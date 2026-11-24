"""Auth handler, password hashing, JWT validation, and shared pre-auth helpers."""

from __future__ import annotations

import base64
import importlib
import json
import sys
from types import SimpleNamespace

import bcrypt
import jwt
import pytest
from fastapi import HTTPException
from starlette.status import HTTP_403_FORBIDDEN

from lightrag.api.passwords import BCRYPT_PASSWORD_PREFIX, hash_password
from lightrag.tools.hash_password import main as hash_password_main
from lightrag.utils import logger as lightrag_logger
from tests.api._cheap_bcrypt import CHEAP_DUMMY_SPEC, cheap_bcrypt_salts

# Every test here checks which verification ran and its result, never its
# cost, so hashes and the dummy spec use bcrypt's minimum cost.
pytestmark = [pytest.mark.offline, pytest.mark.usefixtures(cheap_bcrypt_salts.__name__)]

_JWT_SECRET = "test-jwt-secret-that-is-at-least-32-bytes"


def import_real_api_module(module_name: str):
    sys.modules.pop(module_name, None)

    package_name, _, child_name = module_name.rpartition(".")
    package = sys.modules.get(package_name)
    if package is not None and hasattr(package, child_name):
        delattr(package, child_name)

    return importlib.import_module(module_name)


@pytest.fixture
def auth_module(monkeypatch):
    config = import_real_api_module("lightrag.api.config")

    mock_global_args = SimpleNamespace(
        token_secret=_JWT_SECRET,
        jwt_algorithm="HS256",
        token_expire_hours=48,
        guest_token_expire_hours=24,
        auth_accounts="admin:admin_pass",
    )

    monkeypatch.setattr(config, "global_args", mock_global_args)

    module = import_real_api_module("lightrag.api.auth")
    module = importlib.reload(module)
    monkeypatch.setattr(module, "_DUMMY_VERIFY_SPEC", CHEAP_DUMMY_SPEC)
    yield module
    sys.modules.pop("lightrag.api.auth", None)


def build_bcrypt_value(password: str) -> str:
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    return f"{BCRYPT_PASSWORD_PREFIX}{hashed}"


# --------------------------------------------------------------------------- #
# Password verification, account parsing, and combined auth
# --------------------------------------------------------------------------- #


def test_verify_plaintext_password(auth_module):
    handler = auth_module.AuthHandler()
    handler.accounts = {"admin": "admin_pass"}

    assert handler.verify_password("admin", "admin_pass")
    assert not handler.verify_password("admin", "wrong_pass")


def test_verify_prefixed_bcrypt_password(auth_module):
    handler = auth_module.AuthHandler()
    handler.accounts = {"user": build_bcrypt_value("user_pass")}

    assert handler.verify_password("user", "user_pass")
    assert not handler.verify_password("user", "wrong_pass")


def test_plaintext_password_with_bcrypt_prefix_stays_plaintext(auth_module):
    handler = auth_module.AuthHandler()
    handler.accounts = {"user": "$2b$not-a-real-hash"}

    assert handler.verify_password("user", "$2b$not-a-real-hash")
    assert not handler.verify_password("user", "anything-else")


def test_plaintext_verify_uses_constant_time_comparison():
    """Plaintext specs must be compared with hmac.compare_digest rather than a
    short-circuiting ``==`` that leaks length/content via timing (CWE-208,
    GHSA-c759-cx9p-mrwq). Behavior across lengths and unicode is the regression
    anchor for the constant-time path.
    """
    from lightrag.api.passwords import verify_password as verify

    assert verify("secret", "secret")
    assert not verify("secret", "secre")  # shorter guess
    assert not verify("secret", "secretx")  # longer guess
    assert verify("pässwörd-пароль", "pässwörd-пароль")  # unicode via utf-8
    assert not verify("pässwörd-пароль", "wrong")


def test_verify_password_unknown_user_runs_dummy_verification(auth_module, monkeypatch):
    """Unknown usernames must still trigger a password verification so response
    time does not reveal whether the account exists (username enumeration via
    the bcrypt timing delta, CWE-208). The old code returned early without
    comparing anything.
    """
    handler = auth_module.AuthHandler()
    handler.accounts = {"admin": "admin_pass"}

    calls = []
    real_verify = auth_module.verify_password

    def spy(plain, stored):
        calls.append((plain, stored))
        return real_verify(plain, stored)

    monkeypatch.setattr(auth_module, "verify_password", spy)

    assert handler.verify_password("does-not-exist", "whatever") is False
    # Verification ran against the dummy spec instead of being skipped.
    assert calls == [("whatever", auth_module._DUMMY_VERIFY_SPEC)]


def test_known_plaintext_user_also_runs_dummy_bcrypt(auth_module, monkeypatch):
    """A known plaintext account must run the dummy bcrypt in addition to the
    microsecond-scale plaintext compare, so its timing matches the unknown-user
    and {bcrypt} paths. Otherwise a fast response would reveal that a username
    exists and is stored as plaintext (username enumeration, CWE-208). The real
    comparison result must still be returned.
    """
    handler = auth_module.AuthHandler()
    handler.accounts = {"alice": "password123"}

    specs = []
    real_verify = auth_module.verify_password

    def spy(plain, stored):
        specs.append(stored)
        return real_verify(plain, stored)

    monkeypatch.setattr(auth_module, "verify_password", spy)

    # Correct password -> True; both the real compare and the dummy bcrypt ran.
    assert handler.verify_password("alice", "password123") is True
    assert specs == ["password123", auth_module._DUMMY_VERIFY_SPEC]

    # Wrong password -> False, with the same two-step timing profile.
    specs.clear()
    assert handler.verify_password("alice", "nope") is False
    assert specs == ["password123", auth_module._DUMMY_VERIFY_SPEC]


def test_known_bcrypt_user_runs_single_verification(auth_module, monkeypatch):
    """A known {bcrypt} account already costs one bcrypt; it must NOT run an
    extra dummy bcrypt, which would make it slower than the other paths and
    reintroduce a timing signal.
    """
    handler = auth_module.AuthHandler()
    bcrypt_spec = build_bcrypt_value("user_pass")
    handler.accounts = {"user": bcrypt_spec}

    specs = []
    real_verify = auth_module.verify_password

    def spy(plain, stored):
        specs.append(stored)
        return real_verify(plain, stored)

    monkeypatch.setattr(auth_module, "verify_password", spy)

    assert handler.verify_password("user", "user_pass") is True
    # Exactly one verification (the real bcrypt), no dummy appended.
    assert specs == [bcrypt_spec]


def test_verify_password_unicode_plaintext(auth_module):
    """Non-ASCII plaintext passwords still verify through the utf-8 encoded
    constant-time comparison path.
    """
    handler = auth_module.AuthHandler()
    handler.accounts = {"admin": "pässw0rd-пароль"}

    assert handler.verify_password("admin", "pässw0rd-пароль")
    assert not handler.verify_password("admin", "pässw0rd-wrong")


def test_invalid_auth_accounts_raises(monkeypatch):
    config = import_real_api_module("lightrag.api.config")

    mock_global_args = SimpleNamespace(
        token_secret=_JWT_SECRET,
        jwt_algorithm="HS256",
        token_expire_hours=48,
        guest_token_expire_hours=24,
        auth_accounts="admin",
    )

    monkeypatch.setattr(config, "global_args", mock_global_args)

    with pytest.raises(ValueError, match="AUTH_ACCOUNTS must use"):
        import_real_api_module("lightrag.api.auth")

    sys.modules.pop("lightrag.api.auth", None)


def load_auth_handler(monkeypatch, auth_accounts: str):
    """Build the module-level AuthHandler from an AUTH_ACCOUNTS value."""
    config = import_real_api_module("lightrag.api.config")

    mock_global_args = SimpleNamespace(
        token_secret=_JWT_SECRET,
        jwt_algorithm="HS256",
        token_expire_hours=48,
        guest_token_expire_hours=24,
        auth_accounts=auth_accounts,
    )

    monkeypatch.setattr(config, "global_args", mock_global_args)

    try:
        module = import_real_api_module("lightrag.api.auth")
        monkeypatch.setattr(module, "_DUMMY_VERIFY_SPEC", CHEAP_DUMMY_SPEC)
        return module.auth_handler
    finally:
        sys.modules.pop("lightrag.api.auth", None)


def test_auth_accounts_ignore_whitespace_around_usernames(monkeypatch):
    """A space after the comma (``admin:a, user1:b``) or before the colon must
    not become part of the username: the account would be registered as
    `` user1`` and every login as ``user1`` would fail with no hint why.
    """
    handler = load_auth_handler(monkeypatch, "admin:admin_pass, user1 :user_pass")

    assert handler.accounts == {"admin": "admin_pass", "user1": "user_pass"}


@pytest.mark.parametrize(
    ("auth_accounts", "expected"),
    [
        ("admin:secret ", {"admin": "secret "}),
        ("admin: secret", {"admin": " secret"}),
        ("admin:   ", {"admin": "   "}),
        ("admin:a:b, user1:c", {"admin": "a:b", "user1": "c"}),
    ],
)
def test_auth_accounts_keep_password_bytes(monkeypatch, auth_accounts, expected):
    """Everything after the first ``:`` is the credential, whitespace included.
    Plaintext verification compares exact bytes, so trimming ``secret `` to
    ``secret`` would reject the configured password (PR #3944 review).
    """
    handler = load_auth_handler(monkeypatch, auth_accounts)

    assert handler.accounts == expected
    for username, password in expected.items():
        assert handler.verify_password(username, password)
    assert not handler.verify_password("admin", "secret")


def test_auth_accounts_bcrypt_entry_after_comma_space(monkeypatch):
    spec = hash_password("user_pass")
    handler = load_auth_handler(monkeypatch, f"admin:admin_pass, user1:{spec}")

    assert handler.accounts["user1"] == spec
    assert handler.verify_password("user1", "user_pass")


def test_initialize_config_rejects_default_token_secret_with_auth_accounts():
    config = import_real_api_module("lightrag.api.config")

    insecure_args = SimpleNamespace(
        auth_accounts="admin:admin_pass",
        token_secret=config.DEFAULT_TOKEN_SECRET,
    )

    with pytest.raises(ValueError, match="TOKEN_SECRET must be explicitly set"):
        config.initialize_config(insecure_args, force=True)


def test_initialize_config_allows_custom_token_secret_with_auth_accounts():
    config = import_real_api_module("lightrag.api.config")

    secure_args = SimpleNamespace(
        auth_accounts="admin:admin_pass",
        token_secret="custom-jwt-secret-that-is-at-least-32-bytes",
    )

    initialized = config.initialize_config(secure_args, force=True)

    assert initialized is secure_args


def test_guest_tokens_fall_back_to_default_secret_when_token_secret_missing(
    monkeypatch,
):
    config = import_real_api_module("lightrag.api.config")

    mock_global_args = SimpleNamespace(
        token_secret=None,
        jwt_algorithm="HS256",
        token_expire_hours=48,
        guest_token_expire_hours=24,
        auth_accounts="",
    )

    monkeypatch.setattr(config, "global_args", mock_global_args)
    warning_messages = []

    def capture_warning(message):
        warning_messages.append(message)

    monkeypatch.setattr(lightrag_logger, "warning", capture_warning)

    module = import_real_api_module("lightrag.api.auth")
    module = importlib.reload(module)
    handler = module.AuthHandler()

    token = handler.create_token("guest", role="guest")
    token_info = handler.validate_token(token)

    assert handler.secret == config.DEFAULT_TOKEN_SECRET
    assert token_info["username"] == "guest"
    assert token_info["role"] == "guest"
    assert any(
        "Falling back to the default guest-mode JWT secret" in msg
        for msg in warning_messages
    )

    sys.modules.pop("lightrag.api.auth", None)


def _load_api_key_only_modules(monkeypatch):
    """Import auth + utils_api in the API-key-only profile.

    API-key-only = an API key is set but AUTH_ACCOUNTS is empty, so
    ``auth_configured`` is False and AuthHandler falls back to the public
    DEFAULT_TOKEN_SECRET. Returns (config, auth, utils_api).
    """
    config = import_real_api_module("lightrag.api.config")

    mock_global_args = SimpleNamespace(
        token_secret=None,  # -> AuthHandler falls back to DEFAULT_TOKEN_SECRET
        jwt_algorithm="HS256",
        token_expire_hours=48,
        guest_token_expire_hours=24,
        auth_accounts="",  # no password accounts -> auth_configured is False
        whitelist_paths="/health",  # consumed by utils_api at import time
        token_auto_renew=False,
    )
    monkeypatch.setattr(config, "global_args", mock_global_args)

    auth = import_real_api_module("lightrag.api.auth")
    auth = importlib.reload(auth)
    utils_api = import_real_api_module("lightrag.api.utils_api")
    return config, auth, utils_api


async def test_combined_auth_rejects_guest_token_in_api_key_mode(monkeypatch):
    """A forged/obtained guest token must not bypass the X-API-Key check.

    Regression for GHSA-f4vv-55c2-5789 / GHSA-xr5c-v5r6-c9f9: in the
    API-key-only profile an anonymous caller can mint a guest JWT with the
    public default secret (or fetch one from /auth-status) and previously
    short-circuited authorization before the API key was ever checked.
    """
    config, auth, utils_api = _load_api_key_only_modules(monkeypatch)
    try:
        assert utils_api.auth_configured is False
        assert auth.auth_handler.secret == config.DEFAULT_TOKEN_SECRET

        forged_guest = auth.auth_handler.create_token(username="guest", role="guest")

        api_key = "operator-secret-key"
        dependency = utils_api.get_combined_auth_dependency(api_key=api_key)

        request = SimpleNamespace(url=SimpleNamespace(path="/documents"), scope={})
        response = SimpleNamespace(headers={})

        # Forged guest token, no API key -> rejected (was HTTP 200 before the fix).
        with pytest.raises(HTTPException) as rejected:
            await dependency(
                request=request,
                response=response,
                token=forged_guest,
                api_key_header_value=None,
            )
        assert rejected.value.status_code == HTTP_403_FORBIDDEN

        # Guest token together with the real API key -> allowed (the WebUI sends
        # both Authorization and X-API-Key headers).
        result = await dependency(
            request=request,
            response=response,
            token=forged_guest,
            api_key_header_value=api_key,
        )
        assert result is None
    finally:
        sys.modules.pop("lightrag.api.utils_api", None)
        sys.modules.pop("lightrag.api.auth", None)


async def test_combined_auth_allows_guest_token_when_fully_open(monkeypatch):
    """Fully-open mode (no API key, no AUTH_ACCOUNTS) keeps accepting guest tokens.

    Guards against over-fixing: the bypass fix must not break zero-config guest
    access, which is the intended convenience mode.
    """
    _config, auth, utils_api = _load_api_key_only_modules(monkeypatch)
    try:
        guest_token = auth.auth_handler.create_token(username="guest", role="guest")

        dependency = utils_api.get_combined_auth_dependency(api_key=None)
        request = SimpleNamespace(url=SimpleNamespace(path="/documents"), scope={})
        response = SimpleNamespace(headers={})

        result = await dependency(
            request=request,
            response=response,
            token=guest_token,
            api_key_header_value=None,
        )
        assert result is None
    finally:
        sys.modules.pop("lightrag.api.utils_api", None)
        sys.modules.pop("lightrag.api.auth", None)


def test_hash_password_returns_prefixed_value(auth_module):
    hashed = hash_password("new_password")

    assert hashed.startswith(BCRYPT_PASSWORD_PREFIX)
    raw_hash = hashed[len(BCRYPT_PASSWORD_PREFIX) :]
    assert bcrypt.checkpw("new_password".encode("utf-8"), raw_hash.encode("utf-8"))


def test_hash_password_cli_outputs_auth_accounts_entry(capsys):
    exit_code = hash_password_main(["--username", "admin", "secret"])

    assert exit_code == 0
    output = capsys.readouterr().out.strip()
    username, hashed = output.split(":", 1)
    assert username == "admin"
    assert hashed.startswith(BCRYPT_PASSWORD_PREFIX)
    raw_hash = hashed[len(BCRYPT_PASSWORD_PREFIX) :]
    assert bcrypt.checkpw("secret".encode("utf-8"), raw_hash.encode("utf-8"))


@pytest.mark.parametrize(
    "password",
    [
        pytest.param("a" * 73, id="ascii-73-bytes"),
        # 37 two-byte characters: 37 characters but 74 bytes in UTF-8.
        pytest.param("é" * 37, id="utf8-74-bytes"),
    ],
)
def test_hash_password_rejects_password_over_bcrypt_limit(password):
    with pytest.raises(ValueError, match="at most 72"):
        hash_password(password)


def test_hash_password_accepts_password_at_bcrypt_limit():
    password = "a" * 72
    hashed = hash_password(password)

    raw_hash = hashed[len(BCRYPT_PASSWORD_PREFIX) :]
    assert bcrypt.checkpw(password.encode("utf-8"), raw_hash.encode("utf-8"))


def test_hash_password_cli_reports_password_over_bcrypt_limit(capsys):
    with pytest.raises(SystemExit) as exc_info:
        hash_password_main(["--username", "admin", "a" * 73])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "password is 73 bytes in UTF-8; bcrypt accepts at most 72" in captured.err


# --------------------------------------------------------------------------- #
# JWT validation guards (forged / tampered tokens)
# --------------------------------------------------------------------------- #


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _assert_rejected(handler, token: str) -> None:
    with pytest.raises(HTTPException) as exc_info:
        handler.validate_token(token)
    assert exc_info.value.status_code == 401


class TestJWTValidation:
    """Offline guards that forged or tampered JWTs are rejected after a PyJWT upgrade."""

    @pytest.fixture
    def handler(self, auth_module):
        return auth_module.AuthHandler()

    def test_valid_token_round_trips(self, handler):
        token = handler.create_token("alice", role="user")

        info = handler.validate_token(token)

        assert info["username"] == "alice"
        assert info["role"] == "user"

    def test_tampered_signature_is_rejected(self, handler):
        header, payload, signature = handler.create_token("alice").split(".")
        flipped = ("A" if signature[0] != "A" else "B") + signature[1:]

        _assert_rejected(handler, ".".join([header, payload, flipped]))

    def test_token_signed_with_other_secret_is_rejected(self, handler):
        forged = jwt.encode(
            {"sub": "alice", "role": "user", "exp": 4102444800},
            "some-other-secret-that-is-32-bytes-long",
            algorithm="HS256",
        )

        _assert_rejected(handler, forged)

    def test_unsigned_alg_none_token_is_rejected(self, handler):
        header = _b64(json.dumps({"alg": "none", "typ": "JWT"}).encode())
        payload = _b64(
            json.dumps({"sub": "admin", "role": "user", "exp": 4102444800}).encode()
        )

        _assert_rejected(handler, f"{header}.{payload}.")

    def test_swapped_payload_with_old_signature_is_rejected(self, handler):
        header, _, signature = handler.create_token("alice").split(".")
        forged_payload = _b64(
            json.dumps({"sub": "admin", "role": "user", "exp": 4102444800}).encode()
        )

        _assert_rejected(handler, ".".join([header, forged_payload, signature]))


# --------------------------------------------------------------------------- #
# Shared pre-auth helpers: path_is_whitelisted, get_route_path, credentials_accepted
# --------------------------------------------------------------------------- #


def _scope(path: str, root_path: str = "") -> dict:
    """A minimal ASGI scope. ``path`` is in canonical form: it includes
    ``root_path``, the way FastAPI hands it to the route dependency."""
    return {"type": "http", "path": path, "root_path": root_path}


class TestSharedCredentialCheck:
    """The shared pre-auth helpers used by admission middleware and /health."""

    @pytest.fixture
    def utils_api(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", [sys.argv[0]])
        return importlib.import_module("lightrag.api.utils_api")

    @pytest.fixture
    def patterns(self, monkeypatch, utils_api):
        def _set(entries):
            compiled = []
            for entry in entries:
                if entry.endswith("/*"):
                    compiled.append((entry[:-2], True))
                else:
                    compiled.append((entry, False))
            monkeypatch.setattr(utils_api, "whitelist_patterns", compiled)

        return _set

    def test_exact_and_prefix_whitelist_entries(self, utils_api, patterns):
        patterns(["/health", "/api/*"])

        assert utils_api.path_is_whitelisted(_scope("/health")) is True
        assert utils_api.path_is_whitelisted(_scope("/api/chat")) is True
        assert utils_api.path_is_whitelisted(_scope("/healthz")) is False
        assert utils_api.path_is_whitelisted(_scope("/documents/upload")) is False

    def test_catch_all_whitelist_covers_ingestion(self, utils_api, patterns):
        """An operator who whitelists everything must not get 401s from the
        middleware on paths the route dependency waves through."""
        patterns(["/*"])

        assert utils_api.path_is_whitelisted(_scope("/documents/upload")) is True

    def test_entries_are_matched_against_the_route_path(self, utils_api, patterns):
        """Entries are unprefixed route paths, so the mount prefix must come off
        before matching — in either forwarding form."""
        patterns(["/health", "/api/*"])

        # Verbatim forwarding: the proxy passes the prefixed path through.
        assert (
            utils_api.path_is_whitelisted(_scope("/api/v1/health", "/api/v1")) is True
        )
        assert (
            utils_api.path_is_whitelisted(_scope("/api/v1/documents", "/api/v1"))
            is False
        )
        assert (
            utils_api.path_is_whitelisted(_scope("/site01/health", "/site01")) is True
        )
        assert (
            utils_api.path_is_whitelisted(_scope("/site01/api/tags", "/site01")) is True
        )
        assert (
            utils_api.path_is_whitelisted(_scope("/site01/documents", "/site01"))
            is False
        )

        # Proxy-strip forwarding: bare path with root_path already set.
        assert utils_api.path_is_whitelisted(_scope("/health", "/api/v1")) is True
        assert utils_api.path_is_whitelisted(_scope("/documents", "/api/v1")) is False

    def test_prefix_entries_match_on_segment_boundaries(self, utils_api, patterns):
        """A ``/*`` entry exempts the prefix and what is under it, nothing else."""
        patterns(["/graph/*"])

        assert utils_api.path_is_whitelisted(_scope("/graph")) is True
        assert utils_api.path_is_whitelisted(_scope("/graph/label/list")) is True
        assert utils_api.path_is_whitelisted(_scope("/graphs")) is False

        patterns(["/api/*"])
        assert utils_api.path_is_whitelisted(_scope("/api/tags")) is True
        assert utils_api.path_is_whitelisted(_scope("/apikeys")) is False

    def test_mount_prefix_falls_back_to_the_explicit_argument(
        self, utils_api, patterns
    ):
        """Callers whose scope was not built by FastAPI (hand-rolled ASGI scopes)
        pass the prefix explicitly."""
        patterns(["/health"])

        assert (
            utils_api.path_is_whitelisted(
                {"type": "http", "path": "/site01/health"}, mount_prefix="/site01"
            )
            is True
        )

    def test_route_path_never_cuts_mid_segment(self, utils_api):
        """``/api`` is not a segment prefix of ``/apikeys``, so nothing comes off."""
        assert utils_api.get_route_path(_scope("/apikeys", "/api")) == "/apikeys"
        assert utils_api.get_route_path(_scope("/api/tags", "/api")) == "/tags"
        # An exact-prefix hit normalizes to "/" (Starlette returns "" here).
        assert utils_api.get_route_path(_scope("/site01", "/site01")) == "/"
        assert utils_api.get_route_path(_scope("/documents")) == "/documents"

    def test_fully_open_mode_accepts_anything(self, monkeypatch, utils_api):
        monkeypatch.setattr(utils_api, "auth_configured", False)

        assert (
            utils_api.credentials_accepted(
                token=None, api_key_header_value=None, api_key=None
            )
            is True
        )

    def test_api_key_mode_requires_the_key(self, monkeypatch, utils_api):
        monkeypatch.setattr(utils_api, "auth_configured", False)

        assert (
            utils_api.credentials_accepted(
                token=None, api_key_header_value=None, api_key="secret"
            )
            is False
        )
        assert (
            utils_api.credentials_accepted(
                token=None, api_key_header_value="wrong", api_key="secret"
            )
            is False
        )
        assert (
            utils_api.credentials_accepted(
                token=None, api_key_header_value="secret", api_key="secret"
            )
            is True
        )

    def test_guest_token_never_substitutes_for_the_api_key(
        self, monkeypatch, utils_api
    ):
        """GHSA-f4vv-55c2-5789: a guest token must not authenticate in
        API-key-only mode."""
        monkeypatch.setattr(utils_api, "auth_configured", False)
        guest = utils_api.auth_handler.create_token(username="guest", role="guest")

        assert (
            utils_api.credentials_accepted(
                token=guest, api_key_header_value=None, api_key="secret"
            )
            is False
        )

    def test_password_mode_accepts_a_non_guest_token(self, monkeypatch, utils_api):
        monkeypatch.setattr(utils_api, "auth_configured", True)
        user = utils_api.auth_handler.create_token(username="alice", role="user")
        guest = utils_api.auth_handler.create_token(username="guest", role="guest")

        assert (
            utils_api.credentials_accepted(
                token=user, api_key_header_value=None, api_key=None
            )
            is True
        )
        # A guest token is not a login.
        assert (
            utils_api.credentials_accepted(
                token=guest, api_key_header_value=None, api_key=None
            )
            is False
        )

    def test_garbage_token_is_rejected_not_raised(self, monkeypatch, utils_api):
        monkeypatch.setattr(utils_api, "auth_configured", True)

        assert (
            utils_api.credentials_accepted(
                token="not-a-jwt", api_key_header_value=None, api_key=None
            )
            is False
        )
