"""Ollama bearer API-key auth, prefixed WHITELIST_PATHS, and WebUI auth discovery."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from tests.api.test_auth_verify import _client as _verify_client
from tests.api.test_webui_and_docs import _ENV_VARS_TO_ISOLATE

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_ollama_api = importlib.import_module("lightrag.api.routers.ollama_api")
_utils_api = importlib.import_module("lightrag.api.utils_api")
_auth = importlib.import_module("lightrag.api.auth")
_server = importlib.import_module("lightrag.api.lightrag_server")
sys.argv = _original_argv

pytestmark = pytest.mark.offline

API_KEY = "the-operators-secret-api-key"

# The shipped default: WHITELIST_PATHS=/health,/api/*
_DEFAULT_PATTERNS = [("/health", False), ("/api", True)]


# --------------------------------------------------------------------------- #
# Ollama bearer API-key authentication
# --------------------------------------------------------------------------- #


class _StubRAG:
    def __init__(self):
        self.ollama_server_infos = SimpleNamespace(
            LIGHTRAG_NAME="lightrag",
            LIGHTRAG_TAG="latest",
            LIGHTRAG_MODEL="lightrag:latest",
            LIGHTRAG_SIZE=1,
            LIGHTRAG_CREATED_AT="now",
            LIGHTRAG_DIGEST="sha",
        )


def _ollama_client(monkeypatch, *, accounts: bool, api_key: str | None = API_KEY):
    """``/api/*`` from the real Ollama router, ``/documents`` from the default
    dependency, under the whitelist the server defaults to once a key is set."""
    monkeypatch.setattr(_utils_api, "whitelist_patterns", [("/health", False)])
    monkeypatch.setattr(_utils_api, "auth_configured", accounts)

    app = FastAPI()
    app.include_router(
        _ollama_api.OllamaAPI(_StubRAG(), api_key=api_key).router, prefix="/api"
    )
    default_dependency = _utils_api.get_combined_auth_dependency(api_key)

    @app.get("/documents", dependencies=[Depends(default_dependency)])
    async def documents():
        return {"statuses": {}}

    return TestClient(app)


def _bearer(value: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {value}"}


class TestOllamaBearerApiKey:
    """The Ollama-compatible routes accept LIGHTRAG_API_KEY as a bearer token."""

    @pytest.mark.parametrize("accounts", [False, True])
    def test_bearer_api_key_authenticates_the_ollama_routes(
        self, monkeypatch, accounts
    ):
        client = _ollama_client(monkeypatch, accounts=accounts)

        assert client.get("/api/version", headers=_bearer(API_KEY)).status_code == 200
        assert client.get("/api/tags", headers=_bearer(API_KEY)).status_code == 200

    @pytest.mark.parametrize("accounts", [False, True])
    def test_x_api_key_still_authenticates_the_ollama_routes(
        self, monkeypatch, accounts
    ):
        client = _ollama_client(monkeypatch, accounts=accounts)

        response = client.get("/api/version", headers={"X-API-Key": API_KEY})
        assert response.status_code == 200

    def test_ollama_routes_require_a_credential_once_a_key_is_set(self, monkeypatch):
        """No credential: refused, and the message names the form a client can send."""
        client = _ollama_client(monkeypatch, accounts=False)

        response = client.get("/api/version")
        assert response.status_code == 403
        assert "Authorization: Bearer <key>" in response.json()["detail"]

    @pytest.mark.parametrize("accounts", [False, True])
    def test_a_wrong_bearer_key_is_refused(self, monkeypatch, accounts):
        client = _ollama_client(monkeypatch, accounts=accounts)

        response = client.get("/api/version", headers=_bearer("not-the-key"))
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid API key or token."

    def test_a_logged_in_user_token_still_works_on_the_ollama_routes(self, monkeypatch):
        client = _ollama_client(monkeypatch, accounts=True)
        token = _auth.auth_handler.create_token("alice", role="user")

        assert client.get("/api/version", headers=_bearer(token)).status_code == 200

    def test_bearer_api_key_is_an_ollama_only_allowance(self, monkeypatch):
        """The rest of the API keeps ``X-API-Key`` as the one way to send the key."""
        client = _ollama_client(monkeypatch, accounts=False)

        assert client.get("/documents", headers=_bearer(API_KEY)).status_code == 401
        assert (
            client.get("/documents", headers={"X-API-Key": API_KEY}).status_code == 200
        )

    def test_an_empty_bearer_never_matches_an_unset_key(self, monkeypatch):
        """Without a configured key there is nothing for a bearer value to match:
        under AUTH_ACCOUNTS it is validated as a JWT and refused."""
        client = _ollama_client(monkeypatch, accounts=True, api_key=None)

        assert (
            client.get("/api/version", headers=_bearer("anything")).status_code == 401
        )

    @pytest.mark.parametrize(
        "presented,expected",
        [
            (API_KEY, True),
            (API_KEY + "x", False),
            ("", False),
            (None, False),
            ("ключ", False),  # non-ASCII must compare, not raise
        ],
    )
    def test_api_key_matches(self, presented, expected):
        assert _utils_api.api_key_matches(presented, API_KEY) is expected
        assert _utils_api.api_key_matches(presented, None) is False
        assert _utils_api.api_key_matches(presented, "") is False


# --------------------------------------------------------------------------- #
# WHITELIST_PATHS under LIGHTRAG_API_PREFIX
# --------------------------------------------------------------------------- #


def _whitelist_client(monkeypatch, api_prefix: str) -> TestClient:
    """The real dependency and the real path-normalizing middleware, mounted the
    way ``create_app`` mounts them for a prefixed deployment."""
    monkeypatch.setattr(_utils_api, "whitelist_patterns", list(_DEFAULT_PATTERNS))
    monkeypatch.setattr(_utils_api, "auth_configured", True)

    dependency = _utils_api.get_combined_auth_dependency(API_KEY)
    app = FastAPI()
    app.root_path = api_prefix
    if api_prefix:
        app.add_middleware(_server._RootPathNormalizationMiddleware)

    @app.get("/documents", dependencies=[Depends(dependency)])
    async def list_documents():
        return {"statuses": {}}

    @app.delete("/documents", dependencies=[Depends(dependency)])
    async def clear_documents():
        return {"status": "success"}

    @app.get("/health", dependencies=[Depends(dependency)])
    async def health():
        return {"status": "healthy"}

    # Mirrors the production registration: GET, mounted under the /api router.
    @app.get("/api/tags", dependencies=[Depends(dependency)])
    async def ollama_tags():
        return {"models": []}

    return TestClient(app)


def _request_path(api_prefix: str, route_path: str, mode: str) -> str:
    return route_path if mode == "strip" else f"{api_prefix}{route_path}"


class TestWhitelistPathPrefix:
    """WHITELIST_PATHS under LIGHTRAG_API_PREFIX, over real HTTP."""

    @pytest.mark.parametrize("mode", ["verbatim", "strip"])
    @pytest.mark.parametrize("api_prefix", ["/api/v1", "/api", "/site01"])
    def test_protected_routes_stay_authenticated_under_any_prefix(
        self, monkeypatch, api_prefix, mode
    ):
        """No prefix may exempt a protected route."""
        client = _whitelist_client(monkeypatch, api_prefix)

        path = _request_path(api_prefix, "/documents", mode)
        assert client.get(path).status_code == 401
        assert client.delete(path).status_code == 401

    @pytest.mark.parametrize("mode", ["verbatim", "strip"])
    @pytest.mark.parametrize("api_prefix", ["/api/v1", "/site01"])
    def test_health_stays_exempt_under_any_prefix(self, monkeypatch, api_prefix, mode):
        """``/health`` promises to answer liveness probes unauthenticated."""
        client = _whitelist_client(monkeypatch, api_prefix)

        response = client.get(_request_path(api_prefix, "/health", mode))
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    @pytest.mark.parametrize("mode", ["verbatim", "strip"])
    @pytest.mark.parametrize("api_prefix", ["/api/v1", "/site01"])
    def test_ollama_routes_stay_exempt_under_any_prefix(
        self, monkeypatch, api_prefix, mode
    ):
        """The default whitelist exists for Ollama-client compatibility."""
        client = _whitelist_client(monkeypatch, api_prefix)

        response = client.get(_request_path(api_prefix, "/api/tags", mode))
        assert response.status_code == 200

    def test_no_prefix_behaviour_is_unchanged(self, monkeypatch):
        client = _whitelist_client(monkeypatch, "")

        assert client.get("/documents").status_code == 401
        assert client.delete("/documents").status_code == 401
        assert client.get("/health").status_code == 200
        assert client.get("/api/tags").status_code == 200

    @pytest.mark.parametrize("mode", ["verbatim", "strip"])
    def test_valid_credentials_still_reach_protected_routes(self, monkeypatch, mode):
        client = _whitelist_client(monkeypatch, "/api/v1")

        response = client.get(
            _request_path("/api/v1", "/documents", mode),
            headers={"X-API-Key": API_KEY},
        )
        assert response.status_code == 200


# --------------------------------------------------------------------------- #
# Public WebUI auth discovery
# --------------------------------------------------------------------------- #


class TestWebuiAuthDiscovery:
    """Public WebUI capability metadata is additive; API authentication is unchanged."""

    @pytest.fixture(autouse=True)
    def _isolate_discovery_env(self, monkeypatch):
        import lightrag.api.config as config

        for var in _ENV_VARS_TO_ISOLATE:
            monkeypatch.delenv(var, raising=False)
        for var in ("AUTH_ACCOUNTS", "LIGHTRAG_API_KEY", "TOKEN_SECRET"):
            monkeypatch.setenv(var, "")
        monkeypatch.setenv("LLM_BINDING", "openai")
        monkeypatch.setenv("EMBEDDING_BINDING", "openai")
        monkeypatch.setattr(config, "_global_args", None)
        monkeypatch.setattr(config, "_initialized", False)

    @pytest.mark.parametrize("accounts", [False, True])
    @pytest.mark.parametrize("key_source", ["none", "cli", "env", "both"])
    def test_discovery_preserves_fields_and_uses_effective_key(
        self, tmp_path, monkeypatch, accounts, key_source
    ):
        cli = ("--key", "cli-secret") if key_source in ("cli", "both") else ()
        if key_source in ("env", "both"):
            monkeypatch.setenv("LIGHTRAG_API_KEY", "env-secret")
        client = _verify_client(tmp_path, monkeypatch, *cli)
        import lightrag.api.utils_api as utils_api

        monkeypatch.setattr(
            utils_api.auth_handler,
            "accounts",
            {"alice": "password"} if accounts else {},
        )
        monkeypatch.setattr(utils_api, "auth_configured", accounts)

        response = client.get("/auth-status")
        assert response.status_code == 200
        data = response.json()
        assert data["auth_configured"] is accounts
        assert data["api_key_configured"] is (key_source != "none")
        assert data["auth_mode"] == ("enabled" if accounts else "disabled")
        common = {
            "auth_configured",
            "auth_mode",
            "core_version",
            "api_version",
            "webui_title",
            "webui_description",
            "ai_content_notice_enabled",
        }
        expected = common | {"api_key_configured"}
        if not accounts:
            expected |= {"access_token", "token_type", "message"}
            assert data["token_type"] == "bearer"
            assert (
                utils_api.auth_handler.validate_token(data["access_token"])["role"]
                == "guest"
            )
        assert set(data) == expected
        assert "cli-secret" not in response.text
        assert "env-secret" not in response.text

        if key_source != "none":
            effective_key = (
                "env-secret" if key_source in ("env", "both") else "cli-secret"
            )
            assert (
                client.get(
                    "/auth/verify", headers={"X-API-Key": effective_key}
                ).status_code
                == 200
            )

    def test_key_only_login_still_issues_guest_token(self, tmp_path, monkeypatch):
        client = _verify_client(tmp_path, monkeypatch, "--key", "cli-secret")
        response = client.post(
            "/login", data={"username": "guest", "password": "unused"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["auth_mode"] == "disabled"
        assert data["access_token"]
        assert "api_key_configured" not in data
        assert (
            client.get(
                "/auth/verify",
                headers={"Authorization": f"Bearer {data['access_token']}"},
            ).status_code
            == 403
        )

    def test_empty_environment_key_uses_cli_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LIGHTRAG_API_KEY", "")
        client = _verify_client(tmp_path, monkeypatch, "--key", "cli-secret")
        assert client.get("/auth-status").json()["api_key_configured"] is True
        assert (
            client.get("/auth/verify", headers={"X-API-Key": "cli-secret"}).status_code
            == 200
        )
