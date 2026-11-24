"""CORS middleware configuration and HTTP 500 error message sanitization tests."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi import APIRouter, HTTPException
from fastapi.testclient import TestClient

pytestmark = pytest.mark.offline

_ENV_VARS_TO_ISOLATE = (
    "LLM_BINDING",
    "EMBEDDING_BINDING",
    "LLM_BINDING_HOST",
    "LLM_BINDING_API_KEY",
    "LLM_MODEL",
    "EMBEDDING_BINDING_HOST",
    "EMBEDDING_BINDING_API_KEY",
    "EMBEDDING_MODEL",
    "CORS_ORIGINS",
    "AUTH_ACCOUNTS",
    "TOKEN_SECRET",
    "LIGHTRAG_API_KEY",
    "WHITELIST_PATHS",
    "LIGHTRAG_API_PREFIX",
    "LIGHTRAG_KV_STORAGE",
    "LIGHTRAG_VECTOR_STORAGE",
    "LIGHTRAG_GRAPH_STORAGE",
    "LIGHTRAG_DOC_STATUS_STORAGE",
)

# A stand-in exception message stuffed with the kind of internal detail that
# must never reach a client (mirrors a real asyncpg connection failure).
_SECRET = (
    "connect failed host=db.corp port=5432 user=lightrag key=/etc/lightrag/tls.key"
)
_SECRET_NEEDLES = ("db.corp", "5432", "lightrag", "/etc/lightrag/tls.key", _SECRET)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Isolate tests from developer-local .env pollution and global config state."""
    for var in _ENV_VARS_TO_ISOLATE:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AUTH_ACCOUNTS", "")
    monkeypatch.setenv("LIGHTRAG_API_KEY", "")
    monkeypatch.setenv("TOKEN_SECRET", "")
    monkeypatch.setenv("LLM_BINDING", "ollama")
    monkeypatch.setenv("EMBEDDING_BINDING", "ollama")
    monkeypatch.setenv("EMBEDDING_MODEL", "bge-m3:latest")

    import lightrag.api.config as config

    config._global_args = None
    config._initialized = False
    yield
    config._global_args = None
    config._initialized = False


# --------------------------------------------------------------------------- #
# CORS middleware configuration
# --------------------------------------------------------------------------- #


def _build_cors_client(cors_origins, monkeypatch):
    """Create a TestClient for an app configured with the given CORS_ORIGINS."""
    from lightrag.api.config import initialize_config, parse_args

    monkeypatch.setenv("CORS_ORIGINS", cors_origins)

    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server"]
        args = parse_args()
    finally:
        sys.argv = original_argv

    initialize_config(args, force=True)

    with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
        mock_rag.return_value = MagicMock()
        from lightrag.api.lightrag_server import create_app

        app = create_app(args)
    return TestClient(app)


def _preflight(client, origin):
    """Issue a CORS preflight request and return the response."""
    return client.options(
        "/query",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )


class TestWildcardCorsDisablesCredentials:
    """With CORS_ORIGINS="*" the config must stay spec-compliant."""

    def test_preflight_wildcard_no_credentials(self, monkeypatch):
        client = _build_cors_client("*", monkeypatch)
        resp = _preflight(client, "https://evil.example.com")

        assert resp.headers["access-control-allow-origin"] == "*"
        assert "access-control-allow-credentials" not in resp.headers

    def test_simple_request_wildcard_no_credentials(self, monkeypatch):
        client = _build_cors_client("*", monkeypatch)
        resp = client.get("/health", headers={"Origin": "https://evil.example.com"})

        assert resp.headers.get("access-control-allow-origin") == "*"
        assert "access-control-allow-credentials" not in resp.headers

    def test_wildcard_mixed_with_explicit_origin_no_credentials(self, monkeypatch):
        client = _build_cors_client("*,https://app.example.com", monkeypatch)
        resp = _preflight(client, "https://evil.example.com")

        assert "access-control-allow-credentials" not in resp.headers

    def test_wildcard_trailing_comma_no_credentials(self, monkeypatch):
        client = _build_cors_client("*,", monkeypatch)
        resp = _preflight(client, "https://evil.example.com")

        assert "access-control-allow-credentials" not in resp.headers


class TestEmptyCorsFailsClosed:
    """An explicitly empty CORS_ORIGINS disables cross-origin access (fail closed)."""

    def test_empty_string_allows_no_origin(self, monkeypatch):
        client = _build_cors_client("", monkeypatch)
        resp = _preflight(client, "https://evil.example.com")

        assert "access-control-allow-origin" not in resp.headers
        assert "access-control-allow-credentials" not in resp.headers

    def test_only_commas_allows_no_origin(self, monkeypatch):
        client = _build_cors_client(",,", monkeypatch)
        resp = _preflight(client, "https://evil.example.com")

        assert "access-control-allow-origin" not in resp.headers
        assert "access-control-allow-credentials" not in resp.headers


class TestExplicitAllowlistEnablesCredentials:
    """An explicit origin allowlist reflects the origin and allows credentials."""

    def test_preflight_allowed_origin_reflected_with_credentials(self, monkeypatch):
        client = _build_cors_client(
            "https://app.example.com,https://dash.example.com", monkeypatch
        )
        resp = _preflight(client, "https://app.example.com")

        assert resp.headers["access-control-allow-origin"] == "https://app.example.com"
        assert resp.headers["access-control-allow-credentials"] == "true"

    def test_preflight_disallowed_origin_not_reflected(self, monkeypatch):
        client = _build_cors_client("https://app.example.com", monkeypatch)
        resp = _preflight(client, "https://evil.example.com")

        assert (
            resp.headers.get("access-control-allow-origin")
            != "https://evil.example.com"
        )


# --------------------------------------------------------------------------- #
# Error message sanitization (CWE-209 / GHSA-964x-f4fr-585m)
# --------------------------------------------------------------------------- #


class _FakeLightRAG:
    """Minimal stand-in for LightRAG so create_app touches no backend I/O."""

    def __init__(self, *_args, **_kwargs):
        pass

    def register_role_llm_builder(self, _builder):
        return None

    def set_role_llm_metadata(self, _role, **_metadata):
        return None

    def get_llm_role_config(self):
        return {}


class _FakeOllamaAPI:
    def __init__(self, *_args, **_kwargs):
        self.router = APIRouter()


class TestErrorMessageSanitization:
    """API error responses never leak raw exception text."""

    @pytest.fixture(autouse=True)
    def _init_sanitization_config(self, monkeypatch):
        monkeypatch.setenv("WHITELIST_PATHS", "/_sanitization_probe/*")
        original_argv = sys.argv.copy()
        sys.argv = ["lightrag-server"]
        try:
            from lightrag.api.config import initialize_config, parse_args

            initialize_config(parse_args(), force=True)
        finally:
            sys.argv = original_argv

    def _build_sanitization_client(self, monkeypatch):
        from lightrag.api.config import global_args
        from lightrag.api.utils_api import internal_server_error
        import lightrag.api.lightrag_server as lightrag_server

        monkeypatch.setattr(lightrag_server, "LightRAG", _FakeLightRAG)
        monkeypatch.setattr(
            lightrag_server, "check_frontend_build", lambda: (True, False)
        )
        monkeypatch.setattr(
            lightrag_server, "create_document_routes", lambda *_a, **_k: APIRouter()
        )
        monkeypatch.setattr(
            lightrag_server, "create_query_routes", lambda *_a, **_k: APIRouter()
        )
        monkeypatch.setattr(
            lightrag_server, "create_graph_routes", lambda *_a, **_k: APIRouter()
        )
        monkeypatch.setattr(lightrag_server, "OllamaAPI", _FakeOllamaAPI)

        app = lightrag_server.create_app(global_args)

        @app.get("/_sanitization_probe/via_helper")
        async def _via_helper():
            try:
                raise RuntimeError(_SECRET)
            except Exception as e:
                raise internal_server_error(e)

        @app.get("/_sanitization_probe/bare")
        async def _bare():
            raise RuntimeError(_SECRET)

        @app.get("/_sanitization_probe/legacy_leak")
        async def _legacy_leak():
            try:
                raise RuntimeError(_SECRET)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        return TestClient(app, raise_server_exceptions=False)

    def test_internal_server_error_strips_exception_text(self):
        from lightrag.api.utils_api import internal_server_error

        http_exc = internal_server_error(RuntimeError(_SECRET))
        assert isinstance(http_exc, HTTPException)
        assert http_exc.status_code == 500
        for needle in _SECRET_NEEDLES:
            assert needle not in http_exc.detail
        assert "Internal server error" in http_exc.detail
        assert "error_id" in http_exc.detail

    def test_internal_server_error_uses_fresh_correlation_id_per_call(self):
        from lightrag.api.utils_api import internal_server_error

        first = internal_server_error(ValueError("x")).detail
        second = internal_server_error(ValueError("x")).detail
        assert first != second

    @pytest.mark.parametrize(
        "path",
        ["/_sanitization_probe/via_helper", "/_sanitization_probe/bare"],
    )
    def test_http_500_body_is_sanitized(self, monkeypatch, path):
        resp = self._build_sanitization_client(monkeypatch).get(path)
        assert resp.status_code == 500
        body = resp.text
        for needle in _SECRET_NEEDLES:
            assert needle not in body
        assert resp.json()["detail"].startswith("Internal server error")

    def test_legacy_pattern_leaks_control(self, monkeypatch):
        """Control: the old ``detail=str(e)`` pattern DOES leak."""
        resp = self._build_sanitization_client(monkeypatch).get(
            "/_sanitization_probe/legacy_leak"
        )
        assert resp.status_code == 500
        assert _SECRET in resp.text
