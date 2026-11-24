"""Verify /query and /query/stream response types, NDJSON protocol order, cancellation cleanup, and streamed error sanitization."""

from __future__ import annotations

import asyncio
import importlib
import json
import re
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_ollama_api = importlib.import_module("lightrag.api.routers.ollama_api")
_query_routes = importlib.import_module("lightrag.api.routers.query_routes")
_utils_api = importlib.import_module("lightrag.api.utils_api")
sys.argv = _original_argv

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
    "LIGHTRAG_API_PREFIX",
    "LIGHTRAG_KV_STORAGE",
    "LIGHTRAG_VECTOR_STORAGE",
    "LIGHTRAG_GRAPH_STORAGE",
    "LIGHTRAG_DOC_STATUS_STORAGE",
    "AUTH_ACCOUNTS",
    "TOKEN_SECRET",
    "WHITELIST_PATHS",
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    for var in _ENV_VARS_TO_ISOLATE:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_BINDING", "ollama")
    monkeypatch.setenv("EMBEDDING_BINDING", "ollama")
    monkeypatch.setenv("EMBEDDING_MODEL", "bge-m3:latest")
    monkeypatch.setenv("AUTH_ACCOUNTS", "")
    monkeypatch.setenv("TOKEN_SECRET", "")
    monkeypatch.setenv("LIGHTRAG_API_KEY", "")

    auth_module = sys.modules.get("lightrag.api.auth")
    if auth_module is not None:
        monkeypatch.setattr(auth_module.auth_handler, "accounts", {}, raising=False)
    monkeypatch.setattr(_utils_api, "auth_configured", False, raising=False)


def _build_client():
    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server"]
        from lightrag.api.config import parse_args
        from lightrag.api.lightrag_server import create_app

        args = parse_args()
        with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
            mock_rag.return_value = MagicMock()
            return TestClient(create_app(args))
    finally:
        sys.argv = original_argv


class TestQueryRouteJsonOnly:
    """The /query endpoint must stay JSON-only to preserve backward compatibility."""

    def test_openapi_spec_declares_json_response(self):
        client = _build_client()
        response = client.get("/openapi.json")
        assert response.status_code == 200
        spec = response.json()

        paths = spec.get("paths", {})
        query_path = paths.get("/query", {})
        assert query_path, "/query must be in OpenAPI paths"

        post_op = query_path.get("post", {})
        content = post_op.get("responses", {}).get("200", {}).get("content", {})

        assert "application/json" in content
        assert "application/x-ndjson" not in content


class TestQueryStreamRoute:
    """The /query/stream endpoint must serve application/x-ndjson."""

    def test_openapi_spec_declares_ndjson_response(self):
        client = _build_client()
        response = client.get("/openapi.json")
        assert response.status_code == 200
        spec = response.json()

        paths = spec.get("paths", {})
        stream_path = paths.get("/query/stream", {})
        assert stream_path, "/query/stream must be in OpenAPI paths"

        post_op = stream_path.get("post", {})
        content = post_op.get("responses", {}).get("200", {}).get("content", {})

        assert "application/x-ndjson" in content


class TestQueryStreamResponseContentType:
    """When the mock LLM returns a non-streaming result, /query/stream still sets NDJSON Content-Type."""

    def test_stream_response_has_ndjson_content_type(self):
        original_argv = sys.argv.copy()
        try:
            sys.argv = ["lightrag-server"]
            from lightrag.api.config import parse_args
            from lightrag.api.lightrag_server import create_app

            args = parse_args()
            mock_rag = MagicMock()
            mock_result = {
                "llm_response": {"is_streaming": False, "content": "test response"},
                "data": {"references": []},
            }

            async def _fake_aquery(*a, **kw):
                return mock_result

            mock_rag.aquery_llm = MagicMock(side_effect=_fake_aquery)

            with patch("lightrag.api.lightrag_server.LightRAG", return_value=mock_rag):
                app = create_app(args)

            client = TestClient(app)
            response = client.post(
                "/query/stream",
                json={"query": "test", "mode": "mix", "include_references": True},
            )
            content_type = response.headers.get("content-type", "")
            assert "application/x-ndjson" in content_type
        finally:
            sys.argv = original_argv


class TestQueryStreamProtocolOrder:
    """Verify NDJSON line ordering and cancellation cleanup."""

    @staticmethod
    def _build_client_with_mock(query_error: Exception | None = None):
        original_argv = sys.argv.copy()
        sys.argv = ["lightrag-server"]
        from lightrag.api.config import parse_args
        from lightrag.api.lightrag_server import create_app

        args = parse_args()
        mock_rag = MagicMock()
        mock_result = {
            "llm_response": {"is_streaming": False, "content": "test response"},
            "data": {"references": [{"reference_id": "1", "file_path": "/doc.pdf"}]},
        }

        async def _fake_aquery(*a, **kw):
            cb = kw.get("progress_callback")
            if cb:
                await cb("extracting_keywords")
            if query_error:
                raise query_error
            return mock_result

        mock_rag.aquery_llm = MagicMock(side_effect=_fake_aquery)

        with patch("lightrag.api.lightrag_server.LightRAG", return_value=mock_rag):
            app = create_app(args)

        return TestClient(app), original_argv

    @staticmethod
    def _parse_ndjson(body: str) -> list[dict]:
        return [json.loads(line.strip()) for line in body.splitlines() if line.strip()]

    def test_references_first_without_progress(self):
        client, original_argv = self._build_client_with_mock()
        try:
            response = client.post(
                "/query/stream",
                json={"query": "test", "mode": "mix", "include_references": True},
            )
            assert response.status_code == 200
            lines = self._parse_ndjson(response.text)
            assert len(lines) > 0
            assert "references" in lines[0]
            assert not any("progress" in item for item in lines)
            assert not any("response_time" in item for item in lines)
        finally:
            sys.argv = original_argv

    def test_progress_precedes_references_when_opted_in(self):
        client, original_argv = self._build_client_with_mock()
        try:
            response = client.post(
                "/query/stream",
                json={
                    "query": "test",
                    "mode": "mix",
                    "include_references": True,
                    "include_progress": True,
                },
            )
            assert response.status_code == 200
            lines = self._parse_ndjson(response.text)
            assert len(lines) >= 2
            assert "progress" in lines[0]
            ref_lines = [item for item in lines if "references" in item]
            assert len(ref_lines) > 0
            first_progress_idx = next(
                i for i, item in enumerate(lines) if "progress" in item
            )
            first_ref_idx = next(
                i for i, item in enumerate(lines) if "references" in item
            )
            assert first_progress_idx < first_ref_idx
            assert "response_time" in lines[-1]
        finally:
            sys.argv = original_argv

    def test_progress_query_failure_emits_structured_error(self):
        client, original_argv = self._build_client_with_mock(
            RuntimeError("query failed")
        )
        try:
            response = client.post(
                "/query/stream",
                json={"query": "test", "mode": "mix", "include_progress": True},
            )
            assert response.status_code == 200
            lines = self._parse_ndjson(response.text)
            assert set(lines[-1]) == {"error"}
            assert lines[-1]["error"].startswith("Internal server error (error_id: ")
            assert "query failed" not in response.text
            assert not any("response_time" in item for item in lines)
        finally:
            sys.argv = original_argv

    @pytest.mark.asyncio
    async def test_disconnect_awaits_background_query_cancellation(self):
        from lightrag.api.routers.query_routes import QueryRequest, create_query_routes

        cleanup_complete = asyncio.Event()

        class HangingRag:
            async def aquery_llm(self, *args, **kwargs):
                callback = kwargs["progress_callback"]
                await callback("extracting_keywords")
                try:
                    await asyncio.Event().wait()
                finally:
                    await asyncio.sleep(0)
                    cleanup_complete.set()

        router = create_query_routes(HangingRag())
        endpoint = next(
            route.endpoint for route in router.routes if route.path == "/query/stream"
        )
        response = await endpoint(
            QueryRequest(query="test", mode="mix", include_progress=True)
        )
        iterator = response.body_iterator

        first_line = await anext(iterator)
        assert json.loads(first_line) == {"progress": "extracting_keywords"}

        await iterator.aclose()
        assert cleanup_complete.is_set()


# --------------------------------------------------------------------------- #
# Streamed error line sanitization (CWE-209)
# --------------------------------------------------------------------------- #

_SECRET = "Upstream failure at https://internal-llm.corp.example:8443/v1 deployment=secret-deploy-123"
_NEEDLES = ("internal-llm.corp.example", ":8443/v1", "secret-deploy-123", _SECRET)
_SAFE_MESSAGE = re.compile(r"^Internal server error \(error_id: ([0-9a-f]{12})\)$")


class _ProviderError(RuntimeError):
    """Sentinel provider exception for streaming sanitization tests."""


async def _failing_stream(*_args, **_kwargs):
    yield "partial "
    raise _ProviderError(_SECRET)


def _assert_no_secret(text: str) -> None:
    for needle in _NEEDLES:
        assert needle not in text, f"stream body leaked {needle!r}"


def _ndjson(body: str) -> list[dict]:
    return [json.loads(line) for line in body.splitlines() if line.strip()]


def _error_id(message: str) -> str:
    match = _SAFE_MESSAGE.match(message)
    assert match, f"not a client-safe error message: {message!r}"
    return match.group(1)


def _assert_logged_under(records, error_id: str) -> None:
    matching = [r for r in records if f"error_id={error_id}" in r.getMessage()]
    assert len(matching) == 1
    assert _SECRET in matching[0].getMessage()
    assert matching[0].exc_info is not None


class _QueryRAG:
    def __init__(self, *, fail_before_stream: bool = False):
        self.fail_before_stream = fail_before_stream

    async def aquery_llm(self, *_args, **kwargs):
        callback = kwargs.get("progress_callback")
        if callback:
            await callback("extracting_keywords")
        if self.fail_before_stream:
            raise _ProviderError(_SECRET)
        return {
            "llm_response": {
                "is_streaming": True,
                "response_iterator": _failing_stream(),
            },
            "data": {"references": []},
        }


def _query_client(rag) -> TestClient:
    app = FastAPI()
    app.include_router(_query_routes.create_query_routes(rag))
    return TestClient(app)


class _OllamaRAG:
    def __init__(self):
        self.ollama_server_infos = SimpleNamespace(
            LIGHTRAG_MODEL="lightrag:latest",
            LIGHTRAG_SIZE=1,
            LIGHTRAG_CREATED_AT="now",
            LIGHTRAG_DIGEST="sha",
        )
        self.role_llm_kwargs = {"query": None}
        self.llm_model_kwargs = {}
        self.role_llm_funcs = {"query": self._llm}

    async def _llm(self, *_args, **_kwargs):
        return _failing_stream()

    async def aquery(self, *_args, **_kwargs):
        return _failing_stream()


def _assert_ollama_error_stream(response, records, content_of) -> None:
    assert response.status_code == 200
    _assert_no_secret(response.text)
    lines = _ndjson(response.text)
    assert content_of(lines[0]) == "partial "
    error_line = next(line for line in lines if "error" in line)
    message = error_line["error"].removeprefix("\n\nError: ")
    assert content_of(error_line) == f"\n\nError: {message}"
    _assert_logged_under(records, _error_id(message))
    assert lines[-1]["done"] is True


class TestStreamErrorSanitization:
    """Streamed error lines must never carry raw exception text (CWE-209)."""

    @pytest.fixture
    def ollama_client(self, monkeypatch):
        async def _count_tokens(_text):
            return 0

        monkeypatch.setattr(_ollama_api, "aestimate_tokens", _count_tokens)
        app = FastAPI()
        app.include_router(_ollama_api.OllamaAPI(_OllamaRAG()).router, prefix="/api")
        return TestClient(app)

    def test_stream_error_message_hides_text_and_logs_it_with_the_id(
        self, lightrag_log_records
    ):
        try:
            raise _ProviderError(_SECRET)
        except _ProviderError as e:
            message = _utils_api.stream_error_message(e, "Streaming probe")

        _assert_no_secret(message)
        _assert_logged_under(lightrag_log_records, _error_id(message))

    def test_stream_error_message_mints_a_fresh_id_per_call(self):
        exc = _ProviderError(_SECRET)
        first = _utils_api.stream_error_message(exc, "probe")
        second = _utils_api.stream_error_message(exc, "probe")
        assert _error_id(first) != _error_id(second)

    @pytest.mark.parametrize("include_progress", [False, True])
    def test_query_stream_mid_stream_failure_is_sanitized(
        self, include_progress, lightrag_log_records
    ):
        response = _query_client(_QueryRAG()).post(
            "/query/stream",
            json={"query": "test", "mode": "mix", "include_progress": include_progress},
        )

        assert response.status_code == 200
        _assert_no_secret(response.text)
        lines = _ndjson(response.text)
        assert {"response": "partial "} in lines
        errors = [line["error"] for line in lines if "error" in line]
        assert len(errors) == 1
        _assert_logged_under(lightrag_log_records, _error_id(errors[0]))

    def test_query_stream_background_query_failure_is_sanitized(
        self, lightrag_log_records
    ):
        response = _query_client(_QueryRAG(fail_before_stream=True)).post(
            "/query/stream",
            json={"query": "test", "mode": "mix", "include_progress": True},
        )

        assert response.status_code == 200
        _assert_no_secret(response.text)
        lines = _ndjson(response.text)
        assert lines[0] == {"progress": "extracting_keywords"}
        assert set(lines[-1]) == {"error"}
        assert not any("response_time" in line for line in lines)
        _assert_logged_under(lightrag_log_records, _error_id(lines[-1]["error"]))

    def test_generate_stream_failure_is_sanitized(
        self, ollama_client, lightrag_log_records
    ):
        response = ollama_client.post(
            "/api/generate",
            json={"model": "lightrag:latest", "prompt": "hello", "stream": True},
        )
        _assert_ollama_error_stream(
            response, lightrag_log_records, lambda line: line["response"]
        )

    @pytest.mark.parametrize(
        "content",
        [
            "/bypass hello",
            "what does the knowledge base say about widgets",
        ],
    )
    def test_chat_stream_failure_is_sanitized(
        self, ollama_client, lightrag_log_records, content
    ):
        response = ollama_client.post(
            "/api/chat",
            json={
                "model": "lightrag:latest",
                "messages": [{"role": "user", "content": content}],
                "stream": True,
            },
        )
        _assert_ollama_error_stream(
            response, lightrag_log_records, lambda line: line["message"]["content"]
        )
