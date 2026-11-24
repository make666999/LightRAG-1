"""Query route responses: failure status reporting, llm_generated verdict, validation responses, and concurrent query tokenizer offload."""

from __future__ import annotations

import asyncio
import importlib
import json
import re
import sys
import threading
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from lightrag import LightRAG, operate
from lightrag.base import QueryResult
from lightrag.prompt import PROMPTS
from lightrag.utils import Tokenizer

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_ollama_api = importlib.import_module("lightrag.api.routers.ollama_api")
_query_routes = importlib.import_module("lightrag.api.routers.query_routes")
_utils_api = importlib.import_module("lightrag.api.utils_api")
_server = importlib.import_module("lightrag.api.lightrag_server")
sys.argv = _original_argv

QueryRequest = _query_routes.QueryRequest

pytestmark = pytest.mark.offline


# --------------------------------------------------------------------------- #
# Failed query reporting (HTTP 500 / inline error without raw provider text)
# --------------------------------------------------------------------------- #

_SECRET = "Request to https://internal-llm.corp.example:8443/v1 timed out (deployment=secret-deploy-123)"
_NEEDLES = ("internal-llm.corp.example", ":8443/v1", "secret-deploy-123", "timed out")
_SAFE_MESSAGE = re.compile(r"^Internal server error \(error_id: [0-9a-f]{12}\)$")
_OLD_PLACEHOLDERS = ("No relevant context found", "No response generated")
_KEYWORDS = json.dumps(
    {"high_level_keywords": ["evidence"], "low_level_keywords": ["evidence"]}
)


class _CharacterTokenizer:
    def encode(self, content):
        return [ord(char) for char in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


def _assert_failure_body_is_safe(text: str) -> None:
    for needle in _NEEDLES + _OLD_PLACEHOLDERS:
        assert needle not in text, f"response body carries {needle!r}"


def _query_client(rag) -> TestClient:
    app = FastAPI()
    app.include_router(_query_routes.create_query_routes(rag))
    return TestClient(app)


class TestQueryFailureStatus:
    """A failed query is reported as a failure with no raw exception text."""

    @pytest.fixture(autouse=True)
    def _open_auth(self, monkeypatch):
        monkeypatch.setattr(_utils_api, "auth_configured", False)

    @pytest.fixture
    def failing_rag(self, monkeypatch):
        chunk = {
            "id": "chunk-1",
            "chunk_id": "chunk-1",
            "content": "Retrieved evidence.",
            "file_path": "source.md",
        }
        search = {
            "final_entities": [],
            "final_relations": [],
            "vector_chunks": [chunk],
            "chunk_tracking": {"chunk-1": {"source": "C", "frequency": 1, "order": 1}},
            "query_embedding": None,
        }
        monkeypatch.setattr(
            "lightrag.operate._perform_kg_search", AsyncMock(return_value=search)
        )
        answering_llm = AsyncMock(side_effect=TimeoutError(_SECRET))
        config = {
            "tokenizer": Tokenizer("character", _CharacterTokenizer()),
            "role_llm_funcs": {
                "query": answering_llm,
                "keyword": AsyncMock(return_value=_KEYWORDS),
            },
            "addon_params": {"language": "en"},
        }
        rag = SimpleNamespace(
            _build_global_config=lambda: config,
            _query_done=AsyncMock(),
            chunk_entity_relation_graph=None,
            entities_vdb=None,
            relationships_vdb=None,
            text_chunks=SimpleNamespace(global_config=config),
            chunks_vdb=SimpleNamespace(
                cosine_better_than_threshold=0.0, query=AsyncMock(return_value=[chunk])
            ),
            llm_response_cache=SimpleNamespace(
                global_config={"enable_llm_cache": False},
                get_by_id=AsyncMock(return_value=None),
                upsert=AsyncMock(),
            ),
            ollama_server_infos=SimpleNamespace(
                LIGHTRAG_MODEL="lightrag:latest",
                LIGHTRAG_SIZE=1,
                LIGHTRAG_CREATED_AT="now",
                LIGHTRAG_DIGEST="sha",
            ),
            role_llm_funcs=config["role_llm_funcs"],
            role_llm_kwargs={"query": None},
            llm_model_kwargs={},
        )
        rag.aquery_llm = partial(LightRAG.aquery_llm, rag)
        rag.aquery = partial(LightRAG.aquery, rag)
        return rag, answering_llm

    @pytest.fixture
    def ollama_client(self, monkeypatch, failing_rag):
        async def _count_tokens(_text):
            return 0

        monkeypatch.setattr(_ollama_api, "aestimate_tokens", _count_tokens)
        app = FastAPI()
        app.include_router(
            _ollama_api.OllamaAPI(failing_rag[0]).router, prefix="/api"
        )
        return TestClient(app)

    @pytest.mark.parametrize("mode", ["mix", "naive", "bypass"])
    def test_query_reports_a_failed_llm_call_as_500(self, failing_rag, mode):
        rag, answering_llm = failing_rag

        response = _query_client(rag).post(
            "/query", json={"query": "Find evidence", "mode": mode}
        )

        assert response.status_code == 500
        assert _SAFE_MESSAGE.match(response.json()["detail"])
        _assert_failure_body_is_safe(response.text)
        answering_llm.assert_awaited_once()

    @pytest.mark.parametrize("stream", [True, False])
    @pytest.mark.parametrize("mode", ["mix", "bypass"])
    def test_query_stream_reports_a_failure_before_the_stream_as_500(
        self, failing_rag, mode, stream
    ):
        rag, answering_llm = failing_rag

        response = _query_client(rag).post(
            "/query/stream",
            json={"query": "Find evidence", "mode": mode, "stream": stream},
        )

        assert response.status_code == 500
        assert _SAFE_MESSAGE.match(response.json()["detail"])
        _assert_failure_body_is_safe(response.text)
        answering_llm.assert_awaited_once()

    def test_query_stream_with_progress_reports_the_failure_as_an_error_line(
        self, failing_rag
    ):
        rag, answering_llm = failing_rag

        response = _query_client(rag).post(
            "/query/stream",
            json={"query": "Find evidence", "mode": "mix", "include_progress": True},
        )

        assert response.status_code == 200
        _assert_failure_body_is_safe(response.text)
        lines = [json.loads(line) for line in response.text.splitlines() if line]
        assert set(lines[-1]) == {"error"}
        assert _SAFE_MESSAGE.match(lines[-1]["error"])
        assert not any("response" in line for line in lines)
        answering_llm.assert_awaited_once()

    @pytest.mark.parametrize("stream", [True, False])
    def test_ollama_chat_reports_a_failed_query_as_500(
        self, failing_rag, ollama_client, stream
    ):
        _, answering_llm = failing_rag

        response = ollama_client.post(
            "/api/chat",
            json={
                "model": "lightrag:latest",
                "messages": [{"role": "user", "content": "Find evidence"}],
                "stream": stream,
            },
        )

        assert response.status_code == 500
        assert _SAFE_MESSAGE.match(response.json()["detail"])
        _assert_failure_body_is_safe(response.text)
        answering_llm.assert_awaited_once()


# --------------------------------------------------------------------------- #
# llm_generated verdict on /query and /query/stream
# --------------------------------------------------------------------------- #

_LLM_GEN_ENV_VARS = (
    "LLM_BINDING",
    "EMBEDDING_BINDING",
    "OPENAI_API_KEY",
    "AUTH_ACCOUNTS",
    "TOKEN_SECRET",
    "LIGHTRAG_API_KEY",
    "WHITELIST_PATHS",
    "LIGHTRAG_API_PREFIX",
)


def _build_llm_generated_client(llm_response: dict) -> TestClient:
    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server"]
        from lightrag.api.config import parse_args
        from lightrag.api.lightrag_server import create_app

        args = parse_args()
        mock_rag = MagicMock()

        async def _fake_aquery(*_a, **_kw):
            return {"llm_response": llm_response, "data": {"references": []}}

        mock_rag.aquery_llm = MagicMock(side_effect=_fake_aquery)

        with patch("lightrag.api.lightrag_server.LightRAG", return_value=mock_rag):
            app = create_app(args)
        return TestClient(app)
    finally:
        sys.argv = original_argv


def _ndjson_lines(response) -> list[dict]:
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


class TestQueryLlmGenerated:
    """The `llm_generated` verdict on /query and /query/stream."""

    @pytest.fixture(autouse=True)
    def _isolate_env(self, monkeypatch):
        for var in _LLM_GEN_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("AUTH_ACCOUNTS", "")
        monkeypatch.setenv("TOKEN_SECRET", "")
        monkeypatch.setenv("LIGHTRAG_API_KEY", "")
        monkeypatch.setenv("LLM_BINDING", "openai")
        monkeypatch.setenv("EMBEDDING_BINDING", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        auth_module = sys.modules.get("lightrag.api.auth")
        if auth_module is not None:
            monkeypatch.setattr(
                auth_module.auth_handler, "accounts", {}, raising=False
            )
        utils_api_module = sys.modules.get("lightrag.api.utils_api")
        if utils_api_module is not None:
            monkeypatch.setattr(
                utils_api_module, "auth_configured", False, raising=False
            )

        import lightrag.api.config as config

        config._global_args = None
        config._initialized = False
        yield
        config._global_args = None
        config._initialized = False

    def test_query_result_is_llm_generated_by_default(self):
        assert QueryResult(content="an answer").llm_generated is True

    @pytest.mark.asyncio
    async def test_canned_no_context_result_is_not_llm_generated(self):
        from lightrag.operate import kg_query

        result = await kg_query("", None, None, None, None, MagicMock(), {})

        assert result.content == PROMPTS["fail_response"]
        assert result.llm_generated is False

    def test_query_reports_a_generated_answer(self):
        client = _build_llm_generated_client(
            {"is_streaming": False, "content": "A real answer.", "llm_generated": True}
        )

        body = client.post("/query", json={"query": "test query", "mode": "mix"}).json()

        assert body["response"] == "A real answer."
        assert body["llm_generated"] is True

    def test_query_reports_the_canned_no_context_reply(self):
        client = _build_llm_generated_client(
            {
                "is_streaming": False,
                "content": PROMPTS["fail_response"],
                "llm_generated": False,
            }
        )

        body = client.post("/query", json={"query": "test query", "mode": "mix"}).json()

        assert body["response"] == PROMPTS["fail_response"]
        assert body["llm_generated"] is False

    def test_query_placeholder_for_empty_content_is_not_generated(self):
        client = _build_llm_generated_client({"is_streaming": False, "content": ""})

        body = client.post("/query", json={"query": "test query", "mode": "mix"}).json()

        assert body["response"] == "No relevant context found for the query."
        assert body["llm_generated"] is False

    def test_query_defaults_to_generated_when_the_result_omits_the_flag(self):
        client = _build_llm_generated_client(
            {"is_streaming": False, "content": "An answer."}
        )

        body = client.post("/query", json={"query": "test query", "mode": "mix"}).json()

        assert body["llm_generated"] is True

    def test_stream_complete_response_line_carries_the_verdict(self):
        client = _build_llm_generated_client(
            {
                "is_streaming": False,
                "content": PROMPTS["fail_response"],
                "llm_generated": False,
            }
        )

        response = client.post(
            "/query/stream", json={"query": "test query", "mode": "mix"}
        )
        lines = _ndjson_lines(response)

        content_lines = [line for line in lines if "response" in line]
        assert len(content_lines) == 1
        assert content_lines[0]["response"] == PROMPTS["fail_response"]
        assert content_lines[0]["llm_generated"] is False

    def test_streamed_chunks_carry_no_verdict(self):
        async def _chunks():
            for part in ("Hello", " world"):
                yield part

        client = _build_llm_generated_client(
            {"is_streaming": True, "response_iterator": _chunks()}
        )

        response = client.post(
            "/query/stream", json={"query": "test query", "mode": "mix"}
        )
        lines = _ndjson_lines(response)

        content_lines = [line for line in lines if "response" in line]
        assert [line["response"] for line in content_lines] == ["Hello", " world"]
        assert all("llm_generated" not in line for line in content_lines)


# --------------------------------------------------------------------------- #
# Validation exception handler serializability (/query 422 and /query/data 400)
# --------------------------------------------------------------------------- #


class TestQueryValidationResponses:
    """Validation refusals reach the client as 422/400, not as 500s."""

    @pytest.fixture
    def client(self):
        app = FastAPI()
        app.add_exception_handler(
            RequestValidationError, _server.validation_exception_handler
        )

        @app.post("/query")
        async def _query(request: QueryRequest):  # pragma: no cover - never reached
            return {"response": "ok"}

        @app.post("/query/data")
        async def _data(request: QueryRequest):  # pragma: no cover - never reached
            return {"status": "success"}

        return TestClient(app, raise_server_exceptions=False)

    @pytest.mark.parametrize("body", [{"query": "中"}, {"query": "   "}])
    def test_query_refusals_are_422_with_a_serializable_body(self, client, body):
        response = client.post("/query", json=body)

        assert response.status_code == 422
        detail = response.json()["detail"]
        assert isinstance(detail, list) and detail
        assert detail[0]["type"] == "value_error"
        assert detail[0]["loc"][0] == "body"
        assert detail[0]["msg"].startswith("Value error, ")

    @pytest.mark.parametrize("body", [{"query": "中"}, {"query": "   "}])
    def test_query_data_refusals_keep_their_own_400_envelope(self, client, body):
        response = client.post("/query/data", json=body)

        assert response.status_code == 400
        payload = response.json()
        assert payload["status"] == "failure"
        assert payload["message"].startswith("Validation error: body")
        assert "Value error, " in payload["message"]
        assert payload["data"] == {} and payload["metadata"] == {}

    def test_structural_errors_still_answer_422(self, client):
        response = client.post("/query", json={})

        assert response.status_code == 422
        assert response.json()["detail"][0]["type"] == "missing"


# --------------------------------------------------------------------------- #
# Concurrent queries sharing tokenizer during heading backfill
# --------------------------------------------------------------------------- #


class _SlowTokenizer:
    """Encodes slowly and records whether it is ever entered concurrently."""

    def __init__(self, delay: float = 0.02):
        self._delay = delay
        self._inside = False
        self._guard = threading.Lock()
        self.reentered = False
        self.calls = 0

    def encode(self, content: str):
        with self._guard:
            if self._inside:
                self.reentered = True
            self._inside = True
            self.calls += 1
        try:
            threading.Event().wait(self._delay)
            return list(range(len(content)))
        finally:
            with self._guard:
                self._inside = False

    def decode(self, tokens):
        return "x" * len(tokens)


class _TextChunksDB:
    def __init__(self, tokenizer, rows):
        self.global_config = {"tokenizer": tokenizer}
        self._rows = rows

    async def get_by_ids(self, ids):
        return [self._rows.get(chunk_id) for chunk_id in ids]


def _heading_rows(count: int) -> dict:
    return {
        f"chunk-{i}": {
            "heading": {
                "parent_headings": [
                    "A very long top level heading that will need truncating",
                    "A second level heading that is also quite long indeed",
                    "A third level heading pushing the breadcrumb over budget",
                ]
            }
        }
        for i in range(count)
    }


async def _heartbeat(stop: asyncio.Event) -> int:
    beats = 0
    while not stop.is_set():
        beats += 1
        await asyncio.sleep(0)
    return beats


async def test_heading_backfill_does_not_freeze_the_event_loop():
    tokenizer = _SlowTokenizer(delay=0.02)
    rows = _heading_rows(4)
    db = _TextChunksDB(tokenizer, rows)
    chunks = [{"chunk_id": chunk_id} for chunk_id in rows]

    stop = asyncio.Event()
    pulse = asyncio.create_task(_heartbeat(stop))
    try:
        await operate._attach_content_headings(chunks, db)
    finally:
        stop.set()
    beats = await pulse

    assert tokenizer.calls > 0
    assert beats > 1


async def test_two_concurrent_queries_share_the_tokenizer_without_reentering_it():
    tokenizer = _SlowTokenizer(delay=0.01)
    rows = _heading_rows(3)

    async def _one_query():
        db = _TextChunksDB(tokenizer, rows)
        chunks = [{"chunk_id": chunk_id} for chunk_id in rows]
        await operate._attach_content_headings(chunks, db)
        return chunks

    stop = asyncio.Event()
    pulse = asyncio.create_task(_heartbeat(stop))
    try:
        first, second = await asyncio.wait_for(
            asyncio.gather(_one_query(), _one_query()), timeout=10.0
        )
    finally:
        stop.set()
    beats = await pulse

    assert tokenizer.reentered is False
    assert beats > 1
    assert all("content_headings" in chunk for chunk in first)
    assert all("content_headings" in chunk for chunk in second)


async def test_backfill_output_is_unchanged_by_the_offload():
    tokenizer = _SlowTokenizer(delay=0.0)
    rows = _heading_rows(2)
    db = _TextChunksDB(tokenizer, rows)
    chunks = [{"chunk_id": chunk_id} for chunk_id in rows]

    await operate._attach_content_headings(chunks, db)

    expected = operate._truncate_section_context(
        operate.format_parent_headings(next(iter(rows.values()))),
        tokenizer,
        operate.DEFAULT_MAX_SECTION_CONTEXT_TOKENS,
    )
    assert all(chunk["content_headings"] == expected for chunk in chunks)


async def test_missing_rows_are_skipped_without_touching_the_tokenizer():
    tokenizer = _SlowTokenizer(delay=0.0)
    db = _TextChunksDB(tokenizer, {})
    chunks = [{"chunk_id": "absent"}]

    await operate._attach_content_headings(chunks, db)

    assert chunks == [{"chunk_id": "absent"}]
    assert tokenizer.calls == 0


async def test_no_chunks_is_a_no_op():
    tokenizer = _SlowTokenizer(delay=0.0)
    db = _TextChunksDB(tokenizer, {})
    await operate._attach_content_headings([], db)
    await operate._attach_content_headings([{"chunk_id": "x"}], None)
    assert tokenizer.calls == 0


def test_slow_tokenizer_stub_matches_the_real_interface():
    assert isinstance(SimpleNamespace(), object)
    tokenizer = _SlowTokenizer(delay=0.0)
    assert tokenizer.decode(tokenizer.encode("abc")) == "xxx"
