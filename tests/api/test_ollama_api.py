"""Offline tests for Ollama-compatible API role kwargs and query mode parsing."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_ollama_api = importlib.import_module("lightrag.api.routers.ollama_api")
sys.argv = _original_argv

SearchMode = _ollama_api.SearchMode
parse_query_mode = _ollama_api.parse_query_mode

pytestmark = pytest.mark.offline


# --------------------------------------------------------------------------- #
# Query-role kwargs isolation on /api/generate and /api/chat
# --------------------------------------------------------------------------- #


class _FakeRag:
    def __init__(self):
        self.base_calls = []
        self.query_calls = []
        self.llm_model_kwargs = {"route": "base"}
        self._query_kwargs = {"route": "query"}
        self.ollama_server_infos = SimpleNamespace(
            LIGHTRAG_MODEL="lightrag:latest",
            LIGHTRAG_CREATED_AT="2026-03-14T00:00:00Z",
            LIGHTRAG_SIZE=0,
        )

        async def base_func(*args, **kwargs):
            self.base_calls.append(kwargs)
            return "base"

        async def query_func(*args, **kwargs):
            self.query_calls.append(kwargs)
            return "query"

        self.llm_model_func = base_func
        self.role_llm_funcs = {"query": query_func}
        self.role_llm_kwargs = {"query": self._query_kwargs}

    async def aquery(self, *args, **kwargs):
        return "aquery"


def _make_client(monkeypatch) -> tuple[TestClient, _FakeRag]:
    monkeypatch.setattr(sys, "argv", [sys.argv[0]])
    for module_name in [
        "lightrag.api.config",
        "lightrag.api.auth",
        "lightrag.api.utils_api",
        "lightrag.api.routers",
        "lightrag.api.routers.ollama_api",
    ]:
        sys.modules.pop(module_name, None)
    ollama_api_module = importlib.import_module("lightrag.api.routers.ollama_api")
    rag = _FakeRag()
    api = ollama_api_module.OllamaAPI(rag, top_k=20, api_key=None)
    app = FastAPI()
    app.include_router(api.router, prefix="/api")
    return TestClient(app), rag


def test_generate_non_stream_uses_query_role_kwargs_without_mutating_base(monkeypatch):
    client, rag = _make_client(monkeypatch)

    response = client.post(
        "/api/generate",
        json={
            "model": "lightrag:latest",
            "prompt": "Summarize this",
            "stream": False,
            "system": "custom system",
        },
    )

    assert response.status_code == 200
    assert rag.base_calls == []
    assert rag.query_calls[-1]["route"] == "query"
    assert rag.query_calls[-1]["system_prompt"] == "custom system"
    assert "system_prompt" not in rag.llm_model_kwargs
    assert "system_prompt" not in rag.role_llm_kwargs["query"]


def test_chat_bypass_stream_uses_query_role_kwargs_without_mutating_base(monkeypatch):
    client, rag = _make_client(monkeypatch)

    with client.stream(
        "POST",
        "/api/chat",
        json={
            "model": "lightrag:latest",
            "stream": True,
            "system": "chat system",
            "messages": [
                {"role": "assistant", "content": "history"},
                {"role": "user", "content": "/bypass give me a title"},
            ],
        },
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    assert rag.base_calls == []
    assert rag.query_calls[-1]["route"] == "query"
    assert rag.query_calls[-1]["system_prompt"] == "chat system"
    assert rag.query_calls[-1]["history_messages"] == [
        {"role": "assistant", "content": "history"}
    ]
    assert "system_prompt" not in rag.llm_model_kwargs
    assert "system_prompt" not in rag.role_llm_kwargs["query"]


# --------------------------------------------------------------------------- #
# parse_query_mode prefix & bracket prompt parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "query,expected",
    [
        # Bracket prompt with a query: already worked, must keep working.
        (
            "/local[use mermaid] tell me about X",
            ("tell me about X", SearchMode.local, False, "use mermaid"),
        ),
        # Bracket prompt with no query text: the mode was lost and the literal
        # "/local" became the query.
        ("/local[use mermaid]", ("", SearchMode.local, False, "use mermaid")),
        # No mode prefix: a stray "/ " was prepended to the user's question.
        (
            "/[use mermaid] tell me about X",
            ("tell me about X", SearchMode.mix, False, "use mermaid"),
        ),
        # Context modes never depended on the separator.
        ("/mixcontext[use mermaid]", ("", SearchMode.mix, True, "use mermaid")),
        # An unknown mode keeps its existing pass-through behavior.
        (
            "/nosuchmode[use mermaid] tell me about X",
            ("/nosuchmode tell me about X", SearchMode.mix, False, "use mermaid"),
        ),
        # Non-bracket forms are untouched by the fix.
        ("/local  tell me about X", ("tell me about X", SearchMode.local, False, None)),
        ("/globalcontext X", ("X", SearchMode.global_, True, None)),
        ("tell me about X", ("tell me about X", SearchMode.mix, False, None)),
    ],
)
def test_parse_query_mode(query, expected):
    assert parse_query_mode(query) == expected


@pytest.mark.parametrize(
    "query,expected",
    [
        (
            "/local[be brief] line one\nline two",
            ("line one\nline two", SearchMode.local, False, "be brief"),
        ),
        (
            "/[be brief] line one\nline two",
            ("line one\nline two", SearchMode.mix, False, "be brief"),
        ),
    ],
)
def test_bracket_prompt_keeps_multiline_query(query, expected):
    """A multi-line question must survive the bracket form, as it does without it."""
    assert parse_query_mode(query) == expected
    assert parse_query_mode("/local line one\nline two")[0] == "line one\nline two"


@pytest.mark.parametrize(
    "query,expected",
    [
        (
            "/contextualize the following passage",
            ("/contextualize the following passage", SearchMode.mix, False, None),
        ),
        (
            "/context how does docker networking work",
            ("how does docker networking work", SearchMode.mix, True, None),
        ),
        ("/context", ("", SearchMode.mix, True, None)),
        ("/localcontext foo", ("foo", SearchMode.local, True, None)),
    ],
)
def test_context_keys_end_at_a_word_boundary(query, expected):
    """A context mode key must not prefix-match a longer word."""
    assert parse_query_mode(query) == expected


_PLAIN_MODES = [
    ("local", SearchMode.local),
    ("global", SearchMode.global_),
    ("naive", SearchMode.naive),
    ("hybrid", SearchMode.hybrid),
    ("mix", SearchMode.mix),
    ("bypass", SearchMode.bypass),
]


@pytest.mark.parametrize("separator", ["\n"])
def test_mode_key_accepts_any_whitespace_separator(separator):
    """A newline after the prefix selects the mode, as it does for "/context"."""
    assert parse_query_mode(f"/bypass{separator}line one\nline two") == (
        "line one\nline two",
        SearchMode.bypass,
        False,
        None,
    )


@pytest.mark.parametrize("mode_prefix,mode", _PLAIN_MODES)
def test_bare_mode_key_is_consumed(mode_prefix, mode):
    """A mode key at the end of the query is a prefix, not the question."""
    assert parse_query_mode(f"/{mode_prefix}") == ("", mode, False, None)


@pytest.mark.parametrize("query", ["/localize the failing test"])
def test_mode_keys_end_at_a_word_boundary(query):
    """A longer word that starts with a mode key stays an ordinary query."""
    assert parse_query_mode(query) == (query, SearchMode.mix, False, None)
