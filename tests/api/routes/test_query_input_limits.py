"""Input ceilings on QueryRequest (GHSA-r8jh-295g-vv42).

Fixing the whitelist would narrow who can trigger the stall but not remove it:
an authenticated 8 MiB ``/query`` blocks the loop too, because the graph modes
tokenize the keyword prompt before retrieval runs. These tests bound the fields
that feed that work.

They validate the model rather than driving HTTP: ``QueryRequest`` is what the
three query routes share, and a rejection here is by construction a rejection
before the handler — and therefore before any tokenizer or LLM call — runs.
"""

from __future__ import annotations

import importlib
import sys

import pytest
from pydantic import ValidationError

# lightrag.api.config parses argv at import time, so the import has to be guarded
# the way the rest of the API tests guard it.
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_query_routes = importlib.import_module("lightrag.api.routers.query_routes")
sys.argv = _original_argv

QueryRequest = _query_routes.QueryRequest

from lightrag.constants import (  # noqa: E402
    MAX_KEYWORD_CHARS,
    MAX_KEYWORDS_PER_LIST,
    MAX_MESSAGE_CHARS,
    MAX_MESSAGES_PER_REQUEST,
    MAX_QUERY_CHARS,
    MAX_QUERY_TOKEN_BUDGET,
    MAX_QUERY_TOP_K,
    MAX_REQUEST_TEXT_CHARS,
    MAX_RESPONSE_TYPE_CHARS,
    MAX_ROLE_CHARS,
)

pytestmark = pytest.mark.offline


def _build(**overrides) -> QueryRequest:
    payload = {"query": "what is this about"}
    payload.update(overrides)
    return QueryRequest(**payload)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        pytest.param({"query": "q" * (MAX_QUERY_CHARS + 1)}, None, id="query"),
        pytest.param(
            {"user_prompt": "p" * (MAX_QUERY_CHARS + 1)}, None, id="user-prompt"
        ),
        pytest.param(
            {"response_type": "r" * (MAX_RESPONSE_TYPE_CHARS + 1)},
            None,
            id="response-type",
        ),
        pytest.param(
            {
                "conversation_history": [
                    {"role": "u" * (MAX_ROLE_CHARS + 1), "content": "hi"}
                ]
            },
            "Each message 'role' must be at most",
            id="history-role",
        ),
        pytest.param(
            {
                "conversation_history": [
                    {"role": "user", "content": "c" * (MAX_MESSAGE_CHARS + 1)}
                ]
            },
            None,
            id="history-content",
        ),
        pytest.param(
            {
                "conversation_history": [
                    {"role": "user", "content": "hi"}
                    for _ in range(MAX_MESSAGES_PER_REQUEST + 1)
                ]
            },
            None,
            id="history-message-count",
        ),
        pytest.param(
            {"hl_keywords": ["k"] * (MAX_KEYWORDS_PER_LIST + 1)},
            None,
            id="hl-keywords-count",
        ),
        pytest.param(
            {"ll_keywords": ["k" * (MAX_KEYWORD_CHARS + 1)]},
            None,
            id="ll-keyword-length",
        ),
    ],
)
def test_per_field_limit_refused(overrides, match):
    with pytest.raises(ValidationError, match=match):
        _build(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param(
            {
                "query": "q" * MAX_QUERY_CHARS,
                "user_prompt": "p" * MAX_QUERY_CHARS,
                "conversation_history": [{"role": "user", "content": "one more"}],
            },
            id="query-plus-user-prompt-plus-message",
        ),
        pytest.param(
            {
                "query": "q" * MAX_QUERY_CHARS,
                "conversation_history": [
                    {
                        "role": "user",
                        "content": "hi",
                        "metadata": "x" * (MAX_REQUEST_TEXT_CHARS - MAX_QUERY_CHARS),
                    }
                ],
            },
            id="query-plus-unenumerated-history-extra",
        ),
        pytest.param(
            {
                "conversation_history": [
                    {
                        "role": "user",
                        "content": "hi",
                        "metadata": "x" * (MAX_REQUEST_TEXT_CHARS // 4),
                    }
                    for _ in range(8)
                ]
            },
            id="history-extras-alone",
        ),
    ],
)
def test_aggregate_text_budget_refused(overrides):
    """Per-field legal inputs still cannot combine past MAX_REQUEST_TEXT_CHARS."""
    assert MAX_QUERY_CHARS * 2 == MAX_REQUEST_TEXT_CHARS
    with pytest.raises(ValidationError, match="total request text"):
        _build(**overrides)


@pytest.mark.parametrize(
    ("field", "ceiling"),
    [
        ("top_k", MAX_QUERY_TOP_K),
        ("chunk_top_k", MAX_QUERY_TOP_K),
        ("max_entity_tokens", MAX_QUERY_TOKEN_BUDGET),
        ("max_relation_tokens", MAX_QUERY_TOKEN_BUDGET),
        ("max_total_tokens", MAX_QUERY_TOKEN_BUDGET),
    ],
)
def test_numeric_knobs_are_bounded(field, ceiling):
    """Character limits alone do not bound retrieval width or token budgets."""
    _build(**{field: ceiling})
    with pytest.raises(ValidationError):
        _build(**{field: ceiling + 1})


def test_ordinary_request_and_extra_history_keys_accepted():
    request = _build(
        query="  what is this about  ",
        top_k=60,
        chunk_top_k=20,
        hl_keywords=["alpha", "beta"],
        conversation_history=[
            {"role": "user", "content": "hi", "name": "alice", "tool_call_id": "t-1"}
        ],
    )
    assert request.query == "what is this about"
    assert request.top_k == 60
    assert request.conversation_history[0]["name"] == "alice"
    assert len(request.query) + len("hi") < MAX_REQUEST_TEXT_CHARS


def test_a_realistic_conversation_is_accepted():
    """The count cap is a sanity guard, not a boundary; it must not bite."""
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": "hello there " * 20}
        for i in range(MAX_MESSAGES_PER_REQUEST)
    ]
    request = _build(conversation_history=history)
    assert len(request.conversation_history) == MAX_MESSAGES_PER_REQUEST
