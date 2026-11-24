"""QueryRequest field validation: conversation_history, whitespace stripping, and disable_user_prompt_prefix."""

from __future__ import annotations

import importlib
import sys

import pytest
from pydantic import ValidationError

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_query_routes = importlib.import_module("lightrag.api.routers.query_routes")
sys.argv = _original_argv

QueryRequest = _query_routes.QueryRequest

from lightrag.constants import (  # noqa: E402
    MAX_QUERY_CHARS,
    MAX_REQUEST_TEXT_CHARS,
)

pytestmark = pytest.mark.offline


# --------------------------------------------------------------------------- #
# conversation_history content type
# --------------------------------------------------------------------------- #


def test_conversation_history_rejects_non_string_content():
    with pytest.raises(ValidationError):
        QueryRequest(
            query="hello world",
            conversation_history=[{"role": "user", "content": 123}],
        )


def test_conversation_history_rejects_missing_content():
    with pytest.raises(ValidationError):
        QueryRequest(
            query="hello world",
            conversation_history=[{"role": "user"}],
        )


def test_conversation_history_accepts_string_content():
    req = QueryRequest(
        query="hello world",
        conversation_history=[{"role": "user", "content": "hi"}],
    )
    assert req.conversation_history == [{"role": "user", "content": "hi"}]


# --------------------------------------------------------------------------- #
# Whitespace stripping and weighted RAG minimum
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("query", ["ab ", "中"])
def test_query_request_rejects_queries_below_weighted_minimum(query):
    with pytest.raises(ValidationError):
        QueryRequest(query=query)


@pytest.mark.parametrize(
    "query, expected",
    [
        ("  abc  ", "abc"),
        ("中a", "中a"),
    ],
)
def test_query_request_accepts_english_equivalent_weight_three(query, expected):
    assert QueryRequest(query=query).query == expected


@pytest.mark.parametrize("query", [" a "])
def test_query_request_does_not_apply_rag_minimum_to_bypass(query):
    assert QueryRequest(query=query, mode="bypass").query == query.strip()


@pytest.mark.parametrize("mode", ["mix", "bypass"])
@pytest.mark.parametrize("query", ["", "   "])
def test_query_request_rejects_an_empty_query_in_every_mode(mode, query):
    with pytest.raises(ValidationError):
        QueryRequest(query=query, mode=mode)


# --------------------------------------------------------------------------- #
# disable_user_prompt_prefix field and aggregate text budget
# --------------------------------------------------------------------------- #


def _build(**overrides) -> QueryRequest:
    payload = {"query": "what is this about"}
    payload.update(overrides)
    return QueryRequest(**payload)


def test_omitted_switch_falls_back_to_the_queryparam_default():
    param = _build().to_query_params(is_stream=False)
    assert param.disable_user_prompt_prefix is False


@pytest.mark.parametrize("value", [True, False])
def test_explicit_switch_is_forwarded(value):
    param = _build(disable_user_prompt_prefix=value).to_query_params(is_stream=False)
    assert param.disable_user_prompt_prefix is value


def test_switch_survives_alongside_a_user_prompt():
    param = _build(
        user_prompt="Reply in one sentence.", disable_user_prompt_prefix=True
    ).to_query_params(is_stream=False)
    assert param.user_prompt == "Reply in one sentence."
    assert param.disable_user_prompt_prefix is True


def test_switch_is_not_leaked_as_a_queryparam_prefix_field():
    param = _build().to_query_params(is_stream=False)
    assert not hasattr(param, "user_prompt_prefix")


def test_request_at_the_text_budget_still_validates():
    query = "x" * MAX_QUERY_CHARS
    user_prompt = "y" * (MAX_REQUEST_TEXT_CHARS - MAX_QUERY_CHARS)
    request = _build(query=query, user_prompt=user_prompt)
    assert len(request.query) + len(request.user_prompt) == MAX_REQUEST_TEXT_CHARS
    param = request.to_query_params(is_stream=False)
    assert param.disable_user_prompt_prefix is False
