"""Tests for <think> tag stripping, reasoning-only answer rejection, and query/cache handling."""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from tenacity import wait_none

from lightrag import LightRAG
from lightrag.base import QueryContextResult, QueryParam
from lightrag.exceptions import EmptyTruncatedResponseError, ReasoningOnlyResponseError
from lightrag.llm.openai import InvalidResponseError, openai_complete_if_cache
from lightrag.operate import kg_query, naive_query
from lightrag.utils import (
    Tokenizer,
    TruncatedResponse,
    is_reasoning_only_answer,
    is_truncated_response,
    reject_reasoning_only_answer,
    remove_think_tags,
)

pytestmark = pytest.mark.offline

QUERY = "How do I reset the device?"
REASONING = "SECRET-REASONING the manual says to hold the reset button"
ANSWER = "Hold the reset button for ten seconds."


# --------------------------------------------------------------------------- #
# remove_think_tags
# --------------------------------------------------------------------------- #


class TestRemoveThinkTags:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            pytest.param(
                "<think>Let me reason about this...</think>The actual answer.",
                "The actual answer.",
                id="standard-block-at-start",
            ),
            pytest.param(
                "partial reasoning content</think>The actual answer.",
                "The actual answer.",
                id="orphaned-close-tag-at-start",
            ),
            pytest.param(
                "Just a normal response with no tags.",
                "Just a normal response with no tags.",
                id="no-think-tags",
            ),
            pytest.param(
                "Answer about xxx<think>reasoning</think>xxx more content",
                "Answer about xxxxxx more content",
                id="embedded-in-middle",
            ),
            pytest.param(
                "<think>r1</think>Answer<think>r2</think> more",
                "Answer more",
                id="multiple-blocks",
            ),
            pytest.param(
                "<think>line1\nline2\nline3</think>Result.",
                "Result.",
                id="multiline-block",
            ),
            pytest.param("<think></think>Content.", "Content.", id="empty-block"),
            pytest.param("<think>only reasoning</think>", "", id="only-think-block"),
            pytest.param(
                "  <think>reasoning</think>  Answer  ",
                "Answer",
                id="whitespace-stripping",
            ),
            pytest.param(
                "<think>thinking</think>"
                "The path is .//postTransactionAmounts/sharesOwnedFollowingTransaction/value",
                "The path is .//postTransactionAmounts/sharesOwnedFollowingTransaction/value",
                id="xpath-like-content-preserved",
            ),
            pytest.param(
                "2 < 3 reasoning</think>final answer",
                "final answer",
                id="orphaned-close-with-angle-brackets",
            ),
            pytest.param(
                "check <b>bold</b> reasoning</think>The answer.",
                "The answer.",
                id="orphaned-close-with-html",
            ),
        ],
    )
    def test_remove_think_tags_cases(self, text, expected):
        assert remove_think_tags(text) == expected

    def test_truncation_marker_survives_think_tag_removal(self):
        result = remove_think_tags(
            TruncatedResponse("<think>reasoning</think>Partial answer")
        )
        assert result == "Partial answer"
        assert is_truncated_response(result)


# --------------------------------------------------------------------------- #
# reject_reasoning_only_answer / is_reasoning_only_answer
# --------------------------------------------------------------------------- #


class TestRejectReasoningOnlyAnswer:
    @pytest.mark.parametrize(
        "response",
        [
            pytest.param(f"<think>{REASONING}</think>", id="closed-block"),
            pytest.param(f"  <think>{REASONING}</think>  \n", id="padded-block"),
            pytest.param(f"{REASONING}</think>", id="orphaned-close-tag"),
            pytest.param(f"<think>{REASONING}", id="unclosed-block"),
        ],
    )
    def test_reasoning_only_answer_is_rejected(self, response):
        assert is_reasoning_only_answer(response) is True
        with pytest.raises(ReasoningOnlyResponseError):
            reject_reasoning_only_answer(response)

    def test_truncated_reasoning_only_answer_is_a_truncation_error(self):
        with pytest.raises(EmptyTruncatedResponseError):
            reject_reasoning_only_answer(
                TruncatedResponse(f"<think>{REASONING}</think>")
            )

    @pytest.mark.parametrize(
        "response",
        [
            pytest.param(ANSWER, id="plain-answer"),
            pytest.param(f"<think>{REASONING}</think>{ANSWER}", id="trace-plus-answer"),
            pytest.param(
                TruncatedResponse("Hold the reset button for"), id="truncated-partial"
            ),
            pytest.param("", id="empty-string"),
            pytest.param(None, id="none"),
        ],
    )
    def test_answers_and_non_answers_pass_through(self, response):
        assert is_reasoning_only_answer(response) is False
        reject_reasoning_only_answer(response)

    def test_streaming_iterator_passes_through(self):
        async def _stream():
            yield "<think>"

        reject_reasoning_only_answer(_stream())

    def test_log_and_error_carry_lengths_never_content(self, lightrag_log_records):
        response = f"<think>{REASONING}</think>"

        with pytest.raises(ReasoningOnlyResponseError) as caught:
            reject_reasoning_only_answer(response)

        errors = [r for r in lightrag_log_records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert f"response_len={len(response)}" in errors[0].getMessage()
        assert not any(
            "SECRET-REASONING" in r.getMessage() for r in lightrag_log_records
        )
        assert "SECRET-REASONING" not in str(caught.value)
        assert f"response_len={len(response)}" in str(caught.value)


# --------------------------------------------------------------------------- #
# End-to-end query & cache behavior with reasoning-only responses
# --------------------------------------------------------------------------- #


class _FakeTokenizerImpl:
    """Character-level tokenizer stand-in for offline query tests."""

    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


class _RecordingKV:
    """In-memory KV stub recording cache writes and serving an optional hit."""

    def __init__(self, cached_answer=None):
        self.global_config = {"enable_llm_cache": True}
        self.writes = []
        self._cached_answer = cached_answer

    async def get_by_id(self, _key):
        if self._cached_answer is None:
            return None
        return {"return": self._cached_answer, "create_time": 0}

    async def upsert(self, entries):
        self.writes.append(entries)


class _ChunksVDB:
    """Stub vector DB returning one fixed chunk for naive_query."""

    cosine_better_than_threshold = 0.0

    async def query(self, *_args, **_kwargs):
        return [
            {
                "id": "chunk-1",
                "content": ANSWER,
                "file_path": "device/reset.md",
            }
        ]


class _BypassRAG:
    """Minimal stand-in exposing only what LightRAG.aquery_llm's bypass branch uses."""

    def __init__(self, llm_func):
        self._llm = llm_func

    def _build_global_config(self):
        return {"role_llm_funcs": {"query": self._llm}}


def _completion(content, reasoning_content, finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content=content, parsed=None, reasoning_content=reasoning_content
                ),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )


def _openai_query_llm(*completions):
    responses = list(completions)

    async def _create(*_args, **_kwargs):
        return responses.pop(0) if len(responses) > 1 else responses[0]

    create = AsyncMock(side_effect=_create)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        close=AsyncMock(),
    )

    async def _llm(prompt, system_prompt=None, history_messages=None, **kwargs):
        with patch(
            "lightrag.llm.openai.create_openai_async_client", return_value=client
        ):
            return await openai_complete_if_cache(
                "reasoning-model",
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                api_key="test-key",
                **kwargs,
            )

    _llm.create = create
    return _llm


async def _inline_reasoning_only(*_args, **_kwargs):
    return f"<think>{REASONING}</think>"


def _query_config(query_llm) -> dict:
    return {
        "tokenizer": Tokenizer("fake", _FakeTokenizerImpl()),
        "role_llm_funcs": {"keyword": query_llm, "query": query_llm},
        "addon_params": {"language": "en"},
        "min_rerank_score": 0.0,
        "max_total_tokens": 4096,
    }


async def _kg_query(llm, cache):
    param = QueryParam(
        mode="mix",
        enable_rerank=False,
        stream=False,
        hl_keywords=["reset"],
        ll_keywords=["device"],
    )
    return await kg_query(
        QUERY, None, None, None, None, param, _query_config(llm), hashing_kv=cache
    )


async def _naive_query(llm, cache):
    return await naive_query(
        QUERY,
        _ChunksVDB(),
        QueryParam(mode="naive", enable_rerank=False, stream=False),
        _query_config(llm),
        hashing_kv=cache,
    )


class TestQueryReasoningOnlyAnswer:
    @pytest.fixture(autouse=True)
    def _no_retry_wait(self, monkeypatch):
        monkeypatch.setattr(openai_complete_if_cache.retry, "wait", wait_none())

    @pytest.fixture
    def stub_query_context(self, monkeypatch):
        async def _fake_build_query_context(*_args, **_kwargs):
            return QueryContextResult(
                context="KG CONTEXT",
                raw_data={"metadata": {"processing_info": {"final_chunks_count": 1}}},
            )

        monkeypatch.setattr(
            "lightrag.operate._build_query_context", _fake_build_query_context
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("enable_cot", [True, False])
    async def test_binding_treats_reasoning_only_as_empty_and_retries(self, enable_cot):
        llm = _openai_query_llm(_completion("", REASONING))

        with pytest.raises(InvalidResponseError) as caught:
            await llm(QUERY, enable_cot=enable_cot)

        assert llm.create.await_count == 3
        assert "reasoning-only output" in str(caught.value)
        assert REASONING not in str(caught.value)

    @pytest.mark.asyncio
    async def test_binding_truncated_reasoning_only_fails_fast(self):
        llm = _openai_query_llm(_completion("", REASONING, finish_reason="length"))

        with pytest.raises(EmptyTruncatedResponseError):
            await llm(QUERY, enable_cot=True)

        assert llm.create.await_count == 1

    @pytest.mark.asyncio
    async def test_kg_query_retry_recovers_the_answer(self, stub_query_context):
        cache = _RecordingKV()
        llm = _openai_query_llm(
            _completion("", REASONING), _completion(ANSWER, REASONING)
        )

        result = await _kg_query(llm, cache)

        assert result.content == ANSWER
        assert llm.create.await_count == 2
        assert len(cache.writes) == 1
        assert REASONING not in str(cache.writes)

    @pytest.mark.asyncio
    async def test_provider_without_reasoning_content_is_unaffected(
        self, stub_query_context
    ):
        completion = _completion(ANSWER, None)
        delattr(completion.choices[0].message, "reasoning_content")
        cache = _RecordingKV()

        result = await _kg_query(_openai_query_llm(completion), cache)

        assert result.content == ANSWER
        assert len(cache.writes) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("make_llm", "expected_exc"),
        [
            pytest.param(
                lambda: _openai_query_llm(_completion("", REASONING)),
                InvalidResponseError,
                id="openai-reasoning-only",
            ),
            pytest.param(
                lambda: _inline_reasoning_only,
                ReasoningOnlyResponseError,
                id="inline-think-only",
            ),
            pytest.param(
                lambda: _openai_query_llm(
                    _completion("", REASONING, finish_reason="length")
                ),
                EmptyTruncatedResponseError,
                id="truncated-reasoning-only",
            ),
        ],
    )
    async def test_kg_query_rejects_and_never_caches_reasoning_only(
        self, stub_query_context, make_llm, expected_exc
    ):
        cache = _RecordingKV()

        with pytest.raises(expected_exc):
            await _kg_query(make_llm(), cache)

        assert cache.writes == []

    @pytest.mark.asyncio
    async def test_naive_query_rejects_reasoning_only_answer_and_does_not_cache_it(
        self,
    ):
        cache = _RecordingKV()

        with pytest.raises(ReasoningOnlyResponseError):
            await _naive_query(_inline_reasoning_only, cache)

        assert cache.writes == []

    @pytest.mark.asyncio
    async def test_answer_with_reasoning_trace_and_truncated_partial_controls(
        self, stub_query_context
    ):
        cache = _RecordingKV()
        response = f"<think>{REASONING}</think>{ANSWER}"

        async def _llm_with_trace(*_args, **_kwargs):
            return response

        result = await _kg_query(_llm_with_trace, cache)
        assert result.content == response
        assert len(cache.writes) == 1

        cache2 = _RecordingKV()

        async def _llm_partial(*_args, **_kwargs):
            return TruncatedResponse("Hold the reset button for")

        result2 = await _kg_query(_llm_partial, cache2)
        assert result2.content == "Hold the reset button for"
        assert cache2.writes == []

    @pytest.mark.asyncio
    async def test_bypass_rejects_reasoning_only_and_keeps_trace_plus_answer(self):
        with pytest.raises(ReasoningOnlyResponseError):
            await LightRAG.aquery_llm(
                _BypassRAG(_inline_reasoning_only),
                QUERY,
                param=QueryParam(mode="bypass", stream=False),
            )

        response = f"<think>{REASONING}</think>{ANSWER}"

        async def _llm(*_args, **_kwargs):
            return response

        result = await LightRAG.aquery_llm(
            _BypassRAG(_llm), QUERY, param=QueryParam(mode="bypass", stream=False)
        )
        assert result["llm_response"]["content"] == response

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "run_query", [_kg_query, _naive_query], ids=["kg", "naive"]
    )
    async def test_cached_reasoning_only_answer_is_regenerated(
        self, stub_query_context, run_query
    ):
        cache = _RecordingKV(cached_answer=f"<think>{REASONING}</think>")
        calls = []

        async def _llm(*_args, **_kwargs):
            calls.append(1)
            return ANSWER

        result = await run_query(_llm, cache)

        assert result.content == ANSWER
        assert len(calls) == 1
        assert len(cache.writes) == 1
        assert ANSWER in str(cache.writes)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "run_query", [_kg_query, _naive_query], ids=["kg", "naive"]
    )
    async def test_cached_answer_with_reasoning_trace_is_still_served(
        self, stub_query_context, run_query
    ):
        cached = f"<think>{REASONING}</think>{ANSWER}"
        cache = _RecordingKV(cached_answer=cached)
        calls = []

        async def _llm(*_args, **_kwargs):
            calls.append(1)
            return "should not be generated"

        result = await run_query(_llm, cache)

        assert result.content == cached
        assert calls == []
        assert cache.writes == []
