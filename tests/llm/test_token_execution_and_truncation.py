"""Async tokenizer executor (GHSA-r8jh-295g-vv42) and two-stage chunk truncation tests.

Covers:
1. Offloading token counting / list truncation to the single-worker tokenizer
   executor keeps the event loop responsive, matches synchronous results, and
   uses a single submission per call (including ``process_chunks_unified``'s
   two-stage truncation).
2. ``process_chunks_unified`` verifies truncation against the real rendered
   ``generate_reference_list_from_chunks`` / ``render_chunks_context_text`` output,
   preserves ``content_headings``, and logs budget warnings only when the token
   budget itself drops every retrieved chunk.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading

import pytest

from lightrag import utils as lr_utils
from lightrag.base import QueryParam
from lightrag.utils import (
    Tokenizer,
    TokenizerInterface,
    generate_reference_list_from_chunks,
    process_chunks_unified,
    render_chunks_context_text,
)

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


class _SlowCharTokenizer(TokenizerInterface):
    """1-token-per-char tokenizer with optional blocking delay to simulate CPU work."""

    def __init__(self, delay: float = 0.0):
        self._delay = delay
        self.calls = 0

    def encode(self, content: str) -> list[int]:
        self.calls += 1
        if self._delay:
            threading.Event().wait(self._delay)
        return list(range(len(content)))

    def decode(self, tokens: list[int]) -> str:
        return "x" * len(tokens)


def _tok(delay: float = 0.0) -> Tokenizer:
    return Tokenizer("char", _SlowCharTokenizer(delay=delay))


async def _heartbeat(stop: asyncio.Event) -> int:
    beats = 0
    while not stop.is_set():
        beats += 1
        await asyncio.sleep(0)
    return beats


async def _count_beats_during(coro):
    stop = asyncio.Event()
    pulse = asyncio.create_task(_heartbeat(stop))
    try:
        result = await coro
    finally:
        stop.set()
    return result, await pulse


async def _run_unified(chunks, chunk_token_limit, tokenizer):
    global_config = {"tokenizer": tokenizer}
    query_param = QueryParam(enable_rerank=False, chunk_top_k=None)
    return await process_chunks_unified(
        query="",
        unique_chunks=chunks,
        query_param=query_param,
        global_config=global_config,
        chunk_token_limit=chunk_token_limit,
    )


def _budget_warnings(records):
    return [
        record.getMessage()
        for record in records
        if record.levelno == logging.WARNING
        and "Token budget left no room" in record.getMessage()
    ]


# ---------------------------------------------------------------------------
# 1. Async tokenizer executor behavior (GHSA-r8jh-295g-vv42)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("operation", ["acount_tokens", "atruncate_list"])
async def test_async_token_helpers_do_not_block_the_event_loop(operation):
    if operation == "acount_tokens":
        tokenizer = _tok(delay=0.05)
        result, beats = await _count_beats_during(
            lr_utils.acount_tokens(tokenizer, "abcd")
        )
        assert result == 4
    else:
        tokenizer = _tok(delay=0.01)
        data = [{"text": "abcde"} for _ in range(10)]
        _, beats = await _count_beats_during(
            lr_utils.atruncate_list_by_token_size(
                data,
                key=lambda x: x["text"],
                separator="",
                max_token_size=12,
                tokenizer=tokenizer,
            )
        )
    assert beats > 1


async def test_acount_and_aencode_match_sync_results():
    tokenizer = _tok()
    assert await lr_utils.acount_tokens(tokenizer, "hello") == len(
        tokenizer.encode("hello")
    )
    assert await lr_utils.aencode(tokenizer, "hello") == tokenizer.encode("hello")


@pytest.mark.parametrize("budget", [0, 1, 5, 11, 12, 1000])
async def test_atruncate_matches_the_sync_result(budget):
    tokenizer = _tok()
    data = [{"text": "abcde"} for _ in range(6)]

    expected = lr_utils.truncate_list_by_token_size(
        data,
        key=lambda x: x["text"],
        separator="",
        max_token_size=budget,
        tokenizer=tokenizer,
    )
    actual = await lr_utils.atruncate_list_by_token_size(
        data,
        key=lambda x: x["text"],
        separator="",
        max_token_size=budget,
        tokenizer=tokenizer,
    )
    assert actual == expected


async def test_executor_is_single_worker_singleton():
    executor = lr_utils.get_tokenizer_executor()
    assert executor._max_workers == 1
    assert lr_utils.get_tokenizer_executor() is executor


@pytest.mark.parametrize("kind", ["atruncate_list", "process_chunks_unified"])
async def test_truncation_uses_a_single_executor_submission(monkeypatch, kind):
    """Both list truncation and two-stage unified truncation stay O(1) submissions."""
    submissions = 0
    real_executor = lr_utils.get_tokenizer_executor()

    class _CountingProxy:
        """Forwards submissions while counting how many tasks hit the executor."""

        def submit(self, fn, /, *args, **kwargs):
            nonlocal submissions
            submissions += 1
            return real_executor.submit(fn, *args, **kwargs)

    monkeypatch.setattr(lr_utils, "get_tokenizer_executor", lambda: _CountingProxy())

    if kind == "atruncate_list":
        data = [{"text": "abcde"} for _ in range(1000)]
        await lr_utils.atruncate_list_by_token_size(
            data,
            key=lambda x: x["text"],
            separator="",
            max_token_size=10_000,
            tokenizer=_tok(),
        )
    else:
        chunks = [
            {"content": "X" * 20, "file_path": f"f{i}.txt", "chunk_id": f"c{i}"}
            for i in range(30)
        ]
        await _run_unified(chunks, chunk_token_limit=100, tokenizer=_tok())

    assert submissions == 1


# ---------------------------------------------------------------------------
# 2. process_chunks_unified two-stage truncation and budget warning logs
# ---------------------------------------------------------------------------


async def test_truncation_result_matches_the_real_renderer_exactly():
    """Frequency-shift case: cutting the trailing chunk changes ``common.txt``'s
    frequency (2 -> 1) and reference_id, yet the final rendered text still fits.
    """
    chunks = [
        {"content": "A" * 10, "file_path": "common.txt", "chunk_id": "c1"},
        {"content": "B" * 10, "file_path": "rare.txt", "chunk_id": "c2"},
        {"content": "C" * 200, "file_path": "common.txt", "chunk_id": "c3"},
    ]
    tokenizer = _tok()

    result = await _run_unified(chunks, chunk_token_limit=150, tokenizer=tokenizer)

    kept_ids = {c["chunk_id"] for c in result}
    assert "c3" not in kept_ids
    assert {"c1", "c2"} <= kept_ids

    _, rendered = generate_reference_list_from_chunks(
        [{k: v for k, v in c.items() if k != "id"} for c in result]
    )
    text = render_chunks_context_text(rendered)
    assert len(tokenizer.encode(text)) <= 150


async def test_empty_input_fit_all_and_content_headings_preserved():
    assert await _run_unified([], chunk_token_limit=100, tokenizer=_tok()) == []

    chunks = [
        {
            "content": "hello",
            "content_headings": "Intro > Section 1",
            "file_path": "a.txt",
            "chunk_id": "c1",
        },
        {"content": "world", "file_path": "b.txt", "chunk_id": "c2"},
    ]
    result = await _run_unified(chunks, chunk_token_limit=1000, tokenizer=_tok())
    assert len(result) == 2

    _, rendered = generate_reference_list_from_chunks(result)
    text = render_chunks_context_text(rendered)
    parsed = [json.loads(line) for line in text.splitlines()]
    assert parsed[0]["content_headings"] == "Intro > Section 1"


@pytest.mark.parametrize("source_type", ["vector", "mix"])
async def test_budget_that_drops_every_chunk_is_logged(
    lightrag_log_records, source_type
):
    chunks = [{"content": "x" * 200, "file_path": "a.md"}]
    result = await process_chunks_unified(
        query="",
        unique_chunks=chunks,
        query_param=QueryParam(enable_rerank=False, chunk_top_k=None),
        global_config={"tokenizer": _tok()},
        source_type=source_type,
        chunk_token_limit=50,
    )
    assert result == []
    warnings = _budget_warnings(lightrag_log_records)
    assert len(warnings) == 1
    assert "any of 1 retrieved chunks" in warnings[0]
    assert ("max_entity_tokens" in warnings[0]) == (source_type != "vector")


@pytest.mark.parametrize("limit", [-5, 0])
async def test_non_positive_budget_is_logged(lightrag_log_records, limit):
    assert await _run_unified([{"content": "x", "file_path": "a.md"}], limit, _tok()) == []
    assert len(_budget_warnings(lightrag_log_records)) == 1


async def test_no_budget_warning_for_partial_truncation_or_rerank_filtering(
    lightrag_log_records,
):
    partial_chunks = [
        {"content": "a" * 20, "file_path": "a.md"},
        {"content": "b" * 400, "file_path": "b.md"},
    ]
    assert len(await _run_unified(partial_chunks, 120, _tok())) == 1
    assert _budget_warnings(lightrag_log_records) == []

    async def _rerank(**_kwargs):
        return [{"index": 0, "relevance_score": 0.1}]

    filtered = await process_chunks_unified(
        query="q",
        unique_chunks=[{"content": "x", "file_path": "a.md", "rerank_score": 0.1}],
        query_param=QueryParam(enable_rerank=True, chunk_top_k=None),
        global_config={
            "tokenizer": _tok(),
            "rerank_model_func": _rerank,
            "min_rerank_score": 0.5,
        },
        chunk_token_limit=1000,
    )
    assert filtered == []
    assert _budget_warnings(lightrag_log_records) == []
