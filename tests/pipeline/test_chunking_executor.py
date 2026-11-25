"""Chunking executor isolation, bounds, extension contract, and snapshot replay.

Consolidates:
- off-event-loop execution across all built-in chunking strategies, pre-embedding
  hard split, sidecar backfill, and multimodal chunk assembly;
- single-worker pool & submit semaphore bounds (cancellation does not release
  permits early while a worker thread is still running);
- ``chunking_func`` extension contract (custom sync/async chunkers stay on the
  event loop while the built-in default runs on the chunking executor);
- poisoned persisted separator snapshots bounded before logging, persistence, or
  dispatch.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

import lightrag.chunker as chunker_pkg
import lightrag.chunker.recursive_character as rc_mod
import lightrag.chunker.semantic_vector as sv_mod
import lightrag.pipeline as pipeline_mod
import lightrag.sidecar as sidecar_mod
from lightrag import LightRAG
from lightrag import utils as lr_utils
from lightrag.base import DocStatus
from lightrag.constants import CHUNKING_SUBMIT_LIMIT, MAX_R_SEPARATORS
from lightrag.utils import Tokenizer

from ._fakes import (
    make_embedding_func,
    make_lightrag,
    make_tokenizer,
    mock_json_llm,
)

pytestmark = pytest.mark.offline


def _new_rag(
    tmp_path: Path,
    *,
    with_embedding: bool = True,
    tokenizer: Tokenizer | None = None,
    **kwargs,
) -> LightRAG:
    if not with_embedding:
        kwargs["embedding_func"] = None
    return make_lightrag(
        tmp_path,
        subdir=None,
        workspace_prefix="chunk-exec",
        llm_model_func=mock_json_llm,
        embedding_dim=32,
        max_token_size=4096,
        tokenizer=tokenizer,
        **kwargs,
    )


async def _ingest(
    rag: LightRAG,
    *,
    doc_id: str,
    process_options: str = "",
    body: str | None = None,
) -> None:
    await rag.apipeline_enqueue_documents(
        body if body is not None else f"body text for {doc_id}",
        ids=[doc_id],
        file_paths=f"{doc_id}.txt",
        track_id=f"track-{doc_id}",
        process_options=process_options,
    )
    await rag.apipeline_process_enqueue_documents()


# ---------------------------------------------------------------------------
# Off-event-loop chunking execution
# ---------------------------------------------------------------------------


class _Heartbeat:
    """Counts loop iterations, and lets a worker thread sample the count."""

    def __init__(self):
        self.beats = 0
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self.observed: tuple[int, int] | None = None

    def start(self) -> None:
        async def _run():
            while not self._stop.is_set():
                self.beats += 1
                await asyncio.sleep(0)

        self._task = asyncio.create_task(_run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task

    def sample_across(self, seconds: float = 0.05) -> None:
        before = self.beats
        threading.Event().wait(seconds)
        self.observed = (before, self.beats)

    def assert_loop_stayed_alive(self) -> None:
        assert self.observed is not None, "the instrumented chunker never ran"
        before, after = self.observed
        assert after > before, (
            "the event loop did not advance while the chunker was working "
            f"(beats {before} -> {after})"
        )


def _run_with_heartbeat(coro_factory):
    heartbeat = _Heartbeat()

    async def _main():
        heartbeat.start()
        try:
            await coro_factory(heartbeat)
        finally:
            await heartbeat.stop()

    asyncio.run(_main())
    return heartbeat


def _with_blocks_path(monkeypatch, blocks_path: Path):
    """Inject a ``blocks_path`` at the process-stage boundary for plain-text tests."""
    original = pipeline_mod._PipelineMixin.process_single_document

    async def _patched(self, *, doc_id, status_doc, parsed_data, ctx):
        parsed_data["blocks_path"] = str(blocks_path)
        return await original(
            self, doc_id=doc_id, status_doc=status_doc, parsed_data=parsed_data, ctx=ctx
        )

    monkeypatch.setattr(
        pipeline_mod._PipelineMixin, "process_single_document", _patched
    )


@pytest.mark.parametrize(
    "option,chunker_name",
    [
        ("P", "chunking_by_paragraph_semantic"),
        ("R", "chunking_by_recursive_character"),
        ("F", "chunking_by_fixed_token"),
    ],
)
def test_explicit_strategies_do_not_hold_the_loop(
    tmp_path, monkeypatch, option, chunker_name
):
    def _make_stub(heartbeat):
        def _stub(tokenizer, content, chunk_token_size, **kwargs):
            heartbeat.sample_across()
            return [{"tokens": 5, "content": "stub", "chunk_order_index": 0}]

        return _stub

    async def _body(heartbeat):
        monkeypatch.setattr(chunker_pkg, chunker_name, _make_stub(heartbeat))
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            await _ingest(
                rag, doc_id=f"doc-{option}", process_options=option, body="body text"
            )
        finally:
            await rag.finalize_storages()

    _run_with_heartbeat(_body).assert_loop_stayed_alive()


def test_the_builtin_legacy_chunker_does_not_hold_the_loop(tmp_path, monkeypatch):
    """No explicit selector: the ``self.chunking_func`` path."""

    async def _body(heartbeat):
        rag = _new_rag(tmp_path)
        original = rag.chunking_func

        def _stub(*args, **kwargs):
            heartbeat.sample_across()
            return original(*args, **kwargs)

        monkeypatch.setattr(rag, "chunking_func", _stub)
        monkeypatch.setattr(chunker_pkg, "chunking_by_token_size", _stub)

        await rag.initialize_storages()
        try:
            await _ingest(
                rag, doc_id="doc-legacy", process_options="", body="body text"
            )
        finally:
            await rag.finalize_storages()

    _run_with_heartbeat(_body).assert_loop_stayed_alive()


def test_v_without_an_embedding_function_does_not_hold_the_loop(monkeypatch):
    """Without an embedding function, V falls back to R and must still offload."""

    async def _body(heartbeat):
        original = rc_mod.chunking_by_recursive_character

        def _stub(*args, **kwargs):
            heartbeat.sample_across()
            return original(*args, **kwargs)

        monkeypatch.setattr(rc_mod, "chunking_by_recursive_character", _stub)
        chunks = await sv_mod.chunking_by_semantic_vector(
            make_tokenizer(),
            "body text " * 200,
            120,
            embedding_func=None,
        )
        assert chunks

    _run_with_heartbeat(_body).assert_loop_stayed_alive()


def test_v_post_processing_does_not_hold_the_loop(monkeypatch):
    """The post-grouping recursive re-split pass in V must stay off the event loop."""

    async def _body(heartbeat):
        monkeypatch.setattr(
            sv_mod,
            "_semantic_groups_with_spans",
            lambda splitter, content: [(content, 0, len(content))],
        )
        monkeypatch.setattr(sv_mod, "SemanticChunker", lambda **kwargs: object())
        monkeypatch.setattr(sv_mod, "_LANGCHAIN_EXPERIMENTAL_AVAILABLE", True)

        original = rc_mod.chunking_by_recursive_character

        def _stub(*args, **kwargs):
            heartbeat.sample_across()
            return original(*args, **kwargs)

        monkeypatch.setattr(rc_mod, "chunking_by_recursive_character", _stub)
        chunks = await sv_mod.chunking_by_semantic_vector(
            make_tokenizer(),
            "body text " * 400,
            50,
            embedding_func=make_embedding_func(dim=32, max_token_size=4096),
        )
        assert chunks

    _run_with_heartbeat(_body).assert_loop_stayed_alive()


def test_the_pre_embedding_hard_split_does_not_hold_the_loop(tmp_path, monkeypatch):
    async def _body(heartbeat):
        original = pipeline_mod.enforce_chunk_token_limit_before_embedding

        def _stub(*args, **kwargs):
            heartbeat.sample_across()
            return original(*args, **kwargs)

        monkeypatch.setattr(
            pipeline_mod, "enforce_chunk_token_limit_before_embedding", _stub
        )
        rag = _new_rag(tmp_path)
        rag.embedding_token_limit = 32
        await rag.initialize_storages()
        try:
            await _ingest(
                rag, doc_id="doc-hardsplit", process_options="F", body="body text " * 80
            )
        finally:
            await rag.finalize_storages()

    _run_with_heartbeat(_body).assert_loop_stayed_alive()


async def test_a_chunker_failure_still_fails_the_document(tmp_path, monkeypatch):
    """Exceptions must survive the executor hop with their handling intact."""

    def _boom(*args, **kwargs):
        raise RuntimeError("chunker exploded")

    monkeypatch.setattr(chunker_pkg, "chunking_by_fixed_token", _boom)

    rag = _new_rag(tmp_path)
    await rag.initialize_storages()
    try:
        await _ingest(rag, doc_id="doc-boom", process_options="F", body="body text")
        row = await rag.doc_status.get_by_id("doc-boom")
        assert row is not None
        assert DocStatus(row["status"]) is DocStatus.FAILED
    finally:
        await rag.finalize_storages()


def test_the_sidecar_backfill_does_not_hold_the_loop(tmp_path, monkeypatch):
    async def _body(heartbeat):
        def _stub(chunking_result, blocks_path):
            heartbeat.sample_across()

        monkeypatch.setattr(sidecar_mod, "backfill_chunk_sidecars", _stub)
        _with_blocks_path(monkeypatch, tmp_path / "doc-backfill.blocks.jsonl")

        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            await _ingest(
                rag, doc_id="doc-backfill", process_options="F", body="body text " * 200
            )
        finally:
            await rag.finalize_storages()

    _run_with_heartbeat(_body).assert_loop_stayed_alive()


def test_the_multimodal_chunk_builder_does_not_hold_the_loop(tmp_path, monkeypatch):
    async def _body(heartbeat):
        def _stub(self, **kwargs):
            heartbeat.sample_across()
            return []

        monkeypatch.setattr(
            pipeline_mod._PipelineMixin, "_build_mm_chunks_from_sidecars", _stub
        )
        _with_blocks_path(monkeypatch, tmp_path / "doc-mmchunks.blocks.jsonl")

        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            await _ingest(
                rag, doc_id="doc-mmchunks", process_options="F", body="body text " * 200
            )
        finally:
            await rag.finalize_storages()

    _run_with_heartbeat(_body).assert_loop_stayed_alive()


# ---------------------------------------------------------------------------
# Chunking executor pool & submission permit bounds
# ---------------------------------------------------------------------------


class _CountingProxy:
    """Wraps the real executor and records accepted submissions."""

    def __init__(self, real):
        self._real = real
        self.submitted = 0

    def submit(self, fn, /, *args, **kwargs):
        self.submitted += 1
        return self._real.submit(fn, *args, **kwargs)


@pytest.fixture
def counting_executor(monkeypatch):
    proxy = _CountingProxy(lr_utils.get_chunking_executor())
    monkeypatch.setattr(lr_utils, "get_chunking_executor", lambda: proxy)
    return proxy


def _blocking(release: threading.Event, marker: str) -> str:
    release.wait(5.0)
    return marker


def test_the_pool_has_exactly_one_worker():
    """Preserves the concurrency the event loop already imposed on chunking."""
    assert lr_utils.get_chunking_executor()._max_workers == 1


async def test_cancelling_a_submission_does_not_free_its_slot_early(counting_executor):
    """The thread keeps running on cancel, so the permit stays held until completion."""
    release = threading.Event()
    try:
        tasks = [
            asyncio.create_task(
                lr_utils.run_in_chunking_executor(_blocking, release, f"m{i}")
            )
            for i in range(CHUNKING_SUBMIT_LIMIT)
        ]
        await asyncio.sleep(0.05)
        assert counting_executor.submitted == CHUNKING_SUBMIT_LIMIT

        for task in tasks:
            task.cancel()
        for task in tasks:
            with pytest.raises(asyncio.CancelledError):
                await task

        follow_up = asyncio.create_task(
            lr_utils.run_in_chunking_executor(lambda: "later")
        )
        await asyncio.sleep(0.1)

        assert counting_executor.submitted == CHUNKING_SUBMIT_LIMIT
        assert not follow_up.done()

        release.set()
        assert await asyncio.wait_for(follow_up, timeout=5.0) == "later"
        assert counting_executor.submitted == CHUNKING_SUBMIT_LIMIT + 1
    finally:
        release.set()


async def test_saturation_waits_rather_than_refusing(counting_executor):
    release = threading.Event()
    try:
        tasks = [
            asyncio.create_task(
                lr_utils.run_in_chunking_executor(_blocking, release, f"m{i}")
            )
            for i in range(CHUNKING_SUBMIT_LIMIT + 3)
        ]
        await asyncio.sleep(0.05)
        assert counting_executor.submitted == CHUNKING_SUBMIT_LIMIT

        release.set()
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=10.0)
        assert len(results) == CHUNKING_SUBMIT_LIMIT + 3
    finally:
        release.set()


async def test_an_exception_propagates_to_the_caller(counting_executor):
    """Chunker failures must keep reaching the pipeline's error handling."""

    def _boom():
        raise RuntimeError("chunker exploded")

    with pytest.raises(RuntimeError, match="chunker exploded"):
        await lr_utils.run_in_chunking_executor(_boom)


# ---------------------------------------------------------------------------
# ``chunking_func`` extension contract
# ---------------------------------------------------------------------------


def _stub_chunks():
    return [{"tokens": 5, "content": "stub", "chunk_order_index": 0}]


async def _run_custom_chunker(tmp_path: Path, chunking_func, doc_id: str) -> None:
    rag = _new_rag(tmp_path, chunking_func=chunking_func)
    await rag.initialize_storages()
    try:
        await _ingest(rag, doc_id=doc_id)
    finally:
        await rag.finalize_storages()


async def test_an_async_custom_chunker_is_still_awaited(tmp_path):
    seen = {}

    async def _custom(tokenizer, content, *args, **kwargs):
        seen["called"] = True
        return _stub_chunks()

    await _run_custom_chunker(tmp_path, _custom, "doc-async")
    assert seen.get("called") is True


async def test_a_custom_chunker_that_touches_the_running_loop_still_works(tmp_path):
    """A synchronous factory calling ``get_running_loop()`` / ``create_task()`` stays on loop."""
    seen = {}

    def _custom(tokenizer, content, *args, **kwargs):
        loop = asyncio.get_running_loop()
        seen["loop"] = loop is not None

        async def _produce():
            return _stub_chunks()

        return loop.create_task(_produce())

    await _run_custom_chunker(tmp_path, _custom, "doc-loop-aware")
    assert seen.get("loop") is True


async def test_a_custom_chunker_runs_on_the_event_loop_and_the_builtin_does_not(
    tmp_path, monkeypatch
):
    """Pins the dispatch split in both directions: custom on loop, built-in off loop."""
    observations: dict = {}

    def _has_running_loop() -> bool:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return False
        return True

    def _custom(tokenizer, content, *args, **kwargs):
        observations["custom_on_loop"] = _has_running_loop()
        return _stub_chunks()

    await _run_custom_chunker(tmp_path, _custom, "doc-custom-side")

    original_builtin = chunker_pkg.chunking_by_token_size

    def _builtin_spy(*args, **kwargs):
        observations["builtin_on_loop"] = _has_running_loop()
        observations["builtin_thread"] = threading.current_thread().name
        return original_builtin(*args, **kwargs)

    rag = _new_rag(tmp_path)
    monkeypatch.setattr(chunker_pkg, "chunking_by_token_size", _builtin_spy)
    rag.chunking_func = _builtin_spy
    await rag.initialize_storages()
    try:
        await _ingest(rag, doc_id="doc-builtin-side")
    finally:
        await rag.finalize_storages()

    assert observations["custom_on_loop"] is True
    assert observations["builtin_on_loop"] is False
    assert observations["builtin_thread"].startswith("lightrag-chunking")


# ---------------------------------------------------------------------------
# Persisted poisoned separator snapshot replay bounds
# ---------------------------------------------------------------------------

POISONED_SEPARATORS = ["Q"] * 900


class _CountingTokenizerImpl:
    def __init__(self):
        self.encodes = 0

    def encode(self, content: str):
        self.encodes += 1
        return [ord(c) for c in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


async def test_a_poisoned_snapshot_is_bounded_at_process_time(tmp_path, monkeypatch):
    """A stored 900-separator snapshot reaches the R chunker already bounded."""
    captured: dict = {}
    original_chunker = chunker_pkg.chunking_by_recursive_character
    original_normalize = rc_mod.normalize_r_separators

    def _chunker_spy(tokenizer, content, chunk_token_size, **kwargs):
        captured["dispatched"] = kwargs.get("separators")
        return original_chunker(tokenizer, content, chunk_token_size, **kwargs)

    def _normalize_spy(separators, **kwargs):
        result = original_normalize(separators, **kwargs)
        captured["normalized"] = result
        return result

    monkeypatch.setattr(chunker_pkg, "chunking_by_recursive_character", _chunker_spy)
    monkeypatch.setattr(rc_mod, "normalize_r_separators", _normalize_spy)

    rag = _new_rag(
        tmp_path, tokenizer=Tokenizer("mock-tokenizer", _CountingTokenizerImpl())
    )
    await rag.initialize_storages()
    try:
        await rag.apipeline_enqueue_documents(
            "Q" + ("X " * 2000),
            ids=["doc-poisoned-snapshot"],
            file_paths="poisoned.txt",
            track_id="track-poisoned",
            process_options="R",
            chunk_options={"recursive_character": {"separators": POISONED_SEPARATORS}},
        )
        row = await rag.full_docs.get_by_id("doc-poisoned-snapshot")
        stored = row["chunk_options"]["recursive_character"]["separators"]
        assert len(stored) == len(POISONED_SEPARATORS)

        await rag.apipeline_process_enqueue_documents()
    finally:
        await rag.finalize_storages()

    assert captured.get("dispatched") is not None, "the R chunker never ran"
    assert len(captured["dispatched"]) <= MAX_R_SEPARATORS
    assert captured.get("normalized") is not None
    assert len(captured["normalized"]) <= MAX_R_SEPARATORS


async def test_replaying_a_poisoned_snapshot_costs_a_bounded_number_of_encodes(
    tmp_path,
):
    tokenizer_impl = _CountingTokenizerImpl()
    rag = _new_rag(tmp_path, tokenizer=Tokenizer("mock-tokenizer", tokenizer_impl))
    await rag.initialize_storages()
    try:
        await rag.apipeline_enqueue_documents(
            "Q" + ("X " * 2000),
            ids=["doc-poisoned-cost"],
            file_paths="poisoned-cost.txt",
            track_id="track-poisoned-cost",
            process_options="R",
            chunk_options={"recursive_character": {"separators": POISONED_SEPARATORS}},
        )
        before = tokenizer_impl.encodes
        await rag.apipeline_process_enqueue_documents()
        encodes = tokenizer_impl.encodes - before
    finally:
        await rag.finalize_storages()

    assert encodes < len(POISONED_SEPARATORS) / 4


async def test_an_oversized_snapshot_never_reaches_the_log_or_doc_status(tmp_path):
    """Bounding happens before ``_format_chunking_params`` renders ``chunk_opts``."""
    huge_separator = "Z" * (5 * 1024 * 1024)
    rag = _new_rag(
        tmp_path, tokenizer=Tokenizer("mock-tokenizer", _CountingTokenizerImpl())
    )
    await rag.initialize_storages()
    try:
        await rag.apipeline_enqueue_documents(
            "Q" + ("X " * 2000),
            ids=["doc-huge-separator"],
            file_paths="huge-separator.txt",
            track_id="track-huge-separator",
            process_options="R",
            chunk_options={"recursive_character": {"separators": [huge_separator]}},
        )
        await rag.apipeline_process_enqueue_documents()
        row = await rag.doc_status.get_by_id("doc-huge-separator")
    finally:
        await rag.finalize_storages()

    rendered = (row.get("metadata") or {}).get("chunk_opts", "")
    assert len(rendered) < 4096, f"chunk_opts rendered {len(rendered)} characters"
    assert huge_separator not in rendered
