"""Consolidated pipeline ingress exit/handoff and in-batch feeder contract tests.

Covers (PipelineConcurrencyContract — Phases 2 & 3):
  * Quiescence decision priority (manual > auto > document > release),
    signal-consumption semantics, stale-message compaction, refetch failure
    compensation, atomic publish-under-status-lock, cancel/error signal
    restoration, journaled custom-chunk non-livelock, and direct-delete /
    custom-chunks exit handoffs.
  * In-batch feeder: mid-batch feeding while an earlier doc blocks, sequential
    multi-file upload wakeups, inflight deduplication, non-busy-looping on
    pending manual signals, cancellation / manual-request / paging-disabled
    yielding, teardown message re-publication, SKIP auto-rescan arming, and
    per-admission cancellation responsiveness.
"""

from __future__ import annotations

import asyncio
import threading
from collections import Counter
from pathlib import Path

import pytest

import lightrag.kg.shared_storage as shared_storage_module
import lightrag.lightrag as lightrag_module
from lightrag import LightRAG
from lightrag.base import CURSOR_END, CURSOR_START
from lightrag.kg.pipeline_ingress import PipelineIngressMessage
from lightrag.kg.shared_storage import get_pipeline_ingress
from lightrag.pipeline import (
    _AUTO_RESUME_DOC_STATUSES,
    PipelineNextDecision,
    PipelineNextStep,
    _BatchRunContext,
    _ManualDrainProgress,
)
from lightrag.utils import compute_mdhash_id
from lightrag.utils_pipeline import CUSTOM_CHUNK_PATCH_METADATA_KEY

from ._fakes import (
    build_initialized_rag,
    pipeline_status_handles,
    single_chunk_chunking,
)

pytestmark = pytest.mark.offline


class _MarkerExtract:
    """Process-layer extraction that counts calls per document marker and can
    block whichever marker is armed until released."""

    def __init__(self):
        self.calls: Counter = Counter()
        self.block_marker: str | None = None
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, chunks, *args, **kwargs):
        content = " ".join(str(v.get("content", "")) for v in chunks.values())
        marker = content.split(" ", 1)[0] if content else ""
        self.calls[marker] += 1
        if self.block_marker and self.block_marker in content:
            self.started.set()
            await self.release.wait()
        return [({}, {}) for _ in chunks]


async def _build_rag(
    tmp_path: Path,
    extract: _MarkerExtract,
    *,
    max_parallel_insert: int = 1,
    workspace_prefix: str = "ingress",
) -> LightRAG:
    rag = await build_initialized_rag(
        tmp_path,
        workspace_prefix=workspace_prefix,
        chunking_func=single_chunk_chunking,
        max_parallel_insert=max_parallel_insert,
    )
    rag._process_extract_entities = extract  # type: ignore[assignment]
    return rag


def _status_text(row) -> str:
    status = row["status"]
    return getattr(status, "value", str(status)).replace("DocStatus.", "").lower()


def _doc_msg(doc_id: str) -> PipelineIngressMessage:
    return PipelineIngressMessage(kind="document", doc_id=doc_id)


def _manual_msg(request_id: str) -> PipelineIngressMessage:
    return PipelineIngressMessage(
        kind="rescan", retry_failed=True, request_id=request_id
    )


def _make_feeder_ctx() -> _BatchRunContext:
    """Minimal batch context for driving _pipeline_feeder directly."""
    return _BatchRunContext(
        pipeline_status={"docs": 0, "batchs": 0, "cur_batch": 0},
        pipeline_status_lock=asyncio.Lock(),
        semaphore=asyncio.Semaphore(1),
        total_files=0,
        parse_queues={"native": asyncio.Queue()},
        parser_specs={},
        q_analyze=asyncio.Queue(),
        q_process=asyncio.Queue(),
        pipeline_cancel_event=threading.Event(),
    )


async def _wait_status(rag: LightRAG, doc_id: str, target: str, timeout: float = 5.0):
    async def _poll():
        while True:
            row = await rag.doc_status.get_by_id(doc_id)
            if row and _status_text(row) == target:
                return
            await asyncio.sleep(0.02)

    await asyncio.wait_for(_poll(), timeout=timeout)


# ---------------------------------------------------------------------------
# Phase 3: has_work-only atomic exit & cancellation/recovery closure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decision_priority_and_consumption_semantics(tmp_path):
    """Priority is manual > auto > document > release, and each step consumes
    ONLY its own signal."""
    rag = await _build_rag(tmp_path, _MarkerExtract(), workspace_prefix="exit")
    try:
        status, lock = await pipeline_status_handles(rag)
        ingress = await get_pipeline_ingress(rag.workspace)

        ingress.request_manual_retry("req-1", _manual_msg("req-1"))
        ingress.request_auto_rescan()
        ingress.put_document(_doc_msg("doc-x"))
        async with lock:
            status.update({"busy": True, "busy_owner": None})

        d1 = await rag._decide_pipeline_next_step(status, lock, ingress)
        assert d1.step is PipelineNextStep.CONTINUE_MANUAL
        assert d1.manual_request_id == "req-1"
        counts = ingress.counts()
        assert counts["manual_retries"] == 1
        assert counts["auto_rescan_pending"] is True
        assert counts["documents"] == 1

        ingress.ack_manual_retry("req-1")
        d2 = await rag._decide_pipeline_next_step(status, lock, ingress)
        assert d2.step is PipelineNextStep.CONTINUE_AUTO
        counts = ingress.counts()
        assert counts["auto_rescan_pending"] is False
        assert counts["documents"] == 1

        d3 = await rag._decide_pipeline_next_step(status, lock, ingress)
        assert d3.step is PipelineNextStep.CONTINUE_DOCUMENT
        assert ingress.counts()["documents"] == 1

        ingress.drain_documents()
        d4 = await rag._decide_pipeline_next_step(status, lock, ingress)
        assert d4.step is PipelineNextStep.RELEASED
        assert status.get("busy") is False
        assert status.get("busy_owner") is None
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_document_published_at_quiescence_starts_next_batch(tmp_path):
    """Fix-proof: a document whose ONLY signal is its mailbox message published
    after the batch's feeder stopped keeps the run busy (CONTINUE_DOCUMENT)."""
    extract = _MarkerExtract()
    rag = await _build_rag(tmp_path, extract, workspace_prefix="exit")
    try:
        status, _lock = await pipeline_status_handles(rag)
        await rag.apipeline_enqueue_documents(input="AAAA body", file_paths="a.txt")
        b_id = compute_mdhash_id("b.txt", prefix="doc-")

        orig_batch = rag._run_pipeline_batch
        injected = False

        async def batch_then_inject(to_process_docs, **kwargs):
            nonlocal injected
            await orig_batch(to_process_docs, **kwargs)
            if not injected:
                injected = True
                await rag.apipeline_enqueue_documents(
                    input="BBBB body", file_paths="b.txt"
                )

        rag._run_pipeline_batch = batch_then_inject

        await asyncio.wait_for(rag.apipeline_process_enqueue_documents(), timeout=10)

        assert _status_text(await rag.doc_status.get_by_id(b_id)) == "processed"
        assert extract.calls["BBBB"] == 1
        assert status.get("busy") is False
        ingress = await get_pipeline_ingress(rag.workspace)
        assert ingress.counts()["documents"] == 0
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_stale_document_message_is_compacted_and_released(tmp_path):
    """Fix-proof: a notification for a ghost doc is compacted after strict scan
    and busy is released without livelock."""
    rag = await _build_rag(tmp_path, _MarkerExtract(), workspace_prefix="exit")
    try:
        status, _lock = await pipeline_status_handles(rag)
        ingress = await get_pipeline_ingress(rag.workspace)
        ingress.put_document(_doc_msg("doc-stale-ghost"))

        await asyncio.wait_for(rag.apipeline_process_enqueue_documents(), timeout=10)

        assert status.get("busy") is False
        assert ingress.counts()["documents"] == 0
        assert ingress.has_work() is False
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_refetch_document_failure_republishes_and_arms_auto(tmp_path):
    """Compensation: a strict-scan failure after CONTINUE_DOCUMENT drain restores
    the drained messages and arms auto-rescan before propagating."""
    rag = await _build_rag(tmp_path, _MarkerExtract(), workspace_prefix="exit")
    try:
        rag.pipeline_scheduling_page_size = 0
        ingress = await get_pipeline_ingress(rag.workspace)
        ingress.consume_auto_rescan()
        ingress.put_document(_doc_msg("doc-a"))
        ingress.put_document(_doc_msg("doc-b"))

        async def dead_scan(*args, **kwargs):
            raise ConnectionError("doc_status backend transient failure")

        rag.doc_status.get_docs_by_statuses = dead_scan

        status, lock = await pipeline_status_handles(rag)
        decision = PipelineNextDecision(PipelineNextStep.CONTINUE_DOCUMENT)
        with pytest.raises(ConnectionError):
            await rag._refetch_for_decision(
                decision,
                _AUTO_RESUME_DOC_STATUSES,
                CURSOR_START,
                ingress,
                token="tok",
                pipeline_status=status,
                pipeline_status_lock=lock,
                drain_progress=_ManualDrainProgress(),
            )

        republished = {m.doc_id for m in ingress.drain_documents()}
        assert republished == {"doc-a", "doc-b"}
        assert ingress.counts()["auto_rescan_pending"] is True
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_enqueue_publishes_documents_inside_status_lock(tmp_path, monkeypatch):
    """Fix-proof: document publish happens INSIDE the pipeline_status lock."""
    rag = await _build_rag(tmp_path, _MarkerExtract(), workspace_prefix="exit")
    try:
        held: dict = {}
        orig_aenter = shared_storage_module.NamespaceLock.__aenter__
        orig_aexit = shared_storage_module.NamespaceLock.__aexit__

        async def tracing_aenter(self):
            result = await orig_aenter(self)
            held.setdefault(asyncio.current_task(), set()).add(self._namespace)
            return result

        async def tracing_aexit(self, *args):
            held.get(asyncio.current_task(), set()).discard(self._namespace)
            return await orig_aexit(self, *args)

        monkeypatch.setattr(
            shared_storage_module.NamespaceLock, "__aenter__", tracing_aenter
        )
        monkeypatch.setattr(
            shared_storage_module.NamespaceLock, "__aexit__", tracing_aexit
        )

        ingress = await get_pipeline_ingress(rag.workspace)
        publish_lock_states: list[bool] = []
        published_ids: list[str] = []
        orig_put = ingress.put_documents

        def probing_put(msgs):
            namespaces = held.get(asyncio.current_task(), set())
            publish_lock_states.append("pipeline_status" in namespaces)
            published_ids.extend(m.doc_id for m in msgs)
            return orig_put(msgs)

        ingress.put_documents = probing_put

        await rag.apipeline_enqueue_documents(input="AAAA body", file_paths="a.txt")

        a_id = compute_mdhash_id("a.txt", prefix="doc-")
        assert published_ids == [a_id]
        assert publish_lock_states == [True]
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_refetch_compensation_failure_does_not_mask_original_error(tmp_path):
    """When even compensation RPCs fail, the ORIGINAL strict-scan failure
    still propagates."""
    rag = await _build_rag(tmp_path, _MarkerExtract(), workspace_prefix="exit")
    try:
        rag.pipeline_scheduling_page_size = 0
        ingress = await get_pipeline_ingress(rag.workspace)
        ingress.put_document(_doc_msg("doc-a"))

        async def dead_scan(*args, **kwargs):
            raise ConnectionError("original strict-scan failure")

        def dead_put(msg):
            raise RuntimeError("manager down: put_document")

        def dead_arm():
            raise RuntimeError("manager down: request_auto_rescan")

        rag.doc_status.get_docs_by_statuses = dead_scan
        ingress.put_document = dead_put
        ingress.request_auto_rescan = dead_arm

        status, lock = await pipeline_status_handles(rag)
        decision = PipelineNextDecision(PipelineNextStep.CONTINUE_DOCUMENT)
        with pytest.raises(ConnectionError, match="original strict-scan failure"):
            await rag._refetch_for_decision(
                decision,
                _AUTO_RESUME_DOC_STATUSES,
                CURSOR_START,
                ingress,
                token="tok",
                pipeline_status=status,
                pipeline_status_lock=lock,
                drain_progress=_ManualDrainProgress(),
            )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_user_cancel_preserves_ingress_and_stops_run(tmp_path):
    """Fix-proof: cancellation is checked BEFORE the quiescence decision so an
    auto-rescan flag armed during the batch survives to the next trigger."""
    extract = _MarkerExtract()
    extract.block_marker = "AAAA"
    rag = await _build_rag(tmp_path, extract, workspace_prefix="exit")
    try:
        status, lock = await pipeline_status_handles(rag)
        await rag.apipeline_enqueue_documents(input="AAAA body", file_paths="a.txt")

        proc = asyncio.create_task(rag.apipeline_process_enqueue_documents())
        await asyncio.wait_for(extract.started.wait(), timeout=5)

        async with lock:
            status["cancellation_requested"] = True
        ingress = await get_pipeline_ingress(rag.workspace)
        ingress.consume_auto_rescan()
        ingress.request_auto_rescan()

        extract.release.set()
        await asyncio.wait_for(proc, timeout=10)

        assert status.get("busy") is False
        assert status.get("cancellation_requested") is False
        assert ingress.counts()["auto_rescan_pending"] is True
    finally:
        extract.release.set()
        await rag.finalize_storages()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "target_step"),
    [
        ("auto", PipelineNextStep.CONTINUE_AUTO),
        ("document", PipelineNextStep.CONTINUE_DOCUMENT),
    ],
    ids=["auto", "document"],
)
async def test_cancel_after_decision_restores_consumed_signal(
    tmp_path, mode, target_step
):
    """Fix-proof (consume-then-discard closure): a cancel landing AFTER the
    decision consumed CONTINUE_AUTO or CONTINUE_DOCUMENT but BEFORE its
    refetched docs enter a batch restores the auto-rescan signal."""
    extract = _MarkerExtract()
    extract.block_marker = "AAAA"
    rag = await _build_rag(tmp_path, extract, workspace_prefix="exit")
    try:
        status, _lock = await pipeline_status_handles(rag)
        await rag.apipeline_enqueue_documents(input="AAAA body", file_paths="a.txt")
        b_id = compute_mdhash_id("b.txt", prefix="doc-")
        ingress = await get_pipeline_ingress(rag.workspace)

        orig_batch = rag._run_pipeline_batch
        injected = False

        async def batch_then_inject(to_process_docs, **kwargs):
            nonlocal injected
            await orig_batch(to_process_docs, **kwargs)
            if not injected:
                injected = True
                await rag.apipeline_enqueue_documents(
                    input="BBBB body", file_paths="b.txt"
                )
                if mode == "auto":
                    ingress.drain_documents()
                    ingress.request_auto_rescan()

        rag._run_pipeline_batch = batch_then_inject

        orig_decide = rag._decide_pipeline_next_step
        cancelled_after = []

        async def decide_then_cancel(ps, ps_lock, ing, sweep_cursor=CURSOR_END):
            decision = await orig_decide(ps, ps_lock, ing, sweep_cursor)
            if decision.step is target_step and not cancelled_after:
                cancelled_after.append(True)
                async with ps_lock:
                    ps["cancellation_requested"] = True
            return decision

        rag._decide_pipeline_next_step = decide_then_cancel

        extract.release.set()
        await asyncio.wait_for(rag.apipeline_process_enqueue_documents(), timeout=10)

        assert cancelled_after
        assert status.get("busy") is False
        assert _status_text(await rag.doc_status.get_by_id(b_id)) == "pending"
        assert ingress.counts()["auto_rescan_pending"] is True
    finally:
        extract.release.set()
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_initial_scan_cancellation_restores_absorbed_auto_signal(tmp_path):
    """Fix-proof (entry-window closure): a CancelledError inside the initial
    strict scan re-arms the absorbed auto-rescan flag in the finally block."""
    rag = await _build_rag(tmp_path, _MarkerExtract(), workspace_prefix="exit")
    try:
        rag.pipeline_scheduling_page_size = 0
        status, _lock = await pipeline_status_handles(rag)
        ingress = await get_pipeline_ingress(rag.workspace)
        ingress.request_auto_rescan()

        async def cancelled_scan(*args, **kwargs):
            raise asyncio.CancelledError()

        rag.doc_status.get_docs_by_statuses = cancelled_scan

        with pytest.raises(asyncio.CancelledError):
            await rag.apipeline_process_enqueue_documents()

        assert status.get("busy") is False
        assert ingress.counts()["auto_rescan_pending"] is True
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["document", "auto"])
async def test_validation_failure_after_decision_restores_signal(tmp_path, mode):
    """Fix-proof (non-cancel escape closure): a transient validation failure
    after CONTINUE_DOCUMENT or CONTINUE_AUTO re-arms auto-rescan on exit."""
    extract = _MarkerExtract()
    extract.block_marker = "AAAA"
    rag = await _build_rag(tmp_path, extract, workspace_prefix="exit")
    try:
        status, _lock = await pipeline_status_handles(rag)
        await rag.apipeline_enqueue_documents(input="AAAA body", file_paths="a.txt")
        b_id = compute_mdhash_id("b.txt", prefix="doc-")
        ingress = await get_pipeline_ingress(rag.workspace)

        orig_batch = rag._run_pipeline_batch
        orig_validate = rag._validate_and_fix_document_consistency
        injected = False

        async def failing_validate(*args, **kwargs):
            raise ConnectionError("full_docs transient failure in validation")

        async def batch_then_inject(to_process_docs, **kwargs):
            nonlocal injected
            await orig_batch(to_process_docs, **kwargs)
            if not injected:
                injected = True
                await rag.apipeline_enqueue_documents(
                    input="BBBB body", file_paths="b.txt"
                )
                if mode == "auto":
                    ingress.drain_documents()
                    ingress.request_auto_rescan()
                rag._validate_and_fix_document_consistency = failing_validate

        rag._run_pipeline_batch = batch_then_inject

        extract.release.set()
        with pytest.raises(ConnectionError):
            await asyncio.wait_for(
                rag.apipeline_process_enqueue_documents(), timeout=10
            )

        rag._validate_and_fix_document_consistency = orig_validate
        assert status.get("busy") is False
        assert _status_text(await rag.doc_status.get_by_id(b_id)) == "pending"
        assert ingress.counts()["auto_rescan_pending"] is True
    finally:
        extract.release.set()
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_internal_error_halt_preserves_ingress_and_surfaces_reason(tmp_path):
    """An internal-error abort retains ingress and surfaces the halt reason."""
    extract = _MarkerExtract()
    extract.block_marker = "AAAA"
    rag = await _build_rag(tmp_path, extract, workspace_prefix="exit")
    try:
        status, lock = await pipeline_status_handles(rag)
        await rag.apipeline_enqueue_documents(input="AAAA body", file_paths="a.txt")

        proc = asyncio.create_task(rag.apipeline_process_enqueue_documents())
        await asyncio.wait_for(extract.started.wait(), timeout=5)

        async with lock:
            status.update(
                {
                    "cancellation_requested": True,
                    "cancellation_reason": "internal_error",
                    "cancellation_detail": "storage boom sentinel",
                }
            )
        ingress = await get_pipeline_ingress(rag.workspace)
        ingress.consume_auto_rescan()
        ingress.request_auto_rescan()

        extract.release.set()
        await asyncio.wait_for(proc, timeout=10)

        assert status.get("busy") is False
        assert status.get("cancellation_requested") is False
        assert ingress.counts()["auto_rescan_pending"] is True
        assert "Pipeline halted on internal storage error" in status.get(
            "latest_message", ""
        )
        assert "storage boom sentinel" in status.get("latest_message", "")
    finally:
        extract.release.set()
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_journaled_pending_doc_does_not_hold_busy(tmp_path):
    """A journaled custom-chunk PENDING doc is compacted out of the mailbox and
    does not livelock busy."""
    rag = await _build_rag(tmp_path, _MarkerExtract(), workspace_prefix="exit")
    try:
        status, _lock = await pipeline_status_handles(rag)
        await rag.apipeline_enqueue_documents(input="JJJJ body", file_paths="j.txt")
        j_id = compute_mdhash_id("j.txt", prefix="doc-")

        row = await rag.doc_status.get_by_id(j_id)
        row["metadata"] = {
            **(row.get("metadata") or {}),
            CUSTOM_CHUNK_PATCH_METADATA_KEY: {"phase": "staged"},
        }
        await rag.doc_status.upsert({j_id: row})

        await asyncio.wait_for(rag.apipeline_process_enqueue_documents(), timeout=10)

        assert status.get("busy") is False
        row = await rag.doc_status.get_by_id(j_id)
        assert _status_text(row) == "pending"
        assert (row.get("metadata") or {}).get(CUSTOM_CHUNK_PATCH_METADATA_KEY) == {
            "phase": "staged"
        }
        ingress = await get_pipeline_ingress(rag.workspace)
        assert ingress.counts()["documents"] == 0
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_direct_delete_hands_off_mailbox_work(tmp_path):
    """Fix-proof: direct SDK ``adelete_by_doc_id`` exit probes ``has_work()``
    and hands the slot to a processing run for concurrent inserts."""
    extract = _MarkerExtract()
    rag = await _build_rag(tmp_path, extract, workspace_prefix="exit")
    try:
        await rag.apipeline_enqueue_documents(input="AAAA body", file_paths="a.txt")
        await asyncio.wait_for(rag.apipeline_process_enqueue_documents(), timeout=10)
        a_id = compute_mdhash_id("a.txt", prefix="doc-")
        assert _status_text(await rag.doc_status.get_by_id(a_id)) == "processed"

        orig_get_by_id = rag.doc_status.get_by_id
        delete_entered = asyncio.Event()
        release_delete = asyncio.Event()

        async def gated_get_by_id(doc_id):
            if doc_id == a_id and not delete_entered.is_set():
                delete_entered.set()
                await release_delete.wait()
            return await orig_get_by_id(doc_id)

        rag.doc_status.get_by_id = gated_get_by_id
        try:
            delete_task = asyncio.create_task(rag.adelete_by_doc_id(a_id))
            await asyncio.wait_for(delete_entered.wait(), timeout=5)

            await asyncio.wait_for(
                rag.ainsert("BBBB body", file_paths="b.txt"), timeout=10
            )
            ingress = await get_pipeline_ingress(rag.workspace)
            assert ingress.has_work() is True

            release_delete.set()
            result = await asyncio.wait_for(delete_task, timeout=15)
            assert result.status == "success"
        finally:
            release_delete.set()
            rag.doc_status.get_by_id = orig_get_by_id

        b_id = compute_mdhash_id("b.txt", prefix="doc-")
        assert _status_text(await rag.doc_status.get_by_id(b_id)) == "processed"
        assert extract.calls["BBBB"] == 1
        status, _lock = await pipeline_status_handles(rag)
        assert status.get("busy") is False
        assert ingress.has_work() is False
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_direct_delete_acquire_window_failure_hands_off(tmp_path, monkeypatch):
    """Fix-proof: an escape in the delete acquire window still reaches the
    handoff-aware exit."""
    extract = _MarkerExtract()
    rag = await _build_rag(tmp_path, extract, workspace_prefix="exit")
    try:
        await rag.apipeline_enqueue_documents(input="AAAA body", file_paths="a.txt")
        await asyncio.wait_for(rag.apipeline_process_enqueue_documents(), timeout=10)
        a_id = compute_mdhash_id("a.txt", prefix="doc-")

        real_acquire = lightrag_module.acquire_reservation
        acquired_gate = asyncio.Event()
        release_gate = asyncio.Event()

        async def gated_acquire(*args, **kwargs):
            reservation = await real_acquire(*args, **kwargs)
            assert reservation.acquired is True
            acquired_gate.set()
            await release_gate.wait()
            raise RuntimeError("post-acquire window failure")

        monkeypatch.setattr(lightrag_module, "acquire_reservation", gated_acquire)
        delete_task = asyncio.create_task(rag.adelete_by_doc_id(a_id))
        await asyncio.wait_for(acquired_gate.wait(), timeout=5)

        await asyncio.wait_for(rag.ainsert("BBBB body", file_paths="b.txt"), timeout=10)
        ingress = await get_pipeline_ingress(rag.workspace)
        assert ingress.has_work() is True

        release_gate.set()
        result = await asyncio.wait_for(delete_task, timeout=15)
        assert result.status == "fail"
        assert "post-acquire window failure" in result.message
        b_id = compute_mdhash_id("b.txt", prefix="doc-")
        assert _status_text(await rag.doc_status.get_by_id(b_id)) == "processed"
        assert extract.calls["BBBB"] == 1
        status, _lock = await pipeline_status_handles(rag)
        assert status.get("busy") is False
        assert ingress.has_work() is False
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_direct_delete_acquire_window_cancel_releases_and_retains(
    tmp_path, monkeypatch
):
    """Cancellation counterpart of the acquire-window escape: releases slot,
    propagates CancelledError, and never loses concurrent work."""
    extract = _MarkerExtract()
    rag = await _build_rag(tmp_path, extract, workspace_prefix="exit")
    try:
        await rag.apipeline_enqueue_documents(input="AAAA body", file_paths="a.txt")
        await asyncio.wait_for(rag.apipeline_process_enqueue_documents(), timeout=10)
        a_id = compute_mdhash_id("a.txt", prefix="doc-")

        real_acquire = lightrag_module.acquire_reservation
        acquired_gate = asyncio.Event()
        never = asyncio.Event()

        async def gated_acquire(*args, **kwargs):
            reservation = await real_acquire(*args, **kwargs)
            assert reservation.acquired is True
            acquired_gate.set()
            await never.wait()
            return reservation

        monkeypatch.setattr(lightrag_module, "acquire_reservation", gated_acquire)
        delete_task = asyncio.create_task(rag.adelete_by_doc_id(a_id))
        await asyncio.wait_for(acquired_gate.wait(), timeout=5)

        await asyncio.wait_for(rag.ainsert("BBBB body", file_paths="b.txt"), timeout=10)
        ingress = await get_pipeline_ingress(rag.workspace)
        assert ingress.has_work() is True

        delete_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await delete_task

        status, _lock = await pipeline_status_handles(rag)
        assert status.get("busy") is False
        if ingress.has_work():
            await asyncio.wait_for(
                rag.apipeline_process_enqueue_documents(), timeout=10
            )
        b_id = compute_mdhash_id("b.txt", prefix="doc-")
        assert _status_text(await rag.doc_status.get_by_id(b_id)) == "processed"
        assert extract.calls["BBBB"] == 1
        assert ingress.has_work() is False
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_custom_chunks_exit_hands_off_on_ingress_only_work(tmp_path):
    """Fix-proof: ``ainsert_custom_chunks`` exit consults ``has_work()`` and
    hands off to a processing run for docs enqueued during custom-chunk insert."""
    extract = _MarkerExtract()
    rag = await _build_rag(tmp_path, extract, workspace_prefix="exit")
    try:
        status, _lock = await pipeline_status_handles(rag)
        b_id = compute_mdhash_id("b.txt", prefix="doc-")

        orig_cleanup = rag._insert_done_with_cleanup
        injected = False

        async def cleanup_then_inject():
            nonlocal injected
            if not injected:
                injected = True
                await rag.apipeline_enqueue_documents(
                    input="BBBB body", file_paths="b.txt"
                )
            await orig_cleanup()

        rag._insert_done_with_cleanup = cleanup_then_inject

        await asyncio.wait_for(
            rag.ainsert_custom_chunks("base text", ["alice is here"], doc_id="doc-cc"),
            timeout=10,
        )

        assert _status_text(await rag.doc_status.get_by_id(b_id)) == "processed"
        assert extract.calls["BBBB"] == 1
        assert status.get("busy") is False
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_custom_chunks_exit_hands_off_when_probe_fails(tmp_path):
    """A ``has_work`` probe failure fails TOWARD handoff so enqueued docs still run."""
    extract = _MarkerExtract()
    rag = await _build_rag(tmp_path, extract, workspace_prefix="exit")
    try:
        status, _lock = await pipeline_status_handles(rag)
        b_id = compute_mdhash_id("b.txt", prefix="doc-")
        ingress = await get_pipeline_ingress(rag.workspace)

        orig_cleanup = rag._insert_done_with_cleanup
        orig_has_work = ingress.has_work
        injected = False

        def flaky_has_work():
            ingress.has_work = orig_has_work  # one transient flake
            raise RuntimeError("manager down: has_work")

        async def cleanup_then_inject():
            nonlocal injected
            if not injected:
                injected = True
                await rag.apipeline_enqueue_documents(
                    input="BBBB body", file_paths="b.txt"
                )
                ingress.has_work = flaky_has_work
            await orig_cleanup()

        rag._insert_done_with_cleanup = cleanup_then_inject

        await asyncio.wait_for(
            rag.ainsert_custom_chunks("base text", ["alice is here"], doc_id="doc-cc"),
            timeout=10,
        )

        assert _status_text(await rag.doc_status.get_by_id(b_id)) == "processed"
        assert status.get("busy") is False
    finally:
        await rag.finalize_storages()


# ---------------------------------------------------------------------------
# Phase 2: In-batch feeder (latency win + safety invariants)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_document_arriving_mid_batch_is_fed_without_waiting(tmp_path):
    """Fix-proof: doc B enqueued while doc A is blocked completes without waiting."""
    extract = _MarkerExtract()
    extract.block_marker = "AAAA"
    rag = await _build_rag(
        tmp_path, extract, max_parallel_insert=2, workspace_prefix="feed"
    )
    try:
        await rag.apipeline_enqueue_documents(input="AAAA body", file_paths="a.txt")
        a_id = compute_mdhash_id("a.txt", prefix="doc-")
        b_id = compute_mdhash_id("b.txt", prefix="doc-")

        proc = asyncio.create_task(rag.apipeline_process_enqueue_documents())
        await asyncio.wait_for(extract.started.wait(), timeout=5)

        await rag.apipeline_enqueue_documents(input="BBBB body", file_paths="b.txt")

        await _wait_status(rag, b_id, "processed")
        a_row = await rag.doc_status.get_by_id(a_id)
        assert _status_text(a_row) != "processed"

        extract.release.set()
        await asyncio.wait_for(proc, timeout=5)
        assert _status_text(await rag.doc_status.get_by_id(a_id)) == "processed"
        assert extract.calls["AAAA"] == 1
        assert extract.calls["BBBB"] == 1
    finally:
        extract.release.set()
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_sequential_upload_wakeups_do_not_strand_later_files(tmp_path):
    """WebUI multi-file upload path: busy-refused process wakeups must not stop
    the feeder from routing later uploads while doc A blocks."""
    extract = _MarkerExtract()
    extract.block_marker = "AAAA"
    rag = await _build_rag(
        tmp_path, extract, max_parallel_insert=3, workspace_prefix="feed"
    )
    try:
        await rag.apipeline_enqueue_documents(input="AAAA body", file_paths="a.txt")
        a_id = compute_mdhash_id("a.txt", prefix="doc-")

        proc = asyncio.create_task(rag.apipeline_process_enqueue_documents())
        await asyncio.wait_for(extract.started.wait(), timeout=5)

        for name, body in (("b.txt", "BBBB"), ("c.txt", "CCCC")):
            await rag.apipeline_enqueue_documents(
                input=f"{body} body", file_paths=name
            )
            await rag.apipeline_process_enqueue_documents()  # refused: busy

        for name in ("b.txt", "c.txt"):
            await _wait_status(
                rag, compute_mdhash_id(name, prefix="doc-"), "processed"
            )
        assert _status_text(await rag.doc_status.get_by_id(a_id)) != "processed"

        extract.release.set()
        await asyncio.wait_for(proc, timeout=10)
        assert _status_text(await rag.doc_status.get_by_id(a_id)) == "processed"
        assert extract.calls["BBBB"] == 1
        assert extract.calls["CCCC"] == 1
    finally:
        extract.release.set()
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_feeder_deduplicates_inflight_document_message(tmp_path):
    """A document message echoing an inflight doc id is dropped, not re-routed."""
    extract = _MarkerExtract()
    extract.block_marker = "AAAA"
    rag = await _build_rag(
        tmp_path, extract, max_parallel_insert=2, workspace_prefix="feed"
    )
    try:
        await rag.apipeline_enqueue_documents(input="AAAA body", file_paths="a.txt")
        a_id = compute_mdhash_id("a.txt", prefix="doc-")

        proc = asyncio.create_task(rag.apipeline_process_enqueue_documents())
        await asyncio.wait_for(extract.started.wait(), timeout=5)

        ingress = await get_pipeline_ingress(rag.workspace)
        ingress.put_document(_doc_msg(a_id))
        await asyncio.sleep(0.1)

        extract.release.set()
        await asyncio.wait_for(proc, timeout=5)
        assert _status_text(await rag.doc_status.get_by_id(a_id)) == "processed"
        assert extract.calls["AAAA"] == 1
    finally:
        extract.release.set()
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_pending_manual_signal_does_not_busy_loop_the_feeder(tmp_path):
    """The feeder waits ONLY on the document channel: a sticky manual request
    must not spin it."""
    extract = _MarkerExtract()
    rag = await _build_rag(
        tmp_path, extract, max_parallel_insert=1, workspace_prefix="feed"
    )
    try:
        await rag.apipeline_enqueue_documents(input="AAAA body", file_paths="a.txt")
        a_id = compute_mdhash_id("a.txt", prefix="doc-")

        ingress = await get_pipeline_ingress(rag.workspace)
        ingress.request_manual_retry("req-x", _manual_msg("req-x"))

        await rag.apipeline_process_enqueue_documents()
        assert _status_text(await rag.doc_status.get_by_id(a_id)) == "processed"
        assert extract.calls["AAAA"] == 1
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_feeder_yields_on_cancellation(tmp_path):
    """The feeder stops admitting before draining when cancellation is set."""
    rag = await _build_rag(
        tmp_path, _MarkerExtract(), max_parallel_insert=1, workspace_prefix="feed"
    )
    try:
        ingress = await get_pipeline_ingress(rag.workspace)
        for i in range(50):
            ingress.put_document(_doc_msg(f"doc-{i}"))
        ctx = _make_feeder_ctx()
        ctx.pipeline_cancel_event.set()

        await asyncio.wait_for(rag._pipeline_feeder(ctx, ingress), timeout=3)

        assert ctx.parse_queues["native"].qsize() == 0
        assert ingress.has_work()
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_feeder_yields_when_manual_request_waiting(tmp_path):
    """The feeder stops admitting when a sticky manual retry request is waiting."""
    rag = await _build_rag(
        tmp_path, _MarkerExtract(), max_parallel_insert=1, workspace_prefix="feed"
    )
    try:
        ingress = await get_pipeline_ingress(rag.workspace)
        for i in range(20):
            ingress.put_document(_doc_msg(f"doc-{i}"))
        ingress.request_manual_retry("req-y", _manual_msg("req-y"))
        ctx = _make_feeder_ctx()

        await asyncio.wait_for(rag._pipeline_feeder(ctx, ingress), timeout=3)

        assert ctx.parse_queues["native"].qsize() == 0
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_feeder_teardown_republishes_undrained_messages(tmp_path):
    """On cancel teardown, drained-but-unresolved messages return to the mailbox."""
    rag = await _build_rag(
        tmp_path, _MarkerExtract(), max_parallel_insert=1, workspace_prefix="feed"
    )
    try:
        for i in range(3):
            await rag.apipeline_enqueue_documents(
                input=f"BODY{i}", file_paths=f"d{i}.txt"
            )
        ids = {compute_mdhash_id(f"d{i}.txt", prefix="doc-") for i in range(3)}

        ingress = await get_pipeline_ingress(rag.workspace)
        ingress.drain_documents()
        for doc_id in ids:
            ingress.put_document(_doc_msg(doc_id))

        entered = asyncio.Event()
        gate = asyncio.Event()
        orig_get = rag.full_docs.get_by_id

        async def blocking_get(doc_id):
            entered.set()
            await gate.wait()
            return await orig_get(doc_id)

        rag.full_docs.get_by_id = blocking_get

        ctx = _make_feeder_ctx()
        feeder = asyncio.create_task(rag._pipeline_feeder(ctx, ingress))
        await asyncio.wait_for(entered.wait(), timeout=3)
        feeder.cancel()
        with pytest.raises(asyncio.CancelledError):
            await feeder

        republished = {m.doc_id for m in ingress.drain_documents()}
        assert republished == ids
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_feeder_arms_auto_rescan_and_republishes_on_skip(tmp_path):
    """A SKIP (full_docs not yet visible) arms auto-rescan and re-publishes on exit."""
    rag = await _build_rag(
        tmp_path, _MarkerExtract(), max_parallel_insert=1, workspace_prefix="feed"
    )
    try:
        await rag.apipeline_enqueue_documents(input="BODYB", file_paths="b.txt")
        b_id = compute_mdhash_id("b.txt", prefix="doc-")

        ingress = await get_pipeline_ingress(rag.workspace)
        ingress.drain_documents()
        ingress.consume_auto_rescan()
        ingress.put_document(_doc_msg(b_id))

        orig_get = rag.full_docs.get_by_id

        async def not_visible(doc_id):
            return None if doc_id == b_id else await orig_get(doc_id)

        rag.full_docs.get_by_id = not_visible

        ctx = _make_feeder_ctx()
        feeder = asyncio.create_task(rag._pipeline_feeder(ctx, ingress))

        async def _armed():
            while not ingress.counts()["auto_rescan_pending"]:
                await asyncio.sleep(0.02)

        await asyncio.wait_for(_armed(), timeout=3)
        feeder.cancel()
        await asyncio.gather(feeder, return_exceptions=True)

        assert ctx.parse_queues["native"].qsize() == 0
        assert ingress.counts()["auto_rescan_pending"] is True
        republished = {m.doc_id for m in ingress.drain_documents()}
        assert b_id in republished
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_feeder_keeps_routing_while_auto_rescan_pending(tmp_path):
    """With paging on, a dirty auto-rescan flag must NOT stop the feeder."""
    rag = await _build_rag(
        tmp_path, _MarkerExtract(), max_parallel_insert=1, workspace_prefix="feed"
    )
    try:
        await rag.apipeline_enqueue_documents(input="BODYB", file_paths="b.txt")
        b_id = compute_mdhash_id("b.txt", prefix="doc-")

        ingress = await get_pipeline_ingress(rag.workspace)
        ingress.drain_documents()
        ingress.consume_auto_rescan()
        ingress.request_auto_rescan()
        ingress.put_document(_doc_msg(b_id))

        ctx = _make_feeder_ctx()
        feeder = asyncio.create_task(rag._pipeline_feeder(ctx, ingress))
        try:

            async def _routed():
                while ctx.parse_queues["native"].qsize() == 0:
                    await asyncio.sleep(0.02)

            await asyncio.wait_for(_routed(), timeout=3)
            assert b_id in ctx.inflight_doc_ids
            assert ingress.counts()["auto_rescan_pending"] is True
        finally:
            feeder.cancel()
            await asyncio.gather(feeder, return_exceptions=True)
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_feeder_yields_on_auto_rescan_when_paging_disabled(tmp_path):
    """With PIPELINE_SCHEDULING_PAGE_SIZE=0, a dirty auto-rescan flag yields
    immediately so the boundary rescan runs."""
    rag = await _build_rag(
        tmp_path, _MarkerExtract(), max_parallel_insert=1, workspace_prefix="feed"
    )
    try:
        rag.pipeline_scheduling_page_size = 0
        await rag.apipeline_enqueue_documents(input="BODYB", file_paths="b.txt")
        b_id = compute_mdhash_id("b.txt", prefix="doc-")

        ingress = await get_pipeline_ingress(rag.workspace)
        ingress.drain_documents()
        ingress.consume_auto_rescan()
        ingress.put_document(_doc_msg(b_id))
        ingress.request_auto_rescan()

        ctx = _make_feeder_ctx()
        await asyncio.wait_for(rag._pipeline_feeder(ctx, ingress), timeout=3)

        assert ctx.parse_queues["native"].qsize() == 0
        assert ingress.counts()["auto_rescan_pending"] is True
        assert {m.doc_id for m in ingress.drain_documents()} == {b_id}
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_feeder_yields_within_one_admission_on_mid_drain_cancel(tmp_path):
    """Control signals are re-checked before EVERY admission so a cancel stops
    a large drain after one doc and re-publishes the tail."""
    rag = await _build_rag(
        tmp_path, _MarkerExtract(), max_parallel_insert=1, workspace_prefix="feed"
    )
    try:
        n = 8
        for i in range(n):
            await rag.apipeline_enqueue_documents(
                input=f"BODY{i}", file_paths=f"m{i}.txt"
            )
        ids = {compute_mdhash_id(f"m{i}.txt", prefix="doc-") for i in range(n)}

        ingress = await get_pipeline_ingress(rag.workspace)
        ingress.drain_documents()
        for doc_id in ids:
            ingress.put_document(_doc_msg(doc_id))

        ctx = _make_feeder_ctx()
        orig_put = ctx.parse_queues["native"].put

        async def cancel_after_first_put(item):
            await orig_put(item)
            ctx.pipeline_cancel_event.set()

        ctx.parse_queues["native"].put = cancel_after_first_put

        await asyncio.wait_for(rag._pipeline_feeder(ctx, ingress), timeout=3)

        admitted = ctx.parse_queues["native"].qsize()
        assert admitted == 1
        republished = {m.doc_id for m in ingress.drain_documents()}
        assert admitted + len(republished) == n
    finally:
        await rag.finalize_storages()
