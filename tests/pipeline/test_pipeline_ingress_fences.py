"""Consolidated enqueue ingress exclusion and FAILED-retry semantics contract tests.

Covers (PipelineConcurrencyContract — Phase 0 & LR2 §9.1 / §9.2):
  * Mutual exclusion between enqueue writes and manual retry freezes:
    mid-enqueue freeze refusals (primary and ``dup-*`` records), reservation
    visibility during storage writes, pre-freeze reserved-caller completion
    exemption, rejection of unregistered self-attested tokens, scan enqueue
    exemption, and structured ``PipelineReservationConflictError`` metadata.
  * FAILED-retry semantics split: automatic runs skip FAILED documents and
    auto-resume interrupted orphans, sticky manual retry requests execute FIFO
    (one attempt each, ACKed only after complete scan/reset, terminal against
    replay), strict refetch error compensation, and empty-queue manual ACKs.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from lightrag import LightRAG
from lightrag.base import CURSOR_START, DocStatus
from lightrag.exceptions import PipelineReservationConflictError
from lightrag.kg.pipeline_ingress import (
    ManualRetryPublishResult,
    PipelineIngressMessage,
)
from lightrag.kg.shared_storage import (
    PipelineReservationConflict,
    acquire_enqueue_reservation,
    finalize_share_data,
    get_pipeline_ingress,
    initialize_share_data,
    make_owner_record,
)
from lightrag.pipeline import (
    _AUTO_RESUME_DOC_STATUSES,
    PipelineNextDecision,
    PipelineNextStep,
    _ManualDrainProgress,
)
from lightrag.utils import compute_mdhash_id

from ._fakes import (
    build_initialized_rag,
    pipeline_status_handles,
    single_chunk_chunking,
)
from .conftest import request_failed_retry

pytestmark = pytest.mark.offline


class _CountingExtract:
    """Instance-level extraction stub: succeeds or always fails, counting calls."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = 0

    async def __call__(self, chunks, *args, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("extract fail sentinel")
        return [({}, {}) for _ in chunks]


async def _status_of(rag: LightRAG, doc_id: str) -> str:
    row = await rag.doc_status.get_by_id(doc_id)
    raw = (row or {}).get("status")
    return raw.value if isinstance(raw, DocStatus) else str(raw or "<missing>")


# ---------------------------------------------------------------------------
# Enqueue write-side mutual exclusion with manual retry & ingress fences
# ---------------------------------------------------------------------------


class TestEnqueueIngressExclusion:
    @pytest.fixture(autouse=True)
    def _shared(self):
        initialize_share_data()
        yield
        finalize_share_data()

    @staticmethod
    async def _build_rag(tmp_path: Path, *, capacity: int = 0) -> LightRAG:
        return await build_initialized_rag(
            tmp_path,
            workspace_prefix="excl",
            max_parallel_insert=1,
            max_pending_documents=capacity,
        )

    @staticmethod
    def _freeze_during_dedup(rag: LightRAG, pipeline_status, lock) -> None:
        original = rag.doc_status.filter_keys

        async def _filter_keys(keys):
            async with lock:
                pipeline_status["manual_freeze_requested"] = True
            return await original(keys)

        rag.doc_status.filter_keys = _filter_keys

    @staticmethod
    def _spy_on_first_write(rag: LightRAG, pipeline_status, lock) -> dict:
        seen: dict = {}
        original = rag.full_docs.upsert

        async def _upsert(data):
            async with lock:
                seen["pending_enqueues"] = pipeline_status.get("pending_enqueues", 0)
            return await original(data)

        rag.full_docs.upsert = _upsert
        return seen

    @staticmethod
    async def _hold_reservation(rag: LightRAG, token: str, *, capacity: int = 10):
        pipeline_status, lock = await pipeline_status_handles(rag)
        result = await acquire_enqueue_reservation(
            pipeline_status,
            lock,
            token=token,
            reject_when=(),
            weight=1,
            capacity=capacity,
            active_count=0,
        )
        assert result.acquired is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("capacity", [0, 10], ids=["admission-off", "admission-on"])
    async def test_a_freeze_raised_mid_enqueue_refuses_the_write(
        self, tmp_path, capacity
    ):
        """Fix-proof: freeze raised after entry check refuses writes regardless
        of admission capacity setting."""
        rag = await self._build_rag(tmp_path, capacity=capacity)
        try:
            pipeline_status, lock = await pipeline_status_handles(rag)
            self._freeze_during_dedup(rag, pipeline_status, lock)

            with pytest.raises(RuntimeError, match="manual retry"):
                await rag.apipeline_enqueue_documents(
                    "hello world", file_paths="report.pdf", track_id="t-1"
                )

            assert dict(rag.doc_status._data) == {}
        finally:
            await rag.finalize_storages()

    @pytest.mark.asyncio
    async def test_the_freeze_refusal_also_covers_the_duplicate_records(self, tmp_path):
        """Duplicate ``dup-*`` FAILED rows must also be refused once frozen."""
        rag = await self._build_rag(tmp_path, capacity=10)
        try:
            await rag.apipeline_enqueue_documents(
                "hello world", file_paths="report.pdf", track_id="t-1"
            )
            before = dict(rag.doc_status._data)

            pipeline_status, lock = await pipeline_status_handles(rag)
            self._freeze_during_dedup(rag, pipeline_status, lock)

            with pytest.raises(RuntimeError, match="manual retry"):
                await rag.apipeline_enqueue_documents(
                    "different body", file_paths="report.pdf", track_id="t-2"
                )

            assert dict(rag.doc_status._data) == before
        finally:
            await rag.finalize_storages()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("capacity", [0, 10], ids=["admission-off", "admission-on"])
    async def test_the_enqueue_holds_a_reservation_when_it_writes(
        self, tmp_path, capacity
    ):
        """DRAIN_TO_IDLE proves idleness from ``pending_enqueues``, so writes
        must occur under an active reservation and release it afterwards."""
        rag = await self._build_rag(tmp_path, capacity=capacity)
        try:
            pipeline_status, lock = await pipeline_status_handles(rag)
            seen = self._spy_on_first_write(rag, pipeline_status, lock)

            await rag.apipeline_enqueue_documents(
                "hello world", file_paths="report.pdf", track_id="t-1"
            )

            assert seen["pending_enqueues"] == 1
            assert pipeline_status.get("pending_enqueues", 0) == 0
            assert dict(pipeline_status.get("pending_enqueue_tokens", {})) == {}
        finally:
            await rag.finalize_storages()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("capacity", [0, 10], ids=["admission-off", "admission-on"])
    async def test_a_caller_admitted_before_the_freeze_is_allowed_to_finish(
        self, tmp_path, capacity
    ):
        """§9.2: an endpoint reserved before a freeze is allowed to finish."""
        rag = await self._build_rag(tmp_path, capacity=capacity)
        try:
            pipeline_status, lock = await pipeline_status_handles(rag)
            await self._hold_reservation(rag, "endpoint-token", capacity=capacity)
            async with lock:
                pipeline_status["manual_freeze_requested"] = True

            await rag.apipeline_enqueue_documents(
                "hello world",
                file_paths="report.pdf",
                track_id="t-1",
                admission_token="endpoint-token",
            )
            assert len(dict(rag.doc_status._data)) == 1
            assert "endpoint-token" in pipeline_status["pending_enqueue_tokens"]
        finally:
            await rag.finalize_storages()

    @pytest.mark.asyncio
    async def test_an_unregistered_token_buys_no_exemption(self, tmp_path):
        """A self-attested token string that was never reserved is refused."""
        rag = await self._build_rag(tmp_path, capacity=10)
        try:
            pipeline_status, lock = await pipeline_status_handles(rag)
            async with lock:
                pipeline_status["manual_freeze_requested"] = True

            with pytest.raises(RuntimeError, match="manual retry"):
                await rag.apipeline_enqueue_documents(
                    "hello world",
                    file_paths="report.pdf",
                    track_id="t-1",
                    admission_token="never-reserved",
                )
            assert dict(rag.doc_status._data) == {}
            assert "never-reserved" not in (
                pipeline_status.get("pending_enqueue_tokens") or {}
            )
        finally:
            await rag.finalize_storages()

    @pytest.mark.asyncio
    async def test_a_freeze_raised_mid_enqueue_still_lets_a_reserved_caller_write(
        self, tmp_path
    ):
        """A pre-reserved caller is not refused when freeze rises mid-enqueue."""
        rag = await self._build_rag(tmp_path, capacity=10)
        try:
            pipeline_status, lock = await pipeline_status_handles(rag)
            await self._hold_reservation(rag, "endpoint-token")
            self._freeze_during_dedup(rag, pipeline_status, lock)

            await rag.apipeline_enqueue_documents(
                "hello world",
                file_paths="report.pdf",
                track_id="t-1",
                admission_token="endpoint-token",
            )
            assert len(dict(rag.doc_status._data)) == 1
        finally:
            await rag.finalize_storages()

    @pytest.mark.asyncio
    async def test_scan_enqueues_are_exempt_from_the_fence_and_the_reservation(
        self, tmp_path
    ):
        """§9.1: scan enqueues hold ``scanning_exclusive`` and bypass reservation."""
        rag = await self._build_rag(tmp_path, capacity=1)
        try:
            pipeline_status, lock = await pipeline_status_handles(rag)
            seen = self._spy_on_first_write(rag, pipeline_status, lock)
            async with lock:
                pipeline_status["manual_freeze_requested"] = True
                pipeline_status["scanning_exclusive"] = True

            await rag.apipeline_enqueue_documents(
                ["a", "b"],
                file_paths=["a.txt", "b.txt"],
                track_id="t-scan",
                from_scan=True,
            )
            assert len(dict(rag.doc_status._data)) == 2
            assert seen["pending_enqueues"] == 0
        finally:
            await rag.finalize_storages()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "fence,expected_conflict,expected_recovery",
        [
            (
                "manual_freeze_requested",
                PipelineReservationConflict.MANUAL_FREEZE,
                False,
            ),
            ("scanning_exclusive", PipelineReservationConflict.SCANNING, False),
            ("destructive_busy", PipelineReservationConflict.DESTRUCTIVE, False),
        ],
    )
    async def test_sdk_fence_refusal_is_structured(
        self, tmp_path, fence, expected_conflict, expected_recovery
    ):
        """§9.1: SDK fence refusals raise ``PipelineReservationConflictError``
        (a ``RuntimeError`` subclass) carrying conflict, fence, and recovery flag."""
        rag = await self._build_rag(tmp_path, capacity=0)
        try:
            pipeline_status, lock = await pipeline_status_handles(rag)
            async with lock:
                pipeline_status[fence] = True

            with pytest.raises(PipelineReservationConflictError) as excinfo:
                await rag.apipeline_enqueue_documents(
                    "hello world", file_paths="report.pdf", track_id="t-1"
                )

            error = excinfo.value
            assert isinstance(error, RuntimeError)
            assert error.conflict is expected_conflict
            assert error.fence == fence
            assert error.recovery_required is expected_recovery
            assert str(error)
            assert dict(rag.doc_status._data) == {}
        finally:
            await rag.finalize_storages()

    @pytest.mark.asyncio
    async def test_sdk_recovery_fence_refusal_is_distinguishable(self, tmp_path):
        """The 503 case: ``recovery_required`` reports ``recovery_required=True``
        and ``fence == 'recovery_required'``."""
        rag = await self._build_rag(tmp_path, capacity=0)
        try:
            pipeline_status, lock = await pipeline_status_handles(rag)
            async with lock:
                pipeline_status["recovery_required"] = {
                    "kind": "manual_drain_stalled",
                    "owner_key": "busy_owner",
                    "operation_record": None,
                    "message": "a drain could not advance.",
                }

            with pytest.raises(PipelineReservationConflictError) as excinfo:
                await rag.apipeline_enqueue_documents(
                    "hello world", file_paths="report.pdf", track_id="t-1"
                )

            error = excinfo.value
            assert error.conflict is PipelineReservationConflict.RECOVERY_REQUIRED
            assert error.recovery_required is True
            assert error.fence == "recovery_required"
            assert dict(rag.doc_status._data) == {}
        finally:
            await rag.finalize_storages()


# ---------------------------------------------------------------------------
# FAILED-retry semantics split (Phase 0 of ingress-mailbox migration)
# ---------------------------------------------------------------------------


async def _build_retry_rag(tmp_path: Path, extract: _CountingExtract) -> LightRAG:
    rag = await build_initialized_rag(
        tmp_path,
        workspace=f"frs-{uuid4().hex[:8]}",
        chunking_func=single_chunk_chunking,
        max_parallel_insert=1,
    )
    rag._process_extract_entities = extract  # type: ignore[assignment]
    return rag


async def _make_failed_doc(rag: LightRAG, name: str, extract: _CountingExtract) -> str:
    extract.fail = True
    await rag.apipeline_enqueue_documents(input=f"body of {name}", file_paths=name)
    await rag.apipeline_process_enqueue_documents()
    doc_id = compute_mdhash_id(name, prefix="doc-")
    assert await _status_of(rag, doc_id) == DocStatus.FAILED.value
    return doc_id


@pytest.mark.asyncio
async def test_upload_triggered_run_does_not_retry_failed(tmp_path):
    """Fix-proof: automatic runs triggered by uploads skip FAILED documents."""
    extract = _CountingExtract()
    rag = await _build_retry_rag(tmp_path, extract)
    try:
        failed_id = await _make_failed_doc(rag, "a.txt", extract)
        failed_calls = extract.calls

        extract.fail = False
        await rag.apipeline_enqueue_documents(input="body b", file_paths="b.txt")
        await rag.apipeline_process_enqueue_documents()

        b_id = compute_mdhash_id("b.txt", prefix="doc-")
        assert await _status_of(rag, b_id) == DocStatus.PROCESSED.value
        assert await _status_of(rag, failed_id) == DocStatus.FAILED.value
        assert extract.calls == failed_calls + 1
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_manual_retry_processes_failed_and_acks(tmp_path):
    extract = _CountingExtract()
    rag = await _build_retry_rag(tmp_path, extract)
    try:
        failed_id = await _make_failed_doc(rag, "a.txt", extract)
        ingress = await get_pipeline_ingress(rag.workspace)

        extract.fail = False
        request_id = await request_failed_retry(rag)
        await rag.apipeline_process_enqueue_documents()

        assert await _status_of(rag, failed_id) == DocStatus.PROCESSED.value
        assert ingress.snapshot_manual_retries() == []
        assert (
            ingress.request_manual_retry(
                request_id,
                PipelineIngressMessage(
                    kind="rescan", retry_failed=True, request_id=request_id
                ),
            )
            is ManualRetryPublishResult.ALREADY_TERMINAL
        )

        second = await request_failed_retry(rag)
        await rag.apipeline_process_enqueue_documents()
        assert ingress.snapshot_manual_retries() == []
        assert second != request_id
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_two_requests_fifo_one_attempt_each(tmp_path):
    """Two pending manual requests grant exactly one attempt each, FIFO."""
    extract = _CountingExtract()
    rag = await _build_retry_rag(tmp_path, extract)
    try:
        failed_id = await _make_failed_doc(rag, "a.txt", extract)
        calls_after_failure = extract.calls
        ingress = await get_pipeline_ingress(rag.workspace)

        r1 = await request_failed_retry(rag)
        r2 = await request_failed_retry(rag)
        assert [m.request_id for m in ingress.snapshot_manual_retries()] == [r1, r2]

        await rag.apipeline_process_enqueue_documents()

        assert await _status_of(rag, failed_id) == DocStatus.FAILED.value
        assert extract.calls == calls_after_failure + 2
        assert ingress.snapshot_manual_retries() == []

        await rag.apipeline_process_enqueue_documents()
        assert extract.calls == calls_after_failure + 2
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_strict_scan_failure_keeps_request_sticky(tmp_path):
    """A failing strict scan keeps the manual request sticky for the next run."""
    extract = _CountingExtract()
    rag = await _build_retry_rag(tmp_path, extract)
    try:
        rag.pipeline_scheduling_page_size = 0
        failed_id = await _make_failed_doc(rag, "a.txt", extract)
        ingress = await get_pipeline_ingress(rag.workspace)
        request_id = await request_failed_retry(rag)

        original = rag.doc_status.get_docs_by_statuses
        boom = {"armed": True}

        async def flaky(statuses, strict=False):
            if strict and boom["armed"]:
                boom["armed"] = False
                raise RuntimeError("strict scan boom")
            return await original(statuses, strict=strict)

        rag.doc_status.get_docs_by_statuses = flaky
        with pytest.raises(RuntimeError, match="strict scan boom"):
            await rag.apipeline_process_enqueue_documents()

        assert [m.request_id for m in ingress.snapshot_manual_retries()] == [request_id]

        extract.fail = False
        await rag.apipeline_process_enqueue_documents()
        assert await _status_of(rag, failed_id) == DocStatus.PROCESSED.value
        assert ingress.snapshot_manual_retries() == []
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_auto_rescan_rearmed_when_strict_refetch_fails(tmp_path):
    """CONTINUE_AUTO re-arms auto-rescan on refetch failure; CONTINUE_MANUAL
    stays sticky on its own and raises the manual freeze."""
    extract = _CountingExtract()
    rag = await _build_retry_rag(tmp_path, extract)
    try:
        rag.pipeline_scheduling_page_size = 0
        ingress = await get_pipeline_ingress(rag.workspace)
        status, lock = await pipeline_status_handles(rag)
        token = "owner-tok"
        status.update(
            {"busy": True, "busy_owner": make_owner_record(token, "processing")}
        )

        async def boom(statuses, strict=False):
            raise RuntimeError("refetch boom")

        rag.doc_status.get_docs_by_statuses = boom
        decision = PipelineNextDecision(PipelineNextStep.CONTINUE_AUTO)
        with pytest.raises(RuntimeError, match="refetch boom"):
            await rag._refetch_for_decision(
                decision,
                _AUTO_RESUME_DOC_STATUSES,
                CURSOR_START,
                ingress,
                token=token,
                pipeline_status=status,
                pipeline_status_lock=lock,
                drain_progress=_ManualDrainProgress(),
            )
        assert ingress.consume_auto_rescan() is True

        with pytest.raises(RuntimeError, match="refetch boom"):
            await rag._refetch_for_decision(
                PipelineNextDecision(
                    PipelineNextStep.CONTINUE_MANUAL, manual_request_id="r1"
                ),
                _AUTO_RESUME_DOC_STATUSES,
                CURSOR_START,
                ingress,
                token=token,
                pipeline_status=status,
                pipeline_status_lock=lock,
                drain_progress=_ManualDrainProgress(),
            )
        assert ingress.consume_auto_rescan() is False
        assert status["manual_freeze_requested"] is True
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_interrupted_statuses_still_auto_resume(tmp_path):
    """Dead-process orphans (PROCESSING/PARSING/ANALYZING) self-heal on any run."""
    extract = _CountingExtract()
    rag = await _build_retry_rag(tmp_path, extract)
    try:
        await rag.apipeline_enqueue_documents(input="body a", file_paths="a.txt")
        doc_id = compute_mdhash_id("a.txt", prefix="doc-")
        row = await rag.doc_status.get_by_id(doc_id)
        await rag.doc_status.upsert({doc_id: {**row, "status": DocStatus.PROCESSING}})

        await rag.apipeline_process_enqueue_documents()
        assert await _status_of(rag, doc_id) == DocStatus.PROCESSED.value
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_manual_request_on_empty_queue_acks_as_complete(tmp_path):
    """A legitimately-empty complete manual scan ACKs and releases."""
    extract = _CountingExtract()
    rag = await _build_retry_rag(tmp_path, extract)
    try:
        ingress = await get_pipeline_ingress(rag.workspace)
        await request_failed_retry(rag)
        await rag.apipeline_process_enqueue_documents()
        assert ingress.snapshot_manual_retries() == []
        assert not ingress.has_work()
    finally:
        await rag.finalize_storages()
