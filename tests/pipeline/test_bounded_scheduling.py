"""Consolidated bounded scheduling, scheduler memory, admission capacity, and metrics tests.

Covers (PipelineConcurrencyContract — LR2 Phases 2, 5-a, 6):
  * Bounded keyset scheduling sweeps (`PIPELINE_SCHEDULING_PAGE_SIZE`), legacy
    single-scan mode (`page_size=0`), mid-sweep page-failure auto-rescan
    recovery, and single-epoch feeder capping.
  * Allocation ratio bounds via ``tracemalloc`` over production
    ``_next_scheduling_page`` and ``_next_failed_page`` + ``_reset_failed_page``
    (process-level RSS E2E stays in ``test_scheduler_memory_bounded_e2e.py``).
  * Admission capacity enforcement (``MAX_PENDING_DOCUMENTS``): disabled
    default, structured ``PipelineBackpressureError``, post-dedup charging,
    whole-batch refusal, concurrent reservation weight accounting, token
    re-weighting without self-collision, fail-closed count errors, scan
    exemption, and construction validation.
  * Fixed-cardinality ``pipeline_metrics`` registry and load-bearing
    instrumentation (freeze rejects and active-count latency on failure).
"""

from __future__ import annotations

import asyncio
import tracemalloc
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lightrag import LightRAG, pipeline_metrics
from lightrag.base import (
    CURSOR_END,
    CURSOR_START,
    CursorAfter,
    DocProcessingStatus,
    DocSchedulingRecord,
    DocStatus,
    DocStatusPage,
)
from lightrag.exceptions import PipelineBackpressureError, StorageControlPlaneError
from lightrag.kg.shared_storage import (
    _conflict_for_status_flag,
    acquire_enqueue_reservation,
    finalize_share_data,
    get_pipeline_ingress,
    initialize_share_data,
)
from lightrag.pipeline import (
    _AUTO_RESUME_DOC_STATUSES,
    _BatchRunContext,
    _PipelineMixin,
)
from lightrag.utils import compute_mdhash_id
from lightrag.utils_pipeline import count_active_documents

from ._fakes import (
    build_initialized_rag,
    make_lightrag,
    pipeline_status_handles,
    single_chunk_chunking,
)

pytestmark = pytest.mark.offline

_DEFAULT_MEM_PAGE = 200
_SYNTH_BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
_SYNTH_TOKEN = "owner-token"


class _NoopExtract:
    def __init__(self, fail: bool = False):
        self.fail = fail

    async def __call__(self, chunks, *args, **kwargs):
        if self.fail:
            raise RuntimeError("extract fail sentinel")
        return [({}, {}) for _ in chunks]


async def _build_sweep_rag(tmp_path: Path, *, page_size: int) -> LightRAG:
    rag = await build_initialized_rag(
        tmp_path,
        workspace_prefix="bs",
        chunking_func=single_chunk_chunking,
        max_parallel_insert=1,
        pipeline_scheduling_page_size=page_size,
    )
    rag._process_extract_entities = _NoopExtract()  # type: ignore[assignment]
    return rag


async def _status_of(rag: LightRAG, doc_id: str) -> str:
    row = await rag.doc_status.get_by_id(doc_id)
    raw = (row or {}).get("status")
    return raw.value if isinstance(raw, DocStatus) else str(raw or "<missing>")


async def _enqueue_n(rag: LightRAG, n: int) -> list[str]:
    ids = []
    for i in range(n):
        name = f"doc{i}.txt"
        await rag.apipeline_enqueue_documents(input=f"body {i}", file_paths=name)
        ids.append(compute_mdhash_id(name, prefix="doc-"))
    return ids


def _count_page_calls(rag: LightRAG) -> dict[str, int]:
    counts = {"page": 0, "scan": 0}
    orig_page = rag.doc_status.get_docs_by_statuses_page
    orig_scan = rag.doc_status.get_docs_by_statuses

    async def counting_page(statuses, *, limit, position=CURSOR_START, strict=False):
        counts["page"] += 1
        return await orig_page(statuses, limit=limit, position=position, strict=strict)

    async def counting_scan(statuses, strict=False):
        counts["scan"] += 1
        return await orig_scan(statuses, strict=strict)

    rag.doc_status.get_docs_by_statuses_page = counting_page
    rag.doc_status.get_docs_by_statuses = counting_scan
    return counts


# ---------------------------------------------------------------------------
# Bounded scheduling acceptance tests (LR2 Phase 2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bounded_sweep_processes_all_docs_across_pages(tmp_path):
    """A backlog larger than one page drains page-by-page without starving the tail."""
    rag = await _build_sweep_rag(tmp_path, page_size=2)
    try:
        ids = await _enqueue_n(rag, 5)
        counts = _count_page_calls(rag)

        await rag.apipeline_process_enqueue_documents()

        for doc_id in ids:
            assert await _status_of(rag, doc_id) == DocStatus.PROCESSED.value
        assert counts["page"] >= 3, counts
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_page_size_zero_uses_legacy_single_scan(tmp_path):
    """PIPELINE_SCHEDULING_PAGE_SIZE=0 disables paging and uses single scan."""
    rag = await _build_sweep_rag(tmp_path, page_size=0)
    try:
        ids = await _enqueue_n(rag, 3)
        counts = _count_page_calls(rag)

        await rag.apipeline_process_enqueue_documents()

        for doc_id in ids:
            assert await _status_of(rag, doc_id) == DocStatus.PROCESSED.value
        assert counts["page"] == 0, "paging off -> page API never called"
        assert counts["scan"] >= 1, "legacy single scan drove the sweep"
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_sweep_page_failure_rearms_auto_and_recovers(tmp_path):
    """A mid-sweep page failure re-arms auto-rescan and recovers cleanly on rerun."""
    rag = await _build_sweep_rag(tmp_path, page_size=1)
    try:
        ids = await _enqueue_n(rag, 3)
        ingress = await get_pipeline_ingress(rag.workspace)
        ingress.consume_auto_rescan()

        orig_page = rag.doc_status.get_docs_by_statuses_page
        calls = {"n": 0}

        async def flaky_page(statuses, *, limit, position=CURSOR_START, strict=False):
            calls["n"] += 1
            if calls["n"] == 2:
                raise ConnectionError("backend page failure")
            return await orig_page(
                statuses, limit=limit, position=position, strict=strict
            )

        rag.doc_status.get_docs_by_statuses_page = flaky_page

        with pytest.raises(ConnectionError, match="backend page failure"):
            await rag.apipeline_process_enqueue_documents()

        assert ingress.counts()["auto_rescan_pending"] is True
        statuses = [await _status_of(rag, d) for d in ids]
        assert DocStatus.PENDING.value in statuses, statuses

        rag.doc_status.get_docs_by_statuses_page = orig_page
        await rag.apipeline_process_enqueue_documents()
        for doc_id in ids:
            assert await _status_of(rag, doc_id) == DocStatus.PROCESSED.value
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_feeder_epoch_full_gate(tmp_path):
    """_feeder_epoch_full is True once inflight reaches page_size, and False when paging is off."""
    rag = await _build_sweep_rag(tmp_path, page_size=2)
    try:
        ctx = _BatchRunContext(
            pipeline_status={},
            pipeline_status_lock=asyncio.Lock(),
            semaphore=asyncio.Semaphore(1),
            total_files=0,
            parse_queues={},
            parser_specs={},
            q_analyze=asyncio.Queue(),
            q_process=asyncio.Queue(),
            pipeline_cancel_event=None,
        )
        assert rag._feeder_epoch_full(ctx) is False
        ctx.inflight_doc_ids.update({"a", "b"})
        assert rag._feeder_epoch_full(ctx) is True

        rag.pipeline_scheduling_page_size = 0
        assert rag._feeder_epoch_full(ctx) is False
    finally:
        await rag.finalize_storages()


# ---------------------------------------------------------------------------
# Scheduler memory does not grow with backlog (tracemalloc ratio bounds)
# ---------------------------------------------------------------------------


class _SyntheticDocStatus:
    """Serves pages on demand in O(1) memory so traced growth belongs to the scheduler."""

    supports_strict_point_reads = True

    def __init__(self, total: int, status: DocStatus = DocStatus.PENDING):
        self.total = total
        self.status = status
        self.pages_served = 0
        self.upserted = 0

    def _key(self, index: int) -> tuple[str, str]:
        return (
            (_SYNTH_BASE + timedelta(seconds=index)).isoformat(),
            f"doc-{index:09d}",
        )

    def _index_of(self, doc_id: str) -> int:
        return int(doc_id.rsplit("-", 1)[1])

    async def get_docs_by_statuses_page(
        self, statuses, *, limit, position=CURSOR_START, strict=False
    ):
        start = 0
        if isinstance(position, CursorAfter):
            start = int(position.opaque) + 1
        end = min(start + limit, self.total)
        docs = {}
        for i in range(start, end):
            created_at, doc_id = self._key(i)
            docs[doc_id] = DocSchedulingRecord(
                id=doc_id,
                status=self.status,
                created_at=created_at,
                updated_at=created_at,
                file_path=f"file-{i}.md",
                track_id="t",
                has_custom_chunk_journal=False,
            )
        self.pages_served += 1
        next_position = CURSOR_END if end >= self.total else CursorAfter(str(end - 1))
        return DocStatusPage(docs=docs, next_position=next_position)

    async def get_full_docs_by_ids(self, ids, *, strict=False):
        hydrated = {}
        for doc_id in ids:
            index = self._index_of(doc_id)
            created_at, _ = self._key(index)
            hydrated[doc_id] = DocProcessingStatus(
                content_summary=f"summary-{index}",
                content_length=64,
                status=self.status,
                created_at=created_at,
                updated_at=created_at,
                file_path=f"file-{index}.md",
                track_id="t",
                chunks_count=0,
                chunks_list=[],
                metadata={},
            )
        return hydrated

    async def get_by_id_strict(self, doc_id: str):
        return {"content": "body", "file_path": f"file-{self._index_of(doc_id)}.md"}

    async def upsert(self, data: dict) -> None:
        self.upserted += len(data)


def _pipeline_over(
    doc_status: _SyntheticDocStatus, *, page_size: int = _DEFAULT_MEM_PAGE
) -> _PipelineMixin:
    class _Pipeline(_PipelineMixin):
        def __init__(self):
            self.doc_status = doc_status
            self.full_docs = doc_status
            self.pipeline_scheduling_page_size = page_size

    return _Pipeline()


async def _sweep_peak_kib(
    total: int, *, page_size: int = _DEFAULT_MEM_PAGE
) -> tuple[float, int]:
    doc_status = _SyntheticDocStatus(total)
    pipeline = _pipeline_over(doc_status, page_size=page_size)

    tracemalloc.start()
    tracemalloc.reset_peak()
    try:
        position = CURSOR_START
        routed = 0
        while True:
            docs, position = await pipeline._next_scheduling_page(
                _AUTO_RESUME_DOC_STATUSES, position
            )
            for doc_id in docs:
                routed += len(doc_id)
            if position is CURSOR_END:
                break
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert routed > 0
    return peak / 1024, doc_status.pages_served


async def _reset_peak_kib(
    total: int, *, page_size: int = _DEFAULT_MEM_PAGE
) -> tuple[float, int]:
    doc_status = _SyntheticDocStatus(total, status=DocStatus.FAILED)
    pipeline = _pipeline_over(doc_status, page_size=page_size)
    pipeline_status = {"busy_owner": {"token": _SYNTH_TOKEN}, "history_messages": []}
    lock = asyncio.Lock()

    tracemalloc.start()
    tracemalloc.reset_peak()
    try:
        position = CURSOR_START
        while True:
            docs, position = await pipeline._next_failed_page(position)
            if docs:
                await pipeline._reset_failed_page(
                    docs, _SYNTH_TOKEN, pipeline_status, lock
                )
            if position is CURSOR_END:
                break
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert doc_status.upserted == total
    return peak / 1024, doc_status.pages_served


@pytest.mark.asyncio
async def test_sweep_peak_memory_does_not_track_the_backlog():
    """10x the documents, same page size: memory must stay flat, not scale."""
    small_kib, small_pages = await _sweep_peak_kib(1_000)
    large_kib, large_pages = await _sweep_peak_kib(10_000)

    assert large_pages == 10 * small_pages
    assert large_kib < max(3.0 * small_kib, small_kib + 256), (
        f"peak grew with the backlog: {small_kib:.1f} KiB at 1k docs, "
        f"{large_kib:.1f} KiB at 10k docs"
    )


@pytest.mark.asyncio
async def test_failed_reset_peak_memory_does_not_track_the_backlog():
    """Manual FAILED->PENDING reset releases each page's dict instead of accumulating."""
    small_kib, small_pages = await _reset_peak_kib(1_000)
    large_kib, large_pages = await _reset_peak_kib(10_000)

    assert large_pages == 10 * small_pages
    assert large_kib < max(3.0 * small_kib, small_kib + 256), (
        f"reset peak grew with the backlog: {small_kib:.1f} KiB at 1k FAILED, "
        f"{large_kib:.1f} KiB at 10k FAILED"
    )


@pytest.mark.asyncio
async def test_page_size_not_backlog_size_sets_the_peak():
    """Complement: peak memory DOES grow when page_size grows."""
    small_page_kib, _ = await _sweep_peak_kib(10_000, page_size=50)
    large_page_kib, _ = await _sweep_peak_kib(10_000, page_size=2_000)

    assert large_page_kib > small_page_kib, (
        "peak did not respond to page size at all, so the measurement is not "
        f"observing the page: {small_page_kib:.1f} vs {large_page_kib:.1f} KiB"
    )


@pytest.mark.asyncio
async def test_a_backlog_sized_accumulator_is_caught():
    """Fix-proof for the measurement itself: an O(backlog) list fails the ratio."""

    async def _leaky_sweep_peak_kib(total: int) -> float:
        doc_status = _SyntheticDocStatus(total)
        tracemalloc.start()
        tracemalloc.reset_peak()
        try:
            position = CURSOR_START
            accumulated = []
            while True:
                page = await doc_status.get_docs_by_statuses_page(
                    [DocStatus.PENDING],
                    limit=_DEFAULT_MEM_PAGE,
                    position=position,
                    strict=True,
                )
                accumulated.extend(page.docs.values())
                if page.next_position is CURSOR_END:
                    break
                position = page.next_position
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert accumulated
        return peak / 1024

    small_kib = await _leaky_sweep_peak_kib(1_000)
    large_kib = await _leaky_sweep_peak_kib(10_000)

    assert large_kib >= 3.0 * small_kib, (
        "the ratio assertion would not have caught a backlog-sized accumulator: "
        f"{small_kib:.1f} -> {large_kib:.1f} KiB"
    )


# ---------------------------------------------------------------------------
# Admission capacity at the enqueue chokepoint (LR2 Phase 5-a)
# ---------------------------------------------------------------------------


class TestAdmissionCapacity:
    @pytest.fixture(autouse=True)
    def _shared(self):
        initialize_share_data()
        yield
        finalize_share_data()

    @staticmethod
    async def _build_rag(tmp_path: Path, *, capacity: int) -> LightRAG:
        return await build_initialized_rag(
            tmp_path,
            workspace_prefix="adm",
            chunking_func=single_chunk_chunking,
            max_parallel_insert=1,
            max_pending_documents=capacity,
        )

    @staticmethod
    def _count_spy(rag: LightRAG) -> dict:
        state = {"calls": 0, "raise": None}
        original = rag.doc_status.count_docs_by_statuses

        async def _counting(statuses, *, strict=True):
            state["calls"] += 1
            if state["raise"] is not None:
                raise state["raise"]
            return await original(statuses, strict=strict)

        rag.doc_status.count_docs_by_statuses = _counting
        return state

    @staticmethod
    async def _pending_count(rag: LightRAG) -> int:
        return await rag.doc_status.count_docs_by_statuses(
            [
                DocStatus.PENDING,
                DocStatus.PARSING,
                DocStatus.ANALYZING,
                DocStatus.PROCESSING,
            ]
        )

    @pytest.mark.asyncio
    async def test_admission_disabled_by_default_takes_no_count(self, tmp_path):
        rag = await self._build_rag(tmp_path, capacity=0)
        try:
            spy = self._count_spy(rag)
            for i in range(5):
                await rag.apipeline_enqueue_documents(
                    input=f"body {i}", file_paths=f"doc{i}.txt"
                )
            assert spy["calls"] == 0
            assert await self._pending_count(rag) == 5
        finally:
            await rag.finalize_storages()

    @pytest.mark.asyncio
    async def test_capacity_refuses_with_structured_backpressure(self, tmp_path):
        rag = await self._build_rag(tmp_path, capacity=2)
        try:
            await rag.apipeline_enqueue_documents(input="a", file_paths="a.txt")
            await rag.apipeline_enqueue_documents(input="b", file_paths="b.txt")

            with pytest.raises(PipelineBackpressureError) as excinfo:
                await rag.apipeline_enqueue_documents(input="c", file_paths="c.txt")

            error = excinfo.value
            assert (error.current, error.requested, error.capacity) == (2, 1, 2)
            assert "capacity" in str(error)
            assert await rag.doc_status.get_by_id_strict("doc-anything") is None
            assert await self._pending_count(rag) == 2
        finally:
            await rag.finalize_storages()

    @pytest.mark.asyncio
    async def test_batch_is_charged_after_dedup(self, tmp_path):
        rag = await self._build_rag(tmp_path, capacity=1)
        try:
            await rag.apipeline_enqueue_documents(
                input=["same body", "same body", "same body"],
                file_paths=["a.txt", "b.txt", "c.txt"],
            )
            assert await self._pending_count(rag) == 1
        finally:
            await rag.finalize_storages()

    @pytest.mark.asyncio
    async def test_whole_batch_is_refused_when_it_does_not_fit(self, tmp_path):
        rag = await self._build_rag(tmp_path, capacity=2)
        try:
            with pytest.raises(PipelineBackpressureError) as excinfo:
                await rag.apipeline_enqueue_documents(
                    input=["a", "b", "c"],
                    file_paths=["a.txt", "b.txt", "c.txt"],
                )
            assert excinfo.value.requested == 3
            assert await self._pending_count(rag) == 0
        finally:
            await rag.finalize_storages()

    @pytest.mark.asyncio
    async def test_other_requests_reservation_weight_counts(self, tmp_path):
        rag = await self._build_rag(tmp_path, capacity=1)
        try:
            pipeline_status, lock = await pipeline_status_handles(rag)
            await acquire_enqueue_reservation(
                pipeline_status,
                lock,
                token="other-request",
                reject_when=(),
                weight=1,
                capacity=1,
                active_count=0,
            )

            with pytest.raises(PipelineBackpressureError) as excinfo:
                await rag.apipeline_enqueue_documents(input="a", file_paths="a.txt")
            assert excinfo.value.current == 1
        finally:
            await rag.finalize_storages()

    @pytest.mark.asyncio
    async def test_reweighting_own_token_does_not_collide_with_itself(self, tmp_path):
        rag = await self._build_rag(tmp_path, capacity=3)
        try:
            pipeline_status, lock = await pipeline_status_handles(rag)
            for weight in (1, 3):
                result = await acquire_enqueue_reservation(
                    pipeline_status,
                    lock,
                    token="mine",
                    reject_when=(),
                    weight=weight,
                    capacity=3,
                    active_count=0,
                )
                assert result.acquired is True
            tokens = pipeline_status["pending_enqueue_tokens"]
            assert tokens["mine"]["weight"] == 3
            assert pipeline_status["pending_enqueues"] == 1

            await rag.apipeline_enqueue_documents(
                input=["a", "b", "c"],
                file_paths=["a.txt", "b.txt", "c.txt"],
                admission_token="mine",
            )
            assert await self._pending_count(rag) == 3
        finally:
            await rag.finalize_storages()

    @pytest.mark.asyncio
    async def test_count_failure_fails_closed(self, tmp_path):
        rag = await self._build_rag(tmp_path, capacity=10)
        try:
            spy = self._count_spy(rag)
            spy["raise"] = StorageControlPlaneError("index unavailable")

            with pytest.raises(StorageControlPlaneError):
                await rag.apipeline_enqueue_documents(input="a", file_paths="a.txt")
            spy["raise"] = None
            assert await self._pending_count(rag) == 0
        finally:
            await rag.finalize_storages()

    @pytest.mark.asyncio
    async def test_scan_breaks_through_the_cap(self, tmp_path):
        rag = await self._build_rag(tmp_path, capacity=1)
        try:
            for i in range(3):
                await rag.apipeline_enqueue_documents(
                    input=f"scanned {i}",
                    file_paths=f"scan{i}.txt",
                    from_scan=True,
                )
            assert await self._pending_count(rag) == 3

            with pytest.raises(PipelineBackpressureError):
                await rag.apipeline_enqueue_documents(
                    input="ordinary", file_paths="ordinary.txt"
                )
        finally:
            await rag.finalize_storages()

    @pytest.mark.asyncio
    async def test_capacity_requires_a_count(self, tmp_path):
        rag = await self._build_rag(tmp_path, capacity=1)
        try:
            pipeline_status, lock = await pipeline_status_handles(rag)
            with pytest.raises(ValueError, match="active_count"):
                await acquire_enqueue_reservation(
                    pipeline_status,
                    lock,
                    token="t",
                    reject_when=(),
                    weight=1,
                    capacity=1,
                )
        finally:
            await rag.finalize_storages()

    def test_negative_capacity_is_rejected_at_construction(self, tmp_path):
        with pytest.raises(ValueError, match="MAX_PENDING_DOCUMENTS"):
            make_lightrag(
                tmp_path,
                subdir="wd2",
                workspace="adm-negative",
                chunking_func=single_chunk_chunking,
                max_pending_documents=-1,
            )


# ---------------------------------------------------------------------------
# Bounded scheduling metrics registry & instrumentation (LR2 Phase 6)
# ---------------------------------------------------------------------------


class TestPipelineMetrics:
    @pytest.fixture(autouse=True)
    def _clean_metrics(self):
        pipeline_metrics.reset()
        yield
        pipeline_metrics.reset()

    def test_counters_and_durations_start_at_zero(self):
        snapshot = pipeline_metrics.snapshot()

        assert snapshot["counters"][pipeline_metrics.FREEZE_REJECTS] == 0
        summary = snapshot["durations"][pipeline_metrics.ACTIVE_COUNT_SECONDS]
        assert summary == {
            "count": 0,
            "total_seconds": 0.0,
            "max_seconds": 0.0,
            "last_seconds": 0.0,
            "mean_seconds": 0.0,
        }

    def test_duration_summary_tracks_count_max_and_mean(self):
        pipeline_metrics.observe(pipeline_metrics.SCHEDULING_PAGE_SECONDS, 0.1)
        pipeline_metrics.observe(pipeline_metrics.SCHEDULING_PAGE_SECONDS, 0.3)

        summary = pipeline_metrics.snapshot()["durations"][
            pipeline_metrics.SCHEDULING_PAGE_SECONDS
        ]
        assert summary["count"] == 2
        assert summary["max_seconds"] == 0.3
        assert summary["last_seconds"] == 0.3
        assert summary["mean_seconds"] == 0.2

    def test_unknown_names_are_dropped_not_inserted(self):
        pipeline_metrics.increment("not_a_declared_counter")
        pipeline_metrics.observe("not_a_declared_duration", 1.0)

        snapshot = pipeline_metrics.snapshot()
        assert "not_a_declared_counter" not in snapshot["counters"]
        assert "not_a_declared_duration" not in snapshot["durations"]

    def test_negative_and_zero_samples_cannot_corrupt_a_summary(self):
        pipeline_metrics.increment(pipeline_metrics.FREEZE_REJECTS, 0)
        pipeline_metrics.increment(pipeline_metrics.FREEZE_REJECTS, -5)
        pipeline_metrics.observe(pipeline_metrics.MANUAL_DRAIN_SECONDS, -2.0)

        snapshot = pipeline_metrics.snapshot()
        assert snapshot["counters"][pipeline_metrics.FREEZE_REJECTS] == 0
        drain = snapshot["durations"][pipeline_metrics.MANUAL_DRAIN_SECONDS]
        assert drain["count"] == 1 and drain["max_seconds"] == 0.0

    def test_freeze_rejection_is_counted_at_the_shared_chokepoint(self):
        before = pipeline_metrics.snapshot()["counters"][
            pipeline_metrics.FREEZE_REJECTS
        ]

        _conflict_for_status_flag("manual_freeze_requested")
        _conflict_for_status_flag("destructive_busy")  # different fence: not counted

        after = pipeline_metrics.snapshot()["counters"][pipeline_metrics.FREEZE_REJECTS]
        assert after == before + 1

    @pytest.mark.asyncio
    async def test_active_count_latency_is_recorded_even_when_the_count_fails(self):
        class _CountingDocStatus:
            def __init__(self, error=None):
                self.error = error

            async def count_docs_by_statuses(self, statuses, *, strict=True):
                assert set(statuses) == {
                    DocStatus.PENDING,
                    DocStatus.PARSING,
                    DocStatus.ANALYZING,
                    DocStatus.PROCESSING,
                }
                if self.error:
                    raise self.error
                return 3

        assert await count_active_documents(_CountingDocStatus()) == 3
        with pytest.raises(StorageControlPlaneError):
            await count_active_documents(
                _CountingDocStatus(StorageControlPlaneError("index down"))
            )

        summary = pipeline_metrics.snapshot()["durations"][
            pipeline_metrics.ACTIVE_COUNT_SECONDS
        ]
        assert summary["count"] == 2
