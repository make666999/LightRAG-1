"""Scan lifecycle tests: classification exits, bounded scan job records, manual retry ordering, and unsafe source rejection."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

from lightrag import LightRAG  # noqa: E402
from lightrag.base import (  # noqa: E402
    DocProcessingStatus,
    DocSchedulingRecord,
    DocStatus,
    SourceAbsent,
    SourceConflict,
    SourceUnique,
)
from lightrag.constants import PARSED_DIR_NAME  # noqa: E402
from lightrag.exceptions import StorageCapabilityError  # noqa: E402
from lightrag.kg.pipeline_ingress import (  # noqa: E402
    ManualRetryPublishResult,
    PipelineIngressMessage,
)
from lightrag.kg.scan_job_store import (  # noqa: E402
    SCAN_JOB_SAMPLE_LIMIT,
    AsyncioScanJobStore,
    ScanJobStatus,
)
from lightrag.kg.shared_storage import (  # noqa: E402
    MANUAL_PHASE_IDLE,
    acquire_reservation,
    get_namespace_data,
    get_namespace_lock,
    get_pipeline_ingress,
    get_scan_job_store,
    initialize_pipeline_status,
)
from lightrag.utils import EmbeddingFunc, Tokenizer, compute_mdhash_id  # noqa: E402

DocumentManager = _document_routes.DocumentManager
create_document_routes = _document_routes.create_document_routes
run_scanning_process = _document_routes.run_scanning_process
classify_scan_file = _document_routes.classify_scan_file
pipeline_enqueue_file = _document_routes.pipeline_enqueue_file
_ScanFileClass = _document_routes._ScanFileClass

pytestmark = pytest.mark.offline

UNSAFE_NAME = "scan\x0bsource.txt"


@pytest.fixture(autouse=True)
def _ensure_shared_storage_initialized():
    importlib.import_module("lightrag.kg.shared_storage").initialize_share_data()
    yield


def _endpoint(router, name):
    return [
        route.endpoint for route in router.routes if getattr(route, "name", "") == name
    ][-1]


# --------------------------------------------------------------------------- #
# Scan classification exits (§8.3)
# --------------------------------------------------------------------------- #


class _ResolverDocStatus:
    """doc_status double driven by an explicit ``{canonical_key: resolution}`` map."""

    def __init__(
        self,
        resolutions: dict,
        rows: dict | None = None,
        *,
        delete_error: Exception | None = None,
    ):
        self.resolutions = resolutions
        self.rows = rows or {}
        self.deleted_ids: list[str] = []
        self.hydrated: list[str] = []
        self.delete_error = delete_error

    async def resolve_doc_source_strict(self, canonical_source_key):
        return self.resolutions.get(canonical_source_key, SourceAbsent())

    async def get_full_docs_by_ids(self, doc_ids, *, strict=False):
        self.hydrated.extend(doc_ids)
        return {doc_id: self.rows[doc_id] for doc_id in doc_ids if doc_id in self.rows}

    async def delete(self, ids):
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted_ids.extend(ids)


class _StrictFullDocs:
    supports_strict_point_reads = True

    def __init__(self, docs_by_id, *, raise_on_strict=False):
        self.docs_by_id = docs_by_id
        self.raise_on_strict = raise_on_strict

    async def get_by_id(self, doc_id):
        return self.docs_by_id.get(doc_id)

    async def get_by_id_strict(self, doc_id):
        if self.raise_on_strict:
            raise RuntimeError("strict point read boom")
        return self.docs_by_id.get(doc_id)


class _WeakFullDocs:
    """A backend WITHOUT strict point reads (the base-class default)."""

    supports_strict_point_reads = False

    def __init__(self, docs_by_id):
        self.docs_by_id = docs_by_id

    async def get_by_id(self, doc_id):
        return self.docs_by_id.get(doc_id)

    async def get_by_id_strict(self, doc_id):
        raise StorageCapabilityError("no strict point reads")


class _ClassifyRag:
    def __init__(self, resolutions, rows=None, full_docs=None, delete_error=None):
        self.workspace = f"scanexit-{uuid4().hex[:8]}"
        self.doc_status = _ResolverDocStatus(
            resolutions, rows, delete_error=delete_error
        )
        self.full_docs = full_docs if full_docs is not None else _StrictFullDocs({})
        self.process_calls = 0

    async def arollback_failed_custom_chunk_patches(self, **_kwargs):
        return {"rolled_back": [], "failed": []}

    async def apipeline_process_enqueue_documents(self):
        self.process_calls += 1


def _record(doc_id: str, status: DocStatus) -> DocSchedulingRecord:
    return DocSchedulingRecord(
        id=doc_id,
        status=status,
        created_at="",
        updated_at="",
        file_path=Path(doc_id).name,
        track_id=None,
        has_custom_chunk_journal=False,
    )


def _row(doc_id: str, status: DocStatus, metadata: dict) -> DocProcessingStatus:
    return DocProcessingStatus(
        content_summary="",
        content_length=0,
        file_path=Path(doc_id).name,
        status=status,
        created_at="",
        updated_at="",
        metadata=metadata,
    )


def _unique(doc_id: str, status: DocStatus) -> SourceUnique:
    return SourceUnique(doc_id=doc_id, doc=_record(doc_id, status))


def _scan_rig(tmp_path, monkeypatch, rag):
    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(scan_enqueue_batch_size=8)
    )
    doc_manager = DocumentManager(str(tmp_path))
    batched: list[Path] = []

    async def _capture_batch(_rag, candidates, _track_id):
        batched.extend(c.path for c in candidates)
        return len(candidates)

    monkeypatch.setattr(_document_routes, "pipeline_enqueue_scan_batch", _capture_batch)
    return doc_manager, batched


def test_absent_is_claimed_new():
    async def _run():
        rag = _ClassifyRag({})
        decision = await classify_scan_file(rag, Path("fresh.txt"), "fresh.txt")
        assert decision.kind is _ScanFileClass.CLAIMED_NEW
        assert rag.doc_status.hydrated == []

    asyncio.run(_run())


def test_deleting_keeps_source_and_alias_without_resuming(tmp_path, monkeypatch):
    async def _run():
        rag = _ClassifyRag({"report.txt": _unique("doc-x", DocStatus.DELETING)})
        doc_manager, batched = _scan_rig(tmp_path, monkeypatch, rag)
        source = doc_manager.input_dir / "report.txt"
        alias = doc_manager.input_dir / "report.[native].txt"
        source.write_text("body", encoding="utf-8")
        alias.write_text("body", encoding="utf-8")
        warnings = []

        async def capture_warning(_rag, message):
            warnings.append(message)

        monkeypatch.setattr(_document_routes, "record_scan_warning", capture_warning)
        await run_scanning_process(rag, doc_manager, "track-deleting")
        assert batched == []
        assert rag.doc_status.deleted_ids == []
        assert source.exists() and alias.exists()
        assert len(warnings) == 2
        assert all("re-run deletion first" in message for message in warnings)
        assert rag.doc_status.hydrated == []

    asyncio.run(_run())


def test_conflict_reports_bounded_candidate_sample():
    async def _run():
        rag = _ClassifyRag(
            {
                "dup.txt": SourceConflict(
                    candidate_count=3, sample_doc_ids=("doc-a", "doc-b")
                )
            }
        )
        decision = await classify_scan_file(rag, Path("dup.txt"), "dup.txt")
        assert decision.kind is _ScanFileClass.SOURCE_CONFLICT
        assert "doc-a, doc-b" in decision.detail
        assert "3 primary documents" in decision.detail
        assert rag.doc_status.hydrated == []

    asyncio.run(_run())


def test_processed_wins_over_alias_comparison():
    async def _run():
        rag = _ClassifyRag(
            {"done.txt": _unique("doc-done", DocStatus.PROCESSED)},
            rows={
                "doc-done": _row(
                    "doc-done", DocStatus.PROCESSED, {"source_file": "done.txt"}
                )
            },
        )
        decision = await classify_scan_file(rag, Path("done.[native].txt"), "done.txt")
        assert decision.kind is _ScanFileClass.PROCESSED
        assert decision.doc_id == "doc-done"

    asyncio.run(_run())


def test_failed_without_confirmed_content_is_a_stale_stub():
    async def _run():
        rag = _ClassifyRag(
            {"broken.txt": _unique("doc-broken", DocStatus.FAILED)},
            rows={
                "doc-broken": _row(
                    "doc-broken", DocStatus.FAILED, {"source_file": "broken.txt"}
                )
            },
            full_docs=_StrictFullDocs({}),
        )
        decision = await classify_scan_file(rag, Path("broken.txt"), "broken.txt")
        assert decision.kind is _ScanFileClass.STALE_STUB
        assert decision.doc_id == "doc-broken"

    asyncio.run(_run())


@pytest.mark.parametrize(
    "full_docs",
    [
        _WeakFullDocs({}),
        _StrictFullDocs({}, raise_on_strict=True),
    ],
    ids=["no_strict_capability", "strict_read_failed"],
)
def test_unconfirmed_absence_never_deletes_the_stub(full_docs):
    async def _run():
        rag = _ClassifyRag(
            {"broken.txt": _unique("doc-broken", DocStatus.FAILED)},
            rows={
                "doc-broken": _row(
                    "doc-broken", DocStatus.FAILED, {"source_file": "broken.txt"}
                )
            },
            full_docs=full_docs,
        )
        decision = await classify_scan_file(rag, Path("broken.txt"), "broken.txt")
        assert decision.kind is _ScanFileClass.RESUME_SAME_PHYSICAL_SOURCE
        assert rag.doc_status.deleted_ids == []

    asyncio.run(_run())


def test_missing_source_file_is_identity_unknown_not_an_alias():
    async def _run():
        rag = _ClassifyRag(
            {"legacy.txt": _unique("custom-id", DocStatus.PENDING)},
            rows={"custom-id": _row("custom-id", DocStatus.PENDING, {})},
        )
        decision = await classify_scan_file(rag, Path("legacy.txt"), "legacy.txt")
        assert decision.kind is _ScanFileClass.SOURCE_IDENTITY_UNKNOWN
        assert "records no source file" in decision.detail

    asyncio.run(_run())


def test_same_physical_source_resumes_and_a_different_one_is_an_alias():
    async def _run():
        rows = {
            "doc-x": _row(
                "doc-x", DocStatus.PARSING, {"source_file": "report.[native].txt"}
            )
        }
        rag = _ClassifyRag(
            {"report.txt": _unique("doc-x", DocStatus.PARSING)}, rows=rows
        )

        same = await classify_scan_file(rag, Path("report.[native].txt"), "report.txt")
        assert same.kind is _ScanFileClass.RESUME_SAME_PHYSICAL_SOURCE

        alias = await classify_scan_file(rag, Path("report.txt"), "report.txt")
        assert alias.kind is _ScanFileClass.ALIAS_DUPLICATE
        assert "report.[native].txt" in alias.detail

    asyncio.run(_run())


def test_source_conflict_keeps_the_file_and_never_enqueues(tmp_path, monkeypatch):
    async def _run():
        rag = _ClassifyRag(
            {
                "shared.txt": SourceConflict(
                    candidate_count=None, sample_doc_ids=("doc-1", "doc-2")
                )
            }
        )
        doc_manager, batched = _scan_rig(tmp_path, monkeypatch, rag)
        conflicted = doc_manager.input_dir / "shared.txt"
        conflicted.write_text("body", encoding="utf-8")

        await run_scanning_process(rag, doc_manager, "track-conflict")

        assert batched == []
        assert rag.doc_status.deleted_ids == []
        assert conflicted.exists()
        assert not (doc_manager.input_dir / PARSED_DIR_NAME).exists()

    asyncio.run(_run())


def test_alias_duplicate_is_archived_and_the_row_is_untouched(tmp_path, monkeypatch):
    async def _run():
        rag = _ClassifyRag(
            {"report.txt": _unique("doc-x", DocStatus.PENDING)},
            rows={
                "doc-x": _row(
                    "doc-x", DocStatus.PENDING, {"source_file": "report.[native].txt"}
                )
            },
        )
        doc_manager, batched = _scan_rig(tmp_path, monkeypatch, rag)
        alias = doc_manager.input_dir / "report.txt"
        alias.write_text("alias body", encoding="utf-8")

        await run_scanning_process(rag, doc_manager, "track-alias")

        assert batched == []
        assert rag.doc_status.deleted_ids == []
        assert not alias.exists()
        assert (doc_manager.input_dir / PARSED_DIR_NAME / "report.txt").exists()

    asyncio.run(_run())


def test_stale_stub_is_deleted_then_enqueued_as_new(tmp_path, monkeypatch):
    async def _run():
        rag = _ClassifyRag(
            {"fixed.txt": _unique("doc-stub", DocStatus.FAILED)},
            rows={
                "doc-stub": _row(
                    "doc-stub", DocStatus.FAILED, {"source_file": "fixed.txt"}
                )
            },
            full_docs=_StrictFullDocs({}),
        )
        doc_manager, batched = _scan_rig(tmp_path, monkeypatch, rag)
        fixed = doc_manager.input_dir / "fixed.txt"
        fixed.write_text("fixed body", encoding="utf-8")

        await run_scanning_process(rag, doc_manager, "track-stub")

        assert rag.doc_status.deleted_ids == ["doc-stub"]
        assert batched == [fixed]
        assert fixed.exists()

    asyncio.run(_run())


def test_failed_stub_delete_is_counted_as_an_error_not_a_resume(tmp_path, monkeypatch):
    async def _run():
        rag = _ClassifyRag(
            {"fixed.txt": _unique("doc-stub", DocStatus.FAILED)},
            rows={
                "doc-stub": _row(
                    "doc-stub", DocStatus.FAILED, {"source_file": "fixed.txt"}
                )
            },
            full_docs=_StrictFullDocs({}),
            delete_error=RuntimeError("doc_status delete boom"),
        )
        doc_manager, batched = _scan_rig(tmp_path, monkeypatch, rag)
        fixed = doc_manager.input_dir / "fixed.txt"
        fixed.write_text("fixed body", encoding="utf-8")

        reported: dict = {}

        class _Reporter:
            enabled = False

            def count(self, key, amount=1):
                reported[key] = reported.get(key, 0) + amount

            def sample(self, *_a, **_k):
                return None

            def flush(self):
                return None

            def renew(self):
                return None

            def finish(self, *_a, **_k):
                return None

        monkeypatch.setattr(
            _document_routes, "_ScanJobReporter", lambda *_a: _Reporter()
        )

        await run_scanning_process(rag, doc_manager, "track-stub-fail")

        assert batched == []
        assert rag.doc_status.deleted_ids == []
        assert fixed.exists()
        assert reported.get("errors") == 1
        assert _ScanFileClass.RESUME_SAME_PHYSICAL_SOURCE.value not in reported
        assert _ScanFileClass.STALE_STUB.value not in reported

    asyncio.run(_run())


def test_identity_unknown_keeps_both_the_file_and_the_row(tmp_path, monkeypatch):
    async def _run():
        rag = _ClassifyRag(
            {"legacy.txt": _unique("custom-id", DocStatus.PENDING)},
            rows={"custom-id": _row("custom-id", DocStatus.PENDING, {})},
        )
        doc_manager, batched = _scan_rig(tmp_path, monkeypatch, rag)
        legacy = doc_manager.input_dir / "legacy.txt"
        legacy.write_text("legacy body", encoding="utf-8")

        await run_scanning_process(rag, doc_manager, "track-unknown")

        assert batched == []
        assert rag.doc_status.deleted_ids == []
        assert legacy.exists()
        assert not (doc_manager.input_dir / PARSED_DIR_NAME).exists()
        assert rag.process_calls == 1

    asyncio.run(_run())


# --------------------------------------------------------------------------- #
# Bounded scan job record wired into /scan (§8.6)
# --------------------------------------------------------------------------- #


class _StubScanJobRag:
    """Minimal rag surface the scan path touches with an EMPTY input dir."""

    def __init__(self):
        self.workspace = f"scanjob-{uuid4().hex[:8]}"
        self.process_calls = 0
        self.reset_calls = []
        self.drive_gate: asyncio.Event | None = None

    async def arollback_failed_custom_chunk_patches(self, **kwargs):
        return {"rolled_back": [], "failed": []}

    async def apipeline_reset_failed_for_scan(
        self, request_id, *, scan_owner_token=None
    ):
        self.reset_calls.append((request_id, scan_owner_token))
        return True

    async def apipeline_process_enqueue_documents(self):
        self.process_calls += 1
        if self.drive_gate is not None:
            await self.drive_gate.wait()


async def _prepare_job_router(tmp_path):
    rag = _StubScanJobRag()
    await initialize_pipeline_status(workspace=rag.workspace)
    doc_manager = DocumentManager(str(tmp_path))
    router = create_document_routes(rag, doc_manager)
    return rag, router


async def _await_managed(managed: set):
    for task in list(managed):
        try:
            await task
        except asyncio.CancelledError:
            pass


def test_job_record_is_queryable_before_the_scan_finishes(tmp_path):
    async def _run():
        rag, router = await _prepare_job_router(tmp_path)
        scan = _endpoint(router, "scan_for_new_documents")
        status = _endpoint(router, "get_scan_job_status")

        rag.drive_gate = asyncio.Event()
        managed: set = set()
        response = await scan(managed)
        assert response.status == "scanning_started"

        running = await status(response.track_id)
        assert running.track_id == response.track_id
        assert running.status == "running"
        assert running.counts.get("discovered") == 0

        rag.drive_gate.set()
        await _await_managed(managed)

        done = await status(response.track_id)
        assert done.status == "completed"
        assert "0 discovered" in done.message
        assert set(done.samples) == {"processed", "warning", "error"}

    asyncio.run(_run())


def test_capacity_exceeded_refuses_with_429_and_no_side_effects(tmp_path):
    async def _run():
        rag, router = await _prepare_job_router(tmp_path)
        store = get_scan_job_store(rag.workspace)
        for index in range(200):
            store.create(f"filler-{index}", uuid4().hex)

        scan = _endpoint(router, "scan_for_new_documents")
        managed: set = set()
        with pytest.raises(_document_routes.HTTPException) as excinfo:
            await scan(managed)
        assert excinfo.value.status_code == 429

        assert managed == set()
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
        assert pipeline_status["scanning"] is False
        assert pipeline_status["scanning_exclusive"] is False
        assert pipeline_status["scanning_owner"] is None
        ingress = await get_pipeline_ingress(rag.workspace)
        assert ingress.snapshot_manual_retries() == []
        assert rag.reset_calls == []

    asyncio.run(_run())


def test_precommit_failure_cancels_the_job_and_releases_the_reservation(tmp_path):
    async def _run():
        rag, router = await _prepare_job_router(tmp_path)
        ingress = await get_pipeline_ingress(rag.workspace)

        def _publish_boom(*args, **kwargs):
            raise RuntimeError("mailbox publish boom")

        ingress.request_manual_retry = _publish_boom

        scan = _endpoint(router, "scan_for_new_documents")
        managed: set = set()
        with pytest.raises(RuntimeError, match="failed to start"):
            await scan(managed)

        store = get_scan_job_store(rag.workspace)
        records = store.snapshot()
        assert [record["status"] for record in records] == ["cancelled"]
        assert "aborted before" in records[0]["message"]

        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
        assert pipeline_status["scanning"] is False
        assert pipeline_status["scanning_owner"] is None
        assert rag.reset_calls == []

    asyncio.run(_run())


def test_classification_failure_marks_the_job_failed_with_an_error_sample(tmp_path):
    async def _run():
        rag = _StubScanJobRag()
        await initialize_pipeline_status(workspace=rag.workspace)
        store = get_scan_job_store(rag.workspace)
        track_id, job_token = "scan-fail", uuid4().hex
        assert store.create(track_id, job_token).record is not None

        doc_manager = DocumentManager(str(tmp_path))

        def _boom():
            raise RuntimeError("classification boom")

        doc_manager.iter_new_files = _boom
        await run_scanning_process(
            rag,
            doc_manager,
            track_id,
            manual_request_id=uuid4().hex,
            job_owner_token=job_token,
        )

        record = store.get(track_id)
        assert record["status"] == "failed"
        assert "classification boom" in record["message"]
        assert any(
            "classification boom" in item
            for item in record["samples"]["error"]["items"]
        )

    asyncio.run(_run())


def test_cancelled_scan_marks_the_job_cancelled(tmp_path):
    async def _run():
        rag = _StubScanJobRag()
        await initialize_pipeline_status(workspace=rag.workspace)
        store = get_scan_job_store(rag.workspace)
        track_id, job_token = "scan-cancel", uuid4().hex
        store.create(track_id, job_token)

        doc_manager = DocumentManager(str(tmp_path))

        def _cancelled():
            raise asyncio.CancelledError()

        doc_manager.iter_new_files = _cancelled
        with pytest.raises(asyncio.CancelledError):
            await run_scanning_process(
                rag,
                doc_manager,
                track_id,
                manual_request_id=uuid4().hex,
                job_owner_token=job_token,
            )

        assert store.get(track_id)["status"] == "cancelled"

    asyncio.run(_run())


def test_status_endpoint_404s_for_an_unknown_track_id(tmp_path):
    async def _run():
        _rag, router = await _prepare_job_router(tmp_path)
        status = _endpoint(router, "get_scan_job_status")
        with pytest.raises(_document_routes.HTTPException) as excinfo:
            await status("scan-does-not-exist")
        assert excinfo.value.status_code == 404

    asyncio.run(_run())


def test_a_long_phase_does_not_lose_the_lease_under_a_live_owner(tmp_path, monkeypatch):
    async def _run():
        rag = _StubScanJobRag()
        await initialize_pipeline_status(workspace=rag.workspace)
        store = AsyncioScanJobStore(rag.workspace, lease_seconds=0.15)
        monkeypatch.setattr(
            _document_routes, "_resolve_scan_job_store", lambda _rag: store
        )
        monkeypatch.setattr(_document_routes, "_SCAN_JOB_RENEW_SECONDS", 0.025)

        track_id, job_token = "scan-slow", uuid4().hex
        store.create(track_id, job_token)
        doc_manager = DocumentManager(str(tmp_path))

        rag.drive_gate = asyncio.Event()
        scan = asyncio.create_task(
            run_scanning_process(
                rag,
                doc_manager,
                track_id,
                manual_request_id=uuid4().hex,
                job_owner_token=job_token,
            )
        )
        await asyncio.sleep(0.45)
        assert rag.process_calls == 1
        assert store.get(track_id)["status"] == "running"

        rag.drive_gate.set()
        await scan
        assert store.get(track_id)["status"] == "completed"

    asyncio.run(_run())


def test_the_lease_heartbeat_never_outlives_its_scan(monkeypatch):
    async def _run():
        monkeypatch.setattr(_document_routes, "_SCAN_JOB_RENEW_SECONDS", 0.01)
        store = AsyncioScanJobStore("heartbeat-test", lease_seconds=0.05)
        store.create("job", "owner")
        reporter = _document_routes._ScanJobReporter(store, "job", "owner")

        async def _scan_that_already_ended():
            return None

        scan_task = asyncio.create_task(_scan_that_already_ended())
        await scan_task

        heartbeat = asyncio.create_task(
            _document_routes._renew_scan_job_lease(reporter, scan_task)
        )
        await asyncio.wait_for(heartbeat, timeout=1.0)

        await asyncio.sleep(0.08)
        assert store.get("job")["status"] == "abandoned"

    asyncio.run(_run())


def test_reporter_bounds_sample_calls_and_resyncs_the_cas_version():
    store = AsyncioScanJobStore("reporter-test")
    token = uuid4().hex
    store.create("job", token)
    reporter = _document_routes._ScanJobReporter(store, "job", token)

    for index in range(SCAN_JOB_SAMPLE_LIMIT + 5):
        reporter.sample("warning", f"warning {index}")
    reporter.count("discovered", 3)
    reporter.flush()

    record = store.get("job")
    assert len(record["samples"]["warning"]["items"]) == SCAN_JOB_SAMPLE_LIMIT
    assert record["samples"]["warning"]["dropped"] == 0
    assert record["counts"]["warning_samples_suppressed"] == 5
    assert record["counts"]["discovered"] == 3

    store.update(
        "job", token, count_deltas={"other": 1}, expected_version=record["version"]
    )
    reporter.count("discovered", 1)
    reporter.flush()
    assert store.get("job")["counts"]["discovered"] == 4

    reporter.finish(ScanJobStatus.COMPLETED, "done")
    assert store.get("job")["status"] == "completed"
    assert reporter.enabled is False
    reporter.count("discovered", 99)
    reporter.flush()
    assert store.get("job")["counts"]["discovered"] == 4


# --------------------------------------------------------------------------- #
# Scan-driven manual retry intent (§8.1)
# --------------------------------------------------------------------------- #


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


async def _dummy_llm(*args, **kwargs) -> str:
    return "ok"


def _chunking(tokenizer, content, *args) -> list[dict]:
    return [{"tokens": 1, "content": f"{content}::chunk1", "chunk_order_index": 0}]


class _FlippableExtract:
    def __init__(self):
        self.fail = True

    async def __call__(self, chunks, *args, **kwargs):
        if self.fail:
            raise RuntimeError("extract fail sentinel")
        return [({}, {}) for _ in chunks]


class _SelectiveExtract:
    """Fail extraction for chunks whose content carries a marked substring."""

    def __init__(self, *markers: str):
        self.markers = set(markers)
        self.attempts: list[str] = []

    async def __call__(self, chunks, *args, **kwargs):
        items = chunks.values() if isinstance(chunks, dict) else chunks
        contents = [(chunk or {}).get("content", "") for chunk in items]
        self.attempts.extend(contents)
        if any(marker in content for content in contents for marker in self.markers):
            raise RuntimeError("extract fail sentinel")
        return [({}, {}) for _ in contents]

    def attempts_for(self, marker: str) -> int:
        return sum(1 for content in self.attempts if marker in content)


async def _build_rag(tmp_path, extract) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"scanmr-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        chunking_func=_chunking,
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    rag._process_extract_entities = extract
    return rag


async def _make_failed_doc(rag, extract) -> str:
    extract.fail = True
    await rag.apipeline_enqueue_documents(input="doc body", file_paths="a.txt")
    await rag.apipeline_process_enqueue_documents()
    doc_id = compute_mdhash_id("a.txt", prefix="doc-")
    row = await rag.doc_status.get_by_id(doc_id)
    assert row["status"] == DocStatus.FAILED
    return doc_id


async def _publish_manual(rag) -> str:
    ingress = await get_pipeline_ingress(rag.workspace)
    request_id = uuid4().hex
    ingress.request_manual_retry(
        request_id,
        PipelineIngressMessage(kind="rescan", retry_failed=True, request_id=request_id),
    )
    return request_id


def test_scan_serves_manual_request_and_retries_failed(tmp_path):
    async def _run():
        extract = _FlippableExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            doc_id = await _make_failed_doc(rag, extract)
            extract.fail = False

            request_id = await _publish_manual(rag)
            doc_manager = DocumentManager(str(tmp_path / "inputs"))
            await run_scanning_process(rag, doc_manager, manual_request_id=request_id)

            row = await rag.doc_status.get_by_id(doc_id)
            assert row["status"] == DocStatus.PROCESSED
            ingress = await get_pipeline_ingress(rag.workspace)
            assert ingress.snapshot_manual_retries() == []
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_classification_failure_fallback_drives_sticky_request(tmp_path):
    async def _run():
        extract = _FlippableExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            doc_id = await _make_failed_doc(rag, extract)
            extract.fail = False

            request_id = await _publish_manual(rag)
            doc_manager = DocumentManager(str(tmp_path / "inputs"))

            def boom():
                raise RuntimeError("classification boom")

            doc_manager.iter_new_files = boom
            await run_scanning_process(rag, doc_manager, manual_request_id=request_id)

            row = await rag.doc_status.get_by_id(doc_id)
            assert row["status"] == DocStatus.PROCESSED
            ingress = await get_pipeline_ingress(rag.workspace)
            assert ingress.snapshot_manual_retries() == []
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_new_files_all_fail_to_enqueue_still_drives_sticky_request(
    tmp_path, monkeypatch
):
    async def _run():
        extract = _FlippableExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            doc_id = await _make_failed_doc(rag, extract)
            extract.fail = False

            request_id = await _publish_manual(rag)
            doc_manager = DocumentManager(str(tmp_path / "inputs"))
            doc_manager.iter_new_files = lambda: [
                Path(str(tmp_path / "inputs" / "brand_new.txt"))
            ]

            async def _enqueue_fails(rag, file_path, track_id=None, from_scan=False):
                return (False, None)

            monkeypatch.setattr(
                _document_routes, "pipeline_enqueue_file", _enqueue_fails
            )

            await run_scanning_process(rag, doc_manager, manual_request_id=request_id)

            row = await rag.doc_status.get_by_id(doc_id)
            assert row["status"] == DocStatus.PROCESSED
            ingress = await get_pipeline_ingress(rag.workspace)
            assert ingress.snapshot_manual_retries() == []
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_cancelled_scan_does_not_drive_processing_after_reset(tmp_path):
    async def _run():
        extract = _FlippableExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            doc_id = await _make_failed_doc(rag, extract)
            extract.fail = False

            request_id = await _publish_manual(rag)
            doc_manager = DocumentManager(str(tmp_path / "inputs"))

            def cancelled():
                raise asyncio.CancelledError()

            doc_manager.iter_new_files = cancelled
            with pytest.raises(asyncio.CancelledError):
                await run_scanning_process(
                    rag, doc_manager, manual_request_id=request_id
                )

            row = await rag.doc_status.get_by_id(doc_id)
            assert row["status"] == DocStatus.PENDING
            ingress = await get_pipeline_ingress(rag.workspace)
            assert ingress.snapshot_manual_retries() == []
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_failed_reset_precedes_new_file_enqueue(tmp_path, monkeypatch):
    async def _run():
        extract = _SelectiveExtract("OLDDOC", "BOOM")
        rag = await _build_rag(tmp_path, extract)
        try:
            await rag.apipeline_enqueue_documents(
                input="OLDDOC body", file_paths="old.txt"
            )
            await rag.apipeline_process_enqueue_documents()
            old_id = compute_mdhash_id("old.txt", prefix="doc-")
            assert (await rag.doc_status.get_by_id(old_id))[
                "status"
            ] == DocStatus.FAILED
            extract.markers.discard("OLDDOC")

            input_dir = tmp_path / "inputs"
            input_dir.mkdir(parents=True, exist_ok=True)
            monkeypatch.setenv("INPUT_DIR", str(input_dir))
            (input_dir / "new.txt").write_text("BOOM payload", encoding="utf-8")

            request_id = await _publish_manual(rag)
            doc_manager = DocumentManager(str(input_dir))
            await run_scanning_process(rag, doc_manager, manual_request_id=request_id)

            assert (await rag.doc_status.get_by_id(old_id))[
                "status"
            ] == DocStatus.PROCESSED
            failed = await rag.doc_status.get_docs_by_statuses([DocStatus.FAILED])
            assert [doc.file_path for doc in failed.values()] == ["new.txt"]
            assert extract.attempts_for("BOOM") == 1
            assert extract.attempts_for("OLDDOC") == 2
            ingress = await get_pipeline_ingress(rag.workspace)
            assert ingress.snapshot_manual_retries() == []
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_abandoned_reset_skips_discovery_and_leaves_request_sticky(
    tmp_path, monkeypatch
):
    async def _run():
        extract = _FlippableExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            doc_id = await _make_failed_doc(rag, extract)
            extract.fail = False

            input_dir = tmp_path / "inputs"
            input_dir.mkdir(parents=True, exist_ok=True)
            monkeypatch.setenv("INPUT_DIR", str(input_dir))
            (input_dir / "new.txt").write_text("fresh body", encoding="utf-8")

            request_id = await _publish_manual(rag)

            async def _reset_abandoned(request, *, scan_owner_token=None):
                return False

            monkeypatch.setattr(
                rag, "apipeline_reset_failed_for_scan", _reset_abandoned
            )

            doc_manager = DocumentManager(str(input_dir))
            await run_scanning_process(rag, doc_manager, manual_request_id=request_id)

            rows = await rag.doc_status.get_docs_by_statuses(list(DocStatus))
            assert all("new.txt" not in (doc.file_path or "") for doc in rows.values())
            assert (input_dir / "new.txt").exists()
            assert (await rag.doc_status.get_by_id(doc_id))[
                "status"
            ] == DocStatus.PROCESSED
            ingress = await get_pipeline_ingress(rag.workspace)
            assert ingress.snapshot_manual_retries() == []
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_scan_reset_takes_busy_under_its_own_exclusive_fence(tmp_path):
    async def _run():
        extract = _FlippableExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            doc_id = await _make_failed_doc(rag, extract)
            extract.fail = False
            request_id = await _publish_manual(rag)

            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            pipeline_status_lock = get_namespace_lock(
                "pipeline_status", workspace=rag.workspace
            )
            scanning_token = uuid4().hex
            reservation = await acquire_reservation(
                pipeline_status,
                pipeline_status_lock,
                owner_key="scanning_owner",
                owner=scanning_token,
                owner_kind="scan",
                flags={"scanning": True, "scanning_exclusive": True},
                reject_when=(),
            )
            assert reservation.acquired

            ingress = await get_pipeline_ingress(rag.workspace)
            assert (
                await rag.apipeline_reset_failed_for_scan(
                    request_id, scan_owner_token=uuid4().hex
                )
                is False
            )
            assert (await rag.doc_status.get_by_id(doc_id))[
                "status"
            ] == DocStatus.FAILED
            assert [m.request_id for m in ingress.snapshot_manual_retries()] == [
                request_id
            ]

            assert (
                await rag.apipeline_reset_failed_for_scan(
                    request_id, scan_owner_token=scanning_token
                )
                is True
            )
            assert (await rag.doc_status.get_by_id(doc_id))[
                "status"
            ] == DocStatus.PENDING
            assert ingress.snapshot_manual_retries() == []
            assert pipeline_status["busy"] is False
            assert pipeline_status["busy_owner"] is None
            assert pipeline_status["manual_freeze_requested"] is False
            assert pipeline_status["manual_resetting"] is False
            assert pipeline_status["manual_owner"] is None
            assert pipeline_status["manual_phase"] == MANUAL_PHASE_IDLE
            assert pipeline_status["scanning"] is True
            assert pipeline_status["scanning_exclusive"] is True
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_scan_endpoint_refused_while_earlier_manual_request_queued(tmp_path):
    async def _run():
        extract = _FlippableExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            await initialize_pipeline_status(workspace=rag.workspace)
            request_id = await _publish_manual(rag)

            doc_manager = DocumentManager(str(tmp_path / "inputs"))
            router = create_document_routes(rag, doc_manager)
            scan_endpoint = _endpoint(router, "scan_for_new_documents")

            response = await scan_endpoint(set())
            assert response.status == "scanning_skipped_pipeline_busy"
            assert request_id[:8] in response.message

            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            assert pipeline_status["scanning"] is False
            assert pipeline_status["scanning_exclusive"] is False
            assert pipeline_status["scanning_owner"] is None
            ingress = await get_pipeline_ingress(rag.workspace)
            assert [m.request_id for m in ingress.snapshot_manual_retries()] == [
                request_id
            ]
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_scan_refuses_when_the_manual_channel_is_full(tmp_path, monkeypatch):
    async def _run():
        extract = _FlippableExtract()
        rag = await _build_rag(tmp_path, extract)
        try:
            await initialize_pipeline_status(workspace=rag.workspace)
            ingress = await get_pipeline_ingress(rag.workspace)
            monkeypatch.setattr(
                ingress,
                "request_manual_retry",
                lambda *args, **kwargs: ManualRetryPublishResult.CAPACITY_EXCEEDED,
            )

            doc_manager = DocumentManager(str(tmp_path / "inputs"))
            router = create_document_routes(rag, doc_manager)
            scan_endpoint = _endpoint(router, "scan_for_new_documents")

            with pytest.raises(_document_routes.HTTPException) as excinfo:
                await scan_endpoint(set())

            assert excinfo.value.status_code == 429
            assert excinfo.value.headers["Retry-After"]

            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            assert pipeline_status["scanning"] is False
            assert pipeline_status["scanning_exclusive"] is False
            assert pipeline_status["busy"] is False
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


# --------------------------------------------------------------------------- #
# Unsafe scanned basename rejection (§8.3.0 / GHSA-c922-pw4m-4wcv)
# --------------------------------------------------------------------------- #


class _NoRowsDocStatus:
    def __init__(self):
        self.lookups: list[str] = []

    async def resolve_doc_source_strict(self, canonical_source_key):
        self.lookups.append(canonical_source_key)
        return SourceAbsent()

    async def get_full_docs_by_ids(self, doc_ids, *, strict=False):
        return {}


class _UnsafeScanRag:
    def __init__(self):
        self.workspace = f"unsafe-source-{uuid4().hex[:8]}"
        self.doc_status = _NoRowsDocStatus()
        self.full_docs = SimpleNamespace(
            get_by_id=self._missing,
            get_by_id_strict=self._missing,
            supports_strict_point_reads=True,
        )

    async def _missing(self, _doc_id):
        return None

    async def arollback_failed_custom_chunk_patches(self, **_kwargs):
        return {"rolled_back": [], "failed": []}

    async def apipeline_process_enqueue_documents(self):
        return None


async def _history(workspace) -> list[str]:
    status = await get_namespace_data("pipeline_status", workspace=workspace)
    return list(status.get("history_messages", []))


async def _scan_unsafe(
    tmp_path, monkeypatch, *, filenames
) -> tuple[_UnsafeScanRag, list[str]]:
    rag = _UnsafeScanRag()
    await initialize_pipeline_status(workspace=rag.workspace)
    doc_manager = DocumentManager(str(tmp_path))
    for name in filenames:
        try:
            (doc_manager.input_dir / name).write_text("body", encoding="utf-8")
        except (OSError, UnicodeEncodeError) as exc:
            if name != UNSAFE_NAME:
                raise
            pytest.skip(f"filesystem rejects the unsafe test file name: {exc}")

    enqueued: list[str] = []

    async def _capture_batch(_rag, candidates, _track_id):
        enqueued.extend(candidate.path.name for candidate in candidates)
        return len(candidates)

    monkeypatch.setattr(_document_routes, "pipeline_enqueue_scan_batch", _capture_batch)
    await run_scanning_process(rag, doc_manager, "track-unsafe")
    return rag, enqueued


def test_scan_refuses_an_unsafe_basename_and_still_ingests_its_neighbour(
    tmp_path, monkeypatch
):
    async def _run():
        _rag, enqueued = await _scan_unsafe(
            tmp_path, monkeypatch, filenames=[UNSAFE_NAME, "clean.txt"]
        )
        assert enqueued == ["clean.txt"]

    asyncio.run(_run())


def test_the_refused_file_stays_in_the_input_directory(tmp_path, monkeypatch):
    async def _run():
        await _scan_unsafe(tmp_path, monkeypatch, filenames=[UNSAFE_NAME])

        assert (tmp_path / UNSAFE_NAME).exists()
        parsed_dir = tmp_path / PARSED_DIR_NAME
        archived = (
            [path.name for path in parsed_dir.iterdir()] if parsed_dir.exists() else []
        )
        assert archived == []

    asyncio.run(_run())


def test_the_refusal_happens_before_any_identity_resolution(tmp_path, monkeypatch):
    async def _run():
        rag, _enqueued = await _scan_unsafe(
            tmp_path, monkeypatch, filenames=[UNSAFE_NAME, "clean.txt"]
        )
        assert rag.doc_status.lookups == ["clean.txt"]

    asyncio.run(_run())


def test_the_scan_warning_names_the_offending_code_point(tmp_path, monkeypatch):
    async def _run():
        rag, _enqueued = await _scan_unsafe(
            tmp_path, monkeypatch, filenames=[UNSAFE_NAME]
        )

        history = await _history(rag.workspace)
        matching = [message for message in history if "U+000B" in message]
        assert matching, history
        assert "unsafe document source" in matching[-1]
        assert UNSAFE_NAME not in matching[-1]
        assert "scan\\x0bsource.txt" in matching[-1]

    asyncio.run(_run())


def test_the_recorded_message_survives_json_rendering(tmp_path, monkeypatch):
    surrogate_name = "scan\udcffsource.txt"
    assert (
        _document_routes.find_unsafe_document_source_character(surrogate_name)
        == "\udcff"
    )
    rendered = _document_routes.describe_rejected_document_source(surrogate_name)
    payload = {"history_messages": [f"Skipping {rendered}"]}
    json.dumps(payload, ensure_ascii=False).encode("utf-8")

    with pytest.raises(UnicodeEncodeError):
        json.dumps({"raw": surrogate_name}, ensure_ascii=False).encode("utf-8")


def test_a_surrogate_basename_is_refused_end_to_end(tmp_path, monkeypatch):
    async def _run():
        rag = _UnsafeScanRag()
        await initialize_pipeline_status(workspace=rag.workspace)
        doc_manager = DocumentManager(str(tmp_path))
        try:
            (doc_manager.input_dir / "scan\udcffsource.txt").write_text("body")
        except (OSError, UnicodeEncodeError) as exc:
            pytest.skip(f"filesystem rejects non-UTF-8 filenames: {exc}")

        enqueued: list[str] = []

        async def _capture_batch(_rag, candidates, _track_id):
            enqueued.extend(candidate.path.name for candidate in candidates)
            return len(candidates)

        monkeypatch.setattr(
            _document_routes, "pipeline_enqueue_scan_batch", _capture_batch
        )
        await run_scanning_process(rag, doc_manager, "track-surrogate")

        assert enqueued == []
        history = await _history(rag.workspace)
        json.dumps(history, ensure_ascii=False).encode("utf-8")

    asyncio.run(_run())


def test_the_unsafe_exit_is_part_of_the_scan_taxonomy():
    assert _ScanFileClass.UNSAFE_SOURCE.value == "unsafe_source"


def test_pipeline_enqueue_file_refuses_an_unsafe_basename_without_a_doc_row(tmp_path):
    error_batches: list[list[dict]] = []

    class _EnqueueRag:
        async def apipeline_enqueue_documents(self, _content, **_kwargs):
            raise AssertionError("an unsafe source must not reach doc_status")

        async def apipeline_enqueue_error_documents(self, error_files, _track_id):
            error_batches.append(error_files)

    async def _run():
        unsafe_file = tmp_path / UNSAFE_NAME
        try:
            unsafe_file.write_text("body", encoding="utf-8")
        except (OSError, UnicodeEncodeError) as exc:
            pytest.skip(f"filesystem rejects the unsafe test file name: {exc}")

        success, track_id = await pipeline_enqueue_file(
            _EnqueueRag(), unsafe_file, "track-backstop"
        )

        assert success is False
        assert track_id == "track-backstop"
        assert error_batches == []
        assert unsafe_file.exists()

    asyncio.run(_run())


def test_a_safe_basename_still_reaches_the_enqueue(tmp_path):
    seen: list[str] = []

    class _EnqueueRag:
        addon_params: dict = {}

        async def apipeline_enqueue_documents(self, _content, **kwargs):
            seen.append(kwargs["file_paths"])
            return {"ok": True}

        async def apipeline_enqueue_error_documents(self, error_files, _track_id):
            raise AssertionError(f"unexpected error document: {error_files}")

    async def _run():
        clean_file = tmp_path / "clean.txt"
        clean_file.write_text("body", encoding="utf-8")

        success, _track_id = await pipeline_enqueue_file(
            _EnqueueRag(), clean_file, "track-clean"
        )
        assert success is True
        assert seen == [str(clean_file)]

    asyncio.run(_run())


def test_the_escaped_name_stays_recognisable():
    rendered = _document_routes.describe_rejected_document_source(UNSAFE_NAME)
    assert rendered.isascii()
    assert "scan" in rendered and "source.txt" in rendered
    assert Path(UNSAFE_NAME).suffix == ".txt"
