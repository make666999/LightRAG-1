"""Consolidated strict storage read, capability gate, and orchestrator read-failure tests.

Covers (PipelineConcurrencyContract — LR2 §5.4 / §7.2 / Phase 6):
  * ``PIPELINE_REQUIRE_STRICT_STORAGE_READS`` startup capability gate and
    operator-facing consequence warnings.
  * Batch consistency validator strict ``full_docs`` content probes: preserving
    unconfirmable best-effort misses, deleting confirmed absences, propagating
    strict read failures, and reading content once per document.
  * Orchestrator-level storage failures during batch enqueue: per-doc isolation
    without orphaned worker tasks, fail-closed ``filter_keys`` enqueue dedup,
    and full-outage worker cleanup + ``busy`` release.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest

import lightrag.utils_pipeline as utils_pipeline
from lightrag import LightRAG
from lightrag.base import DocStatus, DocStatusStorage
from lightrag.exceptions import StorageCapabilityError
from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage
from lightrag.utils import compute_mdhash_id
from lightrag.utils_pipeline import (
    _CAPABILITY_CONSEQUENCES,
    enforce_strict_storage_capabilities,
)

from ._fakes import (
    build_initialized_rag,
    pipeline_status_handles,
    single_chunk_chunking,
    status_value_text,
)

pytestmark = pytest.mark.offline


async def _build_rag(tmp_path: Path, *, workspace_prefix: str) -> LightRAG:
    return await build_initialized_rag(
        tmp_path,
        workspace_prefix=workspace_prefix,
        chunking_func=single_chunk_chunking,
        max_parallel_insert=1,
    )


# ---------------------------------------------------------------------------
# PIPELINE_REQUIRE_STRICT_STORAGE_READS startup capability gate
# ---------------------------------------------------------------------------


def _minimal_backend():
    """A third-party doc_status implementing only abstract methods (fail-closed defaults)."""

    class _MinimalDocStatus(DocStatusStorage):
        supports_strict_point_reads = False

        async def get_docs_by_statuses_page(self, *args, **kwargs): ...
        async def get_docs_by_ids(self, *args, **kwargs): ...
        async def resolve_doc_source_strict(self, *args, **kwargs): ...
        async def get_full_docs_by_ids(self, *args, **kwargs): ...

    _MinimalDocStatus.__abstractmethods__ = frozenset()
    return _MinimalDocStatus.__new__(_MinimalDocStatus)


def test_a_first_party_backend_warns_about_nothing(lightrag_log_records):
    capabilities = enforce_strict_storage_capabilities(
        JsonDocStatusStorage.__new__(JsonDocStatusStorage)
    )

    assert all(capabilities.values())
    assert [r for r in lightrag_log_records if r.levelno >= logging.WARNING] == []


def test_gaps_are_warned_with_their_operator_facing_consequence(lightrag_log_records):
    enforce_strict_storage_capabilities(_minimal_backend())

    warnings = [r for r in lightrag_log_records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    for capability in (
        "strict_active_count",
        "source_conflict_listing",
        "source_conflict_repair",
        "strict_point_reads",
    ):
        assert capability in message
    assert "503" in message and "501" in message
    assert "PIPELINE_REQUIRE_STRICT_STORAGE_READS=true" in message


def test_require_turns_the_gaps_into_a_startup_failure():
    with pytest.raises(StorageCapabilityError) as excinfo:
        enforce_strict_storage_capabilities(_minimal_backend(), require=True)

    assert "PIPELINE_REQUIRE_STRICT_STORAGE_READS" in str(excinfo.value)
    assert "strict_active_count" in str(excinfo.value)


def test_require_does_not_fail_a_backend_that_has_everything():
    enforce_strict_storage_capabilities(
        JsonDocStatusStorage.__new__(JsonDocStatusStorage), require=True
    )


def test_a_broken_probe_never_blocks_startup(monkeypatch):
    """Refusing to boot because the diagnostic broke would be worse than the degradation."""

    def _boom(_doc_status):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(utils_pipeline, "describe_doc_status_capabilities", _boom)

    assert utils_pipeline.enforce_strict_storage_capabilities({}, require=True) == {}


def test_the_paging_capabilities_are_not_part_of_the_gate():
    """Paging and typed resolution are @abstractmethod, stronger than an opt-in gate."""
    assert "scheduling_pages" not in _CAPABILITY_CONSEQUENCES
    assert "typed_source_resolution" not in _CAPABILITY_CONSEQUENCES

    for name in ("get_docs_by_statuses_page", "resolve_doc_source_strict"):
        assert getattr(DocStatusStorage, name).__isabstractmethod__


# ---------------------------------------------------------------------------
# Batch consistency validator strict content probe (LR2 §5.4 / §7.2)
# ---------------------------------------------------------------------------


async def _seed_active_row(rag: LightRAG, name: str) -> str:
    """An interrupted (PROCESSING) row — the shape the validator judges."""
    doc_id = compute_mdhash_id(name, prefix="doc-")
    now = datetime.now(timezone.utc).isoformat()
    await rag.doc_status.upsert(
        {
            doc_id: {
                "status": DocStatus.PROCESSING,
                "content_summary": "body",
                "content_length": 4,
                "chunks_count": 0,
                "chunks_list": [],
                "created_at": now,
                "updated_at": now,
                "file_path": name,
                "track_id": "t",
                "error_msg": "",
                "metadata": {},
            }
        }
    )
    return doc_id


async def _validate_docs(rag: LightRAG, doc_ids: list[str]) -> dict:
    status, lock = await pipeline_status_handles(rag)
    hydrated = await rag.doc_status.get_full_docs_by_ids(doc_ids, strict=True)
    return await rag._validate_and_fix_document_consistency(hydrated, status, lock)


@pytest.mark.asyncio
async def test_unconfirmable_miss_keeps_the_row(tmp_path):
    """A backend WITHOUT strict point reads keeps the row and excludes it from
    the batch instead of deleting it on a best-effort miss."""
    rag = await _build_rag(tmp_path, workspace_prefix="vscp")
    try:
        doc_id = await _seed_active_row(rag, "blip.txt")
        rag.full_docs.supports_strict_point_reads = False

        async def best_effort_miss(_doc_id):
            return None

        rag.full_docs.get_by_id = best_effort_miss

        survivors = await _validate_docs(rag, [doc_id])

        assert doc_id not in survivors
        row = await rag.doc_status.get_by_id(doc_id)
        assert row is not None
        assert status_value_text(row["status"]) == DocStatus.PROCESSING.value
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_confirmed_absence_still_deletes(tmp_path):
    """With a strict point read confirming absence, the inconsistent row is deleted."""
    rag = await _build_rag(tmp_path, workspace_prefix="vscp")
    try:
        doc_id = await _seed_active_row(rag, "really-gone.txt")
        rag.full_docs.supports_strict_point_reads = True

        async def confirmed_absent(_doc_id):
            return None

        rag.full_docs.get_by_id_strict = confirmed_absent

        survivors = await _validate_docs(rag, [doc_id])

        assert doc_id not in survivors
        assert await rag.doc_status.get_by_id(doc_id) is None
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_strict_read_failure_propagates(tmp_path):
    """A strict point read raising must abort the batch and keep rows intact."""
    rag = await _build_rag(tmp_path, workspace_prefix="vscp")
    try:
        doc_id = await _seed_active_row(rag, "flaky.txt")
        rag.full_docs.supports_strict_point_reads = True

        async def boom(_doc_id):
            raise ConnectionError("full_docs transport failure")

        rag.full_docs.get_by_id_strict = boom

        with pytest.raises(ConnectionError, match="transport failure"):
            await _validate_docs(rag, [doc_id])

        assert await rag.doc_status.get_by_id(doc_id) is not None
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_content_is_read_once_per_document(tmp_path):
    """The consistency verdict and PENDING-reset payload come from ONE strict read."""
    rag = await _build_rag(tmp_path, workspace_prefix="vscp")
    try:
        doc_id = await _seed_active_row(rag, "counted.txt")
        await rag.full_docs.upsert({doc_id: {"content": "body"}})

        rag.full_docs.supports_strict_point_reads = True
        calls: list[str] = []
        real_strict = rag.full_docs.get_by_id_strict

        async def counting(target):
            calls.append(target)
            return await real_strict(target)

        rag.full_docs.get_by_id_strict = counting

        survivors = await _validate_docs(rag, [doc_id])

        assert doc_id in survivors
        row = await rag.doc_status.get_by_id(doc_id)
        assert status_value_text(row["status"]) == DocStatus.PENDING.value
        assert calls == [doc_id]
    finally:
        await rag.finalize_storages()


# ---------------------------------------------------------------------------
# Orchestrator-level storage failures during batch enqueue
# ---------------------------------------------------------------------------


def _live_worker_tasks() -> list[asyncio.Task]:
    """Pipeline worker tasks (parse/analyze/process) that are still pending."""
    live = []
    for task in asyncio.all_tasks():
        qualname = getattr(task.get_coro(), "__qualname__", "")
        if any(
            marker in qualname
            for marker in ("_parse_worker", "_analyze_worker", "_process_worker")
        ):
            if not task.done():
                live.append(task)
    return live


async def _passthrough_validate(self_docs, pipeline_status, pipeline_status_lock):
    return self_docs


@pytest.mark.asyncio
async def test_single_doc_get_by_id_failure_is_isolated_no_orphans(
    tmp_path, monkeypatch
):
    """One doc's enqueue get_by_id fails -> that doc FAILED, the rest PROCESSED,
    busy released, and no worker tasks are orphaned."""
    rag = await _build_rag(tmp_path, workspace_prefix="orphan")
    try:
        bad_path = "bad.txt"
        good_path = "good.txt"
        bad_id = compute_mdhash_id(bad_path, prefix="doc-")
        good_id = compute_mdhash_id(good_path, prefix="doc-")
        await rag.apipeline_enqueue_documents(
            input=["bad content", "good content"],
            file_paths=[bad_path, good_path],
        )

        monkeypatch.setattr(
            rag, "_validate_and_fix_document_consistency", _passthrough_validate
        )

        orig_get = rag.full_docs.get_by_id

        async def flaky_get(doc_id):
            if doc_id == bad_id:
                raise ConnectionError("redis down for this doc")
            return await orig_get(doc_id)

        monkeypatch.setattr(rag.full_docs, "get_by_id", flaky_get)

        await rag.apipeline_process_enqueue_documents()
        await asyncio.sleep(0)

        assert _live_worker_tasks() == [], "worker tasks were orphaned"

        bad_status = await rag.doc_status.get_by_id(bad_id)
        good_status = await rag.doc_status.get_by_id(good_id)
        assert status_value_text(bad_status["status"]) == "failed"
        assert status_value_text(good_status["status"]) == "processed"

        pipeline_status, _ = await pipeline_status_handles(rag)
        assert pipeline_status.get("busy") is False
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_enqueue_dedup_failure_aborts_before_any_upsert(tmp_path, monkeypatch):
    """Transient ``doc_status.filter_keys`` failure aborts enqueue before any upsert."""
    rag = await _build_rag(tmp_path, workspace_prefix="orphan")
    try:
        full_docs_upserts: list = []
        doc_status_upserts: list = []
        orig_full_upsert = rag.full_docs.upsert
        orig_status_upsert = rag.doc_status.upsert

        async def spy_full_upsert(*args, **kwargs):
            full_docs_upserts.append(args)
            return await orig_full_upsert(*args, **kwargs)

        async def spy_status_upsert(*args, **kwargs):
            doc_status_upserts.append(args)
            return await orig_status_upsert(*args, **kwargs)

        async def dead_filter_keys(keys):
            raise ConnectionError("doc_status backend transient failure")

        monkeypatch.setattr(rag.full_docs, "upsert", spy_full_upsert)
        monkeypatch.setattr(rag.doc_status, "upsert", spy_status_upsert)
        monkeypatch.setattr(rag.doc_status, "filter_keys", dead_filter_keys)

        with pytest.raises(ConnectionError):
            await rag.apipeline_enqueue_documents(input="body", file_paths="a.txt")

        assert full_docs_upserts == [], "full_docs written despite dedup failure"
        assert doc_status_upserts == [], "doc_status written despite dedup failure"
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_full_outage_escape_still_cancels_workers(tmp_path, monkeypatch):
    """When even FAILED-status write fails, workers are still cancelled and busy released."""
    rag = await _build_rag(tmp_path, workspace_prefix="orphan")
    try:
        await rag.apipeline_enqueue_documents(
            input="outage content", file_paths="outage.txt"
        )

        monkeypatch.setattr(
            rag, "_validate_and_fix_document_consistency", _passthrough_validate
        )

        async def dead_get(doc_id):
            raise ConnectionError("redis fully down")

        async def dead_upsert(*args, **kwargs):
            raise ConnectionError("redis fully down")

        monkeypatch.setattr(rag.full_docs, "get_by_id", dead_get)
        monkeypatch.setattr(rag.doc_status, "upsert", dead_upsert)

        with pytest.raises(Exception):
            await rag.apipeline_process_enqueue_documents()
        await asyncio.sleep(0)

        assert _live_worker_tasks() == [], "worker tasks were orphaned on escape"

        pipeline_status, _ = await pipeline_status_handles(rag)
        assert pipeline_status.get("busy") is False
    finally:
        await rag.finalize_storages()
