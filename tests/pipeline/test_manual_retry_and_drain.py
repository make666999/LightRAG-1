"""Consolidated manual retry EXCLUSIVE_RESET and DRAIN_TO_IDLE contract tests.

Covers (PipelineConcurrencyContract — LR2 Phase 3 / §7.2 / §7.7):
  * Manual retry EXCLUSIVE_RESET protocol: draining AUTO backlog before reset,
    no retry loops on re-failure, mid-reset page failure idempotency, bounded
    FAILED paging, ``created_at`` preservation, content-less stub preservation,
    enqueue freeze reservation conflict, owner-checked reset/worker/parse
    writes, strict ``full_docs`` content probes, and targeted ``content_hash``
    field patching.
  * DRAIN_TO_IDLE forward-progress guarantees: routing sweeps skipping
    journaled custom-chunk rows, drain confirmation paging to ``CURSOR_END``,
    ``recovery_required`` fencing (``manual_drain_blocked`` and
    ``manual_drain_stalled``) with bounded samples, and real
    ``POST /documents/recovery/force_reset`` atomic cancel-before-clear
    recovery ordering.
  * Bounded in-flight enqueue wait fence (``manual_drain_enqueue_stalled``):
    independent stall trackers, progress resets on shrinking/empty/changed
    sets, stale-snapshot race protection, token/PID redaction on public
    messages, and retained source-conflict repair guard messaging.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from lightrag import LightRAG
from lightrag import pipeline as pipeline_module
from lightrag.base import CURSOR_END, CURSOR_START, DocStatus
from lightrag.constants import CUSTOM_CHUNK_PATCH_METADATA_KEY
from lightrag.exceptions import PipelineRecoveryRequiredError
from lightrag.kg.shared_storage import (
    MANUAL_PHASE_DRAIN_TO_IDLE,
    SOURCE_REPAIR_RESERVATION_KIND,
    PipelineReservationConflict,
    acquire_enqueue_reservation,
    acquire_reservation,
    finalize_share_data,
    get_pipeline_ingress,
    initialize_share_data,
    make_owner_record,
    release_token_set_reservation,
)
from lightrag.parser.registry import parser_specs_snapshot
from lightrag.pipeline import (
    _MANUAL_DRAIN_STALL_ROUNDS,
    PipelineNextDecision,
    PipelineNextStep,
    _BatchRunContext,
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

_SPIN_TIMEOUT_SECONDS = 20
_STALL_ROUNDS = 3


class _CountingExtract:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = 0

    async def __call__(self, chunks, *args, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("extract fail sentinel")
        return [({}, {}) for _ in chunks]


async def _noop_extract(chunks, *args, **kwargs):
    return [({}, {}) for _ in chunks]


async def _build_rag(
    tmp_path: Path,
    *,
    extract=_noop_extract,
    workspace_prefix: str = "mxr",
) -> LightRAG:
    rag = await build_initialized_rag(
        tmp_path,
        workspace_prefix=workspace_prefix,
        chunking_func=single_chunk_chunking,
        max_parallel_insert=1,
    )
    if extract is not None:
        rag._process_extract_entities = extract  # type: ignore[assignment]
    return rag


async def _status_of(rag: LightRAG, doc_id: str) -> str:
    row = await rag.doc_status.get_by_id(doc_id)
    raw = (row or {}).get("status")
    return raw.value if isinstance(raw, DocStatus) else str(raw or "<missing>")


async def _make_failed(rag: LightRAG, name: str, extract: _CountingExtract) -> str:
    extract.fail = True
    await rag.apipeline_enqueue_documents(input=f"body of {name}", file_paths=name)
    await rag.apipeline_process_enqueue_documents()
    doc_id = compute_mdhash_id(name, prefix="doc-")
    assert await _status_of(rag, doc_id) == DocStatus.FAILED.value
    return doc_id


async def _seed_journaled_row(rag: LightRAG, name: str, status: DocStatus) -> str:
    """Write the row a SIGKILLed ``ainsert_custom_chunks`` leaves behind."""
    doc_id = compute_mdhash_id(name, prefix="doc-")
    now = datetime.now(timezone.utc).isoformat()
    await rag.full_docs.upsert({doc_id: {"content": f"body of {name}"}})
    await rag.doc_status.upsert(
        {
            doc_id: {
                "status": status,
                "content_summary": f"body of {name}",
                "content_length": 10,
                "chunks_count": 0,
                "chunks_list": [],
                "created_at": now,
                "updated_at": now,
                "file_path": name,
                "track_id": "t-journal",
                "error_msg": "",
                "metadata": {
                    CUSTOM_CHUNK_PATCH_METADATA_KEY: {
                        "schema_version": 1,
                        "operation_id": "op-1",
                        "mode": "patch",
                        "chunk_ids": [],
                        "phase": "prepared",
                    }
                },
            }
        }
    )
    return doc_id


async def _seed_failed_with_content(rag: LightRAG, name: str) -> str:
    """A plain retryable FAILED row: content present, no journal."""
    doc_id = compute_mdhash_id(name, prefix="doc-")
    now = datetime.now(timezone.utc).isoformat()
    await rag.full_docs.upsert({doc_id: {"content": f"body of {name}"}})
    await rag.doc_status.upsert(
        {
            doc_id: {
                "status": DocStatus.FAILED,
                "content_summary": f"body of {name}",
                "content_length": 10,
                "chunks_count": 0,
                "chunks_list": [],
                "created_at": now,
                "updated_at": now,
                "file_path": name,
                "track_id": "t-failed",
                "error_msg": "boom",
                "metadata": {},
            }
        }
    )
    return doc_id


def _load_document_routes(monkeypatch):
    """Import document_routes without letting pytest argv trip its CLI parser."""
    monkeypatch.setattr(sys, "argv", [sys.argv[0]])
    return importlib.import_module("lightrag.api.routers.document_routes")


def _get_force_reset_endpoint(dr, rag: LightRAG, tmp_path: Path):
    router = dr.create_document_routes(rag, dr.DocumentManager(str(tmp_path)))
    return [
        r.endpoint
        for r in router.routes
        if getattr(r, "name", "") == "force_reset_recovery"
    ][-1]


# ---------------------------------------------------------------------------
# Manual retry EXCLUSIVE_RESET protocol (LR2 Phase 3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_manual_drains_auto_backlog_before_reset(tmp_path):
    """LR2 §13.2 case 5: a manual retry arriving with an AUTO backlog freezes
    new ingress, drains the AUTO (PENDING) backlog to idle FIRST, then resets
    FAILED->PENDING and processes those too — all in one run."""
    extract = _CountingExtract()
    rag = await _build_rag(tmp_path, extract=extract)
    try:
        failed_id = await _make_failed(rag, "old.txt", extract)
        extract.fail = False
        await rag.apipeline_enqueue_documents(input="fresh body", file_paths="new.txt")
        pending_id = compute_mdhash_id("new.txt", prefix="doc-")
        assert await _status_of(rag, pending_id) == DocStatus.PENDING.value

        await request_failed_retry(rag)
        await rag.apipeline_process_enqueue_documents()

        assert await _status_of(rag, pending_id) == DocStatus.PROCESSED.value
        assert await _status_of(rag, failed_id) == DocStatus.PROCESSED.value
        ingress = await get_pipeline_ingress(rag.workspace)
        assert ingress.snapshot_manual_retries() == []  # ACKed
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_refail_during_process_stays_failed(tmp_path):
    """LR2 §7.4 / §13.2 case 3: a doc the manual reset returns to PENDING that
    FAILS AGAIN during the PROCESS phase stays FAILED — the SAME request never
    resets it a second time (no retry loop)."""
    extract = _CountingExtract()
    rag = await _build_rag(tmp_path, extract=extract)
    try:
        failed_id = await _make_failed(rag, "a.txt", extract)
        calls_after_failure = extract.calls

        await request_failed_retry(rag)
        await rag.apipeline_process_enqueue_documents()

        assert await _status_of(rag, failed_id) == DocStatus.FAILED.value
        assert extract.calls == calls_after_failure + 1
        # A second run with no new signal must not retry again.
        await rag.apipeline_process_enqueue_documents()
        assert extract.calls == calls_after_failure + 1
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_reset_idempotent_across_mid_reset_failure(tmp_path):
    """LR2 §13.2 case 2: a reset that fails on a later FAILED page leaves the
    already-reset rows PENDING; the next run re-runs the reset from Start and
    does NOT re-reset them. Every doc is processed exactly once."""
    extract = _CountingExtract()
    rag = await _build_rag(tmp_path, extract=extract)
    try:
        rag.pipeline_scheduling_page_size = 1  # one FAILED per page
        id_a = await _make_failed(rag, "a.txt", extract)
        id_b = await _make_failed(rag, "b.txt", extract)
        calls_after_setup = extract.calls

        original_page = rag.doc_status.get_docs_by_statuses_page
        state = {"failed_pages": 0, "armed": True}

        async def flaky(statuses, *, limit, position, strict=False):
            if list(statuses) == [DocStatus.FAILED]:
                state["failed_pages"] += 1
                if state["failed_pages"] == 2 and state["armed"]:
                    state["armed"] = False
                    raise RuntimeError("mid-reset page boom")
            return await original_page(
                statuses, limit=limit, position=position, strict=strict
            )

        rag.doc_status.get_docs_by_statuses_page = flaky

        extract.fail = False
        await request_failed_retry(rag)
        with pytest.raises(RuntimeError, match="mid-reset page boom"):
            await rag.apipeline_process_enqueue_documents()

        ingress = await get_pipeline_ingress(rag.workspace)
        assert ingress.snapshot_manual_retries()  # still sticky, not ACKed
        status, _ = await pipeline_status_handles(rag)
        assert status["manual_freeze_requested"] is False

        # Next run completes: both docs processed exactly once, no double reset.
        await rag.apipeline_process_enqueue_documents()
        assert await _status_of(rag, id_a) == DocStatus.PROCESSED.value
        assert await _status_of(rag, id_b) == DocStatus.PROCESSED.value
        assert extract.calls == calls_after_setup + 2
        assert ingress.snapshot_manual_retries() == []  # ACKed
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_exclusive_reset_pages_failed_backlog(tmp_path):
    """With paging on and a small page, the exclusive reset pages the FAILED
    backlog (never one giant scan) and every FAILED doc is reset+reprocessed."""
    extract = _CountingExtract()
    rag = await _build_rag(tmp_path, extract=extract)
    try:
        rag.pipeline_scheduling_page_size = 2
        ids = [await _make_failed(rag, f"f{i}.txt", extract) for i in range(5)]

        page_calls = {"failed": 0}
        original_page = rag.doc_status.get_docs_by_statuses_page

        async def counting_page(statuses, *, limit, position, strict=False):
            if list(statuses) == [DocStatus.FAILED]:
                page_calls["failed"] += 1
            return await original_page(
                statuses, limit=limit, position=position, strict=strict
            )

        rag.doc_status.get_docs_by_statuses_page = counting_page

        extract.fail = False
        await request_failed_retry(rag)
        await rag.apipeline_process_enqueue_documents()

        for doc_id in ids:
            assert await _status_of(rag, doc_id) == DocStatus.PROCESSED.value
        assert page_calls["failed"] >= 3
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_exclusive_reset_preserves_created_at(tmp_path):
    """FAILED->PENDING reset preserves the immutable created_at (LR2 §4.4/§7.3)."""
    extract = _CountingExtract()
    rag = await _build_rag(tmp_path, extract=extract)
    try:
        doc_id = await _make_failed(rag, "keep.txt", extract)
        created_before = (await rag.doc_status.get_by_id(doc_id))["created_at"]

        extract.fail = False
        await request_failed_retry(rag)
        await rag.apipeline_process_enqueue_documents()

        row = await rag.doc_status.get_by_id(doc_id)
        assert row["status"] in (DocStatus.PROCESSED, DocStatus.PROCESSED.value)
        assert row["created_at"] == created_before
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_exclusive_reset_leaves_stub_without_content_failed(tmp_path):
    """A FAILED doc whose full_docs content is gone is unprocessable: the
    exclusive reset must leave it FAILED, never reset it into a re-fail loop."""
    extract = _CountingExtract()
    rag = await _build_rag(tmp_path, extract=extract)
    try:
        doc_id = await _make_failed(rag, "stub.txt", extract)
        await rag.full_docs.delete([doc_id])

        extract.fail = False
        await request_failed_retry(rag)
        await rag.apipeline_process_enqueue_documents()

        assert await _status_of(rag, doc_id) == DocStatus.FAILED.value
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_begin_manual_drain_freezes_new_enqueue_reservation(tmp_path):
    """While a manual drain holds the freeze, a NEW enqueue reservation is
    rejected with MANUAL_FREEZE (-> 409); clearing the freeze re-admits it."""
    extract = _CountingExtract()
    rag = await _build_rag(tmp_path, extract=extract)
    try:
        status, lock = await pipeline_status_handles(rag)
        token = "busy-tok"
        status.update(
            {"busy": True, "busy_owner": make_owner_record(token, "processing")}
        )

        assert await rag._begin_manual_drain("req-1", token, status, lock) is True
        assert status["manual_freeze_requested"] is True
        assert status["manual_phase"] == MANUAL_PHASE_DRAIN_TO_IDLE
        assert status["manual_owner"]["request_id"] == "req-1"
        assert status["manual_owner"]["owner_token"] == token

        reject_when = (("manual_freeze_requested", "manual retry draining"),)
        blocked = await acquire_enqueue_reservation(
            status, lock, token="up-1", reject_when=reject_when
        )
        assert blocked.acquired is False
        assert blocked.conflict is PipelineReservationConflict.MANUAL_FREEZE

        await rag._end_manual_drain(token, status, lock)
        assert status["manual_freeze_requested"] is False
        assert status["manual_owner"] is None
        admitted = await acquire_enqueue_reservation(
            status, lock, token="up-2", reject_when=reject_when
        )
        assert admitted.acquired is True
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_exclusive_reset_returns_false_when_not_owner(tmp_path):
    """LR2 §7.7 item 8 / §13.2 case 11: if the busy owner changed, a stale
    run's reset returns False and touches nothing."""
    extract = _CountingExtract()
    rag = await _build_rag(tmp_path, extract=extract)
    try:
        doc_id = await _make_failed(rag, "a.txt", extract)
        status, lock = await pipeline_status_handles(rag)
        status.update(
            {"busy": True, "busy_owner": make_owner_record("owner-B", "processing")}
        )

        wrote = {"called": False}
        orig_upsert = rag.doc_status.upsert

        async def spy_upsert(data):
            wrote["called"] = True
            return await orig_upsert(data)

        rag.doc_status.upsert = spy_upsert

        ok = await rag._run_exclusive_failed_reset("req-stale", "owner-A", status, lock)
        assert ok is False
        assert wrote["called"] is False
        assert status["manual_resetting"] is False
        assert await _status_of(rag, doc_id) == DocStatus.FAILED.value
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_reset_page_rejects_write_when_owner_changed(tmp_path):
    """_reset_failed_page raises rather than persist a FAILED->PENDING page
    once the freeze owner no longer matches."""
    extract = _CountingExtract()
    rag = await _build_rag(tmp_path, extract=extract)
    try:
        await _make_failed(rag, "a.txt", extract)
        status, lock = await pipeline_status_handles(rag)
        status.update(
            {"busy": True, "busy_owner": make_owner_record("owner-B", "processing")}
        )

        docs, _ = await rag._next_failed_page(CURSOR_START)
        assert docs

        wrote = {"called": False}
        orig_upsert = rag.doc_status.upsert

        async def spy_upsert(data):
            wrote["called"] = True
            return await orig_upsert(data)

        rag.doc_status.upsert = spy_upsert

        with pytest.raises(RuntimeError, match="lost freeze ownership"):
            await rag._reset_failed_page(docs, "owner-A", status, lock)
        assert wrote["called"] is False
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_next_failed_page_single_scan_when_paging_off(tmp_path):
    """page_size<=0 collapses the FAILED reset to one strict scan ending at
    CURSOR_END (legacy path); paging on returns a keyset cursor."""
    extract = _CountingExtract()
    rag = await _build_rag(tmp_path, extract=extract)
    try:
        await _make_failed(rag, "a.txt", extract)
        await _make_failed(rag, "b.txt", extract)

        rag.pipeline_scheduling_page_size = 0
        docs, cursor = await rag._next_failed_page(CURSOR_START)
        assert set(docs) and cursor is not None
        assert cursor is CURSOR_END
        assert all(
            d.status in (DocStatus.FAILED, DocStatus.FAILED.value)
            for d in docs.values()
        )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_reset_page_propagates_a_strict_full_docs_failure(tmp_path):
    """Fix-proof: the content probe must use strict point reads so a transport
    failure propagates instead of looking like an unprocessable stub."""
    extract = _CountingExtract()
    rag = await _build_rag(tmp_path, extract=extract)
    try:
        await _make_failed(rag, "a.txt", extract)
        status, lock = await pipeline_status_handles(rag)
        docs, _ = await rag._next_failed_page(CURSOR_START)
        assert docs
        assert rag.full_docs.supports_strict_point_reads is True

        async def _boom(doc_id):
            raise RuntimeError("full_docs transport failure")

        rag.full_docs.get_by_id_strict = _boom
        rag.full_docs.get_by_id = _boom

        wrote = {"called": False}
        orig_upsert = rag.doc_status.upsert

        async def spy_upsert(data):
            wrote["called"] = True
            return await orig_upsert(data)

        rag.doc_status.upsert = spy_upsert

        with pytest.raises(RuntimeError, match="full_docs transport failure"):
            await rag._reset_failed_page(docs, "tok", status, lock)
        assert wrote["called"] is False
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_reset_page_counts_unconfirmed_misses_without_strict_reads(tmp_path):
    """A backend that cannot confirm absence keeps the conservative skip, but
    surfaces the skipped docs in history_messages."""
    extract = _CountingExtract()
    rag = await _build_rag(tmp_path, extract=extract)
    try:
        await _make_failed(rag, "a.txt", extract)
        status, lock = await pipeline_status_handles(rag)
        status["history_messages"] = []
        docs, _ = await rag._next_failed_page(CURSOR_START)
        assert docs

        type(rag.full_docs).supports_strict_point_reads = False
        try:

            async def _miss(doc_id):
                return None

            rag.full_docs.get_by_id = _miss
            reset = await rag._reset_failed_page(docs, "tok", status, lock)
        finally:
            type(rag.full_docs).supports_strict_point_reads = True

        assert reset == 0
        assert any(
            "cannot confirm whether their content is really absent" in m
            for m in status["history_messages"]
        ), status["history_messages"]
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_worker_status_write_is_discarded_when_ownership_was_reclaimed(tmp_path):
    """LR2 §7.7 items 3/4/7: a run whose ownership was reclaimed must DISCARD
    its late status write, not stamp a final status over the new owner's work."""
    extract = _CountingExtract()
    rag = await _build_rag(tmp_path, extract=extract)
    try:
        doc_id = await _make_failed(rag, "a.txt", extract)
        status, lock = await pipeline_status_handles(rag)
        status["history_messages"] = []
        status.update(
            {"busy": True, "busy_owner": make_owner_record("owner-B", "processing")}
        )

        ctx = _BatchRunContext(
            pipeline_status=status,
            pipeline_status_lock=lock,
            semaphore=asyncio.Semaphore(1),
            total_files=1,
            parse_queues={"native": asyncio.Queue()},
            parser_specs=parser_specs_snapshot(),
            q_analyze=asyncio.Queue(),
            q_process=asyncio.Queue(),
            run_owner_token="owner-A",
        )

        rows = await rag.doc_status.get_by_id(doc_id)
        status_doc = (await rag._next_failed_page(CURSOR_START))[0][doc_id]
        before = rows.get("status")

        wrote = {"called": False}
        orig_upsert = rag.doc_status.upsert

        async def spy_upsert(data):
            wrote["called"] = True
            return await orig_upsert(data)

        rag.doc_status.upsert = spy_upsert

        await rag._upsert_doc_status_transition(
            doc_id=doc_id,
            status=DocStatus.PROCESSED,
            status_doc=status_doc,
            file_path="a.txt",
            ctx=ctx,
        )

        assert wrote["called"] is False
        after = (await rag.doc_status.get_by_id(doc_id)).get("status")
        assert after == before

        # The current owner's write still goes through.
        ctx.run_owner_token = "owner-B"
        await rag._upsert_doc_status_transition(
            doc_id=doc_id,
            status=DocStatus.PROCESSED,
            status_doc=status_doc,
            file_path="a.txt",
            ctx=ctx,
        )
        assert wrote["called"] is True
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_parsed_content_write_is_discarded_when_ownership_was_reclaimed(tmp_path):
    """§7.7 guarantee at ``_persist_parsed_full_docs``: a parse result returning
    after a reaper reclaim must touch neither store and return None."""
    extract = _CountingExtract()
    rag = await _build_rag(tmp_path, extract=extract)
    try:
        doc_id = await _make_failed(rag, "a.txt", extract)
        status, lock = await pipeline_status_handles(rag)
        status.update(
            {"busy": True, "busy_owner": make_owner_record("owner-B", "processing")}
        )

        ctx = _BatchRunContext(
            pipeline_status=status,
            pipeline_status_lock=lock,
            semaphore=asyncio.Semaphore(1),
            total_files=1,
            parse_queues={"native": asyncio.Queue()},
            parser_specs=parser_specs_snapshot(),
            q_analyze=asyncio.Queue(),
            q_process=asyncio.Queue(),
            run_owner_token="owner-A",
        )
        rag._active_run_ctx = ctx

        full_before = await rag.full_docs.get_by_id(doc_id)
        status_before = await rag.doc_status.get_by_id(doc_id)

        result = await rag._persist_parsed_full_docs(
            doc_id,
            {
                "content": "text a stale run re-parsed",
                "file_path": "a.txt",
                "parse_format": "raw",
            },
        )

        assert result is None
        assert await rag.full_docs.get_by_id(doc_id) == full_before
        assert await rag.doc_status.get_by_id(doc_id) == status_before

        # The current owner's parse result still lands.
        ctx.run_owner_token = "owner-B"
        content_hash = await rag._persist_parsed_full_docs(
            doc_id,
            {
                "content": "text the real owner parsed",
                "file_path": "a.txt",
                "parse_format": "raw",
            },
        )
        assert content_hash
        after = await rag.full_docs.get_by_id(doc_id)
        assert after["content"] == "text the real owner parsed"
        assert (await rag.doc_status.get_by_id(doc_id))["content_hash"] == content_hash
    finally:
        rag._active_run_ctx = None
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_persist_parsed_patches_content_hash_without_rewriting_the_row(tmp_path):
    """LR2 §5.6: ``_persist_parsed_full_docs`` syncs ``content_hash`` through
    the TARGETED field update, not a read-modify-write ``upsert``."""
    extract = _CountingExtract()
    rag = await _build_rag(tmp_path, extract=extract)
    try:
        doc_id = "doc-patch"
        chunks = [f"chunk-{index}" for index in range(50)]
        await rag.full_docs.upsert({doc_id: {"content": "old body"}})
        await rag.doc_status.upsert(
            {
                doc_id: {
                    "status": DocStatus.PARSING,
                    "content_summary": "old body",
                    "content_length": 8,
                    "chunks_count": len(chunks),
                    "chunks_list": chunks,
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "file_path": "patch.txt",
                    "track_id": "t-patch",
                    "error_msg": "",
                    "metadata": {"keep": "me"},
                }
            }
        )

        targeted: list[tuple[str, dict, bool]] = []
        full_upserts: list[dict] = []
        real_update = rag.doc_status.update_doc_status_fields
        real_upsert = rag.doc_status.upsert

        async def spy_update(target, fields, *, missing_ok=False):
            targeted.append((target, dict(fields), missing_ok))
            return await real_update(target, fields, missing_ok=missing_ok)

        async def spy_upsert(data):
            full_upserts.append(dict(data))
            return await real_upsert(data)

        rag.doc_status.update_doc_status_fields = spy_update
        rag.doc_status.upsert = spy_upsert

        content_hash = await rag._persist_parsed_full_docs(
            doc_id,
            {
                "content": "freshly parsed body",
                "file_path": "patch.txt",
                "parse_format": "raw",
            },
        )

        assert len(targeted) == 1
        patched_id, fields, missing_ok = targeted[0]
        assert patched_id == doc_id
        assert set(fields) == {"content_hash", "updated_at"}
        assert fields["content_hash"] == content_hash
        assert missing_ok is True
        assert full_upserts == []

        row = await rag.doc_status.get_by_id(doc_id)
        assert row["chunks_list"] == chunks
        assert row["chunks_count"] == len(chunks)
        assert row["metadata"] == {"keep": "me"}
        assert row["created_at"] == "2026-01-01T00:00:00+00:00"
        assert row["content_hash"] == content_hash
    finally:
        await rag.finalize_storages()


# ---------------------------------------------------------------------------
# DRAIN_TO_IDLE forward progress, blocker detection, and force_reset ordering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("page_size", [500, 0], ids=["paged", "paging-disabled"])
async def test_scheduling_sweep_skips_journaled_rows(tmp_path, page_size):
    """``_next_scheduling_page`` drops rows with an unfinished custom-chunk
    journal in BOTH paged and legacy single-scan modes."""
    rag = await _build_rag(tmp_path, workspace_prefix="mdp")
    try:
        rag.pipeline_scheduling_page_size = page_size
        journaled = await _seed_journaled_row(
            rag, "journaled.txt", DocStatus.PROCESSING
        )
        plain = await _seed_journaled_row(rag, "plain.txt", DocStatus.PENDING)
        row = await rag.doc_status.get_by_id(plain)
        row["metadata"] = {}
        await rag.doc_status.upsert({plain: row})

        docs, _cursor = await rag._next_scheduling_page(
            (
                DocStatus.PENDING,
                DocStatus.PROCESSING,
                DocStatus.PARSING,
                DocStatus.ANALYZING,
            ),
            CURSOR_START,
        )

        assert journaled not in docs
        assert plain in docs
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_journaled_active_row_fences_instead_of_resetting(tmp_path):
    """A journaled ACTIVE row means the pipeline is NOT idle, so the exclusive
    reset must fence with ``manual_drain_blocked`` instead of resetting or spinning."""
    rag = await _build_rag(tmp_path, workspace_prefix="mdp")
    try:
        journaled = await _seed_journaled_row(
            rag, "journaled.txt", DocStatus.PROCESSING
        )
        failed = await _seed_failed_with_content(rag, "retry-me.txt")
        journaled_row_before = await rag.doc_status.get_by_id(journaled)

        await request_failed_retry(rag)
        with pytest.raises(PipelineRecoveryRequiredError) as excinfo:
            await asyncio.wait_for(
                rag.apipeline_process_enqueue_documents(),
                timeout=_SPIN_TIMEOUT_SECONDS,
            )

        assert excinfo.value.blocked_doc_ids == (journaled,)
        detail = str(excinfo.value)
        assert "/documents/recovery/force_reset" in detail
        assert "/documents/scan" in detail

        assert await _status_of(rag, failed) == DocStatus.FAILED.value
        ingress = await get_pipeline_ingress(rag.workspace)
        assert len(ingress.snapshot_manual_retries()) == 1
        assert await rag.doc_status.get_by_id(journaled) == journaled_row_before

        status, _ = await pipeline_status_handles(rag)
        fence = status.get("recovery_required")
        assert isinstance(fence, dict)
        assert fence["kind"] == "manual_drain_blocked"
        assert status.get("busy") is False
        assert status.get("manual_freeze_requested") is False
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_force_reset_endpoint_unblocks_the_documented_recovery_order(
    tmp_path, monkeypatch
):
    """Full recovery path through the REAL ``force_reset`` endpoint and REAL
    scan reservation."""
    dr = _load_document_routes(monkeypatch)
    rag = await _build_rag(tmp_path, workspace_prefix="mdp")
    try:
        await _seed_journaled_row(rag, "journaled.txt", DocStatus.PROCESSING)
        failed = await _seed_failed_with_content(rag, "retry-me.txt")

        await request_failed_retry(rag)
        with pytest.raises(PipelineRecoveryRequiredError):
            await asyncio.wait_for(
                rag.apipeline_process_enqueue_documents(),
                timeout=_SPIN_TIMEOUT_SECONDS,
            )

        status, lock = await pipeline_status_handles(rag)
        ingress = await get_pipeline_ingress(rag.workspace)

        async def _try_scan_reservation():
            return await acquire_reservation(
                status,
                lock,
                owner_key="scanning_owner",
                owner=uuid4().hex,
                owner_kind="scan",
                flags={"scanning": True, "scanning_exclusive": True},
                reject_when=(
                    ("busy", "busy"),
                    ("scanning", "scanning"),
                    ("pending_enqueues", "pending"),
                    ("manual_freeze_requested", "freeze"),
                ),
                pipeline_ingress=ingress,
                refuse_when_manual_pending=True,
            )

        # (1) Fenced -> the scan that would fix this is refused.
        refused = await _try_scan_reservation()
        assert refused.acquired is False
        assert refused.conflict is PipelineReservationConflict.RECOVERY_REQUIRED

        # (2) The real endpoint, one call.
        force_reset = _get_force_reset_endpoint(dr, rag, tmp_path)
        resp = await force_reset(dr.ForceResetRecoveryRequest(confirm=True))
        assert resp.status == "reset"
        assert resp.cancelled_manual_retries == 1
        assert status.get("recovery_required") is None
        assert ingress.snapshot_manual_retries() == []

        # (3) The scan is now allowed — the recovery path is reachable.
        granted = await _try_scan_reservation()
        assert granted.acquired is True, granted.message
        assert await _status_of(rag, failed) == DocStatus.FAILED.value
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_force_reset_fails_closed_when_the_cancellation_fails(
    tmp_path, monkeypatch
):
    """A force-reset that cannot cancel the queued intents must NOT drop the fence."""
    dr = _load_document_routes(monkeypatch)
    rag = await _build_rag(tmp_path, workspace_prefix="mdp")
    try:
        await _seed_journaled_row(rag, "journaled.txt", DocStatus.PROCESSING)
        await request_failed_retry(rag)
        with pytest.raises(PipelineRecoveryRequiredError):
            await asyncio.wait_for(
                rag.apipeline_process_enqueue_documents(),
                timeout=_SPIN_TIMEOUT_SECONDS,
            )

        status, _ = await pipeline_status_handles(rag)
        ingress = await get_pipeline_ingress(rag.workspace)
        fence_before = dict(status.get("recovery_required"))

        def _boom():
            raise ConnectionError("manager mailbox RPC failed")

        ingress.cancel_manual_retries = _boom

        force_reset = _get_force_reset_endpoint(dr, rag, tmp_path)
        with pytest.raises(dr.HTTPException) as excinfo:
            await force_reset(dr.ForceResetRecoveryRequest(confirm=True))

        assert excinfo.value.status_code == 503
        assert dict(status.get("recovery_required")) == fence_before
        assert len(ingress.snapshot_manual_retries()) == 1
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_force_reset_cancels_before_it_drops_the_fence(tmp_path, monkeypatch):
    """The cancellation and fence drop are ONE indivisible step, cancel first."""
    dr = _load_document_routes(monkeypatch)
    rag = await _build_rag(tmp_path, workspace_prefix="mdp")
    try:
        await _seed_journaled_row(rag, "journaled.txt", DocStatus.PROCESSING)
        await request_failed_retry(rag)
        with pytest.raises(PipelineRecoveryRequiredError):
            await asyncio.wait_for(
                rag.apipeline_process_enqueue_documents(),
                timeout=_SPIN_TIMEOUT_SECONDS,
            )

        status, _ = await pipeline_status_handles(rag)
        ingress = await get_pipeline_ingress(rag.workspace)

        observed: dict = {}
        real_cancel = ingress.cancel_manual_retries

        def _spy():
            observed["fence_still_set"] = status.get("recovery_required") is not None
            return real_cancel()

        ingress.cancel_manual_retries = _spy

        bad_samples: list[tuple] = []
        stop = asyncio.Event()

        async def _observer():
            while not stop.is_set():
                fenced = status.get("recovery_required") is not None
                queued = len(ingress.snapshot_manual_retries())
                if not fenced and queued:
                    bad_samples.append((fenced, queued))
                await asyncio.sleep(0)

        watcher = asyncio.create_task(_observer())

        force_reset = _get_force_reset_endpoint(dr, rag, tmp_path)
        resp = await force_reset(dr.ForceResetRecoveryRequest(confirm=True))

        stop.set()
        await watcher

        assert resp.status == "reset"
        assert observed["fence_still_set"] is True
        assert bad_samples == []
        assert status.get("recovery_required") is None
        assert ingress.snapshot_manual_retries() == []
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_clearing_only_the_fence_still_blocks_the_scan(tmp_path):
    """Fix-proof: dropping the fence WITHOUT cancelling the queued retry leaves
    ``/documents/scan`` refused for jumping the manual FIFO."""
    rag = await _build_rag(tmp_path, workspace_prefix="mdp")
    try:
        await _seed_journaled_row(rag, "journaled.txt", DocStatus.PROCESSING)
        await request_failed_retry(rag)
        with pytest.raises(PipelineRecoveryRequiredError):
            await asyncio.wait_for(
                rag.apipeline_process_enqueue_documents(),
                timeout=_SPIN_TIMEOUT_SECONDS,
            )

        status, lock = await pipeline_status_handles(rag)
        ingress = await get_pipeline_ingress(rag.workspace)

        async with lock:
            status["recovery_required"] = None

        result = await acquire_reservation(
            status,
            lock,
            owner_key="scanning_owner",
            owner=uuid4().hex,
            owner_kind="scan",
            flags={"scanning": True, "scanning_exclusive": True},
            reject_when=(),
            pipeline_ingress=ingress,
            refuse_when_manual_pending=True,
        )
        assert result.acquired is False
        assert result.conflict is PipelineReservationConflict.MANUAL_FREEZE
        assert "manual retry" in (result.message or "").lower()
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_drain_confirmation_pages_past_a_fully_filtered_page(tmp_path):
    """The final confirmation must page to CURSOR_END, not stop at one empty page."""
    rag = await _build_rag(tmp_path, workspace_prefix="mdp")
    try:
        rag.pipeline_scheduling_page_size = 1

        journaled = compute_mdhash_id("aaa-journaled.txt", prefix="doc-")
        await rag.full_docs.upsert({journaled: {"content": "body"}})
        await rag.doc_status.upsert(
            {
                journaled: {
                    "status": DocStatus.PROCESSING,
                    "content_summary": "body",
                    "content_length": 4,
                    "chunks_count": 0,
                    "chunks_list": [],
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "file_path": "aaa-journaled.txt",
                    "track_id": "t",
                    "error_msg": "",
                    "metadata": {
                        CUSTOM_CHUNK_PATCH_METADATA_KEY: {
                            "schema_version": 1,
                            "operation_id": "op-1",
                            "mode": "patch",
                            "chunk_ids": [],
                            "phase": "prepared",
                        }
                    },
                }
            }
        )
        routable = compute_mdhash_id("zzz-pending.txt", prefix="doc-")
        await rag.full_docs.upsert({routable: {"content": "body"}})
        await rag.doc_status.upsert(
            {
                routable: {
                    "status": DocStatus.PENDING,
                    "content_summary": "body",
                    "content_length": 4,
                    "chunks_count": 0,
                    "chunks_list": [],
                    "created_at": "2026-01-02T00:00:00+00:00",
                    "updated_at": "2026-01-02T00:00:00+00:00",
                    "file_path": "zzz-pending.txt",
                    "track_id": "t",
                    "error_msg": "",
                    "metadata": {},
                }
            }
        )

        docs, _cursor, blocking = await rag._confirm_auto_drained()
        assert set(docs) == {routable}
        assert blocking == ()
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_drain_confirmation_reports_a_blocker_only_after_cursor_end(tmp_path):
    """The blocker verdict requires reaching CURSOR_END across all pages."""
    rag = await _build_rag(tmp_path, workspace_prefix="mdp")
    try:
        rag.pipeline_scheduling_page_size = 1
        journaled_a = await _seed_journaled_row(
            rag, "a-journaled.txt", DocStatus.PROCESSING
        )
        journaled_b = await _seed_journaled_row(
            rag, "b-journaled.txt", DocStatus.PARSING
        )

        docs, cursor, blocking = await rag._confirm_auto_drained()

        assert docs == {}
        assert cursor is CURSOR_END
        assert set(blocking) == {journaled_a, journaled_b}
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_stalled_drain_fences_with_bounded_sample(tmp_path):
    """LR2 §7.2 / §13.2 case 16: when the same active rows block DRAIN_TO_IDLE
    round after round with no state change, the run fences with a BOUNDED sample."""
    rag = await _build_rag(tmp_path, workspace_prefix="mdp")
    try:
        blocked = []
        for index in range(12):
            doc_id = compute_mdhash_id(f"stuck-{index}.txt", prefix="doc-")
            now = datetime.now(timezone.utc).isoformat()
            await rag.full_docs.upsert({doc_id: {"content": "body"}})
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
                        "file_path": f"stuck-{index}.txt",
                        "track_id": "t",
                        "error_msg": "",
                        "metadata": {},
                    }
                }
            )
            blocked.append(doc_id)

        async def drop_everything(to_process_docs, *_args, **_kwargs):
            return {}

        rag._validate_and_fix_document_consistency = drop_everything

        await request_failed_retry(rag)
        with pytest.raises(PipelineRecoveryRequiredError) as excinfo:
            await asyncio.wait_for(
                rag.apipeline_process_enqueue_documents(),
                timeout=_SPIN_TIMEOUT_SECONDS,
            )

        sample = excinfo.value.blocked_doc_ids
        assert 0 < len(sample) <= 8
        assert set(sample).issubset(set(blocked))

        status, _ = await pipeline_status_handles(rag)
        fence = status.get("recovery_required")
        assert isinstance(fence, dict)
        assert fence["kind"] == "manual_drain_stalled"
        assert "drain stalled" in fence["message"]
        assert status.get("busy") is False
        assert status.get("manual_freeze_requested") is False

        ingress = await get_pipeline_ingress(rag.workspace)
        assert len(ingress.snapshot_manual_retries()) == 1

        for doc_id in blocked:
            assert await _status_of(rag, doc_id) == DocStatus.PROCESSING.value
    finally:
        await rag.finalize_storages()


def test_drain_progress_counter_resets_on_a_changing_id_set():
    """The stall guard keys on the blocking id SET, resetting on any change."""
    progress = _ManualDrainProgress()

    for round_index in range(_MANUAL_DRAIN_STALL_ROUNDS * 4):
        assert progress.observe({f"doc-{round_index}"}) is True

    assert progress.observe({"a", "b", "c"}) is True
    assert progress.observe({"a", "b"}) is True
    assert progress.observe({"a"}) is True

    fresh = _ManualDrainProgress()
    for _ in range(_MANUAL_DRAIN_STALL_ROUNDS - 1):
        assert fresh.observe({"a"}) is True
    assert fresh.observe({"a"}) is False
    assert fresh.observe({"a"}) is False


# ---------------------------------------------------------------------------
# DRAIN_TO_IDLE bounded wait on in-flight enqueues (LR2 §7.2 step 5)
# ---------------------------------------------------------------------------


class TestManualDrainEnqueueStallFence:
    @pytest.fixture(autouse=True)
    def _shared(self):
        initialize_share_data()
        yield
        finalize_share_data()

    @pytest.fixture(autouse=True)
    def _short_stall_window(self, monkeypatch):
        """Shrink the 10-minute window so the bound is testable in milliseconds."""
        monkeypatch.setattr(
            pipeline_module, "_MANUAL_DRAIN_ENQUEUE_STALL_ROUNDS", _STALL_ROUNDS
        )
        monkeypatch.setattr(pipeline_module, "_MANUAL_DRAIN_POLL_SECONDS", 0)

    @staticmethod
    async def _reserve(
        pipeline_status, lock, token: str, *, kind: str | None = None
    ) -> None:
        result = await acquire_enqueue_reservation(
            pipeline_status, lock, token=token, reject_when=(), weight=1, kind=kind
        )
        assert result.acquired

    @staticmethod
    async def _drain_wait_round(rag, pipeline_status, lock, progress):
        """One CONTINUE_DRAIN_WAIT refetch, exactly as the run performs it."""
        return await rag._refetch_for_decision(
            PipelineNextDecision(PipelineNextStep.CONTINUE_DRAIN_WAIT),
            (),
            CURSOR_START,
            None,
            token="busy-token",
            pipeline_status=pipeline_status,
            pipeline_status_lock=lock,
            drain_progress=progress,
        )

    def test_the_two_drain_waits_do_not_reset_each_other(self):
        """One drain interleaves both waits — shared evidence would hide a stall."""
        progress = _ManualDrainProgress()
        for _ in range(_STALL_ROUNDS - 1):
            assert progress.observe_enqueue({"a": {}}) is True
            assert progress.observe({"doc-1"}) is True
        assert progress.observe_enqueue({"a": {}}) is False

    @pytest.mark.asyncio
    async def test_a_stalled_in_flight_set_fences_the_workspace(self, tmp_path):
        """Fix-proof: this wait had no bound at all — the run spun on it forever."""
        rag = await _build_rag(tmp_path, extract=None, workspace_prefix="stall")
        try:
            pipeline_status, lock = await pipeline_status_handles(rag)
            token = uuid4().hex
            await self._reserve(pipeline_status, lock, token)
            progress = _ManualDrainProgress()

            for _ in range(_STALL_ROUNDS - 1):
                docs, _statuses, _cursor = await self._drain_wait_round(
                    rag, pipeline_status, lock, progress
                )
                assert docs == {}
                async with lock:
                    assert not pipeline_status.get("recovery_required")

            with pytest.raises(PipelineRecoveryRequiredError, match="drain stalled"):
                await self._drain_wait_round(rag, pipeline_status, lock, progress)

            async with lock:
                fence = pipeline_status.get("recovery_required")
            assert fence["kind"] == "manual_drain_enqueue_stalled"
            assert "force_reset" in fence["message"]
            assert token in fence["operation_record"]["scope"]
        finally:
            await rag.finalize_storages()

    @pytest.mark.asyncio
    async def test_an_enqueue_that_finishes_keeps_the_drain_waiting(self, tmp_path):
        """A producer that lands its documents must never be fenced."""
        rag = await _build_rag(tmp_path, extract=None, workspace_prefix="stall")
        try:
            pipeline_status, lock = await pipeline_status_handles(rag)
            slow = uuid4().hex
            finishing = uuid4().hex
            await self._reserve(pipeline_status, lock, slow)
            await self._reserve(pipeline_status, lock, finishing)
            progress = _ManualDrainProgress()

            for _ in range(_STALL_ROUNDS - 1):
                await self._drain_wait_round(rag, pipeline_status, lock, progress)

            await release_token_set_reservation(
                rag.workspace, tokens_key="pending_enqueue_tokens", token=finishing
            )
            for _ in range(_STALL_ROUNDS - 1):
                await self._drain_wait_round(rag, pipeline_status, lock, progress)

            async with lock:
                assert not pipeline_status.get("recovery_required")
                assert pipeline_status["pending_enqueues"] == 1
        finally:
            await rag.finalize_storages()

    @pytest.mark.asyncio
    async def test_an_empty_in_flight_set_is_not_a_stall(self, tmp_path):
        """``pending_enqueues == 0`` must never fence even on repeated refetches."""
        rag = await _build_rag(tmp_path, extract=None, workspace_prefix="stall")
        try:
            pipeline_status, lock = await pipeline_status_handles(rag)
            progress = _ManualDrainProgress()
            for _ in range(_STALL_ROUNDS * 2):
                await self._drain_wait_round(rag, pipeline_status, lock, progress)
            async with lock:
                assert not pipeline_status.get("recovery_required")
        finally:
            await rag.finalize_storages()

    @pytest.mark.asyncio
    async def test_a_set_that_changed_since_the_poll_is_not_fenced(self, tmp_path):
        """A producer releasing between poll and fence write must NOT be fenced."""
        rag = await _build_rag(tmp_path, extract=None, workspace_prefix="stall")
        try:
            pipeline_status, lock = await pipeline_status_handles(rag)
            finishing = uuid4().hex
            await self._reserve(pipeline_status, lock, finishing)
            async with lock:
                observed = dict(pipeline_status.get("pending_enqueue_tokens") or {})
            assert set(observed) == {finishing}

            await release_token_set_reservation(
                rag.workspace, tokens_key="pending_enqueue_tokens", token=finishing
            )

            await rag._fence_stalled_enqueue_drain(observed, pipeline_status, lock)

            async with lock:
                assert not pipeline_status.get("recovery_required")
                assert pipeline_status["pending_enqueues"] == 0
        finally:
            await rag.finalize_storages()

    @pytest.mark.asyncio
    async def test_a_shrink_since_the_poll_is_not_fenced_either(self, tmp_path):
        """When one of several producers finishes, the shrink resets evidence."""
        rag = await _build_rag(tmp_path, extract=None, workspace_prefix="stall")
        try:
            pipeline_status, lock = await pipeline_status_handles(rag)
            slow = uuid4().hex
            finishing = uuid4().hex
            await self._reserve(pipeline_status, lock, slow)
            await self._reserve(pipeline_status, lock, finishing)
            async with lock:
                observed = dict(pipeline_status.get("pending_enqueue_tokens") or {})

            progress = _ManualDrainProgress()
            for _ in range(_STALL_ROUNDS - 1):
                await self._drain_wait_round(rag, pipeline_status, lock, progress)

            await release_token_set_reservation(
                rag.workspace, tokens_key="pending_enqueue_tokens", token=finishing
            )
            await rag._fence_stalled_enqueue_drain(observed, pipeline_status, lock)

            async with lock:
                assert not pipeline_status.get("recovery_required")
                assert set(pipeline_status["pending_enqueue_tokens"]) == {slow}

            for _ in range(_STALL_ROUNDS - 1):
                await self._drain_wait_round(rag, pipeline_status, lock, progress)
            async with lock:
                assert not pipeline_status.get("recovery_required")
            with pytest.raises(PipelineRecoveryRequiredError, match="drain stalled"):
                await self._drain_wait_round(rag, pipeline_status, lock, progress)
        finally:
            await rag.finalize_storages()

    @pytest.mark.asyncio
    async def test_the_fence_message_carries_no_token_or_pid(self, tmp_path):
        """Public recovery/history surfaces must not leak reservation tokens or PIDs."""
        rag = await _build_rag(tmp_path, extract=None, workspace_prefix="stall")
        try:
            pipeline_status, lock = await pipeline_status_handles(rag)
            token = uuid4().hex
            await self._reserve(pipeline_status, lock, token)
            progress = _ManualDrainProgress()

            for _ in range(_STALL_ROUNDS - 1):
                await self._drain_wait_round(rag, pipeline_status, lock, progress)
            with pytest.raises(PipelineRecoveryRequiredError) as excinfo:
                await self._drain_wait_round(rag, pipeline_status, lock, progress)

            async with lock:
                fence = pipeline_status.get("recovery_required")
                latest = pipeline_status.get("latest_message") or ""
                history = " ".join(pipeline_status.get("history_messages") or [])
            pid = str(os.getpid())
            for surface in (fence["message"], latest, history, str(excinfo.value)):
                assert token not in surface
                assert pid not in surface
            assert token in fence["operation_record"]["scope"]
        finally:
            await rag.finalize_storages()

    @pytest.mark.asyncio
    async def test_the_fence_message_says_a_repair_guard_is_not_dropped(self, tmp_path):
        """force_reset keeps a source-conflict repair guard; the message must say so."""
        rag = await _build_rag(tmp_path, extract=None, workspace_prefix="stall")
        try:
            pipeline_status, lock = await pipeline_status_handles(rag)
            await self._reserve(pipeline_status, lock, uuid4().hex)
            await self._reserve(
                pipeline_status,
                lock,
                f"source-repair-{uuid4().hex}",
                kind=SOURCE_REPAIR_RESERVATION_KIND,
            )
            progress = _ManualDrainProgress()

            for _ in range(_STALL_ROUNDS - 1):
                await self._drain_wait_round(rag, pipeline_status, lock, progress)
            with pytest.raises(PipelineRecoveryRequiredError):
                await self._drain_wait_round(rag, pipeline_status, lock, progress)

            async with lock:
                message = pipeline_status.get("recovery_required")["message"]
            assert "1 upload/insert, 1 source-conflict repair guard(s)" in message
            assert "does NOT drop" in message
            assert "restart the process holding it" in message
            assert "AND drops these reservations" not in message
        finally:
            await rag.finalize_storages()
