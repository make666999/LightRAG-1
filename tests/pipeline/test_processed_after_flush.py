"""PROCESSED must be the commit record (issue #3400, Phase 2).

``_process_document`` historically upserted ``doc_status=PROCESSED`` (which
doc-status backends persist immediately) BEFORE ``_insert_done()`` flushed the
graph/vector/chunk stores — so a crash or flush failure in that window left a
durably-PROCESSED document whose data never hit disk.

These tests drive the real pipeline (JSON storages, offline) and assert the
reordered semantics:

- happy path: the PROCESSED upsert happens strictly AFTER the merge-stage
  data flush;
- flush failure: NO PROCESSED write ever reaches doc_status — the document
  goes straight to FAILED.
"""

from __future__ import annotations

import pytest

from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.utils import compute_mdhash_id

from ._fakes import build_initialized_rag, single_chunk_chunking

pytestmark = pytest.mark.offline


async def _build_rag(tmp_path) -> LightRAG:
    return await build_initialized_rag(
        tmp_path,
        workspace_prefix="flushorder",
        chunking_func=single_chunk_chunking,
        max_parallel_insert=1,
    )


def _status_value(status: object) -> str:
    if isinstance(status, DocStatus):
        return status.value
    return str(status).replace("DocStatus.", "").lower()


@pytest.mark.asyncio
async def test_processed_upsert_happens_after_data_flush(tmp_path, monkeypatch):
    rag = await _build_rag(tmp_path)
    try:
        await rag.apipeline_enqueue_documents(input="order doc", file_paths="order.txt")
        doc_id = compute_mdhash_id("order.txt", prefix="doc-")

        events: list[str] = []

        orig_insert_done = rag._insert_done

        async def spy_insert_done(*a, **k):
            events.append("insert_done")
            await orig_insert_done(*a, **k)

        monkeypatch.setattr(rag, "_insert_done", spy_insert_done)

        orig_upsert = rag.doc_status.upsert

        async def spy_upsert(data):
            for rec_id, rec in data.items():
                if rec_id == doc_id:
                    events.append(f"status:{_status_value(rec.get('status'))}")
            await orig_upsert(data)

        monkeypatch.setattr(rag.doc_status, "upsert", spy_upsert)

        await rag.apipeline_process_enqueue_documents()

        doc_status = await rag.doc_status.get_by_id(doc_id)
        assert _status_value(doc_status["status"]) == "processed"

        processed_idx = events.index("status:processed")
        flush_indexes = [i for i, e in enumerate(events) if e == "insert_done"]
        assert flush_indexes, f"no _insert_done recorded: {events}"
        # The merge-stage flush precedes the PROCESSED commit record.
        assert any(i < processed_idx for i in flush_indexes), (
            f"PROCESSED (at {processed_idx}) was written before any data "
            f"flush: {events}"
        )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_flush_failure_never_writes_processed(tmp_path, monkeypatch):
    """With the merge-stage flush failing, doc_status must never see a
    PROCESSED write for the document — not even transiently (the old order
    durably wrote PROCESSED first and only later corrected it to FAILED)."""
    rag = await _build_rag(tmp_path)
    try:
        await rag.apipeline_enqueue_documents(input="fail doc", file_paths="fail.txt")
        doc_id = compute_mdhash_id("fail.txt", prefix="doc-")

        statuses_written: list[str] = []
        orig_upsert = rag.doc_status.upsert

        async def spy_upsert(data):
            for rec_id, rec in data.items():
                if rec_id == doc_id:
                    statuses_written.append(_status_value(rec.get("status")))
            await orig_upsert(data)

        monkeypatch.setattr(rag.doc_status, "upsert", spy_upsert)

        async def flush_boom():
            raise RuntimeError("vdb flush boom")

        monkeypatch.setattr(rag.entities_vdb, "index_done_callback", flush_boom)

        await rag.apipeline_process_enqueue_documents()

        doc_status = await rag.doc_status.get_by_id(doc_id)
        assert _status_value(doc_status["status"]) == "failed"
        assert "processed" not in statuses_written, (
            f"a PROCESSED write reached doc_status despite the flush "
            f"failure: {statuses_written}"
        )
    finally:
        await rag.finalize_storages()


def _retain_tracking(monkeypatch, rag, answers: list[bool]) -> list[int]:
    """Make ``entity_chunks`` report a write its commit kept (the shape of an
    ``OpenSearchKVStorage`` 429) for as many asks as ``answers`` holds, then
    report it landed. Returns the ask counter."""
    monkeypatch.setattr(
        type(rag.entity_chunks), "buffers_writes_per_process", True, raising=False
    )
    monkeypatch.setattr("lightrag.lightrag._TRACKING_RETAINED_FLUSH_BACKOFF_SECONDS", 0)
    asks = [0]

    async def has_retained_writes():
        asks[0] += 1
        return answers.pop(0) if answers else False

    monkeypatch.setattr(rag.entity_chunks, "has_retained_writes", has_retained_writes)
    return asks


@pytest.mark.asyncio
async def test_retained_tracking_write_never_writes_processed(tmp_path, monkeypatch):
    """A tracking write the flush kept would land after ``busy`` is released,
    over a newer commit. The document must fail like a raised flush."""
    from lightrag.lightrag import _TRACKING_RETAINED_FLUSH_RETRIES

    rag = await _build_rag(tmp_path)
    try:
        await rag.apipeline_enqueue_documents(input="keep doc", file_paths="keep.txt")
        doc_id = compute_mdhash_id("keep.txt", prefix="doc-")
        asks = _retain_tracking(
            monkeypatch, rag, [True] * (_TRACKING_RETAINED_FLUSH_RETRIES + 1)
        )

        statuses_written: list[str] = []
        orig_upsert = rag.doc_status.upsert

        async def spy_upsert(data):
            for rec_id, rec in data.items():
                if rec_id == doc_id:
                    statuses_written.append(_status_value(rec.get("status")))
            await orig_upsert(data)

        monkeypatch.setattr(rag.doc_status, "upsert", spy_upsert)

        await rag.apipeline_process_enqueue_documents()

        doc_status = await rag.doc_status.get_by_id(doc_id)
        assert _status_value(doc_status["status"]) == "failed"
        assert "processed" not in statuses_written, statuses_written
        assert "not made durable" in doc_status["error_msg"]
        assert asks[0] == _TRACKING_RETAINED_FLUSH_RETRIES + 1
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_retained_tracking_write_that_lands_on_reflush_is_processed(
    tmp_path, monkeypatch
):
    rag = await _build_rag(tmp_path)
    try:
        await rag.apipeline_enqueue_documents(input="heal doc", file_paths="heal.txt")
        doc_id = compute_mdhash_id("heal.txt", prefix="doc-")
        asks = _retain_tracking(monkeypatch, rag, [True])

        flushes = [0]
        orig_flush = rag.entity_chunks.index_done_callback

        async def counting_flush():
            flushes[0] += 1
            return await orig_flush()

        monkeypatch.setattr(rag.entity_chunks, "index_done_callback", counting_flush)

        await rag.apipeline_process_enqueue_documents()

        doc_status = await rag.doc_status.get_by_id(doc_id)
        assert _status_value(doc_status["status"]) == "processed"
        assert asks[0] == 2  # retained, then landed after one re-flush
        assert flushes[0] >= 2  # the document's commit plus the re-flush
    finally:
        await rag.finalize_storages()
