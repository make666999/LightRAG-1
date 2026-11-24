"""Consolidated enqueue deduplication, identity, and metadata contract tests.

Covers:
  * Strict basename source resolution and conflict refusal during enqueue
    (``resolve_doc_source_strict`` / ``resolve_existing_doc_source``).
  * Immutable ``doc_status.created_at`` stamping and scheduler ordering.
  * Direct-enqueue seeding of ``DOCX_SMART_HEADING`` for pending-parse vs raw
    documents.
  * Duplicate-document retrieval by the duplicate upload's own ``track_id``
    while preserving ``original_doc_id`` / ``original_track_id`` metadata.
  * Persistence of source-conflict demotion (``metadata.is_duplicate``) across
    manual FAILED->PENDING retries and normal pipeline status transitions.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lightrag import LightRAG
from lightrag.base import (
    CURSOR_START,
    DocStatus,
    SourceAbsent,
    SourceConflict,
    SourceUnique,
)
from lightrag.constants import FULL_DOCS_FORMAT_PENDING_PARSE, FULL_DOCS_FORMAT_RAW
from lightrag.exceptions import StorageControlPlaneError
from lightrag.kg.shared_storage import (
    finalize_share_data,
    initialize_share_data,
    make_owner_record,
)
from lightrag.parser.registry import parser_specs_snapshot
from lightrag.pipeline import _BatchRunContext
from lightrag.utils import compute_mdhash_id
from lightrag.utils_pipeline import resolve_existing_doc_source

from ._fakes import (
    build_initialized_rag,
    pipeline_status_handles,
    single_chunk_chunking,
)

pytestmark = pytest.mark.offline

_CONFLICT_SOURCE = "collision.pdf"
_OWNER_TOKEN = "owner-A"


@pytest.fixture(autouse=True)
def _shared():
    initialize_share_data()
    yield
    finalize_share_data()


async def _build_rag(tmp_path: Path, **kwargs) -> LightRAG:
    defaults = {
        "workspace_prefix": "enq",
        "max_parallel_insert": 1,
    }
    defaults.update(kwargs)
    return await build_initialized_rag(tmp_path, **defaults)


async def _rows(rag: LightRAG) -> dict:
    return dict(rag.doc_status._data)


def _doc_id(name: str) -> str:
    return compute_mdhash_id(name, prefix="doc-")


# ---------------------------------------------------------------------------
# Strict basename source resolution during enqueue
# ---------------------------------------------------------------------------


class _RaisingDocStatus:
    """Stands in for a backend whose identity query fails."""

    def __init__(self):
        self.calls = 0

    async def resolve_doc_source_strict(self, canonical_source_key):
        self.calls += 1
        raise StorageControlPlaneError("backend down")


@pytest.mark.asyncio
async def test_resolver_treats_unknown_source_as_absent_without_querying():
    doc_status = _RaisingDocStatus()
    assert isinstance(
        await resolve_existing_doc_source(doc_status, "unknown_source"), SourceAbsent
    )
    assert doc_status.calls == 0


@pytest.mark.asyncio
async def test_enqueue_refuses_when_the_identity_query_fails(tmp_path, monkeypatch):
    """Fix-proof: the swallowed error used to look like "not a duplicate", so the
    document was admitted and a second row for an existing filename was written.
    Enqueue must now fail instead of admitting it."""
    rag = await _build_rag(tmp_path, workspace_prefix="basename")
    try:

        async def _boom(canonical_source_key):
            raise StorageControlPlaneError("doc_status unavailable")

        monkeypatch.setattr(rag.doc_status, "resolve_doc_source_strict", _boom)

        with pytest.raises(StorageControlPlaneError):
            await rag.apipeline_enqueue_documents(
                "hello world", file_paths="report.pdf", track_id="t-1"
            )

        # Nothing was admitted on the failure.
        assert await _rows(rag) == {}
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_enqueue_marks_a_unique_existing_basename_as_duplicate(tmp_path):
    rag = await _build_rag(tmp_path, workspace_prefix="basename")
    try:
        await rag.apipeline_enqueue_documents(
            "hello world", file_paths="report.pdf", track_id="t-1"
        )
        original_id = next(iter(await _rows(rag)))

        await rag.apipeline_enqueue_documents(
            "different body", file_paths="report.pdf", track_id="t-2"
        )
        rows = await _rows(rag)
        dup_rows = {k: v for k, v in rows.items() if k.startswith("dup-")}
        assert len(dup_rows) == 1
        (dup,) = dup_rows.values()
        assert dup["status"] == DocStatus.FAILED
        assert dup["metadata"]["duplicate_kind"] == "filename"
        assert dup["metadata"]["original_doc_id"] == original_id
        assert "File name already exists" in dup["error_msg"]
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_enqueue_refuses_an_unresolved_basename_conflict(tmp_path, monkeypatch):
    """A historical collision leaves several primaries on one basename. Attaching
    the new document to an arbitrary one hides the collision, so enqueue refuses
    and records WHY, with the bounded candidate sample."""
    rag = await _build_rag(tmp_path, workspace_prefix="basename")
    try:

        async def _conflict(canonical_source_key):
            return SourceConflict(
                candidate_count=2, sample_doc_ids=("doc-old-a", "doc-old-b")
            )

        monkeypatch.setattr(rag.doc_status, "resolve_doc_source_strict", _conflict)

        await rag.apipeline_enqueue_documents(
            "hello world", file_paths="report.pdf", track_id="t-1"
        )

        rows = await _rows(rag)
        # Refused: only the trackable conflict record, no PENDING work.
        assert all(k.startswith("dup-") for k in rows), rows
        assert len(rows) == 1
        (record,) = rows.values()
        assert record["status"] == DocStatus.FAILED
        assert record["metadata"]["duplicate_kind"] == "filename_conflict"
        # No original is asserted; the sample is surfaced instead.
        assert "doc-old-a" in record["error_msg"]
        assert "doc-old-b" in record["error_msg"]
        assert "Repair the conflict by doc id" in record["error_msg"]
        assert "repair required" in record["content_summary"]
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_enqueue_uses_the_strict_resolver_not_the_legacy_lookup(
    tmp_path, monkeypatch
):
    """Guard against a regression to the best-effort path: the legacy
    basename lookup must not be what decides the duplicate."""
    rag = await _build_rag(tmp_path, workspace_prefix="basename")
    try:
        seen = {"strict": 0, "legacy": 0}
        original = rag.doc_status.resolve_doc_source_strict

        async def _counting_strict(canonical_source_key):
            seen["strict"] += 1
            return await original(canonical_source_key)

        async def _counting_legacy(basename):
            seen["legacy"] += 1
            return None

        monkeypatch.setattr(
            rag.doc_status, "resolve_doc_source_strict", _counting_strict
        )
        monkeypatch.setattr(
            rag.doc_status, "get_doc_by_file_basename", _counting_legacy
        )

        await rag.apipeline_enqueue_documents(
            "hello world", file_paths="report.pdf", track_id="t-1"
        )

        assert seen["strict"] >= 1
        assert seen["legacy"] == 0
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_unique_resolution_reuses_the_page_projection(tmp_path):
    """SourceUnique already carries status/track_id, so the duplicate record is
    built without a second read of the original row."""
    rag = await _build_rag(tmp_path, workspace_prefix="basename")
    try:
        await rag.apipeline_enqueue_documents(
            "hello world", file_paths="report.pdf", track_id="t-orig"
        )
        resolution = await rag.doc_status.resolve_doc_source_strict("report.pdf")
        assert isinstance(resolution, SourceUnique)
        assert resolution.doc.track_id == "t-orig"
        assert resolution.doc.status is DocStatus.PENDING
    finally:
        await rag.finalize_storages()


# ---------------------------------------------------------------------------
# doc_status.created_at first-persistence timestamp & scheduler ordering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_created_at_is_the_first_persistence_time(tmp_path):
    rag = await _build_rag(
        tmp_path, workspace_prefix="ca", chunking_func=single_chunk_chunking
    )
    try:
        before = datetime.now(timezone.utc)
        await rag.apipeline_enqueue_documents(input="body", file_paths="now.txt")
        after = datetime.now(timezone.utc)

        row = await rag.doc_status.get_by_id(_doc_id("now.txt"))
        created = datetime.fromisoformat(row["created_at"])
        assert before <= created <= after
        assert row["updated_at"] >= row["created_at"]
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_created_at_records_enqueue_order_for_the_scheduler(tmp_path):
    """After scan has mtime-sorted its candidates, sequential persistence fixes
    that order into the ordinary ``(created_at, id)`` scheduling key."""
    rag = await _build_rag(
        tmp_path, workspace_prefix="ca", chunking_func=single_chunk_chunking
    )
    try:
        arrivals = ["oldest.txt", "middle.txt", "newest.txt"]
        for name in arrivals:
            await rag.apipeline_enqueue_documents(
                input=f"body of {name}",
                file_paths=name,
            )
            # Make the assertion independent of platform clock resolution.
            await asyncio.sleep(0.001)

        page = await rag.doc_status.get_docs_by_statuses_page(
            [DocStatus.PENDING], limit=10
        )
        assert [record.file_path for record in page.docs.values()] == arrivals
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_created_at_never_reaches_full_docs(tmp_path):
    """Creation/scheduling metadata belongs to doc_status alone."""
    rag = await _build_rag(
        tmp_path, workspace_prefix="ca", chunking_func=single_chunk_chunking
    )
    try:
        await rag.apipeline_enqueue_documents(input="body", file_paths="only.txt")
        row = await rag.full_docs.get_by_id(_doc_id("only.txt"))
        assert row is not None
        assert "created_at" not in row
    finally:
        await rag.finalize_storages()


# ---------------------------------------------------------------------------
# Direct-enqueue DOCX_SMART_HEADING seeding
# ---------------------------------------------------------------------------


async def _enqueue_and_read_engines(
    tmp_path: Path,
    *,
    parse_engine: list[str],
    file_paths: list[str],
    docs_format: str = FULL_DOCS_FORMAT_PENDING_PARSE,
    input_docs: list[str] | None = None,
) -> list[str | None]:
    """Enqueue docs and return each persisted full_docs parse_engine."""
    rag = await _build_rag(
        tmp_path,
        workspace_prefix="enqueue-seed",
        subdir=None,
        embedding_dim=32,
        max_token_size=4096,
    )
    try:
        await rag.apipeline_enqueue_documents(
            input_docs if input_docs is not None else [""] * len(file_paths),
            docs_format=docs_format,
            parse_engine=parse_engine,
            file_paths=file_paths,
        )
        engines = []
        for name in file_paths:
            row = await rag.full_docs.get_by_id(_doc_id(name))
            assert row is not None, f"full_docs row missing for {name}"
            engines.append(row.get("parse_engine"))
        return engines
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_direct_enqueue_seeds_bare_native_docx(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCX_SMART_HEADING", "true")
    engines = await _enqueue_and_read_engines(
        tmp_path,
        parse_engine=["native", "native(smart_heading=false)", "native"],
        file_paths=["seeded.docx", "optout.docx", "notes.md"],
    )
    assert engines == [
        "native(smart_heading=true)",  # bare native on .docx gets the seed
        "native(smart_heading=false)",  # explicit param stays the opt-out
        "native",  # non-docx never carries the seed
    ]


@pytest.mark.asyncio
async def test_direct_enqueue_raw_docx_metadata_not_seeded(tmp_path, monkeypatch):
    """RAW enqueue: parse_engine records the engine that ALREADY extracted
    the content — no docx parser will run on this doc, so the seed must not
    rewrite the metadata even for a .docx source with the switch on."""
    monkeypatch.setenv("DOCX_SMART_HEADING", "true")
    engines = await _enqueue_and_read_engines(
        tmp_path,
        docs_format=FULL_DOCS_FORMAT_RAW,
        input_docs=["Already extracted body text with enough words."],
        parse_engine=["native"],
        file_paths=["extracted.docx"],
    )
    assert engines == ["native"]


@pytest.mark.asyncio
async def test_direct_enqueue_switch_off_keeps_bare_native(tmp_path, monkeypatch):
    monkeypatch.delenv("DOCX_SMART_HEADING", raising=False)
    engines = await _enqueue_and_read_engines(
        tmp_path,
        parse_engine=["native"],
        file_paths=["plain.docx"],
    )
    assert engines == ["native"]


# ---------------------------------------------------------------------------
# Duplicate track_id retrieval (issue #2468)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_body", "first_path", "second_body", "second_path", "dup_kind"),
    [
        (
            "the original body",
            "report.txt",
            "different body, same file name",
            "report.txt",
            "filename",
        ),
        (
            "identical content, different file names",
            "first.txt",
            "identical content, different file names",
            "second.txt",
            "content_hash",
        ),
    ],
    ids=["filename", "content_hash"],
)
async def test_duplicate_is_retrievable_by_its_own_track_id(
    tmp_path,
    first_body,
    first_path,
    second_body,
    second_path,
    dup_kind,
):
    """Each of the two uploads must be answerable under the track_id ITS OWN
    call was handed, and the duplicate must name the document it collided with."""
    rag = await _build_rag(tmp_path, workspace_prefix="trk")
    try:
        original_track_id = "track-original"
        dup_track_id = f"track-dup-{dup_kind}"

        await rag.apipeline_enqueue_documents(
            input=first_body,
            file_paths=first_path,
            track_id=original_track_id,
        )
        await rag.apipeline_enqueue_documents(
            input=second_body,
            file_paths=second_path,
            track_id=dup_track_id,
        )

        original_docs = await rag.aget_docs_by_track_id(original_track_id)
        assert len(original_docs) == 1, (
            "the original document's track_id must answer with only its own record "
            "— a duplicate stamped with the original's track_id lands here instead"
        )
        original_doc_id, original_status = next(iter(original_docs.items()))
        assert original_status.track_id == original_track_id
        assert original_status.status == DocStatus.PENDING

        dup_docs = await rag.aget_docs_by_track_id(dup_track_id)
        assert len(dup_docs) == 1, (
            "the duplicate upload's own track_id must resolve to its "
            "FAILED duplicate record, not come back empty"
        )
        (dup_status,) = dup_docs.values()
        assert dup_status.track_id == dup_track_id
        assert dup_status.status == DocStatus.FAILED
        assert dup_status.metadata.get("is_duplicate") is True
        assert dup_status.metadata.get("duplicate_kind") == dup_kind
        # The displaced original identity: without these the client reaches its own
        # record but can no longer name what it collided with.
        assert dup_status.metadata.get("original_doc_id") == original_doc_id
        assert dup_status.metadata.get("original_track_id") == original_track_id
    finally:
        await rag.finalize_storages()


# ---------------------------------------------------------------------------
# Source-conflict demotion survives retries and transitions (LR2 §5.5)
# ---------------------------------------------------------------------------


def _primary_candidate_row(doc_id: str, status: DocStatus, seconds: int) -> dict:
    """A primary-candidate row: real content, no duplicate marker."""
    stamp = datetime(2026, 1, 1, 0, 0, seconds, tzinfo=timezone.utc).isoformat()
    return {
        "status": status,
        "content_summary": f"summary-{doc_id}",
        "content_length": 12,
        "chunks_count": 0,
        "chunks_list": [],
        "created_at": stamp,
        "updated_at": stamp,
        "file_path": _CONFLICT_SOURCE,
        "track_id": "t",
        "metadata": {"process_options": "R!"},
    }


async def _seed_conflict(rag: LightRAG, loser_status: DocStatus) -> tuple[str, str]:
    """Two primaries on one canonical source, then repair keeping ``doc-keep``."""
    keep, loser = "doc-keep", "doc-loser"
    await rag.doc_status.upsert(
        {
            keep: _primary_candidate_row(keep, DocStatus.PROCESSED, 1),
            loser: _primary_candidate_row(loser, loser_status, 2),
        }
    )
    # The reset probes full_docs for content; a demoted row keeps its own.
    await rag.full_docs.upsert(
        {
            keep: {"content": "keep body", "file_path": _CONFLICT_SOURCE},
            loser: {"content": "loser body", "file_path": _CONFLICT_SOURCE},
        }
    )
    assert isinstance(
        await rag.doc_status.resolve_doc_source_strict(_CONFLICT_SOURCE), SourceConflict
    )

    dry = await rag.doc_status.repair_source_conflict(
        _CONFLICT_SOURCE,
        primary_doc_id=keep,
        expected_candidate_count=0,
        expected_candidate_fingerprint="",
        dry_run=True,
    )
    committed = await rag.doc_status.repair_source_conflict(
        _CONFLICT_SOURCE,
        primary_doc_id=keep,
        expected_candidate_count=dry.candidate_count,
        expected_candidate_fingerprint=dry.fingerprint,
        dry_run=False,
    )
    assert committed.committed is True
    resolution = await rag.doc_status.resolve_doc_source_strict(_CONFLICT_SOURCE)
    assert isinstance(resolution, SourceUnique) and resolution.doc_id == keep
    return keep, loser


async def _busy_status_handles(rag: LightRAG, token: str):
    status, lock = await pipeline_status_handles(rag)
    status["history_messages"] = []
    status.update({"busy": True, "busy_owner": make_owner_record(token, "processing")})
    return status, lock


@pytest.mark.asyncio
async def test_a_manual_failed_retry_does_not_undo_the_repair(tmp_path):
    """Fix-proof: the operator repairs the conflict, then anyone clicking "retry
    failed documents" (or running /scan, which drives the same reset) handed the
    canonical source back to the demoted row."""
    rag = await _build_rag(tmp_path, workspace_prefix="demote")
    try:
        keep, loser = await _seed_conflict(rag, DocStatus.FAILED)
        status, lock = await _busy_status_handles(rag, _OWNER_TOKEN)

        docs, _ = await rag._next_failed_page(CURSOR_START)
        assert loser in docs
        reset = await rag._reset_failed_page(docs, _OWNER_TOKEN, status, lock)
        assert reset == 1  # it really was reset, not skipped

        row = await rag.doc_status.get_by_id(loser)
        assert row["status"] == DocStatus.PENDING  # retried, as asked
        assert row["metadata"]["is_duplicate"] is True  # but still demoted
        assert row["metadata"]["original_doc_id"] == keep
        assert row["metadata"]["process_options"] == "R!"  # directives kept

        resolution = await rag.doc_status.resolve_doc_source_strict(_CONFLICT_SOURCE)
        assert isinstance(resolution, SourceUnique)
        assert resolution.doc_id == keep
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_a_status_transition_does_not_undo_the_repair(tmp_path):
    """The wider half: a demoted row in any non-terminal status is resumed by the
    ordinary AUTO sweep, and its first transition rebuilt metadata from the
    carry-over whitelist — so the demotion did not even need a manual retry to
    disappear."""
    rag = await _build_rag(tmp_path, workspace_prefix="demote")
    try:
        keep, loser = await _seed_conflict(rag, DocStatus.PENDING)
        status, lock = await _busy_status_handles(rag, _OWNER_TOKEN)

        hydrated = await rag.doc_status.get_full_docs_by_ids([loser], strict=True)
        ctx = _BatchRunContext(
            pipeline_status=status,
            pipeline_status_lock=lock,
            semaphore=asyncio.Semaphore(1),
            total_files=1,
            parse_queues={"native": asyncio.Queue()},
            parser_specs=parser_specs_snapshot(),
            q_analyze=asyncio.Queue(),
            q_process=asyncio.Queue(),
            run_owner_token=_OWNER_TOKEN,
        )

        await rag._upsert_doc_status_transition(
            doc_id=loser,
            status=DocStatus.PROCESSED,
            status_doc=hydrated[loser],
            file_path=_CONFLICT_SOURCE,
            ctx=ctx,
        )

        row = await rag.doc_status.get_by_id(loser)
        assert row["status"] == DocStatus.PROCESSED
        assert row["metadata"]["is_duplicate"] is True
        assert row["metadata"]["original_doc_id"] == keep

        resolution = await rag.doc_status.resolve_doc_source_strict(_CONFLICT_SOURCE)
        assert isinstance(resolution, SourceUnique)
        assert resolution.doc_id == keep
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_the_repair_tool_cannot_undo_a_resurrection_itself(tmp_path):
    """Why the two tests above are load-bearing rather than cosmetic: there is no
    supported way back. ``repair_source_conflict`` refuses a primary that is not a
    current candidate, so once a demoted row is re-promoted the only exit is
    repairing the conflict again from scratch."""
    rag = await _build_rag(tmp_path, workspace_prefix="demote")
    try:
        _keep, loser = await _seed_conflict(rag, DocStatus.FAILED)
        with pytest.raises(ValueError):
            await rag.doc_status.repair_source_conflict(
                _CONFLICT_SOURCE,
                primary_doc_id=loser,  # demoted: no longer a candidate
                expected_candidate_count=1,
                expected_candidate_fingerprint="whatever",
                dry_run=True,
            )
    finally:
        await rag.finalize_storages()
