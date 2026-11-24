"""Consolidated ``doc_status`` metadata, hydration, truncation, and source-preservation tests.

Covers:
  * Directive-only metadata cleanup when (re)queuing to PENDING
    (``doc_status_reset_metadata``, ``doc_status_metadata_has_attempt_fields``,
    ``doc_status_transition_metadata``, and ``_validate_and_fix_document_consistency``).
  * Tolerant rehydration via ``DocProcessingStatus.from_stored`` (ignoring
    undeclared fields from external producers while still validating required ones).
  * Durable per-document LLM truncation tallies (``metadata.llm_truncation``)
    across extraction, multimodal analysis hand-off, transitions, and resets.
  * Deferred parser source preservation until parsing completes.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from lightrag import LightRAG
from lightrag.base import DocProcessingStatus, DocStatus
from lightrag.constants import (
    KG_PURGE_METADATA_KEY,
    KG_PURGE_PHASE_ANCHORS_PENDING,
    KG_WRITE_STATE_GRAPH_MUTATION_STARTED,
    KG_WRITE_STATE_METADATA_KEY,
    PARSED_DIR_NAME,
)
from lightrag.utils import LLM_TRUNCATION_METADATA_KEY, TruncatedResponse
from lightrag.utils_pipeline import (
    _DOC_STATUS_METADATA_ATTEMPT_KEYS,
    doc_status_metadata_has_attempt_fields,
    doc_status_reset_metadata,
    doc_status_transition_metadata,
)

from ._fakes import (
    build_delete_test_rag,
    build_initialized_rag,
    raw_single_chunk_chunking,
)

pytestmark = pytest.mark.offline


# A metadata blob carrying every per-attempt field a document could have
# accumulated across parse/analyze/process plus the two long-lived directives.
_FULL_ATTEMPT_METADATA = {
    "process_options": "iF",
    "source_file": "report.pdf",
    "parse_start_time": 100,
    "parse_end_time": 200,
    "parse_stage_skipped": True,
    "parse_warnings": {"docx": "missing paraId"},
    "analyzing_start_time": 300,
    "analyzing_end_time": 400,
    "analyzing_stage_skipped": True,
    "chunk_opts": "size=512, overlap=256",
    "process_start_time": 500,
    "process_end_time": 600,
    "parse_format": "raw",
    "parse_engine": "native",
    "chunk_method": "fixed_token",
    "skip_kg": True,
    "mm_chunks": 3,
    "hard_fallback_split": "256 -> 512",
}

_EXTRACTION_RESULT = "(entity<|#|>ALICE<|#|>person<|#|>An analyst)<|COMPLETE|>"


def _status_to_text(status: object) -> str:
    if isinstance(status, DocStatus):
        return status.value
    return str(status).replace("DocStatus.", "").lower()


async def _build_reset_rag(tmp_path: Path, test_name: str) -> LightRAG:
    return await build_initialized_rag(
        tmp_path,
        workspace=f"{test_name}_{uuid4().hex[:8]}",
        subdir=test_name,
        tokenizer_name="test-tokenizer",
        chunking_func=raw_single_chunk_chunking,
        max_parallel_insert=1,
    )


async def _build_truncation_rag(tmp_path: Path, llm_func) -> LightRAG:
    return await build_initialized_rag(
        tmp_path,
        workspace=f"truncation-{tmp_path.name}-{uuid4().hex[:8]}",
        subdir=None,
        llm_model_func=llm_func,
        embedding_dim=32,
        max_token_size=4096,
    )


async def _ingest_doc(rag: LightRAG, *, doc_id: str, body: str) -> dict:
    await rag.apipeline_enqueue_documents(
        body, ids=[doc_id], file_paths=f"{doc_id}.txt", track_id=f"track-{doc_id}"
    )
    await rag.apipeline_process_enqueue_documents()
    return await rag.doc_status.get_by_id(doc_id)


# ---------------------------------------------------------------------------
# doc_status_reset_metadata & transition metadata unit tests
# ---------------------------------------------------------------------------


def test_reset_metadata_keeps_directives_drops_attempt_fields():
    result = doc_status_reset_metadata({"metadata": dict(_FULL_ATTEMPT_METADATA)})
    assert result == {"process_options": "iF", "source_file": "report.pdf"}


def test_reset_metadata_normalizes_legacy_source_file_name():
    result = doc_status_reset_metadata(
        {"metadata": {"source_file_name": "legacy.docx", "process_options": "t"}}
    )
    assert result == {"process_options": "t", "source_file": "legacy.docx"}


def test_reset_metadata_prefers_new_source_file_over_legacy():
    result = doc_status_reset_metadata(
        {"metadata": {"source_file": "new.pdf", "source_file_name": "old.pdf"}}
    )
    assert result == {"source_file": "new.pdf"}


def test_reset_metadata_handles_empty_or_invalid_metadata():
    assert doc_status_reset_metadata({"metadata": {}}) == {}
    assert doc_status_reset_metadata({"metadata": None}) == {}
    assert doc_status_reset_metadata({}) == {}
    assert doc_status_reset_metadata(None) == {}
    # blank directive values are not carried
    assert (
        doc_status_reset_metadata(
            {"metadata": {"process_options": "", "source_file": ""}}
        )
        == {}
    )


def test_reset_metadata_keeps_kg_recovery_state():
    """The KG write marker and purge journal must survive a FAILED->PENDING
    reset (issue #3400 fail-closed purge)."""
    journal = {
        "schema_version": 1,
        "operation_id": "purge-abc",
        "phase": KG_PURGE_PHASE_ANCHORS_PENDING,
        "chunk_count": 2,
    }
    result = doc_status_reset_metadata(
        {
            "metadata": {
                **_FULL_ATTEMPT_METADATA,
                KG_WRITE_STATE_METADATA_KEY: KG_WRITE_STATE_GRAPH_MUTATION_STARTED,
                KG_PURGE_METADATA_KEY: journal,
            }
        }
    )

    assert result == {
        "process_options": "iF",
        "source_file": "report.pdf",
        KG_WRITE_STATE_METADATA_KEY: KG_WRITE_STATE_GRAPH_MUTATION_STARTED,
        KG_PURGE_METADATA_KEY: journal,
    }


def test_transition_metadata_carries_kg_recovery_state_and_drop_retires_it():
    """Carry-over keeps both keys across every status transition, and ``drop``
    is the only way to retire one."""
    status_doc = {
        "metadata": {
            "process_options": "iF",
            KG_WRITE_STATE_METADATA_KEY: KG_WRITE_STATE_GRAPH_MUTATION_STARTED,
            KG_PURGE_METADATA_KEY: {"phase": KG_PURGE_PHASE_ANCHORS_PENDING},
        }
    }

    carried = doc_status_transition_metadata(status_doc)
    assert carried[KG_WRITE_STATE_METADATA_KEY] == (
        KG_WRITE_STATE_GRAPH_MUTATION_STARTED
    )
    assert KG_PURGE_METADATA_KEY in carried

    # What the PROCESSED transition does: retire the journal, keep the marker.
    retired = doc_status_transition_metadata(status_doc, drop=(KG_PURGE_METADATA_KEY,))
    assert KG_PURGE_METADATA_KEY not in retired
    assert retired[KG_WRITE_STATE_METADATA_KEY] == (
        KG_WRITE_STATE_GRAPH_MUTATION_STARTED
    )
    assert retired["process_options"] == "iF"

    # The source metadata must not be mutated in place.
    assert KG_PURGE_METADATA_KEY in status_doc["metadata"]


def test_reset_metadata_drops_degraded_recovery_warnings():
    """A full reprocess is the repair boundary for prior degraded rollback."""
    assert (
        doc_status_reset_metadata(
            {
                "metadata": {
                    "kg_recovery_warnings": [{"code": "degraded_custom_chunk_rollback"}]
                }
            }
        )
        == {}
    )


def test_has_attempt_fields_trigger():
    assert doc_status_metadata_has_attempt_fields({"metadata": {"parse_end_time": 1}})
    assert doc_status_metadata_has_attempt_fields(
        {"metadata": {"process_options": "F", "chunk_opts": "x"}}
    )
    # only directives / custom non-attempt fields -> no trigger
    assert not doc_status_metadata_has_attempt_fields(
        {"metadata": {"process_options": "F", "source_file": "a.pdf"}}
    )
    assert not doc_status_metadata_has_attempt_fields(
        {"metadata": {"custom_field": "x"}}
    )
    assert not doc_status_metadata_has_attempt_fields({"metadata": {}})
    assert not doc_status_metadata_has_attempt_fields({"metadata": None})


# ---------------------------------------------------------------------------
# _validate_and_fix_document_consistency integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interrupted_reset_clears_attempt_metadata_in_storage_and_memory(
    tmp_path,
):
    rag = await _build_reset_rag(tmp_path, "reset_clears_attempt_metadata")
    try:
        doc_id = "doc-interrupted"
        now = datetime.now(timezone.utc).isoformat()
        await rag.full_docs.upsert(
            {doc_id: {"content": "interrupted doc", "file_path": "report.pdf"}}
        )
        await rag.doc_status.upsert(
            {
                doc_id: {
                    "status": DocStatus.PROCESSING,
                    "content_summary": "interrupted",
                    "content_length": 15,
                    "chunks_count": 1,
                    "chunks_list": ["c-1"],
                    "created_at": now,
                    "updated_at": now,
                    "file_path": "report.pdf",
                    "track_id": "track-1",
                    "error_msg": "old error",
                    "metadata": dict(_FULL_ATTEMPT_METADATA),
                }
            }
        )

        to_process_docs = await rag.doc_status.get_docs_by_statuses(
            [DocStatus.PROCESSING]
        )
        pipeline_status = {"latest_message": "", "history_messages": []}
        await rag._validate_and_fix_document_consistency(
            to_process_docs=to_process_docs,
            pipeline_status=pipeline_status,
            pipeline_status_lock=asyncio.Lock(),
        )

        # Storage: directives only, error cleared, chunks preserved.
        stored = await rag.doc_status.get_by_id(doc_id)
        assert stored is not None
        assert _status_to_text(stored["status"]) == "pending"
        assert stored["metadata"] == {
            "process_options": "iF",
            "source_file": "report.pdf",
        }
        assert stored["error_msg"] == ""
        assert stored["chunks_list"] == ["c-1"]

        # In-memory status_doc carried forward by workers is cleaned too.
        assert to_process_docs[doc_id].metadata == {
            "process_options": "iF",
            "source_file": "report.pdf",
        }
        assert _status_to_text(to_process_docs[doc_id].status) == "pending"
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_stale_pending_normalized_clean_pending_untouched(tmp_path):
    rag = await _build_reset_rag(tmp_path, "stale_pending_normalized")
    try:
        stale_id = "doc-stale-pending"
        clean_id = "doc-clean-pending"
        now = datetime.now(timezone.utc).isoformat()
        await rag.full_docs.upsert(
            {
                stale_id: {"content": "stale", "file_path": "stale.pdf"},
                clean_id: {"content": "clean", "file_path": "clean.pdf"},
            }
        )
        await rag.doc_status.upsert(
            {
                stale_id: {
                    "status": DocStatus.PENDING,
                    "content_summary": "stale",
                    "content_length": 5,
                    "chunks_count": 0,
                    "chunks_list": [],
                    "created_at": now,
                    "updated_at": now,
                    "file_path": "stale.pdf",
                    "track_id": "track-stale",
                    "error_msg": "",
                    "metadata": {
                        "process_options": "F",
                        "parse_end_time": 999,
                        "analyzing_start_time": 111,
                    },
                },
                clean_id: {
                    "status": DocStatus.PENDING,
                    "content_summary": "clean",
                    "content_length": 5,
                    "chunks_count": 0,
                    "chunks_list": [],
                    "created_at": now,
                    "updated_at": now,
                    "file_path": "clean.pdf",
                    "track_id": "track-clean",
                    "error_msg": "",
                    "metadata": {"process_options": "F", "source_file": "clean.pdf"},
                },
            }
        )

        to_process_docs = await rag.doc_status.get_docs_by_statuses([DocStatus.PENDING])
        pipeline_status = {"latest_message": "", "history_messages": []}
        await rag._validate_and_fix_document_consistency(
            to_process_docs=to_process_docs,
            pipeline_status=pipeline_status,
            pipeline_status_lock=asyncio.Lock(),
        )

        # Stale PENDING normalized to directives-only (storage + memory).
        stale_stored = await rag.doc_status.get_by_id(stale_id)
        assert stale_stored["metadata"] == {"process_options": "F"}
        assert to_process_docs[stale_id].metadata == {"process_options": "F"}

        # Clean PENDING left untouched: no rewrite (updated_at unchanged).
        clean_stored = await rag.doc_status.get_by_id(clean_id)
        assert clean_stored["metadata"] == {
            "process_options": "F",
            "source_file": "clean.pdf",
        }
        assert clean_stored["updated_at"] == now
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_pending_custom_metadata_semantic_lock(tmp_path):
    """Locks the chosen action semantic for already-PENDING docs:

    * only custom (non-attempt) metadata, no stale key -> left untouched;
    * stale key + custom field -> collapsed to directives-only (custom
      dropped), matching the interrupted-reset path.
    """
    rag = await _build_reset_rag(tmp_path, "pending_custom_metadata_lock")
    try:
        custom_only_id = "doc-custom-only"
        stale_plus_custom_id = "doc-stale-plus-custom"
        now = datetime.now(timezone.utc).isoformat()
        await rag.full_docs.upsert(
            {
                custom_only_id: {"content": "c", "file_path": "custom.pdf"},
                stale_plus_custom_id: {"content": "s", "file_path": "mixed.pdf"},
            }
        )
        await rag.doc_status.upsert(
            {
                custom_only_id: {
                    "status": DocStatus.PENDING,
                    "content_summary": "c",
                    "content_length": 1,
                    "chunks_count": 0,
                    "chunks_list": [],
                    "created_at": now,
                    "updated_at": now,
                    "file_path": "custom.pdf",
                    "track_id": "track-custom",
                    "error_msg": "",
                    "metadata": {"custom_field": "keep-me"},
                },
                stale_plus_custom_id: {
                    "status": DocStatus.PENDING,
                    "content_summary": "s",
                    "content_length": 1,
                    "chunks_count": 0,
                    "chunks_list": [],
                    "created_at": now,
                    "updated_at": now,
                    "file_path": "mixed.pdf",
                    "track_id": "track-mixed",
                    "error_msg": "",
                    "metadata": {
                        "process_options": "F",
                        "custom_field": "drop-me",
                        "parse_start_time": 42,
                    },
                },
            }
        )

        to_process_docs = await rag.doc_status.get_docs_by_statuses([DocStatus.PENDING])
        pipeline_status = {"latest_message": "", "history_messages": []}
        await rag._validate_and_fix_document_consistency(
            to_process_docs=to_process_docs,
            pipeline_status=pipeline_status,
            pipeline_status_lock=asyncio.Lock(),
        )

        custom_stored = await rag.doc_status.get_by_id(custom_only_id)
        assert custom_stored["metadata"] == {"custom_field": "keep-me"}
        assert custom_stored["updated_at"] == now

        mixed_stored = await rag.doc_status.get_by_id(stale_plus_custom_id)
        assert mixed_stored["metadata"] == {"process_options": "F"}
        assert "custom_field" not in mixed_stored["metadata"]
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_reset_and_normalize_with_string_status(tmp_path):
    """Non-JSON backends hydrate ``DocProcessingStatus.status`` as a raw string
    (``"processing"`` / ``"pending"``); pin that both clean paths still match."""
    rag = await _build_reset_rag(tmp_path, "reset_string_status")
    try:
        processing_id = "doc-string-processing"
        pending_id = "doc-string-pending"
        now = datetime.now(timezone.utc).isoformat()
        await rag.full_docs.upsert(
            {
                processing_id: {"content": "p", "file_path": "proc.pdf"},
                pending_id: {"content": "q", "file_path": "pend.pdf"},
            }
        )

        # Status is a plain string, exactly as a non-JSON backend hydrates it.
        to_process_docs = {
            processing_id: DocProcessingStatus(
                content_summary="p",
                content_length=1,
                file_path="proc.pdf",
                status="processing",
                created_at=now,
                updated_at=now,
                track_id="track-proc",
                chunks_count=1,
                chunks_list=["c-1"],
                error_msg="old error",
                metadata=dict(_FULL_ATTEMPT_METADATA),
            ),
            pending_id: DocProcessingStatus(
                content_summary="q",
                content_length=1,
                file_path="pend.pdf",
                status="pending",
                created_at=now,
                updated_at=now,
                track_id="track-pend",
                chunks_count=0,
                chunks_list=[],
                error_msg="",
                metadata={"process_options": "F", "analyzing_start_time": 7},
            ),
        }
        assert isinstance(to_process_docs[processing_id].status, str)
        assert not isinstance(to_process_docs[processing_id].status, DocStatus)

        pipeline_status = {"latest_message": "", "history_messages": []}
        await rag._validate_and_fix_document_consistency(
            to_process_docs=to_process_docs,
            pipeline_status=pipeline_status,
            pipeline_status_lock=asyncio.Lock(),
        )

        # String "processing" is reset to a clean PENDING.
        proc_stored = await rag.doc_status.get_by_id(processing_id)
        assert proc_stored is not None
        assert _status_to_text(proc_stored["status"]) == "pending"
        assert proc_stored["metadata"] == {
            "process_options": "iF",
            "source_file": "report.pdf",
        }
        assert to_process_docs[processing_id].metadata == {
            "process_options": "iF",
            "source_file": "report.pdf",
        }

        # String "pending" carrying a stale field is normalized in place.
        pend_stored = await rag.doc_status.get_by_id(pending_id)
        assert pend_stored is not None
        assert pend_stored["metadata"] == {"process_options": "F"}
        assert to_process_docs[pending_id].metadata == {"process_options": "F"}
    finally:
        await rag.finalize_storages()


# ---------------------------------------------------------------------------
# DocProcessingStatus.from_stored tolerant rehydration
# ---------------------------------------------------------------------------


def _valid_stored_row(**extra) -> dict:
    return {
        "content_summary": "Summary",
        "content_length": 100,
        "file_path": "doc.pdf",
        "status": DocStatus.PROCESSED,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        **extra,
    }


def test_undeclared_fields_are_ignored():
    # The RAG-Anything shape from HKUDS/RAG-Anything#73
    row = _valid_stored_row(
        scheme_name="default",
        multimodal_content=[{"type": "image"}],
        field_from_the_future="value",
    )

    status = DocProcessingStatus.from_stored(row)

    assert status.file_path == "doc.pdf"
    assert status.status == DocStatus.PROCESSED
    assert not hasattr(status, "scheme_name")
    assert not hasattr(status, "field_from_the_future")


def test_declared_fields_all_pass_through():
    row = _valid_stored_row(
        track_id="track_1",
        chunks_count=3,
        chunks_list=["c1", "c2", "c3"],
        error_msg=None,
        metadata={"k": "v"},
    )

    status = DocProcessingStatus.from_stored(row)

    assert status.track_id == "track_1"
    assert status.chunks_list == ["c1", "c2", "c3"]
    assert status.metadata == {"k": "v"}


def test_missing_required_field_still_raises_type_error():
    row = _valid_stored_row()
    del row["status"]

    with pytest.raises(TypeError):
        DocProcessingStatus.from_stored(row)


def test_post_init_status_conversion_still_applies():
    # __post_init__ downgrades PROCESSED to PREPROCESSED when multimodal
    # processing is pending; from_stored must not bypass it.
    row = _valid_stored_row(multimodal_processed=False)

    status = DocProcessingStatus.from_stored(row)

    assert status.status == DocStatus.PREPROCESSED


def test_input_row_is_not_mutated():
    row = _valid_stored_row(field_from_the_future="value")
    snapshot = dict(row)

    DocProcessingStatus.from_stored(row)

    assert row == snapshot


# ---------------------------------------------------------------------------
# Token-limit truncation metadata (issue #3601)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_truncated_run_records_the_summary_on_the_document(tmp_path):
    """The document still reaches PROCESSED, but it now SAYS its graph is partial."""

    async def _truncating_llm(prompt, **kwargs):
        return TruncatedResponse(_EXTRACTION_RESULT)

    rag = await _build_truncation_rag(tmp_path, _truncating_llm)
    try:
        status = await _ingest_doc(rag, doc_id="doc-trunc", body="Alice is an analyst.")
    finally:
        await rag.finalize_storages()

    assert status["status"] == DocStatus.PROCESSED
    summary = status["metadata"][LLM_TRUNCATION_METADATA_KEY]
    assert summary["events"] >= 1
    assert summary["affected"] == 1
    assert "initial" in summary["stages"]
    assert summary["samples"] == status["chunks_list"][:1]


@pytest.mark.asyncio
async def test_a_clean_run_leaves_no_truncation_record(tmp_path):
    async def _complete_llm(prompt, **kwargs):
        return _EXTRACTION_RESULT

    rag = await _build_truncation_rag(tmp_path, _complete_llm)
    try:
        status = await _ingest_doc(rag, doc_id="doc-clean", body="Alice is an analyst.")
    finally:
        await rag.finalize_storages()

    assert status["status"] == DocStatus.PROCESSED
    assert LLM_TRUNCATION_METADATA_KEY not in status["metadata"]


def test_the_summary_is_not_carried_over_to_the_next_attempt():
    """Carry-over would resurrect a stale summary on a re-run that fixed the
    budget: the PROCESSING transition must drop it, and only the terminal
    transition of the NEW attempt may write it back."""
    status_doc = {
        "metadata": {
            LLM_TRUNCATION_METADATA_KEY: {"events": 12, "affected": 12},
            "process_options": "R",
        }
    }

    carried = doc_status_transition_metadata(status_doc)

    assert LLM_TRUNCATION_METADATA_KEY not in carried
    assert carried["process_options"] == "R"


def test_a_reset_to_pending_drops_the_summary():
    status_doc = {
        "metadata": {
            LLM_TRUNCATION_METADATA_KEY: {"events": 12, "affected": 12},
            "process_options": "R",
        }
    }

    reset = doc_status_reset_metadata(status_doc)

    assert LLM_TRUNCATION_METADATA_KEY not in reset
    assert reset["process_options"] == "R"


def test_the_summary_marks_a_pending_row_as_carrying_stale_attempt_metadata():
    assert LLM_TRUNCATION_METADATA_KEY in _DOC_STATUS_METADATA_ATTEMPT_KEYS


@pytest.mark.asyncio
async def test_analyze_stage_truncation_reaches_the_processed_row(tmp_path):
    """Multimodal analysis truncations arrive on the analyze->process hand-off
    dict and must be merged into the terminal metadata alongside extraction."""

    async def _truncating_llm(prompt, **kwargs):
        return TruncatedResponse(_EXTRACTION_RESULT)

    rag = await _build_truncation_rag(tmp_path, _truncating_llm)
    original_analyze = rag.analyze_multimodal

    async def _analyze_with_truncation(*args, **kwargs):
        parsed_data = await original_analyze(*args, **kwargs)
        parsed_data["llm_truncation"] = {
            "events": 2,
            "affected": 2,
            "stages": {"multimodal": 2},
            "samples": ["drawings/im-001", "table/tb-001"],
        }
        return parsed_data

    rag.analyze_multimodal = _analyze_with_truncation
    try:
        status = await _ingest_doc(rag, doc_id="doc-mm", body="Alice is an analyst.")
    finally:
        await rag.finalize_storages()

    assert status["status"] == DocStatus.PROCESSED
    summary = status["metadata"][LLM_TRUNCATION_METADATA_KEY]
    assert summary["stages"]["multimodal"] == 2
    assert summary["stages"]["initial"] >= 1
    assert "drawings/im-001" in summary["samples"]


# ---------------------------------------------------------------------------
# Deferred parser source preservation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("from_scan", [False, True])
@pytest.mark.parametrize("filename", ["__tmp__report.txt", "report.txt"])
async def test_deferred_parser_preserves_source_until_it_is_parsed(
    tmp_path, monkeypatch, from_scan, filename
):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("LIGHTRAG_PARSER", "txt:legacy")
    monkeypatch.setenv("INPUT_DIR", str(tmp_path))
    from lightrag.api.routers.document_routes import pipeline_enqueue_file

    rag = await build_delete_test_rag(tmp_path, f"deferred-source-{uuid4().hex}")
    source = tmp_path / filename
    content = "alice works at acme"
    source.write_text(content, encoding="utf-8")
    try:
        success, _ = await pipeline_enqueue_file(
            rag, source, "test-deferred-source", from_scan=from_scan
        )
        assert success
        await rag.apipeline_process_enqueue_documents()

        match = await rag.doc_status.get_doc_by_file_basename(filename)
        assert match is not None
        _, row = match
        assert row["status"] == DocStatus.PROCESSED.value, row
        assert (tmp_path / PARSED_DIR_NAME / filename).read_text(
            encoding="utf-8"
        ) == content
    finally:
        await rag.finalize_storages()
