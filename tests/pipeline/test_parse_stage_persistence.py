"""Parse-stage persistence contracts across doc_status and full_docs.

Consolidates:
- early persistence of ``parse_engine`` / ``parse_format`` at PARSING (with no
  value jump through PROCESSED) and carry-over allowlist checks;
- parse-stage FAILED ``doc_status`` field mapping (``content_summary`` budget,
  ``error_type`` / ``error_stage`` / ``parse_engine``, retry reset/clearing,
  and logging when FAILED-status upsert itself fails);
- smart-heading ``parse_warnings`` split (non-smart -> doc_status, smart ->
  sidecar ``smart_audit.json``);
- control-character sanitization in ``_persist_parsed_full_docs`` and parse
  worker alignment between ``full_docs`` and ``doc_status``;
- empty-chunk documents reaching PROCESSED with present-but-empty recovery
  anchors and zero LLM invocations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pytest

from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.constants import (
    DOC_STATUS_CONTENT_SUMMARY_MAX_LENGTH,
    FULL_DOCS_FORMAT_LIGHTRAG,
    FULL_DOCS_FORMAT_PENDING_PARSE,
    FULL_DOCS_FORMAT_RAW,
    PARSER_ENGINE_DOCLING,
    PARSER_ENGINE_LEGACY,
    PARSER_ENGINE_MINERU,
    PARSER_ENGINE_NATIVE,
)
from lightrag.parser import registry
from lightrag.parser.debug import build_debug_rag
from lightrag.utils import compute_mdhash_id
from lightrag.utils_pipeline import (
    _DOC_STATUS_METADATA_CARRY_OVER_KEYS,
    compute_text_content_hash,
    doc_status_metadata_carry_over,
    doc_status_parse_failure_fields,
    doc_status_transition_metadata,
    make_lightrag_doc_content,
    resolve_doc_status_parse_engine,
    strip_lightrag_doc_prefix,
)

from ._fakes import (
    build_initialized_rag,
    make_lightrag,
    mock_json_llm,
    single_chunk_chunking,
)
from .conftest import request_failed_retry

pytestmark = pytest.mark.offline


async def _build_rag(tmp_path: Path, **overrides) -> LightRAG:
    return await build_initialized_rag(
        tmp_path,
        workspace_prefix="parse-stage",
        chunking_func=overrides.pop("chunking_func", single_chunk_chunking),
        max_parallel_insert=1,
        **overrides,
    )


def _new_early_persist_rag(tmp_path: Path) -> LightRAG:
    return make_lightrag(
        tmp_path,
        subdir=None,
        workspace_prefix="parse-engine-early",
        llm_model_func=mock_json_llm,
        embedding_dim=32,
        max_token_size=4096,
        max_parallel_insert=1,
    )


# ---------------------------------------------------------------------------
# Early parse_engine / parse_format persistence
# ---------------------------------------------------------------------------


def test_resolve_engine_explicit_wins():
    assert (
        resolve_doc_status_parse_engine(FULL_DOCS_FORMAT_LIGHTRAG, PARSER_ENGINE_MINERU)
        == PARSER_ENGINE_MINERU
    )
    assert (
        resolve_doc_status_parse_engine(FULL_DOCS_FORMAT_RAW, PARSER_ENGINE_DOCLING)
        == PARSER_ENGINE_DOCLING
    )


def test_resolve_engine_format_based_fallback():
    assert (
        resolve_doc_status_parse_engine(FULL_DOCS_FORMAT_LIGHTRAG, None)
        == PARSER_ENGINE_NATIVE
    )
    assert (
        resolve_doc_status_parse_engine(FULL_DOCS_FORMAT_RAW, None)
        == PARSER_ENGINE_LEGACY
    )
    assert resolve_doc_status_parse_engine(None, None) == PARSER_ENGINE_LEGACY
    assert (
        resolve_doc_status_parse_engine(FULL_DOCS_FORMAT_RAW, "")
        == PARSER_ENGINE_LEGACY
    )


def test_carry_over_preserves_parse_engine_and_format():
    assert "parse_format" in _DOC_STATUS_METADATA_CARRY_OVER_KEYS
    assert "parse_engine" in _DOC_STATUS_METADATA_CARRY_OVER_KEYS
    carried = doc_status_metadata_carry_over(
        {
            "metadata": {
                "parse_format": FULL_DOCS_FORMAT_LIGHTRAG,
                "parse_engine": PARSER_ENGINE_MINERU,
                "process_options": "iF",
            }
        }
    )
    assert carried["parse_format"] == FULL_DOCS_FORMAT_LIGHTRAG
    assert carried["parse_engine"] == PARSER_ENGINE_MINERU


def _attach_transition_spy(rag: LightRAG) -> list[tuple[str, dict]]:
    records: list[tuple[str, dict]] = []
    orig = rag._upsert_doc_status_transition

    async def _spy(doc_id, status, status_doc, file_path, **kwargs):
        written = doc_status_transition_metadata(
            status_doc, extra=kwargs.get("metadata_extra")
        )
        status_text = status.value if isinstance(status, DocStatus) else str(status)
        records.append((status_text, dict(written)))
        return await orig(doc_id, status, status_doc, file_path, **kwargs)

    rag._upsert_doc_status_transition = _spy
    return records


def _assert_early_and_stable(records, *, expected_format, expected_engine):
    with_engine = [(status, md) for status, md in records if "parse_engine" in md]
    assert with_engine, "parse_engine never written to any doc_status transition"
    assert with_engine[0][0] == DocStatus.PARSING.value
    assert {md["parse_engine"] for _, md in with_engine} == {expected_engine}
    assert {
        md["parse_format"] for _, md in with_engine if "parse_format" in md
    } == {expected_format}


async def test_raw_doc_records_legacy_engine_at_parsing(tmp_path):
    rag = _new_early_persist_rag(tmp_path / "raw")
    await rag.initialize_storages()
    records = _attach_transition_spy(rag)
    try:
        await rag.apipeline_enqueue_documents(
            "Alpha body paragraph with enough words to look real.",
            file_paths="early_raw.txt",
            track_id="track-raw",
        )
        await rag.apipeline_process_enqueue_documents()

        _assert_early_and_stable(
            records,
            expected_format=FULL_DOCS_FORMAT_RAW,
            expected_engine=PARSER_ENGINE_LEGACY,
        )
        doc_id = compute_mdhash_id("early_raw.txt", prefix="doc-")
        stored = await rag.doc_status.get_by_id(doc_id)
        assert stored is not None
        metadata = (
            stored.get("metadata")
            if isinstance(stored, dict)
            else getattr(stored, "metadata", None)
        )
        assert metadata.get("parse_format") == FULL_DOCS_FORMAT_RAW
        assert metadata.get("parse_engine") == PARSER_ENGINE_LEGACY
    finally:
        await rag.finalize_storages()


async def test_lightrag_doc_records_native_engine_at_parsing(tmp_path):
    rag = _new_early_persist_rag(tmp_path / "work")
    await rag.initialize_storages()
    records = _attach_transition_spy(rag)
    try:
        doc_id = "doc-early-lr"
        body = "Body paragraph for lightrag doc."
        now = datetime.now(timezone.utc).isoformat()
        await rag.full_docs.upsert(
            {
                doc_id: {
                    "content": make_lightrag_doc_content(body),
                    "file_path": "early.lightrag",
                    "parse_format": FULL_DOCS_FORMAT_LIGHTRAG,
                }
            }
        )
        await rag.doc_status.upsert(
            {
                doc_id: {
                    "status": DocStatus.PENDING.value,
                    "content_summary": body,
                    "content_length": len(body),
                    "file_path": "early.lightrag",
                    "created_at": now,
                    "updated_at": now,
                    "track_id": "track-lr",
                }
            }
        )
        await rag.apipeline_process_enqueue_documents()

        _assert_early_and_stable(
            records,
            expected_format=FULL_DOCS_FORMAT_LIGHTRAG,
            expected_engine=PARSER_ENGINE_NATIVE,
        )
    finally:
        await rag.finalize_storages()


# ---------------------------------------------------------------------------
# Parse-stage FAILED doc_status field mapping
# ---------------------------------------------------------------------------


def test_empty_summary_gets_error_description():
    extra, meta = doc_status_parse_failure_fields(
        ValueError("boom"), status_doc={"content_summary": "", "metadata": {}}
    )
    assert extra["error_msg"] == "boom"
    assert extra["content_summary"] == "[File Extraction]boom"
    assert meta["error_type"] == "file_extraction_error"
    assert meta["error_stage"] == "parse"


def test_empty_error_message_falls_back_to_class_name():
    extra, _meta = doc_status_parse_failure_fields(
        ConnectionError(""), status_doc={"content_summary": "", "metadata": {}}
    )
    assert extra["error_msg"] == "ConnectionError"
    assert extra["content_summary"] == "[File Extraction]ConnectionError"


def test_existing_summary_is_not_overwritten():
    extra, _meta = doc_status_parse_failure_fields(
        ValueError("boom"),
        status_doc={"content_summary": "real document summary", "metadata": {}},
    )
    assert "content_summary" not in extra
    assert extra["error_msg"] == "boom"


def test_generated_summary_from_previous_failure_is_replaced():
    extra, _meta = doc_status_parse_failure_fields(
        ValueError("second failure"),
        status_doc={
            "content_summary": "[File Extraction]first failure",
            "metadata": {},
        },
    )
    assert extra["error_msg"] == "second failure"
    assert extra["content_summary"] == "[File Extraction]second failure"


def test_whitespace_summary_counts_as_empty():
    extra, _meta = doc_status_parse_failure_fields(
        ValueError("boom"), status_doc={"content_summary": "   \n", "metadata": {}}
    )
    assert extra["content_summary"] == "[File Extraction]boom"


def test_engine_hint_fills_missing_parse_engine():
    _extra, meta = doc_status_parse_failure_fields(
        ValueError("boom"),
        status_doc={"content_summary": "", "metadata": {}},
        engine_hint="mineru",
    )
    assert meta["parse_engine"] == "mineru"


def test_engine_hint_does_not_override_existing_parse_engine():
    _extra, meta = doc_status_parse_failure_fields(
        ValueError("boom"),
        status_doc={"content_summary": "", "metadata": {"parse_engine": "docling"}},
        engine_hint="native",
    )
    assert "parse_engine" not in meta


def test_long_error_text_is_truncated_in_summary():
    long_error = "x" * 1000
    extra, _meta = doc_status_parse_failure_fields(
        ValueError(long_error), status_doc={"content_summary": "", "metadata": {}}
    )
    summary = extra["content_summary"]
    assert summary.startswith("[File Extraction]")
    assert summary.endswith("...")
    assert extra["error_msg"] == long_error


@pytest.mark.parametrize(
    "error_text",
    [
        "x" * 1000,
        "CUDA out of memory. Tried to allocate 360.00 MiB. GPU 0 has a total "
        "capacity of 22.03 GiB of which 109.94 MiB is free. Process 853701 has "
        "3.85 GiB memory in use. Process 854340 has 18.06 GiB memory in use. Of "
        "the allocated memory 3.08 GiB is allocated by PyTorch, and 344.55 MiB "
        "is reserved by PyTorch but unallocated. If reserved but unallocated "
        "memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments"
        ":True to avoid fragmentation.  See documentation for Memory Management",
        "y" * (DOC_STATUS_CONTENT_SUMMARY_MAX_LENGTH - len("[File Extraction]") - 2),
    ],
)
def test_generated_summary_fits_the_doc_status_column(error_text):
    extra, _meta = doc_status_parse_failure_fields(
        ValueError(error_text), status_doc={"content_summary": "", "metadata": {}}
    )
    summary = extra["content_summary"]
    assert summary.startswith("[File Extraction]")
    assert len(summary) <= DOC_STATUS_CONTENT_SUMMARY_MAX_LENGTH
    assert extra["error_msg"] == error_text


@pytest.mark.asyncio
async def test_legacy_parse_failure_maps_ui_fields_and_retry_clears_them(
    tmp_path, monkeypatch
):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    rag = await _build_rag(tmp_path)
    try:
        source_path = input_dir / "missing.txt"
        await rag.apipeline_enqueue_documents(
            "",
            file_paths=str(source_path),
            docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
            parse_engine="legacy",
            process_options="i",
        )
        doc_id = compute_mdhash_id("missing.txt", prefix="doc-")

        await rag.apipeline_process_enqueue_documents()

        status = await rag.doc_status.get_by_id(doc_id)
        assert status["status"] == DocStatus.FAILED
        assert "legacy source file not found" in status["error_msg"]
        assert status["content_summary"].startswith("[File Extraction]")
        assert "legacy source file not found" in status["content_summary"]
        metadata = status["metadata"]
        assert metadata["error_type"] == "file_extraction_error"
        assert metadata["error_stage"] == "parse"
        assert metadata["parse_engine"] == "legacy"
        assert metadata["process_options"] == "i"

        source_path.write_text("recovered body text for retry")
        await request_failed_retry(rag)
        await rag.apipeline_process_enqueue_documents()

        status = await rag.doc_status.get_by_id(doc_id)
        assert status["status"] == DocStatus.PROCESSED
        assert not status.get("error_msg")
        assert status["content_summary"] == "recovered body text for retry"
        metadata = status["metadata"]
        assert "error_type" not in metadata
        assert "error_stage" not in metadata
        assert metadata["parse_engine"] == "legacy"
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_third_party_engine_failure_records_engine_name(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setenv("INPUT_DIR", str(input_dir))

    class _BoomParser:
        engine_name = "boomengine"

        async def parse(self, ctx):
            raise RuntimeError("boom engine exploded")

    fake_mod = types.ModuleType("_test_boom_parser_mod")
    fake_mod.BoomParser = _BoomParser
    monkeypatch.setitem(sys.modules, "_test_boom_parser_mod", fake_mod)
    registry.register_parser(
        registry.ParserSpec(
            engine_name="boomengine",
            impl="_test_boom_parser_mod:BoomParser",
            suffixes=frozenset({"txt"}),
            queue_group="native",
        )
    )
    rag = await _build_rag(tmp_path)
    try:
        source_path = input_dir / "doc.txt"
        source_path.write_text("some body")
        await rag.apipeline_enqueue_documents(
            "",
            file_paths=str(source_path),
            docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
            parse_engine="boomengine",
        )
        doc_id = compute_mdhash_id("doc.txt", prefix="doc-")

        await rag.apipeline_process_enqueue_documents()

        status = await rag.doc_status.get_by_id(doc_id)
        assert status["status"] == DocStatus.FAILED
        assert status["error_msg"] == "boom engine exploded"
        assert status["content_summary"] == "[File Extraction]boom engine exploded"
        metadata = status["metadata"]
        assert metadata["error_type"] == "file_extraction_error"
        assert metadata["error_stage"] == "parse"
        assert metadata["parse_engine"] == "boomengine"
    finally:
        registry._REGISTRY.pop("boomengine", None)
        registry._INSTANCE_CACHE.pop(
            ("boomengine", "_test_boom_parser_mod:BoomParser"), None
        )
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_repeat_failure_replaces_stale_generated_summary(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setenv("INPUT_DIR", str(input_dir))

    class _FlakyParser:
        engine_name = "flakyengine"
        attempts = 0

        async def parse(self, ctx):
            _FlakyParser.attempts += 1
            raise RuntimeError(f"failure #{_FlakyParser.attempts}")

    fake_mod = types.ModuleType("_test_flaky_parser_mod")
    fake_mod.FlakyParser = _FlakyParser
    monkeypatch.setitem(sys.modules, "_test_flaky_parser_mod", fake_mod)
    registry.register_parser(
        registry.ParserSpec(
            engine_name="flakyengine",
            impl="_test_flaky_parser_mod:FlakyParser",
            suffixes=frozenset({"txt"}),
            queue_group="native",
        )
    )
    rag = await _build_rag(tmp_path)
    try:
        source_path = input_dir / "doc.txt"
        source_path.write_text("some body")
        await rag.apipeline_enqueue_documents(
            "",
            file_paths=str(source_path),
            docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
            parse_engine="flakyengine",
        )
        doc_id = compute_mdhash_id("doc.txt", prefix="doc-")

        await rag.apipeline_process_enqueue_documents()
        status = await rag.doc_status.get_by_id(doc_id)
        assert status["status"] == DocStatus.FAILED
        assert status["content_summary"] == "[File Extraction]failure #1"

        await request_failed_retry(rag)
        await rag.apipeline_process_enqueue_documents()
        status = await rag.doc_status.get_by_id(doc_id)
        assert status["status"] == DocStatus.FAILED
        assert status["error_msg"] == "failure #2"
        assert status["content_summary"] == "[File Extraction]failure #2"
    finally:
        registry._REGISTRY.pop("flakyengine", None)
        registry._INSTANCE_CACHE.pop(
            ("flakyengine", "_test_flaky_parser_mod:FlakyParser"), None
        )
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_raw_document_failure_keeps_real_summary(tmp_path, monkeypatch):
    rag = await _build_rag(tmp_path)
    try:
        await rag.apipeline_enqueue_documents(
            "raw body that becomes the summary", file_paths="raw.txt"
        )
        doc_id = compute_mdhash_id("raw.txt", prefix="doc-")

        async def _passthrough_validate(docs, pipeline_status, pipeline_status_lock):
            return docs

        monkeypatch.setattr(
            rag, "_validate_and_fix_document_consistency", _passthrough_validate
        )
        orchestrator_read_done = {"flag": False}
        orig_get = rag.full_docs.get_by_id

        async def _flaky_get(d_id):
            if d_id == doc_id and orchestrator_read_done["flag"]:
                raise ConnectionError("storage briefly down")
            if d_id == doc_id:
                orchestrator_read_done["flag"] = True
            return await orig_get(d_id)

        monkeypatch.setattr(rag.full_docs, "get_by_id", _flaky_get)

        await rag.apipeline_process_enqueue_documents()
        await asyncio.sleep(0)

        status = await rag.doc_status.get_by_id(doc_id)
        assert status["status"] == DocStatus.FAILED
        assert status["content_summary"] == "raw body that becomes the summary"
        assert "storage briefly down" in status["error_msg"]
        assert status["metadata"]["error_type"] == "file_extraction_error"
        assert status["metadata"]["error_stage"] == "parse"
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_parse_worker_logs_when_failed_upsert_itself_fails(
    tmp_path, monkeypatch, lightrag_caplog
):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setenv("INPUT_DIR", str(input_dir))

    class _BoomParser:
        engine_name = "boomengine"

        async def parse(self, ctx):
            raise RuntimeError("boom engine exploded")

    fake_mod = types.ModuleType("_test_boom_upsert_mod")
    fake_mod.BoomParser = _BoomParser
    monkeypatch.setitem(sys.modules, "_test_boom_upsert_mod", fake_mod)
    registry.register_parser(
        registry.ParserSpec(
            engine_name="boomengine",
            impl="_test_boom_upsert_mod:BoomParser",
            suffixes=frozenset({"txt"}),
            queue_group="native",
        )
    )
    rag = await _build_rag(tmp_path)
    try:
        source_path = input_dir / "doc.txt"
        source_path.write_text("some body")
        await rag.apipeline_enqueue_documents(
            "",
            file_paths=str(source_path),
            docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
            parse_engine="boomengine",
        )

        orig_upsert = rag._upsert_doc_status_transition

        async def _flaky_upsert(*args, status=None, **kwargs):
            if status == DocStatus.FAILED:
                raise RuntimeError("doc_status backend unavailable")
            return await orig_upsert(*args, status=status, **kwargs)

        monkeypatch.setattr(rag, "_upsert_doc_status_transition", _flaky_upsert)

        with lightrag_caplog.at_level(logging.ERROR, logger="lightrag"):
            await rag.apipeline_process_enqueue_documents()

        assert any(
            "Failed to record FAILED status" in record.getMessage()
            for record in lightrag_caplog.records
        )
    finally:
        registry._REGISTRY.pop("boomengine", None)
        registry._INSTANCE_CACHE.pop(
            ("boomengine", "_test_boom_upsert_mod:BoomParser"), None
        )
        await rag.finalize_storages()


# ---------------------------------------------------------------------------
# Smart-heading parse_warnings split persistence
# ---------------------------------------------------------------------------

_STUB_BLOCKS = [
    {
        "uuid": "p1",
        "heading": "Chapter One",
        "content": "# Chapter One\nBody paragraph with enough words to look real.",
        "type": "text",
        "parent_headings": [],
        "level": 1,
    }
]


def _stub_extract_mixed(
    file_path,
    *,
    drawing_context=None,
    parse_warnings=None,
    parse_metadata=None,
    **_kwargs,
):
    if parse_warnings is not None:
        parse_warnings.update(
            {
                "smart_cb1_tripped": 2,
                "title_block_empty_members": 1,
                "missing_paraid_count": 4,
            }
        )
    if parse_metadata is not None:
        parse_metadata["smart_audit"] = {"shadow_diff": {}}
    return [dict(b) for b in _STUB_BLOCKS]


async def test_pipeline_diverts_smart_warnings_keeps_nonsmart_on_doc_status(
    tmp_path, monkeypatch
):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    rag = await _build_rag(tmp_path)
    try:
        source_path = input_dir / "doc.docx"
        source_path.write_bytes(b"fake-docx")

        with mock.patch(
            "lightrag.parser.docx.parse_document.extract_docx_blocks",
            _stub_extract_mixed,
        ):
            await rag.apipeline_enqueue_documents(
                "",
                file_paths=str(source_path),
                docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
                parse_engine="native",
            )
            doc_id = compute_mdhash_id("doc.docx", prefix="doc-")
            await rag.apipeline_process_enqueue_documents()

        status = await rag.doc_status.get_by_id(doc_id)
        assert status is not None
        assert status["status"] == DocStatus.PROCESSED
        persisted = status["metadata"].get("parse_warnings")
        assert persisted == {"missing_paraid_count": 4}
        assert not any(
            k.startswith(("smart_", "title_block_")) for k in (persisted or {})
        )

        (audit_path,) = input_dir.glob("**/*.smart_audit.json")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        assert audit["parse_warnings"] == {
            "smart_cb1_tripped": 2,
            "title_block_empty_members": 1,
        }
        assert "shadow_diff" in audit
    finally:
        await rag.finalize_storages()


# ---------------------------------------------------------------------------
# Control-character sanitization (full_docs & parse worker doc_status alignment)
# ---------------------------------------------------------------------------


async def test_legacy_parse_worker_sanitizes_doc_status_and_full_docs(
    tmp_path, monkeypatch
):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    rag = await _build_rag(tmp_path)
    try:
        dirty = "Alpha\x1cbody\x1d中\x1f文\x00 paragraph with real words."
        clean = "Alphabody中文 paragraph with real words."
        source_path = input_dir / "doc.txt"
        source_path.write_text(dirty, encoding="utf-8")

        await rag.apipeline_enqueue_documents(
            "",
            file_paths=str(source_path),
            docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
            parse_engine="legacy",
        )
        doc_id = compute_mdhash_id("doc.txt", prefix="doc-")
        await rag.apipeline_process_enqueue_documents()

        status = await rag.doc_status.get_by_id(doc_id)
        assert status["status"] == DocStatus.PROCESSED
        assert not any(c in status["content_summary"] for c in "\x1c\x1d\x1f\x00")

        full = await rag.full_docs.get_by_id(doc_id)
        body = strip_lightrag_doc_prefix(full["content"], full.get("parse_format"))
        assert body == clean
        assert status["content_length"] == len(clean)
    finally:
        await rag.finalize_storages()


async def test_persist_strips_control_chars_from_raw_content():
    rag = build_debug_rag()
    dirty = "head\x1cbody\x1d中\x1f文\x00\x7f tail"

    await rag._persist_parsed_full_docs(
        "doc-raw",
        {
            "content": dirty,
            "file_path": "f.pdf",
            "parse_format": FULL_DOCS_FORMAT_RAW,
            "parse_engine": "legacy",
        },
    )

    stored = rag.full_docs.data["doc-raw"]
    assert not any(c in stored["content"] for c in "\x1c\x1d\x1f\x00\x7f")
    assert stored["content"] == "headbody中文 tail"


async def test_persist_content_hash_matches_cleaned_body():
    rag = build_debug_rag()
    dirty = "alpha\x1ebeta\x1f"

    await rag._persist_parsed_full_docs(
        "doc-hash",
        {
            "content": dirty,
            "file_path": "f.pdf",
            "parse_format": FULL_DOCS_FORMAT_RAW,
            "parse_engine": "legacy",
        },
    )

    stored = rag.full_docs.data["doc-hash"]
    expected = compute_text_content_hash(
        strip_lightrag_doc_prefix("alphabeta", FULL_DOCS_FORMAT_RAW)
    )
    assert stored["content_hash"] == expected


async def test_persist_noop_for_clean_content():
    rag = build_debug_rag()
    clean = "perfectly clean body\nwith newline\ttab"

    await rag._persist_parsed_full_docs(
        "doc-clean",
        {
            "content": clean,
            "file_path": "f.pdf",
            "parse_format": FULL_DOCS_FORMAT_RAW,
            "parse_engine": "legacy",
        },
    )

    assert rag.full_docs.data["doc-clean"]["content"] == clean


# ---------------------------------------------------------------------------
# Empty-chunk document commit & anchor persistence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_chunk_document_reaches_processed_with_empty_anchors(tmp_path):
    """An empty chunk set still persists present-but-empty recovery anchors without LLM calls."""
    llm_calls: list[bool] = []

    async def _forbid_llm(*args, **kwargs) -> str:
        llm_calls.append(True)
        raise AssertionError("an empty chunk set must not invoke the LLM")

    rag = await _build_rag(
        tmp_path,
        llm_model_func=_forbid_llm,
        chunking_func=lambda *args, **kwargs: [],
    )
    try:
        doc_id = compute_mdhash_id("empty.txt", prefix="doc-")
        await rag.apipeline_enqueue_documents(
            "content intentionally removed by the chunker",
            ids=[doc_id],
            file_paths=["empty.txt"],
        )
        await rag.apipeline_process_enqueue_documents()

        assert llm_calls == []
        status = await rag.doc_status.get_by_id(doc_id)
        assert status["status"] == DocStatus.PROCESSED
        assert status["chunks_count"] == 0
        assert status["chunks_list"] == []
        entity_anchor = await rag.full_entities.get_by_id(doc_id)
        relation_anchor = await rag.full_relations.get_by_id(doc_id)
        assert entity_anchor["entity_names"] == []
        assert entity_anchor["count"] == 0
        assert relation_anchor["relation_pairs"] == []
        assert relation_anchor["count"] == 0
    finally:
        await rag.finalize_storages()
