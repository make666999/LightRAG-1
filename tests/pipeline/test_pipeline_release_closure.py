"""End-to-end and contract release-closure checks across the document pipeline.

Covers parser routing/hints, doc_status metadata carry-over, stage-end timing,
resume purge gating, enqueue/busy/ingress guards, dedup/content-hash persistence,
multimodal analysis & sidecar chunk building, MinerU persistence, and VDB timeouts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pytest

import lightrag.lightrag as lightrag_module
import lightrag.pipeline as pipeline_module
from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.chunk_schema import strip_internal_multimodal_markup_for_extraction
from lightrag.constants import (
    FULL_DOCS_FORMAT_PENDING_PARSE,
    PARSED_DIR_NAME,
    PARSER_ENGINE_MINERU,
    PARSER_ENGINE_NATIVE,
)
from lightrag.kg.shared_storage import get_namespace_data, get_pipeline_ingress
from lightrag.operate import (
    _get_relationship_vdb_timeout_seconds,
    _parse_mm_display_name,
)
from lightrag.parser.base import ParseContext
from lightrag.parser.registry import get_parser
from lightrag.parser.routing import (
    FilenameParserHintError,
    ParserRoutingConfigError,
    canonicalize_parser_hinted_basename,
    filename_parser_directives,
    parse_process_options,
    resolve_file_parser_directives,
    resolve_file_parser_engine,
    resolve_stored_document_parser_engine,
    validate_parser_routing_config,
    validate_process_options,
)
from lightrag.utils import compute_mdhash_id, safe_vdb_operation_with_exception
from lightrag.utils_pipeline import (
    _DOC_STATUS_METADATA_CARRY_OVER_KEYS,
    build_chunks_dict_from_chunking_result,
    doc_status_transition_metadata,
    resolve_sidecar_uri,
)

from ._fakes import (
    PNG_BYTES,
    make_lightrag,
    mock_json_llm,
    pipeline_status_handles,
    status_value_text,
)

pytestmark = pytest.mark.offline


async def _parse_via_registry(rag, engine, doc_id, file_path, content_data):
    """Drive a parser the way the pipeline worker does (registry dispatch)."""
    result = await get_parser(engine).parse(
        ParseContext(rag, doc_id, file_path, content_data)
    )
    return result.to_dict()


def _new_rag(tmp_path: Path, **kwargs) -> LightRAG:
    # analyze_multimodal short-circuits when vlm_process_enable is False;
    # default the switch ON for the VLM checks in this module.
    kwargs.setdefault("vlm_process_enable", True)
    return make_lightrag(
        tmp_path,
        subdir=None,
        workspace_prefix="release-closure",
        llm_model_func=mock_json_llm,
        embedding_dim=32,
        max_token_size=4096,
        **kwargs,
    )


@asynccontextmanager
async def _open_rag(tmp_path: Path, **kwargs):
    rag = _new_rag(tmp_path, **kwargs)
    await rag.initialize_storages()
    try:
        yield rag
    finally:
        await rag.finalize_storages()


def _write_sidecar_doc(
    base_dir: Path,
    stem: str = "demo",
    *,
    blocks_lines: list[dict] | None = None,
    drawings: dict | None = None,
    tables: dict | None = None,
    equations: dict | None = None,
) -> tuple[Path, dict[str, Path]]:
    """Write ``<stem>.blocks.jsonl`` plus optional modality sidecars."""
    blocks = base_dir / f"{stem}.blocks.jsonl"
    if blocks_lines is None:
        blocks_lines = [
            {"type": "meta", "format_version": "1.0"},
            {"type": "content", "content": "body"},
        ]
    text = (
        "\n".join(json.dumps(line, ensure_ascii=False) for line in blocks_lines) + "\n"
        if blocks_lines
        else ""
    )
    blocks.write_text(text, encoding="utf-8")

    sidecars: dict[str, Path] = {}
    for kind, payload in (
        ("drawings", drawings),
        ("tables", tables),
        ("equations", equations),
    ):
        if payload is not None:
            path = base_dir / f"{stem}.{kind}.json"
            path.write_text(
                json.dumps({"version": "1.0", kind: payload}, ensure_ascii=False),
                encoding="utf-8",
            )
            sidecars[kind] = path
    return blocks, sidecars


def _install_mineru_fake_download(
    monkeypatch,
    *,
    content_list: list[dict] | None,
    assets: dict[str, bytes] | None = None,
    expected_source: Path | None = None,
    expected_upload_name: str | None = None,
    options_params: dict | None = None,
) -> None:
    """Stub ``MinerURawClient.download_into`` to emit a compliant local bundle."""
    from lightrag.parser.external.mineru import compute_size_and_hash
    from lightrag.parser.external.mineru.cache import current_mineru_options_signature
    from lightrag.parser.external.mineru.client import MinerURawClient
    from lightrag.parser.external.mineru.manifest import (
        Manifest,
        ManifestFile,
        write_manifest,
    )

    async def _fake_download(self, raw_dir, source_file_path, **kwargs):
        if expected_source is not None:
            assert source_file_path == expected_source
        if expected_upload_name is not None:
            assert kwargs.get("upload_name") == expected_upload_name
        raw_dir.mkdir(parents=True, exist_ok=True)
        if content_list is None:
            return None

        content_path = raw_dir / "content_list.json"
        content_path.write_text(
            json.dumps(content_list, ensure_ascii=False),
            encoding="utf-8",
        )
        manifest_files: list[ManifestFile] = []
        for rel_path, blob in (assets or {}).items():
            dest = raw_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(blob)
            manifest_files.append(
                ManifestFile(path=rel_path, size=dest.stat().st_size)
            )
        src_size, src_hash = compute_size_and_hash(source_file_path)
        crit_size, crit_hash = compute_size_and_hash(content_path)
        manifest = Manifest(
            source_content_hash=src_hash,
            source_size_bytes=src_size,
            source_filename_at_parse=kwargs.get("upload_name")
            or source_file_path.name,
            critical_file=ManifestFile(
                path="content_list.json", size=crit_size, sha256=crit_hash
            ),
            files=manifest_files,
            total_size_bytes=crit_size,
            task_id="fake-task",
            api_mode="local",
            options_signature=current_mineru_options_signature(options_params),
        )
        write_manifest(raw_dir, manifest)
        return manifest

    monkeypatch.setattr(MinerURawClient, "download_into", _fake_download)


# ---------------------------------------------------------------------------
# Parser routing, filename hints, and process options
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env, filename, content_data, expected",
    [
        pytest.param(
            {"DOCLING_ENDPOINT": "http://fake-docling"},
            "a.[docling-iet].docx",
            {},
            "docling",
            id="filename-hint-docling",
        ),
        pytest.param(
            {
                "MINERU_LOCAL_ENDPOINT": "http://fake-mineru",
                "LIGHTRAG_PARSER": "pdf:mineru-iet,*:native",
            },
            "paper.pdf",
            {},
            "mineru",
            id="env-rule-mineru",
        ),
        pytest.param(
            {"MINERU_LOCAL_ENDPOINT": "http://fake-mineru"},
            "paper.pdf",
            {"parse_engine": "native"},
            "passthrough",
            id="already-extracted-passthrough",
        ),
        pytest.param(
            {"LIGHTRAG_PARSER": "pdf:native,*:legacy"},
            "paper.pdf",
            {},
            "legacy",
            id="unsupported-suffix-falls-back-to-legacy",
        ),
        pytest.param(
            {"LIGHTRAG_PARSER": "pptx:docling,*:legacy", "DOCLING_ENDPOINT": None},
            "slides.pptx",
            {},
            "legacy",
            id="missing-endpoint-falls-back-to-legacy",
        ),
        pytest.param(
            {"LIGHTRAG_PARSER": None, "MINERU_LOCAL_ENDPOINT": None},
            "slides.pptx",
            {},
            "legacy",
            id="unset-env-defaults-to-legacy",
        ),
    ],
)
def test_resolve_stored_document_parser_engine(
    monkeypatch, env, filename, content_data, expected
):
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    assert resolve_stored_document_parser_engine(filename, content_data) == expected


def test_canonicalize_parser_hinted_basename():
    cases = [
        ("abc.[native].docx", "abc.docx"),
        ("/tmp/a.b.[mineru].pdf", "a.b.pdf"),
        ("abc.[draft].docx", "abc.[draft].docx"),
        ("abc.[NATIVE].docx", "abc.docx"),
        ("abc.[mineru-iet].pdf", "abc.pdf"),
        ("abc.[native]", "abc.[native]"),
        ("abc.docx", "abc.docx"),
        ("[native].docx", "[native].docx"),
        ("name.[native].[mineru].pdf", "name.[native].pdf"),
        ("foo.[-!].docx", "foo.docx"),
        ("foo.[native-iet].docx", "foo.docx"),
        ("foo.[mineru-R!].pdf", "foo.pdf"),
        ("foo.[!].docx", "foo.[!].docx"),
        ("foo.[native-].docx", "foo.[native-].docx"),
        ("foo.[xyz].docx", "foo.[xyz].docx"),
    ]
    mismatches = [
        (raw, expected, canonicalize_parser_hinted_basename(raw))
        for raw, expected in cases
        if canonicalize_parser_hinted_basename(raw) != expected
    ]
    assert mismatches == []


def test_filename_parser_directives_decodes_engine_and_options():
    cases = [
        ("paper.[native-iet].docx", ("native", "iet")),
        ("memo.[native-R!].md", ("native", "R!")),
        ("custom.[native-Cite].docx", ("native", "Cite")),
        ("report.[-!].pdf", (None, "!")),
        ("doc.[mineru].docx", ("mineru", "")),
        ("foo.docx", (None, "")),
        ("foo.[!].docx", (None, "")),
        ("foo.[draft].docx", (None, "")),
    ]
    mismatches = [
        (raw, expected, filename_parser_directives(raw))
        for raw, expected in cases
        if filename_parser_directives(raw) != expected
    ]
    assert mismatches == []


@pytest.mark.parametrize(
    "name, match",
    [
        ("foo.[native-FR].docx", "multiple chunking modes"),
        ("foo.[native-Q].docx", "unsupported character"),
        ("foo.[!].docx", "options-only filename hints"),
        ("foo.[iet].docx", "options-only filename hints"),
        ("foo.[abc].docx", "unsupported parser engine 'abc'"),
        ("foo.[xyz].docx", "unsupported parser engine 'xyz'"),
        ("foo.[].docx", "is empty"),
        ("foo.[-].docx", "empty process options"),
        ("foo.[native-].docx", "empty process options"),
        ("foo.[ - ].docx", "empty process options"),
    ],
)
def test_filename_hint_rejects_invalid_directives(name, match):
    """Low-level helpers stay non-throwing while directive resolution fails fast."""
    assert filename_parser_directives(name) == (None, "")
    assert canonicalize_parser_hinted_basename(name) == name
    with pytest.raises(FilenameParserHintError, match=match):
        resolve_file_parser_directives(name)


@pytest.mark.parametrize(
    "name",
    [
        "foo.[ -iet].docx",
        "foo.[-iet ].docx",
        "foo.[ native-iet ].docx",
        "foo.[ -R(chunk_ts=800) ].md",
    ],
)
def test_filename_hint_validation_ignores_surrounding_whitespace(name):
    """Resolution must accept every hint the non-throwing classifier accepts."""
    engine, options = filename_parser_directives(name)
    assert options
    resolved_engine, resolved_options = resolve_file_parser_directives(
        name, parser_rules="", require_external_endpoint=False
    )
    assert resolved_options == options
    if engine:
        assert resolved_engine == engine


@pytest.mark.parametrize(
    "name", ["foo.[xyz-R(chunk_ts=800)].docx", "foo.[R(chunk_ts=800)].docx"]
)
def test_parameterised_hint_without_engine_or_hyphen_stays_unrecognised(name):
    """A parameter block must not turn an unrecognised hint into a recognised one."""
    assert canonicalize_parser_hinted_basename(name) == name
    assert filename_parser_directives(name) == (None, "")
    with pytest.raises(FilenameParserHintError):
        resolve_file_parser_directives(name)


def test_filename_hint_missing_required_endpoint_rejects(monkeypatch):
    monkeypatch.delenv("DOCLING_ENDPOINT", raising=False)
    with pytest.raises(FilenameParserHintError, match="requires DOCLING_ENDPOINT"):
        resolve_file_parser_directives("foo.[docling].docx")


def test_parse_process_options_decodes_flags():
    opts = parse_process_options("iet")
    assert opts.images and opts.tables and opts.equations
    assert not opts.skip_kg and opts.chunking == "F"

    opts = parse_process_options("R!")
    assert opts.skip_kg and opts.chunking == "R"
    assert not (opts.images or opts.tables or opts.equations)

    assert parse_process_options("P").chunking == "P"

    opts = parse_process_options("Cite!")
    assert opts.chunking == "C" and opts.chunking_explicit
    assert opts.images and opts.tables and opts.equations and opts.skip_kg

    opts = parse_process_options("")
    assert not (opts.images or opts.tables or opts.equations or opts.skip_kg)
    assert opts.chunking == "F"


def test_validate_process_options_rejects_invalid_combos():
    assert validate_process_options("iet") == []
    assert validate_process_options("R!") == []
    assert validate_process_options("Cite!") == []

    assert any("multiple chunking modes" in m for m in validate_process_options("FR"))
    errs = validate_process_options("CF")
    assert any("multiple chunking modes" in m for m in errs)
    assert any("F/R/V/P/C" in m for m in errs)
    assert any("'f'" in m for m in validate_process_options("f"))
    assert sum(1 for m in validate_process_options("xyz") if "unsupported character" in m) == 3


def test_lightrag_parser_rule_supports_options_suffix(monkeypatch):
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://fake-mineru")
    monkeypatch.delenv("DOCLING_ENDPOINT", raising=False)
    validate_parser_routing_config("docx:native-iet,*:legacy")
    validate_parser_routing_config("docx:native-Cite,*:legacy")

    with pytest.raises(ParserRoutingConfigError, match="multiple chunking modes"):
        validate_parser_routing_config("docx:native-FR,*:legacy")
    with pytest.raises(ParserRoutingConfigError, match="unsupported character"):
        validate_parser_routing_config("docx:native-Q,*:legacy")


def test_resolve_file_parser_directives_priority(monkeypatch):
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://fake-mineru")
    monkeypatch.setenv("LIGHTRAG_PARSER", "docx:native-iet,*:legacy")

    assert resolve_file_parser_directives("paper.[native-R!].docx") == ("native", "R!")
    assert resolve_file_parser_directives("plain.docx") == ("native", "iet")
    assert resolve_file_parser_directives("plain.[-!].docx") == ("native", "!")


def test_parser_routing_accepts_semicolon_rules(monkeypatch):
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://fake-mineru")
    monkeypatch.setenv("DOCLING_ENDPOINT", "http://fake-docling")

    rules = "*:mineru;html:docling"
    validate_parser_routing_config(rules)
    assert resolve_file_parser_engine("paper.pdf", parser_rules=rules) == "mineru"
    assert resolve_file_parser_engine("index.html", parser_rules=rules) == "docling"
    assert resolve_file_parser_engine("notes.txt", parser_rules=rules) == "legacy"


@pytest.mark.parametrize(
    "env, rules, match",
    [
        pytest.param(
            {"MINERU_LOCAL_ENDPOINT": None, "DOCLING_ENDPOINT": "http://fake-docling"},
            "*:mineru;html:docling",
            "MINERU_LOCAL_ENDPOINT",
            id="missing-mineru-endpoint",
        ),
        pytest.param(
            {"MINERU_LOCAL_ENDPOINT": "http://fake-mineru", "DOCLING_ENDPOINT": None},
            "*:mineru;html:docling",
            "DOCLING_ENDPOINT",
            id="missing-docling-endpoint",
        ),
        pytest.param(
            {"MINERU_LOCAL_ENDPOINT": "http://fake-mineru"},
            "*.pdf:mineru",
            r"\*\.pdf",
            id="glob-suffix-rejected",
        ),
        pytest.param(
            {"MINERU_LOCAL_ENDPOINT": "http://fake-mineru"},
            "pdf:unknown",
            "unsupported parser engine",
            id="unknown-engine",
        ),
        pytest.param(
            {"MINERU_LOCAL_ENDPOINT": "http://fake-mineru"},
            "pdf:native",
            "does not match any suffix",
            id="unsupported-engine-suffix",
        ),
    ],
)
def test_parser_routing_validation_rejects_misconfigured_rules(
    monkeypatch, env, rules, match
):
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    with pytest.raises(ParserRoutingConfigError, match=match):
        validate_parser_routing_config(rules)


def test_parser_routing_validation_honors_mineru_api_mode(monkeypatch):
    monkeypatch.setenv("DOCLING_ENDPOINT", "http://fake-docling")

    monkeypatch.setenv("MINERU_API_MODE", "official")
    monkeypatch.delenv("MINERU_API_TOKEN", raising=False)
    with pytest.raises(ParserRoutingConfigError, match="MINERU_API_TOKEN"):
        validate_parser_routing_config("pdf:mineru")
    monkeypatch.setenv("MINERU_API_TOKEN", "token")
    validate_parser_routing_config("pdf:mineru")

    monkeypatch.setenv("MINERU_API_MODE", "local")
    monkeypatch.delenv("MINERU_LOCAL_ENDPOINT", raising=False)
    with pytest.raises(ParserRoutingConfigError, match="MINERU_LOCAL_ENDPOINT"):
        validate_parser_routing_config("pdf:mineru")
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://fake-local")
    validate_parser_routing_config("pdf:mineru")


# ---------------------------------------------------------------------------
# doc_status metadata carry-over & state-machine stage persistence
# ---------------------------------------------------------------------------


def test_doc_status_metadata_carry_over_helper():
    """``doc_status_transition_metadata`` preserves long-lived per-doc fields
    and layers in transition extras while dropping empty/None directives."""

    class _StubStatusDoc:
        def __init__(self, metadata):
            self.metadata = metadata

    assert doc_status_transition_metadata(_StubStatusDoc({"process_options": "iet"})) == {
        "process_options": "iet"
    }
    assert doc_status_transition_metadata(
        _StubStatusDoc({"source_file": "demo.[mineru].pdf"})
    ) == {"source_file": "demo.[mineru].pdf"}
    assert doc_status_transition_metadata(
        _StubStatusDoc({"process_options": "R!"}),
        extra={"process_start_time": 12345},
    ) == {"process_options": "R!", "process_start_time": 12345}
    assert doc_status_transition_metadata(_StubStatusDoc({})) == {}
    assert doc_status_transition_metadata(None) == {}
    assert doc_status_transition_metadata(_StubStatusDoc({"process_options": ""})) == {}
    assert (
        doc_status_transition_metadata(_StubStatusDoc({"process_options": None})) == {}
    )


def test_carry_over_keys_grouped_by_stage():
    """Strict tuple-equality guard on ``_DOC_STATUS_METADATA_CARRY_OVER_KEYS``.

    Locks WebUI timeline grouping and guarantees ``kg_write_state``,
    ``kg_purge``, ``custom_chunk_patch``, and duplicate-demotion keys survive
    every status transition.
    """
    assert _DOC_STATUS_METADATA_CARRY_OVER_KEYS == (
        "process_options",
        "source_file",
        "parse_warnings",
        "chunk_opts",
        "custom_chunker",
        "parse_start_time",
        "parse_end_time",
        "parse_stage_skipped",
        "parse_format",
        "parse_engine",
        "smartheading_llm_cache_ids",
        "analyzing_start_time",
        "analyzing_end_time",
        "analyzing_stage_skipped",
        "custom_chunk_patch",
        "kg_write_state",
        "kg_purge",
        "is_duplicate",
        "duplicate_kind",
        "original_doc_id",
    )


async def test_doc_status_metadata_survives_processed_transition(tmp_path):
    """End-to-end: process_options and stage-skip signals survive to PROCESSED."""
    async with _open_rag(tmp_path) as rag:
        await rag.apipeline_enqueue_documents(
            "Some content body for chunking.",
            file_paths="metadata_carry.txt",
            track_id="track-md-carry",
            process_options="iet!",
        )
        doc_id = compute_mdhash_id("metadata_carry.txt", prefix="doc-")
        pending_status = await rag.doc_status.get_by_id(doc_id)
        assert pending_status is not None
        assert (pending_status.get("metadata") or {}).get("process_options") == "iet!"

        await rag.apipeline_process_enqueue_documents()

        final_status = await rag.doc_status.get_by_id(doc_id)
        assert final_status is not None
        assert status_value_text(final_status.get("status")) == "processed"
        metadata = final_status.get("metadata") or {}
        assert metadata.get("process_options") == "iet!"
        assert isinstance(metadata.get("parse_start_time"), int)
        assert metadata.get("parse_stage_skipped") is True
        assert "parse_end_time" not in metadata
        assert isinstance(metadata.get("analyzing_start_time"), int)
        assert metadata.get("analyzing_stage_skipped") is True
        assert "analyzing_end_time" not in metadata


async def test_legacy_source_file_name_migrated_and_carried_over(tmp_path):
    """Regression: pre-rename ``metadata['source_file_name']`` migrates to ``source_file``."""
    async with _open_rag(tmp_path) as rag:
        doc_id = compute_mdhash_id("legacy_hint.txt", prefix="doc-")
        await rag.full_docs.upsert(
            {
                doc_id: {
                    "content": "legacy body for chunking.",
                    "file_path": "legacy_hint.txt",
                    "parse_format": "raw",
                    "parse_engine": "legacy",
                    "content_hash": "legacyhash",
                }
            }
        )
        await rag.doc_status.upsert(
            {
                doc_id: {
                    "status": DocStatus.PENDING,
                    "content_summary": "legacy body for chunking.",
                    "content_length": len("legacy body for chunking."),
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:01+00:00",
                    "file_path": "legacy_hint.txt",
                    "track_id": "track-legacy",
                    "content_hash": "legacyhash",
                    "metadata": {"source_file_name": "legacy_hint.[mineru].pdf"},
                }
            }
        )

        await rag.apipeline_process_enqueue_documents()

        final_status = await rag.doc_status.get_by_id(doc_id)
        assert final_status is not None
        assert status_value_text(final_status.get("status")) == "processed"
        assert (final_status.get("metadata") or {}).get(
            "source_file"
        ) == "legacy_hint.[mineru].pdf"


async def test_analyze_soft_failure_writes_neither_end_time_nor_skipped(tmp_path):
    """Soft-failing analyze_multimodal leaves both analyzing_end_time and skipped absent."""
    async with _open_rag(tmp_path) as rag:
        original_analyze = rag.analyze_multimodal

        async def _soft_failing_analyze(*args, **kwargs):
            result = await original_analyze(*args, **kwargs)
            result.pop("analyzing_stage_skipped", None)
            result.pop("multimodal_processed", None)
            return result

        rag.analyze_multimodal = _soft_failing_analyze  # type: ignore[assignment]

        await rag.apipeline_enqueue_documents(
            "Soft-fail body for analyze stage.",
            file_paths="analyze_soft_fail.txt",
            track_id="track-analyze-softfail",
            process_options="iet!",
        )
        doc_id = compute_mdhash_id("analyze_soft_fail.txt", prefix="doc-")
        await rag.apipeline_process_enqueue_documents()

        final_status = await rag.doc_status.get_by_id(doc_id)
        assert final_status is not None
        assert status_value_text(final_status.get("status")) == "processed"
        metadata = final_status.get("metadata") or {}
        assert isinstance(metadata.get("analyzing_start_time"), int)
        assert "analyzing_end_time" not in metadata
        assert "analyzing_stage_skipped" not in metadata


async def test_stage_end_outcomes_persist_within_their_own_stage(tmp_path):
    """Full PARSING -> ANALYZING -> PROCESSING -> PROCESSED flow with per-stage end signals."""
    calls: list[tuple[str, set[str]]] = []
    async with _open_rag(tmp_path) as rag:
        original_upsert = rag.doc_status.upsert

        async def _recording_upsert(data, *args, **kwargs):
            for payload in data.values():
                calls.append(
                    (
                        status_value_text(payload.get("status")),
                        set((payload.get("metadata") or {}).keys()),
                    )
                )
            return await original_upsert(data, *args, **kwargs)

        rag.doc_status.upsert = _recording_upsert  # type: ignore[assignment]

        await rag.apipeline_enqueue_documents(
            "Some content body for chunking.",
            file_paths="stage_end_timing.txt",
            track_id="track-stage-end-timing",
        )
        await rag.apipeline_process_enqueue_documents()

    first_analyzing = next(
        (i for i, (s, _) in enumerate(calls) if s == "analyzing"), None
    )
    first_processing = next(
        (i for i, (s, _) in enumerate(calls) if s == "processing"), None
    )
    first_processed = next(
        (i for i, (s, _) in enumerate(calls) if s == "processed"), None
    )
    assert first_analyzing is not None, f"no ANALYZING upsert; sequence: {calls!r}"
    assert first_processing is not None, f"no PROCESSING upsert; sequence: {calls!r}"
    assert first_processed is not None, f"no PROCESSED upsert; sequence: {calls!r}"
    assert first_analyzing < first_processing < first_processed
    assert any(
        s == "parsing" and "parse_stage_skipped" in keys
        for s, keys in calls[:first_analyzing]
    )
    assert any(
        s == "analyzing" and "analyzing_stage_skipped" in keys
        for s, keys in calls[first_analyzing:first_processing]
    )


async def test_apipeline_enqueue_persists_process_options(tmp_path):
    async with _open_rag(tmp_path) as rag:
        await rag.apipeline_enqueue_documents(
            "alpha body",
            file_paths="abc.[native-R!].docx",
            track_id="track-opts",
            process_options="R!",
        )
        doc_id = compute_mdhash_id("abc.docx", prefix="doc-")
        full_doc = await rag.full_docs.get_by_id(doc_id)
        assert full_doc is not None
        assert full_doc["file_path"] == "abc.docx"
        assert "canonical_basename" not in full_doc
        assert full_doc.get("process_options") == "R!"

        status_doc = await rag.doc_status.get_by_id(doc_id)
        assert status_doc is not None
        metadata = (
            status_doc.get("metadata")
            if isinstance(status_doc, dict)
            else getattr(status_doc, "metadata", {})
        )
        assert metadata.get("process_options") == "R!"


# ---------------------------------------------------------------------------
# Resume purge gating
# ---------------------------------------------------------------------------


async def test_resume_purges_old_chunks_when_content_already_extracted(tmp_path):
    """Resume calls ``_purge_doc_chunks_and_kg`` with prior stale chunk IDs."""
    async with _open_rag(tmp_path) as rag:
        doc_id = compute_mdhash_id("resume.txt", prefix="doc-")
        await rag.full_docs.upsert(
            {
                doc_id: {
                    "content": "previously extracted body",
                    "file_path": "resume.txt",
                    "parse_format": "raw",
                    "parse_engine": "legacy",
                    "content_hash": "deadbeef",
                }
            }
        )
        stale_chunks = [f"{doc_id}-chunk-{i:03d}" for i in range(2)]
        await rag.doc_status.upsert(
            {
                doc_id: {
                    "status": DocStatus.PROCESSING,
                    "content_summary": "previously extracted body",
                    "content_length": len("previously extracted body"),
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:01+00:00",
                    "file_path": "resume.txt",
                    "track_id": "track-resume",
                    "content_hash": "deadbeef",
                    "chunks_list": stale_chunks,
                    "chunks_count": len(stale_chunks),
                }
            }
        )

        calls: list[tuple[str, set[str]]] = []
        original = rag._purge_doc_chunks_and_kg

        class _ResumePurged(Exception):
            """Raised after the real purge ran, ending the pipeline there."""

        async def _wrapped(doc_id_arg, chunk_ids_arg, **kwargs):
            calls.append((doc_id_arg, set(chunk_ids_arg)))
            await original(doc_id_arg, chunk_ids_arg, **kwargs)
            raise _ResumePurged()

        rag._purge_doc_chunks_and_kg = _wrapped  # type: ignore[method-assign]

        await rag.apipeline_process_enqueue_documents()

        assert calls == [(doc_id, set(stale_chunks))]


async def test_resume_skips_purge_when_chunks_list_empty(tmp_path):
    """Resume skips ``_purge_doc_chunks_and_kg`` when ``chunks_list`` is empty."""
    async with _open_rag(tmp_path) as rag:
        doc_id = compute_mdhash_id("noskip.txt", prefix="doc-")
        await rag.full_docs.upsert(
            {
                doc_id: {
                    "content": "fresh body",
                    "file_path": "noskip.txt",
                    "parse_format": "raw",
                    "parse_engine": "legacy",
                    "content_hash": "fresh",
                }
            }
        )
        await rag.doc_status.upsert(
            {
                doc_id: {
                    "status": DocStatus.PARSING,
                    "content_summary": "fresh body",
                    "content_length": len("fresh body"),
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:01+00:00",
                    "file_path": "noskip.txt",
                    "track_id": "track-noskip",
                    "content_hash": "fresh",
                    "chunks_list": [],
                    "chunks_count": 0,
                }
            }
        )

        calls: list[tuple[str, set[str]]] = []

        async def _spy(doc_id_arg, chunk_ids_arg, **kwargs):
            calls.append((doc_id_arg, set(chunk_ids_arg)))
            raise RuntimeError("test stop after purge check")

        rag._purge_doc_chunks_and_kg = _spy  # type: ignore[method-assign]

        await rag.apipeline_process_enqueue_documents()

        assert calls == []


# ---------------------------------------------------------------------------
# Enqueue guards, ingress wakeup, busy slot handoff & bookkeeping
# ---------------------------------------------------------------------------


async def test_apipeline_enqueue_rejects_when_scanning(tmp_path):
    """``scanning_exclusive`` blocks enqueue; scan processing phase and idle allow it."""
    async with _open_rag(tmp_path) as rag:
        pipeline_status, lock = await pipeline_status_handles(rag)

        async with lock:
            pipeline_status["scanning"] = True
            pipeline_status["scanning_exclusive"] = True
        try:
            with pytest.raises(RuntimeError, match="scan is classifying"):
                await rag.apipeline_enqueue_documents(
                    "should not enqueue", file_paths="scan.txt", track_id="track-scan"
                )
        finally:
            async with lock:
                pipeline_status["scanning"] = False
                pipeline_status["scanning_exclusive"] = False

        async with lock:
            pipeline_status["scanning"] = True
            pipeline_status["scanning_exclusive"] = False
        try:
            track = await rag.apipeline_enqueue_documents(
                "allowed during scan processing",
                file_paths="scan_processing.txt",
                track_id="track-scan-processing",
            )
            assert track == "track-scan-processing"
        finally:
            async with lock:
                pipeline_status["scanning"] = False

        await rag.apipeline_enqueue_documents(
            "now allowed", file_paths="ok.txt", track_id="track-ok"
        )


async def test_apipeline_enqueue_rejects_when_destructive_busy(tmp_path):
    """``destructive_busy`` rejects both normal and ``from_scan=True`` enqueues."""
    async with _open_rag(tmp_path) as rag:
        pipeline_status, lock = await pipeline_status_handles(rag)
        async with lock:
            pipeline_status["busy"] = True
            pipeline_status["destructive_busy"] = True
        try:
            with pytest.raises(RuntimeError, match="clearing or deleting"):
                await rag.apipeline_enqueue_documents(
                    "should not enqueue",
                    file_paths="while_clearing.txt",
                    track_id="track-clearing",
                )
            with pytest.raises(RuntimeError, match="clearing or deleting"):
                await rag.apipeline_enqueue_documents(
                    "should not enqueue",
                    file_paths="while_clearing_scan.txt",
                    track_id="track-clearing-scan",
                    from_scan=True,
                )
        finally:
            async with lock:
                pipeline_status["busy"] = False
                pipeline_status["destructive_busy"] = False


async def test_enqueue_publishes_document_messages(tmp_path):
    """Enqueue publishes one document message per stored doc, busy or idle."""
    async with _open_rag(tmp_path) as rag:
        pipeline_status, lock = await pipeline_status_handles(rag)
        ingress = await get_pipeline_ingress(rag.workspace)

        async with lock:
            pipeline_status["busy"] = True
        try:
            await rag.apipeline_enqueue_documents(
                "first while busy", file_paths="first.txt", track_id="track-first"
            )
            assert ingress.counts()["documents"] == 1
            await rag.apipeline_enqueue_documents(
                "second while busy", file_paths="second.txt", track_id="track-second"
            )
            assert ingress.counts()["documents"] == 2
        finally:
            async with lock:
                pipeline_status["busy"] = False

        await rag.apipeline_enqueue_documents(
            "while idle", file_paths="idle.txt", track_id="track-idle"
        )
        assert ingress.counts()["documents"] == 3


async def test_enqueue_status_upsert_failure_still_nudges_busy_loop(
    tmp_path, monkeypatch
):
    """Partial DocStatus upsert failure while busy still arms auto-rescan."""
    async with _open_rag(tmp_path) as rag:
        pipeline_status, lock = await pipeline_status_handles(rag)
        ingress = await get_pipeline_ingress(rag.workspace)
        async with lock:
            pipeline_status["busy"] = True

        original_upsert = rag.doc_status.upsert
        process_calls: list[bool] = []

        async def _partial_upsert_then_raise(data):
            await original_upsert(data)
            raise RuntimeError("partial doc-status bulk failure")

        async def _unexpected_process():
            process_calls.append(True)
            raise AssertionError("busy enqueue should nudge, not start processing")

        monkeypatch.setattr(rag.doc_status, "upsert", _partial_upsert_then_raise)
        monkeypatch.setattr(
            rag, "apipeline_process_enqueue_documents", _unexpected_process
        )
        try:
            with pytest.raises(RuntimeError, match="partial doc-status"):
                await rag.apipeline_enqueue_documents(
                    "partial busy",
                    file_paths="partial-busy.txt",
                    track_id="track-partial-busy",
                )
            assert ingress.counts()["auto_rescan_pending"] is True
            assert process_calls == []
        finally:
            ingress.consume_auto_rescan()
            async with lock:
                pipeline_status["busy"] = False


async def test_acquire_failure_does_not_stamp_stopped_bookkeeping(
    tmp_path, monkeypatch
):
    """A run whose acquire raises never owned the slot and stamps no stop record."""
    async with _open_rag(tmp_path) as rag:
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
        baseline_history = list(pipeline_status["history_messages"])

        async def _acquire_boom(*args, **kwargs):
            raise RuntimeError("arm transport down")

        monkeypatch.setattr(
            pipeline_module, "acquire_processing_reservation", _acquire_boom
        )
        with pytest.raises(RuntimeError, match="arm transport down"):
            await rag.apipeline_process_enqueue_documents()

        assert pipeline_status.get("busy") is False
        assert "stopped" not in (pipeline_status.get("latest_message") or "").lower()
        assert list(pipeline_status["history_messages"]) == baseline_history


async def test_enqueue_publish_failure_falls_back_to_auto_rescan(tmp_path, monkeypatch):
    """Failed document publish after commit falls back to arming auto-rescan."""
    async with _open_rag(tmp_path) as rag:
        ingress = await get_pipeline_ingress(rag.workspace)

        def _publish_boom(messages):
            raise RuntimeError("publish transport down")

        monkeypatch.setattr(ingress, "put_documents", _publish_boom)

        returned = await rag.apipeline_enqueue_documents(
            "publish fails",
            file_paths="publish-fails.txt",
            track_id="track-publish-fails",
        )
        assert returned == "track-publish-fails"
        assert ingress.counts()["documents"] == 0
        assert ingress.counts()["auto_rescan_pending"] is True


async def test_enqueue_status_upsert_failure_starts_idle_processing(
    tmp_path, monkeypatch
):
    """Partial DocStatus upsert failure while idle triggers best-effort processing."""
    async with _open_rag(tmp_path) as rag:
        original_upsert = rag.doc_status.upsert
        process_called = False

        async def _partial_upsert_then_raise(data):
            await original_upsert(data)
            raise RuntimeError("partial doc-status bulk failure")

        async def _record_process():
            nonlocal process_called
            process_called = True
            assert await rag.doc_status.get_docs_by_statuses([DocStatus.PENDING])

        monkeypatch.setattr(rag.doc_status, "upsert", _partial_upsert_then_raise)
        monkeypatch.setattr(rag, "apipeline_process_enqueue_documents", _record_process)

        with pytest.raises(RuntimeError, match="partial doc-status"):
            await rag.apipeline_enqueue_documents(
                "partial idle",
                file_paths="partial-idle.txt",
                track_id="track-partial-idle",
            )
        assert process_called is True


async def test_enqueue_status_upsert_failure_survives_wake_failure(
    tmp_path, monkeypatch, lightrag_caplog
):
    """A failing best-effort wake logs a warning and still surfaces the storage error."""
    async with _open_rag(tmp_path) as rag:
        original_upsert = rag.doc_status.upsert

        async def _partial_upsert_then_raise(data):
            await original_upsert(data)
            raise RuntimeError("partial doc-status bulk failure")

        async def _failing_process():
            raise RuntimeError("wake boom")

        monkeypatch.setattr(rag.doc_status, "upsert", _partial_upsert_then_raise)
        monkeypatch.setattr(
            rag, "apipeline_process_enqueue_documents", _failing_process
        )

        with lightrag_caplog.at_level(logging.WARNING):
            with pytest.raises(RuntimeError, match="partial doc-status"):
                await rag.apipeline_enqueue_documents(
                    "wake failure",
                    file_paths="wake-failure.txt",
                    track_id="track-wake-failure",
                )
        assert "Failed to start document processing" in lightrag_caplog.text
        assert "wake boom" in lightrag_caplog.text


async def test_process_enqueue_holding_busy_releases_when_no_docs(tmp_path):
    """``_holding_busy=True`` takes over an already-held slot and releases it when idle."""
    async with _open_rag(tmp_path) as rag:
        pipeline_status, lock = await pipeline_status_handles(rag)
        handoff_token = "handoff-token"
        async with lock:
            pipeline_status.update({"busy": True, "busy_owner": handoff_token})

        await rag.apipeline_process_enqueue_documents(
            _holding_busy=True, token=handoff_token
        )
        assert pipeline_status.get("busy") is False
        assert pipeline_status.get("busy_owner") is None


async def test_finalize_does_not_clobber_new_owner_bookkeeping(tmp_path):
    """Regression: owner-checked finally must not clear a new owner's flags/message."""
    async with _open_rag(tmp_path) as rag:
        pipeline_status, lock = await pipeline_status_handles(rag)
        token = "holder-token"
        async with lock:
            pipeline_status.update({"busy": True, "busy_owner": token})

        real_rtc = pipeline_module.run_to_completion

        async def _inject(factory):
            async with lock:
                pipeline_status.update(
                    {
                        "busy": True,
                        "busy_owner": "new-owner",
                        "cancellation_requested": True,
                        "latest_message": "new job running",
                    }
                )
            return await real_rtc(factory)

        with mock.patch.object(pipeline_module, "run_to_completion", _inject):
            await rag.apipeline_process_enqueue_documents(
                _holding_busy=True, token=token
            )

        assert pipeline_status.get("cancellation_requested") is True
        assert pipeline_status.get("busy_owner") == "new-owner"
        assert pipeline_status.get("latest_message") == "new job running"


async def test_idle_trigger_releases_slot_silently_without_stop_message(
    tmp_path, monkeypatch
):
    """Empty-queue trigger takes the busy slot for its read and releases silently."""
    calls = {"acquire": 0}
    real_acquire = pipeline_module.acquire_processing_reservation

    async def _spy_acquire(*args, **kwargs):
        calls["acquire"] += 1
        return await real_acquire(*args, **kwargs)

    monkeypatch.setattr(
        pipeline_module, "acquire_processing_reservation", _spy_acquire
    )

    async with _open_rag(tmp_path) as rag:
        await rag.apipeline_process_enqueue_documents()

        assert calls["acquire"] == 1
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
        assert not pipeline_status.get("busy")
        assert pipeline_status.get("busy_owner") is None
        history = [
            str(m).lower() for m in (pipeline_status.get("history_messages") or [])
        ]
        assert not any("stopped" in m for m in history)
        assert (
            "stopped" not in str(pipeline_status.get("latest_message") or "").lower()
        )


# ---------------------------------------------------------------------------
# Enqueue dedup, file-path canonicalization & content-hash persistence
# ---------------------------------------------------------------------------


async def test_concurrent_enqueue_dedupes_same_content_different_filenames(tmp_path):
    """Workspace serialization prevents two concurrent enqueues from both landing PENDING."""
    async with _open_rag(tmp_path) as rag:
        original = pipeline_module.get_existing_doc_by_content_hash

        async def yielding_get_by_content_hash(doc_status, content_hash, **kwargs):
            await asyncio.sleep(0)
            return await original(doc_status, content_hash, **kwargs)

        with mock.patch.object(
            pipeline_module,
            "get_existing_doc_by_content_hash",
            yielding_get_by_content_hash,
        ):
            shared = "shared content for dedup race"
            results = await asyncio.gather(
                rag.apipeline_enqueue_documents(
                    shared, file_paths="first.txt", track_id="track-first"
                ),
                rag.apipeline_enqueue_documents(
                    shared, file_paths="second.txt", track_id="track-second"
                ),
            )
            assert results == ["track-first", None]

        pending = await rag.doc_status.get_docs_by_statuses([DocStatus.PENDING])
        assert len(pending) == 1
        failed = await rag.doc_status.get_docs_by_statuses([DocStatus.FAILED])
        dups = [
            d
            for d in failed.values()
            if getattr(d, "metadata", None)
            and d.metadata.get("duplicate_kind") == "content_hash"
        ]
        assert len(dups) == 1
        assert dups[0].metadata["is_duplicate"] is True
        assert dups[0].metadata["original_doc_id"]


async def test_enqueue_dedupes_by_filename_and_content_hash(tmp_path):
    async with _open_rag(tmp_path) as rag:
        await rag.apipeline_enqueue_documents(
            ["alpha body", "beta body"],
            file_paths=["first.txt", "second.txt"],
            track_id="track-a",
        )
        first_id = compute_mdhash_id("first.txt", prefix="doc-")
        second_id = compute_mdhash_id("second.txt", prefix="doc-")
        first_doc = await rag.full_docs.get_by_id(first_id)
        second_doc = await rag.full_docs.get_by_id(second_id)
        assert first_doc is not None and second_doc is not None
        assert first_doc["content_hash"] != second_doc["content_hash"]

        await rag.apipeline_enqueue_documents(
            "changed content", file_paths="/tmp/first.txt", track_id="track-b"
        )
        assert (await rag.full_docs.get_by_id(first_id))["content"] == "alpha body"

        await rag.apipeline_enqueue_documents(
            "alpha body", file_paths="third.txt", track_id="track-c"
        )
        assert (
            await rag.full_docs.get_by_id(compute_mdhash_id("third.txt", prefix="doc-"))
            is None
        )

        failed = await rag.doc_status.get_docs_by_statuses([DocStatus.FAILED])
        kinds = {
            getattr(doc, "metadata", {}).get("duplicate_kind")
            for doc in failed.values()
            if getattr(doc, "metadata", {}).get("is_duplicate")
        }
        assert {"filename", "content_hash"}.issubset(kinds)


async def test_enqueue_dedupes_parser_hinted_filename_variants(tmp_path):
    async with _open_rag(tmp_path) as rag:
        await rag.apipeline_enqueue_documents(
            "alpha body", file_paths="abc.docx", track_id="track-a"
        )
        first_id = compute_mdhash_id("abc.docx", prefix="doc-")
        first_doc = await rag.full_docs.get_by_id(first_id)
        assert first_doc is not None
        assert first_doc["file_path"] == "abc.docx"
        assert "canonical_basename" not in first_doc

        await rag.apipeline_enqueue_documents(
            "changed body",
            file_paths="/tmp/abc.[native].docx",
            track_id="track-b",
        )
        assert (await rag.full_docs.get_by_id(first_id))["content"] == "alpha body"

        failed = await rag.doc_status.get_docs_by_statuses([DocStatus.FAILED])
        assert any(
            getattr(doc, "metadata", {}).get("duplicate_kind") == "filename"
            and getattr(doc, "file_path", "") == "abc.docx"
            for doc in failed.values()
        )


async def test_delete_result_uses_canonical_file_path(tmp_path):
    async with _open_rag(tmp_path) as rag:
        await rag.apipeline_enqueue_documents(
            "",
            file_paths=str(tmp_path / "abc.[native].docx"),
            docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
            parse_engine=PARSER_ENGINE_NATIVE,
            track_id="track-delete-source",
        )
        doc_id = compute_mdhash_id("abc.docx", prefix="doc-")
        result = await rag.adelete_by_doc_id(doc_id)

        assert result.status == "success"
        assert result.file_path == "abc.docx"
        assert not hasattr(result, "source_path")


async def test_enqueue_without_file_paths_uses_content_ids(tmp_path):
    async with _open_rag(tmp_path) as rag:
        docs = ["alpha without source", "beta without source"]
        await rag.apipeline_enqueue_documents(docs, track_id="track-no-source")

        for content in docs:
            doc_id = compute_mdhash_id(content, prefix="doc-")
            full_doc = await rag.full_docs.get_by_id(doc_id)
            status = await rag.doc_status.get_by_id(doc_id)
            assert full_doc is not None and status is not None
            assert full_doc["content"] == content
            assert full_doc["file_path"] == "unknown_source"
            assert status["file_path"] == "unknown_source"
            assert full_doc.get("content_hash")
            assert status.get("content_hash") == full_doc.get("content_hash")

        failed = await rag.doc_status.get_docs_by_statuses([DocStatus.FAILED])
        assert [
            doc
            for doc in failed.values()
            if getattr(doc, "track_id", "") == "track-no-source"
            and getattr(doc, "metadata", {}).get("is_duplicate")
        ] == []


async def test_legacy_empty_file_paths_do_not_block_unsourced_insert(tmp_path):
    async with _open_rag(tmp_path) as rag:
        await rag.doc_status.upsert(
            {
                "legacy-empty": {
                    "status": DocStatus.PROCESSED,
                    "content_summary": "legacy empty",
                    "content_length": 0,
                    "file_path": "",
                    "track_id": "legacy",
                    "created_at": "2025-01-01T00:00:00+00:00",
                    "updated_at": "2025-01-01T00:00:00+00:00",
                    "chunks_list": [],
                },
                "legacy-no-file": {
                    "status": DocStatus.PROCESSED,
                    "content_summary": "legacy no-file",
                    "content_length": 0,
                    "file_path": "no-file-path",
                    "track_id": "legacy",
                    "created_at": "2025-01-01T00:00:00+00:00",
                    "updated_at": "2025-01-01T00:00:00+00:00",
                    "chunks_list": [],
                },
            }
        )
        content = "fresh unsourced body"
        await rag.apipeline_enqueue_documents(content, track_id="track-fresh")
        doc_id = compute_mdhash_id(content, prefix="doc-")
        full_doc = await rag.full_docs.get_by_id(doc_id)
        status = await rag.doc_status.get_by_id(doc_id)
        assert full_doc is not None and status is not None
        assert full_doc["file_path"] == "unknown_source"
        assert status["file_path"] == "unknown_source"


async def test_basename_lookup_requires_canonical_stored_file_path(tmp_path):
    async with _open_rag(tmp_path) as rag:
        await rag.doc_status.upsert(
            {
                "doc-noncanonical-1": {
                    "status": DocStatus.PROCESSED,
                    "content_summary": "noncanonical",
                    "content_length": 7,
                    "file_path": "/inputs/legacy.txt",
                    "track_id": "noncanonical-track",
                    "created_at": "2025-01-01T00:00:00+00:00",
                    "updated_at": "2025-01-01T00:00:00+00:00",
                    "chunks_list": [],
                }
            }
        )
        assert await rag.doc_status.get_doc_by_file_basename("legacy.txt") is None

        await rag.apipeline_enqueue_documents(
            "fresh body", file_paths="legacy.txt", track_id="track-x"
        )
        new_id = compute_mdhash_id("legacy.txt", prefix="doc-")
        assert (await rag.full_docs.get_by_id(new_id))["file_path"] == "legacy.txt"
        assert (await rag.doc_status.get_by_id(new_id))["file_path"] == "legacy.txt"


async def test_enqueue_rejects_removed_or_unknown_docs_format(tmp_path):
    async with _open_rag(tmp_path) as rag:
        for docs_format in ("lightrag", "bogus"):
            with pytest.raises(ValueError, match="Unsupported docs_format"):
                await rag.apipeline_enqueue_documents(
                    "some content",
                    file_paths="doc.txt",
                    docs_format=docs_format,
                )
        with pytest.raises(TypeError):
            await rag.apipeline_enqueue_documents(  # type: ignore[call-arg]
                "",
                file_paths="first.lightrag",
                lightrag_document_paths="__parsed__/doc.blocks.jsonl",
            )
        assert await rag.doc_status.get_docs_by_statuses([DocStatus.FAILED]) == {}


async def test_persist_parsed_full_docs_syncs_hash_to_doc_status(tmp_path):
    async with _open_rag(tmp_path) as rag:
        await rag.apipeline_enqueue_documents(
            "",
            file_paths="pending.txt",
            docs_format="pending_parse",
            track_id="track-pending",
        )
        doc_id = compute_mdhash_id("pending.txt", prefix="doc-")
        assert (await rag.full_docs.get_by_id(doc_id)).get("content_hash") in (None, "")
        assert (await rag.doc_status.get_by_id(doc_id)).get("content_hash") in (
            None,
            "",
        )

        content = "extracted body text"
        await rag._persist_parsed_full_docs(
            doc_id,
            {
                "content": content,
                "file_path": "pending.txt",
                "parse_format": "raw",
                "parse_engine": "native",
            },
        )
        expected_hash = compute_mdhash_id(content, prefix="")
        assert (await rag.full_docs.get_by_id(doc_id))["content_hash"] == expected_hash
        assert (await rag.doc_status.get_by_id(doc_id))["content_hash"] == expected_hash
        match = await rag.doc_status.get_doc_by_content_hash(expected_hash)
        assert match is not None and match[0] == doc_id


async def test_persist_parsed_full_docs_preserves_pending_metadata(tmp_path):
    async with _open_rag(tmp_path) as rag:
        await rag.apipeline_enqueue_documents(
            "",
            file_paths="report.[native-iet!].docx",
            docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
            parse_engine=PARSER_ENGINE_NATIVE,
            process_options="iet!",
            track_id="track-merge",
        )
        doc_id = compute_mdhash_id("report.docx", prefix="doc-")
        pre = await rag.full_docs.get_by_id(doc_id)
        assert pre is not None
        assert pre.get("process_options") == "iet!"
        assert "canonical_basename" not in pre
        assert pre.get("file_path") == "report.docx"

        await rag._persist_parsed_full_docs(
            doc_id,
            {
                "content": "extracted body",
                "file_path": "report.docx",
                "parse_format": "raw",
                "parse_engine": PARSER_ENGINE_NATIVE,
                "update_time": 12345,
            },
        )
        post = await rag.full_docs.get_by_id(doc_id)
        assert post is not None
        assert post["content"] == "extracted body"
        assert post["parse_format"] == "raw"
        assert post.get("process_options") == "iet!"
        assert post.get("file_path") == "report.docx"
        assert post["content_hash"] == compute_mdhash_id("extracted body", prefix="")


async def test_state_machine_upsert_preserves_content_hash(tmp_path):
    async with _open_rag(tmp_path) as rag:
        await rag.apipeline_enqueue_documents(
            "raw body", file_paths="alpha.txt", track_id="track-state"
        )
        doc_id = compute_mdhash_id("alpha.txt", prefix="doc-")
        initial_hash = (await rag.doc_status.get_by_id(doc_id))["content_hash"]
        assert initial_hash

        status_doc = (
            await rag.doc_status.get_docs_by_statuses([DocStatus.PENDING])
        )[doc_id]
        for next_status in (
            DocStatus.PARSING,
            DocStatus.ANALYZING,
            DocStatus.PROCESSED,
        ):
            await rag.doc_status.upsert(
                {
                    doc_id: {
                        "status": next_status,
                        "content_summary": status_doc.content_summary,
                        "content_length": status_doc.content_length,
                        "created_at": status_doc.created_at,
                        "updated_at": "now",
                        "file_path": status_doc.file_path,
                        "track_id": status_doc.track_id,
                        "content_hash": status_doc.content_hash,
                    }
                }
            )
            current = await rag.doc_status.get_by_id(doc_id)
            assert current.get("content_hash") == initial_hash
            match = await rag.doc_status.get_doc_by_content_hash(initial_hash)
            assert match is not None and match[0] == doc_id


async def test_pending_parse_duplicate_hash_fails_and_archives_source(
    tmp_path, monkeypatch
):
    """Two PENDING_PARSE docx files with identical extracted bodies deduplicate & archive."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setenv("INPUT_DIR", str(input_dir))

    class _FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: D401
            return datetime(2026, 1, 1, tzinfo=tz or timezone.utc)

    monkeypatch.setattr(lightrag_module, "datetime", _FrozenDateTime)
    monkeypatch.setattr(pipeline_module, "datetime", _FrozenDateTime)
    stable_block = {
        "uuid": "p1",
        "uuid_end": "p1",
        "heading": "",
        "content": "same extracted body",
        "type": "text",
        "parent_headings": [],
        "level": 0,
        "table_chunk_role": "none",
    }
    monkeypatch.setattr(
        "lightrag.parser.docx.parse_document.extract_docx_blocks",
        lambda *args, **kwargs: [dict(stable_block)],
    )

    async with _open_rag(tmp_path / "work") as rag:
        original_path = input_dir / "original.docx"
        original_path.write_bytes(b"original docx bytes")
        await rag.apipeline_enqueue_documents(
            "",
            file_paths=str(original_path),
            docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
            parse_engine=PARSER_ENGINE_NATIVE,
            track_id="track-original",
        )
        await rag.apipeline_process_enqueue_documents()
        original_id = compute_mdhash_id("original.docx", prefix="doc-")
        original_status = await rag.doc_status.get_by_id(original_id)
        assert original_status is not None
        original_status["status"] = DocStatus.PROCESSED
        await rag.doc_status.upsert({original_id: original_status})

        source_path = input_dir / "duplicate.docx"
        source_path.write_bytes(b"docx bytes")
        extract_calls: list[bool] = []

        async def _fail_extract(*args, **kwargs):
            extract_calls.append(True)
            raise AssertionError("duplicate document should not reach extraction")

        monkeypatch.setattr(rag, "_process_extract_entities", _fail_extract)

        await rag.apipeline_enqueue_documents(
            "",
            file_paths=str(source_path),
            docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
            parse_engine=PARSER_ENGINE_NATIVE,
            track_id="track-dup",
        )
        await rag.apipeline_process_enqueue_documents()

        duplicate_id = compute_mdhash_id("duplicate.docx", prefix="doc-")
        duplicate_status = await rag.doc_status.get_by_id(duplicate_id)
        assert duplicate_status["status"] == DocStatus.FAILED
        assert duplicate_status["metadata"]["is_duplicate"] is True
        assert duplicate_status["metadata"]["duplicate_kind"] == "content_hash"
        assert duplicate_status["metadata"]["original_doc_id"] == original_id
        assert extract_calls == []
        assert not source_path.exists()
        assert (input_dir / PARSED_DIR_NAME / source_path.name).exists()


# ---------------------------------------------------------------------------
# Multimodal analysis & VDB timeouts
# ---------------------------------------------------------------------------


async def test_analyze_multimodal_overwrites_already_analyzed_items(tmp_path):
    """Re-running analyze_multimodal recomputes enabled modalities and overwrites prior results."""
    vlm_calls = {"n": 0}
    extract_calls = {"n": 0}

    async def _vlm(_prompt, **_kwargs):
        vlm_calls["n"] += 1
        return json.dumps({"name": "Image", "type": "Chart", "description": "details"})

    async def _extract(_prompt, **_kwargs):
        extract_calls["n"] += 1
        return json.dumps({"name": "Item", "description": "table content summary"})

    img_path = tmp_path / "img1.png"
    img_path.write_bytes(PNG_BYTES)
    blocks, sidecars = _write_sidecar_doc(
        tmp_path,
        drawings={
            "id1": {
                "id": "id1",
                "caption": "fig1",
                "path": str(img_path),
                "llm_analyze_result": {
                    "name": "Existing",
                    "type": "Photo",
                    "description": "kept as-is",
                    "analyze_time": 1700000000,
                    "status": "success",
                    "message": "",
                },
            }
        },
        tables={
            "tbl1": {
                "id": "tbl1",
                "caption": "tbl",
                "format": "html",
                "content": "Header|Row",
            }
        },
    )

    async with _open_rag(
        tmp_path, vlm_llm_model_func=_vlm, extract_llm_model_func=_extract
    ) as rag:
        parsed = {
            "doc_id": "doc-1",
            "file_path": "demo.pdf",
            "blocks_path": str(blocks),
            "content": "body",
        }
        await rag.analyze_multimodal("doc-1", "demo.pdf", parsed, process_options="it")

    existing = json.loads(sidecars["drawings"].read_text(encoding="utf-8"))["drawings"][
        "id1"
    ]["llm_analyze_result"]
    assert existing["name"] == "Image"
    assert existing["description"] == "details"
    assert existing["status"] == "success"

    new_result = json.loads(sidecars["tables"].read_text(encoding="utf-8"))["tables"][
        "tbl1"
    ]["llm_analyze_result"]
    assert new_result["name"] == "Item"
    assert new_result["status"] == "success"
    assert vlm_calls["n"] == 1
    assert extract_calls["n"] == 1


async def test_analyze_multimodal_uses_effective_vlm_max_async_when_role_none(tmp_path):
    rag = _new_rag(tmp_path, llm_model_max_async=3, vlm_llm_model_max_async=None)
    blocks, _ = _write_sidecar_doc(
        tmp_path, blocks_lines=[{"type": "meta", "format_version": "1.0"}]
    )
    parsed = {
        "doc_id": "doc-1",
        "file_path": "demo.pdf",
        "blocks_path": str(blocks),
        "content": "body",
    }
    result = await rag.analyze_multimodal(
        "doc-1", "demo.pdf", parsed, process_options="ite"
    )
    assert result["multimodal_processed"] is True


async def test_analyze_multimodal_unknown_image_type_folds_to_other(tmp_path):
    """Out-of-enum VLM ``type`` folds to ``Other`` rather than failing the document."""

    async def _vlm(_prompt, **_kwargs):
        return json.dumps(
            {"name": "Figure A", "type": "diagram", "description": "details"},
            ensure_ascii=False,
        )

    img_path = tmp_path / "img1.png"
    img_path.write_bytes(PNG_BYTES)
    blocks, sidecars = _write_sidecar_doc(
        tmp_path,
        drawings={
            "id1": {
                "id": "id1",
                "caption": "图1 测试图",
                "footnotes": [],
                "path": str(img_path),
            }
        },
    )
    async with _open_rag(tmp_path, vlm_llm_model_func=_vlm) as rag:
        parsed = {
            "doc_id": "doc-1",
            "file_path": "demo.pdf",
            "blocks_path": str(blocks),
            "content": "body",
        }
        await rag.analyze_multimodal("doc-1", "demo.pdf", parsed, process_options="i")

    result = json.loads(sidecars["drawings"].read_text(encoding="utf-8"))["drawings"][
        "id1"
    ]["llm_analyze_result"]
    assert result["status"] == "success"
    assert result["type"] == "Other"
    assert result["description"] == "details"
    assert "analyze_time" in result


async def test_safe_vdb_operation_times_out_with_context():
    async def _hang():
        await asyncio.sleep(0.2)

    with pytest.raises(TimeoutError) as exc_info:
        await safe_vdb_operation_with_exception(
            operation=_hang,
            operation_name="relationship_upsert",
            entity_name="A->B",
            max_retries=1,
            retry_delay=0,
            timeout_seconds=0.05,
        )
    message = str(exc_info.value)
    assert "relationship_upsert" in message
    assert "A->B" in message
    assert "timeout" in message.lower()


def test_relationship_vdb_timeout_has_120s_floor():
    assert _get_relationship_vdb_timeout_seconds({}) == 120.0
    assert (
        _get_relationship_vdb_timeout_seconds({"default_embedding_timeout": 10})
        == 120.0
    )
    assert (
        _get_relationship_vdb_timeout_seconds({"default_embedding_timeout": 50})
        == 150.0
    )


# ---------------------------------------------------------------------------
# Parser source resolver & MinerU pipeline persistence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "create_exact, expected_name",
    [
        pytest.param(False, "demo.[mineru].pdf", id="finds-hint-variant"),
        pytest.param(True, "demo.pdf", id="prefers-exact-canonical"),
    ],
)
def test_parser_source_resolver_canonical_vs_hint_variant(
    tmp_path, monkeypatch, create_exact, expected_name
):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    if create_exact:
        (input_dir / "demo.pdf").write_bytes(b"canonical")
    (input_dir / "demo.[mineru].pdf").write_bytes(b"hinted")
    rag = _new_rag(tmp_path / "work")

    resolved = rag._resolve_source_file_for_parser(
        "demo.pdf", parser_engine=PARSER_ENGINE_MINERU
    )
    assert Path(resolved) == input_dir / expected_name


def _workspace_resolver_rag(tmp_path, monkeypatch, workspace: str):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    if workspace:
        (input_dir / workspace).mkdir()
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    monkeypatch.chdir(tmp_path)
    rag = _new_rag(tmp_path / "work")
    rag.workspace = workspace
    return rag, input_dir


@pytest.mark.parametrize(
    "location",
    [
        pytest.param(("demo.pdf",), id="input-root"),
        pytest.param((PARSED_DIR_NAME, "demo.pdf"), id="input-parsed"),
        pytest.param(("demo.[mineru].pdf",), id="input-root-hint-variant"),
        pytest.param(("@cwd", "inputs", "demo.pdf"), id="cwd-inputs"),
    ],
)
def test_parser_source_resolver_keeps_a_scoped_workspace_out_of_the_default_one(
    tmp_path, monkeypatch, location
):
    """A workspace with its own input directory never parses another's file."""
    rag, input_dir = _workspace_resolver_rag(tmp_path, monkeypatch, "team")
    foreign = (
        tmp_path.joinpath(*location[1:])
        if location[0] == "@cwd"
        else input_dir.joinpath(*location)
    )
    foreign.parent.mkdir(parents=True, exist_ok=True)
    foreign.write_bytes(b"default workspace's document")

    assert (
        rag._resolve_source_file_for_parser(
            "demo.pdf", parser_engine=PARSER_ENGINE_MINERU
        )
        == "demo.pdf"
    )


def test_parser_source_resolver_finds_a_scoped_workspace_file_in_its_own_dir(
    tmp_path, monkeypatch
):
    rag, input_dir = _workspace_resolver_rag(tmp_path, monkeypatch, "team")
    (input_dir / "demo.pdf").write_bytes(b"default workspace's document")
    own = input_dir / "team" / "demo.pdf"
    own.write_bytes(b"team's document")

    assert Path(rag._resolve_source_file_for_parser("demo.pdf")) == own


def test_parser_source_resolver_keeps_the_base_dir_for_an_unscoped_workspace(
    tmp_path, monkeypatch
):
    """SDK caller with a workspace but no ``INPUT_DIR/<workspace>/`` uses base dir."""
    rag, input_dir = _workspace_resolver_rag(tmp_path, monkeypatch, "")
    rag.workspace = "team"
    base = input_dir / "demo.pdf"
    base.write_bytes(b"the SDK caller's document")

    assert Path(rag._resolve_source_file_for_parser("demo.pdf")) == base


async def test_parse_mineru_to_lightrag_document(tmp_path, monkeypatch):
    """End-to-end: MinerU parse persists ``{{LRdoc}}`` summary & ``file://`` sidecar URI."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://fake-mineru")
    src_file = input_dir / "demo.pdf"
    src_file.write_bytes(b"fake-pdf")

    _install_mineru_fake_download(
        monkeypatch,
        expected_source=src_file,
        content_list=[
            {"type": "text", "text": "第一段正文"},
            {
                "type": "image",
                "img_path": "assets/img1.png",
                "image_caption": ["图1 架构图"],
                "image_footnote": ["示意图"],
            },
            {
                "type": "table",
                "table_body": "<table><tr><td>A</td></tr></table>",
                "table_caption": ["表1 指标"],
                "table_footnote": ["单位：%"],
            },
            {"type": "equation", "text": "$$E=mc^2$$"},
        ],
        assets={"assets/img1.png": b"\x89PNGfake"},
    )

    async with _open_rag(tmp_path / "work") as rag:
        parsed = await _parse_via_registry(
            rag, "mineru", doc_id="doc-1", file_path=str(src_file), content_data={"content": ""}
        )
        assert parsed["parse_format"] == "lightrag"
        blocks_path = Path(parsed["blocks_path"])
        assert blocks_path.exists()

        meta = json.loads(blocks_path.read_text(encoding="utf-8").splitlines()[0])
        assert meta["type"] == "meta" and meta["format"] == "lightrag"
        assert meta["drawing_file"] and meta["table_file"] and meta["equation_file"]

        full_doc = await rag.full_docs.get_by_id("doc-1")
        assert full_doc["parse_format"] == "lightrag"
        assert full_doc["content"].startswith("{{LRdoc}}")
        assert full_doc["sidecar_location"].startswith("file://")
        assert full_doc["sidecar_location"].endswith("/")
        assert resolve_sidecar_uri(full_doc["sidecar_location"]) == (
            blocks_path.parent.resolve()
        )


async def test_parse_mineru_uses_hint_source_and_canonical_upload_name(
    tmp_path, monkeypatch
):
    """Enqueue -> MinerU parse uses hinted file on disk, canonical upload name, and archives."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://fake-mineru")

    hinted_name = "LightRAG - Simple and Fast RAG.[mineru].pdf"
    canonical_name = "LightRAG - Simple and Fast RAG.pdf"
    src_file = input_dir / hinted_name
    src_file.write_bytes(b"fake-pdf")

    _install_mineru_fake_download(
        monkeypatch,
        expected_source=src_file,
        expected_upload_name=canonical_name,
        content_list=[{"type": "text", "text": "第一段正文"}],
    )

    async with _open_rag(tmp_path / "work") as rag:
        await rag.apipeline_enqueue_documents(
            "",
            file_paths=str(src_file),
            track_id="track-hint",
            docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
            parse_engine=PARSER_ENGINE_MINERU,
        )
        doc_id = compute_mdhash_id(canonical_name, prefix="doc-")
        status = await rag.doc_status.get_by_id(doc_id)
        assert status is not None
        assert status["file_path"] == canonical_name
        assert status["metadata"]["source_file"] == hinted_name

        content_data = await rag.full_docs.get_by_id(doc_id)
        assert content_data is not None
        content_data["source_file"] = status["metadata"]["source_file"]

        parsed = await _parse_via_registry(
            rag,
            "mineru",
            doc_id=doc_id,
            file_path=status["file_path"],
            content_data=content_data,
        )
        blocks_path = Path(parsed["blocks_path"])
        assert blocks_path.parent == input_dir / PARSED_DIR_NAME / f"{canonical_name}.parsed"
        assert (input_dir / PARSED_DIR_NAME / f"{canonical_name}.mineru_raw").is_dir()
        assert not src_file.exists()
        assert (input_dir / PARSED_DIR_NAME / hinted_name).is_file()


async def test_parse_mineru_empty_service_result_raises_without_fallback(
    tmp_path, monkeypatch
):
    """Missing ``content_list.json`` fails fast without falling back to raw content."""
    src_file = tmp_path / "demo.pdf"
    src_file.write_bytes(b"fake-pdf")
    _install_mineru_fake_download(monkeypatch, content_list=None)
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://fake")

    async with _open_rag(tmp_path) as rag:
        with pytest.raises(FileNotFoundError, match="content_list.json"):
            await _parse_via_registry(
                rag,
                "mineru",
                doc_id="doc-local-1",
                file_path=str(src_file),
                content_data={"content": "native fallback content"},
            )


async def test_engine_params_survive_persist_to_full_docs(tmp_path, monkeypatch):
    """Per-file engine params encoded in ``parse_engine`` survive parse persistence."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    monkeypatch.setenv("MINERU_API_MODE", "local")
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://fake-mineru")
    src_file = input_dir / "demo.pdf"
    src_file.write_bytes(b"fake-pdf")

    _install_mineru_fake_download(
        monkeypatch,
        content_list=[{"type": "text", "text": "正文"}],
        options_params={"page_range": "1-3"},
    )

    async with _open_rag(tmp_path / "work") as rag:
        await _parse_via_registry(
            rag,
            "mineru",
            doc_id="doc-ep",
            file_path=str(src_file),
            content_data={"content": "", "parse_engine": "mineru(page_range=1-3)"},
        )
        stored = await rag.full_docs.get_by_id("doc-ep")
        assert stored is not None
        assert stored["parse_engine"] == "mineru(page_range=1-3)"


# ---------------------------------------------------------------------------
# Multimodal chunk assembly & internal markup stripping
# ---------------------------------------------------------------------------


async def test_mm_chunks_and_modality_relations_from_sidecars(tmp_path):
    blocks, _ = _write_sidecar_doc(
        tmp_path,
        blocks_lines=[
            {"type": "meta", "format": "lightrag", "version": "1.0", "doc_id": "doc-1"},
            {"type": "content", "blockid": "b1", "format": "plain_text", "content": "正文"},
        ],
        drawings={
            "d1": {
                "id": "d1",
                "heading": "章节A",
                "caption": "图1 架构",
                "llm_cache_list": ["default:analysis:abc123"],
                "llm_analyze_result": {
                    "name": "系统架构图",
                    "type": "Chart",
                    "description": "模块交互关系",
                    "analyze_time": 1700000000,
                    "status": "success",
                    "message": "",
                },
            }
        },
    )
    async with _open_rag(tmp_path) as rag:
        mm_chunks = rag._build_mm_chunks_from_sidecars(
            doc_id="doc-1",
            file_path="demo.pdf",
            blocks_path=str(blocks),
            base_order_index=2,
        )
    assert len(mm_chunks) == 1
    chunk = mm_chunks[0]
    assert chunk["content"].startswith("[Image Name]")
    assert "[Image Type]Chart" in chunk["content"]
    assert chunk["sidecar"] == {
        "type": "drawing",
        "id": "d1",
        "refs": [{"type": "drawing", "id": "d1"}],
    }
    assert chunk["heading"] == {"level": 0, "heading": "章节A", "parent_headings": []}
    assert chunk["llm_cache_list"] == ["default:analysis:abc123"]


async def test_mm_chunks_sanitize_vlm_control_characters(tmp_path):
    """Regression: XML-illegal control characters in VLM JSON fields are stripped."""
    blocks, _ = _write_sidecar_doc(
        tmp_path,
        blocks_lines=[
            {"type": "meta", "format": "lightrag", "version": "1.0", "doc_id": "doc-1"}
        ],
        tables={
            "t1": {
                "id": "t1",
                "heading": "章节\x0bB",
                "footnotes": ["脚注\x00一"],
                "llm_analyze_result": {
                    "name": "成本对比表\x00",
                    "description": "GraphRAG消耗$\x0crac{610}{C}$次调用",
                    "analyze_time": 1700000000,
                    "status": "success",
                    "message": "",
                },
            }
        },
    )
    async with _open_rag(tmp_path) as rag:
        mm_chunks = rag._build_mm_chunks_from_sidecars(
            doc_id="doc-1",
            file_path="demo.pdf",
            blocks_path=str(blocks),
            base_order_index=0,
        )
    assert len(mm_chunks) == 1
    chunk = mm_chunks[0]
    illegal = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
    assert not illegal.search(chunk["content"])
    assert not illegal.search(chunk["heading"]["heading"])
    assert "[Table Name]成本对比表" in chunk["content"]
    assert "$rac{610}{C}$" in chunk["content"]
    assert "[Table Footnotes]脚注一" in chunk["content"]
    assert chunk["heading"]["heading"] == "章节B"


def test_parse_mm_display_name_matches_chunk_format():
    """Pin builder/consumer label format contract across all three modalities."""
    assert (
        _parse_mm_display_name(
            "[Image Name]系统架构图\n[Image Type]Chart\n\n模块交互关系\n\n[Image Footnotes]脚注1; 脚注2",
            "d1",
        )
        == "系统架构图"
    )
    assert _parse_mm_display_name("[Table Name]性能对比表\n\n各方法的指标对比", "t1") == "性能对比表"
    assert (
        _parse_mm_display_name("E = mc^2\n[Equation Name]质能方程\n\n爱因斯坦的质能等效公式", "e1")
        == "质能方程"
    )
    assert _parse_mm_display_name("no marker here", "fallback-id") == "fallback-id"
    assert _parse_mm_display_name("", "fallback-id") == "fallback-id"
    assert _parse_mm_display_name("[Image Name]   ", "fallback-id") == "fallback-id"


async def test_parse_mm_display_name_on_real_builder_output(tmp_path):
    """End-to-end pin: feed real ``_build_mm_chunks_from_sidecars`` chunks to ``_parse_mm_display_name``."""
    heading = {"level": 1, "heading": "章节A", "parent_headings": []}
    blocks, _ = _write_sidecar_doc(
        tmp_path,
        stem="demo.pdf",
        blocks_lines=[],
        drawings={
            "d1": {
                "heading": heading,
                "footnotes": [],
                "llm_cache_list": [],
                "llm_analyze_result": {
                    "name": "系统架构图",
                    "type": "Chart",
                    "description": "模块交互关系",
                    "analyze_time": 1700000000,
                    "status": "success",
                    "message": "",
                },
            }
        },
        tables={
            "t1": {
                "heading": heading,
                "footnotes": [],
                "llm_cache_list": [],
                "llm_analyze_result": {
                    "name": "性能对比表",
                    "description": "各方法的指标对比",
                    "analyze_time": 1700000001,
                    "status": "success",
                    "message": "",
                },
            }
        },
        equations={
            "e1": {
                "heading": heading,
                "footnotes": [],
                "llm_cache_list": [],
                "llm_analyze_result": {
                    "name": "质能方程",
                    "equation": "E = mc^2",
                    "description": "爱因斯坦的质能等效公式",
                    "analyze_time": 1700000002,
                    "status": "success",
                    "message": "",
                },
            }
        },
    )
    async with _open_rag(tmp_path) as rag:
        mm_chunks = rag._build_mm_chunks_from_sidecars(
            doc_id="doc-1",
            file_path="demo.pdf",
            blocks_path=str(blocks),
            base_order_index=0,
            process_options="ite",
        )
    by_type = {chunk["sidecar"]["type"]: chunk for chunk in mm_chunks}
    assert set(by_type.keys()) == {"drawing", "table", "equation"}
    for kind, (sidecar_id, name) in {
        "drawing": ("d1", "系统架构图"),
        "table": ("t1", "性能对比表"),
        "equation": ("e1", "质能方程"),
    }.items():
        assert _parse_mm_display_name(by_type[kind]["content"], sidecar_id) == name


def test_build_chunks_dict_preserves_existing_llm_cache_list():
    """Regression: pre-existing multimodal ``llm_cache_list`` is preserved and deduplicated."""
    chunks = build_chunks_dict_from_chunking_result(
        [
            {"chunk_order_index": 0, "content": "first chunk", "tokens": 4},
            {
                "chunk_id": "doc-1-mm-drawing-000",
                "chunk_order_index": 1,
                "content": "second chunk",
                "tokens": 6,
                "llm_cache_list": [
                    "default:analysis:abc",
                    "default:analysis:abc",
                    "default:analysis:def",
                ],
            },
        ],
        doc_id="doc-1",
        file_path="demo.pdf",
    )
    mm_chunk = next(
        v for v in chunks.values() if v.get("chunk_id") == "doc-1-mm-drawing-000"
    )
    text_chunk = next(
        v for v in chunks.values() if v.get("chunk_id") != "doc-1-mm-drawing-000"
    )
    assert mm_chunk["llm_cache_list"] == [
        "default:analysis:abc",
        "default:analysis:def",
    ]
    assert text_chunk["llm_cache_list"] == []


async def test_build_mm_chunks_respects_process_options_filter(tmp_path):
    """Regression: sidecar reads are gated by active ``process_options`` (empty skips stale)."""
    blocks, _ = _write_sidecar_doc(
        tmp_path,
        drawings={
            "d1": {
                "id": "d1",
                "llm_analyze_result": {
                    "name": "old",
                    "type": "Chart",
                    "description": "stale drawing",
                    "analyze_time": 1700000000,
                    "status": "success",
                    "message": "",
                },
            }
        },
        tables={
            "t1": {
                "id": "t1",
                "llm_analyze_result": {
                    "name": "old",
                    "description": "stale table",
                    "analyze_time": 1700000000,
                    "status": "success",
                    "message": "",
                },
            }
        },
    )
    async with _open_rag(tmp_path) as rag:
        only_tables = rag._build_mm_chunks_from_sidecars(
            doc_id="doc-1",
            file_path="demo.pdf",
            blocks_path=str(blocks),
            base_order_index=0,
            process_options="t",
        )
        assert len(only_tables) == 1
        assert only_tables[0]["sidecar"]["type"] == "table"

        assert (
            rag._build_mm_chunks_from_sidecars(
                doc_id="doc-1",
                file_path="demo.pdf",
                blocks_path=str(blocks),
                base_order_index=0,
                process_options="",
            )
            == []
        )

        legacy = rag._build_mm_chunks_from_sidecars(
            doc_id="doc-1",
            file_path="demo.pdf",
            blocks_path=str(blocks),
            base_order_index=0,
        )
        assert {ch["sidecar"]["type"] for ch in legacy} == {"drawing", "table"}


def test_strip_internal_multimodal_markup_cleans_table_id():
    source = (
        '<table id="tb-1-0001" format="json" caption="Indicator metrics">'
        '[["a","b"],["1","2"]]'
        "</table>"
    )
    cleaned = strip_internal_multimodal_markup_for_extraction(source)
    assert "tb-1-0001" not in cleaned
    assert 'format="json"' in cleaned
    assert 'caption="Indicator metrics"' in cleaned
    assert '[["a","b"],["1","2"]]' in cleaned


def test_strip_internal_multimodal_markup_cite_default_unwraps():
    """Default (``keep_cite_tag=False``) unwraps ``<cite>`` for entity extraction."""
    source = (
        'see <cite type="table" refid="tb-1-0001">表 1</cite> and '
        '<cite type="equation" refid="eq-1-0002">公式 2</cite> for details.'
    )
    cleaned = strip_internal_multimodal_markup_for_extraction(source)
    assert "<cite" not in cleaned and "refid=" not in cleaned
    assert "tb-1-0001" not in cleaned and "eq-1-0002" not in cleaned
    assert "表 1" in cleaned and "公式 2" in cleaned


def test_strip_internal_multimodal_markup_cite_keep_tag_strips_refid_only():
    """``keep_cite_tag=True`` (surrounding-context path) keeps ``<cite type=...>`` without ``refid``."""
    source = (
        'see <cite type="table" refid="tb-1-0001">表 1</cite>; '
        '<drawing id="im-1" path="a.png" src="a" caption="Fig" />'
    )
    cleaned = strip_internal_multimodal_markup_for_extraction(
        source, keep_cite_tag=True
    )
    assert '<cite type="table">表 1</cite>' in cleaned
    assert "refid=" not in cleaned and "tb-1-0001" not in cleaned
    assert '<drawing caption="Fig" />' in cleaned
    assert 'id="im-1"' not in cleaned and "path=" not in cleaned
