import asyncio
import importlib
import io
import sys
import threading
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from lightrag.chunker import chunking_by_token_size

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
_lightrag = importlib.import_module("lightrag.lightrag")
_base = importlib.import_module("lightrag.base")
_constants = importlib.import_module("lightrag.constants")
_utils = importlib.import_module("lightrag.utils")
_parser_routing = importlib.import_module("lightrag.parser.routing")
_parser_registry = importlib.import_module("lightrag.parser.registry")
_parser_base = importlib.import_module("lightrag.parser.base")
sys.argv = _original_argv

DocStatus = _base.DocStatus
DocProcessingStatus = _base.DocProcessingStatus
DocSchedulingRecord = _base.DocSchedulingRecord
SourceAbsent = _base.SourceAbsent
SourceConflict = _base.SourceConflict
SourceUnique = _base.SourceUnique
FULL_DOCS_FORMAT_LIGHTRAG = _constants.FULL_DOCS_FORMAT_LIGHTRAG
FULL_DOCS_FORMAT_PENDING_PARSE = _constants.FULL_DOCS_FORMAT_PENDING_PARSE
PARSED_DIR_NAME = _constants.PARSED_DIR_NAME
PROCESS_OPTION_CHUNK_FIXED = _constants.PROCESS_OPTION_CHUNK_FIXED
compute_mdhash_id = _utils.compute_mdhash_id
LightRAG = _lightrag.LightRAG
resolve_stored_document_parser_engine = (
    _parser_routing.resolve_stored_document_parser_engine
)
get_parser = _parser_registry.get_parser
ParseContext = _parser_base.ParseContext

pipeline_index_file = _document_routes.pipeline_index_file
pipeline_enqueue_scan_batch = _document_routes.pipeline_enqueue_scan_batch
_ScanCandidate = _document_routes._ScanCandidate
pipeline_index_texts = _document_routes.pipeline_index_texts
pipeline_enqueue_file = _document_routes.pipeline_enqueue_file
run_scanning_process = _document_routes.run_scanning_process
DocumentManager = _document_routes.DocumentManager
create_document_routes = _document_routes.create_document_routes

pytestmark = pytest.mark.offline


def _get_endpoint(router, name: str):
    return [
        route.endpoint
        for route in router.routes
        if getattr(route, "name", "") == name
    ][-1]


async def _parse_via_registry(rag, engine, doc_id, file_path, content_data):
    """Drive a parser the way the pipeline worker does (registry dispatch)."""
    result = await get_parser(engine).parse(
        ParseContext(rag, doc_id, file_path, content_data)
    )
    return result.to_dict()


async def _await_managed(managed_tasks):
    """Await every managed background task to completion so each child's finally
    runs (releasing its reservation). Snapshot first — the task's done-callback
    discards it from the set."""
    for task in list(managed_tasks):
        try:
            await task
        except asyncio.CancelledError:
            pass


@pytest.fixture(autouse=True)
def _ensure_shared_storage_initialized():
    """Initialize shared_storage dicts before each test and reset flags after."""
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    shared_storage.initialize_share_data()
    yield
    if shared_storage._shared_dicts is not None:
        for key in list(shared_storage._shared_dicts.keys()):
            if key.endswith("pipeline_status") or key == "pipeline_status":
                ns = shared_storage._shared_dicts[key]
                if isinstance(ns, dict):
                    ns["busy"] = False
                    ns["scanning"] = False


class _FakeDocStatus:
    def __init__(self):
        self.docs = {}

    async def get_by_id(self, doc_id):
        return self.docs.get(doc_id)


class _FakeRag:
    def __init__(self, final_status=DocStatus.PROCESSED):
        self.doc_status = _FakeDocStatus()
        self.final_status = final_status
        self.enqueued = []
        self.errors = []
        # _resolve_text_chunking reads addon_params; {} -> default chunker config.
        self.addon_params = {}
        self.chunking_func = chunking_by_token_size

    async def apipeline_enqueue_documents(
        self,
        input,
        ids=None,
        file_paths=None,
        track_id=None,
        docs_format=None,
        parse_engine=None,
        process_options=None,
        chunk_options=None,
        from_scan=False,
    ):
        item = {
            "input": input,
            "file_path": file_paths,
            "track_id": track_id,
            "docs_format": docs_format,
            "parse_engine": parse_engine,
            "process_options": process_options,
            "chunk_options": chunk_options,
            "from_scan": from_scan,
        }
        self.enqueued.append(item)
        return track_id

    async def apipeline_process_enqueue_documents(self):
        for item in self.enqueued:
            file_path = item["file_path"]
            doc_id = compute_mdhash_id(file_path, prefix="doc-")
            self.doc_status.docs[doc_id] = {
                "status": self.final_status,
                "file_path": file_path,
                "track_id": item["track_id"],
            }

    async def apipeline_enqueue_error_documents(self, error_files, track_id=None):
        self.errors.append((error_files, track_id))


class _DuplicateEnqueueRag(_FakeRag):
    async def apipeline_enqueue_documents(self, *args, **kwargs):
        self.enqueued.append({"args": args, "kwargs": kwargs})
        return None


class _ScanDocStatus:
    """doc_status double implementing the typed source-resolution contract.

    Rows are keyed by the string that doubles as their doc id; a canonical source
    key matches a row when it equals that key's basename. Unless a fixture pins
    ``metadata`` explicitly, a row is treated as scan/upload-origin — those always
    record the physical basename that created them in ``metadata.source_file``,
    which is what tells a resumed file apart from an alias (LR2 §8.3.E/F/G).
    """

    def __init__(self, docs_by_path):
        self.docs_by_path = docs_by_path
        self.deleted_ids: list[str] = []

    async def get_doc_by_file_path(self, file_path):
        return self.docs_by_path.get(file_path)

    async def get_doc_by_file_basename(self, basename):
        for stored_path, doc in self.docs_by_path.items():
            if Path(stored_path).name == basename:
                return stored_path, doc
        return None

    def _metadata(self, doc_id, row):
        metadata = row.get("metadata")
        if metadata is None:
            return {"source_file": Path(doc_id).name}
        return metadata

    def _status(self, row):
        status = row.get("status")
        return status if isinstance(status, DocStatus) else DocStatus(status)

    async def resolve_doc_source_strict(self, canonical_source_key):
        candidates = [
            (doc_id, row)
            for doc_id, row in self.docs_by_path.items()
            if Path(doc_id).name == canonical_source_key
            and not (self._metadata(doc_id, row) or {}).get("is_duplicate")
        ]
        if not candidates:
            return SourceAbsent()
        if len(candidates) > 1:
            return SourceConflict(
                candidate_count=len(candidates),
                sample_doc_ids=tuple(doc_id for doc_id, _ in candidates[:2]),
            )
        doc_id, row = candidates[0]
        return SourceUnique(
            doc_id=doc_id,
            doc=DocSchedulingRecord(
                id=doc_id,
                status=self._status(row),
                created_at=row.get("created_at", ""),
                updated_at=row.get("updated_at", ""),
                file_path=row.get("file_path") or Path(doc_id).name,
                track_id=row.get("track_id"),
                has_custom_chunk_journal=False,
            ),
        )

    async def get_full_docs_by_ids(self, doc_ids, *, strict=False):
        hydrated = {}
        for doc_id in doc_ids:
            row = self.docs_by_path.get(doc_id)
            if row is None:
                continue
            hydrated[doc_id] = DocProcessingStatus(
                content_summary=row.get("content_summary", ""),
                content_length=row.get("content_length", 0),
                file_path=row.get("file_path") or Path(doc_id).name,
                status=self._status(row),
                created_at=row.get("created_at", ""),
                updated_at=row.get("updated_at", ""),
                track_id=row.get("track_id"),
                metadata=self._metadata(doc_id, row),
            )
        return hydrated

    async def delete(self, ids):
        for doc_id in ids:
            self.docs_by_path.pop(doc_id, None)
            self.deleted_ids.append(doc_id)


class _ScanFullDocs:
    """Minimal full_docs double for run_scanning_process.

    Declares ``supports_strict_point_reads = True`` per LR2 §8.3.D so the scan
    can distinguish a resumable FAILED row from an extraction-error stub.
    """

    supports_strict_point_reads = True

    def __init__(self, docs_by_id):
        self.docs_by_id = docs_by_id

    async def get_by_id(self, doc_id):
        return self.docs_by_id.get(doc_id)

    async def get_by_id_strict(self, doc_id):
        return self.docs_by_id.get(doc_id)


class _ScanRag:
    def __init__(self, docs_by_path, full_docs_by_id=None):
        self.doc_status = _ScanDocStatus(docs_by_path)
        if full_docs_by_id is None:
            full_docs_by_id = {path: {"content": ""} for path in docs_by_path}
        self.full_docs = _ScanFullDocs(full_docs_by_id)
        self.process_calls = 0
        self.workspace = f"scan-test-{uuid4().hex}"
        self.reset_calls = []
        self.enqueued = []
        self.errors = []

    async def apipeline_enqueue_documents(
        self,
        input,
        ids=None,
        file_paths=None,
        track_id=None,
        docs_format=None,
        parse_engine=None,
        process_options=None,
        from_scan=False,
    ):
        item = {
            "input": input,
            "file_path": file_paths,
            "track_id": track_id,
            "docs_format": docs_format,
            "parse_engine": parse_engine,
            "process_options": process_options,
            "from_scan": from_scan,
        }
        self.enqueued.append(item)
        return track_id

    async def apipeline_enqueue_error_documents(self, error_files, track_id=None):
        self.errors.append((error_files, track_id))

    async def apipeline_process_enqueue_documents(self):
        self.process_calls += 1

    async def apipeline_reset_failed_for_scan(
        self, request_id, *, scan_owner_token=None
    ):
        self.reset_calls.append((request_id, scan_owner_token))
        return True


class _DuplicateUploadRag:
    def __init__(self, docs_by_path, chunking_func=chunking_by_token_size):
        self.doc_status = _ScanDocStatus(docs_by_path)
        self.workspace = f"upload-test-{uuid4().hex}"
        self.chunking_func = chunking_func
        self.addon_params = {}


class _ParseFullDocs:
    def __init__(self, source_path):
        self.source_path = source_path
        self.events = []
        self.data = {}

    async def get_by_id(self, doc_id):
        record = self.data.get(doc_id)
        return dict(record) if record is not None else None

    async def upsert(self, data):
        self.events.append("upsert")
        self.data.update(data)

    async def index_done_callback(self):
        self.events.append("index_done")
        assert self.source_path.exists()


class _ParseDocStatus:
    """Minimal doc_status double for the parse_* archive tests."""

    async def get_by_id(self, doc_id):
        return None

    async def upsert(self, data):
        return None

    async def update_doc_status_fields(self, doc_id, fields, *, missing_ok=False):
        return None


class _ParseTokenizer(_utils.TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


class _ParseRag:
    _persist_parsed_full_docs = LightRAG._persist_parsed_full_docs

    def __init__(self, working_dir, source_path):
        self.working_dir = str(working_dir)
        self.full_docs = _ParseFullDocs(source_path)
        self.doc_status = _ParseDocStatus()
        self.tokenizer = _utils.Tokenizer(
            model_name="char", tokenizer=_ParseTokenizer()
        )

    def _resolve_source_file_for_parser(self, file_path):
        return file_path


async def test_pipeline_index_file_leaves_lightrag_document_docx_for_parser_archive(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("LIGHTRAG_PARSER", "docx:native")
    file_path = tmp_path / "sample.docx"
    file_path.write_bytes(b"docx bytes")
    rag = _FakeRag()

    await pipeline_index_file(rag, file_path, "track-docx")

    assert file_path.exists()
    assert not (tmp_path / PARSED_DIR_NAME / file_path.name).exists()
    assert rag.enqueued[0]["file_path"] == str(file_path)
    assert rag.enqueued[0]["docs_format"] == FULL_DOCS_FORMAT_PENDING_PARSE
    assert rag.enqueued[0]["parse_engine"] == "native"


async def test_pipeline_enqueue_md_defers_to_legacy_parser(tmp_path, monkeypatch):
    # Unhinted .md defaults to the legacy engine and now defers extraction to
    # the worker stage (PENDING_PARSE), like every other engine.
    monkeypatch.delenv("LIGHTRAG_PARSER", raising=False)
    file_path = tmp_path / "notes.md"
    file_path.write_text("# Notes\n\nmarkdown content", encoding="utf-8")
    rag = _FakeRag()

    success, returned_track_id = await pipeline_enqueue_file(rag, file_path, "track-md")

    assert success is True
    assert returned_track_id == "track-md"
    assert rag.enqueued == [
        {
            "input": "",
            "file_path": str(file_path),
            "track_id": "track-md",
            "docs_format": FULL_DOCS_FORMAT_PENDING_PARSE,
            "parse_engine": "legacy",
            "process_options": PROCESS_OPTION_CHUNK_FIXED,
            "chunk_options": None,
            "from_scan": False,
        }
    ]
    # Deferred: the source stays in place until the worker archives it.
    assert file_path.exists()


async def test_pipeline_enqueue_legacy_duplicate_archives_with_unique_name(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("LIGHTRAG_PARSER", raising=False)
    file_path = tmp_path / "duplicate.md"
    file_path.write_text("duplicate content", encoding="utf-8")
    parsed_dir = tmp_path / PARSED_DIR_NAME
    parsed_dir.mkdir()
    (parsed_dir / file_path.name).write_text("existing", encoding="utf-8")
    rag = _DuplicateEnqueueRag()

    success, returned_track_id = await pipeline_enqueue_file(
        rag, file_path, "track-dup"
    )

    assert success is False
    assert returned_track_id == "track-dup"
    assert not file_path.exists()
    assert (parsed_dir / file_path.name).read_text(encoding="utf-8") == "existing"
    assert (parsed_dir / "duplicate_001.md").read_text(
        encoding="utf-8"
    ) == "duplicate content"


async def test_pipeline_enqueue_passes_process_options_from_filename_hint(
    tmp_path, monkeypatch
):
    """Filename hint ``[native-iet]`` flows into apipeline_enqueue_documents."""
    monkeypatch.setenv("LIGHTRAG_PARSER", "docx:native")
    file_path = tmp_path / "report.[native-iet].docx"
    file_path.write_bytes(b"docx-bytes")
    rag = _FakeRag()

    success, returned_track_id = await pipeline_enqueue_file(
        rag, file_path, "track-options"
    )

    assert success is True
    assert returned_track_id == "track-options"
    assert rag.enqueued == [
        {
            "input": "",
            "file_path": str(file_path),
            "track_id": "track-options",
            "docs_format": FULL_DOCS_FORMAT_PENDING_PARSE,
            "parse_engine": "native",
            "process_options": "iet",
            "chunk_options": None,
            "from_scan": False,
        }
    ]
    # Native engine deferral keeps the source file in place for the parser.
    assert file_path.exists()


async def test_scan_enqueue_accepts_c_hint_for_observable_runtime_fallback(
    tmp_path, monkeypatch
):
    """Scan has no caller to receive a 422, so ``C`` remains persisted and the
    runtime decides between the injected callback and warned F fallback."""
    monkeypatch.setenv("LIGHTRAG_PARSER", "docx:native")
    file_path = tmp_path / "report.[native-Cite].docx"
    file_path.write_bytes(b"docx-bytes")
    rag = _FakeRag()

    enqueued = await pipeline_enqueue_scan_batch(
        rag,
        [_ScanCandidate(file_path, len(b"docx-bytes"))],
        "track-custom-scan",
    )

    assert enqueued == 1
    assert rag.enqueued[0]["process_options"] == "Cite"


async def test_pipeline_enqueue_rejects_invalid_filename_hint(tmp_path, monkeypatch):
    """Bad filename processing hints must become file-processing errors."""
    monkeypatch.setenv("LIGHTRAG_PARSER", "docx:native")
    file_path = tmp_path / "report.[abc].docx"
    file_path.write_bytes(b"docx-bytes")
    rag = _FakeRag()

    success, returned_track_id = await pipeline_enqueue_file(
        rag, file_path, "track-bad-hint"
    )

    assert success is False
    assert returned_track_id == "track-bad-hint"
    assert rag.enqueued == []
    assert len(rag.errors) == 1
    error_files, track_id = rag.errors[0]
    assert track_id == "track-bad-hint"
    assert error_files[0]["file_path"] == file_path.name
    assert error_files[0]["error_description"] == (
        "[File Extraction]Filename hint error"
    )
    assert "unsupported parser engine 'abc'" in error_files[0]["original_error"]
    assert file_path.exists()


async def test_pipeline_enqueue_scan_batch_leaves_lightrag_document_docx_batch(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("LIGHTRAG_PARSER", "docx:native")
    first = tmp_path / "first.docx"
    second = tmp_path / "second.[mineru].docx"
    first.write_bytes(b"first docx bytes")
    second.write_bytes(b"second docx bytes")
    rag = _FakeRag()

    await pipeline_enqueue_scan_batch(
        rag,
        [_ScanCandidate(second, 0), _ScanCandidate(first, 0)],
        "track-scan",
    )

    # ``second.[mineru].docx`` needs MINERU_LOCAL_ENDPOINT, so only first.docx
    # actually enqueues here. Assert that explicitly: an empty ``rag.enqueued``
    # would satisfy every ``all(...)`` below and the test would prove nothing.
    assert [item["file_path"] for item in rag.enqueued] == [str(first)]

    assert first.exists()
    assert second.exists()
    assert not (tmp_path / PARSED_DIR_NAME / first.name).exists()
    assert not (tmp_path / PARSED_DIR_NAME / second.name).exists()
    assert all(
        item["docs_format"] == FULL_DOCS_FORMAT_PENDING_PARSE for item in rag.enqueued
    )
    assert all(item["parse_engine"] == "native" for item in rag.enqueued)
    assert all(
        item["process_options"] == PROCESS_OPTION_CHUNK_FIXED for item in rag.enqueued
    )


async def test_pipeline_enqueue_scan_batch_preserves_the_spool_order(
    tmp_path, monkeypatch
):
    """Fix-proof: the batch is enqueued in the order the spool handed it over.

    Global oldest-file-first ordering is recorded into doc_status one
    ``created_at=now()`` row at a time, so this helper is the last place it can
    be silently destroyed. It used to re-sort each batch with
    ``sorted(..., key=get_pinyin_sort_key)``; with these two names that sort
    reverses the batch, so re-introducing it fails here.
    """
    monkeypatch.setenv("LIGHTRAG_PARSER", "docx:native")
    older = tmp_path / "zebra.docx"
    newer = tmp_path / "alpha.docx"
    older.write_bytes(b"older docx bytes")
    newer.write_bytes(b"newer docx bytes")
    rag = _FakeRag()

    # Spool order (by mtime) deliberately opposes alphabetical/pinyin order.
    enqueued = await pipeline_enqueue_scan_batch(
        rag,
        [_ScanCandidate(older, 0), _ScanCandidate(newer, 0)],
        "track-scan",
    )

    assert enqueued == 2
    assert [item["file_path"] for item in rag.enqueued] == [str(older), str(newer)]


async def test_pipeline_enqueue_scan_batch_reuses_the_spooled_file_size(
    tmp_path, monkeypatch
):
    """The size discovery stat'ed travels with the candidate — no second stat.

    INPUT_DIR is frequently a network mount, so the enqueue phase must not pay
    another metadata round-trip per file just to fill in an error-report field.
    """
    monkeypatch.setenv("LIGHTRAG_PARSER", "docx:native")
    doc = tmp_path / "sized.docx"
    doc.write_bytes(b"docx bytes")
    rag = _FakeRag()

    stats: list[Path] = []
    real_stat = Path.stat

    def _counting_stat(self, *args, **kwargs):
        stats.append(self)
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", _counting_stat)

    await pipeline_enqueue_scan_batch(rag, [_ScanCandidate(doc, 4096)], "track-scan")

    assert len(rag.enqueued) == 1
    assert doc not in stats


async def test_pipeline_enqueue_file_stats_when_no_size_is_supplied(
    tmp_path, monkeypatch
):
    """The non-scan ingress paths still get their size the ordinary way.

    Counterpart to the test above: the stat is SKIPPED only because the caller
    supplied the size, not because it was dropped for everyone.
    """
    doc = tmp_path / "unsized.docx"
    doc.write_bytes(b"docx bytes")
    rag = _FakeRag()

    stats: list[Path] = []
    real_stat = Path.stat

    def _counting_stat(self, *args, **kwargs):
        stats.append(self)
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", _counting_stat)

    success, _track = await pipeline_enqueue_file(rag, doc, "track-upload")

    assert success is True
    assert doc in stats


async def test_pipeline_index_texts_sets_api_default_process_options():
    rag = _FakeRag()

    await pipeline_index_texts(
        rag,
        ["first text", "second text"],
        file_sources=["first.txt", "second.txt"],
        track_id="track-texts",
    )

    assert len(rag.enqueued) == 1
    item = rag.enqueued[0]
    assert item["input"] == ["first text", "second text"]
    assert item["file_path"] == ["first.txt", "second.txt"]
    assert item["track_id"] == "track-texts"
    assert item["docs_format"] is None
    assert item["parse_engine"] is None
    assert item["process_options"] == PROCESS_OPTION_CHUNK_FIXED
    assert item["from_scan"] is False
    # No chunking config -> default F snapshot is still passed through.
    assert isinstance(item["chunk_options"], dict)
    assert "fixed_token" in item["chunk_options"]


async def test_scan_processed_same_name_archives_with_unique_name(
    tmp_path, monkeypatch
):
    file_path = tmp_path / "already-parsed.docx"
    file_path.write_bytes(b"docx bytes")
    parsed_dir = tmp_path / PARSED_DIR_NAME
    parsed_dir.mkdir()
    (parsed_dir / file_path.name).write_bytes(b"previous parsed file")
    doc_manager = DocumentManager(str(tmp_path))
    rag = _ScanRag(
        {
            str(file_path): {
                "status": DocStatus.PROCESSED.value,
                "file_path": str(file_path),
                "track_id": "track-existing",
            }
        }
    )

    async def fail_if_reenqueue(*args, **kwargs):
        raise AssertionError("existing docx should not be re-enqueued")

    monkeypatch.setattr(
        _document_routes, "pipeline_enqueue_scan_batch", fail_if_reenqueue
    )

    await run_scanning_process(rag, doc_manager, "track-scan")

    assert not file_path.exists()
    assert (parsed_dir / file_path.name).read_bytes() == b"previous parsed file"
    assert (parsed_dir / "already-parsed_001.docx").read_bytes() == b"docx bytes"


async def test_scan_processed_canonical_name_archives_hinted_file(
    tmp_path, monkeypatch
):
    file_path = tmp_path / "already-parsed.[native].docx"
    file_path.write_bytes(b"docx bytes")
    doc_manager = DocumentManager(str(tmp_path))
    rag = _ScanRag(
        {
            "already-parsed.docx": {
                "status": DocStatus.PROCESSED.value,
                "file_path": "already-parsed.docx",
                "track_id": "track-existing",
            }
        }
    )

    async def fail_if_reenqueue(*args, **kwargs):
        raise AssertionError("canonical duplicate should not be re-enqueued")

    monkeypatch.setattr(
        _document_routes, "pipeline_enqueue_scan_batch", fail_if_reenqueue
    )

    await run_scanning_process(rag, doc_manager, "track-scan")

    assert not file_path.exists()
    assert (tmp_path / PARSED_DIR_NAME / file_path.name).read_bytes() == b"docx bytes"


async def test_scan_archives_scan_wide_canonical_duplicates(tmp_path, monkeypatch):
    plain_file = tmp_path / "same.docx"
    hinted_file = tmp_path / "same.[native].docx"
    plain_file.write_bytes(b"plain docx bytes")
    hinted_file.write_bytes(b"hinted docx bytes")
    doc_manager = DocumentManager(str(tmp_path))
    rag = _ScanRag({})
    calls = []

    async def capture_pipeline(rag_arg, candidates, track_id, **kwargs):
        calls.append(
            {
                "rag": rag_arg,
                "file_paths": [candidate.path for candidate in candidates],
                "track_id": track_id,
                "kwargs": kwargs,
            }
        )
        return len(candidates)

    monkeypatch.setattr(
        _document_routes, "pipeline_enqueue_scan_batch", capture_pipeline
    )

    await run_scanning_process(rag, doc_manager, "track-scan")

    # LR2 §8.4/§8.5: the winner is whichever physical file claimed the canonical
    # key FIRST in the scan-wide disk spool. Discovery is a single unordered
    # pass, so the previous "hinted variant always wins" preference is gone by
    # design. Exactly one variant is enqueued and the other is ARCHIVED, never
    # deleted; which one wins is not promised.
    assert len(calls) == 1
    assert len(calls[0]["file_paths"]) == 1
    winner = calls[0]["file_paths"][0]
    loser = plain_file if winner == hinted_file else hinted_file
    assert calls[0]["kwargs"] == {}
    archived_names = {
        path.name for path in (tmp_path / PARSED_DIR_NAME).iterdir() if path.is_file()
    }
    assert archived_names == {loser.name}
    assert winner.exists()
    assert not loser.exists()


async def test_scan_rejects_invalid_filename_hint(tmp_path, monkeypatch):
    monkeypatch.setenv("LIGHTRAG_PARSER", "docx:native")
    file_path = tmp_path / "bad-scan.[native-FR].docx"
    file_path.write_bytes(b"docx bytes")
    doc_manager = DocumentManager(str(tmp_path))
    rag = _ScanRag({})

    await run_scanning_process(rag, doc_manager, "track-scan")

    assert rag.enqueued == []
    assert len(rag.errors) == 1
    error_files, track_id = rag.errors[0]
    assert track_id == "track-scan"
    assert error_files[0]["file_path"] == file_path.name
    assert error_files[0]["error_description"] == (
        "[File Extraction]Filename hint error"
    )
    assert "multiple chunking modes" in error_files[0]["original_error"]
    # LR2 §8.1: every scan ends with exactly ONE processing drive, even when no
    # file was enqueued — resume targets and the FAILED rows the pre-discovery
    # reset turned into PENDING have no other trigger.
    assert rag.process_calls == 1
    assert file_path.exists()


async def test_scan_existing_non_processed_reprocesses_file(tmp_path, monkeypatch):
    file_path = tmp_path / "retry.docx"
    file_path.write_bytes(b"docx bytes")
    doc_manager = DocumentManager(str(tmp_path))
    rag = _ScanRag(
        {
            str(file_path): {
                "status": DocStatus.PARSING.value,
                "file_path": str(file_path),
                "track_id": "track-existing",
            }
        }
    )
    calls = []

    async def capture_pipeline(rag_arg, candidates, track_id, **kwargs):
        calls.append(
            {
                "rag": rag_arg,
                "file_paths": [candidate.path for candidate in candidates],
                "track_id": track_id,
                "kwargs": kwargs,
            }
        )
        return len(candidates)

    monkeypatch.setattr(
        _document_routes, "pipeline_enqueue_scan_batch", capture_pipeline
    )

    await run_scanning_process(rag, doc_manager, "track-scan")

    # Resume targets bypass pipeline_index_files entirely and trigger
    # apipeline_process_enqueue_documents directly while leaving the source in place.
    assert calls == []
    assert rag.process_calls == 1
    assert file_path.exists()


async def test_scan_mixed_new_and_resume_routes_only_new_through_enqueue(
    tmp_path, monkeypatch
):
    """When a scan finds both a new file and one matching an unfinished
    doc_status row, only the new file goes through pipeline_index_files;
    the resume target stays in place and processing is still driven once."""
    new_file = tmp_path / "fresh.docx"
    resume_file = tmp_path / "retry.docx"
    new_file.write_bytes(b"fresh docx bytes")
    resume_file.write_bytes(b"retry docx bytes")
    doc_manager = DocumentManager(str(tmp_path))
    rag = _ScanRag(
        {
            str(resume_file): {
                "status": DocStatus.FAILED.value,
                "file_path": str(resume_file),
                "track_id": "track-existing",
            }
        }
    )
    calls = []

    async def capture_pipeline(rag_arg, candidates, track_id, **kwargs):
        calls.append(
            {
                "rag": rag_arg,
                "file_paths": [candidate.path for candidate in candidates],
                "track_id": track_id,
                "kwargs": kwargs,
            }
        )
        return len(candidates)

    monkeypatch.setattr(
        _document_routes, "pipeline_enqueue_scan_batch", capture_pipeline
    )

    await run_scanning_process(rag, doc_manager, "track-scan")

    assert len(calls) == 1
    assert calls[0]["file_paths"] == [new_file]
    assert calls[0]["kwargs"] == {}
    assert rag.process_calls == 1
    assert resume_file.exists()
    assert new_file.exists()


async def test_scan_failed_extraction_record_without_full_docs_is_retried(
    tmp_path, monkeypatch
):
    """Stub doc_status rows recorded by apipeline_enqueue_error_documents have
    no full_docs entry; re-scanning drops the stale stub and routes as new."""
    file_path = tmp_path / "fixed.docx"
    file_path.write_bytes(b"now-readable bytes")
    doc_manager = DocumentManager(str(tmp_path))
    rag = _ScanRag(
        {
            str(file_path): {
                "status": DocStatus.FAILED.value,
                "file_path": str(file_path),
                "track_id": "track-old",
                "metadata": {"error_type": "file_extraction_error"},
            }
        },
        full_docs_by_id={},
    )
    calls = []

    async def capture_pipeline(rag_arg, candidates, track_id, **kwargs):
        calls.append(
            {
                "rag": rag_arg,
                "file_paths": [candidate.path for candidate in candidates],
                "track_id": track_id,
                "kwargs": kwargs,
            }
        )
        return len(candidates)

    monkeypatch.setattr(
        _document_routes, "pipeline_enqueue_scan_batch", capture_pipeline
    )

    await run_scanning_process(rag, doc_manager, "track-scan")

    assert rag.doc_status.deleted_ids == [str(file_path)]
    assert len(calls) == 1
    assert calls[0]["file_paths"] == [file_path]
    assert calls[0]["kwargs"] == {}
    assert rag.process_calls == 1
    assert file_path.exists()


async def test_scan_failed_with_full_docs_resumes_normally(tmp_path, monkeypatch):
    """A FAILED row that DOES have a full_docs entry resumes without stub deletion."""
    file_path = tmp_path / "downstream-failed.docx"
    file_path.write_bytes(b"docx bytes")
    doc_manager = DocumentManager(str(tmp_path))
    rag = _ScanRag(
        {
            str(file_path): {
                "status": DocStatus.FAILED.value,
                "file_path": str(file_path),
                "track_id": "track-old",
            }
        }
    )
    calls = []

    async def capture_pipeline(rag_arg, candidates, track_id, **kwargs):
        calls.append([candidate.path for candidate in candidates])

    monkeypatch.setattr(
        _document_routes, "pipeline_enqueue_scan_batch", capture_pipeline
    )

    await run_scanning_process(rag, doc_manager, "track-scan")

    assert rag.doc_status.deleted_ids == []
    assert calls == []
    assert rag.process_calls == 1
    assert file_path.exists()


async def test_upload_rejects_same_name_failed_doc_status_without_full_docs(
    tmp_path, monkeypatch
):
    # Other tests (e.g. test_auth.py) may replace global_args with a SimpleNamespace
    # that lacks max_upload_size; pin a known state so the upload endpoint runs.
    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(max_upload_size=None)
    )
    doc_manager = DocumentManager(str(tmp_path))
    rag = _DuplicateUploadRag(
        {
            "failed.docx": {
                "status": DocStatus.FAILED.value,
                "file_path": "failed.docx",
                "track_id": "track-failed",
                "metadata": {"error_type": "file_extraction_error"},
            }
        }
    )
    router = create_document_routes(rag, doc_manager)
    upload_endpoint = _get_endpoint(router, "upload_to_input_dir")
    upload_file = _document_routes.UploadFile(
        filename="failed.docx",
        file=BytesIO(b"replacement docx bytes"),
    )

    with pytest.raises(_document_routes.HTTPException) as excinfo:
        await upload_endpoint(set(), upload_file)
    assert excinfo.value.status_code == 409
    assert "failed.docx" in excinfo.value.detail
    assert "Status: failed" in excinfo.value.detail
    assert not (tmp_path / "failed.docx").exists()


async def test_upload_rejects_parser_hinted_filesystem_duplicate(tmp_path, monkeypatch):
    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(max_upload_size=None)
    )
    (tmp_path / "existing.docx").write_bytes(b"existing docx bytes")
    doc_manager = DocumentManager(str(tmp_path))
    rag = _DuplicateUploadRag({})
    router = create_document_routes(rag, doc_manager)
    upload_endpoint = _get_endpoint(router, "upload_to_input_dir")
    upload_file = _document_routes.UploadFile(
        filename="existing.[native].docx",
        file=BytesIO(b"replacement docx bytes"),
    )

    with pytest.raises(_document_routes.HTTPException) as excinfo:
        await upload_endpoint(set(), upload_file)
    assert excinfo.value.status_code == 409
    assert "existing.docx" in excinfo.value.detail
    assert not (tmp_path / "existing.[native].docx").exists()


@pytest.mark.parametrize(
    ("case", "leaked_text"),
    [
        ("opener_construction", "input directory unavailable"),
        ("open_time_permission", "Permission denied"),
    ],
)
async def test_upload_non_conflict_opener_oserror_returns_500_not_409(
    tmp_path, monkeypatch, case, leaked_text
):
    """Non-conflict OSErrors during opener setup or file creation (permissions,
    missing input dir, ENOSPC, etc.) are server-side faults and must return 500
    without echoing raw error details, rather than hitting the O_EXCL 409 path."""
    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(max_upload_size=None)
    )
    doc_manager = DocumentManager(str(tmp_path))
    router = create_document_routes(_DuplicateUploadRag({}), doc_manager)
    upload_endpoint = _get_endpoint(router, "upload_to_input_dir")
    upload_file = _document_routes.UploadFile(
        filename="safe.pdf", file=BytesIO(b"content")
    )

    if case == "opener_construction":

        def _failing_opener(_input_dir):
            raise OSError("input directory unavailable")

    else:

        def _failing_opener(_input_dir):
            def opener(path, flags):
                raise PermissionError(13, "Permission denied")

            return opener

    monkeypatch.setattr(_document_routes, "upload_file_opener", _failing_opener)

    with pytest.raises(_document_routes.HTTPException) as excinfo:
        await upload_endpoint(set(), upload_file)

    assert excinfo.value.status_code == 500
    assert leaked_text not in str(excinfo.value.detail)
    assert not (tmp_path / "safe.pdf").exists()


async def test_upload_write_failure_cleans_up_partial_file_and_returns_500(
    tmp_path, monkeypatch
):
    """An OSError raised after the exclusive-create opener succeeded (e.g. ENOSPC
    mid-stream) must delete the partial file and surface as a sanitized 500."""
    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(max_upload_size=None)
    )
    doc_manager = DocumentManager(str(tmp_path))
    router = create_document_routes(_DuplicateUploadRag({}), doc_manager)
    upload_endpoint = _get_endpoint(router, "upload_to_input_dir")
    upload_file = _document_routes.UploadFile(
        filename="safe.pdf", file=BytesIO(b"content")
    )

    call_count = 0

    async def _read_then_fail(_size):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return b"partial content"
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(upload_file, "read", _read_then_fail)

    with pytest.raises(_document_routes.HTTPException) as excinfo:
        await upload_endpoint(set(), upload_file)

    assert excinfo.value.status_code == 500
    assert "No space left on device" not in str(excinfo.value.detail)
    assert not (tmp_path / "safe.pdf").exists()


@pytest.mark.parametrize("failure", ["cancel", "read_error"])
async def test_incomplete_upload_is_removed_and_can_be_retried(
    tmp_path, monkeypatch, failure
):
    from lightrag.kg.shared_storage import (
        get_namespace_data,
        initialize_pipeline_status,
    )

    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(max_upload_size=None)
    )
    doc_manager = DocumentManager(str(tmp_path))
    rag = _DuplicateUploadRag({})
    await initialize_pipeline_status(workspace=rag.workspace)
    router = create_document_routes(rag, doc_manager)
    endpoint = _get_endpoint(router, "upload_to_input_dir")
    managed = set()
    indexed = []

    async def index_file(rag_arg, file_path, track_id, **kwargs):
        indexed.append(file_path.read_bytes())

    monkeypatch.setattr(_document_routes, "pipeline_index_file", index_file)
    upload = _document_routes.UploadFile(filename="report.txt", file=BytesIO())
    reading = asyncio.Event()
    calls = 0

    async def read(size):
        nonlocal calls
        calls += 1
        if calls == 1:
            return b"truncated report"
        reading.set()
        if failure == "read_error":
            raise ValueError("broken upload stream")
        await asyncio.Event().wait()

    monkeypatch.setattr(upload, "read", read)
    task = asyncio.create_task(endpoint(managed, upload))
    await asyncio.wait_for(reading.wait(), 5)
    if failure == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(_document_routes.HTTPException) as excinfo:
            await task
        assert excinfo.value.status_code == 500

    assert not (tmp_path / "report.txt").exists()
    assert list(doc_manager.iter_new_files()) == []
    status = await get_namespace_data("pipeline_status", workspace=rag.workspace)
    assert status["pending_enqueues"] == 0
    assert indexed == []

    retry = _document_routes.UploadFile(
        filename="report.txt", file=BytesIO(b"complete")
    )
    response = await endpoint(managed, retry)
    await _await_managed(managed)
    assert response.status == "success"
    assert indexed == [b"complete"]


@pytest.mark.parametrize("stage", ["open", "write", "close"])
async def test_upload_cancellation_waits_for_inflight_file_io(
    tmp_path, monkeypatch, stage
):
    """Cancel while a real aiofiles executor thread owns the output file."""
    from lightrag.kg.shared_storage import (
        get_namespace_data,
        initialize_pipeline_status,
    )

    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(max_upload_size=None)
    )
    doc_manager = DocumentManager(str(tmp_path))
    rag = _DuplicateUploadRag({})
    await initialize_pipeline_status(workspace=rag.workspace)
    router = create_document_routes(rag, doc_manager)
    endpoint = _get_endpoint(router, "upload_to_input_dir")
    started = threading.Event()
    release = threading.Event()
    original_open = _document_routes.aiofiles.threadpool.sync_open
    handles = []

    def pause():
        started.set()
        assert release.wait(5), "test did not release the file IO thread"

    class ControlledWriter(io.BufferedWriter):
        def write(self, data):
            result = super().write(data)
            if stage == "write":
                pause()
            return result

        def close(self):
            if stage == "close" and not self.closed:
                pause()
            super().close()

    def open_file(path, mode, *args, **kwargs):
        kwargs["buffering"] = 0
        handle = ControlledWriter(original_open(path, mode, *args, **kwargs))
        handles.append(handle)
        if stage == "open":
            pause()
        return handle

    monkeypatch.setattr(_document_routes.aiofiles.threadpool, "sync_open", open_file)
    upload = _document_routes.UploadFile(
        filename="report.txt", file=BytesIO(b"complete report")
    )
    task = asyncio.create_task(endpoint(set(), upload))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel("shutdown")
        await asyncio.sleep(0)
        task.cancel("shutdown again")
        await asyncio.sleep(0)
        assert not task.done()
        status = await get_namespace_data("pipeline_status", workspace=rag.workspace)
        assert status["pending_enqueues"] == 1
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await asyncio.wait_for(task, 5)
    # The write settled after the cancel was requested, but the file it wrote
    # was removed: the cancellation must not claim a durable write.
    assert not _utils.cancellation_was_deferred(cancelled.value)
    await asyncio.to_thread(lambda: None)
    try:
        assert handles and all(handle.closed for handle in handles)
        assert not (tmp_path / "report.txt").exists()
        assert list(doc_manager.iter_new_files()) == []
        status = await get_namespace_data("pipeline_status", workspace=rag.workspace)
        assert status["pending_enqueues"] == 0
    finally:
        for handle in handles:
            handle.close()


@pytest.mark.parametrize("cancel", [False, True])
async def test_upload_open_conflict_never_removes_another_writers_file(
    tmp_path, monkeypatch, cancel, lightrag_log_records
):
    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(max_upload_size=None)
    )
    doc_manager = DocumentManager(str(tmp_path))
    router = create_document_routes(_DuplicateUploadRag({}), doc_manager)
    endpoint = _get_endpoint(router, "upload_to_input_dir")
    original_open = _document_routes.aiofiles.threadpool.sync_open
    started = threading.Event()
    release = threading.Event()
    path = tmp_path / "report.txt"

    def competing_open(*args, **kwargs):
        # Another writer wins the exclusive create after the route's precheck.
        path.write_bytes(b"other writer's complete report")
        started.set()
        assert release.wait(5)
        return original_open(*args, **kwargs)

    monkeypatch.setattr(
        _document_routes.aiofiles.threadpool, "sync_open", competing_open
    )
    upload = _document_routes.UploadFile(
        filename=path.name, file=BytesIO(b"replacement")
    )
    task = asyncio.create_task(endpoint(set(), upload))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        if cancel:
            task.cancel()
    finally:
        release.set()
    if cancel:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
    else:
        with pytest.raises(_document_routes.HTTPException) as excinfo:
            await asyncio.wait_for(task, 5)
        assert excinfo.value.status_code == 409
    assert path.read_bytes() == b"other writer's complete report"
    # Losing the name is the caller's ordinary conflict, cancelled or not --
    # never reported as a server-side file IO failure.
    assert [
        r.getMessage() for r in lightrag_log_records if r.levelname == "ERROR"
    ] == []


async def test_upload_streaming_size_limit_removes_owned_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(max_upload_size=4)
    )
    doc_manager = DocumentManager(str(tmp_path))
    router = create_document_routes(_DuplicateUploadRag({}), doc_manager)
    endpoint = _get_endpoint(router, "upload_to_input_dir")
    upload = _document_routes.UploadFile(filename="report.txt", file=BytesIO())
    chunks = iter([b"four", b"extra"])

    async def read(size):
        return next(chunks, b"")

    monkeypatch.setattr(upload, "read", read)
    with pytest.raises(_document_routes.HTTPException) as excinfo:
        await endpoint(set(), upload)
    assert excinfo.value.status_code == 413
    assert not (tmp_path / "report.txt").exists()
    assert list(doc_manager.iter_new_files()) == []


@pytest.mark.parametrize(
    ("filename", "body", "expected_status", "expected_detail"),
    [
        pytest.param(
            "bad.[native-FR].docx",
            b"docx bytes",
            400,
            "multiple chunking modes",
            id="malformed_hint",
        ),
        pytest.param(
            "small.[-R(chunk_ts=32,chunk_ol=0)].md",
            b"markdown body",
            400,
            "must be >= 64, got 32",
            id="chunk_size_below_minimum",
        ),
        pytest.param(
            "custom.[native-C].docx",
            b"docx bytes",
            422,
            "custom chunking requires",
            id="c_without_custom_callback",
        ),
    ],
)
async def test_upload_rejects_invalid_filename_hints_before_writing(
    tmp_path, monkeypatch, filename, body, expected_status, expected_detail
):
    """Malformed hints, below-minimum chunk sizes, and C without a custom
    callback are all rejected synchronously before any file is written."""
    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(max_upload_size=None)
    )
    doc_manager = DocumentManager(str(tmp_path))
    router = create_document_routes(_DuplicateUploadRag({}), doc_manager)
    upload_endpoint = _get_endpoint(router, "upload_to_input_dir")
    upload_file = _document_routes.UploadFile(filename=filename, file=BytesIO(body))

    with pytest.raises(_document_routes.HTTPException) as excinfo:
        await upload_endpoint(set(), upload_file)

    assert excinfo.value.status_code == expected_status
    assert expected_detail in str(excinfo.value.detail)
    assert not (tmp_path / filename).exists()


async def test_upload_accepts_c_when_custom_callback_is_injected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(max_upload_size=None)
    )
    doc_manager = DocumentManager(str(tmp_path))
    rag = _DuplicateUploadRag({}, chunking_func=lambda *args, **kwargs: [])

    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)

    async def _no_op_index(rag_arg, file_path, track_id=None, admission_token=None):
        return None

    monkeypatch.setattr(_document_routes, "pipeline_index_file", _no_op_index)
    router = create_document_routes(rag, doc_manager)
    upload_endpoint = _get_endpoint(router, "upload_to_input_dir")
    upload_file = _document_routes.UploadFile(
        filename="custom.[native-C].docx",
        file=BytesIO(b"docx bytes"),
    )

    managed: set = set()
    response = await upload_endpoint(managed, upload_file)
    await _await_managed(managed)

    assert response.status == "success"
    assert (tmp_path / "custom.[native-C].docx").exists()


async def test_docx_archive_failure_is_best_effort(tmp_path, monkeypatch):
    file_path = tmp_path / "archive-failure.docx"
    file_path.write_bytes(b"docx bytes")

    async def _raise_archive_failure(*args, **kwargs):
        raise OSError("simulated archive failure")

    import lightrag.utils_pipeline as _utils_pipeline
    from lightrag.utils_pipeline import archive_docx_source_after_full_docs_sync

    monkeypatch.setattr(
        _utils_pipeline, "move_file_to_parsed_dir", _raise_archive_failure
    )

    archived_path = await archive_docx_source_after_full_docs_sync(str(file_path))

    assert archived_path is None
    assert file_path.exists()


async def test_parse_native_archives_docx_after_full_docs_sync(tmp_path, monkeypatch):
    monkeypatch.setenv("INPUT_DIR", str(tmp_path))
    source_path = tmp_path / "parsed-after-sync.docx"
    source_path.write_bytes(b"docx bytes")
    rag = _ParseRag(tmp_path / "work", source_path)

    def _fake_extract(file_path, drawing_context=None, **kwargs):
        return [
            {
                "uuid": "p1",
                "uuid_end": "p1",
                "heading": "",
                "content": "parsed",
                "type": "text",
                "parent_headings": [],
                "level": 0,
                "table_chunk_role": "none",
            }
        ]

    monkeypatch.setattr(
        "lightrag.parser.docx.parse_document.extract_docx_blocks",
        _fake_extract,
    )

    result = await _parse_via_registry(
        rag,
        "native",
        "doc-test",
        str(source_path),
        {"parse_format": FULL_DOCS_FORMAT_PENDING_PARSE, "content": ""},
    )

    # parse_native returns LIGHTRAG-format parsed_data with merged_text
    # (not the {{LRdoc}} marker — that's only in the persisted full_docs row).
    assert result["content"]
    assert result["parse_format"] == "lightrag"
    assert result["blocks_path"]
    assert rag.full_docs.events == ["upsert", "index_done"]
    assert not source_path.exists()
    assert (tmp_path / PARSED_DIR_NAME / source_path.name).exists()
    parsed_artifact_dir = tmp_path / PARSED_DIR_NAME / f"{source_path.name}.parsed"
    assert parsed_artifact_dir.is_dir()
    assert (parsed_artifact_dir / "parsed-after-sync.blocks.jsonl").is_file()
    assert rag.full_docs.data["doc-test"]["parse_engine"] == "native"
    assert rag.full_docs.data["doc-test"]["parse_format"] == "lightrag"
    assert rag.full_docs.data["doc-test"]["content"].startswith("{{LRdoc}}")


def test_parsed_artifact_dir_uses_unique_suffix_when_path_is_file(
    tmp_path, monkeypatch
):
    from lightrag.utils_pipeline import parsed_artifact_dir_for

    monkeypatch.setenv("INPUT_DIR", str(tmp_path))
    parsed_dir = tmp_path / PARSED_DIR_NAME
    parsed_dir.mkdir()
    (parsed_dir / "demo.docx.parsed").write_text("legacy file", encoding="utf-8")

    artifact_dir = parsed_artifact_dir_for("demo.docx")

    assert artifact_dir == parsed_dir / "demo.docx.parsed_001"


def test_parsed_artifact_dir_reuses_existing_parsed_parent(tmp_path, monkeypatch):
    from lightrag.utils_pipeline import parsed_artifact_dir_for

    monkeypatch.setenv("INPUT_DIR", str(tmp_path))
    parsed_dir = tmp_path / PARSED_DIR_NAME
    parsed_dir.mkdir()

    artifact_dir = parsed_artifact_dir_for("demo.docx")

    assert artifact_dir == parsed_dir / "demo.docx.parsed"


@pytest.mark.parametrize(
    ("filename", "fake_extractor", "exc_type", "match"),
    [
        pytest.param(
            "content-list-failure.docx",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("content list boom")
            ),
            RuntimeError,
            "content list boom",
            id="extractor_raises",
        ),
        pytest.param(
            "empty-content-list.docx",
            lambda *args, **kwargs: [],
            ValueError,
            "empty content",
            id="empty_content_list",
        ),
    ],
)
async def test_parse_native_docx_failure_raises_without_fallback(
    tmp_path, monkeypatch, filename, fake_extractor, exc_type, match
):
    source_path = tmp_path / filename
    source_path.write_bytes(b"docx bytes")
    rag = _ParseRag(tmp_path / "work", source_path)

    monkeypatch.setattr(
        "lightrag.parser.docx.parse_document.extract_docx_blocks",
        fake_extractor,
    )

    with pytest.raises(exc_type, match=match):
        await _parse_via_registry(
            rag,
            "native",
            "doc-test",
            str(source_path),
            {"parse_format": FULL_DOCS_FORMAT_PENDING_PARSE, "content": ""},
        )

    assert source_path.exists()
    assert rag.full_docs.events == []


def test_lightrag_document_reprocess_uses_full_docs_without_reparse():
    engine = resolve_stored_document_parser_engine(
        "report.[mineru].docx",
        {
            "parse_format": FULL_DOCS_FORMAT_LIGHTRAG,
            "sidecar_location": "file:///tmp/report.docx.parsed/",
            "parse_engine": "mineru",
        },
    )

    # All lightrag rows route to the internal reuse handler (reuse the stored
    # sidecar without re-parsing) regardless of the originating engine.
    assert engine == "reuse"


@pytest.mark.parametrize(
    # A control character is the shared character rule, pinned in
    # test_document_routes_ingress_and_types.py.
    "filename",
    [
        pytest.param("../report.pdf", id="traversal"),
        pytest.param("nested/report.pdf", id="posix_separator"),
        pytest.param("nested\\report.pdf", id="windows_separator"),
        pytest.param("safe.pdf:payload.pdf", id="ntfs_ads_colon"),
        pytest.param("C:report.pdf", id="drive_relative_colon"),
    ],
)
def test_sanitize_filename_rejects_unsafe_names(tmp_path, filename):
    with pytest.raises(_document_routes.HTTPException) as exc:
        _document_routes.sanitize_filename(filename, tmp_path)

    assert exc.value.status_code == 400


@pytest.mark.parametrize(
    "filename",
    [
        # Unlike ':', glob metacharacters don't let the name be reinterpreted
        # as a different path (no shell-out or unescaped glob() call site uses
        # upload filenames in this codebase).
        pytest.param("glob?name.pdf", id="question_mark"),
        pytest.param("glob*name.pdf", id="asterisk"),
        pytest.param("report..final.pdf", id="non_traversal_dots"),
        pytest.param("report.[native].docx", id="parser_hint"),
    ],
)
def test_sanitize_filename_allows_safe_names(tmp_path, filename):
    assert _document_routes.sanitize_filename(filename, tmp_path) == filename


def test_upload_file_opener_refuses_existing_symlink(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("keep", encoding="utf-8")
    link = tmp_path / "safe.pdf"
    link.symlink_to(outside)

    opener = _document_routes.upload_file_opener(tmp_path)
    with pytest.raises(FileExistsError):
        with open(link, "xb", opener=opener):
            pass

    assert outside.read_text(encoding="utf-8") == "keep"


def test_upload_file_opener_creates_new_file_with_private_permissions(tmp_path):
    path = tmp_path / "safe.pdf"

    opener = _document_routes.upload_file_opener(tmp_path)
    with open(path, "xb", opener=opener) as fh:
        fh.write(b"content")

    assert path.read_bytes() == b"content"
    if sys.platform != "win32":  # POSIX chmod semantics; NTFS reports 0o666
        assert (path.stat().st_mode & 0o777) == 0o600


def test_upload_file_opener_falls_back_when_dir_fd_unsupported(tmp_path, monkeypatch):
    """Simulates Windows: os.open() cannot bind a directory fd there at all,
    so the opener must skip that step entirely rather than crash on it."""
    monkeypatch.setattr(_document_routes, "_UPLOAD_DIR_FD_SUPPORTED", False)

    real_open = _document_routes.os.open
    directory_open_attempts = []

    def spying_open(path, flags, mode=0o777, *, dir_fd=None):
        if dir_fd is not None or str(path) == str(tmp_path):
            directory_open_attempts.append((str(path), dir_fd))
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(_document_routes.os, "open", spying_open)

    path = tmp_path / "safe.pdf"
    opener = _document_routes.upload_file_opener(tmp_path)
    with open(path, "xb", opener=opener) as fh:
        fh.write(b"content")

    assert path.read_bytes() == b"content"
    if sys.platform != "win32":  # POSIX chmod semantics; NTFS reports 0o666
        assert (path.stat().st_mode & 0o777) == 0o600
    assert directory_open_attempts == []

    with pytest.raises(FileExistsError):
        with open(path, "xb", opener=_document_routes.upload_file_opener(tmp_path)):
            pass


def test_default_allowlist_equals_local_engine_suffixes(tmp_path, monkeypatch):
    """With no external endpoints and no routing rules, the registry-derived
    allowlist must equal the local engines' (legacy ∪ native) suffixes —
    i.e. exactly the historical hardcoded upload allowlist."""
    from lightrag.parser import registry

    for var in (
        "MINERU_LOCAL_ENDPOINT",
        "MINERU_API_TOKEN",
        "DOCLING_ENDPOINT",
        "LIGHTRAG_PARSER",
    ):
        monkeypatch.delenv(var, raising=False)

    dm = DocumentManager(str(tmp_path))
    local = {f".{s}" for s in registry.suffix_capabilities("legacy")} | {
        f".{s}" for s in registry.suffix_capabilities("native")
    }
    assert set(dm.supported_extensions) == local
    # External-only suffixes stay out while their endpoints are unset.
    assert ".png" not in dm.supported_extensions
    assert ".doc" not in dm.supported_extensions


def test_unroutable_suffix_needs_rule_or_hint(tmp_path, monkeypatch):
    """An endpoint-configured engine's suffix is accepted only when routing
    actually reaches that engine: a bare filename needs a LIGHTRAG_PARSER
    rule; a per-file hint works without one."""
    monkeypatch.delenv("DOCLING_ENDPOINT", raising=False)
    monkeypatch.delenv("LIGHTRAG_PARSER", raising=False)
    monkeypatch.setenv("MINERU_API_MODE", "local")
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://fake-mineru")

    dm = DocumentManager(str(tmp_path))
    assert ".png" not in dm.supported_extensions
    assert not dm.is_supported_file("scan.png")
    assert dm.is_supported_file("scan.[mineru].png")
    monkeypatch.setenv("LIGHTRAG_PARSER", "png:mineru")
    assert ".png" in dm.supported_extensions
    assert dm.is_supported_file("scan.png")
    assert ".xhtml" not in dm.supported_extensions


def test_third_party_engine_suffixes_join_allowlist_and_scan(tmp_path, monkeypatch):
    """A registered third-party engine's suffixes become uploadable and
    scannable once routable (rule for bare names, hint for individual
    files), and revert on unregister."""
    from lightrag.parser import registry

    monkeypatch.delenv("LIGHTRAG_PARSER", raising=False)
    dm = DocumentManager(str(tmp_path))

    assert not dm.is_supported_file("sample.foo")
    (dm.input_dir / "sample.foo").write_text("x", encoding="utf-8")
    (dm.input_dir / "hinted.[fooengine].foo").write_text("x", encoding="utf-8")
    assert not [p for p in dm.iter_new_files() if p.suffix == ".foo"]

    registry.register_parser(
        registry.ParserSpec(
            engine_name="fooengine",
            impl="x:Y",
            suffixes=frozenset({"foo"}),
            queue_group="fooengine",
            concurrency=1,
        )
    )
    try:
        assert not dm.is_supported_file("sample.foo")
        assert dm.is_supported_file("hinted.[fooengine].foo")
        scanned = {p.name for p in dm.iter_new_files()}
        assert "hinted.[fooengine].foo" in scanned
        assert "sample.foo" not in scanned
        monkeypatch.setenv("LIGHTRAG_PARSER", "foo:fooengine")
        assert ".foo" in dm.supported_extensions
        assert dm.is_supported_file("sample.foo")
        assert "sample.foo" in {p.name for p in dm.iter_new_files()}
    finally:
        registry._REGISTRY.pop("fooengine", None)
    assert not dm.is_supported_file("sample.foo")
