import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
_base = importlib.import_module("lightrag.base")
_constants = importlib.import_module("lightrag.constants")
sys.argv = _original_argv

DeletionResult = _base.DeletionResult
SourceAbsent = _base.SourceAbsent
SourceConflict = _base.SourceConflict
SourceUnique = _base.SourceUnique
PARSED_DIR_NAME = _constants.PARSED_DIR_NAME
DocumentManager = _document_routes.DocumentManager

pytestmark = pytest.mark.offline


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


class _DeleteDocStatus:
    def __init__(self, rows: dict | None = None):
        self.rows = rows or {}

    def _primary_matches(self, basename: str) -> list[tuple[str, dict]]:
        return [
            (doc_id, row)
            for doc_id, row in self.rows.items()
            if row.get("file_path") == basename
            and not (row.get("metadata") or {}).get("is_duplicate")
        ]

    async def get_doc_by_file_basename(self, basename: str):
        # Legacy best-effort lookup; real backends expose both methods. The
        # route's ownership check must NOT use this one (mongo/redis/opensearch
        # swallow query failures into None), pinned by the fail-closed test below.
        matches = self._primary_matches(basename)
        return matches[0] if matches else None

    async def resolve_doc_source_strict(self, canonical_source_key: str):
        matches = self._primary_matches(canonical_source_key)
        if not matches:
            return SourceAbsent()
        if len(matches) == 1:
            doc_id, row = matches[0]
            return SourceUnique(doc_id=doc_id, doc=row)
        return SourceConflict(
            candidate_count=len(matches),
            sample_doc_ids=tuple(doc_id for doc_id, _ in matches[:2]),
        )


class _DeleteRag:
    def __init__(self, result, status_rows=None):
        self.result = result
        self.workspace = f"delete-test-{uuid4().hex}"
        self.deleted_doc_ids = []
        self.doc_status = _DeleteDocStatus(status_rows)

    async def adelete_by_doc_id(self, doc_id, delete_llm_cache=False):
        self.deleted_doc_ids.append((doc_id, delete_llm_cache))
        self.doc_status.rows.pop(doc_id, None)
        return self.result

    async def apipeline_process_enqueue_documents(self):
        return None


class _OtherSourceLookupFailsDocStatus(_DeleteDocStatus):
    def __init__(self, rows, failing_key):
        super().__init__(rows)
        self.failing_key = failing_key

    async def resolve_doc_source_strict(self, canonical_source_key: str):
        if canonical_source_key == self.failing_key:
            raise RuntimeError("simulated backend read failure")
        return await super().resolve_doc_source_strict(canonical_source_key)


class _FailingOwnershipDocStatus(_DeleteDocStatus):
    """Simulates a transient backend read failure during the ownership check.

    Mirrors real backend semantics: the legacy ``get_doc_by_file_basename``
    swallows the failure and returns ``None``, while ``resolve_doc_source_strict``
    raises per its fail-closed contract.
    """

    async def get_doc_by_file_basename(self, basename: str):
        return None

    async def resolve_doc_source_strict(self, canonical_source_key: str):
        raise RuntimeError("simulated backend read failure")


def _seed_archive_family(tmp_path):
    """``scan.pdf`` archived twice, then ``scan_001.pdf`` archived once."""
    parsed_dir = tmp_path / PARSED_DIR_NAME
    parsed_dir.mkdir()
    files = {}
    for name in ("scan.pdf", "scan_001.pdf", "scan_001_001.pdf", "scan_002.pdf"):
        files[name] = parsed_dir / name
        files[name].write_bytes(name.encode())
    return files


async def _run_background_delete(tmp_path, rag, doc_id, *, delete_llm_cache=False):
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    shared_storage.initialize_share_data()
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    await _document_routes.background_delete_documents(
        rag,
        DocumentManager(str(tmp_path)),
        [doc_id],
        delete_file=True,
        delete_llm_cache=delete_llm_cache,
    )
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    return list(pipeline_status.get("history_messages", []))


async def test_delete_release_drives_processing_on_mailbox_work(tmp_path):
    """The post-delete queue drive keys off the ingress ``has_work`` probe
    inside the owner-checked release: work arriving while the delete holds the
    slot triggers one drive; an empty mailbox releases without driving."""
    workspace = f"delete-drive-{uuid4().hex}"
    destructive_token = uuid4().hex

    class _Rag:
        pass

    rag = _Rag()
    rag.workspace = workspace
    doc_manager = DocumentManager(str(tmp_path))

    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=workspace)
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=workspace
    )
    ingress = await shared_storage.get_pipeline_ingress(workspace)

    drives = {"n": 0}

    async def _process():
        drives["n"] += 1

    rag.apipeline_process_enqueue_documents = _process

    async def _adelete_and_signal(doc_id, delete_llm_cache=False):
        ingress.request_auto_rescan()
        return SimpleNamespace(status="success", file_path="-", message="ok")

    rag.adelete_by_doc_id = _adelete_and_signal

    pipeline_status.update(
        {"busy": True, "destructive_busy": True, "busy_owner": destructive_token}
    )
    await _document_routes.background_delete_documents(
        rag, doc_manager, ["doc-1"], token=destructive_token
    )
    assert drives["n"] == 1
    assert pipeline_status.get("busy") is False
    assert pipeline_status.get("destructive_busy") is False

    # Empty mailbox -> release without driving.
    ingress.consume_auto_rescan()

    async def _adelete_quiet(doc_id, delete_llm_cache=False):
        return SimpleNamespace(status="success", file_path="-", message="ok")

    rag.adelete_by_doc_id = _adelete_quiet
    pipeline_status.update(
        {"busy": True, "destructive_busy": True, "busy_owner": destructive_token}
    )
    await _document_routes.background_delete_documents(
        rag, doc_manager, ["doc-2"], token=destructive_token
    )
    assert drives["n"] == 1
    assert pipeline_status.get("busy") is False


async def test_delete_release_fails_toward_drive_on_ingress_failures(
    tmp_path, monkeypatch
):
    """Both delete-exit failure arms fail TOWARD the post-delete drive: an
    ingress resolve failure and a ``has_work`` probe failure each still
    release the destructive slot and fire exactly one busy-gated drive."""
    workspace = f"delete-drive-fail-{uuid4().hex}"
    destructive_token = uuid4().hex

    class _Rag:
        pass

    rag = _Rag()
    rag.workspace = workspace
    doc_manager = DocumentManager(str(tmp_path))

    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=workspace)
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=workspace
    )

    drives = {"n": 0}

    async def _process():
        drives["n"] += 1

    rag.apipeline_process_enqueue_documents = _process

    async def _adelete(doc_id, delete_llm_cache=False):
        return SimpleNamespace(status="success", file_path="-", message="ok")

    rag.adelete_by_doc_id = _adelete

    # Arm 1: the resolve itself fails -> delete_exit_ingress=None -> drive.
    async def _broken_resolve(workspace=None):
        raise RuntimeError("manager connection lost")

    monkeypatch.setattr(shared_storage, "get_pipeline_ingress", _broken_resolve)
    pipeline_status.update(
        {"busy": True, "destructive_busy": True, "busy_owner": destructive_token}
    )
    await _document_routes.background_delete_documents(
        rag, doc_manager, ["doc-1"], token=destructive_token
    )
    assert drives["n"] == 1
    assert pipeline_status.get("busy") is False
    assert pipeline_status.get("destructive_busy") is False

    # Arm 2: the resolve succeeds but the in-lock has_work probe fails.
    class _ProbeBroken:
        def has_work(self):
            raise RuntimeError("probe transport down")

    async def _resolve_probe_broken(workspace=None):
        return _ProbeBroken()

    monkeypatch.setattr(shared_storage, "get_pipeline_ingress", _resolve_probe_broken)
    pipeline_status.update(
        {"busy": True, "destructive_busy": True, "busy_owner": destructive_token}
    )
    await _document_routes.background_delete_documents(
        rag, doc_manager, ["doc-2"], token=destructive_token
    )
    assert drives["n"] == 2
    assert pipeline_status.get("busy") is False
    assert pipeline_status.get("destructive_busy") is False


def test_delete_file_variants_removes_canonical_hint_variants(tmp_path):
    parsed_dir = tmp_path / PARSED_DIR_NAME
    parsed_dir.mkdir()
    files_to_delete = [
        tmp_path / "report.docx",
        tmp_path / "report.[native].docx",
        parsed_dir / "report.[mineru].docx",
        parsed_dir / "report.[native]_001.docx",
        parsed_dir / "report_001.docx",
    ]
    for path in files_to_delete:
        path.write_bytes(b"delete me")
    unrelated_files = [
        tmp_path / "report_001.docx",
        tmp_path / "report_2024.docx",
        parsed_dir / "other.[native].docx",
    ]
    for path in unrelated_files:
        path.write_bytes(b"keep me")
    artifact_dirs = [
        parsed_dir / "report.docx.parsed",
        parsed_dir / "report.docx.parsed_001",
    ]
    for artifact_dir in artifact_dirs:
        artifact_dir.mkdir()
        (artifact_dir / "report.blocks.jsonl").write_text("{}", encoding="utf-8")
    unrelated_artifact_dir = parsed_dir / "other.docx.parsed"
    unrelated_artifact_dir.mkdir()
    (unrelated_artifact_dir / "other.blocks.jsonl").write_text("{}", encoding="utf-8")

    deleted_files, errors = _document_routes.delete_file_variants_by_file_path(
        tmp_path,
        "report.docx",
    )

    assert errors == []
    assert set(deleted_files) == {
        "report.docx",
        "report.[native].docx",
        str(Path(PARSED_DIR_NAME) / "report.[mineru].docx"),
        str(Path(PARSED_DIR_NAME) / "report.[native]_001.docx"),
        str(Path(PARSED_DIR_NAME) / "report_001.docx"),
        str(Path(PARSED_DIR_NAME) / "report.docx.parsed"),
        str(Path(PARSED_DIR_NAME) / "report.docx.parsed_001"),
    }
    assert all(not path.exists() for path in files_to_delete)
    assert all(path.exists() for path in unrelated_files)
    assert all(not artifact_dir.exists() for artifact_dir in artifact_dirs)
    assert unrelated_artifact_dir.is_dir()


@pytest.mark.parametrize(
    ("file_path", "numbered_copy"),
    [
        ("scan_001.pdf", "scan_001_001.pdf"),
        ("export_1700000000.csv", "export_1700000000_001.csv"),
    ],
)
def test_delete_file_variants_removes_archive_of_name_ending_in_archive_suffix(
    tmp_path, file_path, numbered_copy
):
    # A document whose own name ends like an archive suffix keeps that name
    # when it is first archived into __parsed__; delete_file must still find it.
    parsed_dir = tmp_path / PARSED_DIR_NAME
    parsed_dir.mkdir()
    archived = parsed_dir / file_path
    archived.write_bytes(b"delete me")
    (parsed_dir / numbered_copy).write_bytes(b"delete me")
    unrelated = parsed_dir / "other_001.pdf"
    unrelated.write_bytes(b"keep me")

    deleted_files, errors = _document_routes.delete_file_variants_by_file_path(
        tmp_path,
        file_path,
    )

    assert errors == []
    assert set(deleted_files) == {
        str(Path(PARSED_DIR_NAME) / file_path),
        str(Path(PARSED_DIR_NAME) / numbered_copy),
    }
    assert not archived.exists()
    assert unrelated.exists()


def test_find_archived_variants_shared_with_other_sources(tmp_path):
    parsed_dir = tmp_path / PARSED_DIR_NAME
    parsed_dir.mkdir()
    for name in ("scan.pdf", "scan_001.pdf", "scan_001_001.pdf", "other_001.pdf"):
        (parsed_dir / name).write_bytes(b"x")
    (parsed_dir / "scan_002.pdf.parsed").mkdir()

    find = _document_routes.find_archived_variants_shared_with_other_sources
    assert find(tmp_path, "scan.pdf") == {"scan_001.pdf": ["scan_001.pdf"]}
    assert find(tmp_path, "scan_001.pdf") == {
        "scan.pdf": ["scan_001.pdf"],
        "scan_001_001.pdf": ["scan_001_001.pdf"],
    }
    assert find(tmp_path, "unknown.pdf") == {}
    assert find(tmp_path / "missing", "scan.pdf") == {}


@pytest.mark.parametrize(
    ("deleted_source", "live_source", "kept", "removed"),
    [
        # Deleting scan.pdf must not take the first archive of live scan_001.pdf.
        (
            "scan.pdf",
            "scan_001.pdf",
            {"scan_001.pdf", "scan_001_001.pdf"},
            {"scan.pdf", "scan_002.pdf"},
        ),
        # Deleting scan_001.pdf must not take a numbered copy of live scan.pdf.
        (
            "scan_001.pdf",
            "scan.pdf",
            {"scan.pdf", "scan_001.pdf", "scan_002.pdf"},
            {"scan_001_001.pdf"},
        ),
    ],
)
async def test_background_delete_keeps_archive_shared_with_live_source(
    tmp_path, deleted_source, live_source, kept, removed
):
    files = _seed_archive_family(tmp_path)
    rag = _DeleteRag(
        DeletionResult(
            status="success",
            doc_id="doc-deleted",
            message="deleted",
            file_path=deleted_source,
        ),
        status_rows={
            "doc-deleted": {"file_path": deleted_source, "metadata": {}},
            "doc-live": {"file_path": live_source, "metadata": {}},
        },
    )

    history = await _run_background_delete(tmp_path, rag, "doc-deleted")

    assert {name for name, path in files.items() if path.exists()} == kept
    assert not any(files[name].exists() for name in removed)
    assert any(
        "Archived file kept" in msg and f"'{live_source}'" in msg for msg in history
    )
    assert not any("missing or unsafe" in msg for msg in history)


async def test_background_delete_removes_shared_archive_without_live_source(tmp_path):
    files = _seed_archive_family(tmp_path)
    rag = _DeleteRag(
        DeletionResult(
            status="success",
            doc_id="doc-deleted",
            message="deleted",
            file_path="scan_001.pdf",
        ),
        status_rows={"doc-deleted": {"file_path": "scan_001.pdf", "metadata": {}}},
    )

    history = await _run_background_delete(tmp_path, rag, "doc-deleted")

    assert not files["scan_001.pdf"].exists()
    assert not files["scan_001_001.pdf"].exists()
    assert files["scan.pdf"].exists()
    assert files["scan_002.pdf"].exists()
    assert not any("Archived file kept" in msg for msg in history)


async def test_background_delete_keeps_shared_archive_when_lookup_fails(tmp_path):
    files = _seed_archive_family(tmp_path)
    rag = _DeleteRag(
        DeletionResult(
            status="success",
            doc_id="doc-deleted",
            message="deleted",
            file_path="scan_001.pdf",
        ),
    )
    rag.doc_status = _OtherSourceLookupFailsDocStatus(
        {"doc-deleted": {"file_path": "scan_001.pdf", "metadata": {}}},
        failing_key="scan.pdf",
    )

    history = await _run_background_delete(tmp_path, rag, "doc-deleted")

    assert files["scan_001.pdf"].exists()
    assert not files["scan_001_001.pdf"].exists()
    assert any(
        "Archived file kept" in msg and "could not be checked" in msg for msg in history
    )


async def test_background_delete_removes_parser_hint_file_variants(tmp_path):
    parsed_dir = tmp_path / PARSED_DIR_NAME
    parsed_dir.mkdir()
    source_file = tmp_path / "paper.[native].docx"
    source_file.write_bytes(b"source")
    parsed_variant = parsed_dir / "paper.[mineru]_001.docx"
    parsed_variant.write_bytes(b"parsed")
    unrelated_file = tmp_path / "other.[native].docx"
    unrelated_file.write_bytes(b"keep")
    rag = _DeleteRag(
        DeletionResult(
            status="success",
            doc_id="doc-paper",
            message="deleted",
            file_path="paper.docx",
        )
    )

    await _run_background_delete(tmp_path, rag, "doc-paper", delete_llm_cache=True)

    assert rag.deleted_doc_ids == [("doc-paper", True)]
    assert not source_file.exists()
    assert not parsed_variant.exists()
    assert unrelated_file.exists()


async def test_background_delete_duplicate_marker_preserves_shared_source_file(
    tmp_path,
):
    """Deleting a ``dup-*`` status row must not delete the primary's file."""
    source_file = tmp_path / "paper.pdf"
    source_file.write_bytes(b"primary source")
    duplicate_id = "dup-attempt"
    rag = _DeleteRag(
        DeletionResult(
            status="success",
            doc_id=duplicate_id,
            message="deleted duplicate marker",
            file_path="paper.pdf",
        ),
        status_rows={
            duplicate_id: {
                "file_path": "paper.pdf",
                "metadata": {
                    "is_duplicate": True,
                    "original_doc_id": "doc-primary",
                },
            },
            "doc-primary": {
                "file_path": "paper.pdf",
                "metadata": {},
            },
        },
    )

    await _run_background_delete(tmp_path, rag, duplicate_id)

    assert rag.deleted_doc_ids == [(duplicate_id, False)]
    assert source_file.read_bytes() == b"primary source"


async def test_background_delete_post_parse_duplicate_removes_owned_archive(
    tmp_path,
):
    """A demoted ``doc-*`` row still owns its unique archived source."""
    parsed_dir = tmp_path / PARSED_DIR_NAME
    parsed_dir.mkdir()
    archived_source = parsed_dir / "twin.pdf"
    archived_source.write_bytes(b"duplicate content under a unique filename")
    artifact_dir = parsed_dir / "twin.pdf.parsed"
    artifact_dir.mkdir()
    sidecar = artifact_dir / "twin.blocks.jsonl"
    sidecar.write_text("{}", encoding="utf-8")
    primary_source = parsed_dir / "primary.pdf"
    primary_source.write_bytes(b"primary source")
    duplicate_id = "doc-twin"
    rag = _DeleteRag(
        DeletionResult(
            status="success",
            doc_id=duplicate_id,
            message="deleted post-parse duplicate",
            file_path="twin.pdf",
        ),
        status_rows={
            duplicate_id: {
                "file_path": "twin.pdf",
                "metadata": {
                    "is_duplicate": True,
                    "duplicate_kind": "content_hash",
                    "original_doc_id": "doc-primary",
                },
            },
            "doc-primary": {
                "file_path": "primary.pdf",
                "metadata": {},
            },
        },
    )

    await _run_background_delete(tmp_path, rag, duplicate_id)

    assert rag.deleted_doc_ids == [(duplicate_id, False)]
    assert not archived_source.exists()
    assert not artifact_dir.exists()
    assert not sidecar.exists()
    assert primary_source.read_bytes() == b"primary source"


async def test_background_delete_primary_still_removes_source_file(tmp_path):
    """The duplicate guard must not weaken ordinary ``delete_file=True``."""
    source_file = tmp_path / "paper.pdf"
    source_file.write_bytes(b"primary source")
    primary_id = "doc-primary"
    rag = _DeleteRag(
        DeletionResult(
            status="success",
            doc_id=primary_id,
            message="deleted primary document",
            file_path="paper.pdf",
        ),
        status_rows={
            primary_id: {
                "file_path": "paper.pdf",
                "metadata": {},
            }
        },
    )

    await _run_background_delete(tmp_path, rag, primary_id)

    assert rag.deleted_doc_ids == [(primary_id, False)]
    assert not source_file.exists()


@pytest.mark.parametrize(
    "case",
    ["ownership_lookup_failure", "source_conflict", "stale_self_match"],
)
async def test_background_delete_preserves_file_when_ownership_uncertain(
    tmp_path, case
):
    """Fail-closed ownership guard: lookup failure, surviving conflicting
    primaries, or an eventually-consistent stale self-match must keep the
    physical source file while still succeeding on the record deletion."""
    source_file = tmp_path / "paper.pdf"
    body = b"conflicted source" if case == "source_conflict" else b"primary source"
    source_file.write_bytes(body)

    if case == "ownership_lookup_failure":
        deleted_id = "dup-attempt"
        rag = _DeleteRag(
            DeletionResult(
                status="success",
                doc_id=deleted_id,
                message="deleted duplicate marker",
                file_path="paper.pdf",
            ),
            status_rows={
                deleted_id: {
                    "file_path": "paper.pdf",
                    "metadata": {"is_duplicate": True},
                },
                "doc-primary": {"file_path": "paper.pdf", "metadata": {}},
            },
        )
        rag.doc_status = _FailingOwnershipDocStatus(rag.doc_status.rows)
    elif case == "source_conflict":
        deleted_id = "doc-a"
        rag = _DeleteRag(
            DeletionResult(
                status="success",
                doc_id=deleted_id,
                message="deleted one conflicting primary",
                file_path="paper.pdf",
            ),
            status_rows={
                deleted_id: {"file_path": "paper.pdf", "metadata": {}},
                "doc-b": {"file_path": "paper.pdf", "metadata": {}},
                "doc-c": {"file_path": "paper.pdf", "metadata": {}},
            },
        )
    else:
        assert case == "stale_self_match"
        deleted_id = "doc-stale"

        class _StaleDocStatus(_DeleteDocStatus):
            async def resolve_doc_source_strict(self, canonical_source_key: str):
                return SourceUnique(
                    doc_id=deleted_id, doc={"file_path": canonical_source_key}
                )

        rag = _DeleteRag(
            DeletionResult(
                status="success",
                doc_id=deleted_id,
                message="deleted document",
                file_path="paper.pdf",
            ),
        )
        rag.doc_status = _StaleDocStatus()

    await _run_background_delete(tmp_path, rag, deleted_id)

    assert rag.deleted_doc_ids == [(deleted_id, False)]
    assert source_file.read_bytes() == body
