"""Disk-spooled scan ordering, mtime enqueue boundary, and streaming scan batch tests."""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

from lightrag.base import SourceAbsent  # noqa: E402
from lightrag.constants import PARSED_DIR_NAME  # noqa: E402
from tests._log_capture import capture_lightrag_logs  # noqa: E402

DocumentManager = _document_routes.DocumentManager
UNKNOWN_FILE_SOURCE = _document_routes.UNKNOWN_FILE_SOURCE
_ScanCandidateSpool = _document_routes._ScanCandidateSpool
_scan_spool_base_dir = _document_routes._scan_spool_base_dir
pipeline_enqueue_file = _document_routes.pipeline_enqueue_file
run_scanning_process = _document_routes.run_scanning_process

pytestmark = pytest.mark.offline

_MTIME_EPOCH = 1565157904


@pytest.fixture(autouse=True)
def _ensure_shared_storage_initialized():
    importlib.import_module("lightrag.kg.shared_storage").initialize_share_data()
    yield


def _aged(path: Path, epoch: int, body: str = "body") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    os.utime(path, (epoch, epoch))
    return path


def _drain(spool: _ScanCandidateSpool, batch_size: int) -> list[list[str]]:
    return [
        [candidate.path.name for candidate in batch]
        for batch in spool.ordered_batches(batch_size)
    ]


def _spool_rag(working_dir: str | None = None, workspace: str = ""):
    attrs = {"workspace": workspace}
    if working_dir is not None:
        attrs["working_dir"] = working_dir
    return type("_Rag", (), attrs)()


# --------------------------------------------------------------------------- #
# Scan candidate spool placement, claim semantics, ordering, bounded reads
# --------------------------------------------------------------------------- #


def test_base_dir_prefers_scan_spool_dir_then_working_dir(tmp_path, monkeypatch):
    rag = _spool_rag(str(tmp_path / "rag_storage"))

    monkeypatch.setattr(
        _document_routes.global_args, "scan_spool_dir", "", raising=False
    )
    assert (
        _scan_spool_base_dir(rag) == tmp_path / "rag_storage" / "scan_spool" / "unnamed"
    )

    monkeypatch.setattr(
        _document_routes.global_args,
        "scan_spool_dir",
        str(tmp_path / "fast-disk"),
        raising=False,
    )
    assert _scan_spool_base_dir(rag) == tmp_path / "fast-disk" / "unnamed"


def test_each_workspace_gets_its_own_spool_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        _document_routes.global_args, "scan_spool_dir", "", raising=False
    )
    base = tmp_path / "rag_storage" / "scan_spool"

    assert _scan_spool_base_dir(
        _spool_rag(str(tmp_path / "rag_storage"), "alpha")
    ) == (base / "named" / "alpha")
    assert _scan_spool_base_dir(_spool_rag(str(tmp_path / "rag_storage"), "beta")) == (
        base / "named" / "beta"
    )


def test_no_named_workspace_can_collide_with_the_unnamed_one(tmp_path, monkeypatch):
    monkeypatch.setattr(
        _document_routes.global_args, "scan_spool_dir", "", raising=False
    )
    working_dir = str(tmp_path / "rag_storage")

    unnamed = _scan_spool_base_dir(_spool_rag(working_dir, ""))
    for lookalike in ("_default", "unnamed", "named", "default"):
        assert _scan_spool_base_dir(_spool_rag(working_dir, lookalike)) != unnamed


def test_the_unnamed_and_a_lookalike_workspace_hold_spools_concurrently(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        _document_routes.global_args, "scan_spool_dir", "", raising=False
    )
    working_dir = str(tmp_path / "rag_storage")
    unnamed_dir = _scan_spool_base_dir(_spool_rag(working_dir, ""))
    named_dir = _scan_spool_base_dir(_spool_rag(working_dir, "_default"))

    with _ScanCandidateSpool(4, unnamed_dir) as unnamed:
        asyncio.run(unnamed.claim(tmp_path / "a.txt", "a.txt", 1))
        unnamed_db = unnamed_dir / "candidates.sqlite3"
        inode_before = unnamed_db.stat().st_ino

        with _ScanCandidateSpool(4, named_dir) as named:
            asyncio.run(named.claim(tmp_path / "b.txt", "b.txt", 1))

            assert unnamed_db.stat().st_ino == inode_before
            assert _drain(unnamed, 4) == [["a.txt"]]
            assert _drain(named, 4) == [["b.txt"]]

        assert unnamed_db.exists()


def test_a_traversing_workspace_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        _document_routes.global_args, "scan_spool_dir", "", raising=False
    )
    with pytest.raises(ValueError):
        _scan_spool_base_dir(_spool_rag(str(tmp_path / "rag_storage"), "../escape"))


def test_base_dir_is_none_only_without_any_working_dir(monkeypatch):
    monkeypatch.setattr(
        _document_routes.global_args, "scan_spool_dir", "", raising=False
    )
    assert _scan_spool_base_dir(_spool_rag(None)) is None


def test_spool_is_created_under_the_requested_dir_and_removed_on_close(tmp_path):
    base = tmp_path / "rag_storage" / "scan_spool" / "unnamed"
    spool = _ScanCandidateSpool(4, base)
    try:
        assert (base / "candidates.sqlite3").exists()
    finally:
        spool.close()

    assert list(base.glob("candidates.sqlite3*")) == []


def test_an_unusable_base_dir_fails_the_scan_instead_of_relocating(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("regular file", encoding="utf-8")

    with pytest.raises(RuntimeError, match="SCAN_SPOOL_DIR"):
        _ScanCandidateSpool(4, blocker / "spool")


def test_a_crashed_scans_leftover_spool_is_reclaimed_not_accumulated(tmp_path):
    base = tmp_path / "scan_spool" / "unnamed"
    base.mkdir(parents=True)
    leftover = base / "candidates.sqlite3"
    leftover.write_bytes(b"corpse of a killed scan")
    (base / "candidates.sqlite3-journal").write_bytes(b"stale journal")

    with _ScanCandidateSpool(4, base) as spool:
        assert list(base.glob("candidates.sqlite3*")) == [leftover]
        assert asyncio.run(spool.claim(tmp_path / "a.txt", "a.txt", 1)) is None
        assert _drain(spool, 4) == [["a.txt"]]

    assert list(base.glob("candidates.sqlite3*")) == []


def test_first_claim_wins_and_the_loser_is_told_the_winners_full_path(tmp_path):
    winner = _aged(tmp_path / "one" / "same.txt", 1_600_000_000)
    loser = _aged(tmp_path / "two" / "same.txt", 1_600_000_001)

    with _ScanCandidateSpool(4, tmp_path / "spool") as spool:
        assert asyncio.run(spool.claim(winner, "same.txt", 1)) is None
        claimer = asyncio.run(spool.claim(loser, "same.txt", 2))

        assert claimer == str(winner)
        assert claimer != str(loser)
        assert _drain(spool, 4) == [["same.txt"]]


def test_unknown_source_candidates_never_claim_each_others_slot(tmp_path):
    first = _aged(tmp_path / "a.txt", 1_600_000_000)
    second = _aged(tmp_path / "b.txt", 1_600_000_001)
    third = _aged(tmp_path / "c.txt", 1_600_000_002)

    with _ScanCandidateSpool(4, tmp_path / "spool") as spool:
        for index, path in enumerate((first, second, third), start=1):
            assert asyncio.run(spool.claim(path, UNKNOWN_FILE_SOURCE, index)) is None

        assert _drain(spool, 8) == [["a.txt", "b.txt", "c.txt"]]


def test_a_rejected_row_without_a_canonical_claim_raises(tmp_path):
    first = _aged(tmp_path / "a.txt", 1_600_000_000)
    second = _aged(tmp_path / "b.txt", 1_600_000_001)

    with _ScanCandidateSpool(4, tmp_path / "spool") as spool:
        assert asyncio.run(spool.claim(first, UNKNOWN_FILE_SOURCE, 7)) is None

        with pytest.raises(RuntimeError, match="without a canonical claim"):
            asyncio.run(spool.claim(second, UNKNOWN_FILE_SOURCE, 7))


def test_batches_are_globally_ordered_and_bounded_by_the_page_size(tmp_path):
    paths = [
        _aged(tmp_path / f"f{index}.txt", 1_600_000_000 + index) for index in range(5)
    ]
    arrival = [paths[3], paths[0], paths[4], paths[1], paths[2]]

    with _ScanCandidateSpool(2, tmp_path / "spool") as spool:
        for index, path in enumerate(arrival, start=1):
            asyncio.run(spool.claim(path, path.name, index))

        assert _drain(spool, 2) == [
            ["f0.txt", "f1.txt"],
            ["f2.txt", "f3.txt"],
            ["f4.txt"],
        ]


def test_an_unreadable_mtime_sorts_last_without_losing_the_candidate(tmp_path):
    readable = _aged(tmp_path / "middle.txt", 1_600_000_000)
    gone_z = tmp_path / "z-gone.txt"
    gone_a = tmp_path / "a-gone.txt"

    with _ScanCandidateSpool(4, tmp_path / "spool") as spool:
        for index, path in enumerate((gone_z, readable, gone_a), start=1):
            asyncio.run(spool.claim(path, path.name, index))

        assert _drain(spool, 8) == [["middle.txt", "a-gone.txt", "z-gone.txt"]]


def test_the_candidate_carries_the_size_discovery_stated(tmp_path):
    sized = _aged(tmp_path / "sized.txt", 1_600_000_000, body="0123456789")
    vanished = tmp_path / "vanished.txt"

    with _ScanCandidateSpool(4, tmp_path / "spool") as spool:
        asyncio.run(spool.claim(sized, "sized.txt", 1))
        asyncio.run(spool.claim(vanished, "vanished.txt", 2))

        batches = list(spool.ordered_batches(8))

        assert [(c.path.name, c.size) for c in batches[0]] == [
            ("sized.txt", 10),
            ("vanished.txt", 0),
        ]


def test_uncommitted_claims_are_visible_to_ordered_batches(tmp_path):
    path = _aged(tmp_path / "lonely.txt", 1_600_000_000)

    with _ScanCandidateSpool(1000, tmp_path / "spool") as spool:
        asyncio.run(spool.claim(path, "lonely.txt", 1))

        assert _drain(spool, 8) == [["lonely.txt"]]


# --------------------------------------------------------------------------- #
# Scan mtime ordering boundary (never forges persistent created_at)
# --------------------------------------------------------------------------- #


def _recording_rag() -> tuple[SimpleNamespace, dict]:
    captured: dict = {}

    async def fake_enqueue(content, **kwargs):
        captured.update(kwargs)
        return "track-xyz"

    async def fake_error(files, track_id):
        captured["error_files"] = files

    rag = SimpleNamespace(
        addon_params={},
        apipeline_enqueue_documents=fake_enqueue,
        apipeline_enqueue_error_documents=fake_error,
    )
    return rag, captured


def test_scan_enqueue_does_not_supply_created_at(tmp_path):
    rag, captured = _recording_rag()
    aged = _aged(tmp_path / "corpus.txt", _MTIME_EPOCH, body="hello world")

    ok, _track = asyncio.run(pipeline_enqueue_file(rag, aged, from_scan=True))

    assert ok is True
    assert "error_files" not in captured
    assert "created_at" not in captured


def test_upload_enqueue_leaves_created_at_to_the_pipeline(tmp_path):
    rag, captured = _recording_rag()
    aged = _aged(tmp_path / "corpus.txt", _MTIME_EPOCH, body="hello world")

    ok, _track = asyncio.run(pipeline_enqueue_file(rag, aged, from_scan=False))

    assert ok is True
    assert "created_at" not in captured


def test_scan_enqueue_survives_a_vanished_file(tmp_path):
    rag, captured = _recording_rag()

    ok, _track = asyncio.run(
        pipeline_enqueue_file(rag, tmp_path / "gone.txt", from_scan=True)
    )

    assert ok is True
    assert "created_at" not in captured


# --------------------------------------------------------------------------- #
# Streaming scan batches through run_scanning_process
# --------------------------------------------------------------------------- #


class _NoRowsDocStatus:
    """Every canonical source key resolves to SourceAbsent (all files are new)."""

    def __init__(self):
        self.lookups = []

    async def resolve_doc_source_strict(self, canonical_source_key):
        self.lookups.append(canonical_source_key)
        return SourceAbsent()

    async def get_full_docs_by_ids(self, doc_ids, *, strict=False):
        return {}


class _StreamRag:
    def __init__(self):
        self.workspace = f"scanstream-{uuid4().hex[:8]}"
        self.doc_status = _NoRowsDocStatus()
        self.full_docs = SimpleNamespace(
            get_by_id=self._missing,
            get_by_id_strict=self._missing,
            supports_strict_point_reads=True,
        )
        self.process_calls = 0

    async def _missing(self, _doc_id):
        return None

    async def arollback_failed_custom_chunk_patches(self, **_kwargs):
        return {"rolled_back": [], "failed": []}

    async def apipeline_process_enqueue_documents(self):
        self.process_calls += 1


def _batch_size(monkeypatch, size: int) -> None:
    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(scan_enqueue_batch_size=size)
    )


def test_candidates_are_spooled_then_written_in_global_mtime_order(
    tmp_path, monkeypatch
):
    async def _run():
        rag = _StreamRag()
        doc_manager = DocumentManager(str(tmp_path))
        _batch_size(monkeypatch, 2)

        by_age = [tmp_path / f"age-{index}.txt" for index in range(5)]
        for index, file_path in enumerate(by_age):
            file_path.write_text(str(index), encoding="utf-8")
            stamp = 1_600_000_000 + index
            os.utime(file_path, (stamp, stamp))
        files = [by_age[4], by_age[1], by_age[3], by_age[0], by_age[2]]
        events: list[tuple[str, object]] = []

        def _iter_new_files():
            for file_path in files:
                events.append(("yield", file_path.name))
                yield file_path

        doc_manager.iter_new_files = _iter_new_files

        async def _capture_batch(_rag, candidates, _track_id):
            events.append(("flush", [c.path.name for c in candidates]))
            return len(candidates)

        monkeypatch.setattr(
            _document_routes, "pipeline_enqueue_scan_batch", _capture_batch
        )

        await run_scanning_process(rag, doc_manager, "track-stream")

        flushes = [event for event in events if event[0] == "flush"]
        assert [len(names) for _, names in flushes] == [2, 2, 1]
        assert [name for _, names in flushes for name in names] == [
            path.name for path in by_age
        ]
        first_flush = events.index(flushes[0])
        last_yield = max(
            index for index, event in enumerate(events) if event[0] == "yield"
        )
        assert first_flush > last_yield
        assert rag.process_calls == 1

    asyncio.run(_run())


def test_first_scan_wide_claim_wins_and_the_alias_is_archived(tmp_path, monkeypatch):
    async def _run():
        rag = _StreamRag()
        doc_manager = DocumentManager(str(tmp_path))
        _batch_size(monkeypatch, 1)

        plain = doc_manager.input_dir / "same.txt"
        hinted = doc_manager.input_dir / "same.[native].txt"
        plain.write_text("plain", encoding="utf-8")
        hinted.write_text("hinted", encoding="utf-8")
        ordered = [plain, hinted]
        doc_manager.iter_new_files = lambda: iter(ordered)

        batched: list[Path] = []

        async def _capture_batch(_rag, candidates, _track_id):
            batched.extend(c.path for c in candidates)
            return len(candidates)

        monkeypatch.setattr(
            _document_routes, "pipeline_enqueue_scan_batch", _capture_batch
        )

        await run_scanning_process(rag, doc_manager, "track-claim")

        assert batched == [plain]
        assert plain.exists()
        assert not hinted.exists()
        archived = {
            path.name
            for path in (doc_manager.input_dir / PARSED_DIR_NAME).iterdir()
            if path.is_file()
        }
        assert archived == {hinted.name}
        assert rag.doc_status.lookups == ["same.txt", "same.txt"]

    asyncio.run(_run())


def test_unreadable_mtime_sorts_after_readable_candidates(tmp_path, monkeypatch):
    async def _run():
        rag = _StreamRag()
        doc_manager = DocumentManager(str(tmp_path))
        _batch_size(monkeypatch, 2)

        readable = tmp_path / "middle.txt"
        readable.write_text("body", encoding="utf-8")
        missing_b = tmp_path / "z-gone.txt"
        missing_a = tmp_path / "a-gone.txt"
        doc_manager.iter_new_files = lambda: iter([missing_b, readable, missing_a])

        ordered: list[str] = []

        async def _capture_batch(_rag, candidates, _track_id):
            ordered.extend(c.path.name for c in candidates)
            return len(candidates)

        monkeypatch.setattr(
            _document_routes, "pipeline_enqueue_scan_batch", _capture_batch
        )

        await run_scanning_process(rag, doc_manager, "track-missing-mtime")

        assert ordered == ["middle.txt", "a-gone.txt", "z-gone.txt"]

    asyncio.run(_run())


def test_an_unusable_spool_directory_fails_the_scan_closed(
    tmp_path, monkeypatch, caplog
):
    async def _run():
        rag = _StreamRag()
        doc_manager = DocumentManager(str(tmp_path))
        blocker = tmp_path / "blocked-by-a-regular-file"
        blocker.write_text("not a directory", encoding="utf-8")
        monkeypatch.setattr(
            _document_routes,
            "global_args",
            SimpleNamespace(
                scan_enqueue_batch_size=2,
                scan_spool_dir=str(blocker / "spool"),
            ),
        )

        candidate = doc_manager.input_dir / "candidate.txt"
        candidate.write_text("body", encoding="utf-8")

        batches: list[object] = []

        async def _capture_batch(_rag, candidates, _track_id):
            batches.append(candidates)
            return len(candidates)

        monkeypatch.setattr(
            _document_routes, "pipeline_enqueue_scan_batch", _capture_batch
        )

        lightrag_logger = importlib.import_module("lightrag.utils").logger
        with capture_lightrag_logs(caplog):
            with caplog.at_level("ERROR", logger=lightrag_logger.name):
                await run_scanning_process(rag, doc_manager, "track-bad-spool")

        assert batches == []
        assert candidate.exists()
        assert not (doc_manager.input_dir / PARSED_DIR_NAME).exists()
        assert "SCAN_SPOOL_DIR" in caplog.text

    asyncio.run(_run())


def test_iter_new_files_is_lazy_and_skips_directories(tmp_path, monkeypatch):
    doc_manager = DocumentManager(str(tmp_path))
    (doc_manager.input_dir / "a.txt").write_text("a", encoding="utf-8")
    (doc_manager.input_dir / "b.txt").write_text("b", encoding="utf-8")
    (doc_manager.input_dir / "trap.txt").mkdir()
    (doc_manager.input_dir / "ignored.bin").write_text("x", encoding="utf-8")

    class _OneThenTrap:
        """A scandir iterator that fails if discovery reads ahead."""

        def __init__(self, directory):
            self._first = SimpleNamespace(path=str(Path(directory) / "a.txt"))
            self._yielded = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            return self

        def __next__(self):
            if self._yielded:
                raise AssertionError("directory scan read ahead before yielding")
            self._yielded = True
            return self._first

    with monkeypatch.context() as patch:
        patch.setattr(_document_routes.os, "scandir", _OneThenTrap)
        stream = doc_manager.iter_new_files()
        assert not isinstance(stream, list)
        first = next(stream)
        assert first.name == "a.txt"
        stream.close()

    names = {path.name for path in doc_manager.iter_new_files()}
    assert names == {"a.txt", "b.txt"}


def test_batch_size_falls_back_when_unconfigured(monkeypatch):
    from lightrag.constants import DEFAULT_SCAN_ENQUEUE_BATCH_SIZE

    monkeypatch.setattr(_document_routes, "global_args", SimpleNamespace())
    assert (
        _document_routes._scan_enqueue_batch_size() == DEFAULT_SCAN_ENQUEUE_BATCH_SIZE
    )

    _batch_size(monkeypatch, 0)
    assert (
        _document_routes._scan_enqueue_batch_size() == DEFAULT_SCAN_ENQUEUE_BATCH_SIZE
    )

    _batch_size(monkeypatch, 7)
    assert _document_routes._scan_enqueue_batch_size() == 7
