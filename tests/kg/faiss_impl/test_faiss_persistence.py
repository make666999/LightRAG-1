"""Atomic-write and off-event-loop persistence tests for ``FaissVectorDBStorage``.

Covers:
- per-file tmp + rename atomicity and crash preservation for ``.index`` and
  ``.meta.json``;
- event-loop liveness, pre-offload metadata snapshotting, uncancellable
  bookkeeping on cancelled commits, and non-fatal notification failure handling.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from unittest.mock import patch

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from lightrag.kg import faiss_impl  # noqa: E402
from lightrag.kg.faiss_impl import FaissVectorDBStorage  # noqa: E402
from lightrag.kg.shared_storage import (  # noqa: E402
    finalize_share_data,
    initialize_share_data,
)
from lightrag.utils import EmbeddingFunc  # noqa: E402
from tests.kg._file_backed_vector_contracts import BLOCK_SECONDS, Heartbeat  # noqa: E402

pytestmark = pytest.mark.offline


def _make_storage(tmp_path, namespace: str = "vectors") -> FaissVectorDBStorage:
    def _unused(*_args, **_kwargs):  # pragma: no cover — never called here
        raise AssertionError("embedding_func must not be invoked by save path")

    return FaissVectorDBStorage(
        namespace=namespace,
        workspace="",
        global_config={
            "working_dir": str(tmp_path),
            "embedding_batch_num": 1,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        embedding_func=EmbeddingFunc(embedding_dim=4, func=_unused),
    )


async def _noop() -> None:
    """Post-commit hook stand-in for direct ``_save_faiss_index`` tests."""
    return None


def _seed(storage: FaissVectorDBStorage, marker: str) -> None:
    storage._index.add(np.ones((1, 4), dtype=np.float32))
    storage._id_to_meta = {
        storage._index.ntotal - 1: {"__id__": marker, "content": marker}
    }


# ---------------------------------------------------------------------------
# Atomic per-file saves
# ---------------------------------------------------------------------------


async def test_save_faiss_index_publishes_both_files_cleanly(tmp_path):
    storage = _make_storage(tmp_path)
    _seed(storage, "v1")
    await storage._save_faiss_index(_noop)

    assert os.path.exists(storage._faiss_index_file)
    assert os.path.exists(storage._meta_file)
    with open(storage._meta_file, encoding="utf-8") as fh:
        meta = json.load(fh)
    assert next(iter(meta.values()))["__id__"] == "v1"
    leftovers = [p for p in os.listdir(tmp_path) if ".tmp." in p]
    assert leftovers == [], f"Unexpected tmp residue: {leftovers}"


async def test_save_faiss_index_replace_crash_preserves_prior_index(tmp_path):
    storage = _make_storage(tmp_path)
    _seed(storage, "v1")
    await storage._save_faiss_index(_noop)
    assert os.path.exists(storage._faiss_index_file)

    storage._index.reset()
    _seed(storage, "v2")
    with patch(
        "lightrag.file_atomic.os.replace",
        side_effect=OSError("simulated crash"),
    ):
        with pytest.raises(OSError, match="simulated crash"):
            await storage._save_faiss_index(_noop)

    reloaded = faiss.read_index(storage._faiss_index_file)
    assert reloaded.ntotal == 1
    leftovers = [p for p in os.listdir(tmp_path) if ".tmp." in p]
    assert leftovers == [], f"Python-exception path must clean tmp, got {leftovers}"


async def test_save_faiss_meta_write_failure_preserves_prior_meta(tmp_path):
    storage = _make_storage(tmp_path)
    _seed(storage, "v1")
    await storage._save_faiss_index(_noop)
    with open(storage._meta_file, encoding="utf-8") as fh:
        assert json.load(fh)

    real_dump = json.dump
    seen: list[bool] = []

    def explode_on_second_dump(*args, **kwargs):
        seen.append(True)
        raise RuntimeError("simulated meta failure")

    storage._index.reset()
    _seed(storage, "v2")
    with patch("lightrag.kg.faiss_impl.json.dump", side_effect=explode_on_second_dump):
        with pytest.raises(RuntimeError, match="simulated meta failure"):
            await storage._save_faiss_index(_noop)
    assert seen, "patched json.dump must have been invoked"

    with open(storage._meta_file, encoding="utf-8") as fh:
        meta = json.load(fh)
    assert any(entry["__id__"] == "v1" for entry in meta.values())
    leftovers = [p for p in os.listdir(tmp_path) if ".tmp." in p]
    assert leftovers == [], f"meta-write failure must clean tmp, got {leftovers}"
    assert json.dump is real_dump


# ---------------------------------------------------------------------------
# Off-event-loop persistence and cancellation safety
# ---------------------------------------------------------------------------


async def test_save_keeps_the_event_loop_running(tmp_path, monkeypatch):
    storage = _make_storage(tmp_path)
    _seed(storage, "v1")

    real_write_index = faiss_impl.faiss.write_index
    observed: list[tuple[int, int]] = []

    async with Heartbeat() as heartbeat:

        def slow_write_index(index, path):
            before = heartbeat.beats
            time.sleep(BLOCK_SECONDS)
            observed.append((before, heartbeat.beats))
            return real_write_index(index, path)

        monkeypatch.setattr(faiss_impl.faiss, "write_index", slow_write_index)
        await storage._save_faiss_index(_noop)

    assert observed, "patched write_index was never called"
    before, after = observed[0]
    assert after > before, f"event loop stalled during save ({before} -> {after})"
    assert faiss.read_index(storage._faiss_index_file).ntotal == 1
    with open(storage._meta_file, encoding="utf-8") as fh:
        meta = json.load(fh)
    assert next(iter(meta.values()))["__id__"] == "v1"


async def test_metadata_snapshot_is_taken_before_the_offload(tmp_path, monkeypatch):
    storage = _make_storage(tmp_path)
    _seed(storage, "v1")

    real_write_index = faiss_impl.faiss.write_index
    inside_write = threading.Event()
    may_finish = threading.Event()

    def parked_write_index(index, path):
        inside_write.set()
        assert may_finish.wait(timeout=5), "writer was never released"
        return real_write_index(index, path)

    monkeypatch.setattr(faiss_impl.faiss, "write_index", parked_write_index)

    save = asyncio.create_task(storage._save_faiss_index(_noop))
    while not inside_write.is_set():
        await asyncio.sleep(0.01)

    storage._id_to_meta[999] = {"__id__": "late", "content": "late"}
    may_finish.set()
    await save

    with open(storage._meta_file, encoding="utf-8") as fh:
        meta = json.load(fh)
    assert "999" not in meta, "the worker serialized the live dict, not a snapshot"
    assert [entry["__id__"] for entry in meta.values()] == ["v1"]


async def test_cancelled_save_still_completes_both_files(tmp_path, monkeypatch):
    storage = _make_storage(tmp_path)
    _seed(storage, "v1")

    real_write_index = faiss_impl.faiss.write_index
    inside_write = threading.Event()
    may_finish = threading.Event()

    def parked_write_index(index, path):
        inside_write.set()
        assert may_finish.wait(timeout=5), "writer was never released"
        return real_write_index(index, path)

    monkeypatch.setattr(faiss_impl.faiss, "write_index", parked_write_index)

    save = asyncio.create_task(storage._save_faiss_index(_noop))
    while not inside_write.is_set():
        await asyncio.sleep(0.01)

    save.cancel()
    for _ in range(10):
        await asyncio.sleep(0)
    assert not save.done(), "caller returned while the worker was still writing"

    may_finish.set()
    with pytest.raises(asyncio.CancelledError):
        await save

    assert os.path.exists(storage._faiss_index_file)
    assert os.path.exists(storage._meta_file)
    assert faiss.read_index(storage._faiss_index_file).ntotal == 1
    leftovers = [p for p in os.listdir(tmp_path) if ".tmp." in p]
    assert leftovers == [], f"unexpected tmp residue: {leftovers}"


async def test_cancelled_commit_still_notifies_and_retires_the_redo_logs(
    tmp_path, monkeypatch
):
    finalize_share_data()
    initialize_share_data()
    try:
        storage = _make_storage(tmp_path)
        await storage.initialize()
        _seed(storage, "v1")
        storage._index_dirty = True

        flagged: list[str] = []
        real_write_index = faiss_impl.faiss.write_index
        inside_write = threading.Event()
        may_finish = threading.Event()

        async def spy_set_all_update_flags(namespace, workspace=None):
            flagged.append(namespace)

        def parked_write_index(index, path):
            inside_write.set()
            assert may_finish.wait(timeout=5), "writer was never released"
            return real_write_index(index, path)

        monkeypatch.setattr(
            faiss_impl, "set_all_update_flags", spy_set_all_update_flags
        )
        monkeypatch.setattr(faiss_impl.faiss, "write_index", parked_write_index)

        commit = asyncio.create_task(storage.index_done_callback())
        while not inside_write.is_set():
            await asyncio.sleep(0.01)

        commit.cancel()
        may_finish.set()

        with pytest.raises(asyncio.CancelledError):
            await commit

        assert flagged == ["vectors"]
        assert storage._index_dirty is False
        assert storage._unsaved_upserts == {}
    finally:
        finalize_share_data()


async def test_a_failed_notification_is_not_reported_as_a_failed_save(
    tmp_path, monkeypatch, lightrag_caplog
):
    finalize_share_data()
    initialize_share_data()
    try:
        storage = _make_storage(tmp_path)
        await storage.initialize()
        _seed(storage, "v1")
        storage._index_dirty = True

        async def failing_set_all_update_flags(namespace, workspace=None):
            raise RuntimeError("shared-storage manager is down")

        monkeypatch.setattr(
            faiss_impl, "set_all_update_flags", failing_set_all_update_flags
        )

        with lightrag_caplog.at_level(logging.ERROR, logger="lightrag"):
            committed = await storage.index_done_callback()

        assert committed is True
        assert os.path.exists(storage._faiss_index_file)
        assert os.path.exists(storage._meta_file)
        assert storage._index_dirty is True
        assert any(
            "publishing that write failed" in record.getMessage()
            for record in lightrag_caplog.records
        ), f"the deferred publication was not logged: {lightrag_caplog.text}"
    finally:
        finalize_share_data()
