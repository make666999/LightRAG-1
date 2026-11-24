"""Off-event-loop persistence and publication-recovery tests for NanoVectorDBStorage.

Covers:
- event-loop liveness while saving off the loop;
- restoring ``client.storage_file`` and running post-commit bookkeeping on
  cancelled commits;
- non-fatal notification failure handling, redo-log retention/replay against
  an unnotified peer, and resilience against a broken log sink.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time

import pytest

nano_vectordb = pytest.importorskip("nano_vectordb")

import lightrag.kg.nano_vector_db_impl as nano_impl  # noqa: E402
from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage  # noqa: E402
from lightrag.kg.shared_storage import (  # noqa: E402
    finalize_share_data,
    initialize_share_data,
)
from lightrag.utils import EmbeddingFunc  # noqa: E402
from tests.kg._file_backed_vector_contracts import (  # noqa: E402
    BLOCK_SECONDS,
    DIM,
    CountingEmbed,
    Heartbeat,
)

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


async def _make_storage(tmp_path) -> NanoVectorDBStorage:
    storage = NanoVectorDBStorage(
        namespace="test_vectors",
        workspace="ws",
        global_config={
            "working_dir": str(tmp_path),
            "embedding_batch_num": 32,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=DIM, max_token_size=512, func=CountingEmbed()
        ),
        meta_fields={"content"},
    )
    await storage.initialize()
    return storage


async def _wait_for(predicate, *, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        assert loop.time() < deadline, "timed out waiting for condition"
        await asyncio.sleep(0.01)


async def test_commit_keeps_the_event_loop_running(tmp_path, monkeypatch):
    storage = await _make_storage(tmp_path)
    await storage.upsert({"id1": {"content": "alpha"}})

    observed = []
    real_atomic_write = nano_impl.atomic_write

    async with Heartbeat() as heartbeat:

        def blocking_atomic_write(path, write_fn, workspace):
            before = heartbeat.beats
            time.sleep(BLOCK_SECONDS)
            observed.append((before, heartbeat.beats))
            return real_atomic_write(path, write_fn, workspace)

        monkeypatch.setattr(nano_impl, "atomic_write", blocking_atomic_write)
        await storage.index_done_callback()

    assert observed, "the save never ran"
    before, after = observed[0]
    assert after > before, "event loop was blocked for the whole save"
    assert (await storage.get_by_id("id1"))["content"] == "alpha"


async def test_cancelled_commit_restores_storage_file(tmp_path):
    storage = await _make_storage(tmp_path)
    await storage.upsert({"id1": {"content": "alpha"}})

    entered = threading.Event()
    release = threading.Event()
    observed_during_write = []
    real_save = storage._client.save

    def blocking_save():
        observed_during_write.append(storage._client.storage_file)
        entered.set()
        assert release.wait(timeout=5), "write was never released"
        return real_save()

    storage._client.save = blocking_save

    commit = asyncio.create_task(storage.index_done_callback())
    await _wait_for(entered.is_set)

    commit.cancel()
    for _ in range(10):
        await asyncio.sleep(0)
    assert not commit.done(), "caller returned mid-save with storage_file swapped"

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await commit

    assert observed_during_write, "the write body never ran"
    assert observed_during_write[0] != storage._client_file_name
    assert storage._client.storage_file == storage._client_file_name


async def test_cancelled_commit_still_notifies_and_retires_the_redo_logs(
    tmp_path, monkeypatch
):
    storage = await _make_storage(tmp_path)
    await storage.upsert({"id1": {"content": "alpha"}})

    flagged: list[str] = []
    inside_save = threading.Event()
    may_finish = threading.Event()
    real_save = nano_vectordb.NanoVectorDB.save

    async def spy_set_all_update_flags(namespace, workspace=None):
        flagged.append(namespace)

    def parked_save(self):
        inside_save.set()
        assert may_finish.wait(timeout=5), "writer was never released"
        return real_save(self)

    monkeypatch.setattr(nano_impl, "set_all_update_flags", spy_set_all_update_flags)
    monkeypatch.setattr(nano_vectordb.NanoVectorDB, "save", parked_save)

    commit = asyncio.create_task(storage.index_done_callback())
    while not inside_save.is_set():
        await asyncio.sleep(0.01)

    commit.cancel()
    may_finish.set()

    with pytest.raises(asyncio.CancelledError):
        await commit

    assert flagged == ["test_vectors"]
    assert storage._client_dirty is False
    assert storage._unsaved_upserts == {}


async def test_a_failed_notification_is_not_reported_as_a_failed_save(
    tmp_path, monkeypatch, lightrag_caplog
):
    storage = await _make_storage(tmp_path)
    await storage.upsert({"id1": {"content": "alpha"}})

    async def failing_set_all_update_flags(namespace, workspace=None):
        raise RuntimeError("shared-storage manager is down")

    monkeypatch.setattr(nano_impl, "set_all_update_flags", failing_set_all_update_flags)

    with lightrag_caplog.at_level(logging.ERROR, logger="lightrag"):
        committed = await storage.index_done_callback()

    assert committed is True
    with open(storage._client_file_name, encoding="utf-8") as f:
        persisted = json.load(f)
    assert persisted["data"]
    assert storage._client_dirty is True
    assert set(storage._unsaved_upserts) == {"id1"}
    assert any(
        "publishing that write failed" in record.getMessage()
        for record in lightrag_caplog.records
    ), f"the deferred publication was not logged: {lightrag_caplog.text}"


async def test_rows_lost_to_an_unnotified_peer_are_replayed_back(tmp_path, monkeypatch):
    writer = await _make_storage(tmp_path)
    other = await _make_storage(tmp_path)

    await writer.upsert({"id1": {"content": "ours"}})

    async def failing_set_all_update_flags(namespace, workspace=None):
        raise RuntimeError("shared-storage manager is down")

    monkeypatch.setattr(nano_impl, "set_all_update_flags", failing_set_all_update_flags)
    assert await writer.index_done_callback() is True
    monkeypatch.undo()

    await other.upsert({"id2": {"content": "theirs"}})
    assert await other.index_done_callback() is True
    assert await other.get_by_id("id1") is None

    assert await writer.index_done_callback() is True

    with open(writer._client_file_name, encoding="utf-8") as f:
        persisted = json.load(f)
    ids = {row["__id__"] for row in persisted["data"]}
    assert ids == {"id1", "id2"}


async def test_a_broken_sink_cannot_turn_a_landed_save_into_a_failure(
    tmp_path, monkeypatch
):
    storage = await _make_storage(tmp_path)
    await storage.upsert({"id1": {"content": "alpha"}})

    async def failing_set_all_update_flags(namespace, workspace=None):
        raise RuntimeError("shared-storage manager is down")

    def log_boom(msg):
        raise RuntimeError("log sink boom")

    monkeypatch.setattr(nano_impl, "set_all_update_flags", failing_set_all_update_flags)
    monkeypatch.setattr(nano_impl.logger, "error", log_boom)

    assert await storage.index_done_callback() is True

    with open(storage._client_file_name, encoding="utf-8") as f:
        persisted = json.load(f)
    assert persisted["data"]
    assert set(storage._unsaved_upserts) == {"id1"}
