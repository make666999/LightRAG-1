"""NetworkXStorage persistence contracts: atomic writes, commit gate, dirty-discard backstop, and save recovery.

Pins the invariants in ``docs/design/NetworkXSingleWriterContract.md``:
- Atomic GraphML writes under concurrent threads and crash during ``os.replace``.
- Off-event-loop serialization guarded by ``_commit_gate`` (including mutators
  suspended in ``NamespaceLock.__aexit__``, waiter re-checks, error/cancellation
  restoration, event-loop rebinding, and post-write notification resilience).
- Dirty-graph discard backstop (``GraphMutationsDiscardedError`` on writer commit,
  innocent reader non-failure, single-process & multiprocess file-channel paths,
  exempt recovery/decline paths, and non-exempt fingerprint-colliding commits).
- ``index_done_callback`` raising on save failure and restoring/retrying in-memory
  graph coherence without leaking failed batches into subsequent commits.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import networkx as nx
import numpy as np
import pytest

from lightrag.exceptions import GraphMutationsDiscardedError
from lightrag.kg import file_fingerprint, networkx_impl, shared_storage
from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import EmbeddingFunc

pytestmark = pytest.mark.offline

# A blocked loop gives zero beats; a live one beats within one 5 ms GIL switch.
BLOCK_SECONDS = 0.05


@pytest.fixture(autouse=True)
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


@pytest.fixture
def multiprocess(monkeypatch):
    monkeypatch.setattr(file_fingerprint, "is_multiprocess_mode", lambda: True)


@pytest.fixture
def lost_notification(monkeypatch):
    async def _noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(networkx_impl, "set_all_update_flags", _noop)


async def _embed(texts):
    return np.random.rand(len(texts), 8)


def _make_storage(
    tmp_path,
    *,
    namespace: str = "test_graph",
    with_embedding: bool = True,
) -> NetworkXStorage:
    global_config: dict = {"working_dir": str(tmp_path)}
    embedding_func = None
    if with_embedding:
        global_config["embedding_batch_num"] = 10
        global_config["vector_db_storage_cls_kwargs"] = {
            "cosine_better_than_threshold": 0.5
        }
        embedding_func = EmbeddingFunc(
            embedding_dim=8, max_token_size=512, func=_embed
        )
    return NetworkXStorage(
        namespace=namespace,
        workspace="ws",
        global_config=global_config,
        embedding_func=embedding_func,
    )


async def _worker(
    tmp_path, *, namespace: str = "test_graph", with_embedding: bool = True
) -> NetworkXStorage:
    storage = _make_storage(
        tmp_path, namespace=namespace, with_embedding=with_embedding
    )
    await storage.initialize()
    return storage


class _Heartbeat:
    """Counts event-loop iterations to verify commits stay off the event loop."""

    def __init__(self):
        self.beats = 0
        self._stop = False
        self._task = None

    async def _run(self):
        while not self._stop:
            self.beats += 1
            await asyncio.sleep(0)

    def start(self):
        self._task = asyncio.create_task(self._run())
        return self

    async def stop(self):
        self._stop = True
        if self._task is not None:
            await self._task


async def _wait_for(predicate, *, timeout: float = 2.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        assert loop.time() < deadline, "timed out waiting for condition"
        await asyncio.sleep(0.01)


# ---------------------------------------------------------------------------
# Atomic GraphML writes
# ---------------------------------------------------------------------------


def test_write_nx_graph_concurrent_writers_no_race(tmp_path):
    dst = str(tmp_path / "graph_concurrent.graphml")
    barrier = threading.Barrier(5, timeout=10)

    def writer(tid: int) -> None:
        g = nx.Graph()
        g.add_node(f"node-{tid}", label=f"label-{tid}")
        barrier.wait()
        NetworkXStorage.write_nx_graph(g, dst, workspace=f"w{tid}")

    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(writer, range(5)))

    loaded = nx.read_graphml(dst)
    assert loaded.number_of_nodes() == 1
    leftovers = [p for p in os.listdir(tmp_path) if ".tmp." in p]
    assert leftovers == [], f"Unexpected tmp residue: {leftovers}"


def test_write_nx_graph_crash_preserves_prior_snapshot(tmp_path):
    dst = str(tmp_path / "graph_crash.graphml")

    v1 = nx.Graph()
    v1.add_node("v1-node")
    NetworkXStorage.write_nx_graph(v1, dst, workspace="crash")
    assert os.path.exists(dst)

    v2 = nx.Graph()
    v2.add_node("v2-node")
    with patch(
        "lightrag.file_atomic.os.replace",
        side_effect=OSError("simulated crash"),
    ):
        with pytest.raises(OSError, match="simulated crash"):
            NetworkXStorage.write_nx_graph(v2, dst, workspace="crash")

    reloaded = nx.read_graphml(dst)
    assert "v1-node" in reloaded.nodes
    assert "v2-node" not in reloaded.nodes
    leftovers = [p for p in os.listdir(tmp_path) if ".tmp." in p]
    assert leftovers == [], f"Python-exception path must clean tmp, got {leftovers}"


# ---------------------------------------------------------------------------
# Off-event-loop commit gate and publication bookkeeping
# ---------------------------------------------------------------------------


async def test_commit_keeps_the_event_loop_running(tmp_path, monkeypatch):
    storage = await _worker(
        tmp_path, namespace="chunk_entity_relation", with_embedding=False
    )
    await storage.upsert_node("n1", {"entity_id": "n1"})

    observed = []
    heartbeat = _Heartbeat().start()
    real_write = NetworkXStorage.write_nx_graph

    def blocking_write(graph, file_name, workspace="_"):
        before = heartbeat.beats
        time.sleep(BLOCK_SECONDS)
        observed.append((before, heartbeat.beats))
        return real_write(graph, file_name, workspace)

    monkeypatch.setattr(NetworkXStorage, "write_nx_graph", staticmethod(blocking_write))

    await storage.index_done_callback()
    await heartbeat.stop()

    assert observed, "write_nx_graph was never called"
    before, after = observed[0]
    assert after > before, "event loop was blocked for the whole write"
    assert await storage.has_node("n1")


async def test_a_mutator_suspended_in_lock_release_cannot_mutate_during_a_commit(
    tmp_path, monkeypatch
):
    storage = await _worker(
        tmp_path, namespace="chunk_entity_relation", with_embedding=False
    )
    await storage.upsert_node("n1", {"entity_id": "n1"})

    release_reached = asyncio.Event()
    let_release_finish = asyncio.Event()
    arm = {"on": False}
    real_aexit = shared_storage.NamespaceLock.__aexit__

    async def parking_aexit(self, *exc_info):
        result = await real_aexit(self, *exc_info)
        if arm["on"]:
            arm["on"] = False
            release_reached.set()
            await let_release_finish.wait()
        return result

    monkeypatch.setattr(shared_storage.NamespaceLock, "__aexit__", parking_aexit)

    async def suspended_mutator():
        arm["on"] = True
        await storage.upsert_node("n2", {"entity_id": "n2"})

    mutator = asyncio.create_task(suspended_mutator())
    await asyncio.wait_for(release_reached.wait(), timeout=2)

    entered = threading.Event()
    release_write = threading.Event()
    stable = {"ok": None}
    real_write = NetworkXStorage.write_nx_graph

    def blocking_write(graph, file_name, workspace="_"):
        seen = graph.number_of_nodes()
        entered.set()
        assert release_write.wait(timeout=5), "write was never released"
        stable["ok"] = graph.number_of_nodes() == seen
        return real_write(graph, file_name, workspace)

    monkeypatch.setattr(NetworkXStorage, "write_nx_graph", staticmethod(blocking_write))

    commit = asyncio.create_task(storage.index_done_callback())
    await _wait_for(entered.is_set)

    let_release_finish.set()
    for _ in range(20):
        await asyncio.sleep(0)
    assert not mutator.done(), "mutator was not held back by the gate"

    release_write.set()
    await commit
    await mutator

    assert stable["ok"] is True, "graph mutated while the writer was iterating it"
    assert await storage.has_node("n2"), "the mutation must still land afterwards"


async def test_gate_is_restored_when_the_write_raises(tmp_path, monkeypatch):
    storage = await _worker(
        tmp_path, namespace="chunk_entity_relation", with_embedding=False
    )
    await storage.upsert_node("n1", {"entity_id": "n1"})

    def boom(graph, file_name, workspace="_"):
        raise OSError("save boom")

    monkeypatch.setattr(NetworkXStorage, "write_nx_graph", staticmethod(boom))

    with pytest.raises(OSError, match="save boom"):
        await storage.index_done_callback()

    assert storage._gate().is_set()
    await asyncio.wait_for(storage.upsert_node("n2", {"entity_id": "n2"}), timeout=2)


async def test_gate_is_restored_when_the_committer_is_cancelled(tmp_path, monkeypatch):
    storage = await _worker(
        tmp_path, namespace="chunk_entity_relation", with_embedding=False
    )
    await storage.upsert_node("n1", {"entity_id": "n1"})

    entered = threading.Event()
    release_write = threading.Event()
    real_write = NetworkXStorage.write_nx_graph

    def blocking_write(graph, file_name, workspace="_"):
        entered.set()
        assert release_write.wait(timeout=5), "write was never released"
        return real_write(graph, file_name, workspace)

    monkeypatch.setattr(NetworkXStorage, "write_nx_graph", staticmethod(blocking_write))

    commit = asyncio.create_task(storage.index_done_callback())
    await _wait_for(entered.is_set)

    commit.cancel()
    for _ in range(10):
        await asyncio.sleep(0)
    assert not commit.done(), "committer returned while the writer was running"
    assert not storage._gate().is_set(), "gate must stay closed during the write"

    release_write.set()
    with pytest.raises(asyncio.CancelledError):
        await commit

    assert storage._gate().is_set()
    await asyncio.wait_for(storage.upsert_node("n2", {"entity_id": "n2"}), timeout=2)


async def test_a_woken_waiter_rechecks_the_gate_before_proceeding(tmp_path):
    storage = await _worker(
        tmp_path, namespace="chunk_entity_relation", with_embedding=False
    )
    await storage.upsert_node("n1", {"entity_id": "n1"})

    gate = storage._gate()
    gate.clear()

    mutator = asyncio.create_task(storage.upsert_node("n2", {"entity_id": "n2"}))
    for _ in range(20):
        await asyncio.sleep(0)
    assert not mutator.done(), "mutator did not stop at the closed gate"
    assert not storage._graph.has_node("n2")

    gate.set()
    gate.clear()
    for _ in range(20):
        await asyncio.sleep(0)
    assert not mutator.done(), "waiter slipped through after being woken"
    assert not storage._graph.has_node("n2")

    gate.set()
    await asyncio.wait_for(mutator, timeout=2)
    assert storage._graph.has_node("n2")


async def test_cancelled_commit_still_notifies_the_other_processes(
    tmp_path, monkeypatch
):
    storage = await _worker(
        tmp_path, namespace="chunk_entity_relation", with_embedding=False
    )
    await storage.upsert_node("A", {"entity_id": "A"})

    flagged: list[str] = []
    real_write = NetworkXStorage.write_nx_graph
    inside_write = threading.Event()
    may_finish = threading.Event()

    async def spy_set_all_update_flags(namespace, workspace=None):
        flagged.append(namespace)

    def parked_write(graph, path, workspace="_"):
        inside_write.set()
        assert may_finish.wait(timeout=5), "writer was never released"
        return real_write(graph, path, workspace)

    monkeypatch.setattr(
        "lightrag.kg.networkx_impl.set_all_update_flags", spy_set_all_update_flags
    )
    monkeypatch.setattr(NetworkXStorage, "write_nx_graph", staticmethod(parked_write))

    commit = asyncio.create_task(storage.index_done_callback())
    await _wait_for(inside_write.is_set)

    commit.cancel()
    may_finish.set()

    with pytest.raises(asyncio.CancelledError):
        await commit

    assert flagged == ["chunk_entity_relation"]
    await asyncio.wait_for(storage.upsert_node("B", {"entity_id": "B"}), timeout=2)


def test_gate_is_rebound_after_the_owning_loop_closes(tmp_path):
    """Deliberately sync: exercises one NetworkXStorage across two real event loops."""
    initialize_share_data()
    try:
        storage = None

        async def first_loop():
            nonlocal storage
            storage = await _worker(
                tmp_path, namespace="chunk_entity_relation", with_embedding=False
            )
            await storage.upsert_node("A", {"entity_id": "A"})
            gate = storage._gate()
            gate.clear()
            waiter = asyncio.create_task(storage._get_graph())
            await _wait_for(lambda: not waiter.done())
            for _ in range(5):
                await asyncio.sleep(0)
            gate.set()
            await waiter
            assert storage._commit_gate_loop is asyncio.get_running_loop()

        asyncio.run(first_loop())
        bound_to = storage._commit_gate

        async def second_loop():
            await storage.index_done_callback()
            await asyncio.wait_for(
                storage.upsert_node("B", {"entity_id": "B"}), timeout=2
            )
            assert storage._commit_gate is not bound_to, "gate was not rebound"
            assert storage._commit_gate_loop is asyncio.get_running_loop()

        asyncio.run(second_loop())
    finally:
        finalize_share_data()


async def test_a_failed_notification_is_not_reported_as_a_failed_write(
    tmp_path, monkeypatch, lightrag_caplog
):
    storage = await _worker(
        tmp_path, namespace="chunk_entity_relation", with_embedding=False
    )
    await storage.upsert_node("A", {"entity_id": "A"})

    async def failing_set_all_update_flags(namespace, workspace=None):
        raise RuntimeError("shared-storage manager is down")

    monkeypatch.setattr(
        "lightrag.kg.networkx_impl.set_all_update_flags",
        failing_set_all_update_flags,
    )

    with lightrag_caplog.at_level(logging.ERROR, logger="lightrag"):
        committed = await storage.index_done_callback()

    assert committed is True
    persisted = NetworkXStorage.load_nx_graph(storage._graphml_xml_file)
    assert persisted is not None and persisted.has_node("A")
    assert any(
        "publishing that write failed" in record.getMessage()
        for record in lightrag_caplog.records
    )


async def test_a_failed_writer_flag_reset_is_not_reported_as_a_failed_write(
    tmp_path, monkeypatch, lightrag_caplog
):
    storage = await _worker(
        tmp_path, namespace="chunk_entity_relation", with_embedding=False
    )
    await storage.upsert_node("A", {"entity_id": "A"})

    class _FlagOnADeadManager:
        """ValueProxy double that fails when the writer clears its own flag."""

        @property
        def value(self):
            return False

        @value.setter
        def value(self, _new):
            raise RuntimeError("manager connection refused")

    monkeypatch.setattr(storage, "storage_updated", _FlagOnADeadManager())

    with lightrag_caplog.at_level(logging.ERROR, logger="lightrag"):
        committed = await storage.index_done_callback()

    assert committed is True
    persisted = NetworkXStorage.load_nx_graph(storage._graphml_xml_file)
    assert persisted is not None and persisted.has_node("A")
    assert any(
        "publishing that write failed" in record.getMessage()
        for record in lightrag_caplog.records
    )


async def test_a_broken_sink_cannot_turn_a_landed_commit_into_a_failure(
    tmp_path, monkeypatch
):
    storage = await _worker(
        tmp_path, namespace="chunk_entity_relation", with_embedding=False
    )
    await storage.upsert_node("A", {"entity_id": "A"})

    async def failing_set_all_update_flags(namespace, workspace=None):
        raise RuntimeError("shared-storage manager is down")

    def log_boom(msg):
        raise RuntimeError("log sink boom")

    monkeypatch.setattr(
        "lightrag.kg.networkx_impl.set_all_update_flags",
        failing_set_all_update_flags,
    )
    monkeypatch.setattr("lightrag.kg.networkx_impl.logger.error", log_boom)

    assert await storage.index_done_callback() is True
    persisted = NetworkXStorage.load_nx_graph(storage._graphml_xml_file)
    assert persisted is not None and persisted.has_node("A")
    assert await storage.has_node("A")


# ---------------------------------------------------------------------------
# Dirty-graph discard backstop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_writer_commit_raises_and_reader_does_not_single_process(tmp_path):
    assert file_fingerprint.fence_enabled() is False
    worker_a = await _worker(tmp_path)
    worker_b = await _worker(tmp_path)
    try:
        await worker_b.upsert_node("from_b", {"entity_id": "from_b"})
        assert worker_b._graph_dirty is True

        await worker_a.upsert_node("from_a", {"entity_id": "from_a"})
        assert await worker_a.index_done_callback() is True
        assert worker_a._graph_dirty is False
        assert worker_b.storage_updated.value is True

        assert await worker_b.has_node("from_a") is True
        assert await worker_b.has_node("from_b") is False
        assert worker_b._dirty_discard_pending is True
        assert worker_b._graph_dirty is False

        with pytest.raises(GraphMutationsDiscardedError):
            await worker_b.index_done_callback()

        assert worker_b._dirty_discard_pending is False
        assert await worker_b.index_done_callback() is True

        on_disk = NetworkXStorage.load_nx_graph(worker_b._graphml_xml_file)
        assert on_disk.has_node("from_a")
        assert not on_disk.has_node("from_b")
    finally:
        await worker_a.finalize()
        await worker_b.finalize()


@pytest.mark.asyncio
async def test_concurrent_reader_takes_no_error_while_writer_is_refused(tmp_path):
    worker_a = await _worker(tmp_path)
    worker_b = await _worker(tmp_path)
    try:
        parked = asyncio.Event()
        resume = asyncio.Event()

        async def writer():
            await worker_b.upsert_node("step_1", {"entity_id": "step_1"})
            parked.set()
            await resume.wait()
            await worker_b.upsert_node("step_2", {"entity_id": "step_2"})
            return await worker_b.index_done_callback()

        writer_task = asyncio.create_task(writer())
        await parked.wait()

        await worker_a.upsert_node("from_a", {"entity_id": "from_a"})
        assert await worker_a.index_done_callback() is True

        assert await worker_b.get_node("from_a") == {"entity_id": "from_a"}
        assert worker_b._dirty_discard_pending is True

        resume.set()
        with pytest.raises(GraphMutationsDiscardedError):
            await writer_task

        assert worker_b._recovery_reload_pending is True
        assert await worker_b.has_node("step_2") is False
        assert worker_b._recovery_reload_pending is False
        assert await worker_b.index_done_callback() is True
        on_disk = NetworkXStorage.load_nx_graph(worker_b._graphml_xml_file)
        assert not on_disk.has_node("step_1") and not on_disk.has_node("step_2")
    finally:
        await worker_a.finalize()
        await worker_b.finalize()


@pytest.mark.asyncio
async def test_reachable_through_the_file_channel_under_multiprocess(
    tmp_path, multiprocess, lost_notification
):
    worker_a = await _worker(tmp_path)
    worker_b = await _worker(tmp_path)
    try:
        await worker_b.upsert_node("from_b", {"entity_id": "from_b"})
        await worker_a.upsert_node("from_a", {"entity_id": "from_a"})
        assert await worker_a.index_done_callback() is True
        assert worker_b.storage_updated.value is False

        assert await worker_b.has_node("from_a") is True
        assert worker_b._dirty_discard_pending is True
        with pytest.raises(GraphMutationsDiscardedError):
            await worker_b.index_done_callback()
        assert worker_b._dirty_discard_pending is False
        assert await worker_b.index_done_callback() is True
    finally:
        await worker_a.finalize()
        await worker_b.finalize()


@pytest.mark.asyncio
async def test_decline_path_does_not_arm_the_backstop(
    tmp_path, multiprocess, lost_notification
):
    worker_a = await _worker(tmp_path)
    worker_b = await _worker(tmp_path)
    try:
        await worker_b.upsert_node("from_b", {"entity_id": "from_b"})
        await worker_a.upsert_node("from_a", {"entity_id": "from_a"})
        assert await worker_a.index_done_callback() is True

        assert await worker_b.index_done_callback() is False
        assert worker_b._dirty_discard_pending is False
        assert worker_b._graph_dirty is False
        assert await worker_b.index_done_callback() is True
    finally:
        await worker_a.finalize()
        await worker_b.finalize()


@pytest.mark.asyncio
async def test_recovery_reload_discards_silently(tmp_path):
    worker = await _worker(tmp_path)
    try:
        await worker.upsert_node("durable", {"entity_id": "durable"})
        assert await worker.index_done_callback() is True

        await worker.upsert_node("failed_op", {"entity_id": "failed_op"})
        assert worker._graph_dirty is True
        worker._recovery_reload_pending = True

        assert await worker.has_node("failed_op") is False
        assert worker._dirty_discard_pending is False
        assert worker._recovery_reload_pending is False
        assert await worker.index_done_callback() is True
    finally:
        await worker.finalize()


@pytest.mark.asyncio
async def test_a_notification_matching_the_loaded_fingerprint_still_arms(
    tmp_path, multiprocess
):
    worker = await _worker(tmp_path)
    try:
        await worker.upsert_node("durable", {"entity_id": "durable"})
        assert await worker.index_done_callback() is True

        await worker.upsert_node("unpublished", {"entity_id": "unpublished"})
        worker.storage_updated.value = True

        assert await worker.has_node("unpublished") is False
        assert worker._dirty_discard_pending is True
        with pytest.raises(GraphMutationsDiscardedError):
            await worker.index_done_callback()
        assert worker._dirty_discard_pending is False
        assert await worker.index_done_callback() is True
    finally:
        await worker.finalize()


@pytest.mark.asyncio
async def test_a_fingerprint_colliding_peer_commit_is_not_lost_silently(
    tmp_path, multiprocess, monkeypatch
):
    worker_a = await _worker(tmp_path)
    worker_b = await _worker(tmp_path)
    try:
        await worker_b.upsert_node("from_b", {"entity_id": "from_b"})
        await worker_a.upsert_node("from_a", {"entity_id": "from_a"})
        assert await worker_a.index_done_callback() is True

        collided = worker_b._loaded_fingerprint
        monkeypatch.setattr(type(worker_b), "_stat_fingerprint", lambda self: collided)
        assert worker_b._peer_commit_detected(collided) is False
        assert worker_b.storage_updated.value is True

        assert await worker_b.has_node("from_a") is True
        assert await worker_b.has_node("from_b") is False
        assert worker_b._dirty_discard_pending is True
        with pytest.raises(GraphMutationsDiscardedError):
            await worker_b.index_done_callback()
    finally:
        await worker_a.finalize()
        await worker_b.finalize()


@pytest.mark.asyncio
async def test_every_mutator_marks_the_graph_dirty_and_commit_clears_it(tmp_path):
    worker = await _worker(tmp_path)
    try:
        assert worker._graph_dirty is False
        mutations = [
            lambda: worker.upsert_node("n1", {"entity_id": "n1"}),
            lambda: worker.upsert_edge("n1", "n2", {"weight": "1.0"}),
            lambda: worker.upsert_nodes_batch([("n3", {"entity_id": "n3"})]),
            lambda: worker.upsert_edges_batch([("n3", "n1", {"weight": "1.0"})]),
            lambda: worker.remove_edges([("n3", "n1")]),
            lambda: worker.remove_nodes(["n3"]),
            lambda: worker.delete_node("n2"),
        ]
        for mutate in mutations:
            await mutate()
            assert worker._graph_dirty is True
            assert await worker.index_done_callback() is True
            assert worker._graph_dirty is False

        await worker.delete_node("never_there")
        await worker.remove_nodes(["never_there"])
        await worker.remove_edges([("never", "there")])
        assert worker._graph_dirty is False
    finally:
        await worker.finalize()


@pytest.mark.asyncio
async def test_drop_clears_both_backstop_records(tmp_path):
    worker_a = await _worker(tmp_path)
    worker_b = await _worker(tmp_path)
    try:
        await worker_b.upsert_node("from_b", {"entity_id": "from_b"})
        await worker_a.upsert_node("from_a", {"entity_id": "from_a"})
        assert await worker_a.index_done_callback() is True
        await worker_b.has_node("from_a")
        assert worker_b._dirty_discard_pending is True

        assert (await worker_b.drop())["status"] == "success"
        assert worker_b._dirty_discard_pending is False
        assert worker_b._graph_dirty is False
        assert await worker_b.index_done_callback() is True
    finally:
        await worker_a.finalize()
        await worker_b.finalize()


# ---------------------------------------------------------------------------
# index_done_callback save-failure handling and in-memory restoration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_index_done_callback_raises_on_save_failure(tmp_path, monkeypatch):
    storage = await _worker(tmp_path)
    try:
        await storage.upsert_node("n1", {"entity_id": "n1", "description": "x"})

        def boom(graph, file_name, workspace):
            raise OSError("save boom")

        monkeypatch.setattr(NetworkXStorage, "write_nx_graph", staticmethod(boom))

        with pytest.raises(OSError, match="save boom"):
            await storage.index_done_callback()
    finally:
        await storage.finalize()


@pytest.mark.asyncio
async def test_failed_save_restores_the_in_memory_graph(tmp_path, monkeypatch):
    storage = await _worker(tmp_path)
    try:
        await storage.upsert_node("n1", {"entity_id": "n1", "description": "x"})
        await storage.index_done_callback()
        await storage.delete_node("n1")

        def boom(graph, file_name, workspace):
            raise OSError("save boom")

        monkeypatch.setattr(NetworkXStorage, "write_nx_graph", staticmethod(boom))
        with pytest.raises(OSError, match="save boom"):
            await storage.index_done_callback()

        assert await storage.has_node("n1")
        persisted = NetworkXStorage.load_nx_graph(storage._graphml_xml_file)
        assert persisted is not None and persisted.has_node("n1")
    finally:
        await storage.finalize()


@pytest.mark.asyncio
async def test_failed_save_retries_restore_when_the_first_reload_fails(
    tmp_path, monkeypatch
):
    storage = await _worker(tmp_path)
    try:
        await storage.upsert_node("n1", {"entity_id": "n1", "description": "x"})
        await storage.index_done_callback()
        await storage.delete_node("n1")

        original_write = NetworkXStorage.write_nx_graph
        original_load = NetworkXStorage.load_nx_graph
        reload_attempts = 0

        def save_boom(graph, file_name, workspace):
            raise OSError("save boom")

        def first_reload_boom(file_name):
            nonlocal reload_attempts
            reload_attempts += 1
            if reload_attempts == 1:
                raise OSError("reload boom")
            return original_load(file_name)

        monkeypatch.setattr(NetworkXStorage, "write_nx_graph", staticmethod(save_boom))
        monkeypatch.setattr(
            NetworkXStorage, "load_nx_graph", staticmethod(first_reload_boom)
        )
        with pytest.raises(OSError, match="save boom"):
            await storage.index_done_callback()

        monkeypatch.setattr(
            NetworkXStorage, "write_nx_graph", staticmethod(original_write)
        )

        assert await storage.has_node("n1")
        assert reload_attempts == 2
    finally:
        await storage.finalize()


@pytest.mark.asyncio
async def test_a_failed_save_does_not_publish_the_batch_on_the_next_commit(
    tmp_path, monkeypatch
):
    storage = await _worker(tmp_path)
    try:

        def boom(graph, file_name, workspace):
            raise OSError("save boom")

        original = NetworkXStorage.write_nx_graph
        monkeypatch.setattr(NetworkXStorage, "write_nx_graph", staticmethod(boom))
        await storage.upsert_node("failed_batch", {"entity_id": "failed_batch"})
        with pytest.raises(OSError, match="save boom"):
            await storage.index_done_callback()

        monkeypatch.setattr(NetworkXStorage, "write_nx_graph", staticmethod(original))
        await storage.upsert_node("later_doc", {"entity_id": "later_doc"})
        await storage.index_done_callback()

        persisted = NetworkXStorage.load_nx_graph(storage._graphml_xml_file)
        assert persisted is not None
        assert persisted.has_node("later_doc")
        assert not persisted.has_node("failed_batch")
    finally:
        await storage.finalize()


@pytest.mark.asyncio
async def test_a_failing_reload_does_not_mask_the_save_error(tmp_path, monkeypatch):
    storage = await _worker(tmp_path)
    try:
        await storage.upsert_node("n1", {"entity_id": "n1"})

        def boom(graph, file_name, workspace):
            raise OSError("save boom")

        def unreadable(file_name):
            raise OSError("reload boom")

        monkeypatch.setattr(NetworkXStorage, "write_nx_graph", staticmethod(boom))
        monkeypatch.setattr(NetworkXStorage, "load_nx_graph", staticmethod(unreadable))

        with pytest.raises(OSError, match="save boom"):
            await storage.index_done_callback()
    finally:
        await storage.finalize()
