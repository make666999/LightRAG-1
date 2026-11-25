"""Tests for the keyed-lock server-side holder table and per-process async lock registry.

Covers:
- ``KeyedHolderTable`` kill safety under SIGKILL, zombie deadness, single-RPC
  uncontended cycles, reclaim CAS single-winner, real-liveness mutual exclusion,
  clock-adjustment-safe ``start_delta`` process identity, PID-aware ``_my_start_id``
  cache across fork, no-psutil fallback, spawn-mode Manager compatibility, and
  multiprocess ``/health`` lock status.
- ``KeyedUnifiedLock`` async registry delete-on-release lifecycle (uncontended,
  contended, cancelled waiter rollback, fresh reacquire, unmatched release,
  duplicate-key self-deadlock guard, and status schema compatibility).
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

import lightrag.kg.shared_storage as shared_storage
from lightrag.kg.shared_storage import (
    KeyedHolderTable,
    _get_combined_key,
    _holder_dead,
    _pid_alive,
    _start_delta,
    cleanup_keyed_lock,
    finalize_share_data,
    get_keyed_lock_status,
    get_storage_keyed_lock,
    initialize_share_data,
)

pytestmark = pytest.mark.offline

STATUS_KEYS = {
    "total_mp_locks",
    "pending_mp_cleanup",
    "total_async_locks",
    "pending_async_cleanup",
}


def _dead_pid() -> int:
    """A confirmed-dead PID: spawn a child, SIGKILL it, reap it."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    pid = proc.pid
    proc.kill()
    proc.wait()
    return pid


def _seed_direct(table: KeyedHolderTable, key: str, record: dict) -> None:
    """Forge an arbitrary holder record on a direct ``KeyedHolderTable`` instance."""
    with table._lock:
        table._holders[key] = dict(record)


def _registry():
    return shared_storage._storage_keyed_lock


async def _settle(ticks: int = 5):
    """Let already-started coroutines run up to their next await point."""
    for _ in range(ticks):
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Multiprocess KeyedHolderTable: kill safety, zombies, RPC budget, CAS, stress
# ---------------------------------------------------------------------------


def _kill_safety_holder(acquired_evt) -> None:  # pragma: no cover - child process
    async def _run():
        ctx = shared_storage.get_storage_keyed_lock("victim", namespace="killsafe")
        await ctx.__aenter__()
        acquired_evt.set()
        await asyncio.sleep(120)

    asyncio.run(_run())


def _kill_safety_contender() -> None:  # pragma: no cover - child process
    async def _run():
        async with asyncio.timeout(15):
            async with shared_storage.get_storage_keyed_lock(
                "victim", namespace="killsafe"
            ):
                pass

    try:
        asyncio.run(_run())
    except (TimeoutError, asyncio.TimeoutError):
        os._exit(2)
    os._exit(0)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs the fork start method")
@pytest.mark.filterwarnings("ignore:.*fork.*:DeprecationWarning")
def test_sigkilled_holder_never_deadlocks_other_processes():
    finalize_share_data()
    initialize_share_data(2)
    try:
        ctx = multiprocessing.get_context("fork")
        acquired = ctx.Event()
        holder = ctx.Process(target=_kill_safety_holder, args=(acquired,), daemon=True)
        holder.start()
        assert acquired.wait(10), "holder child failed to acquire the keyed lock"

        os.kill(holder.pid, signal.SIGKILL)
        holder.join(10)

        contender = ctx.Process(target=_kill_safety_contender, daemon=True)
        contender.start()
        contender.join(30)
        if contender.is_alive():
            contender.kill()
            contender.join(5)
            pytest.fail("contender deadlocked on a SIGKILLed holder's keyed lock")
        assert contender.exitcode == 0
    finally:
        finalize_share_data()


@pytest.mark.skipif(
    sys.platform == "win32" or shared_storage.psutil is None,
    reason="needs psutil and a POSIX zombie",
)
def test_zombie_holder_is_confirmed_dead():
    psutil = shared_storage.psutil
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        os.kill(proc.pid, signal.SIGKILL)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if psutil.Process(proc.pid).status() == psutil.STATUS_ZOMBIE:
                break
            time.sleep(0.01)
        else:
            pytest.fail("child never became a zombie")

        assert _pid_alive(proc.pid) is False
        assert shared_storage._process_alive(proc.pid, None) is False
        assert (
            _holder_dead({"owner_pid": proc.pid, "lease_id": "z", "start_delta": None})
            is True
        )
    finally:
        proc.wait()


async def test_uncontended_cycle_is_one_rpc_each_way(monkeypatch):
    finalize_share_data()
    initialize_share_data(2)
    try:
        calls = []
        original = shared_storage.BaseProxy._callmethod

        def counting(self, methodname, args=(), kwds=None):
            calls.append(methodname)
            return original(self, methodname, args, kwds or {})

        monkeypatch.setattr(shared_storage._HolderTableProxy, "_callmethod", counting)

        ctx = get_storage_keyed_lock("k", namespace="rpc")
        await ctx.__aenter__()
        assert calls == ["try_acquire"]
        calls.clear()
        await ctx.__aexit__(None, None, None)
        assert calls == ["release"]
    finally:
        finalize_share_data()


def test_proxy_grant_is_single_winner_and_owner_checked():
    finalize_share_data()
    initialize_share_data(2)
    try:
        table = shared_storage._keyed_holder_table
        assert table.try_acquire("w", {"owner_pid": os.getpid(), "lease_id": "a"})
        assert (
            table.try_acquire("w", {"owner_pid": os.getpid(), "lease_id": "b"}) is False
        )
        assert table.release("w", "b") is False
        assert table.release("w", "a") is True
        assert table.release("w", "a") is False
    finally:
        finalize_share_data()


def test_dead_record_reclaim_has_exactly_one_winner():
    table = KeyedHolderTable()
    key = "cas"
    _seed_direct(
        table,
        key,
        {"owner_pid": _dead_pid(), "lease_id": "dead-lease", "start_delta": None},
    )

    n = 16
    barrier = threading.Barrier(n, timeout=10)

    def contender(i: int) -> bool:
        record = {"owner_pid": os.getpid(), "lease_id": f"lease-{i}"}
        barrier.wait()
        return table.try_acquire(key, record)

    with ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(contender, range(n)))

    assert len(results) == n
    assert sum(results) == 1


def test_mutual_exclusion_stress_with_real_liveness():
    table = KeyedHolderTable()
    key = "stress"
    active = 0
    max_active = 0
    meta = threading.Lock()

    def worker(_idx: int) -> None:
        nonlocal active, max_active
        for _ in range(40):
            record = {"owner_pid": os.getpid(), "lease_id": uuid.uuid4().hex}
            while not table.try_acquire(key, record):
                time.sleep(0.0002)
            with meta:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.0005)
            with meta:
                active -= 1
            assert table.release(key, record["lease_id"]) is True

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(worker, range(6)))

    assert max_active == 1
    assert table.holders_snapshot() == {}


def test_no_unbounded_growth_over_distinct_keys():
    table = KeyedHolderTable()
    for i in range(2000):
        record = {"owner_pid": os.getpid(), "lease_id": f"l{i}"}
        assert table.try_acquire(f"key-{i}", record)
        assert table.release(f"key-{i}", f"l{i}")
    assert table.holders_snapshot() == {}


# ---------------------------------------------------------------------------
# start_delta identity, _my_start_id cache, no-psutil fallback, spawn Manager
# ---------------------------------------------------------------------------


def test_start_delta_baseline_and_forged_reuse_self():
    d0 = _start_delta(os.getpid())
    if d0 is None:
        pytest.skip("no process start identity available on this platform")
    assert _start_delta(os.getpid()) == d0
    me = {"owner_pid": os.getpid(), "lease_id": "x"}
    assert _holder_dead({**me, "start_delta": d0}) is False
    assert _holder_dead({**me, "start_delta": d0 - 2}) is True


def test_start_delta_common_mode_offset_cancels(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    base = {os.getpid(): 1000.0, 4242: 1234.5}
    state = {"offset": 0.0}
    monkeypatch.setattr(
        shared_storage, "_read_create_time", lambda pid: base[pid] + state["offset"]
    )

    clean = shared_storage._start_delta(4242)
    assert clean == pytest.approx(234.5)
    state["offset"] = 500.0
    assert shared_storage._start_delta(4242) == pytest.approx(clean)

    monkeypatch.setattr(shared_storage, "_pid_alive", lambda pid: True)
    record = {"owner_pid": 4242, "lease_id": "x", "start_delta": clean}
    assert shared_storage._holder_dead(record) is False


def test_start_delta_polluted_window_returns_none_and_never_kills(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    state = {"clock": 1000.0}

    def jumping_read(pid):
        state["clock"] += 100.0
        return state["clock"]

    monkeypatch.setattr(shared_storage, "_read_create_time", jumping_read)
    assert shared_storage._start_delta(4242) is None

    monkeypatch.setattr(shared_storage, "_pid_alive", lambda pid: True)
    record = {"owner_pid": 4242, "lease_id": "x", "start_delta": 1.0}
    assert shared_storage._holder_dead(record) is False


def test_start_delta_retry_recovers_from_one_polluted_sample(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    reads = iter([1000.0, 1234.5, 1600.0, 2000.0, 2234.5, 2000.0])
    monkeypatch.setattr(shared_storage, "_read_create_time", lambda pid: next(reads))
    assert shared_storage._start_delta(4242) == pytest.approx(234.5)


def test_holder_dead_non_linux_one_sided_tolerance(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(shared_storage, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(shared_storage, "_start_delta", lambda pid: 100.0)

    def record(d0):
        return {"owner_pid": 4242, "lease_id": "x", "start_delta": d0}

    assert shared_storage._holder_dead(record(100.0)) is False
    assert shared_storage._holder_dead(record(99.0)) is False
    assert shared_storage._holder_dead(record(98.9)) is True
    assert shared_storage._holder_dead(record(150.0)) is False


def test_holder_dead_linux_any_tick_difference(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(shared_storage, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(shared_storage, "_start_delta", lambda pid: 100)

    def record(d0):
        return {"owner_pid": 4242, "lease_id": "x", "start_delta": d0}

    assert shared_storage._holder_dead(record(100)) is False
    assert shared_storage._holder_dead(record(99)) is True
    assert shared_storage._holder_dead(record(101)) is True


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="Linux /proc tick track"
)
def test_start_delta_linux_never_touches_wall_clock(monkeypatch):
    wall_reads = []
    monkeypatch.setattr(
        shared_storage,
        "_read_create_time",
        lambda pid: wall_reads.append(pid) or 0.0,
    )
    assert _start_delta(os.getpid()) == 0
    assert wall_reads == []


def test_holder_dead_none_identity_falls_back_to_pid_probe(monkeypatch):
    dead = _dead_pid()
    assert _holder_dead({"owner_pid": dead, "lease_id": "x", "start_delta": None})
    samples = []
    monkeypatch.setattr(
        shared_storage, "_start_delta", lambda pid: samples.append(pid) or 0
    )
    assert (
        shared_storage._holder_dead(
            {"owner_pid": os.getpid(), "lease_id": "x", "start_delta": None}
        )
        is False
    )
    assert samples == []
    assert shared_storage._holder_dead({"lease_id": "x"}) is False


def test_grant_stamps_identity_and_live_rejection_stamps_nothing(monkeypatch):
    table = KeyedHolderTable()
    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        sampled = []
        real = shared_storage._start_delta

        def counting(pid):
            sampled.append(pid)
            return real(pid)

        monkeypatch.setattr(shared_storage, "_start_delta", counting)

        assert table.try_acquire("k", {"owner_pid": holder.pid, "lease_id": "l1"})
        assert sampled == [holder.pid]
        stamped = table.holders_snapshot()["k"]
        assert "start_delta" in stamped

        sampled.clear()
        rejected = table.try_acquire("k", {"owner_pid": os.getpid(), "lease_id": "l2"})
        assert rejected is False
        assert os.getpid() not in sampled
        assert sampled == [holder.pid]
    finally:
        holder.kill()
        holder.wait()


def test_my_start_id_cache_is_pid_aware(monkeypatch):
    monkeypatch.setattr(
        shared_storage, "_read_proc_starttime", lambda pid: f"token-{pid}"
    )
    monkeypatch.setattr(shared_storage, "_MY_START_ID_CACHE", "parent-token")
    monkeypatch.setattr(shared_storage, "_MY_START_ID_PID", os.getpid() - 1)
    assert shared_storage._my_start_id() == f"token-{os.getpid()}"
    monkeypatch.setattr(
        shared_storage, "_read_proc_starttime", lambda pid: "should-not-be-read"
    )
    assert shared_storage._my_start_id() == f"token-{os.getpid()}"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="real /proc assertion")
@pytest.mark.filterwarnings("ignore:.*fork.*:DeprecationWarning")
def test_my_start_id_recomputed_after_real_fork():
    parent_id = shared_storage._my_start_id()
    assert parent_id is not None

    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            os.close(read_fd)
            child_id = shared_storage._my_start_id()
            expected = shared_storage._read_proc_starttime(os.getpid())
            if child_id == expected and child_id != parent_id:
                code = 0
            os.write(write_fd, b"1" if code == 0 else b"0")
        finally:
            os._exit(code)

    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as fh:
        verdict = fh.read(1)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert verdict == b"1"
    assert shared_storage._my_start_id() == parent_id


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="os.kill(pid, 0) is not a liveness probe on Windows",
)
def test_pid_alive_fallback_without_psutil(monkeypatch):
    monkeypatch.setattr(shared_storage, "psutil", None)
    assert shared_storage._pid_alive(os.getpid()) is True
    assert shared_storage._pid_alive(_dead_pid()) is False
    assert shared_storage._read_create_time(os.getpid()) is None
    monkeypatch.setattr(sys, "platform", "darwin")
    assert shared_storage._start_delta(os.getpid()) is None


@pytest.mark.skipif(
    sys.platform == "win32" or shared_storage.psutil is None,
    reason="needs a POSIX zombie and psutil to observe it",
)
def test_pid_alive_fallback_keeps_zombies_alive(monkeypatch):
    psutil = shared_storage.psutil
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        os.kill(proc.pid, signal.SIGKILL)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if psutil.Process(proc.pid).status() == psutil.STATUS_ZOMBIE:
                break
            time.sleep(0.01)
        else:
            pytest.fail("child never became a zombie")
        monkeypatch.setattr(shared_storage, "psutil", None)
        assert shared_storage._pid_alive(proc.pid) is True
    finally:
        proc.wait()


def test_custom_manager_type_works_under_spawn():
    ctx = multiprocessing.get_context("spawn")
    manager = shared_storage._LightRAGManager(ctx=ctx)
    manager.start()
    try:
        table = manager.KeyedHolderTable()
        record = {"owner_pid": os.getpid(), "lease_id": "spawn-lease"}
        assert table.try_acquire("k", record) is True
        snapshot = table.holders_snapshot()
        assert snapshot["k"]["owner_pid"] == os.getpid()
        assert table.holder_count() == 1
        assert table.release("k", "spawn-lease") is True
        assert table.holder_count() == 0
    finally:
        manager.shutdown()


async def test_lock_status_counts_currently_held_keys():
    finalize_share_data()
    initialize_share_data(2)
    try:
        async with get_storage_keyed_lock(["k1", "k2"], namespace="status"):
            status = shared_storage._storage_keyed_lock.get_lock_status()
            assert status["total_mp_locks"] == 2
            assert status["pending_mp_cleanup"] == 0

            public = get_keyed_lock_status()
            assert public["total_mp_locks"] == 2
            assert public["pending_mp_cleanup"] == 0
            assert public["process_id"] == os.getpid()

        assert (
            shared_storage._storage_keyed_lock.get_lock_status()["total_mp_locks"] == 0
        )

        cleanup = shared_storage.cleanup_keyed_lock()
        assert cleanup["cleanup_performed"]["mp_cleaned"] == 0
    finally:
        finalize_share_data()


# ---------------------------------------------------------------------------
# Async keyed-lock registry: delete-on-release & /health schema
# ---------------------------------------------------------------------------


async def test_entry_dropped_immediately_on_release():
    finalize_share_data()
    initialize_share_data(1)
    try:
        keyed = _registry()
        combined = _get_combined_key("ns", "k")

        async with get_storage_keyed_lock("k", namespace="ns"):
            assert combined in keyed._async_lock
            assert keyed._async_lock_count[combined] == 1
            assert keyed.get_lock_status()["total_async_locks"] == 1

        assert combined not in keyed._async_lock
        assert combined not in keyed._async_lock_count
        assert keyed.get_lock_status()["total_async_locks"] == 0
    finally:
        finalize_share_data()


async def test_waiter_keeps_entry_alive_and_mutual_exclusion_holds():
    finalize_share_data()
    initialize_share_data(1)
    try:
        keyed = _registry()
        combined = _get_combined_key("ns", "contended")
        active = 0
        max_active = 0
        holder_entered = asyncio.Event()
        release_holder = asyncio.Event()

        async def holder():
            nonlocal active, max_active
            async with get_storage_keyed_lock("contended", namespace="ns"):
                active += 1
                max_active = max(max_active, active)
                holder_entered.set()
                await release_holder.wait()
                active -= 1

        async def waiter():
            nonlocal active, max_active
            async with get_storage_keyed_lock("contended", namespace="ns"):
                active += 1
                max_active = max(max_active, active)
                await asyncio.sleep(0)
                active -= 1

        a = asyncio.ensure_future(holder())
        await asyncio.wait_for(holder_entered.wait(), timeout=1.0)
        b = asyncio.ensure_future(waiter())
        await _settle()

        assert keyed._async_lock_count[combined] == 2
        assert combined in keyed._async_lock

        release_holder.set()
        await asyncio.gather(a, b)
        assert max_active == 1
        assert combined not in keyed._async_lock
        assert combined not in keyed._async_lock_count
    finally:
        finalize_share_data()


async def test_cancelled_waiter_rolls_back_reference_without_deleting_entry():
    finalize_share_data()
    initialize_share_data(1)
    try:
        keyed = _registry()
        combined = _get_combined_key("ns", "cancelwait")

        holder_ctx = get_storage_keyed_lock("cancelwait", namespace="ns")
        await holder_ctx.__aenter__()
        try:
            lock_obj = keyed._async_lock[combined]

            async def waiter():
                async with get_storage_keyed_lock("cancelwait", namespace="ns"):
                    pass  # pragma: no cover

            wtask = asyncio.ensure_future(waiter())
            await _settle()
            assert keyed._async_lock_count[combined] == 2

            wtask.cancel()
            with pytest.raises(asyncio.CancelledError):
                await wtask

            assert keyed._async_lock_count[combined] == 1
            assert keyed._async_lock[combined] is lock_obj
        finally:
            await holder_ctx.__aexit__(None, None, None)

        assert combined not in keyed._async_lock
        assert combined not in keyed._async_lock_count
    finally:
        finalize_share_data()


async def test_reacquire_after_full_release_uses_fresh_lock():
    finalize_share_data()
    initialize_share_data(1)
    try:
        keyed = _registry()
        combined = _get_combined_key("ns", "again")

        async with get_storage_keyed_lock("again", namespace="ns"):
            first = keyed._async_lock[combined]

        async with get_storage_keyed_lock("again", namespace="ns"):
            second = keyed._async_lock[combined]
            assert second is not first

        assert combined not in keyed._async_lock
    finally:
        finalize_share_data()


async def test_unmatched_release_is_ignored_and_leaves_no_phantom():
    finalize_share_data()
    initialize_share_data(1)
    try:
        keyed = _registry()
        combined = _get_combined_key("ns", "never-acquired")

        keyed._release_async_lock(combined)

        assert combined not in keyed._async_lock_count
        assert combined not in keyed._async_lock
        assert keyed.get_lock_status()["total_async_locks"] == 0
    finally:
        finalize_share_data()


async def test_distinct_keys_do_not_accumulate():
    finalize_share_data()
    initialize_share_data(1)
    try:
        keyed = _registry()
        for i in range(500):
            async with get_storage_keyed_lock(f"entity-{i}", namespace="idx"):
                assert _get_combined_key("idx", f"entity-{i}") in keyed._async_lock
        assert keyed._async_lock == {}
        assert keyed._async_lock_count == {}
    finally:
        finalize_share_data()


async def test_multiprocess_gate_entry_dropped_immediately_too():
    finalize_share_data()
    initialize_share_data(2)
    try:
        keyed = _registry()
        combined = _get_combined_key("ns", "mp")

        async with get_storage_keyed_lock("mp", namespace="ns"):
            assert combined in keyed._async_lock
            assert keyed._async_lock_count[combined] == 1

        assert combined not in keyed._async_lock
        assert combined not in keyed._async_lock_count
    finally:
        finalize_share_data()


async def test_status_shells_keep_health_schema():
    finalize_share_data()
    initialize_share_data(1)
    try:
        async with get_storage_keyed_lock("k", namespace="ns"):
            info = cleanup_keyed_lock()
            assert info["cleanup_performed"] == {"mp_cleaned": 0, "async_cleaned": 0}
            assert set(info["current_status"]) == STATUS_KEYS
            assert info["current_status"]["total_async_locks"] == 1
            assert info["current_status"]["pending_async_cleanup"] == 0

            public = get_keyed_lock_status()
            assert set(public) == STATUS_KEYS | {"process_id"}
            assert public["pending_async_cleanup"] == 0

        info = cleanup_keyed_lock()
        assert set(info["current_status"]) == STATUS_KEYS
        assert info["current_status"]["total_async_locks"] == 0
        assert info["current_status"]["pending_async_cleanup"] == 0
    finally:
        finalize_share_data()

    info = cleanup_keyed_lock()
    assert info["cleanup_performed"] == {"mp_cleaned": 0, "async_cleaned": 0}
    assert set(info["current_status"]) == STATUS_KEYS
    assert all(v == 0 for v in info["current_status"].values())
    public = get_keyed_lock_status()
    assert set(public) == STATUS_KEYS | {"process_id"}


async def test_duplicate_keys_acquire_once_instead_of_self_deadlocking():
    finalize_share_data()
    initialize_share_data(1)
    try:
        keyed = _registry()
        combined = _get_combined_key("ns", "k")

        async def _enter_and_exit():
            async with get_storage_keyed_lock(["k", "k"], namespace="ns"):
                assert keyed._async_lock_count[combined] == 1

        await asyncio.wait_for(_enter_and_exit(), timeout=5)

        assert combined not in keyed._async_lock
        assert combined not in keyed._async_lock_count
    finally:
        finalize_share_data()
