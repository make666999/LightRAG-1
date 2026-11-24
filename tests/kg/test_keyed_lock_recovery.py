"""Tests for keyed-lock cancellation safety and multiprocess dead-worker recovery.

Covers:
- Cancellation safety of ``NamespaceLock.__aexit__`` (clearing ``_ctx_var``
  before/despite cancellation) and ``_KeyedLockContext.__aexit__`` (entry
  snapshotting and resistant wait until release completes).
- Multiprocess dead-worker holder-lease recovery: dead owner reclaim, live owner
  preservation, owner-checked release, self-PID-reuse detection, full-stack
  ``NamespaceLock`` reclaim, and single-process lock bypass.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

import lightrag.kg.shared_storage as shared_storage
from lightrag.kg.shared_storage import (
    KeyedHolderTable,
    KeyedUnifiedLock,
    NamespaceLock,
    _get_combined_key,
    _KeyedLockContext,
    _pid_alive,
    finalize_share_data,
    get_final_namespace,
    get_namespace_lock,
    get_storage_keyed_lock,
    initialize_share_data,
)

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _GatedLeafLock:
    """A leaf lock whose ``__aexit__`` blocks until the test opens a gate."""

    def __init__(self) -> None:
        self.exit_started = asyncio.Event()
        self.release_gate = asyncio.Event()
        self.exit_completed = False

    async def __aexit__(self, exc_type, exc, tb):
        self.exit_started.set()
        await self.release_gate.wait()
        self.exit_completed = True
        return False


def _context_holding(leaf: _GatedLeafLock) -> _KeyedLockContext:
    """Build a real ``_KeyedLockContext`` whose single acquired entry is ``leaf``."""
    parent = KeyedUnifiedLock(default_enable_logging=False)
    parent._release_lock_for_key = MagicMock()
    ctx = _KeyedLockContext(parent, namespace="ns", keys=["k"], enable_logging=False)
    ctx._ul = [
        {
            "key": "k",
            "lock": leaf,
            "entered": True,
            "debug_inc": False,
            "ref_incremented": False,
        }
    ]
    return ctx


class _EnteredCtx:
    """Wrapper yielding an already-entered context manager for ``asyncio.wait_for``."""

    def __init__(self, ctx):
        self._ctx = ctx

    async def __aenter__(self):
        return self._ctx

    async def __aexit__(self, exc_type, exc, tb):
        return await self._ctx.__aexit__(exc_type, exc, tb)


async def _acquire_ctx(namespace: str, key: str):
    ctx = get_storage_keyed_lock([key], namespace=namespace)
    await ctx.__aenter__()
    return _EnteredCtx(ctx)


class _FakeKeyedCtx:
    """Minimal ctx whose ``__aexit__`` raises CancelledError on first call."""

    def __init__(self) -> None:
        self._raised = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if not self._raised:
            self._raised = True
            raise asyncio.CancelledError()
        return False


def _table():
    return shared_storage._keyed_holder_table


def _snapshot():
    return _table().holders_snapshot()


def _seed_holder(key: str, owner_pid: int, lease_id: str) -> None:
    assert _table().try_acquire(key, {"owner_pid": owner_pid, "lease_id": lease_id})


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    pid = proc.pid
    proc.kill()
    proc.wait()
    return pid


# ---------------------------------------------------------------------------
# Cancellation safety of release paths
# ---------------------------------------------------------------------------


async def test_keyed_lock_aexit_waits_for_release_under_cancellation():
    leaf = _GatedLeafLock()
    ctx = _context_holding(leaf)

    task = asyncio.ensure_future(ctx.__aexit__(None, None, None))
    await asyncio.wait_for(leaf.exit_started.wait(), timeout=1.0)
    assert ctx._ul is None

    task.cancel()
    for _ in range(3):
        await asyncio.sleep(0)

    assert not leaf.exit_completed
    assert not task.done()

    leaf.release_gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert leaf.exit_completed


async def test_keyed_lock_aexit_raises_if_release_task_directly_cancelled():
    leaf = _GatedLeafLock()
    ctx = _context_holding(leaf)

    task = asyncio.ensure_future(ctx.__aexit__(None, None, None))
    await asyncio.wait_for(leaf.exit_started.wait(), timeout=1.0)

    for child in asyncio.all_tasks():
        coro = child.get_coro()
        if getattr(coro, "__name__", "") == "release_all_locks":
            child.cancel()
            break
    else:  # pragma: no cover
        pytest.fail("release_all_locks task not found")

    with pytest.raises(asyncio.CancelledError):
        await task
    assert not leaf.exit_completed


async def test_keyed_lock_real_release_and_reacquire_after_cancel():
    finalize_share_data()
    initialize_share_data(1)
    try:
        namespace = "cancel_ns"
        combined = _get_combined_key(namespace, "k")
        holder_running = asyncio.Event()

        async def holder():
            async with get_storage_keyed_lock(["k"], namespace=namespace):
                holder_running.set()
                await asyncio.sleep(3600)

        htask = asyncio.ensure_future(holder())
        await asyncio.wait_for(holder_running.wait(), timeout=1.0)

        htask.cancel()
        with pytest.raises(asyncio.CancelledError):
            await htask

        async with await asyncio.wait_for(_acquire_ctx(namespace, "k"), timeout=1.0):
            assert combined in shared_storage._storage_keyed_lock._async_lock

        keyed = shared_storage._storage_keyed_lock
        assert combined not in keyed._async_lock_count
        assert combined not in keyed._async_lock
    finally:
        finalize_share_data()


async def test_namespace_lock_reentry_after_cancel_in_exit(monkeypatch):
    fake = _FakeKeyedCtx()
    monkeypatch.setattr(shared_storage, "get_storage_keyed_lock", lambda *a, **k: fake)

    lock = NamespaceLock("myns", "myws")
    await lock.__aenter__()
    assert lock._ctx_var.get() is fake

    with pytest.raises(asyncio.CancelledError):
        await lock.__aexit__(None, None, None)

    assert lock._ctx_var.get() is None

    await lock.__aenter__()
    assert lock._ctx_var.get() is fake
    await lock.__aexit__(None, None, None)
    assert lock._ctx_var.get() is None


# ---------------------------------------------------------------------------
# Multiprocess dead-worker lease recovery
# ---------------------------------------------------------------------------


async def test_lease_records_and_releases_holder():
    finalize_share_data()
    initialize_share_data(2)
    try:
        key = _get_combined_key("ns", "k")
        assert key not in _snapshot()

        async with get_storage_keyed_lock("k", namespace="ns"):
            rec = _snapshot().get(key)
            assert rec is not None
            assert rec["owner_pid"] == os.getpid()
            assert rec["lease_id"]
            assert "start_delta" in rec

        assert key not in _snapshot()
    finally:
        finalize_share_data()


async def test_lease_reclaims_confirmed_dead_holder():
    finalize_share_data()
    initialize_share_data(2)
    try:
        key = _get_combined_key("ns", "dead")
        dead = _dead_pid()
        assert _pid_alive(dead) is False

        _seed_holder(key, dead, "stale")

        async with asyncio.timeout(2):
            async with get_storage_keyed_lock("dead", namespace="ns"):
                rec = _snapshot().get(key)
                assert rec["owner_pid"] == os.getpid()
                assert rec["lease_id"] != "stale"

        assert key not in _snapshot()
    finally:
        finalize_share_data()


async def test_lease_does_not_reclaim_live_owner():
    finalize_share_data()
    initialize_share_data(2)
    try:
        key = _get_combined_key("ns", "live")
        _seed_holder(key, os.getpid(), "held-by-someone-else")

        async def _try_acquire():
            async with get_storage_keyed_lock("live", namespace="ns"):
                pass

        with pytest.raises((asyncio.TimeoutError, TimeoutError)):
            await asyncio.wait_for(_try_acquire(), timeout=0.5)

        assert _snapshot().get(key, {}).get("lease_id") == "held-by-someone-else"
        assert _table().release(key, "held-by-someone-else") is True
    finally:
        finalize_share_data()


async def test_lease_release_is_owner_checked():
    finalize_share_data()
    initialize_share_data(2)
    try:
        key = _get_combined_key("ns", "ownercheck")
        lock = get_storage_keyed_lock("ownercheck", namespace="ns")
        await lock.__aenter__()
        our_lease = _snapshot()[key]["lease_id"]
        assert _snapshot()[key]["owner_pid"] == os.getpid()

        assert _table().release(key, our_lease) is True
        _seed_holder(key, os.getpid(), "new-owner")

        await lock.__aexit__(None, None, None)
        assert _snapshot().get(key, {}).get("lease_id") == "new-owner"
        assert _table().release(key, "new-owner") is True
    finally:
        finalize_share_data()


def test_process_alive_detects_self_pid_reuse(monkeypatch):
    monkeypatch.setattr(shared_storage, "_my_start_id", lambda: "our-start-id")
    mypid = os.getpid()
    assert shared_storage._process_alive(mypid, "predecessor-start-id") is False
    assert shared_storage._process_alive(mypid, "our-start-id") is True
    assert shared_storage._process_alive(mypid, None) is True
    monkeypatch.setattr(shared_storage, "_my_start_id", lambda: None)
    assert shared_storage._process_alive(mypid, "predecessor-start-id") is True


async def test_lease_reclaims_on_self_pid_reuse(monkeypatch):
    finalize_share_data()
    initialize_share_data(2)
    try:
        our_delta = shared_storage._start_delta(os.getpid())
        if our_delta is None:
            pytest.skip("no process start identity available on this platform")

        table = KeyedHolderTable()
        key = _get_combined_key("ns", "reuse")
        with table._lock:
            table._holders[key] = {
                "owner_pid": os.getpid(),
                "lease_id": "stale",
                "start_delta": our_delta - 2,
            }
        monkeypatch.setattr(shared_storage, "_keyed_holder_table", table)

        async with asyncio.timeout(5):
            async with get_storage_keyed_lock("reuse", namespace="ns"):
                assert table.holders_snapshot()[key]["lease_id"] != "stale"
        assert key not in table.holders_snapshot()
    finally:
        finalize_share_data()


async def test_namespace_lock_reclaims_dead_holder():
    finalize_share_data()
    initialize_share_data(2)
    try:
        final_ns = get_final_namespace("pipeline_status", "ws")
        key = _get_combined_key(final_ns, "default_key")
        dead = _dead_pid()
        _seed_holder(key, dead, "stale")

        lock = get_namespace_lock("pipeline_status", workspace="ws")
        async with asyncio.timeout(2):
            async with lock:
                assert _snapshot()[key]["owner_pid"] == os.getpid()
        assert key not in _snapshot()
    finally:
        finalize_share_data()


async def test_single_process_keyed_lock_has_no_holder_records():
    finalize_share_data()
    initialize_share_data(1)
    try:
        assert shared_storage._keyed_holder_table is None
        async with get_storage_keyed_lock("k", namespace="ns"):
            assert shared_storage._keyed_holder_table is None
    finally:
        finalize_share_data()
