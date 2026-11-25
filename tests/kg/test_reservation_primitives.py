"""Pipeline reservation primitives, architecture guards, and dead-process recovery.

Covers:
- Architecture guards keeping ``reconcile_dead_pipeline_reservations`` and
  ``make_owner_record`` centralized in ``lightrag/kg/shared_storage.py``.
- Cancellation-safe reservation primitives (``acquire_reservation``,
  ``acquire_enqueue_reservation``, ``acquire_processing_reservation``,
  ``with_reservation_lock``, ``with_token_set_reservation_lock``,
  ``release_owned_reservation``, ``release_token_set_reservation``,
  ``run_to_completion``, ``fence_workspace_for_recovery``,
  ``reservation_owner_kind``).
- Dead-process reconciliation (re-runnable vs. fenced non-rerunnable owners,
  manual-state cleanup, enqueue token reaping) and the ``recovery_required``
  guard across REST routes, SDK paths, ``/pipeline_status`` projection, and
  ``run_scanning_process`` deferred-processing handoff.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

import lightrag.kg.shared_storage as shared_storage
from lightrag.kg.pipeline_ingress import (
    ManualRetryPublishResult,
    PipelineIngressMessage,
)
from lightrag.kg.shared_storage import (
    _INTERNAL_PIPELINE_STATUS_FIELDS,
    acquire_enqueue_reservation,
    acquire_processing_reservation,
    acquire_reservation,
    check_pipeline_status_mutation,
    fence_workspace_for_recovery,
    finalize_share_data,
    get_namespace_data,
    get_namespace_lock,
    get_pipeline_ingress,
    has_scan_deferred_processing,
    initialize_pipeline_status,
    initialize_share_data,
    make_owner_record,
    pipeline_recovery_blocked_message,
    reconcile_dead_pipeline_reservations,
    release_owned_reservation,
    release_token_set_reservation,
    reservation_owner_kind,
    run_to_completion,
    with_reservation_lock,
    with_token_set_reservation_lock,
)

# Import document_routes under a clean argv — its module chain parses CLI args at
# import time and would otherwise choke on pytest's argv. Done via importlib so
# it is not flagged as a late module-level import (E402).
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
dr = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

pytestmark = pytest.mark.offline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIGHTRAG_ROOT = PROJECT_ROOT / "lightrag"
SHARED_STORAGE = LIGHTRAG_ROOT / "kg" / "shared_storage.py"


def _production_python_sources():
    for path in LIGHTRAG_ROOT.rglob("*.py"):
        if path != SHARED_STORAGE:
            yield path


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    pid = proc.pid
    proc.kill()
    proc.wait()
    return pid


def _dead_owner(kind: str) -> dict:
    return {
        "token": "t",
        "pid": _dead_pid(),
        "process_start_id": "gone",
        "kind": kind,
    }


@pytest.fixture
def recovery_enabled(monkeypatch):
    """Force the Linux-multiworker gate on so reconcile logic runs anywhere."""
    monkeypatch.setattr(shared_storage, "_reservation_recovery_enabled", lambda: True)


class _CountingStatus(dict):
    """DictProxy-shaped fake that counts remote-style method calls."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.copy_calls = 0
        self.get_calls = 0
        self.update_calls = 0

    def copy(self):
        self.copy_calls += 1
        return dict.copy(self)

    def get(self, key, default=None):
        self.get_calls += 1
        return super().get(key, default)

    def update(self, *args, **kwargs):
        self.update_calls += 1
        return super().update(*args, **kwargs)


class _IngressArmSpy:
    """Minimal pipeline-ingress stand-in counting auto-rescan arms."""

    def __init__(self):
        self.armed = 0

    def request_auto_rescan(self):
        self.armed += 1


# ---------------------------------------------------------------------------
# Architecture guards
# ---------------------------------------------------------------------------


def test_dead_reservation_reconcile_is_shared_storage_private():
    offenders = [
        str(path.relative_to(PROJECT_ROOT))
        for path in _production_python_sources()
        if "reconcile_dead_pipeline_reservations" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_reservation_owner_records_are_created_only_in_shared_storage():
    offenders = [
        str(path.relative_to(PROJECT_ROOT))
        for path in _production_python_sources()
        if "make_owner_record(" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


# ---------------------------------------------------------------------------
# acquire_reservation / acquire_enqueue_reservation / acquire_processing_reservation
# ---------------------------------------------------------------------------


async def test_acquire_reservation_takes_atomically_and_rejects():
    ps = {"busy": False, "scanning": False, "busy_owner": None}
    lock = asyncio.Lock()

    result = await acquire_reservation(
        ps,
        lock,
        owner_key="busy_owner",
        owner="tok1",
        flags={"busy": True},
        reject_when=[("busy", "pipeline busy"), ("scanning", "scanning")],
    )
    assert result.acquired is True and result.message is None
    assert ps["busy"] is True and ps["busy_owner"] == "tok1"

    result2 = await acquire_reservation(
        ps,
        lock,
        owner_key="busy_owner",
        owner="tok2",
        flags={"busy": True},
        reject_when=[("busy", "pipeline busy")],
    )
    assert result2.acquired is False and result2.message == "pipeline busy"
    assert ps["busy_owner"] == "tok1"


async def test_mutation_check_uses_one_snapshot_and_no_update_when_idle(monkeypatch):
    monkeypatch.setattr(shared_storage, "_reservation_recovery_enabled", lambda: False)
    ps = _CountingStatus(
        {
            "busy": False,
            "busy_owner": None,
            "scanning_owner": None,
            "pending_enqueue_tokens": {},
            "pending_enqueues": 0,
        }
    )

    result = await check_pipeline_status_mutation(ps, asyncio.Lock())

    assert result.acquired is True
    assert ps.copy_calls == 1
    assert ps.get_calls == 0
    assert ps.update_calls == 0


async def test_enqueue_acquire_combines_recovery_and_reservation_update(monkeypatch):
    monkeypatch.setattr(shared_storage, "_reservation_recovery_enabled", lambda: True)
    monkeypatch.setattr(shared_storage, "_process_alive", lambda *_: False)
    ps = _CountingStatus(
        {
            "busy": False,
            "busy_owner": None,
            "scanning": False,
            "scanning_exclusive": False,
            "scanning_owner": None,
            "destructive_busy": False,
            "pending_enqueue_tokens": {"dead": {"pid": 999999}},
            "pending_enqueues": 1,
        }
    )

    result = await acquire_enqueue_reservation(
        ps,
        asyncio.Lock(),
        token="new",
        reject_when=(("destructive_busy", "destructive"),),
    )

    assert result.acquired is True
    assert set(ps["pending_enqueue_tokens"]) == {"new"}
    assert ps["pending_enqueues"] == 1
    assert ps.copy_calls == 1
    assert ps.get_calls == 0
    assert ps.update_calls == 1


async def test_enqueue_acquire_rejects_under_manual_freeze():
    ps = {
        "busy": False,
        "busy_owner": None,
        "scanning_owner": None,
        "destructive_busy": False,
        "manual_freeze_requested": True,
        "pending_enqueue_tokens": {},
        "pending_enqueues": 0,
    }

    result = await acquire_enqueue_reservation(
        ps,
        asyncio.Lock(),
        token="new",
        reject_when=(
            ("destructive_busy", "destructive"),
            ("manual_freeze_requested", "manual retry draining"),
        ),
    )

    assert result.acquired is False
    assert result.conflict is shared_storage.PipelineReservationConflict.MANUAL_FREEZE
    assert result.message == "manual retry draining"
    assert ps["pending_enqueue_tokens"] == {} and ps["pending_enqueues"] == 0


async def test_enqueue_reservation_kind_survives_a_reweight():
    ps = {
        "busy": False,
        "busy_owner": None,
        "scanning_owner": None,
        "destructive_busy": False,
        "pending_enqueue_tokens": {},
        "pending_enqueues": 0,
    }
    lock = asyncio.Lock()

    minted = await acquire_enqueue_reservation(
        ps,
        lock,
        token="guard",
        reject_when=(),
        weight=0,
        kind=shared_storage.SOURCE_REPAIR_RESERVATION_KIND,
    )
    assert minted.acquired

    reweighted = await acquire_enqueue_reservation(
        ps, lock, token="guard", reject_when=(), weight=4
    )
    assert reweighted.acquired

    meta = ps["pending_enqueue_tokens"]["guard"]
    assert meta["weight"] == 4
    assert (
        shared_storage.reservation_kind(meta)
        == shared_storage.SOURCE_REPAIR_RESERVATION_KIND
    )


async def test_enqueue_reservation_defaults_to_the_enqueue_kind():
    ps = {
        "busy": False,
        "busy_owner": None,
        "scanning_owner": None,
        "destructive_busy": False,
        "pending_enqueue_tokens": {},
        "pending_enqueues": 0,
    }

    result = await acquire_enqueue_reservation(
        ps, asyncio.Lock(), token="upload", reject_when=(), weight=1
    )
    assert result.acquired
    assert (
        shared_storage.reservation_kind(ps["pending_enqueue_tokens"]["upload"])
        == shared_storage.ENQUEUE_RESERVATION_KIND
    )
    assert (
        shared_storage.reservation_kind({"pid": 1})
        == shared_storage.ENQUEUE_RESERVATION_KIND
    )
    assert (
        shared_storage.reservation_kind(None) == shared_storage.ENQUEUE_RESERVATION_KIND
    )


async def test_fence_precondition_is_evaluated_inside_the_writing_lock():
    ps = {"pending_enqueue_tokens": {}}

    fenced = await fence_workspace_for_recovery(
        ps,
        asyncio.Lock(),
        kind="manual_drain_enqueue_stalled",
        message="stalled",
        precondition=lambda snapshot: bool(snapshot.get("pending_enqueue_tokens")),
    )

    assert fenced is False
    assert ps.get("recovery_required") is None


async def test_fence_writes_when_the_precondition_still_holds():
    ps = {"pending_enqueue_tokens": {"t": {"pid": 1}}}

    fenced = await fence_workspace_for_recovery(
        ps,
        asyncio.Lock(),
        kind="manual_drain_enqueue_stalled",
        message="stalled",
        precondition=lambda snapshot: bool(snapshot.get("pending_enqueue_tokens")),
    )

    assert fenced is True
    assert ps["recovery_required"]["kind"] == "manual_drain_enqueue_stalled"


async def test_an_existing_fence_is_kept_without_consulting_the_precondition():
    calls = []
    ps = {"recovery_required": {"kind": "delete", "message": "worker died"}}

    def _precondition(snapshot):
        calls.append(snapshot)
        return False

    fenced = await fence_workspace_for_recovery(
        ps,
        asyncio.Lock(),
        kind="manual_drain_enqueue_stalled",
        message="stalled",
        precondition=_precondition,
    )

    assert fenced is True
    assert ps["recovery_required"]["kind"] == "delete"
    assert calls == []


def test_manual_freeze_flag_maps_to_manual_freeze_conflict():
    assert (
        shared_storage._conflict_for_status_flag("manual_freeze_requested")
        is shared_storage.PipelineReservationConflict.MANUAL_FREEZE
    )


async def test_single_owner_acquire_always_honors_recovery_fence(monkeypatch):
    monkeypatch.setattr(shared_storage, "_reservation_recovery_enabled", lambda: True)
    monkeypatch.setattr(shared_storage, "_process_alive", lambda *_: False)
    ps = _CountingStatus(
        {
            "busy": True,
            "destructive_busy": True,
            "busy_owner": {"token": "dead", "pid": 999999, "kind": "clear"},
            "scanning_owner": None,
            "pending_enqueue_tokens": {},
            "pending_enqueues": 0,
        }
    )

    result = await acquire_reservation(
        ps,
        asyncio.Lock(),
        owner_key="busy_owner",
        owner="new",
        owner_kind="processing",
        flags={"busy": True},
        reject_when=(),
    )

    assert result.acquired is False
    assert (
        result.conflict is shared_storage.PipelineReservationConflict.RECOVERY_REQUIRED
    )
    assert ps["busy"] is False
    assert ps["busy_owner"] is None
    assert ps["recovery_required"]["kind"] == "clear"
    assert ps.copy_calls == 1
    assert ps.get_calls == 0
    assert ps.update_calls == 1


async def test_owner_with_no_pid_fences_instead_of_holding_forever(monkeypatch):
    monkeypatch.setattr(shared_storage, "_reservation_recovery_enabled", lambda: True)
    ps = _CountingStatus(
        {
            "busy": True,
            "destructive_busy": False,
            "busy_owner": {"token": "orphan", "kind": "processing"},
            "scanning_owner": None,
            "manual_freeze_requested": True,
            "pending_enqueue_tokens": {},
            "pending_enqueues": 0,
        }
    )

    result = await acquire_reservation(
        ps,
        asyncio.Lock(),
        owner_key="busy_owner",
        owner="new",
        owner_kind="processing",
        flags={"busy": True},
        reject_when=(),
    )

    assert result.acquired is False
    assert (
        result.conflict is shared_storage.PipelineReservationConflict.RECOVERY_REQUIRED
    )
    assert ps["recovery_required"]["owner_key"] == "busy_owner"
    assert "no process identity" in ps["recovery_required"]["message"]
    assert ps["busy"] is True
    assert ps["busy_owner"] == {"token": "orphan", "kind": "processing"}
    assert ps["manual_freeze_requested"] is True


async def test_dead_owner_without_start_id_is_still_reclaimed(monkeypatch):
    monkeypatch.setattr(shared_storage, "_reservation_recovery_enabled", lambda: True)
    monkeypatch.setattr(shared_storage, "_process_alive", lambda *_: False)
    ps = _CountingStatus(
        {
            "busy": True,
            "destructive_busy": False,
            "busy_owner": {"token": "dead", "pid": 999999, "kind": "processing"},
            "scanning_owner": None,
            "pending_enqueue_tokens": {},
            "pending_enqueues": 0,
        }
    )

    result = await acquire_reservation(
        ps,
        asyncio.Lock(),
        owner_key="busy_owner",
        owner="new",
        owner_kind="processing",
        flags={"busy": True},
        reject_when=(),
    )

    assert result.acquired is True
    assert ps.get("recovery_required") in (None, {}, False)


async def test_processing_reservation_fences_busy_and_scanning(monkeypatch):
    monkeypatch.setattr(shared_storage, "_reservation_recovery_enabled", lambda: False)
    lock = asyncio.Lock()
    flags = {"job_name": "Default Job"}
    ingress = _IngressArmSpy()

    busy_ps = {
        "busy": True,
        "scanning_exclusive": False,
        "busy_owner": {"token": "destructive", "pid": 1, "kind": "clear"},
        "history_messages": [],
    }
    busy_res = await acquire_processing_reservation(
        busy_ps,
        lock,
        token="proc",
        already_held=False,
        pipeline_ingress=ingress,
        flags=flags,
    )
    assert busy_res.acquired is False
    assert busy_res.conflict is shared_storage.PipelineReservationConflict.BUSY
    assert busy_ps["busy"] is True
    assert busy_ps["busy_owner"]["token"] == "destructive"
    assert ingress.armed == 1

    scan_ps = {
        "busy": False,
        "scanning_exclusive": True,
        "scanning_owner": {"token": "scan", "pid": 1, "kind": "scan"},
        "history_messages": [],
    }
    scan_res = await acquire_processing_reservation(
        scan_ps,
        lock,
        token="proc",
        already_held=False,
        pipeline_ingress=ingress,
        flags=flags,
    )
    assert scan_res.acquired is False
    assert scan_res.conflict is shared_storage.PipelineReservationConflict.SCANNING
    assert scan_ps["busy"] is False
    assert scan_ps.get("busy_owner") is None
    assert scan_ps["scan_deferred_processing"] is True
    assert ingress.armed == 1

    handoff_ps = {
        "busy": True,
        "scanning_exclusive": True,
        "busy_owner": {"token": "proc", "pid": 1, "kind": "processing"},
        "history_messages": ["stale"],
    }
    handoff_res = await acquire_processing_reservation(
        handoff_ps,
        lock,
        token="proc",
        already_held=True,
        pipeline_ingress=ingress,
        flags=flags,
    )
    assert handoff_res.acquired is True
    assert handoff_ps["busy"] is True
    assert handoff_ps["busy_owner"]["token"] == "proc"
    assert list(handoff_ps["history_messages"]) == []
    assert handoff_ps["scan_deferred_processing"] is False
    assert ingress.armed == 1


async def test_processing_reservation_busy_arm_failure_propagates(monkeypatch):
    monkeypatch.setattr(shared_storage, "_reservation_recovery_enabled", lambda: False)
    lock = asyncio.Lock()

    class _BrokenIngress:
        """Ingress stand-in that fails when auto-rescan is armed."""

        def request_auto_rescan(self):
            raise RuntimeError("manager connection lost")

    busy_ps = {
        "busy": True,
        "scanning_exclusive": False,
        "busy_owner": {"token": "destructive", "pid": 1, "kind": "clear"},
        "history_messages": [],
    }
    with pytest.raises(RuntimeError, match="manager connection lost"):
        await acquire_processing_reservation(
            busy_ps,
            lock,
            token="proc",
            already_held=False,
            pipeline_ingress=_BrokenIngress(),
            flags={"job_name": "Default Job"},
        )
    assert busy_ps["busy"] is True
    assert busy_ps["busy_owner"]["token"] == "destructive"


async def test_has_scan_deferred_processing_is_read_only():
    lock = asyncio.Lock()
    ps = {"scan_deferred_processing": True}
    assert await has_scan_deferred_processing(ps, lock) is True
    assert ps["scan_deferred_processing"] is True
    assert await has_scan_deferred_processing({}, lock) is False
    assert (
        await has_scan_deferred_processing({"scan_deferred_processing": False}, lock)
        is False
    )


# ---------------------------------------------------------------------------
# run_to_completion & owner-checked reservation locks
# ---------------------------------------------------------------------------


async def test_run_to_completion_returns_result():
    async def work():
        return 42

    assert await run_to_completion(work) == 42


async def test_run_to_completion_defers_caller_cancel_until_work_done():
    gate = asyncio.Event()
    completed = False

    async def work():
        nonlocal completed
        await gate.wait()
        completed = True
        return "done"

    task = asyncio.ensure_future(run_to_completion(work))
    await asyncio.sleep(0)

    task.cancel()
    for _ in range(3):
        await asyncio.sleep(0)

    assert not completed
    assert not task.done()

    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert completed


async def test_run_to_completion_restarts_when_work_task_cancelled():
    calls = 0

    async def work():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise asyncio.CancelledError()
        return "recovered"

    assert await run_to_completion(work) == "recovered"
    assert calls == 2


def _release_action(status):
    status.update({"busy": False, "busy_owner": None})
    return "released"


async def test_with_reservation_lock_is_owner_checked():
    finalize_share_data()
    initialize_share_data(1)
    try:
        ws = "resv_ws"
        await initialize_pipeline_status(ws)
        ps = await get_namespace_data("pipeline_status", workspace=ws)
        lock = get_namespace_lock("pipeline_status", workspace=ws)

        result = await acquire_reservation(
            ps,
            lock,
            owner_key="busy_owner",
            owner="tok",
            flags={"busy": True},
            reject_when=[("busy", "busy")],
        )
        assert result.acquired

        res = await with_reservation_lock(
            ps, lock, owner_key="busy_owner", token="tok", action=_release_action
        )
        assert res == "released"
        assert ps["busy"] is False and ps["busy_owner"] is None

        await acquire_reservation(
            ps,
            lock,
            owner_key="busy_owner",
            owner="tok2",
            flags={"busy": True},
            reject_when=[("busy", "busy")],
        )
        ran = []

        def bad(status):
            ran.append(1)
            status.update({"busy": False, "busy_owner": None})
            return "oops"

        res2 = await with_reservation_lock(
            ps, lock, owner_key="busy_owner", token="WRONG", action=bad
        )
        assert res2 is None
        assert ran == []
        assert ps["busy"] is True and ps["busy_owner"] == "tok2"
    finally:
        finalize_share_data()


async def test_with_reservation_lock_matches_owner_record_token():
    finalize_share_data()
    initialize_share_data(1)
    try:
        ws = "resv_ws2"
        await initialize_pipeline_status(ws)
        ps = await get_namespace_data("pipeline_status", workspace=ws)
        lock = get_namespace_lock("pipeline_status", workspace=ws)

        owner = {
            "token": "tokX",
            "pid": 1,
            "process_start_id": None,
            "kind": "processing",
        }
        await acquire_reservation(
            ps,
            lock,
            owner_key="busy_owner",
            owner=owner,
            flags={"busy": True},
            reject_when=[("busy", "busy")],
        )
        assert ps["busy_owner"] == owner

        res = await with_reservation_lock(
            ps, lock, owner_key="busy_owner", token="tokX", action=_release_action
        )
        assert res == "released"
        assert ps["busy"] is False and ps["busy_owner"] is None
    finally:
        finalize_share_data()


async def test_enqueue_token_set_acquire_and_idempotent_release():
    finalize_share_data()
    initialize_share_data(1)
    try:
        ws = "enq_ws"
        await initialize_pipeline_status(ws)
        ps = await get_namespace_data("pipeline_status", workspace=ws)
        lock = get_namespace_lock("pipeline_status", workspace=ws)

        res1 = await acquire_enqueue_reservation(
            ps, lock, token="e1", reject_when=[("destructive_busy", "d")]
        )
        res2 = await acquire_enqueue_reservation(
            ps, lock, token="e2", reject_when=[("destructive_busy", "d")]
        )
        assert res1.acquired and res2.acquired
        assert ps["pending_enqueues"] == 2
        assert set(ps["pending_enqueue_tokens"]) == {"e1", "e2"}

        await with_token_set_reservation_lock(
            ps, lock, tokens_key="pending_enqueue_tokens", token="e1"
        )
        assert ps["pending_enqueues"] == 1
        assert set(ps["pending_enqueue_tokens"]) == {"e2"}

        await with_token_set_reservation_lock(
            ps, lock, tokens_key="pending_enqueue_tokens", token="e1"
        )
        assert ps["pending_enqueues"] == 1
        assert set(ps["pending_enqueue_tokens"]) == {"e2"}
    finally:
        finalize_share_data()


async def test_release_owned_reservation_survives_fetch_cancellation(monkeypatch):
    finalize_share_data()
    initialize_share_data(1)
    try:
        ws = "rel_owned_cancel_ws"
        await initialize_pipeline_status(ws)
        ps = await get_namespace_data("pipeline_status", workspace=ws)
        lock = get_namespace_lock("pipeline_status", workspace=ws)
        await acquire_reservation(
            ps,
            lock,
            owner_key="busy_owner",
            owner="tok",
            flags={"busy": True, "destructive_busy": True},
            reject_when=[("busy", "busy")],
        )

        real_get = shared_storage.get_namespace_data
        entered = asyncio.Event()

        async def _slow_get(name, workspace=None):
            entered.set()
            await asyncio.sleep(0.05)
            return await real_get(name, workspace=workspace)

        monkeypatch.setattr(shared_storage, "get_namespace_data", _slow_get)

        def _release(status):
            status.update(
                {"busy": False, "destructive_busy": False, "busy_owner": None}
            )

        task = asyncio.ensure_future(
            release_owned_reservation(
                ws, owner_key="busy_owner", token="tok", action=_release
            )
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert ps["busy"] is False
        assert ps["destructive_busy"] is False
        assert ps["busy_owner"] is None
    finally:
        finalize_share_data()


async def test_release_token_set_reservation_survives_fetch_cancellation(monkeypatch):
    finalize_share_data()
    initialize_share_data(1)
    try:
        ws = "rel_tokset_cancel_ws"
        await initialize_pipeline_status(ws)
        ps = await get_namespace_data("pipeline_status", workspace=ws)
        lock = get_namespace_lock("pipeline_status", workspace=ws)
        await acquire_enqueue_reservation(
            ps, lock, token="e1", reject_when=[("destructive_busy", "d")]
        )
        assert ps["pending_enqueues"] == 1

        real_get = shared_storage.get_namespace_data
        entered = asyncio.Event()

        async def _slow_get(name, workspace=None):
            entered.set()
            await asyncio.sleep(0.05)
            return await real_get(name, workspace=workspace)

        monkeypatch.setattr(shared_storage, "get_namespace_data", _slow_get)

        task = asyncio.ensure_future(
            release_token_set_reservation(
                ws, tokens_key="pending_enqueue_tokens", token="e1"
            )
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert ps["pending_enqueues"] == 0
        assert dict(ps["pending_enqueue_tokens"]) == {}
    finally:
        finalize_share_data()


def test_reservation_owner_kind_reads_full_record_and_rejects_unidentifiable():
    assert reservation_owner_kind(make_owner_record("t", "admin")) == "admin"
    for invalid in (
        None,
        "bare-token",
        {"token": "t"},
        {"token": "t", "kind": None},
        42,
    ):
        assert reservation_owner_kind(invalid) is None


# ---------------------------------------------------------------------------
# Dead-process reconciliation (reconcile_dead_pipeline_reservations)
# ---------------------------------------------------------------------------


def test_reconcile_reclaims_dead_processing_owner(recovery_enabled):
    status = {"busy": True, "busy_owner": _dead_owner("processing")}
    reconcile_dead_pipeline_reservations(status)
    assert status["busy"] is False
    assert status["busy_owner"] is None
    assert "recovery_required" not in status


def test_reconcile_clears_manual_state_with_dead_busy_owner(recovery_enabled):
    status = {
        "busy": True,
        "busy_owner": _dead_owner("processing"),
        "manual_freeze_requested": True,
        "manual_resetting": True,
        "manual_phase": shared_storage.MANUAL_PHASE_EXCLUSIVE_RESET,
        "manual_owner": {"request_id": "r1", "token": "t", "pid": 999999},
    }
    reconcile_dead_pipeline_reservations(status)
    assert status["busy"] is False and status["busy_owner"] is None
    assert status["manual_freeze_requested"] is False
    assert status["manual_resetting"] is False
    assert status["manual_phase"] == shared_storage.MANUAL_PHASE_IDLE
    assert status["manual_owner"] is None


def test_reconcile_reclaims_dead_scan_owner(recovery_enabled):
    status = {
        "scanning": True,
        "scanning_exclusive": True,
        "scanning_owner": _dead_owner("scan"),
    }
    reconcile_dead_pipeline_reservations(status)
    assert status["scanning"] is False
    assert status["scanning_exclusive"] is False
    assert status["scanning_owner"] is None
    assert "recovery_required" not in status


@pytest.mark.parametrize("kind", ["custom_chunks", "delete", "clear"])
def test_reconcile_fences_dead_non_rerunnable_owner(recovery_enabled, kind):
    status = {
        "busy": True,
        "destructive_busy": kind in ("delete", "clear"),
        "busy_owner": _dead_owner(kind),
        "operation_record": {"kind": kind, "doc_id": "doc-1"},
    }
    reconcile_dead_pipeline_reservations(status)
    assert status["busy"] is False
    assert status["destructive_busy"] is False
    assert status["busy_owner"] is None
    rec = status["recovery_required"]
    assert rec["kind"] == kind
    assert rec["owner_key"] == "busy_owner"
    assert rec["operation_record"] == {"kind": kind, "doc_id": "doc-1"}


def test_reconcile_keeps_live_owner(recovery_enabled):
    status = {
        "busy": True,
        "busy_owner": make_owner_record("t", "processing"),
    }
    reconcile_dead_pipeline_reservations(status)
    assert status["busy"] is True
    assert status["busy_owner"] is not None


def test_reconcile_drops_dead_enqueue_tokens(recovery_enabled):
    status = {
        "pending_enqueues": 2,
        "pending_enqueue_tokens": {
            "live": {"pid": os.getpid(), "process_start_id": None},
            "dead": {"pid": _dead_pid(), "process_start_id": "gone"},
        },
    }
    reconcile_dead_pipeline_reservations(status)
    assert set(status["pending_enqueue_tokens"]) == {"live"}
    assert status["pending_enqueues"] == 1


def test_reconcile_preserves_the_surviving_tokens_kind(recovery_enabled):
    status = {
        "pending_enqueues": 2,
        "pending_enqueue_tokens": {
            "repair": {
                "pid": os.getpid(),
                "process_start_id": None,
                "weight": 0,
                "kind": shared_storage.SOURCE_REPAIR_RESERVATION_KIND,
            },
            "dead": {"pid": _dead_pid(), "process_start_id": "gone"},
        },
    }
    reconcile_dead_pipeline_reservations(status)
    surviving = status["pending_enqueue_tokens"]["repair"]
    assert (
        shared_storage.reservation_kind(surviving)
        == shared_storage.SOURCE_REPAIR_RESERVATION_KIND
    )


def test_reconcile_recalibrates_enqueue_count(recovery_enabled):
    status = {
        "pending_enqueues": 5,
        "pending_enqueue_tokens": {"live": {"pid": os.getpid()}},
    }
    reconcile_dead_pipeline_reservations(status)
    assert status["pending_enqueues"] == 1


def test_reconcile_is_noop_when_disabled():
    status = {"busy": True, "busy_owner": _dead_owner("processing")}
    reconcile_dead_pipeline_reservations(status)
    assert status["busy"] is True
    assert status["busy_owner"] is not None


async def test_reap_dead_reservations_locked_drops_dead_token(recovery_enabled):
    status = {
        "pending_enqueues": 2,
        "pending_enqueue_tokens": {
            "live": {"pid": os.getpid(), "process_start_id": None},
            "dead": {"pid": _dead_pid(), "process_start_id": "gone"},
        },
    }
    await shared_storage.reap_dead_reservations_locked(status, asyncio.Lock())
    assert set(status["pending_enqueue_tokens"]) == {"live"}
    assert status["pending_enqueues"] == 1


def test_pipeline_recovery_blocked_message():
    status = {
        "recovery_required": {
            "kind": "delete",
            "owner_key": "busy_owner",
            "operation_record": {"kind": "delete", "doc_id": "doc-9"},
        }
    }
    msg = pipeline_recovery_blocked_message(status)
    assert "delete" in msg and "doc-9" in msg
    assert (
        pipeline_recovery_blocked_message({}) == "Pipeline is not fenced for recovery."
    )


def test_internal_fields_constant_covers_recovery_state():
    for field in (
        "busy_owner",
        "scanning_owner",
        "pending_enqueue_tokens",
        "operation_record",
        "recovery_required",
        "manual_owner",
        "manual_freeze_requested",
        "manual_resetting",
        "manual_phase",
    ):
        assert field in _INTERNAL_PIPELINE_STATUS_FIELDS


async def test_initialize_pipeline_status_seeds_manual_state_at_idle():
    ws = "manual-init-ws"
    initialize_share_data()
    try:
        await initialize_pipeline_status(workspace=ws)
        ps = await get_namespace_data("pipeline_status", workspace=ws)
        assert ps["manual_freeze_requested"] is False
        assert ps["manual_resetting"] is False
        assert ps["manual_phase"] == shared_storage.MANUAL_PHASE_IDLE
        assert ps["manual_owner"] is None
    finally:
        finalize_share_data()


# ---------------------------------------------------------------------------
# recovery_required guard across REST, SDK, status projection, and scan handoff
# ---------------------------------------------------------------------------


class _Rag:
    """Minimal RAG stand-in exposing a workspace for document-route guards."""

    workspace = "recovery-guard-ws"


async def _seed_recovery_required(rag):
    await initialize_pipeline_status(workspace=rag.workspace)
    ps = await get_namespace_data("pipeline_status", workspace=rag.workspace)
    ps["recovery_required"] = {
        "kind": "delete",
        "owner_key": "busy_owner",
        "operation_record": {"kind": "delete", "doc_id": "doc-1"},
    }
    return ps


def _endpoint(router, name):
    return [r.endpoint for r in router.routes if getattr(r, "name", "") == name][-1]


async def test_acquire_destructive_refuses_when_recovery_required():
    finalize_share_data()
    initialize_share_data(1)
    try:
        rag = _Rag()
        await _seed_recovery_required(rag)
        acquired, reason = await dr._acquire_destructive_busy(
            rag,
            "tok",
            kind="clear",
            operation_record={"kind": "clear", "scope": "all"},
        )
        assert acquired is False
        assert "fenced" in reason.lower()
    finally:
        finalize_share_data()


async def test_reserve_enqueue_slot_raises_when_recovery_required():
    finalize_share_data()
    initialize_share_data(1)
    try:
        rag = _Rag()
        await _seed_recovery_required(rag)
        with pytest.raises(dr.HTTPException) as excinfo:
            await dr._reserve_enqueue_slot(rag, "enq-tok")
        assert excinfo.value.status_code == 503
    finally:
        finalize_share_data()


async def test_apipeline_enqueue_documents_refuses_when_recovery_required():
    from lightrag import LightRAG

    finalize_share_data()
    initialize_share_data(1)
    try:
        rag = LightRAG.__new__(LightRAG)
        rag.workspace = "recovery-enqueue-ws"
        await initialize_pipeline_status(workspace=rag.workspace)
        ps = await get_namespace_data("pipeline_status", workspace=rag.workspace)
        ps["recovery_required"] = {
            "kind": "clear",
            "owner_key": "busy_owner",
            "operation_record": {"kind": "clear", "scope": "all"},
        }
        with pytest.raises(RuntimeError) as excinfo:
            await rag.apipeline_enqueue_documents(["hello"])
        assert "fenced" in str(excinfo.value).lower()
    finally:
        finalize_share_data()


async def test_processing_loop_bails_when_recovery_required():
    from lightrag import LightRAG

    finalize_share_data()
    initialize_share_data(1)
    try:
        rag = LightRAG.__new__(LightRAG)
        rag.workspace = "recovery-process-ws"
        await initialize_pipeline_status(workspace=rag.workspace)
        ps = await get_namespace_data("pipeline_status", workspace=rag.workspace)
        ps["busy"] = False
        ps["recovery_required"] = {
            "kind": "delete",
            "owner_key": "busy_owner",
            "operation_record": {"kind": "delete", "doc_id": "doc-1"},
        }
        await rag.apipeline_process_enqueue_documents()
        assert ps.get("busy") is False
        assert ps.get("recovery_required") is not None
    finally:
        finalize_share_data()


async def test_graph_mutation_sdk_refuses_when_recovery_required():
    from lightrag import LightRAG

    finalize_share_data()
    initialize_share_data(1)
    try:
        rag = LightRAG.__new__(LightRAG)
        rag.workspace = "recovery-graph-ws"
        await initialize_pipeline_status(workspace=rag.workspace)
        ps = await get_namespace_data("pipeline_status", workspace=rag.workspace)
        ps["recovery_required"] = {
            "kind": "delete",
            "owner_key": "busy_owner",
            "operation_record": {"kind": "delete", "doc_id": "doc-1"},
        }
        with pytest.raises(RuntimeError, match="fenced"):
            await rag.acreate_entity("E", {"description": "x"})
        with pytest.raises(RuntimeError, match="fenced"):
            await rag.adelete_by_entity("E")
        with pytest.raises(RuntimeError, match="fenced"):
            await rag.amerge_entities(["A"], "B")
        with pytest.raises(RuntimeError, match="fenced"):
            await rag.ainsert_custom_kg({"chunks": []})
    finally:
        finalize_share_data()


async def test_reprocess_endpoint_refuses_when_recovery_required(tmp_path):
    finalize_share_data()
    initialize_share_data(1)
    try:
        rag = _Rag()
        await _seed_recovery_required(rag)
        router = dr.create_document_routes(rag, dr.DocumentManager(str(tmp_path)))
        reprocess = _endpoint(router, "reprocess_failed_documents")
        with pytest.raises(dr.HTTPException) as excinfo:
            await reprocess(set())
        assert excinfo.value.status_code == 503
    finally:
        finalize_share_data()


async def test_force_reset_recovery_endpoint(tmp_path):
    finalize_share_data()
    initialize_share_data(1)
    try:
        rag = _Rag()
        ps = await _seed_recovery_required(rag)
        router = dr.create_document_routes(rag, dr.DocumentManager(str(tmp_path)))
        force_reset = _endpoint(router, "force_reset_recovery")

        with pytest.raises(dr.HTTPException) as excinfo:
            await force_reset(dr.ForceResetRecoveryRequest(confirm=False))
        assert excinfo.value.status_code == 400
        assert ps.get("recovery_required") is not None

        resp = await force_reset(dr.ForceResetRecoveryRequest(confirm=True))
        assert resp.status == "reset"
        assert ps.get("recovery_required") is None
        assert ps.get("busy") is False
        assert ps.get("busy_owner") is None
        assert ps.get("manual_freeze_requested") is False
        assert ps.get("manual_owner") is None

        resp2 = await force_reset(dr.ForceResetRecoveryRequest(confirm=True))
        assert resp2.status == "no_recovery_required"
    finally:
        finalize_share_data()


async def test_force_reset_cancels_the_queued_manual_retries(tmp_path):
    finalize_share_data()
    initialize_share_data(1)
    try:
        rag = _Rag()
        ps = await _seed_recovery_required(rag)
        ingress = await get_pipeline_ingress(rag.workspace)
        for request_id in ("r1", "r2"):
            assert (
                ingress.request_manual_retry(
                    request_id,
                    PipelineIngressMessage(
                        kind="rescan", retry_failed=True, request_id=request_id
                    ),
                )
                is ManualRetryPublishResult.ACCEPTED
            )
        ingress.put_document(PipelineIngressMessage(kind="document", doc_id="doc-a"))

        router = dr.create_document_routes(rag, dr.DocumentManager(str(tmp_path)))
        force_reset = _endpoint(router, "force_reset_recovery")
        resp = await force_reset(dr.ForceResetRecoveryRequest(confirm=True))

        assert resp.status == "reset"
        assert resp.cancelled_manual_retries == 2
        assert "2 queued manual retry request(s) cancelled" in resp.message
        assert ps.get("recovery_required") is None
        assert ingress.snapshot_manual_retries() == []
        assert ingress.counts()["documents"] == 1

        assert (
            ingress.request_manual_retry(
                "r1",
                PipelineIngressMessage(
                    kind="rescan", retry_failed=True, request_id="r1"
                ),
            )
            is ManualRetryPublishResult.ALREADY_TERMINAL
        )
    finally:
        finalize_share_data()


async def test_force_reset_reports_zero_when_nothing_was_queued(tmp_path):
    finalize_share_data()
    initialize_share_data(1)
    try:
        rag = _Rag()
        await _seed_recovery_required(rag)
        router = dr.create_document_routes(rag, dr.DocumentManager(str(tmp_path)))
        force_reset = _endpoint(router, "force_reset_recovery")
        resp = await force_reset(dr.ForceResetRecoveryRequest(confirm=True))

        assert resp.status == "reset"
        assert resp.cancelled_manual_retries == 0
        assert "cancelled" not in resp.message
    finally:
        finalize_share_data()


async def test_pipeline_status_filters_internal_fields(tmp_path):
    finalize_share_data()
    initialize_share_data(1)
    try:
        rag = _Rag()
        await initialize_pipeline_status(workspace=rag.workspace)
        ps = await get_namespace_data("pipeline_status", workspace=rag.workspace)
        ps.update(
            {
                "busy_owner": {"token": "secret", "pid": 1, "kind": "processing"},
                "scanning_owner": {"token": "s", "pid": 2, "kind": "scan"},
                "pending_enqueue_tokens": {"e": {"pid": 3}},
                "operation_record": {"kind": "delete", "doc_id": "d"},
                "recovery_required": {
                    "kind": "delete",
                    "owner_key": "busy_owner",
                    "operation_record": {"kind": "delete", "doc_id": "doc-target"},
                    "owner_token": "tok-must-not-leak",
                    "pid": 4242,
                },
            }
        )
        router = dr.create_document_routes(rag, dr.DocumentManager(str(tmp_path)))
        get_status = _endpoint(router, "get_pipeline_status")
        resp = await get_status()
        dumped = resp.model_dump()
        for field in _INTERNAL_PIPELINE_STATUS_FIELDS:
            if field == "recovery_required":
                continue
            assert field not in dumped, f"{field} leaked to /pipeline_status"

        assert dumped["recovery_required"] is True
        assert dumped["recovery_kind"] == "delete"
        assert "a worker died mid 'delete'" in dumped["recovery_message"]
        assert "doc-target" in dumped["recovery_message"]

        rendered = repr(dumped)
        assert "tok-must-not-leak" not in rendered
        assert "4242" not in rendered
        assert "secret" not in rendered
        assert "owner_key" not in rendered
    finally:
        finalize_share_data()


async def test_pipeline_status_reads_one_proxy_snapshot(monkeypatch, tmp_path):
    class SnapshotOnlyStatus(dict):
        """Status proxy double enforcing single-copy reads."""

        copy_calls = 0

        def copy(self):
            self.copy_calls += 1
            return dict.copy(self)

        def get(self, *_args, **_kwargs):
            raise AssertionError("pipeline status must be read from one snapshot")

    status = SnapshotOnlyStatus(
        {"busy": False, "history_messages": [], "job_start": None}
    )

    async def fake_get_namespace_data(*_args, **_kwargs):
        return status

    async def fake_update_flags(*_args, **_kwargs):
        return {}

    monkeypatch.setattr(shared_storage, "get_namespace_data", fake_get_namespace_data)
    monkeypatch.setattr(
        shared_storage, "get_namespace_lock", lambda *_a, **_k: asyncio.Lock()
    )
    monkeypatch.setattr(
        shared_storage, "get_all_update_flags_status", fake_update_flags
    )

    rag = _Rag()
    router = dr.create_document_routes(rag, dr.DocumentManager(str(tmp_path)))
    response = await _endpoint(router, "get_pipeline_status")()

    assert response.busy is False
    assert status.copy_calls == 1


class _DeferRag:
    """Minimal rag double for run_scanning_process's deferred-drive path."""

    def __init__(self, workspace, process_calls):
        self.workspace = workspace
        self._process_calls = process_calls

    async def arollback_failed_custom_chunk_patches(self, **_kwargs):
        return None

    async def apipeline_process_enqueue_documents(self):
        self._process_calls.append(1)


class _BoomDocManager:
    """Raises during classification so no scan branch drives the queue."""

    def iter_new_files(self):
        raise RuntimeError("classification boom")


class _CancelDocManager:
    """Raises CancelledError to simulate a scan cancelled by server shutdown."""

    def iter_new_files(self):
        raise asyncio.CancelledError()


async def test_scan_drives_deferred_processing_on_error_path():
    finalize_share_data()
    initialize_share_data(1)
    try:
        ws = "scan-defer-ws"
        await initialize_pipeline_status(workspace=ws)
        ps = await get_namespace_data("pipeline_status", workspace=ws)
        ps["scan_deferred_processing"] = True

        process_calls: list[int] = []
        await dr.run_scanning_process(
            _DeferRag(ws, process_calls), _BoomDocManager(), "track-defer"
        )

        assert process_calls == [1]
        ps_after = await get_namespace_data("pipeline_status", workspace=ws)
        assert ps_after.get("scan_deferred_processing") is True
    finally:
        finalize_share_data()


async def test_scan_without_deferred_flag_does_not_extra_drive():
    finalize_share_data()
    initialize_share_data(1)
    try:
        ws = "scan-nodefer-ws"
        await initialize_pipeline_status(workspace=ws)

        process_calls: list[int] = []
        await dr.run_scanning_process(
            _DeferRag(ws, process_calls), _BoomDocManager(), "track-nodefer"
        )

        assert process_calls == []
    finally:
        finalize_share_data()


async def test_cancelled_scan_skips_deferred_drive():
    finalize_share_data()
    initialize_share_data(1)
    try:
        ws = "scan-cancel-ws"
        await initialize_pipeline_status(workspace=ws)
        ps = await get_namespace_data("pipeline_status", workspace=ws)
        ps["scan_deferred_processing"] = True

        process_calls: list[int] = []
        with pytest.raises(asyncio.CancelledError):
            await dr.run_scanning_process(
                _DeferRag(ws, process_calls), _CancelDocManager(), "track-cancel"
            )

        assert process_calls == []
        ps_after = await get_namespace_data("pipeline_status", workspace=ws)
        assert ps_after.get("scan_deferred_processing") is True
    finally:
        finalize_share_data()
