"""Unit tests for reserved and two-state committed background tasks plus manual-retry commit.

Covers:
- ``start_reserved_background_task``, ``_join_resistant``, and
  ``drain_reserved_background_tasks`` (start-barrier takeover, backstop release
  on startup failure / pre-takeover cancel, and shutdown drain).
- ``start_committed_background_task`` (two-state commit/ownership handoff:
  pre-commit cancel tears down cleanly, post-commit cancel leaves child running,
  commit crash backstops reservation while preserving published intent).
- ``commit_manual_retry_request`` (atomic fence + publish, destructive refusal,
  terminal-id refusal, pending-id idempotency, and capacity refusal).
"""

from __future__ import annotations

import asyncio

import pytest

from lightrag.kg.pipeline_ingress import (
    AsyncioPipelineIngress,
    PipelineIngressMessage,
)
from lightrag.kg.shared_storage import (
    ManualIntentRefused,
    _join_resistant,
    commit_manual_retry_request,
    drain_reserved_background_tasks,
    finalize_share_data,
    get_pipeline_ingress,
    initialize_share_data,
    start_committed_background_task,
    start_reserved_background_task,
)

pytestmark = pytest.mark.offline


async def _noop_backstop():
    return None


# ---------------------------------------------------------------------------
# start_reserved_background_task / drain_reserved_background_tasks
# ---------------------------------------------------------------------------


async def test_start_hands_over_to_child():
    bg: set = set()
    events = []
    gate = asyncio.Event()

    async def work(started):
        try:
            started.set()
            events.append("set")
            await gate.wait()
        finally:
            events.append("released")

    async def backstop():
        events.append("backstop")

    task = await start_reserved_background_task(
        bg, work=work, backstop_release=backstop
    )
    assert task in bg
    assert events == ["set"]

    gate.set()
    await task
    assert "released" in events and "backstop" not in events


async def test_start_backstop_and_raises_when_child_fails_before_signalling():
    bg: set = set()
    released = []

    async def work(started):
        raise RuntimeError("boom before started.set()")

    async def backstop():
        released.append("backstop")

    with pytest.raises(RuntimeError, match="failed to start"):
        await start_reserved_background_task(bg, work=work, backstop_release=backstop)
    assert released == ["backstop"]


async def test_start_teardown_on_caller_cancel_before_takeover():
    bg: set = set()
    events = []
    gate = asyncio.Event()

    async def work(started):
        try:
            await gate.wait()
            started.set()
            events.append("ran_unreserved_work")
        finally:
            events.append("child_finally")

    async def backstop():
        events.append("backstop")

    helper = asyncio.ensure_future(
        start_reserved_background_task(bg, work=work, backstop_release=backstop)
    )
    await asyncio.sleep(0)

    helper.cancel()
    for _ in range(3):
        await asyncio.sleep(0)

    with pytest.raises(asyncio.CancelledError):
        await helper

    assert "ran_unreserved_work" not in events
    assert "backstop" in events


async def test_join_resistant_returns_caller_cancel_without_propagating_task_error():
    async def ok():
        return 1

    t = asyncio.ensure_future(ok())
    assert await _join_resistant(t) is None

    async def boom():
        raise ValueError("x")

    t2 = asyncio.ensure_future(boom())
    assert await _join_resistant(t2) is None


async def test_drain_cancels_and_joins_all_tasks():
    bg: set = set()
    released = []

    async def work(started):
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            released.append(1)

    await start_reserved_background_task(bg, work=work, backstop_release=_noop_backstop)
    await start_reserved_background_task(bg, work=work, backstop_release=_noop_backstop)
    assert len(bg) == 2

    pending = await drain_reserved_background_tasks(bg)
    assert pending is None
    assert released == [1, 1]

    await asyncio.sleep(0)
    assert len(bg) == 0


# ---------------------------------------------------------------------------
# start_committed_background_task
# ---------------------------------------------------------------------------


class _Probe:
    """Tracks work execution, publication, and backstop calls for commit tests."""

    def __init__(self):
        self.work_started = asyncio.Event()
        self.work_release = asyncio.Event()
        self.work_completed = False
        self.backstop_calls = 0
        self.published = False

    async def backstop(self):
        self.backstop_calls += 1

    async def work(self):
        self.work_started.set()
        await self.work_release.wait()
        self.work_completed = True


def _commit_ok(probe):
    async def commit(state):
        probe.published = True
        state["committed"] = True
        return None

    return commit


async def test_success_path_runs_work_and_returns_task():
    probe = _Probe()
    tasks: set = set()
    task = await start_committed_background_task(
        tasks,
        commit=_commit_ok(probe),
        work=probe.work,
        backstop_release=probe.backstop,
    )
    await probe.work_started.wait()
    probe.work_release.set()
    await task
    assert probe.work_completed
    assert probe.backstop_calls == 0


async def test_refusal_raises_and_backstops_without_side_effects():
    probe = _Probe()

    async def commit(state):
        return "fenced: destructive job running"

    with pytest.raises(ManualIntentRefused, match="fenced"):
        await start_committed_background_task(
            set(), commit=commit, work=probe.work, backstop_release=probe.backstop
        )
    assert probe.backstop_calls == 1
    assert not probe.work_started.is_set()


async def test_commit_crash_backstops_and_raises():
    probe = _Probe()

    async def commit(state):
        raise RuntimeError("commit boom")

    with pytest.raises(RuntimeError, match="failed to start"):
        await start_committed_background_task(
            set(), commit=commit, work=probe.work, backstop_release=probe.backstop
        )
    assert probe.backstop_calls == 1


async def test_caller_cancel_before_commit_cancels_child_and_backstops():
    probe = _Probe()
    commit_entered = asyncio.Event()
    commit_release = asyncio.Event()
    child_cancelled = asyncio.Event()

    async def commit(state):
        commit_entered.set()
        try:
            await commit_release.wait()
        except asyncio.CancelledError:
            child_cancelled.set()
            raise
        state["committed"] = True
        return None

    caller = asyncio.create_task(
        start_committed_background_task(
            set(), commit=commit, work=probe.work, backstop_release=probe.backstop
        )
    )
    await commit_entered.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert child_cancelled.is_set()
    assert probe.backstop_calls == 1
    assert not probe.work_started.is_set()


async def test_caller_cancel_after_commit_leaves_child_running():
    probe = _Probe()
    committed = asyncio.Event()
    commit_release = asyncio.Event()

    async def commit(state):
        probe.published = True
        state["committed"] = True
        committed.set()
        await commit_release.wait()
        return None

    tasks: set = set()
    caller = asyncio.create_task(
        start_committed_background_task(
            tasks, commit=commit, work=probe.work, backstop_release=probe.backstop
        )
    )
    await committed.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    commit_release.set()
    probe.work_release.set()
    child = next(iter(tasks))
    await child
    assert probe.work_completed
    assert probe.backstop_calls == 0


async def test_commit_crash_after_commit_still_backstops_reservation():
    probe = _Probe()

    async def commit(state):
        probe.published = True
        state["committed"] = True
        raise RuntimeError("lock release boom")

    with pytest.raises(RuntimeError, match="failed to start"):
        await start_committed_background_task(
            set(), commit=commit, work=probe.work, backstop_release=probe.backstop
        )
    assert probe.backstop_calls == 1
    assert probe.published


# ---------------------------------------------------------------------------
# commit_manual_retry_request
# ---------------------------------------------------------------------------


@pytest.fixture
def share_data():
    finalize_share_data()
    initialize_share_data(workers=1)
    yield
    finalize_share_data()


def _base_status() -> dict:
    return {
        "busy": False,
        "busy_owner": None,
        "scanning": False,
        "scanning_owner": None,
        "scanning_exclusive": False,
        "destructive_busy": False,
        "pending_enqueues": 0,
        "pending_enqueue_tokens": {},
    }


async def test_commit_manual_retry_publishes_and_marks_committed(share_data):
    ingress = await get_pipeline_ingress("wsC")
    state = {"committed": False}
    refusal = await commit_manual_retry_request(
        _base_status(), asyncio.Lock(), ingress, "req-1", state
    )
    assert refusal is None
    assert state["committed"] is True
    assert ingress.peek_next_manual_retry().request_id == "req-1"


async def test_commit_manual_retry_refuses_on_destructive_without_publish(
    share_data,
):
    ingress = await get_pipeline_ingress("wsC")
    status = _base_status()
    status.update({"busy": True, "destructive_busy": True})
    state = {"committed": False}
    refusal = await commit_manual_retry_request(
        status, asyncio.Lock(), ingress, "req-1", state
    )
    assert refusal is not None and "clear/delete" in refusal
    assert state["committed"] is False
    assert not ingress.has_work()

    status = _base_status()
    status["busy"] = True
    status["busy_owner"] = {"token": "t", "kind": "processing", "pid": 1}
    refusal = await commit_manual_retry_request(
        status, asyncio.Lock(), ingress, "req-2", state
    )
    assert refusal is None
    assert ingress.peek_next_manual_retry().request_id == "req-2"


async def test_commit_manual_retry_refuses_terminal_id_without_commit(share_data):
    ingress = await get_pipeline_ingress("wsC")
    ingress.request_manual_retry(
        "req-done",
        PipelineIngressMessage(kind="rescan", retry_failed=True, request_id="req-done"),
    )
    ingress.ack_manual_retry("req-done")

    state = {"committed": False}
    refusal = await commit_manual_retry_request(
        _base_status(), asyncio.Lock(), ingress, "req-done", state
    )
    assert refusal is not None and "already" in refusal.message
    assert refusal.capacity_exceeded is False
    assert state["committed"] is False
    assert not ingress.has_work()


async def test_commit_manual_retry_is_idempotent_for_pending_id(share_data):
    ingress = await get_pipeline_ingress("wsC")
    state = {"committed": False}
    msg = PipelineIngressMessage(kind="rescan", retry_failed=True, request_id="req-1")
    ingress.request_manual_retry("req-1", msg)
    refusal = await commit_manual_retry_request(
        _base_status(), asyncio.Lock(), ingress, "req-1", state
    )
    assert refusal is None
    assert len(ingress.snapshot_manual_retries()) == 1


async def test_commit_manual_retry_reports_capacity_as_retryable():
    ingress = AsyncioPipelineIngress(manual_capacity=1)
    ingress.request_manual_retry(
        "req-first",
        PipelineIngressMessage(
            kind="rescan", retry_failed=True, request_id="req-first"
        ),
    )

    state = {"committed": False}
    refusal = await commit_manual_retry_request(
        _base_status(), asyncio.Lock(), ingress, "req-second", state
    )

    assert refusal is not None
    assert refusal.capacity_exceeded is True
    assert state["committed"] is False
    assert [m.request_id for m in ingress.snapshot_manual_retries()] == ["req-first"]
