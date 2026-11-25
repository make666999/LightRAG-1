"""Offline tests for the storage-IO offload helpers.

The file backends used to rewrite their whole file synchronously on the event
loop. ``run_in_storage_io`` moves that write to a single-worker pool, which
introduces a window the synchronous code did not have: the caller holds its
namespace lock across the write, so a cancelled caller must NOT return while the
worker is still touching shared state.

Pinned here:

* once the work is submitted, the caller waits for it — through repeated
  cancellation, which a lone ``asyncio.shield`` does not survive;
* while merely waiting for a submission permit, the caller stays cancellable —
  nothing is in flight yet, and refusing to return there would keep the caller's
  lock held for the whole queue;
* splitting the body out of ``bounded_submit`` did not change its forwarding
  contract: it still declares no keyword arguments of its own;
* ``commit_in_storage_io`` keeps a landed write and its bookkeeping together:
  the hook runs inside the same uncancellable region, and only if the write
  actually happened;
* the three outcomes stay distinguishable: ``fn`` raising propagates as itself
  (nothing persisted), a failing hook becomes ``CommitBookkeepingError`` (the
  write is durable, only its publication failed), and a cancelled caller still
  gets ``CancelledError``.
"""

import asyncio
import itertools
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from lightrag import utils as lr_utils
from lightrag.exceptions import CommitBookkeepingError
from lightrag.utils import (
    _bounded_submit_impl,
    _wait_deferring_cancellation,
    bounded_submit,
    cancellation_was_deferred,
    commit_in_storage_io,
    get_storage_io_executor,
    mark_cancellation_deferred,
    retract_cancellation_deferred,
    run_in_storage_io,
)
from tests._log_capture import capture_lightrag_logs

pytestmark = pytest.mark.offline


class _CountingExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor that records how many submissions it accepted."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submissions = 0

    def submit(self, *args, **kwargs):
        self.submissions += 1
        return super().submit(*args, **kwargs)


class _BlockingWork:
    """A callable that parks in the worker thread until released."""

    def __init__(self, result="value", exc=None):
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished_at = None
        self._result = result
        self._exc = exc

    def __call__(self):
        self.started.set()
        assert self.release.wait(timeout=5), "work was never released"
        self.finished_at = next(_ORDER)
        if self._exc is not None:
            raise self._exc
        return self._result


_ORDER = itertools.count()


async def _wait_for(predicate, *, timeout=2.0):
    """Await a thread-set condition without blocking the loop."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        assert loop.time() < deadline, "timed out waiting for condition"
        await asyncio.sleep(0.01)


async def test_cancelled_caller_waits_for_the_submitted_worker():
    """A cancelled caller must not return while its worker is still running.

    Fix-proof: with the pre-change semantics (``wait_for_completion=False``,
    i.e. a bare ``bounded_submit``) the task completes as soon as it is
    cancelled, so ``task.done()`` is True before the worker is released and the
    ordering assertion at the end sees the caller return first.
    """
    work = _BlockingWork()
    task = asyncio.create_task(run_in_storage_io(work))

    await _wait_for(work.started.is_set)
    task.cancel()
    # Give the cancellation every chance to land: if the caller were allowed to
    # return early it would already be done here.
    for _ in range(10):
        await asyncio.sleep(0)
    assert not task.done(), "caller returned while the worker was still running"

    work.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    returned_at = next(_ORDER)

    assert work.finished_at is not None
    assert work.finished_at < returned_at


async def test_repeated_cancellation_does_not_release_the_caller(monkeypatch):
    """Three cancellations, each proven to land in a distinct shield round.

    A lone ``await asyncio.shield(fut)`` returns immediately when the awaiting
    coroutine is cancelled again, so an implementation that catches
    ``CancelledError`` once and shields once would escape on the second cancel.
    Sleeping between cancels cannot prove they landed in different rounds — the
    scheduler is free to collapse them — so each cancel here waits for the spy
    to observe the NEXT entry into ``asyncio.shield``.
    """
    entries = itertools.count(1)
    seen = {}
    real_shield = asyncio.shield

    def spy_shield(awaitable, **kwargs):
        seen.setdefault(next(entries), asyncio.Event()).set()
        return real_shield(awaitable, **kwargs)

    def entered(n):
        return seen.setdefault(n, asyncio.Event()).is_set()

    monkeypatch.setattr(asyncio, "shield", spy_shield)

    work = _BlockingWork()
    task = asyncio.create_task(run_in_storage_io(work))

    await _wait_for(work.started.is_set)
    for round_number in range(1, 4):
        # Round N's shield is already active; cancel it and require the helper
        # to re-enter shield for round N+1 before cancelling again.
        await _wait_for(lambda n=round_number: entered(n))
        task.cancel()
        await _wait_for(lambda n=round_number: entered(n + 1))
        assert not task.done(), "caller escaped after a repeated cancellation"

    work.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert work.finished_at is not None


async def test_worker_exception_propagates_when_not_cancelled():
    boom = OSError("disk full")
    work = _BlockingWork(exc=boom)
    task = asyncio.create_task(run_in_storage_io(work))

    await _wait_for(work.started.is_set)
    work.release.set()

    with pytest.raises(OSError, match="disk full"):
        await task


async def test_cancellation_wins_over_a_worker_exception(caplog, monkeypatch):
    """Cancel + IO failure: CancelledError wins, the failure is still logged.

    The worker's exception can never reach the caller once the caller is being
    cancelled, so it must not vanish silently, and it must not resurface as an
    "exception was never retrieved" callback either.

    The cancellation must be observed by the helper BEFORE the worker fails,
    otherwise this races: a worker that finishes first ends the wait loop with
    no deferred cancellation and the caller legitimately sees the OSError. The
    shield spy below pins the ordering instead of hoping for it.
    """
    handled = []
    asyncio.get_running_loop().set_exception_handler(
        lambda _loop, context: handled.append(context)
    )

    entries = itertools.count(1)
    seen = {}
    real_shield = asyncio.shield

    def spy_shield(awaitable, **kwargs):
        seen.setdefault(next(entries), asyncio.Event()).set()
        return real_shield(awaitable, **kwargs)

    monkeypatch.setattr(asyncio, "shield", spy_shield)

    work = _BlockingWork(exc=OSError("disk full"))
    task = asyncio.create_task(run_in_storage_io(work))

    await _wait_for(work.started.is_set)
    await _wait_for(lambda: seen.setdefault(1, asyncio.Event()).is_set())
    task.cancel()
    # Re-entering shield proves the cancellation was caught and deferred.
    await _wait_for(lambda: seen.setdefault(2, asyncio.Event()).is_set())
    work.release.set()

    with capture_lightrag_logs(caplog):
        with caplog.at_level(logging.ERROR, logger="lightrag"):
            with pytest.raises(asyncio.CancelledError):
                await task

    assert any("disk full" in record.getMessage() for record in caplog.records)

    # Force the finalizer that would report an unretrieved exception.
    del work, task
    for _ in range(3):
        await asyncio.sleep(0)
    assert not any(
        "never retrieved" in str(context.get("message", "")) for context in handled
    )


async def test_cancellation_while_waiting_for_a_permit_never_submits():
    """Waiting for a permit stays cancellable, and cancels before submitting.

    Nothing is in flight while a caller waits for a permit, so refusing to
    return there would keep its storage lock held for the whole queue ahead of
    it — and would still submit the work afterwards, producing a disk write
    after the cancellation.
    """
    executor = _CountingExecutor(max_workers=1)
    semaphore = asyncio.Semaphore(1)
    holder = _BlockingWork()

    try:
        held = asyncio.create_task(
            _bounded_submit_impl(
                executor, semaphore, holder, (), {}, wait_for_completion=True
            )
        )
        await _wait_for(holder.started.is_set)
        assert executor.submissions == 1

        queued = _BlockingWork()
        waiting = asyncio.create_task(
            _bounded_submit_impl(
                executor, semaphore, queued, (), {}, wait_for_completion=True
            )
        )
        await _wait_for(lambda: semaphore.locked())
        for _ in range(10):
            await asyncio.sleep(0)

        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting

        assert executor.submissions == 1, "cancelled caller still submitted work"
        assert not queued.started.is_set()

        holder.release.set()
        await held
    finally:
        holder.release.set()
        executor.shutdown(wait=True)


async def test_bounded_submit_still_forwards_every_keyword_to_fn():
    """``bounded_submit`` declares no keyword arguments of its own.

    The control flag lives on the private ``_bounded_submit_impl`` precisely so
    that a target function may use the name ``wait_for_completion`` itself. Put
    the flag on ``bounded_submit`` after ``*args`` and this fails: the value is
    swallowed as the control flag instead of reaching ``fn``.
    """
    executor = ThreadPoolExecutor(max_workers=1)
    semaphore = asyncio.Semaphore(1)

    def target(*, wait_for_completion):
        return wait_for_completion

    try:
        result = await bounded_submit(
            executor, semaphore, target, wait_for_completion="payload"
        )
    finally:
        executor.shutdown(wait=True)

    assert result == "payload"


def test_storage_io_executor_is_a_single_worker_named_pool():
    """Configuration pin: the pool's shape is load-bearing, not incidental.

    One worker keeps the commits as serialized as they are today; the name makes
    the thread identifiable in a stack dump taken during a long write.
    """
    executor = get_storage_io_executor()

    assert executor is get_storage_io_executor()
    assert executor._max_workers == 1
    assert executor._thread_name_prefix == "lightrag-storage-io"
    assert lr_utils._STORAGE_IO_EXECUTOR is executor


# ---------------------------------------------------------------------------
# commit_in_storage_io: a landed write and its bookkeeping must not come apart
# ---------------------------------------------------------------------------


async def test_commit_hook_completes_before_a_deferred_cancellation():
    """Cancelled mid-write, the bookkeeping still runs — before CancelledError.

    This is the case Codex flagged on #3740: the GraphML file is already
    published, so skipping ``set_all_update_flags`` leaves every other worker
    reading the previous snapshot indefinitely.

    Fix-proof: route the same work through ``run_in_storage_io`` and inline the
    hook after it, and the hook never runs at all — the ordering list ends at
    ["write", "cancelled"].
    """
    order = []
    work = _BlockingWork()

    async def on_committed():
        # An await here on purpose: a second cancel would otherwise land on it,
        # which is exactly what the deferring wait has to absorb.
        await asyncio.sleep(0)
        order.append("committed")

    task = asyncio.create_task(commit_in_storage_io(work, on_committed))

    await _wait_for(work.started.is_set)
    task.cancel()
    work.release.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    order.append("cancelled")

    assert order == ["committed", "cancelled"], order


async def test_commit_hook_does_not_run_when_waiting_for_a_permit_is_cancelled():
    """No submission means no write, so the bookkeeping must NOT run.

    Load-bearing rather than tidiness: Nano's hook retires the redo logs. Running
    it without a write would discard rows that were never persisted.
    """
    executor = _CountingExecutor(max_workers=1)
    semaphore = asyncio.Semaphore(1)
    holder = _BlockingWork()
    hook_ran = []

    async def on_committed():
        hook_ran.append(1)

    try:
        held = asyncio.create_task(
            _bounded_submit_impl(
                executor,
                semaphore,
                holder,
                (),
                {},
                wait_for_completion=True,
                on_committed=on_committed,
            )
        )
        await _wait_for(holder.started.is_set)

        queued = _BlockingWork()
        waiting = asyncio.create_task(
            _bounded_submit_impl(
                executor,
                semaphore,
                queued,
                (),
                {},
                wait_for_completion=True,
                on_committed=on_committed,
            )
        )
        await _wait_for(lambda: semaphore.locked())
        for _ in range(10):
            await asyncio.sleep(0)

        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting

        assert executor.submissions == 1
        assert not queued.started.is_set()
        assert hook_ran == [], "bookkeeping ran for a write that never happened"

        holder.release.set()
        await held
        assert hook_ran == [1], "the write that DID land must run its bookkeeping"
    finally:
        holder.release.set()
        executor.shutdown(wait=True)


async def test_commit_hook_does_not_run_when_the_write_raises():
    """A failed write must not have its bookkeeping applied."""
    hook_ran = []

    async def on_committed():
        hook_ran.append(1)

    work = _BlockingWork(exc=OSError("disk full"))
    task = asyncio.create_task(commit_in_storage_io(work, on_committed))

    await _wait_for(work.started.is_set)
    work.release.set()

    with pytest.raises(OSError, match="disk full"):
        await task
    assert hook_ran == []


async def test_commit_hook_failure_surfaces_when_not_cancelled():
    """Bookkeeping that fails must be reported, not swallowed.

    The write landed but the namespace was never flagged, so the caller has to
    hear about it. It hears about it as ``CommitBookkeepingError``, never as the
    hook's own exception: re-raising that made "the write is on disk and its
    publication failed" indistinguishable from "nothing was persisted", and
    every call site's ``except Exception`` then handled it as the latter —
    rolling in-memory state back to a file that already has the mutation, and
    reporting a durable write as one that never happened.
    """

    cause = RuntimeError("flag update failed")

    async def on_committed():
        raise cause

    work = _BlockingWork(result="written")
    work.release.set()

    with pytest.raises(CommitBookkeepingError) as excinfo:
        await commit_in_storage_io(work, on_committed)

    assert excinfo.value.__cause__ is cause
    assert "flag update failed" in str(excinfo.value)
    # The write's own answer travels with it, so a handler that needs it does
    # not have to re-run the write to find out.
    assert excinfo.value.result == "written"


async def test_a_failed_write_is_not_reported_as_a_landed_one():
    """The contrast that gives the typed error its meaning.

    ``fn`` raising means nothing was persisted, and that must keep arriving as
    the backend's own exception — a handler that treats it as
    ``CommitBookkeepingError`` would skip the rollback the failure needs.
    """
    hook_ran = []

    async def on_committed():  # pragma: no cover — must not run
        hook_ran.append(1)

    work = _BlockingWork(exc=OSError("disk full"))
    work.release.set()

    with pytest.raises(OSError) as excinfo:
        await commit_in_storage_io(work, on_committed)

    assert not isinstance(excinfo.value, CommitBookkeepingError)
    assert hook_ran == []


async def test_a_cancelled_caller_still_gets_cancelled_not_the_typed_error():
    """R2 covers the hook-failure path only; cancellation is unchanged.

    A deferred cancellation outranks the hook's failure, because a caller that
    was cancelled must not be told a storage error happened instead. The
    deletion paths in ``utils_graph`` depend on this: they give up their
    tracking cleanup on ``CancelledError`` (it carries no evidence about what
    landed) and proceed on ``CommitBookkeepingError`` (which does).
    """

    async def on_committed():
        raise RuntimeError("flag update failed")

    work = _BlockingWork()
    task = asyncio.create_task(commit_in_storage_io(work, on_committed))

    await _wait_for(work.started.is_set)
    task.cancel()
    work.release.set()

    with pytest.raises(asyncio.CancelledError):
        await task


async def test_on_committed_requires_wait_for_completion():
    """The hook is meaningless without the uncancellable wait — reject it."""
    executor = ThreadPoolExecutor(max_workers=1)
    semaphore = asyncio.Semaphore(1)

    async def on_committed():  # pragma: no cover — never reached
        raise AssertionError("must not run")

    try:
        with pytest.raises(ValueError, match="wait_for_completion"):
            await _bounded_submit_impl(
                executor,
                semaphore,
                lambda: None,
                (),
                {},
                wait_for_completion=False,
                on_committed=on_committed,
            )
    finally:
        executor.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Deferred cancellation durability stamp
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_withheld_cancellation_is_stamped():
    """The caller is cancelled mid-region; the region still finishes, and the
    cancellation handed back says so."""
    started = asyncio.Event()
    release = asyncio.Event()
    landed = []

    async def _uninterruptible():
        started.set()
        await release.wait()
        landed.append("committed")

    async def _region():
        future = asyncio.ensure_future(_uninterruptible())
        return await _wait_deferring_cancellation(future, None)

    task = asyncio.ensure_future(_region())
    await started.wait()
    task.cancel()
    for _ in range(3):
        await asyncio.sleep(0)
    assert not task.done()
    release.set()
    pending = await task

    assert landed == ["committed"]
    assert isinstance(pending, asyncio.CancelledError)
    assert cancellation_was_deferred(pending) is True


@pytest.mark.asyncio
async def test_work_that_FAILED_is_not_stamped():
    """The stamp claims success, not merely that the region ran."""
    started = asyncio.Event()
    release = asyncio.Event()

    async def _fails():
        started.set()
        await release.wait()
        raise OSError(28, "No space left on device")

    async def _region():
        future = asyncio.ensure_future(_fails())
        return await _wait_deferring_cancellation(future, None)

    task = asyncio.ensure_future(_region())
    await started.wait()
    task.cancel()
    for _ in range(3):
        await asyncio.sleep(0)
    assert not task.done()
    release.set()
    pending = await task

    assert isinstance(pending, asyncio.CancelledError)
    assert cancellation_was_deferred(pending) is False


@pytest.mark.asyncio
async def test_a_stamp_from_an_earlier_step_survives_a_later_failure():
    """A write that succeeded IS durable, so a failing hook afterwards must
    not retract the stamp."""
    write_started = asyncio.Event()
    write_release = asyncio.Event()
    hook_started = asyncio.Event()
    hook_release = asyncio.Event()

    async def _ok():
        write_started.set()
        await write_release.wait()

    async def _fails():
        hook_started.set()
        await hook_release.wait()
        raise RuntimeError("publication failed")

    async def _region():
        write = asyncio.ensure_future(_ok())
        pending = await _wait_deferring_cancellation(write, None)
        hook = asyncio.ensure_future(_fails())
        return await _wait_deferring_cancellation(hook, pending)

    task = asyncio.ensure_future(_region())
    await write_started.wait()
    task.cancel()
    for _ in range(3):
        await asyncio.sleep(0)
    write_release.set()
    await hook_started.wait()
    hook_release.set()
    pending = await task

    assert cancellation_was_deferred(pending) is True


@pytest.mark.asyncio
async def test_no_cancellation_means_no_stamp():
    """Nothing was withheld, so there is nothing to warn a caller about."""

    async def _work():
        return "done"

    future = asyncio.ensure_future(_work())
    assert await _wait_deferring_cancellation(future, None) is None


@pytest.mark.asyncio
async def test_the_stamp_survives_the_re_raise_the_caller_does():
    """The stamp rides the exception object itself across the re-raise."""
    started = asyncio.Event()
    release = asyncio.Event()

    async def _uninterruptible():
        started.set()
        await release.wait()

    async def _region():
        future = asyncio.ensure_future(_uninterruptible())
        pending = await _wait_deferring_cancellation(future, None)
        if pending is not None:
            raise pending

    task = asyncio.ensure_future(_region())
    await started.wait()
    task.cancel()
    for _ in range(3):
        await asyncio.sleep(0)
    release.set()

    with pytest.raises(asyncio.CancelledError) as excinfo:
        await task
    assert cancellation_was_deferred(excinfo.value) is True


@pytest.mark.asyncio
async def test_a_plain_cancellation_at_a_suspension_point_is_not_stamped():
    """A cancellation delivered at an ordinary await is not stamped."""

    async def _plain():
        await asyncio.sleep(3600)

    task = asyncio.ensure_future(_plain())
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError) as excinfo:
        await task
    assert cancellation_was_deferred(excinfo.value) is False


def test_cancellation_was_deferred_is_false_for_anything_else():
    assert cancellation_was_deferred(RuntimeError("boom")) is False
    assert cancellation_was_deferred(asyncio.CancelledError()) is False


async def _run_bounded(fn, on_committed):
    """Submit ``fn`` through ``_bounded_submit_impl`` and cancel the caller
    once the commit hook has started -- the window the stamp has to cover."""
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        semaphore = asyncio.Semaphore(1)
        hook_started = asyncio.Event()
        hook_release = asyncio.Event()

        async def _hook():
            hook_started.set()
            await hook_release.wait()
            return await on_committed()

        task = asyncio.ensure_future(
            _bounded_submit_impl(
                executor,
                semaphore,
                fn,
                (),
                {},
                wait_for_completion=True,
                on_committed=_hook,
            )
        )
        await hook_started.wait()
        task.cancel()
        for _ in range(3):
            await asyncio.sleep(0)
        hook_release.set()
        return task
    finally:
        executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_a_write_that_landed_is_stamped_even_when_its_commit_hook_fails():
    """The write is on disk; only its publication failed."""
    written = []

    def _write():
        written.append("landed")
        return "result"

    async def _publish():
        raise RuntimeError("could not flip the reload flags")

    task = await _run_bounded(_write, _publish)

    with pytest.raises(asyncio.CancelledError) as excinfo:
        await task
    assert written == ["landed"]
    assert cancellation_was_deferred(excinfo.value) is True


@pytest.mark.asyncio
async def test_a_write_that_RAISED_is_still_not_stamped():
    """``fn`` raising means nothing was persisted, the hook never runs, and
    the cancellation must carry no claim of durability."""
    published: list[int] = []

    def _write():
        raise OSError(28, "No space left on device")

    async def _publish():  # pragma: no cover - must never run
        published.append(1)
        raise AssertionError("the commit hook must not run after a failed write")

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        semaphore = asyncio.Semaphore(1)

        async def _region():
            future = asyncio.ensure_future(
                _bounded_submit_impl(
                    executor,
                    semaphore,
                    _write,
                    (),
                    {},
                    wait_for_completion=True,
                    on_committed=_publish,
                )
            )
            await asyncio.sleep(0)
            return future

        task = await _region()
        task.cancel()
        with pytest.raises(BaseException) as excinfo:
            await task
    finally:
        executor.shutdown(wait=True)

    assert published == []
    assert cancellation_was_deferred(excinfo.value) is False
    assert not isinstance(excinfo.value, CommitBookkeepingError)


def test_a_rolled_back_write_retracts_the_stamp():
    """A caller that undid the deferred write withdraws the durability claim."""
    cancel = asyncio.CancelledError()
    mark_cancellation_deferred(cancel)
    assert cancellation_was_deferred(cancel)

    retract_cancellation_deferred(cancel)

    assert not cancellation_was_deferred(cancel)


@pytest.mark.parametrize(
    "exc", [asyncio.CancelledError(), ValueError("boom")], ids=["unstamped", "other"]
)
def test_retracting_leaves_an_unstamped_exception_untouched(exc):
    before = dict(vars(exc))

    retract_cancellation_deferred(exc)

    assert vars(exc) == before
    assert not cancellation_was_deferred(exc)
