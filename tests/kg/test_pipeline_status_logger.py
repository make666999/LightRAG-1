"""Unit tests for ``PipelineStatusLogger`` and the bounded pipeline-history funnel.

Coverage:
- Ring-buffer message count cap and per-message UTF-8 byte cap in
  ``append_pipeline_history`` / ``clamp_pipeline_history_message`` /
  ``PipelineStatusLogger``, plus package-wide guard against raw
  ``["history_messages"].append(...)`` writes.
- History-handle caching, cache recovery on fetch/extend errors, in-place reset
  identity preservation vs. replacement orphaning, and never-raise contract.
- Real Manager ``DictProxy`` / ``ListProxy`` fork inheritance and concurrent
  group integrity.
"""

from __future__ import annotations

import copy
import logging
import multiprocessing as mp
import re
from pathlib import Path

import pytest

from lightrag.constants import (
    PIPELINE_HISTORY_MAX_MESSAGES,
    PIPELINE_HISTORY_MESSAGE_MAX_BYTES,
)
from lightrag.kg import shared_storage
from lightrag.kg.shared_storage import (
    PipelineStatusLogger,
    append_pipeline_history,
    clamp_pipeline_history_message,
)

pytestmark = pytest.mark.offline

_INTERVAL = shared_storage._HISTORY_TRIM_CHECK_INTERVAL


@pytest.fixture(autouse=True)
def _reset_trim_counter():
    """Reset the amortized trim counter between tests."""
    shared_storage._history_appends_since_trim = 0
    yield
    shared_storage._history_appends_since_trim = 0


# ---------------------------------------------------------------------------
# Bounded pipeline status history funnel
# ---------------------------------------------------------------------------


def test_a_writer_that_never_reaches_the_extraction_loop_is_still_bounded():
    history = []
    status = {"history_messages": history}

    total = PIPELINE_HISTORY_MAX_MESSAGES + 3 * _INTERVAL
    for i in range(total):
        append_pipeline_history(status, f"deleting document {i}")

    assert len(history) <= PIPELINE_HISTORY_MAX_MESSAGES + _INTERVAL
    assert history[-1] == f"deleting document {total - 1}"
    assert "deleting document 0" not in history
    assert status["history_messages"] is history


def test_a_group_write_advances_the_bound_by_its_own_length():
    history = []
    status = {"history_messages": history}

    group = 8
    for i in range(0, PIPELINE_HISTORY_MAX_MESSAGES + 3 * _INTERVAL, group):
        append_pipeline_history(status, *[f"msg {i}.{k}" for k in range(group)])

    assert len(history) <= PIPELINE_HISTORY_MAX_MESSAGES + _INTERVAL


def test_a_traceback_sized_message_is_capped():
    history = []
    traceback_text = "Traceback (most recent call last):\n" + (
        '  File "lightrag/pipeline.py", line 1, in _process\n    raise RuntimeError\n'
        * 400
    )
    assert len(traceback_text.encode()) > 5 * PIPELINE_HISTORY_MESSAGE_MAX_BYTES

    append_pipeline_history({"history_messages": history}, traceback_text)

    stored = history[0]
    assert len(stored.encode("utf-8")) <= PIPELINE_HISTORY_MESSAGE_MAX_BYTES
    assert stored.startswith("Traceback (most recent call last):")
    assert f"[truncated, {len(traceback_text.encode())} bytes]" in stored


def test_a_cjk_message_is_cut_on_a_byte_boundary():
    message = "文档解析失败：" + "中" * 5000
    stored = clamp_pipeline_history_message(message)

    assert len(stored.encode("utf-8")) <= PIPELINE_HISTORY_MESSAGE_MAX_BYTES
    assert stored.startswith("文档解析失败：")
    assert "\ufffd" not in stored


def test_a_message_at_the_cap_is_stored_verbatim():
    message = "x" * PIPELINE_HISTORY_MESSAGE_MAX_BYTES
    assert clamp_pipeline_history_message(message) == message
    assert "truncated" not in clamp_pipeline_history_message("short line")


def test_no_raw_history_append_survives_in_the_package():
    package = Path(shared_storage.__file__).resolve().parents[1]
    raw_append = re.compile(r'\["history_messages"\]\s*\.\s*append\(')

    offenders = []
    for path in sorted(package.rglob("*.py")):
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if raw_append.search(line) and not line.lstrip().startswith("#"):
                offenders.append(f"{path.relative_to(package)}:{lineno}")

    assert offenders == [], (
        "route these through append_pipeline_history: " + ", ".join(offenders)
    )


@pytest.mark.parametrize("status", [None, {}], ids=["none-status", "missing-key"])
def test_a_missing_history_is_skipped_not_raised(status):
    before = copy.deepcopy(status)
    append_pipeline_history(status, "message")
    assert status == before


def test_a_failing_extend_is_swallowed():
    class _DeadProxy(list):
        """ListProxy double whose ``extend`` fails as if the Manager died."""

        def extend(self, _iterable):
            raise ConnectionRefusedError("manager is gone")

    history = _DeadProxy()
    append_pipeline_history({"history_messages": history}, "message")
    assert history == []


def test_a_failing_len_does_not_fail_the_append():
    class _UnmeasurableProxy(list):
        """ListProxy double whose ``__len__`` raises during the trim check."""

        def __len__(self):
            raise BrokenPipeError("manager is gone")

    history = _UnmeasurableProxy()
    shared_storage._history_appends_since_trim = _INTERVAL - 1

    append_pipeline_history({"history_messages": history}, "message")
    assert list.__len__(history) == 1


def test_a_non_string_message_is_coerced_not_dropped():
    history = []
    append_pipeline_history({"history_messages": history}, RuntimeError("boom"), 42)
    assert history == ["boom", "42"]


def test_the_lock_free_logger_clamps_and_trims_too():
    history = []
    status = {"history_messages": history}
    status_logger = PipelineStatusLogger(status)

    for i in range(PIPELINE_HISTORY_MAX_MESSAGES + 2 * _INTERVAL):
        status_logger.log(f"entity {i}")
    status_logger.log("失败" * 5000)

    assert len(history) <= PIPELINE_HISTORY_MAX_MESSAGES + _INTERVAL
    assert status["history_messages"] is history
    assert len(history[-1].encode("utf-8")) <= PIPELINE_HISTORY_MESSAGE_MAX_BYTES


def test_latest_message_is_clamped_as_well():
    status = {"history_messages": []}
    PipelineStatusLogger(status).log("x" * (4 * PIPELINE_HISTORY_MESSAGE_MAX_BYTES))
    assert (
        len(status["latest_message"].encode("utf-8"))
        <= PIPELINE_HISTORY_MESSAGE_MAX_BYTES
    )


# ---------------------------------------------------------------------------
# PipelineStatusLogger test doubles & caching / recovery / never-raise tests
# ---------------------------------------------------------------------------


class _CountingStatus:
    """Mapping-shaped double counting ``.get("history_messages")`` calls."""

    def __init__(self, history=()):
        self.get_calls = 0
        self.latest = None
        self.latest_writes = []
        self.history = list(history)
        self.fail_next_gets = 0

    def __setitem__(self, key, value):
        if key == "latest_message":
            self.latest = value
            self.latest_writes.append(value)

    def get(self, key, default=None):
        if key != "history_messages":
            return default
        self.get_calls += 1
        if self.fail_next_gets > 0:
            self.fail_next_gets -= 1
            raise RuntimeError("get boom")
        return self.history


class _FlakyHistory(list):
    """List whose ``extend`` raises for the first ``fail_times`` calls."""

    def __init__(self, *args, fail_times=1):
        super().__init__(*args)
        self.fail_times = fail_times

    def extend(self, iterable):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("extend boom")
        super().extend(iterable)


class _RaisingSetitemStatus(_CountingStatus):
    """Status double whose ``__setitem__`` always raises."""

    def __setitem__(self, key, value):
        raise RuntimeError("setitem boom")


def test_construction_is_fetch_free():
    status = _CountingStatus()
    PipelineStatusLogger(status)
    assert status.get_calls == 0
    assert status.latest_writes == []


def test_first_write_fetches_history_once_then_reuses_cached_handle():
    status = _CountingStatus()
    status_logger = PipelineStatusLogger(status)
    status_logger.log("m0")
    assert status.get_calls == 1
    for i in range(1, 11):
        status_logger.log(f"m{i}")
    assert status.get_calls == 1
    assert status.history == [f"m{i}" for i in range(11)]
    assert status.latest == "m10"


def test_multi_message_log_sets_latest_to_last_and_appends_in_order():
    status = _CountingStatus()
    status_logger = PipelineStatusLogger(status)
    status_logger.log("a", "b", "c")
    assert status.latest == "c"
    assert status.history == ["a", "b", "c"]


def test_get_failure_is_retried_next_write_without_resending_lost_message():
    status = _CountingStatus()
    status.fail_next_gets = 1
    status_logger = PipelineStatusLogger(status)
    status_logger.log("lost")
    assert status.get_calls == 1
    assert status.history == []
    assert status.latest == "lost"
    status_logger.log("kept")
    assert status.get_calls == 2
    assert status.history == ["kept"]
    status_logger.log("more")
    assert status.get_calls == 2


def test_get_returning_none_is_not_cached_and_retried_after_late_init():
    status = _CountingStatus()
    status.history = None
    status_logger = PipelineStatusLogger(status)
    status_logger.log("lost")
    assert status.get_calls == 1
    assert status.latest == "lost"
    status.history = []
    status_logger.log("kept")
    assert status.get_calls == 2
    assert status.history == ["kept"]


def test_extend_failure_drops_cache_and_refetches_without_resending():
    history = _FlakyHistory(fail_times=1)
    status = _CountingStatus()
    status.history = history
    status_logger = PipelineStatusLogger(status)
    status_logger.log("lost")
    assert status.get_calls == 1
    assert list(history) == []
    status_logger.log("kept")
    assert status.get_calls == 2
    assert list(history) == ["kept"]


def test_in_place_clear_keeps_cached_handle_live():
    status = _CountingStatus()
    status_logger = PipelineStatusLogger(status)
    status_logger.log("a")
    del status.history[:]
    status_logger.log("b")
    assert status.history == ["b"]
    assert status.get_calls == 1


def test_in_place_slice_assignment_keeps_cached_handle_live():
    status = _CountingStatus()
    status_logger = PipelineStatusLogger(status)
    status_logger.log("a")
    status.history[:] = ["preset"]
    status_logger.log("c")
    assert status.history == ["preset", "c"]
    assert status.get_calls == 1


def test_replacing_history_object_orphans_the_cache():
    status = _CountingStatus()
    status_logger = PipelineStatusLogger(status)
    status_logger.log("a")
    old_history = status.history
    status.history = []
    status_logger.log("b")
    assert old_history == ["a", "b"]
    assert status.history == []


def test_none_status_and_empty_log_are_noops():
    PipelineStatusLogger(None).log("x")
    status = _CountingStatus()
    PipelineStatusLogger(status).log()
    assert status.get_calls == 0
    assert status.latest is None


def test_never_raises_when_setitem_fails_but_still_appends_history():
    status = _RaisingSetitemStatus()
    PipelineStatusLogger(status).log("a", "b")
    assert status.history == ["a", "b"]


def test_never_raises_even_when_diagnostic_logging_itself_fails(monkeypatch):
    def _boom_debug(*_args, **_kwargs):
        raise RuntimeError("logging subsystem down")

    monkeypatch.setattr(logging.getLogger("lightrag"), "debug", _boom_debug)
    status = _CountingStatus()
    status.fail_next_gets = 1
    PipelineStatusLogger(status).log("a")
    assert status.fail_next_gets == 0
    assert status.latest == "a"
    assert status.history == []


# ---------------------------------------------------------------------------
# Real Manager DictProxy / ListProxy backing path
# ---------------------------------------------------------------------------


def _fork_ctx():
    try:
        return mp.get_context("fork")
    except (ValueError, RuntimeError):
        return None


_FORK = _fork_ctx()
_skip_no_fork = pytest.mark.skipif(
    _FORK is None, reason="fork start method unavailable (Manager proxy sharing test)"
)


def _child_logs(status_logger, message):
    status_logger.log(message)


@_skip_no_fork
@pytest.mark.filterwarnings(
    r"ignore:This process .* is multi-threaded, use of fork\(\) may lead "
    r"to deadlocks in the child\.:DeprecationWarning"
)
def test_fork_inherited_cached_proxy_writes_visible_in_parent():
    mgr = _FORK.Manager()
    try:
        status = mgr.dict({"latest_message": "", "history_messages": mgr.list()})
        status_logger = PipelineStatusLogger(status)
        status_logger.log("parent-1")
        proc = _FORK.Process(target=_child_logs, args=(status_logger, "child-1"))
        proc.start()
        proc.join(timeout=30)
        assert proc.exitcode == 0
        assert list(status["history_messages"]) == ["parent-1", "child-1"]
        assert status["latest_message"] == "child-1"
    finally:
        mgr.shutdown()


def _mp_cached_writer(status, prefix, n):
    status_logger = PipelineStatusLogger(status)
    for i in range(n):
        status_logger.log(f"{prefix}-{i}-begin", f"{prefix}-{i}-end")


@_skip_no_fork
@pytest.mark.filterwarnings(
    r"ignore:This process .* is multi-threaded, use of fork\(\) may lead "
    r"to deadlocks in the child\.:DeprecationWarning"
)
def test_concurrent_cached_writers_lose_no_messages_and_groups_stay_intact():
    mgr = _FORK.Manager()
    try:
        status = mgr.dict({"latest_message": "", "history_messages": mgr.list()})
        n_writers, per_writer = 4, 50
        procs = [
            _FORK.Process(target=_mp_cached_writer, args=(status, f"w{w}", per_writer))
            for w in range(n_writers)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=30)
            assert p.exitcode == 0

        history = list(status["history_messages"])
        assert len(history) == n_writers * per_writer * 2
        expected = {
            f"w{w}-{i}-{end}"
            for w in range(n_writers)
            for i in range(per_writer)
            for end in ("begin", "end")
        }
        assert set(history) == expected
        for idx, msg in enumerate(history):
            if msg.endswith("-begin"):
                assert history[idx + 1] == msg[: -len("-begin")] + "-end", (
                    f"group torn at {idx}: {history[idx : idx + 2]}"
                )
    finally:
        mgr.shutdown()
