"""Shared-storage contracts: update flags, UnifiedLock safety, RPC counts, and write_seq ordering.

Covers:
- ``get_all_update_flags_status`` returning native Python ``bool`` values in
  single-process mode.
- ``UnifiedLock`` factory None-safety checks (``get_internal_lock`` /
  ``get_data_init_lock``) and single-release guarantee when ``async_lock.release()``
  fails in ``__aexit__``.
- Manager-proxy RPC-count reductions (single slice for update-flag updates,
  per-process ``get_namespace_data`` cache, and real-Manager slice + fork
  survival).
- Per-write ordering tokens in ``lightrag.kg.write_seq`` (monotonicity under
  frozen/backward clocks, token precedence over whole seconds, and legacy row
  tie semantics).
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

import lightrag.kg.shared_storage as ss
from lightrag.exceptions import PipelineNotInitializedError
from lightrag.kg import write_seq
from lightrag.kg.shared_storage import (
    UnifiedLock,
    clear_all_update_flags,
    finalize_share_data,
    get_all_update_flags_status,
    get_data_init_lock,
    get_final_namespace,
    get_internal_lock,
    get_namespace_data,
    get_update_flag,
    initialize_pipeline_status,
    initialize_share_data,
    set_all_update_flags,
)
from lightrag.kg.write_seq import (
    WRITE_SEQ_FIELD,
    next_write_seq,
    row_is_strictly_newer,
)

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Counting stand-ins and fixture
# ---------------------------------------------------------------------------


class _CountingValue:
    """ValueProxy-shaped fake counting ``.value`` get/set operations."""

    def __init__(self, initial=False):
        self._v = initial
        self.get_calls = 0
        self.set_calls = 0

    @property
    def value(self):
        self.get_calls += 1
        return self._v

    @value.setter
    def value(self, v):
        self.set_calls += 1
        self._v = v


class _CountingList(list):
    """ListProxy-shaped fake distinguishing slice vs. index ``__getitem__``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slice_getitem_calls = 0
        self.index_getitem_calls = 0
        self.len_calls = 0

    def __getitem__(self, key):
        if isinstance(key, slice):
            self.slice_getitem_calls += 1
        else:
            self.index_getitem_calls += 1
        return super().__getitem__(key)

    def __len__(self):
        self.len_calls += 1
        return super().__len__()


class _CountingDict(dict):
    """DictProxy-shaped fake counting contains/getitem/setitem operations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contains_calls = 0
        self.getitem_calls = 0
        self.setitem_calls = 0

    def __contains__(self, key):
        self.contains_calls += 1
        return super().__contains__(key)

    def __getitem__(self, key):
        self.getitem_calls += 1
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        self.setitem_calls += 1
        return super().__setitem__(key, value)


@pytest.fixture
def single_process_shared_data():
    finalize_share_data()
    initialize_share_data(1)
    yield
    finalize_share_data()


# ---------------------------------------------------------------------------
# get_all_update_flags_status boolean contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_all_update_flags_status_returns_booleans(
    single_process_shared_data,
):
    flag = await get_update_flag("test_ns", workspace="default")
    assert flag.value is False

    status_before = await get_all_update_flags_status(workspace="default")
    assert "default:test_ns" in status_before
    statuses = status_before["default:test_ns"]
    assert len(statuses) == 1
    assert type(statuses[0]) is bool
    assert statuses[0] is False

    await set_all_update_flags("test_ns", workspace="default")
    status_set = await get_all_update_flags_status(workspace="default")
    assert type(status_set["default:test_ns"][0]) is bool
    assert status_set["default:test_ns"][0] is True

    await clear_all_update_flags("test_ns", workspace="default")
    status_cleared = await get_all_update_flags_status(workspace="default")
    assert type(status_cleared["default:test_ns"][0]) is bool
    assert status_cleared["default:test_ns"][0] is False


# ---------------------------------------------------------------------------
# UnifiedLock safety
# ---------------------------------------------------------------------------


class TestUnifiedLockSafety:
    """UnifiedLock factory initialization guards and single-release on failure."""

    def setup_method(self):
        finalize_share_data()

    def teardown_method(self):
        finalize_share_data()

    @pytest.mark.parametrize(
        "getter",
        [
            pytest.param(get_internal_lock, id="internal-lock"),
            pytest.param(get_data_init_lock, id="data-init-lock"),
        ],
    )
    def test_lock_factory_raises_when_not_initialized(self, getter):
        with pytest.raises(
            RuntimeError, match="Shared data not initialized.*initialize_share_data"
        ):
            getter()

    async def test_aexit_no_double_release_on_async_lock_failure(self):
        main_lock = MagicMock()
        main_lock.acquire = MagicMock()
        main_lock.release = MagicMock()

        async_lock = AsyncMock()
        async_lock.acquire = AsyncMock()

        release_call_count = 0

        def mock_release_fail():
            nonlocal release_call_count
            release_call_count += 1
            raise RuntimeError("Async lock release failed")

        async_lock.release = MagicMock(side_effect=mock_release_fail)

        lock = UnifiedLock(
            lock=main_lock,
            is_async=False,
            name="test_double_release",
            enable_logging=False,
        )
        lock._async_lock = async_lock

        with pytest.raises(RuntimeError, match="Async lock release failed"):
            async with lock:
                main_lock.acquire.assert_called_once()

        assert release_call_count == 1
        main_lock.release.assert_called_once()


# ---------------------------------------------------------------------------
# Shared-storage Manager-proxy RPC counts and cache behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_all_update_flags_slices_once(single_process_shared_data):
    ws = "w"
    await get_update_flag("ns", workspace=ws)
    await get_update_flag("ns", workspace=ws)
    await get_update_flag("ns", workspace=ws)

    final_ns = get_final_namespace("ns", ws)
    values = [_CountingValue(), _CountingValue(), _CountingValue()]
    ss._update_flags[final_ns] = _CountingList(values)
    counting_list = ss._update_flags[final_ns]

    await set_all_update_flags("ns", workspace=ws)

    assert counting_list.slice_getitem_calls == 1
    assert counting_list.index_getitem_calls == 0
    assert counting_list.len_calls == 0
    assert all(v.set_calls == 1 for v in values)
    assert all(v._v is True for v in values)


@pytest.mark.asyncio
async def test_clear_all_update_flags_slices_once(single_process_shared_data):
    ws = "w"
    await get_update_flag("ns", workspace=ws)
    await get_update_flag("ns", workspace=ws)

    final_ns = get_final_namespace("ns", ws)
    values = [_CountingValue(True), _CountingValue(True)]
    ss._update_flags[final_ns] = _CountingList(values)
    counting_list = ss._update_flags[final_ns]

    await clear_all_update_flags("ns", workspace=ws)

    assert counting_list.slice_getitem_calls == 1
    assert counting_list.index_getitem_calls == 0
    assert counting_list.len_calls == 0
    assert all(v.set_calls == 1 for v in values)
    assert all(v._v is False for v in values)


@pytest.mark.asyncio
async def test_namespace_cache_hot_hit_skips_shared_dict(single_process_shared_data):
    counting = _CountingDict()
    ss._shared_dicts = counting

    first = await get_namespace_data("ns", workspace="w")
    assert counting.contains_calls == 1
    assert counting.setitem_calls == 1

    counting.contains_calls = 0
    counting.getitem_calls = 0
    counting.setitem_calls = 0

    second = await get_namespace_data("ns", workspace="w")
    assert second is first
    assert counting.contains_calls == 0
    assert counting.getitem_calls == 0
    assert counting.setitem_calls == 0


@pytest.mark.asyncio
async def test_namespace_cache_isolates_workspaces(single_process_shared_data):
    a = await get_namespace_data("ns", workspace="w1")
    b = await get_namespace_data("ns", workspace="w2")
    assert a is not b
    assert ss._namespace_data_cache[get_final_namespace("ns", "w1")] is a
    assert ss._namespace_data_cache[get_final_namespace("ns", "w2")] is b


@pytest.mark.asyncio
async def test_namespace_cache_preserves_pipeline_not_initialized(
    single_process_shared_data,
):
    with pytest.raises(PipelineNotInitializedError):
        await get_namespace_data("pipeline_status", workspace="w")

    await initialize_pipeline_status(workspace="w")
    ps = await get_namespace_data("pipeline_status", workspace="w")
    assert ss._namespace_data_cache[get_final_namespace("pipeline_status", "w")] is ps


@pytest.mark.asyncio
async def test_namespace_cache_rebuilt_after_finalize():
    finalize_share_data()
    initialize_share_data(1)
    try:
        old = await get_namespace_data("ns", workspace="w")
    finally:
        finalize_share_data()
    assert ss._namespace_data_cache is None

    initialize_share_data(1)
    try:
        assert ss._namespace_data_cache == {}
        new = await get_namespace_data("ns", workspace="w")
        assert new is not old
    finally:
        finalize_share_data()


@pytest.mark.asyncio
async def test_history_messages_slice_real_manager():
    finalize_share_data()
    initialize_share_data(2)
    try:
        await initialize_pipeline_status(workspace="w")
        ps = await get_namespace_data("pipeline_status", workspace="w")
        async with ss.get_internal_lock():
            ps["history_messages"].extend([f"line {i}" for i in range(5)])

        snapshot = ps.copy()
        history = snapshot["history_messages"][:]
        assert history == [f"line {i}" for i in range(5)]
        assert isinstance(history, list)
    finally:
        finalize_share_data()


@pytest.mark.skipif(
    not hasattr(os, "fork"), reason="cross-fork cache test requires os.fork"
)
@pytest.mark.filterwarnings("ignore:.*fork.*may lead to deadlocks:DeprecationWarning")
@pytest.mark.asyncio
async def test_namespace_cache_survives_fork_real_manager():
    finalize_share_data()
    initialize_share_data(2)
    try:
        final_ns = get_final_namespace("fork_ns", "w")
        ns = await get_namespace_data("fork_ns", workspace="w")
        ns["seed"] = "parent"
        assert ss._namespace_data_cache[final_ns] is ns

        pid = os.fork()
        if pid == 0:
            code = 0
            try:
                cached = ss._namespace_data_cache[final_ns]
                cached["child_key"] = "child_wrote"
            except Exception:
                code = 1
            finally:
                os._exit(code)

        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0

        parent_ns = await get_namespace_data("fork_ns", workspace="w")
        assert parent_ns.get("child_key") == "child_wrote"
    finally:
        finalize_share_data()


# ---------------------------------------------------------------------------
# Per-write ordering tokens (lightrag.kg.write_seq)
# ---------------------------------------------------------------------------


def _seq_row(created_at: int, seq: int | None = None) -> dict:
    row = {"__created_at__": created_at, "content": "c"}
    if seq is not None:
        row[WRITE_SEQ_FIELD] = seq
    return row


def test_successive_tokens_strictly_increase():
    tokens = [next_write_seq() for _ in range(100)]
    assert tokens == sorted(tokens)
    assert len(set(tokens)) == len(tokens)


def test_a_frozen_or_backwards_clock_still_yields_ordered_tokens(monkeypatch):
    readings = iter([500, 500, 400, 300])
    monkeypatch.setattr(write_seq.time, "time_ns", lambda: next(readings))
    monkeypatch.setattr(write_seq, "_last_seq", 499)

    tokens = [next_write_seq() for _ in range(4)]
    assert tokens == [500, 501, 502, 503]


@pytest.mark.parametrize(
    ("candidate", "current", "expected"),
    [
        pytest.param(
            _seq_row(99, 501), _seq_row(100, 500), True, id="token-beats-clock-step"
        ),
        pytest.param(
            _seq_row(100, 500), _seq_row(99, 501), False, id="older-token-loses"
        ),
        pytest.param(
            _seq_row(101), _seq_row(100, 999), True, id="legacy-newer-second-wins"
        ),
        pytest.param(
            _seq_row(100, 999), _seq_row(101), False, id="legacy-older-second-loses"
        ),
        pytest.param(_seq_row(100, 5), _seq_row(100, 5), False, id="equal-tokens-tie"),
        pytest.param(
            _seq_row(100), _seq_row(100, 5), False, id="missing-token-candidate-tie"
        ),
        pytest.param(
            _seq_row(100, 5), _seq_row(100), False, id="missing-token-current-tie"
        ),
        pytest.param(None, _seq_row(100, 1), False, id="absent-row-never-supersedes"),
    ],
)
def test_row_is_strictly_newer(candidate, current, expected):
    assert row_is_strictly_newer(candidate, current) is expected
