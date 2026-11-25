"""Offline unit tests for RedisKVStorage: iter_rows enumeration and create_time preservation.

Runs against the in-memory FakeRedis stand-in (no live Redis required).
The Lua script's real server-side concurrency races are covered in
``tests/kg/redis_impl/test_redis_kv_create_time_integration.py``.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("redis", reason="redis is required for Redis storage tests")

from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data  # noqa: E402
from lightrag.namespace import NameSpace  # noqa: E402

from .fake_redis import FakeRedis  # noqa: E402

pytestmark = pytest.mark.offline


class _DummyEmbeddingFunc:
    embedding_dim = 1
    max_token_size = 1

    async def __call__(self, texts, **kwargs):
        return [[0.0] for _ in texts]


@pytest.fixture(autouse=True)
def _shared():
    initialize_share_data()
    yield
    finalize_share_data()


@pytest.fixture(autouse=True)
def _reset_eviction_cache():
    from lightrag.kg import redis_impl

    cache = getattr(redis_impl, "_eviction_checked", None)
    if cache is not None:
        cache.clear()
    yield
    if cache is not None:
        cache.clear()


@pytest.fixture
def fake(monkeypatch):
    instance = FakeRedis()
    monkeypatch.setattr(
        "lightrag.kg.redis_impl.RedisConnectionManager.get_pool",
        lambda redis_url: MagicMock(name="fake_pool"),
    )
    monkeypatch.setattr(
        "lightrag.kg.redis_impl.Redis", lambda connection_pool=None, **_: instance
    )
    return instance


def _kv_storage(
    workspace: str = "ct-ws",
    namespace: str = NameSpace.KV_STORE_ENTITY_CHUNKS,
):
    from lightrag.kg.redis_impl import RedisKVStorage

    return RedisKVStorage(
        namespace=namespace,
        global_config={},
        embedding_func=_DummyEmbeddingFunc(),
        workspace=workspace,
    )


def _stored(fake: FakeRedis, storage, key: str) -> dict:
    return json.loads(fake.store[f"{storage.final_namespace}:{key}"])


# ---------------------------------------------------------------------------
# iter_rows enumeration (ConfigurationStorageContract scenario 16)
# ---------------------------------------------------------------------------


async def test_iter_rows_yields_every_row_across_scan_pages(fake):
    storage = _kv_storage(workspace="iterws", namespace="config")
    await storage.initialize()
    await storage.upsert({f"k{i:02d}": {"value": {"n": i}} for i in range(7)})

    # A sibling workspace must not leak in.
    other = _kv_storage(workspace="otherws", namespace="config")
    await other.initialize()
    await other.upsert({"k99": {"value": {"n": 99}}})

    scans_before = (
        len([c for c in fake.calls if c[0] == "scan"])
        if hasattr(fake, "calls")
        else None
    )
    rows = [row async for row in storage.iter_rows(page_size=3)]

    assert sorted(r["_id"] for r in rows) == [f"k{i:02d}" for i in range(7)]
    by_id = {r["_id"]: r for r in rows}
    point = await storage.get_by_id("k04")
    point["_id"] = "k04"
    assert by_id["k04"] == point
    if scans_before is not None:
        assert len([c for c in fake.calls if c[0] == "scan"]) - scans_before >= 3


async def test_iter_rows_empty_namespace_yields_nothing(fake):
    storage = _kv_storage(workspace="iterws", namespace="config")
    await storage.initialize()
    assert [row async for row in storage.iter_rows()] == []


# ---------------------------------------------------------------------------
# Replacement upsert create_time preservation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replacement_upsert_preserves_create_time(fake):
    storage = _kv_storage()
    await storage.initialize()

    with patch("time.time", return_value=1_700_000_000):
        await storage.upsert({"E": {"chunk_ids": ["c1"], "count": 1}})

    created = await storage.get_by_id("E")
    assert created is not None
    assert created["create_time"] == 1_700_000_000
    assert created["update_time"] == 1_700_000_000

    with patch("time.time", return_value=1_700_000_100):
        await storage.upsert({"E": {"chunk_ids": ["c1", "c2"], "count": 2}})

    updated = await storage.get_by_id("E")
    assert updated is not None
    assert updated["create_time"] == 1_700_000_000
    assert updated["update_time"] == 1_700_000_100
    assert updated["chunk_ids"] == ["c1", "c2"]
    assert updated["count"] == 2

    persisted = _stored(fake, storage, "E")
    assert persisted["create_time"] == 1_700_000_000
    assert persisted["update_time"] == 1_700_000_100


@pytest.mark.asyncio
async def test_value_is_serialized_create_time_first(fake):
    """The prefix read only works while create_time leads the value."""
    storage = _kv_storage()
    await storage.initialize()

    with patch("time.time", return_value=1_700_000_000):
        await storage.upsert({"E": {"chunk_ids": ["c1"], "count": 1}})

    raw = fake.store[f"{storage.final_namespace}:E"]
    assert next(iter(json.loads(raw))) == "create_time"
    assert raw.startswith('{"create_time"')


@pytest.mark.asyncio
async def test_update_is_one_script_call_and_never_pulls_the_whole_value(fake):
    """One round trip, a prefix read, no full GET."""
    storage = _kv_storage()
    await storage.initialize()

    with patch("time.time", return_value=1_700_000_000):
        await storage.upsert({"E": {"chunk_ids": ["c1"], "count": 1}})

    fake.command_counts.clear()
    with patch("time.time", return_value=1_700_000_100):
        await storage.upsert({"E": {"chunk_ids": ["c1", "c2"], "count": 2}})

    assert fake.command_counts["script"] == 1
    assert fake.command_counts["getrange"] == 1  # inside the script
    assert fake.command_counts["get"] == 0


@pytest.mark.asyncio
async def test_insert_is_one_script_call(fake):
    """A brand-new key must not trigger the legacy hint round."""
    storage = _kv_storage()
    await storage.initialize()

    fake.command_counts.clear()
    with patch("time.time", return_value=1_700_000_300):
        await storage.upsert({"new": {"chunk_ids": ["c9"], "count": 1}})

    assert fake.command_counts["script"] == 1
    assert fake.command_counts["get"] == 0
    row = await storage.get_by_id("new")
    assert row["create_time"] == 1_700_000_300
    assert row["update_time"] == 1_700_000_300


@pytest.mark.asyncio
async def test_legacy_row_with_trailing_create_time_falls_back_to_full_read(fake):
    """A row written before the ordering existed keeps its real create_time."""
    storage = _kv_storage()
    await storage.initialize()

    key = f"{storage.final_namespace}:legacy"
    fake.store[key] = json.dumps(
        {
            "chunk_ids": ["c1"],
            "count": 1,
            "create_time": 1_650_000_000,
            "update_time": 1_650_000_000,
        }
    )

    fake.command_counts.clear()
    with patch("time.time", return_value=1_700_000_200):
        await storage.upsert({"legacy": {"chunk_ids": ["c1", "c2"], "count": 2}})

    # needs_hint, then a full read, then the write.
    assert fake.command_counts["get"] == 1
    assert fake.command_counts["script"] == 2
    row = await storage.get_by_id("legacy")
    assert row["create_time"] == 1_650_000_000
    assert row["update_time"] == 1_700_000_200

    # Rewritten in the prefix layout, so the next update takes the fast path.
    fake.command_counts.clear()
    with patch("time.time", return_value=1_700_000_400):
        await storage.upsert({"legacy": {"chunk_ids": ["c3"], "count": 1}})
    assert fake.command_counts["get"] == 0
    assert fake.command_counts["script"] == 1
    assert _stored(fake, storage, "legacy")["create_time"] == 1_650_000_000


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("row_id", "stored_payload"),
    [
        pytest.param(
            "legacy",
            {
                "chunk_ids": ["c1"],
                "count": 1,
                "update_time": 1_600_000_000,
                "_id": "legacy",
            },
            id="missing_create_time",
        ),
        pytest.param(
            "inf",
            {"x": 1, "create_time": json.loads("1e309")},
            id="infinite_create_time",
        ),
    ],
)
async def test_legacy_unusable_create_time_falls_back_to_zero(
    fake, row_id, stored_payload
):
    """Legacy rows missing create_time or holding non-finite 1e309 take the 0 fallback."""
    storage = _kv_storage()
    await storage.initialize()

    fake.store[f"{storage.final_namespace}:{row_id}"] = json.dumps(stored_payload)

    with patch("time.time", return_value=1_700_000_200):
        await storage.upsert({row_id: {"chunk_ids": ["c1", "c2"], "count": 2}})

    row = await storage.get_by_id(row_id)
    assert row is not None
    assert row["create_time"] == 0
    assert row["update_time"] == 1_700_000_200
    assert _stored(fake, storage, row_id)["create_time"] == 0


@pytest.mark.asyncio
async def test_legacy_float_create_time_is_normalized(fake):
    """An older release could store time.time() unrounded."""
    storage = _kv_storage()
    await storage.initialize()

    key = f"{storage.final_namespace}:f"
    fake.store[key] = json.dumps({"chunk_ids": ["c1"], "create_time": 1_650_000_000.75})

    with patch("time.time", return_value=1_700_000_200):
        await storage.upsert({"f": {"chunk_ids": ["c2"]}})

    stored = _stored(fake, storage, "f")
    assert stored["create_time"] == 1_650_000_000
    assert isinstance(stored["create_time"], int)


@pytest.mark.asyncio
async def test_empty_value_row_records_zero_not_a_new_timestamp(fake, lightrag_caplog):
    """An empty value is a stored row with no usable timestamp, not an insert."""
    storage = _kv_storage()
    await storage.initialize()

    fake.store[f"{storage.final_namespace}:empty"] = ""

    with lightrag_caplog.at_level(logging.WARNING, logger="lightrag"):
        with patch("time.time", return_value=1_700_000_700):
            await storage.upsert({"empty": {"x": 1}})

    stored = _stored(fake, storage, "empty")
    assert stored["create_time"] == 0
    assert stored["update_time"] == 1_700_000_700
    assert any("not decodable JSON" in r.message for r in lightrag_caplog.records)
    assert fake.command_counts["get"] == 1


@pytest.mark.asyncio
async def test_caller_supplied_create_time_ignored_on_update(fake):
    storage = _kv_storage()
    await storage.initialize()

    with patch("time.time", return_value=1_700_000_000):
        await storage.upsert({"E": {"chunk_ids": ["c1"]}})
    with patch("time.time", return_value=1_700_000_100):
        await storage.upsert({"E": {"chunk_ids": ["c2"], "create_time": 1}})

    assert _stored(fake, storage, "E")["create_time"] == 1_700_000_000


@pytest.mark.asyncio
async def test_legacy_row_deleted_before_the_hint_read_stamps_this_write(fake):
    """A legacy row deleted before the hint read stamps the write's own clock."""
    storage = _kv_storage()
    await storage.initialize()

    key = f"{storage.final_namespace}:gone"
    fake.store[key] = json.dumps({"x": 1, "create_time": 1_650_000_000})

    real_apply = fake._apply

    def apply_and_vanish(op):
        result = real_apply(op)
        if op[0] == "script" and op[1] == key and result[0] == "needs_hint":
            fake.store.pop(key, None)
        return result

    fake._apply = apply_and_vanish

    with patch("time.time", return_value=1_700_000_500):
        await storage.upsert({"gone": {"x": 2}})

    stored = _stored(fake, storage, "gone")
    assert stored["create_time"] == 1_700_000_500
    assert stored["update_time"] == 1_700_000_500


@pytest.mark.asyncio
async def test_the_callers_payload_ends_up_matching_storage(fake):
    """The dict the caller passed in carries what the server decided."""
    storage = _kv_storage()
    await storage.initialize()

    with patch("time.time", return_value=100):
        await storage.upsert({"K": {"x": 1}})

    payload = {"x": 2}
    with patch("time.time", return_value=200):
        await storage.upsert({"K": payload})

    assert payload["create_time"] == 100
    assert payload["update_time"] == 200
    assert payload["create_time"] == _stored(fake, storage, "K")["create_time"]


@pytest.mark.asyncio
async def test_concurrent_updates_agree_without_coordination(fake):
    """Two writers updating an existing row derive the same create_time."""
    worker_a = _kv_storage()
    worker_b = _kv_storage()
    await worker_a.initialize()
    await worker_b.initialize()

    with patch("time.time", return_value=100):
        await worker_a.upsert({"K": {"x": 1}})

    with patch("time.time", return_value=200):
        await worker_a.upsert({"K": {"x": 2}})
        await worker_b.upsert({"K": {"x": 3}})

    row = _stored(fake, worker_b, "K")
    assert row["create_time"] == 100
    assert row["x"] == 3
