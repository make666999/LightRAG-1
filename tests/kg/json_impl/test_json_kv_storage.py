"""Tests for JsonKVStorage: copy-on-read, corrupt records, iter_rows, create_time, and config identity."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from lightrag import config_anchor as ca
from lightrag import config_store as cs
from lightrag.exceptions import (
    ConfigurationStorageError,
    CorruptStorageRecordError,
    StorageNotInitializedError,
)
from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.namespace import CONFIG_JSON_FILE_NAME, NameSpace
from lightrag.utils import update_chunk_cache_list

pytestmark = pytest.mark.offline


class _DummyEmbeddingFunc:
    """Minimal embedding stand-in for storages that do not embed."""

    embedding_dim = 1
    max_token_size = 1

    async def __call__(self, texts, **kwargs):
        return [[0.0] for _ in texts]


class _CountingDict(dict):
    """Stand-in for the multiprocess ``Manager().dict()`` proxy counting lookups."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = {"get": 0, "getitem": 0, "contains": 0}

    def get(self, key, default=None):
        self.calls["get"] += 1
        return super().get(key, default)

    def __getitem__(self, key):
        self.calls["getitem"] += 1
        return super().__getitem__(key)

    def __contains__(self, key):
        self.calls["contains"] += 1
        return super().__contains__(key)


@pytest.fixture(autouse=True)
def setup_shared_data():
    initialize_share_data(workers=1)
    yield
    finalize_share_data()


async def _make_kv(
    tmp_path,
    *,
    namespace: str = NameSpace.KV_STORE_TEXT_CHUNKS,
    workspace: str = "test",
    initialize: bool = True,
) -> JsonKVStorage:
    storage = JsonKVStorage(
        namespace=namespace,
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace=workspace,
    )
    if initialize:
        await storage.initialize()
    return storage


async def _storage_over(tmp_path, payload: dict) -> JsonKVStorage:
    """A storage whose file already holds ``payload``, written behind its back."""
    workspace_dir = tmp_path / "ws"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / f"kv_store_{NameSpace.KV_STORE_TEXT_CHUNKS}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return await _make_kv(tmp_path, workspace="ws")


# ---------------------------------------------------------------------------
# Copy-on-read and file-path preservation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_by_id_returns_deep_copy(tmp_path):
    storage = await _make_kv(tmp_path)
    await storage.upsert(
        {
            "chunk1": {
                "content": "hello",
                "llm_cache_list": ["cache-a"],
                "metadata": {"key": "original"},
            }
        }
    )

    doc = await storage.get_by_id("chunk1")
    assert doc is not None
    doc["llm_cache_list"].append("cache-b")
    doc["metadata"]["key"] = "modified_nested"

    stored = await storage.get_by_id("chunk1")
    assert stored is not None
    assert stored["llm_cache_list"] == ["cache-a"]
    assert stored["metadata"]["key"] == "original"


@pytest.mark.asyncio
async def test_get_by_ids_returns_deep_copy(tmp_path):
    storage = await _make_kv(tmp_path)
    await storage.upsert(
        {
            "chunk2": {
                "content": "hello",
                "llm_cache_list": ["cache-a"],
                "metadata": {"key": "original"},
            }
        }
    )

    docs = await storage.get_by_ids(["chunk2"])
    assert len(docs) == 1 and docs[0] is not None
    docs[0]["llm_cache_list"].append("cache-b")
    docs[0]["metadata"]["key"] = "modified_nested"

    stored = await storage.get_by_id("chunk2")
    assert stored is not None
    assert stored["llm_cache_list"] == ["cache-a"]
    assert stored["metadata"]["key"] == "original"


@pytest.mark.asyncio
async def test_update_chunk_cache_list_does_not_corrupt_storage_on_read(tmp_path):
    """A snapshot taken before update_chunk_cache_list stays frozen (no aliasing)."""
    storage = await _make_kv(tmp_path)
    await storage.upsert(
        {"chunk3": {"content": "hello", "llm_cache_list": ["cache-a"]}}
    )

    snapshot = await storage.get_by_id("chunk3")
    await update_chunk_cache_list("chunk3", storage, ["cache-b"])

    assert snapshot["llm_cache_list"] == ["cache-a"]
    stored = await storage.get_by_id("chunk3")
    assert stored is not None
    assert stored["llm_cache_list"] == ["cache-a", "cache-b"]


@pytest.mark.asyncio
async def test_json_kv_upsert_preserves_caller_file_paths(tmp_path):
    full_docs = await _make_kv(tmp_path, namespace=NameSpace.KV_STORE_FULL_DOCS)
    text_chunks = await _make_kv(tmp_path, namespace=NameSpace.KV_STORE_TEXT_CHUNKS)

    await full_docs.upsert(
        {
            "doc-1": {
                "content": "full text",
                "file_path": "/tmp/uploads/report.[native-Fi].pdf",
            }
        }
    )
    await text_chunks.upsert(
        {
            "chunk-1": {
                "content": "chunk text",
                "tokens": 2,
                "chunk_order_index": 0,
                "full_doc_id": "doc-1",
                "file_path": "/tmp/uploads/report.[native-Fi].pdf",
            }
        }
    )

    assert (await full_docs.get_by_id("doc-1"))["file_path"] == (
        "/tmp/uploads/report.[native-Fi].pdf"
    )
    assert (await text_chunks.get_by_id("chunk-1"))["file_path"] == (
        "/tmp/uploads/report.[native-Fi].pdf"
    )


# ---------------------------------------------------------------------------
# Corrupt record handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_by_id_names_the_damaged_record(tmp_path):
    storage = await _storage_over(
        tmp_path, {"chunk-1": "garbage", "chunk-2": {"content": "fine"}}
    )

    with pytest.raises(CorruptStorageRecordError) as excinfo:
        await storage.get_by_id("chunk-1")

    message = str(excinfo.value)
    assert "chunk-1" in message, "the damaged key must be named"
    assert "str" in message and "not a mapping" in message
    assert (await storage.get_by_id("chunk-2"))["content"] == "fine"


@pytest.mark.asyncio
async def test_get_by_ids_names_the_damaged_record(tmp_path):
    storage = await _storage_over(tmp_path, {"chunk-1": ["not", "a", "row"]})

    with pytest.raises(CorruptStorageRecordError):
        await storage.get_by_ids(["chunk-1"])


@pytest.mark.asyncio
async def test_iter_rows_raises_rather_than_yielding_a_damaged_record(tmp_path):
    storage = await _storage_over(tmp_path, {"chunk-1": 42})

    with pytest.raises(CorruptStorageRecordError):
        async for _row in storage.iter_rows(page_size=10):
            pass


@pytest.mark.asyncio
async def test_a_missing_key_is_still_a_confirmed_absence(tmp_path):
    storage = await _storage_over(tmp_path, {"chunk-1": {"content": "fine"}})

    assert await storage.get_by_id("nope") is None
    assert await storage.get_by_id_strict("nope") is None


# ---------------------------------------------------------------------------
# iter_rows enumeration surface
# ---------------------------------------------------------------------------


async def test_every_row_is_yielded_across_pages_with_the_point_read_shape(tmp_path):
    storage = await _make_kv(tmp_path, namespace="config", workspace="iterws")
    await storage.upsert(
        {f"k{i:02d}": {"value": {"n": i}, "nested": [i]} for i in range(7)}
    )

    rows = [row async for row in storage.iter_rows(page_size=3)]

    assert sorted(r["_id"] for r in rows) == [f"k{i:02d}" for i in range(7)]
    by_id = {r["_id"]: r for r in rows}
    point = await storage.get_by_id("k03")
    assert by_id["k03"] == point
    by_id["k03"]["nested"].append("x")
    assert (await storage.get_by_id("k03"))["nested"] == [3]


async def test_an_empty_namespace_yields_nothing(tmp_path):
    storage = await _make_kv(tmp_path, namespace="config", workspace="iterws")
    assert [row async for row in storage.iter_rows()] == []


async def test_a_row_deleted_mid_scan_is_skipped(tmp_path):
    storage = await _make_kv(tmp_path, namespace="config", workspace="iterws")
    await storage.upsert({f"k{i}": {"value": {}} for i in range(4)})

    seen = []
    async for row in storage.iter_rows(page_size=2):
        seen.append(row["_id"])
        if len(seen) == 1:
            await storage.delete(["k3"])

    assert "k3" not in seen
    assert len(seen) == 3


async def test_an_uninitialized_storage_raises_instead_of_yielding_nothing(tmp_path):
    storage = await _make_kv(
        tmp_path,
        namespace="config",
        workspace="never-initialized",
        initialize=False,
    )
    with pytest.raises(StorageNotInitializedError):
        async for _ in storage.iter_rows():
            pass


# ---------------------------------------------------------------------------
# Replacement upsert create_time preservation & lookup count
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replacement_upsert_preserves_create_time(tmp_path):
    storage = await _make_kv(
        tmp_path, namespace=NameSpace.KV_STORE_ENTITY_CHUNKS, workspace="ct-ws"
    )

    with patch("time.time", return_value=1_700_000_000):
        await storage.upsert({"E": {"chunk_ids": ["c1"], "count": 1}})

    created = await storage.get_by_id("E")
    assert created is not None
    assert created["create_time"] == 1_700_000_000
    assert created["update_time"] == 1_700_000_000
    assert created["chunk_ids"] == ["c1"]
    assert created["count"] == 1

    with patch("time.time", return_value=1_700_000_100):
        await storage.upsert({"E": {"chunk_ids": ["c1", "c2"], "count": 2}})

    updated = await storage.get_by_id("E")
    assert updated is not None
    assert updated["create_time"] == 1_700_000_000
    assert updated["update_time"] == 1_700_000_100
    assert updated["chunk_ids"] == ["c1", "c2"]
    assert updated["count"] == 2
    assert "llm_cache_list" not in updated

    await storage.index_done_callback()
    persisted = storage._data["E"]
    assert persisted["create_time"] == 1_700_000_000
    assert persisted["update_time"] == 1_700_000_100


@pytest.mark.asyncio
async def test_legacy_row_missing_create_time_keeps_zero(tmp_path):
    storage = await _make_kv(
        tmp_path, namespace=NameSpace.KV_STORE_ENTITY_CHUNKS, workspace="ct-ws"
    )
    storage._data["legacy"] = {
        "chunk_ids": ["c1"],
        "count": 1,
        "update_time": 1_600_000_000,
        "_id": "legacy",
    }

    with patch("time.time", return_value=1_700_000_200):
        await storage.upsert({"legacy": {"chunk_ids": ["c1", "c2"], "count": 2}})

    row = await storage.get_by_id("legacy")
    assert row is not None
    assert row["create_time"] == 0
    assert row["update_time"] == 1_700_000_200
    assert storage._data["legacy"]["create_time"] == 0


@pytest.mark.asyncio
async def test_update_uses_one_lookup_per_key(tmp_path):
    storage = await _make_kv(
        tmp_path, namespace=NameSpace.KV_STORE_ENTITY_CHUNKS, workspace="ct-ws"
    )
    counting = _CountingDict()
    storage._data = counting

    with patch("time.time", return_value=1_700_000_000):
        await storage.upsert({"A": {"x": 1}, "B": {"x": 1}})
    with patch("time.time", return_value=1_700_000_100):
        counting.calls.update({"get": 0, "getitem": 0, "contains": 0})
        await storage.upsert({"A": {"x": 2}, "B": {"x": 2}})

    assert counting.calls["get"] == 2
    assert counting.calls["getitem"] == 0
    assert counting.calls["contains"] == 0
    assert counting["A"]["create_time"] == 1_700_000_000
    assert counting["A"]["update_time"] == 1_700_000_100


@pytest.mark.asyncio
async def test_legacy_float_create_time_is_normalized(tmp_path):
    storage = await _make_kv(
        tmp_path, namespace=NameSpace.KV_STORE_ENTITY_CHUNKS, workspace="ct-ws"
    )
    storage._data["f"] = {"x": 1, "create_time": 1_650_000_000.75}

    with patch("time.time", return_value=1_700_000_200):
        await storage.upsert({"f": {"x": 2}})

    stored = storage._data["f"]
    assert stored["create_time"] == 1_650_000_000
    assert isinstance(stored["create_time"], int)


@pytest.mark.asyncio
async def test_caller_supplied_create_time_ignored_on_update(tmp_path):
    storage = await _make_kv(
        tmp_path, namespace=NameSpace.KV_STORE_ENTITY_CHUNKS, workspace="ct-ws"
    )

    with patch("time.time", return_value=1_700_000_000):
        await storage.upsert({"E": {"x": 1}})
    with patch("time.time", return_value=1_700_000_100):
        await storage.upsert({"E": {"x": 2, "create_time": 1}})

    assert storage._data["E"]["create_time"] == 1_700_000_000


@pytest.mark.asyncio
async def test_infinite_create_time_does_not_abort_the_upsert(tmp_path):
    storage = await _make_kv(
        tmp_path, namespace=NameSpace.KV_STORE_ENTITY_CHUNKS, workspace="ct-ws"
    )
    storage._data["inf"] = {"x": 1, "create_time": json.loads("1e309")}

    with patch("time.time", return_value=1_700_000_600):
        await storage.upsert({"inf": {"x": 2}})

    assert storage._data["inf"]["create_time"] == 0
    assert storage._data["inf"]["update_time"] == 1_700_000_600


# ---------------------------------------------------------------------------
# Configuration container identity row
# ---------------------------------------------------------------------------


async def _open_config(tmp_path) -> JsonKVStorage:
    storage = cs.create_configuration_storage(
        JsonKVStorage,
        global_config={"working_dir": str(tmp_path)},
        embedding_func=None,
    )
    await storage.initialize()
    return storage


async def _bind_config(storage, tmp_path):
    return await cs.bind_configuration_identity(
        storage,
        working_dir=str(tmp_path),
        backend="JsonKVStorage",
        container=f"JsonKVStorage at {tmp_path}",
        workspace="",
    )


async def test_the_identity_is_durable_and_verified_after_a_restart(tmp_path):
    storage = await _open_config(tmp_path)
    created = await _bind_config(storage, tmp_path)
    await storage.finalize()
    assert created.action == "registered"

    config_file = tmp_path / CONFIG_JSON_FILE_NAME
    on_disk = json.loads(config_file.read_text(encoding="utf-8"))
    assert on_disk[cs.storage_identity_key()]["value"] == {"uuid": created.storage_uuid}
    assert on_disk[cs.json_shard_owner_key()]["value"] == {"workspace": ""}
    anchor = ca.read_anchor(str(tmp_path))
    assert anchor.storage_uuid == created.storage_uuid
    assert anchor.members == ("",)

    finalize_share_data()
    initialize_share_data(workers=1)
    storage = await _open_config(tmp_path)
    verified = await _bind_config(storage, tmp_path)
    await storage.finalize()
    assert verified.action == "verified"
    assert verified.storage_uuid == created.storage_uuid


async def test_a_damaged_identity_row_refuses_and_is_never_regenerated(tmp_path):
    path = tmp_path / CONFIG_JSON_FILE_NAME
    path.write_text(
        json.dumps({cs.storage_identity_key(): "not a row"}), encoding="utf-8"
    )
    storage = await _open_config(tmp_path)
    with pytest.raises(ConfigurationStorageError):
        await _bind_config(storage, tmp_path)
    await storage.finalize()
    assert json.loads(path.read_text(encoding="utf-8")) == {
        cs.storage_identity_key(): "not a row"
    }
    assert ca.read_anchor(str(tmp_path)) is None
