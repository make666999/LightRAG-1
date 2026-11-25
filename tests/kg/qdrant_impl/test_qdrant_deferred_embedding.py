"""Unit tests for QdrantVectorDBStorage deferred-embedding flush and batching.

All tests use mocks — no running Qdrant instance required.
"""

import asyncio
import os
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip(
    "qdrant_client",
    reason="qdrant-client is required for Qdrant storage tests",
)

from qdrant_client import models  # noqa: E402

from lightrag.kg.qdrant_impl import (  # noqa: E402
    QdrantVectorDBStorage,
    compute_mdhash_id_for_qdrant,
)

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


class MockEmbeddingFunc:
    def __init__(self, dim=8):
        self.embedding_dim = dim
        self.max_token_size = 512
        self.model_name = "mock-embed"

    async def __call__(self, texts, **kwargs):
        return np.random.rand(len(texts), self.embedding_dim).astype(np.float32)


class CountingEmbeddingFunc(MockEmbeddingFunc):
    def __init__(self, dim=8, fail_times=0):
        super().__init__(dim=dim)
        self.fail_times = fail_times
        self.call_count = 0
        self.batches: list[list[str]] = []
        self.texts: list[str] = []

    async def __call__(self, texts, **kwargs):
        self.call_count += 1
        batch = list(texts)
        self.batches.append(batch)
        self.texts.extend(batch)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("embedding failed")
        return await super().__call__(texts, **kwargs)


@pytest.fixture(autouse=True)
def patch_namespace_lock():
    """Cache real asyncio.Locks per (namespace, workspace) for shared semantics."""
    cache: dict[tuple[str, str | None], asyncio.Lock] = {}

    def factory(namespace, workspace=None, enable_logging=False):
        key = (namespace, workspace or "")
        lock = cache.get(key)
        if lock is None:
            lock = asyncio.Lock()
            cache[key] = lock
        return lock

    with patch("lightrag.kg.qdrant_impl.get_namespace_lock", side_effect=factory):
        yield cache


def _make_storage(
    embed_func,
    *,
    namespace="entities",
    workspace="test_ws",
    meta_fields=None,
):
    if meta_fields is None:
        meta_fields = {"content", "entity_name", "src_id", "tgt_id"}

    storage = QdrantVectorDBStorage.__new__(QdrantVectorDBStorage)
    storage.workspace = workspace
    storage.namespace = namespace
    storage.effective_workspace = workspace
    storage.model_suffix = "mock"
    storage.final_namespace = f"lightrag_vdb_{namespace}_mock"
    storage.meta_fields = meta_fields
    storage.embedding_func = embed_func
    storage._max_batch_size = 10
    storage._max_upsert_payload_bytes = 16 * 1024 * 1024
    storage._max_upsert_points_per_batch = 128
    storage._max_delete_points_per_batch = 1000
    storage._pending_vector_docs = {}
    storage._pending_vector_deletes = set()

    storage._client = MagicMock()
    storage._client.upsert = MagicMock()
    storage._client.delete = MagicMock()
    # drop() looks up a legacy collection to clear its workspace points;
    # default to "the suffixed collection exists, no legacy collection" so
    # unrelated tests are unaffected.
    storage._client.collection_exists = MagicMock(
        side_effect=lambda name: name == storage.final_namespace
    )
    storage._client.retrieve = MagicMock(return_value=[])
    storage._client.scroll = MagicMock(return_value=([], None))

    from lightrag.kg.qdrant_impl import get_namespace_lock

    storage._flush_lock = get_namespace_lock(
        namespace=storage.final_namespace, workspace=storage.effective_workspace
    )
    return storage


def _make_point(point_id: str, content: str) -> models.PointStruct:
    return models.PointStruct(
        id=point_id,
        vector=[0.1, 0.2, 0.3],
        payload={"id": point_id, "content": content},
    )


# ---------------------------------------------------------------------------
# Deferred-embedding buffering, flush, reads, and teardown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_buffers_without_embedding():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    await s.upsert({"v1": {"content": "hello"}, "v2": {"content": "world"}})

    assert embed.call_count == 0
    assert set(s._pending_vector_docs.keys()) == {"v1", "v2"}
    assert s._pending_vector_docs["v1"].vector is None
    s._client.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_index_done_callback_triggers_flush():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    await s.upsert({"v1": {"content": "hello"}, "v2": {"content": "world"}})
    await s.index_done_callback()

    assert embed.call_count == 1
    s._client.upsert.assert_called_once()
    kwargs = s._client.upsert.call_args.kwargs
    assert kwargs["collection_name"] == s.final_namespace
    points = kwargs["points"]
    assert len(points) == 2
    expected_ids = {
        compute_mdhash_id_for_qdrant("v1", prefix=s.effective_workspace),
        compute_mdhash_id_for_qdrant("v2", prefix=s.effective_workspace),
    }
    assert {p.id for p in points} == expected_ids
    assert s._pending_vector_docs == {}


@pytest.mark.asyncio
async def test_repeated_upsert_same_id_embeds_once():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    await s.upsert({"v1": {"content": "first"}})
    await s.upsert({"v1": {"content": "second"}})
    await s.upsert({"v1": {"content": "third"}})
    await s.index_done_callback()

    assert embed.call_count == 1
    assert embed.texts == ["third"]
    s._client.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_deferred_embeddings_respect_batch_size():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)
    s._max_batch_size = 2

    await s.upsert({f"v{i}": {"content": f"doc {i}"} for i in range(5)})
    await s.index_done_callback()

    assert embed.call_count == 3
    assert [len(b) for b in embed.batches] == [2, 2, 1]


@pytest.mark.asyncio
async def test_get_vectors_by_ids_lazy_embed_then_reuse_in_flush():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    await s.upsert({"v1": {"content": "hello"}})
    vectors = await s.get_vectors_by_ids(["v1"])
    assert "v1" in vectors
    assert embed.call_count == 1
    assert s._pending_vector_docs["v1"].vector is not None

    await s.index_done_callback()
    assert embed.call_count == 1
    s._client.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_get_vectors_by_ids_falls_back_to_requested_id_when_payload_id_missing():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed, namespace="chunks", workspace="test_ws")

    qdrant_id = compute_mdhash_id_for_qdrant("chunk-1", prefix=s.effective_workspace)
    s._client.retrieve.return_value = [
        SimpleNamespace(
            id=qdrant_id,
            payload={"content": "legacy chunk"},
            vector=[0.1, 0.2, 0.3],
        )
    ]

    assert await s.get_by_ids(["chunk-1"]) == [
        {"content": "legacy chunk", "created_at": None}
    ]
    assert await s.get_vectors_by_ids(["chunk-1"]) == {"chunk-1": [0.1, 0.2, 0.3]}


@pytest.mark.asyncio
async def test_id_lookups_fallback_normalize_hyphenated_qdrant_id():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed, namespace="chunks", workspace="test_ws")

    qdrant_id = compute_mdhash_id_for_qdrant("chunk-1", prefix=s.effective_workspace)
    s._client.retrieve.return_value = [
        SimpleNamespace(
            id=str(uuid.UUID(qdrant_id)),
            payload={"content": "legacy chunk"},
            vector=[0.1, 0.2, 0.3],
        )
    ]

    assert await s.get_by_ids(["chunk-1"]) == [
        {"content": "legacy chunk", "created_at": None}
    ]
    assert await s.get_vectors_by_ids(["chunk-1"]) == {"chunk-1": [0.1, 0.2, 0.3]}


@pytest.mark.asyncio
async def test_flush_failure_keeps_buffer_no_double_embed_on_retry():
    embed = CountingEmbeddingFunc(fail_times=1)
    s = _make_storage(embed)

    await s.upsert({"v1": {"content": "hello"}})

    with pytest.raises(RuntimeError, match="embedding failed"):
        await s.index_done_callback()

    assert "v1" in s._pending_vector_docs
    assert s._pending_vector_docs["v1"].vector is None
    s._client.upsert.assert_not_called()

    await s.index_done_callback()
    assert embed.call_count == 2
    s._client.upsert.assert_called_once()
    assert s._pending_vector_docs == {}


@pytest.mark.asyncio
async def test_server_upsert_failure_keeps_buffer():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)
    s._client.upsert.side_effect = RuntimeError("qdrant down")

    await s.upsert({"v1": {"content": "hello"}})

    with pytest.raises(RuntimeError, match="qdrant down"):
        await s.index_done_callback()

    assert "v1" in s._pending_vector_docs
    assert s._pending_vector_docs["v1"].vector is not None

    s._client.upsert.side_effect = None
    await s.index_done_callback()
    assert embed.call_count == 1


@pytest.mark.asyncio
async def test_finalize_raises_when_buffer_unflushed():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)
    s._client.upsert.side_effect = RuntimeError("transient qdrant error")

    await s.upsert({"v1": {"content": "hello"}})

    with pytest.raises(RuntimeError, match="finalize.*flush raised"):
        await s.finalize()
    assert "v1" in s._pending_vector_docs


@pytest.mark.asyncio
async def test_delete_then_upsert_same_id_keeps_upsert():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    await s.delete(["v1"])
    assert "v1" in s._pending_vector_deletes

    await s.upsert({"v1": {"content": "hello"}})
    assert "v1" in s._pending_vector_docs
    assert "v1" not in s._pending_vector_deletes

    await s.index_done_callback()
    s._client.upsert.assert_called_once()
    s._client.delete.assert_not_called()


@pytest.mark.asyncio
async def test_upsert_then_delete_same_id_keeps_delete():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    await s.upsert({"v1": {"content": "hello"}})
    await s.delete(["v1"])
    assert "v1" not in s._pending_vector_docs
    assert "v1" in s._pending_vector_deletes

    await s.index_done_callback()
    s._client.upsert.assert_not_called()
    s._client.delete.assert_called_once()
    qdrant_delete_ids = s._client.delete.call_args.kwargs["points_selector"].points
    assert qdrant_delete_ids == [
        compute_mdhash_id_for_qdrant("v1", prefix=s.effective_workspace)
    ]


@pytest.mark.asyncio
async def test_delete_entity_relation_raises_on_server_failure():
    """scroll-then-delete pattern: server-side failure must bubble up."""
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)
    fake_point = MagicMock()
    fake_point.id = "qid1"
    s._client.scroll.return_value = ([fake_point], None)
    s._client.delete.side_effect = RuntimeError("qdrant delete failed")

    with pytest.raises(RuntimeError, match="qdrant delete failed"):
        await s.delete_entity_relation("X")


@pytest.mark.asyncio
async def test_delete_entity_relation_prunes_pending_buffer():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    await s.upsert(
        {
            "rel-A-B": {"content": "A→B", "src_id": "A", "tgt_id": "B"},
            "rel-C-D": {"content": "C→D", "src_id": "C", "tgt_id": "D"},
        }
    )
    s._client.scroll.return_value = ([], None)

    await s.delete_entity_relation("A")
    assert "rel-A-B" not in s._pending_vector_docs
    assert "rel-C-D" in s._pending_vector_docs


@pytest.mark.asyncio
async def test_get_by_id_reads_pending_buffer_without_vector():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    await s.upsert({"v1": {"content": "hello", "entity_name": "E1"}})
    doc = await s.get_by_id("v1")
    assert doc is not None
    assert doc.get("entity_name") == "E1"
    assert "vector" not in doc
    s._client.retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_get_by_id_returns_none_for_pending_delete():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)

    await s.delete(["v1"])
    assert await s.get_by_id("v1") is None
    s._client.retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_env_workspace_override_shares_flush_lock(patch_namespace_lock):
    cache = patch_namespace_lock
    embed = CountingEmbeddingFunc()

    with patch.dict(os.environ, {"QDRANT_WORKSPACE": "shared_ws"}, clear=False):
        a = _make_storage(embed, workspace="shared_ws")
        b = _make_storage(embed, workspace="shared_ws")
        assert a.final_namespace == b.final_namespace
        assert a.effective_workspace == b.effective_workspace == "shared_ws"
        assert a._flush_lock is b._flush_lock
        assert len([k for k in cache if k == (a.final_namespace, "shared_ws")]) == 1


@pytest.mark.asyncio
async def test_distinct_workspaces_in_same_collection_get_independent_locks(
    patch_namespace_lock,
):
    """Same final_namespace but different workspaces → independent locks."""
    embed = CountingEmbeddingFunc()
    a = _make_storage(embed, workspace="ws_a")
    b = _make_storage(embed, workspace="ws_b")
    assert a.final_namespace == b.final_namespace
    assert a.effective_workspace != b.effective_workspace
    assert a._flush_lock is not b._flush_lock


@pytest.mark.asyncio
async def test_drop_clears_pending_buffers():
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)
    await s.upsert({"v1": {"content": "hello"}})
    await s.delete(["v2"])
    assert s._pending_vector_docs and s._pending_vector_deletes

    result = await s.drop()
    assert result["status"] == "success"
    assert s._pending_vector_docs == {}
    assert s._pending_vector_deletes == set()


@pytest.mark.asyncio
async def test_drop_clears_workspace_points_from_workspace_tagged_legacy():
    """drop() removes only this workspace's points from a workspace-tagged legacy
    collection, so the next startup does not re-migrate the cleared data back."""
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)
    legacy_collection = f"lightrag_vdb_{s.namespace}"
    s._client.collection_exists = MagicMock(
        side_effect=lambda name: name in (legacy_collection, s.final_namespace)
    )
    legacy_info = MagicMock()
    legacy_info.payload_schema = {"workspace_id": MagicMock()}
    s._client.get_collection = MagicMock(return_value=legacy_info)

    result = await s.drop()
    assert result["status"] == "success"

    deleted_collections = [
        call.kwargs.get("collection_name") for call in s._client.delete.call_args_list
    ]
    assert s.final_namespace in deleted_collections
    assert legacy_collection in deleted_collections
    s._client.delete_collection.assert_not_called()


@pytest.mark.asyncio
async def test_drop_drops_untagged_legacy_collection():
    """For an untagged (pre-isolation) legacy collection, setup_collection would
    migrate ALL of its points back with no workspace filter, so a filtered delete
    misses them. drop() must drop the whole legacy collection instead."""
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)
    legacy_collection = f"lightrag_vdb_{s.namespace}"
    s._client.collection_exists = MagicMock(
        side_effect=lambda name: name in (legacy_collection, s.final_namespace)
    )
    legacy_info = MagicMock()
    legacy_info.payload_schema = {}
    s._client.get_collection = MagicMock(return_value=legacy_info)
    s._client.scroll = MagicMock(return_value=([], None))

    result = await s.drop()
    assert result["status"] == "success"

    s._client.delete_collection.assert_called_once_with(
        collection_name=legacy_collection
    )
    filtered_deletes = [
        call.kwargs.get("collection_name") for call in s._client.delete.call_args_list
    ]
    assert legacy_collection not in filtered_deletes


@pytest.mark.asyncio
async def test_drop_reports_error_when_legacy_tagging_undetermined():
    """If the legacy collection's workspace tagging cannot be determined (a
    transient metadata error), drop() must NOT drop the whole collection AND
    must report an error so the caller can retry."""
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)
    legacy_collection = f"lightrag_vdb_{s.namespace}"
    s._client.collection_exists = MagicMock(
        side_effect=lambda name: name in (legacy_collection, s.final_namespace)
    )
    s._client.get_collection = MagicMock(side_effect=RuntimeError("qdrant unavailable"))

    result = await s.drop()
    assert result["status"] == "error"

    legacy_deletes = [
        call.kwargs.get("collection_name") for call in s._client.delete.call_args_list
    ]
    assert legacy_collection not in legacy_deletes
    s._client.delete_collection.assert_not_called()


@pytest.mark.asyncio
async def test_drop_pending_index_ops_clears_buffers():
    """On an internal-error abort the pipeline calls drop_pending_index_ops to
    discard buffered upserts/deletes without flushing them."""
    embed = CountingEmbeddingFunc()
    s = _make_storage(embed)
    await s.upsert({"v1": {"content": "x"}, "v2": {"content": "y"}})
    s._pending_vector_deletes.add("old-id")
    assert s._pending_vector_docs
    await s.drop_pending_index_ops()
    assert not s._pending_vector_docs
    assert not s._pending_vector_deletes


@pytest.mark.asyncio
async def test_finalize_closes_qdrant_client():
    """finalize() must release the Qdrant client transport instead of leaving
    it for GC."""
    s = _make_storage(MockEmbeddingFunc())
    client = s._client
    assert client is not None

    await s.finalize()

    client.close.assert_called_once()


# ---------------------------------------------------------------------------
# Upsert & delete batch sizing and fail-fast semantics
# ---------------------------------------------------------------------------


def test_build_upsert_batches_respects_point_limit():
    points = [_make_point(str(i), "x" * 10) for i in range(5)]

    batches = QdrantVectorDBStorage._build_upsert_batches(
        points, max_payload_bytes=1024 * 1024, max_points_per_batch=2
    )

    assert [len(batch_points) for batch_points, _ in batches] == [2, 2, 1]


def test_build_upsert_batches_exact_payload_boundary_no_split():
    point_a = _make_point("a", "x" * 32)
    point_b = _make_point("b", "y" * 32)

    size_a = QdrantVectorDBStorage._estimate_point_payload_bytes(point_a)
    size_b = QdrantVectorDBStorage._estimate_point_payload_bytes(point_b)
    # JSON array envelope: [] => 2 bytes, and comma between two elements => 1 byte
    exact_limit = 2 + size_a + 1 + size_b

    batches = QdrantVectorDBStorage._build_upsert_batches(
        [point_a, point_b],
        max_payload_bytes=exact_limit,
        max_points_per_batch=128,
    )

    assert len(batches) == 1
    assert len(batches[0][0]) == 2
    assert batches[0][1] == exact_limit


def test_build_upsert_batches_single_oversized_point_gets_own_batch():
    """An oversized point is emitted as its own batch rather than raising.

    Failing here would poison the whole flush; instead the server is left as
    the final arbiter on whether the (conservatively estimated) point fits.
    """
    point = _make_point("oversized", "x" * 64)
    point_size = QdrantVectorDBStorage._estimate_point_payload_bytes(point)
    too_small_limit = point_size - 1

    batches = QdrantVectorDBStorage._build_upsert_batches(
        [point],
        max_payload_bytes=too_small_limit,
        max_points_per_batch=128,
    )

    assert len(batches) == 1
    assert len(batches[0][0]) == 1
    assert batches[0][0][0].id == "oversized"


def test_build_upsert_batches_isolates_oversized_point_between_normal_ones():
    """A mid-stream oversized point lands alone; neighbors are not polluted."""
    small_a = _make_point("a", "x" * 4)
    huge = _make_point("HUGE", "x" * 4096)
    small_b = _make_point("b", "y" * 4)

    budget = QdrantVectorDBStorage._estimate_point_payload_bytes(small_a) + 16

    batches = QdrantVectorDBStorage._build_upsert_batches(
        [small_a, huge, small_b],
        max_payload_bytes=budget,
        max_points_per_batch=128,
    )

    huge_batches = [b for b, _ in batches if any(p.id == "HUGE" for p in b)]
    assert len(huge_batches) == 1 and len(huge_batches[0]) == 1
    assert sum(len(b) for b, _ in batches) == 3


@pytest.mark.asyncio
async def test_flush_fail_fast_stops_on_first_failed_batch():
    """Flush-time fail-fast: once any batch raises, subsequent batches are skipped
    and the buffer is preserved so the next flush can retry."""
    storage = _make_storage(
        MockEmbeddingFunc(dim=2), namespace="chunks", meta_fields={"content"}
    )
    storage._max_batch_size = 16
    storage._max_upsert_payload_bytes = 1024 * 1024
    storage._max_upsert_points_per_batch = 2
    storage.final_namespace = "test_collection"
    storage._client.upsert.side_effect = [None, RuntimeError("batch failed"), None]

    data = {f"chunk-{i}": {"content": f"content-{i}"} for i in range(5)}

    await storage.upsert(data)
    assert len(storage._pending_vector_docs) == 5

    with pytest.raises(RuntimeError, match="batch failed"):
        await storage._flush_pending_vector_ops()

    # 5 items with max 2 points per batch => stop at batch #2 on error.
    assert storage._client.upsert.call_count == 2
    assert len(storage._client.upsert.call_args_list[0].kwargs["points"]) == 2
    assert len(storage._client.upsert.call_args_list[1].kwargs["points"]) == 2
    assert len(storage._pending_vector_docs) == 5


@pytest.mark.asyncio
async def test_flush_chunks_deletes_by_point_count():
    """Deletes are split into chunks of at most _max_delete_points_per_batch."""
    storage = _make_storage(
        MockEmbeddingFunc(dim=2), namespace="chunks", meta_fields={"content"}
    )
    storage._max_delete_points_per_batch = 2
    storage.final_namespace = "test_collection"
    storage._pending_vector_deletes = {f"chunk-{i}" for i in range(5)}

    await storage._flush_pending_vector_ops()

    assert storage._client.delete.call_count == 3
    chunk_sizes = sorted(
        len(call.kwargs["points_selector"].points)
        for call in storage._client.delete.call_args_list
    )
    assert chunk_sizes == [1, 2, 2]
    assert len(storage._pending_vector_deletes) == 0
