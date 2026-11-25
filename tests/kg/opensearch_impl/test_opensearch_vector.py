"""Offline unit tests for OpenSearchVectorDBStorage: k-NN search, cosine scoring,
readiness probes, deferred write batching, durability, and vector-space provenance."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

pytest.importorskip(
    "opensearchpy",
    reason="opensearchpy is required for OpenSearch storage tests",
)

from opensearchpy.exceptions import (  # type: ignore  # noqa: E402
    NotFoundError,
    OpenSearchException,
)

from lightrag.exceptions import (  # noqa: E402
    StorageCapabilityError,
    VectorSpaceMismatchError,
)
from lightrag.kg.opensearch_impl import (  # noqa: E402
    _FINAL_NAMESPACE_META_KEY,
    _WORKSPACE_META_KEY,
    ClientManager,
    OpenSearchVectorDBStorage,
    WorkspaceIndexCollisionError,
    _workspace_index_meta,
)
from lightrag.kg.shared_storage import (  # noqa: E402
    finalize_share_data,
    initialize_share_data,
)
from lightrag.kg.vector_space import (  # noqa: E402
    VECTOR_SPACE_DIM_KEY,
    VECTOR_SPACE_MODEL_KEY,
)
from lightrag.utils import EmbeddingFunc, compute_mdhash_id  # noqa: E402

from ._helpers import (  # noqa: E402
    BufferedWriteBatchingContractMixin,
    CountingEmbeddingFunc,
    FakeCluster,
    FakeRateLimitedServer,
    FixedEmbeddingFunc,
    MockEmbeddingFunc,
    ReadinessDemotionMixin,
    default_global_config,
    foreign_probe,
    gone_probe,
    make_client,
    missing_index_error,
    owned_probe,
    owned_probe_with_dim,
    patch_opensearch_locks,
)

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def patch_locks():
    with patch_opensearch_locks(patch_shard_doc=True):
        yield


@pytest.fixture
def global_config():
    return default_global_config()


@pytest.fixture
def embed_func():
    return MockEmbeddingFunc()


@pytest.fixture
def mock_client():
    return make_client()


@pytest.fixture
def cluster():
    return FakeCluster()


@pytest.fixture
def shared_storage_env():
    initialize_share_data(workers=1)
    try:
        yield
    finally:
        finalize_share_data()


def _make_vector(
    global_config, embed_func, workspace: str = "test"
) -> OpenSearchVectorDBStorage:
    return OpenSearchVectorDBStorage(
        namespace="entities",
        global_config=global_config,
        embedding_func=embed_func,
        workspace=workspace,
        meta_fields={"content", "entity_name", "src_id", "tgt_id"},
    )


# ---------------------------------------------------------------------------
# Vector Storage k-NN, cosine conversion, exact count, and presence recheck
# ---------------------------------------------------------------------------


class TestVectorStorage(ReadinessDemotionMixin):
    """Tests for OpenSearchVectorDBStorage k-NN index, cosine conversion, and probes."""

    missing_read_rpc = "search"

    def _make(self, global_config, embed_func, workspace: str = "test"):
        return _make_vector(global_config, embed_func, workspace)

    def _configure_missing_read(self, storage, mock_client) -> None:
        super()._configure_missing_read(storage, mock_client)
        mock_client.indices.get_mapping = gone_probe()

    async def _invoke_missing_read(self, storage):
        return await storage.query("test", top_k=5)

    def _expected_missing_read_result(self):
        return []

    @pytest.mark.asyncio
    async def test_cosine_threshold_required(self, embed_func):
        with pytest.raises(ValueError, match="cosine_better_than_threshold"):
            OpenSearchVectorDBStorage(
                namespace="v",
                global_config={
                    "embedding_batch_num": 10,
                    "vector_db_storage_cls_kwargs": {},
                },
                embedding_func=embed_func,
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "refresh_shards,count_shards,expected",
        [
            (
                {"total": 4, "successful": 2, "failed": 0},
                {"total": 2, "successful": 2, "failed": 0},
                5,
            ),
            (
                {"total": 4, "successful": 2, "failed": 0},
                {"total": 2, "successful": 1, "failed": 1},
                "partial_shard_count",
            ),
            ({"total": 4, "successful": 2, "failed": 0}, None, "partial_shard_count"),
            (
                {"total": 4, "successful": 1, "failed": 1},
                {"total": 2, "successful": 2, "failed": 0},
                "partial_refresh",
            ),
        ],
        ids=["all-shards", "failed-count-shard", "no-count-header", "failed-refresh"],
    )
    async def test_exact_count_needs_every_shard_refreshed_and_counted(
        self,
        global_config,
        embed_func,
        mock_client,
        refresh_shards,
        count_shards,
        expected,
    ):
        order: list[str] = []

        async def record_refresh(**kwargs):
            order.append("refresh")
            return {"_shards": refresh_shards}

        async def record_count(**kwargs):
            order.append("count")
            response = {"count": 5}
            if count_shards is not None:
                response["_shards"] = count_shards
            return response

        mock_client.indices.refresh = AsyncMock(side_effect=record_refresh)
        mock_client.count = AsyncMock(side_effect=record_count)
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            order.clear()
            if isinstance(expected, int):
                assert await s.get_exact_count() == expected
            else:
                with pytest.raises(StorageCapabilityError, match=expected):
                    await s.get_exact_count()
            refreshed_only = expected == "partial_refresh"
            assert order == (["refresh"] if refreshed_only else ["refresh", "count"])

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("mget", "strict_outcome"),
        [
            (
                AsyncMock(side_effect=OpenSearchException("transport error")),
                OpenSearchException,
            ),
            (
                AsyncMock(
                    return_value={
                        "docs": [
                            {"_id": "a", "found": True, "_source": {"content": "x"}},
                            {"_id": "b", "error": {"type": "shard_failure"}},
                        ]
                    }
                ),
                RuntimeError,
            ),
            (
                AsyncMock(
                    return_value={
                        "docs": [
                            {"_id": "a", "found": True, "_source": {"content": "x"}},
                            {"_id": "b", "found": False},
                        ]
                    }
                ),
                ["a", None],
            ),
        ],
        ids=["transport-error", "item-error", "confirmed-absent"],
    )
    async def test_strict_read_returns_none_only_for_confirmed_absence(
        self, global_config, embed_func, mock_client, mget, strict_outcome
    ):
        mock_client.mget = mget
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert (await s.get_by_ids(["a", "b"]))[1] is None
            if isinstance(strict_outcome, type):
                with pytest.raises(strict_outcome) as excinfo:
                    await s.get_by_ids_strict(["a", "b"])
                assert type(excinfo.value) is strict_outcome
            else:
                rows = await s.get_by_ids_strict(["a", "b"])
                assert [row and row["id"] for row in rows] == strict_outcome

    @pytest.mark.asyncio
    async def test_initialize_creates_knn_index(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            mock_client.indices.create.assert_awaited_once()
            body = mock_client.indices.create.call_args.kwargs["body"]
            assert body["settings"]["index"]["knn"] is True
            assert body["mappings"]["properties"]["vector"]["dimension"] == 128
            method = body["mappings"]["properties"]["vector"]["method"]
            assert method["engine"] == "lucene"
            assert method["space_type"] == "cosinesimil"

    @pytest.mark.asyncio
    async def test_query_cosine_score_conversion(
        self, global_config, embed_func, mock_client
    ):
        mock_client.search = AsyncMock(
            return_value={
                "hits": {
                    "hits": [
                        {
                            "_id": "v1",
                            "_score": 0.85,
                            "_source": {"content": "match", "entity_name": "E1"},
                        },
                    ],
                    "total": {"value": 1},
                },
                "aggregations": {
                    "status_counts": {"buckets": []},
                    "src": {"buckets": []},
                    "tgt": {"buckets": []},
                },
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            with patch("lightrag.kg.opensearch_impl.logger.info") as mock_log:
                results = await s.query("test", top_k=5)
            assert len(results) == 1
            assert results[0]["distance"] == pytest.approx(0.7)
            assert "cosine_range=[0.7000, 0.7000]" in mock_log.call_args.args[0]

    @pytest.mark.asyncio
    async def test_query_rejects_unconverted_score_that_would_have_passed(
        self, global_config, embed_func, mock_client
    ):
        mock_client.search = AsyncMock(
            return_value={
                "hits": {
                    "hits": [
                        {
                            "_id": "v1",
                            "_score": 0.3,
                            "_source": {"content": "unrelated"},
                        },
                    ],
                    "total": {"value": 1},
                },
                "aggregations": {
                    "status_counts": {"buckets": []},
                    "src": {"buckets": []},
                    "tgt": {"buckets": []},
                },
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            results = await s.query("test", top_k=5)
            assert len(results) == 0

    @pytest.mark.asyncio
    async def test_query_with_provided_embedding(self, global_config, mock_client):
        mock_client.search = AsyncMock(
            return_value={
                "hits": {
                    "hits": [
                        {"_id": "v1", "_score": 1.0, "_source": {"content": "exact"}},
                    ],
                    "total": {"value": 1},
                },
                "aggregations": {
                    "status_counts": {"buckets": []},
                    "src": {"buckets": []},
                    "tgt": {"buckets": []},
                },
            }
        )
        embed_func = CountingEmbeddingFunc()
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            vec = np.random.rand(128).astype(np.float32)
            results = await s.query("test", top_k=5, query_embedding=vec)
            assert len(results) == 1
            assert results[0]["distance"] == 1.0
            assert embed_func.call_count == 0
            body = mock_client.search.call_args.kwargs["body"]
            assert body["query"]["knn"]["vector"]["vector"] == vec.tolist()

    @pytest.mark.asyncio
    async def test_get_vectors_by_ids(self, global_config, embed_func, mock_client):
        vec = [0.1] * 128
        mock_client.mget = AsyncMock(
            return_value={
                "docs": [
                    {"_id": "v1", "found": True, "_source": {"vector": vec}},
                    {"_id": "v2", "found": False},
                ]
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            result = await s.get_vectors_by_ids(["v1", "v2"])
            assert "v1" in result
            assert "v2" not in result
            assert result["v1"] == vec

    @pytest.mark.asyncio
    async def test_delete(self, global_config, embed_func, mock_client):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (2, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.delete(["v1", "v2"])
                mock_bulk.assert_not_awaited()
                assert s._pending_vector_deletes == {"v1", "v2"}
                await s.index_done_callback()
                mock_bulk.assert_awaited_once()
                actions = mock_bulk.call_args[0][1]
                assert len(actions) == 2
                assert all(a["_op_type"] == "delete" for a in actions)

    @pytest.mark.asyncio
    async def test_delete_entity(self, global_config, embed_func, mock_client):
        entity_id = compute_mdhash_id("Alice", prefix="ent-")
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.upsert({entity_id: {"content": "Alice"}, "other": {"content": "x"}})
            await s.delete_entity("Alice")
            mock_client.delete.assert_not_awaited()
            assert s._pending_vector_deletes == {entity_id}
            assert set(s._pending_vector_docs) == {"other"}

    @pytest.mark.asyncio
    async def test_drop_recreates_index(self, global_config, embed_func, mock_client):
        mock_client.indices.exists = AsyncMock(return_value=False)
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            result = await s.drop()
            assert result["status"] == "success"
            mock_client.indices.delete.assert_awaited_once()
            assert mock_client.indices.create.await_count == 2

    @pytest.mark.asyncio
    async def test_drop_recreate_error_marks_index_not_ready(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            with patch.object(
                s,
                "_create_knn_index_if_not_exists",
                new=AsyncMock(side_effect=OpenSearchException("recreate failed")),
            ):
                result = await s.drop()
                assert result["status"] == "error"
                assert s._index_ready is False

    @pytest.mark.asyncio
    async def test_drop_recreates_index_when_missing(
        self, global_config, embed_func, mock_client
    ):
        mock_client.indices.exists = AsyncMock(return_value=False)
        mock_client.indices.delete = AsyncMock(
            side_effect=NotFoundError(404, "not found")
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            result = await s.drop()
            assert result["status"] == "success"
            assert mock_client.indices.create.await_count == 2

    @pytest.mark.asyncio
    async def test_reads_recover_once_the_index_is_back(
        self, global_config, embed_func, mock_client
    ):
        mock_client.search = AsyncMock(
            return_value={
                "hits": {
                    "hits": [{"_id": "v1", "_score": 0.9, "_source": {"content": "c"}}]
                }
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            mock_client.indices.create.reset_mock()
            s._index_ready = False
            mock_client.indices.get_mapping = owned_probe(s)

            results = await s.query("test", top_k=5)

            assert s._index_ready is True
            assert [r["id"] for r in results] == ["v1"]
            mock_client.indices.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_presence_recheck_never_creates_the_index(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            mock_client.indices.create.reset_mock()
            mock_client.indices.get_mapping = gone_probe()
            s._index_ready = False

            assert await s.query("test", top_k=5) == []
            assert await s.get_by_id("v1") is None
            assert await s.get_by_ids(["v1"]) == [None]
            assert await s.get_vectors_by_ids(["v1"]) == {}

            assert s._index_ready is False
            assert mock_client.indices.get_mapping.await_count == 4
            mock_client.indices.create.assert_not_awaited()
            mock_client.search.assert_not_awaited()
            mock_client.mget.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_presence_recheck_is_free_while_the_index_is_ready(
        self, global_config, embed_func, mock_client
    ):
        mock_client.search = AsyncMock(return_value={"hits": {"hits": []}})
        mock_client.mget = AsyncMock(
            return_value={"docs": [{"_id": "v1", "found": False}]}
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            mock_client.indices.get_mapping.reset_mock()

            await s.query("test", top_k=5)
            assert await s.get_by_id("v1") is None
            mock_client.get.assert_not_awaited()
            await s.get_vectors_by_ids(["v1"])

            mock_client.indices.get_mapping.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_presence_recheck_runs_outside_the_flush_lock(
        self, global_config, embed_func, mock_client
    ):
        observed: list[bool] = []

        async def _probe(**_kwargs):
            observed.append(s._flush_lock.locked())
            raise missing_index_error()

        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            mock_client.indices.get_mapping = AsyncMock(side_effect=_probe)
            s._index_ready = False

            assert await s.get_by_id("v1") is None
            assert await s.get_vectors_by_ids(["v1"]) == {}

            assert observed == [False, False]

    @pytest.mark.asyncio
    async def test_presence_recheck_failure_keeps_the_mark(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            mock_client.indices.get_mapping = AsyncMock(
                side_effect=OpenSearchException("transport down")
            )
            s._index_ready = False

            with pytest.raises(OpenSearchException):
                await s.query("test", top_k=5)
            assert await s.get_by_id("v1") is None
            assert await s.get_vectors_by_ids(["v1"]) == {}
            assert s._index_ready is False
            mock_client.search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_presence_recheck_refuses_a_foreign_index(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            s._index_ready = False
            mock_client.indices.get_mapping = foreign_probe(s)

            for read in (
                lambda: s.query("test", top_k=5),
                lambda: s.get_by_id("v1"),
                lambda: s.get_by_ids(["v1"]),
                lambda: s.get_vectors_by_ids(["v1"]),
            ):
                with pytest.raises(WorkspaceIndexCollisionError):
                    await read()

            assert s._index_ready is False
            mock_client.search.assert_not_awaited()
            mock_client.mget.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_presence_recheck_adopts_an_unmarked_index(
        self, global_config, embed_func, mock_client
    ):
        mock_client.search = AsyncMock(return_value={"hits": {"hits": []}})
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            s._index_ready = False
            mock_client.indices.get_mapping = AsyncMock(
                return_value={s._index_name: {"mappings": {"properties": {}}}}
            )

            assert await s.query("test", top_k=5) == []
            assert s._index_ready is True

    @pytest.mark.asyncio
    async def test_probe_in_flight_does_not_outrank_a_later_missing_mark(
        self, global_config, embed_func, mock_client
    ):
        drop_ran = asyncio.Event()

        async def _probe_then_drop(**_kwargs):
            mock_client.indices.delete = AsyncMock()
            mock_client.indices.create = AsyncMock(
                side_effect=OpenSearchException("recreate failed")
            )
            await storage.drop()
            drop_ran.set()
            return {
                storage._index_name: {
                    "mappings": {
                        "_meta": _workspace_index_meta(
                            storage.workspace, storage.final_namespace
                        )
                    }
                }
            }

        with patch.object(ClientManager, "get_client", return_value=mock_client):
            storage = self._make(global_config, embed_func)
            await storage.initialize()
            storage._index_ready = False
            mock_client.indices.get_mapping = AsyncMock(side_effect=_probe_then_drop)

            assert await storage.query("test", top_k=5) == []

            assert drop_ran.is_set()
            assert storage._index_ready is False
            mock_client.search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_concurrent_404_probe_outranks_one_that_saw_the_index(
        self, global_config, embed_func, mock_client
    ):
        started = asyncio.Event()
        released = asyncio.Event()
        seen: list[str] = []

        async def _probe(**_kwargs):
            if not seen:
                seen.append("stale")
                started.set()
                await released.wait()
                return {
                    storage._index_name: {
                        "mappings": {
                            "_meta": _workspace_index_meta(
                                storage.workspace, storage.final_namespace
                            )
                        }
                    }
                }
            seen.append("fresh")
            raise missing_index_error()

        with patch.object(ClientManager, "get_client", return_value=mock_client):
            storage = self._make(global_config, embed_func)
            await storage.initialize()
            storage._index_ready = False
            mock_client.indices.get_mapping = AsyncMock(side_effect=_probe)

            stale = asyncio.create_task(storage.query("a", top_k=5))
            await started.wait()

            assert await storage.query("b", top_k=5) == []
            assert storage._index_ready is False

            released.set()
            assert await stale == []

            assert seen == ["stale", "fresh"]
            assert storage._index_ready is False
            mock_client.search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_presence_recheck_refuses_a_foreign_dimension(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            s._index_ready = False
            wrong = s.embedding_func.embedding_dim + 1
            mock_client.indices.get_mapping = owned_probe_with_dim(s, wrong)

            for read in (
                lambda: s.query("test", top_k=5),
                lambda: s.get_by_id("v1"),
                lambda: s.get_by_ids(["v1"]),
                lambda: s.get_vectors_by_ids(["v1"]),
            ):
                with pytest.raises(VectorSpaceMismatchError, match="dimension"):
                    await read()

            assert s._index_ready is False
            mock_client.search.assert_not_awaited()
            mock_client.mget.assert_not_awaited()


# ---------------------------------------------------------------------------
# Vector storage write batching (shared contract mixin + vector-specific behavior)
# ---------------------------------------------------------------------------


class TestVectorStorageBatching(BufferedWriteBatchingContractMixin):
    """Buffered upsert/delete + deferred embedding + flush behaviour."""

    upsert_op_type = "index"
    op_label = "vector"
    id_field = "id"

    def _make(self, global_config, embed_func, workspace: str = "test"):
        return _make_vector(global_config, embed_func, workspace)

    def _pending_upserts(self, storage):
        return storage._pending_vector_docs

    def _pending_deletes(self, storage):
        return storage._pending_vector_deletes

    def _action_content(self, action):
        return action["_source"]["content"]

    def _mget_expected_kwargs(self, storage, ids):
        return {
            "index": storage._index_name,
            "body": {"ids": ids},
            "_source_excludes": ["vector"],
        }

    def _mget_index_source(self):
        return {"content": "from_index", "vector": [0.2] * 128}

    def _assert_read_doc_clean(self, doc):
        assert "vector" not in doc

    def _assert_after_overwrite_flush(self, embed_func):
        assert embed_func.call_count == 1
        assert embed_func.texts == ["second"]

    def _assert_after_retryable_first_flush(self, storage, retained_id, embed_func):
        assert storage._pending_vector_docs[retained_id].vector is not None
        assert embed_func.call_count == 1

    def _assert_after_retryable_second_flush(self, embed_func):
        assert embed_func.call_count == 1

    @pytest.mark.asyncio
    async def test_repeated_upserts_flush_in_single_bulk(
        self, global_config, embed_func, mock_client
    ):
        """Deferred embeddings are batched by ``embedding_batch_num`` during flush."""
        counting_embed = CountingEmbeddingFunc()
        config = {**global_config, "embedding_batch_num": 2}
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (5, [])
                s = self._make(config, counting_embed)
                await s.initialize()
                for i in range(5):
                    await s.upsert({f"v{i}": {"content": f"doc {i}"}})
                mock_bulk.assert_not_awaited()
                assert counting_embed.call_count == 0
                await s.index_done_callback()
                assert counting_embed.batches == [
                    ["doc 0", "doc 1"],
                    ["doc 2", "doc 3"],
                    ["doc 4"],
                ]
                mock_bulk.assert_awaited_once()
                actions = mock_bulk.call_args[0][1]
                assert len(actions) == 5
                assert {a["_id"] for a in actions} == {f"v{i}" for i in range(5)}
                assert "refresh" not in mock_bulk.call_args.kwargs

    @pytest.mark.asyncio
    async def test_get_vectors_by_ids_embeds_buffer_once_and_flush_reuses_it(
        self, global_config, mock_client
    ):
        embed_func = CountingEmbeddingFunc()
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (1, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.upsert({"v1": {"content": "x"}})
                assert embed_func.call_count == 0
                vecs = await s.get_vectors_by_ids(["v1"])
                assert len(vecs["v1"]) == 128
                assert embed_func.call_count == 1
                mock_client.mget.assert_not_awaited()
                await s.index_done_callback()
                assert embed_func.call_count == 1
                actions = mock_bulk.call_args[0][1]
                assert actions[0]["_source"]["vector"] == vecs["v1"]

    @pytest.mark.asyncio
    async def test_embedding_failure_leaves_pending_for_retry(
        self, global_config, mock_client
    ):
        embed_func = CountingEmbeddingFunc(fail_times=1)
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (1, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.upsert({"v1": {"content": "retry me"}})

                with pytest.raises(RuntimeError, match="embedding failed"):
                    await s.index_done_callback()
                mock_bulk.assert_not_awaited()
                assert "v1" in s._pending_vector_docs
                assert s._pending_vector_docs["v1"].vector is None

                await s.index_done_callback()
                mock_bulk.assert_awaited_once()
                assert "v1" not in s._pending_vector_docs
                assert embed_func.call_count == 2

    @pytest.mark.asyncio
    async def test_delete_entity_relation_prunes_pending_buffer(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk",
                new_callable=AsyncMock,
                return_value=(1, []),
            ):
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.upsert(
                    {
                        "rel-1": {
                            "content": "Alice -> Bob",
                            "src_id": "Alice",
                            "tgt_id": "Bob",
                        },
                        "rel-2": {
                            "content": "Carol -> Dave",
                            "src_id": "Carol",
                            "tgt_id": "Dave",
                        },
                    }
                )
                await s.delete_entity_relation("Alice")
                assert "rel-1" not in s._pending_vector_docs
                assert "rel-2" in s._pending_vector_docs
                mock_client.delete_by_query.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_concurrent_delete_during_flush_supersedes_retried_upsert(
        self, global_config, embed_func, mock_client
    ):
        flush_started = asyncio.Event()
        flush_can_finish = asyncio.Event()

        async def slow_bulk(client, actions, raise_on_error=False, **kwargs):
            flush_started.set()
            await flush_can_finish.wait()
            return (
                0,
                [{"index": {"_id": "v1", "status": 503, "error": "down"}}],
            )

        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch("lightrag.kg.opensearch_impl.helpers.async_bulk", new=slow_bulk):
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.upsert({"v1": {"content": "first"}})

                flush_task = asyncio.create_task(s.index_done_callback())
                await flush_started.wait()
                delete_task = asyncio.create_task(s.delete(["v1"]))
                for _ in range(5):
                    await asyncio.sleep(0)
                assert not delete_task.done()

                flush_can_finish.set()
                await flush_task
                await delete_task
                assert "v1" not in s._pending_vector_docs
                assert "v1" in s._pending_vector_deletes

    @pytest.mark.asyncio
    async def test_get_by_id_strips_vector_from_mget_path(
        self, global_config, embed_func, mock_client
    ):
        mock_client.mget = AsyncMock(
            return_value={
                "docs": [
                    {
                        "_id": "v1",
                        "found": True,
                        "_source": {"content": "from_index", "vector": [0.1] * 128},
                    }
                ]
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            doc = await s.get_by_id("v1")
            assert doc is not None
            assert doc["id"] == "v1"
            assert doc["content"] == "from_index"
            assert "vector" not in doc
            mock_client.mget.assert_awaited_once_with(
                index=s._index_name,
                body={"ids": ["v1"]},
                _source_excludes=["vector"],
            )

    @pytest.mark.asyncio
    async def test_index_done_callback_flushes_when_index_recreated(
        self, global_config, embed_func, mock_client
    ):
        exists_responses = [False, False]
        mock_client.indices.exists = AsyncMock(
            side_effect=lambda **kw: (
                exists_responses.pop(0) if exists_responses else False
            )
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (1, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.upsert({"v1": {"content": "ok"}})
                s._mark_index_missing()
                await s.index_done_callback()
                mock_bulk.assert_awaited_once()
                assert s._pending_vector_docs == {}
                assert mock_client.indices.create.await_count >= 2

    @pytest.mark.asyncio
    async def test_delete_entity_relation_serialised_with_flush(
        self, global_config, embed_func, mock_client
    ):
        flush_started = asyncio.Event()
        flush_can_finish = asyncio.Event()
        delete_started = asyncio.Event()

        async def slow_bulk(client, actions, raise_on_error=False, **kwargs):
            flush_started.set()
            await flush_can_finish.wait()
            return (len(actions), [])

        async def watch_delete_by_query(**kwargs):
            delete_started.set()
            return {"deleted": 0}

        mock_client.delete_by_query = AsyncMock(side_effect=watch_delete_by_query)

        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch("lightrag.kg.opensearch_impl.helpers.async_bulk", new=slow_bulk):
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.upsert(
                    {
                        "rel-1": {
                            "content": "X",
                            "src_id": "Alice",
                            "tgt_id": "Bob",
                        }
                    }
                )

                flush_task = asyncio.create_task(s.index_done_callback())
                await flush_started.wait()
                rel_task = asyncio.create_task(s.delete_entity_relation("Alice"))
                for _ in range(5):
                    await asyncio.sleep(0)
                assert not delete_started.is_set()
                assert not rel_task.done()

                flush_can_finish.set()
                await flush_task
                await rel_task
                assert delete_started.is_set()

    @pytest.mark.asyncio
    async def test_drop_serialised_with_flush_embedding_phase(
        self, global_config, mock_client
    ):
        embedding_started = asyncio.Event()
        embedding_can_finish = asyncio.Event()
        drop_delete_started = asyncio.Event()

        class GatedEmbeddingFunc(MockEmbeddingFunc):
            async def __call__(self, texts, **kwargs):
                embedding_started.set()
                await embedding_can_finish.wait()
                return await super().__call__(texts, **kwargs)

        async def watch_indices_delete(**kwargs):
            drop_delete_started.set()

        mock_client.indices.delete = AsyncMock(side_effect=watch_indices_delete)
        embed_func = GatedEmbeddingFunc()

        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk",
                new_callable=AsyncMock,
                return_value=(1, []),
            ):
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.upsert({"v1": {"content": "x"}})

                flush_task = asyncio.create_task(s.index_done_callback())
                await embedding_started.wait()
                drop_task = asyncio.create_task(s.drop())
                for _ in range(5):
                    await asyncio.sleep(0)
                assert not drop_delete_started.is_set()
                assert not drop_task.done()

                embedding_can_finish.set()
                await flush_task
                await drop_task
                assert drop_delete_started.is_set()


# ---------------------------------------------------------------------------
# Vector flush durability (has_pending_index_ops against rate-limited 429s)
# ---------------------------------------------------------------------------


async def _ones_embed(texts, **kwargs):
    return np.ones((len(texts), 8), dtype=np.float32)


async def _make_durability_vector(
    server: FakeRateLimitedServer,
) -> OpenSearchVectorDBStorage:
    with patch.object(ClientManager, "get_client", return_value=server.client):
        storage = OpenSearchVectorDBStorage(
            namespace="entities",
            workspace="tenant",
            global_config=default_global_config(),
            embedding_func=EmbeddingFunc(
                embedding_dim=8,
                max_token_size=1024,
                func=_ones_embed,
                model_name="bge-m3",
            ),
            meta_fields={"entity_name"},
        )
        await storage.initialize()
    return storage


@pytest.mark.usefixtures("shared_storage_env")
class TestVectorDurability:
    """``OpenSearchVectorDBStorage.has_pending_index_ops`` against 429 bulk failures."""

    async def test_a_rate_limited_flush_leaves_the_vectors_pending(self):
        server = FakeRateLimitedServer(rate_limited=True)
        storage = await _make_durability_vector(server)

        with patch("lightrag.kg.opensearch_impl.helpers.async_bulk", new=server.bulk):
            await storage.upsert(
                {"ent-1": {"content": "Alice", "entity_name": "Alice"}},
            )
            await storage.index_done_callback()

            assert server.bulk_calls == 1
            assert server.docs == {}
            assert await storage.has_pending_index_ops() is True
            assert await storage.has_pending_index_ops(include_deletes=True) is True

    async def test_a_healthy_flush_leaves_nothing_pending(self):
        server = FakeRateLimitedServer(rate_limited=False)
        storage = await _make_durability_vector(server)

        with patch("lightrag.kg.opensearch_impl.helpers.async_bulk", new=server.bulk):
            await storage.upsert(
                {"ent-1": {"content": "Alice", "entity_name": "Alice"}}
            )
            await storage.index_done_callback()

            assert server.docs == {"ent-1": {"_id": "ent-1"}}
            assert await storage.has_pending_index_ops() is False
            assert await storage.has_pending_index_ops(include_deletes=True) is False

    async def test_a_retained_delete_is_counted_only_when_asked_for(self):
        server = FakeRateLimitedServer(rate_limited=False)
        storage = await _make_durability_vector(server)

        with patch("lightrag.kg.opensearch_impl.helpers.async_bulk", new=server.bulk):
            await storage.upsert(
                {"ent-1": {"content": "Alice", "entity_name": "Alice"}}
            )
            await storage.index_done_callback()

            server.rate_limited = True
            await storage.delete(["ent-1"])
            await storage.index_done_callback()

            assert await storage.has_pending_index_ops() is False
            assert await storage.has_pending_index_ops(include_deletes=True) is True


# ---------------------------------------------------------------------------
# Embedding-space provenance on the OpenSearch vector index
# ---------------------------------------------------------------------------


_VS_INDEX = "ws_entities"


def _vs_storage(
    global_config, embed, workspace: str = "ws", namespace: str = "entities"
):
    return OpenSearchVectorDBStorage(
        namespace=namespace,
        global_config=global_config,
        embedding_func=embed,
        workspace=workspace,
    )


async def _vs_initialize(storage, cluster: FakeCluster):
    with patch.object(ClientManager, "get_client", return_value=cluster.client()):
        await storage.initialize()
    return storage


class TestVectorSpaceProvenance:
    """Embedding-space provenance marker recording, attach gates, drop recovery, adoption."""

    @pytest.mark.asyncio
    async def test_every_created_index_records_its_embedding_space(
        self, global_config, cluster
    ):
        await _vs_initialize(
            _vs_storage(global_config, FixedEmbeddingFunc("bge-m3", 8)), cluster
        )

        meta = cluster.meta_of(_VS_INDEX)
        assert meta[VECTOR_SPACE_MODEL_KEY] == "bge-m3"
        assert meta[VECTOR_SPACE_DIM_KEY] == 8
        assert meta[_WORKSPACE_META_KEY] == "ws"
        assert meta[_FINAL_NAMESPACE_META_KEY] == "ws_entities"

    @pytest.mark.asyncio
    async def test_an_unknown_model_records_no_model_key(self, global_config, cluster):
        await _vs_initialize(
            _vs_storage(global_config, FixedEmbeddingFunc(None, 8)), cluster
        )

        meta = cluster.meta_of(_VS_INDEX)
        assert VECTOR_SPACE_MODEL_KEY not in meta
        assert meta[VECTOR_SPACE_DIM_KEY] == 8

    @pytest.mark.asyncio
    async def test_attaching_refuses_an_index_built_by_another_model(
        self, global_config, cluster
    ):
        cluster.seed_index(_VS_INDEX, model="bge-m3", dim=8)

        with pytest.raises(VectorSpaceMismatchError) as excinfo:
            await _vs_initialize(
                _vs_storage(global_config, FixedEmbeddingFunc("e5-large", 8)), cluster
            )

        message = str(excinfo.value)
        assert "'bge-m3' -> 'e5-large'" in message
        assert "lightrag-rebuild-vdb" in message
        assert "_mapping" not in message

    @pytest.mark.asyncio
    async def test_attaching_accepts_the_same_model(self, global_config, cluster):
        cluster.seed_index(_VS_INDEX, model="bge-m3", dim=8)
        storage = await _vs_initialize(
            _vs_storage(global_config, FixedEmbeddingFunc("bge-m3", 8)), cluster
        )
        assert storage._index_ready is True

    @pytest.mark.asyncio
    async def test_an_index_predating_the_marker_is_still_servable(
        self, global_config, cluster
    ):
        cluster.seed_index(_VS_INDEX, model=None, dim=8)
        storage = await _vs_initialize(
            _vs_storage(global_config, FixedEmbeddingFunc("bge-m3", 8)), cluster
        )
        assert storage._index_ready is True

    @pytest.mark.asyncio
    async def test_attaching_is_not_a_backfill(self, global_config, cluster):
        cluster.seed_index(_VS_INDEX, model=None, dim=8)
        await _vs_initialize(
            _vs_storage(global_config, FixedEmbeddingFunc("bge-m3", 8)), cluster
        )
        assert VECTOR_SPACE_MODEL_KEY not in cluster.meta_of(_VS_INDEX)

    @pytest.mark.asyncio
    async def test_attaching_refuses_a_foreign_dimension(self, global_config, cluster):
        cluster.seed_index(_VS_INDEX, model="bge-m3", dim=16)

        with pytest.raises(VectorSpaceMismatchError, match="16 -> 8"):
            await _vs_initialize(
                _vs_storage(global_config, FixedEmbeddingFunc("bge-m3", 8)), cluster
            )

    @pytest.mark.asyncio
    async def test_losing_the_create_race_still_validates_compatibility(
        self, global_config, cluster
    ):
        storage = _vs_storage(global_config, FixedEmbeddingFunc("e5-large", 8))
        client = cluster.client()

        async def _exists_then_lose(index):
            cluster.seed_index(index, model="bge-m3", dim=8)
            return False

        client.indices.exists = AsyncMock(side_effect=_exists_then_lose)

        with patch.object(ClientManager, "get_client", return_value=client):
            with pytest.raises(
                VectorSpaceMismatchError, match="'bge-m3' -> 'e5-large'"
            ):
                await storage.initialize()

        assert storage._index_ready is False

    @pytest.mark.asyncio
    async def test_losing_the_create_race_accepts_a_compatible_index(
        self, global_config, cluster
    ):
        storage = _vs_storage(global_config, FixedEmbeddingFunc("bge-m3", 8))
        client = cluster.client()

        async def _exists_then_lose(index):
            cluster.seed_index(index, model="bge-m3", dim=8)
            return False

        client.indices.exists = AsyncMock(side_effect=_exists_then_lose)

        with patch.object(ClientManager, "get_client", return_value=client):
            await storage.initialize()

        assert storage._index_ready is True

    @pytest.mark.asyncio
    async def test_an_unreadable_mapping_after_the_create_race_fails_closed(
        self, global_config, cluster
    ):
        storage = _vs_storage(global_config, FixedEmbeddingFunc("bge-m3", 8))
        client = cluster.client()
        calls = {"n": 0}

        async def _exists_then_lose(index):
            cluster.seed_index(index, model="bge-m3", dim=8)
            return False

        async def _get_mapping(index):
            calls["n"] += 1
            if calls["n"] <= 1:
                return {index: {"mappings": cluster.mappings[index]}}
            raise ConnectionError("cluster unreachable")

        client.indices.exists = AsyncMock(side_effect=_exists_then_lose)
        client.indices.get_mapping = AsyncMock(side_effect=_get_mapping)

        with patch.object(ClientManager, "get_client", return_value=client):
            with pytest.raises(ConnectionError):
                await storage.initialize()

        assert storage._index_ready is False

    @pytest.mark.asyncio
    async def test_presence_recheck_refuses_a_foreign_model(
        self, global_config, cluster
    ):
        storage = await _vs_initialize(
            _vs_storage(global_config, FixedEmbeddingFunc("bge-m3", 8)), cluster
        )
        storage._mark_index_missing()
        cluster.seed_index(_VS_INDEX, model="e5-large", dim=8)

        with pytest.raises(VectorSpaceMismatchError, match="'e5-large' -> 'bge-m3'"):
            await storage.query("anything", top_k=5)

        assert storage._index_ready is False

    @pytest.mark.asyncio
    async def test_presence_recheck_accepts_the_same_model(
        self, global_config, cluster
    ):
        storage = await _vs_initialize(
            _vs_storage(global_config, FixedEmbeddingFunc("bge-m3", 8)), cluster
        )
        storage._mark_index_missing()

        assert await storage.query("anything", top_k=5) == []
        assert storage._index_ready is True

    @pytest.mark.asyncio
    async def test_a_refused_instance_can_still_drop(self, global_config, cluster):
        cluster.seed_index(_VS_INDEX, model="bge-m3", dim=8)
        storage = _vs_storage(global_config, FixedEmbeddingFunc("e5-large", 8))

        with patch.object(ClientManager, "get_client", return_value=cluster.client()):
            with pytest.raises(VectorSpaceMismatchError):
                await storage.initialize()

            assert storage._flush_lock is not None
            assert storage.client is not None

            result = await storage.drop()

        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_drop_reprovisions_the_index_in_the_current_space(
        self, global_config, cluster
    ):
        cluster.seed_index(_VS_INDEX, model="bge-m3", dim=8)
        storage = _vs_storage(global_config, FixedEmbeddingFunc("e5-large", 16))
        client = cluster.client()

        with patch.object(ClientManager, "get_client", return_value=client):
            with pytest.raises(VectorSpaceMismatchError):
                await storage.initialize()

            assert (await storage.drop())["status"] == "success"

            meta = cluster.meta_of(_VS_INDEX)
            assert meta[VECTOR_SPACE_MODEL_KEY] == "e5-large"
            assert meta[VECTOR_SPACE_DIM_KEY] == 16
            assert (
                cluster.mappings[_VS_INDEX]["properties"]["vector"]["dimension"] == 16
            )

            await storage.initialize()

        assert storage._index_ready is True

    @pytest.mark.asyncio
    async def test_an_unmarked_index_reports_adoption_pending(
        self, global_config, cluster
    ):
        cluster.seed_index(_VS_INDEX, model=None, dim=8)
        storage = await _vs_initialize(
            _vs_storage(global_config, FixedEmbeddingFunc("bge-m3", 8)), cluster
        )

        assert await storage.vector_space_adoption_pending() is True

    @pytest.mark.asyncio
    async def test_a_marked_index_reports_nothing_pending(self, global_config, cluster):
        cluster.seed_index(_VS_INDEX, model="bge-m3", dim=8)
        storage = await _vs_initialize(
            _vs_storage(global_config, FixedEmbeddingFunc("bge-m3", 8)), cluster
        )

        assert await storage.vector_space_adoption_pending() is False

    @pytest.mark.asyncio
    async def test_an_index_this_process_created_reports_nothing_pending(
        self, global_config, cluster
    ):
        storage = await _vs_initialize(
            _vs_storage(global_config, FixedEmbeddingFunc("bge-m3", 8)), cluster
        )

        assert await storage.vector_space_adoption_pending() is False

    @pytest.mark.asyncio
    async def test_a_process_with_no_model_has_nothing_to_adopt(
        self, global_config, cluster
    ):
        cluster.seed_index(_VS_INDEX, model=None, dim=8)
        storage = await _vs_initialize(
            _vs_storage(global_config, FixedEmbeddingFunc(None, 8)), cluster
        )

        assert await storage.vector_space_adoption_pending() is False

    @pytest.mark.asyncio
    async def test_adoption_records_the_marker_and_ends_the_silence(
        self, global_config, cluster
    ):
        cluster.seed_index(_VS_INDEX, model=None, dim=8)
        storage = await _vs_initialize(
            _vs_storage(global_config, FixedEmbeddingFunc("bge-m3", 8)), cluster
        )

        assert await storage.adopt_vector_space() is True
        assert cluster.meta_of(_VS_INDEX)[VECTOR_SPACE_MODEL_KEY] == "bge-m3"
        assert await storage.vector_space_adoption_pending() is False

        peer = _vs_storage(global_config, FixedEmbeddingFunc("e5-large", 8))
        with pytest.raises(VectorSpaceMismatchError):
            await _vs_initialize(peer, cluster)

    @pytest.mark.asyncio
    async def test_adoption_keeps_the_workspace_identity(self, global_config, cluster):
        cluster.seed_index(_VS_INDEX, model=None, dim=8)
        storage = await _vs_initialize(
            _vs_storage(global_config, FixedEmbeddingFunc("bge-m3", 8)), cluster
        )

        await storage.adopt_vector_space()

        meta = cluster.meta_of(_VS_INDEX)
        assert meta[_WORKSPACE_META_KEY] == "ws"
        assert meta[_FINAL_NAMESPACE_META_KEY] == "ws_entities"

    @pytest.mark.asyncio
    async def test_a_denied_mapping_write_reports_failure_instead_of_raising(
        self, global_config, cluster
    ):
        cluster.seed_index(_VS_INDEX, model=None, dim=8)
        storage = await _vs_initialize(
            _vs_storage(global_config, FixedEmbeddingFunc("bge-m3", 8)), cluster
        )
        storage.client.indices.put_mapping = AsyncMock(
            side_effect=RuntimeError("cluster_block_exception")
        )

        assert await storage.adopt_vector_space() is False
        assert VECTOR_SPACE_MODEL_KEY not in cluster.meta_of(_VS_INDEX)
