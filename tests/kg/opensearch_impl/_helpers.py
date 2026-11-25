"""Shared test doubles, lock fixtures, and storage contract mixins for OpenSearch tests.

This module intentionally has no ``test_`` prefix and its mixins do not start
with ``Test`` so pytest never collects it directly. Offline test modules import
these helpers and wrap ``patch_opensearch_locks`` in their own autouse fixtures
so ``test_opensearch_kv_create_time_integration.py`` stays completely isolated.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

pytest.importorskip(
    "opensearchpy",
    reason="opensearchpy is required for OpenSearch storage tests",
)

from opensearchpy import AsyncOpenSearch  # noqa: E402
from opensearchpy.exceptions import (  # type: ignore  # noqa: E402
    NotFoundError,
    OpenSearchException,
    RequestError,
    TransportError,
)

from lightrag.exceptions import flush_may_have_lost_reference  # noqa: E402
from lightrag.kg.opensearch_impl import (  # noqa: E402
    _FINAL_NAMESPACE_META_KEY,
    _WORKSPACE_META_KEY,
    ClientManager,
    OpenSearchReferencesIntactError,
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


# ---------------------------------------------------------------------------
# Lock patching & shared-storage context helpers
# ---------------------------------------------------------------------------


@asynccontextmanager
async def mock_lock():
    """No-op async lock used to bypass shared-storage init lock in unit tests."""
    yield


def mock_lock_factory(*_args, **_kwargs):
    return mock_lock()


@contextmanager
def patch_opensearch_locks(
    *,
    patch_shard_doc: bool = True,
    init_shared_storage: bool = False,
):
    """Patch OpenSearch storage locks to per-(namespace, workspace) asyncio locks.

    Returning real ``asyncio.Lock`` instances preserves the in-process blocking
    semantics the storages rely on, so concurrent flush / read / write tests
    observe genuine serialization.
    """
    cache: dict[tuple[str, str], asyncio.Lock] = {}

    def namespace_lock(namespace, workspace=None, enable_logging=False):
        return cache.setdefault((namespace, workspace or ""), asyncio.Lock())

    if init_shared_storage:
        initialize_share_data(workers=1)
    try:
        with (
            patch(
                "lightrag.kg.opensearch_impl.get_data_init_lock",
                side_effect=mock_lock_factory,
            ),
            patch(
                "lightrag.kg.opensearch_impl.get_namespace_lock",
                side_effect=namespace_lock,
            ),
        ):
            if patch_shard_doc:
                with patch("lightrag.kg.opensearch_impl._shard_doc_supported", True):
                    yield
            else:
                yield
    finally:
        if init_shared_storage:
            finalize_share_data()


# ---------------------------------------------------------------------------
# Standard OpenSearch error factories & readiness probes
# ---------------------------------------------------------------------------


def missing_index_error() -> NotFoundError:
    return NotFoundError(404, "index_not_found_exception", "no such index")


def doc_missing_error() -> NotFoundError:
    """Document-level 404 on an existing index (never mentions index_not_found)."""
    return NotFoundError(404, "document_missing_exception", "no such document")


def transient_error() -> TransportError:
    return TransportError(503, "search_phase_execution_exception", "shard failure")


def gone_probe() -> AsyncMock:
    """``indices.get_mapping`` for an index that is still absent."""
    return AsyncMock(side_effect=missing_index_error())


def owned_probe(storage) -> AsyncMock:
    """``indices.get_mapping`` for an index that is back and marked as ours."""
    return AsyncMock(
        return_value={
            storage._index_name: {
                "mappings": {
                    "_meta": _workspace_index_meta(
                        storage.workspace, storage.final_namespace
                    )
                }
            }
        }
    )


def owned_probe_with_dim(storage, dimension: int) -> AsyncMock:
    """``indices.get_mapping`` for our index declaring a knn_vector dimension."""
    return AsyncMock(
        return_value={
            storage._index_name: {
                "mappings": {
                    "_meta": _workspace_index_meta(
                        storage.workspace, storage.final_namespace
                    ),
                    "properties": {
                        "vector": {"type": "knn_vector", "dimension": dimension}
                    },
                }
            }
        }
    )


def foreign_probe(storage) -> AsyncMock:
    """``indices.get_mapping`` for an index a different workspace has claimed."""
    return AsyncMock(
        return_value={
            storage._index_name: {
                "mappings": {"_meta": _workspace_index_meta("other_ws", "other_ns")}
            }
        }
    )


# ---------------------------------------------------------------------------
# Embedding test doubles & default config
# ---------------------------------------------------------------------------


class MockEmbeddingFunc:
    """Mock embedding function returning random float32 vectors."""

    def __init__(self, dim: int = 128, model_name: str | None = "mock-embed"):
        self.embedding_dim = dim
        self.max_token_size = 512
        self.model_name = model_name

    async def __call__(self, texts, **kwargs):
        return np.random.rand(len(texts), self.embedding_dim).astype(np.float32)


class CountingEmbeddingFunc(MockEmbeddingFunc):
    """Embedding test double that records calls and can fail N times."""

    def __init__(self, dim: int = 128, fail_times: int = 0):
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


class FixedEmbeddingFunc:
    """Deterministic zero-vector embedding double with configurable model/dim."""

    max_token_size = 100

    def __init__(self, model_name: str | None = "model-a", embedding_dim: int = 8):
        self.model_name = model_name
        self.embedding_dim = embedding_dim

    async def __call__(self, texts, **kwargs):
        return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)


class RejectingEmbedFunc:
    """Embedding double that fails loudly if a non-embedding path invokes it."""

    embedding_dim = 8
    max_token_size = 512
    model_name = "mock-embed"

    async def __call__(self, texts, **kwargs):
        raise AssertionError("read/scheduling paths must not embed")


def default_global_config() -> dict[str, Any]:
    return {
        "embedding_batch_num": 10,
        "max_graph_nodes": 1000,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
    }


# ---------------------------------------------------------------------------
# Client factories & mget side effects
# ---------------------------------------------------------------------------


def migrated_mapping(index: str, **_kwargs) -> dict[str, Any]:
    """Mapping of an index that already carries the ``__mirrored_id`` tiebreaker."""
    return {index: {"mappings": {"properties": {"__mirrored_id": {"type": "keyword"}}}}}


def node_mget_side_effect(existing: set[str]):
    """Well-formed node ``mget`` echoing every requested id, found iff in ``existing``."""

    async def _mget(index=None, body=None, **kwargs):
        return {
            "docs": [
                {"_id": node_id, "found": True, "_source": {"entity_id": node_id}}
                if node_id in existing
                else {"_id": node_id, "found": False}
                for node_id in body["ids"]
            ]
        }

    return _mget


def mget_by_ids_side_effect(node_sources: dict[str, dict[str, Any]]):
    """Build an ``mget`` side effect returning one doc per requested id."""

    async def _side_effect(index=None, body=None, **kwargs):
        docs = []
        for doc_id in (body or {}).get("ids", []):
            source = node_sources.get(doc_id)
            if source is None:
                docs.append({"_id": doc_id, "found": False})
            else:
                docs.append({"_id": doc_id, "found": True, "_source": dict(source)})
        return {"docs": docs}

    return _side_effect


def make_client() -> AsyncMock:
    """Create a fully-mocked ``AsyncOpenSearch`` client with spec validation."""
    client = AsyncMock(spec=AsyncOpenSearch)
    client.indices = AsyncMock()
    client.indices.exists = AsyncMock(return_value=False)
    client.indices.create = AsyncMock()
    client.indices.delete = AsyncMock()
    client.indices.refresh = AsyncMock()
    client.indices.get_mapping = AsyncMock(return_value={})
    client.indices.put_mapping = AsyncMock()
    client.transport = AsyncMock()
    client.transport.perform_request = AsyncMock(
        side_effect=Exception("PPL not available")
    )
    client.exists = AsyncMock(return_value=False)
    client.index = AsyncMock()
    client.delete = AsyncMock()
    client.delete_by_query = AsyncMock()
    client.get = AsyncMock(
        return_value={
            "_id": "doc1",
            "_source": {"content": "hello", "create_time": 0, "update_time": 0},
        }
    )
    client.mget = AsyncMock(
        return_value={
            "docs": [
                {"_id": "id1", "found": True, "_source": {"content": "c1"}},
                {"_id": "id2", "found": True, "_source": {"content": "c2"}},
            ]
        }
    )
    client.count = AsyncMock(return_value={"count": 5})
    client.search = AsyncMock(
        return_value={
            "hits": {"hits": [], "total": {"value": 0}},
            "aggregations": {
                "status_counts": {"buckets": []},
                "src": {"buckets": []},
                "tgt": {"buckets": []},
                "source_degrees": {"ids": {"buckets": []}},
                "target_degrees": {"ids": {"buckets": []}},
            },
        }
    )
    client.create_pit = AsyncMock(return_value={"pit_id": "mock_pit_id_123"})
    client.delete_pit = AsyncMock()
    return client


def make_strict_client() -> AsyncMock:
    """Mock client for pre-existing migrated indices (strict reads / scheduling)."""
    client = AsyncMock(spec=AsyncOpenSearch)
    client.indices = AsyncMock()
    client.indices.exists = AsyncMock(return_value=True)
    client.indices.create = AsyncMock()
    client.indices.get_mapping = AsyncMock(side_effect=migrated_mapping)
    client.indices.put_mapping = AsyncMock()
    client.get = AsyncMock(return_value={})
    client.index = AsyncMock(return_value={})
    client.update = AsyncMock(return_value={})
    client.count = AsyncMock(return_value={"count": 0})
    client.search = AsyncMock(return_value={"hits": {"hits": []}})
    client.update_by_query = AsyncMock(return_value={"updated": 0, "failures": []})
    client.create_pit = AsyncMock(return_value={"pit_id": "pit-1"})
    client.delete_pit = AsyncMock()
    return client


# ---------------------------------------------------------------------------
# Stateful OpenSearch cluster & rate-limited server fakes
# ---------------------------------------------------------------------------


class FakeCluster:
    """Minimal stateful stand-in for one shared OpenSearch cluster.

    Index mappings live in a dict so two storage instances built against one
    ``FakeCluster`` genuinely meet on the same index (used for vector-space
    provenance and workspace-index collision tests).
    """

    def __init__(self):
        self.mappings: dict[str, dict[str, Any]] = {}

    async def exists(self, index: str) -> bool:
        return index in self.mappings

    async def create(self, index: str, body: dict[str, Any] | None = None) -> None:
        if index in self.mappings:
            raise RequestError(
                400,
                "resource_already_exists_exception",
                {"error": "resource_already_exists_exception"},
            )
        self.mappings[index] = dict((body or {}).get("mappings", {}))

    async def get_mapping(self, index: str) -> dict[str, Any]:
        if index not in self.mappings:
            return {}
        return {index: {"mappings": self.mappings[index]}}

    async def put_mapping(self, index: str, body: dict[str, Any]) -> None:
        mappings = self.mappings[index]
        if "_meta" in body:
            mappings["_meta"] = body["_meta"]
        if "properties" in body:
            mappings.setdefault("properties", {}).update(body["properties"])

    async def delete(self, index: str, **_kwargs) -> None:
        self.mappings.pop(index, None)

    async def refresh(self, index: str | None = None, **_kwargs) -> None:
        return None

    def client(self) -> AsyncMock:
        client = AsyncMock(spec=AsyncOpenSearch)
        client.indices = AsyncMock()
        client.indices.exists = AsyncMock(side_effect=self.exists)
        client.indices.create = AsyncMock(side_effect=self.create)
        client.indices.get_mapping = AsyncMock(side_effect=self.get_mapping)
        client.indices.put_mapping = AsyncMock(side_effect=self.put_mapping)
        client.indices.delete = AsyncMock(side_effect=self.delete)
        client.indices.refresh = AsyncMock(side_effect=self.refresh)
        client.transport = AsyncMock()
        client.transport.perform_request = AsyncMock(
            side_effect=Exception("PPL not available")
        )
        client.count = AsyncMock(return_value={"count": 0})
        client.search = AsyncMock(
            return_value={"hits": {"hits": [], "total": {"value": 0}}}
        )
        client.mget = AsyncMock(return_value={"docs": []})
        return client

    def meta_of(self, index: str) -> dict[str, Any]:
        return self.mappings[index].get("_meta", {})

    def seed_index(
        self,
        index: str,
        *,
        model: str | None,
        dim: int | None,
        workspace: str = "ws",
        namespace: str = "entities",
    ) -> None:
        """Place a vector index built by another process/configuration."""
        meta: dict[str, Any] = {
            _WORKSPACE_META_KEY: workspace,
            _FINAL_NAMESPACE_META_KEY: f"{workspace}_{namespace}",
        }
        if model is not None:
            meta[VECTOR_SPACE_MODEL_KEY] = model
        if dim is not None:
            meta[VECTOR_SPACE_DIM_KEY] = dim
        self.mappings[index] = {
            "properties": {"vector": {"type": "knn_vector", "dimension": dim}},
            "_meta": meta,
        }


class FakeRateLimitedServer:
    """Document map + bulk endpoint that either applies actions or rejects with 429.

    Shared by configuration, chunk-tracking, and vector durability tests.
    """

    def __init__(self, *, rate_limited: bool = False):
        self.docs: dict[str, dict[str, Any]] = {}
        self.rate_limited = rate_limited
        self.bulk_calls = 0
        self.client = AsyncMock()
        self.client.indices = AsyncMock()
        self.client.indices.exists = AsyncMock(return_value=True)
        self.client.indices.create = AsyncMock()
        self.client.indices.refresh = AsyncMock()
        self.client.indices.get_mapping = AsyncMock(return_value={})
        self.client.mget = AsyncMock(side_effect=self._mget)

    async def _mget(self, *, index, body, **_kwargs):
        items = []
        for doc_id in body["ids"]:
            source = self.docs.get(doc_id)
            if source is None:
                items.append({"_id": doc_id, "found": False})
            else:
                items.append({"_id": doc_id, "found": True, "_source": dict(source)})
        return {"docs": items}

    async def bulk(self, client, actions, **_kwargs):
        self.bulk_calls += 1
        actions = list(actions)
        if self.rate_limited:
            failed = [
                {
                    action["_op_type"]: {
                        "_id": action["_id"],
                        "status": 429,
                        "error": "rate limited",
                    }
                }
                for action in actions
            ]
            return 0, failed
        for action in actions:
            if action["_op_type"] == "delete":
                self.docs.pop(action["_id"], None)
            elif "script" in action:
                self.docs[action["_id"]] = dict(action["script"]["params"]["doc"])
            else:
                self.docs[action["_id"]] = {"_id": action["_id"]}
        return len(actions), []


# ---------------------------------------------------------------------------
# Shared readiness / drop contract mixins (KV, DocStatus, Graph, Vector)
# ---------------------------------------------------------------------------


class ReadinessDemotionMixin:
    """Shared contract: a live missing-index read error demotes readiness once."""

    readiness_attr: str = "_index_ready"
    missing_read_rpc: str = "mget"

    def _configure_missing_read(self, storage, mock_client) -> None:
        setattr(
            mock_client,
            self.missing_read_rpc,
            AsyncMock(side_effect=missing_index_error()),
        )

    async def _invoke_missing_read(self, storage) -> Any:
        raise NotImplementedError

    def _expected_missing_read_result(self) -> Any:
        return None

    @pytest.mark.asyncio
    async def test_read_missing_index_demotes_readiness(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            self._configure_missing_read(s, mock_client)

            expected = self._expected_missing_read_result()
            assert await self._invoke_missing_read(s) == expected
            assert getattr(s, self.readiness_attr) is False
            assert await self._invoke_missing_read(s) == expected
            assert getattr(mock_client, self.missing_read_rpc).await_count == 1


class LazyDropRecreateMixin:
    """Shared contract for storages whose ``drop()`` recreates lazily on next write."""

    readiness_attr: str = "_index_ready"
    create_method_name: str = "_create_index_if_not_exists"
    expected_drop_error_substr: str = ""

    def _configure_drop_failure(self, mock_client) -> None:
        mock_client.indices.delete = AsyncMock(
            side_effect=OpenSearchException("drop failed")
        )

    async def _write_to_recreate(self, storage) -> None:
        raise NotImplementedError

    @pytest.mark.asyncio
    async def test_drop_error_marks_not_ready_and_next_write_recreates_index(
        self, global_config, embed_func, mock_client
    ):
        self._configure_drop_failure(mock_client)
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk",
                new_callable=AsyncMock,
                return_value=(1, []),
            ):
                s = self._make(global_config, embed_func)
                await s.initialize()
                with patch.object(
                    s, self.create_method_name, new_callable=AsyncMock
                ) as mock_create:
                    result = await s.drop()
                    assert result["status"] == "error"
                    if self.expected_drop_error_substr:
                        assert self.expected_drop_error_substr in result["message"]
                    assert getattr(s, self.readiness_attr) is False
                    await self._write_to_recreate(s)
                    mock_create.assert_awaited_once()


# ---------------------------------------------------------------------------
# Unified KV / Vector buffered-write batching contract mixin
# ---------------------------------------------------------------------------


class BufferedWriteBatchingContractMixin:
    """Shared contract tests for buffered upsert/delete + flush behavior.

    Inherited by ``TestKVStorageBatching`` (KV) and ``TestVectorStorageBatching``
    (Vector) so every shared invariant is pinned identically on both storages
    without duplicating test bodies.
    """

    upsert_op_type: str  # "update" (KV) or "index" (Vector)
    op_label: str  # "KV" or "vector"
    id_field: str  # "_id" (KV) or "id" (Vector)

    def _make(self, global_config, embed_func, workspace: str = "test"):
        raise NotImplementedError

    def _pending_upserts(self, storage) -> dict[str, Any]:
        raise NotImplementedError

    def _pending_deletes(self, storage) -> set[str]:
        raise NotImplementedError

    def _action_content(self, action: dict[str, Any]) -> str:
        raise NotImplementedError

    def _mget_expected_kwargs(self, storage, ids: list[str]) -> dict[str, Any]:
        return {"index": storage._index_name, "body": {"ids": ids}}

    def _mget_index_source(self) -> dict[str, Any]:
        return {"content": "from_index"}

    def _assert_read_doc_clean(self, doc: dict[str, Any]) -> None:
        pass

    def _assert_after_retryable_first_flush(
        self, storage, retained_id: str, embed_func: CountingEmbeddingFunc
    ) -> None:
        pass

    def _assert_after_retryable_second_flush(
        self, embed_func: CountingEmbeddingFunc
    ) -> None:
        pass

    def _assert_after_overwrite_flush(self, embed_func: CountingEmbeddingFunc) -> None:
        pass

    @pytest.mark.asyncio
    async def test_repeated_upserts_flush_in_single_bulk(
        self, global_config, embed_func, mock_client
    ):
        """Many small upsert() calls collapse to one async_bulk on flush."""
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (5, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                for i in range(5):
                    await s.upsert({f"k{i}": {"content": f"doc {i}"}})
                mock_bulk.assert_not_awaited()
                await s.index_done_callback()
                mock_bulk.assert_awaited_once()
                actions = mock_bulk.call_args[0][1]
                assert len(actions) == 5
                assert {a["_id"] for a in actions} == {f"k{i}" for i in range(5)}
                # Flush must not request a per-operation refresh.
                assert "refresh" not in mock_bulk.call_args.kwargs

    @pytest.mark.asyncio
    async def test_upsert_overwrites_pending_doc_for_same_id(
        self, global_config, mock_client
    ):
        """Upserting the same id twice keeps only the latest payload."""
        embed_func = CountingEmbeddingFunc()
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (1, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.upsert({"k1": {"content": "first"}})
                await s.upsert({"k1": {"content": "second"}})
                await s.index_done_callback()
                actions = mock_bulk.call_args[0][1]
                assert len(actions) == 1
                assert self._action_content(actions[0]) == "second"
                self._assert_after_overwrite_flush(embed_func)

    @pytest.mark.asyncio
    async def test_delete_cancels_pending_upsert(
        self, global_config, embed_func, mock_client
    ):
        """A delete after a buffered upsert removes the upsert from the buffer."""
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (1, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.upsert({"k1": {"content": "doomed"}})
                await s.delete(["k1"])
                assert "k1" not in self._pending_upserts(s)
                assert "k1" in self._pending_deletes(s)
                await s.index_done_callback()
                actions = mock_bulk.call_args[0][1]
                assert len(actions) == 1
                assert actions[0]["_op_type"] == "delete"

    @pytest.mark.asyncio
    async def test_upsert_cancels_pending_delete(
        self, global_config, embed_func, mock_client
    ):
        """An upsert after a buffered delete removes the tombstone."""
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (1, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.delete(["k1"])
                await s.upsert({"k1": {"content": "resurrected"}})
                assert "k1" not in self._pending_deletes(s)
                assert "k1" in self._pending_upserts(s)
                await s.index_done_callback()
                actions = mock_bulk.call_args[0][1]
                assert len(actions) == 1
                assert actions[0]["_op_type"] == self.upsert_op_type

    @pytest.mark.asyncio
    async def test_get_by_id_reads_pending_buffer(
        self, global_config, embed_func, mock_client
    ):
        """Buffered upserts are visible to get_by_id without hitting OpenSearch."""
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.upsert({"k1": {"content": "buffered"}})
            doc = await s.get_by_id("k1")
            assert doc is not None
            assert doc[self.id_field] == "k1"
            assert doc["content"] == "buffered"
            self._assert_read_doc_clean(doc)
            mock_client.mget.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_by_id_returns_none_for_pending_delete(
        self, global_config, embed_func, mock_client
    ):
        """A pending tombstone shadows any persisted doc without an mget RTT."""
        mock_client.mget = AsyncMock()
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.delete(["k1"])
            assert await s.get_by_id("k1") is None
            mock_client.mget.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_by_ids_merges_buffer_and_index(
        self, global_config, embed_func, mock_client
    ):
        """get_by_ids returns buffered docs and falls back to mget for the rest."""
        mock_client.mget = AsyncMock(
            return_value={
                "docs": [
                    {
                        "_id": "k2",
                        "found": True,
                        "_source": self._mget_index_source(),
                    }
                ]
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.upsert({"k1": {"content": "buffered"}})
            docs = await s.get_by_ids(["k1", "k2"])
            assert [d[self.id_field] for d in docs] == ["k1", "k2"]
            assert docs[0]["content"] == "buffered"
            assert docs[1]["content"] == "from_index"
            for doc in docs:
                self._assert_read_doc_clean(doc)
            mock_client.mget.assert_awaited_once_with(
                **self._mget_expected_kwargs(s, ["k2"])
            )

    @pytest.mark.asyncio
    async def test_finalize_flushes_pending_ops(
        self, global_config, embed_func, mock_client
    ):
        """finalize() flushes buffered writes before releasing the client."""
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (1, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.upsert({"k1": {"content": "to flush"}})
                await s.finalize()
                mock_bulk.assert_awaited_once()
                assert s.client is None

    @pytest.mark.asyncio
    async def test_finalize_raises_when_retryable_buffer_remains(
        self, global_config, embed_func, mock_client
    ):
        """finalize() raises RuntimeError when retryable bulk failures remain,
        while still releasing the client."""
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch.object(
                ClientManager, "release_client", new_callable=AsyncMock
            ) as mock_release:
                with patch(
                    "lightrag.kg.opensearch_impl.helpers.async_bulk",
                    new_callable=AsyncMock,
                ) as mock_bulk:
                    mock_bulk.return_value = (
                        0,
                        [
                            {
                                self.upsert_op_type: {
                                    "_id": "k1",
                                    "status": 503,
                                    "error": "down",
                                }
                            }
                        ],
                    )
                    s = self._make(global_config, embed_func)
                    await s.initialize()
                    await s.upsert({"k1": {"content": "stuck"}})
                    with pytest.raises(RuntimeError, match="pending upserts"):
                        await s.finalize()
                    mock_release.assert_awaited_once()
                    assert s.client is None

    @pytest.mark.asyncio
    async def test_finalize_propagates_flush_exception(
        self, global_config, embed_func, mock_client
    ):
        """If async_bulk raises during finalize(), release the client and wrap
        the original error in RuntimeError."""
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch.object(
                ClientManager, "release_client", new_callable=AsyncMock
            ) as mock_release:
                with patch(
                    "lightrag.kg.opensearch_impl.helpers.async_bulk",
                    new_callable=AsyncMock,
                ) as mock_bulk:
                    mock_bulk.side_effect = OpenSearchException("connection reset")
                    s = self._make(global_config, embed_func)
                    await s.initialize()
                    await s.upsert({"k1": {"content": "stuck"}})
                    with pytest.raises(RuntimeError) as exc_info:
                        await s.finalize()
                    assert isinstance(exc_info.value.__cause__, OpenSearchException)
                    mock_release.assert_awaited_once()
                    assert s.client is None

    @pytest.mark.asyncio
    async def test_finalize_propagates_cancellation(
        self, global_config, embed_func, mock_client
    ):
        """asyncio.CancelledError propagates unwrapped while still releasing the client."""
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch.object(
                ClientManager, "release_client", new_callable=AsyncMock
            ) as mock_release:
                with patch(
                    "lightrag.kg.opensearch_impl.helpers.async_bulk",
                    new_callable=AsyncMock,
                ) as mock_bulk:
                    mock_bulk.side_effect = asyncio.CancelledError()
                    s = self._make(global_config, embed_func)
                    await s.initialize()
                    await s.upsert({"k1": {"content": "stuck"}})
                    with pytest.raises(asyncio.CancelledError):
                        await s.finalize()
                    mock_release.assert_awaited_once()
                    assert s.client is None

    @pytest.mark.asyncio
    async def test_drop_discards_pending_buffers(
        self, global_config, embed_func, mock_client
    ):
        """drop() discards pending writes without flushing them to a deleted index."""
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.upsert({"k1": {"content": "doomed"}})
                await s.delete(["k2"])
                await s.drop()
                assert self._pending_upserts(s) == {}
                assert self._pending_deletes(s) == set()
                mock_bulk.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_drop_serialised_with_flush(
        self, global_config, embed_func, mock_client
    ):
        """drop() is serialised with any in-flight flush so indices.delete cannot land mid-bulk."""
        flush_started = asyncio.Event()
        flush_can_finish = asyncio.Event()
        drop_delete_started = asyncio.Event()

        async def slow_bulk(client, actions, raise_on_error=False, **kwargs):
            flush_started.set()
            await flush_can_finish.wait()
            return (len(actions), [])

        async def watch_indices_delete(**kwargs):
            drop_delete_started.set()

        mock_client.indices.delete = AsyncMock(side_effect=watch_indices_delete)
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch("lightrag.kg.opensearch_impl.helpers.async_bulk", new=slow_bulk):
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.upsert({"k1": {"content": "x"}})
                await s.delete(["k2"])

                flush_task = asyncio.create_task(s.index_done_callback())
                await asyncio.wait_for(flush_started.wait(), 5)
                drop_task = asyncio.create_task(s.drop())
                for _ in range(5):
                    await asyncio.sleep(0)
                assert not drop_delete_started.is_set(), (
                    "indices.delete should be blocked behind the flush lock"
                )
                assert not drop_task.done()

                flush_can_finish.set()
                await flush_task
                await drop_task
                assert drop_delete_started.is_set()
                assert self._pending_upserts(s) == {}
                assert self._pending_deletes(s) == set()

    @pytest.mark.asyncio
    async def test_failed_flush_retains_retryable(self, global_config, mock_client):
        """Transient (5xx) per-doc failures stay buffered and land on the next flush."""
        embed_func = CountingEmbeddingFunc()
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.side_effect = [
                    (
                        1,
                        [
                            {
                                self.upsert_op_type: {
                                    "_id": "k2",
                                    "status": 503,
                                    "error": "down",
                                }
                            }
                        ],
                    ),
                    (1, []),
                ]
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.upsert({"k1": {"content": "ok"}, "k2": {"content": "boom"}})
                await s.index_done_callback()
                assert "k1" not in self._pending_upserts(s)
                assert "k2" in self._pending_upserts(s)
                self._assert_after_retryable_first_flush(s, "k2", embed_func)

                await s.index_done_callback()
                assert "k2" not in self._pending_upserts(s)
                assert mock_bulk.await_count == 2
                self._assert_after_retryable_second_flush(embed_func)

    @pytest.mark.asyncio
    async def test_failed_flush_raises_on_non_retryable(
        self, global_config, embed_func, mock_client
    ):
        """Permanent 4xx failures raise with a sample, drop the permanent ops,
        retain retryable ops, and never claim references stayed intact."""
        failed = [
            {
                self.upsert_op_type: {
                    "_id": f"k{i}",
                    "status": 400,
                    "error": {
                        "type": "mapper_parsing_exception",
                        "reason": f"bad field {i}",
                    },
                }
            }
            for i in range(6)
        ] + [
            {
                self.upsert_op_type: {
                    "_id": "k6",
                    "status": 503,
                    "error": "down",
                }
            }
        ]
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (0, failed)
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.upsert({f"k{i}": {"content": f"d{i}"} for i in range(7)})
                with pytest.raises(RuntimeError, match="failed permanently") as excinfo:
                    await s.index_done_callback()
                assert set(self._pending_upserts(s)) == {"k6"}
                assert not isinstance(excinfo.value, OpenSearchReferencesIntactError)
                assert flush_may_have_lost_reference(excinfo.value) is True
        error_text = str(excinfo.value)
        for i in range(5):
            assert f"k{i}" in error_text
        assert "status=400" in error_text
        assert "bad field" in error_text
        assert f"6 {self.op_label} ops" in error_text

    @pytest.mark.asyncio
    async def test_drop_pending_index_ops_clears_buffers(
        self, global_config, embed_func, mock_client
    ):
        """drop_pending_index_ops clears both pending upserts and pending deletes."""
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.upsert({"k1": {"content": "x"}, "k2": {"content": "y"}})
            self._pending_deletes(s).add("old-id")
            assert self._pending_upserts(s)
            await s.drop_pending_index_ops()
            assert not self._pending_upserts(s)
            assert not self._pending_deletes(s)

    @pytest.mark.asyncio
    async def test_concurrent_writes_during_flush_are_serialised(
        self, global_config, embed_func, mock_client
    ):
        """An upsert issued while async_bulk is in flight blocks until flush finishes."""
        flush_started = asyncio.Event()
        flush_can_finish = asyncio.Event()

        async def slow_bulk(client, actions, raise_on_error=False, **kwargs):
            flush_started.set()
            await flush_can_finish.wait()
            return (len(actions), [])

        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch("lightrag.kg.opensearch_impl.helpers.async_bulk", new=slow_bulk):
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.upsert({"k1": {"content": "first"}})

                flush_task = asyncio.create_task(s.index_done_callback())
                await asyncio.wait_for(flush_started.wait(), 5)
                concurrent_task = asyncio.create_task(
                    s.upsert({"k2": {"content": "concurrent"}})
                )
                for _ in range(5):
                    await asyncio.sleep(0)
                assert not concurrent_task.done(), (
                    "concurrent upsert should be blocked by the flush lock"
                )
                assert "k2" not in self._pending_upserts(s)

                flush_can_finish.set()
                await flush_task
                await concurrent_task
                assert "k1" not in self._pending_upserts(s)
                assert "k2" in self._pending_upserts(s)

    @pytest.mark.asyncio
    async def test_flush_splits_delete_and_upsert_into_separate_phases(
        self, global_config, embed_func, mock_client
    ):
        """A mixed flush issues separate delete and upsert bulks with distinct caps."""
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (1, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                s._max_upsert_records_per_batch = 11
                s._max_delete_records_per_batch = 22
                await s.upsert({"keep": {"content": "v"}})
                await s.delete(["gone"])
                await s.index_done_callback()

                assert mock_bulk.await_count == 2
                by_op = {
                    call.args[1][0]["_op_type"]: call.kwargs["chunk_size"]
                    for call in mock_bulk.call_args_list
                }
                assert by_op == {"delete": 22, self.upsert_op_type: 11}

    @pytest.mark.asyncio
    async def test_bulk_transport_and_refresh_failures_keep_references_intact(
        self, global_config, embed_func, mock_client
    ):
        """Transport errors on bulk (items still buffered) and on refresh (items
        already durable) both raise ``OpenSearchReferencesIntactError``."""
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.side_effect = OpenSearchException("bulk died mid-stream")
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.upsert({"k1": {"content": "v1"}})

                with pytest.raises(OpenSearchReferencesIntactError):
                    await s.index_done_callback()
                assert "k1" in self._pending_upserts(s)

                mock_bulk.side_effect = None
                mock_bulk.return_value = (1, [])
                mock_client.indices.refresh = AsyncMock(
                    side_effect=OpenSearchException("transient refresh failure")
                )
                with pytest.raises(OpenSearchReferencesIntactError):
                    await s.index_done_callback()
                assert len(self._pending_upserts(s)) == 0
