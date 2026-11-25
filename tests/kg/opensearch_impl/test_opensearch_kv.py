"""Offline unit tests for OpenSearchKVStorage, shared OpenSearch infrastructure,
KV durability contracts, PIT iteration, create_time preservation, and LLM cache tools."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

pytest.importorskip(
    "opensearchpy",
    reason="opensearchpy is required for OpenSearch storage tests",
)

from opensearchpy.exceptions import (  # type: ignore  # noqa: E402
    ConnectionTimeout,
    NotFoundError,
    OpenSearchException,
)

from lightrag import config_anchor as ca  # noqa: E402
from lightrag import config_store as cs  # noqa: E402
from lightrag.exceptions import (  # noqa: E402
    ConfigurationStorageError,
    IndexFlushError,
    StorageControlPlaneError,
    StorageNotInitializedError,
)
from lightrag.kg.opensearch_impl import (  # noqa: E402
    _BULK_ERROR_SUMMARY_MAX_LEN,
    _KV_CREATE_TIME_SENTINEL,
    _OPENSEARCH_UNBOUNDED_PAYLOAD_BYTES,
    DEFAULT_OPENSEARCH_DELETE_MAX_RECORDS_PER_BATCH,
    DEFAULT_OPENSEARCH_UPSERT_MAX_PAYLOAD_BYTES,
    DEFAULT_OPENSEARCH_UPSERT_MAX_RECORDS_PER_BATCH,
    ClientManager,
    OpenSearchKVStorage,
    OpenSearchReferencesIntactError,
    _extract_bulk_failed_ids,
    _resolve_bulk_batch_limits,
    _run_chunked_async_bulk,
    _sanitize_index_name,
    _verify_mirrored_id_mapping,
)
from lightrag.kg.shared_storage import (  # noqa: E402
    finalize_share_data,
    initialize_share_data,
)
from lightrag.lightrag import (  # noqa: E402
    _TRACKING_RETAINED_FLUSH_RETRIES,
    LightRAG,
)
from lightrag.tools.clean_llm_query_cache import CleanupStats, CleanupTool  # noqa: E402
from lightrag.tools.migrate_llm_cache import MigrationTool  # noqa: E402
from lightrag.utils import EmbeddingFunc  # noqa: E402
from lightrag.utils_graph import _persist_graph_updates  # noqa: E402

from ._helpers import (  # noqa: E402
    BufferedWriteBatchingContractMixin,
    FakeRateLimitedServer,
    LazyDropRecreateMixin,
    MockEmbeddingFunc,
    ReadinessDemotionMixin,
    default_global_config,
    make_client,
    missing_index_error,
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
def shared_storage_env():
    initialize_share_data(workers=1)
    try:
        yield
    finally:
        finalize_share_data()


# ---------------------------------------------------------------------------
# Shared OpenSearch infrastructure helpers, bulk batching, ClientManager
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for module-level helper functions (_sanitize_index_name)."""

    def test_sanitize_index_name(self):
        assert _sanitize_index_name("Hello_World") == "hello_world"
        assert _sanitize_index_name("-bad") == "x-bad"
        assert _sanitize_index_name("a.b/c") == "a_b_c"


class TestBulkBatchLimits:
    """Tests for bulk batch limits, chunked async_bulk, and bulk error classification."""

    def test_resolve_defaults(self):
        with patch.dict("os.environ", {}, clear=True):
            payload, upserts, deletes = _resolve_bulk_batch_limits()
        assert payload == DEFAULT_OPENSEARCH_UPSERT_MAX_PAYLOAD_BYTES
        assert upserts == DEFAULT_OPENSEARCH_UPSERT_MAX_RECORDS_PER_BATCH
        assert deletes == DEFAULT_OPENSEARCH_DELETE_MAX_RECORDS_PER_BATCH

    def test_resolve_env_override(self):
        env = {
            "OPENSEARCH_UPSERT_MAX_PAYLOAD_BYTES": "12345",
            "OPENSEARCH_UPSERT_MAX_RECORDS_PER_BATCH": "7",
            "OPENSEARCH_DELETE_MAX_RECORDS_PER_BATCH": "9",
        }
        with patch.dict("os.environ", env, clear=True):
            assert _resolve_bulk_batch_limits() == (12345, 7, 9)

    def test_resolve_non_positive_disables_and_warns(self):
        env = {
            "OPENSEARCH_UPSERT_MAX_PAYLOAD_BYTES": "0",
            "OPENSEARCH_UPSERT_MAX_RECORDS_PER_BATCH": "-1",
            "OPENSEARCH_DELETE_MAX_RECORDS_PER_BATCH": "0",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("lightrag.kg.opensearch_impl.logger") as mock_logger:
                payload, upserts, deletes = _resolve_bulk_batch_limits()
        assert (payload, upserts, deletes) == (0, -1, 0)
        warnings = [c.args[0] for c in mock_logger.warning.call_args_list]
        assert sum("non-positive" in msg for msg in warnings) == 3

    @pytest.mark.asyncio
    async def test_run_chunked_forwards_limits(self):
        actions = [{"_op_type": "index", "_id": str(i)} for i in range(3)]
        with patch(
            "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
        ) as mock_bulk:
            mock_bulk.return_value = (3, [])
            await _run_chunked_async_bulk(
                "client",
                actions,
                max_payload_bytes=4096,
                max_records_per_batch=2,
                log_prefix="[ws] ns:",
                what="upsert",
                refresh="wait_for",
            )
        _, kwargs = mock_bulk.call_args
        assert kwargs["chunk_size"] == 2
        assert kwargs["max_chunk_bytes"] == 4096
        assert kwargs["raise_on_error"] is False
        assert kwargs["refresh"] == "wait_for"

    @pytest.mark.asyncio
    async def test_run_chunked_non_positive_uses_sentinels(self):
        actions = [{"_op_type": "index", "_id": str(i)} for i in range(3)]
        with patch(
            "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
        ) as mock_bulk:
            mock_bulk.return_value = (3, [])
            await _run_chunked_async_bulk(
                "client",
                actions,
                max_payload_bytes=0,
                max_records_per_batch=0,
                log_prefix="x",
                what="y",
            )
        _, kwargs = mock_bulk.call_args
        assert kwargs["chunk_size"] == len(actions)
        assert kwargs["max_chunk_bytes"] == _OPENSEARCH_UNBOUNDED_PAYLOAD_BYTES

    def test_extract_bulk_failed_ids_classifies_by_status(self):
        retryable, non_retryable = _extract_bulk_failed_ids(None)
        assert retryable == set()
        assert non_retryable == []
        retryable, non_retryable = _extract_bulk_failed_ids([])
        assert retryable == set()
        assert non_retryable == []

        retryable, non_retryable = _extract_bulk_failed_ids(
            [
                {"index": {"_id": "r-408", "status": 408}},
                {"index": {"_id": "r-429", "status": 429}},
                {"index": {"_id": "r-500", "status": 500}},
                {"index": {"_id": "r-502", "status": 502}},
                {"index": {"_id": "r-503", "status": 503}},
                {"index": {"_id": "r-504", "status": 504}},
                {"create": {"_id": "r-none"}},
                {
                    "index": {
                        "_id": "n-400",
                        "status": 400,
                        "error": {
                            "type": "mapper_parsing_exception",
                            "reason": "vector must be array",
                        },
                    }
                },
                {"update": {"_id": "n-404", "status": 404, "error": "not found"}},
                {"delete": {"_id": "drop-404", "status": 404}},
                "garbage",
                {"update": {}},
            ]
        )
        assert retryable == {
            "r-408",
            "r-429",
            "r-500",
            "r-502",
            "r-503",
            "r-504",
            "r-none",
        }
        assert {op.doc_id for op in non_retryable} == {"n-400", "n-404"}
        by_id = {op.doc_id: op for op in non_retryable}
        assert by_id["n-400"].op == "index"
        assert by_id["n-400"].status == 400
        assert "vector must be array" in by_id["n-400"].error
        assert by_id["n-404"].op == "update"
        assert by_id["n-404"].status == 404
        assert by_id["n-404"].error == "not found"

    def test_extract_bulk_failed_ids_truncates_long_errors(self):
        long_reason = "x" * 1000
        _, non_retryable = _extract_bulk_failed_ids(
            [
                {
                    "index": {
                        "_id": "n-400",
                        "status": 400,
                        "error": {"reason": long_reason},
                    }
                }
            ]
        )
        assert len(non_retryable) == 1
        assert len(non_retryable[0].error) <= _BULK_ERROR_SUMMARY_MAX_LEN
        assert non_retryable[0].error.endswith("...")


class TestClientManager:
    """Tests for ClientManager singleton pattern and reference counting."""

    @staticmethod
    def _stub_client(version: str = "3.3.0") -> AsyncMock:
        client = AsyncMock()
        client.info = AsyncMock(return_value={"version": {"number": version}})
        return client

    @pytest.mark.asyncio
    async def test_singleton_and_refcount(self):
        ClientManager._instances = {"client": None, "ref_count": 0}
        with patch("lightrag.kg.opensearch_impl.AsyncOpenSearch") as mock_cls:
            inner = self._stub_client()
            mock_cls.return_value = inner
            c1 = await ClientManager.get_client()
            c2 = await ClientManager.get_client()
            assert c1 is c2
            assert ClientManager._instances["ref_count"] == 2
            await ClientManager.release_client(c1)
            assert ClientManager._instances["ref_count"] == 1
            inner.close.assert_not_awaited()
            await ClientManager.release_client(c2)
            assert ClientManager._instances["ref_count"] == 0
            assert ClientManager._instances["client"] is None
            inner.close.assert_awaited_once()

    @pytest.mark.parametrize(
        ("use_ssl", "verify_certs", "config_values", "expected", "warn_keys"),
        [
            ("on", "false", {}, (True, False), set()),
            ("false", "on", {}, (False, True), set()),
            (" True ", "t", {}, (True, True), set()),
            ("ture", "true", {}, (True, True), {"OPENSEARCH_USE_SSL"}),
            ("true", "ture", {}, (True, False), {"OPENSEARCH_VERIFY_CERTS"}),
            (None, None, {"verify_certs": "on"}, (True, True), set()),
        ],
    )
    @pytest.mark.asyncio
    async def test_boolean_connection_settings(
        self,
        monkeypatch,
        lightrag_log_records,
        use_ssl,
        verify_certs,
        config_values,
        expected,
        warn_keys,
    ):
        for key, value in (
            ("OPENSEARCH_USE_SSL", use_ssl),
            ("OPENSEARCH_VERIFY_CERTS", verify_certs),
        ):
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)

        ClientManager._instances = {"client": None, "ref_count": 0}
        with (
            patch("lightrag.kg.opensearch_impl.config") as mock_config,
            patch("lightrag.kg.opensearch_impl.AsyncOpenSearch") as mock_cls,
        ):
            mock_config.get.side_effect = lambda _section, key, fallback: (
                config_values.get(key, fallback)
            )
            inner = self._stub_client()
            mock_cls.return_value = inner
            client = await ClientManager.get_client()
            kwargs = mock_cls.call_args.kwargs
            assert (kwargs["use_ssl"], kwargs["verify_certs"]) == expected
            await ClientManager.release_client(client)
        warned = {
            record.args[0]
            for record in lightrag_log_records
            if "is not a recognized boolean" in record.msg
        }
        assert warned == warn_keys


class TestMirroredIdVerification:
    """Tests for the _verify_mirrored_id_mapping fail-fast helper."""

    @pytest.mark.asyncio
    async def test_skipped_on_modern_cluster(self, mock_client):
        await _verify_mirrored_id_mapping(mock_client, "any_index")
        mock_client.indices.get_mapping.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_passes_when_mapping_present(self, mock_client):
        mock_client.indices.get_mapping = AsyncMock(
            return_value={
                "my_index": {
                    "mappings": {"properties": {"__mirrored_id": {"type": "keyword"}}}
                }
            }
        )
        with patch("lightrag.kg.opensearch_impl._shard_doc_supported", False):
            await _verify_mirrored_id_mapping(mock_client, "my_index")
        mock_client.indices.get_mapping.assert_awaited_once_with(index="my_index")

    @pytest.mark.asyncio
    async def test_swallows_get_mapping_error(self, mock_client):
        mock_client.indices.get_mapping = AsyncMock(
            side_effect=OpenSearchException("transport error")
        )
        with patch("lightrag.kg.opensearch_impl._shard_doc_supported", False):
            await _verify_mirrored_id_mapping(mock_client, "my_index")
        mock_client.indices.get_mapping.assert_awaited_once_with(index="my_index")


# ---------------------------------------------------------------------------
# KV Storage core CRUD, drop/readiness, and raw PIT iteration
# ---------------------------------------------------------------------------


def _make_kv(global_config, embed_func, workspace: str = "test") -> OpenSearchKVStorage:
    return OpenSearchKVStorage(
        namespace="text_chunks",
        global_config=global_config,
        embedding_func=embed_func,
        workspace=workspace,
    )


class TestKVStorage(LazyDropRecreateMixin, ReadinessDemotionMixin):
    """Tests for OpenSearchKVStorage CRUD operations, readiness, and PIT scanning."""

    def _make(self, global_config, embed_func, workspace: str = "test"):
        return _make_kv(global_config, embed_func, workspace)

    async def _invoke_missing_read(self, storage):
        return await storage.get_by_id("doc1")

    async def _write_to_recreate(self, storage):
        await storage.upsert({"k1": {"content": "v1"}})

    @pytest.mark.asyncio
    async def test_initialize_fails_on_legacy_index_without_mirrored_id(
        self, global_config, embed_func, mock_client
    ):
        mock_client.indices.exists = AsyncMock(return_value=True)
        mock_client.indices.get_mapping = AsyncMock(
            return_value={
                "test_text_chunks": {
                    "mappings": {"properties": {"content": {"type": "text"}}}
                }
            }
        )
        with (
            patch.object(ClientManager, "get_client", return_value=mock_client),
            patch("lightrag.kg.opensearch_impl._shard_doc_supported", False),
        ):
            s = self._make(global_config, embed_func)
            with pytest.raises(RuntimeError, match="__mirrored_id"):
                await s.initialize()
            mock_client.indices.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_by_id(self, global_config, embed_func, mock_client):
        mock_client.mget = AsyncMock(
            return_value={
                "docs": [
                    {
                        "_id": "doc1",
                        "found": True,
                        "_source": {
                            "content": "hello",
                            "create_time": 0,
                            "update_time": 0,
                        },
                    }
                ]
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            doc = await s.get_by_id("doc1")
            assert doc is not None
            assert doc["content"] == "hello"
            assert doc["_id"] == "doc1"
            mock_client.mget.assert_awaited_once_with(
                index=s._index_name, body={"ids": ["doc1"]}
            )

    @pytest.mark.asyncio
    async def test_reads_short_circuit_after_drop(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.upsert({"buffered": {"content": "x"}})
            await s.delete(["tombstoned"])
            result = await s.drop()

            assert result["status"] == "success"
            mock_client.indices.delete.assert_awaited_once()
            assert s._pending_upserts == {}
            assert s._pending_kv_deletes == set()

            assert await s.get_by_id("doc1") is None
            assert await s.get_by_ids(["doc1", "doc2"]) == [None, None]
            assert await s.is_empty() is True

            mock_client.mget.assert_not_awaited()
            mock_client.count.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_iter_raw_docs_uses_pit_and_search_after(
        self, global_config, embed_func, mock_client
    ):
        mock_client.search = AsyncMock(
            side_effect=[
                {
                    "hits": {
                        "hits": [
                            {"_id": "d1", "_source": {"content": "a"}, "sort": [1]},
                            {"_id": "d2", "_source": {"content": "b"}, "sort": [2]},
                        ]
                    }
                },
                {
                    "hits": {
                        "hits": [
                            {"_id": "d3", "_source": {"content": "c"}, "sort": [3]}
                        ]
                    }
                },
            ]
        )

        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()

            batches = [batch async for batch in s._iter_raw_docs(batch_size=2)]

            assert [[doc["_id"] for doc in batch] for batch in batches] == [
                ["d1", "d2"],
                ["d3"],
            ]
            assert (
                "search_after"
                not in mock_client.search.await_args_list[0].kwargs["body"]
            )
            assert mock_client.search.await_args_list[1].kwargs["body"][
                "search_after"
            ] == [2]
            mock_client.create_pit.assert_awaited_once()
            mock_client.delete_pit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_iter_raw_docs_missing_index_demotes_readiness_and_raises(
        self, global_config, embed_func, mock_client
    ):
        mock_client.search = AsyncMock(side_effect=missing_index_error())

        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()

            batches = []
            with pytest.raises(StorageControlPlaneError, match="unexpectedly missing"):
                async for batch in s._iter_raw_docs(batch_size=2):
                    batches.append(batch)

            assert batches == []
            assert s._index_ready is False
            mock_client.create_pit.assert_awaited_once()
            mock_client.delete_pit.assert_awaited_once()


# ---------------------------------------------------------------------------
# KV storage write batching (shared contract mixin + KV-specific behavior)
# ---------------------------------------------------------------------------


class TestKVStorageBatching(BufferedWriteBatchingContractMixin):
    """Buffered upsert/delete + flush behaviour for OpenSearchKVStorage."""

    upsert_op_type = "update"
    op_label = "KV"
    id_field = "_id"

    def _make(self, global_config, embed_func, workspace: str = "test"):
        return _make_kv(global_config, embed_func, workspace)

    def _pending_upserts(self, storage):
        return storage._pending_upserts

    def _pending_deletes(self, storage):
        return storage._pending_kv_deletes

    def _action_content(self, action):
        return action["script"]["params"]["doc"]["content"]

    def _assert_read_doc_clean(self, doc):
        assert "__mirrored_id" not in doc

    @pytest.mark.asyncio
    async def test_drops_do_not_require_an_initialized_lock(
        self, global_config, embed_func
    ):
        """Both drops run safely before ``initialize()`` assigns ``_flush_lock``."""
        s = self._make(global_config, embed_func)
        assert s._flush_lock is None
        s._pending_upserts["k1"] = {"content": "x"}
        s._pending_kv_deletes.add("gone")

        await s.drop_pending_upserts()
        assert not s._pending_upserts
        assert s._pending_kv_deletes == {"gone"}

        s._pending_upserts["k2"] = {"content": "y"}
        await s.drop_pending_index_ops()
        assert not s._pending_upserts
        assert not s._pending_kv_deletes

    @pytest.mark.asyncio
    async def test_drop_pending_upserts_narrows_to_the_named_cache_types(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.upsert(
                {
                    "default:extract:aaa": {"return": "e1"},
                    "default:extract:bbb": {"return": "e2"},
                    "default:query:ccc": {"return": "an expensive answer"},
                    "default:keywords:ddd": {"return": "kw"},
                    "not-a-cache-key": {"return": "?"},
                }
            )
            await s.delete(["gone-1"])

            dropped = await s.drop_pending_upserts(cache_types={"extract"})

            assert dropped == 2
            assert set(s._pending_upserts) == {
                "default:query:ccc",
                "default:keywords:ddd",
                "not-a-cache-key",
            }
            assert s._pending_kv_deletes == {"gone-1"}

            assert await s.drop_pending_upserts() == 3
            assert not s._pending_upserts
            assert s._pending_kv_deletes == {"gone-1"}

    @pytest.mark.asyncio
    async def test_has_pending_index_ops_reports_retained_upserts_only(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.has_pending_index_ops() is False

            await s.upsert({"default:extract:aaa": {"return": "e1"}})
            assert await s.has_pending_index_ops() is True

            await s.drop_pending_upserts()
            assert await s.has_pending_index_ops() is False

            await s.delete(["gone-1"])
            assert await s.has_pending_index_ops() is False
            assert await s.has_pending_index_ops(include_deletes=True) is True

    @pytest.mark.asyncio
    async def test_kv_delete_works_when_index_not_ready(
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
                await s.upsert({"k1": {"content": "x"}})
                s._mark_index_missing()
                await s.delete(["k1"])
                assert "k1" not in s._pending_upserts
                assert "k1" in s._pending_kv_deletes

    @pytest.mark.asyncio
    async def test_kv_filter_keys_excludes_buffered_upserts(
        self, global_config, embed_func, mock_client
    ):
        mock_client.mget = AsyncMock(
            return_value={"docs": [{"_id": "k2", "found": False}]}
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.upsert({"k1": {"content": "x"}})
            missing = await s.filter_keys({"k1", "k2"})
            assert missing == {"k2"}
            ((_, kwargs),) = mock_client.mget.await_args_list[0:1]
            assert kwargs["body"] == {"ids": ["k2"]}

    @pytest.mark.asyncio
    async def test_kv_filter_keys_treats_buffered_deletes_as_missing(
        self, global_config, embed_func, mock_client
    ):
        mock_client.mget = AsyncMock(
            return_value={"docs": [{"_id": "k3", "found": True}]}
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.delete(["k1"])
            missing = await s.filter_keys({"k1", "k3"})
            assert "k1" in missing
            assert "k3" not in missing
            mget_kwargs = mock_client.mget.await_args_list[0].kwargs
            assert mget_kwargs["body"] == {"ids": ["k3"]}

    @pytest.mark.asyncio
    async def test_kv_is_empty_returns_false_with_pending_upsert(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.upsert({"k1": {"content": "x"}})
            assert await s.is_empty() is False
            mock_client.count.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_missing_index_on_refresh_still_makes_no_claim(
        self, global_config, embed_func, mock_client
    ):
        """Missing-index during refresh returns without raising and marks not ready."""
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk",
                new_callable=AsyncMock,
                return_value=(1, []),
            ):
                s = self._make(global_config, embed_func)
                await s.initialize()
                mock_client.indices.refresh = AsyncMock(
                    side_effect=NotFoundError(404, "index_not_found_exception", {})
                )
                await s.upsert({"k1": {"content": "v1"}})

                await s.index_done_callback()
                assert s._index_ready is False


# ---------------------------------------------------------------------------
# KV refresh gating (cited by docs/design/PurgeRecoveryContract.md)
# ---------------------------------------------------------------------------


class TestKVRefreshGating:
    """The commit-side refresh is owed per-storage, not unconditional."""

    def _make(self, global_config, embed_func, workspace: str = "test"):
        return _make_kv(global_config, embed_func, workspace)

    @pytest.mark.asyncio
    async def test_an_idle_commit_does_not_refresh(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (1, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.upsert({"k1": {"content": "v1"}})
                await s.index_done_callback()
                assert mock_client.indices.refresh.await_count == 1

                await s.index_done_callback()
                await s.index_done_callback()
                assert mock_client.indices.refresh.await_count == 1
                assert mock_bulk.await_count == 1

    @pytest.mark.asyncio
    async def test_a_failed_refresh_is_retried_by_the_next_idle_commit(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (1, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                mock_client.indices.refresh = AsyncMock(
                    side_effect=OpenSearchException("transient refresh failure")
                )
                await s.upsert({"k1": {"content": "v1"}})
                with pytest.raises(OpenSearchReferencesIntactError):
                    await s.index_done_callback()
                assert mock_bulk.await_count == 1
                assert len(s._pending_upserts) == 0

                mock_client.indices.refresh = AsyncMock()
                await s.index_done_callback()
                assert mock_client.indices.refresh.await_count == 1
                assert mock_bulk.await_count == 1

                await s.index_done_callback()
                assert mock_client.indices.refresh.await_count == 1

    @pytest.mark.asyncio
    async def test_a_write_landing_during_a_refresh_is_refreshed_again(
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

                refresh_entered = asyncio.Event()
                release_refresh = asyncio.Event()

                async def blocking_refresh(**kwargs):
                    refresh_entered.set()
                    await release_refresh.wait()

                mock_client.indices.refresh = AsyncMock(side_effect=blocking_refresh)

                await s.upsert({"k1": {"content": "v1"}})
                commit = asyncio.create_task(s.index_done_callback())
                await asyncio.wait_for(refresh_entered.wait(), 5)

                await s.upsert({"k2": {"content": "v2"}})
                await s._flush_pending_kv_ops()

                release_refresh.set()
                await commit
                assert mock_client.indices.refresh.await_count == 1

                await s.index_done_callback()
                assert mock_client.indices.refresh.await_count == 2

    @pytest.mark.asyncio
    async def test_a_flush_that_raised_still_owes_a_refresh(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.side_effect = OpenSearchException("bulk died mid-stream")
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.upsert({"k1": {"content": "v1"}})
                with pytest.raises(OpenSearchException):
                    await s.index_done_callback()
                assert mock_client.indices.refresh.await_count == 0

                await s.drop_pending_index_ops()
                mock_bulk.side_effect = None
                mock_bulk.return_value = (0, [])
                await s.index_done_callback()
                assert mock_client.indices.refresh.await_count == 1
                assert mock_bulk.await_count == 1

    @pytest.mark.asyncio
    async def test_a_missing_index_defers_the_debt_rather_than_settling_it(
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
                mock_client.indices.refresh = AsyncMock(
                    side_effect=OpenSearchException("transient refresh failure")
                )
                await s.upsert({"k1": {"content": "v1"}})
                with pytest.raises(OpenSearchException):
                    await s.index_done_callback()

                await s.drop()
                mock_client.indices.refresh = AsyncMock()

                await s.index_done_callback()
                assert mock_client.indices.refresh.await_count == 0

                await s._ensure_index_ready()
                await s.index_done_callback()
                await s.index_done_callback()
                assert mock_client.indices.refresh.await_count == 1

    @pytest.mark.asyncio
    async def test_a_read_seeing_a_missing_index_does_not_settle_the_debt(
        self, global_config, embed_func, mock_client
    ):
        mget_entered = asyncio.Event()
        release_mget = asyncio.Event()
        bulk_entered = asyncio.Event()
        release_bulk = asyncio.Event()

        async def blocking_missing_mget(**kwargs):
            mget_entered.set()
            await release_mget.wait()
            raise NotFoundError(404, "index_not_found_exception", {})

        mock_client.mget = AsyncMock(side_effect=blocking_missing_mget)

        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:

                async def blocking_bulk(*args, **kwargs):
                    bulk_entered.set()
                    await release_bulk.wait()
                    return (1, [])

                mock_bulk.side_effect = blocking_bulk
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.upsert({"k1": {"content": "v1"}})

                read = asyncio.create_task(s.get_by_id("absent-id"))
                await asyncio.wait_for(mget_entered.wait(), 5)

                commit = asyncio.create_task(s.index_done_callback())
                await asyncio.wait_for(bulk_entered.wait(), 5)

                release_mget.set()
                assert await read is None
                assert s._index_ready is False

                release_bulk.set()
                await commit
                assert mock_client.indices.refresh.await_count == 0

                await s._ensure_index_ready()
                await s.index_done_callback()
                assert mock_client.indices.refresh.await_count == 1

    @pytest.mark.asyncio
    async def test_is_empty_refreshes_before_counting(
        self, global_config, embed_func, mock_client
    ):
        order: list[str] = []

        async def record_refresh(**kwargs):
            order.append("refresh")

        async def record_count(**kwargs):
            order.append("count")
            return {"count": 0}

        mock_client.indices.refresh = AsyncMock(side_effect=record_refresh)
        mock_client.count = AsyncMock(side_effect=record_count)
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.is_empty() is True
            assert order == ["refresh", "count"]

    @pytest.mark.asyncio
    async def test_is_empty_survives_a_failed_refresh(
        self, global_config, embed_func, mock_client
    ):
        mock_client.indices.refresh = AsyncMock(
            side_effect=OpenSearchException("refresh unavailable")
        )
        mock_client.count = AsyncMock(return_value={"count": 3})
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.is_empty() is False
            mock_client.count.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_is_empty_reports_empty_when_the_refresh_finds_no_index(
        self, global_config, embed_func, mock_client
    ):
        mock_client.indices.refresh = AsyncMock(
            side_effect=NotFoundError(404, "index_not_found_exception", {})
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.is_empty() is True
            mock_client.count.assert_not_awaited()
            assert s._index_ready is False


# ---------------------------------------------------------------------------
# KV iter_rows enumeration surface (scenario 16 in ConfigurationStorageContract)
# ---------------------------------------------------------------------------


def _iter_hits(ids_and_sources):
    return [
        {
            "_id": doc_id,
            "_source": {**source, "__mirrored_id": doc_id},
            "sort": [doc_id],
        }
        for doc_id, source in ids_and_sources
    ]


def _iter_storage(pages, *, pending=None, deleted=()):
    storage = OpenSearchKVStorage.__new__(OpenSearchKVStorage)
    storage.namespace = "config"
    storage.workspace = "iterws"
    storage.final_namespace = "iterws_config"
    storage._index_name = "iterws_config"
    storage._index_ready = True
    storage._pending_upserts = dict(pending or {})
    storage._pending_kv_deletes = set(deleted)
    storage._flush_lock = asyncio.Lock()
    storage._write_generation = 0
    storage._refreshed_generation = 0
    client = MagicMock()
    client.indices = MagicMock()
    client.indices.refresh = AsyncMock()
    client.create_pit = AsyncMock(return_value={"pit_id": "pit-1"})
    client.delete_pit = AsyncMock()
    responses = [{"hits": {"hits": page}} for page in pages] + [{"hits": {"hits": []}}]
    client.search = AsyncMock(side_effect=responses)
    storage.client = client
    return storage, client


class TestKVIterRows:
    """``OpenSearchKVStorage.iter_rows``: PIT + search_after + read-your-writes."""

    async def test_rows_stream_across_pit_pages_in_the_point_read_shape(self):
        storage, client = _iter_storage(
            [
                _iter_hits([("a", {"value": {"n": 1}}), ("b", {"value": {"n": 2}})]),
                _iter_hits([("c", {"value": {"n": 3}})]),
            ]
        )

        rows = [row async for row in storage.iter_rows(page_size=2)]

        assert [r["_id"] for r in rows] == ["a", "b", "c"]
        assert all("__mirrored_id" not in r for r in rows)
        assert all(r["create_time"] == 0 and r["update_time"] == 0 for r in rows)
        client.create_pit.assert_awaited_once()
        client.delete_pit.assert_awaited_once()
        assert all(
            call.kwargs["body"]["size"] == 2 for call in client.search.call_args_list
        )

    async def test_the_pending_buffer_is_read_your_writes(self):
        storage, _ = _iter_storage(
            [
                _iter_hits(
                    [
                        ("indexed", {"value": 1}),
                        ("replaced", {"value": "old"}),
                        ("gone", {"value": 1}),
                    ]
                )
            ],
            pending={"replaced": {"value": "new"}, "fresh": {"value": "buffered"}},
            deleted={"gone"},
        )

        rows = {r["_id"]: r for r in [row async for row in storage.iter_rows()]}

        assert set(rows) == {"indexed", "replaced", "fresh"}
        assert rows["replaced"]["value"] == "new"
        assert rows["fresh"]["value"] == "buffered"

    async def test_an_uninitialized_storage_raises(self):
        storage, _ = _iter_storage([])
        storage._flush_lock = None
        with pytest.raises(StorageNotInitializedError):
            async for _ in storage.iter_rows():
                pass

    async def test_a_dropped_index_raises_instead_of_ending_the_scan(self):
        storage, client = _iter_storage([])
        storage._index_ready = False

        rows = []
        with pytest.raises(StorageControlPlaneError, match="not ready"):
            async for row in storage.iter_rows():
                rows.append(row)

        assert rows == []
        client.create_pit.assert_not_awaited()

    async def test_an_index_that_vanishes_mid_scan_raises_and_marks_it_missing(self):
        storage, client = _iter_storage([])
        client.create_pit = AsyncMock(
            side_effect=OpenSearchException("index_not_found_exception: iterws_config")
        )

        with pytest.raises(StorageControlPlaneError, match="unexpectedly missing"):
            async for _ in storage.iter_rows():
                pass

        assert storage._index_ready is False

    async def test_the_pending_buffer_alone_is_not_yielded_as_the_listing(self):
        storage, _ = _iter_storage([], pending={"buffered": {"value": 1}})
        storage._index_ready = False

        rows = []
        with pytest.raises(StorageControlPlaneError):
            async for row in storage.iter_rows():
                rows.append(row)

        assert rows == []

    async def test_a_failed_refresh_raises_instead_of_scanning_a_stale_view(self):
        storage, client = _iter_storage([])
        client.indices.refresh = AsyncMock(
            side_effect=OpenSearchException("refresh rejected: too many requests")
        )

        with pytest.raises(StorageControlPlaneError, match="not yet searchable"):
            async for _ in storage.iter_rows(page_size=10):
                pass

        client.create_pit.assert_not_awaited()

    async def test_a_refresh_failure_outside_a_scan_is_still_best_effort(self):
        storage, client = _iter_storage([])
        client.indices.refresh = AsyncMock(
            side_effect=OpenSearchException("refresh rejected: too many requests")
        )

        await storage._refresh_for_search()
        client.indices.refresh.assert_awaited_once()
        assert storage._index_ready is True


# ---------------------------------------------------------------------------
# KV upsert server-side create_time preservation (scripted_upsert)
# ---------------------------------------------------------------------------


def _upsert_actions(mock_bulk: AsyncMock) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for call in mock_bulk.await_args_list:
        for action in call.args[1]:
            if action.get("_op_type") != "delete":
                actions.append(action)
    return actions


class TestKVUpsertCreateTime:
    """OpenSearchKVStorage preserves create_time server-side via scripted_upsert."""

    @pytest.fixture
    def ct_storage(self, global_config, embed_func):
        client = AsyncMock()
        client.indices = AsyncMock()
        client.indices.exists = AsyncMock(return_value=True)
        client.indices.get_mapping = AsyncMock(
            return_value={
                "ct_text_chunks": {
                    "mappings": {
                        "_meta": {
                            "lightrag_workspace": "ct",
                            "lightrag_final_namespace": "ct_text_chunks",
                        },
                        "properties": {"__mirrored_id": {"type": "keyword"}},
                    }
                }
            }
        )
        client.indices.refresh = AsyncMock(return_value={})
        with patch.object(ClientManager, "get_client", return_value=client):
            s = OpenSearchKVStorage(
                namespace="text_chunks",
                global_config=global_config,
                embedding_func=embed_func,
                workspace="ct",
            )
            s.client = client
            s._index_ready = True
            s._flush_lock = asyncio.Lock()
            yield s, client

    @pytest.mark.asyncio
    async def test_upsert_reads_nothing(self, ct_storage):
        storage, client = ct_storage
        with patch("time.time", return_value=1_700_000_000):
            await storage.upsert({"k1": {"content": "a"}})
            await storage.upsert({"k1": {"content": "b"}, "k2": {"content": "c"}})

        client.mget.assert_not_awaited()
        client.get.assert_not_awaited()
        client.search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_flush_emits_scripted_upsert_action(self, ct_storage):
        storage, _ = ct_storage
        with patch("time.time", return_value=1_700_000_000):
            await storage.upsert({"k1": {"content": "a", "tokens": 3}})

        with patch(
            "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
        ) as mock_bulk:
            mock_bulk.return_value = (1, [])
            await storage._flush_pending_kv_ops()

        actions = _upsert_actions(mock_bulk)
        assert len(actions) == 1
        action = actions[0]
        assert action["_op_type"] == "update"
        assert action["_id"] == "k1"
        assert action["scripted_upsert"] is True
        assert action["upsert"] == {_KV_CREATE_TIME_SENTINEL: True}
        assert action["retry_on_conflict"] >= 1
        assert "_source" not in action
        doc = action["script"]["params"]["doc"]
        assert doc["content"] == "a"
        assert doc["tokens"] == 3
        assert doc["__mirrored_id"] == "k1"
        assert doc["update_time"] == 1_700_000_000
        source = action["script"]["source"]
        assert "create_time" in source
        assert _KV_CREATE_TIME_SENTINEL in source

    @pytest.mark.asyncio
    async def test_buffered_create_time_is_the_write_time(self, ct_storage):
        storage, _ = ct_storage
        with patch("time.time", return_value=1_700_000_000):
            await storage.upsert({"k1": {"content": "a"}})

        buffered = await storage.get_by_id("k1")
        assert buffered["create_time"] == 1_700_000_000
        assert buffered["update_time"] == 1_700_000_000

    @pytest.mark.asyncio
    async def test_caller_supplied_create_time_is_overwritten(self, ct_storage):
        storage, _ = ct_storage
        with patch("time.time", return_value=1_700_000_000):
            await storage.upsert({"k1": {"content": "a", "create_time": 1}})

        with patch(
            "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
        ) as mock_bulk:
            mock_bulk.return_value = (1, [])
            await storage._flush_pending_kv_ops()

        doc = _upsert_actions(mock_bulk)[0]["script"]["params"]["doc"]
        assert doc["create_time"] == 1_700_000_000

    @pytest.mark.asyncio
    async def test_upsert_after_delete_flushes_only_the_upsert(self, ct_storage):
        storage, _ = ct_storage
        await storage.delete(["k1"])
        with patch("time.time", return_value=1_700_000_400):
            await storage.upsert({"k1": {"content": "fresh"}})

        with patch(
            "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
        ) as mock_bulk:
            mock_bulk.return_value = (1, [])
            await storage._flush_pending_kv_ops()

        deletes = [
            action
            for call in mock_bulk.await_args_list
            for action in call.args[1]
            if action.get("_op_type") == "delete"
        ]
        assert deletes == []
        actions = _upsert_actions(mock_bulk)
        assert len(actions) == 1
        assert actions[0]["script"]["params"]["doc"]["create_time"] == 1_700_000_400


# ---------------------------------------------------------------------------
# Configuration-store durability on OpenSearchKVStorage
# ---------------------------------------------------------------------------


_DURABILITY_WORKSPACE = "tenant"


async def _dummy_embed(texts, **kwargs):  # pragma: no cover - never called
    return np.zeros((len(texts), 8), dtype=np.float32)


def _config_embedding() -> EmbeddingFunc:
    return EmbeddingFunc(
        embedding_dim=8, max_token_size=1024, func=_dummy_embed, model_name="bge-m3"
    )


async def _make_config_storage(server: FakeRateLimitedServer) -> OpenSearchKVStorage:
    with patch.object(ClientManager, "get_client", return_value=server.client):
        storage = cs.create_configuration_storage(
            OpenSearchKVStorage,
            global_config={
                "embedding_batch_num": 10,
                "max_graph_nodes": 1000,
                "vector_db_storage_cls_kwargs": {},
            },
            embedding_func=None,
        )
        await storage.initialize()
    return storage


def _seed_config_records(server: FakeRateLimitedServer) -> None:
    for target in cs.EMBEDDING_TARGETS:
        server.docs[cs.embedding_baseline_key(_DURABILITY_WORKSPACE, target)] = (
            cs.make_config_row(
                scope_workspace=_DURABILITY_WORKSPACE,
                suffix=cs.embedding_baseline_suffix(target),
                value={"model": "old-model", "dim": 8, "origin": "probe"},
                updated_by="test",
            )
        )


async def _bind_config_identity(storage, tmp_path):
    return await cs.bind_configuration_identity(
        storage,
        working_dir=str(tmp_path),
        backend="OpenSearchKVStorage",
        container="OpenSearchKVStorage (_lightrag_config)",
    )


@pytest.mark.usefixtures("shared_storage_env")
class TestKVConfigDurability:
    """Configuration-store durability against a rate-limited (429) OpenSearch server."""

    async def test_a_rate_limited_claim_is_a_failure_and_writes_nothing(self):
        server = FakeRateLimitedServer(rate_limited=True)
        storage = await _make_config_storage(server)
        with patch(
            "lightrag.kg.opensearch_impl.helpers.async_bulk", side_effect=server.bulk
        ):
            with pytest.raises(ConfigurationStorageError, match="retained"):
                await cs.claim_embedding_baseline(
                    storage,
                    workspace=_DURABILITY_WORKSPACE,
                    target="entities",
                    candidate=cs.EmbeddingBaseline("bge-m3", 8, "empty"),
                    embedding_func=_config_embedding(),
                )
        assert server.bulk_calls == 1
        assert server.docs == {}
        assert await storage.has_pending_index_ops(include_deletes=True) is False

    async def test_a_rate_limited_rebuild_record_is_a_failure_and_keeps_the_old_model(
        self,
    ):
        server = FakeRateLimitedServer(rate_limited=True)
        _seed_config_records(server)
        storage = await _make_config_storage(server)
        with patch(
            "lightrag.kg.opensearch_impl.helpers.async_bulk", side_effect=server.bulk
        ):
            with pytest.raises(ConfigurationStorageError, match="retained"):
                await cs.record_embedding_baseline(
                    storage,
                    workspace=_DURABILITY_WORKSPACE,
                    target="chunks",
                    embedding_func=_config_embedding(),
                )
        key = cs.embedding_baseline_key(_DURABILITY_WORKSPACE, "chunks")
        assert server.docs[key]["value"]["model"] == "old-model"

    async def test_a_rate_limited_drop_is_a_failure_and_the_rows_survive(self):
        server = FakeRateLimitedServer(rate_limited=True)
        _seed_config_records(server)
        storage = await _make_config_storage(server)
        with patch(
            "lightrag.kg.opensearch_impl.helpers.async_bulk", side_effect=server.bulk
        ):
            with pytest.raises(ConfigurationStorageError, match="retained"):
                await cs.delete_workspace_configuration(storage, _DURABILITY_WORKSPACE)
        assert len(server.docs) == 3
        assert await storage.has_pending_index_ops(include_deletes=True) is False

    async def test_a_healthy_server_confirms_the_claim_from_the_server_not_the_buffer(
        self,
    ):
        server = FakeRateLimitedServer(rate_limited=False)
        storage = await _make_config_storage(server)
        with patch(
            "lightrag.kg.opensearch_impl.helpers.async_bulk", side_effect=server.bulk
        ):
            recorded = await cs.claim_embedding_baseline(
                storage,
                workspace=_DURABILITY_WORKSPACE,
                target="entities",
                candidate=cs.EmbeddingBaseline("bge-m3", 8, "empty"),
                embedding_func=_config_embedding(),
            )
        assert recorded.model == "bge-m3"
        assert (
            cs.embedding_baseline_key(_DURABILITY_WORKSPACE, "entities") in server.docs
        )
        assert await storage.has_pending_index_ops(include_deletes=True) is False

    async def test_a_refresh_failure_over_a_landed_bulk_still_claims(self):
        server = FakeRateLimitedServer(rate_limited=False)
        server.client.indices.refresh = AsyncMock(
            side_effect=OpenSearchException("refresh unavailable")
        )
        storage = await _make_config_storage(server)

        with patch("lightrag.kg.opensearch_impl.helpers.async_bulk", new=server.bulk):
            baseline = await cs.claim_embedding_baseline(
                storage,
                workspace=_DURABILITY_WORKSPACE,
                target="entities",
                candidate=cs.EmbeddingBaseline("bge-m3", 8, "empty"),
                embedding_func=_config_embedding(),
            )

        assert baseline.model == "bge-m3"
        key = cs.embedding_baseline_key(_DURABILITY_WORKSPACE, "entities")
        assert server.docs[key]["value"]["model"] == "bge-m3"
        assert await storage.has_pending_index_ops(include_deletes=True) is False

    async def test_a_rate_limited_identity_write_binds_nothing(self, tmp_path):
        server = FakeRateLimitedServer(rate_limited=True)
        storage = await _make_config_storage(server)
        with patch(
            "lightrag.kg.opensearch_impl.helpers.async_bulk", side_effect=server.bulk
        ):
            with pytest.raises(ConfigurationStorageError, match="retained"):
                await _bind_config_identity(storage, tmp_path)
        assert server.docs == {}
        assert await storage.has_pending_index_ops(include_deletes=True) is False
        assert ca.read_anchor(str(tmp_path)) is None

    async def test_a_healthy_server_records_the_identity_and_verifies_it(
        self, tmp_path
    ):
        server = FakeRateLimitedServer(rate_limited=False)
        storage = await _make_config_storage(server)
        with patch(
            "lightrag.kg.opensearch_impl.helpers.async_bulk", side_effect=server.bulk
        ):
            created = await _bind_config_identity(storage, tmp_path)
            assert created.action == "created"
            assert server.docs[cs.storage_identity_key()]["value"] == {
                "uuid": created.storage_uuid
            }
            bulk_calls = server.bulk_calls
            verified = await _bind_config_identity(storage, tmp_path)
        assert verified.storage_uuid == created.storage_uuid
        assert verified.action == "verified"
        assert server.bulk_calls == bulk_calls
        assert ca.read_anchor(str(tmp_path)).backend == "OpenSearchKVStorage"


# ---------------------------------------------------------------------------
# Chunk-tracking durability on OpenSearchKVStorage (buffers_writes_per_process)
# ---------------------------------------------------------------------------


class _RecordingGraph:
    """Minimal graph stand-in counting phase-2 commits."""

    def __init__(self):
        self.commits = 0

    async def index_done_callback(self):
        self.commits += 1


async def _make_tracking_storage(server: FakeRateLimitedServer) -> OpenSearchKVStorage:
    with patch.object(ClientManager, "get_client", return_value=server.client):
        storage = OpenSearchKVStorage(
            namespace="entity_chunks",
            workspace="tenant",
            global_config={
                "embedding_batch_num": 10,
                "max_graph_nodes": 1000,
                "vector_db_storage_cls_kwargs": {},
            },
            embedding_func=None,
        )
        await storage.initialize()
    return storage


def _tracking_commit_owner(entity_chunks):
    owner = SimpleNamespace(
        entity_chunks=entity_chunks,
        relation_chunks=None,
        text_chunks=None,
        llm_response_cache=None,
    )
    owner._flush_storages = lambda storages, **kw: LightRAG._flush_storages(
        owner, storages, **kw
    )
    return owner, LightRAG._require_tracking_commit_landed


@pytest.mark.usefixtures("shared_storage_env")
class TestKVTrackingDurability:
    """A chunk-tracking flush that retains a write fails before the graph commits."""

    async def test_a_retained_tracking_write_fails_the_commit_and_is_discarded(self):
        server = FakeRateLimitedServer()
        storage = await _make_tracking_storage(server)
        graph = _RecordingGraph()
        with patch(
            "lightrag.kg.opensearch_impl.helpers.async_bulk", side_effect=server.bulk
        ):
            server.rate_limited = True
            await storage.upsert({"Foo": {"chunk_ids": ["old", "custom"], "count": 2}})
            with pytest.raises(RuntimeError, match="not made durable"):
                await _persist_graph_updates(
                    entity_chunks_storage=storage,
                    chunk_entity_relation_graph=graph,
                    context="test",
                )

            assert server.bulk_calls == 1
            assert graph.commits == 0
            assert await storage.has_pending_index_ops(include_deletes=True) is False

            server.rate_limited = False
            server.docs["Foo"] = {"chunk_ids": ["old", "document"], "count": 2}
            await storage.index_done_callback()

        assert server.docs["Foo"]["chunk_ids"] == ["old", "document"]

    async def test_a_tracking_write_that_lands_commits_the_graph(self):
        server = FakeRateLimitedServer()
        storage = await _make_tracking_storage(server)
        graph = _RecordingGraph()
        with patch(
            "lightrag.kg.opensearch_impl.helpers.async_bulk", side_effect=server.bulk
        ):
            await storage.upsert({"Foo": {"chunk_ids": ["old", "custom"], "count": 2}})
            await _persist_graph_updates(
                entity_chunks_storage=storage,
                chunk_entity_relation_graph=graph,
                context="test",
            )

        assert server.docs["Foo"]["chunk_ids"] == ["old", "custom"]
        assert graph.commits == 1

    async def test_a_raising_tracking_flush_discards_its_buffer_too(self):
        server = FakeRateLimitedServer()
        storage = await _make_tracking_storage(server)
        graph = _RecordingGraph()
        await storage.upsert({"Foo": {"chunk_ids": ["old", "custom"], "count": 2}})

        with patch(
            "lightrag.kg.opensearch_impl.helpers.async_bulk",
            side_effect=ConnectionTimeout("TIMEOUT", "timed out", Exception("slow")),
        ):
            with pytest.raises(
                OpenSearchReferencesIntactError, match="ConnectionTimeout"
            ):
                await _persist_graph_updates(
                    entity_chunks_storage=storage,
                    chunk_entity_relation_graph=graph,
                    context="test",
                )
        assert graph.commits == 0
        assert await storage.has_pending_index_ops(include_deletes=True) is False

        with patch(
            "lightrag.kg.opensearch_impl.helpers.async_bulk", side_effect=server.bulk
        ):
            server.rate_limited = True
            await storage.index_done_callback()
            server.rate_limited = False
            server.docs["Foo"] = {"chunk_ids": ["old", "document"], "count": 2}
            await storage.index_done_callback()

        assert server.docs["Foo"]["chunk_ids"] == ["old", "document"]

    async def test_the_synchronous_discard_empties_the_buffer(self):
        server = FakeRateLimitedServer()
        storage = await _make_tracking_storage(server)
        await storage.upsert({"Foo": {"chunk_ids": ["old", "edit"], "count": 2}})
        await storage.delete(["Bar"])

        assert storage.discard_buffered_writes() == 2
        assert await storage.has_pending_index_ops(include_deletes=True) is False

        with patch(
            "lightrag.kg.opensearch_impl.helpers.async_bulk", side_effect=server.bulk
        ):
            server.docs["Foo"] = {"chunk_ids": ["old", "document"], "count": 2}
            await storage.index_done_callback()

        assert server.bulk_calls == 0
        assert server.docs["Foo"]["chunk_ids"] == ["old", "document"]

    async def test_has_retained_writes_counts_only_what_a_flush_attempted(self):
        server = FakeRateLimitedServer()
        storage = await _make_tracking_storage(server)
        with patch(
            "lightrag.kg.opensearch_impl.helpers.async_bulk", side_effect=server.bulk
        ):
            await storage.upsert({"Foo": {"chunk_ids": ["old"], "count": 1}})
            await storage.index_done_callback()
            await storage.upsert({"Bar": {"chunk_ids": ["fresh"], "count": 1}})
            assert await storage.has_pending_index_ops() is True
            assert await storage.has_retained_writes() is False

            server.rate_limited = True
            await storage.index_done_callback()
            assert await storage.has_retained_writes() is True

            server.rate_limited = False
            await storage.index_done_callback()
            assert await storage.has_retained_writes() is False
        assert server.docs["Bar"]["chunk_ids"] == ["fresh"]

    @pytest.mark.parametrize("op", ["upsert", "delete"])
    async def test_has_retained_writes_counts_a_retained_tombstone(self, op):
        server = FakeRateLimitedServer()
        storage = await _make_tracking_storage(server)
        with patch(
            "lightrag.kg.opensearch_impl.helpers.async_bulk", side_effect=server.bulk
        ):
            if op == "upsert":
                await storage.upsert({"Foo": {"chunk_ids": ["old"], "count": 1}})
            else:
                await storage.delete(["Foo"])
            server.rate_limited = True
            await storage.index_done_callback()
        assert await storage.has_retained_writes() is True
        await storage.drop_pending_index_ops()
        assert await storage.has_retained_writes() is False

    async def test_a_retained_pipeline_tracking_write_is_reflushed_until_it_lands(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            "lightrag.lightrag._TRACKING_RETAINED_FLUSH_BACKOFF_SECONDS", 0
        )
        server = FakeRateLimitedServer()
        storage = await _make_tracking_storage(server)
        owner, require_landed = _tracking_commit_owner(storage)
        original_bulk = server.bulk

        async def recovering_bulk(client, actions, **kw):
            result = await original_bulk(client, actions, **kw)
            server.rate_limited = False
            return result

        with patch(
            "lightrag.kg.opensearch_impl.helpers.async_bulk",
            side_effect=recovering_bulk,
        ):
            await storage.upsert(
                {"Foo": {"chunk_ids": ["old", "document"], "count": 2}}
            )
            server.rate_limited = True
            await storage.index_done_callback()
            assert await storage.has_retained_writes() is True
            await require_landed(owner)

        assert server.bulk_calls == 2
        assert server.docs["Foo"]["chunk_ids"] == ["old", "document"]
        assert await storage.has_pending_index_ops(include_deletes=True) is False

    async def test_a_tracking_write_that_never_lands_fails_without_discarding(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            "lightrag.lightrag._TRACKING_RETAINED_FLUSH_BACKOFF_SECONDS", 0
        )
        server = FakeRateLimitedServer()
        storage = await _make_tracking_storage(server)
        owner, require_landed = _tracking_commit_owner(storage)
        with patch(
            "lightrag.kg.opensearch_impl.helpers.async_bulk", side_effect=server.bulk
        ):
            await storage.upsert(
                {"Foo": {"chunk_ids": ["old", "document"], "count": 2}}
            )
            server.rate_limited = True
            await storage.index_done_callback()
            with pytest.raises(IndexFlushError, match="not made durable") as exc_info:
                await require_landed(owner)

        assert exc_info.value.storage_name == "OpenSearchKVStorage"
        assert server.bulk_calls == 1 + _TRACKING_RETAINED_FLUSH_RETRIES
        assert await storage.has_retained_writes() is True
        assert "Foo" not in server.docs


# ---------------------------------------------------------------------------
# OpenSearch support in LLM cache cleanup & migration CLI tools
# ---------------------------------------------------------------------------


class _FakeCacheToolStorage:
    """Minimal storage double exposing ``_iter_raw_docs`` and ``delete``."""

    def __init__(self, batches, workspace: str = "test-workspace"):
        self._batches = batches
        self.workspace = workspace
        self.deleted_batches: list[list[str]] = []

    async def _iter_raw_docs(self, batch_size: int = 1000):
        for batch in self._batches:
            yield batch

    async def delete(self, ids):
        self.deleted_batches.append(list(ids))


class TestCleanupToolOpenSearch:
    """OpenSearch support in lightrag.tools.clean_llm_query_cache."""

    @pytest.mark.asyncio
    async def test_count_query_caches_opensearch(self):
        tool = CleanupTool()
        storage = _FakeCacheToolStorage(
            [
                [
                    {"_id": "mix:query:1", "_source": {}},
                    {"_id": "mix:keywords:1", "_source": {}},
                    {"_id": "default:extract:1", "_source": {}},
                ],
                [
                    {"_id": "hybrid:query:1", "_source": {}},
                    {"_id": "local:keywords:1", "_source": {}},
                    {"_id": "other:key:1", "_source": {}},
                ],
            ]
        )

        counts = await tool.count_query_caches(storage, "OpenSearchKVStorage")

        assert counts["mix"] == {"query": 1, "keywords": 1}
        assert counts["hybrid"] == {"query": 1, "keywords": 0}
        assert counts["local"] == {"query": 0, "keywords": 1}
        assert counts["global"] == {"query": 0, "keywords": 0}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("cleanup_type", "expected_ids"),
        [
            (
                "all",
                [
                    "mix:query:1",
                    "mix:keywords:1",
                    "global:query:1",
                    "local:keywords:1",
                ],
            ),
            ("query", ["mix:query:1", "global:query:1"]),
            ("keywords", ["mix:keywords:1", "local:keywords:1"]),
        ],
    )
    async def test_delete_query_caches_opensearch(self, cleanup_type, expected_ids):
        tool = CleanupTool()
        tool.batch_size = 2
        storage = _FakeCacheToolStorage(
            [
                [
                    {"_id": "mix:query:1", "_source": {}},
                    {"_id": "mix:keywords:1", "_source": {}},
                ],
                [
                    {"_id": "global:query:1", "_source": {}},
                    {"_id": "local:keywords:1", "_source": {}},
                    {"_id": "default:extract:1", "_source": {}},
                ],
            ]
        )
        stats = CleanupStats()

        await tool.delete_query_caches(
            storage, "OpenSearchKVStorage", cleanup_type, stats
        )

        flattened = [item for batch in storage.deleted_batches for item in batch]
        assert flattened == expected_ids
        assert all(len(batch) <= 2 for batch in storage.deleted_batches)
        assert stats.successfully_deleted == len(expected_ids)
        assert stats.successful_batches == len(storage.deleted_batches)

    def test_check_config_ini_for_storage_opensearch(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.ini").write_text("[opensearch]\nhosts = localhost:9200\n")

        assert CleanupTool().check_config_ini_for_storage("OpenSearchKVStorage")

    def test_get_storage_class_opensearch(self):
        cleanup_cls = CleanupTool().get_storage_class("OpenSearchKVStorage")
        migrate_cls = MigrationTool().get_storage_class("OpenSearchKVStorage")

        assert cleanup_cls.__name__ == "OpenSearchKVStorage"
        assert migrate_cls.__name__ == "OpenSearchKVStorage"


class TestMigrationToolOpenSearch:
    """OpenSearch support in lightrag.tools.migrate_llm_cache."""

    @pytest.mark.asyncio
    async def test_count_and_stream_default_caches_opensearch(self):
        tool = MigrationTool()
        storage = _FakeCacheToolStorage(
            [
                [
                    {"_id": "default:extract:1", "_source": {"return": "a"}},
                    {"_id": "mix:query:1", "_source": {"return": "ignored"}},
                ],
                [
                    {"_id": "default:summary:1", "_source": {"return": "b"}},
                    {"_id": "default:analysis:1", "_source": {"return": "vlm"}},
                    {"_id": "default:extract:2", "_source": {"return": "c"}},
                ],
            ]
        )

        count = await tool.count_default_caches(storage, "OpenSearchKVStorage")
        streamed = [
            batch
            async for batch in tool.stream_default_caches(
                storage, "OpenSearchKVStorage", batch_size=2
            )
        ]

        assert count == 4
        assert streamed == [
            {
                "default:extract:1": {"return": "a"},
                "default:summary:1": {"return": "b"},
            },
            {
                "default:analysis:1": {"return": "vlm"},
                "default:extract:2": {"return": "c"},
            },
        ]

    def test_count_available_storage_types_includes_opensearch(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.ini").write_text("[opensearch]\nhosts = localhost:9200\n")

        with patch.dict("os.environ", {}, clear=True):
            assert MigrationTool().count_available_storage_types() == 2

    @pytest.mark.asyncio
    async def test_setup_storage_returns_effective_workspace(self, monkeypatch):
        tool = MigrationTool()
        fake_storage = SimpleNamespace(workspace="forced-workspace")

        monkeypatch.setattr(tool, "check_env_vars", lambda _: True)
        monkeypatch.setattr(
            tool, "initialize_storage", AsyncMock(return_value=fake_storage)
        )
        monkeypatch.setattr(tool, "count_default_caches", AsyncMock(return_value=3))

        with patch("builtins.input", return_value="5"):
            storage, storage_name, workspace, total_count = await tool.setup_storage(
                "Source", use_streaming=True
            )

        assert storage is fake_storage
        assert storage_name == "OpenSearchKVStorage"
        assert workspace == "forced-workspace"
        assert total_count == 3
