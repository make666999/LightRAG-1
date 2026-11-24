"""Offline tests for Qdrant collection migration, vector-space refusal, and sibling-model cleanup.

All tests use mocks — no running Qdrant instance required.
"""

import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

pytest.importorskip(
    "qdrant_client",
    reason="qdrant-client is required for Qdrant storage tests",
)

from qdrant_client import models  # noqa: E402

from lightrag.exceptions import (  # noqa: E402
    DataMigrationError,
    VectorSpaceMismatchError,
)
from lightrag.kg.qdrant_impl import QdrantVectorDBStorage  # noqa: E402
from lightrag.kg.shared_storage import initialize_share_data  # noqa: E402
from lightrag.utils import EmbeddingFunc  # noqa: E402

pytestmark = pytest.mark.offline

_GLOBAL_CONFIG = {
    "embedding_batch_num": 10,
    "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
}


def _embedding_func(model_name: str = "test-model", dim: int = 768) -> EmbeddingFunc:
    async def embed(texts, **kwargs):
        return np.array([[0.1] * dim for _ in texts])

    return EmbeddingFunc(embedding_dim=dim, func=embed, model_name=model_name)


@pytest.fixture(autouse=True)
def _shared_storage():
    """The storages take their data-init and flush locks from shared storage."""
    initialize_share_data(workers=1)


# ---------------------------------------------------------------------------
# Legacy collection migration & multi-model coexistence
# ---------------------------------------------------------------------------


class TestQdrantMigration:
    @pytest.fixture
    def mock_qdrant_client(self):
        with patch("lightrag.kg.qdrant_impl.QdrantClient") as mock_client_cls:
            client = mock_client_cls.return_value
            client.collection_exists.return_value = False
            client.count.return_value.count = 0
            collection_info = MagicMock()
            collection_info.payload_schema = {}
            collection_info.config.params.vectors.size = 768
            client.get_collection.return_value = collection_info
            yield client

    @pytest.fixture(autouse=True)
    def mock_data_init_lock(self):
        with patch("lightrag.kg.qdrant_impl.get_data_init_lock") as mock_lock:
            mock_lock.return_value = AsyncMock()
            yield mock_lock

    async def test_qdrant_migration_trigger(self, mock_qdrant_client):
        """Legacy collection without model suffix triggers sampling and batched migration."""
        storage = QdrantVectorDBStorage(
            namespace="chunks",
            global_config=_GLOBAL_CONFIG,
            embedding_func=_embedding_func(),
            workspace="test_ws",
        )
        legacy_collection = "lightrag_vdb_chunks"
        mock_qdrant_client.collection_exists.side_effect = (
            lambda name: name == legacy_collection
        )

        migration_state = {"new_workspace_count": 0}

        def count_mock(collection_name, exact=True, count_filter=None):
            mock_result = MagicMock()
            if collection_name == legacy_collection:
                mock_result.count = 100
            elif collection_name == storage.final_namespace:
                mock_result.count = migration_state["new_workspace_count"]
            else:
                mock_result.count = 0
            return mock_result

        mock_qdrant_client.count.side_effect = count_mock

        mock_point = MagicMock()
        mock_point.id = "old_id"
        mock_point.vector = [0.1] * 768
        mock_point.payload = {"content": "test"}
        mock_qdrant_client.scroll.side_effect = [
            ([mock_point], "_"),  # Sampling scroll - no workspace_id found
            ([mock_point], "next_offset"),  # Migration batch
            ([], None),  # End of migration
        ]

        def upsert_mock(*args, **kwargs):
            migration_state["new_workspace_count"] = 100
            return None

        mock_qdrant_client.upsert.side_effect = upsert_mock

        await storage.initialize()

        mock_qdrant_client.count.assert_any_call(
            collection_name=legacy_collection, exact=True
        )
        mock_qdrant_client.create_collection.assert_called()
        assert mock_qdrant_client.scroll.call_count >= 2
        sampling_call = mock_qdrant_client.scroll.call_args_list[0]
        assert sampling_call.kwargs["collection_name"] == legacy_collection
        assert sampling_call.kwargs["limit"] == 10
        migration_call = mock_qdrant_client.scroll.call_args_list[1]
        assert migration_call.kwargs["collection_name"] == legacy_collection
        assert migration_call.kwargs["limit"] == 500
        mock_qdrant_client.upsert.assert_called()
        mock_qdrant_client.create_payload_index.assert_called()

    async def test_qdrant_no_migration_needed(self, mock_qdrant_client):
        """When only the new collection exists, ensure payload index without scrolling."""
        storage = QdrantVectorDBStorage(
            namespace="chunks",
            global_config=_GLOBAL_CONFIG,
            embedding_func=_embedding_func(),
            workspace="test_ws",
        )
        mock_qdrant_client.collection_exists.side_effect = (
            lambda name: name == storage.final_namespace
        )

        await storage.initialize()

        mock_qdrant_client.create_payload_index.assert_called_with(
            collection_name=storage.final_namespace,
            field_name="workspace_id",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
                is_tenant=True,
            ),
        )
        mock_qdrant_client.scroll.assert_not_called()

    async def test_scenario_1_new_workspace_creation(self, mock_qdrant_client):
        """New workspace creates suffixed collection directly with no migration."""
        storage = QdrantVectorDBStorage(
            namespace="chunks",
            global_config=_GLOBAL_CONFIG,
            embedding_func=_embedding_func("text-embedding-3-large", 3072),
            workspace="test_new",
        )
        mock_qdrant_client.collection_exists.return_value = False

        await storage.initialize()

        expected_collection = "lightrag_vdb_chunks_text_embedding_3_large_3072d"
        assert storage.final_namespace == expected_collection
        create_calls = mock_qdrant_client.create_collection.call_args_list
        assert len(create_calls) > 0
        assert (
            create_calls[0][0][0] == expected_collection
            or create_calls[0].kwargs.get("collection_name") == expected_collection
        )
        mock_qdrant_client.scroll.assert_not_called()

    async def test_scenario_2_legacy_upgrade_migration(self, mock_qdrant_client):
        """Upgrading from untagged legacy collection migrates into model-suffixed collection."""
        storage = QdrantVectorDBStorage(
            namespace="chunks",
            global_config=_GLOBAL_CONFIG,
            embedding_func=_embedding_func("text-embedding-ada-002", 1536),
            workspace="test_legacy",
        )
        legacy_collection = "lightrag_vdb_chunks"
        new_collection = storage.final_namespace
        mock_qdrant_client.collection_exists.side_effect = (
            lambda name: name == legacy_collection
        )

        legacy_collection_info = MagicMock()
        legacy_collection_info.payload_schema = {}
        legacy_collection_info.config.params.vectors.size = 1536
        mock_qdrant_client.get_collection.return_value = legacy_collection_info

        migration_state = {"new_workspace_count": 0}

        def count_mock(collection_name, exact=True, count_filter=None):
            mock_result = MagicMock()
            if collection_name == legacy_collection:
                mock_result.count = 150
            elif collection_name == new_collection:
                mock_result.count = migration_state["new_workspace_count"]
            else:
                mock_result.count = 0
            return mock_result

        mock_qdrant_client.count.side_effect = count_mock

        mock_points = []
        for i in range(10):
            point = MagicMock()
            point.id = f"legacy-{i}"
            point.vector = [0.1] * 1536
            point.payload = {"content": f"Legacy document {i}", "id": f"doc-{i}"}
            mock_points.append(point)

        mock_qdrant_client.scroll.side_effect = [
            (mock_points, "_"),
            (mock_points, "offset1"),
            ([], None),
        ]

        def upsert_mock(*args, **kwargs):
            migration_state["new_workspace_count"] = 150
            return None

        mock_qdrant_client.upsert.side_effect = upsert_mock

        await storage.initialize()

        expected_new_collection = "lightrag_vdb_chunks_text_embedding_ada_002_1536d"
        assert storage.final_namespace == expected_new_collection
        mock_qdrant_client.count.assert_any_call(
            collection_name=legacy_collection, exact=True
        )
        mock_qdrant_client.create_collection.assert_called()
        assert (
            mock_qdrant_client.scroll.call_args_list[0].kwargs["collection_name"]
            == legacy_collection
        )
        assert (
            mock_qdrant_client.upsert.call_args_list[0].kwargs["collection_name"]
            == new_collection
        )

    async def test_scenario_3_multi_model_coexistence(self, mock_qdrant_client):
        """Distinct embedding models target independent model-suffixed collections."""
        storage_a = QdrantVectorDBStorage(
            namespace="chunks",
            global_config=_GLOBAL_CONFIG,
            embedding_func=_embedding_func("bge-small", 768),
            workspace="workspace_a",
        )
        storage_b = QdrantVectorDBStorage(
            namespace="chunks",
            global_config=_GLOBAL_CONFIG,
            embedding_func=_embedding_func("bge-large", 1024),
            workspace="workspace_b",
        )

        assert storage_a.final_namespace == "lightrag_vdb_chunks_bge_small_768d"
        assert storage_b.final_namespace == "lightrag_vdb_chunks_bge_large_1024d"
        assert storage_a.embedding_func.embedding_dim == 768
        assert storage_b.embedding_func.embedding_dim == 1024

    @pytest.mark.parametrize(
        ("legacy_count", "expect_deleted"),
        [
            pytest.param(0, True, id="empty_legacy_auto_cleaned"),
            pytest.param(50, False, id="nonempty_legacy_preserved"),
        ],
    )
    async def test_case1_both_collections_exist_legacy_cleanup_policy(
        self, mock_qdrant_client, legacy_count, expect_deleted
    ):
        """Empty legacy collection is auto-deleted; non-empty legacy is preserved."""
        storage = QdrantVectorDBStorage(
            namespace="chunks",
            global_config=_GLOBAL_CONFIG,
            embedding_func=_embedding_func(),
            workspace="test_ws",
        )
        legacy_collection = "lightrag_vdb_chunks"
        new_collection = storage.final_namespace

        mock_qdrant_client.collection_exists.side_effect = lambda name: name in (
            legacy_collection,
            new_collection,
        )

        def count_mock(collection_name, exact=True, count_filter=None):
            mock_result = MagicMock()
            mock_result.count = (
                legacy_count if collection_name == legacy_collection else 100
            )
            return mock_result

        mock_qdrant_client.count.side_effect = count_mock

        collection_info = MagicMock()
        collection_info.payload_schema = {"workspace_id": True}
        mock_qdrant_client.get_collection.return_value = collection_info

        await storage.initialize()

        deleted_names = [
            call[0][0] if call[0] else call.kwargs.get("collection_name")
            for call in mock_qdrant_client.delete_collection.call_args_list
        ]
        if expect_deleted:
            assert legacy_collection in deleted_names
        else:
            assert legacy_collection not in deleted_names


# ---------------------------------------------------------------------------
# Vector-space mismatch refusal (docs/design/VectorSpaceProvenance.md)
# ---------------------------------------------------------------------------


class TestQdrantVectorSpaceRefusal:
    @staticmethod
    def _client(*, legacy_name, legacy_dim, legacy_total, workspace_count=None):
        """A Qdrant client holding only ``legacy_name``, at ``legacy_dim``."""
        client = MagicMock()
        client.collection_exists.side_effect = lambda name: name == legacy_name

        legacy_info = MagicMock()
        legacy_info.config.params.vectors.size = legacy_dim
        legacy_info.payload_schema = (
            {"workspace_id": MagicMock()} if workspace_count is not None else {}
        )
        client.get_collection.return_value = legacy_info
        client.scroll.return_value = ([], None)

        def count(collection_name, exact=True, count_filter=None):
            result = MagicMock()
            if collection_name != legacy_name:
                result.count = 0
            elif count_filter is not None and workspace_count is not None:
                result.count = workspace_count
            else:
                result.count = legacy_total
            return result

        client.count.side_effect = count
        return client

    @staticmethod
    def _setup(client, *, dim, workspace="test_ws"):
        return QdrantVectorDBStorage.setup_collection(
            client,
            f"lightrag_vdb_chunks_test_model_{dim}d",
            namespace="chunks",
            workspace=workspace,
            vectors_config=models.VectorParams(
                size=dim, distance=models.Distance.COSINE
            ),
            hnsw_config=models.HnswConfigDiff(payload_m=16, m=0),
            model_suffix=f"test_model_{dim}d",
        )

    def test_legacy_at_another_dimension_raises_the_typed_refusal(self):
        """Refusal is VectorSpaceMismatchError (and NOT DataMigrationError) before writes."""
        client = self._client(
            legacy_name="lightrag_vdb_chunks", legacy_dim=1536, legacy_total=100
        )

        with pytest.raises(VectorSpaceMismatchError) as excinfo:
            self._setup(client, dim=768)

        error = excinfo.value
        assert not isinstance(error, DataMigrationError)
        assert error.container == "lightrag_vdb_chunks"
        assert (error.stored_dim, error.expected_dim) == (1536, 768)
        client.create_collection.assert_not_called()
        client.upsert.assert_not_called()

    def test_a_tagged_legacy_holding_no_rows_for_this_workspace_does_not_refuse(self):
        """The refusal is scoped to what this workspace would migrate."""
        client = self._client(
            legacy_name="lightrag_vdb_chunks",
            legacy_dim=1536,
            legacy_total=100,
            workspace_count=0,
        )

        self._setup(client, dim=768)

        client.create_collection.assert_called_once()

    def test_a_tagged_legacy_holding_rows_for_this_workspace_still_refuses(self):
        client = self._client(
            legacy_name="lightrag_vdb_chunks",
            legacy_dim=1536,
            legacy_total=100,
            workspace_count=7,
        )

        with pytest.raises(VectorSpaceMismatchError) as excinfo:
            self._setup(client, dim=768)

        assert "7 record(s)" in str(excinfo.value)

    async def test_a_refused_instance_can_still_be_dropped(self):
        """drop() then initialize() is the recovery, so the refusal may not
        leave the instance without its flush lock."""
        client = self._client(
            legacy_name="lightrag_vdb_chunks", legacy_dim=1536, legacy_total=100
        )

        with patch("lightrag.kg.qdrant_impl.QdrantClient", return_value=client):
            storage = QdrantVectorDBStorage(
                namespace="chunks",
                global_config=_GLOBAL_CONFIG,
                embedding_func=_embedding_func(),
                workspace="test_ws",
            )
            with pytest.raises(VectorSpaceMismatchError):
                await storage.initialize()

            assert storage._flush_lock is not None
            result = await storage.drop()

        assert result["status"] == "success"
        client.delete_collection.assert_called_once_with(
            collection_name="lightrag_vdb_chunks"
        )


# ---------------------------------------------------------------------------
# Sibling-model cleanup (drop_sibling_models)
# ---------------------------------------------------------------------------


class TestQdrantDropSiblingModels:
    @staticmethod
    def _storage(client, embedding_func=None):
        with patch("lightrag.kg.qdrant_impl.QdrantClient", return_value=client):
            storage = QdrantVectorDBStorage(
                namespace="chunks",
                global_config=_GLOBAL_CONFIG,
                embedding_func=embedding_func or _embedding_func("model-b", 768),
                workspace="ws",
            )
        storage._client = client
        return storage

    @staticmethod
    def _client(collections: dict[str, dict]):
        """``collections`` maps name -> payload_schema."""
        client = MagicMock()
        client.get_collections.return_value = SimpleNamespace(
            collections=[SimpleNamespace(name=n) for n in collections]
        )
        client.get_collection.side_effect = lambda n: SimpleNamespace(
            payload_schema=collections[n]
        )
        return client

    async def test_clears_only_this_workspaces_points_from_each_sibling_model_collection(
        self,
    ):
        client = self._client(
            {
                "lightrag_vdb_chunks_model_b_768d": {"workspace_id": object()},
                "lightrag_vdb_chunks_model_a_1536d": {"workspace_id": object()},
                "lightrag_vdb_chunks_model_c_1024d": {"workspace_id": object()},
            }
        )
        storage = self._storage(client)

        result = await storage.drop_sibling_models()

        assert result["status"] == "success"
        cleared = {c.kwargs["collection_name"] for c in client.delete.call_args_list}
        assert cleared == {
            "lightrag_vdb_chunks_model_a_1536d",
            "lightrag_vdb_chunks_model_c_1024d",
        }
        for call in client.delete.call_args_list:
            selector = call.kwargs["points_selector"]
            (condition,) = selector.filter.must
            assert (condition.key, condition.match.value) == ("workspace_id", "ws")
            assert call.kwargs["wait"] is True
        # Shared collections are emptied of this workspace, never removed.
        client.delete_collection.assert_not_called()

    async def test_leaves_collections_it_cannot_prove_it_owns(self):
        client = self._client(
            {
                "lightrag_vdb_chunks_foreign_512d": {},
                "lightrag_vdb_chunks_scratch": {"workspace_id": object()},
                "lightrag_vdb_entities_model_a_1536d": {"workspace_id": object()},
            }
        )
        storage = self._storage(client)

        result = await storage.drop_sibling_models()

        assert result["status"] == "success"
        client.delete.assert_not_called()

    async def test_reports_a_backend_failure(self):
        client = self._client({"lightrag_vdb_chunks_model_a_1536d": {"workspace_id": 1}})
        client.delete.side_effect = RuntimeError("boom")
        storage = self._storage(client)

        result = await storage.drop_sibling_models()

        assert result["status"] == "error"
        assert "boom" in result["message"]

    async def test_every_client_call_runs_off_the_event_loop(self):
        """qdrant-client is synchronous: a delete with ``wait=True`` on the loop
        thread stalls every request the server is serving."""
        loop_thread = threading.get_ident()
        callers: dict[str, int] = {}
        collections = {"lightrag_vdb_chunks_model_a_1536d": {"workspace_id": object()}}
        client = self._client(collections)

        def recording(name, result):
            def call(*args, **kwargs):
                callers[name] = threading.get_ident()
                return result(*args, **kwargs)

            return call

        client.get_collections.side_effect = recording(
            "get_collections",
            lambda: SimpleNamespace(
                collections=[SimpleNamespace(name=n) for n in collections]
            ),
        )
        client.get_collection.side_effect = recording(
            "get_collection", lambda n: SimpleNamespace(payload_schema=collections[n])
        )
        client.delete.side_effect = recording("delete", lambda **kwargs: None)

        result = await self._storage(client).drop_sibling_models()

        assert result["status"] == "success"
        assert sorted(callers) == ["delete", "get_collection", "get_collections"]
        assert loop_thread not in callers.values()

    def test_model_container_name_is_the_collection_that_model_would_attach_to(self):
        other = self._storage(MagicMock(), _embedding_func("Model-A", 1536))
        assert (
            self._storage(MagicMock()).model_container_name("Model-A", 1536)
            == other.final_namespace
        )
