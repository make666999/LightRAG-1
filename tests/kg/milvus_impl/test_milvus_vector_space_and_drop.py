"""Milvus vector-space refusal, ownership markers, and workspace/sibling-model drops.

Covers:
- Embedding-space refusal raises ``VectorSpaceMismatchError`` on dimension mismatch,
  keeps malformed/undeclared schemas as ``ValueError``, and skips incompatible legacy
  migration sources.
- Ownership markers on new collections and ``model_container_name`` computation.
- ``drop_sibling_models()`` and ``drop()`` removing owned backups/legacy collections
  in crash-safe order without touching foreign or unvouched collections.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("pymilvus", reason="pymilvus is required for Milvus storage tests")

from lightrag.exceptions import VectorSpaceMismatchError  # noqa: E402
from lightrag.kg.milvus_impl import MilvusVectorDBStorage  # noqa: E402
from lightrag.kg.shared_storage import (  # noqa: E402
    finalize_share_data,
    initialize_share_data,
)
from lightrag.utils import EmbeddingFunc  # noqa: E402

pytestmark = pytest.mark.offline

OURS = "[lightrag:workspace=a;namespace=chunks]"
VECTOR_SCHEMA = [{"name": "id"}, {"name": "vector"}]


@pytest.fixture(autouse=True)
def _shared_storage():
    initialize_share_data(workers=1)
    yield
    finalize_share_data()


def _embedding_func(
    dim: int = 768, model_name: str | None = "test-model"
) -> EmbeddingFunc:
    async def embed(texts, **kwargs):
        return np.array([[0.1] * dim for _ in texts])

    return EmbeddingFunc(embedding_dim=dim, func=embed, model_name=model_name)


def fake_milvus(collections=None, fields=None):
    """A client that remembers which collections exist and records drop order."""
    client = MagicMock()
    client.collections = dict(collections or {})
    client.dropped = []
    schemas = dict(fields or {})

    def drop(name, *args, **kwargs):
        del client.collections[name]
        client.dropped.append(name)

    def create(collection_name=None, schema=None, **kwargs):
        client.collections[collection_name] = getattr(schema, "description", "")

    def rename(old_name, new_name, *args, **kwargs):
        client.collections[new_name] = client.collections.pop(old_name)

    client.has_collection.side_effect = lambda name, *a, **k: name in client.collections
    client.list_collections.side_effect = lambda *a, **k: list(client.collections)
    client.describe_collection.side_effect = lambda name, *a, **k: {
        "description": client.collections[name],
        "fields": schemas.get(name, VECTOR_SCHEMA),
    }
    client.drop_collection.side_effect = drop
    client.create_collection.side_effect = create
    client.rename_collection.side_effect = rename
    return client


def _storage(
    client, *, model: str | None = "model-b", dim: int = 768, workspace="a", namespace="chunks"
) -> MilvusVectorDBStorage:
    storage = MilvusVectorDBStorage(
        namespace=namespace,
        workspace=workspace,
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        embedding_func=_embedding_func(dim=dim, model_name=model),
        meta_fields={"content"},
    )
    storage._client = client
    return storage


# ---------------------------------------------------------------------------
# Embedding-space refusal
# ---------------------------------------------------------------------------


class TestMilvusVectorSpaceRefusal:
    @staticmethod
    def _bare_storage(dim: int = 768) -> MilvusVectorDBStorage:
        storage = MilvusVectorDBStorage.__new__(MilvusVectorDBStorage)
        storage.workspace = "test_ws"
        storage.namespace = "chunks"
        storage.legacy_namespace = "test_ws_chunks"
        storage.final_namespace = f"test_ws_chunks_test_model_{dim}d"
        storage.embedding_func = _embedding_func(dim)
        storage._client = MagicMock()
        return storage

    @staticmethod
    def _collection_info(dim):
        return {
            "fields": [
                {"name": "vector", "type": "FloatVector", "params": {"dim": dim}}
            ]
        }

    def test_dimension_mismatch_raises_the_typed_refusal(self):
        storage = self._bare_storage()

        with pytest.raises(VectorSpaceMismatchError) as excinfo:
            storage._check_vector_dimension(self._collection_info(1536))

        error = excinfo.value
        assert error.container == storage.final_namespace
        assert (error.stored_dim, error.expected_dim) == (1536, 768)

    @pytest.mark.parametrize(
        ("info", "declared_dim"),
        [
            pytest.param(
                {"fields": [{"name": "vector", "type": "FloatVector", "params": {}}]},
                768,
                id="missing-stored-dim",
            ),
            pytest.param(
                {
                    "fields": [
                        {"name": "vector", "type": "FloatVector", "params": {"dim": 1536}}
                    ]
                },
                None,
                id="undeclared-embedding-dim",
            ),
            pytest.param(
                {
                    "fields": [
                        {
                            "name": "vector",
                            "type": "FloatVector",
                            "params": {"dim": "not-a-number"},
                        }
                    ]
                },
                768,
                id="unparseable-stored-dim",
            ),
        ],
    )
    def test_unreadable_or_missing_dimensions_stay_value_error_not_refusal(
        self, info, declared_dim
    ):
        """A schema we cannot read or declare must not trigger a rebuild drop."""
        storage = self._bare_storage()
        if declared_dim is None:
            storage.embedding_func.embedding_dim = None

        with pytest.raises(ValueError) as excinfo:
            storage._check_vector_dimension(info)

        assert not isinstance(excinfo.value, VectorSpaceMismatchError)

    def test_validation_does_not_reframe_the_refusal_as_a_migration_failure(self):
        storage = self._bare_storage()
        storage._client.describe_collection.return_value = self._collection_info(1536)

        with pytest.raises(VectorSpaceMismatchError):
            storage._validate_collection_and_load()

    def test_an_incompatible_legacy_collection_is_skipped_not_refused(self):
        storage = self._bare_storage()
        storage._client.has_collection.side_effect = lambda name: (
            name == storage.legacy_namespace
        )
        storage._client.describe_collection.return_value = self._collection_info(1536)
        storage._create_collection_with_schema = MagicMock()
        storage._ensure_collection_loaded = MagicMock()
        storage._migrate_collection_schema = MagicMock()

        storage._create_collection_if_not_exist()

        storage._create_collection_with_schema.assert_called_once_with(
            storage.final_namespace
        )
        storage._migrate_collection_schema.assert_not_called()


# ---------------------------------------------------------------------------
# Ownership marker
# ---------------------------------------------------------------------------


def test_new_collections_carry_the_ownership_marker():
    storage = _storage(fake_milvus())
    schema = storage._create_schema_for_namespace()
    assert storage._ownership_marker() == OURS
    assert OURS in schema.description


def test_model_container_name_is_the_name_that_model_would_attach_to():
    other = _storage(fake_milvus(), model="Model-A", dim=1536)
    assert (
        _storage(fake_milvus()).model_container_name("Model-A", 1536)
        == other.final_namespace
    )


# ---------------------------------------------------------------------------
# drop_sibling_models
# ---------------------------------------------------------------------------


async def test_drops_marked_siblings_only():
    client = fake_milvus(
        {
            "a_chunks_model_b_768d": f"LightRAG chunks vector storage {OURS}",
            "a_chunks_model_a_1536d": f"LightRAG chunks vector storage {OURS}",
            "a_chunks_m_entities_model_a_1536d": (
                "LightRAG entities vector storage "
                "[lightrag:workspace=a_chunks_m;namespace=entities]"
            ),
        }
    )
    result = await _storage(client).drop_sibling_models()

    assert result["status"] == "success"
    assert client.dropped == ["a_chunks_model_a_1536d"]
    assert sorted(client.collections) == [
        "a_chunks_m_entities_model_a_1536d",
        "a_chunks_model_b_768d",
    ]


async def test_a_siblings_backups_go_before_the_sibling_itself():
    client = fake_milvus(
        {
            "a_chunks_model_a_1536d": OURS,
            "a_chunks_model_a_1536d_old": OURS,
            "a_chunks_model_a_1536d_temp": OURS,
        }
    )
    result = await _storage(client).drop_sibling_models()

    assert result["status"] == "success"
    assert client.dropped == [
        "a_chunks_model_a_1536d_temp",
        "a_chunks_model_a_1536d_old",
        "a_chunks_model_a_1536d",
    ]


async def test_a_backup_shares_the_verdict_of_the_collection_it_belongs_to():
    client = fake_milvus(
        {
            "a_chunks_model_a_1536d": OURS,
            "a_chunks_model_a_1536d_old": "LightRAG chunks vector storage",
        }
    )
    result = await _storage(client).drop_sibling_models()

    assert result["status"] == "success"
    assert client.collections == {}


async def test_the_active_models_own_backups_are_not_siblings():
    client = fake_milvus(
        {
            "a_chunks_model_b_768d": OURS,
            "a_chunks_model_b_768d_old": OURS,
            "a_chunks_model_b_768d_temp": OURS,
        }
    )
    result = await _storage(client).drop_sibling_models()

    assert result["status"] == "success"
    assert client.dropped == []


async def test_unmarked_candidates_are_kept_and_reported():
    client = fake_milvus({"a_chunks_model_a_1536d": "LightRAG chunks vector storage"})
    storage = _storage(client)

    result = await storage.drop_sibling_models()

    assert result["status"] == "error"
    assert "a_chunks_model_a_1536d" in result["message"]
    assert client.dropped == []
    assert await storage.unvouched_sibling_containers() == ["a_chunks_model_a_1536d"]


async def test_a_vouched_unmarked_sibling_is_dropped_with_its_backups():
    client = fake_milvus(
        {
            "a_chunks_model_a_1536d": "LightRAG chunks vector storage",
            "a_chunks_model_a_1536d_old": "LightRAG chunks vector storage",
            "a_chunks_model_c_8d": "LightRAG chunks vector storage",
        }
    )
    result = await _storage(client).drop_sibling_models(
        vouched=["a_chunks_model_a_1536d"]
    )

    assert result["status"] == "error"
    assert "a_chunks_model_c_8d" in result["message"]
    assert "a_chunks_model_a_1536d" not in result["message"]
    assert client.dropped == ["a_chunks_model_a_1536d_old", "a_chunks_model_a_1536d"]


async def test_a_voucher_never_overrides_another_owners_marker():
    foreign = "[lightrag:workspace=a_chunks_m;namespace=entities]"
    client = fake_milvus({"a_chunks_m_entities_model_a_1536d": foreign})
    storage = _storage(client)

    result = await storage.drop_sibling_models(
        vouched=["a_chunks_m_entities_model_a_1536d"]
    )

    assert result["status"] == "success"
    assert client.dropped == []
    assert await storage.unvouched_sibling_containers() == []


@pytest.mark.parametrize(
    "family",
    [
        pytest.param(
            {
                "a_chunks_m_entities_x_8d": "FOREIGN",
                "a_chunks_m_entities_x_8d_old": OURS,
            },
            id="our-marker-on-the-backup",
        ),
        pytest.param(
            {
                "a_chunks_m_entities_x_8d": OURS,
                "a_chunks_m_entities_x_8d_old": "FOREIGN",
            },
            id="our-marker-on-the-collection",
        ),
    ],
)
@pytest.mark.parametrize("vouched", [(), ("a_chunks_m_entities_x_8d",)])
async def test_a_family_marked_for_two_owners_is_refused_whole(family, vouched):
    foreign = "[lightrag:workspace=a_chunks_m;namespace=entities]"
    client = fake_milvus(
        {name: foreign if d == "FOREIGN" else d for name, d in family.items()}
    )
    storage = _storage(client)

    result = await storage.drop_sibling_models(vouched=vouched)

    assert result["status"] == "error"
    assert "another owner" in result["message"]
    assert "a_chunks_m_entities_x_8d" in result["message"]
    assert client.dropped == []
    assert await storage.unvouched_sibling_containers() == []


async def test_listing_unvouched_siblings_drops_nothing():
    client = fake_milvus(
        {
            "a_chunks_model_a_1536d": "",
            "a_chunks_model_c_8d": OURS,
            "a_chunks_model_b_768d": "",  # active collection
            "a_chunks": "",  # legacy collection
        }
    )
    assert await _storage(client).unvouched_sibling_containers() == [
        "a_chunks_model_a_1536d"
    ]
    assert client.dropped == []


async def test_a_backend_error_is_an_error_result():
    client = fake_milvus()
    client.list_collections.side_effect = ConnectionError("milvus down")

    result = await _storage(client).drop_sibling_models()

    assert result == {"status": "error", "message": "milvus down"}


# ---------------------------------------------------------------------------
# drop(): active model's backups and legacy collection
# ---------------------------------------------------------------------------


async def test_drop_removes_every_backup_before_it_rebuilds_the_collection():
    client = fake_milvus(
        {
            "a_chunks_model_b_768d": OURS,
            "a_chunks_model_b_768d_old": OURS,
            "a_chunks_model_b_768d_temp": OURS,
            "a_chunks": "LightRAG chunks vector storage",
            "a_chunks_old": "LightRAG chunks vector storage",
            "a_chunks_temp": "LightRAG chunks vector storage",
        }
    )
    result = await _storage(client).drop()

    assert result["status"] == "success"
    assert client.dropped[-1] == "a_chunks_model_b_768d"
    assert sorted(client.dropped[:-1]) == [
        "a_chunks",
        "a_chunks_model_b_768d_old",
        "a_chunks_model_b_768d_temp",
        "a_chunks_old",
        "a_chunks_temp",
    ]
    assert list(client.collections) == ["a_chunks_model_b_768d"]
    assert OURS in client.collections["a_chunks_model_b_768d"]


async def test_drop_leaves_a_legacy_collection_nothing_would_migrate(
    lightrag_log_records,
):
    client = fake_milvus(
        {"a_chunks_model_b_768d": OURS, "a_chunks": ""},
        fields={"a_chunks": [{"name": "id"}, {"name": "text"}]},
    )
    result = await _storage(client).drop()

    assert result["status"] == "success"
    assert client.dropped == ["a_chunks_model_b_768d"]
    assert "a_chunks" in client.collections
    assert any(
        "'a_chunks'" in r.getMessage() and "no vector field" in r.getMessage()
        for r in lightrag_log_records
    )


async def test_drop_leaves_a_backup_marked_as_another_owners(lightrag_log_records):
    foreign = "[lightrag:workspace=a_chunks_model;namespace=entities]"
    client = fake_milvus(
        {"a_chunks_model_b_768d": OURS, "a_chunks_model_b_768d_old": foreign}
    )
    result = await _storage(client).drop()

    assert result["status"] == "success"
    assert client.dropped == ["a_chunks_model_b_768d"]
    assert client.collections["a_chunks_model_b_768d_old"] == foreign
    assert any("another owner's" in r.getMessage() for r in lightrag_log_records)


async def test_drop_without_a_model_suffix_treats_no_collection_as_legacy():
    client = fake_milvus({"a_chunks": "", "a_chunks_old": ""})
    storage = _storage(client, model=None)
    assert storage.final_namespace == storage.legacy_namespace == "a_chunks"

    result = await storage.drop()

    assert result["status"] == "success"
    assert client.dropped == ["a_chunks_old", "a_chunks"]
    assert list(client.collections) == ["a_chunks"]


# ---------------------------------------------------------------------------
# Regression: nothing comes back when the operator switches models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "leftover",
    [
        pytest.param({"a_chunks": "LightRAG chunks vector storage"}, id="legacy"),
        pytest.param({"a_chunks_model_a_768d_old": OURS}, id="sibling-old-backup"),
        pytest.param({"a_chunks_model_a_768d_temp": OURS}, id="sibling-temp"),
    ],
)
async def test_a_cleared_sibling_is_created_empty_when_its_model_returns(leftover):
    client = fake_milvus(
        {
            "a_chunks_model_b_768d": OURS,
            "a_chunks_model_a_768d": OURS,
            **leftover,
        }
    )
    result = await _storage(client).drop_all_models()
    assert result["status"] == "success"
    assert list(client.collections) == ["a_chunks_model_b_768d"]

    returning = _storage(client, model="model-a")
    returning._create_collection_if_not_exist()

    assert "a_chunks_model_a_768d" in client.collections
    client.rename_collection.assert_not_called()
    client.query_iterator.assert_not_called()
    client.insert.assert_not_called()
    assert client.create_collection.call_args.kwargs["collection_name"] == (
        "a_chunks_model_a_768d"
    )
