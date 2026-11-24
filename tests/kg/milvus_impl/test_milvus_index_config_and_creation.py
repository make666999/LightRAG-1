"""Tests for Milvus index configuration, kwargs bridging, metric policy, and collection/index creation."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import lightrag.kg.milvus_impl as milvus_impl
from lightrag.kg.milvus_impl import (
    MILVUS_MAX_VARCHAR_BYTES,
    MilvusException,
    MilvusIndexConfig,
    MilvusVectorDBStorage,
)

pytestmark = pytest.mark.offline


class _EmbeddingFunc:
    """Minimal embedding stand-in exposing ``embedding_dim`` and ``model_name``."""

    def __init__(self, dim: int = 128, model_name="text-embedding-3-small"):
        self.embedding_dim = dim
        self.model_name = model_name


def _make_storage(
    *,
    namespace: str = "entities",
    workspace: str = "test_workspace",
    dim: int = 128,
    embedding_func=None,
    extra_kwargs: dict | None = None,
    working_dir: str | None = None,
    embedding_batch_num: int = 100,
) -> MilvusVectorDBStorage:
    if embedding_func is None:
        embedding_func = MagicMock()
        embedding_func.embedding_dim = dim
    kwargs = {"cosine_better_than_threshold": 0.3}
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    global_config: dict = {
        "embedding_batch_num": embedding_batch_num,
        "vector_db_storage_cls_kwargs": kwargs,
    }
    if working_dir is not None:
        global_config["working_dir"] = working_dir
    return MilvusVectorDBStorage(
        namespace=namespace,
        workspace=workspace,
        global_config=global_config,
        embedding_func=embedding_func,
        meta_fields=set(),
    )


def _make_model_storage(
    namespace: str = "entities", workspace: str = "test_workspace", dim: int = 128
) -> MilvusVectorDBStorage:
    return _make_storage(
        namespace=namespace,
        workspace=workspace,
        embedding_func=_EmbeddingFunc(dim=dim),
        working_dir="/tmp/lightrag",
    )


def _field_max_length(field) -> int:
    return int(field.params["max_length"])


def _collection_info(field_names) -> dict:
    fields = [
        {"name": "id", "type": "VarChar", "is_primary": True},
        {"name": "vector", "type": "FloatVector", "params": {"dim": 128}},
        {"name": "created_at", "type": "Int64"},
    ]
    fields.extend(
        {
            "name": field_name,
            "type": "VarChar",
            "params": {"max_length": MILVUS_MAX_VARCHAR_BYTES},
        }
        for field_name in field_names
    )
    return {"fields": fields}


def _wire_collection_state(storage, collections, describe_by_name=None):
    storage._client = MagicMock()
    describe_by_name = describe_by_name or {}

    def has_collection(collection_name):
        return collection_name in collections

    def create_collection(collection_name, schema):
        collections.add(collection_name)

    def drop_collection(collection_name):
        collections.discard(collection_name)

    def rename_collection(source, target):
        collections.discard(source)
        collections.add(target)

    def describe_collection(collection_name):
        return describe_by_name.get(collection_name, _collection_info([]))

    storage._client.has_collection.side_effect = has_collection
    storage._client.create_collection.side_effect = create_collection
    storage._client.drop_collection.side_effect = drop_collection
    storage._client.rename_collection.side_effect = rename_collection
    storage._client.describe_collection.side_effect = describe_collection
    return storage._client


# ---------------------------------------------------------------------------
# MilvusIndexConfig
# ---------------------------------------------------------------------------


class TestMilvusIndexConfig:
    """MilvusIndexConfig defaults, validation, and param-builder unit tests."""

    def test_defaults_and_field_names(self):
        config = MilvusIndexConfig()
        assert config.index_type == "AUTOINDEX"
        assert config.metric_type == "COSINE"
        assert config.hnsw_m == 16
        assert config.hnsw_ef_construction == 360
        assert config.hnsw_ef == 200
        assert config.sq_type == "SQ8"
        assert not config.sq_refine
        assert config.sq_refine_type == "FP32"
        assert config.sq_refine_k == 10
        assert config.ivf_nlist == 1024
        assert config.ivf_nprobe == 16

        assert MilvusIndexConfig.get_config_field_names() == {
            "index_type",
            "metric_type",
            "hnsw_m",
            "hnsw_ef_construction",
            "hnsw_ef",
            "sq_type",
            "sq_refine",
            "sq_refine_type",
            "sq_refine_k",
            "ivf_nlist",
            "ivf_nprobe",
        }

    def test_env_override_case_normalization_and_boundaries(self):
        with patch.dict(
            os.environ, {"MILVUS_INDEX_TYPE": "HNSW", "MILVUS_HNSW_M": "64"}
        ):
            config = MilvusIndexConfig()
            assert config.index_type == "HNSW"
            assert config.hnsw_m == 64

        with patch.dict(os.environ, {"MILVUS_HNSW_M": "invalid"}):
            assert MilvusIndexConfig().hnsw_m == 16

        assert MilvusIndexConfig(index_type="hnsw").index_type == "HNSW"
        assert MilvusIndexConfig(hnsw_m=2).hnsw_m == 2
        assert MilvusIndexConfig(hnsw_m=2048).hnsw_m == 2048

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("true", True), ("false", False), ("1", True), ("0", False)],
    )
    def test_env_bool_parsing(self, raw, expected):
        with patch.dict(os.environ, {"MILVUS_HNSW_SQ_REFINE": raw}):
            assert MilvusIndexConfig(index_type="HNSW_SQ").sq_refine is expected

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            pytest.param(
                {"index_type": "INVALID_INDEX"},
                "Unsupported index type",
                id="invalid-index-type",
            ),
            pytest.param({"hnsw_m": 1}, "hnsw_m must be in", id="hnsw-m-low"),
            pytest.param({"hnsw_m": 3000}, "hnsw_m must be in", id="hnsw-m-high"),
            pytest.param(
                {"hnsw_ef_construction": 0},
                "hnsw_ef_construction must be",
                id="hnsw-ef-construction-zero",
            ),
            pytest.param({"ivf_nlist": 0}, "ivf_nlist must be in", id="ivf-nlist-low"),
            pytest.param(
                {"ivf_nlist": 70000}, "ivf_nlist must be in", id="ivf-nlist-high"
            ),
            pytest.param(
                {"index_type": "HNSW_SQ", "sq_type": "INVALID"},
                "Unsupported sq_type",
                id="invalid-sq-type",
            ),
            pytest.param(
                {
                    "index_type": "HNSW_SQ",
                    "sq_refine": True,
                    "sq_refine_type": "INVALID",
                },
                "Unsupported refine_type",
                id="invalid-refine-type",
            ),
        ],
    )
    def test_invalid_config_parameters_raise(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            MilvusIndexConfig(**kwargs)

    @pytest.mark.parametrize(
        ("index_type", "version"),
        [
            ("HNSW_SQ", "2.6.8"),
            ("HNSW_SQ", "2.6.9"),
            ("HNSW_SQ", "2.7.0"),
            ("HNSW_SQ", "2.6.9-dev"),
            ("HNSW", "2.4.0"),
            ("HNSW", "2.6.9"),
        ],
    )
    def test_version_validation_accepts_compatible_versions(self, index_type, version):
        config = MilvusIndexConfig(index_type=index_type)
        assert config.validate_milvus_version(version) is None

    @pytest.mark.parametrize("version", ["2.6.7", "2.5.0"])
    def test_version_validation_rejects_old_milvus_for_hnsw_sq(self, version):
        config = MilvusIndexConfig(index_type="HNSW_SQ")
        with pytest.raises(ValueError, match="HNSW_SQ requires Milvus 2.6.8"):
            config.validate_milvus_version(version)

    @pytest.mark.parametrize(
        ("kwargs", "expected_index_type", "expected_params", "absent_keys"),
        [
            pytest.param(
                {"index_type": "AUTOINDEX", "metric_type": "COSINE"},
                "AUTOINDEX",
                {},
                (),
                id="autoindex",
            ),
            pytest.param(
                {
                    "index_type": "HNSW",
                    "metric_type": "COSINE",
                    "hnsw_m": 32,
                    "hnsw_ef_construction": 256,
                },
                "HNSW",
                {"M": 32, "efConstruction": 256},
                (),
                id="hnsw",
            ),
            pytest.param(
                {
                    "index_type": "HNSW_SQ",
                    "sq_type": "SQ8",
                    "sq_refine": True,
                    "sq_refine_type": "FP32",
                },
                "HNSW_SQ",
                {"sq_type": "SQ8", "refine": True, "refine_type": "FP32"},
                (),
                id="hnsw-sq-with-refine",
            ),
            pytest.param(
                {"index_type": "HNSW_SQ", "sq_type": "SQ8", "sq_refine": False},
                "HNSW_SQ",
                {"sq_type": "SQ8"},
                ("refine", "refine_type"),
                id="hnsw-sq-no-refine",
            ),
            pytest.param(
                {"index_type": "IVF_FLAT", "ivf_nlist": 2048},
                "IVF_FLAT",
                {"nlist": 2048},
                (),
                id="ivf-flat",
            ),
        ],
    )
    def test_build_index_params(
        self, kwargs, expected_index_type, expected_params, absent_keys
    ):
        config = MilvusIndexConfig(**kwargs)
        mock_index_params = MagicMock()

        result = config.build_index_params(mock_index_params)

        assert result == mock_index_params
        mock_index_params.add_index.assert_called_once()
        call_kwargs = mock_index_params.add_index.call_args[1]
        assert call_kwargs["field_name"] == "vector"
        assert call_kwargs["index_type"] == expected_index_type
        if "metric_type" in kwargs:
            assert call_kwargs["metric_type"] == kwargs["metric_type"]
        for key, value in expected_params.items():
            assert call_kwargs["params"][key] == value
        for key in absent_keys:
            assert key not in call_kwargs["params"]

    def test_build_index_params_when_index_params_is_none(self):
        custom = MilvusIndexConfig(index_type="HNSW", metric_type="COSINE")
        with pytest.raises(RuntimeError, match="IndexParams not available"):
            custom.build_index_params(None)

        auto = MilvusIndexConfig(index_type="AUTOINDEX", metric_type="COSINE")
        assert auto.build_index_params(None) == {
            "field_name": "vector",
            "index_type": "AUTOINDEX",
            "metric_type": "COSINE",
            "params": {},
        }

    @pytest.mark.parametrize(
        ("kwargs", "expected_params", "absent_keys"),
        [
            pytest.param(
                {"index_type": "HNSW", "hnsw_ef": 150}, {"ef": 150}, (), id="hnsw"
            ),
            pytest.param(
                {
                    "index_type": "HNSW_SQ",
                    "hnsw_ef": 200,
                    "sq_refine": True,
                    "sq_refine_k": 20,
                },
                {"ef": 200, "refine_k": 20},
                (),
                id="hnsw-sq-with-refine",
            ),
            pytest.param(
                {"index_type": "HNSW_SQ", "hnsw_ef": 200, "sq_refine": False},
                {"ef": 200},
                ("refine_k",),
                id="hnsw-sq-no-refine",
            ),
            pytest.param(
                {"index_type": "IVF_FLAT", "ivf_nprobe": 32},
                {"nprobe": 32},
                (),
                id="ivf-flat",
            ),
            pytest.param({"index_type": "AUTOINDEX"}, None, (), id="autoindex"),
        ],
    )
    def test_build_search_params(self, kwargs, expected_params, absent_keys):
        params = MilvusIndexConfig(**kwargs).build_search_params()
        if expected_params is None:
            assert params == {}
        else:
            for key, value in expected_params.items():
                assert params["params"][key] == value
            for key in absent_keys:
                assert key not in params["params"]

    @pytest.mark.parametrize(
        ("kwargs", "expected_fields"),
        [
            pytest.param(
                {"index_type": "HNSW"},
                {
                    "index_type": "HNSW",
                    "hnsw_m": 16,
                    "sq_type": None,
                    "ivf_nlist": None,
                },
                id="hnsw",
            ),
            pytest.param(
                {"index_type": "HNSW_SQ", "sq_type": "SQ8"},
                {"index_type": "HNSW_SQ", "sq_type": "SQ8", "ivf_nlist": None},
                id="hnsw-sq",
            ),
            pytest.param(
                {"index_type": "IVF_FLAT"},
                {"index_type": "IVF_FLAT", "ivf_nlist": 1024, "sq_type": None},
                id="ivf-flat",
            ),
        ],
    )
    def test_to_dict(self, kwargs, expected_fields):
        d = MilvusIndexConfig(**kwargs).to_dict()
        for key, value in expected_fields.items():
            assert d[key] == value


# ---------------------------------------------------------------------------
# vector_db_storage_cls_kwargs parameter bridge
# ---------------------------------------------------------------------------


class TestMilvusKwargsParameterBridge:
    """Parameter bridging from vector_db_storage_cls_kwargs to MilvusIndexConfig."""

    def test_kwargs_params_override_environment_variables(self):
        with patch.dict(
            "os.environ",
            {"MILVUS_INDEX_TYPE": "IVF_FLAT", "MILVUS_HNSW_M": "16"},
        ):
            storage = _make_storage(
                namespace="test_entities",
                extra_kwargs={"index_type": "HNSW", "hnsw_m": 64},
            )
            assert storage.index_config.index_type == "HNSW"
            assert storage.index_config.hnsw_m == 64

    def test_raganything_framework_integration_scenario(self):
        """Frameworks can pass index config + custom keys via kwargs without env vars."""
        storage = _make_storage(
            namespace="test_entities",
            workspace="raganything_workspace",
            extra_kwargs={
                "cosine_better_than_threshold": 0.2,
                "index_type": "HNSW",
                "metric_type": "L2",
                "hnsw_m": 48,
                "hnsw_ef_construction": 400,
                "hnsw_ef": 200,
                "framework_version": "1.0.0",
                "custom_setting": "value",
            },
        )

        assert storage.index_config.index_type == "HNSW"
        assert storage.index_config.metric_type == "COSINE"  # deprecated, ignored
        assert storage.index_config.hnsw_m == 48
        assert storage.index_config.hnsw_ef_construction == 400
        assert storage.index_config.hnsw_ef == 200
        assert not hasattr(storage.index_config, "framework_version")
        assert not hasattr(storage.index_config, "custom_setting")
        assert storage.workspace == "raganything_workspace"

    def test_all_milvus_parameters_supported_via_kwargs(self):
        """All 11 MilvusIndexConfig parameters can be configured via kwargs."""
        storage = _make_storage(
            namespace="test_entities",
            extra_kwargs={
                "index_type": "HNSW_SQ",
                "metric_type": "IP",
                "hnsw_m": 64,
                "hnsw_ef_construction": 512,
                "hnsw_ef": 256,
                "sq_type": "SQ8",
                "sq_refine": True,
                "sq_refine_type": "FP16",
                "sq_refine_k": 30,
                "ivf_nlist": 4096,
                "ivf_nprobe": 64,
            },
        )

        assert storage.index_config.index_type == "HNSW_SQ"
        assert storage.index_config.metric_type == "COSINE"  # deprecated, ignored
        assert storage.index_config.hnsw_m == 64
        assert storage.index_config.hnsw_ef_construction == 512
        assert storage.index_config.hnsw_ef == 256
        assert storage.index_config.sq_type == "SQ8"
        assert storage.index_config.sq_refine is True
        assert storage.index_config.sq_refine_type == "FP16"
        assert storage.index_config.sq_refine_k == 30
        assert storage.index_config.ivf_nlist == 4096
        assert storage.index_config.ivf_nprobe == 64


# ---------------------------------------------------------------------------
# Metric type policy (COSINE-only + in-place re-index at startup)
# ---------------------------------------------------------------------------

_REBUILD_CALLS = {"release_collection", "drop_index", "create_index", "load_collection"}


class TestMilvusMetricType:
    """Milvus always uses COSINE and re-indexes non-COSINE collections on startup."""

    @staticmethod
    def _metric_storage(metric_kwarg=None):
        extra = {"cosine_better_than_threshold": 0.2}
        if metric_kwarg is not None:
            extra["metric_type"] = metric_kwarg
        return _make_storage(
            namespace="entities",
            workspace="test",
            dim=8,
            embedding_batch_num=10,
            extra_kwargs=extra,
        )

    @pytest.mark.parametrize(
        ("source", "value", "warns"),
        [
            ("kwarg", "L2", True),
            ("env", "ip", True),
            ("kwarg", "cosine", False),
            ("env", None, False),
        ],
    )
    def test_requested_metric_is_ignored_and_warned_once(
        self, monkeypatch, lightrag_caplog, source, value, warns
    ):
        monkeypatch.setattr(milvus_impl, "_IGNORED_METRIC_TYPES_WARNED", set())
        monkeypatch.delenv("MILVUS_METRIC_TYPE", raising=False)
        if source == "env" and value is not None:
            monkeypatch.setenv("MILVUS_METRIC_TYPE", value)
        kwarg = value if source == "kwarg" else None

        storages = [self._metric_storage(kwarg), self._metric_storage(kwarg)]

        assert [s.index_config.metric_type for s in storages] == ["COSINE", "COSINE"]
        warnings = [
            r
            for r in lightrag_caplog.records
            if "no longer supported" in r.getMessage()
        ]
        assert len(warnings) == (1 if warns else 0)

    @pytest.mark.parametrize(
        ("metric", "rebuilt"),
        [("L2", True), ("COSINE", False), (None, False)],
    )
    def test_initialize_reindexes_a_non_cosine_vector_index(self, metric, rebuilt):
        storage = self._metric_storage()
        client = MagicMock()
        client.has_collection.return_value = True
        client.list_indexes.return_value = [] if metric is None else ["vector"]
        client.describe_index.return_value = {"metric_type": metric}
        storage._client = client

        with (
            patch(
                "lightrag.kg.milvus_impl.get_data_init_lock", return_value=AsyncMock()
            ),
            patch.object(storage, "_create_collection_if_not_exist"),
        ):
            asyncio.run(storage.initialize())

        calls = [c[0] for c in client.method_calls if c[0] in _REBUILD_CALLS]
        if rebuilt:
            assert calls == [
                "release_collection",
                "drop_index",
                "create_index",
                "load_collection",
            ]
            add_index = client.prepare_index_params.return_value.add_index
            assert add_index.call_args.kwargs["metric_type"] == "COSINE"
        else:
            assert calls == []
        assert storage._initialized is True

    def test_failed_reindex_fails_initialize(self):
        storage = self._metric_storage()
        client = MagicMock()
        client.has_collection.return_value = True
        client.list_indexes.return_value = ["vector"]
        client.describe_index.return_value = {"metric_type": "L2"}
        client.create_index.side_effect = RuntimeError("index build refused")
        storage._client = client

        with (
            patch(
                "lightrag.kg.milvus_impl.get_data_init_lock", return_value=AsyncMock()
            ),
            patch.object(storage, "_create_collection_if_not_exist"),
            pytest.raises(RuntimeError, match="index build refused"),
        ):
            asyncio.run(storage.initialize())

        client.drop_index.assert_called_once()
        assert storage._initialized is False


# ---------------------------------------------------------------------------
# Schema creation, collection naming, database bootstrap, and index creation
# ---------------------------------------------------------------------------


class TestMilvusIndexCreation:
    """Schema promotion, collection setup/migration dispatch, DB bootstrap, and index creation."""

    @pytest.mark.parametrize(
        ("namespace", "expected_fields"),
        [
            ("entities", {"content", "source_id"}),
            ("relationships", {"content", "source_id"}),
            ("chunks", {"content"}),
        ],
    )
    def test_schema_promotes_core_metadata_fields(self, namespace, expected_fields):
        storage = _make_storage(namespace=namespace)

        fields_by_name = {
            field.name: field for field in storage._create_schema_for_namespace().fields
        }

        assert expected_fields.issubset(fields_by_name)
        for field_name in expected_fields:
            assert (
                _field_max_length(fields_by_name[field_name])
                == MILVUS_MAX_VARCHAR_BYTES
            )

    @pytest.mark.parametrize(
        ("namespace", "old_fields"),
        [
            ("entities", ["entity_name", "file_path"]),
            ("relationships", ["src_id", "tgt_id", "file_path"]),
            ("chunks", ["full_doc_id", "file_path"]),
        ],
    )
    def test_missing_core_metadata_fields_trigger_schema_migration(
        self, namespace, old_fields
    ):
        storage = _make_storage(namespace=namespace)

        with patch.object(storage, "_migrate_collection_schema") as migrate:
            storage._check_schema_compatibility(_collection_info(old_fields))

        migrate.assert_called_once_with()

    def test_migration_sanitizes_varchar_rows_before_insert(self):
        storage = _make_storage(namespace="entities")
        storage.final_namespace = "test_entities"
        storage._client = MagicMock()
        iterator = MagicMock()
        iterator.next.side_effect = [
            [
                {
                    "id": "ent-1",
                    "vector": [0.0] * 128,
                    "content": "x" * (MILVUS_MAX_VARCHAR_BYTES + 10),
                    "source_id": "源" * (MILVUS_MAX_VARCHAR_BYTES // 3 + 10),
                }
            ],
            [],
        ]
        storage._client.query_iterator.return_value = iterator

        with patch.object(storage, "_create_indexes_after_collection"):
            storage._migrate_collection_schema()

        inserted = storage._client.insert.call_args.kwargs["data"][0]
        assert len(inserted["content"].encode("utf-8")) <= MILVUS_MAX_VARCHAR_BYTES
        assert len(inserted["source_id"].encode("utf-8")) <= MILVUS_MAX_VARCHAR_BYTES
        inserted["source_id"].encode("utf-8").decode("utf-8")

    def test_migration_row_normalization_rules(self):
        storage = _make_storage(namespace="entities")

        # Oversized non-primary identity field is truncated rather than aborting migration.
        truncated = storage._normalize_migration_row(
            {"id": "ent-1", "entity_name": "e" * 513, "content": "body"}
        )
        assert len(truncated["entity_name"].encode("utf-8")) == 512

        # Explicit None is backfilled from $meta; extra $meta keys are kept.
        backfilled = storage._normalize_migration_row(
            {
                "id": "ent-1",
                "content": None,
                "$meta": {"content": "body", "extra": "kept"},
            }
        )
        assert backfilled["content"] == "body"
        assert backfilled["extra"] == "kept"

        # Explicit non-None value is never overridden by $meta.
        preserved = storage._normalize_migration_row(
            {
                "id": "ent-1",
                "content": "explicit",
                "$meta": {"content": "stale"},
            }
        )
        assert preserved["content"] == "explicit"

        # Primary key is never truncated (would silently collapse distinct rows).
        with pytest.raises(ValueError, match="primary keys cannot be truncated"):
            storage._normalize_migration_row({"id": "i" * 65, "content": "body"})

    def test_legacy_without_vector_field_creates_fresh_suffixed_collection(self):
        storage = _make_model_storage()
        legacy_info = {
            "fields": [
                {"name": "id", "type": "VarChar", "is_primary": True},
                {
                    "name": "entity_name",
                    "type": "VarChar",
                    "params": {"max_length": 512},
                },
            ]
        }
        client = _wire_collection_state(
            storage,
            {storage.legacy_namespace},
            {storage.legacy_namespace: legacy_info},
        )

        with patch.object(storage, "_create_indexes_after_collection"):
            with patch.object(storage, "_migrate_collection_schema") as migrate:
                storage._create_collection_if_not_exist()

        migrate.assert_not_called()
        client.query_iterator.assert_not_called()
        client.create_collection.assert_called_once()
        assert client.create_collection.call_args.kwargs["collection_name"] == (
            storage.final_namespace
        )
        client.load_collection.assert_called_with(
            storage.final_namespace, timeout=storage._load_timeout
        )

    @pytest.mark.parametrize(
        ("namespace", "workspace", "dim", "model_name", "expected_legacy", "expected_final"),
        [
            pytest.param(
                "chunks",
                "space1",
                3072,
                "text-embedding-3-large",
                "space1_chunks",
                "space1_chunks_text_embedding_3_large_3072d",
                id="with-workspace",
            ),
            pytest.param(
                "entities",
                "",
                2560,
                " qwen3-embedding:4b ",
                "entities",
                "entities_qwen3_embedding_4b_2560d",
                id="without-workspace",
            ),
        ],
    )
    def test_model_suffix_collection_naming(
        self,
        namespace,
        workspace,
        dim,
        model_name,
        expected_legacy,
        expected_final,
    ):
        storage = _make_storage(
            namespace=namespace,
            workspace=workspace,
            embedding_func=_EmbeddingFunc(dim=dim, model_name=model_name),
        )
        assert storage.legacy_namespace == expected_legacy
        assert storage.final_namespace == expected_final

    @pytest.mark.parametrize("model_name", ["   ", 123])
    def test_missing_or_invalid_model_name_keeps_legacy_collection_name(
        self, model_name
    ):
        storage = _make_storage(
            namespace="entities",
            workspace="space1",
            embedding_func=_EmbeddingFunc(model_name=model_name),
        )

        assert storage.model_suffix is None
        assert storage.legacy_namespace == "space1_entities"
        assert storage.final_namespace == "space1_entities"

    def test_creates_suffixed_collection_when_no_collection_exists(self):
        storage = _make_model_storage()
        client = _wire_collection_state(storage, set())

        with patch.object(storage, "_create_indexes_after_collection"):
            storage._create_collection_if_not_exist()

        client.create_collection.assert_called_once()
        assert client.create_collection.call_args.kwargs["collection_name"] == (
            storage.final_namespace
        )
        client.query_iterator.assert_not_called()
        client.load_collection.assert_called_with(
            storage.final_namespace, timeout=storage._load_timeout
        )

    def test_existing_suffixed_collection_is_validated_and_used(self):
        storage = _make_model_storage()
        client = _wire_collection_state(
            storage,
            {storage.final_namespace},
            {
                storage.final_namespace: _collection_info(
                    ["entity_name", "content", "source_id", "file_path"]
                )
            },
        )

        with patch.object(storage, "_migrate_collection_schema") as migrate:
            storage._create_collection_if_not_exist()

        migrate.assert_not_called()
        client.create_collection.assert_not_called()
        client.load_collection.assert_called_with(
            storage.final_namespace, timeout=storage._load_timeout
        )

    def test_legacy_same_dimension_migrates_to_suffixed_collection(self):
        storage = _make_model_storage()
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(
            storage,
            collections,
            {
                storage.legacy_namespace: _collection_info(
                    ["entity_name", "content", "source_id"]
                )
            },
        )

        def migrate_collection(**_kwargs):
            collections.add(storage.final_namespace)

        with patch.object(
            storage, "_migrate_collection_schema", side_effect=migrate_collection
        ) as migrate:
            storage._create_collection_if_not_exist()

        migrate.assert_called_once_with(
            source_collection_name=storage.legacy_namespace,
            target_collection_name=storage.final_namespace,
        )
        client.describe_collection.assert_called_once_with(storage.legacy_namespace)
        client.query_iterator.assert_not_called()
        client.create_collection.assert_not_called()
        client.load_collection.assert_called_with(
            storage.final_namespace, timeout=storage._load_timeout
        )

    def test_no_model_suffix_old_meta_schema_migrates_in_place(self):
        storage = _make_storage(namespace="entities")
        storage.model_suffix = None
        storage.legacy_namespace = storage.final_namespace
        client = _wire_collection_state(
            storage,
            {storage.final_namespace},
            {storage.final_namespace: _collection_info(["entity_name", "file_path"])},
        )

        with patch.object(storage, "_migrate_collection_schema") as migrate:
            storage._create_collection_if_not_exist()

        migrate.assert_called_once_with()
        client.create_collection.assert_not_called()
        client.load_collection.assert_called_with(
            storage.final_namespace, timeout=storage._load_timeout
        )

    def test_legacy_describe_failure_raises_without_creating_suffixed_collection(self):
        storage = _make_model_storage()
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(storage, collections)
        client.describe_collection.side_effect = RuntimeError("milvus unavailable")

        with pytest.raises(RuntimeError, match="milvus unavailable"):
            storage._create_collection_if_not_exist()

        client.describe_collection.assert_called_once_with(storage.legacy_namespace)
        client.query_iterator.assert_not_called()
        client.create_collection.assert_not_called()
        assert storage.legacy_namespace in collections
        assert storage.final_namespace not in collections

    def test_legacy_migration_failure_keeps_legacy_collection(self):
        storage = _make_model_storage()
        legacy_info = _collection_info(["entity_name", "content", "source_id"])
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(
            storage,
            collections,
            {storage.legacy_namespace: legacy_info},
        )

        with patch.object(
            storage,
            "_migrate_collection_schema",
            side_effect=RuntimeError("migration failed"),
        ) as migrate:
            with pytest.raises(RuntimeError, match="migration failed"):
                storage._create_collection_if_not_exist()

        migrate.assert_called_once_with(
            source_collection_name=storage.legacy_namespace,
            target_collection_name=storage.final_namespace,
        )
        client.query_iterator.assert_not_called()
        client.create_collection.assert_not_called()
        assert storage.legacy_namespace in collections
        assert storage.final_namespace not in collections

    def test_migration_insert_batches_use_build_upsert_batches(self):
        storage = _make_model_storage()
        storage._max_upsert_payload_bytes = 1024
        storage._max_upsert_records_per_batch = 2000
        client = _wire_collection_state(storage, {storage.legacy_namespace})
        iterator = MagicMock()
        iterator.next.side_effect = [
            [
                {
                    "id": f"ent-{i}",
                    "vector": [0.0] * 128,
                    "content": "x" * 300,
                }
                for i in range(2000)
            ],
            [],
        ]
        client.query_iterator.return_value = iterator

        with patch.object(storage, "_create_indexes_after_collection"):
            with patch.object(
                storage, "_build_upsert_batches", wraps=storage._build_upsert_batches
            ) as build_batches:
                with patch.object(storage, "_flush_pending_vector_ops") as flush:
                    storage._migrate_collection_schema(
                        source_collection_name=storage.legacy_namespace,
                        target_collection_name=storage.final_namespace,
                    )

        build_batches.assert_called()
        assert client.insert.call_count > 1
        flush.assert_not_called()

    def test_failed_legacy_migration_cleans_temp_and_keeps_legacy_collection(self):
        storage = _make_model_storage()
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(storage, collections)
        iterator = MagicMock()
        iterator.next.side_effect = [
            [{"id": "ent-1", "vector": [0.0] * 128, "content": "body"}],
        ]
        client.query_iterator.return_value = iterator
        client.insert.side_effect = RuntimeError("insert failed")

        with patch.object(storage, "_create_indexes_after_collection"):
            with pytest.raises(RuntimeError, match="Iterator-based migration failed"):
                storage._migrate_collection_schema(
                    source_collection_name=storage.legacy_namespace,
                    target_collection_name=storage.final_namespace,
                )

        assert storage.legacy_namespace in collections
        assert f"{storage.final_namespace}_temp" not in collections
        assert storage.final_namespace not in collections

    def test_vector_index_creation_failure_is_raised(self):
        storage = _make_storage(
            namespace="test_entities", extra_kwargs={"index_type": "HNSW"}
        )
        mock_client = MagicMock()
        mock_index_params = MagicMock()
        storage._client = mock_client
        storage.final_namespace = "test_entities"

        with patch.object(storage, "_get_index_params", return_value=mock_index_params):
            with patch.object(
                storage.index_config,
                "build_index_params",
                return_value=mock_index_params,
            ):
                mock_client.create_index.side_effect = Exception(
                    "Index creation failed"
                )
                with pytest.raises(Exception, match="Index creation failed"):
                    storage._create_indexes_after_collection()

    def test_scalar_index_creation_failure_is_logged_not_raised(self, lightrag_caplog):
        storage = _make_storage(
            namespace="test_entities", extra_kwargs={"index_type": "AUTOINDEX"}
        )
        mock_client = MagicMock()
        mock_index_params = MagicMock()
        storage._client = mock_client
        storage.final_namespace = "test_entities"

        calls = {"n": 0}

        def fail_after_vector(*_args, **_kwargs):
            calls["n"] += 1
            if calls["n"] >= 2:
                raise Exception("Scalar index creation failed")

        with patch.object(storage, "_get_index_params", return_value=mock_index_params):
            mock_client.create_index.side_effect = fail_after_vector
            storage._create_indexes_after_collection()

        assert calls["n"] >= 2
        assert any(
            "Scalar index creation failed" in r.getMessage()
            for r in lightrag_caplog.records
        )

    def test_create_indexes_uses_compatibility_helper(self):
        storage = _make_storage(
            namespace="test_entities", extra_kwargs={"index_type": "HNSW"}
        )
        storage._client = MagicMock()
        storage.final_namespace = "test_entities"

        with patch.object(
            storage, "_get_index_params", return_value=MagicMock()
        ) as mock_get_index_params:
            storage._create_indexes_after_collection()
            assert mock_get_index_params.call_count >= 1

    @pytest.mark.parametrize(
        ("index_type", "expect_probe"),
        [("HNSW", False), ("HNSW_SQ", True)],
    )
    def test_version_probing_gated_on_index_type(self, index_type, expect_probe):
        storage = _make_storage(
            namespace="test_entities", extra_kwargs={"index_type": index_type}
        )
        mock_client = MagicMock()
        mock_client.get_server_version.return_value = "2.6.9"
        storage._client = mock_client

        with patch(
            "lightrag.kg.milvus_impl.get_data_init_lock", return_value=AsyncMock()
        ):
            with patch.object(storage, "_create_collection_if_not_exist"):
                asyncio.run(storage.initialize())

        if expect_probe:
            mock_client.get_server_version.assert_called_once()
        else:
            mock_client.get_server_version.assert_not_called()

    @pytest.mark.parametrize(
        ("existing_dbs", "expect_create"),
        [
            pytest.param(["default"], True, id="creates-missing-database"),
            pytest.param(["default", "lightrag"], False, id="uses-existing-database"),
        ],
    )
    def test_initialize_database_bootstrap(self, existing_dbs, expect_create):
        storage = _make_storage(
            namespace="test_entities", workspace="space1", working_dir="/tmp/lightrag"
        )
        bootstrap_client = MagicMock()
        bootstrap_client.list_databases.return_value = existing_dbs

        with patch.dict(
            "os.environ",
            {"MILVUS_URI": "http://milvus:19530", "MILVUS_DB_NAME": "lightrag"},
            clear=False,
        ):
            with patch(
                "lightrag.kg.milvus_impl.MilvusClient", return_value=bootstrap_client
            ) as mock_client_cls:
                with patch(
                    "lightrag.kg.milvus_impl.get_data_init_lock",
                    return_value=AsyncMock(),
                ):
                    with patch.object(storage, "_create_collection_if_not_exist"):
                        asyncio.run(storage.initialize())

        mock_client_cls.assert_called_once_with(
            uri="http://milvus:19530", user=None, password=None, token=None
        )
        bootstrap_client.list_databases.assert_called_once_with()
        if expect_create:
            bootstrap_client.create_database.assert_called_once_with("lightrag")
        else:
            bootstrap_client.create_database.assert_not_called()
        bootstrap_client.use_database.assert_called_once_with("lightrag")

    @pytest.mark.parametrize(
        "message",
        ["Database already exist[database=lightrag]", "Database Already Exists"],
        ids=["exact-milvus-message", "case-insensitive"],
    )
    def test_create_database_race_already_exists_is_swallowed(self, message):
        storage = _make_model_storage()
        bootstrap_client = MagicMock()
        bootstrap_client.list_databases.return_value = ["default"]
        bootstrap_client.create_database.side_effect = MilvusException(message=message)

        with patch.dict("os.environ", {"MILVUS_DB_NAME": "lightrag"}, clear=False):
            with patch(
                "lightrag.kg.milvus_impl.MilvusClient", return_value=bootstrap_client
            ):
                client = storage._create_milvus_client()

        bootstrap_client.create_database.assert_called_once_with("lightrag")
        bootstrap_client.use_database.assert_called_once_with("lightrag")
        assert client is bootstrap_client

    @pytest.mark.parametrize(
        ("exc", "exc_type", "match"),
        [
            pytest.param(
                MilvusException(message="permission denied"),
                MilvusException,
                "permission denied",
                id="unrelated-milvus-exception",
            ),
            pytest.param(
                RuntimeError("connection reset"),
                RuntimeError,
                "connection reset",
                id="non-milvus-exception",
            ),
        ],
    )
    def test_create_database_non_race_errors_are_reraised(self, exc, exc_type, match):
        storage = _make_model_storage()
        bootstrap_client = MagicMock()
        bootstrap_client.list_databases.return_value = ["default"]
        bootstrap_client.create_database.side_effect = exc

        with patch.dict("os.environ", {"MILVUS_DB_NAME": "lightrag"}, clear=False):
            with patch(
                "lightrag.kg.milvus_impl.MilvusClient", return_value=bootstrap_client
            ):
                with pytest.raises(exc_type, match=match):
                    storage._create_milvus_client()

        bootstrap_client.use_database.assert_not_called()

    def test_existing_collection_missing_vector_index_is_repaired(self):
        storage = _make_storage(
            namespace="entities", workspace="space1", working_dir="/tmp/lightrag"
        )
        storage.final_namespace = "space1_entities"
        storage._client = MagicMock()
        storage._client.has_collection.return_value = True

        load_error = RuntimeError(
            "there is no vector index on field: [vector], please create index firstly"
        )

        with patch.object(storage._client, "describe_collection", return_value={}):
            with patch.object(storage, "_validate_collection_compatibility"):
                with patch.object(
                    storage,
                    "_ensure_collection_loaded",
                    side_effect=[load_error, None],
                ) as mock_load:
                    with patch.object(
                        storage, "_repair_missing_vector_index"
                    ) as mock_repair:
                        storage._create_collection_if_not_exist()

        assert mock_load.call_count == 2
        mock_repair.assert_called_once_with()

    def test_existing_collection_index_repair_failure_has_precise_error(self):
        storage = _make_storage(
            namespace="entities", workspace="space1", working_dir="/tmp/lightrag"
        )
        storage.final_namespace = "space1_entities"
        storage._client = MagicMock()
        storage._client.has_collection.return_value = True

        load_error = RuntimeError(
            "there is no vector index on field: [vector], please create index firstly"
        )

        with patch.object(storage._client, "describe_collection", return_value={}):
            with patch.object(storage, "_validate_collection_compatibility"):
                with patch.object(
                    storage, "_ensure_collection_loaded", side_effect=load_error
                ):
                    with patch.object(
                        storage,
                        "_repair_missing_vector_index",
                        side_effect=RuntimeError("create index failed"),
                    ):
                        with pytest.raises(
                            RuntimeError,
                            match="Index repair failed for collection 'space1_entities'",
                        ):
                            storage._create_collection_if_not_exist()

    def test_existing_collection_non_index_validation_failure_still_raises(self):
        storage = _make_storage(
            namespace="entities", workspace="space1", working_dir="/tmp/lightrag"
        )
        storage.final_namespace = "space1_entities"
        storage._client = MagicMock()
        storage._client.has_collection.return_value = True

        with patch.object(storage._client, "describe_collection", return_value={}):
            with patch.object(
                storage,
                "_validate_collection_compatibility",
                side_effect=RuntimeError("dimension mismatch"),
            ):
                with pytest.raises(
                    RuntimeError,
                    match="Collection validation failed for 'space1_entities'",
                ):
                    storage._create_collection_if_not_exist()
