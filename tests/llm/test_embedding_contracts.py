"""Offline contracts for asymmetric embeddings, batched KG search embeddings,
vector dimension mismatch guards (#3978), and no-model-suffix startup safety.
"""

from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from lightrag.api import config as api_config
from lightrag.base import QueryParam
from lightrag.exceptions import VectorSpaceMismatchError
from lightrag.kg.postgres_impl import PGVectorStorage
from lightrag.kg.qdrant_impl import QdrantVectorDBStorage
from lightrag.operate import _perform_kg_search
from lightrag.utils import EmbeddingFunc, wrap_embedding_func_with_attrs
from tests._pg_legacy_migration import answer_legacy_migration

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# 1. Asymmetric embedding decorator, EmbeddingFunc, config, and providers
# ---------------------------------------------------------------------------


def test_wrap_embedding_func_supports_asymmetric_detection():
    """Auto-detects ``supports_asymmetric`` per function and honors explicit override."""

    @wrap_embedding_func_with_attrs(embedding_dim=4, max_token_size=64)
    async def context_embed(texts, context="document"):
        return np.zeros((len(texts), 4), dtype=np.float32)

    @wrap_embedding_func_with_attrs(embedding_dim=4, max_token_size=64)
    async def legacy_embed(texts):
        return np.zeros((len(texts), 4), dtype=np.float32)

    @wrap_embedding_func_with_attrs(
        embedding_dim=4, max_token_size=64, supports_asymmetric=False
    )
    async def forced_off(texts, context="document"):
        return np.zeros((len(texts), 4), dtype=np.float32)

    shared_decorator = wrap_embedding_func_with_attrs(
        embedding_dim=4, max_token_size=64
    )

    @shared_decorator
    async def reused_legacy(texts):
        return np.zeros((len(texts), 4), dtype=np.float32)

    @shared_decorator
    async def reused_aware(texts, context="document"):
        return np.zeros((len(texts), 4), dtype=np.float32)

    assert context_embed.supports_asymmetric is True
    assert legacy_embed.supports_asymmetric is False
    assert forced_off.supports_asymmetric is False
    assert reused_legacy.supports_asymmetric is False
    assert reused_aware.supports_asymmetric is True


@pytest.mark.asyncio
async def test_embedding_func_context_forwarding():
    """Strips ``context`` for legacy functions and forwards it when supported."""
    legacy_seen: list[dict] = []

    async def legacy_embed(texts):
        legacy_seen.append({"texts": texts})
        return np.zeros((len(texts), 4), dtype=np.float32)

    legacy_func = EmbeddingFunc(
        embedding_dim=4, max_token_size=64, supports_asymmetric=False, func=legacy_embed
    )
    out = await legacy_func(["a", "b"], context="query")
    assert out.shape == (2, 4)
    assert legacy_seen == [{"texts": ["a", "b"]}]

    aware_seen: list[str] = []

    async def aware_embed(texts, context="document"):
        aware_seen.append(context)
        return np.zeros((len(texts), 4), dtype=np.float32)

    aware_func = EmbeddingFunc(
        embedding_dim=4, max_token_size=64, supports_asymmetric=True, func=aware_embed
    )
    await aware_func(["a"], context="query")
    await aware_func(["b"], context="document")
    assert aware_seen == ["query", "document"]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        pytest.param(
            dict(
                binding="ollama",
                embedding_asymmetric=False,
                embedding_asymmetric_configured=False,
                query_prefix="search_query: ",
                query_prefix_configured=True,
                document_prefix=None,
                document_prefix_configured=False,
            ),
            False,
            id="unset-toggle-with-prefix",
        ),
        pytest.param(
            dict(
                binding="ollama",
                embedding_asymmetric=False,
                embedding_asymmetric_configured=True,
                query_prefix="search_query: ",
                query_prefix_configured=True,
                document_prefix=None,
                document_prefix_configured=False,
            ),
            False,
            id="explicit-false-with-prefix",
        ),
        pytest.param(
            dict(
                binding="jina",
                embedding_asymmetric=True,
                embedding_asymmetric_configured=True,
                query_prefix=None,
                query_prefix_configured=False,
                document_prefix=None,
                document_prefix_configured=False,
            ),
            True,
            id="provider-level-jina",
        ),
        pytest.param(
            dict(
                binding="gemini",
                embedding_asymmetric=True,
                embedding_asymmetric_configured=True,
                query_prefix=None,
                query_prefix_configured=False,
                document_prefix=None,
                document_prefix_configured=False,
            ),
            True,
            id="provider-level-gemini",
        ),
        pytest.param(
            dict(
                binding="voyageai",
                embedding_asymmetric=True,
                embedding_asymmetric_configured=True,
                query_prefix=None,
                query_prefix_configured=False,
                document_prefix=None,
                document_prefix_configured=False,
            ),
            True,
            id="provider-level-voyageai",
        ),
        pytest.param(
            dict(
                binding="jina",
                embedding_asymmetric=True,
                embedding_asymmetric_configured=True,
                query_prefix="search_query: ",
                query_prefix_configured=True,
                document_prefix=None,
                document_prefix_configured=False,
            ),
            True,
            id="provider-level-ignores-prefixes",
        ),
        pytest.param(
            dict(
                binding="ollama",
                embedding_asymmetric=True,
                embedding_asymmetric_configured=True,
                query_prefix="search_query: ",
                query_prefix_configured=True,
                document_prefix="",
                document_prefix_configured=True,
            ),
            True,
            id="prefix-binding-accepts-one-sentinel-side",
        ),
    ],
)
def test_resolve_asymmetric_embedding_opt_in_valid(kwargs, expected):
    assert api_config.resolve_asymmetric_embedding_opt_in(**kwargs) is expected


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param(
            dict(
                binding="ollama",
                embedding_asymmetric=True,
                embedding_asymmetric_configured=True,
                query_prefix="search_query: ",
                query_prefix_configured=True,
                document_prefix=None,
                document_prefix_configured=False,
            ),
            "requires both",
            id="missing-document-prefix",
        ),
        pytest.param(
            dict(
                binding="ollama",
                embedding_asymmetric=True,
                embedding_asymmetric_configured=True,
                query_prefix="",
                query_prefix_configured=True,
                document_prefix="",
                document_prefix_configured=True,
            ),
            "At least one",
            id="both-sides-no-prefix",
        ),
    ],
)
def test_resolve_asymmetric_embedding_opt_in_invalid(kwargs, match):
    with pytest.raises(ValueError, match=match):
        api_config.resolve_asymmetric_embedding_opt_in(**kwargs)


def test_get_embedding_prefix_config_sentinel_and_empty_error(monkeypatch):
    monkeypatch.setenv("EMBEDDING_DOCUMENT_PREFIX", api_config.NO_PREFIX_SENTINEL)
    assert api_config.get_embedding_prefix_config("EMBEDDING_DOCUMENT_PREFIX") == (
        "",
        True,
    )

    monkeypatch.setenv("EMBEDDING_DOCUMENT_PREFIX", "")
    with pytest.raises(ValueError, match=api_config.NO_PREFIX_SENTINEL):
        api_config.get_embedding_prefix_config("EMBEDDING_DOCUMENT_PREFIX")


def _fake_jina_response(num: int, dim: int = 4) -> list[dict]:
    arr = np.zeros((num, dim), dtype=np.float32)
    return [
        {"embedding": base64.b64encode(arr[i].tobytes()).decode()} for i in range(num)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("texts", "context", "explicit_task", "expected_task"),
    [
        pytest.param(["q1"], "query", None, "retrieval.query", id="query-default"),
        pytest.param(
            ["d1", "d2"], "document", None, "retrieval.passage", id="document-default"
        ),
        pytest.param(
            ["x"], "query", "text-matching", "text-matching", id="explicit-override"
        ),
    ],
)
async def test_jina_embed_task_selection(
    monkeypatch, texts, context, explicit_task, expected_task
):
    monkeypatch.setenv("JINA_API_KEY", "fake")
    from lightrag.llm import jina as jina_mod

    captured: list[dict] = []

    async def fake_fetch(url, headers, data, timeout=None):
        captured.append(data)
        return _fake_jina_response(len(data["input"]))

    call_kwargs = {"texts": texts, "context": context}
    if explicit_task is not None:
        call_kwargs["task"] = explicit_task

    with patch.object(jina_mod, "fetch_data", side_effect=fake_fetch):
        await jina_mod.jina_embed.func(**call_kwargs)
    assert captured[0]["task"] == expected_task


@pytest.fixture
def gemini_client_cache_cleared():
    """gemini.py caches its Client via lru_cache; clear it between tests."""
    pytest.importorskip("google.genai")
    from lightrag.llm import gemini as gemini_mod

    gemini_mod._get_gemini_client.cache_clear()
    yield
    gemini_mod._get_gemini_client.cache_clear()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("texts", "context", "task_type", "expected"),
    [
        pytest.param(["q"], "query", None, "RETRIEVAL_QUERY", id="query-default"),
        pytest.param(
            ["d"], "document", None, "RETRIEVAL_DOCUMENT", id="document-default"
        ),
        pytest.param(
            ["x"], "query", "CLASSIFICATION", "CLASSIFICATION", id="explicit-override"
        ),
    ],
)
async def test_gemini_embed_task_type_selection(
    gemini_client_cache_cleared, texts, context, task_type, expected
):
    pytest.importorskip("google.genai")
    from lightrag.llm import gemini as gemini_mod

    captured: list[dict] = []

    async def fake_embed_content(*, model, contents, config):
        captured.append({"task_type": getattr(config, "task_type", None)})
        resp = MagicMock()
        resp.embeddings = [MagicMock(values=[0.1] * 4) for _ in contents]
        return resp

    fake_client = MagicMock()
    fake_client.aio.models.embed_content = fake_embed_content

    with patch.object(gemini_mod.genai, "Client", return_value=fake_client):
        await gemini_mod.gemini_embed.func(
            texts=texts, api_key="fake", context=context, task_type=task_type
        )
    assert captured[0]["task_type"] == expected


# ---------------------------------------------------------------------------
# 2. Batch embedding pre-computation in _perform_kg_search()
# ---------------------------------------------------------------------------


def _make_mock_embedding_func(dim=1536):
    async def _embed(texts, **kwargs):
        return np.array(
            [np.full(dim, i + 1, dtype=np.float32) for i in range(len(texts))]
        )

    return AsyncMock(side_effect=_embed)


def _make_kg_search_doubles(embed_func):
    text_chunks_db = MagicMock()
    text_chunks_db.embedding_func = embed_func
    text_chunks_db.global_config = {"kg_chunk_pick_method": "VECTOR"}

    def _vdb():
        mock = AsyncMock()
        mock.query = AsyncMock(return_value=[])
        mock.cosine_better_than_threshold = 0.2
        return mock

    return text_chunks_db, _vdb(), _vdb(), AsyncMock()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected_batch", "check_vdb_forwarding"),
    [
        pytest.param(
            "hybrid",
            ["test query", "entity keywords", "theme keywords"],
            True,
            id="hybrid",
        ),
        pytest.param(
            "local",
            ["test query", "entity keywords"],
            False,
            id="local",
        ),
        pytest.param(
            "global",
            ["test query", "theme keywords"],
            False,
            id="global",
        ),
    ],
)
async def test_kg_search_batches_embeddings_by_mode(
    mode, expected_batch, check_vdb_forwarding
):
    embed_func = _make_mock_embedding_func()
    text_chunks_db, entities_vdb, relationships_vdb, knowledge_graph = (
        _make_kg_search_doubles(embed_func)
    )

    await _perform_kg_search(
        query="test query",
        ll_keywords="entity keywords",
        hl_keywords="theme keywords",
        knowledge_graph_inst=knowledge_graph,
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        text_chunks_db=text_chunks_db,
        query_param=QueryParam(mode=mode, top_k=5),
    )

    assert embed_func.call_count == 1
    assert embed_func.call_args[0][0] == expected_batch

    if check_vdb_forwarding:
        ll_embedding = entities_vdb.query.call_args.kwargs.get("query_embedding")
        assert ll_embedding is not None and np.all(ll_embedding == 2.0)
        hl_embedding = relationships_vdb.query.call_args.kwargs.get("query_embedding")
        assert hl_embedding is not None and np.all(hl_embedding == 3.0)


@pytest.mark.asyncio
async def test_kg_search_embedding_failure_falls_back_gracefully():
    embed_func = AsyncMock(side_effect=RuntimeError("API error"))
    text_chunks_db, entities_vdb, relationships_vdb, knowledge_graph = (
        _make_kg_search_doubles(embed_func)
    )

    await _perform_kg_search(
        query="test query",
        ll_keywords="entity keywords",
        hl_keywords="theme keywords",
        knowledge_graph_inst=knowledge_graph,
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        text_chunks_db=text_chunks_db,
        query_param=QueryParam(mode="hybrid", top_k=5),
    )

    assert entities_vdb.query.call_args.kwargs.get("query_embedding") is None
    assert relationships_vdb.query.call_args.kwargs.get("query_embedding") is None


# ---------------------------------------------------------------------------
# 3. Vector dimension mismatch & no-model-suffix startup safety (#3978)
# ---------------------------------------------------------------------------


class TestQdrantMigrationSafety:
    """Qdrant dimension mismatch refusal, matching migration, and suffix cleanup safety."""

    def test_qdrant_dimension_mismatch_raises_error(self):
        from qdrant_client import models

        client = MagicMock()
        legacy_collection_info = MagicMock()
        legacy_collection_info.config.params.vectors.size = 1536
        legacy_collection_info.payload_schema = {}

        client.collection_exists.side_effect = (
            lambda name: name == "lightrag_vdb_chunks"
        )
        client.get_collection.return_value = legacy_collection_info
        client.count.return_value.count = 100
        client.scroll.return_value = ([], None)

        with patch(
            "lightrag.kg.qdrant_impl._find_legacy_collection",
            return_value="lightrag_vdb_chunks",
        ):
            with pytest.raises(VectorSpaceMismatchError) as exc_info:
                QdrantVectorDBStorage.setup_collection(
                    client,
                    "lightrag_chunks_model_3072d",
                    namespace="chunks",
                    workspace="test",
                    vectors_config=models.VectorParams(
                        size=3072, distance=models.Distance.COSINE
                    ),
                    hnsw_config=models.HnswConfigDiff(payload_m=16, m=0),
                    model_suffix="model_3072d",
                )

        assert "3072" in str(exc_info.value) or "1536" in str(exc_info.value)
        client.create_collection.assert_not_called()
        client.upsert.assert_not_called()

    def test_qdrant_dimension_match_proceed_migration(self):
        from qdrant_client import models

        client = MagicMock()
        legacy_collection_info = MagicMock()
        legacy_collection_info.config.params.vectors.size = 1536

        client.collection_exists.side_effect = lambda name: name == "lightrag_chunks"
        client.get_collection.return_value = legacy_collection_info

        migration_done = {"value": False}

        def upsert_side_effect(*_args, **_kwargs):
            migration_done["value"] = True
            return MagicMock()

        client.upsert.side_effect = upsert_side_effect

        def count_side_effect(collection_name, **_kwargs):
            result = MagicMock()
            if collection_name == "lightrag_chunks":
                result.count = 1
            elif collection_name == "lightrag_chunks_model_1536d":
                result.count = 1 if migration_done["value"] else 0
            else:
                result.count = 0
            return result

        client.count.side_effect = count_side_effect

        sample_point = MagicMock()
        sample_point.id = "test_id"
        sample_point.vector = [0.1] * 1536
        sample_point.payload = {"id": "test"}
        client.scroll.return_value = ([sample_point], None)

        with patch(
            "lightrag.kg.qdrant_impl._find_legacy_collection",
            return_value="lightrag_chunks",
        ):
            QdrantVectorDBStorage.setup_collection(
                client,
                "lightrag_chunks_model_1536d",
                namespace="chunks",
                workspace="test",
                vectors_config=models.VectorParams(
                    size=1536, distance=models.Distance.COSINE
                ),
                hnsw_config=models.HnswConfigDiff(payload_m=16, m=0),
                model_suffix="model_1536d",
            )

        client.create_collection.assert_called_once()
        client.scroll.assert_called()
        client.upsert.assert_called()

    @pytest.mark.parametrize(
        ("collection_name", "model_suffix", "expect_delete"),
        [
            pytest.param("lightrag_vdb_chunks", "", False, id="no-suffix-safe"),
            pytest.param(
                "lightrag_vdb_chunks_ada_002_1536d",
                "ada_002_1536d",
                True,
                id="with-suffix-deletes-legacy",
            ),
        ],
    )
    def test_qdrant_model_suffix_cleanup_safety(
        self, collection_name, model_suffix, expect_delete
    ):
        from qdrant_client import models

        client = MagicMock()
        legacy_collection = "lightrag_vdb_chunks"
        client.collection_exists.side_effect = lambda name: name in (
            collection_name,
            legacy_collection,
        )
        client.count.return_value.count = 0

        ctx = (
            patch(
                "lightrag.kg.qdrant_impl._find_legacy_collection",
                return_value=legacy_collection,
            )
            if not model_suffix
            else patch.object(client, "delete_collection", wraps=client.delete_collection)
        )
        with ctx:
            QdrantVectorDBStorage.setup_collection(
                client,
                collection_name,
                namespace="chunks",
                workspace="_",
                vectors_config=models.VectorParams(
                    size=1536, distance=models.Distance.COSINE
                ),
                hnsw_config=models.HnswConfigDiff(payload_m=16, m=0),
                model_suffix=model_suffix,
            )

        if expect_delete:
            client.delete_collection.assert_called_once_with(
                collection_name=legacy_collection
            )
        else:
            client.delete_collection.assert_not_called()
            assert client.collection_exists.call_count >= 1


class TestPostgresMigrationSafety:
    """PostgreSQL dimension mismatch refusal, matching migration, and suffix cleanup safety."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "sample_vector",
        [
            pytest.param([0.1] * 1536, id="raw-list"),
            pytest.param(json.dumps([0.1] * 1536), id="json-string-sampling"),
        ],
    )
    async def test_postgres_dimension_mismatch_raises_error(self, sample_vector):
        db = AsyncMock()
        db.vector_index_type = None
        db.check_table_exists = AsyncMock(
            side_effect=lambda name: name == "LIGHTRAG_DOC_CHUNKS"
        )

        async def query_side_effect(query, params, **_kwargs):
            if "information_schema.tables" in query:
                return {"exists": params[0] == "LIGHTRAG_DOC_CHUNKS"}
            if "COUNT(*)" in query:
                return {"count": 100}
            if "SELECT content_vector FROM" in query:
                return {"content_vector": sample_vector}
            return {}

        db.query.side_effect = query_side_effect
        db.execute = AsyncMock()
        db._create_vector_index = AsyncMock()

        with pytest.raises(VectorSpaceMismatchError) as exc_info:
            await PGVectorStorage.setup_table(
                db,
                "LIGHTRAG_DOC_CHUNKS_model_3072d",
                legacy_table_name="LIGHTRAG_DOC_CHUNKS",
                base_table="LIGHTRAG_DOC_CHUNKS",
                embedding_dim=3072,
                workspace="test",
            )

        assert "3072" in str(exc_info.value) or "1536" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_postgres_dimension_match_proceed_migration(self):
        db = AsyncMock()
        db.vector_index_type = None
        migration_done = {"value": False}
        mock_records = [
            {"id": "test1", "content_vector": [0.1] * 1536, "workspace": "test"},
            {"id": "test2", "content_vector": [0.2] * 1536, "workspace": "test"},
        ]

        db.check_table_exists = AsyncMock(
            side_effect=lambda name: name == "LIGHTRAG_DOC_CHUNKS"
        )

        async def query_side_effect(query, params, **kwargs):
            multirows = kwargs.get("multirows", False)
            query_upper = query.upper()
            if "information_schema.tables" in query:
                return {"exists": params[0] == "LIGHTRAG_DOC_CHUNKS"}
            if "COUNT(*)" in query_upper:
                if "LIGHTRAG_DOC_CHUNKS_MODEL_1536D" in query_upper:
                    return {
                        "count": len(mock_records) if migration_done["value"] else 0
                    }
                return {"count": len(mock_records)}
            if "PG_ATTRIBUTE" in query_upper:
                return {"vector_dim": 1536}
            if "SELECT" in query_upper and "FROM" in query_upper and multirows:
                if "id >" in query.lower():
                    last_id = params[1] if len(params) > 1 else None
                    for i, rec in enumerate(mock_records):
                        if rec["id"] == last_id:
                            return mock_records[i + 1 :]
                    return []
                return mock_records
            return {}

        migration_executed = []

        def record_migration(_sql, _params):
            migration_executed.append(True)
            migration_done["value"] = True

        db.query = AsyncMock(
            side_effect=answer_legacy_migration(query_side_effect, record_migration)
        )
        db.execute = AsyncMock()
        db._create_vector_index = AsyncMock()

        await PGVectorStorage.setup_table(
            db,
            "LIGHTRAG_DOC_CHUNKS_model_1536d",
            legacy_table_name="LIGHTRAG_DOC_CHUNKS",
            base_table="LIGHTRAG_DOC_CHUNKS",
            embedding_dim=1536,
            workspace="test",
        )

        assert len(migration_executed) > 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("table_name", "expect_drop"),
        [
            pytest.param("LIGHTRAG_VDB_CHUNKS", False, id="no-suffix-safe"),
            pytest.param(
                "LIGHTRAG_VDB_CHUNKS_ADA_002_1536D",
                True,
                id="with-suffix-drops-legacy",
            ),
        ],
    )
    async def test_postgres_model_suffix_cleanup_safety(self, table_name, expect_drop):
        db = AsyncMock()
        db.vector_index_type = None
        db.check_table_exists = AsyncMock(return_value=True)
        db.query.return_value = {"count": 0}
        db._create_vector_index.return_value = None

        legacy_table_name = "LIGHTRAG_VDB_CHUNKS"
        await PGVectorStorage.setup_table(
            db,
            table_name,
            workspace="test_workspace",
            embedding_dim=1536,
            legacy_table_name=legacy_table_name,
            base_table="LIGHTRAG_VDB_CHUNKS",
        )

        drop_calls = [
            call
            for call in db.execute.call_args_list
            if call[0][0] and "DROP TABLE" in call[0][0]
        ]
        if expect_drop:
            assert len(drop_calls) == 1
            assert legacy_table_name in drop_calls[0][0][0]
        else:
            assert len(drop_calls) == 0
