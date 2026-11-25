"""Offline unit tests for OpenSearchGraphStorage: node/edge CRUD, canonical edge
migration, PPL graphlookup, read contracts, placeholder safety, BFS tie-breaking
and subgraph truncation, batch edge degrees, edge payload merging, self-loops,
and delete result handling."""

from __future__ import annotations

import contextlib
import sys
from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip(
    "opensearchpy",
    reason="opensearchpy is required for OpenSearch storage tests",
)

from opensearchpy.exceptions import (  # type: ignore  # noqa: E402
    ConflictError,
    NotFoundError,
    OpenSearchException,
    TransportError,
)

import lightrag.kg.opensearch_impl as opensearch_module  # noqa: E402
from lightrag.kg.opensearch_impl import (  # noqa: E402
    _EDGE_ID_CANONICAL_META_FLAG,
    _GRAPH_DEGREE_RANK_MAX_CANDIDATES,
    ClientManager,
    OpenSearchGraphStorage,
    _canonical_edge_id,
    _merge_edge_payloads,
)

from ._helpers import (  # noqa: E402
    LazyDropRecreateMixin,
    MockEmbeddingFunc,
    ReadinessDemotionMixin,
    RejectingEmbedFunc,
    default_global_config,
    make_client,
    make_strict_client,
    mget_by_ids_side_effect,
    missing_index_error,
    node_mget_side_effect,
    patch_opensearch_locks,
    transient_error,
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


def _make_graph(
    global_config, embed_func, workspace: str = "test"
) -> OpenSearchGraphStorage:
    return OpenSearchGraphStorage(
        namespace="chunk_entity_relation",
        global_config=global_config,
        embedding_func=embed_func,
        workspace=workspace,
    )


# ---------------------------------------------------------------------------
# Graph Storage CRUD, canonical edge migration, labels, drop, and construction
# ---------------------------------------------------------------------------


class TestGraphStorage(LazyDropRecreateMixin, ReadinessDemotionMixin):
    """Tests for OpenSearchGraphStorage node/edge CRUD, batch ops, migration, and labels."""

    readiness_attr = "_indices_ready"
    create_method_name = "_create_indices_if_not_exist"
    expected_drop_error_substr = "edges drop failed"

    def _make(self, global_config, embed_func, workspace: str = "test"):
        return _make_graph(global_config, embed_func, workspace)

    def _configure_drop_failure(self, mock_client) -> None:
        mock_client.mget = AsyncMock(side_effect=node_mget_side_effect(set()))
        mock_client.indices.delete = AsyncMock(
            side_effect=[None, OpenSearchException("edges drop failed")]
        )

    async def _write_to_recreate(self, storage) -> None:
        await storage.upsert_edge("A", "B", {"weight": "1.0"})

    async def _invoke_missing_read(self, storage):
        return await storage.get_node("A")

    @pytest.mark.asyncio
    async def test_has_edge(self, global_config, embed_func, mock_client):
        mock_client.exists = AsyncMock(return_value=True)
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.has_edge("A", "B") is True
            mock_client.exists.assert_awaited_once_with(
                index=s._edges_index, id=_canonical_edge_id("A", "B")
            )
            mock_client.mget.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_node(self, global_config, embed_func, mock_client):
        mock_client.mget = AsyncMock(
            return_value={
                "docs": [
                    {
                        "_id": "Alice",
                        "found": True,
                        "_source": {
                            "entity_type": "person",
                            "description": "A researcher",
                        },
                    }
                ]
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            node = await s.get_node("Alice")
            assert node["entity_type"] == "person"
            assert node["_id"] == "Alice"
            mock_client.mget.assert_awaited_once_with(
                index=s._nodes_index, body={"ids": ["Alice"]}
            )

    @pytest.mark.asyncio
    async def test_get_node_not_found(self, global_config, embed_func, mock_client):
        mock_client.mget = AsyncMock(
            return_value={"docs": [{"_id": "Nobody", "found": False}]}
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.get_node("Nobody") is None
            mock_client.get.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_edge(self, global_config, embed_func, mock_client):
        canonical = _canonical_edge_id("A", "B")
        mock_client.mget = AsyncMock(
            return_value={
                "docs": [
                    {
                        "_id": canonical,
                        "found": True,
                        "_source": {
                            "source_node_id": "A",
                            "target_node_id": "B",
                            "weight": 1.0,
                        },
                    },
                ]
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            edge = await s.get_edge("A", "B")
            assert edge is not None
            assert edge["weight"] == 1.0
            assert edge["_id"] == canonical
            mock_client.mget.assert_awaited_once_with(
                index=s._edges_index, body={"ids": [canonical]}
            )

    @pytest.mark.asyncio
    async def test_get_node_edges(self, global_config, embed_func, mock_client):
        mock_client.search = AsyncMock(
            return_value={
                "hits": {
                    "hits": [
                        {
                            "_id": "e1",
                            "_source": {"source_node_id": "A", "target_node_id": "B"},
                            "sort": [1],
                        },
                        {
                            "_id": "e2",
                            "_source": {"source_node_id": "C", "target_node_id": "A"},
                            "sort": [2],
                        },
                    ],
                    "total": {"value": 2},
                },
            }
        )
        mock_client.exists = AsyncMock(return_value=True)
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            edges = await s.get_node_edges("A")
            assert edges == [("A", "B"), ("C", "A")]

    @pytest.mark.asyncio
    async def test_node_degrees_batch(self, global_config, embed_func, mock_client):
        mock_client.search = AsyncMock(
            return_value={
                "hits": {"hits": [], "total": {"value": 0}},
                "aggregations": {
                    "source_degrees": {
                        "ids": {"buckets": [{"key": "A", "doc_count": 2}]}
                    },
                    "target_degrees": {
                        "ids": {
                            "buckets": [
                                {"key": "A", "doc_count": 1},
                                {"key": "B", "doc_count": 3},
                            ]
                        }
                    },
                    "status_counts": {"buckets": []},
                    "src": {"buckets": []},
                    "tgt": {"buckets": []},
                },
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            degrees = await s.node_degrees_batch(["A", "B"])
            assert degrees["A"] == 3
            assert degrees["B"] == 3

    @pytest.mark.asyncio
    async def test_upsert_node(self, global_config, embed_func, mock_client):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.upsert_node(
                "Alice", {"entity_type": "person", "source_id": "c1<SEP>c2"}
            )
            mock_client.index.assert_awaited()
            call_kwargs = mock_client.index.call_args
            assert call_kwargs.kwargs["id"] == "Alice"
            body = call_kwargs.kwargs["body"]
            assert body["source_ids"] == ["c1", "c2"]
            assert body["entity_id"] == "Alice"

    @pytest.mark.asyncio
    async def test_upsert_edge_uses_canonical_id_without_reverse_lookup(
        self, global_config, embed_func, mock_client
    ):
        mock_client.mget = AsyncMock(side_effect=node_mget_side_effect({"A", "B"}))
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()

            await s.upsert_edge("A", "B", {"weight": "1.0"})
            await s.upsert_edge("B", "A", {"weight": "2.0"})

            for call in mock_client.exists.await_args_list:
                assert call.kwargs["index"] != s._edges_index
            edge_ids = [
                c.kwargs["id"]
                for c in mock_client.index.await_args_list
                if c.kwargs["index"] == s._edges_index
            ]
            assert edge_ids[0] == edge_ids[1] == _canonical_edge_id("A", "B")
            mock_client.delete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_upsert_edges_batch_collapses_reciprocal_edges(
        self, global_config, embed_func, mock_client
    ):
        mock_client.mget = AsyncMock(side_effect=node_mget_side_effect(set()))
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()

            bulk_calls = []

            async def capture_bulk(_client, actions, *args, **kwargs):
                bulk_calls.append(list(actions))
                return (len(bulk_calls[-1]), [])

            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk",
                new=AsyncMock(side_effect=capture_bulk),
            ):
                await s.upsert_edges_batch(
                    [
                        ("A", "B", {"weight": "1.0"}),
                        ("B", "A", {"weight": "2.0"}),
                    ]
                )

            edge_acts = [
                a for call in bulk_calls for a in call if a["_index"] == s._edges_index
            ]
            index_actions = [a for a in edge_acts if a["_op_type"] == "index"]
            assert not [a for a in edge_acts if a["_op_type"] == "delete"]
            assert len(index_actions) == 1
            assert index_actions[0]["_id"] == _canonical_edge_id("A", "B")
            assert index_actions[0]["_source"]["weight"] == "2.0"

            node_ids = sorted(
                a["_id"]
                for call in bulk_calls
                for a in call
                if a["_index"] == s._nodes_index
            )
            assert node_ids == ["A", "B"]

    @pytest.mark.asyncio
    async def test_migrate_edges_to_canonical_id_reindexes_legacy_docs(
        self, global_config, embed_func, mock_client
    ):
        s = self._make(global_config, embed_func)
        s.client = mock_client
        mock_client.indices.exists = AsyncMock(return_value=True)
        mock_client.indices.get_mapping = AsyncMock(
            return_value={s._edges_index: {"mappings": {}}}
        )
        mock_client.indices.put_mapping = AsyncMock()
        mock_client.search = AsyncMock(
            return_value={
                "_scroll_id": "s1",
                "hits": {
                    "hits": [
                        {
                            "_id": "edge-legacy-noncanonical",
                            "_source": {
                                "source_node_id": "A",
                                "target_node_id": "B",
                                "weight": 1.0,
                            },
                        }
                    ]
                },
            }
        )
        mock_client.scroll = AsyncMock(
            return_value={"_scroll_id": "s1", "hits": {"hits": []}}
        )
        mock_client.clear_scroll = AsyncMock()

        bulk_calls = []

        async def capture_bulk(_client, actions, *args, **kwargs):
            bulk_calls.append(list(actions))
            return (len(bulk_calls[-1]), [])

        with patch(
            "lightrag.kg.opensearch_impl.helpers.async_bulk",
            new=AsyncMock(side_effect=capture_bulk),
        ):
            await s._migrate_edges_to_canonical_id_if_needed()

        canonical = _canonical_edge_id("A", "B")
        all_actions = [a for call in bulk_calls for a in call]
        create_ops = [a for a in all_actions if a["_op_type"] == "create"]
        delete_ops = [a for a in all_actions if a["_op_type"] == "delete"]
        assert not [a for a in all_actions if a["_op_type"] == "index"]
        assert create_ops[0]["_id"] == canonical
        assert create_ops[0]["_source"]["source_node_id"] == "A"
        assert delete_ops[0]["_id"] == "edge-legacy-noncanonical"

        put_body = mock_client.indices.put_mapping.await_args.kwargs["body"]
        assert put_body["_meta"][_EDGE_ID_CANONICAL_META_FLAG] is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "create_errors, delete_errors, expect_raise",
        [
            ([], [{"delete": {"_id": "edge-old", "status": 404}}], False),
            ([], [{"delete": {"_id": "edge-old", "status": 500}}], True),
        ],
        ids=["benign-delete-404", "delete-500-fails-fast"],
    )
    async def test_migrate_edges_phase_error_handling(
        self,
        global_config,
        embed_func,
        mock_client,
        create_errors,
        delete_errors,
        expect_raise,
    ):
        s = self._make(global_config, embed_func)
        s.client = mock_client
        mock_client.indices.exists = AsyncMock(return_value=True)
        mock_client.indices.get_mapping = AsyncMock(
            return_value={s._edges_index: {"mappings": {}}}
        )
        mock_client.indices.put_mapping = AsyncMock()
        mock_client.search = AsyncMock(
            return_value={
                "_scroll_id": "s1",
                "hits": {
                    "hits": [
                        {
                            "_id": "edge-old",
                            "_source": {"source_node_id": "A", "target_node_id": "B"},
                        }
                    ]
                },
            }
        )
        mock_client.scroll = AsyncMock(
            return_value={"_scroll_id": "s1", "hits": {"hits": []}}
        )
        mock_client.clear_scroll = AsyncMock()

        with patch(
            "lightrag.kg.opensearch_impl.helpers.async_bulk",
            new=AsyncMock(side_effect=[(1, create_errors), (1, delete_errors)]),
        ):
            if expect_raise:
                with pytest.raises(RuntimeError):
                    await s._migrate_edges_to_canonical_id_if_needed()
                mock_client.indices.put_mapping.assert_not_awaited()
            else:
                await s._migrate_edges_to_canonical_id_if_needed()
                mock_client.indices.put_mapping.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_migrate_edges_fail_fast_preserves_source_when_create_fails(
        self, global_config, embed_func, mock_client
    ):
        s = self._make(global_config, embed_func)
        s.client = mock_client
        mock_client.indices.exists = AsyncMock(return_value=True)
        mock_client.indices.get_mapping = AsyncMock(
            return_value={s._edges_index: {"mappings": {}}}
        )
        mock_client.indices.put_mapping = AsyncMock()
        mock_client.search = AsyncMock(
            return_value={
                "_scroll_id": "s1",
                "hits": {
                    "hits": [
                        {
                            "_id": "edge-old",
                            "_source": {"source_node_id": "A", "target_node_id": "B"},
                        }
                    ]
                },
            }
        )
        mock_client.scroll = AsyncMock(
            return_value={"_scroll_id": "s1", "hits": {"hits": []}}
        )
        mock_client.clear_scroll = AsyncMock()

        canonical = _canonical_edge_id("A", "B")
        bulk_calls = []

        async def capture_bulk(_client, actions, *args, **kwargs):
            acts = list(actions)
            bulk_calls.append(acts)
            if acts and acts[0]["_op_type"] == "create":
                return (0, [{"create": {"_id": canonical, "status": 503}}])
            return (len(acts), [])

        with patch(
            "lightrag.kg.opensearch_impl.helpers.async_bulk",
            new=AsyncMock(side_effect=capture_bulk),
        ):
            with pytest.raises(RuntimeError):
                await s._migrate_edges_to_canonical_id_if_needed()

        all_actions = [a for call in bulk_calls for a in call]
        assert not [a for a in all_actions if a["_op_type"] == "delete"]
        mock_client.indices.put_mapping.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_migrate_merges_reverse_payload_into_canonical_on_conflict(
        self, global_config, embed_func, mock_client
    ):
        s = self._make(global_config, embed_func)
        s.client = mock_client
        canonical = _canonical_edge_id("A", "B")
        mock_client.indices.exists = AsyncMock(return_value=True)
        mock_client.indices.get_mapping = AsyncMock(
            return_value={s._edges_index: {"mappings": {}}}
        )
        mock_client.indices.put_mapping = AsyncMock()
        mock_client.search = AsyncMock(
            return_value={
                "_scroll_id": "s1",
                "hits": {
                    "hits": [
                        {
                            "_id": "edge-reverse",
                            "_source": {
                                "source_node_id": "B",
                                "target_node_id": "A",
                                "description": "d2",
                                "keywords": "beta,gamma",
                                "weight": 2.0,
                                "source_ids": ["c2"],
                                "file_path": "f2",
                            },
                        }
                    ]
                },
            }
        )
        mock_client.scroll = AsyncMock(
            return_value={"_scroll_id": "s1", "hits": {"hits": []}}
        )
        mock_client.clear_scroll = AsyncMock()
        mock_client.get = AsyncMock(
            return_value={
                "_id": canonical,
                "_seq_no": 7,
                "_primary_term": 1,
                "_source": {
                    "source_node_id": "A",
                    "target_node_id": "B",
                    "description": "d1",
                    "keywords": "alpha,beta",
                    "weight": 1.0,
                    "source_ids": ["c1"],
                    "file_path": "f1",
                },
            }
        )
        mock_client.index = AsyncMock()

        async def capture_bulk(_client, actions, *args, **kwargs):
            acts = list(actions)
            if acts and acts[0]["_op_type"] == "create":
                return (0, [{"create": {"_id": canonical, "status": 409}}])
            return (len(acts), [])

        with patch(
            "lightrag.kg.opensearch_impl.helpers.async_bulk",
            new=AsyncMock(side_effect=capture_bulk),
        ):
            await s._migrate_edges_to_canonical_id_if_needed()

        index_kwargs = mock_client.index.await_args.kwargs
        assert index_kwargs["id"] == canonical
        assert index_kwargs["if_seq_no"] == 7
        assert index_kwargs["if_primary_term"] == 1
        merged = index_kwargs["body"]
        assert merged["description"] == "d1<SEP>d2"
        assert merged["keywords"] == "alpha,beta,gamma"
        assert merged["weight"] == 3.0
        assert merged["source_ids"] == ["c1", "c2"]
        assert merged["file_path"] == "f1<SEP>f2"
        assert merged["source_node_id"] == "A"
        mock_client.indices.put_mapping.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_merge_into_canonical_retries_on_version_conflict(
        self, global_config, embed_func, mock_client
    ):
        s = self._make(global_config, embed_func)
        s.client = mock_client
        mock_client.get = AsyncMock(
            return_value={
                "_id": "edge-canon",
                "_seq_no": 1,
                "_primary_term": 1,
                "_source": {"source_node_id": "A", "target_node_id": "B"},
            }
        )
        mock_client.index = AsyncMock(
            side_effect=[ConflictError(409, "version_conflict", {}), None]
        )

        await s._merge_into_canonical_edge(
            "edge-canon", [("edge-rev9", {"source_id": "c9"})]
        )

        assert mock_client.get.await_count == 2
        assert mock_client.index.await_count == 2
        mock_client.delete.assert_awaited_once_with(
            index=s._edges_index, id="edge-rev9"
        )

    @pytest.mark.asyncio
    async def test_merge_into_canonical_recreates_when_canonical_vanished(
        self, global_config, embed_func, mock_client
    ):
        s = self._make(global_config, embed_func)
        s.client = mock_client
        mock_client.get = AsyncMock(side_effect=NotFoundError(404, "not_found", {}))
        mock_client.index = AsyncMock()

        await s._merge_into_canonical_edge(
            "edge-canon",
            [
                (
                    "edge-rev1",
                    {"source_ids": ["c1"], "weight": 1.0, "source_node_id": "A"},
                ),
                ("edge-rev2", {"source_ids": ["c2"], "weight": 2.0}),
            ],
        )

        kwargs = mock_client.index.await_args.kwargs
        assert kwargs["id"] == "edge-canon"
        assert "if_seq_no" not in kwargs
        body = kwargs["body"]
        assert body["source_ids"] == ["c1", "c2"]
        assert body["weight"] == 3.0
        assert body["source_node_id"] == "A"
        deleted_ids = {c.kwargs["id"] for c in mock_client.delete.await_args_list}
        assert deleted_ids == {"edge-rev1", "edge-rev2"}

    @pytest.mark.asyncio
    async def test_merge_into_canonical_aborts_after_persistent_conflicts(
        self, global_config, embed_func, mock_client
    ):
        s = self._make(global_config, embed_func)
        s.client = mock_client
        mock_client.get = AsyncMock(
            return_value={
                "_id": "edge-canon",
                "_seq_no": 1,
                "_primary_term": 1,
                "_source": {"source_node_id": "A", "target_node_id": "B"},
            }
        )
        mock_client.index = AsyncMock(
            side_effect=ConflictError(409, "version_conflict", {})
        )

        with pytest.raises(RuntimeError, match="could not merge into edge-canon"):
            await s._merge_into_canonical_edge(
                "edge-canon", [("edge-rev9", {"source_id": "c9"})]
            )
        assert mock_client.index.await_count == 3
        mock_client.delete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_migrate_merges_multiple_reverse_docs_into_one_canonical(
        self, global_config, embed_func, mock_client
    ):
        s = self._make(global_config, embed_func)
        s.client = mock_client
        canonical = _canonical_edge_id("A", "B")
        mock_client.indices.exists = AsyncMock(return_value=True)
        mock_client.indices.get_mapping = AsyncMock(
            return_value={s._edges_index: {"mappings": {}}}
        )
        mock_client.indices.put_mapping = AsyncMock()
        mock_client.search = AsyncMock(
            return_value={
                "_scroll_id": "s1",
                "hits": {
                    "hits": [
                        {
                            "_id": "edge-rev1",
                            "_source": {
                                "source_node_id": "B",
                                "target_node_id": "A",
                                "source_ids": ["c2"],
                                "weight": 2.0,
                            },
                        },
                        {
                            "_id": "edge-rev2",
                            "_source": {
                                "source_node_id": "B",
                                "target_node_id": "A",
                                "source_ids": ["c3"],
                                "weight": 3.0,
                            },
                        },
                    ]
                },
            }
        )
        mock_client.scroll = AsyncMock(
            return_value={"_scroll_id": "s1", "hits": {"hits": []}}
        )
        mock_client.clear_scroll = AsyncMock()
        mock_client.get = AsyncMock(
            return_value={
                "_id": canonical,
                "_seq_no": 7,
                "_primary_term": 1,
                "_source": {
                    "source_node_id": "A",
                    "target_node_id": "B",
                    "source_ids": ["c1"],
                    "weight": 1.0,
                },
            }
        )
        mock_client.index = AsyncMock()

        async def capture_bulk(_client, actions, *args, **kwargs):
            acts = list(actions)
            if acts and acts[0]["_op_type"] == "create":
                return (1, [{"create": {"_id": canonical, "status": 409}}])
            return (len(acts), [])

        with patch(
            "lightrag.kg.opensearch_impl.helpers.async_bulk",
            new=AsyncMock(side_effect=capture_bulk),
        ):
            await s._migrate_edges_to_canonical_id_if_needed()

        assert mock_client.index.await_count == 1
        merged = mock_client.index.await_args.kwargs["body"]
        assert merged["source_ids"] == ["c1", "c2", "c3"]
        assert merged["weight"] == 6.0

    @pytest.mark.asyncio
    async def test_migrate_premerges_same_canonical_docs_into_one_create(
        self, global_config, embed_func, mock_client
    ):
        s = self._make(global_config, embed_func)
        s.client = mock_client
        canonical = _canonical_edge_id("A", "B")
        mock_client.indices.exists = AsyncMock(return_value=True)
        mock_client.indices.get_mapping = AsyncMock(
            return_value={s._edges_index: {"mappings": {}}}
        )
        mock_client.indices.put_mapping = AsyncMock()
        mock_client.search = AsyncMock(
            return_value={
                "_scroll_id": "s1",
                "hits": {
                    "hits": [
                        {
                            "_id": "edge-rev1",
                            "_source": {
                                "source_node_id": "B",
                                "target_node_id": "A",
                                "source_ids": ["c2"],
                                "weight": 2.0,
                            },
                        },
                        {
                            "_id": "edge-rev2",
                            "_source": {
                                "source_node_id": "B",
                                "target_node_id": "A",
                                "source_ids": ["c3"],
                                "weight": 3.0,
                            },
                        },
                    ]
                },
            }
        )
        mock_client.scroll = AsyncMock(
            return_value={"_scroll_id": "s1", "hits": {"hits": []}}
        )
        mock_client.clear_scroll = AsyncMock()
        mock_client.index = AsyncMock()

        created = []

        async def capture_bulk(_client, actions, *args, **kwargs):
            acts = list(actions)
            if acts and acts[0]["_op_type"] == "create":
                created.extend(acts)
                return (len(acts), [])
            return (len(acts), [])

        with patch(
            "lightrag.kg.opensearch_impl.helpers.async_bulk",
            new=AsyncMock(side_effect=capture_bulk),
        ):
            await s._migrate_edges_to_canonical_id_if_needed()

        create_for_canon = [a for a in created if a["_id"] == canonical]
        assert len(create_for_canon) == 1
        body = create_for_canon[0]["_source"]
        assert body["weight"] == 5.0
        assert body["source_ids"] == ["c2", "c3"]
        mock_client.index.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_migrate_edges_logs_progress_for_large_scan(
        self, global_config, embed_func, mock_client
    ):
        s = self._make(global_config, embed_func)
        s.client = mock_client
        mock_client.indices.exists = AsyncMock(return_value=True)
        mock_client.indices.get_mapping = AsyncMock(
            return_value={s._edges_index: {"mappings": {}}}
        )
        mock_client.indices.put_mapping = AsyncMock()
        mock_client.count = AsyncMock(return_value={"count": 3})
        mock_client.search = AsyncMock(
            return_value={
                "_scroll_id": "s1",
                "hits": {
                    "hits": [
                        {
                            "_id": _canonical_edge_id(n, "Z"),
                            "_source": {"source_node_id": n, "target_node_id": "Z"},
                        }
                        for n in ("A", "B", "C")
                    ]
                },
            }
        )
        mock_client.scroll = AsyncMock(
            return_value={"_scroll_id": "s1", "hits": {"hits": []}}
        )
        mock_client.clear_scroll = AsyncMock()

        fake_logger = MagicMock()
        with (
            patch("lightrag.kg.opensearch_impl._EDGE_MIGRATION_PROGRESS_INTERVAL", 2),
            patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk",
                new=AsyncMock(return_value=(0, [])),
            ),
            patch("lightrag.kg.opensearch_impl.logger", fake_logger),
        ):
            await s._migrate_edges_to_canonical_id_if_needed()

        info_lines = [c.args[0] for c in fake_logger.info.call_args_list]
        progress_lines = [m for m in info_lines if "progress: scanned" in m]
        assert progress_lines and "/3" in progress_lines[0]
        assert mock_client.indices.put_mapping.await_count == 1

    @pytest.mark.asyncio
    async def test_migrate_edges_skipped_when_flag_present(
        self, global_config, embed_func, mock_client
    ):
        s = self._make(global_config, embed_func)
        s.client = mock_client
        mock_client.indices.exists = AsyncMock(return_value=True)
        mock_client.indices.get_mapping = AsyncMock(
            return_value={
                s._edges_index: {
                    "mappings": {"_meta": {_EDGE_ID_CANONICAL_META_FLAG: True}}
                }
            }
        )
        mock_client.indices.put_mapping = AsyncMock()
        mock_client.search = AsyncMock()

        await s._migrate_edges_to_canonical_id_if_needed()

        mock_client.search.assert_not_awaited()
        mock_client.indices.put_mapping.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reads_short_circuit_after_drop(
        self, global_config, embed_func, mock_client
    ):
        mock_client.transport = AsyncMock()
        mock_client.transport.perform_request = AsyncMock(
            side_effect=Exception("PPL not available")
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.drop()

            graph = await s.get_knowledge_graph("A", max_depth=2)

            assert await s.get_node("A") is None
            assert await s.get_all_labels() == []
            assert await s.has_edge("A", "B") is False
            assert await s.node_degree("A") == 0
            assert len(graph.nodes) == 0
            assert len(graph.edges) == 0

            mock_client.mget.assert_not_awaited()
            mock_client.search.assert_not_awaited()
            mock_client.create_pit.assert_not_awaited()
            mock_client.count.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_delete_node(self, global_config, embed_func, mock_client):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.delete_node("Alice")
            mock_client.delete_by_query.assert_awaited_once()
            mock_client.delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_remove_nodes(self, global_config, embed_func, mock_client):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (2, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.remove_nodes(["A", "B"])
                mock_client.delete_by_query.assert_awaited_once()
                mock_bulk.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_remove_edges(self, global_config, embed_func, mock_client):
        mock_client.bulk = AsyncMock(return_value={"errors": False, "items": []})
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.remove_edges([("A", "B"), ("B", "A"), ("C", "D")])
            mock_client.bulk.assert_awaited_once()
            call_body = mock_client.bulk.call_args.kwargs["body"]
            assert len(call_body) == 2
            assert {op["delete"]["_id"] for op in call_body} == {
                _canonical_edge_id("A", "B"),
                _canonical_edge_id("C", "D"),
            }

    @pytest.mark.asyncio
    async def test_get_all_labels(self, global_config, embed_func, mock_client):
        mock_client.search = AsyncMock(
            return_value={
                "hits": {
                    "hits": [
                        {"_id": "Bob", "sort": [1]},
                        {"_id": "Alice", "sort": [2]},
                    ],
                    "total": {"value": 2},
                },
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            labels = await s.get_all_labels()
            assert labels == ["Alice", "Bob"]

    @pytest.mark.asyncio
    async def test_get_popular_labels(self, global_config, embed_func, mock_client):
        mock_client.search = AsyncMock(
            return_value={
                "hits": {"hits": [], "total": {"value": 0}},
                "aggregations": {
                    "src": {
                        "buckets": [
                            {"key": "B", "doc_count": 3},
                            {"key": "A", "doc_count": 2},
                        ]
                    },
                    "tgt": {"buckets": [{"key": "A", "doc_count": 2}]},
                    "status_counts": {"buckets": []},
                },
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            labels = await s.get_popular_labels(limit=2)
            assert labels == ["A", "B"]

    @pytest.mark.asyncio
    async def test_get_knowledge_graph_all_backfills_isolated_nodes_when_truncated(
        self, global_config, embed_func, mock_client
    ):
        mock_client.count = AsyncMock(return_value={"count": 5})
        mock_client.search = AsyncMock(
            side_effect=[
                {
                    "hits": {"hits": [], "total": {"value": 1}},
                    "aggregations": {
                        "src": {"buckets": [{"key": "A", "doc_count": 1}]},
                        "tgt": {"buckets": [{"key": "B", "doc_count": 1}]},
                        "status_counts": {"buckets": []},
                    },
                },
                {
                    "hits": {
                        "hits": [
                            {"_id": "A", "sort": [1]},
                            {"_id": "B", "sort": [2]},
                            {"_id": "C", "sort": [3]},
                            {"_id": "D", "sort": [4]},
                            {"_id": "E", "sort": [5]},
                        ],
                        "total": {"value": 5},
                    }
                },
                {
                    "hits": {
                        "hits": [
                            {
                                "_id": "edge-ab",
                                "_source": {
                                    "source_node_id": "A",
                                    "target_node_id": "B",
                                    "relationship": "knows",
                                },
                            }
                        ],
                        "total": {"value": 1},
                    }
                },
            ]
        )
        mock_client.mget = AsyncMock(
            side_effect=mget_by_ids_side_effect(
                {node_id: {"entity_type": "person"} for node_id in "ABCD"}
            )
        )

        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()

            result = await s.get_knowledge_graph("*", max_nodes=4)

            assert result.is_truncated is True
            assert [node.id for node in result.nodes] == ["A", "B", "C", "D"]
            assert len(result.edges) == 1
            assert result.edges[0].source == "A"
            assert result.edges[0].target == "B"
            assert mock_client.create_pit.await_count == 2

    @pytest.mark.asyncio
    async def test_get_knowledge_graph_all_paginates_edges_between_selected_nodes(
        self, global_config, embed_func, mock_client
    ):
        mock_client.count = AsyncMock(return_value={"count": 2})
        first_edge_page = [
            {
                "_id": f"edge-{i}",
                "_source": {
                    "source_node_id": "A",
                    "target_node_id": "B",
                    "relationship": "knows",
                },
                "sort": [i],
            }
            for i in range(10000)
        ]
        mock_client.search = AsyncMock(
            side_effect=[
                {
                    "hits": {
                        "hits": [{"_id": "A"}, {"_id": "B"}],
                        "total": {"value": 2},
                    }
                },
                {"hits": {"hits": first_edge_page, "total": {"value": 10001}}},
                {
                    "hits": {
                        "hits": [
                            {
                                "_id": "edge-last",
                                "_source": {
                                    "source_node_id": "B",
                                    "target_node_id": "A",
                                    "relationship": "knows",
                                },
                                "sort": [10000],
                            }
                        ],
                        "total": {"value": 10001},
                    }
                },
            ]
        )
        mock_client.mget = AsyncMock(
            return_value={
                "docs": [
                    {"_id": "A", "found": True, "_source": {"entity_type": "person"}},
                    {"_id": "B", "found": True, "_source": {"entity_type": "person"}},
                ]
            }
        )

        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()

            result = await s.get_knowledge_graph("*", max_nodes=2)

            assert len(result.nodes) == 2
            assert len(result.edges) == 2
            assert {(edge.source, edge.target) for edge in result.edges} == {
                ("A", "B"),
                ("B", "A"),
            }
            assert mock_client.search.await_count == 3

    @pytest.mark.asyncio
    async def test_search_labels_empty_query(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.search_labels("  ") == []
            mock_client.search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_drop_treats_missing_graph_indices_as_success(
        self, global_config, embed_func, mock_client
    ):
        mock_client.indices.delete = AsyncMock(
            side_effect=[missing_index_error(), missing_index_error()]
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            result = await s.drop()
            assert result["status"] == "success"
            assert s._indices_ready is False

    @pytest.mark.asyncio
    async def test_construct_graph_node(self, global_config, embed_func):
        s = self._make(global_config, embed_func)
        node = s._construct_graph_node(
            "Alice",
            {
                "entity_type": "person",
                "description": "A researcher",
                "_id": "Alice",
                "entity_id": "Alice",
            },
        )
        assert node.id == "Alice"
        assert "entity_type" in node.properties
        assert "_id" not in node.properties
        assert node.properties["entity_id"] == "Alice"

    @pytest.mark.asyncio
    async def test_construct_graph_edge(self, global_config, embed_func):
        s = self._make(global_config, embed_func)
        edge = s._construct_graph_edge(
            "e1",
            {
                "source_node_id": "A",
                "target_node_id": "B",
                "relationship": "knows",
                "weight": 1.0,
            },
        )
        assert edge.source == "A"
        assert edge.target == "B"
        assert edge.type == "knows"
        assert "source_node_id" not in edge.properties

    @pytest.mark.asyncio
    async def test_bfs_subgraph_start_not_found(
        self, global_config, embed_func, mock_client
    ):
        mock_client.mget = AsyncMock(
            return_value={"docs": [{"_id": "NonExistent", "found": False}]}
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            result = await s.get_knowledge_graph("NonExistent", max_depth=2)
            assert len(result.nodes) == 0
            assert len(result.edges) == 0


# ---------------------------------------------------------------------------
# PPL graphlookup detection and server-side BFS
# ---------------------------------------------------------------------------


class TestGraphPPLDetection:
    """Tests for PPL graphlookup detection and server-side BFS."""

    def _make(self, global_config, embed_func, workspace: str = "test"):
        return _make_graph(global_config, embed_func, workspace)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("env_value", [None, "", "ture"])
    async def test_ppl_detected_when_available(
        self, global_config, embed_func, mock_client, monkeypatch, env_value
    ):
        if env_value is None:
            monkeypatch.delenv("OPENSEARCH_USE_PPL_GRAPHLOOKUP", raising=False)
        else:
            monkeypatch.setenv("OPENSEARCH_USE_PPL_GRAPHLOOKUP", env_value)
        mock_client.transport = AsyncMock()
        mock_client.transport.perform_request = AsyncMock(
            return_value={"datarows": [], "schema": []}
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert s._ppl_graphlookup_available is True
            mock_client.transport.perform_request.assert_awaited()

    @pytest.mark.asyncio
    async def test_ppl_not_detected_when_endpoint_fails(
        self, global_config, embed_func, mock_client
    ):
        mock_client.transport = AsyncMock()
        mock_client.transport.perform_request = AsyncMock(
            side_effect=Exception("PPL not supported")
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert s._ppl_graphlookup_available is False

    @pytest.mark.asyncio
    async def test_env_override_true(self, global_config, embed_func, mock_client):
        with patch.dict("os.environ", {"OPENSEARCH_USE_PPL_GRAPHLOOKUP": " Yes "}):
            with patch.object(ClientManager, "get_client", return_value=mock_client):
                s = self._make(global_config, embed_func)
                await s.initialize()
                assert s._ppl_graphlookup_available is True
                mock_client.transport.perform_request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_env_override_false(self, global_config, embed_func, mock_client):
        mock_client.transport = AsyncMock()
        mock_client.transport.perform_request = AsyncMock(
            return_value={"datarows": [], "schema": []}
        )
        with patch.dict("os.environ", {"OPENSEARCH_USE_PPL_GRAPHLOOKUP": "0"}):
            with patch.object(ClientManager, "get_client", return_value=mock_client):
                s = self._make(global_config, embed_func)
                await s.initialize()
                assert s._ppl_graphlookup_available is False

    @pytest.mark.asyncio
    async def test_ppl_bfs_calls_ppl_endpoint(
        self, global_config, embed_func, mock_client
    ):
        mock_client.transport = AsyncMock()
        ppl_response = {
            "schema": [
                {"name": "entity_id", "type": "string"},
                {"name": "connected_edges", "type": "struct"},
            ],
            "datarows": [
                [
                    "A",
                    [
                        {
                            "source_node_id": "A",
                            "target_node_id": "B",
                            "weight": 1.0,
                            "_depth": 0,
                        },
                        {
                            "source_node_id": "B",
                            "target_node_id": "C",
                            "weight": 0.5,
                            "_depth": 1,
                        },
                    ],
                ]
            ],
        }
        mock_client.transport.perform_request = AsyncMock(return_value=ppl_response)
        mock_client.get = AsyncMock(
            return_value={
                "_id": "A",
                "_source": {"entity_type": "person", "description": "Node A"},
            }
        )
        mock_client.mget = AsyncMock(
            side_effect=mget_by_ids_side_effect(
                {
                    "A": {"entity_type": "person", "description": "Node A"},
                    "B": {"entity_type": "person"},
                    "C": {"entity_type": "person"},
                }
            )
        )
        mock_client.search = AsyncMock(
            return_value={
                "hits": {
                    "hits": [
                        {
                            "_id": "e1",
                            "_source": {
                                "source_node_id": "A",
                                "target_node_id": "B",
                                "relationship": "knows",
                            },
                        },
                        {
                            "_id": "e2",
                            "_source": {
                                "source_node_id": "B",
                                "target_node_id": "C",
                                "relationship": "knows",
                            },
                        },
                    ],
                    "total": {"value": 2},
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
            assert s._ppl_graphlookup_available is True

            result = await s.get_knowledge_graph("A", max_depth=2)
            assert len(result.nodes) == 3
            assert len(result.edges) == 2
            assert mock_client.transport.perform_request.await_count == 3
            actual_query = mock_client.transport.perform_request.call_args_list[2]
            ppl_query = actual_query.kwargs["body"]["query"]
            assert ppl_query.startswith(f"source = {s._nodes_index} ")

    @pytest.mark.asyncio
    async def test_ppl_bfs_falls_back_on_query_failure(
        self, global_config, embed_func, mock_client
    ):
        call_count = {"n": 0}

        async def ppl_side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                return {"datarows": [], "schema": []}
            raise Exception("PPL query timeout")

        mock_client.transport = AsyncMock()
        mock_client.transport.perform_request = AsyncMock(side_effect=ppl_side_effect)
        mock_client.mget = AsyncMock(
            side_effect=mget_by_ids_side_effect({"A": {"entity_type": "person"}})
        )

        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert s._ppl_graphlookup_available is True

            result = await s.get_knowledge_graph("A", max_depth=2)
            assert call_count["n"] == 3
            mock_client.search.assert_awaited()
            assert [node.id for node in result.nodes] == ["A"]

    @pytest.mark.asyncio
    async def test_escape_ppl(self, global_config, embed_func):
        s = self._make(global_config, embed_func)
        assert s._escape_ppl("it's") == "it\\'s"
        assert s._escape_ppl("normal") == "normal"
        assert s._escape_ppl("back\\slash") == "back\\\\slash"
        assert s._escape_ppl("both\\and'quote") == "both\\\\and\\'quote"

    @pytest.mark.asyncio
    async def test_ppl_bfs_depth_zero_returns_start_only(
        self, global_config, embed_func, mock_client
    ):
        mock_client.transport = AsyncMock()
        mock_client.transport.perform_request = AsyncMock(
            return_value={"datarows": [], "schema": []}
        )
        mock_client.mget = AsyncMock(
            side_effect=mget_by_ids_side_effect({"A": {"entity_type": "person"}})
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert s._ppl_graphlookup_available is True
            result = await s.get_knowledge_graph("A", max_depth=0)
            assert len(result.nodes) == 1
            assert result.nodes[0].id == "A"
            assert len(result.edges) == 0
            assert mock_client.transport.perform_request.await_count == 2

    @pytest.mark.asyncio
    async def test_ppl_bfs_empty_connected_edges(
        self, global_config, embed_func, mock_client
    ):
        mock_client.transport = AsyncMock()
        ppl_response = {
            "schema": [
                {"name": "entity_id", "type": "string"},
                {"name": "connected_edges", "type": "struct"},
            ],
            "datarows": [["A", []]],
        }
        mock_client.transport.perform_request = AsyncMock(return_value=ppl_response)
        mock_client.mget = AsyncMock(
            side_effect=mget_by_ids_side_effect({"A": {"entity_type": "person"}})
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            result = await s.get_knowledge_graph("A", max_depth=2)
            assert len(result.nodes) == 1
            assert result.nodes[0].id == "A"
            mock_client.search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ppl_bfs_ranks_same_depth_nodes_by_degree_then_id(
        self, global_config, embed_func, mock_client
    ):
        mock_client.transport = AsyncMock()
        ppl_response = {
            "schema": [
                {"name": "entity_id", "type": "string"},
                {"name": "connected_edges", "type": "struct"},
            ],
            "datarows": [
                [
                    "A",
                    [
                        {
                            "source_node_id": "A",
                            "target_node_id": "C",
                            "weight": 1.0,
                            "_depth": 1,
                        },
                        {
                            "source_node_id": "B",
                            "target_node_id": "D",
                            "weight": 10.0,
                            "_depth": 1,
                        },
                        {
                            "source_node_id": "A",
                            "target_node_id": "B",
                            "weight": 1.0,
                            "_depth": 0,
                        },
                    ],
                ]
            ],
        }
        mock_client.transport.perform_request = AsyncMock(return_value=ppl_response)
        mock_client.mget = AsyncMock(
            side_effect=mget_by_ids_side_effect(
                {
                    "A": {"entity_type": "person"},
                    "B": {"entity_type": "person"},
                    "C": {"entity_type": "person"},
                    "D": {"entity_type": "person"},
                }
            )
        )

        edge_hits = {
            "hits": {
                "hits": [
                    {
                        "_id": "e1",
                        "_source": {
                            "source_node_id": "A",
                            "target_node_id": "B",
                            "relationship": "knows",
                        },
                        "sort": [1],
                    },
                    {
                        "_id": "e2",
                        "_source": {
                            "source_node_id": "A",
                            "target_node_id": "C",
                            "relationship": "knows",
                        },
                        "sort": [2],
                    },
                ],
                "total": {"value": 2},
            }
        }
        degree_aggs = {
            "hits": {"hits": []},
            "aggregations": {
                "source_degrees": {
                    "ids": {
                        "buckets": [
                            {"key": "C", "doc_count": 5},
                            {"key": "B", "doc_count": 1},
                        ]
                    }
                },
                "target_degrees": {"ids": {"buckets": [{"key": "D", "doc_count": 1}]}},
            },
        }

        async def _search(index=None, body=None, **kwargs):
            return degree_aggs if "aggs" in (body or {}) else edge_hits

        mock_client.search = AsyncMock(side_effect=_search)

        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()

            result = await s.get_knowledge_graph("A", max_depth=2, max_nodes=3)

            assert sorted(node.id for node in result.nodes) == ["A", "B", "C"]
            assert result.is_truncated is True
            assert {(edge.source, edge.target) for edge in result.edges} == {
                ("A", "B"),
                ("A", "C"),
            }

    @pytest.mark.asyncio
    async def test_node_degree_uses_count_api(
        self, global_config, embed_func, mock_client
    ):
        mock_client.count = AsyncMock(return_value={"count": 7})
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            degree = await s.node_degree("X")
            assert degree == 7
            mock_client.count.assert_awaited()
            assert s._edges_index in str(mock_client.count.call_args)


# ---------------------------------------------------------------------------
# Graph read contracts (isolated entities, tie-breaking, missing edge index)
# ---------------------------------------------------------------------------


class TestGraphReadContract:
    """Isolated entities stay ranked, absent nodes stay distinct from isolated ones."""

    def _make(self, global_config, embed_func, workspace: str = "test"):
        return _make_graph(global_config, embed_func, workspace)

    @staticmethod
    def _aggs(src_buckets, tgt_buckets):
        return {
            "hits": {"hits": [], "total": {"value": 0}},
            "aggregations": {
                "src": {"buckets": src_buckets},
                "tgt": {"buckets": tgt_buckets},
                "status_counts": {"buckets": []},
            },
        }

    @staticmethod
    def _node_page(ids):
        return {
            "hits": {
                "hits": [{"_id": i, "sort": [i]} for i in ids],
                "total": {"value": len(ids)},
            }
        }

    @pytest.mark.asyncio
    async def test_popular_labels_backfills_isolated_nodes(
        self, global_config, embed_func, mock_client
    ):
        mock_client.search = AsyncMock(
            side_effect=[
                self._aggs(
                    [{"key": "A", "doc_count": 5}], [{"key": "B", "doc_count": 2}]
                ),
                self._node_page(["A", "B", "Orphan1", "Orphan2"]),
            ]
        )
        mock_client.mget = AsyncMock(side_effect=node_mget_side_effect({"A", "B"}))
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            labels = await s.get_popular_labels(limit=10)

        assert labels == ["A", "B", "Orphan1", "Orphan2"]
        mock_client.create_pit.assert_awaited_once()
        mock_client.delete_pit.assert_awaited_once()
        scan_body = mock_client.search.call_args_list[-1].kwargs["body"]
        assert scan_body["sort"][0] == {"entity_id": {"order": "asc"}}, scan_body

    @pytest.mark.asyncio
    async def test_popular_labels_skips_node_scan_when_limit_already_filled(
        self, global_config, embed_func, mock_client
    ):
        mock_client.search = AsyncMock(
            return_value=self._aggs(
                [{"key": "A", "doc_count": 5}, {"key": "B", "doc_count": 2}], []
            )
        )
        mock_client.mget = AsyncMock(side_effect=node_mget_side_effect({"A", "B"}))
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            labels = await s.get_popular_labels(limit=2)

        assert labels == ["A", "B"]
        mock_client.create_pit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_popular_labels_breaks_degree_ties_on_label(
        self, global_config, embed_func, mock_client
    ):
        mock_client.search = AsyncMock(
            return_value=self._aggs(
                [{"key": "Zeta", "doc_count": 1}, {"key": "Alpha", "doc_count": 1}], []
            )
        )
        mock_client.mget = AsyncMock(
            side_effect=node_mget_side_effect({"Zeta", "Alpha"})
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.get_popular_labels(limit=2) == ["Alpha", "Zeta"]

    @pytest.mark.asyncio
    async def test_popular_labels_missing_edge_index_ranks_from_node_index(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()

            async def _search(index=None, body=None, **kwargs):
                if index == s._edges_index:
                    raise missing_index_error()
                return self._node_page(["Orphan1", "Orphan2"])

            mock_client.search = AsyncMock(side_effect=_search)
            labels = await s.get_popular_labels(limit=10)

        assert labels == ["Orphan1", "Orphan2"]
        mock_client.create_pit.assert_awaited_once()
        assert s._indices_ready is False

    @pytest.mark.asyncio
    async def test_popular_labels_missing_node_index_still_returns_empty(
        self, global_config, embed_func, mock_client
    ):
        mock_client.search = AsyncMock(return_value=self._aggs([], []))
        mock_client.create_pit = AsyncMock(side_effect=missing_index_error())
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            labels = await s.get_popular_labels(limit=10)

        assert labels == []
        assert s._indices_ready is False

    @pytest.mark.asyncio
    async def test_popular_labels_edge_transport_error_still_raises(
        self, global_config, embed_func, mock_client
    ):
        mock_client.search = AsyncMock(side_effect=transient_error())
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            with pytest.raises(TransportError):
                await s.get_popular_labels(limit=10)

        mock_client.create_pit.assert_not_awaited()
        assert s._indices_ready is True

    @pytest.mark.asyncio
    async def test_knowledge_graph_all_missing_edge_index_keeps_nodes(
        self, global_config, embed_func, mock_client
    ):
        mock_client.count = AsyncMock(return_value={"count": 5})
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()

            async def _search(index=None, body=None, **kwargs):
                if index == s._edges_index:
                    raise missing_index_error()
                assert index == s._nodes_index
                return self._node_page(["A", "B", "C", "D"])

            async def _create_pit(index=None, **kwargs):
                if index == s._edges_index:
                    raise missing_index_error()
                return {"pit_id": "pit-nodes"}

            mock_client.search = AsyncMock(side_effect=_search)
            mock_client.create_pit = AsyncMock(side_effect=_create_pit)
            mock_client.mget = AsyncMock(
                side_effect=mget_by_ids_side_effect(
                    {node_id: {"entity_type": "person"} for node_id in "ABCD"}
                )
            )
            result = await s.get_knowledge_graph("*", max_nodes=4)

        assert result.is_truncated is True
        assert [node.id for node in result.nodes] == ["A", "B", "C", "D"]
        assert result.edges == []
        assert s._indices_ready is False

    @pytest.mark.asyncio
    async def test_knowledge_graph_all_breaks_degree_ties_on_label(
        self, global_config, embed_func, mock_client
    ):
        mock_client.count = AsyncMock(return_value={"count": 3})
        mock_client.search = AsyncMock(
            return_value=self._aggs(
                [
                    {"key": "Zeta", "doc_count": 1},
                    {"key": "Beta", "doc_count": 1},
                    {"key": "Alpha", "doc_count": 1},
                ],
                [],
            )
        )
        mock_client.mget = AsyncMock(
            side_effect=node_mget_side_effect({"Zeta", "Beta", "Alpha"})
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            kg = await s.get_knowledge_graph("*", max_depth=1, max_nodes=2)

        assert sorted(node.id for node in kg.nodes) == ["Alpha", "Beta"]
        assert kg.is_truncated is True

    @pytest.mark.asyncio
    async def test_knowledge_graph_all_refills_from_the_ranked_band(
        self, global_config, embed_func, mock_client
    ):
        mock_client.count = AsyncMock(return_value={"count": 9})
        mock_client.search = AsyncMock(
            return_value=self._aggs(
                [
                    {"key": "Dangling", "doc_count": 9},
                    {"key": "Real", "doc_count": 5},
                    {"key": "Lower", "doc_count": 1},
                ],
                [],
            )
        )
        mock_client.mget = AsyncMock(
            side_effect=node_mget_side_effect({"Real", "Lower"})
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            kg = await s.get_knowledge_graph("*", max_depth=1, max_nodes=2)

        assert [node.id for node in kg.nodes] == ["Real", "Lower"]
        requested = [
            call.kwargs["body"]["ids"] for call in mock_client.mget.await_args_list
        ]
        assert requested == [["Dangling", "Real"], ["Lower"]], requested

    @pytest.mark.asyncio
    async def test_knowledge_graph_all_isolated_topup_sorts_by_label(
        self, global_config, embed_func, mock_client
    ):
        mock_client.count = AsyncMock(return_value={"count": 3})
        mock_client.search = AsyncMock(
            side_effect=[
                self._aggs([], []),
                self._node_page(["Alpha", "Beta"]),
                {"hits": {"hits": [], "total": {"value": 0}}},
            ]
        )
        mock_client.mget = AsyncMock(
            side_effect=node_mget_side_effect({"Alpha", "Beta"})
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            kg = await s.get_knowledge_graph("*", max_depth=1, max_nodes=2)

        scan_body = next(
            call.kwargs["body"]
            for call in mock_client.search.call_args_list
            if "match_all" in call.kwargs.get("body", {}).get("query", {})
        )
        assert scan_body["sort"][0] == {"entity_id": {"order": "asc"}}, scan_body
        assert [node.id for node in kg.nodes] == ["Alpha", "Beta"]

    @pytest.mark.asyncio
    async def test_knowledge_graph_all_topup_scan_sorts_by_label_not_shard_order(
        self, global_config, embed_func, mock_client
    ):
        mock_client.count = AsyncMock(return_value={"count": 9})
        mock_client.search = AsyncMock(
            side_effect=[
                self._aggs([{"key": "Connected", "doc_count": 3}], []),
                self._node_page(["Aardvark", "Orphan"]),
                {"hits": {"hits": [], "total": {"value": 0}}},
            ]
        )
        mock_client.mget = AsyncMock(
            side_effect=node_mget_side_effect({"Connected", "Aardvark", "Orphan"})
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.get_knowledge_graph("*", max_depth=1, max_nodes=3)

        scan_body = next(
            call.kwargs["body"]
            for call in mock_client.search.call_args_list
            if "pit" in call.kwargs.get("body", {})
        )
        assert scan_body["sort"][0] == {"entity_id": {"order": "asc"}}, scan_body

    @pytest.mark.asyncio
    async def test_get_node_edges_returns_empty_list_for_isolated_node(
        self, global_config, embed_func, mock_client
    ):
        mock_client.search = AsyncMock(
            return_value={"hits": {"hits": [], "total": {"value": 0}}}
        )
        mock_client.exists = AsyncMock(return_value=True)
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.get_node_edges("Lonely") == []

    @pytest.mark.asyncio
    async def test_get_node_edges_returns_none_for_a_dangling_target_endpoint(
        self, global_config, embed_func, mock_client
    ):
        mock_client.search = AsyncMock(
            return_value={
                "hits": {
                    "hits": [
                        {
                            "_id": "e1",
                            "_source": {
                                "source_node_id": "A",
                                "target_node_id": "Target",
                            },
                            "sort": [1],
                        }
                    ],
                    "total": {"value": 1},
                }
            }
        )
        mock_client.exists = AsyncMock(return_value=False)
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.get_node_edges("Target") is None
            mock_client.create_pit.assert_not_awaited()


# ---------------------------------------------------------------------------
# Endpoint placeholder safety
# ---------------------------------------------------------------------------


class TestGraphEndpointPlaceholderSafety:
    """Placeholder writes for edge endpoints must never overwrite a real node."""

    def _make(self, global_config, embed_func, workspace: str = "test"):
        return _make_graph(global_config, embed_func, workspace)

    @pytest.mark.asyncio
    async def test_has_nodes_batch_raises_on_short_docs_array(
        self, global_config, embed_func, mock_client
    ):
        mock_client.mget = AsyncMock(
            return_value={
                "docs": [{"_id": "A", "found": True, "_source": {"entity_id": "A"}}]
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            with pytest.raises(RuntimeError, match="expected exactly 2"):
                await s.has_nodes_batch(["A", "B"])

    @pytest.mark.asyncio
    async def test_ambiguous_probe_never_reaches_the_placeholder_write(
        self, global_config, embed_func, mock_client
    ):
        mock_client.mget = AsyncMock(
            return_value={
                "docs": [
                    {"_id": "A", "error": {"type": "shard_unavailable"}},
                    {"_id": "B", "found": False},
                ]
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            with pytest.raises(RuntimeError, match="mget item error"):
                await s.upsert_edge("A", "B", {"weight": "1.0"})
            assert mock_client.index.await_count == 0

    @pytest.mark.asyncio
    async def test_placeholder_write_is_insert_only(
        self, global_config, embed_func, mock_client
    ):
        mock_client.mget = AsyncMock(side_effect=node_mget_side_effect(set()))
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.upsert_edge("A", "B", {"weight": "1.0"})

        node_writes = [
            c
            for c in mock_client.index.await_args_list
            if c.kwargs["index"] == s._nodes_index
        ]
        assert [c.kwargs["id"] for c in node_writes] == ["A", "B"]
        assert all(c.kwargs.get("op_type") == "create" for c in node_writes)
        edge_writes = [
            c
            for c in mock_client.index.await_args_list
            if c.kwargs["index"] == s._edges_index
        ]
        assert len(edge_writes) == 1
        assert edge_writes[0].kwargs.get("op_type") is None

    @pytest.mark.asyncio
    async def test_placeholder_conflict_is_not_an_error(
        self, global_config, embed_func, mock_client
    ):
        mock_client.mget = AsyncMock(side_effect=node_mget_side_effect(set()))

        async def _index(index=None, id=None, body=None, **kwargs):
            if kwargs.get("op_type") == "create":
                raise ConflictError(409, "version_conflict_engine_exception", {})
            return {"result": "created"}

        mock_client.index = AsyncMock(side_effect=_index)
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.upsert_edge("A", "B", {"weight": "1.0"})

        edge_writes = [
            c
            for c in mock_client.index.await_args_list
            if c.kwargs["index"] == s._edges_index
        ]
        assert len(edge_writes) == 1

    @pytest.mark.asyncio
    async def test_batch_placeholders_are_insert_only_and_tolerate_conflicts(
        self, global_config, embed_func, mock_client
    ):
        mock_client.mget = AsyncMock(side_effect=node_mget_side_effect(set()))
        bulk_calls = []

        async def capture_bulk(_client, actions, *args, **kwargs):
            acts = list(actions)
            bulk_calls.append(acts)
            node_acts = [a for a in acts if a["_index"].endswith("-nodes")]
            if node_acts:
                return (
                    len(acts) - 1,
                    [{"create": {"_id": node_acts[0]["_id"], "status": 409}}],
                )
            return (len(acts), [])

        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk",
                new=AsyncMock(side_effect=capture_bulk),
            ):
                await s.upsert_edges_batch([("A", "B", {"weight": "1.0"})])

        node_acts = [
            a for call in bulk_calls for a in call if a["_index"] == s._nodes_index
        ]
        assert sorted(a["_id"] for a in node_acts) == ["A", "B"]
        assert all(a["_op_type"] == "create" for a in node_acts)

    @pytest.mark.asyncio
    async def test_batch_placeholder_non_conflict_error_fails_the_write(
        self, global_config, embed_func, mock_client
    ):
        mock_client.mget = AsyncMock(side_effect=node_mget_side_effect(set()))

        async def capture_bulk(_client, actions, *args, **kwargs):
            acts = list(actions)
            if any(a["_index"].endswith("-nodes") for a in acts):
                return (0, [{"create": {"_id": "A", "status": 503}}])
            return (len(acts), [])

        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk",
                new=AsyncMock(side_effect=capture_bulk),
            ):
                with pytest.raises(RuntimeError, match="endpoint placeholder"):
                    await s.upsert_edges_batch([("A", "B", {"weight": "1.0"})])


# ---------------------------------------------------------------------------
# BFS level tie-breaking & subgraph truncation / dangling-endpoint semantics
# ---------------------------------------------------------------------------


_TIE_BREAK_EDGES = [
    {"source_node_id": "A", "target_node_id": "X"},
    {"source_node_id": "A", "target_node_id": "Y"},
    {"source_node_id": "A", "target_node_id": "Z"},
    {"source_node_id": "Z", "target_node_id": "P"},
    {"source_node_id": "Z", "target_node_id": "Q"},
    {"source_node_id": "X", "target_node_id": "P"},
    {"source_node_id": "Y", "target_node_id": "Q"},
]


def _bfs_search_side_effect(edges):
    """Answer client-side BFS edge scan, degree aggregation, and final edge fetch."""

    async def _search(index=None, body=None, **kwargs):
        bool_query = body["query"]["bool"]
        if "aggs" in body:
            ids = set(bool_query["should"][0]["terms"]["source_node_id"])
            matching = [
                e
                for e in edges
                if e["source_node_id"] in ids or e["target_node_id"] in ids
            ]

            def _agg(name, field):
                spec = body["aggs"][name]
                docs, filtered = matching, "filter" in spec
                if filtered:
                    allowed = set(spec["filter"]["terms"][field])
                    docs = [e for e in matching if e[field] in allowed]
                    inner = spec["aggs"]["ids"]["terms"]
                else:
                    inner = spec["terms"]
                counts = Counter(e[field] for e in docs)
                buckets = [
                    {"key": key, "doc_count": count}
                    for key, count in counts.most_common(inner["size"])
                ]
                return (
                    {"ids": {"buckets": buckets}} if filtered else {"buckets": buckets}
                )

            return {
                "hits": {"hits": []},
                "aggregations": {
                    "source_degrees": _agg("source_degrees", "source_node_id"),
                    "target_degrees": _agg("target_degrees", "target_node_id"),
                },
            }
        if "should" in bool_query:
            ids = set(bool_query["should"][0]["terms"]["source_node_id"])
            hits = [
                {"_source": e}
                for e in edges
                if e["source_node_id"] in ids or e["target_node_id"] in ids
            ]
        else:
            ids = set(bool_query["must"][0]["terms"]["source_node_id"])
            hits = [
                {"_source": e}
                for e in edges
                if e["source_node_id"] in ids and e["target_node_id"] in ids
            ]
        return {"hits": {"hits": hits}}

    return _search


def _make_bare_graph_storage(
    edges=_TIE_BREAK_EDGES,
    real_nodes: dict[str, dict] | set[str] | None = None,
) -> OpenSearchGraphStorage:
    if real_nodes is None:
        node_sources = {
            nid: {"entity_type": "person"} for nid in ("A", "X", "Y", "Z", "P", "Q")
        }
    elif isinstance(real_nodes, set):
        node_sources = {nid: {"entity_type": "person"} for nid in real_nodes}
    else:
        node_sources = real_nodes

    storage = OpenSearchGraphStorage.__new__(OpenSearchGraphStorage)
    storage.workspace = "test"
    storage.global_config = {"max_graph_nodes": 1000}
    storage._nodes_index = "test-nodes"
    storage._edges_index = "test-edges"
    storage._indices_ready = True
    storage._ppl_graphlookup_available = False
    storage._refresh_graph_indices_if_dirty = AsyncMock(return_value=None)
    storage.client = AsyncMock()
    storage.client.search = AsyncMock(side_effect=_bfs_search_side_effect(edges))
    storage.client.mget = AsyncMock(side_effect=mget_by_ids_side_effect(node_sources))
    storage.client.get = AsyncMock(
        return_value={"_id": "A", "_source": {"entity_type": "person"}}
    )
    storage.client.create_pit = AsyncMock(return_value={"pit_id": "pit-1"})
    storage.client.delete_pit = AsyncMock(return_value=None)
    return storage


def _make_ppl_graph_storage(edges, real_ids):
    storage = _make_bare_graph_storage(real_nodes=set(real_ids))
    storage._ppl_graphlookup_available = True
    storage.client.transport = AsyncMock()
    storage.client.transport.perform_request = AsyncMock(
        return_value={
            "schema": [{"name": "connected_edges"}],
            "datarows": [[[dict(edge, _depth=depth) for depth, edge in edges]]],
        }
    )
    return storage


class TestGraphBFSTieBreakAndSubgraph:
    """BFS level ranking (degree desc, id asc), caps, truncation, and dangling endpoints."""

    @pytest.mark.asyncio
    async def test_bfs_level_admits_by_degree_then_id(self):
        storage = _make_bare_graph_storage()
        result = await storage.get_knowledge_graph("A", max_depth=2, max_nodes=3)
        assert sorted(node.id for node in result.nodes) == ["A", "X", "Z"]
        assert result.is_truncated is True

    @pytest.mark.asyncio
    async def test_bfs_level_that_fits_keeps_every_node(self):
        storage = _make_bare_graph_storage()
        result = await storage.get_knowledge_graph("A", max_depth=1, max_nodes=10)
        assert sorted(node.id for node in result.nodes) == ["A", "X", "Y", "Z"]
        assert result.is_truncated is False

    @pytest.mark.asyncio
    async def test_ppl_degree_lookup_skips_levels_that_cannot_reach_the_cap(self):
        edges = [
            (1, {"source_node_id": "A", "target_node_id": f"n{i}"}) for i in range(3)
        ]
        edges += [
            (2, {"source_node_id": "n0", "target_node_id": f"d{i}"}) for i in range(500)
        ]
        storage = _make_ppl_graph_storage(edges, {"A"} | {f"n{i}" for i in range(3)})
        storage.node_degrees_batch = AsyncMock(return_value={})

        await storage.get_knowledge_graph("A", max_depth=3, max_nodes=3)

        ranked = storage.node_degrees_batch.await_args.args[0]
        assert set(ranked) == {"n0", "n1", "n2"}

    @pytest.mark.asyncio
    async def test_ppl_degree_lookup_is_capped_on_a_single_wide_level(self):
        width = _GRAPH_DEGREE_RANK_MAX_CANDIDATES + 500
        edges = [
            (1, {"source_node_id": "A", "target_node_id": f"n{i}"})
            for i in range(width)
        ]
        storage = _make_ppl_graph_storage(edges, {"A"})
        storage.node_degrees_batch = AsyncMock(return_value={})

        await storage.get_knowledge_graph("A", max_depth=1, max_nodes=1000)

        ranked = storage.node_degrees_batch.await_args.args[0]
        assert len(ranked) == _GRAPH_DEGREE_RANK_MAX_CANDIDATES

    @pytest.mark.asyncio
    async def test_bfs_degree_lookup_is_capped_on_a_single_wide_level(self):
        width = _GRAPH_DEGREE_RANK_MAX_CANDIDATES + 500
        neighbours = [f"n{i}" for i in range(width)]
        edges = [{"source_node_id": "A", "target_node_id": n} for n in neighbours]

        storage = _make_bare_graph_storage(edges=edges, real_nodes={"A", *neighbours})
        storage.node_degrees_batch = AsyncMock(return_value={})

        await storage.get_knowledge_graph("A", max_depth=1, max_nodes=1000)

        ranked = storage.node_degrees_batch.await_args.args[0]
        assert len(ranked) == _GRAPH_DEGREE_RANK_MAX_CANDIDATES

    @pytest.mark.asyncio
    async def test_degree_aggregation_confines_its_buckets_to_the_requested_ids(self):
        edges = [
            {"source_node_id": f"s{i}", "target_node_id": "H"}
            for i in range(50)
            for _ in range(2)
        ]
        edges.append({"source_node_id": "L", "target_node_id": "t"})

        storage = _make_bare_graph_storage(edges=edges)
        degrees = await storage.node_degrees_batch(["L", "H"])

        assert degrees["L"] == 1
        assert degrees["H"] == 100

    @pytest.mark.asyncio
    async def test_bfs_skips_the_degree_lookup_once_the_cap_is_full(self):
        storage = _make_bare_graph_storage()
        storage.node_degrees_batch = AsyncMock(return_value={})

        result = await storage.get_knowledge_graph("A", max_depth=3, max_nodes=4)

        storage.node_degrees_batch.assert_not_awaited()
        assert sorted(node.id for node in result.nodes) == ["A", "X", "Y", "Z"]
        assert result.is_truncated is True

    @pytest.mark.asyncio
    async def test_bfs_stops_expanding_once_truncation_is_proven(self):
        storage = _make_bare_graph_storage()
        await storage.get_knowledge_graph("A", max_depth=3, max_nodes=3)
        assert storage.client.mget.await_count == 2

    @pytest.mark.asyncio
    async def test_ppl_skips_the_degree_lookup_when_no_slots_remain(self):
        edges = [
            (1, {"source_node_id": "A", "target_node_id": f"n{i}"}) for i in range(5)
        ]
        storage = _make_ppl_graph_storage(edges, {"A"} | {f"n{i}" for i in range(5)})
        storage.node_degrees_batch = AsyncMock(return_value={})

        result = await storage.get_knowledge_graph("A", max_depth=2, max_nodes=1)

        storage.node_degrees_batch.assert_not_awaited()
        assert [node.id for node in result.nodes] == ["A"]
        assert result.is_truncated is True

    @pytest.mark.asyncio
    async def test_bfs_subgraph_transient_error_raises_not_reports_false_complete(
        self, global_config
    ):
        """A mid-BFS transport error must raise rather than return a partial subgraph
        with ``is_truncated=False``."""
        client = make_strict_client()
        client.indices.exists = AsyncMock(return_value=False)
        client.transport = AsyncMock()
        client.transport.perform_request = AsyncMock(side_effect=Exception("no PPL"))
        with patch.object(ClientManager, "get_client", return_value=client):
            storage = OpenSearchGraphStorage(
                namespace="chunk_entity_relation",
                global_config=global_config,
                embedding_func=RejectingEmbedFunc(),
                workspace="strictws",
            )
            await storage.initialize()
        storage._ppl_graphlookup_available = False
        client.mget = AsyncMock(
            return_value={
                "docs": [
                    {
                        "_id": "start",
                        "found": True,
                        "_source": {"entity_type": "person"},
                    }
                ]
            }
        )
        client.search = AsyncMock(side_effect=transient_error())
        with pytest.raises(TransportError):
            await storage.get_knowledge_graph("start", max_depth=2, max_nodes=100)

    @pytest.mark.asyncio
    async def test_bfs_subgraph_counter_example_not_falsely_truncated(self):
        real_nodes = {n: {"entity_type": "person"} for n in ["A", "B", "C"]}
        edges = [
            {"source_node_id": "A", "target_node_id": "B"},
            {"source_node_id": "A", "target_node_id": "C"},
        ]
        storage = _make_bare_graph_storage(edges=edges, real_nodes=real_nodes)

        result = await storage.get_knowledge_graph("A", max_depth=2, max_nodes=3)

        assert {n.id for n in result.nodes} == {"A", "B", "C"}
        assert result.is_truncated is False

    @pytest.mark.asyncio
    async def test_bfs_subgraph_diamond_not_truncated(self):
        real_nodes = {n: {"entity_type": "person"} for n in ["A", "B", "C", "D"]}
        edges = [
            {"source_node_id": "A", "target_node_id": "B"},
            {"source_node_id": "A", "target_node_id": "C"},
            {"source_node_id": "B", "target_node_id": "D"},
            {"source_node_id": "C", "target_node_id": "D"},
        ]
        storage = _make_bare_graph_storage(edges=edges, real_nodes=real_nodes)

        result = await storage.get_knowledge_graph("A", max_depth=2, max_nodes=4)

        assert {n.id for n in result.nodes} == {"A", "B", "C", "D"}
        assert result.is_truncated is False

    @pytest.mark.asyncio
    async def test_bfs_subgraph_star_reports_truncated_and_respects_cap(self):
        leaves = ["B", "C", "D", "E", "F"]
        real_nodes = {n: {"entity_type": "person"} for n in ["A"] + leaves}
        edges = [{"source_node_id": "A", "target_node_id": leaf} for leaf in leaves]
        storage = _make_bare_graph_storage(edges=edges, real_nodes=real_nodes)

        result = await storage.get_knowledge_graph("A", max_depth=2, max_nodes=3)

        assert result.is_truncated is True
        assert len(result.nodes) <= 3

    @pytest.mark.asyncio
    async def test_bfs_subgraph_dangling_only_neighbor_not_falsely_truncated(self):
        real_nodes = {"A": {"entity_type": "person"}}
        edges = [{"source_node_id": "A", "target_node_id": "X"}]
        storage = _make_bare_graph_storage(edges=edges, real_nodes=real_nodes)

        result = await storage.get_knowledge_graph("A", max_depth=2, max_nodes=1)

        assert {n.id for n in result.nodes} == {"A"}
        assert result.is_truncated is False

    @pytest.mark.asyncio
    async def test_bfs_subgraph_dangling_candidate_does_not_steal_real_node_slot(self):
        real_nodes = {"A": {"entity_type": "person"}, "B": {"entity_type": "person"}}
        edges = [
            {"source_node_id": "A", "target_node_id": "X"},
            {"source_node_id": "A", "target_node_id": "B"},
        ]
        storage = _make_bare_graph_storage(edges=edges, real_nodes=real_nodes)

        result = await storage.get_knowledge_graph("A", max_depth=2, max_nodes=2)

        assert {n.id for n in result.nodes} == {"A", "B"}
        assert result.is_truncated is False


# ---------------------------------------------------------------------------
# Batch edge degrees (edge_degrees_batch & node_degrees_batch edge cases)
# ---------------------------------------------------------------------------


class TestGraphEdgeDegreesBatch:
    """``OpenSearchGraphStorage.edge_degrees_batch`` resolves distinct ids once."""

    _PAIRS = [("A", "B"), ("B", "C"), ("A", "C")]

    @pytest.mark.asyncio
    async def test_resolves_distinct_ids_in_one_batch_call(self):
        s = _make_bare_graph_storage()
        s.node_degrees_batch = AsyncMock(return_value={"A": 2, "B": 2, "C": 2})
        s.node_degree = AsyncMock(side_effect=AssertionError("scalar must not be used"))

        result = await s.edge_degrees_batch(self._PAIRS)

        assert result == {("A", "B"): 4, ("B", "C"): 4, ("A", "C"): 4}
        s.node_degrees_batch.assert_awaited_once()
        assert sorted(s.node_degrees_batch.await_args.args[0]) == ["A", "B", "C"]
        s.node_degree.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_chunks_ids_so_a_large_batch_cannot_breach_the_terms_limit(self):
        n = _GRAPH_DEGREE_RANK_MAX_CANDIDATES * 2 + 1
        ids = [f"N{i}" for i in range(n)]
        pairs = [(ids[i], ids[i + 1]) for i in range(n - 1)]

        s = _make_bare_graph_storage()
        seen_chunks = []

        async def _batch(chunk):
            seen_chunks.append(list(chunk))
            return {nid: 1 for nid in chunk}

        s.node_degrees_batch = AsyncMock(side_effect=_batch)

        result = await s.edge_degrees_batch(pairs)

        assert len(seen_chunks) == 3, [len(c) for c in seen_chunks]
        assert max(len(c) for c in seen_chunks) <= _GRAPH_DEGREE_RANK_MAX_CANDIDATES
        flat = [nid for chunk in seen_chunks for nid in chunk]
        assert sorted(flat) == sorted(ids)
        assert set(result.values()) == {2}

    @pytest.mark.asyncio
    async def test_empty_input_issues_no_query(self):
        s = _make_bare_graph_storage()
        s.node_degrees_batch = AsyncMock(side_effect=AssertionError("must not query"))
        assert await s.edge_degrees_batch([]) == {}
        s.node_degrees_batch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_endpoint_counts_as_zero(self):
        s = _make_bare_graph_storage()
        s.node_degrees_batch = AsyncMock(return_value={"A": 3})
        assert await s.edge_degrees_batch([("A", "Ghost")]) == {("A", "Ghost"): 3}

    @pytest.mark.asyncio
    async def test_node_degrees_batch_answers_every_requested_id(self):
        s = _make_bare_graph_storage()
        s.client.search = AsyncMock(
            return_value={
                "aggregations": {
                    "source_degrees": {"ids": {"buckets": []}},
                    "target_degrees": {"ids": {"buckets": []}},
                }
            }
        )
        assert await s.node_degrees_batch(["A", "B"]) == {"A": 0, "B": 0}

    @pytest.mark.asyncio
    async def test_node_degrees_batch_answers_every_id_when_indices_are_not_ready(self):
        s = _make_bare_graph_storage()
        s._indices_ready = False

        assert await s.node_degrees_batch(["A", "B"]) == {"A": 0, "B": 0}
        s._refresh_graph_indices_if_dirty.assert_not_awaited()
        s.client.search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_node_degrees_batch_answers_every_id_when_edge_index_disappears(self):
        s = _make_bare_graph_storage()
        s.client.search = AsyncMock(
            side_effect=NotFoundError(
                404, "index_not_found_exception", "no such edge index"
            )
        )

        assert await s.node_degrees_batch(["A", "B"]) == {"A": 0, "B": 0}
        assert s._indices_ready is False


# ---------------------------------------------------------------------------
# _merge_edge_payloads weight & provenance contract
# ---------------------------------------------------------------------------


class TestMergeEdgePayloads:
    """``_merge_edge_payloads`` honors the relation weight floor and sums fragment weights."""

    def test_sums_every_fragment_weight_disjoint_and_same_source(self):
        disjoint = _merge_edge_payloads(
            [
                {"source_ids": ["c1"], "weight": 1.0},
                {"source_ids": ["c2"], "weight": 2.0},
            ]
        )
        assert disjoint["weight"] == 3.0
        assert disjoint["source_ids"] == ["c1", "c2"]

        same_source = _merge_edge_payloads(
            [
                {"source_ids": ["c1"], "weight": 1.0},
                {"source_ids": ["c1"], "weight": 2.0},
            ]
        )
        assert same_source["weight"] == 3.0
        assert same_source["source_ids"] == ["c1"]

    def test_fragment_with_missing_weight_does_not_undercut_evidence_floor(self):
        merged = _merge_edge_payloads(
            [
                {"source_ids": ["chunk1"], "weight": 1.0},
                {"source_ids": ["chunk2"]},
            ]
        )
        assert merged["source_ids"] == ["chunk1", "chunk2"]
        assert merged["weight"] >= 2

    def test_all_fragments_missing_weight_still_get_the_evidence_floor(self):
        merged = _merge_edge_payloads(
            [{"source_ids": ["chunk1"]}, {"source_ids": ["chunk2"]}]
        )
        assert merged["weight"] == 2

    def test_weight_sum_overflow_is_clamped_to_the_largest_finite_weight(self):
        merged = _merge_edge_payloads(
            [
                {"source_ids": ["chunk1"], "weight": 1e308},
                {"source_ids": ["chunk2"], "weight": 1e308},
            ]
        )
        assert merged["weight"] == sys.float_info.max

    @pytest.mark.parametrize(
        "bad_weight", [float("nan"), float("inf"), "NaN", "-Infinity"]
    )
    def test_non_finite_legacy_weight_is_skipped_not_propagated(self, bad_weight):
        merged = _merge_edge_payloads(
            [
                {"source_ids": ["chunk1"], "weight": bad_weight},
                {"source_ids": ["chunk2"]},
            ]
        )
        assert merged["weight"] == 2

    def test_unstorable_legacy_source_id_does_not_raise(self):
        merged = _merge_edge_payloads(
            [
                {"source_ids": ["chunk-\x0bbad"], "weight": 1.0},
                {"source_ids": ["chunk2"]},
            ]
        )
        assert merged["weight"] >= 2

    def test_placeholder_only_evidence_leaves_a_missing_weight_alone(self):
        merged = _merge_edge_payloads(
            [{"source_ids": ["UNKNOWN"]}, {"source_ids": ["manual_creation"]}]
        )
        assert "weight" not in merged
        assert merged["source_ids"] == ["UNKNOWN", "manual_creation"]

    def test_placeholder_only_evidence_keeps_a_real_weight(self):
        merged = _merge_edge_payloads(
            [
                {"source_ids": ["UNKNOWN"], "weight": 0.5},
                {"source_ids": ["manual_creation"]},
            ]
        )
        assert merged["weight"] == 0.5

    def test_negative_weight_sum_overflow_takes_the_evidence_floor(self):
        merged = _merge_edge_payloads(
            [
                {"source_ids": ["chunk1"], "weight": -1e308},
                {"source_ids": ["chunk2"], "weight": -1e308},
            ]
        )
        assert merged["weight"] == 2


# ---------------------------------------------------------------------------
# Self-loop listing contract
# ---------------------------------------------------------------------------


def _make_self_loop_storage(edges: list[dict]) -> OpenSearchGraphStorage:
    async def _search(*args, **kwargs):
        body = kwargs.get("body") or (args[0] if args else {})
        if body.get("search_after"):
            return {"hits": {"hits": []}}
        should = body["query"]["bool"]["should"]
        ids = set(should[0]["terms"]["source_node_id"])
        return {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "source_node_id": e["source_node_id"],
                            "target_node_id": e["target_node_id"],
                        },
                        "sort": [e["source_node_id"], e["target_node_id"]],
                    }
                    for e in edges
                    if e["source_node_id"] in ids or e["target_node_id"] in ids
                ]
            }
        }

    s = _make_bare_graph_storage()
    s.client.search = AsyncMock(side_effect=_search)
    return s


class TestGetNodesEdgesBatchSelfLoop:
    """A self-loop appears once per endpoint in ``get_nodes_edges_batch``."""

    _SELF_LOOP = [{"source_node_id": "Loop", "target_node_id": "Loop"}]
    _PLAIN = [
        {"source_node_id": "A", "target_node_id": "B"},
        {"source_node_id": "C", "target_node_id": "A"},
    ]

    @pytest.mark.asyncio
    async def test_self_loop_is_listed_once(self):
        s = _make_self_loop_storage(self._SELF_LOOP)
        result = await s.get_nodes_edges_batch(["Loop"])
        assert result["Loop"] == [("Loop", "Loop")]

    @pytest.mark.asyncio
    async def test_ordinary_edge_still_reaches_both_endpoints(self):
        s = _make_self_loop_storage(self._PLAIN)
        result = await s.get_nodes_edges_batch(["A", "B"])
        assert ("A", "B") in result["A"]
        assert ("A", "B") in result["B"]
        assert ("C", "A") in result["A"]

    @pytest.mark.asyncio
    async def test_self_loop_alongside_ordinary_edges(self):
        s = _make_self_loop_storage(
            self._PLAIN + [{"source_node_id": "A", "target_node_id": "A"}]
        )
        result = await s.get_nodes_edges_batch(["A"])
        assert result["A"].count(("A", "A")) == 1
        assert len(result["A"]) == 3


# ---------------------------------------------------------------------------
# Graph delete per-item results & concurrent refresh dirty-flag safety
# ---------------------------------------------------------------------------


class TestGraphDeleteResults:
    """Graph deletes honor per-item failures without rejecting success or absence."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["remove_edges", "remove_nodes"])
    @pytest.mark.parametrize("status", [200, 404, 403])
    async def test_graph_delete_item_results(self, monkeypatch, method, status):
        item = {"delete": {"_id": "row", "status": status}}
        storage = OpenSearchGraphStorage("graph", {}, None, "test")
        storage._indices_ready = True
        storage.client = SimpleNamespace(
            bulk=AsyncMock(return_value={"errors": status >= 400, "items": [item]}),
            delete_by_query=AsyncMock(return_value={"failures": []}),
        )
        monkeypatch.setattr(
            opensearch_module,
            "_run_chunked_async_bulk",
            AsyncMock(
                return_value=(int(status == 200), [] if status == 200 else [item])
            ),
        )
        values = [("A", "B")] if method == "remove_edges" else ["A", "B"]
        if status in (200, 404):
            await getattr(storage, method)(values)
        else:
            with pytest.raises(OpenSearchException, match="bulk deletion failed"):
                await getattr(storage, method)(values)
        assert storage._edges_dirty
        assert storage._nodes_dirty == (method == "remove_nodes")

    @pytest.mark.asyncio
    async def test_delete_node_marks_views_dirty_when_node_delete_fails(self):
        storage = OpenSearchGraphStorage("graph", {}, None, "test")
        storage._indices_ready = True
        storage.client = SimpleNamespace(
            delete_by_query=AsyncMock(return_value={"deleted": 1, "failures": []}),
            delete=AsyncMock(side_effect=OpenSearchException("unavailable")),
        )
        with pytest.raises(OpenSearchException, match="unavailable"):
            await storage.delete_node("A")
        assert storage._edges_dirty
        assert storage._nodes_dirty

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["remove_edges", "remove_nodes", "delete_node"])
    async def test_refresh_during_in_flight_delete_leaves_views_dirty(
        self, monkeypatch, method
    ):
        monkeypatch.setattr(
            opensearch_module,
            "get_data_init_lock",
            lambda: contextlib.AsyncExitStack(),
        )
        storage = OpenSearchGraphStorage("graph", {}, None, "test")
        storage._indices_ready = True
        refreshes: list[str] = []

        async def concurrent_read_refresh(*_args, **_kwargs):
            await storage._refresh_graph_indices_if_dirty(
                refresh_nodes=True, refresh_edges=True
            )
            return {
                "errors": False,
                "items": [{"delete": {"_id": "e", "status": 200}}],
            }

        storage.client = SimpleNamespace(
            bulk=AsyncMock(side_effect=concurrent_read_refresh),
            delete_by_query=AsyncMock(side_effect=concurrent_read_refresh),
            delete=AsyncMock(return_value={"result": "deleted"}),
            indices=SimpleNamespace(
                refresh=AsyncMock(side_effect=lambda index: refreshes.append(index))
            ),
        )

        async def node_bulk_with_concurrent_refresh(*args, **kwargs):
            await concurrent_read_refresh()
            return 1, []

        monkeypatch.setattr(
            opensearch_module,
            "_run_chunked_async_bulk",
            node_bulk_with_concurrent_refresh,
        )
        storage._nodes_dirty = True
        storage._edges_dirty = True

        if method == "remove_edges":
            await storage.remove_edges([("A", "B")])
        elif method == "remove_nodes":
            await storage.remove_nodes(["A"])
        else:
            await storage.delete_node("A")

        assert refreshes, "the concurrent refresh did not run"
        assert storage._edges_dirty
        assert storage._nodes_dirty == (method != "remove_edges")
