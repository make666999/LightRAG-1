"""Offline unit tests for MongoGraphStorage operations: attribute name validation,
BFS tie-breaking, legacy edge deduplication, batch read contracts, and self-loop listing.
"""

from __future__ import annotations

import sys
from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

pytest.importorskip("pymongo", reason="pymongo is required for Mongo storage tests")

from lightrag.kg.mongo_impl import (  # noqa: E402
    MongoGraphStorage,
    _canonical_edge_endpoints,
)

pytestmark = pytest.mark.offline


class _AsyncCursor:
    def __init__(self, docs):
        self._docs = list(docs)

    def __aiter__(self):
        self._iter = iter(self._docs)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


def _make_graph_storage(*, with_collections: bool = False) -> MongoGraphStorage:
    storage = MongoGraphStorage.__new__(MongoGraphStorage)
    storage.workspace = "test"
    storage.namespace = "chunk_entity_relation"
    storage.global_config = {"max_graph_nodes": 1000}
    storage._collection_name = "test_nodes"
    storage._edge_collection_name = "test_edges"
    storage._max_upsert_payload_bytes = 1_000_000
    storage._max_upsert_records_per_batch = 100
    if with_collections:
        storage.collection = SimpleNamespace(
            update_one=AsyncMock(), bulk_write=AsyncMock()
        )
        storage.edge_collection = SimpleNamespace(
            update_one=AsyncMock(), bulk_write=AsyncMock()
        )
    else:
        storage.edge_collection = Mock()
        storage.edge_collection.find = Mock()
    return storage


def _make_edge_aggregate_side_effect(edges: list[dict]):
    async def _aggregate(pipeline, **kwargs):
        match = pipeline[0]["$match"]
        field = "source_node_id" if "source_node_id" in match else "target_node_id"
        ids = set(match[field]["$in"])
        counts = Counter(e[field] for e in edges if e[field] in ids)
        return _AsyncCursor(
            [{"_id": key, "degree": count} for key, count in counts.items()]
        )

    return _aggregate


# ---------------------------------------------------------------------------
# Attribute name validation (field path / operator hazards)
# ---------------------------------------------------------------------------

UNSAFE_NAMES = [
    pytest.param("source_ids.0", id="dotted-array-element"),
    pytest.param("a.b", id="dotted"),
    pytest.param("$set", id="update-operator"),
    pytest.param("$where", id="query-operator"),
    pytest.param("", id="empty"),
]

LEGACY_STORED_NAMES = [
    pytest.param("display-name", id="dashed"),
    pytest.param("has space", id="space"),
    pytest.param("9legacy", id="leading-digit"),
]


def _assert_no_server_call(storage):
    storage.collection.update_one.assert_not_awaited()
    storage.collection.bulk_write.assert_not_awaited()
    storage.edge_collection.update_one.assert_not_awaited()
    storage.edge_collection.bulk_write.assert_not_awaited()


class TestUpsertNode:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", UNSAFE_NAMES)
    async def test_unsafe_name_is_refused(self, name):
        storage = _make_graph_storage(with_collections=True)

        with pytest.raises(ValueError, match="invalid attribute name"):
            await storage.upsert_node("n1", {"entity_id": "n1", name: "x"})

        _assert_no_server_call(storage)

    @pytest.mark.asyncio
    async def test_reserved_id_is_refused(self):
        storage = _make_graph_storage(with_collections=True)

        with pytest.raises(ValueError, match="reserved by MongoDB"):
            await storage.upsert_node("n1", {"entity_id": "n1", "_id": "elsewhere"})

        _assert_no_server_call(storage)

    @pytest.mark.asyncio
    async def test_ordinary_names_still_reach_the_server(self):
        storage = _make_graph_storage(with_collections=True)

        await storage.upsert_node(
            "n1",
            {
                "entity_id": "n1",
                "description": "d",
                "source_id": "chunk-1",
                "created_at": 1,
            },
        )

        storage.collection.update_one.assert_awaited_once()
        _filter, update = storage.collection.update_one.await_args.args
        assert update["$set"]["description"] == "d"
        assert update["$set"]["source_ids"] == ["chunk-1"]


class TestUpsertEdge:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", UNSAFE_NAMES)
    async def test_unsafe_name_is_refused(self, name):
        storage = _make_graph_storage(with_collections=True)

        with pytest.raises(ValueError, match="invalid attribute name"):
            await storage.upsert_edge("a", "b", {"weight": 1.0, name: "x"})

        _assert_no_server_call(storage)


class TestBatchesValidateBeforeWriting:
    @pytest.mark.asyncio
    async def test_nodes_batch_refuses_whole_batch(self):
        storage = _make_graph_storage(with_collections=True)
        nodes = [
            ("good", {"entity_id": "good", "description": "a"}),
            ("bad", {"entity_id": "bad", "source_ids.0": "x"}),
        ]

        with pytest.raises(ValueError, match="invalid attribute name"):
            await storage.upsert_nodes_batch(nodes)

        _assert_no_server_call(storage)

    @pytest.mark.asyncio
    async def test_edges_batch_refuses_before_creating_endpoints(self):
        storage = _make_graph_storage(with_collections=True)
        edges = [
            ("a", "b", {"weight": 1.0}),
            ("c", "d", {"weight": 1.0, "$set": "x"}),
        ]

        with pytest.raises(ValueError, match="invalid attribute name"):
            await storage.upsert_edges_batch(edges)

        _assert_no_server_call(storage)


class TestLegacyStoredNamesStillWrite:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", LEGACY_STORED_NAMES)
    async def test_node_rewrite_carrying_a_legacy_name_succeeds(self, name):
        storage = _make_graph_storage(with_collections=True)

        await storage.upsert_node(
            "n1", {"entity_id": "n1", name: "legacy", "description": "updated"}
        )

        storage.collection.update_one.assert_awaited_once()
        _filter, update = storage.collection.update_one.await_args.args
        assert update["$set"][name] == "legacy"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", LEGACY_STORED_NAMES)
    async def test_edge_rewrite_carrying_a_legacy_name_succeeds(self, name):
        storage = _make_graph_storage(with_collections=True)

        await storage.upsert_edge("a", "b", {"weight": 1.0, name: "legacy"})

        storage.edge_collection.update_one.assert_awaited_once()
        _filter, update = storage.edge_collection.update_one.await_args.args
        assert update["$set"][name] == "legacy"

    @pytest.mark.asyncio
    async def test_a_dollar_sign_inside_the_name_is_accepted(self):
        storage = _make_graph_storage(with_collections=True)

        await storage.upsert_node("n1", {"entity_id": "n1", "cost$usd": "5"})

        _filter, update = storage.collection.update_one.await_args.args
        assert update["$set"]["cost$usd"] == "5"


class TestNoValidationToMutationWindow:
    @pytest.mark.asyncio
    async def test_a_mutation_during_the_endpoint_write_does_not_reach_mongo(self):
        storage = _make_graph_storage(with_collections=True)
        payload = {"weight": 1.0, "description": "clean"}

        async def poison_during_bulk_write(*args, **kwargs):
            payload["source_ids.0"] = "attribution rewritten"

        storage.collection.bulk_write = AsyncMock(side_effect=poison_during_bulk_write)

        await storage.upsert_edge("a", "b", payload)

        storage.edge_collection.update_one.assert_awaited_once()
        _filter, update = storage.edge_collection.update_one.await_args.args
        assert "source_ids.0" not in update["$set"]
        assert update["$set"]["description"] == "clean"

    @pytest.mark.asyncio
    async def test_the_batch_path_uses_the_validated_snapshots(self, monkeypatch):
        storage = _make_graph_storage(with_collections=True)
        payload = {"weight": 1.0, "description": "clean"}
        captured: list = []

        async def fake_bulk(collection, ops, **kwargs):
            if not captured:
                payload["source_ids.0"] = "attribution rewritten"
                captured.append("endpoints")
            else:
                captured.append(ops)

        monkeypatch.setattr("lightrag.kg.mongo_impl._run_batched_bulk_write", fake_bulk)

        await storage.upsert_edges_batch([("a", "b", payload)])

        edge_ops = captured[-1]
        assert edge_ops, "edge ops were never written"
        update = edge_ops[0][0]._doc
        assert "source_ids.0" not in update["$set"]
        assert update["$set"]["description"] == "clean"


# ---------------------------------------------------------------------------
# Bidirectional BFS level ranking & tie-break
# ---------------------------------------------------------------------------

_BFS_EDGES = [
    {"source_node_id": "A", "target_node_id": "X"},
    {"source_node_id": "A", "target_node_id": "Y"},
    {"source_node_id": "A", "target_node_id": "Z"},
    {"source_node_id": "Z", "target_node_id": "P"},
    {"source_node_id": "Z", "target_node_id": "Q"},
    {"source_node_id": "X", "target_node_id": "P"},
    {"source_node_id": "Y", "target_node_id": "Q"},
]
_BFS_REAL_IDS = {"A", "X", "Y", "Z", "P", "Q"}


class TestMongoBfsTieBreak:
    @staticmethod
    def _bfs_storage(edges=_BFS_EDGES, real_ids=_BFS_REAL_IDS):
        def _node_find(query, projection=None):
            return _AsyncCursor(
                [
                    {"_id": nid, "entity_type": "person"}
                    for nid in query["_id"]["$in"]
                    if nid in real_ids
                ]
            )

        def _edge_find(query):
            if "$or" in query:
                ids = set(query["$or"][0]["source_node_id"]["$in"])
                return _AsyncCursor(
                    [
                        e
                        for e in edges
                        if e["source_node_id"] in ids or e["target_node_id"] in ids
                    ]
                )
            ids = set(query["$and"][0]["source_node_id"]["$in"])
            return _AsyncCursor(
                [
                    e
                    for e in edges
                    if e["source_node_id"] in ids and e["target_node_id"] in ids
                ]
            )

        storage = _make_graph_storage()
        storage.collection = SimpleNamespace(find=Mock(side_effect=_node_find))
        storage.edge_collection = SimpleNamespace(
            find=Mock(side_effect=_edge_find),
            aggregate=AsyncMock(side_effect=_make_edge_aggregate_side_effect(edges)),
        )
        return storage

    @pytest.mark.asyncio
    async def test_bfs_level_admits_by_degree_then_id_and_projects_away_source_ids(
        self,
    ):
        storage = self._bfs_storage()

        result = await storage.get_knowledge_subgraph_bidirectional_bfs(
            "A", 0, max_depth=1, max_nodes=3
        )

        assert sorted(node.id for node in result.nodes) == ["A", "X", "Z"]
        assert result.is_truncated is True
        assert storage.collection.find.call_args_list
        for call in storage.collection.find.call_args_list:
            assert call.args[1:] == ({"source_ids": 0},)

    @pytest.mark.asyncio
    async def test_bfs_level_that_fits_keeps_every_node(self):
        storage = self._bfs_storage()

        result = await storage.get_knowledge_subgraph_bidirectional_bfs(
            "A", 0, max_depth=1, max_nodes=10
        )

        assert sorted(node.id for node in result.nodes) == ["A", "X", "Y", "Z"]
        assert result.is_truncated is False

    @pytest.mark.asyncio
    async def test_bfs_degree_lookup_is_capped_on_a_single_wide_level(self):
        width = 8692
        neighbours = [f"n{i}" for i in range(width)]
        edges = [{"source_node_id": "A", "target_node_id": n} for n in neighbours]
        storage = self._bfs_storage(edges=edges, real_ids={"A", *neighbours})
        storage.node_degrees_batch = AsyncMock(return_value={})

        await storage.get_knowledge_subgraph_bidirectional_bfs(
            "A", 0, max_depth=1, max_nodes=1000
        )

        ranked = storage.node_degrees_batch.await_args.args[0]
        assert len(ranked) < width
        from lightrag.kg.mongo_impl import _GRAPH_DEGREE_RANK_MAX_CANDIDATES

        assert len(ranked) == _GRAPH_DEGREE_RANK_MAX_CANDIDATES


# ---------------------------------------------------------------------------
# Legacy reciprocal-edge deduplication & relation weight floor
# ---------------------------------------------------------------------------


class TestMongoDedupeLegacyEdges:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("docs", "expected_weight", "expected_sources"),
        [
            pytest.param(
                [
                    {
                        "_id": "survivor",
                        "source_ids": ["chunk1"],
                        "weight": 1.0,
                        "created_at": 2,
                    },
                    {"_id": "dup", "source_ids": ["chunk2"], "created_at": 1},
                ],
                2,
                ["chunk1", "chunk2"],
                id="duplicate_missing_weight_hits_evidence_floor",
            ),
            pytest.param(
                [
                    {"_id": "survivor", "source_ids": ["chunk1"], "created_at": 2},
                    {"_id": "dup", "source_ids": ["chunk2"], "created_at": 1},
                ],
                2,
                ["chunk1", "chunk2"],
                id="all_missing_weight_gets_evidence_floor",
            ),
            pytest.param(
                [
                    {
                        "_id": "survivor",
                        "source_ids": ["chunk1"],
                        "weight": 1e308,
                        "created_at": 2,
                    },
                    {
                        "_id": "dup",
                        "source_ids": ["chunk2"],
                        "weight": 1e308,
                        "created_at": 1,
                    },
                ],
                sys.float_info.max,
                ["chunk1", "chunk2"],
                id="positive_overflow_clamped_to_max_finite",
            ),
            pytest.param(
                [
                    {
                        "_id": "survivor",
                        "source_ids": ["chunk1"],
                        "weight": float("nan"),
                        "created_at": 2,
                    },
                    {"_id": "dup", "source_ids": ["chunk2"], "created_at": 1},
                ],
                2,
                ["chunk1", "chunk2"],
                id="nan_weight_skipped_for_evidence_floor",
            ),
            pytest.param(
                [
                    {
                        "_id": "survivor",
                        "source_ids": ["chunk-\x0bbad"],
                        "weight": 1.0,
                        "created_at": 2,
                    },
                    {"_id": "dup", "source_ids": ["chunk2"], "created_at": 1},
                ],
                2,
                ["chunk-\x0bbad", "chunk2"],
                id="unstorable_legacy_source_id_does_not_abort",
            ),
            pytest.param(
                [
                    {"_id": "survivor", "source_ids": ["UNKNOWN"], "created_at": 2},
                    {
                        "_id": "dup",
                        "source_ids": ["manual_creation"],
                        "created_at": 1,
                    },
                ],
                None,
                ["UNKNOWN", "manual_creation"],
                id="placeholder_only_missing_weight_left_unset",
            ),
            pytest.param(
                [
                    {
                        "_id": "survivor",
                        "source_ids": ["UNKNOWN"],
                        "weight": 1.0,
                        "created_at": 2,
                    },
                    {
                        "_id": "dup",
                        "source_ids": ["manual_creation"],
                        "created_at": 1,
                    },
                ],
                1.0,
                ["UNKNOWN", "manual_creation"],
                id="placeholder_only_keeps_real_weight",
            ),
            pytest.param(
                [
                    {
                        "_id": "survivor",
                        "source_ids": ["chunk1"],
                        "weight": -1e308,
                        "created_at": 2,
                    },
                    {
                        "_id": "dup",
                        "source_ids": ["chunk2"],
                        "weight": -1e308,
                        "created_at": 1,
                    },
                ],
                2,
                ["chunk1", "chunk2"],
                id="negative_overflow_takes_evidence_floor",
            ),
        ],
    )
    async def test_dedupe_legacy_edges_weight_contract(
        self, docs, expected_weight, expected_sources
    ):
        storage = MongoGraphStorage.__new__(MongoGraphStorage)
        storage.workspace = "test"
        group = {"_id": {"lo": "A", "hi": "B"}, "docs": docs, "count": len(docs)}
        storage.edge_collection = SimpleNamespace(
            aggregate=AsyncMock(return_value=_AsyncCursor([group])),
            update_one=AsyncMock(),
            delete_many=AsyncMock(),
        )

        removed = await storage._dedupe_legacy_edges()

        assert removed == 1
        set_fields = storage.edge_collection.update_one.await_args[0][1]["$set"]
        assert set_fields["source_ids"] == expected_sources
        if expected_weight is None:
            assert "weight" not in set_fields
        else:
            assert set_fields["weight"] == expected_weight


# ---------------------------------------------------------------------------
# Batch read contract: node_degrees_batch, edge_degrees_batch, get_edges_batch
# ---------------------------------------------------------------------------


def _make_edge_find_side_effect(docs: list[dict]):
    def _find(query, *args, **kwargs):
        wanted = {(c["edge_lo"], c["edge_hi"]) for c in query["$or"]}
        matches = [d for d in docs if (d["edge_lo"], d["edge_hi"]) in wanted]
        return _AsyncCursor(matches)

    return _find


class TestNodeDegreesBatch:
    @pytest.mark.asyncio
    async def test_sums_outbound_and_inbound_degree_and_seeds_isolated_nodes_to_zero(
        self,
    ):
        s = _make_graph_storage()
        edges = [
            {"source_node_id": "A", "target_node_id": "B"},
            {"source_node_id": "A", "target_node_id": "C"},
            {"source_node_id": "D", "target_node_id": "A"},
        ]
        s.edge_collection.aggregate = AsyncMock(
            side_effect=_make_edge_aggregate_side_effect(edges)
        )

        result = await s.node_degrees_batch(["A", "Isolated"])

        assert result == {"A": 3, "Isolated": 0}
        assert s.edge_collection.aggregate.await_count == 2

    @pytest.mark.asyncio
    async def test_isolated_nodes_only_get_zero(self):
        s = _make_graph_storage()
        s.edge_collection.aggregate = AsyncMock(
            side_effect=[_AsyncCursor([]), _AsyncCursor([])]
        )

        assert await s.node_degrees_batch(["A", "B"]) == {"A": 0, "B": 0}

    @pytest.mark.asyncio
    async def test_chunks_in_when_ids_exceed_the_chunk_size(self):
        s = _make_graph_storage()
        node_ids = [f"n{i}" for i in range(5)]
        edges = [{"source_node_id": nid, "target_node_id": "sink"} for nid in node_ids]
        s.edge_collection.aggregate = AsyncMock(
            side_effect=_make_edge_aggregate_side_effect(edges)
        )

        with patch("lightrag.kg.mongo_impl._NODE_DEGREES_BATCH_CHUNK_SIZE", 2):
            result = await s.node_degrees_batch(node_ids)

        assert s.edge_collection.aggregate.await_count == 6
        for call in s.edge_collection.aggregate.await_args_list:
            match = call.args[0][0]["$match"]
            field = next(iter(match))
            assert len(match[field]["$in"]) <= 2
        assert result == {nid: 1 for nid in node_ids}

    @pytest.mark.asyncio
    async def test_dedupes_ids_before_chunking(self):
        s = _make_graph_storage()
        s.edge_collection.aggregate = AsyncMock(
            side_effect=_make_edge_aggregate_side_effect([])
        )

        with patch("lightrag.kg.mongo_impl._NODE_DEGREES_BATCH_CHUNK_SIZE", 2):
            await s.node_degrees_batch(["A", "A", "A"])

        assert s.edge_collection.aggregate.await_count == 2

    @pytest.mark.asyncio
    async def test_chunks_in_by_payload_bytes_even_under_the_count_cap(self):
        s = _make_graph_storage()
        node_ids = [f"very-long-entity-id-{i:040d}" for i in range(5)]
        edges = [{"source_node_id": nid, "target_node_id": "sink"} for nid in node_ids]
        s.edge_collection.aggregate = AsyncMock(
            side_effect=_make_edge_aggregate_side_effect(edges)
        )

        with patch("lightrag.kg.mongo_impl._GRAPH_BATCH_QUERY_MAX_PAYLOAD_BYTES", 80):
            result = await s.node_degrees_batch(node_ids)

        assert s.edge_collection.aggregate.await_count == 10
        for call in s.edge_collection.aggregate.await_args_list:
            match = call.args[0][0]["$match"]
            field = next(iter(match))
            assert len(match[field]["$in"]) == 1
        assert result == {nid: 1 for nid in node_ids}


class TestEdgeDegreesBatch:
    @pytest.mark.asyncio
    async def test_skips_query_for_empty_input(self):
        s = _make_graph_storage()
        s.edge_collection.aggregate = AsyncMock()

        assert await s.edge_degrees_batch([]) == {}
        s.edge_collection.aggregate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_sums_source_and_target_degree_in_one_round_trip_pair(self):
        s = _make_graph_storage()
        edges = [
            {"source_node_id": "A", "target_node_id": "B"},
            {"source_node_id": "A", "target_node_id": "C"},
            {"source_node_id": "A", "target_node_id": "D"},
        ]
        s.edge_collection.aggregate = AsyncMock(
            side_effect=_make_edge_aggregate_side_effect(edges)
        )

        result = await s.edge_degrees_batch([("A", "B"), ("A", "D"), ("A", "Ghost")])

        assert result == {("A", "B"): 4, ("A", "D"): 4, ("A", "Ghost"): 3}
        assert s.edge_collection.aggregate.await_count == 2


class TestGetEdgesBatch:
    @pytest.mark.asyncio
    async def test_skips_query_for_empty_input(self):
        s = _make_graph_storage()

        assert await s.get_edges_batch([]) == {}
        s.edge_collection.find.assert_not_called()

    @pytest.mark.asyncio
    async def test_omits_missing_edges(self):
        s = _make_graph_storage()
        s.edge_collection.find.return_value = _AsyncCursor([])

        assert await s.get_edges_batch([{"src": "A", "tgt": "B"}]) == {}

        query = s.edge_collection.find.call_args.args[0]
        lo, hi = _canonical_edge_endpoints("A", "B")
        assert query == {"$or": [{"edge_lo": lo, "edge_hi": hi}]}

    @pytest.mark.asyncio
    async def test_returns_existing_edge_for_each_requested_direction(self):
        s = _make_graph_storage()
        lo, hi = _canonical_edge_endpoints("A", "B")
        doc = {"_id": "x", "edge_lo": lo, "edge_hi": hi, "weight": 1.0}
        s.edge_collection.find.return_value = _AsyncCursor([doc])

        result = await s.get_edges_batch([{"src": "A", "tgt": "B"}])

        assert result == {("A", "B"): {"edge_lo": lo, "edge_hi": hi, "weight": 1.0}}

    @pytest.mark.asyncio
    async def test_undirected_property_forward_and_reverse_return_independent_dicts(
        self,
    ):
        s = _make_graph_storage()
        lo, hi = _canonical_edge_endpoints("A", "B")
        doc = {"_id": "x", "edge_lo": lo, "edge_hi": hi, "weight": 2.0}
        s.edge_collection.find.return_value = _AsyncCursor([doc])

        result = await s.get_edges_batch(
            [{"src": "A", "tgt": "B"}, {"src": "B", "tgt": "A"}]
        )

        expected = {"edge_lo": lo, "edge_hi": hi, "weight": 2.0}
        assert result[("A", "B")] == expected
        assert result[("B", "A")] == expected
        assert result[("A", "B")] is not result[("B", "A")]
        result[("A", "B")]["source"] = "A"
        assert "source" not in result[("B", "A")]

    @pytest.mark.asyncio
    async def test_chunks_or_when_pairs_exceed_the_chunk_size(self):
        s = _make_graph_storage()
        pairs = [{"src": f"n{i}", "tgt": f"n{i + 1}"} for i in range(5)]
        docs = []
        for i, p in enumerate(pairs):
            lo, hi = _canonical_edge_endpoints(p["src"], p["tgt"])
            docs.append({"edge_lo": lo, "edge_hi": hi, "weight": float(i)})
        s.edge_collection.find = Mock(side_effect=_make_edge_find_side_effect(docs))

        with patch("lightrag.kg.mongo_impl._GET_EDGES_BATCH_CHUNK_SIZE", 2):
            result = await s.get_edges_batch(pairs)

        assert s.edge_collection.find.call_count == 3
        for call in s.edge_collection.find.call_args_list:
            assert len(call.args[0]["$or"]) <= 2
        for i, p in enumerate(pairs):
            assert result[(p["src"], p["tgt"])]["weight"] == float(i)

    @pytest.mark.asyncio
    async def test_single_find_call_when_pairs_fit_in_one_chunk(self):
        s = _make_graph_storage()
        pairs = [{"src": f"n{i}", "tgt": f"n{i + 1}"} for i in range(5)]
        s.edge_collection.find.return_value = _AsyncCursor([])

        await s.get_edges_batch(pairs)

        assert s.edge_collection.find.call_count == 1
        assert len(s.edge_collection.find.call_args.args[0]["$or"]) == 5

    @pytest.mark.asyncio
    async def test_chunks_or_by_payload_bytes_even_under_the_count_cap(self):
        s = _make_graph_storage()
        pairs = [
            {"src": f"very-long-entity-id-{i:040d}", "tgt": f"other-{i:040d}"}
            for i in range(5)
        ]
        docs = []
        for i, p in enumerate(pairs):
            lo, hi = _canonical_edge_endpoints(p["src"], p["tgt"])
            docs.append({"edge_lo": lo, "edge_hi": hi, "weight": float(i)})
        s.edge_collection.find = Mock(side_effect=_make_edge_find_side_effect(docs))

        with patch("lightrag.kg.mongo_impl._GRAPH_BATCH_QUERY_MAX_PAYLOAD_BYTES", 200):
            result = await s.get_edges_batch(pairs)

        assert s.edge_collection.find.call_count == 5
        for call in s.edge_collection.find.call_args_list:
            assert len(call.args[0]["$or"]) == 1
        for i, p in enumerate(pairs):
            assert result[(p["src"], p["tgt"])]["weight"] == float(i)


# ---------------------------------------------------------------------------
# Self-loop listing contract
# ---------------------------------------------------------------------------


class TestGetNodesEdgesBatchSelfLoop:
    _SELF_LOOP = [{"source_node_id": "Loop", "target_node_id": "Loop"}]
    _PLAIN = [
        {"source_node_id": "A", "target_node_id": "B"},
        {"source_node_id": "C", "target_node_id": "A"},
    ]

    @staticmethod
    def _self_loop_storage(edges: list[dict]):
        def _find(query, projection=None, **kwargs):
            if "$or" in query:
                node_id = query["$or"][0]["source_node_id"]
                return _AsyncCursor(
                    [
                        e
                        for e in edges
                        if node_id in (e["source_node_id"], e["target_node_id"])
                    ]
                )
            field = "source_node_id" if "source_node_id" in query else "target_node_id"
            ids = set(query[field]["$in"])
            return _AsyncCursor([e for e in edges if e[field] in ids])

        s = _make_graph_storage()
        s.edge_collection.find = Mock(side_effect=_find)
        return s

    @pytest.mark.asyncio
    async def test_self_loop_matches_get_node_edges(self):
        s = self._self_loop_storage(self._SELF_LOOP)
        s.has_node = AsyncMock(return_value=True)

        single = await s.get_node_edges("Loop")
        batch = await s.get_nodes_edges_batch(["Loop"])

        assert single == batch["Loop"] == [("Loop", "Loop")]

    @pytest.mark.asyncio
    async def test_ordinary_edge_still_reaches_both_endpoints(self):
        s = self._self_loop_storage(self._PLAIN)

        result = await s.get_nodes_edges_batch(["A", "B"])

        assert ("A", "B") in result["A"]
        assert ("A", "B") in result["B"]
        assert ("C", "A") in result["A"]

    @pytest.mark.asyncio
    async def test_self_loop_alongside_ordinary_edges(self):
        s = self._self_loop_storage(
            self._PLAIN + [{"source_node_id": "A", "target_node_id": "A"}]
        )

        result = await s.get_nodes_edges_batch(["A"])

        assert result["A"].count(("A", "A")) == 1
        assert len(result["A"]) == 3
