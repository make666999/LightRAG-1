"""Offline graph-query contract tests for ``Neo4JStorage``.

Covers:
- directed edge reads (``get_all_edges`` / ``iter_edges``), degree tie-breaking,
  and ``get_node_edges`` / ``get_edges_batch`` read contracts;
- single-walk APOC subgraph truncation without APOC's ``limit`` key;
- pure-Cypher ``_robust_fallback`` truncation reporting.
"""

from __future__ import annotations

import pytest

from lightrag.kg.neo4j_impl import Neo4JStorage
from tests.kg._cypher_fakes import (
    CypherGraphQueryContractMixin,
    FakeDriver,
    FakeResult,
)

pytestmark = pytest.mark.offline


class _FakeNode(dict):
    """Neo4j node stand-in supporting dict access, ``.id``, and ``._properties``."""

    def __init__(self, entity_id: str, node_id: int | None = None):
        props = {"entity_id": entity_id}
        super().__init__(props)
        self._properties = props
        if node_id is not None:
            self.id = node_id


class _FakeRel(dict):
    """Neo4j relationship stand-in supporting dict access and graph endpoints."""

    def __init__(
        self,
        rel_id: int = 0,
        start=None,
        end=None,
        rel_type: str = "DIRECTED",
    ):
        super().__init__(weight=1.0)
        self.id = rel_id
        self.type = rel_type
        self.start_node = start
        self.end_node = end


def _make_storage(
    driver: FakeDriver,
    *,
    namespace: str = "chunk_entity_relation",
    workspace: str = "test",
    database: str | None = "neo4j",
    global_config: dict | None = None,
) -> Neo4JStorage:
    storage = Neo4JStorage(
        namespace=namespace,
        global_config=(
            {"max_graph_nodes": 1000} if global_config is None else global_config
        ),
        embedding_func=None,
        workspace=workspace,
    )
    storage._driver = driver
    storage._DATABASE = database
    return storage


class TestNeo4jGraphQueries(CypherGraphQueryContractMixin):
    def make_storage(self, driver: FakeDriver) -> Neo4JStorage:
        return _make_storage(driver)

    def isolated_node_record(self, entity_id: str) -> dict:
        return {"n": _FakeNode(entity_id), "r": None, "connected": None}

    @pytest.mark.asyncio
    async def test_get_node_edges_returns_connected_pairs(self):
        storage = _make_storage(
            FakeDriver(
                records=[
                    {
                        "n": _FakeNode("Alpha"),
                        "r": object(),
                        "connected": _FakeNode("Beta"),
                    },
                    {
                        "n": _FakeNode("Alpha"),
                        "r": object(),
                        "connected": _FakeNode("Gamma"),
                    },
                ]
            )
        )

        assert await storage.get_node_edges("Alpha") == [
            ("Alpha", "Beta"),
            ("Alpha", "Gamma"),
        ]

    @pytest.mark.asyncio
    async def test_get_edges_batch_skips_query_for_empty_input(self):
        driver = FakeDriver(records=[])
        storage = _make_storage(driver)

        assert await storage.get_edges_batch([]) == {}
        assert driver.calls == []

    @pytest.mark.asyncio
    async def test_get_edges_batch_omits_missing_edges(self):
        driver = FakeDriver(records=[])
        storage = _make_storage(driver)

        assert await storage.get_edges_batch([{"src": "A", "tgt": "B"}]) == {}

        query, params = driver.calls[0]
        assert "-[r:DIRECTED]-" in query
        assert params == {"pairs": [{"src": "A", "tgt": "B"}]}

    @pytest.mark.asyncio
    async def test_get_edges_batch_returns_existing_edge_with_defaults(self):
        stored_properties = {"weight": 2.0, "description": "A to B"}
        storage = _make_storage(
            FakeDriver(
                records=[{"src_id": "A", "tgt_id": "B", "edges": [stored_properties]}]
            )
        )

        assert await storage.get_edges_batch([{"src": "A", "tgt": "B"}]) == {
            ("A", "B"): {
                "weight": 2.0,
                "source_id": None,
                "description": "A to B",
                "keywords": None,
            }
        }
        assert stored_properties == {"weight": 2.0, "description": "A to B"}


# ---------------------------------------------------------------------------
# APOC subgraph single-walk truncation
# ---------------------------------------------------------------------------


def _apoc_walk():
    """Six nodes in traversal order from n0, and the relationships among them."""
    n = [_FakeNode(f"e{i}", node_id=i) for i in range(6)]
    rels = [
        _FakeRel(100, n[0], n[1]),
        _FakeRel(101, n[2], n[1]),
        _FakeRel(102, n[0], n[3]),
        _FakeRel(103, n[3], n[4]),
        _FakeRel(104, n[4], n[5]),
        _FakeRel(105, n[2], n[5]),
    ]
    return {
        "node_info": [{"node": node} for node in n],
        "relationships": rels,
        "total_nodes": len(n),
    }


async def _get_apoc_graph(max_nodes: int):
    driver = FakeDriver(records=[_apoc_walk()])
    storage = _make_storage(driver, namespace="test", workspace="ws", database=None)
    graph = await storage.get_knowledge_graph("e0", max_depth=3, max_nodes=max_nodes)
    return graph, driver.queries


@pytest.mark.asyncio
async def test_oversized_subgraph_is_truncated_from_the_single_walk():
    graph, queries = await _get_apoc_graph(max_nodes=3)

    assert graph.is_truncated is True
    assert [node.id for node in graph.nodes] == ["0", "1", "2"]
    # Only relationships with both ends kept: n0-n1 and n2-n1.
    assert sorted(edge.id for edge in graph.edges) == ["100", "101"]
    assert all({edge.source, edge.target} <= {"0", "1", "2"} for edge in graph.edges)
    # One walk, and never APOC's `limit` key.
    subgraph_queries = [q for q in queries if "apoc.path.subgraphAll" in q]
    assert len(subgraph_queries) == 1
    assert "limit:" not in subgraph_queries[0]


@pytest.mark.asyncio
async def test_subgraph_within_max_nodes_is_returned_whole():
    graph, queries = await _get_apoc_graph(max_nodes=6)

    assert graph.is_truncated is False
    assert [node.id for node in graph.nodes] == ["0", "1", "2", "3", "4", "5"]
    assert len(graph.edges) == 6
    assert len([q for q in queries if "apoc.path.subgraphAll" in q]) == 1


# ---------------------------------------------------------------------------
# Pure-Cypher _robust_fallback BFS truncation
# ---------------------------------------------------------------------------


def _fallback_storage(node_docs: dict, edges_by_node: dict) -> Neo4JStorage:
    def _route(query: str, params: dict) -> FakeResult:
        entity_id = params.get("entity_id")
        if "RETURN id(n) as node_id, n" in query:
            node = node_docs.get(entity_id)
            return FakeResult([{"n": node, "node_id": 1}] if node else [])
        return FakeResult(edges_by_node.get(entity_id, []))

    return _make_storage(
        FakeDriver(handler=_route),
        namespace="test",
        workspace="ws",
        database=None,
        global_config={},
    )


def _fallback_edge(edge_id: str, target_entity_id: str) -> dict:
    return {
        "r": _FakeRel(rel_type="RELATED"),
        "b": _FakeNode(target_entity_id),
        "edge_id": edge_id,
    }


@pytest.mark.asyncio
async def test_robust_fallback_reports_truncated_over_cap():
    """Star graph (A + 5 leaves) with max_nodes=3 must be flagged truncated."""
    leaves = ["B", "C", "D", "E", "F"]
    node_docs = {n: _FakeNode(n) for n in ["A", *leaves]}
    edges_by_node = {
        "A": [_fallback_edge(f"e{i}", leaf) for i, leaf in enumerate(leaves)]
    }
    for i, leaf in enumerate(leaves):
        edges_by_node[leaf] = [_fallback_edge(f"e{i}", "A")]
    storage = _fallback_storage(node_docs, edges_by_node)

    result = await storage._robust_fallback("A", max_depth=5, max_nodes=3)

    assert result.is_truncated is True
    assert len(result.nodes) <= 3


@pytest.mark.asyncio
async def test_robust_fallback_duplicate_queue_entries_not_falsely_truncated():
    """Diamond graph (4 nodes) with max_nodes=4 must not be reported truncated."""
    node_docs = {n: _FakeNode(n) for n in ["A", "B", "C", "D"]}
    edges_by_node = {
        "A": [_fallback_edge("e1", "B"), _fallback_edge("e2", "C")],
        "B": [_fallback_edge("e1", "A"), _fallback_edge("e3", "D")],
        "C": [_fallback_edge("e2", "A"), _fallback_edge("e4", "D")],
        "D": [_fallback_edge("e3", "B"), _fallback_edge("e4", "C")],
    }
    storage = _fallback_storage(node_docs, edges_by_node)

    result = await storage._robust_fallback("A", max_depth=5, max_nodes=4)

    assert {n.id for n in result.nodes} == {"A", "B", "C", "D"}
    assert result.is_truncated is False
