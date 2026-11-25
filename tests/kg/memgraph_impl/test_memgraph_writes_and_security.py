"""Offline batch-read, subgraph capacity, and workspace-quoting tests for Memgraph."""

from __future__ import annotations

import asyncio

import pytest

from lightrag.kg import memgraph_impl
from lightrag.kg.memgraph_impl import MemgraphStorage
from tests.kg._cypher_fakes import FakeDriver

pytestmark = pytest.mark.offline


class _FakeNode(dict):
    """Minimal Memgraph node stand-in carrying ``.id`` and entity properties."""

    def __init__(self, node_id: int, entity_id: str, **properties):
        super().__init__(entity_id=entity_id, **properties)
        self.id = node_id


def _make_storage(driver: FakeDriver, *, workspace: str = "test") -> MemgraphStorage:
    storage = MemgraphStorage(
        namespace="chunk_entity_relation",
        global_config={"max_graph_nodes": 1000},
        embedding_func=None,
        workspace=workspace,
    )
    storage._driver = driver
    storage._DATABASE = "memgraph"
    return storage


# ---------------------------------------------------------------------------
# Batch reads: single UNWIND round trip matching scalar read contracts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_nodes_batch_is_one_query_with_get_node_results():
    label = "test"
    driver = FakeDriver(
        records=[
            {
                "entity_id": "A",
                "n": {"entity_id": "A", "labels": [label, "concept"]},
            },
            # A duplicate keeps the first node, matching get_node.
            {"entity_id": "A", "n": {"entity_id": "A", "description": "dup"}},
            {"entity_id": "B", "n": {"entity_id": "B"}},
        ]
    )
    storage = _make_storage(driver)
    assert storage._get_workspace_label() == label

    nodes = await storage.get_nodes_batch(["A", "B", "Missing", "A"])

    assert len(driver.calls) == 1
    query, params = driver.calls[0]
    assert "UNWIND $ids" in query and f"`{label}`" in query
    assert params == {"ids": ["A", "B", "Missing"]}
    assert nodes == {
        "A": {"entity_id": "A", "labels": ["concept"]},
        "B": {"entity_id": "B"},
    }
    assert driver.result.consumed


@pytest.mark.asyncio
async def test_get_edges_batch_is_one_query_keyed_as_requested():
    driver = FakeDriver(
        records=[
            {
                "src": "B",
                "tgt": "A",
                "edge_properties": {
                    "weight": 2.0,
                    "source_id": "c1",
                    "description": "d",
                    "keywords": "k",
                },
            },
            {"src": "B", "tgt": "A", "edge_properties": {"weight": 9.0}},
            # A missing core property receives get_edge's default.
            {"src": "A", "tgt": "C", "edge_properties": {"source_id": "c2"}},
        ]
    )
    storage = _make_storage(driver)

    edges = await storage.get_edges_batch(
        [{"src": "B", "tgt": "A"}, {"src": "A", "tgt": "C"}, {"src": "X", "tgt": "Y"}]
    )

    assert len(driver.calls) == 1
    query, params = driver.calls[0]
    assert "UNWIND $pairs" in query and "-[r]-" in query
    assert params == {
        "pairs": [
            {"src": "B", "tgt": "A"},
            {"src": "A", "tgt": "C"},
            {"src": "X", "tgt": "Y"},
        ]
    }
    assert edges == {
        ("B", "A"): {
            "weight": 2.0,
            "source_id": "c1",
            "description": "d",
            "keywords": "k",
        },
        ("A", "C"): {
            "weight": 1.0,
            "source_id": "c2",
            "description": None,
            "keywords": None,
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "arg"),
    [("get_nodes_batch", ["A"]), ("get_edges_batch", [{"src": "A", "tgt": "B"}])],
)
async def test_batch_read_failure_propagates(method, arg):
    driver = FakeDriver(run_error=RuntimeError("memgraph down"))
    with pytest.raises(RuntimeError, match="memgraph down"):
        await getattr(_make_storage(driver), method)(arg)
    assert len(driver.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["get_nodes_batch", "get_edges_batch"])
async def test_empty_batch_issues_no_query(method):
    driver = FakeDriver()
    assert await getattr(_make_storage(driver), method)([]) == {}
    assert driver.calls == []


# ---------------------------------------------------------------------------
# Subgraph capacity reservation and workspace label quoting / index creation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_knowledge_graph_preserves_isolated_start_node():
    start_node = _FakeNode(1, "Start", description="isolated")
    driver = FakeDriver(
        records=[
            {
                "node_info": [{"node": start_node}],
                "relationships": [],
                "is_truncated": False,
            }
        ]
    )
    storage = _make_storage(driver)

    result = await storage.get_knowledge_graph("Start", max_depth=0, max_nodes=1)

    assert len(result.nodes) == 1
    assert result.nodes[0].labels == ["Start"]
    assert result.nodes[0].properties["entity_id"] == "Start"
    assert result.edges == []
    assert result.is_truncated is False

    assert len(driver.calls) == 1
    _, params = driver.calls[0]
    assert params["entity_id"] == "Start"
    assert params["max_nodes"] == 1
    assert params["max_other_nodes"] == 0


@pytest.mark.asyncio
async def test_get_knowledge_graph_reserves_capacity_for_start_node_when_truncating():
    start_node = _FakeNode(1, "Start")
    driver = FakeDriver(
        records=[
            {
                "node_info": [{"node": start_node}],
                "relationships": [],
                "is_truncated": True,
            }
        ]
    )
    storage = _make_storage(driver)

    result = await storage.get_knowledge_graph("Start", max_depth=2, max_nodes=2)

    assert result.is_truncated is True
    assert len(result.nodes) == 1
    assert result.edges == []

    assert len(driver.calls) == 1
    _, params = driver.calls[0]
    assert params["max_nodes"] == 2
    assert params["max_other_nodes"] == 1


@pytest.mark.asyncio
async def test_get_knowledge_graph_max_nodes_zero_does_not_underflow():
    """``max_other_nodes`` must not go negative when ``max_nodes=0``."""
    driver = FakeDriver(
        records=[{"node_info": [], "relationships": [], "is_truncated": False}]
    )
    storage = _make_storage(driver)

    await storage.get_knowledge_graph("Start", max_depth=1, max_nodes=0)

    _, params = driver.calls[0]
    assert params["max_other_nodes"] == 0


_QUOTING_WORKSPACES = ["o'brien", "a`b", "my-ws"]


@pytest.mark.asyncio
@pytest.mark.parametrize("workspace", _QUOTING_WORKSPACES)
async def test_get_knowledge_graph_binds_raw_workspace_label(workspace):
    """The BFS path filter compares ``labels(n)`` to the raw workspace parameter."""
    driver = FakeDriver(
        records=[{"node_info": [], "relationships": [], "is_truncated": False}]
    )
    storage = _make_storage(driver, workspace=workspace)

    await storage.get_knowledge_graph("Start", max_depth=2, max_nodes=5)

    assert len(driver.calls) == 1
    query, params = driver.calls[0]
    assert "$workspace_label IN labels(n)" in query
    assert f"'{storage._get_workspace_label()}'" not in query
    assert params["workspace_label"] == workspace


@pytest.mark.asyncio
@pytest.mark.parametrize("workspace", _QUOTING_WORKSPACES)
async def test_initialize_creates_index_on_quoted_label(workspace, monkeypatch):
    driver = FakeDriver(records=[None])
    init_lock = asyncio.Lock()
    monkeypatch.setattr(memgraph_impl, "get_data_init_lock", lambda: init_lock)
    monkeypatch.setattr(
        memgraph_impl.AsyncGraphDatabase,
        "driver",
        lambda *args, **kwargs: driver,
    )
    storage = MemgraphStorage(
        namespace="chunk_entity_relation",
        global_config={"max_graph_nodes": 1000},
        embedding_func=None,
        workspace=workspace,
    )

    await storage.initialize()

    index_queries = [q for q in driver.queries if q.startswith("CREATE INDEX")]
    escaped = workspace.replace("`", "``")
    assert index_queries == [f"CREATE INDEX ON :`{escaped}`(entity_id)"]
