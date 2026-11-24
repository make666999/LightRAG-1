"""NetworkXStorage graph operation contracts: batch snapshots, bounded iteration, copy-on-read, truncation, and ranking.

Covers:
- Batch graph reads answering from a single ``_get_graph()`` snapshot across a
  mid-batch peer commit.
- Bounded ``iter_labels`` / ``iter_edges`` batch sizing, lazy first-batch read,
  and loud failure on live mutation between batches.
- Copy-on-read isolation for ``get_node``, ``get_edge``, and ``get_nodes_batch``.
- ``get_knowledge_graph`` non-truncation on diamond duplicates and deterministic
  label tie-breaking for ``*`` and BFS cutoffs.
- ``get_popular_labels`` degree ranking with alphabetical tie-break and isolated
  nodes ranked last.
"""

from __future__ import annotations

import networkx as nx
import pytest

from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


def _storage(
    tmp_path, namespace: str, *, max_graph_nodes: int | None = None
) -> NetworkXStorage:
    global_config: dict = {"working_dir": str(tmp_path)}
    if max_graph_nodes is not None:
        global_config["max_graph_nodes"] = max_graph_nodes
    return NetworkXStorage(
        namespace=namespace,
        workspace="ws",
        global_config=global_config,
        embedding_func=None,
    )


def _memory_storage(graph) -> NetworkXStorage:
    """Lightweight NetworkXStorage backed directly by an in-memory graph object."""
    storage = NetworkXStorage.__new__(NetworkXStorage)
    storage.workspace = "test"
    storage.namespace = "chunk_entity_relation"

    async def _get_graph():
        return graph

    storage._get_graph = _get_graph
    return storage


# ---------------------------------------------------------------------------
# Batch read single-snapshot isolation
# ---------------------------------------------------------------------------


async def _seed_batch_graph(storage: NetworkXStorage) -> None:
    for node_id in ("A", "B", "X"):
        await storage.upsert_node(node_id, {"entity_id": node_id, "val": "old"})
    await storage.upsert_edge("A", "X", {"weight": 1.0, "val": "old"})
    await storage.upsert_edge("B", "X", {"weight": 1.0, "val": "old"})


async def _peer_commit(peer: NetworkXStorage) -> None:
    """One commit that changes every entry each batch below reads, together."""
    await peer.upsert_node("A", {"entity_id": "A", "val": "new"})
    await peer.upsert_node("B", {"entity_id": "B", "val": "new"})
    await peer.upsert_edge("A", "X", {"weight": 2.0, "val": "new"})
    await peer.upsert_edge("B", "X", {"weight": 2.0, "val": "new"})
    await peer.upsert_node("C", {"entity_id": "C", "val": "new"})
    await peer.upsert_edge("A", "C", {"weight": 1.0})
    await peer.upsert_edge("B", "C", {"weight": 1.0})
    assert await peer.index_done_callback() is True


def _normalized(result):
    return {
        key: sorted(tuple(sorted(edge)) for edge in value)
        if isinstance(value, list)
        else value
        for key, value in result.items()
    }


BATCH_READS = [
    pytest.param(lambda s: s.get_nodes_batch(["A", "B"]), id="get_nodes_batch"),
    pytest.param(lambda s: s.node_degrees_batch(["A", "B"]), id="node_degrees_batch"),
    pytest.param(
        lambda s: s.get_edges_batch(
            [{"src": "A", "tgt": "X"}, {"src": "B", "tgt": "X"}]
        ),
        id="get_edges_batch",
    ),
    pytest.param(
        lambda s: s.edge_degrees_batch([("A", "X"), ("B", "X")]),
        id="edge_degrees_batch",
    ),
    pytest.param(
        lambda s: s.get_nodes_edges_batch(["A", "B"]), id="get_nodes_edges_batch"
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_read", BATCH_READS)
async def test_batch_read_does_not_tear_across_a_mid_batch_peer_commit(
    tmp_path, batch_read
):
    storage = _storage(tmp_path, "batch_tear")
    await storage.initialize()
    await _seed_batch_graph(storage)
    assert await storage.index_done_callback() is True

    peer = _storage(tmp_path, "batch_tear")
    await peer.initialize()

    try:
        before = _normalized(await batch_read(storage))

        original_get_graph = storage._get_graph
        commits = []

        async def get_graph_then_peer_commits():
            graph = await original_get_graph()
            if not commits:
                commits.append("peer")
                await _peer_commit(peer)
            return graph

        storage._get_graph = get_graph_then_peer_commits
        try:
            during = _normalized(await batch_read(storage))
        finally:
            storage._get_graph = original_get_graph

        assert commits == ["peer"], "the mid-batch peer commit never fired"
        after = _normalized(await batch_read(storage))
        assert after != before, "the peer commit is not visible to this storage"
        assert during == before, f"torn batch read: {during!r}"
    finally:
        await peer.finalize()
        await storage.finalize()


# ---------------------------------------------------------------------------
# Bounded graph iteration (iter_labels / iter_edges)
# ---------------------------------------------------------------------------


class _CountingGraph:
    """Wraps a graph so a test can observe how much of each view was read."""

    def __init__(self, graph: nx.Graph):
        self._graph = graph
        self.reads = 0

    def _counted(self, view):
        for item in view:
            self.reads += 1
            yield item

    def nodes(self):
        return self._counted(self._graph.nodes())

    def edges(self, data=False):
        return self._counted(self._graph.edges(data=data))


@pytest.mark.asyncio
async def test_graph_iteration_never_exceeds_the_requested_batch_size():
    graph = nx.Graph()
    graph.add_edges_from((f"N{index}", f"N{index + 1}") for index in range(9))
    storage = _memory_storage(graph)

    label_batches = [batch async for batch in storage.iter_labels(3)]
    edge_batches = [batch async for batch in storage.iter_edges(4)]

    assert max(map(len, label_batches)) == 3
    assert max(map(len, edge_batches)) == 4
    assert {label for batch in label_batches for label in batch} == set(graph.nodes)
    assert {
        frozenset((edge["source"], edge["target"]))
        for batch in edge_batches
        for edge in batch
    } == {frozenset(edge) for edge in graph.edges}


@pytest.mark.asyncio
async def test_graph_iteration_rejects_non_positive_batch_size():
    storage = _memory_storage(nx.Graph())

    with pytest.raises(ValueError, match="positive"):
        await anext(storage.iter_labels(0))
    with pytest.raises(ValueError, match="positive"):
        await anext(storage.iter_edges(0))


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["iter_labels", "iter_edges"])
async def test_first_batch_reads_only_one_batch_of_the_graph(method):
    graph = nx.Graph()
    graph.add_edges_from((f"N{index}", f"N{index + 1}") for index in range(50))
    counting = _CountingGraph(graph)
    storage = _memory_storage(counting)

    gen = getattr(storage, method)(1)
    assert len(await anext(gen)) == 1
    await gen.aclose()

    assert counting.reads == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "mutate"),
    [
        pytest.param(
            "iter_labels", lambda graph: graph.add_node("LATE_ARRIVAL"), id="node"
        ),
        pytest.param(
            "iter_edges",
            lambda graph: graph.add_edge("N0", "LATE_ARRIVAL"),
            id="edge",
        ),
    ],
)
async def test_mutation_between_batches_raises_an_error_naming_the_cause(
    method, mutate
):
    graph = nx.Graph()
    graph.add_edges_from((f"N{index}", f"N{index + 1}") for index in range(5))
    storage = _memory_storage(graph)

    gen = getattr(storage, method)(2)
    assert len(await anext(gen)) == 2
    mutate(graph)

    with pytest.raises(RuntimeError, match=f"{method}: the graph was modified") as info:
        await anext(gen)
    assert "changed size during iteration" in str(info.value.__cause__)


# ---------------------------------------------------------------------------
# Copy-on-read isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_node_returns_copy_not_live_alias(tmp_path):
    storage = _storage(tmp_path, "copy_on_read_node")
    await storage.initialize()
    await storage.upsert_node(
        "A",
        {"entity_type": "PERSON", "description": "orig", "source_id": "c1"},
    )

    node = await storage.get_node("A")
    assert node is not None
    node["description"] = "MUTATED"

    reread = await storage.get_node("A")
    assert reread is not None
    assert reread["description"] == "orig"


@pytest.mark.asyncio
async def test_get_edge_returns_copy_not_live_alias(tmp_path):
    storage = _storage(tmp_path, "copy_on_read_edge")
    await storage.initialize()
    await storage.upsert_node("A", {"entity_type": "PERSON", "description": "a"})
    await storage.upsert_node("B", {"entity_type": "PERSON", "description": "b"})
    await storage.upsert_edge("A", "B", {"weight": 1.0, "description": "rel-orig"})

    edge = await storage.get_edge("A", "B")
    assert edge is not None
    edge["description"] = "REL-MUTATED"

    reread = await storage.get_edge("A", "B")
    assert reread is not None
    assert reread["description"] == "rel-orig"


@pytest.mark.asyncio
async def test_get_nodes_batch_uses_copied_nodes(tmp_path):
    storage = _storage(tmp_path, "copy_on_read_batch")
    await storage.initialize()
    await storage.upsert_node("A", {"entity_type": "ORG", "description": "keep-me"})

    batch = await storage.get_nodes_batch(["A"])
    batch["A"]["description"] = "batch-mutated"

    reread = await storage.get_node("A")
    assert reread["description"] == "keep-me"


# ---------------------------------------------------------------------------
# Knowledge graph truncation & deterministic tie-breaking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_knowledge_graph_is_not_truncated_when_all_nodes_returned_with_duplicates(
    tmp_path,
):
    storage = _storage(tmp_path, "test_graph_diamond")
    await storage.initialize()

    g = nx.Graph()
    g.add_edge("A", "B")
    g.add_edge("A", "C")
    g.add_edge("B", "D")
    g.add_edge("C", "D")
    storage._graph = g

    result = await storage.get_knowledge_graph("A", max_depth=2, max_nodes=4)
    assert len(result.nodes) == 4
    assert result.is_truncated is False


@pytest.mark.asyncio
async def test_get_knowledge_graph_star_breaks_degree_ties_by_label(tmp_path):
    storage = _storage(tmp_path, "kg_star_ties", max_graph_nodes=1000)
    await storage.initialize()

    g = nx.Graph()
    for name in ["Zeta", "Alpha", "Beta"]:
        g.add_node(name, entity_type="CONCEPT", description=name)
    storage._graph = g

    result = await storage.get_knowledge_graph("*", max_depth=1, max_nodes=2)
    ids = sorted(node.id for node in result.nodes)
    assert ids == ["Alpha", "Beta"]
    assert result.is_truncated is True


@pytest.mark.asyncio
async def test_get_knowledge_graph_bfs_breaks_degree_ties_by_label(tmp_path):
    storage = _storage(tmp_path, "kg_bfs_ties", max_graph_nodes=1000)
    await storage.initialize()

    g = nx.Graph()
    g.add_node("A", entity_type="CONCEPT", description="A")
    for name in ["Zeta", "Alpha", "Beta"]:
        g.add_node(name, entity_type="CONCEPT", description=name)
        g.add_edge("A", name)
    storage._graph = g

    result = await storage.get_knowledge_graph("A", max_depth=1, max_nodes=3)
    ids = sorted(node.id for node in result.nodes)
    assert ids == ["A", "Alpha", "Beta"]
    assert result.is_truncated is True


# ---------------------------------------------------------------------------
# Popular labels ranking contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ties_break_on_label_not_insertion_order():
    graph = nx.Graph()
    for name in ["Zeta", "Mu", "Alpha"]:
        graph.add_node(name)
    graph.add_edge("Hub", "Zeta")
    graph.add_edge("Hub", "Mu")
    graph.add_edge("Hub", "Alpha")

    storage = _memory_storage(graph)

    assert await storage.get_popular_labels() == ["Hub", "Alpha", "Mu", "Zeta"]
    assert await storage.get_popular_labels(limit=2) == ["Hub", "Alpha"]


@pytest.mark.asyncio
async def test_isolated_nodes_are_ranked_last_but_still_ranked():
    graph = nx.Graph()
    graph.add_edge("Connected", "Peer")
    graph.add_node("Orphan")
    graph.add_node("Aardvark")

    storage = _memory_storage(graph)

    labels = await storage.get_popular_labels()
    assert set(labels) == {"Connected", "Peer", "Orphan", "Aardvark"}
    assert labels[-2:] == ["Aardvark", "Orphan"]
