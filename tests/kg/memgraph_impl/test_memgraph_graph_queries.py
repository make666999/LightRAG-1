"""Offline graph-query contract tests for ``MemgraphStorage``.

Covers:
- directed edge reads (``get_all_edges`` / ``iter_edges``), degree tie-breaking,
  and ``get_node_edges`` missing-vs-isolated contracts via the shared Cypher
  contract mixin;
- edge direction preservation in ``get_node_edges``;
- fail-loud propagation in ``get_popular_labels`` / ``search_labels``.
"""

from __future__ import annotations

import pytest

from lightrag.kg.memgraph_impl import MemgraphStorage
from tests.kg._cypher_fakes import CypherGraphQueryContractMixin, FakeDriver

pytestmark = pytest.mark.offline


def _make_storage(driver: FakeDriver) -> MemgraphStorage:
    storage = MemgraphStorage(
        namespace="chunk_entity_relation",
        global_config={"max_graph_nodes": 1000},
        embedding_func=None,
        workspace="test",
    )
    storage._driver = driver
    storage._DATABASE = "memgraph"
    return storage


class TestMemgraphGraphQueries(CypherGraphQueryContractMixin):
    def make_storage(self, driver: FakeDriver) -> MemgraphStorage:
        return _make_storage(driver)

    def isolated_node_record(self, entity_id: str) -> dict:
        return {
            "node_entity_id": entity_id,
            "connected_entity_id": None,
            "start_entity_id": None,
        }

    @pytest.mark.asyncio
    async def test_get_node_edges_preserves_edge_direction(self):
        storage = _make_storage(
            FakeDriver(
                records=[
                    {
                        "node_entity_id": "Alpha",
                        "connected_entity_id": "Beta",
                        "start_entity_id": "Alpha",
                    },
                    {
                        "node_entity_id": "Alpha",
                        "connected_entity_id": "Gamma",
                        "start_entity_id": "Gamma",
                    },
                ]
            )
        )

        assert await storage.get_node_edges("Alpha") == [
            ("Alpha", "Beta"),
            ("Gamma", "Alpha"),
        ]

    @pytest.mark.asyncio
    async def test_get_popular_labels_raises_on_query_error(self):
        storage = _make_storage(FakeDriver(run_error=RuntimeError("memgraph is down")))

        with pytest.raises(RuntimeError, match="memgraph is down"):
            await storage.get_popular_labels(limit=10)

    @pytest.mark.asyncio
    async def test_get_popular_labels_raises_on_iteration_error(self):
        """A mid-stream failure must not return a partial ranking."""
        storage = _make_storage(FakeDriver(iter_error=RuntimeError("connection reset")))

        with pytest.raises(RuntimeError, match="connection reset"):
            await storage.get_popular_labels(limit=10)

    @pytest.mark.asyncio
    async def test_search_labels_raises_on_query_error(self):
        storage = _make_storage(FakeDriver(run_error=RuntimeError("memgraph is down")))

        with pytest.raises(RuntimeError, match="memgraph is down"):
            await storage.search_labels("alpha")

    @pytest.mark.asyncio
    async def test_search_labels_still_short_circuits_on_blank_query(self):
        """A blank query is a legitimate empty result and never hits the DB."""
        storage = _make_storage(
            FakeDriver(run_error=RuntimeError("must not be reached"))
        )

        assert await storage.search_labels("   ") == []
