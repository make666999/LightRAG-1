"""Shared Cypher test doubles and graph-query contract mixins.

Used by ``tests/kg/neo4j_impl/`` and ``tests/kg/memgraph_impl/`` so both
Bolt/Cypher graph backends exercise the same read/query contracts without
duplicating driver/session/result boilerplate.

This module intentionally has no ``test_`` prefix and its mixins do not start
with ``Test`` so pytest never collects it directly; each backend subclasses
the mixin inside its own mirror directory.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

import pytest


def normalize_cypher(query: str) -> str:
    """Collapse whitespace so multi-line Cypher can be matched as one string."""
    return re.sub(r"\s+", " ", query).strip()


class FakeResult:
    """Async Neo4j/Memgraph result supporting iteration, single, fetch, data, consume."""

    def __init__(
        self, records: Sequence[Any] = (), *, error: BaseException | None = None
    ):
        self._records = list(records)
        self._error = error
        self.consumed = False

    def __aiter__(self):
        self._iter = iter(self._records)
        return self

    async def __anext__(self):
        if self._error is not None:
            raise self._error
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration

    async def single(self):
        if self._error is not None:
            raise self._error
        return self._records[0] if self._records else None

    async def fetch(self, n: int):
        if self._error is not None:
            raise self._error
        return self._records[:n]

    async def data(self):
        if self._error is not None:
            raise self._error
        return list(self._records)

    async def consume(self):
        self.consumed = True
        return None


class FakeSession:
    """Async Neo4j/Memgraph session stand-in recording every ``run`` invocation."""

    def __init__(self, driver: FakeDriver):
        self._driver = driver

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def run(self, query: str, parameters: dict[str, Any] | None = None, **kwargs):
        params = dict(parameters) if parameters is not None else {}
        if kwargs:
            params.update(kwargs)
        self._driver.calls.append((query, params))
        if self._driver.run_error is not None:
            raise self._driver.run_error
        if self._driver.handler is not None:
            return self._driver.handler(query, params)
        return self._driver.next_result()


class FakeDriver:
    """Async Neo4j/Memgraph driver stand-in returning scripted or handled results."""

    def __init__(
        self,
        results: Sequence[FakeResult] | None = None,
        *,
        records: Sequence[Any] = (),
        run_error: BaseException | None = None,
        iter_error: BaseException | None = None,
        handler: Callable[[str, dict[str, Any]], FakeResult] | None = None,
    ):
        if results is not None:
            self._results = list(results)
            self.result = self._results[0] if self._results else FakeResult()
        else:
            self.result = FakeResult(records, error=iter_error)
            self._results = [self.result]
        self.run_error = run_error
        self.handler = handler
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def queries(self) -> list[str]:
        return [query for query, _ in self.calls]

    def next_result(self) -> FakeResult:
        # Keep the last result once scripted ones are exhausted so an unexpected
        # extra query surfaces as a bad record rather than an IndexError that a
        # storage's exception handler might swallow.
        if len(self._results) > 1:
            return self._results.pop(0)
        return self._results[0]

    def session(self, **kwargs):
        return FakeSession(self)


def edge_record(source: str, target: str, properties: dict[str, Any]) -> dict[str, Any]:
    """Row shape returned by a directed ``get_all_edges`` / ``iter_edges`` match."""
    return {"source": source, "target": target, "properties": dict(properties)}


class CypherGraphQueryContractMixin:
    """Shared Cypher query contracts for Neo4j and Memgraph graph backends.

    Subclasses implement ``make_storage(driver)`` and ``isolated_node_record(entity_id)``.
    """

    def make_storage(self, driver: FakeDriver):
        raise NotImplementedError

    def isolated_node_record(self, entity_id: str) -> dict[str, Any]:
        raise NotImplementedError

    @pytest.mark.asyncio
    async def test_get_all_edges_returns_one_row_per_relationship(self):
        driver = FakeDriver(records=[edge_record("Alpha", "Beta", {"weight": 1.0})])
        storage = self.make_storage(driver)

        edges = await storage.get_all_edges()

        assert len(edges) == 1
        assert edges[0]["source"] == "Alpha"
        assert edges[0]["target"] == "Beta"
        query = driver.queries[0]
        assert "-[r]->" in query
        assert "DISTINCT" not in query
        assert "id(r)" not in query

    @pytest.mark.asyncio
    async def test_iter_edges_returns_one_row_per_relationship(self):
        driver = FakeDriver(records=[edge_record("Alpha", "Beta", {"weight": 1.0})])
        storage = self.make_storage(driver)

        batches = [batch async for batch in storage.iter_edges(batch_size=10)]
        edges = [edge for batch in batches for edge in batch]

        assert len(edges) == 1
        assert edges[0]["source"] == "Alpha"
        assert edges[0]["target"] == "Beta"
        query = driver.queries[0]
        assert "-[r]->" in query
        assert "DISTINCT" not in query
        assert "id(r)" not in query

    @pytest.mark.asyncio
    async def test_star_mode_ranks_by_degree_then_entity_id(self):
        """``get_knowledge_graph('*')`` must break degree ties on ``entity_id``."""
        driver = FakeDriver(
            results=[
                FakeResult([{"total": 5}]),
                FakeResult([{"node_info": [], "relationships": []}]),
            ]
        )
        storage = self.make_storage(driver)

        result = await storage.get_knowledge_graph("*", max_depth=1, max_nodes=2)

        assert len(driver.queries) == 2, driver.queries
        main_query = normalize_cypher(driver.queries[1])
        assert "ORDER BY degree DESC, n.entity_id ASC" in main_query, main_query
        assert "ORDER BY degree DESC LIMIT" not in main_query, main_query
        assert result.is_truncated is True

    @pytest.mark.asyncio
    async def test_get_node_edges_returns_none_for_missing_node(self):
        """Zero rows means the anchor MATCH never bound n: no such node."""
        storage = self.make_storage(FakeDriver(records=[]))

        assert await storage.get_node_edges("NoSuchEntity") is None

    @pytest.mark.asyncio
    async def test_get_node_edges_returns_empty_list_for_isolated_node(self):
        """An existing node with no relations yields one row with NULL neighbor."""
        storage = self.make_storage(
            FakeDriver(records=[self.isolated_node_record("Lonely")])
        )

        assert await storage.get_node_edges("Lonely") == []
