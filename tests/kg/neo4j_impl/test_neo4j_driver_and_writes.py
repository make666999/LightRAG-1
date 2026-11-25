"""Offline driver-configuration and write-retry exhaustion tests for Neo4j."""

from __future__ import annotations

import logging

import pytest
from neo4j import exceptions as neo4jExceptions
from tenacity import wait_none

from lightrag.kg import neo4j_impl
from lightrag.kg.neo4j_impl import Neo4JStorage
from lightrag.kg.shared_storage import initialize_share_data
from tests.kg._cypher_fakes import FakeDriver

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# NEO4J_KEEP_ALIVE / [neo4j] keep_alive boolean parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env_value", "config_value", "expected", "warn_keys"),
    [
        pytest.param("t", "off", True, set(), id="env-true-alias"),
        pytest.param("off", "true", False, set(), id="env-false-spelling"),
        pytest.param(
            "ture",
            "off",
            True,
            {"NEO4J_KEEP_ALIVE"},
            id="env-typo-defaults-and-warns",
        ),
        pytest.param(None, "off", False, set(), id="config-false-spelling"),
    ],
)
async def test_keep_alive_setting_is_parsed_for_driver(
    monkeypatch,
    lightrag_log_records: list[logging.LogRecord],
    env_value,
    config_value,
    expected,
    warn_keys,
):
    if env_value is None:
        monkeypatch.delenv("NEO4J_KEEP_ALIVE", raising=False)
    else:
        monkeypatch.setenv("NEO4J_KEEP_ALIVE", env_value)

    monkeypatch.setattr(
        neo4j_impl.config,
        "get",
        lambda _section, key, fallback=None: (
            config_value if key == "keep_alive" else fallback
        ),
    )
    driver = FakeDriver(records=[])
    driver_kwargs = {}

    def fake_driver(_uri, **kwargs):
        driver_kwargs.update(kwargs)
        return driver

    monkeypatch.setattr(neo4j_impl.AsyncGraphDatabase, "driver", fake_driver)
    initialize_share_data()

    await Neo4JStorage(
        namespace="test", global_config={}, embedding_func=None, workspace="test"
    ).initialize()

    assert driver_kwargs["keep_alive"] is expected
    warned = {
        record.args[0]
        for record in lightrag_log_records
        if "is not a recognized boolean" in record.msg
    }
    assert warned == warn_keys


# ---------------------------------------------------------------------------
# Write retry exhaustion reraises the original Neo4j exception
# ---------------------------------------------------------------------------


class _UnavailableDriver:
    """Driver whose every session fails to open, counting the attempts."""

    def __init__(self):
        self.attempts = 0

    def session(self, **kwargs):
        return self

    async def __aenter__(self):
        self.attempts += 1
        raise neo4jExceptions.ServiceUnavailable("Neo4j unavailable")

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _make_unavailable_storage() -> tuple[Neo4JStorage, _UnavailableDriver]:
    storage = Neo4JStorage(
        namespace="chunk_entity_relation",
        global_config={"max_graph_nodes": 1000},
        embedding_func=None,
        workspace="test",
    )
    driver = _UnavailableDriver()
    storage._driver = driver
    storage._DATABASE = "neo4j"
    return storage, driver


WRITE_CASES = [
    pytest.param("upsert_node", ("node", {"entity_id": "node"}), id="upsert_node"),
    pytest.param(
        "upsert_nodes_batch",
        ([("node", {"entity_id": "node"})],),
        id="upsert_nodes_batch",
    ),
    pytest.param(
        "upsert_edges_batch",
        ([("source", "target", {})],),
        id="upsert_edges_batch",
    ),
    pytest.param("upsert_edge", ("source", "target", {}), id="upsert_edge"),
    pytest.param("delete_node", ("node",), id="delete_node"),
    pytest.param("remove_edges", ([("source", "target")],), id="remove_edges"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("method_name", "args"), WRITE_CASES)
async def test_write_retry_exhaustion_reraises_original_exception(
    monkeypatch, method_name, args
):
    storage, driver = _make_unavailable_storage()
    method = getattr(Neo4JStorage, method_name)
    monkeypatch.setattr(method.retry, "wait", wait_none())

    with pytest.raises(neo4jExceptions.ServiceUnavailable):
        await getattr(storage, method_name)(*args)

    assert driver.attempts == 3


@pytest.mark.asyncio
async def test_remove_nodes_does_not_multiply_delete_node_retries(monkeypatch):
    storage, driver = _make_unavailable_storage()
    monkeypatch.setattr(Neo4JStorage.delete_node.retry, "wait", wait_none())

    with pytest.raises(neo4jExceptions.ServiceUnavailable):
        await storage.remove_nodes(["node"])

    assert driver.attempts == 3
