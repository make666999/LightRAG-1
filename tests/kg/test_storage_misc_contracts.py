"""Miscellaneous cross-backend storage contracts: import-time env limits, degree return types, and deprecations.

Covers:
- Unusable/empty env-backed storage limits not breaking ``redis_impl`` or
  ``memgraph_impl`` imports (verified in a fresh subprocess).
- ``node_degree`` / ``edge_degree`` returning ``int`` (defaulting to ``0`` on
  missing entries) on ``PGGraphStorage`` and ``NetworkXStorage``.
- Startup deprecation metadata and warnings for deprecated storage backends.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import networkx as nx
import numpy as np
import pytest

from lightrag.kg import (
    DEPRECATED_STORAGE_MIGRATION_HINTS,
    DEPRECATED_STORAGES,
    STORAGE_IMPLEMENTATIONS,
    deprecated_storage_message,
)
from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.postgres_impl import PGGraphStorage
from lightrag.utils import EmbeddingFunc

pytestmark = pytest.mark.offline

# ---------------------------------------------------------------------------
# Import-time env-backed storage limits
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

ENV_KEYS = (
    "REDIS_MAX_CONNECTIONS",
    "REDIS_SOCKET_TIMEOUT",
    "REDIS_CONNECT_TIMEOUT",
    "REDIS_RETRY_ATTEMPTS",
    "MAX_GRAPH_NODES",
)

PROBE = """
import importlib.util, json, os, sys, traceback
import pipmaster
pipmaster.install = lambda *a, **k: (_ for _ in ()).throw(
    RuntimeError(f"package install refused: {a}"))
keys, cases = json.loads(sys.argv[1]), json.loads(sys.argv[2])

def fresh(module, case):
    origin = importlib.util.find_spec(module).origin
    name = f"{module}_probe_{case}"
    spec = importlib.util.spec_from_file_location(name, origin)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded

out = {}
for case, raw in cases.items():
    for key in keys:
        os.environ.pop(key, None)
    if raw is not None:
        os.environ.update(dict.fromkeys(keys, raw))
    try:
        r = fresh("lightrag.kg.redis_impl", case)
        m = fresh("lightrag.kg.memgraph_impl", case)
        out[case] = repr((r.MAX_CONNECTIONS, r.SOCKET_TIMEOUT,
                          r.SOCKET_CONNECT_TIMEOUT, r.RETRY_ATTEMPTS,
                          m.MAX_GRAPH_NODES))
    except Exception:
        out[case] = traceback.format_exc()
print(json.dumps(out))
"""

DEFAULTS = (200, 30.0, 10.0, 3, 1000)
CASES = {
    "unset": (None, DEFAULTS),
    "empty": ("", DEFAULTS),
    "blank": ("  ", DEFAULTS),
    "garbage": ("off", DEFAULTS),
    "populated": ("17", (17, 17.0, 17.0, 17, 17)),
}


@functools.lru_cache(maxsize=None)
def _import_results() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ENV_KEYS}
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")])
    )
    raws = {case: raw for case, (raw, _) in CASES.items()}
    result = subprocess.run(
        [sys.executable, "-c", PROBE, json.dumps(ENV_KEYS), json.dumps(raws)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("case", list(CASES))
def test_storage_limits_on_import(case: str) -> None:
    for sdk in ("redis", "neo4j"):
        if importlib.util.find_spec(sdk) is None:
            pytest.skip(f"{sdk} SDK not installed")
    assert _import_results()[case] == repr(CASES[case][1])


# ---------------------------------------------------------------------------
# Graph degree return-type contract
# ---------------------------------------------------------------------------


def _make_pg_storage(**attrs) -> PGGraphStorage:
    instance = object.__new__(PGGraphStorage)
    for k, v in attrs.items():
        setattr(instance, k, v)
    return instance


def _make_nx_storage(**attrs) -> NetworkXStorage:
    instance = object.__new__(NetworkXStorage)
    for k, v in attrs.items():
        setattr(instance, k, v)
    return instance


class TestPGGraphStorageDegree:
    """Verify PGGraphStorage.node_degree / edge_degree return int (0 for missing)."""

    @pytest.mark.asyncio
    async def test_node_degree_missing_and_existing(self):
        missing = _make_pg_storage(node_degrees_batch=AsyncMock(return_value={}))
        res_missing = await missing.node_degree("nonexistent_node")
        assert res_missing == 0 and isinstance(res_missing, int)

        existing = _make_pg_storage(
            node_degrees_batch=AsyncMock(return_value={"node_a": 5})
        )
        res_existing = await existing.node_degree("node_a")
        assert res_existing == 5 and isinstance(res_existing, int)

    @pytest.mark.asyncio
    async def test_edge_degree_missing_and_existing(self):
        missing = _make_pg_storage(edge_degrees_batch=AsyncMock(return_value={}))
        res_missing = await missing.edge_degree("src", "tgt")
        assert res_missing == 0 and isinstance(res_missing, int)

        existing = _make_pg_storage(
            edge_degrees_batch=AsyncMock(return_value={("src", "tgt"): 7})
        )
        res_existing = await existing.edge_degree("src", "tgt")
        assert res_existing == 7 and isinstance(res_existing, int)


class TestNetworkXStorageDegree:
    """Verify NetworkXStorage.node_degree / edge_degree return int (0 for missing)."""

    @pytest.mark.asyncio
    async def test_node_degree_missing_and_existing(self):
        graph = nx.Graph()
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")
        storage = _make_nx_storage(_get_graph=AsyncMock(return_value=graph))

        res_missing = await storage.node_degree("nonexistent")
        assert res_missing == 0 and isinstance(res_missing, int)

        res_existing = await storage.node_degree("a")
        assert res_existing == 2 and isinstance(res_existing, int)

    @pytest.mark.asyncio
    async def test_edge_degree_missing_and_existing(self):
        graph = nx.Graph()
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")
        graph.add_edge("b", "d")
        storage = _make_nx_storage(_get_graph=AsyncMock(return_value=graph))

        res_missing = await storage.edge_degree("x", "y")
        assert res_missing == 0 and isinstance(res_missing, int)

        res_existing = await storage.edge_degree("a", "b")
        assert res_existing == 4 and isinstance(res_existing, int)


# ---------------------------------------------------------------------------
# Deprecated storage warnings
# ---------------------------------------------------------------------------


async def _embed(texts, **kwargs):
    return np.zeros((len(texts), 8), dtype=np.float32)


_EMBEDDING = EmbeddingFunc(embedding_dim=8, max_token_size=512, func=_embed)


def test_age_graph_storage_is_deprecated_in_favor_of_plain_tables():
    message = deprecated_storage_message("PGGraphStorage")
    assert message is not None and "PGTableGraphStorage" in message
    assert "lightrag.tools.migrate_graph_storage" in message


def test_every_migration_hint_belongs_to_a_deprecated_backend():
    assert set(DEPRECATED_STORAGE_MIGRATION_HINTS) <= set(DEPRECATED_STORAGES)


@pytest.mark.parametrize("name", ["PGTableGraphStorage", "Unknown"])
def test_other_backends_are_not_deprecated(name):
    assert deprecated_storage_message(name) is None


def test_every_deprecated_backend_is_still_registered():
    registered = {
        name
        for info in STORAGE_IMPLEMENTATIONS.values()
        for name in info["implementations"]
    }
    for name, replacement in DEPRECATED_STORAGES.items():
        assert name in registered
        assert replacement in registered


def _deprecation_records(
    lightrag_caplog, monkeypatch, tmp_path, **storages
) -> list[str]:
    from lightrag import LightRAG

    monkeypatch.setenv("REDIS_URI", "redis://localhost:6379")
    lightrag_caplog.clear()
    with lightrag_caplog.at_level(logging.WARNING, logger="lightrag"):
        LightRAG(
            working_dir=str(tmp_path),
            llm_model_func=AsyncMock(return_value=""),
            embedding_func=_EMBEDDING,
            **storages,
        )
    return [
        record.getMessage()
        for record in lightrag_caplog.records
        if record.levelno == logging.WARNING and "deprecated" in record.getMessage()
    ]


def test_constructing_with_redis_logs_one_warning_per_backend(
    lightrag_caplog, monkeypatch, tmp_path
):
    messages = _deprecation_records(
        lightrag_caplog,
        monkeypatch,
        tmp_path,
        kv_storage="RedisKVStorage",
        doc_status_storage="RedisDocStatusStorage",
    )

    assert len(messages) == 2
    assert any("RedisKVStorage" in m and "PGKVStorage" in m for m in messages)
    assert any(
        "RedisDocStatusStorage" in m and "PGDocStatusStorage" in m for m in messages
    )


def test_constructing_with_age_graph_storage_logs_the_warning(
    lightrag_caplog, monkeypatch, tmp_path
):
    for key in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DATABASE"):
        monkeypatch.setenv(key, "x")
    messages = _deprecation_records(
        lightrag_caplog, monkeypatch, tmp_path, graph_storage="PGGraphStorage"
    )

    assert len(messages) == 1
    assert "PGGraphStorage" in messages[0] and "PGTableGraphStorage" in messages[0]


def test_default_storages_log_no_deprecation(lightrag_caplog, monkeypatch, tmp_path):
    assert _deprecation_records(lightrag_caplog, monkeypatch, tmp_path) == []
