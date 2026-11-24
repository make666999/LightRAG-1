"""Strict-read contracts for the OpenSearch backend (Phase 0 groundwork).

Contracts pinned here:
* ``get_docs_by_statuses(..., strict=True)`` — complete-or-raise on PIT, page,
  and deserialization errors while keeping ``strict=False`` best-effort.
* Scheduling-safe reads across KV, doc_status, graph, and vector storages —
  ``None`` / absent / empty means CONFIRMED absent; whole-call ``TransportError``,
  per-item ``mget`` errors, malformed items, and short ``docs`` arrays must raise.
* KV ``get_by_id_strict`` — three-state contract over pending buffers and index
  readiness.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip(
    "opensearchpy",
    reason="opensearchpy is required for OpenSearch storage tests",
)

from opensearchpy.exceptions import TransportError  # type: ignore  # noqa: E402

from lightrag.base import DocStatus  # noqa: E402
from lightrag.exceptions import StorageControlPlaneError  # noqa: E402
from lightrag.kg.opensearch_impl import (  # noqa: E402
    ClientManager,
    OpenSearchDocStatusStorage,
    OpenSearchGraphStorage,
    OpenSearchKVStorage,
    OpenSearchVectorDBStorage,
    _canonical_edge_id,
)

from ._helpers import (  # noqa: E402
    RejectingEmbedFunc,
    default_global_config,
    make_strict_client,
    missing_index_error,
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


def _status_source(doc_id: str, status: str = "failed") -> dict:
    return {
        "__mirrored_id": doc_id,
        "content_summary": f"summary-{doc_id}",
        "content_length": 10,
        "file_path": f"{doc_id}.txt",
        "status": status,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }


def _hit(doc_id: str, source: dict) -> dict:
    return {"_id": doc_id, "_source": source, "sort": [doc_id]}


def _page(hits: list[dict]) -> dict:
    return {"hits": {"hits": hits}}


def _make_graph_client() -> AsyncMock:
    client = make_strict_client()
    client.indices.exists = AsyncMock(return_value=False)
    client.transport = AsyncMock()
    client.transport.perform_request = AsyncMock(side_effect=Exception("no PPL"))
    return client


async def _build_storage(kind: str, global_config):
    """Instantiate and initialize a storage + client pair for ``kind``."""
    client = _make_graph_client() if kind == "graph" else make_strict_client()
    cls_and_ns = {
        "kv": (OpenSearchKVStorage, "full_docs"),
        "doc_status": (OpenSearchDocStatusStorage, "doc_status"),
        "graph": (OpenSearchGraphStorage, "chunk_entity_relation"),
        "vector": (OpenSearchVectorDBStorage, "chunks"),
    }
    cls, namespace = cls_and_ns[kind]
    with patch.object(ClientManager, "get_client", return_value=client):
        storage = cls(
            namespace=namespace,
            global_config=global_config,
            embedding_func=RejectingEmbedFunc(),
            workspace="strictws",
        )
        await storage.initialize()
    return storage, client


def _mget_error_for(bad_id: str, *, source: dict | None = None):
    """Echo requested-id order: ``bad_id`` gets an item-level error, others succeed."""

    async def _side_effect(index=None, body=None, **kwargs):
        docs = []
        for doc_id in (body or {}).get("ids", []):
            if doc_id == bad_id:
                docs.append({"_id": doc_id, "error": {"type": "shard"}, "status": 503})
            elif source is None:
                docs.append({"_id": doc_id, "found": True})
            else:
                docs.append({"_id": doc_id, "found": True, "_source": dict(source)})
        return {"docs": docs}

    return _side_effect


def _edge_doc(edge_id, source=None, error=None):
    if error is not None:
        return {"_id": edge_id, "error": error}
    if source is None:
        return {"_id": edge_id, "found": False}
    return {"_id": edge_id, "found": True, "_source": source}


# ---------------------------------------------------------------------------
# get_docs_by_statuses strict contract
# ---------------------------------------------------------------------------


async def test_second_page_failure_raises_in_strict_partial_in_relaxed(global_config):
    """A mid-pagination transport error must not silently drop the tail."""
    storage, client = await _build_storage("doc_status", global_config)

    client.search = AsyncMock(side_effect=transient_error())
    with pytest.raises(TransportError):
        await storage.get_docs_by_statuses([DocStatus.FAILED], strict=True)

    client.search = AsyncMock(side_effect=transient_error())
    assert await storage.get_docs_by_statuses([DocStatus.FAILED]) == {}

    big_page = [_hit(f"d{i}", _status_source(f"d{i}")) for i in range(10000)]
    client.search = AsyncMock(side_effect=[_page(big_page), transient_error()])
    with pytest.raises(TransportError):
        await storage.get_docs_by_statuses([DocStatus.FAILED], strict=True)

    client.search = AsyncMock(side_effect=[_page(big_page), transient_error()])
    partial = await storage.get_docs_by_statuses([DocStatus.FAILED])
    assert len(partial) == 10000


async def test_pit_creation_failure_raises_in_strict(global_config):
    storage, client = await _build_storage("doc_status", global_config)
    client.create_pit = AsyncMock(side_effect=transient_error())
    with pytest.raises(TransportError):
        await storage.get_docs_by_statuses([DocStatus.PENDING], strict=True)
    assert await storage.get_docs_by_statuses([DocStatus.PENDING]) == {}


async def test_undeserializable_record_raises_in_strict_skips_in_relaxed(global_config):
    storage, client = await _build_storage("doc_status", global_config)
    bad = _hit("bad", {"__mirrored_id": "bad", "status": "failed"})
    good = _hit("good", _status_source("good"))

    client.search = AsyncMock(return_value=_page([good, bad]))
    with pytest.raises(TypeError):
        await storage.get_docs_by_statuses([DocStatus.FAILED], strict=True)

    client.search = AsyncMock(return_value=_page([good, bad]))
    relaxed = await storage.get_docs_by_statuses([DocStatus.FAILED])
    assert set(relaxed) == {"good"}


async def test_missing_index_is_complete_empty_in_both_modes(global_config):
    storage, client = await _build_storage("doc_status", global_config)
    client.search = AsyncMock(side_effect=missing_index_error())
    assert await storage.get_docs_by_statuses([DocStatus.PENDING], strict=True) == {}
    storage._index_ready = True
    assert await storage.get_docs_by_statuses([DocStatus.PENDING]) == {}


async def test_pit_delete_failure_stays_best_effort(global_config):
    storage, client = await _build_storage("doc_status", global_config)
    client.search = AsyncMock(return_value=_page([_hit("d1", _status_source("d1"))]))
    client.delete_pit = AsyncMock(side_effect=transient_error())
    result = await storage.get_docs_by_statuses([DocStatus.FAILED], strict=True)
    assert set(result) == {"d1"}


# ---------------------------------------------------------------------------
# Parametrized TransportError, item-level mget error, short response, missing index
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "rpc_attr", "invoke"),
    [
        ("kv", "mget", lambda s: s.get_by_id("doc-1")),
        ("kv", "mget", lambda s: s.get_by_ids(["doc-1", "doc-2"])),
        ("kv", "mget", lambda s: s.filter_keys({"a", "b"})),
        ("kv", "mget", lambda s: s.get_by_id_strict("doc-1")),
        ("doc_status", "mget", lambda s: s.filter_keys({"a", "b"})),
        ("doc_status", "mget", lambda s: s.get_by_ids(["doc-1", "doc-2"])),
        ("graph", "exists", lambda s: s.has_node("Alice")),
        ("graph", "mget", lambda s: s.get_node("Alice")),
        ("graph", "mget", lambda s: s.get_nodes_batch(["Alice", "Bob"])),
        ("graph", "mget", lambda s: s.get_edges_batch([{"src": "A", "tgt": "B"}])),
        ("graph", "mget", lambda s: s.has_nodes_batch(["Alice", "Bob"])),
        ("graph", "search", lambda s: s.get_all_nodes()),
        (
            "vector",
            "search",
            lambda s: s.query("q", top_k=5, query_embedding=[0.1] * 8),
        ),
        ("graph", "mget", lambda s: s.upsert_edge("A", "B", {})),
        ("graph", "mget", lambda s: s.upsert_edges_batch([("A", "B", {})])),
        ("graph", "index", lambda s: s.upsert_node("A", {})),
        ("graph", "delete_by_query", lambda s: s.delete_node("A")),
    ],
    ids=[
        "kv-get_by_id",
        "kv-get_by_ids",
        "kv-filter_keys",
        "kv-get_by_id_strict",
        "doc_status-filter_keys",
        "doc_status-get_by_ids",
        "graph-has_node",
        "graph-get_node",
        "graph-get_nodes_batch",
        "graph-get_edges_batch",
        "graph-has_nodes_batch",
        "graph-get_all_nodes",
        "vector-query",
        "graph-upsert_edge-endpoint-check",
        "graph-upsert_edges_batch-endpoint-check",
        "graph-upsert_node",
        "graph-delete_node",
    ],
)
async def test_transient_transport_error_raises(global_config, kind, rpc_attr, invoke):
    """Whole-call TransportError must propagate rather than degrade to absence/empty."""
    storage, client = await _build_storage(kind, global_config)
    setattr(client, rpc_attr, AsyncMock(side_effect=transient_error()))
    with pytest.raises(TransportError):
        await invoke(storage)


@pytest.mark.parametrize(
    ("kind", "bad_id", "source", "invoke"),
    [
        ("kv", "doc-1", {"k": "v"}, lambda s: s.get_by_id("doc-1")),
        ("kv", "doc-2", {"k": "v"}, lambda s: s.get_by_ids(["doc-1", "doc-2"])),
        ("kv", "bad", None, lambda s: s.filter_keys({"good", "bad"})),
        (
            "doc_status",
            "bad",
            {"status": "failed"},
            lambda s: s.get_by_ids(["good", "bad"]),
        ),
        ("doc_status", "bad", None, lambda s: s.filter_keys({"good", "bad"})),
        (
            "graph",
            "Bob",
            {"entity_id": "Alice"},
            lambda s: s.get_nodes_batch(["Alice", "Bob"]),
        ),
        (
            "graph",
            _canonical_edge_id("A", "B"),
            {"source_id": "c1"},
            lambda s: s.get_edges_batch([{"src": "A", "tgt": "B"}]),
        ),
    ],
    ids=[
        "kv-get_by_id",
        "kv-get_by_ids",
        "kv-filter_keys",
        "doc_status-get_by_ids",
        "doc_status-filter_keys",
        "graph-get_nodes_batch",
        "graph-get_edges_batch",
    ],
)
async def test_mget_item_level_error_raises(
    global_config, kind, bad_id, source, invoke
):
    """An HTTP-200 mget item carrying ``error`` must raise, never read as absent."""
    storage, client = await _build_storage(kind, global_config)
    client.mget = AsyncMock(side_effect=_mget_error_for(bad_id, source=source))
    with pytest.raises(RuntimeError, match="item error"):
        await invoke(storage)


@pytest.mark.parametrize(
    ("kind", "docs", "match", "invoke"),
    [
        (
            "kv",
            [{"_id": "doc-1", "found": True, "_source": {"k": "v"}}],
            "for 2 requested ids",
            lambda s: s.get_by_ids(["doc-1", "doc-2"]),
        ),
        (
            "graph",
            [{"_id": "Alice", "found": True, "_source": {"entity_id": "Alice"}}],
            "expected exactly 2",
            lambda s: s.get_nodes_batch(["Alice", "Bob"]),
        ),
        (
            "graph",
            [],
            "expected exactly 1",
            lambda s: s.get_edges_batch([{"src": "A", "tgt": "B"}]),
        ),
    ],
    ids=["kv-get_by_ids", "graph-get_nodes_batch", "graph-get_edges_batch"],
)
async def test_mget_short_response_raises(global_config, kind, docs, match, invoke):
    """A short ``docs`` array (omitted id) must raise instead of treating it as absent."""
    storage, client = await _build_storage(kind, global_config)
    client.mget = AsyncMock(return_value={"docs": docs})
    with pytest.raises(RuntimeError, match=match):
        await invoke(storage)


@pytest.mark.parametrize(
    ("kind", "rpc_attr", "invoke", "expected"),
    [
        ("kv", "mget", lambda s: s.filter_keys({"a", "b"}), {"a", "b"}),
        ("doc_status", "mget", lambda s: s.filter_keys({"a", "b"}), {"a", "b"}),
        ("graph", "exists", lambda s: s.has_node("Alice"), False),
        (
            "vector",
            "search",
            lambda s: s.query("q", top_k=5, query_embedding=[0.1] * 8),
            [],
        ),
    ],
    ids=["kv-filter_keys", "doc_status-filter_keys", "graph-has_node", "vector-query"],
)
async def test_missing_index_returns_confirmed_empty(
    global_config, kind, rpc_attr, invoke, expected
):
    """A genuinely missing index stays confirmed-empty across fail-open reads."""
    storage, client = await _build_storage(kind, global_config)
    setattr(client, rpc_attr, AsyncMock(side_effect=missing_index_error()))
    assert await invoke(storage) == expected


# ---------------------------------------------------------------------------
# KV strict point reads, malformed items, and confirmed-absent happy paths
# ---------------------------------------------------------------------------


async def test_kv_get_by_id_none_only_for_confirmed_absent(global_config):
    storage, client = await _build_storage("kv", global_config)
    client.mget = AsyncMock(return_value={"docs": [{"_id": "doc-1", "found": False}]})
    assert await storage.get_by_id("doc-1") is None
    client.mget = AsyncMock(side_effect=missing_index_error())
    assert await storage.get_by_id("doc-1") is None


@pytest.mark.parametrize(
    "docs",
    [
        [],
        [{"_id": "doc-1"}],
        [{"_id": "doc-1", "found": True}],
        [{"_id": "other", "found": True, "_source": {"k": 1}}],
    ],
    ids=["empty-docs", "missing-found-flag", "found-without-source", "id-mismatch"],
)
async def test_kv_get_by_id_raises_on_malformed_or_missing_item(global_config, docs):
    storage, client = await _build_storage("kv", global_config)
    client.mget = AsyncMock(return_value={"docs": docs})
    with pytest.raises(RuntimeError):
        await storage.get_by_id("doc-1")


async def test_kv_get_by_ids_maps_found_and_confirmed_absent(global_config):
    storage, client = await _build_storage("kv", global_config)
    client.mget = AsyncMock(
        return_value={
            "docs": [
                {"_id": "doc-1", "found": True, "_source": {"k": "v1"}},
                {"_id": "doc-2", "found": False},
            ]
        }
    )
    result = await storage.get_by_ids(["doc-1", "doc-2"])
    assert result[0]["k"] == "v1"
    assert result[0]["_id"] == "doc-1"
    assert result[1] is None


async def test_kv_filter_keys_returns_only_confirmed_absent(global_config):
    storage, client = await _build_storage("kv", global_config)

    async def _side_effect(index=None, body=None, **kwargs):
        return {
            "docs": [
                {"_id": doc_id, "found": doc_id == "here"} for doc_id in body["ids"]
            ]
        }

    client.mget = AsyncMock(side_effect=_side_effect)
    assert await storage.filter_keys({"here", "gone"}) == {"gone"}


async def test_kv_strict_point_read_three_state(global_config):
    """``OpenSearchKVStorage.get_by_id_strict`` three-state contract over buffers,
    healthy misses, not-ready indices, and live index_not_found."""
    # Buffered upsert -> present without hitting mget.
    storage, client = await _build_storage("kv", global_config)
    client.mget = AsyncMock()
    await storage.upsert({"doc-1": {"content": "x"}})
    doc = await storage.get_by_id_strict("doc-1")
    assert doc["content"] == "x"
    assert doc["_id"] == "doc-1"
    client.mget.assert_not_awaited()

    # Pending delete -> confirmed absent without hitting mget.
    await storage.delete(["doc-1"])
    assert await storage.get_by_id_strict("doc-1") is None
    client.mget.assert_not_awaited()

    # Healthy mget miss -> confirmed absent (None).
    storage2, client2 = await _build_storage("kv", global_config)
    client2.mget = AsyncMock(return_value={"docs": [{"_id": "doc-1", "found": False}]})
    assert await storage2.get_by_id_strict("doc-1") is None

    # Live index_not_found -> raises StorageControlPlaneError and marks not ready.
    client2.mget = AsyncMock(side_effect=missing_index_error())
    with pytest.raises(StorageControlPlaneError):
        await storage2.get_by_id_strict("doc-1")
    assert storage2._index_ready is False

    # _index_ready=False -> strict raises while relaxed get_by_id returns None.
    with pytest.raises(StorageControlPlaneError):
        await storage2.get_by_id_strict("doc-1")
    assert await storage2.get_by_id("doc-1") is None


# ---------------------------------------------------------------------------
# Graph batch-read happy paths & regression pins
# ---------------------------------------------------------------------------


async def test_graph_get_nodes_batch_dedups_and_omits_confirmed_absent(global_config):
    storage, client = await _build_storage("graph", global_config)
    client.mget = AsyncMock(
        return_value={
            "docs": [
                {"_id": "Alice", "found": True, "_source": {"entity_id": "Alice"}},
                {"_id": "Bob", "found": False},
            ]
        }
    )
    result = await storage.get_nodes_batch(["Alice", "Bob", "Alice"])
    assert client.mget.await_args.kwargs["body"] == {"ids": ["Alice", "Bob"]}
    assert result == {"Alice": {"entity_id": "Alice", "_id": "Alice"}}


async def test_graph_get_edges_batch_is_one_mget_keyed_as_requested(global_config):
    storage, client = await _build_storage("graph", global_config)
    ab = _canonical_edge_id("A", "B")
    ac = _canonical_edge_id("A", "C")
    client.mget = AsyncMock(
        return_value={
            "docs": [
                _edge_doc(ab, {"source_id": "chunk-1", "weight": 1.0}),
                _edge_doc(ac),
            ]
        }
    )

    result = await storage.get_edges_batch(
        [{"src": "B", "tgt": "A"}, {"src": "A", "tgt": "C"}, {"src": "A", "tgt": "B"}]
    )

    assert client.mget.await_count == 1
    assert client.mget.await_args.kwargs["body"] == {"ids": [ab, ac]}
    assert sorted(result) == [("A", "B"), ("B", "A")]
    assert result[("B", "A")] == {"source_id": "chunk-1", "weight": 1.0, "_id": ab}
    assert result[("A", "B")] is not result[("B", "A")]


async def test_graph_get_edges_batch_empty_input_reads_nothing(global_config):
    storage, client = await _build_storage("graph", global_config)
    client.mget = AsyncMock(side_effect=AssertionError("must not be called"))
    assert await storage.get_edges_batch([]) == {}
    assert client.mget.await_count == 0


async def test_get_nodes_edges_batch_success_path_returns_dict_not_none(global_config):
    storage, client = await _build_storage("graph", global_config)
    client.search = AsyncMock(
        return_value={
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "source_node_id": "A",
                            "target_node_id": "B",
                        },
                        "sort": ["A", "B"],
                    }
                ]
            }
        }
    )
    result = await storage.get_nodes_edges_batch(["A", "B"])
    assert result is not None
    assert result["A"] == [("A", "B")]
    assert result["B"] == [("A", "B")]
