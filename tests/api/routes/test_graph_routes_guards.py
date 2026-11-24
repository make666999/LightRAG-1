"""Graph mutation guards: pipeline-busy / admin-write-gate refusal, self-loop rejection, and rename self-loop preservation."""

from __future__ import annotations

import importlib
import sys
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import networkx as nx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from lightrag import utils_graph
from lightrag.utils import compute_mdhash_id, make_relation_chunk_key

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_graph_routes = importlib.import_module("lightrag.api.routers.graph_routes")
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

create_graph_routes = _graph_routes.create_graph_routes
create_document_routes = _document_routes.create_document_routes
check_pipeline_busy_or_raise = _document_routes.check_pipeline_busy_or_raise

pytestmark = pytest.mark.offline

_API_KEY = "test-key"
_HEADERS = {"X-API-Key": _API_KEY}


# --------------------------------------------------------------------------- #
# Pipeline-busy and admin-write-gate guard tests
# --------------------------------------------------------------------------- #


def _make_mock_rag() -> SimpleNamespace:
    return SimpleNamespace(
        workspace="",
        aedit_entity=AsyncMock(
            return_value={
                "entity_name": "Alice",
                "description": "updated",
                "operation_summary": {
                    "merged": False,
                    "merge_status": "not_attempted",
                    "merge_error": None,
                    "operation_status": "success",
                    "target_entity": None,
                    "final_entity": "Alice",
                    "renamed": False,
                },
            }
        ),
        aedit_relation=AsyncMock(return_value={"description": "updated"}),
        acreate_entity=AsyncMock(return_value={"entity_name": "Alice"}),
        acreate_relation=AsyncMock(return_value={"src_id": "a", "tgt_id": "b"}),
        amerge_entities=AsyncMock(return_value={"merged_entity": "Alice"}),
        adelete_by_entity=AsyncMock(
            return_value=SimpleNamespace(
                status="success", message="deleted", doc_id="ignored"
            )
        ),
        adelete_by_relation=AsyncMock(
            return_value=SimpleNamespace(
                status="success", message="deleted", doc_id="ignored"
            )
        ),
    )


def _build_client(rag: SimpleNamespace) -> TestClient:
    app = FastAPI()
    app.include_router(create_graph_routes(rag, api_key=_API_KEY))
    app.include_router(create_document_routes(rag, SimpleNamespace(), api_key=_API_KEY))
    return TestClient(app)


async def _force_busy_guard(_rag) -> None:
    raise HTTPException(
        status_code=409,
        detail=(
            "Pipeline is busy with another operation. "
            "Wait for the running job to finish before editing "
            "the knowledge graph."
        ),
    )


async def _noop_guard(_rag) -> None:
    return None


def _patch_guard(monkeypatch, replacement) -> None:
    monkeypatch.setattr(_graph_routes, "check_pipeline_busy_or_raise", replacement)
    monkeypatch.setattr(_document_routes, "check_pipeline_busy_or_raise", replacement)


_ENDPOINTS = [
    pytest.param(
        "POST",
        "/graph/entity/edit",
        {"entity_name": "Alice", "updated_data": {"description": "x"}},
        id="update_entity",
    ),
    pytest.param(
        "POST",
        "/graph/relation/edit",
        {
            "source_id": "Alice",
            "target_id": "Bob",
            "updated_data": {"description": "x"},
        },
        id="update_relation",
    ),
    pytest.param(
        "POST",
        "/graph/entity/create",
        {"entity_name": "Alice", "entity_data": {"description": "x"}},
        id="create_entity",
    ),
    pytest.param(
        "POST",
        "/graph/relation/create",
        {
            "source_entity": "Alice",
            "target_entity": "Bob",
            "relation_data": {"description": "x"},
        },
        id="create_relation",
    ),
    pytest.param(
        "POST",
        "/graph/entities/merge",
        {"entities_to_change": ["Alic"], "entity_to_change_into": "Alice"},
        id="merge_entities",
    ),
    pytest.param(
        "DELETE",
        "/graph/entity/delete",
        {"entity_name": "Alice"},
        id="delete_entity",
    ),
    pytest.param(
        "DELETE",
        "/graph/relation/delete",
        {"source_entity": "Alice", "target_entity": "Bob"},
        id="delete_relation",
    ),
]

_GATE_METHOD_FOR_PATH = {
    "/graph/entity/edit": "aedit_entity",
    "/graph/relation/edit": "aedit_relation",
    "/graph/entity/create": "acreate_entity",
    "/graph/relation/create": "acreate_relation",
    "/graph/entities/merge": "amerge_entities",
    "/graph/entity/delete": "adelete_by_entity",
    "/graph/relation/delete": "adelete_by_relation",
}


@pytest.mark.parametrize("method, path, body", _ENDPOINTS)
def test_endpoint_refuses_with_409_when_pipeline_busy(method, path, body, monkeypatch):
    rag = _make_mock_rag()
    client = _build_client(rag)
    _patch_guard(monkeypatch, _force_busy_guard)

    response = client.request(method, path, json=body, headers=_HEADERS)

    assert response.status_code == 409, response.text
    payload = response.json()
    assert "Pipeline is busy" in payload["detail"]
    for attr in (
        "aedit_entity",
        "aedit_relation",
        "acreate_entity",
        "acreate_relation",
        "amerge_entities",
        "adelete_by_entity",
        "adelete_by_relation",
    ):
        getattr(rag, attr).assert_not_awaited()


def test_endpoint_passes_through_when_pipeline_idle(monkeypatch):
    rag = _make_mock_rag()
    client = _build_client(rag)
    _patch_guard(monkeypatch, _noop_guard)

    response = client.post(
        "/graph/entity/edit",
        json={"entity_name": "Alice", "updated_data": {"description": "x"}},
        headers=_HEADERS,
    )

    assert response.status_code == 200, response.text
    rag.aedit_entity.assert_awaited_once()


def test_create_entity_message_uses_normalized_result_name(monkeypatch):
    rag = _make_mock_rag()
    rag.acreate_entity.return_value = {"entity_name": "A公司"}
    client = _build_client(rag)
    _patch_guard(monkeypatch, _noop_guard)

    response = client.post(
        "/graph/entity/create",
        json={"entity_name": "“Ａ 公 司”", "entity_data": {"description": "x"}},
        headers=_HEADERS,
    )

    assert response.status_code == 200, response.text
    assert response.json()["message"] == "Entity 'A公司' created successfully"


def test_merge_entity_message_uses_normalized_result_name(monkeypatch):
    rag = _make_mock_rag()
    rag.amerge_entities.return_value = {"entity_name": "T目标"}
    client = _build_client(rag)
    _patch_guard(monkeypatch, _noop_guard)

    response = client.post(
        "/graph/entities/merge",
        json={
            "entities_to_change": ["Ｓｏｕｒｃｅ 公 司"],
            "entity_to_change_into": "“Ｔ 目 标”",
        },
        headers=_HEADERS,
    )

    assert response.status_code == 200, response.text
    assert response.json()["message"] == "Successfully merged 1 entities into 'T目标'"


async def _with_pipeline_status(action):
    from lightrag.kg.shared_storage import (
        finalize_share_data,
        get_namespace_data,
        initialize_pipeline_status,
        initialize_share_data,
    )

    initialize_share_data()
    try:
        await initialize_pipeline_status(workspace="")
        pipeline_status = await get_namespace_data("pipeline_status", workspace="")
        await action(pipeline_status)
    finally:
        finalize_share_data()


async def test_helper_raises_409_when_busy_flag_set():
    async def _do(pipeline_status):
        pipeline_status["busy"] = True
        rag = SimpleNamespace(workspace="")
        with pytest.raises(HTTPException) as exc_info:
            await check_pipeline_busy_or_raise(rag)
        assert exc_info.value.status_code == 409
        assert "Pipeline is busy" in exc_info.value.detail

    await _with_pipeline_status(_do)


async def test_helper_returns_silently_when_pipeline_idle():
    async def _do(pipeline_status):
        pipeline_status["busy"] = False
        rag = SimpleNamespace(workspace="")
        await check_pipeline_busy_or_raise(rag)

    await _with_pipeline_status(_do)


async def test_helper_raises_503_when_recovery_required():
    async def _do(pipeline_status):
        pipeline_status["busy"] = False
        pipeline_status["recovery_required"] = {
            "kind": "delete",
            "owner_key": "busy_owner",
            "operation_record": {"kind": "delete", "doc_id": "doc-1"},
        }
        rag = SimpleNamespace(workspace="")
        with pytest.raises(HTTPException) as exc_info:
            await check_pipeline_busy_or_raise(rag)
        assert exc_info.value.status_code == 503
        assert "fenced" in exc_info.value.detail.lower()

    await _with_pipeline_status(_do)


async def test_helper_is_noop_when_pipeline_status_uninitialized():
    from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data

    initialize_share_data()
    try:
        rag = SimpleNamespace(workspace="__never_bootstrapped__")
        await check_pipeline_busy_or_raise(rag)
    finally:
        finalize_share_data()


@pytest.mark.parametrize("method, path, body", _ENDPOINTS)
def test_admin_lock_refusal_from_the_core_maps_to_409(method, path, body, monkeypatch):
    from lightrag.exceptions import (
        ADMIN_WRITE_LOCK_BUSY_PREFIX,
        AdminWriteGateRefusedError,
    )

    rag = _make_mock_rag()
    getattr(rag, _GATE_METHOD_FOR_PATH[path]).side_effect = AdminWriteGateRefusedError(
        f"{ADMIN_WRITE_LOCK_BUSY_PREFIX}: `x` waited 30s for the workspace admin lock.",
        fence="admin_lock",
    )
    client = _build_client(rag)
    _patch_guard(monkeypatch, _noop_guard)

    response = client.request(method, path, json=body, headers=_HEADERS)

    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail.startswith(ADMIN_WRITE_LOCK_BUSY_PREFIX)
    assert not detail.startswith("Pipeline is busy")


def test_pipeline_busy_refusal_from_the_core_maps_to_409_with_its_own_phrase(
    monkeypatch,
):
    from lightrag.exceptions import (
        ADMIN_WRITE_LOCK_BUSY_PREFIX,
        ADMIN_WRITE_PIPELINE_BUSY_PREFIX,
        AdminWriteGateRefusedError,
    )
    from lightrag.kg.shared_storage import PipelineReservationConflict

    rag = _make_mock_rag()
    rag.acreate_entity.side_effect = AdminWriteGateRefusedError(
        f"{ADMIN_WRITE_PIPELINE_BUSY_PREFIX}. Wait for the running job to finish.",
        conflict=PipelineReservationConflict.BUSY,
        fence="busy",
    )
    client = _build_client(rag)
    _patch_guard(monkeypatch, _noop_guard)

    response = client.post(
        "/graph/entity/create",
        json={"entity_name": "Alice", "entity_data": {"description": "x"}},
        headers=_HEADERS,
    )

    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail.startswith(ADMIN_WRITE_PIPELINE_BUSY_PREFIX)
    assert not detail.startswith(ADMIN_WRITE_LOCK_BUSY_PREFIX)


def test_recovery_required_refusal_from_the_core_maps_to_503(monkeypatch):
    from lightrag.exceptions import AdminWriteGateRefusedError
    from lightrag.kg.shared_storage import PipelineReservationConflict

    rag = _make_mock_rag()
    rag.acreate_entity.side_effect = AdminWriteGateRefusedError(
        "Workspace is fenced for recovery.",
        conflict=PipelineReservationConflict.RECOVERY_REQUIRED,
    )
    client = _build_client(rag)
    _patch_guard(monkeypatch, _noop_guard)

    response = client.post(
        "/graph/entity/create",
        json={"entity_name": "Alice", "entity_data": {"description": "x"}},
        headers=_HEADERS,
    )

    assert response.status_code == 503, response.text


def _hold_exceeded(operation: str):
    from lightrag.exceptions import AdminWriteHoldExceededError

    return AdminWriteHoldExceededError(
        f"Admin write `{operation}` exceeded the admin-write hold ceiling of 180s "
        "(LIGHTRAG_ADMIN_WRITE_MAX_HOLD_SECONDS) and was stopped so it stops "
        "deferring document ingestion. It was inside a region that must not be "
        "interrupted, so that region ran to completion first: the storage commit "
        "it had started IS durable and only the work after it was skipped. "
        "Re-read the entity or relation before retrying. Raise the ceiling if "
        "the embedding round-trip legitimately takes that long."
    )


@pytest.mark.parametrize("method, path, body", _ENDPOINTS)
def test_hold_ceiling_expiry_reaches_the_client_with_actionable_detail(
    method, path, body, monkeypatch
):
    rag = _make_mock_rag()
    gate_method = _GATE_METHOD_FOR_PATH[path]
    getattr(rag, gate_method).side_effect = _hold_exceeded(gate_method)
    client = _build_client(rag)
    _patch_guard(monkeypatch, _noop_guard)

    response = client.request(method, path, json=body, headers=_HEADERS)

    assert response.status_code == 500, response.text
    detail = response.json()["detail"]
    assert "IS durable" in detail
    assert "Re-read the entity or relation before retrying" in detail
    assert "LIGHTRAG_ADMIN_WRITE_MAX_HOLD_SECONDS" in detail
    assert "Internal server error" not in detail
    assert "error_id" not in detail


def test_an_ordinary_failure_still_gets_the_sanitized_500(monkeypatch):
    rag = _make_mock_rag()
    rag.acreate_entity.side_effect = RuntimeError(
        "connection to postgres://user:pw@db.internal:5432 failed"
    )
    client = _build_client(rag)
    _patch_guard(monkeypatch, _noop_guard)

    response = client.post(
        "/graph/entity/create",
        json={"entity_name": "Alice", "entity_data": {"description": "x"}},
        headers=_HEADERS,
    )

    assert response.status_code == 500, response.text
    detail = response.json()["detail"]
    assert "Internal server error" in detail
    assert "db.internal" not in detail
    assert "postgres" not in detail


# --------------------------------------------------------------------------- #
# Self-loop rejection on create and self-loop preservation on entity rename
# --------------------------------------------------------------------------- #


class _NoopLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _VectorStorage:
    def __init__(self):
        self.global_config = {"workspace": ""}
        self.records = {}

    async def upsert(self, data):
        self.records.update(deepcopy(data))

    async def delete(self, record_ids):
        for record_id in record_ids:
            self.records.pop(record_id, None)

    async def index_done_callback(self):
        return None


@pytest.mark.asyncio
async def test_acreate_relation_rejects_self_loop():
    with pytest.raises(ValueError, match="self-loop relation on 'A'"):
        await utils_graph.acreate_relation(
            chunk_entity_relation_graph=None,
            entities_vdb=None,
            relationships_vdb=None,
            source_entity="A",
            target_entity="A",
            relation_data={"description": "A refers to itself"},
        )


@pytest.mark.asyncio
async def test_acreate_relation_still_accepts_distinct_endpoints(monkeypatch):
    monkeypatch.setattr(
        utils_graph,
        "get_storage_keyed_lock",
        lambda *args, **kwargs: _NoopLock(),
    )

    upserted_edges = []

    class _Graph:
        async def has_node(self, entity_name):
            return True

        async def has_edge(self, source_entity, target_entity):
            return False

        async def get_edge(self, source_entity, target_entity):
            return {"description": "A relates to B", "keywords": "k", "weight": 1.0}

        async def upsert_edge(self, source_entity, target_entity, edge_data):
            upserted_edges.append((source_entity, target_entity))

        async def index_done_callback(self):
            return None

    await utils_graph.acreate_relation(
        _Graph(),
        _VectorStorage(),
        _VectorStorage(),
        "A",
        "B",
        {"description": "A relates to B", "keywords": "k"},
    )

    assert upserted_edges == [("A", "B")]


class _NxGraph:
    """NetworkX-backed double with real incident-edge deletion semantics."""

    def __init__(self):
        self.graph = nx.Graph()

    async def has_node(self, entity_name):
        return self.graph.has_node(entity_name)

    async def get_node(self, entity_name):
        data = self.graph.nodes.get(entity_name)
        return deepcopy(data) if data is not None else None

    async def upsert_node(self, entity_name, node_data):
        self.graph.add_node(entity_name, **deepcopy(node_data))

    async def get_node_edges(self, entity_name):
        return list(self.graph.edges(entity_name))

    async def get_edge(self, source_entity, target_entity):
        data = self.graph.get_edge_data(source_entity, target_entity)
        return deepcopy(data) if data is not None else None

    async def upsert_edge(self, source_entity, target_entity, edge_data):
        self.graph.add_edge(source_entity, target_entity, **deepcopy(edge_data))

    async def delete_node(self, entity_name):
        self.graph.remove_node(entity_name)

    async def index_done_callback(self):
        return None


class _ChunkTrackingStorage:
    def __init__(self, records):
        self.records = deepcopy(records)

    async def get_by_id(self, record_id):
        value = self.records.get(record_id)
        return deepcopy(value) if value is not None else None

    async def upsert(self, data):
        self.records.update(deepcopy(data))

    async def delete(self, record_ids):
        for record_id in record_ids:
            self.records.pop(record_id, None)

    async def index_done_callback(self):
        return None


@pytest.mark.asyncio
async def test_entity_rename_preserves_self_loop_across_all_storages(monkeypatch):
    monkeypatch.setattr(
        utils_graph,
        "get_storage_keyed_lock",
        lambda *args, **kwargs: _NoopLock(),
    )

    graph = _NxGraph()
    await graph.upsert_node(
        "A", {"entity_id": "A", "description": "source", "source_id": "chunk-a"}
    )
    await graph.upsert_node(
        "C", {"entity_id": "C", "description": "neighbour", "source_id": "chunk-c"}
    )
    self_loop = {
        "description": "A refers to itself",
        "keywords": "self",
        "source_id": "chunk-self",
        "weight": 2.0,
    }
    neighbour = {
        "description": "A relates to C",
        "keywords": "neighbour",
        "source_id": "chunk-neighbour",
        "weight": 1.0,
    }
    await graph.upsert_edge("A", "A", self_loop)
    await graph.upsert_edge("A", "C", neighbour)

    entities_vdb = _VectorStorage()
    relationships_vdb = _VectorStorage()
    old_self_id = compute_mdhash_id("AA", prefix="rel-")
    old_neighbour_id = compute_mdhash_id("AC", prefix="rel-")
    relationships_vdb.records = {
        old_self_id: {"src_id": "A", "tgt_id": "A"},
        old_neighbour_id: {"src_id": "A", "tgt_id": "C"},
    }

    old_self_key = make_relation_chunk_key("A", "A")
    old_neighbour_key = make_relation_chunk_key("A", "C")
    relation_chunks = _ChunkTrackingStorage(
        {
            old_self_key: {"chunk_ids": ["chunk-self"], "count": 1},
            old_neighbour_key: {"chunk_ids": ["chunk-neighbour"], "count": 1},
        }
    )

    result = await utils_graph.aedit_entity(
        graph,
        entities_vdb,
        relationships_vdb,
        "A",
        {"entity_name": "B"},
        relation_chunks_storage=relation_chunks,
    )

    assert result["operation_summary"]["operation_status"] == "success"
    assert set(graph.graph.nodes) == {"B", "C"}
    assert {frozenset(edge) for edge in graph.graph.edges} == {
        frozenset(("B", "B")),
        frozenset(("B", "C")),
    }
    assert graph.graph.get_edge_data("B", "B") == self_loop
    assert graph.graph.get_edge_data("B", "C") == neighbour

    new_self_id = compute_mdhash_id("BB", prefix="rel-")
    new_neighbour_id = compute_mdhash_id("BC", prefix="rel-")
    assert set(relationships_vdb.records) == {new_self_id, new_neighbour_id}
    assert relationships_vdb.records[new_self_id]["src_id"] == "B"
    assert relationships_vdb.records[new_self_id]["tgt_id"] == "B"

    assert set(relation_chunks.records) == {
        make_relation_chunk_key("B", "B"),
        make_relation_chunk_key("B", "C"),
    }
    assert relation_chunks.records[make_relation_chunk_key("B", "B")]["chunk_ids"] == [
        "chunk-self"
    ]
