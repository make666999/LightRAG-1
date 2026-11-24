"""Backend-specific deletion failures must preserve document retry state.

Exercises the real MongoDB, OpenSearch, and PostgreSQL storage ``delete`` /
``remove_*`` methods against failing transport seams inside ``adelete_by_doc_id``
so a swallowed delete error can never report success over undeleted data.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

import lightrag.kg.opensearch_impl as opensearch_module
from lightrag.kg.opensearch_impl import (
    OpenSearchDocStatusStorage,
    OpenSearchGraphStorage,
)

from ._fakes import (
    build_delete_test_rag as _build_rag,
    chunk_ids_for_doc as _chunk_ids,
    ingest_test_doc as _ingest,
)

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# MongoDB
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mongo_chunk_delete_failure_preserves_retry_state(tmp_path):
    pytest.importorskip("pymongo", reason="pymongo is required for Mongo storage tests")
    from pymongo.errors import PyMongoError

    from lightrag.kg.mongo_impl import MongoKVStorage

    rag = await _build_rag(tmp_path, f"mongo-delete-{uuid4().hex[:8]}")
    original_delete = rag.text_chunks.delete
    try:
        doc_id = await _ingest(rag)
        chunk_ids = await _chunk_ids(rag, doc_id)
        mongo = MongoKVStorage("text_chunks", {}, None, rag.workspace)
        mongo._data = SimpleNamespace(
            delete_many=AsyncMock(side_effect=PyMongoError("delete unavailable"))
        )
        rag.text_chunks.delete = mongo.delete

        result = await rag.adelete_by_doc_id(doc_id)

        assert result.status == "fail"
        row = await rag.doc_status.get_by_id(doc_id)
        assert row is not None
        assert row["metadata"]["deletion_failure_stage"] == "delete_chunks"
        assert await rag.full_docs.get_by_id(doc_id) is not None
        assert all(await rag.text_chunks.get_by_ids(chunk_ids))

        rag.text_chunks.delete = original_delete
        retry = await rag.adelete_by_doc_id(doc_id)
        assert retry.status == "success"
        assert await rag.doc_status.get_by_id(doc_id) is None
        assert await rag.full_docs.get_by_id(doc_id) is None
        assert not any(await rag.text_chunks.get_by_ids(chunk_ids))
    finally:
        rag.text_chunks.delete = original_delete
        await rag.finalize_storages()


# ---------------------------------------------------------------------------
# OpenSearch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["remove_edges", "remove_nodes"])
async def test_failed_graph_bulk_delete_preserves_document(
    tmp_path, monkeypatch, method
):
    rag = await _build_rag(tmp_path, f"os-delete-{uuid4().hex[:8]}")
    graph = rag.chunk_entity_relation_graph
    original_delete = getattr(graph, method)
    try:
        doc_id = await _ingest(rag)
        chunk_ids = await _chunk_ids(rag, doc_id)
        failure = {
            "delete": {"_id": "failed-id", "status": 503, "error": "unavailable"}
        }
        storage = OpenSearchGraphStorage("graph", {}, None, rag.workspace)
        storage._indices_ready = True
        storage.client = SimpleNamespace(
            bulk=AsyncMock(return_value={"errors": True, "items": [failure]}),
            delete_by_query=AsyncMock(return_value={"failures": []}),
        )
        monkeypatch.setattr(
            opensearch_module,
            "_run_chunked_async_bulk",
            AsyncMock(return_value=(0, [failure])),
        )
        setattr(graph, method, getattr(storage, method))

        result = await rag.adelete_by_doc_id(doc_id)

        assert result.status == "fail"
        assert await rag.doc_status.get_by_id(doc_id) is not None
        assert await rag.full_docs.get_by_id(doc_id) is not None
        assert all(await rag.text_chunks.get_by_ids(chunk_ids))

        setattr(graph, method, original_delete)
        retry = await rag.adelete_by_doc_id(doc_id)
        assert retry.status == "success"
        assert await rag.doc_status.get_by_id(doc_id) is None
    finally:
        setattr(graph, method, original_delete)
        await rag.finalize_storages()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ["item", "transport"])
async def test_failed_status_delete_preserves_retry_record(
    tmp_path, monkeypatch, failure_kind
):
    rag = await _build_rag(tmp_path, f"os-status-delete-{uuid4().hex[:8]}")
    original_delete = rag.doc_status.delete
    try:
        doc_id = await _ingest(rag)
        storage = OpenSearchDocStatusStorage("doc_status", {}, None, rag.workspace)
        storage._index_ready = True
        storage.client = SimpleNamespace()
        failure = {"delete": {"_id": doc_id, "status": 503, "error": "unavailable"}}
        bulk = AsyncMock(return_value=(0, [failure]))
        if failure_kind == "transport":
            bulk.side_effect = opensearch_module.OpenSearchException("unavailable")
        monkeypatch.setattr(opensearch_module, "_run_chunked_async_bulk", bulk)
        rag.doc_status.delete = storage.delete

        result = await rag.adelete_by_doc_id(doc_id)

        assert result.status == "fail"
        assert await rag.doc_status.get_by_id(doc_id) is not None

        rag.doc_status.delete = original_delete
        retry = await rag.adelete_by_doc_id(doc_id)
        assert retry.status == "success"
        assert await rag.doc_status.get_by_id(doc_id) is None
    finally:
        rag.doc_status.delete = original_delete
        await rag.finalize_storages()


# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------


class _FailingPGDB:
    async def _run_with_retry(self, operation):
        raise RuntimeError("delete unavailable")


def _failing_pg_delete(cls, namespace: str, workspace: str):
    storage = object.__new__(cls)
    storage.namespace = namespace
    storage.workspace = workspace
    storage._max_delete_records_per_batch = 0
    storage.db = _FailingPGDB()
    return storage.delete


@pytest.mark.asyncio
async def test_pg_chunk_delete_failure_preserves_retry_state(tmp_path):
    pytest.importorskip("asyncpg")
    from lightrag.kg.postgres_impl import PGKVStorage

    rag = await _build_rag(tmp_path, f"pg-chunks-{uuid4().hex[:8]}")
    original_delete = rag.text_chunks.delete
    try:
        doc_id = await _ingest(rag)
        chunk_ids = await _chunk_ids(rag, doc_id)
        rag.text_chunks.delete = _failing_pg_delete(
            PGKVStorage, rag.text_chunks.namespace, rag.workspace
        )

        result = await rag.adelete_by_doc_id(doc_id)

        assert result.status == "fail"
        row = await rag.doc_status.get_by_id(doc_id)
        assert row is not None
        assert row["metadata"]["deletion_failure_stage"] == "delete_chunks"
        assert await rag.full_docs.get_by_id(doc_id) is not None
        assert all(await rag.text_chunks.get_by_ids(chunk_ids))

        rag.text_chunks.delete = original_delete
        retry = await rag.adelete_by_doc_id(doc_id)
        assert retry.status == "success"
        assert await rag.doc_status.get_by_id(doc_id) is None
        assert await rag.full_docs.get_by_id(doc_id) is None
        assert not any(await rag.text_chunks.get_by_ids(chunk_ids))
    finally:
        rag.text_chunks.delete = original_delete
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_pg_doc_status_delete_failure_keeps_document_intact(tmp_path):
    pytest.importorskip("asyncpg")
    from lightrag.kg.postgres_impl import PGDocStatusStorage

    rag = await _build_rag(tmp_path, f"pg-status-{uuid4().hex[:8]}")
    original_delete = rag.doc_status.delete
    try:
        doc_id = await _ingest(rag)
        rag.doc_status.delete = _failing_pg_delete(
            PGDocStatusStorage, rag.doc_status.namespace, rag.workspace
        )

        result = await rag.adelete_by_doc_id(doc_id)

        assert result.status == "fail"
        assert await rag.doc_status.get_by_id(doc_id) is not None
        assert await rag.full_docs.get_by_id(doc_id) is not None

        rag.doc_status.delete = original_delete
        retry = await rag.adelete_by_doc_id(doc_id)
        assert retry.status == "success"
        assert await rag.doc_status.get_by_id(doc_id) is None
        assert await rag.full_docs.get_by_id(doc_id) is None
    finally:
        rag.doc_status.delete = original_delete
        await rag.finalize_storages()
