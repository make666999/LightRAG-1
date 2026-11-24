"""Tests for ``DELETE /documents`` (clear all): configuration records, ingress mailbox, LLM cache, parsed files, and drop status handling."""

from __future__ import annotations

import asyncio
import importlib
import shutil
import sys
import threading
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pytest

from lightrag import config_store as cs

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

from lightrag.constants import PARSED_DIR_NAME  # noqa: E402
from lightrag.kg.networkx_impl import NetworkXStorage  # noqa: E402
from lightrag.kg.pipeline_ingress import (  # noqa: E402
    ManualRetryPublishResult,
    PipelineIngressMessage,
)
from lightrag.kg.scan_job_store import ScanJobStatus  # noqa: E402
from lightrag.kg.shared_storage import get_pipeline_ingress  # noqa: E402
from lightrag.utils import EmbeddingFunc  # noqa: E402

DocumentManager = _document_routes.DocumentManager
create_document_routes = _document_routes.create_document_routes

pytestmark = pytest.mark.offline


# --------------------------------------------------------------------------- #
# Shared storage doubles and helpers
# --------------------------------------------------------------------------- #


class _NoopStorage:
    """Minimal storage the clear endpoint can ``drop()`` and report on."""

    namespace = "noop"

    async def drop(self):
        return {"status": "success", "message": "data dropped"}


class _FailingStorage:
    namespace = "failing"

    async def drop(self):
        return {"status": "error", "message": "backend refused the drop"}


class _DropResultStorage:
    def __init__(self, result: dict, namespace: str = "drop"):
        self._result = result
        self.namespace = namespace

    async def drop(self):
        return self._result


class _CancelledStorage:
    """A drop whose task is cancelled while the others run."""

    namespace = "cancelled"

    async def drop(self):
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        await asyncio.sleep(0)
        raise AssertionError("the drop was expected to be cancelled")


class _TrackedDocStatusStorage(_NoopStorage):
    """Records whether ``initialize()`` ran again after ``drop()``."""

    def __init__(self):
        self.dropped = False
        self.reinitialized_after_drop = False

    async def drop(self):
        self.dropped = True
        return await super().drop()

    async def initialize(self):
        if self.dropped:
            self.reinitialized_after_drop = True


class _VectorStorage(_document_routes.BaseVectorStorage):
    """Real ``BaseVectorStorage`` with abstract surface waived to test ``drop_all_models``."""

    namespace = "vectors"

    def __init__(self, workspace, sibling_result=None):
        self.workspace = workspace
        self.calls: list[str] = []
        self.sibling_result = sibling_result or {"status": "success", "message": ""}

    async def drop(self):
        self.calls.append("drop")
        return {"status": "success", "message": "data dropped"}

    async def drop_sibling_models(self, vouched=()):
        self.calls.append("siblings")
        return self.sibling_result


_VectorStorage.__abstractmethods__ = frozenset()


class _FakeConfigKV:
    supports_strict_point_reads = True

    def __init__(self, workspace):
        self.rows = {
            cs.embedding_baseline_key(workspace, t): cs.make_config_row(
                scope_workspace=workspace,
                suffix=cs.embedding_baseline_suffix(t),
                value={"model": "m", "dim": 8, "origin": "probe"},
                updated_by="test",
            )
            for t in cs.EMBEDDING_TARGETS
        }
        self.deleted: list[list[str]] = []
        self.flushes = 0

    async def get_by_id_strict(self, key):
        row = self.rows.get(key)
        return None if row is None else dict(row)

    async def delete(self, ids):
        self.deleted.append(list(ids))
        for key in ids:
            self.rows.pop(key, None)

    async def index_done_callback(self):
        self.flushes += 1


class _ClearRag:
    def __init__(
        self,
        workspace: str,
        *,
        failing_chunks: bool = False,
        cancelled_chunks: bool = False,
        cache_error: Exception | None = None,
        with_config_storage: bool = False,
        track_doc_status_reinit: bool = False,
        chunks_vdb_drop_result: dict | None = None,
    ):
        self.workspace = workspace
        self.cache_error = cache_error
        self.aclear_cache_calls = 0
        self.busy_during_cache_clear: bool | None = None
        self.destructive_busy_during_cache_clear: bool | None = None

        storage = _NoopStorage()
        storage.workspace = workspace
        if failing_chunks or cancelled_chunks:
            chunks = _FailingStorage() if failing_chunks else _CancelledStorage()
            chunks.workspace = workspace
            self.text_chunks = chunks
        else:
            self.text_chunks = storage
        self.full_docs = storage
        self.full_entities = storage
        self.full_relations = storage
        self.entity_chunks = storage
        self.relation_chunks = storage
        self.entities_vdb = storage
        self.relationships_vdb = storage
        if chunks_vdb_drop_result is not None:
            chunks_vdb = _DropResultStorage(chunks_vdb_drop_result, "chunks")
            chunks_vdb.workspace = workspace
            self.chunks_vdb = chunks_vdb
        else:
            self.chunks_vdb = storage
        self.chunk_entity_relation_graph = storage
        if track_doc_status_reinit:
            doc_status = _TrackedDocStatusStorage()
            doc_status.workspace = workspace
            self.doc_status = doc_status
        else:
            self.doc_status = storage
        if with_config_storage:
            self.configuration_storage = _FakeConfigKV(workspace)

    async def aclear_cache(self, modes=None):
        from lightrag.kg.shared_storage import get_namespace_data

        self.aclear_cache_calls += 1
        status = await get_namespace_data("pipeline_status", workspace=self.workspace)
        self.busy_during_cache_clear = bool(status.get("busy"))
        self.destructive_busy_during_cache_clear = bool(status.get("destructive_busy"))
        if self.cache_error is not None:
            raise self.cache_error


def _clear_endpoint(rag, input_dir):
    router = create_document_routes(rag, DocumentManager(str(input_dir)))
    return [
        route.endpoint
        for route in router.routes
        if getattr(route, "name", "") == "clear_documents"
    ][-1]


async def _init_workspace(workspace):
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    shared_storage.initialize_share_data()
    await shared_storage.initialize_pipeline_status(workspace=workspace)
    return shared_storage


# --------------------------------------------------------------------------- #
# Configuration records cleanup ordering (Scenarios 13 & 14)
# --------------------------------------------------------------------------- #


async def test_every_drop_succeeding_deletes_the_three_records_last(tmp_path):
    """Scenario 14, the endpoint's half."""
    workspace = f"clear-config-ok-{uuid4().hex[:8]}"
    await _init_workspace(workspace)
    rag = _ClearRag(workspace, with_config_storage=True)

    response = await _clear_endpoint(rag, tmp_path)()

    assert response.status == "success"
    config = rag.configuration_storage
    assert config.rows == {}
    assert len(config.deleted) == 1
    assert set(config.deleted[0]) == {
        cs.embedding_baseline_key(workspace, t) for t in cs.EMBEDDING_TARGETS
    }
    assert config.flushes == 1


async def test_a_partial_drop_keeps_all_three_records(tmp_path):
    """Scenario 13: partial drop keeps all configuration records."""
    workspace = f"clear-config-partial-{uuid4().hex[:8]}"
    await _init_workspace(workspace)
    rag = _ClearRag(workspace, failing_chunks=True, with_config_storage=True)

    response = await _clear_endpoint(rag, tmp_path)()

    assert response.status == "partial_success"
    config = rag.configuration_storage
    assert config.deleted == []
    assert len(config.rows) == 3


async def test_a_failed_record_delete_is_reported_not_hidden(tmp_path):
    class _Stubborn(_FakeConfigKV):
        async def delete(self, ids):
            self.deleted.append(list(ids))

    workspace = f"clear-config-stuck-{uuid4().hex[:8]}"
    await _init_workspace(workspace)
    rag = _ClearRag(workspace, with_config_storage=True)
    rag.configuration_storage = _Stubborn(workspace)

    response = await _clear_endpoint(rag, tmp_path)()

    assert response.status == "partial_success"
    assert "configuration records could not be deleted" in response.message


async def test_a_failed_cache_drop_keeps_all_three_records(tmp_path):
    """A failed opt-in cache drop keeps all three configuration records."""
    workspace = f"clear-config-cache-{uuid4().hex[:8]}"
    await _init_workspace(workspace)
    rag = _ClearRag(
        workspace,
        cache_error=RuntimeError("cache backend refused the drop"),
        with_config_storage=True,
    )

    response = await _clear_endpoint(rag, tmp_path)(clear_llm_cache=True)

    assert response.status == "partial_success"
    assert "LLM response cache could not be cleared" in response.message
    config = rag.configuration_storage
    assert config.deleted == []
    assert len(config.rows) == 3


async def test_a_successful_cache_drop_still_lets_the_records_go(tmp_path):
    workspace = f"clear-config-cache-ok-{uuid4().hex[:8]}"
    await _init_workspace(workspace)
    rag = _ClearRag(workspace, with_config_storage=True)

    response = await _clear_endpoint(rag, tmp_path)(clear_llm_cache=True)

    assert response.status == "success"
    assert rag.aclear_cache_calls == 1
    assert rag.configuration_storage.rows == {}
    assert len(rag.configuration_storage.deleted) == 1


async def test_a_cancelled_drop_keeps_all_three_records(tmp_path):
    """A cancelled drop is not a success; configuration records and LLM cache stay."""
    workspace = f"clear-config-cancelled-{uuid4().hex[:8]}"
    await _init_workspace(workspace)
    rag = _ClearRag(workspace, cancelled_chunks=True, with_config_storage=True)

    response = await _clear_endpoint(rag, tmp_path)(clear_llm_cache=True)

    assert response.status == "partial_success"
    config = rag.configuration_storage
    assert config.deleted == [], (
        "a cancelled drop was counted as a success and the records went with it"
    )
    assert len(config.rows) == 3
    assert rag.aclear_cache_calls == 0, (
        "a cancelled drop was counted as a success and the LLM cache went too"
    )


async def test_a_vector_storage_is_cleared_for_every_embedding_model(tmp_path):
    workspace = f"clear-config-siblings-{uuid4().hex[:8]}"
    await _init_workspace(workspace)
    rag = _ClearRag(workspace, with_config_storage=True)
    vdbs = [_VectorStorage(workspace) for _ in range(3)]
    rag.entities_vdb, rag.relationships_vdb, rag.chunks_vdb = vdbs

    response = await _clear_endpoint(rag, tmp_path)()

    assert response.status == "success"
    assert [vdb.calls for vdb in vdbs] == [["drop", "siblings"]] * 3
    assert rag.configuration_storage.rows == {}


async def test_a_sibling_left_behind_keeps_all_three_records(tmp_path):
    workspace = f"clear-config-sibling-left-{uuid4().hex[:8]}"
    await _init_workspace(workspace)
    rag = _ClearRag(workspace, with_config_storage=True)
    rag.chunks_vdb = _VectorStorage(
        workspace, sibling_result={"status": "error", "message": "left a_chunks_x_8d"}
    )

    response = await _clear_endpoint(rag, tmp_path)()

    assert response.status == "partial_success"
    assert rag.chunks_vdb.calls == ["drop", "siblings"]
    config = rag.configuration_storage
    assert config.deleted == []
    assert len(config.rows) == 3
    assert "a_chunks_x_8d" not in response.message


# --------------------------------------------------------------------------- #
# Ingress mailbox, NetworkX drop notification, scan jobs, and doc_status reinit
# --------------------------------------------------------------------------- #


async def test_clear_documents_clears_ingress_and_refuses_replay(tmp_path):
    workspace = f"clear-ingress-{uuid4().hex[:8]}"
    shared_storage = await _init_workspace(workspace)

    ingress = await get_pipeline_ingress(workspace)
    ingress.put_document(PipelineIngressMessage(kind="document", doc_id="doc-x"))
    ingress.request_auto_rescan()
    manual_msg = PipelineIngressMessage(
        kind="rescan", retry_failed=True, request_id="req-cleared"
    )
    assert (
        ingress.request_manual_retry("req-cleared", manual_msg)
        is ManualRetryPublishResult.ACCEPTED
    )

    rag = _ClearRag(workspace, track_doc_status_reinit=True)
    response = await _clear_endpoint(rag, tmp_path)()
    assert response.status in ("success", "partial_success")

    counts = ingress.counts()
    assert counts["documents"] == 0
    assert counts["auto_rescan_pending"] is False
    assert counts["manual_retries"] == 0
    assert ingress.has_work() is False

    assert (
        ingress.request_manual_retry("req-cleared", manual_msg)
        is ManualRetryPublishResult.ALREADY_TERMINAL
    )
    assert ingress.snapshot_manual_retries() == []

    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=workspace
    )
    assert pipeline_status.get("busy") is False
    assert pipeline_status.get("destructive_busy") is False


async def test_clear_documents_survives_ingress_clear_failure(tmp_path):
    workspace = f"clear-ingress-fail-{uuid4().hex[:8]}"
    shared_storage = await _init_workspace(workspace)

    ingress = await get_pipeline_ingress(workspace)
    ingress.put_document(PipelineIngressMessage(kind="document", doc_id="doc-x"))

    def dead_clear():
        raise RuntimeError("manager down: clear")

    ingress.clear = dead_clear

    rag = _ClearRag(workspace, track_doc_status_reinit=True)
    response = await _clear_endpoint(rag, tmp_path)()
    assert response.status in ("success", "partial_success")

    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=workspace
    )
    assert pipeline_status.get("busy") is False
    assert pipeline_status.get("destructive_busy") is False


async def test_clear_documents_counts_notification_failed_networkx_drop_as_success(
    tmp_path, monkeypatch
):
    workspace = f"clear-networkx-notify-{uuid4().hex[:8]}"
    await _init_workspace(workspace)
    networkx_impl = importlib.import_module("lightrag.kg.networkx_impl")

    async def embed(texts):
        return np.random.rand(len(texts), 8)

    graph = NetworkXStorage(
        namespace="chunk_entity_relation",
        workspace=workspace,
        global_config={
            "working_dir": str(tmp_path),
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.5},
        },
        embedding_func=EmbeddingFunc(embedding_dim=8, max_token_size=512, func=embed),
    )
    await graph.initialize()
    try:
        await graph.upsert_node("n1", {"entity_id": "n1"})
        await graph.index_done_callback()

        async def notification_boom(namespace, workspace=None):
            raise RuntimeError("notification boom")

        monkeypatch.setattr(networkx_impl, "set_all_update_flags", notification_boom)
        input_file = tmp_path / "cleared-input.txt"
        input_file.write_text("already cleared graph data")

        rag = _ClearRag(workspace, track_doc_status_reinit=True)
        rag.chunk_entity_relation_graph = graph

        logged_errors = []
        monkeypatch.setattr(networkx_impl.logger, "error", logged_errors.append)
        response = await _clear_endpoint(rag, tmp_path)()

        assert response.status == "success"
        assert not input_file.exists()
        assert not Path(graph._graphml_xml_file).exists()
        assert any("some processes may not reload" in msg for msg in logged_errors)
    finally:
        await graph.finalize()


async def test_clear_documents_retires_finished_scan_jobs_only(tmp_path):
    workspace = f"clear-jobs-{uuid4().hex[:8]}"
    shared_storage = await _init_workspace(workspace)

    store = shared_storage.get_scan_job_store(workspace)
    live_token, done_token = uuid4().hex, uuid4().hex
    store.create("scan-live", live_token)
    store.create("scan-done", done_token)
    finished = store.get("scan-done")
    assert store.set_status(
        "scan-done",
        done_token,
        ScanJobStatus.COMPLETED,
        expected_version=finished["version"],
    ).ok

    rag = _ClearRag(workspace, track_doc_status_reinit=True)
    response = await _clear_endpoint(rag, tmp_path)()
    assert response.status in ("success", "partial_success")

    assert [record["track_id"] for record in store.snapshot()] == ["scan-live"]
    assert store.get("scan-live")["status"] == "running"


async def test_clear_documents_reinitializes_doc_status_after_drop(tmp_path):
    workspace = f"clear-reinit-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    rag = _ClearRag(workspace, track_doc_status_reinit=True)
    response = await _clear_endpoint(rag, tmp_path)()
    assert response.status in ("success", "partial_success")

    assert rag.doc_status.dropped is True
    assert rag.doc_status.reinitialized_after_drop is True


# --------------------------------------------------------------------------- #
# LLM response cache opt-in drop and error sanitization
# --------------------------------------------------------------------------- #


async def test_clear_documents_preserves_llm_cache_by_default(tmp_path):
    workspace = f"clear-cache-default-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    rag = _ClearRag(workspace)
    response = await _clear_endpoint(rag, tmp_path)()

    assert response.status == "success"
    assert rag.aclear_cache_calls == 0
    assert "LLM response cache" not in response.message


async def test_clear_documents_drops_llm_cache_when_opted_in(tmp_path):
    workspace = f"clear-cache-optin-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    rag = _ClearRag(workspace)
    response = await _clear_endpoint(rag, tmp_path)(clear_llm_cache=True)

    assert response.status == "success"
    assert rag.aclear_cache_calls == 1
    assert "Cleared the LLM response cache." in response.message


async def test_llm_cache_drop_runs_inside_the_destructive_reservation(tmp_path):
    workspace = f"clear-cache-reserved-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    rag = _ClearRag(workspace)
    response = await _clear_endpoint(rag, tmp_path)(clear_llm_cache=True)

    assert response.status == "success"
    assert rag.busy_during_cache_clear is True
    assert rag.destructive_busy_during_cache_clear is True


async def test_clear_documents_refuses_while_pipeline_busy(tmp_path):
    workspace = f"clear-cache-busy-{uuid4().hex[:8]}"
    shared_storage = await _init_workspace(workspace)

    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=workspace
    )
    pipeline_status["busy"] = True
    try:
        rag = _ClearRag(workspace)
        response = await _clear_endpoint(rag, tmp_path)(clear_llm_cache=True)

        assert response.status == "busy"
        assert rag.aclear_cache_calls == 0
    finally:
        pipeline_status["busy"] = False


async def test_llm_cache_drop_failure_degrades_to_partial_success(tmp_path):
    workspace = f"clear-cache-failure-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    rag = _ClearRag(workspace, cache_error=RuntimeError("cache backend down"))
    response = await _clear_endpoint(rag, tmp_path)(clear_llm_cache=True)

    assert response.status == "partial_success"
    assert rag.aclear_cache_calls == 1
    assert "Cleared the LLM response cache." not in response.message


async def test_partial_success_names_the_cache_failure(tmp_path):
    workspace = f"clear-cache-named-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    rag = _ClearRag(workspace, cache_error=RuntimeError("cache backend down"))
    response = await _clear_endpoint(rag, tmp_path)(clear_llm_cache=True)

    assert response.status == "partial_success"
    assert "LLM response cache" in response.message
    assert "error_id:" in response.message


async def test_the_cache_failure_does_not_leak_the_backend_text(tmp_path):
    workspace = f"clear-cache-leak-{uuid4().hex[:8]}"
    shared_storage = await _init_workspace(workspace)

    rag = _ClearRag(workspace, cache_error=RuntimeError("cache backend down"))
    response = await _clear_endpoint(rag, tmp_path)(clear_llm_cache=True)

    assert response.status == "partial_success"
    assert "cache backend down" not in response.message

    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=workspace
    )
    history = list(pipeline_status.get("history_messages", []))
    assert not any("cache backend down" in entry for entry in history)


async def test_a_failed_storage_drop_does_not_leak_the_backend_text(tmp_path):
    workspace = f"clear-storage-leak-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    rag = _ClearRag(workspace, failing_chunks=True)
    response = await _clear_endpoint(rag, tmp_path)(clear_llm_cache=True)

    assert response.status == "partial_success"
    assert "backend refused the drop" not in response.message
    assert "_FailingStorage drop failed" in response.message
    assert "error_id:" in response.message


async def test_standalone_clear_cache_route_is_gone(tmp_path):
    workspace = f"clear-cache-route-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    rag = _ClearRag(workspace)
    router = create_document_routes(rag, DocumentManager(str(tmp_path)))

    paths = {getattr(route, "path", "") for route in router.routes}
    names = {getattr(route, "name", "") for route in router.routes}

    assert "/clear_cache" not in paths
    assert "clear_cache" not in names


async def test_a_partial_storage_drop_preserves_the_llm_cache(tmp_path):
    workspace = f"clear-cache-partial-drop-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    rag = _ClearRag(workspace, failing_chunks=True)
    response = await _clear_endpoint(rag, tmp_path)(clear_llm_cache=True)

    assert response.status == "partial_success"
    assert rag.aclear_cache_calls == 0
    assert "LLM cache preserved" in response.message
    assert "Re-run the clear" in response.message


async def test_the_preserved_cache_is_dropped_once_every_storage_succeeds(tmp_path):
    workspace = f"clear-cache-partial-retry-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    failing = _ClearRag(workspace, failing_chunks=True)
    assert (
        await _clear_endpoint(failing, tmp_path)(clear_llm_cache=True)
    ).status == "partial_success"
    assert failing.aclear_cache_calls == 0

    healthy = _ClearRag(workspace)
    response = await _clear_endpoint(healthy, tmp_path)(clear_llm_cache=True)

    assert response.status == "success"
    assert healthy.aclear_cache_calls == 1


# --------------------------------------------------------------------------- #
# __parsed__ directory preservation and opt-in deletion
# --------------------------------------------------------------------------- #


async def test_clear_documents_preserves_parsed_dir_by_default(tmp_path):
    workspace = f"clear-parsed-default-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    parsed_dir = tmp_path / PARSED_DIR_NAME
    parsed_dir.mkdir()
    (parsed_dir / "a.parsed.json").write_text("{}")

    rag = _ClearRag(workspace)
    response = await _clear_endpoint(rag, tmp_path)()

    assert response.status in ("success", "partial_success")
    assert parsed_dir.exists()
    assert (parsed_dir / "a.parsed.json").exists()
    assert "__parsed__ preserved" in response.message
    assert "delete_parsed_files=true" in response.message


async def test_clear_documents_deletes_parsed_dir_when_opted_in(tmp_path):
    workspace = f"clear-parsed-optin-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    parsed_dir = tmp_path / PARSED_DIR_NAME
    parsed_dir.mkdir()
    (parsed_dir / "a.parsed.json").write_text("{}")

    rag = _ClearRag(workspace)
    response = await _clear_endpoint(rag, tmp_path)(delete_parsed_files=True)

    assert response.status in ("success", "partial_success")
    assert not parsed_dir.exists()
    assert "__parsed__" in response.message


async def test_clear_documents_opt_in_is_a_noop_without_parsed_dir(tmp_path):
    workspace = f"clear-parsed-missing-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    rag = _ClearRag(workspace)
    response = await _clear_endpoint(rag, tmp_path)(delete_parsed_files=True)

    assert response.status in ("success", "partial_success")


async def test_clear_documents_deletes_parsed_dir_off_the_event_loop_thread(tmp_path):
    workspace = f"clear-parsed-thread-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    parsed_dir = tmp_path / PARSED_DIR_NAME
    parsed_dir.mkdir()
    (parsed_dir / "a.parsed.json").write_text("{}")

    main_thread_id = threading.get_ident()
    call_thread_id = {}
    real_rmtree = shutil.rmtree

    def fake_rmtree(path, *args, **kwargs):
        call_thread_id["id"] = threading.get_ident()
        return real_rmtree(path, *args, **kwargs)

    rag = _ClearRag(workspace)
    endpoint = _clear_endpoint(rag, tmp_path)

    with patch.object(shutil, "rmtree", side_effect=fake_rmtree):
        response = await endpoint(delete_parsed_files=True)

    assert response.status in ("success", "partial_success")
    assert not parsed_dir.exists()
    assert call_thread_id["id"] != main_thread_id


async def test_clear_documents_cancel_defers_until_rmtree_finishes_before_releasing_lock(
    tmp_path,
):
    workspace = f"clear-parsed-cancel-{uuid4().hex[:8]}"
    shared_storage = await _init_workspace(workspace)

    parsed_dir = tmp_path / PARSED_DIR_NAME
    parsed_dir.mkdir()
    (parsed_dir / "a.parsed.json").write_text("{}")

    call_started = threading.Event()
    release_call = threading.Event()
    real_rmtree = shutil.rmtree

    def fake_rmtree(path, *args, **kwargs):
        call_started.set()
        release_call.wait(timeout=5)
        return real_rmtree(path, *args, **kwargs)

    rag = _ClearRag(workspace)
    endpoint = _clear_endpoint(rag, tmp_path)

    with patch.object(shutil, "rmtree", side_effect=fake_rmtree):
        task = asyncio.ensure_future(endpoint(delete_parsed_files=True))
        for _ in range(500):
            if call_started.is_set():
                break
            await asyncio.sleep(0.01)
        assert call_started.is_set()

        task.cancel()
        for _ in range(10):
            await asyncio.sleep(0.01)

        pipeline_status = await shared_storage.get_namespace_data(
            "pipeline_status", workspace=workspace
        )
        assert pipeline_status["destructive_busy"] is True

        release_call.set()

        with pytest.raises(asyncio.CancelledError):
            await task

    assert not parsed_dir.exists()
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=workspace
    )
    assert pipeline_status["destructive_busy"] is False


async def test_clear_documents_default_message_silent_without_parsed_dir(tmp_path):
    workspace = f"clear-parsed-silent-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    rag = _ClearRag(workspace)
    response = await _clear_endpoint(rag, tmp_path)()

    assert response.status in ("success", "partial_success")
    assert "__parsed__" not in response.message


# --------------------------------------------------------------------------- #
# Storage drop status handling
# --------------------------------------------------------------------------- #


async def test_clear_documents_honors_drop_error_status(tmp_path):
    """A storage whose drop() returns {"status": "error"} (without raising) must
    NOT be counted as a success: the clear is reported as partial_success."""
    workspace = f"clear-drop-err-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    rag = _ClearRag(
        workspace,
        chunks_vdb_drop_result={
            "status": "error",
            "message": "legacy tagging undetermined",
        },
    )
    response = await _clear_endpoint(rag, tmp_path)()

    assert response.status == "partial_success"


async def test_clear_documents_succeeds_when_all_drops_succeed(tmp_path):
    """Baseline: when every storage drop() returns success the clear reports success."""
    workspace = f"clear-drop-ok-{uuid4().hex[:8]}"
    await _init_workspace(workspace)

    rag = _ClearRag(
        workspace,
        chunks_vdb_drop_result={"status": "success", "message": "data dropped"},
    )
    response = await _clear_endpoint(rag, tmp_path)()

    assert response.status == "success"
