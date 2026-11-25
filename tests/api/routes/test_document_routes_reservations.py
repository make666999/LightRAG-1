import asyncio
import importlib
import sys
from io import BytesIO
from types import SimpleNamespace
from uuid import uuid4

import pytest

from lightrag.chunker import chunking_by_token_size

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
_base = importlib.import_module("lightrag.base")
sys.argv = _original_argv

DeletionResult = _base.DeletionResult
SourceAbsent = _base.SourceAbsent
DocumentManager = _document_routes.DocumentManager
create_document_routes = _document_routes.create_document_routes
run_scanning_process = _document_routes.run_scanning_process

pytestmark = pytest.mark.offline


def _get_endpoint(router, name: str):
    return [
        route.endpoint for route in router.routes if getattr(route, "name", "") == name
    ][-1]


async def _await_managed(managed_tasks):
    """Await every managed background task to completion so each child's finally
    runs (releasing its reservation). Snapshot first — the task's done-callback
    discards it from the set."""
    for task in list(managed_tasks):
        try:
            await task
        except asyncio.CancelledError:
            pass


@pytest.fixture(autouse=True)
def _ensure_shared_storage_initialized():
    """Initialize shared_storage dicts before each test and reset flags after."""
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    shared_storage.initialize_share_data()
    yield
    if shared_storage._shared_dicts is not None:
        for key in list(shared_storage._shared_dicts.keys()):
            if key.endswith("pipeline_status") or key == "pipeline_status":
                ns = shared_storage._shared_dicts[key]
                if isinstance(ns, dict):
                    ns["busy"] = False
                    ns["scanning"] = False


class _EmptyDocStatus:
    async def get_doc_by_file_basename(self, basename: str):
        return None

    async def resolve_doc_source_strict(self, canonical_source_key: str):
        return SourceAbsent()


class _ReservationRag:
    """Minimal LightRAG double covering upload, scan, delete, and reprocess routes."""

    def __init__(self, prefix: str = "res-test"):
        self.workspace = f"{prefix}-{uuid4().hex}"
        self.doc_status = _EmptyDocStatus()
        self.chunking_func = chunking_by_token_size
        self.addon_params = {}
        self.process_calls = 0
        self.reset_calls = []

    async def apipeline_process_enqueue_documents(self):
        self.process_calls += 1

    async def apipeline_reset_failed_for_scan(
        self, request_id, *, scan_owner_token=None
    ):
        self.reset_calls.append((request_id, scan_owner_token))
        return True

    async def adelete_by_doc_id(self, doc_id, delete_llm_cache=False):
        return DeletionResult(status="success", message="ok", doc_id=doc_id)


@pytest.mark.parametrize(
    ("status_flags", "filename", "expected_detail"),
    [
        pytest.param(
            {"scanning": True, "scanning_exclusive": True},
            "while_scanning.docx",
            "classifying",
            id="scanning_classification",
        ),
        pytest.param(
            {"busy": True, "manual_freeze_requested": True},
            "while_freezing.docx",
            "retry of failed documents",
            id="manual_freeze",
        ),
    ],
)
async def test_upload_returns_409_when_ingress_blocked(
    tmp_path, monkeypatch, status_flags, filename, expected_detail
):
    """Upload must refuse with 409 during scan classification
    (``scanning_exclusive=True``) and manual-retry freeze
    (``manual_freeze_requested=True``)."""
    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(max_upload_size=None)
    )
    doc_manager = DocumentManager(str(tmp_path))
    rag = _ReservationRag("upload-409")

    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status.update(status_flags)

    router = create_document_routes(rag, doc_manager)
    upload_endpoint = _get_endpoint(router, "upload_to_input_dir")
    upload_file = _document_routes.UploadFile(
        filename=filename,
        file=BytesIO(b"docx bytes"),
    )

    with pytest.raises(_document_routes.HTTPException) as excinfo:
        await upload_endpoint(set(), upload_file)
    assert excinfo.value.status_code == 409
    assert expected_detail in excinfo.value.detail.lower()
    assert not (tmp_path / filename).exists()


async def test_upload_succeeds_during_scan_processing_phase(tmp_path, monkeypatch):
    """While the pipeline is in scan-driven processing (``scanning=True`` but
    ``scanning_exclusive=False``), new uploads must be accepted and coexist via
    the ingress mailbox."""
    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(max_upload_size=None)
    )
    doc_manager = DocumentManager(str(tmp_path))
    rag = _ReservationRag("upload-scan-proc")

    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    # Classification done; scan is now driving the processing pipeline.
    pipeline_status["scanning"] = True
    pipeline_status["scanning_exclusive"] = False
    pipeline_status["busy"] = True
    pipeline_status["pending_enqueues"] = 0

    router = create_document_routes(rag, doc_manager)
    upload_endpoint = _get_endpoint(router, "upload_to_input_dir")
    upload_file = _document_routes.UploadFile(
        filename="upload_during_scan_processing.docx",
        file=BytesIO(b"docx bytes"),
    )

    gate = asyncio.Event()

    async def _gated_index(rag_arg, file_path, track_id=None, admission_token=None):
        await gate.wait()

    monkeypatch.setattr(_document_routes, "pipeline_index_file", _gated_index)

    managed: set = set()
    response = await upload_endpoint(managed, upload_file)

    assert response.status == "success"
    assert (tmp_path / "upload_during_scan_processing.docx").exists()
    assert pipeline_status["pending_enqueues"] == 1
    assert len(managed) == 1

    gate.set()
    await _await_managed(managed)
    assert pipeline_status["pending_enqueues"] == 0


@pytest.mark.parametrize(
    ("initial_flags", "expected_scanning", "expected_pending"),
    [
        pytest.param({"busy": True}, False, 0, id="pipeline_busy"),
        pytest.param({"scanning": True}, True, 0, id="already_scanning"),
        pytest.param({"pending_enqueues": 1}, False, 1, id="enqueue_pending"),
    ],
)
async def test_scan_endpoint_returns_skipped_when_pipeline_active(
    tmp_path, initial_flags, expected_scanning, expected_pending
):
    """Scan endpoint must return ``scanning_skipped_pipeline_busy`` with zero
    side effects (no managed task, no flag mutation, no manual retry published)
    when ``busy``, ``scanning``, or ``pending_enqueues`` is active."""
    doc_manager = DocumentManager(str(tmp_path))
    rag = _ReservationRag("scan-skip")

    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status["busy"] = False
    pipeline_status["scanning"] = False
    pipeline_status["pending_enqueues"] = 0
    pipeline_status.update(initial_flags)

    router = create_document_routes(rag, doc_manager)
    scan_endpoint = _get_endpoint(router, "scan_for_new_documents")

    managed: set = set()
    response = await scan_endpoint(managed)

    assert response.status == "scanning_skipped_pipeline_busy"
    assert len(managed) == 0
    assert pipeline_status.get("scanning") is expected_scanning
    assert pipeline_status["pending_enqueues"] == expected_pending
    # Zero side effects: a refused scan publishes NO manual retry intent.
    ingress = await shared_storage.get_pipeline_ingress(rag.workspace)
    assert ingress.snapshot_manual_retries() == []


async def test_scan_endpoint_acquires_and_releases_scanning_flag(tmp_path):
    """The scan endpoint must atomically set ``scanning=True`` and
    ``run_scanning_process`` must clear it in finally."""
    doc_manager = DocumentManager(str(tmp_path))
    rag = _ReservationRag("scan-acquire")

    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status["busy"] = False
    pipeline_status["scanning"] = False

    router = create_document_routes(rag, doc_manager)
    scan_endpoint = _get_endpoint(router, "scan_for_new_documents")

    managed: set = set()
    response = await scan_endpoint(managed)

    assert response.status == "scanning_started"

    await _await_managed(managed)
    assert pipeline_status["scanning"] is False
    assert rag.process_calls == 1


async def test_reserve_enqueue_slot_blocks_concurrent_scan_until_release(tmp_path):
    """End-to-end on the reservation primitive: reserving a slot makes the scan
    endpoint refuse; releasing it lets the next scan in."""
    doc_manager = DocumentManager(str(tmp_path))
    rag = _ReservationRag("enq-blocks-scan")

    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status["busy"] = False
    pipeline_status["scanning"] = False
    pipeline_status["pending_enqueues"] = 0

    enq_token = "enq-blocks-scan"
    reserved = await _document_routes._reserve_enqueue_slot(rag, enq_token)
    assert reserved is True
    assert pipeline_status["pending_enqueues"] == 1

    router = create_document_routes(rag, doc_manager)
    scan_endpoint = _get_endpoint(router, "scan_for_new_documents")

    managed: set = set()
    blocked = await scan_endpoint(managed)
    assert blocked.status == "scanning_skipped_pipeline_busy"
    assert len(managed) == 0

    await _document_routes._release_enqueue_slot(rag, enq_token)
    assert pipeline_status["pending_enqueues"] == 0

    managed2: set = set()
    allowed = await scan_endpoint(managed2)
    assert allowed.status == "scanning_started"
    await _await_managed(managed2)
    assert pipeline_status["scanning"] is False


async def test_release_enqueue_slot_decrements_per_call():
    """Two-reservation cohort: each release removes its own token and decrements
    ``pending_enqueues`` independently."""
    rag = _ReservationRag("enq-cohort")
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status["busy"] = False
    pipeline_status["scanning"] = False
    pipeline_status["scanning_exclusive"] = False
    pipeline_status["pending_enqueues"] = 0

    assert await _document_routes._reserve_enqueue_slot(rag, "enq-a") is True
    assert await _document_routes._reserve_enqueue_slot(rag, "enq-b") is True
    assert pipeline_status["pending_enqueues"] == 2

    await _document_routes._release_enqueue_slot(rag, "enq-a")
    assert pipeline_status["pending_enqueues"] == 1

    await _document_routes._release_enqueue_slot(rag, "enq-b")
    assert pipeline_status["pending_enqueues"] == 0


@pytest.mark.parametrize(
    "case",
    [
        "enqueue_acquire_lock_exit",
        "scan_before_handoff",
        "clear_during_setup",
        "clear_acquire_lock_exit",
    ],
)
async def test_endpoint_releases_reservation_on_cancellation_before_handoff(
    tmp_path, monkeypatch, case
):
    """Regression (#3408): cancellations delivered during reservation acquire,
    pre-handoff start-barrier, or clear setup must release the slot in finally
    and leave zero leaked flags or manual retry entries."""
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    rag = _ReservationRag(f"cancel-{case}")
    doc_manager = DocumentManager(str(tmp_path))
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )

    if case == "enqueue_acquire_lock_exit":

        async def _reserve_then_cancel(rag_arg, token):
            lock = shared_storage.get_namespace_lock(
                "pipeline_status", workspace=rag_arg.workspace
            )
            async with lock:
                tokens = dict(pipeline_status.get("pending_enqueue_tokens", {}))
                tokens[token] = {"pid": 0, "process_start_id": None}
                pipeline_status.update(
                    {"pending_enqueue_tokens": tokens, "pending_enqueues": len(tokens)}
                )
            raise asyncio.CancelledError()

        monkeypatch.setattr(
            _document_routes, "_reserve_enqueue_slot", _reserve_then_cancel
        )
        router = create_document_routes(rag, doc_manager)
        endpoint = _get_endpoint(router, "insert_text")
        with pytest.raises(asyncio.CancelledError):
            await endpoint(
                _document_routes.InsertTextRequest(text="hello", file_source="a.md"),
                set(),
            )
        assert pipeline_status.get("pending_enqueues", 0) == 0
        assert pipeline_status.get("pending_enqueue_tokens", {}) == {}

    elif case == "scan_before_handoff":

        async def _cancel_before_takeover(*args, **kwargs):
            raise asyncio.CancelledError()

        monkeypatch.setattr(
            shared_storage, "start_committed_background_task", _cancel_before_takeover
        )
        router = create_document_routes(rag, doc_manager)
        endpoint = _get_endpoint(router, "scan_for_new_documents")
        with pytest.raises(asyncio.CancelledError):
            await endpoint(set())
        assert pipeline_status.get("scanning") is False
        assert pipeline_status.get("scanning_exclusive") is False
        assert pipeline_status.get("scanning_owner") is None
        ingress = await shared_storage.get_pipeline_ingress(rag.workspace)
        assert ingress.snapshot_manual_retries() == []

    elif case == "clear_during_setup":

        class _CancelNow:
            @staticmethod
            def now():
                raise asyncio.CancelledError()

        monkeypatch.setattr(_document_routes, "datetime", _CancelNow)
        router = create_document_routes(rag, doc_manager)
        endpoint = _get_endpoint(router, "clear_documents")
        with pytest.raises(asyncio.CancelledError):
            await endpoint()
        assert pipeline_status.get("busy") is False
        assert pipeline_status.get("destructive_busy") is False
        assert pipeline_status.get("busy_owner") is None

    else:
        assert case == "clear_acquire_lock_exit"

        async def _acquire_then_cancel(
            rag_arg, token, *, kind=None, operation_record=None
        ):
            lock = shared_storage.get_namespace_lock(
                "pipeline_status", workspace=rag_arg.workspace
            )
            ps = await shared_storage.get_namespace_data(
                "pipeline_status", workspace=rag_arg.workspace
            )
            async with lock:
                ps.update({"busy": True, "destructive_busy": True, "busy_owner": token})
            raise asyncio.CancelledError()

        monkeypatch.setattr(
            _document_routes, "_acquire_destructive_busy", _acquire_then_cancel
        )
        router = create_document_routes(rag, doc_manager)
        endpoint = _get_endpoint(router, "clear_documents")
        with pytest.raises(asyncio.CancelledError):
            await endpoint()
        assert pipeline_status.get("busy") is False
        assert pipeline_status.get("destructive_busy") is False
        assert pipeline_status.get("busy_owner") is None


@pytest.mark.parametrize("task_kind", ["scan_task", "delete_task"])
async def test_background_task_releases_reservation_on_initial_fetch_cancellation(
    tmp_path, monkeypatch, task_kind
):
    """Regression (#3408 review P2): background tasks that already hold a
    reservation when started must still release it if cancelled during their
    initial ``get_namespace_data`` fetch."""
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    rag = _ReservationRag(f"fetch-cancel-{task_kind}")
    token = uuid4().hex
    doc_manager = DocumentManager(str(tmp_path))

    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    if task_kind == "scan_task":
        pipeline_status.update(
            {"scanning": True, "scanning_exclusive": True, "scanning_owner": token}
        )
    else:
        pipeline_status.update(
            {"busy": True, "destructive_busy": True, "busy_owner": token}
        )

    real_get = shared_storage.get_namespace_data
    calls = {"n": 0}

    async def _get_ns_cancel_first(name, workspace=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise asyncio.CancelledError()
        return await real_get(name, workspace=workspace)

    monkeypatch.setattr(shared_storage, "get_namespace_data", _get_ns_cancel_first)

    with pytest.raises(asyncio.CancelledError):
        if task_kind == "scan_task":
            await run_scanning_process(rag, doc_manager, "track-scan", token)
        else:
            await _document_routes.background_delete_documents(
                rag, doc_manager, ["doc-x"], token=token
            )

    if task_kind == "scan_task":
        assert pipeline_status.get("scanning") is False
        assert pipeline_status.get("scanning_exclusive") is False
        assert pipeline_status.get("scanning_owner") is None
    else:
        assert pipeline_status.get("busy") is False
        assert pipeline_status.get("destructive_busy") is False
        assert pipeline_status.get("busy_owner") is None


@pytest.mark.parametrize(
    ("helper_name", "seed_fn", "expected_cleared"),
    [
        pytest.param(
            "_release_destructive_busy",
            lambda token: {
                "busy": True,
                "destructive_busy": True,
                "busy_owner": token,
            },
            {"busy": False, "destructive_busy": False, "busy_owner": None},
            id="destructive",
        ),
        pytest.param(
            "_release_scanning_reservation",
            lambda token: {
                "scanning": True,
                "scanning_exclusive": True,
                "scanning_owner": token,
            },
            {"scanning": False, "scanning_exclusive": False, "scanning_owner": None},
            id="scanning",
        ),
        pytest.param(
            "_release_enqueue_slot",
            lambda token: {
                "pending_enqueue_tokens": {token: {"pid": 0, "process_start_id": None}},
                "pending_enqueues": 1,
            },
            {"pending_enqueues": 0, "pending_enqueue_tokens": {}},
            id="enqueue",
        ),
    ],
)
async def test_release_helper_survives_fetch_cancellation(
    monkeypatch, helper_name, seed_fn, expected_cleared
):
    """Regression (#3408 review P2): reservation release helpers run their
    ``get_namespace_data`` fetch + owner-checked release inside
    ``run_to_completion``, so a cancellation delivered mid-fetch is deferred
    until the slot is freed and then re-raised."""
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    rag = _ReservationRag(f"rel-{helper_name}")
    token = uuid4().hex

    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status.update(seed_fn(token))

    real_get = shared_storage.get_namespace_data
    entered = asyncio.Event()
    release_fetch = asyncio.Event()

    async def _gated_get(name, workspace=None):
        # Gate the fetch deterministically so we deliver cancellation strictly
        # while the fetch is in flight; run_to_completion shields the inner task.
        entered.set()
        await release_fetch.wait()
        return await real_get(name, workspace=workspace)

    monkeypatch.setattr(shared_storage, "get_namespace_data", _gated_get)

    release_fn = getattr(_document_routes, helper_name)
    task = asyncio.create_task(release_fn(rag, token))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release_fetch.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)

    for key, expected in expected_cleared.items():
        assert pipeline_status.get(key) == expected


async def test_two_concurrent_uploads_both_succeed_when_pipeline_busy(
    tmp_path, monkeypatch
):
    """Two upload requests arriving while the pipeline is busy must both reserve
    slots, track managed tasks, and release pending_enqueues back to 0."""
    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(max_upload_size=None)
    )
    doc_manager = DocumentManager(str(tmp_path))
    rag = _ReservationRag("two-uploads")

    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status["scanning"] = False
    pipeline_status["pending_enqueues"] = 0
    pipeline_status["busy"] = True

    router = create_document_routes(rag, doc_manager)
    upload_endpoint = _get_endpoint(router, "upload_to_input_dir")

    gate = asyncio.Event()

    async def _gated_index(rag_arg, file_path, track_id=None, admission_token=None):
        await gate.wait()

    monkeypatch.setattr(_document_routes, "pipeline_index_file", _gated_index)

    managed: set = set()

    upload_a = _document_routes.UploadFile(filename="a.docx", file=BytesIO(b"a bytes"))
    response_a = await upload_endpoint(managed, upload_a)
    assert response_a.status == "success"
    assert pipeline_status["pending_enqueues"] == 1

    upload_b = _document_routes.UploadFile(filename="b.docx", file=BytesIO(b"b bytes"))
    response_b = await upload_endpoint(managed, upload_b)
    assert response_b.status == "success"
    assert pipeline_status["pending_enqueues"] == 2
    assert (tmp_path / "a.docx").exists()
    assert (tmp_path / "b.docx").exists()
    assert len(managed) == 2

    gate.set()
    await _await_managed(managed)
    assert pipeline_status["pending_enqueues"] == 0


async def test_reserve_enqueue_slot_allows_busy_and_scan_processing_phase():
    """Reservation allows ``busy=True`` and ``scanning=True`` (processing phase)
    while still rejecting ``scanning_exclusive=True`` (classification phase)."""
    rag = _ReservationRag("enq-allow")
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status["busy"] = False
    pipeline_status["scanning"] = False
    pipeline_status["scanning_exclusive"] = False
    pipeline_status["pending_enqueues"] = 0

    pipeline_status["busy"] = True
    assert await _document_routes._reserve_enqueue_slot(rag, "enq") is True
    await _document_routes._release_enqueue_slot(rag, "enq")
    pipeline_status["busy"] = False

    pipeline_status["scanning"] = True
    assert await _document_routes._reserve_enqueue_slot(rag, "enq") is True
    await _document_routes._release_enqueue_slot(rag, "enq")
    pipeline_status["scanning"] = False

    pipeline_status["scanning"] = True
    pipeline_status["scanning_exclusive"] = True
    with pytest.raises(_document_routes.HTTPException) as exc:
        await _document_routes._reserve_enqueue_slot(rag, "enq")
    assert exc.value.status_code == 409
    assert "classifying" in exc.value.detail.lower()
    assert pipeline_status["pending_enqueues"] == 0


async def test_reserve_enqueue_slot_rejects_destructive_busy():
    """``destructive_busy`` (set by /documents/clear and per-doc delete) must
    reject enqueue reservations with 409 until released."""
    rag = _ReservationRag("enq-destr")
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status["busy"] = False
    pipeline_status["scanning"] = False
    pipeline_status["scanning_exclusive"] = False
    pipeline_status["pending_enqueues"] = 0

    pipeline_status["busy"] = True
    pipeline_status["destructive_busy"] = True
    with pytest.raises(_document_routes.HTTPException) as exc:
        await _document_routes._reserve_enqueue_slot(rag, "enq")
    assert exc.value.status_code == 409
    assert "clearing or deleting" in exc.value.detail.lower()
    assert pipeline_status["pending_enqueues"] == 0

    pipeline_status["destructive_busy"] = False
    pipeline_status["busy"] = False
    assert await _document_routes._reserve_enqueue_slot(rag, "enq") is True
    await _document_routes._release_enqueue_slot(rag, "enq")


async def test_clear_documents_sets_and_clears_destructive_busy(tmp_path):
    """``/documents/clear`` must set ``destructive_busy=True`` while dropping
    storages and clear it back to False on completion."""
    workspace = f"clear-test-{uuid4().hex}"
    observed = {"destructive_busy": None}

    class _DropSpy:
        def __init__(self, ws):
            self.namespace = "spy"
            self.workspace = ws

        async def drop(self):
            shared_storage_inner = importlib.import_module("lightrag.kg.shared_storage")
            ns = await shared_storage_inner.get_namespace_data(
                "pipeline_status", workspace=self.workspace
            )
            observed["destructive_busy"] = ns.get("destructive_busy")
            return None

    spy = _DropSpy(workspace)

    class _ClearRag:
        def __init__(self):
            self.workspace = workspace
            self.text_chunks = spy
            self.full_docs = spy
            self.full_entities = spy
            self.full_relations = spy
            self.entity_chunks = spy
            self.relation_chunks = spy
            self.entities_vdb = spy
            self.relationships_vdb = spy
            self.chunks_vdb = spy
            self.chunk_entity_relation_graph = spy
            self.doc_status = spy

        async def aclear_cache(self, modes=None):
            return None

    doc_manager = DocumentManager(str(tmp_path))
    rag = _ClearRag()

    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )

    router = create_document_routes(rag, doc_manager)
    clear_endpoint = _get_endpoint(router, "clear_documents")

    response = await clear_endpoint()
    assert response.status == "success"
    assert observed["destructive_busy"] is True
    assert pipeline_status.get("destructive_busy") is False
    assert pipeline_status.get("busy") is False


async def test_clear_documents_refuses_when_scanning_or_pending_enqueues(tmp_path):
    """``/documents/clear`` must refuse atomically when scanning, pending
    enqueues, or busy is active, without mutating existing flags."""
    rag = _ReservationRag("clear-refuse")
    doc_manager = DocumentManager(str(tmp_path))

    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )

    router = create_document_routes(rag, doc_manager)
    clear_endpoint = _get_endpoint(router, "clear_documents")

    pipeline_status["busy"] = False
    pipeline_status["scanning"] = True
    pipeline_status["pending_enqueues"] = 0
    response = await clear_endpoint()
    assert response.status == "busy"
    assert pipeline_status["scanning"] is True
    assert pipeline_status.get("destructive_busy", False) is False

    pipeline_status["scanning"] = False
    pipeline_status["pending_enqueues"] = 1
    response = await clear_endpoint()
    assert response.status == "busy"
    assert pipeline_status["pending_enqueues"] == 1
    assert pipeline_status.get("destructive_busy", False) is False

    pipeline_status["pending_enqueues"] = 0
    pipeline_status["busy"] = True
    response = await clear_endpoint()
    assert response.status == "busy"
    assert pipeline_status.get("destructive_busy", False) is False


async def test_delete_document_reserves_destructive_busy_synchronously(tmp_path):
    """``/documents/delete_document`` must reserve the destructive slot
    synchronously BEFORE returning ``deletion_started``, and refuse when
    scanning or pending_enqueues is active."""
    rag = _ReservationRag("delete-sync")
    doc_manager = DocumentManager(str(tmp_path))

    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )

    router = create_document_routes(rag, doc_manager)
    delete_endpoint = _get_endpoint(router, "delete_document")
    DeleteDocRequest = _document_routes.DeleteDocRequest

    pipeline_status["busy"] = False
    pipeline_status["scanning"] = False
    pipeline_status["pending_enqueues"] = 0
    gate = asyncio.Event()
    _orig_delete = rag.adelete_by_doc_id

    async def _gated_delete(doc_id, delete_llm_cache=False):
        await gate.wait()
        return await _orig_delete(doc_id, delete_llm_cache=delete_llm_cache)

    rag.adelete_by_doc_id = _gated_delete

    managed: set = set()
    response = await delete_endpoint(DeleteDocRequest(doc_ids=["doc-1"]), managed)
    assert response.status == "deletion_started"
    assert pipeline_status["busy"] is True
    assert pipeline_status["destructive_busy"] is True
    assert len(managed) == 1

    gate.set()
    await _await_managed(managed)
    assert pipeline_status["busy"] is False
    assert pipeline_status["destructive_busy"] is False
    rag.adelete_by_doc_id = _orig_delete

    pipeline_status["scanning"] = True
    managed = set()
    response = await delete_endpoint(DeleteDocRequest(doc_ids=["doc-1"]), managed)
    assert response.status == "busy"
    assert len(managed) == 0
    assert pipeline_status.get("destructive_busy", False) is False
    pipeline_status["scanning"] = False

    pipeline_status["pending_enqueues"] = 1
    managed = set()
    response = await delete_endpoint(DeleteDocRequest(doc_ids=["doc-1"]), managed)
    assert response.status == "busy"
    assert len(managed) == 0
    assert pipeline_status["pending_enqueues"] == 1
    assert pipeline_status.get("destructive_busy", False) is False
    pipeline_status["pending_enqueues"] = 0


@pytest.mark.parametrize("kind", ["scan", "upload", "delete"])
async def test_managed_task_released_on_shutdown_drain(tmp_path, monkeypatch, kind):
    """PR-1b: scan, upload, and delete reservations are held by managed asyncio
    tasks; a shutdown drain cancels the parked task and its cancellation-resistant
    finally releases the reservation."""
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    doc_manager = DocumentManager(str(tmp_path))
    rag = _ReservationRag(f"drain-{kind}")

    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)
    pipeline_status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status["busy"] = False
    pipeline_status["scanning"] = False
    pipeline_status["pending_enqueues"] = 0

    gate = asyncio.Event()
    managed: set = set()

    if kind == "scan":

        async def _gated_process():
            await gate.wait()

        rag.apipeline_process_enqueue_documents = _gated_process
        router = create_document_routes(rag, doc_manager)
        response = await _get_endpoint(router, "scan_for_new_documents")(managed)
        assert response.status == "scanning_started"
        assert len(managed) == 1
        assert pipeline_status["scanning"] is True
        expected_after = {
            "scanning": False,
            "scanning_exclusive": False,
            "scanning_owner": None,
        }

    elif kind == "upload":
        monkeypatch.setattr(
            _document_routes, "global_args", SimpleNamespace(max_upload_size=None)
        )

        async def _gated_index(rag_arg, file_path, track_id=None, admission_token=None):
            await gate.wait()

        monkeypatch.setattr(_document_routes, "pipeline_index_file", _gated_index)
        router = create_document_routes(rag, doc_manager)
        upload_file = _document_routes.UploadFile(
            filename="drain.docx", file=BytesIO(b"bytes")
        )
        response = await _get_endpoint(router, "upload_to_input_dir")(
            managed, upload_file
        )
        assert response.status == "success"
        assert len(managed) == 1
        assert pipeline_status["pending_enqueues"] == 1
        expected_after = {"pending_enqueues": 0, "pending_enqueue_tokens": {}}

    else:
        assert kind == "delete"

        async def _gated_delete(doc_id, delete_llm_cache=False):
            await gate.wait()

        rag.adelete_by_doc_id = _gated_delete
        router = create_document_routes(rag, doc_manager)
        response = await _get_endpoint(router, "delete_document")(
            _document_routes.DeleteDocRequest(doc_ids=["doc-1"]),
            managed,
        )
        assert response.status == "deletion_started"
        assert len(managed) == 1
        assert pipeline_status["busy"] is True
        assert pipeline_status["destructive_busy"] is True
        expected_after = {
            "busy": False,
            "destructive_busy": False,
            "busy_owner": None,
        }

    pending = await shared_storage.drain_reserved_background_tasks(managed)
    assert pending is None
    for key, expected in expected_after.items():
        assert pipeline_status.get(key) == expected


async def test_reprocess_starts_managed_task(tmp_path):
    """PR-1b: /reprocess_failed runs apipeline_process_enqueue_documents as a
    MANAGED task (tracked for shutdown drain) rather than a Starlette callback."""
    doc_manager = DocumentManager(str(tmp_path))
    rag = _ReservationRag("reprocess")

    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    await shared_storage.initialize_pipeline_status(workspace=rag.workspace)

    gate = asyncio.Event()

    async def _gated_process():
        await gate.wait()
        rag.process_calls += 1

    rag.apipeline_process_enqueue_documents = _gated_process

    router = create_document_routes(rag, doc_manager)
    reprocess_endpoint = _get_endpoint(router, "reprocess_failed_documents")

    managed: set = set()
    response = await reprocess_endpoint(managed)
    assert response.status == "reprocessing_started"
    assert len(managed) == 1

    gate.set()
    await _await_managed(managed)
    assert rag.process_calls == 1
