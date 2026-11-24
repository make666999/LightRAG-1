"""Document route admission control, slot release before processing, force-reset reservation cleanup, and stall fence exemptions."""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
from fastapi import HTTPException

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
_shared_storage = importlib.import_module("lightrag.kg.shared_storage")
sys.argv = _original_argv

from lightrag import LightRAG  # noqa: E402
from lightrag.base import DocStatus  # noqa: E402
from lightrag.exceptions import (  # noqa: E402
    PipelineReservationConflictError,
    StorageControlPlaneError,
)
from lightrag.kg.pipeline_ingress import PipelineIngressMessage  # noqa: E402
from lightrag.kg.shared_storage import (  # noqa: E402
    MANUAL_DRAIN_ENQUEUE_STALL_FENCE,
    MANUAL_PHASE_DRAIN_TO_IDLE,
    SOURCE_REPAIR_RESERVATION_KIND,
    acquire_enqueue_reservation,
    fence_workspace_for_recovery,
    finalize_share_data,
    get_namespace_data,
    get_namespace_lock,
    get_pipeline_ingress,
    initialize_pipeline_status,
    initialize_share_data,
)
from lightrag.utils import EmbeddingFunc, Tokenizer, compute_mdhash_id  # noqa: E402

DocumentManager = _document_routes.DocumentManager
ForceResetRecoveryRequest = _document_routes.ForceResetRecoveryRequest
_acquire_destructive_busy = _document_routes._acquire_destructive_busy
_release_enqueue_slot = _document_routes._release_enqueue_slot
_reserve_enqueue_slot = _document_routes._reserve_enqueue_slot
pipeline_index_file = _document_routes.pipeline_index_file
pipeline_index_texts = _document_routes.pipeline_index_texts

pytestmark = pytest.mark.offline


# --------------------------------------------------------------------------- #
# Shared fixtures and LightRAG builders
# --------------------------------------------------------------------------- #


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


async def _dummy_llm(*args, **kwargs) -> str:
    return "ok"


async def _noop_process(*args, **kwargs) -> None:
    return None


@pytest.fixture(autouse=True)
def _shared():
    initialize_share_data()
    yield
    finalize_share_data()


async def _build_rag(
    tmp_path,
    *,
    prefix: str = "adm",
    capacity: int = 0,
    init_pipeline: bool = False,
    chunking_func=None,
    extract=None,
    noop_process: bool = False,
) -> LightRAG:
    kwargs = {}
    if chunking_func is not None:
        kwargs["chunking_func"] = chunking_func
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"{prefix}-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        max_parallel_insert=1,
        max_pending_documents=capacity,
        **kwargs,
    )
    await rag.initialize_storages()
    if init_pipeline:
        await initialize_pipeline_status(workspace=rag.workspace)
    if extract is not None:
        rag._process_extract_entities = extract
    if noop_process:
        rag.apipeline_process_enqueue_documents = _noop_process
    return rag


async def _status_handles(rag):
    return (
        await get_namespace_data("pipeline_status", workspace=rag.workspace),
        get_namespace_lock("pipeline_status", workspace=rag.workspace),
    )


async def _fence(rag, kind: str, *, operation_record=None) -> None:
    pipeline_status, lock = await _status_handles(rag)
    kwargs = {"kind": kind, "message": f"{kind} for test"}
    if operation_record is not None:
        kwargs["operation_record"] = operation_record
    await fence_workspace_for_recovery(pipeline_status, lock, **kwargs)


# --------------------------------------------------------------------------- #
# Endpoint-side admission control (§9.1/§9.2)
# --------------------------------------------------------------------------- #


class _CountingDocStatus:
    def __init__(self, active: int):
        self.active = active
        self.error: Exception | None = None
        self.calls = 0

    async def count_docs_by_statuses(self, statuses, *, strict=True):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.active


async def _counting_rag(*, capacity: int, active: int = 0):
    workspace = f"admission-{uuid4().hex[:8]}"
    await initialize_pipeline_status(workspace=workspace)
    return SimpleNamespace(
        workspace=workspace,
        doc_status=_CountingDocStatus(active),
        max_pending_documents=capacity,
    )


async def _status(rag):
    return await get_namespace_data("pipeline_status", workspace=rag.workspace)


async def test_reservation_refuses_over_capacity_with_429():
    rag = await _counting_rag(capacity=3, active=3)

    with pytest.raises(HTTPException) as excinfo:
        await _reserve_enqueue_slot(rag, "tok-1")

    error = excinfo.value
    assert error.status_code == 429
    assert "3 document(s) already active" in error.detail
    assert "capacity 3" in error.detail
    assert error.headers["Retry-After"]
    status = await _status(rag)
    assert status["pending_enqueue_tokens"] == {}


async def test_reservation_admits_up_to_capacity_and_records_weight():
    rag = await _counting_rag(capacity=2, active=1)

    assert await _reserve_enqueue_slot(rag, "tok-1") is True

    status = await _status(rag)
    assert status["pending_enqueue_tokens"]["tok-1"]["weight"] == 1
    assert status["pending_enqueues"] == 1


async def test_count_failure_is_503_not_capacity():
    rag = await _counting_rag(capacity=5)
    rag.doc_status.error = StorageControlPlaneError("doc_status index unavailable")

    with pytest.raises(HTTPException) as excinfo:
        await _reserve_enqueue_slot(rag, "tok-1")

    assert excinfo.value.status_code == 503
    assert "index unavailable" not in excinfo.value.detail


async def test_fence_wins_over_available_capacity():
    rag = await _counting_rag(capacity=100)
    status = await _status(rag)
    status["manual_freeze_requested"] = True

    with pytest.raises(HTTPException) as excinfo:
        await _reserve_enqueue_slot(rag, "tok-1")

    assert excinfo.value.status_code == 409
    assert rag.doc_status.calls == 1


async def test_reweight_survives_a_freeze_that_started_after_admission():
    rag = await _counting_rag(capacity=10)
    await _reserve_enqueue_slot(rag, "tok-1")
    status = await _status(rag)
    status["manual_freeze_requested"] = True

    await _document_routes._reweight_enqueue_slot(rag, "tok-1", 4)

    status = await _status(rag)
    assert status["pending_enqueue_tokens"]["tok-1"]["weight"] == 4
    assert status["pending_enqueues"] == 1


async def test_reweight_refuses_a_batch_that_does_not_fit():
    rag = await _counting_rag(capacity=3, active=1)
    await _reserve_enqueue_slot(rag, "tok-1")

    with pytest.raises(HTTPException) as excinfo:
        await _document_routes._reweight_enqueue_slot(rag, "tok-1", 5)

    assert excinfo.value.status_code == 429
    status = await _status(rag)
    assert status["pending_enqueue_tokens"]["tok-1"]["weight"] == 1


async def test_reweight_is_a_noop_when_admission_is_disabled():
    rag = await _counting_rag(capacity=0)
    await _reserve_enqueue_slot(rag, "tok-1")

    await _document_routes._reweight_enqueue_slot(rag, "tok-1", 1000)

    assert rag.doc_status.calls == 0
    status = await _status(rag)
    assert status["pending_enqueues"] == 1


async def test_disabled_admission_never_counts():
    rag = await _counting_rag(capacity=0, active=10_000)

    assert await _reserve_enqueue_slot(rag, "tok-1") is True

    assert rag.doc_status.calls == 0


# --------------------------------------------------------------------------- #
# Admission slot released before background task drives processing loop
# --------------------------------------------------------------------------- #


def _spy_on_processing(rag, pipeline_status, lock, token: str) -> dict:
    seen: dict = {}

    async def _process():
        async with lock:
            seen["pending_enqueues"] = pipeline_status.get("pending_enqueues", 0)
            seen["token_registered"] = token in (
                pipeline_status.get("pending_enqueue_tokens") or {}
            )

    rag.apipeline_process_enqueue_documents = _process
    return seen


async def _freeze_as_manual_retry(pipeline_status, lock) -> None:
    async with lock:
        pipeline_status["manual_freeze_requested"] = True
        pipeline_status["manual_phase"] = MANUAL_PHASE_DRAIN_TO_IDLE


@pytest.mark.parametrize("capacity", [0, 10], ids=["admission-off", "admission-on"])
def test_texts_releases_its_slot_before_driving_the_loop(tmp_path, capacity):
    async def _run():
        rag = await _build_rag(tmp_path, capacity=capacity)
        try:
            pipeline_status, lock = await _status_handles(rag)
            token = uuid4().hex
            assert await _reserve_enqueue_slot(rag, token)
            seen = _spy_on_processing(rag, pipeline_status, lock, token)
            await _freeze_as_manual_retry(pipeline_status, lock)

            try:
                await pipeline_index_texts(
                    rag,
                    ["a document about admission tokens"],
                    file_sources=["admission.txt"],
                    track_id="t-texts",
                    admission_token=token,
                )
            finally:
                await _release_enqueue_slot(rag, token)

            assert seen["pending_enqueues"] == 0
            assert seen["token_registered"] is False
            pending = await rag.doc_status.get_docs_by_statuses([DocStatus.PENDING])
            assert len(pending) == 1
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_texts_release_survives_the_endpoint_double_release(tmp_path):
    async def _run():
        rag = await _build_rag(tmp_path, capacity=0)
        try:
            pipeline_status, lock = await _status_handles(rag)
            token = uuid4().hex
            sibling = uuid4().hex
            assert await _reserve_enqueue_slot(rag, token)
            assert await _reserve_enqueue_slot(rag, sibling)
            _spy_on_processing(rag, pipeline_status, lock, token)

            await pipeline_index_texts(
                rag,
                ["another document"],
                file_sources=["sibling.txt"],
                track_id="t-sibling",
                admission_token=token,
            )
            await _release_enqueue_slot(rag, token)

            async with lock:
                assert pipeline_status["pending_enqueues"] == 1
                assert sibling in pipeline_status["pending_enqueue_tokens"]
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.parametrize("capacity", [0, 10], ids=["admission-off", "admission-on"])
def test_upload_releases_its_slot_before_driving_the_loop(tmp_path, capacity):
    async def _run():
        rag = await _build_rag(tmp_path, capacity=capacity)
        try:
            pipeline_status, lock = await _status_handles(rag)
            source: Path = tmp_path / "upload.txt"
            source.write_text("a document arriving through /documents/upload")

            token = uuid4().hex
            assert await _reserve_enqueue_slot(rag, token)
            seen = _spy_on_processing(rag, pipeline_status, lock, token)
            await _freeze_as_manual_retry(pipeline_status, lock)

            try:
                await pipeline_index_file(
                    rag, source, "t-upload", admission_token=token
                )
            finally:
                await _release_enqueue_slot(rag, token)

            assert seen["pending_enqueues"] == 0
            assert seen["token_registered"] is False
            pending = await rag.doc_status.get_docs_by_statuses([DocStatus.PENDING])
            assert len(pending) == 1
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_upload_releases_its_slot_even_when_the_enqueue_rejects_the_file(tmp_path):
    async def _run():
        rag = await _build_rag(tmp_path, capacity=0)
        try:
            pipeline_status, lock = await _status_handles(rag)
            refused = tmp_path / "bad\nname.txt"

            token = uuid4().hex
            assert await _reserve_enqueue_slot(rag, token)
            seen = _spy_on_processing(rag, pipeline_status, lock, token)

            await pipeline_index_file(rag, refused, "t-rejected", admission_token=token)

            assert seen == {}
            async with lock:
                assert pipeline_status["pending_enqueues"] == 0
                assert token not in pipeline_status["pending_enqueue_tokens"]
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def _chunking(tokenizer, content, *a, **k) -> list[dict]:
    return [{"tokens": 1, "content": f"{content}::chunk1", "chunk_order_index": 0}]


class _Extract:
    """Entity extraction stub with a hook that fires inside the running run."""

    def __init__(self) -> None:
        self.fail = False
        self.on_call = None

    async def __call__(self, chunks, *args, **kwargs):
        if self.on_call is not None:
            hook, self.on_call = self.on_call, None
            await hook()
        if self.fail:
            raise RuntimeError("extract fail sentinel")
        return [({}, {}) for _ in chunks]


async def _status_of(rag: LightRAG, doc_id: str) -> str:
    row = await rag.doc_status.get_by_id(doc_id)
    raw = (row or {}).get("status")
    return raw.value if isinstance(raw, DocStatus) else str(raw or "<missing>")


def test_a_manual_retry_queued_mid_texts_run_completes(tmp_path):
    async def _run():
        extract = _Extract()
        rag = await _build_rag(
            tmp_path, prefix="e2e", chunking_func=_chunking, extract=extract
        )
        try:
            extract.fail = True
            await rag.apipeline_enqueue_documents(
                input="body of old.txt", file_paths="old.txt"
            )
            await rag.apipeline_process_enqueue_documents()
            failed_id = compute_mdhash_id("old.txt", prefix="doc-")
            assert await _status_of(rag, failed_id) == DocStatus.FAILED.value

            extract.fail = False
            ingress = await get_pipeline_ingress(rag.workspace)
            request_id = uuid4().hex

            async def _queue_manual_retry():
                ingress.request_manual_retry(
                    request_id,
                    PipelineIngressMessage(
                        kind="rescan", retry_failed=True, request_id=request_id
                    ),
                )

            extract.on_call = _queue_manual_retry

            token = uuid4().hex
            assert await _reserve_enqueue_slot(rag, token)
            try:
                await asyncio.wait_for(
                    pipeline_index_texts(
                        rag,
                        ["fresh body"],
                        file_sources=["new.txt"],
                        track_id="t-e2e",
                        admission_token=token,
                    ),
                    timeout=30,
                )
            finally:
                await _release_enqueue_slot(rag, token)

            new_id = compute_mdhash_id("new.txt", prefix="doc-")
            assert await _status_of(rag, new_id) == DocStatus.PROCESSED.value
            assert await _status_of(rag, failed_id) == DocStatus.PROCESSED.value
            assert ingress.snapshot_manual_retries() == []
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


# --------------------------------------------------------------------------- #
# force_reset drops stalled enqueue reservations (while sparing repair guards)
# --------------------------------------------------------------------------- #


def _force_reset_endpoint(rag, tmp_path):
    doc_manager = DocumentManager(str(tmp_path / "inputs"))
    router = _document_routes.create_document_routes(rag, doc_manager)
    return [
        route.endpoint
        for route in router.routes
        if getattr(route, "name", "") == "force_reset_recovery"
    ][-1]


async def _reserve_repair_guard(rag, token: str) -> None:
    pipeline_status, lock = await _status_handles(rag)
    result = await acquire_enqueue_reservation(
        pipeline_status,
        lock,
        token=token,
        reject_when=(),
        weight=0,
        capacity=0,
        kind=SOURCE_REPAIR_RESERVATION_KIND,
    )
    assert result.acquired


def test_force_reset_drops_the_reservations_the_enqueue_fence_waited_on(tmp_path):
    async def _run():
        rag = await _build_rag(tmp_path, prefix="fr", init_pipeline=True)
        try:
            pipeline_status, _ = await _status_handles(rag)
            stuck = uuid4().hex
            assert await _reserve_enqueue_slot(rag, stuck)
            await _fence(
                rag, "manual_drain_enqueue_stalled", operation_record={"scope": "t"}
            )

            endpoint = _force_reset_endpoint(rag, tmp_path)
            response = await endpoint(ForceResetRecoveryRequest(confirm=True))

            assert response.status == "reset"
            acquired, reason = await _acquire_destructive_busy(
                rag,
                uuid4().hex,
                kind="clear",
                operation_record={"kind": "clear"},
            )
            assert acquired, reason

            assert pipeline_status.get("pending_enqueues") == 0
            assert dict(pipeline_status.get("pending_enqueue_tokens") or {}) == {}
            assert response.dropped_enqueue_reservations == 1
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.parametrize(
    "kind",
    [
        "manual_drain_stalled",
        "manual_drain_blocked",
        "clear",
    ],
)
def test_force_reset_keeps_reservations_for_every_other_fence_kind(tmp_path, kind):
    async def _run():
        rag = await _build_rag(tmp_path, prefix="fr", init_pipeline=True)
        try:
            pipeline_status, _ = await _status_handles(rag)
            live = uuid4().hex
            assert await _reserve_enqueue_slot(rag, live)
            await _fence(rag, kind, operation_record={"scope": "t"})

            endpoint = _force_reset_endpoint(rag, tmp_path)
            response = await endpoint(ForceResetRecoveryRequest(confirm=True))

            assert response.status == "reset"
            assert response.dropped_enqueue_reservations == 0
            assert pipeline_status.get("pending_enqueues") == 1
            assert live in dict(pipeline_status.get("pending_enqueue_tokens") or {})
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_force_reset_keeps_a_source_repair_guard_it_cannot_safely_drop(tmp_path):
    async def _run():
        rag = await _build_rag(tmp_path, prefix="fr", init_pipeline=True)
        try:
            pipeline_status, _ = await _status_handles(rag)
            stuck = uuid4().hex
            guard = f"source-repair-{uuid4().hex}"
            assert await _reserve_enqueue_slot(rag, stuck)
            await _reserve_repair_guard(rag, guard)
            await _fence(
                rag, "manual_drain_enqueue_stalled", operation_record={"scope": "t"}
            )

            endpoint = _force_reset_endpoint(rag, tmp_path)
            response = await endpoint(ForceResetRecoveryRequest(confirm=True))

            assert response.status == "reset"
            tokens = dict(pipeline_status.get("pending_enqueue_tokens") or {})
            assert list(tokens) == [guard]
            assert pipeline_status.get("pending_enqueues") == 1
            assert response.dropped_enqueue_reservations == 1
            assert response.retained_enqueue_reservations == 1

            acquired, reason = await _acquire_destructive_busy(
                rag,
                uuid4().hex,
                kind="clear",
                operation_record={"kind": "clear"},
            )
            assert acquired is False
            assert "in flight" in (reason or "")
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


# --------------------------------------------------------------------------- #
# Enqueue-stall fence spares already-admitted text payloads
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("capacity", [0, 10], ids=["admission-off", "admission-on"])
def test_an_admitted_text_still_lands_under_the_stall_fence(tmp_path, capacity):
    async def _run():
        rag = await _build_rag(
            tmp_path, prefix="stall", capacity=capacity, noop_process=True
        )
        try:
            token = uuid4().hex
            assert await _reserve_enqueue_slot(rag, token)
            await _fence(rag, MANUAL_DRAIN_ENQUEUE_STALL_FENCE)

            try:
                await pipeline_index_texts(
                    rag,
                    ["a payload that lives only in this task"],
                    file_sources=["slow.txt"],
                    track_id="t-stalled",
                    admission_token=token,
                )
            finally:
                await _release_enqueue_slot(rag, token)

            pending = await rag.doc_status.get_docs_by_statuses([DocStatus.PENDING])
            assert len(pending) == 1

            pipeline_status, lock = await _status_handles(rag)
            async with lock:
                fence = pipeline_status.get("recovery_required")
            assert fence["kind"] == MANUAL_DRAIN_ENQUEUE_STALL_FENCE
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.parametrize(
    "kind",
    [
        "clear",
        "manual_drain_stalled",
        "manual_drain_blocked",
    ],
)
def test_every_other_fence_kind_still_refuses_the_same_holder(tmp_path, kind):
    async def _run():
        rag = await _build_rag(tmp_path, prefix="stall", noop_process=True)
        try:
            token = uuid4().hex
            assert await _reserve_enqueue_slot(rag, token)
            await _fence(rag, kind)

            with pytest.raises(PipelineReservationConflictError):
                await pipeline_index_texts(
                    rag,
                    ["refused"],
                    file_sources=["refused.txt"],
                    track_id="t-refused",
                    admission_token=token,
                )

            pending = await rag.doc_status.get_docs_by_statuses([DocStatus.PENDING])
            assert len(pending) == 0
        finally:
            await _release_enqueue_slot(rag, token)
            await rag.finalize_storages()

    asyncio.run(_run())


def test_a_holder_the_fence_never_knew_is_not_exempt(tmp_path):
    async def _run():
        rag = await _build_rag(tmp_path, prefix="stall", noop_process=True)
        try:
            await _fence(rag, MANUAL_DRAIN_ENQUEUE_STALL_FENCE)

            with pytest.raises(PipelineReservationConflictError):
                await pipeline_index_texts(
                    rag,
                    ["not admitted"],
                    file_sources=["stranger.txt"],
                    track_id="t-stranger",
                    admission_token=uuid4().hex,
                )

            pending = await rag.doc_status.get_docs_by_statuses([DocStatus.PENDING])
            assert len(pending) == 0
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
