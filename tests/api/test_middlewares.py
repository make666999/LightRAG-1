"""Pre-body admission middleware and raw request-body ceiling middleware tests."""

from __future__ import annotations

import asyncio
import importlib
import sys
import tempfile
from io import BytesIO
from types import SimpleNamespace
from uuid import uuid4

import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
_shared_storage = importlib.import_module("lightrag.kg.shared_storage")
_admission = importlib.import_module("lightrag.api.admission")
_admission_mod = importlib.import_module("lightrag.api.admission_middleware")
_body_limit_mod = importlib.import_module("lightrag.api.body_limit_middleware")
_utils_api = importlib.import_module("lightrag.api.utils_api")
sys.argv = _original_argv

AdmissionMiddleware = _admission_mod.AdmissionMiddleware
BodyLimitMiddleware = _body_limit_mod.BodyLimitMiddleware
resolve_body_limits = _body_limit_mod.resolve_body_limits

pytestmark = pytest.mark.offline

DEFAULT_LIMIT = 1024 * 1024
INGEST_LIMIT = 50 * 1024 * 1024
UPLOAD_LIMIT = 101 * 1024 * 1024


class _CountingDocStatus:
    def __init__(self, active: int):
        self.active = active

    async def count_docs_by_statuses(self, statuses, *, strict=True):
        return self.active


async def _rag(*, capacity: int, active: int = 0, prefix: str = "mw"):
    workspace = f"{prefix}-{uuid4().hex[:8]}"
    _shared_storage.initialize_share_data()
    await _shared_storage.initialize_pipeline_status(workspace=workspace)
    return SimpleNamespace(
        workspace=workspace,
        doc_status=_CountingDocStatus(active),
        max_pending_documents=capacity,
    )


async def _tokens(rag) -> dict:
    status = await _shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    return dict(status.get("pending_enqueue_tokens") or {})


def _scope(path="/documents/upload", method="POST", headers=None):
    return {
        "type": "http",
        "method": method,
        "path": path,
        "headers": headers or [],
    }


async def _no_existing_doc(*args, **kwargs):
    return None


@pytest.fixture
def _open_auth(monkeypatch):
    """Default to the fully-open auth profile; auth tests opt back in."""
    monkeypatch.setattr(_utils_api, "auth_configured", False)


# --------------------------------------------------------------------------- #
# Pre-body admission middleware
# --------------------------------------------------------------------------- #


class _Downstream:
    """Records whether the app was reached, whether the body was read, and what
    ticket (if any) the route would have adopted."""

    def __init__(self, *, adopt: bool = True, fail: bool = False):
        self.calls = 0
        self.body_reads = 0
        self.tickets: list = []
        self._adopt = adopt
        self._fail = fail

    async def __call__(self, scope, receive, send):
        self.calls += 1
        ticket = (scope.get("state") or {}).get(_admission.ADMISSION_STATE_KEY)
        self.tickets.append(ticket)
        if self._adopt and ticket is not None:
            ticket.adopted = True
        message = await receive()
        self.body_reads += 1
        assert message["type"] == "http.request"
        if self._fail:
            raise RuntimeError("route blew up after adopting")
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": b"{}"})


class _AdmissionRecorder:
    """Captures the ASGI response and counts ``receive()`` calls."""

    def __init__(self):
        self.messages: list[dict] = []
        self.receives = 0

    async def receive(self):
        self.receives += 1
        return {"type": "http.request", "body": b"x", "more_body": False}

    async def send(self, message):
        self.messages.append(message)

    @property
    def status(self):
        for message in self.messages:
            if message["type"] == "http.response.start":
                return message["status"]
        return None

    @property
    def headers(self) -> dict[str, str]:
        for message in self.messages:
            if message["type"] == "http.response.start":
                return {
                    k.decode("latin-1"): v.decode("latin-1")
                    for k, v in message["headers"]
                }
        return {}

    def body(self) -> bytes:
        return b"".join(
            m.get("body", b"")
            for m in self.messages
            if m["type"] == "http.response.body"
        )


def _admission_mw(rag, downstream, **kwargs):
    return AdmissionMiddleware(downstream, rag_getter=lambda: rag, **kwargs)


@pytest.mark.usefixtures("_open_auth")
class TestAdmissionMiddleware:
    """Pre-body admission middleware tests."""

    async def test_refused_request_never_reads_the_body(self):
        rag = await _rag(capacity=1, active=1)
        downstream = _Downstream()
        recorder = _AdmissionRecorder()

        await _admission_mw(rag, downstream)(
            _scope(), recorder.receive, recorder.send
        )

        assert recorder.status == 429
        assert recorder.receives == 0
        assert downstream.calls == 0
        assert recorder.headers["retry-after"]
        assert b"capacity" in recorder.body()
        assert await _tokens(rag) == {}

    async def test_capacity_one_admits_exactly_one_of_many_concurrent_requests(self):
        rag = await _rag(capacity=1, active=0)
        downstream = _Downstream()
        middleware = _admission_mw(rag, downstream)

        async def _one():
            recorder = _AdmissionRecorder()
            await middleware(_scope(), recorder.receive, recorder.send)
            return recorder

        recorders = await asyncio.gather(*[_one() for _ in range(50)])

        admitted = [r for r in recorders if r.status == 200]
        refused = [r for r in recorders if r.status == 429]
        assert len(admitted) == 1
        assert len(refused) == 49
        assert downstream.body_reads == 1
        assert sum(r.receives for r in refused) == 0

    async def test_unauthenticated_request_is_refused_before_reserving(self):
        rag = await _rag(capacity=10)
        downstream = _Downstream()
        recorder = _AdmissionRecorder()

        await _admission_mw(rag, downstream, api_key="secret")(
            _scope(), recorder.receive, recorder.send
        )

        assert recorder.status == 401
        assert recorder.receives == 0
        assert downstream.calls == 0
        assert await _tokens(rag) == {}

    async def test_valid_api_key_is_admitted(self):
        rag = await _rag(capacity=10)
        downstream = _Downstream()
        recorder = _AdmissionRecorder()

        await _admission_mw(rag, downstream, api_key="secret")(
            _scope(headers=[(b"x-api-key", b"secret")]),
            recorder.receive,
            recorder.send,
        )

        assert recorder.status == 200
        assert downstream.calls == 1

    async def test_whitelisted_path_skips_pre_auth_like_the_route_does(
        self, monkeypatch
    ):
        monkeypatch.setattr(_utils_api, "whitelist_patterns", [("", True)])  # "/*"
        rag = await _rag(capacity=10)
        downstream = _Downstream()
        recorder = _AdmissionRecorder()

        await _admission_mw(rag, downstream, api_key="secret")(
            _scope(), recorder.receive, recorder.send
        )

        assert recorder.status == 200

    async def test_ticket_is_published_and_adopted_exactly_once(self):
        rag = await _rag(capacity=10)
        downstream = _Downstream(adopt=True)
        recorder = _AdmissionRecorder()

        await _admission_mw(rag, downstream)(
            _scope(), recorder.receive, recorder.send
        )

        ticket = downstream.tickets[0]
        assert ticket is not None and ticket.adopted is True
        assert list((await _tokens(rag)).keys()) == [ticket.token]

    async def test_unadopted_ticket_is_released_by_the_middleware(self):
        rag = await _rag(capacity=10)
        downstream = _Downstream(adopt=False)
        recorder = _AdmissionRecorder()

        await _admission_mw(rag, downstream)(
            _scope(), recorder.receive, recorder.send
        )

        assert recorder.status == 200
        assert await _tokens(rag) == {}

    async def test_route_exception_after_adoption_leaves_ownership_downstream(self):
        rag = await _rag(capacity=10)
        downstream = _Downstream(adopt=True, fail=True)
        recorder = _AdmissionRecorder()

        with pytest.raises(RuntimeError, match="blew up"):
            await _admission_mw(rag, downstream)(
                _scope(), recorder.receive, recorder.send
            )

        ticket = downstream.tickets[0]
        assert list((await _tokens(rag)).keys()) == [ticket.token]

    async def test_fenced_pipeline_is_409_before_the_body(self):
        rag = await _rag(capacity=10)
        status = await _shared_storage.get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
        status["manual_freeze_requested"] = True
        downstream = _Downstream()
        recorder = _AdmissionRecorder()

        await _admission_mw(rag, downstream)(
            _scope(), recorder.receive, recorder.send
        )

        assert recorder.status == 409
        assert recorder.receives == 0
        assert downstream.calls == 0

    @pytest.mark.parametrize(
        "scope_kwargs",
        [
            {"path": "/documents/scan"},
            {"path": "/query"},
            {"path": "/documents/upload", "method": "GET"},
        ],
    )
    async def test_non_ingestion_requests_pass_through_untouched(self, scope_kwargs):
        rag = await _rag(capacity=1, active=99)
        downstream = _Downstream()
        recorder = _AdmissionRecorder()

        await _admission_mw(rag, downstream)(
            _scope(**scope_kwargs), recorder.receive, recorder.send
        )

        assert downstream.calls == 1
        assert downstream.tickets == [None]

    async def test_api_prefix_is_stripped_before_matching(self):
        rag = await _rag(capacity=1, active=1)
        downstream = _Downstream()
        recorder = _AdmissionRecorder()

        await _admission_mw(rag, downstream, api_prefix="/lightrag")(
            _scope(path="/lightrag/documents/upload"),
            recorder.receive,
            recorder.send,
        )

        assert recorder.status == 429
        assert downstream.calls == 0

    @pytest.mark.parametrize(
        "path",
        [
            "/api/v1/documents/upload",  # verbatim forwarding
            "/documents/upload",  # nginx stripped the prefix
        ],
    )
    async def test_mount_prefix_never_collapses_the_pre_auth_check(
        self, monkeypatch, path
    ):
        monkeypatch.setattr(_utils_api, "auth_configured", True)
        monkeypatch.setattr(
            _utils_api, "whitelist_patterns", [("/health", False), ("/api", True)]
        )
        rag = await _rag(capacity=10)
        downstream = _Downstream()
        recorder = _AdmissionRecorder()

        scope = _scope(path=path)
        scope["root_path"] = "/api/v1"
        await _admission_mw(rag, downstream, api_key="secret")(
            scope, recorder.receive, recorder.send
        )

        assert recorder.status == 401
        assert downstream.calls == 0
        assert recorder.receives == 0
        assert await _tokens(rag) == {}

    async def test_disabled_capacity_is_a_pure_passthrough(self):
        rag = await _rag(capacity=0, active=10_000)
        downstream = _Downstream()
        recorder = _AdmissionRecorder()

        await _admission_mw(rag, downstream)(
            _scope(), recorder.receive, recorder.send
        )

        assert downstream.calls == 1
        assert downstream.tickets == [None]
        assert await _tokens(rag) == {}

    async def test_unresolvable_rag_degrades_to_the_route_reservation(self):
        downstream = _Downstream()
        recorder = _AdmissionRecorder()

        def _boom():
            raise RuntimeError("rag not built yet")

        middleware = AdmissionMiddleware(downstream, rag_getter=_boom)
        await middleware(_scope(), recorder.receive, recorder.send)

        assert downstream.calls == 1
        assert downstream.tickets == [None]

    async def test_route_adopts_the_middleware_reservation_instead_of_taking_a_second(
        self, monkeypatch
    ):
        rag = await _rag(capacity=1, active=0)
        seen: dict = {}

        async def _fake_index(rag_arg, file_path, track_id=None, admission_token=None):
            seen["token"] = admission_token
            seen["tokens_during"] = await _tokens(rag)

        reserved_tokens: list[str] = []
        original_reserve = _document_routes._reserve_enqueue_slot

        async def _tracking_reserve(rag_arg, token, **kwargs):
            reserved_tokens.append(token)
            return await original_reserve(rag_arg, token, **kwargs)

        doc_manager = _document_routes.DocumentManager(str(tempfile.mkdtemp()))
        rag.doc_status.resolve_doc_source_strict = None
        rag.doc_status.get_doc_by_file_basename = lambda *a, **k: None

        monkeypatch.setattr(_document_routes, "pipeline_index_file", _fake_index)
        monkeypatch.setattr(
            _document_routes, "_reserve_enqueue_slot", _tracking_reserve
        )
        monkeypatch.setattr(
            _document_routes,
            "global_args",
            SimpleNamespace(max_upload_size=None),
        )
        monkeypatch.setattr(
            _document_routes,
            "get_existing_doc_by_file_path_candidates",
            _no_existing_doc,
        )
        router = _document_routes.create_document_routes(rag, doc_manager)
        upload_endpoint = [
            route.endpoint
            for route in router.routes
            if getattr(route, "name", "") == "upload_to_input_dir"
        ][-1]

        ticket = _admission.AdmissionTicket(token="mw-token", weight=1)
        await _document_routes._reserve_enqueue_slot(rag, ticket.token, weight=1)
        request = SimpleNamespace(
            state=SimpleNamespace(**{_admission.ADMISSION_STATE_KEY: ticket})
        )

        upload_file = _document_routes.UploadFile(
            filename="adopted.txt", file=BytesIO(b"hello")
        )
        response = await upload_endpoint(set(), upload_file, request)
        assert response.status == "success"

        assert ticket.adopted is True
        assert reserved_tokens == ["mw-token"]
        assert seen["token"] == "mw-token"
        assert list(seen["tokens_during"].keys()) == ["mw-token"]
        assert await _tokens(rag) == {}


# --------------------------------------------------------------------------- #
# Raw request-body ceiling middleware (GHSA-r8jh-295g-vv42)
# --------------------------------------------------------------------------- #


class _BodyRecorder:
    """Supplies receive/send and records what the body-limit middleware did."""

    def __init__(self, chunks: list[bytes] | None = None):
        self._chunks = list(chunks or [])
        self.receives = 0
        self.delivered = 0
        self.status = None
        self.body = b""

    async def receive(self):
        self.receives += 1
        if not self._chunks:
            return {"type": "http.disconnect"}
        chunk = self._chunks.pop(0)
        self.delivered += 1
        return {
            "type": "http.request",
            "body": chunk,
            "more_body": bool(self._chunks),
        }

    async def send(self, message):
        if message["type"] == "http.response.start":
            self.status = message["status"]
        elif message["type"] == "http.response.body":
            self.body += message.get("body", b"")


class _BodyReader:
    """Downstream app that drains the whole body, like multipart parsing does."""

    def __init__(self):
        self.received = b""
        self.completed = False
        self.calls = 0

    async def __call__(self, scope, receive, send):
        self.calls += 1
        while True:
            message = await receive()
            if message["type"] != "http.request":
                break
            self.received += message.get("body", b"")
            if not message.get("more_body"):
                break
        self.completed = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})


def _body_limit_mw(app, **kwargs):
    limits = {
        "default_limit": DEFAULT_LIMIT,
        "ingest_limit": INGEST_LIMIT,
        "upload_limit": UPLOAD_LIMIT,
    }
    limits.update(kwargs)
    return BodyLimitMiddleware(app, **limits)


def _body_limit_args(**kwargs):
    base = {
        "max_request_body_bytes": DEFAULT_LIMIT,
        "max_request_body_bytes_explicit": False,
        "max_upload_size": 104857600,
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


class TestBodyLimitMiddleware:
    """Raw request-body ceilings and interaction with admission."""

    async def test_oversized_declared_length_is_refused_before_the_body(self):
        app = _BodyReader()
        recorder = _BodyRecorder([b"x" * 100])

        await _body_limit_mw(app, default_limit=50)(
            _scope(path="/api/chat", headers=[(b"content-length", b"100")]),
            recorder.receive,
            recorder.send,
        )

        assert recorder.status == 413
        assert recorder.receives == 0
        assert app.completed is False

    async def test_understated_length_is_still_cut_off_mid_stream(self):
        app = _BodyReader()
        recorder = _BodyRecorder([b"x" * 100, b"y" * 100, b"z" * 100])

        await _body_limit_mw(app, default_limit=150)(
            _scope(path="/api/chat", headers=[(b"content-length", b"10")]),
            recorder.receive,
            recorder.send,
        )

        assert recorder.status == 413
        assert app.completed is False
        assert recorder.delivered == 2

    async def test_body_within_the_limit_streams_through_untouched(self):
        app = _BodyReader()
        recorder = _BodyRecorder([b"a" * 40, b"b" * 40])

        await _body_limit_mw(app, default_limit=100)(
            _scope(path="/api/chat"), recorder.receive, recorder.send
        )

        assert recorder.status == 200
        assert app.completed is True
        assert app.received == b"a" * 40 + b"b" * 40

    async def test_zero_limit_leaves_receive_unwrapped(self):
        app = _BodyReader()
        recorder = _BodyRecorder([b"x" * 10_000])

        await _body_limit_mw(app, default_limit=0)(
            _scope(path="/api/chat"), recorder.receive, recorder.send
        )

        assert recorder.status == 200
        assert len(app.received) == 10_000

    async def test_non_http_scopes_pass_through(self):
        app = _BodyReader()
        recorder = _BodyRecorder([b"x"])

        await _body_limit_mw(app)({"type": "lifespan"}, recorder.receive, recorder.send)

        assert app.calls == 1

    @pytest.mark.parametrize(
        "path,accepted,refused",
        [
            ("/api/chat", DEFAULT_LIMIT, DEFAULT_LIMIT + 1),
            ("/query", DEFAULT_LIMIT, DEFAULT_LIMIT + 1),
            ("/documents/text", DEFAULT_LIMIT + 1, INGEST_LIMIT + 1),
            ("/documents/texts", INGEST_LIMIT, INGEST_LIMIT + 1),
            ("/documents/upload", INGEST_LIMIT + 1, UPLOAD_LIMIT + 1),
        ],
    )
    async def test_each_route_tier_is_applied(self, path, accepted, refused):
        for declared, expected in ((accepted, 200), (refused, 413)):
            app = _BodyReader()
            recorder = _BodyRecorder([b""])
            await _body_limit_mw(app)(
                _scope(path=path, headers=[(b"content-length", str(declared).encode())]),
                recorder.receive,
                recorder.send,
            )
            assert recorder.status == expected, (path, declared)

    async def test_api_prefix_is_stripped_before_matching(self):
        app = _BodyReader()
        recorder = _BodyRecorder([b""])

        await _body_limit_mw(app, api_prefix="/site01")(
            {
                "type": "http",
                "method": "POST",
                "path": "/site01/documents/texts",
                "root_path": "/site01",
                "headers": [(b"content-length", str(DEFAULT_LIMIT + 1).encode())],
            },
            recorder.receive,
            recorder.send,
        )

        assert recorder.status == 200

    async def test_a_limit_applies_to_methods_other_than_post(self):
        app = _BodyReader()
        recorder = _BodyRecorder([b"x" * 200])

        await _body_limit_mw(app, default_limit=50)(
            _scope(path="/api/chat", method="PUT"), recorder.receive, recorder.send
        )

        assert recorder.status == 413

    async def test_mid_body_413_still_releases_the_admission_reservation(
        self, _open_auth
    ):
        rag = await _rag(capacity=10, prefix="bl")
        app = _BodyReader()
        admission = _admission_mod.AdmissionMiddleware(app, rag_getter=lambda: rag)
        stacked = _body_limit_mw(admission, upload_limit=50)
        recorder = _BodyRecorder([b"x" * 200])

        await stacked(_scope(path="/documents/upload"), recorder.receive, recorder.send)

        assert recorder.status == 413
        assert await _tokens(rag) == {}

    async def test_oversized_declared_length_never_reaches_admission(self, _open_auth):
        rag = await _rag(capacity=10, prefix="bl")
        app = _BodyReader()
        admission = _admission_mod.AdmissionMiddleware(app, rag_getter=lambda: rag)
        stacked = _body_limit_mw(admission, upload_limit=50)
        recorder = _BodyRecorder([b"x" * 200])

        await stacked(
            _scope(path="/documents/upload", headers=[(b"content-length", b"200")]),
            recorder.receive,
            recorder.send,
        )

        assert recorder.status == 413
        assert recorder.receives == 0
        assert app.calls == 0
        assert await _tokens(rag) == {}

    def test_defaults_produce_the_three_tiers(self):
        limits = resolve_body_limits(_body_limit_args())
        assert limits == {
            "default_limit": DEFAULT_LIMIT,
            "ingest_limit": INGEST_LIMIT,
            "upload_limit": 104857600 + 1024 * 1024,
        }

    def test_an_explicit_value_governs_every_non_upload_route(self):
        limits = resolve_body_limits(
            _body_limit_args(
                max_request_body_bytes=4096, max_request_body_bytes_explicit=True
            )
        )
        assert limits["default_limit"] == 4096
        assert limits["ingest_limit"] == 4096
        assert limits["upload_limit"] == 104857600 + 1024 * 1024

    def test_an_explicit_value_equal_to_the_default_still_governs_ingestion(self):
        limits = resolve_body_limits(
            _body_limit_args(
                max_request_body_bytes=DEFAULT_LIMIT,
                max_request_body_bytes_explicit=True,
            )
        )
        assert limits["default_limit"] == DEFAULT_LIMIT
        assert limits["ingest_limit"] == DEFAULT_LIMIT

    def test_missing_provenance_is_treated_as_configured(self):
        args = SimpleNamespace(
            max_request_body_bytes=DEFAULT_LIMIT, max_upload_size=104857600
        )
        assert resolve_body_limits(args)["ingest_limit"] == DEFAULT_LIMIT

    def test_zero_disables_every_ceiling_including_the_derived_upload_one(self):
        assert resolve_body_limits(_body_limit_args(max_request_body_bytes=0)) is None

    @pytest.mark.parametrize("value", [None, 0, -1])
    def test_unlimited_upload_size_leaves_the_upload_route_uncapped(self, value):
        limits = resolve_body_limits(_body_limit_args(max_upload_size=value))
        assert limits["upload_limit"] == 0
        assert limits["default_limit"] == DEFAULT_LIMIT
