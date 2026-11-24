"""LoLLMs adapter tests: HTTP error handling, max_tokens alias, retries, and raw-text streaming."""

import asyncio
import threading
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import aiohttp
import pytest
from aiohttp import ClientResponseError, RequestInfo, web
from aiohttp.test_utils import TestServer
from tenacity import wait_none
from yarl import URL

from lightrag.llm.lollms import lollms_embed, lollms_model_if_cache

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]

_URL = URL("http://127.0.0.1:9600/lollms_generate")


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr(lollms_model_if_cache.retry, "wait", wait_none())


@asynccontextmanager
async def _error_server(status, body):
    requests = []

    async def respond(request):
        requests.append(request.path)
        await request.read()
        return web.Response(status=status, text=body, content_type="application/json")

    app = web.Application()
    app.router.add_post("/lollms_generate", respond)
    app.router.add_post("/lollms_embed", respond)
    async with TestServer(app) as server:
        yield str(server.make_url("")).rstrip("/"), requests


def _http_status_error(status):
    request_info = RequestInfo(_URL, "POST", {}, _URL)
    return ClientResponseError(request_info, (), status=status, message="unavailable")


class _RaisingResponse:
    """Async context manager that records an attempt and raises an error."""

    def __init__(self, counter, error):
        self._counter = counter
        self._error = error

    async def __aenter__(self):
        self._counter["n"] += 1
        raise self._error

    async def __aexit__(self, *exc_info):
        return False


class _RaisingSession:
    """Stub aiohttp.ClientSession returning _RaisingResponse."""

    def __init__(self, counter, error):
        self._counter = counter
        self._error = error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    def post(self, url, json):
        return _RaisingResponse(self._counter, self._error)


class _FlakyServer:
    """Drops the first ``drops`` connections, then answers normally."""

    def __init__(self, drops, body=b"model answer"):
        self.drops = drops
        self.body = body
        self.requests = 0
        server = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):  # noqa: N802
                server.requests += 1
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                if server.requests <= server.drops:
                    self.close_connection = True
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(server.body)))
                self.end_headers()
                self.wfile.write(server.body)

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, args=(0.01,), daemon=True
        )

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self._server.server_address[1]}"


@pytest.fixture
def http_resources(monkeypatch):
    sessions = []
    responses = []
    headers_received = asyncio.Event()
    client_session = aiohttp.ClientSession

    async def record_response(session, context, params):
        responses.append(params.response)
        headers_received.set()

    def create_session(*args, **kwargs):
        trace = aiohttp.TraceConfig()
        trace.on_request_end.append(record_response)
        session = client_session(*args, trace_configs=[trace], **kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr("lightrag.llm.lollms.aiohttp.ClientSession", create_session)
    return sessions, responses, headers_received


# --------------------------------------------------------------------------- #
# HTTP error statuses
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("status", "attempts"),
    [
        pytest.param(503, 3, id="transient-retried"),
        pytest.param(401, 1, id="permanent-fails-fast"),
    ],
)
async def test_generation_raises_on_error_status(status, attempts):
    async with _error_server(status, "provider error") as (base_url, requests):
        with pytest.raises(ClientResponseError) as exc_info:
            await lollms_model_if_cache("test-model", "hello", base_url=base_url)

    assert exc_info.value.status == status
    assert len(requests) == attempts


async def test_stream_raises_before_yielding_the_error_body():
    chunks = []
    async with _error_server(503, "provider error") as (base_url, requests):
        stream = await lollms_model_if_cache(
            "test-model", "hello", base_url=base_url, stream=True
        )
        with pytest.raises(ClientResponseError) as exc_info:
            async for chunk in stream:
                chunks.append(chunk)

    assert exc_info.value.status == 503
    assert chunks == []
    assert len(requests) == 1


async def test_embedding_raises_before_parsing_the_body():
    async with _error_server(503, '{"vector": [1.0, 2.0]}') as (base_url, _):
        with pytest.raises(ClientResponseError) as exc_info:
            await lollms_embed.func(["hello"], base_url=base_url)

    assert exc_info.value.status == 503


# --------------------------------------------------------------------------- #
# max_tokens -> n_predict alias and streaming session lifetime
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("generation_kwargs", "expected_n_predict"),
    [
        pytest.param({}, None, id="provider-default-is-preserved"),
        pytest.param(
            {"max_tokens": 37},
            37,
            id="generic-limit-maps-to-native-field",
        ),
        pytest.param(
            {"max_tokens": 37, "n_predict": 41},
            41,
            id="native-limit-takes-precedence",
        ),
        pytest.param(
            {"max_tokens": 37, "n_predict": 0},
            0,
            id="explicit-native-zero-takes-precedence",
        ),
    ],
)
async def test_max_tokens_alias_precedence(
    monkeypatch, generation_kwargs, expected_n_predict
):
    sent_requests = []

    class FakeResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def text(self):
            return "ok"

    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def post(self, url, json):
            sent_requests.append(json)
            return FakeResponse()

    monkeypatch.setattr("lightrag.llm.lollms.aiohttp.ClientSession", FakeSession)

    result = await lollms_model_if_cache(
        model="test-model", prompt="hello", **generation_kwargs
    )

    assert result == "ok"
    assert len(sent_requests) == 1
    assert sent_requests[0]["n_predict"] == expected_n_predict
    assert "max_tokens" not in sent_requests[0]


async def test_stream_keeps_client_session_open_while_consumed(monkeypatch):
    instances = []

    class StreamingFakeResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        @property
        def content(self):
            return self

        def iter_any(self):
            async def chunks():
                yield b"first chunk\n"
                yield b"second chunk\n"

            return chunks()

    class StreamingFakeSession:
        def __init__(self, *args, **kwargs):
            self.closed = False
            instances.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            self.closed = True
            return False

        def post(self, url, json):
            if self.closed:
                raise AssertionError("stream attempted to use a closed ClientSession")
            return StreamingFakeResponse()

    monkeypatch.setattr(
        "lightrag.llm.lollms.aiohttp.ClientSession", StreamingFakeSession
    )

    stream = await lollms_model_if_cache(
        model="test-model", prompt="hello", stream=True
    )

    assert [chunk async for chunk in stream] == ["first chunk\n", "second chunk\n"]
    assert len(instances) == 1
    assert instances[0].closed


# --------------------------------------------------------------------------- #
# Retry classification & end-to-end retry recovery
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("error", "attempts"),
    [
        pytest.param(
            aiohttp.ClientConnectionError("Cannot connect to host"),
            3,
            id="connection",
        ),
        pytest.param(_http_status_error(503), 3, id="transient-status"),
        pytest.param(_http_status_error(401), 1, id="permanent-status"),
        pytest.param(aiohttp.InvalidURL("htp//bad"), 1, id="invalid-url"),
    ],
)
async def test_retry_follows_the_shared_classification(monkeypatch, error, attempts):
    counter = {"n": 0}
    monkeypatch.setattr(
        "lightrag.llm.lollms.aiohttp.ClientSession",
        lambda *args, **kwargs: _RaisingSession(counter, error),
    )

    with pytest.raises(type(error)) as exc_info:
        await lollms_model_if_cache("test-model", "hello")

    assert exc_info.value is error
    assert counter["n"] == attempts


async def test_a_server_that_comes_back_is_recovered_not_failed():
    with _FlakyServer(drops=2) as server:
        result = await lollms_model_if_cache(
            "test-model", "hello", base_url=server.base_url
        )

        assert result == "model answer"
        assert server.requests == 3


async def test_an_expired_total_timeout_is_retried_end_to_end():
    connections = 0

    async def _never_answer(reader, writer):
        nonlocal connections
        connections += 1
        try:
            await reader.read()
        finally:
            writer.close()

    server = await asyncio.start_server(_never_answer, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        with pytest.raises(asyncio.TimeoutError):
            await lollms_model_if_cache(
                "test-model",
                "hello",
                base_url=f"http://127.0.0.1:{port}",
                timeout=0.1,
            )
    finally:
        server.close()
        await server.wait_closed()

    assert connections == 3


# --------------------------------------------------------------------------- #
# Raw-text streaming behavior
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "expected",
    [
        pytest.param(
            '  # Answer\r\n\n```python\n    print("世界 🌍")  \n```\n\n  ',
            id="markdown-and-code",
        ),
        pytest.param("x" * 150_000, id="long-line"),
        pytest.param("", id="empty"),
        pytest.param(" \t\r\n  ", id="whitespace-only"),
    ],
)
async def test_stream_preserves_response_text(expected, http_resources):
    async def generate(request):
        assert (await request.json())["stream"] is True
        return web.Response(text=expected)

    app = web.Application()
    app.router.add_post("/lollms_generate", generate)
    async with TestServer(app) as server:
        stream = await lollms_model_if_cache(
            "offline-model", "hello", base_url=str(server.make_url("")), stream=True
        )
        actual = "".join([part async for part in stream])

    assert actual == expected
    sessions, responses, _ = http_resources
    assert len(sessions) == len(responses) == 1
    assert sessions[0].closed
    assert responses[0].closed


async def test_stream_delivers_text_before_newline_or_eof():
    prefix_sent = asyncio.Event()
    allow_tail = asyncio.Event()

    async def generate(request):
        response = web.StreamResponse(headers={"Content-Type": "text/plain"})
        await response.prepare(request)
        await response.write(b"first ")
        prefix_sent.set()
        await allow_tail.wait()
        await response.write(b"second\n")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/lollms_generate", generate)
    async with TestServer(app) as server:
        stream = await lollms_model_if_cache(
            "offline-model", "hello", base_url=str(server.make_url("")), stream=True
        )
        reader = asyncio.create_task(anext(stream))
        try:
            await asyncio.wait_for(prefix_sent.wait(), timeout=2)
            first = await asyncio.wait_for(asyncio.shield(reader), timeout=2)
            assert first == "first "
            allow_tail.set()
            assert first + "".join([part async for part in stream]) == "first second\n"
        finally:
            allow_tail.set()
            await asyncio.gather(reader, return_exceptions=True)
            await stream.aclose()


@pytest.mark.parametrize("character", ["世", "🌍"])
async def test_stream_decodes_utf8_split_across_http_chunks(character):
    prefix_sent = asyncio.Event()
    allow_tail = asyncio.Event()
    encoded = character.encode("utf-8")

    async def generate(request):
        response = web.StreamResponse(headers={"Content-Type": "text/plain"})
        await response.prepare(request)
        await response.write(b"head " + encoded[:1])
        prefix_sent.set()
        await allow_tail.wait()
        await response.write(encoded[1:] + b" tail")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/lollms_generate", generate)
    async with TestServer(app) as server:
        stream = await lollms_model_if_cache(
            "offline-model", "hello", base_url=str(server.make_url("")), stream=True
        )
        reader = asyncio.create_task(anext(stream))
        try:
            await asyncio.wait_for(prefix_sent.wait(), timeout=2)
            first = await asyncio.wait_for(asyncio.shield(reader), timeout=2)
            assert first == "head "
            allow_tail.set()
            parts = [part async for part in stream]
            assert all(parts)
            assert first + "".join(parts) == f"head {character} tail"
        finally:
            allow_tail.set()
            await asyncio.gather(reader, return_exceptions=True)
            await stream.aclose()


@pytest.mark.parametrize("payload", [b"\xff", b"\xf0\x9f"])
async def test_stream_rejects_invalid_or_truncated_utf8_and_closes(
    payload, http_resources
):
    async def generate(request):
        return web.Response(body=payload, content_type="text/plain")

    app = web.Application()
    app.router.add_post("/lollms_generate", generate)
    async with TestServer(app) as server:
        stream = await lollms_model_if_cache(
            "offline-model", "hello", base_url=str(server.make_url("")), stream=True
        )
        with pytest.raises(UnicodeDecodeError):
            _ = [part async for part in stream]

    sessions, responses, _ = http_resources
    assert len(sessions) == len(responses) == 1
    assert sessions[0].closed
    assert responses[0].closed


async def test_stream_early_close_releases_response_and_session(http_resources):
    allow_finish = asyncio.Event()

    async def generate(request):
        response = web.StreamResponse(headers={"Content-Type": "text/plain"})
        await response.prepare(request)
        await response.write(b"first")
        await allow_finish.wait()
        return response

    app = web.Application()
    app.router.add_post("/lollms_generate", generate)
    async with TestServer(app) as server:
        stream = await lollms_model_if_cache(
            "offline-model", "hello", base_url=str(server.make_url("")), stream=True
        )
        try:
            assert await asyncio.wait_for(anext(stream), timeout=2) == "first"
            await stream.aclose()
            sessions, responses, _ = http_resources
            assert len(sessions) == len(responses) == 1
            assert sessions[0].closed
            assert responses[0].closed
        finally:
            allow_finish.set()
            await stream.aclose()


async def test_stream_cancellation_releases_response_and_session(http_resources):
    allow_finish = asyncio.Event()

    async def generate(request):
        response = web.StreamResponse(headers={"Content-Type": "text/plain"})
        await response.prepare(request)
        await allow_finish.wait()
        return response

    app = web.Application()
    app.router.add_post("/lollms_generate", generate)
    async with TestServer(app) as server:
        stream = await lollms_model_if_cache(
            "offline-model", "hello", base_url=str(server.make_url("")), stream=True
        )
        reader = asyncio.create_task(anext(stream))
        try:
            sessions, responses, headers_received = http_resources
            await asyncio.wait_for(headers_received.wait(), timeout=2)
            reader.cancel()
            with pytest.raises(asyncio.CancelledError):
                await reader
            assert len(sessions) == len(responses) == 1
            assert sessions[0].closed
            assert responses[0].closed
        finally:
            allow_finish.set()
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            await stream.aclose()
