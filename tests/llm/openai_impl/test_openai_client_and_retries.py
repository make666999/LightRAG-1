"""OpenAI client configuration, HTTP client ownership, retry classification, and streaming COT cleanup."""

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from tenacity import wait_none

from lightrag.llm._error_utils import is_permanent_rate_limit_error
from lightrag.llm.openai import (
    TransientBadRequestError,
    _caller_supplied_http_client,
    create_openai_async_client,
    openai_complete_if_cache,
    openai_embed,
)

pytestmark = pytest.mark.offline


def _chat_success_body(content: str = "OK") -> dict:
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _embed_success_body(values=(0.1, 0.2, 0.3)) -> dict:
    embedding = np.array(values, dtype=np.float32)
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": 0,
                "embedding": base64.b64encode(embedding.tobytes()).decode("ascii"),
            }
        ],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }


def _make_error_client(error: Exception) -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(side_effect=error))
        ),
        close=AsyncMock(),
    )


@pytest.fixture
def no_retry_wait(monkeypatch):
    monkeypatch.setattr(openai_complete_if_cache.retry, "wait", wait_none())
    monkeypatch.setattr(openai_embed.func.retry, "wait", wait_none())


# --------------------------------------------------------------------------- #
# Client headers
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["completion", "embedding"])
async def test_client_config_headers_reach_requests(monkeypatch, operation):
    monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID", "environment-workspace")
    captured = []

    def respond(request):
        captured.append(request)
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": 0, "embedding": [0.0] * 1536}
                    ],
                    "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
            )
        return httpx.Response(200, json=_chat_success_body("done"))

    headers = {
        "X-Custom-Route": "caller-route",
        "User-Agent": "caller-agent",
        "X-DashScope-Workspace": "caller-workspace",
    }
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    configs = {"default_headers": headers, "http_client": http_client}
    options = {"api_key": "test-key", "base_url": "https://example.invalid/v1"}
    try:
        if operation == "completion":
            assert (
                await openai_complete_if_cache(
                    "test-model", "hello", openai_client_configs=configs, **options
                )
                == "done"
            )
        else:
            result = await openai_embed(["hello"], client_configs=configs, **options)
            assert result.shape == (1, 1536)
    finally:
        await http_client.aclose()

    assert len(captured) == 1
    sent = captured[0].headers
    assert sent["X-Custom-Route"] == "caller-route"
    assert sent["User-Agent"] == "caller-agent"
    assert sent["X-DashScope-Workspace"] == "caller-workspace"
    assert sent["Content-Type"] == "application/json"
    assert configs["default_headers"] is headers
    assert headers == {
        "X-Custom-Route": "caller-route",
        "User-Agent": "caller-agent",
        "X-DashScope-Workspace": "caller-workspace",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("configs", [None, {"default_headers": None}])
async def test_missing_custom_headers_keep_lightrag_defaults(monkeypatch, configs):
    monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID", "  environment-workspace  ")
    async with create_openai_async_client(
        api_key="test-key", client_configs=configs
    ) as client:
        assert "LightRAG/" in client.default_headers["User-Agent"]
        assert client.default_headers["Content-Type"] == "application/json"
        assert (
            client.default_headers["X-DashScope-Workspace"] == "environment-workspace"
        )


@pytest.mark.asyncio
async def test_azure_custom_headers_are_preserved(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID", "environment-workspace")
    headers = {"X-Custom-Route": "azure-route"}
    async with create_openai_async_client(
        api_key="test-key",
        base_url="https://example.invalid",
        use_azure=True,
        api_version="2024-02-01",
        client_configs={"default_headers": headers},
    ) as client:
        assert client.default_headers["X-Custom-Route"] == "azure-route"
        assert "X-DashScope-Workspace" not in client.default_headers
    assert headers == {"X-Custom-Route": "azure-route"}


@pytest.mark.asyncio
async def test_caller_headers_override_defaults_case_insensitively(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID", "environment-workspace")
    captured = []

    def respond(request):
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.0] * 8}],
                "model": "test-embedding",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    headers = {
        "user-agent": "caller-agent",
        "x-dashscope-workspace": "caller-workspace",
    }
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        await openai_embed.func(
            ["hello"],
            model="test-embedding",
            api_key="test-key",
            base_url="https://example.invalid/v1",
            client_configs={"default_headers": headers, "http_client": http_client},
        )
    finally:
        await http_client.aclose()

    assert len(captured) == 1
    sent = captured[0].headers
    assert sent.get_list("User-Agent") == ["caller-agent"]
    assert sent.get_list("X-DashScope-Workspace") == ["caller-workspace"]
    assert sent.get_list("Content-Type") == ["application/json"]
    assert headers == {
        "user-agent": "caller-agent",
        "x-dashscope-workspace": "caller-workspace",
    }


# --------------------------------------------------------------------------- #
# HTTP client ownership
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "configs, expected",
    [
        (None, False),
        ({}, False),
        ({"http_client": None}, False),
        ({"http_async_client": object()}, False),
        ({"http_client": object()}, True),
    ],
)
def test_caller_supplied_http_client_requires_a_live_instance(configs, expected):
    assert _caller_supplied_http_client(configs) is expected


@pytest.mark.asyncio
async def test_complete_retries_on_injected_http_client(no_retry_wait):
    calls = {"http": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["http"] += 1
        if calls["http"] == 1:
            raise httpx.ConnectError("transient blip", request=request)
        return httpx.Response(200, json=_chat_success_body())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        out = await openai_complete_if_cache(
            "gpt-4o-mini",
            "hi",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            openai_client_configs={"http_client": client, "max_retries": 0},
        )
        assert out == "OK"
        assert calls["http"] == 2
        assert client.is_closed is False

        out_again = await openai_complete_if_cache(
            "gpt-4o-mini",
            "hi",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            openai_client_configs={"http_client": client, "max_retries": 0},
        )
        assert out_again == "OK"
        assert calls["http"] == 3
        assert client.is_closed is False
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_embed_retries_on_injected_http_client(no_retry_wait):
    calls = {"http": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["http"] += 1
        if calls["http"] == 1:
            raise httpx.ConnectError("transient blip", request=request)
        return httpx.Response(200, json=_embed_success_body())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        first = await openai_embed.func(
            texts=["hi"],
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            client_configs={"http_client": client, "max_retries": 0},
        )
        np.testing.assert_allclose(first, [[0.1, 0.2, 0.3]], rtol=1e-6)
        assert calls["http"] == 2
        assert client.is_closed is False

        second = await openai_embed.func(
            texts=["hi"],
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            client_configs={"http_client": client, "max_retries": 0},
        )
        np.testing.assert_allclose(second, [[0.1, 0.2, 0.3]], rtol=1e-6)
        assert calls["http"] == 3
        assert client.is_closed is False
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_http_client_none_still_closes_sdk_owned_client():
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    error = APIConnectionError(request=request, message="boom")
    fake = _make_error_client(error)

    with patch("lightrag.llm.openai.create_openai_async_client", return_value=fake):
        with pytest.raises(APIConnectionError):
            await openai_complete_if_cache.__wrapped__(
                model="gpt-4o-mini",
                prompt="hi",
                api_key="sk-test",
                openai_client_configs={"http_client": None},
            )

    fake.close.assert_awaited()


class _TrackingAsyncByteStream(httpx.AsyncByteStream):
    """Yield SSE events lazily so early response closure is observable."""

    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.chunks_yielded = 0
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            self.chunks_yielded += 1
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_real_sdk_stream_early_stop_closes_response_and_reuses_client():
    events = [
        {
            "id": "x",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [
                {"index": 0, "delta": {"content": content}, "finish_reason": None}
            ],
        }
        for content in ("hello", " world")
    ]
    body = _TrackingAsyncByteStream(
        [f"data: {json.dumps(event)}\n\n".encode() for event in events]
        + [b"data: [DONE]\n\n"]
    )
    response = httpx.Response(
        200, headers={"content-type": "text/event-stream"}, stream=body
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            assert json.loads(request.content)["stream"] is True
            return response
        assert json.loads(request.content).get("stream", False) is False
        return httpx.Response(200, json=_chat_success_body())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        configs = {"http_client": client, "max_retries": 0}
        gen = await openai_complete_if_cache(
            model="gpt-4o-mini",
            prompt="hi",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            stream=True,
            openai_client_configs=configs,
        )
        try:
            assert await gen.__anext__() == "hello"
            assert body.chunks_yielded == 1
            assert body.closed is False
            assert response.is_closed is False
        finally:
            await gen.aclose()

        assert body.closed is True
        assert response.is_closed is True
        assert body.chunks_yielded == 1
        assert client.is_closed is False

        result = await openai_complete_if_cache(
            model="gpt-4o-mini",
            prompt="hi again",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            openai_client_configs=configs,
        )
        assert result == "OK"
        assert len(requests) == 2
        assert client.is_closed is False


class _FailingAfterFirstChunkStream(httpx.AsyncByteStream):
    """Yield one SSE event, then fail the read like a dropped connection."""

    def __init__(self, first_chunk: bytes) -> None:
        self.first_chunk = first_chunk
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.first_chunk
        raise httpx.ReadError("mid-stream failure")

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_real_sdk_stream_error_after_chunk_propagates_and_keeps_client():
    event = {
        "id": "x",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "delta": {"content": "A"}, "finish_reason": None}],
    }
    body = _FailingAfterFirstChunkStream(f"data: {json.dumps(event)}\n\n".encode())
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=body
            )
        return httpx.Response(200, json=_chat_success_body())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        configs = {"http_client": client, "max_retries": 0}
        gen = await openai_complete_if_cache(
            model="gpt-4o-mini",
            prompt="hi",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            stream=True,
            openai_client_configs=configs,
        )
        received: list[str] = []
        with pytest.raises(httpx.ReadError, match="mid-stream failure"):
            async for piece in gen:
                received.append(piece)

        assert received == ["A"]
        assert body.closed is True
        assert client.is_closed is False

        result = await openai_complete_if_cache(
            model="gpt-4o-mini",
            prompt="hi again",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            openai_client_configs=configs,
        )
        assert result == "OK"
        assert len(requests) == 2
        assert client.is_closed is False


@pytest.mark.asyncio
async def test_cancelled_request_keeps_shared_client_open_for_concurrent_calls():
    slow_request_started = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["messages"][-1]["content"] == "slow":
            slow_request_started.set()
            await asyncio.Event().wait()
        return httpx.Response(200, json=_chat_success_body())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        configs = {"http_client": client, "max_retries": 0}
        slow = asyncio.create_task(
            openai_complete_if_cache(
                model="gpt-4o-mini",
                prompt="slow",
                api_key="sk-test",
                base_url="https://api.openai.com/v1",
                openai_client_configs=configs,
            )
        )
        await asyncio.wait_for(slow_request_started.wait(), timeout=5)
        slow.cancel()
        with pytest.raises(asyncio.CancelledError):
            await slow

        assert client.is_closed is False
        result = await openai_complete_if_cache(
            model="gpt-4o-mini",
            prompt="fast",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            openai_client_configs=configs,
        )
        assert result == "OK"
        assert client.is_closed is False


@pytest.mark.asyncio
async def test_cancelled_request_still_closes_sdk_owned_client():
    started = asyncio.Event()

    async def hang(**_kwargs):
        started.set()
        await asyncio.Event().wait()

    fake = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=hang)),
        close=AsyncMock(),
    )
    with patch("lightrag.llm.openai.create_openai_async_client", return_value=fake):
        task = asyncio.create_task(
            openai_complete_if_cache.__wrapped__(
                model="gpt-4o-mini", prompt="hi", api_key="sk-test"
            )
        )
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    fake.close.assert_awaited()


# --------------------------------------------------------------------------- #
# Permanent rate limit detection & HTTP-level retry classification
# --------------------------------------------------------------------------- #

BUDGET_BODY = {
    "message": (
        "Budget has been exceeded! Team=6ef183de-04ba-4114-8ef6-4b628370bc8c "
        "Current cost: 180.1173603404399, Max budget: 180.0"
    ),
    "type": "budget_exceeded",
    "param": None,
    "code": "429",
}
QUOTA_BODY = {
    "message": (
        "You exceeded your current quota, please check your plan and billing details."
    ),
    "type": "insufficient_quota",
    "param": None,
    "code": "insufficient_quota",
}
THROTTLE_BODY = {
    "message": "Rate limit reached for gpt-4o-mini. Please try again in 1s.",
    "type": "requests",
    "param": None,
    "code": "rate_limit_exceeded",
}


def _make_rate_limit_error(body: object, message: str | None = None) -> RateLimitError:
    request = httpx.Request("POST", "https://proxy.example/v1/chat/completions")
    response = httpx.Response(status_code=429, request=request)
    return RateLimitError(
        message or f"Error code: 429 - {body}", response=response, body=body
    )


def _counting_transport(
    counter: list[int], body: dict | None, status_code: int = 429
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        counter[0] += 1
        return httpx.Response(status_code, json={"error": body} if body else {})

    return httpx.MockTransport(handler)


def _disconnecting_transport(counter: list[int]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        counter[0] += 1
        raise httpx.ConnectError("connection refused", request=request)

    return httpx.MockTransport(handler)


def _client_factory(transport: httpx.MockTransport):
    def factory(**_kwargs):
        return create_openai_async_client(
            api_key="test-key",
            base_url="https://proxy.example/v1",
            client_configs={"http_client": httpx.AsyncClient(transport=transport)},
        )

    return factory


async def _call_complete():
    return await openai_complete_if_cache(model="gpt-4o-mini", prompt="hello")


async def _call_embed():
    return await openai_embed(["hello"])


@pytest.mark.parametrize(
    "body, message",
    [
        pytest.param(BUDGET_BODY, "Error code: 429", id="budget-unwrapped-body"),
        pytest.param(
            {"error": QUOTA_BODY}, "Error code: 429", id="quota-nested-envelope"
        ),
        pytest.param(
            None,
            "Error code: 429 - {'error': {'message': 'Budget has been exceeded! "
            "Team=x', 'type': 'budget_exceeded', 'param': None, 'code': '429'}}",
            id="message-only",
        ),
    ],
)
def test_permanent_spend_stop_is_detected(body, message):
    assert is_permanent_rate_limit_error(_make_rate_limit_error(body, message)) is True


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"error": THROTTLE_BODY}, id="nested-throughput-throttle"),
        pytest.param(None, id="no-body"),
        pytest.param("plain text error page", id="non-mapping-body"),
    ],
)
def test_ordinary_rate_limit_is_not_a_spend_stop(body):
    assert is_permanent_rate_limit_error(_make_rate_limit_error(body)) is False


@pytest.mark.parametrize(
    "call, body",
    [
        pytest.param(_call_complete, BUDGET_BODY, id="complete-budget-exceeded"),
        pytest.param(_call_embed, QUOTA_BODY, id="embed-insufficient-quota"),
    ],
)
@pytest.mark.asyncio
async def test_spend_stop_costs_exactly_one_http_request(
    monkeypatch, no_retry_wait, call, body
):
    calls = [0]
    monkeypatch.setattr(
        "lightrag.llm.openai.create_openai_async_client",
        _client_factory(_counting_transport(calls, body)),
    )

    with pytest.raises(RateLimitError) as excinfo:
        await call()

    assert calls[0] == 1
    assert body["message"] in str(excinfo.value)


@pytest.mark.parametrize(
    "call", [_call_complete, _call_embed], ids=["complete", "embed"]
)
@pytest.mark.asyncio
async def test_ordinary_429_costs_exactly_the_tenacity_attempts(
    monkeypatch, no_retry_wait, call
):
    calls = [0]
    monkeypatch.setattr(
        "lightrag.llm.openai.create_openai_async_client",
        _client_factory(_counting_transport(calls, THROTTLE_BODY)),
    )

    with pytest.raises(RateLimitError) as excinfo:
        await call()

    assert calls[0] == 3
    assert "Rate limit reached" in str(excinfo.value)


def test_client_configs_can_restore_sdk_retries():
    client = create_openai_async_client(api_key="k", client_configs={"max_retries": 2})
    assert client.max_retries == 2


@pytest.mark.parametrize(
    ("call", "status_code", "expected_calls"),
    [
        pytest.param(_call_complete, 408, 3, id="complete-408-retried"),
        pytest.param(_call_complete, 409, 3, id="complete-409-retried"),
        pytest.param(_call_embed, 408, 3, id="embed-408-retried"),
        pytest.param(_call_complete, 400, 1, id="complete-400-fail-fast"),
        pytest.param(_call_complete, 422, 1, id="complete-422-fail-fast"),
    ],
)
@pytest.mark.asyncio
async def test_status_code_retry_policy(
    monkeypatch, no_retry_wait, call, status_code, expected_calls
):
    calls = [0]
    monkeypatch.setattr(
        "lightrag.llm.openai.create_openai_async_client",
        _client_factory(_counting_transport(calls, None, status_code)),
    )

    with pytest.raises(APIStatusError) as excinfo:
        await call()

    assert calls[0] == expected_calls
    assert excinfo.value.status_code == status_code


@pytest.mark.parametrize(
    "call", [_call_complete, _call_embed], ids=["complete", "embed"]
)
@pytest.mark.asyncio
async def test_exhausted_5xx_reraises_internal_server_error(
    monkeypatch, no_retry_wait, call
):
    calls = [0]
    monkeypatch.setattr(
        "lightrag.llm.openai.create_openai_async_client",
        _client_factory(
            _counting_transport(calls, {"message": "upstream connect error"}, 503)
        ),
    )

    with pytest.raises(InternalServerError) as excinfo:
        await call()

    assert calls[0] == 3
    assert excinfo.value.status_code == 503
    assert "upstream connect error" in str(excinfo.value)


@pytest.mark.parametrize(
    "call", [_call_complete, _call_embed], ids=["complete", "embed"]
)
@pytest.mark.asyncio
async def test_exhausted_connection_error_reraises_api_connection_error(
    monkeypatch, no_retry_wait, call
):
    calls = [0]
    monkeypatch.setattr(
        "lightrag.llm.openai.create_openai_async_client",
        _client_factory(_disconnecting_transport(calls)),
    )

    with pytest.raises(APIConnectionError):
        await call()

    assert calls[0] == 3


# --------------------------------------------------------------------------- #
# Transient JSON-parse 400 wrapping
# --------------------------------------------------------------------------- #


def _retry_exception_types(func) -> set[type]:
    types: set[type] = set()

    def _walk(retry_obj):
        for child in getattr(retry_obj, "retries", ()):
            _walk(child)
        exc_types = getattr(retry_obj, "exception_types", ())
        if isinstance(exc_types, type):
            exc_types = (exc_types,)
        types.update(exc_types)

    _walk(func.retry.retry)
    return types


def _make_bad_request(message: str) -> BadRequestError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status_code=400, request=request)
    return BadRequestError(message, response=response, body=None)


def test_complete_retries_transient_400():
    assert TransientBadRequestError in _retry_exception_types(openai_complete_if_cache)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "expected_exc"),
    [
        pytest.param(
            "Error code: 400 - We could not parse the JSON body of your request.",
            TransientBadRequestError,
            id="transient-json-parse-wrapped",
        ),
        pytest.param(
            "Error code: 400 - Invalid value for 'temperature'.",
            BadRequestError,
            id="genuine-400-propagates",
        ),
    ],
)
async def test_bad_request_transient_wrapping(message, expected_exc):
    fake_client = _make_error_client(_make_bad_request(message))
    with patch(
        "lightrag.llm.openai.create_openai_async_client", return_value=fake_client
    ):
        with pytest.raises(expected_exc):
            await openai_complete_if_cache.__wrapped__(
                model="gpt-4o-mini", prompt="hello"
            )
    fake_client.close.assert_awaited()


# --------------------------------------------------------------------------- #
# Streaming COT GeneratorExit & CancelledError cleanup
# --------------------------------------------------------------------------- #


class _FakeOpenAIStream:
    """Async-iterable stand-in for openai.AsyncStream."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.aclose = AsyncMock()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class _FailingOpenAIStream(_FakeOpenAIStream):
    """Stream that raises after yielding its seeded chunks."""

    def __init__(self, chunks, error):
        super().__init__(chunks)
        self._error = error

    async def __anext__(self):
        if not self._chunks:
            raise self._error
        return self._chunks.pop(0)


class _StallingOpenAIStream(_FakeOpenAIStream):
    """Stream whose next read blocks forever once seeded chunks are drained."""

    async def __anext__(self):
        if not self._chunks:
            await asyncio.Event().wait()
        return self._chunks.pop(0)


def _make_cot_chunk(content=None, reasoning_content=""):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=content, reasoning_content=reasoning_content
                )
            )
        ],
        usage=None,
    )


def _make_stream_client(stream: _FakeOpenAIStream) -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=stream))
        ),
        close=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_stream_closed_on_early_close_while_cot_open():
    stream = _FakeOpenAIStream(
        [
            _make_cot_chunk(reasoning_content="Let me think"),
            _make_cot_chunk(reasoning_content=" about this."),
            _make_cot_chunk(reasoning_content=" still reasoning..."),
        ]
    )
    fake_client = _make_stream_client(stream)

    with patch(
        "lightrag.llm.openai.create_openai_async_client", return_value=fake_client
    ):
        gen = await openai_complete_if_cache.__wrapped__(
            model="deepseek-reasoner",
            prompt="hello",
            enable_cot=True,
            stream=True,
            api_key="test-key",
        )
        first = await gen.__anext__()
        second = await gen.__anext__()
        await gen.aclose()

    assert first == "<think>"
    assert second == "Let me think"
    stream.aclose.assert_awaited()
    fake_client.close.assert_awaited()


@pytest.mark.asyncio
async def test_stream_closed_after_full_consumption_with_cot():
    stream = _FakeOpenAIStream(
        [
            _make_cot_chunk(reasoning_content="thinking"),
            _make_cot_chunk(content="answer"),
        ]
    )
    fake_client = _make_stream_client(stream)

    with patch(
        "lightrag.llm.openai.create_openai_async_client", return_value=fake_client
    ):
        gen = await openai_complete_if_cache.__wrapped__(
            model="deepseek-reasoner",
            prompt="hello",
            enable_cot=True,
            stream=True,
            api_key="test-key",
        )
        chunks = [chunk async for chunk in gen]

    assert chunks == ["<think>", "thinking", "</think>", "answer"]
    stream.aclose.assert_awaited()
    fake_client.close.assert_awaited()


@pytest.mark.asyncio
async def test_stream_closed_on_disconnect_during_error_path_cot_close():
    stream = _FailingOpenAIStream(
        [_make_cot_chunk(reasoning_content="thinking...")],
        RuntimeError("upstream 500 mid-reasoning"),
    )
    fake_client = _make_stream_client(stream)

    yielded = []
    with patch(
        "lightrag.llm.openai.create_openai_async_client", return_value=fake_client
    ):
        gen = await openai_complete_if_cache.__wrapped__(
            model="deepseek-reasoner",
            prompt="hello",
            enable_cot=True,
            stream=True,
            api_key="test-key",
        )
        async for piece in gen:
            yielded.append(piece)
            if piece == "</think>":
                break
        await gen.aclose()

    assert yielded == ["<think>", "thinking...", "</think>"]
    stream.aclose.assert_awaited()
    fake_client.close.assert_awaited()


@pytest.mark.asyncio
async def test_cancelled_read_mid_cot_propagates_and_cleans_up():
    stream = _StallingOpenAIStream([_make_cot_chunk(reasoning_content="thinking")])
    fake_client = _make_stream_client(stream)

    with patch(
        "lightrag.llm.openai.create_openai_async_client", return_value=fake_client
    ):
        gen = await openai_complete_if_cache.__wrapped__(
            model="deepseek-reasoner",
            prompt="hello",
            enable_cot=True,
            stream=True,
            api_key="test-key",
        )
        assert await gen.__anext__() == "<think>"
        assert await gen.__anext__() == "thinking"

        pending = asyncio.ensure_future(gen.__anext__())
        await asyncio.sleep(0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

    stream.aclose.assert_awaited()
    fake_client.close.assert_awaited()
