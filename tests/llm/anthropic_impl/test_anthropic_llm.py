"""Anthropic LLM adapter tests: SDK retry budget, token usage, truncation, and Unicode streaming."""

import inspect
import json
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic._base_client as anthropic_base_client
import httpx
import pytest
from anthropic import APIConnectionError, APIStatusError, AsyncAnthropic
from starlette.applications import Starlette
from starlette.responses import StreamingResponse
from starlette.routing import Route
from tenacity import RetryError

from lightrag.exceptions import EmptyTruncatedResponseError
from lightrag.llm import anthropic as anthropic_binding
from lightrag.llm.anthropic import InvalidResponseError, anthropic_complete_if_cache
from lightrag.utils import TokenTracker, is_truncated_response

pytestmark = pytest.mark.offline

# The SDK accepts an HTTP client from the package it is built on (httpx2 or httpx).
sdk_http = getattr(anthropic_base_client, "httpx2", None) or anthropic_base_client.httpx
DEFAULT_ATTEMPTS = anthropic_binding._DEFAULT_MAX_RETRIES + 1


@pytest.fixture
def mock_anthropic_http(monkeypatch):
    clients = []
    monkeypatch.setattr(anthropic_base_client, "INITIAL_RETRY_DELAY", 0.001)

    def install(handler):
        def create_client(**kwargs):
            client = AsyncAnthropic(
                http_client=sdk_http.AsyncClient(
                    transport=sdk_http.MockTransport(handler)
                ),
                **kwargs,
            )
            clients.append(client)
            return client

        monkeypatch.setattr(anthropic_binding, "AsyncAnthropic", create_client)
        return clients

    return install


def _error_response(status, headers=None):
    return sdk_http.Response(
        status,
        headers=headers or {},
        json={
            "type": "error",
            "error": {"type": "api_error", "message": "synthetic failure"},
        },
    )


def _usage(*, input_tokens, output_tokens, cache_creation=None, cache_read=None):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
    )


class _EventStream:
    """Async iterator standing in for Anthropic's streaming response."""

    def __init__(self, events):
        self._events = iter(events)
        self.close = AsyncMock()

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._events)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def _make_stub_client(
    *, stop_reason: str, text: str | None = None, content: list | None = None
) -> SimpleNamespace:
    if content is None:
        content = [SimpleNamespace(type="text", text=text)]
    response = SimpleNamespace(content=content, stop_reason=stop_reason)
    return SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=response)),
        close=AsyncMock(),
    )


# --------------------------------------------------------------------------- #
# Retries
# --------------------------------------------------------------------------- #


def test_outer_tenacity_layer_is_gone():
    assert not hasattr(anthropic_complete_if_cache, "retry")
    assert not hasattr(anthropic_complete_if_cache, "__wrapped__")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,headers,expected_requests",
    [
        (429, {}, DEFAULT_ATTEMPTS),
        (529, {}, DEFAULT_ATTEMPTS),
        (400, {}, 1),
        (429, {"x-should-retry": "false"}, 1),
    ],
)
async def test_http_error_has_one_sdk_retry_budget(
    mock_anthropic_http, status, headers, expected_requests
):
    requests = []

    def respond(request):
        requests.append(request)
        return _error_response(status, headers)

    clients = mock_anthropic_http(respond)
    with pytest.raises((APIStatusError, RetryError)) as caught:
        await anthropic_complete_if_cache(
            model="claude-test", prompt="hello", api_key="offline-key"
        )

    assert len(requests) == expected_requests
    assert isinstance(caught.value, APIStatusError)
    assert caught.value.status_code == status
    assert all(client.is_closed() for client in clients)


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", ["ConnectError"])
@pytest.mark.parametrize("stream", [False, True])
async def test_transport_failure_does_not_restart_sdk_retries(
    mock_anthropic_http, error_type, stream
):
    requests = []

    def respond(request):
        requests.append(request)
        raise getattr(sdk_http, error_type)(
            "synthetic connection failure", request=request
        )

    clients = mock_anthropic_http(respond)
    with pytest.raises((APIConnectionError, RetryError)) as caught:
        await anthropic_complete_if_cache(
            model="claude-test",
            prompt="hello",
            api_key="offline-key",
            base_url="https://anthropic.invalid",
            stream=stream,
        )

    assert len(requests) == DEFAULT_ATTEMPTS
    assert isinstance(caught.value, APIConnectionError)
    assert all(client.is_closed() for client in clients)


@pytest.mark.asyncio
@pytest.mark.parametrize("max_retries,expected_requests", [(0, 1), (1, 2)])
async def test_max_retries_overrides_the_sdk_budget(
    mock_anthropic_http, max_retries, expected_requests
):
    requests = []

    def respond(request):
        requests.append(request)
        return _error_response(529)

    clients = mock_anthropic_http(respond)
    with pytest.raises(APIStatusError):
        await anthropic_complete_if_cache(
            model="claude-test",
            prompt="hello",
            api_key="offline-key",
            max_retries=max_retries,
        )

    assert len(requests) == expected_requests
    assert all(client.is_closed() for client in clients)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call_kwargs,expected_max_retries",
    [
        ({"max_retries": None}, anthropic_binding._DEFAULT_MAX_RETRIES),
        ({"max_retries": 0}, 0),
        ({"base_url": "https://proxy.invalid", "max_retries": 7}, 7),
    ],
)
async def test_max_retries_is_a_client_option_not_a_request_parameter(
    call_kwargs, expected_max_retries
):
    message = MagicMock()
    message.content = [MagicMock(type="text", text="hello")]
    message.stop_reason = "end_turn"
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=message)
    client.close = AsyncMock()

    with patch(
        "lightrag.llm.anthropic.AsyncAnthropic", return_value=client
    ) as client_cls:
        result = await anthropic_complete_if_cache(
            model="claude-test", prompt="hello", api_key="offline-key", **call_kwargs
        )

    assert result == "hello"
    assert client_cls.call_args.kwargs["max_retries"] == expected_max_retries
    assert "max_retries" not in client.messages.create.await_args.kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_transient_rate_limit_recovers_within_sdk_budget(
    mock_anthropic_http, stream
):
    requests = []
    message = {
        "id": "msg_offline",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }

    def respond(request):
        requests.append(request)
        if len(requests) < DEFAULT_ATTEMPTS:
            return sdk_http.Response(
                429,
                headers={"retry-after-ms": "1"},
                json={
                    "type": "error",
                    "error": {"type": "rate_limit_error", "message": "retry"},
                },
            )
        if not stream:
            return sdk_http.Response(200, json=message)
        events = [
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hello"},
            },
            {"type": "message_stop"},
        ]
        body = "".join(
            f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
        )
        return sdk_http.Response(
            200, headers={"content-type": "text/event-stream"}, content=body
        )

    clients = mock_anthropic_http(respond)
    result = await anthropic_complete_if_cache(
        model="claude-test", prompt="hello", api_key="offline-key", stream=stream
    )
    if stream:
        result = "".join([chunk async for chunk in result])

    assert result == "hello"
    assert len(requests) == DEFAULT_ATTEMPTS
    assert all(client.is_closed() for client in clients)


# --------------------------------------------------------------------------- #
# Token usage
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_non_streaming_usage_is_tracked_without_reaching_sdk():
    tracker = TokenTracker()
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="answer")],
        stop_reason="end_turn",
        usage=_usage(input_tokens=10, output_tokens=4, cache_creation=3, cache_read=2),
    )
    create = AsyncMock(return_value=response)
    client = SimpleNamespace(messages=SimpleNamespace(create=create), close=AsyncMock())

    with patch("lightrag.llm.anthropic.AsyncAnthropic", return_value=client):
        result = await anthropic_complete_if_cache(
            model="claude-test",
            prompt="hello",
            api_key="test-key",
            token_tracker=tracker,
        )

    assert result == "answer"
    assert tracker.get_usage() == {
        "prompt_tokens": 15,
        "completion_tokens": 4,
        "total_tokens": 19,
        "call_count": 1,
    }
    assert "token_tracker" not in create.await_args.kwargs


def test_image_inputs_keeps_its_position_ahead_of_token_tracker():
    params = list(inspect.signature(anthropic_complete_if_cache).parameters)
    assert params.index("image_inputs") < params.index("token_tracker")


@pytest.mark.asyncio
async def test_usage_recorded_even_when_content_extraction_fails():
    tracker = TokenTracker()
    response = SimpleNamespace(
        content=[],
        stop_reason="end_turn",
        usage=_usage(input_tokens=10, output_tokens=4),
    )
    create = AsyncMock(return_value=response)
    client = SimpleNamespace(messages=SimpleNamespace(create=create), close=AsyncMock())

    with patch("lightrag.llm.anthropic.AsyncAnthropic", return_value=client):
        with pytest.raises(InvalidResponseError):
            await anthropic_complete_if_cache(
                model="claude-test",
                prompt="hello",
                api_key="test-key",
                token_tracker=tracker,
            )

    assert tracker.get_usage() == {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
        "call_count": 1,
    }


@pytest.mark.asyncio
async def test_streaming_usage_is_tracked_after_full_consumption():
    tracker = TokenTracker()
    stream = _EventStream(
        [
            SimpleNamespace(
                message=SimpleNamespace(
                    usage=_usage(
                        input_tokens=8,
                        output_tokens=1,
                        cache_creation=2,
                        cache_read=3,
                    )
                )
            ),
            SimpleNamespace(delta=SimpleNamespace(text="answer")),
            SimpleNamespace(
                delta=SimpleNamespace(),
                usage=_usage(input_tokens=None, output_tokens=5),
            ),
        ]
    )
    create = AsyncMock(return_value=stream)
    client = SimpleNamespace(messages=SimpleNamespace(create=create), close=AsyncMock())

    with patch("lightrag.llm.anthropic.AsyncAnthropic", return_value=client):
        result = await anthropic_complete_if_cache(
            model="claude-test",
            prompt="hello",
            api_key="test-key",
            stream=True,
            token_tracker=tracker,
        )
        chunks = [chunk async for chunk in result]

    assert chunks == ["answer"]
    assert tracker.get_usage() == {
        "prompt_tokens": 13,
        "completion_tokens": 5,
        "total_tokens": 18,
        "call_count": 1,
    }
    assert "token_tracker" not in create.await_args.kwargs
    stream.close.assert_awaited_once()


# --------------------------------------------------------------------------- #
# Truncation & thinking blocks
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_reason", ["max_tokens", "model_context_window_exceeded"])
async def test_anthropic_truncation_stop_reason_marks_result_truncated(stop_reason):
    partial = '{"entities":[{"name":"Ali'
    client = _make_stub_client(text=partial, stop_reason=stop_reason)

    with patch("lightrag.llm.anthropic.AsyncAnthropic", return_value=client):
        result = await anthropic_complete_if_cache(
            model="claude-test", prompt="Extract", api_key="test-key"
        )

    assert result == partial
    assert is_truncated_response(result) is True
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_anthropic_end_turn_and_thinking_block_skipping():
    complete = '{"entities":[]}'
    client = _make_stub_client(text=complete, stop_reason="end_turn")
    with patch("lightrag.llm.anthropic.AsyncAnthropic", return_value=client):
        result = await anthropic_complete_if_cache(
            model="claude-test", prompt="Extract", api_key="test-key"
        )
    assert result == complete
    assert type(result) is str
    assert is_truncated_response(result) is False
    client.close.assert_awaited_once()

    client2 = _make_stub_client(
        content=[
            SimpleNamespace(type="thinking", thinking="step one"),
            SimpleNamespace(type="text", text="Part one. "),
            SimpleNamespace(type="redacted_thinking", data="opaque"),
            SimpleNamespace(type="text", text="Part two."),
        ],
        stop_reason="end_turn",
    )
    with patch("lightrag.llm.anthropic.AsyncAnthropic", return_value=client2):
        result2 = await anthropic_complete_if_cache(
            model="claude-test", prompt="hi", api_key="test-key"
        )
    assert result2 == "Part one. Part two."
    assert type(result2) is str
    client2.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "thinking_block",
    [
        SimpleNamespace(type="thinking", thinking="x" * 50),
        SimpleNamespace(type="redacted_thinking", data="opaque"),
    ],
    ids=["thinking", "redacted_thinking"],
)
async def test_anthropic_thinking_only_max_tokens_raises_empty_truncated(
    thinking_block,
):
    client = _make_stub_client(content=[thinking_block], stop_reason="max_tokens")

    with patch("lightrag.llm.anthropic.AsyncAnthropic", return_value=client):
        with pytest.raises(EmptyTruncatedResponseError) as excinfo:
            await anthropic_complete_if_cache(
                model="claude-test", prompt="hi", api_key="test-key"
            )

    message = str(excinfo.value)
    assert "budget consumed by reasoning" in message
    assert "max_tokens" in message and "thinking.budget_tokens" in message
    client.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stop_reason", "content", "expected_hint", "absent_hint"),
    [
        (
            "model_context_window_exceeded",
            [],
            "shorten the input",
            "raising max_tokens",
        ),
        (
            "model_context_window_exceeded",
            [SimpleNamespace(type="text", text=" ")],
            "shorten the input",
            "raising max_tokens",
        ),
        (
            "max_tokens",
            [SimpleNamespace(type="text", text=" \n")],
            "raising max_tokens",
            "shorten the input",
        ),
    ],
    ids=["context-no-text-block", "context-blank-text", "max-tokens-blank-text"],
)
async def test_anthropic_blank_truncated_response_raises_with_remedy(
    stop_reason, content, expected_hint, absent_hint
):
    client = _make_stub_client(content=content, stop_reason=stop_reason)

    with patch("lightrag.llm.anthropic.AsyncAnthropic", return_value=client):
        with pytest.raises(EmptyTruncatedResponseError) as excinfo:
            await anthropic_complete_if_cache(
                model="claude-test", prompt="Extract", api_key="test-key"
            )

    message = str(excinfo.value)
    assert f"stop_reason={stop_reason}" in message
    assert expected_hint in message
    assert absent_hint not in message
    client.close.assert_awaited_once()


# --------------------------------------------------------------------------- #
# Escaped Unicode streaming through ASGI
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_surrogate_pair_in_anthropic_stream_is_utf8_encodable(monkeypatch):
    event = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": r"Answer: \ud83d\ude00"},
    }
    sse = f"event: content_block_delta\ndata: {json.dumps(event)}\n\n"

    def provider_response(request):
        assert request.url.path == "/v1/messages"
        assert json.loads(request.content)["stream"] is True
        return sdk_http.Response(
            200, headers={"content-type": "text/event-stream"}, content=sse.encode()
        )

    provider_http = sdk_http.AsyncClient(
        transport=sdk_http.MockTransport(provider_response)
    )
    monkeypatch.setattr(
        anthropic_binding,
        "AsyncAnthropic",
        partial(AsyncAnthropic, http_client=provider_http),
    )

    async def answer(request):
        stream = await anthropic_complete_if_cache(
            model="claude-test", prompt="Hello", api_key="test-key", stream=True
        )
        return StreamingResponse(stream, media_type="text/plain")

    app = Starlette(routes=[Route("/", answer)])
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as consumer:
        response = await consumer.get("/")

    assert response.status_code == 200
    assert response.content == "Answer: 😀".encode("utf-8")
    assert provider_http.is_closed
