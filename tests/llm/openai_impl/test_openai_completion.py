"""OpenAI completion response handling: cached prompt tokens, invalid gateway responses, truncation, and reasoning diagnostics."""

import json
import logging
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import AsyncOpenAI
from tenacity import wait_none

import lightrag.llm.openai as openai_module
from lightrag.exceptions import EmptyTruncatedResponseError
from lightrag.llm.openai import (
    InvalidResponseError,
    azure_openai_complete_if_cache,
    openai_complete_if_cache,
)
from lightrag.utils import TokenTracker, is_truncated_response

pytestmark = pytest.mark.offline


# --------------------------------------------------------------------------- #
# Cached prompt tokens reporting
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stream", "cached_tokens"),
    [
        pytest.param(False, None, id="absent"),
        pytest.param(False, 0, id="reported-zero"),
        pytest.param(False, 2, id="reported"),
        pytest.param(True, 2, id="stream-reported"),
    ],
)
async def test_openai_chat_reports_cached_prompt_tokens(
    monkeypatch, stream, cached_tokens
):
    usage = {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}
    if cached_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    response = {"id": "completion", "model": "test-model", "created": 0}

    def respond(request):
        if not stream:
            return httpx.Response(
                200,
                json={
                    **response,
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "answer"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": usage,
                },
            )
        chunk = {**response, "object": "chat.completion.chunk"}
        events = [
            {
                **chunk,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "answer"},
                        "finish_reason": "stop",
                    }
                ],
            },
            {**chunk, "choices": [], "usage": usage},
        ]
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text="".join(f"data: {json.dumps(event)}\n\n" for event in events)
            + "data: [DONE]\n\n",
        )

    tracker = TokenTracker()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        sdk = AsyncOpenAI(
            api_key="placeholder",
            base_url="https://example.invalid/v1",
            http_client=http_client,
        )
        monkeypatch.setattr(
            openai_module, "create_openai_async_client", lambda **_: sdk
        )
        result = await openai_complete_if_cache(
            "test-model",
            "extract entities",
            stream=stream,
            token_tracker=tracker,
        )
        if stream:
            result = "".join([part async for part in result])

    assert result == "answer"
    counts = tracker.get_usage()
    assert counts["prompt_tokens"] == 4
    assert counts["total_tokens"] == 5
    if cached_tokens is None:
        assert "cached_prompt_tokens" not in counts
    else:
        assert counts["cached_prompt_tokens"] == cached_tokens


# --------------------------------------------------------------------------- #
# Invalid gateway responses
# --------------------------------------------------------------------------- #


def _gateway_completion() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "synthetic-completion",
            "object": "chat.completion",
            "created": 0,
            "model": "synthetic-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Recovered"},
                    "finish_reason": "stop",
                }
            ],
        },
    )


def _mock_gateway(
    monkeypatch: pytest.MonkeyPatch, responses: list[httpx.Response]
) -> tuple[list[httpx.Request], list[AsyncOpenAI]]:
    requests: list[httpx.Request] = []
    clients: list[AsyncOpenAI] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert responses, "Unexpected extra gateway request"
        return responses.pop(0)

    def create_client(**_kwargs: object) -> AsyncOpenAI:
        client = AsyncOpenAI(
            api_key="synthetic-not-a-credential",
            base_url="https://synthetic.invalid/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        )
        clients.append(client)
        return client

    monkeypatch.setattr(openai_module, "create_openai_async_client", create_client)
    monkeypatch.setattr(openai_complete_if_cache.retry, "wait", wait_none())
    return requests, clients


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_response",
    [
        pytest.param(
            lambda: httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text="<html><body>Synthetic gateway error</body></html>",
            ),
            id="html",
        ),
        pytest.param(
            lambda: httpx.Response(200, json=["unexpected", "array"]),
            id="json-array",
        ),
        pytest.param(
            lambda: httpx.Response(200, json={"choices": []}),
            id="empty-choices-control",
        ),
    ],
)
async def test_invalid_gateway_response_retries_then_recovers(
    monkeypatch: pytest.MonkeyPatch,
    invalid_response: Callable[[], httpx.Response],
) -> None:
    requests, clients = _mock_gateway(
        monkeypatch, [invalid_response(), _gateway_completion()]
    )

    result = await openai_complete_if_cache(
        model="synthetic-model", prompt="Synthetic chunk"
    )

    assert result == "Recovered"
    assert len(requests) == 2
    assert all(client.is_closed() for client in clients)


@pytest.mark.asyncio
async def test_persistent_html_response_exhausts_existing_retry_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests, clients = _mock_gateway(
        monkeypatch,
        [
            httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text="<html>Unavailable</html>",
            )
            for _ in range(3)
        ],
    )

    with pytest.raises(InvalidResponseError) as caught:
        await openai_complete_if_cache(
            model="synthetic-model", prompt="Synthetic chunk"
        )

    assert "got str" in str(caught.value)
    assert "<html>Unavailable</html>" in str(caught.value)
    assert len(requests) == 3
    assert all(client.is_closed() for client in clients)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        pytest.param(None, "", id="none"),
        pytest.param(SimpleNamespace(choices=[]), "", id="completion-no-choices"),
        pytest.param(
            "<html>\n  <body>Bad   gateway</body>\n</html>",
            ": expected a chat completion, got str: "
            "'<html> <body>Bad gateway</body> </html>'",
            id="html-whitespace-collapsed",
        ),
        pytest.param(
            ["unexpected", "array"],
            ": expected a chat completion, got list: \"['unexpected', 'array']\"",
            id="json-array",
        ),
        pytest.param(
            "", ": expected a chat completion, got str: ''", id="empty-string"
        ),
    ],
)
def test_describe_invalid_response(response: object, expected: str) -> None:
    assert openai_module._describe_invalid_response(response) == expected


def test_describe_invalid_response_truncates_long_bodies() -> None:
    limit = openai_module._INVALID_RESPONSE_SNIPPET_CHARS
    detail = openai_module._describe_invalid_response("x" * (limit * 10))
    assert detail.endswith("'" + "x" * limit + "...'")


# --------------------------------------------------------------------------- #
# finish_reason, structured outputs, and empty-content diagnostics
# --------------------------------------------------------------------------- #


def _make_completion(
    content: str,
    finish_reason: str = "stop",
    reasoning_content: str = "",
    reasoning_tokens: int | None = None,
):
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    if reasoning_tokens is not None:
        usage.completion_tokens_details = SimpleNamespace(
            reasoning_tokens=reasoning_tokens
        )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content=content,
                    parsed=None,
                    reasoning_content=reasoning_content,
                ),
            )
        ],
        usage=usage,
    )


def _make_fake_client(completion):
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=completion))
        ),
        close=AsyncMock(),
    )


class _FakeAsyncStream:
    """Async iterator yielding stub chat completion stream chunks."""

    def __init__(self, chunks):
        self._chunks = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration

    async def aclose(self):
        return None


def _make_stream_chunk(content=None, reasoning_content=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=content, reasoning_content=reasoning_content
                )
            )
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("finish_reason", "truncated"),
    [("length", True), ("stop", False)],
)
async def test_finish_reason_sets_truncation_marker(finish_reason, truncated):
    raw_json = '{"entities":[{"name":"Alice","type":"Person"'
    fake_client = _make_fake_client(
        _make_completion(raw_json, finish_reason=finish_reason)
    )

    with patch(
        "lightrag.llm.openai.create_openai_async_client", return_value=fake_client
    ):
        result = await openai_complete_if_cache(
            model="test-model",
            prompt="Extract entities",
            response_format={"type": "json_object"},
        )

    assert result == raw_json
    assert is_truncated_response(result) is truncated
    fake_client.chat.completions.create.assert_awaited_once()
    assert fake_client.chat.completions.create.await_args.kwargs["response_format"] == {
        "type": "json_object"
    }
    fake_client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_azure_length_finish_reason_marks_result_truncated():
    raw_json = '{"entities":[{"name":"Alice","type":"Person"'
    fake_client = _make_fake_client(_make_completion(raw_json, finish_reason="length"))

    with patch(
        "lightrag.llm.openai.create_openai_async_client", return_value=fake_client
    ):
        result = await azure_openai_complete_if_cache(
            model="test-deployment",
            prompt="Extract entities",
            response_format={"type": "json_object"},
        )

    assert result == raw_json
    assert is_truncated_response(result) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", ["entity_extraction", "keyword_extraction"])
async def test_legacy_extraction_flag_emits_deprecation_warning(flag):
    fake_client = _make_fake_client(
        _make_completion('{"entities":[],"relationships":[]}')
    )

    with patch(
        "lightrag.llm.openai.create_openai_async_client", return_value=fake_client
    ):
        with pytest.warns(DeprecationWarning):
            await openai_complete_if_cache(
                model="test-model", prompt="Extract", **{flag: True}
            )

    fake_client.chat.completions.create.assert_awaited_once()
    assert fake_client.chat.completions.create.await_args.kwargs["response_format"] == {
        "type": "json_object"
    }


@pytest.mark.asyncio
async def test_typed_response_format_is_rejected():
    fake_client = _make_fake_client(_make_completion("{}"))

    class FakeSchemaModel:
        """Stand-in typed response schema class."""

    with patch(
        "lightrag.llm.openai.create_openai_async_client", return_value=fake_client
    ):
        with pytest.raises(TypeError, match="typed/Pydantic"):
            await openai_complete_if_cache(
                model="test-model",
                prompt="Extract entities",
                response_format=FakeSchemaModel,
            )

    fake_client.chat.completions.create.assert_not_awaited()
    fake_client.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_streaming_structured_output_disables_cot():
    fake_stream = _FakeAsyncStream(
        [
            _make_stream_chunk(reasoning_content="this should not be included"),
            _make_stream_chunk(content='{"answer":"ok"}'),
        ]
    )
    fake_client = _make_fake_client(fake_stream)

    with patch(
        "lightrag.llm.openai.create_openai_async_client", return_value=fake_client
    ):
        stream = await openai_complete_if_cache(
            model="test-model",
            prompt="Extract entities",
            stream=True,
            enable_cot=True,
            response_format={"type": "json_object"},
        )
        chunks = [chunk async for chunk in stream]

    assert "".join(chunks) == '{"answer":"ok"}'
    fake_client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_empty_content_reasoning_only_diagnostics(lightrag_caplog):
    reasoning_text = "thinking about the diagram..."
    fake_client = _make_fake_client(
        _make_completion(
            "",
            finish_reason="stop",
            reasoning_content=reasoning_text,
            reasoning_tokens=800,
        )
    )

    with lightrag_caplog.at_level(logging.ERROR):
        with patch(
            "lightrag.llm.openai.create_openai_async_client", return_value=fake_client
        ):
            with pytest.raises(InvalidResponseError) as excinfo:
                await openai_complete_if_cache.__wrapped__(
                    model="test-model",
                    prompt="Describe the image",
                    response_format={"type": "json_object"},
                )

    message = str(excinfo.value)
    assert "finish_reason=stop" in message
    assert "reasoning_tokens=800" in message
    assert f"reasoning_content_len={len(reasoning_text)}" in message
    assert "reasoning-only" in lightrag_caplog.text
    fake_client.close.assert_awaited()


@pytest.mark.asyncio
async def test_empty_content_length_truncation_diagnostics(lightrag_caplog):
    fake_client = _make_fake_client(_make_completion("", finish_reason="length"))
    tracked: list[dict] = []
    tracker = SimpleNamespace(add_usage=tracked.append)

    with lightrag_caplog.at_level(logging.ERROR):
        with patch(
            "lightrag.llm.openai.create_openai_async_client", return_value=fake_client
        ):
            with pytest.raises(EmptyTruncatedResponseError) as excinfo:
                await openai_complete_if_cache(
                    model="test-model",
                    prompt="Describe the image",
                    response_format={"type": "json_object"},
                    token_tracker=tracker,
                )

    assert fake_client.chat.completions.create.await_count == 1
    message = str(excinfo.value)
    assert "finish_reason=length" in message
    assert "reasoning_tokens=n/a" in message
    assert "reasoning_content_len=0" in message
    assert "hit the token limit" in lightrag_caplog.text
    assert "hit the token limit" in message
    assert "consider raising max_tokens" in message
    fake_client.close.assert_awaited()
    assert tracked == [
        {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    ]
