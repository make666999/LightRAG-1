"""Offline tests for the OpenAI Responses (``/v1/responses``) LLM binding.

The driver is exercised against a fake client so the whole file runs in the
offline CI job with no network and no API key.
"""

from types import SimpleNamespace
from typing import Any

import pytest
from tenacity import wait_none

from lightrag.exceptions import EmptyTruncatedResponseError
from lightrag.llm.openai_responses import (
    InvalidResponseError,
    ResponseRefusedError,
    openai_responses_complete_if_cache,
)
from lightrag.utils import TruncatedResponse, is_truncated_response

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


@pytest.fixture(autouse=True)
def _no_retry_wait(monkeypatch):
    monkeypatch.setattr(openai_responses_complete_if_cache.retry, "wait", wait_none())


class _FakeResponses:
    """Captures the create() call and returns a canned payload."""

    def __init__(self, payload: Any):
        self._payload = payload
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _FakeClient:
    """Minimal AsyncOpenAI stand-in recording close() and responses.create()."""

    def __init__(self, payload: Any):
        self.responses = _FakeResponses(payload)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _message(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="message",
        content=[SimpleNamespace(type="output_text", text=text)],
    )


def _refusal(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="message",
        content=[SimpleNamespace(type="refusal", refusal=text)],
    )


def _reasoning(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="reasoning", summary=[SimpleNamespace(text=text)])


def _response(
    *,
    output: list[Any] | None = None,
    status: str = "completed",
    incomplete_reason: str | None = None,
    usage: Any = None,
    output_text: str | None = None,
) -> SimpleNamespace:
    payload = SimpleNamespace(
        output=output or [],
        status=status,
        incomplete_details=(
            SimpleNamespace(reason=incomplete_reason) if incomplete_reason else None
        ),
        usage=usage,
    )
    if output_text is not None:
        payload.output_text = output_text
    return payload


def _usage(input_tokens: int, output_tokens: int, reasoning_tokens: int = 0):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
    )


def _patch_client(monkeypatch, payload: Any) -> _FakeClient:
    client = _FakeClient(payload)
    monkeypatch.setattr(
        "lightrag.llm.openai_responses.create_openai_async_client",
        lambda **kwargs: client,
    )
    return client


async def _non_streaming_failure(monkeypatch, payload, **kwargs) -> BaseException:
    _patch_client(monkeypatch, payload)
    with pytest.raises(Exception) as excinfo:
        await openai_responses_complete_if_cache("gpt-5.6", "hi", **kwargs)
    return excinfo.value


# --------------------------------------------------------------------------- #
# Request construction & parameter translation
# --------------------------------------------------------------------------- #


async def test_returns_output_text_and_closes_client(monkeypatch):
    client = _patch_client(monkeypatch, _response(output=[_message("hello")]))

    result = await openai_responses_complete_if_cache("gpt-5.6", "hi")

    assert result == "hello"
    assert client.closed


async def test_system_prompt_is_sent_as_instructions(monkeypatch):
    client = _patch_client(monkeypatch, _response(output=[_message("ok")]))

    await openai_responses_complete_if_cache("gpt-5.6", "hi", system_prompt="be terse")

    call = client.responses.calls[0]
    assert call["instructions"] == "be terse"
    assert all(item["role"] != "system" for item in call["input"])


async def test_history_assistant_turn_uses_output_text_part(monkeypatch):
    client = _patch_client(monkeypatch, _response(output=[_message("ok")]))

    await openai_responses_complete_if_cache(
        "gpt-5.6",
        "next",
        history_messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
        ],
    )

    api_input = client.responses.calls[0]["input"]
    assert api_input[0]["content"][0]["type"] == "input_text"
    assert api_input[1]["content"][0]["type"] == "output_text"
    assert api_input[-1]["content"][0]["type"] == "input_text"


@pytest.mark.parametrize(
    "alias", ["max_output_tokens", "max_completion_tokens", "max_tokens"]
)
async def test_output_budget_aliases_map_to_max_output_tokens(monkeypatch, alias):
    client = _patch_client(monkeypatch, _response(output=[_message("ok")]))

    await openai_responses_complete_if_cache("gpt-5.6", "hi", **{alias: 321})

    call = client.responses.calls[0]
    assert call["max_output_tokens"] == 321
    assert "max_completion_tokens" not in call
    assert "max_tokens" not in call


@pytest.mark.parametrize(
    ("kwargs", "expected_reasoning"),
    [
        pytest.param({"reasoning_effort": "low"}, {"effort": "low"}, id="flat-alias"),
        pytest.param(
            {"reasoning": {"effort": "high"}, "reasoning_effort": "low"},
            {"effort": "high"},
            id="explicit-dict-wins",
        ),
    ],
)
async def test_reasoning_param_translation(monkeypatch, kwargs, expected_reasoning):
    client = _patch_client(monkeypatch, _response(output=[_message("ok")]))

    await openai_responses_complete_if_cache("gpt-5.6", "hi", **kwargs)

    call = client.responses.calls[0]
    assert call["reasoning"] == expected_reasoning
    assert "reasoning_effort" not in call


_SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}}


@pytest.mark.parametrize(
    ("response_format", "expected_format"),
    [
        pytest.param(
            {"type": "json_object"}, {"type": "json_object"}, id="json-object"
        ),
        pytest.param(
            {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": _SCHEMA, "strict": True},
            },
            {
                "type": "json_schema",
                "name": "answer",
                "schema": _SCHEMA,
                "strict": True,
            },
            id="nested-json-schema-flattened",
        ),
        pytest.param(
            {"type": "json_schema", "name": "answer", "schema": {"type": "object"}},
            {"type": "json_schema", "name": "answer", "schema": {"type": "object"}},
            id="flat-json-schema-untouched",
        ),
    ],
)
async def test_response_format_translation(
    monkeypatch, response_format, expected_format
):
    client = _patch_client(monkeypatch, _response(output=[_message('{"a":1}')]))

    await openai_responses_complete_if_cache(
        "gpt-5.6", "hi", response_format=dict(response_format)
    )

    call = client.responses.calls[0]
    assert call["text"] == {"format": expected_format}
    assert "response_format" not in call


async def test_unsupported_chat_completions_params_are_dropped(monkeypatch):
    client = _patch_client(monkeypatch, _response(output=[_message("ok")]))

    await openai_responses_complete_if_cache(
        "gpt-5.6", "hi", frequency_penalty=0.5, presence_penalty=0.2, n=2
    )

    call = client.responses.calls[0]
    for dropped in ("frequency_penalty", "presence_penalty", "n"):
        assert dropped not in call


# --------------------------------------------------------------------------- #
# Non-streaming response extraction, COT, usage, truncation, and errors
# --------------------------------------------------------------------------- #


async def test_reasoning_only_body_raises_retryable_even_with_cot(monkeypatch):
    client = _patch_client(
        monkeypatch,
        _response(output=[_reasoning("thinking hard"), _message("")]),
    )

    with pytest.raises(InvalidResponseError) as excinfo:
        await openai_responses_complete_if_cache("gpt-5.6", "hi", enable_cot=True)

    assert not isinstance(excinfo.value, EmptyTruncatedResponseError)
    assert "reasoning-only output" in str(excinfo.value)
    assert "thinking hard" not in str(excinfo.value)
    assert len(client.responses.calls) == 3


@pytest.mark.parametrize(
    ("body", "extra_kwargs"),
    [
        pytest.param("answer", {}, id="plain-body"),
        pytest.param(
            '{"a": 1}', {"response_format": {"type": "json_object"}}, id="structured"
        ),
    ],
)
async def test_cot_dropped_when_body_text_present(monkeypatch, body, extra_kwargs):
    _patch_client(
        monkeypatch,
        _response(output=[_reasoning("trace"), _message(body)]),
    )

    result = await openai_responses_complete_if_cache(
        "gpt-5.6", "hi", enable_cot=True, **extra_kwargs
    )

    assert result == body


async def test_token_tracker_maps_responses_usage_keys(monkeypatch):
    _patch_client(
        monkeypatch,
        _response(output=[_message("ok")], usage=_usage(11, 7)),
    )
    recorded: list[dict[str, int]] = []
    tracker = SimpleNamespace(add_usage=recorded.append)

    await openai_responses_complete_if_cache("gpt-5.6", "hi", token_tracker=tracker)

    assert recorded == [
        {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    ]


async def test_truncated_output_is_wrapped_for_the_cache_layer(monkeypatch):
    _patch_client(
        monkeypatch,
        _response(
            output=[_message('{"partial": ')],
            status="incomplete",
            incomplete_reason="max_output_tokens",
        ),
    )

    result = await openai_responses_complete_if_cache("gpt-5.6", "hi")

    assert isinstance(result, TruncatedResponse)
    assert is_truncated_response(result)
    assert str(result) == '{"partial": '


async def test_empty_truncated_output_raises_non_retryable_and_counts_usage(
    monkeypatch,
):
    client = _patch_client(
        monkeypatch,
        _response(
            output=[],
            status="incomplete",
            incomplete_reason="max_output_tokens",
            usage=_usage(9, 400, reasoning_tokens=400),
        ),
    )
    recorded: list[dict[str, int]] = []
    tracker = SimpleNamespace(add_usage=recorded.append)

    with pytest.raises(EmptyTruncatedResponseError) as excinfo:
        await openai_responses_complete_if_cache("gpt-5.6", "hi", token_tracker=tracker)

    assert "max_output_tokens" in str(excinfo.value)
    assert len(client.responses.calls) == 1
    assert recorded == [
        {"prompt_tokens": 9, "completion_tokens": 400, "total_tokens": 409}
    ]


async def test_empty_output_without_truncation_raises_retryable(monkeypatch):
    _patch_client(monkeypatch, _response(output=[_message("")]))

    with pytest.raises(InvalidResponseError) as excinfo:
        await openai_responses_complete_if_cache("gpt-5.6", "hi")

    assert not isinstance(excinfo.value, EmptyTruncatedResponseError)


async def test_output_text_property_is_preferred(monkeypatch):
    _patch_client(
        monkeypatch,
        _response(output=[_message("from-walk")], output_text="from-property"),
    )

    assert await openai_responses_complete_if_cache("gpt-5.6", "hi") == (
        "from-property"
    )


@pytest.mark.parametrize(
    "output",
    [
        pytest.param([], id="empty-output"),
        pytest.param([_message("partial")], id="partial-output-not-masked"),
    ],
)
async def test_non_streaming_failed_status_raises_and_counts_usage(monkeypatch, output):
    payload = _response(output=output, status="failed", usage=_usage(5, 2))
    payload.error = SimpleNamespace(code="server_error", message="upstream exploded")
    recorded: list[dict[str, int]] = []
    tracker = SimpleNamespace(add_usage=recorded.append)

    error = await _non_streaming_failure(monkeypatch, payload, token_tracker=tracker)

    assert isinstance(error, InvalidResponseError)
    assert "server_error" in str(error)
    assert "upstream exploded" in str(error)
    assert recorded
    assert all(
        counts == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        for counts in recorded
    )


async def test_refusal_content_part_raises_rather_than_reporting_empty_output(
    monkeypatch,
):
    payload = _response(output=[_refusal("I will not answer that")])

    error = await _non_streaming_failure(monkeypatch, payload)

    assert isinstance(error, ResponseRefusedError)
    assert "I will not answer that" in str(error)


async def test_non_budget_incomplete_status_raises(monkeypatch):
    payload = _response(
        output=[_message("partial")],
        status="incomplete",
        incomplete_reason="content_filter",
    )

    error = await _non_streaming_failure(monkeypatch, payload)

    assert isinstance(error, InvalidResponseError)
    assert "content_filter" in str(error)


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #


class _FakeStream:
    """Async iterable yielding stub Responses API stream events."""

    def __init__(self, events: list[Any]):
        self._events = events
        self.aclosed = False

    def __aiter__(self):
        async def gen():
            for event in self._events:
                yield event

        return gen()

    async def aclose(self) -> None:
        self.aclosed = True


def _delta(event_type: str, delta: str) -> SimpleNamespace:
    return SimpleNamespace(type=event_type, delta=delta)


def _completed(usage: Any = None) -> SimpleNamespace:
    return SimpleNamespace(
        type="response.completed",
        response=SimpleNamespace(status="completed", usage=usage),
    )


def _incomplete(reason: str, usage: Any = None) -> SimpleNamespace:
    return SimpleNamespace(
        type="response.incomplete",
        response=SimpleNamespace(
            status="incomplete",
            incomplete_details=SimpleNamespace(reason=reason),
            usage=usage,
        ),
    )


async def _collect(agen) -> str:
    return "".join([chunk async for chunk in agen])


async def test_streaming_yields_output_text_deltas(monkeypatch):
    stream = _FakeStream(
        [
            _delta("response.output_text.delta", "hel"),
            _delta("response.output_text.delta", "lo"),
            _completed(),
        ]
    )
    client = _patch_client(monkeypatch, stream)

    result = await openai_responses_complete_if_cache("gpt-5.6", "hi", stream=True)

    assert await _collect(result) == "hello"
    assert stream.aclosed
    assert client.closed


@pytest.mark.parametrize(
    ("events", "enable_cot", "expected"),
    [
        pytest.param(
            [
                _delta("response.reasoning_summary_text.delta", "step one"),
                _delta("response.output_text.delta", "answer"),
                _completed(),
            ],
            True,
            "<think>step one</think>answer",
            id="reasoning-then-body",
        ),
        pytest.param(
            [
                _delta("response.reasoning_summary_text.delta", "only thinking"),
                _completed(_usage(3, 4)),
            ],
            True,
            "<think>only thinking</think>",
            id="reasoning-only-closes-tag",
        ),
        pytest.param(
            [
                _delta("response.reasoning_summary_text.delta", "trace"),
                _delta("response.output_text.delta", "answer"),
                _completed(),
            ],
            False,
            "answer",
            id="cot-disabled-drops-reasoning",
        ),
    ],
)
async def test_streaming_cot_wrapping_and_filtering(
    monkeypatch, events, enable_cot, expected
):
    _patch_client(monkeypatch, _FakeStream(events))

    result = await openai_responses_complete_if_cache(
        "gpt-5.6", "hi", stream=True, enable_cot=enable_cot
    )

    assert await _collect(result) == expected


@pytest.mark.parametrize(
    ("events", "expected_text"),
    [
        pytest.param(
            [_delta("response.output_text.delta", "ok"), _completed(_usage(3, 4))],
            "ok",
            id="completed-event",
        ),
        pytest.param(
            [
                _delta("response.output_text.delta", "partial"),
                _incomplete("max_output_tokens", _usage(3, 4)),
            ],
            "partial",
            id="incomplete-budget-event",
        ),
    ],
)
async def test_streaming_records_usage_from_terminal_event(
    monkeypatch, events, expected_text
):
    _patch_client(monkeypatch, _FakeStream(events))
    recorded: list[dict[str, int]] = []
    tracker = SimpleNamespace(add_usage=recorded.append)

    result = await openai_responses_complete_if_cache(
        "gpt-5.6", "hi", stream=True, token_tracker=tracker
    )

    assert await _collect(result) == expected_text
    assert recorded == [{"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}]


async def test_streaming_empty_truncation_raises_non_retryable(monkeypatch):
    _patch_client(
        monkeypatch,
        _FakeStream(
            [_incomplete("max_output_tokens", _usage(3, 120, reasoning_tokens=120))]
        ),
    )

    result = await openai_responses_complete_if_cache("gpt-5.6", "hi", stream=True)

    with pytest.raises(EmptyTruncatedResponseError):
        await _collect(result)


@pytest.mark.parametrize(
    ("prefix_events", "expected_partial"),
    [
        pytest.param([], "", id="no-text"),
        pytest.param(
            [_delta("response.output_text.delta", "par")],
            "par",
            id="partial-text-not-masked",
        ),
    ],
)
async def test_streaming_non_budget_incomplete_raises_and_counts_usage(
    monkeypatch, prefix_events, expected_partial
):
    _patch_client(
        monkeypatch,
        _FakeStream([*prefix_events, _incomplete("content_filter", _usage(4, 1))]),
    )
    recorded: list[dict[str, int]] = []
    result = await openai_responses_complete_if_cache(
        "gpt-5.6",
        "hi",
        stream=True,
        token_tracker=SimpleNamespace(add_usage=recorded.append),
    )

    chunks: list[str] = []
    with pytest.raises(InvalidResponseError) as excinfo:
        async for chunk in result:
            chunks.append(chunk)

    assert "content_filter" in str(excinfo.value)
    assert "".join(chunks) == expected_partial
    assert recorded == [{"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}]


async def test_streaming_unreadable_incomplete_terminal_raises(monkeypatch):
    _patch_client(
        monkeypatch,
        _FakeStream([SimpleNamespace(type="response.incomplete", response=None)]),
    )

    result = await openai_responses_complete_if_cache("gpt-5.6", "hi", stream=True)

    with pytest.raises(InvalidResponseError) as excinfo:
        await _collect(result)
    assert "unknown" in str(excinfo.value)


@pytest.mark.parametrize(
    ("events", "expected_text"),
    [
        pytest.param(
            [
                _delta("response.refusal.delta", "I cannot "),
                _delta("response.refusal.delta", "help with that"),
                _completed(_usage(6, 4)),
            ],
            "I cannot help with that",
            id="refusal-deltas",
        ),
        pytest.param(
            [
                SimpleNamespace(type="response.refusal.done", refusal="policy: no"),
                _completed(),
            ],
            "policy: no",
            id="refusal-done-without-deltas",
        ),
        pytest.param(
            [
                _delta("response.refusal.delta", ""),
                SimpleNamespace(type="response.refusal.done", refusal="policy: no"),
                _completed(),
            ],
            "policy: no",
            id="empty-delta-keeps-done-fallback",
        ),
        pytest.param(
            [_delta("response.refusal.delta", ""), _completed()],
            None,
            id="textless-refusal",
        ),
        pytest.param(
            [
                SimpleNamespace(
                    type="response.completed",
                    response=_response(output=[_refusal("I will not answer that")]),
                )
            ],
            "I will not answer that",
            id="terminal-content-part-fallback",
        ),
        pytest.param(
            [
                _delta("response.refusal.delta", "no can do"),
                SimpleNamespace(type="response.refusal.done", refusal="no can do"),
                _completed(),
            ],
            "no can do",
            id="done-does-not-duplicate-deltas",
        ),
    ],
)
async def test_streaming_refusal_raises_response_refused_error(
    monkeypatch, events, expected_text
):
    _patch_client(monkeypatch, _FakeStream(events))

    result = await openai_responses_complete_if_cache("gpt-5.6", "hi", stream=True)

    with pytest.raises(ResponseRefusedError) as excinfo:
        await _collect(result)
    if expected_text is not None:
        assert str(excinfo.value).count(expected_text) == 1


async def test_streaming_reasoning_only_truncation_is_not_treated_as_empty(monkeypatch):
    _patch_client(
        monkeypatch,
        _FakeStream(
            [
                _delta("response.reasoning_summary_text.delta", "thinking"),
                _incomplete("max_output_tokens", _usage(3, 9, reasoning_tokens=9)),
            ]
        ),
    )

    result = await openai_responses_complete_if_cache(
        "gpt-5.6", "hi", stream=True, enable_cot=True
    )

    assert await _collect(result) == "<think>thinking</think>"


async def test_streaming_failed_terminal_raises_and_counts_usage(monkeypatch):
    _patch_client(
        monkeypatch,
        _FakeStream(
            [
                _delta("response.output_text.delta", "partial"),
                SimpleNamespace(
                    type="response.failed",
                    response=SimpleNamespace(
                        status="failed",
                        error=SimpleNamespace(
                            code="server_error", message="upstream exploded"
                        ),
                        usage=_usage(3, 4),
                    ),
                ),
            ]
        ),
    )
    recorded: list[dict[str, int]] = []
    result = await openai_responses_complete_if_cache(
        "gpt-5.6",
        "hi",
        stream=True,
        token_tracker=SimpleNamespace(add_usage=recorded.append),
    )

    chunks: list[str] = []
    with pytest.raises(InvalidResponseError) as excinfo:
        async for chunk in result:
            chunks.append(chunk)

    assert "".join(chunks) == "partial"
    assert "server_error" in str(excinfo.value)
    assert "upstream exploded" in str(excinfo.value)
    assert recorded == [{"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}]


async def test_streaming_failed_terminal_closes_an_open_think_tag(monkeypatch):
    _patch_client(
        monkeypatch,
        _FakeStream(
            [
                _delta("response.reasoning_summary_text.delta", "thinking"),
                SimpleNamespace(
                    type="response.failed",
                    response=SimpleNamespace(
                        status="failed",
                        error=SimpleNamespace(code="server_error", message="boom"),
                        usage=None,
                    ),
                ),
            ]
        ),
    )

    result = await openai_responses_complete_if_cache(
        "gpt-5.6", "hi", stream=True, enable_cot=True
    )

    chunks: list[str] = []
    with pytest.raises(InvalidResponseError):
        async for chunk in result:
            chunks.append(chunk)
    assert "".join(chunks) == "<think>thinking</think>"


async def test_streaming_completed_with_no_text_raises_like_non_streaming(monkeypatch):
    _patch_client(monkeypatch, _FakeStream([_completed(_usage(3, 0))]))

    result = await openai_responses_complete_if_cache("gpt-5.6", "hi", stream=True)
    with pytest.raises(InvalidResponseError) as excinfo:
        await _collect(result)

    assert not isinstance(excinfo.value, EmptyTruncatedResponseError)
