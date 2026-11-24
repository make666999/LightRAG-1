"""Ollama chat/completion tests: client cleanup, max_tokens alias, response_format, retries, think option, token usage, and truncation."""

import copy
import json
import logging
import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np
import ollama
import pydantic
import pytest
from tenacity import wait_none

import lightrag.llm.ollama as ollama_binding
from lightrag.llm.ollama import (
    InvalidResponseError,
    _ollama_model_if_cache,
    _ollama_usage_counts,
    ollama_embed,
)
from lightrag.utils import TokenTracker, is_truncated_response

pytestmark = pytest.mark.offline


def _make_fake_client(response=None):
    if response is None:
        response = {"message": {"content": "ok"}, "done_reason": "stop"}
    return SimpleNamespace(
        chat=AsyncMock(return_value=response),
        _client=SimpleNamespace(aclose=AsyncMock()),
    )


async def _call_chat(response=None, *, prompt="hello", **kwargs):
    fake_client = _make_fake_client(response)
    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        result = await _ollama_model_if_cache(
            model="test-model", prompt=prompt, **kwargs
        )
    return result, fake_client


# --------------------------------------------------------------------------- #
# Client cleanup on failure
# --------------------------------------------------------------------------- #


async def _cleanup_chat(**kwargs):
    return await _ollama_model_if_cache(model="test-model", prompt="hello", **kwargs)


async def _cleanup_chat_stream_consumed():
    stream = await _cleanup_chat(stream=True)
    return [part async for part in stream]


async def _cleanup_chat_stream_setup_fails():
    return await _ollama_model_if_cache(model="", prompt="hello", stream=True)


async def _cleanup_embed():
    return await ollama_embed.func(["hello"], embed_model="test-model")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "error", "paths"),
    [
        pytest.param(_cleanup_chat, ollama.ResponseError, ["/api/chat"], id="chat"),
        pytest.param(
            _cleanup_chat_stream_consumed,
            ollama.ResponseError,
            ["/api/chat"],
            id="stream",
        ),
        pytest.param(
            _cleanup_chat_stream_setup_fails,
            pydantic.ValidationError,
            [],
            id="stream-setup",
        ),
        pytest.param(_cleanup_embed, ollama.ResponseError, ["/api/embed"], id="embed"),
    ],
)
async def test_request_failure_closes_owned_client_once(
    monkeypatch, operation, error, paths
):
    requested = []
    close_calls = []
    create = ollama.AsyncClient

    def handle(request):
        requested.append(request.url.path)
        return httpx.Response(400, json={"error": "bad request"})

    def client_factory(**kwargs):
        client = create(transport=httpx.MockTransport(handle), **kwargs)
        close = client._client.aclose

        async def close_client():
            close_calls.append(client)
            await close()

        client._client.aclose = close_client
        return client

    monkeypatch.setattr(ollama, "AsyncClient", client_factory)

    with pytest.raises(error):
        await operation()

    assert requested == paths
    assert len(close_calls) == 1


# --------------------------------------------------------------------------- #
# max_tokens -> options.num_predict alias
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("generation_kwargs", "expected_options"),
    [
        pytest.param(
            {"max_tokens": 37},
            {"num_predict": 37},
            id="generic-limit-maps-to-num-predict",
        ),
        pytest.param(
            {"max_tokens": 37, "options": {"temperature": 0.2}},
            {"temperature": 0.2, "num_predict": 37},
            id="generic-limit-preserves-other-options",
        ),
        pytest.param(
            {"options": {"num_predict": 41}},
            {"num_predict": 41},
            id="native-limit-is-preserved",
        ),
        pytest.param(
            {"max_tokens": 37, "options": {"num_predict": 41}},
            {"num_predict": 41},
            id="native-limit-takes-precedence",
        ),
    ],
)
async def test_max_tokens_alias_precedence(generation_kwargs, expected_options):
    caller_kwargs = copy.deepcopy(generation_kwargs)
    result, fake_client = await _call_chat(**caller_kwargs)

    assert result == "ok"
    assert fake_client.chat.call_args.kwargs.get("options") == expected_options
    assert caller_kwargs == generation_kwargs


@pytest.mark.asyncio
async def test_ollama_options_object_alias_copies_and_respects_native():
    shared = ollama_binding.ollama.Options(temperature=0.2)
    _, fake_client = await _call_chat(max_tokens=37, options=shared)
    sent = fake_client.chat.call_args.kwargs.get("options")
    assert sent.num_predict == 37
    assert sent.temperature == 0.2
    assert shared.num_predict is None

    native = ollama_binding.ollama.Options(num_predict=41)
    _, fake_client2 = await _call_chat(max_tokens=37, options=native)
    sent2 = fake_client2.chat.call_args.kwargs.get("options")
    assert sent2 is native
    assert sent2.num_predict == 41


# --------------------------------------------------------------------------- #
# response_format & deprecated extraction flags
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["direct", "wrapper"])
@pytest.mark.parametrize(
    ("flags", "response_format", "warned_flag"),
    [
        pytest.param(
            {"entity_extraction": True, "keyword_extraction": False},
            None,
            "entity",
            id="entity",
        ),
        pytest.param({"keyword_extraction": True}, None, "keyword", id="keyword"),
        pytest.param(
            {"entity_extraction": True, "keyword_extraction": True},
            None,
            "entity",
            id="both",
        ),
        pytest.param(
            {"entity_extraction": True, "keyword_extraction": True},
            {"type": "json_object"},
            None,
            id="both-explicit-format",
        ),
    ],
)
async def test_extraction_flags_use_real_ollama_signature(
    monkeypatch, entry, flags, response_format, warned_flag
):
    requests = []
    clients = []
    real_client = ollama_binding.ollama.AsyncClient

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "message": {"role": "assistant", "content": "{}"},
                "done": True,
                "done_reason": "stop",
            },
        )

    def client_factory(**kwargs):
        client = real_client(transport=httpx.MockTransport(handle), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(ollama_binding.ollama, "AsyncClient", client_factory)
    kwargs = {
        **flags,
        "response_format": response_format,
        "host": "http://ollama.test",
    }
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always", DeprecationWarning)
        if entry == "direct":
            result = await ollama_binding._ollama_model_if_cache(
                "test-model", "hello", **kwargs
            )
        else:
            cache = SimpleNamespace(global_config={"llm_model_name": "test-model"})
            result = await ollama_binding.ollama_model_complete(
                "hello", hashing_kv=cache, **kwargs
            )

    assert result == "{}"
    assert len(requests) == 1
    assert requests[0]["format"] == "json"
    assert "entity_extraction" not in requests[0]
    assert "keyword_extraction" not in requests[0]
    legacy_warnings = [
        str(notice.message)
        for notice in notices
        if issubclass(notice.category, DeprecationWarning)
        and "extraction=" in str(notice.message)
    ]
    if warned_flag is None:
        assert legacy_warnings == []
    else:
        assert len(legacy_warnings) == 1
        assert f"{warned_flag}_extraction=True" in legacy_warnings[0]
    assert len(clients) == 1
    assert clients[0]._client.is_closed


_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


@pytest.mark.parametrize(
    ("kwargs", "expected_format"),
    [
        pytest.param(
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "answer_payload", "schema": _SCHEMA},
                }
            },
            _SCHEMA,
            id="json-schema-unwrapped",
        ),
        pytest.param(
            {"response_format": {"type": "json_schema", "json_schema": _SCHEMA}},
            _SCHEMA,
            id="json-schema-without-schema-key",
        ),
        pytest.param({"response_format": _SCHEMA}, _SCHEMA, id="raw-schema"),
        pytest.param(
            {"format": "json", "response_format": {"type": "json_schema"}},
            "json",
            id="native-format-wins",
        ),
    ],
)
def test_response_format_maps_to_native_format(kwargs, expected_format):
    ollama_binding._normalize_ollama_response_format(kwargs)
    assert kwargs == {"format": expected_format}


# --------------------------------------------------------------------------- #
# Retry classification (chat and embed)
# --------------------------------------------------------------------------- #

_CONNECTION_ERROR_MESSAGE = (
    "Failed to connect to Ollama. Please check that Ollama is downloaded, "
    "running and accessible. https://ollama.com/download"
)
_EOF_ERROR_TEXT = "do embedding request: Post http://127.0.0.1:50830/embedding: EOF"


class TestOllamaRetry:
    @pytest.fixture(autouse=True)
    def _no_retry_wait(self, monkeypatch):
        monkeypatch.setattr(_ollama_model_if_cache.retry, "wait", wait_none())
        monkeypatch.setattr(ollama_embed.func.retry, "wait", wait_none())

    @staticmethod
    def _make_flaky_client(error: BaseException, *, failures: int):
        calls = {"count": 0}

        async def _attempt(result):
            calls["count"] += 1
            if calls["count"] <= failures:
                raise error
            return result

        async def chat(*args, **kwargs):
            return await _attempt({"message": {"content": "ok"}, "done_reason": "stop"})

        async def embed(*args, **kwargs):
            return await _attempt({"embeddings": [[0.1, 0.2, 0.3]]})

        client = SimpleNamespace(
            chat=chat, embed=embed, _client=SimpleNamespace(aclose=AsyncMock())
        )
        return client, calls

    @staticmethod
    async def _run(entry, fake_client):
        with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
            if entry == "chat":
                return await _ollama_model_if_cache(model="test-model", prompt="hello")
            return await ollama_embed.func(["hello"], embed_model="bge-m3:latest")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("entry", "error"),
        [
            pytest.param(
                "chat", ConnectionError(_CONNECTION_ERROR_MESSAGE), id="chat-connect"
            ),
            pytest.param(
                "chat", httpx.ReadTimeout("timed out"), id="chat-read-timeout"
            ),
            pytest.param(
                "chat", httpx.ReadError("connection reset"), id="chat-network"
            ),
            pytest.param(
                "chat",
                httpx.RemoteProtocolError("server disconnected"),
                id="chat-remote-protocol",
            ),
            pytest.param(
                "chat",
                ollama.ResponseError("runner died", status_code=500),
                id="chat-5xx",
            ),
            pytest.param(
                "chat",
                ollama.ResponseError("rate limited", status_code=429),
                id="chat-429",
            ),
            pytest.param(
                "embed", ConnectionError(_CONNECTION_ERROR_MESSAGE), id="embed-connect"
            ),
            pytest.param(
                "embed", ollama.ResponseError(_EOF_ERROR_TEXT, 500), id="embed-5xx"
            ),
        ],
    )
    async def test_transient_error_is_retried(self, entry, error):
        fake_client, calls = self._make_flaky_client(error, failures=1)

        result = await self._run(entry, fake_client)

        if entry == "chat":
            assert result == "ok"
        else:
            assert np.array_equal(result, np.array([[0.1, 0.2, 0.3]]))
        assert calls["count"] == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("entry", "error", "message"),
        [
            pytest.param(
                "chat",
                ConnectionError(_CONNECTION_ERROR_MESSAGE),
                "Failed to connect to Ollama",
                id="chat-connect",
            ),
            pytest.param(
                "embed",
                ollama.ResponseError(_EOF_ERROR_TEXT, 500),
                "EOF (status code: 500)",
                id="embed-5xx",
            ),
        ],
    )
    async def test_persistent_error_reraises_original_after_three_attempts(
        self, entry, error, message
    ):
        fake_client, calls = self._make_flaky_client(error, failures=10)

        with pytest.raises(type(error)) as excinfo:
            await self._run(entry, fake_client)

        assert type(excinfo.value) is type(error)
        assert message in str(excinfo.value)
        assert calls["count"] == 3

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("entry", "status_code"),
        [("chat", 400), ("chat", 404), ("chat", 408), ("chat", 409), ("embed", 404)],
    )
    async def test_client_error_response_is_not_retried(self, entry, status_code):
        fake_client, calls = self._make_flaky_client(
            ollama.ResponseError("bad request", status_code=status_code), failures=10
        )
        with pytest.raises(ollama.ResponseError) as excinfo:
            await self._run(entry, fake_client)

        assert excinfo.value.status_code == status_code
        assert calls["count"] == 1


# --------------------------------------------------------------------------- #
# think option lifting and version gate
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize("think", [False, "high"])
async def test_think_is_lifted_out_of_a_shared_options_dict(think):
    fake_client = _make_fake_client()
    shared_options = {"think": think, "num_predict": 4096}

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        for prompt in ("first", "second"):
            await _ollama_model_if_cache(
                model="test-model", prompt=prompt, options=shared_options
            )

    for call in fake_client.chat.call_args_list:
        assert call.kwargs["think"] == think
        assert type(call.kwargs["think"]) is type(think)
        assert call.kwargs["options"] == {"num_predict": 4096}
    assert len(fake_client.chat.call_args_list) == 2
    assert shared_options == {"think": think, "num_predict": 4096}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [
        pytest.param({"options": {"num_predict": 4096}}, id="options-without-think"),
        pytest.param({}, id="no-options"),
    ],
)
async def test_absent_think_adds_no_think_kwarg(extra):
    _, fake_client = await _call_chat(prompt="Extract", **extra)

    call_kwargs = fake_client.chat.call_args.kwargs
    assert "think" not in call_kwargs
    assert call_kwargs.get("options") == extra.get("options")


@pytest.mark.asyncio
async def test_think_on_an_old_ollama_is_refused_before_the_call(monkeypatch):
    monkeypatch.setattr(ollama_binding, "_OLLAMA_SUPPORTS_THINK", False)
    fake_client = _make_fake_client()

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        with pytest.raises(RuntimeError, match="ollama>=0.5.4"):
            await _ollama_model_if_cache(
                model="test-model", prompt="Extract", options={"think": False}
            )

    fake_client.chat.assert_not_called()


# --------------------------------------------------------------------------- #
# Token usage reporting
# --------------------------------------------------------------------------- #


class _FakeStream:
    """Async iterable yielding pre-seeded Ollama stream chunks."""

    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for chunk in self._chunks:
                yield chunk

        return gen()


@pytest.mark.asyncio
async def test_completion_token_usage_recording():
    tracker = TokenTracker()
    result, fake_client = await _call_chat(
        {
            "message": {"content": "answer"},
            "done_reason": "stop",
            "prompt_eval_count": 31,
            "eval_count": 11,
        },
        prompt="Q",
        token_tracker=tracker,
    )

    assert result == "answer"
    assert tracker.get_usage() == {
        "prompt_tokens": 31,
        "completion_tokens": 11,
        "total_tokens": 42,
        "call_count": 1,
    }
    assert "token_tracker" not in fake_client.chat.await_args.kwargs

    # Missing counters must not bill as zero-token call.
    empty_tracker = TokenTracker()
    await _call_chat(
        {"message": {"content": "answer"}, "done_reason": "stop"},
        prompt="Q",
        token_tracker=empty_tracker,
    )
    assert empty_tracker.get_usage()["call_count"] == 0


@pytest.mark.asyncio
async def test_streaming_records_usage_from_the_terminal_chunk():
    tracker = TokenTracker()
    stream = _FakeStream(
        [
            {"message": {"content": "hel"}},
            {"message": {"content": "lo"}},
            {
                "message": {"content": ""},
                "done": True,
                "prompt_eval_count": 7,
                "eval_count": 2,
            },
        ]
    )

    result, fake_client = await _call_chat(
        stream, prompt="Q", stream=True, token_tracker=tracker
    )
    collected = "".join([chunk async for chunk in result])

    assert collected == "hello"
    fake_client._client.aclose.assert_awaited_once()
    assert tracker.get_usage() == {
        "prompt_tokens": 7,
        "completion_tokens": 2,
        "total_tokens": 9,
        "call_count": 1,
    }


@pytest.mark.asyncio
async def test_streaming_without_counters_or_disconnected_is_not_billed():
    tracker = TokenTracker()
    stream = _FakeStream([{"message": {"content": "hi"}}])
    result, _ = await _call_chat(stream, prompt="Q", stream=True, token_tracker=tracker)
    assert "".join([chunk async for chunk in result]) == "hi"
    assert tracker.get_usage()["call_count"] == 0

    disconnect_tracker = TokenTracker()
    stream2 = _FakeStream(
        [
            {"message": {"content": "hel"}},
            {"message": {"content": "lo"}},
            {"message": {"content": ""}, "done": True, "eval_count": 2},
        ]
    )
    result2, _ = await _call_chat(
        stream2, prompt="Q", stream=True, token_tracker=disconnect_tracker
    )
    agen = result2.__aiter__()
    assert await agen.__anext__() == "hel"
    await agen.aclose()
    assert disconnect_tracker.get_usage()["call_count"] == 0


def test_usage_counts_mapping_shapes():
    from ollama import ChatResponse

    assert _ollama_usage_counts({"message": {"content": "x"}}) is None
    assert _ollama_usage_counts({"eval_count": 3}) == {
        "prompt_tokens": 0,
        "completion_tokens": 3,
        "total_tokens": 3,
    }
    response = ChatResponse(
        model="test-model",
        message={"role": "assistant", "content": "hi"},
        prompt_eval_count=8,
        eval_count=4,
    )
    assert _ollama_usage_counts(response) == {
        "prompt_tokens": 8,
        "completion_tokens": 4,
        "total_tokens": 12,
    }
    assert _ollama_usage_counts(object()) is None


# --------------------------------------------------------------------------- #
# Truncation handling
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_ollama_length_done_reason_marks_partial_content_truncated():
    raw_json = ' {"entities":[{"name":"Ali '

    result, _ = await _call_chat(
        {"message": {"content": raw_json}, "done_reason": "length"}, prompt="Extract"
    )

    assert result == raw_json
    assert is_truncated_response(result) is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        pytest.param(
            {"message": {"content": '{"entities":[]}'}, "done_reason": "stop"},
            id="stop",
        ),
        pytest.param({"message": {"content": "answer"}}, id="missing-done-reason"),
        pytest.param(
            {"message": {"content": ""}, "done_reason": "stop"}, id="empty-stop"
        ),
    ],
)
async def test_ollama_non_length_response_is_returned_unmarked(response):
    result, _ = await _call_chat(response, prompt="Extract")

    assert result == response["message"]["content"]
    assert is_truncated_response(result) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["", "   \n  "])
async def test_ollama_empty_length_truncated_response_raises(lightrag_caplog, content):
    fake_client = _make_fake_client(
        {
            "message": {"content": content},
            "done_reason": "length",
            "eval_count": 64,
            "prompt_eval_count": 900,
        }
    )

    with lightrag_caplog.at_level(logging.ERROR):
        with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
            with pytest.raises(InvalidResponseError) as excinfo:
                await _ollama_model_if_cache(model="test-model", prompt="Extract")

    message = str(excinfo.value)
    assert "Received empty content from Ollama API" in message
    assert "done_reason=length" in message
    assert "eval_count=64" in message
    assert "OLLAMA_LLM_NUM_PREDICT" in message
    assert "hit the token limit" in lightrag_caplog.text
    fake_client._client.aclose.assert_awaited()


@pytest.mark.asyncio
async def test_ollama_empty_length_response_names_reasoning_and_is_billed():
    tracker = TokenTracker()

    with pytest.raises(InvalidResponseError) as excinfo:
        await _call_chat(
            {
                "message": {"content": "", "thinking": "let me carefully consider..."},
                "done_reason": "length",
                "prompt_eval_count": 12,
                "eval_count": 400,
            },
            prompt="Extract",
            token_tracker=tracker,
        )

    message = str(excinfo.value)
    assert "budget consumed by reasoning" in message
    assert "thinking_len=28" in message
    assert tracker.get_usage()["total_tokens"] == 412
