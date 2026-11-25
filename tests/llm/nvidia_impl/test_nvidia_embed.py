"""NVIDIA OpenAI-compatible embedding tests: client lifetime, retries, encoding, ordering, and asymmetric input_type."""

import base64
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np
import pytest
from openai import AsyncOpenAI, RateLimitError
from tenacity import wait_none

from lightrag.llm import nvidia_openai
from lightrag.llm.nvidia_openai import nvidia_openai_embed

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


class _FakeAsyncOpenAI:
    """Mimics ``AsyncOpenAI`` as an async context manager that closes on exit."""

    def __init__(self, *, create):
        self.embeddings = SimpleNamespace(create=create)
        self.close = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()


def _row(index, embedding):
    row = {"object": "embedding", "embedding": embedding}
    if index is not None:
        row["index"] = index
    return row


def _mock_sdk(monkeypatch, rows):
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "nvidia/llama-3.2-nv-embedqa-1b-v1",
                "data": rows,
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    client = AsyncOpenAI(
        api_key="test-key",
        base_url="https://nvidia.example/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    monkeypatch.setattr(nvidia_openai, "AsyncOpenAI", lambda **kwargs: client)
    return client, requests


async def test_exhausted_retries_reraise_the_sdk_error(monkeypatch):
    monkeypatch.setattr(nvidia_openai_embed.func.retry, "wait", wait_none())
    request = httpx.Request("POST", "https://integrate.api.nvidia.com/v1/embeddings")
    error = RateLimitError(
        "rate limited", response=httpx.Response(429, request=request), body=None
    )
    fake = _FakeAsyncOpenAI(create=AsyncMock(side_effect=error))

    with patch("lightrag.llm.nvidia_openai.AsyncOpenAI", return_value=fake):
        with pytest.raises(RateLimitError) as caught:
            await nvidia_openai_embed(texts=["hello"], api_key="k")

    assert caught.value is error
    assert fake.embeddings.create.await_count == 3
    assert fake.close.await_count == 3


async def test_explicit_api_key_is_request_local(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "shared-openai-key")
    fake = _FakeAsyncOpenAI(
        create=AsyncMock(
            return_value=SimpleNamespace(data=[SimpleNamespace(embedding=[0.1])])
        )
    )

    with patch(
        "lightrag.llm.nvidia_openai.AsyncOpenAI", return_value=fake
    ) as client_cls:
        await nvidia_openai_embed.func.__wrapped__(
            texts=["hello"], api_key="request-nvidia-key"
        )

    client_cls.assert_called_once_with(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="request-nvidia-key",
    )
    assert os.environ["OPENAI_API_KEY"] == "shared-openai-key"


@pytest.mark.parametrize(
    ("encode", "response_format"),
    [("float", "float"), ("base64", "base64"), ("base64", "float")],
)
async def test_embedding_encoding_preserves_numeric_vectors(
    monkeypatch, encode, response_format
):
    expected = (np.arange(2 * 2048, dtype=np.float32).reshape(2, 2048) - 1024) / 4096
    client, requests = _mock_sdk(
        monkeypatch,
        [
            _row(
                index,
                base64.b64encode(vector.tobytes()).decode("ascii")
                if response_format == "base64"
                else vector.tolist(),
            )
            for index, vector in enumerate(expected)
        ],
    )
    texts = ["document 0", "document 1"]

    result = await nvidia_openai_embed(texts, encode=encode)

    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, expected)
    assert requests[0]["input"] == texts
    assert requests[0]["encoding_format"] == encode
    assert client.is_closed()


async def test_short_base64_vector_fails_dimension_validation(monkeypatch):
    vector = np.zeros(3, dtype=np.float32)
    client, _ = _mock_sdk(
        monkeypatch, [_row(0, base64.b64encode(vector.tobytes()).decode("ascii"))]
    )

    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        await nvidia_openai_embed(["document"], encode="base64")
    assert client.is_closed()


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        pytest.param(
            [_row(1, [10.0]), _row(0, [20.0])], [[20.0], [10.0]], id="shuffled"
        ),
        pytest.param(
            [_row(None, [10.0]), _row(None, [20.0])],
            [[10.0], [20.0]],
            id="unindexed",
        ),
        pytest.param([_row(0, [10.0]), _row(0, [20.0])], None, id="duplicate"),
        pytest.param([_row(1, [10.0]), _row(None, [20.0])], None, id="partial"),
        pytest.param(
            [_row(0, [10.0]), _row(1, [20.0]), _row(1, [30.0])], None, id="extra-row"
        ),
    ],
)
async def test_embedding_rows_follow_input_indices(monkeypatch, rows, expected):
    client, _ = _mock_sdk(monkeypatch, rows)
    call = nvidia_openai_embed.func(["first", "second"])

    if expected is None:
        with pytest.raises(ValueError, match="invalid embedding indices for 2 texts"):
            await call
    else:
        np.testing.assert_array_equal(await call, expected)
    assert client.is_closed()


@pytest.mark.parametrize(
    ("kwargs", "expected_input_type"),
    [
        pytest.param({"context": "query"}, "query", id="query-context"),
        pytest.param({"context": "document"}, "passage", id="document-context"),
        pytest.param({}, "passage", id="no-context"),
        pytest.param(
            {"context": "query", "input_type": "passage"},
            "passage",
            id="explicit-input-type-wins",
        ),
    ],
)
async def test_input_type_follows_lightrag_context(
    monkeypatch, kwargs, expected_input_type
):
    _, requests = _mock_sdk(monkeypatch, [_row(0, [0.0] * 2048)])

    await nvidia_openai_embed(["text"], trunc="END", **kwargs)

    assert requests[0]["input_type"] == expected_input_type
    assert requests[0]["truncate"] == "END"
