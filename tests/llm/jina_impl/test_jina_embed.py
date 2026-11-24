"""Jina embedding tests: request-local API key isolation, index alignment, and retry classification."""

import base64
import os
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import numpy as np
import pytest
from tenacity import wait_none

from lightrag.llm import jina
from lightrag.llm.jina import jina_embed

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]

_URL = "https://api.jina.ai/v1/embeddings"


def _encoded_embedding(values=(0.1, 0.2, 0.3)) -> str:
    return base64.b64encode(np.asarray(values, dtype=np.float32).tobytes()).decode()


def _embedding_row(value, **metadata):
    encoded = base64.b64encode(np.asarray([value, 0], dtype=np.float32)).decode()
    return {"embedding": encoded, **metadata}


def _client_response_error(status: int) -> aiohttp.ClientResponseError:
    request_info = aiohttp.RequestInfo(
        url=_URL, method="POST", headers={}, real_url=_URL
    )
    return aiohttp.ClientResponseError(
        request_info=request_info,
        history=(),
        status=status,
        message=f"Jina API error: HTTP {status}",
    )


@pytest.fixture
def serve_embeddings(monkeypatch):
    def serve(rows):
        response = MagicMock(status=200)
        response.json = AsyncMock(return_value={"data": rows})
        session = MagicMock()
        session.post.return_value.__aenter__.return_value = response
        factory = MagicMock()
        factory.return_value.__aenter__.return_value = session
        monkeypatch.setattr(jina.aiohttp, "ClientSession", factory)
        return session.post

    return serve


async def test_explicit_api_key_is_request_local(monkeypatch):
    """An explicit Jina key must not replace the process-wide default key."""
    monkeypatch.setenv("JINA_API_KEY", "shared-jina-key")
    fetch = AsyncMock(return_value=[{"embedding": _encoded_embedding()}])

    with patch("lightrag.llm.jina.fetch_data", fetch):
        result = await jina_embed.func.__wrapped__(
            texts=["hello"], api_key="request-jina-key", embedding_dim=3
        )

    _, headers, _ = fetch.await_args.args
    assert headers["Authorization"] == "Bearer request-jina-key"
    assert os.environ["JINA_API_KEY"] == "shared-jina-key"
    np.testing.assert_allclose(result, [[0.1, 0.2, 0.3]])


@pytest.mark.parametrize(
    "rows",
    [
        pytest.param(
            [_embedding_row(2, index=1), _embedding_row(1, index=0)], id="reversed"
        ),
        pytest.param([_embedding_row(1), _embedding_row(2)], id="unindexed"),
    ],
)
async def test_embeddings_follow_input_order(serve_embeddings, rows):
    post = serve_embeddings(rows)

    result = await jina.jina_embed.func(
        ["first", "second"], api_key="test-key", embedding_dim=2
    )

    np.testing.assert_array_equal(result, [[1, 0], [2, 0]])
    assert post.call_args.kwargs["json"]["input"] == ["first", "second"]


@pytest.mark.parametrize(
    "second_index",
    [
        pytest.param(0, id="duplicate"),
        pytest.param(2, id="out-of-range"),
        pytest.param(True, id="boolean"),
        pytest.param(None, id="missing"),
    ],
)
async def test_invalid_indices_do_not_return_misaligned_vectors(
    serve_embeddings, second_index
):
    second = (
        _embedding_row(2)
        if second_index is None
        else _embedding_row(2, index=second_index)
    )
    serve_embeddings([_embedding_row(1, index=0), second])

    with pytest.raises(ValueError, match="embedding indices"):
        await jina.jina_embed.func(
            ["first", "second"], api_key="test-key", embedding_dim=2
        )


@pytest.mark.parametrize(
    ("exc", "attempts"),
    [
        pytest.param(_client_response_error(401), 1, id="permanent-status"),
        pytest.param(_client_response_error(503), 3, id="transient-status"),
        pytest.param(aiohttp.InvalidURL("not-a-valid-url"), 1, id="invalid-url"),
        pytest.param(aiohttp.ClientConnectionError("reset"), 3, id="connection"),
    ],
)
async def test_retry_follows_the_shared_classification(monkeypatch, exc, attempts):
    """jina_embed retries transient aiohttp errors up to 3 times and re-raises original."""
    monkeypatch.setenv("JINA_API_KEY", "irrelevant")
    monkeypatch.setattr(jina_embed.func.retry, "wait", wait_none())
    call_count = 0

    async def fake_fetch_data(url, headers, data, timeout=None):
        nonlocal call_count
        call_count += 1
        raise exc

    with patch("lightrag.llm.jina.fetch_data", fake_fetch_data):
        with pytest.raises(type(exc)) as excinfo:
            await jina_embed.func(texts=["hello"], api_key="key", embedding_dim=3)

    assert call_count == attempts
    assert excinfo.value is exc
