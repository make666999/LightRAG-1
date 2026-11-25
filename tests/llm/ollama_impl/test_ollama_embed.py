"""Ollama embedding tests: embedding_dim forwarding and prompt token usage tracking."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from lightrag.llm.ollama import ollama_embed
from lightrag.utils import TokenTracker

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


@pytest.mark.parametrize(
    ("args", "kwargs", "expected_extra"),
    [
        pytest.param(
            (),
            {"embed_model": "test-model", "embedding_dim": 256},
            {"dimensions": 256},
            id="explicit-dim-forwarded",
        ),
        pytest.param((), {"embed_model": "test-model"}, {}, id="unset-dim-omitted"),
        pytest.param(
            ("test-model", 4096), {}, {}, id="positional-third-arg-is-max-token-size"
        ),
    ],
)
async def test_embedding_dim_forwarding(args, kwargs, expected_extra):
    fake_client = SimpleNamespace(
        embed=AsyncMock(return_value={"embeddings": [[0.1, 0.2, 0.3]]}),
        _client=SimpleNamespace(aclose=AsyncMock()),
    )

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        result = await ollama_embed.func(["hello"], *args, **kwargs)

    fake_client.embed.assert_awaited_once_with(
        model="test-model", input=["hello"], options={}, **expected_extra
    )
    assert np.array_equal(result, np.array([[0.1, 0.2, 0.3]]))


async def test_embed_records_prompt_tokens_only():
    """An embed call has no generation, so Ollama reports no `eval_count`."""
    tracker = TokenTracker()
    fake_client = SimpleNamespace(
        embed=AsyncMock(
            return_value={"embeddings": [[0.1, 0.2]], "prompt_eval_count": 6}
        ),
        _client=SimpleNamespace(aclose=AsyncMock()),
    )

    with patch("lightrag.llm.ollama.ollama.AsyncClient", return_value=fake_client):
        vectors = await ollama_embed.func(["hello"], token_tracker=tracker)

    assert isinstance(vectors, np.ndarray)
    assert tracker.get_usage() == {
        "prompt_tokens": 6,
        "completion_tokens": 0,
        "total_tokens": 6,
        "call_count": 1,
    }
