"""OpenAI embedding tests: special-token truncation, prefix budgeting, and response index ordering."""

import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from lightrag.llm.openai import _get_tiktoken_encoding_for_model, openai_embed

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]

MODEL = "text-embedding-3-small"
MARKER = "<|endoftext|>"
REPLACEMENT = "\ufffd"


class _FakeEmbeddingClient:
    """Captures the ``input`` list sent to the embeddings API."""

    def __init__(self, captured):
        self._captured = captured
        self.embeddings = SimpleNamespace(create=self._create)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def _create(self, **params):
        self._captured.append(params)
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.0, 1.0]) for _ in params["input"]]
        )


async def _embed(monkeypatch, texts, *, max_token_size, **kwargs):
    captured = []
    monkeypatch.setattr(
        "lightrag.llm.openai.create_openai_async_client",
        lambda **_: _FakeEmbeddingClient(captured),
    )
    await openai_embed.func(
        texts, model=MODEL, api_key="test-key", max_token_size=max_token_size, **kwargs
    )
    assert len(captured) == 1
    return captured[0]["input"]


def _token_count(text):
    encoding = _get_tiktoken_encoding_for_model(MODEL)
    return len(encoding.encode(text, disallowed_special=()))


async def test_special_token_and_short_texts_pass_through_unchanged(monkeypatch):
    texts = ["", "a perfectly ordinary chunk", f"another chunk mentioning {MARKER}"]
    assert await _embed(monkeypatch, texts, max_token_size=8192) == texts


async def test_oversized_text_containing_a_special_token(monkeypatch):
    text = f"filler words here {MARKER} more filler " * 40
    budget = 24

    (sent,) = await _embed(monkeypatch, [text], max_token_size=budget)

    assert sent != text
    assert _token_count(sent) <= budget


async def test_truncation_still_fills_the_budget(monkeypatch):
    text = "hello world " * 100
    budget = 24

    (sent,) = await _embed(monkeypatch, [text], max_token_size=budget)

    assert _token_count(sent) == budget


async def test_a_cut_inside_a_code_point_costs_one_trailing_replacement(monkeypatch):
    text = "hello world " * 10 + "\U0001f44d" * 5
    budget = 24

    (sent,) = await _embed(monkeypatch, [text], max_token_size=budget)

    assert sent.count(REPLACEMENT) <= 1
    assert REPLACEMENT not in sent[:-1]
    assert _token_count(sent) <= budget


async def test_prefix_is_applied_before_truncation(monkeypatch):
    prefix = "search_document: "
    budget = 12

    (sent,) = await _embed(
        monkeypatch,
        ["hello world " * 10],
        max_token_size=budget,
        context="document",
        document_prefix=prefix,
    )

    assert sent.startswith(prefix)
    assert _token_count(sent) <= budget


def _row(index, value):
    row = {"object": "embedding", "embedding": [value]}
    if index is not None:
        row["index"] = index
    return row


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        pytest.param([_row(1, 1.0), _row(0, 0.0)], [[0.0], [1.0]], id="shuffled"),
        pytest.param(
            [_row(None, 1.0), _row(None, 0.0)], [[1.0], [0.0]], id="unindexed"
        ),
        pytest.param([_row(0, 0.0), _row(0, 1.0)], None, id="duplicate"),
        pytest.param([_row(1, 1.0), _row(None, 0.0)], None, id="partial"),
        pytest.param([_row(True, 1.0), _row(0, 0.0)], None, id="non-int"),
    ],
)
async def test_openai_embedding_input_order(rows, expected):
    def respond(request):
        assert json.loads(request.content)["input"] == ["first", "second"]
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "test-embedding",
                "data": rows,
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        call = openai_embed.func(
            ["first", "second"],
            api_key="test-key",
            base_url="https://example.invalid/v1",
            client_configs={"http_client": client},
        )
        if expected is None:
            with pytest.raises(ValueError, match="invalid embedding indices"):
                await call
        else:
            np.testing.assert_array_equal(await call, expected)
