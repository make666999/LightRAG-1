"""LlamaIndex adapter tests: completion/embedding entry points and Settings configuration."""

from __future__ import annotations

import asyncio
from copy import copy
from functools import partial

import numpy as np
import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM
from llama_index.core.settings import Settings
from llama_index.core.storage.kvstore import SimpleKVStore
from pydantic import Field

from lightrag.exceptions import APIConnectionError
from lightrag.llm import llama_index_impl
from lightrag.llm.llama_index_impl import (
    configure_llama_index,
    llama_index_complete,
    llama_index_complete_if_cache,
    llama_index_embed,
)
from lightrag.utils import EmbeddingFunc

pytestmark = pytest.mark.offline


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChatResponse:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeLLM:
    def __init__(self, content: str = "answer") -> None:
        self.content = content
        self.received_kwargs = None

    async def achat(self, messages, **kwargs):
        self.received_kwargs = kwargs
        return _FakeChatResponse(self.content)


class _RecordingEmbedModel(BaseEmbedding):
    vectors: dict[str, list[float]]
    batches: list[list[str]] = Field(default_factory=list)

    def _get_query_embedding(self, query: str) -> list[float]:
        raise AssertionError("Query embedding is not used")

    async def _aget_query_embedding(self, query: str) -> list[float]:
        raise AssertionError("Query embedding is not used")

    def _get_text_embedding(self, text: str) -> list[float]:
        raise AssertionError("Synchronous embedding must not be used")

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        await asyncio.sleep(0)
        return [self.vectors[text] for text in texts]


class _FailingModel:
    """Stand-in raising the same exception for both LLM and embed calls."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    async def achat(self, messages, **kwargs):
        self.calls += 1
        raise self.error

    async def aget_text_embedding_batch(self, texts):
        self.calls += 1
        raise self.error


# --------------------------------------------------------------------------- #
# llama_index_complete & llama_index_embed
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("extra_kwargs", "expected_chat_kwargs"),
    [
        pytest.param({}, {}, id="llm-instance-only"),
        pytest.param(
            {"hashing_kv": object(), "max_tokens": 256, "stream": False},
            {"max_tokens": 256},
            id="role-wrapper-kwargs-and-max-tokens",
        ),
        pytest.param(
            {"chat_kwargs": {"temperature": 0.1}},
            {"temperature": 0.1},
            id="explicit-chat-kwargs",
        ),
        pytest.param(
            {
                "max_tokens": 256,
                "chat_kwargs": {"max_tokens": 64, "temperature": 0.1},
            },
            {"max_tokens": 64, "temperature": 0.1},
            id="explicit-chat-kwargs-wins-over-max-tokens",
        ),
    ],
)
async def test_llama_index_complete_kwargs_handling(extra_kwargs, expected_chat_kwargs):
    llm = _FakeLLM(content="hello back")

    result = await llama_index_complete("hi", llm_instance=llm, **extra_kwargs)

    assert result == "hello back"
    assert llm.received_kwargs == expected_chat_kwargs


@pytest.mark.asyncio
async def test_llama_index_complete_ignores_settings(monkeypatch):
    configured = []
    monkeypatch.setattr(
        llama_index_impl,
        "configure_llama_index",
        lambda *args, **kwargs: configured.append((args, kwargs)),
    )
    llm = _FakeLLM(content="hello back")

    result = await llama_index_complete("hi", llm_instance=llm, settings=Settings)

    assert result == "hello back"
    assert configured == []
    assert llm.received_kwargs == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("texts", [["a", "b"], []])
async def test_llama_index_embed_uses_async_batch_method(texts: list[str]) -> None:
    model = _RecordingEmbedModel(vectors={"a": [0.1, 0.2], "b": [0.3, 0.4]})

    result = await llama_index_embed.func(texts, embed_model=model)

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(
        result, np.array([model.vectors[text] for text in texts])
    )


@pytest.mark.asyncio
async def test_llama_index_embed_respects_batch_size_order_and_embedding_func_contract() -> (
    None
):
    texts = ["c", "a", "b", "d", "e"]
    model = _RecordingEmbedModel(
        vectors={text: [float(i), 0.5] for i, text in enumerate(texts)},
        embed_batch_size=2,
    )
    embedding_func = EmbeddingFunc(
        embedding_dim=2,
        max_token_size=64,
        func=partial(llama_index_embed.func, embed_model=model),
    )

    result = await embedding_func(texts)

    assert result.shape == (len(texts), 2)
    assert sorted(model.batches) == sorted([["c", "a"], ["b", "d"], ["e"]])
    np.testing.assert_array_equal(
        result, np.array([model.vectors[text] for text in texts])
    )


@pytest.mark.asyncio
async def test_llama_index_embed_reuses_cache_and_preserves_order() -> None:
    model = _RecordingEmbedModel(
        vectors={"a": [0.1, 0.2], "b": [0.3, 0.4], "c": [0.5, 0.6]},
        embeddings_cache=SimpleKVStore(),
    )
    await model.aget_text_embedding_batch(["b"])
    model.batches.clear()
    texts = ["c", "b", "a"]

    result = await llama_index_embed.func(texts, embed_model=model)

    assert model.batches == [["c", "a"]]
    np.testing.assert_array_equal(
        result, np.array([model.vectors[text] for text in texts])
    )
    model.batches.clear()
    repeated = await llama_index_embed.func(texts, embed_model=model)
    assert model.batches == []
    np.testing.assert_array_equal(repeated, result)


@pytest.mark.asyncio
async def test_llama_index_embed_requires_embed_model():
    with pytest.raises(ValueError, match="embed_model must be provided"):
        await llama_index_embed.func(["a"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda m: llama_index_complete_if_cache(m, "hi"), id="complete"),
        pytest.param(
            lambda m: llama_index_embed.func(["a"], embed_model=m), id="embed"
        ),
    ],
)
async def test_llama_index_adds_no_retry_layer(call):
    error = APIConnectionError(request=None)
    model = _FailingModel(error)

    with pytest.raises(APIConnectionError) as exc_info:
        await call(model)

    assert exc_info.value is error
    assert model.calls == 1


# --------------------------------------------------------------------------- #
# configure_llama_index Settings behavior
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "arg", [pytest.param(None, id="default"), pytest.param(Settings, id="explicit")]
)
def test_settings_returns_sdk_singleton_unchanged(arg):
    before = vars(Settings).copy()
    result = configure_llama_index() if arg is None else configure_llama_index(arg)
    assert result is Settings
    assert vars(Settings) == before


@pytest.mark.asyncio
async def test_embedding_accepts_sdk_settings():
    before = vars(Settings).copy()
    model = MockEmbedding(embed_dim=3)
    actual = await llama_index_embed.func(
        ["alpha", "beta"], embed_model=model, settings=Settings
    )
    np.testing.assert_array_equal(actual, np.full((2, 3), 0.5))
    assert vars(Settings) == before


def test_provided_settings_are_not_copied_to_the_global_singleton():
    settings = copy(Settings)
    model = MockEmbedding(embed_dim=3)
    settings.embed_model = model
    before = vars(Settings).copy()
    assert configure_llama_index(settings) is settings
    assert settings.embed_model is model
    assert vars(Settings) == before


def test_unknown_setting_remains_a_warning(lightrag_log_records):
    settings = copy(Settings)
    assert configure_llama_index(settings, not_a_setting=1) is settings
    assert not hasattr(settings, "not_a_setting")
    assert any(
        "Unknown LlamaIndex setting: not_a_setting" in record.getMessage()
        for record in lightrag_log_records
    )


def test_lazy_model_settings_are_assigned_without_loading_defaults(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = copy(Settings)
    embed_model = MockEmbedding(embed_dim=3)
    llm = MockLLM()
    before = vars(Settings).copy()

    assert configure_llama_index(settings, embed_model=embed_model, llm=llm) is settings
    assert settings.embed_model is embed_model
    assert settings.llm is llm
    assert vars(Settings) == before


def test_instance_attributes_of_other_settings_objects_are_configurable():
    class _PlainSettings:
        def __init__(self) -> None:
            self.chunk_size = 1024

    settings = _PlainSettings()
    assert configure_llama_index(settings, chunk_size=256) is settings
    assert settings.chunk_size == 256
