"""Query error handling, empty final evidence, and bypass mode contracts.

Covers:
1. Final filtering (rerank / token budget) removing all evidence uses the same
   no-results failure path as an empty search, while surviving graph evidence
   still generates an answer.
2. Answering LLM failures in ``aquery_llm`` / ``aquery`` propagate instead of
   collapsing into empty / ``None`` results (#4475).
3. ``mode="bypass"`` queues non-streaming answers (including ``TruncatedResponse``)
   at query priority and preserves streaming async iterators.
"""

from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lightrag import LightRAG
from lightrag.base import QueryParam
from lightrag.constants import DEFAULT_QUERY_PRIORITY, DEFAULT_SUMMARY_PRIORITY
from lightrag.prompt import PROMPTS
from lightrag.utils import Tokenizer, TruncatedResponse

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]

_TIMEOUT_MESSAGE = "query LLM func: Worker execution timeout after 2s"


class _CharacterTokenizer:
    """1-char-per-token tokenizer stub for deterministic budget checks."""

    def encode(self, content):
        return [ord(char) for char in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


def _make_query_rag(
    monkeypatch,
    *,
    model: AsyncMock,
    enable_cache: bool,
    reranker: AsyncMock | None = None,
):
    chunk = {
        "id": "chunk-1",
        "chunk_id": "chunk-1",
        "content": "Retrieved evidence.",
        "file_path": "source.md",
    }
    search = {
        "final_entities": [],
        "final_relations": [],
        "vector_chunks": [chunk],
        "chunk_tracking": {"chunk-1": {"source": "C", "frequency": 1, "order": 1}},
        "query_embedding": None,
    }
    monkeypatch.setattr(
        "lightrag.operate._perform_kg_search", AsyncMock(return_value=search)
    )
    config = {
        "tokenizer": Tokenizer("character", _CharacterTokenizer()),
        "role_llm_funcs": {"query": model},
        "addon_params": {"language": "en"},
    }
    if reranker is not None:
        config["min_rerank_score"] = 0.5
        config["rerank_model_func"] = reranker
    cache = SimpleNamespace(
        global_config={"enable_llm_cache": enable_cache},
        get_by_id=AsyncMock(return_value=None),
        upsert=AsyncMock(),
    )
    rag = SimpleNamespace(
        _build_global_config=lambda: config,
        _query_done=AsyncMock(),
        chunk_entity_relation_graph=None,
        entities_vdb=None,
        relationships_vdb=None,
        text_chunks=SimpleNamespace(global_config=config),
        chunks_vdb=SimpleNamespace(
            cosine_better_than_threshold=0.0, query=AsyncMock(return_value=[chunk])
        ),
        llm_response_cache=cache,
    )
    rag.aquery_llm = partial(LightRAG.aquery_llm, rag)
    return rag, search, cache


@pytest.fixture
def query_setup(monkeypatch):
    model = AsyncMock(return_value="Grounded answer.")
    reranker = AsyncMock(return_value=[{"index": 0, "relevance_score": 0.1}])
    rag, search, cache = _make_query_rag(
        monkeypatch, model=model, enable_cache=True, reranker=reranker
    )
    return rag, search, model, cache, reranker


@pytest.fixture
def failing_rag(monkeypatch):
    model = AsyncMock(side_effect=TimeoutError(_TIMEOUT_MESSAGE))
    rag, _, _ = _make_query_rag(monkeypatch, model=model, enable_cache=False)
    return rag, model


async def _query(rag, mode, **options):
    slot = "content_data" if mode == "naive" else "context_data"
    return await LightRAG.aquery_llm(
        rag,
        "Find evidence",
        param=QueryParam(mode=mode, ll_keywords=["evidence"], **options),
        system_prompt="Answer using: {" + slot + "}",
    )


# ---------------------------------------------------------------------------
# 1. Empty final evidence after rerank / budget filtering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["naive", "mix"])
@pytest.mark.parametrize("cause", ["rerank", "budget"])
@pytest.mark.parametrize("output", ["answer", "stream", "context", "prompt"])
async def test_all_final_evidence_removed_returns_no_results(
    query_setup, mode, cause, output
):
    rag, _, model, cache, reranker = query_setup
    result = await _query(
        rag,
        mode,
        enable_rerank=cause == "rerank",
        max_total_tokens=4096 if cause == "rerank" else 120,
        stream=output == "stream",
        only_need_context=output == "context",
        only_need_prompt=output == "prompt",
    )

    assert result["status"] == "failure"
    assert result["metadata"]["failure_reason"] == "no_results"
    assert result["llm_response"] == {
        "content": PROMPTS["fail_response"],
        "response_iterator": None,
        "is_streaming": False,
        "llm_generated": False,
    }
    model.assert_not_awaited()
    cache.get_by_id.assert_not_awaited()
    cache.upsert.assert_not_awaited()
    if cause == "rerank":
        reranker.assert_awaited_once()


@pytest.mark.parametrize("mode", ["naive", "mix"])
@pytest.mark.parametrize("cause", ["rerank", "budget"])
async def test_data_only_query_reports_final_empty_evidence(query_setup, mode, cause):
    rag, _, model, cache, _ = query_setup
    result = await LightRAG.aquery_data(
        rag,
        "Find evidence",
        param=QueryParam(
            mode=mode,
            ll_keywords=["evidence"],
            enable_rerank=cause == "rerank",
            max_total_tokens=4096 if cause == "rerank" else 120,
        ),
    )

    assert result["status"] == "failure"
    assert result["metadata"]["failure_reason"] == "no_results"
    assert result["data"] == {}
    model.assert_not_awaited()
    cache.get_by_id.assert_not_awaited()
    cache.upsert.assert_not_awaited()


@pytest.mark.parametrize("mode", ["hybrid", "mix"])
@pytest.mark.parametrize("evidence", ["entity", "relationship"])
async def test_filtered_chunks_preserve_graph_evidence(query_setup, mode, evidence):
    rag, search, model, _, _ = query_setup
    if evidence == "entity":
        search["final_entities"] = [
            {"entity_name": "Evidence", "description": "A surviving graph fact."}
        ]
    else:
        search["final_relations"] = [
            {"src_id": "Evidence", "tgt_id": "Fact", "description": "Related facts."}
        ]

    result = await _query(rag, mode, enable_rerank=True, max_total_tokens=4096)

    assert result["status"] == "success"
    assert result["data"]["chunks"] == []
    graph_key = "entities" if evidence == "entity" else "relationships"
    assert len(result["data"][graph_key]) == 1
    assert result["llm_response"]["content"] == "Grounded answer."
    assert result["llm_response"]["llm_generated"] is True
    model.assert_awaited_once()


@pytest.mark.parametrize("mode", ["naive", "mix"])
async def test_retained_chunks_still_generate_answers(query_setup, mode):
    rag, _, model, _, _ = query_setup
    result = await _query(rag, mode, enable_rerank=False, max_total_tokens=4096)

    assert result["status"] == "success"
    assert len(result["data"]["chunks"]) == 1
    assert result["llm_response"]["llm_generated"] is True
    model.assert_awaited_once()


# ---------------------------------------------------------------------------
# 2. Answering LLM failures propagate instead of returning empty/None (#4475)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mode", ["naive", "mix", "bypass"])
async def test_aquery_llm_raises_when_the_answering_llm_fails(
    failing_rag, mode, stream
):
    rag, model = failing_rag

    with pytest.raises(TimeoutError, match="Worker execution timeout"):
        await rag.aquery_llm(
            "Find evidence",
            param=QueryParam(mode=mode, ll_keywords=["evidence"], stream=stream),
        )

    model.assert_awaited_once()


async def test_aquery_raises_instead_of_returning_none(failing_rag):
    rag, model = failing_rag

    with pytest.raises(TimeoutError, match="Worker execution timeout"):
        await LightRAG.aquery(
            rag,
            "Find evidence",
            param=QueryParam(mode="mix", ll_keywords=["evidence"], stream=False),
        )

    model.assert_awaited_once()


# ---------------------------------------------------------------------------
# 3. mode="bypass" priority and streaming classification
# ---------------------------------------------------------------------------


class _FakeBypassRAG:
    """Minimal stand-in exposing only what the bypass branch touches."""

    def __init__(self, llm_func):
        self._llm = llm_func

    def _build_global_config(self):
        return {"role_llm_funcs": {"query": self._llm}}


async def test_bypass_non_streaming_answer_at_query_priority():
    """A TruncatedResponse stays non-streaming, queues at query priority,
    and allows a short prompt below the RAG retrieval minimum.
    """
    seen: list[dict] = []

    async def llm(prompt, **kwargs):
        seen.append(kwargs)
        return TruncatedResponse("partial answer")

    result = await LightRAG.aquery_llm(
        _FakeBypassRAG(llm), "中", param=QueryParam(mode="bypass", stream=False)
    )

    llm_response = result["llm_response"]
    assert llm_response["is_streaming"] is False
    assert llm_response["content"] == "partial answer"
    assert llm_response["response_iterator"] is None
    assert len(seen) == 1
    assert DEFAULT_QUERY_PRIORITY != DEFAULT_SUMMARY_PRIORITY
    assert seen[0]["_priority"] == DEFAULT_QUERY_PRIORITY


async def test_bypass_iterator_stays_streaming():
    async def _chunks():
        yield "part"

    iterator = _chunks()

    async def llm(prompt, **_kwargs):
        return iterator

    result = await LightRAG.aquery_llm(
        _FakeBypassRAG(llm), "question", param=QueryParam(mode="bypass", stream=True)
    )

    llm_response = result["llm_response"]
    assert llm_response["is_streaming"] is True
    assert llm_response["response_iterator"] is iterator
