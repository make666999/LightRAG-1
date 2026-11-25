"""Tests for keyword extraction prompts, LLM driver guards, and KG search keyword fallbacks."""

from __future__ import annotations

import json
import re
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from lightrag.base import QueryParam
from lightrag.llm.lmdeploy import lmdeploy_model_if_cache
from lightrag.llm.lollms import lollms_model_complete, lollms_model_if_cache
from lightrag.operate import _perform_kg_search
from lightrag.prompt import PROMPTS

pytestmark = pytest.mark.offline


# --------------------------------------------------------------------------- #
# 1. Keyword extraction prompt template & format-only examples
# --------------------------------------------------------------------------- #


def test_keywords_extraction_prompt_template_formats_with_literal_json_braces():
    rendered = PROMPTS["keywords_extraction"].format(
        query="hello",
        examples="example",
        language="en",
    )

    assert "first character of your response must be `{`" in rendered
    assert '{"high_level_keywords": [], "low_level_keywords": []}' in rendered


def test_keywords_extraction_examples_are_format_only():
    """Keyword examples must be placeholder-only JSON templates, not sample demos."""
    placeholder = re.compile(r"<[^<>]+>")

    for example in PROMPTS["keywords_extraction_examples"]:
        assert "Query:" not in example
        assert "Output:" not in example

        parsed = json.loads(example)
        assert set(parsed) == {"high_level_keywords", "low_level_keywords"}
        keywords = parsed["high_level_keywords"] + parsed["low_level_keywords"]
        assert keywords
        for keyword in keywords:
            assert placeholder.fullmatch(keyword), keyword


def test_keywords_extraction_prompt_labels_template_as_not_source_text():
    prompt = PROMPTS["keywords_extraction"]

    assert "---Output Format Template---" in prompt
    assert "---Examples---" not in prompt
    assert "Apple Inc." not in prompt
    assert "output JSON format template only" in prompt
    assert "not source text" in prompt
    assert "must never be used as keyword extraction content" in prompt
    assert (
        "derived only from the `User Query` in the `---Real Data---` section" in prompt
    )


def test_keywords_extraction_prompt_keeps_single_real_user_query_section():
    rendered = PROMPTS["keywords_extraction"].format(
        query="How did LightRAG improve retrieval?",
        examples="\n".join(PROMPTS["keywords_extraction_examples"]),
        language="English",
    )

    assert rendered.count("User Query:") == 1
    assert "User Query: How did LightRAG improve retrieval?" in rendered
    assert "User Query:" not in "\n".join(PROMPTS["keywords_extraction_examples"])


# --------------------------------------------------------------------------- #
# 2. Provider driver response_format / keyword_extraction handling
# --------------------------------------------------------------------------- #


def _make_fake_lollms_session(captured_requests: list[dict] | None = None):
    class FakeResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def text(self):
            return "{}"

    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        def post(self, url, json):
            if captured_requests is not None:
                captured_requests.append(json)
            return FakeResponse()

    return FakeSession


@pytest.mark.asyncio
async def test_lollms_if_cache_strips_response_format_before_request():
    """lollms_model_if_cache drops response_format; lollms has no JSON mode."""
    captured_requests: list[dict] = []

    with patch(
        "lightrag.llm.lollms.aiohttp.ClientSession",
        _make_fake_lollms_session(captured_requests),
    ):
        result = await lollms_model_if_cache(
            model="lollms-model",
            prompt="hello",
            response_format={"type": "json_object"},
        )

    assert result == "{}"
    assert captured_requests
    assert "response_format" not in captured_requests[0]


@pytest.mark.asyncio
async def test_lollms_if_cache_emits_deprecation_warning():
    with patch(
        "lightrag.llm.lollms.aiohttp.ClientSession",
        _make_fake_lollms_session(),
    ):
        with pytest.warns(DeprecationWarning):
            await lollms_model_if_cache(
                model="lollms-model",
                prompt="hello",
                keyword_extraction=True,
            )


@pytest.mark.asyncio
async def test_lollms_complete_forwards_legacy_flag_downstream():
    hashing_kv = SimpleNamespace(global_config={"llm_model_name": "lollms-model"})

    with patch(
        "lightrag.llm.lollms.lollms_model_if_cache",
        AsyncMock(return_value="{}"),
    ) as mocked_complete:
        await lollms_model_complete(
            prompt="hello",
            hashing_kv=hashing_kv,
            keyword_extraction=True,
        )

    assert mocked_complete.await_args.kwargs.get("keyword_extraction") is True


@pytest.mark.asyncio
async def test_lmdeploy_strips_response_format_before_generation_config(monkeypatch):
    captured_gen_config_kwargs = {}

    class FakeGenerationConfig:
        def __init__(self, **kwargs):
            captured_gen_config_kwargs.update(kwargs)

    class FakeVersion:
        def __lt__(self, other):
            return False

    async def fake_generate(*_args, **_kwargs):
        yield SimpleNamespace(response="{}")

    monkeypatch.setattr(
        "lightrag.llm.lmdeploy.initialize_lmdeploy_pipeline",
        lambda **_kwargs: SimpleNamespace(generate=fake_generate),
    )
    monkeypatch.setitem(
        sys.modules,
        "lmdeploy",
        SimpleNamespace(
            __version__="0.6.0",
            version_info=FakeVersion(),
            GenerationConfig=FakeGenerationConfig,
        ),
    )

    result = await lmdeploy_model_if_cache(
        model="lmdeploy-model",
        prompt="hello",
        response_format={"type": "json_object"},
    )

    assert result == "{}"
    assert "response_format" not in captured_gen_config_kwargs
    assert "keyword_extraction" not in captured_gen_config_kwargs


# --------------------------------------------------------------------------- #
# 3. KG search empty-keyword fallbacks
# --------------------------------------------------------------------------- #

_LL = "Tesla, SpaceX"
_HL = "launch costs, tariffs"


def _make_vdb():
    vdb = AsyncMock()
    vdb.query = AsyncMock(return_value=[])
    vdb.cosine_better_than_threshold = 0.2
    return vdb


async def _search(mode: str, ll_keywords: str, hl_keywords: str):
    async def _embed(texts, **kwargs):
        return np.array(
            [np.full(8, i + 1, dtype=np.float32) for i in range(len(texts))]
        )

    embed_func = AsyncMock(side_effect=_embed)
    text_chunks_db = MagicMock()
    text_chunks_db.embedding_func = embed_func
    text_chunks_db.global_config = {"kg_chunk_pick_method": "VECTOR"}
    entities_vdb = _make_vdb()
    relationships_vdb = _make_vdb()

    await _perform_kg_search(
        query="test query",
        ll_keywords=ll_keywords,
        hl_keywords=hl_keywords,
        knowledge_graph_inst=AsyncMock(),
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        text_chunks_db=text_chunks_db,
        query_param=QueryParam(mode=mode, top_k=5),
    )
    return entities_vdb, relationships_vdb, embed_func


def _queried_with(vdb) -> str | None:
    assert vdb.query.call_count <= 1
    return vdb.query.call_args.args[0] if vdb.query.call_count else None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode, ll_keywords, hl_keywords, entity_query, relation_query",
    [
        pytest.param("local", _LL, _HL, _LL, None, id="local"),
        pytest.param("global", _LL, _HL, None, _HL, id="global"),
        pytest.param("hybrid", _LL, _HL, _LL, _HL, id="hybrid"),
        pytest.param("mix", _LL, _HL, _LL, _HL, id="mix"),
        pytest.param("local", "", _HL, _HL, None, id="local-ll-empty"),
        pytest.param("hybrid", "", _HL, _HL, _HL, id="hybrid-ll-empty"),
        pytest.param("mix", "", _HL, _HL, _HL, id="mix-ll-empty"),
        pytest.param("global", "", _HL, None, _HL, id="global-ll-empty"),
        pytest.param("global", _LL, "", None, _LL, id="global-hl-empty"),
        pytest.param("local", _LL, "", _LL, None, id="local-hl-empty"),
        pytest.param("hybrid", _LL, "", _LL, None, id="hybrid-hl-empty"),
        pytest.param("local", "", "", None, None, id="local-both-empty"),
        pytest.param("global", "", "", None, None, id="global-both-empty"),
    ],
)
async def test_kg_search_queries_each_store_with_the_resolved_keywords(
    mode, ll_keywords, hl_keywords, entity_query, relation_query
):
    entities_vdb, relationships_vdb, _ = await _search(mode, ll_keywords, hl_keywords)

    assert _queried_with(entities_vdb) == entity_query
    assert _queried_with(relationships_vdb) == relation_query


@pytest.mark.asyncio
async def test_fallback_shared_by_both_stores_is_embedded_once():
    entities_vdb, relationships_vdb, embed_func = await _search("hybrid", "", _HL)

    assert embed_func.call_count == 1
    assert embed_func.call_args.args[0] == ["test query", _HL]
    entity_embedding = entities_vdb.query.call_args.kwargs["query_embedding"]
    relation_embedding = relationships_vdb.query.call_args.kwargs["query_embedding"]
    assert entity_embedding is not None
    assert np.array_equal(entity_embedding, relation_embedding)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode, ll_keywords, hl_keywords, expected",
    [
        pytest.param("local", "", _HL, "querying entities", id="entity-fallback"),
        pytest.param(
            "global", _LL, "", "querying relationships", id="relation-fallback"
        ),
    ],
)
async def test_fallback_is_logged(
    mode, ll_keywords, hl_keywords, expected, lightrag_log_records
):
    await _search(mode, ll_keywords, hl_keywords)

    warnings = [
        record.getMessage()
        for record in lightrag_log_records
        if record.levelname == "WARNING"
    ]
    assert sum(expected in message for message in warnings) == 1
