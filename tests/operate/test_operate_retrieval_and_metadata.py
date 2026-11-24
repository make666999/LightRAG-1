"""Tests for operate retrieval error propagation and keyword metadata.

Covers:
- Query metadata preserving multi-word keywords that contain commas verbatim
  instead of splitting on ``", "``.
- ``_get_vector_context`` propagating backend errors rather than turning them
  into empty result lists, while keeping confirmed-empty and hit paths intact.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lightrag import operate
from lightrag.base import QueryParam
from lightrag.operate import _get_vector_context
from lightrag.utils import Tokenizer

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Keyword metadata preservation
# ---------------------------------------------------------------------------


class _Characters:
    """One token per character: no encoding data to load."""

    def encode(self, content):
        return list(map(ord, content))

    def decode(self, tokens):
        return "".join(map(chr, tokens))


async def test_keyword_containing_a_comma_is_reported_whole(monkeypatch):
    high_level = ["Treaty negotiations, 1871"]
    low_level = ["Washington, D.C.", "Hamilton Fish"]
    keyword_model = AsyncMock(
        return_value=json.dumps(
            {"high_level_keywords": high_level, "low_level_keywords": low_level}
        )
    )
    config = {
        "tokenizer": Tokenizer("characters", _Characters()),
        "role_llm_funcs": {"query": AsyncMock(), "keyword": keyword_model},
    }
    search = {
        "final_entities": [
            {
                "entity_name": "Hamilton Fish",
                "entity_type": "person",
                "description": "Secretary of State",
                "source_id": "chunk-1",
            }
        ],
        "final_relations": [],
        "vector_chunks": [],
        "chunk_tracking": {},
        "query_embedding": None,
    }
    monkeypatch.setattr(operate, "_perform_kg_search", AsyncMock(return_value=search))
    monkeypatch.setattr(operate, "_merge_all_chunks", AsyncMock(return_value=[]))

    result = await operate.kg_query(
        "Who negotiated the treaty signed in Washington?",
        None,
        None,
        None,
        SimpleNamespace(global_config=config),
        QueryParam(mode="hybrid", only_need_context=True, enable_rerank=False),
        config,
    )

    assert result.raw_data["metadata"]["keywords"] == {
        "high_level": high_level,
        "low_level": low_level,
    }


# ---------------------------------------------------------------------------
# _get_vector_context error propagation
# ---------------------------------------------------------------------------


class _FailingVDB:
    cosine_better_than_threshold = 0.2

    async def query(self, *args, **kwargs):
        raise RuntimeError("vector backend unavailable")


class _EmptyVDB:
    cosine_better_than_threshold = 0.2

    async def query(self, *args, **kwargs):
        return []


class _HitVDB:
    cosine_better_than_threshold = 0.2

    async def query(self, *args, **kwargs):
        return [{"content": "hello", "id": "chunk-1", "file_path": "a.txt"}]


async def test_get_vector_context_propagates_backend_error():
    """A transport/backend failure must raise, not be reported as no-results."""
    with pytest.raises(RuntimeError, match="vector backend unavailable"):
        await _get_vector_context(
            "query", _FailingVDB(), QueryParam(), query_embedding=[0.1]
        )


async def test_get_vector_context_confirmed_empty_returns_empty_list():
    """Genuinely zero hits stay a plain empty list, not an error."""
    result = await _get_vector_context(
        "query", _EmptyVDB(), QueryParam(), query_embedding=[0.1]
    )
    assert result == []


async def test_get_vector_context_happy_path_unaffected():
    result = await _get_vector_context(
        "query", _HitVDB(), QueryParam(), query_embedding=[0.1]
    )
    assert len(result) == 1
    assert result[0]["content"] == "hello"
    assert result[0]["chunk_id"] == "chunk-1"
