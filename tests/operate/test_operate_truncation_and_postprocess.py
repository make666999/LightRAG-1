"""Tests for stage-2 truncation ordering and query answer postprocessing.

Covers:
- Stage-2 truncation order propagating intact into stage-3 ``filtered_*`` lists
  and chunk quota allocation.
- Non-streaming ``kg_query`` and ``naive_query`` returning long LLM and cached
  answers verbatim without legacy prompt-echo string replacements.
"""

from __future__ import annotations

import logging

import pytest

from lightrag.base import QueryContextResult, QueryParam
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.operate import (
    _apply_token_truncation,
    _find_related_text_unit_from_entities,
    kg_query,
    naive_query,
)
from lightrag.utils import Tokenizer, logger

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Stage-2 truncation order propagation
# ---------------------------------------------------------------------------

GLOBAL_CONFIG = {"tokenizer": object()}


def _entity(name: str, **extra) -> dict:
    record = {
        "entity_name": name,
        "entity_type": "person",
        "description": f"description of {name}",
        "file_path": f"{name}.txt",
    }
    record.update(extra)
    return record


def _relation(src: str, tgt: str, **extra) -> dict:
    record = {
        "src_id": src,
        "tgt_id": tgt,
        "description": f"{src} -> {tgt}",
        "file_path": f"{src}-{tgt}.txt",
    }
    record.update(extra)
    return record


def _search_result(entities: list[dict], relations: list[dict]) -> dict:
    return {"final_entities": entities, "final_relations": relations}


def _install_selector(monkeypatch, selector):
    """Replace the truncation step with ``selector(list_data) -> list_data``."""

    async def fake_atruncate_list_by_token_size(list_data, **kwargs):
        return selector(list_data)

    monkeypatch.setattr(
        "lightrag.operate.atruncate_list_by_token_size",
        fake_atruncate_list_by_token_size,
    )


def _names(entities: list[dict]) -> list[str]:
    return [e["entity_name"] for e in entities]


def _pairs(relations: list[dict]) -> list[tuple[str, str]]:
    return [(r["src_id"], r["tgt_id"]) for r in relations]


async def test_reordering_selector_propagates_into_filtered_lists(monkeypatch):
    """A stage-2 reorder must show up in ``filtered_*``, not just the prompt."""
    _install_selector(monkeypatch, lambda list_data: list(reversed(list_data)))

    result = await _apply_token_truncation(
        _search_result(
            [_entity("alice"), _entity("bob"), _entity("carol")],
            [_relation("alice", "bob"), _relation("bob", "carol")],
        ),
        QueryParam(),
        GLOBAL_CONFIG,
    )

    assert [e["entity"] for e in result["entities_context"]] == [
        "carol",
        "bob",
        "alice",
    ]
    assert _names(result["filtered_entities"]) == ["carol", "bob", "alice"]

    assert [(r["entity1"], r["entity2"]) for r in result["relations_context"]] == [
        ("bob", "carol"),
        ("alice", "bob"),
    ]
    assert _pairs(result["filtered_relations"]) == [
        ("bob", "carol"),
        ("alice", "bob"),
    ]


async def test_reordering_selector_that_also_drops_records(monkeypatch):
    """Reorder plus selection: order follows stage 2, membership follows it too."""
    _install_selector(monkeypatch, lambda list_data: [list_data[2], list_data[0]])

    result = await _apply_token_truncation(
        _search_result(
            [_entity("alice"), _entity("bob"), _entity("carol")],
            [
                _relation("alice", "bob"),
                _relation("bob", "carol"),
                _relation("carol", "dave"),
            ],
        ),
        QueryParam(),
        GLOBAL_CONFIG,
    )

    assert _names(result["filtered_entities"]) == ["carol", "alice"]
    assert _pairs(result["filtered_relations"]) == [
        ("carol", "dave"),
        ("alice", "bob"),
    ]
    assert list(result["entity_id_to_original"]) == ["carol", "alice"]
    assert list(result["relation_id_to_original"]) == [
        ("carol", "dave"),
        ("alice", "bob"),
    ]


async def test_prefix_truncation_is_unchanged(monkeypatch):
    """The shipped default path -- a prefix -- keeps stage-1 order, as before."""
    _install_selector(monkeypatch, lambda list_data: list_data[:2])

    result = await _apply_token_truncation(
        _search_result(
            [_entity("alice"), _entity("bob"), _entity("carol")],
            [
                _relation("alice", "bob"),
                _relation("bob", "carol"),
                _relation("carol", "dave"),
            ],
        ),
        QueryParam(),
        GLOBAL_CONFIG,
    )

    assert _names(result["filtered_entities"]) == ["alice", "bob"]
    assert _pairs(result["filtered_relations"]) == [
        ("alice", "bob"),
        ("bob", "carol"),
    ]


async def test_non_prefix_subset_without_reorder(monkeypatch):
    """A packing selector that only drops records still resolves correctly."""
    _install_selector(monkeypatch, lambda list_data: [list_data[0], list_data[2]])

    result = await _apply_token_truncation(
        _search_result(
            [_entity("alice"), _entity("bob"), _entity("carol")],
            [],
        ),
        QueryParam(),
        GLOBAL_CONFIG,
    )

    assert _names(result["filtered_entities"]) == ["alice", "carol"]
    assert result["filtered_relations"] == []
    assert result["relations_context"] == []


async def test_duplicate_names_keep_the_first_retrieved_record(monkeypatch):
    """Dedup resolves to the first occurrence in stage-1 order, as it always did.

    Stage 2 decides *order*; it does not get to promote a later duplicate of a
    name over the record that was actually retrieved first.
    """
    _install_selector(monkeypatch, lambda list_data: list(reversed(list_data)))

    result = await _apply_token_truncation(
        _search_result(
            [
                _entity("alice", description="first alice"),
                _entity("bob"),
                _entity("alice", description="second alice"),
            ],
            [
                _relation("alice", "bob", description="first edge"),
                _relation("alice", "bob", description="second edge"),
            ],
        ),
        QueryParam(),
        GLOBAL_CONFIG,
    )

    assert _names(result["filtered_entities"]) == ["alice", "bob"]
    assert result["filtered_entities"][0]["description"] == "first alice"
    assert result["entity_id_to_original"]["alice"]["description"] == "first alice"

    assert _pairs(result["filtered_relations"]) == [("alice", "bob")]
    assert result["filtered_relations"][0]["description"] == "first edge"


async def test_src_tgt_relation_format_survives_reordering(monkeypatch):
    """Relations carrying ``src_tgt`` instead of ``src_id``/``tgt_id`` resolve too."""
    _install_selector(monkeypatch, lambda list_data: list(reversed(list_data)))

    result = await _apply_token_truncation(
        _search_result(
            [],
            [
                {"src_tgt": ("alice", "bob"), "description": "a-b"},
                {"src_tgt": ("bob", "carol"), "description": "b-c"},
            ],
        ),
        QueryParam(),
        GLOBAL_CONFIG,
    )

    assert [r["src_tgt"] for r in result["filtered_relations"]] == [
        ("bob", "carol"),
        ("alice", "bob"),
    ]
    assert result["filtered_entities"] == []
    assert result["entities_context"] == []


async def test_missing_tokenizer_short_circuits(monkeypatch):
    """No tokenizer: nothing is truncated and the originals pass straight through."""
    entities = [_entity("alice")]
    relations = [_relation("alice", "bob")]

    result = await _apply_token_truncation(
        _search_result(entities, relations),
        QueryParam(),
        {},
    )

    assert result["entities_context"] == []
    assert result["relations_context"] == []
    assert result["filtered_entities"] is entities
    assert result["filtered_relations"] is relations


async def test_unresolved_context_record_is_dropped_and_reported(
    monkeypatch, lightrag_caplog
):
    """A record a selector invents cannot reach chunk selection -- warn, don't hide."""
    _install_selector(
        monkeypatch,
        lambda list_data: [{**list_data[0], "entity": "ghost"}] + list_data,
    )

    with lightrag_caplog.at_level(logging.WARNING, logger=logger.name):
        result = await _apply_token_truncation(
            _search_result([_entity("alice"), _entity("bob")], []),
            QueryParam(),
            GLOBAL_CONFIG,
        )

    assert [e["entity"] for e in result["entities_context"]][0] == "ghost"
    assert _names(result["filtered_entities"]) == ["alice", "bob"]
    assert "ghost" in lightrag_caplog.text
    assert "absent from the pre-truncation map" in lightrag_caplog.text


class _FakeTextChunksDB:
    """Minimal stand-in for the chunk KV storage on the WEIGHT pick path."""

    def __init__(self, global_config: dict):
        self.global_config = global_config
        self.embedding_func = None

    async def get_by_ids(self, ids: list[str]) -> list[dict]:
        return [{"content": f"content of {chunk_id}"} for chunk_id in ids]


async def test_reordering_reaches_stage_3_chunk_quota(monkeypatch):
    """The end of the chain: stage-2 order changes which chunks survive stage 3.

    ``pick_by_weighted_polling`` hands position 0 the full
    ``related_chunk_number`` quota and the last position one chunk, so reversing
    stage 2's order must reverse which entity's chunks get the budget. Three
    entities with three disjoint chunks each and a quota of 3 give an exact
    ``[3, 2, 1]`` split with no second-round redistribution to blur the result.
    """
    _install_selector(monkeypatch, lambda list_data: list(reversed(list_data)))

    truncation = await _apply_token_truncation(
        _search_result(
            [
                _entity(name, source_id=GRAPH_FIELD_SEP.join(chunks))
                for name, chunks in (
                    ("alice", ["a1", "a2", "a3"]),
                    ("bob", ["b1", "b2", "b3"]),
                    ("carol", ["c1", "c2", "c3"]),
                )
            ],
            [],
        ),
        QueryParam(),
        GLOBAL_CONFIG,
    )

    chunks = await _find_related_text_unit_from_entities(
        truncation["filtered_entities"],
        QueryParam(),
        _FakeTextChunksDB(
            {"kg_chunk_pick_method": "WEIGHT", "related_chunk_number": 3}
        ),
        None,  # knowledge_graph_inst is unused on the WEIGHT path
    )

    # Reranked order carol > bob > alice, so carol takes 3 chunks and alice 1.
    # Stage-1 order would have produced ["a1", "a2", "a3", "b1", "b2", "c1"].
    assert [chunk["chunk_id"] for chunk in chunks] == [
        "c1",
        "c2",
        "c3",
        "b1",
        "b2",
        "a1",
    ]


# ---------------------------------------------------------------------------
# Non-streaming query answer postprocessing
# ---------------------------------------------------------------------------


class _CharTokenizer:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


class _ChunksVDB:
    cosine_better_than_threshold = 0.0

    async def query(self, *_args, **_kwargs):
        return [
            {
                "id": "chunk-1",
                "content": "The user manual describes the model settings.",
                "file_path": "manual.md",
            }
        ]


class _TextChunks:
    def __init__(self, global_config):
        self.global_config = global_config


# Longer than any system prompt these tests render (~3.3k chars), and it
# contains "user" / "model" as parts of ordinary words. The query text is kept
# out of it so the assertions do not depend on `.replace(query, "")`.
LONG_ANSWER = (
    "The user manual explains how each model is configured. "
    + "Every setting is described step by step for the operator. " * 80
    + "Finally, users can reset the model to its defaults."
)

ANSWER_QUERY = "How is it configured?"


class _AnswerCacheKV:
    """Answer-cache storage that hits on every key with ``answer``."""

    def __init__(self, answer: str):
        self.global_config = {"enable_llm_cache": True}
        self._answer = answer

    async def get_by_id(self, _key):
        return {"return": self._answer, "create_time": 0}


def _answer_config(answer: str, seen: dict) -> dict:
    async def query_model(*_args, system_prompt=None, **_kwargs):
        seen["sys_prompt"] = system_prompt or ""
        return answer

    return {
        "tokenizer": Tokenizer("fake", _CharTokenizer()),
        "role_llm_funcs": {"query": query_model},
        "addon_params": {"language": "en"},
        "min_rerank_score": 0.0,
        "max_total_tokens": 30000,
        "related_chunk_number": 1,
        "kg_chunk_pick_method": "WEIGHT",
    }


async def _run_query(kind: str, monkeypatch, config: dict, hashing_kv=None):
    if kind == "naive":
        return await naive_query(
            ANSWER_QUERY,
            _ChunksVDB(),
            QueryParam(mode="naive", enable_rerank=False, stream=False),
            config,
            hashing_kv=hashing_kv,
        )

    async def fake_keywords(*_args, **_kwargs):
        return ["configuration"], ["model settings"]

    async def fake_context(*_args, **_kwargs):
        return QueryContextResult(context="small context", raw_data={})

    monkeypatch.setattr("lightrag.operate.get_keywords_from_query", fake_keywords)
    monkeypatch.setattr("lightrag.operate._build_query_context", fake_context)
    return await kg_query(
        ANSWER_QUERY,
        None,
        None,
        None,
        _TextChunks(config),
        QueryParam(mode="mix", enable_rerank=False, stream=False),
        config,
        hashing_kv=hashing_kv,
    )


def _assert_old_block_would_fire(seen: dict) -> None:
    # The removed block only ran when the answer outgrew the rendered system
    # prompt. If prompt templates ever grow past LONG_ANSWER, this test would
    # pass even with the block restored, so pin the precondition explicitly.
    assert seen["sys_prompt"], "stub LLM did not receive a system prompt"
    assert len(LONG_ANSWER) > len(seen["sys_prompt"]), (
        "LONG_ANSWER no longer outgrows the system prompt; lengthen it so "
        "this test still guards the removed clean-up block"
    )


@pytest.mark.parametrize("kind", ["naive", "kg"])
async def test_query_returns_long_answer_unmodified(kind, monkeypatch):
    seen: dict = {}
    result = await _run_query(kind, monkeypatch, _answer_config(LONG_ANSWER, seen))

    _assert_old_block_would_fire(seen)
    assert result.content == LONG_ANSWER


@pytest.mark.parametrize("kind", ["naive", "kg"])
async def test_query_cache_hit_returns_cached_answer_unmodified(kind, monkeypatch):
    # The cache stores the raw answer and the clean-up block used to run on
    # every hit as well. Render the prompt once without a cache to prove the
    # precondition, then serve the same request from the cache.
    seen: dict = {}
    await _run_query(kind, monkeypatch, _answer_config(LONG_ANSWER, seen))
    _assert_old_block_would_fire(seen)

    calls: list[tuple] = []

    async def must_not_call(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("cache hit must not call the LLM")

    config = _answer_config(LONG_ANSWER, {})
    config["role_llm_funcs"] = {"query": must_not_call}
    result = await _run_query(
        kind, monkeypatch, config, hashing_kv=_AnswerCacheKV(LONG_ANSWER)
    )

    assert calls == []
    assert result.content == LONG_ANSWER
