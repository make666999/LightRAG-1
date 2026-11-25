"""Tests for merge description deduplication, sanitization, empty-summary guard, and map-reduce perf.

Covers:
1. ``_combine_descriptions_dedup`` and merge round-trips preventing duplicate
   description accumulation across reprocesses (#3367), plus entity-type voting.
2. XML/GraphML control-char sanitization across LLM summaries, single-item early
   return, and no-LLM join, plus ``EmptySummaryResponseError`` on empty LLM
   summary output.
3. Map-reduce description chunking behavior and single-pass tokenization memoization.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import networkx as nx
import pytest

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.exceptions import EmptySummaryResponseError
from lightrag.operate import (
    _combine_descriptions_dedup,
    _handle_entity_relation_summary,
    _merge_edges_then_upsert,
    _merge_nodes_then_upsert,
    _summarize_descriptions,
)
from lightrag.utils import Tokenizer

from ._helpers import (
    CountingTokenizer,
    MemGraph,
    MemKV,
    make_dummy_tokenizer,
    make_merge_config,
)

pytestmark = pytest.mark.offline


def _node_fragments(graph: MemGraph, name: str) -> list[str]:
    return graph.nodes[name]["description"].split(GRAPH_FIELD_SEP)


def _edge_fragments(graph: MemGraph, src: str, tgt: str) -> list[str]:
    return graph.edges[(src, tgt)]["description"].split(GRAPH_FIELD_SEP)


# --------------------------------------------------------------------------- #
# 1. Description deduplication & entity-type voting (#3367)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("stored", "new", "expected_combined", "expected_already"),
    [
        pytest.param(["A", "B"], ["B", "C"], ["A", "B", "C"], 2, id="cross_boundary"),
        pytest.param(
            ["A", "A", "A"], ["A"], ["A"], 1, id="collapses_legacy_stored_duplicates"
        ),
        pytest.param(
            ["Alice is a software engineer at Acme."],
            ["Alice\x08 is a software engineer at Acme."],
            ["Alice is a software engineer at Acme."],
            1,
            id="sanitizes_before_compare",
        ),
        pytest.param(
            ["Bob\x08 builds pipelines.", "Bob builds pipelines."],
            ["Bob builds pipelines."],
            ["Bob builds pipelines."],
            1,
            id="collapses_dirty_stored_fragments",
        ),
        pytest.param(
            ["A", "\x08"],
            ["\x00", "B"],
            ["A", "B"],
            1,
            id="drops_empty_after_sanitize",
        ),
    ],
)
def test_combine_descriptions_dedup(stored, new, expected_combined, expected_already):
    combined, already = _combine_descriptions_dedup(stored, new)
    assert combined == expected_combined
    assert already == expected_already


@pytest.mark.asyncio
async def test_node_reprocess_with_dirty_char_does_not_accumulate():
    """A re-extracted description with an XML-illegal control char dedups against
    the sanitized stored copy instead of accumulating on reprocess."""
    graph = MemGraph()
    cfg = make_merge_config()
    clean = "Alice is a software engineer at Acme."
    dirty = "Alice\x08 is a software engineer at Acme."
    base = {
        "entity_name": "ALICE",
        "entity_type": "person",
        "source_id": "chunk-1",
        "file_path": "doc1.txt",
        "timestamp": 1,
    }

    await _merge_nodes_then_upsert(
        "ALICE", [dict(base, description=clean)], graph, None, cfg
    )
    await _merge_nodes_then_upsert(
        "ALICE", [dict(base, description=dirty, timestamp=2)], graph, None, cfg
    )

    assert _node_fragments(graph, "ALICE") == [clean]


@pytest.mark.asyncio
async def test_node_reprocess_does_not_accumulate_description():
    graph = MemGraph()
    cfg = make_merge_config()
    batch = {
        "entity_name": "ALICE",
        "entity_type": "person",
        "description": "Alice is a software engineer at Acme.",
        "source_id": "chunk-1",
        "file_path": "doc1.txt",
        "timestamp": 1,
    }

    counts = []
    for _ in range(3):
        await _merge_nodes_then_upsert("ALICE", [dict(batch)], graph, None, cfg)
        counts.append(len(_node_fragments(graph, "ALICE")))

    assert counts == [1, 1, 1]
    assert _node_fragments(graph, "ALICE") == ["Alice is a software engineer at Acme."]


@pytest.mark.asyncio
async def test_node_merge_keeps_distinct_descriptions():
    """Negative twin: genuinely different descriptions are still accumulated."""
    graph = MemGraph()
    cfg = make_merge_config()
    base = {
        "entity_name": "ALICE",
        "entity_type": "person",
        "source_id": "chunk-1",
        "file_path": "doc1.txt",
        "timestamp": 1,
    }

    await _merge_nodes_then_upsert(
        "ALICE", [dict(base, description="Alice is an engineer.")], graph, None, cfg
    )
    await _merge_nodes_then_upsert(
        "ALICE",
        [dict(base, description="Alice leads the platform team.", timestamp=2)],
        graph,
        None,
        cfg,
    )

    frags = _node_fragments(graph, "ALICE")
    assert set(frags) == {
        "Alice is an engineer.",
        "Alice leads the platform team.",
    }
    assert len(frags) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stored_type", "new_types", "expected"),
    [
        pytest.param("person", ["organization"], "person", id="tie-keeps-stored"),
        pytest.param(
            "person", ["organization", "organization"], "organization", id="majority"
        ),
        pytest.param("UNKNOWN", ["person"], "person", id="placeholder-yields"),
    ],
)
async def test_node_merge_entity_type_vote(stored_type, new_types, expected):
    graph = MemGraph(
        nodes={
            "ALICE": {
                "entity_id": "ALICE",
                "entity_type": stored_type,
                "description": "Alice.",
                "source_id": "chunk-0",
                "file_path": "doc0.txt",
            }
        }
    )
    nodes_data = [
        {
            "entity_name": "ALICE",
            "entity_type": entity_type,
            "description": f"Alice {i}.",
            "source_id": f"chunk-{i}",
            "file_path": "doc1.txt",
            "timestamp": i,
        }
        for i, entity_type in enumerate(new_types, start=1)
    ]

    await _merge_nodes_then_upsert(
        "ALICE", nodes_data, graph, None, make_merge_config()
    )

    assert graph.nodes["ALICE"]["entity_type"] == expected


@pytest.mark.asyncio
async def test_edge_reprocess_does_not_accumulate_description():
    graph = MemGraph(
        nodes={
            "ALICE": {
                "entity_type": "person",
                "description": "x",
                "source_id": "chunk-1",
                "file_path": "doc1.txt",
            },
            "ACME": {
                "entity_type": "organization",
                "description": "y",
                "source_id": "chunk-1",
                "file_path": "doc1.txt",
            },
        }
    )
    cfg = make_merge_config()
    batch = {
        "src_id": "ALICE",
        "tgt_id": "ACME",
        "description": "Alice works at Acme.",
        "keywords": "employment",
        "weight": 1.0,
        "source_id": "chunk-1",
        "file_path": "doc1.txt",
        "timestamp": 1,
    }

    counts = []
    for _ in range(3):
        await _merge_edges_then_upsert(
            "ALICE", "ACME", [dict(batch)], graph, None, None, cfg
        )
        counts.append(len(_edge_fragments(graph, "ALICE", "ACME")))

    assert counts == [1, 1, 1]
    assert _edge_fragments(graph, "ALICE", "ACME") == ["Alice works at Acme."]


# --------------------------------------------------------------------------- #
# 2. Summary sanitization & empty-summary rejection
# --------------------------------------------------------------------------- #

_DIRTY_SUMMARY = "Clean text\x00with\x07control\x1fchars\x7f\tand\nnewlines"
_EXPECTED_CLEAN = "Clean textwithcontrolchars\tand\nnewlines"


def _make_summary_config(*summary_returns: str) -> dict:
    extract_func = AsyncMock(side_effect=list(summary_returns))
    return {
        "role_llm_funcs": {"extract": extract_func},
        "addon_params": {},
        "_resolved_summary_language": "English",
        "summary_length_recommended": 100,
        "summary_context_size": 10_000,
        "tokenizer": make_dummy_tokenizer(),
    }


async def _summarize(global_config: dict, cache=None) -> str:
    return await _summarize_descriptions(
        "Entity",
        "TEST_ENTITY",
        ["first description", "second description"],
        global_config,
        llm_response_cache=cache,
    )


@pytest.mark.asyncio
async def test_summarized_description_is_clean_and_graphml_writable(tmp_path):
    """LLM summary strips XML-incompatible control chars while keeping tabs/newlines."""
    description = await _summarize(_make_summary_config(_DIRTY_SUMMARY))

    assert description == _EXPECTED_CLEAN

    graph = nx.Graph()
    graph.add_node("TEST_ENTITY", description=description)
    out = tmp_path / "graph.graphml"
    nx.write_graphml(graph, out)
    assert nx.read_graphml(out).nodes["TEST_ENTITY"]["description"] == _EXPECTED_CLEAN


@pytest.mark.asyncio
async def test_single_description_early_return_is_sanitized():
    """Single-description early return skips the LLM yet still sanitizes."""
    description, llm_was_used = await _handle_entity_relation_summary(
        "Entity",
        "TEST_ENTITY",
        [_DIRTY_SUMMARY],
        "<SEP>",
        global_config={},
        llm_response_cache=None,
    )

    assert llm_was_used is False
    assert description == _EXPECTED_CLEAN


@pytest.mark.asyncio
async def test_join_without_llm_is_sanitized():
    """The no-LLM join path sanitizes descriptions read back from graph nodes."""
    global_config = {
        "tokenizer": make_dummy_tokenizer(),
        "summary_context_size": 100_000,
        "summary_max_tokens": 100_000,
        "force_llm_summary_on_merge": 10,
    }

    description, llm_was_used = await _handle_entity_relation_summary(
        "Entity",
        "TEST_ENTITY",
        [_DIRTY_SUMMARY, "clean second description"],
        "<SEP>",
        global_config,
        llm_response_cache=None,
    )

    assert llm_was_used is False
    assert description == f"{_EXPECTED_CLEAN}<SEP>clean second description"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "empty_answer, retry_reaches_llm",
    [
        pytest.param("", True, id="empty"),
        pytest.param("<think>reasoning, no answer</think>", True, id="think-only"),
        # Accepted residue: cached before sanitization empties it, so the
        # retry replays it from the cache and fails again.
        pytest.param("\x00\x07", False, id="control-chars-only"),
    ],
)
async def test_empty_summary_fails_and_retry_asks_the_llm_again(
    empty_answer, retry_reaches_llm
):
    """Empty summaries raise EmptySummaryResponseError instead of erasing stored text."""
    global_config = _make_summary_config(empty_answer, "Recovered summary.")
    llm = global_config["role_llm_funcs"]["extract"]
    cache = MemKV(global_config={"enable_llm_cache_for_entity_extract": True})

    with pytest.raises(EmptySummaryResponseError, match="Entity:TEST_ENTITY"):
        await _summarize(global_config, cache)

    if retry_reaches_llm:
        assert await _summarize(global_config, cache) == "Recovered summary."
        assert llm.await_count == 2
    else:
        with pytest.raises(EmptySummaryResponseError):
            await _summarize(global_config, cache)
        assert llm.await_count == 1


# --------------------------------------------------------------------------- #
# 3. Map-reduce chunking behavior & tokenization memoization
# --------------------------------------------------------------------------- #


def _make_chunking_config(mock_extract) -> tuple[dict, CountingTokenizer]:
    """Config whose summary_context_size forces map-reduce chunking (4x5 > 10)."""
    counter = CountingTokenizer()
    global_config = {
        "role_llm_funcs": {"extract": mock_extract},
        "addon_params": {},
        "_resolved_summary_language": "English",
        "summary_length_recommended": 100,
        "summary_context_size": 10,
        "summary_max_tokens": 100_000,
        "force_llm_summary_on_merge": 10,
        "tokenizer": Tokenizer("dummy", counter),
    }
    return global_config, counter


@pytest.mark.asyncio
async def test_map_reduce_chunking_is_preserved():
    """Memoizing token counts keeps map-reduce group boundaries and joined output."""
    mock_extract = AsyncMock(side_effect=["SUM_ONE", "SUM_TWO", "SUM_THREE"])
    global_config, _ = _make_chunking_config(mock_extract)

    description, llm_was_used = await _handle_entity_relation_summary(
        "Entity",
        "TEST_ENTITY",
        ["alpha", "beta", "gamma", "delta"],
        "<SEP>",
        global_config,
        llm_response_cache=None,
    )

    assert llm_was_used is True
    assert description == "SUM_ONE<SEP>SUM_TWO"


@pytest.mark.asyncio
async def test_map_phase_encodes_each_description_once():
    """The map phase tokenizes each input description once per iteration."""
    mock_extract = AsyncMock(side_effect=["SUM_ONE", "SUM_TWO", "SUM_THREE"])
    global_config, counter = _make_chunking_config(mock_extract)
    inputs = ["alpha", "beta", "gamma", "delta"]

    await _handle_entity_relation_summary(
        "Entity",
        "TEST_ENTITY",
        inputs,
        "<SEP>",
        global_config,
        llm_response_cache=None,
    )

    for desc in inputs:
        assert counter.encoded.count(desc) == 1, (
            f"description {desc!r} was encoded {counter.encoded.count(desc)} "
            "time(s) by the map phase; expected exactly 1 (memoized counts)"
        )
