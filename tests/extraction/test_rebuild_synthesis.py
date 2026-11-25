"""Tests for KG rebuild synthesis, history scoping, fallback parsing, and multimodal evidence."""

from __future__ import annotations

import asyncio
import inspect
import logging

import pytest

import lightrag.operate as operate
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.operate import (
    _rebuild_from_extraction_result,
    _rebuild_single_entity,
    _rebuild_single_relationship,
    rebuild_knowledge_from_chunks,
)

from ._helpers import MemGraph, MemKV, MemVDB, make_merge_config

pytestmark = pytest.mark.offline

_DELIMITER_WARNING = "Complete delimiter can not be found"
_JSON_WARNING = "JSON extraction result is empty or unrecoverable"


@pytest.fixture(autouse=True)
def _shared_data():
    """Ensure the single-process shared-data registry exists for keyed locks."""
    initialize_share_data()


# --------------------------------------------------------------------------- #
# Single-entity & single-relationship rebuild synthesis
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_rebuild_single_entity_most_common_type():
    graph = MemGraph()
    global_config = make_merge_config(
        max_source_ids_per_entity=10,
        max_file_paths=10,
    )
    await graph.upsert_node(
        "ALICE",
        {
            "description": "Initial description",
            "entity_type": "UNKNOWN",
            "source_id": "c1",
            "file_path": "f1.txt",
        },
    )

    chunk_entities = {
        "c1": {
            "ALICE": [{"entity_type": "PERSON", "description": "Alice is a person"}]
        },
        "c2": {
            "ALICE": [{"entity_type": "ORGANIZATION", "description": "Alice Org 1"}]
        },
        "c3": {
            "ALICE": [{"entity_type": "ORGANIZATION", "description": "Alice Org 2"}]
        },
        "c4": {
            "ALICE": [{"entity_type": "ORGANIZATION", "description": "Alice Org 3"}]
        },
    }

    await _rebuild_single_entity(
        knowledge_graph_inst=graph,
        entities_vdb=MemVDB(),
        entity_name="ALICE",
        chunk_ids=["c1", "c2", "c3", "c4"],
        chunk_entities=chunk_entities,
        llm_response_cache=MemKV(),
        global_config=global_config,
    )

    updated_node = await graph.get_node("ALICE")
    assert updated_node["entity_type"] == "ORGANIZATION"


@pytest.mark.asyncio
async def test_rebuild_single_relationship_keyword_deduplication_and_formatting():
    graph = MemGraph()
    global_config = make_merge_config(
        max_source_ids_per_relation=10,
        max_file_paths=10,
    )
    await graph.upsert_node("A", {"entity_type": "ENT", "description": "Node A"})
    await graph.upsert_node("B", {"entity_type": "ENT", "description": "Node B"})
    await graph.upsert_edge(
        "A",
        "B",
        {
            "description": "Relation AB",
            "keywords": "legacy_kw",
            "weight": 1.0,
            "source_id": "c1",
            "file_path": "f1.txt",
        },
    )

    chunk_relationships = {
        "c1": {
            ("A", "B"): [
                {"keywords": "AI, Machine Learning", "description": "rel desc 1"}
            ]
        },
        "c2": {
            ("A", "B"): [
                {
                    "keywords": "Machine Learning, Deep Learning",
                    "description": "rel desc 2",
                }
            ]
        },
    }

    await _rebuild_single_relationship(
        knowledge_graph_inst=graph,
        relationships_vdb=MemVDB(),
        entities_vdb=MemVDB(),
        src="A",
        tgt="B",
        chunk_ids=["c1", "c2"],
        chunk_relationships=chunk_relationships,
        llm_response_cache=MemKV(),
        global_config=global_config,
    )

    updated_edge = await graph.get_edge("A", "B")
    assert updated_edge["keywords"] == "AI, Deep Learning, Machine Learning"


@pytest.mark.asyncio
async def test_rebuild_single_entity_from_edges_takes_file_paths_from_surviving_chunks():
    """An entity named only as a relation endpoint is re-summarized from its
    edges, while its file_path follows the surviving chunks capped at
    max_file_paths rather than copying edge file_path values."""
    graph = MemGraph()
    global_config = make_merge_config(
        max_source_ids_per_entity=10,
        max_file_paths=2,
        file_path_more_placeholder="more",
    )
    await graph.upsert_node(
        "ALICE",
        {
            "description": "Initial description",
            "entity_type": "PERSON",
            "source_id": "c0",
            "file_path": "deleted.txt",
        },
    )
    for other in ("BOB", "CARL"):
        await graph.upsert_edge(
            "ALICE",
            other,
            {
                "description": f"Alice knows {other}",
                "file_path": GRAPH_FIELD_SEP.join(
                    [f"{other}-1.txt", f"{other}-2.txt", "...more...(KEEP 2/5)"]
                ),
            },
        )

    degraded = await _rebuild_single_entity(
        knowledge_graph_inst=graph,
        entities_vdb=MemVDB(),
        entity_name="ALICE",
        chunk_ids=["c3", "c1", "c2"],
        chunk_entities={},
        llm_response_cache=MemKV(),
        global_config=global_config,
        chunk_data_by_id={
            "c1": {"file_path": "f1.txt"},
            "c2": {"file_path": "f2.txt"},
            "c3": {"file_path": "f3.txt"},
        },
    )

    node = await graph.get_node("ALICE")
    assert degraded is False
    assert node["description"] != "Initial description"
    assert node["source_id"] == GRAPH_FIELD_SEP.join(["c3", "c1", "c2"])
    assert node["file_path"].split(GRAPH_FIELD_SEP) == [
        "f3.txt",
        "f1.txt",
        "...more...(KEEP 2/3)",
    ]


# --------------------------------------------------------------------------- #
# KG-rebuild pipeline history scope
# --------------------------------------------------------------------------- #


class _AbsentGraph:
    """Graph with no nodes/edges: rebuilds return early and succeed."""

    async def get_node(self, _name):
        return None

    async def get_edge(self, _src, _tgt):
        return None


class _FailingNodeGraph(_AbsentGraph):
    async def get_node(self, _name):
        raise RuntimeError("node lookup boom")


class _FailingEdgeGraph(_AbsentGraph):
    async def get_edge(self, _src, _tgt):
        raise RuntimeError("edge lookup boom")


def _rebuild_history_kwargs(**overrides):
    text_chunks = MemKV({"c1": {"content": "x", "llm_cache_list": ["k1"]}})
    llm_cache = MemKV(
        {
            "k1": {
                "cache_type": "extract",
                "chunk_id": "c1",
                "return": "r",
                "create_time": 1,
            }
        }
    )
    kwargs = dict(
        entities_to_rebuild={"ALICE": ["c1"]},
        relationships_to_rebuild={},
        knowledge_graph_inst=_AbsentGraph(),
        entities_vdb=None,
        relationships_vdb=None,
        text_chunks_storage=text_chunks,
        llm_response_cache=llm_cache,
        global_config={"llm_model_max_async": 1},
        pipeline_status={"latest_message": "", "history_messages": []},
        pipeline_status_lock=asyncio.Lock(),
    )
    kwargs.update(overrides)
    return kwargs


async def _boom_parse(**_kwargs):
    raise ValueError("bad cache")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "overrides", "failure_fragment"),
    [
        (
            "parse_failure",
            {"rebuild_policy": "rollback"},
            "Failed to parse cached extraction",
        ),
        (
            "entity_failure",
            {"knowledge_graph_inst": _FailingNodeGraph()},
            "Failed to rebuild `ALICE`",
        ),
        (
            "relationship_failure",
            {
                "entities_to_rebuild": {},
                "relationships_to_rebuild": {("A", "B"): ["c1"]},
                "knowledge_graph_inst": _FailingEdgeGraph(),
            },
            "Failed to rebuild `A`~`B`",
        ),
    ],
)
async def test_per_item_rebuild_failures_reach_history_but_completions_do_not(
    monkeypatch, case, overrides, failure_fragment
):
    """Per-item failures stay in pipeline history; per-item completions do not."""
    if case == "parse_failure":
        monkeypatch.setattr(operate, "_rebuild_from_extraction_result", _boom_parse)

    kwargs = _rebuild_history_kwargs(**overrides)
    history = kwargs["pipeline_status"]["history_messages"]
    if case == "parse_failure":
        await rebuild_knowledge_from_chunks(**kwargs)
        summary_prefix = "KG rebuild completed:"
    else:
        with pytest.raises(RuntimeError, match="lookup boom"):
            await rebuild_knowledge_from_chunks(**kwargs)
        summary_prefix = "KG rebuild failed:"

    assert any(failure_fragment in message for message in history), history
    assert not any("Rebuild `" in message for message in history), history
    assert any(
        message.startswith("Rebuilding knowledge from") for message in history
    ), history
    assert any(
        message.startswith("Starting parallel rebuild") for message in history
    ), history
    assert any(message.startswith(summary_prefix) for message in history), history


@pytest.mark.asyncio
async def test_failure_counts_still_reach_the_summary():
    """The failure summary carries aggregated failure counts into history."""
    kwargs = _rebuild_history_kwargs(knowledge_graph_inst=_FailingNodeGraph())
    history = kwargs["pipeline_status"]["history_messages"]
    with pytest.raises(RuntimeError, match="node lookup boom"):
        await rebuild_knowledge_from_chunks(**kwargs)
    assert any(
        message.startswith("KG rebuild failed:")
        and "Failed: 1 entities, 0 relationships" in message
        for message in history
    ), history


def test_rebuild_helpers_cannot_receive_pipeline_status():
    """Static pin: single-item rebuild helpers do not accept pipeline_status."""
    for helper in (_rebuild_single_entity, _rebuild_single_relationship):
        parameters = inspect.signature(helper).parameters
        assert "pipeline_status" not in parameters
        assert "pipeline_status_lock" not in parameters


# --------------------------------------------------------------------------- #
# Cached payload parsing fallback & warning deduplication
# --------------------------------------------------------------------------- #


class _FilePathKV:
    async def get_by_id(self, _key):
        return {"file_path": "doc.pdf"}


def _warning_messages(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING
    ]


@pytest.mark.asyncio
async def test_unparseable_json_payload_reports_the_failure_exactly_once(
    caplog, lightrag_caplog
) -> None:
    """Speculative delimiter rescue after a failed JSON parse must not emit a
    second WARNING for the missing delimiter."""
    with caplog.at_level(logging.DEBUG, logger="lightrag"):
        nodes, edges = await _rebuild_from_extraction_result(
            _FilePathKV(),
            "{not json at all, no delimiter either",
            chunk_id="chunk-1",
            timestamp=1,
        )

    assert nodes == {}
    assert edges == {}
    warnings = _warning_messages(caplog)
    assert len(warnings) == 1, warnings
    assert _JSON_WARNING in warnings[0]
    assert _DELIMITER_WARNING in caplog.text


@pytest.mark.asyncio
async def test_delimiter_payload_missing_its_terminator_still_warns(
    caplog, lightrag_caplog
) -> None:
    """A genuine delimiter payload missing its terminator still logs a WARNING."""
    with caplog.at_level(logging.WARNING, logger="lightrag"):
        await _rebuild_from_extraction_result(
            _FilePathKV(),
            '("entity"<|#|>Alice<|#|>person<|#|>an engineer)',
            chunk_id="chunk-2",
            timestamp=1,
        )

    warnings = _warning_messages(caplog)
    assert any(_DELIMITER_WARNING in message for message in warnings), warnings


@pytest.mark.asyncio
async def test_json_payload_rescued_by_the_delimiter_parser_keeps_its_records(
    caplog, lightrag_caplog
) -> None:
    """Demoting the fallback warning must not disturb delimiter rescue itself."""
    with caplog.at_level(logging.WARNING, logger="lightrag"):
        nodes, _edges = await _rebuild_from_extraction_result(
            _FilePathKV(),
            '{}\n("entity"<|#|>Alice<|#|>person<|#|>an engineer)\n<|COMPLETE|>',
            chunk_id="chunk-3",
            timestamp=1,
        )

    assert "Alice" in nodes


@pytest.mark.asyncio
async def test_wrong_schema_json_payload_never_reaches_the_fallback(
    monkeypatch, caplog, lightrag_caplog
) -> None:
    """A valid JSON payload carrying no extraction fields returns early without
    invoking the delimiter parser or warning."""
    calls: list[str] = []

    async def _spy(result, chunk_key, *args, **kwargs):
        calls.append(chunk_key)
        return {}, {}

    monkeypatch.setattr(operate, "_process_extraction_result", _spy)

    with caplog.at_level(logging.WARNING, logger="lightrag"):
        nodes, edges = await _rebuild_from_extraction_result(
            _FilePathKV(),
            '{"answer": "no entities found"}',
            chunk_id="chunk-4",
            timestamp=1,
        )

    assert nodes == {}
    assert edges == {}
    assert calls == [], calls
    assert _warning_messages(caplog) == []


@pytest.mark.asyncio
async def test_rebuild_accepts_valid_empty_json_without_delimiter_warning(
    caplog, lightrag_caplog
) -> None:
    """Valid empty JSON extraction output must not fall through to delimiter parsing."""
    with caplog.at_level(logging.WARNING, logger="lightrag"):
        nodes, edges = await _rebuild_from_extraction_result(
            _FilePathKV(),
            '{"entities": [], "relationships": []}',
            chunk_id="chunk-1",
            timestamp=1,
        )
    assert nodes == {}
    assert edges == {}
    assert _DELIMITER_WARNING not in caplog.text


@pytest.mark.asyncio
async def test_rebuild_still_returns_entities_from_nonempty_json() -> None:
    nodes, _edges = await _rebuild_from_extraction_result(
        _FilePathKV(),
        '{"entities":[{"name":"Alice","type":"person",'
        '"description":"an engineer"}],"relationships":[]}',
        chunk_id="chunk-2",
        timestamp=1,
    )
    assert "Alice" in nodes


# --------------------------------------------------------------------------- #
# Multimodal sidecar evidence synthesis during rebuild
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["best_effort", "rollback"])
async def test_multimodal_entity_is_rebuilt_from_its_sidecar_chunk(policy):
    """The deletion rebuild synthesizes core-generated multimodal entity and
    association-edge records from the chunk's sidecar block."""
    content = "Table T1: quarterly revenue"
    text_chunks = MemKV(
        {
            "c1": {
                "content": content,
                "file_path": "report.pdf",
                "sidecar": {"type": "table", "id": "tbl-1"},
                "llm_cache_list": ["default:extract:k1"],
            }
        }
    )
    llm_cache = MemKV(
        {
            "default:extract:k1": {
                "cache_type": "extract",
                "return": "entity<|#|>Revenue<|#|>concept<|#|>Quarterly revenue\n"
                "<|COMPLETE|>",
                "original_prompt": "user prompt\nsystem prompt",
            }
        }
    )
    stale = {"source_id": "c1<SEP>c-deleted", "file_path": "old.pdf"}
    graph = MemGraph(
        nodes={
            "tbl-1": {
                "description": "stale table text",
                "entity_type": "table",
                **stale,
            },
        },
        edges={
            ("tbl-1", "Revenue"): {
                "description": "stale edge text",
                "keywords": "old",
                "weight": 2.0,
                **stale,
            }
        },
    )
    global_config = make_merge_config(
        llm_model_max_async=1,
        max_source_ids_per_entity=10,
        max_source_ids_per_relation=10,
        max_file_paths=10,
    )

    report = await rebuild_knowledge_from_chunks(
        entities_to_rebuild={"tbl-1": ["c1"]},
        relationships_to_rebuild={("tbl-1", "Revenue"): ["c1"]},
        knowledge_graph_inst=graph,
        entities_vdb=MemKV(),
        relationships_vdb=MemKV(),
        text_chunks_storage=text_chunks,
        llm_response_cache=llm_cache,
        global_config=global_config,
        rebuild_policy=policy,
    )

    assert report.degraded_entities == {}
    assert report.degraded_relationships == {}
    assert report.reextracted_chunk_ids == set()
    node = graph.nodes["tbl-1"]
    assert node["description"] == content
    assert (node["source_id"], node["file_path"]) == ("c1", "report.pdf")
    edge = graph.edges[("tbl-1", "Revenue")]
    assert edge["description"].startswith("Revenue is associated with table")
    assert (edge["source_id"], edge["file_path"]) == ("c1", "report.pdf")
