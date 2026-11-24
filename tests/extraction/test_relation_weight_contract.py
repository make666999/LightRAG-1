"""Relation weight contract tests across merge, rebuild, entity merge, and null-VDB paths.

Covers the unified Relation Weight Contract invariants:
1. ``_merge_edges_then_upsert`` counts each distinct real source once across
   reprocess/resume, within a single batch, when ``relation_chunks`` is ahead,
   under KEEP caps, and ignores falsy source IDs (#3367).
2. ``_rebuild_single_relationship`` enforces ``weight >= distinct real sources``,
   preserves explicit boosts on degraded rebuilds, re-derives weight from cached
   fragments, floors on the post-limit list, and excludes no-evidence
   placeholders (``manual_creation`` / ``UNKNOWN``) from tracking and ``source_id``
   (#3676 follow-up).
3. ``_merge_entities_impl`` preserves ``max(input weights, distinct merged real
   source IDs)`` in both graph and relationship VDB payloads.
4. ``_merge_edges_then_upsert`` tolerates ``entity_vdb=None`` without crashing.
"""

from __future__ import annotations

import asyncio

import pytest

from lightrag import utils_graph
from lightrag.constants import GRAPH_FIELD_SEP, RELATION_NO_EVIDENCE_SOURCE_IDS
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.operate import (
    _merge_edges_then_upsert,
    _rebuild_single_relationship,
    merge_nodes_and_edges,
)
from lightrag.utils import make_relation_chunk_key

from ._helpers import MemGraph, MemKV, MemVDB, make_merge_config

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


async def _edge_graph_with_nodes(
    *, edge: dict | None = None, node_source_id: str = "c1"
) -> MemGraph:
    graph = MemGraph()
    for name in ("A", "B"):
        await graph.upsert_node(
            name,
            {
                "entity_id": name,
                "entity_type": "ENT",
                "description": f"Node {name}",
                "source_id": node_source_id,
                "file_path": "f1.txt",
            },
        )
    if edge is not None:
        await graph.upsert_edge("A", "B", edge)
    return graph


def _rel(src, weight: float = 1.0) -> dict:
    return {
        "weight": weight,
        "source_id": src,
        "description": "rel",
        "keywords": "k",
        "file_path": "f",
    }


def _fragment(weight: float = 1.0) -> dict:
    return {
        "weight": weight,
        "description": "rel desc",
        "keywords": "kw",
        "file_path": "f1.txt",
    }


# --------------------------------------------------------------------------- #
# 1. Edge merge weight idempotence & evidence floor (#3367)
# --------------------------------------------------------------------------- #


async def test_edge_weight_counts_each_source_once_across_reprocess():
    """Re-feeding an already-stored source adds nothing; each new source adds its weight."""
    g = await _edge_graph_with_nodes()
    cfg = make_merge_config()
    weights: list[float] = []
    for src in ["c1", "c1", "c2", "c2", "c3"]:
        await _merge_edges_then_upsert(
            "A", "B", [_rel(src)], g, MemVDB(), MemVDB(), cfg
        )
        weights.append((await g.get_edge("A", "B"))["weight"])
    assert weights == [1.0, 1.0, 2.0, 2.0, 3.0]


async def test_edge_weight_not_double_counted_within_one_batch():
    """Two records for the same source in one batch contribute weight once."""
    g = await _edge_graph_with_nodes()
    await _merge_edges_then_upsert(
        "A",
        "B",
        [_rel("c1"), _rel("c1"), _rel("c2")],
        g,
        MemVDB(),
        MemVDB(),
        make_merge_config(),
    )
    edge = await g.get_edge("A", "B")
    assert edge["source_id"].split(GRAPH_FIELD_SEP) == ["c1", "c2"]
    assert edge["weight"] == 2.0


async def test_edge_weight_adds_new_source_when_chunks_store_is_ahead():
    """When relation_chunks is ahead of the graph edge, the filter uses the
    edge's own source_ids so a genuinely new source still adds its weight."""
    g = await _edge_graph_with_nodes(edge=_rel("c1", weight=5.0))
    rcs = MemKV({make_relation_chunk_key("A", "B"): {"chunk_ids": ["c1", "c2"]}})
    await _merge_edges_then_upsert(
        "A",
        "B",
        [_rel("c2")],
        g,
        MemVDB(),
        MemVDB(),
        make_merge_config(),
        relation_chunks_storage=rcs,
    )
    assert (await g.get_edge("A", "B"))["weight"] == 6.0


async def test_edge_merge_repairs_legacy_weight_below_evidence_floor():
    g = await _edge_graph_with_nodes(edge=_rel("c1", weight=0.25))

    await _merge_edges_then_upsert(
        "A", "B", [_rel("c2")], g, MemVDB(), MemVDB(), make_merge_config()
    )

    edge = await g.get_edge("A", "B")
    assert set(edge["source_id"].split(GRAPH_FIELD_SEP)) == {"c1", "c2"}
    assert edge["weight"] == 2.0


async def test_edge_weight_stays_consistent_when_new_source_hits_id_limit():
    """A new source dropped by KEEP cap must not inflate the stored weight."""
    g = await _edge_graph_with_nodes(edge=_rel("c1"))
    cfg = make_merge_config(max_source_ids_per_relation=1)
    await _merge_edges_then_upsert("A", "B", [_rel("c2")], g, MemVDB(), MemVDB(), cfg)
    edge = await g.get_edge("A", "B")
    persisted = len([s for s in edge["source_id"].split(GRAPH_FIELD_SEP) if s])
    assert persisted == 1
    assert edge["weight"] == 1.0


@pytest.mark.parametrize("falsy_source", [None, ""])
async def test_edge_weight_ignores_falsy_source(falsy_source):
    """A falsy source_id is excluded from source_id and must not add weight."""
    g = await _edge_graph_with_nodes(edge=_rel("c1"))
    await _merge_edges_then_upsert(
        "A", "B", [_rel(falsy_source)], g, MemVDB(), MemVDB(), make_merge_config()
    )
    edge = await g.get_edge("A", "B")
    persisted = len([s for s in edge["source_id"].split(GRAPH_FIELD_SEP) if s])
    assert persisted == 1
    assert edge["weight"] == 1.0


async def test_edge_merge_with_entity_vdb_none_does_not_crash():
    """entity_vdb=None must not raise UnboundLocalError when an edge merge
    adds or updates endpoint entities."""
    g = await _edge_graph_with_nodes()
    cfg = make_merge_config()
    for src in ("c1", "c2", "c3"):
        await _merge_edges_then_upsert("A", "B", [_rel(src)], g, None, None, cfg)
    assert await g.get_edge("A", "B") is not None


async def test_orchestrator_weight_and_description_track_distinct_sources():
    """Through the real merge_nodes_and_edges orchestrator, reprocessing a doc
    grows neither description nor weight, while a distinct doc grows both; a
    chunk stating the relation in both directions counts once."""
    initialize_share_data()
    g = MemGraph()
    stores = {
        "entity_vdb": MemVDB(),
        "relationships_vdb": MemVDB(),
        "full_entities": MemKV(),
        "full_relations": MemKV(),
        "entity_chunks": MemKV(),
        "relation_chunks": MemKV(),
    }
    cfg = make_merge_config()

    def _one_chunk(src: str, alice_desc: str):
        maybe_nodes = {
            "ALICE": [
                {
                    "entity_name": "ALICE",
                    "entity_type": "person",
                    "description": alice_desc,
                    "source_id": src,
                    "file_path": "d.txt",
                    "timestamp": 1,
                }
            ],
            "ACME": [
                {
                    "entity_name": "ACME",
                    "entity_type": "person",
                    "description": "Acme is a company.",
                    "source_id": src,
                    "file_path": "d.txt",
                    "timestamp": 1,
                }
            ],
        }
        maybe_edges = {
            ("ALICE", "ACME"): [
                {
                    "src_id": "ALICE",
                    "tgt_id": "ACME",
                    "weight": 1.0,
                    "description": "rel",
                    "keywords": "k",
                    "source_id": src,
                    "file_path": "d.txt",
                    "timestamp": 1,
                }
            ],
            ("ACME", "ALICE"): [
                {
                    "src_id": "ACME",
                    "tgt_id": "ALICE",
                    "weight": 1.0,
                    "description": "rel",
                    "keywords": "k",
                    "source_id": src,
                    "file_path": "d.txt",
                    "timestamp": 1,
                }
            ],
        }
        return [(maybe_nodes, maybe_edges)]

    observed = []
    for src, desc, doc_id in [
        ("c1", "Alice is an engineer.", "d1"),
        ("c1", "Alice is an engineer.", "d1"),
        ("c2", "Alice leads the team.", "d2"),
        ("c1", "Alice is an engineer.", "d1"),
    ]:
        await merge_nodes_and_edges(
            _one_chunk(src, desc),
            g,
            stores["entity_vdb"],
            stores["relationships_vdb"],
            cfg,
            full_entities_storage=stores["full_entities"],
            full_relations_storage=stores["full_relations"],
            doc_id=doc_id,
            pipeline_status={"history_messages": []},
            pipeline_status_lock=asyncio.Lock(),
            entity_chunks_storage=stores["entity_chunks"],
            relation_chunks_storage=stores["relation_chunks"],
        )
        frag_count = len(
            [d for d in g.nodes["ALICE"]["description"].split(GRAPH_FIELD_SEP) if d]
        )
        observed.append((frag_count, (await g.get_edge("ALICE", "ACME"))["weight"]))
    assert observed == [(1, 1.0), (1, 1.0), (2, 2.0), (2, 2.0)]


# --------------------------------------------------------------------------- #
# 2. Rebuild evidence floor & placeholder exclusion (#3676 follow-up)
# --------------------------------------------------------------------------- #


async def _graph_with_edge(weight: float, source_id: str) -> MemGraph:
    return await _edge_graph_with_nodes(
        node_source_id=source_id,
        edge={
            "description": "Relation AB",
            "keywords": "legacy_kw",
            "weight": weight,
            "source_id": source_id,
            "file_path": "f1.txt",
        },
    )


async def _rebuild(
    graph: MemGraph,
    chunk_ids: list[str],
    chunk_relationships: dict,
    *,
    global_config: dict | None = None,
    relation_chunks_storage=None,
) -> dict:
    await _rebuild_single_relationship(
        knowledge_graph_inst=graph,
        relationships_vdb=MemVDB(),
        entities_vdb=MemVDB(),
        src="A",
        tgt="B",
        chunk_ids=chunk_ids,
        chunk_relationships=chunk_relationships,
        llm_response_cache=MemKV(),
        global_config=global_config
        or make_merge_config(
            max_source_ids_per_entity=10,
            max_source_ids_per_relation=10,
            max_file_paths=10,
        ),
        relation_chunks_storage=relation_chunks_storage,
        structural_fallback=True,
    )
    return await graph.get_edge("A", "B")


async def test_rebuild_lifts_legacy_weight_when_no_fragment_cache_survives():
    graph = await _graph_with_edge(weight=1.0, source_id="c1")
    edge = await _rebuild(graph, ["c1", "c2", "c3"], {})
    assert edge["source_id"] == GRAPH_FIELD_SEP.join(["c1", "c2", "c3"])
    assert edge["weight"] == 3.0


async def test_rebuild_lifts_weight_when_fragment_cache_is_partial():
    graph = await _graph_with_edge(weight=1.0, source_id="c1")
    edge = await _rebuild(
        graph,
        ["c1", "c2", "c3"],
        {"c1": {("A", "B"): [_fragment(1.0)]}},
    )
    assert edge["source_id"] == GRAPH_FIELD_SEP.join(["c1", "c2", "c3"])
    assert edge["weight"] == 3.0


async def test_degraded_rebuild_preserves_a_larger_explicit_boost():
    graph = await _graph_with_edge(weight=10.0, source_id="c1")
    edge = await _rebuild(graph, ["c1", "c2"], {})
    assert edge["weight"] == 10.0


async def test_cached_rebuild_rederives_weight_instead_of_keeping_the_boost():
    graph = await _graph_with_edge(weight=10.0, source_id="c1")
    edge = await _rebuild(
        graph,
        ["c1", "c2"],
        {
            "c1": {("A", "B"): [_fragment(1.0)]},
            "c2": {("A", "B"): [_fragment(1.0)]},
        },
    )
    assert edge["weight"] == 2.0


async def test_rebuild_floors_on_the_post_limit_chunk_list():
    graph = await _graph_with_edge(weight=1.0, source_id="c1")
    edge = await _rebuild(
        graph,
        ["c1", "c2", "c3"],
        {},
        global_config=make_merge_config(
            max_source_ids_per_entity=10,
            max_source_ids_per_relation=2,
            max_file_paths=10,
        ),
    )
    assert edge["source_id"] == GRAPH_FIELD_SEP.join(["c1", "c2"])
    assert edge["weight"] == 2.0


async def test_rebuild_counts_each_chunk_once():
    """Multiple fragments for the same chunk (in either direction) count once."""
    graph = await _graph_with_edge(weight=2.0, source_id="c1")
    edge = await _rebuild(
        graph,
        ["c1", "c2"],
        {
            "c1": {
                ("A", "B"): [_fragment(1.0), _fragment(1.0)],
                ("B", "A"): [_fragment(1.0)],
            },
            "c2": {("A", "B"): [_fragment(1.0)]},
        },
    )
    assert edge["weight"] == 2.0


@pytest.mark.parametrize("placeholder", sorted(RELATION_NO_EVIDENCE_SOURCE_IDS))
async def test_merge_keeps_legacy_placeholder_out_of_chunk_tracking(placeholder):
    graph = await _graph_with_edge(weight=0.25, source_id=placeholder)
    storage_key = make_relation_chunk_key("A", "B")
    tracking = MemKV()

    await _merge_edges_then_upsert(
        "A",
        "B",
        [{**_fragment(1.0), "source_id": "c1"}],
        graph,
        MemVDB(),
        MemVDB(),
        make_merge_config(
            max_source_ids_per_entity=10,
            max_source_ids_per_relation=10,
            max_file_paths=10,
        ),
        relation_chunks_storage=tracking,
    )

    assert tracking.data[storage_key]["chunk_ids"] == ["c1"]
    assert tracking.data[storage_key]["count"] == 1
    edge = await graph.get_edge("A", "B")
    assert edge["source_id"] == "c1"
    assert edge["weight"] == 1.25


async def test_merge_repairs_a_tracking_row_that_already_holds_a_placeholder():
    graph = await _graph_with_edge(
        weight=2.0, source_id=GRAPH_FIELD_SEP.join(["manual_creation", "c1"])
    )
    storage_key = make_relation_chunk_key("A", "B")
    tracking = MemKV({storage_key: {"chunk_ids": ["manual_creation", "c1"]}})

    await _merge_edges_then_upsert(
        "A",
        "B",
        [{**_fragment(1.0), "source_id": "c2"}],
        graph,
        MemVDB(),
        MemVDB(),
        make_merge_config(
            max_source_ids_per_entity=10,
            max_source_ids_per_relation=10,
            max_file_paths=10,
        ),
        relation_chunks_storage=tracking,
    )

    assert tracking.data[storage_key]["chunk_ids"] == ["c1", "c2"]
    edge = await graph.get_edge("A", "B")
    assert edge["source_id"] == GRAPH_FIELD_SEP.join(["c1", "c2"])
    assert edge["weight"] >= 2.0


@pytest.mark.parametrize("placeholder", sorted(RELATION_NO_EVIDENCE_SOURCE_IDS))
async def test_rebuild_keeps_placeholders_out_of_tracking_and_source_id(placeholder):
    graph = await _graph_with_edge(weight=1.0, source_id=placeholder)
    storage_key = make_relation_chunk_key("A", "B")
    tracking = MemKV({storage_key: {"chunk_ids": [placeholder, "c1"]}})

    edge = await _rebuild(
        graph,
        [placeholder, "c1", "c2"],
        {"c1": {("A", "B"): [_fragment(1.0)]}},
        relation_chunks_storage=tracking,
    )

    assert tracking.data[storage_key]["chunk_ids"] == ["c1", "c2"]
    assert tracking.data[storage_key]["count"] == 2
    assert edge["source_id"] == GRAPH_FIELD_SEP.join(["c1", "c2"])
    assert edge["weight"] == 2.0


async def test_rebuild_rewrites_an_all_placeholder_row_empty():
    """An all-placeholder row is rewritten empty (authoritative empty tracking),
    leaving source_id empty and preserving the source-less fractional weight."""
    graph = await _graph_with_edge(weight=0.5, source_id="manual_creation")
    storage_key = make_relation_chunk_key("A", "B")
    tracking = MemKV({storage_key: {"chunk_ids": ["manual_creation"]}})

    edge = await _rebuild(
        graph,
        ["manual_creation"],
        {},
        relation_chunks_storage=tracking,
    )

    assert tracking.data[storage_key]["chunk_ids"] == []
    assert tracking.data[storage_key]["count"] == 0
    assert edge["source_id"] == ""
    assert edge["weight"] == 0.5


# --------------------------------------------------------------------------- #
# 3. Entity merge relation weight contract (_merge_entities_impl)
# --------------------------------------------------------------------------- #


def _make_merge_entities_graph(edges: dict[tuple[str, str], dict]) -> MemGraph:
    nodes = {
        name: {
            "entity_id": name,
            "description": f"Entity {name}",
            "entity_type": "ORG",
            "source_id": f"chunk-{name.lower()}-node",
            "file_path": f"doc{name}.txt",
        }
        for name in ("A", "B", "C")
    }
    return MemGraph(nodes=nodes, edges=edges)


@pytest.mark.parametrize(
    (
        "edges",
        "source_entities",
        "expected_sources",
        "expected_weight",
        "exact_source_str",
    ),
    [
        pytest.param(
            {
                ("A", "C"): {
                    "description": "A relates to C",
                    "keywords": "k1",
                    "source_id": GRAPH_FIELD_SEP.join(
                        ["chunk-a1", "chunk-a2", "chunk-a3"]
                    ),
                    "weight": 3.0,
                    "file_path": "docA.txt",
                },
                ("B", "C"): {
                    "description": "B relates to C",
                    "keywords": "k2",
                    "source_id": GRAPH_FIELD_SEP.join(["chunk-b1", "chunk-b2"]),
                    "weight": 2.0,
                    "file_path": "docB.txt",
                },
            },
            ["A", "B"],
            {"chunk-a1", "chunk-a2", "chunk-a3", "chunk-b1", "chunk-b2"},
            5.0,
            None,
            id="distinct_source_count_sum",
        ),
        pytest.param(
            {
                ("A", "C"): {
                    "description": "A relates to C",
                    "keywords": "k1",
                    "source_id": "manual_creation",
                    "weight": 10.0,
                    "file_path": "docA.txt",
                },
                ("B", "C"): {
                    "description": "B relates to C",
                    "keywords": "k2",
                    "source_id": "manual_creation",
                    "weight": 8.0,
                    "file_path": "docB.txt",
                },
            },
            ["A", "B"],
            None,
            10.0,
            "manual_creation",
            id="explicit_boost_never_regresses_below_max",
        ),
        pytest.param(
            {
                ("A", "C"): {
                    "description": "A relates to C",
                    "keywords": "k1",
                    "source_id": "manual_creation",
                    "weight": 0.25,
                    "file_path": "docA.txt",
                },
                ("B", "C"): {
                    "description": "B relates to C",
                    "keywords": "k2",
                    "source_id": "manual_creation",
                    "weight": 0.5,
                    "file_path": "docB.txt",
                },
            },
            ["A", "B"],
            None,
            0.5,
            "manual_creation",
            id="legacy_sourceless_fractional_weight_preserved",
        ),
        pytest.param(
            {
                ("A", "C"): {
                    "description": "A relates to C",
                    "keywords": "k1",
                    "source_id": "manual-a",
                    "weight": 0.25,
                    "file_path": "docA.txt",
                },
                ("B", "C"): {
                    "description": "B relates to C",
                    "keywords": "k2",
                    "source_id": "manual-b",
                    "weight": 0.5,
                    "file_path": "docB.txt",
                },
            },
            ["A", "B"],
            {"manual-a", "manual-b"},
            2.0,
            None,
            id="sourced_low_weight_lifted_to_floor",
        ),
        pytest.param(
            {
                ("A", "C"): {
                    "description": "A relates to C",
                    "keywords": "k1",
                    "source_id": GRAPH_FIELD_SEP.join(["chunk-1", "chunk-2"]),
                    "weight": 0.25,
                    "file_path": "docA.txt",
                }
            },
            ["A"],
            {"chunk-1", "chunk-2"},
            2.0,
            None,
            id="single_redirected_low_weight_lifted_to_floor",
        ),
    ],
)
async def test_merge_entities_relation_weight_contract(
    edges, source_entities, expected_sources, expected_weight, exact_source_str
):
    """Entity merge keeps ``max(all input weights, distinct merged real sources)``
    in both graph and relationships VDB."""
    graph = _make_merge_entities_graph(edges)
    entities_vdb = MemVDB()
    relationships_vdb = MemVDB()

    await utils_graph._merge_entities_impl(
        chunk_entity_relation_graph=graph,
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        source_entities=source_entities,
        target_entity="AB",
    )

    merged_edge = graph.edges[("AB", "C")]
    if exact_source_str is not None:
        assert merged_edge["source_id"] == exact_source_str
    if expected_sources is not None:
        source_chunks = [
            s for s in merged_edge["source_id"].split(GRAPH_FIELD_SEP) if s
        ]
        assert set(source_chunks) == expected_sources
    assert merged_edge["weight"] == expected_weight

    vdb_payload = relationships_vdb.upserts[-1]
    persisted = next(iter(vdb_payload.values()))
    assert persisted["weight"] == expected_weight
