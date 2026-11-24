"""Chunk-tracking presence and upsert-before-delete ordering contract tests (#3609).

Covers the PurgeRecoveryContract chunk-tracking invariants across:
1. Ingestion merges (``_merge_nodes_then_upsert``, ``_merge_edges_then_upsert``,
   and edge-endpoint node updates): present-but-empty rows stay authoritative
   while absent rows fall back to graph ``source_id``.
2. Entity merges (``_merge_entities_impl``): per-input presence resolution,
   curated-empty preservation, unknown-stays-unknown, and upsert-before-delete
   ordering for entity and relation tracking rows.
3. Entity renames (``aedit_entity``): new entity and relation tracking keys are
   upserted before old keys are deleted, including curated-empty rows.
"""

from __future__ import annotations

import pytest

from lightrag import utils_graph
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.operate import _merge_edges_then_upsert, _merge_nodes_then_upsert
from lightrag.utils import make_relation_chunk_key

from ._helpers import MemGraph, MemKV, MemVDB, make_merge_config

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


def _op_index(ops: list[tuple[str, tuple[str, ...]]], wanted_op: str, key: str) -> int:
    return next(
        i for i, (op, keys) in enumerate(ops) if op == wanted_op and key in keys
    )


# --------------------------------------------------------------------------- #
# 1. Ingestion merges: present-but-empty vs absent-row presence model
# --------------------------------------------------------------------------- #


def _node_input(source_id: str) -> dict:
    return {
        "entity_name": "A",
        "entity_type": "person",
        "description": "Entity A",
        "source_id": source_id,
        "file_path": "doc.txt",
        "timestamp": 1,
    }


def _rel_input(source_id: str) -> dict:
    return {
        "weight": 1.0,
        "source_id": source_id,
        "description": "rel",
        "keywords": "k",
        "file_path": "doc.txt",
    }


async def _graph_with_nodes(
    names: tuple[str, ...], source_id: str
) -> MemGraph:
    g = MemGraph()
    for name in names:
        await g.upsert_node(
            name,
            {
                "entity_id": name,
                "entity_type": "person",
                "description": f"Entity {name}",
                "source_id": source_id,
                "file_path": "doc.txt",
            },
        )
    return g


@pytest.mark.parametrize(
    ("initial_rows", "graph_source", "expected_chunks"),
    [
        pytest.param(
            {"A": {"chunk_ids": [], "count": 0}},
            "c-stale",
            ["c-new"],
            id="present_empty_not_reseeded",
        ),
        pytest.param(
            {},
            "c-old",
            ["c-old", "c-new"],
            id="absent_row_seeds_from_graph",
        ),
    ],
)
async def test_node_merge_chunk_tracking_presence(
    initial_rows, graph_source, expected_chunks
):
    """_merge_nodes_then_upsert trusts a present empty row and seeds only when absent."""
    g = await _graph_with_nodes(("A",), graph_source)
    ecs = MemKV(initial_rows)

    await _merge_nodes_then_upsert(
        "A",
        [_node_input("c-new")],
        g,
        None,
        make_merge_config(),
        entity_chunks_storage=ecs,
    )

    assert ecs.data["A"]["chunk_ids"] == expected_chunks


@pytest.mark.parametrize(
    ("initial_rows", "edge_source", "expected_chunks"),
    [
        pytest.param(
            {make_relation_chunk_key("A", "B"): {"chunk_ids": [], "count": 0}},
            "c-stale",
            ["c-new"],
            id="present_empty_not_reseeded",
        ),
        pytest.param(
            {},
            "c-old",
            ["c-old", "c-new"],
            id="absent_row_seeds_from_graph",
        ),
    ],
)
async def test_edge_merge_chunk_tracking_presence(
    initial_rows, edge_source, expected_chunks
):
    """_merge_edges_then_upsert trusts a present empty relation row and seeds only when absent."""
    g = await _graph_with_nodes(("A", "B"), "c-node")
    await g.upsert_edge("A", "B", _rel_input(edge_source))
    rcs = MemKV(initial_rows)

    await _merge_edges_then_upsert(
        "A",
        "B",
        [_rel_input("c-new")],
        g,
        MemVDB(),
        MemVDB(),
        make_merge_config(),
        relation_chunks_storage=rcs,
    )

    assert rcs.data[make_relation_chunk_key("A", "B")]["chunk_ids"] == expected_chunks


@pytest.mark.parametrize(
    ("initial_rows", "node_source", "expected_chunks"),
    [
        pytest.param(
            {
                "A": {"chunk_ids": [], "count": 0},
                "B": {"chunk_ids": [], "count": 0},
            },
            "c-stale",
            ["c-new"],
            id="present_empty_not_reseeded",
        ),
        pytest.param(
            {},
            "c-node",
            ["c-node", "c-new"],
            id="absent_row_seeds_from_graph",
        ),
    ],
)
async def test_edge_endpoint_update_chunk_tracking_presence(
    initial_rows, node_source, expected_chunks
):
    """The endpoint-node update branch inside _merge_edges_then_upsert obeys row presence."""
    g = await _graph_with_nodes(("A", "B"), node_source)
    ecs = MemKV(initial_rows)

    await _merge_edges_then_upsert(
        "A",
        "B",
        [_rel_input("c-new")],
        g,
        MemVDB(),
        MemVDB(),
        make_merge_config(),
        entity_chunks_storage=ecs,
    )

    assert ecs.data["A"]["chunk_ids"] == expected_chunks
    assert ecs.data["B"]["chunk_ids"] == expected_chunks


# --------------------------------------------------------------------------- #
# 2. Entity merge chunk-tracking presence & upsert-before-delete ordering
# --------------------------------------------------------------------------- #


def _make_merge_graph() -> MemGraph:
    return MemGraph(
        nodes={
            name: {
                "entity_id": name,
                "description": f"Entity {name}",
                "entity_type": "ORG",
                "source_id": f"chunk-{name.lower()}-node",
                "file_path": f"doc{name}.txt",
            }
            for name in ("A", "B", "C")
        },
        edges={
            ("A", "C"): {
                "description": "A relates to C",
                "keywords": "k1",
                "source_id": GRAPH_FIELD_SEP.join(["chunk-a1"]),
                "weight": 1.0,
                "file_path": "docA.txt",
            },
            ("B", "C"): {
                "description": "B relates to C",
                "keywords": "k2",
                "source_id": GRAPH_FIELD_SEP.join(["chunk-b1"]),
                "weight": 1.0,
                "file_path": "docB.txt",
            },
        },
    )


async def _run_merge(
    ecs: MemKV,
    *,
    source_entities: list[str] | None = None,
    target_entity: str = "AB",
    rcs: MemKV | None = None,
) -> None:
    await utils_graph._merge_entities_impl(
        chunk_entity_relation_graph=_make_merge_graph(),
        entities_vdb=MemVDB(),
        relationships_vdb=MemVDB(),
        source_entities=source_entities if source_entities is not None else ["A", "B"],
        target_entity=target_entity,
        entity_chunks_storage=ecs,
        relation_chunks_storage=rcs,
    )


async def test_merge_preserves_curated_empty_entity_rows():
    """All inputs curated-empty: target keeps a present empty row and sources are removed."""
    ecs = MemKV(
        {
            "A": {"chunk_ids": [], "count": 0},
            "B": {"chunk_ids": [], "count": 0},
        }
    )

    await _run_merge(ecs)

    assert ecs.data["AB"] == {"chunk_ids": [], "count": 0}
    assert "A" not in ecs.data
    assert "B" not in ecs.data


async def test_merge_leaves_target_row_absent_when_no_source_has_one():
    """Unknown stays unknown: with no input row present, target row is not fabricated."""
    ecs = MemKV()

    await _run_merge(ecs)

    assert "AB" not in ecs.data


async def test_merge_mixed_empty_row_and_absent_row_falls_back_per_entity():
    """A's curated-empty row applies only to A; B falls back to its own graph source_id."""
    ecs = MemKV({"A": {"chunk_ids": [], "count": 0}})

    await _run_merge(ecs)

    assert ecs.data["AB"] == {"chunk_ids": ["chunk-b-node"], "count": 1}


async def test_merge_mixed_valid_row_and_malformed_row_falls_back_per_entity():
    """A valid row and a malformed row resolve independently per input entity."""
    ecs = MemKV(
        {
            "A": {"chunk_ids": ["c-a"], "count": 1},
            "B": {"chunk_ids": None},
        }
    )

    await _run_merge(ecs)

    assert ecs.data["AB"] == {"chunk_ids": ["c-a", "chunk-b-node"], "count": 2}


async def test_merge_into_existing_target_with_absent_row_uses_its_pre_merge_source_id():
    """Existing target with absent row falls back to its pre-merge graph source_id."""
    ecs = MemKV({"A": {"chunk_ids": ["c-a"], "count": 1}})

    await _run_merge(ecs, source_entities=["A"], target_entity="B")

    assert ecs.data["B"] == {"chunk_ids": ["c-a", "chunk-b-node"], "count": 2}
    assert "A" not in ecs.data


async def test_merge_upserts_target_row_before_deleting_source_rows():
    """Crash-ordering pin: target entity row is upserted before source rows are deleted."""
    ecs = MemKV(
        {
            "A": {"chunk_ids": ["c-a"], "count": 1},
            "B": {"chunk_ids": ["c-b"], "count": 1},
        }
    )

    await _run_merge(ecs)

    assert ecs.data["AB"]["chunk_ids"] == ["c-a", "c-b"]
    assert _op_index(ecs.ops, "upsert", "AB") < _op_index(ecs.ops, "delete", "A")


async def test_merge_upserts_relation_rows_before_deleting_only_stale_keys():
    """Migrated relation union is upserted before stale old keys are deleted,
    and keys that are both old and new are never deleted."""
    old_key = make_relation_chunk_key("A", "C")
    shared_key = make_relation_chunk_key("B", "C")
    ecs = MemKV({"A": {"chunk_ids": ["c-a"], "count": 1}})
    rcs = MemKV(
        {
            old_key: {"chunk_ids": ["c-ac"], "count": 1},
            shared_key: {"chunk_ids": ["c-bc"], "count": 1},
        }
    )

    await _run_merge(ecs, source_entities=["A"], target_entity="B", rcs=rcs)

    assert rcs.data[shared_key]["chunk_ids"] == ["c-ac", "c-bc"]
    assert old_key not in rcs.data
    assert _op_index(rcs.ops, "upsert", shared_key) < _op_index(
        rcs.ops, "delete", old_key
    )
    assert all(shared_key not in keys for op, keys in rcs.ops if op == "delete")


# --------------------------------------------------------------------------- #
# 3. Entity rename tracking migration ordering (upsert new key before delete)
# --------------------------------------------------------------------------- #


class _NoopLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


async def _rename_a_to_b(
    entity_chunks: MemKV, relation_chunks: MemKV, monkeypatch
) -> None:
    monkeypatch.setattr(
        utils_graph,
        "get_storage_keyed_lock",
        lambda *args, **kwargs: _NoopLock(),
    )

    graph = MemGraph(
        nodes={
            "A": {
                "entity_id": "A",
                "description": "source",
                "entity_type": "ORG",
                "source_id": "chunk-a",
                "file_path": "docA.txt",
            },
            "C": {
                "entity_id": "C",
                "description": "neighbour",
                "entity_type": "ORG",
                "source_id": "chunk-c",
                "file_path": "docC.txt",
            },
        },
        edges={
            ("A", "C"): {
                "description": "A relates to C",
                "keywords": "k",
                "source_id": "chunk-rel",
                "weight": 1.0,
                "file_path": "docA.txt",
            }
        },
    )

    result = await utils_graph.aedit_entity(
        graph,
        MemVDB({"workspace": ""}),
        MemVDB({"workspace": ""}),
        "A",
        {"entity_name": "B"},
        entity_chunks_storage=entity_chunks,
        relation_chunks_storage=relation_chunks,
    )
    assert result["operation_summary"]["operation_status"] == "success"


async def test_rename_upserts_new_entity_key_before_deleting_old(monkeypatch):
    entity_chunks = MemKV({"A": {"chunk_ids": ["chunk-a"], "count": 1}})
    relation_chunks = MemKV(
        {make_relation_chunk_key("A", "C"): {"chunk_ids": ["chunk-rel"], "count": 1}}
    )

    await _rename_a_to_b(entity_chunks, relation_chunks, monkeypatch)

    assert entity_chunks.data["B"]["chunk_ids"] == ["chunk-a"]
    assert "A" not in entity_chunks.data
    assert _op_index(entity_chunks.ops, "upsert", "B") < _op_index(
        entity_chunks.ops, "delete", "A"
    )


@pytest.mark.parametrize(
    ("entity_row", "relation_row"),
    [
        pytest.param(
            {"chunk_ids": ["chunk-a"], "count": 1},
            {"chunk_ids": ["chunk-rel"], "count": 1},
            id="populated_row",
        ),
        pytest.param(
            {"chunk_ids": [], "count": 0},
            {"chunk_ids": [], "count": 0},
            id="curated_empty_row",
        ),
    ],
)
async def test_rename_upserts_new_relation_key_before_deleting_old(
    monkeypatch, entity_row, relation_row
):
    """Rename writes the migrated relation tracking row (populated or curated-empty)
    under the new key before deleting the old key."""
    old_key = make_relation_chunk_key("A", "C")
    new_key = make_relation_chunk_key("B", "C")
    entity_chunks = MemKV({"A": entity_row})
    relation_chunks = MemKV({old_key: relation_row})

    await _rename_a_to_b(entity_chunks, relation_chunks, monkeypatch)

    assert relation_chunks.data[new_key] == relation_row
    assert old_key not in relation_chunks.data
    assert _op_index(relation_chunks.ops, "upsert", new_key) < _op_index(
        relation_chunks.ops, "delete", old_key
    )
