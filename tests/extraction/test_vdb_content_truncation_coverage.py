"""VDB content-truncation coverage across every entity/relation write path.

Covers:
1. ``operate._truncate_vdb_content`` safe-truncate contract (no-ops, verified
   prefix truncation, impossible-budget ``TokenBudgetError``).
2. Front-loaded per-object truncation across ``_rebuild_single_entity``,
   ``_rebuild_single_relationship``, ``_merge_nodes_then_upsert``,
   ``_merge_edges_then_upsert``, ``acreate_entity``, ``acreate_relation``,
   ``aedit_relation``, and ``ainsert_custom_kg`` — ensuring oversized payloads
   are truncated while ``TokenBudgetError`` prevents that object's own graph
   write without rolling back already-written earlier objects in multi-object
   operations.
3. Post-graph-write VDB failures on ``aedit_relation`` (transient VDB delete),
   ``_edit_entity_impl`` (rename), and ``_merge_entities_impl`` wrapping as
   ``VectorStorageConsistencyError`` while preserving the durable graph write.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

import lightrag.operate as operate
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.lightrag import LightRAG
from lightrag.operate import (
    _merge_edges_then_upsert,
    _merge_nodes_then_upsert,
    _rebuild_single_entity,
    _rebuild_single_relationship,
    _truncate_vdb_content,
)
from lightrag.utils import (
    TokenBudgetError,
    Tokenizer,
    TokenizerInterface,
    VectorStorageConsistencyError,
)
from lightrag.utils_graph import (
    _edit_entity_impl,
    _merge_entities_impl,
    acreate_entity,
    acreate_relation,
    aedit_relation,
)

from ._helpers import (
    CharTokenizer,
    MemGraph,
    MemKV,
    MemVDB,
    make_dummy_tokenizer,
    make_merge_config,
)

pytestmark = pytest.mark.offline

LONG_DESCRIPTION = "x" * 500


@pytest.fixture(autouse=True)
def _shared_data():
    """Ensure the single-process shared-data registry exists for keyed locks."""
    initialize_share_data()


class _AlwaysFailsTruncateTokenizer(Tokenizer):
    """Tokenizer whose ``truncate_by_token_limit`` always raises ``TokenBudgetError``."""

    def __init__(self):
        super().__init__("hostile", CharTokenizer())

    def truncate_by_token_limit(self, content, max_tokens):
        raise TokenBudgetError(max_tokens, 999, content[:20])


class _FailsOnlyForRelationContentTokenizer(Tokenizer):
    """Truncates entity content fine, but raises ``TokenBudgetError`` for relation
    content (discriminated by the literal ``\\t`` in relation VDB payloads)."""

    def __init__(self):
        super().__init__("relation-hostile", CharTokenizer())

    def truncate_by_token_limit(self, content, max_tokens):
        if "\t" in content:
            raise TokenBudgetError(max_tokens, 999, content[:20])
        return super().truncate_by_token_limit(content, max_tokens)


def _cfg(fail: bool = False, *, tokenizer: Tokenizer | None = None, limit: int = 20):
    tok = (
        tokenizer
        if tokenizer is not None
        else (_AlwaysFailsTruncateTokenizer() if fail else make_dummy_tokenizer("char"))
    )
    return make_merge_config(tokenizer=tok, embedding_token_limit=limit)


def _make_custom_kg_rag(global_config: dict):
    """Bare LightRAG instance wired for ``ainsert_custom_kg`` graph/VDB writes."""
    rag = LightRAG.__new__(LightRAG)
    rag.workspace = ""
    rag.tokenizer = global_config["tokenizer"]
    rag._build_global_config = lambda: global_config
    rag.chunk_entity_relation_graph = MemGraph()
    rag.entities_vdb = MemVDB(global_config)
    rag.relationships_vdb = MemVDB(global_config)
    rag.entity_chunks = MemKV()
    rag.relation_chunks = MemKV()
    rag._insert_done = AsyncMock(return_value=None)
    return rag


# --------------------------------------------------------------------------- #
# 1. Direct _truncate_vdb_content unit contract
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("content", "global_config"),
    [
        pytest.param("hello", {}, id="no_limit"),
        pytest.param("hello", {"embedding_token_limit": 5}, id="no_tokenizer"),
        pytest.param(
            "",
            {"embedding_token_limit": 5, "tokenizer": make_dummy_tokenizer("char")},
            id="empty_content",
        ),
        pytest.param(
            "entity_name\nshort description",
            {"embedding_token_limit": 100, "tokenizer": make_dummy_tokenizer("char")},
            id="within_limit",
        ),
        pytest.param(
            "x" * 50,
            {"embedding_token_limit": 0, "tokenizer": make_dummy_tokenizer("char")},
            id="non_positive_limit",
        ),
    ],
)
def test_truncate_vdb_content_noop_cases(content: str, global_config: dict):
    assert _truncate_vdb_content(content, global_config, "entity:x") == content


def test_truncate_vdb_content_truncates_oversized_to_verified_prefix():
    tok = make_dummy_tokenizer("char")
    gc = {"embedding_token_limit": 10, "tokenizer": tok}
    content = "x" * 500
    out = _truncate_vdb_content(content, gc, "entity:x")
    assert out == content[:10]
    assert len(tok.encode(out)) <= 10


def test_truncate_vdb_content_impossible_budget_raises_token_budget_error():
    class _HeavyTokenizer(TokenizerInterface):
        def encode(self, content: str) -> list[int]:
            return [0] * (5 * len(content))

        def decode(self, tokens: list[int]) -> str:
            return ""

    tok = Tokenizer("heavy", _HeavyTokenizer())
    gc = {"embedding_token_limit": 1, "tokenizer": tok}
    with pytest.raises(TokenBudgetError):
        _truncate_vdb_content("hello", gc, "entity:x")


# --------------------------------------------------------------------------- #
# 2. operate.py: rebuild paths
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_truncation", [False, True], ids=["truncates", "fails"])
async def test_rebuild_single_entity_truncation(fail_truncation: bool):
    graph = MemGraph(
        nodes={
            "ALICE": {
                "entity_id": "ALICE",
                "description": "short",
                "entity_type": "PERSON",
                "source_id": "c1",
                "file_path": "f",
            }
        }
    )
    cfg = _cfg(fail=fail_truncation)
    vdb = MemVDB(cfg)
    node_before = dict(graph.nodes["ALICE"])

    coro = _rebuild_single_entity(
        graph,
        vdb,
        "ALICE",
        ["c1"],
        chunk_entities={"c1": {"ALICE": [{"description": LONG_DESCRIPTION}]}},
        llm_response_cache=None,
        global_config=cfg,
    )
    if fail_truncation:
        with pytest.raises(TokenBudgetError):
            await coro
        assert graph.nodes["ALICE"] == node_before
        assert vdb.upsert_calls == 0
    else:
        await coro
        record = vdb.records[operate.compute_mdhash_id("ALICE", prefix="ent-")]
        assert len(cfg["tokenizer"].encode(record["content"])) <= 20
        assert record["content"] != "ALICE\n" + LONG_DESCRIPTION


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_truncation", [False, True], ids=["truncates", "fails"])
async def test_rebuild_single_relationship_truncation(fail_truncation: bool):
    graph = MemGraph(
        nodes={
            name: {"entity_id": name, "description": name, "source_id": "c1"}
            for name in ("A", "B")
        },
        edges={
            ("A", "B"): {
                "description": "short",
                "keywords": "k",
                "weight": 1.0,
                "source_id": "c1",
                "file_path": "f",
            }
        },
    )
    cfg = _cfg(fail=fail_truncation)
    entities_vdb = MemVDB(cfg)
    relationships_vdb = MemVDB(cfg)
    edge_before = dict(graph.edges[("A", "B")])

    coro = _rebuild_single_relationship(
        graph,
        relationships_vdb,
        entities_vdb,
        "A",
        "B",
        ["c1"],
        chunk_relationships={
            "c1": {("A", "B"): [{"description": LONG_DESCRIPTION, "keywords": "k"}]}
        },
        llm_response_cache=None,
        global_config=cfg,
    )
    if fail_truncation:
        with pytest.raises(TokenBudgetError):
            await coro
        assert graph.edges[("A", "B")] == edge_before
        assert relationships_vdb.upsert_calls == 0
    else:
        degraded = await coro
        assert degraded is False
        rel_id = operate.compute_mdhash_id("A" + "B", prefix="rel-")
        record = relationships_vdb.records[rel_id]
        assert len(cfg["tokenizer"].encode(record["content"])) <= 20


@pytest.mark.asyncio
async def test_rebuild_single_relationship_endpoint_created_when_relation_content_fails():
    """Per-object front-loading: missing endpoint B is created (graph + VDB),
    while the failing relationship leaves the edge unchanged and VDB unwritten."""
    graph = MemGraph(
        nodes={"A": {"entity_id": "A", "description": "short", "source_id": "c1"}},
        edges={
            ("A", "B"): {
                "description": "short",
                "keywords": "k",
                "weight": 1.0,
                "source_id": "c1",
                "file_path": "f",
            }
        },
    )
    cfg = _cfg(tokenizer=_FailsOnlyForRelationContentTokenizer())
    entities_vdb = MemVDB(cfg)
    relationships_vdb = MemVDB(cfg)

    with pytest.raises(TokenBudgetError):
        await _rebuild_single_relationship(
            graph,
            relationships_vdb,
            entities_vdb,
            "A",
            "B",
            ["c1"],
            chunk_relationships={
                "c1": {("A", "B"): [{"description": "short", "keywords": "k"}]}
            },
            llm_response_cache=None,
            global_config=cfg,
            structural_fallback=True,
        )

    assert "B" in graph.nodes
    assert entities_vdb.upsert_calls == 1
    assert graph.edges[("A", "B")]["description"] == "short"
    assert relationships_vdb.upsert_calls == 0


# --------------------------------------------------------------------------- #
# 3. operate.py: merge (ingestion) paths
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_truncation", [False, True], ids=["truncates", "fails"])
async def test_merge_nodes_then_upsert_truncation(fail_truncation: bool):
    graph = MemGraph()
    cfg = _cfg(fail=fail_truncation)
    vdb = MemVDB(cfg)

    coro = _merge_nodes_then_upsert(
        "ALICE",
        [
            {
                "entity_type": "PERSON",
                "description": LONG_DESCRIPTION,
                "source_id": "c1",
                "file_path": "f",
            }
        ],
        graph,
        vdb,
        cfg,
    )
    if fail_truncation:
        with pytest.raises(TokenBudgetError):
            await coro
        assert "ALICE" not in graph.nodes
        assert vdb.upsert_calls == 0
    else:
        node_data = await coro
        assert node_data["entity_id"] == "ALICE"
        record = vdb.records[operate.compute_mdhash_id("ALICE", prefix="ent-")]
        assert len(cfg["tokenizer"].encode(record["content"])) <= 20


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_truncation", [False, True], ids=["truncates", "fails"])
async def test_merge_edges_then_upsert_truncation(fail_truncation: bool):
    graph = MemGraph(
        nodes={
            # Matching source_id skips endpoint-node updates so truncation
            # failure is isolated to the relationship's own VDB payload.
            name: {"entity_id": name, "description": name, "source_id": "c1"}
            for name in ("A", "B")
        }
    )
    cfg = _cfg(fail=fail_truncation)
    entities_vdb = MemVDB(cfg)
    relationships_vdb = MemVDB(cfg)

    coro = _merge_edges_then_upsert(
        "A",
        "B",
        [
            {
                "description": LONG_DESCRIPTION,
                "keywords": "k",
                "weight": 1.0,
                "source_id": "c1",
            }
        ],
        graph,
        relationships_vdb,
        entities_vdb,
        cfg,
    )
    if fail_truncation:
        with pytest.raises(TokenBudgetError):
            await coro
        assert ("A", "B") not in graph.edges and ("B", "A") not in graph.edges
        assert relationships_vdb.upsert_calls == 0
    else:
        await coro
        rel_id = operate.compute_mdhash_id("A" + "B", prefix="rel-")
        record = relationships_vdb.records[rel_id]
        assert len(cfg["tokenizer"].encode(record["content"])) <= 20


@pytest.mark.asyncio
async def test_merge_edges_then_upsert_new_endpoint_entity_content_truncated():
    """The new-endpoint-entity branch (added_entities path) also truncates."""
    graph = MemGraph()
    cfg = _cfg(fail=False)
    entities_vdb = MemVDB(cfg)
    relationships_vdb = MemVDB(cfg)

    await _merge_edges_then_upsert(
        "NEW_A",
        "NEW_B",
        [
            {
                "description": LONG_DESCRIPTION,
                "keywords": "k",
                "weight": 1.0,
                "source_id": "c1",
            }
        ],
        graph,
        relationships_vdb,
        entities_vdb,
        cfg,
    )
    for name in ("NEW_A", "NEW_B"):
        record = entities_vdb.records[operate.compute_mdhash_id(name, prefix="ent-")]
        assert len(cfg["tokenizer"].encode(record["content"])) <= 20


@pytest.mark.asyncio
async def test_merge_edges_then_upsert_new_endpoints_created_when_relation_content_fails():
    """Per-object front-loading: both new endpoints are created (graph + VDB),
    while the failing relationship leaves the edge unwritten."""
    graph = MemGraph()
    cfg = _cfg(tokenizer=_FailsOnlyForRelationContentTokenizer())
    entities_vdb = MemVDB(cfg)
    relationships_vdb = MemVDB(cfg)

    with pytest.raises(TokenBudgetError):
        await _merge_edges_then_upsert(
            "NEW_A",
            "NEW_B",
            [
                {
                    "description": "short",
                    "keywords": "k",
                    "weight": 1.0,
                    "source_id": "c1",
                }
            ],
            graph,
            relationships_vdb,
            entities_vdb,
            cfg,
        )

    assert "NEW_A" in graph.nodes
    assert "NEW_B" in graph.nodes
    assert ("NEW_A", "NEW_B") not in graph.edges
    assert entities_vdb.upsert_calls == 2
    assert relationships_vdb.upsert_calls == 0


# --------------------------------------------------------------------------- #
# 4. utils_graph.py: create / edit / rename / merge
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target", "fail_truncation"),
    [
        pytest.param("entity", False, id="entity_truncates"),
        pytest.param("entity", True, id="entity_fails"),
        pytest.param("relation", False, id="relation_truncates"),
        pytest.param("relation", True, id="relation_fails"),
    ],
)
async def test_acreate_entity_and_relation_truncation(
    target: str, fail_truncation: bool
):
    graph = MemGraph(
        nodes={
            name: {"entity_id": name, "description": name}
            for name in (("A", "B") if target == "relation" else ())
        }
    )
    cfg = _cfg(fail=fail_truncation)
    entities_vdb = MemVDB(cfg)
    relationships_vdb = MemVDB(cfg)

    if target == "entity":
        coro = acreate_entity(
            graph,
            entities_vdb,
            relationships_vdb,
            "ALICE",
            {"description": LONG_DESCRIPTION, "entity_type": "PERSON"},
        )
    else:
        coro = acreate_relation(
            graph,
            entities_vdb,
            relationships_vdb,
            "A",
            "B",
            {"description": LONG_DESCRIPTION, "keywords": "k"},
        )

    if fail_truncation:
        with pytest.raises(TokenBudgetError):
            await coro
        if target == "entity":
            assert "ALICE" not in graph.nodes
            assert entities_vdb.upsert_calls == 0
        else:
            assert ("A", "B") not in graph.edges and ("B", "A") not in graph.edges
            assert relationships_vdb.upsert_calls == 0
    else:
        await coro
        if target == "entity":
            record = entities_vdb.records[
                operate.compute_mdhash_id("ALICE", prefix="ent-")
            ]
        else:
            record = relationships_vdb.records[
                operate.compute_mdhash_id("A" + "B", prefix="rel-")
            ]
        assert len(cfg["tokenizer"].encode(record["content"])) <= 20


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_truncation", [False, True], ids=["truncates", "fails"])
async def test_edit_relation_truncation(fail_truncation: bool):
    original_edge = {
        "description": "short",
        "keywords": "k",
        "weight": 1.0,
        "source_id": "c1",
        "file_path": "f",
    }
    graph = MemGraph(
        nodes={name: {"entity_id": name, "description": name} for name in ("A", "B")},
        edges={("A", "B"): dict(original_edge)},
    )
    cfg = _cfg(fail=fail_truncation)
    entities_vdb = MemVDB(cfg)
    relationships_vdb = MemVDB(cfg)
    rel_id = operate.compute_mdhash_id("A" + "B", prefix="rel-")
    if fail_truncation:
        relationships_vdb.records[rel_id] = {"content": "pre-existing"}

    coro = aedit_relation(
        graph,
        entities_vdb,
        relationships_vdb,
        "A",
        "B",
        {"description": LONG_DESCRIPTION},
    )
    if fail_truncation:
        with pytest.raises(TokenBudgetError):
            await coro
        assert graph.edges[("A", "B")] == original_edge
        assert relationships_vdb.upsert_calls == 0
        assert rel_id in relationships_vdb.records  # delete never happened either
    else:
        await coro
        record = relationships_vdb.records[rel_id]
        assert len(cfg["tokenizer"].encode(record["content"])) <= 20


@pytest.mark.asyncio
async def test_edit_relation_vdb_delete_failure_still_leaves_graph_updated():
    """A transient VDB delete failure happens after the graph write and surfaces
    as ``VectorStorageConsistencyError`` while leaving the graph updated."""
    graph = MemGraph(
        nodes={name: {"entity_id": name, "description": name} for name in ("A", "B")},
        edges={
            ("A", "B"): {
                "description": "short",
                "keywords": "k",
                "weight": 1.0,
                "source_id": "c1",
                "file_path": "f",
            }
        },
    )
    cfg = _cfg(fail=False, limit=1000)
    entities_vdb = MemVDB(cfg)
    relationships_vdb = MemVDB(cfg)

    async def _boom(_ids):
        raise RuntimeError("transient VDB backend error")

    relationships_vdb.delete = _boom

    with pytest.raises(
        VectorStorageConsistencyError, match="transient VDB backend error"
    ):
        await aedit_relation(
            graph,
            entities_vdb,
            relationships_vdb,
            "A",
            "B",
            {"description": "updated description"},
        )

    assert graph.edges[("A", "B")]["description"] == "updated description"
    assert relationships_vdb.upsert_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "fail_truncation"),
    [
        pytest.param("rename", False, id="rename_truncates"),
        pytest.param("rename", True, id="rename_wraps_consistency_error"),
        pytest.param("merge", False, id="merge_truncates"),
        pytest.param("merge", True, id="merge_wraps_consistency_error"),
    ],
)
async def test_post_graph_mutation_truncation_wraps_consistency_error(
    operation: str, fail_truncation: bool
):
    """Rename and merge mutate the graph before building the entity VDB payload,
    so a truncation failure surfaces as ``VectorStorageConsistencyError``."""
    graph = MemGraph(
        nodes={
            "ALICE": {"entity_id": "ALICE", "description": "short", "source_id": "c1"}
        }
    )
    cfg = _cfg(fail=fail_truncation)
    entities_vdb = MemVDB(cfg)
    relationships_vdb = MemVDB(cfg)
    target_name = "ALICIA" if operation == "rename" else "ALICE_MERGED"

    if operation == "rename":
        coro = _edit_entity_impl(
            graph,
            entities_vdb,
            relationships_vdb,
            "ALICE",
            {"entity_name": target_name, "description": LONG_DESCRIPTION},
        )
    else:
        coro = _merge_entities_impl(
            graph,
            entities_vdb,
            relationships_vdb,
            ["ALICE"],
            target_name,
            target_entity_data={"description": LONG_DESCRIPTION},
        )

    if fail_truncation:
        with pytest.raises(VectorStorageConsistencyError):
            await coro
        assert target_name in graph.nodes
        assert entities_vdb.upsert_calls == 0
    else:
        await coro
        assert target_name in graph.nodes
        record = entities_vdb.records[
            operate.compute_mdhash_id(target_name, prefix="ent-")
        ]
        assert len(cfg["tokenizer"].encode(record["content"])) <= 20


# --------------------------------------------------------------------------- #
# 5. lightrag.py: ainsert_custom_kg
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_truncation", [False, True], ids=["truncates", "fails"])
async def test_ainsert_custom_kg_entity_truncation(fail_truncation: bool):
    cfg = _cfg(fail=fail_truncation)
    rag = _make_custom_kg_rag(cfg)

    coro = rag.ainsert_custom_kg(
        {
            "chunks": [],
            "entities": [
                {
                    "entity_name": "Alice",
                    "entity_type": "PERSON",
                    "description": LONG_DESCRIPTION,
                    "source_id": "chunk-1",
                    "file_path": "f",
                }
            ],
            "relationships": [],
        }
    )
    if fail_truncation:
        with pytest.raises(TokenBudgetError):
            await coro
        assert "Alice" not in rag.chunk_entity_relation_graph.nodes
        assert rag.entities_vdb.upsert_calls == 0
    else:
        await coro
        record = rag.entities_vdb.records[
            operate.compute_mdhash_id("Alice", prefix="ent-")
        ]
        assert len(cfg["tokenizer"].encode(record["content"])) <= 20


@pytest.mark.asyncio
async def test_ainsert_custom_kg_relationship_endpoints_keep_real_entity_data():
    """Explicit batch entities are upserted before missing-endpoint discovery so
    relationship endpoints never get clobbered with UNKNOWN placeholders."""
    cfg = _cfg(fail=False, limit=1000)
    rag = _make_custom_kg_rag(cfg)

    await rag.ainsert_custom_kg(
        {
            "chunks": [],
            "entities": [
                {
                    "entity_name": "Alice",
                    "entity_type": "PERSON",
                    "description": "Alice is a scientist.",
                    "source_id": "chunk-1",
                    "file_path": "f",
                },
                {
                    "entity_name": "Bob",
                    "entity_type": "PERSON",
                    "description": "Bob is an engineer.",
                    "source_id": "chunk-1",
                    "file_path": "f",
                },
            ],
            "relationships": [
                {
                    "src_id": "Alice",
                    "tgt_id": "Bob",
                    "description": "Alice and Bob are colleagues.",
                    "keywords": "colleague",
                    "weight": 1.0,
                    "source_id": "chunk-1",
                    "file_path": "f",
                }
            ],
        }
    )

    alice = rag.chunk_entity_relation_graph.nodes["Alice"]
    bob = rag.chunk_entity_relation_graph.nodes["Bob"]
    assert alice["description"] == "Alice is a scientist."
    assert alice["entity_type"] == "PERSON"
    assert bob["description"] == "Bob is an engineer."
    assert bob["entity_type"] == "PERSON"
    assert ("Alice", "Bob") in rag.chunk_entity_relation_graph.edges


@pytest.mark.asyncio
async def test_ainsert_custom_kg_entities_created_but_not_edge_when_relationship_content_fails():
    """Per-object front-loading: valid entities land in the graph before a
    failing relationship aborts the edge write and both VDB upserts."""
    cfg = _cfg(tokenizer=_FailsOnlyForRelationContentTokenizer())
    rag = _make_custom_kg_rag(cfg)

    with pytest.raises(TokenBudgetError):
        await rag.ainsert_custom_kg(
            {
                "chunks": [],
                "entities": [
                    {
                        "entity_name": "Alice",
                        "entity_type": "PERSON",
                        "description": "short",
                        "source_id": "chunk-1",
                        "file_path": "f",
                    },
                    {
                        "entity_name": "Bob",
                        "entity_type": "PERSON",
                        "description": "short",
                        "source_id": "chunk-1",
                        "file_path": "f",
                    },
                ],
                "relationships": [
                    {
                        "src_id": "Alice",
                        "tgt_id": "Bob",
                        "description": "short",
                        "keywords": "k",
                        "weight": 1.0,
                        "source_id": "chunk-1",
                        "file_path": "f",
                    }
                ],
            }
        )

    assert "Alice" in rag.chunk_entity_relation_graph.nodes
    assert "Bob" in rag.chunk_entity_relation_graph.nodes
    assert ("Alice", "Bob") not in rag.chunk_entity_relation_graph.edges
    assert rag.entities_vdb.upsert_calls == 0
    assert rag.relationships_vdb.upsert_calls == 0


@pytest.mark.asyncio
async def test_ainsert_custom_kg_empty_batch_never_builds_global_config():
    """The empty-batch fast path skips _build_global_config completely."""

    def _boom():
        raise AssertionError("_build_global_config should not be called")

    cfg = _cfg(fail=False)
    rag = _make_custom_kg_rag(cfg)
    rag._build_global_config = _boom

    await rag.ainsert_custom_kg({"chunks": [], "entities": [], "relationships": []})
