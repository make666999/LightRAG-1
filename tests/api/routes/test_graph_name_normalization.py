"""Graph entity/relation name normalization, whitespace validation, and REST manual name resolution lifecycle."""

from __future__ import annotations

import asyncio
import importlib
import sys
from copy import deepcopy
from uuid import uuid4

import numpy as np
import pytest
from pydantic import ValidationError

from lightrag import LightRAG, utils_graph
from lightrag.utils import (
    EmbeddingFunc,
    Tokenizer,
    compute_mdhash_id,
    make_relation_chunk_key,
)

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_gr = importlib.import_module("lightrag.api.routers.graph_routes")
sys.argv = _original_argv

EntityCreateRequest = _gr.EntityCreateRequest
EntityUpdateRequest = _gr.EntityUpdateRequest
EntityMergeRequest = _gr.EntityMergeRequest
RelationCreateRequest = _gr.RelationCreateRequest
RelationUpdateRequest = _gr.RelationUpdateRequest
DeleteEntityRequest = _gr.DeleteEntityRequest

pytestmark = pytest.mark.offline


class _NoopLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _VectorStorage:
    def __init__(self):
        self.global_config = {"workspace": ""}
        self.records = {}
        self.deleted_ids = []

    async def upsert(self, data):
        self.records.update(deepcopy(data))

    async def delete(self, ids):
        for record_id in ids:
            self.deleted_ids.append(record_id)
            self.records.pop(record_id, None)

    async def index_done_callback(self):
        return None


class _KVStorage:
    """Minimal KV storage for chunk tracking; ``get_by_id`` returns None for empty rows."""

    def __init__(self):
        self.global_config = {"workspace": ""}
        self.records = {}

    async def get_by_id(self, key):
        return self.records.get(key)

    async def upsert(self, data):
        self.records.update(deepcopy(data))

    async def delete(self, ids):
        for record_id in ids:
            self.records.pop(record_id, None)

    async def index_done_callback(self):
        return None


# --------------------------------------------------------------------------- #
# Entity/relation creation, edit, and merge name normalization (unit doubles)
# --------------------------------------------------------------------------- #


class _EntityGraph:
    def __init__(self, nodes=None):
        self.nodes = deepcopy(nodes or {})
        self.edges = {}
        self.deleted_nodes = []

    async def has_node(self, entity_name):
        return entity_name in self.nodes

    async def get_node(self, entity_name):
        node = self.nodes.get(entity_name)
        return deepcopy(node) if node is not None else None

    async def upsert_node(self, entity_name, node_data):
        self.nodes[entity_name] = deepcopy(node_data)

    async def get_node_edges(self, entity_name):
        return []

    async def has_edge(self, source_entity, target_entity):
        return (source_entity, target_entity) in self.edges or (
            target_entity,
            source_entity,
        ) in self.edges

    async def get_edge(self, source_entity, target_entity):
        edge = self.edges.get((source_entity, target_entity))
        if edge is None:
            edge = self.edges.get((target_entity, source_entity))
        return deepcopy(edge) if edge is not None else None

    async def upsert_edge(self, source_entity, target_entity, edge_data):
        self.edges[(source_entity, target_entity)] = deepcopy(edge_data)

    async def delete_node(self, entity_name):
        self.deleted_nodes.append(entity_name)
        self.nodes.pop(entity_name, None)

    async def index_done_callback(self):
        return None


def _source_company_graph():
    return _EntityGraph(
        {
            "Source公司": {
                "entity_id": "Source公司",
                "description": "old",
                "entity_type": "organization",
                "source_id": "manual_creation",
            }
        }
    )


class TestEntityNameNormalization:
    """Business-layer normalization for manual entity mutations."""

    @pytest.fixture(autouse=True)
    def patch_graph_lock(self, monkeypatch):
        monkeypatch.setattr(
            utils_graph,
            "get_storage_keyed_lock",
            lambda *args, **kwargs: _NoopLock(),
        )

    @pytest.mark.asyncio
    async def test_create_entity_uses_extraction_name_normalization(self):
        graph = _EntityGraph()
        entities_vdb = _VectorStorage()
        relationships_vdb = _VectorStorage()

        result = await utils_graph.acreate_entity(
            graph,
            entities_vdb,
            relationships_vdb,
            "  “Ａ 公 司”  ",
            {"description": "Company description", "entity_type": "organization"},
        )

        assert result["entity_name"] == "A公司"
        assert set(graph.nodes) == {"A公司"}
        entity_id = compute_mdhash_id("A公司", prefix="ent-")
        assert entities_vdb.records[entity_id]["entity_name"] == "A公司"

    @pytest.mark.asyncio
    async def test_create_entity_rejects_name_removed_by_normalization(self):
        with pytest.raises(ValueError, match="empty after normalization"):
            await utils_graph.acreate_entity(
                _EntityGraph(),
                _VectorStorage(),
                _VectorStorage(),
                "1",
                {"description": "Invalid numeric identifier"},
            )

    @pytest.mark.asyncio
    async def test_edit_without_rename_resolves_normalized_source(self):
        graph = _source_company_graph()

        result = await utils_graph.aedit_entity(
            graph,
            _VectorStorage(),
            _VectorStorage(),
            "Ｓｏｕｒｃｅ 公 司",
            {"description": "updated"},
            allow_rename=False,
        )

        assert set(graph.nodes) == {"Source公司"}
        assert graph.nodes["Source公司"]["description"] == "updated"
        assert result["entity_name"] == "Source公司"

    @pytest.mark.parametrize("allow_merge", [False, True])
    @pytest.mark.asyncio
    async def test_rename_refuses_a_source_spelling_that_is_not_stored(
        self, allow_merge
    ):
        graph = _source_company_graph()

        with pytest.raises(ValueError, match="it is stored as 'Source公司'"):
            await utils_graph.aedit_entity(
                graph,
                _VectorStorage(),
                _VectorStorage(),
                "Ｓｏｕｒｃｅ 公 司",
                {"entity_name": "T目标", "description": "renamed"},
                allow_rename=True,
                allow_merge=allow_merge,
            )

        assert set(graph.nodes) == {"Source公司"}
        assert graph.nodes["Source公司"]["description"] == "old"

    @pytest.mark.asyncio
    async def test_rename_by_stored_name_normalizes_rename_target(self):
        graph = _source_company_graph()
        entities_vdb = _VectorStorage()
        relationships_vdb = _VectorStorage()
        updated_data = {"entity_name": " “Ｔ 目 标” ", "description": "renamed"}

        result = await utils_graph.aedit_entity(
            graph,
            entities_vdb,
            relationships_vdb,
            "Source公司",
            updated_data,
            allow_rename=True,
        )

        assert updated_data["entity_name"] == " “Ｔ 目 标” "
        assert "Source公司" not in graph.nodes
        assert graph.nodes["T目标"]["entity_id"] == "T目标"
        assert result["entity_name"] == "T目标"
        assert result["operation_summary"]["final_entity"] == "T目标"
        assert result["operation_summary"]["renamed"] is True

    @pytest.mark.asyncio
    async def test_edit_prefers_exact_legacy_entity_key(self):
        legacy_name = "“Ａ 公 司”"
        graph = _EntityGraph(
            {
                legacy_name: {
                    "entity_id": legacy_name,
                    "description": "old",
                    "entity_type": "organization",
                    "source_id": "manual_creation",
                }
            }
        )

        result = await utils_graph.aedit_entity(
            graph,
            _VectorStorage(),
            _VectorStorage(),
            legacy_name,
            {"description": "updated"},
            allow_rename=False,
        )

        assert set(graph.nodes) == {legacy_name}
        assert graph.nodes[legacy_name]["description"] == "updated"
        assert result["entity_name"] == legacy_name

    @pytest.mark.asyncio
    async def test_edit_preserves_exact_legacy_name_that_normalizes_to_empty(self):
        legacy_name = "1"
        graph = _EntityGraph(
            {
                legacy_name: {
                    "entity_id": legacy_name,
                    "description": "old",
                    "source_id": "manual_creation",
                }
            }
        )

        result = await utils_graph.aedit_entity(
            graph,
            _VectorStorage(),
            _VectorStorage(),
            legacy_name,
            {"description": "updated"},
            allow_rename=False,
        )

        assert graph.nodes[legacy_name]["description"] == "updated"
        assert result["entity_name"] == legacy_name

    @pytest.mark.asyncio
    async def test_create_refuses_duplicate_exact_legacy_key(self):
        legacy_name = "“Ａ 公 司”"
        graph = _EntityGraph(
            {
                legacy_name: {
                    "entity_id": legacy_name,
                    "description": "legacy",
                }
            }
        )

        with pytest.raises(ValueError, match="already exists"):
            await utils_graph.acreate_entity(
                graph,
                _VectorStorage(),
                _VectorStorage(),
                legacy_name,
                {"description": "duplicate"},
            )

        assert set(graph.nodes) == {legacy_name}

    @pytest.mark.asyncio
    async def test_merge_refuses_a_source_spelling_that_is_not_stored(self):
        graph = _source_company_graph()

        with pytest.raises(ValueError, match="it is stored as 'Source公司'"):
            await utils_graph.amerge_entities(
                graph,
                _VectorStorage(),
                _VectorStorage(),
                ["Ｓｏｕｒｃｅ 公 司"],
                "T目标",
            )

        assert set(graph.nodes) == {"Source公司"}
        assert graph.deleted_nodes == []

    @pytest.mark.asyncio
    async def test_merge_dedupes_sources_and_creates_normalized_target_once(self):
        graph = _EntityGraph(
            {
                "Source公司": {
                    "entity_id": "Source公司",
                    "description": "source",
                    "entity_type": "organization",
                    "source_id": "manual_creation",
                }
            }
        )
        entities_vdb = _VectorStorage()

        result = await utils_graph.amerge_entities(
            graph,
            entities_vdb,
            _VectorStorage(),
            ["Source公司", "Source公司"],
            " “Ｔ 目 标” ",
        )

        assert result["entity_name"] == "T目标"
        assert set(graph.nodes) == {"T目标"}
        assert graph.nodes["T目标"]["entity_id"] == "T目标"
        assert graph.deleted_nodes == ["Source公司"]
        target_id = compute_mdhash_id("T目标", prefix="ent-")
        assert entities_vdb.records[target_id]["entity_name"] == "T目标"

    @pytest.mark.asyncio
    async def test_merge_preserves_exact_legacy_source_and_target_keys(self):
        legacy_source = "1"
        legacy_target = "“Ａ 公 司”"
        graph = _EntityGraph(
            {
                legacy_source: {
                    "entity_id": legacy_source,
                    "description": "source",
                    "source_id": "manual_creation",
                },
                legacy_target: {
                    "entity_id": legacy_target,
                    "description": "target",
                    "source_id": "manual_creation",
                },
            }
        )

        result = await utils_graph.amerge_entities(
            graph,
            _VectorStorage(),
            _VectorStorage(),
            [legacy_source],
            legacy_target,
        )

        assert result["entity_name"] == legacy_target
        assert set(graph.nodes) == {legacy_target}
        assert graph.deleted_nodes == [legacy_source]

    @pytest.mark.asyncio
    async def test_create_relation_accepts_the_name_create_entity_normalized(self):
        graph = _EntityGraph()
        entities_vdb = _VectorStorage()
        relationships_vdb = _VectorStorage()

        await utils_graph.acreate_entity(
            graph,
            entities_vdb,
            relationships_vdb,
            "GoodA",
            {"description": "supplier", "entity_type": "organization"},
        )
        await utils_graph.acreate_entity(
            graph,
            entities_vdb,
            relationships_vdb,
            "Tesla（US）",
            {"description": "electric vehicles", "entity_type": "organization"},
        )

        assert set(graph.nodes) == {"GoodA", "Tesla(US)"}

        result = await utils_graph.acreate_relation(
            graph,
            entities_vdb,
            relationships_vdb,
            "GoodA",
            "Tesla（US）",
            {"description": "supplies"},
        )

        assert set(graph.edges) == {("GoodA", "Tesla(US)")}
        assert {result["src_entity"], result["tgt_entity"]} == {"GoodA", "Tesla(US)"}

    @pytest.mark.asyncio
    async def test_create_relation_accepts_fully_normalized_spellings(self):
        graph = _EntityGraph(
            {
                "A公司": {
                    "entity_id": "A公司",
                    "description": "old",
                    "entity_type": "organization",
                    "source_id": "manual_creation",
                },
                "B公司": {
                    "entity_id": "B公司",
                    "description": "old",
                    "entity_type": "organization",
                    "source_id": "manual_creation",
                },
            }
        )

        result = await utils_graph.acreate_relation(
            graph,
            _VectorStorage(),
            _VectorStorage(),
            " “Ｂ 公 司” ",
            "  “Ａ 公 司”  ",
            {"description": "B supplies A"},
        )

        assert set(graph.edges) == {("B公司", "A公司")}
        assert {result["src_entity"], result["tgt_entity"]} == {"A公司", "B公司"}

    @pytest.mark.asyncio
    async def test_create_relation_prefers_exact_legacy_endpoint_key(self):
        legacy_name = "“Ａ 公 司”"
        graph = _EntityGraph(
            {
                legacy_name: {
                    "entity_id": legacy_name,
                    "description": "legacy",
                    "entity_type": "organization",
                    "source_id": "manual_creation",
                },
                "B公司": {
                    "entity_id": "B公司",
                    "description": "old",
                    "entity_type": "organization",
                    "source_id": "manual_creation",
                },
            }
        )

        result = await utils_graph.acreate_relation(
            graph,
            _VectorStorage(),
            _VectorStorage(),
            "B公司",
            legacy_name,
            {"description": "legacy endpoint"},
        )

        assert set(graph.nodes) == {legacy_name, "B公司"}
        assert set(graph.edges) == {("B公司", legacy_name)}
        assert {result["src_entity"], result["tgt_entity"]} == {"B公司", legacy_name}

    @pytest.mark.asyncio
    async def test_create_relation_rejects_endpoints_that_collapse_into_one_node(self):
        graph = _EntityGraph(
            {
                "Tesla(US)": {
                    "entity_id": "Tesla(US)",
                    "description": "old",
                    "entity_type": "organization",
                    "source_id": "manual_creation",
                }
            }
        )

        with pytest.raises(ValueError, match=r"self-loop relation on 'Tesla\(US\)'"):
            await utils_graph.acreate_relation(
                graph,
                _VectorStorage(),
                _VectorStorage(),
                "Tesla（US）",
                "Tesla(US)",
                {"description": "would be a self-loop"},
            )

        assert graph.edges == {}

    @pytest.mark.asyncio
    async def test_create_relation_prefers_exact_legacy_source_key(self):
        legacy_name = "“Ａ 公 司”"
        graph = _EntityGraph(
            {
                legacy_name: {
                    "entity_id": legacy_name,
                    "description": "legacy",
                    "entity_type": "organization",
                    "source_id": "manual_creation",
                },
                "B公司": {
                    "entity_id": "B公司",
                    "description": "old",
                    "entity_type": "organization",
                    "source_id": "manual_creation",
                },
            }
        )

        result = await utils_graph.acreate_relation(
            graph,
            _VectorStorage(),
            _VectorStorage(),
            legacy_name,
            "B公司",
            {"description": "legacy endpoint"},
        )

        assert set(graph.nodes) == {legacy_name, "B公司"}
        assert set(graph.edges) == {(legacy_name, "B公司")}
        assert {result["src_entity"], result["tgt_entity"]} == {legacy_name, "B公司"}

    @pytest.mark.asyncio
    async def test_create_relation_rejects_endpoints_removed_by_normalization(self):
        graph = _EntityGraph(
            {
                "B公司": {
                    "entity_id": "B公司",
                    "description": "old",
                    "entity_type": "organization",
                    "source_id": "manual_creation",
                }
            }
        )

        with pytest.raises(ValueError, match="Source entity name cannot be empty"):
            await utils_graph.acreate_relation(
                graph,
                _VectorStorage(),
                _VectorStorage(),
                "1",
                "B公司",
                {"description": "erased source"},
            )

        with pytest.raises(ValueError, match="Target entity name cannot be empty"):
            await utils_graph.acreate_relation(
                graph,
                _VectorStorage(),
                _VectorStorage(),
                "B公司",
                "1",
                {"description": "erased target"},
            )

        assert graph.edges == {}

    @pytest.mark.asyncio
    async def test_create_relation_sees_one_edge_through_either_spelling(self):
        graph = _EntityGraph()
        entities_vdb = _VectorStorage()
        relationships_vdb = _VectorStorage()

        for name in ("GoodA", "Tesla（US）"):
            await utils_graph.acreate_entity(
                graph,
                entities_vdb,
                relationships_vdb,
                name,
                {"description": "d", "entity_type": "organization"},
            )

        await utils_graph.acreate_relation(
            graph,
            entities_vdb,
            relationships_vdb,
            "GoodA",
            "Tesla（US）",
            {"description": "supplies"},
        )

        with pytest.raises(ValueError, match="already exists"):
            await utils_graph.acreate_relation(
                graph,
                entities_vdb,
                relationships_vdb,
                "GoodA",
                "Tesla(US)",
                {"description": "supplies again"},
            )

        assert set(graph.edges) == {("GoodA", "Tesla(US)")}

    @pytest.mark.asyncio
    async def test_create_relation_locks_every_spelling(self, monkeypatch):
        captured = []

        def _spy(keys, **kwargs):
            captured.append((list(keys), kwargs.get("namespace")))
            return _NoopLock()

        monkeypatch.setattr(utils_graph, "get_storage_keyed_lock", _spy)

        graph = _EntityGraph(
            {
                "A公司": {
                    "entity_id": "A公司",
                    "description": "old",
                    "entity_type": "organization",
                    "source_id": "manual_creation",
                },
                "B公司": {
                    "entity_id": "B公司",
                    "description": "old",
                    "entity_type": "organization",
                    "source_id": "manual_creation",
                },
            }
        )

        await utils_graph.acreate_relation(
            graph,
            _VectorStorage(),
            _VectorStorage(),
            "Ｂ公司",
            "Ａ公司",
            {"description": "d"},
        )

        assert captured == [(["A公司", "B公司", "Ａ公司", "Ｂ公司"], "GraphDB")]


# --------------------------------------------------------------------------- #
# Whitespace validation on Pydantic request schemas
# --------------------------------------------------------------------------- #


class TestEntityNameWhitespace:
    """Graph create/update/merge reject whitespace-only entity names like delete."""

    @pytest.mark.parametrize("name", ["  "])
    def test_entity_create_rejects_whitespace_only_name(self, name):
        with pytest.raises(ValidationError):
            EntityCreateRequest(entity_name=name, entity_data={"description": "x"})

    def test_entity_create_strips_surrounding_whitespace(self):
        req = EntityCreateRequest(
            entity_name="  Tesla  ", entity_data={"description": "x"}
        )
        assert req.entity_name == "Tesla"

    @pytest.mark.parametrize("name", ["  "])
    def test_entity_update_rejects_whitespace_only_name(self, name):
        with pytest.raises(ValidationError):
            EntityUpdateRequest(entity_name=name, updated_data={"description": "x"})

    @pytest.mark.parametrize("name", ["  "])
    def test_entity_update_rejects_whitespace_only_rename_target(self, name):
        with pytest.raises(ValidationError):
            EntityUpdateRequest(
                entity_name="Tesla",
                updated_data={"entity_name": name},
                allow_rename=True,
            )

    def test_entity_update_strips_rename_target_name(self):
        req = EntityUpdateRequest(
            entity_name="Tesla",
            updated_data={"entity_name": "  Elon Musk  "},
            allow_rename=True,
        )
        assert req.updated_data["entity_name"] == "Elon Musk"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"entities_to_change": ["Keep"], "entity_to_change_into": "  "},
            {"entities_to_change": ["A", "  "], "entity_to_change_into": "Keep"},
        ],
    )
    def test_entity_merge_rejects_whitespace_only_names(self, kwargs):
        with pytest.raises(ValidationError):
            EntityMergeRequest(**kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {
                "source_entity": "  ",
                "target_entity": "B",
                "relation_data": {"weight": 1.0},
            },
            {
                "source_entity": "A",
                "target_entity": "  ",
                "relation_data": {"weight": 1.0},
            },
        ],
    )
    def test_relation_create_rejects_whitespace_only_endpoints(self, kwargs):
        with pytest.raises(ValidationError):
            RelationCreateRequest(**kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"source_id": "  ", "target_id": "B", "updated_data": {"weight": 1.0}},
            {"source_id": "A", "target_id": "\t", "updated_data": {"weight": 1.0}},
        ],
    )
    def test_relation_update_rejects_whitespace_only_endpoints(self, kwargs):
        with pytest.raises(ValidationError):
            RelationUpdateRequest(**kwargs)

    def test_delete_entity_still_rejects_whitespace_only_name(self):
        with pytest.raises(ValidationError):
            DeleteEntityRequest(entity_name="  ")


# --------------------------------------------------------------------------- #
# Relation edit/delete name normalization (unit doubles)
# --------------------------------------------------------------------------- #


class _RelationGraph:
    """Minimal in-process graph storage sufficient for aedit_relation / adelete_by_relation."""

    def __init__(self, nodes=None, edges=None):
        self.nodes = deepcopy(nodes or {})
        self.edges = {
            tuple(sorted(key)): deepcopy(value) for key, value in (edges or {}).items()
        }
        self.removed_edges = []

    async def has_node(self, entity_name):
        return entity_name in self.nodes

    async def has_edge(self, source, target):
        canonical = tuple(sorted([source, target]))
        return canonical in self.edges

    async def get_edge(self, source, target):
        canonical = tuple(sorted([source, target]))
        edge = self.edges.get(canonical)
        return deepcopy(edge) if edge is not None else None

    async def upsert_edge(self, source, target, edge_data):
        canonical = tuple(sorted([source, target]))
        self.edges[canonical] = deepcopy(edge_data)

    async def remove_edges(self, edges):
        for source, target in edges:
            canonical = tuple(sorted([source, target]))
            self.removed_edges.append(canonical)
            self.edges.pop(canonical, None)

    async def index_done_callback(self):
        return None


class TestRelationEditDeleteNameNormalization:
    """Business-layer normalization for manual relation edit and delete."""

    @pytest.fixture(autouse=True)
    def captured_lock_keys(self, monkeypatch):
        captured: list[list[str]] = []

        def _capturing_lock(*args, **kwargs):
            keys = list(args[0]) if args else list(kwargs.get("keys", []))
            captured.append(keys)
            return _NoopLock()

        monkeypatch.setattr(utils_graph, "get_storage_keyed_lock", _capturing_lock)
        return captured

    @pytest.mark.asyncio
    async def test_edit_relation_resolves_normalized_endpoint_spelling(
        self, captured_lock_keys
    ):
        norm_a = "Tesla(US)"
        norm_b = "SpaceX"
        raw_a = "Tesla（US）"
        graph = _RelationGraph(
            edges={(norm_a, norm_b): {"description": "old", "source_id": "chunk-1"}}
        )

        result = await utils_graph.aedit_relation(
            graph,
            _VectorStorage(),
            _VectorStorage(),
            raw_a,
            norm_b,
            {"description": "new", "keywords": "k"},
        )

        assert (result["src_entity"], result["tgt_entity"]) == tuple(
            sorted([norm_a, norm_b])
        )
        assert result["graph_data"]["description"] == "new"
        canonical_key = tuple(sorted([norm_a, norm_b]))
        assert graph.edges[canonical_key]["description"] == "new"
        assert graph.edges[canonical_key]["keywords"] == "k"
        assert graph.edges[canonical_key]["source_id"] == "chunk-1"
        assert captured_lock_keys, "expected get_storage_keyed_lock to be called"
        assert {raw_a, norm_a, norm_b}.issubset(set(captured_lock_keys[0]))

    @pytest.mark.asyncio
    async def test_edit_relation_prefers_exact_legacy_endpoint_key(self):
        legacy = "“Ａ 公 司”"
        partner = "SpaceX"
        graph = _RelationGraph(
            nodes={legacy: {"entity_id": legacy, "description": "legacy"}},
            edges={
                (legacy, partner): {
                    "description": "legacy edge",
                    "source_id": "manual_creation",
                },
            },
        )

        result = await utils_graph.aedit_relation(
            graph,
            _VectorStorage(),
            _VectorStorage(),
            legacy,
            partner,
            {"description": "renamed"},
        )

        assert (result["src_entity"], result["tgt_entity"]) == tuple(
            sorted([legacy, partner])
        )
        assert result["graph_data"]["description"] == "renamed"
        canonical_key = tuple(sorted([legacy, partner]))
        assert graph.edges[canonical_key]["description"] == "renamed"

    @pytest.mark.asyncio
    async def test_edit_relation_uses_canonical_storage_key_for_vdb(self):
        norm_a = "Tesla(US)"
        norm_b = "SpaceX"
        raw_a = "Tesla（US）"
        graph = _RelationGraph(
            edges={(norm_a, norm_b): {"description": "old", "source_id": "chunk-1"}}
        )
        relationships_vdb = _VectorStorage()
        canonical_vdb_src, canonical_vdb_tgt = sorted([norm_a, norm_b])
        canonical_rel_id = compute_mdhash_id(
            canonical_vdb_src + canonical_vdb_tgt, prefix="rel-"
        )
        relationships_vdb.records[canonical_rel_id] = {"content": "old"}

        await utils_graph.aedit_relation(
            graph,
            _VectorStorage(),
            relationships_vdb,
            raw_a,
            norm_b,
            {"description": "new", "keywords": "k"},
        )

        reverse_rel_id = compute_mdhash_id(
            canonical_vdb_tgt + canonical_vdb_src, prefix="rel-"
        )
        assert {canonical_rel_id, reverse_rel_id}.issubset(
            set(relationships_vdb.deleted_ids)
        )
        upserted_ids = set(relationships_vdb.records)
        assert canonical_rel_id in upserted_ids
        assert compute_mdhash_id(raw_a + norm_b, prefix="rel-") not in upserted_ids
        assert compute_mdhash_id(norm_a + norm_b, prefix="rel-") not in upserted_ids

    @pytest.mark.asyncio
    async def test_edit_relation_refuses_normalized_self_loop(self):
        norm = "Tesla(US)"
        graph = _RelationGraph(
            edges={(norm, norm): {"description": "self", "source_id": "chunk-1"}}
        )

        with pytest.raises(ValueError, match="self-loop"):
            await utils_graph.aedit_relation(
                graph,
                _VectorStorage(),
                _VectorStorage(),
                "Tesla（US）",
                "Tesla(US)",
                {"description": "would-be self"},
            )

    @pytest.mark.asyncio
    async def test_delete_relation_resolves_normalized_endpoint_spelling(
        self, captured_lock_keys
    ):
        norm_a = "Tesla(US)"
        norm_b = "SpaceX"
        raw_a = "Tesla（US）"
        graph = _RelationGraph(
            edges={(norm_a, norm_b): {"description": "old", "source_id": "chunk-1"}}
        )

        result = await utils_graph.adelete_by_relation(
            graph,
            _VectorStorage(),
            raw_a,
            norm_b,
        )

        assert result.status == "success"
        assert result.status_code == 200
        assert graph.edges == {}
        assert graph.removed_edges == [tuple(sorted([norm_a, norm_b]))]
        assert captured_lock_keys, "expected get_storage_keyed_lock to be called"
        assert {raw_a, norm_a, norm_b}.issubset(set(captured_lock_keys[0]))

    @pytest.mark.asyncio
    async def test_delete_relation_prefers_exact_legacy_endpoint_key(self):
        legacy = "“Ａ 公 司”"
        partner = "SpaceX"
        graph = _RelationGraph(
            nodes={legacy: {"entity_id": legacy, "description": "legacy"}},
            edges={
                (legacy, partner): {
                    "description": "legacy edge",
                    "source_id": "manual_creation",
                },
            },
        )

        result = await utils_graph.adelete_by_relation(
            graph,
            _VectorStorage(),
            legacy,
            partner,
        )

        assert result.status == "success"
        assert graph.removed_edges == [tuple(sorted([legacy, partner]))]
        assert graph.edges == {}

    @pytest.mark.asyncio
    async def test_delete_relation_returns_not_found_for_missing_edge_under_normalized_spelling(
        self,
    ):
        graph = _RelationGraph(edges={})

        result = await utils_graph.adelete_by_relation(
            graph,
            _VectorStorage(),
            "Tesla（US）",
            "SpaceX",
        )

        assert result.status == "not_found"
        assert result.status_code == 404
        assert result.message == "Relation from 'SpaceX' to 'Tesla(US)' does not exist"

    @pytest.mark.parametrize(
        "spellings",
        [("Tesla(US)", "Tesla(US)"), ("Tesla（US）", "Tesla(US)")],
        ids=["identical", "two-spellings-of-one-node"],
    )
    @pytest.mark.asyncio
    async def test_delete_relation_removes_stored_self_loop(self, spellings):
        norm = "Tesla(US)"
        graph = _RelationGraph(
            edges={(norm, norm): {"description": "self", "source_id": "chunk-1"}}
        )
        relation_chunks = _KVStorage()
        relation_chunks.records[make_relation_chunk_key(norm, norm)] = {
            "chunk_ids": ["chunk-1"]
        }

        result = await utils_graph.adelete_by_relation(
            graph,
            _VectorStorage(),
            *spellings,
            relation_chunks_storage=relation_chunks,
        )

        assert result.status == "success"
        assert graph.edges == {}
        assert graph.removed_edges == [(norm, norm)]
        assert relation_chunks.records == {}


# --------------------------------------------------------------------------- #
# REST manual name resolution lifecycle (real LightRAG + real keyed locks)
# --------------------------------------------------------------------------- #

_API_KEY = "fixture-key"
_HEADERS = {"X-API-Key": _API_KEY}

_RAW_TO_STORED = [
    ("Tesla（US）", "Tesla(US)"),
    ("Tesla&nbsp;Inc", "Tesla Inc"),
    ('"Quoted"', "Quoted"),
    ("Tesla　Motors", "Tesla Motors"),
]


class _Characters:
    def encode(self, content):
        return [ord(c) for c in content]

    def decode(self, tokens):
        return "".join(chr(c) for c in tokens)


async def _embed(texts):
    return np.ones((len(texts), 8))


async def _no_llm(*args, **kwargs):
    raise AssertionError("Manual graph edits must not call an LLM")


async def _create_entity(client, name):
    response = await client.post(
        "/graph/entity/create",
        json={"entity_name": name, "entity_data": {"description": "d"}},
        headers=_HEADERS,
    )
    assert response.status_code == 200, response.text


async def _delete(client, resource, payload):
    return await client.request(
        "DELETE", f"/graph/{resource}/delete", json=payload, headers=_HEADERS
    )


async def _legacy_and_normalized_entities(rag):
    graph = rag.chunk_entity_relation_graph
    legacy = "Tesla（US）"
    await graph.upsert_node(
        legacy,
        {
            "entity_id": legacy,
            "entity_type": "ORG",
            "description": "legacy",
            "source_id": "",
            "file_path": "",
        },
    )
    await rag.entity_chunks.upsert({legacy: {"chunk_ids": ["chunk-1"], "count": 1}})
    await rag.acreate_entity("Tesla(US)", {"description": "a different entity"})
    await rag.acreate_entity("SpaceX", {"description": "partner"})
    await rag.acreate_entity("Tesla Group", {"description": "merge target"})
    await rag.acreate_relation("Tesla(US)", "SpaceX", {"description": "its own"})
    return graph, legacy


class TestManualNameResolutionLifecycle:
    """REST routes against a real LightRAG (real locks and admin write gate)."""

    @pytest.fixture
    async def rag(self, tmp_path):
        instance = LightRAG(
            working_dir=str(tmp_path),
            workspace=f"names_{uuid4().hex}",
            llm_model_func=_no_llm,
            embedding_func=EmbeddingFunc(embedding_dim=8, func=_embed),
            tokenizer=Tokenizer("characters", _Characters()),
        )
        await instance.initialize_storages()
        try:
            yield instance
        finally:
            await instance.finalize_storages()

    @pytest.fixture
    async def client(self, rag, monkeypatch):
        monkeypatch.setattr(sys, "argv", [sys.argv[0]])
        from fastapi import FastAPI
        from httpx import ASGITransport, AsyncClient

        from lightrag.api.routers.graph_routes import create_graph_routes

        app = FastAPI()
        app.include_router(create_graph_routes(rag, api_key=_API_KEY))
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as http:
            yield http

    @pytest.mark.asyncio
    @pytest.mark.parametrize("raw, stored", _RAW_TO_STORED)
    async def test_raw_spelling_drives_the_whole_lifecycle(
        self, rag, client, raw, stored
    ):
        graph = rag.chunk_entity_relation_graph
        await _create_entity(client, raw)
        await _create_entity(client, "SpaceX")
        assert await graph.has_node(stored)

        exists = await client.get(
            "/graph/entity/exists", params={"name": raw}, headers=_HEADERS
        )
        assert exists.json() == {"exists": True}

        created = await client.post(
            "/graph/relation/create",
            json={
                "source_entity": raw,
                "target_entity": "SpaceX",
                "relation_data": {"description": "rel"},
            },
            headers=_HEADERS,
        )
        assert created.status_code == 200, created.text
        assert await graph.has_edge(stored, "SpaceX")

        duplicate = await client.post(
            "/graph/relation/create",
            json={
                "source_entity": stored,
                "target_entity": "SpaceX",
                "relation_data": {"description": "rel"},
            },
            headers=_HEADERS,
        )
        assert duplicate.status_code == 400
        assert "already exists" in duplicate.json()["detail"]

        edited = await client.post(
            "/graph/relation/edit",
            json={
                "source_id": "SpaceX",
                "target_id": raw,
                "updated_data": {"description": "edited"},
            },
            headers=_HEADERS,
        )
        assert edited.status_code == 200, edited.text
        assert (await graph.get_edge(stored, "SpaceX"))["description"] == "edited"

        info = await rag.get_relation_info(raw, "SpaceX")
        assert info["graph_data"]["description"] == "edited"
        assert (await rag.get_entity_info(raw))["entity_name"] == stored

        deleted = await _delete(
            client, "relation", {"source_entity": raw, "target_entity": "SpaceX"}
        )
        assert deleted.status_code == 200, deleted.text
        assert not await graph.has_edge(stored, "SpaceX")
        key = make_relation_chunk_key(stored, "SpaceX")
        assert await rag.relation_chunks.get_by_id(key) is None

        refused = await _delete(client, "entity", {"entity_name": raw})
        assert refused.status_code == 404
        assert f"it is stored as '{stored}'" in refused.json()["detail"]
        assert await graph.has_node(stored)

        deleted = await _delete(client, "entity", {"entity_name": stored})
        assert deleted.status_code == 200, deleted.text
        assert not await graph.has_node(stored)

    @pytest.mark.asyncio
    async def test_exact_legacy_node_wins_over_its_normalized_spelling(
        self, rag, client
    ):
        graph = rag.chunk_entity_relation_graph
        legacy = "Tesla（US）"
        await graph.upsert_node(
            legacy,
            {
                "entity_id": legacy,
                "entity_type": "ORG",
                "description": "legacy",
                "source_id": "",
                "file_path": "",
            },
        )
        await _create_entity(client, "Tesla(US)")
        await _create_entity(client, "SpaceX")

        created = await client.post(
            "/graph/relation/create",
            json={
                "source_entity": legacy,
                "target_entity": "SpaceX",
                "relation_data": {"description": "rel"},
            },
            headers=_HEADERS,
        )
        assert created.status_code == 200, created.text
        assert await graph.has_edge(legacy, "SpaceX")
        assert not await graph.has_edge("Tesla(US)", "SpaceX")

        deleted = await _delete(client, "entity", {"entity_name": legacy})
        assert deleted.status_code == 200, deleted.text
        assert not await graph.has_node(legacy)
        assert await graph.has_node("Tesla(US)")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "spellings",
        [("Loop", "Loop"), ('"Loop"', "Loop")],
        ids=["identical", "two-spellings-of-one-node"],
    )
    async def test_stored_self_loop_is_deleted_without_hanging(
        self, rag, client, spellings
    ):
        graph = rag.chunk_entity_relation_graph
        await _create_entity(client, "Loop")
        await graph.upsert_edge(
            "Loop",
            "Loop",
            {"description": "self", "keywords": "", "source_id": "", "weight": 1.0},
        )
        await graph.index_done_callback()

        response = await asyncio.wait_for(
            _delete(
                client,
                "relation",
                {"source_entity": spellings[0], "target_entity": spellings[1]},
            ),
            timeout=30,
        )

        assert response.status_code == 200, response.text
        assert not await graph.has_edge("Loop", "Loop")

    @pytest.mark.asyncio
    async def test_two_spellings_of_one_node_are_refused_by_create_and_edit(
        self, rag, client
    ):
        await _create_entity(client, "Loop")
        created = await client.post(
            "/graph/relation/create",
            json={
                "source_entity": '"Loop"',
                "target_entity": "Loop",
                "relation_data": {"description": "rel"},
            },
            headers=_HEADERS,
        )
        assert created.status_code == 400
        assert "self-loop" in created.json()["detail"]

        edited = await client.post(
            "/graph/relation/edit",
            json={
                "source_id": '"Loop"',
                "target_id": "Loop",
                "updated_data": {"description": "edited"},
            },
            headers=_HEADERS,
        )
        assert edited.status_code == 400
        assert "self-loop" in edited.json()["detail"]
        assert not await rag.chunk_entity_relation_graph.has_edge("Loop", "Loop")

    @pytest.mark.asyncio
    async def test_names_that_normalize_to_nothing_are_client_errors(self, rag, client):
        await _create_entity(client, "SpaceX")

        relation = await _delete(
            client, "relation", {"source_entity": "“”", "target_entity": "SpaceX"}
        )
        assert relation.status_code == 400
        assert "cannot be empty after normalization" in relation.json()["detail"]

        entity = await _delete(client, "entity", {"entity_name": "“”"})
        assert entity.status_code == 404

        assert await rag.chunk_entity_relation_graph.has_node("SpaceX")

    @pytest.mark.asyncio
    async def test_non_string_names_are_refused_without_raising(self, rag):
        relation = await rag.adelete_by_relation(None, "SpaceX")
        assert (relation.status, relation.status_code) == ("fail", 400)

        entity = await rag.adelete_by_entity(None)
        assert (entity.status, entity.status_code) == ("fail", 400)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "first_attempt", ["fails-after-node-removal", "succeeds"]
    )
    async def test_repeated_entity_delete_never_reaches_another_entity(
        self, rag, monkeypatch, first_attempt
    ):
        graph = rag.chunk_entity_relation_graph
        legacy = "Tesla（US）"
        await graph.upsert_node(
            legacy,
            {
                "entity_id": legacy,
                "entity_type": "ORG",
                "description": "legacy",
                "source_id": "",
                "file_path": "",
            },
        )
        await rag.entity_chunks.upsert(
            {legacy: {"chunk_ids": ["chunk-1"], "count": 1}}
        )
        await rag.acreate_entity("Tesla(US)", {"description": "a different entity"})

        if first_attempt == "fails-after-node-removal":

            async def _fail_tracking_cleanup(ids):
                raise RuntimeError("injected tracking failure")

            with monkeypatch.context() as patch:
                patch.setattr(rag.entity_chunks, "delete", _fail_tracking_cleanup)
                first = await rag.adelete_by_entity(legacy)
            assert (first.status, first.status_code) == ("fail", 500)
        else:
            first = await rag.adelete_by_entity(legacy)
            assert first.status == "success"
        assert not await graph.has_node(legacy)

        second = await rag.adelete_by_entity(legacy)

        assert (second.status, second.status_code) == ("not_found", 404)
        assert await graph.has_node("Tesla(US)")
        assert await rag.entity_chunks.get_by_id(legacy) is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "first_attempt", ["fails-after-node-removal", "succeeds"]
    )
    @pytest.mark.parametrize(
        "operation", ["merge_entities", "rename_with_allow_merge"]
    )
    async def test_repeated_merge_never_reaches_another_entity(
        self, rag, monkeypatch, operation, first_attempt
    ):
        graph, legacy = await _legacy_and_normalized_entities(rag)

        def _call():
            if operation == "merge_entities":
                return rag.amerge_entities([legacy], "Tesla Group")
            return rag.aedit_entity(
                legacy,
                {"entity_name": "Tesla Group"},
                allow_rename=True,
                allow_merge=True,
            )

        if first_attempt == "fails-after-node-removal":

            async def _fail_tracking_cleanup(ids):
                raise RuntimeError("injected tracking failure")

            with monkeypatch.context() as patch:
                patch.setattr(rag.entity_chunks, "delete", _fail_tracking_cleanup)
                with pytest.raises(Exception, match="tracking"):
                    await _call()
        else:
            await _call()
        assert not await graph.has_node(legacy)

        with pytest.raises(ValueError, match="it is stored as 'Tesla\\(US\\)'"):
            await _call()

        assert await graph.has_node("Tesla(US)")
        assert await graph.has_edge("Tesla(US)", "SpaceX")
        assert not await graph.has_edge("Tesla Group", "SpaceX")

    @pytest.mark.asyncio
    async def test_rest_merge_and_rename_name_the_stored_spelling(self, rag, client):
        await _create_entity(client, "Tesla（US）")
        await _create_entity(client, "Tesla Group")

        merged = await client.post(
            "/graph/entities/merge",
            json={
                "entities_to_change": ["Tesla（US）"],
                "entity_to_change_into": "Tesla Group",
            },
            headers=_HEADERS,
        )
        assert merged.status_code == 400
        assert "it is stored as 'Tesla(US)'" in merged.json()["detail"]

        renamed = await client.post(
            "/graph/entity/edit",
            json={
                "entity_name": "Tesla（US）",
                "updated_data": {"entity_name": "Tesla Motors"},
                "allow_rename": True,
            },
            headers=_HEADERS,
        )
        assert renamed.status_code == 400
        assert "it is stored as 'Tesla(US)'" in renamed.json()["detail"]

        edited = await client.post(
            "/graph/entity/edit",
            json={"entity_name": "Tesla（US）", "updated_data": {"description": "new"}},
            headers=_HEADERS,
        )
        assert edited.status_code == 200, edited.text
        graph = rag.chunk_entity_relation_graph
        assert (await graph.get_node("Tesla(US)"))["description"] == "new"
        assert await graph.has_node("Tesla Group")
