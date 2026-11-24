"""Attribute and description validation for manual entity/relation mutation APIs."""

from __future__ import annotations

from copy import deepcopy

import pytest

from lightrag import utils_graph
from lightrag.constants import GRAPH_FIELD_SEP, SOURCE_IDS_LIMIT_METHOD_KEEP
from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.operate import (
    _handle_single_entity_extraction,
    _handle_single_relationship_extraction,
    _merge_nodes_then_upsert,
    _normalize_text_extraction_record_attributes,
)
from lightrag.utils import VectorStorageConsistencyError, make_relation_chunk_key

pytestmark = pytest.mark.offline

NESTED = {"any": ["object"]}


class _NoopLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _KVStorage:
    def __init__(self):
        self.records: dict = {}

    async def get_by_id(self, key):
        return self.records.get(key)

    async def upsert(self, data):
        self.records.update(deepcopy(data))

    async def delete(self, ids):
        for key in ids:
            self.records.pop(key, None)

    async def index_done_callback(self):
        return None


class _VectorStorage(_KVStorage):
    def __init__(self, global_config):
        super().__init__()
        self.global_config = global_config


class _Fixture:
    """A real NetworkXStorage plus in-memory VDB/KV doubles."""

    def __init__(self, tmp_path):
        self.global_config = {
            "working_dir": str(tmp_path),
            "workspace": "",
            "embedding_batch_num": 10,
            "max_total_tokens": 30000,
        }
        self.graph = NetworkXStorage(
            namespace="chunk_entity_relation",
            workspace="",
            global_config=self.global_config,
            embedding_func=None,
        )
        self.entities_vdb = _VectorStorage(self.global_config)
        self.relationships_vdb = _VectorStorage(self.global_config)
        self.entity_chunks = _KVStorage()
        self.relation_chunks = _KVStorage()

    async def start(self):
        await self.graph.initialize()
        return self

    async def create_entity(self, name, **overrides):
        data = {"description": "d", "entity_type": "t", "source_id": "chunk-1"}
        data.update(overrides)
        return await utils_graph.acreate_entity(
            self.graph,
            self.entities_vdb,
            self.relationships_vdb,
            name,
            data,
            entity_chunks_storage=self.entity_chunks,
            relation_chunks_storage=self.relation_chunks,
        )

    async def edit_entity(self, name, updated_data):
        return await utils_graph.aedit_entity(
            self.graph,
            self.entities_vdb,
            self.relationships_vdb,
            name,
            updated_data,
            entity_chunks_storage=self.entity_chunks,
            relation_chunks_storage=self.relation_chunks,
        )

    async def create_relation(self, source, target, **overrides):
        data = {
            "description": "d",
            "keywords": "k",
            "source_id": "chunk-1",
            "weight": 1.0,
        }
        data.update(overrides)
        return await utils_graph.acreate_relation(
            self.graph,
            self.entities_vdb,
            self.relationships_vdb,
            source,
            target,
            data,
            relation_chunks_storage=self.relation_chunks,
        )

    async def edit_relation(self, source, target, updated_data):
        return await utils_graph.aedit_relation(
            self.graph,
            self.entities_vdb,
            self.relationships_vdb,
            source,
            target,
            updated_data,
            relation_chunks_storage=self.relation_chunks,
        )

    def graphml_text(self):
        with open(self.graph._graphml_xml_file, encoding="utf-8") as handle:
            return handle.read()


@pytest.fixture
async def rag(tmp_path, monkeypatch):
    finalize_share_data()
    initialize_share_data()
    monkeypatch.setattr(
        utils_graph, "get_storage_keyed_lock", lambda *a, **k: _NoopLock()
    )
    try:
        yield await _Fixture(tmp_path).start()
    finally:
        finalize_share_data()


# --------------------------------------------------------------------------- #
# Fix proofs: rejected attribute payloads leave NetworkX write path healthy
# --------------------------------------------------------------------------- #


class TestEntityEditWritePathStaysHealthy:
    @pytest.mark.asyncio
    async def test_unknown_key_with_non_scalar_value_is_refused(self, rag):
        await rag.create_entity("VICTIM")

        with pytest.raises(ValueError, match="Unknown entity field 'poc_nested'"):
            await rag.edit_entity("VICTIM", {"poc_nested": NESTED})

        assert "poc_nested" not in await rag.graph.get_node("VICTIM")
        await rag.create_entity("AFTER")
        assert "AFTER" in rag.graphml_text()

    @pytest.mark.asyncio
    async def test_null_on_an_allowed_field_is_refused(self, rag):
        await rag.create_entity("VICTIM")

        with pytest.raises(ValueError, match="'file_path' must be a string"):
            await rag.edit_entity("VICTIM", {"file_path": None})

        await rag.create_entity("AFTER")
        assert "AFTER" in rag.graphml_text()

    @pytest.mark.asyncio
    async def test_control_character_on_an_allowed_field_is_refused(self, rag):
        await rag.create_entity("VICTIM")

        with pytest.raises(ValueError, match=r"must not contain the character U\+000B"):
            await rag.edit_entity("VICTIM", {"file_path": "a\x0bb"})

        await rag.create_entity("AFTER")
        assert "AFTER" in rag.graphml_text()

    @pytest.mark.asyncio
    async def test_wrong_scalar_type_on_source_id_is_refused(self, rag):
        await rag.create_entity("VICTIM")

        with pytest.raises(ValueError, match="'source_id' must be a string, got int"):
            await rag.edit_entity("VICTIM", {"source_id": 123})

        assert (await rag.graph.get_node("VICTIM"))["source_id"] == "chunk-1"
        await rag.edit_entity("VICTIM", {"description": "an ordinary update"})
        assert (await rag.graph.get_node("VICTIM"))["description"] == (
            "an ordinary update"
        )


class TestRelationEditWritePathStaysHealthy:
    @pytest.mark.asyncio
    async def test_unknown_key_with_non_scalar_value_is_refused(self, rag):
        await rag.create_entity("REL_A")
        await rag.create_entity("REL_B")
        await rag.create_relation("REL_A", "REL_B")

        with pytest.raises(ValueError, match="Unknown relation field 'poc_nested'"):
            await rag.edit_relation("REL_A", "REL_B", {"poc_nested": NESTED})

        assert "poc_nested" not in await rag.graph.get_edge("REL_A", "REL_B")
        await rag.create_entity("AFTER")
        assert "AFTER" in rag.graphml_text()


class TestRelationWeightEvidenceFloor:
    @staticmethod
    async def _create_endpoints(rag):
        await rag.create_entity("REL_A")
        await rag.create_entity("REL_B")

    @pytest.mark.asyncio
    async def test_create_rejects_weight_below_distinct_source_count(self, rag):
        await self._create_endpoints(rag)

        with pytest.raises(ValueError, match="distinct-source evidence count 2"):
            await rag.create_relation(
                "REL_A",
                "REL_B",
                source_id=GRAPH_FIELD_SEP.join(["chunk-1", "chunk-2"]),
                weight=1.5,
            )

        assert not await rag.graph.has_edge("REL_A", "REL_B")

    @pytest.mark.asyncio
    async def test_source_less_relation_allows_fractional_weight(self, rag):
        await self._create_endpoints(rag)

        await rag.create_relation("REL_A", "REL_B", source_id="", weight=0.25)

        edge = await rag.graph.get_edge("REL_A", "REL_B")
        assert edge["source_id"] == ""
        assert edge["weight"] == 0.25
        assert rag.relation_chunks.records == {
            make_relation_chunk_key("REL_A", "REL_B"): {"chunk_ids": [], "count": 0}
        }

    @pytest.mark.asyncio
    async def test_create_rejects_negative_source_less_weight(self, rag):
        await self._create_endpoints(rag)

        with pytest.raises(ValueError, match="distinct-source evidence count 0"):
            await rag.create_relation("REL_A", "REL_B", source_id="", weight=-0.25)

        assert not await rag.graph.has_edge("REL_A", "REL_B")

    @pytest.mark.asyncio
    async def test_distinct_real_sources_define_floor_and_chunk_tracking(self, rag):
        await self._create_endpoints(rag)
        source_id = GRAPH_FIELD_SEP.join(
            ["chunk-1", "chunk-1", "manual_creation", "UNKNOWN"]
        )

        await rag.create_relation("REL_A", "REL_B", source_id=source_id, weight=1.0)

        edge = await rag.graph.get_edge("REL_A", "REL_B")
        assert edge["source_id"] == source_id
        tracking = next(iter(rag.relation_chunks.records.values()))
        assert tracking == {"chunk_ids": ["chunk-1"], "count": 1}

    @pytest.mark.asyncio
    async def test_edit_validates_source_and_weight_together(self, rag):
        await self._create_endpoints(rag)
        await rag.create_relation("REL_A", "REL_B", source_id="", weight=0.25)

        with pytest.raises(ValueError, match="distinct-source evidence count 1"):
            await rag.edit_relation("REL_A", "REL_B", {"source_id": "chunk-1"})

        unchanged = await rag.graph.get_edge("REL_A", "REL_B")
        assert unchanged["source_id"] == ""
        assert unchanged["weight"] == 0.25

        await rag.edit_relation(
            "REL_A",
            "REL_B",
            {"source_id": "chunk-1", "weight": 1.0},
        )
        updated = await rag.graph.get_edge("REL_A", "REL_B")
        assert updated["source_id"] == "chunk-1"
        assert updated["weight"] == 1.0

    @pytest.mark.asyncio
    async def test_edit_can_remove_sources_before_lowering_weight(self, rag):
        await self._create_endpoints(rag)
        await rag.create_relation("REL_A", "REL_B", weight=1.0)

        await rag.edit_relation(
            "REL_A",
            "REL_B",
            {"source_id": "", "weight": 0.25},
        )

        edge = await rag.graph.get_edge("REL_A", "REL_B")
        assert edge["source_id"] == ""
        assert edge["weight"] == 0.25

    @pytest.mark.asyncio
    @pytest.mark.parametrize("legacy_source_id", ["manual_creation", "UNKNOWN"])
    async def test_edit_clears_legacy_tracking_when_source_is_removed(
        self, rag, legacy_source_id
    ):
        await self._create_endpoints(rag)
        await rag.create_relation(
            "REL_A", "REL_B", source_id=legacy_source_id, weight=0.25
        )
        storage_key = make_relation_chunk_key("REL_A", "REL_B")
        await rag.relation_chunks.upsert(
            {storage_key: {"chunk_ids": [legacy_source_id], "count": 1}}
        )

        await rag.edit_relation(
            "REL_A",
            "REL_B",
            {"source_id": "", "weight": 0.25},
        )

        edge = await rag.graph.get_edge("REL_A", "REL_B")
        assert edge["source_id"] == ""
        assert rag.relation_chunks.records[storage_key] == {
            "chunk_ids": [],
            "count": 0,
        }

    @pytest.mark.asyncio
    async def test_unrelated_edit_cleans_legacy_tracking_placeholder(self, rag):
        await self._create_endpoints(rag)
        await rag.create_relation(
            "REL_A", "REL_B", source_id="manual_creation", weight=0.25
        )
        storage_key = make_relation_chunk_key("REL_A", "REL_B")
        await rag.relation_chunks.upsert(
            {storage_key: {"chunk_ids": ["manual_creation"], "count": 1}}
        )

        await rag.edit_relation(
            "REL_A", "REL_B", {"description": "updated description"}
        )

        edge = await rag.graph.get_edge("REL_A", "REL_B")
        assert edge["source_id"] == "manual_creation"
        assert edge["weight"] == 0.25
        assert rag.relation_chunks.records[storage_key] == {
            "chunk_ids": [],
            "count": 0,
        }

    @pytest.mark.asyncio
    async def test_unrelated_edit_preserves_authoritative_empty_tracking(self, rag):
        await self._create_endpoints(rag)
        await rag.create_relation("REL_A", "REL_B", source_id="chunk-1", weight=1.0)
        storage_key = make_relation_chunk_key("REL_A", "REL_B")
        await rag.relation_chunks.upsert({storage_key: {"chunk_ids": [], "count": 0}})

        await rag.edit_relation(
            "REL_A", "REL_B", {"description": "updated description"}
        )

        assert rag.relation_chunks.records[storage_key] == {
            "chunk_ids": [],
            "count": 0,
        }

    @pytest.mark.asyncio
    async def test_unrelated_edit_repairs_legacy_weight_below_floor(self, rag):
        await self._create_endpoints(rag)
        await rag.create_relation("REL_A", "REL_B", weight=1.0)
        legacy = await rag.graph.get_edge("REL_A", "REL_B")
        legacy["weight"] = 0.25
        await rag.graph.upsert_edge("REL_A", "REL_B", legacy)

        await rag.edit_relation(
            "REL_A", "REL_B", {"description": "updated description"}
        )

        repaired = await rag.graph.get_edge("REL_A", "REL_B")
        assert repaired["weight"] == 1.0

    @pytest.mark.asyncio
    async def test_entity_rename_repairs_legacy_relation_weight(self, rag):
        await self._create_endpoints(rag)
        await rag.create_relation("REL_A", "REL_B", weight=1.0)
        legacy = await rag.graph.get_edge("REL_A", "REL_B")
        legacy["weight"] = 0.25
        await rag.graph.upsert_edge("REL_A", "REL_B", legacy)

        await rag.edit_entity("REL_A", {"entity_name": "RENAMED_A"})

        repaired = await rag.graph.get_edge("RENAMED_A", "REL_B")
        assert repaired["weight"] == 1.0


class TestEntityCreateWritePathStaysHealthy:
    @pytest.mark.asyncio
    async def test_non_scalar_on_a_known_field_is_refused(self, rag):
        await rag.create_entity("CTRL")

        with pytest.raises(ValueError, match="'entity_type' must be a string"):
            await rag.create_entity("ATTACK", entity_type=NESTED)

        assert not await rag.graph.has_node("ATTACK")
        await rag.create_entity("AFTER")
        assert "AFTER" in rag.graphml_text()

    @pytest.mark.asyncio
    async def test_unknown_key_is_still_ignored_not_rejected(self, rag):
        await rag.create_entity("CREATE_EXTRA", poc_nested=NESTED)

        node = await rag.graph.get_node("CREATE_EXTRA")
        assert "poc_nested" not in node
        assert "CREATE_EXTRA" in rag.graphml_text()


class TestAllowedFieldsAndTypes:
    @pytest.mark.asyncio
    async def test_multiline_description_is_accepted(self, rag):
        await rag.create_entity("VICTIM")
        text = "first line\nsecond\tline\rthird"

        await rag.edit_entity("VICTIM", {"description": text})

        assert (await rag.graph.get_node("VICTIM"))["description"] == text
        assert "VICTIM" in rag.graphml_text()

    @pytest.mark.asyncio
    async def test_rename_field_is_still_allowed(self, rag):
        await rag.create_entity("OLD_NAME")

        await rag.edit_entity("OLD_NAME", {"entity_name": "NEW_NAME"})

        assert await rag.graph.has_node("NEW_NAME")
        assert not await rag.graph.has_node("OLD_NAME")

    @pytest.mark.asyncio
    async def test_entity_id_is_not_an_editable_field(self, rag):
        await rag.create_entity("VICTIM")

        with pytest.raises(ValueError, match="Unknown entity field 'entity_id'"):
            await rag.edit_entity("VICTIM", {"entity_id": "SOMETHING_ELSE"})

    @pytest.mark.asyncio
    async def test_error_message_lists_the_allowed_fields(self, rag):
        await rag.create_entity("VICTIM")

        with pytest.raises(ValueError) as excinfo:
            await rag.edit_entity("VICTIM", {"nope": "x"})

        message = str(excinfo.value)
        for field in ("entity_name", "entity_type", "description", "source_id"):
            assert field in message

    @pytest.mark.asyncio
    @pytest.mark.parametrize("weight, expected", [(2, 2.0), ("2.5", 2.5)])
    async def test_relation_weight_is_normalized_to_float(self, rag, weight, expected):
        await rag.create_entity("REL_A")
        await rag.create_entity("REL_B")
        await rag.create_relation("REL_A", "REL_B")

        await rag.edit_relation("REL_A", "REL_B", {"weight": weight})

        stored = (await rag.graph.get_edge("REL_A", "REL_B"))["weight"]
        assert isinstance(stored, float)
        assert stored == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize("weight", ["nan", "1e999"])
    async def test_relation_weight_rejects_strings_that_coerce_to_non_finite(
        self, rag, weight
    ):
        await rag.create_entity("REL_A")
        await rag.create_entity("REL_B")
        await rag.create_relation("REL_A", "REL_B")

        with pytest.raises(ValueError, match="'weight' must be a finite number"):
            await rag.edit_relation("REL_A", "REL_B", {"weight": weight})

        assert (await rag.graph.get_edge("REL_A", "REL_B"))["weight"] == 1.0
        await rag.create_entity("AFTER")
        assert "AFTER" in rag.graphml_text()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("weight", [2**63, -(10**400)])
    async def test_relation_weight_rejects_oversized_integers(self, rag, weight):
        await rag.create_entity("REL_A")
        await rag.create_entity("REL_B")
        await rag.create_relation("REL_A", "REL_B")

        with pytest.raises(ValueError, match="must be a 64-bit integer"):
            await rag.edit_relation("REL_A", "REL_B", {"weight": weight})

        assert (await rag.graph.get_edge("REL_A", "REL_B"))["weight"] == 1.0
        await rag.create_entity("AFTER")
        assert "AFTER" in rag.graphml_text()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("weight", [True, "abc", None])
    async def test_relation_weight_rejects_non_numbers(self, rag, weight):
        await rag.create_entity("REL_A")
        await rag.create_entity("REL_B")
        await rag.create_relation("REL_A", "REL_B")

        with pytest.raises(ValueError, match="'weight'"):
            await rag.edit_relation("REL_A", "REL_B", {"weight": weight})

        await rag.create_entity("AFTER")
        assert "AFTER" in rag.graphml_text()

    @pytest.mark.asyncio
    async def test_merge_target_entity_data_is_validated(self, rag):
        await rag.create_entity("SOURCE")
        await rag.create_entity("TARGET")

        with pytest.raises(ValueError, match="Unknown entity field 'poc_nested'"):
            await utils_graph.amerge_entities(
                rag.graph,
                rag.entities_vdb,
                rag.relationships_vdb,
                ["SOURCE"],
                "TARGET",
                target_entity_data={"poc_nested": NESTED},
                entity_chunks_storage=rag.entity_chunks,
                relation_chunks_storage=rag.relation_chunks,
            )

        assert await rag.graph.has_node("SOURCE")
        await rag.create_entity("AFTER")
        assert "AFTER" in rag.graphml_text()


# --------------------------------------------------------------------------- #
# Description validation, record normalization, and merge consistency
# --------------------------------------------------------------------------- #


class _DummyGraphStorage:
    def __init__(self, node=None):
        self.node = node
        self.upserted_nodes = []

    async def get_node(self, node_id):
        return self.node

    async def upsert_node(self, node_id, node_data):
        self.upserted_nodes.append((node_id, node_data))
        self.node = dict(node_data)


class _DummyVectorStorage:
    def __init__(self):
        self.global_config = {"workspace": "test"}
        self.upserts = []
        self.deletes = []

    async def upsert(self, data):
        self.upserts.append(data)
        return None

    async def delete(self, ids):
        self.deletes.append(ids)
        return None

    async def get_by_id(self, id_):
        return None

    async def index_done_callback(self):
        return True


class _DummyMergeGraphStorage:
    def __init__(self):
        self.nodes = {
            "Canonical": {
                "entity_id": "Canonical",
                "description": "canonical desc",
                "entity_type": "ORG",
                "source_id": "chunk-1",
                "file_path": "canonical.md",
            },
            "Alias": {
                "entity_id": "Alias",
                "description": "alias desc",
                "entity_type": "ORG",
                "source_id": "chunk-2",
                "file_path": "alias.md",
            },
            "Neighbor": {
                "entity_id": "Neighbor",
                "description": "neighbor desc",
                "entity_type": "ORG",
                "source_id": "chunk-3",
                "file_path": "neighbor.md",
            },
        }
        self.edges = {
            ("Alias", "Neighbor"): {
                "description": "rel desc",
                "keywords": "alias",
                "source_id": "chunk-rel",
                "weight": 1.0,
                "file_path": "rel.md",
            }
        }

    async def has_node(self, node_id):
        return node_id in self.nodes

    async def get_node(self, node_id):
        return self.nodes[node_id]

    async def upsert_node(self, node_id, node_data):
        self.nodes[node_id] = dict(node_data)

    async def get_node_edges(self, node_id):
        return [
            (src, tgt)
            for src, tgt in self.edges
            if src == node_id or tgt == node_id
        ]

    async def get_edge(self, src, tgt):
        return self.edges.get((src, tgt)) or self.edges.get((tgt, src))

    async def upsert_edge(self, src, tgt, edge_data):
        self.edges[(src, tgt)] = dict(edge_data)

    async def delete_node(self, node_id):
        self.nodes.pop(node_id, None)
        self.edges = {
            (src, tgt): data
            for (src, tgt), data in self.edges.items()
            if src != node_id and tgt != node_id
        }

    async def index_done_callback(self):
        return True


@pytest.mark.asyncio
async def test_merge_nodes_then_upsert_handles_missing_legacy_description():
    graph = _DummyGraphStorage(node={"source_id": "chunk-1"})
    global_config = {
        "source_ids_limit_method": SOURCE_IDS_LIMIT_METHOD_KEEP,
        "max_source_ids_per_entity": 20,
    }

    result = await _merge_nodes_then_upsert(
        entity_name="LegacyEntity",
        nodes_data=[],
        knowledge_graph_inst=graph,
        entity_vdb=None,
        global_config=global_config,
    )

    assert result["description"] == "Entity LegacyEntity"
    assert graph.upserted_nodes[-1][1]["description"] == "Entity LegacyEntity"


@pytest.mark.asyncio
async def test_acreate_entity_rejects_empty_description():
    with pytest.raises(ValueError, match="description cannot be empty"):
        await utils_graph.acreate_entity(
            chunk_entity_relation_graph=None,
            entities_vdb=None,
            relationships_vdb=None,
            entity_name="EntityA",
            entity_data={"description": "   "},
        )


@pytest.mark.asyncio
async def test_acreate_relation_rejects_empty_description():
    with pytest.raises(ValueError, match="description cannot be empty"):
        await utils_graph.acreate_relation(
            chunk_entity_relation_graph=None,
            entities_vdb=None,
            relationships_vdb=None,
            source_entity="A",
            target_entity="B",
            relation_data={"description": ""},
        )


@pytest.mark.asyncio
async def test_aedit_entity_rejects_empty_description():
    with pytest.raises(ValueError, match="description cannot be empty"):
        await utils_graph.aedit_entity(
            chunk_entity_relation_graph=None,
            entities_vdb=None,
            relationships_vdb=None,
            entity_name="EntityA",
            updated_data={"description": None},
        )


@pytest.mark.asyncio
async def test_aedit_relation_rejects_empty_description():
    with pytest.raises(ValueError, match="description cannot be empty"):
        await utils_graph.aedit_relation(
            chunk_entity_relation_graph=None,
            entities_vdb=None,
            relationships_vdb=None,
            source_entity="A",
            target_entity="B",
            updated_data={"description": "   "},
        )


@pytest.mark.asyncio
async def test_aedit_entity_allows_updates_without_description(monkeypatch):
    async def fake_edit_impl(*args, **kwargs):
        return {"entity_name": "EntityA", "description": "kept", "source_id": "chunk-1"}

    monkeypatch.setattr(
        utils_graph, "get_storage_keyed_lock", lambda *a, **k: _NoopLock()
    )
    monkeypatch.setattr(utils_graph, "_edit_entity_impl", fake_edit_impl)

    result = await utils_graph.aedit_entity(
        chunk_entity_relation_graph=None,
        entities_vdb=_DummyVectorStorage(),
        relationships_vdb=_DummyVectorStorage(),
        entity_name="EntityA",
        updated_data={"entity_type": "ORG"},
    )

    assert result["operation_summary"]["operation_status"] == "success"


def test_four_part_entity_row_remains_entity():
    record = _normalize_text_extraction_record_attributes(
        ["entity", "Alice", "Person", "Alice is the founder of Acme Corp."],
        chunk_key="chunk-1",
    )

    entity = _handle_single_entity_extraction(
        record,
        chunk_key="chunk-1",
        timestamp=1,
    )

    assert entity is not None
    assert entity["entity_name"] == "Alice"


def test_malformed_recovered_relation_still_fails():
    record = _normalize_text_extraction_record_attributes(
        ["entity", "Alice", "Acme Corp", "founded", "   "],
        chunk_key="chunk-1",
    )

    relation = _handle_single_relationship_extraction(
        record,
        chunk_key="chunk-1",
        timestamp=1,
    )

    assert relation is None


def test_unrelated_five_part_prefix_remains_invalid():
    record = _normalize_text_extraction_record_attributes(
        ["edge", "Alice", "Acme Corp", "founded", "Alice founded Acme Corp."],
        chunk_key="chunk-1",
    )

    relation = _handle_single_relationship_extraction(
        record,
        chunk_key="chunk-1",
        timestamp=1,
    )

    assert relation is None


@pytest.mark.asyncio
async def test_merge_entities_preserves_file_path_in_vector_updates(monkeypatch):
    graph = _DummyMergeGraphStorage()
    entities_vdb = _DummyVectorStorage()
    relationships_vdb = _DummyVectorStorage()

    async def fake_get_entity_info(*args, **kwargs):
        return {"entity_name": "Canonical"}

    monkeypatch.setattr(utils_graph, "get_entity_info", fake_get_entity_info)

    await utils_graph._merge_entities_impl(
        chunk_entity_relation_graph=graph,
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        source_entities=["Alias", "Canonical"],
        target_entity="Canonical",
    )

    relationship_payload = relationships_vdb.upserts[-1]
    entity_payload = entities_vdb.upserts[-1]

    assert next(iter(relationship_payload.values()))["file_path"] == "rel.md"
    assert set(
        next(iter(entity_payload.values()))["file_path"].split(GRAPH_FIELD_SEP)
    ) == {
        "alias.md",
        "canonical.md",
    }


@pytest.mark.asyncio
async def test_merge_response_is_built_from_graph_without_reading_vdb():
    graph = _DummyMergeGraphStorage()
    entities_vdb = _DummyVectorStorage()
    relationships_vdb = _DummyVectorStorage()

    vdb_reads = []
    inner_get_by_id = entities_vdb.get_by_id

    async def _tracked_get_by_id(id_):
        vdb_reads.append(id_)
        return await inner_get_by_id(id_)

    entities_vdb.get_by_id = _tracked_get_by_id

    result = await utils_graph._merge_entities_impl(
        chunk_entity_relation_graph=graph,
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        source_entities=["Alias", "Canonical"],
        target_entity="Canonical",
    )

    assert "graph_data" in result
    assert "vector_data" not in result
    assert vdb_reads == []


@pytest.mark.asyncio
async def test_aedit_entity_merge_propagates_consistency_error(monkeypatch):
    graph = _DummyMergeGraphStorage()
    entities_vdb = _DummyVectorStorage()
    relationships_vdb = _DummyVectorStorage()

    async def _boom():
        raise RuntimeError("embedder down")

    relationships_vdb.index_done_callback = _boom

    async def _single_attempt(operation, **kwargs):
        await operation()

    monkeypatch.setattr(
        utils_graph, "safe_vdb_operation_with_exception", _single_attempt
    )
    monkeypatch.setattr(
        utils_graph, "get_storage_keyed_lock", lambda *a, **k: _NoopLock()
    )

    async def fake_get_entity_info(*args, **kwargs):
        return {"entity_name": "Canonical"}

    monkeypatch.setattr(utils_graph, "get_entity_info", fake_get_entity_info)

    with pytest.raises(VectorStorageConsistencyError) as excinfo:
        await utils_graph.aedit_entity(
            chunk_entity_relation_graph=graph,
            entities_vdb=entities_vdb,
            relationships_vdb=relationships_vdb,
            entity_name="Alias",
            updated_data={"entity_name": "Canonical"},
            allow_rename=True,
            allow_merge=True,
        )

    assert "lightrag-rebuild-vdb" in str(excinfo.value)
    assert "Alias" in graph.nodes
