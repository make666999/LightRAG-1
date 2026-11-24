"""``ainsert_custom_kg`` provenance merge, file-path validation, and self-loop guards.

Consolidates:
- provenance merge into existing objects (so deleting a document cannot orphan
  or drop curated custom-KG entities/relations) and tracking-before-graph commit;
- pre-write validation rejecting unstorable ``file_path`` characters and preserving
  one-shot chunk iterables;
- pre-write rejection of self-loop relationships (raw and post-normalization).
"""

from __future__ import annotations

import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from lightrag import LightRAG
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import compute_mdhash_id, make_relation_chunk_key

from ._fakes import build_initialized_rag, noop_llm

pytestmark = pytest.mark.offline

DOC_ID = "doc-v3"
DOC_CHUNK = "doc-v3-chunk-000"
CURATED_TEXT = "Curated note about Foo."
CURATED_CHUNK = compute_mdhash_id(CURATED_TEXT, prefix="chunk-")


# ---------------------------------------------------------------------------
# Provenance merge & tracking-before-graph commit
# ---------------------------------------------------------------------------


async def _llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    if "V3ONLY paragraph" not in prompt:
        raise AssertionError(f"unexpected LLM prompt: {prompt[:200]!r}")
    return (
        "entity<|#|>Foo<|#|>concept<|#|>Foo fact from v3\n"
        "entity<|#|>Bar<|#|>concept<|#|>Bar fact from v3\n"
        "relation<|#|>Foo<|#|>Bar<|#|>linked<|#|>Foo links Bar in v3\n"
        "<|COMPLETE|>"
    )


@pytest.fixture
async def rag(tmp_path):
    instance = await build_initialized_rag(
        tmp_path,
        subdir=None,
        workspace_prefix="custom_kg_merge",
        llm_model_func=_llm,
        tokenizer_name="characters",
        entity_extract_max_gleaning=0,
        max_parallel_insert=1,
    )
    try:
        yield instance
    finally:
        await instance.finalize_storages()


async def _insert_doc(rag: LightRAG) -> None:
    await rag.ainsert("V3ONLY paragraph.", ids=[DOC_ID], file_paths=["doc-v3.txt"])
    node = await rag.chunk_entity_relation_graph.get_node("Foo")
    assert node["source_id"] == DOC_CHUNK


def _custom_kg(*, entities=True, relationships=True, weight=1.0) -> dict:
    return {
        "chunks": [
            {
                "content": CURATED_TEXT,
                "source_id": "curated",
                "file_path": "curated.md",
            }
        ],
        "entities": [
            {
                "entity_name": "Foo",
                "entity_type": "concept",
                "description": "Curated Foo",
                "source_id": "curated",
                "file_path": "curated.md",
            }
        ]
        if entities
        else [],
        "relationships": [
            {
                "src_id": "Foo",
                "tgt_id": "Bar",
                "description": "Curated link",
                "keywords": "curated",
                "source_id": "curated",
                "file_path": "curated.md",
                "weight": weight,
            }
        ]
        if relationships
        else [],
    }


def _ids(value: str) -> list[str]:
    return value.split(GRAPH_FIELD_SEP)


async def _tracking(rag: LightRAG) -> tuple[list[str], list[str], list[str]]:
    foo, bar = await rag.entity_chunks.get_by_ids(["Foo", "Bar"])
    (rel,) = await rag.relation_chunks.get_by_ids(
        [make_relation_chunk_key("Foo", "Bar")]
    )
    return foo["chunk_ids"], bar["chunk_ids"], rel["chunk_ids"]


@pytest.mark.asyncio
async def test_issue_reproduction_curated_objects_survive_document_deletion(rag):
    await _insert_doc(rag)
    await rag.ainsert_custom_kg(_custom_kg())

    graph = rag.chunk_entity_relation_graph
    foo = await graph.get_node("Foo")
    edge = await graph.get_edge("Foo", "Bar")
    assert foo["description"] == "Curated Foo"
    assert _ids(foo["source_id"]) == [DOC_CHUNK, CURATED_CHUNK]
    assert _ids(foo["file_path"]) == ["doc-v3.txt", "curated.md"]
    assert edge["description"] == "Curated link"
    assert _ids(edge["source_id"]) == [DOC_CHUNK, CURATED_CHUNK]
    assert _ids((await graph.get_node("Bar"))["source_id"]) == [
        DOC_CHUNK,
        CURATED_CHUNK,
    ]
    assert await _tracking(rag) == ([DOC_CHUNK, CURATED_CHUNK],) * 3

    result = await rag.adelete_by_doc_id(DOC_ID)
    assert result.status == "success", result.message

    foo = await graph.get_node("Foo")
    edge = await graph.get_edge("Foo", "Bar")
    assert foo is not None and edge is not None
    assert foo["description"] == "Curated Foo"
    assert foo["source_id"] == CURATED_CHUNK
    assert edge["description"] == "Curated link"
    assert edge["source_id"] == CURATED_CHUNK
    assert await graph.get_node("Bar") is not None
    assert await _tracking(rag) == ([CURATED_CHUNK],) * 3


@pytest.mark.asyncio
async def test_endpoint_entity_vector_carries_merged_source(rag):
    await _insert_doc(rag)
    await rag.ainsert_custom_kg(_custom_kg(entities=False))

    (bar_vector,) = await rag.entities_vdb.get_by_ids(
        [compute_mdhash_id("Bar", prefix="ent-")]
    )
    assert _ids(bar_vector["source_id"]) == [DOC_CHUNK, CURATED_CHUNK]
    bar = await rag.chunk_entity_relation_graph.get_node("Bar")
    assert bar["description"] == "Bar fact from v3"


@pytest.mark.asyncio
async def test_empty_tracking_row_is_not_reseeded_from_stale_source_id(rag):
    await _insert_doc(rag)
    await rag.entity_chunks.upsert({"Foo": {"chunk_ids": [], "count": 0}})
    await rag.ainsert_custom_kg(_custom_kg(relationships=False))

    (foo_row,) = await rag.entity_chunks.get_by_ids(["Foo"])
    assert foo_row["chunk_ids"] == [CURATED_CHUNK]
    foo = await rag.chunk_entity_relation_graph.get_node("Foo")
    assert foo["source_id"] == CURATED_CHUNK


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stored_weight", "curated_weight", "expected"),
    [
        (1.0, 1.0, 2.0),  # raised to the merged evidence count
        (5.0, 1.0, 5.0),  # an accumulated weight is not lowered
        (1.0, 7.5, 7.5),  # a curated boost still applies
    ],
)
async def test_merged_relation_weight(rag, stored_weight, curated_weight, expected):
    await _insert_doc(rag)
    graph = rag.chunk_entity_relation_graph
    edge = await graph.get_edge("Foo", "Bar")
    await graph.upsert_edge("Foo", "Bar", {**edge, "weight": stored_weight})

    await rag.ainsert_custom_kg(_custom_kg(weight=curated_weight))

    assert (await graph.get_edge("Foo", "Bar"))["weight"] == expected


@pytest.mark.asyncio
async def test_new_objects_get_tracking_rows(rag):
    await rag.ainsert_custom_kg(_custom_kg())

    graph = rag.chunk_entity_relation_graph
    foo = await graph.get_node("Foo")
    assert foo["source_id"] == CURATED_CHUNK
    assert foo["file_path"] == "curated.md"
    bar = await graph.get_node("Bar")
    assert (bar["description"], bar["source_id"]) == ("UNKNOWN", CURATED_CHUNK)
    assert await _tracking(rag) == ([CURATED_CHUNK],) * 3


@pytest.mark.asyncio
async def test_unmapped_source_writes_no_tracking_row(rag):
    await rag.ainsert_custom_kg(
        {
            "chunks": [],
            "entities": [
                {
                    "entity_name": "Foo",
                    "entity_type": "concept",
                    "description": "Curated Foo",
                    "source_id": "no-such-alias",
                }
            ],
            "relationships": [],
        }
    )

    foo = await rag.chunk_entity_relation_graph.get_node("Foo")
    assert foo["source_id"] == "UNKNOWN"
    assert await rag.entity_chunks.get_by_ids(["Foo"]) == [None]


@pytest.mark.asyncio
async def test_edges_are_read_in_one_batch_call(rag, monkeypatch):
    """All pairs go through a single get_edges_batch round trip."""
    await _insert_doc(rag)
    graph = rag.chunk_entity_relation_graph
    requested: list[list[dict]] = []
    original = graph.get_edges_batch

    async def _recording(pairs):
        requested.append(list(pairs))
        return await original(pairs)

    monkeypatch.setattr(graph, "get_edges_batch", _recording)
    custom_kg = _custom_kg()
    custom_kg["relationships"].append(
        {
            "src_id": "Foo",
            "tgt_id": "Newcomer",
            "description": "Curated newcomer",
            "keywords": "curated",
            "source_id": "curated",
        }
    )
    await rag.ainsert_custom_kg(custom_kg)

    assert requested == [
        [{"src": "Bar", "tgt": "Foo"}, {"src": "Foo", "tgt": "Newcomer"}]
    ]
    assert (await graph.get_edge("Foo", "Newcomer"))["source_id"] == CURATED_CHUNK


@pytest.mark.asyncio
async def test_edge_with_a_missing_endpoint_node_is_still_merged(rag, monkeypatch):
    """Endpoint absence is not proof of edge absence: merge weight and source_id."""
    await _insert_doc(rag)
    graph = rag.chunk_entity_relation_graph
    edge = await graph.get_edge("Foo", "Bar")
    await graph.upsert_edge("Foo", "Bar", {**edge, "weight": 10.0})
    original = graph.get_nodes_batch

    async def _bar_has_no_node(node_ids):
        nodes = await original(node_ids)
        nodes.pop("Bar", None)
        return nodes

    monkeypatch.setattr(graph, "get_nodes_batch", _bar_has_no_node)
    await rag.ainsert_custom_kg(_custom_kg(entities=False, weight=1.0))

    edge = await graph.get_edge("Foo", "Bar")
    assert edge["weight"] == 10.0
    assert _ids(edge["source_id"]) == [DOC_CHUNK, CURATED_CHUNK]


@pytest.mark.asyncio
async def test_tracking_rows_are_committed_before_the_graph_is_written(
    rag, monkeypatch
):
    """Tracking rows commit before the first graph mutation."""
    await _insert_doc(rag)
    events: list[str] = []
    graph = rag.chunk_entity_relation_graph

    def _record(name, original):
        async def wrapper(*args, **kwargs):
            events.append(name)
            return await original(*args, **kwargs)

        return wrapper

    for storage, name in (
        (rag.entity_chunks, "entity_chunks"),
        (rag.relation_chunks, "relation_chunks"),
    ):
        monkeypatch.setattr(
            storage,
            "index_done_callback",
            _record(f"commit:{name}", storage.index_done_callback),
        )
    for method in ("upsert_nodes_batch", "upsert_edges_batch"):
        monkeypatch.setattr(graph, method, _record(method, getattr(graph, method)))

    await rag.ainsert_custom_kg(_custom_kg())

    first_graph_write = events.index("upsert_nodes_batch")
    assert {"commit:entity_chunks", "commit:relation_chunks"} <= set(
        events[:first_graph_write]
    )


# ---------------------------------------------------------------------------
# Pre-write validation: unstorable file_path & self-loop rejection
# ---------------------------------------------------------------------------


@pytest.fixture
def single_process_shared_data():
    """``ainsert_custom_kg`` reads the ``pipeline_status`` namespace."""
    finalize_share_data()
    initialize_share_data(1)
    yield
    finalize_share_data()


def _make_storage_mock():
    storage = MagicMock()
    storage.global_config = {"workspace": ""}
    storage.upsert = AsyncMock(return_value=None)
    storage.index_done_callback = AsyncMock(return_value=None)
    return storage


def _make_mock_rag() -> LightRAG:
    rag = LightRAG.__new__(LightRAG)
    rag.workspace = ""
    rag.tokenizer = MagicMock()
    rag.tokenizer.encode = lambda _content: []
    rag.chunks_vdb = _make_storage_mock()
    rag.text_chunks = _make_storage_mock()
    rag.chunk_entity_relation_graph = MagicMock()
    rag.chunk_entity_relation_graph.upsert_node = AsyncMock(return_value=None)
    rag.chunk_entity_relation_graph.upsert_edge = AsyncMock(return_value=None)
    rag.chunk_entity_relation_graph.upsert_nodes_batch = AsyncMock(return_value=None)
    rag.chunk_entity_relation_graph.upsert_edges_batch = AsyncMock(return_value=None)
    rag.entities_vdb = _make_storage_mock()
    rag.relationships_vdb = _make_storage_mock()
    rag._insert_done = AsyncMock(return_value=None)
    return rag


def _file_path_payload(bad_section: str, bad_path: str) -> dict:
    def path_for(section: str) -> str:
        return bad_path if section == bad_section else "doc.txt"

    return {
        "chunks": [
            {"content": "Alpha meets Beta.", "source_id": "c1", "file_path": "ok.txt"},
            {
                "content": "Beta meets Gamma.",
                "source_id": "c2",
                "file_path": path_for("chunks"),
            },
        ],
        "entities": [
            {"entity_name": "Alpha", "source_id": "c1", "file_path": "ok.txt"},
            {
                "entity_name": "Beta",
                "source_id": "c2",
                "file_path": path_for("entities"),
            },
        ],
        "relationships": [
            {
                "src_id": "Alpha",
                "tgt_id": "Gamma",
                "source_id": "c1",
                "file_path": "ok.txt",
            },
            {
                "src_id": "Beta",
                "tgt_id": "Gamma",
                "source_id": "c2",
                "file_path": path_for("relationships"),
            },
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("section", ["chunks", "entities", "relationships"])
@pytest.mark.parametrize(
    ("bad_char", "code_point"),
    [pytest.param("\ud800", "U+D800", id="lone-surrogate")],
)
async def test_unstorable_file_path_is_rejected_before_any_write(
    single_process_shared_data, section, bad_char, code_point
):
    rag = _make_mock_rag()

    with pytest.raises(
        ValueError,
        match=rf"Custom KG {section}\[1\]\.file_path must not contain the "
        rf"character {re.escape(code_point)}",
    ):
        await rag.ainsert_custom_kg(
            _file_path_payload(section, f"bad{bad_char}name.txt")
        )

    rag.chunks_vdb.upsert.assert_not_called()
    rag.text_chunks.upsert.assert_not_called()
    rag.chunk_entity_relation_graph.upsert_nodes_batch.assert_not_called()
    rag.chunk_entity_relation_graph.upsert_edges_batch.assert_not_called()
    rag.entities_vdb.upsert.assert_not_called()
    rag.relationships_vdb.upsert.assert_not_called()
    rag._insert_done.assert_not_called()


@pytest.mark.asyncio
async def test_rejected_chunk_path_leaves_default_storages_clean_and_writable(tmp_path):
    """Rejected batch leaves file storages clean and able to commit the next batch."""
    finalize_share_data()
    rag = await build_initialized_rag(
        tmp_path,
        subdir=None,
        workspace="",
        llm_model_func=noop_llm,
        max_token_size=512,
    )
    try:
        bad_chunk = {"content": "Poisoned chunk.", "source_id": "c1"}
        with pytest.raises(ValueError, match=r"chunks\[0\]\.file_path"):
            await rag.ainsert_custom_kg(
                {
                    "chunks": [{**bad_chunk, "file_path": "bad\udcffname.txt"}],
                    "entities": [],
                    "relationships": [],
                }
            )
        bad_id = compute_mdhash_id("Poisoned chunk.", prefix="chunk-")
        assert await rag.text_chunks.get_by_id(bad_id) is None
        assert await rag.chunks_vdb.get_by_id(bad_id) is None

        await rag.ainsert_custom_kg(
            {
                "chunks": [
                    {
                        "content": "Clean chunk.",
                        "source_id": "c2",
                        "file_path": "notes\tv2.txt",
                    }
                ],
                "entities": [
                    {
                        "entity_name": "Alpha",
                        "description": "A clean entity.",
                        "source_id": "c2",
                        "file_path": "notes\tv2.txt",
                    }
                ],
                "relationships": [],
            }
        )
        good_id = compute_mdhash_id("Clean chunk.", prefix="chunk-")
        assert (await rag.chunks_vdb.get_by_id(good_id))["file_path"] == "notes\tv2.txt"
        node = await rag.chunk_entity_relation_graph.get_node("Alpha")
        assert node["file_path"] == "notes\tv2.txt"
    finally:
        await rag.finalize_storages()
        finalize_share_data()


@pytest.mark.asyncio
async def test_one_shot_chunk_iterable_is_still_stored(tmp_path):
    """Validation must not exhaust a one-shot generator before storage reads it."""
    finalize_share_data()
    rag = await build_initialized_rag(
        tmp_path,
        subdir=None,
        workspace="",
        llm_model_func=noop_llm,
        max_token_size=512,
    )
    try:
        contents = ["First chunk.", "Second chunk."]
        await rag.ainsert_custom_kg(
            {
                "chunks": (
                    {"content": c, "source_id": f"c{i}", "file_path": "a.txt"}
                    for i, c in enumerate(contents)
                ),
                "entities": [
                    {
                        "entity_name": "Alpha",
                        "description": "Linked to the second chunk.",
                        "source_id": "c1",
                    }
                ],
                "relationships": [],
            }
        )
        ids = [compute_mdhash_id(c, prefix="chunk-") for c in contents]
        assert [await rag.text_chunks.get_by_id(i) is not None for i in ids] == [
            True,
            True,
        ]
        node = await rag.chunk_entity_relation_graph.get_node("Alpha")
        assert node["source_id"] == ids[1]
    finally:
        await rag.finalize_storages()
        finalize_share_data()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "relationships, match",
    [
        pytest.param(
            [
                {
                    "src_id": "A",
                    "tgt_id": "A",
                    "description": "A refers to itself",
                    "keywords": "self",
                }
            ],
            r"relationships\[0\] is a self-loop on 'A'",
            id="exact-self-loop",
        ),
        pytest.param(
            [
                {
                    "src_id": '"A"',
                    "tgt_id": "A",
                    "description": "A refers to itself",
                    "keywords": "self",
                }
            ],
            r"relationships\[0\] is a self-loop",
            id="normalized-self-loop",
        ),
        pytest.param(
            [
                {
                    "src_id": "A",
                    "tgt_id": "B",
                    "description": "A relates to B",
                    "keywords": "k",
                },
                {
                    "src_id": "B",
                    "tgt_id": "B",
                    "description": "B refers to itself",
                    "keywords": "self",
                },
            ],
            r"relationships\[1\] is a self-loop on 'B'",
            id="reports-offending-index",
        ),
    ],
)
async def test_ainsert_custom_kg_rejects_self_loop_relationships(
    single_process_shared_data, relationships, match
):
    """Self-loops (raw or post-normalization) fail validation before any write."""
    rag = _make_mock_rag()

    with pytest.raises(ValueError, match=match):
        await rag.ainsert_custom_kg(
            {"chunks": [], "entities": [], "relationships": relationships}
        )

    rag.chunk_entity_relation_graph.upsert_edge.assert_not_called()
    rag.chunk_entity_relation_graph.upsert_node.assert_not_called()
    rag.relationships_vdb.upsert.assert_not_called()
