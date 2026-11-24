"""Tests for ``aexport_data`` across Excel, Markdown, and CSV/VDB backends.

Covers:
- Excel export: one sheet per non-empty section, ``index=False``, and ``None``
  cells left empty rather than stringified.
- Markdown export: escaping literal pipes, backslashes, and line breaks so
  every value stays inside its cell and decodes unchanged.
- Portable VectorDB ``client_storage`` handling: backends without
  ``client_storage``, async property (NanoVectorDB), single descriptor access
  (no ``hasattr`` coroutine leak), sync property with staleness warning
  (Faiss), and omitting/including ``__vector__`` embeddings per flag.
- Undirected relations exported once rather than once per orientation.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import networkx as nx
import pytest

from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.utils import aexport_data

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Shared storage stand-ins
# ---------------------------------------------------------------------------


def _make_networkx_storage(graph: nx.Graph) -> NetworkXStorage:
    """A ``NetworkXStorage`` reading ``graph`` with no file or lock behind it."""
    storage = NetworkXStorage.__new__(NetworkXStorage)
    storage.workspace = "test"
    storage.namespace = "chunk_entity_relation"

    async def _get_graph():
        return graph

    storage._get_graph = _get_graph
    return storage


class _NullVdb:
    """A vector storage with no records and no ``client_storage``."""

    async def get_by_id(self, _id: str) -> None:
        return None


class _FakeGraph:
    """Minimal graph storage surface consumed by ``aexport_data``."""

    def __init__(self, nodes: dict[str, Any], edges: dict[tuple[str, str], Any]):
        self._nodes = nodes
        self._edges = edges

    async def get_all_labels(self) -> list[str]:
        return list(self._nodes)

    async def get_node(self, entity_name: str) -> Any:
        return self._nodes.get(entity_name)

    async def has_edge(self, src: str, tgt: str) -> bool:
        return (src, tgt) in self._edges

    async def get_edge(self, src: str, tgt: str) -> Any:
        return self._edges.get((src, tgt))


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------


class _ExcelFakeVdb:
    def __init__(self, data: list[dict[str, Any]] | None = None):
        self._data = data or []
        self.get_by_id_calls: list[str] = []

    @property
    async def client_storage(self) -> dict[str, Any]:
        return {"data": self._data}

    async def get_by_id(self, item_id: str) -> None:  # pragma: no cover - unused here
        self.get_by_id_calls.append(item_id)
        raise AssertionError("include_vector_data=False must not query the VDB")


def _read_sheets(path: Path) -> dict[str, list[list[Any]]]:
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.load_workbook(path)
    try:
        return {
            name: [list(row) for row in workbook[name].iter_rows(values_only=True)]
            for name in workbook.sheetnames
        }
    finally:
        workbook.close()


@pytest.mark.asyncio
async def test_excel_export_writes_one_sheet_per_non_empty_section(
    tmp_path: Path,
) -> None:
    graph = _FakeGraph(
        nodes={
            "ALICE": {"entity_type": "person", "source_id": "chunk-1"},
            "BOB": {"entity_type": "person", "source_id": "chunk-2"},
        },
        edges={("ALICE", "BOB"): {"weight": 1.0, "source_id": "chunk-1"}},
    )
    entities_vdb = _ExcelFakeVdb()
    relationships_vdb = _ExcelFakeVdb([{"__id__": "rel-1", "src_id": "ALICE"}])
    output = tmp_path / "export.xlsx"

    await aexport_data(
        graph,
        entities_vdb,
        relationships_vdb,
        str(output),
        file_format="excel",
    )

    assert entities_vdb.get_by_id_calls == []
    assert relationships_vdb.get_by_id_calls == []
    sheets = _read_sheets(output)
    assert set(sheets) == {"Entities", "Relations", "Relationships"}

    # index=False: the first column is a real field, not the DataFrame index.
    assert sheets["Entities"][0] == ["entity_name", "source_id", "graph_data"]
    assert sheets["Entities"][1][:2] == ["ALICE", "chunk-1"]
    assert sheets["Entities"][2][:2] == ["BOB", "chunk-2"]

    assert sheets["Relations"][0] == [
        "src_entity",
        "tgt_entity",
        "source_id",
        "graph_data",
    ]
    assert len(sheets["Relations"]) == 2
    assert sheets["Relations"][1][:3] == ["ALICE", "BOB", "chunk-1"]

    assert sheets["Relationships"][0] == ["relationship_id", "data"]
    assert sheets["Relationships"][1][0] == "rel-1"


@pytest.mark.asyncio
async def test_excel_export_leaves_missing_source_ids_empty(tmp_path: Path) -> None:
    """A source-less node yields source_id=None, which must not become text."""
    graph = _FakeGraph(
        nodes={"ALICE": {"entity_type": "person"}},
        edges={},
    )
    output = tmp_path / "export.xlsx"

    await aexport_data(
        graph,
        _ExcelFakeVdb(),
        _ExcelFakeVdb([{"__id__": "rel-1"}]),
        str(output),
        file_format="excel",
    )

    sheets = _read_sheets(output)
    # No edges and no relations rows -> that sheet is skipped entirely.
    assert set(sheets) == {"Entities", "Relationships"}
    header, row = sheets["Entities"]
    assert row[header.index("source_id")] is None
    assert row[header.index("graph_data")] == str({"entity_type": "person"})


# ---------------------------------------------------------------------------
# Markdown export cell escaping
# ---------------------------------------------------------------------------

_MD_RELATIONSHIP = {"__id__": "rel-1", "content": "net | gross"}


class _MdVdbWithClientStorage(_NullVdb):
    """Feeds the raw VectorDB Relationships section (FaissVectorDBStorage shape)."""

    @property
    def client_storage(self) -> dict[str, Any]:
        return {"data": [dict(_MD_RELATIONSHIP)]}


def _cells(row: str) -> list[str]:
    """Split a row the way a backslash-parity renderer does, then decode."""
    cells, current, i = [], "", 0
    body = row.strip()[1:]
    while i < len(body):
        char = body[i]
        if char == "\\" and i + 1 < len(body):
            current += char + body[i + 1]
            i += 2
            continue
        if char == "|":
            cells.append(current)
            current = ""
        else:
            current += char
        i += 1
    assert current.strip() == "", f"row does not end with a pipe: {row!r}"
    return [
        re.sub(
            r"(\\*)\\\|",
            lambda m: m.group(1)[: len(m.group(1)) // 2] + "|",
            cell.strip().replace("<br>", "\n"),
        )
        for cell in cells
    ]


@pytest.mark.asyncio
async def test_values_stay_in_their_cells_and_read_back_unchanged(
    tmp_path: Path,
) -> None:
    graph = nx.Graph()
    graph.add_node(
        "REVENUE",
        description="Q1 | Q2 totals for `List<T>` *net*",
        source_id="chunk-1<SEP>chunk-2",
    )
    graph.add_node("ACME\nCORP", description="vendor", source_id="chunk-3")
    graph.add_node("A\\|B", description="path C:\\x", source_id="chunk-5")
    graph.add_edge(
        "REVENUE", "ACME\nCORP", description="reported | audited", source_id="chunk-4"
    )
    output = tmp_path / "export.md"

    await aexport_data(
        _make_networkx_storage(graph),
        _NullVdb(),
        _MdVdbWithClientStorage(),
        str(output),
        file_format="md",
    )

    lines = output.read_text(encoding="utf-8").splitlines()
    tables = {}
    for heading in ("## Entities", "## Relations", "## Relationships"):
        start = lines.index(heading) + 2
        end = lines.index("", start) if "" in lines[start:] else len(lines)
        header, _rule, *rows = lines[start:end]
        assert [len(_cells(r)) for r in rows] == [len(_cells(header))] * len(rows)
        tables[heading] = [dict(zip(_cells(header), _cells(r))) for r in rows]

    entities = {r["entity_name"]: r for r in tables["## Entities"]}
    assert set(entities) == set(graph.nodes)
    for name, row in entities.items():
        assert row["source_id"] == graph.nodes[name]["source_id"]
        assert row["graph_data"] == str(graph.nodes[name])
    (relation,) = tables["## Relations"]
    edge = graph.edges["REVENUE", "ACME\nCORP"]
    assert relation["graph_data"] == str(edge)
    (relationship,) = tables["## Relationships"]
    assert relationship["data"] == str(_MD_RELATIONSHIP)


# ---------------------------------------------------------------------------
# Portable VectorDB client_storage handling
# ---------------------------------------------------------------------------


class _VdbWithAsyncClientStorage(_NullVdb):
    """Stands in for NanoVectorDBStorage: client_storage is an async property."""

    @property
    async def client_storage(self) -> dict[str, Any]:
        return {"data": [{"__id__": "rel-1", "weight": 1.0}]}


class _VdbCountingAsyncClientStorage(_NullVdb):
    """Same observable shape as above, but counts descriptor invocations."""

    def __init__(self) -> None:
        self.getter_calls = 0

    async def _snapshot(self) -> dict[str, Any]:
        return {"data": [{"__id__": "rel-1", "weight": 1.0}]}

    @property
    def client_storage(self) -> Any:
        self.getter_calls += 1
        return self._snapshot()


class _VdbWithSyncClientStorage(_NullVdb):
    """Stands in for FaissVectorDBStorage: client_storage is a sync property."""

    @property
    def client_storage(self) -> dict[str, Any]:
        return {"data": [{"__id__": "rel-2", "weight": 2.0}]}


class _VdbWithVectorBearingRows(_NullVdb):
    """Faiss shape, faithfully: the embedding lives IN the metadata row."""

    @property
    def client_storage(self) -> dict[str, Any]:
        return {
            "data": [
                {
                    "__id__": "rel-3",
                    "src_id": "ALICE",
                    "__vector__": [0.125, 0.25, 0.375],
                }
            ]
        }


async def _run_csv_export(
    tmp_path: Path,
    relationships_vdb: Any,
    *,
    include_vector_data: bool = False,
) -> str:
    graph = _FakeGraph(
        nodes={
            "ALICE": {"description": "ALICE node", "source_id": "chunk-1"},
            "BOB": {"description": "BOB node", "source_id": "chunk-1"},
        },
        edges={("ALICE", "BOB"): {"description": "knows", "source_id": "chunk-1"}},
    )
    output_path = tmp_path / "export.csv"
    await aexport_data(
        graph,
        _NullVdb(),
        relationships_vdb,
        str(output_path),
        file_format="csv",
        include_vector_data=include_vector_data,
    )
    return output_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_export_does_not_crash_without_client_storage(tmp_path: Path) -> None:
    content = await _run_csv_export(tmp_path, _NullVdb())

    assert "# ENTITIES" in content
    assert "ALICE" in content
    assert "# RELATIONS" in content
    assert "# RELATIONSHIPS" not in content


@pytest.mark.asyncio
async def test_export_includes_relationships_dump_for_async_client_storage(
    tmp_path: Path,
    lightrag_caplog: pytest.LogCaptureFixture,
) -> None:
    with lightrag_caplog.at_level("WARNING", logger="lightrag"):
        content = await _run_csv_export(tmp_path, _VdbWithAsyncClientStorage())

    assert "# ENTITIES" in content
    assert "# RELATIONS" in content
    assert "# RELATIONSHIPS" in content
    assert "rel-1" in content
    # The async shape awaits a reload-if-stale path: nothing to warn about.
    assert not [
        record
        for record in lightrag_caplog.records
        if "does not check for a newer on-disk commit" in record.message
    ], lightrag_caplog.text


@pytest.mark.asyncio
async def test_export_probes_client_storage_on_the_class_not_the_instance(
    tmp_path: Path,
) -> None:
    """The attribute is read exactly once, to use it -- never via hasattr."""
    relationships_vdb = _VdbCountingAsyncClientStorage()

    content = await _run_csv_export(tmp_path, relationships_vdb)

    assert "rel-1" in content
    assert relationships_vdb.getter_calls == 1


@pytest.mark.asyncio
async def test_export_includes_relationships_dump_for_sync_client_storage(
    tmp_path: Path,
    lightrag_caplog: pytest.LogCaptureFixture,
) -> None:
    """Faiss's sync client_storage is not awaited and warns about staleness."""
    with lightrag_caplog.at_level("WARNING", logger="lightrag"):
        content = await _run_csv_export(tmp_path, _VdbWithSyncClientStorage())

    assert "# ENTITIES" in content
    assert "# RELATIONS" in content
    assert "# RELATIONSHIPS" in content
    assert "rel-2" in content
    assert any(
        "does not check for a newer on-disk commit" in record.message
        for record in lightrag_caplog.records
    ), lightrag_caplog.text


@pytest.mark.asyncio
async def test_raw_dump_omits_embeddings_when_vector_data_is_not_requested(
    tmp_path: Path,
) -> None:
    """``include_vector_data=False`` must strip ``__vector__`` from Faiss rows."""
    content = await _run_csv_export(tmp_path, _VdbWithVectorBearingRows())

    assert "rel-3" in content
    assert "__vector__" not in content
    assert "0.375" not in content


@pytest.mark.asyncio
async def test_raw_dump_keeps_embeddings_when_vector_data_is_requested(
    tmp_path: Path,
) -> None:
    """The flag adds vector data; it must not strip what the caller asked for."""
    content = await _run_csv_export(
        tmp_path, _VdbWithVectorBearingRows(), include_vector_data=True
    )

    assert "__vector__" in content
    assert "0.375" in content


# ---------------------------------------------------------------------------
# Undirected relation deduplication
# ---------------------------------------------------------------------------


def _relation_rows(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = lines.index("# RELATIONS") + 1
    end = lines.index("", start)
    return list(csv.DictReader(lines[start:end]))


@pytest.mark.asyncio
async def test_undirected_relation_is_exported_once(tmp_path: Path) -> None:
    graph = nx.Graph()
    graph.add_edge("ALICE", "BOB", description="knows", source_id="chunk-1")
    graph.add_edge("CAROL", "BOB", description="works with", source_id="chunk-2")
    graph.add_node("DAVE", source_id="chunk-3")
    output = tmp_path / "export.csv"

    await aexport_data(
        _make_networkx_storage(graph),
        _NullVdb(),
        _NullVdb(),
        str(output),
        file_format="csv",
    )

    rows = _relation_rows(output)
    exported = sorted(
        tuple(sorted((row["src_entity"], row["tgt_entity"]))) for row in rows
    )
    assert exported == [("ALICE", "BOB"), ("BOB", "CAROL")]
    assert {row["source_id"] for row in rows} == {"chunk-1", "chunk-2"}
