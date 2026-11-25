"""Unit tests for entity/relation extraction parsing and recovery.

Covers:
- Tuple delimiter repair without forging separators in free text.
- Inline single-line record separation and record-keyword collision handling.
- JSON extraction recovery from trailing prose braces and top-level array rejection.
- Reserved JavaScript prototype member name rejection on both text and JSON paths.
- Numeric relationship descriptions not hijacking edge weights.
"""

from __future__ import annotations

import pytest

from lightrag.operate import (
    _handle_single_entity_extraction,
    _handle_single_relationship_extraction,
    _normalize_and_validate_entity_type,
    _process_extraction_result,
    _process_json_extraction_result,
)
from lightrag.utils import (
    fix_tuple_delimiter_corruption,
    split_string_by_multi_markers,
)

pytestmark = pytest.mark.offline


# --------------------------------------------------------------------------- #
# Tuple delimiter repair in free text
# --------------------------------------------------------------------------- #

_DELIM = "<|#|>"
_CORE = "#"


@pytest.mark.parametrize(
    "name, description, token",
    [
        pytest.param(
            "ShellRedirect",
            "Documents the shell redirect form a <|> b used in pipelines",
            "<|>",
            id="spaced_angle_pipe",
        ),
        pytest.param(
            "Guard",
            "Pseudocode guard: if (x <|| y) then skip",
            "<||",
            id="spaced_double_pipe",
        ),
    ],
)
def test_spaced_delimiter_lookalike_in_description_does_not_split_entity(
    name: str, description: str, token: str
) -> None:
    """Prose/code mentioning ``<|>`` or ``<||`` must stay one description field."""
    record = f"entity{_DELIM}{name}{_DELIM}concept{_DELIM}{description}"
    fixed = fix_tuple_delimiter_corruption(record, _CORE, _DELIM)
    attrs = split_string_by_multi_markers(fixed, [_DELIM])
    assert len(attrs) == 4
    entity = _handle_single_entity_extraction(attrs, "chunk-1", 1, "docs.md")
    assert entity is not None
    assert entity["entity_name"] == name
    assert token in entity["description"]


def test_glued_missing_core_delimiter_still_repaired() -> None:
    """LLM field separators with a missing core (entity<|>name) still repair."""
    broken = "entity<|>ShellRedirect<|>concept<|>Kept description"
    fixed = fix_tuple_delimiter_corruption(broken, _CORE, _DELIM)
    attrs = split_string_by_multi_markers(fixed, [_DELIM])
    assert attrs == ["entity", "ShellRedirect", "concept", "Kept description"]
    entity = _handle_single_entity_extraction(attrs, "chunk-3", 1, "docs.md")
    assert entity is not None
    assert entity["description"] == "Kept description"


# --------------------------------------------------------------------------- #
# Inline single-line record separation
# --------------------------------------------------------------------------- #

_ALICE = f"entity{_DELIM}Alice{_DELIM}person{_DELIM}Alice is an engineer."
_BOB = f"entity{_DELIM}Bob{_DELIM}person{_DELIM}Bob is a manager."
_CAROL = f"entity{_DELIM}Carol{_DELIM}person{_DELIM}Carol is a designer."
_REL = (
    f"relation{_DELIM}Alice{_DELIM}Bob{_DELIM}reports to{_DELIM}Alice reports to Bob."
)


async def _parse_text(text: str):
    return await _process_extraction_result(text, "chunk-1", 0)


@pytest.mark.asyncio
async def test_entities_on_one_line_are_all_kept() -> None:
    multi_line_nodes, multi_line_edges = await _parse_text(
        "\n".join([_ALICE, _BOB, _CAROL, _REL]) + "\n<|COMPLETE|>"
    )
    one_line_nodes, one_line_edges = await _parse_text(
        _DELIM.join([_ALICE, _BOB, _CAROL, _REL]) + "\n<|COMPLETE|>"
    )

    assert set(multi_line_nodes) == {"Alice", "Bob", "Carol"}
    assert set(one_line_nodes) == set(multi_line_nodes)
    assert set(one_line_edges) == set(multi_line_edges) == {("Alice", "Bob")}
    assert (
        one_line_nodes["Bob"][0]["description"]
        == multi_line_nodes["Bob"][0]["description"]
    )


@pytest.mark.asyncio
async def test_entity_named_like_a_record_keyword_on_one_line() -> None:
    relational = (
        f"entity{_DELIM}relational database{_DELIM}concept{_DELIM}A kind of database."
    )
    nodes, _ = await _parse_text(_DELIM.join([_ALICE, relational]) + "\n<|COMPLETE|>")
    assert set(nodes) == {"Alice", "relational database"}


@pytest.mark.asyncio
async def test_parenthesised_first_record_still_parses() -> None:
    nodes, edges = await _parse_text(
        f"(entity{_DELIM}Carol{_DELIM}person{_DELIM}A designer.\n<|COMPLETE|>"
    )
    assert set(nodes) == {"Carol"}
    assert not edges


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source",
    ["entity framework", "relation store", "relationship graph"],
)
async def test_relation_endpoint_named_like_a_record_keyword_on_one_line(
    source: str,
) -> None:
    relation = f"relation{_DELIM}{source}{_DELIM}Alice{_DELIM}uses{_DELIM}{source} is used by Alice."
    nodes, edges = await _parse_text(_DELIM.join([_ALICE, relation]) + "\n<|COMPLETE|>")
    assert set(nodes) == {"Alice"}
    assert set(edges) == {(source, "Alice")}


@pytest.mark.asyncio
async def test_line_opening_with_the_delimiter_keeps_its_first_entity() -> None:
    nodes, edges = await _parse_text(
        _DELIM + _DELIM.join([_ALICE, _BOB, _REL]) + "\n<|COMPLETE|>"
    )
    assert set(nodes) == {"Alice", "Bob"}
    assert set(edges) == {("Alice", "Bob")}


# --------------------------------------------------------------------------- #
# JSON extraction recovery
# --------------------------------------------------------------------------- #

_TRAILING_BRACE_JSON = (
    '{"entities":[{"name":"Alice","type":"person","description":"an engineer"}],'
    '"relationships":[{"source":"Alice","target":"Acme",'
    '"keywords":"works at","description":"Alice works at Acme"}]}'
    " trailing {brace}"
)


@pytest.mark.asyncio
async def test_process_json_extraction_recovers_from_trailing_brace_prose() -> None:
    nodes, edges = await _process_json_extraction_result(
        _TRAILING_BRACE_JSON,
        chunk_key="chunk-1",
        timestamp=1,
        file_path="doc.pdf",
    )
    assert "Alice" in nodes, "entity dropped — object not recovered from prose"
    assert nodes["Alice"][0]["entity_type"] == "person"
    assert ("Alice", "Acme") in edges
    assert edges[("Alice", "Acme")][0]["description"] == "Alice works at Acme"


@pytest.mark.asyncio
async def test_process_json_extraction_top_level_array_yields_nothing() -> None:
    raw = '[{"name":"Alice","type":"person","description":"x"}]'
    nodes, edges = await _process_json_extraction_result(
        raw, chunk_key="chunk-1", timestamp=1, file_path="doc.pdf"
    )
    assert nodes == {}
    assert edges == {}


# --------------------------------------------------------------------------- #
# Reserved JS prototype entity_type rejection (text + JSON paths)
# --------------------------------------------------------------------------- #

_RESERVED_ENTITY_TYPES = ["__proto__", "constructor", "prototype"]


def _json_with_type(entity_type: str) -> str:
    return (
        '{"entities":[{"name":"Globex","type":"' + entity_type + '",'
        '"description":"Globex supplies equipment"}],"relationships":[]}'
    )


def test_shared_helper_takes_first_token_of_comma_list() -> None:
    assert _normalize_and_validate_entity_type("person, extra", "ctx") == "person"


@pytest.mark.parametrize("reserved", _RESERVED_ENTITY_TYPES)
def test_text_path_drops_reserved_entity_type(reserved: str) -> None:
    attrs = ["entity", "GLOBEX LTD", reserved, "Globex Ltd supplies equipment."]
    assert _handle_single_entity_extraction(attrs, "chunk-1", 1, "doc.txt") is None


def test_text_path_keeps_ordinary_entity() -> None:
    attrs = ["entity", "GLOBEX LTD", "organization", "Globex Ltd supplies equipment."]
    entity = _handle_single_entity_extraction(attrs, "chunk-1", 1, "doc.txt")
    assert entity is not None
    assert entity["entity_type"] == "organization"


@pytest.mark.asyncio
@pytest.mark.parametrize("reserved", _RESERVED_ENTITY_TYPES[:1])
async def test_json_path_drops_reserved_entity_type(reserved: str) -> None:
    nodes, _edges = await _process_json_extraction_result(
        _json_with_type(reserved), chunk_key="chunk-1", timestamp=1, file_path="doc.pdf"
    )
    assert "Globex" not in nodes


@pytest.mark.asyncio
async def test_json_path_keeps_ordinary_entity() -> None:
    nodes, _edges = await _process_json_extraction_result(
        _json_with_type("organization"),
        chunk_key="chunk-1",
        timestamp=1,
        file_path="doc.pdf",
    )
    assert "Globex" in nodes
    assert nodes["Globex"][0]["entity_type"] == "organization"


# --------------------------------------------------------------------------- #
# Numeric relationship description must not hijack weight
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("description,expected_weight", [("12.345", 1.0)])
def test_numeric_description_keeps_default_weight(
    description: str, expected_weight: float
) -> None:
    result = _handle_single_relationship_extraction(
        ["relationship", "Alice", "Bob", "knows", description],
        "chunk-1",
        1,
    )
    assert result is not None
    assert result["description"] == description
    assert result["weight"] == expected_weight
