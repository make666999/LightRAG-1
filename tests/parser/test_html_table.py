"""HTML table parsing: dimensions, header detection, and Unicode slicing."""

from __future__ import annotations

import pytest

from lightrag.parser._html_table import (
    extract_html_table_info,
    extract_thead_html,
    html_table_inner_body,
    unwrap_html_table,
)
from lightrag.parser.markdown.extract import table_marker
from lightrag.parser.markdown.ir_builder import NativeMarkdownIRBuilder

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Dimensions & optional end tags
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        (
            "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>",
            (2, 2),
        ),
        (
            "<table><tr><td>a<td>b</tr><tr><td>c<td>d</tr></table>",
            (2, 2),
        ),
        (
            "<table><tr><td>a</td><td>b</td><tr><td>c</td><td>d</td></table>",
            (2, 2),
        ),
        ("<table><tr><td>a<td>b<tr><td>c<td>d</table>", (2, 2)),
        (
            "<table><thead><tr><th>A<th>B</thead><tbody><tr><td>a<td>b</tbody></table>",
            (2, 2),
        ),
        (
            '<table><tr><th colspan="3">Heading<tr><td>a<td>b<td>c</table>',
            (2, 3),
        ),
        ("<tr><td>a<td>b<tr><td>c<td>d", (2, 2)),
        (
            "<table><tr><td>outer<table><tr><td>nested-a<td>nested-b"
            "</table><td>peer</table>",
            (1, 2),
        ),
        (
            "<table><tr><td>outer<table><tr><td>nested-a</td>"
            "<td>nested-b</td></tr></table></td><td>peer</td></tr></table>",
            (1, 2),
        ),
        ("<table><tr></tr><tr><td>a</td></tr></table>", (1, 1)),
    ],
)
def test_table_dimensions_with_optional_end_tags(html, expected):
    info = extract_html_table_info(html)
    assert (info.num_rows, info.num_cols) == expected


# ---------------------------------------------------------------------------
# Blank HTML header detection & rendered character references
# ---------------------------------------------------------------------------


# Tag removal happens before entity decoding; the ordinary header is
# tests/parser/markdown/test_ir_builder.py::test_html_table_becomes_html_irtable_with_thead_and_dims.
@pytest.mark.parametrize("text", ["&lt;ID&gt;"])
def test_visible_headers_keep_their_original_markup(text: str) -> None:
    header = f"<thead><tr><th colspan='2'>{text}</th></tr></thead>"

    assert extract_thead_html(f"<table>{header}</table>") == header


def test_markdown_ir_keeps_the_table_but_not_a_blank_header() -> None:
    html = (
        "<table><thead><tr><th>&nbsp;</th></tr></thead>"
        "<tbody><tr><td>42</td></tr></tbody></table>"
    )
    ir = NativeMarkdownIRBuilder().normalize(
        [{"content": table_marker("t1")}],
        document_name="table.md",
        asset_dir_name="table.blocks.assets",
        parse_metadata={"md_tables": {"t1": {"kind": "html", "html": html}}},
    )
    (table,) = ir.blocks[0].tables

    assert table.table_header is None
    assert table.html == html
    assert table.body_override and "<td>42</td>" in table.body_override
    assert (table.num_rows, table.num_cols) == (2, 1)


# ---------------------------------------------------------------------------
# Unicode offset preservation across slicing helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prefix", ["", "<p>İstanbul</p>"])
@pytest.mark.parametrize("tag", ["table", "TABLE"])
def test_unwrap_preserves_unicode_and_complete_closing_tag(prefix, tag):
    table = f'<{tag} data-city="İstanbul"><tr><td>İzmir</td></tr></{tag}>'
    assert unwrap_html_table(f"<html><body>{prefix}{table}</body></html>") == table


@pytest.mark.parametrize("tag", ["thead", "THEAD"])
def test_header_slice_preserves_unicode_before_and_inside_header(tag):
    header = f"<{tag}><tr><th>İstanbul</th></tr></{tag}>"
    assert extract_thead_html(f'<table title="İzmir">{header}</table>') == header


def test_inner_body_does_not_include_part_of_closing_tag():
    body = "<tr><td>İstanbul</td></tr>"
    assert html_table_inner_body(f'<TABLE title="İzmir">{body}</TABLE>') == body
