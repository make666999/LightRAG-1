"""Consolidated multimodal context, content truncation, and sidecar path containment tests.

Covers:
  * ``trim_content_to_budget`` & table row/char trimming (JSON/HTML tables,
    equations, malformed markup, logarithmic bisection probe bounds, and
    preserving cells containing literal ``</table>``).
  * Sidecar image path containment (``_resolve_sidecar_image_path``,
    GHSA-8rgj-chc2-6chv): refusing absolute/traversal/symlink escapes,
    malformed/non-string/NUL values, and symlink loops while keeping
    ``RESOLVED``, ``REFUSED``, and ``MISSING`` distinct.
  * Surrounding-context extraction (``find_target_span``, ``build_surrounding``,
    ``enrich_sidecars_with_surrounding``, ``_resolve_surrounding_budget``):
    table/cite stripping, separator-driven segmentation, char fallback,
    strip-before-budget ordering, LF line endings, and env overrides.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import lightrag.multimodal_context as mc
from lightrag.multimodal_context import (
    DEFAULT_SURROUNDING_MAX_TOKENS,
    _resolve_surrounding_budget,
    _row_trim_table_leading,
    _row_trim_table_trailing,
    build_surrounding,
    enrich_sidecars_with_surrounding,
    find_target_span,
    load_chunk_separators,
    parse_table_tag,
    serialize_html_rows,
    split_html_rows,
    trim_content_to_budget,
)
from lightrag.pipeline import _resolve_sidecar_image_path, _SidecarPathOutcome
from lightrag.table_markup import dumps_json_table_body
from lightrag.utils import Tokenizer

from ._fakes import SimpleTokenizerImpl, make_tokenizer

pytestmark = pytest.mark.offline

_MARKER_RE = re.compile(
    r"<!-- content truncated from (\d+) to (\d+) tokens, head preserved -->"
)
_MARKUP_ROWS = [["Tag", "Meaning"], ["</table>", "closes"], ["<tr>", "row"]]


def _tokenizer() -> Tokenizer:
    return make_tokenizer("char")


def _ntok(text: str) -> int:
    return len(_tokenizer().encode(text))


class _CountingCharTokenizer(SimpleTokenizerImpl):
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, content: str) -> list[int]:
        self.calls += 1
        return super().encode(content)


# ---------------------------------------------------------------------------
# trim_content_to_budget & table row/char trimming
# ---------------------------------------------------------------------------


def test_short_content_passes_through():
    tok = _tokenizer()
    content = "<table><tr><td>cell</td></tr></table>"
    out, was_trimmed = trim_content_to_budget(
        content, kind="tables", max_tokens=10_000, tokenizer=tok
    )
    assert out == content
    assert was_trimmed is False
    assert _MARKER_RE.search(out) is None


def test_table_json_row_trim_keeps_head():
    tok = _tokenizer()
    rows = [[f"r{i}c0", f"r{i}c1"] for i in range(10)]
    content = '<table id="t-json" format="json">' + json.dumps(rows) + "</table>"
    out, was_trimmed = trim_content_to_budget(
        content, kind="tables", max_tokens=150, tokenizer=tok
    )
    assert was_trimmed is True
    assert "<table " in out
    assert "</table>" in out
    assert "r0c0" in out
    assert "r9c0" not in out
    table_close = out.rfind("</table>")
    marker_match = _MARKER_RE.search(out)
    assert marker_match is not None
    assert marker_match.start() > table_close
    assert len(tok.encode(out)) <= 150


def test_table_char_fallback_when_single_row_oversized():
    tok = _tokenizer()
    long_cell = "X" * 400
    content = (
        '<table id="t-big" format="json">'
        + json.dumps([[long_cell]], ensure_ascii=False)
        + "</table>"
    )
    out, was_trimmed = trim_content_to_budget(
        content, kind="tables", max_tokens=120, tokenizer=tok
    )
    assert was_trimmed is True
    assert out.lstrip().startswith("<table ")
    assert "</table>" in out
    assert _MARKER_RE.search(out) is not None
    assert len(tok.encode(out)) <= 120


def test_equation_char_trim_keeps_head():
    tok = _tokenizer()
    content = "HEAD_" + "A" * 500 + "_TAIL"
    out, was_trimmed = trim_content_to_budget(
        content, kind="equations", max_tokens=100, tokenizer=tok
    )
    assert was_trimmed is True
    assert out.startswith("HEAD_")
    assert "_TAIL" not in out
    assert _MARKER_RE.search(out) is not None
    assert len(tok.encode(out)) <= 100


def test_malformed_table_falls_back_to_char_trim():
    tok = _tokenizer()
    content = "<table><tr><td>" + "Z" * 500 + "</td></tr>"
    out, was_trimmed = trim_content_to_budget(
        content, kind="tables", max_tokens=100, tokenizer=tok
    )
    assert was_trimmed is True
    assert out.startswith("<table>")
    assert _MARKER_RE.search(out) is not None
    assert len(tok.encode(out)) <= 100


def test_zero_budget_returns_input_unchanged():
    tok = _tokenizer()
    content = "x" * 5000
    out, was_trimmed = trim_content_to_budget(
        content, kind="tables", max_tokens=0, tokenizer=tok
    )
    assert out == content
    assert was_trimmed is False


def test_tokenizer_none_returns_input_unchanged():
    content = "x" * 5000
    out, was_trimmed = trim_content_to_budget(
        content, kind="tables", max_tokens=100, tokenizer=None
    )
    assert out == content
    assert was_trimmed is False


def test_marker_reports_original_and_final_token_counts():
    tok = _tokenizer()
    content = "x" * 500
    out, was_trimmed = trim_content_to_budget(
        content, kind="equations", max_tokens=100, tokenizer=tok
    )
    assert was_trimmed is True
    match = _MARKER_RE.search(out)
    assert match is not None
    original_in_marker = int(match.group(1))
    final_in_marker = int(match.group(2))
    assert original_in_marker == 500
    assert final_in_marker < original_in_marker
    assert len(tok.encode(out)) <= 100


def test_empty_content_returns_unchanged():
    tok = _tokenizer()
    out, was_trimmed = trim_content_to_budget(
        "", kind="tables", max_tokens=100, tokenizer=tok
    )
    assert out == ""
    assert was_trimmed is False


def test_budget_smaller_than_marker_stays_within_cap():
    tok = _tokenizer()
    content = "hello world " * 50
    out, was_trimmed = trim_content_to_budget(
        content, kind="equations", max_tokens=20, tokenizer=tok
    )
    assert was_trimmed is True
    assert len(tok.encode(out)) <= 20
    assert len(out) == 20
    assert out.startswith("hello")
    assert _MARKER_RE.search(out) is None


def test_table_budget_too_small_for_wrapper_avoids_partial_tags():
    tok = _tokenizer()
    rows = [[f"r{i}c0", f"r{i}c1"] for i in range(10)]
    content = '<table id="t-tiny" format="json">' + json.dumps(rows) + "</table>"
    out, was_trimmed = trim_content_to_budget(
        content, kind="tables", max_tokens=20, tokenizer=tok
    )
    assert was_trimmed is True
    assert len(tok.encode(out)) <= 20
    assert not (out.lstrip().startswith("<table") and "</table>" not in out)
    assert _MARKER_RE.search(out) is None


@pytest.mark.parametrize("fmt", ["json", "html"])
@pytest.mark.parametrize("keep", ["head", "tail"])
def test_row_trim_keeps_most_rows_with_logarithmic_probes(fmt, keep):
    """Row trimming bisects to the maximal fitting row count in O(log N) probes."""
    row_count = 4000
    if fmt == "json":
        rows = [[f"r{i}c0", f"r{i}c1"] for i in range(row_count)]
        attrs = 'id="t" format="json"'
        content = f"<table {attrs}>{json.dumps(rows)}</table>"
    else:
        rows_html = "".join(
            f"<tr><td>r{i}c0</td><td>r{i}c1</td></tr>" for i in range(row_count)
        )
        attrs = 'id="t" format="html"'
        content = f"<table {attrs}>{rows_html}</table>"

    counting = _CountingCharTokenizer()
    tok = Tokenizer(model_name="char", tokenizer=counting)
    budget = len(content) // 3
    trim = _row_trim_table_trailing if keep == "head" else _row_trim_table_leading
    out = trim(content, budget, tok)

    assert out is not None
    assert len(out) <= budget
    assert counting.calls < 40, f"{counting.calls} tokenizer calls"

    if fmt == "json":
        _, kept = parse_table_tag(out)
        k = len(kept)
        more = rows[: k + 1] if keep == "head" else rows[-(k + 1) :]
        bigger = f"<table {attrs}>{json.dumps(more, ensure_ascii=False)}</table>"
    else:
        all_rows = split_html_rows(rows_html)
        k = len(split_html_rows(out[out.index(">") + 1 : -len("</table>")]))
        more = all_rows[: k + 1] if keep == "head" else all_rows[-(k + 1) :]
        bigger = f"<table {attrs}>{serialize_html_rows(more)}</table>"
    assert 0 < k < row_count
    assert len(bigger) > budget
    edge = "r0c0" if keep == "head" else f"r{row_count - 1}c0"
    assert edge in out


def _markup_table() -> str:
    return (
        f'<table id="tb-1" format="json">{dumps_json_table_body(_MARKUP_ROWS)}</table>'
    )


@pytest.mark.parametrize(
    ("trim", "kept"),
    [
        ("_row_trim_table_trailing", _MARKUP_ROWS[:2]),
        ("_row_trim_table_leading", _MARKUP_ROWS[1:]),
    ],
)
def test_row_trim_keeps_a_closing_tag_cell_inside_the_tag(trim, kept):
    tok = _tokenizer()
    table = _markup_table()

    out = getattr(mc, trim)(table, len(table) - 5, tok)

    assert out is not None
    assert out.count("</table>") == 1
    parsed = parse_table_tag(out)
    assert parsed is not None, out
    assert parsed[1] == kept
    assert len(tok.encode(out)) <= len(table) - 5


def test_char_fallback_keeps_a_closing_tag_cell_inside_the_tag():
    tok = _tokenizer()
    rows = [["</table>" + "X" * 400]]
    table = f'<table id="tb-1" format="json">{dumps_json_table_body(rows)}</table>'

    out = _row_trim_table_trailing(table, 120, tok)

    assert out is not None
    assert out.count("</table>") == 1
    parsed = parse_table_tag(out)
    assert parsed is not None, out
    assert "</table>" in parsed[1][0][0]
    assert len(tok.encode(out)) <= 120


# ---------------------------------------------------------------------------
# Sidecar image path containment (GHSA-8rgj-chc2-6chv)
# ---------------------------------------------------------------------------


@pytest.fixture
def sidecar_dir(tmp_path: Path) -> Path:
    """A document's parsed dir, with one legitimately-placed asset."""
    d = tmp_path / "ws" / "tenantA" / "__parsed__" / "report.docx.parsed"
    assets = d / "report.blocks.assets"
    assets.mkdir(parents=True)
    (assets / "image1.png").write_bytes(b"\x89PNG\r\n\x1a\nLEGIT")
    return d


def test_absolute_path_outside_the_sidecar_is_refused(sidecar_dir, tmp_path):
    victim = tmp_path / "victim_home" / "id_rsa_screenshot.png"
    victim.parent.mkdir(parents=True)
    victim.write_bytes(b"\x89PNG\r\n\x1a\nSECRET")

    path, outcome = _resolve_sidecar_image_path(str(victim), sidecar_dir)
    assert path is None
    assert outcome is _SidecarPathOutcome.REFUSED


def test_dotdot_escape_is_refused(sidecar_dir, tmp_path):
    tenant_b = (
        tmp_path
        / "ws"
        / "tenantB"
        / "__parsed__"
        / "victim.docx.parsed"
        / "victim.blocks.assets"
    )
    tenant_b.mkdir(parents=True)
    (tenant_b / "image1.png").write_bytes(b"\x89PNG\r\n\x1a\nTENANT-B-CONFIDENTIAL")

    traversal = (
        "../../../tenantB/__parsed__/victim.docx.parsed/victim.blocks.assets/image1.png"
    )
    path, outcome = _resolve_sidecar_image_path(traversal, sidecar_dir)
    assert path is None
    assert outcome is _SidecarPathOutcome.REFUSED


def test_absolute_path_to_a_non_image_is_refused(sidecar_dir, tmp_path):
    target = tmp_path / "hostname"
    target.write_text("victim-host\n")
    path, outcome = _resolve_sidecar_image_path(str(target), sidecar_dir)
    assert path is None
    assert outcome is _SidecarPathOutcome.REFUSED


@pytest.mark.parametrize("bad", [123])
def test_non_string_path_value_is_refused_not_raised(sidecar_dir, bad):
    path, outcome = _resolve_sidecar_image_path(bad, sidecar_dir)
    assert path is None
    assert outcome is _SidecarPathOutcome.REFUSED


def test_asset_inside_the_document_directory_resolves(sidecar_dir):
    resolved, outcome = _resolve_sidecar_image_path(
        "report.blocks.assets/image1.png", sidecar_dir
    )
    assert outcome is _SidecarPathOutcome.RESOLVED
    assert resolved is not None
    assert resolved.read_bytes() == b"\x89PNG\r\n\x1a\nLEGIT"


def test_resolution_survives_a_symlinked_prefix(tmp_path):
    real = tmp_path / "real" / "report.docx.parsed"
    (real / "report.blocks.assets").mkdir(parents=True)
    (real / "report.blocks.assets" / "image1.png").write_bytes(b"PNGDATA")

    link = tmp_path / "linked"
    link.symlink_to(tmp_path / "real", target_is_directory=True)

    resolved, outcome = _resolve_sidecar_image_path(
        "report.blocks.assets/image1.png", link / "report.docx.parsed"
    )
    assert outcome is _SidecarPathOutcome.RESOLVED
    assert resolved is not None
    assert resolved.read_bytes() == b"PNGDATA"


def test_missing_and_empty_values_are_refused(sidecar_dir):
    assert _resolve_sidecar_image_path(None, sidecar_dir) == (
        None,
        _SidecarPathOutcome.REFUSED,
    )
    assert _resolve_sidecar_image_path("", sidecar_dir) == (
        None,
        _SidecarPathOutcome.REFUSED,
    )


def test_a_contained_but_absent_file_is_missing_not_refused(sidecar_dir):
    path, outcome = _resolve_sidecar_image_path(
        "report.blocks.assets/absent.png", sidecar_dir
    )
    assert path is None
    assert outcome is _SidecarPathOutcome.MISSING


def test_a_directory_inside_the_sidecar_is_not_a_file(sidecar_dir):
    path, outcome = _resolve_sidecar_image_path("report.blocks.assets", sidecar_dir)
    assert path is None
    assert outcome is _SidecarPathOutcome.MISSING


def test_symlink_escaping_the_sidecar_is_refused(sidecar_dir, tmp_path):
    victim = tmp_path / "outside_secret.png"
    victim.write_bytes(b"\x89PNG\r\n\x1a\nSECRET")
    (sidecar_dir / "report.blocks.assets" / "escape.png").symlink_to(victim)

    path, outcome = _resolve_sidecar_image_path(
        "report.blocks.assets/escape.png", sidecar_dir
    )
    assert path is None
    assert outcome is _SidecarPathOutcome.REFUSED


def test_embedded_nul_is_skipped_not_raised(sidecar_dir):
    path, outcome = _resolve_sidecar_image_path(
        "report.blocks.assets/a\x00b.png", sidecar_dir
    )
    assert path is None
    assert outcome is _SidecarPathOutcome.REFUSED


def test_symlink_loop_is_skipped_not_raised(sidecar_dir):
    assets = sidecar_dir / "report.blocks.assets"
    (assets / "loopA.png").symlink_to(assets / "loopB.png")
    (assets / "loopB.png").symlink_to(assets / "loopA.png")

    path, outcome = _resolve_sidecar_image_path(
        "report.blocks.assets/loopA.png", sidecar_dir
    )
    assert path is None
    assert outcome is not _SidecarPathOutcome.RESOLVED


def test_symlink_loop_in_the_sidecar_dir_itself_is_skipped(tmp_path):
    a = tmp_path / "dirA"
    b = tmp_path / "dirB"
    a.symlink_to(b, target_is_directory=True)
    b.symlink_to(a, target_is_directory=True)

    path, outcome = _resolve_sidecar_image_path("report.blocks.assets/x.png", a)
    assert path is None
    assert outcome is not _SidecarPathOutcome.RESOLVED


# ---------------------------------------------------------------------------
# Surrounding-context extraction & sidecar enrichment
# ---------------------------------------------------------------------------


def _surround(
    kind: str,
    item_id: str,
    block: str,
    *,
    leading: int = 2000,
    trailing: int = 2000,
    separators: list[str] | None = None,
) -> dict[str, str]:
    span = find_target_span(kind, item_id, block)
    assert span is not None, f"{kind}/{item_id} not found in block"
    return build_surrounding(
        kind=kind,
        block_content=block,
        span=span,
        tokenizer=_tokenizer(),
        leading_max_tokens=leading,
        trailing_max_tokens=trailing,
        separators=load_chunk_separators() if separators is None else separators,
    )


@pytest.mark.parametrize(
    ("kind", "item_id", "content", "expected"),
    [
        pytest.param(
            "tables",
            "tb-abcd-0007",
            'before <table format="json" id="tb-abcd-0007">[[1,2]]</table> after',
            '<table format="json" id="tb-abcd-0007">[[1,2]]</table>',
            id="table-id-not-first-attribute",
        ),
        pytest.param(
            "drawings",
            "im-other",
            '<drawing id="im-1" />',
            None,
            id="unknown-id",
        ),
    ],
)
def test_find_target_span(kind, item_id, content, expected):
    span = find_target_span(kind, item_id, content)
    got = None if span is None else content[span[0] : span[1]]
    assert got == expected


def test_table_surrounding_strips_other_tables_before_counting():
    surr = _surround(
        "tables",
        "tb-target",
        '<table id="tb-other" format="json">[["a","b"],["c","d"]]</table> '
        "narrative text describing the report. "
        '<table id="tb-target" format="json">[["x","y"]]</table>'
        " concluding remarks.",
    )
    assert surr == {
        "leading": " narrative text describing the report. ",
        "trailing": " concluding remarks.",
    }


def test_table_surrounding_supports_cite_marker_and_strips_sibling_cites():
    surr = _surround(
        "tables",
        "tb-target",
        'prefix <cite type="table" refid="tb-other">表0</cite> '
        'narrative <cite type="table" refid="tb-target">表1</cite> suffix',
    )
    assert surr == {"leading": "prefix  narrative ", "trailing": " suffix"}


def test_table_surrounding_anchors_on_the_tag_not_a_preceding_cite():
    surr = _surround(
        "tables",
        "tb-1",
        "as shown in Table 1, "
        '<cite type="table" refid="tb-1">Table 1</cite>: '
        '<table id="tb-1" format="json">[[1,2]]</table>'
        " concluding remarks.",
    )
    assert surr == {
        "leading": "as shown in Table 1, : ",
        "trailing": " concluding remarks.",
    }


def test_chunk_r_separators_env_drives_segment_boundary(monkeypatch):
    monkeypatch.setenv("CHUNK_R_SEPARATORS", json.dumps(["|"]))
    seps = load_chunk_separators()
    assert seps == ["|"]
    surr = _surround(
        "drawings",
        "d",
        'aaaaaaaaa|bbbbbbbbb|<drawing id="d" />|ccccccccc|ddddddddd',
        leading=12,
        trailing=12,
        separators=seps,
    )
    assert surr["leading"] == "bbbbbbbbb|"


@pytest.mark.parametrize(
    ("block", "side", "kept_char"),
    [
        pytest.param("X" * 5000 + '<drawing id="d" />', "leading", "X", id="leading"),
        pytest.param('<drawing id="d" />' + "Y" * 5000, "trailing", "Y", id="trailing"),
    ],
)
def test_oversized_closest_segment_char_truncated(block, side, kept_char):
    surr = _surround("drawings", "d", block, leading=200, trailing=200)
    assert surr[side] == kept_char * 200


def test_drawing_surrounding_row_trims_oversized_json_table():
    rows = [[f"r{i}c0", f"r{i}c1"] for i in range(10)]
    block = (
        '<table id="tb-big" format="json">'
        + json.dumps(rows)
        + '</table> <drawing id="d" />'
    )
    surr = _surround("drawings", "d", block, leading=80, trailing=80)
    assert surr["leading"] == (
        '<table format="json">[["r8c0", "r8c1"], ["r9c0", "r9c1"]]</table> '
    )


def test_drawing_surrounding_row_trims_oversized_html_table():
    rows_html = "".join(f"<tr><td>r{i}c0</td><td>r{i}c1</td></tr>" for i in range(10))
    block = (
        '<drawing id="d" /> '
        f'<table id="tb-h" format="html"><tbody>{rows_html}</tbody></table>'
    )
    trailing = _surround("drawings", "d", block, leading=120, trailing=120)["trailing"]
    assert trailing.startswith(" <table ")
    assert trailing.endswith("</tbody></table>")
    assert _ntok(trailing) <= 120
    assert "r0c0" in trailing
    assert "r9c0" not in trailing


def test_drawing_surrounding_char_trims_oversized_single_json_row():
    big_table = (
        '<table id="tb-big" format="json">'
        + json.dumps([["A" * 200 + "TAIL"]])
        + "</table>"
    )
    leading = _surround(
        "drawings", "d", big_table + '<drawing id="d" />', leading=90, trailing=90
    )["leading"]
    assert leading.startswith("<table ")
    assert leading.endswith("</table>")
    assert "TAIL" in leading
    assert _ntok(leading) <= 90
    body = leading[leading.index(">") + 1 : -len("</table>")]
    assert isinstance(json.loads(body), list)


def test_drawing_surrounding_char_trims_oversized_single_html_row():
    big_table = (
        '<table id="tb-h" format="html">'
        f"<tbody><tr><td>HEAD{'B' * 200}</td></tr></tbody>"
        "</table>"
    )
    trailing = _surround(
        "drawings", "d", f'<drawing id="d" />{big_table}', leading=100, trailing=100
    )["trailing"]
    assert trailing.startswith("<table ")
    assert trailing.endswith("</table>")
    assert "<tr><td>HEAD" in trailing
    assert _ntok(trailing) <= 100


def test_surrounding_strips_cite_refid_keeping_the_wrapper():
    surr = _surround(
        "drawings",
        "im-target",
        'see <cite type="table" refid="tb-x">Table 1</cite> for details. '
        '<drawing id="im-target" caption="Fig 3" /> end.',
    )
    assert surr["leading"] == 'see <cite type="table">Table 1</cite> for details. '


def test_strip_happens_before_budget_truncation():
    surr = _surround(
        "equations",
        "eq-1",
        '<drawing id="im-prev" path="some/long/path.png" src="raw/long/path.png"'
        ' caption="C" />'
        '<equation id="eq-1" format="latex">y</equation>'
        " tail.",
        leading=30,
    )
    assert surr["leading"] == '<drawing caption="C" />'


def _write_blocks(tmp_path: Path, base: str, blocks: list[tuple[str, str]]) -> str:
    blocks_path = tmp_path / f"{base}.blocks.jsonl"
    lines = [json.dumps({"type": "meta", "format": "lightrag"})]
    for blockid, content in blocks:
        lines.append(
            json.dumps(
                {
                    "type": "content",
                    "blockid": blockid,
                    "format": "plain_text",
                    "content": content,
                    "heading": "h",
                    "parent_headings": [],
                    "level": 1,
                },
                ensure_ascii=False,
            )
        )
    blocks_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(blocks_path)


def _write_sidecar(tmp_path: Path, base: str, root_key: str, items: dict) -> Path:
    path = tmp_path / f"{base}.{root_key}.json"
    path.write_text(
        json.dumps({"version": "1.0", root_key: items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _sidecar_item(item_id: str, blockid: str = "b1", **extra) -> dict:
    return {item_id: {"id": item_id, "blockid": blockid, "heading": "h", **extra}}


def _read_item(path: Path, root_key: str, item_id: str) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))[root_key][item_id]


def test_enrich_only_updates_enabled_modalities(tmp_path):
    blocks_path = _write_blocks(
        tmp_path,
        "doc",
        [
            (
                "b1",
                'intro. <drawing id="im-1" path="a.png" src="a" />'
                ' middle <table id="tb-1" format="json">[["a"]]</table>'
                ' tail <equation id="eq-1" format="latex">e</equation> end.',
            )
        ],
    )
    drawings = _write_sidecar(tmp_path, "doc", "drawings", _sidecar_item("im-1"))
    tables = _write_sidecar(tmp_path, "doc", "tables", _sidecar_item("tb-1"))
    equations = _write_sidecar(tmp_path, "doc", "equations", _sidecar_item("eq-1"))

    counts = enrich_sidecars_with_surrounding(
        blocks_path=blocks_path,
        enabled_modalities={"drawings"},
        tokenizer=_tokenizer(),
        leading_max_tokens=2000,
        trailing_max_tokens=2000,
    )
    assert counts == {"drawings": 1, "tables": 0, "equations": 0}
    assert _read_item(drawings, "drawings", "im-1")["surrounding"][
        "leading"
    ].startswith("intro.")
    assert "surrounding" not in _read_item(tables, "tables", "tb-1")
    assert "surrounding" not in _read_item(equations, "equations", "eq-1")


def test_enrich_does_not_cross_block_boundaries(tmp_path):
    blocks_path = _write_blocks(
        tmp_path,
        "doc",
        [
            ("bA", "earlier block content."),
            ("bB", 'later block. <drawing id="im-1" path="a.png" src="a" /> tail.'),
        ],
    )
    drawings = _write_sidecar(tmp_path, "doc", "drawings", _sidecar_item("im-1", "bB"))

    enrich_sidecars_with_surrounding(
        blocks_path=blocks_path,
        enabled_modalities={"drawings"},
        tokenizer=_tokenizer(),
        leading_max_tokens=2000,
        trailing_max_tokens=2000,
    )
    assert _read_item(drawings, "drawings", "im-1")["surrounding"] == {
        "leading": "later block. ",
        "trailing": " tail.",
    }


def test_enrich_rewrite_overwrites_surrounding_and_keeps_other_fields(
    tmp_path, emulate_windows_newlines
):
    blocks_path = _write_blocks(
        tmp_path,
        "doc",
        [("b1", "L" * 500 + '<drawing id="im-1" caption="C" />' + "T" * 500)],
    )
    analysis = {"name": "x", "summary": "", "detail_description": ""}
    drawings = _write_sidecar(
        tmp_path,
        "doc",
        "drawings",
        _sidecar_item("im-1", llm_analyze_result=analysis),
    )

    def run(budget: int):
        counts = enrich_sidecars_with_surrounding(
            blocks_path=blocks_path,
            enabled_modalities={"drawings"},
            tokenizer=_tokenizer(),
            leading_max_tokens=budget,
            trailing_max_tokens=budget,
        )
        assert counts["drawings"] == 1
        return _read_item(drawings, "drawings", "im-1")

    first = run(300)
    assert first["surrounding"] == {"leading": "L" * 300, "trailing": "T" * 300}
    assert first["llm_analyze_result"] == analysis

    second = run(50)
    assert second["surrounding"] == {"leading": "L" * 50, "trailing": "T" * 50}
    assert second["llm_analyze_result"] == analysis
    assert b"\r\n" not in drawings.read_bytes(), "the rewrite wrote CRLF"


def test_env_var_leading_and_trailing_budgets_apply_independently(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SURROUNDING_LEADING_MAX_TOKENS", "5")
    monkeypatch.setenv("SURROUNDING_TRAILING_MAX_TOKENS", "20")
    blocks_path = _write_blocks(
        tmp_path,
        "doc",
        [("b1", "X" * 200 + '<drawing id="im-1" path="a.png" src="a" />' + "Y" * 200)],
    )
    drawings = _write_sidecar(tmp_path, "doc", "drawings", _sidecar_item("im-1"))

    enrich_sidecars_with_surrounding(
        blocks_path=blocks_path,
        enabled_modalities={"drawings"},
        tokenizer=_tokenizer(),
    )
    assert _read_item(drawings, "drawings", "im-1")["surrounding"] == {
        "leading": "X" * 5,
        "trailing": "Y" * 20,
    }


def test_env_var_invalid_value_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("SURROUNDING_LEADING_MAX_TOKENS", "not-a-number")
    monkeypatch.setenv("SURROUNDING_TRAILING_MAX_TOKENS", "not-a-number")
    assert _resolve_surrounding_budget(None, None) == (
        DEFAULT_SURROUNDING_MAX_TOKENS,
        DEFAULT_SURROUNDING_MAX_TOKENS,
    )
