"""Unit tests for ``lightrag.parser.markdown.extract`` (pure extraction).

Locks in the supported-subset contract: ATX heading splitting, fenced-code
suppression, pipe / HTML table recognition with headers, block-level ``$$``
math (and inline ``$`` left alone), literal-image code-span/escape preservation,
and inline image resolution via a stubbed resolver.
"""

from __future__ import annotations

import itertools
import random
import re
import time
import tracemalloc
from collections.abc import Callable

import pytest

from lightrag.parser.markdown import extract as extract_module
from lightrag.parser.markdown.extract import (
    PREFACE_HEADING,
    ResolvedImage,
    _PAREN_SOURCE_MAX_CHARS,
    _has_unescaped_pipe,
    _replace_inline_images,
    _split_pipe_row,
    extract_markdown,
)


pytestmark = pytest.mark.offline


def _reference_tail_end(line: str, source_end: int) -> int | None:
    index = source_end
    while index < len(line) and line[index].isspace():
        index += 1
    if index < len(line) and line[index] == ")":
        return index + 1
    if index == source_end or index >= len(line) or line[index] != '"':
        return None
    index = line.find('"', index + 1)
    if index < 0:
        return None
    index += 1
    while index < len(line) and line[index].isspace():
        index += 1
    return index + 1 if index < len(line) and line[index] == ")" else None


def _reference_match(line: str, start: int) -> tuple[int, str] | None:
    alt_end = line.find("]", start + 2)
    if alt_end < 0 or line[alt_end + 1 : alt_end + 2] != "(":
        return None
    source_start = alt_end + 2
    while source_start < len(line) and line[source_start].isspace():
        source_start += 1
    if source_start >= len(line):
        return None
    if line[source_start] == "<":
        angle_end = line.find(">", source_start + 1)
        if angle_end >= 0:
            end = _reference_tail_end(line, angle_end + 1)
            if end is not None:
                return end, line[source_start : angle_end + 1]
    depth = 0
    source_end = source_start
    while source_end < len(line) and not line[source_end].isspace():
        if line[source_end] == "(":
            depth += 1
        elif line[source_end] == ")":
            if depth == 0:
                break
            depth -= 1
        source_end += 1
    if source_end == source_start or depth != 0:
        return None
    end = _reference_tail_end(line, source_end)
    return None if end is None else (end, line[source_start:source_end])


_REFERENCE_CANDIDATE_RE = re.compile(r"`+|!\[")
_REFERENCE_BACKTICKS_RE = re.compile(r"`+")


def _reference_escaped(line: str, index: int) -> bool:
    backslashes = len(line[:index]) - len(line[:index].rstrip("\\"))
    return backslashes % 2 == 1


def _reference_skip_backticks(line: str, run: re.Match[str]) -> int:
    opener_start = run.start() + _reference_escaped(line, run.start())
    length = run.end() - opener_start
    if length:
        for closer in _REFERENCE_BACKTICKS_RE.finditer(line, run.end()):
            if closer.end() - closer.start() == length:
                return closer.end()
    return run.end()


def _reference_alt_end_is_literal(line: str, start: int) -> bool:
    alt_end = line.find("]", start + 2)
    if alt_end < 0:
        return False
    if _reference_escaped(line, alt_end):
        return True
    scan = start + 2
    while run := _REFERENCE_BACKTICKS_RE.search(line, scan, alt_end):
        scan = _reference_skip_backticks(line, run)
        if scan > alt_end:
            return True
    return False


def _reference_replace_inline_images(line: str) -> str:
    """The strict image grammar, written naively (quadratic, no caches).

    It is the oracle for the scanner's cached searches, code spans and
    escapes included. It has no lenient retry, so it is only valid for lines
    without an image file extension.
    """
    parts: list[str] = []
    position = 0
    cursor = 0
    while candidate := _REFERENCE_CANDIDATE_RE.search(line, cursor):
        start = candidate.start()
        if candidate.group() != "![":
            cursor = _reference_skip_backticks(line, candidate)
            continue
        matched = None
        if not _reference_escaped(line, start) and not (
            _reference_alt_end_is_literal(line, start)
        ):
            matched = _reference_match(line, start)
        if matched is None:
            cursor = start + 2
            continue
        end, src = matched
        parts.append(line[position:start])
        parts.append(f"<{src}>")
        position = cursor = end
    parts.append(line[position:])
    return "".join(parts)


class _StubResolver:
    """Resolves ``http(s)`` -> external link, everything else -> local bytes."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def resolve(self, src: str) -> ResolvedImage:
        self.calls.append(src)
        if src.startswith(("http://", "https://")):
            return ResolvedImage(kind="external", url=src, fmt="png")
        return ResolvedImage(
            kind="local",
            asset_ref="sha:" + src,
            data=b"BYTES:" + src.encode(),
            suggested_name="img.png",
            fmt="png",
        )


def _scan_time_ratio(small: str, large: str) -> float:
    """How much longer ``large`` takes to scan than ``small``.

    Timing on a shared machine is noisy in one direction only: a scan can be
    delayed, never sped up. So each line is timed by its fastest scan, and the
    lowest ratio of a few attempts is returned. An eightfold input that
    scanned quadratically would still give about 64 every time.
    """

    def fastest(line: str) -> float:
        best = float("inf")
        for _ in range(5):
            started = time.perf_counter()
            _replace_inline_images(line, lambda src: src)
            best = min(best, time.perf_counter() - started)
        return best

    ratio = float("inf")
    for _ in range(3):
        ratio = min(ratio, fastest(large) / max(fastest(small), 1e-9))
        if ratio < 12:
            break
    return ratio


def _extract(md: str):
    return extract_markdown(md, image_resolver=_StubResolver())


def _resolved(line: str) -> tuple[list[str], str]:
    calls: list[str] = []
    out = _replace_inline_images(line, lambda src: calls.append(src) or "<img>")
    return calls, out


# ---------------------------------------------------------------------------
# Headings & fenced code blocks
# ---------------------------------------------------------------------------


def test_headings_split_blocks_and_track_parents():
    md = "# A\nintro\n## B\nbody\n### C\ndeep\n# D\nlast"
    ex = _extract(md)
    summary = [(b["heading"], b["level"], b["parent_headings"]) for b in ex.blocks]
    assert summary == [
        ("A", 1, []),
        ("B", 2, ["A"]),
        ("C", 3, ["A", "B"]),
        ("D", 1, []),
    ]
    # Heading line is rendered into the block body (markdown-style).
    assert ex.blocks[0]["content"].startswith("# A")


def test_skipped_heading_levels_keep_same_level_headings_as_siblings():
    ex = _extract("# A\n### B\n### C")
    summary = [(b["heading"], b["parent_headings"]) for b in ex.blocks]
    assert summary == [
        ("A", []),
        ("B", ["A"]),
        ("C", ["A"]),
    ]


def test_content_before_first_heading_is_preface():
    ex = _extract("loose intro line\n\n# Real")
    assert ex.blocks[0]["heading"] == PREFACE_HEADING
    assert ex.blocks[0]["level"] == 0
    assert "loose intro line" in ex.blocks[0]["content"]


@pytest.mark.parametrize(
    ("md", "snippets"),
    [
        pytest.param(
            "# H\n```python\n# not a heading\n$$ not eq $$\n| not | table |\n![x](no.png)\n```\n",
            ("# not a heading", "$$ not eq $$"),
            id="backtick-fence",
        ),
        pytest.param(
            "# H\n~~~\n## still code\n~~~\nafter",
            ("## still code",),
            id="tilde-fence",
        ),
    ],
)
def test_fenced_code_suppresses_detection(md: str, snippets: tuple[str, ...]):
    ex = _extract(md)
    assert len(ex.blocks) == 1
    assert not ex.tables and not ex.equations and not ex.drawings
    for snippet in snippets:
        assert snippet in ex.blocks[0]["content"]


def test_one_line_backtick_code_span_does_not_open_a_fence():
    """A backtick run followed by text holding a backtick is an inline code
    span, not a fence opener, so it must not swallow the rest of the file."""
    md = "# Install\n```pip install lightrag```\n## Usage\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    ex = _extract(md)
    assert [b["heading"] for b in ex.blocks] == ["Install", "Usage"]
    assert "```pip install lightrag```" in ex.blocks[0]["content"]
    assert len(ex.tables) == 1


# ---------------------------------------------------------------------------
# Pipe tables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("md", "expected_header", "expected_rows", "not_in_content"),
    [
        pytest.param(
            "| Name | Age |\n|------|-----|\n| Bob | 30 |\n| Sue | 25 |\n",
            [["Name", "Age"]],
            [["Bob", "30"], ["Sue", "25"]],
            None,
            id="basic-header-and-rows",
        ),
        pytest.param(
            # ``\|`` is cell content, not a column separator.
            "| h1 | h2 |\n| --- | --- |\n| a \\| b | c |\n",
            [["h1", "h2"]],
            [["a | b", "c"]],
            None,
            id="escaped-pipe-in-cell",
        ),
        pytest.param(
            # Escaped pipe in the header must not inflate its column count.
            "| a \\| b | h2 |\n| --- | --- |\n| 1 | 2 |\n",
            [["a | b", "h2"]],
            [["1", "2"]],
            None,
            id="escaped-pipe-in-header",
        ),
        pytest.param(
            # Once the delimiter establishes the table, an escaped-pipe-only
            # line stays in the table body (GFM pads it to the header width).
            "| h1 | h2 |\n| --- | --- |\n| a | b |\ntail \\| text\n",
            [["h1", "h2"]],
            [["a", "b"], ["tail | text"]],
            "tail \\| text",
            id="escaped-pipe-only-body-row",
        ),
        pytest.param(
            # ``\\|`` is an escaped backslash followed by a real separator.
            "| h1 | h2 |\n| --- | --- |\n| a | b |\nc \\\\| d\n",
            [["h1", "h2"]],
            [["a", "b"], ["c \\", "d"]],
            None,
            id="escaped-backslash-before-separator",
        ),
    ],
)
def test_pipe_table_extraction(
    md: str,
    expected_header: list[list[str]],
    expected_rows: list[list[str]],
    not_in_content: str | None,
):
    ex = _extract(md)
    assert len(ex.tables) == 1
    (table,) = ex.tables.values()
    assert table["kind"] == "pipe"
    assert table["header"] == expected_header
    assert table["rows"] == expected_rows
    if not_in_content is not None:
        assert not_in_content not in ex.blocks[0]["content"]


def test_pipe_table_stops_at_atx_heading_with_pipe():
    md = (
        "# Regional inventory\n"
        "| Item | Quantity |\n"
        "| --- | --- |\n"
        "| # Available | 2 |\n"
        "## Delivery | Next steps\n"
        "Confirm the shipment date.\n"
    )
    ex = _extract(md)

    (table,) = ex.tables.values()
    assert table["rows"] == [["# Available", "2"]]
    assert [block["heading"] for block in ex.blocks] == [
        "Regional inventory",
        "Delivery | Next steps",
    ]
    assert ex.blocks[1]["parent_headings"] == ["Regional inventory"]
    assert "Confirm the shipment date." in ex.blocks[1]["content"]


@pytest.mark.parametrize(
    ("md", "preserved_text"),
    [
        pytest.param(
            "| just | text |\nnot a delimiter\n",
            None,
            id="missing-delimiter-row",
        ),
        pytest.param(
            # Mismatched column counts (2 vs 1) over bare thematic break.
            "foo | bar\n---\nnext paragraph\n",
            "foo | bar",
            id="pipe-line-over-thematic-break",
        ),
        pytest.param(
            # Delimiter column count differs from header.
            "| a | b | c |\n| --- | --- |\n| 1 | 2 | 3 |\n",
            None,
            id="column-count-mismatch",
        ),
        pytest.param(
            # No unescaped pipe -> paragraph, not a 1-column table over ``---``.
            "foo \\| bar\n---\nnext paragraph\n",
            "foo \\| bar",
            id="escaped-pipe-only-over-thematic-break",
        ),
    ],
)
def test_pipe_table_rejected_when_invalid(md: str, preserved_text: str | None):
    ex = _extract(md)
    assert not ex.tables
    if preserved_text is not None:
        assert preserved_text in ex.blocks[0]["content"]


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("foo", False),
        ("a | b", True),
        ("| foo |", True),  # a one-column table still carries separators
        ("a \\| b", False),  # escaped pipe: cell content, no separator
        ("a \\\\| b", True),  # escaped backslash, then a real separator
        ("a \\\\\\| b", False),  # escaped backslash, then an escaped pipe
    ],
)
def test_has_unescaped_pipe_agrees_with_the_row_splitter(line, expected):
    # The table gate and the row splitter read one scan, so they cannot
    # disagree about what an escape is; this pins the rule both of them see.
    # The prepended ``|`` absorbs the splitter's opening-pipe strip, leaving a
    # cell count that reflects exactly the separators in ``line``.
    assert _has_unescaped_pipe(line) is expected
    assert (len(_split_pipe_row("|" + line)) > 1) is expected


# ---------------------------------------------------------------------------
# HTML tables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("md", "expected_html", "expected_trailing"),
    [
        pytest.param(
            "<table>\n<thead><tr><th>K</th></tr></thead>\n"
            "<tbody><tr><td>a</td></tr></tbody>\n</table>\n",
            "<table>\n<thead><tr><th>K</th></tr></thead>\n"
            "<tbody><tr><td>a</td></tr></tbody>\n</table>",
            "",
            id="multiline-verbatim-thead-tbody",
        ),
        pytest.param(
            "<table><tr><td>a</td></tr></table> trailing prose",
            "<table><tr><td>a</td></tr></table>",
            " trailing prose",
            id="same-line-trailing-text",
        ),
        pytest.param(
            "<table>\n<tr><td>a</td></tr>\n</table> trailing prose",
            "<table>\n<tr><td>a</td></tr>\n</table>",
            " trailing prose",
            id="multiline-close-trailing-text",
        ),
        pytest.param(
            "<table><tr><td>İ</td></tr></table>tail",
            "<table><tr><td>İ</td></tr></table>",
            "tail",
            id="unicode-content-trailing-text",
        ),
        pytest.param(
            '<table data-note="</table>"><tr><td>a</td></tr></table> tail',
            '<table data-note="</table>"><tr><td>a</td></tr></table>',
            " tail",
            id="closing-tag-inside-attribute",
        ),
        pytest.param(
            "<table><!-- don't remove --><tr><td>a</td></tr></table> tail",
            "<table><!-- don't remove --><tr><td>a</td></tr></table>",
            " tail",
            id="quotes-inside-comments",
        ),
        pytest.param(
            # Unmatched quote inside RCDATA (<textarea>) must not stall close scan.
            "<table>\n<tr><td><textarea>x<y's value</textarea></td></tr>\n</table> tail",
            "<table>\n<tr><td><textarea>x<y's value</textarea></td></tr>\n</table>",
            " tail",
            id="unmatched-quote-inside-textarea",
        ),
        pytest.param(
            # Raw mode waits until the opening <textarea ...> tag itself closes.
            '<table><textarea data-note="</textarea>">x<y\'s value</textarea></table> tail',
            '<table><textarea data-note="</textarea>">x<y\'s value</textarea></table>',
            " tail",
            id="raw-mode-waits-for-opening-tag-close",
        ),
        pytest.param(
            # Valid HTML closing tag broken by whitespace/newline before ``>``.
            "<table><textarea>x</textarea\n>\n</table> tail",
            "<table><textarea>x</textarea\n>\n</table>",
            " tail",
            id="raw-mode-closing-tag-split-across-lines",
        ),
        pytest.param(
            # Bare ``>`` inside foreign-content CDATA must not expose inner ``</table>``.
            "<table><tr><td><svg><![CDATA[x > </table> y]]></svg></td></tr></table> tail",
            "<table><tr><td><svg><![CDATA[x > </table> y]]></svg></td></tr></table>",
            " tail",
            id="closing-tag-inside-cdata",
        ),
        pytest.param(
            # Same-line nested table depth is tracked; inner stays verbatim.
            "<table><tr><td><table>x</table></td></tr></table> tail",
            "<table><tr><td><table>x</table></td></tr></table>",
            " tail",
            id="nested-same-line-table-depth",
        ),
        pytest.param(
            # Custom/namespaced tag ``<textarea:widget>`` needs real name boundary.
            "<table><textarea:widget>x</textarea:widget></table> tail",
            "<table><textarea:widget>x</textarea:widget></table>",
            " tail",
            id="boundary-after-raw-element-name",
        ),
        pytest.param(
            # Literal ``</table>`` inside <script> raw-text content is ignored.
            '<table><tr><td><script>if (a < b) { x.close("</table>"); }</script>'
            "</td></tr></table> tail",
            '<table><tr><td><script>if (a < b) { x.close("</table>"); }</script>'
            "</td></tr></table>",
            " tail",
            id="tag-like-text-inside-script",
        ),
        pytest.param(
            # Unmatched quote inside <style> raw-text content is ignored.
            '<table><tr><td><style>content: "it\'s";</style></td></tr></table> tail',
            '<table><tr><td><style>content: "it\'s";</style></td></tr></table>',
            " tail",
            id="quote-inside-style",
        ),
        pytest.param(
            # Non-ASCII letter after ``<`` is not an HTML tag opener.
            "<table><td>x <é's text</td></table> tail",
            "<table><td>x <é's text</td></table>",
            " tail",
            id="non-ascii-tag-open",
        ),
        pytest.param(
            # RCDATA (<textarea>) must not treat literal ``<!--`` as a comment.
            "<table><tr><td><textarea>a <!-- not a comment</textarea></td></tr></table> tail",
            "<table><tr><td><textarea>a <!-- not a comment</textarea></td></tr></table>",
            " tail",
            id="comment-marker-inside-rcdata-textarea",
        ),
        pytest.param(
            # RCDATA (<title>) must not treat literal ``<!--`` as a comment.
            "<table><tr><td><title>a <!-- not a comment</title></td></tr></table> tail",
            "<table><tr><td><title>a <!-- not a comment</title></td></tr></table>",
            " tail",
            id="comment-marker-inside-rcdata-title",
        ),
    ],
)
def test_html_table_single_block_extraction(
    md: str, expected_html: str, expected_trailing: str
):
    ex = _extract(md)
    assert len(ex.tables) == 1
    (table,) = ex.tables.values()
    assert table["kind"] == "html"
    assert table["html"] == expected_html
    if expected_trailing:
        assert ex.blocks[0]["content"].endswith(expected_trailing)


def test_html_table_processes_inline_image_in_trailing_text():
    resolver = _StubResolver()
    ex = extract_markdown(
        "<table><tr><td>a</td></tr></table> ![plot](plot.png)",
        image_resolver=resolver,
    )

    assert len(ex.drawings) == 1
    assert resolver.calls == ["plot.png"]
    assert "![plot](plot.png)" not in ex.blocks[0]["content"]


def test_html_table_tag_name_split_across_lines_is_not_recognized():
    """A line break inside a tag NAME itself is not valid HTML, so the element
    never closes and the unclosed table stays plain text."""
    ex = _extract("<table><textarea>x</texta\nrea></table> tail")

    assert not ex.tables
    assert ex.blocks[0]["content"] == "<table><textarea>x</texta\nrea></table> tail"


def test_html_table_recognizes_second_same_line_table():
    """A second supported table on the same line as the first's close gets its
    own entry via the targeted table-only suffix retry."""
    ex = _extract("<table>A</table><table>B</table>")

    assert [t["html"] for t in ex.tables.values()] == [
        "<table>A</table>",
        "<table>B</table>",
    ]
    (ref1, ref2) = ex.tables.keys()
    assert (
        ex.blocks[0]["content"] == f'<mdtable ref="{ref1}"/>\n<mdtable ref="{ref2}"/>'
    )


def test_html_table_trailing_suffix_does_not_gain_line_anchored_semantics():
    """Re-queuing the suffix after ``</table>`` must not let it open a fence or
    start a heading it never was in the source."""
    ex = _extract("<table>A</table># literal")
    assert ex.blocks[0]["heading"] == PREFACE_HEADING
    assert ex.blocks[0]["content"].endswith("# literal")
    assert len(ex.blocks) == 1

    ex2 = _extract("<table>A</table>```\n# actual heading")
    assert ex2.blocks[0]["heading"] == PREFACE_HEADING
    assert ex2.blocks[0]["content"].endswith("```")
    assert ex2.blocks[1]["heading"] == "actual heading"


# ---------------------------------------------------------------------------
# Block equations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("md", "expected_equations", "expected_headings", "expected_snippet"),
    [
        pytest.param(
            "$$ E = mc^2 $$\n\ntext\n\n$$\n\\sum x\n$$\n",
            ["E = mc^2", "\\sum x"],
            None,
            None,
            id="single-and-multiline",
        ),
        pytest.param(
            "cost is $5 and $10 today",
            [],
            None,
            "$5",
            id="inline-dollar-ignored",
        ),
        pytest.param(
            "# A\n$$E = mc^2$$ relates mass and energy.\n## B\n$$\np = mv\n$$\n",
            ["p = mv"],
            ["A", "B"],
            None,
            id="display-math-with-trailing-prose-does-not-open-block",
        ),
        pytest.param(
            "$$\nE = mc^2\nno closing",
            [],
            None,
            "$$",
            id="unclosed-block-is-plain-text",
        ),
    ],
)
def test_block_equation_extraction(
    md: str,
    expected_equations: list[str],
    expected_headings: list[str] | None,
    expected_snippet: str | None,
):
    ex = _extract(md)
    assert list(ex.equations.values()) == expected_equations
    if expected_headings is not None:
        assert [b["heading"] for b in ex.blocks] == expected_headings
    if expected_snippet is not None:
        assert expected_snippet in ex.blocks[0]["content"]


# ---------------------------------------------------------------------------
# Inline images, literal images, grammar & resource bounds
# ---------------------------------------------------------------------------


def test_inline_image_local_and_external_and_dedup():
    md = "![a](one.png) and ![b](one.png) and ![c](http://x/y.png)"
    resolver = _StubResolver()
    ex = extract_markdown(md, image_resolver=resolver)
    # Three occurrences -> three drawings; same local src deduped to one asset.
    assert len(ex.drawings) == 3
    assert len(ex.assets) == 1
    kinds = sorted(d["kind"] for d in ex.drawings.values())
    assert kinds == ["external", "local", "local"]
    # Asset dedup is by ``asset_ref`` in extract (the resolver itself owns any
    # per-src call caching), so the stub sees one call per occurrence.
    assert resolver.calls.count("one.png") == 2


def test_base64_src_not_echoed_into_drawing_src():
    md = "![x](data:image/png;base64,QUJD)"

    class _B64:
        """Minimal resolver returning a local image for data URLs."""

        def resolve(self, src):
            return ResolvedImage(
                kind="local",
                asset_ref="sha:b64",
                data=b"ABC",
                suggested_name="image.png",
                fmt="png",
            )

    ex = extract_markdown(md, image_resolver=_B64())
    (drawing,) = ex.drawings.values()
    assert drawing["src"] == ""


def test_reference_style_image_not_recognized():
    # ``![alt][id]`` reference-style is out of the supported subset.
    ex = _extract("![alt][id]\n\n[id]: real.png")
    assert not ex.drawings


@pytest.mark.parametrize(
    "literal",
    [
        r"\![example](example.png)",
        "`![example](example.png)`",
        "``a ` tick ![example](example.png)``",
        "``![example](example.png)` still code``",
        # A backslash escapes one backtick; the rest of the run still opens.
        r"\``![example](example.png)`",
        # The code span opened in the alt text swallows its ``]``.
        "![a `b](example.png)`",
        # An escaped ``]`` never ends the alt text.
        r"![a\](example.png)",
    ],
)
def test_literal_images_are_preserved_without_resolving(literal):
    calls: list[str] = []

    class _Resolver:
        """Records resolved image sources and returns an external stub."""

        def resolve(self, src):
            calls.append(src)
            return ResolvedImage(kind="external", url=src)

    result = extract_markdown(
        literal + " and ![real](real.png)", image_resolver=_Resolver()
    )
    assert calls == ["real.png"]
    assert len(result.drawings) == 1
    assert literal in result.blocks[0]["content"]


@pytest.mark.parametrize(
    "prefix",
    [
        r"\\",  # an escaped backslash escapes nothing after it
        "`unclosed ",  # an unclosed run protects nothing
        r"\`",  # an escaped single backtick opens nothing
        r"`a\`",  # a backslash does not escape a closing run
        "``a`` ",  # a closed span ends before the image
    ],
)
def test_escaped_backslash_or_unclosed_code_does_not_hide_image(prefix):
    calls: list[str] = []

    class _Resolver:
        """Records resolved image sources and returns an external stub."""

        def resolve(self, src):
            calls.append(src)
            return ResolvedImage(kind="external", url=src)

    result = extract_markdown(prefix + "![real](real.png)", image_resolver=_Resolver())
    assert calls == ["real.png"]
    assert len(result.drawings) == 1


@pytest.mark.parametrize(
    ("line", "expected_calls", "expected_out"),
    [
        ("![a `b` c](real.png)", ["real.png"], "<img>"),
        (r"![a\\](real.png)", ["real.png"], "<img>"),
        ('![a ``b`` c](real.png "title")', ["real.png"], "<img>"),
        # After an unclosed ``x run scans the line, a closed `a` span must not
        # make a later `b ...` span look unclosed or hide a later closer.
        ("``x `a` `b ![i](i.png)` `", [], None),
        ("``x `a` `in ``` t` ![real](real.png) ```", ["real.png"], None),
        (
            "`a` ``b ` c`` ![x](x.png) `![y](y.png)` tail",
            ["x.png"],
            "`a` ``b ` c`` <img> `![y](y.png)` tail",
        ),
    ],
)
def test_literal_image_scanner_code_span_and_escape_handling(
    line: str, expected_calls: list[str], expected_out: str | None
):
    calls, out = _resolved(line)
    assert calls == expected_calls
    if expected_out is not None:
        assert out == expected_out


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("before ![alt](image.png) after", "before <image.png> after"),
        ('![alt](<a b.png> "caption")', "<<a b.png>>"),
        ("![](<a>suffix)", "<<a>suffix>"),
        ("![broken ![nested](ok.png)", "<ok.png>"),
        # CommonMark: a bare destination may hold balanced parentheses.
        ("![x](shot(1).png)", "<shot(1).png>"),
        ("see ![x](a(b(c))d) end", "see <a(b(c))d> end"),
        ('![x](shot(1).png "caption")', "<shot(1).png>"),
        (
            "![x](https://en.wikipedia.org/wiki/File:Foo_(bar).svg) tail)",
            "<https://en.wikipedia.org/wiki/File:Foo_(bar).svg> tail)",
        ),
        ("(![x](a(1).png))", "(<a(1).png>)"),
        ("![x](a.png)(b)", "<a.png>(b)"),
        # Unbalanced: not an image (CommonMark asks for escapes or <...>).
        ("![x](foo(and(bar))", "![x](foo(and(bar))"),
        ('![x](foo(bar "title")', '![x](foo(bar "title")'),
        ("![x]()", "![x]()"),
    ],
)
def test_inline_image_scanner_preserves_supported_image_grammar(line, expected):
    assert _replace_inline_images(line, lambda src: f"<{src}>") == expected


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        # Typora-style local path: spaces, no angle brackets.
        ("![alt](./my doc.assets/a.png)", "<./my doc.assets/a.png>"),
        ("a ![x](my doc/a b.JPG) b", "a <my doc/a b.JPG> b"),
        ('![x](my doc/a.png "the caption")', "<my doc/a.png>"),
        ("![x](  my doc/a.webp  )", "<my doc/a.webp>"),
        # Balanced parentheses inside such a path.
        ("![x](shots/shot (1).png)", "<shots/shot (1).png>"),
        # The first extension is followed by more path, so the source runs on.
        ("![x](a.png copy.png)", "<a.png copy.png>"),
        ("![x](pics (old.png) new.svg)", "<pics (old.png) new.svg>"),
        ("![x](a b.png) and ![y](c d.gif)", "<a b.png> and <c d.gif>"),
        ("![broken ![nested](a b.png)", "<a b.png>"),
    ],
)
def test_inline_image_scanner_accepts_bare_source_with_spaces(line, expected):
    assert _replace_inline_images(line, lambda src: f"<{src}>") == expected


@pytest.mark.parametrize(
    "line",
    [
        # No image extension: still not an image.
        "![alt](some words here)",
        "![alt](my doc/readme.md)",
        # The candidate's own parenthesis closes before any image extension,
        # so a later parenthetical or link is not swallowed into its source.
        "![alt](not an image) see (figure one.png)",
        "![alt](two words) and [a link](my pic.png)",
        # Unbalanced parentheses and quotes are not paths.
        "![alt](shot (1.png)",
        '![alt](my "doc".png)',
        '![alt](a b.png "unterminated)',
        # The extension must end the source.
        "![alt](my doc.png?raw=1 x)",
        "![alt](my doc.png extra)",
    ],
)
def test_inline_image_scanner_lenient_retry_leaves_non_images_alone(line):
    assert _replace_inline_images(line, lambda src: f"<{src}>") == line


@pytest.mark.parametrize(
    ("line", "expected", "drawing_src"),
    [
        pytest.param(
            # ``old.png "...")`` must not hide the real image inside its title.
            '(old.png "![x](my pic.png)")',
            '(old.png "<my pic.png>")',
            "my pic.png",
            id="image-inside-earlier-tail",
        ),
        pytest.param(
            # Rejected candidate's retry must stop at the next ``![``.
            "![a](two words ![b](ok.png)",
            "![a](two words <ok.png>",
            None,
            id="stops-before-next-candidate",
        ),
        pytest.param(
            # Strict grammar already matched ``a.png``; retry must not extend it.
            "![x](a.png) trailing (b c.png)",
            "<a.png> trailing (b c.png)",
            None,
            id="strict-wins-over-lenient",
        ),
    ],
)
def test_lenient_retry_interaction_boundaries(
    line: str, expected: str, drawing_src: str | None
):
    assert _replace_inline_images(line, lambda src: f"<{src}>") == expected
    if drawing_src is not None:
        ex = _extract(line)
        (drawing,) = ex.drawings.values()
        assert drawing["src"] == drawing_src


def test_image_path_with_spaces_becomes_a_drawing():
    resolver = _StubResolver()
    ex = extract_markdown(
        "Intro\n\n![shot](./my doc.assets/shot (1).png)\n", image_resolver=resolver
    )
    assert resolver.calls == ["./my doc.assets/shot (1).png"]
    (drawing,) = ex.drawings.values()
    assert drawing["src"] == "./my doc.assets/shot (1).png"
    assert "my doc.assets" not in ex.blocks[0]["content"]


def test_parenthesised_source_is_limited_in_length():
    limit = _PAREN_SOURCE_MAX_CHARS
    fits = "a(1)" + "x" * (limit - 4)
    assert len(fits) == limit
    assert _replace_inline_images(f"![x]({fits})", lambda src: f"<{src}>") == (
        f"<{fits}>"
    )
    assert _replace_inline_images(f'![x]({fits} "t")', lambda src: "<>") == "<>"
    too_long = f"![x]({fits}x)"
    assert _replace_inline_images(too_long, lambda src: "<>") == too_long
    too_long = f'![x]({fits}x "t")'
    assert _replace_inline_images(too_long, lambda src: "<>") == too_long
    # The limit is on sources WITH parentheses only: a base64 data URL has
    # none and may be any length.
    data_url = "data:image/png;base64," + "QUJD" * limit
    assert _replace_inline_images(f"![x]({data_url})", lambda src: "<>") == "<>"


@pytest.mark.parametrize(
    "line",
    [
        # Whole-line parenthesis index once grew 230 MiB for 1 MB suffix.
        "![x](a.png) " + ")" * 1_000_000,
        "![x](a(1).png) " + ")" * 1_000_000,
        # Shorter under tracemalloc: whole-line index still takes ~4x bound.
        "![x](" + "(" * 250_000,
        "![x](a(" + "()" * 125_000 + ")",
        "![x](a b" + ".png)" * 200_000,
        "![x](a (b" + ".png) " * 200_000,
        # Code spans are skipped, never indexed.
        "![x](a.png) " + "`a" * 100_000,
        "![x](a.png) " + "``a`" * 50_000,
        "![x](a.png) " + "".join("`" * k + " " for k in range(1, 1_400)),
    ],
)
def test_inline_image_scanner_memory_stays_bounded(line):
    tracemalloc.start()
    try:
        _replace_inline_images(line, lambda src: "")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # The output is a copy of (most of) the line; allow that and a fixed
    # overhead, far below what any per-character index would take.
    assert peak < 2 * len(line) + 4_000_000


def _repeat(prefix: str, unit: str, suffix: str) -> Callable[[int], str]:
    return lambda n: prefix + unit * n + suffix


@pytest.mark.parametrize(
    ("build_line", "small_n", "large_n", "paren_max_chars", "verify_unchanged"),
    [
        # Parenthesis matching for malformed candidates (small window = 64).
        pytest.param(_repeat("", "![](a(", ""), 1_024, 8_192, 64, False, id="paren-open"),
        pytest.param(_repeat("", "![](a)", ""), 1_024, 8_192, 64, False, id="paren-close"),
        pytest.param(
            _repeat("", "![](a(b)", ""), 1_024, 8_192, 64, False, id="paren-nested"
        ),
        pytest.param(
            _repeat("", "![]((", ""), 1_024, 8_192, 64, False, id="paren-bare-open"
        ),
        pytest.param(
            _repeat("", "![](a (", ""), 1_024, 8_192, 64, False, id="paren-space-open"
        ),
        # Parenthesis window rebuilds (every candidate beyond previous window).
        pytest.param(
            _repeat("", "![](a(" + "x" * 5_000, ""),
            32,
            256,
            None,
            False,
            id="paren-window-rebuilds",
        ),
        # Lenient retry for malformed candidates sharing a suffix.
        pytest.param(
            _repeat("", "![](a b", "](a ( b.png)"),
            1_024,
            8_192,
            None,
            False,
            id="lenient-no-ext",
        ),
        pytest.param(
            _repeat("", "![](a b.png", "](a ( b.png)"),
            1_024,
            8_192,
            None,
            False,
            id="lenient-unclosed",
        ),
        pytest.param(
            _repeat("", "![](a (b.png) ", "](a ( b.png)"),
            1_024,
            8_192,
            None,
            False,
            id="lenient-wrong-depth",
        ),
        pytest.param(
            _repeat("", '![](a b.png "', "](a ( b.png)"),
            1_024,
            8_192,
            None,
            False,
            id="lenient-dangling-title",
        ),
        pytest.param(
            _repeat("", "![", "](a ( b.png)"),
            1_024,
            8_192,
            None,
            False,
            id="lenient-nested-alt",
        ),
        # Code span and escape checks.
        pytest.param(
            _repeat("![x](a.png) ", "`a", ""), 1_024, 8_192, None, False, id="code-closed"
        ),
        pytest.param(
            _repeat("![x](a.png) ", "``a`", ""),
            1_024,
            8_192,
            None,
            False,
            id="code-unclosed-opener",
        ),
        pytest.param(
            _repeat("![x](a.png) ", "\\``a", ""),
            1_024,
            8_192,
            None,
            False,
            id="code-escaped-opener",
        ),
        pytest.param(
            _repeat("![x](a.png) ", "![", "`](x.png)`"),
            1_024,
            8_192,
            None,
            False,
            id="code-alt-ends-in-span",
        ),
        pytest.param(
            _repeat("![x](a.png) ", "![ `a` ", "`](x.png)`"),
            1_024,
            8_192,
            None,
            False,
            id="code-alt-after-closed-span",
        ),
        pytest.param(
            _repeat("![x](a.png) ", "![", "\\](x.png)"),
            1_024,
            8_192,
            None,
            False,
            id="code-escaped-alt-end",
        ),
        # Shared alt end behind a long odd backslash run.
        pytest.param(
            lambda n: "![" * n + "\\" * (2 * n + 1) + "](x.png)",
            1_024,
            8_192,
            None,
            False,
            id="shared-alt-end-backslashes",
        ),
        # Bare malformed candidate flood.
        pytest.param(
            _repeat("", "![](", ""),
            1_024,
            8_192,
            None,
            True,
            id="malformed-candidate-flood",
        ),
    ],
)
def test_inline_image_scanner_scales_linearly(
    build_line: Callable[[int], str],
    small_n: int,
    large_n: int,
    paren_max_chars: int | None,
    verify_unchanged: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    if paren_max_chars is not None:
        monkeypatch.setattr(extract_module, "_PAREN_SOURCE_MAX_CHARS", paren_max_chars)
    small = build_line(small_n)
    large = build_line(large_n)
    if verify_unchanged:
        assert _replace_inline_images(small, lambda src: src) == small
        assert _replace_inline_images(large, lambda src: src) == large
    # Eight times more input must not become a quadratic (~64x) workload.
    assert _scan_time_ratio(small, large) < 16


class _CopyCountingLine(str):
    """A line that counts the characters its slices copy."""

    copied = 0

    def __getitem__(self, key):
        if isinstance(key, slice):
            self.copied += len(range(*key.indices(len(self))))
        return str.__getitem__(self, key)


@pytest.mark.parametrize("alt_end", ["\\]", "`]", " `a` `]"])
@pytest.mark.parametrize("source", ["<" + "a" * 4_096 + ">", "a" * 4_096])
def test_rejected_alt_text_does_not_copy_the_source_per_candidate(alt_end, source):
    # Nested candidates share one literal ``]`` and a long source. Matching
    # the source before rejecting copied it once per candidate.
    line = _CopyCountingLine("![" * 2_048 + alt_end + "(" + source + ") `")
    assert _replace_inline_images(line, lambda src: "<>") == line
    assert line.copied <= 2 * len(line)


def test_inline_image_scanner_matches_reference_grammar_for_short_inputs():
    # The first case previously exposed a cache query that fell back to an
    # earlier start after a title search. The generated short corpora protect
    # the title, alternate-source and parenthesis grammar without a timing
    # dependency.
    lines = (
        'srcb"bb![a]( ![a]( " "title")<![a](>',
        '![a](src "title") ![b](other "caption")',
        '![a](<a b> "title") ![b](<c>d)',
        '![a](x(y)z) ![b]((x) "t") ![c](a(b c)',
    )
    pieces = ("![a](", "![", "]", "(", ")", "<", ">", " ", '"', "title", "src")
    paren_pieces = ("![a](", "(", ")", " ", "x", '"')
    corpus = itertools.chain(
        lines,
        ("".join(parts) for parts in itertools.product(pieces, repeat=4)),
        ("".join(parts) for parts in itertools.product(paren_pieces, repeat=6)),
    )
    for line in corpus:
        assert _replace_inline_images(
            line, lambda src: f"<{src}>"
        ) == _reference_replace_inline_images(line), line


def test_code_spans_and_escapes_match_reference_for_short_inputs():
    # The cached closing-run search once moved its per-length memory back to
    # an earlier run, which hid a later closer from every following opener.
    pieces = ("`", "``", "\\", "![a](", "![", "]", "(", ")", "x", " ")
    corpus = itertools.chain(
        (
            "``x `a` `b ![i](i) ` `",
            "``u `a` `in ``` t` ![r](r) ```",
        ),
        ("".join(parts) for parts in itertools.product(pieces, repeat=5)),
    )
    for line in corpus:
        assert _replace_inline_images(
            line, lambda src: f"<{src}>"
        ) == _reference_replace_inline_images(line), line


def test_code_spans_and_escapes_match_reference_for_random_lines():
    pieces = ("`", "``", "```", "\\", "![a](b)", "![", "](c)", "]", " ", "x")
    rng = random.Random(4133)
    for _ in range(20_000):
        line = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 24)))
        assert _replace_inline_images(
            line, lambda src: f"<{src}>"
        ) == _reference_replace_inline_images(line), line
