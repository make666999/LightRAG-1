"""Unit tests for OMML (Word equation) -> LaTeX conversion.

Locks in:
- function names spread across multiple OMML text runs (and empty/missing names),
- delimiter conversion: escaped curly braces, paired ``\\left``/``\\right``
  with ``.`` for one-sided delimiters, fixed-size brackets, position-based
  side selection, NFKC/CJK delimiter folding, matrix environments, and
  separator normalization/allowlist guards.
"""

from __future__ import annotations

import unicodedata
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

import pytest
from docx import Document
from docx.oxml import parse_xml

from lightrag.parser.docx.omml import convert_omml_to_latex
from lightrag.parser.docx.omml.ommlparser import (
    DELIMITER_MAP,
    FIXED_SIZE_DELIMITER_MAP,
)
from lightrag.parser.docx.parse_document import extract_docx_blocks

pytestmark = pytest.mark.offline

MATH_NS = 'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"'


# ---------------------------------------------------------------------------
# Function names across runs
# ---------------------------------------------------------------------------


def _function_xml(parts: tuple[str, ...], *, scripted: bool = False) -> str:
    runs = "".join(f"<m:r><m:t>{escape(part)}</m:t></m:r>" for part in parts)
    if scripted:
        runs = (
            f"<m:sSubSup><m:e>{runs}</m:e>"
            "<m:sub><m:r><m:t>1</m:t></m:r></m:sub>"
            "<m:sup><m:r><m:t>2</m:t></m:r></m:sup></m:sSubSup>"
        )
    return (
        f"<m:oMath {MATH_NS}><m:func><m:fName>{runs}</m:fName>"
        "<m:e><m:r><m:t>x</m:t></m:r></m:e></m:func></m:oMath>"
    )


@pytest.mark.parametrize(
    ("parts", "scripted", "expected"),
    [
        (("c", "os"), False, r"\cos_{}^{}{x}"),
        (("l", "im"), False, r"\lim\limits_{}^{}{x}"),
        (("foo", "bar"), False, r"{foobar}^{}_{}{x}"),
        (("cos",), False, r"\cos_{}^{}{x}"),
        (("c", "os"), True, r"\cos_{1}^{2}{x}"),
        # A whitespace-only run parses to " "; it must not keep the name out
        # of FUNCTION_MAP or the limit forms.
        (("cos", " "), False, r"\cos_{}^{}{x}"),
        ((" ", "lim"), False, r"\lim\limits_{}^{}{x}"),
    ],
)
def test_function_name_keeps_all_runs(parts, scripted, expected):
    assert (
        convert_omml_to_latex(ET.fromstring(_function_xml(parts, scripted=scripted)))
        == expected
    )


def _raw_function_xml(fname: str | None) -> str:
    name = "" if fname is None else f"<m:fName>{fname}</m:fName>"
    return (
        f"<m:oMath {MATH_NS}><m:func>{name}"
        "<m:e><m:r><m:t>x</m:t></m:r></m:e></m:func></m:oMath>"
    )


@pytest.mark.parametrize(
    ("fname", "expected"),
    [
        # A name wrapped in any structure the parser knows is read through it.
        ("<m:box><m:e><m:r><m:t>cos</m:t></m:r></m:e></m:box>", r"\cos_{}^{}{x}"),
        # A name that yields no text is never replaced by an invented "sin".
        ("<m:r><m:t/></m:r>", r"{}^{}_{}{x}"),
        ("<m:limUpp><m:e><m:r><m:t>lim</m:t></m:r></m:e></m:limUpp>", r"{}^{}_{}{x}"),
        (None, r"{}^{}_{}{x}"),
    ],
)
def test_function_name_is_read_not_invented(fname, expected):
    assert convert_omml_to_latex(ET.fromstring(_raw_function_xml(fname))) == expected


@pytest.mark.parametrize("parts", [("c", "os")])
def test_split_function_name_survives_docx_extraction(tmp_path, parts):
    doc = Document()
    paragraph = doc.add_paragraph("Before ")
    paragraph._p.append(parse_xml(_function_xml(parts)))
    paragraph.add_run(" after")
    path = tmp_path / "function.docx"
    doc.save(path)

    blocks = extract_docx_blocks(str(path))

    assert len(blocks) == 1
    assert r"Before <equation>\cos_{}^{}{x}</equation> after" in blocks[0]["content"]


# ---------------------------------------------------------------------------
# Delimiters, matrices, and separators
# ---------------------------------------------------------------------------


def _delimited(beg: str, end: str) -> ET.Element:
    """Build an ``m:oMath`` holding a single delimited ``x``."""
    return ET.fromstring(
        f"<m:oMath {MATH_NS}><m:d>"
        f'<m:dPr><m:begChr m:val="{beg}"/><m:endChr m:val="{end}"/></m:dPr>'
        "<m:e><m:r><m:t>x</m:t></m:r></m:e></m:d></m:oMath>"
    )


def _matrix(beg: str, end: str) -> ET.Element:
    """Build an ``m:oMath`` holding a one-cell matrix inside a delimiter."""
    return ET.fromstring(
        f"<m:oMath {MATH_NS}><m:d>"
        f'<m:dPr><m:begChr m:val="{beg}"/><m:endChr m:val="{end}"/></m:dPr>'
        "<m:e><m:m><m:mr><m:e><m:r><m:t>a</m:t></m:r></m:e></m:mr></m:m></m:e>"
        "</m:d></m:oMath>"
    )


def _separated(sep: str) -> ET.Element:
    """Build an ``m:oMath`` holding two elements split by ``sep``."""
    return ET.fromstring(
        f"<m:oMath {MATH_NS}><m:d>"
        f'<m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/><m:sepChr m:val="{sep}"/></m:dPr>'
        "<m:e><m:r><m:t>x</m:t></m:r></m:e>"
        "<m:e><m:r><m:t>y</m:t></m:r></m:e>"
        "</m:d></m:oMath>"
    )


@pytest.mark.parametrize(
    ("beg", "end", "expected"),
    [
        # Curly braces escaped; standard scalable delimiters preserved.
        ("{", "}", r"\left\{ x \right\}"),
        ("(", ")", r"\left( x \right)"),
        ("[", "]", r"\left[ x \right]"),
        ("⌊", "⌋", r"\left\lfloor x \right\rfloor"),
        # One-sided delimiters paired with ``.``.
        ("{", "", r"\left\{ x \right."),
        ("", "}", r"\left. x \right\}"),
        # Fixed-size double brackets only pair opposite a scalable side.
        ("⟦", "⟧", r"[\![ x ]\!]"),
        ("⟦", "", r"[\![ x"),
        ("", "⟧", r"x ]\!]"),
        ("⟦", ")", r"\left. [\![ x \right)"),
        ("(", "⟧", r"\left( x ]\!] \right."),
        # Side follows position, not character (half-open intervals / mirror).
        ("[", "[", r"\left[ x \right["),
        ("(", "(", r"\left( x \right("),
        ("{", "{", r"\left\{ x \right\{"),
        ("⌈", "⌊", r"\left\lceil x \right\lfloor"),
        ("]", "[", r"\left] x \right["),
        (")", ")", r"\left) x \right)"),
        ("}", "}", r"\left\} x \right\}"),
        ("⌋", "⌉", r"\left\rfloor x \right\rceil"),
        ("", "(", r"\left. x \right("),
        (")", "", r"\left) x \right."),
        ("", "{", r"\left. x \right\{"),
        ("}", "", r"\left\} x \right."),
        # Fullwidth delimiters fold onto ASCII counterparts via NFKC.
        ("｛", "｝", r"\left\{ x \right\}"),
        ("｜", "｜", r"\left| x \right|"),
        # Both NFC (U+3008/U+3009) and U+2329/U+232A angle brackets reach map.
        ("\u3008", "\u3009", r"\left\langle x \right\rangle"),
        ("\u2329", "\u232a", r"\left\langle x \right\rangle"),
        # CJK brackets map to nearest scalable delimiter or fixed-size sequence.
        ("【", "】", r"\left[ x \right]"),
        ("〔", "〕", r"\left[ x \right]"),
        ("《", "》", r"\langle\!\langle x \rangle\!\rangle"),
        # Unknown LaTeX-special delimiter characters fall back to parentheses.
        ("＆", "％", r"\left( x \right)"),
    ],
)
def test_delimiter_conversion(beg: str, end: str, expected: str):
    assert convert_omml_to_latex(_delimited(beg, end)) == expected


@pytest.mark.parametrize(
    ("beg", "end", "expected"),
    [
        # Matrix flavour follows the normalized delimiter.
        ("（", "）", r"\begin{pmatrix} a \end{pmatrix}"),
        ("｜", "｜", r"\begin{vmatrix} a \end{vmatrix}"),
        ("‖", "‖", r"\begin{Vmatrix} a \end{Vmatrix}"),
        ("[", "]", r"\begin{bmatrix} a \end{bmatrix}"),
        ("【", "】", r"\begin{bmatrix} a \end{bmatrix}"),
        # Matrix without a named environment keeps its own delimiters.
        (
            "《",
            "》",
            r"\langle\!\langle \begin{matrix} a \end{matrix} \rangle\!\rangle",
        ),
        ("〈", "〉", r"\left\langle \begin{matrix} a \end{matrix} \right\rangle"),
        ("[", "", r"\left[ \begin{matrix} a \end{matrix} \right."),
    ],
)
def test_matrix_delimiter_conversion(beg: str, end: str, expected: str):
    assert convert_omml_to_latex(_matrix(beg, end)) == expected


@pytest.mark.parametrize(
    ("sep", "expected"),
    [("｜", "|"), ("，", ","), ("；", ";"), ("：", ":"), ("．", "."), ("／", "/")],
)
def test_fullwidth_separator_is_normalized(sep, expected):
    assert convert_omml_to_latex(_separated(sep)) == f"\\left( x{expected}y \\right)"


@pytest.mark.parametrize(
    "sep",
    [
        # LaTeX's own special characters.
        "＃",
        "＄",
        "％",
        "＆",
        "＼",
        "＾",
        "＿",
        "｛",
        "｝",
        "～",
        # Structural in math mode without being special.
        "＇",
        "｀",
        "＂",
        "Ｘ",
        "５",
    ],
)
def test_separator_keeps_the_character_word_wrote_unless_the_fold_is_plain(sep):
    assert convert_omml_to_latex(_separated(sep)) == f"\\left( x{sep}y \\right)"


def test_delimiter_map_keys_are_nfkc_normalized():
    keys = [*DELIMITER_MAP, *FIXED_SIZE_DELIMITER_MAP]
    assert [k for k in keys if unicodedata.normalize("NFKC", k) != k] == []
