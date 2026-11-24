"""Tests for line-break and HTML-tag handling in ``normalize_extracted_info``.

The delimited extraction format is one record per line, so an LLM writes a line
break inside a value as ``<br>`` / ``<p>`` in any of their spellings, and
documents bring their own line break characters. What they become depends on
the field:

- single-line values (entity names, relation endpoints, types, keywords): tags
  are removed and every run of whitespace collapses into one space, after the
  rules that produce the value. A name is an entity's identity in the graph and
  a rebuild re-parses cached LLM output with the current rules, so a rule
  change may only merge names that used to differ, never split names that used
  to be equal (see
  "Entity name normalization changes" in docs/design/PurgeRecoveryContract.md).
- free text (descriptions, ``multiline=True``): tags and line break characters
  become a line break, so the two lines are not glued into one word.
"""

from __future__ import annotations

import itertools
import re

import pytest

from lightrag.operate import (
    _handle_single_entity_extraction,
    _handle_single_relationship_extraction,
    _process_json_extraction_result,
)
from lightrag.utils import (
    normalize_entity_name,
    normalize_extracted_info,
    sanitize_and_normalize_extracted_text,
    sanitize_text_for_encoding,
)

pytestmark = pytest.mark.offline

LINE_BREAK_TAGS = ["<br>", "<br >", "<br/>", "<br />", "<br  />", "<BR />", "</br>"]
PARAGRAPH_TAGS = ["<p>", "<p >", "<p/>", "<p />", "<P />", "</p>"]
LINE_BREAK_CHARS = ["\n", "\r\n", "\r", "\x85", "\u2028", "\u2029"]


# --- single-line values: only merge, never split ----------------------------


@pytest.mark.parametrize("tag", LINE_BREAK_TAGS + PARAGRAPH_TAGS)
def test_every_tag_spelling_is_removed_from_a_name(tag):
    assert normalize_extracted_info(f"Acme{tag}Corp") == "AcmeCorp"


@pytest.mark.parametrize(
    "spelling",
    [
        "Acme<br>Corp",
        "Acme<br />Corp",
        "Acme<p />Corp",
        "Acme&lt;br /&gt;Corp",
        "Acme\x0bCorp",  # deleted by the sanitizer as a control character
        "Acme\x0cCorp",
    ],
)
def test_a_tag_or_control_break_does_not_split_a_name_from_its_glued_form(spelling):
    # A stored "AcmeCorp" may have been extracted from any of these spellings.
    # A rebuild re-parses them and keeps only the records whose name equals
    # the stored key, so they must all keep landing on it.
    assert normalize_entity_name(spelling) == normalize_entity_name("AcmeCorp")


@pytest.mark.parametrize("char", LINE_BREAK_CHARS + ["\t", "  ", " \n\t "])
def test_whitespace_and_line_breaks_collapse_into_one_space_in_a_name(char):
    name = normalize_entity_name(f"Acme{char}Corp")

    assert name == "Acme Corp"
    assert name.splitlines() == ["Acme Corp"]


def test_non_breaking_spaces_merge_with_ordinary_spaces_in_a_name():
    # The collapse runs after the non-breaking space conversion; before it,
    # the first spelling kept two spaces and split from the second.
    assert normalize_entity_name("Acme&nbsp;&nbsp;Corp") == "Acme Corp"
    assert normalize_entity_name("Acme\u00a0 Corp") == "Acme Corp"
    assert normalize_entity_name("Acme  Corp") == "Acme Corp"


@pytest.mark.parametrize(
    "spelling",
    [
        "Acme&nbsp;&nbsp;Corp",
        "Acme \u2028 Corp",
        "<p> Acme\t\tCorp </p>",
        "北京 \n 大学",
        "Acme\u202f\u202fCorp",
    ],
)
def test_normalizing_a_name_twice_changes_no_whitespace(spelling):
    once = normalize_entity_name(spelling)

    assert normalize_entity_name(once) == once


def test_a_break_between_chinese_characters_leaves_no_space_in_a_name():
    assert normalize_entity_name("北京<br />大学") == "北京大学"
    assert normalize_entity_name("北京\u2028大学") == "北京大学"
    assert normalize_entity_name("北京\x0b大学") == "北京大学"


def test_outer_quotes_next_to_a_paragraph_tag_are_still_removed():
    assert normalize_entity_name('<p>"Acme Corp"</p>') == "Acme Corp"


def test_narrow_non_breaking_space_after_a_digit_is_kept():
    assert normalize_entity_name("10\u202fkm") == "10\u202fkm"


# --- free text ---------------------------------------------------------------


@pytest.mark.parametrize("tag", LINE_BREAK_TAGS)
def test_every_line_break_tag_spelling_becomes_a_line_break_in_free_text(tag):
    assert (
        normalize_extracted_info(f"first line{tag}second line", multiline=True)
        == "first line\nsecond line"
    )


@pytest.mark.parametrize("char", LINE_BREAK_CHARS + ["\x0b", "\x0c"])
def test_every_line_break_character_becomes_a_line_break_in_free_text(char):
    text = f"first line{char}second line"

    assert (
        sanitize_and_normalize_extracted_text(text, multiline=True)
        == "first line\nsecond line"
    )
    assert normalize_extracted_info(text, multiline=True) == "first line\nsecond line"


def test_escaped_line_break_tag_becomes_a_line_break_too():
    # sanitize_text_for_encoding unescapes entities before the tag cleanup runs
    assert (
        sanitize_and_normalize_extracted_text(
            "first line&lt;br /&gt;second line", multiline=True
        )
        == "first line\nsecond line"
    )


def test_paragraphs_are_separated_by_one_blank_line_at_most():
    assert (
        normalize_extracted_info("<p>first</p><p>second</p>", multiline=True)
        == "first\n\nsecond"
    )
    assert (
        normalize_extracted_info("first<br><br><br><br>second", multiline=True)
        == "first\n\nsecond"
    )


def test_spaces_around_a_line_break_are_dropped():
    assert (
        normalize_extracted_info("first line <br />\t second line", multiline=True)
        == "first line\nsecond line"
    )


def test_a_line_break_between_chinese_characters_stays_in_free_text():
    assert normalize_extracted_info("第一行<br>第二行", multiline=True) == (
        "第一行\n第二行"
    )
    assert (
        sanitize_and_normalize_extracted_text("第一行\x0b第二行", multiline=True)
        == "第一行\n第二行"
    )
    # horizontal spaces between Chinese characters are still removed
    assert normalize_extracted_info("中文 字符", multiline=True) == "中文字符"


# --- both modes --------------------------------------------------------------


@pytest.mark.parametrize("multiline", [False, True])
@pytest.mark.parametrize("text", ["<pre>code</pre>", "<brand>", "<path />"])
def test_tags_that_only_start_like_p_or_br_are_kept(text, multiline):
    assert normalize_extracted_info(text, multiline=multiline) == text


# --- through the extraction parsers ------------------------------------------


def test_delimited_records_apply_the_rule_per_field():
    entity = _handle_single_entity_extraction(
        ["entity", "Acme<br />Corp", "organization", "Makes widgets.<br />Since 1990."],
        "chunk-1",
        1,
        "doc.txt",
    )
    relation = _handle_single_relationship_extraction(
        [
            "relation",
            "Acme  Corp",
            "Globex",
            "supply chain",
            "Acme supplies Globex.<br />Since 2001.",
        ],
        "chunk-1",
        1,
        "doc.txt",
    )

    assert entity["entity_name"] == "AcmeCorp"
    assert entity["description"] == "Makes widgets.\nSince 1990."
    assert relation["src_id"] == "Acme Corp"
    assert relation["description"] == "Acme supplies Globex.\nSince 2001."


@pytest.mark.asyncio
async def test_json_results_put_an_entity_and_its_relation_on_one_node():
    raw = (
        '{"entities":[{"name":"Acme&nbsp;&nbsp;Corp","type":"organization",'
        '"description":"Makes widgets.<br />Since 1990."}],'
        '"relationships":[{"source":"Acme\\nCorp","target":"Globex",'
        '"keywords":"supply chain","description":"Supplies Globex.\\u2028Since 2001."}]}'
    )

    nodes, edges = await _process_json_extraction_result(
        raw, chunk_key="chunk-1", timestamp=1, file_path="doc.txt"
    )

    assert list(nodes) == ["Acme Corp"]
    assert nodes["Acme Corp"][0]["description"] == "Makes widgets.\nSince 1990."
    assert list(edges) == [("Acme Corp", "Globex")]
    assert edges[("Acme Corp", "Globex")][0]["description"] == (
        "Supplies Globex.\nSince 2001."
    )


# --- quoted Chinese second space pass (only merge, never split) --------------


@pytest.mark.parametrize(
    "spelling, expected",
    [
        ("Apple 苹果", "Apple苹果"),
        ('Apple "苹果"', "Apple苹果"),
        ("Apple '苹果'", "Apple苹果"),
        ("Apple “苹果”", "Apple苹果"),
        ("Apple ‘苹果’", "Apple苹果"),
        ('"苹果" Apple', "苹果Apple"),
        ("'苹果' Apple", "苹果Apple"),
        ("“苹果” Apple", "苹果Apple"),
        ("苹果 公司", "苹果公司"),
        ('"苹果" 公司', "苹果公司"),
        ('苹果 "公司"', "苹果公司"),
        ("‘苹果’ 公司", "苹果公司"),
    ],
)
def test_a_quoted_spelling_lands_on_the_unquoted_name(spelling, expected):
    name = normalize_entity_name(spelling)

    assert name == expected
    assert normalize_entity_name(name) == name


def test_a_space_next_to_a_quote_that_stays_is_kept():
    # No Chinese character touches these quotes, so they are not removed and
    # the space next to them is still not one the space rules match.
    assert normalize_entity_name('Apple "Inc" 苹果') == 'Apple "Inc" 苹果'
    assert normalize_entity_name("Apple Inc") == "Apple Inc"


def _name_before_the_second_pass(raw: str) -> str:
    """``normalize_entity_name`` as it was before the second pass. Frozen.

    A copy of the single-line path of ``normalize_extracted_info`` with
    ``remove_inner_quotes=True``, taken at 75d6a0c. It is the "old result" of
    the merge-only rule, so it must not follow later changes to the function.
    """
    name = sanitize_text_for_encoding(raw)
    if not name:
        return ""

    name = re.sub(r"</p\s*>|<p\s*/?>", "", name, flags=re.IGNORECASE)
    name = re.sub(r"</br\s*>|<br\s*/?>", "", name, flags=re.IGNORECASE)
    name = name.translate(
        str.maketrans(
            "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        )
    )
    name = name.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    name = name.replace("－", "-")
    name = name.replace("＋", "+")
    name = name.replace("／", "/")
    name = name.replace("＊", "*")
    name = name.replace("（", "(").replace("）", ")")
    name = name.replace("—", "-").replace("－", "-")
    name = name.replace("　", " ")

    name = re.sub(r"(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])", "", name)
    name = re.sub(
        r"(?<=[\u4e00-\u9fa5])\s+(?=[a-zA-Z0-9\(\)\[\]@#$%!&\*\-=+_])", "", name
    )
    name = re.sub(
        r"(?<=[a-zA-Z0-9\(\)\[\]@#$%!&\*\-=+_])\s+(?=[\u4e00-\u9fa5])", "", name
    )

    if len(name) >= 2:
        if name.startswith('"') and name.endswith('"'):
            inner_content = name[1:-1]
            if '"' not in inner_content:
                name = inner_content
        if name.startswith("'") and name.endswith("'"):
            inner_content = name[1:-1]
            if "'" not in inner_content:
                name = inner_content
        if name.startswith("“") and name.endswith("”"):
            inner_content = name[1:-1]
            if "“" not in inner_content and "”" not in inner_content:
                name = inner_content
        if name.startswith("‘") and name.endswith("’"):
            inner_content = name[1:-1]
            if "‘" not in inner_content and "’" not in inner_content:
                name = inner_content
        if name.startswith("《") and name.endswith("》"):
            inner_content = name[1:-1]
            if "《" not in inner_content and "》" not in inner_content:
                name = inner_content

    name = name.replace("“", "").replace("”", "").replace("‘", "").replace("’", "")
    name = re.sub(r"['\"]+(?=[\u4e00-\u9fa5])", "", name)
    name = re.sub(r"(?<=[\u4e00-\u9fa5])['\"]+", "", name)
    name = name.replace("\u00a0", " ")
    name = re.sub(r"(?<=[^\d])\u202F", " ", name)

    name = re.sub(r"[ \t\n\r\x85\u2028\u2029]+", " ", name)
    name = name.strip()

    if len(name) < 3 and re.match(r"^[0-9]+$", name):
        return ""
    if len(name) < 6 and "." in name and all(c.isdigit() or c == "." for c in name):
        return ""
    return name


_PIECES = ["A", "1", "苹", " ", "(", ".", "<br>", "\n", *"\"'“”‘’《》"]
_WHITESPACE_PIECES = ["A", "1", "苹", " ", "\t", "\u00a0", "\u202f", "\u3000", '"', "“"]


def _generated_names(pieces):
    for length in range(1, 5):
        for combination in itertools.product(pieces, repeat=length):
            yield "".join(combination)


@pytest.mark.parametrize(
    "pieces", [_PIECES, _WHITESPACE_PIECES], ids=["quotes", "whitespace"]
)
def test_the_second_pass_only_merges_names(pieces):
    new_name_by_old_name: dict[str, str] = {}
    for raw in _generated_names(pieces):
        old_name = _name_before_the_second_pass(raw)
        new_name = normalize_entity_name(raw)

        # Two spellings that shared a name must still share one.
        assert new_name_by_old_name.setdefault(old_name, new_name) == new_name, raw

    # ...and the pass did merge some: the check above is not vacuous.
    assert len(set(new_name_by_old_name.values())) < len(new_name_by_old_name)
