"""G9 tests: TOC detection, content preservation (I1), I2/I3 checks,
canonicalization, and the 30% length gate."""

from __future__ import annotations

import pytest

from lightrag.parser.docx.parse_document import ParagraphRecord
from lightrag.parser.docx.smart_heading.guardrails import (
    TOC_ELLIPSIS,
    TocOutputPlan,
    canonicalize_paragraph_text,
    detect_toc_records,
    plan_toc_output,
    shadow_baseline_diff,
    smart_output_length_ok,
    toc_audit_entries,
    verify_anchor_semantics,
    verify_baseline_heading_retention,
    verify_content_preservation,
)
from lightrag.parser.docx.smart_heading.heading_flow import HeadingDecision
from tests.parser.docx._smart_heading_helpers import make_para as _make_para

pytestmark = pytest.mark.offline


def _para(text: str, **kw) -> ParagraphRecord:
    return _make_para(text, size=None, **kw)


def _p(text: str, size: float = 10.5) -> ParagraphRecord:
    return _make_para(text, size=size)


# ---------------------------------------------------------------------------
# canonicalization (G9-3 normalization side)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b",
    [
        ("## 第一章 绪论", "第一章绪论"),  # markdown prefix + spaces
        ("  中 文  分 词 ", "中文分词"),  # ALL whitespace (CJK zero-width splits)
        ("# Heading Line", "HeadingLine"),
        ("plain\ttext", "plaintext"),
    ],
)
def test_canonicalization_pairs(a: str, b: str) -> None:
    assert canonicalize_paragraph_text(a) == canonicalize_paragraph_text(b)


def test_canonicalization_hash_only_strips_leading_hashes() -> None:
    assert canonicalize_paragraph_text("A # B") == "A#B"  # inner # kept


# ---------------------------------------------------------------------------
# English trailing-period disambiguation (review D1)
# ---------------------------------------------------------------------------


class _FakeSent:
    """Minimal sentence span double for spaCy doc stub."""

    def __init__(self, start: int, end: int) -> None:
        self.start_char = start
        self.end_char = end


class _FakeDoc:
    """Minimal spaCy Doc double exposing ``sents``."""

    def __init__(self, sents) -> None:
        self.sents = sents


@pytest.mark.parametrize(
    ("text", "spans", "expected"),
    [
        # Phantom 'Next' becomes its own sentence -> terminal period.
        pytest.param(
            "This is a sentence.",
            [(0, 19), (20, 24)],
            True,
            id="terminal-dot",
        ),
        # Abbreviation absorbs 'Next' into one sentence -> not a terminator.
        pytest.param("See Fig.", [(0, 13)], False, id="abbreviation-dot"),
    ],
)
def test_ends_with_sentence_period(monkeypatch, text, spans, expected) -> None:
    from lightrag.parser.docx.smart_heading import nlp

    monkeypatch.setattr(
        nlp, "analyze", lambda t: _FakeDoc([_FakeSent(s, e) for s, e in spans])
    )
    assert nlp.ends_with_sentence_period(text) is expected


# ---------------------------------------------------------------------------
# strong-body: a numbering label must not read as a sentence break (I2 bug)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("3.1.1 桌面及办公设备运维服务", "桌面及办公设备运维服务"),
        ("3.1.1桌面及办公设备运维服务", "桌面及办公设备运维服务"),  # no space
        ("2.4.5 系统安全与容灾", "系统安全与容灾"),
        ("1. 概述", "概述"),
        ("第3章 项目背景", "项目背景"),
        ("（一）建设目标", "建设目标"),
        ("没有编号的标题", "没有编号的标题"),  # no numbering -> no-op
        ("第二章 \xa0列入条件和管理措施", "列入条件和管理措施"),  # NBSP separator
        ("第3章　项目背景", "项目背景"),  # full-width space separator
    ],
)
def test_strip_leading_numbering(text: str, expected: str) -> None:
    from lightrag.parser.docx.smart_heading.guardrails import _strip_leading_numbering

    assert _strip_leading_numbering(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2.3   投标单位廉洁自律承诺书", False),
        ("第一章\xa0总 则", False),
        ("3.2.2.2.2.4   禁限用工艺", False),
        ("3.2.2.1.1.1.3   4路AD_1～AD_4信号高速采集电路设计", False),
        ("第一阶段已经完成。下一阶段继续推进", True),
        ("第一阶段已经完成。” 下一阶段继续推进", True),
        ("Version 2.0 design overview", False),
        ("项目范围：", False),
    ],
)
def test_explicit_internal_sentence_boundary(text: str, expected: bool) -> None:
    from lightrag.parser.docx.smart_heading.guardrails import (
        has_explicit_internal_sentence_boundary,
    )

    assert has_explicit_internal_sentence_boundary(text) is expected


@pytest.mark.requires_spacy_models
def test_strong_body_spares_multi_level_numbered_heading() -> None:
    """Regression: zh spaCy splits a dotted multi-level number into two
    pseudo-sentences; strong_body_reason judges the label-stripped prose."""
    from lightrag.parser.docx.smart_heading.guardrails import strong_body_reason

    assert strong_body_reason("3.1.1 桌面及办公设备运维服务") is None
    assert strong_body_reason("3.1.2 政务信息化基础设施运维服务") is None
    assert strong_body_reason("2.4.5 系统安全与容灾") is None
    assert strong_body_reason("本节介绍运维范围。") == "strong_body_sentence_end"


@pytest.mark.requires_spacy_models
def test_strong_body_spares_heading_with_nbsp_separator() -> None:
    """Regression: an NBSP after the numbering label must be stripped and
    whitespace-only pseudo-sentences must not inflate sentence_count."""
    from lightrag.parser.docx.smart_heading import nlp
    from lightrag.parser.docx.smart_heading.guardrails import strong_body_reason

    assert strong_body_reason("第二章 \xa0列入条件和管理措施") is None
    assert strong_body_reason("第三章 \xa0列入和移出程序") is None
    assert nlp.sentence_count("\xa0列入条件和管理措施") == 1
    assert strong_body_reason("本节介绍列入条件。") == "strong_body_sentence_end"


@pytest.mark.requires_spacy_models
def test_strong_body_multi_sentence_needs_explicit_terminator() -> None:
    """Regression: zh spaCy mid-word pseudo-splits without a deterministic
    internal terminator must not mark short title phrases as strong body."""
    from lightrag.parser.docx.smart_heading import nlp
    from lightrag.parser.docx.smart_heading.guardrails import (
        has_explicit_internal_sentence_boundary,
        strong_body_reason,
    )

    lead_in = "广州市增城区“7.19”索菲亚定制家居项目"
    phrase = "投标单位廉洁自律承诺书"
    assert nlp.sentence_count(lead_in) >= 2
    assert not has_explicit_internal_sentence_boundary(lead_in)
    assert strong_body_reason(lead_in) is None
    assert strong_body_reason(phrase) is None
    assert (
        strong_body_reason("项目背景已经说明。后续要求继续执行")
        == "strong_body_multi_sentence"
    )


# ---------------------------------------------------------------------------
# TOC two-channel detection (G9-1 / G9-2 / G9-5)
# ---------------------------------------------------------------------------


def test_structural_toc_evidence() -> None:
    records = [
        _para("目录", is_toc_field=True),
        _para("第一章 绪论……3", is_toc_link=True),
        _para("普通正文段落。"),
    ]
    toc, _events = detect_toc_records(records)
    assert toc == {0, 1}
    entries = toc_audit_entries(records, toc)
    assert len(entries) == 2 and all("hash" in e for e in entries)


@pytest.mark.parametrize(
    ("records", "expected_toc"),
    [
        pytest.param(
            # G9-2: 4-line dot-leader run is TOC; isolated trailing line stays.
            [
                _para("第一章 绪论............3"),
                _para("第二章 方法............12"),
                _para("第三章 实验……25"),
                _para("第四章 结论·····31"),
                _para("正文开始。"),
                _para("附录 A............99"),
            ],
            {0, 1, 2, 3},
            id="four-line-run-spares-isolated-trailing",
        ),
        pytest.param(
            # 3+ dot-leader lines inside one soft-break paragraph qualify.
            [
                _para(
                    "第一章 绪论............3\n"
                    "第二章 方法............12\n"
                    "第三章 实验............25"
                ),
                _para("正文开始。"),
            ],
            {0},
            id="single-para-three-soft-break-lines",
        ),
        pytest.param(
            # 2-line soft-break paragraph is below the 3-line threshold.
            [
                _para("第一章 绪论............3\n第二章 方法............12"),
                _para("正文开始。"),
            ],
            set(),
            id="single-para-two-lines-below-threshold",
        ),
        pytest.param(
            # G9-5: body text adjacent to a TOC without leader shape stays.
            [
                _para("第一章 绪论............3"),
                _para("第二章 方法............12"),
                _para("第三章 实验............25"),
                _para("以下正文引用了第一章 绪论的内容。"),
            ],
            {0, 1, 2},
            id="adjacent-body-without-leader-spared",
        ),
    ],
)
def test_heuristic_toc_detection(
    records: list[ParagraphRecord], expected_toc: set[int]
) -> None:
    toc, _events = detect_toc_records(records)
    assert toc == expected_toc


# ---------------------------------------------------------------------------
# TOC third channel: plain (leaderless) numbered run under a 目次 title
# ---------------------------------------------------------------------------

_LONG_BODY = "这是用于分隔目录与正文的较长正文段落其加权字符数明显超过标题长度上限以确保它不会被误判为目录成员并作为一个天然的分隔项目描述若干"


@pytest.mark.parametrize(
    ("records", "expected_toc", "expected_event"),
    [
        pytest.param(
            # GB/T shape: leaderless numbered run under 目次, duplicated in body.
            [
                _p("目  次", 16.0),
                _p("1 范围"),
                _p("2 规范性引用文件"),
                _p("3 术语和定义"),
                _p(_LONG_BODY),
                _p("1 范围", 14.0),
                _p("本章规定了适用范围。"),
                _p("2 规范性引用文件", 14.0),
                _p("下列引用文件适用。"),
                _p("3 术语和定义", 14.0),
                _p("下列术语和定义适用。"),
            ],
            {1, 2, 3},
            {"rule": "toc_plain_numbered_run", "anchor": 0, "start": 1, "end": 3},
            id="plain-numbered-run",
        ),
        pytest.param(
            # Immediately followed by body's first heading at same 10.5pt size.
            [
                _p("目  次", 16.0),
                _p("1 范围"),
                _p("2 规范性引用文件"),
                _p("3 术语和定义"),
                _p("1 范围"),
                _p("本章规定了适用范围。"),
                _p("2 规范性引用文件"),
                _p("下列引用文件适用。"),
                _p("3 术语和定义"),
                _p("下列术语和定义适用。"),
            ],
            {1, 2, 3},
            {"rule": "toc_plain_numbered_run", "anchor": 0, "start": 1, "end": 3},
            id="truncates-before-same-size-body-heading",
        ),
        pytest.param(
            # Strictly larger body heading (16pt 前言) terminates the run.
            [
                _p("目  次", 16.0),
                _p("1 范围"),
                _p("2 规范性引用文件"),
                _p("3 术语和定义"),
                _p("前  言", 16.0),
                _p("前  言", 16.0),
                _p("1 范围"),
                _p("2 规范性引用文件"),
                _p("3 术语和定义"),
            ],
            {1, 2, 3},
            {"rule": "toc_plain_numbered_run", "anchor": 0, "start": 1, "end": 3},
            id="font-size-terminates-run",
        ),
        pytest.param(
            # One paragraph with 3 soft-break lines is counted line-by-line.
            [
                _p("目  次", 16.0),
                _p("1 范围\n2 规范性引用文件\n3 术语和定义"),
                _p(_LONG_BODY),
                _p("1 范围", 14.0),
                _p("2 规范性引用文件", 14.0),
                _p("3 术语和定义", 14.0),
            ],
            {1},
            {"rule": "toc_plain_numbered_run", "anchor": 0, "start": 1, "end": 1},
            id="soft-break-member-counts-lines",
        ),
        pytest.param(
            # Trailing page number is stripped before duplicate comparison.
            [
                _p("目  次", 16.0),
                _p("1 范围　3"),
                _p("2 规范性引用文件　5"),
                _p("3 术语和定义　8"),
                _p(_LONG_BODY),
                _p("1 范围", 14.0),
                _p("2 规范性引用文件", 14.0),
                _p("3 术语和定义", 14.0),
            ],
            {1, 2, 3},
            {"rule": "toc_plain_numbered_run", "anchor": 0, "start": 1, "end": 3},
            id="strips-trailing-pageno",
        ),
        pytest.param(
            # English Table of Contents matches body headings case-insensitively.
            [
                _p("Table of Contents", 16.0),
                _p("1 Scope"),
                _p("2 Normative References"),
                _p("3 Terms and Definitions"),
                _p(_LONG_BODY),
                _p("1 SCOPE", 14.0),
                _p("2 normative references", 14.0),
                _p("3 Terms and Definitions", 14.0),
            ],
            {1, 2, 3},
            {"rule": "toc_plain_numbered_run", "anchor": 0, "start": 1, "end": 3},
            id="english-contents-casefold",
        ),
    ],
)
def test_toc_third_channel_detects_plain_numbered_runs(
    records: list[ParagraphRecord],
    expected_toc: set[int],
    expected_event: dict,
) -> None:
    toc, events = detect_toc_records(records)
    assert toc == expected_toc
    assert 0 not in toc  # the TOC title itself stays a real heading
    assert events == [expected_event]


@pytest.mark.parametrize(
    "records",
    [
        pytest.param(
            # All-or-nothing: one run member lacks a body duplicate -> reject all.
            [
                _p("目  次", 16.0),
                _p("1 范围"),
                _p("2 规范性引用文件"),
                _p("3 术语和定义"),
                _p(_LONG_BODY),
                _p("1 范围", 14.0),
                _p("正文。"),
                _p("2 规范性引用文件", 14.0),
                _p("正文。"),
            ],
            id="all-or-nothing-missing-duplicate",
        ),
        pytest.param(
            # Mid-document Contents after 3 long body paras -> past front limit.
            [_p(_LONG_BODY) for _ in range(3)]
            + [
                _p("Contents", 16.0),
                _p("1 范围"),
                _p("2 规范性引用文件"),
                _p("3 术语和定义"),
                _p(_LONG_BODY),
                _p("1 范围", 14.0),
                _p("2 规范性引用文件", 14.0),
                _p("3 术语和定义", 14.0),
            ],
            id="anchor-past-front-part-limit",
        ),
        pytest.param(
            # Fewer than DOCX_SMART_TOC_MIN_LINES members -> not detected.
            [
                _p("目  次", 16.0),
                _p("1 范围"),
                _p("2 规范性引用文件"),
                _p(_LONG_BODY),
                _p("1 范围", 14.0),
                _p("2 规范性引用文件", 14.0),
            ],
            id="below-min-lines",
        ),
        pytest.param(
            # Numbered run with body duplicates but no 目次/Contents anchor.
            [
                _p("1 范围"),
                _p("2 规范性引用文件"),
                _p("3 术语和定义"),
                _p(_LONG_BODY),
                _p("1 范围", 14.0),
                _p("2 规范性引用文件", 14.0),
                _p("3 术语和定义", 14.0),
            ],
            id="missing-title-anchor",
        ),
    ],
)
def test_toc_third_channel_rejects_non_toc_runs(
    records: list[ParagraphRecord],
) -> None:
    toc, events = detect_toc_records(records)
    assert toc == set()
    assert events == []


# ---------------------------------------------------------------------------
# TOC retention implemented by plan_toc_output
# ---------------------------------------------------------------------------


def _toc(*texts: str) -> tuple[list[ParagraphRecord], set[int]]:
    recs = [_p(t) for t in texts]
    return recs, set(range(len(recs)))


@pytest.mark.parametrize(
    (
        "recs",
        "toc",
        "keep_lines",
        "expected_kept",
        "expected_anchor",
        "expected_counts",
        "expected_full_removed",
    ),
    [
        pytest.param(
            *_toc("第一章 绪论……3", "第二章 方法……12"),
            5,
            {0: "第一章 绪论……3", 1: "第二章 方法……12"},
            None,
            (2, 0),
            0,
            id="under-budget-no-ellipsis",
        ),
        pytest.param(
            *_toc("1", "2", "3", "4", "5"),
            5,
            {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"},
            None,
            (5, 0),
            0,
            id="exactly-at-budget-no-ellipsis",
        ),
        pytest.param(
            *_toc(*[f"第{i} 条……{i}" for i in range(8)]),
            5,
            {i: f"第{i} 条……{i}" for i in range(5)},
            4,
            (5, 3),
            3,
            id="over-budget-truncates-with-ellipsis",
        ),
        pytest.param(
            # Single paragraph with 8 soft-break lines, budget 5: per-line budget.
            *_toc("\n".join(f"第{i} 条……{i}" for i in range(8))),
            5,
            {0: "\n".join(f"第{i} 条……{i}" for i in range(5))},
            0,
            (5, 3),
            0,
            id="straddling-soft-break-paragraph",
        ),
        pytest.param(
            *_toc("第一章……3", "第二章……12", "第三章……25"),
            0,
            {},
            0,
            (0, 3),
            3,
            id="keep-zero-collapses-to-ellipsis",
        ),
        pytest.param(
            *_toc("第一章……3", "第二章……12"),
            -3,
            {},
            0,
            (0, 2),
            2,
            id="negative-budget-clamps-to-zero",
        ),
        pytest.param(
            # Two TOC runs separated by a body record share one global budget.
            [_p(t) for t in ("A", "B", "sep-body", "C", "D", "E")],
            {0, 1, 3, 4, 5},
            3,
            {0: "A", 1: "B", 3: "C"},
            3,
            (3, 2),
            2,
            id="non-contiguous-segments-single-anchor",
        ),
        pytest.param(
            # Whitespace-only record skipped entirely; surrounding spaces stripped.
            [_p("   "), _p("  第一章 绪论……3  "), _p("第二章……12")],
            {0, 1, 2},
            5,
            {1: "第一章 绪论……3", 2: "第二章……12"},
            None,
            (2, 0),
            0,
            id="skips-whitespace-and-strips",
        ),
        pytest.param(
            # Elided demoted_body_text tail counts as removed and raises ellipsis.
            [
                ParagraphRecord(
                    kind="para",
                    text="第一章 绪论……3",
                    demoted_body_text="尾部第一行\n尾部第二行",
                )
            ],
            {0},
            5,
            {0: "第一章 绪论……3"},
            0,
            (1, 2),
            0,
            id="counts-demoted-tail-as-removed",
        ),
    ],
)
def test_plan_toc_output_cases(
    recs: list[ParagraphRecord],
    toc: set[int],
    keep_lines: int,
    expected_kept: dict[int, str],
    expected_anchor: int | None,
    expected_counts: tuple[int, int],
    expected_full_removed: int,
) -> None:
    plan = plan_toc_output(recs, toc, keep_lines=keep_lines)
    assert isinstance(plan, TocOutputPlan)
    assert plan.kept_text == expected_kept
    assert plan.ellipsis_anchor == expected_anchor
    assert (plan.kept_lines, plan.removed_lines) == expected_counts
    assert plan.fully_removed_records == expected_full_removed


# ---------------------------------------------------------------------------
# I1 content preservation (G9-3)
# ---------------------------------------------------------------------------


def _blocks(*contents: str) -> list[dict]:
    return [{"content": c, "heading": "", "level": 1} for c in contents]


@pytest.mark.parametrize(
    ("records", "blocks", "kwargs"),
    [
        pytest.param(
            [_para("第一章 绪论"), _para("正文第一段。"), _para("正文第二段。")],
            _blocks("# 第一章 绪论\n正文第一段。", "正文第二段。"),
            {},
            id="faithful-output",
        ),
        pytest.param(
            # Merged headings + use_raw_text rebuild pass ordered concatenation.
            [
                _para("中华人民共和国"),
                _para("某某管理办法"),
                ParagraphRecord(
                    kind="para",
                    text="超长标题首行",
                    demoted_body_text="被拆出的余部内容",
                ),
            ],
            _blocks("# 中华人民共和国某某管理办法", "超长标题首行被拆出的余部内容"),
            {},
            id="heading-merge-and-oversize-rebuild",
        ),
        pytest.param(
            # A13 positive: consecutive source pieces composing merged line pass.
            [_para("第一部分"), _para("总体要求"), _para("独立正文段。")],
            _blocks("# 第一部分总体要求", "独立正文段。"),
            {},
            id="ordered-concatenation-merge",
        ),
        pytest.param(
            # Review C1: soft break inside body paragraph splits symmetrically.
            [_para("标题甲"), _para("正文第一行\n正文第二行")],
            _blocks("# 标题甲\n正文第一行\n正文第二行"),
            {},
            id="body-paragraph-soft-break",
        ),
        pytest.param(
            # Review C1: demoted_body_text carrying soft breaks passes I1.
            [
                ParagraphRecord(
                    kind="para",
                    text="超长标题首行",
                    demoted_body_text="余部第一行\n余部第二行",
                )
            ],
            _blocks("超长标题首行\n余部第一行\n余部第二行"),
            {},
            id="demoted-body-text-soft-break",
        ),
        pytest.param(
            # Paired control: injected TOC copy subtracted, body copy satisfies I1.
            [_para("第一章 绪论", is_toc_link=True), _para("第一章 绪论"), _para("正文段落。")],
            _blocks("第一章 绪论\n# 第一章 绪论\n正文段落。"),
            {"toc_indices": {0}, "ignored_output_texts": ["第一章 绪论"]},
            id="injected-toc-copy-with-body-present",
        ),
    ],
)
def test_i1_preservation_passing_cases(
    records: list[ParagraphRecord], blocks: list[dict], kwargs: dict
) -> None:
    assert verify_content_preservation(records, blocks, **kwargs) == []


@pytest.mark.parametrize(
    ("records", "blocks", "kwargs", "expected_missing_text"),
    [
        pytest.param(
            [_para("第一章 绪论"), _para("正文第一段。"), _para("被丢失的段落。")],
            _blocks("# 第一章 绪论\n正文第一段。"),
            {},
            "被丢失的段落。",
            id="injected-paragraph-loss",
        ),
        pytest.param(
            # A13: lost short paragraph cannot hide as substring of merged line.
            [_para("第一部分"), _para("总体要求"), _para("要求")],
            _blocks("# 第一部分总体要求"),
            {},
            "要求",
            id="concatenation-rejects-substring-hiding",
        ),
        pytest.param(
            # Retained TOC copy subtracted first, surfacing lost body heading.
            [_para("第一章 绪论", is_toc_link=True), _para("第一章 绪论"), _para("正文段落。")],
            _blocks("第一章 绪论\n正文段落。"),
            {"toc_indices": {0}, "ignored_output_texts": ["第一章 绪论"]},
            "第一章 绪论",
            id="subtracts-injected-toc-copy-reveals-body-loss",
        ),
    ],
)
def test_i1_preservation_detects_missing_content(
    records: list[ParagraphRecord],
    blocks: list[dict],
    kwargs: dict,
    expected_missing_text: str,
) -> None:
    missing = verify_content_preservation(records, blocks, **kwargs)
    assert missing == [canonicalize_paragraph_text(expected_missing_text)]


def test_i1_whitelists_toc_indices_only() -> None:
    records = [
        _para("第一章……3", is_toc_link=True),
        _para("正文段落。"),
    ]
    blocks = _blocks("正文段落。")
    assert verify_content_preservation(records, blocks, toc_indices={0}) == []
    assert verify_content_preservation(records, blocks) != []


def test_i1_subtracts_ellipsis_without_masking_a_source_ellipsis() -> None:
    """The injected ellipsis is subtracted; a genuine body '……' line still
    needs its own output copy."""
    records = [
        _para("目录条目……3", is_toc_link=True),
        _para(TOC_ELLIPSIS),
    ]
    assert verify_content_preservation(
        records,
        _blocks(TOC_ELLIPSIS),
        toc_indices={0},
        ignored_output_texts=[TOC_ELLIPSIS],
    ) == [canonicalize_paragraph_text(TOC_ELLIPSIS)]
    assert (
        verify_content_preservation(
            records,
            _blocks(f"{TOC_ELLIPSIS}\n{TOC_ELLIPSIS}"),
            toc_indices={0},
            ignored_output_texts=[TOC_ELLIPSIS],
        )
        == []
    )


# ---------------------------------------------------------------------------
# I2 / I3
# ---------------------------------------------------------------------------


def test_i2_outline_paragraph_must_stay_or_be_rule_tagged() -> None:
    records = [
        _para("样式标题甲", outline_level=0),
        _para("样式标题乙", outline_level=0),
        _para("样式标题丙", outline_level=0),
    ]
    kept = HeadingDecision(record_index=0, text="样式标题甲", is_heading=True, level=1)
    demoted = HeadingDecision(record_index=1, text="样式标题乙", is_heading=False)
    demoted.note("strong_body_demoted")
    silently_dropped = HeadingDecision(
        record_index=2, text="样式标题丙", is_heading=False
    )
    assert verify_baseline_heading_retention(
        records, [kept, demoted, silently_dropped]
    ) == [2]


@pytest.mark.parametrize(
    ("decisions", "expected_violations"),
    [
        pytest.param(
            # Level 0 is reserved for title-block roots.
            [
                HeadingDecision(
                    record_index=0,
                    text="主标题",
                    is_heading=True,
                    is_title_block=True,
                    level=0,
                ),
                HeadingDecision(
                    record_index=1, text="普通标题", is_heading=True, level=0
                ),
            ],
            [1],
            id="non-title-block-level-zero",
        ),
        pytest.param(
            # Non-numbered anchored heading must match outline_level + 1.
            [
                HeadingDecision(
                    record_index=0,
                    text="锚定标题",
                    is_heading=True,
                    level=3,
                    outline_level=2,
                    anchored=True,
                ),
                HeadingDecision(
                    record_index=1,
                    text="漂移标题",
                    is_heading=True,
                    level=5,
                    outline_level=2,
                    anchored=True,
                ),
            ],
            [1],
            id="non-numbered-anchor-drift",
        ),
    ],
)
def test_i3_anchor_semantics_violations(
    decisions: list[HeadingDecision], expected_violations: list[int]
) -> None:
    assert verify_anchor_semantics(decisions) == expected_violations


# ---------------------------------------------------------------------------
# 30% gate + shadow diff (G9-4)
# ---------------------------------------------------------------------------


def test_length_gate_triggers_below_30_percent() -> None:
    baseline = _blocks("x" * 1000)
    assert smart_output_length_ok(_blocks("x" * 300), baseline)
    assert not smart_output_length_ok(_blocks("x" * 299), baseline)


def test_shadow_diff_summary() -> None:
    smart = [
        {"content": "abc", "level": 0},
        {"content": "def", "level": 2},
    ]
    base = [{"content": "abcdef", "level": 1}]
    diff = shadow_baseline_diff(smart, base)
    assert diff["heading_count_smart"] == 2
    assert diff["heading_count_baseline"] == 1
    assert diff["level_distribution_smart"] == {0: 1, 2: 1}
    assert diff["content_char_delta"] == 0


# ---------------------------------------------------------------------------
# heading length cap reader + weighted truncation (merged doc-title support)
# ---------------------------------------------------------------------------


def test_heading_max_chars_reads_env_and_floors_below_three(monkeypatch) -> None:
    from lightrag.constants import DEFAULT_DOCX_SMART_HEADING_MAX_CHARS
    from lightrag.parser.docx.smart_heading.guardrails import heading_max_chars

    monkeypatch.delenv("DOCX_SMART_HEADING_MAX_CHARS", raising=False)
    assert heading_max_chars() == DEFAULT_DOCX_SMART_HEADING_MAX_CHARS
    monkeypatch.setenv("DOCX_SMART_HEADING_MAX_CHARS", "250")
    assert heading_max_chars() == 250
    for bad in ("2", "0", "-5"):
        monkeypatch.setenv("DOCX_SMART_HEADING_MAX_CHARS", bad)
        assert heading_max_chars() == DEFAULT_DOCX_SMART_HEADING_MAX_CHARS


@pytest.mark.parametrize(
    ("cap_env", "text", "mode"),
    [
        pytest.param("180", "广州市平台  建设方案", "unchanged", id="short-unchanged"),
        pytest.param("default", "标" * 80, "weighted", id="weighted-cap"),
        pytest.param("2", "标" * 100, "tiny-floor", id="floors-tiny-cap"),
        pytest.param("300", "x" * 250, "raw-ceiling", id="raw-ceiling-above-cap"),
    ],
)
def test_truncate_to_heading_length_cases(
    monkeypatch, cap_env: str, text: str, mode: str
) -> None:
    from lightrag.constants import (
        DEFAULT_DOCX_SMART_HEADING_MAX_CHARS,
        MAX_HEADING_LENGTH,
    )
    from lightrag.parser.docx.smart_heading.guardrails import (
        truncate_to_heading_length,
        weighted_char_length,
    )

    env_val = (
        str(DEFAULT_DOCX_SMART_HEADING_MAX_CHARS) if cap_env == "default" else cap_env
    )
    monkeypatch.setenv("DOCX_SMART_HEADING_MAX_CHARS", env_val)
    out = truncate_to_heading_length(text)

    if mode == "unchanged":
        assert out == text
    elif mode == "weighted":
        assert weighted_char_length(out) <= DEFAULT_DOCX_SMART_HEADING_MAX_CHARS
        assert out.endswith("...")
    elif mode == "tiny-floor":
        assert out != "..."
        assert weighted_char_length(out) <= DEFAULT_DOCX_SMART_HEADING_MAX_CHARS
    else:
        assert len(out) == MAX_HEADING_LENGTH
        assert out.endswith("...")


@pytest.mark.requires_spacy_models
def test_strong_body_length_floors_tiny_cap(monkeypatch) -> None:
    """strong_body_reason shares heading_max_chars: a nonsensical cap<3 must
    not demote an ordinary short heading via the length rule."""
    from lightrag.parser.docx.smart_heading.guardrails import strong_body_reason

    monkeypatch.setenv("DOCX_SMART_HEADING_MAX_CHARS", "2")
    assert strong_body_reason("某某管理办法实施细则条例") is None


def test_validate_heading_max_chars_env_returns_parsed(monkeypatch) -> None:
    from lightrag.parser.docx.smart_heading.guardrails import (
        validate_heading_max_chars_env,
    )

    monkeypatch.delenv("DOCX_SMART_HEADING_MAX_CHARS", raising=False)
    assert validate_heading_max_chars_env() is None
    monkeypatch.setenv("DOCX_SMART_HEADING_MAX_CHARS", "180")
    assert validate_heading_max_chars_env() == 180
    monkeypatch.setenv("DOCX_SMART_HEADING_MAX_CHARS", "50")
    assert validate_heading_max_chars_env() == 50


@pytest.mark.parametrize("bad", ["abc", "2"])
def test_validate_heading_max_chars_env_raises_on_invalid(monkeypatch, bad) -> None:
    from lightrag.parser.docx.smart_heading.guardrails import (
        validate_heading_max_chars_env,
    )

    monkeypatch.setenv("DOCX_SMART_HEADING_MAX_CHARS", bad)
    with pytest.raises(ValueError, match="DOCX_SMART_HEADING_MAX_CHARS"):
        validate_heading_max_chars_env()
