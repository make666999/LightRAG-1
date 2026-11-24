"""G4 tests (judgment layer): title-block candidates and LLM verdict parsing.

Strong-body / homophone judgments are injected as deterministic stubs so
these tests run without spaCy; the real implementations are covered by
test_smart_heading_guards.py.
"""

from __future__ import annotations

import json

import pytest

from lightrag.parser.docx.parse_document import ParagraphRecord
from lightrag.parser.docx.smart_heading import guardrails
from lightrag.parser.docx.smart_heading.title_block import (
    TitleBlockCandidate,
    TitleBlockLLMError,
    compose_doc_title,
    compose_title_heading,
    detect_imprint_regions,
    find_title_block_candidates,
    flatten_heading_line,
    judge_title_block,
)

from tests.parser.docx._smart_heading_helpers import (
    make_empty_para as _empty,
    make_para as _para,
    stub_always_veto as _stub_always_veto,
    stub_no_veto as _stub_no_veto,
    stub_strong_body as _stub_strong_body,
)

pytestmark = pytest.mark.offline


def _find(records, **kw):
    kw.setdefault("fs_base_pt", 12.0)
    kw.setdefault("strong_body", _stub_strong_body)
    kw.setdefault("numbering_veto", _stub_no_veto)
    return find_title_block_candidates(records, **kw)


# ---------------------------------------------------------------------------
# candidate discovery
# ---------------------------------------------------------------------------


@pytest.mark.requires_spacy_models
def test_two_line_cover_with_decimal_lead_in_forms_window() -> None:
    """Regression (test13 事故调查报告): a two-line 22pt cover whose first line
    is a lead-in carrying a decimal must form ONE multi-window together with
    the main title under the REAL ``strong_body_reason``."""
    records = [
        _para("广州市增城区“7.19”索菲亚定制家居项目", size=22.0),  # lead-in (引题)
        _para("建筑工地塔式起重机坍塌较大事故调查报告", size=22.0),  # main title
        _para("2015年7月19日发生塔式起重机坍塌事故，造成4人死亡。", size=16.0),
    ]
    cands = find_title_block_candidates(
        records,
        fs_base_pt=16.0,
        strong_body=guardrails.strong_body_reason,
        numbering_veto=lambda _cls, _t: None,
    )
    assert len(cands) == 1
    c = cands[0]
    assert (c.start, c.end, c.single, c.trigger) == (0, 2, False, "multi_window")


def test_single_candidate_first_eligible_paragraph() -> None:
    """A lone big opening line is a single-paragraph candidate after leading
    empty and recognized TOC paragraphs; a digit-led title survives when
    the homophone veto revokes its numbering."""
    records = [
        _empty(),
        _para("第一章 目录条目............3", size=12.0, is_toc_field=True),
        _para("2026年度工作报告", size=18.0),
        _para("正文第一段说明了年度工作的总体情况。", size=12.0),
        _para("正文第二段继续说明。", size=12.0),
    ]
    cands = _find(records, numbering_veto=_stub_always_veto)
    assert len(cands) == 1 and cands[0].single and cands[0].start == 2
    assert _find(records, numbering_veto=_stub_no_veto) == []


@pytest.mark.parametrize(
    "records",
    [
        pytest.param(
            [_para(f"正文段落{i}，以句号结尾。", size=12.0) for i in range(3)]
            + [_para("突然的大字号行", size=18.0)]
            + [_para(f"后续正文{i}，以句号结尾。", size=12.0) for i in range(3)],
            id="after-non-title-first-paragraph",
        ),
        pytest.param(
            [
                _para("开头正文，以句号结尾。", size=12.0),
                _empty(),
                _para("居中的独立标题", size=18.0, alignment="center"),
                _empty(),
                _para("后续正文，以句号结尾。", size=12.0),
            ],
            id="centered-line-with-blank-flanks",
        ),
        pytest.param(
            [
                _para("前一篇正文结束。", size=12.0),
                _para("另一篇文章的标题", size=18.0, page_break_before=True),
                _para("这一篇正文开始，以句号结尾。", size=12.0),
            ],
            id="after-page-break-before",
        ),
        pytest.param(
            [
                _para(
                    "上一篇的收尾正文，以句号结尾。",
                    size=12.0,
                    has_page_break_run=True,
                    has_nonleading_page_break_run=True,
                ),
                _para("下一篇文章的标题", size=18.0),
                _para("下一篇正文开始，以句号结尾。", size=12.0),
            ],
            id="after-previous-nonleading-page-break-run",
        ),
        pytest.param(
            [
                _para("前文正文结束。", size=12.0),
                _para("段内分页后的标题", size=18.0, has_leading_page_break_run=True),
                _para("后续正文开始。", size=12.0),
            ],
            id="leading-page-run",
        ),
        pytest.param(
            [
                _para("前文正文结束。", size=12.0),
                ParagraphRecord(kind="section_break"),
                _para("分节后的标题", size=18.0),
                _para("后续正文开始。", size=12.0),
            ],
            id="section-record",
        ),
        pytest.param(
            [
                _para("前文正文结束。", size=12.0, ends_section=True),
                _para("段尾分节后的标题", size=18.0),
                _para("后续正文开始。", size=12.0),
            ],
            id="paragraph-section",
        ),
        pytest.param(
            [
                _para("前文正文结束。", size=12.0),
                _para("字号变化处的标题", size=18.0),
                _para("后续正文开始。", size=10.5),
            ],
            id="font-change",
        ),
    ],
)
def test_mid_document_boundaries_do_not_seed_single_candidate(records) -> None:
    assert _find(records) == []


def test_single_channel_never_scans_past_first_eligible_paragraph(monkeypatch) -> None:
    """The retired review-cap setting is ignored because the single channel
    can now emit at most one candidate and never scans for later replacements."""
    monkeypatch.setenv("DOCX_SMART_SINGLE_TITLE_LLM_MAX", "2")
    records = []
    for i in range(4):
        records.append(_para(f"独立文章标题{i}", size=18.0, page_break_before=True))
        records.append(_para(f"文章{i}的正文，以句号结尾。", size=12.0))
    cands = _find(records)
    assert len(cands) == 1
    assert cands[0].start == 0


# ---------------------------------------------------------------------------
# LLM verdict parsing + validation
# ---------------------------------------------------------------------------

_TITLE_RECORDS = [
    _para("中华人民共和国某某管理办法", size=22.0),
    _para("（2026年修订）", size=12.0),
    _empty(),
    _para("国某发〔2026〕12号", size=12.0),
    _para("正文第一条从这里开始。", size=12.0),
]
_CANDIDATE = TitleBlockCandidate(start=0, end=4, single=False, trigger="multi_window")


def _judge_with(payload: dict | str, records=None, candidate=None, warnings=None):
    raw = (
        payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    )

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return raw

    return judge_title_block(
        candidate or _CANDIDATE,
        records or _TITLE_RECORDS,
        _llm,
        warnings=warnings if warnings is not None else {},
    )


_COLON_LEAD_IN = (
    "为确保本项目产品的售后服务需求得以及时响应和解决，公司将组建该项目"
    "售后服务团队，具体负责人员事项如下表："
)


def test_body_font_colon_lead_in_never_forms_title_block() -> None:
    """Stability guard for the multi-sentence gate change (test13): a long
    multi-clause colon lead-in at BODY font, in a body context, is never a
    title-block candidate — it is not a title line, so it can neither seed nor
    grow a qualifying window. Behaviour is identical before and after the gate
    (this asserts the fix does NOT newly promote body-region lead-ins)."""
    records = [
        _para("2015年7月19日发生事故，造成4人死亡。", size=16.0),
        _para(_COLON_LEAD_IN, size=16.0),
        _para("事故原因分析如下。", size=16.0),
    ]
    assert _find(records, fs_base_pt=16.0) == []


def test_title_font_colon_lead_in_absorbed_but_llm_gate_rejects() -> None:
    """The multi-sentence gate change means a colon lead-in that spaCy used to
    pseudo-split is no longer strong body. When such a lead-in is ALSO at title
    font and sits right after the main title, it now joins the multi-window
    (a known edge the fix introduces). The mandatory LLM judge is the backstop:
    given an ``is_title_block: false`` verdict the candidate is rejected wholesale.

    The default ``_stub_strong_body`` returns None for this short, non-terminated
    lead-in, mirroring the post-fix ``strong_body_reason`` verdict, so the
    candidate here reflects the real post-fix behaviour."""
    records = [
        _para("建筑工地塔式起重机坍塌较大事故调查报告", size=22.0),  # main title
        _para(_COLON_LEAD_IN, size=22.0),  # colon lead-in at TITLE font (edge)
        _para("正文从这里开始。", size=16.0),
    ]
    cands = _find(records, fs_base_pt=16.0)
    # The lead-in is absorbed into the cover window (documented edge).
    assert len(cands) == 1
    assert (cands[0].start, cands[0].end, cands[0].single) == (0, 2, False)
    # Mandatory LLM gate rejects the whole candidate → nothing promoted.
    decision = _judge_with(
        {"is_title_block": False, "headings": [0], "body": [1]},
        records=records,
        candidate=cands[0],
    )
    assert not decision.is_title_block


@pytest.mark.parametrize(
    (
        "payload",
        "records",
        "candidate",
        "expected_main",
        "expected_sub",
        "expected_doc_num",
        "expected_members",
        "expected_composed",
        "expect_relaxed",
    ),
    [
        pytest.param(
            # Verbatim answer -> exact locate tier, no relaxed warning.
            {
                "is_title_block": True,
                "main_title": "中华人民共和国某某管理办法",
                "sub_title": "（2026年修订）",
                "doc_number": "国某发〔2026〕12号",
                "classification": None,
                "publisher": None,
                "date": "2026",
            },
            None,
            None,
            "中华人民共和国某某管理办法",
            "（2026年修订）",
            "国某发〔2026〕12号",
            (0, 1, 3),
            "中华人民共和国某某管理办法 — （2026年修订） — 国某发〔2026〕12号(2026)",
            False,
            id="parsed-and-composed-exact",
        ),
        pytest.param(
            # Concatenated two-line main title locates cleanly.
            {"is_title_block": True, "main_title": "中华人民共和国某某管理办法"},
            [
                _para("中华人民共和国", size=22.0),
                _para("某某管理办法", size=22.0),
                _para("正文开始。", size=12.0),
            ],
            TitleBlockCandidate(start=0, end=2, single=False, trigger="t"),
            "中华人民共和国某某管理办法",
            None,
            None,
            (0, 1),
            None,
            False,
            id="concatenated-main-title-locates",
        ),
        pytest.param(
            # Overlapping boundary word deduped by LLM -> assembled relaxed tier.
            {
                "is_title_block": True,
                "main_title": "2027-2028年信息化运维项目方案",
                "publisher": "广州市节能中心",
            },
            [
                _para("广州市节能中心", size=24.0),
                _para("2027-2028年信息化运维项目", size=24.0),
                _para("项目方案", size=24.0),
                _para("正文开始。", size=12.0),
            ],
            TitleBlockCandidate(start=0, end=3, single=False, trigger="t"),
            "2027-2028年信息化运维项目方案",
            None,
            None,
            (0, 1, 2),
            None,
            True,
            id="overlap-dedup-main-title-relaxed",
        ),
        pytest.param(
            # Full-width -> half-width punctuation and Latin case folding.
            {
                "is_title_block": True,
                "main_title": "某某平台(2026年修订)",
                "sub_title": "Q1-Q2 Roadmap",
                "doc_number": "国某发[2026]12号",
            },
            [
                _para("某某平台（2026年修订）", size=22.0),
                _para("Q1—Q2 ROADMAP", size=16.0),
                _para("国某发〔2026〕12号", size=12.0),
                _para("正文开始。", size=12.0),
            ],
            TitleBlockCandidate(start=0, end=3, single=False, trigger="t"),
            "某某平台(2026年修订)",
            "Q1-Q2 Roadmap",
            "国某发[2026]12号",
            (0, 1, 2),
            None,
            True,
            id="punctuation-and-case-normalized-relaxed",
        ),
    ],
)
def test_title_verdict_locate_and_parse_cases(
    payload,
    records,
    candidate,
    expected_main: str,
    expected_sub: str | None,
    expected_doc_num: str | None,
    expected_members: tuple[int, ...],
    expected_composed: str | None,
    expect_relaxed: bool,
) -> None:
    warnings: dict = {}
    decision = _judge_with(
        payload, records=records, candidate=candidate, warnings=warnings
    )
    assert decision.is_title_block
    assert decision.main_title == expected_main
    assert decision.member_indices == expected_members
    if expected_sub is not None:
        assert decision.sub_title == expected_sub
    if expected_doc_num is not None:
        assert decision.doc_number == expected_doc_num
    if expected_composed is not None:
        assert compose_title_heading(decision) == expected_composed
    if expect_relaxed:
        assert warnings["title_block_locate_relaxed"] == 1
    else:
        assert "title_block_locate_relaxed" not in warnings


def test_title_verdict_multiline_fields_flattened(monkeypatch) -> None:
    """Title material renders single-line at verdict construction."""
    from lightrag.constants import DEFAULT_DOCX_SMART_HEADING_MAX_CHARS

    monkeypatch.setenv(
        "DOCX_SMART_HEADING_MAX_CHARS", str(DEFAULT_DOCX_SMART_HEADING_MAX_CHARS)
    )
    records = [
        _para("中华人民共和国\n某某管理办法", size=22.0),
        _para("Implementation\nGuide", size=16.0),
        _para("正文开始。", size=12.0),
    ]
    decision = _judge_with(
        {
            "is_title_block": True,
            "main_title": "中华人民共和国\n某某管理办法",
            "sub_title": "Implementation\r\nGuide",
        },
        records=records,
        candidate=TitleBlockCandidate(start=0, end=2, single=False, trigger="t"),
    )
    assert decision.main_title == "中华人民共和国某某管理办法"
    assert decision.sub_title == "Implementation Guide"
    assert "\n" not in compose_title_heading(decision)
    doc_title = compose_doc_title(decision.main_title, decision.sub_title)
    assert doc_title == "中华人民共和国某某管理办法  Implementation Guide"
    assert "\n" not in doc_title


def test_compose_doc_title_joins_with_double_space(monkeypatch) -> None:
    from lightrag.constants import DEFAULT_DOCX_SMART_HEADING_MAX_CHARS

    monkeypatch.setenv(
        "DOCX_SMART_HEADING_MAX_CHARS", str(DEFAULT_DOCX_SMART_HEADING_MAX_CHARS)
    )
    assert compose_doc_title("广州市平台", "建设方案") == "广州市平台  建设方案"
    assert compose_doc_title("广州市平台", "") == "广州市平台"
    assert compose_doc_title("广州市平台", None) == "广州市平台"


def test_compose_doc_title_bounds_length(monkeypatch) -> None:
    """Both the merged and the main-only paths are bounded to the strong-body
    weighted cap (unified handling, user ruling)."""
    from lightrag.constants import DEFAULT_DOCX_SMART_HEADING_MAX_CHARS

    monkeypatch.setenv(
        "DOCX_SMART_HEADING_MAX_CHARS", str(DEFAULT_DOCX_SMART_HEADING_MAX_CHARS)
    )
    cap = DEFAULT_DOCX_SMART_HEADING_MAX_CHARS
    merged = compose_doc_title("标" * 40, "案" * 40)
    assert guardrails.weighted_char_length(merged) <= cap
    assert merged.endswith("...")
    main_only = compose_doc_title("则" * 70, None)
    assert guardrails.weighted_char_length(main_only) <= cap
    assert main_only.endswith("...")


def test_flatten_heading_line_unicode_breaks() -> None:
    assert flatten_heading_line("总则\u2028细则") == "总则细则"
    assert flatten_heading_line("  Part A \n\n Part B ") == "Part A Part B"
    assert flatten_heading_line("单行标题") == "单行标题"
    assert flatten_heading_line("\n \n") == ""


@pytest.mark.parametrize(
    ("payload", "records", "candidate", "match"),
    [
        pytest.param(
            {"is_title_block": True, "main_title": "完全虚构的标题"},
            None,
            None,
            "cannot be located",
            id="hallucinated-title",
        ),
        pytest.param(
            # Scattered single chars cannot be assembled into a title.
            {"is_title_block": True, "main_title": "中共某国某人管某某某和办"},
            [_para("中某人办法某某共和国管理", size=22.0), _para("正文开始。", size=12.0)],
            TitleBlockCandidate(start=0, end=1, single=False, trigger="t"),
            "cannot be located",
            id="stitched-from-stray-chars",
        ),
        pytest.param(
            # Reversed source fragments must not locate.
            {"is_title_block": True, "main_title": "广州市节能中心信息化运维项目"},
            [
                _para("信息化运维项目", size=22.0),
                _para("广州市节能中心", size=22.0),
                _para("正文开始。", size=12.0),
            ],
            TitleBlockCandidate(start=0, end=2, single=False, trigger="t"),
            "cannot be located",
            id="out-of-order-fragments",
        ),
        pytest.param(
            "I think this is a title block.",
            None,
            None,
            "no JSON object",
            id="invalid-json",
        ),
    ],
)
def test_title_verdict_invalid_hard_fails(payload, records, candidate, match: str) -> None:
    with pytest.raises(TitleBlockLLMError, match=match):
        _judge_with(payload, records=records, candidate=candidate)


@pytest.mark.parametrize(
    (
        "payload",
        "records",
        "candidate",
        "expected_headings",
        "expected_body",
        "expect_incomplete",
    ),
    [
        pytest.param(
            {"is_title_block": False, "headings": [0], "body": [1, 2]},
            None,
            None,
            (0,),
            (1, 3),
            False,
            id="classifies-every-paragraph",
        ),
        pytest.param(
            # Unmentioned window index 2 abstains rather than hard-failing.
            {"is_title_block": False, "headings": [0], "body": [1]},
            None,
            None,
            (0,),
            (1,),
            True,
            id="incomplete-partition-recovers",
        ),
        pytest.param(
            # Metadata-routed standard number omitted from headings/body abstains.
            {
                "is_title_block": False,
                "headings": [1],
                "body": [],
                "doc_number": "GB/T 9704—2012",
            },
            [_para("GB/T 9704—2012", size=12.0), _para("前　言", size=14.0)],
            TitleBlockCandidate(start=0, end=2, single=False, trigger="multi_window"),
            (1,),
            (),
            True,
            id="metadata-dropped-index-recovers",
        ),
    ],
)
def test_non_title_verdict_partition_cases(
    payload,
    records,
    candidate,
    expected_headings: tuple[int, ...],
    expected_body: tuple[int, ...],
    expect_incomplete: bool,
) -> None:
    warnings: dict = {}
    decision = _judge_with(
        payload, records=records, candidate=candidate, warnings=warnings
    )
    assert not decision.is_title_block
    assert decision.heading_indices == expected_headings
    assert decision.body_indices == expected_body
    if expect_incomplete:
        assert warnings["title_block_partition_incomplete"] == 1
    else:
        assert "title_block_partition_incomplete" not in warnings


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        pytest.param(
            {"is_title_block": False, "headings": [0, 1], "body": [1, 2]},
            "malformed",
            id="overlap-heading-and-body",
        ),
        pytest.param(
            {"is_title_block": False, "headings": [99], "body": []},
            "malformed",
            id="out-of-range-index",
        ),
        pytest.param(
            {"is_title_block": False, "headings": [0, 0], "body": [1, 2]},
            "malformed",
            id="duplicate-in-one-list",
        ),
        pytest.param(
            {"is_title_block": False},
            "must be",
            id="missing-fields",
        ),
        pytest.param(
            {"is_title_block": False, "headings": None, "body": None},
            "must be",
            id="explicit-null",
        ),
        pytest.param(
            {"is_title_block": False, "headings": [True], "body": [1]},
            "must be",
            id="bool-not-an-index",
        ),
    ],
)
def test_non_title_verdict_malformed_partition_hard_fails(
    payload: dict, match: str
) -> None:
    """Only omission is forgiven; malformed partitions hard-fail."""
    with pytest.raises(TitleBlockLLMError, match=match):
        _judge_with(payload)


def test_outline_paragraph_never_demoted_by_llm(monkeypatch) -> None:
    """G4-5 / I2: an outlineLvl paragraph voted 'body' keeps heading standing."""
    records = [
        _para("大标题", size=22.0),
        _para("带物理大纲的段落", size=12.0, outline_level_raw=1),
        _para("正文开始。", size=12.0),
    ]
    warnings: dict = {}
    decision = _judge_with(
        {"is_title_block": False, "headings": [0], "body": [1]},
        records=records,
        candidate=TitleBlockCandidate(start=0, end=2, single=False, trigger="t"),
        warnings=warnings,
    )
    assert decision.heading_indices == (0, 1)
    assert decision.body_indices == ()
    assert warnings["title_block_llm_outline_demotion_blocked"] == 1


def test_window_token_cap_truncates(monkeypatch) -> None:
    """G4-4: an over-long window is tail-truncated with a warning."""
    monkeypatch.setenv("DOCX_SMART_LLM_WINDOW_TOKENS", "30")
    records = [
        _para("标题块第一行内容比较长" * 3, size=22.0),
        _para("标题块第二行内容也比较长" * 3, size=12.0),
        _para("标题块第三行内容同样很长" * 3, size=12.0),
        _para("正文开始。", size=12.0),
    ]
    seen_prompt: dict = {}

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        seen_prompt["prompt"] = prompt
        return json.dumps(
            {"is_title_block": True, "main_title": records[0].text},
            ensure_ascii=False,
        )

    warnings: dict = {}
    decision = judge_title_block(
        TitleBlockCandidate(start=0, end=3, single=False, trigger="t"),
        records,
        _llm,
        warnings=warnings,
    )
    assert warnings["title_block_window_truncated"] == 1
    # cap=30 only fits the first (always-kept) line; the tail is dropped.
    assert "[1]" not in seen_prompt["prompt"]
    assert decision.member_indices == (0,)


def test_missing_llm_hard_fails() -> None:
    with pytest.raises(TitleBlockLLMError, match="none is configured"):
        judge_title_block(_CANDIDATE, _TITLE_RECORDS, None)


def test_single_candidate_window_includes_context() -> None:
    records = [
        _para("上文一。", size=12.0),
        _para("上文二。", size=12.0),
        _para("独立主标题", size=18.0, page_break_before=True),
        _para("下文一。", size=12.0),
        _para("下文二。", size=12.0),
    ]
    seen: dict = {}

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        seen["prompt"] = prompt
        return json.dumps(
            {"is_title_block": True, "main_title": "独立主标题"}, ensure_ascii=False
        )

    decision = judge_title_block(
        TitleBlockCandidate(start=2, end=3, single=True, trigger="single_line"),
        records,
        _llm,
    )
    # Window shows ±2 context paragraphs; the block itself is only the line.
    assert "上文一。" in seen["prompt"] and "下文二。" in seen["prompt"]
    assert decision.is_title_block
    assert decision.member_indices == (2,)


def test_single_candidate_line_survives_truncation(monkeypatch) -> None:
    """Review D2: when huge context would blow the token cap, the candidate
    line must still be shown to the LLM (it is pre-reserved) — otherwise a
    true verdict composes a level-0 heading from context alone."""
    monkeypatch.setenv("DOCX_SMART_LLM_WINDOW_TOKENS", "30")
    records = [
        _para("很长的上文内容" * 20, size=12.0),  # ~oversized context before
        _para("很长的上文内容之二" * 20, size=12.0),
        _para("独立主标题", size=18.0, page_break_before=True),  # the candidate
        _para("下文。", size=12.0),
    ]
    seen: dict = {}

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        seen["prompt"] = prompt
        return json.dumps(
            {"is_title_block": True, "main_title": "独立主标题"}, ensure_ascii=False
        )

    decision = judge_title_block(
        TitleBlockCandidate(start=2, end=3, single=True, trigger="single_line"),
        records,
        _llm,
        warnings={},
    )
    assert "独立主标题" in seen["prompt"]  # candidate never truncated away
    assert decision.is_title_block and decision.member_indices == (2,)


def test_single_candidate_main_title_must_be_the_candidate_line() -> None:
    """Review D4: the single-candidate hallucination guard is scoped to the
    candidate paragraph — a main_title lifted from reference-only context
    must NOT validate."""
    records = [
        _para("上下文里的另一句话", size=12.0),
        _para("真正的独立主标题", size=18.0, page_break_before=True),
        _para("下文。", size=12.0),
    ]
    # LLM (wrongly) returns a context paragraph as the main title.
    with pytest.raises(TitleBlockLLMError, match="cannot be located"):
        _judge_with(
            {"is_title_block": True, "main_title": "上下文里的另一句话"},
            records=records,
            candidate=TitleBlockCandidate(
                start=1, end=2, single=True, trigger="single_line"
            ),
        )


def test_single_candidate_false_verdict_leaves_context_untouched() -> None:
    """Review D3: a non-title verdict on a single candidate neither grants
    nor vetoes the reference-only context — the candidate re-enters the
    normal flow. The LLM's headings/body partition over context is ignored,
    and the candidate is NOT demoted."""
    records = [
        _para("上文一。", size=12.0),
        _para("上文二。", size=12.0),
        _para("独立大字号行", size=18.0, page_break_before=True),
        _para("下文一。", size=12.0),
        _para("下文二。", size=12.0),
    ]
    decision = _judge_with(
        # Even a hostile partition that votes the candidate + context as body
        # must not propagate.
        {"is_title_block": False, "headings": [], "body": [0, 1, 2, 3, 4]},
        records=records,
        candidate=TitleBlockCandidate(
            start=2, end=3, single=True, trigger="single_line"
        ),
    )
    assert not decision.is_title_block
    assert decision.heading_indices == ()
    assert decision.body_indices == ()  # candidate not demoted; context untouched


# ---------------------------------------------------------------------------
# spec-vs-implementation audit regressions (A7 / A9)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("records", "expected_span"),
    [
        pytest.param(
            # Basic multi-window: big main title + metadata + blank + publisher.
            [
                _para("公司数字化转型白皮书", size=22.0),
                _para("（2026年版）", size=12.0),
                _empty(),
                _para("某某咨询公司发布", size=12.0),
                _para("正文从这里开始，介绍研究的背景与目标。", size=12.0),
            ],
            (0, 4),
            id="basic-cover-stops-before-strong-body",
        ),
        pytest.param(
            # +2pt over FS_base (12pt) clears the unified bar.
            [
                _para("产品技术白皮书", size=14.0),
                _para("研发中心", size=12.0),
                _para("这一段是以句号结尾的正式正文内容，用来终止候选窗口的生长。", size=12.0),
            ],
            (0, 2),
            id="two-pt-over-body-clears-bar",
        ),
        pytest.param(
            # Real outline heading terminates the cover window cleanly.
            [
                _para("公司数字化转型白皮书", size=22.0),
                _para("研发中心发布", size=12.0),
                _para("概述", size=14.0, outline_level_raw=0),
                _para("这一段是以句号结尾的正文内容。", size=12.0),
            ],
            (0, 2),
            id="stops-before-outline-heading",
        ),
        pytest.param(
            # Genuine numbering (no physical outline) terminates the window.
            [
                _para("公司数字化转型白皮书", size=22.0),
                _para("研发中心发布", size=12.0),
                _para("第二章 总体设计", size=14.0),
                _para("这一段是以句号结尾的正文内容。", size=12.0),
            ],
            (0, 2),
            id="stops-before-numbered-heading",
        ),
        pytest.param(
            # Length-demoted heading (outline_level=None, outline_level_raw=1)
            # still serves as a boundary.
            [
                _para("公司数字化转型白皮书", size=22.0),
                _para("研发中心发布", size=12.0),
                _para(
                    "某个被样式标成标题却因过长而被降级的段落标题行内容",
                    size=14.0,
                    outline_level=None,
                    outline_level_raw=1,
                ),
                _para("这一段是以句号结尾的正文内容。", size=12.0),
            ],
            (0, 2),
            id="boundary-uses-raw-outline-level",
        ),
        pytest.param(
            # OR dominance: 18pt beats 16pt following heading (behind imprint).
            [
                _para("第一章 绪论", size=20.0, outline_level_raw=0),
                _para("抄送：各相关单位", size=12.0),
                _para("某某办公室 2026年6月30日 印发", size=12.0),
                _para("某某产品发布公告", size=18.0),
                _para("配套的说明行", size=12.0),
                _para("一段以句号结尾的正文内容，用来终止窗口生长。", size=12.0),
                _para("第二章 概览", size=16.0, outline_level_raw=0),
            ],
            (3, 5),
            id="dominant-over-one-flank",
        ),
        pytest.param(
            # Unstructured doc (no neighbor headings): +2pt absolute bar suffices.
            [
                _para("公司数字化转型白皮书", size=14.0),
                _para("研发中心发布", size=12.0),
                _para("这一段是以句号结尾的正文内容，用来终止窗口生长。", size=12.0),
            ],
            (0, 2),
            id="dominance-passes-without-neighbor-headings",
        ),
    ],
)
def test_multi_window_admitted_cases(
    records, expected_span: tuple[int, int]
) -> None:
    cands = _find(records)
    assert len(cands) == 1
    c = cands[0]
    assert not c.single and c.trigger == "multi_window"
    assert (c.start, c.end) == expected_span


@pytest.mark.parametrize(
    ("records", "assert_no_suppression_events"),
    [
        pytest.param(
            # Only one non-body line at base size -> no multi or single candidate.
            [
                _para("孤立小字号行", size=12.0),
                _para("这是很长的正文，它明确地以句号结尾。", size=12.0),
            ],
            False,
            id="requires-two-paragraphs-and-big-line",
        ),
        pytest.param(
            # +1pt over base: below the unified +2pt bar.
            [
                _para("产品技术白皮书", size=13.0),
                _para("研发中心", size=12.0),
                _para("这一段是以句号结尾的正式正文内容，用来终止候选窗口的生长。", size=12.0),
            ],
            False,
            id="plus-one-pt-below-two-pt-bar",
        ),
        pytest.param(
            # Flat at base size -> no title line.
            [
                _para("产品技术白皮书", size=12.0),
                _para("研发中心", size=12.0),
                _para("这一段是以句号结尾的正式正文内容，用来终止候选窗口的生长。", size=12.0),
            ],
            False,
            id="flat-base-size-no-title-line",
        ),
        pytest.param(
            # M1212 repro: body-sized references + numbered outline headings.
            [
                _para("GJB/Z 106A-2005  工艺标准化大纲编制指南", size=12.0),
                _para("GJB/Z 114A-2015  产品标准化大纲编制指南", size=12.0),
                _para("《M1212(ML001) 模块技术协议书》", size=12.0),
                _para("1.4   产品简介", size=14.0, outline_level_raw=1),
                _para("1.4.1   产品组成", size=14.0, outline_level_raw=2),
                _para("这一段是以句号结尾的正式正文内容，用来终止候选窗口的生长。", size=12.0),
            ],
            False,
            id="excludes-numbered-outline-headings",
        ),
        pytest.param(
            # Window never starts on an outline heading, even an 18pt one.
            [
                _para("引言", size=18.0, outline_level_raw=0),
                _para("研发中心", size=12.0),
                _para("这一段是以句号结尾的正文内容，用来终止窗口生长。", size=12.0),
            ],
            False,
            id="outline-heading-not-window-start",
        ),
        pytest.param(
            # Not dominant over 18pt outline headings on both flanks (ev == []).
            [
                _para("第一章 绪论", size=18.0, outline_level_raw=0),
                _para("一段以句号结尾的正文内容，用来隔开上一个标题。", size=12.0),
                _para("强调小标题", size=14.0),
                _para("配套的说明行", size=12.0),
                _para("一段以句号结尾的正文内容，用来终止窗口生长。", size=12.0),
                _para("第二章 概览", size=18.0, outline_level_raw=0),
            ],
            True,
            id="not-dominant-over-outline-headings",
        ),
        pytest.param(
            # Dominance reference includes numbered-but-unoutlined headings.
            [
                _para("第一章 绪论", size=18.0),
                _para("一段以句号结尾的正文内容，用来隔开上一个标题。", size=12.0),
                _para("强调小标题", size=14.0),
                _para("配套的说明行", size=12.0),
                _para("一段以句号结尾的正文内容，用来终止窗口生长。", size=12.0),
            ],
            True,
            id="not-dominant-over-numbered-heading",
        ),
    ],
)
def test_multi_window_rejected_cases(
    records, assert_no_suppression_events: bool
) -> None:
    ev: list = []
    assert _find(records, suppressed_events=ev) == []
    if assert_no_suppression_events:
        assert ev == []


def test_title_line_uses_first_line_size_for_split_records() -> None:
    """A soft-break-split heading is judged by its first line's size."""
    split = _para(
        "被拆分的超长主标题首行",
        size=12.0,
        first_line_font_size_pt=15.0,
        demoted_body_text="降级为正文的余部内容",
    )
    records = [
        split,
        _para("编制单位", size=12.0),
        _para("这一段是以句号结尾的正式正文内容，用来终止候选窗口的生长。", size=12.0),
    ]
    assert [c.trigger for c in _find(records)] == ["multi_window"]

    single = _para(
        "整篇文档开头的孤立大标题",
        size=12.0,
        first_line_font_size_pt=14.5,
        demoted_body_text="软回车拆出的正文余部",
    )
    solo = [single] + [
        _para("这一段是以句号结尾的正式正文内容，用来终止候选窗口的生长。", size=12.0)
        for _ in range(3)
    ]
    assert [c.trigger for c in _find(solo)] == ["single_line"]


def test_single_candidate_outline_heading_excluded() -> None:
    """The single-paragraph channel excludes real outline headings."""
    records = [
        _para("引言", size=18.0, outline_level_raw=0),
        _para("正文第一段说明了总体情况。", size=12.0),
        _para("正文第二段继续说明。", size=12.0),
    ]
    assert _find(records) == []

    plain = [
        _para("某某产品发布公告", size=18.0),
        _para("正文第一段说明了总体情况。", size=12.0),
        _para("正文第二段继续说明。", size=12.0),
    ]
    cands = _find(plain)
    assert len(cands) == 1 and cands[0].single and cands[0].start == 0


def test_dominance_ignores_headings_beyond_flank_window(monkeypatch) -> None:
    """Section headings farther than ``_FLANK_WINDOW`` paragraphs are ignored."""
    import lightrag.parser.docx.smart_heading.title_block as tb

    monkeypatch.setattr(tb, "_FLANK_WINDOW", 1)
    records = [
        _para("某某产品发布公告", size=14.0),
        _para("配套的说明行", size=12.0),
        _para("一段以句号结尾的正文内容，用来终止窗口生长。", size=12.0),
        _para("第一章 绪论", size=20.0, outline_level_raw=0),
    ]
    cands = _find(records)
    assert len(cands) == 1 and (cands[0].start, cands[0].end) == (0, 2)


# ---------------------------------------------------------------------------
# Table-channel windows for covers laid out inside tables
# ---------------------------------------------------------------------------


def _table(rows, **kw) -> ParagraphRecord:
    """Fake table record; ``rows`` = per-row lists of (text, size, outline)."""
    return ParagraphRecord(
        kind="table", text="<table>[]</table>", table_cell_features=rows, **kw
    )


def _cover_tables() -> list[ParagraphRecord]:
    """M1212-style cover: form tables (multi-cell rows, short text) + a title
    table (single-cell big rows) + a publisher table."""
    return [
        _table(
            [
                [("档 号", 10.5, False), ("", None, False)],
                [("版 本 号", 10.5, False), ("1V1.0.0", 10.5, False)],
            ]
        ),
        _empty(),
        _table(
            [
                [("产品标准化大纲", 22.0, False)],  # title rows: single cell
                [("某某模块", 22.0, False)],
            ]
        ),
        _table([[("某某电子股份有限公司", 16.0, False)]]),
    ]


@pytest.mark.parametrize(
    ("records_factory", "expected_span"),
    [
        pytest.param(
            # Adjacent qualifying tables (blank lines between ok) merge into one.
            lambda: _cover_tables() + [_para("正文从这里开始，以句号结尾。", size=12.0)],
            (0, 4),
            id="merges-adjacent-qualifying-tables",
        ),
        pytest.param(
            # First disqualified table ends the window AND the run.
            lambda: [
                _table([[("产品标准化大纲", 22.0, False)]]),
                _table(
                    [
                        [
                            ("序号", 10.5, False),
                            ("这一格是以句号结尾的数据内容。", 10.5, False),
                        ]
                    ]
                ),
                _table([[("另一份大字标题", 22.0, False)]]),
                _para("正文从这里开始，以句号结尾。", size=12.0),
            ],
            (0, 1),
            id="prefix-breaks-at-disqualified-table",
        ),
        pytest.param(
            # Disqualified first table yields no candidate even if another follows.
            lambda: [
                _table(
                    [
                        [
                            ("序号", 10.5, False),
                            ("这一格是以句号结尾的数据内容。", 10.5, False),
                        ]
                    ]
                ),
                _table([[("产品标准化大纲", 22.0, False)]]),
                _para("正文从这里开始，以句号结尾。", size=12.0),
            ],
            None,
            id="requires-qualifying-first-table",
        ),
        pytest.param(
            # Dominance gate rejects table smaller than flanking section headings.
            lambda: [
                _para("第一章 绪论", size=20.0, outline_level_raw=0),
                _para("一段以句号结尾的正文，用来隔开标题。", size=12.0),
                _table([[("强调框标题", 14.0, False)]]),
                _para("一段以句号结尾的正文。", size=12.0),
                _para("第二章 概览", size=20.0, outline_level_raw=0),
            ],
            None,
            id="dominance-gate-rejects-smaller-table",
        ),
    ],
)
def test_table_window_discovery_cases(
    records_factory, expected_span: tuple[int, int] | None
) -> None:
    cands = _find(records_factory())
    if expected_span is None:
        assert cands == []
    else:
        assert len(cands) == 1
        c = cands[0]
        assert c.table and c.trigger == "table_window" and not c.single
        assert (c.start, c.end) == expected_span


@pytest.mark.parametrize(
    "bad_cell",
    [
        pytest.param(("这一格是以句号结尾的正文语句。", 10.5, False), id="strong-body"),
        pytest.param(("超" * 40, 10.5, False), id="over-90-weighted-cap"),
        pytest.param(("带物理大纲的格", 10.5, True), id="physical-outline"),
        pytest.param(("抄送：市委各部门", 10.5, False), id="imprint-marker"),
        pytest.param(("某某办公室 2026年6月30日 印发", 10.5, False), id="imprint-closer"),
    ],
)
def test_table_member_gate_rejects_bad_cells(bad_cell) -> None:
    """Rule 2: ONE offending cell disqualifies the whole table."""
    records = [
        _table([[("产品标准化大纲", 22.0, False)], [bad_cell]]),
        _para("正文从这里开始，以句号结尾。", size=12.0),
    ]
    assert _find(records) == []


def test_table_title_row_requires_single_big_unnumbered_cell() -> None:
    """Rule 1: the title row is a single-PHYSICAL-cell row at the +2pt tier
    whose text has no genuine numbering."""
    # (a) all sizes below fs_base+2 -> no title row.
    records = [
        _table([[("产品标准化大纲", 13.0, False)]]),
        _para("正文从这里开始，以句号结尾。", size=12.0),
    ]
    assert _find(records) == []

    # (b) big text but in a MULTI-cell row -> not a title row.
    records = [
        _table([[("产品标准化大纲", 22.0, False), ("旁边一格", 10.5, False)]]),
        _para("正文从这里开始，以句号结尾。", size=12.0),
    ]
    assert _find(records) == []

    # (c) genuine numbering disqualifies the title row...
    records = [
        _table([[("第二章 总体设计", 22.0, False)]]),
        _para("正文从这里开始，以句号结尾。", size=12.0),
    ]
    assert _find(records) == []

    # ...but the homophone veto keeps a date-opener title alive.
    records = [
        _table([[("2026年度工作报告", 22.0, False)]]),
        _para("正文从这里开始，以句号结尾。", size=12.0),
    ]
    cands = _find(records, numbering_veto=_stub_always_veto)
    assert len(cands) == 1 and cands[0].table


# --- judge: table candidates ------------------------------------------------

_TABLE_RECORDS = [
    _table(
        [
            [("产品标准化大纲", 22.0, False)],
            [("某某模块", 22.0, False)],
        ]
    ),
    _table([[("某某电子股份有限公司", 16.0, False)]]),
    _para("正文从这里开始，以句号结尾。", size=12.0),
]
_TABLE_CANDIDATE = TitleBlockCandidate(
    start=0, end=2, single=False, trigger="table_window", table=True
)


def test_table_verdict_members_are_table_records() -> None:
    decision = _judge_with(
        {
            "is_title_block": True,
            "main_title": "产品标准化大纲某某模块",  # concatenated cells
            "publisher": "某某电子股份有限公司",
        },
        records=_TABLE_RECORDS,
        candidate=_TABLE_CANDIDATE,
    )
    assert decision.is_title_block
    assert decision.member_indices == (0, 1)  # the TABLE record indices
    assert (
        compose_title_heading(decision)
        == "产品标准化大纲某某模块(某某电子股份有限公司)"
    )


def test_table_verdict_locates_against_cell_texts() -> None:
    """Locate-back runs against the CELL texts (the record text is a <table>
    placeholder); a hallucinated title hard-fails."""
    with pytest.raises(TitleBlockLLMError):
        _judge_with(
            {"is_title_block": True, "main_title": "凭空捏造的标题"},
            records=_TABLE_RECORDS,
            candidate=_TABLE_CANDIDATE,
        )


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            {"is_title_block": False, "headings": [0, 0], "body": []},
            id="malformed-partition",
        ),
        pytest.param(
            {"is_title_block": False, "headings": [0, 1], "body": [2]},
            id="well-formed-partition",
        ),
    ],
)
def test_table_non_title_verdict_always_ignored(payload: dict) -> None:
    """A non-title verdict for a table window short-circuits before partition
    parsing: cells can never be named headings or vetoed."""
    decision = _judge_with(
        payload, records=_TABLE_RECORDS, candidate=_TABLE_CANDIDATE
    )
    assert not decision.is_title_block
    assert decision.heading_indices == () and decision.body_indices == ()


# ---------------------------------------------------------------------------
# font-size evidence legend & table region separators
# ---------------------------------------------------------------------------


def _prompt_for(records, candidate, *, fs_base_pt=None, payload=None):
    seen: dict = {}

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        seen["prompt"] = prompt
        return json.dumps(
            payload or {"is_title_block": False, "headings": [], "body": []},
            ensure_ascii=False,
        )

    judge_title_block(candidate, records, _llm, warnings={}, fs_base_pt=fs_base_pt)
    return seen["prompt"]


@pytest.mark.parametrize(
    ("records", "expected_snippets"),
    [
        pytest.param(
            # M1212 shape: 22pt title table + 16pt publisher/absorbed heading.
            [
                _table(
                    [
                        [("档 号", 10.5, False), ("", None, False)],
                        [("版 本 号", 10.5, False), ("1V1.0.0", 10.5, False)],
                    ]
                ),
                _table([[("产品标准化大纲", 22.0, False)], [("某某模块", 22.0, False)]]),
                _table([[("某某电子股份有限公司", 16.0, False)]]),
                _para("更 改 记 录", size=16.0),
            ],
            (
                "Font-size evidence — largest tier: [3]=22pt, [4]=22pt",
                "second tier: [5]=16pt, [6]=16pt",
                "(body ≈ 12pt)",
            ),
            id="largest-over-enlarged",
        ),
        pytest.param(
            # Three distinct enlarged sizes split into largest / second / other.
            [
                _table([[("最大标题", 22.0, False)]]),
                _table([[("第二大行", 16.0, False)]]),
                _table([[("第三档行", 14.0, False)]]),
            ],
            (
                "Font-size evidence — largest tier: [0]=22pt",
                "second tier: [1]=16pt",
                "other enlarged: [2]=14pt",
                "two largest tiers",
            ),
            id="three-tiers",
        ),
    ],
)
def test_table_window_font_size_legend_tiers(
    records, expected_snippets: tuple[str, ...]
) -> None:
    cand = TitleBlockCandidate(
        start=0,
        end=len(records),
        single=False,
        trigger="table_window",
        table=True,
        members=tuple(range(len(records))),
    )
    prompt = _prompt_for(records, cand, fs_base_pt=12.0)
    for snippet in expected_snippets:
        assert snippet in prompt


@pytest.mark.parametrize(
    (
        "records",
        "expected_separated",
        "expected_sep_count",
        "expected_sizes_len",
        "window_snippet",
        "expect_layout_note",
        "prompt_checks",
    ),
    [
        pytest.param(
            # Three tables -> two separators, unindexed, exact prompt index list.
            [
                _table([[("档 号", 12.0, False)], [("版 本 号", 12.0, False)]]),
                _table(
                    [
                        [("M1212(ML001) 模块", 22.0, False)],
                        [("标准化大纲", 22.0, False)],
                    ]
                ),
                _table([[("长沙电子股份有限公司", 16.0, False)]]),
            ],
            True,
            2,
            5,
            "----------\n[2] M1212(ML001) 模块\n[3] 标准化大纲\n----------",
            True,
            ("— 0, 1, 2, 3, 4 —", "0, 1, 2, 3, 4, 5"),
            id="three-frames-two-separators",
        ),
        pytest.param(
            # table -> pure-<drawing/> -> table: image renders nothing, 1 separator.
            [
                _table([[("甲表内容", 12.0, False)]]),
                _para('<drawing id="1" name="印章"/>', size=12.0),
                _table([[("乙表内容", 12.0, False)]]),
            ],
            True,
            1,
            2,
            None,
            None,
            None,
            id="pure-image-gap-yields-one-separator",
        ),
        pytest.param(
            # Consecutive absorbed paragraphs share one region (no separator between).
            [
                _table([[("表格行", 12.0, False)]]),
                _para("封面材料段一", size=12.0),
                _para("封面材料段二", size=12.0),
            ],
            True,
            1,
            3,
            "[1] 封面材料段一\n[2] 封面材料段二",
            None,
            None,
            id="consecutive-paras-share-one-region",
        ),
        pytest.param(
            # Single-table window: no separator and no layout note.
            [_table([[("某某公司管理办法", 22.0, False)]])],
            False,
            0,
            1,
            None,
            False,
            None,
            id="single-table-no-separator-no-note",
        ),
        pytest.param(
            # Two tables: layout note appears and precedes the font-size legend.
            [
                _table([[("档 号", 12.0, False)]]),
                _table([[("某某白皮书", 22.0, False)]]),
            ],
            True,
            1,
            2,
            None,
            True,
            None,
            id="two-tables-layout-note-precedes-legend",
        ),
    ],
)
def test_table_window_region_separators(
    records,
    expected_separated: bool,
    expected_sep_count: int,
    expected_sizes_len: int,
    window_snippet: str | None,
    expect_layout_note: bool | None,
    prompt_checks: tuple[str, str] | None,
) -> None:
    from lightrag.parser.docx.smart_heading.title_block import (
        _TABLE_LAYOUT_NOTE,
        _TABLE_REGION_SEPARATOR,
        _render_table_window,
    )

    cand = TitleBlockCandidate(
        start=0,
        end=len(records),
        single=False,
        trigger="table_window",
        table=True,
        members=tuple(range(len(records))),
    )
    window, members, canon, sizes, separated = _render_table_window(records, cand, {})
    assert separated is expected_separated
    assert window.count(_TABLE_REGION_SEPARATOR) == expected_sep_count
    assert len(sizes) == expected_sizes_len
    assert list(members) == list(range(len(records)))
    assert not window.endswith(_TABLE_REGION_SEPARATOR)
    assert _TABLE_REGION_SEPARATOR not in canon
    if window_snippet is not None:
        assert window_snippet in window
    if expect_layout_note is not None or prompt_checks is not None:
        prompt = _prompt_for(records, cand, fs_base_pt=12.0)
        if expect_layout_note is True:
            assert _TABLE_LAYOUT_NOTE in prompt
            assert prompt.index(_TABLE_LAYOUT_NOTE) < prompt.index("Font-size evidence")
        elif expect_layout_note is False:
            assert _TABLE_LAYOUT_NOTE not in prompt
        if prompt_checks is not None:
            present, absent = prompt_checks
            assert present in prompt
            assert absent not in prompt


def test_table_window_truncation_at_boundary_drops_separator_whole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cap hit at a region boundary drops the separator AND its following
    line together — never a dangling separator."""
    from lightrag.parser.docx.smart_heading.title_block import (
        _TABLE_REGION_SEPARATOR,
        _render_table_window,
    )

    records = [
        _table([[("甲", 12.0, False)]]),
        _table([[("乙表内容较长足以触发截断的一行文本", 12.0, False)]]),
    ]
    cand = TitleBlockCandidate(
        start=0,
        end=2,
        single=False,
        trigger="table_window",
        table=True,
        members=(0, 1),
    )
    monkeypatch.setenv("DOCX_SMART_LLM_WINDOW_TOKENS", "5")
    warnings: dict = {}
    window, _m, canon, sizes, separated = _render_table_window(records, cand, warnings)
    assert separated is False
    assert _TABLE_REGION_SEPARATOR not in window
    assert not window.endswith(_TABLE_REGION_SEPARATOR)
    assert len(window.splitlines()) == len(sizes) == 1
    assert _TABLE_REGION_SEPARATOR not in canon
    assert warnings.get("title_block_window_truncated") == 1


def test_paragraph_window_legend_uses_indexed_lines() -> None:
    """[BLANK] rows carry no index, so the legend must number by indexed
    lines — a raw-row count would point at the wrong text."""
    records = [
        _para("公司数字化转型白皮书", size=22.0),
        _empty(),
        _para("（2026年版）", size=12.0),
        _para("某某咨询公司发布", size=16.0),
    ]
    candidate = TitleBlockCandidate(
        start=0, end=4, single=False, trigger="multi_window"
    )
    prompt = _prompt_for(records, candidate, fs_base_pt=12.0)
    assert "Font-size evidence — largest tier: [0]=22pt" in prompt
    # record 3 renders as indexed line [2] (the blank row shifts raw rows).
    assert "second tier: [2]=16pt" in prompt


def test_legend_absent_without_fs_base_or_dominant_line() -> None:
    """No baseline (hand-built candidates) or no dominant line → the prompt
    is legend-free, byte-identical to the pre-evidence form."""
    records = [
        _para("普通一行", size=12.0),
        _para("另一行", size=12.0),
    ]
    candidate = TitleBlockCandidate(
        start=0, end=2, single=False, trigger="multi_window"
    )
    assert "Font-size evidence" not in _prompt_for(records, candidate)
    assert "Font-size evidence" not in _prompt_for(records, candidate, fs_base_pt=12.0)


def test_single_window_legend_covers_only_the_candidate_line() -> None:
    """Single candidates: the ±context lines are reference-only and a true
    verdict's main_title is locate-back-scoped to the candidate line alone —
    a LARGER context line must never be advertised as the largest line, or
    the LLM would follow the cue into a guaranteed locate-back hard fail."""
    records = [
        _para("上一篇文档的巨大结尾行", size=26.0),  # bigger, but context-only
        _para("独立主标题", size=18.0, page_break_before=True),
        _para("正文从这里开始。", size=12.0),
    ]
    candidate = TitleBlockCandidate(start=1, end=2, single=True, trigger="single_line")
    prompt = _prompt_for(
        records,
        candidate,
        fs_base_pt=12.0,
        payload={"is_title_block": True, "main_title": "独立主标题"},
    )
    assert "Font-size evidence — largest tier: [1]=18pt" in prompt
    assert "26pt" not in prompt  # the context line carries no evidence


def test_legend_skips_truncated_lines(monkeypatch) -> None:
    """A token-truncated line is not emitted, so it must not surface in the
    legend either — the legend only ever cites lines the LLM can see."""
    monkeypatch.setenv("DOCX_SMART_LLM_WINDOW_TOKENS", "30")
    records = [
        _para("标题块第一行内容比较长" * 3, size=22.0),
        _para("被截掉的大字号尾行" * 3, size=22.0),
    ]
    candidate = TitleBlockCandidate(
        start=0, end=2, single=False, trigger="multi_window"
    )
    prompt = _prompt_for(
        records,
        candidate,
        fs_base_pt=12.0,
        payload={"is_title_block": True, "main_title": records[0].text},
    )
    assert "largest tier: [0]=22pt" in prompt
    assert "[1]" not in prompt  # neither as window line nor in the legend


def test_prompt_front_matter_negative_list() -> None:
    """Guard the front-matter negative list against prompt refactors.

    The list deliberately lives INSIDE the user template's "If true:
    main_title" field rule — as a selection constraint only — and NOT in the
    system prompt as a window-verdict rule: A/B probes against the deployed
    judge showed any verdict-level phrasing flips genuine covers that merely
    CONTAIN such a heading (the M1212 table window absorbs a trailing
    更 改 记 录 line) to false, while the field constraint keeps every
    verdict intact. The needles cover the listed names, the Chinese
    whitespace-insensitivity note, and the substring-acceptable escape."""
    from lightrag.parser.docx.smart_heading.title_block import (
        _SYSTEM_PROMPT,
        _USER_TEMPLATE,
    )

    for needle in (
        "NEVER a front-matter/bookkeeping heading",
        "目次",
        "更改记录",
        "Revision History",
        "ignoring whitespace between Chinese characters",
        "更改记录管理规范",
        # org-name is a last-resort title (not banned — 机构简介 covers exist)
        "LAST-RESORT main_title",
        # red-header masthead (发文机关…文件) belongs in publisher, not title
        "red-header masthead",
    ):
        assert needle in _USER_TEMPLATE, needle
    # These selection constraints live in the field rule, never the verdict
    # level (a verdict-level phrasing flips genuine covers — see above).
    assert "更改记录" not in _SYSTEM_PROMPT
    assert "LAST-RESORT" not in _SYSTEM_PROMPT
    assert "masthead" not in _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# 公文版记 (imprint) veto: the marker line and its 2 preceding non-blank
# paragraphs are barred from every title-block channel
# ---------------------------------------------------------------------------


def test_imprint_breaks_multi_window_and_vetoes_two_neighbors() -> None:
    """The window ends 2 non-blank paragraphs BEFORE the imprint line; with
    the veto injected away, the imprint line and its neighbours would be
    absorbed as cover-title material (the bug being fixed)."""
    records = [
        _para("公司数字化转型白皮书", size=22.0),  # 0 big main title
        _para("（2026年版）", size=12.0),  # 1
        _para("某某市人民政府办公室", size=12.0),  # 2 ← vetoed (2nd preceding)
        _para("研究策划部编写", size=12.0),  # 3 ← vetoed (1st preceding)
        _para("抄送：各区人民政府", size=12.0),  # 4 anchor
        _para("正文从这里开始，以句号结尾。", size=12.0),
    ]
    cands = _find(records)
    assert len(cands) == 1
    assert not cands[0].single
    assert (cands[0].start, cands[0].end) == (0, 2)

    # Injectable: a no-op marker restores the pre-fix absorption behaviour.
    cands = _find(records, imprint_marker=lambda t: None)
    assert len(cands) == 1
    assert (cands[0].start, cands[0].end) == (0, 5)


def test_imprint_line_never_single_candidate() -> None:
    """Even at the only position eligible for the single channel, an imprint
    line is vetoed (the strong-body stub deliberately does not recognize it)."""
    records = [
        _para("抄送：各市人民政府", size=18.0, alignment="center"),
        _para("后续正文，以句号结尾。", size=12.0),
    ]
    assert _find(records) == []
    cands = _find(records, imprint_marker=lambda t: None)
    assert len(cands) == 1 and cands[0].single and cands[0].start == 0


@pytest.mark.parametrize(
    ("records", "expected_spans", "control_records", "control_kwargs", "control_start"),
    [
        pytest.param(
            # Tail signature/date lines above an imprint line are vetoed.
            [_para(f"正文段落{i}，以句号结尾。", size=12.0) for i in range(3)]
            + [
                _empty(),
                _para("某某市人民政府办公室", size=18.0, alignment="center"),
                _empty(),
                _para("二〇二六年七月一日", size=12.0),
                _para("抄送：各县区人民政府", size=12.0),
            ],
            [],
            [_para(f"正文段落{i}，以句号结尾。", size=12.0) for i in range(3)]
            + [
                _empty(),
                _para("某某市人民政府办公室", size=18.0, alignment="center"),
                _empty(),
                _para("二〇二六年七月一日", size=12.0),
            ],
            {"imprint_boundary_indices": {2}},
            4,
            id="veto-blocks-single-candidate",
        ),
        pytest.param(
            # Backward walk skips empty/whitespace paras without spending budget.
            [
                _para("某某公司发文稿纸", size=22.0),
                _empty(),
                _para("某某办公室", size=12.0),
                _para("   ", size=12.0),
                _para("抄送：各成员单位", size=12.0),
                _para("正文从这里开始，以句号结尾。", size=12.0),
            ],
            [],
            None,
            {"imprint_marker": lambda t: None},
            0,
            id="walk-skips-blank-paras",
        ),
        pytest.param(
            # Non-paragraph record (table) stops backward veto walk.
            [
                _para("产品发布白皮书", size=22.0),
                _para("某某公司编", size=12.0),
                _table([[("附件清单", 10.5, False)]]),
                _para("某某办公室代拟", size=12.0),
                _para("抄送：各部门", size=12.0),
                _para("正文从这里开始，以句号结尾。", size=12.0),
            ],
            [(0, 2)],
            None,
            None,
            None,
            id="walk-stops-at-table",
        ),
    ],
)
def test_imprint_neighbor_veto_walk_cases(
    records, expected_spans, control_records, control_kwargs, control_start
) -> None:
    cands = _find(records)
    assert [(c.start, c.end) for c in cands] == expected_spans
    if control_kwargs is not None:
        ctrl = _find(
            control_records if control_records is not None else records,
            **control_kwargs,
        )
        assert len(ctrl) == 1 and ctrl[0].start == control_start


# ---------------------------------------------------------------------------
# mixed table/paragraph covers: absorbable paragraphs join the table run
# (image-only, cover-material, paragraph-borne main title, paragraph tail)
# ---------------------------------------------------------------------------

_FORM_TABLE_ROWS = [
    [("档 号", 10.5, False), ("", None, False)],
    [("密 级", 10.5, False), ("公开", 10.5, False)],
]
_PUB_TABLE_ROWS = [[("某某电子股份有限公司", 16.0, False)]]


@pytest.mark.parametrize(
    ("records", "expected_span", "expected_members"),
    [
        pytest.param(
            # Pure <drawing/> between cover tables is absorbed in source order.
            [
                _table([[("产品标准化大纲", 22.0, False)]]),
                _para('<drawing id="1" name="图"/>', size=12.0),
                _table(_PUB_TABLE_ROWS),
                _para("正文从这里开始，以句号结尾。", size=12.0),
            ],
            (0, 3),
            (0, 1, 2),
            id="image-para-between-tables",
        ),
        pytest.param(
            # Paragraph carries the 22pt main title between two tables.
            [
                _table(_FORM_TABLE_ROWS),
                _para("某某管理办法", size=22.0),
                _table(_PUB_TABLE_ROWS),
                _para("正文从这里开始，以句号结尾。", size=12.0),
            ],
            (0, 3),
            (0, 1, 2),
            id="paragraph-carries-main-title",
        ),
        pytest.param(
            # Mixed <drawing/> + title text stands on semantic text length.
            [
                _table(_FORM_TABLE_ROWS),
                _para(
                    '<drawing id="1" name="logo" path="assets/very-long-path-to-'
                    'image-file-name-here.png"/> 某某管理办法',
                    size=22.0,
                ),
                _table(_PUB_TABLE_ROWS),
                _para("正文从这里开始，以句号结尾。", size=12.0),
            ],
            (0, 3),
            (0, 1, 2),
            id="mixed-image-and-title-para",
        ),
        pytest.param(
            # Cover-material paragraph with inline formula joins the cover.
            [
                _table([[("产品标准化大纲", 22.0, False)]]),
                _para("副标题 <equation>x</equation>", size=16.0),
                _table(_PUB_TABLE_ROWS),
                _para("正文从这里开始，以句号结尾。", size=12.0),
            ],
            (0, 3),
            (0, 1, 2),
            id="cover-material-para-with-formula",
        ),
        pytest.param(
            # Cover ending in a paragraph (no trailing table) stays one candidate.
            [
                _table(_FORM_TABLE_ROWS),
                _para("某某管理办法", size=22.0),
                _para("正文从这里开始，以句号结尾。", size=12.0),
            ],
            (0, 2),
            (0, 1),
            id="paragraph-tail-cover",
        ),
        pytest.param(
            # Trailing pure-image paragraph joins the cover members.
            [
                _table([[("产品标准化大纲", 22.0, False)]]),
                _para('<drawing id="9" name="印章"/>', size=12.0),
                _para("正文从这里开始，以句号结尾。", size=12.0),
            ],
            (0, 2),
            (0, 1),
            id="trailing-image-para-joins-cover",
        ),
        pytest.param(
            # Imprint-preceding signature line is barred from table absorption.
            [
                _table([[("产品标准化大纲", 22.0, False)]]),
                _para("某某市人民政府办公室", size=12.0),
                _para("抄送：各区人民政府", size=12.0),
                _para("正文从这里开始，以句号结尾。", size=12.0),
            ],
            (0, 1),
            (0,),
            id="imprint-preceding-line-not-absorbed",
        ),
    ],
)
def test_mixed_table_cover_absorption_cases(
    records, expected_span: tuple[int, int], expected_members: tuple[int, ...]
) -> None:
    cands = _find(records)
    assert len(cands) == 1
    c = cands[0]
    assert c.table
    assert (c.start, c.end) == expected_span
    assert c.members == expected_members


@pytest.mark.parametrize(
    ("middle_para", "kwargs", "expected_spans"),
    [
        pytest.param(
            _para("这是一段以句号结尾的正文。", size=12.0),
            {"imprint_boundary_indices": {1}},
            [(0, 1), (2, 3)],
            id="strong-body-para-splits-run",
        ),
        pytest.param(
            _para("第一章 总则", size=12.0, outline_level_raw=0),
            {},
            None,
            id="outline-heading-para-splits-run",
        ),
        pytest.param(
            _para('<drawing id="1" name="装饰"/> 第一章 总则', size=12.0),
            {},
            None,
            id="drawing-prefixed-numbered-heading-splits-run",
        ),
    ],
)
def test_mixed_table_cover_breaks_on_non_cover_para(
    middle_para, kwargs, expected_spans
) -> None:
    records = [
        _table([[("产品标准化大纲", 22.0, False)]]),
        middle_para,
        _table(_PUB_TABLE_ROWS),
        _para("正文从这里开始，以句号结尾。", size=12.0),
    ]
    cands = _find(records, **kwargs)
    assert all(1 not in c.members for c in cands)
    if expected_spans is not None:
        assert [(c.start, c.end) for c in cands] == expected_spans


@pytest.mark.parametrize(
    "title_text",
    [
        pytest.param("某某管理办法", id="plain-paragraph-main-title"),
        pytest.param(
            '某某<drawing id="1" name="纹章"/>管理办法',
            id="mixed-drawing-semantic-main-title",
        ),
    ],
)
def test_table_verdict_locates_absorbed_paragraph_main_title(title_text: str) -> None:
    records = [
        _table(_FORM_TABLE_ROWS),
        _para(title_text, size=22.0),
        _table(_PUB_TABLE_ROWS),
    ]
    candidate = TitleBlockCandidate(
        start=0,
        end=3,
        single=False,
        trigger="table_window",
        table=True,
        members=(0, 1, 2),
    )
    decision = _judge_with(
        {"is_title_block": True, "main_title": "某某管理办法"},
        records=records,
        candidate=candidate,
    )
    assert decision.is_title_block
    assert decision.member_indices == (0, 1, 2)


def test_llm_window_contains_no_image_placeholders() -> None:
    """Images are removed from the LLM window ENTIRELY (least noise, nothing
    for the LLM to echo back): a pure-image member contributes no window line,
    a mixed line reads as its clean semantic text — while both stay members
    so assembly keeps them in source order."""
    prompts: list[str] = []

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        prompts.append(prompt)
        return json.dumps(
            {"is_title_block": True, "main_title": "某某管理办法"},
            ensure_ascii=False,
        )

    # Table channel: pure-image member + mixed title member.
    records = [
        _table(_FORM_TABLE_ROWS),
        _para('<drawing id="9" name="印章" path="a/b.png"/>', size=12.0),
        _para('某某<drawing id="1" name="纹章"/>管理办法', size=22.0),
        _table(_PUB_TABLE_ROWS),
    ]
    candidate = TitleBlockCandidate(
        start=0,
        end=4,
        single=False,
        trigger="table_window",
        table=True,
        members=(0, 1, 2, 3),
    )
    decision = judge_title_block(candidate, records, _llm, warnings={})
    assert decision.is_title_block
    assert decision.member_indices == (0, 1, 2, 3)  # image kept as a member
    assert "<drawing" not in prompts[-1]
    assert "[IMAGE]" not in prompts[-1]
    assert "某某管理办法" in prompts[-1]  # mixed line reads clean

    # Paragraph channel (multi-window): same stripping + semantic locate; the
    # pure-image paragraph contributes no window line at all.
    records2 = [
        _para('某某<drawing id="1" name="纹章"/>管理办法', size=22.0),
        _para('<drawing id="9" name="印章"/>', size=12.0),
        _para("（2026年版）", size=12.0),
    ]
    cand2 = TitleBlockCandidate(start=0, end=3, single=False, trigger="multi_window")
    decision2 = judge_title_block(cand2, records2, _llm, warnings={})
    assert decision2.is_title_block
    assert "<drawing" not in prompts[-1]
    assert "[IMAGE]" not in prompts[-1]
    assert "某某管理办法" in prompts[-1]
    assert "（2026年版）" in prompts[-1]  # image line skipped, context intact
    # The unrendered image paragraph STAYS a member (source-order assembly);
    # dropping it would re-emit the image after the block's members.
    assert decision2.member_indices == (0, 1, 2)


# ---------------------------------------------------------------------------
# 版记 region model: detect_imprint_regions (抄送 anchor → 印发 closer, forward
# window, TOC/blank skip, structural boundary)
# ---------------------------------------------------------------------------


def _regions(records, **kw):
    kw.setdefault("imprint_marker", guardrails.imprint_marker_reason)
    kw.setdefault("imprint_closer", guardrails.imprint_closer_reason)
    return detect_imprint_regions(records, **kw)


@pytest.mark.parametrize(
    (
        "records",
        "kwargs",
        "expected_anchor",
        "expected_closer",
        "expected_members",
        "expected_preceding",
        "non_anchor_idx",
    ),
    [
        pytest.param(
            # 抄送 anchor closes on first 印发 inside 3-para forward window.
            [
                _para("抄送：各区人民政府", size=12.0),
                _para("中间行一", size=12.0),
                _para("某某办公室 2026年6月30日 印发", size=12.0),
            ],
            {},
            0,
            2,
            {0, 1, 2},
            None,
            None,
            id="closer-within-forward-window",
        ),
        pytest.param(
            # 印发 as 4th non-blank paragraph is out of window -> anchor alone.
            [
                _para("抄送：各区人民政府", size=12.0),
                _para("中间行一", size=12.0),
                _para("中间行二", size=12.0),
                _para("中间行三", size=12.0),
                _para("某某办公室 2026年 印发", size=12.0),
            ],
            {},
            0,
            None,
            {0},
            None,
            None,
            id="closer-beyond-window-falls-back",
        ),
        pytest.param(
            # N2: TOC lines (skip_indices) are stepped over without spending budget.
            [
                _para("上文", size=12.0),
                _para("目录行", size=12.0),
                _para("署名行", size=12.0),
                _para("抄送：各区", size=12.0),
                _para("目录行二", size=12.0),
                _para("某某 2026年 印发", size=12.0),
            ],
            {"skip_indices": {1, 4}},
            3,
            5,
            {3, 5},
            {0, 2},
            None,
            id="walks-skip-toc",
        ),
        pytest.param(
            # Non-paragraph record (table) stops forward walk -> anchor alone.
            [
                _para("抄送：各部门", size=12.0),
                _table([[("附件清单", 10.5, False)]]),
                _para("某某办公室 2026年 印发", size=12.0),
            ],
            {},
            0,
            None,
            {0},
            None,
            None,
            id="forward-walk-stops-at-table",
        ),
        pytest.param(
            # 印发机关 is a closer (never an anchor) that closes 抄送.
            [
                _para("抄送：各区", size=12.0),
                _para("印发机关　某某市人民政府办公厅", size=12.0),
            ],
            {},
            0,
            1,
            {0, 1},
            None,
            1,
            id="closer-is-yinfa-jiguan",
        ),
        pytest.param(
            # Mis-ordered 成文日期 right after 印发 closer is absorbed.
            [
                _para("抄送：各设区市城乡规划局", size=12.0),
                _para("河北省住房和城乡建设厅办公室   2009年7月6日印发", size=12.0),
                _para("二○○九年七月六日", size=12.0),
                _para("正文从这里开始，以句号结尾。", size=12.0),
            ],
            {},
            0,
            1,
            {0, 1, 2},
            None,
            None,
            id="absorbs-trailing-document-date",
        ),
        pytest.param(
            # Date walk stops at first non-date paragraph (later date stays out).
            [
                _para("抄送：各区", size=12.0),
                _para("某某办公室 2009年 印发", size=12.0),
                _para("二○○九年七月六日", size=12.0),
                _para("附件：", size=12.0),
                _para("2009年7月7日", size=12.0),
            ],
            {},
            0,
            1,
            {0, 1, 2},
            None,
            None,
            id="date-absorb-stops-at-non-date",
        ),
    ],
)
def test_imprint_region_detection_cases(
    records,
    kwargs,
    expected_anchor: int,
    expected_closer: int | None,
    expected_members: set[int],
    expected_preceding: set[int] | None,
    non_anchor_idx: int | None,
) -> None:
    if non_anchor_idx is not None:
        assert guardrails.imprint_marker_reason(records[non_anchor_idx].text) is None
    regs = _regions(records, **kwargs)
    assert len(regs) == 1
    r = regs[0]
    assert r.anchor == expected_anchor
    assert r.closer == expected_closer
    assert r.members == expected_members
    if expected_preceding is not None:
        assert r.preceding == expected_preceding


def test_imprint_region_start_marker_is_middle_content() -> None:
    """A 主题词-opened region runs THROUGH a following 抄送 (middle content, not
    a closer) to reach the 印发 closer; both anchors are vetoed."""
    records = [
        _para("主题词：经济 管理", size=12.0),
        _para("抄送：各区人民政府", size=12.0),
        _para("某某办公室 2026年 印发", size=12.0),
    ]
    by_anchor = {r.anchor: r for r in _regions(records)}
    assert by_anchor[0].closer == 2 and by_anchor[0].members == {0, 1, 2}
    assert by_anchor[1].closer == 2 and by_anchor[1].members == {1, 2}


def test_trailing_document_date_vetoed_from_next_cover() -> None:
    """End-to-end (mirrors test5-红头文件): the absorbed 成文日期 no longer seeds
    the following 附件 cover — the window starts at the real title, not the
    date. Injecting a no-op document_date restores the bug (date seeds)."""
    records = [
        _para("抄送：各设区市城乡规划局", size=12.0),  # 0 anchor
        _para("河北省住房和城乡建设厅办公室  2009年7月6日印发", size=12.0),  # 1 closer
        _para("二○○九年七月六日", size=12.0),  # 2 成文日期
        _para("附件：", size=12.0),  # 3
        _para("河北省城市控制性详细规划备案工作规程", size=22.0),  # 4 cover title
        _para("第一条 为规范……以句号结尾。", size=12.0),  # 5 body
    ]
    cands = _find(records)
    assert len(cands) == 1
    assert (cands[0].start, cands[0].end) == (3, 5)  # window skips the date (2)

    # Control: date not recognized → it seeds the window (the pre-fix bug).
    cands2 = _find(records, document_date=lambda t: False)
    assert cands2 and cands2[0].start == 2


def test_imprint_forward_region_vetoes_middle_single_candidate() -> None:
    """The middle line of a 抄送…印发 region is barred from the single-line
    channel; disabling the closer detector restores it as a candidate (proving
    the forward region — not strong_body — is what vetoes it)."""
    records = [
        _para("正文一，以句号结尾。", size=12.0),  # 0
        _para("抄送：各区人民政府", size=12.0),  # 1 anchor
        _empty(),  # 2
        _para("通知标题", size=18.0, alignment="center"),  # 3 middle (would-be)
        _empty(),  # 4
        _para("某某办公室2026年6月30日印发", size=12.0),  # 5 closer (trailing)
        _para("正文二，以句号结尾。", size=12.0),  # 6
    ]
    assert _find(records) == []
    # Control: no-op closer → region is anchor-only → idx 3 leads a candidate
    # (a tail window absorbing the 版记 material — exactly the bug being fixed).
    # A stub boundary index props the mid-document gate open (the anchor line
    # precedes the window) so the control isolates the forward-region veto.
    cands = _find(records, imprint_closer=lambda t: None, imprint_boundary_indices={1})
    assert len(cands) == 1 and cands[0].start == 3


# ---------------------------------------------------------------------------
# A 组：页/节边界断窗（title block 是单页单元）
# ---------------------------------------------------------------------------


def _ev(records, **kw):
    """_find + suppression ledger, returned as (candidates, events).

    ``head_zone_records=1`` pins the STRICT head-zone semantics (a window
    with ANY content record before it is mid-document) so these small
    synthetic shapes exercise the gate itself; the production default
    (DOCX_SMART_TITLE_HEAD_ZONE_RECORDS=8, corpus-calibrated so real covers
    behind a few leading tables/title lines stay head-zone) is covered by
    test_head_zone_default_tolerates_leading_cover_material."""
    ev: list = []
    kw.setdefault("head_zone_records", 1)
    cands = _find(records, suppressed_events=ev, **kw)
    return cands, ev


def test_page_break_splits_multi_window() -> None:
    """A1: a page break BEFORE a paragraph ends the open window; the two
    fragments are separate windows — the reseeded mid-doc pair reaches the
    position gate (suppressed, event visible) instead of joining the first."""
    records = [
        _para("正文内容以句号结尾。", size=12.0),
        _para("短行材料", size=12.0),
        _para("十六磅大标题行", size=16.0, page_break_before=True),
        _para("配套副行", size=12.0),
    ]
    cands, ev = _ev(records)
    assert cands == []
    assert [(e["trigger"], e["start"], e["end"]) for e in ev] == [
        ("multi_window", 2, 4)
    ]


def test_page_break_seed_still_opens_imprint_cover() -> None:
    """A2: the seed itself carrying a page break is NOT a boundary — a 公文汇编
    second cover opens on a page break right after the imprint tail."""
    records = [
        _para("正文一以句号结尾。", size=12.0),
        _para("抄送：各单位", size=12.0),
        _para("某某办公室 2026年6月30日 印发", size=12.0),
        _para("新文档大标题", size=18.0, page_break_before=True),
        _para("（2026年版）", size=12.0),
        _para("正文二以句号结尾。", size=12.0),
    ]
    cands, ev = _ev(records)
    assert [(c.start, c.end, c.trigger) for c in cands] == [(3, 5, "multi_window")]
    assert ev == []


def test_empty_para_page_break_splits_window() -> None:
    """A3: a page break carried on an EMPTY paragraph splits the window: the
    head cover stays intact, the post-break pair reseeds and hits the gate."""
    records = [
        _para("二十磅标题行", size=20.0),
        _para("副行材料", size=12.0),
        ParagraphRecord(
            kind="empty_para",
            has_page_break_run=True,
            has_leading_page_break_run=True,
        ),
        _para("另一大行", size=20.0),
        _para("配套副行", size=12.0),
        _para("正文以句号结尾。", size=12.0),
    ]
    cands, ev = _ev(records)
    assert [(c.start, c.end) for c in cands] == [(0, 2)]
    assert [(e["start"], e["end"]) for e in ev] == [(3, 5)]


def test_ends_section_breaks_window_after_member() -> None:
    """A4: a paragraph-level sectPr ends the window AFTER that member — the
    next-section line never joins it (a head-zone (0,3) window would form
    otherwise)."""
    records = [
        _para("二十磅标题行", size=20.0, ends_section=True),
        _para("次页大行", size=20.0),
        _para("副行", size=12.0),
        _para("正文以句号结尾。", size=12.0),
    ]
    cands, ev = _ev(records)
    # The lone pre-boundary line falls to the single channel — proof it never
    # joined the (1,3) window; that window reseeds and hits the gate.
    assert [(c.start, c.end, c.trigger) for c in cands] == [(0, 1, "single_line")]
    assert [(e["trigger"], e["start"], e["end"]) for e in ev] == [
        ("multi_window", 1, 3)
    ]


def test_table_run_resets_after_page_boundary() -> None:
    """A5a: a page boundary ends the table run BEFORE the boundary record and
    the remainder reseeds a NEW run (run_end must not swallow it) — here the
    second run carries attachment evidence, so BOTH become candidates."""
    records = [
        _table([[("产品标准化大纲", 22.0, False)]]),
        _para("档案编号 2026-001", size=12.0),
        _para("附件：", size=12.0, page_break_before=True),
        _table([[("附件封面标题", 22.0, False)]]),
        _para("正文以句号结尾。", size=12.0),
    ]
    cands, ev = _ev(records)
    assert [(c.start, c.end, c.members) for c in cands] == [
        (0, 2, (0, 1)),
        (3, 4, (3,)),
    ]
    assert ev == []


def test_table_run_reset_without_evidence_suppressed() -> None:
    """A5b: same split, but the post-boundary run has no imprint/attachment
    evidence — the first run stands, the second leaves a suppression event
    (visible, not silently dropped)."""
    records = [
        _table([[("产品标准化大纲", 22.0, False)]]),
        _para("档案编号 2026-001", size=12.0),
        _para("普通说明行", size=12.0, page_break_before=True),
        _table([[("第二个大标题表", 22.0, False)]]),
        _para("正文以句号结尾。", size=12.0),
    ]
    cands, ev = _ev(records)
    assert [(c.start, c.end) for c in cands] == [(0, 2)]
    assert [(e["trigger"], e["start"], e["end"]) for e in ev] == [
        ("table_window", 3, 4)
    ]


def test_multi_windows_on_both_sides_of_boundary() -> None:
    """A6: two covers split by a page break — the imprint-backed first window
    opens, the evidence-less second window suppresses (A/B interplay plus
    reset completeness in one shape)."""
    records = [
        _para("正文一以句号结尾。", size=12.0),
        _para("抄送：各单位", size=12.0),
        _para("某某办公室 2026年6月30日 印发", size=12.0),
        _para("汇编另一份文档标题", size=18.0, page_break_before=True),
        _para("（甲份副题行）", size=12.0),
        _para("再一份文档标题", size=18.0, page_break_before=True),
        _para("（乙份副题行）", size=12.0),
        _para("正文二以句号结尾。", size=12.0),
    ]
    cands, ev = _ev(records)
    assert [(c.start, c.end) for c in cands] == [(3, 5)]
    assert [(e["trigger"], e["start"], e["end"]) for e in ev] == [
        ("multi_window", 5, 7)
    ]


def test_leading_break_title_keeps_subtitle_in_window(tmp_path) -> None:
    """A7 (real DOCX): a leading w:br page inside the title run must count as
    ONE boundary (before the title), not two — the subtitle stays in the same
    window (asserted via the suppression event span; the pair is mid-document
    without evidence, so the gate rejects it as one unit). A second break
    AFTER the title text is a genuine trailing boundary: the subtitle is cut
    off and neither fragment can form a window."""
    from docx import Document
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt

    from lightrag.parser.docx.numbering_resolver import NumberingResolver
    from lightrag.parser.docx.parse_document import (
        _read_document_records,
        parse_styles_outline_levels,
    )
    from lightrag.parser.docx.smart_heading.features import parse_styles_attributes

    def _build(trailing_break: bool):
        doc = Document()
        for i in range(2):
            doc.add_paragraph(f"正文第{i}段，本段以句号结尾。").runs
        title = doc.add_paragraph()
        run = title.add_run("换页后的主标题")
        run.font.size = Pt(18)
        br = OxmlElement("w:br")
        br.set(qn("w:type"), "page")
        run._r.insert(0, br)
        if trailing_break:
            tail = OxmlElement("w:br")
            tail.set(qn("w:type"), "page")
            run._r.append(tail)
        doc.add_paragraph("配套副标题行")
        path = tmp_path / f"lead_{trailing_break}.docx"
        doc.save(str(path))
        doc2 = Document(str(path))
        return _read_document_records(
            doc2,
            NumberingResolver(str(path)),
            parse_styles_outline_levels(str(path)),
            None,
            {},
            style_attributes=parse_styles_attributes(str(path)),
        )

    records = _build(trailing_break=False)
    title_idx = next(i for i, r in enumerate(records) if "主标题" in r.text)
    assert records[title_idx].has_leading_page_break_run
    assert not records[title_idx].has_nonleading_page_break_run
    cands, ev = _ev(records)
    assert cands == []
    assert [(e["start"], e["end"]) for e in ev] == [(title_idx, title_idx + 2)]

    records2 = _build(trailing_break=True)
    title_idx2 = next(i for i, r in enumerate(records2) if "主标题" in r.text)
    assert records2[title_idx2].has_nonleading_page_break_run
    cands2, ev2 = _ev(records2)
    assert cands2 == [] and ev2 == []  # both fragments are lone lines


def test_table_para_adjacency_boundaries_split_runs() -> None:
    """A8: boundary evidence on the PARAGRAPH side of a table↔para adjacency
    still splits the run — table records carry default-False fields but must
    not shield their neighbours."""
    # ends_section on the absorbed paragraph: run keeps it, drops table 2.
    records = [
        _table([[("产品标准化大纲", 22.0, False)]]),
        _para("档案编号 2026-001", size=12.0, ends_section=True),
        _table([[("第二个大标题表", 22.0, False)]]),
        _para("正文以句号结尾。", size=12.0),
    ]
    cands, ev = _ev(records)
    assert [(c.start, c.end, c.members) for c in cands] == [(0, 2, (0, 1))]
    assert [(e["trigger"], e["start"]) for e in ev] == [("table_window", 2)]

    # pageBreakBefore on the paragraph: run ends at table 1 alone.
    records2 = [
        _table([[("产品标准化大纲", 22.0, False)]]),
        _para("档案编号 2026-001", size=12.0, page_break_before=True),
        _table([[("第二个大标题表", 22.0, False)]]),
        _para("正文以句号结尾。", size=12.0),
    ]
    cands2, ev2 = _ev(records2)
    assert [(c.start, c.end, c.members) for c in cands2] == [(0, 1, (0,))]
    assert [(e["trigger"], e["start"]) for e in ev2] == [("table_window", 2)]


# ---------------------------------------------------------------------------
# B 组：mid-document 位置门（形状不带换页，隔离 B 层）
# ---------------------------------------------------------------------------


def test_mid_document_windows_suppressed_without_evidence() -> None:
    """B1 (the test11 shape, sans page breaks): a mid-doc metadata line + big
    table caption pairs into a multi window AND a mixed table run — both are
    suppressed with scan-ordered events, and neither reaches the LLM."""
    records = [
        _para("正文内容以句号结尾。", size=12.0),
        _table([[("数据", 12.0, False)]]),
        _para("填报单位：某某公司", size=12.0),
        _para("外购外协价格明细表", size=16.0),
        _table([[("数据2", 12.0, False)]]),
    ]
    cands, ev = _ev(records)
    assert cands == []
    assert [(e["trigger"], e["start"], e["end"]) for e in ev] == [
        ("multi_window", 2, 4),
        ("table_window", 1, 5),
    ]


def test_imprint_tail_adjacency_opens_mid_document_window() -> None:
    """B2: an imprint tail right above (blank absorbed) opens the mid-doc
    window; disabling the closer detector removes the evidence and the same
    shape suppresses."""
    records = [
        _para("正文一以句号结尾。", size=12.0),
        _para("正文二以句号结尾。", size=12.0),
        _para("抄送：各区人民政府", size=12.0),
        _para("某某办公室 2026年6月30日 印发", size=12.0),
        _empty(),
        _para("新文档标题", size=18.0),
        _para("（副题行）", size=12.0),
        _para("正文三以句号结尾。", size=12.0),
    ]
    cands, ev = _ev(records)
    assert [(c.start, c.end) for c in cands] == [(5, 7)]
    assert ev == []
    cands2, ev2 = _ev(records, imprint_closer=lambda t: None)
    assert cands2 == []
    assert len(ev2) == 1


def test_head_zone_reaches_past_blank_toc_and_section_records() -> None:
    """B3: empties, TOC lines and section breaks before the first window do
    not end the head zone — the cover stays gate-free."""
    records = [
        _empty(),
        _para("目 录", size=12.0, is_toc_field=True),
        ParagraphRecord(kind="section_break"),
        _para("大标题", size=22.0),
        _para("副题行", size=12.0),
        _para("正文以句号结尾。", size=12.0),
    ]
    cands, ev = _ev(records)
    assert [(c.start, c.end) for c in cands] == [(3, 5)]
    assert ev == []


def test_logo_paragraph_keeps_table_cover_in_head_zone() -> None:
    """B4: a pure <drawing/> logo paragraph is decoration, not content — the
    table cover behind it still counts as the document head."""
    records = [
        _para('<drawing id="1" />', size=12.0, visible_char_count=0),
        _table([[("产品标准化大纲", 22.0, False)]]),
        _para("正文以句号结尾。", size=12.0),
    ]
    cands, ev = _ev(records)
    assert [(c.start, c.trigger) for c in cands] == [(1, "table_window")]
    assert ev == []


def test_mid_document_table_window_suppressed() -> None:
    """B5: a lone mid-doc title table is suppressed with a visible event."""
    records = [
        _para("正文内容以句号结尾。", size=12.0),
        _table([[("大标题表", 22.0, False)]]),
        _para("正文二以句号结尾。", size=12.0),
    ]
    cands, ev = _ev(records)
    assert cands == []
    assert [(e["trigger"], e["start"], e["end"]) for e in ev] == [
        ("table_window", 1, 2)
    ]


def test_attachment_opener_admits_mid_document_window() -> None:
    """B6: a bare 附件 marker line opening the window is document-boundary
    evidence; an attachment-like BODY phrase is not."""
    records = [
        _para("正文以句号结尾。", size=12.0),
        _para("附件：", size=12.0),
        _para("附件方案大标题", size=18.0),
        _para("（附件副题）", size=12.0),
        _para("正文二以句号结尾。", size=12.0),
    ]
    cands, ev = _ev(records)
    assert [(c.start, c.end) for c in cands] == [(1, 4)]
    assert ev == []

    records2 = [
        _para("正文以句号结尾。", size=12.0),
        _para("附件：见附表", size=12.0),
        _para("附件方案大标题", size=18.0),
        _para("（附件副题）", size=12.0),
        _para("正文二以句号结尾。", size=12.0),
    ]
    cands2, ev2 = _ev(records2)
    assert cands2 == []
    assert len(ev2) == 1


def test_imprint_single_line_candidate_after_boundary() -> None:
    """B7: the imprint tail absorbs page/section breaks and admits the SINGLE
    big line after it (user ruling: 版记 is strong boundary evidence, exempt
    from the >=2-member requirement)."""
    records = [
        _para("正文一以句号结尾。", size=12.0),
        _para("抄送：各区人民政府", size=12.0),
        _para("某某办公室 2026年6月30日 印发", size=12.0),
        ParagraphRecord(kind="section_break"),
        ParagraphRecord(
            kind="empty_para",
            has_page_break_run=True,
            has_leading_page_break_run=True,
        ),
        _para("新文档单行标题", size=18.0),
        _table([[("数据", 12.0, False)]]),
        _para("正文二以句号结尾。", size=12.0),
    ]
    cands, ev = _ev(records)
    assert [(c.start, c.end, c.single, c.trigger) for c in cands] == [
        (5, 6, True, "imprint_single")
    ]

    # Without the imprint: nothing (no channel admits the lone line).
    no_imprint = records[:1] + records[3:]
    cands2, _ = _ev(no_imprint)
    assert cands2 == []

    # A multi window already covering the line wins; no duplicate emission.
    covered = records[:6] + [_para("（副题行）", size=12.0)] + records[6:]
    cands3, _ = _ev(covered)
    assert [(c.start, c.trigger) for c in cands3] == [(5, "multi_window")]


def test_imprint_single_walk_skips_logo_paragraph() -> None:
    """B7-logo: a pure <drawing/> seal/logo between the imprint tail and the
    title is decoration — the cover after it is still found (here the logo
    joins the window as an absorbed member, exactly like a head-zone cover)."""
    records = [
        _para("正文一以句号结尾。", size=12.0),
        _para("抄送：各区人民政府", size=12.0),
        _para("某某办公室 2026年6月30日 印发", size=12.0),
        ParagraphRecord(
            kind="empty_para",
            has_page_break_run=True,
            has_leading_page_break_run=True,
        ),
        _para('<drawing id="1" />', size=12.0, visible_char_count=0),
        _para("新文档单行标题", size=18.0),
        _table([[("数据", 12.0, False)]]),
        _para("正文二以句号结尾。", size=12.0),
    ]
    cands, _ = _ev(records)
    title_idx = 5
    assert len(cands) == 1
    c = cands[0]
    assert c.start <= title_idx < c.end


def test_page_break_single_line_without_imprint_never_candidate() -> None:
    """B8: the original flow:1861 negative — a page break + one big line with
    NO imprint stays a plain paragraph (no candidate, no event: the lone
    fragment never forms a window)."""
    records = [
        _para("正文一以句号结尾。", size=12.0),
        _para("孤立大标题", size=18.0, page_break_before=True),
        _para("正文二以句号结尾。", size=12.0),
    ]
    cands, ev = _ev(records)
    assert cands == [] and ev == []


def test_attachment_opener_admits_following_table_cover() -> None:
    """B9: an attachment marker right above a title TABLE opens the table
    window (the marker is not members[0], so the preceding-record walk must
    recognize it); a plain phrase does not."""
    records = [
        _para("正文以句号结尾。", size=12.0),
        _para("附件：", size=12.0),
        _table([[("附件封面大标题", 22.0, False)]]),
        _para("正文二以句号结尾。", size=12.0),
    ]
    cands, ev = _ev(records)
    assert [(c.start, c.trigger) for c in cands] == [(2, "table_window")]
    assert ev == []

    records2 = [
        _para("正文以句号结尾。", size=12.0),
        _para("普通说明行", size=12.0),
        _table([[("附件封面大标题", 22.0, False)]]),
        _para("正文二以句号结尾。", size=12.0),
    ]
    cands2, ev2 = _ev(records2)
    assert cands2 == []
    assert len(ev2) == 1


def test_head_zone_default_tolerates_leading_cover_material() -> None:
    """Default head zone (8 content records): a cover window behind a couple
    of leading tables / long title lines (the test9 国标 / test13 报告 head
    shapes) still opens without boundary evidence; the same window behind a
    page's worth of content is mid-document and suppresses."""
    cover = [
        _table([[("GB", 10.5, False)]]),
        _para("ICS 35.240.20", size=10.5),
        _para("大标题封面行", size=22.0),
        _para("（副题行）", size=12.0),
        _para("正文从这里开始，以句号结尾。", size=12.0),
    ]
    ev: list = []
    cands = _find(cover, suppressed_events=ev)  # production default zone
    # Both channels open in the head zone: the GB label table absorbs the
    # cover paragraphs (table run) AND the paragraph pair forms a window.
    assert {(c.start, c.end, c.trigger) for c in cands} == {
        (0, 4, "table_window"),
        (1, 4, "multi_window"),
    }
    assert ev == []

    deep = [
        _para(f"垫层正文第{i}段，本段以句号结尾。", size=12.0) for i in range(8)
    ] + cover
    ev2: list = []
    cands2 = _find(deep, suppressed_events=ev2)
    assert cands2 == []
    assert [(e["trigger"], e["start"]) for e in ev2] == [
        ("multi_window", 9),
        ("table_window", 8),
    ]


def test_head_zone_closes_at_first_body_sentence_under_default() -> None:
    """Review regression (production default zone): a SHORT document whose
    top is ordinary body prose must not reopen the gate the record-count cap
    guards on long documents — the first body sentence closes the head zone,
    so the test11 shape behind only 3 body paragraphs still suppresses."""
    records = [_para(f"正文第{i}段，本段以句号结尾。", size=12.0) for i in range(3)] + [
        _para("填报单位：某某公司", size=12.0),
        _para("外购外协价格明细表", size=16.0),
    ]
    ev: list = []
    cands = _find(records, suppressed_events=ev)  # production default zone
    assert cands == []
    assert [(e["trigger"], e["start"], e["end"]) for e in ev] == [
        ("multi_window", 3, 5)
    ]


def test_head_zone_closes_at_body_data_table_under_default() -> None:
    """Review regression (production default zone): a DATA table — cells the
    cover-material gate rejects (full sentences) — is body evidence exactly
    like a sentence paragraph; the test11 shape right behind one must
    suppress even though fewer than 8 content records precede it."""
    records = [
        _table([[("本表第一格是完整的正文句子，明确以句号结尾。", 12.0, False)]]),
        _para("填报单位：某某公司", size=12.0),
        _para("外购外协价格明细表", size=16.0),
    ]
    ev: list = []
    cands = _find(records, suppressed_events=ev)  # production default zone
    assert cands == []
    assert [(e["trigger"], e["start"], e["end"]) for e in ev] == [
        ("multi_window", 1, 3)
    ]

    # Control: a COVER-shaped table (short label cells) is not body evidence —
    # the same window stays head-zone under the default.
    records2 = [
        _table([[("档 号", 10.5, False)]]),
        _para("填报单位：某某公司", size=12.0),
        _para("外购外协价格明细表", size=16.0),
    ]
    ev2: list = []
    cands2 = _find(records2, suppressed_events=ev2)
    # Head zone stays open: the paragraph window opens AND the cover table
    # absorbs the pair into a table window — both reach the LLM.
    assert {(c.start, c.end, c.trigger) for c in cands2} == {
        (0, 3, "table_window"),
        (1, 3, "multi_window"),
    }
    assert ev2 == []
