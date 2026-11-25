"""G5/G6 tests: candidate gate, size leveling, backfill, series alignment."""

from __future__ import annotations

import logging

import pytest

from lightrag.parser.docx.parse_document import ParagraphRecord
from lightrag.parser.docx.smart_heading.heading_flow import (
    HeadingDecision,
    align_numbering_series,
    anchor_outline_levels,
    assign_levels_by_size,
    backfill_top_level,
    close_unnumbered_level_gaps,
    demote_strong_body_headings,
    document_fs_base,
    gate_candidates,
    gate_with_cb1,
    nest_numbered_under_parent,
)
from lightrag.parser.docx.smart_heading.style_key import classify_numbering
from tests.parser.docx._smart_heading_helpers import (
    log_messages as _log_messages,
    make_body_paras as _body,
    make_decision as _make_decision,
    make_para as _para,
    stub_no_caption as _stub_no_caption,
    stub_no_veto as _stub_no_veto,
    stub_strong_body as _stub_strong_body,
)

pytestmark = pytest.mark.offline


def _no_title_llm(prompt: str, *, system_prompt: str | None = None) -> str:
    """Deterministic judge that says 'not a title block, everything is body'."""
    import json
    import re

    ids = [int(m) for m in re.findall(r"^\[(\d+)\]", prompt, re.M)]
    return json.dumps(
        {"is_title_block": False, "headings": [], "body": ids}, ensure_ascii=False
    )


def _gate(records, *, fs_base=None, **kw):
    indices = list(range(len(records)))
    fs = fs_base or document_fs_base(records, indices)
    kw.setdefault("strong_body", _stub_strong_body)
    kw.setdefault("numbering_veto", _stub_no_veto)
    kw.setdefault("caption_veto", _stub_no_caption)
    return gate_candidates(records, indices, fs_base=fs, **kw)


def _texts(result) -> list[str]:
    return [d.text for d in result.decisions]


# ---------------------------------------------------------------------------
# G5: candidate gate
# ---------------------------------------------------------------------------


def test_outline_candidate_survives_smaller_font() -> None:
    """G5-1: an outlineLvl paragraph at 10.5pt < FS_base=12 stays a candidate."""
    records = _body() + [_para("小字号样式标题", size=10.5, outline_level=0)]
    result = _gate(records)
    assert "小字号样式标题" in _texts(result)
    d = next(x for x in result.decisions if x.text == "小字号样式标题")
    assert d.rule_trail[0] == "outline"


@pytest.mark.parametrize("fmt", ["lowerLetter", "upperLetter"])
def test_automatic_alpha_keeps_roman_looking_ordinals(fmt):
    labels = [("i", 9), ("ii", 35), ("vv", 48), ("xx", 50)]
    records = _body() + [
        _para(
            f"{label.upper() if fmt == 'upperLetter' else label}. Heading",
            outline_level=0,
            numbering_format=fmt,
        )
        for label, _ in labels
    ]
    # A real, typed Roman companion must not promote the explicit alpha I.
    records.append(_para("III. Roman heading", outline_level=0))
    result = _gate(records)
    classifications = [d.numbering for d in result.decisions]
    assert [(c.style_key, c.ordinal) for c in classifications] == [
        ("EnAlpha", ordinal) for _, ordinal in labels
    ] + [("RomanNum", 3)]


def test_outline_strong_body_demotion_is_rule_tagged() -> None:
    """Review C2: an outline paragraph the recognition-time strong-body check
    demotes must leave a rule-tagged demoted decision (not a silent drop), so
    the I2 retention check sees an explicit demotion instead of a violation."""
    from lightrag.parser.docx.smart_heading.guardrails import (
        verify_baseline_heading_retention,
    )

    records = _body(20) + [
        _para("这段正文被误加了大纲级别，它以句号结束。", size=12.0, outline_level=1)
    ]
    result = _gate(records)
    # Not admitted as a heading candidate…
    assert "这段正文被误加了大纲级别，它以句号结束。" not in _texts(result)
    # …but recorded as an explicit, rule-tagged demotion.
    assert len(result.demoted) == 1
    dem = result.demoted[0]
    assert dem.record_index == len(records) - 1
    assert dem.is_heading is False
    assert "strong_body_demoted" in dem.rule_trail
    # Merging the demoted decisions (as run_smart_heading does) makes I2 pass.
    merged = list(result.decisions) + result.demoted
    assert verify_baseline_heading_retention(records, merged) == []


def test_outline_nlp_only_multisentence_is_spared_contextually() -> None:
    """An outline candidate beats an NLP-only multi-sentence vote, while the
    same text without outline and an outline paragraph with visible sentence
    punctuation retain the original strong-body demotion semantics."""

    def _multi(_text: str) -> str:
        return "strong_body_multi_sentence"

    spared_text = "2.3   投标单位廉洁自律承诺书"
    non_outline_text = "投标单位廉洁自律承诺书"
    explicit_text = "项目背景已经说明。后续要求继续执行"
    records = _body(20) + [
        _para(spared_text, size=14.0, outline_level=1),
        _para(non_outline_text, size=14.0),
        _para(explicit_text, size=14.0, outline_level=1),
    ]
    result = _gate(
        records,
        strong_body=_multi,
        numbering_veto=lambda _cls, text: (
            "homophone_ner_entity" if text == spared_text else None
        ),
    )

    spared = next(d for d in result.decisions if d.text == spared_text)
    assert spared.is_heading
    assert "outline_multisentence_spared" in spared.rule_trail
    assert "strong_body_demoted" not in spared.rule_trail

    demoted = {d.text: d for d in result.demoted}
    assert not demoted[non_outline_text].is_heading
    assert "strong_body_demoted" in demoted[non_outline_text].rule_trail
    assert not demoted[explicit_text].is_heading
    assert "strong_body_demoted" in demoted[explicit_text].rule_trail


def test_outline_other_strong_body_reasons_keep_full_force() -> None:
    """The outline guard is exact to multi-sentence: length and terminal
    punctuation remain direct, explicit demotion evidence."""
    from lightrag.parser.docx.smart_heading import guardrails

    length_text = "这是一段确实超过标题长度上限的正文内容" * 8
    terminal_text = "这段正文被误加了物理大纲级别，但仍然以句号结束。"
    records = _body(30) + [
        _para(length_text, size=14.0, outline_level=1),
        _para(terminal_text, size=14.0, outline_level=1),
    ]
    result = _gate(records, strong_body=guardrails.strong_body_reason)
    demoted = {d.text: d for d in result.demoted}

    assert "strong_body_length" in demoted[length_text].rule_trail
    assert "strong_body_sentence_end" in demoted[terminal_text].rule_trail
    assert all("strong_body_demoted" in d.rule_trail for d in demoted.values())


def test_cb1_projection_keeps_outline_multisentence_spared_candidates(
    monkeypatch,
) -> None:
    """CB1's scratch demotion uses the same outline guard: spared candidates
    remain in projected density instead of falsely recovering the overflow by
    projecting them all away."""

    def _multi_for_candidates(text: str) -> str | None:
        return "strong_body_multi_sentence" if text.startswith("候选标题") else None

    monkeypatch.setenv("DOCX_SMART_MIN_INTER_HEADING_CHARS", "0")
    records = _body(5) + [
        _para(f"候选标题{i}", size=14.0, outline_level=1) for i in range(5)
    ]
    fs = document_fs_base(records, range(len(records)))
    warnings: dict = {}
    result = gate_with_cb1(
        records,
        list(range(len(records))),
        fs_base=fs,
        warnings=warnings,
        strong_body=_multi_for_candidates,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )

    assert result.cb1_tripped
    assert not result.cb1_strong_body_recovered
    assert len(result.decisions) == 5
    assert all("outline_multisentence_spared" in d.rule_trail for d in result.decisions)


def test_post_merge_outline_nlp_only_multisentence_is_spared() -> None:
    """The same outline guard applies when numbered headings defer their
    strong-body judgment to the authoritative post-merge sweep."""

    def _multi(_text: str) -> str:
        return "strong_body_multi_sentence"

    spared = HeadingDecision(
        record_index=0,
        text="2.3 投标单位廉洁自律承诺书",
        is_heading=True,
        level=2,
        outline_level=1,
        numbering=classify_numbering("2.3 投标单位廉洁自律承诺书"),
    )
    explicit = HeadingDecision(
        record_index=1,
        text="2.4 项目背景已经说明。后续要求继续执行",
        is_heading=True,
        level=2,
        outline_level=1,
        numbering=classify_numbering("2.4 项目背景已经说明。后续要求继续执行"),
    )

    demote_strong_body_headings([spared, explicit], strong_body=_multi, warnings={})

    assert spared.is_heading
    assert spared.rule_trail == ["outline_multisentence_spared"]
    assert not explicit.is_heading
    assert "strong_body_demoted" in explicit.rule_trail


def _hd(
    idx: int,
    text: str,
    level: int,
    *,
    outline_level: int | None = None,
    is_title_block: bool = False,
    anchored: bool = False,
) -> HeadingDecision:
    return _make_decision(
        text,
        level,
        idx=idx,
        size=None,
        numbered=True,
        outline=outline_level,
        anchored=anchored,
        is_title_block=is_title_block,
    )


@pytest.mark.parametrize(
    "case_id,builder,expected_is_heading,subtree_idxs,non_subtree_strong_idxs,expected_subtree_warn",
    [
        pytest.param(
            "cascades_to_subtree",
            lambda: [
                _hd(0, "一、对节能汽车，减半征收车船税。", 1),
                _hd(1, "（一）减半征收车船税的节能乘用车应同时符合以下标准：", 2),
                _hd(2, "（二）减半征收车船税的节能商用车应同时符合以下标准：", 2),
                _hd(3, "二、对新能源车船，免征车船税。", 1),
            ],
            [False, False, False, False],
            (1, 2),
            (0, 3),
            2,
            id="cascades_to_subtree",
        ),
        pytest.param(
            "stops_at_outline_member",
            lambda: [
                _hd(0, "一、总体要求。", 1),
                _hd(1, "（一）明确目标", 2, outline_level=1),
                _hd(2, "1）具体分工", 3),
            ],
            [False, True, True],
            (),
            (0,),
            None,
            id="stops_at_outline_member",
        ),
        pytest.param(
            "outline_member_demoted_by_own_evidence",
            lambda: [
                _hd(0, "一、总体要求。", 1),
                _hd(1, "（一）本项要求已经明确。", 2, outline_level=1),
            ],
            [False, False],
            (),
            (0, 1),
            None,
            id="outline_member_demoted_by_own_evidence",
        ),
        pytest.param(
            "stops_at_title_block_boundary",
            lambda: [
                _hd(0, "一、总体要求。", 1),
                _hd(1, "第二篇 附件材料", 0, is_title_block=True),
                _hd(2, "（一）附件要点", 2),
            ],
            [False, True, True],
            (),
            (0,),
            None,
            id="stops_at_title_block_boundary",
        ),
        pytest.param(
            "follows_cb2_propagation",
            lambda: [
                _hd(0, "一、对节能汽车减半征收车船税。", 1),
                _hd(1, "（一）节能乘用车标准", 2),
                _hd(2, "二、对新能源车船免征车船税", 1),
                _hd(3, "（二）新能源汽车标准", 2),
            ],
            [False, False, False, False],
            (1, 3),
            (0, 2),
            2,
            id="follows_cb2_propagation",
        ),
        pytest.param(
            "leaves_unnumbered_child_to_gap_close",
            lambda: [
                _hd(0, "一、总体要求。", 1),
                _hd(1, "实施要点", 2),
            ],
            [False, True],
            (),
            (0,),
            None,
            id="leaves_unnumbered_child_to_gap_close",
        ),
        pytest.param(
            "multilevel_and_sibling_boundary",
            lambda: [
                _hd(0, "一、总体安排。", 1),
                _hd(1, "（一）职责分工", 2),
                _hd(2, "（1）牵头单位", 3),
                _hd(3, "第一章 概述", 1),
            ],
            [False, False, False, True],
            (1, 2),
            (0,),
            2,
            id="multilevel_and_sibling_boundary",
        ),
        pytest.param(
            "ignores_plain_anchored_child",
            lambda: [
                _hd(0, "一、对节能汽车，减半征收车船税。", 1),
                _hd(1, "（一）标准甲", 2, anchored=True),
            ],
            [False, False],
            (1,),
            (0,),
            1,
            id="ignores_plain_anchored_child",
        ),
    ],
)
def test_subtree_demotion_cases(
    case_id: str,
    builder,
    expected_is_heading: list[bool],
    subtree_idxs: tuple[int, ...],
    non_subtree_strong_idxs: tuple[int, ...],
    expected_subtree_warn: int | None,
) -> None:
    """Post-merge subtree demotion cascades to numbered descendants (including
    after CB2 propagation and ignoring plain size-anchored bookkeeping), stops
    at outline/title-block/sibling boundaries, and leaves unnumbered children."""
    decs = builder()
    warnings: dict = {}
    demote_strong_body_headings(decs, strong_body=_stub_strong_body, warnings=warnings)
    assert [d.is_heading for d in decs] == expected_is_heading
    for idx in subtree_idxs:
        assert "subtree_demoted" in decs[idx].rule_trail
        assert "strong_body_demoted" in decs[idx].rule_trail
        assert decs[idx].use_raw_text
    for idx in non_subtree_strong_idxs:
        assert "strong_body_demoted" in decs[idx].rule_trail
        assert "subtree_demoted" not in decs[idx].rule_trail
    for idx, d in enumerate(decs):
        if d.is_heading:
            assert "subtree_demoted" not in d.rule_trail
    if expected_subtree_warn is None:
        assert "smart_subtree_demotions" not in warnings
    else:
        assert warnings["smart_subtree_demotions"] == expected_subtree_warn


def test_subtree_demotion_keeps_outline_anchored_series_together() -> None:
    """Round-1 anchoring locks a whole numbered series to its outline.

    When at least one member is outlined, a no-outline member marked
    ``outline_anchored`` must not be split off
    by the parent's subtree demotion while its outlined siblings survive — else
    the series ends up part body and part heading, breaking same-series
    consistency. 1）has no outline; 2）/3）have outlines.
    """
    decs = [
        _hd(0, "一、总体安排。", 1),  # 句号 → hit
        _hd(1, "1）甲项内容", 2),  # EnSingleParen, no own outline
        _hd(2, "2）乙项内容", 2, outline_level=1),
        _hd(3, "3）丙项内容", 2, outline_level=1),
    ]
    anchor_outline_levels(decs, warnings={})
    assert decs[1].outline_anchored  # propagated from the series' outlined members
    warnings: dict = {}
    demote_strong_body_headings(decs, strong_body=_stub_strong_body, warnings=warnings)
    assert not decs[0].is_heading  # parent demoted by its own 句号
    assert [d.is_heading for d in decs[1:]] == [True, True, True]  # series intact
    assert "subtree_demoted" not in decs[1].rule_trail
    assert "smart_subtree_demotions" not in warnings


@pytest.mark.requires_spacy_models
def test_source_multisentence_gate_real_spacy_end_to_end(monkeypatch) -> None:
    """test11 regression through real NLP and the block assembler, now via the
    source-level multi-sentence gate in ``strong_body_reason`` (a spaCy ≥2 vote
    counts only WITH a visible internal terminator):

    - the outline-1 ``2.3   投标单位廉洁自律承诺书`` survives the model's
      dependency-ROOT pseudo split (承诺|书) and stays a level-2 heading — the
      pseudo split now yields ``strong_body_reason is None`` at the source, so it
      is never demoted (no ``outline_multisentence_spared`` detour needed);
    - a non-outline body lead-in (no visible terminator) is likewise not strong
      body and simply stays plain body content (it is body-font, so it is not
      promoted to a heading either);
    - an outline paragraph carrying a genuine internal ``。`` still trips
      ``strong_body_multi_sentence`` and is demoted.
    """
    import json

    from lightrag.parser.docx.parse_document import _assemble_blocks_smart
    from lightrag.parser.docx.smart_heading import guardrails
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")

    target = "2.3   投标单位廉洁自律承诺书"
    body_lead = (
        "为确保本项目产品的售后服务需求得以及时响应和解决，公司将组建该项目"
        "售后服务团队，具体负责人员事项如下表："
    )
    explicit = "项目背景已经说明。后续要求继续执行"

    # target/body_lead: spaCy pseudo-splits without any visible terminator →
    # the source gate returns None (previously target was a false multi_sentence
    # rescued only by the outline-context wrapper).
    assert not guardrails.has_explicit_internal_sentence_boundary(target)
    assert guardrails.strong_body_reason(target) is None
    assert guardrails.strong_body_reason(body_lead) is None
    # explicit: a genuine internal terminator corroborates the ≥2 vote.
    assert guardrails.has_explicit_internal_sentence_boundary(explicit)
    assert guardrails.strong_body_reason(explicit) == "strong_body_multi_sentence"

    records = _body(20) + [
        _para("2 商务文件", size=16.0, outline_level=0),
        *_body(5),
        _para("2.2   中标服务费承诺函", size=14.0, outline_level=1),
        *_body(5),
        _para(body_lead),  # body-font lead-in (a real one is not heading-sized)
        _para(target, size=14.0, outline_level=1),
        *_body(5),
        _para("2.4   公司实力", size=14.0, outline_level=1),
        *_body(5),
        _para(explicit, size=14.0, outline_level=1),
        *_body(5),
    ]

    def _llm(_prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps(
            {"is_title_block": False, "headings": [], "body": []},
            ensure_ascii=False,
        )

    warnings: dict = {}
    result = run_smart_heading(records, llm_judge=_llm, warnings=warnings)
    assert result is not None
    blocks = _assemble_blocks_smart(records, result, {}, {})
    # No spared detour anymore — the source gate handles it silently.
    assert "smart_outline_multisentence_spared" not in warnings

    target_decision = next(d for d in result.decisions.values() if d.text == target)
    assert target_decision.is_heading and target_decision.level == 2
    assert "homophone_ner_entity" in target_decision.rule_trail
    assert "strong_body_demoted" not in target_decision.rule_trail
    assert "outline_multisentence_spared" not in target_decision.rule_trail

    target_block = next(b for b in blocks if b.get("heading") == target)
    assert target_block["level"] == 2
    assert target_block["parent_headings"] == ["2 商务文件"]
    for sibling in ("2.2   中标服务费承诺函", "2.4   公司实力"):
        block = next(b for b in blocks if b.get("heading") == sibling)
        assert block["level"] == 2
        assert block["parent_headings"] == ["2 商务文件"]

    # body_lead is not strong and body-font → plain body, never a heading, and
    # not fetched from the demotion ledger (it produces no HeadingDecision).
    assert not any(d.text == body_lead for d in result.decisions.values())
    assert not any(b.get("heading") == body_lead for b in blocks)
    assert any(body_lead in (b.get("content") or "") for b in blocks)

    # explicit: genuine multi-sentence prose is still demoted.
    explicit_decision = next(d for d in result.decisions.values() if d.text == explicit)
    assert not explicit_decision.is_heading
    assert "strong_body_multi_sentence" in explicit_decision.rule_trail
    assert "strong_body_demoted" in explicit_decision.rule_trail


def test_non_outline_strong_body_demotion_is_audited_output_neutral() -> None:
    """Every recognition-time strong-body demotion is audited, not just the
    I2 outline ones: a NON-outline candidate that the check demotes leaves a
    rule-tagged demoted decision so the per-paragraph ledger matches the
    demotion counter — but stays output-neutral (``use_raw_text`` off, no
    physical outline, so it renders as plain body just like no decision)."""
    records = _body(30) + [
        # +4pt over the 12pt body → a size_strong candidate, but strong-body
        # (ends with a period) and NOT carrying any physical outline level.
        _para(
            "这是一段被字号误判为标题候选但其实是正文的内容，它以句号收尾。", size=16.0
        )
    ]
    warnings: dict = {}
    result = _gate(records, warnings=warnings)
    # Not admitted as a heading…
    assert (
        "这是一段被字号误判为标题候选但其实是正文的内容，它以句号收尾。"
        not in _texts(result)
    )
    # …but recorded as an explicit, rule-tagged, audited demotion.
    assert warnings.get("smart_strong_body_demotions") == 1
    assert len(result.demoted) == 1
    dem = result.demoted[0]
    assert dem.record_index == len(records) - 1
    assert dem.is_heading is False
    assert "strong_body_demoted" in dem.rule_trail
    # Non-outline: output-neutral (renders rec.text, not full_text_raw) and no
    # physical-outline weight, so it never touches the I2 retention check.
    assert dem.use_raw_text is False
    assert dem.outline_level is None


def test_numbering_veto_suppression_is_audited_output_neutral() -> None:
    """A paragraph whose numbering identity a veto revokes, and which then
    matches no promotion rule, is NOT silently dropped: it leaves a rule-tagged
    output-neutral ledger row carrying the veto reason so the audit records WHY
    a would-be numbered candidate became body."""

    def _veto_the_item(_cls, text):
        return "homophone_ner_entity" if text.startswith("1. ") else None

    records = _body(20) + [_para("1. 建立上岗人员培训制度", size=12.0)]
    result = _gate(records, numbering_veto=_veto_the_item)

    # Not admitted as a heading (its numbering was revoked, no other signal)…
    assert "1. 建立上岗人员培训制度" not in _texts(result)
    # …but recorded in the veto-suppression ledger, rule-tagged.
    assert len(result.veto_suppressed) == 1
    supp = result.veto_suppressed[0]
    assert supp.record_index == len(records) - 1
    assert supp.is_heading is False
    assert "homophone_ner_entity" in supp.rule_trail
    assert "numbering_veto_suppressed" in supp.rule_trail
    # Non-outline → output-neutral: renders as plain body, same as no decision.
    assert supp.use_raw_text is False
    assert supp.outline_level is None


@pytest.mark.parametrize(
    "case_id,extra_builder,expected_in,expected_out",
    [
        pytest.param(
            "strong_vs_isolated_weak",
            lambda: [
                _para("强信号标题", size=13.0),
                _para("孤立弱信号", size=12.5),
            ],
            ("强信号标题",),
            ("孤立弱信号",),
            id="strong_vs_isolated_weak",
        ),
        pytest.param(
            "weak_pair_with_companion",
            lambda: [
                _para("弱信号甲", size=12.5),
                *_body(3),
                _para("弱信号乙", size=12.5),
            ],
            ("弱信号甲", "弱信号乙"),
            (),
            id="weak_pair_with_companion",
        ),
        pytest.param(
            "weak_pair_ignores_strong_body_companion",
            lambda: [
                _para("孤立弱信号标题", size=12.5),
                *_body(3),
                _para("这是一句以句号结尾的同字号长正文。", size=12.5),
            ],
            (),
            ("孤立弱信号标题",),
            id="weak_pair_ignores_strong_body_companion",
        ),
        pytest.param(
            "base_size_needs_numbering_or_bold",
            lambda: [
                _para("一、成套编号甲", size=12.0),
                *_body(3),
                _para("二、成套编号乙", size=12.0),
                _para("整段加粗短语", size=12.0, all_bold=True),
                _para("普通同字号段落", size=12.0),
            ],
            ("一、成套编号甲", "二、成套编号乙", "整段加粗短语"),
            ("普通同字号段落",),
            id="base_size_needs_numbering_or_bold",
        ),
        pytest.param(
            "isolated_numbering_blocked_by_p2",
            lambda: [_para("一、只有这一个编号", size=12.0)],
            (),
            ("一、只有这一个编号",),
            id="isolated_numbering_blocked_by_p2",
        ),
        pytest.param(
            "smaller_than_base_never_heading_without_outline",
            lambda: [
                _para("一、小字号编号", size=10.5),
                _para("二、小字号编号乙", size=10.5),
            ],
            (),
            ("一、小字号编号", "二、小字号编号乙"),
            id="smaller_than_base_never_heading_without_outline",
        ),
    ],
)
def test_high_confidence_gate_admission_rules(
    case_id: str,
    extra_builder,
    expected_in: tuple[str, ...],
    expected_out: tuple[str, ...],
) -> None:
    """High-confidence FS_base gate rules: strong/weak size tiers, strong-body
    companion exclusion, base-size composite requirements, P2 isolation, and
    strict below-base rejection without outline."""
    records = _body(30) + extra_builder()
    texts = _texts(_gate(records))
    for t in expected_in:
        assert t in texts
    for t in expected_out:
        assert t not in texts
    if case_id == "smaller_than_base_never_heading_without_outline":
        assert texts == []


def test_low_confidence_disables_bare_size_advantage() -> None:
    """G3-3: mixed sizes (<60% dominant) → bare size > FS_base is not enough;
    composite paths (series / bold / centered companions) still work."""
    records = []
    records += _body(10, size=10.0)
    records += _body(8, size=12.0)
    records += _body(7, size=14.0)
    indices = list(range(len(records)))
    fs = document_fs_base(records, indices)
    assert not fs.confidence_high

    records.append(_para("纯字号优势标题", size=16.0))
    records.append(_para("一、复合信号标题", size=16.0))
    records += _body(2, size=10.0)
    records.append(_para("二、复合信号标题乙", size=16.0))
    result = _gate(records)
    texts = _texts(result)
    assert "纯字号优势标题" not in texts
    assert "一、复合信号标题" in texts and "二、复合信号标题乙" in texts


@pytest.mark.parametrize(
    "case_id,extra_builder,expected_in,expected_out",
    [
        pytest.param(
            "cross_run_companions_vs_adjacent_run",
            lambda: [
                _para("居中标题一", size=14.0, alignment="center"),
                *_body(3, size=10.0),
                _para("居中标题二", size=14.0, alignment="center"),
                *_body(3, size=10.0),
                _para("孤立居中行", size=14.0, alignment="center"),
                *_body(2, size=10.0),
                _para("连续居中甲", size=13.0, alignment="center"),
                _para("连续居中乙", size=13.0, alignment="center"),
            ],
            ("居中标题一", "居中标题二"),
            ("连续居中甲", "连续居中乙"),
            id="cross_run_companions_vs_adjacent_run",
        ),
        pytest.param(
            "globally_unique_isolated_rejected",
            lambda: [_para("全局唯一居中行", size=14.0, alignment="center")],
            (),
            ("全局唯一居中行",),
            id="globally_unique_isolated_rejected",
        ),
    ],
)
def test_low_confidence_centered_channel_rules(
    case_id: str,
    extra_builder,
    expected_in: tuple[str, ...],
    expected_out: tuple[str, ...],
) -> None:
    """Under low-confidence FS_base, centered lines require a cross-run
    companion: adjacent same-run pairs and globally unique lines are rejected."""
    records = _body(6, size=10.0) + _body(5, size=12.0) + _body(4, size=14.0)
    fs = document_fs_base(records, list(range(len(records))))
    assert not fs.confidence_high

    records += extra_builder()
    texts = _texts(_gate(records))
    for t in expected_in:
        assert t in texts
    for t in expected_out:
        assert t not in texts


@pytest.mark.parametrize(
    "case_id",
    [
        pytest.param(
            "poem_run_longer_than_four_loses_channel",
            id="poem_run_longer_than_four_loses_channel",
        ),
        pytest.param("space_padded_loses_channel", id="space_padded_loses_channel"),
        pytest.param(
            "padded_admitted_by_other_channel_uncentered",
            id="padded_admitted_by_other_channel_uncentered",
        ),
        pytest.param("solo_centered_line_admitted", id="solo_centered_line_admitted"),
        pytest.param(
            "two_line_centered_title_both_admitted",
            id="two_line_centered_title_both_admitted",
        ),
        pytest.param(
            "run_of_four_loses_channel_not_size_path",
            id="run_of_four_loses_channel_not_size_path",
        ),
    ],
)
def test_high_confidence_centered_channel_rules(case_id: str) -> None:
    """High-confidence centered channel: solo and 2-line titles admit, space
    padding strips centeredness (without vetoing other channels), and runs
    >=4 lose only the centered channel (preserving size_strong)."""
    from lightrag.parser.docx.smart_heading.guardrails import (
        CENTER_MAX_LEADING_PAD_EM,
    )

    records = _body(30)
    if case_id == "poem_run_longer_than_four_loses_channel":
        records += [
            _para(f"诗歌居中行{i}", size=12.0, alignment="center") for i in range(8)
        ]
        records += _body(3)
        records.append(_para("居中标题一", size=12.0, alignment="center"))
        records += _body(2)
        records.append(_para("居中标题二", size=12.0, alignment="center"))
        texts = _texts(_gate(records))
        assert all(f"诗歌居中行{i}" not in texts for i in range(8))
        assert "居中标题一" in texts and "居中标题二" in texts
    elif case_id == "space_padded_loses_channel":
        records.append(_para("孤立居中标题", size=12.0, alignment="center"))
        records += _body(3)
        records.append(
            _para(
                "某某市科学技术局",
                size=12.0,
                alignment="center",
                leading_pad_em=CENTER_MAX_LEADING_PAD_EM * 3,
            )
        )
        result = _gate(records)
        texts = _texts(result)
        assert "孤立居中标题" in texts
        assert "某某市科学技术局" not in texts
        assert next(x for x in result.decisions if x.text == "孤立居中标题").centered
    elif case_id == "padded_admitted_by_other_channel_uncentered":
        records.append(
            _para("一、工作安排", size=12.0, alignment="center", leading_pad_em=14.5)
        )
        records += _body(3)
        records.append(_para("二、联系方式", size=12.0, alignment="center"))
        by_text = {d.text: d for d in _gate(records).decisions}
        assert by_text["一、工作安排"].rule_trail[0] == "base_series"
        assert by_text["二、联系方式"].rule_trail[0] == "base_series"
        assert by_text["一、工作安排"].centered is False
        assert by_text["二、联系方式"].centered is True
    elif case_id == "solo_centered_line_admitted":
        records.append(_para("孤立居中标题", size=12.0, alignment="center"))
        result = _gate(records)
        d = next(x for x in result.decisions if x.text == "孤立居中标题")
        assert d.rule_trail[0] == "base_center"
    elif case_id == "two_line_centered_title_both_admitted":
        records.append(
            _para("广州市城市安全风险综合监测预警平台", size=12.0, alignment="center")
        )
        records.append(_para("建设的工作方案", size=12.0, alignment="center"))
        texts = _texts(_gate(records))
        assert "广州市城市安全风险综合监测预警平台" in texts
        assert "建设的工作方案" in texts
    else:
        records += [
            _para("居中簇一", size=12.0, alignment="center"),
            _para("居中簇二", size=12.0, alignment="center"),
            _para("大字号居中行", size=16.0, alignment="center"),
            _para("居中簇四", size=12.0, alignment="center"),
        ]
        result = _gate(records)
        texts = _texts(result)
        assert "居中簇一" not in texts
        assert "居中簇二" not in texts
        assert "居中簇四" not in texts
        d = next(x for x in result.decisions if x.text == "大字号居中行")
        assert d.rule_trail[0] == "size_strong"


def test_solo_centered_channel_exclusions() -> None:
    """The solo channel's safety exclusions: a centered 成文日期 (every
    format), a letter-free decoration line, and a caption never enter it."""
    from lightrag.parser.docx.smart_heading import guardrails

    excluded = (
        "二〇二六年七月六日",
        "2026.7.31",
        "2026/7/31",
        "2026-7-31",
        "- 1 -",
        "***",
        "——",
        "图 1 系统架构",
    )
    records = _body(30)
    for t in excluded:
        records.append(_para(t, size=12.0, alignment="center"))
        records += _body(2)  # separate runs: exclusion, not anti-poetry
    result = _gate(records, caption_veto=guardrails.caption_prefix_reason)
    texts = _texts(result)
    for t in excluded:
        assert t not in texts


def test_cb1_reestimation_recovers_density(monkeypatch, lightrag_caplog) -> None:
    """G3-4 branch 1: dominant fake-heading size folds into the body; the
    re-gated density collapses and the breaker does NOT trip."""
    records = _body(60, size=10.5)
    # 30 short 12pt lines with no composite signal — they clear the strong
    # +1.5pt tier at first, flooding density past 30%.
    for i in range(30):
        records.append(_para(f"伪标题短语{i}", size=12.0))
    indices = list(range(len(records)))
    fs = document_fs_base(records, indices)
    assert fs.confidence_high

    warnings: dict = {}
    with lightrag_caplog.at_level(logging.INFO, logger="lightrag"):
        result = gate_with_cb1(
            records,
            indices,
            fs_base=fs,
            warnings=warnings,
            strong_body=_stub_strong_body,
            numbering_veto=_stub_no_veto,
            caption_veto=_stub_no_caption,
        )
    assert warnings.get("smart_cb1_reestimated") == 1
    assert result.cb1_reestimated and not result.cb1_tripped
    # 12pt became the body baseline; the bare lines lost candidacy.
    assert result.decisions == []
    # Re-gate outcome logged as a converged (INFO) event.
    regate = next(m for m in _log_messages(lightrag_caplog) if "CB1 re-gate:" in m)
    assert "converged" in regate and "threshold" in regate


def test_cb1_extreme_sample_trips_breaker(monkeypatch, lightrag_caplog) -> None:
    """G3-4 branch 2: a question-bank doc stays dense after one
    re-estimation → the breaker trips (sub-document falls back)."""
    records = []
    for i in range(40):
        records.append(_para(f"{i + 1}. 选择题题干第{i}题", size=12.0))
        records.append(_para("A. 选项甲  B. 选项乙", size=10.5))
    indices = list(range(len(records)))
    fs = document_fs_base(records, indices)

    warnings: dict = {}
    with lightrag_caplog.at_level(logging.INFO, logger="lightrag"):
        result = gate_with_cb1(
            records,
            indices,
            fs_base=fs,
            warnings=warnings,
            strong_body=_stub_strong_body,
            numbering_veto=_stub_no_veto,
            caption_veto=_stub_no_caption,
        )
    assert result.cb1_reestimated
    assert result.cb1_tripped
    assert warnings.get("smart_cb1_tripped") == 1
    # Re-gate outcome logged as a still-over (WARNING) fallback event.
    regate = next(
        r for r in lightrag_caplog.records if "CB1 re-gate:" in r.getMessage()
    )
    assert regate.levelno == logging.WARNING
    assert "outline-only fallback" in regate.getMessage()


# ---------------------------------------------------------------------------
# CB1 graduated demotion peels same-size candidates one evidence tier
# at a time instead of the blanket re-estimation wiping every same-size heading.
# ---------------------------------------------------------------------------

# ~70-CJK body paragraph — long enough that a couple between candidates keeps
# average inter-heading spacing above DOCX_SMART_MIN_INTER_HEADING_CHARS (200),
# so density (not sparsity) is the binding CB1 constraint in these fixtures.
_GRAD_BODY = (
    "本段为用于支撑基准字号统计权重并拉开相邻标题间距的正文内容占位文字总长约七十"
    "余个汉字确保标题间平均正文字符数稳定高于阈值不触发稀疏条件仅让密度成为约束若干"
)


def _grad_body_para() -> ParagraphRecord:
    return _para(_GRAD_BODY, size=12.0)


def _gbt_shape_records(
    *, ennum_root: bool = True, strong_body_clauses: int = 0
) -> list[ParagraphRecord]:
    """A GB/T-shaped sub-document: same-size (12pt = FS_base) numbered headings
    of several tiers, spread among long body so density is ~0.49 (over the 0.35
    bar) but spacing is healthy. ``ennum_root`` controls whether the EnNum
    ordinals cover the MultiLevelNum leading components (making EnNum the
    family root) or not.

    ``strong_body_clauses`` adds N CnClause candidates whose text ends in a
    period, so they are admitted first-pass (clause defers strong-body to the
    post-merge sweep) but removed by the look-ahead projection — used to force
    ``density`` (real) and ``cb1_graduated_density`` (projection) apart.
    """
    cands: list[ParagraphRecord] = []
    ennum_labels = range(1, 11) if ennum_root else range(101, 111)
    for n in ennum_labels:  # 10 EnNum "1 …" (root parents 5.1 / 5.1.1 when 1..10)
        cands.append(_para(f"{n} 概述章节{n}", size=12.0))
    for i in range(8):  # 8 MultiLevelNum raw-level 2, tops 1..8
        cands.append(_para(f"{i + 1}.1 子节标题{i}", size=12.0))
    for i in range(14):  # 14 MultiLevelNum raw-level 3, tops cycle 1..7
        cands.append(_para(f"{(i % 7) + 1}.1.{i + 1} 细则条目{i}", size=12.0))
    for i in range(6):  # 6 EnSingleParen list items "a) …"
        cands.append(_para(f"a) 列表项目内容{i}", size=12.0))
    for i in range(strong_body_clauses):  # strong-body CnClause (body-in-disguise)
        cands.append(
            _para(
                f"第{'一二三四五六'[i]}条 本条为伪装成标题的正文内容以句号结尾说明。",
                size=12.0,
            )
        )
    recs = [_grad_body_para(), _grad_body_para()]
    for c in cands:
        recs.append(c)
        recs.append(_grad_body_para())
    return recs


def _grad_gate(records, warnings):
    idx = list(range(len(records)))
    return gate_with_cb1(
        records,
        idx,
        fs_base=document_fs_base(records, idx),
        warnings=warnings,
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )


@pytest.mark.parametrize(
    "case_id",
    [
        pytest.param(
            "recovers_same_size_numbered_headings",
            id="recovers_same_size_numbered_headings",
        ),
        pytest.param(
            "ennum_root_not_demoted_before_mln",
            id="ennum_root_not_demoted_before_mln",
        ),
        pytest.param("sparse_only_trigger", id="sparse_only_trigger"),
        pytest.param(
            "exhausted_falls_back_to_blanket",
            id="exhausted_falls_back_to_blanket",
        ),
    ],
)
def test_cb1_graduated_gate_scenarios(lightrag_caplog, case_id: str) -> None:
    """CB1 graduated demotion peels weak same-size tiers, respects EnNum-root
    deferral, handles spacing-only triggers, and falls back cleanly when exhausted."""
    warnings: dict = {}
    if case_id == "recovers_same_size_numbered_headings":
        records = _gbt_shape_records(ennum_root=True)
        with lightrag_caplog.at_level(logging.INFO, logger="lightrag"):
            result = _grad_gate(records, warnings)
        assert not result.cb1_reestimated
        assert not result.cb1_tripped
        assert not result.cb1_strong_body_recovered
        assert result.cb1_graduated_stages == ["en_single_paren", "mln_raw3"]
        assert result.cb1_graduated_ennum_root is True
        assert result.cb1_graduated_demoted == 20
        styles = sorted(d.numbering.style_key for d in result.decisions)
        assert styles == ["EnNum"] * 10 + ["MultiLevelNum"] * 8
        demoted = [d for d in result.demoted if "cb1_graduated_demoted" in d.rule_trail]
        assert len(demoted) == 20
        for d in demoted:
            assert not d.is_heading
            assert d.use_raw_text is False
            assert any(t.startswith("cb1_stage:") for t in d.rule_trail)
            assert d.rule_trail[0] in {"base_series", "base_bold", "base_center"}
        assert warnings["smart_cb1_graduated_demotions"] == 20
        assert result.density == pytest.approx(18 / 78, abs=1e-4)
        assert result.cb1_graduated_density == pytest.approx(18 / 78, abs=1e-4)
        assert result.cb1_graduated_inter_chars is not None
        assert result.cb1_graduated_inter_chars >= 200
        assert any("CB1 graduated" in m for m in _log_messages(lightrag_caplog))
    elif case_id == "ennum_root_not_demoted_before_mln":
        records = _gbt_shape_records(ennum_root=False)
        result = _grad_gate(records, warnings)
        assert result.cb1_graduated_ennum_root is False
        assert result.cb1_graduated_stages == ["en_single_paren", "en_num"]
        styles = sorted(d.numbering.style_key for d in result.decisions)
        assert styles == ["MultiLevelNum"] * 22
    elif case_id == "sparse_only_trigger":
        records = [_grad_body_para(), _grad_body_para()]
        for i in range(4):
            records += [
                _para(f"{i + 1}.1 子节标题{i}", size=12.0),
                _grad_body_para(),
                _grad_body_para(),
            ]
        records.append(_grad_body_para())
        for i in range(6):
            records.append(_para(f"a) 列表项{i}", size=12.0))
        records += [_grad_body_para(), _grad_body_para()]
        result = _grad_gate(records, warnings)
        assert not result.cb1_reestimated and not result.cb1_tripped
        assert result.cb1_graduated_stages == ["en_single_paren"]
        assert result.cb1_graduated_density is not None
        assert result.cb1_graduated_density <= 0.35
        assert result.cb1_graduated_inter_chars >= 200
        assert sorted(d.numbering.style_key for d in result.decisions) == (
            ["MultiLevelNum"] * 4
        )
    else:
        records = [_para(_GRAD_BODY, size=10.5) for _ in range(20)]
        for i in range(30):
            records.append(_para(f"大纲标题{i}", size=10.5, outline_level=0))
        for i in range(10):
            records.append(_para(f"a) 列表项{i}", size=10.5))
        result = _grad_gate(records, warnings)
        assert result.cb1_graduated_stages == []
        assert result.cb1_graduated_demoted == 0
        assert "smart_cb1_graduated_demotions" not in warnings
        assert result.cb1_reestimated
        assert not any("cb1_graduated_demoted" in d.rule_trail for d in result.demoted)


def test_cb1_graduated_density_reflects_real_decisions_not_projection() -> None:
    """``result.density`` keeps its real-decisions definition; the projection
    basis is a SEPARATE field. Strong-body clause candidates are admitted
    first-pass (so they stay in ``decisions``) but removed by the look-ahead
    projection — so the two densities genuinely diverge, and writing the
    projection value back to ``density`` would be caught here. The authoritative
    post-merge sweep then demotes the clauses, converging the final heading
    count on the projection."""
    records = _gbt_shape_records(ennum_root=True, strong_body_clauses=4)
    warnings: dict = {}
    result = _grad_gate(records, warnings)

    # At the 0.40 floor the ladder converges after peeling the single weakest
    # tier (en_single_paren); the deeper mln_raw3 tier is no longer needed.
    assert result.cb1_graduated_stages == ["en_single_paren"]
    # The 4 strong-body clauses survive the first-pass gate (clause defers
    # strong-body) and are NOT touched by the ladder, so the real ratio counts
    # them while the projection ratio does not — a strict inequality.
    clauses = [
        d
        for d in result.decisions
        if d.numbering is not None and d.numbering.style_key == "CnClause"
    ]
    assert len(clauses) == 4 and all(d.is_heading for d in clauses)
    assert result.cb1_graduated_density is not None
    assert result.density > result.cb1_graduated_density

    # The authoritative post-merge sweep demotes the body-in-disguise clauses;
    # the final heading count matches the projection the ladder decided on.
    ds = result.decisions
    assign_levels_by_size(ds)
    demote_strong_body_headings(ds, strong_body=_stub_strong_body, warnings=warnings)
    assert not any(
        d.numbering is not None and d.numbering.style_key == "CnClause" and d.is_heading
        for d in ds
    )
    final_headings = sum(1 for d in ds if d.is_heading)
    assert final_headings == round(result.cb1_graduated_density * result.non_empty)


def test_cb1_graduated_ladder_and_root_helpers() -> None:
    """The ladder orders weakest body-prior first, then MLN deepest-first,
    deferring EnNum to the end only when >=2 overlapping ordinals cover >=half
    of the MLN leading components."""
    from lightrag.parser.docx.smart_heading.heading_flow import (
        _cb1_ennum_is_mln_root,
        _cb1_ladder,
    )

    present = {
        "en_single_paren": [],
        "en_num": [],
        "cn_num": [],
        "mln_raw2": [],
        "mln_raw3": [],
    }
    assert _cb1_ladder(present, ennum_root=False) == [
        "en_single_paren",
        "en_num",
        "cn_num",
        "mln_raw3",
        "mln_raw2",
    ]
    assert _cb1_ladder(present, ennum_root=True) == [
        "en_single_paren",
        "cn_num",
        "mln_raw3",
        "mln_raw2",
        "en_num",
    ]

    mlns = [_make_decision(f"{t}.1 子节", idx=t, numbered=True) for t in (1, 2, 3, 4)]

    def _en(ordinal: int) -> HeadingDecision:
        return _make_decision(f"{ordinal} 章节", idx=1000 + ordinal, numbered=True)

    assert _cb1_ennum_is_mln_root(mlns, [_en(o) for o in (1, 2, 3, 4)]) is True
    assert _cb1_ennum_is_mln_root(mlns, [_en(1)]) is False  # single {1} overlap
    assert _cb1_ennum_is_mln_root(mlns, [_en(o) for o in (90, 91)]) is False  # none
    assert _cb1_ennum_is_mln_root([], [_en(1), _en(2)]) is False  # no MLN family


def test_cb1_graduated_outline_and_larger_size_exempt() -> None:
    """Outline headings and candidates whose size is above FS_base are never
    demoted by the ladder — only same-size non-outline tiers are."""
    from lightrag.parser.docx.smart_heading.heading_flow import _cb1_stage_tag

    outline = HeadingDecision(
        record_index=0,
        text="1 大纲",
        is_heading=True,
        font_size_pt=12.0,
        outline_level=0,
        numbering=classify_numbering("1 大纲"),
    )
    outline.note("outline")
    bigger = HeadingDecision(
        record_index=1,
        text="1) 列表",
        is_heading=True,
        font_size_pt=14.0,
        numbering=classify_numbering("1) 列表"),
    )
    bigger.note("size_strong")
    same_size = HeadingDecision(
        record_index=2,
        text="a) 列表",
        is_heading=True,
        font_size_pt=12.0,
        numbering=classify_numbering("a) 列表"),
    )
    same_size.note("base_series")

    assert _cb1_stage_tag(outline, 12.0) is None
    assert _cb1_stage_tag(bigger, 12.0) is None
    assert _cb1_stage_tag(same_size, 12.0) == "en_single_paren"


def test_cb1_graduated_flows_through_downstream_leveling() -> None:
    """After graduated demotion the kept candidates are ordinary headings: the
    normal leveling / backfill pipeline runs over them without incident and the
    EnNum root is promoted to the shallowest level."""
    records = _gbt_shape_records(ennum_root=True)
    warnings: dict = {}
    result = _grad_gate(records, warnings)

    ds = result.decisions
    assign_levels_by_size(ds)
    backfill_top_level(ds, warnings=warnings)
    align_numbering_series(ds)
    assert ds  # non-empty
    assert all(d.is_heading and d.level is not None and d.level >= 1 for d in ds)
    root_levels = [d.level for d in ds if d.text.startswith("1 概述")]
    assert root_levels and min(d.level for d in ds) in root_levels


def test_cb1_graduated_audit_fields(monkeypatch) -> None:
    """run_smart_heading surfaces the graduated-demotion audit block."""
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    monkeypatch.setenv("DOCX_SMART_SUBDOC_MIN_TOKENS", "10")
    records = _gbt_shape_records(ennum_root=True)

    warnings: dict = {}
    result = run_smart_heading(
        records,
        llm_judge=None,
        warnings=warnings,
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    subs = result.audit["sub_documents"]
    grad = next(s["cb1_graduated"] for s in subs if "cb1_graduated" in s)
    assert grad["stages_applied"] == ["en_single_paren", "mln_raw3"]
    assert grad["ennum_root"] is True
    assert grad["demoted_count"] == 20
    assert grad["projected_density"] == pytest.approx(18 / 78, abs=1e-3)
    assert grad["projected_inter_chars"] >= 200


# ---------------------------------------------------------------------------
# G6: leveling / backfill / alignment
# ---------------------------------------------------------------------------


def _decision(
    text: str,
    *,
    size: float | None = None,
    centered: bool = False,
    numbered: bool = True,
    idx: int = 0,
) -> HeadingDecision:
    return _make_decision(
        text, size=size, centered=centered, numbered=numbered, idx=idx
    )


@pytest.mark.parametrize(
    "case_id,specs,expected_levels",
    [
        pytest.param(
            "consecutive_size_bands",
            [
                ("总览", 16.0, False, True),
                ("第一章 分述", 14.0, True, False),
                ("一、细项", 12.0, True, False),
            ],
            [1, 2, 3],
            id="consecutive_size_bands",
        ),
        pytest.param(
            "same_band_centered_numbered_uncentered",
            [
                ("居中无编号", 14.0, False, True),
                ("一、编号标题", 14.0, True, False),
                ("非居中无编号", 14.0, False, False),
            ],
            [1, 2, 3],
            id="same_band_centered_numbered_uncentered",
        ),
        pytest.param(
            "step2_spec_example_table",
            [
                ("居中无编号标题", 14.0, False, True),
                ("1.2 多级编号", 14.0, True, False),
                ("一、中文数字", 14.0, True, False),
                ("2.3 多级编号", 14.0, True, False),
                ("1.1.2 多级编号", 14.0, True, False),
                ("（一）中文括号", 14.0, True, False),
            ],
            [1, 2, 3, 2, 3, 4],
            id="step2_spec_example_table",
        ),
    ],
)
def test_assign_levels_by_size_cases(
    case_id: str,
    specs: list[tuple[str, float, bool, bool]],
    expected_levels: list[int],
) -> None:
    """G6 size banding assigns consecutive levels across bands and
    centered < numbered < uncentered within one band."""
    ds = [
        _decision(text, size=size, numbered=numbered, centered=centered, idx=i)
        for i, (text, size, numbered, centered) in enumerate(specs)
    ]
    assign_levels_by_size(ds)
    assert [d.level for d in ds] == expected_levels


@pytest.mark.parametrize(
    "case_id,rows,size",
    [
        pytest.param(
            "step4_spec_example_table",
            [
                ("居中无编号标题", 2, 2, False),
                ("1.2 节", 3, 3, True),
                ("一、目标", 4, 4, True),
                ("二、方法", 5, 4, True),
                ("1.1.2 小节", 4, 4, True),
                ("1.3 节", 4, 3, True),
                ("三、总结", 3, 3, True),
                ("（一）分项", 5, 4, True),
                ("（二）分项", 4, 4, True),
            ],
            14.0,
            id="step4_spec_example_table",
        ),
        pytest.param(
            "alignment_shifts_subtree",
            [
                ("一、第一部分", 4, 4, True),
                ("子标题甲", 5, 5, False),
                ("二、第二部分", 5, 4, True),
                ("子标题乙", 6, 5, False),
                ("孙标题", 7, 6, False),
            ],
            None,
            id="alignment_shifts_subtree",
        ),
    ],
)
def test_align_numbering_series_cases(
    case_id: str,
    rows: list[tuple[str, int, int, bool]],
    size: float | None,
) -> None:
    """Same-series alignment uses modal level with shallow tie-break and shifts
    the aligned heading's subtree along with it."""
    ds = []
    for i, (text, level, _expected, numbered) in enumerate(rows):
        d = _decision(text, size=size, numbered=numbered, idx=i)
        d.level = level
        ds.append(d)
    align_numbering_series(ds)
    assert [d.level for d in ds] == [r[2] for r in rows]


@pytest.mark.parametrize(
    "case_id,specs,promoted_idxs,expected_levels,expected_styles,expected_warnings",
    [
        pytest.param(
            "absorbs_ennum_with_ordinal_linkage",
            [
                ("1. 总则", 14.0, 2),
                ("1.1 范围", 12.0, 3),
                ("1.2 定义", 12.0, 3),
                ("2. 要求", 14.0, 2),
                ("2.1 基本要求", 12.0, 3),
                ("第一章 干扰章节", 14.0, 2),
            ],
            (0, 3),
            {0: 2, 3: 2},
            {5: "CnChapter"},
            {},
            id="absorbs_ennum_with_ordinal_linkage",
        ),
        pytest.param(
            "linkage_waives_size_threshold",
            [
                ("1. 总则", 10.5, 4),
                ("1.1 范围", 12.0, 3),
                ("1.2 定义", 12.0, 3),
                ("2. 要求", 10.5, 4),
                ("2.1 基本要求", 12.0, 3),
            ],
            (0, 3),
            {0: 2, 3: 2},
            {},
            {},
            id="linkage_waives_size_threshold",
        ),
        pytest.param(
            "single_linkage_pair_skipped_with_warning",
            [
                ("1. 总则", 10.5, 4),
                ("1.1 范围", 12.0, 3),
                ("1.2 定义", 12.0, 3),
                ("2. 附则", 10.5, 4),
            ],
            (),
            {},
            {0: "EnNum"},
            {"smart_backfill_single_linkage": 1},
            id="single_linkage_pair_skipped_with_warning",
        ),
        pytest.param(
            "counter_example_blocks_linkage",
            [
                ("1. 总则", 10.5, 4),
                ("2.1 错位小节", 12.0, 3),
                ("2. 要求", 10.5, 4),
                ("2.2 又一节", 12.0, 3),
            ],
            (),
            {},
            {0: "EnNum"},
            {},
            id="counter_example_blocks_linkage",
        ),
        pytest.param(
            "only_promotes_top_level_roots",
            [
                ("1. 价格文件", 10.5, 4),
                ("1.1 开标一览表", 14.0, 3),
                ("2. 商务文件", 10.5, 4),
                ("2.1 投标函", 14.0, 3),
                ("1. 我方按招标文件递交投标文件正本。", 10.5, 6),
                ("2. 我方承认招标人有权决定中标人。", 10.5, 6),
            ],
            (0, 2),
            {},
            {4: "EnNum", 5: "EnNum"},
            {},
            id="only_promotes_top_level_roots",
        ),
        pytest.param(
            "noop_without_mln",
            [
                ("1. 总则", 14.0, 2),
                ("2. 附则", 14.0, 2),
            ],
            (),
            {},
            {0: "EnNum", 1: "EnNum"},
            {},
            id="noop_without_mln",
        ),
    ],
)
def test_backfill_top_level_cases(
    case_id: str,
    specs: list[tuple[str, float, int]],
    promoted_idxs: tuple[int, ...],
    expected_levels: dict[int, int],
    expected_styles: dict[int, str],
    expected_warnings: dict[str, int],
) -> None:
    """backfill_top_level promotes only genuine ordinal-linked top-level roots
    (waiving size threshold with >=2 links), skips single/refuted links and
    MLN-free docs, and never lifts colliding nested body lists."""
    ds = []
    for i, (text, size, level) in enumerate(specs):
        d = _decision(text, size=size, idx=i)
        d.level = level
        ds.append(d)
    warnings: dict = {}
    backfill_top_level(ds, warnings=warnings)

    for idx in promoted_idxs:
        assert ds[idx].numbering.style_key == "MultiLevelNum"
        assert ds[idx].numbering.raw_level == 1
        assert "backfill_top_level" in ds[idx].rule_trail
    for idx in range(len(ds)):
        if idx not in promoted_idxs:
            assert "backfill_top_level" not in ds[idx].rule_trail
    assert sum("backfill_top_level" in d.rule_trail for d in ds) == len(promoted_idxs)
    for idx, level in expected_levels.items():
        assert ds[idx].level == level
    for idx, style in expected_styles.items():
        assert ds[idx].numbering.style_key == style
    for key, count in expected_warnings.items():
        assert warnings.get(key) == count


def test_outlined_part_root_survives_end_to_end(monkeypatch) -> None:
    """test11 shape end to end (through the assembler, where parent_headings
    is generated): outline-0 EnNum part roots with same-key body lists under
    their sub-sections must SURVIVE as headings — the body lists demote, the
    roots do not. Sizes are cleanly banded (root 16 > child 14 > body 12) so
    the nesting is deterministic; the same-size test11 shape is covered by the
    real-doc rerun in verification."""
    import json

    from lightrag.parser.docx.parse_document import _assemble_blocks_smart
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")

    def _sents(n, p):
        return [
            _para(
                f"{p}第{i}段正常长度的正文内容，用来撑起基准字号统计，本段以句号结尾。",
                size=12.0,
            )
            for i in range(n)
        ]

    records = (
        _sents(15, "引")
        + [_para("1. 价格文件", size=16.0, outline_level=0)]
        + _sents(6, "价")
        + [_para("1.1 开标一览表", size=14.0)]
        + _sents(6, "开")
        + [_para("2. 商务文件", size=16.0, outline_level=0)]
        + _sents(6, "商")
        + [_para("2.1 投标函", size=14.0)]
        + [
            _para("1. 我方按招标文件递交投标文件正本以及相关材料。", size=12.0),
            _para("2. 我方承认招标人有权决定中标人的相关权利。", size=12.0),
        ]
        + _sents(6, "尾")
    )

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps(
            {"is_title_block": False, "headings": [], "body": []}, ensure_ascii=False
        )

    result = run_smart_heading(
        records,
        llm_judge=_llm,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    meta: dict = {}
    blocks = _assemble_blocks_smart(records, result, {}, meta)

    def _block(needle):
        return next(b for b in blocks if needle in (b.get("heading") or ""))

    assert _block("1. 价格文件")["level"] == 1
    assert _block("2. 商务文件")["level"] == 1
    assert _block("1.1 开标一览表")["parent_headings"] == ["1. 价格文件"]
    assert _block("2.1 投标函")["parent_headings"] == ["2. 商务文件"]
    # The body EnNum lists never became headings, and never got the top-level
    # backfill / outline-anchor rules.
    assert not any(
        (b.get("heading") or "").startswith(("1. 我方", "2. 我方")) for b in blocks
    )
    for idx, r in enumerate(records):
        if r.text.startswith(("1. 我方", "2. 我方")):
            d = result.decisions.get(idx)
            assert d is None or (
                not d.is_heading
                and "backfill_top_level" not in d.rule_trail
                and "anchor_outline_series" not in d.rule_trail
            )


def test_fs_base_excludes_toc_lines() -> None:
    records = _body(10, size=12.0)
    records += [
        ParagraphRecord(
            kind="para", text="目录行" * 10, font_size_pt=20.0, is_toc_link=True
        )
        for _ in range(20)
    ]
    fs = document_fs_base(records, range(len(records)))
    assert fs.size_pt == 12.0


# ---------------------------------------------------------------------------
# spec-vs-implementation audit regressions (A1 / A2 / A5 / A6 / A10)
# ---------------------------------------------------------------------------


def test_cb1_second_pass_keeps_sizes_above_reestimated_base() -> None:
    """CB1 re-estimation disables same-size composite admission paths.

    Sizes strictly above the re-estimated body size remain auto-admitted.
    """
    records = _body(60, size=10.5)
    for i in range(30):
        records.append(_para(f"伪标题短语{i}", size=12.0))
    records.append(_para("一、同字号编号甲", size=12.0))
    records.append(_para("二、同字号编号乙", size=12.0))
    records.append(_para("同字号加粗题", size=12.0, all_bold=True))
    records.append(_para("真标题一", size=16.0))
    records.append(_para("真标题二", size=16.0))
    indices = list(range(len(records)))
    fs = document_fs_base(records, indices)

    result = gate_with_cb1(
        records,
        indices,
        fs_base=fs,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result.cb1_reestimated
    texts = _texts(result)
    # Above the folded-in 12pt body size: auto-admitted without any signal.
    assert "真标题一" in texts and "真标题二" in texts
    # Same-size series / bold paths are off in the CB1 second pass.
    assert "一、同字号编号甲" not in texts and "二、同字号编号乙" not in texts
    assert "同字号加粗题" not in texts
    assert all(f"伪标题短语{i}" not in texts for i in range(30))


def _cb1_sparse_or_long_records(*, sparse: bool) -> list[ParagraphRecord]:
    records: list[ParagraphRecord] = []
    if sparse:
        for i in range(4):
            records.append(_para(f"密集标题{i}", size=16.0))
            records += [_para("句", size=12.0), _para("短", size=12.0)]
        records += _body(60, size=12.0)
    else:
        for i in range(4):
            records.append(_para(f"正常标题{i}", size=16.0))
            records += _body(3, size=12.0)
        records += _body(40, size=12.0)
    return records


def _cb1_baseline_density_records() -> list:
    """4 headings (16pt) with long body gaps → 44.4% candidate density but the
    mean inter-heading spacing stays well over 200 weighted chars (no
    sparse-body trip). Layout: H B B H B B H B H → 4 headings / 9 paras = 0.444."""
    long_body = "这是一段刻意写长的正文内容，用来把相邻标题之间的正文字符数撑到二百加权字符以上，从而避开稀疏正文触发条件的判定。"
    gaps = (2, 2, 1)
    records: list = []
    for i in range(4):
        records.append(_para(f"标题第{i}节", size=16.0))
        if i < 3:
            for _ in range(gaps[i]):
                records.append(_para(long_body, size=12.0))
    return records


@pytest.mark.parametrize(
    "case_id,builder,baseline_density,min_inter_env,expect_reestimated,expected_headings",
    [
        pytest.param(
            "sparse_body_triggers",
            lambda: _cb1_sparse_or_long_records(sparse=True),
            None,
            None,
            True,
            (),
            id="sparse_body_triggers",
        ),
        pytest.param(
            "long_inter_heading_body_spared",
            lambda: _cb1_sparse_or_long_records(sparse=False),
            None,
            None,
            False,
            tuple(f"正常标题{i}" for i in range(4)),
            id="long_inter_heading_body_spared",
        ),
        pytest.param(
            "min_inter_heading_chars_env_override",
            lambda: _cb1_sparse_or_long_records(sparse=False),
            None,
            "400",
            True,
            (),
            id="min_inter_heading_chars_env_override",
        ),
        pytest.param(
            "baseline_aware_high_baseline_spared",
            _cb1_baseline_density_records,
            0.35,
            None,
            False,
            tuple(f"标题第{i}节" for i in range(4)),
            id="baseline_aware_high_baseline_spared",
        ),
        pytest.param(
            "baseline_aware_low_baseline_trips",
            _cb1_baseline_density_records,
            0.0,
            None,
            True,
            (),
            id="baseline_aware_low_baseline_trips",
        ),
    ],
)
def test_cb1_reestimation_threshold_cases(
    monkeypatch,
    case_id: str,
    builder,
    baseline_density: float | None,
    min_inter_env: str | None,
    expect_reestimated: bool,
    expected_headings: tuple[str, ...],
) -> None:
    """CB1 re-estimation thresholds: sparse inter-heading spacing, env override
    of MIN_INTER_HEADING_CHARS, and baseline-aware density ceiling."""
    if min_inter_env is not None:
        monkeypatch.setenv("DOCX_SMART_MIN_INTER_HEADING_CHARS", min_inter_env)

    records = builder()
    indices = list(range(len(records)))
    fs = document_fs_base(records, indices)
    assert fs.confidence_high

    warnings: dict = {}
    kw = {}
    if baseline_density is not None:
        kw["baseline_density"] = baseline_density
    result = gate_with_cb1(
        records,
        indices,
        fs_base=fs,
        warnings=warnings,
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
        **kw,
    )
    assert result.cb1_reestimated is expect_reestimated
    if expect_reestimated:
        assert warnings.get("smart_cb1_reestimated") == 1
    for heading in expected_headings:
        assert heading in _texts(result)


# ---------------------------------------------------------------------------
# test15 regression: a compact single-tier 公文 outline (一、二、三… at body
# font size) must survive at the default 0.40 density floor. The real notice
# had 9 same-size CN_NUM section headings (density 9/24 = 0.375) isolated into
# their own sub-document; at the old 0.35 floor CB1 tripped, and because cn_num
# was the ONLY ladder tier, graduated demotion peeled it to zero and wiped
# every heading. Raising the floor to 0.40 lets the raw 0.375 density pass.
# ---------------------------------------------------------------------------

_CN_ORDINALS = "一二三四五六七八九十"


def _compact_gongwen_records(*, heading_texts, body_per_gap, size: float = 16.0):
    """CN_NUM headings at FS_base size, each followed by ``body_per_gap[i]`` long
    body paragraphs — ALL at the same size, so fs_dominant_ratio is 1.0 and the
    headings carry no size signal (mirrors the real notice)."""
    recs: list = []
    for i, htext in enumerate(heading_texts):
        recs.append(_para(htext, size=size))
        for _ in range(body_per_gap[i]):
            recs.append(_para(_GRAD_BODY, size=size))
    return recs


def test_cb1_compact_gongwen_outline_survives_at_default_floor(monkeypatch) -> None:
    """Fix-proof (test15): 9 same-size ``一、``…``九、`` CN_NUM section headings,
    one substantial body paragraph each — density 9/24 = 0.375, healthy (>200)
    inter-heading spacing. At the default 0.40 floor CB1 short-circuits (raw
    density under the ceiling, not sparse) so every heading survives. Pinning the
    OLD 0.35 floor reproduces the failure: the sole cn_num tier is peeled to
    zero."""
    heads = [f"{_CN_ORDINALS[i]}、第{i}节标题内容" for i in range(9)]
    # 8 internal gaps carry all 15 body paras (2×7 + 1), no leading/trailing, so
    # every body sits between the first and last candidate → spacing well >200.
    gaps = (2, 2, 2, 2, 2, 2, 2, 1, 0)
    records = _compact_gongwen_records(heading_texts=heads, body_per_gap=gaps)
    indices = list(range(len(records)))
    assert len([i for i in indices if records[i].text.strip()]) == 24  # 9 + 15

    # Default floor 0.40: 0.375 ≤ 0.40 and spacing healthy → CB1 does not engage.
    result = _grad_gate(records, {})
    assert not result.cb1_reestimated
    assert not result.cb1_strong_body_recovered
    assert result.cb1_graduated_stages == []
    assert result.density == pytest.approx(9 / 24, abs=1e-4)
    assert sum(1 for d in result.decisions if d.is_heading) == 9
    assert all(h in _texts(result) for h in heads)

    # Fix-proof: pin the OLD 0.35 floor → 0.375 > 0.35 trips, the sole cn_num
    # ladder tier is peeled to zero, every heading is demoted.
    monkeypatch.setenv("DOCX_SMART_DENSITY_MAX", "0.35")
    warnings: dict = {}
    demoted = _grad_gate(records, warnings)
    assert demoted.cb1_graduated_stages == ["cn_num"]
    assert demoted.cb1_graduated_demoted == 9
    assert sum(1 for d in demoted.decisions if d.is_heading) == 0
    assert warnings["smart_cb1_graduated_demotions"] == 9


def test_cb1_sparse_backstop_still_engages_below_density_floor() -> None:
    """Negative control: raising the density floor to 0.40 must NOT disable the
    sparse-body backstop (``MIN_INTER_HEADING_CHARS=200``). A packed run of
    same-size ``一、``…``六、`` CN_NUM candidates has density 6/16 = 0.375 (UNDER
    the 0.40 ceiling) but ~0 inter-heading spacing → CB1 still engages via
    sparsity and the sole cn_num tier is demoted. ``cb1_tripped`` stays False
    (graduated handled it), so we assert the graduated stages, not the trip
    flag."""
    records = _body(5, size=16.0)  # leading body keeps density under the ceiling
    for i in range(6):  # packed CN_NUM run: no body between → sparse spacing
        records.append(_para(f"{_CN_ORDINALS[i]}、紧凑小节{i}", size=16.0))
    records += _body(5, size=16.0)  # trailing body
    assert len(records) == 16  # density 6/16 = 0.375 ≤ 0.40

    result = _grad_gate(records, {})
    # Density under the ceiling → ONLY sparsity could have engaged CB1.
    assert result.cb1_graduated_stages == ["cn_num"]
    assert result.cb1_graduated_demoted == 6
    assert sum(1 for d in result.decisions if d.is_heading) == 0


def test_cb1_body_in_disguise_still_demoted_after_ceiling_raise() -> None:
    """Negative control: raising the floor to 0.40 must NOT let numbered
    body-in-disguise become headings. Nine ``一、``…``九、`` CN_NUM candidates
    that are full sentences (strong-body) at density 9/24 = 0.375 slip UNDER the
    0.40 ceiling with healthy spacing, so CB1 short-circuits and they survive the
    gate first-pass (CN_NUM defers strong-body). The downstream sweep —
    ``assign_levels_by_size`` + ``demote_strong_body_headings``, exactly as
    run_smart_heading runs it — then demotes every one. Asserting the gate return
    alone would falsely 'pass'."""
    heads = [
        f"{_CN_ORDINALS[i]}、这是一段伪装成小节标题实则为完整正文的句子内容以句号收尾。"
        for i in range(9)
    ]
    gaps = (2, 2, 2, 2, 2, 2, 2, 1, 0)
    records = _compact_gongwen_records(heading_texts=heads, body_per_gap=gaps)
    warnings: dict = {}
    result = _grad_gate(records, warnings)

    # 0.375 ≤ 0.40 and spacing healthy → CB1 short-circuits; candidates are still
    # is_heading at the gate boundary (CN_NUM defers strong-body to the sweep).
    assert not result.cb1_reestimated and result.cb1_graduated_stages == []
    assert sum(1 for d in result.decisions if d.is_heading) == 9

    # Downstream sweep demotes the body-in-disguise — the real guard for this
    # class, unchanged by the ceiling raise.
    ds = result.decisions
    assign_levels_by_size(ds)
    demote_strong_body_headings(ds, strong_body=_stub_strong_body, warnings=warnings)
    assert not any(
        d.numbering is not None and d.numbering.style_key == "CnNum" and d.is_heading
        for d in ds
    )


def test_caption_veto_spares_outline_paragraphs() -> None:
    """A5 (P3 × I2): an outline paragraph hitting the caption blacklist keeps
    its candidacy (silently dropping it would trip the I2 machine check);
    the caption veto still rejects non-outline paragraphs."""
    from lightrag.parser.docx.smart_heading import guardrails

    records = _body(30)
    records.append(_para("图 1 系统架构", size=12.0, outline_level=1))
    records.append(_para("图 2 数据流转示意", size=16.0))
    result = _gate(records, caption_veto=guardrails.caption_prefix_reason)
    texts = _texts(result)
    assert "图 1 系统架构" in texts  # outline is immune to P3
    assert "图 2 数据流转示意" not in texts  # P3 vetoes the size-tier path


def test_llm_body_veto_revokes_candidacy() -> None:
    """An LLM body vote strips an otherwise strong-size candidate.

    Only the veto side of the LLM partition carries force; without the
    vote it is admitted."""
    records = _body(30)
    records.append(_para("被判为正文的大字号行", size=16.0))
    idx = len(records) - 1

    admitted = _gate(records)
    assert "被判为正文的大字号行" in _texts(admitted)

    vetoed = _gate(records, llm_body_vetoes={idx})
    assert "被判为正文的大字号行" not in _texts(vetoed)


@pytest.mark.parametrize(
    "case_id,target,veto_fn,expect_admitted_rule,expected_rejected_marks",
    [
        pytest.param(
            "no_admission_force",
            _para("GB/T 9704—2012", size=12.0, alignment="right"),
            None,
            None,
            ("llm_grant_rejected",),
            id="no_admission_force",
        ),
        pytest.param(
            "does_not_block_normal_admission",
            _para("真正的大字号标题", size=16.0),
            None,
            "size_strong",
            (),
            id="does_not_block_normal_admission",
        ),
        pytest.param(
            "rejected_with_numbering_veto_single_ledger_row",
            _para("一、被撤销编号且被点名的行", size=12.0),
            lambda _c, _t: "homophone_stub",
            None,
            ("homophone_stub", "numbering_veto_suppressed", "llm_grant_rejected"),
            id="rejected_with_numbering_veto_single_ledger_row",
        ),
    ],
)
def test_llm_grant_gate_cases(
    case_id: str,
    target: ParagraphRecord,
    veto_fn,
    expect_admitted_rule: str | None,
    expected_rejected_marks: tuple[str, ...],
) -> None:
    """LLM heading grants are audit-only: they neither admit on their own nor
    block normal admission, and combine into one ledger row with numbering veto."""
    records = _body(30) + [target]
    idx = len(records) - 1
    gate_kw = {"llm_heading_grants": {idx}}
    if veto_fn is not None:
        gate_kw["numbering_veto"] = veto_fn

    result = _gate(records, **gate_kw)
    if expect_admitted_rule is not None:
        d = next(x for x in result.decisions if x.record_index == idx)
        assert d.rule_trail[0] == expect_admitted_rule
        assert result.grant_rejected == []
        return

    assert target.text not in _texts(result)
    assert result.veto_suppressed == []
    assert [d.record_index for d in result.grant_rejected] == [idx]
    rej = result.grant_rejected[0]
    assert rej.is_heading is False
    assert rej.level is None
    assert rej.use_raw_text is False
    for mark in expected_rejected_marks:
        assert mark in rej.rule_trail


def test_llm_grant_rejected_end_to_end_audit(monkeypatch) -> None:
    """test9 国标范例 end-to-end: the LLM rejects the 目次-page window as a
    title block but names every line a heading (headings=[0,1,2]). Only the
    16pt 目次 line survives (size_strong, on its own signal); the base-size
    running-header and TOC lines are rejected grants — counted once against
    the final gate result and ledgered in ``audit["decisions"]`` as
    output-neutral non-headings."""
    import json

    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")

    records = [
        _para("GB/T 9704—2012", size=10.5, alignment="right"),  # running header
        _para("目  次", size=16.0, alignment="center"),
        _para("前言", size=10.5),  # a plain-text TOC entry
    ]
    records += _body(30, size=10.5)

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps(
            {
                "is_title_block": False,
                "main_title": None,
                "sub_title": None,
                "doc_number": None,
                "classification": None,
                "publisher": None,
                "date": None,
                "headings": [0, 1, 2],
                "body": [],
            },
            ensure_ascii=False,
        )

    warnings: dict = {}
    result = run_smart_heading(
        records,
        llm_judge=_llm,
        warnings=warnings,
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    assert result.decisions[1].is_heading  # 目次: admitted by size_strong
    for idx in (0, 2):  # granted base-size lines stay body, rule-tagged
        d = result.decisions[idx]
        assert d.is_heading is False
        assert "llm_grant_rejected" in d.rule_trail
    assert warnings["smart_llm_grant_rejected"] == 2
    rows = [r for r in result.audit["decisions"] if "llm_grant_rejected" in r["rules"]]
    assert len(rows) == 2
    assert all(r["is_heading"] is False and r["level"] is None for r in rows)


def test_merge_ledger_only_merges_and_counts_once() -> None:
    """The ledger merge shared by the normal path and the CB1 fallback:
    rows ``setdefault`` in (an existing real decision wins) and the
    grant-rejected warning counts once per FINAL gate result."""
    from lightrag.parser.docx.smart_heading.heading_flow import (
        GateResult,
        _merge_ledger_only,
    )

    records = _body(5)
    fs = document_fs_base(records, range(len(records)))
    rej = HeadingDecision(record_index=3, text="x", is_heading=False)
    rej.note("llm_grant_rejected")
    supp = HeadingDecision(record_index=4, text="y", is_heading=False)
    supp.note("numbering_veto_suppressed")
    gate = GateResult(
        decisions=[],
        fs_base=fs,
        density=0.0,
        veto_suppressed=[supp],
        grant_rejected=[rej],
    )
    existing = HeadingDecision(record_index=3, text="real", is_heading=True)
    decisions = {3: existing}
    warnings: dict = {}
    _merge_ledger_only(decisions, gate, warnings)
    assert decisions[3] is existing
    assert decisions[4] is supp
    assert warnings["smart_llm_grant_rejected"] == 1


def test_title_block_gate_uses_char_weighted_mode_not_mean(monkeypatch) -> None:
    """The title-block baseline is the initial global FS_base mode.

    Here the char-weighted mode is 12pt; a long large-font
    paragraph inflating the weighted mean must not raise the gate."""
    import json

    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")

    records = [
        _para("行业观察白皮书", size=14.0),  # title line: 14 ≥ mode12+1
        _para("某某研究院", size=12.0),
    ]
    records += _body(30, size=12.0)
    records.append(_para("超大字号的引言长段落" * 40 + "。", size=30.0))

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps(
            {
                "is_title_block": True,
                "main_title": "行业观察白皮书",
                "sub_title": None,
                "doc_number": None,
                "classification": None,
                "publisher": "某某研究院",
                "date": None,
                "headings": [],
                "body": [],
            },
            ensure_ascii=False,
        )

    result = run_smart_heading(
        records,
        llm_judge=_llm,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    assert result.doc_title == "行业观察白皮书"
    assert any(d.is_title_block and d.level == 0 for d in result.decisions.values())


def test_title_block_multiline_main_title_flattened(monkeypatch) -> None:
    """A soft-break cover title echoed by the LLM with its ``\\n`` intact must
    land single-line everywhere it fans out: result.doc_title, the meta
    doc_title key, the block heading, and every descendant's
    parent_headings."""
    import json

    from lightrag.parser.docx.parse_document import _assemble_blocks_smart
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")

    records = [
        _para("行业观察\n白皮书", size=14.0),
        _para("某某研究院", size=12.0),
    ]
    records += _body(15, size=12.0)
    records.append(_para("第一章 总则", size=12.0, outline_level=0))
    records += _body(15, size=12.0)

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps(
            {"is_title_block": True, "main_title": "行业观察\n白皮书"},
            ensure_ascii=False,
        )

    result = run_smart_heading(
        records,
        llm_judge=_llm,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    assert result.doc_title == "行业观察白皮书"

    meta: dict = {}
    blocks = _assemble_blocks_smart(records, result, {}, meta)
    assert meta["doc_title"] == "行业观察白皮书"
    title = next(b for b in blocks if b.get("is_title_block"))
    assert title["heading"] == "行业观察白皮书"
    chapter = next(b for b in blocks if b["heading"] == "第一章 总则")
    assert chapter["parent_headings"] == ["行业观察白皮书"]
    for b in blocks:
        assert "\n" not in b["heading"]
        assert all("\n" not in h for h in b["parent_headings"])


def test_assembler_doc_title_empty_without_title_block() -> None:
    """Smart mode: the title block is the ONLY doc_title source. With no
    title verdict the assembler records an explicitly EMPTY doc_title (a
    "前言"-style first heading must not masquerade as the document title),
    while first_heading keeps its legacy any-heading semantics."""
    from lightrag.parser.docx.parse_document import _assemble_blocks_smart
    from lightrag.parser.docx.smart_heading.heading_flow import SmartHeadingResult

    records = [_para("前言", outline_level=0), _para("正文一句。")]
    result = SmartHeadingResult(
        decisions={
            0: HeadingDecision(record_index=0, text="前言", is_heading=True, level=1)
        },
        toc_indices=set(),
        doc_title=None,
        audit={},
    )
    meta: dict = {}
    _assemble_blocks_smart(records, result, {}, meta)
    assert meta["doc_title"] == ""
    assert meta["first_heading"] == "前言"


def test_assembler_toc_retention_keeps_first_lines_as_body() -> None:
    """TOC retention emits the first N lines as body under the 目录 heading.

    The rest collapse to a single '……', and no retained line is
    ever a heading (the toc_indices branch continues before any heading path)."""
    from lightrag.parser.docx.parse_document import _assemble_blocks_smart
    from lightrag.parser.docx.smart_heading.guardrails import TOC_ELLIPSIS
    from lightrag.parser.docx.smart_heading.heading_flow import SmartHeadingResult

    entries = [
        "第一章 绪论……3",
        "第二章 方法……12",
        "第三章 实验……25",
        "第四章 结果……40",
        "第五章 讨论……55",
        "第六章 结论……70",
        "参考文献……80",
        "附录……90",
    ]
    records = [_para("目录")]
    records += [_para(t) for t in entries]  # records[1..8]
    records.append(_para("这是正文第一段，跟在目录后面。"))
    toc_indices = set(range(1, 9))

    def _sentinel(i: int) -> HeadingDecision:
        d = HeadingDecision(record_index=i, text="")
        d.note("toc_removed")
        return d

    result = SmartHeadingResult(
        decisions={
            0: HeadingDecision(record_index=0, text="目录", is_heading=True, level=1),
            **{i: _sentinel(i) for i in range(1, 9)},
        },
        toc_indices=toc_indices,
        doc_title=None,
        audit={},
        toc_kept_text={i: records[i].text for i in range(1, 6)},  # first 5
        toc_ellipsis_anchor=5,
        toc_kept_lines=5,
        toc_removed_lines=3,
    )

    blocks = _assemble_blocks_smart(records, result, {}, {})
    lines = [ln for b in blocks for ln in b["content"].split("\n")]

    # first 5 entries survive as body lines (verbatim, dot leaders + page nos)
    for i in range(1, 6):
        assert records[i].text in lines
    # entries 6-8 are dropped
    for i in range(6, 9):
        assert records[i].text not in lines
    # exactly one standalone ellipsis line, right after the last kept entry
    assert lines.count(TOC_ELLIPSIS) == 1
    assert lines.index(TOC_ELLIPSIS) == lines.index(records[5].text) + 1
    # never a heading: no block heading equals a TOC entry, no kept line is
    # rendered with a markdown heading prefix, and each sentinel is non-heading
    assert not ({b["heading"] for b in blocks} & set(entries))
    assert not any(ln.startswith("# 第") for ln in lines)
    for i in range(1, 6):
        assert result.decisions[i].is_heading is False
    # the retained entries land under the 目录 section, not orphaned
    toc_block = next(b for b in blocks if b["heading"] == "目录")
    assert "第一章 绪论……3" in toc_block["content"]
    assert "这是正文第一段，跟在目录后面。" in toc_block["content"]


def _run_toc(records, monkeypatch):
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    monkeypatch.setenv("DOCX_SMART_SUBDOC_MIN_TOKENS", "10")
    return run_smart_heading(
        records,
        llm_judge=_no_title_llm,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )


def test_toc_keep_lines_env_wiring(monkeypatch) -> None:
    """DOCX_SMART_TOC_KEEP_LINES must be read through run_smart_heading (not
    just the pure planner): default when unset, _env_int fallback on garbage,
    negative clamps to 0, huge keeps all. Eight one-line dot-leader TOC paras."""
    records = [_para(f"第{i}章 标题............{i}", size=12.0) for i in range(8)]
    records += _body(20, size=12.0)

    monkeypatch.delenv("DOCX_SMART_TOC_KEEP_LINES", raising=False)
    r = _run_toc(records, monkeypatch)
    assert r is not None and r.toc_indices == set(range(8))
    assert (r.toc_kept_lines, r.toc_removed_lines) == (5, 3)  # unset → default 5

    monkeypatch.setenv("DOCX_SMART_TOC_KEEP_LINES", "2")
    r = _run_toc(records, monkeypatch)
    assert (r.toc_kept_lines, r.toc_removed_lines) == (2, 6)

    monkeypatch.setenv("DOCX_SMART_TOC_KEEP_LINES", "not-an-int")
    r = _run_toc(records, monkeypatch)
    assert r.toc_kept_lines == 5  # _env_int falls back to the default

    monkeypatch.setenv("DOCX_SMART_TOC_KEEP_LINES", "-3")
    r = _run_toc(records, monkeypatch)
    assert (r.toc_kept_lines, r.toc_removed_lines) == (0, 8)  # clamped to 0

    monkeypatch.setenv("DOCX_SMART_TOC_KEEP_LINES", "9999")
    r = _run_toc(records, monkeypatch)
    assert (r.toc_kept_lines, r.toc_removed_lines) == (8, 0)
    assert r.toc_ellipsis_anchor is None  # nothing elided


def test_toc_retention_straddling_paragraph_line_accurate(monkeypatch) -> None:
    """A single paragraph carrying 8 soft-break (\\n) dot-leader lines, keep 5:
    the record keeps its first 5 lines and the 3-line tail counts as removed —
    proving the budget/warning are per visible LINE, not per record."""
    toc_para = _para(
        "\n".join(f"第{i}章 标题............{i}" for i in range(8)), size=12.0
    )
    records = [toc_para] + _body(20, size=12.0)

    monkeypatch.delenv("DOCX_SMART_TOC_KEEP_LINES", raising=False)  # default 5
    r = _run_toc(records, monkeypatch)
    assert r is not None and r.toc_indices == {0}
    assert (r.toc_kept_lines, r.toc_removed_lines) == (5, 3)
    assert r.toc_ellipsis_anchor == 0  # the straddling record itself
    kept = r.toc_kept_text[0].split("\n")
    assert len(kept) == 5 and kept[0] == "第0章 标题............0"


def test_title_block_judge_receives_fs_base(monkeypatch) -> None:
    """Call-site guard: run_smart_heading must hand the global FS_base
    initial value to judge_title_block — the font-size evidence legend
    depends on it, and a dropped kwarg silently renders the legend-free
    prompt (the M1212 regression would come back unnoticed)."""
    import json

    from lightrag.parser.docx.smart_heading import title_block as tb
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")

    records = [
        _para("行业观察白皮书", size=14.0),
        _para("某某研究院", size=12.0),
    ]
    records += _body(30, size=12.0)

    captured: dict = {}
    real_judge = tb.judge_title_block

    def _spy(candidate, recs, judge, **kw):
        captured["fs_base_pt"] = kw.get("fs_base_pt")
        return real_judge(candidate, recs, judge, **kw)

    # run_smart_heading imports title_block function-locally, so patching the
    # module attribute intercepts its call.
    monkeypatch.setattr(tb, "judge_title_block", _spy)

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps(
            {"is_title_block": True, "main_title": "行业观察白皮书"},
            ensure_ascii=False,
        )

    result = run_smart_heading(
        records,
        llm_judge=_llm,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    assert captured["fs_base_pt"] == 12.0  # the char-weighted body mode


def test_seq_break_knob_splits_distant_series(monkeypatch) -> None:
    """A12 (DOCX_SMART_SEQ_BREAK_PARAS): a long body run between same-key
    members closes the open sequence; the default 0 keeps one series."""
    from lightrag.parser.docx.smart_heading.heading_flow import (
        align_numbering_series,
    )

    def _series(levels_and_idx):
        ds = []
        for text, level, idx in levels_and_idx:
            d = _decision(text, idx=idx)
            d.level = level
            ds.append(d)
        return ds

    spec = [("一、条目甲", 3, 0), ("二、条目乙", 4, 1), ("三、远端条目", 5, 60)]

    monkeypatch.delenv("DOCX_SMART_SEQ_BREAK_PARAS", raising=False)
    ds = _series(spec)
    align_numbering_series(ds)
    assert [d.level for d in ds] == [3, 3, 3]  # one series, ties → shallow

    monkeypatch.setenv("DOCX_SMART_SEQ_BREAK_PARAS", "5")
    ds = _series(spec)
    align_numbering_series(ds)
    # 三、 sits 58 record slots away → its sequence is closed and it aligns
    # alone (not pulled to 3); the 5→4 move is the pre-existing subtree
    # shift riding along when 二、 aligned 4→3.
    assert [d.level for d in ds] == [3, 3, 4]


def test_run_level_audit_exports_metrics(monkeypatch) -> None:
    """The audit payload includes calls, verdicts, and the decision ledger."""
    import json

    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    records = [
        _para("行业观察白皮书", size=16.0),
        _para("某某研究院", size=12.0),
    ]
    records += _body(30, size=12.0)

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps(
            {"is_title_block": True, "main_title": "行业观察白皮书"},
            ensure_ascii=False,
        )

    result = run_smart_heading(
        records,
        llm_judge=_llm,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    audit = result.audit
    assert audit["llm_calls"] >= 1
    assert [c["is_title_block"] for c in audit["title_block_candidates"]] == [True]
    rows = audit["decisions"]
    assert rows
    assert all(
        {
            "hash",
            "summary",
            "font_size_pt",
            "sub_fs_base",
            "rules",
            "level",
            "is_heading",
        }
        <= set(r)
        for r in rows
    )
    assert any(any("title_block" in rule for rule in r["rules"]) for r in rows)
    # Enriched fields present and the plaintext preview is capped. Value
    # assertions live in test_audit_decision_fields_enriched below.
    from lightrag.parser.docx.smart_heading.heading_flow import _AUDIT_SUMMARY_CHARS

    assert all(len(r["summary"]) <= _AUDIT_SUMMARY_CHARS + 3 for r in rows)  # +"..."
    # Every sub-document (fallbacks included) now records its fs_base.
    assert all("fs_base" in s for s in audit["sub_documents"])


def test_audit_decision_fields_enriched(monkeypatch) -> None:
    """Requirement 2: each decision row carries a plaintext summary, the
    paragraph mode font off the record, and its sub-document fs_base."""
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    # No title block (llm_judge=None) → the whole doc is one sub-document.
    records = [_para("第一章 绪论", size=16.0, outline_level=0)]
    records += _body(30, size=12.0)

    result = run_smart_heading(
        records,
        llm_judge=None,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    rows = result.audit["decisions"]
    heading_row = next(r for r in rows if r["summary"] == "第一章 绪论")
    assert heading_row["is_heading"] is True
    assert heading_row["font_size_pt"] == 16.0  # record mode, not FS_base
    assert heading_row["sub_fs_base"] == 12.0  # the sub-document body baseline


def test_run_logs_physical_feature_summary(monkeypatch, lightrag_caplog) -> None:
    """Requirement 1: a document-level info line (FS_base, paragraph count,
    outline histogram) fires once physical features are extracted."""
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    records = [_para("第一章 绪论", size=16.0, outline_level=0)]
    records += _body(30, size=12.0)

    with lightrag_caplog.at_level(logging.INFO, logger="lightrag"):
        run_smart_heading(
            records,
            llm_judge=None,
            warnings={},
            strong_body=_stub_strong_body,
            numbering_veto=_stub_no_veto,
            caption_veto=_stub_no_caption,
        )

    summary = next(
        (m for m in _log_messages(lightrag_caplog) if "physical features:" in m), None
    )
    assert summary is not None
    assert "FS_base=12.0pt" in summary
    assert "31 paragraphs" in summary
    assert "L1: 1" in summary and "none: 30" in summary


def test_table_title_block_end_to_end_assembly(monkeypatch) -> None:
    """A cover laid out inside tables becomes a complete title block.

    End to end, it is judged a level-0 block whose heading is the main title
    and whose content carries
    every absorbed cover-cell text verbatim (as plain text, not a ``<table>``
    placeholder). With no section heading in the document, the title block also
    OWNS the following body and the trailing data table (block-boundary merge),
    and the data table keeps its ``<table>`` placeholder; I1 passes (absorption
    is lossless)."""
    import json

    from lightrag.parser.docx.parse_document import (
        ParagraphRecord,
        _assemble_blocks_smart,
    )
    from lightrag.parser.docx.smart_heading import guardrails as g
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")

    def _table(rows, text="<table>[]</table>"):
        return ParagraphRecord(kind="table", text=text, table_cell_features=rows)

    cover_title = _table(
        [[("产品标准化大纲", 22.0, False)], [("某某模块", 22.0, False)]],
        text='<table>[["产品标准化大纲"], ["某某模块"]]</table>',
    )
    cover_publisher = _table(
        [[("某某电子股份有限公司", 16.0, False)]],
        text='<table>[["某某电子股份有限公司"]]</table>',
    )
    data_table = _table(
        [[("序号", 10.5, False), ("这一格是以句号结尾的数据。", 10.5, False)]],
        text='<table>[["序号", "这一格是以句号结尾的数据。"]]</table>',
    )
    records = [cover_title, cover_publisher] + _body(30, size=12.0) + [data_table]

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        if "产品标准化大纲" in prompt:
            return json.dumps(
                {
                    "is_title_block": True,
                    "main_title": "产品标准化大纲某某模块",
                    "publisher": "某某电子股份有限公司",
                },
                ensure_ascii=False,
            )
        return json.dumps({"is_title_block": False}, ensure_ascii=False)

    result = run_smart_heading(
        records,
        llm_judge=_llm,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    assert result.doc_title == "产品标准化大纲某某模块"
    lead = result.decisions[0]
    assert lead.is_title_block and lead.level == 0
    assert lead.member_indices == (0, 1)

    meta: dict = {}
    blocks = _assemble_blocks_smart(records, result, {}, meta)
    title_blocks = [b for b in blocks if b.get("is_title_block")]
    assert len(title_blocks) == 1
    tb = title_blocks[0]
    assert tb["level"] == 0
    # heading is the plain main title; the publisher stays in the content.
    assert tb["heading"] == "产品标准化大纲某某模块"
    for cell in ("产品标准化大纲", "某某模块", "某某电子股份有限公司"):
        assert cell in tb["content"]  # absorption is lossless
    assert meta["first_heading"] == "产品标准化大纲某某模块"
    assert meta["doc_title"] == "产品标准化大纲某某模块"

    # No section heading separates the cover from the rest, so the whole
    # document is one title block: cover cells carried as PLAIN text (no
    # <table> placeholder), while the ordinary data table keeps its placeholder.
    assert len(blocks) == 1
    assert '["产品标准化大纲"]' not in tb["content"]
    assert '<table>[["序号"' in tb["content"]

    # I1: absorbed cell texts never trip the paragraph-preservation check.
    assert g.verify_content_preservation(records, blocks, toc_indices=set()) == []


def test_title_block_empty_members_defense() -> None:
    """Defensive: a title verdict whose members yield no emittable content (an
    empty-cell cover table) must warn + skip the title treatment rather than
    crash (``_build_unsplit_block`` indexes ``paragraphs[-1]``) or mis-label.
    The degenerate lead never becomes a level-1 heading and no stale pin leaks
    to the following body block. (Unreachable via ``run_smart_heading`` — its
    locate-back requires a non-empty window — so it is exercised directly.)"""
    from lightrag.parser.docx.parse_document import (
        ParagraphRecord,
        _assemble_blocks_smart,
    )
    from lightrag.parser.docx.smart_heading.heading_flow import SmartHeadingResult

    empty_cover = ParagraphRecord(
        kind="table",
        text='<table>[[""]]</table>',
        table_cell_features=[[("   ", 22.0, False)]],
        para_id="t0",
    )
    body = ParagraphRecord(
        kind="para", text="正文一句。", font_size_pt=12.0, para_id="p1"
    )
    records = [empty_cover, body]
    result = SmartHeadingResult(
        decisions={
            0: HeadingDecision(
                record_index=0,
                text="<table>…",
                is_heading=True,
                level=0,
                is_title_block=True,
                member_indices=(0,),
            )
        },
        toc_indices=set(),
        doc_title=None,
        audit={},
    )
    warnings: dict = {}
    blocks = _assemble_blocks_smart(records, result, warnings, {})

    assert warnings.get("title_block_empty_members") == 1
    assert not any(b.get("is_title_block") for b in blocks)
    # The following body survived as an ordinary (non-title) block — no crash,
    # no level-1 mis-labelling of the degenerate title decision.
    joined = "\n".join(b["content"] for b in blocks)
    assert "正文一句" in joined


# ---------------------------------------------------------------------------
# CB1 look-ahead evaluates density after strong-body and series propagation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case_id,second_heading,clause_texts,check_logs",
    [
        pytest.param(
            "all_strong_clauses",
            "二、主要任务",
            (
                "加强组织领导，落实工作责任，明确任务分工，确保各项措施落地见效。",
                "加强组织领导，落实工作责任，明确任务分工，确保各项措施落地见效。",
                "加强组织领导，落实工作责任，明确任务分工，确保各项措施落地见效。",
            ),
            True,
            id="all_strong_clauses",
        ),
        pytest.param(
            "partial_series_propagation",
            "二、保障措施",
            (
                "加强组织领导，落实工作责任，明确任务分工，确保各项措施落地见效。",
                "完善制度机制，强化监督问责，明确任务分工，压实各方责任",
                "完善制度机制，强化监督问责，明确任务分工，压实各方责任",
            ),
            False,
            id="partial_series_propagation",
        ),
    ],
)
def test_cb1_strong_body_lookahead_cases(
    lightrag_caplog,
    case_id: str,
    second_heading: str,
    clause_texts: tuple[str, str, str],
    check_logs: bool,
) -> None:
    """CB1 look-ahead evaluates density after strong-body + CB2 series
    propagation, sparing the trip without re-estimation so same-size CnNum
    headings survive the downstream sweep."""
    records = [
        _para("一、总体要求"),
        _para(f"（一）{clause_texts[0]}"),
        _para(f"（二）{clause_texts[1]}"),
        _para(f"（三）{clause_texts[2]}"),
        _para(second_heading),
        _para(f"（四）{clause_texts[0]}"),
        _para(f"（五）{clause_texts[1]}"),
        _para(f"（六）{clause_texts[2]}"),
    ]
    indices = list(range(len(records)))
    fs = document_fs_base(records, indices)
    assert fs.size_pt == 12.0 and fs.confidence_high

    warnings: dict = {}
    with lightrag_caplog.at_level(logging.INFO, logger="lightrag"):
        gate = gate_with_cb1(
            records,
            indices,
            fs_base=fs,
            warnings=warnings,
            strong_body=_stub_strong_body,
            numbering_veto=_stub_no_veto,
            caption_veto=_stub_no_caption,
        )
    assert warnings.get("smart_cb1_strong_body_recovered") == 1
    assert gate.cb1_strong_body_recovered
    assert not gate.cb1_reestimated and not gate.cb1_tripped
    if check_logs:
        regate = next(
            r for r in lightrag_caplog.records if "not tripping" in r.getMessage()
        )
        assert regate.levelno == logging.INFO
        assert not any(
            "re-estimated FS_base" in m for m in _log_messages(lightrag_caplog)
        )

    ds = gate.decisions
    assign_levels_by_size(ds)
    demote_strong_body_headings(ds, strong_body=_stub_strong_body, warnings=warnings)
    assert {d.text for d in ds if d.is_heading} == {"一、总体要求", second_heading}


# ---------------------------------------------------------------------------
# clause-class numbering (条/款/项, Article/§) defers to the post-merge sweep
# ---------------------------------------------------------------------------


def test_early_strong_body_keys_covers_only_chapter_classes() -> None:
    """Chapter classes are strong-body-checked at recognition (per-paragraph,
    no propagation); clause classes are NOT — they defer to the post-merge
    sweep so CB2 can demote a body-in-disguise run as a whole series."""
    from lightrag.parser.docx.smart_heading.heading_flow import (
        EARLY_STRONG_BODY_KEYS,
    )
    from lightrag.parser.docx.smart_heading.style_key import (
        CN_CHAPTER,
        CN_CLAUSE,
        EN_CHAPTER,
        EN_CLAUSE,
    )

    assert CN_CHAPTER in EARLY_STRONG_BODY_KEYS
    assert EN_CHAPTER in EARLY_STRONG_BODY_KEYS
    assert CN_CLAUSE not in EARLY_STRONG_BODY_KEYS
    assert EN_CLAUSE not in EARLY_STRONG_BODY_KEYS


@pytest.mark.parametrize(
    "case_id,clauses,expected_headings,demoted_survivors",
    [
        pytest.param(
            "cn_clause_mostly_body",
            [
                "第一条  为了加强安全生产领域信用体系建设，规范严重失信主体名单管理，依法制定本办法。",
                "第二条  为了加强安全生产领域信用体系建设，规范严重失信主体名单管理，依法制定本办法。",
                "第三条  为了加强安全生产领域信用体系建设，规范严重失信主体名单管理，依法制定本办法。",
                "第六条  下列单位及其人员应当列入严重失信主体名单：",
                "第八条  应急管理部门可以采取下列管理措施：",
            ],
            set(),
            (
                "第六条  下列单位及其人员应当列入严重失信主体名单：",
                "第八条  应急管理部门可以采取下列管理措施：",
            ),
            id="cn_clause_mostly_body",
        ),
        pytest.param(
            "cn_clause_short_bare_survive",
            ["第一条", "第二条", "第三条", "第四条"],
            {"第一条", "第二条", "第三条", "第四条"},
            (),
            id="cn_clause_short_bare_survive",
        ),
        pytest.param(
            "en_clause_mostly_body",
            [
                "Article 1  This article establishes the comprehensive requirements and the detailed obligations applicable to all covered entities herein.",
                "Article 2  This article establishes the comprehensive requirements and the detailed obligations applicable to all covered entities herein.",
                "Article 3  This article establishes the comprehensive requirements and the detailed obligations applicable to all covered entities herein.",
                "Article 5  introduces the following measures:",
            ],
            set(),
            ("Article 5  introduces the following measures:",),
            id="en_clause_mostly_body",
        ),
    ],
)
def test_clause_series_post_merge_sweep_cases(
    case_id: str,
    clauses: list[str],
    expected_headings: set[str],
    demoted_survivors: tuple[str, ...],
) -> None:
    """Deferred CnClause/EnClause series demote as a whole via CB2 when mostly
    body (dragging short colon-lead survivors down), while bare short clause
    headings with zero hits survive intact."""
    records = _body(20) + [_para(t) for t in clauses]
    warnings: dict = {}
    result = _gate(records, warnings=warnings)
    ds = result.decisions
    assign_levels_by_size(ds)
    demote_strong_body_headings(ds, strong_body=_stub_strong_body, warnings=warnings)

    by_text = {d.text: d for d in ds}
    for survivor in demoted_survivors:
        assert by_text[survivor].is_heading is False
    assert {d.text for d in ds if d.is_heading} == expected_headings


# ---------------------------------------------------------------------------
# 公文版记 (imprint): strong-body rule 0 in the gate/sweep, and the
# outline-only fallback exception
# ---------------------------------------------------------------------------
# These tests use the REAL guardrails.strong_body_reason: every record either
# opens with an imprint marker (regex, returns before any NLP) or ends with a
# CJK sentence terminator (string check), so no spaCy models are needed.


@pytest.mark.parametrize(
    "case_id,para_kwargs,expect_raw_text,check_i2",
    [
        pytest.param("size_channel", {"size": 16.0}, False, False, id="size_channel"),
        pytest.param(
            "outline_channel",
            {"size": 12.0, "outline_level": 1},
            True,
            True,
            id="outline_channel",
        ),
    ],
)
def test_imprint_gate_demotion_cases(
    case_id: str, para_kwargs: dict, expect_raw_text: bool, check_i2: bool
) -> None:
    """Imprint lines demote at recognition time; outline ones keep I2 retention."""
    from lightrag.parser.docx.smart_heading import guardrails

    records = _body(20) + [_para("抄送：各区人民政府", **para_kwargs)]
    warnings: dict = {}
    result = _gate(
        records, strong_body=guardrails.strong_body_reason, warnings=warnings
    )
    assert "抄送：各区人民政府" not in _texts(result)
    assert len(result.demoted) == 1
    dem = result.demoted[0]
    assert dem.is_heading is False
    assert dem.rule_trail == ["imprint_marker", "strong_body_demoted"]
    assert dem.use_raw_text is expect_raw_text
    assert warnings["smart_strong_body_demotions"] == 1
    if check_i2:
        merged = list(result.decisions) + result.demoted
        assert guardrails.verify_baseline_heading_retention(records, merged) == []


def test_postmerge_sweep_demotes_imprint_heading() -> None:
    """The post-merge sweep catches an imprint that survives earlier gates.

    This covers outline/size admission and merges whose joined text newly reads
    as an imprint; unnumbered headings have no series-propagation side effects.
    """
    from lightrag.parser.docx.smart_heading import guardrails

    d = _decision("抄送：各成员单位", numbered=False)
    demote_strong_body_headings([d], strong_body=guardrails.strong_body_reason)
    assert d.is_heading is False
    assert "imprint_marker" in d.rule_trail
    assert "strong_body_demoted" in d.rule_trail


def test_outline_only_fallback_demotes_imprint() -> None:
    """The sub-document fallback keeps outlineLvl headings EXCEPT imprint
    lines, which get an explicit rule-tagged demotion (a silent skip would be
    an I2 violation). Both fallback call sites share this helper."""
    from lightrag.parser.docx.smart_heading.guardrails import (
        verify_baseline_heading_retention,
    )
    from lightrag.parser.docx.smart_heading.heading_flow import (
        _outline_only_decisions,
    )

    records = [
        _para("第一章 总体要求", size=12.0, outline_level=0),
        _para("正文段落，以句号结尾。", size=12.0),
        _para("抄送：各成员单位", size=12.0, outline_level=1),
    ]
    warnings: dict = {}
    out = _outline_only_decisions(records, range(len(records)), warnings=warnings)
    assert len(out) == 2
    normal = next(d for d in out if d.record_index == 0)
    assert normal.is_heading and normal.level == 1
    assert "subdoc_fallback_outline_only" in normal.rule_trail
    dem = next(d for d in out if d.record_index == 2)
    assert dem.is_heading is False and dem.use_raw_text is True
    assert dem.rule_trail == ["imprint_marker", "strong_body_demoted"]
    assert warnings["smart_strong_body_demotions"] == 1
    assert verify_baseline_heading_retention(records, out) == []


def test_cb4_short_subdoc_fallback_wires_imprint(monkeypatch) -> None:
    """Wiring: run_smart_heading's cb4_short_subdoc fallback demotes an
    outline imprint line and counts it — even with an injected stub
    strong_body that does not know imprint (the fallback uses the guardrails
    default marker, not the injected strong_body)."""
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    monkeypatch.setenv("DOCX_SMART_SUBDOC_MIN_TOKENS", "1000000")

    records = _body(20) + [
        _para("附则说明", size=12.0, outline_level=0),
        _para("抄送：各成员单位", size=12.0, outline_level=1),
    ]
    imprint_idx = len(records) - 1
    warnings: dict = {}
    result = run_smart_heading(
        records,
        llm_judge=None,  # uniform 12pt: no title-block candidates, no LLM
        warnings=warnings,
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    assert result.audit["sub_documents"][0]["fallback"] == "cb4_short_subdoc"
    dem = result.decisions[imprint_idx]
    assert dem.is_heading is False and dem.use_raw_text is True
    assert "imprint_marker" in dem.rule_trail
    assert result.decisions[imprint_idx - 1].is_heading  # plain outline kept
    assert warnings["smart_strong_body_demotions"] == 1


@pytest.mark.parametrize(
    "tail_records",
    [
        pytest.param(
            [_para("抄送：各成员单位", size=12.0, outline_level=1)],
            id="imprint_line",
        ),
        pytest.param(
            [
                _para("抄送：各区", size=12.0),
                _para("某某办公室 2026年 印发", size=12.0, outline_level=1),
            ],
            id="imprint_region",
        ),
    ],
)
def test_below_whole_doc_cb4_gate_returns_none(monkeypatch, tail_records) -> None:
    """Guarantee boundary: below the whole-document CB4 gate smart never runs
    (returns None) — imprint line and region rules never reach baseline output."""
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "1000000")
    records = _body(5) + list(tail_records)
    result = run_smart_heading(
        records,
        llm_judge=None,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is None


# ---------------------------------------------------------------------------
# 版记 region conditional demotion: a 抄送…印发 span followed by a valid title
# block (a 公文汇编 boundary) is force-demoted to body; otherwise veto-only.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param("multi_line_cover", id="multi_line_cover"),
        pytest.param("single_line_cover", id="single_line_cover"),
        pytest.param("no_following_title_block", id="no_following_title_block"),
    ],
)
def test_imprint_region_conditional_demotion_cases(monkeypatch, mode: str) -> None:
    """版记 region demotes when followed by a confirmed cover (multi-line or
    imprint-anchored single-line), and stays veto-only without a following cover."""
    import json

    from lightrag.parser.docx.smart_heading.guardrails import (
        verify_baseline_heading_retention,
    )
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    doc1 = _body(20)
    region = [
        _para("抄送：各区人民政府", size=12.0),
        _para("中间说明行不带大纲", size=12.0),
        _para("某某办公室 2026年6月30日 印发", size=12.0, outline_level=1),
    ]
    if mode == "multi_line_cover":
        region += [
            ParagraphRecord(kind="empty_para"),
            _para("数字政府建设白皮书", size=18.0, page_break_before=True),
            _para("（2026年版）", size=12.0),
        ]
    elif mode == "single_line_cover":
        region += [
            ParagraphRecord(kind="empty_para"),
            _para("数字政府建设白皮书", size=18.0, page_break_before=True),
        ]
    records = doc1 + region + _body(20)
    closer_idx = len(doc1) + 2
    cover_idx = len(doc1) + 4

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps(
            {"is_title_block": True, "main_title": "数字政府建设白皮书"},
            ensure_ascii=False,
        )

    warnings: dict = {}
    result = run_smart_heading(
        records,
        llm_judge=None if mode == "no_following_title_block" else _llm,
        warnings=warnings,
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    if mode == "no_following_title_block":
        assert result.decisions[closer_idx].is_heading is True
        assert "smart_imprint_region_demotions" not in warnings
        return

    d_cover = result.decisions[cover_idx]
    assert d_cover.is_title_block
    if mode == "single_line_cover":
        assert d_cover.level == 0
        assert "title_block:imprint_single" in d_cover.rule_trail
    dem = result.decisions[closer_idx]
    assert dem.is_heading is False and dem.use_raw_text is True
    assert dem.rule_trail[-2:] == ["imprint_region", "strong_body_demoted"]
    assert warnings["smart_imprint_region_demotions"] == 1
    assert (
        verify_baseline_heading_retention(records, list(result.decisions.values()))
        == []
    )


# ---------------------------------------------------------------------------
# mixed table/paragraph covers, end to end: absorbed paragraphs stay members
# in SOURCE ORDER through judgment and assembly
# ---------------------------------------------------------------------------


def test_mixed_cover_assembly_preserves_source_order(monkeypatch) -> None:
    """表A + 封面材料段 + 表B: the absorbed paragraph rides in member_indices
    between the tables, assembly renders 表A cells → 段落 → 表B cells in source
    order with no duplication, and I1 passes."""
    import json

    from lightrag.parser.docx.parse_document import (
        ParagraphRecord,
        _assemble_blocks_smart,
    )
    from lightrag.parser.docx.smart_heading import guardrails as g
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")

    def _table(rows, text="<table>[]</table>"):
        return ParagraphRecord(kind="table", text=text, table_cell_features=rows)

    records = [
        _table([[("产品标准化大纲", 22.0, False)]]),
        _para("研究策划部编写", size=16.0),  # absorbed cover-material paragraph
        _table([[("某某电子股份有限公司", 16.0, False)]]),
    ] + _body(30, size=12.0)

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps(
            {"is_title_block": True, "main_title": "产品标准化大纲"},
            ensure_ascii=False,
        )

    result = run_smart_heading(
        records,
        llm_judge=_llm,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    lead = result.decisions[0]
    assert lead.is_title_block
    assert lead.member_indices == (0, 1, 2)  # 表A, 段, 表B — source order

    blocks = _assemble_blocks_smart(records, result, {}, {})
    tb = next(b for b in blocks if b.get("is_title_block"))
    content = tb["content"]
    # Source order preserved, each fragment exactly once.
    for frag in ("产品标准化大纲", "研究策划部编写", "某某电子股份有限公司"):
        assert content.count(frag) == 1
    assert (
        content.index("产品标准化大纲")
        < content.index("研究策划部编写")
        < content.index("某某电子股份有限公司")
    )
    assert g.verify_content_preservation(records, blocks, toc_indices=set()) == []


def test_paragraph_tail_cover_end_to_end(monkeypatch) -> None:
    """档号表 + 主标题段 + body(无后置表格): the paragraph-tail cover completes
    the title-block path — the candidate stands on the paragraph's size, the
    heading is the paragraph-borne main title, assembly is lossless."""
    import json

    from lightrag.parser.docx.parse_document import (
        ParagraphRecord,
        _assemble_blocks_smart,
    )
    from lightrag.parser.docx.smart_heading import guardrails as g
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")

    form_table = ParagraphRecord(
        kind="table",
        text="<table>[]</table>",
        table_cell_features=[
            [("档 号", 10.5, False), ("", None, False)],
            [("密 级", 10.5, False), ("公开", 10.5, False)],
        ],
    )
    records = [form_table, _para("某某管理办法", size=22.0)] + _body(30, size=12.0)

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps(
            {"is_title_block": True, "main_title": "某某管理办法"},
            ensure_ascii=False,
        )

    result = run_smart_heading(
        records,
        llm_judge=_llm,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    lead = result.decisions[0]
    assert lead.is_title_block
    assert lead.member_indices == (0, 1)

    blocks = _assemble_blocks_smart(records, result, {}, {})
    tb = next(b for b in blocks if b.get("is_title_block"))
    assert tb["heading"] == "某某管理办法"
    assert tb["content"].count("某某管理办法") == 1  # member, not re-emitted
    assert g.verify_content_preservation(records, blocks, toc_indices=set()) == []


def test_rejected_mixed_cover_paragraph_regains_heading_path(monkeypatch) -> None:
    """LLM 否决取舍固化: when the LLM rejects the mixed-cover candidate, the
    absorbed 22pt paragraph re-enters the normal gate and becomes an ORDINARY
    heading (not a title block) — bounded loss, mirroring multi-window."""
    import json

    from lightrag.parser.docx.parse_document import ParagraphRecord
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    monkeypatch.setenv("DOCX_SMART_SUBDOC_MIN_TOKENS", "10")

    form_table = ParagraphRecord(
        kind="table",
        text="<table>[]</table>",
        table_cell_features=[[("档 号", 10.5, False), ("", None, False)]],
    )
    records = [
        form_table,
        _para("某某管理办法", size=22.0),
        ParagraphRecord(
            kind="table",
            text="<table>[]</table>",
            table_cell_features=[[("某某电子股份有限公司", 16.0, False)]],
        ),
    ] + _body(30, size=12.0)

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps({"is_title_block": False}, ensure_ascii=False)

    result = run_smart_heading(
        records,
        llm_judge=_llm,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    d = result.decisions.get(1)
    assert d is not None and d.is_heading and not d.is_title_block


def test_trailing_document_date_not_absorbed_into_next_cover(monkeypatch) -> None:
    """End-to-end (mirrors test5-红头文件): a 成文日期 mis-ordered right after the
    版记 must NOT become the leading member of the following 附件 cover — it
    stays body under the previous section; the cover roots at the real title."""
    import json

    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    doc1 = _body(20)
    tail = [
        _para("抄送：各设区市城乡规划局", size=12.0),  # anchor
        _para("河北省住房和城乡建设厅办公室  2009年7月6日印发", size=12.0),  # closer
        _para("二○○九年七月六日", size=12.0),  # 成文日期 (mis-ordered)
        _para("附件：", size=12.0),  # 附件 cover root
        _para("城市控制性详细规划备案工作规程", size=18.0),  # cover title
    ]
    records = doc1 + tail + _body(20)
    date_idx = len(doc1) + 2

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps(
            {"is_title_block": True, "main_title": "城市控制性详细规划备案工作规程"},
            ensure_ascii=False,
        )

    result = run_smart_heading(
        records,
        llm_judge=_llm,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    tb_roots = [d for d in result.decisions.values() if d.is_title_block]
    assert tb_roots  # 附件 cover confirmed
    # The date is never a title-block member, and is body (not a heading).
    assert all(date_idx not in d.member_indices for d in tb_roots)
    d_date = result.decisions.get(date_idx)
    assert d_date is None or not (d_date.is_heading or d_date.is_title_block)


# ---------------------------------------------------------------------------
# close_unnumbered_level_gaps: post-demotion lift for unnumbered headings
# ---------------------------------------------------------------------------


def _leveled(
    idx: int,
    text: str,
    level: int,
    *,
    plain: bool = False,
    outline: int | None = None,
    title_block: bool = False,
    anchored: bool = False,
) -> HeadingDecision:
    """A surviving post-leveling decision (numbering derived from text)."""
    return _make_decision(
        text,
        level,
        idx=idx,
        size=12.0,
        numbered=not plain,
        outline=outline,
        anchored=anchored or outline is not None,
        is_title_block=title_block,
    )


@pytest.mark.parametrize(
    ("ds_factory", "expected_levels", "moved_indices"),
    [
        pytest.param(
            # test7-专利说明书 shape: sections lift 3 -> 2 over vacated L2.
            lambda: [
                _leveled(0, "一种层级感知的文档语义分块方法及系统", 1, plain=True),
                _leveled(1, "技术领域", 3, plain=True),
                _leveled(2, "背景技术", 3, plain=True),
            ],
            [1, 2, 2],
            (1, 2),
            id="test7-vacated-class-slot",
        ),
        pytest.param(
            # Deliberate MLN raw-level gap (2 -> 4): pinned, trailing plain stays.
            lambda: [
                _leveled(0, "总标题", 1, plain=True),
                _leveled(1, "1.1 概述", 2),
                _leveled(2, "1.1.1.1 深层小节", 4),
                _leveled(3, "普通小节", 5, plain=True),
            ],
            [1, 2, 4, 5],
            (),
            id="preserves-mln-deliberate-gaps",
        ),
        pytest.param(
            # Cascading movable levels re-pack consecutively (1/3/4 -> 1/2/3).
            lambda: [
                _leveled(0, "主标题", 1, plain=True),
                _leveled(1, "加粗章", 3, plain=True),
                _leveled(2, "加粗节", 4, plain=True),
            ],
            [1, 2, 3],
            (1, 2),
            id="cascading-gaps",
        ),
        pytest.param(
            # Gap between two pinned numbered levels stays untouched.
            lambda: [
                _leveled(0, "第一章 总则", 1),
                _leveled(1, "（一）分项", 3),
                _leveled(2, "加粗小节", 4, plain=True),
            ],
            [1, 3, 4],
            (),
            id="numbered-numbered-gap-out-of-scope",
        ),
        pytest.param(
            # Mixed level (numbered + unnumbered at L3) is conservatively pinned.
            lambda: [
                _leveled(0, "主标题", 1, plain=True),
                _leveled(1, "一、编号章", 3),
                _leveled(2, "加粗章", 3, plain=True),
            ],
            [1, 3, 3],
            (),
            id="mixed-level-pinned",
        ),
        pytest.param(
            # Outline-bearing is pinned; round-2 anchored plain heading still lifts.
            lambda: [
                _leveled(0, "样式标题", 1, outline=0),
                _leveled(1, "加粗小节", 4, plain=True, anchored=True),
            ],
            [1, 2],
            (1,),
            id="outline-pinned-round2-anchored-movable",
        ),
        pytest.param(
            # Title-block root stays pinned even off a vacated shallower level.
            lambda: [
                _leveled(0, "封面主标题", 2, plain=True, title_block=True),
                _leveled(1, "加粗章", 4, plain=True),
            ],
            [2, 3],
            (1,),
            id="title-block-pinned",
        ),
    ],
)
def test_close_unnumbered_level_gaps_cases(
    ds_factory, expected_levels: list[int], moved_indices: tuple[int, ...]
) -> None:
    ds = ds_factory()
    warnings: dict = {}
    close_unnumbered_level_gaps(ds, warnings=warnings)
    assert [d.level for d in ds] == expected_levels
    for idx, d in enumerate(ds):
        if idx in moved_indices:
            assert "unnumbered_gap_closed" in d.rule_trail
        else:
            assert "unnumbered_gap_closed" not in d.rule_trail
    if moved_indices:
        assert warnings["smart_unnumbered_gap_closed"] == len(moved_indices)
    else:
        assert "smart_unnumbered_gap_closed" not in warnings


@pytest.mark.parametrize(
    ("ds_factory", "after_gap_close", "after_nest_child_level"),
    [
        pytest.param(
            # Child at L2 <= lifted parent (4 -> 3): nest lands on corrected 3+1=4.
            lambda: [
                _leveled(0, "主标题", 1, plain=True),
                _leveled(1, "加粗章", 4, plain=True),
                _leveled(2, "1. 子项", 2),
            ],
            [1, 3, 2],
            4,
            id="child-lands-on-corrected-parent",
        ),
        pytest.param(
            # Child at L4 > lifted parent (3 -> 2): nest is one-directional, stays 4.
            lambda: [
                _leveled(0, "主标题", 1, plain=True),
                _leveled(1, "加粗章", 3, plain=True),
                _leveled(2, "1. 子项", 4),
            ],
            [1, 2, 4],
            4,
            id="deeper-child-keeps-documented-gap",
        ),
    ],
)
def test_gap_close_then_nest_interactions(
    ds_factory, after_gap_close: list[int], after_nest_child_level: int
) -> None:
    ds = ds_factory()
    close_unnumbered_level_gaps(ds)
    assert [d.level for d in ds] == after_gap_close
    nest_numbered_under_parent(ds)
    assert ds[2].level == after_nest_child_level
    assert ds[2].level > ds[1].level


def test_gap_close_end_to_end_bold_sections_reach_level_2(monkeypatch) -> None:
    """End-to-end mirror of test7-专利说明书: no outline, single font size,
    a centered+bold main title, an EnNum claims series that the post-merge
    sweep demotes wholesale, and bold left-aligned section headings — the
    sections must land at L2 (not L3) after the vacated class slot closes."""
    import json

    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    records = (
        [
            _para(
                "一种层级感知的文档语义分块方法及系统",
                size=12.0,
                all_bold=True,
                alignment="center",
            )
        ]
        + [
            _para(
                f"{i}、根据权利要求所述的方法，其特征在于，包括对应的处理步骤。",
                size=12.0,
            )
            for i in range(1, 9)
        ]
        + [_para("技术领域", size=12.0, all_bold=True)]
        + _body(10)
        + [_para("背景技术", size=12.0, all_bold=True)]
        + _body(10)
    )
    title_idx = 0
    section_indices = [9, 20]

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps({"is_title_block": False}, ensure_ascii=False)

    result = run_smart_heading(
        records,
        llm_judge=_llm,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    d_title = result.decisions[title_idx]
    assert d_title.is_heading and d_title.level == 1
    for i in section_indices:
        d = result.decisions[i]
        assert d.is_heading, records[i].text
        assert d.level == 2, (records[i].text, d.level, d.rule_trail)
        assert "unnumbered_gap_closed" in d.rule_trail
    for i in range(1, 9):
        d = result.decisions.get(i)
        assert d is None or not d.is_heading


# ---------------------------------------------------------------------------
# Zero-visible-char placeholder paragraphs (pure <drawing>/<object>/<equation>)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tail_records", "expected_texts"),
    [
        pytest.param(
            # 14pt drawing-only paragraph over FS_base=12 must not take size_strong.
            [_para('<drawing id="1" path="x.emf" />', size=14.0, visible_char_count=0)],
            [],
            id="not-size-promoted",
        ),
        pytest.param(
            # Pure-OMML paragraph (size=None, centered, 0 visible chars) never candidate.
            [
                _para(
                    "<equation>E=mc^2</equation>",
                    size=None,
                    alignment="center",
                    visible_char_count=0,
                )
            ],
            [],
            id="pure-equation-never-candidate",
        ),
        pytest.param(
            # Placeholder at fs+0.5 does not act as a weak-pair companion.
            [
                _para("候选标题甲", size=12.5),
                _para('<drawing id="1" />', size=12.5, visible_char_count=0),
            ],
            [],
            id="not-weak-pair-companion",
        ),
        pytest.param(
            # Auto-numbered image line does not lend the second numbering-series vote.
            [
                _para("3.1 真实编号标题", size=12.0),
                _para('3.2 <drawing id="1" />', size=12.0, visible_char_count=0),
            ],
            [],
            id="not-numbering-series-companion",
        ),
        pytest.param(
            # Centered placeholder between centered titles is transparent in run.
            [
                _para("居中标题一", size=12.0, alignment="center"),
                _para(
                    '<drawing id="1" />',
                    size=12.0,
                    alignment="center",
                    visible_char_count=0,
                ),
                _para("居中标题二", size=12.0, alignment="center"),
                _para("居中标题三", size=12.0, alignment="center"),
            ],
            ["居中标题一", "居中标题二", "居中标题三"],
            id="transparent-in-centered-run",
        ),
    ],
)
def test_zero_weight_placeholder_gate_channels(
    tail_records: list[ParagraphRecord], expected_texts: list[str]
) -> None:
    result = _gate(_body(20) + tail_records)
    assert _texts(result) == expected_texts
    assert result.demoted == []
    assert result.veto_suppressed == []


@pytest.mark.parametrize("path", ["gate", "outline_only_fallback"])
def test_zero_weight_outline_placeholder_demotion(lightrag_caplog, path: str) -> None:
    """An outline-bearing zero-weight placeholder takes an explicit I2 demotion
    in both the normal gate and the CB4/CB1 outline-only fallback."""
    from lightrag.parser.docx.smart_heading.guardrails import (
        verify_baseline_heading_retention,
    )
    from lightrag.parser.docx.smart_heading.heading_flow import (
        _outline_only_decisions,
    )

    if path == "gate":
        records = _body(20) + [
            _para(
                '<drawing id="1" />', size=14.0, outline_level=0, visible_char_count=0
            )
        ]
        result = _gate(records)
        assert result.decisions == []
        assert len(result.demoted) == 1
        dem = result.demoted[0]
        assert dem.is_heading is False and dem.use_raw_text is True
        assert "placeholder_demoted" in dem.rule_trail
        assert "zero_visible_chars" in dem.rule_trail
        merged = list(result.decisions) + result.demoted
        assert verify_baseline_heading_retention(records, merged) == []
    else:
        records = _body(5) + [
            _para("正常大纲标题", size=12.0, outline_level=0),
            _para(
                '<drawing id="1" />', size=12.0, outline_level=0, visible_char_count=0
            ),
        ]
        warnings: dict = {}
        with lightrag_caplog.at_level(logging.WARNING, logger="lightrag"):
            out = _outline_only_decisions(
                records, range(len(records)), warnings=warnings
            )
        by_idx = {d.record_index: d for d in out}
        real = by_idx[len(records) - 2]
        assert real.is_heading and "subdoc_fallback_outline_only" in real.rule_trail
        dem = by_idx[len(records) - 1]
        assert dem.is_heading is False and dem.use_raw_text is True
        assert "placeholder_demoted" in dem.rule_trail
        assert warnings["smart_placeholder_demotions"] == 1
        assert (
            len([m for m in _log_messages(lightrag_caplog) if "placeholder" in m]) == 1
        )


@pytest.mark.parametrize("cb1_terminal", ["converged", "tripped"])
def test_placeholder_demotion_counted_once_across_cb1(
    monkeypatch, lightrag_caplog, cb1_terminal
) -> None:
    """``smart_placeholder_demotions`` counts at the FINAL adoption point
    only across both CB1 terminal states (re-estimated converged vs tripped)."""
    import json

    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")

    if cb1_terminal == "converged":
        records = _body(60, size=10.5)
        records += [_para(f"伪标题短语{i}", size=12.0) for i in range(30)]
    else:
        records = []
        for i in range(40):
            records.append(_para(f"{i + 1}. 选择题题干第{i}题", size=12.0))
            records.append(_para("A. 选项甲  B. 选项乙", size=10.5))
    records.append(
        _para('<drawing id="1" />', size=12.0, outline_level=0, visible_char_count=0)
    )
    placeholder_idx = len(records) - 1

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps({"is_title_block": False}, ensure_ascii=False)

    warnings: dict = {}
    with lightrag_caplog.at_level(logging.WARNING, logger="lightrag"):
        result = run_smart_heading(
            records,
            llm_judge=_llm,
            warnings=warnings,
            strong_body=_stub_strong_body,
            numbering_veto=_stub_no_veto,
            caption_veto=_stub_no_caption,
        )
    assert result is not None
    sub_docs = result.audit["sub_documents"]
    assert any("cb1_reestimated_fs" in s for s in sub_docs)
    if cb1_terminal == "tripped":
        assert any(s.get("fallback") == "cb1_density" for s in sub_docs)
    else:
        assert not any(s.get("fallback") == "cb1_density" for s in sub_docs)
    assert warnings["smart_placeholder_demotions"] == 1
    dem = result.decisions[placeholder_idx]
    assert dem.is_heading is False
    assert "placeholder_demoted" in dem.rule_trail
    i2_logs = [
        m
        for m in _log_messages(lightrag_caplog)
        if "zero-visible-char outline paragraph" in m
    ]
    assert len(i2_logs) == 1


def test_zero_weight_llm_grant_leaves_rejected_ledger() -> None:
    """A granted placeholder line lands in ``grant_rejected`` and the
    final-adoption merge counts it once and inserts the ledger row."""
    from lightrag.parser.docx.smart_heading.heading_flow import _merge_ledger_only

    records = _body(20) + [_para('<drawing id="1" />', size=14.0, visible_char_count=0)]
    idx = len(records) - 1
    result = _gate(records, llm_heading_grants={idx})
    assert result.decisions == []
    assert [d.record_index for d in result.grant_rejected] == [idx]
    trail = result.grant_rejected[0].rule_trail
    assert "zero_visible_chars" in trail
    assert "llm_grant_rejected" in trail

    decisions: dict = {}
    warnings: dict = {}
    _merge_ledger_only(decisions, result, warnings)
    assert warnings["smart_llm_grant_rejected"] == 1
    assert decisions[idx] is result.grant_rejected[0]


# ---------------------------------------------------------------------------
# mid-document title-block gate: end-to-end + audit enrichment
# ---------------------------------------------------------------------------


def test_mid_document_isolated_title_never_reaches_llm(monkeypatch) -> None:
    """The test11 shape end to end: a page-break 16pt caption line beside a
    metadata line must not form any title-block window — zero LLM calls, no
    candidates, no doc_title; the line stands as an ordinary size_strong
    level-1 heading and the document stays ONE sub-document."""
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    records = (
        _body(20)
        + [
            _para("填报单位：某某公司", size=12.0),
            _para("外购外协价格明细表", size=16.0, page_break_before=True),
        ]
        + _body(20)
    )
    title_idx = 21

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        raise AssertionError("LLM must not be called for a gated document")

    warnings: dict = {}
    result = run_smart_heading(
        records,
        llm_judge=_llm,
        warnings=warnings,
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    assert result.audit["llm_calls"] == 0
    assert result.audit["title_block_candidates"] == []
    assert result.doc_title is None
    d = result.decisions[title_idx]
    assert d.is_heading and not d.is_title_block
    assert d.level == 1
    assert "size_strong" in d.rule_trail
    assert len(result.audit["sub_documents"]) == 1


def test_title_block_candidate_audit_records_verdict_content(monkeypatch) -> None:
    """Audit enrichment: each candidate row carries the window end, the LLM's
    main_title (true verdicts) and the partition strength (false verdicts)."""
    import json

    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    records = [
        _para("产品发布白皮书", size=22.0),
        _para("（2026年版）", size=12.0),
    ] + _body(30)

    def _true_llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps(
            {"is_title_block": True, "main_title": "产品发布白皮书"},
            ensure_ascii=False,
        )

    result = run_smart_heading(
        records,
        llm_judge=_true_llm,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    (row,) = result.audit["title_block_candidates"]
    assert row["trigger"] == "multi_window"
    assert (row["start"], row["end"]) == (0, 2)
    assert row["is_title_block"] is True
    assert row["main_title"] == "产品发布白皮书"
    assert row["heading_grants"] == 0 and row["body_vetoes"] == 0

    def _false_llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps(
            {"is_title_block": False, "headings": [], "body": [0, 1]},
            ensure_ascii=False,
        )

    result2 = run_smart_heading(
        records,
        llm_judge=_false_llm,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result2 is not None
    (row2,) = result2.audit["title_block_candidates"]
    assert row2["is_title_block"] is False
    assert row2["main_title"] is None
    assert row2["body_vetoes"] == 2


def test_imprint_single_cover_behind_logo_confirms_region(monkeypatch) -> None:
    """Review regression: the imprint confirm scan must walk with the SAME
    eyes as the imprint_single channel — a pure logo paragraph + section
    break between the 版记 tail and the single-line cover must not stop the
    scan, or the confirmed cover leaves the region undemoted."""
    import json

    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    doc1 = _body(20)
    records = (
        doc1
        + [
            _para("抄送：各区人民政府", size=12.0),
            _para("中间说明行不带大纲", size=12.0),
            _para("某某办公室 2026年6月30日 印发", size=12.0, outline_level=1),
            _para('<drawing id="seal-1" />', size=12.0, visible_char_count=0),
            ParagraphRecord(kind="section_break"),
        ]
        + [_para("数字政府建设白皮书", size=18.0)]
        + _body(20)
    )
    closer_idx = len(doc1) + 2
    cover_idx = len(doc1) + 5

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps(
            {"is_title_block": True, "main_title": "数字政府建设白皮书"},
            ensure_ascii=False,
        )

    warnings: dict = {}
    result = run_smart_heading(
        records,
        llm_judge=_llm,
        warnings=warnings,
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    d_cover = result.decisions[cover_idx]
    assert d_cover.is_title_block and d_cover.level == 0
    assert "title_block:imprint_single" in d_cover.rule_trail
    dem = result.decisions[closer_idx]
    assert dem.is_heading is False and dem.use_raw_text is True
    assert dem.rule_trail[-2:] == ["imprint_region", "strong_body_demoted"]
    assert warnings["smart_imprint_region_demotions"] == 1


def test_logo_led_multi_cover_still_confirms_region(monkeypatch) -> None:
    """A multi window that OPENS on a seal/logo image paragraph keys its
    decision on that image; the confirm scan (which skips non-content
    records) lands on the block's title LINE — matching the full member set
    keeps the 版记 confirmation working (matching starts only would not)."""
    import json

    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    doc1 = _body(20)
    records = (
        doc1
        + [
            _para("抄送：各区人民政府", size=12.0),
            _para("中间说明行不带大纲", size=12.0),
            _para("某某办公室 2026年6月30日 印发", size=12.0, outline_level=1),
            ParagraphRecord(kind="empty_para"),
            _para('<drawing id="seal-1" />', size=12.0, visible_char_count=0),
            _para("数字政府建设白皮书", size=18.0),
        ]
        + _body(20)
    )
    closer_idx = len(doc1) + 2
    logo_idx = len(doc1) + 4

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps(
            {"is_title_block": True, "main_title": "数字政府建设白皮书"},
            ensure_ascii=False,
        )

    warnings: dict = {}
    result = run_smart_heading(
        records,
        llm_judge=_llm,
        warnings=warnings,
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    d_cover = result.decisions[logo_idx]  # image-led window keys on the logo
    assert d_cover.is_title_block and d_cover.level == 0
    assert "title_block:multi_window" in d_cover.rule_trail
    dem = result.decisions[closer_idx]
    assert dem.is_heading is False
    assert dem.rule_trail[-2:] == ["imprint_region", "strong_body_demoted"]
    assert warnings["smart_imprint_region_demotions"] == 1


# ---------------------------------------------------------------------------
# merged doc-title length bounding — four-way consistency (H1 / doc_title /
# first_heading / descendant parent_headings root) under truncation
# ---------------------------------------------------------------------------


def _assemble_bounded_title(monkeypatch, cap_env: str, main: str, sub):
    """Drive the assembler for a level-0 title block whose composed doc-title is
    (potentially) truncated, plus one child heading, and return
    (doc_heading, blocks, meta)."""
    from lightrag.parser.docx.parse_document import _assemble_blocks_smart
    from lightrag.parser.docx.smart_heading.heading_flow import SmartHeadingResult
    from lightrag.parser.docx.smart_heading.title_block import compose_doc_title

    monkeypatch.setenv("DOCX_SMART_HEADING_MAX_CHARS", cap_env)
    doc_heading = compose_doc_title(main, sub)

    records = [
        _para(main, size=22.0),  # 0: title-block lead
        _para("封面说明正文一句。", size=12.0),  # 1: body owned by the title block
        _para("第一章 总则", size=14.0),  # 2: child heading
        _para("章节正文一句。", size=12.0),  # 3: body under the child
    ]
    result = SmartHeadingResult(
        decisions={
            0: HeadingDecision(
                record_index=0,
                text=main,
                is_heading=True,
                is_title_block=True,
                level=0,
                doc_title_heading=doc_heading,
                member_indices=(0,),
            ),
            2: HeadingDecision(
                record_index=2, text="第一章 总则", is_heading=True, level=1
            ),
        },
        toc_indices=set(),
        doc_title=doc_heading,
        audit={},
    )
    meta: dict = {}
    blocks = _assemble_blocks_smart(records, result, {}, meta)
    return doc_heading, blocks, meta


def _assert_four_way(doc_heading, blocks, meta) -> None:
    title = next(b for b in blocks if b.get("is_title_block"))
    child = next(b for b in blocks if b["heading"] == "第一章 总则")
    assert title["heading"] == doc_heading
    assert meta["doc_title"] == doc_heading
    assert meta["first_heading"] == doc_heading
    assert child["parent_headings"][0] == doc_heading


@pytest.mark.parametrize(
    ("cap_env", "main", "sub", "check_raw_ceiling"),
    [
        pytest.param(
            "default", "标" * 60, "案" * 30, False, id="merged-weighted-truncation"
        ),
        pytest.param("300", "x" * 130, "y" * 130, True, id="raw-ceiling-above-cap"),
        pytest.param("default", "则" * 70, None, False, id="main-only-long-title"),
    ],
)
def test_merged_doc_title_truncation_four_way(
    monkeypatch, cap_env: str, main: str, sub: str | None, check_raw_ceiling: bool
) -> None:
    """Merged and main-only long doc titles truncate identically across all
    four landing sites (H1, doc_title, first_heading, child parent_headings)."""
    from lightrag.constants import (
        DEFAULT_DOCX_SMART_HEADING_MAX_CHARS,
        MAX_HEADING_LENGTH,
    )
    from lightrag.parser.docx.smart_heading.guardrails import weighted_char_length

    env_val = (
        str(DEFAULT_DOCX_SMART_HEADING_MAX_CHARS) if cap_env == "default" else cap_env
    )
    doc_heading, blocks, meta = _assemble_bounded_title(monkeypatch, env_val, main, sub)
    assert doc_heading.endswith("...")
    if check_raw_ceiling:
        assert len(doc_heading) == MAX_HEADING_LENGTH
    else:
        assert weighted_char_length(doc_heading) <= DEFAULT_DOCX_SMART_HEADING_MAX_CHARS
    _assert_four_way(doc_heading, blocks, meta)


def test_run_smart_heading_rejects_invalid_max_chars_env(monkeypatch) -> None:
    """Parse-time entry check: a structurally invalid DOCX_SMART_HEADING_MAX_CHARS
    hard-fails the parse on EVERY entry point (library / per-file hint), matching
    the API startup check, instead of silently defaulting."""
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_HEADING_MAX_CHARS", "2")
    with pytest.raises(ValueError, match="DOCX_SMART_HEADING_MAX_CHARS"):
        run_smart_heading([], llm_judge=None, warnings={})

    monkeypatch.setenv("DOCX_SMART_HEADING_MAX_CHARS", "abc")
    with pytest.raises(ValueError, match="DOCX_SMART_HEADING_MAX_CHARS"):
        run_smart_heading([], llm_judge=None, warnings={})


def test_smart_assembly_gives_bibliography_its_own_references_block(monkeypatch):
    """Records inside a bibliography control form a session_type=references
    block in the smart assembler too, keeping the heading they continue."""
    from lightrag.parser.docx.parse_document import _assemble_blocks_smart
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    records = _body(20) + [
        _para("1 结论", size=16.0, outline_level=0),
        *_body(3),
        _para("[1] 张三：知识图谱综述", in_references=True),
        _para("[2] 李四：检索增强生成", in_references=True),
        _para("以上为全文结束语。"),
    ]

    result = run_smart_heading(
        records,
        llm_judge=_no_title_llm,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    blocks = _assemble_blocks_smart(records, result, {}, {})

    refs = [b for b in blocks if b.get("session_type") == "references"]
    assert len(refs) == 1
    assert refs[0]["content"] == "[1] 张三：知识图谱综述\n[2] 李四：检索增强生成"
    assert refs[0]["heading"] == "1 结论"
    closing = next(b for b in blocks if "以上为全文结束语。" in b["content"])
    assert "session_type" not in closing


def test_title_block_never_takes_a_bibliography_record(monkeypatch):
    """A bibliography entry right before the cover cannot join the title
    block: drop_references would drop the title and author with it."""
    import json

    from lightrag.parser.docx.parse_document import _assemble_blocks_smart
    from lightrag.parser.docx.smart_heading.heading_flow import run_smart_heading

    monkeypatch.setenv("DOCX_SMART_MIN_TOKENS", "10")
    records = [
        _para("[1] 张三：某论文", size=14.0, in_references=True),
        _para("行业观察白皮书", size=14.0),
        _para("某某研究院", size=12.0),
    ]
    records += _body(15, size=12.0)
    records.append(_para("第一章 总则", size=12.0, outline_level=0))
    records += _body(15, size=12.0)

    def _llm(prompt: str, *, system_prompt: str | None = None) -> str:
        return json.dumps(
            {"is_title_block": True, "main_title": "行业观察白皮书"},
            ensure_ascii=False,
        )

    result = run_smart_heading(
        records,
        llm_judge=_llm,
        warnings={},
        strong_body=_stub_strong_body,
        numbering_veto=_stub_no_veto,
        caption_veto=_stub_no_caption,
    )
    assert result is not None
    blocks = _assemble_blocks_smart(records, result, {}, {})

    title = next(b for b in blocks if b.get("is_title_block"))
    assert title["heading"] == "行业观察白皮书"
    assert "session_type" not in title
    assert "某某研究院" in title["content"]
    refs = [b for b in blocks if b.get("session_type") == "references"]
    assert [b["content"] for b in refs] == ["[1] 张三：某论文"]
