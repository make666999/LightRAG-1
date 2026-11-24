"""Shared non-collected test helpers for ``tests/parser/docx/test_smart_heading_*.py``.

Centralizes the minimal record/decision factories, deterministic guard stubs,
and ``lightrag_caplog`` helpers reused across the smart-heading test modules.
"""

from __future__ import annotations

from lightrag.parser.docx.parse_document import ParagraphRecord
from lightrag.parser.docx.smart_heading.heading_flow import HeadingDecision
from lightrag.parser.docx.smart_heading.style_key import classify_numbering


def make_para(text: str, *, size: float | None = 12.0, **kw) -> ParagraphRecord:
    """Build a ``ParagraphRecord(kind="para", ...)`` with a default 12pt font."""
    return ParagraphRecord(kind="para", text=text, font_size_pt=size, **kw)


def make_empty_para(**kw) -> ParagraphRecord:
    """Build an empty paragraph record."""
    return ParagraphRecord(kind="empty_para", **kw)


def make_body_paras(n: int = 20, size: float = 12.0) -> list[ParagraphRecord]:
    """Build ``n`` normal-length body paragraphs to anchor ``FS_base`` weight."""
    return [
        make_para(
            f"这是第{i}段正常长度的正文内容，用来撑起基准字号的统计权重。", size=size
        )
        for i in range(n)
    ]


def make_decision(
    text: str,
    level: int | None = None,
    *,
    idx: int = 0,
    size: float | None = 14.0,
    numbered: bool = False,
    outline: int | None = None,
    anchored: bool = False,
    centered: bool = False,
    is_title_block: bool = False,
    is_heading: bool = True,
) -> HeadingDecision:
    """Build a ``HeadingDecision`` with optional numbering classification."""
    return HeadingDecision(
        record_index=idx,
        text=text,
        is_heading=is_heading,
        level=level,
        font_size_pt=size,
        centered=centered,
        outline_level=outline,
        anchored=anchored,
        is_title_block=is_title_block,
        numbering=classify_numbering(text) if numbered else None,
    )


def stub_strong_body(text: str) -> str | None:
    """Deterministic strong-body stand-in (no spaCy): sentence enders / long text."""
    stripped = text.strip()
    if stripped.endswith(("。", "？", "！")) or len(stripped) > 60:
        return "strong_body_stub"
    return None


def stub_no_veto(_classification, _text) -> str | None:
    """Deterministic numbering-veto stub that never vetoes."""
    return None


def stub_always_veto(_classification, _text) -> str | None:
    """Deterministic numbering-veto stub that always vetoes."""
    return "homophone_stub"


def stub_no_caption(_text) -> str | None:
    """Deterministic caption-veto stub that never vetoes."""
    return None


def log_messages(caplog) -> list[str]:
    """Return formatted messages captured by ``lightrag_caplog``."""
    return [record.getMessage() for record in caplog.records]
