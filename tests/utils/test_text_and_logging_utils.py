"""Tests for text sanitization, Unicode decoding, safe logging, and exception prefixing.

Covers:
- ``strip_control_characters``: C0/DEL/surrogate deletion, whitespace and
  markup preservation, CJK continuity, and custom ``replacement_char``.
- ``safe_unicode_decode``: supplementary Unicode escapes, already-decoded
  UTF-8, and malformed escape preservation.
- ``safe_log_value``: log-injection control-character neutralization, Unicode
  preservation, and truncate-before-sanitize ordering.
- ``log_without_raising``: normal emission and sink-exception suppression.
- ``create_prefixed_exception``: reconstructable exceptions, ``JSONDecodeError``,
  and keyword-only exception constructors.
"""

from __future__ import annotations

import json

import pytest

from lightrag.utils import (
    create_prefixed_exception,
    log_without_raising,
    safe_log_value,
    safe_unicode_decode,
    strip_control_characters,
)

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# strip_control_characters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("a\x1cb\x1dc\x1ed\x1fe", "abcde", id="fs-gs-rs-us-separators"),
        pytest.param("x\x00\x08\x0b\x0c\x0e\x1f\x7fy", "xy", id="other-c0-and-del"),
        pytest.param(
            "line1\tcol\nline2\r\nline3",
            "line1\tcol\nline2\r\nline3",
            id="preserves-tab-lf-crlf",
        ),
        pytest.param(
            "  普通文本 with <table id='t'>markup</table> &lt;kept&gt;  ",
            "  普通文本 with <table id='t'>markup</table> &lt;kept&gt;  ",
            id="clean-text-and-markup-unchanged",
        ),
        pytest.param("", "", id="empty"),
        pytest.param("ok\ud800tail", "oktail", id="surrogates"),
        # Critical: deletion (not space-substitution) so CJK text is not split.
        pytest.param("中\x1f文\x1c内\x1d容", "中文内容", id="cjk-without-spaces"),
    ],
)
def test_strip_control_characters_default_deletion(raw, expected):
    assert strip_control_characters(raw) == expected


def test_strip_control_characters_replacement_char_can_substitute():
    assert strip_control_characters("a\x1fb", replacement_char="_") == "a_b"


# ---------------------------------------------------------------------------
# safe_unicode_decode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (rb"\ud83d\ude00", "😀"),
        (rb"\uD834\uDD1E", "𝄞"),
        (rb"\u4f60\ud83d\ude00\u597d", "你😀好"),
        ("Already decoded: 😀".encode("utf-8"), "Already decoded: 😀"),
        (rb"Keep \utils, \n and \uZZZZ", r"Keep \utils, \n and \uZZZZ"),
        (b"", ""),
    ],
)
def test_safe_unicode_decode_keeps_utf8_text(content: bytes, expected: str):
    result = safe_unicode_decode(content)

    assert result == expected
    assert result.encode("utf-8").decode("utf-8") == expected


# ---------------------------------------------------------------------------
# safe_log_value
# ---------------------------------------------------------------------------


def test_safe_log_value_neutralizes_control_characters():
    # CR/LF would otherwise let a caller forge an extra log line.
    assert safe_log_value("a\r\nb") == "a??b"
    assert safe_log_value("admin\nCRITICAL forged") == "admin?CRITICAL forged"


def test_safe_log_value_truncates_over_long_values():
    long = safe_log_value("x" * 500, max_length=100)
    assert long.startswith("x" * 100)
    assert "truncated" in long
    assert len(long) < 500


def test_safe_log_value_preserves_printable_unicode():
    assert safe_log_value("报告.docx") == "报告.docx"


def test_safe_log_value_truncates_before_sanitizing():
    assert safe_log_value("ab\ncd" + "\n" * 50, max_length=5) == "ab?cd…(truncated)"
    assert safe_log_value("abcde", max_length=5) == "abcde"


# ---------------------------------------------------------------------------
# log_without_raising
# ---------------------------------------------------------------------------


def test_log_without_raising_emits_message():
    emitted: list[str] = []

    log_without_raising(emitted.append, "hello")

    assert emitted == ["hello"]


def test_log_without_raising_swallows_sink_failure():
    error = RuntimeError("sink boom")

    def boom(message):
        raise error

    assert log_without_raising(boom, "hello") is None


# ---------------------------------------------------------------------------
# create_prefixed_exception
# ---------------------------------------------------------------------------


def test_reconstructable_exception_keeps_type_and_prefix():
    result = create_prefixed_exception(ValueError("boom"), "`entity`")
    assert isinstance(result, ValueError)
    assert str(result) == "`entity`: boom"


def test_jsondecodeerror_does_not_raise_and_preserves_message():
    with pytest.raises(json.JSONDecodeError) as exc_info:
        json.loads('{\n  "x": "abc')  # unterminated string -> JSONDecodeError
    result = create_prefixed_exception(exc_info.value, "`entity`")
    # JSONDecodeError(msg) is missing (doc, pos): falls back to a clean RuntimeError
    assert isinstance(result, RuntimeError)
    assert result.args
    assert "`entity`" in str(result)
    assert "JSONDecodeError" in str(result)
    assert "Unterminated string" in str(result)


def test_keyword_only_constructor_exception_falls_back_cleanly():
    """Mimics openai.APIStatusError: keyword-only required args."""

    class KeywordOnlyError(Exception):
        def __init__(self, message, *, response, body):
            super().__init__(message)

    exc = KeywordOnlyError("Error code: 500", response=object(), body=None)
    result = create_prefixed_exception(exc, "chunk-1")
    assert isinstance(result, RuntimeError)
    assert str(result) == "chunk-1: KeywordOnlyError: Error code: 500"
