"""Tests for ``get_env_value`` and ``parse_optional_float`` environment parsing.

Covers:
- Unconvertible non-blank values fall back to default and emit a warning naming
  the variable, raw value, and target type.
- Blank/whitespace values fall back silently without warning.
- Valid int/float/bool values (including whitespace-padded bool aliases) parse
  without warning; unrecognized bool spellings fall back with a warning.
- Non-finite floats (``nan``, ``inf``, ``-inf``, ``+inf``) are rejected by both
  ``get_env_value`` and ``parse_optional_float``.
"""

from __future__ import annotations

import logging
import math

import pytest

from lightrag.utils import get_env_value, parse_optional_float

pytestmark = pytest.mark.offline

ENV_KEY = "LIGHTRAG_TEST_CONVERT_ENV"


def _warnings(records, env_key: str = ENV_KEY):
    return [
        r.getMessage()
        for r in records
        if r.levelno == logging.WARNING and env_key in r.getMessage()
    ]


@pytest.mark.parametrize(
    ("raw", "value_type", "default"),
    [
        ("abc", int, 7),
        ("1.5", int, 7),
        ("off", int, 7),
        ("1OO", int, 7),
        ("12x", float, 0.5),
    ],
)
def test_unconvertible_value_falls_back_with_warning(
    raw, value_type, default, monkeypatch, lightrag_log_records
):
    monkeypatch.setenv(ENV_KEY, raw)
    assert get_env_value(ENV_KEY, default, value_type) == default
    messages = _warnings(lightrag_log_records)
    assert len(messages) == 1
    assert repr(raw) in messages[0]
    assert value_type.__name__ in messages[0]


@pytest.mark.parametrize("raw", ["", "  ", "\t"])
def test_blank_value_falls_back_silently(raw, monkeypatch, lightrag_log_records):
    monkeypatch.setenv(ENV_KEY, raw)
    assert get_env_value(ENV_KEY, 7, int) == 7
    assert _warnings(lightrag_log_records) == []


def test_convertible_int_and_float_do_not_warn(monkeypatch, lightrag_log_records):
    monkeypatch.setenv(ENV_KEY, "42")
    assert get_env_value(ENV_KEY, 7, int) == 42
    monkeypatch.setenv(ENV_KEY, "0.42")
    assert get_env_value(ENV_KEY, 1.5, float) == pytest.approx(0.42)
    assert _warnings(lightrag_log_records) == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True),
        (" true ", True),
        ("\tyes\t", True),
        (" 1 ", True),
        (" ON ", True),
        ("off", False),
        ("  false\n", False),
        ("0", False),
        ("", False),
    ],
)
def test_recognized_and_padded_bool_does_not_warn(
    raw, expected, monkeypatch, lightrag_log_records
):
    monkeypatch.setenv(ENV_KEY, raw)
    assert get_env_value(ENV_KEY, not expected, bool) is expected
    assert _warnings(lightrag_log_records) == []


@pytest.mark.parametrize("default", [True, False])
@pytest.mark.parametrize("raw", ["ture", "enabled", "2"])
def test_unrecognized_bool_falls_back_with_warning(
    raw, default, monkeypatch, lightrag_log_records
):
    monkeypatch.setenv(ENV_KEY, raw)
    assert get_env_value(ENV_KEY, default, bool) is default
    messages = _warnings(lightrag_log_records)
    assert len(messages) == 1
    assert repr(raw) in messages[0]


@pytest.mark.parametrize("raw", ["nan", "NaN", "inf", "-inf", "+inf"])
def test_get_env_value_rejects_nonfinite_float(raw, monkeypatch):
    monkeypatch.setenv("LIGHTRAG_TEST_FLOAT_ENV", raw)
    assert get_env_value("LIGHTRAG_TEST_FLOAT_ENV", 1.5, float) == 1.5


def test_get_env_value_missing_uses_default(monkeypatch):
    monkeypatch.delenv("LIGHTRAG_TEST_FLOAT_ENV", raising=False)
    value = get_env_value("LIGHTRAG_TEST_FLOAT_ENV", 1.5, float)
    assert value == 1.5
    assert math.isfinite(value)


@pytest.mark.parametrize("raw", ["nan", "NaN", "inf", "+inf", "-inf"])
def test_parse_optional_float_rejects_nonfinite(raw: str):
    with pytest.raises(ValueError, match="finite"):
        parse_optional_float(raw)


def test_parse_optional_float_accepts_finite_and_unset():
    assert parse_optional_float("1.5") == 1.5
    assert parse_optional_float(None) is None
    assert parse_optional_float("") is None
    assert parse_optional_float("None") is None
