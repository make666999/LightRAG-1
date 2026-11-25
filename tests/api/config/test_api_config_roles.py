"""Per-role API config parsing: MAX_ASYNC_LLM, {ROLE}_LLM_TIMEOUT, and VLM_PROCESS_ENABLE."""

from __future__ import annotations

import sys

import pytest

from lightrag.api.config import parse_args
from lightrag.constants import DEFAULT_MAX_ASYNC

pytestmark = pytest.mark.offline

_ROLE_MAX_ASYNC_ENV_KEYS = (
    "MAX_ASYNC_LLM",
    "MAX_ASYNC",
    "EXTRACT_MAX_ASYNC_LLM",
    "KEYWORD_MAX_ASYNC_LLM",
    "QUERY_MAX_ASYNC_LLM",
    "VLM_MAX_ASYNC_LLM",
)

_VLM_AND_TIMEOUT_ENV_KEYS = (
    "VLM_PROCESS_ENABLE",
    "VLM_LLM_BINDING",
    "VLM_LLM_MODEL",
    "VLM_LLM_BINDING_HOST",
    "VLM_LLM_BINDING_API_KEY",
    "VLM_LLM_TIMEOUT",
    "EXTRACT_LLM_TIMEOUT",
    "KEYWORD_LLM_TIMEOUT",
    "QUERY_LLM_TIMEOUT",
    "LLM_TIMEOUT_VLM_LLM",
    "LLM_TIMEOUT_EXTRACT_LLM",
    "LLM_TIMEOUT_KEYWORD_LLM",
    "LLM_TIMEOUT_QUERY_LLM",
)


class TestRoleMaxAsyncConfig:
    @pytest.fixture(autouse=True)
    def _clear_env(self, monkeypatch):
        for key in _ROLE_MAX_ASYNC_ENV_KEYS:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(sys, "argv", ["lightrag-server"])

    def test_role_max_async_env_override_keeps_other_roles_none(self, monkeypatch):
        monkeypatch.setenv("MAX_ASYNC", "10")
        monkeypatch.setenv("EXTRACT_MAX_ASYNC_LLM", "7")

        args = parse_args()

        assert args.max_async == 10
        assert args.extract_llm_max_async == 7
        assert args.keyword_llm_max_async is None
        assert args.query_llm_max_async is None
        assert args.vlm_llm_max_async is None

    def test_role_max_async_literal_none_string_is_preserved(self, monkeypatch):
        monkeypatch.setenv("MAX_ASYNC", "10")
        monkeypatch.setenv("QUERY_MAX_ASYNC_LLM", "None")

        args = parse_args()

        assert args.max_async == 10
        assert args.query_llm_max_async is None

    def test_max_async_llm_takes_precedence_over_legacy(self, monkeypatch):
        monkeypatch.setenv("MAX_ASYNC_LLM", "8")
        monkeypatch.setenv("MAX_ASYNC", "10")

        assert parse_args().max_async == 8

    def test_max_async_defaults_when_unset(self):
        assert parse_args().max_async == DEFAULT_MAX_ASYNC


class TestRoleTimeoutResolution:
    """``{ROLE}_LLM_TIMEOUT`` overrides ``LLM_TIMEOUT``; explicit 0 means no timeout."""

    @pytest.mark.parametrize(
        ("query_timeout_env", "expected"),
        [
            pytest.param(None, 240, id="unset-inherits-base"),
            pytest.param("120", 120, id="explicit-override"),
            pytest.param("0", 0, id="zero-means-no-timeout"),
        ],
    )
    def test_role_timeout_resolution(
        self, prepare_startup, query_timeout_env, expected
    ):
        startup = prepare_startup(
            {
                "LLM_TIMEOUT": "240",
                "QUERY_LLM_TIMEOUT": query_timeout_env,
                "EXTRACT_LLM_TIMEOUT": None,
            }
        )

        role_configs = startup.constructor_kwargs()["role_llm_configs"]

        assert role_configs["query"].timeout == expected
        assert role_configs["extract"].timeout == 240


class TestVlmEnableAndRoleTimeouts:
    @pytest.fixture(autouse=True)
    def _reset_vlm_env(self, monkeypatch):
        for key in _VLM_AND_TIMEOUT_ENV_KEYS:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(sys, "argv", ["lightrag-server"])

    def test_vlm_process_enable_defaults_to_false(self, monkeypatch):
        monkeypatch.setenv("LLM_BINDING", "openai")
        assert parse_args().vlm_process_enable is False

    def test_vlm_process_enable_true_with_openai_passes(self, monkeypatch):
        monkeypatch.setenv("LLM_BINDING", "openai")
        monkeypatch.setenv("VLM_PROCESS_ENABLE", "true")
        assert parse_args().vlm_process_enable is True

    def test_vlm_process_enable_rejects_lollms_base_binding(self, monkeypatch):
        monkeypatch.setenv("LLM_BINDING", "lollms")
        monkeypatch.setenv("VLM_PROCESS_ENABLE", "true")

        with pytest.raises(SystemExit) as exc:
            parse_args()
        assert "lollms" in str(exc.value).lower()

    def test_vlm_process_enable_rejects_lollms_role_binding(self, monkeypatch):
        monkeypatch.setenv("LLM_BINDING", "openai")
        monkeypatch.setenv("VLM_PROCESS_ENABLE", "true")
        monkeypatch.setenv("VLM_LLM_BINDING", "lollms")
        monkeypatch.setenv("VLM_LLM_MODEL", "anything")
        monkeypatch.setenv("VLM_LLM_BINDING_HOST", "http://localhost:9600")
        monkeypatch.setenv("VLM_LLM_BINDING_API_KEY", "placeholder")

        with pytest.raises(SystemExit) as exc:
            parse_args()
        assert "lollms" in str(exc.value).lower()

    def test_role_timeout_uses_new_variable_names(self, monkeypatch):
        monkeypatch.setenv("LLM_BINDING", "openai")
        monkeypatch.setenv("EXTRACT_LLM_TIMEOUT", "240")
        monkeypatch.setenv("KEYWORD_LLM_TIMEOUT", "120")
        monkeypatch.setenv("QUERY_LLM_TIMEOUT", "60")
        monkeypatch.setenv("VLM_LLM_TIMEOUT", "300")

        args = parse_args()

        assert args.extract_llm_timeout == 240
        assert args.keyword_llm_timeout == 120
        assert args.query_llm_timeout == 60
        assert args.vlm_llm_timeout == 300

    def test_role_timeout_legacy_variables_no_longer_have_effect(self, monkeypatch):
        monkeypatch.setenv("LLM_BINDING", "openai")
        monkeypatch.setenv("LLM_TIMEOUT_EXTRACT_LLM", "999")
        monkeypatch.setenv("LLM_TIMEOUT_VLM_LLM", "888")

        args = parse_args()

        assert args.extract_llm_timeout is None
        assert args.vlm_llm_timeout is None
