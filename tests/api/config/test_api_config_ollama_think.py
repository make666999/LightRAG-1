"""OllamaLLMOptions.think parsing (global and per-role) and startup version validation."""

from __future__ import annotations

import argparse
import sys
from unittest.mock import MagicMock, patch

import pytest

from lightrag.api.config import parse_args
from lightrag.llm.binding_options import OllamaLLMOptions

pytestmark = pytest.mark.offline


class TestOllamaRoleThinkParsing:
    @pytest.fixture(autouse=True)
    def _ollama_llm_binding(self, monkeypatch):
        monkeypatch.setenv("LLM_BINDING", "ollama")
        monkeypatch.setattr(sys, "argv", ["lightrag-server"])

    def test_think_is_absent_when_unset(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_LLM_THINK", raising=False)
        assert "think" not in OllamaLLMOptions.options_dict(parse_args())

    def test_think_is_configurable_per_role(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_LLM_THINK", raising=False)
        monkeypatch.setenv("QUERY_OLLAMA_LLM_THINK", "false")

        args = parse_args()

        assert OllamaLLMOptions.options_dict_for_role(args, "query")["think"] is False
        assert "think" not in OllamaLLMOptions.options_dict_for_role(args, "extract")

    def test_an_empty_value_means_thinking_off_not_unset(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_LLM_THINK", "")
        assert OllamaLLMOptions.options_dict(parse_args())["think"] is False

    @pytest.mark.parametrize("level", ["low", "medium", "high"])
    def test_think_accepts_ollama_reasoning_levels(self, monkeypatch, level):
        monkeypatch.setenv("OLLAMA_LLM_THINK", level.upper())
        monkeypatch.setenv("EXTRACT_OLLAMA_LLM_THINK", level)

        args = parse_args()

        assert OllamaLLMOptions.options_dict(args)["think"] == level
        assert OllamaLLMOptions.options_dict_for_role(args, "extract")["think"] == level

    def test_booleans_still_parse_as_booleans_alongside_the_levels(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_LLM_THINK", "false")
        monkeypatch.setenv("EXTRACT_OLLAMA_LLM_THINK", "on")

        args = parse_args()

        assert OllamaLLMOptions.options_dict(args)["think"] is False
        assert OllamaLLMOptions.options_dict_for_role(args, "extract")["think"] is True

    def test_an_unrecognized_think_value_is_refused(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_LLM_THINK", "hgih")

        with pytest.raises(argparse.ArgumentTypeError, match="hgih"):
            parse_args()

    def test_an_unrecognized_role_think_value_is_refused(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_LLM_THINK", raising=False)
        monkeypatch.setenv("EXTRACT_OLLAMA_LLM_THINK", "hgih")

        args = parse_args()

        with pytest.raises(argparse.ArgumentTypeError, match="hgih"):
            OllamaLLMOptions.options_dict_for_role(args, "extract")


# --------------------------------------------------------------------------- #
# Startup validation against the installed ollama package floor (>=0.5.4)
# --------------------------------------------------------------------------- #

_STARTUP_ENV_VARS_TO_ISOLATE = (
    "LLM_BINDING",
    "EMBEDDING_BINDING",
    "LLM_BINDING_HOST",
    "LLM_BINDING_API_KEY",
    "LLM_MODEL",
    "EMBEDDING_BINDING_HOST",
    "EMBEDDING_BINDING_API_KEY",
    "EMBEDDING_MODEL",
    "OLLAMA_LLM_THINK",
    "EXTRACT_OLLAMA_LLM_THINK",
    "KEYWORD_OLLAMA_LLM_THINK",
    "QUERY_OLLAMA_LLM_THINK",
    "VLM_OLLAMA_LLM_THINK",
    "EXTRACT_LLM_BINDING",
    "EXTRACT_LLM_MODEL",
    "EXTRACT_LLM_BINDING_API_KEY",
    "KEYWORD_LLM_BINDING",
    "QUERY_LLM_BINDING",
    "VLM_LLM_BINDING",
    "LIGHTRAG_KV_STORAGE",
    "LIGHTRAG_VECTOR_STORAGE",
    "LIGHTRAG_GRAPH_STORAGE",
    "LIGHTRAG_DOC_STATUS_STORAGE",
)


class TestOllamaThinkStartupValidation:
    @pytest.fixture(autouse=True)
    def _isolate_env(self, monkeypatch):
        for var in _STARTUP_ENV_VARS_TO_ISOLATE:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("LLM_BINDING", "ollama")
        monkeypatch.setenv("EMBEDDING_BINDING", "ollama")
        monkeypatch.setenv("EMBEDDING_MODEL", "bge-m3:latest")

        import lightrag.api.config as config

        saved_args, saved_initialized = config._global_args, config._initialized
        config._global_args = None
        config._initialized = False
        try:
            yield
        finally:
            config._global_args = saved_args
            config._initialized = saved_initialized

    @staticmethod
    def _create_app(monkeypatch, *, supports_think: bool):
        import lightrag.llm.ollama as ollama_binding
        from lightrag.api.config import initialize_config

        monkeypatch.setattr(ollama_binding, "_OLLAMA_SUPPORTS_THINK", supports_think)
        monkeypatch.setattr(sys, "argv", ["lightrag-server"])
        args = parse_args()
        initialize_config(args, force=True)

        with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
            mock_rag.return_value = MagicMock()
            from lightrag.api.lightrag_server import create_app

            return create_app(args)

    @pytest.mark.parametrize(
        ("env", "match"),
        [
            pytest.param({"OLLAMA_LLM_THINK": "false"}, "ollama>=0.5.4", id="global"),
            pytest.param(
                {"EXTRACT_OLLAMA_LLM_THINK": "false"},
                r"LLM role 'extract'",
                id="role-names-role",
            ),
            pytest.param(
                {
                    "LLM_BINDING": "openai",
                    "EXTRACT_LLM_BINDING": "ollama",
                    "EXTRACT_LLM_MODEL": "qwen3",
                    "EXTRACT_LLM_BINDING_API_KEY": "dummy",
                    "EXTRACT_OLLAMA_LLM_THINK": "false",
                },
                r"LLM role 'extract'",
                id="cross-provider-role",
            ),
            pytest.param({"OLLAMA_LLM_THINK": "high"}, "'high'", id="reasoning-level"),
        ],
    )
    def test_think_is_refused_at_startup_when_ollama_is_too_old(
        self, monkeypatch, env, match
    ):
        for k, v in env.items():
            monkeypatch.setenv(k, v)

        with pytest.raises(RuntimeError, match=match):
            self._create_app(monkeypatch, supports_think=False)

    def test_startup_succeeds_when_installed_ollama_supports_think(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_LLM_THINK", "false")
        monkeypatch.setenv("EXTRACT_OLLAMA_LLM_THINK", "false")

        assert self._create_app(monkeypatch, supports_think=True) is not None

    def test_startup_succeeds_on_old_ollama_when_think_is_unconfigured(
        self, monkeypatch
    ):
        assert self._create_app(monkeypatch, supports_think=False) is not None
