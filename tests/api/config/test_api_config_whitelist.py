"""WHITELIST_PATHS default policy and whitelist_exposes_api_routes matcher (GHSA-mmg5-8x8q-v934)."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest

from lightrag.api.config import default_whitelist_paths, parse_args

pytestmark = pytest.mark.offline


class TestWhitelistDefaultFollowsApiKey:
    @pytest.fixture(autouse=True)
    def _reset_env(self, monkeypatch):
        monkeypatch.delenv("WHITELIST_PATHS", raising=False)
        monkeypatch.delenv("LIGHTRAG_API_KEY", raising=False)
        monkeypatch.delenv("AUTH_ACCOUNTS", raising=False)
        monkeypatch.setattr(sys, "argv", ["lightrag-server"])
        monkeypatch.setenv("LLM_BINDING", "openai")

    def test_without_a_key_the_ollama_routes_stay_open(self):
        assert parse_args().whitelist_paths == "/health,/api/*"

    def test_a_key_from_the_environment_protects_the_ollama_routes(self, monkeypatch):
        monkeypatch.setenv("LIGHTRAG_API_KEY", "an-api-key")
        assert parse_args().whitelist_paths == "/health"

    def test_a_key_from_the_command_line_counts_too(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["lightrag-server", "--key", "an-api-key"])
        assert parse_args().whitelist_paths == "/health"

    def test_accounts_alone_leave_the_default_open(self, monkeypatch):
        monkeypatch.setenv("AUTH_ACCOUNTS", "reader:a-strong-secret")
        monkeypatch.setenv("TOKEN_SECRET", "a-non-default-token-secret")
        assert parse_args().whitelist_paths == "/health,/api/*"

    @pytest.mark.parametrize("explicit", ["/health,/api/*", "/health", ""])
    def test_an_explicit_value_always_wins(self, monkeypatch, explicit):
        monkeypatch.setenv("LIGHTRAG_API_KEY", "an-api-key")
        monkeypatch.setenv("WHITELIST_PATHS", explicit)
        assert parse_args().whitelist_paths == explicit

    def test_an_empty_key_counts_as_no_key(self):
        assert default_whitelist_paths("") == "/health,/api/*"


class TestWhitelistExposesApiRoutes:
    @pytest.fixture
    def whitelist_exposes_api_routes(self, monkeypatch):
        config = importlib.import_module("lightrag.api.config")
        monkeypatch.setattr(
            config,
            "global_args",
            SimpleNamespace(
                token_secret=None,
                jwt_algorithm="HS256",
                token_expire_hours=48,
                guest_token_expire_hours=24,
                auth_accounts="",
                whitelist_paths="/health",
                token_auto_renew=False,
            ),
        )
        sys.modules.pop("lightrag.api.auth", None)
        importlib.reload(importlib.import_module("lightrag.api.auth"))
        sys.modules.pop("lightrag.api.utils_api", None)
        utils_api = importlib.import_module("lightrag.api.utils_api")
        try:
            yield utils_api.whitelist_exposes_api_routes
        finally:
            sys.modules.pop("lightrag.api.utils_api", None)

    @pytest.mark.parametrize(
        "whitelist",
        [
            "/health,/api/*",
            "/api/*",
            "/api",
            "/api/chat",
            "/*",
            " /* ",
            "/health,/*",
            "/a/*",
        ],
    )
    def test_exposes_api_routes(
        self, whitelist_exposes_api_routes, whitelist: str
    ) -> None:
        assert whitelist_exposes_api_routes(whitelist) is True

    @pytest.mark.parametrize(
        "whitelist",
        [
            "/health",
            "/health,/docs",
            "/health/*",
            "/apiary/*",
            "/apidocs",
            "",
            "   ",
        ],
    )
    def test_does_not_expose_api_routes(
        self, whitelist_exposes_api_routes, whitelist: str
    ) -> None:
        assert whitelist_exposes_api_routes(whitelist) is False
