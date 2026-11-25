"""Startup banner notices: deprecated storages, configuration backend, and Ollama open-route warning."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.offline

_DEPRECATED_MARKER = "Deprecated Storage"
_OLLAMA_WARNING_MARKER = "remain publicly accessible"
_DEFAULT = object()


def _render_storage_splash(
    monkeypatch,
    *,
    kv_storage: str,
    doc_status_storage: str,
    graph_storage: str = "NetworkXStorage",
) -> str:
    monkeypatch.setenv("LIGHTRAG_KV_STORAGE", kv_storage)
    monkeypatch.setenv("LIGHTRAG_DOC_STATUS_STORAGE", doc_status_storage)
    monkeypatch.setenv("LIGHTRAG_GRAPH_STORAGE", graph_storage)

    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server"]
        config = importlib.import_module("lightrag.api.config")
        utils_api = importlib.import_module("lightrag.api.utils_api")
        args = config.parse_args()
        colors = MagicMock()
        monkeypatch.setattr(utils_api, "ASCIIColors", colors)
        utils_api.display_splash_screen(args)
    finally:
        sys.argv = original_argv

    return "\n".join(str(arg) for call in colors.mock_calls for arg in call.args)


def _configuration_storage_line(output: str) -> str:
    lines = output.splitlines()
    index = next(i for i, line in enumerate(lines) if "Configuration Storage" in line)
    return lines[index + 1]


class TestSplashDeprecatedStorage:
    def test_redis_storages_are_flagged(self, monkeypatch):
        output = _render_storage_splash(
            monkeypatch,
            kv_storage="RedisKVStorage",
            doc_status_storage="RedisDocStatusStorage",
        )

        assert _DEPRECATED_MARKER in output
        assert "(deprecated)" in output
        assert "PGKVStorage" in output
        assert "PGDocStatusStorage" in output

    def test_age_graph_storage_is_flagged(self, monkeypatch):
        for key in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DATABASE"):
            monkeypatch.setenv(key, "x")
        output = _render_storage_splash(
            monkeypatch,
            kv_storage="JsonKVStorage",
            doc_status_storage="JsonDocStatusStorage",
            graph_storage="PGGraphStorage",
        )

        assert _DEPRECATED_MARKER in output
        assert "(deprecated)" in output
        assert "PGTableGraphStorage" in output
        assert "lightrag.tools.migrate_graph_storage" in output

    def test_default_storages_are_not_flagged(self, monkeypatch):
        output = _render_storage_splash(
            monkeypatch,
            kv_storage="JsonKVStorage",
            doc_status_storage="JsonDocStatusStorage",
        )

        assert _DEPRECATED_MARKER not in output
        assert "(deprecated)" not in output

    @pytest.mark.parametrize(
        ("config_storage", "kv_storage", "doc_status_storage", "expected"),
        [
            (
                "",
                "RedisKVStorage",
                "RedisDocStatusStorage",
                "JsonKVStorage (default for RedisKVStorage)",
            ),
            ("", "PGKVStorage", "PGDocStatusStorage", "(follows PGKVStorage)"),
            ("PGKVStorage", "RedisKVStorage", "RedisDocStatusStorage", "PGKVStorage"),
        ],
        ids=["redis-defaults-to-json", "pg-follows-kv", "explicit-override"],
    )
    def test_banner_names_resolved_configuration_backend(
        self, monkeypatch, config_storage, kv_storage, doc_status_storage, expected
    ):
        monkeypatch.setenv("LIGHTRAG_CONFIG_STORAGE", config_storage)
        output = _render_storage_splash(
            monkeypatch, kv_storage=kv_storage, doc_status_storage=doc_status_storage
        )

        assert _configuration_storage_line(output) == expected


# --------------------------------------------------------------------------- #
# Ollama routes open warning (GHSA-mmg5-8x8q-v934)
# --------------------------------------------------------------------------- #


def _render_ollama_warning_splash(
    monkeypatch,
    *,
    host: str,
    api_prefix: str | None,
    whitelist=_DEFAULT,
    api_key: str | None = "an-api-key",
    accounts: str | None = None,
) -> str:
    importlib.import_module("lightrag.api.config")
    for name, value in (
        ("WHITELIST_PATHS", "/health,/api/*" if whitelist is _DEFAULT else whitelist),
        ("LIGHTRAG_API_KEY", api_key),
        ("AUTH_ACCOUNTS", accounts),
        ("TOKEN_SECRET", "a-non-default-token-secret" if accounts else None),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    argv = ["lightrag-server", "--host", host]
    if api_prefix is not None:
        argv += ["--api-prefix", api_prefix]

    original_argv = sys.argv.copy()
    try:
        sys.argv = argv
        config = importlib.import_module("lightrag.api.config")
        utils_api = importlib.import_module("lightrag.api.utils_api")
        args = config.parse_args()
        patterns = [("/health", False)]
        if "/api/*" in args.whitelist_paths:
            patterns.append(("/api", True))
        monkeypatch.setattr(utils_api, "whitelist_patterns", patterns)
        colors = MagicMock()
        monkeypatch.setattr(utils_api, "ASCIIColors", colors)
        utils_api.display_splash_screen(args)
    finally:
        sys.argv = original_argv

    return "\n".join(str(arg) for call in colors.mock_calls for arg in call.args)


class TestSplashOllamaWarning:
    @pytest.mark.parametrize(
        ("kwargs", "expected_remedy"),
        [
            pytest.param(
                {"host": "0.0.0.0", "api_prefix": None},
                "remove /api from WHITELIST_PATHS",
                id="network-bind",
            ),
            pytest.param(
                {"host": "127.0.0.1", "api_prefix": "/site01"},
                "reverse-proxy prefix",
                id="loopback-with-reverse-proxy-prefix",
            ),
            pytest.param(
                {
                    "host": "0.0.0.0",
                    "api_prefix": None,
                    "whitelist": None,
                    "api_key": None,
                    "accounts": "reader:a-strong-secret",
                },
                "set\n    LIGHTRAG_API_KEY",
                id="accounts-only-without-api-key",
            ),
        ],
    )
    def test_warns_when_ollama_routes_are_exposed(
        self, monkeypatch, kwargs, expected_remedy
    ):
        output = _render_ollama_warning_splash(monkeypatch, **kwargs)
        assert _OLLAMA_WARNING_MARKER in output
        assert expected_remedy in output

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param(
                {"host": "127.0.0.1", "api_prefix": None}, id="loopback-no-prefix"
            ),
            pytest.param(
                {"host": "127.0.0.1", "api_prefix": ""}, id="loopback-empty-prefix"
            ),
            pytest.param(
                {"host": "127.0.0.1", "api_prefix": "/"}, id="loopback-slash-prefix"
            ),
            pytest.param(
                {"host": "127.0.0.1", "api_prefix": "   "},
                id="loopback-whitespace-prefix",
            ),
            pytest.param(
                {"host": "0.0.0.0", "api_prefix": None, "whitelist": None},
                id="key-with-default-whitelist",
            ),
        ],
    )
    def test_silent_when_ollama_routes_not_exposed(self, monkeypatch, kwargs):
        output = _render_ollama_warning_splash(monkeypatch, **kwargs)
        assert _OLLAMA_WARNING_MARKER not in output
