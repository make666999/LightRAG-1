"""WebUI entry mounts, title injection, AI content notice, and API docs toggle tests."""

from __future__ import annotations

import json
import re
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

pytestmark = pytest.mark.offline

PLACEHOLDER = "<!-- __LIGHTRAG_RUNTIME_CONFIG__ -->"
STATIC_TITLE = "<title>LightRAG</title>"

_DOC_SURFACES = (
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
    "/openapi.json",
    "/static/swagger-ui/swagger-ui.css",
)

_ENV_VARS_TO_ISOLATE = (
    "LLM_BINDING",
    "EMBEDDING_BINDING",
    "OPENAI_API_KEY",
    "AUTH_ACCOUNTS",
    "TOKEN_SECRET",
    "LIGHTRAG_API_KEY",
    "WHITELIST_PATHS",
    "LIGHTRAG_API_PREFIX",
    "LIGHTRAG_DEFAULT_UI",
    "ENABLE_API_DOCS",
    "ENABLE_AI_CONTENT_NOTICE",
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Keep tests hermetic from developer-local .env and global config state."""
    import lightrag.api.config as config

    for var in _ENV_VARS_TO_ISOLATE:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AUTH_ACCOUNTS", "")
    monkeypatch.setenv("LIGHTRAG_API_KEY", "")
    monkeypatch.setenv("TOKEN_SECRET", "")
    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("EMBEDDING_BINDING", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    config._global_args = None
    config._initialized = False
    yield
    config._global_args = None
    config._initialized = False


class _FakeLightRAG:
    """Minimal stand-in implementing the async surface /health touches."""

    def __init__(self, *_args, **_kwargs):
        pass

    def register_role_llm_builder(self, _builder):
        return None

    def set_role_llm_metadata(self, _role, **_metadata):
        return None

    def get_llm_role_config(self):
        return {}

    async def get_llm_queue_status(self, include_base=True):
        return {}

    async def get_embedding_queue_status(self):
        return {}

    async def get_rerank_queue_status(self):
        return {}


class _FakeOllamaAPI:
    def __init__(self, *_args, **_kwargs):
        self.router = APIRouter()


def _entry_html(name: str) -> str:
    """Mirror what Vite emits: static <title> followed by the placeholder."""
    return (
        "<!doctype html><html><head>"
        f"{STATIC_TITLE}"
        f"{PLACEHOLDER}"
        f'<script type="module" crossorigin src="./assets/{name}-X.js"></script>'
        f"</head><body><div id=root>{name}</div></body></html>"
    )


def _stage_build(tmp_path, *, webui=True, workspace=True):
    webui_dir = tmp_path / "webui"
    webui_dir.mkdir(exist_ok=True)
    if webui:
        (webui_dir / "index.html").write_text(_entry_html("index"), encoding="utf-8")
    if workspace:
        (webui_dir / "workspace.html").write_text(
            _entry_html("workspace"), encoding="utf-8"
        )
    assets = webui_dir / "assets"
    assets.mkdir(exist_ok=True)
    (assets / "shared-X.js").write_text("//js", encoding="utf-8")


def _patch_server_backends(monkeypatch, lightrag_server):
    monkeypatch.setattr(lightrag_server, "LightRAG", _FakeLightRAG)
    monkeypatch.setattr(
        lightrag_server, "create_document_routes", lambda *_a, **_k: APIRouter()
    )
    monkeypatch.setattr(
        lightrag_server, "create_query_routes", lambda *_a, **_k: APIRouter()
    )
    monkeypatch.setattr(
        lightrag_server, "create_graph_routes", lambda *_a, **_k: APIRouter()
    )
    monkeypatch.setattr(lightrag_server, "OllamaAPI", _FakeOllamaAPI)
    monkeypatch.setattr(
        lightrag_server, "get_namespace_data", AsyncMock(return_value={"busy": False})
    )
    monkeypatch.setattr(lightrag_server, "get_default_workspace", lambda: "default")
    monkeypatch.setattr(
        lightrag_server,
        "cleanup_keyed_lock",
        lambda: {"cleanup_performed": {}, "current_status": {}},
    )


def _parse_and_init(*cli_args):
    from lightrag.api.config import initialize_config, parse_args

    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server", *cli_args]
        args = parse_args()
    finally:
        sys.argv = original_argv
    initialize_config(args, force=True)
    return args


def _build_app(tmp_path, monkeypatch, *cli_args):
    args = _parse_and_init(*cli_args)

    import lightrag.api.lightrag_server as lightrag_server

    monkeypatch.setattr(
        lightrag_server, "__file__", str(tmp_path / "lightrag_server.py")
    )
    _patch_server_backends(monkeypatch, lightrag_server)
    return lightrag_server.create_app(args)


def _client(tmp_path, monkeypatch, *cli_args, **stage_kwargs):
    _stage_build(tmp_path, **stage_kwargs)
    app = _build_app(tmp_path, monkeypatch, *cli_args)
    return TestClient(app)


def _build_toggle_client(
    monkeypatch,
    *,
    notice=None,
    enable_api_docs=None,
    webui_available=True,
    argv=(),
):
    if notice is not None:
        monkeypatch.setenv("ENABLE_AI_CONTENT_NOTICE", notice)
    if enable_api_docs is not None:
        monkeypatch.setenv("ENABLE_API_DOCS", enable_api_docs)

    args = _parse_and_init(*argv)

    import lightrag.api.lightrag_server as lightrag_server

    monkeypatch.setattr(lightrag_server.auth_handler, "accounts", {})
    _patch_server_backends(monkeypatch, lightrag_server)
    monkeypatch.setattr(
        lightrag_server, "check_frontend_build", lambda: (webui_available, False)
    )
    return TestClient(lightrag_server.create_app(args))


# --------------------------------------------------------------------------- #
# ENABLE_AI_CONTENT_NOTICE switch
# --------------------------------------------------------------------------- #


class TestAiContentNotice:
    """Tests for the ENABLE_AI_CONTENT_NOTICE switch."""

    def test_disabled_by_default(self, monkeypatch):
        client = _build_toggle_client(monkeypatch)

        assert client.get("/auth-status").json()["ai_content_notice_enabled"] is False
        assert client.get("/health").json()["ai_content_notice_enabled"] is False
        assert (
            client.post("/login", data={"username": "u", "password": "p"}).json()[
                "ai_content_notice_enabled"
            ]
            is False
        )

    def test_config_default_is_false(self):
        from lightrag.api.config import parse_args

        original_argv = sys.argv.copy()
        try:
            sys.argv = ["lightrag-server"]
            args = parse_args()
        finally:
            sys.argv = original_argv

        assert args.enable_ai_content_notice is False

    def test_auth_status_reports_the_notice_when_enabled(self, monkeypatch):
        client = _build_toggle_client(monkeypatch, notice="true")
        assert client.get("/auth-status").json()["ai_content_notice_enabled"] is True

    def test_login_reports_the_notice_when_enabled(self, monkeypatch):
        client = _build_toggle_client(monkeypatch, notice="true")
        body = client.post("/login", data={"username": "u", "password": "p"}).json()
        assert body["ai_content_notice_enabled"] is True

    def test_health_reports_the_notice_when_enabled(self, monkeypatch):
        client = _build_toggle_client(monkeypatch, notice="true")
        assert client.get("/health").json()["ai_content_notice_enabled"] is True

    def test_health_reports_the_notice_without_authentication(self, monkeypatch):
        import lightrag.api.utils_api as utils_api

        client = _build_toggle_client(monkeypatch, notice="true")
        monkeypatch.setattr(utils_api, "auth_configured", True)
        monkeypatch.setattr(
            utils_api, "whitelist_patterns", [("/health", False), ("/api", True)]
        )

        body = client.get("/health").json()
        assert body["ai_content_notice_enabled"] is True
        assert "configuration" not in body


# --------------------------------------------------------------------------- #
# ENABLE_API_DOCS toggle
# --------------------------------------------------------------------------- #


def _assert_service_info(resp, expected_health_url):
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["service"] == "LightRAG Server"
    assert body["api_version"]
    assert body["health_url"] == expected_health_url
    assert "message" in body


class TestApiDocsToggle:
    """Tests for the ENABLE_API_DOCS switch."""

    def test_docs_surfaces_enabled_by_default(self, monkeypatch):
        client = _build_toggle_client(monkeypatch)

        for path in _DOC_SURFACES:
            resp = client.get(path)
            assert resp.status_code == 200, f"{path} should be served by default"

    def test_docs_surfaces_all_return_404_when_disabled(self, monkeypatch):
        client = _build_toggle_client(monkeypatch, enable_api_docs="false")

        for path in _DOC_SURFACES:
            resp = client.get(path)
            assert resp.status_code == 404, f"{path} must 404 with ENABLE_API_DOCS=false"

    def test_disabling_docs_leaves_other_routes_alive(self, monkeypatch):
        client = _build_toggle_client(monkeypatch, enable_api_docs="false")

        assert client.get("/health").status_code == 200
        assert client.get("/auth-status").status_code == 200

    def test_health_reports_api_docs_available_true(self, monkeypatch):
        client = _build_toggle_client(monkeypatch)
        assert client.get("/health").json()["api_docs_available"] is True

    def test_health_reports_api_docs_available_false(self, monkeypatch):
        client = _build_toggle_client(monkeypatch, enable_api_docs="false")
        assert client.get("/health").json()["api_docs_available"] is False

    def test_health_capability_field_visible_without_authentication(self, monkeypatch):
        import lightrag.api.utils_api as utils_api

        client = _build_toggle_client(monkeypatch, enable_api_docs="false")
        monkeypatch.setattr(utils_api, "auth_configured", True)
        monkeypatch.setattr(
            utils_api, "whitelist_patterns", [("/health", False), ("/api", True)]
        )

        resp = client.get("/health")
        body = resp.json()
        assert resp.status_code == 200
        assert body["api_docs_available"] is False
        assert "configuration" not in body

    def test_missing_webui_still_redirects_to_docs_when_enabled(self, monkeypatch):
        client = _build_toggle_client(monkeypatch, webui_available=False)

        for path in ("/", "/webui", "/webui/"):
            resp = client.get(path, follow_redirects=False)
            assert resp.status_code in (302, 307), path
            assert resp.headers["location"] == "/docs", path

    def test_missing_webui_and_disabled_docs_return_service_info(self, monkeypatch):
        client = _build_toggle_client(
            monkeypatch, enable_api_docs="false", webui_available=False
        )

        for path in ("/", "/webui", "/webui/"):
            _assert_service_info(client.get(path), "/health")

    def test_service_info_health_url_keeps_api_prefix(self, monkeypatch):
        client = _build_toggle_client(
            monkeypatch,
            enable_api_docs="false",
            webui_available=False,
            argv=("--api-prefix", "/site01"),
        )

        for path in (
            "/",
            "/webui",
            "/webui/",
            "/site01/",
            "/site01/webui",
            "/site01/webui/",
        ):
            _assert_service_info(client.get(path), "/site01/health")

    def test_webui_available_ignores_docs_switch_for_root_redirect(self, monkeypatch):
        client = _build_toggle_client(monkeypatch, enable_api_docs="false")

        resp = client.get("/", follow_redirects=False)
        assert resp.status_code in (302, 307)
        assert resp.headers["location"] == "/webui/"


# --------------------------------------------------------------------------- #
# WEBUI_TITLE runtime config injection
# --------------------------------------------------------------------------- #


def _title_client(tmp_path, monkeypatch, title):
    _stage_build(tmp_path)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])

    from lightrag.api import lightrag_server
    from lightrag.api.config import parse_args

    monkeypatch.setattr(
        lightrag_server, "__file__", str(tmp_path / "lightrag_server.py")
    )
    monkeypatch.setattr(lightrag_server, "webui_title", title)

    args = parse_args()
    with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
        mock_rag.return_value = MagicMock()
        app = lightrag_server.create_app(args)
    return TestClient(app)


def _config_payload(body: str) -> dict:
    match = re.search(r"window\.__LIGHTRAG_CONFIG__ = (\{.*?\});", body)
    assert match is not None, body
    return json.loads(match.group(1))


class TestWebuiTitleInjection:
    """WEBUI_TITLE reaches the browser tab title through the entry HTML."""

    def test_custom_title_is_published_and_applied(self, tmp_path, monkeypatch):
        client = _title_client(tmp_path, monkeypatch, "My Graph KB")

        body = client.get("/webui/").text
        assert _config_payload(body)["webuiTitle"] == "My Graph KB"
        assert "document.title=window.__LIGHTRAG_CONFIG__.webuiTitle" in body
        assert STATIC_TITLE in body

    def test_both_entries_get_the_title(self, tmp_path, monkeypatch):
        client = _title_client(tmp_path, monkeypatch, "My Graph KB")

        for path in ("/webui/", "/workspace/"):
            body = client.get(path).text
            assert _config_payload(body)["webuiTitle"] == "My Graph KB", path

    def test_unset_title_leaves_the_static_default(self, tmp_path, monkeypatch):
        client = _title_client(tmp_path, monkeypatch, None)

        body = client.get("/webui/").text
        assert _config_payload(body)["webuiTitle"] is None
        assert STATIC_TITLE in body

    def test_empty_title_is_normalized_to_null(self, tmp_path, monkeypatch):
        client = _title_client(tmp_path, monkeypatch, "")
        assert _config_payload(client.get("/webui/").text)["webuiTitle"] is None

    def test_title_cannot_break_out_of_the_script(self, tmp_path, monkeypatch):
        client = _title_client(
            tmp_path, monkeypatch, "</script><script>alert(1)</script>"
        )

        body = client.get("/webui/").text
        assert "</script><script>alert(1)" not in body
        assert "<\\/script>" in body
        payload = _config_payload(body.replace("<\\/", "</"))
        assert payload["webuiTitle"] == "</script><script>alert(1)</script>"

    def test_quotes_in_title_are_json_escaped(self, tmp_path, monkeypatch):
        client = _title_client(tmp_path, monkeypatch, 'My "Graph" KB')
        payload = _config_payload(client.get("/webui/").text)
        assert payload["webuiTitle"] == 'My "Graph" KB'


# --------------------------------------------------------------------------- #
# Dual-entry WebUI mounts (/webui + /workspace) and LIGHTRAG_DEFAULT_UI
# --------------------------------------------------------------------------- #


class TestDualMount:
    def test_each_mount_serves_its_own_entry(self, tmp_path, monkeypatch):
        client = _client(tmp_path, monkeypatch)

        webui = client.get("/webui/")
        assert webui.status_code == 200
        assert ">index<" in webui.text

        workspace = client.get("/workspace/")
        assert workspace.status_code == 200
        assert ">workspace<" in workspace.text

    def test_trailing_slash_redirects_preserved(self, tmp_path, monkeypatch):
        client = _client(tmp_path, monkeypatch)
        for path, target in (("/webui", "/webui/"), ("/workspace", "/workspace/")):
            resp = client.get(path, follow_redirects=False)
            assert resp.status_code in (301, 307, 308), path
            assert resp.headers["location"].endswith(target)

    def test_cross_entry_html_rejected(self, tmp_path, monkeypatch):
        client = _client(tmp_path, monkeypatch)
        assert client.get("/workspace/index.html").status_code == 404
        assert client.get("/webui/workspace.html").status_code == 404

    def test_cross_entry_html_rejected_case_insensitively(self, tmp_path, monkeypatch):
        client = _client(tmp_path, monkeypatch)
        for path in (
            "/webui/WORKSPACE.HTML",
            "/webui/Workspace.html",
            "/workspace/INDEX.HTML",
            "/workspace/Index.Html",
        ):
            assert client.get(path).status_code == 404, path

    def test_same_entry_explicit_filename_still_served(self, tmp_path, monkeypatch):
        client = _client(tmp_path, monkeypatch)

        webui = client.get("/webui/index.html")
        assert webui.status_code == 200
        assert ">index<" in webui.text
        assert "window.__LIGHTRAG_CONFIG__" in webui.text

        workspace = client.get("/workspace/workspace.html")
        assert workspace.status_code == 200
        assert ">workspace<" in workspace.text
        assert "window.__LIGHTRAG_CONFIG__" in workspace.text

    def test_runtime_config_identical_on_both_mounts(self, tmp_path, monkeypatch):
        client = _client(tmp_path, monkeypatch, "--api-prefix", "/site01")

        def config_script(text):
            match = re.search(
                r"<script>window\.__LIGHTRAG_CONFIG__ = .*?</script>", text
            )
            return match.group(0) if match else None

        webui = client.get("/site01/webui/").text
        workspace = client.get("/site01/workspace/").text
        assert config_script(webui) is not None
        assert config_script(webui) == config_script(workspace)
        assert '"apiPrefix": "/site01"' in webui.replace('":"', '": "')
        assert "uiMode" not in webui and "entryMode" not in workspace

    def test_html_no_cache_and_assets_immutable_on_workspace(
        self, tmp_path, monkeypatch
    ):
        client = _client(tmp_path, monkeypatch)
        html = client.get("/workspace/")
        assert "no-store" in html.headers.get("cache-control", "")
        asset = client.get("/workspace/assets/shared-X.js")
        assert asset.status_code == 200
        assert "immutable" in asset.headers.get("cache-control", "")


class TestWorkspaceDegradation:
    """Old build directory: only index.html exists."""

    @pytest.mark.parametrize("docs_enabled", ["true", "false"])
    def test_workspace_missing_returns_fixed_json(
        self, tmp_path, monkeypatch, docs_enabled
    ):
        monkeypatch.setenv("ENABLE_API_DOCS", docs_enabled)
        client = _client(tmp_path, monkeypatch, workspace=False)

        for path in ("/workspace", "/workspace/"):
            resp = client.get(path, follow_redirects=False)
            assert resp.status_code == 200, path
            data = resp.json()
            assert data["service"] == "LightRAG Server"
            assert data["health_url"] == "/health"
            body = resp.text.lower()
            assert "docs" not in body, path
            assert "redoc" not in body, path

        assert client.get("/webui/").status_code == 200

    def test_health_reports_entries_independently(self, tmp_path, monkeypatch):
        client = _client(tmp_path, monkeypatch, workspace=False)
        health = client.get("/health").json()
        assert health["webui_available"] is True
        assert health["workspace_available"] is False

    def test_health_reports_both_available(self, tmp_path, monkeypatch):
        client = _client(tmp_path, monkeypatch)
        health = client.get("/health").json()
        assert health["webui_available"] is True
        assert health["workspace_available"] is True


class TestDefaultUi:
    def test_default_is_webui(self, tmp_path, monkeypatch):
        client = _client(tmp_path, monkeypatch)
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code in (302, 307)
        assert resp.headers["location"] == "/webui/"

    def test_workspace_via_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LIGHTRAG_DEFAULT_UI", "workspace")
        client = _client(tmp_path, monkeypatch)
        resp = client.get("/", follow_redirects=False)
        assert resp.headers["location"] == "/workspace/"

    def test_workspace_via_cli(self, tmp_path, monkeypatch):
        client = _client(tmp_path, monkeypatch, "--default-ui", "workspace")
        resp = client.get("/", follow_redirects=False)
        assert resp.headers["location"] == "/workspace/"

    def test_root_path_preserved(self, tmp_path, monkeypatch):
        client = _client(
            tmp_path,
            monkeypatch,
            "--default-ui",
            "workspace",
            "--api-prefix",
            "/site01",
        )
        resp = client.get("/site01/", follow_redirects=False)
        assert resp.headers["location"] == "/site01/workspace/"

    def test_illegal_env_value_fails_startup(self, monkeypatch):
        monkeypatch.setenv("LIGHTRAG_DEFAULT_UI", "workspaces")
        monkeypatch.setattr(sys, "argv", ["lightrag-server"])
        from lightrag.api.config import parse_args

        with pytest.raises(SystemExit):
            parse_args()

    def test_illegal_cli_value_fails_startup(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["lightrag-server", "--default-ui", "workspaces"]
        )
        from lightrag.api.config import parse_args

        with pytest.raises(SystemExit):
            parse_args()

    @pytest.mark.parametrize("docs_enabled", ["true", "false"])
    def test_root_follows_workspace_degradation(
        self, tmp_path, monkeypatch, docs_enabled
    ):
        monkeypatch.setenv("ENABLE_API_DOCS", docs_enabled)
        monkeypatch.setenv("LIGHTRAG_DEFAULT_UI", "workspace")
        client = _client(tmp_path, monkeypatch, workspace=False)

        resp = client.get("/", follow_redirects=False)
        assert resp.status_code == 200
        body = resp.text.lower()
        assert "docs" not in body
        assert "webui" not in body
        assert client.get("/webui/").status_code == 200
