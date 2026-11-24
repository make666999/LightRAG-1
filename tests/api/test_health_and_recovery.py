"""Health endpoint auth disclosure, Docling status, scheduling view, and recovery fence projection."""

from __future__ import annotations

import importlib
import sys
import time
from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_server = importlib.import_module("lightrag.api.lightrag_server")
_utils_pipeline = importlib.import_module("lightrag.utils_pipeline")
sys.argv = _original_argv

from lightrag.base import DocStatusStorage  # noqa: E402
from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage  # noqa: E402
from lightrag.kg.shared_storage import (  # noqa: E402
    _INTERNAL_PIPELINE_STATUS_FIELDS,
    describe_recovery_fence,
)

pytestmark = pytest.mark.offline

# Fields that must NEVER appear in an unauthenticated response.
_SENSITIVE_TOP_LEVEL = ("working_directory", "input_directory", "configuration")
# Liveness fields that must always be present.
_LIVENESS_FIELDS = ("status", "auth_mode", "core_version", "api_version")

_ENV_VARS_TO_ISOLATE = (
    "LLM_BINDING",
    "EMBEDDING_BINDING",
    "AUTH_ACCOUNTS",
    "TOKEN_SECRET",
    "LIGHTRAG_API_KEY",
    "WHITELIST_PATHS",
    "LIGHTRAG_API_PREFIX",
)

_OCR_KEYS = {"do_ocr", "force_ocr", "ocr_engine", "ocr_lang", "do_formula_enrichment"}


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


def _build_client(monkeypatch, *, api_key=None, pipeline_status=None):
    """Build a /health-capable TestClient with all backend I/O mocked out."""
    from lightrag.api.config import initialize_config, parse_args

    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server"]
        args = parse_args()
    finally:
        sys.argv = original_argv
    if api_key is not None:
        args.key = api_key
    initialize_config(args, force=True)

    import lightrag.api.lightrag_server as lightrag_server

    monkeypatch.setattr(lightrag_server, "LightRAG", _FakeLightRAG)
    monkeypatch.setattr(lightrag_server, "check_frontend_build", lambda: (True, False))
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
        lightrag_server,
        "get_namespace_data",
        AsyncMock(
            return_value=(
                {"busy": False} if pipeline_status is None else pipeline_status
            )
        ),
    )
    monkeypatch.setattr(lightrag_server, "get_default_workspace", lambda: "default")
    monkeypatch.setattr(
        lightrag_server,
        "cleanup_keyed_lock",
        lambda: {"cleanup_performed": {}, "current_status": {}},
    )

    app = lightrag_server.create_app(args)
    return TestClient(app)


def _set_auth_mode(monkeypatch, *, auth_configured):
    """Override the module-level auth flags the /health gate reads at runtime."""
    import lightrag.api.utils_api as utils_api

    monkeypatch.setattr(utils_api, "auth_configured", auth_configured)
    monkeypatch.setattr(
        utils_api, "whitelist_patterns", [("/health", False), ("/api", True)]
    )


def _assert_liveness_only(body):
    for field in _LIVENESS_FIELDS:
        assert field in body, f"liveness field {field!r} missing"
    for field in _SENSITIVE_TOP_LEVEL:
        assert field not in body, f"sensitive field {field!r} leaked"


def _assert_full_config(body):
    assert "configuration" in body
    assert "working_directory" in body
    assert "llm_binding" in body["configuration"]


# --------------------------------------------------------------------------- #
# Credential-gated configuration disclosure on GET /health
# --------------------------------------------------------------------------- #


class TestHealthAuth:
    """Credential-gated configuration disclosure on ``GET /health``."""

    @pytest.fixture(autouse=True)
    def _isolate_env(self, monkeypatch):
        """Keep tests hermetic from developer-local .env and global config state."""
        for var in _ENV_VARS_TO_ISOLATE:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("AUTH_ACCOUNTS", "")
        monkeypatch.setenv("LIGHTRAG_API_KEY", "")
        monkeypatch.setenv("TOKEN_SECRET", "")
        monkeypatch.setenv("WHITELIST_PATHS", "/health,/api/*")
        monkeypatch.setenv("LLM_BINDING", "ollama")
        monkeypatch.setenv("EMBEDDING_BINDING", "ollama")
        monkeypatch.setenv("EMBEDDING_MODEL", "bge-m3:latest")

        import lightrag.api.config as config

        config._global_args = None
        config._initialized = False
        yield
        config._global_args = None
        config._initialized = False

    def test_open_mode_returns_full_config_to_anonymous(self, monkeypatch):
        client = _build_client(monkeypatch)
        _set_auth_mode(monkeypatch, auth_configured=False)

        resp = client.get("/health")

        assert resp.status_code == 200
        _assert_full_config(resp.json())

    def test_health_reads_pipeline_status_with_one_snapshot(self, monkeypatch):
        class SnapshotOnlyStatus(dict):
            copy_calls = 0

            def copy(self):
                self.copy_calls += 1
                return dict.copy(self)

            def get(self, *_args, **_kwargs):
                raise AssertionError("/health must not issue per-field proxy gets")

        status = SnapshotOnlyStatus(
            {
                "busy": False,
                "scanning": False,
                "destructive_busy": False,
                "pending_enqueues": 0,
            }
        )
        client = _build_client(monkeypatch, pipeline_status=status)
        _set_auth_mode(monkeypatch, auth_configured=False)

        response = client.get("/health")

        assert response.status_code == 200
        assert status.copy_calls == 1

    def test_password_mode_anonymous_gets_liveness_only(self, monkeypatch):
        client = _build_client(monkeypatch)
        _set_auth_mode(monkeypatch, auth_configured=True)

        resp = client.get("/health")

        assert resp.status_code == 200
        _assert_liveness_only(resp.json())

    def test_password_mode_valid_token_unlocks_config(self, monkeypatch):
        import lightrag.api.utils_api as utils_api

        client = _build_client(monkeypatch)
        _set_auth_mode(monkeypatch, auth_configured=True)
        monkeypatch.setattr(
            utils_api.auth_handler,
            "validate_token",
            lambda token: (
                {"username": "admin", "role": "user"}
                if token == "valid-user-token"
                else (_ for _ in ()).throw(ValueError("bad token"))
            ),
        )

        resp = client.get(
            "/health", headers={"Authorization": "Bearer valid-user-token"}
        )

        assert resp.status_code == 200
        _assert_full_config(resp.json())

    def test_password_mode_guest_token_stays_liveness_only(self, monkeypatch):
        import lightrag.api.utils_api as utils_api

        client = _build_client(monkeypatch)
        _set_auth_mode(monkeypatch, auth_configured=True)
        monkeypatch.setattr(
            utils_api.auth_handler,
            "validate_token",
            lambda token: {"username": "guest", "role": "guest"},
        )

        resp = client.get("/health", headers={"Authorization": "Bearer guest-token"})

        assert resp.status_code == 200
        _assert_liveness_only(resp.json())

    def test_api_key_mode_anonymous_gets_liveness_only(self, monkeypatch):
        client = _build_client(monkeypatch, api_key="secret-key")
        _set_auth_mode(monkeypatch, auth_configured=False)

        resp = client.get("/health")

        assert resp.status_code == 200
        _assert_liveness_only(resp.json())

    def test_api_key_mode_valid_key_unlocks_config(self, monkeypatch):
        client = _build_client(monkeypatch, api_key="secret-key")
        _set_auth_mode(monkeypatch, auth_configured=False)

        resp = client.get("/health", headers={"X-API-Key": "secret-key"})

        assert resp.status_code == 200
        _assert_full_config(resp.json())

    def test_combined_mode_anonymous_gets_liveness_only(self, monkeypatch):
        client = _build_client(monkeypatch, api_key="secret-key")
        _set_auth_mode(monkeypatch, auth_configured=True)

        resp = client.get("/health")

        assert resp.status_code == 200
        _assert_liveness_only(resp.json())

    def test_combined_mode_valid_api_key_unlocks_config(self, monkeypatch):
        client = _build_client(monkeypatch, api_key="secret-key")
        _set_auth_mode(monkeypatch, auth_configured=True)

        resp = client.get("/health", headers={"X-API-Key": "secret-key"})

        assert resp.status_code == 200
        _assert_full_config(resp.json())


# --------------------------------------------------------------------------- #
# Docling block of the /health snapshot
# --------------------------------------------------------------------------- #


class TestHealthDoclingStatus:
    """The ``docling`` block of the /health snapshot (``_build_docling_status``)."""

    @pytest.fixture(autouse=True)
    def _docling_env(self, monkeypatch):
        monkeypatch.setenv("DOCLING_ENDPOINT", "http://docling.test")
        for name in ("DOCLING_PIPELINE", "DOCLING_VLM_PRESET"):
            monkeypatch.delenv(name, raising=False)

    def test_no_endpoint_reports_empty_block(self, monkeypatch):
        monkeypatch.delenv("DOCLING_ENDPOINT")
        assert _server._build_docling_status() == {"endpoint": "", "options": {}}

    def test_standard_pipeline_reports_ocr_options(self):
        options = _server._build_docling_status()["options"]
        assert options["pipeline"] == "standard"
        assert _OCR_KEYS <= set(options)
        assert "vlm_pipeline_preset" not in options
        assert "pipeline_error" not in options

    def test_vlm_pipeline_reports_preset_and_hides_ocr_options(self, monkeypatch):
        monkeypatch.setenv("DOCLING_PIPELINE", "vlm")
        monkeypatch.setenv("DOCLING_VLM_PRESET", "minicpm_ocr")
        options = _server._build_docling_status()["options"]
        assert options == {"pipeline": "vlm", "vlm_pipeline_preset": "minicpm_ocr"}

    def test_vlm_pipeline_without_preset_reports_error(self, monkeypatch):
        monkeypatch.setenv("DOCLING_PIPELINE", "vlm")
        options = _server._build_docling_status()["options"]
        assert options["pipeline"] == "vlm"
        assert "DOCLING_VLM_PRESET" in options["pipeline_error"]
        assert not (_OCR_KEYS & set(options))

    def test_unknown_pipeline_reports_error_only(self, monkeypatch):
        monkeypatch.setenv("DOCLING_PIPELINE", "fast")
        status = _server._build_docling_status()
        assert status["endpoint"] == "http://docling.test"
        assert set(status["options"]) == {"pipeline_error"}
        assert "fast" in status["options"]["pipeline_error"]


# --------------------------------------------------------------------------- #
# Curated scheduling / capability view on /health
# --------------------------------------------------------------------------- #


def _idle_snapshot() -> dict:
    return {
        "busy": False,
        "pending_enqueues": 0,
        "manual_phase": "idle",
        "manual_freeze_requested": False,
        "manual_resetting": False,
        "manual_freeze_started_at": None,
        "manual_owner": None,
    }


def test_idle_pipeline_reports_zeros_not_nulls():
    view = _server._build_scheduling_status(
        _idle_snapshot(), {"manual_retries": 0, "manual_retries_capacity": 64}
    )

    assert view["manual_phase"] == "idle"
    assert view["manual_freeze_requested"] is False
    assert view["manual_freeze_seconds"] is None  # nothing is frozen
    assert view["drain_pending_enqueues"] == 0
    assert view["drain_waiting_on_workers"] is False
    assert view["manual_retries_queued"] == 0
    assert view["manual_retries_capacity"] == 64


def test_freeze_is_reported_with_its_holder_and_age():
    snapshot = _idle_snapshot()
    snapshot.update(
        {
            "manual_phase": "exclusive_reset",
            "manual_freeze_requested": True,
            "manual_resetting": True,
            "manual_freeze_started_at": time.time() - 9,
            "manual_owner": {"request_id": "req-42", "pid": 4242},
            "pending_enqueues": 2,
            "busy": True,
        }
    )

    view = _server._build_scheduling_status(snapshot, {})

    assert view["manual_phase"] == "exclusive_reset"
    assert view["manual_resetting"] is True
    assert 8.0 <= view["manual_freeze_seconds"] <= 15.0
    assert view["manual_owner_request_id"] == "req-42"
    assert view["manual_owner_pid"] == 4242
    assert view["drain_pending_enqueues"] == 2
    assert view["drain_waiting_on_workers"] is True


def test_owner_token_is_never_published():
    """The token authorizes releasing another process's reservation."""
    snapshot = _idle_snapshot()
    snapshot["manual_owner"] = {
        "request_id": "req-1",
        "pid": 7,
        "owner_token": "capability-do-not-leak",
        "process_start_id": "abc",
    }

    view = _server._build_scheduling_status(snapshot, {})

    assert "capability-do-not-leak" not in str(view)
    assert not any("token" in key for key in view)


def test_clock_stepping_backwards_never_reports_a_negative_age():
    snapshot = _idle_snapshot()
    snapshot["manual_freeze_started_at"] = time.time() + 60  # future timestamp

    view = _server._build_scheduling_status(snapshot, {})

    assert view["manual_freeze_seconds"] == 0.0


def test_a_first_party_backend_reports_every_capability():
    capabilities = _utils_pipeline.describe_doc_status_capabilities(
        JsonDocStatusStorage.__new__(JsonDocStatusStorage)
    )

    assert all(capabilities.values()), capabilities


def test_a_backend_on_the_fail_closed_defaults_reports_the_gaps():
    """A third-party doc_status that only implements the abstract methods keeps
    the base's fail-closed defaults; health has to say so."""

    class _MinimalDocStatus(DocStatusStorage):
        supports_strict_point_reads = False

        async def get_docs_by_statuses_page(self, *args, **kwargs): ...
        async def get_docs_by_ids(self, *args, **kwargs): ...
        async def resolve_doc_source_strict(self, *args, **kwargs): ...
        async def get_full_docs_by_ids(self, *args, **kwargs): ...

    _MinimalDocStatus.__abstractmethods__ = frozenset()

    capabilities = _utils_pipeline.describe_doc_status_capabilities(
        _MinimalDocStatus.__new__(_MinimalDocStatus)
    )

    assert capabilities["scheduling_pages"] is True
    assert capabilities["typed_source_resolution"] is True
    assert capabilities["strict_active_count"] is False
    assert capabilities["source_conflict_listing"] is False
    assert capabilities["source_conflict_repair"] is False
    assert capabilities["strict_point_reads"] is False


# --------------------------------------------------------------------------- #
# Recovery fence visibility projection
# --------------------------------------------------------------------------- #


def test_unfenced_pipeline_reports_false_not_null():
    """A dashboard should not have to special-case "no fence"."""
    view = describe_recovery_fence({"busy": False})

    assert view == {
        "recovery_required": False,
        "recovery_kind": None,
        "recovery_message": None,
    }


def test_unfenced_pipeline_with_an_empty_record_is_not_fenced():
    """A cleared fence is written as ``None``; an empty dict must read the same."""
    assert (
        describe_recovery_fence({"recovery_required": None})["recovery_required"]
        is False
    )
    assert (
        describe_recovery_fence({"recovery_required": {}})["recovery_required"] is False
    )


def test_stalled_drain_fence_surfaces_kind_and_blocker_sample():
    """The projection carries what an operator acts on: coarse cause and message."""
    view = describe_recovery_fence(
        {
            "recovery_required": {
                "kind": "manual_drain_stalled",
                "owner_key": "busy_owner",
                "operation_record": {"scope": "doc-a, doc-b"},
                "message": (
                    "manual retry drain stalled: 12 active document(s) have "
                    "blocked DRAIN_TO_IDLE for 3 consecutive rounds without "
                    "changing state (blocked doc id sample: doc-a, doc-b)."
                ),
            }
        }
    )

    assert view["recovery_required"] is True
    assert view["recovery_kind"] == "manual_drain_stalled"
    assert "blocked doc id sample: doc-a, doc-b" in view["recovery_message"]
    assert "force-reset" in view["recovery_message"]


def test_dead_owner_fence_keeps_its_derived_wording():
    """A fence with no explicit message still renders."""
    view = describe_recovery_fence(
        {
            "recovery_required": {
                "kind": "clear",
                "owner_key": "busy_owner",
                "operation_record": {"scope": "workspace"},
            }
        }
    )

    assert view["recovery_kind"] == "clear"
    assert "a worker died mid 'clear'" in view["recovery_message"]


def test_the_projection_leaks_no_credentials_or_process_identity():
    """Exactly three keys, and none of the record's credentials."""
    view = describe_recovery_fence(
        {
            "recovery_required": {
                "kind": "custom_chunks",
                "owner_key": "busy_owner",
                "operation_record": {"doc_id": "doc-x"},
                "message": "a worker died mid custom_chunks.",
                "owner_token": "tok-must-not-leak",
                "pid": 4242,
            }
        }
    )

    assert set(view) == {
        "recovery_required",
        "recovery_kind",
        "recovery_message",
    }
    rendered = repr(view)
    assert "tok-must-not-leak" not in rendered
    assert "4242" not in rendered
    assert "owner_key" not in rendered


def test_the_raw_record_is_still_internal():
    """Guards the reason this projection exists: the raw field stays stripped."""
    assert "recovery_required" in _INTERNAL_PIPELINE_STATUS_FIELDS
