"""Document ingress validation (whitespace, texts-per-request limit, source characters) and supported file types."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
_parser_routing = importlib.import_module("lightrag.parser.routing")
_registry = importlib.import_module("lightrag.parser.registry")
sys.argv = _original_argv

DocumentManager = _document_routes.DocumentManager
InsertTextRequest = _document_routes.InsertTextRequest
InsertTextsRequest = _document_routes.InsertTextsRequest
create_document_routes = _document_routes.create_document_routes
find_unsafe_document_source_character = (
    _document_routes.find_unsafe_document_source_character
)
normalize_file_path = _document_routes.normalize_file_path
resolve_file_parser_engine = _parser_routing.resolve_file_parser_engine

pytestmark = pytest.mark.offline

_HEADERS = {"X-API-Key": "test-key"}


# --------------------------------------------------------------------------- #
# Insert text whitespace validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("text", ["  "])
def test_insert_text_rejects_whitespace_only(text):
    with pytest.raises(ValidationError):
        InsertTextRequest(text=text)


def test_insert_text_strips_surrounding_whitespace():
    req = InsertTextRequest(text="  hello  ")
    assert req.text == "hello"


@pytest.mark.parametrize("texts", [["ok", "  "]])
def test_insert_texts_rejects_whitespace_only_entries(texts):
    with pytest.raises(ValidationError):
        InsertTextsRequest(texts=texts)


def test_insert_texts_strips_surrounding_whitespace():
    req = InsertTextsRequest(texts=["  a  ", " b "])
    assert req.texts == ["a", "b"]


# --------------------------------------------------------------------------- #
# MAX_TEXTS_PER_REQUEST per-request fan-out ceiling (§11)
# --------------------------------------------------------------------------- #


class _TextsLimitRag:
    workspace = "texts-limit-test"
    addon_params: dict = {}
    doc_status = SimpleNamespace()


def _make_texts_limit_client(monkeypatch, limit):
    calls = {"lookups": 0, "reservations": 0, "indexed": 0}

    async def _lookup(_doc_status, _file_source):
        calls["lookups"] += 1
        return None

    async def _reserve(*_args, **_kwargs):
        calls["reservations"] += 1
        return True

    async def _release(*_args, **_kwargs):
        return None

    async def _index(*_args, **_kwargs):
        calls["indexed"] += 1

    monkeypatch.setattr(
        _document_routes, "get_existing_doc_by_file_path_candidates", _lookup
    )
    monkeypatch.setattr(_document_routes, "_reserve_enqueue_slot", _reserve)
    monkeypatch.setattr(_document_routes, "_release_enqueue_slot", _release)
    monkeypatch.setattr(_document_routes, "pipeline_index_texts", _index)
    monkeypatch.setattr(
        _document_routes.global_args, "max_texts_per_request", limit, raising=False
    )

    app = FastAPI()
    app.state.background_tasks = set()
    app.include_router(
        create_document_routes(
            _TextsLimitRag(), SimpleNamespace(), api_key="test-key"
        )
    )
    return TestClient(app), calls


def _post_texts(client, count):
    return client.post(
        "/documents/texts",
        headers=_HEADERS,
        json={
            "texts": [f"text {i}" for i in range(count)],
            "file_sources": [f"doc-{i}.md" for i in range(count)],
        },
    )


def test_an_oversized_batch_is_refused_with_413_before_any_storage_lookup(monkeypatch):
    client, calls = _make_texts_limit_client(monkeypatch, 2)

    resp = _post_texts(client, 3)

    assert resp.status_code == 413
    detail = resp.json()["detail"]
    assert "3" in detail and "2" in detail
    assert calls == {"lookups": 0, "reservations": 0, "indexed": 0}


def test_a_batch_exactly_at_the_limit_is_accepted(monkeypatch):
    client, calls = _make_texts_limit_client(monkeypatch, 2)

    resp = _post_texts(client, 2)

    assert resp.status_code == 200
    assert calls["lookups"] == 2


def test_zero_disables_the_check(monkeypatch):
    client, calls = _make_texts_limit_client(monkeypatch, 0)

    resp = _post_texts(client, 200)

    assert resp.status_code == 200
    assert calls["lookups"] == 200


@pytest.mark.parametrize("bad", ["50", True], ids=["str", "bool"])
def test_a_malformed_limit_disables_the_check_instead_of_refusing_everything(
    monkeypatch, bad
):
    client, _calls = _make_texts_limit_client(monkeypatch, bad)

    assert _post_texts(client, 5).status_code == 200


def test_the_limit_is_read_per_request_not_captured_at_import(monkeypatch):
    client, _calls = _make_texts_limit_client(monkeypatch, 1)
    assert _post_texts(client, 4).status_code == 413

    monkeypatch.setattr(
        _document_routes.global_args, "max_texts_per_request", 10, raising=False
    )
    assert _post_texts(client, 4).status_code == 200


# --------------------------------------------------------------------------- #
# Document source character rules (GHSA-c922-pw4m-4wcv)
# --------------------------------------------------------------------------- #

UNSAFE = [
    pytest.param("a\x00b.txt", "U+0000", id="nul"),
    pytest.param("a\x0bb.txt", "U+000B", id="vertical-tab"),
    pytest.param("a\x1fb.txt", "U+001F", id="unit-separator"),
    pytest.param("a\tb.txt", "U+0009", id="tab"),
    pytest.param("a\nb.txt", "U+000A", id="newline"),
    pytest.param("a\x7fb.txt", "U+007F", id="del"),
    pytest.param("a\ud800b.txt", "U+D800", id="lone-surrogate"),
    pytest.param("a\ufffeb.txt", "U+FFFE", id="non-character"),
]

SAFE = [
    "report.pdf",
    "报告 2026.docx",
    "a-b_c.d.txt",
    "emoji \U0001f600.md",
    "spaces are fine.txt",
]

UNSAFE_WIRING = [p for p in UNSAFE if p.id in ("nul", "lone-surrogate")]
SAFE_WIRING = ["报告 2026.docx"]


class TestCharacterRule:
    @pytest.mark.parametrize("value, code", UNSAFE)
    def test_unsafe_characters_are_found(self, value, code):
        offender = find_unsafe_document_source_character(value)
        assert offender is not None
        assert f"U+{ord(offender):04X}" == code

    @pytest.mark.parametrize("value", SAFE)
    def test_ordinary_sources_are_accepted(self, value):
        assert find_unsafe_document_source_character(value) is None


class TestTextEndpointRequests:
    @pytest.mark.parametrize("value, code", UNSAFE_WIRING)
    def test_text_request_rejects_unsafe_file_source(self, value, code):
        with pytest.raises(ValidationError) as excinfo:
            InsertTextRequest(text="hello", file_source=value)

        assert code in str(excinfo.value)

    def test_texts_request_rejects_an_offender_in_any_position(self):
        with pytest.raises(ValidationError, match=r"U\+000B"):
            InsertTextsRequest(
                texts=["a", "b", "c"],
                file_sources=["ok.txt", "bad\x0b.txt", "also-ok.txt"],
            )

    @pytest.mark.parametrize("value", SAFE_WIRING)
    def test_ordinary_file_source_still_accepted(self, value):
        request = InsertTextRequest(text="hello", file_source=value)
        assert request.file_source == normalize_file_path(value)

    def test_absent_file_source_still_accepted(self):
        request = InsertTextRequest(text="hello")
        assert request.file_source is None

    def test_explicit_null_file_source_still_accepted(self):
        request = InsertTextRequest(text="hello", file_source=None)
        assert request.file_source == _document_routes.UNKNOWN_FILE_SOURCE

    def test_absent_file_sources_still_accepted(self):
        request = InsertTextsRequest(texts=["hello"])
        assert request.file_sources is None


class TestUploadPath:
    @pytest.mark.parametrize("value, _code", UNSAFE_WIRING)
    def test_unsafe_filename_is_rejected(self, tmp_path, value, _code):
        with pytest.raises(_document_routes.HTTPException) as excinfo:
            _document_routes.sanitize_filename(value, tmp_path)

        assert excinfo.value.status_code == 400

    @pytest.mark.parametrize("value", SAFE_WIRING)
    def test_ordinary_filename_is_accepted(self, tmp_path, value):
        assert _document_routes.sanitize_filename(value, tmp_path) == value


class TestReadPathsStayNonRaising:
    @pytest.mark.parametrize("value, _code", UNSAFE_WIRING)
    def test_normalize_file_path_never_raises(self, value, _code):
        assert isinstance(normalize_file_path(value), str)


# --------------------------------------------------------------------------- #
# Supported extensions (.md / .textpack) and GET /documents/supported_file_types
# --------------------------------------------------------------------------- #


@pytest.fixture
def doc_manager(tmp_path, monkeypatch):
    monkeypatch.delenv("LIGHTRAG_PARSER", raising=False)
    return DocumentManager(str(tmp_path))


def test_textpack_and_md_are_uploadable(doc_manager):
    extensions = doc_manager.supported_extensions
    assert ".textpack" in extensions
    assert ".md" in extensions


def test_bare_textpack_is_supported_file_via_native(doc_manager, monkeypatch):
    monkeypatch.delenv("LIGHTRAG_PARSER", raising=False)
    assert doc_manager.is_supported_file("note.textpack") is True
    assert resolve_file_parser_engine("note.textpack") == "native"


def test_md_is_supported_file(doc_manager, monkeypatch):
    monkeypatch.delenv("LIGHTRAG_PARSER", raising=False)
    assert doc_manager.is_supported_file("doc.md") is True


@pytest.fixture
def supported_types_client(tmp_path, monkeypatch):
    for var in (
        "LIGHTRAG_PARSER",
        "MINERU_API_MODE",
        "MINERU_LOCAL_ENDPOINT",
        "MINERU_API_TOKEN",
        "DOCLING_ENDPOINT",
    ):
        monkeypatch.delenv(var, raising=False)
    app = FastAPI()
    app.include_router(
        create_document_routes(
            SimpleNamespace(),
            DocumentManager(str(tmp_path)),
            api_key="test-key",
        )
    )
    return TestClient(app)


def test_default_deployment_contract(supported_types_client):
    resp = supported_types_client.get("/documents/supported_file_types", headers=_HEADERS)
    assert resp.status_code == 200

    body = resp.json()
    exts = body["supported_extensions"]
    assert isinstance(exts, list) and exts
    assert all(e.startswith(".") and e == e.lower() for e in exts)
    assert ".md" in exts and ".txt" in exts

    engines = body["engines"]
    assert set(engines) >= {"native", "legacy"}
    assert "mineru" not in engines and "docling" not in engines
    assert "reuse" not in engines and "passthrough" not in engines
    assert engines["native"] == [".docx", ".md", ".textpack"]
    for suffixes in engines.values():
        assert all(s.startswith(".") and s == s.lower() for s in suffixes)


def test_requires_auth(supported_types_client):
    resp = supported_types_client.get("/documents/supported_file_types")
    assert resp.status_code in (401, 403)


def test_cache_headers(supported_types_client):
    resp = supported_types_client.get("/documents/supported_file_types", headers=_HEADERS)
    assert resp.headers["Cache-Control"] == "no-store"
    assert resp.headers["Vary"] == "LIGHTRAG-WORKSPACE"


def test_mineru_routing_splits_allowlist_and_matrix(supported_types_client, monkeypatch):
    monkeypatch.setenv("MINERU_API_MODE", "local")
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://localhost:8888")
    monkeypatch.setenv("LIGHTRAG_PARSER", "jpg:mineru;*:legacy-R")

    body = supported_types_client.get(
        "/documents/supported_file_types", headers=_HEADERS
    ).json()

    assert ".jpg" in body["supported_extensions"]
    assert ".png" not in body["supported_extensions"]
    assert ".png" in body["engines"]["mineru"]
    assert ".jpg" in body["engines"]["mineru"]


def test_matrix_tracks_dynamic_registry(supported_types_client):
    spec = _registry.ParserSpec(
        engine_name="fooengine",
        impl="x:Y",
        suffixes=frozenset({"foo"}),
        queue_group="fooengine",
    )
    try:
        _registry.register_parser(spec)
        body = supported_types_client.get(
            "/documents/supported_file_types", headers=_HEADERS
        ).json()
        assert body["engines"]["fooengine"] == [".foo"]
    finally:
        _registry._REGISTRY.pop("fooengine", None)
