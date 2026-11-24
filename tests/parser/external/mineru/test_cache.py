"""``*.mineru_raw/`` cache validation and artifact-deletion tests.

Covers every failure mode that triggers a re-download:
- missing / malformed / non-UTF-8 / wrong-engine manifest
- source file size mismatch (fast-path) and same-size content_hash mismatch
- parser options signature missing / mismatch (local & official partitions)
- engine version / api_mode / endpoint env mismatch
- critical_file (content_list.json) missing / size / sha256 mismatch
- auxiliary file missing / size mismatch
- ``delete_file_variants_by_file_path`` cleanup of ``*.mineru_raw/`` bundles
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from lightrag.constants import PARSED_DIR_NAME
from lightrag.parser.external.mineru import (
    Manifest,
    ManifestFile,
    clear_dir_contents,
    compute_size_and_hash,
    is_bundle_valid,
    raw_dir_for_parsed_dir,
)
from lightrag.parser.external.mineru.cache import current_mineru_options_signature
from lightrag.parser.external.mineru.manifest import write_manifest

# document_routes pulls in api.auth -> api.config.parse_args(); neutralize
# pytest's argv during that first import and restore it immediately.
_original_argv = sys.argv[:]
sys.argv = sys.argv[:1]
from lightrag.api.routers.document_routes import (  # noqa: E402
    _file_path_for_parsed_artifact_dir,
    delete_file_variants_by_file_path,
)

sys.argv = _original_argv

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def source_file(tmp_path: Path) -> Path:
    p = tmp_path / "src.pdf"
    p.write_bytes(b"Hello PDF" * 100)
    return p


@pytest.fixture
def fresh_bundle(tmp_path: Path, source_file: Path) -> tuple[Path, Manifest]:
    """Build a fully-valid bundle alongside ``source_file`` and return
    ``(raw_dir, manifest)``."""
    raw = tmp_path / "src.mineru_raw"
    raw.mkdir()
    content_list = raw / "content_list.json"
    content_list.write_text('[{"type":"text","text":"hi"}]', encoding="utf-8")
    images = raw / "images"
    images.mkdir()
    (images / "img1.png").write_bytes(b"PNG" * 50)
    (images / "img2.png").write_bytes(b"PNG" * 60)

    src_size, src_hash = compute_size_and_hash(source_file)
    crit_size, crit_hash = compute_size_and_hash(content_list)
    files = [
        ManifestFile(path="images/img1.png", size=(images / "img1.png").stat().st_size),
        ManifestFile(path="images/img2.png", size=(images / "img2.png").stat().st_size),
    ]
    manifest = Manifest(
        source_content_hash=src_hash,
        source_size_bytes=src_size,
        source_filename_at_parse=source_file.name,
        critical_file=ManifestFile(
            path="content_list.json", size=crit_size, sha256=crit_hash
        ),
        files=files,
        total_size_bytes=crit_size + sum(f.size for f in files),
        task_id="task-1",
        api_mode="local",
        options_signature=current_mineru_options_signature(),
    )
    write_manifest(raw, manifest)
    return raw, manifest


def _edit_manifest(raw: Path, fn: Callable[[dict], None]) -> None:
    mp = raw / "_manifest.json"
    payload = json.loads(mp.read_text(encoding="utf-8"))
    fn(payload)
    mp.write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def test_raw_dir_naming(tmp_path: Path) -> None:
    parsed = tmp_path / "report.pdf.parsed"
    raw = raw_dir_for_parsed_dir(parsed)
    assert raw.name == "report.pdf.mineru_raw"
    assert raw.parent == parsed.parent


# ---------------------------------------------------------------------------
# Validation: happy path, env compatibility, and mutation matrix
# ---------------------------------------------------------------------------


def test_is_bundle_valid_happy_path(
    fresh_bundle: tuple[Path, Manifest], source_file: Path
) -> None:
    raw, _ = fresh_bundle
    assert is_bundle_valid(raw, source_file) is True


def _compat_engine_version_match(raw: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _edit_manifest(raw, lambda p: p.__setitem__("engine_version", "magic-pdf 1.5.4"))
    monkeypatch.setenv("MINERU_ENGINE_VERSION", "magic-pdf 1.5.4")


def _compat_engine_version_blank_manifest(
    raw: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Blank manifest engine_version + non-blank env should NOT invalidate.
    monkeypatch.setenv("MINERU_ENGINE_VERSION", "anything")


def _compat_endpoint_trailing_slash(raw: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _update(p: dict) -> None:
        p["api_mode"] = "local"
        p["endpoint_signature"] = "http://old.example"

    _edit_manifest(raw, _update)
    monkeypatch.setenv("MINERU_API_MODE", "local")
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://old.example/")


@pytest.mark.parametrize(
    "setup",
    [
        pytest.param(_compat_engine_version_match, id="engine-version-match"),
        pytest.param(
            _compat_engine_version_blank_manifest, id="engine-version-blank-manifest"
        ),
        pytest.param(_compat_endpoint_trailing_slash, id="endpoint-trailing-slash"),
    ],
)
def test_is_bundle_valid_env_compatibility(
    fresh_bundle: tuple[Path, Manifest],
    source_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    setup: Callable[[Path, pytest.MonkeyPatch], None],
) -> None:
    raw, _ = fresh_bundle
    setup(raw, monkeypatch)
    assert is_bundle_valid(raw, source_file) is True


def _mut_manifest_missing(
    raw: Path, source_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (raw / "_manifest.json").unlink()


def _mut_manifest_malformed(
    raw: Path, source_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (raw / "_manifest.json").write_text("not json")


def _mut_manifest_not_utf8(
    raw: Path, source_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (raw / "_manifest.json").write_bytes(b"\xff\xfe")


def _mut_manifest_wrong_engine(
    raw: Path, source_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _edit_manifest(raw, lambda p: p.__setitem__("engine", "docling"))


def _mut_source_size_changes(
    raw: Path, source_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with source_file.open("ab") as fh:
        fh.write(b"x")


def _mut_source_hash_same_size(
    raw: Path, source_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = source_file.read_bytes()
    mutated = bytes([data[0] ^ 0xFF]) + data[1:]
    assert len(mutated) == len(data)
    source_file.write_bytes(mutated)


def _mut_engine_version_mismatch(
    raw: Path, source_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _edit_manifest(raw, lambda p: p.__setitem__("engine_version", "magic-pdf 1.5.4"))
    monkeypatch.setenv("MINERU_ENGINE_VERSION", "magic-pdf 1.6.0")


def _mut_api_mode_mismatch(
    raw: Path, source_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _edit_manifest(raw, lambda p: p.__setitem__("api_mode", "local"))
    monkeypatch.setenv("MINERU_API_MODE", "official")
    monkeypatch.setenv("MINERU_API_TOKEN", "token")


def _mut_options_signature_missing(
    raw: Path, source_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _edit_manifest(raw, lambda p: p.pop("options_signature", None))


def _mut_endpoint_signature_mismatch(
    raw: Path, source_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _update(p: dict) -> None:
        p["api_mode"] = "local"
        p["endpoint_signature"] = "http://old.example"

    _edit_manifest(raw, _update)
    monkeypatch.setenv("MINERU_API_MODE", "local")
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", "http://new.example")


def _mut_critical_file_missing(
    raw: Path, source_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (raw / "content_list.json").unlink()


def _mut_critical_file_size_changes(
    raw: Path, source_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cl = raw / "content_list.json"
    cl.write_text(cl.read_text() + "/* extra */")


def _mut_critical_file_hash_changes(
    raw: Path, source_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cl = raw / "content_list.json"
    data = cl.read_text()
    mutated = data[:-1] + "X"
    assert len(mutated) == len(data)
    cl.write_text(mutated)


def _mut_aux_file_size_changes(
    raw: Path, source_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = raw / "images" / "img1.png"
    p.write_bytes(p.read_bytes() + b"corruption")


def _mut_aux_file_missing(
    raw: Path, source_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (raw / "images" / "img2.png").unlink()


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(_mut_manifest_missing, id="manifest-missing"),
        pytest.param(_mut_manifest_malformed, id="manifest-malformed"),
        pytest.param(_mut_manifest_not_utf8, id="manifest-not-utf8"),
        pytest.param(_mut_manifest_wrong_engine, id="manifest-wrong-engine"),
        pytest.param(_mut_source_size_changes, id="source-size-changes"),
        pytest.param(_mut_source_hash_same_size, id="source-hash-same-size"),
        pytest.param(_mut_engine_version_mismatch, id="engine-version-mismatch"),
        pytest.param(_mut_api_mode_mismatch, id="api-mode-mismatch"),
        pytest.param(_mut_options_signature_missing, id="options-signature-missing"),
        pytest.param(
            _mut_endpoint_signature_mismatch, id="endpoint-signature-mismatch"
        ),
        pytest.param(_mut_critical_file_missing, id="critical-file-missing"),
        pytest.param(_mut_critical_file_size_changes, id="critical-file-size-changes"),
        pytest.param(_mut_critical_file_hash_changes, id="critical-file-hash-changes"),
        pytest.param(_mut_aux_file_size_changes, id="aux-file-size-changes"),
        pytest.param(_mut_aux_file_missing, id="aux-file-missing"),
    ],
)
def test_is_bundle_valid_rejects_invalid_mutations(
    fresh_bundle: tuple[Path, Manifest],
    source_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[Path, Path, pytest.MonkeyPatch], None],
) -> None:
    raw, _ = fresh_bundle
    mutate(raw, source_file, monkeypatch)
    assert is_bundle_valid(raw, source_file) is False


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("MINERU_LOCAL_BACKEND", "pipeline"),
        ("MINERU_LOCAL_PARSE_METHOD", "ocr"),
        ("MINERU_LOCAL_IMAGE_ANALYSIS", "true"),
        ("MINERU_LOCAL_START_PAGE_ID", "1"),
    ],
)
def test_invalid_when_local_parser_options_change(
    fresh_bundle: tuple[Path, Manifest],
    source_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: str,
) -> None:
    raw, _ = fresh_bundle
    monkeypatch.setenv("MINERU_API_MODE", "local")
    monkeypatch.setenv(key, value)
    assert is_bundle_valid(raw, source_file) is False


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("MINERU_MODEL_VERSION", "pipeline"),
        ("MINERU_IS_OCR", "true"),
        ("MINERU_PAGE_RANGES", "1-5"),
        ("MINERU_LANGUAGE", "en"),
        ("MINERU_ENABLE_TABLE", "false"),
        ("MINERU_ENABLE_FORMULA", "false"),
    ],
)
def test_invalid_when_official_parser_options_change(
    tmp_path: Path,
    source_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: str,
) -> None:
    """Symmetric coverage for the official-mode partition of the signature."""
    monkeypatch.setenv("MINERU_API_MODE", "official")

    raw = tmp_path / "src.mineru_raw"
    raw.mkdir()
    content_list = raw / "content_list.json"
    content_list.write_text('[{"type":"text","text":"hi"}]', encoding="utf-8")
    crit_size, crit_hash = compute_size_and_hash(content_list)
    src_size, src_hash = compute_size_and_hash(source_file)
    manifest = Manifest(
        source_content_hash=src_hash,
        source_size_bytes=src_size,
        source_filename_at_parse=source_file.name,
        critical_file=ManifestFile(
            path="content_list.json", size=crit_size, sha256=crit_hash
        ),
        files=[],
        total_size_bytes=crit_size,
        task_id="task-official",
        api_mode="official",
        options_signature=current_mineru_options_signature(),
    )
    write_manifest(raw, manifest)

    assert is_bundle_valid(raw, source_file) is True
    monkeypatch.setenv(key, value)
    assert is_bundle_valid(raw, source_file) is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_clear_dir_contents_preserves_directory(tmp_path: Path) -> None:
    d = tmp_path / "raw"
    d.mkdir()
    (d / "a.txt").write_text("a")
    (d / "sub").mkdir()
    (d / "sub" / "b.txt").write_text("b")
    clear_dir_contents(d)
    assert d.exists()
    assert list(d.iterdir()) == []


def test_compute_size_and_hash_consistency(tmp_path: Path) -> None:
    """Both values describe the same byte stream."""
    p = tmp_path / "f.bin"
    payload = b"abc" * 1000
    p.write_bytes(payload)
    size, h = compute_size_and_hash(p)
    assert size == len(payload)
    assert h.startswith("sha256:") and len(h) == len("sha256:") + 64


# ---------------------------------------------------------------------------
# delete_file_variants_by_file_path cleanup of *.mineru_raw/ bundles
# ---------------------------------------------------------------------------


def test_canonical_basename_recognizes_both_suffixes() -> None:
    assert _file_path_for_parsed_artifact_dir("foo.pdf.parsed") == "foo.pdf"
    assert _file_path_for_parsed_artifact_dir("foo.pdf.mineru_raw") == "foo.pdf"
    assert _file_path_for_parsed_artifact_dir("foo.pdf.parsed_001") == "foo.pdf"
    assert _file_path_for_parsed_artifact_dir("foo.pdf.mineru_raw_002") == "foo.pdf"
    assert _file_path_for_parsed_artifact_dir("foo.parsed.bak") is None
    assert _file_path_for_parsed_artifact_dir("notes.txt") is None


def test_delete_file_variants_removes_parsed_and_mineru_raw(tmp_path: Path) -> None:
    """Both sidecar dir and raw bundle dir should disappear in one call."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    parsed_root = input_dir / PARSED_DIR_NAME
    parsed_root.mkdir()

    archived_source = parsed_root / "demo.pdf"
    archived_source.write_bytes(b"PDFCONTENT")

    parsed_dir = parsed_root / "demo.pdf.parsed"
    parsed_dir.mkdir()
    (parsed_dir / "demo.blocks.jsonl").write_text("{}\n")

    raw_dir = parsed_root / "demo.pdf.mineru_raw"
    raw_dir.mkdir()
    (raw_dir / "_manifest.json").write_text("{}")
    (raw_dir / "content_list.json").write_text("[]")
    (raw_dir / "images").mkdir()
    (raw_dir / "images" / "img.png").write_bytes(b"png")

    deleted, errors = delete_file_variants_by_file_path(input_dir, file_path="demo.pdf")

    assert errors == []
    deleted_names = {Path(p).name for p in deleted}
    assert "demo.pdf.parsed" in deleted_names
    assert "demo.pdf.mineru_raw" in deleted_names
    assert "demo.pdf" in deleted_names
    assert not parsed_dir.exists()
    assert not raw_dir.exists()
    assert not archived_source.exists()


def test_delete_file_variants_handles_only_raw_dir(tmp_path: Path) -> None:
    """A raw bundle without a sidecar (aborted mid-parse) is still deleted."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    parsed_root = input_dir / PARSED_DIR_NAME
    parsed_root.mkdir()

    raw_dir = parsed_root / "demo.pdf.mineru_raw"
    raw_dir.mkdir()
    (raw_dir / "_manifest.json").write_text("{}")

    deleted, errors = delete_file_variants_by_file_path(input_dir, file_path="demo.pdf")
    assert errors == []
    assert any("demo.pdf.mineru_raw" in p for p in deleted)
    assert not raw_dir.exists()


def test_delete_file_variants_leaves_unrelated_dirs(tmp_path: Path) -> None:
    """Directories that don't match the canonical filename are untouched."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    parsed_root = input_dir / PARSED_DIR_NAME
    parsed_root.mkdir()

    target_raw = parsed_root / "demo.pdf.mineru_raw"
    target_raw.mkdir()
    other_raw = parsed_root / "other.pdf.mineru_raw"
    other_raw.mkdir()
    other_parsed = parsed_root / "other.pdf.parsed"
    other_parsed.mkdir()

    deleted, errors = delete_file_variants_by_file_path(input_dir, file_path="demo.pdf")
    assert errors == []
    assert not target_raw.exists()
    assert other_raw.exists(), "siblings for a different basename must survive"
    assert other_parsed.exists()
