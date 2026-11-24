"""Tests for ``lightrag/parser/external/docling/cache.py`` and ``manifest.py``.

Covers the cache-miss conditions and manifest JSON selection contract:
- missing / malformed / wrong-engine manifest
- source size or hash mismatch
- engine_version / endpoint_signature / options_signature env mismatch
- critical-file sha256 mismatch and non-critical file size mismatch
- legacy unsigned bundle handling under overrides / VLM pipeline
- ``select_main_json`` preferred-path and fallback filtering around ``_manifest.json``
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from lightrag.parser.external import Manifest, ManifestFile, write_manifest
from lightrag.parser.external._common import compute_size_and_hash
from lightrag.parser.external.docling.cache import (
    compute_options_signature,
    is_bundle_valid,
    snapshot_tunable_env,
)
from lightrag.parser.external.docling.client import FIXED_CONSTANTS
from lightrag.parser.external.docling.manifest import select_main_json


pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _clear_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "DOCLING_DO_OCR",
        "DOCLING_FORCE_OCR",
        "DOCLING_OCR_ENGINE",
        "DOCLING_OCR_PRESET",
        "DOCLING_OCR_LANG",
        "DOCLING_DO_FORMULA_ENRICHMENT",
        "DOCLING_ENGINE_VERSION",
        "DOCLING_ENDPOINT",
        "DOCLING_PIPELINE",
        "DOCLING_VLM_PRESET",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DOCLING_ENDPOINT", "http://docling.test")


@pytest.fixture
def source_file(tmp_path: Path) -> Path:
    p = tmp_path / "src.pdf"
    p.write_bytes(b"hello pdf payload" * 64)
    return p


def test_snapshot_tunable_env_uses_effective_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unset = snapshot_tunable_env()

    monkeypatch.setenv("DOCLING_DO_OCR", "true")
    monkeypatch.setenv("DOCLING_FORCE_OCR", "true")
    monkeypatch.setenv("DOCLING_OCR_ENGINE", "auto")
    monkeypatch.setenv("DOCLING_OCR_PRESET", "auto")
    monkeypatch.setenv("DOCLING_OCR_LANG", "")
    monkeypatch.setenv("DOCLING_DO_FORMULA_ENRICHMENT", "false")

    assert snapshot_tunable_env() == unset


def _build_valid_bundle(
    tmp_path: Path,
    source_file: Path,
    *,
    options_signature: str | None = None,
) -> Path:
    raw_dir = tmp_path / "src.docling_raw"
    raw_dir.mkdir()
    main_json = raw_dir / "src.json"
    main_json.write_text('{"schema_name": "DoclingDocument"}', encoding="utf-8")
    md = raw_dir / "src.md"
    md.write_text("# title", encoding="utf-8")

    src_size, src_hash = compute_size_and_hash(source_file)
    crit_size, crit_hash = compute_size_and_hash(main_json)
    sig = options_signature
    if sig is None:
        sig = compute_options_signature(
            tunable_env=snapshot_tunable_env(),
            fixed_constants=FIXED_CONSTANTS,
        )
    manifest = Manifest(
        engine="docling",
        source_content_hash=src_hash,
        source_size_bytes=src_size,
        source_filename_at_parse=source_file.name,
        critical_file=ManifestFile(path="src.json", size=crit_size, sha256=crit_hash),
        files=[ManifestFile(path="src.md", size=md.stat().st_size)],
        total_size_bytes=crit_size + md.stat().st_size,
        task_id="task-1",
        endpoint_signature="http://docling.test",
        engine_version="",
        options_signature=sig,
        extras={"fixed_constants": dict(FIXED_CONSTANTS)},
    )
    write_manifest(raw_dir, manifest)
    return raw_dir


def test_is_bundle_valid_happy_path(tmp_path: Path, source_file: Path) -> None:
    raw = _build_valid_bundle(tmp_path, source_file)
    assert is_bundle_valid(raw, source_file) is True


def _mut_missing_dir(
    tmp_path: Path, source_file: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    return tmp_path / "ghost"


def _mut_missing_manifest(
    tmp_path: Path, source_file: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    raw = tmp_path / "src.docling_raw"
    raw.mkdir()
    (raw / "src.json").write_text("{}")
    return raw


def _mut_manifest_not_utf8(
    tmp_path: Path, source_file: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    raw = _build_valid_bundle(tmp_path, source_file)
    (raw / "_manifest.json").write_bytes(b"\xff\xfe")
    return raw


def _mut_wrong_engine(
    tmp_path: Path, source_file: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    raw = _build_valid_bundle(tmp_path, source_file)
    manifest_path = raw / "_manifest.json"
    data = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(data.replace('"docling"', '"mineru"'), encoding="utf-8")
    return raw


def _mut_source_size(
    tmp_path: Path, source_file: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    raw = _build_valid_bundle(tmp_path, source_file)
    source_file.write_bytes(source_file.read_bytes() + b"!")
    return raw


def _mut_source_hash_same_size(
    tmp_path: Path, source_file: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    raw = _build_valid_bundle(tmp_path, source_file)
    source_file.write_bytes(b"Y" * source_file.stat().st_size)
    return raw


def _mut_endpoint_change(
    tmp_path: Path, source_file: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    raw = _build_valid_bundle(tmp_path, source_file)
    monkeypatch.setenv("DOCLING_ENDPOINT", "http://other:5001")
    return raw


def _mut_options_signature(
    tmp_path: Path, source_file: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    raw = _build_valid_bundle(tmp_path, source_file)
    monkeypatch.setenv("DOCLING_FORCE_OCR", "false")
    return raw


def _mut_critical_file_corrupt(
    tmp_path: Path, source_file: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    raw = _build_valid_bundle(tmp_path, source_file)
    # Same length, different bytes — defeats size check, sha256 must catch it.
    current = (raw / "src.json").read_bytes()
    (raw / "src.json").write_bytes(b"X" * len(current))
    return raw


def _mut_other_file_size(
    tmp_path: Path, source_file: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    raw = _build_valid_bundle(tmp_path, source_file)
    (raw / "src.md").write_text("totally different content here that is longer")
    return raw


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(_mut_missing_dir, id="missing-dir"),
        pytest.param(_mut_missing_manifest, id="missing-manifest"),
        pytest.param(_mut_manifest_not_utf8, id="manifest-not-utf8"),
        pytest.param(_mut_wrong_engine, id="wrong-engine"),
        pytest.param(_mut_source_size, id="source-size-mismatch"),
        pytest.param(_mut_source_hash_same_size, id="source-hash-mismatch"),
        pytest.param(_mut_endpoint_change, id="endpoint-change"),
        pytest.param(_mut_options_signature, id="options-signature-change"),
        pytest.param(_mut_critical_file_corrupt, id="critical-file-corrupt"),
        pytest.param(_mut_other_file_size, id="other-file-size-mismatch"),
    ],
)
def test_is_bundle_valid_rejects_invalid_mutations(
    tmp_path: Path,
    source_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[Path, Path, pytest.MonkeyPatch], Path],
) -> None:
    raw = mutate(tmp_path, source_file, monkeypatch)
    assert is_bundle_valid(raw, source_file) is False


def test_is_bundle_valid_legacy_bundle_misses_with_overrides(
    tmp_path: Path, source_file: Path
) -> None:
    # A legacy bundle predating signature recording (empty options_signature)
    # is leniently accepted with no overrides, but a per-file override
    # (docling(force_ocr=...)) must force a miss.
    raw = _build_valid_bundle(tmp_path, source_file, options_signature="")
    assert is_bundle_valid(raw, source_file) is True
    assert is_bundle_valid(raw, source_file, overrides={"force_ocr": False}) is False
    assert is_bundle_valid(raw, source_file, overrides={"force_ocr": True}) is False


def test_is_bundle_valid_legacy_bundle_misses_under_vlm_pipeline(
    tmp_path: Path, source_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An unsigned legacy bundle can only come from the standard pipeline.
    raw = _build_valid_bundle(tmp_path, source_file, options_signature="")
    assert is_bundle_valid(raw, source_file) is True
    monkeypatch.setenv("DOCLING_PIPELINE", "vlm")
    monkeypatch.setenv("DOCLING_VLM_PRESET", "minicpm_ocr")
    assert is_bundle_valid(raw, source_file) is False
    monkeypatch.setenv("DOCLING_PIPELINE", "standard")
    assert is_bundle_valid(raw, source_file) is True


def test_is_bundle_valid_fixed_constants_code_change(
    tmp_path: Path, source_file: Path
) -> None:
    # Validation must compare against current FIXED_CONSTANTS, never rehydrate
    # stale constants from the manifest's own extras.
    stale_constants = {**FIXED_CONSTANTS, "image_export_mode": "embedded"}
    stale_signature = compute_options_signature(
        tunable_env=snapshot_tunable_env(),
        fixed_constants=stale_constants,
    )
    raw = _build_valid_bundle(tmp_path, source_file, options_signature=stale_signature)
    mp = raw / "_manifest.json"
    data = json.loads(mp.read_text(encoding="utf-8"))
    data["extras"] = {"fixed_constants": stale_constants}
    mp.write_text(json.dumps(data), encoding="utf-8")

    assert is_bundle_valid(raw, source_file) is False


# ---------------------------------------------------------------------------
# Manifest main-JSON selection (lightrag/parser/external/docling/manifest.py)
# ---------------------------------------------------------------------------


def _touch(path: Path, content: str = "{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.mark.parametrize(
    ("source_name", "expected_name"),
    [
        pytest.param("report.pdf", "report.json", id="preferred-path-hits"),
        pytest.param("other.pdf", "report.json", id="fallback-ignores-manifest"),
    ],
)
def test_select_main_json_resolves_bundle_json(
    tmp_path: Path, source_name: str, expected_name: str
) -> None:
    _touch(tmp_path / "report.json")
    _touch(tmp_path / "_manifest.json", '{"engine":"docling"}')
    assert select_main_json(tmp_path, Path(source_name)) == tmp_path / expected_name


@pytest.mark.parametrize(
    ("extra_files", "source_name", "error_match"),
    [
        pytest.param((), "report.pdf", "contains no .json file", id="only-manifest"),
        pytest.param(
            ("report.json", "extra.json"),
            "other.pdf",
            "multiple .json candidates",
            id="real-ambiguity",
        ),
    ],
)
def test_select_main_json_errors(
    tmp_path: Path,
    extra_files: tuple[str, ...],
    source_name: str,
    error_match: str,
) -> None:
    for name in extra_files:
        _touch(tmp_path / name)
    _touch(tmp_path / "_manifest.json", '{"engine":"docling"}')
    with pytest.raises(RuntimeError, match=error_match):
        select_main_json(tmp_path, Path(source_name))
