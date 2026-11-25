"""Tests for hashing, incremental chunk-ID tracking, JSON loading, and archive moves.

Covers:
- ``compute_args_hash`` / ``compute_mdhash_id``: length-prefixed collision
  immunity, single-arg legacy MD5 stability, and surrogate Unicode handling.
- ``compute_incremental_chunk_ids``: tracking authority over stale graph
  ``source_id`` entries, deduplication, and empty-ID stripping.
- ``load_json``: missing/empty/whitespace files vs. valid/invalid JSON.
- ``move_file_to_parsed_dir``: timestamp archive collision preservation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightrag import utils
from lightrag.utils import (
    compute_args_hash,
    compute_incremental_chunk_ids,
    compute_mdhash_id,
    load_json,
)

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# compute_args_hash / compute_mdhash_id
# ---------------------------------------------------------------------------


def test_adjacent_text_field_boundary_does_not_collide():
    a = compute_args_hash("hybrid", "abc", "x", 10, 20, 6000, 8000, 30000)
    b = compute_args_hash("hybrid", "ab", "cx", 10, 20, 6000, 8000, 30000)
    assert a != b, (
        f"Adjacent free-text boundary shift must not collide, both hashed to {a}"
    )


def test_adjacent_integer_field_boundary_does_not_collide():
    c = compute_args_hash("hybrid", "q", "text", 1, 20, 6000, 8000, 30000)
    d = compute_args_hash("hybrid", "q", "text", 12, 0, 6000, 8000, 30000)
    assert c != d, (
        f"Adjacent integer boundary shift must not collide, both hashed to {c}"
    )


def test_inputs_containing_record_separator_do_not_collide():
    """Length-prefixing must stay immune to embedded delimiter characters."""
    e = compute_args_hash("a\x1eb", "c")
    f = compute_args_hash("a", "b\x1ec")
    assert e != f


def test_identical_arguments_hash_identically():
    g = compute_args_hash("hybrid", "abc", "x", 10, 20, 6000, 8000, 30000)
    h = compute_args_hash("hybrid", "abc", "x", 10, 20, 6000, 8000, 30000)
    assert g == h


def test_single_argument_preserves_legacy_hash():
    """Single-arg ``compute_args_hash`` / ``compute_mdhash_id`` stays raw MD5."""
    content = "hello world test content"
    expected_raw_md5 = hashlib.md5(content.encode("utf-8")).hexdigest()

    assert compute_mdhash_id(content) == expected_raw_md5
    assert compute_mdhash_id(content, prefix="doc-") == "doc-" + expected_raw_md5

    empty_md5 = hashlib.md5(b"").hexdigest()
    assert compute_args_hash("") == empty_md5
    assert compute_mdhash_id("") == empty_md5


def test_surrogate_unicode_is_stable_and_distinguishable():
    bad = "before\udcffafter"
    h1 = compute_args_hash(bad, "field2")
    h2 = compute_args_hash(bad, "field2")
    assert isinstance(h1, str) and len(h1) == 32
    assert h1 == h2

    a = compute_args_hash("ab\udcff", "xy")
    b = compute_args_hash("ab", "\udcffxy")
    assert a != b


# ---------------------------------------------------------------------------
# compute_incremental_chunk_ids
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("existing", "old", "new", "expected"),
    [
        pytest.param(
            ["chunk-1", "chunk-2", "chunk-3"],
            ["chunk-1", "chunk-2"],
            ["chunk-2", "chunk-4"],
            ["chunk-2", "chunk-3", "chunk-4"],
            id="standard-update",
        ),
        pytest.param(
            ["chunk-3"],
            ["chunk-1", "chunk-3"],
            ["chunk-1", "chunk-2", "chunk-3"],
            ["chunk-3", "chunk-2"],
            id="does-not-resurrect-stale-graph-ids",
        ),
        pytest.param(
            ["live"],
            ["stale", "live"],
            ["stale", "live"],
            ["live"],
            id="rename-unchanged-source-id-does-not-reseed-pruned-ids",
        ),
        pytest.param(
            ["chunk-a", "chunk-b"],
            ["chunk-b"],
            ["chunk-a", "chunk-b"],
            ["chunk-a", "chunk-b"],
            id="no-duplicate-when-existing-already-holds-addition",
        ),
        pytest.param(
            ["", "chunk-a"],
            [],
            ["", "chunk-b"],
            ["chunk-a", "chunk-b"],
            id="empty-chunk-ids-dropped",
        ),
        pytest.param(
            ["chunk-1", "chunk-2"],
            ["chunk-1", "chunk-2"],
            [],
            [],
            id="all-chunks-removed",
        ),
    ],
)
def test_compute_incremental_chunk_ids(existing, old, new, expected):
    assert compute_incremental_chunk_ids(existing, old, new) == expected


# ---------------------------------------------------------------------------
# load_json
# ---------------------------------------------------------------------------


def test_load_json_missing_file_returns_none(tmp_path: Path) -> None:
    assert load_json(str(tmp_path / "no-such.json")) is None


@pytest.mark.parametrize("payload", [b"", b"   \n\t  "])
def test_load_json_empty_or_whitespace_file_returns_none(
    tmp_path: Path, payload: bytes
) -> None:
    path = tmp_path / "kv.json"
    path.write_bytes(payload)
    assert load_json(str(path)) is None
    assert (load_json(str(path)) or {}) == {}


def test_load_json_invalid_json_still_raises(tmp_path: Path) -> None:
    path = tmp_path / "kv.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_json(str(path))


@pytest.mark.parametrize("text, expected", [('{"a": 1}', {"a": 1}), ("{}", {})])
def test_load_json_valid_json_loads(tmp_path: Path, text: str, expected: dict) -> None:
    path = tmp_path / "kv.json"
    path.write_text(text, encoding="utf-8")
    assert load_json(str(path)) == expected


# ---------------------------------------------------------------------------
# move_file_to_parsed_dir
# ---------------------------------------------------------------------------


async def test_move_preserves_existing_timestamp_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "time", SimpleNamespace(time=lambda: 1_700_000_000))
    parsed_dir = tmp_path / utils.PARSED_DIR_NAME
    parsed_dir.mkdir()
    (parsed_dir / "report.txt").touch()
    for slot in range(1, 1000):
        (parsed_dir / f"report_{slot:03d}.txt").touch()
    existing_archive = parsed_dir / "report_1700000000.txt"
    existing_archive.write_bytes(b"previous report")
    source = tmp_path / "report.txt"
    source.write_bytes(b"new report")

    archived = await utils.move_file_to_parsed_dir(source)

    assert existing_archive.read_bytes() == b"previous report"
    assert archived == parsed_dir / "report_1700000001.txt"
    assert archived.read_bytes() == b"new report"
    assert not source.exists()
