"""Tests for source-ID merging/subtraction, attribute joining, tracking rows, and cache lists.

Covers:
- ``merge_source_ids``: splitting ``GRAPH_FIELD_SEP``, preserving order,
  stripping/deduping, handling empty/None inputs, non-string warning, and
  debug logging on split.
- ``subtract_source_ids``: removal, order preservation, duplicate retention,
  falsy-ID dropping on both empty and non-empty removal paths, and generators.
- ``_merge_attributes`` ``join_unique`` deterministic first-seen order for
  ``source_id`` / ``file_path`` under ``FIFO`` and ``IGNORE_NEW`` limits.
- ``has_chunk_tracking_row`` schema truth table.
- ``update_chunk_cache_list`` durability return value without raising.
"""

from __future__ import annotations

import logging

import pytest

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.utils import (
    apply_source_ids_limit,
    has_chunk_tracking_row,
    merge_source_ids,
    subtract_source_ids,
    update_chunk_cache_list,
)
from lightrag.utils import logger as lightrag_logger
from lightrag.utils_graph import _merge_attributes

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# merge_source_ids
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("existing", "new", "expected"),
    [
        pytest.param(
            ["chunk-1"],
            [f"chunk-2{GRAPH_FIELD_SEP}chunk-3", "chunk-3"],
            ["chunk-1", "chunk-2", "chunk-3"],
            id="splits-new-joined-ids",
        ),
        pytest.param(
            [f"chunk-1{GRAPH_FIELD_SEP}chunk-2"],
            ["chunk-2", "chunk-3"],
            ["chunk-1", "chunk-2", "chunk-3"],
            id="splits-existing-joined-ids",
        ),
        pytest.param(
            ["chunk-b", "chunk-a"],
            ["chunk-a", "chunk-c", "chunk-b"],
            ["chunk-b", "chunk-a", "chunk-c"],
            id="preserves-first-seen-order",
        ),
        pytest.param(
            ["   "],
            [
                f"  chunk-1  {GRAPH_FIELD_SEP}{GRAPH_FIELD_SEP}   {GRAPH_FIELD_SEP}chunk-2"
            ],
            ["chunk-1", "chunk-2"],
            id="strips-and-drops-blank-fragments",
        ),
        pytest.param(
            ["chunk-1"],
            ["  chunk-1  "],
            ["chunk-1"],
            id="dedups-after-stripping",
        ),
        pytest.param(None, None, [], id="both-none"),
        pytest.param([], None, [], id="empty-and-none"),
        pytest.param(None, ["chunk-1"], ["chunk-1"], id="none-and-list"),
        pytest.param(["chunk-1"], [], ["chunk-1"], id="list-and-empty"),
        pytest.param(
            ["", "chunk-1"],
            [None, "chunk-2"],
            ["chunk-1", "chunk-2"],
            id="skips-falsy-elements",
        ),
    ],
)
def test_merge_source_ids_normalization(existing, new, expected):
    merged = merge_source_ids(existing, new)
    assert merged == expected
    assert not any(GRAPH_FIELD_SEP in item for item in merged)


def test_merge_source_ids_coerces_non_string_ids_with_a_warning(lightrag_caplog):
    with lightrag_caplog.at_level(logging.WARNING, logger=lightrag_logger.name):
        merged = merge_source_ids([123], ["chunk-1"])

    assert merged == ["123", "chunk-1"]
    assert any("non-string id" in r.getMessage() for r in lightrag_caplog.records)


def test_merge_source_ids_logs_when_it_has_to_split(lightrag_caplog):
    with lightrag_caplog.at_level(logging.DEBUG, logger=lightrag_logger.name):
        merge_source_ids([], [f"chunk-1{GRAPH_FIELD_SEP}chunk-2"])

    assert any(
        "GRAPH_FIELD_SEP-joined" in r.getMessage() for r in lightrag_caplog.records
    )


def test_merge_source_ids_does_not_log_for_normal_input(lightrag_caplog):
    with lightrag_caplog.at_level(logging.DEBUG, logger=lightrag_logger.name):
        merge_source_ids(["chunk-1"], ["chunk-2"])

    assert not [r for r in lightrag_caplog.records if r.name == lightrag_logger.name]


# ---------------------------------------------------------------------------
# subtract_source_ids
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "removal", "expected"),
    [
        pytest.param(["a", "b", "c"], ["b"], ["a", "c"], id="removes-listed-id"),
        pytest.param(["a", "b"], ["a", "b"], [], id="removes-every-id"),
        pytest.param(
            ["d", "a", "c", "b"], ["c"], ["d", "a", "b"], id="preserves-source-order"
        ),
        pytest.param(
            ["a", "b", "c"],
            {"c", "a"},
            ["b"],
            id="ignores-removal-set-iteration-order",
        ),
        pytest.param(
            ["a", "a", "b"],
            ["b"],
            ["a", "a"],
            id="keeps-duplicate-survivors-as-is",
        ),
        pytest.param(
            (sid for sid in ["a", "b"]), ["b"], ["a"], id="accepts-generator-source"
        ),
    ],
)
def test_subtract_source_ids_removal_and_order(source, removal, expected):
    assert subtract_source_ids(source, removal) == expected


@pytest.mark.parametrize(
    ("source", "removal", "expected"),
    [
        pytest.param(["a", "", "b"], [], ["a", "b"], id="empty-removal-fast-path"),
        pytest.param(["a", "", "b"], ["x"], ["a", "b"], id="non-empty-removal-path"),
        pytest.param(["", ""], [], [], id="all-empty-source"),
        pytest.param(["a", None, "b"], [], ["a", "b"], id="none-entries"),
    ],
)
def test_subtract_source_ids_drops_falsy_ids(source, removal, expected):
    """Falsy ids must never survive either the fast or non-empty removal path."""
    assert subtract_source_ids(source, removal) == expected


# ---------------------------------------------------------------------------
# _merge_attributes join_unique ordering
# ---------------------------------------------------------------------------

# Set iteration order is sensitive to the exact strings, and this spelling is
# the one the seed check covered: do not renumber or re-pad these.
FIRST_HALF = [f"chunk-{i:02d}" for i in range(1, 11)]
SECOND_HALF = [f"chunk-{i:02d}" for i in range(11, 21)]
ALL_CHUNKS = FIRST_HALF + SECOND_HALF
HALF = len(FIRST_HALF)


@pytest.mark.parametrize("key", ["source_id", "file_path"])
def test_join_unique_preserves_first_seen_order(key):
    first = {key: GRAPH_FIELD_SEP.join(FIRST_HALF)}
    second = {key: GRAPH_FIELD_SEP.join(SECOND_HALF)}

    merged = _merge_attributes([first, second], {key: "join_unique"})

    assert merged[key].split(GRAPH_FIELD_SEP) == ALL_CHUNKS


def test_join_unique_dedups_without_reordering():
    overlap = FIRST_HALF[-3:]
    first = {"source_id": GRAPH_FIELD_SEP.join(FIRST_HALF)}
    second = {"source_id": GRAPH_FIELD_SEP.join(overlap + SECOND_HALF)}

    merged = _merge_attributes([first, second], {"source_id": "join_unique"})

    assert merged["source_id"].split(GRAPH_FIELD_SEP) == ALL_CHUNKS


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        pytest.param("FIFO", SECOND_HALF, id="fifo-keeps-newest"),
        pytest.param("IGNORE_NEW", FIRST_HALF, id="ignore-new-keeps-oldest"),
    ],
)
def test_join_unique_order_matches_source_id_limit_methods(method, expected):
    merged = _merge_attributes(
        [
            {"source_id": GRAPH_FIELD_SEP.join(FIRST_HALF)},
            {"source_id": GRAPH_FIELD_SEP.join(SECOND_HALF)},
        ],
        {"source_id": "join_unique"},
    )

    kept = apply_source_ids_limit(
        merged["source_id"].split(GRAPH_FIELD_SEP), HALF, method
    )

    assert kept == expected


# ---------------------------------------------------------------------------
# has_chunk_tracking_row truth table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stored",
    [
        {"chunk_ids": [], "count": 0},  # curated empty row — authoritative
        {"chunk_ids": ["c1"], "count": 1},
        {"chunk_ids": ["c1", "c2"]},  # count missing is irrelevant
    ],
)
def test_has_chunk_tracking_row_present_rows(stored):
    assert has_chunk_tracking_row(stored) is True


@pytest.mark.parametrize(
    "stored",
    [
        None,  # absent row
        {},  # legacy/partial: no chunk_ids key
        {"count": 0},  # legacy/partial: no chunk_ids key
        {"chunk_ids": None},  # malformed: null instead of list
        {"chunk_ids": "c1"},  # malformed: scalar instead of list
        {"chunk_ids": {"c1": 1}},  # malformed: wrong container
        [],  # malformed: not a dict
        "row",  # malformed: not a dict
    ],
)
def test_has_chunk_tracking_row_absent_or_malformed_rows(stored):
    assert has_chunk_tracking_row(stored) is False


# ---------------------------------------------------------------------------
# update_chunk_cache_list durability
# ---------------------------------------------------------------------------


class _FakeKV:
    def __init__(self, rows: dict | None = None):
        self.data = dict(rows or {})
        self.upserts = 0
        self.fail_on_read = False
        self.fail_on_upsert = False

    async def get_by_id(self, key):
        if self.fail_on_read:
            raise RuntimeError("read is down")
        return self.data.get(key)

    async def upsert(self, rows: dict):
        if self.fail_on_upsert:
            raise RuntimeError("write is down")
        self.data.update(rows)
        self.upserts += 1


@pytest.mark.asyncio
async def test_no_keys_to_record_is_durable():
    storage = _FakeKV({"chunk-1": {"content": "c"}})

    assert await update_chunk_cache_list("chunk-1", storage, []) is True
    assert storage.upserts == 0


@pytest.mark.asyncio
async def test_recording_a_new_key_reports_success():
    storage = _FakeKV({"chunk-1": {"content": "c"}})

    assert await update_chunk_cache_list("chunk-1", storage, ["k1"]) is True
    assert storage.data["chunk-1"]["llm_cache_list"] == ["k1"]
    assert storage.upserts == 1


@pytest.mark.asyncio
async def test_an_already_recorded_key_is_durable_without_a_write():
    """The steady state of the re-ingest self-heal must not cost a write."""
    storage = _FakeKV({"chunk-1": {"content": "c", "llm_cache_list": ["k1"]}})

    assert await update_chunk_cache_list("chunk-1", storage, ["k1"]) is True
    assert storage.upserts == 0
    assert storage.data["chunk-1"]["llm_cache_list"] == ["k1"]


@pytest.mark.asyncio
async def test_a_missing_chunk_row_reports_failure():
    """Nothing can carry the reference, so the caller must not write the row."""
    storage = _FakeKV()

    assert await update_chunk_cache_list("chunk-1", storage, ["k1"]) is False
    assert storage.upserts == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_attr", ["fail_on_read", "fail_on_upsert"])
async def test_storage_errors_report_failure_without_raising(fail_attr):
    storage = _FakeKV({"chunk-1": {"content": "c"}})
    setattr(storage, fail_attr, True)

    assert await update_chunk_cache_list("chunk-1", storage, ["k1"]) is False
