"""JsonDocStatusStorage JSON-specific behavior, scheduling details, copy-on-read, and track_id hydration.

Cross-backend scheduling invariants (sort order, cursor pagination, multi-status
merge, projection, strict counts/reads, source resolution, conflict listing and
CAS repair, concurrency) live in ``tests/kg/test_doc_status_scheduling_contract.py``
and run offline against JsonDocStatusStorage. This file covers JSON-specific
details not shared across all five backends.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import patch

import pytest

from lightrag.base import (
    CURSOR_END,
    CURSOR_START,
    CursorAfter,
    DocStatus,
    SourceUnique,
)
from lightrag.exceptions import (
    SourceConflictRepairCASError,
    StorageControlPlaneError,
    StorageRecordNotFoundError,
)
from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage
from lightrag.kg.shared_storage import (
    NamespaceLock,
    finalize_share_data,
    initialize_share_data,
)
from lightrag.namespace import NameSpace

pytestmark = pytest.mark.offline


class _DummyEmbeddingFunc:
    """Minimal embedding stand-in for storages that do not embed."""

    embedding_dim = 1
    max_token_size = 1

    async def __call__(self, texts, **kwargs):
        return [[0.0] for _ in texts]


def _doc(
    status: str,
    file_path: str = "a.pdf",
    created_at: str = "2024-01-01T00:00:00+00:00",
    **extra,
) -> dict:
    row = {
        "content_summary": f"{status} summary",
        "content_length": 10,
        "file_path": file_path,
        "status": status,
        "created_at": created_at,
        "updated_at": created_at,
        "metadata": {},
        "error_msg": None,
        "chunks_list": [],
    }
    row.update(extra)
    return row


async def _storage(
    tmp_path,
    rows: dict | None = None,
    *,
    namespace: str = "doc_status",
) -> JsonDocStatusStorage:
    storage = JsonDocStatusStorage(
        namespace=namespace,
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    await storage.initialize()
    if rows:
        async with storage._storage_lock:
            storage._data.update(rows)
    return storage


async def _sweep_ids(storage, statuses, *, limit):
    ids: list[str] = []
    position = CURSOR_START
    for _ in range(100):
        page = await storage.get_docs_by_statuses_page(
            statuses,
            limit=limit,
            position=position,
            strict=True,
        )
        ids.extend(page.docs.keys())
        if page.next_position is CURSOR_END:
            return ids
        assert isinstance(page.next_position, CursorAfter)
        position = page.next_position
    pytest.fail("sweep failed to terminate")


@pytest.fixture(autouse=True)
def setup_shared_data():
    initialize_share_data()
    yield
    finalize_share_data()


# ---------------------------------------------------------------------------
# Pagination, file paths, and strict/relaxed status queries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_docs_paginated_with_status_filters(tmp_path):
    storage = await _storage(
        tmp_path,
        {
            "doc-1": _doc("preprocessed", "a.pdf"),
            "doc-2": _doc("parsing", "b.pdf"),
            "doc-3": _doc("analyzing", "c.pdf"),
            "doc-4": _doc("processed", "d.pdf"),
        },
    )

    docs, total = await storage.get_docs_paginated(
        status_filter=DocStatus.PROCESSED,
        status_filters=[
            DocStatus.PREPROCESSED,
            DocStatus.PARSING,
            DocStatus.ANALYZING,
        ],
        page=1,
        page_size=10,
        sort_field="id",
        sort_direction="asc",
    )

    assert total == 3
    assert [doc_id for doc_id, _ in docs] == ["doc-1", "doc-2", "doc-3"]
    assert [doc.status for _, doc in docs] == [
        DocStatus.PREPROCESSED,
        DocStatus.PARSING,
        DocStatus.ANALYZING,
    ]


@pytest.mark.asyncio
async def test_doc_status_upsert_preserves_caller_file_path(tmp_path):
    storage = await _storage(tmp_path, namespace=NameSpace.DOC_STATUS)

    await storage.upsert(
        {
            "doc-1": _doc(
                DocStatus.PENDING.value,
                "/tmp/uploads/report.[native-Fi].pdf",
            )
        }
    )

    assert (await storage.get_by_id("doc-1"))["file_path"] == (
        "/tmp/uploads/report.[native-Fi].pdf"
    )
    assert await storage.get_doc_by_file_basename("report.pdf") is None


@pytest.mark.asyncio
async def test_doc_status_basename_lookup_requires_canonical_stored_path(tmp_path):
    storage = await _storage(
        tmp_path,
        {"doc-1": _doc(DocStatus.PROCESSED.value, "report.[native].pdf")},
        namespace=NameSpace.DOC_STATUS,
    )

    assert await storage.get_doc_by_file_basename("report.pdf") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bad_id", "bad_row", "exc_type"),
    [
        pytest.param("doc-bad", {"status": "failed"}, TypeError, id="missing-fields"),
        pytest.param(
            "doc-nostatus", {"file_path": "x.pdf"}, KeyError, id="missing-status-key"
        ),
    ],
)
async def test_get_docs_by_statuses_skips_relaxed_raises_strict(
    tmp_path, bad_id, bad_row, exc_type
):
    storage = await _storage(
        tmp_path,
        {"doc-good": _doc("failed", "good.pdf"), bad_id: bad_row},
    )

    relaxed = await storage.get_docs_by_statuses([DocStatus.FAILED])
    assert set(relaxed) == {"doc-good"}

    with pytest.raises(exc_type):
        await storage.get_docs_by_statuses([DocStatus.FAILED], strict=True)


# ---------------------------------------------------------------------------
# JSON-specific scheduling / cursor / lookup details
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_created_at_sorts_first_and_stays_reachable(tmp_path):
    storage = await _storage(
        tmp_path,
        {
            "doc-real": _doc("pending", created_at="2026-01-01T00:00:00+00:00"),
            "doc-null": _doc("pending", created_at=""),
        },
    )
    assert await _sweep_ids(storage, [DocStatus.PENDING], limit=1) == [
        "doc-null",
        "doc-real",
    ]


@pytest.mark.asyncio
async def test_update_doc_status_fields_missing_row_respects_missing_ok(tmp_path):
    storage = await _storage(tmp_path, {"doc-1": _doc("failed")})
    with pytest.raises(StorageRecordNotFoundError):
        await storage.update_doc_status_fields("missing", {"status": "pending"})
    await storage.update_doc_status_fields(
        "missing", {"status": "pending"}, missing_ok=True
    )
    assert (await storage.get_by_id("doc-1"))["status"] == "failed"


@pytest.mark.asyncio
async def test_get_full_docs_by_ids_strict_raises_on_malformed(tmp_path):
    storage = await _storage(tmp_path, {"doc-1": _doc("pending")})
    async with storage._storage_lock:
        storage._data["bad"] = {"status": "pending"}
    with pytest.raises((KeyError, TypeError)):
        await storage.get_full_docs_by_ids(["doc-1", "bad"], strict=True)
    relaxed = await storage.get_full_docs_by_ids(["doc-1", "bad"], strict=False)
    assert set(relaxed) == {"doc-1"}


@pytest.mark.asyncio
async def test_repair_source_conflict_repeats_converge(tmp_path):
    storage = await _storage(
        tmp_path,
        {
            "doc-1": _doc("pending", file_path="a.pdf"),
            "doc-2": _doc("failed", file_path="a.pdf"),
        },
    )
    first = await storage.repair_source_conflict(
        "a.pdf",
        primary_doc_id="doc-2",
        expected_candidate_count=0,
        expected_candidate_fingerprint="",
    )
    committed = await storage.repair_source_conflict(
        "a.pdf",
        primary_doc_id="doc-2",
        expected_candidate_count=first.candidate_count,
        expected_candidate_fingerprint=first.fingerprint,
        dry_run=False,
    )
    assert committed.committed is True

    with pytest.raises(SourceConflictRepairCASError):
        await storage.repair_source_conflict(
            "a.pdf",
            primary_doc_id="doc-2",
            expected_candidate_count=first.candidate_count,
            expected_candidate_fingerprint=first.fingerprint,
            dry_run=False,
        )

    again = await storage.repair_source_conflict(
        "a.pdf",
        primary_doc_id="doc-2",
        expected_candidate_count=0,
        expected_candidate_fingerprint="",
    )
    assert again.candidate_count == 1
    assert again.demoted_sample_doc_ids == ()
    final = await storage.repair_source_conflict(
        "a.pdf",
        primary_doc_id="doc-2",
        expected_candidate_count=again.candidate_count,
        expected_candidate_fingerprint=again.fingerprint,
        dry_run=False,
    )
    assert final.committed is True
    resolved = await storage.resolve_doc_source_strict("a.pdf")
    assert isinstance(resolved, SourceUnique) and resolved.doc_id == "doc-2"
    row = await storage.get_by_id("doc-1")
    assert row["metadata"]["original_doc_id"] == "doc-2"


@pytest.mark.asyncio
async def test_basename_lookup_returns_primary_only(tmp_path):
    dup = _doc("failed", file_path="a.pdf")
    dup["metadata"] = {"is_duplicate": True, "duplicate_kind": "filename"}
    storage = await _storage(
        tmp_path,
        {"dup-1": dup, "doc-primary": _doc("pending", file_path="a.pdf")},
    )
    match = await storage.get_doc_by_file_basename("a.pdf")
    assert match is not None and match[0] == "doc-primary"
    await storage.delete(["doc-primary"])
    assert await storage.get_doc_by_file_basename("a.pdf") is None


_TWO_DOC_ROWS = {
    "doc-1": _doc("pending", created_at="2026-01-01T00:00:00+00:00"),
    "doc-2": _doc("pending", created_at="2026-02-01T00:00:00+00:00"),
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            json.dumps({"pending": "doc-a", "failed": "doc-b"}), id="two-key-object"
        ),
        pytest.param(json.dumps("ab"), id="two-char-string"),
    ],
)
async def test_cursor_payload_must_be_a_two_element_list(tmp_path, payload):
    storage = await _storage(tmp_path, dict(_TWO_DOC_ROWS))
    with pytest.raises(StorageControlPlaneError):
        await storage.get_docs_by_statuses_page(
            [DocStatus.PENDING],
            limit=1,
            position=CursorAfter(payload),
            strict=True,
        )


@pytest.mark.asyncio
async def test_genuine_cursor_still_resumes_after_consumed_key(tmp_path):
    storage = await _storage(tmp_path, dict(_TWO_DOC_ROWS))
    page = await storage.get_docs_by_statuses_page(
        [DocStatus.PENDING],
        limit=1,
        position=CursorAfter(json.dumps(["2026-01-01T00:00:00+00:00", "doc-1"])),
        strict=True,
    )
    assert list(page.docs) == ["doc-2"]


@pytest.mark.asyncio
async def test_content_hash_lookup_returns_the_earliest_holder(tmp_path):
    storage = await _storage(
        tmp_path,
        {
            "doc-a": _doc(
                "processed",
                file_path="late.pdf",
                created_at="2026-05-05T00:00:00+00:00",
                content_hash="dup",
            ),
            "doc-z": _doc(
                "processed",
                file_path="early.pdf",
                created_at="2024-01-01T00:00:00+00:00",
                content_hash="dup",
            ),
        },
    )

    result = await storage.get_doc_by_content_hash("dup")
    assert result is not None and result[0] == "doc-z"

    result = await storage.get_doc_by_content_hash("dup", exclude_doc_id="doc-z")
    assert result is not None and result[0] == "doc-a"


# ---------------------------------------------------------------------------
# Copy-on-read and paginated snapshot / hydration invariants
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_by_id_returns_copy(tmp_path):
    storage = await _storage(tmp_path)
    await storage.upsert(
        {
            "doc1": {
                "status": "pending",
                "file_path": "test.pdf",
                "metadata": {"key": "original"},
            }
        }
    )

    doc = await storage.get_by_id("doc1")
    assert doc is not None
    doc["status"] = "modified_externally"
    doc["metadata"]["key"] = "modified_nested"
    doc["chunks_list"] = ["chunk_1"]

    doc_after = await storage.get_by_id_strict("doc1")
    assert doc_after is not None
    assert doc_after["status"] == "pending"
    assert doc_after["metadata"]["key"] == "original"
    assert doc_after["chunks_list"] == []


@pytest.mark.asyncio
async def test_get_doc_by_file_path_returns_copy(tmp_path):
    storage = await _storage(tmp_path)
    await storage.upsert(
        {
            "doc2": {
                "status": "pending",
                "file_path": "report.pdf",
                "metadata": {"key": "original"},
            }
        }
    )

    doc = await storage.get_doc_by_file_path("report.pdf")
    assert doc is not None
    doc["status"] = "modified_externally"
    doc["metadata"]["key"] = "modified_nested"

    doc_after = await storage.get_by_id_strict("doc2")
    assert doc_after is not None
    assert doc_after["status"] == "pending"
    assert doc_after["metadata"]["key"] == "original"


@pytest.mark.asyncio
async def test_get_by_ids_returns_deep_copy(tmp_path):
    storage = await _storage(tmp_path)
    await storage.upsert(
        {
            "doc3": {
                "status": "pending",
                "file_path": "data.pdf",
                "metadata": {"kg_purge": {"phase": "prepared"}},
                "chunks_list": ["c1", "c2"],
            }
        }
    )

    docs = await storage.get_by_ids(["doc3"])
    assert len(docs) == 1 and docs[0] is not None
    docs[0]["metadata"]["kg_purge"]["phase"] = "corrupted"
    docs[0]["chunks_list"].append("c3")

    stored = await storage.get_by_id_strict("doc3")
    assert stored is not None
    assert stored["metadata"]["kg_purge"]["phase"] == "prepared"
    assert stored["chunks_list"] == ["c1", "c2"]


@pytest.mark.asyncio
async def test_get_doc_by_file_basename_returns_copy(tmp_path):
    storage = await _storage(tmp_path)
    await storage.upsert(
        {
            "doc4": {
                "status": "pending",
                "file_path": "basename.pdf",
                "metadata": {"key": "original"},
            }
        }
    )

    result = await storage.get_doc_by_file_basename("basename.pdf")
    assert result is not None
    doc_id, doc = result
    assert doc_id == "doc4"
    doc["status"] = "modified_externally"
    doc["metadata"]["key"] = "modified_nested"

    stored = await storage.get_by_id_strict("doc4")
    assert stored is not None
    assert stored["status"] == "pending"
    assert stored["metadata"]["key"] == "original"


@pytest.mark.asyncio
async def test_get_docs_paginated_returns_copy(tmp_path):
    storage = await _storage(tmp_path)
    await storage.upsert(
        {
            "doc5": {
                "status": "pending",
                "file_path": "paginated.pdf",
                "content_summary": "summary",
                "content_length": 100,
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
                "metadata": {"kg_purge": {"phase": "prepared"}},
                "chunks_list": ["c1", "c2"],
            }
        }
    )

    docs, total = await storage.get_docs_paginated()
    assert total == 1
    doc_id, doc_status = docs[0]
    assert doc_id == "doc5"
    assert doc_status.chunks_list == []

    doc_status.metadata["kg_purge"]["phase"] = "corrupted"

    stored = await storage.get_by_id_strict("doc5")
    assert stored is not None
    assert stored["metadata"]["kg_purge"]["phase"] == "prepared"
    assert stored["chunks_list"] == ["c1", "c2"]


@pytest.mark.asyncio
async def test_get_docs_paginated_excludes_rows_missing_required_fields(tmp_path):
    storage = await _storage(tmp_path)
    await storage.upsert(
        {
            "doc7": {
                "status": "pending",
                "file_path": "good.pdf",
                "content_summary": "summary",
                "content_length": 100,
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
            },
            "doc8": {
                "status": "pending",
                "file_path": "malformed.pdf",
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
            },
        }
    )

    docs, total = await storage.get_docs_paginated()
    assert total == 1
    assert [doc_id for doc_id, _ in docs] == ["doc7"]


@pytest.mark.asyncio
async def test_get_docs_paginated_hydrates_rows_with_unexpected_fields(tmp_path):
    storage = await _storage(tmp_path)
    await storage.upsert(
        {
            "doc9": {
                "status": "pending",
                "file_path": "good.pdf",
                "content_summary": "summary",
                "content_length": 100,
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
            },
            "doc10": {
                "status": "pending",
                "file_path": "unexpected_field.pdf",
                "content_summary": "summary",
                "content_length": 100,
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
                "unexpected_field": "x",
            },
        }
    )

    docs, total = await storage.get_docs_paginated()
    assert total == 2
    assert sorted(doc_id for doc_id, _ in docs) == ["doc10", "doc9"]
    hydrated = dict(docs)["doc10"]
    assert not hasattr(hydrated, "unexpected_field")


@pytest.mark.asyncio
async def test_get_docs_paginated_uses_a_single_lock_acquisition(tmp_path):
    storage = await _storage(tmp_path)
    await storage.upsert(
        {
            "doc11": {
                "status": "pending",
                "file_path": "single_lock.pdf",
                "content_summary": "summary",
                "content_length": 100,
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
            }
        }
    )

    real_aenter = NamespaceLock.__aenter__
    acquisitions = []

    async def counting_aenter(self):
        acquisitions.append(1)
        return await real_aenter(self)

    with patch.object(NamespaceLock, "__aenter__", counting_aenter):
        docs, total = await storage.get_docs_paginated()

    assert total == 1
    assert len(acquisitions) == 1


@pytest.mark.asyncio
async def test_get_docs_by_statuses_returns_copy(tmp_path):
    storage = await _storage(tmp_path)
    await storage.upsert(
        {
            "doc6": {
                "status": "pending",
                "file_path": "bystatus.pdf",
                "content_summary": "summary",
                "content_length": 100,
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
                "metadata": {"kg_purge": {"phase": "prepared"}},
                "chunks_list": ["c1", "c2"],
            }
        }
    )

    result = await storage.get_docs_by_statuses([DocStatus.PENDING])
    doc_status = result["doc6"]
    doc_status.metadata["kg_purge"]["phase"] = "corrupted"
    doc_status.chunks_list.append("c3")

    stored = await storage.get_by_id_strict("doc6")
    assert stored is not None
    assert stored["metadata"]["kg_purge"]["phase"] == "prepared"
    assert stored["chunks_list"] == ["c1", "c2"]


# ---------------------------------------------------------------------------
# get_docs_by_track_id hydration resilience
# ---------------------------------------------------------------------------


def _valid_track_row(track_id: str = "track_123") -> dict:
    return {
        "track_id": track_id,
        "status": "processed",
        "content_summary": "Summary of valid doc",
        "content_length": 100,
        "file_path": "valid.txt",
        "created_at": 1000.0,
        "updated_at": 1000.0,
    }


@pytest.mark.asyncio
async def test_get_docs_by_track_id_handles_unhydratable_row_without_crashing(tmp_path):
    storage = await _storage(tmp_path)
    await storage.upsert(
        {
            "valid_doc": _valid_track_row(),
            "invalid_doc": {"track_id": "track_123"},
        }
    )

    docs = await storage.get_docs_by_track_id("track_123")
    assert "valid_doc" in docs
    assert docs["valid_doc"].status == "processed"
    assert "invalid_doc" not in docs


async def test_get_docs_by_track_id_hydrates_row_carrying_an_unknown_field(tmp_path):
    storage = await _storage(tmp_path)
    await storage.upsert(
        {
            "old_doc": _valid_track_row(),
            "written_by_newer_build": {
                **_valid_track_row(),
                "field_from_the_future": "value",
            },
        }
    )

    docs = await storage.get_docs_by_track_id("track_123")
    assert set(docs) == {"old_doc", "written_by_newer_build"}
    assert not hasattr(docs["written_by_newer_build"], "field_from_the_future")


async def test_get_docs_by_track_id_logs_and_skips_a_non_mapping_row(
    tmp_path, lightrag_caplog
):
    storage = await _storage(tmp_path)
    await storage.upsert({"valid_doc": _valid_track_row()})
    storage._data["corrupt_doc"] = "this is not a mapping"

    lightrag_caplog.clear()
    with lightrag_caplog.at_level(logging.ERROR, logger="lightrag"):
        docs = await storage.get_docs_by_track_id("track_123")

    assert "valid_doc" in docs
    assert "corrupt_doc" not in docs
    assert "corrupt_doc" in lightrag_caplog.text
    assert "is not a mapping" in lightrag_caplog.text
