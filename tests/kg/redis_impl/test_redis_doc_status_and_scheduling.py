"""Offline unit tests for RedisDocStatusStorage: lookups, track_id hydration, and scheduling pages.

Runs against the shared in-memory FakeRedis stand-in — no live Redis required.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

pytest.importorskip("redis", reason="redis is required for Redis storage tests")

from redis.exceptions import (  # noqa: E402
    ConnectionError as RedisConnectionError,
    RedisError,
)

from lightrag.base import (  # noqa: E402
    CURSOR_END,
    CURSOR_START,
    CursorAfter,
    DocStatus,
    SourceAbsent,
    SourceConflict,
    SourceUnique,
)
from lightrag.exceptions import (  # noqa: E402
    StorageControlPlaneError,
    StorageRecordNotFoundError,
)
from lightrag.namespace import NameSpace  # noqa: E402

from .fake_redis import FakeRedis  # noqa: E402

pytestmark = pytest.mark.offline


class _DummyEmbeddingFunc:
    embedding_dim = 1
    max_token_size = 1

    async def __call__(self, texts, **kwargs):
        return [[0.0] for _ in texts]


def _doc(
    status: str,
    file_path: str = "a.pdf",
    created_at: str = "2026-01-01T00:00:00+00:00",
    **extra,
) -> dict:
    row = {
        "content_summary": "s",
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


def _dup(status: str, file_path: str = "a.pdf", **extra) -> dict:
    row = _doc(status, file_path=file_path, **extra)
    row["metadata"] = {"is_duplicate": True}
    return row


def _valid_track_row(track_id: str = "track_123") -> dict:
    return {
        "track_id": track_id,
        "status": "processed",
        "content_summary": "summary",
        "content_length": 10,
        "file_path": "report.pdf",
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00",
        "metadata": {},
        "error_msg": None,
    }


@pytest.fixture
def storage(monkeypatch):
    """Construct RedisDocStatusStorage backed by an in-memory FakeRedis."""
    fake = FakeRedis()
    monkeypatch.setattr(
        "lightrag.kg.redis_impl.RedisConnectionManager.get_pool",
        lambda redis_url: MagicMock(name="fake_pool"),
    )
    monkeypatch.setattr(
        "lightrag.kg.redis_impl.RedisConnectionManager.release_pool",
        lambda redis_url: None,
    )
    monkeypatch.setattr(
        "lightrag.kg.redis_impl.Redis", lambda connection_pool=None, **_: fake
    )

    from lightrag.kg.redis_impl import RedisDocStatusStorage

    instance = RedisDocStatusStorage(
        namespace=NameSpace.DOC_STATUS,
        global_config={},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    instance._pool = MagicMock(name="fake_pool")
    instance._redis = fake
    instance._initialized = True
    return instance


def _store_raw(storage, doc_id: str, payload: dict) -> None:
    """Write directly into the fake backing store, bypassing ``upsert``."""
    storage._redis.store[f"{storage.final_namespace}:{doc_id}"] = json.dumps(payload)


async def _bootstrap(storage):
    """Run the derived-sidecar rebuild the way initialize() would."""
    await storage._rebuild_scheduling_sidecar()


def _primary_row_counts(storage) -> dict[str, int]:
    """Reference tally: what a full scan of the primary rows would report."""
    fake = storage._redis
    prefix = f"{storage.final_namespace}:"
    counts = {status.value: 0 for status in DocStatus}
    for key, raw in fake.store.items():
        if key.startswith(prefix):
            counts[json.loads(raw)["status"]] += 1
    return counts


async def _sweep_ids(storage, statuses, *, limit):
    ids: list[str] = []
    pages = 0
    position = CURSOR_START
    while True:
        page = await storage.get_docs_by_statuses_page(
            statuses,
            limit=limit,
            position=position,
            strict=True,
        )
        pages += 1
        ids.extend(page.docs.keys())
        if page.next_position is CURSOR_END:
            return ids, pages
        position = page.next_position
        assert isinstance(position, CursorAfter)
        assert pages < 100, "sweep failed to terminate"


# ---------------------------------------------------------------------------
# Basename, content_hash, and get_by_id lookups
# ---------------------------------------------------------------------------


class TestRedisDocStatusLookup:
    async def test_get_by_id_reraises_redis_error_after_retry_limit(
        self, storage, monkeypatch
    ):
        from lightrag.kg.redis_impl import RETRY_ATTEMPTS, RedisDocStatusStorage

        attempts = 0

        async def failing_get(_key):
            nonlocal attempts
            attempts += 1
            raise RedisConnectionError("connection lost")

        async def no_sleep(_delay):
            pass

        monkeypatch.setattr(storage._redis, "get", failing_get)
        monkeypatch.setattr(RedisDocStatusStorage.get_by_id.retry, "sleep", no_sleep)

        with pytest.raises(RedisConnectionError):
            await storage.get_by_id("doc-1")

        assert attempts == RETRY_ATTEMPTS

    async def test_get_doc_by_file_basename_returns_tuple_on_hit(self, storage):
        await storage.upsert({"doc-1": _doc(DocStatus.PROCESSED.value, "report.pdf")})

        result = await storage.get_doc_by_file_basename("report.pdf")

        assert result is not None
        doc_id, doc_data = result
        assert doc_id == "doc-1"
        assert doc_data["file_path"] == "report.pdf"

    @pytest.mark.parametrize(
        ("stored_path", "query_basename"),
        [
            pytest.param("report.pdf", "other.pdf", id="not_present"),
            pytest.param("report.pdf", "", id="empty_query"),
            pytest.param(
                "unknown_source", "unknown_source", id="unknown_source_sentinel"
            ),
        ],
    )
    async def test_get_doc_by_file_basename_returns_none(
        self, storage, stored_path, query_basename
    ):
        _store_raw(storage, "doc-1", _doc(DocStatus.PROCESSED.value, stored_path))

        assert await storage.get_doc_by_file_basename(query_basename) is None

    async def test_get_doc_by_content_hash_returns_tuple_on_hit(self, storage):
        _store_raw(
            storage,
            "doc-1",
            _doc(DocStatus.PROCESSED.value, "report.pdf", content_hash="abc123"),
        )

        result = await storage.get_doc_by_content_hash("abc123")

        assert result is not None
        doc_id, doc_data = result
        assert doc_id == "doc-1"
        assert doc_data["content_hash"] == "abc123"

    @pytest.mark.parametrize(
        ("stored_kwargs", "query_hash"),
        [
            pytest.param({"content_hash": "abc123"}, "zzz999", id="not_present"),
            pytest.param({}, "", id="empty_query_with_legacy_row"),
        ],
    )
    async def test_get_doc_by_content_hash_returns_none(
        self, storage, stored_kwargs, query_hash
    ):
        _store_raw(
            storage,
            "doc-1",
            _doc(DocStatus.PROCESSED.value, "report.pdf", **stored_kwargs),
        )

        assert await storage.get_doc_by_content_hash(query_hash) is None

    async def test_get_doc_by_content_hash_ignores_legacy_rows(self, storage):
        _store_raw(storage, "doc-legacy", _doc(DocStatus.PROCESSED.value, "old.pdf"))
        _store_raw(
            storage,
            "doc-new",
            _doc(DocStatus.PROCESSED.value, "new.pdf", content_hash="abc123"),
        )

        result = await storage.get_doc_by_content_hash("abc123")

        assert result is not None
        assert result[0] == "doc-new"

    async def test_get_doc_by_content_hash_exclude_doc_id_skips_self(self, storage):
        _store_raw(
            storage,
            "doc-a",
            _doc(DocStatus.PROCESSED.value, "a.pdf", content_hash="dup"),
        )
        _store_raw(
            storage,
            "doc-b",
            _doc(DocStatus.PROCESSED.value, "b.pdf", content_hash="dup"),
        )
        _store_raw(
            storage,
            "doc-solo",
            _doc(DocStatus.PROCESSED.value, "s.pdf", content_hash="uniq"),
        )
        result = await storage.get_doc_by_content_hash("dup", exclude_doc_id="doc-a")
        assert result is not None and result[0] == "doc-b"
        result = await storage.get_doc_by_content_hash("dup", exclude_doc_id="doc-b")
        assert result is not None and result[0] == "doc-a"
        assert (
            await storage.get_doc_by_content_hash("uniq", exclude_doc_id="doc-solo")
            is None
        )

    async def test_get_doc_by_content_hash_returns_the_earliest_holder(self, storage):
        """The base contract asks for the earliest holder by (created_at, id)."""
        late = _doc(
            DocStatus.PROCESSED.value,
            "late.pdf",
            created_at="2026-05-05T00:00:00+00:00",
            content_hash="dup",
        )
        early = _doc(
            DocStatus.PROCESSED.value,
            "early.pdf",
            created_at="2024-01-01T00:00:00+00:00",
            content_hash="dup",
        )
        # Doc IDs sort opposite to created_at so scan order cannot mask a bug.
        _store_raw(storage, "doc-a", late)
        _store_raw(storage, "doc-z", early)

        result = await storage.get_doc_by_content_hash("dup")

        assert result is not None
        assert result[0] == "doc-z"

    async def test_get_doc_by_content_hash_propagates_transport_failure(self, storage):
        """SCAN error must propagate rather than returning None (false miss)."""
        _store_raw(
            storage,
            "doc-1",
            _doc(DocStatus.PROCESSED.value, "a.pdf", content_hash="dup"),
        )
        storage._redis.fail_next["scan"] = RedisError("boom")

        with pytest.raises(RedisError):
            await storage.get_doc_by_content_hash("dup")

    async def test_get_doc_by_content_hash_refuses_to_skip_an_undecodable_row(
        self, storage
    ):
        """An undecodable row might be the holder; fail closed instead of skipping."""
        storage._redis.store[f"{storage.final_namespace}:doc-bad"] = "not-json{"

        with pytest.raises(StorageControlPlaneError):
            await storage.get_doc_by_content_hash("dup")

    async def test_get_doc_by_content_hash_skips_pointer_rows_and_keeps_looking(
        self, storage
    ):
        """Duplicate pointers referencing the excluded doc are skipped without
        stopping the scan for genuine surviving holders."""
        pointer = _doc(
            DocStatus.FAILED.value,
            "b.pdf",
            created_at="2024-01-01T00:00:00+00:00",
            content_hash="dup",
        )
        pointer["metadata"] = {"is_duplicate": True, "original_doc_id": "doc-asking"}
        _store_raw(storage, "doc-pointer", pointer)

        asking = _doc(
            DocStatus.PROCESSED.value,
            "a.pdf",
            created_at="2024-02-01T00:00:00+00:00",
            content_hash="dup",
        )
        _store_raw(storage, "doc-asking", asking)

        third = _doc(
            DocStatus.PROCESSED.value,
            "c.pdf",
            created_at="2024-03-01T00:00:00+00:00",
            content_hash="dup",
        )
        _store_raw(storage, "doc-third", third)

        result = await storage.get_doc_by_content_hash(
            "dup", exclude_doc_id="doc-asking"
        )
        assert result is not None and result[0] == "doc-third"

        await storage.delete(["doc-third"])
        assert (
            await storage.get_doc_by_content_hash("dup", exclude_doc_id="doc-asking")
            is None
        )


# ---------------------------------------------------------------------------
# get_docs_by_track_id hydration resilience
# ---------------------------------------------------------------------------


class TestRedisTrackIdHydration:
    async def test_track_id_listing_survives_a_row_missing_required_fields(
        self, storage
    ):
        _store_raw(storage, "valid_doc", _valid_track_row())
        _store_raw(storage, "invalid_doc", {"track_id": "track_123"})

        docs = await storage.get_docs_by_track_id("track_123")

        assert "valid_doc" in docs
        assert "invalid_doc" not in docs

    async def test_track_id_listing_hydrates_a_row_with_an_unknown_field(self, storage):
        """Undeclared fields from newer builds are ignored by DocProcessingStatus.from_stored."""
        _store_raw(
            storage,
            "doc-a-written-by-newer-build",
            {**_valid_track_row(), "field_from_the_future": "value"},
        )
        _store_raw(storage, "doc-b-old", _valid_track_row())

        docs = await storage.get_docs_by_track_id("track_123")

        assert set(docs) == {"doc-a-written-by-newer-build", "doc-b-old"}
        assert not hasattr(
            docs["doc-a-written-by-newer-build"], "field_from_the_future"
        )

    async def test_a_bad_row_does_not_truncate_the_rest_of_the_scan(self, storage):
        """An unhydratable row scanned first must not abort the SCAN loop."""
        _store_raw(storage, "doc-000-corrupt", {"track_id": "track_123"})
        for i in range(12):
            _store_raw(storage, f"doc-1{i:02d}", _valid_track_row())

        docs = await storage.get_docs_by_track_id("track_123")

        assert set(docs) == {f"doc-1{i:02d}" for i in range(12)}

    async def test_rows_under_other_track_ids_are_not_returned(self, storage):
        _store_raw(storage, "mine", _valid_track_row("track_123"))
        _store_raw(storage, "theirs", _valid_track_row("track_999"))

        docs = await storage.get_docs_by_track_id("track_123")

        assert set(docs) == {"mine"}


# ---------------------------------------------------------------------------
# Phase 1 scheduling sidecar, pagination, source resolution, and conflicts
# ---------------------------------------------------------------------------


class TestRedisSchedulingPages:
    @pytest.mark.asyncio
    async def test_deleting_moves_status_index_and_remains_filterable(self, storage):
        await _bootstrap(storage)
        await storage.upsert({"doc-1": _doc("processed")})
        await storage.update_doc_status_fields("doc-1", {"status": DocStatus.DELETING})
        assert await storage.count_docs_by_statuses([DocStatus.PROCESSED]) == 0
        assert await storage.count_docs_by_statuses([DocStatus.DELETING]) == 1
        page = await storage.get_docs_by_statuses_page([DocStatus.DELETING], limit=10)
        assert set(page.docs) == {"doc-1"}
        assert page.docs["doc-1"].status == DocStatus.DELETING
        counts = await storage.get_all_status_counts()
        assert counts["deleting"] == 1

    @pytest.mark.asyncio
    async def test_status_counts_come_from_sidecar_without_reading_rows(
        self, storage, monkeypatch
    ):
        await _bootstrap(storage)
        await storage.upsert(
            {
                "p1": _doc("processed", file_path="p1.pdf"),
                "p2": _doc("processed", file_path="p2.pdf"),
                "f1": _doc("failed", file_path="f1.pdf"),
                "q1": _doc("pending", file_path="q1.pdf"),
                "dup": _dup("failed", file_path="p1.pdf"),
            }
        )
        await storage.update_doc_status_fields("p2", {"status": DocStatus.DELETING})
        await storage.delete(["q1"])

        fake = storage._redis
        scanned: list[tuple] = []

        async def no_scan(*args, **kwargs):
            scanned.append((args, kwargs))
            raise AssertionError("status counts must not SCAN the primary rows")

        monkeypatch.setattr(fake, "scan", no_scan)
        gets_before = fake.command_counts["get"]

        counts = await storage.get_all_status_counts()

        assert scanned == []
        assert fake.command_counts["get"] == gets_before
        expected = _primary_row_counts(storage)
        assert expected["processed"] == 1 and expected["deleting"] == 1
        assert expected["failed"] == 2 and expected["pending"] == 0
        assert counts == {**expected, "all": 4}

    @pytest.mark.asyncio
    async def test_status_counts_propagate_redis_errors(self, storage):
        await _bootstrap(storage)
        await storage.upsert({"p1": _doc("processed")})
        storage._redis.fail_next["execute"] = RedisError("connection lost")
        with pytest.raises(RedisError, match="connection lost"):
            await storage.get_all_status_counts()

    @pytest.mark.asyncio
    async def test_rebuild_builds_sidecar_from_primary_rows(self, storage):
        fake = storage._redis
        fake.store[f"{storage.final_namespace}:doc-1"] = json.dumps(
            _doc("pending", file_path="a.pdf")
        )
        fake.store[f"{storage.final_namespace}:doc-2"] = json.dumps(
            _doc("failed", file_path="b.pdf")
        )
        fake.store[f"{storage.final_namespace}:dup-1"] = json.dumps(
            _dup("failed", file_path="a.pdf")
        )

        await _bootstrap(storage)

        assert await storage.count_docs_by_statuses([DocStatus.PENDING]) == 1
        assert await storage.count_docs_by_statuses([DocStatus.FAILED]) == 2
        resolved = await storage.resolve_doc_source_strict("a.pdf")
        assert isinstance(resolved, SourceUnique) and resolved.doc_id == "doc-1"
        assert not any(
            k.startswith(f"{storage._sched_prefix}_rebuild:") for k in fake.zsets
        )
        assert not any(
            k.startswith(f"{storage._sched_prefix}_rebuild:") for k in fake.sets
        )
        await _bootstrap(storage)
        assert await storage.count_docs_by_statuses([DocStatus.PENDING]) == 1

    @pytest.mark.asyncio
    async def test_publish_refuses_to_clobber_a_concurrent_writer(
        self, storage, monkeypatch
    ):
        """Publish must refuse instead of reverting a concurrent write."""
        fake = storage._redis
        fake.store[f"{storage.final_namespace}:doc-1"] = json.dumps(
            _doc("pending", file_path="a.pdf")
        )

        primary_pattern = f"{storage.final_namespace}:*"
        real_scan = fake.scan
        primary_scans = {"n": 0}

        async def scan_with_racing_write(
            cursor: int = 0, match: str = "", count: int = 1000
        ):
            result = await real_scan(cursor, match=match, count=count)
            if match == primary_pattern:
                primary_scans["n"] += 1
                if primary_scans["n"] == 2:
                    await storage.upsert(
                        {"doc-new": _doc("pending", file_path="c.pdf")}
                    )
            return result

        monkeypatch.setattr(fake, "scan", scan_with_racing_write)

        with pytest.raises(StorageControlPlaneError, match="stale snapshot"):
            await storage._rebuild_scheduling_sidecar()

        pending_members = fake.zsets[f"{storage._sched_prefix}:status:pending"]
        assert any("doc-new" in member for member in pending_members)

    @pytest.mark.asyncio
    async def test_sidecar_not_ready_raises_never_empty(self, storage):
        storage._initialized = False
        with pytest.raises(StorageControlPlaneError):
            await storage.get_status_counts()
        with pytest.raises(StorageControlPlaneError):
            await storage.count_docs_by_statuses([DocStatus.PENDING])
        with pytest.raises(StorageControlPlaneError):
            await storage.get_docs_by_statuses_page([DocStatus.PENDING], limit=1)
        with pytest.raises(StorageControlPlaneError):
            await storage.resolve_doc_source_strict("a.pdf")

    @pytest.mark.asyncio
    async def test_page_kway_merge_and_consumed_position(self, storage):
        await _bootstrap(storage)
        await storage.upsert(
            {
                "doc-c": _doc("pending", created_at="2026-01-01T00:00:00+00:00"),
                "doc-a": _doc("failed", created_at="2026-01-02T00:00:00+00:00"),
                "doc-b": _doc("pending", created_at="2026-01-03T00:00:00+00:00"),
            }
        )
        ids, pages = await _sweep_ids(
            storage, [DocStatus.PENDING, DocStatus.FAILED], limit=1
        )
        assert ids == ["doc-c", "doc-a", "doc-b"]
        assert pages >= 3

    @pytest.mark.asyncio
    async def test_page_prefetched_head_not_consumed(self, storage):
        await _bootstrap(storage)
        await storage.upsert(
            {
                "doc-p": _doc("pending", created_at="2026-01-01T00:00:00+00:00"),
                "doc-f": _doc("failed", created_at="2026-01-01T00:00:00+00:00"),
            }
        )
        page1 = await storage.get_docs_by_statuses_page(
            [DocStatus.PENDING, DocStatus.FAILED], limit=1, strict=True
        )
        assert list(page1.docs) == ["doc-f"]
        page2 = await storage.get_docs_by_statuses_page(
            [DocStatus.PENDING, DocStatus.FAILED],
            limit=1,
            position=page1.next_position,
            strict=True,
        )
        assert list(page2.docs) == ["doc-p"]

    @pytest.mark.asyncio
    async def test_status_transition_moves_zset_membership(self, storage):
        await _bootstrap(storage)
        await storage.upsert({"doc-1": _doc("pending")})
        await storage.update_doc_status_fields("doc-1", {"status": "processing"})
        assert await storage.count_docs_by_statuses([DocStatus.PENDING]) == 0
        assert await storage.count_docs_by_statuses([DocStatus.PROCESSING]) == 1
        ids, _ = await _sweep_ids(storage, [DocStatus.PROCESSING], limit=10)
        assert ids == ["doc-1"]
        with pytest.raises(ValueError, match="created_at"):
            await storage.update_doc_status_fields(
                "doc-1", {"created_at": "2030-01-01"}
            )
        with pytest.raises(StorageRecordNotFoundError):
            await storage.update_doc_status_fields("missing", {"status": "pending"})

    @pytest.mark.asyncio
    async def test_upsert_with_a_live_docstatus_enum_is_discoverable(self, storage):
        """Live DocStatus enum values must key ZSETs by .value, not Enum.__str__."""
        await _bootstrap(storage)
        await storage.upsert({"doc-1": _doc(DocStatus.PENDING, file_path="a.pdf")})

        assert (await storage.get_by_id("doc-1"))["status"] == "pending"
        assert await storage.count_docs_by_statuses([DocStatus.PENDING]) == 1
        ids, _ = await _sweep_ids(storage, [DocStatus.PENDING], limit=10)
        assert ids == ["doc-1"]

        await storage.update_doc_status_fields(
            "doc-1", {"status": DocStatus.PROCESSING}
        )
        assert await storage.count_docs_by_statuses([DocStatus.PENDING]) == 0
        assert await storage.count_docs_by_statuses([DocStatus.PROCESSING]) == 1

    @pytest.mark.asyncio
    async def test_atomic_write_retries_on_watch_conflict(self, storage):
        await _bootstrap(storage)
        await storage.upsert({"doc-1": _doc("pending")})
        fake = storage._redis
        main_key = f"{storage.final_namespace}:doc-1"

        original = FakeRedis.get
        tripped = {"done": False}

        async def get_with_interleaving(self, key):
            result = await original(self, key)
            if not tripped["done"] and key == main_key:
                tripped["done"] = True
                self._bump(key)
            return result

        fake.get = get_with_interleaving.__get__(fake)
        try:
            await storage.update_doc_status_fields("doc-1", {"status": "processing"})
        finally:
            fake.get = original.__get__(fake)
        assert tripped["done"] is True
        assert (await storage.get_by_id("doc-1"))["status"] == "processing"
        assert await storage.count_docs_by_statuses([DocStatus.PROCESSING]) == 1
        assert await storage.count_docs_by_statuses([DocStatus.PENDING]) == 0

    @pytest.mark.asyncio
    async def test_atomic_write_transport_failure_propagates(self, storage):
        await _bootstrap(storage)
        await storage.upsert({"doc-1": _doc("pending")})
        fake = storage._redis
        fake.fail_next["execute"] = RedisError("boom")
        with pytest.raises(RedisError):
            await storage.update_doc_status_fields("doc-1", {"status": "processing"})
        assert (await storage.get_by_id("doc-1"))["status"] == "pending"

    @pytest.mark.asyncio
    async def test_source_multimap_eligibility_state_machine(self, storage):
        await _bootstrap(storage)
        await storage.upsert({"doc-1": _doc("pending", file_path="a.pdf")})
        resolved = await storage.resolve_doc_source_strict("a.pdf")
        assert isinstance(resolved, SourceUnique) and resolved.doc_id == "doc-1"

        await storage.upsert(
            {"doc-1": _dup("failed", file_path="a.pdf", duplicate_kind="content_hash")}
        )
        assert isinstance(
            await storage.resolve_doc_source_strict("a.pdf"), SourceAbsent
        )

        await storage.upsert({"doc-2": _doc("pending", file_path="a.pdf")})
        resolved = await storage.resolve_doc_source_strict("a.pdf")
        assert isinstance(resolved, SourceUnique) and resolved.doc_id == "doc-2"

        await storage.delete(["doc-1"])
        resolved = await storage.resolve_doc_source_strict("a.pdf")
        assert isinstance(resolved, SourceUnique) and resolved.doc_id == "doc-2"

        await storage.delete(["doc-2"])
        assert isinstance(
            await storage.resolve_doc_source_strict("a.pdf"), SourceAbsent
        )

    @pytest.mark.asyncio
    async def test_resolve_source_conflict(self, storage):
        await _bootstrap(storage)
        await storage.upsert(
            {
                "doc-1": _doc("pending", file_path="a.pdf"),
                "doc-2": _doc("failed", file_path="a.pdf"),
            }
        )
        resolved = await storage.resolve_doc_source_strict("a.pdf")
        assert isinstance(resolved, SourceConflict)
        assert resolved.candidate_count is None
        assert set(resolved.sample_doc_ids) == {"doc-1", "doc-2"}

    @pytest.mark.asyncio
    async def test_resolve_source_stale_member_self_heals(self, storage):
        await _bootstrap(storage)
        await storage.upsert({"doc-1": _doc("pending", file_path="a.pdf")})
        fake = storage._redis
        set_key = storage._basename_key("a.pdf")
        fake.sets[set_key].add("gone")

        resolved = await storage.resolve_doc_source_strict("a.pdf")
        assert isinstance(resolved, SourceUnique) and resolved.doc_id == "doc-1"
        assert "gone" not in fake.sets.get(set_key, set())

    @pytest.mark.asyncio
    async def test_resolve_source_stops_after_two_valid_candidates(self, storage):
        await _bootstrap(storage)
        await storage.upsert(
            {
                "doc-a": _doc("pending", file_path="a.pdf"),
                "doc-b": _doc("failed", file_path="a.pdf"),
            }
        )
        fake = storage._redis
        set_key = storage._basename_key("a.pdf")
        fake.sets[set_key].add("zzz-stale")

        resolved = await storage.resolve_doc_source_strict("a.pdf")
        assert isinstance(resolved, SourceConflict)
        assert set(resolved.sample_doc_ids) == {"doc-a", "doc-b"}
        assert "zzz-stale" in fake.sets.get(set_key, set())

    @pytest.mark.asyncio
    async def test_get_docs_by_ids_present_and_missing(self, storage):
        await _bootstrap(storage)
        await storage.upsert({"doc-1": _doc("pending"), "doc-2": _doc("failed")})
        result = await storage.get_docs_by_ids(["doc-1", "doc-2", "ghost"], strict=True)
        assert set(result) == {"doc-1", "doc-2"}
        assert result["doc-2"].status is DocStatus.FAILED
        assert not hasattr(result["doc-1"], "chunks_list")

    @pytest.mark.asyncio
    async def test_conflict_listing_is_bounded_and_resumable(
        self, storage, monkeypatch
    ):
        await _bootstrap(storage)
        rows = {}
        for i in range(40):
            rows[f"solo-{i:03d}"] = _doc("pending", file_path=f"solo-{i:03d}.pdf")
        for i in range(4):
            rows[f"conf-{i}-a"] = _doc("pending", file_path=f"dup-{i}.pdf")
            rows[f"conf-{i}-b"] = _doc("failed", file_path=f"dup-{i}.pdf")
        await storage.upsert(rows)

        validated: list[str] = []
        real_valid = storage._valid_primary_ids

        async def counting_valid_primary_ids(redis, canonical):
            validated.append(canonical)
            return await real_valid(redis, canonical)

        monkeypatch.setattr(storage, "_valid_primary_ids", counting_valid_primary_ids)
        monkeypatch.setattr(type(storage), "_CONFLICT_SCAN_BATCH", 8)

        page = await storage.list_source_conflicts_page(limit=1)
        assert len(page.conflicts) >= 1
        assert isinstance(page.next_position, CursorAfter)
        assert len(validated) <= 8
        assert all(name.startswith("dup-") for name in validated)

        seen = {c.canonical_source_key for c in page.conflicts}
        position = page.next_position
        for _ in range(20):
            nxt = await storage.list_source_conflicts_page(limit=1, position=position)
            seen.update(c.canonical_source_key for c in nxt.conflicts)
            if nxt.next_position is CURSOR_END:
                break
            position = nxt.next_position
        else:
            raise AssertionError("conflict paging failed to terminate")
        assert seen == {f"dup-{i}.pdf" for i in range(4)}

    @pytest.mark.asyncio
    async def test_get_full_docs_by_ids_present_and_missing(self, storage):
        await _bootstrap(storage)
        await storage.upsert({"doc-1": _doc("pending"), "doc-2": _doc("failed")})
        result = await storage.get_full_docs_by_ids(
            ["doc-1", "doc-2", "ghost"], strict=True
        )
        assert set(result) == {"doc-1", "doc-2"}
        assert result["doc-2"].status == DocStatus.FAILED
        assert result["doc-1"].content_summary == "s"
        assert result["doc-1"].content_length == 10
        assert result["doc-1"].chunks_list == []

    @pytest.mark.asyncio
    async def test_get_full_docs_by_ids_strict_transport_error_raises(self, storage):
        await _bootstrap(storage)
        await storage.upsert({"doc-1": _doc("pending")})
        storage._redis.fail_next["execute"] = RedisError("boom")
        with pytest.raises(RedisError):
            await storage.get_full_docs_by_ids(["doc-1"], strict=True)

    @pytest.mark.asyncio
    async def test_get_full_docs_by_ids_relaxed_skips_undecodable_row(self, storage):
        await _bootstrap(storage)
        await storage.upsert({"doc-1": _doc("pending")})
        storage._redis.store[f"{storage.final_namespace}:bad"] = "not-json{"
        result = await storage.get_full_docs_by_ids(["doc-1", "bad"], strict=False)
        assert set(result) == {"doc-1"}
        assert result["doc-1"].content_summary == "s"

    @pytest.mark.asyncio
    async def test_list_and_repair_source_conflict(self, storage):
        await _bootstrap(storage)
        await storage.upsert(
            {
                "doc-1": _doc("pending", file_path="a.pdf"),
                "doc-2": _doc("failed", file_path="a.pdf"),
                "doc-3": _doc("pending", file_path="a.pdf"),
                "solo": _doc("pending", file_path="b.pdf"),
            }
        )
        page = await storage.list_source_conflicts_page(limit=10)
        assert len(page.conflicts) == 1
        conflict = page.conflicts[0]
        assert conflict.canonical_source_key == "a.pdf"
        assert conflict.candidate_count == 3
        assert set(conflict.sample_doc_ids) == {"doc-1", "doc-2", "doc-3"}

        dry = await storage.repair_source_conflict(
            "a.pdf",
            primary_doc_id="doc-2",
            expected_candidate_count=0,
            expected_candidate_fingerprint="ignored-in-dry-run",
        )
        assert dry.committed is False
        assert dry.candidate_count == 3
        assert isinstance(
            await storage.resolve_doc_source_strict("a.pdf"), SourceConflict
        )

        with pytest.raises(ValueError):
            await storage.repair_source_conflict(
                "a.pdf",
                primary_doc_id="not-a-candidate",
                expected_candidate_count=dry.candidate_count,
                expected_candidate_fingerprint=dry.fingerprint,
                dry_run=False,
            )

        with pytest.raises(StorageControlPlaneError):
            await storage.repair_source_conflict(
                "a.pdf",
                primary_doc_id="doc-2",
                expected_candidate_count=99,
                expected_candidate_fingerprint=dry.fingerprint,
                dry_run=False,
            )

        result = await storage.repair_source_conflict(
            "a.pdf",
            primary_doc_id="doc-2",
            expected_candidate_count=dry.candidate_count,
            expected_candidate_fingerprint=dry.fingerprint,
            dry_run=False,
        )
        assert result.committed is True
        resolved = await storage.resolve_doc_source_strict("a.pdf")
        assert isinstance(resolved, SourceUnique) and resolved.doc_id == "doc-2"
        for loser in ("doc-1", "doc-3"):
            row = await storage.get_by_id(loser)
            assert row["metadata"]["is_duplicate"] is True
            assert row["metadata"]["original_doc_id"] == "doc-2"
        page = await storage.list_source_conflicts_page(limit=10)
        assert page.conflicts == ()

    @pytest.mark.asyncio
    async def test_count_and_strict_lookup_fail_closed(self, storage):
        await _bootstrap(storage)
        await storage.upsert({"doc-1": _doc("pending")})
        fake = storage._redis

        fake.fail_next["zcard"] = RedisError("boom")
        with pytest.raises(RedisError):
            await storage.count_docs_by_statuses([DocStatus.PENDING])

        fake.fail_next["sscan"] = RedisError("boom")
        with pytest.raises(RedisError):
            await storage.resolve_doc_source_strict("a.pdf")

        fake.fail_next["sscan"] = RedisError("boom")
        assert await storage.get_doc_by_file_basename("a.pdf") is None

    @pytest.mark.asyncio
    async def test_malformed_cursor_raises_control_plane_error(self, storage):
        await _bootstrap(storage)
        with pytest.raises(StorageControlPlaneError):
            await storage.get_docs_by_statuses_page(
                [DocStatus.PENDING], limit=1, position=CursorAfter("[1,2]")
            )

    @pytest.mark.asyncio
    async def test_drop_clears_rows_and_sidecar(self, storage):
        await _bootstrap(storage)
        await storage.upsert({"doc-1": _doc("pending", file_path="a.pdf")})
        await storage.drop()
        assert await storage.count_docs_by_statuses([DocStatus.PENDING]) == 0
        assert isinstance(
            await storage.resolve_doc_source_strict("a.pdf"), SourceAbsent
        )

    @pytest.mark.asyncio
    async def test_the_rebuild_switch_is_bounded_not_one_transaction_per_workspace(
        self, storage, monkeypatch
    ):
        fake = storage._redis
        total = 5 * storage._PUBLISH_BATCH
        for index in range(total):
            fake.store[f"{storage.final_namespace}:doc-{index:05d}"] = json.dumps(
                _doc("pending", file_path=f"f{index}.pdf")
            )

        widest = {"renames": 0}
        real_pipeline = fake.pipeline

        def counting_pipeline(transaction: bool = True):
            pipe = real_pipeline(transaction=transaction)
            real_execute = pipe.execute

            async def _execute(raise_on_error: bool = True):
                renames = sum(1 for op in pipe._ops if op[0] == "rename")
                widest["renames"] = max(widest["renames"], renames)
                return await real_execute(raise_on_error=raise_on_error)

            pipe.execute = _execute
            return pipe

        monkeypatch.setattr(fake, "pipeline", counting_pipeline)

        await storage._rebuild_scheduling_sidecar()

        assert 0 < widest["renames"] <= storage._PUBLISH_BATCH, (
            f"a single round trip queued {widest['renames']} renames for {total} "
            "docs; the switch is not bounded"
        )
        members = fake.zsets[f"{storage._sched_prefix}:status:pending"]
        assert len(members) == total
        assert len(fake.sets) == total
        assert not [k for k in fake.zsets if "_rebuild:" in k]
        assert not [k for k in fake.sets if "_rebuild:" in k]

    @pytest.mark.asyncio
    async def test_publish_survives_a_duplicate_scan_return(self, storage, monkeypatch):
        fake = storage._redis
        monkeypatch.setattr(storage, "_PUBLISH_BATCH", 1)
        for index in range(3):
            fake.store[f"{storage.final_namespace}:doc-{index}"] = json.dumps(
                _doc("pending", file_path=f"f{index}.pdf")
            )

        real_scan = fake.scan
        basename_pattern = f"{storage._sched_prefix}_rebuild:basename:*"
        state = {"first_key": None, "injected": False}

        async def duplicating_scan(cursor=0, match="", count=1000):
            next_cursor, keys = await real_scan(cursor, match=match, count=count)
            if match == basename_pattern:
                if cursor == 0 and keys and state["first_key"] is None:
                    state["first_key"] = keys[0]
                elif (
                    cursor != 0
                    and keys
                    and not state["injected"]
                    and state["first_key"] is not None
                ):
                    state["injected"] = True
                    keys = [state["first_key"], *keys]
            return next_cursor, keys

        monkeypatch.setattr(fake, "scan", duplicating_scan)

        await storage._rebuild_scheduling_sidecar()

        assert not [k for k in fake.zsets if "_rebuild:" in k]
        assert not [k for k in fake.sets if "_rebuild:" in k]
        assert len(fake.sets) == 3
        assert len(fake.zsets[f"{storage._sched_prefix}:status:pending"]) == 3

    @pytest.mark.asyncio
    async def test_a_half_published_index_is_rebuilt_not_trusted(self, storage):
        fake = storage._redis
        for index in range(3):
            fake.store[f"{storage.final_namespace}:doc-{index}"] = json.dumps(
                _doc("pending", file_path=f"f{index}.pdf")
            )
        fake.zsets[f"{storage._sched_prefix}:status:pending"] = {"x|doc-0"}
        fake.zsets[f"{storage._sched_prefix}_rebuild:status:pending"] = {"x|doc-1"}

        await storage._rebuild_scheduling_sidecar()

        members = fake.zsets[f"{storage._sched_prefix}:status:pending"]
        assert len(members) == 3
        assert not [k for k in fake.zsets if "_rebuild:" in k]
