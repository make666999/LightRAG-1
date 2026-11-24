"""Scheduling base defaults, backend fixture workspace isolation, and strict/relaxed parse branches.

Covers:
- ``DocStatusStorage`` mandatory ``@abstractmethod`` surface and fail-closed
  concrete defaults (``count_docs_by_statuses``, ``get_by_id_strict``,
  ``update_doc_status_fields``, ``list_source_conflicts_page``,
  ``repair_source_conflict``), plus lightweight ``DocSchedulingRecord`` fields.
- ``_workspace_env_isolated`` / ``_unique_workspace`` fixture isolation against
  ``*_WORKSPACE`` environment overrides.
- Per-record relaxed-skip vs. strict-raise parse branches for Mongo, Redis, and
  PostgreSQL ``get_docs_by_statuses`` using in-process fakes.
"""

from __future__ import annotations

import asyncio
import json
import os
from abc import abstractmethod
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from lightrag.base import (
    CURSOR_END,
    DocProcessingStatus,
    DocSchedulingRecord,
    DocStatus,
    DocStatusPage,
    DocStatusStorage,
    SourceAbsent,
    SourceResolution,
)
from lightrag.exceptions import (
    StorageCapabilityError,
    StorageRecordNotFoundError,
)
from lightrag.kg.mongo_impl import MongoDocStatusStorage
from lightrag.kg.postgres_impl import PGDocStatusStorage
from lightrag.kg.redis_impl import RedisDocStatusStorage

from .conftest_scheduling_backends import _unique_workspace, _workspace_env_isolated

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# DocStatusStorage base defaults
# ---------------------------------------------------------------------------


@dataclass
class _MinimalDocStatusStorage(DocStatusStorage):
    """Minimal concrete subclass exercising inherited base defaults."""

    embedding_func: Any = None
    namespace: str = "test"
    workspace: str = "test"
    global_config: dict = field(default_factory=dict)
    data: dict[str, dict[str, Any]] = field(default_factory=dict)

    async def initialize(self):  # pragma: no cover
        pass

    async def finalize(self):  # pragma: no cover
        pass

    async def index_done_callback(self) -> None:
        pass

    async def drop(self) -> dict[str, str]:
        self.data.clear()
        return {"status": "success", "message": "dropped"}

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        row = self.data.get(id)
        return dict(row) if row is not None else None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        return [dict(self.data[i]) for i in ids if i in self.data]

    async def filter_keys(self, keys: set[str]) -> set[str]:
        return {k for k in keys if k not in self.data}

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        for key, value in data.items():
            self.data[key] = dict(value)

    async def delete(self, ids: list[str]) -> None:
        for i in ids:
            self.data.pop(i, None)

    async def is_empty(self) -> bool:
        return not self.data

    async def get_status_counts(self) -> dict[str, int]:  # pragma: no cover
        return {}

    async def get_docs_by_statuses(
        self, statuses: list[DocStatus], strict: bool = False
    ) -> dict[str, DocProcessingStatus]:  # pragma: no cover
        return {}

    async def get_docs_by_track_id(
        self, track_id: str
    ) -> dict[str, DocProcessingStatus]:  # pragma: no cover
        return {}

    async def get_docs_paginated(
        self,
        status_filter=None,
        status_filters=None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
    ):  # pragma: no cover
        return [], 0

    async def get_all_status_counts(self) -> dict[str, int]:  # pragma: no cover
        return {}

    async def get_doc_by_file_path(
        self, file_path: str
    ) -> dict[str, Any] | None:  # pragma: no cover
        return None

    async def get_doc_by_file_basename(
        self, basename: str
    ) -> tuple[str, dict[str, Any]] | None:  # pragma: no cover
        return None

    async def get_doc_by_content_hash(
        self, content_hash: str
    ) -> tuple[str, dict[str, Any]] | None:  # pragma: no cover
        return None

    async def get_docs_by_statuses_page(
        self, statuses, *, limit, position=None, strict=False
    ) -> DocStatusPage:  # pragma: no cover
        return DocStatusPage(docs={}, next_position=CURSOR_END)

    async def get_docs_by_ids(
        self, doc_ids: Sequence[str], *, strict: bool = False
    ) -> dict[str, DocSchedulingRecord]:  # pragma: no cover
        return {}

    async def get_full_docs_by_ids(
        self, doc_ids: Sequence[str], *, strict: bool = False
    ) -> dict[str, DocProcessingStatus]:  # pragma: no cover
        return {}

    async def resolve_doc_source_strict(
        self, canonical_source_key: str
    ) -> SourceResolution:  # pragma: no cover
        return SourceAbsent()


def _row(status: DocStatus, created_at: str = "2026-01-01T00:00:00") -> dict:
    return {
        "status": status.value,
        "content_summary": "s",
        "content_length": 1,
        "file_path": "a.pdf",
        "created_at": created_at,
        "updated_at": created_at,
        "track_id": "t1",
        "metadata": {},
    }


def _storage(**rows: dict) -> _MinimalDocStatusStorage:
    storage = _MinimalDocStatusStorage()
    storage.data.update(rows)
    return storage


def test_scheduling_methods_are_mandatory_abstractmethods():
    assert "get_by_id_strict" not in DocStatusStorage.__abstractmethods__
    for name in (
        "get_docs_by_statuses_page",
        "get_docs_by_ids",
        "get_full_docs_by_ids",
        "resolve_doc_source_strict",
    ):
        assert name in DocStatusStorage.__abstractmethods__, name


def test_get_by_id_strict_is_optional_capability():
    async def _run():
        storage = _storage(d1=_row(DocStatus.PENDING))
        assert storage.supports_strict_point_reads is False
        with pytest.raises(StorageCapabilityError):
            await storage.get_by_id_strict("d1")

    asyncio.run(_run())


def test_subclass_missing_an_abstract_cannot_instantiate():
    class _Incomplete(_MinimalDocStatusStorage):
        @abstractmethod
        async def resolve_doc_source_strict(
            self, canonical_source_key: str
        ) -> SourceResolution: ...

    assert "resolve_doc_source_strict" in _Incomplete.__abstractmethods__
    with pytest.raises(TypeError):
        _Incomplete()


def test_scheduling_record_projection_is_lightweight():
    projected = {f.name for f in fields(DocSchedulingRecord)}
    assert "chunks_list" not in projected
    assert "error_msg" not in projected
    assert "content_length" not in projected


def test_count_default_raises_capability_error():
    async def _run():
        storage = _storage(d1=_row(DocStatus.PENDING))
        with pytest.raises(StorageCapabilityError):
            await storage.count_docs_by_statuses([DocStatus.PENDING])

    asyncio.run(_run())


def test_update_fields_refuses_created_at_and_missing_ids():
    async def _run():
        storage = _storage(d1=_row(DocStatus.PENDING))
        with pytest.raises(ValueError):
            await storage.update_doc_status_fields(
                "d1", {"created_at": "2027-01-01T00:00:00"}
            )
        with pytest.raises(StorageRecordNotFoundError):
            await storage.update_doc_status_fields("missing", {"status": "failed"})
        await storage.update_doc_status_fields(
            "missing", {"status": "failed"}, missing_ok=True
        )
        await storage.update_doc_status_fields("d1", {"track_id": "t2"})
        assert storage.data["d1"]["track_id"] == "t2"
        assert storage.data["d1"]["created_at"] == "2026-01-01T00:00:00"

    asyncio.run(_run())


def test_source_conflict_methods_raise_capability_error():
    async def _run():
        storage = _storage(d1=_row(DocStatus.PENDING))
        with pytest.raises(StorageCapabilityError):
            await storage.list_source_conflicts_page(limit=10)
        with pytest.raises(StorageCapabilityError):
            await storage.repair_source_conflict(
                "a.pdf",
                primary_doc_id="d1",
                expected_candidate_count=2,
                expected_candidate_fingerprint="deadbeef",
            )

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Scheduling backend fixture workspace env isolation
# ---------------------------------------------------------------------------


def _resolved_namespace(workspace: str) -> str:
    class _Embed:
        """Minimal embedding stand-in for RedisDocStatusStorage construction."""

        embedding_dim = 4

        async def __call__(self, texts):  # pragma: no cover
            return [[0.0] * 4 for _ in texts]

    storage = RedisDocStatusStorage(
        namespace="doc_status",
        global_config={},
        embedding_func=_Embed(),
        workspace=workspace,
    )
    return storage.final_namespace


def test_the_env_override_really_does_beat_the_constructor(monkeypatch):
    monkeypatch.setenv("REDIS_WORKSPACE", "shared_by_everyone")
    assert _resolved_namespace(_unique_workspace()) == "shared_by_everyone_doc_status"


def test_isolation_restores_the_unique_workspace_and_puts_the_env_back(monkeypatch):
    monkeypatch.setenv("REDIS_WORKSPACE", "shared_by_everyone")

    unique = _unique_workspace()
    with _workspace_env_isolated("REDIS_WORKSPACE"):
        assert "REDIS_WORKSPACE" not in os.environ
        assert _resolved_namespace(unique) == f"{unique}_doc_status"

    assert os.environ["REDIS_WORKSPACE"] == "shared_by_everyone"


def test_isolation_leaves_an_unset_variable_unset(monkeypatch):
    monkeypatch.delenv("REDIS_WORKSPACE", raising=False)
    with _workspace_env_isolated("REDIS_WORKSPACE"):
        assert "REDIS_WORKSPACE" not in os.environ
    assert "REDIS_WORKSPACE" not in os.environ


def test_two_cases_never_collide(monkeypatch):
    monkeypatch.setenv("REDIS_WORKSPACE", "shared_by_everyone")
    with _workspace_env_isolated("REDIS_WORKSPACE"):
        first = _resolved_namespace(_unique_workspace())
    with _workspace_env_isolated("REDIS_WORKSPACE"):
        second = _resolved_namespace(_unique_workspace())
    assert first != second


# ---------------------------------------------------------------------------
# Per-record strict/relaxed parse branches (Mongo, Redis, Postgres)
# ---------------------------------------------------------------------------

_GOOD_FIELDS = {
    "content_summary": "summary",
    "content_length": 10,
    "file_path": "good.txt",
    "status": "failed",
    "created_at": "2026-01-01T00:00:00+00:00",
    "updated_at": "2026-01-01T00:00:00+00:00",
}


class _FakeCursor:
    """In-memory Mongo cursor stand-in."""

    def __init__(self, docs):
        self._docs = docs

    async def to_list(self, length=None):
        return self._docs


class _FakeCollection:
    """In-memory Mongo collection stand-in filtering on ``status.$in``."""

    def __init__(self, docs):
        self._docs = docs

    def find(self, query):
        wanted = set(query["status"]["$in"])
        return _FakeCursor([d for d in self._docs if d["status"] in wanted])


def _mongo_storage(docs) -> MongoDocStatusStorage:
    storage = MongoDocStatusStorage.__new__(MongoDocStatusStorage)
    storage.workspace = "t"
    storage._data = _FakeCollection(docs)
    return storage


async def test_mongo_relaxed_skips_bad_record_strict_raises():
    docs = [
        {"_id": "good", **_GOOD_FIELDS},
        {"_id": "bad", "status": "failed"},
    ]
    storage = _mongo_storage(docs)

    relaxed = await storage.get_docs_by_statuses([DocStatus.FAILED])
    assert set(relaxed) == {"good"}

    with pytest.raises(TypeError):
        await storage.get_docs_by_statuses([DocStatus.FAILED], strict=True)


class _FakePipe:
    """In-memory Redis pipeline stand-in recording queued keys."""

    def __init__(self, entries):
        self._entries = entries
        self._keys = []

    def get(self, key):
        self._keys.append(key)

    async def execute(self):
        return [self._entries[k] for k in self._keys]


class _FakeRedis:
    """In-memory Redis client stand-in supporting ``scan`` and ``pipeline``."""

    def __init__(self, entries):
        self._entries = entries

    async def scan(self, cursor, match=None, count=None):
        return 0, list(self._entries)

    def pipeline(self):
        return _FakePipe(self._entries)


def _redis_storage(entries) -> RedisDocStatusStorage:
    storage = RedisDocStatusStorage.__new__(RedisDocStatusStorage)
    storage.workspace = "t"
    storage.final_namespace = "ns"

    @asynccontextmanager
    async def _conn():
        yield _FakeRedis(entries)

    storage._get_redis_connection = _conn
    return storage


async def test_redis_relaxed_skips_bad_record_strict_raises():
    entries = {
        "ns:good": json.dumps({**_GOOD_FIELDS, "metadata": {}, "error_msg": None}),
        "ns:bad": json.dumps({"status": "failed"}),
    }
    storage = _redis_storage(entries)

    relaxed = await storage.get_docs_by_statuses([DocStatus.FAILED])
    assert set(relaxed) == {"good"}

    with pytest.raises(TypeError):
        await storage.get_docs_by_statuses([DocStatus.FAILED], strict=True)


def _pg_storage(rows) -> PGDocStatusStorage:
    storage = PGDocStatusStorage.__new__(PGDocStatusStorage)
    storage.workspace = "t"

    async def _query(sql, params, multirows=False):
        return rows

    storage.db = SimpleNamespace(query=_query)
    return storage


async def test_postgres_relaxed_skips_bad_row_strict_raises():
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    good_row = {
        "id": "good",
        "content_summary": "summary",
        "content_length": 10,
        "file_path": "good.txt",
        "status": "failed",
        "created_at": ts,
        "updated_at": ts,
        "chunks_count": 0,
        "chunks_list": [],
        "metadata": {},
        "error_msg": None,
    }
    bad_row = {"id": "bad", "status": "failed"}
    storage = _pg_storage([good_row, bad_row])

    relaxed = await storage.get_docs_by_statuses([DocStatus.FAILED])
    assert set(relaxed) == {"good"}

    with pytest.raises(KeyError):
        await storage.get_docs_by_statuses([DocStatus.FAILED], strict=True)
