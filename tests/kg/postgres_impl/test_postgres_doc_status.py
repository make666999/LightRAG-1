"""Offline unit tests for PGDocStatusStorage lookups, track_id hydration, and doc_full parse_engine DDL/migration."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("asyncpg", reason="asyncpg is required for PostgreSQL storage tests")

from lightrag.kg.postgres_impl import (  # noqa: E402
    TABLES,
    PGDocStatusStorage,
    PostgreSQLDB,
)
from lightrag.namespace import NameSpace  # noqa: E402

pytestmark = pytest.mark.offline


def _make_storage(rows=None) -> PGDocStatusStorage:
    storage = PGDocStatusStorage.__new__(PGDocStatusStorage)
    storage.namespace = NameSpace.DOC_STATUS
    storage.workspace = "test_ws"
    storage.global_config = {"embedding_batch_num": 10}
    storage.db = MagicMock()
    storage.db.query = AsyncMock(return_value=rows)
    return storage


def _row(doc_id: str = "doc-1", **overrides):
    base = {
        "id": doc_id,
        "content_summary": "summary",
        "content_length": 12,
        "chunks_count": 1,
        "status": "processed",
        "file_path": "report.pdf",
        "chunks_list": "[]",
        "metadata": "{}",
        "error_msg": None,
        "track_id": "track_123",
        "content_hash": "abc123",
        "created_at": datetime(2024, 1, 1, 0, 0, 0),
        "updated_at": datetime(2024, 1, 1, 0, 0, 0),
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Database-native basename & content_hash lookups
# ---------------------------------------------------------------------------


class TestPGDocStatusLookups:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("basename", ["", "unknown_source"])
    async def test_get_doc_by_file_basename_short_circuits_empty_or_sentinel(
        self, basename
    ):
        storage = _make_storage()
        assert await storage.get_doc_by_file_basename(basename) is None
        storage.db.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_doc_by_file_basename_exact_match(self):
        storage = _make_storage([_row(file_path="report.pdf", track_id=None)])

        result = await storage.get_doc_by_file_basename("report.pdf")

        assert result is not None
        doc_id, doc = result
        assert doc_id == "doc-1"
        assert doc["file_path"] == "report.pdf"
        assert doc["content_hash"] == "abc123"

        sql, params = storage.db.query.call_args.args[:2]
        assert "LIGHTRAG_DOC_STATUS" in sql
        assert "workspace=$1" in sql
        assert params == ["test_ws", "report.pdf"]
        assert "LIKE" not in sql
        assert "file_path = $2" in sql
        assert "created_at ASC" in sql
        assert "id ASC" in sql

    @pytest.mark.asyncio
    async def test_get_doc_by_file_basename_uses_exact_match_for_like_metacharacters(
        self,
    ):
        storage = _make_storage([])

        assert await storage.get_doc_by_file_basename("100%_off.pdf") is None

        sql, params = storage.db.query.call_args.args[:2]
        assert "LIKE" not in sql
        assert params == ["test_ws", "100%_off.pdf"]

    @pytest.mark.asyncio
    async def test_get_doc_by_content_hash_empty_returns_none(self):
        storage = _make_storage()
        assert await storage.get_doc_by_content_hash("") is None
        storage.db.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_doc_by_content_hash_match(self):
        storage = _make_storage([_row(content_hash="hash-abc", track_id=None)])

        result = await storage.get_doc_by_content_hash("hash-abc")

        assert result is not None
        doc_id, doc = result
        assert doc_id == "doc-1"
        assert doc["content_hash"] == "hash-abc"

        sql, params = storage.db.query.call_args.args[:2]
        assert "content_hash=$2" in sql
        assert "id <>" not in sql
        assert "ORDER BY created_at ASC, id ASC" in sql
        assert "LIMIT 1" in sql
        assert params == ["test_ws", "hash-abc"]

    @pytest.mark.asyncio
    async def test_get_doc_by_content_hash_no_match_returns_none(self):
        storage = _make_storage([])
        assert await storage.get_doc_by_content_hash("nope") is None

    @pytest.mark.asyncio
    async def test_get_doc_by_content_hash_also_excludes_rows_pointing_at_the_excluded_id(
        self,
    ):
        storage = _make_storage([_row(id="doc-3", content_hash="hash-abc")])

        result = await storage.get_doc_by_content_hash(
            "hash-abc", exclude_doc_id="doc-1"
        )

        assert result is not None and result[0] == "doc-3"
        sql, params = storage.db.query.call_args.args[:2]
        assert "id <> $3" in sql
        assert "COALESCE((metadata->>'is_duplicate')::boolean, false)" in sql
        assert "COALESCE(metadata->>'original_doc_id', '') = $3" in sql
        assert params == ["test_ws", "hash-abc", "doc-1"]
        assert "ORDER BY created_at ASC, id ASC" in sql and "LIMIT 1" in sql

        storage.db.query.reset_mock()
        storage.db.query.return_value = [_row(content_hash="hash-abc")]
        await storage.get_doc_by_content_hash("hash-abc")
        assert "original_doc_id" not in storage.db.query.call_args.args[0]


# ---------------------------------------------------------------------------
# get_docs_by_track_id hydration resilience
# ---------------------------------------------------------------------------


class TestPGTrackIdHydration:
    @pytest.mark.parametrize("missing_col", ["content_summary", "chunks_count"])
    async def test_track_id_listing_survives_a_row_missing_a_column_without_hiding_siblings(
        self, missing_col
    ):
        bad = _row("doc-bad")
        del bad[missing_col]
        storage = _make_storage([_row(f"doc-{i}") for i in range(5)] + [bad])

        docs = await storage.get_docs_by_track_id("track_123")

        assert set(docs) == {f"doc-{i}" for i in range(5)}
        assert "doc-bad" not in docs

    async def test_an_unknown_column_is_ignored_rather_than_fatal(self):
        storage = _make_storage(
            [
                _row("old_doc"),
                _row("written_by_newer_build", field_from_the_future="value"),
            ]
        )

        docs = await storage.get_docs_by_track_id("track_123")

        assert set(docs) == {"old_doc", "written_by_newer_build"}

    async def test_hydration_decodes_json_columns_and_normalizes_file_path(self):
        storage = _make_storage(
            [
                _row(
                    "doc-1",
                    chunks_list='["chunk-a", "chunk-b"]',
                    metadata='{"k": "v"}',
                    file_path="",
                )
            ]
        )

        docs = await storage.get_docs_by_track_id("track_123")

        doc = docs["doc-1"]
        assert doc.chunks_list == ["chunk-a", "chunk-b"]
        assert doc.metadata == {"k": "v"}
        assert doc.file_path == "no-file-path"
        assert doc.content_hash == "abc123"
        assert doc.created_at.startswith("2024-01-01")

    async def test_empty_result_set_returns_empty_dict(self):
        storage = _make_storage(None)

        assert await storage.get_docs_by_track_id("track_123") == {}


# ---------------------------------------------------------------------------
# LIGHTRAG_DOC_FULL.parse_engine TEXT DDL & widening migration
# ---------------------------------------------------------------------------


class TestDocFullParseEngineText:
    def test_create_ddl_uses_text_for_parse_engine(self):
        ddl = TABLES["LIGHTRAG_DOC_FULL"]["ddl"]
        assert "parse_engine TEXT" in ddl
        assert "parse_engine VARCHAR(32)" not in ddl

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("existing_type", "expect_alter"),
        [
            pytest.param("character varying", True, id="widens_varchar"),
            pytest.param("text", False, id="skips_when_already_text"),
        ],
    )
    async def test_migration_widens_parse_engine_only_when_needed(
        self, existing_type, expect_alter
    ):
        db = PostgreSQLDB.__new__(PostgreSQLDB)
        executed: list[str] = []

        async def fake_query(sql, *args, **kwargs):
            if "information_schema.columns" in sql and "column_name = ANY" in sql:
                return [
                    {"column_name": c}
                    for c in (
                        "sidecar_location",
                        "parse_format",
                        "content_hash",
                        "process_options",
                        "chunk_options",
                        "parse_engine",
                    )
                ]
            if "information_schema.columns" in sql:
                return {"data_type": existing_type}
            return None

        db.query = AsyncMock(side_effect=fake_query)
        db.execute = AsyncMock(side_effect=lambda sql, *a, **k: executed.append(sql))

        await db._migrate_doc_full_add_pipeline_fields()

        altered = any("ALTER COLUMN parse_engine TYPE TEXT" in sql for sql in executed)
        assert altered is expect_alter
