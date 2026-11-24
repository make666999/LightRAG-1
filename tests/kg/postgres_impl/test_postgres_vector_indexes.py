"""Offline unit tests for PostgreSQL vector indexes, HALFVEC, iterative scan, ef_search,
safe index naming, dimension validation, sibling-model cleanup, and vector-space refusal.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("asyncpg", reason="asyncpg is required for PostgreSQL storage tests")

import lightrag.kg.postgres_impl as pg  # noqa: E402
from lightrag.exceptions import (  # noqa: E402
    DataMigrationError,
    StorageCapabilityError,
    VectorSpaceMismatchError,
)
from lightrag.kg.postgres_impl import (  # noqa: E402
    SQL_TEMPLATES,
    PGVectorStorage,
    PostgreSQLDB,
    _check_vector_index_dimension,
    _safe_index_name,
    _with_distance_resort,
)
from lightrag.kg.shared_storage import initialize_share_data  # noqa: E402
from lightrag.namespace import NameSpace  # noqa: E402
from lightrag.utils import EmbeddingFunc  # noqa: E402

pytestmark = pytest.mark.offline

_GLOBAL_CONFIG = {
    "embedding_batch_num": 10,
    "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
}

_QUERY_NAMESPACES = [
    NameSpace.VECTOR_STORE_CHUNKS,
    NameSpace.VECTOR_STORE_ENTITIES,
    NameSpace.VECTOR_STORE_RELATIONSHIPS,
]

HNSW_SQL = "SET LOCAL hnsw.iterative_scan = strict_order"
IVFFLAT_SQL = "SET LOCAL ivfflat.iterative_scan = relaxed_order"


def _embedding_func(dim: int = 768, model_name: str = "test_model") -> EmbeddingFunc:
    async def embed(texts, **kwargs):
        return np.array([[0.1] * dim for _ in texts])

    return EmbeddingFunc(embedding_dim=dim, func=embed, model_name=model_name)


def _scan_db(
    index_type: str | None = "HNSW",
    *,
    enable_vector: bool = True,
    supported=None,
) -> PostgreSQLDB:
    db = PostgreSQLDB.__new__(PostgreSQLDB)
    db.workspace = "gw"
    db.enable_vector = enable_vector
    db.vector_index_type = index_type
    if supported is not None:
        db.iterative_scan_supported = supported
    return db


class _ScanConnection:
    """Records transaction boundaries and statements for scan-setting tests."""

    def __init__(
        self,
        *,
        execute_error: Exception | None = None,
        rows=None,
        assert_no_args: bool = False,
    ):
        self.log: list[str] = []
        self._execute_error = execute_error
        self._rows = rows if rows is not None else [{"id": "a"}]
        self._assert_no_args = assert_no_args
        self.fetch_error: Exception | None = None

    @asynccontextmanager
    async def transaction(self):
        self.log.append("BEGIN")
        try:
            yield
        except BaseException:
            self.log.append("ROLLBACK")
            raise
        else:
            self.log.append("COMMIT")

    async def execute(self, sql, *args):
        if self._assert_no_args:
            assert args == ()
        self.log.append(sql)
        if self._execute_error is not None:
            raise self._execute_error

    async def fetch(self, sql, *args):
        self.log.append("FETCH")
        if self.fetch_error is not None:
            raise self.fetch_error
        return self._rows


# ---------------------------------------------------------------------------
# HALFVEC table creation, query casting, index switching, and dim detection
# ---------------------------------------------------------------------------


class _MockHalfVector:
    """Mimics pgvector.halfvec.HalfVector for testing dimension detection."""

    def __init__(self, dim: int):
        self._dim = dim

    def dimensions(self) -> int:
        return self._dim

    def to_list(self):
        return [0.0] * self._dim


class TestPostgresHalfvec:
    @pytest.fixture
    def mock_pg_db(self):
        db = AsyncMock()
        db.workspace = "test_workspace"
        db.vector_index_type = None

        async def mock_query(sql, params=None, multirows=False, **kwargs):
            if multirows:
                return []
            return {"exists": False, "count": 0}

        db.query = AsyncMock(side_effect=mock_query)
        db.execute = AsyncMock(return_value=None)
        return db

    @pytest.fixture(autouse=True)
    def mock_data_init_lock(self):
        with patch("lightrag.kg.postgres_impl.get_data_init_lock") as mock_lock:
            mock_lock.return_value = AsyncMock()
            yield mock_lock

    @pytest.fixture
    def mock_client_manager(self, mock_pg_db):
        with patch("lightrag.kg.postgres_impl.ClientManager") as mock_manager:
            mock_manager.get_client = AsyncMock(return_value=mock_pg_db)
            mock_manager.release_client = AsyncMock()
            yield mock_manager

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("index_type", "expected_col", "forbidden_col"),
        [
            pytest.param("HNSW_HALFVEC", "HALFVEC(768)", "VECTOR(768)", id="halfvec"),
            pytest.param("HNSW", "VECTOR(768)", "HALFVEC(768)", id="default_vector"),
        ],
    )
    async def test_table_creation_column_type(
        self, mock_client_manager, mock_pg_db, index_type, expected_col, forbidden_col
    ):
        mock_pg_db.vector_index_type = index_type
        mock_pg_db.check_table_exists = AsyncMock(return_value=False)

        storage = PGVectorStorage(
            namespace=NameSpace.VECTOR_STORE_CHUNKS,
            global_config=_GLOBAL_CONFIG,
            embedding_func=_embedding_func(),
            workspace="test_ws",
        )
        await storage.initialize()

        create_table_calls = [
            call
            for call in mock_pg_db.execute.call_args_list
            if "CREATE TABLE" in call[0][0]
        ]
        assert len(create_table_calls) > 0
        create_sql = create_table_calls[0][0][0]
        assert expected_col in create_sql
        assert forbidden_col not in create_sql

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("index_type", "expected_cast", "forbidden_cast"),
        [
            pytest.param("HNSW_HALFVEC", "::halfvec", "::vector", id="halfvec"),
            pytest.param("HNSW", "::vector", "::halfvec", id="default_vector"),
        ],
    )
    @pytest.mark.parametrize("namespace", _QUERY_NAMESPACES)
    async def test_query_uses_matching_vector_cast(
        self,
        mock_client_manager,
        mock_pg_db,
        index_type,
        expected_cast,
        forbidden_cast,
        namespace,
    ):
        mock_pg_db.vector_index_type = index_type
        mock_pg_db.check_table_exists = AsyncMock(return_value=True)

        storage = PGVectorStorage(
            namespace=namespace,
            global_config=_GLOBAL_CONFIG,
            embedding_func=_embedding_func(),
            workspace="test_ws",
        )
        await storage.initialize()

        await storage.query("test query", top_k=5, query_embedding=[0.1] * 768)

        assert mock_pg_db.query.called
        sql = mock_pg_db.query.call_args[0][0]
        assert expected_cast in sql
        assert forbidden_cast not in sql

    @pytest.mark.asyncio
    async def test_create_vector_index_drops_old_indexes_when_switching(
        self, mock_pg_db
    ):
        mock_pg_db.vector_index_type = "HNSW_HALFVEC"
        mock_pg_db.hnsw_m = 16
        mock_pg_db.hnsw_ef = 64
        mock_pg_db.ivfflat_lists = 100
        mock_pg_db.vchordrq_build_options = ""

        table_name = "lightrag_vdb_chunks_test"
        mock_pg_db.query = AsyncMock(return_value=None)
        mock_pg_db.execute = AsyncMock()

        await PostgreSQLDB._create_vector_index(mock_pg_db, table_name, 3072)

        execute_calls = [call[0][0] for call in mock_pg_db.execute.call_args_list]
        drop_calls = [c for c in execute_calls if "DROP INDEX IF EXISTS" in c]
        dropped_names = {
            c.split("DROP INDEX IF EXISTS ")[1].strip() for c in drop_calls
        }
        assert _safe_index_name(table_name, "hnsw_cosine") in dropped_names
        assert _safe_index_name(table_name, "ivfflat_cosine") in dropped_names
        assert _safe_index_name(table_name, "vchordrq_cosine") in dropped_names
        assert _safe_index_name(table_name, "hnsw_halfvec_cosine") not in dropped_names
        assert any("HALFVEC(3072)" in c for c in execute_calls if "ALTER TABLE" in c)
        assert any(
            "halfvec_cosine_ops" in c for c in execute_calls if "CREATE INDEX" in c
        )

    @pytest.mark.asyncio
    async def test_create_vector_index_no_drop_when_index_exists(self, mock_pg_db):
        mock_pg_db.vector_index_type = "HNSW_HALFVEC"
        mock_pg_db.hnsw_m = 16
        mock_pg_db.hnsw_ef = 64
        mock_pg_db.ivfflat_lists = 100
        mock_pg_db.vchordrq_build_options = ""

        mock_pg_db.query = AsyncMock(
            side_effect=lambda sql, *a, **k: {"?column?": 1}
            if "pg_indexes" in sql
            else None
        )
        mock_pg_db.execute = AsyncMock()

        await PostgreSQLDB._create_vector_index(
            mock_pg_db, "lightrag_vdb_chunks_test", 3072
        )

        execute_calls = [call[0][0] for call in mock_pg_db.execute.call_args_list]
        assert not any("DROP INDEX" in c for c in execute_calls)
        assert not any("CREATE INDEX" in c for c in execute_calls)

    @pytest.mark.asyncio
    async def test_setup_table_detects_halfvector_dimension_mismatch(self, mock_pg_db):
        legacy_table = "lightrag_vdb_chunks"
        mock_pg_db.check_table_exists = AsyncMock(
            side_effect=lambda t: t.lower() == legacy_table.lower()
        )

        async def mock_query(sql, params=None, multirows=False, **kwargs):
            if "COUNT(*)" in sql:
                return {"count": 5}
            if "content_vector" in sql:
                return {"content_vector": _MockHalfVector(1024)}
            return None

        mock_pg_db.query = AsyncMock(side_effect=mock_query)
        mock_pg_db.execute = AsyncMock()

        with pytest.raises(VectorSpaceMismatchError, match="dimension 1024 -> 768"):
            await PGVectorStorage.setup_table(
                db=mock_pg_db,
                table_name="lightrag_vdb_chunks_new",
                workspace="test_ws",
                embedding_dim=768,
                legacy_table_name=legacy_table,
                base_table=legacy_table,
            )

    @pytest.mark.asyncio
    async def test_setup_table_accepts_matching_halfvector_dimension(self, mock_pg_db):
        legacy_table = "lightrag_vdb_chunks"
        mock_pg_db.check_table_exists = AsyncMock(
            side_effect=lambda t: t.lower() == legacy_table.lower()
        )
        mock_pg_db.vector_index_type = "HNSW_HALFVEC"

        async def mock_query(sql, params=None, multirows=False, **kwargs):
            if "COUNT(*)" in sql:
                return {"count": 5}
            if "content_vector" in sql:
                return {"content_vector": _MockHalfVector(768)}
            if multirows:
                return []
            return None

        mock_pg_db.query = AsyncMock(side_effect=mock_query)
        mock_pg_db.execute = AsyncMock()

        with patch.object(
            PGVectorStorage, "_pg_create_table", new_callable=AsyncMock
        ) as create_table:
            await PGVectorStorage.setup_table(
                db=mock_pg_db,
                table_name="lightrag_vdb_chunks_new",
                workspace="test_ws",
                embedding_dim=768,
                legacy_table_name=legacy_table,
                base_table=legacy_table,
            )
            create_table.assert_awaited_once()


# ---------------------------------------------------------------------------
# HNSW ef_search widening
# ---------------------------------------------------------------------------


def _ef_sql(value: int) -> str:
    return (
        "SELECT set_config('hnsw.ef_search', GREATEST(COALESCE(NULLIF("
        "current_setting('hnsw.ef_search', true), '')::int, 40), "
        f"{value})::text, true)"
    )


class TestPostgresHnswEfSearch:
    @pytest.mark.parametrize(
        ("index_type", "top_k", "value"),
        [
            ("HNSW", 10, 10),
            ("HNSW", 1000, 1000),
            ("HNSW", 1001, 1000),
            ("HNSW_HALFVEC", 1001, 1000),
        ],
    )
    def test_statement_raises_to_top_k_capped_at_the_guc_maximum(
        self, index_type, top_k, value
    ) -> None:
        assert _scan_db(index_type).ef_search_statement(top_k) == _ef_sql(value)

    @pytest.mark.parametrize("index_type", ["IVFFLAT", None])
    def test_non_hnsw_index_has_no_statement(self, index_type) -> None:
        assert _scan_db(index_type).ef_search_statement(100) is None

    @pytest.mark.parametrize("top_k", [None, 0])
    def test_no_positive_top_k_has_no_statement(self, top_k) -> None:
        assert _scan_db().ef_search_statement(top_k) is None

    @pytest.mark.asyncio
    async def test_both_settings_share_one_round_trip_in_the_query_transaction(
        self,
    ) -> None:
        connection = _ScanConnection(assert_no_args=True)

        rows = await _scan_db()._fetch_with_iterative_scan(
            connection, "SELECT 1", (), top_k=100
        )

        assert connection.log == [
            "BEGIN",
            f"{HNSW_SQL}; {_ef_sql(100)}",
            "FETCH",
            "COMMIT",
        ]
        assert rows == [{"id": "a"}]

    @pytest.mark.asyncio
    async def test_old_pgvector_still_gets_ef_search(self) -> None:
        connection = _ScanConnection(assert_no_args=True)

        await _scan_db(supported=False)._fetch_with_iterative_scan(
            connection, "SELECT 1", (), top_k=100
        )

        assert connection.log == ["BEGIN", _ef_sql(100), "FETCH", "COMMIT"]

    @pytest.mark.asyncio
    async def test_ivfflat_gets_only_its_iterative_scan(self) -> None:
        connection = _ScanConnection(assert_no_args=True)

        await _scan_db("IVFFLAT")._fetch_with_iterative_scan(
            connection, "SELECT 1", (), top_k=100
        )

        assert connection.log == ["BEGIN", IVFFLAT_SQL, "FETCH", "COMMIT"]

    @pytest.mark.asyncio
    async def test_failing_settings_fall_back_to_a_plain_fetch(self) -> None:
        connection = _ScanConnection(
            execute_error=Exception("boom"), assert_no_args=True
        )

        rows = await _scan_db()._fetch_with_iterative_scan(
            connection, "SELECT 1", (), top_k=100
        )

        assert connection.log == [
            "BEGIN",
            f"{HNSW_SQL}; {_ef_sql(100)}",
            "ROLLBACK",
            "FETCH",
        ]
        assert rows == [{"id": "a"}]

    @pytest.mark.asyncio
    async def test_db_query_hands_ann_top_k_to_the_scan_helper(self) -> None:
        db = _scan_db()
        connection = _ScanConnection(assert_no_args=True)

        async def run_with_retry(operation, **kwargs):
            return await operation(connection)

        db._run_with_retry = run_with_retry

        await db.query("SELECT 1", multirows=True, iterative_scan=True, ann_top_k=100)

        assert connection.log[1] == f"{HNSW_SQL}; {_ef_sql(100)}"

    @pytest.mark.asyncio
    async def test_storage_query_passes_its_top_k(self) -> None:
        storage = PGVectorStorage.__new__(PGVectorStorage)
        storage.namespace = "chunks"
        storage.workspace = "gw"
        storage.table_name = "t"
        storage.cosine_better_than_threshold = 0.2
        storage.db = _scan_db()
        storage.db.query = AsyncMock(return_value=[])

        await storage.query("q", top_k=10, query_embedding=[0.1, 0.2])

        call = storage.db.query.await_args
        assert call.kwargs["iterative_scan"] is True
        assert call.kwargs["ann_top_k"] == 10
        assert call.kwargs["params"][2] == 10


# ---------------------------------------------------------------------------
# Iterative scan settings, pgvector version probe, and IVFFLAT distance resort
# ---------------------------------------------------------------------------


class TestPostgresIterativeScan:
    @staticmethod
    def _version_connection(version: str | None) -> AsyncMock:
        connection = AsyncMock()
        connection.fetchrow.return_value = (
            None if version is None else {"extversion": version}
        )
        return connection

    @staticmethod
    def _storage(index_type: str, *, supported=None) -> PGVectorStorage:
        storage = PGVectorStorage.__new__(PGVectorStorage)
        storage.namespace = "chunks"
        storage.workspace = "gw"
        storage.table_name = "t"
        storage.cosine_better_than_threshold = 0.2
        storage.db = _scan_db(index_type, supported=supported)
        storage.db.query = AsyncMock(
            return_value=[{"id": "a", "content": "x", "_distance": 0.1}]
        )
        return storage

    @pytest.mark.parametrize(
        ("index_type", "statement"),
        [
            ("HNSW", HNSW_SQL),
            ("HNSW_HALFVEC", HNSW_SQL),
            ("IVFFLAT", IVFFLAT_SQL),
            ("VCHORDRQ", None),
            (None, None),
        ],
    )
    def test_statement_per_index_type(self, index_type, statement) -> None:
        assert _scan_db(index_type).iterative_scan_statement() == statement

    def test_old_pgvector_has_no_statement(self) -> None:
        db = _scan_db(supported=False)
        assert db.iterative_scan_statement() is None

    @pytest.mark.asyncio
    async def test_setting_is_local_to_the_query_transaction(self) -> None:
        connection = _ScanConnection()

        rows = await _scan_db()._fetch_with_iterative_scan(connection, "SELECT 1", ())

        assert connection.log == ["BEGIN", HNSW_SQL, "FETCH", "COMMIT"]
        assert rows == [{"id": "a"}]

    @pytest.mark.asyncio
    async def test_uncovered_index_runs_a_plain_fetch(self) -> None:
        connection = _ScanConnection()

        await _scan_db("VCHORDRQ")._fetch_with_iterative_scan(
            connection, "SELECT 1", ()
        )

        assert connection.log == ["FETCH"]

    @pytest.mark.asyncio
    async def test_failing_set_falls_back_and_warns_once(self) -> None:
        db = _scan_db()

        with patch("lightrag.kg.postgres_impl.logger") as mock_logger:
            for _ in range(2):
                connection = _ScanConnection(execute_error=Exception("permission denied"))
                rows = await db._fetch_with_iterative_scan(connection, "SELECT 1", ())
                assert connection.log == ["BEGIN", HNSW_SQL, "ROLLBACK", "FETCH"]
                assert rows == [{"id": "a"}]

        assert mock_logger.warning.call_count == 1
        assert "vector scan settings not applied" in str(mock_logger.warning.call_args)

    @pytest.mark.asyncio
    async def test_query_failure_is_reraised_without_a_plain_fallback(self) -> None:
        connection = _ScanConnection()
        connection.fetch_error = RuntimeError("query failed")

        with pytest.raises(RuntimeError, match="query failed"):
            await _scan_db()._fetch_with_iterative_scan(connection, "SELECT 1", ())

        assert connection.log.count("FETCH") == 1

    @pytest.mark.asyncio
    async def test_pool_callbacks_carry_no_session_setting(self) -> None:
        db = _scan_db()
        for name, value in {
            "user": "u",
            "password": "p",
            "database": "d",
            "host": "h",
            "port": 5432,
            "max": 2,
            "statement_cache_size": None,
            "ssl_mode": None,
            "server_settings": None,
            "connection_retry_attempts": 1,
            "connection_retry_backoff": 0,
            "connection_retry_backoff_max": 0,
        }.items():
            setattr(db, name, value)
        db._transient_exceptions = (ConnectionError,)
        db._create_ssl_context = lambda: None
        db.configure_vector_extension = AsyncMock()
        captured: dict = {}

        async def fake_create_pool(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with (
            patch("lightrag.kg.postgres_impl.asyncpg.connect", AsyncMock()),
            patch(
                "lightrag.kg.postgres_impl.asyncpg.create_pool",
                side_effect=fake_create_pool,
            ),
            patch("pgvector.asyncpg.register_vector", AsyncMock()),
        ):
            await db.initdb()
            connection = AsyncMock()
            await captured["init"](connection)
            assert [c.args[0] for c in connection.execute.await_args_list] == []

            connection = MagicMock()
            connection.get_reset_query.return_value = "RESET ALL"
            connection.execute = AsyncMock()
            await captured["reset"](connection)
            assert [c.args[0] for c in connection.execute.await_args_list] == [
                "RESET ALL"
            ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("version", "supported"),
        [("0.7.4", False), ("0.8.0", True), ("1.0.0", True)],
    )
    async def test_version_probe_decides_support(self, version, supported) -> None:
        db = _scan_db()
        with patch("lightrag.kg.postgres_impl.logger") as mock_logger:
            await db._probe_iterative_scan_support(self._version_connection(version))

        assert db.iterative_scan_supported is supported
        assert mock_logger.warning.called is (not supported)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("version", [None, "garbage"])
    async def test_unreadable_version_leaves_support_unknown(self, version) -> None:
        db = _scan_db()
        db.iterative_scan_supported = None

        await db._probe_iterative_scan_support(self._version_connection(version))

        assert db.iterative_scan_supported is None

    @pytest.mark.asyncio
    async def test_version_probe_failure_is_not_fatal(self) -> None:
        connection = AsyncMock()
        connection.fetchrow.side_effect = Exception("boom")
        db = _scan_db()
        db.iterative_scan_supported = None

        await db._probe_iterative_scan_support(connection)

        assert db.iterative_scan_supported is None

    @pytest.mark.asyncio
    async def test_version_probe_ignores_uncovered_index(self) -> None:
        connection = self._version_connection("0.7.0")

        await _scan_db("VCHORDRQ")._probe_iterative_scan_support(connection)

        assert connection.fetchrow.await_count == 0

    @pytest.mark.parametrize(
        ("namespace", "vector_cast"),
        [("chunks", "vector"), ("entities", "halfvec"), ("relationships", "vector")],
    )
    def test_resort_wrapper_restores_distance_order(
        self, namespace, vector_cast
    ) -> None:
        raw = SQL_TEMPLATES[namespace].format(table_name="t", vector_cast=vector_cast)

        wrapped = _with_distance_resort(raw, vector_cast)

        assert wrapped.startswith("WITH relaxed AS MATERIALIZED (")
        assert wrapped.endswith("ORDER BY _distance + 0")
        assert wrapped.count(f"content_vector <=> $4::{vector_cast} AS _distance") == 1
        assert "LIMIT $3" in wrapped
        assert ";" not in wrapped

    def test_resort_wrapper_refuses_a_template_it_cannot_parse(self) -> None:
        with pytest.raises(RuntimeError):
            _with_distance_resort("SELECT 1", "vector")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("index_type", "wrapped"),
        [("IVFFLAT", True), ("HNSW", False)],
    )
    async def test_query_resorts_only_for_ivfflat(self, index_type, wrapped) -> None:
        storage = self._storage(index_type)

        rows = await storage.query("q", top_k=5, query_embedding=[0.1, 0.2])

        call = storage.db.query.await_args
        assert call.args[0].startswith("WITH relaxed AS MATERIALIZED") is wrapped
        assert call.kwargs["iterative_scan"] is True
        if wrapped:
            assert rows == [{"id": "a", "content": "x"}]
        else:
            assert rows == [{"id": "a", "content": "x", "_distance": 0.1}]

    @pytest.mark.asyncio
    async def test_query_skips_resort_when_iterative_scan_unsupported(self) -> None:
        storage = self._storage("IVFFLAT", supported=False)

        await storage.query("q", top_k=5, query_embedding=[0.1])

        assert not storage.db.query.await_args.args[0].startswith("WITH relaxed")


# ---------------------------------------------------------------------------
# Safe index naming (63-byte PostgreSQL identifier cap)
# ---------------------------------------------------------------------------


class TestSafeIndexName:
    def test_short_name_unchanged(self):
        result = _safe_index_name("lightrag_vdb_entity", "hnsw_cosine")
        assert result == "idx_lightrag_vdb_entity_hnsw_cosine"
        assert len(result.encode("utf-8")) <= 63

    def test_long_name_gets_hashed(self):
        long_table_name = "LIGHTRAG_VDB_ENTITY_text_embedding_3_large_3072d"
        result = _safe_index_name(long_table_name, "hnsw_cosine")

        assert len(result.encode("utf-8")) <= 63
        assert result.startswith("idx_")
        assert result.endswith("_hnsw_cosine")
        assert result != f"idx_{long_table_name.lower()}_hnsw_cosine"

    def test_case_insensitive(self):
        assert _safe_index_name("LIGHTRAG_VDB_ENTITY", "hnsw_cosine") == (
            _safe_index_name("lightrag_vdb_entity", "hnsw_cosine")
        )

    def test_boundary_case_exactly_63_bytes(self):
        table_56 = "a" * 56
        result = _safe_index_name(table_56, "id")
        assert result == f"idx_{table_56}_id"
        assert len(result.encode("utf-8")) == 63


# ---------------------------------------------------------------------------
# Vector index dimension limits and fresh-database table probing
# ---------------------------------------------------------------------------


class TestVectorIndexDimension:
    @staticmethod
    def _bare_db(index_type: str) -> PostgreSQLDB:
        db = PostgreSQLDB.__new__(PostgreSQLDB)
        db.vector_index_type = index_type
        db.hnsw_m = 16
        db.hnsw_ef = 200
        db.ivfflat_lists = 100
        db.vchordrq_build_options = ""
        db.query = AsyncMock(return_value=None)
        db.execute = AsyncMock()
        db.check_table_exists = AsyncMock(return_value=True)
        return db

    @pytest.mark.parametrize(
        ("index_type", "dim"),
        [
            ("HNSW", 2000),
            ("IVFFLAT", 2000),
            ("HNSW_HALFVEC", 3072),
            ("HNSW_HALFVEC", 4000),
            ("VCHORDRQ", 8192),
            (None, 8192),
            ("", 8192),
        ],
    )
    def test_supported_dimension_is_accepted(self, index_type, dim):
        assert _check_vector_index_dimension(index_type, dim) is None

    @pytest.mark.parametrize(
        ("index_type", "dim", "suggestion"),
        [
            ("HNSW", 3072, "HNSW_HALFVEC"),
            ("IVFFLAT", 2001, "HNSW_HALFVEC"),
            ("ivfflat", 3072, "HNSW_HALFVEC"),
            ("HNSW_HALFVEC", 4096, "VCHORDRQ"),
        ],
    )
    def test_oversized_dimension_is_refused_with_a_remedy(
        self, index_type, dim, suggestion
    ):
        with pytest.raises(ValueError, match=str(dim)) as exc:
            _check_vector_index_dimension(index_type, dim)
        assert suggestion in str(exc.value)

    @pytest.mark.parametrize("index_type", ["HNSW", "IVFFLAT"])
    def test_remedy_follows_the_dimension_not_the_refused_type(self, index_type):
        with pytest.raises(ValueError) as exc:
            _check_vector_index_dimension(index_type, 4096)
        message = str(exc.value)
        assert "Set POSTGRES_VECTOR_INDEX_TYPE=HNSW_HALFVEC" not in message
        assert "VCHORDRQ" in message

    def test_halfvec_remedy_discloses_the_lossy_conversion(self):
        with pytest.raises(ValueError) as exc:
            _check_vector_index_dimension("HNSW", 3072)
        message = str(exc.value)
        assert "16-bit" in message
        assert "without touching stored vectors" not in message

    @pytest.mark.asyncio
    async def test_create_vector_index_refuses_before_any_ddl(self):
        db = self._bare_db("HNSW")

        with pytest.raises(ValueError, match="HNSW_HALFVEC"):
            await db._create_vector_index("LIGHTRAG_VDB_CHUNKS_model_3072d", 3072)

        db.query.assert_not_awaited()
        db.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_create_vector_index_still_builds_a_supported_index(self):
        db = self._bare_db("HNSW_HALFVEC")

        await db._create_vector_index("LIGHTRAG_VDB_CHUNKS_model_3072d", 3072)

        executed = " ".join(call.args[0] for call in db.execute.await_args_list)
        assert "HALFVEC(3072)" in executed
        assert "halfvec_cosine_ops" in executed

    @pytest.mark.asyncio
    async def test_setup_table_refuses_before_creating_the_table(self):
        db = self._bare_db("HNSW")

        with pytest.raises(ValueError, match="at most 2000"):
            await PGVectorStorage.setup_table(
                db,
                "LIGHTRAG_VDB_CHUNKS_model_3072d",
                "default",
                embedding_dim=3072,
                legacy_table_name="LIGHTRAG_VDB_CHUNKS",
                base_table="LIGHTRAG_VDB_CHUNKS",
            )

        db.check_table_exists.assert_not_awaited()
        db.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_check_tables_probes_existence_without_failing_queries(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            pg,
            "TABLES",
            {
                "LIGHTRAG_PRESENT": {"ddl": "CREATE TABLE LIGHTRAG_PRESENT (id int)"},
                "LIGHTRAG_MISSING": {"ddl": "CREATE TABLE LIGHTRAG_MISSING (id int)"},
            },
        )
        db = self._bare_db("HNSW_HALFVEC")
        db.check_table_exists = AsyncMock(
            side_effect=lambda t: t == "LIGHTRAG_PRESENT"
        )
        db.query = AsyncMock(return_value=[])
        for name in dir(PostgreSQLDB):
            if name.startswith(("_migrate", "_check_llm_cache_needs_migration")):
                monkeypatch.setattr(
                    db, name, AsyncMock(return_value=False), raising=False
                )

        await db.check_tables()

        probed = [call.args[0] for call in db.check_table_exists.await_args_list]
        assert probed[:2] == ["LIGHTRAG_PRESENT", "LIGHTRAG_MISSING"]
        created = [call.args[0] for call in db.execute.await_args_list]
        assert "CREATE TABLE LIGHTRAG_MISSING (id int)" in created
        assert "CREATE TABLE LIGHTRAG_PRESENT (id int)" not in created
        assert not any(
            str(call.args[0]).lstrip().upper().startswith("SELECT 1 FROM")
            for call in db.query.await_args_list
        )


# ---------------------------------------------------------------------------
# Sibling-model cleanup (drop_sibling_models)
# ---------------------------------------------------------------------------


class TestPostgresDropSiblingModels:
    @pytest.fixture(autouse=True)
    def _shared_storage(self):
        initialize_share_data(workers=1)

    @staticmethod
    def _storage(rows, model_name="model-b", dim=768):
        storage = PGVectorStorage(
            namespace="chunks",
            global_config=_GLOBAL_CONFIG,
            embedding_func=_embedding_func(dim=dim, model_name=model_name),
            workspace="ws",
        )
        storage.db = MagicMock()
        storage.db.query = AsyncMock(return_value=rows)
        storage.db.execute = AsyncMock()
        storage._flush_lock = asyncio.Lock()
        return storage

    async def test_deletes_this_workspaces_rows_from_each_sibling_table(self):
        storage = self._storage(
            [
                {"table_name": "lightrag_vdb_chunks_model_a_1536d"},
                {"table_name": "lightrag_vdb_chunks_model_c_1024d"},
            ]
        )

        result = await storage.drop_sibling_models()

        assert result["status"] == "success"
        sqls = [c.args[0] for c in storage.db.execute.call_args_list]
        assert [s.split()[2] for s in sqls] == [
            "lightrag_vdb_chunks_model_a_1536d",
            "lightrag_vdb_chunks_model_c_1024d",
        ]
        assert all(
            c.args[1] == {"workspace": "ws"}
            for c in storage.db.execute.call_args_list
        )
        assert storage.db.query.call_args.args[1][1] == storage.table_name.lower()

    async def test_ignores_tables_without_a_model_dimension_suffix(self):
        storage = self._storage([{"table_name": "lightrag_vdb_chunks_backup"}])

        result = await storage.drop_sibling_models()

        assert result["status"] == "success"
        storage.db.execute.assert_not_called()

    async def test_reports_a_backend_failure(self):
        storage = self._storage([{"table_name": "lightrag_vdb_chunks_model_a_1536d"}])
        storage.db.execute.side_effect = RuntimeError("boom")

        result = await storage.drop_sibling_models()

        assert result["status"] == "error"
        assert "boom" in result["message"]

    @pytest.mark.parametrize(
        "name",
        [
            pytest.param("lightrag_vdbxchunks_model_a_1536d", id="like-wildcard-match"),
            pytest.param('lightrag_vdb_chunks_a"; drop table x; --_8d', id="quoted-name"),
            pytest.param("lightrag_vdb_chunks_Model_8d", id="mixed-case"),
        ],
    )
    async def test_a_name_the_suffix_fold_cannot_produce_is_never_formatted_into_sql(
        self, name
    ):
        storage = self._storage([{"table_name": name}])

        result = await storage.drop_sibling_models()

        assert result["status"] == "success"
        storage.db.execute.assert_not_called()

    async def test_does_not_wait_on_the_flush_lock(self):
        storage = self._storage([{"table_name": "lightrag_vdb_chunks_model_a_1536d"}])
        await storage._flush_lock.acquire()
        try:
            result = await asyncio.wait_for(storage.drop_sibling_models(), timeout=5)
        finally:
            storage._flush_lock.release()

        assert result["status"] == "success"
        assert storage.db.execute.await_count == 1

    def test_model_container_name_is_the_table_that_model_would_attach_to(self):
        other = self._storage([], model_name="Model-A", dim=1536)
        assert (
            self._storage([]).model_container_name("Model-A", 1536)
            == other.table_name.lower()
        )

    async def test_siblings_are_found_by_search_path_visibility_not_one_schema(self):
        storage = self._storage([])

        await storage.drop_sibling_models()

        sql = storage.db.query.call_args.args[0]
        assert "pg_table_is_visible" in sql
        assert "current_schema" not in sql
        assert "information_schema" not in sql


# ---------------------------------------------------------------------------
# Vector-space refusal & fail-loud emptiness / exact count
# ---------------------------------------------------------------------------


class TestPostgresVectorSpaceRefusal:
    @pytest.fixture(autouse=True)
    def _shared_storage(self):
        initialize_share_data(workers=1)

    @staticmethod
    def _db(*, legacy_table, legacy_dim, legacy_rows):
        db = AsyncMock()
        db.vector_index_type = None
        db.check_table_exists = AsyncMock(
            side_effect=lambda name: name.lower() == legacy_table.lower()
        )

        async def query(sql, params=None, multirows=False, **kwargs):
            if "COUNT(*)" in sql:
                return {"count": legacy_rows}
            if "content_vector" in sql:
                return {"content_vector": [0.1] * legacy_dim}
            return {}

        db.query = AsyncMock(side_effect=query)
        db.execute = AsyncMock()
        db._create_vector_index = AsyncMock()
        return db

    async def test_legacy_at_another_dimension_raises_the_typed_refusal(self):
        db = self._db(
            legacy_table="LIGHTRAG_VDB_CHUNKS", legacy_dim=1536, legacy_rows=100
        )

        with pytest.raises(VectorSpaceMismatchError) as excinfo:
            await PGVectorStorage.setup_table(
                db,
                "LIGHTRAG_VDB_CHUNKS_test_model_768d",
                workspace="test_ws",
                embedding_dim=768,
                legacy_table_name="LIGHTRAG_VDB_CHUNKS",
                base_table="LIGHTRAG_VDB_CHUNKS",
            )

        error = excinfo.value
        assert not isinstance(error, DataMigrationError)
        assert error.container == "LIGHTRAG_VDB_CHUNKS"
        assert (error.stored_dim, error.expected_dim) == (1536, 768)

    async def test_an_undeclared_embedding_dimension_is_a_schema_error_not_a_refusal(
        self,
    ):
        db = self._db(
            legacy_table="LIGHTRAG_VDB_CHUNKS", legacy_dim=1536, legacy_rows=100
        )

        with pytest.raises(ValueError) as excinfo:
            await PGVectorStorage.setup_table(
                db,
                "LIGHTRAG_VDB_CHUNKS_test_model_768d",
                workspace="test_ws",
                embedding_dim=None,
                legacy_table_name="LIGHTRAG_VDB_CHUNKS",
                base_table="LIGHTRAG_VDB_CHUNKS",
            )

        error = excinfo.value
        assert not isinstance(error, VectorSpaceMismatchError)
        assert not isinstance(error, DataMigrationError)
        db.execute.assert_not_awaited()

    async def test_a_refused_instance_can_still_be_dropped(self):
        db = self._db(
            legacy_table="LIGHTRAG_VDB_CHUNKS", legacy_dim=1536, legacy_rows=100
        )
        db.workspace = None

        storage = PGVectorStorage(
            namespace="chunks",
            global_config=_GLOBAL_CONFIG,
            embedding_func=_embedding_func(model_name="test-model"),
            workspace="test_ws",
        )
        with patch(
            "lightrag.kg.postgres_impl.ClientManager.get_client",
            AsyncMock(return_value=db),
        ):
            with pytest.raises(VectorSpaceMismatchError):
                await storage.initialize()

            assert storage._flush_lock is not None
            result = await storage.drop()

        assert result["status"] == "success"
        cleared = [call.args[0] for call in db.execute.await_args_list if call.args]
        assert any("LIGHTRAG_VDB_CHUNKS" in sql for sql in cleared)
        assert all("test_model_768d" not in sql for sql in cleared)


class TestPostgresVectorEmptinessIsFailLoud:
    @staticmethod
    def _storage(db):
        storage = PGVectorStorage.__new__(PGVectorStorage)
        storage.workspace = "test_ws"
        storage.namespace = "entities"
        storage.table_name = "LIGHTRAG_VDB_ENTITY_test_model_768d"
        storage.db = db
        storage._pending_vector_docs = {}
        storage._pending_vector_deletes = set()
        storage._flush_lock = asyncio.Lock()
        return storage

    async def test_a_failed_query_propagates_instead_of_reading_as_empty(self):
        db = AsyncMock()
        db.query = AsyncMock(
            side_effect=ConnectionError("server closed the connection")
        )

        with pytest.raises(ConnectionError):
            await self._storage(db).is_empty()

    async def test_a_query_that_returns_no_row_is_also_a_failure(self):
        db = AsyncMock()
        db.query = AsyncMock(return_value=None)

        with pytest.raises(RuntimeError, match="returned no row"):
            await self._storage(db).is_empty()

    @pytest.mark.parametrize("has_data,expected", [(False, True), (True, False)])
    async def test_a_successful_read_answers_from_the_row(self, has_data, expected):
        db = AsyncMock()
        db.query = AsyncMock(return_value={"has_data": has_data})

        assert await self._storage(db).is_empty() is expected

    @pytest.mark.parametrize(
        "pending,row,expected",
        [
            (False, {"row_count": 7}, 7),
            (False, None, RuntimeError),
            (True, {"row_count": 7}, StorageCapabilityError),
        ],
        ids=["counts-the-workspace", "no-row-raises", "buffered-write-raises"],
    )
    async def test_exact_count(self, pending, row, expected):
        db = AsyncMock()
        db.query = AsyncMock(return_value=row)
        storage = self._storage(db)
        if pending:
            storage._pending_vector_deletes = {"ent-1"}

        if isinstance(expected, int):
            assert await storage.get_exact_count() == expected
            sql, params = db.query.await_args.args
            assert "WHERE workspace=$1" in sql and params == ["test_ws"]
        else:
            with pytest.raises(expected):
                await storage.get_exact_count()
            assert db.query.await_count == (0 if pending else 1)

    async def test_strict_read_raises_where_get_by_ids_reports_nothing(self):
        db = AsyncMock()
        db.query = AsyncMock(side_effect=ConnectionError("connection reset"))
        storage = self._storage(db)

        assert await storage.get_by_ids(["ent-1"]) == []
        with pytest.raises(ConnectionError):
            await storage.get_by_ids_strict(["ent-1"])

    async def test_a_buffered_upsert_makes_it_non_empty_without_a_query(self):
        db = AsyncMock()
        called: list[bool] = []

        async def _forbid(*args, **kwargs):
            called.append(True)
            raise AssertionError("must not be reached")

        db.query = AsyncMock(side_effect=_forbid)
        storage = self._storage(db)
        storage._pending_vector_docs = {"ent-1": object()}

        assert await storage.is_empty() is False
        assert called == []
