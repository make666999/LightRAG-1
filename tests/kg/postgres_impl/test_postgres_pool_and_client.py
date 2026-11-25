"""Offline unit tests for PostgreSQL ClientManager, pool reconnect/retry, search_path table checks, and performance timing."""

from __future__ import annotations

import asyncio
import gc
import logging
import weakref
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip(
    "asyncpg", reason="asyncpg is required for PostgreSQL storage tests"
)

import asyncpg  # noqa: E402
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt  # noqa: E402

from lightrag.kg.postgres_impl import (  # noqa: E402
    ClientManager,
    PGGraphStorage,
    PGVectorStorage,
    PostgreSQLDB,
)
from lightrag.namespace import NameSpace  # noqa: E402

pytestmark = pytest.mark.offline


def make_db(attempts: int = 3) -> PostgreSQLDB:
    return PostgreSQLDB(
        {
            "host": "localhost",
            "port": 5432,
            "user": "postgres",
            "password": "postgres",
            "database": "postgres",
            "workspace": "test_ws",
            "max_connections": 10,
            "connection_retry_attempts": attempts,
            "connection_retry_backoff": 0,
            "connection_retry_backoff_max": 0,
            "pool_close_timeout": 5.0,
            "enable_vector": False,
        }
    )


def make_pool(connection: MagicMock) -> MagicMock:
    @asynccontextmanager
    async def acquire():
        yield connection

    pool = MagicMock()
    pool.acquire = acquire
    pool.close = AsyncMock()
    return pool


def refused() -> ConnectionRefusedError:
    return ConnectionRefusedError(111, "refused")


def gated_create_pool(results):
    """A create_pool stand-in whose first call waits until ``release`` is set."""
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def create_pool(**_kwargs):
        nonlocal calls
        calls += 1
        result = results[calls - 1]
        if calls == 1:
            started.set()
            await release.wait()
        if isinstance(result, BaseException):
            raise result
        return result

    return create_pool, started, release, lambda: calls


# ---------------------------------------------------------------------------
# ClientManager shared pool & pgvector requirement
# ---------------------------------------------------------------------------


class TestPostgresClientManager:
    @pytest.fixture(autouse=True)
    def reset_client_manager_state(self):
        pristine = {"db": None, "ref_count": 0, "vector_signature": None}
        ClientManager._instances = dict(pristine)
        yield
        ClientManager._instances = dict(pristine)

    def test_none_vector_storage_disables_vector(self) -> None:
        config = ClientManager.get_config(None)
        assert config["enable_vector"] is False

    def test_only_pg_vector_storage_enables_vector(self) -> None:
        from lightrag.kg import STORAGE_IMPLEMENTATIONS

        for name in STORAGE_IMPLEMENTATIONS["VECTOR_STORAGE"]["implementations"]:
            expected = name == "PGVectorStorage"
            assert ClientManager.get_config(name)["enable_vector"] is expected, name

    @pytest.mark.asyncio
    async def test_get_client_reuses_shared_pool_for_same_vector_settings(self) -> None:
        db = MagicMock()
        db.initdb = AsyncMock()
        db.check_tables = AsyncMock()

        with patch("lightrag.kg.postgres_impl.PostgreSQLDB", return_value=db) as db_cls:
            first = await ClientManager.get_client("PGVectorStorage")
            second = await ClientManager.get_client("PGVectorStorage")

        assert first is db
        assert second is db
        assert ClientManager._instances["ref_count"] == 2
        db_cls.assert_called_once()
        db.initdb.assert_awaited_once()
        db.check_tables.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_client_rejects_conflicting_vector_storage_settings(self) -> None:
        db = MagicMock()
        db.initdb = AsyncMock()
        db.check_tables = AsyncMock()

        with patch("lightrag.kg.postgres_impl.PostgreSQLDB", return_value=db):
            await ClientManager.get_client("NanoVectorDBStorage")

            with pytest.raises(RuntimeError, match="process-wide"):
                await ClientManager.get_client("PGVectorStorage")

    @pytest.mark.asyncio
    async def test_pg_vector_storage_declares_its_own_pgvector_requirement(
        self,
    ) -> None:
        storage = object.__new__(PGVectorStorage)
        storage.db = None
        storage.workspace = "test"
        storage.namespace = "chunks"
        storage.global_config = {}
        storage.table_name = "LIGHTRAG_VDB_CHUNKS"
        storage.legacy_table_name = "LIGHTRAG_VDB_CHUNKS"
        storage.embedding_func = MagicMock(embedding_dim=8)
        storage._flush_lock = MagicMock()

        db = MagicMock()
        db.workspace = None

        @asynccontextmanager
        async def _lock():
            yield

        with (
            patch(
                "lightrag.kg.postgres_impl.ClientManager.get_client",
                new=AsyncMock(return_value=db),
            ) as get_client,
            patch("lightrag.kg.postgres_impl.get_data_init_lock", return_value=_lock()),
            patch.object(PGVectorStorage, "setup_table", new=AsyncMock(), create=True),
        ):
            await PGVectorStorage.initialize(storage)

        get_client.assert_awaited_once_with(vector_storage="PGVectorStorage")
        assert ClientManager.get_config("PGVectorStorage")["enable_vector"] is True


# ---------------------------------------------------------------------------
# Pool reconnect, single-flight waiters, logging, and weakref retention
# ---------------------------------------------------------------------------


class TestPostgresPoolReconnect:
    @pytest.mark.asyncio
    async def test_initdb_retry_does_not_wait_on_the_reconnect_lock(self):
        db = make_db()
        pool = MagicMock()
        create_pool = AsyncMock(side_effect=[refused(), pool])

        with patch("lightrag.kg.postgres_impl.asyncpg.create_pool", create_pool):
            async with db._pool_reconnect_lock:
                await asyncio.wait_for(db.initdb(), timeout=5)

        assert db.pool is pool
        assert create_pool.await_count == 2

    @pytest.mark.asyncio
    async def test_ensure_pool_makes_one_attempt_and_releases_the_lock(self):
        db = make_db()
        create_pool = AsyncMock(side_effect=refused())

        with patch("lightrag.kg.postgres_impl.asyncpg.create_pool", create_pool):
            with pytest.raises(ConnectionRefusedError):
                await asyncio.wait_for(db._ensure_pool(), timeout=5)

        assert db.pool is None
        assert create_pool.await_count == 1
        assert not db._pool_reconnect_lock.locked()

    @pytest.mark.asyncio
    async def test_query_recovers_after_the_database_comes_back(self):
        db = make_db()
        connection = MagicMock()
        connection.fetch = AsyncMock(return_value=[])
        create_pool = AsyncMock(
            side_effect=[refused(), refused(), make_pool(connection)]
        )

        with patch("lightrag.kg.postgres_impl.asyncpg.create_pool", create_pool):
            await asyncio.wait_for(db.query("SELECT 1"), timeout=5)

        connection.fetch.assert_awaited_once_with("SELECT 1")
        assert create_pool.await_count == 3
        assert not db._pool_reconnect_lock.locked()

    @pytest.mark.asyncio
    async def test_query_fails_after_the_configured_attempts(self):
        db = make_db(attempts=2)
        create_pool = AsyncMock(side_effect=refused())

        with patch("lightrag.kg.postgres_impl.asyncpg.create_pool", create_pool):
            with pytest.raises(ConnectionRefusedError):
                await asyncio.wait_for(db.query("SELECT 1"), timeout=5)

        assert create_pool.await_count == 2
        assert not db._pool_reconnect_lock.locked()

    @pytest.mark.asyncio
    async def test_failed_reconnect_does_not_close_a_newer_pool(self):
        db = make_db()
        connection = MagicMock()
        connection.fetch = AsyncMock(return_value=[])
        healthy = make_pool(connection)
        first_attempt_started = asyncio.Event()
        fail_first_attempt = asyncio.Event()
        calls = 0

        async def create_pool(**_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                first_attempt_started.set()
                await fail_first_attempt.wait()
                raise refused()
            return healthy

        with patch("lightrag.kg.postgres_impl.asyncpg.create_pool", create_pool):
            first = asyncio.create_task(db.query("SELECT 1"))
            await first_attempt_started.wait()
            second = asyncio.create_task(db.query("SELECT 2"))
            await asyncio.sleep(0)
            fail_first_attempt.set()
            await asyncio.wait_for(asyncio.gather(first, second), timeout=5)

        healthy.close.assert_not_awaited()
        assert db.pool is healthy
        assert calls == 2

    @pytest.mark.asyncio
    async def test_reset_pool_closes_only_the_pool_that_failed(self):
        db = make_db()
        failed = make_pool(MagicMock())
        newer = make_pool(MagicMock())
        db.pool = newer

        await db._reset_pool(failed)

        newer.close.assert_not_awaited()
        assert db.pool is newer

        await db._reset_pool(newer)

        newer.close.assert_awaited_once()
        assert db.pool is None

    @pytest.mark.asyncio
    async def test_waiters_share_the_failure_of_the_attempt_they_waited_on(self):
        db = make_db()
        create_pool, started, release, calls = gated_create_pool(
            [refused(), make_pool(MagicMock())]
        )

        with patch("lightrag.kg.postgres_impl.asyncpg.create_pool", create_pool):
            leader = asyncio.create_task(db._ensure_pool())
            await started.wait()
            waiters = [asyncio.create_task(db._ensure_pool()) for _ in range(5)]
            await asyncio.sleep(0)
            release.set()
            results = await asyncio.wait_for(
                asyncio.gather(leader, *waiters, return_exceptions=True), timeout=5
            )

            assert all(isinstance(r, ConnectionRefusedError) for r in results)
            assert calls() == 1
            assert db.pool is None

            await asyncio.wait_for(db._ensure_pool(), timeout=5)

        assert calls() == 2
        assert db.pool is not None
        assert not db._pool_reconnect_lock.locked()

    @pytest.mark.asyncio
    async def test_waiters_share_the_pool_of_the_attempt_they_waited_on(self):
        db = make_db()
        healthy = make_pool(MagicMock())
        create_pool, started, release, calls = gated_create_pool([healthy])

        with patch("lightrag.kg.postgres_impl.asyncpg.create_pool", create_pool):
            leader = asyncio.create_task(db._ensure_pool())
            await started.wait()
            waiters = [asyncio.create_task(db._ensure_pool()) for _ in range(5)]
            await asyncio.sleep(0)
            release.set()
            await asyncio.wait_for(asyncio.gather(leader, *waiters), timeout=5)

        assert calls() == 1
        assert db.pool is healthy

    @pytest.mark.asyncio
    async def test_a_cancelled_attempt_is_not_shared_with_its_waiters(self):
        db = make_db()
        healthy = make_pool(MagicMock())
        create_pool, started, _release, calls = gated_create_pool([refused(), healthy])

        with patch("lightrag.kg.postgres_impl.asyncpg.create_pool", create_pool):
            leader = asyncio.create_task(db._ensure_pool())
            await started.wait()
            waiter = asyncio.create_task(db._ensure_pool())
            await asyncio.sleep(0)
            leader.cancel()
            await asyncio.wait_for(waiter, timeout=5)

        with pytest.raises(asyncio.CancelledError):
            await leader
        assert calls() == 2
        assert db.pool is healthy

    @pytest.mark.asyncio
    async def test_a_transient_reconnect_failure_is_a_warning(
        self, lightrag_log_records
    ):
        db = make_db()
        create_pool = AsyncMock(side_effect=refused())

        with patch("lightrag.kg.postgres_impl.asyncpg.create_pool", create_pool):
            with pytest.raises(ConnectionRefusedError):
                await db._ensure_pool()

        messages = [(r.levelno, r.getMessage()) for r in lightrag_log_records]
        assert any(
            level == logging.WARNING and "Failed to reconnect database" in msg
            for level, msg in messages
        )
        assert not any(level >= logging.ERROR for level, _ in messages)

    @pytest.mark.asyncio
    async def test_a_non_transient_reconnect_failure_is_an_error(
        self, lightrag_log_records
    ):
        db = make_db()
        create_pool = AsyncMock(
            side_effect=asyncpg.exceptions.InvalidPasswordError("bad password")
        )

        with patch("lightrag.kg.postgres_impl.asyncpg.create_pool", create_pool):
            with pytest.raises(asyncpg.exceptions.InvalidPasswordError):
                await db._ensure_pool()

        assert any(
            r.levelno == logging.ERROR
            and "Failed to connect database" in r.getMessage()
            for r in lightrag_log_records
        )

    @pytest.mark.asyncio
    async def test_startup_failure_is_an_error_after_all_attempts(
        self, lightrag_log_records
    ):
        db = make_db(attempts=3)
        create_pool = AsyncMock(side_effect=refused())

        with patch("lightrag.kg.postgres_impl.asyncpg.create_pool", create_pool):
            with pytest.raises(ConnectionRefusedError):
                await db.initdb()

        assert create_pool.await_count == 3
        assert any(
            r.levelno == logging.ERROR
            and "Failed to connect database" in r.getMessage()
            for r in lightrag_log_records
        )

    @pytest.mark.asyncio
    async def test_retry_log_reports_the_limit_of_the_running_loop(
        self, lightrag_log_records
    ):
        db = make_db(attempts=7)

        with pytest.raises(ConnectionRefusedError):
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(2),
                retry=retry_if_exception_type(ConnectionError),
                before_sleep=db._log_transient_failure,
                reraise=True,
            ):
                with attempt:
                    raise refused()

        messages = [
            r.getMessage()
            for r in lightrag_log_records
            if "transient connection issue" in r.getMessage()
        ]
        assert len(messages) == 1
        assert "attempt 1/2" in messages[0]

    @pytest.mark.asyncio
    async def test_failed_reconnects_do_not_retain_earlier_callers(self):
        db = make_db(attempts=1)
        create_pool = AsyncMock(
            side_effect=lambda **_kwargs: (_ for _ in ()).throw(refused())
        )

        class Payload:
            """Dummy object tracked via weakref for caller-frame retention."""

        refs = []

        async def caller():
            payload = Payload()
            refs.append(weakref.ref(payload))
            with pytest.raises(ConnectionRefusedError):
                await db.query("SELECT 1")

        with patch("lightrag.kg.postgres_impl.asyncpg.create_pool", create_pool):
            for _ in range(20):
                await caller()

        gc.collect()
        assert create_pool.await_count == 20
        assert sum(ref() is not None for ref in refs) <= 1


# ---------------------------------------------------------------------------
# search_path table existence & performance timing instrumentation
# ---------------------------------------------------------------------------


class TestPostgresSearchPathTableExists:
    @pytest.mark.asyncio
    async def test_check_table_exists_uses_search_path_visible_regclass(self):
        db = PostgreSQLDB.__new__(PostgreSQLDB)
        db.query = AsyncMock(return_value={"exists": True})

        assert await db.check_table_exists("LIGHTRAG_DOC_FULL") is True

        db.query.assert_awaited_once_with(
            "SELECT to_regclass($1) IS NOT NULL AS exists",
            ["lightrag_doc_full"],
        )

    @pytest.mark.asyncio
    async def test_full_entity_relation_migration_reuses_table_exists_helper(self):
        db = PostgreSQLDB.__new__(PostgreSQLDB)
        db.check_table_exists = AsyncMock(return_value=True)
        db.query = AsyncMock()
        db.execute = AsyncMock()

        await db._migrate_create_full_entities_relations_tables()

        assert [call.args[0] for call in db.check_table_exists.await_args_list] == [
            "LIGHTRAG_FULL_ENTITIES",
            "LIGHTRAG_FULL_RELATIONS",
        ]
        db.query.assert_not_called()
        db.execute.assert_not_called()


class TestPostgresPerformanceTiming:
    @pytest.mark.asyncio
    async def test_execute_timing_logs_success(self):
        db = make_db()

        async def fake_run_with_retry(operation, **kwargs):
            conn = AsyncMock()
            conn.execute = AsyncMock(return_value="INSERT 0 1")
            await operation(conn)

        db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)

        with patch("lightrag.kg.postgres_impl.performance_timing_log") as timing_log:
            await db.execute("SELECT 1", timing_label="test label")

        assert any(
            "connection.execute completed" in call.args[0]
            for call in timing_log.call_args_list
        )

    @pytest.mark.asyncio
    async def test_execute_timing_logs_failure(self):
        db = make_db()

        async def fake_run_with_retry(operation, **kwargs):
            conn = AsyncMock()
            conn.execute = AsyncMock(side_effect=RuntimeError("boom"))
            await operation(conn)

        db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)

        with patch("lightrag.kg.postgres_impl.performance_timing_log") as timing_log:
            with pytest.raises(RuntimeError, match="boom"):
                await db.execute("SELECT 1", timing_label="test label")

        assert any(
            "connection.execute failed" in call.args[0]
            for call in timing_log.call_args_list
        )

    @pytest.mark.asyncio
    async def test_graph_upsert_node_passes_timing_label(self):
        storage = PGGraphStorage(
            namespace=NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION,
            workspace="test_ws",
            global_config={},
            embedding_func=AsyncMock(),
        )
        storage.graph_name = "test_graph"
        storage._query = AsyncMock(return_value=[])

        await storage.upsert_node(
            "node-1",
            {
                "entity_id": "node-1",
                "description": "desc",
            },
        )

        assert storage._query.await_args.kwargs["timing_label"] == (
            "test_ws PGGraphStorage.upsert_node"
        )

    @pytest.mark.asyncio
    async def test_graph_upsert_edge_passes_timing_label(self):
        storage = PGGraphStorage(
            namespace=NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION,
            workspace="test_ws",
            global_config={},
            embedding_func=AsyncMock(),
        )
        storage.graph_name = "test_graph"
        storage.db = MagicMock()
        storage.db._run_with_retry = AsyncMock(return_value=None)

        await storage.upsert_edge(
            "node-1",
            "node-2",
            {
                "weight": 1.0,
                "description": "desc",
            },
        )

        assert storage.db._run_with_retry.await_args.kwargs["timing_label"] == (
            "test_ws PGGraphStorage.upsert_edge"
        )
