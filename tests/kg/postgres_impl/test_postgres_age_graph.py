"""Offline unit tests for PGGraphStorage (Apache AGE):
dangling-label retry, one-time graph ensure, quote/parameterized reads,
chunked batch transactions, read contracts, complete-or-raise scans,
full-graph degree selection, search_labels escaping, self-loop listing,
upsert_edge Cypher & advisory locks, and workspace graph naming.
"""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

pytest.importorskip("asyncpg", reason="asyncpg is required for PostgreSQL storage tests")

import asyncpg  # noqa: E402
from tenacity import wait_none  # noqa: E402

from lightrag.kg.postgres_impl import (  # noqa: E402
    _INITIALIZED_GRAPHS_LOOKUP,
    PGGraphEdgeWriteLostError,
    PGGraphQueryException,
    PGGraphStorage,
    PostgreSQLDB,
    _is_transient_graph_write_error,
    _is_valid_age_graph_name,
)

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Shared test doubles & helpers
# ---------------------------------------------------------------------------


class _FakeTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


@asynccontextmanager
async def _null_lock():
    yield


def _make_graph_storage(
    *,
    workspace: str = "test_ws",
    namespace: str = "test_graph",
    graph_name: str = "test_graph",
    post_init: bool = True,
    global_config: dict | None = None,
) -> PGGraphStorage:
    storage = PGGraphStorage.__new__(PGGraphStorage)
    storage.workspace = workspace
    storage.namespace = namespace
    storage.graph_name = graph_name
    if global_config is not None:
        storage.global_config = global_config
    if post_init:
        storage.__post_init__()
    storage.db = MagicMock()
    return storage


def _normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip()


# ---------------------------------------------------------------------------
# 1. Apache AGE dangling-label retry
# ---------------------------------------------------------------------------

_DANGLING_GRAPH = "test_graph"


def _undefined_table(relation: str) -> asyncpg.exceptions.UndefinedTableError:
    return asyncpg.exceptions.UndefinedTableError(
        f'relation "{relation}" does not exist'
    )


def _undecodable(schema: str) -> UnicodeDecodeError:
    head = f'relation "{schema}.'.encode()
    raw = head + b'\x9c\xff" does not exist'
    return UnicodeDecodeError("utf-8", raw, len(head), len(head) + 1, "invalid")


_DANGLING_SHAPES = {
    "undefined_table": lambda: _undefined_table(f"{_DANGLING_GRAPH}.é\u0002garbage"),
    "unicode_decode": lambda: _undecodable(_DANGLING_GRAPH),
}


def _wrapped_query_exc(cause: BaseException, graph_name: str | None = _DANGLING_GRAPH):
    exc = PGGraphQueryException({"message": "write failed", "graph_name": graph_name})
    exc.__cause__ = cause
    return exc


class _AcceptingConnection:
    """Accepts every statement; ``fetch`` returns a row so edge writes succeed."""

    def __init__(self):
        self.statements: list[str] = []

    def transaction(self):
        return _FakeTransaction()

    async def execute(self, sql, *args):
        self.statements.append(sql)
        return ""

    async def fetch(self, sql, *args):
        self.statements.append(sql)
        return [{"r": "edge"}]


@dataclass
class _RetryAttempts:
    calls: int = 0
    succeeded: bool = False


@dataclass(frozen=True)
class _WritePath:
    retried_method: str
    install: Callable[
        [PGGraphStorage, Callable[[], BaseException], int, _RetryAttempts], None
    ]
    invoke: Callable[[PGGraphStorage], Awaitable[Any]]


def _fail_then_run_operation(storage, make_cause, failures, attempts):
    async def fake_run_with_retry(operation, **_kwargs):
        attempts.calls += 1
        if attempts.calls <= failures:
            raise make_cause()
        conn = _AcceptingConnection()
        await operation(conn)
        attempts.succeeded = bool(conn.statements)

    storage.db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)


def _fail_then_execute(storage, make_cause, failures, attempts):
    async def fake_execute(*_args, **_kwargs):
        attempts.calls += 1
        if attempts.calls <= failures:
            raise make_cause()
        attempts.succeeded = True

    storage.db.execute = AsyncMock(side_effect=fake_execute)


_SAMPLE_NODE = ("Alice", {"entity_id": "Alice", "entity_type": "person"})
_SAMPLE_EDGE = ("Alice", "Bob", {"weight": "1.0"})

_WRITE_PATHS = {
    "upsert_node": _WritePath(
        "upsert_node", _fail_then_execute, lambda s: s.upsert_node(*_SAMPLE_NODE)
    ),
    "upsert_edge": _WritePath(
        "upsert_edge", _fail_then_run_operation, lambda s: s.upsert_edge(*_SAMPLE_EDGE)
    ),
    "upsert_nodes_batch": _WritePath(
        "_upsert_node_chunk",
        _fail_then_run_operation,
        lambda s: s.upsert_nodes_batch([_SAMPLE_NODE]),
    ),
    "upsert_edges_batch": _WritePath(
        "_upsert_edge_chunk",
        _fail_then_run_operation,
        lambda s: s.upsert_edges_batch([_SAMPLE_EDGE]),
    ),
}


class TestAgeDanglingLabelRetry:
    @staticmethod
    def _prepare(monkeypatch, path_name, make_cause, failures):
        path = _WRITE_PATHS[path_name]
        monkeypatch.setattr(
            getattr(PGGraphStorage, path.retried_method).retry, "wait", wait_none()
        )
        storage = _make_graph_storage(
            namespace="chunk_entity_relation", graph_name=_DANGLING_GRAPH
        )
        attempts = _RetryAttempts()
        path.install(storage, make_cause, failures, attempts)
        return path, storage, attempts

    @pytest.mark.parametrize("shape", sorted(_DANGLING_SHAPES))
    def test_dangling_label_error_in_graph_schema_is_transient(self, shape):
        assert (
            _is_transient_graph_write_error(
                _wrapped_query_exc(_DANGLING_SHAPES[shape]())
            )
            is True
        )

    @pytest.mark.parametrize(
        "cause",
        [
            pytest.param(
                _undefined_table(f"{_DANGLING_GRAPH}_2.garbage"),
                id="schema-sharing-a-prefix",
            ),
            pytest.param(
                _undefined_table(f"{_DANGLING_GRAPH}.base"), id="real-label-base"
            ),
            pytest.param(
                asyncpg.PostgresError(
                    f'relation "{_DANGLING_GRAPH}.garbage" does not exist'
                ),
                id="not-undefined-table-error",
            ),
            pytest.param(_undecodable("other_graph"), id="undecodable-other-schema"),
        ],
    )
    def test_other_errors_are_not_transient(self, cause):
        assert _is_transient_graph_write_error(_wrapped_query_exc(cause)) is False

    def test_dangling_label_error_without_graph_name_is_not_transient(self):
        cause = _DANGLING_SHAPES["undefined_table"]()
        assert (
            _is_transient_graph_write_error(_wrapped_query_exc(cause, graph_name=None))
            is False
        )
        bare = PGGraphQueryException("write failed")
        bare.__cause__ = cause
        assert _is_transient_graph_write_error(bare) is False

    def test_edge_write_lost_error_keeps_its_graph_name(self):
        err = PGGraphEdgeWriteLostError(_DANGLING_GRAPH, "Alice", "Bob", ("Bob",))
        assert err.graph_name == _DANGLING_GRAPH
        assert _is_transient_graph_write_error(err) is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("path_name", sorted(_WRITE_PATHS))
    async def test_write_path_retries_dangling_label_error(
        self, monkeypatch, path_name
    ):
        shape = "undefined_table"
        path, storage, attempts = self._prepare(
            monkeypatch, path_name, _DANGLING_SHAPES[shape], failures=10
        )

        with pytest.raises(PGGraphQueryException) as excinfo:
            await path.invoke(storage)

        assert type(excinfo.value.__cause__) is type(_DANGLING_SHAPES[shape]())
        assert excinfo.value.graph_name == _DANGLING_GRAPH
        assert attempts.calls == 3
        assert attempts.succeeded is False

    @pytest.mark.asyncio
    async def test_write_path_recovers_after_one_dangling_label_error(
        self, monkeypatch
    ):
        path, storage, attempts = self._prepare(
            monkeypatch,
            "upsert_node",
            _DANGLING_SHAPES["undefined_table"],
            failures=1,
        )

        await path.invoke(storage)

        assert attempts.calls == 2
        assert attempts.succeeded is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("path_name", sorted(set(_WRITE_PATHS) - {"upsert_edge"}))
    async def test_write_path_fails_fast_on_missing_real_label(
        self, monkeypatch, path_name
    ):
        path, storage, attempts = self._prepare(
            monkeypatch,
            path_name,
            lambda: _undefined_table(f"{_DANGLING_GRAPH}.base"),
            failures=10,
        )

        with pytest.raises(PGGraphQueryException) as excinfo:
            await path.invoke(storage)

        assert isinstance(
            excinfo.value.__cause__, asyncpg.exceptions.UndefinedTableError
        )
        assert attempts.calls == 1


# ---------------------------------------------------------------------------
# 2. One-time AGE graph ensure & idempotent initialize()
# ---------------------------------------------------------------------------

_AGE_SESSION_SQL = (
    'SET search_path = ag_catalog, "$user", public; '
    "SELECT 'null'::ag_catalog.agtype"
)
_GRAPH_LOOKUP_SQL = "SELECT 1 FROM ag_catalog.ag_graph WHERE name = left($1, 63)::name"
_DEFAULT_SEARCH_PATH = '"$user", public'


class TestAgeGraphEnsure:
    class _FakeConnection:
        def __init__(
            self, graph_exists: bool = False, create_error: Exception | None = None
        ):
            self.graph_exists = graph_exists
            self.create_error = create_error
            self.executed: list[str] = []
            self.fetched: list[tuple[str, tuple]] = []

        async def execute(self, sql: str, *args):
            self.executed.append(sql)
            if "create_graph" in sql and self.create_error is not None:
                raise self.create_error
            return "OK"

        async def fetchval(self, sql: str, *args):
            self.fetched.append((sql, args))
            return 1 if self.graph_exists else None

        @property
        def create_graph_statements(self) -> list[str]:
            return [sql for sql in self.executed if "create_graph" in sql]

    class _FakeAcquire:
        def __init__(self, connection):
            self._connection = connection

        async def __aenter__(self):
            return self._connection

        async def __aexit__(self, *exc_info):
            return False

    class _FakePool:
        def __init__(self):
            self.connection = TestAgeGraphEnsure._FakeConnection(graph_exists=True)

        def acquire(self):
            return TestAgeGraphEnsure._FakeAcquire(self.connection)

    class _SessionConnection:
        def __init__(self):
            self.search_path = _DEFAULT_SEARCH_PATH
            self.age_loaded = False
            self.executed: list[str] = []

        def get_reset_query(self) -> str:
            return "RESET ALL"

        async def execute(self, sql: str, *args):
            self.executed.append(sql)
            for statement in (s.strip() for s in sql.split(";")):
                if statement.startswith("SET search_path = "):
                    self.search_path = statement.removeprefix("SET search_path = ")
                elif statement == "RESET ALL":
                    self.search_path = _DEFAULT_SEARCH_PATH
                elif statement == "SELECT 'null'::ag_catalog.agtype":
                    self.age_loaded = True
            return "OK"

        async def fetchval(self, sql: str, *args):
            return 1

    class _ReleaseResetAcquire:
        def __init__(self, connection, reset):
            self._connection = connection
            self._reset = reset

        async def __aenter__(self):
            return self._connection

        async def __aexit__(self, *exc_info):
            await self._reset(self._connection)
            return False

    class _ReleaseResetPool:
        def __init__(self, connection, reset):
            self._connection = connection
            self._reset = reset

        def acquire(self):
            return TestAgeGraphEnsure._ReleaseResetAcquire(
                self._connection, self._reset
            )

    @staticmethod
    def _make_db() -> PostgreSQLDB:
        db = PostgreSQLDB.__new__(PostgreSQLDB)
        db._ensured_age_graphs = set()
        db._age_graph_ensure_lock = asyncio.Lock()
        return db

    @staticmethod
    async def _capture_pool_reset_callback(db: PostgreSQLDB):
        for name, value in {
            "workspace": "space1",
            "user": "u",
            "password": "p",
            "database": "d",
            "host": "h",
            "port": 5432,
            "max": 2,
            "statement_cache_size": None,
            "ssl_mode": None,
            "server_settings": None,
            "enable_vector": False,
            "vector_index_type": None,
            "connection_retry_attempts": 1,
            "connection_retry_backoff": 0,
            "connection_retry_backoff_max": 0,
        }.items():
            setattr(db, name, value)
        db._transient_exceptions = (ConnectionError,)
        db._create_ssl_context = lambda: None
        captured: dict = {}

        async def fake_create_pool(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch(
            "lightrag.kg.postgres_impl.asyncpg.create_pool",
            side_effect=fake_create_pool,
        ):
            await db.initdb()
        return captured["reset"]

    @staticmethod
    def _init_storage(existing_labels: list[str]) -> tuple[PGGraphStorage, AsyncMock]:
        storage = PGGraphStorage.__new__(PGGraphStorage)
        storage.workspace = "space1"
        storage.namespace = "chunk_entity_relation"
        storage.global_config = {"vector_storage": "PGVectorStorage"}

        db = AsyncMock()
        db.workspace = None
        db.query = AsyncMock(return_value=[{"name": name} for name in existing_labels])
        db.execute = AsyncMock()
        db._run_with_retry = AsyncMock()
        storage.db = db
        return storage, db

    async def _run_initialize(self, monkeypatch, existing_labels: list[str]) -> list[str]:
        monkeypatch.setattr(
            "lightrag.kg.postgres_impl.get_data_init_lock", lambda: _null_lock()
        )
        storage, db = self._init_storage(existing_labels)
        await storage.initialize()
        return [call.args[0] for call in db.execute.await_args_list]

    @pytest.mark.asyncio
    async def test_existing_graph_is_not_recreated(self):
        db = self._make_db()
        conn = self._FakeConnection(graph_exists=True)

        await db.configure_age(conn, "space1_chunk_entity_relation")

        assert conn.create_graph_statements == []
        assert conn.fetched == [
            (_GRAPH_LOOKUP_SQL, ("space1_chunk_entity_relation",)),
        ]
        assert "space1_chunk_entity_relation" in db._ensured_age_graphs

    @pytest.mark.asyncio
    async def test_age_is_loaded_on_every_checkout_with_an_ensured_graph(self):
        db = self._make_db()
        db._ensured_age_graphs.add("g")
        conn = self._FakeConnection(graph_exists=True)
        replacement = self._FakeConnection(graph_exists=True)

        await db.configure_age(conn, "g")
        await db.configure_age(conn, "g")
        await db.configure_age(replacement, "g")

        assert conn.executed == [_AGE_SESSION_SQL] * 2
        assert replacement.executed == [_AGE_SESSION_SQL]
        assert conn.fetched == replacement.fetched == []

    @pytest.mark.asyncio
    async def test_age_load_failure_does_not_ensure_graph(self):
        db = self._make_db()
        conn = AsyncMock()
        conn.execute.side_effect = asyncpg.exceptions.UndefinedObjectError(
            'type "ag_catalog.agtype" does not exist'
        )

        with pytest.raises(asyncpg.exceptions.UndefinedObjectError):
            await db.configure_age(conn, "g")

        conn.fetchval.assert_not_awaited()
        assert db._ensured_age_graphs == set()

        retry_conn = self._FakeConnection(graph_exists=True)
        await db.configure_age(retry_conn, "g")
        assert retry_conn.executed == [_AGE_SESSION_SQL]
        assert db._ensured_age_graphs == {"g"}

    @pytest.mark.asyncio
    async def test_ensured_graph_issues_no_further_sql(self):
        db = self._make_db()
        conn = self._FakeConnection(graph_exists=False)

        await db.configure_age(conn, "g")
        assert conn.create_graph_statements == ["select create_graph('g')"]
        assert db._ensured_age_graphs == {"g"}
        lookups_after_first = len(conn.fetched)

        for _ in range(20):
            await db.configure_age(conn, "g")

        assert len(conn.fetched) == lookups_after_first
        assert conn.create_graph_statements == ["select create_graph('g')"]

    @pytest.mark.asyncio
    async def test_graphs_are_tracked_independently(self):
        db = self._make_db()
        conn = self._FakeConnection(graph_exists=False)

        await db.configure_age(conn, "graph_a")
        await db.configure_age(conn, "graph_b")
        await db.configure_age(conn, "graph_a")

        assert conn.create_graph_statements == [
            "select create_graph('graph_a')",
            "select create_graph('graph_b')",
        ]
        assert db._ensured_age_graphs == {"graph_a", "graph_b"}

    @pytest.mark.parametrize(
        "race_error",
        [
            asyncpg.exceptions.InvalidSchemaNameError('graph "g" already exists'),
            asyncpg.exceptions.UniqueViolationError("duplicate key value"),
        ],
        ids=["invalid_schema_name_3F000", "unique_violation_23505"],
    )
    @pytest.mark.asyncio
    async def test_creation_race_is_tolerated(self, race_error):
        db = self._make_db()
        conn = self._FakeConnection(graph_exists=False, create_error=race_error)

        await db.configure_age(conn, "g")

        assert db._ensured_age_graphs == {"g"}

    @pytest.mark.asyncio
    async def test_transient_creation_failure_does_not_mark_graph_ensured(self):
        db = self._make_db()
        boom = asyncpg.exceptions.ConnectionDoesNotExistError("connection lost")
        failing_conn = self._FakeConnection(graph_exists=False, create_error=boom)

        with pytest.raises(asyncpg.exceptions.ConnectionDoesNotExistError):
            await db.configure_age(failing_conn, "g")

        assert db._ensured_age_graphs == set()

        retry_conn = self._FakeConnection(graph_exists=False)
        await db.configure_age(retry_conn, "g")
        assert retry_conn.create_graph_statements == ["select create_graph('g')"]

    @pytest.mark.asyncio
    async def test_concurrent_first_use_creates_graph_once(self):
        db = self._make_db()
        conn = self._FakeConnection(graph_exists=False)

        await asyncio.gather(*(db.configure_age(conn, "g") for _ in range(10)))

        assert conn.create_graph_statements == ["select create_graph('g')"]
        assert len(conn.fetched) == 1

    @pytest.mark.asyncio
    async def test_run_with_retry_configures_age_before_the_operation(self):
        db = self._make_db()
        order: list[str] = []

        async def fake_configure_age(connection, graph_name):
            order.append(f"configure_age:{graph_name}")

        db.configure_age = AsyncMock(side_effect=fake_configure_age)
        db.pool = self._FakePool()
        db._ensure_pool = AsyncMock()
        db.connection_retry_attempts = 1
        db.connection_retry_backoff = 0
        db.connection_retry_backoff_max = 0
        db._transient_exceptions = ()

        async def operation(connection):
            order.append("operation")
            return "result"

        result = await db._run_with_retry(
            operation, with_age=True, graph_name="space1_chunk_entity_relation"
        )

        assert result == "result"
        assert order == ["configure_age:space1_chunk_entity_relation", "operation"]

    @pytest.mark.asyncio
    async def test_run_with_retry_skips_age_setup_when_not_requested(self):
        db = self._make_db()
        db.configure_age = AsyncMock()
        db.pool = self._FakePool()
        db._ensure_pool = AsyncMock()
        db.connection_retry_attempts = 1
        db.connection_retry_backoff = 0
        db.connection_retry_backoff_max = 0
        db._transient_exceptions = ()

        async def operation(connection):
            return "result"

        assert await db._run_with_retry(operation) == "result"
        db.configure_age.assert_not_called()

    @pytest.mark.asyncio
    async def test_age_search_path_does_not_leak_across_release_and_reacquire(self):
        db = self._make_db()
        reset = await self._capture_pool_reset_callback(db)
        conn = self._SessionConnection()
        db.pool = self._ReleaseResetPool(conn, reset)
        db._ensure_pool = AsyncMock()
        seen: list[tuple[str, bool]] = []

        async def operation(connection):
            seen.append((connection.search_path, connection.age_loaded))

        await db._run_with_retry(operation, with_age=True, graph_name="g")
        await db._run_with_retry(operation)
        await db._run_with_retry(operation, with_age=True, graph_name="g")

        age_path = 'ag_catalog, "$user", public'
        assert seen == [
            (age_path, True),
            (_DEFAULT_SEARCH_PATH, True),
            (age_path, True),
        ]
        assert conn.executed.count(_AGE_SESSION_SQL) == 2
        assert conn.search_path == _DEFAULT_SEARCH_PATH

    @pytest.mark.asyncio
    async def test_initialize_skips_labels_that_already_exist(self, monkeypatch):
        monkeypatch.setattr(
            "lightrag.kg.postgres_impl.get_data_init_lock", lambda: _null_lock()
        )
        storage, db = self._init_storage(["base", "DIRECTED"])

        await storage.initialize()

        statements = [call.args[0] for call in db.execute.await_args_list]
        assert not [s for s in statements if "create_vlabel" in s]
        assert not [s for s in statements if "create_elabel" in s]
        index_statements = [s for s in statements if "CREATE INDEX" in s]
        assert index_statements
        for statement in index_statements:
            assert "CONCURRENTLY IF NOT EXISTS" in statement, statement

        db.query.assert_awaited_once()
        sql, params = db.query.await_args.args[0], db.query.await_args.args[1]
        assert "ag_catalog.ag_label" in sql
        assert "ag_catalog.ag_graph" in sql
        assert "g.name = left($1, 63)::name" in sql
        assert "name::text = $1" not in sql
        assert params == ["space1_chunk_entity_relation"]
        assert db.query.await_args.kwargs["with_age"] is True
        assert (
            db.query.await_args.kwargs["graph_name"] == "space1_chunk_entity_relation"
        )

    @pytest.mark.asyncio
    async def test_initialize_creates_only_the_missing_label(self, monkeypatch):
        statements = await self._run_initialize(monkeypatch, ["base"])

        assert not [s for s in statements if "create_vlabel" in s]
        assert [s for s in statements if "create_elabel" in s] == [
            "SELECT create_elabel('space1_chunk_entity_relation', 'DIRECTED');"
        ]

    @pytest.mark.asyncio
    async def test_initialize_creates_both_labels_on_a_fresh_graph(self, monkeypatch):
        statements = await self._run_initialize(monkeypatch, [])

        assert statements[0] == (
            "SELECT create_vlabel('space1_chunk_entity_relation', 'base');"
        )
        assert statements[1] == (
            "SELECT create_elabel('space1_chunk_entity_relation', 'DIRECTED');"
        )
        assert not [s for s in statements if "create_graph" in s]

    def test_graph_names_are_ascii_so_bytes_equal_characters(self):
        storage = PGGraphStorage.__new__(PGGraphStorage)
        storage.workspace = "工作区 with spaces/and-punct"
        storage.namespace = "chunk_entity_relation"

        name = storage._get_workspace_graph_name()

        assert name.isascii(), name
        assert len(name) == len(name.encode("utf-8"))


# ---------------------------------------------------------------------------
# 3. Special characters & parameterized Cypher reads (quote fix)
# ---------------------------------------------------------------------------


class TestAgeQuoteAndParameterizedReads:
    def test_normalize_both_special_chars(self):
        assert (
            PGGraphStorage._normalize_node_id('say \\"hello\\"')
            == 'say \\\\\\"hello\\\\\\"'
        )

    @pytest.mark.asyncio
    async def test_get_node_edges_passes_original_id_as_parameter(self):
        storage = _make_graph_storage(post_init=False)
        entity = 'John "Smith"'
        captured_params: list[dict] = []
        captured_sql: list[str] = []

        async def fake_query(sql, **kwargs):
            captured_sql.append(sql)
            if kwargs.get("params"):
                captured_params.append(json.loads(list(kwargs["params"].values())[0]))
            return []

        with patch.object(storage, "_query", side_effect=fake_query):
            await storage.get_node_edges(entity)

        assert len(captured_params) == 1
        assert captured_params[0]["entity_id"] == entity
        assert len(captured_sql) == 1
        assert "$1::agtype" in captured_sql[0]
        assert entity not in captured_sql[0]
        assert '\\"' not in captured_sql[0]

    @pytest.mark.asyncio
    async def test_get_node_edges_returns_edges(self):
        storage = _make_graph_storage(post_init=False)

        async def fake_query(_sql, **_kwargs):
            return [
                {"source_id": "Alice", "connected_id": "Bob"},
                {"source_id": "Alice", "connected_id": None},
                {"source_id": "Alice", "connected_id": "Carol"},
            ]

        with patch.object(storage, "_query", side_effect=fake_query):
            result = await storage.get_node_edges("Alice")

        assert result == [("Alice", "Bob"), ("Alice", "Carol")]

    @pytest.mark.asyncio
    async def test_get_nodes_edges_batch_passes_original_ids_as_parameter(self):
        storage = _make_graph_storage(post_init=False)
        entities = ['John "Smith"', "Alice", "O\\Brien"]
        captured_params: list[dict] = []
        captured_sql: list[str] = []

        async def fake_query(sql, **kwargs):
            captured_sql.append(sql)
            if kwargs.get("params"):
                captured_params.append(json.loads(list(kwargs["params"].values())[0]))
            return []

        with patch.object(storage, "_query", side_effect=fake_query):
            await storage.get_nodes_edges_batch(entities)

        assert len(captured_params) == 2
        assert captured_params[0]["node_ids"] == entities
        assert captured_params[1]["node_ids"] == entities
        assert len(captured_sql) == 2
        for sql in captured_sql:
            assert "$1::agtype" in sql
            assert not [e for e in entities if e in sql]
            assert '\\"' not in sql

    @pytest.mark.asyncio
    async def test_get_nodes_edges_batch_with_quoted_entity(self):
        storage = _make_graph_storage(post_init=False)
        entity = 'John "Smith"'

        async def fake_query(sql, **_kwargs):
            if "OPTIONAL MATCH (n:base)-[]->" in sql:
                return [{"node_id": entity, "connected_id": "Alice"}]
            if "OPTIONAL MATCH (n:base)<-[]-" in sql:
                return [{"node_id": entity, "connected_id": "Bob"}]
            return []

        with patch.object(storage, "_query", side_effect=fake_query):
            result = await storage.get_nodes_edges_batch([entity])

        assert entity in result
        assert (entity, "Alice") in result[entity]
        assert ("Bob", entity) in result[entity]

    @pytest.mark.asyncio
    async def test_get_nodes_edges_batch_no_results(self):
        storage = _make_graph_storage(post_init=False)
        entity = 'Entity "X"'

        with patch.object(storage, "_query", new=AsyncMock(return_value=[])):
            result = await storage.get_nodes_edges_batch([entity])

        assert result == {entity: []}

    @pytest.mark.asyncio
    async def test_get_nodes_edges_batch_deduplication(self):
        storage = _make_graph_storage(post_init=False)
        entity = 'Dup "Entity"'

        async def fake_query(sql, **_kwargs):
            if "OPTIONAL MATCH (n:base)-[]->" in sql:
                return [{"node_id": entity, "connected_id": "Other"}]
            return []

        with patch.object(storage, "_query", side_effect=fake_query):
            result = await storage.get_nodes_edges_batch([entity, entity])

        assert result[entity] == [(entity, "Other")]

    @pytest.mark.asyncio
    async def test_get_nodes_edges_batch_empty_input(self):
        storage = _make_graph_storage(post_init=False)

        with patch.object(storage, "_query") as mock_q:
            result = await storage.get_nodes_edges_batch([])

        assert result == {}
        mock_q.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Chunk-level transaction batch paths
# ---------------------------------------------------------------------------


class TestPGGraphBatchTransactions:
    class _Capture:
        def __init__(self) -> None:
            self.calls: list[dict] = []
            self.tx_count = 0
            self.run_count = 0

    class _RecordingConnection:
        def __init__(self, capture):
            self._capture = capture

        def transaction(self):
            self._capture.tx_count += 1
            return _FakeTransaction()

        async def execute(self, sql, *args):
            self._capture.calls.append({"sql": sql, "args": args})
            return ""

        async def fetch(self, sql, *args):
            self._capture.calls.append({"sql": sql, "args": args})
            return [{"r": "edge"}]

    @classmethod
    def _storage(
        cls,
        *,
        max_upsert_records: int | None = None,
        max_upsert_payload: int | None = None,
        max_delete_records: int | None = None,
    ):
        storage = _make_graph_storage()
        if max_upsert_records is not None:
            storage._max_upsert_records_per_batch = max_upsert_records
        if max_upsert_payload is not None:
            storage._max_upsert_payload_bytes = max_upsert_payload
        if max_delete_records is not None:
            storage._max_delete_records_per_batch = max_delete_records

        capture = cls._Capture()
        conn = cls._RecordingConnection(capture)

        async def fake_run_with_retry(operation, **_kwargs):
            capture.run_count += 1
            return await operation(conn)

        storage.db = AsyncMock()
        storage.db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)
        return storage, capture

    @staticmethod
    def _sql_count(capture, needle: str) -> int:
        return sum(1 for c in capture.calls if needle in c["sql"])

    @pytest.mark.asyncio
    async def test_upsert_nodes_batch_splits_into_chunk_transactions(self):
        storage, cap = self._storage(max_upsert_records=2)
        nodes = [(f"n{i}", {"entity_id": f"n{i}", "v": str(i)}) for i in range(5)]

        await storage.upsert_nodes_batch(nodes)

        assert cap.run_count == 3
        assert cap.tx_count == 3
        assert self._sql_count(cap, "MERGE (n:base") == 5

    @pytest.mark.asyncio
    async def test_upsert_nodes_batch_dedupes_last_write_wins(self):
        storage, cap = self._storage()
        await storage.upsert_nodes_batch(
            [
                ("A", {"entity_id": "A", "weight": "first"}),
                ("A", {"entity_id": "A", "weight": "second"}),
            ]
        )

        merge_calls = [c for c in cap.calls if "MERGE (n:base" in c["sql"]]
        assert len(merge_calls) == 1
        assert '"second"' in merge_calls[0]["sql"]
        assert '"first"' not in merge_calls[0]["sql"]

    @pytest.mark.asyncio
    async def test_upsert_nodes_batch_splits_by_payload_bytes(self):
        storage, cap = self._storage(max_upsert_records=0, max_upsert_payload=200)
        nodes = [
            (f"n{i}", {"entity_id": f"n{i}", "blob": "X" * 150}) for i in range(3)
        ]

        await storage.upsert_nodes_batch(nodes)

        assert cap.run_count == 3

    @pytest.mark.asyncio
    async def test_upsert_nodes_batch_missing_entity_id_raises_before_tx(self):
        storage, cap = self._storage()
        with pytest.raises(ValueError, match="entity_id"):
            await storage.upsert_nodes_batch([("A", {"foo": "bar"})])
        assert cap.run_count == 0

    @pytest.mark.asyncio
    async def test_upsert_edges_batch_splits_into_chunk_transactions(self):
        storage, cap = self._storage(max_upsert_records=2)
        edges = [(f"s{i}", f"t{i}", {"w": str(i)}) for i in range(5)]

        await storage.upsert_edges_batch(edges)

        assert cap.run_count == 3
        assert cap.tx_count == 3
        assert self._sql_count(cap, "CREATE (source)-[r:DIRECTED") == 5
        lock_calls = [c for c in cap.calls if "pg_advisory_xact_lock" in c["sql"]]
        assert len(lock_calls) == 3
        assert {c["args"] for c in lock_calls} == {("test_graph",)}
        assert "pg_advisory_xact_lock" in cap.calls[0]["sql"]

    @pytest.mark.asyncio
    async def test_remove_nodes_chunks_share_one_transaction(self):
        storage, cap = self._storage(max_delete_records=2)
        await storage.remove_nodes([f"n{i}" for i in range(5)])

        assert cap.run_count == 1
        assert cap.tx_count == 1
        assert self._sql_count(cap, "DETACH DELETE n") == 3

    @pytest.mark.asyncio
    async def test_remove_edges_one_transaction_per_chunk(self):
        storage, cap = self._storage(max_delete_records=2)
        edges = [(f"s{i}", f"t{i}") for i in range(5)]

        await storage.remove_edges(edges)

        assert cap.run_count == 3
        assert cap.tx_count == 3
        assert self._sql_count(cap, "DELETE r") == 5

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "method_name",
        [
            "upsert_nodes_batch",
            "upsert_edges_batch",
            "remove_nodes",
            "remove_edges",
        ],
    )
    async def test_batch_methods_empty_noop(self, method_name):
        storage, cap = self._storage()
        await getattr(storage, method_name)([])
        assert cap.run_count == 0


# ---------------------------------------------------------------------------
# 5. Read-path contracts (popular labels, error propagation, missing vs isolated)
# ---------------------------------------------------------------------------


class TestPGGraphReadContract:
    @pytest.mark.asyncio
    async def test_popular_labels_ranks_from_edges_without_scanning_the_vertices(self):
        storage = _make_graph_storage()
        storage._query = AsyncMock(
            return_value=[{"label": f"E{i}"} for i in range(7)]
        )

        labels = await storage.get_popular_labels(limit=7)

        assert labels == [f"E{i}" for i in range(7)]
        assert storage._query.await_count == 1
        sql = _normalize_sql(storage._query.await_args.args[0])
        assert "FROM node_degrees d JOIN test_graph._ag_label_vertex v" in sql, sql
        assert "LIMIT $1" in sql
        assert storage._query.await_args.kwargs["params"] == {"limit": 7}

    @pytest.mark.asyncio
    async def test_popular_labels_tops_up_from_isolated_when_short(self):
        storage = _make_graph_storage()
        storage._query = AsyncMock(
            side_effect=[
                [{"label": "Connected"}],
                [{"label": "Aardvark"}, {"label": "Orphan"}],
            ]
        )

        labels = await storage.get_popular_labels(limit=3)

        assert labels == ["Connected", "Aardvark", "Orphan"]
        assert storage._query.await_count == 2
        isolated_sql = _normalize_sql(storage._query.await_args_list[1].args[0])
        assert (
            "NOT EXISTS ( SELECT 1 FROM test_graph._ag_label_edge e WHERE e.start_id = v.id )"
            in isolated_sql
        )
        assert (
            "NOT EXISTS ( SELECT 1 FROM test_graph._ag_label_edge e WHERE e.end_id = v.id )"
            in isolated_sql
        )
        assert storage._query.await_args_list[1].kwargs["params"] == {"limit": 2}

    @pytest.mark.asyncio
    async def test_popular_labels_orders_by_degree_then_bytewise_label(self):
        storage = _make_graph_storage()
        storage._query = AsyncMock(
            return_value=[{"label": f"E{i}"} for i in range(3)]
        )

        await storage.get_popular_labels(limit=3)

        sql = _normalize_sql(storage._query.await_args.args[0])
        assert 'ORDER BY degree DESC, label COLLATE "C" ASC' in sql

        storage._query = AsyncMock(side_effect=[[], []])
        await storage.get_popular_labels(limit=3)
        isolated_sql = _normalize_sql(storage._query.await_args_list[1].args[0])
        assert 'ORDER BY label COLLATE "C" ASC' in isolated_sql

    @pytest.mark.asyncio
    async def test_get_popular_labels_raises_on_query_error(self):
        storage = _make_graph_storage()
        storage._query = AsyncMock(
            side_effect=PGGraphQueryException({"message": "connection reset"})
        )

        with pytest.raises(PGGraphQueryException):
            await storage.get_popular_labels(limit=10)

    @pytest.mark.asyncio
    async def test_search_labels_raises_on_query_error(self):
        storage = _make_graph_storage()
        storage._query = AsyncMock(
            side_effect=PGGraphQueryException({"message": "boom"})
        )

        with pytest.raises(PGGraphQueryException):
            await storage.search_labels("alpha")

    @pytest.mark.asyncio
    async def test_search_labels_still_short_circuits_on_blank_query(self):
        storage = _make_graph_storage()
        called: list[bool] = []

        async def _forbid(*args, **kwargs):
            called.append(True)
            raise AssertionError("must not be reached")

        storage._query = AsyncMock(side_effect=_forbid)

        assert await storage.search_labels("   ") == []
        assert called == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("rows", "expected"),
        [
            pytest.param([], None, id="missing_node"),
            pytest.param(
                [{"source_id": "Lonely", "connected_id": None}], [], id="isolated_node"
            ),
        ],
    )
    async def test_get_node_edges_distinguishes_missing_and_isolated_node(
        self, rows, expected
    ):
        storage = _make_graph_storage()
        storage._query = AsyncMock(return_value=rows)

        assert await storage.get_node_edges("Entity") == expected


# ---------------------------------------------------------------------------
# 6. get_all_nodes / get_all_edges complete-or-raise
# ---------------------------------------------------------------------------


class TestPGGraphGetAllCompleteOrRaise:
    @pytest.mark.asyncio
    async def test_get_all_nodes_raises_on_corrupt_properties(self):
        storage = _make_graph_storage(post_init=False)
        storage._query = AsyncMock(
            return_value=[
                {"properties": '{"entity_id": "ALICE", "source_id": "chunk-1"}'},
                {"properties": '{"entity_id": "ACME", not-json'},
            ]
        )

        with pytest.raises(PGGraphQueryException) as excinfo:
            await storage.get_all_nodes()

        assert "Corrupt node properties" in excinfo.value.get_message()

    @pytest.mark.asyncio
    async def test_get_all_nodes_returns_every_parsable_row(self):
        storage = _make_graph_storage(post_init=False)
        storage._query = AsyncMock(
            return_value=[
                {"properties": '{"entity_id": "ALICE", "source_id": "chunk-1"}'},
                {"properties": {"entity_id": "ACME", "source_id": "chunk-2"}},
            ]
        )

        nodes = await storage.get_all_nodes()

        assert [n["id"] for n in nodes] == ["ALICE", "ACME"]

    @pytest.mark.asyncio
    async def test_get_all_edges_raises_on_corrupt_properties(self):
        storage = _make_graph_storage(post_init=False)
        storage._query = AsyncMock(
            return_value=[
                {
                    "source": "ACME",
                    "target": "ALICE",
                    "properties": '{"source_id": "chunk-1"',
                },
            ]
        )

        with pytest.raises(PGGraphQueryException) as excinfo:
            await storage.get_all_edges()

        assert "Corrupt edge properties" in excinfo.value.get_message()

    @pytest.mark.asyncio
    async def test_get_all_edges_keeps_source_id_attribution(self):
        storage = _make_graph_storage(post_init=False)
        storage._query = AsyncMock(
            return_value=[
                {
                    "source": "ACME",
                    "target": "ALICE",
                    "properties": '{"source_id": "chunk-1", "weight": 1.0}',
                },
                {
                    "source": "ACME",
                    "target": "BOB",
                    "properties": {"source_id": "chunk-2", "weight": 1.0},
                },
                {
                    "source": '"Al"ice"',
                    "target": "ACME",
                    "properties": {"source_id": "chunk-3", "weight": 1.0},
                },
            ]
        )

        edges = await storage.get_all_edges()

        assert edges[0]["source_id"] == "chunk-1"
        assert edges[0]["source"] == "ACME"
        assert edges[0]["target"] == "ALICE"
        assert edges[1]["source_id"] == "chunk-2"
        assert edges[1]["target"] == "BOB"
        assert edges[2]["source"] == '"Al"ice"'


# ---------------------------------------------------------------------------
# 7. get_knowledge_graph("*") degree selection & directed subgraph dedup
# ---------------------------------------------------------------------------


class TestPGGraphKnowledgeGraphAll:
    class _QueryCapture:
        def __init__(self, *, total_nodes, degree_rows, subgraph_rows):
            self._total_nodes = total_nodes
            self._degree_rows = degree_rows
            self._subgraph_rows = subgraph_rows
            self.count_sql = None
            self.degree_sql = None
            self.degree_params = None
            self.subgraph_sql = None

        def as_side_effect(self):
            async def fake_query(query, **kwargs):
                if "count(distinct n)" in query:
                    self.count_sql = query
                    return [{"total_nodes": self._total_nodes}]
                if "node_degrees" in query:
                    self.degree_sql = query
                    self.degree_params = kwargs.get("params")
                    return self._degree_rows
                self.subgraph_sql = query
                return self._subgraph_rows

            return fake_query

    @staticmethod
    def _node(node_id, entity_id):
        return {"id": node_id, "properties": {"entity_id": entity_id}}

    @staticmethod
    def _edge(edge_id, start_id, end_id, weight="1"):
        return {
            "id": edge_id,
            "label": "DIRECTED",
            "start_id": start_id,
            "end_id": end_id,
            "properties": {"weight": weight},
        }

    @pytest.mark.asyncio
    async def test_degree_selection_sql_is_undirected_and_index_only(self):
        capture = self._QueryCapture(
            total_nodes=2,
            degree_rows=[{"node_id": 1, "degree": 3}, {"node_id": 2, "degree": 0}],
            subgraph_rows=[{"a": self._node(1, "Alice"), "r": None, "b": None}],
        )
        storage = _make_graph_storage(
            post_init=False, global_config={"max_graph_nodes": 1000}
        )

        with patch.object(storage, "_query", side_effect=capture.as_side_effect()):
            await storage.get_knowledge_graph("*", max_nodes=37)

        sql = capture.degree_sql
        assert sql is not None
        assert "LIMIT $1" in sql
        assert "37" not in sql
        assert capture.degree_params == {"limit": 37}
        assert "start_id" in sql and "end_id" in sql and "UNION ALL" in sql
        assert "LEFT JOIN" in sql and "COALESCE" in sql
        normalized = " ".join(sql.split())
        assert "ORDER BY degree DESC, v.id ASC" in normalized
        assert "properties" not in normalized
        assert "agtype_access_operator" not in normalized
        assert "-[r]->()" not in sql
        assert "OPTIONAL MATCH (n)-[r]->()" not in sql

    @pytest.mark.asyncio
    async def test_isolated_node_survives_end_to_end(self):
        capture = self._QueryCapture(
            total_nodes=2,
            degree_rows=[{"node_id": 1, "degree": 2}, {"node_id": 2, "degree": 0}],
            subgraph_rows=[
                {"a": self._node(1, "Alice"), "r": None, "b": None},
                {"a": self._node(2, "Bob"), "r": None, "b": None},
            ],
        )
        storage = _make_graph_storage(
            post_init=False, global_config={"max_graph_nodes": 1000}
        )

        with patch.object(storage, "_query", side_effect=capture.as_side_effect()):
            kg = await storage.get_knowledge_graph("*", max_nodes=50)

        assert "2" in capture.subgraph_sql
        assert {node.id for node in kg.nodes} == {"1", "2"}
        assert kg.is_truncated is False

    @pytest.mark.asyncio
    async def test_subgraph_read_stays_directed_and_dedupes_edges(self):
        duplicate_edge = self._edge(10, 1, 2)
        capture = self._QueryCapture(
            total_nodes=2,
            degree_rows=[{"node_id": 1, "degree": 1}, {"node_id": 2, "degree": 1}],
            subgraph_rows=[
                {
                    "a": self._node(1, "Alice"),
                    "r": duplicate_edge,
                    "b": self._node(2, "Bob"),
                },
                {
                    "a": self._node(1, "Alice"),
                    "r": duplicate_edge,
                    "b": self._node(2, "Bob"),
                },
            ],
        )
        storage = _make_graph_storage(
            post_init=False, global_config={"max_graph_nodes": 1000}
        )

        with patch.object(storage, "_query", side_effect=capture.as_side_effect()):
            kg = await storage.get_knowledge_graph("*", max_nodes=50)

        assert "(a)-[r]->(b)" in capture.subgraph_sql
        assert len(kg.edges) == 1
        assert (kg.edges[0].source, kg.edges[0].target) == ("1", "2")


# ---------------------------------------------------------------------------
# 8. search_labels LIKE escaping & scoring
# ---------------------------------------------------------------------------


class TestPGGraphSearchLabels:
    @staticmethod
    def _like_matches(pattern: str, text: str) -> bool:
        regex = []
        i = 0
        while i < len(pattern):
            ch = pattern[i]
            if ch == "\\" and i + 1 < len(pattern):
                regex.append(re.escape(pattern[i + 1]))
                i += 2
                continue
            if ch == "%":
                regex.append(".*")
            elif ch == "_":
                regex.append(".")
            else:
                regex.append(re.escape(ch))
            i += 1
        return re.fullmatch("".join(regex), text, flags=re.DOTALL) is not None

    @staticmethod
    async def _captured_search(query: str) -> tuple[str, list]:
        storage = _make_graph_storage()
        storage._query = AsyncMock(return_value=[])
        await storage.search_labels(query)
        call = storage._query.await_args
        return call.args[0], list(call.kwargs["params"].values())

    @pytest.mark.asyncio
    async def test_underscore_in_query_is_not_a_wildcard(self):
        sql, params = await self._captured_search("a_b")
        assert "ESCAPE" in sql
        contains = params[0]
        assert self._like_matches(contains, "a_b")
        assert self._like_matches(contains, "x a_b y")
        assert not self._like_matches(contains, "axb")

    @pytest.mark.asyncio
    async def test_percent_in_query_is_not_a_wildcard(self):
        _, params = await self._captured_search("50%")
        contains, prefix = params[0], params[2]
        assert self._like_matches(contains, "rate 50% off")
        assert not self._like_matches(contains, "500 units")
        assert self._like_matches(prefix, "50% off")
        assert not self._like_matches(prefix, "500 units")

    @pytest.mark.asyncio
    async def test_exact_match_parameter_is_the_raw_query(self):
        _, params = await self._captured_search("A_B%")
        assert params[1] == "a_b%"

    @pytest.mark.asyncio
    async def test_underscore_boundary_bonus_needs_a_literal_underscore(self):
        _, params = await self._captured_search("foo")
        space_boundary, underscore_boundary = params[3], params[4]
        assert self._like_matches(underscore_boundary, "xx_foo")
        assert not self._like_matches(underscore_boundary, "barfoo")
        assert self._like_matches(space_boundary, "bar foo")
        assert not self._like_matches(space_boundary, "barfoo")

    @pytest.mark.asyncio
    async def test_boundary_bonus_only_applies_to_contains_matches(self):
        sql, _ = await self._captured_search("foo")
        normalized = re.sub(r"\s+", " ", sql)
        assert re.search(
            r"ELSE \(100 - LENGTH\(label\)\) \+ CASE WHEN label_lower LIKE \$4",
            normalized,
        )
        assert not re.search(r"END \+ CASE", normalized)


# ---------------------------------------------------------------------------
# 9. Self-loop listing contract
# ---------------------------------------------------------------------------


class TestPGGraphSelfLoopListing:
    @staticmethod
    def _storage(outgoing, incoming):
        s = _make_graph_storage()

        async def _query(query, **kwargs):
            return outgoing if "-[]->" in query else incoming

        s._query = AsyncMock(side_effect=_query)
        return s

    @pytest.mark.asyncio
    async def test_self_loop_is_listed_once(self):
        s = self._storage(
            outgoing=[{"node_id": "loop", "connected_id": "loop"}],
            incoming=[{"node_id": "loop", "connected_id": "loop"}],
        )

        result = await s.get_nodes_edges_batch(["loop"])

        assert result["loop"] == [("loop", "loop")]

    @pytest.mark.asyncio
    async def test_ordinary_edge_still_reaches_both_endpoints(self):
        s = self._storage(
            outgoing=[{"node_id": "A", "connected_id": "B"}],
            incoming=[{"node_id": "B", "connected_id": "A"}],
        )

        result = await s.get_nodes_edges_batch(["A", "B"])

        assert result["A"] == [("A", "B")]
        assert result["B"] == [("A", "B")]

    @pytest.mark.asyncio
    async def test_self_loop_alongside_ordinary_edges(self):
        s = self._storage(
            outgoing=[
                {"node_id": "A", "connected_id": "A"},
                {"node_id": "A", "connected_id": "B"},
            ],
            incoming=[
                {"node_id": "A", "connected_id": "A"},
                {"node_id": "A", "connected_id": "C"},
            ],
        )

        result = await s.get_nodes_edges_batch(["A"])

        assert result["A"].count(("A", "A")) == 1
        assert len(result["A"]) == 3


# ---------------------------------------------------------------------------
# 10. upsert_edge Cypher generation, advisory locks, and missing endpoints
# ---------------------------------------------------------------------------


class TestPGGraphUpsertEdgeCypher:
    class _FakeConnection:
        def __init__(self, edge_rows: list | None = None):
            self.calls: list[dict] = []
            self._edge_rows = [{"r": "edge"}] if edge_rows is None else edge_rows

        def transaction(self):
            return _FakeTransaction()

        async def execute(self, sql, *args):
            self.calls.append({"sql": sql, "args": args})
            return ""

        async def fetch(self, sql, *args):
            self.calls.append({"sql": sql, "args": args})
            return self._edge_rows

    class _MissingEndpointConnection(_FakeConnection):
        def __init__(self, present: set[str] | None = None):
            super().__init__(edge_rows=[])
            self._present = present or set()

        async def fetch(self, sql, *args):
            self.calls.append({"sql": sql, "args": args})
            if "unnest(" in sql:
                return [
                    {"entity_id": node_id}
                    for node_id in args[0]
                    if node_id in self._present
                ]
            return []

    @classmethod
    async def _capture_upsert_edge(
        cls, storage: PGGraphStorage, src: str, tgt: str, edge_data
    ):
        conn = cls._FakeConnection()

        async def fake_run_with_retry(operation, **_kwargs):
            return await operation(conn)

        storage.db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)
        await storage.upsert_edge(src, tgt, edge_data)
        return conn.calls

    @classmethod
    async def _capture_upsert_edges_batch(cls, storage: PGGraphStorage, edges):
        conn = cls._FakeConnection()

        async def fake_run_with_retry(operation, **_kwargs):
            return await operation(conn)

        storage.db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)
        await storage.upsert_edges_batch(edges)
        return conn.calls

    @staticmethod
    async def _run_upsert_edge(storage: PGGraphStorage, conn, src, tgt, edge_data=None):
        async def fake_run_with_retry(operation, **_kwargs):
            return await operation(conn)

        storage.db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)
        await storage.upsert_edge(src, tgt, edge_data or {"weight": "1.0"})

    @pytest.mark.asyncio
    async def test_upsert_edge_contains_optional_match_delete_create(self):
        storage = _make_graph_storage()
        calls = await self._capture_upsert_edge(
            storage, "Alice", "Bob", {"weight": "0.5"}
        )

        cypher_call = calls[2]
        cypher_sql = cypher_call["sql"]
        assert "SET r" not in cypher_sql
        assert "ON CREATE SET" not in cypher_sql
        assert "ON MATCH SET" not in cypher_sql
        assert "entity_id: $src_id" in cypher_sql
        assert "entity_id: $tgt_id" in cypher_sql
        params = json.loads(cypher_call["args"][0])
        assert params["src_id"] == "Alice"
        assert params["tgt_id"] == "Bob"
        assert "OPTIONAL MATCH (source)-[old:DIRECTED]-(target)" in cypher_sql
        assert "DELETE old" in cypher_sql
        assert "CREATE (source)-[r:DIRECTED" in cypher_sql
        assert "]->(target)" in cypher_sql
        assert '`weight`: "0.5"' in cypher_sql
        assert "RETURN r" in cypher_sql

    @pytest.mark.asyncio
    async def test_upsert_edge_handles_empty_props(self):
        storage = _make_graph_storage()
        calls = await self._capture_upsert_edge(storage, "Alice", "Bob", {})

        assert "CREATE (source)-[r:DIRECTED {}]->(target)" in calls[2]["sql"]

    @pytest.mark.asyncio
    async def test_upsert_edge_serialises_with_advisory_lock(self):
        storage = _make_graph_storage()
        calls = await self._capture_upsert_edge(
            storage, "Alice", "Bob", {"weight": "0.5"}
        )

        assert len(calls) == 3
        lock_sql = calls[0]["sql"]
        assert "pg_advisory_xact_lock(" in lock_sql
        assert "$1::text || E'\\x01' ||" in lock_sql
        assert "LEAST($2::text, $3::text)" in lock_sql
        assert "GREATEST($2::text, $3::text)" in lock_sql
        assert "test_graph" not in lock_sql
        assert "Alice" not in lock_sql and "Bob" not in lock_sql
        assert calls[0]["args"] == ("test_graph", "Alice", "Bob")

        assert "pg_advisory_xact_lock_shared" in calls[1]["sql"]
        assert calls[1]["args"] == ("test_graph",)
        assert "pg_advisory_xact_lock" not in calls[2]["sql"]

    @pytest.mark.asyncio
    async def test_upsert_edge_lock_key_includes_graph_name(self):
        storage_a = _make_graph_storage(graph_name="graph_a")
        storage_b = _make_graph_storage(graph_name="graph_b")

        calls_a = await self._capture_upsert_edge(storage_a, "Alice", "Bob", {})
        calls_b = await self._capture_upsert_edge(storage_b, "Alice", "Bob", {})

        assert calls_a[0]["args"] == ("graph_a", "Alice", "Bob")
        assert calls_b[0]["args"] == ("graph_b", "Alice", "Bob")
        lock_sql = calls_a[0]["sql"]
        assert "$1::text" in lock_sql
        assert "LEAST($2::text, $3::text)" in lock_sql
        assert "GREATEST($2::text, $3::text)" in lock_sql

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("exc_factory", "expected_transient", "expected_attempts"),
        [
            pytest.param(
                lambda: asyncpg.exceptions.DeadlockDetectedError("simulated deadlock"),
                True,
                3,
                id="transient_deadlock_retried",
            ),
            pytest.param(
                lambda: ValueError("not a transient db error"),
                False,
                1,
                id="non_transient_fails_fast",
            ),
        ],
    )
    async def test_upsert_edge_retry_classification(
        self, monkeypatch, exc_factory, expected_transient, expected_attempts
    ):
        monkeypatch.setattr(PGGraphStorage.upsert_edge.retry, "wait", wait_none())
        storage = _make_graph_storage()
        cause = exc_factory()
        call_count = 0

        async def fake_run_with_retry(_operation, **_kwargs):
            nonlocal call_count
            call_count += 1
            raise cause

        storage.db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)

        with pytest.raises(PGGraphQueryException) as excinfo:
            await storage.upsert_edge("Alice", "Bob", {"weight": "1.0"})

        assert excinfo.value.__cause__ is cause
        assert _is_transient_graph_write_error(excinfo.value) is expected_transient
        assert call_count == expected_attempts

    @pytest.mark.asyncio
    async def test_upsert_edges_batch_iterates_in_sorted_order(self):
        storage = _make_graph_storage()
        calls = await self._capture_upsert_edges_batch(
            storage,
            [
                ("B", "A", {"weight": "1"}),
                ("C", "A", {"weight": "2"}),
                ("D", "A", {"weight": "3"}),
            ],
        )

        lock_calls = [c for c in calls if "pg_advisory_xact_lock" in c["sql"]]
        assert len(lock_calls) == 1
        assert lock_calls[0]["args"] == ("test_graph",)
        cypher_calls = [c for c in calls if "CREATE (source)-[r:DIRECTED" in c["sql"]]
        edge_pairs = [
            (json.loads(c["args"][0])["src_id"], json.loads(c["args"][0])["tgt_id"])
            for c in cypher_calls
        ]
        canonical_keys = [tuple(sorted(pair)) for pair in edge_pairs]
        assert canonical_keys == sorted(canonical_keys)
        assert canonical_keys == [("A", "B"), ("A", "C"), ("A", "D")]

    @pytest.mark.asyncio
    async def test_upsert_edges_batch_dedupes_last_write_wins(self):
        storage = _make_graph_storage()
        calls = await self._capture_upsert_edges_batch(
            storage,
            [
                ("A", "B", {"weight": "first"}),
                ("B", "A", {"weight": "second"}),
            ],
        )

        cypher_calls = [c for c in calls if "CREATE (source)-[r:DIRECTED" in c["sql"]]
        assert len(cypher_calls) == 1
        assert '"second"' in cypher_calls[0]["sql"]
        assert '"first"' not in cypher_calls[0]["sql"]
        assert json.loads(cypher_calls[0]["args"][0]) == {"src_id": "B", "tgt_id": "A"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("present", "expected_missing", "expected_msg_substring"),
        [
            pytest.param({"Alice"}, ("Bob",), "`Bob`", id="one_endpoint_missing"),
            pytest.param(set(), ("Alice", "Bob"), "Alice", id="both_endpoints_missing"),
            pytest.param(
                {"Alice", "Bob"},
                (),
                "concurrent",
                id="probe_finds_both_endpoints",
            ),
        ],
    )
    async def test_upsert_edge_raises_when_no_edge_was_created(
        self, monkeypatch, present, expected_missing, expected_msg_substring
    ):
        monkeypatch.setattr(PGGraphStorage.upsert_edge.retry, "wait", wait_none())
        storage = _make_graph_storage()
        conn = self._MissingEndpointConnection(present=present)

        with pytest.raises(PGGraphEdgeWriteLostError) as excinfo:
            await self._run_upsert_edge(storage, conn, "Alice", "Bob")

        error = excinfo.value
        assert error.missing_endpoints == expected_missing
        assert expected_msg_substring in str(error)
        assert "test_graph" in str(error)

    @pytest.mark.asyncio
    async def test_upsert_edge_write_lost_error_is_not_retried(self, monkeypatch):
        monkeypatch.setattr(PGGraphStorage.upsert_edge.retry, "wait", wait_none())
        storage = _make_graph_storage()
        conn = self._MissingEndpointConnection()
        call_count = 0

        async def fake_run_with_retry(operation, **_kwargs):
            nonlocal call_count
            call_count += 1
            return await operation(conn)

        storage.db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)

        with pytest.raises(PGGraphEdgeWriteLostError):
            await storage.upsert_edge("Alice", "Bob", {"weight": "1.0"})

        assert call_count == 1
        assert issubclass(PGGraphEdgeWriteLostError, PGGraphQueryException)
        assert not _is_transient_graph_write_error(
            PGGraphEdgeWriteLostError("g", "A", "B")
        )

    @pytest.mark.asyncio
    async def test_upsert_edges_batch_raises_and_rolls_back_the_chunk(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            PGGraphStorage._upsert_edge_chunk.retry, "wait", wait_none()
        )
        storage = _make_graph_storage()
        conn = self._MissingEndpointConnection(present={"A"})

        async def fake_run_with_retry(operation, **_kwargs):
            return await operation(conn)

        storage.db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)

        with pytest.raises(PGGraphEdgeWriteLostError) as excinfo:
            await storage.upsert_edges_batch(
                [("A", "B", {"weight": "1"}), ("A", "C", {"weight": "2"})]
            )

        assert (excinfo.value.source_node_id, excinfo.value.target_node_id) == (
            "A",
            "B",
        )
        cypher_calls = [
            c for c in conn.calls if "CREATE (source)-[r:DIRECTED" in c["sql"]
        ]
        assert len(cypher_calls) == 1

    @pytest.mark.asyncio
    async def test_edge_endpoint_probe_failure_does_not_mask_the_error(
        self, monkeypatch
    ):
        monkeypatch.setattr(PGGraphStorage.upsert_edge.retry, "wait", wait_none())
        storage = _make_graph_storage()

        class _ProbeExplodes(TestPGGraphUpsertEdgeCypher._MissingEndpointConnection):
            async def fetch(self, sql, *args):
                if "unnest(" in sql:
                    raise RuntimeError("transaction is aborted")
                return await super().fetch(sql, *args)

        with pytest.raises(PGGraphEdgeWriteLostError) as excinfo:
            await self._run_upsert_edge(storage, _ProbeExplodes(), "Alice", "Bob")

        assert excinfo.value.missing_endpoints == ()


# ---------------------------------------------------------------------------
# 11. Workspace graph name derivation & legacy clipped-graph resolution
# ---------------------------------------------------------------------------


class TestPGWorkspaceGraphName:
    _NAMESPACE = "chunk_entity_relation"

    @classmethod
    def _storage(cls, workspace: str) -> PGGraphStorage:
        storage = PGGraphStorage.__new__(PGGraphStorage)
        storage.workspace = workspace
        storage.namespace = cls._NAMESPACE
        return storage

    @classmethod
    def _graph_name(cls, workspace: str) -> str:
        return cls._storage(workspace)._get_workspace_graph_name()

    @classmethod
    def _initializable_storage(cls, monkeypatch, workspace: str, initialized_graphs=()):
        monkeypatch.setattr(
            "lightrag.kg.postgres_impl.get_data_init_lock", lambda: _null_lock()
        )
        storage = cls._storage(workspace)
        storage.global_config = {"vector_storage": "PGVectorStorage"}
        calls: list[dict] = []

        async def query(sql, params=None, multirows=False, with_age=False, **kwargs):
            calls.append(
                {
                    "sql": sql,
                    "params": params,
                    "with_age": with_age,
                    "graph_name": kwargs.get("graph_name"),
                }
            )
            if sql == _INITIALIZED_GRAPHS_LOOKUP:
                (candidates,) = params
                return [{"name": n} for n in initialized_graphs if n in candidates]
            return [{"name": "base"}, {"name": "DIRECTED"}]

        db = AsyncMock()
        db.workspace = None
        db.query = AsyncMock(side_effect=query)
        db.execute = AsyncMock()
        db._run_with_retry = AsyncMock()
        storage.db = db
        return storage, calls

    @pytest.mark.parametrize(
        ("workspace", "expected"),
        [
            ("space1", "space1_chunk_entity_relation"),
            ("_team", "_team_chunk_entity_relation"),
            ("v1.0", "v1_0_chunk_entity_relation"),
            ("w" * 41, "w" * 41 + "_chunk_entity_relation"),
            ("", "chunk_entity_relation"),
            ("default", "chunk_entity_relation"),
        ],
    )
    def test_names_that_already_worked_are_unchanged(self, workspace, expected):
        assert self._graph_name(workspace) == expected

    @pytest.mark.parametrize(
        "workspace",
        [
            "w" * 42,
            "acme_corp_customer_support_knowledge_base_production_eu",
            "2026kb",
            "1abc",
        ],
    )
    def test_invalid_names_fall_back_to_a_valid_hashed_name(self, workspace):
        name = self._graph_name(workspace)

        assert _is_valid_age_graph_name(name), name
        assert name.startswith("ws_")
        assert name.endswith("_chunk_entity_relation")
        assert name.isascii()
        assert name == self._graph_name(workspace)

    def test_long_workspaces_sharing_a_63_byte_prefix_get_separate_graphs(self):
        prefix = "x" * 63
        assert self._graph_name(prefix + "_eu") != self._graph_name(prefix + "_us")

    def test_digit_workspace_does_not_take_the_graph_of_a_ws_prefixed_one(self):
        assert self._graph_name("2026kb") != self._graph_name("ws_2026kb")

    def test_hashed_names_follow_the_same_sanitizing_rule_as_short_names(self):
        assert self._graph_name("v1.0") == self._graph_name("v1_0")
        assert self._graph_name("v1.0" + "x" * 50) == self._graph_name(
            "v1_0" + "x" * 50
        )

    @pytest.mark.parametrize(
        ("workspace", "expected"),
        [
            ("w" * 42, ("w" * 42 + "_chunk_entity_relation")[:63]),
            ("w" * 41, None),
            ("space1", None),
            ("2026kb", None),
            ("1" + "x" * 60, None),
            ("", None),
            ("default", None),
        ],
    )
    def test_legacy_clipped_graph_name(self, workspace, expected):
        assert self._storage(workspace)._get_legacy_clipped_graph_name() == expected

    @pytest.mark.parametrize(
        ("name", "valid"),
        [
            ("tenant_a_chunk_entity_relation", True),
            ("_team_chunk_entity_relation", True),
            ("a" * 63, True),
            ("a" * 64, False),
            ("2026kb_chunk_entity_relation", False),
        ],
    )
    def test_is_valid_age_graph_name(self, name, valid):
        assert _is_valid_age_graph_name(name) is valid

    @pytest.mark.asyncio
    async def test_initialize_keeps_an_existing_clipped_graph(self, monkeypatch):
        warning = Mock()
        monkeypatch.setattr("lightrag.kg.postgres_impl.logger.warning", warning)
        clipped = ("w" * 42 + "_chunk_entity_relation")[:63]
        storage, calls = self._initializable_storage(monkeypatch, "w" * 42, [clipped])

        await storage.initialize()

        assert storage.graph_name == clipped
        warning.assert_called_once()
        lookup = next(
            i for i, c in enumerate(calls) if c["sql"] == _INITIALIZED_GRAPHS_LOOKUP
        )
        first_age = next(i for i, c in enumerate(calls) if c["with_age"])
        assert lookup < first_age
        assert calls[lookup]["params"] == [[self._graph_name("w" * 42), clipped]]
        assert calls[first_age]["graph_name"] == clipped

    @pytest.mark.asyncio
    async def test_initialize_uses_the_hashed_name_when_no_clipped_graph_exists(
        self, monkeypatch
    ):
        warning = Mock()
        monkeypatch.setattr("lightrag.kg.postgres_impl.logger.warning", warning)
        storage, calls = self._initializable_storage(monkeypatch, "w" * 42)

        await storage.initialize()

        assert storage.graph_name == self._graph_name("w" * 42)
        assert storage.graph_name.startswith("ws_")
        assert [c for c in calls if c["sql"] == _INITIALIZED_GRAPHS_LOOKUP]
        warning.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_stays_on_the_hashed_graph_when_a_clipped_one_appears(
        self, monkeypatch
    ):
        warning = Mock()
        monkeypatch.setattr("lightrag.kg.postgres_impl.logger.warning", warning)
        hashed = self._graph_name("w" * 42)
        clipped = ("w" * 42 + "_chunk_entity_relation")[:63]
        storage, calls = self._initializable_storage(
            monkeypatch, "w" * 42, [hashed, clipped]
        )

        await storage.initialize()

        assert storage.graph_name == hashed
        first_age = next(c for c in calls if c["with_age"])
        assert first_age["graph_name"] == hashed
        warning.assert_called_once()
        assert hashed in warning.call_args.args[0]
        assert clipped in warning.call_args.args[0]

    @pytest.mark.asyncio
    async def test_initialize_reopens_its_hashed_graph_without_a_warning(
        self, monkeypatch
    ):
        warning = Mock()
        monkeypatch.setattr("lightrag.kg.postgres_impl.logger.warning", warning)
        hashed = self._graph_name("w" * 42)
        storage, _ = self._initializable_storage(monkeypatch, "w" * 42, [hashed])

        await storage.initialize()

        assert storage.graph_name == hashed
        warning.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("workspace", ["space1", "w" * 41, "2026kb"])
    async def test_initialize_skips_the_lookup_when_no_clipped_name_applies(
        self, monkeypatch, workspace
    ):
        storage, calls = self._initializable_storage(monkeypatch, workspace)

        await storage.initialize()

        assert storage.graph_name == self._graph_name(workspace)
        assert not [c for c in calls if c["sql"] == _INITIALIZED_GRAPHS_LOOKUP]
