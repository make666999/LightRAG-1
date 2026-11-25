import asyncio
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

import lightrag.kg.postgres_impl as pg_impl
from lightrag.exceptions import DataMigrationError
from lightrag.kg.postgres_impl import PGVectorStorage
from lightrag.namespace import NameSpace
from lightrag.utils import EmbeddingFunc
from tests._pg_legacy_migration import answer_legacy_migration


pytestmark = pytest.mark.offline


# Mock PostgreSQLDB
@pytest.fixture
def mock_pg_db():
    """Mock PostgreSQL database connection"""
    db = AsyncMock()
    # Real default; an auto-created mock breaks _check_vector_index_dimension.
    db.vector_index_type = None
    db.workspace = "test_workspace"

    # Mock query responses with multirows support
    async def mock_query(sql, params=None, multirows=False, **kwargs):
        # Default return value
        if multirows:
            return []  # Return empty list for multirows
        return {"exists": False, "count": 0}

    # Mock for execute that mimics PostgreSQLDB.execute() behavior
    async def mock_execute(sql, data=None, **kwargs):
        """
        Mock that mimics PostgreSQLDB.execute() behavior:
        - Accepts data as dict[str, Any] | None (second parameter)
        - Internally converts dict.values() to tuple for AsyncPG
        """
        # Mimic real execute() which accepts dict and converts to tuple
        if data is not None and not isinstance(data, dict):
            raise TypeError(
                f"PostgreSQLDB.execute() expects data as dict, got {type(data).__name__}"
            )
        return None

    db.query = AsyncMock(side_effect=mock_query)
    db.execute = AsyncMock(side_effect=mock_execute)

    return db


# Mock get_data_init_lock to avoid async lock issues in tests
@pytest.fixture(autouse=True)
def mock_data_init_lock():
    with patch("lightrag.kg.postgres_impl.get_data_init_lock") as mock_lock:
        mock_lock_ctx = AsyncMock()
        mock_lock.return_value = mock_lock_ctx
        yield mock_lock


# Mock ClientManager
@pytest.fixture
def mock_client_manager(mock_pg_db):
    with patch("lightrag.kg.postgres_impl.ClientManager") as mock_manager:
        mock_manager.get_client = AsyncMock(return_value=mock_pg_db)
        mock_manager.release_client = AsyncMock()
        yield mock_manager


# Mock Embedding function
@pytest.fixture
def mock_embedding_func():
    async def embed_func(texts, **kwargs):
        return np.array([[0.1] * 768 for _ in texts])

    func = EmbeddingFunc(embedding_dim=768, func=embed_func, model_name="test_model")
    return func


async def test_postgres_table_naming(
    mock_client_manager, mock_pg_db, mock_embedding_func
):
    """Test if table name is correctly generated with model suffix"""
    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
    }

    storage = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        global_config=config,
        embedding_func=mock_embedding_func,
        workspace="test_ws",
    )

    # Verify table name contains model suffix
    expected_suffix = "test_model_768d"
    assert expected_suffix in storage.table_name
    assert storage.table_name == f"LIGHTRAG_VDB_CHUNKS_{expected_suffix}"

    # Verify legacy table name
    assert storage.legacy_table_name == "LIGHTRAG_VDB_CHUNKS"


async def test_postgres_no_migration_needed(
    mock_client_manager, mock_pg_db, mock_embedding_func
):
    """Test scenario where new table already exists (no migration needed)"""
    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
    }

    storage = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        global_config=config,
        embedding_func=mock_embedding_func,
        workspace="test_ws",
    )

    # Mock: new table already exists
    async def mock_check_table_exists(table_name):
        return table_name == storage.table_name

    mock_pg_db.check_table_exists = AsyncMock(side_effect=mock_check_table_exists)

    with patch(
        "lightrag.kg.postgres_impl.PGVectorStorage._pg_create_table", AsyncMock()
    ) as mock_create:
        await storage.initialize()

        # Verify no table creation was attempted
        mock_create.assert_not_called()


async def test_scenario_1_new_workspace_creation(
    mock_client_manager, mock_pg_db, mock_embedding_func
):
    """
    Scenario 1: New workspace creation

    Expected behavior:
    - No legacy table exists
    - Directly create new table with model suffix
    - No migration needed
    """
    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
    }

    embedding_func = EmbeddingFunc(
        embedding_dim=3072,
        func=mock_embedding_func.func,
        model_name="text-embedding-3-large",
    )

    storage = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        global_config=config,
        embedding_func=embedding_func,
        workspace="new_workspace",
    )

    # Mock: neither table exists
    async def mock_check_table_exists(table_name):
        return False

    mock_pg_db.check_table_exists = AsyncMock(side_effect=mock_check_table_exists)

    with patch(
        "lightrag.kg.postgres_impl.PGVectorStorage._pg_create_table", AsyncMock()
    ) as mock_create:
        await storage.initialize()

        # Verify table name format
        assert "text_embedding_3_large_3072d" in storage.table_name

        # Verify new table creation was called
        mock_create.assert_called_once()
        call_args = mock_create.call_args
        assert (
            call_args[0][1] == storage.table_name
        )  # table_name is second positional arg


async def test_scenario_2_legacy_upgrade_migration(
    mock_client_manager, mock_pg_db, mock_embedding_func
):
    """
    Scenario 2: Upgrade from legacy version

    Expected behavior:
    - Legacy table exists (without model suffix)
    - New table doesn't exist
    - Automatically migrate data to new table with suffix
    """
    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
    }

    embedding_func = EmbeddingFunc(
        embedding_dim=1536,
        func=mock_embedding_func.func,
        model_name="text-embedding-ada-002",
    )

    storage = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        global_config=config,
        embedding_func=embedding_func,
        workspace="legacy_workspace",
    )

    # Mock: only legacy table exists
    async def mock_check_table_exists(table_name):
        return table_name == storage.legacy_table_name

    mock_pg_db.check_table_exists = AsyncMock(side_effect=mock_check_table_exists)

    # Mock: legacy table has 50 records
    mock_rows = [
        {
            "id": f"legacy_id_{i}",
            "content": f"legacy_content_{i}",
            "workspace": "legacy_workspace",
        }
        for i in range(50)
    ]

    # Track which queries have been made for proper response
    query_history = []
    migration_state = {"new_table_count": 0}

    async def mock_query(sql, params=None, multirows=False, **kwargs):
        query_history.append(sql)

        if "COUNT(*)" in sql:
            # Determine table type:
            # - Legacy: contains base name but NOT model suffix
            # - New: contains model suffix (e.g., text_embedding_ada_002_1536d)
            sql_upper = sql.upper()
            base_name = storage.legacy_table_name.upper()

            # Check if this is querying the new table (has model suffix)
            has_model_suffix = storage.table_name.upper() in sql_upper

            is_legacy_table = base_name in sql_upper and not has_model_suffix
            has_workspace_filter = "WHERE workspace" in sql

            if is_legacy_table and has_workspace_filter:
                # Count for legacy table with workspace filter (before migration)
                return {"count": 50}
            elif is_legacy_table and not has_workspace_filter:
                # Total count for legacy table
                return {"count": 50}
            else:
                # New table count (before/after migration)
                return {"count": migration_state["new_table_count"]}
        elif multirows and "SELECT *" in sql:
            # Mock batch fetch for migration using keyset pagination
            # New pattern: WHERE workspace = $1 AND id > $2 ORDER BY id LIMIT $3
            # or first batch: WHERE workspace = $1 ORDER BY id LIMIT $2
            if "WHERE workspace" in sql:
                if "id >" in sql:
                    # Keyset pagination: params = [workspace, last_id, limit]
                    last_id = params[1] if len(params) > 1 else None
                    # Find rows after last_id
                    start_idx = 0
                    for i, row in enumerate(mock_rows):
                        if row["id"] == last_id:
                            start_idx = i + 1
                            break
                    limit = params[2] if len(params) > 2 else 500
                else:
                    # First batch (no last_id): params = [workspace, limit]
                    start_idx = 0
                    limit = params[1] if len(params) > 1 else 500
            else:
                # No workspace filter with keyset
                if "id >" in sql:
                    last_id = params[0] if params else None
                    start_idx = 0
                    for i, row in enumerate(mock_rows):
                        if row["id"] == last_id:
                            start_idx = i + 1
                            break
                    limit = params[1] if len(params) > 1 else 500
                else:
                    start_idx = 0
                    limit = params[0] if params else 500
            end = min(start_idx + limit, len(mock_rows))
            return mock_rows[start_idx:end]
        return {}

    mock_pg_db.query = AsyncMock(side_effect=mock_query)

    # Track migration through its server-side copy statement
    migration_executed = []

    def record_migration(sql, params):
        # Track that migration batch operation was called
        migration_executed.append(True)
        migration_state["new_table_count"] = 50

    mock_pg_db.query = AsyncMock(
        side_effect=answer_legacy_migration(
            mock_pg_db.query.side_effect, record_migration
        )
    )

    with patch(
        "lightrag.kg.postgres_impl.PGVectorStorage._pg_create_table", AsyncMock()
    ) as mock_create:
        await storage.initialize()

        # Verify table name contains ada-002
        assert "text_embedding_ada_002_1536d" in storage.table_name

        # Verify migration was executed
        assert len(migration_executed) > 0, "Migration should have been executed"
        mock_create.assert_called_once()


async def test_scenario_3_multi_model_coexistence(
    mock_client_manager, mock_pg_db, mock_embedding_func
):
    """
    Scenario 3: Multiple embedding models coexist

    Expected behavior:
    - Different embedding models create separate tables
    - Tables are isolated by model suffix
    - No interference between different models
    """
    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
    }

    # Workspace A: uses bge-small (768d)
    embedding_func_a = EmbeddingFunc(
        embedding_dim=768, func=mock_embedding_func.func, model_name="bge-small"
    )

    storage_a = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        global_config=config,
        embedding_func=embedding_func_a,
        workspace="workspace_a",
    )

    # Workspace B: uses bge-large (1024d)
    async def embed_func_b(texts, **kwargs):
        return np.array([[0.1] * 1024 for _ in texts])

    embedding_func_b = EmbeddingFunc(
        embedding_dim=1024, func=embed_func_b, model_name="bge-large"
    )

    storage_b = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        global_config=config,
        embedding_func=embedding_func_b,
        workspace="workspace_b",
    )

    # Verify different table names
    assert storage_a.table_name != storage_b.table_name
    assert "bge_small_768d" in storage_a.table_name
    assert "bge_large_1024d" in storage_b.table_name

    # Mock: both tables don't exist yet
    async def mock_check_table_exists(table_name):
        return False

    mock_pg_db.check_table_exists = AsyncMock(side_effect=mock_check_table_exists)

    with patch(
        "lightrag.kg.postgres_impl.PGVectorStorage._pg_create_table", AsyncMock()
    ) as mock_create:
        # Initialize both storages
        await storage_a.initialize()
        await storage_b.initialize()

        # Verify two separate tables were created
        assert mock_create.call_count == 2

        # Verify table names are different
        call_args_list = mock_create.call_args_list
        table_names = [call[0][1] for call in call_args_list]  # Second positional arg
        assert len(set(table_names)) == 2  # Two unique table names
        assert storage_a.table_name in table_names
        assert storage_b.table_name in table_names


async def test_case1_empty_legacy_auto_cleanup(
    mock_client_manager, mock_pg_db, mock_embedding_func
):
    """
    Case 1a: Both new and legacy tables exist, but legacy is EMPTY
    Expected: Automatically delete empty legacy table (safe cleanup)
    """
    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
    }

    embedding_func = EmbeddingFunc(
        embedding_dim=1536,
        func=mock_embedding_func.func,
        model_name="test-model",
    )

    storage = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        global_config=config,
        embedding_func=embedding_func,
        workspace="test_ws",
    )

    # Mock: Both tables exist
    async def mock_check_table_exists(table_name):
        return True  # Both new and legacy exist

    mock_pg_db.check_table_exists = AsyncMock(side_effect=mock_check_table_exists)

    # Mock: Legacy table is empty (0 records)
    async def mock_query(sql, params=None, multirows=False, **kwargs):
        if "COUNT(*)" in sql:
            if storage.legacy_table_name in sql:
                return {"count": 0}  # Empty legacy table
            else:
                return {"count": 100}  # New table has data
        return {}

    mock_pg_db.query = AsyncMock(side_effect=mock_query)

    with patch("lightrag.kg.postgres_impl.logger"):
        await storage.initialize()

        # Verify: Empty legacy table should be automatically cleaned up
        # Empty tables are safe to delete without data loss risk
        delete_calls = [
            call
            for call in mock_pg_db.execute.call_args_list
            if call[0][0] and "DROP TABLE" in call[0][0]
        ]
        assert len(delete_calls) >= 1, "Empty legacy table should be auto-deleted"
        # Check if legacy table was dropped
        dropped_table = storage.legacy_table_name
        assert any(dropped_table in str(call) for call in delete_calls), (
            f"Expected to drop empty legacy table '{dropped_table}'"
        )

        print(
            f"✅ Case 1a: Empty legacy table '{dropped_table}' auto-deleted successfully"
        )


async def test_case1_nonempty_legacy_warning(
    mock_client_manager, mock_pg_db, mock_embedding_func
):
    """
    Case 1b: Both new and legacy tables exist, and legacy HAS DATA
    Expected: Log warning, do not delete legacy (preserve data)
    """
    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
    }

    embedding_func = EmbeddingFunc(
        embedding_dim=1536,
        func=mock_embedding_func.func,
        model_name="test-model",
    )

    storage = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        global_config=config,
        embedding_func=embedding_func,
        workspace="test_ws",
    )

    # Mock: Both tables exist
    async def mock_check_table_exists(table_name):
        return True  # Both new and legacy exist

    mock_pg_db.check_table_exists = AsyncMock(side_effect=mock_check_table_exists)

    # Mock: Legacy table has data (50 records)
    async def mock_query(sql, params=None, multirows=False, **kwargs):
        if "COUNT(*)" in sql:
            if storage.legacy_table_name in sql:
                return {"count": 50}  # Legacy has data
            else:
                return {"count": 100}  # New table has data
        return {}

    mock_pg_db.query = AsyncMock(side_effect=mock_query)

    with patch("lightrag.kg.postgres_impl.logger"):
        await storage.initialize()

        # Verify: Legacy table with data should be preserved
        # We never auto-delete tables that contain data to prevent accidental data loss
        delete_calls = [
            call
            for call in mock_pg_db.execute.call_args_list
            if call[0][0] and "DROP TABLE" in call[0][0]
        ]
        # Check if legacy table was deleted (it should not be)
        dropped_table = storage.legacy_table_name
        legacy_deleted = any(dropped_table in str(call) for call in delete_calls)
        assert not legacy_deleted, "Legacy table with data should NOT be auto-deleted"

        print(
            f"✅ Case 1b: Legacy table '{dropped_table}' with data preserved (warning only)"
        )


async def test_case1_sequential_workspace_migration(
    mock_client_manager, mock_pg_db, mock_embedding_func
):
    """
    Case 1c: Sequential workspace migration (Multi-tenant scenario)

    Critical bug fix verification:
    Timeline:
    1. Legacy table has workspace_a (3 records) + workspace_b (3 records)
    2. Workspace A initializes first → Case 3 (only legacy exists) → migrates A's data
    3. Workspace B initializes later → Case 3 (both tables exist, legacy has B's data) → should migrate B's data
    4. Verify workspace B's data is correctly migrated to new table

    This test verifies the migration logic correctly handles multi-tenant scenarios
    where different workspaces migrate sequentially.
    """
    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
    }

    embedding_func = EmbeddingFunc(
        embedding_dim=1536,
        func=mock_embedding_func.func,
        model_name="test-model",
    )

    # Mock data: Legacy table has 6 records total (3 from workspace_a, 3 from workspace_b)
    mock_rows_a = [
        {"id": f"a_{i}", "content": f"A content {i}", "workspace": "workspace_a"}
        for i in range(3)
    ]
    mock_rows_b = [
        {"id": f"b_{i}", "content": f"B content {i}", "workspace": "workspace_b"}
        for i in range(3)
    ]

    # Track migration state
    migration_state = {
        "new_table_exists": False,
        "workspace_a_migrated": False,
        "workspace_a_migration_count": 0,
        "workspace_b_migration_count": 0,
    }

    # Step 1: Simulate workspace_a initialization (Case 3 - only legacy exists)
    # CRITICAL: Set db.workspace to workspace_a
    mock_pg_db.workspace = "workspace_a"

    storage_a = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        global_config=config,
        embedding_func=embedding_func,
        workspace="workspace_a",
    )

    # Mock table_exists for workspace_a
    async def mock_check_table_exists_a(table_name):
        if table_name == storage_a.legacy_table_name:
            return True
        if table_name == storage_a.table_name:
            return migration_state["new_table_exists"]
        return False

    mock_pg_db.check_table_exists = AsyncMock(side_effect=mock_check_table_exists_a)

    # Mock query for workspace_a (Case 3)
    async def mock_query_a(sql, params=None, multirows=False, **kwargs):
        sql_upper = sql.upper()
        base_name = storage_a.legacy_table_name.upper()

        if "COUNT(*)" in sql:
            has_model_suffix = "TEST_MODEL_1536D" in sql_upper
            is_legacy = base_name in sql_upper and not has_model_suffix
            has_workspace_filter = "WHERE workspace" in sql

            if is_legacy and has_workspace_filter:
                workspace = params[0] if params and len(params) > 0 else None
                if workspace == "workspace_a":
                    return {"count": 3}
                elif workspace == "workspace_b":
                    return {"count": 3}
            elif is_legacy and not has_workspace_filter:
                # Global count in legacy table
                return {"count": 6}
            elif has_model_suffix:
                if has_workspace_filter:
                    workspace = params[0] if params and len(params) > 0 else None
                    if workspace == "workspace_a":
                        return {"count": migration_state["workspace_a_migration_count"]}
                    if workspace == "workspace_b":
                        return {"count": migration_state["workspace_b_migration_count"]}
                return {
                    "count": migration_state["workspace_a_migration_count"]
                    + migration_state["workspace_b_migration_count"]
                }
        elif multirows and "SELECT *" in sql:
            if "WHERE workspace" in sql:
                workspace = params[0] if params and len(params) > 0 else None
                if workspace == "workspace_a":
                    # Handle keyset pagination
                    if "id >" in sql:
                        # params = [workspace, last_id, limit]
                        last_id = params[1] if len(params) > 1 else None
                        start_idx = 0
                        for i, row in enumerate(mock_rows_a):
                            if row["id"] == last_id:
                                start_idx = i + 1
                                break
                        limit = params[2] if len(params) > 2 else 500
                    else:
                        # First batch: params = [workspace, limit]
                        start_idx = 0
                        limit = params[1] if len(params) > 1 else 500
                    end = min(start_idx + limit, len(mock_rows_a))
                    return mock_rows_a[start_idx:end]
        return {}

    mock_pg_db.query = AsyncMock(side_effect=mock_query_a)

    # Track migration through its server-side copy statement
    migration_a_executed = []

    def record_migration_a(sql, params):
        migration_a_executed.append(True)
        migration_state["workspace_a_migration_count"] = len(mock_rows_a)

    mock_pg_db.query = AsyncMock(
        side_effect=answer_legacy_migration(
            mock_pg_db.query.side_effect, record_migration_a
        )
    )

    # Initialize workspace_a (Case 3)
    with patch("lightrag.kg.postgres_impl.logger"):
        await storage_a.initialize()
        migration_state["new_table_exists"] = True
        migration_state["workspace_a_migrated"] = True

    print("✅ Step 1: Workspace A initialized")
    # Verify migration was executed
    assert len(migration_a_executed) > 0, (
        "Migration should have been executed for workspace_a"
    )
    print(f"✅ Step 1: Migration executed {len(migration_a_executed)} batch(es)")

    # Step 2: Simulate workspace_b initialization (Case 3 - both exist, but legacy has B's data)
    # CRITICAL: Set db.workspace to workspace_b
    mock_pg_db.workspace = "workspace_b"

    storage_b = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        global_config=config,
        embedding_func=embedding_func,
        workspace="workspace_b",
    )

    mock_pg_db.reset_mock()

    # Mock table_exists for workspace_b (both exist)
    async def mock_check_table_exists_b(table_name):
        return True  # Both tables exist

    mock_pg_db.check_table_exists = AsyncMock(side_effect=mock_check_table_exists_b)

    # Mock query for workspace_b (Case 3)
    async def mock_query_b(sql, params=None, multirows=False, **kwargs):
        sql_upper = sql.upper()
        base_name = storage_b.legacy_table_name.upper()

        if "COUNT(*)" in sql:
            has_model_suffix = "TEST_MODEL_1536D" in sql_upper
            is_legacy = base_name in sql_upper and not has_model_suffix
            has_workspace_filter = "WHERE workspace" in sql

            if is_legacy and has_workspace_filter:
                workspace = params[0] if params and len(params) > 0 else None
                if workspace == "workspace_b":
                    return {"count": 3}  # workspace_b still has data in legacy
                elif workspace == "workspace_a":
                    return {"count": 0}  # workspace_a already migrated
            elif is_legacy and not has_workspace_filter:
                # Global count: only workspace_b data remains
                return {"count": 3}
            elif has_model_suffix:
                if has_workspace_filter:
                    workspace = params[0] if params and len(params) > 0 else None
                    if workspace == "workspace_b":
                        return {"count": migration_state["workspace_b_migration_count"]}
                    elif workspace == "workspace_a":
                        return {"count": 3}
                else:
                    return {"count": 3 + migration_state["workspace_b_migration_count"]}
        elif multirows and "SELECT *" in sql:
            if "WHERE workspace" in sql:
                workspace = params[0] if params and len(params) > 0 else None
                if workspace == "workspace_b":
                    # Handle keyset pagination
                    if "id >" in sql:
                        # params = [workspace, last_id, limit]
                        last_id = params[1] if len(params) > 1 else None
                        start_idx = 0
                        for i, row in enumerate(mock_rows_b):
                            if row["id"] == last_id:
                                start_idx = i + 1
                                break
                        limit = params[2] if len(params) > 2 else 500
                    else:
                        # First batch: params = [workspace, limit]
                        start_idx = 0
                        limit = params[1] if len(params) > 1 else 500
                    end = min(start_idx + limit, len(mock_rows_b))
                    return mock_rows_b[start_idx:end]
        return {}

    mock_pg_db.query = AsyncMock(side_effect=mock_query_b)

    # Track migration through its server-side copy statement
    migration_b_executed = []

    def record_migration_b(sql, params):
        migration_b_executed.append(True)
        migration_state["workspace_b_migration_count"] = len(mock_rows_b)

    mock_pg_db.query = AsyncMock(
        side_effect=answer_legacy_migration(
            mock_pg_db.query.side_effect, record_migration_b
        )
    )

    # Initialize workspace_b (Case 3 - both tables exist)
    with patch("lightrag.kg.postgres_impl.logger"):
        await storage_b.initialize()

    print("✅ Step 2: Workspace B initialized")

    # Verify workspace_b migration happens when new table has no workspace_b data
    # but legacy table still has workspace_b data.
    assert len(migration_b_executed) > 0, (
        "Migration should have been executed for workspace_b"
    )
    print("✅ Step 2: Migration executed for workspace_b")

    print("\n🎉 Case 1c: Sequential workspace migration verification complete!")
    print("   - Workspace A: Migrated successfully (only legacy existed)")
    print("   - Workspace B: Migrated successfully (new table empty for workspace_b)")


# ---------------------------------------------------------------------------
# Legacy vector migration: segmented copy, in-progress marker, resume
# ---------------------------------------------------------------------------


_RESTART_STATE_TABLE = pg_impl.LEGACY_VECTOR_MIGRATION_STATE_TABLE
_RESTART_COLUMNS = ["id", "workspace", "content_vector"]
_RESTART_SETUP_ARGS = dict(
    table_name="target",
    workspace="tenant",
    embedding_dim=3,
    legacy_table_name="legacy",
    base_table="LIGHTRAG_VDB_CHUNKS",
)


class _RestartFakeServer:
    """Committed database state, shared by every _RestartFakeDB facade."""

    def __init__(self, rows: int = 750):
        self.source = {f"row-{i:04}": [1.0, 0.0, float(i)] for i in range(rows)}
        self.target: dict[str, list[float]] = {}
        self.state_table_exists = False
        self.markers: set[tuple[str, str]] = set()
        self.fail_on_id: str | None = None
        self.after_segment = None
        self.events: list[str] = []
        self.segment_sql: list[str] = []


class _RestartFakeDB:
    vector_index_type = None

    def __init__(self, server: _RestartFakeServer):
        self.server = server
        self._create_vector_index = AsyncMock()

    async def check_table_exists(self, name: str) -> bool:
        if name == _RESTART_STATE_TABLE:
            return self.server.state_table_exists
        return True

    async def query(self, sql, params=None, multirows=False, **kwargs):
        server = self.server
        if "pg_attribute" in sql:
            assert params == ["legacy"]
            return [{"attname": name} for name in _RESTART_COLUMNS]
        if "in_progress" in sql:
            return {"in_progress": True} if tuple(params) in server.markers else None
        if sql.startswith("WITH guard"):
            return self._copy_segment(sql, params)
        if "AS cleared" in sql:
            return self._verify_and_clear(params)
        assert "COUNT(*)" in sql, sql
        if "FROM legacy" in sql:
            return {"count": len(server.source)}
        return {"count": len(server.target)}

    def _copy_segment(self, sql, params):
        server = self.server
        server.segment_sql.append(sql)
        if "id > $3" in sql:
            workspace, target, last_id, limit = params
        else:
            (workspace, target, limit), last_id = params, None
        assert workspace == "tenant"
        if (target, workspace) not in server.markers:
            server.events.append("segment:fenced")
            return {"batch_rows": 0, "last_id": None, "active": False}
        ids = sorted(i for i in server.source if last_id is None or i > last_id)
        batch = ids[:limit]
        if server.fail_on_id in batch:
            raise RuntimeError("injected segment failure")
        for row_id in batch:
            server.target.setdefault(row_id, server.source[row_id])
        server.events.append(f"segment:{len(batch)}")
        if server.after_segment:
            server.after_segment()
        return {
            "batch_rows": len(batch),
            "last_id": batch[-1] if batch else None,
            "active": True,
        }

    def _verify_and_clear(self, params):
        server = self.server
        workspace, target = params
        server.events.append("verify")
        missing = len(set(server.source) - set(server.target))
        cleared = int(missing == 0 and (target, workspace) in server.markers)
        if cleared:
            server.markers.discard((target, workspace))
            server.events.append("marker:clear")
        return {"missing": missing, "cleared": cleared}

    async def execute(self, sql, data=None, **kwargs):
        server = self.server
        if sql.startswith("CREATE TABLE"):
            assert _RESTART_STATE_TABLE in sql
            server.state_table_exists = True
        elif sql.startswith(f"INSERT INTO {_RESTART_STATE_TABLE}"):
            assert "WHERE NOT EXISTS (SELECT 1 FROM target WHERE workspace = $2)" in sql
            if not server.target:
                server.markers.add((data["target_table"], data["workspace"]))
                server.events.append("marker:set")
        else:
            raise AssertionError(f"unexpected statement: {sql}")


@pytest.mark.asyncio
class TestPostgresLegacyVectorMigrationRestart:
    @pytest.fixture
    def small_batches(self, monkeypatch):
        monkeypatch.setattr(pg_impl, "LEGACY_VECTOR_MIGRATION_BATCH_SIZE", 200)

    async def test_interrupted_migration_is_resumed_not_accepted(self, small_batches):
        server = _RestartFakeServer()
        server.fail_on_id = "row-0600"
        with pytest.raises(DataMigrationError):
            await PGVectorStorage.setup_table(
                db=_RestartFakeDB(server), **_RESTART_SETUP_ARGS
            )
        assert len(server.target) == 600
        assert server.markers == {("target", "tenant")}

        with pytest.raises(DataMigrationError):
            await PGVectorStorage.setup_table(
                db=_RestartFakeDB(server), **_RESTART_SETUP_ARGS
            )
        assert len(server.target) == 600

        server.fail_on_id = None
        await PGVectorStorage.setup_table(
            db=_RestartFakeDB(server), **_RESTART_SETUP_ARGS
        )
        assert server.target == server.source
        assert server.markers == set()

        server.events.clear()
        await PGVectorStorage.setup_table(
            db=_RestartFakeDB(server), **_RESTART_SETUP_ARGS
        )
        assert server.events == []

    async def test_marker_brackets_the_copy_and_verification(self, small_batches):
        server = _RestartFakeServer()
        await PGVectorStorage.setup_table(
            db=_RestartFakeDB(server), **_RESTART_SETUP_ARGS
        )
        assert server.events == [
            "marker:set",
            "segment:200",
            "segment:200",
            "segment:200",
            "segment:150",
            "verify",
            "marker:clear",
        ]

    async def test_segments_are_server_side_keyset_statements(self, small_batches):
        server = _RestartFakeServer()
        await PGVectorStorage.setup_table(
            db=_RestartFakeDB(server), **_RESTART_SETUP_ARGS
        )
        names = '"id", "workspace", "content_vector"'
        first, later = server.segment_sql[0], server.segment_sql[1]
        assert "id >" not in first
        assert "AND id > $3 ORDER BY id LIMIT $4" in later
        for sql in (first, later):
            assert (
                f"FROM {_RESTART_STATE_TABLE} WHERE target_table = $2 AND workspace = $1"
                in sql
            )
            assert "FOR SHARE" in sql
            assert "AND EXISTS (SELECT 1 FROM guard)" in sql
            assert f"INSERT INTO target ({names}) SELECT {names} FROM batch" in sql
            assert f"SELECT {names} FROM legacy WHERE workspace = $1" in sql
            assert "ON CONFLICT (workspace, id) DO NOTHING" in sql

    async def test_exact_multiple_of_batch_size_terminates(self, small_batches):
        server = _RestartFakeServer(rows=400)
        await PGVectorStorage.setup_table(
            db=_RestartFakeDB(server), **_RESTART_SETUP_ARGS
        )
        assert server.target == server.source
        assert server.events.count("segment:200") == 2
        assert server.events[-3:] == ["segment:0", "verify", "marker:clear"]

    async def test_failed_verification_keeps_the_marker(self, small_batches):
        server = _RestartFakeServer()
        server.after_segment = lambda: server.target.pop("row-0001", None)
        with pytest.raises(DataMigrationError, match="verification failed"):
            await PGVectorStorage.setup_table(
                db=_RestartFakeDB(server), **_RESTART_SETUP_ARGS
            )
        assert server.markers == {("target", "tenant")}

        server.after_segment = None
        await PGVectorStorage.setup_table(
            db=_RestartFakeDB(server), **_RESTART_SETUP_ARGS
        )
        assert server.target == server.source
        assert server.markers == set()

    async def test_cancellation_leaves_a_resumable_marker(self, small_batches):
        server = _RestartFakeServer()

        def cancel():
            raise asyncio.CancelledError()

        server.after_segment = cancel
        with pytest.raises(asyncio.CancelledError):
            await PGVectorStorage.setup_table(
                db=_RestartFakeDB(server), **_RESTART_SETUP_ARGS
            )
        assert len(server.target) == 200
        assert server.markers == {("target", "tenant")}

    async def test_resume_does_not_overwrite_rows_already_in_the_target(
        self, small_batches
    ):
        server = _RestartFakeServer()
        server.state_table_exists = True
        server.markers.add(("target", "tenant"))
        server.target["row-0003"] = [9.0, 9.0, 9.0]
        await PGVectorStorage.setup_table(
            db=_RestartFakeDB(server), **_RESTART_SETUP_ARGS
        )
        assert len(server.target) == 750
        assert server.target["row-0003"] == [9.0, 9.0, 9.0]

    async def test_follower_stops_when_another_instance_finishes(self, small_batches):
        server = _RestartFakeServer()
        server.state_table_exists = True
        server.markers.add(("target", "tenant"))
        segments = 0

        def leader_finishes_and_serves_a_deletion():
            nonlocal segments
            segments += 1
            if segments == 1:
                server.target = dict(server.source)
                server.markers.clear()
                del server.target["row-0700"]

        server.after_segment = leader_finishes_and_serves_a_deletion
        await PGVectorStorage.setup_table(
            db=_RestartFakeDB(server), **_RESTART_SETUP_ARGS
        )
        assert "row-0700" not in server.target
        assert len(server.target) == 749
        assert server.events == ["segment:200", "segment:fenced"]

    async def test_deletion_between_last_segment_and_verification_is_not_a_failure(
        self, small_batches
    ):
        server = _RestartFakeServer(rows=150)
        server.state_table_exists = True
        server.markers.add(("target", "tenant"))

        def leader_finishes_and_serves_a_deletion():
            server.markers.clear()
            del server.target["row-0001"]

        server.after_segment = leader_finishes_and_serves_a_deletion
        await PGVectorStorage.setup_table(
            db=_RestartFakeDB(server), **_RESTART_SETUP_ARGS
        )
        assert "row-0001" not in server.target
        assert server.markers == set()

    async def test_resume_never_recreates_the_marker(self, small_batches):
        server = _RestartFakeServer()
        server.state_table_exists = True
        await PGVectorStorage._pg_migrate_workspace_data(
            _RestartFakeDB(server), "legacy", "target", "tenant", 750, resume=True
        )
        assert server.markers == set()
        assert server.target == {}
        assert server.events == ["segment:fenced"]

    async def test_fresh_start_does_not_mark_a_target_populated_meanwhile(
        self, small_batches
    ):
        server = _RestartFakeServer()
        server.target = {"row-0000": server.source["row-0000"]}
        await PGVectorStorage._pg_migrate_workspace_data(
            _RestartFakeDB(server), "legacy", "target", "tenant", 750
        )
        assert server.markers == set()
        assert len(server.target) == 1
        assert server.events == ["segment:fenced"]

    async def test_populated_target_without_marker_is_left_alone(self, lightrag_caplog):
        server = _RestartFakeServer()
        server.target = {"row-0000": [9.0, 9.0, 9.0]}
        await PGVectorStorage.setup_table(
            db=_RestartFakeDB(server), **_RESTART_SETUP_ARGS
        )
        assert server.target == {"row-0000": [9.0, 9.0, 9.0]}
        assert server.events == []
        assert "fewer than the 750" in lightrag_caplog.text

    async def test_progress_is_logged_per_segment(self, small_batches, lightrag_caplog):
        server = _RestartFakeServer()
        await PGVectorStorage.setup_table(
            db=_RestartFakeDB(server), **_RESTART_SETUP_ARGS
        )
        for done in (200, 400, 600, 750):
            assert f"{done}/750 records migrated" in lightrag_caplog.text

    async def test_malformed_segment_result_is_not_read_as_end_of_data(self):
        server = _RestartFakeServer()
        db = _RestartFakeDB(server)
        db._copy_segment = lambda sql, params: {}
        with pytest.raises(DataMigrationError, match="segment result"):
            await PGVectorStorage.setup_table(db=db, **_RESTART_SETUP_ARGS)
        assert server.markers == {("target", "tenant")}

    async def test_helper_rejects_empty_workspace_before_touching_the_database(self):
        server = _RestartFakeServer()
        with pytest.raises(ValueError, match="workspace must be provided"):
            await PGVectorStorage._pg_migrate_workspace_data(
                _RestartFakeDB(server), "legacy", "target", "", 750
            )
        assert server.events == [] and not server.state_table_exists
