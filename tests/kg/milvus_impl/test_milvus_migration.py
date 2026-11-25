"""Tests for Milvus schema-migration resilience, retry, memory back-pressure, and recovery.

Covers:
1. Connection-class retry with client rebuild and capped backoff vs. fail-fast
   non-connection errors and cause/context chain classification.
2. Force-create guard (never drops an existing collection or shadows legacy/_old/_temp).
3. Memory footprint (source loaded before iterator, temp never loaded, backups
   released, periodic/final flush back-pressure, and optional inter-batch sleep).
4. In-place commit-window recovery (stale _old cleanup, orphaned _temp/_old
   startup promotion/restoration, and commit-window retry).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import grpc
import pytest
from pymilvus import MilvusException

from lightrag.kg.milvus_impl import (
    MilvusCollectionLoadError,
    MilvusVectorDBStorage,
)
from tests._sleep_patch import patch_module_sleep

pytestmark = pytest.mark.offline


class _EmbeddingFunc:
    """Minimal embedding stand-in exposing ``embedding_dim`` and ``model_name``."""

    def __init__(self, dim: int = 128, model_name: str = "text-embedding-3-small"):
        self.embedding_dim = dim
        self.model_name = model_name


def _make_model_storage(
    namespace: str = "entities", workspace: str = "test_workspace", dim: int = 128
) -> MilvusVectorDBStorage:
    return MilvusVectorDBStorage(
        namespace=namespace,
        workspace=workspace,
        global_config={
            "embedding_batch_num": 100,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.3,
            },
        },
        embedding_func=_EmbeddingFunc(dim=dim),
        meta_fields=set(),
    )


def _wire_collection_state(storage, collections):
    storage._client = MagicMock()

    def has_collection(collection_name):
        return collection_name in collections

    def create_collection(collection_name, schema):
        collections.add(collection_name)

    def drop_collection(collection_name):
        collections.discard(collection_name)

    def rename_collection(source, target):
        collections.discard(source)
        collections.add(target)

    storage._client.has_collection.side_effect = has_collection
    storage._client.create_collection.side_effect = create_collection
    storage._client.drop_collection.side_effect = drop_collection
    storage._client.rename_collection.side_effect = rename_collection
    return storage._client


def _rows(n: int = 1, start: int = 0) -> list[dict]:
    return [
        {"id": f"ent-{i}", "vector": [0.0] * 128, "content": "body"}
        for i in range(start, start + n)
    ]


def _wire_iterator(client, batches):
    """query_iterator yields the given list of row-batches then []."""

    def make_iterator(**kwargs):
        iterator = MagicMock()
        iterator.next.side_effect = list(batches) + [[]]
        return iterator

    client.query_iterator.side_effect = make_iterator


def _wire_fresh_iterator_per_attempt(client, rows=None):
    """query_iterator returns a new exhausted-after-one-batch iterator per call."""
    _wire_iterator(client, [rows if rows is not None else _rows(1)])


class _FakeRpcError(grpc.RpcError):
    """Minimal grpc.RpcError double returning a fixed status code."""

    def __init__(self, status_code):
        self._status_code = status_code

    def code(self):
        return self._status_code


_CONNECT_FAILED = MilvusException(
    code=2, message="Fail connecting to server on host:19530"
)


def _fail_specific_rename_once(client, collections, fail_source, fail_target, error):
    """Wrap the wired rename so one specific rename raises ``error`` on first call."""
    state = {"n": 0}

    def rename(source, target):
        if source == fail_source and target == fail_target:
            state["n"] += 1
            if state["n"] == 1:
                raise error
        collections.discard(source)
        collections.add(target)

    client.rename_collection.side_effect = rename


# ---------------------------------------------------------------------------
# Flush back-pressure and inter-batch throttling
# ---------------------------------------------------------------------------


class TestMigrationFlushBackpressure:
    def test_periodic_and_final_flush_by_row_interval(self):
        storage = _make_model_storage()
        storage._migration_flush_interval_rows = 3
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(storage, collections)
        temp = f"{storage.final_namespace}_temp"
        # Three iterator batches of 2 rows: cumulative 2, 4, 6.
        # Periodic flush trips once (at 4 >= 3), then a final flush at the end.
        _wire_iterator(client, [_rows(2), _rows(2, 2), _rows(2, 4)])

        with patch.object(storage, "_create_indexes_after_collection"):
            storage._migrate_collection_schema(
                source_collection_name=storage.legacy_namespace,
                target_collection_name=storage.final_namespace,
            )

        flush_targets = [call.args[0] for call in client.flush.call_args_list]
        assert flush_targets == [temp, temp]  # one periodic + one final
        assert storage.final_namespace in collections

    def test_flush_disabled_when_interval_zero(self):
        storage = _make_model_storage()
        storage._migration_flush_interval_rows = 0
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(storage, collections)
        _wire_iterator(client, [_rows(2), _rows(2, 2)])

        with patch.object(storage, "_create_indexes_after_collection"):
            storage._migrate_collection_schema(
                source_collection_name=storage.legacy_namespace,
                target_collection_name=storage.final_namespace,
            )

        client.flush.assert_not_called()
        assert storage.final_namespace in collections

    def test_no_final_flush_for_empty_source(self):
        storage = _make_model_storage()
        storage._migration_flush_interval_rows = 50000
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(storage, collections)
        _wire_iterator(client, [])  # empty source

        with patch.object(storage, "_create_indexes_after_collection"):
            storage._migrate_collection_schema(
                source_collection_name=storage.legacy_namespace,
                target_collection_name=storage.final_namespace,
            )

        client.flush.assert_not_called()

    def test_flush_connection_error_is_retryable(self):
        storage = _make_model_storage()
        storage._migration_flush_interval_rows = 1
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(storage, collections)
        _wire_iterator(client, [_rows(2)])

        flush_calls = {"n": 0}

        def flush(collection_name, **kwargs):
            flush_calls["n"] += 1
            if flush_calls["n"] == 1:
                raise MilvusException(
                    code=2, message="Fail connecting to server on host:19530"
                )

        client.flush.side_effect = flush

        with patch.object(storage, "_create_indexes_after_collection"):
            with patch.object(storage, "_rebuild_milvus_client") as rebuild:
                with patch_module_sleep("lightrag.kg.milvus_impl"):
                    storage._migrate_collection_schema(
                        source_collection_name=storage.legacy_namespace,
                        target_collection_name=storage.final_namespace,
                    )

        rebuild.assert_called_once()
        assert storage.final_namespace in collections


class TestMigrationBatchThrottle:
    def test_batch_sleep_throttles_between_batches(self):
        storage = _make_model_storage()
        storage._migration_flush_interval_rows = 0
        storage._migration_batch_sleep = 0.05
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(storage, collections)
        _wire_iterator(client, [_rows(2), _rows(2, 2)])

        with patch.object(storage, "_create_indexes_after_collection"):
            with patch_module_sleep("lightrag.kg.milvus_impl") as sleep:
                storage._migrate_collection_schema(
                    source_collection_name=storage.legacy_namespace,
                    target_collection_name=storage.final_namespace,
                )

        # One sleep per non-empty iterator batch, none for the terminating [].
        assert [call.args[0] for call in sleep.call_args_list] == [0.05, 0.05]

    def test_no_sleep_when_disabled(self):
        storage = _make_model_storage()
        storage._migration_flush_interval_rows = 0
        storage._migration_batch_sleep = 0.0
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(storage, collections)
        _wire_iterator(client, [_rows(2)])

        with patch.object(storage, "_create_indexes_after_collection"):
            with patch_module_sleep("lightrag.kg.milvus_impl") as sleep:
                storage._migrate_collection_schema(
                    source_collection_name=storage.legacy_namespace,
                    target_collection_name=storage.final_namespace,
                )

        sleep.assert_not_called()


class TestMigrationBackpressureConfig:
    @pytest.mark.parametrize(
        ("flush_env", "sleep_env", "expected_flush", "expected_sleep"),
        [
            pytest.param("1234", "0.25", 1234, 0.25, id="env-overrides"),
            pytest.param("-5", "-1", 0, 0.0, id="negative-clamped"),
        ],
    )
    def test_backpressure_env_config(
        self, monkeypatch, flush_env, sleep_env, expected_flush, expected_sleep
    ):
        monkeypatch.setenv("MILVUS_MIGRATION_FLUSH_INTERVAL_ROWS", flush_env)
        monkeypatch.setenv("MILVUS_MIGRATION_BATCH_SLEEP", sleep_env)
        storage = _make_model_storage()
        assert storage._migration_flush_interval_rows == expected_flush
        assert storage._migration_batch_sleep == expected_sleep


# ---------------------------------------------------------------------------
# Retry, error classification, and client rebuild
# ---------------------------------------------------------------------------


class TestMigrationRetry:
    def test_retryable_error_retries_with_rebuilt_client(self):
        storage = _make_model_storage()
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(storage, collections)
        _wire_fresh_iterator_per_attempt(client)

        insert_calls = {"n": 0}

        def insert(collection_name, data):
            insert_calls["n"] += 1
            if insert_calls["n"] == 1:
                raise _CONNECT_FAILED

        client.insert.side_effect = insert

        with patch.object(storage, "_create_indexes_after_collection"):
            with patch.object(storage, "_rebuild_milvus_client") as rebuild:
                with patch_module_sleep("lightrag.kg.milvus_impl") as sleep:
                    storage._migrate_collection_schema(
                        source_collection_name=storage.legacy_namespace,
                        target_collection_name=storage.final_namespace,
                    )

        rebuild.assert_called_once()
        sleep.assert_called_once()
        assert sleep.call_args.args[0] == pytest.approx(5.0)
        assert client.query_iterator.call_count == 2
        assert storage.final_namespace in collections
        assert f"{storage.final_namespace}_temp" not in collections
        assert storage.legacy_namespace in collections

    def test_non_retryable_error_fails_fast(self):
        storage = _make_model_storage()
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(storage, collections)
        _wire_fresh_iterator_per_attempt(client)
        client.insert.side_effect = RuntimeError("insert failed")

        with patch.object(storage, "_create_indexes_after_collection"):
            with patch.object(storage, "_rebuild_milvus_client") as rebuild:
                with patch_module_sleep("lightrag.kg.milvus_impl") as sleep:
                    with pytest.raises(
                        RuntimeError, match="Iterator-based migration failed"
                    ):
                        storage._migrate_collection_schema(
                            source_collection_name=storage.legacy_namespace,
                            target_collection_name=storage.final_namespace,
                        )

        rebuild.assert_not_called()
        sleep.assert_not_called()
        assert client.query_iterator.call_count == 1
        assert storage.legacy_namespace in collections
        assert storage.final_namespace not in collections

    def test_retries_exhausted_raises_with_backoff_sequence(self):
        storage = _make_model_storage()
        storage._migration_max_retries = 2
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(storage, collections)
        _wire_fresh_iterator_per_attempt(client)
        client.insert.side_effect = ValueError("Cannot invoke RPC on closed channel!")

        with patch.object(storage, "_create_indexes_after_collection"):
            with patch.object(storage, "_rebuild_milvus_client"):
                with patch_module_sleep("lightrag.kg.milvus_impl") as sleep:
                    with pytest.raises(
                        RuntimeError, match="Iterator-based migration failed"
                    ):
                        storage._migrate_collection_schema(
                            source_collection_name=storage.legacy_namespace,
                            target_collection_name=storage.final_namespace,
                        )

        assert client.query_iterator.call_count == 3
        assert [call.args[0] for call in sleep.call_args_list] == [
            pytest.approx(5.0),
            pytest.approx(15.0),
        ]
        assert storage.legacy_namespace in collections

    def test_max_backoff_is_capped(self):
        storage = _make_model_storage()
        storage._migration_max_retries = 4
        storage._migration_retry_max_backoff = 20.0
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(storage, collections)
        _wire_fresh_iterator_per_attempt(client)
        client.insert.side_effect = _CONNECT_FAILED

        with patch.object(storage, "_create_indexes_after_collection"):
            with patch.object(storage, "_rebuild_milvus_client"):
                with patch_module_sleep("lightrag.kg.milvus_impl") as sleep:
                    with pytest.raises(RuntimeError):
                        storage._migrate_collection_schema(
                            source_collection_name=storage.legacy_namespace,
                            target_collection_name=storage.final_namespace,
                        )

        assert [call.args[0] for call in sleep.call_args_list] == [
            pytest.approx(5.0),
            pytest.approx(15.0),
            pytest.approx(20.0),
            pytest.approx(20.0),
        ]


class TestRetryableErrorClassification:
    @pytest.mark.parametrize(
        "error",
        [
            _FakeRpcError(grpc.StatusCode.UNAVAILABLE),
            _FakeRpcError(grpc.StatusCode.DEADLINE_EXCEEDED),
            MilvusException(code=2, message="Fail connecting to server on host:19530"),
            MilvusException(code=1, message="server unavailable: ping timeout"),
            ValueError("Cannot invoke RPC on closed channel!"),
        ],
        ids=[
            "grpc-unavailable",
            "grpc-deadline-exceeded",
            "milvus-connect-failed",
            "milvus-connection-message",
            "closed-channel-valueerror",
        ],
    )
    def test_retryable_errors(self, error):
        assert MilvusVectorDBStorage._is_retryable_connection_error(error) is True

    @pytest.mark.parametrize(
        "error",
        [
            ValueError("primary keys cannot be truncated"),
            MilvusException(code=1100, message="schema mismatch"),
            RuntimeError("Target collection already exists: foo"),
        ],
        ids=["value-error", "milvus-schema-error", "runtime-error"],
    )
    def test_non_retryable_errors(self, error):
        assert MilvusVectorDBStorage._is_retryable_connection_error(error) is False

    def test_cause_context_and_cycle_chains(self):
        inner_cause = ValueError("Cannot invoke RPC on closed channel!")
        outer_cause = RuntimeError("Iterator-based migration failed for collection foo")
        outer_cause.__cause__ = inner_cause
        assert MilvusVectorDBStorage._is_retryable_connection_error(outer_cause) is True

        inner_ctx = MilvusException(code=2, message="Fail connecting to server")
        outer_ctx = RuntimeError("wrapper")
        outer_ctx.__context__ = inner_ctx
        assert MilvusVectorDBStorage._is_retryable_connection_error(outer_ctx) is True

        loop = RuntimeError("loop")
        loop.__cause__ = loop
        assert MilvusVectorDBStorage._is_retryable_connection_error(loop) is False


class TestRebuildMilvusClient:
    def test_rebuild_replaces_client_and_closes_old(self):
        storage = _make_model_storage()
        old_client = MagicMock()
        new_client = MagicMock()
        storage._client = old_client

        with patch.object(
            storage, "_create_milvus_client", return_value=new_client
        ) as create:
            storage._rebuild_milvus_client()

        old_client.close.assert_called_once()
        create.assert_called_once()
        assert storage._client is new_client

    def test_rebuild_tolerates_close_failure_on_dead_client(self):
        storage = _make_model_storage()
        old_client = MagicMock()
        old_client.close.side_effect = ValueError(
            "Cannot invoke RPC on closed channel!"
        )
        new_client = MagicMock()
        storage._client = old_client

        with patch.object(storage, "_create_milvus_client", return_value=new_client):
            storage._rebuild_milvus_client()

        assert storage._client is new_client


# ---------------------------------------------------------------------------
# Force-create guard
# ---------------------------------------------------------------------------


class TestForceCreateGuard:
    def test_connection_error_propagates_without_force_create(self):
        storage = _make_model_storage()
        storage._client = MagicMock()
        storage._client.has_collection.side_effect = _CONNECT_FAILED

        with pytest.raises(MilvusException):
            storage._create_collection_if_not_exist()

        storage._client.create_collection.assert_not_called()
        storage._client.drop_collection.assert_not_called()

    def test_non_connection_error_still_force_creates(self):
        storage = _make_model_storage()
        storage._client = MagicMock()
        checks = iter([KeyError("boom")])

        def has_collection(name):
            error = next(checks, None)
            if error is not None:
                raise error
            return False

        storage._client.has_collection.side_effect = has_collection

        with patch.object(storage, "_create_indexes_after_collection"):
            with patch.object(storage, "_ensure_collection_loaded"):
                storage._create_collection_if_not_exist()

        storage._client.create_collection.assert_called_once()

    def test_fallback_never_drops_an_existing_collection(self):
        storage = _make_model_storage()
        storage._client = MagicMock()
        storage._client.has_collection.side_effect = [
            MilvusException(message="server is busy"),
            True,
        ]

        with pytest.raises(RuntimeError, match="refusing to drop and recreate"):
            storage._create_collection_if_not_exist()

        storage._client.drop_collection.assert_not_called()
        storage._client.create_collection.assert_not_called()

    @pytest.mark.parametrize("leftover", ["legacy", "_old", "_temp"])
    def test_fallback_never_creates_over_data_it_would_shadow(self, leftover):
        storage = _make_model_storage()
        storage._client = MagicMock()
        existing = {
            "legacy": storage.legacy_namespace,
            "_old": f"{storage.final_namespace}_old",
            "_temp": f"{storage.final_namespace}_temp",
        }[leftover]
        checks = iter([MilvusException(message="server is busy")])

        def has_collection(name):
            error = next(checks, None)
            if error is not None:
                raise error
            return name == existing

        storage._client.has_collection.side_effect = has_collection

        with pytest.raises(RuntimeError, match="refusing to create it over existing"):
            storage._create_collection_if_not_exist()

        storage._client.drop_collection.assert_not_called()
        storage._client.create_collection.assert_not_called()

    def test_fallback_fails_closed_when_existence_stays_unknown(self):
        storage = _make_model_storage()
        storage._client = MagicMock()
        storage._client.has_collection.side_effect = MilvusException(
            message="server is busy"
        )

        with pytest.raises(MilvusException, match="server is busy"):
            storage._create_collection_if_not_exist()

        storage._client.drop_collection.assert_not_called()
        storage._client.create_collection.assert_not_called()

    @pytest.mark.parametrize(
        ("message", "match"),
        [
            pytest.param(
                "call query coordinator LoadCollection: resource group node "
                "not enough[rg=__default_resource_group][currentNodeNum=0]",
                "resource group node not enough",
                id="resource-group-refused",
            ),
            pytest.param(
                "wait for loading collection timeout, collection: x",
                "loading collection timeout",
                id="load-timeout",
            ),
        ],
    )
    def test_load_failure_on_existing_collection_never_drops_or_force_creates(
        self, message, match
    ):
        storage = _make_model_storage()
        storage._client = MagicMock()
        storage._client.has_collection.side_effect = lambda name: (
            name == storage.final_namespace
        )
        storage._client.load_collection.side_effect = MilvusException(message=message)

        with patch.object(storage, "_validate_collection_compatibility"):
            with pytest.raises(MilvusCollectionLoadError, match=match) as exc_info:
                storage._create_collection_if_not_exist()

        assert isinstance(exc_info.value.__cause__, MilvusException)
        storage._client.drop_collection.assert_not_called()
        storage._client.create_collection.assert_not_called()


# ---------------------------------------------------------------------------
# Memory footprint & backup release
# ---------------------------------------------------------------------------


class TestMigrationMemoryFootprint:
    def test_temp_collection_is_not_loaded_during_migration(self):
        storage = _make_model_storage()
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(storage, collections)
        _wire_fresh_iterator_per_attempt(client)

        with patch.object(storage, "_create_indexes_after_collection"):
            storage._migrate_collection_schema(
                source_collection_name=storage.legacy_namespace,
                target_collection_name=storage.final_namespace,
            )

        loaded = [call.args[0] for call in client.load_collection.call_args_list]
        assert f"{storage.final_namespace}_temp" not in loaded
        assert storage.legacy_namespace in loaded
        assert storage.final_namespace in collections

    def test_source_is_loaded_before_query_iterator(self):
        storage = _make_model_storage()
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(storage, collections)

        order = []
        client.load_collection.side_effect = lambda name, **kw: order.append(
            ("load", name)
        )

        def make_iterator(**kwargs):
            order.append(("query_iterator", kwargs["collection_name"]))
            iterator = MagicMock()
            iterator.next.side_effect = [_rows(1), []]
            return iterator

        client.query_iterator.side_effect = make_iterator

        with patch.object(storage, "_create_indexes_after_collection"):
            storage._migrate_collection_schema(
                source_collection_name=storage.legacy_namespace,
                target_collection_name=storage.final_namespace,
            )

        assert ("load", storage.legacy_namespace) in order
        assert order.index(("load", storage.legacy_namespace)) < order.index(
            ("query_iterator", storage.legacy_namespace)
        )

    def test_failed_suffix_migration_releases_loaded_source(self):
        storage = _make_model_storage()
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(storage, collections)
        _wire_fresh_iterator_per_attempt(client)
        client.insert.side_effect = RuntimeError("insert failed")

        with patch.object(storage, "_create_indexes_after_collection"):
            with pytest.raises(RuntimeError, match="Iterator-based migration failed"):
                storage._migrate_collection_schema(
                    source_collection_name=storage.legacy_namespace,
                    target_collection_name=storage.final_namespace,
                )

        client.release_collection.assert_any_call(storage.legacy_namespace)

    def test_failed_inplace_migration_does_not_release_active_source(self):
        storage = _make_model_storage()
        collections = {storage.final_namespace}
        client = _wire_collection_state(storage, collections)
        _wire_fresh_iterator_per_attempt(client)
        client.insert.side_effect = RuntimeError("insert failed")

        with patch.object(storage, "_create_indexes_after_collection"):
            with pytest.raises(RuntimeError, match="Iterator-based migration failed"):
                storage._migrate_collection_schema()  # in-place

        client.release_collection.assert_not_called()

    def test_suffix_migration_releases_legacy_backup(self):
        storage = _make_model_storage()
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(storage, collections)
        _wire_fresh_iterator_per_attempt(client)

        with patch.object(storage, "_create_indexes_after_collection"):
            storage._migrate_collection_schema(
                source_collection_name=storage.legacy_namespace,
                target_collection_name=storage.final_namespace,
            )

        client.release_collection.assert_called_once_with(storage.legacy_namespace)
        assert storage.legacy_namespace in collections

    def test_in_place_migration_releases_old_backup(self):
        storage = _make_model_storage()
        collections = {storage.final_namespace}
        client = _wire_collection_state(storage, collections)
        _wire_fresh_iterator_per_attempt(client)

        with patch.object(storage, "_create_indexes_after_collection"):
            storage._migrate_collection_schema()

        backup_name = f"{storage.final_namespace}_old"
        client.release_collection.assert_called_once_with(backup_name)
        assert backup_name in collections
        assert storage.final_namespace in collections

    def test_release_failure_does_not_fail_migration(self):
        storage = _make_model_storage()
        collections = {storage.legacy_namespace}
        client = _wire_collection_state(storage, collections)
        _wire_fresh_iterator_per_attempt(client)
        client.release_collection.side_effect = MilvusException(
            code=1, message="release failed"
        )

        with patch.object(storage, "_create_indexes_after_collection"):
            storage._migrate_collection_schema(
                source_collection_name=storage.legacy_namespace,
                target_collection_name=storage.final_namespace,
            )

        assert storage.final_namespace in collections


# ---------------------------------------------------------------------------
# In-place migration safety and commit-window recovery
# ---------------------------------------------------------------------------


class TestInPlaceMigrationSafety:
    def test_stale_old_backup_is_dropped_before_rename(self):
        storage = _make_model_storage()
        backup_name = f"{storage.final_namespace}_old"
        collections = {storage.final_namespace, backup_name}
        client = _wire_collection_state(storage, collections)
        _wire_fresh_iterator_per_attempt(client)

        with patch.object(storage, "_create_indexes_after_collection"):
            storage._migrate_collection_schema()

        assert backup_name in collections
        assert storage.final_namespace in collections
        drop_calls = [call.args[0] for call in client.drop_collection.call_args_list]
        assert backup_name in drop_calls

    @pytest.mark.parametrize("suffix", ["_temp", "_old"])
    def test_orphaned_backup_is_recovered_on_startup(self, suffix):
        storage = _make_model_storage()
        orphan_name = f"{storage.final_namespace}{suffix}"
        collections = {orphan_name}
        client = _wire_collection_state(storage, collections)

        with patch.object(storage, "_ensure_collection_loaded"):
            storage._create_collection_if_not_exist()

        assert storage.final_namespace in collections
        assert orphan_name not in collections
        client.create_collection.assert_not_called()

    def test_inplace_recovery_precedes_legacy_migration(self):
        storage = _make_model_storage()
        final = storage.final_namespace
        legacy = storage.legacy_namespace
        temp = f"{final}_temp"
        old = f"{final}_old"
        collections = {legacy, old, temp}
        client = _wire_collection_state(storage, collections)

        with patch.object(storage, "_ensure_collection_loaded"):
            with patch.object(storage, "_migrate_collection_schema") as migrate:
                storage._create_collection_if_not_exist()

        migrate.assert_not_called()
        assert final in collections
        assert temp not in collections
        assert legacy in collections
        client.create_collection.assert_not_called()
        drops = [call.args[0] for call in client.drop_collection.call_args_list]
        assert temp not in drops

    def test_partial_suffix_temp_with_legacy_is_remigrated_not_promoted(self):
        storage = _make_model_storage()
        final = storage.final_namespace
        legacy = storage.legacy_namespace
        temp = f"{final}_temp"
        collections = {legacy, temp}
        client = _wire_collection_state(storage, collections)

        with patch.object(storage, "_has_vector_field", return_value=True):
            with patch.object(storage, "_check_vector_dimension"):
                with patch.object(storage, "_migrate_collection_schema") as migrate:
                    with patch.object(storage, "_ensure_collection_loaded"):
                        storage._create_collection_if_not_exist()

        migrate.assert_called_once_with(
            source_collection_name=legacy,
            target_collection_name=final,
        )
        client.rename_collection.assert_not_called()


class TestInPlaceCommitWindowRecovery:
    """The in-place commit window must treat temp/_old as recoverable state."""

    def test_step4_connection_failure_retry_promotes_temp(self):
        storage = _make_model_storage()
        final = storage.final_namespace
        temp = f"{final}_temp"
        old = f"{final}_old"
        collections = {final}
        client = _wire_collection_state(storage, collections)
        _wire_fresh_iterator_per_attempt(client)
        _fail_specific_rename_once(client, collections, temp, final, _CONNECT_FAILED)

        with patch.object(storage, "_create_indexes_after_collection"):
            with patch.object(storage, "_rebuild_milvus_client") as rebuild:
                with patch_module_sleep("lightrag.kg.milvus_impl") as sleep:
                    storage._migrate_collection_schema()

        rebuild.assert_called_once()
        sleep.assert_called_once()
        assert final in collections
        assert temp not in collections
        assert old in collections
        assert storage.final_namespace == final
        drops = [call.args[0] for call in client.drop_collection.call_args_list]
        assert temp not in drops

    def test_drop_source_fallback_then_step4_failure_recovers_temp(self):
        storage = _make_model_storage()
        final = storage.final_namespace
        temp = f"{final}_temp"
        old = f"{final}_old"
        collections = {final}
        client = _wire_collection_state(storage, collections)
        _wire_fresh_iterator_per_attempt(client)
        state = {"temp_to_final": 0}

        def rename(source, target):
            if source == final and target == old:
                raise RuntimeError("rename to _old unsupported")
            if source == temp and target == final:
                state["temp_to_final"] += 1
                if state["temp_to_final"] == 1:
                    raise _CONNECT_FAILED
            collections.discard(source)
            collections.add(target)

        client.rename_collection.side_effect = rename

        with patch.object(storage, "_create_indexes_after_collection"):
            with patch.object(storage, "_rebuild_milvus_client") as rebuild:
                with patch_module_sleep("lightrag.kg.milvus_impl"):
                    storage._migrate_collection_schema()

        rebuild.assert_called_once()
        assert final in collections
        assert temp not in collections
        assert old not in collections
        assert storage.final_namespace == final
        drops = [call.args[0] for call in client.drop_collection.call_args_list]
        assert temp not in drops

    def test_non_retryable_commit_failure_keeps_temp_for_startup_recovery(self):
        storage = _make_model_storage()
        final = storage.final_namespace
        temp = f"{final}_temp"
        old = f"{final}_old"
        collections = {final}
        client = _wire_collection_state(storage, collections)
        _wire_fresh_iterator_per_attempt(client)

        def rename(source, target):
            if source == temp and target == final:
                raise RuntimeError("non-retryable rename failure")
            collections.discard(source)
            collections.add(target)

        client.rename_collection.side_effect = rename

        with patch.object(storage, "_create_indexes_after_collection"):
            with pytest.raises(RuntimeError, match="Iterator-based migration failed"):
                storage._migrate_collection_schema()

        assert temp in collections
        assert old in collections
        drops = [call.args[0] for call in client.drop_collection.call_args_list]
        assert temp not in drops
