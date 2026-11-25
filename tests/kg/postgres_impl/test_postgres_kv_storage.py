"""
Unit tests for PGKVStorage.upsert batch optimization (PR #2742 fixes).

Verifies:
1. Each namespace builds correct tuple ordering matching SQL positional params.
2. _run_with_retry is used (not the removed PostgreSQLDB.executemany wrapper).
3. Sub-batching splits data when len(data) > _max_batch_size.
4. Unknown namespace raises ValueError.
5. Empty data returns without any DB call.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from lightrag import config_anchor as ca
from lightrag import config_store as cs
from lightrag.exceptions import ConfigurationStorageError
from lightrag.kg.postgres_impl import (
    DEFAULT_PG_DELETE_MAX_RECORDS_PER_BATCH,
    DEFAULT_PG_UPSERT_MAX_PAYLOAD_BYTES,
    DEFAULT_PG_UPSERT_MAX_RECORDS_PER_BATCH,
    NAMESPACE_TABLE_MAP,
    SQL_TEMPLATES,
    TABLES,
    PGDocStatusStorage,
    PGKVStorage,
    PGVectorStorage,
    _chunk_by_budget,
    _config_row_from_pg,
    _config_row_payload,
    _estimate_record_bytes,
    _resolve_pg_batch_limits,
    namespace_to_table_name,
)
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.namespace import CONFIG_CONTAINER_TAG, NameSpace
from lightrag.utils import EmbeddingFunc


pytestmark = pytest.mark.offline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GLOBAL_CONFIG = {"embedding_batch_num": 10}


def make_storage(namespace: str) -> PGKVStorage:
    """Construct a PGKVStorage instance with a mocked db."""
    db = MagicMock()
    captured: list[tuple] = []
    captured_execute: list[tuple] = []
    retry_kwargs: list[dict] = []

    async def fake_run_with_retry(operation, **kwargs):
        """Call the closure with a mock connection to capture executemany args."""
        retry_kwargs.append(kwargs)
        mock_conn = AsyncMock()
        tx = AsyncMock()
        tx.__aenter__.return_value = tx
        tx.__aexit__.return_value = False
        mock_conn.transaction = MagicMock(return_value=tx)
        await operation(mock_conn)
        # Store (sql, data) from each executemany call
        for call in mock_conn.executemany.call_args_list:
            captured.append((call.args[0], call.args[1]))
        # Store (sql, args) from each execute call (chunked delete path).
        for call in mock_conn.execute.call_args_list:
            captured_execute.append((call.args[0], call.args[1:]))

    db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)
    db.workspace = "test_ws"

    storage = PGKVStorage.__new__(PGKVStorage)
    storage.namespace = namespace
    storage.workspace = "test_ws"
    storage.global_config = GLOBAL_CONFIG
    storage.db = db
    storage.__post_init__()

    storage._captured = captured
    storage._captured_execute = captured_execute
    storage._retry_kwargs = retry_kwargs
    return storage


def make_doc_status_storage() -> PGDocStatusStorage:
    """Construct a PGDocStatusStorage instance with a mocked db."""
    db = MagicMock()
    captured: list[tuple] = []
    captured_execute: list[tuple] = []
    retry_kwargs: list[dict] = []

    async def fake_run_with_retry(operation, **kwargs):
        retry_kwargs.append(kwargs)
        mock_conn = AsyncMock()
        tx = AsyncMock()
        tx.__aenter__.return_value = tx
        tx.__aexit__.return_value = False
        mock_conn.transaction = MagicMock(return_value=tx)
        await operation(mock_conn)
        for call in mock_conn.executemany.call_args_list:
            captured.append((call.args[0], call.args[1]))
        for call in mock_conn.execute.call_args_list:
            captured_execute.append((call.args[0], call.args[1:]))

    db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)
    db.workspace = "test_ws"

    storage = PGDocStatusStorage.__new__(PGDocStatusStorage)
    storage.namespace = NameSpace.DOC_STATUS
    storage.workspace = "test_ws"
    storage.global_config = GLOBAL_CONFIG
    storage.db = db
    storage.__post_init__()  # resolves batch-limit attrs (payload/records caps)
    storage._captured = captured
    storage._captured_execute = captured_execute
    storage._retry_kwargs = retry_kwargs
    return storage


def make_vector_storage(namespace: str) -> PGVectorStorage:
    """Construct a PGVectorStorage instance with a mocked db and embedding func."""
    db = MagicMock()
    captured: list[tuple] = []
    retry_kwargs: list[dict] = []

    async def fake_run_with_retry(operation, **kwargs):
        retry_kwargs.append(kwargs)
        mock_conn = AsyncMock()
        # connection.transaction() is a synchronous factory in asyncpg that
        # returns an async context manager. AsyncMock would return a coroutine,
        # so swap in a MagicMock that returns an AsyncMock-backed context.
        tx_cm = AsyncMock()
        tx_cm.__aenter__.return_value = None
        tx_cm.__aexit__.return_value = None
        mock_conn.transaction = MagicMock(return_value=tx_cm)
        await operation(mock_conn)
        for call in mock_conn.executemany.call_args_list:
            captured.append((call.args[0], call.args[1]))
        # Also capture connection.execute calls (delete SQL inside flush).
        for call in mock_conn.execute.call_args_list:
            captured.append((call.args[0], call.args[1:]))

    db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)
    db.workspace = "test_ws"

    async def embed_func(texts, **kwargs):
        return np.array([[0.1, 0.2, 0.3] for _ in texts], dtype=np.float32)

    embedding = EmbeddingFunc(
        embedding_dim=3,
        func=embed_func,
        model_name="test_model",
    )
    storage = PGVectorStorage(
        namespace=namespace,
        workspace="test_ws",
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.5,
            },
        },
        embedding_func=embedding,
    )
    storage.db = db
    storage._flush_lock = asyncio.Lock()
    storage._captured = captured
    storage._retry_kwargs = retry_kwargs
    return storage


# ---------------------------------------------------------------------------
# 1. _max_batch_size is always 200 (not embedding_batch_num)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 2. Namespace: TEXT_CHUNKS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_text_chunks_tuple_order():
    storage = make_storage(NameSpace.KV_STORE_TEXT_CHUNKS)
    data = {
        "chunk-1": {
            "tokens": 42,
            "chunk_order_index": 0,
            "full_doc_id": "doc-1",
            "content": "hello world",
            "file_path": "/a/b.[native-Fi].txt",
            "llm_cache_list": ["cache-key"],
            "heading": {"level": 2, "text": "Section A"},
            "sidecar": {"type": "drawing", "id": "img-1", "refs": []},
        }
    }
    await storage.upsert(data)

    assert len(storage._captured) == 1
    sql, rows = storage._captured[0]
    assert "LIGHTRAG_DOC_CHUNKS" in sql
    assert len(rows) == 1
    row = rows[0]
    # SQL: (workspace, id, tokens, chunk_order_index, full_doc_id,
    #        content, file_path, llm_cache_list, heading, sidecar,
    #        create_time, update_time)
    assert row[0] == "test_ws"  # workspace
    assert row[1] == "chunk-1"  # id
    assert row[2] == 42  # tokens
    assert row[3] == 0  # chunk_order_index
    assert row[4] == "doc-1"  # full_doc_id
    assert row[5] == "hello world"  # content
    assert row[6] == "/a/b.[native-Fi].txt"  # file_path
    assert json.loads(row[7]) == ["cache-key"]  # llm_cache_list
    assert json.loads(row[8]) == {"level": 2, "text": "Section A"}  # heading
    assert json.loads(row[9]) == {
        "type": "drawing",
        "id": "img-1",
        "refs": [],
    }  # sidecar


@pytest.mark.asyncio
async def test_upsert_text_chunks_missing_heading_sidecar_defaults_to_empty_dict():
    """Plain-text chunks without heading/sidecar should serialize to '{}'."""
    storage = make_storage(NameSpace.KV_STORE_TEXT_CHUNKS)
    data = {
        "chunk-1": {
            "tokens": 10,
            "chunk_order_index": 0,
            "full_doc_id": "doc-1",
            "content": "plain text",
            "file_path": "/a/b.txt",
        }
    }
    await storage.upsert(data)

    _, rows = storage._captured[0]
    row = rows[0]
    assert json.loads(row[8]) == {}  # heading
    assert json.loads(row[9]) == {}  # sidecar


# ---------------------------------------------------------------------------
# 3. Namespace: FULL_DOCS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_full_docs_tuple_order():
    storage = make_storage(NameSpace.KV_STORE_FULL_DOCS)
    data = {
        "doc-1": {
            "content": "full text",
            "file_path": "/path/doc.[mineru-Fi].pdf",
            "sidecar_location": "lightrag://sidecar/doc-1",
            "parse_format": "lightrag",
            "content_hash": "deadbeef",
            "process_options": "Fi",
            "chunk_options": {"chunk_token_size": 1200, "chunk_overlap": 100},
            "parse_engine": "mineru",
        }
    }
    await storage.upsert(data)

    assert len(storage._captured) == 1
    _, rows = storage._captured[0]
    row = rows[0]
    # SQL: (id, content, doc_name, workspace, sidecar_location, parse_format,
    #       content_hash, process_options, chunk_options, parse_engine)
    assert row[0] == "doc-1"
    assert row[1] == "full text"
    assert row[2] == "/path/doc.[mineru-Fi].pdf"
    assert row[3] == "test_ws"
    assert row[4] == "lightrag://sidecar/doc-1"
    assert row[5] == "lightrag"
    assert row[6] == "deadbeef"
    assert row[7] == "Fi"
    assert json.loads(row[8]) == {"chunk_token_size": 1200, "chunk_overlap": 100}
    assert row[9] == "mineru"


@pytest.mark.asyncio
async def test_upsert_full_docs_missing_pipeline_fields_pass_through_as_none():
    """Missing pipeline-derived fields must serialize as None at the Python
    layer so the SQL-level COALESCE guard can distinguish "caller did not
    supply" from "caller supplied a real value".

    The 'raw' default for parse_format is provided by the column DDL on
    initial insert; the Python layer must NOT inject it, otherwise the
    COALESCE guard never triggers on subsequent partial writes (a follow-up
    upsert with no parse_format would re-stamp the column with 'raw' and
    blow away a previously-set 'lightrag').
    """
    storage = make_storage(NameSpace.KV_STORE_FULL_DOCS)
    data = {"doc-1": {"content": "full text", "file_path": "/path/doc.pdf"}}
    await storage.upsert(data)

    _, rows = storage._captured[0]
    row = rows[0]
    assert row[4] is None  # sidecar_location
    assert row[5] is None  # parse_format — DDL supplies 'raw' default on insert
    assert row[6] is None  # content_hash
    assert row[7] is None  # process_options
    assert json.loads(row[8]) == {}  # chunk_options default
    assert row[9] is None  # parse_engine


@pytest.mark.asyncio
async def test_upsert_full_docs_none_chunk_options_defaults_to_empty_dict():
    storage = make_storage(NameSpace.KV_STORE_FULL_DOCS)
    data = {
        "doc-1": {
            "content": "full text",
            "file_path": "/path/doc.pdf",
            "chunk_options": None,
        }
    }
    await storage.upsert(data)

    _, rows = storage._captured[0]
    assert json.loads(rows[0][8]) == {}


@pytest.mark.asyncio
async def test_upsert_full_docs_sql_protects_partial_writes():
    """The ON CONFLICT clause must COALESCE+NULLIF every pipeline-derived
    column so a follow-up upsert that only carries ``content`` + ``doc_name``
    does not silently overwrite previously-recorded metadata back to defaults.

    We assert this at the SQL-template level since the actual COALESCE
    behavior is executed by Postgres. The presence of the protective
    expression in the SQL is the single source of truth for the guarantee.
    """
    storage = make_storage(NameSpace.KV_STORE_FULL_DOCS)
    await storage.upsert(
        {"doc-1": {"content": "full text", "file_path": "/path/doc.pdf"}}
    )
    sql, _ = storage._captured[0]
    normalized = " ".join(sql.split()).lower()

    # Each pipeline-derived string column must be COALESCE/NULLIF-guarded
    for col in (
        "sidecar_location",
        "parse_format",
        "content_hash",
        "process_options",
        "parse_engine",
    ):
        assert f"coalesce( nullif(excluded.{col}, '')" in normalized, (
            f"upsert_doc_full must guard {col} via COALESCE+NULLIF"
        )
        assert f"lightrag_doc_full.{col}" in normalized, (
            f"upsert_doc_full must preserve existing {col} on partial write"
        )

    # chunk_options (JSONB) is guarded via CASE on NULL/empty-object literal
    assert "excluded.chunk_options is null" in normalized
    assert "excluded.chunk_options = '{}'::jsonb" in normalized
    assert "lightrag_doc_full.chunk_options" in normalized

    # content / doc_name remain straight overwrites — they ARE the payload
    assert "content = excluded.content" in normalized
    assert "doc_name = excluded.doc_name" in normalized


# ---------------------------------------------------------------------------
# 4. Namespace: LLM_RESPONSE_CACHE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_llm_cache_tuple_order():
    storage = make_storage(NameSpace.KV_STORE_LLM_RESPONSE_CACHE)
    data = {
        "key-1": {
            "original_prompt": "what is X?",
            "return": "X is Y",
            "chunk_id": "chunk-1",
            "cache_type": "query",
            "queryparam": {"mode": "hybrid"},
        }
    }
    await storage.upsert(data)

    assert len(storage._captured) == 1
    _, rows = storage._captured[0]
    row = rows[0]
    # SQL: (workspace, id, original_prompt, return_value, chunk_id, cache_type, queryparam)
    assert row[0] == "test_ws"
    assert row[1] == "key-1"
    assert row[2] == "what is X?"
    assert row[3] == "X is Y"
    assert row[4] == "chunk-1"
    assert row[5] == "query"
    assert json.loads(row[6]) == {"mode": "hybrid"}


@pytest.mark.asyncio
async def test_upsert_llm_cache_null_queryparam():
    storage = make_storage(NameSpace.KV_STORE_LLM_RESPONSE_CACHE)
    data = {
        "key-2": {
            "original_prompt": "prompt",
            "return": "answer",
            "cache_type": "extract",
        }
    }
    await storage.upsert(data)
    _, rows = storage._captured[0]
    assert rows[0][6] is None  # queryparam should be None


# ---------------------------------------------------------------------------
# 5. Namespace: FULL_ENTITIES
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_full_entities_tuple_order():
    storage = make_storage(NameSpace.KV_STORE_FULL_ENTITIES)
    data = {"ent-1": {"entity_names": ["EntityA", "EntityB"], "count": 2}}
    await storage.upsert(data)

    _, rows = storage._captured[0]
    row = rows[0]
    # SQL: (workspace, id, entity_names, count, create_time, update_time)
    assert row[0] == "test_ws"
    assert row[1] == "ent-1"
    assert json.loads(row[2]) == ["EntityA", "EntityB"]
    assert row[3] == 2


# ---------------------------------------------------------------------------
# 6. Namespace: FULL_RELATIONS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_full_relations_tuple_order():
    storage = make_storage(NameSpace.KV_STORE_FULL_RELATIONS)
    data = {"rel-1": {"relation_pairs": [["A", "B"]], "count": 1}}
    await storage.upsert(data)

    _, rows = storage._captured[0]
    row = rows[0]
    # SQL: (workspace, id, relation_pairs, count, create_time, update_time)
    assert row[0] == "test_ws"
    assert row[1] == "rel-1"
    assert json.loads(row[2]) == [["A", "B"]]
    assert row[3] == 1


# ---------------------------------------------------------------------------
# 7. Namespace: ENTITY_CHUNKS / RELATION_CHUNKS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_entity_chunks_tuple_order():
    storage = make_storage(NameSpace.KV_STORE_ENTITY_CHUNKS)
    data = {"ec-1": {"chunk_ids": ["c1", "c2"], "count": 2}}
    await storage.upsert(data)

    _, rows = storage._captured[0]
    row = rows[0]
    # SQL: (workspace, id, chunk_ids, count, create_time, update_time)
    assert row[0] == "test_ws"
    assert row[1] == "ec-1"
    assert json.loads(row[2]) == ["c1", "c2"]
    assert row[3] == 2


@pytest.mark.asyncio
async def test_upsert_relation_chunks_tuple_order():
    storage = make_storage(NameSpace.KV_STORE_RELATION_CHUNKS)
    data = {"rc-1": {"chunk_ids": ["c3"], "count": 1}}
    await storage.upsert(data)

    _, rows = storage._captured[0]
    row = rows[0]
    assert row[0] == "test_ws"
    assert row[1] == "rc-1"
    assert json.loads(row[2]) == ["c3"]
    assert row[3] == 1


# ---------------------------------------------------------------------------
# 8. Sub-batching: data > _max_batch_size splits into multiple _run_with_retry calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sub_batching_splits_correctly():
    storage = make_storage(NameSpace.KV_STORE_FULL_DOCS)
    # upsert splits by _max_upsert_records_per_batch (byte cap left at default).
    storage._max_upsert_records_per_batch = 3

    data = {f"doc-{i}": {"content": f"text {i}", "file_path": ""} for i in range(7)}
    await storage.upsert(data)

    # 7 records / batch_size 3 => 3 batches (3 + 3 + 1)
    assert len(storage._captured) == 3
    assert len(storage._captured[0][1]) == 3
    assert len(storage._captured[1][1]) == 3
    assert len(storage._captured[2][1]) == 1


@pytest.mark.asyncio
async def test_sub_batching_exact_multiple():
    storage = make_storage(NameSpace.KV_STORE_FULL_DOCS)
    storage._max_upsert_records_per_batch = 3

    data = {f"doc-{i}": {"content": f"text {i}", "file_path": ""} for i in range(6)}
    await storage.upsert(data)

    # 6 / 3 => exactly 2 batches
    assert len(storage._captured) == 2
    assert len(storage._captured[0][1]) == 3
    assert len(storage._captured[1][1]) == 3


# ---------------------------------------------------------------------------
# 9. Empty data: no DB call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_empty_data_no_db_call():
    storage = make_storage(NameSpace.KV_STORE_FULL_DOCS)
    await storage.upsert({})
    assert len(storage._captured) == 0
    storage.db._run_with_retry.assert_not_called()


# ---------------------------------------------------------------------------
# 10. Unknown namespace raises ValueError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_unknown_namespace_raises():
    storage = make_storage("unknown_namespace")
    with pytest.raises(ValueError, match="Unknown namespace"):
        await storage.upsert({"k": {"v": 1}})


# ---------------------------------------------------------------------------
# 11. Multiple records go into one batch when within limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_records_single_batch():
    storage = make_storage(NameSpace.KV_STORE_FULL_DOCS)
    data = {
        "doc-1": {"content": "text 1", "file_path": "/a"},
        "doc-2": {"content": "text 2", "file_path": "/b"},
        "doc-3": {"content": "text 3", "file_path": "/c"},
    }
    await storage.upsert(data)

    # All 3 fit within default batch size of 200
    assert len(storage._captured) == 1
    _, rows = storage._captured[0]
    assert len(rows) == 3
    ids = {row[0] for row in rows}  # id is $1 for FULL_DOCS
    assert ids == {"doc-1", "doc-2", "doc-3"}


@pytest.mark.asyncio
async def test_kv_upsert_passes_timing_label():
    storage = make_storage(NameSpace.KV_STORE_FULL_DOCS)
    await storage.upsert({"doc-1": {"content": "text 1", "file_path": "/a"}})

    assert storage._retry_kwargs[0]["timing_label"] == (
        f"test_ws PGKVStorage.upsert[{NameSpace.KV_STORE_FULL_DOCS}]"
    )


@pytest.mark.asyncio
async def test_doc_status_upsert_passes_timing_label():
    storage = make_doc_status_storage()
    await storage.upsert(
        {
            "doc-1": {
                "content_summary": "summary",
                "content_length": 12,
                "chunks_count": 1,
                "status": "processed",
                "file_path": "/a.txt",
                "chunks_list": ["chunk-1"],
                "metadata": {"source": "test"},
                "created_at": "2024-01-01T00:00:00+00:00",
                "updated_at": "2024-01-01T00:00:00+00:00",
            }
        }
    )

    assert storage._retry_kwargs[0]["timing_label"] == (
        "test_ws PGDocStatusStorage.upsert"
    )


# ---------------------------------------------------------------------------
# doc_status: content_hash tuple + COALESCE SQL guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_doc_status_upsert_includes_content_hash():
    storage = make_doc_status_storage()
    await storage.upsert(
        {
            "doc-1": {
                "content_summary": "summary",
                "content_length": 12,
                "chunks_count": 1,
                "status": "processed",
                "file_path": "/uploads/a.[native-Fi].txt",
                "chunks_list": ["chunk-1"],
                "metadata": {"source": "test"},
                "content_hash": "abc123",
                "created_at": "2024-01-01T00:00:00+00:00",
                "updated_at": "2024-01-01T00:00:00+00:00",
            }
        }
    )

    sql, rows = storage._captured[0]
    # content_hash should be present in the INSERT column list and tuple
    assert "content_hash" in sql
    row = rows[0]
    # Tuple layout: workspace, id, content_summary, content_length, chunks_count,
    # status, file_path, chunks_list, track_id, metadata, error_msg,
    # content_hash, created_at, updated_at
    assert row[6] == "/uploads/a.[native-Fi].txt"
    assert row[11] == "abc123"


@pytest.mark.asyncio
async def test_doc_status_upsert_sql_protects_existing_content_hash():
    """The ON CONFLICT clause must COALESCE+NULLIF to preserve a previously
    set content_hash when a subsequent state-transition upsert carries no
    hash (None) or an empty string.

    We assert this at the SQL-template level since the actual COALESCE
    behavior is executed by Postgres. The presence of the protective
    expression in the SQL is the single source of truth for the guarantee.
    """
    storage = make_doc_status_storage()
    await storage.upsert(
        {
            "doc-1": {
                "content_summary": "summary",
                "content_length": 12,
                "chunks_count": 1,
                "status": "processed",
                "file_path": "/a.txt",
                "chunks_list": [],
                "metadata": {},
                "created_at": "2024-01-01T00:00:00+00:00",
                "updated_at": "2024-01-01T00:00:00+00:00",
            }
        }
    )

    sql, rows = storage._captured[0]
    # A caller that passes no content_hash still produces a valid tuple.
    assert rows[0][11] is None
    normalized = " ".join(sql.split()).lower()
    assert "coalesce(" in normalized
    assert "nullif(excluded.content_hash, '')" in normalized
    assert "lightrag_doc_status.content_hash" in normalized


@pytest.mark.asyncio
async def test_vector_flush_passes_timing_label():
    """upsert() now buffers; the timing_label is emitted by the flush path
    that runs from index_done_callback() / finalize().
    """
    storage = make_vector_storage(NameSpace.VECTOR_STORE_CHUNKS)
    await storage.upsert(
        {
            "chunk-1": {
                "tokens": 42,
                "chunk_order_index": 0,
                "full_doc_id": "doc-1",
                "content": "hello world",
                "file_path": "/a/b.txt",
            }
        }
    )
    # No retry call until flush.
    assert storage._retry_kwargs == []

    await storage.index_done_callback()

    assert storage._retry_kwargs[0]["timing_label"] == (
        f"test_ws PGVectorStorage.flush[{NameSpace.VECTOR_STORE_CHUNKS}]"
    )


# ---------------------------------------------------------------------------
# Batch limits: KV byte-budget split + delete chunking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kv_upsert_splits_by_payload_bytes():
    storage = make_storage(NameSpace.KV_STORE_FULL_DOCS)
    # Disable the record cap; force a byte cap that no two rows can share.
    storage._max_upsert_records_per_batch = 0
    storage._max_upsert_payload_bytes = 200

    data = {f"doc-{i}": {"content": "X" * 150, "file_path": ""} for i in range(3)}
    await storage.upsert(data)

    # Each ~150-byte content row exceeds half the budget -> 3 separate batches.
    assert len(storage._captured) == 3


@pytest.mark.asyncio
async def test_kv_delete_splits_by_id_cap():
    storage = make_storage(NameSpace.KV_STORE_FULL_DOCS)
    storage._max_delete_records_per_batch = 2

    await storage.delete([f"doc-{i}" for i in range(5)])

    # 5 ids / cap 2 => 3 bounded ANY($2) DELETE statements, all in ONE
    # transaction (a single _run_with_retry closure) for all-or-nothing semantics.
    assert len(storage._retry_kwargs) == 1
    assert len(storage._captured_execute) == 3
    slices = [args[1] for _, args in storage._captured_execute]
    assert [len(s) for s in slices] == [2, 2, 1]
    assert all("DELETE FROM" in sql for sql, _ in storage._captured_execute)


@pytest.mark.asyncio
async def test_kv_delete_chunks_share_one_transaction():
    """All delete chunks run inside a single connection.transaction()."""
    storage = make_storage(NameSpace.KV_STORE_FULL_DOCS)
    storage._max_delete_records_per_batch = 2

    tx_calls = {"n": 0}

    async def run(operation, **kwargs):
        mock_conn = AsyncMock()
        tx = AsyncMock()
        tx.__aenter__.return_value = tx
        tx.__aexit__.return_value = False

        def _make_tx():
            tx_calls["n"] += 1
            return tx

        mock_conn.transaction = MagicMock(side_effect=_make_tx)
        await operation(mock_conn)

    storage.db._run_with_retry = AsyncMock(side_effect=run)

    await storage.delete([f"doc-{i}" for i in range(5)])

    # One transaction wrapping all 3 chunks, not one per chunk.
    assert tx_calls["n"] == 1


# ---------------------------------------------------------------------------
# Batch limits: DocStatus record-cap split + delete chunking
# ---------------------------------------------------------------------------


def _doc_status_payload(i: int) -> dict:
    return {
        "content_summary": f"summary {i}",
        "content_length": 10,
        "chunks_count": 1,
        "status": "processed",
        "file_path": f"/{i}.txt",
        "chunks_list": ["chunk-1"],
        "metadata": {"source": "test"},
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00",
    }


@pytest.mark.asyncio
async def test_doc_status_upsert_splits_by_record_cap():
    storage = make_doc_status_storage()
    storage._max_upsert_records_per_batch = 2

    data = {f"doc-{i}": _doc_status_payload(i) for i in range(5)}
    await storage.upsert(data)

    # 5 records / cap 2 => 3 executemany calls.
    assert len(storage._captured) == 3
    assert [len(rows) for _, rows in storage._captured] == [2, 2, 1]


@pytest.mark.asyncio
async def test_doc_status_delete_splits_by_id_cap():
    storage = make_doc_status_storage()
    storage._max_delete_records_per_batch = 2

    await storage.delete([f"doc-{i}" for i in range(5)])

    # One transaction (single _run_with_retry) wrapping 3 bounded DELETEs.
    assert len(storage._retry_kwargs) == 1
    assert len(storage._captured_execute) == 3
    slices = [args[1] for _, args in storage._captured_execute]
    assert [len(s) for s in slices] == [2, 2, 1]


# ---------------------------------------------------------------------------
# Delete error propagation & set-typed IDs
# ---------------------------------------------------------------------------


class _DeleteUnavailable(Exception):
    """Sentinel database error raised by _FailingDeleteDB."""


class _FailingDeleteDB:
    def __init__(self):
        self.calls = 0

    async def _run_with_retry(self, operation):
        self.calls += 1
        raise _DeleteUnavailable("delete unavailable")


class _FakeDeleteTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _RecordingDeleteConnection:
    """Captures the id-array argument of each ``execute`` call."""

    def __init__(self):
        self.deleted_id_batches: list[list[str]] = []

    def transaction(self):
        return _FakeDeleteTransaction()

    async def execute(self, sql, workspace, id_slice):
        assert isinstance(id_slice, list)
        self.deleted_id_batches.append(id_slice)


class _RecordingDeleteDB:
    def __init__(self, connection):
        self._connection = connection

    async def _run_with_retry(self, operation):
        return await operation(self._connection)


class TestPostgresDeleteSemantics:
    @staticmethod
    def _failing_storage(cls, namespace):
        storage = object.__new__(cls)
        storage.namespace = namespace
        storage.workspace = "test_ws"
        storage._max_delete_records_per_batch = 0
        storage.db = _FailingDeleteDB()
        return storage

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("cls", "namespace"),
        [
            (PGKVStorage, NameSpace.KV_STORE_TEXT_CHUNKS),
            (PGDocStatusStorage, NameSpace.DOC_STATUS),
        ],
    )
    async def test_delete_propagates_database_failure(self, cls, namespace):
        storage = self._failing_storage(cls, namespace)

        with pytest.raises(_DeleteUnavailable, match="delete unavailable"):
            await storage.delete(["id-1", "id-2"])

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("cls", "namespace"),
        [
            (PGKVStorage, NameSpace.KV_STORE_TEXT_CHUNKS),
            (PGDocStatusStorage, NameSpace.DOC_STATUS),
        ],
    )
    async def test_empty_delete_does_not_touch_the_database(self, cls, namespace):
        storage = self._failing_storage(cls, namespace)

        await storage.delete([])

        assert storage.db.calls == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("cls", "namespace", "prefix"),
        [
            pytest.param(
                PGKVStorage, NameSpace.KV_STORE_TEXT_CHUNKS, "chunk", id="pgkv"
            ),
            pytest.param(
                PGDocStatusStorage, NameSpace.DOC_STATUS, "doc", id="pgdocstatus"
            ),
        ],
    )
    async def test_delete_accepts_set_and_chunks_all_ids(self, cls, namespace, prefix):
        """Passing a set of IDs with a positive batch cap must chunk and delete every ID."""
        ids = {f"{prefix}-{i}" for i in range(5)}
        storage = object.__new__(cls)
        storage.namespace = namespace
        storage.workspace = "test_ws"
        storage._max_delete_records_per_batch = 2
        connection = _RecordingDeleteConnection()
        storage.db = _RecordingDeleteDB(connection)

        await storage.delete(ids)

        flattened = [cid for batch in connection.deleted_id_batches for cid in batch]
        assert set(flattened) == ids
        assert len(flattened) == len(ids)
        assert all(len(batch) <= 2 for batch in connection.deleted_id_batches)


# ---------------------------------------------------------------------------
# Configuration container identity & config namespace
# ---------------------------------------------------------------------------


class TestPostgresConfigIdentity:
    CONTAINER = "PGKVStorage (_lightrag_config)"

    @pytest.fixture(autouse=True)
    def _shared(self):
        initialize_share_data(workers=1)
        yield
        finalize_share_data()

    class _Table:
        def __init__(self, *, read_error=None):
            self.rows: dict[tuple[str, str], str] = {}
            self.read_error = read_error
            self.writes = 0

        async def query(self, sql, params=None, multirows=False, **kwargs):
            if self.read_error is not None:
                raise self.read_error
            assert sql == SQL_TEMPLATES["get_by_id_config"]
            workspace, key = params
            value = self.rows.get((workspace, key))
            if value is None:
                return None
            return {"id": key, "value": value, "create_time": 1, "update_time": 1}

        async def run_with_retry(self, operation, **kwargs):
            conn = AsyncMock()
            tx = AsyncMock()
            tx.__aenter__.return_value = tx
            tx.__aexit__.return_value = False
            conn.transaction = MagicMock(return_value=tx)
            await operation(conn)
            for call in conn.executemany.call_args_list:
                assert call.args[0] == SQL_TEMPLATES["upsert_config"]
                for workspace, key, payload in call.args[1]:
                    self.writes += 1
                    self.rows[(workspace, key)] = payload

    @staticmethod
    def _storage(table) -> PGKVStorage:
        storage = cs.create_configuration_storage(
            PGKVStorage, global_config={}, embedding_func=None
        )
        db = MagicMock()
        db.query = AsyncMock(side_effect=table.query)
        db._run_with_retry = AsyncMock(side_effect=table.run_with_retry)
        db.workspace = None
        storage.db = db
        return storage

    async def _bind(self, storage, tmp_path):
        return await cs.bind_configuration_identity(
            storage,
            working_dir=str(tmp_path),
            backend="PGKVStorage",
            container=self.CONTAINER,
        )

    async def test_the_identity_is_created_read_back_and_then_verified(self, tmp_path):
        table = self._Table()
        storage = self._storage(table)
        created = await self._bind(storage, tmp_path)
        assert created.action == "created"
        payload = json.loads(
            table.rows[(CONFIG_CONTAINER_TAG, cs.storage_identity_key())]
        )
        assert payload["value"] == {"uuid": created.storage_uuid}
        assert ca.read_anchor(str(tmp_path)).backend == "PGKVStorage"

        verified = await self._bind(storage, tmp_path)
        assert verified.action == "verified"
        assert verified.storage_uuid == created.storage_uuid
        assert table.writes == 1

    async def test_a_read_failure_creates_neither_identity_nor_anchor(self, tmp_path):
        table = self._Table(read_error=ConnectionError("server closed the connection"))
        with pytest.raises(ConfigurationStorageError):
            await self._bind(self._storage(table), tmp_path)
        assert table.rows == {}
        assert ca.read_anchor(str(tmp_path)) is None


class TestPostgresConfigNamespace:
    @staticmethod
    def _storage(query_side_effect):
        storage = PGKVStorage.__new__(PGKVStorage)
        storage.namespace = "config"
        storage.workspace = CONFIG_CONTAINER_TAG
        storage.global_config = {}
        db = MagicMock()
        db.query = AsyncMock(side_effect=query_side_effect)
        db.workspace = None
        storage.db = db
        storage.__post_init__()
        return storage, db

    def test_the_namespace_maps_to_its_own_table_and_ddl(self):
        assert NAMESPACE_TABLE_MAP[NameSpace.KV_STORE_CONFIG] == "LIGHTRAG_CONFIG"
        assert namespace_to_table_name("config") == "LIGHTRAG_CONFIG"
        ddl = TABLES["LIGHTRAG_CONFIG"]["ddl"]
        assert "value JSONB NOT NULL" in ddl
        assert "PRIMARY KEY (workspace, id)" in ddl
        for namespace in (
            "full_docs",
            "text_chunks",
            "llm_response_cache",
            "full_entities",
            "full_relations",
            "entity_chunks",
            "relation_chunks",
            "doc_status",
            "entities",
            "relationships",
            "chunks",
        ):
            assert namespace_to_table_name(namespace) != "LIGHTRAG_CONFIG", namespace

    def test_the_templates_read_and_write_the_config_table(self):
        assert (
            "FROM LIGHTRAG_CONFIG WHERE workspace=$1 AND id=$2"
            in SQL_TEMPLATES["get_by_id_config"]
        )
        assert "id = ANY($2)" in SQL_TEMPLATES["get_by_ids_config"]
        upsert = SQL_TEMPLATES["upsert_config"]
        assert "INSERT INTO LIGHTRAG_CONFIG (workspace, id, value)" in upsert
        assert "ON CONFLICT (workspace,id) DO UPDATE" in upsert
        assert "update_time = CURRENT_TIMESTAMP" in upsert

    def test_row_shaping_round_trips_the_payload(self):
        row = {
            "_id": "ws/embedding/entities",
            "create_time": 1,
            "update_time": 2,
            "schema_version": 1,
            "workspace": "ws",
            "updated_at": "now",
            "updated_by": "test",
            "value": {"model": "m", "dim": 8, "origin": "probe"},
        }
        payload = _config_row_payload(row)
        assert set(payload) == {
            "schema_version",
            "workspace",
            "updated_at",
            "updated_by",
            "value",
        }

        rebuilt = _config_row_from_pg(
            {
                "id": "ws/embedding/entities",
                "value": json.dumps(payload),
                "create_time": 5,
                "update_time": 0,
            }
        )
        assert rebuilt["value"] == {"model": "m", "dim": 8, "origin": "probe"}
        assert rebuilt["workspace"] == "ws"
        assert rebuilt["_id"] == rebuilt["id"] == "ws/embedding/entities"
        assert rebuilt["create_time"] == 5 and rebuilt["update_time"] == 5

        assert _config_row_from_pg({"id": "k", "value": {"value": {}}})["value"] == {}
        assert (
            _config_row_from_pg({"id": "k", "value": "not json"}).get("value") is None
        )

    async def test_upsert_builds_the_config_tuple(self):
        captured = []

        async def fake_run_with_retry(operation, **kwargs):
            conn = AsyncMock()
            tx = AsyncMock()
            tx.__aenter__.return_value = tx
            tx.__aexit__.return_value = False
            conn.transaction = MagicMock(return_value=tx)
            await operation(conn)
            for call in conn.executemany.call_args_list:
                captured.append((call.args[0], call.args[1]))

        storage, db = self._storage(None)
        db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)

        await storage.upsert(
            {
                "ws/embedding/chunks": {
                    "schema_version": 1,
                    "workspace": "ws",
                    "updated_at": "now",
                    "updated_by": "t",
                    "value": {"model": "m", "dim": 8, "origin": "rebuild"},
                    "_id": "ws/embedding/chunks",
                    "create_time": 1,
                    "update_time": 2,
                }
            }
        )

        assert len(captured) == 1
        sql, rows = captured[0]
        assert sql == SQL_TEMPLATES["upsert_config"]
        workspace, key, payload = rows[0]
        assert (workspace, key) == (CONFIG_CONTAINER_TAG, "ws/embedding/chunks")
        decoded = json.loads(payload)
        assert decoded["value"] == {"model": "m", "dim": 8, "origin": "rebuild"}
        assert "create_time" not in decoded and "_id" not in decoded

    async def test_iter_rows_pages_the_ids_and_reads_each_page_through_get_by_ids(self):
        ids = [f"ws{i}/embedding/entities" for i in range(5)]
        table = {
            key: {
                "id": key,
                "value": json.dumps({"workspace": f"ws{i}", "value": {"n": i}}),
                "create_time": 1,
                "update_time": 1,
            }
            for i, key in enumerate(ids)
        }
        page_calls = []

        async def query(sql, params=None, multirows=False, **kwargs):
            if sql.startswith("SELECT id FROM LIGHTRAG_CONFIG"):
                workspace, last_id, limit = params
                assert workspace == CONFIG_CONTAINER_TAG
                page_calls.append((last_id, limit))
                remaining = [k for k in sorted(table) if last_id is None or k > last_id]
                return [{"id": k} for k in remaining[:limit]]
            assert sql == SQL_TEMPLATES["get_by_ids_config"]
            workspace, wanted = params
            return [dict(table[k]) for k in wanted]

        storage, _ = self._storage(query)

        rows = [row async for row in storage.iter_rows(page_size=2)]

        assert [r["_id"] for r in rows] == sorted(ids)
        assert [r["workspace"] for r in rows] == [f"ws{i}" for i in range(5)]
        assert page_calls == [(None, 2), (sorted(ids)[1], 2), (sorted(ids)[3], 2)]

    async def test_iter_rows_on_an_empty_table_yields_nothing(self):
        storage, _ = self._storage(lambda *a, **k: [])
        assert [row async for row in storage.iter_rows()] == []


# ---------------------------------------------------------------------------
# Batch-limit helpers (_resolve_pg_batch_limits, _estimate_record_bytes, _chunk_by_budget)
# ---------------------------------------------------------------------------


class TestResolvePgBatchLimits:
    def test_defaults(self):
        with patch.dict("os.environ", {}, clear=True):
            payload, upserts, deletes = _resolve_pg_batch_limits()
        assert payload == DEFAULT_PG_UPSERT_MAX_PAYLOAD_BYTES
        assert upserts == DEFAULT_PG_UPSERT_MAX_RECORDS_PER_BATCH == 200
        assert deletes == DEFAULT_PG_DELETE_MAX_RECORDS_PER_BATCH

    def test_env_override(self):
        env = {
            "POSTGRES_UPSERT_MAX_PAYLOAD_BYTES": "12345",
            "POSTGRES_UPSERT_MAX_RECORDS_PER_BATCH": "7",
            "POSTGRES_DELETE_MAX_RECORDS_PER_BATCH": "9",
        }
        with patch.dict("os.environ", env, clear=True):
            assert _resolve_pg_batch_limits() == (12345, 7, 9)

    def test_non_positive_disables_and_warns(self):
        env = {
            "POSTGRES_UPSERT_MAX_PAYLOAD_BYTES": "0",
            "POSTGRES_UPSERT_MAX_RECORDS_PER_BATCH": "-1",
            "POSTGRES_DELETE_MAX_RECORDS_PER_BATCH": "0",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("lightrag.kg.postgres_impl.logger") as mock_logger:
                payload, upserts, deletes = _resolve_pg_batch_limits()
        assert (payload, upserts, deletes) == (0, -1, 0)
        warnings = [c.args[0] for c in mock_logger.warning.call_args_list]
        assert sum("non-positive" in msg for msg in warnings) == 3


class TestEstimateRecordBytes:
    @pytest.mark.parametrize(
        ("record", "expected"),
        [
            pytest.param(
                ("id", np.zeros(128, dtype=np.float32)),
                128 * 4 + 2,
                id="ndarray_nbytes",
            ),
            pytest.param(("abc",), 3, id="ascii_str"),
            pytest.param(("中文",), 6, id="multibyte_str"),
            pytest.param((b"abcd", None), 4, id="bytes_and_none"),
            pytest.param(
                ({"a": 1, "b": [1, 2, 3]},),
                len(
                    json.dumps(
                        {"a": 1, "b": [1, 2, 3]},
                        ensure_ascii=False,
                        separators=(",", ":"),
                        default=str,
                    ).encode("utf-8")
                ),
                id="dict_and_list_compact_json",
            ),
            pytest.param((1, 2.0, None), 32, id="scalar_constant"),
            pytest.param(
                ("ws", "id", "hello", np.ones(4, dtype=np.float32), None, 42),
                41,
                id="mixed_tuple",
            ),
        ],
    )
    def test_estimate_record_bytes(self, record, expected):
        assert _estimate_record_bytes(record) == expected


class TestChunkByBudget:
    @pytest.mark.parametrize(
        ("items", "size_fn", "max_bytes", "max_records", "expected_sizes"),
        [
            pytest.param([], lambda x: 1, 100, 100, [], id="empty"),
            pytest.param(
                list(range(7)), lambda x: 1, 0, 3, [3, 3, 1], id="split_by_record_count"
            ),
            pytest.param(
                ["x" * 10, "y" * 10, "z" * 10, "w" * 10],
                len,
                25,
                0,
                [2, 2],
                id="split_by_payload_bytes",
            ),
            pytest.param(
                ["x" * 5] * 10,
                len,
                10_000,
                4,
                [4, 4, 2],
                id="both_dimensions_record_cap_wins",
            ),
            pytest.param(
                list(range(100)),
                lambda x: 999,
                0,
                0,
                [100],
                id="non_positive_disables_both",
            ),
        ],
    )
    def test_chunk_by_budget_splits(
        self, items, size_fn, max_bytes, max_records, expected_sizes
    ):
        batches = _chunk_by_budget(
            items,
            size_fn,
            max_payload_bytes=max_bytes,
            max_records_per_batch=max_records,
        )
        assert [len(b) for b, _ in batches] == expected_sizes

    def test_oversized_single_item_becomes_own_batch(self):
        items = ["small", "X" * 1000, "small2"]
        batches = _chunk_by_budget(
            items, len, max_payload_bytes=50, max_records_per_batch=0
        )
        assert [len(b) for b, _ in batches] == [1, 1, 1]
        assert batches[1][0] == ["X" * 1000]
