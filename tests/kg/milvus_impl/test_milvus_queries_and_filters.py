"""Tests for Milvus filter escaping, paginated queries/bisection, and Strong read consistency.

Covers:
- Filter literal escaping (``_escape_milvus_str``) and structured-value preservation
  on ``get_by_id``, ``get_by_ids``, ``delete_entity_relation``, ``upsert``, and ``delete``.
- ``get_vectors_by_ids`` and ``get_by_ids`` pagination, gRPC oversize page bisection,
  and cooperative event-loop yielding between pages.
- Strong consistency level on every point read / count / deletion lookup while
  vector search stays on the collection default.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from collections.abc import Callable, Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import grpc  # type: ignore
import numpy as np
import pytest
from pymilvus import MilvusException

from lightrag.kg.milvus_impl import MilvusVectorDBStorage, _escape_milvus_str

pytestmark = pytest.mark.offline

_GRPC_MAX_MESSAGE_BYTES = 4 * 1024 * 1024
_ROW_OVERHEAD_BYTES = 256


class _MockEmbeddingFunc:
    """Mock embedding function returning random vectors of configurable dimension."""

    def __init__(self, dim: int = 8) -> None:
        self.embedding_dim = dim
        self.max_token_size = 512
        self.model_name = "mock-embed"

    async def __call__(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        return np.random.rand(len(texts), self.embedding_dim).astype(np.float32)


@pytest.fixture(autouse=True)
def patch_namespace_lock() -> Generator[
    dict[tuple[str, str], asyncio.Lock], None, None
]:
    """Cache real asyncio.Locks per (namespace, workspace) for shared semantics."""
    cache: dict[tuple[str, str], asyncio.Lock] = {}

    def factory(
        namespace: str, workspace: str | None = None, enable_logging: bool = False
    ) -> asyncio.Lock:
        return cache.setdefault((namespace, workspace or ""), asyncio.Lock())

    with patch("lightrag.kg.milvus_impl.get_namespace_lock", side_effect=factory):
        yield cache


def _make_storage(
    embed_func: _MockEmbeddingFunc | None = None,
    *,
    namespace: str = "entities",
    workspace: str = "test",
    meta_fields: set[str] | None = None,
) -> MilvusVectorDBStorage:
    """Build a MilvusVectorDBStorage skipping ``initialize()`` (no real client)."""
    if embed_func is None:
        embed_func = _MockEmbeddingFunc(dim=8)
    if meta_fields is None:
        meta_fields = {"content", "entity_name", "src_id", "tgt_id"}
    storage = MilvusVectorDBStorage(
        namespace=namespace,
        workspace=workspace,
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        embedding_func=embed_func,
        meta_fields=meta_fields,
    )
    storage._client = MagicMock()
    storage._client.has_collection.return_value = True
    storage._client.upsert = MagicMock(return_value={"upsert_count": 0})
    storage._client.delete = MagicMock(return_value={"delete_count": 0})
    storage._client.query = MagicMock(return_value=[])
    storage._client.search = MagicMock(return_value=[[]])
    storage._client.load_collection = MagicMock()
    storage._initialized = True
    return storage


def _requested_id_count(filter: str) -> int:
    """Number of ids packed into an ``id in [...]`` filter."""
    return filter.count(",") + 1


def _resource_exhausted_query_side_effect(
    embedding_dim: int,
) -> Callable[..., list[dict[str, Any]]]:
    row_bytes = embedding_dim * 4 + _ROW_OVERHEAD_BYTES
    row_ids = itertools.count()

    def _side_effect(*, filter: str, **kwargs: Any) -> list[dict[str, Any]]:
        count = _requested_id_count(filter)
        response_bytes = count * row_bytes
        if response_bytes > _GRPC_MAX_MESSAGE_BYTES:
            raise MilvusException(
                message=(
                    "grpc: received message larger than max "
                    f"({response_bytes} vs. {_GRPC_MAX_MESSAGE_BYTES})"
                )
            )
        return [
            {"id": f"row-{next(row_ids)}", "vector": [0.0] * embedding_dim}
            for _ in range(count)
        ]

    return _side_effect


def _echo_query_side_effect(
    embedding_dim: int, call_sizes: list[int]
) -> Callable[..., list[dict[str, Any]]]:
    row_ids = itertools.count()

    def _side_effect(*, filter: str, **kwargs: Any) -> list[dict[str, Any]]:
        count = _requested_id_count(filter)
        call_sizes.append(count)
        return [
            {"id": f"row-{next(row_ids)}", "vector": [0.0] * embedding_dim}
            for _ in range(count)
        ]

    return _side_effect


async def _page_and_record(
    embedding_dim: int, ids: list[str]
) -> tuple[dict[str, list[float]], list[int]]:
    call_sizes: list[int] = []
    s = _make_storage(_MockEmbeddingFunc(dim=embedding_dim))
    s._client.query.side_effect = _echo_query_side_effect(embedding_dim, call_sizes)
    vectors = await s.get_vectors_by_ids(ids)
    return vectors, call_sizes


def _oversize_above_side_effect(
    max_ids_per_response: int, call_sizes: list[int]
) -> Callable[..., list[dict[str, Any]]]:
    def _side_effect(*, filter: str, **kwargs: Any) -> list[dict[str, Any]]:
        count = _requested_id_count(filter)
        call_sizes.append(count)
        if count > max_ids_per_response:
            raise MilvusException(
                message=(
                    "<MilvusException: (code=1, message=grpc: received message "
                    f"larger than max ({count} rows vs. {max_ids_per_response}))>"
                )
            )
        return []

    return _side_effect


class _QuotaExhaustedRpcError(grpc.RpcError):
    """RESOURCE_EXHAUSTED carrying no size marker (rate-limit/quota rejection)."""

    def code(self) -> grpc.StatusCode:
        return grpc.StatusCode.RESOURCE_EXHAUSTED

    def __str__(self) -> str:
        return (
            "<_InactiveRpcError of RPC that terminated with: "
            "StatusCode.RESOURCE_EXHAUSTED, rate limit exceeded>"
        )


def _ordered_echo_side_effect(
    ordered_ids: list[str], captured_filters: list[str]
) -> Callable[..., list[dict[str, Any]]]:
    cursor = list(ordered_ids)

    def _side_effect(*, filter: str, **kwargs: Any) -> list[dict[str, Any]]:
        captured_filters.append(filter)
        count = _requested_id_count(filter)
        page = cursor[:count]
        del cursor[:count]
        return [{"id": doc_id, "content": f"server-{doc_id}"} for doc_id in page]

    return _side_effect


# ---------------------------------------------------------------------------
# Filter escaping and structured-value boundaries
# ---------------------------------------------------------------------------


class TestMilvusEscapeFilter:
    @pytest.mark.parametrize(
        ("raw", "escaped"),
        [
            ('doc"1', 'doc\\"1'),
            ("doc\\2", "doc\\\\2"),
            ("line\nbreak", "line\\nbreak"),
            ("column\tbreak", "column\\tbreak"),
            ("carriage\rreturn", "carriage\\rreturn"),
        ],
    )
    def test_escape_milvus_str_covers_filter_literal_escapes(self, raw, escaped):
        assert _escape_milvus_str(raw) == escaped

    @pytest.mark.asyncio
    async def test_get_by_id_escapes_special_characters(self):
        storage = _make_storage(
            _MockEmbeddingFunc(dim=128),
            namespace="test_entities",
            meta_fields={"content", "created_at"},
        )
        raw_id = 'doc"\\1\nline\tcolumn\rreturn'
        storage._client.query.return_value = [{"id": raw_id, "content": "test"}]

        result = await storage.get_by_id(raw_id)
        assert result == {"id": raw_id, "content": "test"}

        storage._client.query.assert_called_once()
        _, kwargs = storage._client.query.call_args
        assert kwargs["filter"] == 'id == "doc\\"\\\\1\\nline\\tcolumn\\rreturn"'

    @pytest.mark.asyncio
    async def test_get_by_ids_escapes_special_characters(self):
        storage = _make_storage(
            _MockEmbeddingFunc(dim=128),
            namespace="test_entities",
            meta_fields={"content", "created_at"},
        )
        storage._client.query.return_value = [{"id": 'doc"1', "content": "test"}]

        result = await storage.get_by_ids(['doc"1', "doc\\2"])
        assert result == [{"id": 'doc"1', "content": "test"}, None]

        storage._client.query.assert_called_once()
        _, kwargs = storage._client.query.call_args
        assert kwargs["filter"] == 'id in ["doc\\"1", "doc\\\\2"]'

    @pytest.mark.asyncio
    async def test_delete_entity_relation_escapes_special_characters(self):
        storage = _make_storage(
            _MockEmbeddingFunc(dim=128),
            namespace="test_entities",
            meta_fields={"content", "created_at"},
        )
        storage._client.query.return_value = [{"id": "rel1"}]

        await storage.delete_entity_relation('ENTITY "A"\\PATH\nLINE\tCOLUMN\rRETURN')

        storage._client.query.assert_called_once()
        _, kwargs = storage._client.query.call_args
        assert (
            kwargs["filter"]
            == 'src_id == "ENTITY \\"A\\"\\\\PATH\\nLINE\\tCOLUMN\\rRETURN" '
            'or tgt_id == "ENTITY \\"A\\"\\\\PATH\\nLINE\\tCOLUMN\\rRETURN"'
        )

    @pytest.mark.asyncio
    async def test_upsert_preserves_structured_values_without_filter_escaping(self):
        storage = _make_storage(
            _MockEmbeddingFunc(dim=128),
            namespace="test_relationships",
            meta_fields={"content", "src_id", "tgt_id", "created_at"},
        )
        raw_id = 'rel"\\1\nline\tcolumn\rreturn'
        raw_src = 'ENTITY "A"\\PATH\nLINE'
        raw_tgt = "TARGET\tCOLUMN\rRETURN"

        await storage.upsert(
            {
                raw_id: {
                    "content": "relationship content",
                    "src_id": raw_src,
                    "tgt_id": raw_tgt,
                }
            }
        )

        pending = storage._pending_vector_docs[raw_id].source
        assert pending["id"] == raw_id
        assert pending["src_id"] == raw_src
        assert pending["tgt_id"] == raw_tgt

        storage.embedding_func = AsyncMock(
            return_value=np.zeros((1, storage.embedding_func.embedding_dim))
        )
        await storage.index_done_callback()

        persisted = storage._client.upsert.call_args.kwargs["data"][0]
        assert persisted["id"] == raw_id
        assert persisted["src_id"] == raw_src
        assert persisted["tgt_id"] == raw_tgt

    @pytest.mark.asyncio
    async def test_delete_preserves_primary_keys_without_filter_escaping(self):
        storage = _make_storage(
            _MockEmbeddingFunc(dim=128),
            namespace="test_entities",
            meta_fields={"content", "created_at"},
        )
        raw_id = 'doc"\\1\nline\tcolumn\rreturn'

        await storage.delete([raw_id])
        await storage.index_done_callback()

        storage._client.delete.assert_called_once_with(
            collection_name=storage.final_namespace,
            pks=[raw_id],
        )


# ---------------------------------------------------------------------------
# Paginated vector / metadata reads, oversize bisection, and cooperative yield
# ---------------------------------------------------------------------------


class TestGetVectorsByIdsPaging:
    @pytest.mark.asyncio
    async def test_returns_all_vectors_above_grpc_size_limit(self) -> None:
        s = _make_storage(_MockEmbeddingFunc(dim=3072))
        s._client.query.side_effect = _resource_exhausted_query_side_effect(
            embedding_dim=3072
        )

        ids = [f"chunk-{i}" for i in range(600)]
        vectors = await s.get_vectors_by_ids(ids)

        assert len(vectors) == 600
        assert s._client.query.call_count > 1

    @pytest.mark.asyncio
    async def test_skips_id_missing_from_database(self) -> None:
        s = _make_storage(_MockEmbeddingFunc(dim=8))
        s._client.query.return_value = []
        assert await s.get_vectors_by_ids(["missing-1"]) == {}

    @pytest.mark.asyncio
    async def test_page_size_shrinks_with_higher_dimension(self) -> None:
        ids = [f"chunk-{i}" for i in range(1000)]

        vectors_dim8, call_sizes_dim8 = await _page_and_record(8, ids)
        vectors_dim3072, call_sizes_dim3072 = await _page_and_record(3072, ids)

        assert len(vectors_dim8) == len(ids)
        assert len(vectors_dim3072) == len(ids)
        assert sum(call_sizes_dim8) == len(ids)
        assert sum(call_sizes_dim3072) == len(ids)
        assert len(call_sizes_dim3072) > len(call_sizes_dim8)

    @pytest.mark.asyncio
    async def test_record_cap_binds_at_small_dimension(self) -> None:
        ids = [f"chunk-{i}" for i in range(20)]

        with patch("lightrag.kg.milvus_impl.MILVUS_QUERY_MAX_RECORDS_PER_BATCH", 5):
            vectors, call_sizes = await _page_and_record(1, ids)

        assert len(vectors) == len(ids)
        assert sum(call_sizes) == len(ids)
        assert len(call_sizes) > 1
        assert all(size <= 5 for size in call_sizes)

    @pytest.mark.asyncio
    async def test_returns_buffered_portion_on_page_failure(
        self, lightrag_caplog: pytest.LogCaptureFixture
    ) -> None:
        s = _make_storage(_MockEmbeddingFunc(dim=8))
        await s.upsert({"buffered-1": {"content": "hello"}})

        remaining_ids = [f"srv-{i}" for i in range(10)]
        ids = ["buffered-1", *remaining_ids]

        call_count = 0

        def _side_effect(*, filter: str, **kwargs: Any) -> list[dict[str, Any]]:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise MilvusException(message="milvus down mid-page")
            count = _requested_id_count(filter)
            return [
                {"id": f"row-{call_count}-{i}", "vector": [0.0] * 8}
                for i in range(count)
            ]

        s._client.query.side_effect = _side_effect

        with lightrag_caplog.at_level(logging.ERROR, logger="lightrag"):
            with patch("lightrag.kg.milvus_impl.MILVUS_QUERY_MAX_RECORDS_PER_BATCH", 2):
                vectors = await s.get_vectors_by_ids(ids)

        assert vectors == {"buffered-1": s._pending_vector_docs["buffered-1"].vector}
        assert not any(doc_id.startswith("row-") for doc_id in vectors)
        assert "Error retrieving vectors by IDs" in lightrag_caplog.text


class TestGetByIdsPaging:
    @pytest.mark.asyncio
    async def test_preserves_order_and_excludes_buffered_and_deleted(self) -> None:
        s = _make_storage(_MockEmbeddingFunc(dim=8), meta_fields={"content"})

        await s.upsert({"buffered-1": {"content": "buffered content"}})
        await s.delete(["deleted-1"])

        server_ids = [f"srv-{i}" for i in range(12)]
        ids = [
            server_ids[0],
            "buffered-1",
            server_ids[1],
            "deleted-1",
            *server_ids[2:],
        ]

        captured_filters: list[str] = []
        s._client.query.side_effect = _ordered_echo_side_effect(
            server_ids, captured_filters
        )

        with patch("lightrag.kg.milvus_impl.MILVUS_QUERY_MAX_RECORDS_PER_BATCH", 5):
            results = await s.get_by_ids(ids)

        assert len(results) == len(ids)
        for doc_id, row in zip(ids, results):
            if doc_id == "deleted-1":
                assert row is None
            elif doc_id == "buffered-1":
                assert row["id"] == "buffered-1"
                assert row["content"] == "buffered content"
            else:
                assert row == {"id": doc_id, "content": f"server-{doc_id}"}

        for filter_expr in captured_filters:
            assert "buffered-1" not in filter_expr
            assert "deleted-1" not in filter_expr
        assert s._client.query.call_count > 1

    @pytest.mark.asyncio
    async def test_returns_none_for_ids_missing_from_the_database(self) -> None:
        s = _make_storage(_MockEmbeddingFunc(dim=8), meta_fields={"content"})
        assert await s.get_by_ids(["missing-1"]) == [None]

    @pytest.mark.asyncio
    async def test_escapes_special_characters_across_pages(self) -> None:
        s = _make_storage(_MockEmbeddingFunc(dim=8), meta_fields={"content"})

        quote_id = 'doc"1'
        backslash_id = "doc\\2"
        plain_ids = [f"plain-{i}" for i in range(4)]
        ids = [plain_ids[0], quote_id, plain_ids[1], backslash_id, *plain_ids[2:]]

        captured_filters: list[str] = []

        def _side_effect(*, filter: str, **kwargs: Any) -> list[dict[str, Any]]:
            captured_filters.append(filter)
            return []

        s._client.query.side_effect = _side_effect

        with patch("lightrag.kg.milvus_impl.MILVUS_QUERY_MAX_RECORDS_PER_BATCH", 2):
            await s.get_by_ids(ids)

        assert s._client.query.call_count > 1
        joined_filters = " ".join(captured_filters)
        assert f'"{_escape_milvus_str(quote_id)}"' in joined_filters
        assert f'"{_escape_milvus_str(backslash_id)}"' in joined_filters


class TestOversizePageBisection:
    @pytest.mark.asyncio
    async def test_oversize_metadata_page_is_bisected_and_returns_every_row(
        self,
    ) -> None:
        s = _make_storage(_MockEmbeddingFunc(dim=8), meta_fields={"content"})
        server_ids = [f"srv-{i}" for i in range(12)]
        captured_filters: list[str] = []
        echo = _ordered_echo_side_effect(server_ids, captured_filters)

        def _side_effect(*, filter: str, **kwargs: Any) -> list[dict[str, Any]]:
            if _requested_id_count(filter) > 4:
                raise MilvusException(
                    message="grpc: received message larger than max (x vs. y)"
                )
            return echo(filter=filter, **kwargs)

        s._client.query.side_effect = _side_effect

        with patch("lightrag.kg.milvus_impl.MILVUS_QUERY_MAX_RECORDS_PER_BATCH", 8):
            results = await s.get_by_ids(server_ids)

        assert results == [
            {"id": doc_id, "content": f"server-{doc_id}"} for doc_id in server_ids
        ]

    @pytest.mark.asyncio
    async def test_reduced_page_size_carries_to_the_remaining_pages(self) -> None:
        s = _make_storage(_MockEmbeddingFunc(dim=8), meta_fields={"content"})
        call_sizes: list[int] = []
        s._client.query.side_effect = _oversize_above_side_effect(4, call_sizes)

        ids = [f"srv-{i}" for i in range(20)]
        with patch("lightrag.kg.milvus_impl.MILVUS_QUERY_MAX_RECORDS_PER_BATCH", 8):
            await s.get_by_ids(ids)

        assert call_sizes == [8, 4, 4, 4, 4, 4]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("error", "page_size"),
        [
            pytest.param(_QuotaExhaustedRpcError(), 8, id="bare-resource-exhausted"),
            pytest.param(
                MilvusException(message="milvus down"), 4, id="non-size-error"
            ),
        ],
    )
    async def test_non_oversize_errors_are_not_bisected(
        self, error: Exception, page_size: int
    ) -> None:
        s = _make_storage(_MockEmbeddingFunc(dim=8), meta_fields={"content"})
        call_sizes: list[int] = []

        def _side_effect(*, filter: str, **kwargs: Any) -> list[dict[str, Any]]:
            call_sizes.append(_requested_id_count(filter))
            raise error

        s._client.query.side_effect = _side_effect

        ids = [f"srv-{i}" for i in range(8)]
        with patch(
            "lightrag.kg.milvus_impl.MILVUS_QUERY_MAX_RECORDS_PER_BATCH", page_size
        ):
            results = await s.get_by_ids(ids)

        assert call_sizes == [page_size]
        assert results == []

    @pytest.mark.asyncio
    async def test_overflow_surviving_to_a_single_id_returns_no_partial_rows(
        self, lightrag_caplog: pytest.LogCaptureFixture
    ) -> None:
        s = _make_storage(_MockEmbeddingFunc(dim=8), meta_fields={"content"})
        call_sizes: list[int] = []
        s._client.query.side_effect = _oversize_above_side_effect(0, call_sizes)

        ids = [f"srv-{i}" for i in range(8)]
        with lightrag_caplog.at_level(logging.ERROR, logger="lightrag"):
            with patch("lightrag.kg.milvus_impl.MILVUS_QUERY_MAX_RECORDS_PER_BATCH", 8):
                results = await s.get_by_ids(ids)

        assert call_sizes == [8, 4, 2, 1]
        assert results == []
        assert "Error retrieving vector data" in lightrag_caplog.text
        assert "srv-7" not in lightrag_caplog.text


class TestPagingCooperativeYield:
    @pytest.mark.asyncio
    async def test_event_loop_runs_between_pages(self) -> None:
        s = _make_storage(_MockEmbeddingFunc(dim=8), meta_fields={"content"})
        s._client.query.side_effect = lambda **kwargs: []

        observer_ticks = 0

        async def observer() -> None:
            nonlocal observer_ticks
            while True:
                observer_ticks += 1
                await asyncio.sleep(0)

        ids = [f"srv-{i}" for i in range(20)]
        task = asyncio.create_task(observer())
        try:
            with patch("lightrag.kg.milvus_impl.MILVUS_QUERY_MAX_RECORDS_PER_BATCH", 2):
                await s.get_by_ids(ids)  # 10 pages
        finally:
            task.cancel()

        assert observer_ticks >= 9


# ---------------------------------------------------------------------------
# Strong read consistency on point reads vs. default on vector search
# ---------------------------------------------------------------------------


def _assert_every_query_strong(storage: MilvusVectorDBStorage) -> None:
    calls = storage._client.query.call_args_list
    assert calls, "the read never reached client.query"
    levels = [call.kwargs.get("consistency_level") for call in calls]
    assert levels == ["Strong"] * len(calls)


class TestMilvusReadConsistency:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "invoke",
        [
            pytest.param(lambda s: s.get_by_id("rel-1"), id="get_by_id"),
            pytest.param(lambda s: s.get_by_ids(["rel-1", "rel-2"]), id="get_by_ids"),
            pytest.param(lambda s: s.is_empty(), id="is_empty"),
        ],
    )
    async def test_point_reads_use_strong_consistency(self, invoke) -> None:
        storage = _make_storage(
            namespace="relationships", meta_fields={"content", "src_id", "tgt_id"}
        )
        await invoke(storage)
        _assert_every_query_strong(storage)

    @pytest.mark.asyncio
    async def test_get_exact_count_reads_strong(self) -> None:
        storage = _make_storage(
            namespace="relationships", meta_fields={"content", "src_id", "tgt_id"}
        )
        storage._client.query.return_value = [{"count(*)": 3}]
        assert await storage.get_exact_count() == 3
        _assert_every_query_strong(storage)
        assert storage._client.query.call_args.kwargs["output_fields"] == ["count(*)"]

    @pytest.mark.asyncio
    async def test_strict_read_raises_where_get_by_ids_reports_nothing(self) -> None:
        storage = _make_storage(
            namespace="relationships", meta_fields={"content", "src_id", "tgt_id"}
        )
        storage._client.query.side_effect = ConnectionError("milvus down")

        assert await storage.get_by_ids(["rel-1"]) == []
        with pytest.raises(ConnectionError):
            await storage.get_by_ids_strict(["rel-1"])

    @pytest.mark.asyncio
    async def test_delete_entity_relation_victim_lookup_reads_strong(self) -> None:
        storage = _make_storage(
            namespace="relationships", meta_fields={"content", "src_id", "tgt_id"}
        )
        storage._client.query.return_value = [{"id": "rel-A-B"}]
        await storage.delete_entity_relation("A")
        _assert_every_query_strong(storage)
        deleted = [
            pk
            for call in storage._client.delete.call_args_list
            for pk in call.kwargs.get("pks", [])
        ]
        assert deleted == ["rel-A-B"]

    @pytest.mark.asyncio
    async def test_vector_search_stays_on_collection_default(self) -> None:
        storage = _make_storage(
            namespace="relationships", meta_fields={"content", "src_id", "tgt_id"}
        )
        await storage.query("anything", top_k=3, query_embedding=[0.0] * 8)
        storage._client.search.assert_called_once()
        assert "consistency_level" not in storage._client.search.call_args.kwargs
