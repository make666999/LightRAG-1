"""Offline unit tests for OpenSearchDocStatusStorage."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip(
    "opensearchpy",
    reason="opensearchpy is required for OpenSearch storage tests",
)

from opensearchpy.exceptions import OpenSearchException  # type: ignore  # noqa: E402

import lightrag.kg.opensearch_impl as opensearch_module  # noqa: E402
from lightrag.base import DocProcessingStatus, DocStatus  # noqa: E402
from lightrag.kg.opensearch_impl import (  # noqa: E402
    ClientManager,
    OpenSearchDocStatusStorage,
)

from ._helpers import (  # noqa: E402
    LazyDropRecreateMixin,
    MockEmbeddingFunc,
    ReadinessDemotionMixin,
    default_global_config,
    make_client,
    patch_opensearch_locks,
)

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def patch_locks():
    with patch_opensearch_locks(patch_shard_doc=True):
        yield


@pytest.fixture
def global_config():
    return default_global_config()


@pytest.fixture
def embed_func():
    return MockEmbeddingFunc()


@pytest.fixture
def mock_client():
    return make_client()


class TestDocStatusStorage(LazyDropRecreateMixin, ReadinessDemotionMixin):
    """Tests for OpenSearchDocStatusStorage aggregations, pagination, and normalization."""

    missing_read_rpc = "search"

    def _make(self, global_config, embed_func, workspace: str = "test"):
        return OpenSearchDocStatusStorage(
            namespace="doc_status",
            global_config=global_config,
            embedding_func=embed_func,
            workspace=workspace,
        )

    async def _invoke_missing_read(self, storage):
        return await storage.get_all_status_counts()

    def _expected_missing_read_result(self):
        return {}

    async def _write_to_recreate(self, storage):
        await storage.upsert({"d1": {"status": "pending"}})

    @pytest.mark.asyncio
    async def test_get_by_id(self, global_config, embed_func, mock_client):
        mock_client.mget = AsyncMock(
            return_value={
                "docs": [
                    {
                        "_id": "doc-abc",
                        "found": True,
                        "_source": {"status": "processed", "file_path": "/a.txt"},
                    }
                ]
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            doc = await s.get_by_id("doc-abc")
            assert doc["status"] == "processed"
            assert doc["_id"] == "doc-abc"
            mock_client.mget.assert_awaited_once_with(
                index=s._index_name, body={"ids": ["doc-abc"]}
            )

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, global_config, embed_func, mock_client):
        mock_client.mget = AsyncMock(
            return_value={"docs": [{"_id": "missing", "found": False}]}
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.get_by_id("missing") is None
            mock_client.get.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_upsert_sets_chunks_list_default(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (1, [])
                s = self._make(global_config, embed_func)
                await s.initialize()
                await s.upsert({"d1": {"status": "pending"}})
                actions = mock_bulk.call_args[0][1]
                assert actions[0]["_source"]["chunks_list"] == []

    @pytest.mark.asyncio
    async def test_upsert_raises_on_bulk_item_failures(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            with patch(
                "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
            ) as mock_bulk:
                mock_bulk.return_value = (
                    0,
                    [
                        {
                            "index": {
                                "_id": "d1",
                                "status": 400,
                                "error": {
                                    "type": "mapper_parsing_exception",
                                    "reason": "bad status",
                                },
                            }
                        },
                        {"index": {"_id": "d2", "status": 503, "error": "down"}},
                    ],
                )
                s = self._make(global_config, embed_func)
                await s.initialize()
                with pytest.raises(RuntimeError) as excinfo:
                    await s.upsert(
                        {"d1": {"status": "pending"}, "d2": {"status": "pending"}}
                    )
        error_text = str(excinfo.value)
        assert "doc-status ops failed" in error_text
        assert "1 retryable" in error_text
        assert "1 non-retryable" in error_text
        assert "d1" in error_text
        assert "d2" in error_text
        assert "status=400" in error_text
        assert "bad status" in error_text

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [200, 404, 403])
    async def test_status_delete_item_results(self, monkeypatch, status):
        """DocStatus delete honors per-item failures without rejecting 200 or 404."""
        item = {"delete": {"_id": "row", "status": status}}
        storage = OpenSearchDocStatusStorage("doc_status", {}, None, "test")
        storage._index_ready = True
        storage.client = SimpleNamespace()
        monkeypatch.setattr(
            opensearch_module,
            "_run_chunked_async_bulk",
            AsyncMock(
                return_value=(int(status == 200), [] if status == 200 else [item])
            ),
        )
        if status in (200, 404):
            await storage.delete(["row"])
        else:
            with pytest.raises(OpenSearchException, match="bulk deletion failed"):
                await storage.delete(["row"])

    @pytest.mark.asyncio
    async def test_get_status_counts(self, global_config, embed_func, mock_client):
        mock_client.search = AsyncMock(
            return_value={
                "hits": {"hits": [], "total": {"value": 0}},
                "aggregations": {
                    "status_counts": {
                        "buckets": [
                            {"key": "processed", "doc_count": 3},
                            {"key": "pending", "doc_count": 1},
                        ]
                    }
                },
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            counts = await s.get_status_counts()
            assert counts == {"processed": 3, "pending": 1}

    @pytest.mark.asyncio
    async def test_get_all_status_counts_includes_all(
        self, global_config, embed_func, mock_client
    ):
        mock_client.search = AsyncMock(
            return_value={
                "hits": {"hits": [], "total": {"value": 0}},
                "aggregations": {
                    "status_counts": {
                        "buckets": [
                            {"key": "processed", "doc_count": 5},
                            {"key": "failed", "doc_count": 2},
                        ]
                    }
                },
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            counts = await s.get_all_status_counts()
            assert counts["all"] == 7
            assert counts["processed"] == 5

    @pytest.mark.asyncio
    async def test_get_docs_by_statuses(self, global_config, embed_func, mock_client):
        mock_client.search = AsyncMock(
            return_value={
                "hits": {
                    "hits": [
                        {
                            "_id": "d1",
                            "_source": {
                                "status": "processed",
                                "file_path": "/a.txt",
                                "content_summary": "s",
                                "content_length": 10,
                                "chunks_count": 1,
                                "created_at": 100,
                                "updated_at": 200,
                            },
                            "sort": ["d1"],
                        },
                    ],
                    "total": {"value": 1},
                },
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            result = await s.get_docs_by_statuses(
                [DocStatus.PROCESSED, DocStatus.FAILED]
            )
            assert "d1" in result
            assert isinstance(result["d1"], DocProcessingStatus)
            body = mock_client.search.await_args.kwargs["body"]
            assert body["query"] == {"terms": {"status": ["processed", "failed"]}}

    @pytest.mark.asyncio
    async def test_get_docs_paginated(self, global_config, embed_func, mock_client):
        mock_client.count = AsyncMock(return_value={"count": 50})
        mock_client.search = AsyncMock(
            return_value={
                "hits": {
                    "hits": [
                        {
                            "_id": "d1",
                            "_source": {
                                "status": "processed",
                                "file_path": "/a.txt",
                                "content_summary": "s",
                                "content_length": 10,
                                "chunks_count": 1,
                                "created_at": 100,
                                "updated_at": 200,
                            },
                            "sort": [200, "d1"],
                        },
                    ],
                    "total": {"value": 50},
                },
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            docs, total = await s.get_docs_paginated(page=1, page_size=10)
            assert total == 50
            assert len(docs) == 1
            assert docs[0][0] == "d1"
            assert mock_client.search.await_count == 1
            body = mock_client.search.call_args.kwargs.get(
                "body"
            ) or mock_client.search.call_args[1].get("body", {})
            assert "search_after" not in body

    @pytest.mark.asyncio
    async def test_get_docs_paginated_page2_uses_search_after(
        self, global_config, embed_func, mock_client
    ):
        mock_client.count = AsyncMock(return_value={"count": 50})
        call_count = {"n": 0}

        async def search_side_effect(*args, **kwargs):
            call_count["n"] += 1
            body = kwargs.get("body", {})
            if "search_after" not in body:
                return {
                    "hits": {
                        "hits": [
                            {
                                "_id": f"skip{i}",
                                "_source": {
                                    "status": "processed",
                                    "file_path": f"/{i}.txt",
                                    "content_summary": "s",
                                    "content_length": 1,
                                    "chunks_count": 1,
                                    "created_at": 100,
                                    "updated_at": 100 + i,
                                },
                                "sort": [100 + i, f"skip{i}"],
                            }
                            for i in range(10)
                        ],
                        "total": {"value": 50},
                    }
                }
            return {
                "hits": {
                    "hits": [
                        {
                            "_id": "page2_doc",
                            "_source": {
                                "status": "pending",
                                "file_path": "/p2.txt",
                                "content_summary": "s",
                                "content_length": 1,
                                "chunks_count": 1,
                                "created_at": 200,
                                "updated_at": 300,
                            },
                            "sort": [300, "page2_doc"],
                        }
                    ],
                    "total": {"value": 50},
                }
            }

        mock_client.search = AsyncMock(side_effect=search_side_effect)
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            docs, total = await s.get_docs_paginated(page=2, page_size=10)
            assert total == 50
            assert len(docs) == 1
            assert docs[0][0] == "page2_doc"
            assert mock_client.search.await_count == 2

    @pytest.mark.asyncio
    async def test_get_docs_paginated_empty_index(
        self, global_config, embed_func, mock_client
    ):
        mock_client.count = AsyncMock(return_value={"count": 0})
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            docs, total = await s.get_docs_paginated(page=1, page_size=10)
            assert total == 0
            assert docs == []
            mock_client.search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_docs_paginated_page_beyond_total(
        self, global_config, embed_func, mock_client
    ):
        mock_client.count = AsyncMock(return_value={"count": 5})
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            docs, total = await s.get_docs_paginated(page=100, page_size=10)
            assert total == 5
            assert docs == []
            mock_client.create_pit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_docs_paginated_with_status_filter(
        self, global_config, embed_func, mock_client
    ):
        mock_client.count = AsyncMock(return_value={"count": 3})
        mock_client.search = AsyncMock(
            return_value={"hits": {"hits": [], "total": {"value": 3}}}
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            _, total = await s.get_docs_paginated(
                status_filter=DocStatus.PROCESSED, page=1, page_size=10
            )
            assert total == 3
            count_body = mock_client.count.call_args.kwargs.get("body", {})
            assert count_body["query"] == {"term": {"status": "processed"}}

    @pytest.mark.asyncio
    async def test_get_docs_paginated_with_status_filters(
        self, global_config, embed_func, mock_client
    ):
        mock_client.count = AsyncMock(return_value={"count": 2})
        mock_client.search = AsyncMock(
            return_value={"hits": {"hits": [], "total": {"value": 2}}}
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            docs, total = await s.get_docs_paginated(
                status_filter=DocStatus.PROCESSED,
                status_filters=[DocStatus.PARSING, DocStatus.ANALYZING],
                page=1,
                page_size=10,
            )
            assert total == 2
            assert docs == []
            count_body = mock_client.count.call_args.kwargs.get("body", {})
            assert count_body["query"] == {
                "terms": {"status": ["analyzing", "parsing"]}
            }

    @pytest.mark.asyncio
    async def test_get_doc_by_file_path(self, global_config, embed_func, mock_client):
        mock_client.search = AsyncMock(
            return_value={
                "hits": {
                    "hits": [
                        {
                            "_id": "d1",
                            "_source": {
                                "file_path": "/test.txt",
                                "status": "processed",
                            },
                        },
                    ],
                    "total": {"value": 1},
                },
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            doc = await s.get_doc_by_file_path("/test.txt")
            assert doc is not None
            assert doc["_id"] == "d1"

    @pytest.mark.asyncio
    async def test_get_doc_by_file_path_not_found(
        self, global_config, embed_func, mock_client
    ):
        mock_client.search = AsyncMock(
            return_value={"hits": {"hits": [], "total": {"value": 0}}}
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            assert await s.get_doc_by_file_path("/nope.txt") is None

    @pytest.mark.asyncio
    async def test_get_doc_by_file_basename_returns_tuple_on_hit(
        self, global_config, embed_func, mock_client
    ):
        mock_client.search = AsyncMock(
            return_value={
                "hits": {
                    "hits": [
                        {
                            "_id": "doc-1",
                            "_source": {
                                "file_path": "report.pdf",
                                "status": "processed",
                            },
                        },
                    ],
                    "total": {"value": 1},
                },
            }
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            result = await s.get_doc_by_file_basename("report.pdf")
            assert result is not None
            doc_id, doc = result
            assert doc_id == "doc-1"
            assert doc["file_path"] == "report.pdf"
            body = mock_client.search.call_args.kwargs.get(
                "body"
            ) or mock_client.search.call_args[1].get("body", {})
            assert body["query"] == {
                "bool": {
                    "filter": [{"term": {"file_path": "report.pdf"}}],
                    "must_not": [{"term": {"metadata.is_duplicate": True}}],
                }
            }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("method", "arg", "expect_search"),
        [
            ("get_doc_by_file_basename", "", False),
            ("get_doc_by_file_basename", "unknown_source", False),
            ("get_doc_by_file_basename", "missing.pdf", True),
            ("get_doc_by_content_hash", "", False),
        ],
        ids=[
            "basename-empty",
            "basename-unknown-source",
            "basename-miss",
            "content-hash-empty",
        ],
    )
    async def test_doc_lookup_short_circuits_and_miss(
        self, global_config, embed_func, mock_client, method, arg, expect_search
    ):
        mock_client.search = AsyncMock(
            return_value={"hits": {"hits": [], "total": {"value": 0}}}
        )
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            mock_client.search.reset_mock()
            assert await getattr(s, method)(arg) is None
            if expect_search:
                mock_client.search.assert_awaited_once()
            else:
                mock_client.search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ensure_content_hash_mapping_added_when_missing(
        self, global_config, embed_func, mock_client
    ):
        mock_client.indices.exists = AsyncMock(return_value=True)
        mock_client.indices.get_mapping = AsyncMock(
            return_value={
                "test_doc_status": {
                    "mappings": {
                        "properties": {
                            "__mirrored_id": {"type": "keyword"},
                            "status": {"type": "keyword"},
                            "file_path": {"type": "keyword"},
                        }
                    }
                }
            }
        )
        mock_client.indices.put_mapping = AsyncMock()
        mock_client.count = AsyncMock(return_value={"count": 0})
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            property_bodies = [
                call.kwargs["body"]
                for call in mock_client.indices.put_mapping.await_args_list
                if "properties" in call.kwargs["body"]
            ]
            assert property_bodies == [
                {"properties": {"content_hash": {"type": "keyword"}}}
            ]

    @pytest.mark.asyncio
    async def test_ensure_content_hash_mapping_skipped_when_present(
        self, global_config, embed_func, mock_client
    ):
        mock_client.indices.exists = AsyncMock(return_value=True)
        mock_client.indices.get_mapping = AsyncMock(
            return_value={
                "test_doc_status": {
                    "mappings": {
                        "properties": {
                            "__mirrored_id": {"type": "keyword"},
                            "content_hash": {"type": "keyword"},
                        }
                    }
                }
            }
        )
        mock_client.indices.put_mapping = AsyncMock()
        mock_client.count = AsyncMock(return_value={"count": 0})
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            property_calls = [
                call
                for call in mock_client.indices.put_mapping.await_args_list
                if "properties" in call.kwargs["body"]
            ]
            assert property_calls == []

    @pytest.mark.asyncio
    async def test_prepare_doc_status_data(self, global_config, embed_func):
        s = self._make(global_config, embed_func)
        raw = {"_id": "x", "status": "processed", "error": "oops"}
        data = s._prepare_doc_status_data(raw)
        assert "_id" not in data
        assert data["error_msg"] == "oops"
        assert "error" not in data
        assert data["file_path"] == "no-file-path"
        assert data["metadata"] == {}

    @pytest.mark.asyncio
    async def test_reads_short_circuit_after_drop(
        self, global_config, embed_func, mock_client
    ):
        with patch.object(ClientManager, "get_client", return_value=mock_client):
            s = self._make(global_config, embed_func)
            await s.initialize()
            await s.drop()

            assert await s.get_all_status_counts() == {}
            assert await s.get_docs_paginated(page=1, page_size=10) == ([], 0)
            assert await s.get_doc_by_file_path("/a.txt") is None
            assert await s.get_docs_by_statuses([DocStatus.PROCESSED]) == {}

            mock_client.count.assert_not_awaited()
            mock_client.search.assert_not_awaited()
            mock_client.create_pit.assert_not_awaited()
