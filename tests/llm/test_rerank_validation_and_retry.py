"""Rerank result validation, negative top_n rejection, null chunk aggregation, and retry classification."""

from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest
from tenacity import wait_none

from lightrag.base import QueryParam
from lightrag.rerank import aggregate_chunk_scores, cohere_rerank, generic_rerank_api
from lightrag.utils import apply_rerank_if_enabled, process_chunks_unified
from tests._rerank_session import mock_rerank_session

pytestmark = pytest.mark.offline


# --------------------------------------------------------------------------- #
# apply_rerank_if_enabled & process_chunks_unified result validation
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_boolean_index_is_not_treated_as_document_one():
    async def rerank_func(**_kwargs):
        return [{"index": True, "relevance_score": 0.9}]

    documents = [{"content": "first"}, {"content": "second"}]

    result = await apply_rerank_if_enabled(
        query="query",
        retrieved_docs=documents,
        global_config={"rerank_model_func": rerank_func},
    )

    assert result == documents


@pytest.mark.asyncio
async def test_valid_results_survive_malformed_items():
    async def rerank_func(**_kwargs):
        return [None, {"index": 1, "relevance_score": "0.8"}, {"index": 0}]

    documents = [{"content": "first"}, {"content": "second"}]

    result = await apply_rerank_if_enabled(
        query="query",
        retrieved_docs=documents,
        global_config={"rerank_model_func": rerank_func},
    )

    assert result == [{"content": "second", "rerank_score": 0.8}]


@pytest.mark.asyncio
async def test_duplicate_index_keeps_its_best_score_and_one_chunk_top_k_slot():
    async def rerank_func(**_kwargs):
        return [
            {"index": 0, "relevance_score": 0.3},
            {"index": 1, "relevance_score": 0.78},
            {"index": 0, "relevance_score": 0.99},
            {"index": 0, "relevance_score": 0.85},
        ]

    result = await process_chunks_unified(
        query="query",
        unique_chunks=[{"content": "first"}, {"content": "second"}],
        query_param=QueryParam(enable_rerank=True, chunk_top_k=2),
        global_config={"rerank_model_func": rerank_func},
    )

    assert result == [
        {"content": "first", "rerank_score": 0.99, "id": "DC1"},
        {"content": "second", "rerank_score": 0.78, "id": "DC2"},
    ]


@pytest.mark.asyncio
async def test_results_are_sorted_by_score_not_provider_order():
    async def rerank_func(**_kwargs):
        return [
            {"index": 0, "relevance_score": 0.1},
            {"index": 1, "relevance_score": 0.9},
            {"index": 2, "relevance_score": 0.5},
        ]

    documents = [{"content": "low"}, {"content": "high"}, {"content": "mid"}]

    result = await apply_rerank_if_enabled(
        query="query",
        retrieved_docs=documents,
        global_config={"rerank_model_func": rerank_func},
    )

    assert [d["content"] for d in result] == ["high", "mid", "low"]
    assert [d["rerank_score"] for d in result] == [0.9, 0.5, 0.1]


@pytest.mark.asyncio
async def test_legacy_documents_are_not_inferred_from_later_index_metadata():
    legacy_results = [
        {"content": "ranked first"},
        {"content": "ranked second", "index": "source-index"},
    ]

    async def rerank_func(**_kwargs):
        return legacy_results

    result = await apply_rerank_if_enabled(
        query="query",
        retrieved_docs=[{"content": "original"}],
        global_config={"rerank_model_func": rerank_func},
    )

    assert result == legacy_results


@pytest.mark.asyncio
async def test_negative_top_n_keeps_original_documents_without_calling_rerank():
    calls = []

    async def rerank_func(**kwargs):
        calls.append(kwargs)
        return [{"content": "b"}, {"content": "a"}]

    documents = [{"content": "a"}, {"content": "b"}]

    result = await apply_rerank_if_enabled(
        query="query",
        retrieved_docs=documents,
        global_config={"rerank_model_func": rerank_func},
        top_n=-1,
    )

    assert result == documents
    assert calls == []


# --------------------------------------------------------------------------- #
# generic_rerank_api / cohere_rerank negative top_n & aggregate_chunk_scores
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_negative_top_n_raises_before_any_request():
    session, _ = mock_rerank_session({"results": []})

    with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
        with pytest.raises(ValueError, match="top_n"):
            await generic_rerank_api(
                query="q",
                documents=["a", "b"],
                model="m",
                base_url="http://test.com/rerank",
                api_key="k",
                top_n=-1,
            )

    session.post.assert_not_called()

    session2, _ = mock_rerank_session(
        {
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.7},
                {"index": 2, "relevance_score": 0.5},
            ]
        }
    )
    with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session2):
        with pytest.raises(ValueError, match="top_n"):
            await cohere_rerank(
                query="q",
                documents=["doc0", "doc1", "doc2"],
                api_key="k",
                base_url="http://test.com/rerank",
                enable_chunking=True,
                top_n=-1,
            )


@pytest.mark.asyncio
async def test_top_n_none_and_zero_are_unaffected():
    session, captured = mock_rerank_session({"results": []})
    with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
        await generic_rerank_api(
            query="q",
            documents=["a"],
            model="m",
            base_url="http://test.com/rerank",
            api_key="k",
            top_n=None,
        )
    assert "top_n" not in captured["kwargs"]["json"]

    session, captured = mock_rerank_session({"results": []})
    with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
        await generic_rerank_api(
            query="q",
            documents=["a"],
            model="m",
            base_url="http://test.com/rerank",
            api_key="k",
            top_n=0,
        )
    assert captured["kwargs"]["json"]["top_n"] == 0


def test_aggregate_chunk_scores_handles_none_index_and_score():
    chunk_results = [
        {"index": None, "relevance_score": 0.9},
        {"index": 0, "relevance_score": None},
        {"index": 0, "relevance_score": 0.8},
        {"index": "invalid", "relevance_score": 0.7},
    ]

    res = aggregate_chunk_scores(
        chunk_results=chunk_results,
        doc_indices=[0],
        num_original_docs=1,
        aggregation="mean",
    )
    assert res == [{"index": 0, "relevance_score": 0.8}]


# --------------------------------------------------------------------------- #
# generic_rerank_api retry classification
# --------------------------------------------------------------------------- #

_URL = "http://rerank.test/v1/rerank"


class TestRerankRetryClassification:
    @pytest.fixture(autouse=True)
    def _no_backoff(self, monkeypatch):
        monkeypatch.setattr(generic_rerank_api.retry, "wait", wait_none())

    @staticmethod
    def _response(status: int, json_body=None) -> Mock:
        response = Mock()
        response.status = status
        response.headers = {"content-type": "application/json"}
        response.request_info = aiohttp.RequestInfo(
            url=_URL, method="POST", headers={}, real_url=_URL
        )
        response.history = ()
        response.text = AsyncMock(return_value=f'{{"detail": "HTTP {status}"}}')
        response.json = AsyncMock(return_value=json_body)
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)
        return response

    @staticmethod
    def _session(*outcomes) -> Mock:
        posts = []
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                ctx = Mock()
                ctx.__aenter__ = AsyncMock(side_effect=outcome)
                ctx.__aexit__ = AsyncMock(return_value=None)
                posts.append(ctx)
            else:
                posts.append(outcome)
        session = Mock()
        session.post = Mock(side_effect=posts)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        return session

    @staticmethod
    async def _rerank(session: Mock):
        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
            return await generic_rerank_api(
                query="q",
                documents=["a", "b"],
                model="m",
                base_url=_URL,
                api_key="key",
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("status", "attempts"),
        [pytest.param(401, 1, id="permanent"), pytest.param(503, 3, id="transient")],
    )
    async def test_status_retry_follows_the_shared_classification(
        self, status, attempts
    ):
        session = self._session(*[self._response(status) for _ in range(3)])

        with pytest.raises(aiohttp.ClientResponseError) as excinfo:
            await self._rerank(session)

        assert session.post.call_count == attempts
        assert excinfo.value.status == status

    @pytest.mark.asyncio
    async def test_transient_status_then_success_returns_results(self):
        ok = self._response(200, {"results": [{"index": 1, "relevance_score": 0.9}]})
        session = self._session(self._response(503), ok)

        result = await self._rerank(session)

        assert session.post.call_count == 2
        assert result == [{"index": 1, "relevance_score": 0.9}]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("make_exc", "attempts"),
        [
            pytest.param(lambda: aiohttp.InvalidURL("not-a-url"), 1, id="invalid-url"),
            pytest.param(
                lambda: aiohttp.ClientConnectionError("connection reset"),
                3,
                id="connection",
            ),
        ],
    )
    async def test_transport_retry_follows_the_shared_classification(
        self, make_exc, attempts
    ):
        session = self._session(*[make_exc() for _ in range(3)])

        with pytest.raises(type(make_exc())):
            await self._rerank(session)

        assert session.post.call_count == attempts
