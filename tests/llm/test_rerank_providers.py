"""Provider-specific rerank request/response shape and API key resolution (Aliyun DashScope and Jina)."""

from unittest.mock import patch

import pytest

from lightrag.rerank import ali_rerank, jina_rerank
from tests._rerank_session import mock_rerank_session

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


def _wrap_results(provider: str, results: list) -> dict:
    if provider == "ali":
        return {"output": {"results": results}, "usage": {"total_tokens": 10}}
    return {"results": results}


def _provider_func(provider: str):
    return ali_rerank if provider == "ali" else jina_rerank


# --------------------------------------------------------------------------- #
# Request payload shape
# --------------------------------------------------------------------------- #


async def test_ali_request_payload_shape():
    session, captured = mock_rerank_session({"output": {"results": []}})

    with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
        await ali_rerank(
            query="capital of France",
            documents=["Paris is the capital.", "Tokyo is a city."],
            api_key="test-key",
            top_n=1,
            extra_body={"instruction": "focus on dates"},
        )

    payload = captured["kwargs"]["json"]
    assert payload["model"] == "gte-rerank-v2"
    assert "dashscope.aliyuncs.com" in captured["args"][0]
    assert payload["input"] == {
        "query": "capital of France",
        "documents": ["Paris is the capital.", "Tokyo is a city."],
    }
    assert payload["parameters"]["top_n"] == 1
    assert payload["parameters"]["return_documents"] is False
    assert payload["parameters"]["instruction"] == "focus on dates"
    for flat_key in ("query", "documents", "top_n", "instruction"):
        assert flat_key not in payload


async def test_jina_request_payload_shape():
    session, captured = mock_rerank_session({"results": []})

    with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
        await jina_rerank(
            query="capital of France",
            documents=["Paris is the capital.", "Tokyo is a city."],
            api_key="test-key",
            top_n=1,
            extra_body={"instruction": "focus on dates"},
        )

    payload = captured["kwargs"]["json"]
    assert payload["model"] == "jina-reranker-v2-base-multilingual"
    assert captured["args"][0] == "https://api.jina.ai/v1/rerank"
    assert payload["query"] == "capital of France"
    assert payload["documents"] == ["Paris is the capital.", "Tokyo is a city."]
    assert payload["return_documents"] is False
    assert payload["top_n"] == 1
    assert payload["instruction"] == "focus on dates"
    assert "input" not in payload
    assert "parameters" not in payload


# --------------------------------------------------------------------------- #
# Response parsing & degradation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("provider", ["ali", "jina"])
async def test_parses_scored_results(provider):
    items = [
        {"index": 1, "relevance_score": 0.95},
        {"index": 0, "relevance_score": 0.42},
    ]
    session, _ = mock_rerank_session(_wrap_results(provider, items))

    with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
        result = await _provider_func(provider)(
            query="q", documents=["doc0", "doc1"], api_key="test-key"
        )

    assert result == items


@pytest.mark.parametrize("provider", ["ali", "jina"])
@pytest.mark.parametrize(
    ("body_kind", "documents"),
    [
        pytest.param("empty-results", ["doc0", "doc1"], id="empty-results-list"),
        pytest.param("missing-key", ["doc0"], id="missing-results-key"),
        pytest.param("empty-results", [], id="empty-documents"),
    ],
)
async def test_empty_and_missing_shapes_return_empty(provider, body_kind, documents):
    body = (
        _wrap_results(provider, [])
        if body_kind == "empty-results"
        else {"unexpected": "shape"}
    )
    session, _ = mock_rerank_session(body)

    with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
        result = await _provider_func(provider)(
            query="q", documents=documents, api_key="test-key"
        )

    assert result == []


async def test_ali_output_results_not_a_list_returns_empty_and_warns():
    session, _ = mock_rerank_session({"output": {"results": {"not": "a list"}}})

    with (
        patch("lightrag.rerank.aiohttp.ClientSession", return_value=session),
        patch("lightrag.rerank.logger.warning") as mock_warning,
    ):
        result = await ali_rerank(query="q", documents=["doc0"], api_key="test-key")

    assert result == []
    assert mock_warning.call_count == 2
    assert "to be list" in mock_warning.call_args_list[0].args[0]


@pytest.mark.parametrize("provider", ["ali", "jina"])
async def test_malformed_result_items_are_dropped_not_fatal(provider):
    raw = [
        None,
        {"index": 5, "relevance_score": 0.9},
        {"index": 0, "relevance_score": "bad"},
        {"index": 1, "relevance_score": 0.6},
    ]
    session, _ = mock_rerank_session(_wrap_results(provider, raw))

    with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
        result = await _provider_func(provider)(
            query="q", documents=["doc0", "doc1"], api_key="test-key"
        )

    assert result == [{"index": 1, "relevance_score": 0.6}]


# --------------------------------------------------------------------------- #
# API key precedence: explicit > provider env > RERANK_BINDING_API_KEY
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("provider", "provider_env"),
    [("ali", "DASHSCOPE_API_KEY"), ("jina", "JINA_API_KEY")],
)
@pytest.mark.parametrize(
    ("explicit_key", "provider_val", "generic_val", "expected"),
    [
        pytest.param(
            "explicit-key", "env-key", None, "explicit-key", id="explicit-wins"
        ),
        pytest.param(None, "env-key", None, "env-key", id="provider-env-fallback"),
        pytest.param(
            None, None, "generic-key", "generic-key", id="generic-env-fallback"
        ),
        pytest.param(
            None,
            "provider-key",
            "generic-key",
            "provider-key",
            id="provider-over-generic",
        ),
    ],
)
async def test_api_key_resolution_precedence(
    monkeypatch,
    provider,
    provider_env,
    explicit_key,
    provider_val,
    generic_val,
    expected,
):
    if provider_val is None:
        monkeypatch.delenv(provider_env, raising=False)
    else:
        monkeypatch.setenv(provider_env, provider_val)
    if generic_val is None:
        monkeypatch.delenv("RERANK_BINDING_API_KEY", raising=False)
    else:
        monkeypatch.setenv("RERANK_BINDING_API_KEY", generic_val)

    session, captured = mock_rerank_session(_wrap_results(provider, []))
    kwargs = {"api_key": explicit_key} if explicit_key is not None else {}

    with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
        await _provider_func(provider)(query="q", documents=["a"], **kwargs)

    assert captured["kwargs"]["headers"]["Authorization"] == f"Bearer {expected}"
