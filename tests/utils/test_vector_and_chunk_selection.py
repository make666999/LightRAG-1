"""Tests for ``cosine_similarity``, ``pick_by_vector_similarity``, and ``pick_by_weighted_polling``.

Covers:
- ``cosine_similarity`` zero, unit, orthogonal, NaN, and Inf handling.
- ``pick_by_vector_similarity`` missing-vector inconsistency fallback/warning
  and deterministic first-occurrence ordering across tie-breaks, duplicates,
  storage error fallback, empty inputs, and query-embedding computation.
- ``pick_by_weighted_polling`` non-positive and positive max-chunk bounds.
"""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from lightrag.utils import (
    cosine_similarity,
    pick_by_vector_similarity,
    pick_by_weighted_polling,
)
from lightrag.utils import logger as utils_logger

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("v1", "v2", "expected"),
    [
        pytest.param(np.zeros(3), np.zeros(3), 0.0, id="zero-zero"),
        pytest.param(np.zeros(2), np.array([1.0, 0.0]), 0.0, id="zero-nonzero"),
        pytest.param(
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            1.0,
            id="identical-unit",
        ),
        pytest.param(np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.0, id="orthogonal"),
        pytest.param(np.array([np.nan, 1.0]), np.array([1.0, 1.0]), 0.0, id="nan"),
        pytest.param(np.array([np.inf, 1.0]), np.array([1.0, 1.0]), 0.0, id="inf"),
    ],
)
def test_cosine_similarity(v1, v2, expected):
    score = cosine_similarity(v1, v2)
    assert not np.isnan(score)
    assert score == pytest.approx(expected)


# ---------------------------------------------------------------------------
# pick_by_vector_similarity
# ---------------------------------------------------------------------------


class _VectorStore:
    """Stub vector store returning existing vectors in reversed key order."""

    def __init__(self, vectors):
        self.vectors = vectors
        self.requested_ids = None

    async def get_vectors_by_ids(self, ids):
        self.requested_ids = list(ids)
        # The returned mapping's order must not control tie-breaking either.
        return {key: self.vectors[key] for key in reversed(ids) if key in self.vectors}


async def _select(entities, vectors, count, *, embedding=None, query_embedding=None):
    store = _VectorStore(vectors)
    result = await pick_by_vector_similarity(
        query="evidence",
        text_chunks_storage=None,
        chunks_vdb=store,
        num_of_chunks=count,
        entity_info=entities,
        embedding_func=embedding,
        query_embedding=(
            np.array([1.0, 0.0]) if query_embedding is None else query_embedding
        ),
    )
    return result, store


async def test_partial_missing_vectors_falls_back_and_warns():
    query_embedding = np.array([1.0, 0.0, 0.0])
    entity_info = [{"sorted_chunks": ["c0", "c1", "c2"]}]
    # c2 has no stored vector; the function should not rank the partial set.
    vdb = _VectorStore(
        {
            "c0": np.array([1.0, 0.0, 0.0]),
            "c1": np.array([0.9, 0.1, 0.0]),
        }
    )

    with patch.object(utils_logger, "warning") as mock_warning:
        selected = await pick_by_vector_similarity(
            query="q",
            text_chunks_storage=None,
            chunks_vdb=vdb,
            num_of_chunks=5,
            entity_info=entity_info,
            embedding_func=None,
            query_embedding=query_embedding,
        )

    assert selected == []
    assert mock_warning.call_count == 1
    fmt, expected, retrieved, missing, sample = mock_warning.call_args.args
    assert "data inconsistency detected" in fmt
    assert expected == 3
    assert retrieved == 2
    assert missing == 1
    assert "c2" in str(sample)


async def test_no_vectors_at_all_returns_empty():
    entity_info = [{"sorted_chunks": ["c0", "c1"]}]
    vdb = _VectorStore({})

    with patch.object(utils_logger, "warning") as mock_warning:
        selected = await pick_by_vector_similarity(
            query="q",
            text_chunks_storage=None,
            chunks_vdb=vdb,
            num_of_chunks=5,
            entity_info=entity_info,
            embedding_func=None,
            query_embedding=np.array([1.0, 0.0, 0.0]),
        )

    assert selected == []
    assert any(
        "no vectors retrieved from chunks_vdb" in call.args[0]
        for call in mock_warning.call_args_list
    )


# A permutation and its reverse: no hash- or name-ordered selection can pass both.
_ORDERS = [("chunk-a", "chunk-b", "chunk-c"), ("chunk-c", "chunk-b", "chunk-a")]


@pytest.mark.parametrize("order", _ORDERS)
async def test_similarity_ties_preserve_candidate_order(order):
    vectors = {key: np.array([1.0, 0.0]) for key in order}
    result, store = await _select([{"sorted_chunks": list(order)}], vectors, 2)
    assert result == list(order[:2])
    assert store.requested_ids == list(order)


async def test_duplicate_chunks_keep_their_first_position():
    entities = [
        {"sorted_chunks": ["chunk-c", "chunk-a", "chunk-c"]},
        {"sorted_chunks": ["chunk-b", "chunk-a"]},
    ]
    before = deepcopy(entities)
    vectors = {key: np.array([1.0, 0.0]) for key in ["chunk-a", "chunk-b", "chunk-c"]}
    result, store = await _select(entities, vectors, 10)
    assert result == ["chunk-c", "chunk-a", "chunk-b"]
    assert store.requested_ids == result
    assert entities == before


async def test_similarity_still_takes_precedence_over_candidate_order():
    vectors = {
        "low": np.array([-1.0, 0.0]),
        "high": np.array([1.0, 0.0]),
        "middle": np.array([0.0, 1.0]),
    }
    result, _ = await _select(
        [{"sorted_chunks": ["low", "high", "middle"]}], vectors, 2
    )
    assert result == ["high", "middle"]


async def test_ties_only_affect_equal_scores():
    vectors = {
        "tie-b": np.array([0.0, 1.0]),
        "best": np.array([1.0, 0.0]),
        "tie-a": np.array([0.0, -1.0]),
        "worst": np.array([-1.0, 0.0]),
    }
    result, _ = await _select([{"sorted_chunks": list(vectors)}], vectors, 3)
    assert result == ["best", "tie-b", "tie-a"]


@pytest.mark.parametrize("order", _ORDERS)
async def test_storage_error_fallback_preserves_candidate_order(order):
    class FailingStore:
        """Vector store that raises on every lookup."""

        async def get_vectors_by_ids(self, ids):
            raise RuntimeError("storage unavailable")

    result = await pick_by_vector_similarity(
        "evidence",
        None,
        FailingStore(),
        2,
        [{"sorted_chunks": list(order)}],
        None,
        query_embedding=np.array([1.0, 0.0]),
    )
    assert result == list(order[:2])


@pytest.mark.parametrize(
    "entities, count",
    [([], 2), ([{}], 2), ([{"sorted_chunks": ["a"]}], 0)],
)
async def test_empty_selection_does_not_read_vectors(entities, count):
    result, store = await _select(entities, {}, count)
    assert result == []
    assert store.requested_ids is None


async def test_embedding_computation_path_uses_same_ordering():
    embedding = AsyncMock(return_value=np.array([[1.0, 0.0]]))
    vectors = {key: np.array([1.0, 0.0]) for key in ["chunk-c", "chunk-a", "chunk-b"]}
    store = _VectorStore(vectors)
    result = await pick_by_vector_similarity(
        "evidence",
        None,
        store,
        2,
        [{"sorted_chunks": list(vectors)}],
        embedding,
    )
    embedding.assert_awaited_once_with(["evidence"], context="query")
    assert result == ["chunk-c", "chunk-a"]


# ---------------------------------------------------------------------------
# pick_by_weighted_polling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("max_chunks", "expected"),
    [
        # On buggy code, chunks[:-1] returns all-but-last instead of [].
        pytest.param(-1, [], id="negative"),
        pytest.param(0, [], id="zero"),
        pytest.param(2, ["c1", "c2"], id="positive"),
    ],
)
def test_weighted_polling_max_chunk_bounds(max_chunks, expected):
    chunks = [{"sorted_chunks": ["c1", "c2", "c3"]}]
    assert pick_by_weighted_polling(chunks, max_chunks) == expected
