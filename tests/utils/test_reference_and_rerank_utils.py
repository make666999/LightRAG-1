"""Tests for ``generate_reference_list_from_chunks`` and ``normalize_rerank_result``.

Covers:
- ``generate_reference_list_from_chunks``: frequency-descending +
  first-appearance-ascending reference numbering, ``unknown_source`` / missing
  ``file_path`` handling, input non-mutation, and chunk field/order retention.
- ``normalize_rerank_result``: valid normalization, non-dict rejection,
  integer/bool/range index validation, and unconvertible/overflow/bool/non-finite
  relevance score validation.
"""

from __future__ import annotations

import pytest

from lightrag.utils import generate_reference_list_from_chunks, normalize_rerank_result

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# generate_reference_list_from_chunks
# ---------------------------------------------------------------------------


def test_empty_chunks_returns_empty_lists():
    assert generate_reference_list_from_chunks([]) == ([], [])


def test_single_chunk_gets_reference_id_one():
    chunks = [{"content": "hello", "file_path": "a.txt"}]

    reference_list, updated = generate_reference_list_from_chunks(chunks)

    assert reference_list == [{"reference_id": "1", "file_path": "a.txt"}]
    assert updated == [{"content": "hello", "file_path": "a.txt", "reference_id": "1"}]


def test_higher_frequency_file_path_gets_lower_reference_id():
    chunks = [
        {"content": "c1", "file_path": "rare.txt"},
        {"content": "c2", "file_path": "common.txt"},
        {"content": "c3", "file_path": "common.txt"},
    ]

    reference_list, updated = generate_reference_list_from_chunks(chunks)

    assert reference_list == [
        {"reference_id": "1", "file_path": "common.txt"},
        {"reference_id": "2", "file_path": "rare.txt"},
    ]
    assert [c["reference_id"] for c in updated] == ["2", "1", "1"]


def test_tied_frequency_breaks_by_first_appearance_order():
    chunks = [
        {"content": "c1", "file_path": "z.txt"},
        {"content": "c2", "file_path": "a.txt"},
    ]

    reference_list, _ = generate_reference_list_from_chunks(chunks)

    assert [entry["file_path"] for entry in reference_list] == ["z.txt", "a.txt"]


def test_repeated_file_path_keeps_first_appearance_index_for_the_tie_break():
    chunks = [
        {"content": "c1", "file_path": "second.txt"},
        {"content": "c2", "file_path": "first.txt"},
        {"content": "c3", "file_path": "first.txt"},
        {"content": "c4", "file_path": "second.txt"},
    ]

    reference_list, _ = generate_reference_list_from_chunks(chunks)

    assert [entry["file_path"] for entry in reference_list] == [
        "second.txt",
        "first.txt",
    ]


def test_unknown_source_sentinel_and_missing_file_paths_are_excluded():
    chunks = [
        {"content": "c1", "file_path": "unknown_source"},
        {"content": "no key"},
        {"content": "empty", "file_path": ""},
        {"content": "real", "file_path": "a.txt"},
    ]

    reference_list, updated = generate_reference_list_from_chunks(chunks)

    assert reference_list == [{"reference_id": "1", "file_path": "a.txt"}]
    assert [c["reference_id"] for c in updated] == ["", "", "", "1"]


def test_does_not_mutate_input_and_preserves_chunk_fields_and_order():
    original_first = {"content": "c1", "file_path": "b.txt", "chunk_id": "x1"}
    original_second = {"content": "c2", "file_path": "a.txt", "chunk_id": "x2"}
    chunks = [original_first, original_second]

    _, updated = generate_reference_list_from_chunks(chunks)

    assert "reference_id" not in original_first
    assert updated[0] is not original_first
    assert [c["chunk_id"] for c in updated] == ["x1", "x2"]
    assert [c["content"] for c in updated] == ["c1", "c2"]
    assert [c["reference_id"] for c in updated] == ["1", "2"]


# ---------------------------------------------------------------------------
# normalize_rerank_result
# ---------------------------------------------------------------------------


class TestAcceptedRerankResults:
    @pytest.mark.parametrize(
        ("result", "max_index", "expected_score"),
        [
            ({"index": 1, "relevance_score": 0.8}, 2, 0.8),
            # String scores are accepted and coerced to float.
            ({"index": 0, "relevance_score": "0.5"}, 3, 0.5),
            # Integral scores are accepted and normalized to float.
            ({"index": 2, "relevance_score": 3}, 4, 3.0),
            # The highest valid index is ``max_index - 1``.
            ({"index": 3, "relevance_score": 0.1}, 4, 0.1),
        ],
    )
    def test_valid_result_is_normalized(self, result, max_index, expected_score):
        normalized, error = normalize_rerank_result(result, max_index)
        assert error is None
        assert normalized == {
            "index": result["index"],
            "relevance_score": expected_score,
        }
        assert isinstance(normalized["relevance_score"], float)


class TestRerankResultShape:
    @pytest.mark.parametrize(
        "result",
        [
            pytest.param(None, id="none"),
            pytest.param((0, 0.9), id="tuple"),
        ],
    )
    def test_non_dict_result_is_rejected(self, result):
        assert normalize_rerank_result(result, 2) == (None, "not an object")


class TestRerankIndexValidation:
    @pytest.mark.parametrize(
        "result",
        [
            pytest.param({"relevance_score": 0.9}, id="missing-index"),
            pytest.param({"index": "0", "relevance_score": 0.9}, id="string-index"),
        ],
    )
    def test_non_integer_index_is_rejected(self, result):
        assert normalize_rerank_result(result, 2) == (None, "invalid index")

    @pytest.mark.parametrize("index", [True, False])
    def test_bool_index_is_rejected_even_though_bool_is_an_int_subclass(self, index):
        result = {"index": index, "relevance_score": 0.9}
        assert normalize_rerank_result(result, 2) == (None, "invalid index")

    @pytest.mark.parametrize(
        ("result", "max_index"),
        [
            pytest.param({"index": -1, "relevance_score": 0.9}, 2, id="negative"),
            pytest.param({"index": 2, "relevance_score": 0.9}, 2, id="at-max"),
        ],
    )
    def test_out_of_range_index_is_rejected(self, result, max_index):
        assert normalize_rerank_result(result, max_index) == (
            None,
            "index out of range",
        )


class TestRerankScoreValidation:
    @pytest.mark.parametrize(
        "result",
        [
            pytest.param({"index": 0}, id="missing-score"),
            pytest.param(
                {"index": 0, "relevance_score": "not-a-score"}, id="string-score"
            ),
            pytest.param(
                {"index": 0, "relevance_score": {"value": 0.5}}, id="dict-score"
            ),
            # float(10**400) raises OverflowError rather than ValueError.
            pytest.param({"index": 0, "relevance_score": 10**400}, id="overflow-score"),
            # float(True) would succeed as 1.0 unless bool is rejected first.
            pytest.param({"index": 0, "relevance_score": True}, id="bool-score"),
        ],
    )
    def test_invalid_score_is_rejected(self, result):
        assert normalize_rerank_result(result, 2) == (
            None,
            "invalid relevance score",
        )

    @pytest.mark.parametrize(
        "score",
        [
            pytest.param(float("nan"), id="nan"),
            pytest.param(float("inf"), id="positive-inf"),
            pytest.param("inf", id="string-inf"),
            pytest.param("nan", id="string-nan"),
        ],
    )
    def test_non_finite_score_is_rejected(self, score):
        result = {"index": 0, "relevance_score": score}
        assert normalize_rerank_result(result, 2) == (
            None,
            "non-finite relevance score",
        )
