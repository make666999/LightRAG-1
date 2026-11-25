"""Cross-backend vector storage contracts: provenance verdicts, drop_all_models, and strict maintenance reads.

Covers:
- Shared embedding-space provenance helpers (``declared_model_name``,
  ``vector_space_marker``, ``read_vector_space_marker``, ``assert_vector_space_matches``).
- ``BaseVectorStorage.drop_all_models`` / ``drop_sibling_models`` /
  ``_collection_suffix_for`` workspace-clear contract.
- Every bundled ``VECTOR_STORAGE`` implementation overriding ``get_by_ids_strict``
  and ``get_exact_count`` (except ``NoopVectorDBStorage``).
"""

from __future__ import annotations

import importlib
from unittest.mock import AsyncMock

import pytest

from lightrag.base import BaseVectorStorage
from lightrag.exceptions import VectorSpaceMismatchError
from lightrag.kg import STORAGE_IMPLEMENTATIONS, STORAGES
from lightrag.kg.vector_space import (
    VECTOR_SPACE_DIM_KEY,
    VECTOR_SPACE_MODEL_KEY,
    assert_vector_space_matches,
    declared_model_name,
    read_vector_space_marker,
    vector_space_marker,
)

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Shared embedding-space provenance helpers
# ---------------------------------------------------------------------------


class _Embedder:
    """Minimal embedder stand-in exposing ``model_name`` and ``embedding_dim``."""

    def __init__(self, model_name=None, embedding_dim=1024):
        self.model_name = model_name
        self.embedding_dim = embedding_dim


def _assert_space(stored_model, stored_dim, embedder):
    assert_vector_space_matches(
        backend="FakeVectorStorage",
        container="fake_index",
        embedding_func=embedder,
        stored_model=stored_model,
        stored_dim=stored_dim,
    )


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("bge-m3", "bge-m3"),
        ("  bge-m3  ", "bge-m3"),
        ("   ", None),
        (None, None),
        (123, None),
    ],
)
def test_declared_model_name_normalizes_one_way_for_every_backend(configured, expected):
    assert declared_model_name(_Embedder(model_name=configured)) == expected


def test_marker_records_unfolded_model_and_omits_unknown_model():
    marker = vector_space_marker(
        _Embedder(model_name="text-embedding-3-large", embedding_dim=3072)
    )
    assert marker == {
        VECTOR_SPACE_MODEL_KEY: "text-embedding-3-large",
        VECTOR_SPACE_DIM_KEY: 3072,
    }

    unknown_model_marker = vector_space_marker(
        _Embedder(model_name=None, embedding_dim=768)
    )
    assert unknown_model_marker == {VECTOR_SPACE_DIM_KEY: 768}


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({VECTOR_SPACE_MODEL_KEY: "m", VECTOR_SPACE_DIM_KEY: 8}, ("m", 8)),
        ({VECTOR_SPACE_MODEL_KEY: " m ", VECTOR_SPACE_DIM_KEY: "8"}, ("m", 8)),
        ({VECTOR_SPACE_MODEL_KEY: "", VECTOR_SPACE_DIM_KEY: "eight"}, (None, None)),
        ({VECTOR_SPACE_DIM_KEY: True}, (None, None)),
        ({}, (None, None)),
        (None, (None, None)),
    ],
)
def test_unreadable_marker_fields_read_as_absent(payload, expected):
    assert read_vector_space_marker(payload) == expected


@pytest.mark.parametrize(
    ("stored_model", "stored_dim", "declared_model", "declared_dim"),
    [
        pytest.param("bge-m3", 1024, "bge-m3", 1024, id="matching"),
        pytest.param(None, 1024, "bge-m3", 1024, id="absent-stored-model"),
        pytest.param("bge-m3", None, "bge-m3", 1024, id="absent-stored-dim"),
        pytest.param("bge-m3", 1024, None, 1024, id="absent-declared-model"),
        pytest.param(None, None, "bge-m3", 1024, id="container-records-nothing"),
    ],
)
def test_matching_or_absent_evidence_never_refuses(
    stored_model, stored_dim, declared_model, declared_dim
):
    assert (
        _assert_space(stored_model, stored_dim, _Embedder(declared_model, declared_dim))
        is None
    )


def test_dimension_change_refuses():
    with pytest.raises(VectorSpaceMismatchError) as excinfo:
        _assert_space("bge-m3", 1024, _Embedder("bge-m3", 768))
    assert excinfo.value.stored_dim == 1024
    assert excinfo.value.expected_dim == 768
    assert "1024 -> 768" in str(excinfo.value)


def test_same_dimension_model_change_refuses():
    with pytest.raises(VectorSpaceMismatchError) as excinfo:
        _assert_space("bge-m3", 1024, _Embedder("other-model", 1024))
    message = str(excinfo.value)
    assert "'bge-m3' -> 'other-model'" in message
    assert "lightrag-rebuild-vdb" in message


def test_case_and_punctuation_differences_are_a_mismatch():
    with pytest.raises(VectorSpaceMismatchError):
        _assert_space(
            "text-embedding-3-large", 3072, _Embedder("text_embedding_3_large", 3072)
        )


def test_dimension_mismatch_refuses_even_when_the_model_matches_and_vice_versa():
    with pytest.raises(VectorSpaceMismatchError):
        _assert_space("bge-m3", 1024, _Embedder(None, 768))
    with pytest.raises(VectorSpaceMismatchError):
        _assert_space("bge-m3", None, _Embedder("other", 768))


# ---------------------------------------------------------------------------
# BaseVectorStorage.drop_all_models contract
# ---------------------------------------------------------------------------


class _Stub(BaseVectorStorage):
    """BaseVectorStorage with its abstract surface waived to exercise base methods."""


_Stub.__abstractmethods__ = frozenset()


def _storage(drop_result, sibling_result):
    storage = object.__new__(_Stub)
    storage.drop = AsyncMock(return_value=drop_result)
    storage.drop_sibling_models = AsyncMock(return_value=sibling_result)
    return storage


OK = {"status": "success", "message": "ok"}


async def test_clears_the_active_model_then_its_siblings():
    storage = _storage(OK, OK)
    assert (await storage.drop_all_models())["status"] == "success"
    storage.drop.assert_awaited_once()
    storage.drop_sibling_models.assert_awaited_once()


async def test_siblings_are_untouched_when_the_active_drop_fails():
    failed = {"status": "error", "message": "nope"}
    storage = _storage(failed, OK)
    assert await storage.drop_all_models() == failed
    storage.drop_sibling_models.assert_not_awaited()


async def test_a_sibling_failure_is_reported_not_swallowed():
    storage = _storage(OK, {"status": "error", "message": "stuck"})
    result = await storage.drop_all_models()
    assert result["status"] == "error"
    assert "stuck" in result["message"]


async def test_vouched_names_reach_the_sibling_sweep_and_not_drop():
    storage = _storage(OK, OK)
    await storage.drop_all_models(vouched=["a_chunks_m_8d"])
    storage.drop.assert_awaited_once_with()
    storage.drop_sibling_models.assert_awaited_once_with(vouched=["a_chunks_m_8d"])


async def test_default_base_sibling_and_container_helpers():
    s = object.__new__(_Stub)
    assert (await s.drop_sibling_models())["status"] == "success"
    assert s.model_container_name("m", 8) is None
    assert await s.unvouched_sibling_containers() == []


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("Text-Embedding-3.Large", "text_embedding_3_large_8d"),
        ("  m  ", "m_8d"),
        ("   ", None),
        (None, None),
    ],
)
def test_the_suffix_fold_is_one_definition(model_name, expected):
    assert _Stub._collection_suffix_for(model_name, 8) == expected


# ---------------------------------------------------------------------------
# Maintenance strict reads implemented on all bundled vector storages
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", STORAGE_IMPLEMENTATIONS["VECTOR_STORAGE"]["implementations"]
)
def test_bundled_vector_storage_implements_the_maintenance_reads(name):
    try:
        module = importlib.import_module(STORAGES[name], package="lightrag")
    except ImportError as exc:
        pytest.skip(f"{name}: {exc}")
    cls = getattr(module, name)

    assert cls.get_by_ids_strict is not BaseVectorStorage.get_by_ids_strict
    if name != "NoopVectorDBStorage":
        assert cls.get_exact_count is not BaseVectorStorage.get_exact_count
