"""Tests for LightRAG / QueryParam configuration, addon_params, and positional API.

Covers:
- Observable runtime ``addon_params`` atomicity, cache invalidation, and overlay
  copy isolation.
- Non-positive batch/concurrency settings rejected fast at construction and
  runtime update.
- Dataclass positional field-order compatibility for ``QueryParam`` and
  ``LightRAG``.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any

import numpy as np
import pytest

from lightrag import LightRAG, RoleLLMConfig
from lightrag.addon_params import ObservableAddonParams, normalize_addon_params
from lightrag.base import QueryParam
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import EmbeddingFunc, Tokenizer, priority_limit_async_func_call

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Observable runtime addon_params
# ---------------------------------------------------------------------------


def test_update_commits_valid_input_and_notifies_once() -> None:
    changes: list[str] = []
    params = ObservableAddonParams(
        {"language": "English"}, on_change=lambda: changes.append("changed")
    )

    params.update([("language", "French")], entity_types=["person"])

    assert params == {"language": "French", "entity_types": ["person"]}
    assert changes == ["changed"]


def test_update_is_atomic_when_an_iterable_is_malformed() -> None:
    changes: list[str] = []

    def unexpected_change() -> None:
        changes.append("changed")
        raise AssertionError("failed updates must not trigger callbacks")

    params = ObservableAddonParams({"language": "English"}, on_change=unexpected_change)

    def partly_invalid_items():
        yield ("language", "French")
        yield ("malformed",)

    # Match on the exception type only: the "dictionary update sequence
    # element #1 ..." wording is a CPython implementation detail, and what this
    # regression is about is the mapping's state, not the message.
    with pytest.raises(ValueError):
        params.update(partly_invalid_items())

    # A failed live-config update must not leave the mapping ahead of its
    # derived cache or emit a change notification for an update that failed.
    assert params == {"language": "English"}
    assert changes == []


def test_update_does_not_notify_when_validation_fails_before_a_write() -> None:
    changes: list[str] = []
    params = ObservableAddonParams(
        {"language": "English"}, on_change=lambda: changes.append("changed")
    )

    with pytest.raises(TypeError):
        params.update(42)

    assert params == {"language": "English"}
    assert changes == []


def test_ior_is_atomic_when_an_iterable_is_malformed() -> None:
    """``|=`` takes iterables too (PEP 584), so it needs update()'s guarantee.

    ``dict.__ior__`` is not restricted to the ``Mapping`` its signature
    advertises, so without materializing first it applies leading pairs and then
    raises, skipping the change notification exactly like ``update`` did.
    """
    changes: list[str] = []

    def unexpected_change() -> None:
        changes.append("changed")
        raise AssertionError("failed updates must not trigger callbacks")

    params = ObservableAddonParams({"language": "English"}, on_change=unexpected_change)

    def partly_invalid_items():
        yield ("language", "French")
        yield ("malformed",)

    with pytest.raises(ValueError):
        params |= partly_invalid_items()

    assert params == {"language": "English"}
    assert changes == []


def test_ior_commits_a_mapping_and_notifies_once() -> None:
    changes: list[str] = []
    params = ObservableAddonParams(
        {"language": "English"}, on_change=lambda: changes.append("changed")
    )

    params |= {"language": "French"}

    assert params == {"language": "French"}
    assert changes == ["changed"]


def _addon_params_change_observer() -> Any:
    """A minimal stand-in carrying only LightRAG's addon-params cache hooks.

    Constructing a real ``LightRAG`` needs storages, an LLM func and an
    embedding func; the invalidation contract under test is just these two
    methods plus the two attributes they touch.
    """

    class _CacheOwner:
        _on_addon_params_changed = LightRAG._on_addon_params_changed
        _mark_addon_params_dirty = LightRAG._mark_addon_params_dirty

        def __init__(self) -> None:
            self._addon_params_dirty = False
            self._addon_params = ObservableAddonParams(
                {"chunker": {}}, on_change=self._on_addon_params_changed
            )

    return _CacheOwner()


def test_cache_is_invalidated_even_when_the_change_callback_raises() -> None:
    """A raising callback must not leave the mapping ahead of a clean cache.

    ``separators`` holding non-strings reaches ``len()`` on an ``int`` inside
    ``inspect_r_separators``. The assignment itself has already committed by
    then, so skipping the dirty mark would keep the stale summary language and
    prompt profile alive until some unrelated later mutation happened to
    invalidate them.
    """
    owner = _addon_params_change_observer()
    malformed = {"recursive_character": {"separators": [1, 2]}}

    with pytest.raises(TypeError):
        owner._addon_params["chunker"] = malformed

    assert owner._addon_params["chunker"] == malformed
    assert owner._addon_params_dirty is True


def test_cache_is_invalidated_on_a_normal_chunker_replacement() -> None:
    owner = _addon_params_change_observer()

    owner._addon_params["chunker"] = {"recursive_character": {"separators": ["\n"]}}

    assert owner._addon_params_dirty is True


def test_chunk_size_overlay_does_not_mutate_supplied_addon_params() -> None:
    """Resolving runtime defaults must not write into the caller's config.

    A shared mapping reused for a second instance must not carry the first
    instance's resolved sizes, which would otherwise outrank the second
    instance's own ``chunk_token_size`` as an explicit addon_params value.
    """
    supplied = {"chunker": {"fixed_token": {"split_by_character": "|"}}}

    class _OverlayOwner:
        _apply_chunk_size_overlay = LightRAG._apply_chunk_size_overlay

        def __init__(self, chunk_token_size: int) -> None:
            self._addon_params = normalize_addon_params(supplied)
            self.chunk_token_size = chunk_token_size
            self.chunk_overlap_token_size = None

    first = _OverlayOwner(500)
    first._apply_chunk_size_overlay()
    second = _OverlayOwner(2000)
    second._apply_chunk_size_overlay()

    assert supplied == {"chunker": {"fixed_token": {"split_by_character": "|"}}}
    assert (first.chunk_token_size, second.chunk_token_size) == (500, 2000)
    first._addon_params["chunker"]["fixed_token"]["split_by_character"] = "#"
    assert second._addon_params["chunker"]["fixed_token"]["split_by_character"] == "|"


# ---------------------------------------------------------------------------
# Non-positive batch / concurrency configuration validation
# ---------------------------------------------------------------------------


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


async def _dummy_llm(*args, **kwargs) -> str:
    return "ok"


async def _dummy_rerank(query: str, documents: list[str], **kwargs):
    return list(range(len(documents)))


@pytest.fixture
def make_rag(tmp_path):
    initialize_share_data()

    def _build(**kwargs) -> LightRAG:
        return LightRAG(
            working_dir=str(tmp_path / "wd"),
            workspace="nonpositive-config",
            llm_model_func=_dummy_llm,
            embedding_func=EmbeddingFunc(
                embedding_dim=8, max_token_size=8192, func=_dummy_embedding
            ),
            tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
            **kwargs,
        )

    try:
        yield _build
    finally:
        finalize_share_data()


@pytest.mark.parametrize("bad", [0, -1])
def test_embedding_batch_num_nonpositive_is_rejected(make_rag, bad):
    with pytest.raises(ValueError, match="EMBEDDING_BATCH_NUM"):
        make_rag(embedding_batch_num=bad)


@pytest.mark.parametrize("bad", [0, -2])
def test_embedding_func_max_async_nonpositive_is_rejected(make_rag, bad):
    with pytest.raises(ValueError, match="EMBEDDING_FUNC_MAX_ASYNC"):
        make_rag(embedding_func_max_async=bad)


def test_llm_model_max_async_nonpositive_is_rejected(make_rag):
    with pytest.raises(ValueError, match="MAX_ASYNC_LLM"):
        make_rag(llm_model_max_async=0)


def test_rerank_model_max_async_nonpositive_is_rejected(make_rag):
    with pytest.raises(ValueError, match="MAX_ASYNC_RERANK"):
        make_rag(
            rerank_model_func=_dummy_rerank,
            rerank_model_max_async=0,
        )


def test_role_max_async_override_nonpositive_is_rejected(make_rag):
    with pytest.raises(ValueError, match=r"role_llm_configs\['query'\]\.max_async"):
        make_rag(role_llm_configs={"query": {"max_async": 0}})


def test_role_max_async_config_object_nonpositive_is_rejected(make_rag):
    with pytest.raises(ValueError, match=r"role_llm_configs\['vlm'\]\.max_async"):
        make_rag(role_llm_configs={"vlm": RoleLLMConfig(max_async=-1)})


def test_role_max_async_env_nonpositive_is_rejected(make_rag, monkeypatch):
    monkeypatch.setenv("QUERY_MAX_ASYNC_LLM", "0")
    with pytest.raises(ValueError, match="QUERY_MAX_ASYNC_LLM must be >= 1"):
        make_rag()


def test_role_max_async_override_wins_over_nonpositive_env(make_rag, monkeypatch):
    """An explicit override shadows the env seed, so a bad env var it hides
    must not fail construction."""
    monkeypatch.setenv("QUERY_MAX_ASYNC_LLM", "0")
    rag = make_rag(role_llm_configs={"query": {"max_async": 2}})
    assert rag._get_effective_role_llm_max_async("query") == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"embedding_batch_num": 0},
        {"llm_model_max_async": 0},
        {"role_llm_configs": {"query": {"max_async": 0}}},
    ],
)
def test_rejection_happens_before_working_dir_is_created(tmp_path, make_rag, kwargs):
    with pytest.raises(ValueError):
        make_rag(**kwargs)
    assert not (tmp_path / "wd").exists()


@pytest.mark.parametrize("bad", [0, -2])
def test_limiter_rejects_nonpositive_max_size_at_decoration(bad):
    """The decorator is the last line of defence for callers that never
    build a LightRAG: it must refuse rather than queue calls forever."""
    with pytest.raises(ValueError, match="My queue: max_size must be >= 1"):
        priority_limit_async_func_call(bad, queue_name="My queue")


def test_role_max_async_runtime_update_nonpositive_is_rejected(make_rag):
    rag = make_rag()
    with pytest.raises(ValueError, match="max_async for LLM role 'query'"):
        rag.update_llm_role_config("query", max_async=0)
    # The failed update rolls back: the previous limit stays in effect.
    assert rag._get_effective_role_llm_max_async("query") == 4


def test_positive_values_are_accepted(make_rag):
    rag = make_rag(
        embedding_batch_num=1,
        embedding_func_max_async=1,
        llm_model_max_async=1,
        rerank_model_func=_dummy_rerank,
        rerank_model_max_async=1,
        role_llm_configs={"query": {"max_async": 1}},
    )
    assert rag.embedding_batch_num == 1
    assert rag.embedding_func_max_async == 1
    assert rag.llm_model_max_async == 1
    assert rag.rerank_model_max_async == 1
    assert rag._get_effective_role_llm_max_async("query") == 1


# ---------------------------------------------------------------------------
# Dataclass positional field-order compatibility (QueryParam & LightRAG)
# ---------------------------------------------------------------------------

# The positional prefix of the signature as it stood before
# disable_user_prompt_prefix existed. Do NOT reorder to match a future refactor
# -- if this list stops matching, the constructor signature changed and every
# positional caller downstream changed behaviour with it.
_LEGACY_POSITIONAL_ORDER = [
    "mode",
    "only_need_context",
    "only_need_prompt",
    "response_type",
    "stream",
    "top_k",
    "chunk_top_k",
    "max_entity_tokens",
    "max_relation_tokens",
    "max_total_tokens",
    "hl_keywords",
    "ll_keywords",
    "conversation_history",
    "user_prompt",
    "enable_rerank",
    "include_references",
]


def test_legacy_positional_field_order_is_unchanged():
    names = [f.name for f in dataclasses.fields(QueryParam)]
    assert names[: len(_LEGACY_POSITIONAL_ORDER)] == _LEGACY_POSITIONAL_ORDER


def test_new_fields_are_appended_after_the_legacy_ones():
    names = [f.name for f in dataclasses.fields(QueryParam)]
    assert "disable_user_prompt_prefix" in names
    assert names.index("disable_user_prompt_prefix") >= len(_LEGACY_POSITIONAL_ORDER)


def test_positional_construction_still_binds_enable_rerank():
    """The concrete break: the 15th positional argument.

    Before the field was moved to the end, this ``False`` bound to
    ``disable_user_prompt_prefix`` and ``enable_rerank`` silently kept its
    ``RERANK_BY_DEFAULT`` value.
    """
    param = QueryParam(
        "mix",
        False,
        False,
        "Multiple Paragraphs",
        False,
        40,
        20,
        6000,
        8000,
        30000,
        [],
        [],
        [],
        "be concise",
        False,
    )
    assert param.user_prompt == "be concise"
    assert param.enable_rerank is False
    assert param.disable_user_prompt_prefix is False


def test_positional_construction_still_binds_include_references():
    """The 16th positional argument, shifted by the same insertion."""
    param = QueryParam(
        "mix",
        False,
        False,
        "Multiple Paragraphs",
        False,
        40,
        20,
        6000,
        8000,
        30000,
        [],
        [],
        [],
        None,
        True,
        True,
    )
    assert param.enable_rerank is True
    assert param.include_references is True
    assert param.disable_user_prompt_prefix is False


# The constructor's positional order before `user_prompt_prefix` was added.
# Appending a new field at the end means appending one line here too; that tax
# is deliberate, because any change to this list is a change to the public
# constructor API and should be looked at by a human.
_LEGACY_LIGHTRAG_INIT_ORDER = [
    "working_dir",
    "kv_storage",
    "vector_storage",
    "graph_storage",
    "doc_status_storage",
    "workspace",
    "log_level",
    "log_file_path",
    "top_k",
    "chunk_top_k",
    "max_entity_tokens",
    "max_relation_tokens",
    "max_total_tokens",
    "related_chunk_number",
    "kg_chunk_pick_method",
    "enable_content_headings",
    "entity_extract_max_gleaning",
    "entity_extract_max_records",
    "entity_extract_max_entities",
    "force_llm_summary_on_merge",
    "chunk_token_size",
    "chunk_overlap_token_size",
    "tokenizer",
    "tiktoken_model_name",
    "chunking_func",
    "embedding_func",
    "embedding_chunk_overlap_token_size",
    "embedding_batch_num",
    "embedding_func_max_async",
    "embedding_cache_config",
    "default_embedding_timeout",
    "llm_model_func",
    "role_llm_configs",
    "llm_model_name",
    "summary_max_tokens",
    "summary_context_size",
    "summary_length_recommended",
    "llm_model_max_async",
    "llm_model_kwargs",
    "default_llm_timeout",
    "entity_extraction_use_json",
    "rerank_model_func",
    "rerank_model_max_async",
    "default_rerank_timeout",
    "min_rerank_score",
    "vector_db_storage_cls_kwargs",
    "enable_llm_cache",
    "enable_llm_cache_for_entity_extract",
    "vlm_process_enable",
    "max_parallel_insert",
    "max_parallel_parse_native",
    "max_parallel_parse_mineru",
    "max_parallel_parse_docling",
    "max_parallel_analyze",
    "queue_size_parse",
    "queue_size_analyze",
    "queue_size_insert",
    "pipeline_scheduling_page_size",
    "pipeline_require_strict_storage_reads",
    "max_pending_documents",
    "max_graph_nodes",
    "max_source_ids_per_entity",
    "max_source_ids_per_relation",
    "source_ids_limit_method",
    "max_file_paths",
    "file_path_more_placeholder",
    "addon_params",
    "auto_manage_storages_states",
    "cosine_better_than_threshold",
    "ollama_server_infos",
    "_storages_status",
]


def _lightrag_init_params() -> list[str]:
    """Introspect the signature rather than constructing LightRAG.

    ``__post_init__`` does real storage setup, which would make a positional
    construction test slow and brittle. The signature is exactly -- and only --
    the contract at stake here.
    """
    return list(inspect.signature(LightRAG.__init__).parameters)[1:]  # drop self


def test_legacy_lightrag_init_order_is_unchanged():
    params = _lightrag_init_params()
    assert params[: len(_LEGACY_LIGHTRAG_INIT_ORDER)] == _LEGACY_LIGHTRAG_INIT_ORDER


def test_user_prompt_prefix_is_appended_after_every_legacy_field():
    """The rule itself: new fields come after the last pre-existing one."""
    params = _lightrag_init_params()
    assert params.index("user_prompt_prefix") > params.index("_storages_status")


def test_keyword_extraction_json_mode_is_appended_after_every_legacy_field():
    params = _lightrag_init_params()
    assert params.index("keyword_extraction_json_mode") > params.index(
        "_storages_status"
    )


def test_lightrag_positional_anchors_did_not_shift():
    """Spot-checks either end of the range the insertion used to move."""
    params = _lightrag_init_params()
    # First field displaced by the original insertion.
    assert params[16] == "entity_extract_max_gleaning"
    # Last pre-existing field; nothing may be inserted before it.
    assert params[len(_LEGACY_LIGHTRAG_INIT_ORDER) - 1] == "_storages_status"
