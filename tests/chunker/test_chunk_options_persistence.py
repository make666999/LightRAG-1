"""Tests for the ``chunk_options`` snapshot mechanism.

Three properties under test:

1. **env-driven snapshot**: env vars (CHUNK_R_OVERLAP_SIZE,
   CHUNK_V_BREAKPOINT_THRESHOLD_TYPE, …) flow into
   ``addon_params['chunker']`` via
   :func:`lightrag.parser.routing.default_chunker_config`, then into
   ``full_docs[doc_id]['chunk_options']`` at enqueue time via
   :func:`lightrag.parser.routing.resolve_chunk_options`.

2. **caller-supplied chunk_options**: an explicit ``chunk_options``
   kwarg passed to ``apipeline_enqueue_documents`` is persisted
   verbatim and reaches the dispatched chunker as keyword args.

3. **per-file chunk_options as a list**: when chunk_options is a
   ``list[dict]`` aligned with ``input``, each doc gets its own
   independent persisted snapshot.
"""

import asyncio
from pathlib import Path

import numpy as np
import pytest

from lightrag import LightRAG, ROLES, RoleLLMConfig
from lightrag.constants import DEFAULT_CHUNK_P_SIZE, DEFAULT_R_SEPARATORS
from lightrag.utils import EmbeddingFunc, Tokenizer, TokenizerInterface

pytestmark = pytest.mark.offline


class _SimpleTokenizerImpl(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


async def _mock_embedding(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 32)


async def _mock_llm(prompt, **kwargs):
    return '{"name":"x","summary":"s","detail_description":"d"}'


_ROLE_FIELD_SUFFIXES = (
    ("_llm_model_func", "func"),
    ("_llm_model_kwargs", "kwargs"),
    ("_llm_model_max_async", "max_async"),
    ("_llm_timeout", "timeout"),
)


def _new_rag(tmp_path: Path, **kwargs) -> LightRAG:
    role_configs: dict[str, RoleLLMConfig] = {}
    for spec in ROLES:
        bucket = {}
        for suffix, target in _ROLE_FIELD_SUFFIXES:
            key = f"{spec.name}{suffix}"
            if key in kwargs:
                bucket[target] = kwargs.pop(key)
        if bucket:
            role_configs[spec.name] = RoleLLMConfig(**bucket)
    if role_configs:
        kwargs["role_llm_configs"] = role_configs

    return LightRAG(
        working_dir=str(tmp_path),
        workspace=f"chunkopts-{tmp_path.name}",
        llm_model_func=_mock_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=32,
            max_token_size=4096,
            func=_mock_embedding,
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        **kwargs,
    )


@pytest.mark.offline
def test_env_driven_snapshot_persisted_in_full_docs(tmp_path, monkeypatch):
    """Env vars + ainsert split args land in ``full_docs.chunk_options``.

    The persisted snapshot is slim — only the strategy slot selected by
    ``process_options`` survives — so each strategy is verified through
    its own enqueue with the matching selector.
    """
    monkeypatch.setenv("CHUNK_R_OVERLAP_SIZE", "42")
    monkeypatch.setenv("CHUNK_V_BREAKPOINT_THRESHOLD_TYPE", "interquartile")
    monkeypatch.setenv("CHUNK_V_BUFFER_SIZE", "3")

    async def _run():
        from lightrag.parser.routing import resolve_chunk_options

        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            # F slot — mirror what ``LightRAG.ainsert`` does: build the
            # F-scoped chunk_options snapshot from addon_params plus
            # F-strategy runtime args, then hand it to enqueue.
            chunk_opts_f = resolve_chunk_options(
                rag.addon_params,
                process_options="F",
                split_by_character="\n\n",
                split_by_character_only=True,
            )
            await rag.apipeline_enqueue_documents(
                "Body for F-strategy snapshot test.",
                ids=["doc-snap-f"],
                file_paths="snap-f.txt",
                track_id="track-snap-f",
                chunk_options=chunk_opts_f,
            )
            row_f = await rag.full_docs.get_by_id("doc-snap-f")

            # R slot — env-driven CHUNK_R_OVERLAP_SIZE flows through
            # addon_params['chunker'] into the persisted snapshot.
            await rag.apipeline_enqueue_documents(
                "Body for R-strategy snapshot test.",
                ids=["doc-snap-r"],
                file_paths="snap-r.[native-R].txt",
                track_id="track-snap-r",
                process_options="R",
            )
            row_r = await rag.full_docs.get_by_id("doc-snap-r")

            # V slot — env-driven CHUNK_V_* params likewise.
            await rag.apipeline_enqueue_documents(
                "Body for V-strategy snapshot test.",
                ids=["doc-snap-v"],
                file_paths="snap-v.[native-V].txt",
                track_id="track-snap-v",
                process_options="V",
            )
            row_v = await rag.full_docs.get_by_id("doc-snap-v")
        finally:
            await rag.finalize_storages()
        return row_f, row_r, row_v

    row_f, row_r, row_v = asyncio.run(_run())
    assert row_f is not None and row_r is not None and row_v is not None

    f_opts = row_f["chunk_options"]
    assert f_opts["fixed_token"]["split_by_character"] == "\n\n"
    assert f_opts["fixed_token"]["split_by_character_only"] is True
    # Slim contract: only the active strategy survives.
    assert "recursive_character" not in f_opts
    assert "semantic_vector" not in f_opts
    assert "paragraph_semantic" not in f_opts

    r_opts = row_r["chunk_options"]
    assert r_opts["recursive_character"]["chunk_overlap_token_size"] == 42
    assert "fixed_token" not in r_opts

    v_opts = row_v["chunk_options"]
    assert v_opts["semantic_vector"]["breakpoint_threshold_type"] == "interquartile"
    assert v_opts["semantic_vector"]["buffer_size"] == 3
    assert "fixed_token" not in v_opts


@pytest.mark.offline
def test_caller_supplied_chunk_options_reach_chunker(tmp_path, monkeypatch):
    """A caller-supplied ``chunk_options`` dict is persisted verbatim
    and the dispatcher splats it into the chunker call."""
    pytest.importorskip("langchain_text_splitters")

    import lightrag.chunker as chunker_pkg

    custom_options = {
        "chunk_token_size": 100,
        "fixed_token": {
            "chunk_overlap_token_size": 5,
            "split_by_character": None,
            "split_by_character_only": False,
        },
        "recursive_character": {
            "chunk_overlap_token_size": 0,
            "separators": ["|", ""],
        },
        "semantic_vector": {
            "breakpoint_threshold_type": "percentile",
            "breakpoint_threshold_amount": None,
            "buffer_size": 1,
        },
        "paragraph_semantic": {},
    }

    captured: dict = {}

    def _r_spy(tokenizer, content, chunk_token_size, **kwargs):
        captured["chunk_token_size"] = chunk_token_size
        captured["kwargs"] = dict(kwargs)
        return [
            {"tokens": 5, "content": "stub", "chunk_order_index": 0},
        ]

    monkeypatch.setattr(chunker_pkg, "chunking_by_recursive_character", _r_spy)

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                "alpha|beta|gamma|delta",
                file_paths="caller.[native-R].txt",
                track_id="track-caller",
                process_options="R",
                chunk_options=custom_options,
            )
            await rag.apipeline_process_enqueue_documents()
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())

    assert captured.get("chunk_token_size") == 100, (
        f"R chunker must receive caller-supplied chunk_token_size; got {captured!r}"
    )
    assert captured["kwargs"]["separators"] == ["|", ""]
    assert captured["kwargs"]["chunk_overlap_token_size"] == 0


@pytest.mark.offline
def test_per_file_chunk_options_list(tmp_path, monkeypatch):
    """A ``chunk_options`` list aligned with ``input`` writes
    independent snapshots per doc.

    The two docs use ``process_options="R"`` so the slim snapshot
    keeps their distinct R-strategy params; F/V/P sub-dicts in the
    caller-supplied input are dropped by design.
    """

    opts_a = {
        "chunk_token_size": 1200,
        "fixed_token": {
            "chunk_overlap_token_size": 100,
            "split_by_character": None,
            "split_by_character_only": False,
        },
        "recursive_character": {
            "chunk_overlap_token_size": 100,
            "separators": ["A_SEP"],
        },
        "semantic_vector": {
            "breakpoint_threshold_type": "percentile",
            "breakpoint_threshold_amount": None,
            "buffer_size": 1,
        },
        "paragraph_semantic": {},
    }
    opts_b = {
        "chunk_token_size": 1200,
        "fixed_token": {
            "chunk_overlap_token_size": 100,
            "split_by_character": None,
            "split_by_character_only": False,
        },
        "recursive_character": {
            "chunk_overlap_token_size": 100,
            "separators": ["B_SEP"],
        },
        "semantic_vector": {
            "breakpoint_threshold_type": "percentile",
            "breakpoint_threshold_amount": None,
            "buffer_size": 1,
        },
        "paragraph_semantic": {},
    }

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                ["doc one body", "doc two body"],
                ids=["doc-aaaaa-list", "doc-bbbbb-list"],
                file_paths=["a.[native-R].txt", "b.[native-R].txt"],
                track_id="track-list",
                process_options=["R", "R"],
                chunk_options=[opts_a, opts_b],
            )
            row_a = await rag.full_docs.get_by_id("doc-aaaaa-list")
            row_b = await rag.full_docs.get_by_id("doc-bbbbb-list")
        finally:
            await rag.finalize_storages()
        return row_a, row_b

    row_a, row_b = asyncio.run(_run())
    assert row_a is not None and row_b is not None

    sep_a = row_a["chunk_options"]["recursive_character"]["separators"]
    sep_b = row_b["chunk_options"]["recursive_character"]["separators"]
    assert sep_a == ["A_SEP"]
    assert sep_b == ["B_SEP"]

    # Independence: mutating one snapshot must not bleed into the other.
    sep_a.append("MUT")
    assert "MUT" not in row_b["chunk_options"]["recursive_character"]["separators"]

    # Slim contract: non-R strategy slots are dropped from the persisted
    # snapshot since they would never be consumed at process time.
    assert "fixed_token" not in row_a["chunk_options"]
    assert "semantic_vector" not in row_a["chunk_options"]
    assert "paragraph_semantic" not in row_a["chunk_options"]


@pytest.mark.offline
def test_constructor_chunk_size_overlays_addon_params(tmp_path, monkeypatch):
    """``LightRAG(chunk_token_size=N, chunk_overlap_token_size=M)`` must
    actually take effect — the per-doc snapshot is built from
    ``addon_params['chunker']``, so the constructor values have to be
    overlaid onto it (otherwise env-driven defaults would silently win).
    """
    # Set env vars to non-default values so the env path would be
    # observably different from the constructor path.
    monkeypatch.setenv("CHUNK_SIZE", "1200")
    monkeypatch.setenv("CHUNK_OVERLAP_SIZE", "100")

    async def _run():
        rag = _new_rag(
            tmp_path,
            chunk_token_size=77,
            chunk_overlap_token_size=2,
        )
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                "Body for constructor overlay test.",
                ids=["doc-ctor-overlay"],
                file_paths="ctor.txt",
                track_id="track-ctor",
            )
            row = await rag.full_docs.get_by_id("doc-ctor-overlay")
        finally:
            await rag.finalize_storages()
        return row, rag.addon_params

    row, addon_params = asyncio.run(_run())
    assert row is not None
    chunk_opts = row["chunk_options"]
    # Top-level chunk_token_size carries the constructor value.
    assert chunk_opts["chunk_token_size"] == 77
    # Default-F doc: the persisted slim snapshot only carries F's slot.
    assert chunk_opts["fixed_token"]["chunk_overlap_token_size"] == 2
    assert "recursive_character" not in chunk_opts
    assert "semantic_vector" not in chunk_opts
    assert "paragraph_semantic" not in chunk_opts
    # addon_params still reflects the constructor overlay across every
    # strategy so subsequent enqueues with other selectors pick up the
    # same baseline.  V doesn't have chunk_overlap_token_size and must
    # remain unchanged.
    assert addon_params["chunker"]["chunk_token_size"] == 77
    assert addon_params["chunker"]["fixed_token"]["chunk_overlap_token_size"] == 2
    assert (
        addon_params["chunker"]["recursive_character"]["chunk_overlap_token_size"] == 2
    )
    assert (
        addon_params["chunker"]["paragraph_semantic"]["chunk_overlap_token_size"] == 2
    )
    assert "chunk_overlap_token_size" not in addon_params["chunker"]["semantic_vector"]


@pytest.mark.offline
def test_addon_params_chunker_wins_when_constructor_field_unset(tmp_path):
    """If the constructor field is left at its default (``None``), an
    explicit ``addon_params={'chunker': {...}}`` must NOT be clobbered.
    """

    async def _run():
        rag = _new_rag(
            tmp_path,
            addon_params={
                "chunker": {
                    "chunk_token_size": 5000,
                    "fixed_token": {
                        "chunk_overlap_token_size": 250,
                        "split_by_character": None,
                        "split_by_character_only": False,
                    },
                    "recursive_character": {
                        "chunk_overlap_token_size": 250,
                        "separators": ["\n\n", "\n", " ", ""],
                    },
                    "semantic_vector": {
                        "breakpoint_threshold_type": "percentile",
                        "breakpoint_threshold_amount": None,
                        "buffer_size": 1,
                    },
                    "paragraph_semantic": {},
                },
            },
        )
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                "Body for addon-only overlay test.",
                ids=["doc-addon-only"],
                file_paths="addon.txt",
                track_id="track-addon",
            )
            row = await rag.full_docs.get_by_id("doc-addon-only")
        finally:
            await rag.finalize_storages()
        return row, rag.chunk_token_size, rag.chunk_overlap_token_size

    row, ctor_size, ctor_overlap = asyncio.run(_run())
    assert row is not None
    assert row["chunk_options"]["chunk_token_size"] == 5000
    assert row["chunk_options"]["fixed_token"]["chunk_overlap_token_size"] == 250
    # Legacy instance fields back-fill from addon_params (not env defaults).
    assert ctor_size == 5000
    assert ctor_overlap == 250


@pytest.mark.offline
def test_strategy_env_wins_over_legacy_ctor_field(tmp_path, monkeypatch):
    """Specificity-ordered precedence: strategy-specific env vars beat
    the strategy-agnostic legacy constructor field.

    Setup: ``CHUNK_R_OVERLAP_SIZE=42`` is strategy-specific for R.
    ``LightRAG(chunk_overlap_token_size=2)`` is the legacy
    strategy-agnostic field.  R must end up at 42 (env wins on its own
    strategy slot), F at 2 (no F-specific env, so legacy field fills).
    """
    monkeypatch.setenv("CHUNK_R_OVERLAP_SIZE", "42")
    monkeypatch.delenv("CHUNK_F_OVERLAP_SIZE", raising=False)
    monkeypatch.delenv("CHUNK_OVERLAP_SIZE", raising=False)

    async def _run():
        rag = _new_rag(tmp_path, chunk_overlap_token_size=2)
        await rag.initialize_storages()
        try:
            # R-strategy doc — strategy-specific env wins.
            await rag.apipeline_enqueue_documents(
                "Body for R precedence test.",
                ids=["doc-prec-r"],
                file_paths="prec-r.[native-R].txt",
                track_id="track-prec-r",
                process_options="R",
            )
            row_r = await rag.full_docs.get_by_id("doc-prec-r")
            # F-strategy doc — no F-specific env, ctor field fills.
            await rag.apipeline_enqueue_documents(
                "Body for F precedence test.",
                ids=["doc-prec-f"],
                file_paths="prec-f.txt",
                track_id="track-prec-f",
            )
            row_f = await rag.full_docs.get_by_id("doc-prec-f")
            # P-strategy doc — no P-specific env, ctor field fills.
            await rag.apipeline_enqueue_documents(
                "Body for P precedence test.",
                ids=["doc-prec-p"],
                file_paths="prec-p.[native-P].txt",
                track_id="track-prec-p",
                process_options="P",
            )
            row_p = await rag.full_docs.get_by_id("doc-prec-p")
        finally:
            await rag.finalize_storages()
        return row_r, row_f, row_p, rag.chunk_overlap_token_size

    row_r, row_f, row_p, ctor_field = asyncio.run(_run())
    assert (
        row_r["chunk_options"]["recursive_character"]["chunk_overlap_token_size"] == 42
    ), (
        "Strategy-specific CHUNK_R_OVERLAP_SIZE must win over the "
        "legacy chunk_overlap_token_size constructor field."
    )
    assert row_f["chunk_options"]["fixed_token"]["chunk_overlap_token_size"] == 2, (
        "Without a CHUNK_F_OVERLAP_SIZE override, the F slot falls back "
        "to the legacy constructor field."
    )
    assert row_p["chunk_options"]["paragraph_semantic"]["chunk_overlap_token_size"] == 2
    # self.chunk_overlap_token_size mirrors the F-strategy resolved value.
    assert ctor_field == 2


@pytest.mark.offline
def test_legacy_env_is_final_fallback(tmp_path, monkeypatch):
    """When neither a strategy env nor the legacy ctor field is set,
    the legacy ``CHUNK_OVERLAP_SIZE`` env is the final fallback for
    every per-strategy overlap slot."""
    monkeypatch.delenv("CHUNK_F_OVERLAP_SIZE", raising=False)
    monkeypatch.delenv("CHUNK_R_OVERLAP_SIZE", raising=False)
    monkeypatch.setenv("CHUNK_OVERLAP_SIZE", "77")

    async def _run():
        rag = _new_rag(tmp_path)  # no chunk_overlap_token_size kwarg
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                ["F body", "R body", "P body"],
                ids=["doc-legacy-f", "doc-legacy-r", "doc-legacy-p"],
                file_paths=[
                    "legacy-f.txt",
                    "legacy-r.[native-R].txt",
                    "legacy-p.[native-P].txt",
                ],
                track_id="track-legacy",
                process_options=["", "R", "P"],
            )
            row_f = await rag.full_docs.get_by_id("doc-legacy-f")
            row_r = await rag.full_docs.get_by_id("doc-legacy-r")
            row_p = await rag.full_docs.get_by_id("doc-legacy-p")
        finally:
            await rag.finalize_storages()
        return row_f, row_r, row_p, rag.chunk_overlap_token_size

    row_f, row_r, row_p, ctor_field = asyncio.run(_run())
    assert row_f["chunk_options"]["fixed_token"]["chunk_overlap_token_size"] == 77
    assert (
        row_r["chunk_options"]["recursive_character"]["chunk_overlap_token_size"] == 77
    )
    assert (
        row_p["chunk_options"]["paragraph_semantic"]["chunk_overlap_token_size"] == 77
    )
    assert ctor_field == 77

    # Mixed case: F-specific env set, legacy still acts as R's fallback.
    monkeypatch.setenv("CHUNK_F_OVERLAP_SIZE", "10")

    async def _run_mixed():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                ["F mixed body", "R mixed body", "P mixed body"],
                ids=["doc-mixed-f", "doc-mixed-r", "doc-mixed-p"],
                file_paths=[
                    "mixed-f.txt",
                    "mixed-r.[native-R].txt",
                    "mixed-p.[native-P].txt",
                ],
                track_id="track-mixed",
                process_options=["", "R", "P"],
            )
            row_f = await rag.full_docs.get_by_id("doc-mixed-f")
            row_r = await rag.full_docs.get_by_id("doc-mixed-r")
            row_p = await rag.full_docs.get_by_id("doc-mixed-p")
        finally:
            await rag.finalize_storages()
        return row_f, row_r, row_p

    row_f, row_r, row_p = asyncio.run(_run_mixed())
    assert row_f["chunk_options"]["fixed_token"]["chunk_overlap_token_size"] == 10
    assert (
        row_r["chunk_options"]["recursive_character"]["chunk_overlap_token_size"] == 77
    )
    assert (
        row_p["chunk_options"]["paragraph_semantic"]["chunk_overlap_token_size"] == 77
    )


@pytest.mark.parametrize(
    "env_vars, ctor_size, addon_params, runtime_chunker, caller_chunk_options, expected_size",
    [
        pytest.param(
            {"CHUNK_SIZE": "1200", "CHUNK_P_SIZE": "999"},
            None,
            None,
            None,
            None,
            999,
            id="dedicated_env",
        ),
        pytest.param(
            {"CHUNK_P_SIZE": None, "CHUNK_SIZE": None},
            333,
            None,
            None,
            None,
            DEFAULT_CHUNK_P_SIZE,
            id="unset_env_uses_default",
        ),
        pytest.param(
            {"CHUNK_P_SIZE": None, "CHUNK_SIZE": None},
            333,
            {"chunker": {"paragraph_semantic": {}}},
            None,
            None,
            DEFAULT_CHUNK_P_SIZE,
            id="partial_addon_uses_default",
        ),
        pytest.param(
            {"CHUNK_P_SIZE": "4096", "CHUNK_SIZE": None},
            333,
            {"chunker": {"paragraph_semantic": {}}},
            None,
            None,
            4096,
            id="partial_addon_picks_up_env",
        ),
        pytest.param(
            {"CHUNK_P_SIZE": "4096", "CHUNK_SIZE": None},
            333,
            None,
            {"paragraph_semantic": {}},
            None,
            4096,
            id="runtime_mutation_picks_up_env",
        ),
        pytest.param(
            {"CHUNK_P_SIZE": "4096", "CHUNK_SIZE": None},
            333,
            None,
            None,
            {"chunk_token_size": 333, "paragraph_semantic": {}},
            4096,
            id="caller_chunk_options_picks_up_env",
        ),
        pytest.param(
            {"CHUNK_P_SIZE": "4096", "CHUNK_SIZE": None},
            None,
            None,
            None,
            {"paragraph_semantic": {"chunk_token_size": 8192}},
            8192,
            id="caller_chunk_options_explicit_wins",
        ),
        pytest.param(
            {"CHUNK_P_SIZE": None, "CHUNK_SIZE": None},
            None,
            {"chunker": {"paragraph_semantic": {"chunk_token_size": 4096}}},
            None,
            None,
            4096,
            id="addon_params_explicit_wins",
        ),
    ],
)
def test_p_strategy_chunk_size_precedence(
    tmp_path,
    monkeypatch,
    env_vars,
    ctor_size,
    addon_params,
    runtime_chunker,
    caller_chunk_options,
    expected_size,
):
    """P-strategy ``chunk_token_size`` precedence: explicit caller/addon_params
    > ``CHUNK_P_SIZE`` env > ``DEFAULT_CHUNK_P_SIZE`` (never silently inherits
    global ``CHUNK_SIZE`` or ``LightRAG(chunk_token_size=...)``)."""
    import lightrag.chunker as chunker_pkg

    for key, value in env_vars.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)

    captured: dict = {}

    def _p_spy(tokenizer, content, chunk_token_size, *, blocks_path=None, **kwargs):
        captured["chunk_token_size"] = chunk_token_size
        captured["blocks_path"] = blocks_path
        captured["kwargs"] = dict(kwargs)
        return [{"tokens": 5, "content": "stub", "chunk_order_index": 0}]

    monkeypatch.setattr(chunker_pkg, "chunking_by_paragraph_semantic", _p_spy)

    rag_kwargs: dict = {}
    if ctor_size is not None:
        rag_kwargs["chunk_token_size"] = ctor_size
    if addon_params is not None:
        rag_kwargs["addon_params"] = addon_params

    enqueue_kwargs: dict = {}
    if caller_chunk_options is not None:
        enqueue_kwargs["chunk_options"] = caller_chunk_options

    async def _run():
        rag = _new_rag(tmp_path, **rag_kwargs)
        await rag.initialize_storages()
        try:
            if runtime_chunker is not None:
                rag.addon_params["chunker"] = runtime_chunker
            await rag.apipeline_enqueue_documents(
                "stand-in body for paragraph-semantic chunker",
                file_paths="ctor.[native-P].txt",
                track_id="track-p-size",
                process_options="P",
                **enqueue_kwargs,
            )
            await rag.apipeline_process_enqueue_documents()
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
    assert captured.get("chunk_token_size") == expected_size, (
        f"Expected P chunk_token_size={expected_size}; got {captured!r}"
    )
    # Dispatcher must not double-pass chunk_token_size as kwarg.
    assert "chunk_token_size" not in captured["kwargs"]


@pytest.mark.offline
def test_p_strategy_uses_dedicated_overlap_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CHUNK_OVERLAP_SIZE", "11")
    monkeypatch.setenv("CHUNK_P_OVERLAP_SIZE", "66")

    import lightrag.chunker as chunker_pkg

    captured: dict = {}

    def _p_spy(tokenizer, content, chunk_token_size, *, blocks_path=None, **kwargs):
        captured["kwargs"] = dict(kwargs)
        return [{"tokens": 5, "content": "stub", "chunk_order_index": 0}]

    monkeypatch.setattr(chunker_pkg, "chunking_by_paragraph_semantic", _p_spy)

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                "P overlap body",
                ids=["doc-p-overlap"],
                file_paths="ctor.[native-P].txt",
                track_id="track-p-overlap",
                process_options="P",
            )
            row = await rag.full_docs.get_by_id("doc-p-overlap")
            await rag.apipeline_process_enqueue_documents()
        finally:
            await rag.finalize_storages()
        return row

    row = asyncio.run(_run())
    assert row["chunk_options"]["paragraph_semantic"]["chunk_overlap_token_size"] == 66
    assert captured["kwargs"]["chunk_overlap_token_size"] == 66


@pytest.mark.offline
def test_addon_params_strategy_wins_over_strategy_env(tmp_path, monkeypatch):
    """Highest tier check: a value sitting in
    ``addon_params['chunker'][<strategy>]['chunk_overlap_token_size']``
    must beat even a strategy-specific env."""
    monkeypatch.setenv("CHUNK_R_OVERLAP_SIZE", "42")

    async def _run():
        rag = _new_rag(
            tmp_path,
            addon_params={
                "chunker": {
                    "recursive_character": {
                        "chunk_overlap_token_size": 999,
                        "separators": ["\n\n", "\n", " ", ""],
                    },
                },
            },
        )
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                "Body for addon-vs-env precedence test.",
                ids=["doc-addon-vs-env"],
                file_paths="addon.[native-R].txt",
                track_id="track-addon",
                process_options="R",
            )
            row = await rag.full_docs.get_by_id("doc-addon-vs-env")
        finally:
            await rag.finalize_storages()
        return row

    row = asyncio.run(_run())
    chunk_opts = row["chunk_options"]
    assert chunk_opts["recursive_character"]["chunk_overlap_token_size"] == 999, (
        "addon_params explicit value must beat strategy-specific env."
    )


@pytest.mark.offline
def test_runtime_addon_params_mutation_affects_subsequent_enqueue(tmp_path):
    """Mutating ``rag.addon_params['chunker']`` after construction must
    take effect for documents enqueued *after* the mutation, while
    documents enqueued *before* keep their frozen snapshot.
    """

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            # Doc A enqueued under default config (R strategy so the
            # mutated separators land in the persisted slim snapshot).
            await rag.apipeline_enqueue_documents(
                "first doc body",
                ids=["doc-pre-mutation"],
                file_paths=["pre.[native-R].txt"],
                track_id="track-pre",
                process_options="R",
            )
            row_pre = await rag.full_docs.get_by_id("doc-pre-mutation")
            sep_pre = list(
                row_pre["chunk_options"]["recursive_character"]["separators"]
            )

            # Mutate the runtime defaults.
            rag.addon_params["chunker"]["recursive_character"]["separators"] = [
                "##",
                "\n",
            ]

            # Doc B enqueued under the mutated defaults.
            await rag.apipeline_enqueue_documents(
                "second doc body",
                ids=["doc-post-mutation"],
                file_paths=["post.[native-R].txt"],
                track_id="track-post",
                process_options="R",
            )
            row_post = await rag.full_docs.get_by_id("doc-post-mutation")
        finally:
            await rag.finalize_storages()
        return sep_pre, row_post

    sep_pre, row_post = asyncio.run(_run())

    # Pre-mutation doc keeps the env-driven default cascade.
    assert sep_pre == list(DEFAULT_R_SEPARATORS)
    # Post-mutation doc reflects the runtime change.
    assert row_post["chunk_options"]["recursive_character"]["separators"] == [
        "##",
        "\n",
    ]


@pytest.mark.offline
def test_runtime_chunker_config_is_corrected_once_and_cached(tmp_path, monkeypatch):
    """Top-level replacement warns immediately; nested mutation warns once at enqueue.

    Nested ``addon_params`` dicts intentionally remain mutable for compatibility,
    so snapshot construction is the first observable boundary after that style of
    update. It must replace the bad live value as well as return a safe snapshot.
    """
    import lightrag.chunker.recursive_character as recursive_character

    from lightrag.parser.routing import resolve_chunk_options

    rag = _new_rag(tmp_path)
    too_many = [f"runtime-cache-{index}" for index in range(70)]
    warnings: list[str] = []
    monkeypatch.setattr(recursive_character.logger, "warning", warnings.append)

    rag.addon_params["chunker"] = {"recursive_character": {"separators": too_many}}

    assert (
        rag.addon_params["chunker"]["recursive_character"]["separators"]
        == too_many[:64]
    )
    assert (
        sum(
            "[addon_params['chunker']] separator cascade" in message
            for message in warnings
        )
        == 1
    )

    warnings.clear()
    all_overlong = ["x" * 257] * 3
    rag.addon_params["chunker"]["recursive_character"]["separators"] = all_overlong

    first = resolve_chunk_options(rag.addon_params, process_options="R")
    second = resolve_chunk_options(rag.addon_params, process_options="R")

    assert first["recursive_character"]["separators"] == []
    assert second["recursive_character"]["separators"] == []
    assert rag.addon_params["chunker"]["recursive_character"]["separators"] == []
    assert (
        sum("[addon_params['chunker']] dropped" in message for message in warnings) == 1
    )


@pytest.mark.offline
def test_correction_keeps_the_nested_dict_a_caller_still_holds(tmp_path):
    """A correction must not detach the sub-dict the documented idiom hands out.

    ``default_addon_params`` advertises
    ``rag.addon_params["chunker"]["recursive_character"][...] = ...`` as the way
    to retune the chunker at runtime, which means callers legitimately keep a
    reference to that nested dict. Replacing it during a correction leaves the
    caller writing into an orphan: every later retune is silently discarded.
    """
    from lightrag.parser.routing import resolve_chunk_options

    rag = _new_rag(tmp_path)
    recursive = rag.addon_params["chunker"]["recursive_character"]

    recursive["separators"] = [f"identity-{index}" for index in range(70)]
    resolve_chunk_options(rag.addon_params, process_options="R")

    live = rag.addon_params["chunker"]["recursive_character"]
    assert live is recursive
    assert len(recursive["separators"]) == 64

    # The reference must still steer subsequent documents.
    recursive["separators"] = ["|"]
    snapshot = resolve_chunk_options(rag.addon_params, process_options="R")
    assert snapshot["recursive_character"]["separators"] == ["|"]


@pytest.mark.offline
def test_top_level_replacement_also_keeps_the_supplied_nested_dict(tmp_path):
    """Same guarantee for the assignment path, which corrects in place too."""
    rag = _new_rag(tmp_path)
    recursive = {"separators": [f"assigned-{index}" for index in range(70)]}

    rag.addon_params["chunker"] = {"recursive_character": recursive}

    assert rag.addon_params["chunker"]["recursive_character"] is recursive
    assert len(recursive["separators"]) == 64


@pytest.mark.offline
def test_building_a_snapshot_does_not_invalidate_the_prompt_profile_cache(tmp_path):
    """``resolve_chunk_options`` corrects in place, so it must not mark dirty.

    Going through ``ObservableAddonParams.__setitem__`` would fire the change
    callback and force ``_refresh_addon_params_cache`` — re-resolving the entity
    extraction prompt profile (which can read a file) — on an enqueue that only
    ever needed to bound a list.
    """
    from lightrag.parser.routing import resolve_chunk_options

    rag = _new_rag(tmp_path)
    rag.addon_params["chunker"]["recursive_character"]["separators"] = [
        f"dirty-{index}" for index in range(70)
    ]
    rag._addon_params_dirty = False

    resolve_chunk_options(rag.addon_params, process_options="R")

    assert rag._addon_params_dirty is False


@pytest.mark.offline
def test_a_string_separators_value_is_dropped_not_split_into_characters(
    tmp_path, monkeypatch
):
    """A ``str`` satisfies ``Sequence[str]`` — bounding it would corrupt config.

    Iterating a 300-character string yields 300 one-character "separators",
    which the bound then truncates to 64. Persisting that back into
    ``addon_params`` would destroy the operator's original value, replace it
    with something that looks legitimate, and report it as a 300-entry cascade.
    """
    import lightrag.chunker.recursive_character as recursive_character

    from lightrag.parser.routing import resolve_chunk_options

    rag = _new_rag(tmp_path)
    warnings: list[str] = []
    monkeypatch.setattr(recursive_character.logger, "warning", warnings.append)

    rag.addon_params["chunker"] = {"recursive_character": {"separators": "x" * 300}}

    recursive = rag.addon_params["chunker"]["recursive_character"]
    assert "separators" not in recursive
    assert sum("must be a list of strings, got str" in m for m in warnings) == 1

    # And the snapshot takes the documented ``separators=None`` path rather than
    # carrying 64 single characters.
    snapshot = resolve_chunk_options(rag.addon_params, process_options="R")
    assert "separators" not in snapshot["recursive_character"]

    warnings.clear()
    resolve_chunk_options(rag.addon_params, process_options="R")
    assert warnings == []


@pytest.mark.offline
def test_constructor_supplied_chunker_is_bounded_without_mutating_the_caller(
    tmp_path, monkeypatch
):
    """``LightRAG(addon_params=...)`` is a configuration ingress point too.

    It corrects by copying: the caller's mapping is an argument, not the
    instance's live configuration, so mutating it would be a surprise.
    """
    import lightrag.chunker.recursive_character as recursive_character

    warnings: list[str] = []
    monkeypatch.setattr(recursive_character.logger, "warning", warnings.append)

    too_many = [f"ctor-{index}" for index in range(70)]
    supplied = {"chunker": {"recursive_character": {"separators": list(too_many)}}}

    rag = _new_rag(tmp_path, addon_params=supplied)

    assert (
        rag.addon_params["chunker"]["recursive_character"]["separators"]
        == too_many[:64]
    )
    assert supplied["chunker"]["recursive_character"]["separators"] == too_many
    assert (
        sum(
            "[addon_params['chunker']] separator cascade" in message
            for message in warnings
        )
        == 1
    )


@pytest.mark.offline
def test_a_plain_mapping_is_not_corrected_but_is_still_bounded_downstream():
    """Only live ``ObservableAddonParams`` gets the ingress correction.

    A plain mapping has no owner to cache the fix into, so warning about it
    would be the per-call amplification this design removes. The snapshot keeps
    the raw value and the chunker's silent backstop bounds it at execution.
    """
    from lightrag.chunker.recursive_character import normalize_r_separators
    from lightrag.parser.routing import resolve_chunk_options

    too_many = [f"plain-{index}" for index in range(70)]
    plain = {"chunker": {"recursive_character": {"separators": list(too_many)}}}

    snapshot = resolve_chunk_options(plain, process_options="R")

    assert snapshot["recursive_character"]["separators"] == too_many
    assert (
        len(normalize_r_separators(snapshot["recursive_character"]["separators"])) == 64
    )


@pytest.mark.parametrize(
    "strategy, target_attr, is_async, file_path, env_vars, ctor_size, caller_chunk_options, expected_size, expected_overlap",
    [
        pytest.param(
            "R",
            "chunking_by_recursive_character",
            False,
            "ctor.[native-R].txt",
            {"CHUNK_SIZE": "1200", "CHUNK_R_SIZE": "777"},
            None,
            None,
            777,
            None,
            id="r_dedicated_env",
        ),
        pytest.param(
            "R",
            "chunking_by_recursive_character",
            False,
            "ctor.[native-R].txt",
            {"CHUNK_R_SIZE": None, "CHUNK_SIZE": None},
            444,
            None,
            444,
            None,
            id="r_fallback_global",
        ),
        pytest.param(
            "V",
            "chunking_by_semantic_vector",
            True,
            "ctor.[native-V].txt",
            {"CHUNK_SIZE": "1200", "CHUNK_V_SIZE": "2500"},
            None,
            None,
            2500,
            None,
            id="v_dedicated_env",
        ),
        pytest.param(
            "V",
            "chunking_by_semantic_vector",
            True,
            "ctor.[native-V].txt",
            {"CHUNK_V_SIZE": None, "CHUNK_SIZE": None},
            555,
            None,
            555,
            None,
            id="v_fallback_global",
        ),
        pytest.param(
            "F",
            "chunking_by_fixed_token",
            False,
            "ctor-f.txt",
            {"CHUNK_SIZE": "1200"},
            None,
            {
                "chunk_token_size": 1200,
                "fixed_token": {
                    "chunk_token_size": 333,
                    "chunk_overlap_token_size": 7,
                    "split_by_character": None,
                    "split_by_character_only": False,
                },
            },
            333,
            7,
            id="f_subdict_override",
        ),
        pytest.param(
            "F",
            "chunking_by_fixed_token",
            False,
            "ctor-f.txt",
            {"CHUNK_SIZE": None},
            456,
            None,
            456,
            None,
            id="f_fallback_top_level",
        ),
        pytest.param(
            "F",
            "chunking_by_fixed_token",
            False,
            "ctor-f.txt",
            {"CHUNK_F_SIZE": "640", "CHUNK_SIZE": None},
            999,
            None,
            640,
            None,
            id="f_env_wins_over_ctor",
        ),
    ],
)
def test_rvf_strategy_chunk_size_precedence(
    tmp_path,
    monkeypatch,
    strategy,
    target_attr,
    is_async,
    file_path,
    env_vars,
    ctor_size,
    caller_chunk_options,
    expected_size,
    expected_overlap,
):
    """R/V/F strategy ``chunk_token_size`` precedence: explicit sub-dict >
    strategy env (``CHUNK_{F,R,V}_SIZE``) > legacy ctor/global ``CHUNK_SIZE``."""
    import lightrag.chunker as chunker_pkg

    for key, value in env_vars.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)

    captured: dict = {}

    if is_async:

        async def _spy(
            tokenizer, content, chunk_token_size, *, embedding_func=None, **kwargs
        ):
            captured["chunk_token_size"] = chunk_token_size
            captured["kwargs"] = dict(kwargs)
            return [{"tokens": 5, "content": "stub", "chunk_order_index": 0}]

    else:

        def _spy(tokenizer, content, chunk_token_size, **kwargs):
            captured["chunk_token_size"] = chunk_token_size
            captured["kwargs"] = dict(kwargs)
            return [{"tokens": 5, "content": "stub", "chunk_order_index": 0}]

    monkeypatch.setattr(chunker_pkg, target_attr, _spy)

    rag_kwargs = {"chunk_token_size": ctor_size} if ctor_size is not None else {}
    enqueue_kwargs = (
        {"chunk_options": caller_chunk_options}
        if caller_chunk_options is not None
        else {}
    )

    async def _run():
        rag = _new_rag(tmp_path, **rag_kwargs)
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                f"stand-in body for {strategy} chunker",
                file_paths=file_path,
                track_id=f"track-{strategy.lower()}-size",
                process_options=strategy,
                **enqueue_kwargs,
            )
            await rag.apipeline_process_enqueue_documents()
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
    assert captured.get("chunk_token_size") == expected_size, (
        f"Expected {strategy} chunk_token_size={expected_size}; got {captured!r}"
    )
    # Dispatcher must pop chunk_token_size so it isn't double-passed as kwarg.
    assert "chunk_token_size" not in captured["kwargs"]
    if expected_overlap is not None:
        assert captured["kwargs"]["chunk_overlap_token_size"] == expected_overlap


def test_ainsert_legacy_path_honors_f_size_env(tmp_path, monkeypatch):
    """``rag.ainsert()`` runs the legacy ``chunking_func`` branch and must still
    honor ``CHUNK_F_SIZE`` (``fixed_token.chunk_token_size``) over global
    ``CHUNK_SIZE``."""
    monkeypatch.setenv("CHUNK_SIZE", "1200")
    monkeypatch.setenv("CHUNK_F_SIZE", "640")

    captured: dict = {}

    def _chunking_func_spy(
        tokenizer,
        content,
        split_by_character,
        split_by_character_only,
        overlap,
        chunk_token_size,
    ):
        captured["chunk_token_size"] = chunk_token_size
        return [{"tokens": 5, "content": "stub", "chunk_order_index": 0}]

    async def _run():
        rag = _new_rag(tmp_path)
        rag.chunking_func = _chunking_func_spy
        await rag.initialize_storages()
        try:
            await rag.ainsert("legacy path body", file_paths="legacy-f.txt")
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
    assert captured.get("chunk_token_size") == 640, (
        "ainsert legacy chunking_func must receive CHUNK_F_SIZE-derived size, "
        f"not the global CHUNK_SIZE; got {captured!r}"
    )


@pytest.mark.parametrize(
    "env_vars, partial_chunker, expected_sizes, absent_slots, extra_check",
    [
        pytest.param(
            {"CHUNK_F_SIZE": "640", "CHUNK_R_SIZE": "777", "CHUNK_V_SIZE": "888"},
            {"fixed_token": {"split_by_character": "\n"}},
            {"fixed_token": 640, "recursive_character": 777, "semantic_vector": 888},
            (),
            ("fixed_token", "split_by_character", "\n"),
            id="backfills_size_envs",
        ),
        pytest.param(
            {"CHUNK_F_SIZE": "640"},
            {"fixed_token": {"chunk_token_size": 320}},
            {"fixed_token": 320},
            (),
            None,
            id="explicit_size_beats_env",
        ),
        pytest.param(
            {"CHUNK_F_SIZE": None, "CHUNK_R_SIZE": None},
            {"recursive_character": {"separators": ["X"]}},
            {},
            ("recursive_character", "fixed_token"),
            None,
            id="no_env_leaves_slot_absent",
        ),
    ],
)
def test_partial_chunker_config_size_env_resolution(
    tmp_path,
    monkeypatch,
    env_vars,
    partial_chunker,
    expected_sizes,
    absent_slots,
    extra_check,
):
    """Partial ``addon_params['chunker']`` backfills strategy size envs unless
    explicitly supplied, and leaves slots absent when no size env is set."""
    for key, value in env_vars.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)

    rag = _new_rag(tmp_path, addon_params={"chunker": partial_chunker})
    chunker = rag.addon_params["chunker"]
    for slot, expected in expected_sizes.items():
        assert chunker[slot]["chunk_token_size"] == expected
    for slot in absent_slots:
        assert "chunk_token_size" not in chunker[slot]
    if extra_check is not None:
        slot, key, value = extra_check
        assert chunker[slot][key] == value


# --------------------------------------------------------------------------- #
# drop_references: switch is snapshotted (+ recorded in chunk_opts metadata);
# detection knobs (tail/headings) are NOT snapshotted (read live by chunker).
# --------------------------------------------------------------------------- #


def test_drop_references_env_snapshotted_but_knobs_not(tmp_path, monkeypatch):
    """``CHUNK_P_DROP_REFERENCES`` flows into the persisted snapshot and reaches
    the chunker even on a partial ``addon_params`` (runtime-mutation path that
    bypasses ``_apply_chunk_size_overlay``).  The detection knobs
    ``CHUNK_P_REFERENCES_TAIL_N`` / ``CHUNK_P_REFERENCES_HEADINGS`` must NOT be
    snapshotted — the chunker reads them live."""
    monkeypatch.setenv("CHUNK_P_DROP_REFERENCES", "true")
    monkeypatch.setenv("CHUNK_P_REFERENCES_TAIL_N", "3")
    monkeypatch.setenv("CHUNK_P_REFERENCES_HEADINGS", "Foo|Bar")

    import lightrag.chunker as chunker_pkg

    captured: dict = {}

    def _p_spy(tokenizer, content, chunk_token_size, *, blocks_path=None, **kwargs):
        captured.update(kwargs)
        return [{"tokens": 5, "content": "stub", "chunk_order_index": 0}]

    monkeypatch.setattr(chunker_pkg, "chunking_by_paragraph_semantic", _p_spy)

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            rag.addon_params["chunker"] = {"paragraph_semantic": {}}
            await rag.apipeline_enqueue_documents(
                "drop references body",
                ids=["doc-drop-rf"],
                file_paths="ctor.[native-P].txt",
                track_id="track-p-drop-rf",
                process_options="P",
            )
            row = await rag.full_docs.get_by_id("doc-drop-rf")
            await rag.apipeline_process_enqueue_documents()
        finally:
            await rag.finalize_storages()
        return row

    row = asyncio.run(_run())
    assert row is not None
    p_opts = row["chunk_options"]["paragraph_semantic"]
    assert p_opts.get("drop_references") is True
    assert "references_tail_n" not in p_opts
    assert "references_headings" not in p_opts
    assert captured.get("drop_references") is True
    assert "references_tail_n" not in captured
    assert "references_headings" not in captured


def test_explicit_drop_references_false_overrides_env_true(tmp_path, monkeypatch):
    """An explicit ``drop_references=False`` in a caller-supplied
    ``chunk_options`` wins over ``CHUNK_P_DROP_REFERENCES=true``."""
    monkeypatch.setenv("CHUNK_P_DROP_REFERENCES", "true")

    import lightrag.chunker as chunker_pkg
    from lightrag.parser.routing import resolve_chunk_options

    captured: dict = {}

    def _p_spy(tokenizer, content, chunk_token_size, *, blocks_path=None, **kwargs):
        captured.update(kwargs)
        return [{"tokens": 5, "content": "stub", "chunk_order_index": 0}]

    monkeypatch.setattr(chunker_pkg, "chunking_by_paragraph_semantic", _p_spy)

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            opts = resolve_chunk_options(rag.addon_params, process_options="P")
            assert opts["paragraph_semantic"]["drop_references"] is True
            opts["paragraph_semantic"]["drop_references"] = False
            await rag.apipeline_enqueue_documents(
                "explicit override body",
                file_paths="ctor.[native-P].txt",
                track_id="track-p-drop-rf-false",
                process_options="P",
                chunk_options=opts,
            )
            await rag.apipeline_process_enqueue_documents()
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
    assert captured.get("drop_references") is False


def test_chunk_opts_metadata_records_only_drop_rf():
    """``_format_chunking_params`` renders ``drop_references`` under the short
    ``drop_rf`` alias and never the env-only detection knobs."""
    from lightrag.pipeline import _format_chunking_params

    line = _format_chunking_params(2000, {"drop_references": True})
    assert "drop_rf=True" in line
    assert "drop_references" not in line
    assert "rf_tail" not in line and "rf_heads" not in line


@pytest.mark.parametrize(
    "caller_chunk_options",
    [
        pytest.param(None, id="forwarded_from_deployment"),
        pytest.param(
            {
                "semantic_vector": {
                    "embedding_batch_num": 999,
                    "embedding_max_async": 999,
                }
            },
            id="discards_colliding_addon_keys",
        ),
    ],
)
def test_v_strategy_embedding_request_bounds(
    tmp_path, monkeypatch, caller_chunk_options
):
    """The dispatcher forwards deployment ``embedding_batch_num`` /
    ``embedding_max_async`` to V and discards any same-named keys in the V
    addon slot so they neither override nor collide with keyword dispatch."""
    import lightrag.chunker as chunker_pkg

    captured: dict = {}

    async def _v_spy(
        tokenizer, content, chunk_token_size, *, embedding_func=None, **kwargs
    ):
        captured["kwargs"] = dict(kwargs)
        return [{"tokens": 5, "content": "stub", "chunk_order_index": 0}]

    monkeypatch.setattr(chunker_pkg, "chunking_by_semantic_vector", _v_spy)

    enqueue_kwargs = (
        {"chunk_options": caller_chunk_options}
        if caller_chunk_options is not None
        else {}
    )

    async def _run():
        rag = _new_rag(tmp_path, embedding_batch_num=7, embedding_func_max_async=5)
        await rag.initialize_storages()
        try:
            await rag.apipeline_enqueue_documents(
                "stand-in body for semantic-vector chunker",
                file_paths="bounds.[native-V].txt",
                track_id="track-v-bounds",
                process_options="V",
                **enqueue_kwargs,
            )
            await rag.apipeline_process_enqueue_documents()
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())

    assert "kwargs" in captured, "the V chunker was never dispatched"
    kwargs = captured["kwargs"]
    assert kwargs.get("embedding_batch_num") == 7
    assert kwargs.get("embedding_max_async") == 5

