"""Tests for KG query token budgeting, memoization, and chunk selection.

Covers:
- ``kg_chunk_pick_method`` validation and canonicalization at construction.
- Total KG token budget enforcement over graph records, chunks, references, and
  live ``user_prompt_prefix`` overrides.
- Per-request token memoization and one-shot graph record serialization.
- ``_vector_chunk_quota`` floor/truncation and single-group VECTOR selection.
- Entity vs. relation chunk-selection budget parity and VECTOR -> WEIGHT
  fallback when prerequisites are absent.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from lightrag import LightRAG, operate
from lightrag.base import QueryParam
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.operate import (
    _find_related_text_unit_from_entities,
    _find_related_text_unit_from_relations,
    _vector_chunk_quota,
    normalize_kg_chunk_pick_method,
)
from lightrag.prompt import PROMPTS
from lightrag.utils import (
    EmbeddingFunc,
    TiktokenTokenizer,
    Tokenizer,
    resolve_user_prompt,
)

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Shared tokenizers & stubs
# ---------------------------------------------------------------------------


class _Characters:
    def encode(self, content: str) -> list[int]:
        return [ord(char) for char in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


async def _unused_llm(*_args, **_kwargs) -> str:
    return ""


async def _unused_embedding(texts: list[str]) -> np.ndarray:
    return np.zeros((len(texts), 8), dtype=np.float32)


# ---------------------------------------------------------------------------
# kg_chunk_pick_method validation & canonicalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [("vector", "VECTOR"), (" Weight ", "WEIGHT")],
)
def test_normalize_accepts_any_case_and_padding(given, expected):
    assert normalize_kg_chunk_pick_method(given) == expected


@pytest.mark.parametrize("invalid", ["WEIGTH", None], ids=["typo", "not_a_string"])
def test_normalize_rejects_unknown_values(invalid):
    with pytest.raises(ValueError, match="KG_CHUNK_PICK_METHOD"):
        normalize_kg_chunk_pick_method(invalid)


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [("   ", "VECTOR"), ("weight", "weight")],
    ids=["blank_env_uses_default", "env_read_at_construction"],
)
def test_field_default_reads_env_and_treats_blank_as_unset(
    monkeypatch, env_value, expected
):
    """A blank ``KG_CHUNK_PICK_METHOD=`` in .env must not fail the startup check."""
    monkeypatch.setenv("KG_CHUNK_PICK_METHOD", env_value)
    field = LightRAG.__dataclass_fields__["kg_chunk_pick_method"]

    assert field.default_factory() == expected


def test_storage_snapshot_holds_the_canonical_value(tmp_path):
    """The query path reads the storages' config snapshot, never normalizing again."""
    rag = LightRAG(
        working_dir=str(tmp_path),
        llm_model_func=_unused_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_unused_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _Characters()),
        kg_chunk_pick_method="weight",
    )

    assert rag.kg_chunk_pick_method == "WEIGHT"
    assert rag.text_chunks.global_config["kg_chunk_pick_method"] == "WEIGHT"


# ---------------------------------------------------------------------------
# Total KG token budget
# ---------------------------------------------------------------------------

QUERY = "Who knows Alice?"
TEMPLATE = "Evidence:\n{context_data}\nInstructions:{user_prompt}"


def _config(tokenizer=None):
    return {"tokenizer": tokenizer or Tokenizer("characters", _Characters())}


def _entity(name, description="short evidence"):
    return {
        "entity_name": name,
        "entity_type": "person",
        "description": description,
        "source_id": f"chunk-{name}",
    }


def _relation(source, target, description="short relationship"):
    return {
        "src_id": source,
        "tgt_id": target,
        "description": description,
        "source_id": f"chunk-{source}-{target}",
    }


def _search(entities, relations, chunks=None):
    return {
        "final_entities": entities,
        "final_relations": relations,
        "vector_chunks": chunks or [],
        "chunk_tracking": {chunk["chunk_id"]: {} for chunk in chunks or []},
        "query_embedding": None,
    }


def _prompt_tokens(context, param, config, template=TEMPLATE):
    prompt = template.format(
        context_data=context,
        response_type=param.response_type,
        user_prompt=resolve_user_prompt(
            param.user_prompt,
            config.get("user_prompt_prefix", ""),
            param.disable_user_prompt_prefix,
        ).slot,
    )
    tokenizer = config["tokenizer"]
    return len(tokenizer.encode(prompt)) + len(tokenizer.encode(QUERY))


async def _build(monkeypatch, search, param, config, template=TEMPLATE):
    monkeypatch.setattr(operate, "_perform_kg_search", AsyncMock(return_value=search))
    merge = AsyncMock(return_value=search["vector_chunks"])
    monkeypatch.setattr(operate, "_merge_all_chunks", merge)
    result = await operate._build_query_context(
        QUERY,
        "Alice",
        "knows",
        None,
        None,
        None,
        SimpleNamespace(global_config=config),
        param,
        system_prompt=template,
    )
    return result, merge


@pytest.mark.parametrize("mode", ["local", "global", "hybrid", "mix"])
@pytest.mark.parametrize("kind", ["entity", "relation", "both"])
async def test_total_budget_filters_graph_before_chunk_selection(
    monkeypatch, mode, kind
):
    entities = [_entity("Alice"), _entity("Bob", "e" * 3000)]
    relations = [_relation("Alice", "Carol"), _relation("Bob", "Carol", "r" * 3000)]
    if kind == "entity":
        relations = []
    if kind == "relation":
        entities = []
    config = _config()
    param = QueryParam(
        mode=mode,
        max_total_tokens=1000,
        max_entity_tokens=6000,
        max_relation_tokens=8000,
        enable_rerank=False,
    )
    result, merge = await _build(
        monkeypatch, _search(entities, relations), param, config
    )
    assert result is not None
    assert _prompt_tokens(result.context, param, config) <= param.max_total_tokens
    selected = merge.call_args.kwargs
    assert selected["filtered_entities"] == entities[:1]
    assert selected["filtered_relations"] == relations[:1]
    data = result.raw_data["data"]
    assert [row["entity_name"] for row in data["entities"]] == [
        row["entity_name"] for row in selected["filtered_entities"]
    ]
    assert [(row["src_id"], row["tgt_id"]) for row in data["relationships"]] == [
        (row["src_id"], row["tgt_id"]) for row in selected["filtered_relations"]
    ]
    assert all(row["source_id"] for row in data["entities"] + data["relationships"])


async def test_oversized_entity_does_not_starve_fitting_relationship(monkeypatch):
    config = _config()
    param = QueryParam(max_total_tokens=1000, enable_rerank=False)
    relationship = _relation("Alice", "Bob")
    result, merge = await _build(
        monkeypatch,
        _search([_entity("Alice", "e" * 3000)], [relationship]),
        param,
        config,
    )
    assert result is not None
    assert merge.call_args.kwargs["filtered_entities"] == []
    assert merge.call_args.kwargs["filtered_relations"] == [relationship]
    assert _prompt_tokens(result.context, param, config) <= 1000


async def test_only_retained_graph_records_supply_chunks(monkeypatch):
    config = _config()
    param = QueryParam(mode="hybrid", max_total_tokens=1000, enable_rerank=False)
    entities = [_entity("Alice"), _entity("Bob", "e" * 3000)]
    monkeypatch.setattr(
        operate, "_perform_kg_search", AsyncMock(return_value=_search(entities, []))
    )

    async def related_chunks(records, *args, **kwargs):
        return [
            {
                "chunk_id": record["source_id"],
                "content": f"Evidence for {record['entity_name']}",
                "file_path": f"{record['entity_name']}.txt",
            }
            for record in records
        ]

    source_selection = AsyncMock(side_effect=related_chunks)
    monkeypatch.setattr(
        operate, "_find_related_text_unit_from_entities", source_selection
    )
    result = await operate._build_query_context(
        QUERY,
        "Alice",
        "knows",
        None,
        None,
        None,
        SimpleNamespace(global_config=config),
        param,
        system_prompt=TEMPLATE,
    )
    assert result is not None
    assert source_selection.call_args.args[0] == entities[:1]
    assert [chunk["chunk_id"] for chunk in result.raw_data["data"]["chunks"]] == [
        "chunk-Alice"
    ]
    assert "Evidence for Bob" not in result.context
    assert _prompt_tokens(result.context, param, config) <= 1000


@pytest.mark.parametrize(
    "template", [TEMPLATE, "{context_data}{context_data}{user_prompt}"]
)
async def test_graph_only_exact_boundary_and_zero_capacity(template):
    config = _config()
    entities = [{"entity": "Alice", "type": "person", "description": "e" * 300}]
    param = QueryParam(max_total_tokens=10000, enable_rerank=False)
    context, raw = await operate._build_context_str(
        entities, [], [], QUERY, param, config, system_prompt=template
    )
    assert raw["status"] == "success"
    required = _prompt_tokens(context, param, config, template)
    param.max_total_tokens = required
    same, raw = await operate._build_context_str(
        entities, [], [], QUERY, param, config, system_prompt=template
    )
    assert same == context
    assert len(raw["data"]["entities"]) == 1
    param.max_total_tokens = required - 1
    empty, raw = await operate._build_context_str(
        entities, [], [], QUERY, param, config, system_prompt=template
    )
    assert empty == ""
    assert raw["metadata"]["token_budget_exceeded"] is True


async def test_long_reference_paths_are_charged_and_reference_ids_rebuilt():
    config = _config()
    param = QueryParam(mode="mix", max_total_tokens=1800, enable_rerank=False)
    chunks = [
        {"chunk_id": "first", "content": "first evidence", "file_path": "a.md"},
        {"chunk_id": "last", "content": "last evidence", "file_path": "z" * 2000},
    ]
    context, raw = await operate._build_context_str(
        [], [], chunks, QUERY, param, config, system_prompt=TEMPLATE
    )
    assert _prompt_tokens(context, param, config) <= param.max_total_tokens
    assert [chunk["chunk_id"] for chunk in raw["data"]["chunks"]] == ["first"]
    assert raw["data"]["references"] == [{"reference_id": "1", "file_path": "a.md"}]
    assert '"reference_id": "1"' in context
    assert "last evidence" not in context


@pytest.mark.parametrize("limit", [1, 650])
async def test_budget_exhaustion_never_calls_answer_model(monkeypatch, limit):
    config = _config()
    model = AsyncMock(return_value="answer")
    config["role_llm_funcs"] = {"query": model}
    param = QueryParam(
        mode="hybrid",
        max_total_tokens=limit,
        enable_rerank=False,
        hl_keywords=["knows"],
        ll_keywords=["Alice"],
    )
    search = _search([_entity("Alice", "e" * 3000)], [])
    monkeypatch.setattr(operate, "_perform_kg_search", AsyncMock(return_value=search))
    result = await operate.kg_query(
        QUERY,
        None,
        None,
        None,
        SimpleNamespace(global_config=config),
        param,
        config,
        system_prompt=TEMPLATE,
    )
    assert result is None
    model.assert_not_awaited()


async def test_real_tokenizer_default_prompt_and_user_instructions(monkeypatch):
    config = _config(TiktokenTokenizer())
    config["user_prompt_prefix"] = "Use the evidence. " * 10
    param = QueryParam(
        mode="mix", max_total_tokens=1000, user_prompt="Be brief.", enable_rerank=False
    )
    result, merge = await _build(
        monkeypatch,
        _search([_entity("Alice"), _entity("Bob", "Coherent evidence. " * 1200)], []),
        param,
        config,
        template=PROMPTS["rag_response"],
    )
    assert result is not None
    assert (
        _prompt_tokens(result.context, param, config, PROMPTS["rag_response"]) <= 1000
    )
    assert len(merge.call_args.kwargs["filtered_entities"]) == 1
    assert len(result.raw_data["data"]["entities"]) == 1


async def test_repeated_context_placeholder_counts_rendered_chunks():
    config = _config()
    param = QueryParam(mode="mix", max_total_tokens=2300, enable_rerank=False)
    template = "{context_data}{context_data}"
    chunks = [
        {"chunk_id": str(i), "content": str(i) * 500, "file_path": f"{i}.txt"}
        for i in range(2)
    ]
    context, raw = await operate._build_context_str(
        [], [], chunks, QUERY, param, config, system_prompt=template
    )
    assert raw["status"] == "success"
    assert len(raw["data"]["chunks"]) == 1
    assert _prompt_tokens(context, param, config, template) <= 2300


async def test_mix_vector_only_budget_exhaustion_does_not_call_answer(monkeypatch):
    config = _config()
    model = AsyncMock(return_value="answer")
    config["role_llm_funcs"] = {"query": model}
    param = QueryParam(
        mode="mix", max_total_tokens=650, hl_keywords=["knows"], enable_rerank=False
    )
    chunks = [{"chunk_id": "chunk", "content": "e" * 1000, "file_path": "e.txt"}]
    monkeypatch.setattr(
        operate, "_perform_kg_search", AsyncMock(return_value=_search([], [], chunks))
    )
    result = await operate.kg_query(
        QUERY,
        None,
        None,
        None,
        SimpleNamespace(global_config=config),
        param,
        config,
        system_prompt=TEMPLATE,
    )
    assert result is None
    model.assert_not_awaited()


async def test_live_prefix_overrides_storage_snapshot_for_budget_only(monkeypatch):
    storage_config = _config()
    storage_config["user_prompt_prefix"] = ""
    live_config = {**storage_config, "user_prompt_prefix": "P" * 2000}
    model = AsyncMock(return_value="answer")
    live_config["role_llm_funcs"] = {"query": model}
    param = QueryParam(
        mode="hybrid", max_total_tokens=1000, hl_keywords=["knows"], enable_rerank=False
    )
    monkeypatch.setattr(
        operate,
        "_perform_kg_search",
        AsyncMock(return_value=_search([_entity("Alice")], [])),
    )
    source_selection = AsyncMock(return_value=[])
    monkeypatch.setattr(operate, "_merge_all_chunks", source_selection)
    result = await operate.kg_query(
        QUERY,
        None,
        None,
        None,
        SimpleNamespace(global_config=storage_config),
        param,
        live_config,
        system_prompt=TEMPLATE,
    )
    assert result is None
    source_selection.assert_not_awaited()
    model.assert_not_awaited()
    assert storage_config["user_prompt_prefix"] == ""


async def test_live_prefix_does_not_replace_chunk_selection_snapshot(monkeypatch):
    storage_config = _config()
    storage_config.update(user_prompt_prefix="", related_chunk_number=7)
    live_config = {
        **storage_config,
        "user_prompt_prefix": "Live instructions.",
        "related_chunk_number": 2,
    }
    model = AsyncMock(return_value="answer")
    live_config["role_llm_funcs"] = {"query": model}
    param = QueryParam(
        mode="hybrid", max_total_tokens=1000, hl_keywords=["knows"], enable_rerank=False
    )
    monkeypatch.setattr(
        operate,
        "_perform_kg_search",
        AsyncMock(return_value=_search([_entity("Alice")], [])),
    )
    source_selection = AsyncMock(return_value=[])
    monkeypatch.setattr(operate, "_merge_all_chunks", source_selection)
    result = await operate.kg_query(
        QUERY,
        None,
        None,
        None,
        SimpleNamespace(global_config=storage_config),
        param,
        live_config,
        system_prompt=TEMPLATE,
    )
    assert result is not None
    selected_config = source_selection.call_args.kwargs["text_chunks_db"].global_config
    assert selected_config is storage_config
    assert selected_config["related_chunk_number"] == 7
    prompt = model.call_args.kwargs["system_prompt"]
    assert "Live instructions." in prompt
    assert len(storage_config["tokenizer"].encode(prompt)) + len(QUERY) <= 1000


def _no_chunk_room_warnings(records):
    return [
        record.getMessage()
        for record in records
        if record.levelno == logging.WARNING
        and "Token budget left no room" in record.getMessage()
    ]


async def test_graph_filling_the_budget_warns_and_answers_from_graph(
    lightrag_log_records,
):
    config = _config()
    param = QueryParam(mode="mix", max_total_tokens=1000, enable_rerank=False)
    entities = [{"entity": "Alice", "type": "person", "description": "e" * 300}]
    chunks = [{"chunk_id": "c", "content": "c" * 500, "file_path": "c.md"}]
    context, raw = await operate._build_context_str(
        entities, [], chunks, QUERY, param, config, system_prompt=TEMPLATE
    )
    assert raw["status"] == "success"
    assert len(raw["data"]["entities"]) == 1
    assert raw["data"]["chunks"] == []
    assert _prompt_tokens(context, param, config) <= param.max_total_tokens
    warnings = _no_chunk_room_warnings(lightrag_log_records)
    assert len(warnings) == 1
    assert "max_entity_tokens" in warnings[0]


async def test_reference_list_dropping_every_chunk_warns(lightrag_log_records):
    config = _config()
    param = QueryParam(mode="mix", max_total_tokens=1800, enable_rerank=False)
    entities = [{"entity": "Alice", "type": "person", "description": "short"}]
    chunks = [{"chunk_id": "z", "content": "z evidence", "file_path": "z" * 2000}]
    context, raw = await operate._build_context_str(
        entities, [], chunks, QUERY, param, config, system_prompt=TEMPLATE
    )
    assert raw["status"] == "success"
    assert raw["data"]["chunks"] == []
    assert "z evidence" not in context
    warnings = _no_chunk_room_warnings(lightrag_log_records)
    assert len(warnings) == 1
    assert "reference list" in warnings[0]
    assert "graph context only" in warnings[0]


async def test_vector_only_budget_exhaustion_warns(lightrag_log_records):
    config = _config()
    param = QueryParam(mode="mix", max_total_tokens=650, enable_rerank=False)
    chunks = [{"chunk_id": "chunk", "content": "e" * 1000, "file_path": "e.txt"}]
    context, raw = await operate._build_context_str(
        [], [], chunks, QUERY, param, config, system_prompt=TEMPLATE
    )
    assert context == ""
    assert raw["status"] == "failure"
    warnings = _no_chunk_room_warnings(lightrag_log_records)
    assert len(warnings) == 1
    assert "no result" in warnings[0]


async def test_chunks_that_fit_log_no_budget_warning(lightrag_log_records):
    config = _config()
    param = QueryParam(mode="mix", max_total_tokens=10000, enable_rerank=False)
    entities = [{"entity": "Alice", "type": "person", "description": "short"}]
    chunks = [{"chunk_id": "c", "content": "evidence", "file_path": "c.md"}]
    _, raw = await operate._build_context_str(
        entities, [], chunks, QUERY, param, config, system_prompt=TEMPLATE
    )
    assert len(raw["data"]["chunks"]) == 1
    assert _no_chunk_room_warnings(lightrag_log_records) == []


# ---------------------------------------------------------------------------
# Per-request token memo & one-shot graph serialization
# ---------------------------------------------------------------------------

# The baseline measurement classifies prompt-sized encodes as texts over 1KB.
_PROMPT_CHARS = 1024


class _CountingTokenizer(TiktokenTokenizer):
    """Real tiktoken encoding that records every encode call's exact text."""

    def __init__(self):
        super().__init__()
        self.encoded: list[str] = []

    def encode(self, content):
        self.encoded.append(content)
        return super().encode(content)


def _context_entity(name, description="short evidence"):
    return {
        "entity": name,
        "type": "person",
        "description": description,
        "created_at": "2026-01-01 00:00:00",
        "file_path": f"{name}.md",
    }


def _context_relation(source, target, description="short relationship"):
    return {
        "entity1": source,
        "entity2": target,
        "description": description,
        "created_at": "2026-01-01 00:00:00",
        "file_path": f"{source}-{target}.md",
    }


def _graph_records():
    entities = [_context_entity(f"E{i}", f"d{i} " * 40) for i in range(6)]
    relations = [
        _context_relation(f"E{i}", f"E{i + 1}", f"r{i} " * 40) for i in range(6)
    ]
    return entities, relations


def _memo_chunks():
    return [
        {
            "chunk_id": f"c{i}",
            "content": f"chunk evidence {i} " * 30,
            "file_path": f"c{i}.md",
        }
        for i in range(3)
    ]


def _fit(entities, relations, chunks, limit, tokenizer, memo, template=TEMPLATE):
    param = QueryParam(max_total_tokens=limit, enable_rerank=False)
    return operate._fit_kg_context_to_total_budget(
        entities,
        relations,
        chunks,
        QUERY,
        param,
        _config(tokenizer),
        template,
        memo,
    )


def _prompt_texts(texts):
    return [text for text in texts if len(text) > _PROMPT_CHARS]


def test_exact_boundary_hit_keeps_the_surviving_prefix():
    """A memo hit at the exact budget boundary returns the identical count.

    The limit equals the one-entity prompt's direct count, so the binary
    search's re-verification of the winning prefix is a memo hit on the
    boundary: a drifted hit count walks the survivor down to nothing.
    """
    entities = [_context_entity("E0"), _context_entity("E1")]
    tokenizer = TiktokenTokenizer()
    context = PROMPTS["kg_query_context"].format(
        entities_str=json.dumps(entities[0], ensure_ascii=False),
        relations_str="",
        text_chunks_str="",
        reference_list_str="",
    )
    limit = _prompt_tokens(context, QueryParam(enable_rerank=False), _config(tokenizer))
    direct = _fit(entities, [], [], limit, tokenizer, None)
    memoized = _fit(entities, [], [], limit, tokenizer, {})
    assert [record["entity"] for record in direct[0]] == ["E0"]
    assert memoized == direct


def test_lengthened_description_is_reencoded_and_memo_never_drifts():
    """The memo keys on exact text: a changed record misses, and every stored
    count equals a fresh direct encode of the same text."""
    extra_chars = 17
    entities, relations = _graph_records()
    tokenizer = _CountingTokenizer()
    memo = {}
    _fit(entities, relations, [], 1800, tokenizer, memo, template=None)
    keys_before = set(memo)

    lengthened = [
        {**entities[0], "description": entities[0]["description"] + "x" * extra_chars},
        *entities[1:],
    ]
    encoded_before = len(tokenizer.encoded)
    memoized = _fit(lengthened, relations, [], 1800, tokenizer, memo, template=None)
    new_texts = tokenizer.encoded[encoded_before:]
    assert any("x" * extra_chars in text for text in new_texts)
    assert set(memo) - keys_before

    direct = _fit(
        lengthened, relations, [], 1800, _CountingTokenizer(), None, template=None
    )
    assert memoized == direct

    oracle = TiktokenTokenizer()
    drift = [
        (text[:40], count, len(oracle.encode(text)))
        for text, count in memo.items()
        if count != len(oracle.encode(text))
    ]
    assert drift == []


@pytest.mark.parametrize(
    ("limit", "fits"),
    [
        pytest.param(1200, True, id="tight-many-probes"),
        pytest.param(1, False, id="exhausted"),
    ],
)
def test_graph_records_serialize_once_per_call_and_only_after_empty_probe(
    limit, fits, monkeypatch
):
    """Each graph record is serialized once per fitter call, however many
    probes render it, and not at all when the empty-context probe already
    exhausts the budget. Restoring per-probe dumps in render makes the tight
    case count probes x records; hoisting the serialization above the empty
    probe makes the exhausted case dump every record."""
    entities, relations = _graph_records()
    tokenizer = _CountingTokenizer()
    serialized = []
    real_dumps = json.dumps

    def counting_dumps(obj, *args, **kwargs):
        serialized.append(real_dumps(obj, *args, **kwargs))
        return serialized[-1]

    monkeypatch.setattr(operate.json, "dumps", counting_dumps)
    result = _fit(entities, relations, [], limit, tokenizer, {}, template=None)

    if not fits:
        assert result is None
        assert serialized == []
        return
    assert result is not None
    assert Counter(serialized) == Counter(
        real_dumps(record, ensure_ascii=False) for record in entities + relations
    )
    assert len(_prompt_texts(tokenizer.encoded)) > 2


async def test_request_pipeline_shares_one_memo_across_its_three_fits(monkeypatch):
    """Wiring: _build_query_context creates one memo and threads it through
    stage 2 and both stage-4 fits. Dropping the memo on the way in reproduces
    the stock encode count; the wired run eliminates exactly the fitter's
    repeated texts and probes the same distinct prompts."""
    entities = [_entity(f"E{i}", f"d{i} " * 40) for i in range(6)]
    relations = [_relation(f"E{i}", f"E{i + 1}", f"r{i} " * 40) for i in range(6)]
    chunks = _memo_chunks()
    real_fitter = operate._fit_kg_context_to_total_budget

    async def run(drop_memo):
        tokenizer = _CountingTokenizer()
        param = QueryParam(
            mode="mix",
            max_total_tokens=1800,
            max_entity_tokens=6000,
            max_relation_tokens=8000,
            enable_rerank=False,
        )
        fitter_texts = []

        def observed(*args):
            start = len(tokenizer.encoded)
            result = real_fitter(*args[:7]) if drop_memo else real_fitter(*args)
            fitter_texts.extend(tokenizer.encoded[start:])
            return result

        monkeypatch.setattr(
            operate,
            "_perform_kg_search",
            AsyncMock(return_value=_search(entities, relations, chunks)),
        )
        monkeypatch.setattr(
            operate, "_merge_all_chunks", AsyncMock(return_value=chunks)
        )
        monkeypatch.setattr(operate, "_fit_kg_context_to_total_budget", observed)
        result = await operate._build_query_context(
            QUERY,
            "Alice",
            "knows",
            None,
            None,
            None,
            SimpleNamespace(global_config=_config(tokenizer)),
            param,
        )
        return tokenizer.encoded, fitter_texts, result

    wired_texts, wired_fit, wired = await run(False)
    stock_texts, stock_fit, stock = await run(True)
    assert wired.context == stock.context
    assert wired.raw_data == stock.raw_data

    repeats = len(stock_fit) - len(set(stock_fit))
    assert repeats > 0
    assert len(wired_fit) == len(set(wired_fit))
    assert set(wired_fit) == set(stock_fit)
    assert len(wired_texts) == len(stock_texts) - repeats


# ---------------------------------------------------------------------------
# Vector chunk quota
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("max_related_chunks", "group_count", "expected"),
    [
        (0, 5, 0),  # related_chunk_number disabled entirely
        (5, 0, 0),  # defensive branch: no call site reaches this in practice
        (1, 1, 1),  # floor kicks in: 1 * 1 / 2 would round down to 0
        (2, 1, 1),  # 2 * 1 / 2 == 1, floor is a no-op here
        (4, 3, 6),  # normal scaling, no floor/rounding involved
        (3, 1, 1),  # 3 * 1 / 2 == 1.5, int() truncates to 1
    ],
)
def test_vector_chunk_quota(max_related_chunks, group_count, expected):
    assert _vector_chunk_quota(max_related_chunks, group_count) == expected


def _quota_text_chunks_db():
    return SimpleNamespace(
        global_config={
            "kg_chunk_pick_method": "VECTOR",
            "related_chunk_number": 1,
        },
        embedding_func=Mock(),
        get_by_ids=AsyncMock(return_value=[{"content": "chunk"}]),
    )


async def _vector_picker(**kwargs):
    if kwargs["num_of_chunks"] <= 0:
        return []
    return ["chunk-1"]


async def test_single_entity_uses_vector_picker_with_one_chunk(monkeypatch):
    vector_picker = AsyncMock(side_effect=_vector_picker)
    weighted_picker = Mock(return_value=["weighted-chunk"])
    monkeypatch.setattr("lightrag.operate.pick_by_vector_similarity", vector_picker)
    monkeypatch.setattr("lightrag.operate.pick_by_weighted_polling", weighted_picker)

    result = await _find_related_text_unit_from_entities(
        [{"entity_name": "entity", "source_id": "chunk-1"}],
        QueryParam(),
        _quota_text_chunks_db(),
        SimpleNamespace(),
        query="query",
        chunks_vdb=SimpleNamespace(),
        query_embedding=[1.0],
    )

    assert vector_picker.await_args.kwargs["num_of_chunks"] == 1
    weighted_picker.assert_not_called()
    assert [chunk["chunk_id"] for chunk in result] == ["chunk-1"]


async def test_single_relation_uses_vector_picker_with_one_chunk(monkeypatch):
    vector_picker = AsyncMock(side_effect=_vector_picker)
    weighted_picker = Mock(return_value=["weighted-chunk"])
    monkeypatch.setattr("lightrag.operate.pick_by_vector_similarity", vector_picker)
    monkeypatch.setattr("lightrag.operate.pick_by_weighted_polling", weighted_picker)

    result = await _find_related_text_unit_from_relations(
        [{"src_id": "source", "tgt_id": "target", "source_id": "chunk-1"}],
        QueryParam(),
        _quota_text_chunks_db(),
        entity_chunks=[],
        query="query",
        chunks_vdb=SimpleNamespace(),
        query_embedding=[1.0],
    )

    assert vector_picker.await_args.kwargs["num_of_chunks"] == 1
    weighted_picker.assert_not_called()
    assert [chunk["chunk_id"] for chunk in result] == ["chunk-1"]


async def test_zero_related_chunk_limit_remains_disabled(monkeypatch):
    vector_picker = AsyncMock(side_effect=_vector_picker)
    weighted_picker = Mock(return_value=[])
    monkeypatch.setattr("lightrag.operate.pick_by_vector_similarity", vector_picker)
    monkeypatch.setattr("lightrag.operate.pick_by_weighted_polling", weighted_picker)
    text_chunks_db = _quota_text_chunks_db()
    text_chunks_db.global_config["related_chunk_number"] = 0

    result = await _find_related_text_unit_from_entities(
        [{"entity_name": "entity", "source_id": "chunk-1"}],
        QueryParam(),
        text_chunks_db,
        SimpleNamespace(),
        query="query",
        chunks_vdb=SimpleNamespace(),
        query_embedding=[1.0],
    )

    assert vector_picker.await_args.kwargs["num_of_chunks"] == 0
    assert result == []


# ---------------------------------------------------------------------------
# Entity vs. relation chunk-selection budget parity
# ---------------------------------------------------------------------------

CHUNK_IDS = [f"chunk-{index}" for index in range(20)]
SOURCE_ID = GRAPH_FIELD_SEP.join(CHUNK_IDS)
RELATED_CHUNK_NUMBER = 2
DUPLICATE_GROUPS = 5


def _selection_text_chunks_db(
    pick_method: str, related_chunk_number: int = RELATED_CHUNK_NUMBER
) -> AsyncMock:
    text_chunks_db = AsyncMock()
    text_chunks_db.global_config = {
        "kg_chunk_pick_method": pick_method,
        "related_chunk_number": related_chunk_number,
    }
    text_chunks_db.get_by_ids.side_effect = lambda chunk_ids: [
        {"content": f"content-{chunk_id}"} for chunk_id in chunk_ids
    ]
    return text_chunks_db


def _duplicate_entities() -> list[dict]:
    """Entities that all carry the same chunks, so one group survives dedup."""
    return [
        {"entity_name": f"entity-{index}", "source_id": SOURCE_ID}
        for index in range(DUPLICATE_GROUPS)
    ]


def _duplicate_relations() -> list[dict]:
    """The relation-path twin of :func:`_duplicate_entities`."""
    return [
        {"src_tgt": (f"src-{index}", f"tgt-{index}"), "source_id": SOURCE_ID}
        for index in range(DUPLICATE_GROUPS)
    ]


async def _select_from_entities(text_chunks_db: AsyncMock, **kwargs) -> list[dict]:
    return await _find_related_text_unit_from_entities(
        _duplicate_entities(),
        QueryParam(),
        text_chunks_db,
        AsyncMock(),
        **kwargs,
    )


async def _select_from_relations(text_chunks_db: AsyncMock, **kwargs) -> list[dict]:
    return await _find_related_text_unit_from_relations(
        _duplicate_relations(),
        QueryParam(),
        text_chunks_db,
        entity_chunks=[],
        **kwargs,
    )


def _vector_call_args(picker: AsyncMock) -> tuple[int, int]:
    """Return the (budget, group count) a path handed to the vector picker."""
    return (
        picker.call_args.kwargs["num_of_chunks"],
        len(picker.call_args.kwargs["entity_info"]),
    )


@pytest.mark.parametrize(
    "select",
    [_select_from_entities, _select_from_relations],
    ids=["entities", "relations"],
)
async def test_weight_path_honours_the_configured_budget(select):
    """The user-visible symptom: phantom entity groups used to deliver 8, not 2.

    Both paths are pinned because the asymmetry is what makes this easy to
    reintroduce.
    """
    result = await select(_selection_text_chunks_db("WEIGHT"))

    assert [chunk["chunk_id"] for chunk in result] == CHUNK_IDS[:RELATED_CHUNK_NUMBER]


@pytest.mark.parametrize(
    "select",
    [_select_from_entities, _select_from_relations],
    ids=["entities", "relations"],
)
async def test_vector_budget_counts_only_surviving_groups(select):
    """``related_chunk_number=1`` pins both the dedup and the quota floor.

    With the phantom groups still counted the picker would get
    ``int(1 * 5 / 2) == 2`` chunks over 5 groups. With the groups dropped but
    without the floor added with ``_vector_chunk_quota``, ``int(1 * 1 / 2)``
    truncates to 0; ``pick_by_vector_similarity`` reads that as "no chunks" and
    the call site silently downgrades to WEIGHT. Only ``(1, 1)`` has both.
    """
    text_chunks_db = _selection_text_chunks_db("VECTOR", related_chunk_number=1)
    text_chunks_db.embedding_func = AsyncMock()

    with patch(
        "lightrag.operate.pick_by_vector_similarity",
        new=AsyncMock(return_value=[]),
    ) as vector_picker:
        await select(text_chunks_db, query="question", chunks_vdb=AsyncMock())

    assert _vector_call_args(vector_picker) == (1, 1)


@pytest.mark.parametrize(
    ("select", "query", "chunks_vdb"),
    [
        (_select_from_entities, "", AsyncMock()),
        (_select_from_relations, "question", None),
    ],
    ids=["entities_without_query", "relations_without_chunks_vdb"],
)
async def test_vector_without_prerequisites_falls_back_to_weight(
    select, query, chunks_vdb
):
    """VECTOR without a query or ``chunks_vdb`` used to skip both branches and return []."""
    text_chunks_db = _selection_text_chunks_db("VECTOR")
    text_chunks_db.embedding_func = AsyncMock()

    with patch(
        "lightrag.operate.pick_by_vector_similarity", new=AsyncMock()
    ) as vector_picker:
        result = await select(text_chunks_db, query=query, chunks_vdb=chunks_vdb)

    vector_picker.assert_not_awaited()
    assert [chunk["chunk_id"] for chunk in result] == CHUNK_IDS[:RELATED_CHUNK_NUMBER]
