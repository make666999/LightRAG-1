"""Query answer-cache partitioning contracts.

Covers four partitioning axes on ``naive_query`` and ``kg_query``:
1. ``conversation_history`` bypasses both sides of the answer cache (while the
   keywords cache stays shared) and retires pre-v2 answer-cache entries.
2. ``user_prompt_prefix`` and custom ``system_prompt`` partition the answer
   cache in place without invalidating pre-prefix entries when unconfigured,
   respect ``disable_user_prompt_prefix``, avoid leaking the prefix into cache
   metadata, and account for prefix tokens accurately in context budgets.
3. Keyword fallbacks partition the answer cache when triggered and keep the
   existing key otherwise, suppressing bare empty-keyword warnings on fallback.
4. KG chunk-selection settings (``related_chunk_number``,
   ``kg_chunk_pick_method``) partition the KG answer cache.
"""

import pytest

import lightrag.operate as operate
from lightrag.base import QueryContextResult, QueryParam
from lightrag.operate import kg_query, naive_query
from lightrag.utils import (
    Tokenizer,
    compute_args_hash,
    get_llm_cache_identity,
    serialize_llm_cache_identity,
)

pytestmark = pytest.mark.offline

QUERY = "who is Tesla?"
STALE = "STALE-HISTORY-CONDITIONED-ANSWER"
PREFIX_A = "Always answer in Chinese.\n\n"
PREFIX_B = "Always answer in French.\n\n"
USER_PROMPT = "Reply in one sentence."
HL = ["launch costs"]
LL = ["Tesla"]

HISTORY_A = [
    {"role": "user", "content": "we were discussing Tesla the carmaker"},
    {"role": "assistant", "content": "Yes, the automotive company."},
]
HISTORY_B = [
    {"role": "user", "content": "we were discussing Nikola Tesla the physicist"},
    {"role": "assistant", "content": "Yes, the inventor."},
]


class _FakeTokenizerImpl:
    """1-char-per-token tokenizer stub for exact budget delta assertions."""

    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


def _fake_tokenizer() -> Tokenizer:
    return Tokenizer("fake", _FakeTokenizerImpl())


class _FakeKVStorage:
    """In-memory KV storage double with query caching enabled."""

    def __init__(self):
        self.global_config = {"enable_llm_cache": True}
        self._store = {}

    async def get_by_id(self, key):
        return self._store.get(key)

    async def upsert(self, entries):
        self._store.update(entries)


class _FakeChunksVDB:
    """Single-chunk vector store double for naive_query runs."""

    cosine_better_than_threshold = 0.0

    async def query(self, *_args, **_kwargs):
        return [
            {
                "id": "chunk-1",
                "content": "Query cache partitioning test chunk.",
                "file_path": "test.md",
            }
        ]


class _FakeTextChunks:
    """Minimal text_chunks holder carrying global_config for chunk-selection tests."""

    def __init__(self, global_config):
        self.global_config = global_config


class _RecordingModel:
    """Counts calls and records histories and system prompts received."""

    def __init__(self):
        self.calls = 0
        self.histories = []
        self.system_prompts = []

    async def __call__(self, *_args, **kwargs):
        self.calls += 1
        self.histories.append(kwargs.get("history_messages"))
        self.system_prompts.append(kwargs.get("system_prompt"))
        return f"answer-{self.calls}"


def _query_global_config(
    llm_func, *, keyword_func=None, prefix=None, **extra
) -> dict:
    role_llm_funcs = {"query": llm_func}
    if keyword_func is not None:
        role_llm_funcs["keyword"] = keyword_func
    cfg = {
        "tokenizer": _fake_tokenizer(),
        "role_llm_funcs": role_llm_funcs,
        "addon_params": {"language": "en"},
        "min_rerank_score": 0.0,
        "max_total_tokens": 4096,
        **extra,
    }
    if prefix is not None:
        cfg["user_prompt_prefix"] = prefix
    return cfg


def _answer_cache_keys(cache: _FakeKVStorage) -> list[str]:
    return [key for key in cache._store if ":query:" in key]


def _legacy_v1_answer_cache_key(
    param: QueryParam,
    cfg: dict,
    *,
    keywords: tuple[str, str] | None = None,
) -> str:
    """Frozen snapshot of the pre-v2 answer-cache key composition.

    Deliberately duplicates the historical argument list instead of reusing
    production code: these tests prove entries written before v2 are never
    served again. Do NOT refresh this when new key fields are added.
    """
    args = [
        param.mode,
        QUERY,
        param.response_type,
        param.top_k,
        param.chunk_top_k,
        param.max_entity_tokens,
        param.max_relation_tokens,
        param.max_total_tokens,
    ]
    if keywords is not None:
        args.extend(keywords)
    args.extend(
        [
            param.user_prompt or "",
            param.enable_rerank,
            cfg.get("enable_content_headings", False),
            "\n<llm_identity>\n",
            serialize_llm_cache_identity(get_llm_cache_identity(cfg, "query")),
        ]
    )
    return f"{param.mode}:query:{compute_args_hash(*args)}"


def _preprefix_answer_cache_key(
    param: QueryParam,
    cfg: dict,
    *,
    keywords: tuple[str, str] | None = None,
    kg_total_budget: bool = True,
) -> str:
    """Frozen snapshot of the answer-cache key as composed BEFORE user_prompt_prefix.

    Refreshed deliberately in PR #3877 (KG chunk-selection settings) and PR #4221
    (unconditional ``kg_total_budget`` component on KG queries). Naive is untouched.
    """
    args = [
        "query-answer-cache-v2",
        param.mode,
        QUERY,
        param.response_type,
        param.top_k,
        param.chunk_top_k,
        param.max_entity_tokens,
        param.max_relation_tokens,
        param.max_total_tokens,
    ]
    if keywords is not None:
        args.extend(keywords)
    args.extend(
        [
            param.user_prompt or "",
            param.enable_rerank,
            cfg.get("enable_content_headings", False),
        ]
    )
    if keywords is not None:
        args.extend(
            [
                "\n<kg_chunk_selection>\n",
                cfg.get("related_chunk_number", 5),
                cfg.get("kg_chunk_pick_method", "VECTOR"),
            ]
        )
        if kg_total_budget:
            args.append("\n<kg_total_budget>\n")
    args.extend(
        [
            "\n<llm_identity>\n",
            serialize_llm_cache_identity(get_llm_cache_identity(cfg, "query")),
        ]
    )
    return f"{param.mode}:query:{compute_args_hash(*args)}"


@pytest.fixture
def stub_query_context(monkeypatch):
    """Skip retrieval: kg_query only forwards its storage args into this call."""

    async def _fake_build_query_context(*_args, **_kwargs):
        return QueryContextResult(context="KG CONTEXT", raw_data={})

    monkeypatch.setattr(
        "lightrag.operate._build_query_context", _fake_build_query_context
    )


def _naive_param(**overrides) -> QueryParam:
    return QueryParam(mode="naive", enable_rerank=False, **overrides)


def _kg_param(**overrides) -> QueryParam:
    # mode="local" plus preset ll_keywords short-circuits keyword extraction
    # without tripping the empty-hl_keywords warning that hybrid/mix would.
    return QueryParam(
        mode="local", enable_rerank=False, ll_keywords=["Tesla"], **overrides
    )


async def _run_naive(param, cfg, cache, system_prompt=None):
    return await naive_query(
        QUERY,
        _FakeChunksVDB(),
        param,
        cfg,
        hashing_kv=cache,
        system_prompt=system_prompt,
    )


async def _run_kg(param, cfg, cache, system_prompt=None):
    return await kg_query(
        QUERY,
        None,
        None,
        None,
        None,
        param,
        cfg,
        hashing_kv=cache,
        system_prompt=system_prompt,
    )


_QUERY_MODES = [
    pytest.param(_run_naive, _naive_param, id="naive"),
    pytest.param(_run_kg, _kg_param, id="kg"),
]


# ---------------------------------------------------------------------------
# 1. conversation_history bypass and policy version retirement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("runner,param", _QUERY_MODES)
async def test_history_turn_does_not_seed_cache_for_plain_turn(
    runner, param, stub_query_context
):
    """A history-conditioned answer must never become the shared cache entry."""
    cache = _FakeKVStorage()
    model = _RecordingModel()
    cfg = _query_global_config(model)

    first = await runner(param(conversation_history=HISTORY_A), cfg, cache)
    assert first.content == "answer-1"
    assert model.histories[0] == HISTORY_A
    assert cache._store == {}

    second = await runner(param(), cfg, cache)
    assert second.content == "answer-2"
    assert model.calls == 2
    assert model.histories[1] == []
    assert len(_answer_cache_keys(cache)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("runner,param", _QUERY_MODES)
async def test_history_turn_ignores_existing_plain_cache_entry(
    runner, param, stub_query_context
):
    """A multi-turn caller must not be served an answer that ignored its history."""
    cache = _FakeKVStorage()
    model = _RecordingModel()
    cfg = _query_global_config(model)

    first = await runner(param(), cfg, cache)
    assert first.content == "answer-1"
    assert len(_answer_cache_keys(cache)) == 1

    second = await runner(param(conversation_history=HISTORY_A), cfg, cache)
    assert second.content == "answer-2"
    assert model.calls == 2
    assert model.histories[1] == HISTORY_A
    assert len(_answer_cache_keys(cache)) == 1


@pytest.mark.asyncio
async def test_naive_distinct_histories_neither_share_nor_grow_the_cache():
    """Distinct histories get distinct answers and add zero cache entries."""
    cache = _FakeKVStorage()
    model = _RecordingModel()
    cfg = _query_global_config(model)

    first = await _run_naive(_naive_param(conversation_history=HISTORY_A), cfg, cache)
    second = await _run_naive(_naive_param(conversation_history=HISTORY_B), cfg, cache)

    assert (first.content, second.content) == ("answer-1", "answer-2")
    assert model.calls == 2
    assert cache._store == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "history_overrides",
    [
        pytest.param({}, id="unset"),
        pytest.param({"conversation_history": []}, id="empty-list"),
        pytest.param({"conversation_history": None}, id="none"),
    ],
)
async def test_naive_single_turn_query_still_hits_cache(history_overrides):
    """Over-fix guard: history-free turns (including explicit None) share one entry."""
    cache = _FakeKVStorage()
    model = _RecordingModel()
    cfg = _query_global_config(model)

    first = await _run_naive(_naive_param(**history_overrides), cfg, cache)
    second = await _run_naive(_naive_param(**history_overrides), cfg, cache)

    assert first.content == second.content == "answer-1"
    assert model.calls == 1
    assert len(_answer_cache_keys(cache)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runner,param,keywords",
    [
        pytest.param(_run_naive, _naive_param, None, id="naive"),
        pytest.param(_run_kg, _kg_param, ("", "Tesla"), id="kg"),
    ],
)
async def test_pre_v2_cache_entry_is_not_served(
    runner, param, keywords, stub_query_context
):
    """Entries written under the pre-v2 key scheme must be unreachable."""
    cache = _FakeKVStorage()
    model = _RecordingModel()
    cfg = _query_global_config(model)
    p = param()

    legacy_key = _legacy_v1_answer_cache_key(p, cfg, keywords=keywords)
    cache._store[legacy_key] = {"return": STALE, "create_time": 1}

    first = await runner(p, cfg, cache)
    assert first.content == "answer-1"
    assert model.calls == 1

    fresh_keys = [key for key in _answer_cache_keys(cache) if key != legacy_key]
    assert len(fresh_keys) == 1

    second = await runner(p, cfg, cache)
    assert second.content == "answer-1"
    assert model.calls == 1


@pytest.mark.asyncio
async def test_kg_keywords_cache_still_shared_across_histories(stub_query_context):
    """The keywords cache keeps the raw hashing_kv across history-bearing turns."""
    cache = _FakeKVStorage()
    model = _RecordingModel()
    keyword_calls = 0

    async def keyword_model(*_args, **_kwargs):
        nonlocal keyword_calls
        keyword_calls += 1
        return '{"high_level_keywords": ["physics"], "low_level_keywords": ["Tesla"]}'

    cfg = _query_global_config(model, keyword_func=keyword_model)

    for history in (HISTORY_A, HISTORY_B):
        await kg_query(
            QUERY,
            None,
            None,
            None,
            None,
            QueryParam(mode="local", enable_rerank=False, conversation_history=history),
            cfg,
            hashing_kv=cache,
        )

    keyword_keys = [key for key in cache._store if ":keywords:" in key]
    assert keyword_calls == 1
    assert len(keyword_keys) == 1
    assert keyword_keys[0].startswith("local:keywords:")
    assert _answer_cache_keys(cache) == []
    assert model.calls == 2


# ---------------------------------------------------------------------------
# 2. user_prompt_prefix & system_prompt partitioning, budgets, debug preview
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("runner,param", _QUERY_MODES)
@pytest.mark.parametrize(
    "user_prompt",
    [
        pytest.param(USER_PROMPT, id="with-user-prompt"),
        pytest.param(None, id="none-user-prompt"),
        pytest.param("", id="empty-user-prompt"),
    ],
)
async def test_prefix_reaches_system_prompt(
    runner, param, user_prompt, stub_query_context
):
    """A configured prefix reaches the model with or without a caller user_prompt."""
    cache = _FakeKVStorage()
    model = _RecordingModel()
    cfg = _query_global_config(model, prefix=PREFIX_A)

    await runner(param(user_prompt=user_prompt), cfg, cache)

    sys_prompt = model.system_prompts[0]
    assert PREFIX_A.strip() in sys_prompt
    if user_prompt:
        assert f"{PREFIX_A}{user_prompt}" in sys_prompt
    assert "n/a" not in sys_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runner,param,prompt_a,prompt_b",
    [
        pytest.param(
            _run_naive,
            _naive_param,
            "Policy A\n{content_data}\n{response_type}\n{user_prompt}",
            "Policy B\n{content_data}\n{response_type}\n{user_prompt}",
            id="naive",
        ),
        pytest.param(
            _run_kg,
            _kg_param,
            "Policy A\n{context_data}\n{response_type}\n{user_prompt}",
            "Policy B\n{context_data}\n{response_type}\n{user_prompt}",
            id="kg",
        ),
    ],
)
async def test_system_prompt_hits_on_same_and_partitions_on_change(
    runner, param, prompt_a, prompt_b, stub_query_context
):
    cache = _FakeKVStorage()
    model = _RecordingModel()
    cfg = _query_global_config(model)

    first = await runner(param(), cfg, cache, system_prompt=prompt_a)
    repeat = await runner(param(), cfg, cache, system_prompt=prompt_a)
    changed = await runner(param(), cfg, cache, system_prompt=prompt_b)

    assert first.content == repeat.content == "answer-1"
    assert changed.content == "answer-2"
    assert model.calls == 2
    assert len(_answer_cache_keys(cache)) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("runner,param", _QUERY_MODES)
async def test_prefix_hits_on_same_and_partitions_on_change(
    runner, param, stub_query_context
):
    cache = _FakeKVStorage()
    model = _RecordingModel()
    cfg_a = _query_global_config(model, prefix=PREFIX_A)
    cfg_b = _query_global_config(model, prefix=PREFIX_B)

    first = await runner(param(user_prompt=USER_PROMPT), cfg_a, cache)
    repeat = await runner(param(user_prompt=USER_PROMPT), cfg_a, cache)
    changed = await runner(param(user_prompt=USER_PROMPT), cfg_b, cache)

    assert first.content == repeat.content == "answer-1"
    assert changed.content == "answer-2"
    assert model.calls == 2
    assert len(_answer_cache_keys(cache)) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("runner,param", _QUERY_MODES)
async def test_prefix_and_user_prompt_keyed_on_composed_text(
    runner, param, stub_query_context
):
    """``'ab' + 'c'`` and ``'a' + 'bc'`` compose identical prompts and share an entry."""
    cache = _FakeKVStorage()
    model = _RecordingModel()

    first = await runner(
        param(user_prompt="c"), _query_global_config(model, prefix="ab"), cache
    )
    second = await runner(
        param(user_prompt="bc"), _query_global_config(model, prefix="a"), cache
    )

    assert first.content == second.content == "answer-1"
    assert model.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("runner,param", _QUERY_MODES)
async def test_disabled_prefix_excludes_prefix_and_shares_unprefixed_entry(
    runner, param, stub_query_context
):
    """``disable_user_prompt_prefix`` keeps the prefix out of the prompt and shares
    the cache entry with an unprefixed configuration.
    """
    cache = _FakeKVStorage()
    model = _RecordingModel()

    first = await runner(
        param(user_prompt=USER_PROMPT, disable_user_prompt_prefix=True),
        _query_global_config(model, prefix=PREFIX_A),
        cache,
    )
    sys_prompt = model.system_prompts[0]
    assert PREFIX_A.strip() not in sys_prompt
    assert USER_PROMPT in sys_prompt

    second = await runner(
        param(user_prompt=USER_PROMPT),
        _query_global_config(model),
        cache,
    )
    assert first.content == second.content == "answer-1"
    assert model.calls == 1
    assert len(_answer_cache_keys(cache)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runner,param,keywords,prefix,kg_total_budget,expect_hit",
    [
        pytest.param(
            _run_naive, _naive_param, None, None, True, True, id="naive-unprefixed-hits"
        ),
        pytest.param(
            _run_kg,
            _kg_param,
            ("", "Tesla"),
            None,
            True,
            True,
            id="kg-unprefixed-hits",
        ),
        pytest.param(
            _run_kg,
            _kg_param,
            ("", "Tesla"),
            None,
            False,
            False,
            id="kg-pre-total-budget-misses",
        ),
        pytest.param(
            _run_naive,
            _naive_param,
            None,
            PREFIX_A,
            True,
            False,
            id="naive-configured-prefix-misses",
        ),
        pytest.param(
            _run_kg,
            _kg_param,
            ("", "Tesla"),
            PREFIX_A,
            True,
            False,
            id="kg-configured-prefix-misses",
        ),
    ],
)
async def test_pre_prefix_and_pre_budget_cache_compatibility(
    runner, param, keywords, prefix, kg_total_budget, expect_hit, stub_query_context
):
    """Unconfigured prefix preserves pre-prefix hits; configured prefix and pre-budget KG keys miss."""
    cache = _FakeKVStorage()
    model = _RecordingModel()
    cfg = _query_global_config(model, prefix=prefix)
    p = param(user_prompt=USER_PROMPT)

    seed_key = _preprefix_answer_cache_key(
        p, cfg, keywords=keywords, kg_total_budget=kg_total_budget
    )
    cache._store[seed_key] = {"return": "SEEDED-ANSWER", "create_time": 1}

    result = await runner(p, cfg, cache)
    if expect_hit:
        assert result.content == "SEEDED-ANSWER"
        assert model.calls == 0
    else:
        assert result.content == "answer-1"
        assert model.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("runner,param", _QUERY_MODES)
async def test_cache_metadata_records_the_raw_request_user_prompt(
    runner, param, stub_query_context
):
    """``queryparam`` mirrors the request; the operator prefix must not leak into it."""
    cache = _FakeKVStorage()
    model = _RecordingModel()
    cfg = _query_global_config(model, prefix=PREFIX_A)

    await runner(param(user_prompt=USER_PROMPT), cfg, cache)

    entry = cache._store[_answer_cache_keys(cache)[0]]
    queryparam = entry["queryparam"]
    assert queryparam["user_prompt"] == USER_PROMPT
    assert PREFIX_A not in str(queryparam)


@pytest.mark.asyncio
async def test_prefix_token_budget_accounting(monkeypatch):
    """Context chunk budgets charge the prefix only when active and rendered,
    and ``only_need_context`` previews the exact same budget as a live query.
    """
    limits = []
    original = operate.process_chunks_unified

    async def _spy(**kwargs):
        limits.append(kwargs["chunk_token_limit"])
        return await original(**kwargs)

    monkeypatch.setattr(operate, "process_chunks_unified", _spy)

    entities = [{"entity": "Scrooge", "description": "A miser."}]
    relations = [{"src": "Scrooge", "tgt": "Marley", "description": "Partners."}]
    chunks = [
        {"chunk_id": f"c{i}", "content": f"Chunk {i}.", "file_path": "a.md"}
        for i in range(5)
    ]
    prefix = "P" * 3000  # 1 char == 1 token with _fake_tokenizer
    template_without_slot = (
        "Custom template. Context: {context_data} Type: {response_type}"
    )

    async def _budget(
        prefix_value,
        *,
        disable=False,
        system_prompt=None,
        only_need_context=False,
    ):
        cfg = {
            "tokenizer": _fake_tokenizer(),
            "max_total_tokens": 30000,
            "user_prompt_prefix": prefix_value,
        }
        param = QueryParam(
            mode="hybrid",
            enable_rerank=False,
            user_prompt=USER_PROMPT,
            disable_user_prompt_prefix=disable,
            only_need_context=only_need_context,
        )
        await operate._build_context_str(
            entities,
            relations,
            chunks,
            QUERY,
            param,
            cfg,
            system_prompt=system_prompt,
        )
        return limits[-1]

    baseline = await _budget("")
    with_prefix = await _budget(prefix)
    disabled = await _budget(prefix, disable=True)
    assert with_prefix == baseline - len(prefix)
    assert disabled == baseline

    custom_baseline = await _budget("", system_prompt=template_without_slot)
    custom_with_prefix = await _budget(prefix, system_prompt=template_without_slot)
    assert custom_with_prefix == custom_baseline
    assert custom_baseline != baseline

    assert await _budget(prefix, only_need_context=True) == with_prefix


@pytest.mark.asyncio
@pytest.mark.parametrize("runner,param", _QUERY_MODES)
@pytest.mark.parametrize("user_prompt", [USER_PROMPT, ""])
async def test_only_need_prompt_returns_the_composed_prefix(
    runner, param, user_prompt, stub_query_context
):
    """``only_need_prompt`` previews the exact composed prefix without calling the LLM."""
    cache = _FakeKVStorage()
    model = _RecordingModel()
    cfg = _query_global_config(model, prefix=PREFIX_A)

    result = await runner(
        param(user_prompt=user_prompt, only_need_prompt=True), cfg, cache
    )

    assert PREFIX_A.strip() in result.content
    assert "n/a" not in result.content
    if user_prompt:
        assert user_prompt in result.content
    assert model.calls == 0


# ---------------------------------------------------------------------------
# 3. Keyword fallback cache partitioning and warning suppression
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,hl,ll,should_differ",
    [
        pytest.param("local", HL, [], True, id="local-ll-empty-fallback"),
        pytest.param("hybrid", HL, [], True, id="hybrid-ll-empty-fallback"),
        pytest.param("mix", HL, [], True, id="mix-ll-empty-fallback"),
        pytest.param("global", [], LL, True, id="global-hl-empty-fallback"),
        pytest.param("hybrid", HL, LL, False, id="hybrid-no-fallback"),
        pytest.param("local", [], LL, False, id="local-hl-empty-no-fallback"),
        pytest.param("hybrid", [], LL, False, id="hybrid-hl-empty-no-fallback"),
        pytest.param("global", HL, [], False, id="global-ll-empty-no-fallback"),
    ],
)
async def test_keyword_fallback_cache_key_partitioning(
    mode, hl, ll, should_differ, stub_query_context
):
    """A keyword fallback partitions the answer cache; non-fallback modes keep the old key."""
    cache = _FakeKVStorage()
    cfg = _query_global_config(_RecordingModel())
    param = QueryParam(mode=mode, enable_rerank=False, hl_keywords=hl, ll_keywords=ll)

    await _run_kg(param, cfg, cache)

    (written,) = _answer_cache_keys(cache)
    old = _preprefix_answer_cache_key(
        param,
        cfg,
        keywords=(", ".join(param.hl_keywords), ", ".join(param.ll_keywords)),
    )
    assert (written != old) is should_differ


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,hl,ll,expected",
    [
        pytest.param("local", HL, [], [], id="local-ll-empty-fallback"),
        pytest.param("global", [], LL, [], id="global-hl-empty-fallback"),
        pytest.param(
            "hybrid", [], LL, ["high_level_keywords is empty"], id="hybrid-hl-empty"
        ),
        pytest.param("local", HL, LL, [], id="both-present"),
    ],
)
async def test_bare_empty_keyword_warning_only_without_a_fallback(
    mode, hl, ll, expected, stub_query_context, lightrag_log_records
):
    cache = _FakeKVStorage()
    cfg = _query_global_config(_RecordingModel())
    param = QueryParam(mode=mode, enable_rerank=False, hl_keywords=hl, ll_keywords=ll)

    await _run_kg(param, cfg, cache)

    bare = [
        record.getMessage()
        for record in lightrag_log_records
        if record.getMessage().endswith("_level_keywords is empty")
    ]
    assert bare == expected


# ---------------------------------------------------------------------------
# 4. KG chunk-selection cache partitioning
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "setting,first_value,second_value",
    [
        pytest.param("related_chunk_number", 1, 5, id="related-chunk-number"),
        pytest.param("kg_chunk_pick_method", "WEIGHT", "VECTOR", id="pick-method"),
    ],
)
async def test_kg_chunk_selection_cache_partitioning(
    setting, first_value, second_value, monkeypatch
):
    """Changing KG chunk-selection settings partitions the answer cache; same settings hit."""

    async def fake_keywords(*_args, **_kwargs):
        return "", "topic"

    async def fake_context(*args, **_kwargs):
        config = args[6].global_config
        context = (
            f"chunks={config['related_chunk_number']};"
            f"method={config['kg_chunk_pick_method']}"
        )
        return QueryContextResult(context=context, raw_data={})

    monkeypatch.setattr("lightrag.operate.get_keywords_from_query", fake_keywords)
    monkeypatch.setattr("lightrag.operate._build_query_context", fake_context)

    model = _RecordingModel()
    cfg = _query_global_config(
        model, related_chunk_number=1, kg_chunk_pick_method="WEIGHT"
    )
    cache = _FakeKVStorage()
    param = QueryParam(mode="local", enable_rerank=False, ll_keywords=["topic"])

    cfg[setting] = first_value
    first = await kg_query(
        "query", None, None, None, _FakeTextChunks(cfg), param, cfg, hashing_kv=cache
    )
    repeat = await kg_query(
        "query", None, None, None, _FakeTextChunks(cfg), param, cfg, hashing_kv=cache
    )
    assert first.content == repeat.content == "answer-1"
    assert model.calls == 1
    assert len(_answer_cache_keys(cache)) == 1

    cfg[setting] = second_value
    second = await kg_query(
        "query", None, None, None, _FakeTextChunks(cfg), param, cfg, hashing_kv=cache
    )
    assert second.content == "answer-2"
    assert model.calls == 2
    assert len(_answer_cache_keys(cache)) == 2
