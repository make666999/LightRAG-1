"""Unit and integration tests for LLM response caching, truncation guards, and #3833 ordering."""

from unittest.mock import AsyncMock

import pytest

from lightrag.base import QueryParam
from lightrag.exceptions import EmptyTruncatedResponseError
from lightrag.operate import naive_query
from lightrag.utils import (
    Tokenizer,
    TruncatedResponse,
    is_truncated_response,
    use_llm_func_with_cache,
)

pytestmark = pytest.mark.offline


class _FakeTokenizerImpl:
    """1-char-per-token tokenizer stub for deterministic cache tests."""

    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


def _fake_tokenizer() -> Tokenizer:
    return Tokenizer("fake", _FakeTokenizerImpl())


class _FakeKVStorage:
    """In-memory KV storage double with cache enabled for extract and query."""

    def __init__(self):
        self.global_config = {
            "enable_llm_cache": True,
            "enable_llm_cache_for_entity_extract": True,
        }
        self._store = {}

    async def get_by_id(self, key):
        return self._store.get(key)

    async def upsert(self, entries):
        self._store.update(entries)


class _FakeChunksVDB:
    """Single-chunk vector store double for naive_query cache tests."""

    cosine_better_than_threshold = 0.0

    async def query(self, *_args, **_kwargs):
        return [
            {
                "id": "chunk-1",
                "content": "LightRAG cache test chunk.",
                "file_path": "test.md",
            }
        ]


class _FakeChunkKV:
    """text_chunks stand-in for the reference-before-row path (#3833).

    ``writes`` is shared with the cache double so a test can assert the ORDER
    of the two storage writes rather than merely that both happened.
    """

    def __init__(self, rows: dict | None = None, writes: list | None = None):
        self.data = dict(rows or {})
        self.writes = writes if writes is not None else []
        self.fail_on_upsert = False
        self.fail_on_read = False

    async def get_by_id(self, key):
        if self.fail_on_read:
            raise RuntimeError("text_chunks read is down")
        return self.data.get(key)

    async def upsert(self, rows: dict):
        if self.fail_on_upsert:
            raise RuntimeError("text_chunks write is down")
        self.data.update(rows)
        self.writes.append("chunk")

    async def index_done_callback(self):
        return None


class _RecordingCache(_FakeKVStorage):
    """Cache double that records its writes into a shared order log."""

    def __init__(self, writes: list):
        super().__init__()
        self.writes = writes

    async def upsert(self, entries):
        self._store.update(entries)
        self.writes.append("cache")


def _attached(chunks: _FakeChunkKV, chunk_id: str = "chunk-1") -> list[str]:
    return list((chunks.data.get(chunk_id) or {}).get("llm_cache_list") or [])


def _query_global_config(llm_func, *, model: str | None = None) -> dict:
    cfg = {
        "tokenizer": _fake_tokenizer(),
        "role_llm_funcs": {"query": llm_func},
        "min_rerank_score": 0.0,
        "max_total_tokens": 4096,
    }
    if model is not None:
        cfg["llm_cache_identities"] = {
            "query": {
                "role": "query",
                "binding": "openai",
                "model": model,
                "host": "https://api.example.com/v1",
            }
        }
    return cfg


# ---------------------------------------------------------------------------
# use_llm_func_with_cache: partitioning, response_format, truncation guards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first_kwargs,second_kwargs,responses",
    [
        pytest.param(
            {},
            {"response_format": {"type": "json_object"}},
            ["plain-text", '{"answer":"json"}'],
            id="response-format",
        ),
        pytest.param(
            {
                "llm_cache_identity": {
                    "role": "query",
                    "binding": "openai",
                    "model": "model-a",
                    "host": "https://api.example.com/v1",
                }
            },
            {
                "llm_cache_identity": {
                    "role": "query",
                    "binding": "openai",
                    "model": "model-b",
                    "host": "https://api.example.com/v1",
                }
            },
            ["model-a", "model-b"],
            id="llm-identity",
        ),
    ],
)
async def test_use_llm_func_with_cache_partitions_keys(
    first_kwargs, second_kwargs, responses
):
    cache = _FakeKVStorage()
    llm_func = AsyncMock(side_effect=responses)

    first_result, _ = await use_llm_func_with_cache(
        "same prompt",
        llm_func,
        llm_response_cache=cache,
        **first_kwargs,
    )
    second_result, _ = await use_llm_func_with_cache(
        "same prompt",
        llm_func,
        llm_response_cache=cache,
        **second_kwargs,
    )

    assert first_result == responses[0]
    assert second_result == responses[1]
    assert llm_func.await_count == 2
    assert len(cache._store) == 2


@pytest.mark.asyncio
async def test_use_llm_func_with_cache_skips_and_does_not_reuse_truncated_response():
    """A token-limit-truncated response is returned for salvage, never persisted,
    and re-invokes the LLM on the next call until a complete response is cached.
    """
    cache = _FakeKVStorage()
    llm_func = AsyncMock(
        side_effect=[
            TruncatedResponse('{"entities":[{"name":"Ali'),
            '{"entities":[{"name":"Alice"}]}',
        ]
    )

    first, _ = await use_llm_func_with_cache(
        "extract prompt",
        llm_func,
        llm_response_cache=cache,
        response_format={"type": "json_object"},
    )
    assert first == '{"entities":[{"name":"Ali'
    assert is_truncated_response(first)
    assert cache._store == {}
    llm_func.assert_awaited_once()

    second, _ = await use_llm_func_with_cache(
        "extract prompt",
        llm_func,
        llm_response_cache=cache,
        response_format={"type": "json_object"},
    )
    assert second == '{"entities":[{"name":"Alice"}]}'
    assert not is_truncated_response(second)
    assert llm_func.await_count == 2
    assert len(cache._store) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_output,expected",
    [
        pytest.param(
            "<think>reasoning</think>Partial result",
            "Partial result",
            id="plain-partial",
        ),
        pytest.param(
            '<think>reasoning</think>{"entities":[{"name',
            '{"entities":[{"name',
            id="json-salvage",
        ),
    ],
)
async def test_truncation_marker_survives_think_removal_without_cache(
    raw_output, expected
):
    """Callers must observe truncation even without an extraction cache."""
    llm_func = AsyncMock(return_value=TruncatedResponse(raw_output))

    result, _ = await use_llm_func_with_cache(
        "extract prompt",
        llm_func,
        llm_response_cache=None,
    )

    assert result == expected
    assert is_truncated_response(result)


@pytest.mark.asyncio
async def test_use_llm_func_with_cache_rejects_json_schema_response_format():
    llm_func = AsyncMock()

    with pytest.raises(ValueError, match="json_schema"):
        await use_llm_func_with_cache(
            "same prompt",
            llm_func,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer_payload",
                    "schema": {"type": "object"},
                },
            },
        )

    llm_func.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("cache_enabled", [False, True])
async def test_truncated_response_emptied_by_think_removal_is_rejected(cache_enabled):
    """A thinking model that exhausts its budget inside ``<think>...</think>``
    becomes visibly empty only after think-tag removal and must raise rather
    than index an empty graph.
    """
    llm_func = AsyncMock(
        return_value=TruncatedResponse("<think>let me carefully consider</think>")
    )

    with pytest.raises(EmptyTruncatedResponseError) as excinfo:
        await use_llm_func_with_cache(
            "extract prompt",
            llm_func,
            llm_response_cache=_FakeKVStorage() if cache_enabled else None,
            chunk_id="chunk-001",
        )

    message = str(excinfo.value)
    assert "Received empty extract content after think-tag removal" in message
    assert "chunk_id=chunk-001" in message
    assert "reasoning_content_len=40" in message
    assert "budget consumed by reasoning" in message
    assert "output token limit" in message


@pytest.mark.asyncio
async def test_an_untruncated_empty_response_is_still_returned():
    """Only the token-limit case escalates; normal empty/think-only output stays empty."""
    llm_func = AsyncMock(return_value="<think>done thinking</think>")

    result, _ = await use_llm_func_with_cache(
        "extract prompt", llm_func, llm_response_cache=None
    )

    assert result == ""


# ---------------------------------------------------------------------------
# Reference-before-row ordering (#3833)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_cache_row_is_written_only_after_its_reference():
    """The chunk llm_cache_list reference must be durable before the cache row exists."""
    writes: list[str] = []
    chunks = _FakeChunkKV({"chunk-1": {"content": "c"}}, writes=writes)
    cache = _RecordingCache(writes)
    llm_func = AsyncMock(return_value="extracted")

    result, _ = await use_llm_func_with_cache(
        "extract prompt",
        llm_func,
        llm_response_cache=cache,
        chunk_id="chunk-1",
        text_chunks_storage=chunks,
    )

    assert result == "extracted"
    assert writes == ["chunk", "cache"], writes
    assert _attached(chunks) == list(cache._store)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rows,fail_read,fail_upsert",
    [
        pytest.param({"chunk-1": {"content": "c"}}, False, True, id="upsert-fails"),
        pytest.param({"chunk-1": {"content": "c"}}, True, False, id="read-fails"),
        pytest.param({}, False, False, id="missing-chunk-row"),
    ],
)
async def test_cache_write_is_skipped_when_chunk_reference_cannot_be_recorded(
    rows, fail_read, fail_upsert
):
    """No reference, no row: unrecordable keys must never leave orphan cache rows."""
    chunks = _FakeChunkKV(rows)
    chunks.fail_on_read = fail_read
    chunks.fail_on_upsert = fail_upsert
    cache = _FakeKVStorage()

    result, _ = await use_llm_func_with_cache(
        "extract prompt",
        AsyncMock(return_value="extracted"),
        llm_response_cache=cache,
        chunk_id="chunk-1",
        text_chunks_storage=chunks,
    )

    assert result == "extracted"
    assert cache._store == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "llm_output,expected",
    [
        pytest.param("", "", id="empty-response"),
        pytest.param(TruncatedResponse("partial"), "partial", id="truncated-response"),
    ],
)
async def test_uncacheable_response_attaches_no_reference(llm_output, expected):
    """Responses that are not persisted to cache must not attach dangling chunk refs."""
    chunks = _FakeChunkKV({"chunk-1": {"content": "c"}})
    cache = _FakeKVStorage()

    result, _ = await use_llm_func_with_cache(
        "extract prompt",
        AsyncMock(return_value=llm_output),
        llm_response_cache=cache,
        chunk_id="chunk-1",
        text_chunks_storage=chunks,
    )

    assert result == expected
    assert cache._store == {}
    assert _attached(chunks) == []


@pytest.mark.asyncio
async def test_a_cache_hit_reattaches_a_wiped_reference():
    """Re-ingest self-heal: a cache hit restores references wiped by stage-1 chunk overwrite."""
    writes: list[str] = []
    chunks = _FakeChunkKV({"chunk-1": {"content": "c", "llm_cache_list": []}}, writes)
    cache = _RecordingCache(writes)
    llm_func = AsyncMock(return_value="extracted")

    await use_llm_func_with_cache(
        "extract prompt",
        llm_func,
        llm_response_cache=cache,
        chunk_id="chunk-1",
        text_chunks_storage=chunks,
    )
    cache_keys = list(cache._store)
    chunks.data["chunk-1"] = {"content": "c", "llm_cache_list": []}

    await use_llm_func_with_cache(
        "extract prompt",
        llm_func,
        llm_response_cache=cache,
        chunk_id="chunk-1",
        text_chunks_storage=chunks,
    )

    assert llm_func.await_count == 1, "second call must be served from cache"
    assert _attached(chunks) == cache_keys


@pytest.mark.asyncio
async def test_callers_without_a_chunk_storage_keep_the_legacy_order():
    """Callers without an owning text_chunks_storage keep collecting cache keys."""
    cache = _FakeKVStorage()
    collector: list[str] = []

    result, _ = await use_llm_func_with_cache(
        "smartheading prompt",
        AsyncMock(return_value="judged"),
        llm_response_cache=cache,
        cache_type="smartheading",
        chunk_id="chunk-1",
        cache_keys_collector=collector,
    )

    assert result == "judged"
    assert list(cache._store) == collector


# ---------------------------------------------------------------------------
# naive_query cache identity and truncation guards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_naive_query_partitions_query_cache_by_llm_identity():
    cache = _FakeKVStorage()
    chunks_vdb = _FakeChunksVDB()
    calls = 0

    async def query_model(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return f"answer-{calls}"

    param = QueryParam(mode="naive", enable_rerank=False)

    first = await naive_query(
        "same query",
        chunks_vdb,
        param,
        _query_global_config(query_model, model="model-a"),
        hashing_kv=cache,
    )
    second = await naive_query(
        "same query",
        chunks_vdb,
        param,
        _query_global_config(query_model, model="model-b"),
        hashing_kv=cache,
    )

    assert first.content == "answer-1"
    assert second.content == "answer-2"
    assert calls == 2
    assert len(cache._store) == 2


@pytest.mark.asyncio
async def test_naive_query_truncated_answer_not_cached():
    """A token-limit-truncated answer is returned but never enters the query cache."""
    cache = _FakeKVStorage()
    chunks_vdb = _FakeChunksVDB()
    responses = [
        TruncatedResponse("partial ans"),
        "complete answer",
    ]
    calls = 0

    async def query_model(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return responses[calls - 1]

    param = QueryParam(mode="naive", enable_rerank=False)

    first = await naive_query(
        "same query",
        chunks_vdb,
        param,
        _query_global_config(query_model),
        hashing_kv=cache,
    )
    assert first.content == "partial ans"
    assert cache._store == {}

    second = await naive_query(
        "same query",
        chunks_vdb,
        param,
        _query_global_config(query_model),
        hashing_kv=cache,
    )
    assert second.content == "complete answer"
    assert calls == 2
    assert len(cache._store) == 1
