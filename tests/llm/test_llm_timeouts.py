"""Timeout propagation across LLM, embedding, rerank clients, wrapper forwarding, and concurrency queues."""

import asyncio
import base64
import contextlib
import functools
import importlib
from types import SimpleNamespace

import numpy as np
import openai
import pytest
from aiohttp import web
from tenacity import wait_none

from lightrag import LightRAG
from lightrag.llm_roles import _ROLE_LLM_TIMEOUT, apply_llm_timeout
from lightrag.utils import (
    EmbeddingFunc,
    Tokenizer,
    bind_embedding_timeout,
    bind_rerank_timeout,
    embedding_timeout,
    priority_limit_async_func_call,
    provider_timeout,
    rerank_timeout,
)

pytestmark = pytest.mark.offline

_VECTOR = [0.25, 0.5, 0.75, 1.0]
_OMITTED = "omitted"
_SLOW_DELAY = 0.05

_CHAT = {
    "id": "x",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
_ANTHROPIC = {
    "id": "x",
    "type": "message",
    "role": "assistant",
    "model": "m",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 1, "output_tokens": 1},
}
_OLLAMA = {
    "model": "m",
    "created_at": "2026-01-01T00:00:00Z",
    "message": {"role": "assistant", "content": "ok"},
    "done": True,
}


class _Recorded(Exception):
    """Raised by a recording fake once it has seen the client's arguments."""


@contextlib.asynccontextmanager
async def _http_stub(*, first_delay: float = 0.0, every_delay: float = 0.0):
    """Loopback stub answering all bundled LLM, embedding, and rerank HTTP bindings."""
    hits: list[str] = []
    released = asyncio.Event()

    async def answer(request: web.Request) -> web.StreamResponse:
        hits.append(request.path)
        if every_delay:
            await asyncio.sleep(every_delay)
        elif len(hits) == 1 and first_delay:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(released.wait(), first_delay)
        path = request.path
        if path.endswith("/api/embed"):
            return web.json_response({"model": "m", "embeddings": [_VECTOR]})
        if path.endswith("/lollms_embed"):
            return web.json_response({"vector": _VECTOR})
        if path.endswith("/rerank"):
            return web.json_response({"results": [{"index": 0, "relevance_score": 1}]})
        if path.endswith("/jina/embeddings"):
            encoded = base64.b64encode(np.array(_VECTOR, np.float32).tobytes())
            return web.json_response({"data": [{"embedding": encoded.decode()}]})
        if path.endswith("/messages"):
            return web.json_response(_ANTHROPIC)
        if path.endswith("/api/chat"):
            return web.json_response(_OLLAMA)
        if path.endswith("/lollms_generate"):
            return web.Response(text="ok")
        if path.endswith("/embeddings"):
            return web.json_response(
                {
                    "object": "list",
                    "model": "m",
                    "data": [
                        {"object": "embedding", "index": 0, "embedding": _VECTOR}
                    ],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
            )
        return web.json_response(_CHAT)

    app = web.Application()
    app.router.add_post("/{tail:.*}", answer)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", hits
    finally:
        released.set()
        await runner.cleanup()


def _record_calls(monkeypatch, owner, name, calls: list[dict]):
    real = getattr(owner, name)

    def wrapper(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, name, wrapper)


def _session_timeout(kwargs: dict):
    timeout = kwargs.get("timeout")
    return _OMITTED if timeout is None else timeout.total


# --------------------------------------------------------------------------- #
# Section 1: Embedding & rerank client timeouts
# --------------------------------------------------------------------------- #


async def _probe_openai(monkeypatch, url, extra):
    from lightrag.llm import openai as module

    calls: list[dict] = []
    _record_calls(monkeypatch, module, "AsyncOpenAI", calls)
    result = await module.openai_embed.func(["a"], base_url=url, api_key="k", **extra)
    assert result.shape == (1, len(_VECTOR))
    assert len(calls) == 1
    return calls[0].get("timeout", _OMITTED)


async def _probe_azure(monkeypatch, url, extra):
    from lightrag.llm import openai as module

    calls: list[dict] = []
    _record_calls(monkeypatch, openai, "AsyncAzureOpenAI", calls)
    result = await module.azure_openai_embed.func(
        ["a"], model="d", base_url=url, api_key="k", api_version="v", **extra
    )
    assert result.shape == (1, len(_VECTOR))
    assert len(calls) == 1
    return calls[0].get("timeout", _OMITTED)


async def _probe_nvidia(monkeypatch, url, extra):
    from lightrag.llm import nvidia_openai as module

    calls: list[dict] = []
    _record_calls(monkeypatch, module, "AsyncOpenAI", calls)
    result = await module.nvidia_openai_embed.func(
        ["a"], base_url=url, api_key="k", **extra
    )
    assert result.shape == (1, len(_VECTOR))
    assert len(calls) == 1
    return calls[0].get("timeout", _OMITTED)


async def _probe_ollama(monkeypatch, url, extra):
    from lightrag.llm import ollama as module

    calls: list[dict] = []
    _record_calls(monkeypatch, module.ollama, "AsyncClient", calls)
    result = await module.ollama_embed.func(["a"], host=url, **extra)
    assert result.shape == (1, len(_VECTOR))
    assert len(calls) == 1
    timeout = calls[0]["timeout"]
    return _OMITTED if timeout is None else timeout


async def _probe_jina(monkeypatch, url, extra):
    from lightrag.llm import jina as module

    calls: list[dict] = []
    _record_calls(monkeypatch, module.aiohttp, "ClientSession", calls)
    result = await module.jina_embed.func(
        ["a"],
        embedding_dim=len(_VECTOR),
        base_url=f"{url}/jina/embeddings",
        api_key="k",
        **extra,
    )
    assert result.shape == (1, len(_VECTOR))
    assert len(calls) == 1
    return _session_timeout(calls[0])


async def _probe_lollms(monkeypatch, url, extra):
    from lightrag.llm import lollms as module

    calls: list[dict] = []
    _record_calls(monkeypatch, module.aiohttp, "ClientSession", calls)
    result = await module.lollms_embed.func(["a"], base_url=url, **extra)
    assert result.shape == (1, len(_VECTOR))
    assert len(calls) == 1
    return _session_timeout(calls[0])


async def _probe_gemini(monkeypatch, url, extra):
    from lightrag.llm import gemini as module

    calls: list = []

    def fake_client(api_key, base_url, timeout=None):
        calls.append(timeout)
        raise _Recorded

    monkeypatch.setattr(module, "_get_gemini_client", fake_client)
    with pytest.raises(_Recorded):
        await module.gemini_embed.func(["a"], api_key="k", **extra)
    assert len(calls) == 1
    return _OMITTED if calls[0] is None else calls[0] / 1000


async def _probe_bedrock(monkeypatch, url, extra):
    from lightrag.llm import bedrock as module

    calls: list[dict] = []
    real = module._bedrock_client_kwargs

    def recording(*args, **kwargs):
        calls.append(real(*args, **kwargs))
        raise _Recorded

    monkeypatch.setattr(module, "_bedrock_client_kwargs", recording)
    with pytest.raises(_Recorded):
        await module.bedrock_embed.func(["a"], aws_region="us-east-1", **extra)
    assert len(calls) == 1
    config = calls[0].get("config")
    if config is None:
        return _OMITTED
    assert config.connect_timeout == config.read_timeout
    return config.read_timeout


async def _probe_voyageai(monkeypatch, url, extra):
    from lightrag.llm import voyageai as module

    calls: list[dict] = []

    def fake_client(**kwargs):
        calls.append(kwargs)
        raise _Recorded

    monkeypatch.setattr(module.voyageai, "AsyncClient", fake_client)
    with pytest.raises(_Recorded):
        await module.voyageai_embed.func(["a"], api_key="k", **extra)
    assert len(calls) == 1
    return calls[0].get("timeout", _OMITTED)


async def _probe_zhipu(monkeypatch, url, extra):
    from lightrag.llm import zhipu as module

    pytest.importorskip("zai")
    calls: list[dict] = []

    class _Embeddings:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(data=[SimpleNamespace(embedding=_VECTOR)])

    @contextlib.contextmanager
    def fake_build_client(client_cls, api_key):
        yield SimpleNamespace(embeddings=_Embeddings())

    monkeypatch.setattr(module, "_build_client", fake_build_client)
    result = await module.zhipu_embedding.func(["a"], api_key="k", **extra)
    assert result.shape == (1, len(_VECTOR))
    assert len(calls) == 1
    return calls[0].get("timeout", _OMITTED)


async def _probe_rerank(monkeypatch, url, extra):
    from lightrag import rerank as module

    calls: list[dict] = []
    _record_calls(monkeypatch, module.aiohttp, "ClientSession", calls)
    result = await module.cohere_rerank(
        query="q", documents=["a"], api_key="k", base_url=f"{url}/rerank", **extra
    )
    assert [r["index"] for r in result] == [0]
    assert len(calls) == 1
    return _session_timeout(calls[0])


_EMBEDDING_PROBES = [
    pytest.param(_probe_openai, id="openai_embed"),
    pytest.param(_probe_azure, id="azure_openai_embed"),
    pytest.param(_probe_nvidia, id="nvidia_openai_embed"),
    pytest.param(_probe_ollama, id="ollama_embed"),
    pytest.param(_probe_jina, id="jina_embed"),
    pytest.param(_probe_lollms, id="lollms_embed"),
    pytest.param(_probe_gemini, id="gemini_embed"),
    pytest.param(_probe_bedrock, id="bedrock_embed"),
    pytest.param(_probe_voyageai, id="voyageai_embed"),
    pytest.param(_probe_zhipu, id="zhipu_embedding"),
]

_EMBED_CASES = [
    pytest.param(7, {}, 7, id="configured-positive"),
    pytest.param(0, {}, _OMITTED, id="configured-zero-sdk-default"),
    pytest.param(-1, {}, _OMITTED, id="configured-negative-sdk-default"),
    pytest.param("unbound", {}, _OMITTED, id="outside-lightrag-sdk-default"),
    pytest.param(7, {"timeout": 3}, 3, id="explicit-argument-wins"),
    pytest.param(7, {"timeout": 0}, _OMITTED, id="explicit-zero-sdk-default"),
]
_WIRING_CASE_IDS = (
    "configured-positive",
    "configured-zero-sdk-default",
    "explicit-argument-wins",
)
_EMBEDDING_MATRIX = [
    pytest.param(probe.values[0], *case.values, id=f"{case.id}-{probe.id}")
    for probe in _EMBEDDING_PROBES
    for case in _EMBED_CASES
    if probe.id == "openai_embed" or case.id in _WIRING_CASE_IDS
]


async def _run_probe(probe, bind, monkeypatch, configured, extra):
    async with _http_stub() as (url, _hits):

        async def call():
            return await probe(monkeypatch, url, extra)

        if configured != "unbound":
            call = bind(call, configured)
        return await call()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("probe", "configured", "extra", "expected"), _EMBEDDING_MATRIX
)
async def test_embedding_binding_hands_timeout_to_client(
    monkeypatch, probe, configured, extra, expected
):
    assert (
        await _run_probe(probe, bind_embedding_timeout, monkeypatch, configured, extra)
        == expected
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("configured", "extra", "expected"), _EMBED_CASES)
async def test_rerank_binding_hands_timeout_to_client(
    monkeypatch, configured, extra, expected
):
    assert (
        await _run_probe(
            _probe_rerank, bind_rerank_timeout, monkeypatch, configured, extra
        )
        == expected
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "probe", [_probe_openai, _probe_azure], ids=["openai", "azure"]
)
@pytest.mark.parametrize(
    ("configured", "extra", "expected"),
    [
        pytest.param(7, {}, 5, id="beats-lightrag-timeout"),
        pytest.param(0, {}, 5, id="applies-when-lightrag-sets-none"),
        pytest.param(7, {"timeout": 3}, 3, id="positive-argument-beats-it"),
        pytest.param(7, {"timeout": 0}, 5, id="non-positive-argument-keeps-it"),
    ],
)
async def test_openai_client_configs_timeout_precedence(
    monkeypatch, probe, configured, extra, expected
):
    observed = await _run_probe(
        probe,
        bind_embedding_timeout,
        monkeypatch,
        configured,
        {"client_configs": {"timeout": 5}, **extra},
    )
    assert observed == expected


@pytest.mark.asyncio
async def test_embedding_and_rerank_timeouts_do_not_leak_into_each_other():
    async def read_both():
        return embedding_timeout(), rerank_timeout()

    assert await bind_embedding_timeout(read_both, 7)() == (7, None)
    assert await bind_rerank_timeout(read_both, 9)() == (None, 9)
    assert await read_both() == (None, None)


class _CharTokenizer:
    """Character-level tokenizer stand-in for lightweight LightRAG instantiation."""

    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _dummy_llm(*args, **kwargs) -> str:
    return "ok"


def _make_rag(tmp_path, embedding_func, rerank_func, **kwargs) -> LightRAG:
    return LightRAG(
        working_dir=str(tmp_path / "rag"),
        workspace="timeout-forwarding",
        llm_model_func=_dummy_llm,
        embedding_func=embedding_func,
        rerank_model_func=rerank_func,
        tokenizer=Tokenizer("char", _CharTokenizer()),
        **kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("embedding", "rerank", "expected"),
    [
        pytest.param(11, 13, (11, 13), id="positive"),
        pytest.param(0, 0, (None, None), id="zero-sdk-default"),
    ],
)
async def test_lightrag_binds_its_timeouts_around_each_call(
    tmp_path, embedding, rerank, expected
):
    seen: dict[str, object] = {}

    async def embed(texts, max_token_size=None):
        seen["max_token_size"] = max_token_size
        seen["embedding"] = embedding_timeout()
        return np.ones((len(texts), 4))

    async def rerank_func(query, documents, top_n=None):
        seen["rerank"] = rerank_timeout()
        return [{"index": 0, "relevance_score": 1.0}]

    rag = _make_rag(
        tmp_path,
        EmbeddingFunc(embedding_dim=4, max_token_size=123, func=embed),
        rerank_func,
        default_embedding_timeout=embedding,
        default_rerank_timeout=rerank,
    )

    await rag.embedding_func(["a"])
    await rag.rerank_model_func(query="q", documents=["a"])

    assert (seen["embedding"], seen["rerank"]) == expected
    assert seen["max_token_size"] == 123
    assert (embedding_timeout(), rerank_timeout()) == (None, None)


@pytest.mark.asyncio
async def test_lightrag_timeout_reaches_a_partial_of_a_bundled_binding(
    tmp_path, monkeypatch
):
    from lightrag.llm import openai as module

    calls: list[dict] = []
    _record_calls(monkeypatch, module, "AsyncOpenAI", calls)

    async with _http_stub() as (url, _hits):
        rag = _make_rag(
            tmp_path,
            EmbeddingFunc(
                embedding_dim=len(_VECTOR),
                max_token_size=8192,
                func=functools.partial(
                    module.openai_embed.func, base_url=url, api_key="k"
                ),
            ),
            None,
            default_embedding_timeout=17,
        )
        result = await rag.embedding_func(["a"])

    assert result.shape == (1, len(_VECTOR))
    assert [call.get("timeout") for call in calls] == [17]


@pytest.mark.asyncio
async def test_rerank_and_jina_attempts_exceeding_timeout_are_retried(monkeypatch):
    from lightrag import rerank as rerank_module
    from lightrag.llm import jina as jina_module

    monkeypatch.setattr(rerank_module.generic_rerank_api.retry, "wait", wait_none())
    async with _http_stub(first_delay=2.0) as (url, hits):
        res = await bind_rerank_timeout(rerank_module.cohere_rerank, 0.15)(
            query="q", documents=["a"], api_key="k", base_url=f"{url}/rerank"
        )
    assert [r["index"] for r in res] == [0]
    assert len(hits) == 2

    monkeypatch.setattr(jina_module.jina_embed.func.retry, "wait", wait_none())
    async with _http_stub(first_delay=2.0) as (url, hits2):
        res2 = await bind_embedding_timeout(jina_module.jina_embed.func, 0.15)(
            ["a"],
            embedding_dim=len(_VECTOR),
            base_url=f"{url}/jina/embeddings",
            api_key="k",
        )
    assert res2.shape == (1, len(_VECTOR))
    assert len(hits2) == 2


# --------------------------------------------------------------------------- #
# Section 2: LLM SDK default timeout (LLM_TIMEOUT <= 0 leaves SDK default)
# --------------------------------------------------------------------------- #

_LLM_TIMEOUT_CASES = [
    pytest.param(0, None, id="zero-sdk-default"),
    pytest.param(None, None, id="unset-sdk-default"),
    pytest.param(30, 30, id="positive-passed-on"),
]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(0, None, id="zero"),
        pytest.param(-1, None, id="negative"),
        pytest.param(None, None, id="unset"),
        pytest.param(240, 240, id="positive"),
        pytest.param(False, False, id="bool-untouched"),
    ],
)
def test_provider_timeout(value, expected):
    assert provider_timeout(value) is expected or provider_timeout(value) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("timeout", "expected"), _LLM_TIMEOUT_CASES)
async def test_openai_client_timeout(monkeypatch, timeout, expected):
    from lightrag.llm import openai as openai_module

    calls: list[dict] = []
    _record_calls(monkeypatch, openai_module, "AsyncOpenAI", calls)

    async with _http_stub(every_delay=_SLOW_DELAY) as (url, _):
        result = await openai_module.openai_complete_if_cache(
            "m",
            "hi",
            base_url=f"{url}/v1",
            api_key="k",
            timeout=timeout,
            openai_client_configs={"max_retries": 0},
        )

    assert result == "ok"
    assert len(calls) == 1
    if expected is None:
        assert "timeout" not in calls[0]
        assert openai.AsyncOpenAI(api_key="k").timeout == openai.DEFAULT_TIMEOUT
    else:
        assert calls[0]["timeout"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("timeout", "expected"), _LLM_TIMEOUT_CASES)
async def test_anthropic_client_timeout(monkeypatch, timeout, expected):
    from lightrag.llm import anthropic as anthropic_module

    calls: list[dict] = []
    _record_calls(monkeypatch, anthropic_module, "AsyncAnthropic", calls)

    async with _http_stub(every_delay=_SLOW_DELAY) as (url, _):
        result = await anthropic_module.anthropic_complete_if_cache(
            "m", "hi", base_url=url, api_key="k", timeout=timeout, max_retries=0
        )

    assert result == "ok"
    assert len(calls) == 1
    if expected is None:
        assert "timeout" not in calls[0]
    else:
        assert calls[0]["timeout"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("timeout", "expected"), _LLM_TIMEOUT_CASES)
async def test_ollama_client_timeout(monkeypatch, timeout, expected):
    from lightrag.llm import ollama as ollama_module

    calls: list[dict] = []
    _record_calls(monkeypatch, ollama_module.ollama, "AsyncClient", calls)

    async with _http_stub(every_delay=_SLOW_DELAY) as (url, _):
        result = await ollama_module._ollama_model_if_cache(
            "m", "hi", host=url, timeout=timeout
        )

    assert result == "ok"
    assert len(calls) == 1
    assert calls[0]["timeout"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("timeout", "expected"), _LLM_TIMEOUT_CASES)
async def test_zhipu_request_timeout(monkeypatch, timeout, expected):
    from lightrag.llm import zhipu as zhipu_module

    calls: list[dict] = []
    real_build = zhipu_module._build_client

    def build(client_cls, api_key):
        client = real_build(client_cls, api_key)
        _record_calls(monkeypatch, client.chat.completions, "create", calls)
        return client

    monkeypatch.setattr(zhipu_module, "_build_client", build)

    async with _http_stub(every_delay=_SLOW_DELAY) as (url, _):
        monkeypatch.setenv("ZAI_BASE_URL", f"{url}/api/")
        result = await zhipu_module.zhipu_complete_if_cache(
            "hi", model="m", api_key="id.secret", timeout=timeout
        )

    assert result == "ok"
    assert len(calls) == 1
    if expected is None:
        assert "timeout" not in calls[0]
    else:
        assert calls[0]["timeout"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("timeout", "expected"), _LLM_TIMEOUT_CASES)
async def test_lollms_session_timeout(monkeypatch, timeout, expected):
    from lightrag.llm import lollms as lollms_module

    calls: list[dict] = []

    async with _http_stub(every_delay=_SLOW_DELAY) as (url, _):
        _record_calls(monkeypatch, lollms_module.aiohttp, "ClientSession", calls)
        result = await lollms_module.lollms_model_if_cache(
            "m", "hi", base_url=url, timeout=timeout
        )

    assert result == "ok"
    assert len(calls) == 1
    if expected is None:
        assert "timeout" not in calls[0]
    else:
        assert calls[0]["timeout"].total == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("timeout", "expected"), _LLM_TIMEOUT_CASES)
async def test_non_http_llm_client_timeouts(monkeypatch, timeout, expected):
    from lightrag.llm import gemini, openai_responses
    from lightrag.llm.bedrock import _bedrock_client_kwargs

    # OpenAI Responses
    seen_responses: dict = {}

    def _responses_client(**kwargs):
        seen_responses.update(kwargs)
        raise _Recorded

    monkeypatch.setattr(
        openai_responses, "create_openai_async_client", _responses_client
    )
    with pytest.raises(_Recorded):
        await openai_responses.openai_responses_complete_if_cache.__wrapped__(
            "m", "hi", api_key="k", timeout=timeout
        )
    assert seen_responses["timeout"] == expected

    # Gemini (milliseconds)
    seen_gemini: dict = {}

    def _gemini_client(api_key, base_url, timeout_ms):
        seen_gemini["timeout_ms"] = timeout_ms
        raise _Recorded

    monkeypatch.setattr(gemini, "_get_gemini_client", _gemini_client)
    with pytest.raises(_Recorded):
        await gemini.gemini_complete_if_cache.__wrapped__(
            "m", "hi", api_key="k", timeout=timeout
        )
    assert seen_gemini["timeout_ms"] == (None if expected is None else expected * 1000)

    # Bedrock (botocore Config)
    bedrock_kwargs = _bedrock_client_kwargs(
        region="us-east-1", endpoint_url=None, timeout=timeout
    )
    if expected is None:
        assert "config" not in bedrock_kwargs
    else:
        cfg = bedrock_kwargs["config"]
        assert (cfg.connect_timeout, cfg.read_timeout) == (expected, expected)


# --------------------------------------------------------------------------- #
# Section 3: Bundled LLM wrapper timeout forwarding
# --------------------------------------------------------------------------- #

_OPENAI_CLIENT_CONFIGS = "openai_client_configs"
_WRAPPERS = [
    (
        "lightrag.llm.openai",
        "openai_complete",
        "openai_complete_if_cache",
        _OPENAI_CLIENT_CONFIGS,
    ),
    (
        "lightrag.llm.openai",
        "gpt_4o_complete",
        "openai_complete_if_cache",
        _OPENAI_CLIENT_CONFIGS,
    ),
    (
        "lightrag.llm.openai",
        "gpt_4o_mini_complete",
        "openai_complete_if_cache",
        _OPENAI_CLIENT_CONFIGS,
    ),
    (
        "lightrag.llm.openai",
        "nvidia_openai_complete",
        "openai_complete_if_cache",
        _OPENAI_CLIENT_CONFIGS,
    ),
    (
        "lightrag.llm.openai",
        "azure_openai_complete",
        "azure_openai_complete_if_cache",
        _OPENAI_CLIENT_CONFIGS,
    ),
    (
        "lightrag.llm.openai_responses",
        "openai_responses_complete",
        "openai_responses_complete_if_cache",
        _OPENAI_CLIENT_CONFIGS,
    ),
    ("lightrag.llm.anthropic", "anthropic_complete", "anthropic_complete_if_cache", None),
    (
        "lightrag.llm.anthropic",
        "claude_3_opus_complete",
        "anthropic_complete_if_cache",
        None,
    ),
    (
        "lightrag.llm.anthropic",
        "claude_3_sonnet_complete",
        "anthropic_complete_if_cache",
        None,
    ),
    (
        "lightrag.llm.anthropic",
        "claude_3_haiku_complete",
        "anthropic_complete_if_cache",
        None,
    ),
    ("lightrag.llm.gemini", "gemini_model_complete", "gemini_complete_if_cache", None),
    ("lightrag.llm.ollama", "ollama_model_complete", "_ollama_model_if_cache", None),
    ("lightrag.llm.lollms", "lollms_model_complete", "lollms_model_if_cache", None),
    ("lightrag.llm.zhipu", "zhipu_complete", "zhipu_complete_if_cache", None),
    ("lightrag.llm.bedrock", "bedrock_complete", "bedrock_complete_if_cache", None),
]
_CLIENT_CONFIG_WRAPPERS = [case for case in _WRAPPERS if case[3] is not None]


def _wrapper_id(case) -> str:
    return case[1]


async def _call_wrapper(monkeypatch, case, *, role_timeout=None, **call_kwargs):
    module_name, wrapper_name, inner_name, _ = case
    module = importlib.import_module(module_name)
    calls: list[dict] = []

    async def _record(*args, **kwargs):
        calls.append(kwargs)
        return "ok"

    monkeypatch.setattr(module, inner_name, _record)
    hashing_kv = SimpleNamespace(
        global_config={"llm_model_name": "test-model", "default_llm_timeout": 240}
    )
    token = _ROLE_LLM_TIMEOUT.set(role_timeout) if role_timeout is not None else None
    try:
        await getattr(module, wrapper_name)(
            "prompt", hashing_kv=hashing_kv, **call_kwargs
        )
    finally:
        if token is not None:
            _ROLE_LLM_TIMEOUT.reset(token)

    assert len(calls) == 1
    return calls[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _WRAPPERS, ids=_wrapper_id)
async def test_wrapper_uses_role_timeout_inside_role_call(monkeypatch, case):
    forwarded = await _call_wrapper(monkeypatch, case, role_timeout=77)
    assert forwarded["timeout"] == 77


@pytest.mark.asyncio
async def test_wrapper_fallback_and_explicit_timeout(monkeypatch):
    case = _WRAPPERS[0]
    forwarded = await _call_wrapper(monkeypatch, case, openai_client_configs=None)
    assert forwarded["timeout"] == 240

    forwarded_explicit = await _call_wrapper(
        monkeypatch, case, role_timeout=77, timeout=5
    )
    assert forwarded_explicit["timeout"] == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _CLIENT_CONFIG_WRAPPERS, ids=_wrapper_id)
async def test_wrapper_keeps_client_config_timeout(monkeypatch, case):
    forwarded = await _call_wrapper(
        monkeypatch, case, role_timeout=77, **{case[3]: {"timeout": 400}}
    )
    assert "timeout" not in forwarded
    assert forwarded[case[3]] == {"timeout": 400}


def test_openai_client_config_timeout_and_apply_llm_timeout_no_config():
    from lightrag.llm.openai import create_openai_async_client

    client = create_openai_async_client(
        api_key="test-key", client_configs={"timeout": 400}, timeout=None
    )
    assert client.timeout == 400

    for kwargs in ({}, {"hashing_kv": object()}):
        apply_llm_timeout(kwargs)
        assert "timeout" not in kwargs


# --------------------------------------------------------------------------- #
# Section 4: Non-positive llm_timeout in priority_limit_async_func_call
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize("llm_timeout", [0, 0.0, -1, None])
async def test_non_positive_or_unset_llm_timeout_runs_the_call(llm_timeout):
    async def func(value: str, **_kwargs):
        await asyncio.sleep(0.05)
        return value

    kwargs = {"llm_timeout": llm_timeout} if llm_timeout is not None else {}
    wrapped = priority_limit_async_func_call(1, queue_name="timeout-test", **kwargs)(
        func
    )
    try:
        assert await wrapped("hello") == "hello"
    finally:
        await wrapped.shutdown(graceful=True, timeout=1)


@pytest.mark.asyncio
async def test_non_positive_llm_timeout_is_not_logged_as_enforced(lightrag_caplog):
    async def quick_func(value: str, **_kwargs):
        return value

    wrapped = priority_limit_async_func_call(
        1, llm_timeout=0, queue_name="zero-timeout LLM func"
    )(quick_func)
    try:
        with lightrag_caplog.at_level("INFO", logger="lightrag"):
            assert await wrapped("hello") == "hello"
    finally:
        await wrapped.shutdown(graceful=True, timeout=1)

    messages = [record.getMessage() for record in lightrag_caplog.records]
    init_lines = [m for m in messages if "new workers initialized" in m]
    assert init_lines, messages
    assert "Func: 0s" not in init_lines[0]
    assert "Worker: 0s" not in init_lines[0]


@pytest.mark.asyncio
async def test_positive_llm_timeout_still_bounds_execution():
    async def stuck_func(value: str, **_kwargs):
        await asyncio.sleep(30)
        return value

    wrapped = priority_limit_async_func_call(
        1, llm_timeout=0.05, queue_name="bounded LLM func"
    )(stuck_func)
    try:
        with pytest.raises(asyncio.TimeoutError):
            await wrapped("hello")
    finally:
        await wrapped.shutdown(graceful=False, timeout=1)
