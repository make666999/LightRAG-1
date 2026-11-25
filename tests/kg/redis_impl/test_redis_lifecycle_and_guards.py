"""Offline unit tests for Redis eviction guard, finalize/refcount lifecycle, and startup rollback."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("redis", reason="redis is required for Redis storage tests")

from redis.exceptions import ResponseError  # noqa: E402

from lightrag.exceptions import StorageControlPlaneError  # noqa: E402
from lightrag.kg.redis_impl import (  # noqa: E402
    RedisConnectionManager,
    RedisDocStatusStorage,
    RedisKVStorage,
)
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data  # noqa: E402
from lightrag.namespace import NameSpace  # noqa: E402
from lightrag.utils import (  # noqa: E402
    EmbeddingFunc,
    Tokenizer,
    TokenizerInterface,
    logger,
)

from .fake_redis import FakeRedis  # noqa: E402

pytestmark = pytest.mark.offline

_REDIS_URL = "redis://localhost:6379/0"
STORAGE_CLASSES = [RedisKVStorage, RedisDocStatusStorage]


class _DummyEmbeddingFunc:
    embedding_dim = 1
    max_token_size = 1

    async def __call__(self, texts, **kwargs):
        return [[0.0] for _ in texts]


class _StubTokenizer(TokenizerInterface):
    """Avoids tiktoken downloads during offline LightRAG rollback test."""

    def encode(self, content: str) -> list[int]:
        return [ord(c) for c in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


def _namespace_for(cls) -> str:
    return (
        NameSpace.KV_STORE_FULL_DOCS if cls is RedisKVStorage else NameSpace.DOC_STATUS
    )


def _new_storage(cls, url: str = _REDIS_URL, pool=None):
    """Build an initialized storage with mocked client and pool handles."""
    s = cls.__new__(cls)
    s.workspace = "ws"
    s.namespace = "kv" if cls is RedisKVStorage else "doc_status"
    s._redis = AsyncMock()
    s._redis_url = url
    s._pool = pool if pool is not None else MagicMock()
    s._initialized = True
    return s


def _unconnected_storage(cls, url: str = _REDIS_URL):
    """Build a storage in the post-construction, pre-initialize state."""
    s = cls.__new__(cls)
    s.workspace = "ws"
    s.namespace = "kv" if cls is RedisKVStorage else "doc_status"
    s.final_namespace = f"ws_{s.namespace}"
    s._redis_url = url
    s._pool = None
    s._redis = None
    s._initialized = False
    return s


def _drop_from_registry(url: str) -> None:
    RedisConnectionManager._pools.pop(url, None)
    RedisConnectionManager._pool_refs.pop(url, None)


@pytest.fixture
def _shared():
    """Shared-storage data-init lock fixture."""
    initialize_share_data()
    yield
    finalize_share_data()


# ---------------------------------------------------------------------------
# Eviction-policy startup guard
# ---------------------------------------------------------------------------


class TestRedisEvictionGuard:
    @pytest.fixture(autouse=True)
    def _setup_shared_and_cache(self):
        from lightrag.kg import redis_impl

        initialize_share_data()
        cache = getattr(redis_impl, "_eviction_checked", None)
        if cache is not None:
            cache.clear()
        yield
        if cache is not None:
            cache.clear()
        finalize_share_data()

    @pytest.fixture
    def fake(self, monkeypatch):
        instance = FakeRedis()
        monkeypatch.setattr(
            "lightrag.kg.redis_impl.RedisConnectionManager.get_pool",
            lambda redis_url: MagicMock(name="fake_pool"),
        )
        monkeypatch.setattr(
            "lightrag.kg.redis_impl.Redis", lambda connection_pool=None, **_: instance
        )
        return instance

    @staticmethod
    def _doc_status_storage(workspace: str = "evict-ws"):
        return RedisDocStatusStorage(
            namespace=NameSpace.DOC_STATUS,
            global_config={},
            embedding_func=_DummyEmbeddingFunc(),
            workspace=workspace,
        )

    @staticmethod
    def _kv_storage(workspace: str = "evict-ws"):
        return RedisKVStorage(
            namespace=NameSpace.KV_STORE_FULL_DOCS,
            global_config={},
            embedding_func=_DummyEmbeddingFunc(),
            workspace=workspace,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("policy", ["allkeys-lru", "allkeys-lfu", "allkeys-random"])
    async def test_allkeys_policy_with_a_memory_cap_refuses_to_start(
        self, fake, policy
    ):
        """Every allkeys-* policy can evict a TTL-less key, so all of them refuse."""
        fake.config_values = {"maxmemory": "1073741824", "maxmemory-policy": policy}
        storage = self._doc_status_storage()
        with pytest.raises(StorageControlPlaneError, match="evicting cache"):
            await storage.initialize()
        assert storage._initialized is False

    @pytest.mark.asyncio
    async def test_rejected_policy_still_refuses_later_storages(
        self, fake, monkeypatch
    ):
        """Both KV and doc_status run the guard and a rejection is never cached."""
        monkeypatch.delenv("REDIS_ALLOW_EVICTION_POLICY", raising=False)
        fake.config_values = {
            "maxmemory": "1073741824",
            "maxmemory-policy": "allkeys-lru",
        }

        with pytest.raises(StorageControlPlaneError, match="evicting cache"):
            await self._kv_storage().initialize()

        retry = self._doc_status_storage(workspace="retry-ws")
        try:
            with pytest.raises(StorageControlPlaneError, match="evicting cache"):
                await retry.initialize()
        finally:
            await retry.finalize()

    @pytest.mark.asyncio
    async def test_rejected_policy_can_recover_on_later_startup(
        self, fake, monkeypatch, lightrag_caplog
    ):
        monkeypatch.delenv("REDIS_ALLOW_EVICTION_POLICY", raising=False)
        fake.config_values = {
            "maxmemory": "1073741824",
            "maxmemory-policy": "allkeys-lru",
        }
        with pytest.raises(StorageControlPlaneError, match="evicting cache"):
            await self._doc_status_storage().initialize()

        monkeypatch.setenv("REDIS_ALLOW_EVICTION_POLICY", "true")

        retry = self._kv_storage()
        try:
            with lightrag_caplog.at_level(logging.WARNING, logger=logger.name):
                await retry.initialize()
            assert "configured to EVICT" in lightrag_caplog.text
        finally:
            await retry.finalize()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "config_values",
        [
            pytest.param(
                {"maxmemory": "0", "maxmemory-policy": "allkeys-lru"},
                id="allkeys_without_memory_cap",
            ),
            pytest.param(
                {"maxmemory": "1073741824", "maxmemory-policy": "volatile-lru"},
                id="volatile_policy",
            ),
        ],
    )
    async def test_safe_eviction_configs_allow_startup(self, fake, config_values):
        fake.config_values = config_values
        storage = self._doc_status_storage()
        await storage.initialize()
        assert storage._initialized is True

    @pytest.mark.asyncio
    async def test_unreadable_config_warns_but_starts(self, fake, lightrag_caplog):
        fake.fail_next["config_get"] = ResponseError("unknown command 'CONFIG'")
        storage = self._doc_status_storage()
        with lightrag_caplog.at_level(logging.WARNING, logger=logger.name):
            await storage.initialize()
        assert storage._initialized is True
        assert "Could not read the Redis maxmemory policy" in lightrag_caplog.text

    @pytest.mark.asyncio
    async def test_explicit_override_downgrades_to_a_warning(
        self, fake, monkeypatch, lightrag_caplog
    ):
        monkeypatch.setenv("REDIS_ALLOW_EVICTION_POLICY", " Yes ")
        fake.config_values = {
            "maxmemory": "1073741824",
            "maxmemory-policy": "allkeys-lru",
        }
        storage = self._doc_status_storage()
        with lightrag_caplog.at_level(logging.WARNING, logger=logger.name):
            await storage.initialize()
        assert storage._initialized is True
        assert "configured to EVICT" in lightrag_caplog.text

    @pytest.mark.asyncio
    async def test_policy_is_checked_once_per_url(self, fake):
        calls = {"n": 0}
        real_config_get = fake.config_get

        async def counting_config_get(pattern="*"):
            calls["n"] += 1
            return await real_config_get(pattern)

        fake.config_get = counting_config_get

        await self._doc_status_storage().initialize()
        await self._kv_storage().initialize()
        assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Finalize, pool reference counting, and cancellation safety
# ---------------------------------------------------------------------------


class TestRedisFinalize:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("cls", STORAGE_CLASSES)
    async def test_finalize_closes_client_and_releases_pool(self, cls):
        s = _new_storage(cls)
        client = s._redis

        with patch.object(
            RedisConnectionManager, "release_pool", new_callable=AsyncMock
        ) as release_pool:
            await s.finalize()

        client.aclose.assert_awaited_once()
        release_pool.assert_awaited_once_with(_REDIS_URL)
        assert s._redis is None
        assert s._pool is None
        assert s._initialized is False

    @pytest.mark.asyncio
    async def test_finalize_is_idempotent(self):
        s = _new_storage(RedisKVStorage)

        with patch.object(
            RedisConnectionManager, "release_pool", new_callable=AsyncMock
        ) as release_pool:
            await s.finalize()
            await s.finalize()

        release_pool.assert_awaited_once_with(_REDIS_URL)
        assert s._redis is None
        assert s._pool is None

    @pytest.mark.asyncio
    async def test_finalize_after_close_is_safe(self):
        s = _new_storage(RedisDocStatusStorage)

        with patch.object(
            RedisConnectionManager, "release_pool", new_callable=AsyncMock
        ) as release_pool:
            await s.close()
            await s.finalize()

        release_pool.assert_awaited_once_with(_REDIS_URL)
        assert s._redis is None
        assert s._pool is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cls", STORAGE_CLASSES)
    async def test_double_close_does_not_steal_sibling_pool_ref(self, cls):
        url = "redis://localhost:6379/15"
        try:
            pool = RedisConnectionManager.get_pool(url)
            RedisConnectionManager.get_pool(url)
            assert RedisConnectionManager._pool_refs[url] == 2

            a = _new_storage(cls, url, pool)
            await a.close()
            assert RedisConnectionManager._pool_refs[url] == 1
            assert url in RedisConnectionManager._pools

            await a.close()
            assert RedisConnectionManager._pool_refs[url] == 1
            assert url in RedisConnectionManager._pools

            b = _new_storage(cls, url, pool)
            await b.close()
            assert url not in RedisConnectionManager._pools
            assert url not in RedisConnectionManager._pool_refs
        finally:
            _drop_from_registry(url)

    @pytest.mark.asyncio
    async def test_release_pool_closes_only_on_last_reference(self):
        url = "redis://localhost:6379/16"
        fake_pool = MagicMock()
        fake_pool.aclose = AsyncMock()
        try:
            with patch(
                "lightrag.kg.redis_impl.ConnectionPool.from_url", return_value=fake_pool
            ):
                RedisConnectionManager.get_pool(url)
                RedisConnectionManager.get_pool(url)

            await RedisConnectionManager.release_pool(url)
            fake_pool.aclose.assert_not_awaited()
            assert url in RedisConnectionManager._pools

            await RedisConnectionManager.release_pool(url)
            fake_pool.aclose.assert_awaited_once()
            assert url not in RedisConnectionManager._pools
            assert url not in RedisConnectionManager._pool_refs
        finally:
            _drop_from_registry(url)

    @pytest.mark.asyncio
    async def test_close_all_pools_disconnects_and_clears_registry(self):
        url = "redis://localhost:6379/17"
        fake_pool = MagicMock()
        fake_pool.aclose = AsyncMock()
        try:
            with patch(
                "lightrag.kg.redis_impl.ConnectionPool.from_url", return_value=fake_pool
            ):
                RedisConnectionManager.get_pool(url)

            await RedisConnectionManager.close_all_pools()

            fake_pool.aclose.assert_awaited_once()
            assert not RedisConnectionManager._pools
            assert not RedisConnectionManager._pool_refs
        finally:
            _drop_from_registry(url)

    @pytest.mark.parametrize("cls", STORAGE_CLASSES)
    def test_construction_takes_no_pool_reference(self, cls):
        pools_before = dict(RedisConnectionManager._pools)
        refs_before = dict(RedisConnectionManager._pool_refs)

        storage = cls(
            namespace=_namespace_for(cls),
            global_config={},
            embedding_func=_DummyEmbeddingFunc(),
            workspace="leak-ws",
        )

        assert RedisConnectionManager._pools == pools_before
        assert RedisConnectionManager._pool_refs == refs_before
        assert storage._pool is None
        assert storage._redis is None
        assert storage._initialized is False

    @pytest.mark.parametrize("cls", STORAGE_CLASSES)
    def test_discarded_storages_leave_no_pool_reference(self, cls, monkeypatch):
        url = "redis://localhost:6379/20"
        monkeypatch.setenv("REDIS_URI", url)
        try:
            RedisConnectionManager.get_pool(url)
            assert RedisConnectionManager._pool_refs[url] == 1

            for _ in range(3):
                cls(
                    namespace=_namespace_for(cls),
                    global_config={},
                    embedding_func=_DummyEmbeddingFunc(),
                    workspace="leak-ws",
                )

            assert RedisConnectionManager._pool_refs[url] == 1
        finally:
            _drop_from_registry(url)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cls", STORAGE_CLASSES)
    async def test_close_before_initialize_releases_nothing(self, cls):
        url = "redis://localhost:6379/18"
        try:
            RedisConnectionManager.get_pool(url)
            assert RedisConnectionManager._pool_refs[url] == 1

            s = _unconnected_storage(cls, url)
            await s.close()

            assert RedisConnectionManager._pool_refs[url] == 1
            assert url in RedisConnectionManager._pools
        finally:
            _drop_from_registry(url)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cls", STORAGE_CLASSES)
    async def test_initialize_client_failure_releases_the_pool_ref(self, cls, _shared):
        fake_pool = MagicMock()
        fake_pool.aclose = AsyncMock()
        boom = RuntimeError("unique-init-boom")
        url = "redis://localhost:6379/21"
        s = _unconnected_storage(cls, url)
        try:
            with (
                patch(
                    "lightrag.kg.redis_impl.ConnectionPool.from_url",
                    return_value=fake_pool,
                ),
                patch("lightrag.kg.redis_impl.Redis", side_effect=boom),
            ):
                with pytest.raises(RuntimeError, match="unique-init-boom"):
                    await s.initialize()

            assert url not in RedisConnectionManager._pools
            assert url not in RedisConnectionManager._pool_refs
            fake_pool.aclose.assert_awaited_once()
            assert s._pool is None
            assert s._redis is None
            assert s._initialized is False
        finally:
            _drop_from_registry(url)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cls", STORAGE_CLASSES)
    async def test_initialize_ping_failure_releases_the_pool_ref(self, cls, _shared):
        fake_pool = MagicMock()
        fake_pool.aclose = AsyncMock()
        client = AsyncMock()
        client.ping.side_effect = RuntimeError("unique-ping-boom")
        url = "redis://localhost:6379/22"
        s = _unconnected_storage(cls, url)
        try:
            with (
                patch(
                    "lightrag.kg.redis_impl.ConnectionPool.from_url",
                    return_value=fake_pool,
                ),
                patch("lightrag.kg.redis_impl.Redis", return_value=client),
            ):
                with pytest.raises(RuntimeError, match="unique-ping-boom"):
                    await s.initialize()

            client.aclose.assert_awaited_once()
            assert url not in RedisConnectionManager._pools
            assert url not in RedisConnectionManager._pool_refs
            fake_pool.aclose.assert_awaited_once()
            assert s._initialized is False
        finally:
            _drop_from_registry(url)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cls", STORAGE_CLASSES)
    async def test_close_releases_pool_even_if_client_close_cancelled(self, cls):
        s = _new_storage(cls)
        s._redis = AsyncMock()
        s._redis.aclose.side_effect = asyncio.CancelledError

        with patch.object(
            RedisConnectionManager, "release_pool", new_callable=AsyncMock
        ) as release_pool:
            with pytest.raises(asyncio.CancelledError):
                await s.close()

        release_pool.assert_awaited_once_with(_REDIS_URL)
        assert s._redis is None
        assert s._pool is None

    @pytest.mark.asyncio
    async def test_close_cancelled_midflight_still_releases_real_pool_ref(self):
        url = "redis://localhost:6379/19"
        started = asyncio.Event()
        release = asyncio.Event()

        async def blocking_aclose(*args, **kwargs):
            started.set()
            await release.wait()

        try:
            pool = RedisConnectionManager.get_pool(url)
            RedisConnectionManager.get_pool(url)

            a = _new_storage(RedisKVStorage, url, pool)
            a._redis = AsyncMock()
            a._redis.aclose = AsyncMock(side_effect=blocking_aclose)

            task = asyncio.create_task(a.close())
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            assert RedisConnectionManager._pool_refs[url] == 1
            assert url in RedisConnectionManager._pools
        finally:
            _drop_from_registry(url)


# ---------------------------------------------------------------------------
# Full-LightRAG startup rollback
# ---------------------------------------------------------------------------


class TestRedisStartupRollback:
    @pytest.fixture
    def redis_url(self, tmp_path, monkeypatch):
        url = f"redis://rollback-{tmp_path.name}:6379"
        monkeypatch.setenv("REDIS_URI", url)
        monkeypatch.setattr(
            "lightrag.kg.redis_impl.ConnectionPool.from_url",
            lambda *args, **kwargs: MagicMock(name="pool", aclose=AsyncMock()),
        )
        monkeypatch.setattr(
            "lightrag.kg.redis_impl.Redis",
            lambda connection_pool=None, **_: AsyncMock(),
        )
        yield url
        _drop_from_registry(url)

    async def test_a_startup_refused_at_step_1_holds_no_pool_reference(
        self, tmp_path, redis_url
    ):
        """A whole LightRAG on Redis refused before business storages start
        holds no pool reference after construction or rollback."""
        from lightrag import LightRAG

        async def _llm(prompt, **kwargs):  # pragma: no cover - never called
            return ""

        embed_called: list[bool] = []

        async def _embed(texts, **kwargs):  # pragma: no cover - never called
            embed_called.append(True)
            raise AssertionError("startup must refuse before anything embeds")

        initialize_share_data(workers=1)
        try:
            rag = LightRAG(
                working_dir=str(tmp_path),
                workspace="tenant",
                llm_model_func=_llm,
                embedding_func=EmbeddingFunc(
                    embedding_dim=8, func=_embed, model_name="bge-m3"
                ),
                kv_storage="RedisKVStorage",
                doc_status_storage="RedisDocStatusStorage",
                config_storage="JsonKVStorage",
                tokenizer=Tokenizer("stub", _StubTokenizer()),
            )
            assert redis_url not in RedisConnectionManager._pool_refs

            boom = ConnectionError("configuration backend unreachable")

            async def _refuse():
                raise boom

            rag.configuration_storage.initialize = _refuse

            with pytest.raises(ConnectionError) as excinfo:
                await rag.initialize_storages()
            assert excinfo.value is boom
            assert embed_called == []
            assert redis_url not in RedisConnectionManager._pool_refs
            assert redis_url not in RedisConnectionManager._pools
        finally:
            finalize_share_data()
