"""Deferred-embedding and deferred-delete tests for ``NanoVectorDBStorage``.

Subclasses the shared file-backed vector contracts in
``tests/kg/_file_backed_vector_contracts.py`` and adds Nano-specific matrix,
eager ``delete_entity``, and internal-error coverage.
"""

from __future__ import annotations

import contextlib
import json
from unittest.mock import patch

import pytest

nano_vectordb = pytest.importorskip("nano_vectordb")  # noqa: F841

from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage  # noqa: E402
from lightrag.kg.shared_storage import (  # noqa: E402
    finalize_share_data,
    initialize_share_data,
)
from lightrag.kg.write_seq import WRITE_SEQ_FIELD  # noqa: E402
from lightrag.utils import EmbeddingFunc, compute_mdhash_id  # noqa: E402
from tests.kg._file_backed_vector_contracts import (  # noqa: E402
    DIM,
    CountingEmbed,
    DeferredDeleteContractMixin,
    DeferredEmbeddingContractMixin,
)

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


def _spy_client_delete(storage: NanoVectorDBStorage, counter: list[list[str]]) -> None:
    """Wrap the materialized client's ``delete`` to count invocations."""
    original = storage._client.delete

    def spy(ids):
        counter.append(list(ids))
        return original(ids)

    storage._client.delete = spy


class _NanoDeferredAdapter:
    """Adapter binding the shared vector-contract mixins to ``NanoVectorDBStorage``."""

    def make_storage(
        self,
        tmp_path,
        embed,
        *,
        namespace: str = "test_vectors",
        meta_fields: set[str] | None = None,
    ) -> NanoVectorDBStorage:
        return NanoVectorDBStorage(
            namespace=namespace,
            workspace="ws",
            global_config={
                "working_dir": str(tmp_path),
                "embedding_batch_num": 32,
                "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
            },
            embedding_func=EmbeddingFunc(
                embedding_dim=DIM, max_token_size=512, func=embed
            ),
            meta_fields=meta_fields or {"content"},
        )

    def materialized_count(self, storage: NanoVectorDBStorage) -> int:
        return len(storage._client)

    def is_dirty(self, storage: NanoVectorDBStorage) -> bool:
        return storage._client_dirty

    def assert_consistent(self, storage: NanoVectorDBStorage) -> None:
        raw = getattr(storage._client, "_NanoVectorDB__storage")
        assert raw["matrix"].shape == (len(storage._client), DIM)

    def make_save_fail(
        self, storage: NanoVectorDBStorage, *, message: str = "disk full"
    ):
        original = storage._save_to_disk_locked

        async def boom(_on_committed):
            raise OSError(message)

        storage._save_to_disk_locked = boom

        def restore():
            storage._save_to_disk_locked = original

        return restore

    def make_save_fail_once(
        self, storage: NanoVectorDBStorage, *, message: str = "boom"
    ) -> list[int]:
        original = storage._save_to_disk_locked
        calls: list[int] = []

        async def fail_once(on_committed):
            calls.append(1)
            if len(calls) == 1:
                raise OSError(message)
            await original(on_committed)

        storage._save_to_disk_locked = fail_once
        return calls

    @contextlib.contextmanager
    def frozen_clock(self, stamp: int):
        with patch("lightrag.kg.nano_vector_db_impl.time.time", return_value=stamp):
            yield

    def resident_row(self, storage: NanoVectorDBStorage, doc_id: str) -> dict:
        return storage._client.get([doc_id])[0]

    def strip_write_seq_on_disk(self, storage: NanoVectorDBStorage) -> None:
        with open(storage._client_file_name, encoding="utf-8") as f:
            snapshot = json.load(f)
        stripped = [row.pop(WRITE_SEQ_FIELD, None) for row in snapshot["data"]]
        assert any(token is not None for token in stripped), (
            "the token must reach the committed snapshot"
        )
        with open(storage._client_file_name, "w", encoding="utf-8") as f:
            json.dump(snapshot, f)


class TestNanoDeferredEmbedding(_NanoDeferredAdapter, DeferredEmbeddingContractMixin):
    """Shared deferred-embedding contract plus Nano-specific redo-log tests."""

    @pytest.mark.asyncio
    async def test_delete_entity_evicts_the_upsert_redo_entry(self, tmp_path):
        """Eager ``delete_entity`` evicts ``_unsaved_upserts`` unconditionally."""
        writer = await self._make_initialized(tmp_path)
        other = await self._make_initialized(tmp_path)

        entity_id = compute_mdhash_id("EntityA", prefix="ent-")
        await writer.upsert({entity_id: {"content": "entity row"}})
        restore = self.make_save_fail(writer)
        with pytest.raises(OSError):
            await writer.index_done_callback()
        restore()
        assert entity_id in writer._unsaved_upserts

        await other.upsert({"id3": {"content": "gamma"}})
        assert await other.index_done_callback() is True

        await writer.delete_entity("EntityA")
        assert entity_id not in writer._unsaved_upserts

        assert await writer.index_done_callback() is True

        reader = await self._make_initialized(tmp_path)
        assert await reader.get_by_id(entity_id) is None
        assert (await reader.get_by_id("id3"))["content"] == "gamma"

    @pytest.mark.asyncio
    async def test_repeated_failed_saves_do_not_re_upsert_each_time(self, tmp_path):
        embed = CountingEmbed()
        storage = await self._make_initialized(tmp_path, embed)

        await storage.upsert({"id1": {"content": "alpha"}})
        restore = self.make_save_fail(storage)
        with pytest.raises(OSError):
            await storage.index_done_callback()

        upsert_calls: list[list[str]] = []
        real_upsert = storage._client.upsert

        def spy(datas):
            upsert_calls.append([d["__id__"] for d in datas])
            return real_upsert(datas=datas)

        storage._client.upsert = spy

        with pytest.raises(OSError):
            await storage.index_done_callback()
        assert upsert_calls == [], "no reload happened, so there is nothing to redo"

        restore()
        assert await storage.index_done_callback() is True
        assert upsert_calls == [], "the durable retry must not rewrite either"
        assert not storage._unsaved_upserts
        assert embed.call_count == 1
        assert (await storage.get_by_id("id1"))["content"] == "alpha"


class TestNanoDeferredDelete(_NanoDeferredAdapter, DeferredDeleteContractMixin):
    """Shared deferred-delete contract plus Nano-specific batching & eager-delete tests."""

    @pytest.mark.asyncio
    async def test_delete_defers_materialization_to_flush(self, tmp_path):
        storage = await self._make_initialized(tmp_path)
        await self._upsert_and_flush(
            storage,
            {
                "id1": {"content": "alpha"},
                "id2": {"content": "beta"},
                "id3": {"content": "gamma"},
            },
        )

        calls: list[list[str]] = []
        _spy_client_delete(storage, calls)

        await storage.delete(["id1"])
        await storage.delete(["id2"])
        assert calls == [], "delete() must not touch the materialized client"
        assert len(storage._client) == 3, "rows stay materialized until flush"

        await storage.index_done_callback()
        assert len(calls) == 1, "flush applies all queued deletes in ONE batch"
        assert sorted(calls[0]) == ["id1", "id2"]
        assert len(storage._client) == 1

    @pytest.mark.asyncio
    async def test_query_reads_the_materialized_index_only(self, tmp_path):
        """``query`` ranks materialized rows only so queued deletes do not shrink top_k."""
        storage = await self._make_initialized(tmp_path)
        await self._upsert_and_flush(
            storage, {f"id{i}": {"content": f"doc{i}"} for i in range(5)}
        )

        await storage.delete([f"id{i}" for i in range(4)])

        hits = await storage.query("doc0", top_k=3)
        assert len(hits) == 3, "top_k must not be truncated by queued deletes"

        await storage.index_done_callback()
        hits = await storage.query("doc0", top_k=3)
        assert [h["id"] for h in hits] == ["id4"]

    @pytest.mark.asyncio
    async def test_drop_discards_queued_deletes(self, tmp_path):
        storage = await self._make_initialized(tmp_path)
        await self._upsert_and_flush(storage, {"id1": {"content": "alpha"}})

        await storage.delete(["id1"])
        await storage.drop()

        assert storage._pending_deletes == set()
        assert storage._pending_upserts == {}

    @pytest.mark.asyncio
    async def test_delete_only_flush_persists_and_is_idempotent(self, tmp_path):
        storage = await self._make_initialized(tmp_path)
        await self._upsert_and_flush(
            storage, {"id1": {"content": "alpha"}, "id2": {"content": "beta"}}
        )

        await storage.delete(["id2"])
        await storage.index_done_callback()
        assert len(storage._client) == 1

        await storage.index_done_callback()
        assert len(storage._client) == 1

        reloaded = await self._make_initialized(tmp_path)
        assert await reloaded.get_by_id("id2") is None
        assert (await reloaded.get_by_id("id1"))["content"] == "alpha"

    @pytest.mark.asyncio
    async def test_unflushed_deletes_roll_back_like_pending_upserts(self, tmp_path):
        storage = await self._make_initialized(tmp_path)
        await self._upsert_and_flush(storage, {"id1": {"content": "alpha"}})

        await storage.delete(["id1"])

        reloaded = await self._make_initialized(tmp_path)
        assert (await reloaded.get_by_id("id1"))["content"] == "alpha"

    @pytest.mark.asyncio
    async def test_queue_of_absent_ids_is_not_retained(self, tmp_path):
        storage = await self._make_initialized(tmp_path)
        await self._upsert_and_flush(storage, {"id1": {"content": "alpha"}})

        await storage.delete(["never-inserted"])
        await storage.index_done_callback()

        assert storage._pending_deletes == set()
        assert storage._unsaved_deletes == {}
        assert storage._client_dirty is False

    @pytest.mark.asyncio
    async def test_an_id_that_matched_nothing_is_never_replayed(self, tmp_path):
        writer = await self._make_initialized(tmp_path)
        other = await self._make_initialized(tmp_path)
        await self._upsert_and_flush(writer, {"seed": {"content": "s"}})

        restore = self.make_save_fail(writer)
        await writer.upsert({"other": {"content": "o"}})
        with pytest.raises(OSError):
            await writer.index_done_callback()
        await writer.delete(["ghost"])
        with pytest.raises(OSError):
            await writer.index_done_callback()
        restore()

        assert "ghost" not in writer._unsaved_deletes

        await self._upsert_and_flush(other, {"ghost": {"content": "created later"}})
        await writer.index_done_callback()

        reader = await self._make_initialized(tmp_path)
        row = await reader.get_by_id("ghost")
        assert row is not None and row["content"] == "created later"

    @pytest.mark.asyncio
    async def test_repeated_failed_saves_do_not_re_delete_each_time(self, tmp_path):
        storage = await self._make_initialized(tmp_path)
        await self._upsert_and_flush(
            storage, {"id1": {"content": "alpha"}, "id2": {"content": "beta"}}
        )

        calls: list[list[str]] = []
        _spy_client_delete(storage, calls)

        restore = self.make_save_fail(storage)
        await storage.delete(["id1"])
        for _ in range(4):
            with pytest.raises(OSError):
                await storage.index_done_callback()
        restore()
        await storage.index_done_callback()

        assert len(calls) == 1, f"one matrix rebuild, not one per retry: {calls}"
        assert await storage.get_by_id("id1") is None

    @pytest.mark.asyncio
    async def test_redo_log_reads_as_absent_while_the_save_is_pending(self, tmp_path):
        """After a reload brings a logged-removed row back, all read paths hide it."""
        storage = await self._make_initialized(tmp_path)
        await self._upsert_and_flush(
            storage, {"id1": {"content": "alpha"}, "id2": {"content": "beta"}}
        )

        await storage.delete(["id1"])
        restore = self.make_save_fail(storage)
        with pytest.raises(OSError):
            await storage.index_done_callback()
        restore()
        assert set(storage._unsaved_deletes) == {"id1"}

        storage.storage_updated.value = True

        assert await storage.get_by_id("id1") is None
        assert (await storage.get_by_ids(["id1", "id2"]))[0] is None
        assert "id1" not in await storage.get_vectors_by_ids(["id1", "id2"])

    @pytest.mark.asyncio
    async def test_a_rewrite_after_a_logged_delete_survives_the_replay(self, tmp_path):
        storage = await self._make_initialized(tmp_path)
        await self._upsert_and_flush(storage, {"id1": {"content": "old"}})

        await storage.delete(["id1"])
        restore = self.make_save_fail(storage)
        with pytest.raises(OSError):
            await storage.index_done_callback()
        assert set(storage._unsaved_deletes) == {"id1"}

        await storage.upsert({"id1": {"content": "new"}})
        assert set(storage._unsaved_deletes) == {"id1"}, (
            "a buffered row that may never materialize must not clear the log"
        )

        with pytest.raises(OSError):
            await storage.index_done_callback()
        restore()
        assert await storage.index_done_callback() is True

        reader = await self._make_initialized(tmp_path)
        assert (await reader.get_by_id("id1"))["content"] == "new"

    @pytest.mark.asyncio
    async def test_abort_after_a_rewrite_keeps_the_redo_entry(self, tmp_path):
        storage = await self._make_initialized(tmp_path)
        await self._upsert_and_flush(
            storage, {"id1": {"content": "old"}, "id2": {"content": "beta"}}
        )

        await storage.delete(["id1"])
        restore = self.make_save_fail(storage)
        with pytest.raises(OSError):
            await storage.index_done_callback()

        await storage.upsert({"id1": {"content": "new"}})
        await storage.drop_pending_index_ops()
        assert set(storage._unsaved_deletes) == {"id1"}

        storage.storage_updated.value = True
        restore()
        assert await storage.index_done_callback() is True

        reader = await self._make_initialized(tmp_path)
        assert await reader.get_by_id("id1") is None
        assert (await reader.get_by_id("id2"))["content"] == "beta"

    @pytest.mark.asyncio
    async def test_eager_entity_delete_is_replayed_across_a_reload(self, tmp_path):
        storage = await self._make_initialized(tmp_path)
        entity_id = compute_mdhash_id("Alice", prefix="ent-")
        await self._upsert_and_flush(
            storage, {entity_id: {"content": "alice"}, "id2": {"content": "beta"}}
        )

        await storage.delete_entity("Alice")
        assert set(storage._unsaved_deletes) == {entity_id}
        assert storage._client_dirty is True

        storage.storage_updated.value = True
        assert await storage.index_done_callback() is True

        reader = await self._make_initialized(tmp_path)
        assert await reader.get_by_id(entity_id) is None
        assert (await reader.get_by_id("id2"))["content"] == "beta"

    @pytest.mark.asyncio
    async def test_delete_does_not_swallow_an_internal_error(self, tmp_path):
        storage = await self._make_initialized(tmp_path)
        await self._upsert_and_flush(storage, {"id1": {"content": "alpha"}})

        class _BrokenPendingDeletes(set):
            """Set stand-in that raises when an id is queued."""

            def add(self, item):
                raise RuntimeError("boom")

        storage._pending_deletes = _BrokenPendingDeletes()

        with pytest.raises(RuntimeError, match="boom"):
            await storage.delete(["id1"])
