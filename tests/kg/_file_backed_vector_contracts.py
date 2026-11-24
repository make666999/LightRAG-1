"""Shared contract mixins and test doubles for file-backed vector storages.

Used by ``tests/kg/faiss_impl/`` and ``tests/kg/nano_impl/`` to pin the common
contracts documented in ``docs/design/FileBackedSnapshotContract.md`` without
duplicating the deferred-embedding, deferred-delete, fingerprint-fence, and
exact-count test suites across both backends.

This module intentionally has no ``test_`` prefix and its mixins do not start
with ``Test`` so pytest never collects it directly; neither ``faiss`` nor
``nano_vectordb`` is imported here so each backend directory runs standalone.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from lightrag.exceptions import StorageCapabilityError
from lightrag.kg import file_fingerprint, write_seq
from lightrag.kg.write_seq import WRITE_SEQ_FIELD

DIM = 8
# A blocked loop gives zero beats however long the window; a live one beats
# within one 5 ms GIL switch interval, so 0.05 s keeps a wide margin.
BLOCK_SECONDS = 0.05


class CountingEmbed:
    """Async embedding callable recording batch count and embedded texts."""

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.call_count = 0
        self.embedded_texts: list[str] = []

    async def __call__(self, texts, **kwargs):
        self.call_count += 1
        self.embedded_texts.extend(texts)
        return np.array(
            [
                np.full(self.dim, (abs(hash(t)) % 97) + 1, dtype=np.float32)
                for t in texts
            ]
        )


class FailingEmbed:
    """Async embedding callable that always raises ``RuntimeError``."""

    def __init__(self):
        self.call_count = 0

    async def __call__(self, texts, **kwargs):
        self.call_count += 1
        raise RuntimeError("embed boom")


class NoLock:
    """Async no-op lock for unit-testing storage internals without shared_storage."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class Heartbeat:
    """Background task counting how often the event loop gets to schedule it."""

    def __init__(self):
        self.beats = 0
        self._stop = False
        self._task: asyncio.Task[None] | None = None

    async def __aenter__(self):
        async def _run():
            while not self._stop:
                self.beats += 1
                await asyncio.sleep(0)

        self._task = asyncio.create_task(_run())
        await asyncio.sleep(0)
        return self

    async def __aexit__(self, *_exc):
        self._stop = True
        if self._task is not None:
            await self._task


def truncate_file(path: str) -> None:
    """Overwrite a file with its first half to simulate a mid-write crash."""
    payload = Path(path).read_bytes()
    Path(path).write_bytes(payload[: len(payload) // 2])


class _FileBackedVectorHooks:
    """Abstract adapter interface implemented by Faiss and Nano test subclasses."""

    def make_storage(
        self,
        tmp_path,
        embed,
        *,
        namespace: str = "test_vectors",
        meta_fields: set[str] | None = None,
    ):
        raise NotImplementedError

    def materialized_count(self, storage) -> int:
        raise NotImplementedError

    def is_dirty(self, storage) -> bool:
        raise NotImplementedError

    def assert_consistent(self, storage) -> None:
        """Backend-specific structural invariant check (no-op by default)."""

    def make_save_fail(self, storage, *, message: str = "disk full"):
        """Make the backend's save hook raise OSError(message); return restore()."""
        raise NotImplementedError

    def make_save_fail_once(self, storage, *, message: str = "boom") -> list[int]:
        """Fail the first save with OSError(message) then delegate; return call log."""
        raise NotImplementedError

    def frozen_clock(self, stamp: int):
        """Context manager pinning ``time.time`` inside the backend module."""
        raise NotImplementedError

    def resident_row(self, storage, doc_id: str) -> dict:
        """Return the materialized metadata dict for ``doc_id``."""
        raise NotImplementedError

    def strip_write_seq_on_disk(self, storage) -> None:
        """Remove ``WRITE_SEQ_FIELD`` from every row in the committed snapshot."""
        raise NotImplementedError

    async def _make_initialized(
        self,
        tmp_path,
        embed=None,
        *,
        namespace: str = "test_vectors",
        meta_fields: set[str] | None = None,
    ):
        storage = self.make_storage(
            tmp_path,
            embed if embed is not None else CountingEmbed(),
            namespace=namespace,
            meta_fields=meta_fields,
        )
        await storage.initialize()
        return storage

    async def _upsert_and_flush(self, storage, data: dict) -> None:
        await storage.upsert(data)
        assert await storage.index_done_callback() is True

    async def _strand_dirty_with_empty_buffers(self, writer) -> None:
        """Reach ``dirty=True`` with all four pending/redo buffers empty."""
        await writer.upsert(
            {"idX": {"content": "ours-x"}, "idY": {"content": "ours-y"}}
        )
        restore = self.make_save_fail(writer)
        with pytest.raises(OSError):
            await writer.index_done_callback()
        restore()

        await writer.delete(["idX", "idY"])
        await writer.drop_pending_index_ops()

        assert self.is_dirty(writer) is True
        assert not writer._pending_upserts
        assert not writer._pending_deletes
        assert not writer._unsaved_deletes
        assert not writer._unsaved_upserts


class DeferredEmbeddingContractMixin(_FileBackedVectorHooks):
    """Shared deferred-embedding and upsert redo-log contracts for Faiss & Nano."""

    @pytest.mark.asyncio
    async def test_upsert_defers_embedding_to_index_done_callback(self, tmp_path):
        embed = CountingEmbed()
        storage = await self._make_initialized(tmp_path, embed)

        await storage.upsert({"id1": {"content": "alpha"}, "id2": {"content": "beta"}})
        assert embed.call_count == 0, "upsert must not embed"
        assert self.materialized_count(storage) == 0, (
            "nothing should be materialized yet"
        )
        self.assert_consistent(storage)

        await storage.index_done_callback()
        assert embed.call_count == 1, "flush should embed in a single batch"
        assert sorted(embed.embedded_texts) == ["alpha", "beta"]
        assert self.materialized_count(storage) == 2
        self.assert_consistent(storage)

    @pytest.mark.asyncio
    async def test_repeated_upserts_same_id_embed_once_per_flush(self, tmp_path):
        embed = CountingEmbed()
        storage = await self._make_initialized(tmp_path, embed)

        await storage.upsert({"id1": {"content": "v1"}})
        await storage.upsert({"id1": {"content": "v2"}})
        await storage.upsert({"id1": {"content": "v3"}})

        await storage.index_done_callback()

        assert embed.call_count == 1
        assert embed.embedded_texts == ["v3"], "only the latest content is embedded"
        assert self.materialized_count(storage) == 1
        self.assert_consistent(storage)

    @pytest.mark.asyncio
    async def test_get_vectors_caches_and_flush_reuses(self, tmp_path):
        embed = CountingEmbed()
        storage = await self._make_initialized(tmp_path, embed)

        await storage.upsert({"id1": {"content": "alpha"}})

        vecs = await storage.get_vectors_by_ids(["id1"])
        assert "id1" in vecs and len(vecs["id1"]) == DIM
        assert embed.call_count == 1, "get_vectors_by_ids embeds pending lazily"

        await storage.index_done_callback()
        assert embed.call_count == 1, "flush should reuse the cached temp vector"
        assert self.materialized_count(storage) == 1
        self.assert_consistent(storage)

    @pytest.mark.asyncio
    async def test_reupsert_after_get_vectors_clears_cached_vector(self, tmp_path):
        embed = CountingEmbed()
        storage = await self._make_initialized(tmp_path, embed)

        await storage.upsert({"id1": {"content": "old"}})
        await storage.get_vectors_by_ids(["id1"])
        assert embed.call_count == 1

        await storage.upsert({"id1": {"content": "new"}})
        await storage.index_done_callback()

        assert embed.call_count == 2
        assert embed.embedded_texts == ["old", "new"]
        self.assert_consistent(storage)

    @pytest.mark.asyncio
    async def test_delete_cancels_pending_and_defers_materialized_removal(
        self, tmp_path
    ):
        embed = CountingEmbed()
        storage = await self._make_initialized(tmp_path, embed)

        await storage.upsert({"id1": {"content": "alpha"}})
        await storage.index_done_callback()
        await storage.upsert({"id2": {"content": "beta"}})

        await storage.delete(["id1", "id2"])

        assert "id2" not in storage._pending_upserts, "delete cancels pending upsert"
        assert self.materialized_count(storage) == 1, (
            "materialized row survives until the flush"
        )
        assert {"id1", "id2"} <= storage._pending_deletes
        assert await storage.get_by_id("id1") is None
        assert await storage.get_by_id("id2") is None

        assert await storage.index_done_callback() is True
        assert self.materialized_count(storage) == 0, (
            "flush removes the materialized row"
        )
        self.assert_consistent(storage)

    @pytest.mark.asyncio
    async def test_stale_client_reload_still_flushes_pending_upsert(self, tmp_path):
        embed = CountingEmbed()
        writer = await self._make_initialized(tmp_path, embed)
        stale_writer = await self._make_initialized(tmp_path, embed)

        await writer.upsert({"id1": {"content": "alpha"}})
        assert await writer.index_done_callback() is True
        assert stale_writer.storage_updated.value is True

        await stale_writer.upsert({"id2": {"content": "beta"}})
        assert await stale_writer.index_done_callback() is True

        reader = await self._make_initialized(tmp_path, embed)
        rows = await reader.get_by_ids(["id1", "id2"])
        assert [row["id"] for row in rows] == ["id1", "id2"]
        assert stale_writer._pending_upserts == {}
        self.assert_consistent(reader)

    @pytest.mark.asyncio
    async def test_delete_on_stale_client_applies_after_reload_at_flush(self, tmp_path):
        embed = CountingEmbed()
        writer = await self._make_initialized(tmp_path, embed)
        stale_deleter = await self._make_initialized(tmp_path, embed)

        await writer.upsert({"id1": {"content": "alpha"}})
        assert await writer.index_done_callback() is True
        assert stale_deleter.storage_updated.value is True

        await stale_deleter.delete(["id1"])
        assert stale_deleter.storage_updated.value is True
        assert await stale_deleter.index_done_callback() is True
        assert stale_deleter.storage_updated.value is False

        reader = await self._make_initialized(tmp_path, embed)
        assert await reader.get_by_id("id1") is None
        self.assert_consistent(reader)

    @pytest.mark.asyncio
    async def test_finalize_reloads_stale_client_before_flushing(self, tmp_path):
        embed = CountingEmbed()
        writer = await self._make_initialized(tmp_path, embed)
        stale_finalizer = await self._make_initialized(tmp_path, embed)

        await writer.upsert({"id1": {"content": "alpha"}})
        assert await writer.index_done_callback() is True
        assert stale_finalizer.storage_updated.value is True

        await stale_finalizer.upsert({"id2": {"content": "beta"}})
        await stale_finalizer.finalize()

        reader = await self._make_initialized(tmp_path, embed)
        rows = await reader.get_by_ids(["id1", "id2"])
        assert [row["id"] for row in rows] == ["id1", "id2"]
        assert stale_finalizer._pending_upserts == {}
        self.assert_consistent(reader)

    @pytest.mark.asyncio
    async def test_read_your_writes_and_query_after_flush(self, tmp_path):
        embed = CountingEmbed()
        storage = await self._make_initialized(tmp_path, embed)

        await storage.upsert({"id1": {"content": "alpha"}})

        hit = await storage.get_by_id("id1")
        assert hit is not None and hit["id"] == "id1" and hit["content"] == "alpha"
        by_ids = await storage.get_by_ids(["id1", "missing"])
        assert by_ids[0]["id"] == "id1" and by_ids[1] is None
        assert await storage.query("alpha", top_k=5) == [], (
            "query ignores unflushed data"
        )

        await storage.index_done_callback()
        results = await storage.query("alpha", top_k=5)
        assert any(r["id"] == "id1" for r in results)
        self.assert_consistent(storage)

    @pytest.mark.asyncio
    async def test_delete_entity_relation_cancels_pending(self, tmp_path):
        storage = await self._make_initialized(
            tmp_path,
            CountingEmbed(),
            namespace="test_relations",
            meta_fields={"content", "src_id", "tgt_id"},
        )

        await storage.upsert({"r1": {"content": "rel1", "src_id": "A", "tgt_id": "B"}})
        await storage.index_done_callback()
        await storage.upsert(
            {
                "r2": {"content": "rel2", "src_id": "A", "tgt_id": "C"},
                "r3": {"content": "rel3", "src_id": "X", "tgt_id": "Y"},
            }
        )

        await storage.delete_entity_relation("A")

        assert "r2" not in storage._pending_upserts, "incident pending entry cancelled"
        assert "r3" in storage._pending_upserts, "unrelated pending entry preserved"
        assert self.materialized_count(storage) == 0, "materialized A->B removed"
        self.assert_consistent(storage)

    @pytest.mark.asyncio
    async def test_flush_embedding_failure_raises_and_keeps_pending(self, tmp_path):
        storage = await self._make_initialized(tmp_path, FailingEmbed())

        await storage.upsert({"id1": {"content": "alpha"}})

        with pytest.raises(RuntimeError, match="embed boom"):
            await storage.index_done_callback()

        assert "id1" in storage._pending_upserts, "pending preserved for retry"
        assert self.materialized_count(storage) == 0, (
            "nothing materialized on embed failure"
        )
        assert self.is_dirty(storage) is False
        self.assert_consistent(storage)

    @pytest.mark.asyncio
    async def test_drop_discards_pending_without_embedding(self, tmp_path):
        embed = CountingEmbed()
        storage = await self._make_initialized(tmp_path, embed)

        await storage.upsert({"id1": {"content": "alpha"}})
        assert "id1" in storage._pending_upserts

        result = await storage.drop()

        assert result["status"] == "success"
        assert storage._pending_upserts == {}, "drop discards buffered upserts"
        assert embed.call_count == 0, "drop must not embed"
        assert self.is_dirty(storage) is False
        self.assert_consistent(storage)

    @pytest.mark.asyncio
    async def test_finalize_retries_save_after_flush_failure(self, tmp_path):
        embed = CountingEmbed()
        storage = await self._make_initialized(tmp_path, embed)

        await storage.upsert({"id1": {"content": "alpha"}})
        save_calls = self.make_save_fail_once(storage, message="boom")

        with pytest.raises(OSError, match="boom"):
            await storage.finalize()

        assert storage._pending_upserts == {}
        assert self.is_dirty(storage) is True

        await storage.finalize()

        assert len(save_calls) == 2
        assert self.is_dirty(storage) is False

        reader = await self._make_initialized(tmp_path, embed)
        hit = await reader.get_by_id("id1")
        assert hit is not None and hit["id"] == "id1"
        self.assert_consistent(reader)

    @pytest.mark.asyncio
    async def test_drop_pending_does_not_rollback_materialized(self, tmp_path):
        storage = await self._make_initialized(tmp_path, CountingEmbed())

        await storage.upsert({"id1": {"content": "alpha"}})
        self.make_save_fail(storage, message="save boom")
        with pytest.raises(OSError, match="save boom"):
            await storage.index_done_callback()

        assert storage._pending_upserts == {}, "flush succeeded so pending is empty"
        assert self.is_dirty(storage) is True
        assert self.materialized_count(storage) == 1, "id1 materialized, not saved"

        await storage.upsert({"id2": {"content": "beta"}})
        assert "id2" in storage._pending_upserts

        await storage.drop_pending_index_ops()

        assert storage._pending_upserts == {}, "pending id2 dropped"
        assert self.materialized_count(storage) == 1, "materialized id1 NOT rolled back"
        assert self.is_dirty(storage) is True, "still dirty for a later save retry"
        self.assert_consistent(storage)

    @pytest.mark.asyncio
    async def test_index_done_replays_unsaved_upsert_after_foreign_commit(
        self, tmp_path
    ):
        embed = CountingEmbed()
        writer = await self._make_initialized(tmp_path, embed)
        other = await self._make_initialized(tmp_path)

        await writer.upsert({"id2": {"content": "beta"}})
        restore = self.make_save_fail(writer)
        with pytest.raises(OSError, match="disk full"):
            await writer.index_done_callback()
        restore()
        assert writer._pending_upserts == {}, "flush succeeded so pending is empty"
        assert writer._unsaved_upserts, "the flushed doc moved into the redo log"
        calls_after_failure = embed.call_count

        await other.upsert({"id3": {"content": "gamma"}})
        assert await other.index_done_callback() is True
        assert writer.storage_updated.value is True

        assert await writer.index_done_callback() is True
        assert embed.call_count == calls_after_failure, "replay must not re-embed"
        assert not writer._unsaved_upserts, "durable save clears the redo log"
        self.assert_consistent(writer)

        reader = await self._make_initialized(tmp_path)
        assert (await reader.get_by_id("id2"))["content"] == "beta"
        assert (await reader.get_by_id("id3"))["content"] == "gamma"
        self.assert_consistent(reader)

    @pytest.mark.asyncio
    async def test_finalize_replays_unsaved_upserts_after_a_foreign_commit(
        self, tmp_path
    ):
        writer = await self._make_initialized(tmp_path)
        other = await self._make_initialized(tmp_path)
        await self._upsert_and_flush(writer, {"id1": {"content": "alpha"}})

        await writer.upsert({"id2": {"content": "beta"}})
        restore = self.make_save_fail(writer)
        with pytest.raises(OSError):
            await writer.index_done_callback()
        restore()
        assert writer._unsaved_upserts, "the flushed doc moved into the redo log"

        await other.upsert({"id3": {"content": "gamma"}})
        assert await other.index_done_callback() is True

        await writer.finalize()
        self.assert_consistent(writer)

        reader = await self._make_initialized(tmp_path)
        assert (await reader.get_by_id("id2"))["content"] == "beta"
        assert (await reader.get_by_id("id3"))["content"] == "gamma"
        self.assert_consistent(reader)

    @pytest.mark.asyncio
    async def test_reads_serve_an_unsaved_upsert_after_a_foreign_reload(self, tmp_path):
        embed = CountingEmbed()
        writer = await self._make_initialized(tmp_path, embed)
        other = await self._make_initialized(tmp_path)

        await writer.upsert({"id2": {"content": "beta"}})
        restore = self.make_save_fail(writer)
        with pytest.raises(OSError):
            await writer.index_done_callback()
        restore()

        await other.upsert({"id3": {"content": "gamma"}})
        assert await other.index_done_callback() is True

        got = await writer.get_by_id("id2")
        assert got is not None and got["content"] == "beta"
        by_ids = await writer.get_by_ids(["id2", "id3"])
        assert by_ids[0] is not None and by_ids[0]["content"] == "beta"
        assert by_ids[1] is not None and by_ids[1]["content"] == "gamma"
        vectors = await writer.get_vectors_by_ids(["id2"])
        assert "id2" in vectors and len(vectors["id2"]) == DIM
        assert embed.call_count == 1, "the logged vector is reused, never re-embedded"

    @pytest.mark.asyncio
    async def test_replay_overwrites_an_older_foreign_row_under_the_same_id(
        self, tmp_path
    ):
        writer = await self._make_initialized(tmp_path)
        other = await self._make_initialized(tmp_path)

        await other.upsert({"idX": {"content": "theirs"}})
        assert await other.index_done_callback() is True

        await writer.upsert({"idX": {"content": "ours"}})
        restore = self.make_save_fail(writer)
        with pytest.raises(OSError):
            await writer.index_done_callback()
        restore()

        await other.upsert({"idZ": {"content": "unrelated"}})
        assert await other.index_done_callback() is True

        assert await writer.index_done_callback() is True
        self.assert_consistent(writer)

        reader = await self._make_initialized(tmp_path)
        assert (await reader.get_by_id("idX"))["content"] == "ours"
        assert (await reader.get_by_id("idZ"))["content"] == "unrelated"
        self.assert_consistent(reader)

    @pytest.mark.asyncio
    async def test_delete_after_a_failed_save_evicts_the_redo_entry(self, tmp_path):
        writer = await self._make_initialized(tmp_path)
        other = await self._make_initialized(tmp_path)

        await writer.upsert({"id1": {"content": "alpha"}, "id2": {"content": "beta"}})
        restore = self.make_save_fail(writer)
        with pytest.raises(OSError):
            await writer.index_done_callback()
        restore()
        assert "id2" in writer._unsaved_upserts

        await writer.delete(["id2"])
        assert "id2" not in writer._unsaved_upserts, "delete evicts the redo entry"

        await other.upsert({"id3": {"content": "gamma"}})
        assert await other.index_done_callback() is True

        assert await writer.index_done_callback() is True
        self.assert_consistent(writer)

        reader = await self._make_initialized(tmp_path)
        assert await reader.get_by_id("id2") is None, "replay must not resurrect id2"
        assert (await reader.get_by_id("id1"))["content"] == "alpha"
        assert (await reader.get_by_id("id3"))["content"] == "gamma"

    @pytest.mark.asyncio
    async def test_delete_entity_relation_evicts_matching_redo_upserts(self, tmp_path):
        writer = await self._make_initialized(
            tmp_path,
            namespace="test_relations",
            meta_fields={"content", "src_id", "tgt_id"},
        )
        other = await self._make_initialized(
            tmp_path,
            namespace="test_relations",
            meta_fields={"content", "src_id", "tgt_id"},
        )

        await writer.upsert(
            {
                "r1": {"content": "rel1", "src_id": "A", "tgt_id": "B"},
                "r2": {"content": "rel2", "src_id": "X", "tgt_id": "Y"},
            }
        )
        restore = self.make_save_fail(writer)
        with pytest.raises(OSError):
            await writer.index_done_callback()
        restore()
        assert "r1" in writer._unsaved_upserts and "r2" in writer._unsaved_upserts

        await other.upsert({"id3": {"content": "gamma"}})
        assert await other.index_done_callback() is True

        await writer.delete_entity_relation("A")
        assert "r1" not in writer._unsaved_upserts, "incident redo entry evicted"
        assert "r2" in writer._unsaved_upserts, "unrelated redo entry preserved"

        assert await writer.index_done_callback() is True
        self.assert_consistent(writer)

        reader = await self._make_initialized(
            tmp_path,
            namespace="test_relations",
            meta_fields={"content", "src_id", "tgt_id"},
        )
        assert await reader.get_by_id("r1") is None
        assert (await reader.get_by_id("r2"))["content"] == "rel2"

    @pytest.mark.asyncio
    async def test_aborting_batch_keeps_the_upsert_redo_log(self, tmp_path):
        writer = await self._make_initialized(tmp_path)
        other = await self._make_initialized(tmp_path)

        await writer.upsert({"id1": {"content": "alpha"}})
        restore = self.make_save_fail(writer)
        with pytest.raises(OSError):
            await writer.index_done_callback()
        restore()

        await writer.drop_pending_index_ops()
        assert writer._unsaved_upserts, "abort keeps the redo log"

        await other.upsert({"id2": {"content": "beta"}})
        assert await other.index_done_callback() is True

        assert await writer.index_done_callback() is True
        self.assert_consistent(writer)

        reader = await self._make_initialized(tmp_path)
        assert (await reader.get_by_id("id1"))["content"] == "alpha"
        assert (await reader.get_by_id("id2"))["content"] == "beta"

    @pytest.mark.asyncio
    async def test_finalize_reloads_when_only_the_dirty_flag_is_set(self, tmp_path):
        writer = await self._make_initialized(tmp_path)
        other = await self._make_initialized(tmp_path)

        await self._strand_dirty_with_empty_buffers(writer)

        await other.upsert({"foreign": {"content": "theirs"}})
        assert await other.index_done_callback() is True

        await writer.finalize()

        reader = await self._make_initialized(tmp_path)
        foreign = await reader.get_by_id("foreign")
        assert foreign is not None, (
            "finalize saved its stale pre-commit snapshot over the foreign commit"
        )
        assert foreign["content"] == "theirs"
        assert await reader.get_by_id("idX") is None
        assert await reader.get_by_id("idY") is None
        self.assert_consistent(reader)

    @pytest.mark.asyncio
    async def test_replay_declines_a_strictly_newer_foreign_row(self, tmp_path):
        writer = await self._make_initialized(tmp_path)
        other = await self._make_initialized(tmp_path)

        await writer.upsert({"idX": {"content": "ours-stale"}})
        restore = self.make_save_fail(writer)
        with pytest.raises(OSError):
            await writer.index_done_callback()
        restore()
        assert "idX" in writer._unsaved_upserts

        with self.frozen_clock(int(time.time()) + 60):
            await other.upsert({"idX": {"content": "theirs-newer"}})
        assert await other.index_done_callback() is True

        assert await writer.index_done_callback() is True
        assert "idX" not in writer._unsaved_upserts, (
            "a superseded redo entry must be dropped"
        )

        reader = await self._make_initialized(tmp_path)
        assert (await reader.get_by_id("idX"))["content"] == "theirs-newer"
        self.assert_consistent(reader)

    @pytest.mark.asyncio
    async def test_reads_prefer_a_strictly_newer_foreign_row(self, tmp_path):
        embed = CountingEmbed()
        writer = await self._make_initialized(tmp_path, embed)
        other = await self._make_initialized(tmp_path)

        await writer.upsert({"idX": {"content": "ours-stale"}})
        restore = self.make_save_fail(writer)
        with pytest.raises(OSError):
            await writer.index_done_callback()
        restore()

        with self.frozen_clock(int(time.time()) + 60):
            await other.upsert({"idX": {"content": "theirs-newer"}})
        assert await other.index_done_callback() is True

        got = await writer.get_by_id("idX")
        assert got is not None and got["content"] == "theirs-newer"
        by_ids = await writer.get_by_ids(["idX"])
        assert by_ids[0] is not None and by_ids[0]["content"] == "theirs-newer"
        vectors = await writer.get_vectors_by_ids(["idX"])
        assert "idX" in vectors and len(vectors["idX"]) == DIM
        assert embed.call_count == 1, "reads must not re-embed to answer this"

    @pytest.mark.asyncio
    async def test_same_second_tie_is_broken_by_the_write_sequence(self, tmp_path):
        writer = await self._make_initialized(tmp_path)
        other = await self._make_initialized(tmp_path)

        frozen = int(time.time())
        with self.frozen_clock(frozen):
            await writer.upsert({"idX": {"content": "ours-stale"}})
            restore = self.make_save_fail(writer)
            with pytest.raises(OSError):
                await writer.index_done_callback()
            restore()

            await other.upsert({"idX": {"content": "theirs-newer"}})
        assert await other.index_done_callback() is True

        logged = writer._unsaved_upserts["idX"].record
        resident = self.resident_row(other, "idX")
        assert logged["__created_at__"] == resident["__created_at__"]
        assert resident[WRITE_SEQ_FIELD] > logged[WRITE_SEQ_FIELD]

        assert (await writer.get_by_id("idX"))["content"] == "theirs-newer"

        assert await writer.index_done_callback() is True
        assert "idX" not in writer._unsaved_upserts

        reader = await self._make_initialized(tmp_path)
        assert (await reader.get_by_id("idX"))["content"] == "theirs-newer"
        self.assert_consistent(reader)

    @pytest.mark.asyncio
    async def test_same_second_tie_without_a_write_sequence_still_replays(
        self, tmp_path
    ):
        writer = await self._make_initialized(tmp_path)
        other = await self._make_initialized(tmp_path)

        frozen = int(time.time())
        with self.frozen_clock(frozen):
            await writer.upsert({"idX": {"content": "ours"}})
            restore = self.make_save_fail(writer)
            with pytest.raises(OSError):
                await writer.index_done_callback()
            restore()

            await other.upsert({"idX": {"content": "theirs"}})
            assert await other.index_done_callback() is True
            self.strip_write_seq_on_disk(other)

        assert await writer.index_done_callback() is True

        reader = await self._make_initialized(tmp_path)
        assert (await reader.get_by_id("idX"))["content"] == "ours"
        self.assert_consistent(reader)

    @pytest.mark.asyncio
    async def test_a_backward_clock_step_does_not_revert_a_newer_commit(
        self, tmp_path, monkeypatch
    ):
        writer = await self._make_initialized(tmp_path)
        other = await self._make_initialized(tmp_path)

        monkeypatch.setattr(write_seq, "_last_seq", 0)
        with patch.object(write_seq.time, "time_ns", return_value=1_000):
            with self.frozen_clock(100):
                await writer.upsert({"idX": {"content": "ours-stale"}})
        restore = self.make_save_fail(writer)
        with pytest.raises(OSError):
            await writer.index_done_callback()
        restore()

        with patch.object(write_seq.time, "time_ns", return_value=1_001):
            with self.frozen_clock(99):
                await other.upsert({"idX": {"content": "theirs-newer"}})
        assert await other.index_done_callback() is True

        logged = writer._unsaved_upserts["idX"].record
        resident = self.resident_row(other, "idX")
        assert resident["__created_at__"] < logged["__created_at__"]
        assert resident[WRITE_SEQ_FIELD] > logged[WRITE_SEQ_FIELD]

        assert (await writer.get_by_id("idX"))["content"] == "theirs-newer"
        assert await writer.index_done_callback() is True
        assert "idX" not in writer._unsaved_upserts

        reader = await self._make_initialized(tmp_path)
        assert (await reader.get_by_id("idX"))["content"] == "theirs-newer"
        self.assert_consistent(reader)


class DeferredDeleteContractMixin(_FileBackedVectorHooks):
    """Shared deferred-delete and delete redo-log contracts for Faiss & Nano."""

    @pytest.mark.asyncio
    async def test_fresh_upsert_supersedes_queued_delete(self, tmp_path):
        storage = await self._make_initialized(tmp_path)
        await self._upsert_and_flush(
            storage, {"id1": {"content": "old"}, "id2": {"content": "keep"}}
        )

        await storage.delete(["id1", "missing-id"])
        await storage.upsert({"id1": {"content": "new"}})
        assert "id1" not in storage._pending_deletes, (
            "upsert must cancel the queued delete for the same id"
        )
        assert await storage.index_done_callback() is True

        record = await storage.get_by_id("id1")
        assert record is not None and record["content"] == "new"
        assert (await storage.get_by_id("id2"))["content"] == "keep"
        assert self.materialized_count(storage) == 2, "no duplicate or leftover rows"
        self.assert_consistent(storage)

    @pytest.mark.asyncio
    async def test_read_your_writes_hides_queued_deletes(self, tmp_path):
        storage = await self._make_initialized(tmp_path)
        await self._upsert_and_flush(
            storage, {"a": {"content": "alpha"}, "b": {"content": "beta"}}
        )

        await storage.delete(["a"])

        assert await storage.get_by_id("a") is None
        got = await storage.get_by_ids(["a", "b"])
        assert got[0] is None
        assert got[1] is not None and got[1]["content"] == "beta"
        vectors = await storage.get_vectors_by_ids(["a", "b"])
        assert "a" not in vectors and "b" in vectors

    @pytest.mark.asyncio
    async def test_queued_delete_survives_concurrent_commit_and_is_scoped_to_id(
        self, tmp_path
    ):
        """A queued delete is scoped to the id and applies after any peer reload."""
        writer = await self._make_initialized(tmp_path)
        other = await self._make_initialized(tmp_path)
        await self._upsert_and_flush(
            writer, {"shared": {"content": "v1"}, "keep": {"content": "beta"}}
        )

        await writer.delete(["shared"])

        await self._upsert_and_flush(
            other, {"shared": {"content": "rewritten"}, "extra": {"content": "gamma"}}
        )
        assert writer.storage_updated.value is True

        assert await writer.index_done_callback() is True

        reader = await self._make_initialized(tmp_path)
        assert await reader.get_by_id("shared") is None, "the id was asked to go"
        assert (await reader.get_by_id("keep"))["content"] == "beta"
        assert (await reader.get_by_id("extra"))["content"] == "gamma"

    @pytest.mark.asyncio
    async def test_delete_survives_a_failed_save_then_a_concurrent_commit(
        self, tmp_path
    ):
        writer = await self._make_initialized(tmp_path)
        other = await self._make_initialized(tmp_path)
        await self._upsert_and_flush(
            writer, {"victim": {"content": "to be deleted"}, "keep": {"content": "k"}}
        )

        await writer.delete(["victim"])
        restore = self.make_save_fail(writer, message="transient disk error")
        with pytest.raises(OSError, match="transient disk error"):
            await writer.index_done_callback()
        assert writer._pending_deletes == set(), "the queue was consumed by the flush"
        assert "victim" in writer._unsaved_deletes, (
            "the applied removal must be recorded in the redo log"
        )
        assert self.is_dirty(writer) is True
        restore()

        await self._upsert_and_flush(other, {"extra": {"content": "unrelated"}})
        assert writer.storage_updated.value is True

        assert await writer.index_done_callback() is True
        assert await writer.get_by_id("victim") is None
        assert writer._unsaved_deletes == {}, "a landed save clears the redo log"

        reader = await self._make_initialized(tmp_path)
        assert await reader.get_by_id("victim") is None
        assert (await reader.get_by_id("keep"))["content"] == "k"
        assert (await reader.get_by_id("extra"))["content"] == "unrelated"

    @pytest.mark.asyncio
    async def test_replay_preserves_a_newer_row_under_the_same_id(self, tmp_path):
        writer = await self._make_initialized(tmp_path)
        other = await self._make_initialized(tmp_path)
        await self._upsert_and_flush(writer, {"shared": {"content": "v1"}})

        await writer.delete(["shared"])
        restore = self.make_save_fail(writer, message="transient disk error")
        with pytest.raises(OSError, match="transient disk error"):
            await writer.index_done_callback()
        restore()

        # Another writer publishes a successor row inside the same second.
        await self._upsert_and_flush(other, {"shared": {"content": "v2"}})
        assert writer.storage_updated.value is True

        # Read paths hide only the removed version, so the successor stays readable.
        assert (await writer.get_by_id("shared"))["content"] == "v2"
        assert (await writer.get_by_ids(["shared"]))[0]["content"] == "v2"
        assert "shared" in await writer.get_vectors_by_ids(["shared"])

        assert await writer.index_done_callback() is True
        reader = await self._make_initialized(tmp_path)
        record = await reader.get_by_id("shared")
        assert record is not None and record["content"] == "v2", (
            "the replay must not remove the successor row"
        )

    @pytest.mark.asyncio
    async def test_finalize_reloads_before_retrying_a_delete_only_save(self, tmp_path):
        writer = await self._make_initialized(tmp_path)
        other = await self._make_initialized(tmp_path)
        await self._upsert_and_flush(
            writer,
            {"victim": {"content": "to be deleted"}, "keep": {"content": "beta"}},
        )

        await writer.delete(["victim"])
        restore = self.make_save_fail(writer, message="transient disk error")
        with pytest.raises(OSError, match="transient disk error"):
            await writer.index_done_callback()
        restore()
        assert self.is_dirty(writer) is True
        assert not writer._unsaved_upserts, "the dirty state is removals only"

        await self._upsert_and_flush(
            other, {"extra": {"content": "committed meanwhile"}}
        )

        await writer.finalize()

        reader = await self._make_initialized(tmp_path)
        assert (await reader.get_by_id("extra")) is not None, (
            "the delete-only retry save must not clobber the other writer's row"
        )
        assert (await reader.get_by_id("keep"))["content"] == "beta"
        assert await reader.get_by_id("victim") is None, (
            "the removal must still be persisted"
        )

    @pytest.mark.asyncio
    async def test_abort_keeps_the_delete_redo_log(self, tmp_path):
        writer = await self._make_initialized(tmp_path)
        other = await self._make_initialized(tmp_path)
        await self._upsert_and_flush(
            writer,
            {"victim": {"content": "to be deleted"}, "keep": {"content": "beta"}},
        )

        await writer.delete(["victim"])
        restore = self.make_save_fail(writer, message="transient disk error")
        with pytest.raises(OSError, match="transient disk error"):
            await writer.index_done_callback()
        restore()

        await writer.drop_pending_index_ops()
        assert "victim" in writer._unsaved_deletes, (
            "an abort must not discard the redo log"
        )

        await self._upsert_and_flush(other, {"extra": {"content": "unrelated"}})

        await writer.finalize()
        reader = await self._make_initialized(tmp_path)
        assert await reader.get_by_id("victim") is None
        assert (await reader.get_by_id("keep"))["content"] == "beta"
        assert (await reader.get_by_id("extra")) is not None

    @pytest.mark.asyncio
    async def test_identical_reupsert_survives_the_delete_replay(self, tmp_path):
        with self.frozen_clock(1_700_000_000):
            storage = await self._make_initialized(tmp_path)
            await self._upsert_and_flush(storage, {"row": {"content": "same"}})

            await storage.delete(["row"])
            restore = self.make_save_fail(storage, message="transient disk error")
            with pytest.raises(OSError, match="transient disk error"):
                await storage.index_done_callback()
            assert "row" in storage._unsaved_deletes

            await storage.upsert({"row": {"content": "same"}})
            with pytest.raises(OSError, match="transient disk error"):
                await storage.index_done_callback()
            assert "row" in storage._unsaved_deletes, (
                "the entry names the removed version, which the re-upsert is not"
            )
            record = await storage.get_by_id("row")
            assert record is not None and record["content"] == "same"
            restore()

        storage.storage_updated.value = True
        assert await storage.index_done_callback() is True
        reader = await self._make_initialized(tmp_path)
        record = await reader.get_by_id("row")
        assert record is not None and record["content"] == "same"
        assert self.materialized_count(reader) == 1, "exactly one row per id"

    @pytest.mark.asyncio
    async def test_finalize_skips_save_when_flush_changes_nothing(self, tmp_path):
        storage = await self._make_initialized(tmp_path)
        await self._upsert_and_flush(storage, {"keep": {"content": "kept"}})

        watcher = await self._make_initialized(tmp_path)
        assert watcher.storage_updated.value is False

        save_calls = self.make_save_fail_once(
            storage, message="no-op finalize must not save"
        )
        await storage.delete(["ghost"])
        await storage.finalize()

        assert save_calls == [], "no-op delete must not rewrite the snapshot"
        assert storage._pending_deletes == set()
        assert watcher.storage_updated.value is False, (
            "a no-op finalize must not broadcast a reload to other processes"
        )


class FingerprintFenceContractMixin:
    """Shared two-channel reload-fence contracts for Faiss and Nano."""

    absent_fingerprint: tuple = ()

    @pytest.fixture
    def multiprocess(self, monkeypatch):
        """Pretend peer processes exist, so the file channel is armed."""
        monkeypatch.setattr(file_fingerprint, "is_multiprocess_mode", lambda: True)

    @pytest.fixture
    def lost_notification(self, monkeypatch):
        """Every commit lands on disk and notifies nobody."""
        self.patch_lost_notification(monkeypatch)

    async def make_worker(self, tmp_path):
        raise NotImplementedError

    def ids_on_disk(self, worker) -> set[str]:
        raise NotImplementedError

    def patch_lost_notification(self, monkeypatch) -> None:
        raise NotImplementedError

    def patch_reload_failure(self, monkeypatch, broken: list[bool]) -> None:
        raise NotImplementedError

    @pytest.mark.asyncio
    async def test_a_stale_writer_replays_instead_of_overwriting_a_peer_commit(
        self, tmp_path, multiprocess, lost_notification
    ):
        worker_a = await self.make_worker(tmp_path)
        worker_b = await self.make_worker(tmp_path)
        try:
            await worker_b.upsert({"from_b": {"content": "b"}})

            await worker_a.upsert({"from_a": {"content": "a"}})
            assert await worker_a.index_done_callback() is True

            assert worker_b.storage_updated.value is False

            assert await worker_b.index_done_callback() is True
            assert worker_b._missed_notification_reloads == 1
            assert self.ids_on_disk(worker_b) == {"from_a", "from_b"}
        finally:
            await worker_a.finalize()
            await worker_b.finalize()

    @pytest.mark.asyncio
    async def test_reader_picks_up_a_peer_commit_it_was_never_told_about(
        self, tmp_path, multiprocess, lost_notification
    ):
        worker_a = await self.make_worker(tmp_path)
        worker_b = await self.make_worker(tmp_path)
        try:
            await worker_a.upsert({"from_a": {"content": "a"}})
            assert await worker_a.index_done_callback() is True
            assert worker_b.storage_updated.value is False

            assert await worker_b.get_by_id("from_a") is not None
            assert worker_b._missed_notification_reloads == 1
        finally:
            await worker_a.finalize()
            await worker_b.finalize()

    @pytest.mark.asyncio
    async def test_single_process_mode_ignores_a_divergent_file(
        self, tmp_path, lost_notification
    ):
        worker_a = await self.make_worker(tmp_path)
        worker_b = await self.make_worker(tmp_path)
        try:
            await worker_a.upsert({"from_a": {"content": "a"}})
            assert await worker_a.index_done_callback() is True

            assert worker_b._peer_commit_detected() is False
            assert await worker_b.get_by_id("from_a") is None
            assert worker_b._missed_notification_reloads == 0
        finally:
            await worker_a.finalize()
            await worker_b.finalize()

    @pytest.mark.asyncio
    async def test_drop_adopts_the_files_absence(self, tmp_path, multiprocess):
        worker = await self.make_worker(tmp_path)
        try:
            await worker.upsert({"n1": {"content": "x"}})
            assert await worker.index_done_callback() is True
            assert worker._loaded_fingerprint is not None

            assert (await worker.drop())["status"] == "success"
            assert worker._loaded_fingerprint == self.absent_fingerprint
            assert worker._peer_commit_detected() is False
        finally:
            await worker.finalize()

    @pytest.mark.asyncio
    async def test_a_failing_reload_does_not_recount_the_same_peer_commit(
        self, tmp_path, multiprocess, lost_notification, monkeypatch
    ):
        worker_a = await self.make_worker(tmp_path)
        worker_b = await self.make_worker(tmp_path)
        try:
            await worker_a.upsert({"from_a": {"content": "a"}})
            assert await worker_a.index_done_callback() is True
            assert worker_b.storage_updated.value is False

            broken = [True]
            self.patch_reload_failure(monkeypatch, broken)

            for _ in range(5):
                with pytest.raises(OSError, match="unreadable"):
                    await worker_b.get_by_id("from_a")
            assert worker_b._missed_notification_reloads == 1

            await worker_a.upsert({"from_a_again": {"content": "a2"}})
            assert await worker_a.index_done_callback() is True
            with pytest.raises(OSError, match="unreadable"):
                await worker_b.get_by_id("from_a")
            assert worker_b._missed_notification_reloads == 2

            broken[0] = False
            assert await worker_b.get_by_id("from_a_again") is not None
            assert worker_b._missed_notification_reloads == 2
        finally:
            await worker_a.finalize()
            await worker_b.finalize()

    @pytest.mark.asyncio
    async def test_a_recurring_state_is_counted_again_after_a_notified_reload(
        self, tmp_path, multiprocess, lost_notification
    ):
        worker_a = await self.make_worker(tmp_path)
        await worker_a.upsert({"x": {"content": "x"}})
        assert await worker_a.index_done_callback() is True

        worker_b = await self.make_worker(tmp_path)
        try:
            assert worker_b._missed_notification_reloads == 0

            assert (await worker_a.drop())["status"] == "success"
            assert await worker_b.get_by_id("x") is None
            assert worker_b._missed_notification_reloads == 1

            await worker_a.upsert({"y": {"content": "y"}})
            assert await worker_a.index_done_callback() is True
            worker_b.storage_updated.value = True
            assert await worker_b.get_by_id("y") is not None
            assert worker_b._missed_notification_reloads == 1

            assert (await worker_a.drop())["status"] == "success"
            assert await worker_b.get_by_id("y") is None
            assert worker_b._missed_notification_reloads == 2
        finally:
            await worker_b.finalize()
            await worker_a.finalize()

    @pytest.mark.asyncio
    async def test_an_unreadable_adoption_keeps_the_dedupe_marker(
        self, tmp_path, multiprocess, lost_notification
    ):
        worker_a = await self.make_worker(tmp_path)
        worker_b = await self.make_worker(tmp_path)
        try:
            await worker_a.upsert({"from_a": {"content": "a"}})
            assert await worker_a.index_done_callback() is True
            assert worker_b.storage_updated.value is False

            with pytest.MonkeyPatch.context() as broken:
                self.patch_reload_failure(broken, [True])
                with pytest.raises(OSError, match="unreadable"):
                    await worker_b.get_by_id("from_a")
            assert worker_b._missed_notification_reloads == 1

            original = worker_b._stat_fingerprint

            def unreadable_once():
                worker_b._stat_fingerprint = original
                return file_fingerprint.UNREADABLE

            worker_b._stat_fingerprint = unreadable_once
            assert await worker_b.get_by_id("from_a") is not None
            assert worker_b._loaded_fingerprint is None

            assert await worker_b.get_by_id("from_a") is not None
            assert worker_b._missed_notification_reloads == 1
        finally:
            await worker_b.finalize()
            await worker_a.finalize()


class ExactCountContractMixin:
    """Shared ``get_exact_count`` contract for file-backed vector storages."""

    def make_exact_count_storage(self, tmp_path):
        raise NotImplementedError

    @pytest.mark.asyncio
    async def test_count_refuses_unflushed_writes_then_counts_persisted_rows(
        self, tmp_path
    ):
        storage = self.make_exact_count_storage(tmp_path)
        await storage.initialize()
        assert await storage.get_exact_count() == 0

        await storage.upsert({"a": {"content": "alpha"}, "b": {"content": "beta"}})
        with pytest.raises(StorageCapabilityError, match="unflushed_writes"):
            await storage.get_exact_count()

        await storage.index_done_callback()
        assert await storage.get_exact_count() == 2
