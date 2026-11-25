"""Deferred-embedding and deferred-delete tests for ``FaissVectorDBStorage``.

Subclasses the shared file-backed vector contracts in
``tests/kg/_file_backed_vector_contracts.py`` and adds Faiss-specific rebuild,
skew, duplicate-fid, and failed-publication coverage.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from unittest.mock import patch

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from lightrag.kg.faiss_impl import FaissVectorDBStorage  # noqa: E402
from lightrag.kg.shared_storage import (  # noqa: E402
    finalize_share_data,
    initialize_share_data,
)
from lightrag.kg.write_seq import WRITE_SEQ_FIELD  # noqa: E402
from lightrag.utils import EmbeddingFunc  # noqa: E402
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


class _FaissDeferredAdapter:
    """Adapter binding the shared vector-contract mixins to ``FaissVectorDBStorage``."""

    def make_storage(
        self,
        tmp_path,
        embed,
        *,
        namespace: str = "test_vectors",
        meta_fields: set[str] | None = None,
    ) -> FaissVectorDBStorage:
        return FaissVectorDBStorage(
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

    def materialized_count(self, storage: FaissVectorDBStorage) -> int:
        return storage._index.ntotal

    def is_dirty(self, storage: FaissVectorDBStorage) -> bool:
        return storage._index_dirty

    def assert_consistent(self, storage: FaissVectorDBStorage) -> None:
        assert storage._index.ntotal == len(storage._id_to_meta), (
            f"index ntotal ({storage._index.ntotal}) != meta length "
            f"({len(storage._id_to_meta)})"
        )

    def make_save_fail(
        self, storage: FaissVectorDBStorage, *, message: str = "disk full"
    ):
        original = storage._save_faiss_index

        async def boom(_on_committed):
            raise OSError(message)

        storage._save_faiss_index = boom

        def restore():
            storage._save_faiss_index = original

        return restore

    def make_save_fail_once(
        self, storage: FaissVectorDBStorage, *, message: str = "boom"
    ) -> list[int]:
        original = storage._save_faiss_index
        calls: list[int] = []

        async def fail_once(on_committed):
            calls.append(1)
            if len(calls) == 1:
                raise OSError(message)
            await original(on_committed)

        storage._save_faiss_index = fail_once
        return calls

    @contextlib.contextmanager
    def frozen_clock(self, stamp: int):
        with patch("lightrag.kg.faiss_impl.time.time", return_value=stamp):
            yield

    def resident_row(self, storage: FaissVectorDBStorage, doc_id: str) -> dict:
        return next(
            meta for meta in storage._id_to_meta.values() if meta["__id__"] == doc_id
        )

    def strip_write_seq_on_disk(self, storage: FaissVectorDBStorage) -> None:
        with open(storage._meta_file, encoding="utf-8") as f:
            snapshot = json.load(f)
        stripped = [record.pop(WRITE_SEQ_FIELD, None) for record in snapshot.values()]
        assert any(token is not None for token in stripped), (
            "the token must reach the committed snapshot"
        )
        with open(storage._meta_file, "w", encoding="utf-8") as f:
            json.dump(snapshot, f)


class TestFaissDeferredEmbedding(_FaissDeferredAdapter, DeferredEmbeddingContractMixin):
    """Shared deferred-embedding contract plus Faiss-specific rebuild & skew tests."""

    @pytest.mark.asyncio
    async def test_index_done_callback_save_failure_raises(self, tmp_path):
        embed = CountingEmbed()
        storage = await self._make_initialized(tmp_path, embed)

        await storage.upsert({"id1": {"content": "alpha"}})
        restore = self.make_save_fail(storage, message="save boom")

        with pytest.raises(OSError, match="save boom"):
            await storage.index_done_callback()

        assert storage._pending_upserts == {}, "flush succeeded so pending is empty"
        assert storage._index_dirty is True, "save failure preserves dirty for retry"
        assert storage._index.ntotal == 1, "materialized state is preserved"
        self.assert_consistent(storage)

        restore()
        embed_before = embed.call_count
        await storage.finalize()
        assert embed.call_count == embed_before, "save retry must not re-embed"
        assert storage._index_dirty is False

        reader = await self._make_initialized(tmp_path, embed)
        hit = await reader.get_by_id("id1")
        assert hit is not None and hit["id"] == "id1"
        self.assert_consistent(reader)

    @pytest.mark.asyncio
    async def test_reload_warns_on_index_meta_skew(self, tmp_path, lightrag_caplog):
        """An ``index > meta`` skew on disk logs a warning and is not auto-repaired."""
        embed = CountingEmbed()
        writer = await self._make_initialized(tmp_path, embed)

        await writer.upsert({"id1": {"content": "alpha"}, "id2": {"content": "beta"}})
        await writer.index_done_callback()

        with open(writer._meta_file, encoding="utf-8") as f:
            meta = json.load(f)
        assert len(meta) == 2
        del meta[next(iter(meta))]
        with open(writer._meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        lightrag_caplog.clear()
        with lightrag_caplog.at_level(logging.WARNING, logger="lightrag"):
            reader = await self._make_initialized(tmp_path, embed)

        assert reader._index.ntotal == 2
        assert len(reader._id_to_meta) == 1
        skew_messages = [
            rec.message
            for rec in lightrag_caplog.records
            if "skew" in rec.message or "index > meta" in rec.message
        ]
        assert skew_messages, (
            f"expected an index>meta skew warning; got: "
            f"{[r.message for r in lightrag_caplog.records]}"
        )
        assert os.path.exists(writer._faiss_index_file)
        assert os.path.exists(writer._meta_file)

    @pytest.mark.asyncio
    async def test_query_skips_orphan_faiss_hits(self, tmp_path):
        """Orphan vectors from an ``index > meta`` skew are filtered out of query hits."""
        storage = await self._make_initialized(tmp_path)

        await storage.upsert({"id1": {"content": "alpha"}, "id2": {"content": "beta"}})
        await storage.index_done_callback()
        assert storage._index.ntotal == 2

        orphan_fid = next(iter(storage._id_to_meta))
        del storage._id_to_meta[orphan_fid]
        assert storage._index.ntotal == 2
        assert len(storage._id_to_meta) == 1

        results = await storage.query("anything", top_k=5)
        for row in results:
            assert row["id"] is not None, f"orphan hit leaked: {row}"
        surviving_id = next(iter(storage._id_to_meta.values()))["__id__"]
        assert any(r["id"] == surviving_id for r in results)

    @pytest.mark.asyncio
    async def test_reupsert_cleans_duplicate_custom_id_rows(self, tmp_path):
        """Re-upsert + flush collapses duplicate fids; delete removes every copy."""
        storage = await self._make_initialized(tmp_path)

        matrix = np.array([[1.0] * DIM, [2.0] * DIM], dtype=np.float32)
        faiss.normalize_L2(matrix)
        storage._index.add(matrix)
        storage._id_to_meta[0] = {
            "__id__": "dup",
            "__created_at__": 1,
            "content": "v1",
            "__vector__": matrix[0].tolist(),
        }
        storage._id_to_meta[1] = {
            "__id__": "dup",
            "__created_at__": 1,
            "content": "v2",
            "__vector__": matrix[1].tolist(),
        }
        self.assert_consistent(storage)
        assert storage._find_faiss_ids_by_custom_id("dup") == [0, 1]

        await storage.upsert({"dup": {"content": "v3"}})
        await storage.index_done_callback()

        assert storage._index.ntotal == 1, "flush rebuild must drop both duplicates"
        assert len(storage._id_to_meta) == 1
        assert storage._find_faiss_ids_by_custom_id("dup") == list(
            storage._id_to_meta.keys()
        )
        hit = await storage.get_by_id("dup")
        assert hit is not None and hit["content"] == "v3"
        self.assert_consistent(storage)

        matrix2 = np.array([[3.0] * DIM, [4.0] * DIM], dtype=np.float32)
        faiss.normalize_L2(matrix2)
        storage._index.add(matrix2)
        next_fid = max(storage._id_to_meta) + 1
        storage._id_to_meta[next_fid] = {
            "__id__": "dup",
            "__created_at__": 2,
            "content": "dup-a",
            "__vector__": matrix2[0].tolist(),
        }
        storage._id_to_meta[next_fid + 1] = {
            "__id__": "dup",
            "__created_at__": 2,
            "content": "dup-b",
            "__vector__": matrix2[1].tolist(),
        }
        assert len(storage._find_faiss_ids_by_custom_id("dup")) == 3

        await storage.delete(["dup"])
        assert storage._find_faiss_ids_by_custom_id("dup") == [0, 1, 2]
        assert await storage.index_done_callback() is True
        assert storage._find_faiss_ids_by_custom_id("dup") == []
        assert storage._index.ntotal == 0
        self.assert_consistent(storage)

    @pytest.mark.asyncio
    async def test_flush_recovers_from_index_add_failure_without_re_embedding(
        self, tmp_path
    ):
        """A mid-flush ``index.add`` failure keeps cached pending vectors for retry."""

        class _AddFailsOnce:
            """Shims a real faiss index, raising once then unwrapping itself."""

            def __init__(self, storage, real):
                self._storage = storage
                self._real = real
                self._calls = 0

            def __getattr__(self, name):
                return getattr(self._real, name)

            def add(self, arr):
                self._calls += 1
                if self._calls == 1:
                    raise RuntimeError("add boom")
                result = self._real.add(arr)
                self._storage._index = self._real
                return result

        embed = CountingEmbed()
        storage = await self._make_initialized(tmp_path, embed)
        await storage.upsert({"id1": {"content": "alpha"}})

        real_index = storage._index
        storage._index = _AddFailsOnce(storage, real_index)

        with pytest.raises(RuntimeError, match="add boom"):
            await storage.index_done_callback()

        assert embed.call_count == 1
        assert "id1" in storage._pending_upserts
        assert storage._pending_upserts["id1"].vector is not None
        assert storage._index_dirty is False
        assert storage._index.ntotal == 0

        await storage.finalize()

        assert embed.call_count == 1, "retry must reuse cached vectors, not re-embed"
        assert storage._index is real_index
        assert storage._index.ntotal == 1
        assert storage._pending_upserts == {}
        assert storage._index_dirty is False
        self.assert_consistent(storage)

        reader = await self._make_initialized(tmp_path, embed)
        hit = await reader.get_by_id("id1")
        assert hit is not None and hit["content"] == "alpha"

    @pytest.mark.asyncio
    async def test_flush_add_failure_leaves_the_superseded_row_intact(
        self, tmp_path, monkeypatch
    ):
        """A failure preparing an overwrite must leave the prior row intact."""
        storage = await self._make_initialized(tmp_path)
        await self._upsert_and_flush(
            storage, {"doomed": {"content": "x"}, "keep": {"content": "v1"}}
        )
        self.assert_consistent(storage)

        await storage.delete(["doomed"])
        await storage.upsert({"keep": {"content": "v2"}})

        def boom(*_args, **_kwargs):
            raise RuntimeError("faiss add boom")

        monkeypatch.setattr(np, "vstack", boom)
        with pytest.raises(RuntimeError, match="faiss add boom"):
            await storage.index_done_callback()
        monkeypatch.undo()
        self.assert_consistent(storage)

        await storage.drop_pending_index_ops()
        await storage.finalize()

        reloaded = await self._make_initialized(tmp_path)
        survivor = await reloaded.get_by_id("keep")
        assert survivor is not None and survivor["content"] == "v1"
        assert await reloaded.get_by_id("doomed") is None

    @pytest.mark.asyncio
    async def test_overwrite_is_one_rebuild_and_all_or_nothing(
        self, tmp_path, monkeypatch
    ):
        storage = await self._make_initialized(tmp_path)
        await self._upsert_and_flush(
            storage, {"keep": {"content": "v1"}, "other": {"content": "o"}}
        )

        rebuilds = []
        original = storage._rebuild_index_locked

        def counting(drop_fids, add_records=()):
            rebuilds.append((list(drop_fids), len(add_records)))
            return original(drop_fids, add_records)

        monkeypatch.setattr(storage, "_rebuild_index_locked", counting)
        await storage.upsert({"keep": {"content": "v2"}})
        assert await storage.index_done_callback() is True
        monkeypatch.undo()

        assert len(rebuilds) == 1, "an overwrite is a single rebuild"
        assert rebuilds[0][1] == 1
        assert len(storage._find_faiss_ids_by_custom_id("keep")) == 1
        assert (await storage.get_by_id("keep"))["content"] == "v2"
        self.assert_consistent(storage)

        await storage.upsert({"keep": {"content": "v3"}})

        def boom(_drop_fids, _add_records=()):
            raise RuntimeError("faiss rebuild boom")

        monkeypatch.setattr(storage, "_rebuild_index_locked", boom)
        with pytest.raises(RuntimeError, match="faiss rebuild boom"):
            await storage.index_done_callback()
        monkeypatch.undo()

        fids = storage._find_faiss_ids_by_custom_id("keep")
        assert len(fids) == 1
        assert storage._id_to_meta[fids[0]]["content"] == "v2"
        assert storage._index_dirty is False
        self.assert_consistent(storage)

        assert "keep" in storage._pending_upserts
        assert await storage.index_done_callback() is True
        assert len(storage._find_faiss_ids_by_custom_id("keep")) == 1
        assert (await storage.get_by_id("keep"))["content"] == "v3"
        self.assert_consistent(storage)

    @pytest.mark.asyncio
    async def test_insert_only_flush_does_not_rebuild(self, tmp_path, monkeypatch):
        storage = await self._make_initialized(tmp_path)
        await self._upsert_and_flush(
            storage, {f"row{i}": {"content": f"c{i}"} for i in range(3)}
        )

        rebuilds = []
        original = storage._rebuild_index_locked

        def counting(drop_fids, add_records=()):
            rebuilds.append(list(drop_fids))
            return original(drop_fids, add_records)

        monkeypatch.setattr(storage, "_rebuild_index_locked", counting)
        await storage.upsert({"fresh": {"content": "brand new"}})
        assert await storage.index_done_callback() is True
        monkeypatch.undo()

        assert rebuilds == [], "an insert-only flush must not rebuild the index"
        assert storage._index.ntotal == 4
        assert (await storage.get_by_id("fresh"))["content"] == "brand new"
        self.assert_consistent(storage)

    @pytest.mark.asyncio
    async def test_rebuild_failure_leaves_index_and_meta_untouched(
        self, tmp_path, monkeypatch
    ):
        storage = await self._make_initialized(tmp_path)
        await self._upsert_and_flush(
            storage, {f"row{i}": {"content": f"c{i}"} for i in range(4)}
        )
        self.assert_consistent(storage)

        await storage.upsert({"row0": {"content": "new"}})

        real_cls = faiss.IndexFlatIP
        built = {"n": 0}

        def failing_factory(dim):
            index = real_cls(dim)
            built["n"] += 1
            if built["n"] == 1:

                def boom(_arr):
                    raise RuntimeError("rebuild add boom")

                index.add = boom
            return index

        monkeypatch.setattr(faiss, "IndexFlatIP", failing_factory)
        with pytest.raises(RuntimeError, match="rebuild add boom"):
            await storage.index_done_callback()
        monkeypatch.undo()

        self.assert_consistent(storage)
        assert len(storage._find_faiss_ids_by_custom_id("row0")) == 1

        await storage.drop_pending_index_ops()
        await storage.finalize()

        reloaded = await self._make_initialized(tmp_path)
        for i in range(4):
            assert await reloaded.get_by_id(f"row{i}") is not None
        self.assert_consistent(reloaded)

    @pytest.mark.asyncio
    async def test_repeated_failed_saves_do_not_rebuild_each_time(self, tmp_path):
        embed = CountingEmbed()
        storage = await self._make_initialized(tmp_path, embed)

        await storage.upsert({"id1": {"content": "alpha"}})
        restore = self.make_save_fail(storage)
        with pytest.raises(OSError):
            await storage.index_done_callback()

        rebuilds: list[tuple[list[int], int]] = []
        real_rebuild = storage._rebuild_index_locked

        def spy(drop_fids, add_records=()):
            rebuilds.append((list(drop_fids), len(add_records)))
            return real_rebuild(drop_fids, add_records)

        storage._rebuild_index_locked = spy

        with pytest.raises(OSError):
            await storage.index_done_callback()
        assert rebuilds == [], "no reload happened, so there is nothing to redo"

        restore()
        assert await storage.index_done_callback() is True
        assert rebuilds == [], "the durable retry must not rebuild either"
        assert not storage._unsaved_upserts
        assert embed.call_count == 1
        self.assert_consistent(storage)
        assert (await storage.get_by_id("id1"))["content"] == "alpha"

    @pytest.mark.asyncio
    async def test_reads_compare_every_duplicate_row_before_serving_a_redo_entry(
        self, tmp_path
    ):
        storage = await self._make_initialized(tmp_path)
        other = await self._make_initialized(tmp_path)

        matrix = np.zeros((2, DIM), dtype=np.float32)
        matrix[0][0] = 1.0
        matrix[1][1] = 1.0
        storage._index.add(matrix)
        storage._id_to_meta[0] = {
            "__id__": "dup",
            "__created_at__": 1,
            "content": "v1-older",
            "__vector__": matrix[0].tolist(),
        }
        storage._id_to_meta[1] = {
            "__id__": "dup",
            "__created_at__": 3,
            "content": "v2-newer",
            "__vector__": matrix[1].tolist(),
        }
        assert await storage.index_done_callback() is True

        with self.frozen_clock(2):
            await storage.upsert({"dup": {"content": "ours-stale"}})
        restore = self.make_save_fail(storage)
        with pytest.raises(OSError):
            await storage.index_done_callback()
        restore()
        assert "dup" in storage._unsaved_upserts

        await self._upsert_and_flush(other, {"unrelated": {"content": "other"}})

        got = await storage.get_by_id("dup")
        fids = storage._find_faiss_ids_by_custom_id("dup")
        assert len(fids) == 2
        assert storage._id_to_meta[fids[0]]["__created_at__"] == 1
        assert got["content"] == "v2-newer"
        assert (await storage.get_by_ids(["dup"]))[0]["content"] == "v2-newer"
        vectors = await storage.get_vectors_by_ids(["dup"])
        assert vectors["dup"][1] == pytest.approx(1.0)

        assert await storage.index_done_callback() is True
        assert "dup" not in storage._unsaved_upserts
        assert (await storage.get_by_id("dup"))["content"] == "v2-newer"

    @pytest.mark.asyncio
    async def test_a_failed_publication_keeps_the_redo_log(self, tmp_path, monkeypatch):
        storage = await self._make_initialized(tmp_path)
        await storage.upsert({"idA": {"content": "alpha"}})

        async def failing_set_all_update_flags(namespace, workspace=None):
            raise RuntimeError("shared-storage manager is down")

        monkeypatch.setattr(
            "lightrag.kg.faiss_impl.set_all_update_flags",
            failing_set_all_update_flags,
        )
        assert await storage.index_done_callback() is True
        assert "idA" in storage._unsaved_upserts
        assert storage._index_dirty is True

        monkeypatch.undo()
        assert await storage.index_done_callback() is True
        assert storage._unsaved_upserts == {}
        self.assert_consistent(storage)

    @pytest.mark.asyncio
    async def test_rows_lost_to_an_unnotified_peer_are_replayed_back(
        self, tmp_path, monkeypatch
    ):
        writer = await self._make_initialized(tmp_path)
        other = await self._make_initialized(tmp_path)

        await writer.upsert({"idA": {"content": "ours"}})

        async def failing_set_all_update_flags(namespace, workspace=None):
            raise RuntimeError("shared-storage manager is down")

        monkeypatch.setattr(
            "lightrag.kg.faiss_impl.set_all_update_flags",
            failing_set_all_update_flags,
        )
        assert await writer.index_done_callback() is True
        monkeypatch.undo()

        await self._upsert_and_flush(other, {"idB": {"content": "theirs"}})
        assert await other.get_by_id("idA") is None

        assert await writer.index_done_callback() is True

        reader = await self._make_initialized(tmp_path)
        assert (await reader.get_by_id("idA"))["content"] == "ours"
        assert (await reader.get_by_id("idB"))["content"] == "theirs"
        self.assert_consistent(reader)

    @pytest.mark.asyncio
    async def test_a_broken_sink_cannot_turn_a_landed_save_into_a_failure(
        self, tmp_path, monkeypatch
    ):
        storage = await self._make_initialized(tmp_path)
        await storage.upsert({"idA": {"content": "alpha"}})

        async def failing_set_all_update_flags(namespace, workspace=None):
            raise RuntimeError("shared-storage manager is down")

        def log_boom(msg):
            raise RuntimeError("log sink boom")

        monkeypatch.setattr(
            "lightrag.kg.faiss_impl.set_all_update_flags",
            failing_set_all_update_flags,
        )
        monkeypatch.setattr("lightrag.kg.faiss_impl.logger.error", log_boom)

        assert await storage.index_done_callback() is True
        assert os.path.exists(storage._faiss_index_file)
        assert os.path.exists(storage._meta_file)
        assert "idA" in storage._unsaved_upserts
        self.assert_consistent(storage)


class TestFaissDeferredDelete(_FaissDeferredAdapter, DeferredDeleteContractMixin):
    """Shared deferred-delete contract plus Faiss-specific rebuild & dup-fid tests."""

    @pytest.mark.asyncio
    async def test_delete_is_deferred_until_one_batched_flush(
        self, tmp_path, monkeypatch
    ):
        storage = await self._make_initialized(tmp_path)
        await self._upsert_and_flush(
            storage, {f"rel-{i}": {"content": f"relation {i}"} for i in range(4)}
        )
        assert len(storage._id_to_meta) == 4

        rebuild_calls = []
        original_rebuild = storage._remove_faiss_ids_locked

        def counting_rebuild(fid_list):
            rebuild_calls.append(list(fid_list))
            return original_rebuild(fid_list)

        monkeypatch.setattr(storage, "_remove_faiss_ids_locked", counting_rebuild)

        for i in range(4):
            await storage.delete([f"rel-{i}", f"rel-{i}-reverse"])

        assert rebuild_calls == [], "delete must not rebuild the index eagerly"
        assert len(storage._id_to_meta) == 4

        assert await storage.index_done_callback() is True
        assert len(rebuild_calls) == 1, "the whole queued batch is one rebuild"
        assert len(storage._id_to_meta) == 0
        assert await storage.get_by_ids([f"rel-{i}" for i in range(4)]) == [None] * 4

    @pytest.mark.asyncio
    async def test_delete_cancels_pending_upsert_without_embedding(self, tmp_path):
        embed = CountingEmbed()
        storage = await self._make_initialized(tmp_path, embed)

        await storage.upsert({"doomed": {"content": "never embedded"}})
        await storage.delete(["doomed"])

        assert await storage.index_done_callback() is True
        assert "never embedded" not in embed.embedded_texts
        assert embed.call_count == 0
        assert await storage.get_by_id("doomed") is None

    @pytest.mark.asyncio
    async def test_finalize_applies_deletes_without_upserts(self, tmp_path):
        storage = await self._make_initialized(tmp_path)
        await self._upsert_and_flush(storage, {"x": {"content": "ex"}})

        await storage.delete(["x"])
        await storage.finalize()

        reloaded = await self._make_initialized(tmp_path)
        assert await reloaded.get_by_id("x") is None

    @pytest.mark.asyncio
    async def test_drop_pending_index_ops_discards_queued_deletes(self, tmp_path):
        storage = await self._make_initialized(tmp_path)
        await self._upsert_and_flush(storage, {"keep": {"content": "kept"}})

        await storage.delete(["keep"])
        await storage.drop_pending_index_ops()
        assert await storage.index_done_callback() is True

        assert (await storage.get_by_id("keep")) is not None

    @pytest.mark.asyncio
    async def test_failed_delete_rebuild_keeps_tombstones_for_retry(
        self, tmp_path, monkeypatch
    ):
        storage = await self._make_initialized(tmp_path)
        await self._upsert_and_flush(
            storage, {"x": {"content": "ex"}, "y": {"content": "wy"}}
        )
        assert len(storage._id_to_meta) == 2

        await storage.delete(["x", "y"])

        def failing_rebuild(fid_list):
            raise RuntimeError("faiss rebuild boom")

        monkeypatch.setattr(storage, "_remove_faiss_ids_locked", failing_rebuild)

        with pytest.raises(RuntimeError, match="faiss rebuild boom"):
            await storage.index_done_callback()

        assert storage._pending_deletes == {"x", "y"}
        assert storage._unsaved_deletes == {}
        assert len(storage._id_to_meta) == 2
        assert storage._index_dirty is False

        monkeypatch.undo()

        assert await storage.index_done_callback() is True
        assert storage._pending_deletes == set()
        assert len(storage._id_to_meta) == 0
        assert await storage.get_by_ids(["x", "y"]) == [None, None]

        reloaded = await self._make_initialized(tmp_path)
        assert await reloaded.get_by_ids(["x", "y"]) == [None, None]

    @pytest.mark.asyncio
    async def test_eager_entity_relation_removal_survives_reload(self, tmp_path):
        storage = await self._make_initialized(
            tmp_path, meta_fields={"content", "src_id", "tgt_id"}
        )
        await self._upsert_and_flush(
            storage, {"rel-1": {"content": "r1", "src_id": "E", "tgt_id": "F"}}
        )

        await storage.delete_entity_relation("E")
        assert "rel-1" in storage._unsaved_deletes
        restore = self.make_save_fail(storage, message="transient disk error")
        with pytest.raises(OSError, match="transient disk error"):
            await storage.index_done_callback()
        restore()

        other = await self._make_initialized(tmp_path)
        await self._upsert_and_flush(other, {"extra": {"content": "unrelated"}})

        assert await storage.index_done_callback() is True
        reloaded = await self._make_initialized(tmp_path)
        assert await reloaded.get_by_id("rel-1") is None

    @pytest.mark.asyncio
    async def test_reads_show_the_duplicate_a_delete_replay_preserves(self, tmp_path):
        storage = await self._make_initialized(tmp_path)
        await self._upsert_and_flush(storage, {"dup": {"content": "removed"}})
        removed_row = dict(storage._id_to_meta[0])

        await storage.delete(["dup"])
        restore = self.make_save_fail(storage, message="transient disk error")
        with pytest.raises(OSError, match="transient disk error"):
            await storage.index_done_callback()
        assert storage._unsaved_deletes["dup"] == {
            storage._row_fingerprint(removed_row)
        }
        restore()

        matrix = np.zeros((2, DIM), dtype=np.float32)
        matrix[0][0] = 1.0
        matrix[1][1] = 1.0
        storage._index.add(matrix)
        storage._id_to_meta[0] = {**removed_row, "__vector__": matrix[0].tolist()}
        storage._id_to_meta[1] = {
            "__id__": "dup",
            "__created_at__": removed_row["__created_at__"] + 5,
            "content": "survivor",
            "__vector__": matrix[1].tolist(),
        }

        got = await storage.get_by_id("dup")
        assert got is not None and got["content"] == "survivor"
        assert (await storage.get_by_ids(["dup"]))[0]["content"] == "survivor"
        vectors = await storage.get_vectors_by_ids(["dup"])
        assert vectors["dup"][1] == pytest.approx(1.0)

        assert await storage.index_done_callback() is True
        assert storage._find_faiss_ids_by_custom_id("dup") == [0]
        assert (await storage.get_by_id("dup"))["content"] == "survivor"

    @pytest.mark.asyncio
    async def test_replay_removes_every_duplicate_row_under_one_id(self, tmp_path):
        storage = await self._make_initialized(tmp_path)

        matrix = np.array([[1.0] * DIM, [2.0] * DIM], dtype=np.float32)
        faiss.normalize_L2(matrix)
        storage._index.add(matrix)
        storage._id_to_meta[0] = {
            "__id__": "dup",
            "__created_at__": 1,
            "content": "v1",
            "__vector__": matrix[0].tolist(),
        }
        storage._id_to_meta[1] = {
            "__id__": "dup",
            "__created_at__": 1,
            "content": "v2",
            "__vector__": matrix[1].tolist(),
        }
        assert await storage.index_done_callback() is True
        assert len(storage._find_faiss_ids_by_custom_id("dup")) == 2

        await storage.delete(["dup"])
        restore = self.make_save_fail(storage, message="transient disk error")
        with pytest.raises(OSError, match="transient disk error"):
            await storage.index_done_callback()
        assert storage._find_faiss_ids_by_custom_id("dup") == []
        assert len(storage._unsaved_deletes["dup"]) == 2
        restore()

        other = await self._make_initialized(tmp_path)
        await self._upsert_and_flush(other, {"extra": {"content": "unrelated"}})

        assert await storage.index_done_callback() is True
        assert storage._find_faiss_ids_by_custom_id("dup") == []

        reloaded = await self._make_initialized(tmp_path)
        assert reloaded._find_faiss_ids_by_custom_id("dup") == []
        assert (await reloaded.get_by_id("extra")) is not None
