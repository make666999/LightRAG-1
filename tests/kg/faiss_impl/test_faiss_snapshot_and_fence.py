"""Corrupt-snapshot, fingerprint-fence, and exact-count tests for Faiss.

Subclasses the shared fingerprint-fence and exact-count contract mixins in
``tests/kg/_file_backed_vector_contracts.py`` and retains all Faiss-specific
two-file torn-pair, marker corruption, backup-permission, and index/meta-skew
checks.
"""

from __future__ import annotations

import builtins
import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

faiss = pytest.importorskip("faiss")

from lightrag.exceptions import (  # noqa: E402
    CorruptStorageSnapshotError,
    StorageCapabilityError,
)
from lightrag.kg import faiss_impl  # noqa: E402
from lightrag.kg.faiss_impl import FaissVectorDBStorage  # noqa: E402
from lightrag.kg.shared_storage import (  # noqa: E402
    finalize_share_data,
    initialize_share_data,
)
from lightrag.utils import EmbeddingFunc  # noqa: E402
from tests.kg._file_backed_vector_contracts import (  # noqa: E402
    DIM,
    CountingEmbed,
    ExactCountContractMixin,
    FingerprintFenceContractMixin,
    NoLock,
    truncate_file,
)

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


def _make_storage(tmp_path) -> FaissVectorDBStorage:
    return FaissVectorDBStorage(
        namespace="test_vectors",
        workspace="ws",
        global_config={
            "working_dir": str(tmp_path),
            "embedding_batch_num": 32,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=DIM, max_token_size=512, func=CountingEmbed()
        ),
        meta_fields={"content"},
    )


async def _seeded_storage(tmp_path) -> FaissVectorDBStorage:
    storage = _make_storage(tmp_path)
    await storage.initialize()
    await storage.upsert({"v1": {"content": "hello"}})
    await storage.index_done_callback()
    assert Path(storage._faiss_index_file).exists()
    assert Path(storage._meta_file).exists()
    return storage


def _ids_on_disk(worker: FaissVectorDBStorage) -> set[str]:
    with open(worker._meta_file, encoding="utf-8") as fh:
        meta = json.load(fh)
    rows = meta["data"] if isinstance(meta, dict) and "data" in meta else meta
    if isinstance(rows, dict):
        rows = rows.values()
    return {row["__id__"] for row in rows}


# ---------------------------------------------------------------------------
# Corrupt snapshot refusal & offline backup/recovery
# ---------------------------------------------------------------------------


class TestFaissCorruptSnapshot:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("target", ["index", "meta"])
    async def test_initialize_refuses_a_corrupt_pair(self, tmp_path, target):
        storage = await _seeded_storage(tmp_path)
        corrupt_file = (
            storage._faiss_index_file if target == "index" else storage._meta_file
        )
        truncate_file(corrupt_file)
        on_disk = Path(corrupt_file).read_bytes()

        fresh = _make_storage(tmp_path)
        with pytest.raises(CorruptStorageSnapshotError) as exc_info:
            await fresh.initialize()

        assert "lightrag-rebuild-vdb" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None
        assert exc_info.value.container == corrupt_file
        assert set(exc_info.value.artifacts) == {
            storage._faiss_index_file,
            storage._meta_file,
            storage._vector_space_file,
        }
        assert Path(corrupt_file).read_bytes() == on_disk

    @pytest.mark.asyncio
    @pytest.mark.parametrize("payload", ['["a list, not an object"]'])
    async def test_unparsable_metadata_names_the_metadata_file(self, tmp_path, payload):
        storage = await _seeded_storage(tmp_path)
        Path(storage._meta_file).write_text(payload, encoding="utf-8")
        index_bytes = Path(storage._faiss_index_file).read_bytes()

        fresh = _make_storage(tmp_path)
        with pytest.raises(CorruptStorageSnapshotError) as exc_info:
            await fresh.initialize()

        assert exc_info.value.container == storage._meta_file
        assert set(exc_info.value.artifacts) == {
            storage._faiss_index_file,
            storage._meta_file,
            storage._vector_space_file,
        }
        assert Path(storage._faiss_index_file).read_bytes() == index_bytes
        assert Path(storage._meta_file).read_text(encoding="utf-8") == payload

    @pytest.mark.asyncio
    async def test_a_heap_that_ran_out_is_not_a_corrupt_snapshot(
        self, tmp_path, monkeypatch
    ):
        storage = await _seeded_storage(tmp_path)

        def out_of_memory(*args, **kwargs):
            raise MemoryError("cannot allocate the metadata")

        monkeypatch.setattr(json, "load", out_of_memory)

        fresh = _make_storage(tmp_path)
        with pytest.raises(MemoryError):
            await fresh.initialize()
        assert Path(storage._meta_file).exists()
        assert Path(storage._faiss_index_file).exists()

    @pytest.mark.asyncio
    async def test_metadata_nested_past_the_parser_names_the_metadata(
        self, tmp_path, monkeypatch
    ):
        storage = await _seeded_storage(tmp_path)

        def too_deep(*args, **kwargs):
            raise RecursionError("maximum recursion depth exceeded")

        monkeypatch.setattr(json, "load", too_deep)

        fresh = _make_storage(tmp_path)
        with pytest.raises(CorruptStorageSnapshotError) as exc_info:
            await fresh.initialize()
        assert exc_info.value.container == storage._meta_file

    @pytest.mark.asyncio
    async def test_initialize_refuses_an_index_whose_metadata_vanished(self, tmp_path):
        storage = await _seeded_storage(tmp_path)
        Path(storage._meta_file).unlink()

        fresh = _make_storage(tmp_path)
        with pytest.raises(CorruptStorageSnapshotError) as exc_info:
            await fresh.initialize()

        assert exc_info.value.container == storage._meta_file
        assert "missing beside a present index" in str(exc_info.value)
        assert Path(storage._faiss_index_file).exists()

    @pytest.mark.asyncio
    async def test_a_save_never_publishes_emptiness_over_a_corrupt_pair(self, tmp_path):
        storage = await _seeded_storage(tmp_path)
        truncate_file(storage._faiss_index_file)
        index_bytes = Path(storage._faiss_index_file).read_bytes()
        meta_bytes = Path(storage._meta_file).read_bytes()
        storage.storage_updated.value = True

        await storage.upsert({"v2": {"content": "world"}})
        with pytest.raises(CorruptStorageSnapshotError):
            await storage.index_done_callback()

        assert Path(storage._faiss_index_file).read_bytes() == index_bytes
        assert Path(storage._meta_file).read_bytes() == meta_bytes
        with pytest.raises(CorruptStorageSnapshotError):
            await storage.query("hello", top_k=1, query_embedding=[1.0] * DIM)

    @pytest.mark.asyncio
    async def test_a_read_error_is_not_reported_as_corruption(
        self, tmp_path, monkeypatch
    ):
        storage = await _seeded_storage(tmp_path)
        real_open = builtins.open

        def _deny(path, *args, **kwargs):
            if str(path) == storage._meta_file:
                raise PermissionError(13, "Permission denied", str(path))
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", _deny)
        fresh = _make_storage(tmp_path)
        with pytest.raises(PermissionError):
            await fresh.initialize()

    @pytest.mark.asyncio
    async def test_offline_recovery_backs_up_every_file_and_rebuilds(self, tmp_path):
        from lightrag.tools.rebuild_vdb import RebuildTool

        storage = await _seeded_storage(tmp_path)
        truncate_file(storage._faiss_index_file)
        originals = {
            path: Path(path).read_bytes()
            for path in (
                storage._faiss_index_file,
                storage._meta_file,
                storage._vector_space_file,
            )
        }
        refused = _make_storage(tmp_path)
        with pytest.raises(CorruptStorageSnapshotError) as caught:
            await refused.initialize()

        tool = RebuildTool()
        tool.entities_vdb = refused
        tool.corrupt_vdbs = {"entities": caught.value}
        tool.incompatible_vdbs = {"entities": str(caught.value)}
        await tool.recover_incompatible(["entities"])

        backups = sorted(Path(storage._faiss_index_file).parent.glob("*.corrupt-*"))
        assert len(backups) == 3
        tokens = {path.name.rsplit("-", 1)[1] for path in backups}
        assert len(tokens) == 1
        restored = {path.name.rsplit(".corrupt-", 1)[0] for path in backups}
        assert restored == {Path(path).name for path in originals}
        for backup in backups:
            original = backup.parent / backup.name.rsplit(".corrupt-", 1)[0]
            assert backup.read_bytes() == originals[str(original)]

        await refused.upsert({"restored": {"content": "hello"}})
        await refused.index_done_callback()
        fresh = _make_storage(tmp_path)
        await fresh.initialize()
        assert (await fresh.get_by_id("restored"))["content"] == "hello"

    @pytest.mark.asyncio
    async def test_offline_recovery_skips_a_file_the_torn_pair_never_had(
        self, tmp_path
    ):
        from lightrag.tools.rebuild_vdb import RebuildTool

        storage = await _seeded_storage(tmp_path)
        Path(storage._meta_file).unlink()
        refused = _make_storage(tmp_path)
        with pytest.raises(CorruptStorageSnapshotError) as caught:
            await refused.initialize()

        tool = RebuildTool()
        tool.entities_vdb = refused
        tool.corrupt_vdbs = {"entities": caught.value}
        tool.incompatible_vdbs = {"entities": str(caught.value)}
        await tool.recover_incompatible(["entities"])

        backups = sorted(Path(storage._faiss_index_file).parent.glob("*.corrupt-*"))
        preserved = {path.name.rsplit(".corrupt-", 1)[0] for path in backups}
        assert preserved == {
            Path(storage._faiss_index_file).name,
            Path(storage._vector_space_file).name,
        }
        assert Path(storage._meta_file).name not in preserved

    @pytest.mark.asyncio
    async def test_a_marker_that_exists_but_cannot_be_read_is_not_absent(
        self, tmp_path
    ):
        storage = await _seeded_storage(tmp_path)
        marker = Path(storage._vector_space_file)
        assert marker.exists()
        marker.write_bytes(b'{"model": "bge-m3"')

        fresh = _make_storage(tmp_path)
        with pytest.raises(CorruptStorageSnapshotError) as exc_info:
            await fresh.initialize()

        assert exc_info.value.container == storage._vector_space_file
        assert marker.read_bytes() == b'{"model": "bge-m3"'

    @pytest.mark.asyncio
    async def test_a_marker_the_parser_could_not_finish_names_the_marker(
        self, tmp_path, monkeypatch
    ):
        storage = await _seeded_storage(tmp_path)
        marker = Path(storage._vector_space_file)
        marker_bytes = marker.read_bytes()

        real_load = json.load

        def too_deep_for_the_marker(fp, *args, **kwargs):
            if getattr(fp, "name", "") == storage._vector_space_file:
                raise RecursionError("maximum recursion depth exceeded")
            return real_load(fp, *args, **kwargs)

        monkeypatch.setattr(json, "load", too_deep_for_the_marker)

        fresh = _make_storage(tmp_path)
        with pytest.raises(CorruptStorageSnapshotError) as exc_info:
            await fresh.initialize()

        assert exc_info.value.container == storage._vector_space_file
        assert marker.read_bytes() == marker_bytes

    @pytest.mark.asyncio
    async def test_a_marker_that_was_never_written_still_attaches(self, tmp_path):
        storage = await _seeded_storage(tmp_path)
        Path(storage._vector_space_file).unlink()

        fresh = _make_storage(tmp_path)
        await fresh.initialize()
        assert (await fresh.get_by_id("v1"))["content"] == "hello"

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="chmod(0) cannot revoke read access on Windows",
    )
    @pytest.mark.skipif(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        reason="root bypasses the file mode this test relies on",
    )
    @pytest.mark.asyncio
    async def test_an_unreadable_index_is_not_labelled_corrupt(self, tmp_path):
        storage = await _seeded_storage(tmp_path)
        index = Path(storage._faiss_index_file)
        healthy = index.read_bytes()
        index.chmod(0o000)

        fresh = _make_storage(tmp_path)
        try:
            with pytest.raises(PermissionError):
                await fresh.initialize()
        finally:
            index.chmod(0o600)

        assert index.read_bytes() == healthy
        recovered = _make_storage(tmp_path)
        await recovered.initialize()
        assert (await recovered.get_by_id("v1"))["content"] == "hello"

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="st_mode carries no ACL on Windows; Windows uses icacls",
    )
    @pytest.mark.asyncio
    async def test_a_backup_is_never_more_readable_than_what_it_copies(self, tmp_path):
        from lightrag.tools.rebuild_vdb import RebuildTool

        storage = await _seeded_storage(tmp_path)
        Path(storage._faiss_index_file).chmod(0o600)
        Path(storage._meta_file).chmod(0o600)
        truncate_file(storage._faiss_index_file)

        refused = _make_storage(tmp_path)
        with pytest.raises(CorruptStorageSnapshotError) as caught:
            await refused.initialize()
        tool = RebuildTool()
        tool.entities_vdb = refused
        tool.corrupt_vdbs = {"entities": caught.value}
        tool.incompatible_vdbs = {"entities": str(caught.value)}
        saved_umask = os.umask(0o022)
        try:
            await tool.recover_incompatible(["entities"])
        finally:
            os.umask(saved_umask)

        backups = list(Path(storage._faiss_index_file).parent.glob("*.corrupt-*"))
        assert backups
        for backup in backups:
            mode = stat.S_IMODE(backup.stat().st_mode)
            assert mode == 0o600, f"{backup.name} is {oct(mode)}"


# ---------------------------------------------------------------------------
# Two-channel fingerprint fence (shared + Faiss two-file torn-pair contracts)
# ---------------------------------------------------------------------------


class TestFaissFingerprintFence(FingerprintFenceContractMixin):
    absent_fingerprint = (None, None)

    @pytest.fixture(autouse=True)
    def _fine_grained_clock(self, monkeypatch):
        """Stamp every published file strictly after the previous write so
        coarse tmpfs jiffies do not make complete pairs look torn."""
        real_atomic_write = faiss_impl.atomic_write
        last_stamp = [0]

        def stamped_atomic_write(file_name, write_fn, workspace="_", *args, **kwargs):
            result = real_atomic_write(file_name, write_fn, workspace, *args, **kwargs)
            stamp = max(os.stat(file_name).st_mtime_ns, last_stamp[0] + 1)
            os.utime(file_name, ns=(stamp, stamp))
            last_stamp[0] = stamp
            return result

        monkeypatch.setattr(faiss_impl, "atomic_write", stamped_atomic_write)

    async def make_worker(self, tmp_path) -> FaissVectorDBStorage:
        storage = _make_storage(tmp_path)
        await storage.initialize()
        return storage

    def ids_on_disk(self, worker: FaissVectorDBStorage) -> set[str]:
        return _ids_on_disk(worker)

    def patch_lost_notification(self, monkeypatch) -> None:
        async def _noop(*_args, **_kwargs):
            return None

        monkeypatch.setattr(faiss_impl, "set_all_update_flags", _noop)

    def patch_reload_failure(self, monkeypatch, broken: list[bool]) -> None:
        real_load = FaissVectorDBStorage._load_faiss_index

        def maybe_boom(storage_self):
            if broken[0]:
                raise OSError("unreadable")
            return real_load(storage_self)

        monkeypatch.setattr(FaissVectorDBStorage, "_load_faiss_index", maybe_boom)

    @pytest.mark.asyncio
    async def test_an_index_newer_than_its_metadata_is_an_interrupted_publication(
        self, tmp_path, multiprocess
    ):
        worker = await self.make_worker(tmp_path)
        try:
            await worker.upsert({"n1": {"content": "x"}})
            assert await worker.index_done_callback() is True
            assert worker._peer_commit_detected() is False

            meta_mtime = os.stat(worker._meta_file).st_mtime_ns
            os.utime(
                worker._faiss_index_file, ns=(meta_mtime + 1000, meta_mtime + 1000)
            )
            assert worker._peer_commit_detected() is False
        finally:
            await worker.finalize()

    @pytest.mark.asyncio
    async def test_a_peer_generations_behind_still_rejects_a_torn_pair(
        self, tmp_path, multiprocess, lost_notification, monkeypatch
    ):
        worker_a = await self.make_worker(tmp_path)
        worker_b = await self.make_worker(tmp_path)
        try:
            await worker_a.upsert({"A": {"content": "a"}})
            assert await worker_a.index_done_callback() is True
            assert await worker_b.get_by_id("A") is not None
            recorded_a = worker_b._loaded_fingerprint

            await worker_a.upsert({"B": {"content": "b"}})
            assert await worker_a.index_done_callback() is True

            await worker_a.upsert({"C": {"content": "c"}})
            real_atomic_write = faiss_impl.atomic_write

            def fail_on_meta(file_name, write_fn, workspace="_", *args, **kwargs):
                if file_name == worker_a._meta_file:
                    raise OSError("meta write boom")
                return real_atomic_write(
                    file_name, write_fn, workspace, *args, **kwargs
                )

            with monkeypatch.context() as patched:
                patched.setattr(faiss_impl, "atomic_write", fail_on_meta)
                with pytest.raises(OSError, match="meta write boom"):
                    await worker_a.index_done_callback()

            current = worker_b._stat_fingerprint()
            assert all(new != old for new, old in zip(current, recorded_a))
            assert worker_b._peer_commit_detected() is False
            assert await worker_b.get_by_id("C") is None
        finally:
            await worker_a.finalize()
            await worker_b.finalize()

    @pytest.mark.asyncio
    async def test_a_peer_does_not_reload_another_workers_half_published_pair(
        self, tmp_path, multiprocess, lost_notification, monkeypatch
    ):
        worker_a = await self.make_worker(tmp_path)
        worker_b = await self.make_worker(tmp_path)
        try:
            await worker_a.upsert({"A": {"content": "a"}, "B": {"content": "b"}})
            assert await worker_a.index_done_callback() is True
            assert await worker_b.get_by_id("B") is not None
            recoveries = worker_b._missed_notification_reloads

            await worker_a.delete(["A"])
            real_atomic_write = faiss_impl.atomic_write

            def fail_on_meta(file_name, write_fn, workspace="_", *args, **kwargs):
                if file_name == worker_a._meta_file:
                    raise OSError("meta write boom")
                return real_atomic_write(
                    file_name, write_fn, workspace, *args, **kwargs
                )

            with monkeypatch.context() as patched:
                patched.setattr(faiss_impl, "atomic_write", fail_on_meta)
                with pytest.raises(OSError, match="meta write boom"):
                    await worker_a.index_done_callback()

            assert worker_b._peer_commit_detected() is False
            assert worker_b._missed_notification_reloads == recoveries
            assert await worker_b.get_by_id("B") is not None

            assert await worker_a.index_done_callback() is True
            assert worker_b._peer_commit_detected() is True
            assert await worker_b.get_by_id("A") is None
            assert await worker_b.get_by_id("B") is not None
        finally:
            await worker_a.finalize()
            await worker_b.finalize()

    @pytest.mark.asyncio
    async def test_a_partial_two_file_save_is_not_read_as_a_peer_commit(
        self, tmp_path, multiprocess, monkeypatch
    ):
        worker = await self.make_worker(tmp_path)
        try:
            await worker.upsert({"A": {"content": "a"}, "B": {"content": "b"}})
            assert await worker.index_done_callback() is True
            assert _ids_on_disk(worker) == {"A", "B"}

            await worker.delete(["A"])
            real_atomic_write = faiss_impl.atomic_write

            def fail_on_meta(file_name, write_fn, workspace="_", *args, **kwargs):
                if file_name == worker._meta_file:
                    raise OSError("meta write boom")
                return real_atomic_write(
                    file_name, write_fn, workspace, *args, **kwargs
                )

            with monkeypatch.context() as patched:
                patched.setattr(faiss_impl, "atomic_write", fail_on_meta)
                with pytest.raises(OSError, match="meta write boom"):
                    await worker.index_done_callback()

            assert worker._peer_commit_detected() is False

            assert await worker.index_done_callback() is True
            assert _ids_on_disk(worker) == {"B"}
            assert await worker.get_by_id("B") is not None
        finally:
            await worker.finalize()

    @pytest.mark.asyncio
    async def test_a_torn_pair_inside_one_timestamp_tick_is_still_refused(
        self, tmp_path, multiprocess, lost_notification, monkeypatch
    ):
        worker_a = await self.make_worker(tmp_path)
        worker_b = await self.make_worker(tmp_path)
        try:
            await worker_a.upsert({"A": {"content": "a"}, "B": {"content": "b"}})
            assert await worker_a.index_done_callback() is True
            assert await worker_b.get_by_id("B") is not None

            await worker_a.delete(["A"])
            real_atomic_write = faiss_impl.atomic_write

            def fail_on_meta(file_name, write_fn, workspace="_", *args, **kwargs):
                if file_name == worker_a._meta_file:
                    raise OSError("meta write boom")
                return real_atomic_write(
                    file_name, write_fn, workspace, *args, **kwargs
                )

            with monkeypatch.context() as patched:
                patched.setattr(faiss_impl, "atomic_write", fail_on_meta)
                with pytest.raises(OSError, match="meta write boom"):
                    await worker_a.index_done_callback()

            meta_mtime = os.stat(worker_b._meta_file).st_mtime_ns
            os.utime(worker_b._faiss_index_file, ns=(meta_mtime, meta_mtime))
            assert (
                os.stat(worker_b._faiss_index_file).st_mtime_ns
                == os.stat(worker_b._meta_file).st_mtime_ns
            )
            assert worker_b._stat_fingerprint() != worker_b._loaded_fingerprint
            assert worker_b._peer_commit_detected() is False
            assert await worker_b.get_by_id("B") is not None
        finally:
            await worker_a.finalize()
            await worker_b.finalize()


# ---------------------------------------------------------------------------
# Maintenance exact-count contract
# ---------------------------------------------------------------------------


class TestFaissExactCount(ExactCountContractMixin):
    def make_exact_count_storage(self, tmp_path) -> FaissVectorDBStorage:
        return _make_storage(tmp_path)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("ntotal", "id_to_meta"),
        [
            pytest.param(2, {0: {"__id__": "one"}}, id="fewer-rows-than-slots"),
            pytest.param(1, {2: {"__id__": "one"}}, id="position-out-of-range"),
            pytest.param(
                2, {0: {"__id__": "x"}, 1: {"__id__": "x"}}, id="duplicate-id"
            ),
            pytest.param(1, {0: {"content": "orphan"}}, id="missing-id"),
        ],
    )
    async def test_count_refuses_index_metadata_skew(self, ntotal, id_to_meta):
        storage = object.__new__(FaissVectorDBStorage)
        storage._storage_lock = NoLock()
        storage._reload_index_from_disk_locked = MagicMock()
        storage._pending_upserts = storage._unsaved_upserts = {}
        storage._pending_deletes = set()
        storage._unsaved_deletes = {}
        storage._index_dirty = False
        storage._index = SimpleNamespace(ntotal=ntotal)
        storage._id_to_meta = id_to_meta

        with pytest.raises(StorageCapabilityError, match="index_meta_skew"):
            await storage.get_exact_count()
