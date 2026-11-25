"""Corrupt-snapshot, fingerprint-fence, exact-count, and surrogate tests for Nano.

Subclasses the shared fingerprint-fence and exact-count contract mixins in
``tests/kg/_file_backed_vector_contracts.py`` and retains all Nano-specific
corrupt JSON/matrix payloads, malformed-row count refusals, and lone-surrogate
sanitization tests.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

nano_vectordb = pytest.importorskip("nano_vectordb")  # noqa: F841

from lightrag.exceptions import (  # noqa: E402
    CorruptStorageSnapshotError,
    StorageCapabilityError,
    VectorSpaceMismatchError,
)
from lightrag.kg import nano_vector_db_impl  # noqa: E402
from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage  # noqa: E402
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


def _make_storage(
    tmp_path, *, meta_fields: set[str] | None = None
) -> NanoVectorDBStorage:
    return NanoVectorDBStorage(
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
        meta_fields=meta_fields or {"content"},
    )


async def _seeded_storage(tmp_path) -> NanoVectorDBStorage:
    storage = _make_storage(tmp_path)
    await storage.initialize()
    await storage.upsert({"v1": {"content": "hello"}})
    await storage.index_done_callback()
    assert Path(storage._client_file_name).exists()
    return storage


def _write_binary_garbage(path: str) -> None:
    Path(path).write_bytes(b"\xff\xfe\x00\x01 not json at all \x89\x8a")


UNREADABLE_PAYLOADS = {
    "damaged-base64-matrix": b'{"embedding_dim": 8, "data": [], "matrix": "!!not-b64!"}',
    "matrix-shorter-than-a-row": b'{"embedding_dim": 8, "data": [], "matrix": "AAAAAA=="}',
    "matrix-not-a-string": b'{"embedding_dim": 8, "data": [], "matrix": 5}',
    "no-matrix-at-all": b"{}",
    "not-an-object": b"[1, 2, 3]",
    "json-null": b"null",
}


# ---------------------------------------------------------------------------
# Corrupt snapshot refusal & offline backup/recovery
# ---------------------------------------------------------------------------


class TestNanoCorruptSnapshot:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("corrupt", [truncate_file, _write_binary_garbage])
    async def test_initialize_refuses_a_corrupt_snapshot(self, tmp_path, corrupt):
        storage = await _seeded_storage(tmp_path)
        client_file = storage._client_file_name
        corrupt(client_file)

        fresh = _make_storage(tmp_path)
        with pytest.raises(CorruptStorageSnapshotError) as exc_info:
            await fresh.initialize()

        message = str(exc_info.value)
        assert client_file in message
        assert "lightrag-rebuild-vdb" in message
        assert isinstance(
            exc_info.value.__cause__, (json.JSONDecodeError, UnicodeDecodeError)
        )

    @pytest.mark.asyncio
    async def test_a_parse_that_recursed_away_is_corruption(
        self, tmp_path, monkeypatch
    ):
        storage = await _seeded_storage(tmp_path)

        def too_deep(*args, **kwargs):
            raise RecursionError("maximum recursion depth exceeded")

        monkeypatch.setattr(nano_vector_db_impl, "NanoVectorDB", too_deep)

        fresh = _make_storage(tmp_path)
        with pytest.raises(CorruptStorageSnapshotError) as exc_info:
            await fresh.initialize()

        assert exc_info.value.container == storage._client_file_name
        assert isinstance(exc_info.value.__cause__, RecursionError)
        assert Path(storage._client_file_name).exists()

    @pytest.mark.asyncio
    async def test_a_deeply_nested_snapshot_refuses_however_the_parser_fails(
        self, tmp_path
    ):
        storage = await _seeded_storage(tmp_path)
        client_file = storage._client_file_name
        depth = sys.getrecursionlimit() * 2
        Path(client_file).write_text("[" * depth + "]" * depth, encoding="utf-8")

        fresh = _make_storage(tmp_path)
        with pytest.raises(CorruptStorageSnapshotError) as exc_info:
            await fresh.initialize()

        assert exc_info.value.container == client_file
        assert Path(client_file).exists()

    @pytest.mark.asyncio
    async def test_a_heap_that_ran_out_is_not_a_corrupt_snapshot(
        self, tmp_path, monkeypatch
    ):
        storage = await _seeded_storage(tmp_path)

        def out_of_memory(*args, **kwargs):
            raise MemoryError("cannot allocate the snapshot")

        monkeypatch.setattr(nano_vector_db_impl, "NanoVectorDB", out_of_memory)

        fresh = _make_storage(tmp_path)
        with pytest.raises(MemoryError):
            await fresh.initialize()
        assert Path(storage._client_file_name).exists()

    @pytest.mark.asyncio
    async def test_reader_reload_refuses_a_snapshot_a_peer_left_corrupt(self, tmp_path):
        storage = await _seeded_storage(tmp_path)
        client_file = storage._client_file_name

        truncate_file(client_file)
        storage.storage_updated.value = True

        with pytest.raises(CorruptStorageSnapshotError) as exc_info:
            await storage.query("hello", top_k=1, query_embedding=[1.0] * DIM)

        assert client_file in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_offline_recovery_backs_up_and_rebuilds_real_nano(self, tmp_path):
        from lightrag.tools.rebuild_vdb import RebuildTool

        storage = await _seeded_storage(tmp_path)
        truncate_file(storage._client_file_name)
        original = Path(storage._client_file_name).read_bytes()
        refused = _make_storage(tmp_path)
        with pytest.raises(CorruptStorageSnapshotError) as caught:
            await refused.initialize()
        tool = RebuildTool()
        tool.entities_vdb = refused
        tool.corrupt_vdbs = {"entities": caught.value}
        tool.incompatible_vdbs = {"entities": str(caught.value)}
        await tool.recover_incompatible(["entities"])
        backups = list(Path(refused._client_file_name).parent.glob("*.corrupt-*"))
        assert len(backups) == 1
        assert backups[0].read_bytes() == original
        await refused.upsert({"restored": {"content": "hello"}})
        await refused.index_done_callback()
        fresh = _make_storage(tmp_path)
        await fresh.initialize()
        assert (await fresh.get_by_id("restored"))["content"] == "hello"
        assert backups[0].read_bytes() == original

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "payload", UNREADABLE_PAYLOADS.values(), ids=UNREADABLE_PAYLOADS
    )
    async def test_initialize_refuses_a_snapshot_it_cannot_reconstitute(
        self, tmp_path, payload
    ):
        storage = await _seeded_storage(tmp_path)
        client_file = storage._client_file_name
        Path(client_file).write_bytes(payload)

        fresh = _make_storage(tmp_path)
        with pytest.raises(CorruptStorageSnapshotError) as exc_info:
            await fresh.initialize()

        assert client_file in str(exc_info.value)
        assert "lightrag-rebuild-vdb" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None
        assert Path(client_file).read_bytes() == payload

    @pytest.mark.asyncio
    async def test_a_dimension_mismatch_is_not_reported_as_corruption(self, tmp_path):
        storage = await _seeded_storage(tmp_path)

        wider = NanoVectorDBStorage(
            namespace="test_vectors",
            workspace="ws",
            global_config=storage.global_config,
            embedding_func=EmbeddingFunc(
                embedding_dim=DIM * 2, max_token_size=512, func=CountingEmbed(DIM * 2)
            ),
            meta_fields={"content"},
        )
        with pytest.raises(VectorSpaceMismatchError):
            await wider.initialize()

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="chmod(0) sets the read-only attribute on Windows rather than revoking read",
    )
    @pytest.mark.skipif(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        reason="root bypasses the file mode this test relies on",
    )
    @pytest.mark.asyncio
    async def test_an_unreadable_snapshot_is_not_labelled_corrupt(self, tmp_path):
        storage = await _seeded_storage(tmp_path)
        path = Path(storage._client_file_name)
        healthy = path.read_bytes()
        path.chmod(0o000)

        fresh = _make_storage(tmp_path)
        try:
            with pytest.raises(PermissionError):
                await fresh.initialize()
        finally:
            path.chmod(0o600)

        assert path.read_bytes() == healthy
        recovered = _make_storage(tmp_path)
        await recovered.initialize()
        assert (await recovered.get_by_id("v1"))["content"] == "hello"


# ---------------------------------------------------------------------------
# Two-channel fingerprint fence
# ---------------------------------------------------------------------------


class TestNanoFingerprintFence(FingerprintFenceContractMixin):
    absent_fingerprint = (None,)

    async def make_worker(self, tmp_path) -> NanoVectorDBStorage:
        storage = _make_storage(tmp_path)
        await storage.initialize()
        return storage

    def ids_on_disk(self, worker: NanoVectorDBStorage) -> set[str]:
        with open(worker._client_file_name, encoding="utf-8") as fh:
            return {row["__id__"] for row in json.load(fh)["data"]}

    def patch_lost_notification(self, monkeypatch) -> None:
        async def _noop(*_args, **_kwargs):
            return None

        monkeypatch.setattr(nano_vector_db_impl, "set_all_update_flags", _noop)

    def patch_reload_failure(self, monkeypatch, broken: list[bool]) -> None:
        real_client = nano_vector_db_impl.NanoVectorDB

        def maybe_boom(*args, **kwargs):
            if broken[0]:
                raise OSError("unreadable")
            return real_client(*args, **kwargs)

        monkeypatch.setattr(nano_vector_db_impl, "NanoVectorDB", maybe_boom)

    @pytest.mark.asyncio
    async def test_writer_does_not_reload_the_file_it_just_wrote(
        self, tmp_path, multiprocess, monkeypatch
    ):
        worker = await self.make_worker(tmp_path)
        try:
            await worker.upsert({"n1": {"content": "x"}})
            assert await worker.index_done_callback() is True

            reloads = [0]
            original = NanoVectorDBStorage._reload_client_from_disk_locked

            def counting(storage_self, **kwargs):
                did = original(storage_self, **kwargs)
                reloads[0] += int(bool(did))
                return did

            monkeypatch.setattr(
                NanoVectorDBStorage, "_reload_client_from_disk_locked", counting
            )
            assert await worker.get_by_id("n1") is not None
            assert reloads[0] == 0
            assert worker._missed_notification_reloads == 0
        finally:
            await worker.finalize()


# ---------------------------------------------------------------------------
# Maintenance exact-count contract
# ---------------------------------------------------------------------------


class TestNanoExactCount(ExactCountContractMixin):
    def make_exact_count_storage(self, tmp_path) -> NanoVectorDBStorage:
        return _make_storage(tmp_path)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("rows", "reason"),
        [
            ([{"__id__": "same"}, {"__id__": "same"}], "duplicate_persisted_id"),
            ([{"content": "orphan"}], "unidentifiable_persisted_row"),
            (["not a row"], "malformed_persisted_row"),
            ({"not": "a list"}, "malformed_persisted_data"),
        ],
    )
    async def test_count_refuses_malformed_persisted_rows(self, rows, reason):
        storage = object.__new__(NanoVectorDBStorage)
        storage._storage_lock = NoLock()
        storage._reload_client_from_disk_locked = MagicMock()
        storage._pending_upserts = storage._unsaved_upserts = {}
        storage._pending_deletes = set()
        storage._unsaved_deletes = {}
        storage._client_dirty = False
        storage._client = SimpleNamespace(_NanoVectorDB__storage={"data": rows})

        with pytest.raises(StorageCapabilityError, match=reason):
            await storage.get_exact_count()


# ---------------------------------------------------------------------------
# Lone-surrogate sanitization on upsert
# ---------------------------------------------------------------------------


class TestNanoUpsertSurrogate:
    @pytest.mark.asyncio
    async def test_upsert_strips_lone_surrogate_and_commit_succeeds(self, tmp_path):
        storage = _make_storage(tmp_path, meta_fields={"content", "tags", "extra"})
        await storage.initialize()

        await storage.upsert({"bad": {"content": "hello \ud800 world"}})
        await storage.index_done_callback()

        row = await storage.get_by_id("bad")
        assert row["content"] == "hello  world"

        reloaded = _make_storage(tmp_path, meta_fields={"content", "tags", "extra"})
        await reloaded.initialize()
        assert (await reloaded.get_by_id("bad"))["content"] == "hello  world"

    @pytest.mark.asyncio
    async def test_poisoned_row_no_longer_wedges_namespace(self, tmp_path):
        storage = _make_storage(tmp_path, meta_fields={"content", "tags", "extra"})
        await storage.initialize()

        await storage.upsert({"bad": {"content": "hello \ud800 world"}})
        await storage.index_done_callback()

        await storage.upsert({"good": {"content": "clean text"}})
        await storage.index_done_callback()

        assert (await storage.get_by_id("bad"))["content"] == "hello  world"
        assert (await storage.get_by_id("good"))["content"] == "clean text"

    @pytest.mark.asyncio
    async def test_clean_content_and_nested_values_unchanged_except_surrogates(
        self, tmp_path
    ):
        storage = _make_storage(tmp_path, meta_fields={"content", "tags", "extra"})
        await storage.initialize()

        await storage.upsert(
            {
                "id1": {
                    "content": "plain text with emoji \U0001f600 and entity &amp;",
                    "tags": ["keep", "drop\udc00me"],
                    "extra": {"note": "x\udfffy", "count": 3, "nc": "a\ufffeb\uffffc"},
                }
            }
        )
        await storage.index_done_callback()

        expected = {
            "content": "plain text with emoji \U0001f600 and entity &amp;",
            "tags": ["keep", "dropme"],
            "extra": {"note": "xy", "count": 3, "nc": "a\ufffeb\uffffc"},
        }
        row = await storage.get_by_id("id1")
        assert {k: row[k] for k in expected} == expected

        reloaded = _make_storage(tmp_path, meta_fields={"content", "tags", "extra"})
        await reloaded.initialize()
        row = await reloaded.get_by_id("id1")
        assert {k: row[k] for k in expected} == expected

    @pytest.mark.asyncio
    async def test_encodable_identity_field_is_not_rewritten(self, tmp_path):
        """An entity name holding U+FFFE must keep matching its graph key."""
        storage = _make_storage(tmp_path, meta_fields={"entity_name", "content"})
        await storage.initialize()

        await storage.upsert(
            {"ent-x": {"entity_name": "A\ufffeB", "content": "A\ufffeB\ndesc"}}
        )
        await storage.index_done_callback()

        reloaded = _make_storage(tmp_path, meta_fields={"entity_name", "content"})
        await reloaded.initialize()
        assert (await reloaded.get_by_id("ent-x"))["entity_name"] == "A\ufffeB"
