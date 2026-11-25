"""JSON storage persistence contracts: atomic writes, sanitization, commit reconciliation, offload, and RPC counts.

Covers:
- ``write_json`` atomicity on crash and concurrent threads, plus two-stage
  fast/slow surrogate sanitization and ``sanitize_text_for_encoding``.
- ``commit_in_storage_io`` cancellation-safe sanitize reconciliation,
  post-write flag-clear resilience, and fatal reload-failure propagation.
- Off-event-loop commit execution while holding ``_storage_lock``.
- Single-RPC ``DictProxy.copy()`` snapshotting (counting fake + real Manager).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from lightrag.base import DocStatus
from lightrag.kg import json_doc_status_impl, json_kv_impl
from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage
from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.namespace import NameSpace
from lightrag.utils import (
    SanitizingJSONEncoder,
    load_json,
    sanitize_text_for_encoding,
    write_json,
)

pytestmark = pytest.mark.offline

# A blocked loop gives zero beats; a live one beats within one 5 ms GIL switch.
BLOCK_SECONDS = 0.05


class _DummyEmbeddingFunc:
    """Minimal embedding stand-in for storages that do not embed."""

    embedding_dim = 1
    max_token_size = 1

    async def __call__(self, texts, **kwargs):
        return [[0.0] for _ in texts]


class _Heartbeat:
    """Counts event-loop iterations so a blocked loop is directly observable."""

    def __init__(self):
        self.beats = 0
        self._stop = False
        self._task = None

    async def _run(self):
        while not self._stop:
            self.beats += 1
            await asyncio.sleep(0)

    def start(self):
        self._task = asyncio.create_task(self._run())
        return self

    async def stop(self):
        self._stop = True
        if self._task is not None:
            await self._task


class _CountingDict(dict):
    """DictProxy-shaped fake counting remote-style snapshot and key accesses."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.copy_calls = 0
        self.getitem_calls = 0
        self.keys_calls = 0
        self.contains_calls = 0

    def copy(self):
        self.copy_calls += 1
        return dict(self)

    def __getitem__(self, key):
        self.getitem_calls += 1
        return super().__getitem__(key)

    def keys(self):
        self.keys_calls += 1
        return super().keys()

    def __contains__(self, key):
        self.contains_calls += 1
        return super().__contains__(key)


@pytest.fixture
def single_process_shared_data():
    finalize_share_data()
    initialize_share_data(1)
    yield
    finalize_share_data()


async def _make_kv(tmp_path, *, workspace: str = "ws") -> JsonKVStorage:
    storage = JsonKVStorage(
        namespace="text_chunks",
        workspace=workspace,
        global_config={"working_dir": str(tmp_path)},
        embedding_func=None,
    )
    await storage.initialize()
    return storage


async def _make_doc_status(tmp_path, *, workspace: str = "ws") -> JsonDocStatusStorage:
    storage = JsonDocStatusStorage(
        namespace="doc_status",
        workspace=workspace,
        global_config={"working_dir": str(tmp_path)},
        embedding_func=None,
    )
    await storage.initialize()
    return storage


def _doc(status: str, file_path: str) -> dict:
    return {
        "content_summary": f"{status} summary",
        "content_length": 10,
        "file_path": file_path,
        "status": status,
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00",
        "metadata": {},
        "error_msg": None,
        "chunks_list": [],
    }


def _parked_sanitizing_write(inside_write, may_finish, cleaned_payload):
    """Stand in for ``write_json``: park, publish cleaned data, report reload."""

    def _write(data_dict, file_name):
        inside_write.set()
        assert may_finish.wait(timeout=5), "writer was never released"
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(cleaned_payload, f)
        return True

    return _write


def _blocking_spy(module, heartbeat, observed):
    """Wrap the module's ``write_json`` with a real, GIL-releasing block."""
    real_write_json = module.write_json

    def spy(payload, path):
        before = heartbeat.beats
        time.sleep(BLOCK_SECONDS)
        observed.append((before, heartbeat.beats))
        return real_write_json(payload, path)

    return spy


# ---------------------------------------------------------------------------
# write_json atomicity
# ---------------------------------------------------------------------------


def test_write_json_publishes_clean_payload(tmp_path):
    target = str(tmp_path / "kv.json")
    needs_reload = write_json({"a": 1, "b": "hello"}, target)
    assert needs_reload is False
    with open(target, encoding="utf-8") as f:
        assert json.load(f) == {"a": 1, "b": "hello"}
    assert [p for p in os.listdir(tmp_path) if ".tmp." in p] == []


def test_write_json_replace_crash_preserves_prior_snapshot(tmp_path):
    target = str(tmp_path / "kv.json")
    write_json({"v": 1}, target)

    with patch(
        "lightrag.file_atomic.os.replace",
        side_effect=OSError("simulated crash"),
    ):
        with pytest.raises(OSError, match="simulated crash"):
            write_json({"v": 2}, target)

    with open(target, encoding="utf-8") as f:
        assert json.load(f) == {"v": 1}
    leftovers = [p for p in os.listdir(tmp_path) if ".tmp." in p]
    assert leftovers == [], f"write_json must clean tmp on crash, got {leftovers}"


def test_write_json_concurrent_writers_land_intact(tmp_path):
    target = str(tmp_path / "kv.json")
    barrier = threading.Barrier(5, timeout=10)

    def writer(tid: int) -> None:
        barrier.wait()
        write_json({"writer": tid}, target)

    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(writer, range(5)))

    with open(target, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload.keys() == {"writer"}
    assert payload["writer"] in range(5)
    leftovers = [p for p in os.listdir(tmp_path) if ".tmp." in p]
    assert leftovers == [], f"Unexpected tmp residue: {leftovers}"


# ---------------------------------------------------------------------------
# Two-stage write_json sanitization & sanitize_text_for_encoding
# ---------------------------------------------------------------------------


class TestWriteJsonOptimization:
    """Two-stage write_json fast path (clean) and slow path (surrogate sanitization)."""

    def test_sanitizing_encoder_removes_surrogates(self):
        data_with_surrogates = {
            "text": "Hello\ud800\udc00World",
            "clean": "Clean text",
            "nested": {"dirty_key\ud801": "value", "clean_key": "clean\ud802value"},
        }

        encoded = json.dumps(
            data_with_surrogates, cls=SanitizingJSONEncoder, ensure_ascii=False
        )

        for surrogate in ("\ud800", "\udc00", "\ud801", "\ud802"):
            assert surrogate not in encoded
        assert "Clean text" in encoded
        assert "clean_key" in encoded

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            pytest.param(
                {
                    "level1": {
                        "level2": {
                            "level3": {
                                "dirty": "text\ud800here",
                                "clean": "normal text",
                            },
                            "list": ["item1", "item\ud801dirty", "item3"],
                        }
                    }
                },
                {
                    "level1": {
                        "level2": {
                            "level3": {"dirty": "texthere", "clean": "normal text"},
                            "list": ["item1", "itemdirty", "item3"],
                        }
                    }
                },
                id="nested-structures",
            ),
            pytest.param(
                {
                    "clean_field": "This is perfectly fine",
                    "dirty_field": "This has\ud800issues",
                    "number": 42,
                    "boolean": True,
                    "null_value": None,
                    "clean_list": [1, 2, 3],
                    "dirty_list": ["clean", "dirty\ud801item"],
                },
                {
                    "clean_field": "This is perfectly fine",
                    "dirty_field": "This hasissues",
                    "number": 42,
                    "boolean": True,
                    "null_value": None,
                    "clean_list": [1, 2, 3],
                    "dirty_list": ["clean", "dirtyitem"],
                },
                id="mixed-clean-and-dirty-types",
            ),
            pytest.param(
                {
                    "text": "Some text with surrogate\udc9acharacter",
                    "position": 201,
                    "clean_field": "Normal text",
                },
                {
                    "text": "Some text with surrogatecharacter",
                    "position": 201,
                    "clean_field": "Normal text",
                },
                id="specific-surrogate-udc9a",
            ),
            pytest.param(
                {"key1": "\ud800\udc00\ud801", "key2": "\ud802\ud803"},
                {"key1": "", "key2": ""},
                id="all-surrogates-leave-empty-string-values",
            ),
        ],
    )
    def test_write_json_slow_path_sanitizes_surrogates(
        self, tmp_path, payload, expected
    ):
        target = tmp_path / "dirty.json"
        assert write_json(payload, str(target)) is True
        loaded = load_json(str(target))
        assert loaded is not None
        assert loaded == expected

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param(
                {"text1": "Hello\ufffeWorld", "text2": "Test\uffffString"},
                id="utf8-valid-non-characters",
            ),
            pytest.param(
                {
                    "empty": "",
                    "none": None,
                    "zero": 0,
                    "false": False,
                    "empty_list": [],
                    "empty_dict": {},
                },
                id="empty-and-falsy-values",
            ),
            pytest.param({}, id="empty-dict"),
        ],
    )
    def test_write_json_fast_path_preserves_clean_data(self, tmp_path, payload):
        target = tmp_path / "clean.json"
        assert write_json(payload, str(target)) is False
        loaded = load_json(str(target))
        assert loaded is not None
        assert loaded == payload


class TestSanitizeTextForEncoding:
    """Direct unit tests for sanitize_text_for_encoding."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            pytest.param("", "", id="empty"),
            pytest.param("   ", "", id="whitespace-only"),
            pytest.param("hello world", "hello world", id="clean-unchanged"),
            pytest.param("  hello  ", "hello", id="strips-outer-whitespace"),
            pytest.param("hello\ud800world", "helloworld", id="lone-surrogate-removed"),
            pytest.param("\ud800", "", id="surrogate-low-boundary"),
            pytest.param("\udfff", "", id="surrogate-high-boundary"),
            pytest.param("a\ufffeb", "ab", id="non-char-fffe"),
            pytest.param("a\uffffb", "ab", id="non-char-ffff"),
            pytest.param("&amp;", "&", id="html-amp"),
            pytest.param("&lt;p&gt;", "<p>", id="html-tags"),
            pytest.param("&quot;hello&quot;", '"hello"', id="html-quot"),
            pytest.param("\x01hello\x1fworld", "helloworld", id="c0-control-chars"),
            pytest.param("\x00null", "null", id="nul-byte"),
            pytest.param("del\x7f", "del", id="del-char"),
            pytest.param("line1\nline2", "line1\nline2", id="newline-preserved"),
            pytest.param("col1\tcol2", "col1\tcol2", id="tab-preserved"),
            pytest.param("line1\rline2", "line1\rline2", id="cr-preserved"),
            pytest.param("caf\x85e", "caf\x85e", id="c1-control-preserved"),
            pytest.param("\ud800hello\x01", "hello", id="default-deletion"),
            pytest.param("\ud800&amp;\x01clean", "&clean", id="mixed-issues"),
            pytest.param(
                f"prefix\ud800{'a' * 10000}\udfffsuffix",
                f"prefix{'a' * 10000}suffix",
                id="large-text-scattered-surrogates",
            ),
        ],
    )
    def test_default_sanitization(self, raw, expected):
        assert sanitize_text_for_encoding(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            pytest.param("hello\ud800world", "hello?world", id="lone-surrogate"),
            pytest.param("a\x01b\x08c", "a?b?c", id="control-chars"),
        ],
    )
    def test_custom_replacement_char(self, raw, expected):
        assert sanitize_text_for_encoding(raw, replacement_char="?") == expected

    def test_html_entity_for_surrogate_leaves_no_lone_surrogate(self):
        result = sanitize_text_for_encoding("&#xD800;")
        assert "\ud800" not in result


# ---------------------------------------------------------------------------
# Commit reconciliation (cancel-safety, post-write bookkeeping, reload failures)
# ---------------------------------------------------------------------------

_JSON_STORAGES = [
    pytest.param(_make_kv, {"content": "alpha"}, id="kv"),
    pytest.param(
        _make_doc_status,
        {"status": "processed", "file_path": "a.pdf"},
        id="doc_status",
    ),
]


@pytest.mark.parametrize(("factory", "row"), _JSON_STORAGES)
async def test_cancelled_commit_still_reconciles_sanitized_data(
    tmp_path, monkeypatch, single_process_shared_data, factory, row
):
    storage = await factory(tmp_path)
    await storage.upsert({"id1": row})

    module = type(storage).__module__
    inside_write = threading.Event()
    may_finish = threading.Event()
    cleaned_payload = {"id1": dict(row, content_was="sanitized")}

    monkeypatch.setattr(
        f"{module}.write_json",
        _parked_sanitizing_write(inside_write, may_finish, cleaned_payload),
    )
    storage.storage_updated.value = True

    commit = asyncio.create_task(storage.index_done_callback())
    while not inside_write.is_set():
        await asyncio.sleep(0.01)

    commit.cancel()
    may_finish.set()

    with pytest.raises(asyncio.CancelledError):
        await commit

    assert storage._data["id1"].get("content_was") == "sanitized"


@pytest.mark.parametrize(("factory", "row"), _JSON_STORAGES)
async def test_reconciliation_is_skipped_when_the_write_fails(
    tmp_path, monkeypatch, single_process_shared_data, factory, row
):
    storage = await factory(tmp_path)
    await storage.upsert({"id1": row})

    module = type(storage).__module__
    reloads: list[str] = []

    def _boom(data_dict, file_name):
        raise OSError("disk full")

    monkeypatch.setattr(f"{module}.write_json", _boom)
    monkeypatch.setattr(
        f"{module}.load_json", lambda *a, **k: reloads.append("read") or {}
    )
    storage.storage_updated.value = True

    with pytest.raises(OSError, match="disk full"):
        await storage.index_done_callback()

    assert reloads == [], "reconciled a write that never landed"


@pytest.mark.parametrize(("factory", "row"), _JSON_STORAGES)
async def test_a_failed_flag_clear_is_not_reported_as_a_failed_write(
    tmp_path, monkeypatch, single_process_shared_data, factory, row, lightrag_caplog
):
    storage = await factory(tmp_path)
    await storage.upsert({"id1": row})

    module = type(storage).__module__

    async def failing_clear_all_update_flags(namespace, workspace=None):
        raise RuntimeError("shared-storage manager is down")

    monkeypatch.setattr(
        f"{module}.clear_all_update_flags", failing_clear_all_update_flags
    )
    storage.storage_updated.value = True

    with lightrag_caplog.at_level(logging.ERROR, logger="lightrag"):
        await storage.index_done_callback()

    with open(storage._file_name, encoding="utf-8") as f:
        persisted = json.load(f)
    assert "id1" in persisted
    assert storage.storage_updated.value is True
    assert any(
        "post-write bookkeeping failed" in record.getMessage()
        for record in lightrag_caplog.records
    )


@pytest.mark.parametrize(("factory", "row"), _JSON_STORAGES)
async def test_a_broken_sink_cannot_turn_a_landed_write_into_a_failure(
    tmp_path, monkeypatch, single_process_shared_data, factory, row
):
    storage = await factory(tmp_path)
    await storage.upsert({"id1": row})

    module = type(storage).__module__

    async def failing_clear_all_update_flags(namespace, workspace=None):
        raise RuntimeError("shared-storage manager is down")

    def log_boom(msg):
        raise RuntimeError("log sink boom")

    monkeypatch.setattr(
        f"{module}.clear_all_update_flags", failing_clear_all_update_flags
    )
    monkeypatch.setattr(f"{module}.logger.error", log_boom)
    storage.storage_updated.value = True

    await storage.index_done_callback()

    with open(storage._file_name, encoding="utf-8") as f:
        persisted = json.load(f)
    assert "id1" in persisted


@pytest.mark.parametrize(("factory", "row"), _JSON_STORAGES)
async def test_a_failed_sanitize_reload_is_not_absorbed(
    tmp_path, monkeypatch, single_process_shared_data, factory, row
):
    storage = await factory(tmp_path)
    await storage.upsert({"id1": row})

    module = type(storage).__module__

    class _BrokenUpdate(dict):
        """Dict double whose ``update()`` raises to simulate a mid-reconcile RPC error."""

        def update(self, *args, **kwargs):
            raise RuntimeError("shared-dict RPC failed")

    def _sanitizing_write(data_dict, file_name):
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump({"id1": dict(row, content_was="sanitized")}, f)
        return True

    monkeypatch.setattr(f"{module}.write_json", _sanitizing_write)
    storage._data = _BrokenUpdate(storage._data)
    storage.storage_updated.value = True

    with pytest.raises(RuntimeError, match="shared-dict RPC failed"):
        await storage.index_done_callback()

    with open(storage._file_name, encoding="utf-8") as f:
        assert json.load(f)["id1"]["content_was"] == "sanitized"
    assert dict(storage._data) == {}


# ---------------------------------------------------------------------------
# Off-event-loop persistence and storage-lock serialization
# ---------------------------------------------------------------------------


async def test_kv_commit_keeps_the_event_loop_running(
    tmp_path, monkeypatch, single_process_shared_data
):
    storage = await _make_kv(tmp_path, workspace="")
    await storage.upsert({"chunk-1": {"content": "hello"}})

    observed = []
    heartbeat = _Heartbeat().start()
    monkeypatch.setattr(
        json_kv_impl, "write_json", _blocking_spy(json_kv_impl, heartbeat, observed)
    )

    await storage.index_done_callback()
    await heartbeat.stop()

    assert observed, "write_json was never called"
    before, after = observed[0]
    assert after > before, "event loop was blocked for the whole write"
    with open(storage._file_name, encoding="utf-8") as handle:
        assert json.load(handle)["chunk-1"]["content"] == "hello"


async def test_doc_status_commit_keeps_the_event_loop_running(
    tmp_path, monkeypatch, single_process_shared_data
):
    storage = await _make_doc_status(tmp_path, workspace="")

    observed = []
    heartbeat = _Heartbeat().start()
    monkeypatch.setattr(
        json_doc_status_impl,
        "write_json",
        _blocking_spy(json_doc_status_impl, heartbeat, observed),
    )

    await storage.upsert(
        {
            "doc-1": {
                "status": "processed",
                "content_summary": "s",
                "content_length": 1,
                "file_path": "a.pdf",
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
            }
        }
    )
    await heartbeat.stop()

    assert observed, "write_json was never called"
    before, after = observed[0]
    assert after > before, "event loop was blocked for the whole write"
    with open(storage._file_name, encoding="utf-8") as handle:
        assert json.load(handle)["doc-1"]["status"] == "processed"


async def test_a_concurrent_upsert_waits_for_the_in_flight_write(
    tmp_path, monkeypatch, single_process_shared_data
):
    storage = await _make_kv(tmp_path, workspace="")
    await storage.upsert({"chunk-1": {"content": "first"}})

    release = threading.Event()
    entered = threading.Event()
    real_write_json = json_kv_impl.write_json

    def blocking_write(payload, path):
        entered.set()
        assert release.wait(timeout=5), "write was never released"
        return real_write_json(payload, path)

    monkeypatch.setattr(json_kv_impl, "write_json", blocking_write)

    commit = asyncio.create_task(storage.index_done_callback())
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 2
    while not entered.is_set():
        assert loop.time() < deadline, "write never started"
        await asyncio.sleep(0.01)

    upsert = asyncio.create_task(storage.upsert({"chunk-2": {"content": "second"}}))
    for _ in range(20):
        await asyncio.sleep(0)
    assert not upsert.done(), "upsert ran while a commit was in flight"

    release.set()
    await commit
    await upsert

    with open(storage._file_name, encoding="utf-8") as handle:
        published = json.load(handle)
    assert "chunk-1" in published
    assert "chunk-2" not in published

    monkeypatch.setattr(json_kv_impl, "write_json", real_write_json)
    await storage.index_done_callback()
    with open(storage._file_name, encoding="utf-8") as handle:
        final = json.load(handle)
    assert final["chunk-1"]["content"] == "first"
    assert final["chunk-2"]["content"] == "second"


# ---------------------------------------------------------------------------
# Snapshot RPC counts (counting fake + real 2-worker Manager)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kv_index_done_callback_snapshots_with_single_copy(
    tmp_path, single_process_shared_data
):
    storage = JsonKVStorage(
        namespace=NameSpace.KV_STORE_TEXT_CHUNKS,
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    await storage.initialize()

    counting = _CountingDict({f"chunk-{i}": {"content": str(i)} for i in range(50)})
    storage._data = counting
    storage.storage_updated.value = True

    await storage.index_done_callback()

    assert counting.copy_calls == 1
    assert counting.getitem_calls == 0
    assert counting.keys_calls == 0

    with open(storage._file_name, encoding="utf-8") as f:
        on_disk = json.load(f)
    assert len(on_disk) == 50


@pytest.mark.asyncio
async def test_doc_status_index_done_callback_snapshots_with_single_copy(
    tmp_path, single_process_shared_data
):
    storage = JsonDocStatusStorage(
        namespace=NameSpace.DOC_STATUS,
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    await storage.initialize()

    counting = _CountingDict(
        {f"doc-{i}": _doc("processed", f"{i}.pdf") for i in range(20)}
    )
    storage._data = counting
    storage.storage_updated.value = True

    await storage.index_done_callback()

    assert counting.copy_calls == 1
    assert counting.getitem_calls == 0
    assert counting.keys_calls == 0


@pytest.mark.asyncio
async def test_kv_index_done_callback_real_manager_persists(tmp_path):
    finalize_share_data()
    initialize_share_data(2)
    try:
        storage = JsonKVStorage(
            namespace=NameSpace.KV_STORE_TEXT_CHUNKS,
            global_config={"working_dir": str(tmp_path)},
            embedding_func=_DummyEmbeddingFunc(),
            workspace="test",
        )
        await storage.initialize()

        await storage.upsert(
            {
                "chunk-a": {"content": "alpha"},
                "chunk-b": {"content": "beta"},
            }
        )
        await storage.index_done_callback()

        with open(storage._file_name, encoding="utf-8") as f:
            on_disk = json.load(f)
        assert set(on_disk.keys()) == {"chunk-a", "chunk-b"}
        assert on_disk["chunk-a"]["content"] == "alpha"
        assert on_disk["chunk-b"]["content"] == "beta"
    finally:
        finalize_share_data()


@pytest.mark.asyncio
async def test_doc_status_upsert_real_manager_persists(tmp_path):
    finalize_share_data()
    initialize_share_data(2)
    try:
        storage = JsonDocStatusStorage(
            namespace=NameSpace.DOC_STATUS,
            global_config={"working_dir": str(tmp_path)},
            embedding_func=_DummyEmbeddingFunc(),
            workspace="test",
        )
        await storage.initialize()

        await storage.upsert({"doc-1": _doc("processed", "a.pdf")})

        with open(storage._file_name, encoding="utf-8") as f:
            on_disk = json.load(f)
        assert on_disk["doc-1"]["status"] == DocStatus.PROCESSED.value
        assert on_disk["doc-1"]["file_path"] == "a.pdf"
    finally:
        finalize_share_data()
