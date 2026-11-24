"""Regression tests for LLM cache maintenance tools (migration & query-cache cleanup)."""

from __future__ import annotations

import ast
from pathlib import Path
import typing

import pytest

import lightrag
from lightrag.base import QueryParam
from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.tools.clean_llm_query_cache import QUERY_MODES
from lightrag.tools.migrate_llm_cache import (
    DEFAULT_CACHE_TYPES,
    MigrationTool,
    _cache_type_from_key,
    _is_default_cache_key,
)
from lightrag.utils import use_llm_func_with_cache

pytestmark = pytest.mark.offline


# --------------------------------------------------------------------------- #
# LLM cache migration (multimodal analysis + smartheading + completeness)
# --------------------------------------------------------------------------- #


class _FakeLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _FakeJsonStorage:
    def __init__(self, data):
        self._data = data
        self._storage_lock = _FakeLock()


class _DummyEmbeddingFunc:
    """KV storages never embed; the dataclass only requires the field."""

    embedding_dim = 4

    async def __call__(self, texts):  # pragma: no cover - never invoked
        return [[0.0] * self.embedding_dim for _ in texts]


async def _build_cache_storage(tmp_path, workspace: str) -> JsonKVStorage:
    from lightrag.kg.shared_storage import initialize_share_data

    initialize_share_data()
    storage = JsonKVStorage(
        namespace="llm_response_cache",
        global_config={
            "working_dir": str(tmp_path),
            "enable_llm_cache": True,
            "enable_llm_cache_for_entity_extract": True,
        },
        embedding_func=_DummyEmbeddingFunc(),
        workspace=workspace,
    )
    await storage.initialize()
    return storage


async def _write_smartheading_row(storage, prompt: str) -> str:
    """Write one cache row through the real parse-stage writer path."""

    async def _llm(prompt, **kwargs):
        return "judged"

    keys: list[str] = []
    await use_llm_func_with_cache(
        prompt,
        _llm,
        llm_response_cache=storage,
        cache_type="smartheading",
        cache_keys_collector=keys,
    )
    assert len(keys) == 1
    return keys[0]


@pytest.mark.asyncio
async def test_json_migration_includes_multimodal_analysis_caches():
    tool = MigrationTool()
    storage = _FakeJsonStorage(
        {
            "default:extract:extract-hash": {"return": "extract"},
            "default:summary:summary-hash": {"return": "summary"},
            "default:analysis:analysis-hash": {
                "return": '{"name": "diagram"}',
                "cache_type": "analysis",
            },
            "mix:query:query-hash": {"return": "query"},
            "default:unknown:unknown-hash": {"return": "unknown"},
        }
    )

    loaded = await tool.get_default_caches_json(storage)
    counted = await tool.count_default_caches_json(storage)
    streamed = [
        batch async for batch in tool.stream_default_caches_json(storage, batch_size=2)
    ]
    type_counts = await tool.count_cache_types(loaded)

    assert list(loaded) == [
        "default:extract:extract-hash",
        "default:summary:summary-hash",
        "default:analysis:analysis-hash",
    ]
    assert counted == 3
    assert [key for batch in streamed for key in batch] == list(loaded)
    assert type_counts == {
        "extract": 1,
        "summary": 1,
        "analysis": 1,
        "smartheading": 0,
    }


async def test_the_parse_stage_writes_a_default_smartheading_row(tmp_path):
    """The premise: the runtime really does produce ``default:smartheading:*``."""
    storage = await _build_cache_storage(tmp_path, "smartheading_key_shape")
    try:
        key = await _write_smartheading_row(storage, "is this a heading?")

        assert key.startswith("default:smartheading:")
        assert _cache_type_from_key(key) == "smartheading"
        assert await storage.get_by_id(key) is not None
    finally:
        await storage.finalize()


async def test_migration_tool_moves_default_smartheading_caches(tmp_path):
    """Counting, streaming and migrating must all see the smart-heading row."""
    storage = await _build_cache_storage(tmp_path, "smartheading_migration")
    try:
        smartheading_key = await _write_smartheading_row(storage, "title?")

        tool = MigrationTool()

        counted = await tool.count_default_caches_json(storage)
        loaded = await tool.get_default_caches_json(storage)
        streamed = [
            key
            async for batch in tool.stream_default_caches_json(storage, batch_size=2)
            for key in batch
        ]
        type_counts = await tool.count_cache_types(loaded)

        assert counted == 1
        assert list(loaded) == [smartheading_key]
        assert streamed == [smartheading_key]
        assert type_counts["smartheading"] == 1
    finally:
        await storage.finalize()


async def test_a_store_of_only_smartheading_rows_is_not_reported_as_empty(tmp_path):
    """A source with smart-heading rows only must not say 'No records'."""
    source = await _build_cache_storage(tmp_path, "smartheading_only_source")
    target = await _build_cache_storage(tmp_path, "smartheading_only_target")
    try:
        smartheading_key = await _write_smartheading_row(source, "heading?")

        tool = MigrationTool()
        source_data = await tool.get_default_caches(source, "JsonKVStorage")
        stats = await tool.migrate_caches(source_data, target, "JsonKVStorage")

        assert stats.total_source_records == 1
        assert stats.successful_records == 1

        migrated = await target.get_by_id(smartheading_key)
        assert migrated is not None
        assert migrated["return"] == "judged"
        assert migrated["cache_type"] == "smartheading"
    finally:
        await source.finalize()
        await target.finalize()


def _literal(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _argument(call: ast.Call, name: str, position: int) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    if len(call.args) > position:
        return call.args[position]
    return None


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _default_mode_cache_writers() -> dict[str, set[str]]:
    """Map each scanner pathway to the default-mode cache types it found."""
    found: dict[str, set[str]] = {
        "use_llm_func_with_cache": set(),
        "mode_default": set(),
        "generate_cache_key": set(),
    }
    package = Path(lightrag.__file__).parent
    for path in package.rglob("*.py"):
        if package / "tools" in path.parents:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name == "use_llm_func_with_cache":
                keyword = next(
                    (kw.value for kw in node.keywords if kw.arg == "cache_type"),
                    None,
                )
                cache_type = "extract" if keyword is None else _literal(keyword)
                if cache_type is not None:
                    found["use_llm_func_with_cache"].add(cache_type)
            elif name == "generate_cache_key":
                if _literal(_argument(node, "mode", 0)) == "default":
                    cache_type = _literal(_argument(node, "cache_type", 1))
                    if cache_type is not None:
                        found["generate_cache_key"].add(cache_type)
            else:
                mode = next(
                    (kw.value for kw in node.keywords if kw.arg == "mode"), None
                )
                cache_type = next(
                    (kw.value for kw in node.keywords if kw.arg == "cache_type"),
                    None,
                )
                cache_type = _literal(cache_type)
                if _literal(mode) == "default" and cache_type is not None:
                    found["mode_default"].add(cache_type)
    return found


def test_every_default_mode_cache_type_the_runtime_writes_is_migrated():
    """Every default-mode type a writer in ``lightrag/`` names must migrate."""
    found = _default_mode_cache_writers()
    assert all(found.values()), found

    written = set().union(*found.values())
    missing = written - set(DEFAULT_CACHE_TYPES)
    assert not missing, (
        f"default-mode cache types written by lightrag/ but not migrated by "
        f"lightrag/tools/migrate_llm_cache.py: {sorted(missing)}; add them to "
        f"DEFAULT_CACHE_TYPES"
    )
    for cache_type in written:
        assert _is_default_cache_key(f"default:{cache_type}:abc123")
    assert not _is_default_cache_key("mix:query:abc123")
    assert not _is_default_cache_key("default:unknown:abc123")


# --------------------------------------------------------------------------- #
# Query-cache cleanup mode sync with QueryParam.mode
# --------------------------------------------------------------------------- #

_MODES_WITHOUT_CACHE = {"bypass"}


def _declared_query_modes() -> set[str]:
    hints = typing.get_type_hints(QueryParam)
    return set(typing.get_args(hints["mode"]))


def test_cleanup_tool_scans_exactly_the_cache_writing_query_modes():
    """Every mode that writes cache entries, no unknown mode, and not bypass."""
    declared = _declared_query_modes()
    expected = declared - _MODES_WITHOUT_CACHE
    missing = expected - set(QUERY_MODES)
    assert not missing, (
        f"QUERY_MODES misses {sorted(missing)}; their cache entries would be "
        "neither counted nor deleted by the cleanup tool"
    )
    unknown = set(QUERY_MODES) - declared
    assert not unknown, f"QUERY_MODES lists non-existent modes: {sorted(unknown)}"
    assert not set(QUERY_MODES) & _MODES_WITHOUT_CACHE
