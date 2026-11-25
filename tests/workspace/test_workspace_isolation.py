"""Workspace isolation tests across shared storage, JsonKV, LightRAG E2E, and PG migration.

Covers:
- Pipeline status isolation between workspaces
- Namespace lock parallelism across workspaces/namespaces and serialization
  within the same workspace/namespace
- Backward-compatible default/empty workspace handling and error guardrails
- NamespaceLock re-entrance protection and concurrent coroutine reuse
- Update flag workspace isolation and filtering
- JsonKVStorage workspace directory and data isolation
- End-to-end concurrent LightRAG insertion isolation across workspaces
- PostgreSQL legacy-table migration workspace scoping
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.postgres_impl import PGVectorStorage
from lightrag.kg.shared_storage import (
    clear_all_update_flags,
    finalize_share_data,
    get_all_update_flags_status,
    get_default_workspace,
    get_final_namespace,
    get_namespace_data,
    get_namespace_lock,
    get_update_flag,
    initialize_pipeline_status,
    initialize_share_data,
    set_all_update_flags,
    set_default_workspace,
)
from lightrag.utils import EmbeddingFunc, Tokenizer
from tests._pg_legacy_migration import answer_legacy_migration

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def setup_shared_data():
    """Initialize shared data before each test and tear it down after."""
    initialize_share_data()
    yield
    finalize_share_data()


async def _measure_lock_parallelism(
    workload: list[tuple[str, str, str]],
    *,
    expect_parallel: bool,
) -> tuple[int, list[tuple[str, str]]]:
    """Run lock acquisition workload and capture peak concurrency and timeline.

    When ``expect_parallel`` is True, workers synchronize on an event once
    multiple holders are inside their locks concurrently (zero real sleep; a
    broken lock that serializes fails loudly via ``wait_for`` timeout).
    When ``expect_parallel`` is False, workers yield one event-loop turn so
    contending tasks attempt acquisition while the holder owns the lock.
    """
    running = 0
    max_parallel = 0
    timeline: list[tuple[str, str]] = []
    all_entered = asyncio.Event()
    target_parallel = len(workload) if expect_parallel else 1

    async def worker(name: str, workspace: str, namespace: str) -> None:
        nonlocal running, max_parallel
        lock = get_namespace_lock(namespace, workspace)
        async with lock:
            running += 1
            max_parallel = max(max_parallel, running)
            timeline.append((name, "start"))
            if expect_parallel:
                if running >= target_parallel:
                    all_entered.set()
                else:
                    await asyncio.wait_for(all_entered.wait(), timeout=2.0)
            else:
                await asyncio.sleep(0)
            timeline.append((name, "end"))
            running -= 1

    await asyncio.gather(*(worker(*args) for args in workload))
    return max_parallel, timeline


def _assert_no_timeline_overlap(timeline: list[tuple[str, str]]) -> None:
    """Validate sequential execution: only one task holds the lock at a time."""
    active_task = None
    for name, event in timeline:
        if event == "start":
            if active_task is not None:
                raise AssertionError(
                    f"Task '{name}' started before '{active_task}' released the lock"
                )
            active_task = name
        else:
            if active_task != name:
                raise AssertionError(
                    f"Task '{name}' finished while '{active_task}' was expected to hold the lock"
                )
            active_task = None

    if active_task is not None:
        raise AssertionError(f"Task '{active_task}' did not release the lock properly")


async def test_pipeline_status_isolation():
    """Pipeline status shared data remains isolated between workspaces."""
    workspace1 = "test_workspace_1"
    workspace2 = "test_workspace_2"

    await initialize_pipeline_status(workspace1)
    await initialize_pipeline_status(workspace2)

    data1 = await get_namespace_data("pipeline_status", workspace=workspace1)
    data2 = await get_namespace_data("pipeline_status", workspace=workspace2)

    assert data1 is not data2

    data1["test_key"] = "workspace1_value"

    data1_check = await get_namespace_data("pipeline_status", workspace=workspace1)
    data2_check = await get_namespace_data("pipeline_status", workspace=workspace2)

    assert data1_check.get("test_key") == "workspace1_value"
    assert "test_key" not in data2_check


async def test_lock_mechanism(stress_test_mode, parallel_workers):
    """Keyed locks run different workspaces in parallel and serialize same workspace."""
    num_workers = parallel_workers if stress_test_mode else 3
    parallel_workload = [
        (f"ws_{chr(97 + i)}", f"ws_{chr(97 + i)}", "test_namespace")
        for i in range(num_workers)
    ]

    max_parallel, timeline_parallel = await _measure_lock_parallelism(
        parallel_workload, expect_parallel=True
    )
    assert max_parallel >= 2, (
        "Locks for distinct workspaces should overlap; "
        f"observed max concurrency: {max_parallel}, timeline={timeline_parallel}"
    )

    serial_workload = [
        ("serial_run_1", "ws_same", "test_namespace"),
        ("serial_run_2", "ws_same", "test_namespace"),
    ]
    max_parallel_serial, timeline_serial = await _measure_lock_parallelism(
        serial_workload, expect_parallel=False
    )
    assert max_parallel_serial == 1, (
        "Same workspace locks should not overlap; "
        f"observed {max_parallel_serial} with timeline {timeline_serial}"
    )
    _assert_no_timeline_overlap(timeline_serial)


async def test_backward_compatibility():
    """Legacy callers omitting workspace fall back to default / empty workspace."""
    set_default_workspace("my_default_workspace")
    assert (
        get_final_namespace("pipeline_status") == "my_default_workspace:pipeline_status"
    )

    set_default_workspace("test_default")
    assert get_default_workspace() == "test_default"

    set_default_workspace("")
    assert get_final_namespace("pipeline_status", workspace=None) == "pipeline_status"

    set_default_workspace("compat_test_workspace")
    initialize_share_data()
    await initialize_pipeline_status(workspace=None)

    data = await get_namespace_data(
        "pipeline_status", workspace="compat_test_workspace"
    )
    assert data is not None


async def test_namespace_lock_reentrance():
    """NamespaceLock blocks same-coroutine re-entrance and allows concurrent reuse."""
    lock = get_namespace_lock("test_reentrance", "test_ws")

    async with lock:
        with pytest.raises(RuntimeError, match=r"(?i)already acquired"):
            async with lock:
                raise AssertionError("re-entrance must not succeed")

    shared_lock = get_namespace_lock("test_concurrent", "test_ws")
    timeline: list[tuple[str, str]] = []

    async def use_shared_lock(coroutine_id: int) -> None:
        async with shared_lock:
            timeline.append((f"coroutine_{coroutine_id}", "start"))
            await asyncio.sleep(0)
            timeline.append((f"coroutine_{coroutine_id}", "end"))

    await asyncio.gather(use_shared_lock(1), use_shared_lock(2))

    assert len(timeline) == 4
    _assert_no_timeline_overlap(timeline)


async def test_different_namespace_lock_isolation():
    """Locks for different namespaces within the same workspace run concurrently."""
    workload = [
        ("ns_a", "same_ws", "namespace_a"),
        ("ns_b", "same_ws", "namespace_b"),
        ("ns_c", "same_ws", "namespace_c"),
    ]
    max_parallel, timeline = await _measure_lock_parallelism(
        workload, expect_parallel=True
    )

    assert max_parallel >= 2, (
        "Different namespaces within the same workspace should run concurrently; "
        f"observed max concurrency {max_parallel} with timeline {timeline}"
    )


async def test_error_handling():
    """Guardrails for unset default workspace and None/empty normalization."""
    with pytest.raises(ValueError):
        get_final_namespace("test_namespace", workspace=None)

    set_default_workspace(None)
    assert get_default_workspace() == ""
    assert get_final_namespace("test_namespace", workspace="") == "test_namespace"


async def test_update_flags_workspace_isolation():
    """Update flag setters, clearers, and status queries stay scoped per workspace."""
    workspace1 = "update_flags_ws1"
    workspace2 = "update_flags_ws2"
    test_namespace = "test_update_flags_ns"

    await initialize_pipeline_status(workspace1)
    await initialize_pipeline_status(workspace2)

    flag1_obj = await get_update_flag(test_namespace, workspace=workspace1)
    flag2_obj = await get_update_flag(test_namespace, workspace=workspace2)

    assert flag1_obj.value is False
    assert flag2_obj.value is False

    await set_all_update_flags(test_namespace, workspace=workspace1)
    assert flag1_obj.value is True
    assert flag2_obj.value is False

    await set_all_update_flags(test_namespace, workspace=workspace1)
    await set_all_update_flags(test_namespace, workspace=workspace2)
    assert flag1_obj.value is True
    assert flag2_obj.value is True

    await clear_all_update_flags(test_namespace, workspace=workspace1)
    assert flag1_obj.value is False
    assert flag2_obj.value is True

    await get_update_flag("ns_a", workspace=workspace1)
    await get_update_flag("ns_b", workspace=workspace1)
    await get_update_flag("ns_c", workspace=workspace2)

    await set_all_update_flags("ns_a", workspace=workspace1)
    await set_all_update_flags("ns_b", workspace=workspace1)
    await set_all_update_flags("ns_c", workspace=workspace2)

    status1 = await get_all_update_flags_status(workspace=workspace1)
    assert set(status1.keys()) == {
        f"{workspace1}:{test_namespace}",
        f"{workspace1}:ns_a",
        f"{workspace1}:ns_b",
    }
    assert status1[f"{workspace1}:ns_a"] == [True]
    assert status1[f"{workspace1}:ns_b"] == [True]
    # test_namespace was cleared for workspace1 above and must stay cleared.
    assert status1[f"{workspace1}:{test_namespace}"] == [False]

    status2 = await get_all_update_flags_status(workspace=workspace2)
    assert set(status2.keys()) == {
        f"{workspace2}:{test_namespace}",
        f"{workspace2}:ns_c",
    }
    assert status2[f"{workspace2}:{test_namespace}"] == [True]
    assert status2[f"{workspace2}:ns_c"] == [True]


async def test_json_kv_storage_workspace_isolation(tmp_path: Path):
    """JsonKVStorage isolates both persisted directories and in-memory data."""

    async def mock_embedding_func(texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), 8), dtype=np.float32)

    global_config = {
        "working_dir": str(tmp_path),
        "embedding_batch_num": 10,
    }

    storage1 = JsonKVStorage(
        namespace="entities",
        workspace="workspace1",
        global_config=global_config,
        embedding_func=mock_embedding_func,
    )
    storage2 = JsonKVStorage(
        namespace="entities",
        workspace="workspace2",
        global_config=global_config,
        embedding_func=mock_embedding_func,
    )

    await storage1.initialize()
    await storage2.initialize()

    await storage1.upsert(
        {
            "entity1": {
                "content": "Data from workspace1 - AI Research",
                "type": "entity",
            },
            "entity2": {
                "content": "Data from workspace1 - Machine Learning",
                "type": "entity",
            },
        }
    )
    await storage1.index_done_callback()

    await storage2.upsert(
        {
            "entity1": {
                "content": "Data from workspace2 - Deep Learning",
                "type": "entity",
            },
            "entity2": {
                "content": "Data from workspace2 - Neural Networks",
                "type": "entity",
            },
        }
    )
    await storage2.index_done_callback()

    result1_entity1 = await storage1.get_by_id("entity1")
    result1_entity2 = await storage1.get_by_id("entity2")
    result2_entity1 = await storage2.get_by_id("entity1")
    result2_entity2 = await storage2.get_by_id("entity2")

    assert result1_entity1 is not None
    assert result1_entity2 is not None
    assert result2_entity1 is not None
    assert result2_entity2 is not None
    assert result1_entity1.get("content") == "Data from workspace1 - AI Research"
    assert result1_entity2.get("content") == "Data from workspace1 - Machine Learning"
    assert result2_entity1.get("content") == "Data from workspace2 - Deep Learning"
    assert result2_entity2.get("content") == "Data from workspace2 - Neural Networks"
    assert (tmp_path / "workspace1").exists()
    assert (tmp_path / "workspace2").exists()


async def test_lightrag_end_to_end_workspace_isolation(tmp_path: Path):
    """Concurrent LightRAG inserts keep workspace directories and persisted docs isolated."""

    def create_mock_llm_func(workspace_name: str):
        async def mock_llm_func(
            prompt, system_prompt=None, history_messages=[], **kwargs
        ) -> str:
            await asyncio.sleep(0)
            if workspace_name == "project_a":
                return (
                    "entity<|#|>Artificial Intelligence<|#|>concept<|#|>AI is a field of computer science.\n"
                    "entity<|#|>Machine Learning<|#|>concept<|#|>Machine Learning is a subset of AI.\n"
                    "relation<|#|>Machine Learning<|#|>Artificial Intelligence<|#|>subset<|#|>ML is a subset of AI.\n"
                    "<|COMPLETE|>"
                )
            return (
                "entity<|#|>Deep Learning<|#|>concept<|#|>Deep Learning uses multi-layer neural networks.\n"
                "entity<|#|>Neural Networks<|#|>concept<|#|>Computing systems inspired by biological networks.\n"
                "relation<|#|>Deep Learning<|#|>Neural Networks<|#|>uses<|#|>Deep Learning uses Neural Networks.\n"
                "<|COMPLETE|>"
            )

        return mock_llm_func

    async def mock_embedding_func(texts: list[str]) -> np.ndarray:
        await asyncio.sleep(0)
        return np.ones((len(texts), 8), dtype=np.float32)

    class _SimpleTokenizerImpl:
        def encode(self, content: str) -> list[int]:
            return [ord(ch) for ch in content]

        def decode(self, tokens: list[int]) -> str:
            return "".join(chr(t) for t in tokens)

    tokenizer = Tokenizer("mock-tokenizer", _SimpleTokenizerImpl())

    rag1 = LightRAG(
        working_dir=str(tmp_path),
        workspace="project_a",
        llm_model_func=create_mock_llm_func("project_a"),
        embedding_func=EmbeddingFunc(
            embedding_dim=8,
            max_token_size=8192,
            func=mock_embedding_func,
        ),
        tokenizer=tokenizer,
    )
    rag2 = LightRAG(
        working_dir=str(tmp_path),
        workspace="project_b",
        llm_model_func=create_mock_llm_func("project_b"),
        embedding_func=EmbeddingFunc(
            embedding_dim=8,
            max_token_size=8192,
            func=mock_embedding_func,
        ),
        tokenizer=tokenizer,
    )

    await rag1.initialize_storages()
    await rag2.initialize_storages()
    try:
        text_for_project_a = (
            "This document is about Artificial Intelligence and Machine Learning. "
            "AI is transforming the world."
        )
        text_for_project_b = (
            "This document is about Deep Learning and Neural Networks. "
            "Deep learning uses multiple layers."
        )

        await asyncio.gather(
            rag1.ainsert(text_for_project_a),
            rag2.ainsert(text_for_project_b),
        )

        project_a_dir = tmp_path / "project_a"
        project_b_dir = tmp_path / "project_b"
        assert project_a_dir.exists()
        assert project_b_dir.exists()

        docs_a_file = project_a_dir / "kv_store_full_docs.json"
        docs_b_file = project_b_dir / "kv_store_full_docs.json"
        assert docs_a_file.exists(), "project_a full_docs snapshot was not persisted"
        assert docs_b_file.exists(), "project_b full_docs snapshot was not persisted"

        docs_a_content = json.loads(docs_a_file.read_text(encoding="utf-8"))
        docs_b_content = json.loads(docs_b_file.read_text(encoding="utf-8"))
        assert docs_a_content != docs_b_content

        docs_a_str = json.dumps(docs_a_content)
        docs_b_str = json.dumps(docs_b_content)

        assert "Artificial Intelligence" in docs_a_str
        assert "Machine Learning" in docs_a_str
        assert "Deep Learning" not in docs_a_str
        assert "Neural Networks" not in docs_a_str

        assert "Deep Learning" in docs_b_str
        assert "Neural Networks" in docs_b_str
        assert "Artificial Intelligence" not in docs_b_str
    finally:
        await rag1.finalize_storages()
        await rag2.finalize_storages()


class TestWorkspaceMigrationIsolation:
    """Workspace-scoped legacy table migration in PostgreSQL."""

    async def test_migration_without_workspace_raises_error(self):
        """setup_table called with workspace=None raises ValueError."""
        db = AsyncMock()
        # Real default; an auto-created mock breaks _check_vector_index_dimension.
        db.vector_index_type = None

        with pytest.raises(ValueError, match="workspace must be provided"):
            await PGVectorStorage.setup_table(
                db,
                "lightrag_doc_chunks_model_1536d",
                workspace=None,
                embedding_dim=1536,
                legacy_table_name="lightrag_doc_chunks",
                base_table="lightrag_doc_chunks",
            )

    async def test_no_cross_workspace_contamination(self):
        """Migrating workspace_b only queries and copies workspace_b data."""
        db = AsyncMock()
        db.vector_index_type = None
        db._create_vector_index.return_value = None

        queried_workspace = None
        new_table_count = {"count": 0}
        mock_records_b = [
            {
                "id": "b1",
                "workspace": "workspace_b",
                "content": "content_b1",
                "content_vector": [0.3] * 1536,
            },
        ]

        async def query_side_effect(sql, params, **kwargs):
            nonlocal queried_workspace
            multirows = kwargs.get("multirows", False)
            sql_upper = sql.upper()

            if (
                "COUNT(*)" in sql_upper
                and "MODEL_1536D" in sql_upper
                and "WHERE WORKSPACE" in sql_upper
            ):
                return new_table_count
            if "COUNT(*)" in sql_upper and "WHERE WORKSPACE" in sql_upper:
                queried_workspace = params[0] if params else None
                return {"count": 1}
            if (
                "COUNT(*)" in sql_upper
                and "LIGHTRAG" in sql_upper
                and "WHERE WORKSPACE" not in sql_upper
            ):
                return {"count": 3}
            if "SELECT" in sql_upper and "FROM" in sql_upper and multirows:
                workspace = params[0] if params else None
                if workspace == "workspace_b":
                    if "id >" in sql.lower():
                        last_id = params[1] if len(params) > 1 else None
                        found_idx = -1
                        for i, rec in enumerate(mock_records_b):
                            if rec["id"] == last_id:
                                found_idx = i
                                break
                        if found_idx >= 0:
                            return mock_records_b[found_idx + 1 :]
                        return []
                    return mock_records_b
                return []
            return {}

        db.query.side_effect = query_side_effect
        db.execute = AsyncMock()

        async def check_table_exists_side_effect(name):
            return name.lower() == "lightrag_doc_chunks"

        db.check_table_exists = AsyncMock(side_effect=check_table_exists_side_effect)

        migration_executed = []

        def record_migration(sql, params):
            migration_executed.append(True)
            new_table_count["count"] = 1

        db.query = AsyncMock(
            side_effect=answer_legacy_migration(db.query.side_effect, record_migration)
        )

        await PGVectorStorage.setup_table(
            db,
            "LIGHTRAG_DOC_CHUNKS_model_1536d",
            workspace="workspace_b",
            embedding_dim=1536,
            legacy_table_name="LIGHTRAG_DOC_CHUNKS",
            base_table="LIGHTRAG_DOC_CHUNKS",
        )

        assert len(migration_executed) > 0
        assert queried_workspace == "workspace_b"
