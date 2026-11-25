"""Live-Neo4j integration tests for full-text indexes and APOC workspace chars.

Requires a running Neo4j instance configured via ``NEO4J_URI`` (and APOC for
the workspace-character subgraph test).
"""

from __future__ import annotations

import asyncio
import os

import numpy as np
import pytest

from lightrag.kg.shared_storage import initialize_share_data

pytestmark = [pytest.mark.integration, pytest.mark.requires_db]


async def mock_embedding_func(texts):
    return np.random.rand(len(texts), 10)


async def _make_live_storage(namespace: str, workspace: str):
    from lightrag.kg.neo4j_impl import Neo4JStorage

    storage = Neo4JStorage(
        namespace=namespace,
        workspace=workspace,
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.5},
            "working_dir": os.environ.get("WORKING_DIR", "./rag_storage"),
        },
        embedding_func=mock_embedding_func,
    )
    await storage.initialize()
    await storage.drop()
    return storage


async def _has_apoc(storage) -> bool:
    try:
        async with storage._driver.session(database=storage._DATABASE) as session:
            result = await session.run("RETURN apoc.version() AS v")
            await result.consume()
        return True
    except Exception:
        return False


@pytest.fixture
async def neo4j_storage():
    """Initialize Neo4j storage for integration testing."""
    if not os.getenv("NEO4J_URI"):
        pytest.skip("Neo4j not configured (NEO4J_URI not set)")

    initialize_share_data()
    storage = await _make_live_storage("test_fulltext_index", "test_workspace")
    yield storage
    await storage.drop()
    await storage.finalize()


@pytest.mark.integration
@pytest.mark.requires_db
async def test_fulltext_index_creation(neo4j_storage):
    """Full-text index is created with the workspace-specific name."""
    storage = neo4j_storage
    workspace_label = storage._get_workspace_label()
    expected_index_name = f"entity_id_fulltext_idx_{workspace_label}"

    async with storage._driver.session(database=storage._DATABASE) as session:
        result = await session.run("SHOW FULLTEXT INDEXES")
        indexes = await result.data()
        await result.consume()

        index_names = [idx["name"] for idx in indexes]
        assert expected_index_name in index_names, (
            f"Expected index '{expected_index_name}' not found. Found indexes: {index_names}"
        )


@pytest.mark.integration
@pytest.mark.requires_db
async def test_search_labels_with_workspace_index(neo4j_storage):
    """``search_labels`` uses the workspace-specific index and returns matches."""
    storage = neo4j_storage

    test_nodes = [
        {
            "entity_id": "Artificial Intelligence",
            "description": "AI field",
            "keywords": "AI,ML,DL",
            "entity_type": "Technology",
        },
        {
            "entity_id": "Machine Learning",
            "description": "ML subfield",
            "keywords": "supervised,unsupervised",
            "entity_type": "Technology",
        },
        {
            "entity_id": "Deep Learning",
            "description": "DL subfield",
            "keywords": "neural networks",
            "entity_type": "Technology",
        },
        {
            "entity_id": "Natural Language Processing",
            "description": "NLP field",
            "keywords": "text,language",
            "entity_type": "Technology",
        },
    ]

    for node_data in test_nodes:
        await storage.upsert_node(node_data["entity_id"], node_data)

    # Give the eventually-consistent index time to populate.
    await asyncio.sleep(2)

    results = await storage.search_labels("Learning", limit=10)
    assert len(results) > 0, "search_labels should return results for 'Learning'"
    assert any("Learning" in result for result in results)

    results_lower = await storage.search_labels("learning", limit=10)
    assert len(results_lower) > 0, "search_labels should be case-insensitive"

    results_partial = await storage.search_labels("Intelli", limit=10)
    assert len(results_partial) > 0
    assert any("Intelligence" in result for result in results_partial)


@pytest.mark.integration
@pytest.mark.requires_db
async def test_search_labels_with_hyphen(neo4j_storage):
    """Label fragments with Lucene reserved characters still match via full-text."""
    storage = neo4j_storage

    test_nodes = [
        {"entity_id": "tb-alpha", "entity_type": "Test"},
        {"entity_id": "tb-beta", "entity_type": "Test"},
        {"entity_id": "tbcd", "entity_type": "Test"},
    ]

    for node_data in test_nodes:
        await storage.upsert_node(node_data["entity_id"], node_data)

    await asyncio.sleep(2)

    results = await storage.search_labels("tb", limit=10)
    for expected in ("tb-alpha", "tb-beta", "tbcd"):
        assert expected in results, f"'{expected}' should match 'tb', got {results}"

    results_hyphen = await storage.search_labels("tb-", limit=10)
    assert "tb-alpha" in results_hyphen
    assert "tb-beta" in results_hyphen

    results_exact = await storage.search_labels("tb-alpha", limit=10)
    assert "tb-alpha" in results_exact


@pytest.mark.integration
@pytest.mark.requires_db
async def test_search_labels_chinese_text(neo4j_storage):
    """``search_labels`` works with Chinese text using the CJK analyzer."""
    storage = neo4j_storage

    chinese_nodes = [
        {
            "entity_id": "人工智能",
            "description": "人工智能领域",
            "keywords": "AI,机器学习",
            "entity_type": "技术",
        },
        {
            "entity_id": "机器学习",
            "description": "机器学习子领域",
            "keywords": "监督学习,无监督学习",
            "entity_type": "技术",
        },
        {
            "entity_id": "深度学习",
            "description": "深度学习子领域",
            "keywords": "神经网络",
            "entity_type": "技术",
        },
    ]

    for node_data in chinese_nodes:
        await storage.upsert_node(node_data["entity_id"], node_data)

    await asyncio.sleep(2)

    results = await storage.search_labels("学习", limit=10)
    assert len(results) > 0
    assert any("学习" in result for result in results)


@pytest.mark.integration
@pytest.mark.requires_db
async def test_search_labels_fallback_to_contains(neo4j_storage):
    """``search_labels`` returns matches even before full-text indexing settles."""
    storage = neo4j_storage

    test_nodes = [
        {
            "entity_id": "Test Node Alpha",
            "description": "Test node",
            "keywords": "test",
            "entity_type": "Test",
        },
        {
            "entity_id": "Test Node Beta",
            "description": "Test node",
            "keywords": "test",
            "entity_type": "Test",
        },
    ]

    for node_data in test_nodes:
        await storage.upsert_node(node_data["entity_id"], node_data)

    results = await storage.search_labels("Alpha", limit=10)
    assert len(results) > 0
    assert "Test Node Alpha" in results


@pytest.mark.integration
@pytest.mark.requires_db
async def test_multiple_workspaces_have_separate_indexes(neo4j_storage):
    """Different workspaces maintain separate full-text indexes."""
    storage1 = neo4j_storage
    storage2 = await _make_live_storage("test_fulltext_index", "test_workspace_2")

    try:
        async with storage1._driver.session(database=storage1._DATABASE) as session:
            result = await session.run("SHOW FULLTEXT INDEXES")
            indexes = await result.data()
            await result.consume()

            index_names = [idx["name"] for idx in indexes]
            workspace1_index = (
                f"entity_id_fulltext_idx_{storage1._get_workspace_label()}"
            )
            workspace2_index = (
                f"entity_id_fulltext_idx_{storage2._get_workspace_label()}"
            )

            assert workspace1_index in index_names
            assert workspace2_index in index_names
    finally:
        try:
            async with storage2._driver.session(database=storage2._DATABASE) as session:
                index_name = storage2._get_fulltext_index_name(
                    storage2._get_workspace_label()
                )
                result = await session.run(f"DROP INDEX {index_name} IF EXISTS")
                await result.consume()
        except Exception:
            pass
        await storage2.drop()
        await storage2.finalize()


@pytest.mark.integration
@pytest.mark.requires_db
@pytest.mark.parametrize(
    "workspace",
    ["tenant_1", "tenant:1", "acme,eu", "a|b", "+team", ">x"],
)
async def test_get_knowledge_graph_returns_full_subgraph(workspace):
    """``get_knowledge_graph`` returns the full APOC subgraph for any workspace."""
    if not os.getenv("NEO4J_URI"):
        pytest.skip("Neo4j not configured (NEO4J_URI not set)")

    initialize_share_data()
    storage = await _make_live_storage("test_kg_workspace_chars", workspace)
    try:
        if not await _has_apoc(storage):
            pytest.skip("APOC is not installed on this Neo4j instance")

        async def _no_fallback(*args, **kwargs):
            raise AssertionError("APOC path failed and fell back to plain Cypher")

        storage._robust_fallback = _no_fallback

        for entity in ("A", "B", "C"):
            await storage.upsert_node(
                entity, {"entity_id": entity, "entity_type": "concept"}
            )
        await storage.upsert_edge("A", "B", {"weight": 1.0})
        await storage.upsert_edge("B", "C", {"weight": 1.0})

        kg = await storage.get_knowledge_graph("A", max_depth=3)

        assert sorted(n.labels[0] for n in kg.nodes) == ["A", "B", "C"]
        assert len(kg.edges) == 2
        assert kg.is_truncated is False
    finally:
        await storage.drop()
        await storage.finalize()
