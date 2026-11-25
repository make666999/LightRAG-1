"""Offline security & index-isolation tests for the OpenSearch backend.

Covers:
1. CWE-89 injection guards: wildcard escaping in ``search_labels`` and PPL
   string literal escaping in ``_escape_ppl``.
2. Workspace/index collision isolation: ``_meta`` ownership markers across all
   four storage classes, ambiguous ``{workspace}_{namespace}`` joins, legacy
   adoption races, read-only mapping writes, and configuration container guards.
"""

from __future__ import annotations

import asyncio
import logging
import re
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip(
    "opensearchpy",
    reason="opensearchpy is required for OpenSearch storage tests",
)

from opensearchpy.exceptions import OpenSearchException  # type: ignore  # noqa: E402

from lightrag.kg.opensearch_impl import (  # noqa: E402
    _EDGE_ID_CANONICAL_META_FLAG,
    _FINAL_NAMESPACE_META_KEY,
    _WORKSPACE_META_KEY,
    ClientManager,
    OpenSearchDocStatusStorage,
    OpenSearchGraphStorage,
    OpenSearchKVStorage,
    OpenSearchVectorDBStorage,
    WorkspaceIndexCollisionError,
    _build_index_name,
    _claim_index_for_workspace,
    _sanitize_index_name,
)
from lightrag.namespace import CONFIG_CONTAINER_TAG  # noqa: E402
from lightrag.utils import logger as lightrag_logger  # noqa: E402

from ._helpers import (  # noqa: E402
    FakeCluster,
    FixedEmbeddingFunc,
    MockEmbeddingFunc,
    default_global_config,
    make_client,
    patch_opensearch_locks,
)

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def patch_locks():
    with patch_opensearch_locks(patch_shard_doc=True):
        yield


@pytest.fixture
def cluster():
    return FakeCluster()


@pytest.fixture
def global_config():
    return default_global_config()


@pytest.fixture
def graph_storage(global_config):
    client = make_client()
    client.indices.exists = AsyncMock(return_value=True)
    with patch.object(ClientManager, "get_client", return_value=client):
        storage = OpenSearchGraphStorage(
            namespace="test_graph",
            global_config=global_config,
            embedding_func=MockEmbeddingFunc(),
        )
        storage.client = client
        storage._indices_ready = True
        storage._ppl_graphlookup_available = True
        yield storage


# ---------------------------------------------------------------------------
# CWE-89 injection guards: wildcard & PPL escaping
# ---------------------------------------------------------------------------


def _extract_wildcard_value(client: AsyncMock) -> str:
    call_kwargs = client.search.call_args
    body = call_kwargs.kwargs.get("body") or call_kwargs[1].get("body")
    for clause in body["query"]["bool"]["should"]:
        if "wildcard" in clause:
            return clause["wildcard"]["entity_id"]["value"]
    raise AssertionError("wildcard clause should exist")


class TestWildcardInjection:
    """Wildcard special characters (*, ?, \\) in user input must be escaped."""

    @pytest.mark.asyncio
    async def test_wildcard_chars_escaped_in_search_labels(self, graph_storage):
        client = graph_storage.client
        client.search.return_value = {"hits": {"hits": []}}

        await graph_storage.search_labels("test*?foo")

        assert client.search.called
        wildcard_clause = _extract_wildcard_value(client)
        inner_value = wildcard_clause[1:-1]
        assert "\\*" in inner_value, (
            f"User's '*' should be escaped as '\\*' in wildcard, got: {wildcard_clause}"
        )
        assert "\\?" in inner_value, (
            f"User's '?' should be escaped as '\\?' in wildcard, got: {wildcard_clause}"
        )

    @pytest.mark.asyncio
    async def test_wildcard_heavy_pattern_not_exploitable(self, graph_storage):
        client = graph_storage.client
        client.search.return_value = {"hits": {"hits": []}}

        await graph_storage.search_labels("?" * 50)

        wildcard_clause = _extract_wildcard_value(client)
        inner = wildcard_clause[1:-1]
        unescaped_q = re.findall(r"(?<!\\)\?", inner)
        assert len(unescaped_q) == 0, (
            f"Found {len(unescaped_q)} unescaped '?' in wildcard pattern: {wildcard_clause}"
        )

    @pytest.mark.asyncio
    async def test_backslash_escaped_in_wildcard(self, graph_storage):
        client = graph_storage.client
        client.search.return_value = {"hits": {"hits": []}}

        await graph_storage.search_labels("test\\*")

        wildcard_clause = _extract_wildcard_value(client)
        inner = wildcard_clause[1:-1]
        assert "\\\\" in inner and "\\*" in inner, (
            f"Backslash and * from user should be escaped in wildcard: {wildcard_clause}"
        )


class TestPPLInjection:
    """PPL string literal escaping handles quotes, backslashes, and control chars."""

    def test_escape_ppl_basic_quote(self, graph_storage):
        result = graph_storage._escape_ppl("it's a test")
        assert "'" not in result.replace("\\'", ""), (
            f"Unescaped quote found in: {result}"
        )

    def test_escape_ppl_backslash(self, graph_storage):
        assert graph_storage._escape_ppl("test\\path") == "test\\\\path"

    def test_escape_ppl_newline_and_control_chars(self, graph_storage):
        result = graph_storage._escape_ppl("line1\nline2\rline3\t")
        assert "\n" not in result, f"Raw newline in PPL literal: {result!r}"
        assert "\r" not in result, f"Raw carriage return in PPL literal: {result!r}"

    def test_escape_ppl_pipe_in_quotes_safe(self, graph_storage):
        result = graph_storage._escape_ppl("entity | stats count()")
        assert result == "entity | stats count()"


# ---------------------------------------------------------------------------
# Workspace / index collision isolation
# ---------------------------------------------------------------------------


STORAGE_CASES = [
    pytest.param(OpenSearchKVStorage, "text_chunks", id="kv"),
    pytest.param(OpenSearchDocStatusStorage, "doc_status", id="doc_status"),
    pytest.param(OpenSearchGraphStorage, "chunk_entity_relation", id="graph"),
    pytest.param(OpenSearchVectorDBStorage, "entities", id="vector"),
]


async def _initialize_storage(cls, namespace, workspace, global_config, cluster):
    storage = cls(
        namespace=namespace,
        global_config=global_config,
        embedding_func=FixedEmbeddingFunc(model_name="mock-embed", embedding_dim=8),
        workspace=workspace,
    )
    with patch.object(ClientManager, "get_client", return_value=cluster.client()):
        await storage.initialize()
    return storage


class TestCaseVariantWorkspacesCollide:
    def test_index_names_really_collide(self):
        assert _sanitize_index_name("TeamA_text_chunks") == "teama_text_chunks"
        assert _sanitize_index_name("teama_text_chunks") == "teama_text_chunks"
        assert _sanitize_index_name("v1.0_text_chunks") == "v1_0_text_chunks"
        assert _sanitize_index_name("v1_0_text_chunks") == "v1_0_text_chunks"

    @pytest.mark.parametrize("cls,namespace", STORAGE_CASES)
    @pytest.mark.asyncio
    async def test_second_workspace_is_refused(
        self, cls, namespace, global_config, cluster
    ):
        await _initialize_storage(cls, namespace, "TeamA", global_config, cluster)

        with pytest.raises(WorkspaceIndexCollisionError) as excinfo:
            await _initialize_storage(cls, namespace, "teama", global_config, cluster)

        message = str(excinfo.value)
        assert f"TeamA_{namespace}" in message
        assert f"teama_{namespace}" in message
        assert "Rename one of the" in message
        assert message.count("drop") == message.count("Do NOT drop")

    @pytest.mark.parametrize("cls,namespace", STORAGE_CASES)
    @pytest.mark.asyncio
    async def test_punctuation_variant_is_refused(
        self, cls, namespace, global_config, cluster
    ):
        await _initialize_storage(cls, namespace, "v1.0", global_config, cluster)

        with pytest.raises(WorkspaceIndexCollisionError):
            await _initialize_storage(cls, namespace, "v1_0", global_config, cluster)

    @pytest.mark.parametrize("cls,namespace", STORAGE_CASES)
    @pytest.mark.asyncio
    async def test_same_workspace_reattaches(
        self, cls, namespace, global_config, cluster
    ):
        await _initialize_storage(cls, namespace, "TeamA", global_config, cluster)
        await _initialize_storage(cls, namespace, "TeamA", global_config, cluster)
        await _initialize_storage(cls, namespace, "TeamA", global_config, cluster)

    @pytest.mark.parametrize("cls,namespace", STORAGE_CASES)
    @pytest.mark.asyncio
    async def test_distinct_workspaces_are_untouched(
        self, cls, namespace, global_config, cluster
    ):
        await _initialize_storage(cls, namespace, "TeamA", global_config, cluster)
        await _initialize_storage(cls, namespace, "TeamB", global_config, cluster)


class TestAmbiguousJoinedNamespace:
    @pytest.mark.asyncio
    async def test_joined_names_really_collide(self, global_config):
        kv = OpenSearchKVStorage(
            namespace="text_chunks",
            global_config=global_config,
            embedding_func=FixedEmbeddingFunc(),
            workspace="foo",
        )
        vector = OpenSearchVectorDBStorage(
            namespace="chunks",
            global_config=global_config,
            embedding_func=FixedEmbeddingFunc(),
            workspace="foo_text",
        )
        assert kv._index_name == vector._index_name == "foo_text_chunks"
        assert kv.final_namespace == vector.final_namespace == "foo_text_chunks"

    @pytest.mark.asyncio
    async def test_different_workspace_same_joined_name_is_refused(
        self, global_config, cluster
    ):
        await _initialize_storage(
            OpenSearchKVStorage, "text_chunks", "foo", global_config, cluster
        )
        with pytest.raises(WorkspaceIndexCollisionError) as excinfo:
            await _initialize_storage(
                OpenSearchVectorDBStorage, "chunks", "foo_text", global_config, cluster
            )
        message = str(excinfo.value)
        assert "'foo'" in message
        assert "'foo_text'" in message

    @pytest.mark.asyncio
    async def test_partial_marker_is_not_adopted(self, cluster):
        cluster.mappings["teama_text_chunks"] = {
            "_meta": {_WORKSPACE_META_KEY: "TeamA"},
            "properties": {},
        }
        client = cluster.client()
        with pytest.raises(WorkspaceIndexCollisionError):
            await _claim_index_for_workspace(
                client, "teama_text_chunks", "TeamA", "TeamA_text_chunks"
            )
        client.indices.put_mapping.assert_not_awaited()


class TestMarkerIsRecorded:
    @pytest.mark.asyncio
    async def test_every_created_index_carries_the_marker(self, global_config, cluster):
        await _initialize_storage(
            OpenSearchGraphStorage,
            "chunk_entity_relation",
            "TeamA",
            global_config,
            cluster,
        )
        assert set(cluster.mappings) == {
            "teama_chunk_entity_relation-nodes",
            "teama_chunk_entity_relation-edges",
        }
        for mappings in cluster.mappings.values():
            meta = mappings["_meta"]
            assert meta[_WORKSPACE_META_KEY] == "TeamA"
            assert meta[_FINAL_NAMESPACE_META_KEY] == "TeamA_chunk_entity_relation"
        edges_meta = cluster.mappings["teama_chunk_entity_relation-edges"]["_meta"]
        assert edges_meta[_EDGE_ID_CANONICAL_META_FLAG] is True

    @pytest.mark.asyncio
    async def test_drop_and_recreate_remarks_the_index(self, global_config, cluster):
        storage = await _initialize_storage(
            OpenSearchKVStorage, "text_chunks", "TeamA", global_config, cluster
        )
        with patch.object(ClientManager, "get_client", return_value=cluster.client()):
            await storage.drop()
            await storage._ensure_index_ready()
        assert cluster.mappings["teama_text_chunks"]["_meta"] == {
            _WORKSPACE_META_KEY: "TeamA",
            _FINAL_NAMESPACE_META_KEY: "TeamA_text_chunks",
        }


class TestLegacyIndexAdoption:
    @pytest.mark.asyncio
    async def test_unmarked_index_is_adopted(self, global_config, cluster):
        cluster.mappings["teama_text_chunks"] = {
            "properties": {"__mirrored_id": {"type": "keyword"}}
        }
        await _initialize_storage(
            OpenSearchKVStorage, "text_chunks", "TeamA", global_config, cluster
        )
        assert cluster.mappings["teama_text_chunks"]["_meta"] == {
            _WORKSPACE_META_KEY: "TeamA",
            _FINAL_NAMESPACE_META_KEY: "TeamA_text_chunks",
        }

    @pytest.mark.asyncio
    async def test_adoption_is_first_come_first_served(self, global_config, cluster):
        cluster.mappings["teama_text_chunks"] = {"properties": {}}
        await _initialize_storage(
            OpenSearchKVStorage, "text_chunks", "TeamA", global_config, cluster
        )
        with pytest.raises(WorkspaceIndexCollisionError):
            await _initialize_storage(
                OpenSearchKVStorage, "text_chunks", "teama", global_config, cluster
            )

    @pytest.mark.asyncio
    async def test_adoption_preserves_existing_meta(self, cluster):
        cluster.mappings["teama_graph-edges"] = {
            "_meta": {_EDGE_ID_CANONICAL_META_FLAG: True},
            "properties": {},
        }
        client = cluster.client()
        await _claim_index_for_workspace(
            client, "teama_graph-edges", "TeamA", "TeamA_graph"
        )
        assert cluster.mappings["teama_graph-edges"]["_meta"] == {
            _EDGE_ID_CANONICAL_META_FLAG: True,
            _WORKSPACE_META_KEY: "TeamA",
            _FINAL_NAMESPACE_META_KEY: "TeamA_graph",
        }


class TestConcurrentAdoption:
    @pytest.mark.asyncio
    async def test_same_workspace_racing_adoption_is_idempotent(self, cluster):
        cluster.mappings["teama_text_chunks"] = {"properties": {}}
        client = cluster.client()
        await asyncio.gather(
            *[
                _claim_index_for_workspace(
                    client, "teama_text_chunks", "TeamA", "TeamA_text_chunks"
                )
                for _ in range(4)
            ]
        )
        assert cluster.mappings["teama_text_chunks"]["_meta"] == {
            _WORKSPACE_META_KEY: "TeamA",
            _FINAL_NAMESPACE_META_KEY: "TeamA_text_chunks",
        }

    @pytest.mark.asyncio
    async def test_losing_an_adoption_race_is_detected(self, cluster):
        cluster.mappings["teama_text_chunks"] = {"properties": {}}
        client = cluster.client()
        original_put = cluster.put_mapping

        async def steal_after_write(index, body):
            await original_put(index, body)
            cluster.mappings[index]["_meta"] = {
                _WORKSPACE_META_KEY: "teama",
                _FINAL_NAMESPACE_META_KEY: "teama_text_chunks",
            }

        client.indices.put_mapping = AsyncMock(side_effect=steal_after_write)
        with pytest.raises(WorkspaceIndexCollisionError):
            await _claim_index_for_workspace(
                client, "teama_text_chunks", "TeamA", "TeamA_text_chunks"
            )

    @pytest.mark.asyncio
    async def test_invisible_write_is_not_a_collision(self, cluster):
        cluster.mappings["teama_text_chunks"] = {"properties": {}}
        client = cluster.client()

        async def swallow_write(index, body):
            return None

        client.indices.put_mapping = AsyncMock(side_effect=swallow_write)
        await _claim_index_for_workspace(
            client, "teama_text_chunks", "TeamA", "TeamA_text_chunks"
        )
        assert "_meta" not in cluster.mappings["teama_text_chunks"]


class TestUnwritableMapping:
    @pytest.mark.asyncio
    async def test_startup_survives_a_rejected_marker_write(
        self, global_config, cluster, lightrag_caplog
    ):
        cluster.mappings["teama_text_chunks"] = {
            "properties": {"__mirrored_id": {"type": "keyword"}}
        }
        client = cluster.client()
        client.indices.put_mapping = AsyncMock(
            side_effect=OpenSearchException("security_exception: no write privilege")
        )
        storage = OpenSearchKVStorage(
            namespace="text_chunks",
            global_config=global_config,
            embedding_func=FixedEmbeddingFunc(),
            workspace="TeamA",
        )

        with lightrag_caplog.at_level(logging.WARNING, logger=lightrag_logger.name):
            with patch.object(ClientManager, "get_client", return_value=client):
                await storage.initialize()

        assert "teama_text_chunks" in lightrag_caplog.text
        assert "stays unmarked" in lightrag_caplog.text
        assert "_meta" not in cluster.mappings["teama_text_chunks"]

    @pytest.mark.asyncio
    async def test_collision_still_raises_without_write_access(self, cluster):
        cluster.mappings["teama_text_chunks"] = {
            "_meta": {
                _WORKSPACE_META_KEY: "teama",
                _FINAL_NAMESPACE_META_KEY: "teama_text_chunks",
            }
        }
        client = cluster.client()
        client.indices.put_mapping = AsyncMock(
            side_effect=OpenSearchException("security_exception: no write privilege")
        )
        with pytest.raises(WorkspaceIndexCollisionError):
            await _claim_index_for_workspace(
                client, "teama_text_chunks", "TeamA", "TeamA_text_chunks"
            )
        client.indices.put_mapping.assert_not_awaited()


@pytest.mark.parametrize(
    "alias", [".lightrag_config", "x_lightrag_config", "+LightRAG_config", "ordinary"]
)
@pytest.mark.parametrize("via_env", [False, True])
def test_no_workspace_spelling_moves_the_configuration_container(
    monkeypatch, alias, via_env
):
    monkeypatch.delenv("OPENSEARCH_WORKSPACE", raising=False)
    if via_env:
        monkeypatch.setenv("OPENSEARCH_WORKSPACE", alias)
    effective, final_ns, index_name = _build_index_name(
        "ordinary" if via_env else alias, "config"
    )
    assert effective == CONFIG_CONTAINER_TAG
    assert final_ns == "_lightrag_config_config"
    assert index_name == "x_lightrag_config_config"


def test_a_non_configuration_open_may_not_normalize_onto_the_container():
    with pytest.raises(ValueError, match="configuration container's index"):
        _build_index_name("_lightrag", "config_config")


def test_ordinary_names_and_a_tenant_named_after_the_tag_remain_available():
    assert _build_index_name("v1.0", "text_chunks")[2] == "v1_0_text_chunks"
    assert (
        _build_index_name("_lightrag_config", "text_chunks")[2]
        == "x_lightrag_config_text_chunks"
    )
