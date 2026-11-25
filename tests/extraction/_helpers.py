"""Shared lightweight test doubles and config builders for ``tests/extraction/``."""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import AsyncMock

from lightrag.utils import Tokenizer, TokenizerInterface

MINIMAL_EXTRACTION_RESULT = (
    "(entity<|#|>TEST_ENTITY<|#|>CONCEPT<|#|>A test entity)<|COMPLETE|>"
)


class CharTokenizer(TokenizerInterface):
    """Simple 1:1 character-to-token mapping for offline extraction tests."""

    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


class CountingTokenizer(CharTokenizer):
    """``CharTokenizer`` that records every string passed to ``encode()``."""

    def __init__(self) -> None:
        self.encoded: list[str] = []

    def encode(self, content: str) -> list[int]:
        self.encoded.append(content)
        return super().encode(content)


def make_dummy_tokenizer(name: str = "dummy") -> Tokenizer:
    """Build a ``Tokenizer`` backed by :class:`CharTokenizer`."""
    return Tokenizer(name, CharTokenizer())


def make_extract_global_config(
    *,
    extract_func=None,
    entity_extract_max_gleaning: int = 0,
    use_json: bool = False,
    addon_params: dict | None = None,
    prompt_profile: dict | None = None,
    tokenizer: Tokenizer | None = None,
    llm_model_max_async: int = 1,
    enable_llm_cache_for_entity_extract: bool | None = None,
    **extra,
) -> dict:
    """Build a minimal ``global_config`` dict for ``extract_entities``."""
    llm = extract_func if extract_func is not None else AsyncMock(return_value="")
    config = {
        "llm_model_func": llm,
        "role_llm_funcs": {
            "extract": llm,
            "keyword": llm,
            "query": llm,
            "vlm": llm,
        },
        "entity_extract_max_gleaning": entity_extract_max_gleaning,
        "entity_extract_max_records": 100,
        "entity_extract_max_entities": 40,
        "addon_params": addon_params if addon_params is not None else {},
        "tokenizer": tokenizer if tokenizer is not None else make_dummy_tokenizer(),
        "llm_model_max_async": llm_model_max_async,
        "entity_extraction_use_json": use_json,
        "_entity_extraction_prompt_profile": prompt_profile,
    }
    if enable_llm_cache_for_entity_extract is not None:
        config["enable_llm_cache_for_entity_extract"] = (
            enable_llm_cache_for_entity_extract
        )
    config.update(extra)
    return config


def make_chunks(
    count: int = 1,
    *,
    content: str = "Test content.",
    file_path: str | None = None,
    vary_content: bool = False,
    start_index: int = 0,
) -> dict[str, dict]:
    """Build a dict of synthetic chunk payloads keyed ``chunk-001``, ..."""
    chunks: dict[str, dict] = {}
    for index in range(1, count + 1):
        chunk_content = f"{content} {index}" if vary_content else content
        row: dict = {
            "tokens": len(chunk_content),
            "content": chunk_content,
            "full_doc_id": "doc-001",
            "chunk_order_index": start_index + (index - 1),
        }
        if file_path is not None:
            row["file_path"] = file_path
        chunks[f"chunk-{index:03d}"] = row
    return chunks


def make_merge_config(
    *,
    tokenizer: Tokenizer | None = None,
    embedding_token_limit: int | None = None,
    max_source_ids_per_entity: int = 10_000,
    max_source_ids_per_relation: int = 10_000,
    max_file_paths: int = 100,
    file_path_more_placeholder: str = "...",
    source_ids_limit_method: str = "KEEP",
    summary_context_size: int = 1_000_000,
    summary_max_tokens: int = 1_000_000,
    force_llm_summary_on_merge: int = 6,
    **extra,
) -> dict:
    """Build a slack-threshold ``global_config`` for merge/rebuild unit tests."""
    config = {
        "tokenizer": tokenizer if tokenizer is not None else make_dummy_tokenizer(),
        "embedding_token_limit": embedding_token_limit,
        "summary_context_size": summary_context_size,
        "summary_max_tokens": summary_max_tokens,
        "force_llm_summary_on_merge": force_llm_summary_on_merge,
        "source_ids_limit_method": source_ids_limit_method,
        "max_source_ids_per_entity": max_source_ids_per_entity,
        "max_source_ids_per_relation": max_source_ids_per_relation,
        "max_file_paths": max_file_paths,
        "file_path_more_placeholder": file_path_more_placeholder,
    }
    config.update(extra)
    return config


class MemGraph:
    """In-memory graph storage double supporting single and batch operations."""

    def __init__(
        self,
        nodes: dict[str, dict] | None = None,
        edges: dict[tuple[str, str], dict] | None = None,
    ) -> None:
        self.nodes: dict[str, dict] = deepcopy(nodes) if nodes else {}
        self.edges: dict[tuple[str, str], dict] = deepcopy(edges) if edges else {}
        self.upsert_node_calls = 0
        self.upsert_edge_calls = 0
        self.workspace = ""

    async def get_node(self, name: str):
        data = self.nodes.get(name)
        return deepcopy(data) if data is not None else None

    async def has_node(self, name: str) -> bool:
        return name in self.nodes

    async def upsert_node(self, name: str, node_data: dict) -> None:
        self.upsert_node_calls += 1
        self.nodes[name] = deepcopy(node_data)

    async def has_edge(self, src: str, tgt: str) -> bool:
        return (src, tgt) in self.edges or (tgt, src) in self.edges

    async def get_edge(self, src: str, tgt: str):
        data = self.edges.get((src, tgt)) or self.edges.get((tgt, src))
        return deepcopy(data) if data is not None else None

    async def upsert_edge(self, src: str, tgt: str, edge_data: dict) -> None:
        self.upsert_edge_calls += 1
        self.edges[(src, tgt)] = deepcopy(edge_data)

    async def get_node_edges(self, name: str) -> list[tuple[str, str]]:
        return [(src, tgt) for (src, tgt) in self.edges if src == name or tgt == name]

    async def delete_node(self, name: str) -> None:
        self.nodes.pop(name, None)
        self.edges = {
            (src, tgt): data
            for (src, tgt), data in self.edges.items()
            if src != name and tgt != name
        }

    async def remove_nodes(self, nodes: list[str]) -> None:
        for name in nodes:
            await self.delete_node(name)

    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        for src, tgt in edges:
            self.edges.pop((src, tgt), None)
            self.edges.pop((tgt, src), None)

    async def get_nodes_batch(self, names: list[str]) -> dict[str, dict | None]:
        return {
            name: deepcopy(self.nodes[name]) for name in names if name in self.nodes
        }

    async def has_nodes_batch(self, names: list[str]) -> set[str]:
        return {name for name in names if name in self.nodes}

    async def upsert_nodes_batch(self, nodes: list[tuple[str, dict]]) -> None:
        for name, node_data in nodes:
            await self.upsert_node(name, node_data)

    async def get_edges_batch(self, pairs) -> dict:
        result = {}
        for pair in pairs:
            src, tgt = (pair["src"], pair["tgt"]) if isinstance(pair, dict) else pair
            edge = await self.get_edge(src, tgt)
            if edge is not None:
                result[(src, tgt)] = edge
        return result

    async def upsert_edges_batch(self, edges: list[tuple[str, str, dict]]) -> None:
        for src, tgt, edge_data in edges:
            await self.upsert_edge(src, tgt, edge_data)

    async def index_done_callback(self) -> None:
        return None


class MemKV:
    """In-memory KV / cache double that also records upsert/delete call order."""

    def __init__(
        self,
        data: dict | None = None,
        *,
        global_config: dict | None = None,
    ) -> None:
        self.data: dict = deepcopy(data) if data else {}
        self.global_config: dict = dict(global_config) if global_config else {}
        self.ops: list[tuple[str, tuple[str, ...]]] = []
        self.namespace = "test"

    async def get_by_id(self, key: str):
        value = self.data.get(key)
        return deepcopy(value) if value is not None else None

    async def get_by_ids(self, keys: list[str]):
        return [deepcopy(self.data[k]) if k in self.data else None for k in keys]

    async def upsert(self, data: dict) -> None:
        self.ops.append(("upsert", tuple(data.keys())))
        self.data.update(deepcopy(data))

    async def delete(self, keys: list[str]) -> None:
        self.ops.append(("delete", tuple(keys)))
        for key in keys:
            self.data.pop(key, None)

    async def filter_keys(self, keys: set[str]) -> set[str]:
        return {k for k in keys if k not in self.data}

    async def is_empty(self) -> bool:
        return len(self.data) == 0

    async def index_done_callback(self) -> None:
        return None

    async def drop(self) -> dict[str, str]:
        self.data.clear()
        return {"status": "success"}


class MemVDB:
    """In-memory vector storage double recording upsert/delete payloads."""

    def __init__(self, global_config: dict | None = None) -> None:
        self.global_config: dict = (
            dict(global_config) if global_config is not None else {"workspace": "test"}
        )
        self.records: dict[str, dict] = {}
        self.upserts: list[dict] = []
        self.deletes: list = []
        self.upsert_calls = 0
        self.namespace = "test"

    async def upsert(self, data: dict) -> None:
        self.upsert_calls += 1
        copied = deepcopy(data)
        self.upserts.append(copied)
        self.records.update(copied)

    async def delete(self, ids: list[str]) -> None:
        self.deletes.append(list(ids))
        for record_id in ids:
            self.records.pop(record_id, None)

    async def get_by_id(self, record_id: str):
        value = self.records.get(record_id)
        return deepcopy(value) if value is not None else None

    async def get_by_ids(self, ids: list[str]):
        return [deepcopy(self.records[i]) if i in self.records else None for i in ids]

    async def query(self, query: str, top_k: int, query_embedding=None):
        return []

    async def delete_entity(self, entity_name: str) -> None:
        return None

    async def delete_entity_relation(self, entity_name: str) -> None:
        return None

    async def get_vectors_by_ids(self, ids: list[str]):
        return {}

    async def index_done_callback(self) -> None:
        return None

    async def drop(self) -> dict[str, str]:
        self.records.clear()
        return {"status": "success"}
