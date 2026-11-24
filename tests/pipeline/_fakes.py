"""Shared test doubles and LightRAG factory helpers for ``tests/pipeline/``.

Leading underscore ensures pytest does not collect this module as tests.
"""

from __future__ import annotations

import asyncio
import struct
import zlib
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

import lightrag.operate as operate_module
from lightrag import ROLES, LightRAG, RoleLLMConfig
from lightrag.constants import (
    GRAPH_FIELD_SEP,
    KG_PURGE_METADATA_KEY,
    KG_WRITE_STATE_METADATA_KEY,
)
from lightrag.exceptions import StorageRecordNotFoundError
from lightrag.kg.shared_storage import (
    get_namespace_data,
    get_namespace_lock,
)
from lightrag.utils import (
    EmbeddingFunc,
    Tokenizer,
    TokenizerInterface,
    compute_mdhash_id,
)
from lightrag.utils_pipeline import make_kg_purge_operation_id

_UNSET: Any = object()

ROLE_FIELD_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("_llm_model_func", "func"),
    ("_llm_model_kwargs", "kwargs"),
    ("_llm_model_max_async", "max_async"),
    ("_llm_timeout", "timeout"),
)


# ---------------------------------------------------------------------------
# Basic tokenizer / embedding / LLM / chunker doubles
# ---------------------------------------------------------------------------


class SimpleTokenizerImpl(TokenizerInterface):
    """Character-level deterministic tokenizer for offline pipeline tests."""

    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


def make_tokenizer(model_name: str = "mock-tokenizer") -> Tokenizer:
    """Build a deterministic character-level ``Tokenizer``."""
    return Tokenizer(model_name, SimpleTokenizerImpl())


async def dummy_embedding(texts: list[str]) -> np.ndarray:
    """Deterministic 8-dim embedding stub returning all ones."""
    return np.ones((len(texts), 8), dtype=float)


async def mock_embedding_32(texts: list[str]) -> np.ndarray:
    """Deterministic 32-dim float32 embedding stub."""
    return np.full((len(texts), 32), 0.1, dtype=np.float32)


def make_embedding_func(
    *,
    dim: int = 8,
    max_token_size: int = 8192,
    func: Any = None,
) -> EmbeddingFunc:
    """Build an ``EmbeddingFunc`` backed by an offline deterministic stub."""
    if func is None:
        func = mock_embedding_32 if dim == 32 else dummy_embedding
    return EmbeddingFunc(
        embedding_dim=dim,
        max_token_size=max_token_size,
        func=func,
    )


async def dummy_llm(*args: Any, **kwargs: Any) -> str:
    """Minimal async LLM stub returning ``"ok"``."""
    return "ok"


async def mock_json_llm(prompt: Any, **kwargs: Any) -> str:
    """Async LLM stub returning a valid minimal multimodal JSON object."""
    return '{"name":"x","summary":"s","detail_description":"d"}'


async def noop_llm(*args: Any, **kwargs: Any) -> str:  # pragma: no cover
    """Async LLM stub returning empty string (for paths not invoking LLM)."""
    return ""


def single_chunk_chunking(
    tokenizer: Any,
    content: str,
    *args: Any,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Deterministic chunker emitting one ``f"{content}::chunk1"`` chunk."""
    return [{"tokens": 1, "content": f"{content}::chunk1", "chunk_order_index": 0}]


def raw_single_chunk_chunking(
    tokenizer: Any,
    content: str,
    *args: Any,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Deterministic chunker emitting one chunk carrying ``content`` verbatim."""
    return [{"tokens": 1, "content": content, "chunk_order_index": 0}]


def two_chunk_chunking(
    tokenizer: Any,
    content: str,
    *args: Any,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Deterministic chunker emitting two chunks per document."""
    return [
        {"tokens": 1, "content": f"{content}::chunk1", "chunk_order_index": 0},
        {"tokens": 1, "content": f"{content}::chunk2", "chunk_order_index": 1},
    ]


# ---------------------------------------------------------------------------
# LightRAG construction & pipeline status helpers
# ---------------------------------------------------------------------------


def extract_role_llm_configs(kwargs: dict[str, Any]) -> dict[str, RoleLLMConfig]:
    """Pop per-role ``<role>_llm_*`` shorthand kwargs into ``role_llm_configs``."""
    role_configs: dict[str, RoleLLMConfig] = {}
    for spec in ROLES:
        bucket: dict[str, Any] = {}
        for suffix, target in ROLE_FIELD_SUFFIXES:
            key = f"{spec.name}{suffix}"
            if key in kwargs:
                bucket[target] = kwargs.pop(key)
        if bucket:
            role_configs[spec.name] = RoleLLMConfig(**bucket)
    return role_configs


def make_lightrag(
    tmp_path: Path,
    *,
    workspace: str | None = None,
    workspace_prefix: str = "pipe",
    subdir: str | None = "wd",
    llm_model_func: Any = dummy_llm,
    embedding_func: Any = _UNSET,
    embedding_dim: int = 8,
    max_token_size: int = 8192,
    tokenizer: Tokenizer | None = None,
    tokenizer_name: str = "mock-tokenizer",
    **kwargs: Any,
) -> LightRAG:
    """Build an uninitialized ``LightRAG`` with deterministic offline doubles."""
    role_configs = extract_role_llm_configs(kwargs)
    if role_configs and "role_llm_configs" not in kwargs:
        kwargs["role_llm_configs"] = role_configs

    working_dir = str(tmp_path / subdir) if subdir else str(tmp_path)
    resolved_workspace = (
        workspace if workspace is not None else f"{workspace_prefix}-{uuid4().hex[:8]}"
    )
    resolved_embedding = (
        make_embedding_func(dim=embedding_dim, max_token_size=max_token_size)
        if embedding_func is _UNSET
        else embedding_func
    )
    resolved_tokenizer = (
        tokenizer if tokenizer is not None else make_tokenizer(tokenizer_name)
    )
    return LightRAG(
        working_dir=working_dir,
        workspace=resolved_workspace,
        llm_model_func=llm_model_func,
        embedding_func=resolved_embedding,
        tokenizer=resolved_tokenizer,
        **kwargs,
    )


async def build_initialized_rag(tmp_path: Path, **kwargs: Any) -> LightRAG:
    """Build a ``LightRAG`` via :func:`make_lightrag` and initialize storages."""
    rag = make_lightrag(tmp_path, **kwargs)
    await rag.initialize_storages()
    return rag


def status_value_text(status: Any) -> str:
    """Return the string value of a ``DocStatus`` enum or raw status string."""
    return status.value if hasattr(status, "value") else str(status)


async def pipeline_status_handles(rag: LightRAG) -> tuple[dict[str, Any], Any]:
    """Return ``(pipeline_status, pipeline_status_lock)`` for ``rag``'s workspace."""
    status = await get_namespace_data("pipeline_status", workspace=rag.workspace)
    lock = get_namespace_lock("pipeline_status", workspace=rag.workspace)
    return status, lock


async def shutdown_role_workers(rag: LightRAG) -> None:
    """Explicitly shut down each role wrapper's priority-queue workers."""
    seen: set[int] = set()
    for func in rag.role_llm_funcs.values():
        if func is not None and id(func) not in seen:
            seen.add(id(func))
            await rag._shutdown_llm_wrapper(func)


def make_png_bytes(width: int = 64, height: int = 64) -> bytes:
    """Generate a minimal valid RGBA PNG of the requested pixel dimensions."""
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">II", width, height) + b"\x08\x06\x00\x00\x00"
    crc_ihdr = zlib.crc32(b"IHDR" + ihdr).to_bytes(4, "big")
    ihdr_chunk = struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr + crc_ihdr
    idat_payload = b"\x00" * (width * height * 4 + height)
    compressed = zlib.compress(idat_payload)
    crc_idat = zlib.crc32(b"IDAT" + compressed).to_bytes(4, "big")
    idat_chunk = struct.pack(">I", len(compressed)) + b"IDAT" + compressed + crc_idat
    iend_chunk = b"\x00\x00\x00\x00IEND\xaeB`\x82"
    return sig + ihdr_chunk + idat_chunk + iend_chunk


PNG_BYTES: bytes = make_png_bytes(64, 64)


# ---------------------------------------------------------------------------
# End-to-end deletion / ingestion helpers (used by fail-closed & backend suites)
# ---------------------------------------------------------------------------


def wire_fake_alice_acme_extraction(rag: LightRAG) -> None:
    """Wire deterministic extraction producing ALICE, ACME, and ACME--ALICE."""

    async def fake_extract(chunks: dict[str, Any], *args: Any, **kwargs: Any):
        results = []
        for chunk_id in chunks:
            nodes = {
                name: [
                    {
                        "entity_name": name,
                        "entity_type": "person",
                        "description": f"{name} description",
                        "source_id": chunk_id,
                        "file_path": "d.txt",
                        "timestamp": 1,
                    }
                ]
                for name in ("ALICE", "ACME")
            }
            edges = {
                ("ACME", "ALICE"): [
                    {
                        "src_id": "ACME",
                        "tgt_id": "ALICE",
                        "description": "works at",
                        "keywords": "employment",
                        "weight": 1.0,
                        "source_id": chunk_id,
                        "file_path": "d.txt",
                        "timestamp": 1,
                    }
                ]
            }
            results.append((nodes, edges))
        return results

    rag._process_extract_entities = fake_extract  # type: ignore[assignment]


async def build_delete_test_rag(tmp_path: Path, workspace: str) -> LightRAG:
    """Build and initialize a ``LightRAG`` wired with ALICE--ACME extraction."""
    rag = await build_initialized_rag(
        tmp_path,
        workspace=workspace,
        chunking_func=single_chunk_chunking,
        max_parallel_insert=1,
    )
    wire_fake_alice_acme_extraction(rag)
    return rag


async def ingest_test_doc(
    rag: LightRAG,
    file_path: str = "d.txt",
    content: str = "alice works at acme",
) -> str:
    """Enqueue and process a single document, asserting it lands PROCESSED."""
    doc_id = compute_mdhash_id(file_path, prefix="doc-")
    await rag.apipeline_enqueue_documents(content, ids=[doc_id], file_paths=[file_path])
    await rag.apipeline_process_enqueue_documents()
    row = await rag.doc_status.get_by_id(doc_id)
    assert row is not None and status_value_text(row["status"]) == "processed", row
    return doc_id


async def chunk_ids_for_doc(rag: LightRAG, doc_id: str) -> list[str]:
    """Return the stored ``chunks_list`` for ``doc_id``."""
    row = await rag.doc_status.get_by_id(doc_id)
    assert row is not None
    return list(row.get("chunks_list") or [])


# ---------------------------------------------------------------------------
# Graph / chunk-tracking ordering doubles
# ---------------------------------------------------------------------------


class InjectedBackendError(RuntimeError):
    """Sentinel exception raised by injected storage failures."""


class ImmediateTrackingKV:
    """Immediate-write KV double: mutations take effect before the next await.

    Every mutating ``delete`` yields once (``await asyncio.sleep(0)``) so
    pending task cancellations can be delivered inside the store call.
    """

    def __init__(
        self,
        tag: str = "",
        timeline: list[Any] | None = None,
    ) -> None:
        self.tag = tag
        self.records: dict[str, Any] = {}
        self.timeline: list[Any] | None = timeline
        self.flushes = 0
        self.fail_delete_times = 0
        self.delete_error: BaseException | None = None

    async def get_by_id(self, key: str) -> Any:
        return deepcopy(self.records.get(key))

    async def get_by_ids(self, keys: list[str]) -> list[Any]:
        return [deepcopy(self.records.get(key)) for key in keys]

    async def upsert(self, data: dict[str, Any]) -> None:
        if self.timeline is not None:
            self.timeline.append(("upsert", self.tag, sorted(data)))
        self.records.update(deepcopy(data))

    async def delete(self, ids: list[str]) -> None:
        await asyncio.sleep(0)
        if self.delete_error is not None:
            raise self.delete_error
        if self.fail_delete_times > 0:
            self.fail_delete_times -= 1
            raise InjectedBackendError("tracking delete failed")
        if self.timeline is not None:
            self.timeline.append(("delete", self.tag, sorted(ids)))
        for key in ids:
            self.records.pop(key, None)

    async def index_done_callback(self) -> None:
        await asyncio.sleep(0)
        self.flushes += 1

    async def is_empty(self) -> bool:
        return not self.records


class DeferredTrackingKV(ImmediateTrackingKV):
    """Deferred-commit KV double: ``index_done_callback`` persists ``records`` to ``disk``."""

    def __init__(
        self,
        name: str,
        commit_log: list[str] | None = None,
        *,
        timeline: list[Any] | None = None,
    ) -> None:
        super().__init__(tag=name, timeline=timeline)
        self.name = name
        self.commit_log = commit_log
        self.disk: dict[str, Any] = {}
        self.fail_commit_times = 0

    @property
    def persisted(self) -> dict[str, Any]:
        return self.disk

    @persisted.setter
    def persisted(self, value: dict[str, Any]) -> None:
        self.disk = value

    async def index_done_callback(self) -> None:
        await asyncio.sleep(0)
        self.flushes += 1
        if self.commit_log is not None:
            self.commit_log.append(self.name)
        if self.fail_commit_times > 0:
            self.fail_commit_times -= 1
            raise InjectedBackendError(f"{self.name} commit failed")
        self.disk = deepcopy(self.records)


class TrackingVectorDouble:
    """In-memory vector-store double for graph/tracking ordering tests."""

    def __init__(self, global_config: dict[str, Any]) -> None:
        self.global_config = global_config
        self.fail = False
        self.fail_flush = False
        self.flushes = 0
        self.delete_error: BaseException | None = None

    def _check(self) -> None:
        if self.fail:
            raise InjectedBackendError("vector backend unavailable")

    async def delete(self, ids: list[str]) -> None:
        await asyncio.sleep(0)
        if self.delete_error is not None:
            raise self.delete_error
        self._check()

    async def upsert(self, data: dict[str, Any]) -> None:
        self._check()

    async def delete_entity(self, entity_name: str) -> None:
        self._check()

    async def delete_entity_relation(self, entity_name: str) -> None:
        self._check()

    async def index_done_callback(self) -> None:
        await asyncio.sleep(0)
        self.flushes += 1
        if self.fail_flush:
            raise InjectedBackendError("vector flush failed")


# ---------------------------------------------------------------------------
# In-memory purge-contract doubles (shared by purge primitive & fail-closed)
# ---------------------------------------------------------------------------


def patch_storage_keyed_lock_noop(monkeypatch: Any) -> None:
    """Replace ``operate.get_storage_keyed_lock`` with an async no-op context manager."""

    @asynccontextmanager
    async def _noop_lock(*args: Any, **kwargs: Any):
        yield

    monkeypatch.setattr(operate_module, "get_storage_keyed_lock", _noop_lock)


class PurgeKVDouble:
    """In-memory KV double tracking reads, upserts, and deletes for purge tests."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.data: dict[str, Any] = dict(data or {})
        self.deleted: list[str] = []
        self.upserted: list[dict[str, Any]] = []

    async def get_by_id(self, key: str) -> Any:
        return self.data.get(key)

    async def get_by_ids(self, keys: list[str]) -> list[Any]:
        return [self.data.get(k) for k in keys]

    async def upsert(self, data: dict[str, Any]) -> None:
        self.upserted.append(dict(data))
        self.data.update(data)

    async def delete(self, ids: list[str]) -> None:
        self.deleted.extend(ids)
        for k in ids:
            self.data.pop(k, None)

    async def index_done_callback(self) -> None:
        pass


class PurgeDocStatusDouble(PurgeKVDouble):
    """Minimal ``doc_status`` double supporting atomic metadata field updates."""

    async def update_doc_status_fields(
        self,
        doc_id: str,
        fields: dict[str, Any],
        *,
        missing_ok: bool = False,
    ) -> None:
        existing = self.data.get(doc_id)
        if existing is None:
            if missing_ok:
                return
            raise StorageRecordNotFoundError(doc_id)
        self.data[doc_id] = {**existing, **fields}

    def journal(self, doc_id: str = "d1") -> dict[str, Any] | None:
        return (
            (self.data.get(doc_id) or {}).get("metadata", {}).get(KG_PURGE_METADATA_KEY)
        )


class PurgeVdbDouble:
    """In-memory VDB double recording deleted IDs and upserted payloads."""

    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.data: dict[str, Any] = {}

    async def delete(self, ids: list[str]) -> None:
        self.deleted.extend(ids)
        for key in ids:
            self.data.pop(key, None)

    async def upsert(self, data: dict[str, Any]) -> None:
        self.data.update(data)

    async def index_done_callback(self) -> None:
        pass


class PurgeGraphDouble:
    """In-memory graph double recording reads, mutations, and removals."""

    def __init__(
        self,
        nodes: dict[str, dict[str, Any]] | None = None,
        edges: dict[tuple[str, str], dict[str, Any]] | None = None,
    ) -> None:
        self.nodes: dict[str, dict[str, Any]] = dict(nodes or {})
        self.edges: dict[tuple[str, str], dict[str, Any]] = dict(edges or {})
        self.removed_nodes: list[str] = []
        self.removed_edges: list[tuple[str, str]] = []
        self.reads: list[str] = []

    async def get_nodes_batch(self, names: list[str]) -> dict[str, Any]:
        self.reads.append("get_nodes_batch")
        return {n: dict(self.nodes[n]) if n in self.nodes else None for n in names}

    async def get_edges_batch(self, pairs: list[dict[str, str]]) -> dict[Any, Any]:
        self.reads.append("get_edges_batch")
        out: dict[tuple[str, str], Any] = {}
        for p in pairs:
            s, t = p["src"], p["tgt"]
            edge = self.edges.get((s, t)) or self.edges.get((t, s))
            out[(s, t)] = dict(edge) if edge else None
        return out

    async def get_nodes_edges_batch(self, names: list[str]) -> dict[str, list[Any]]:
        self.reads.append("get_nodes_edges_batch")
        return {n: [] for n in names}

    async def get_node(self, name: str) -> dict[str, Any] | None:
        return self.nodes.get(name)

    async def get_edge(self, src: str, tgt: str) -> dict[str, Any] | None:
        return self.edges.get((src, tgt)) or self.edges.get((tgt, src))

    async def get_node_edges(self, name: str) -> list[tuple[str, str]]:
        return [pair for pair in self.edges if name in pair]

    async def has_node(self, name: str) -> bool:
        return name in self.nodes

    async def upsert_node(self, name: str, node_data: dict[str, Any]) -> None:
        self.nodes[name] = dict(node_data)

    async def upsert_edge(
        self, src: str, tgt: str, edge_data: dict[str, Any]
    ) -> None:
        self.edges[(src, tgt)] = dict(edge_data)

    async def remove_nodes(self, names: list[str]) -> None:
        self.removed_nodes.extend(names)
        for n in names:
            self.nodes.pop(n, None)

    async def remove_edges(self, pairs: list[tuple[str, str]]) -> None:
        self.removed_edges.extend(pairs)
        for s, t in pairs:
            self.edges.pop((s, t), None)
            self.edges.pop((t, s), None)


def make_purge_node(name: str, sources: list[str]) -> dict[str, Any]:
    """Construct a minimal graph entity node dict for purge unit tests."""
    return {
        "entity_id": name,
        "description": name,
        "source_id": GRAPH_FIELD_SEP.join(sources),
        "entity_type": "X",
        "file_path": "f",
    }


def make_purge_edge(sources: list[str]) -> dict[str, Any]:
    """Construct a minimal graph relation edge dict for purge unit tests."""
    return {
        "description": "rel",
        "keywords": "k",
        "weight": 1.0,
        "source_id": GRAPH_FIELD_SEP.join(sources),
        "file_path": "f",
    }


def make_purge_status_row(
    *,
    write_state: str | None = None,
    purge_journal: dict[str, Any] | None = None,
    patch_journal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a ``doc_status`` row seeded with optional write-state / journal metadata."""
    metadata: dict[str, Any] = {}
    if write_state is not None:
        metadata[KG_WRITE_STATE_METADATA_KEY] = write_state
    if purge_journal is not None:
        metadata[KG_PURGE_METADATA_KEY] = purge_journal
    if patch_journal is not None:
        metadata["custom_chunk_patch"] = patch_journal
    return {"status": "processed", "chunks_list": [], "metadata": metadata}


def make_purge_journal(
    phase: str, chunk_ids: list[str], *, doc_id: str = "d1"
) -> dict[str, Any]:
    """Build a schema-v1 ``kg_purge`` journal payload."""
    return {
        "schema_version": 1,
        "operation_id": make_kg_purge_operation_id(doc_id, chunk_ids),
        "phase": phase,
        "chunk_count": len(chunk_ids),
        "updated_at": 0,
    }


def make_purge_pipeline_status() -> tuple[dict[str, Any], asyncio.Lock]:
    """Return a fresh ``(pipeline_status, lock)`` pair for purge unit tests."""
    return {"latest_message": "", "history_messages": []}, asyncio.Lock()


def make_purge_rag(
    *,
    graph: PurgeGraphDouble | None = None,
    full_entities: PurgeKVDouble | None = None,
    full_relations: PurgeKVDouble | None = None,
    entity_chunks: PurgeKVDouble | None = None,
    relation_chunks: PurgeKVDouble | None = None,
    text_chunks: PurgeKVDouble | None = None,
    llm_cache: PurgeKVDouble | None = None,
    doc_status: PurgeDocStatusDouble | None = None,
    default_doc_id: str = "d1",
) -> LightRAG:
    """Build a lightweight ``LightRAG.__new__`` harness backed by purge doubles."""
    rag = LightRAG.__new__(LightRAG)
    rag.chunk_entity_relation_graph = graph if graph is not None else PurgeGraphDouble()
    rag.doc_status = (
        doc_status
        if doc_status is not None
        else PurgeDocStatusDouble({default_doc_id: make_purge_status_row()})
    )
    rag.full_entities = full_entities if full_entities is not None else PurgeKVDouble()
    rag.full_relations = (
        full_relations if full_relations is not None else PurgeKVDouble()
    )
    rag.entity_chunks = entity_chunks if entity_chunks is not None else PurgeKVDouble()
    rag.relation_chunks = (
        relation_chunks if relation_chunks is not None else PurgeKVDouble()
    )
    rag.chunks_vdb = PurgeVdbDouble()
    rag.entities_vdb = PurgeVdbDouble()
    rag.relationships_vdb = PurgeVdbDouble()
    rag.text_chunks = text_chunks if text_chunks is not None else PurgeKVDouble()
    rag.llm_response_cache = llm_cache if llm_cache is not None else PurgeKVDouble()

    async def _noop_insert_done(*args: Any, **kwargs: Any) -> None:
        return None

    rag._insert_done = _noop_insert_done  # type: ignore[assignment]
    rag._build_global_config = lambda: {  # type: ignore[assignment]
        "llm_model_max_async": 1,
        "max_source_ids_per_entity": 100,
        "max_source_ids_per_relation": 100,
        "source_ids_limit_method": "KEEP",
        "max_file_paths": 100,
        "file_path_more_placeholder": "more",
    }
    return rag
