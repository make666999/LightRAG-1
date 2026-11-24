"""LLM extraction cache reachability and skip-reporting contract tests (#3833).

Pins the PurgeRecoveryContract / LLM extraction cache reachability invariants:
1. Reference-before-row ordering: every ``extract`` row written to
   ``llm_response_cache`` is already referenced by its owning chunk's
   ``llm_cache_list``, even when a sibling chunk cancels the task mid-flight.
2. Operator-visible skip reporting: when ``text_chunks_storage`` cannot record
   the reference and the cache write is skipped, every occurrence reaches the
   server log, while ``pipeline_status`` gets the first occurrence plus exactly
   one end-of-stage aggregate (even when a chunk raises midway).
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock

import pytest

import lightrag.operate as operate
from lightrag.operate import extract_entities

from ._helpers import (
    MINIMAL_EXTRACTION_RESULT,
    MemKV,
    make_chunks,
    make_extract_global_config,
)

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


def _cache_gates() -> dict:
    return {
        "enable_llm_cache": True,
        "enable_llm_cache_for_entity_extract": True,
    }


class _UnwritableChunks(MemKV):
    """``text_chunks`` double whose writes fail so cache references cannot land."""

    async def upsert(self, entries):
        raise RuntimeError("text_chunks write is down")


@pytest.fixture
def status_messages(monkeypatch) -> list[str]:
    """Capture what extraction publishes to ``pipeline_status``."""
    captured: list[str] = []

    class _CaptureLogger:
        def __init__(self, pipeline_status):
            self.pipeline_status = pipeline_status

        def log(self, *messages):
            captured.extend(messages)

    monkeypatch.setattr(operate, "PipelineStatusLogger", _CaptureLogger)
    return captured


def _skip_lines(messages: list[str]) -> list[str]:
    return [m for m in messages if "cache write" in m.lower()]


def _skip_config(extract_func=None, *, gleaning: int = 0) -> dict:
    return make_extract_global_config(
        extract_func=extract_func or AsyncMock(return_value=MINIMAL_EXTRACTION_RESULT),
        entity_extract_max_gleaning=gleaning,
        enable_llm_cache_for_entity_extract=True,
    )


# --------------------------------------------------------------------------- #
# 1. Operator-visible cache-skip reporting
# --------------------------------------------------------------------------- #


async def test_the_first_skip_is_published_with_chunk_and_file_identity(
    status_messages,
):
    chunks = make_chunks(1, file_path="reports/annual.md", vary_content=True)
    await extract_entities(
        chunks=chunks,
        global_config=_skip_config(),
        llm_response_cache=MemKV(global_config=_cache_gates()),
        text_chunks_storage=_UnwritableChunks(chunks),
    )

    lines = _skip_lines(status_messages)
    assert len(lines) == 2, status_messages
    assert "chunk-001" in lines[0]
    assert "reports/annual.md" in lines[0]
    assert "1 of 1 chunks" in lines[1]
    assert "recomputed" in lines[1]


async def test_status_lines_do_not_grow_with_the_number_of_skipped_chunks(
    status_messages,
):
    """An unwritable text_chunks skips on every chunk; history stays bounded at 2 lines."""
    chunks = make_chunks(6, file_path="reports/annual.md", vary_content=True)
    await extract_entities(
        chunks=chunks,
        global_config=_skip_config(),
        llm_response_cache=MemKV(global_config=_cache_gates()),
        text_chunks_storage=_UnwritableChunks(chunks),
    )

    lines = _skip_lines(status_messages)
    assert len(lines) == 2, lines
    assert "6 of 6 chunks" in lines[1]


async def test_every_occurrence_still_reaches_the_server_log(
    status_messages, lightrag_caplog, caplog
):
    chunks = make_chunks(3, file_path="reports/annual.md", vary_content=True)
    with caplog.at_level(logging.WARNING, logger="lightrag"):
        await extract_entities(
            chunks=chunks,
            global_config=_skip_config(),
            llm_response_cache=MemKV(global_config=_cache_gates()),
            text_chunks_storage=_UnwritableChunks(chunks),
        )

    per_chunk = [
        record.getMessage()
        for record in caplog.records
        if "cache write skipped for chunk" in record.getMessage()
    ]
    assert len(per_chunk) == 3, per_chunk
    for index in (1, 2, 3):
        assert any(f"chunk-{index:03d}" in message for message in per_chunk)


async def test_a_run_whose_cache_writes_all_land_reports_nothing(status_messages):
    """No false alarm: a writable chunk store records every reference."""
    chunks = make_chunks(2, file_path="reports/annual.md", vary_content=True)
    text_chunks = MemKV(chunks)
    await extract_entities(
        chunks=chunks,
        global_config=_skip_config(),
        llm_response_cache=MemKV(global_config=_cache_gates()),
        text_chunks_storage=text_chunks,
    )

    assert _skip_lines(status_messages) == []
    attached = [
        key
        for row in text_chunks.data.values()
        for key in (row.get("llm_cache_list") or [])
    ]
    assert len(attached) == 2, attached


async def test_the_summary_is_published_when_a_chunk_raises_midway(status_messages):
    """The skip aggregate is still published via ``finally`` when a chunk fails."""
    chunks = make_chunks(2, file_path="reports/annual.md", vary_content=True)
    calls: list[str] = []

    async def flaky_llm(prompt: str, *args, **kwargs) -> str:
        calls.append(prompt)
        if len(calls) > 1:
            raise RuntimeError("second chunk exploded")
        return MINIMAL_EXTRACTION_RESULT

    with pytest.raises(Exception, match="second chunk exploded"):
        await extract_entities(
            chunks=chunks,
            global_config=_skip_config(flaky_llm),
            llm_response_cache=MemKV(global_config=_cache_gates()),
            text_chunks_storage=_UnwritableChunks(chunks),
        )

    lines = _skip_lines(status_messages)
    assert len(lines) == 2, lines
    assert "1 of 2 chunks" in lines[1]


# --------------------------------------------------------------------------- #
# 2. Extract cache orphan prevention (reference-before-row invariant)
# --------------------------------------------------------------------------- #

_ALPHA = "Alpha content about Alice."
_BRAVO = "Bravo content about Bob."
_ORPHAN_CHUNK_CONTENTS = {"chunk-alpha": _ALPHA, "chunk-bravo": _BRAVO}


def _named_extraction_result(name: str) -> str:
    return f"(entity<|#|>{name}<|#|>CONCEPT<|#|>Description of {name})<|COMPLETE|>"


def _make_orphan_chunks() -> dict[str, dict]:
    return {
        key: {
            "tokens": len(content),
            "content": content,
            "full_doc_id": "doc-001",
            "chunk_order_index": index,
            "file_path": f"{key}.md",
        }
        for index, (key, content) in enumerate(_ORPHAN_CHUNK_CONTENTS.items())
    }


def _referenced_keys(text_chunks: MemKV) -> set[str]:
    return {
        key
        for row in text_chunks.data.values()
        for key in (row.get("llm_cache_list") or [])
    }


def _extract_rows(llm_cache: MemKV) -> set[str]:
    return {
        key
        for key, row in llm_cache.data.items()
        if isinstance(row, dict) and row.get("cache_type") == "extract"
    }


async def test_no_extract_cache_row_survives_a_cancelled_sibling():
    """A sibling cancellation after an extract cache row lands must not leave
    that row unreferenced in any chunk's ``llm_cache_list``."""
    chunks = _make_orphan_chunks()
    text_chunks = MemKV(chunks)

    row_written = asyncio.Event()
    never = asyncio.Event()

    class _ParkingCache(MemKV):
        async def upsert(self, rows: dict):
            await super().upsert(rows)
            if any(
                isinstance(row, dict) and row.get("cache_type") == "extract"
                for row in rows.values()
            ):
                row_written.set()
                await never.wait()

    llm_cache = _ParkingCache(global_config=_cache_gates())

    async def fake_llm(prompt: str, *args, **kwargs) -> str:
        if _BRAVO in prompt:
            await row_written.wait()
            raise RuntimeError("bravo exploded")
        if _ALPHA in prompt:
            return _named_extraction_result("ALPHA")
        raise AssertionError(f"unexpected prompt: {prompt[:120]!r}")

    with pytest.raises(Exception, match="bravo exploded"):
        await asyncio.wait_for(
            extract_entities(
                chunks=chunks,
                global_config=make_extract_global_config(
                    extract_func=fake_llm,
                    llm_model_max_async=3,
                    enable_llm_cache_for_entity_extract=True,
                    kg_extraction_validator=None,
                ),
                llm_response_cache=llm_cache,
                text_chunks_storage=text_chunks,
            ),
            timeout=10,
        )

    rows = _extract_rows(llm_cache)
    assert rows, "fixture must actually write an extract cache row"
    orphaned = rows - _referenced_keys(text_chunks)
    assert not orphaned, (
        "extract cache rows are unreachable — no chunk's llm_cache_list names "
        f"them, so delete_llm_cache can never remove them: {sorted(orphaned)}"
    )


async def test_every_extract_row_is_referenced_on_the_happy_path():
    """Each call's key is attached exactly once, gleaning included."""
    chunks = _make_orphan_chunks()
    text_chunks = MemKV(chunks)
    llm_cache = MemKV(global_config=_cache_gates())

    calls: list[str] = []

    async def fake_llm(prompt: str, *args, **kwargs) -> str:
        for key, content in _ORPHAN_CHUNK_CONTENTS.items():
            if content in prompt:
                calls.append(key)
                return _named_extraction_result(key.split("-", 1)[1].upper())
        if "last extraction task" in prompt:
            calls.append("gleaning")
            return _named_extraction_result("GLEANED")
        raise AssertionError(f"unexpected prompt: {prompt[:120]!r}")

    await extract_entities(
        chunks=chunks,
        global_config=make_extract_global_config(
            extract_func=fake_llm,
            entity_extract_max_gleaning=1,
            llm_model_max_async=3,
            enable_llm_cache_for_entity_extract=True,
            kg_extraction_validator=None,
        ),
        llm_response_cache=llm_cache,
        text_chunks_storage=text_chunks,
    )

    assert len(calls) == 4, calls
    rows = _extract_rows(llm_cache)
    assert rows == _referenced_keys(text_chunks)
    for chunk_id, row in text_chunks.data.items():
        attached = row.get("llm_cache_list") or []
        assert len(attached) == 2, (chunk_id, attached)
        assert len(set(attached)) == len(attached), (chunk_id, attached)
