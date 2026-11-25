"""Tests for ``extract_entities``: gleaning guard, empty input, cancellation, and ordering."""

from __future__ import annotations

import asyncio
import gc
import json
import re
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

import lightrag.operate as operate
from lightrag.operate import extract_entities
from lightrag.prompt import PROMPTS
from lightrag.utils import Tokenizer

from ._helpers import (
    MINIMAL_EXTRACTION_RESULT,
    CountingTokenizer,
    make_chunks,
    make_extract_global_config,
)

pytestmark = pytest.mark.offline

_JSON_EXTRACTION_RESULT = json.dumps(
    {
        "entities": [
            {
                "name": "TEST_ENTITY",
                "type": "CONCEPT",
                "description": "A test entity",
            }
        ],
        "relationships": [],
    }
)


def _make_counting_config(
    entity_extract_max_gleaning: int = 1,
    use_json: bool = False,
) -> tuple[dict, CountingTokenizer]:
    """Build a ``global_config`` whose tokenizer records every encoded string."""
    inner = CountingTokenizer()
    global_config = make_extract_global_config(
        entity_extract_max_gleaning=entity_extract_max_gleaning,
        use_json=use_json,
        tokenizer=Tokenizer("counting", inner),
    )
    return global_config, inner


# --------------------------------------------------------------------------- #
# Empty input & gleaning token-limit guard
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_extract_entities_empty_chunks():
    """Calling extract_entities with {} returns [] instead of raising."""
    global_config = {
        "role_llm_funcs": {"extract": None},
        "entity_extract_max_gleaning": 0,
        "entity_extract_max_records": 100,
        "entity_extract_max_entities": 100,
    }
    assert await extract_entities({}, global_config) == []


@pytest.mark.asyncio
async def test_gleaning_skipped_when_tokens_exceed_limit(
    monkeypatch, caplog, lightrag_caplog
):
    """Gleaning must be skipped (with a WARNING) when the projected gleaning
    input — system + history(user+assistant) + continue prompt — exceeds
    ``MAX_EXTRACT_INPUT_TOKENS``."""
    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "10")

    global_config = make_extract_global_config(entity_extract_max_gleaning=1)
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = MINIMAL_EXTRACTION_RESULT

    with caplog.at_level("WARNING", logger="lightrag"):
        await extract_entities(
            chunks=make_chunks(),
            global_config=global_config,
        )

    assert llm_func.await_count == 1
    warnings_emitted = [
        rec.getMessage()
        for rec in caplog.records
        if rec.levelname == "WARNING"
        and rec.getMessage().startswith("Gleaning stopped for chunk chunk-001:")
    ]
    assert warnings_emitted, (
        "expected a WARNING log explaining gleaning was skipped due to "
        f"token limit; got: {[r.getMessage() for r in caplog.records]}"
    )
    msg = warnings_emitted[0]
    assert "exceeded limit (10)" in msg
    assert "Input tokens (" in msg


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("max_tokens", "max_gleaning", "expected_awaits"),
    [
        pytest.param("999999", 1, 2, id="within_limit_runs_gleaning"),
        pytest.param("999999", 0, 1, id="max_gleaning_zero_skips_gleaning"),
        pytest.param("0", 1, 2, id="max_tokens_zero_disables_guard"),
    ],
)
async def test_gleaning_round_count_by_flag_and_token_budget(
    monkeypatch, max_tokens: str, max_gleaning: int, expected_awaits: int
):
    """Gleaning runs when enabled and within budget (or guard disabled with 0)."""
    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", max_tokens)

    global_config = make_extract_global_config(entity_extract_max_gleaning=max_gleaning)
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = MINIMAL_EXTRACTION_RESULT

    await extract_entities(
        chunks=make_chunks(),
        global_config=global_config,
    )

    assert llm_func.await_count == expected_awaits


@pytest.mark.asyncio
async def test_gleaning_guard_encodes_system_prompt_once(monkeypatch):
    """The gleaning token guard encodes the chunk-invariant system prompt once
    for the whole extraction run, not once per chunk."""
    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "999999")

    global_config, inner = _make_counting_config(entity_extract_max_gleaning=1)
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = MINIMAL_EXTRACTION_RESULT

    await extract_entities(
        chunks=make_chunks(3, content="Test content number", vary_content=True),
        global_config=global_config,
    )

    assert llm_func.await_count == 6
    longest = max(inner.encoded, key=len)
    assert sum(1 for s in inner.encoded if s == longest) == 1


@pytest.mark.asyncio
async def test_gleaning_json_mode_binds_continue_prompt(monkeypatch):
    """JSON mode + gleaning reaches the gleaning call without UnboundLocalError."""
    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "999999")

    global_config = make_extract_global_config(
        entity_extract_max_gleaning=1, use_json=True
    )
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = "{}"

    await extract_entities(
        chunks=make_chunks(),
        global_config=global_config,
    )

    assert llm_func.await_count == 2


@pytest.mark.asyncio
async def test_json_mode_encodes_both_invariants_once_per_run(monkeypatch):
    """In JSON mode both hoisted strings (system + continue prompt) are encoded
    once for the whole run regardless of chunk count."""
    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "999999")

    global_config, inner = _make_counting_config(
        entity_extract_max_gleaning=1, use_json=True
    )
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _JSON_EXTRACTION_RESULT

    await extract_entities(
        chunks=make_chunks(3, content="Test content number", vary_content=True),
        global_config=global_config,
    )

    assert llm_func.await_count == 6
    for template_key in (
        "entity_extraction_json_system_prompt",
        "entity_continue_extraction_json_user_prompt",
    ):
        prefix = PROMPTS[template_key].split("{", 1)[0]
        assert prefix, f"{template_key} must start with literal text"
        matches = [s for s in inner.encoded if s.startswith(prefix)]
        assert len(matches) == 1, (
            f"{template_key} was encoded {len(matches)} times for a 3-chunk "
            "run; it is chunk-invariant and must be encoded once"
        )


async def _projected_gleaning_tokens(
    monkeypatch, caplog, use_json: bool, continue_prompt_padding: str = ""
) -> int:
    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "10")
    continue_key = (
        "entity_continue_extraction_json_user_prompt"
        if use_json
        else "entity_continue_extraction_user_prompt"
    )
    if continue_prompt_padding:
        monkeypatch.setitem(
            PROMPTS,
            continue_key,
            PROMPTS[continue_key] + continue_prompt_padding,
        )

    global_config, _ = _make_counting_config(
        entity_extract_max_gleaning=1, use_json=use_json
    )
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = (
        _JSON_EXTRACTION_RESULT if use_json else MINIMAL_EXTRACTION_RESULT
    )

    caplog.clear()
    with caplog.at_level("WARNING", logger="lightrag"):
        await extract_entities(
            chunks=make_chunks(),
            global_config=global_config,
        )

    assert llm_func.await_count == 1
    reported = [
        re.search(r"Input tokens \((\d+)\)", rec.getMessage())
        for rec in caplog.records
        if rec.getMessage().startswith("Gleaning stopped for chunk chunk-001:")
    ]
    assert reported and reported[0], (
        "expected a gleaning-stopped WARNING carrying the projected token "
        f"count; got: {[r.getMessage() for r in caplog.records]}"
    )
    return int(reported[0].group(1))


@pytest.mark.asyncio
@pytest.mark.parametrize("use_json", [False, True], ids=["text_mode", "json_mode"])
async def test_guard_still_counts_the_continue_prompt(
    monkeypatch, caplog, lightrag_caplog, use_json
):
    """Lengthening the continue prompt moves the projected gleaning token count
    by the exact padding length in both text and JSON modes."""
    baseline = await _projected_gleaning_tokens(monkeypatch, caplog, use_json)
    padding = "X" * 64
    padded = await _projected_gleaning_tokens(
        monkeypatch, caplog, use_json, continue_prompt_padding=padding
    )

    assert padded - baseline == len(padding), (
        f"projected gleaning input moved by {padded - baseline} tokens when "
        f"the continue prompt grew by {len(padding)}; the guard is no longer "
        "counting the continue prompt"
    )


@pytest.mark.asyncio
async def test_invariant_tokens_not_encoded_when_guard_inactive(monkeypatch):
    """With gleaning off, the system prompt is never fed to the tokenizer."""
    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "999999")

    global_config, inner = _make_counting_config(entity_extract_max_gleaning=0)
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = MINIMAL_EXTRACTION_RESULT

    await extract_entities(
        chunks=make_chunks(),
        global_config=global_config,
    )

    assert llm_func.await_count == 1
    prefix = PROMPTS["entity_extraction_system_prompt"].split("{", 1)[0]
    assert not [s for s in inner.encoded if s.startswith(prefix)], (
        "system prompt was encoded even though the gleaning guard never runs"
    )


# --------------------------------------------------------------------------- #
# Cancellation draining & unretrieved task exception guards
# --------------------------------------------------------------------------- #


@pytest.fixture
async def unretrieved_task_exceptions():
    """Collect what asyncio reports through the loop exception handler."""
    loop = asyncio.get_running_loop()
    reported: list[str] = []
    previous = loop.get_exception_handler()
    loop.set_exception_handler(
        lambda _loop, context: reported.append(context.get("message", ""))
    )
    yield reported
    loop.set_exception_handler(previous)


async def _never_retrieved(reported: list[str]) -> list[str]:
    for _ in range(3):
        await asyncio.sleep(0)
    gc.collect()
    return [m for m in reported if "never retrieved" in m]


@pytest.mark.asyncio
async def test_external_cancellation_cancels_the_running_chunk_tasks(
    monkeypatch, unretrieved_task_exceptions
):
    """External cancellation cancels all in-flight chunk tasks before returning."""
    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "999999")
    config = make_extract_global_config(llm_model_max_async=4)
    entered = 0
    all_entered = asyncio.Event()
    cancelled_calls = 0
    release = asyncio.Event()

    async def _llm(*args, **kwargs):
        nonlocal entered, cancelled_calls
        entered += 1
        if entered == 3:
            all_entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled_calls += 1
            raise
        raise RuntimeError("Queue is shutting down")

    config["role_llm_funcs"]["extract"].side_effect = _llm

    extract_task = asyncio.create_task(
        extract_entities(
            chunks=make_chunks(3, file_path="reports/annual.md"),
            global_config=config,
        )
    )
    await asyncio.wait_for(all_entered.wait(), timeout=2.0)
    extract_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await extract_task

    assert cancelled_calls == 3
    release.set()
    del extract_task
    assert await _never_retrieved(unretrieved_task_exceptions) == []


@pytest.mark.asyncio
async def test_sibling_failures_during_fail_fast_are_retrieved(
    monkeypatch, unretrieved_task_exceptions
):
    """When one chunk fails and a cancelled sibling raises a different error,
    the sibling exception is retrieved while the first failure propagates."""
    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "999999")
    config = make_extract_global_config(llm_model_max_async=4)
    entered = 0
    siblings_parked = asyncio.Event()
    release_first = asyncio.Event()

    async def _llm(*args, **kwargs):
        nonlocal entered
        entered += 1
        if entered == 1:
            await release_first.wait()
            raise ValueError("first failure")
        if entered == 3:
            siblings_parked.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            raise RuntimeError("sibling failed while being cancelled") from None
        return MINIMAL_EXTRACTION_RESULT

    config["role_llm_funcs"]["extract"].side_effect = _llm

    extract_task = asyncio.create_task(
        extract_entities(
            chunks=make_chunks(3, file_path="reports/annual.md"),
            global_config=config,
        )
    )
    await asyncio.wait_for(siblings_parked.wait(), timeout=2.0)
    release_first.set()
    with pytest.raises(ValueError, match="first failure"):
        await extract_task

    del extract_task
    assert await _never_retrieved(unretrieved_task_exceptions) == []


@pytest.mark.asyncio
async def test_cancelled_extraction_drains_running_and_queued_chunks(monkeypatch):
    """Cancelled extraction drains the running chunk and never starts queued ones."""
    prior_tasks = asyncio.all_tasks()
    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "999999")
    entered = asyncio.Event()
    release = asyncio.Event()
    children = []
    writes = []

    async def llm(*args, **kwargs):
        children.append(asyncio.current_task())
        entered.set()
        await release.wait()
        writes.append("late result")
        return "(entity<|#|>A<|#|>CONCEPT<|#|>Description)<|COMPLETE|>"

    config = make_extract_global_config(extract_func=llm, llm_model_max_async=1)
    chunks = make_chunks(3, content="Text", file_path="document.txt")
    parent = asyncio.create_task(extract_entities(chunks, config))
    await asyncio.wait_for(entered.wait(), timeout=5)
    parent.cancel()
    with pytest.raises(asyncio.CancelledError):
        await parent
    try:
        assert all(task.done() for task in children)
        release.set()
        await asyncio.sleep(0.05)
        assert writes == []
        assert len(children) == 1, "queued chunks must not start after cancellation"
    finally:
        release.set()
        remaining = asyncio.all_tasks() - prior_tasks
        for task in remaining:
            task.cancel()
        await asyncio.gather(*remaining, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["entity", "relation"])
async def test_merge_cancelled_during_task_creation_drains_writers(monkeypatch, phase):
    """Cancelled merge_nodes_and_edges drains in-flight writer tasks before returning."""
    prior_tasks = asyncio.all_tasks()
    entered = asyncio.Event()
    writer_entered = asyncio.Event()
    release = asyncio.Event()
    children = []
    writes = []

    @asynccontextmanager
    async def no_lock(*args, **kwargs):
        yield

    async def writer(*args, **kwargs):
        children.append(asyncio.current_task())
        writer_entered.set()
        await release.wait()
        writes.append("late graph write")
        return {} if phase == "entity" else None

    async def scheduling_yield(iteration, every=64):
        if iteration == every and every == (16 if phase == "entity" else 32):
            entered.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(operate, "get_storage_keyed_lock", no_lock)
    monkeypatch.setattr(operate, "_cooperative_yield", scheduling_yield)
    monkeypatch.setattr(operate, "_merge_nodes_then_upsert", writer)
    monkeypatch.setattr(operate, "_merge_edges_then_upsert", writer)
    nodes = {f"E{i}": [{}] for i in range(16)} if phase == "entity" else {}
    edges = {(f"S{i}", f"T{i}"): [{}] for i in range(32)} if phase == "relation" else {}
    parent = asyncio.create_task(
        operate.merge_nodes_and_edges(
            [(nodes, edges)],
            AsyncMock(),
            AsyncMock(),
            AsyncMock(),
            {"llm_model_max_async": 1},
            pipeline_status={"history_messages": []},
            pipeline_status_lock=asyncio.Lock(),
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=5)
    await asyncio.wait_for(writer_entered.wait(), timeout=5)
    parent.cancel()
    with pytest.raises(asyncio.CancelledError):
        await parent
    try:
        assert all(task.done() for task in children)
        release.set()
        await asyncio.sleep(0.05)
        assert writes == []
    finally:
        release.set()
        remaining = asyncio.all_tasks() - prior_tasks
        for task in remaining:
            task.cancel()
        await asyncio.gather(*remaining, return_exceptions=True)


# --------------------------------------------------------------------------- #
# Deterministic chunk_results ordering & fail-fast exception preservation
# --------------------------------------------------------------------------- #

_ORDERED_CHUNK_CONTENTS = {
    "chunk-alpha": "Alpha content.",
    "chunk-bravo": "Bravo content.",
    "chunk-charlie": "Charlie content.",
}


def _make_ordered_chunks() -> dict[str, dict]:
    return {
        key: {
            "tokens": len(content),
            "content": content,
            "full_doc_id": "doc-001",
            "chunk_order_index": index,
            "file_path": f"{key}.md",
        }
        for index, (key, content) in enumerate(_ORDERED_CHUNK_CONTENTS.items())
    }


async def _fake_ordered_extract(prompt: str, *args, **kwargs) -> str:
    for key, content in _ORDERED_CHUNK_CONTENTS.items():
        if content in prompt:
            name = key.split("-", 1)[1].upper()
            return (
                f"(entity<|#|>{name}<|#|>CONCEPT<|#|>Description of {name})<|COMPLETE|>"
            )
    raise AssertionError(f"prompt carried no known chunk content: {prompt[:200]!r}")


@pytest.fixture
def _reverse_done_order(monkeypatch):
    """Make ``asyncio.wait`` return ``done`` in reverse creation order."""
    real_wait = asyncio.wait

    async def _reversed_wait(fs, **kwargs):
        done, pending = await real_wait(fs, **kwargs)
        completed_in_creation_order = [task for task in fs if task in done]
        return list(reversed(completed_in_creation_order)), pending

    monkeypatch.setattr(asyncio, "wait", _reversed_wait)


@pytest.mark.asyncio
async def test_chunk_results_follow_ordered_chunks(_reverse_done_order):
    """``chunk_results[i]`` must correspond to ``ordered_chunks[i]`` even when
    the completed-task container iterates in reverse order."""
    chunks = _make_ordered_chunks()
    chunk_results = await extract_entities(
        chunks=chunks,
        global_config=make_extract_global_config(
            extract_func=AsyncMock(side_effect=_fake_ordered_extract),
            llm_model_max_async=3,
        ),
    )

    assert len(chunk_results) == len(chunks)
    observed_source_ids = [
        record["source_id"]
        for maybe_nodes, _maybe_edges in chunk_results
        for records in maybe_nodes.values()
        for record in records
    ]
    assert observed_source_ids == list(chunks)


@pytest.mark.asyncio
async def test_first_exception_survives_pending_tasks():
    """Fail-fast propagates the failing chunk's exception, not InvalidStateError
    from touching a still-pending sibling task."""

    async def _explode_on_alpha(prompt: str, *args, **kwargs) -> str:
        if _ORDERED_CHUNK_CONTENTS["chunk-alpha"] in prompt:
            raise RuntimeError("chunk alpha exploded")
        await asyncio.Event().wait()
        raise AssertionError("unreachable: sibling should have been cancelled")

    with pytest.raises(Exception) as excinfo:
        await extract_entities(
            chunks=_make_ordered_chunks(),
            global_config=make_extract_global_config(
                extract_func=AsyncMock(side_effect=_explode_on_alpha),
                llm_model_max_async=3,
            ),
        )

    message = str(excinfo.value)
    assert "chunk alpha exploded" in message
    assert "InvalidStateError" not in message
