"""Offline regression tests for the reproduction scripts under ``reproduce/``.

Covers:
- ``reproduce/batch_eval.py`` question-to-answer alignment and pre-submission
  validation.
- ``reproduce/Step_2.py`` ``get_summary`` token window sampling.
- ``reproduce/Step_3*.py`` repeated-run JSON result overwriting and error log
  appending.
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, Mock

import pytest

pytestmark = pytest.mark.offline

REPRODUCE_DIR = Path(__file__).resolve().parents[2] / "reproduce"


# ---------------------------------------------------------------------------
# batch_eval.py question-to-answer alignment
# ---------------------------------------------------------------------------

BATCH_EVAL_SCRIPT = REPRODUCE_DIR / "batch_eval.py"


@contextmanager
def _open_jsonlines(path, mode):
    with open(path, mode, encoding="utf-8") as output:
        writer = Mock()
        writer.write.side_effect = lambda row: output.write(json.dumps(row) + "\n")
        yield writer


@pytest.fixture
def batch_eval_script(monkeypatch):
    jsonlines = ModuleType("jsonlines")
    jsonlines.open = Mock(side_effect=_open_jsonlines)
    monkeypatch.setitem(sys.modules, "jsonlines", jsonlines)
    spec = importlib.util.spec_from_file_location(
        "reproduce_batch_eval", BATCH_EVAL_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "OpenAI", Mock())
    return module


def _row(query, answer):
    return {"query": query, "result": answer}


def _run_batch(tmp_path, script, first, second, questions=("First?", "Second?")):
    question_file = tmp_path / "questions.txt"
    question_file.write_text(
        "\n".join(f"- Question {i}: {q}" for i, q in enumerate(questions, 1))
    )
    paths = []
    for name, rows in (("first", first), ("second", second)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(rows))
        paths.append(path)
    script.batch_eval(question_file, *paths, tmp_path / "requests.jsonl")


def _prompts(tmp_path):
    lines = (tmp_path / "requests.jsonl").read_text().splitlines()
    return [json.loads(line)["body"]["messages"][1]["content"] for line in lines]


def _assert_nothing_submitted(script):
    script.OpenAI.assert_not_called()
    script.jsonlines.open.assert_not_called()


def test_reordered_results_are_matched_by_query(tmp_path, batch_eval_script):
    _run_batch(
        tmp_path,
        batch_eval_script,
        [_row("Second?", "A2"), _row("First?", "A1")],
        [_row("First?", "B1"), _row("Second?", "B2")],
    )
    first, second = _prompts(tmp_path)
    assert ("First?" in first, "A1" in first, "B1" in first) == (True, True, True)
    assert ("A2" in first, "B2" in first) == (False, False)
    assert ("Second?" in second, "A2" in second, "B2" in second) == (
        True,
        True,
        True,
    )
    batch_eval_script.OpenAI.return_value.batches.create.assert_called_once()


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        pytest.param(
            [_row("Second?", "A2")],
            r"first\.json: 1 of 2 questions have no answer \('First\?'\)",
            id="missing",
        ),
        pytest.param(
            [_row("First?", "A1"), _row("First?", "again"), _row("Second?", "A2")],
            r"first\.json: question answered twice: 'First\?'",
            id="duplicate",
        ),
        pytest.param(
            [_row("First?", "A1"), _row("Unknown?", "other")],
            r"first\.json: entry 1 answers a question that is not in the question "
            r"file: 'Unknown\?'",
            id="unknown",
        ),
        pytest.param(
            [_row("First?", "A1"), {"query": "Second?"}],
            r"first\.json: entry 1 has no 'result': 'Second\?'",
            id="no-result",
        ),
        pytest.param(
            [_row("First?", "A1"), "Second?"],
            r"first\.json: entry 1 answers a question that is not in the question "
            r"file: None",
            id="not-an-object",
        ),
    ],
)
def test_bad_results_fail_before_submission(tmp_path, batch_eval_script, rows, message):
    with pytest.raises(ValueError, match=message):
        _run_batch(
            tmp_path,
            batch_eval_script,
            rows,
            [_row("First?", "B1"), _row("Second?", "B2")],
        )
    _assert_nothing_submitted(batch_eval_script)


def test_missing_answers_are_listed_with_a_bound(tmp_path, batch_eval_script):
    questions = [f"Q{i}?" for i in range(1, 6)]
    with pytest.raises(
        ValueError, match=r"5 of 5 .*'Q1\?', 'Q2\?', 'Q3\?' \(\+2 more\)"
    ):
        _run_batch(tmp_path, batch_eval_script, [], [], questions=questions)
    _assert_nothing_submitted(batch_eval_script)


@pytest.mark.parametrize(
    ("questions", "message"),
    [
        pytest.param(("First?", "First?"), r"repeated: 'First\?'", id="duplicate"),
        pytest.param((), r"no '- Question N: \.\.\.' lines found", id="empty"),
    ],
)
def test_bad_question_file_fails_before_submission(
    tmp_path, batch_eval_script, questions, message
):
    with pytest.raises(ValueError, match=message):
        _run_batch(
            tmp_path,
            batch_eval_script,
            [_row("First?", "A1")],
            [_row("First?", "B1")],
            questions=questions,
        )
    _assert_nothing_submitted(batch_eval_script)


def test_markdown_questions_match_query_script_normalization(
    tmp_path, batch_eval_script
):
    _run_batch(
        tmp_path,
        batch_eval_script,
        [_row("First?", "A1")],
        [_row("First?", "B1")],
        questions=("**First?**",),
    )
    (prompt,) = _prompts(tmp_path)
    assert ("First?" in prompt, "**First?**" in prompt) == (True, False)
    batch_eval_script.OpenAI.return_value.batches.create.assert_called_once()


# ---------------------------------------------------------------------------
# Step_2.py get_summary token sampling
# ---------------------------------------------------------------------------


class _WordTokenizer:
    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)


def _load_get_summary():
    # Importing Step_2.py would download GPT-2 and call OpenAI at import time,
    # so compile only the get_summary definition against a stub tokenizer.
    source = REPRODUCE_DIR / "Step_2.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if getattr(n, "name", None) == "get_summary")
    namespace = {"tokenizer": _WordTokenizer()}
    exec(  # noqa: S102 - trusted repository source, compiled for the test only
        compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace["get_summary"]


@pytest.mark.parametrize(
    "count,budget,head,tail",
    [
        (5000, 2000, (1000, 2000), (3000, 4000)),
        (10000, 1000, (1000, 1500), (8500, 9000)),
        (5000, 1999, (1000, 1999), (3001, 4000)),
    ],
)
def test_summary_retains_both_existing_offset_windows(count, budget, head, tail):
    tokens = [str(i) for i in range(count)]
    actual = _load_get_summary()(" ".join(tokens), budget).split()
    expected = [str(i) for i in range(*head)] + [str(i) for i in range(*tail)]
    assert actual == expected
    assert len(actual) <= budget


@pytest.mark.parametrize(
    "count,budget,expected",
    [
        (0, 2000, (0, 0)),
        (300, 2000, (0, 300)),
        (3999, 2000, (0, 2000)),
    ],
)
def test_short_context_is_sampled_from_the_start(count, budget, expected):
    # Below 2 * (1000 + budget // 2) tokens the offset windows would be empty
    # or overlap, which dropped or repeated text instead of summarizing it.
    tokens = [str(i) for i in range(count)]
    actual = _load_get_summary()(" ".join(tokens), budget).split()
    assert actual == [str(i) for i in range(*expected)]


def test_windows_at_threshold_are_adjacent_and_disjoint():
    tokens = [str(i) for i in range(4000)]
    actual = _load_get_summary()(" ".join(tokens), 2000).split()
    assert actual == tokens[1000:3000]


# ---------------------------------------------------------------------------
# Step_3*.py result output & error log
# ---------------------------------------------------------------------------


def _load_step3_script(name: str):
    spec = importlib.util.spec_from_file_location(
        f"reproduce_{Path(name).stem}", REPRODUCE_DIR / name
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json_objects(text: str) -> list:
    """Decode the error log, which is a stream of JSON objects, not an array."""
    decoder, objects, index = json.JSONDecoder(), [], 0
    while index < len(text):
        if text[index].isspace():
            index += 1
            continue
        obj, index = decoder.raw_decode(text, index)
        objects.append(obj)
    return objects


@pytest.mark.parametrize("script", ["Step_3.py", "Step_3_openai_compatible.py"])
def test_repeated_runs_leave_valid_current_results(script, tmp_path, monkeypatch):
    module = _load_step3_script(script)
    loop = asyncio.new_event_loop()
    monkeypatch.setattr(module, "always_get_an_event_loop", lambda: loop)
    output, errors = tmp_path / "results.json", tmp_path / "errors.json"
    answers = ["old answer", RuntimeError("old boom")]
    answers += ["new answer", RuntimeError("new boom")]
    rag = type("Rag", (), {"aquery": AsyncMock(side_effect=answers)})()
    try:
        for run in ["old", "new"]:
            module.run_queries_and_save_to_json(
                [f"{run} question", f"{run} failing"], rag, None, output, errors
            )
    finally:
        loop.close()

    # Results are overwritten: only the latest run, as one JSON array.
    assert json.loads(output.read_text(encoding="utf-8")) == [
        {"query": "new question", "result": "new answer"}
    ]
    # The error log still appends across runs.
    assert _json_objects(errors.read_text(encoding="utf-8")) == [
        {"query": "old failing", "error": "old boom"},
        {"query": "new failing", "error": "new boom"},
    ]
