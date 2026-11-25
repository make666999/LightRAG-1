"""Offline regression tests for the repository's example scripts.

Covers:
- Gemini demo embedding callbacks forwarding declared ``embedding_dim`` and
  ``model_name`` with ``send_dimensions=True``.
- ``examples/generate_query.py`` CLI model selection and reasoning_effort wiring.
- OpenAI/Ollama/Cloudflare demo ``main`` error handling (no UnboundLocalError)
  and storage finalization.
- OpenAI demos' real ``configure_logging`` creating relative and nested
  absolute log directories in an isolated subprocess.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import numpy as np
import pytest

from lightrag.utils import EmbeddingFunc, wrap_embedding_func_with_attrs

pytestmark = pytest.mark.offline

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"


def _load_demo_definition(demo: str, name: str, namespace: dict):
    """Execute only the top-level definition ``name`` of ``examples/<demo>``.

    Importing a demo would validate API keys, create directories, apply
    ``nest_asyncio`` or configure logging, so the definition is compiled on its
    own. ``namespace`` must stub every global the definition reads.
    """
    path = EXAMPLES_DIR / demo
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )
    exec(  # noqa: S102 - trusted repository source, see the docstring
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[name]


# ---------------------------------------------------------------------------
# Gemini demo embedding callbacks
# ---------------------------------------------------------------------------

GEMINI_DEMOS = [
    "lightrag_gemini_demo.py",
    "lightrag_gemini_postgres_demo.py",
    "lightrag_gemini_workspace_demo.py",
]


def _load_embedding_func(demo: str, provider: AsyncMock) -> EmbeddingFunc:
    """Load only the demo's ``embedding_func`` against a stubbed Gemini binding."""
    return _load_demo_definition(
        demo,
        "embedding_func",
        {
            "np": np,
            "os": os,
            "wrap_embedding_func_with_attrs": wrap_embedding_func_with_attrs,
            "gemini_embed": SimpleNamespace(func=provider),
            "GEMINI_API_KEY": "test-key",
        },
    )


@pytest.mark.parametrize("demo", GEMINI_DEMOS)
async def test_gemini_demo_forwards_declared_embedding_dimension(demo, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    provider = AsyncMock()
    embedding_func = _load_embedding_func(demo, provider)
    vectors = np.zeros((1, embedding_func.embedding_dim), dtype=np.float32)
    provider.return_value = vectors

    result = await embedding_func(["sample"])

    assert result is vectors
    # Without send_dimensions, Gemini returns its default size rather than the
    # declared one and the wrapper's shape check rejects every batch.
    assert embedding_func.send_dimensions is True
    provider.assert_awaited_once_with(
        ["sample"],
        api_key="test-key",
        # The declared model_name keys the vector space and the configuration
        # binding, so it must name the model that actually produced the vectors.
        model=embedding_func.model_name,
        embedding_dim=embedding_func.embedding_dim,
    )


# ---------------------------------------------------------------------------
# generate_query.py CLI
# ---------------------------------------------------------------------------

GENERATE_QUERY_SCRIPT = EXAMPLES_DIR / "generate_query.py"


@pytest.mark.parametrize(
    ("args", "model", "reasoning_effort"),
    [
        ([], "gpt-6-luna", "none"),
        (["--model", "gpt-4o-mini"], "gpt-4o-mini", None),
    ],
)
def test_generate_query_uses_selected_model(
    monkeypatch, tmp_path, args, model, reasoning_effort
):
    openai = pytest.importorskip("openai")
    requests = []

    def respond(request):
        assert request.url.path == "/v1/chat/completions"
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-generated-queries",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Sample question"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    with openai.OpenAI(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as client:
        monkeypatch.setattr(openai, "OpenAI", lambda: client)
        monkeypatch.setattr(sys, "argv", [str(GENERATE_QUERY_SCRIPT), *args])
        monkeypatch.chdir(tmp_path)
        runpy.run_path(str(GENERATE_QUERY_SCRIPT), run_name="__main__")

    assert len(requests) == 1
    assert requests[0]["model"] == model
    assert requests[0].get("reasoning_effort") == reasoning_effort
    assert "generate 5 questions" in requests[0]["messages"][0]["content"]
    assert (tmp_path / "queries.txt").read_text() == "Sample question"


# ---------------------------------------------------------------------------
# Demo main() cleanup & finalization
# ---------------------------------------------------------------------------

CLEANUP_DEMOS = [
    "lightrag_openai_demo.py",
    "lightrag_openai_compatible_demo.py",
    "lightrag_ollama_demo.py",
    "unofficial-sample/lightrag_cloudflare_demo.py",
]


def _load_main(demo, working_dir, initialize_rag):
    return _load_demo_definition(
        demo,
        "main",
        {
            "os": os,
            "inspect": inspect,
            "WORKING_DIR": str(working_dir),
            "initialize_rag": initialize_rag,
            "QueryParam": Mock(),
        },
    )


@pytest.mark.parametrize("demo", CLEANUP_DEMOS)
async def test_initialization_failure_does_not_mask_diagnostic(
    demo, tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-placeholder")
    main = _load_main(
        demo,
        tmp_path,
        AsyncMock(side_effect=RuntimeError("storage unavailable")),
    )

    # Without ``rag = None`` before the ``try``, ``finally`` reads the
    # unassigned local and raises UnboundLocalError over the handled failure.
    await main()

    assert "storage unavailable" in capsys.readouterr().out


@pytest.mark.parametrize("demo", CLEANUP_DEMOS)
async def test_initialized_rag_is_finalized(demo, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-placeholder")
    (tmp_path / "book.txt").write_text("A short offline book.", encoding="utf-8")
    rag = Mock(
        embedding_func=AsyncMock(return_value=np.zeros((1, 2))),
        ainsert=AsyncMock(),
        aquery=AsyncMock(return_value="answer"),
        finalize_storages=AsyncMock(),
        llm_response_cache=Mock(index_done_callback=AsyncMock()),
    )
    main = _load_main(demo, tmp_path, AsyncMock(return_value=rag))

    await main()

    assert "An error occurred" not in capsys.readouterr().out
    rag.finalize_storages.assert_awaited_once()


# ---------------------------------------------------------------------------
# OpenAI demo configure_logging in an isolated subprocess
# ---------------------------------------------------------------------------

LOGGING_DEMOS = ("lightrag_openai_demo.py", "lightrag_openai_compatible_demo.py")
RELATIVE = {"relative-log-dir": True, "nested-absolute-log-dir": False}

_LOGGING_CHILD = """
import importlib.util, json, logging, os, sys, traceback
out = {}
for case in json.loads(sys.argv[1]):
    try:
        os.chdir(case["cwd"])
        os.environ["LOG_DIR"] = case["log_dir"]
        spec = importlib.util.spec_from_file_location(
            f"demo_under_test_{len(out)}", case["demo"])
        demo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(demo)
        demo.configure_logging()
        demo.logger.info("demo logging is ready")
        for handler in logging.getLogger("lightrag").handlers:
            handler.flush()
        out[case["id"]] = None
    except Exception:
        out[case["id"]] = traceback.format_exc()
logging.shutdown()
print(json.dumps(out))
"""


def _case_id(demo: str, relative_id: str) -> str:
    return f"{relative_id}-{demo}"


@pytest.fixture(scope="module")
def demo_runs(tmp_path_factory):
    """Run every case in one child; map case id to (error, cwd, log dir)."""
    cases = []
    layout = {}
    for demo in LOGGING_DEMOS:
        for relative_id, relative in RELATIVE.items():
            case_id = _case_id(demo, relative_id)
            cwd = tmp_path_factory.mktemp("demo-logging")
            log_dir = Path("logs") if relative else cwd / "nested" / "logs"
            layout[case_id] = (cwd, log_dir)
            cases.append(
                {
                    "id": case_id,
                    "cwd": str(cwd),
                    "log_dir": str(log_dir),
                    "demo": str(EXAMPLES_DIR / demo),
                }
            )
    # Import this checkout's lightrag, not whichever one is installed: the
    # child's sys.path[0] is not the repository.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO_ROOT), os.environ.get("PYTHONPATH")) if p
    )
    result = subprocess.run(
        [sys.executable, "-c", _LOGGING_CHILD, json.dumps(cases)],
        cwd=cases[0]["cwd"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    errors = json.loads(result.stdout.strip().splitlines()[-1])
    return {case_id: (errors[case_id], *layout[case_id]) for case_id in layout}


@pytest.mark.parametrize("demo", LOGGING_DEMOS)
@pytest.mark.parametrize("relative_id", list(RELATIVE))
def test_log_directory_is_created(demo, relative_id, demo_runs):
    error, cwd, log_dir = demo_runs[_case_id(demo, relative_id)]
    assert error is None, error
    files = list((cwd / log_dir).glob("*.log"))
    assert len(files) == 1
    assert "demo logging is ready" in files[0].read_text(encoding="utf-8")
