"""Server startup tests: custom chunker, embedding model guard, rerank flags, log dir, and startup error diagnostics."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from argparse import Namespace
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from uvicorn import Config
from uvicorn.lifespan.on import LifespanOn

from lightrag.api.startup_errors import (
    ConsoleFormatter,
    StartupDiagnostic,
    StartupErrorMiddleware,
)
from lightrag.chunker import plugins, registry
from lightrag.exceptions import (
    ConfigurationAnchorLockError,
    ConfigurationIdentityError,
    CorruptStorageSnapshotError,
    EmbeddingBaselineMismatchError,
    VectorSpaceMismatchError,
    VectorStorageEmptyError,
    WorkingDirectoryInUseError,
)
from .conftest import LightRAGConstructed

pytestmark = pytest.mark.offline


# --------------------------------------------------------------------------- #
# Shared logging helper
# --------------------------------------------------------------------------- #


@contextmanager
def restored_loggers(*prefixes):
    """Put back every logger a logging configuration under test rewrites."""
    names = [
        name
        for name in logging.root.manager.loggerDict
        if any(name == p or name.startswith(p + ".") for p in prefixes)
    ]
    names += [p for p in prefixes if p not in names]
    saved = {
        name: (lg.handlers[:], lg.filters[:], lg.level, lg.propagate)
        for name in names
        for lg in [logging.getLogger(name)]
    }
    try:
        yield
    finally:
        for name, (handlers, filters, level, propagate) in saved.items():
            lg = logging.getLogger(name)
            for handler in lg.handlers:
                if handler not in handlers:
                    handler.close()
            lg.handlers, lg.filters = handlers, filters
            lg.setLevel(level)
            lg.propagate = propagate


# --------------------------------------------------------------------------- #
# Custom chunker startup discovery and validation
# --------------------------------------------------------------------------- #


@pytest.fixture
def startup(monkeypatch, prepare_startup, lightrag_caplog):
    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.setattr(registry, "_DUPLICATES", set())
    monkeypatch.setattr(plugins, "_loaded", False)
    monkeypatch.setattr(plugins, "_failures", [])
    prepared = prepare_startup({"CUSTOM_CHUNKER": ""})
    return prepared.server, prepared.args, prepared.captured, LightRAGConstructed


@pytest.mark.parametrize("factory", ["create_app", "get_application"])
def test_discovered_chunker_injected_before_constructor(startup, monkeypatch, factory):
    server, args, captured, constructed = startup

    def callback(*args):
        return args

    monkeypatch.setitem(
        sys.modules, "startup_chunker_impl", SimpleNamespace(chunk=callback)
    )

    def register():
        registry.register_chunker(
            registry.ChunkerSpec("acme", "startup_chunker_impl:chunk", "1", "Acme")
        )

    ep = SimpleNamespace(name="acme", value="provider:register", load=lambda: register)
    monkeypatch.setattr(plugins, "entry_points", lambda **kwargs: [ep])

    def validate():
        assert registry.registered_chunker_names() == ("acme",)

    monkeypatch.setattr(server, "validate_parser_routing_config", validate)
    args.custom_chunker = "acme"
    with pytest.raises(constructed):
        getattr(server, factory)(args)
    assert captured[0]["chunking_func"](*range(6)) == tuple(range(6))


def test_unset_does_not_override_constructor_default(startup, monkeypatch):
    server, args, captured, constructed = startup
    monkeypatch.setattr(plugins, "entry_points", lambda **kwargs: [])
    with pytest.raises(constructed):
        server.create_app(args)
    assert "chunking_func" not in captured[0]


@pytest.mark.parametrize(
    "selected,outcome",
    [
        (
            "acme",
            "selected chunker 'acme' validated; C and no-selector chunking remain available",
        ),
        ("partial", "CUSTOM_CHUNKER='partial' could not be activated; startup aborted"),
        (
            "",
            "CUSTOM_CHUNKER is unset; built-in callback and existing C admission/fallback are unchanged",
        ),
    ],
)
def test_failed_provider_log_alone_describes_selected_chunker_outcome(
    startup, monkeypatch, caplog, selected, outcome
):
    server, args, captured, constructed = startup
    monkeypatch.setitem(
        sys.modules, "startup_chunker_impl", SimpleNamespace(chunk=lambda *args: args)
    )

    def good():
        registry.register_chunker(
            registry.ChunkerSpec("acme", "startup_chunker_impl:chunk", "1", "Acme")
        )

    def broken():
        registry.register_chunker(
            registry.ChunkerSpec("partial", "absent:chunk", "1", "Partial")
        )
        raise RuntimeError("provider failed")

    entries = [
        SimpleNamespace(name=name, value=f"{name}:register", load=lambda fn=fn: fn)
        for name, fn in [("good", good), ("broken", broken)]
    ]
    monkeypatch.setattr(plugins, "entry_points", lambda **kwargs: entries)
    args.custom_chunker = selected
    with pytest.raises(ValueError if selected == "partial" else constructed):
        server.create_app(args)
    errors = [
        r.getMessage()
        for r in caplog.records
        if "skipped failed plugin" in r.getMessage()
    ]
    assert len(errors) == 1
    assert "broken:register" in errors[0] and "provider failed" in errors[0]
    assert outcome in errors[0]
    if selected == "partial":
        assert not captured
    elif selected:
        assert captured[0]["chunking_func"](*range(6)) == tuple(range(6))


@pytest.mark.parametrize(
    "name,match", [("missing", "unknown chunker"), ("pkg:func", "import paths")]
)
def test_invalid_selection_fails_before_constructor(startup, monkeypatch, name, match):
    server, args, captured, _ = startup
    monkeypatch.setattr(plugins, "entry_points", lambda **kwargs: [])
    args.custom_chunker = name
    with pytest.raises(ValueError, match=match):
        server.create_app(args)
    assert captured == []


def test_ingress_error_names_config_and_selectable_choices(monkeypatch):
    from lightrag.api.routers.document_routes import _validate_custom_chunking_available
    from lightrag.chunker import chunking_by_token_size

    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.setattr(registry, "_DUPLICATES", set())
    registry.register_chunker(registry.ChunkerSpec("acme", "absent:chunk", "1", "Acme"))
    registry.register_chunker(registry.ChunkerSpec("dup", "absent:chunk", "1", "Dup"))
    registry.register_chunker(registry.ChunkerSpec("dup", "absent:other", "1", "Dup"))
    with pytest.raises(ValueError, match="CUSTOM_CHUNKER.*acme") as excinfo:
        _validate_custom_chunking_available(
            "C", SimpleNamespace(chunking_func=chunking_by_token_size)
        )
    assert "dup" not in str(excinfo.value)


def test_invalid_selection_prints_a_boxed_notice_and_still_aborts(startup, monkeypatch):
    server, args, captured, _ = startup
    monkeypatch.setattr(plugins, "entry_points", lambda **kwargs: [])
    colors = MagicMock()
    monkeypatch.setattr(server, "ASCIIColors", colors)
    args.custom_chunker = "pkg:func"
    with pytest.raises(ValueError, match="import paths"):
        server.create_app(args)
    notice = "\n".join(str(a) for call in colors.mock_calls for a in call.args)
    assert "ERROR: invalid CUSTOM_CHUNKER selection" in notice
    assert "import paths are not supported" in notice
    assert "docs/ThirdPartyChunker.md" in notice
    assert captured == []


def test_invalid_cli_selection_exits_nonzero(monkeypatch):
    monkeypatch.setenv("AUTH_ACCOUNTS", "")
    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("EMBEDDING_BINDING", "openai")
    monkeypatch.setenv("UI_TEMPLATES_DIR", "")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from lightrag.api.config import parse_args; from lightrag.api.lightrag_server import create_app; create_app(parse_args())",
            "--custom-chunker",
            "missing-chunker",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "unknown chunker" in result.stderr


# --------------------------------------------------------------------------- #
# EMBEDDING_MODEL required at server startup
# --------------------------------------------------------------------------- #


def _parse_server_args():
    from lightrag.api.config import initialize_config, parse_args

    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server"]
        args = parse_args()
    finally:
        sys.argv = original_argv
    initialize_config(args, force=True)
    return args


@pytest.fixture
def server_args(monkeypatch):
    monkeypatch.setenv("LLM_BINDING", "ollama")
    monkeypatch.setenv("EMBEDDING_BINDING", "ollama")
    return _parse_server_args


@pytest.mark.parametrize("configured", [None, "   "])
def test_create_app_refuses_without_an_embedding_model(
    monkeypatch, server_args, configured
):
    """Unset, empty and whitespace are the same condition: no name to record."""
    if configured is None:
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    else:
        monkeypatch.setenv("EMBEDDING_MODEL", configured)
    args = server_args()

    from lightrag.api.lightrag_server import create_app

    with pytest.raises(SystemExit) as excinfo:
        create_app(args)

    message = str(excinfo.value)
    assert "EMBEDDING_MODEL" in message
    assert "lightrag-rebuild-vdb" in message
    assert "rolling" in message


def test_an_args_object_without_the_field_is_refused_too():
    from lightrag.api.lightrag_server import create_app

    with pytest.raises(SystemExit, match="EMBEDDING_MODEL"):
        create_app(Namespace())


# --------------------------------------------------------------------------- #
# RERANK_ENABLE_CHUNKING boolean parsing for Cohere binding
# --------------------------------------------------------------------------- #


@pytest.fixture
def cohere_rerank_calls(monkeypatch, prepare_startup):
    """Build the server with a stubbed Cohere rerank and return its call kwargs."""

    def run(enable_chunking: str | None) -> dict:
        import lightrag.rerank as rerank

        rerank_calls: list[dict] = []

        async def fake_cohere_rerank(
            query, documents, top_n=None, api_key=None, model="m", base_url="u", **kw
        ):
            rerank_calls.append(kw)
            return []

        monkeypatch.setattr(rerank, "cohere_rerank", fake_cohere_rerank)
        prepared = prepare_startup(
            {
                "RERANK_BINDING": "cohere",
                "RERANK_MAX_TOKENS_PER_DOC": None,
                "RERANK_ENABLE_CHUNKING": enable_chunking,
            }
        )

        rerank_func = prepared.constructor_kwargs()["rerank_model_func"]
        asyncio.run(rerank_func(query="q", documents=["d"]))
        assert len(rerank_calls) == 1
        return rerank_calls[0]

    return run


@pytest.mark.parametrize("value", ["true", "1", " Yes "])
def test_true_spellings_enable_chunking(cohere_rerank_calls, value):
    assert cohere_rerank_calls(value)["enable_chunking"] is True


@pytest.mark.parametrize("value", [None, "0"])
def test_false_spellings_and_unset_disable_chunking(cohere_rerank_calls, value):
    assert cohere_rerank_calls(value)["enable_chunking"] is False


# --------------------------------------------------------------------------- #
# LOG_DIR directory creation in configure_logging
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "relative", [True, False], ids=["relative-log-dir", "nested-absolute-log-dir"]
)
def test_configure_logging_creates_the_log_dir(relative, tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["lightrag-server"])
    from lightrag.api import lightrag_server

    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "logs" if relative else tmp_path / "nested" / "logs"
    monkeypatch.setenv("LOG_DIR", "logs" if relative else str(log_dir))
    with restored_loggers("uvicorn", "lightrag"):
        lightrag_server.configure_logging()
        logger = logging.getLogger("lightrag")
        logger.info("server logging is ready")
        for handler in logger.handlers:
            handler.flush()

    log_file = log_dir / lightrag_server.DEFAULT_LOG_FILENAME
    assert "server logging is ready" in log_file.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# StartupErrorMiddleware diagnostics through FastAPI and Uvicorn lifespan
# --------------------------------------------------------------------------- #


def make_app(error, cleaned, *, shutdown=False):
    @asynccontextmanager
    async def lifespan(app):
        try:
            if not shutdown:
                raise error
            yield
            raise error
        finally:
            cleaned.append(True)

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        StartupErrorMiddleware,
        workspace="space1",
        vector_storage="FaissVectorDBStorage",
    )
    return app


def driver(app):
    return LifespanOn(Config(app, lifespan="on", log_config=None, ws="none"))


@pytest.fixture
def logs(caplog):
    """caplog over ``uvicorn.error`` AND the ``lightrag`` logger, which does not
    propagate to the root handler caplog listens on."""
    lightrag_logger = logging.getLogger("lightrag")
    lightrag_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.INFO, logger="uvicorn.error"):
            yield caplog
    finally:
        lightrag_logger.removeHandler(caplog.handler)


def _lifespan_messages(caplog):
    return "\n".join(record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize("tty", [True, False])
async def test_empty_index_is_concise_but_still_fails_and_cleans_up(
    monkeypatch, logs, tty
):
    monkeypatch.setattr("sys.stderr.isatty", lambda: tty)
    monkeypatch.delenv("NO_COLOR", raising=False)
    cleaned = []
    error = VectorStorageEmptyError(vdb_name="chunks", source="text chunk storage")
    life = driver(make_app(error, cleaned))
    await life.startup()
    messages = "\n".join(record.getMessage() for record in logs.records)
    assert life.should_exit and life.startup_failed
    assert cleaned == [True]
    assert "Startup blocked" in messages
    assert "space1" in messages and "FaissVectorDBStorage" in messages
    assert "chunks" in messages and "text chunk storage" in messages
    assert "lightrag-rebuild-vdb" in messages
    assert "lightrag-clear-storage" in messages
    assert "Traceback" not in messages
    assert "Application startup complete" not in messages
    assert "\033[" not in messages
    diagnostic = [r for r in logs.records if "Startup blocked" in r.getMessage()]
    assert len(diagnostic) == 1
    assert diagnostic[0].name == "lightrag"
    assert isinstance(diagnostic[0].msg, StartupDiagnostic)


@pytest.mark.parametrize(
    "tty,no_color,colored",
    [(True, False, True), (False, False, False), (True, True, False)],
)
def test_the_console_formatter_highlights_a_diagnostic_only_on_a_terminal(
    monkeypatch, tty, no_color, colored
):
    monkeypatch.setattr("sys.stderr.isatty", lambda: tty)
    monkeypatch.delenv("NO_COLOR", raising=False)
    if no_color:
        monkeypatch.setenv("NO_COLOR", "1")
    formatter = ConsoleFormatter("%(levelname)s: %(message)s")

    def record(msg):
        return logging.LogRecord("uvicorn.error", logging.ERROR, "", 0, msg, None, None)

    assert ("\033[1;31m" in formatter.format(record(StartupDiagnostic("x")))) is colored
    assert "\033[" not in formatter.format(record("Application startup failed."))


@pytest.mark.asyncio
async def test_the_log_file_stays_plain_while_the_terminal_is_highlighted(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr("sys.argv", ["lightrag-server"])
    from lightrag.api import lightrag_server

    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr("sys.stderr.isatty", lambda: True)
    with restored_loggers("uvicorn", "lightrag"):
        lightrag_server.configure_logging()
        life = driver(
            make_app(VectorStorageEmptyError(vdb_name="chunks"), cleaned := [])
        )
        await life.startup()
        for name in ("uvicorn.error", "lightrag"):
            for handler in logging.getLogger(name).handlers:
                handler.flush()
        log_file = (tmp_path / lightrag_server.DEFAULT_LOG_FILENAME).read_text(
            encoding="utf-8"
        )

    stderr = capsys.readouterr().err
    assert life.startup_failed and cleaned == [True]
    assert log_file.count("Startup blocked") == 1
    assert "\033[" not in log_file
    assert stderr.count("Startup blocked") == 1
    assert "\033[1;31mERROR: Startup blocked" in stderr


@pytest.mark.asyncio
async def test_gunicorn_workers_still_report_the_refusal(monkeypatch, tmp_path, capsys):
    from lightrag.api import gunicorn_config

    log_file_path = tmp_path / "lightrag.log"
    monkeypatch.setattr(gunicorn_config, "log_file_path", str(log_file_path))
    monkeypatch.setattr(gunicorn_config, "loglevel", "info")
    with restored_loggers("uvicorn", "lightrag"):
        gunicorn_config.post_fork(server=None, worker=None)
        assert logging.getLogger("uvicorn.error").level == logging.CRITICAL
        life = driver(
            make_app(VectorStorageEmptyError(vdb_name="chunks"), cleaned := [])
        )
        await life.startup()
        for handler in logging.getLogger("lightrag").handlers:
            handler.flush()
        log_file = log_file_path.read_text(encoding="utf-8")

    assert life.startup_failed and cleaned == [True]
    assert log_file.count("Startup blocked") == 1
    assert "lightrag-rebuild-vdb" in log_file
    assert capsys.readouterr().err.count("Startup blocked") == 1


@pytest.mark.asyncio
async def test_the_storage_workspace_wins_over_the_server_workspace(logs):
    error = VectorStorageEmptyError(vdb_name="chunks", workspace="qdrant_scope")
    life = driver(make_app(error, []))
    await life.startup()
    assert "Workspace: qdrant_scope" in logs.text
    assert "space1" not in logs.text


@pytest.mark.asyncio
async def test_unexpected_startup_error_keeps_traceback(caplog):
    cleaned = []
    life = driver(make_app(RuntimeError("connection broke"), cleaned))
    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        await life.startup()
    assert life.should_exit and life.startup_failed
    assert cleaned == [True]
    assert "Traceback" in caplog.text
    assert "connection broke" in caplog.text


@pytest.mark.asyncio
async def test_shutdown_error_is_not_reclassified(caplog):
    cleaned = []
    life = driver(
        make_app(VectorStorageEmptyError(vdb_name="chunks"), cleaned, shutdown=True)
    )
    await life.startup()
    assert not life.should_exit
    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        await life.shutdown()
    assert life.shutdown_failed and life.should_exit
    assert cleaned == [True]
    assert "Traceback" in caplog.text
    assert "Startup blocked" not in caplog.text


@pytest.mark.asyncio
async def test_failure_message_without_exception_is_forwarded(caplog):
    async def app(scope, receive, send):
        await receive()
        await send({"type": "lifespan.startup.failed", "message": "explicit refusal"})

    life = driver(
        StartupErrorMiddleware(app, workspace="space1", vector_storage="test")
    )
    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        await life.startup()
    assert life.startup_failed and life.should_exit
    assert "explicit refusal" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,headline,detail",
    [
        (
            VectorSpaceMismatchError(
                backend="FaissVectorDBStorage",
                container="vdb_chunks.space.json",
                expected_model="new-model",
                expected_dim=1024,
                stored_model="old-model",
                stored_dim=1024,
            ),
            "different embedding space",
            "old-model",
        ),
        (
            EmbeddingBaselineMismatchError(
                workspace="space1",
                mismatches=[
                    {
                        "target": "chunks",
                        "recorded_model": "old-model",
                        "recorded_dim": 1024,
                        "expected_model": "new-model",
                        "expected_dim": 3072,
                    }
                ],
            ),
            "recorded embedding baseline differs",
            "chunks: recorded model 'old-model' dim 1024",
        ),
        (
            CorruptStorageSnapshotError(
                backend="NanoVectorDBStorage",
                container="vdb_entities.json",
                detail="JSONDecodeError: Expecting value",
                artifacts=("vdb_entities.json",),
            ),
            "snapshot is corrupt",
            "JSONDecodeError: Expecting value",
        ),
    ],
    ids=["space-mismatch", "baseline-mismatch", "corrupt-snapshot"],
)
async def test_every_vector_refusal_names_both_recoveries(
    monkeypatch, logs, error, headline, detail
):
    monkeypatch.setattr("sys.stderr.isatty", lambda: False)
    cleaned = []
    life = driver(make_app(error, cleaned))
    await life.startup()
    messages = _lifespan_messages(logs)
    assert life.should_exit and life.startup_failed
    assert cleaned == [True]
    assert "Startup blocked" in messages
    assert headline in messages
    assert detail in messages
    assert "space1" in messages and "FaissVectorDBStorage" in messages
    assert "lightrag-rebuild-vdb" in messages
    assert "lightrag-clear-storage" in messages
    assert "Traceback" not in messages


@pytest.mark.parametrize(
    "error",
    [
        VectorStorageEmptyError(vdb_name="chunks", source="text chunk storage"),
        VectorSpaceMismatchError(
            backend="PGVectorStorage",
            container="lightrag_vdb_chunks_new_model",
            expected_model="new-model",
            stored_model="old-model",
        ),
        EmbeddingBaselineMismatchError(
            workspace="",
            mismatches=[
                {
                    "target": "entities",
                    "recorded_model": "a",
                    "recorded_dim": 8,
                    "expected_model": "b",
                    "expected_dim": 8,
                }
            ],
        ),
        CorruptStorageSnapshotError(
            backend="FaissVectorDBStorage",
            container="vdb_chunks.index",
            detail="truncated",
            artifacts=("vdb_chunks.index",),
        ),
    ],
    ids=["empty", "space-mismatch", "baseline-mismatch", "corrupt-snapshot"],
)
def test_the_exception_text_itself_names_the_clear_tool(error):
    assert "lightrag-rebuild-vdb" in str(error)
    assert "lightrag-clear-storage" in str(error)


@pytest.mark.asyncio
async def test_a_baseline_refusal_reports_every_target_with_its_own_values(
    monkeypatch, logs
):
    monkeypatch.setattr("sys.stderr.isatty", lambda: False)
    error = EmbeddingBaselineMismatchError(
        workspace="space1",
        mismatches=[
            {
                "target": "entities",
                "recorded_model": "model-a",
                "recorded_dim": 768,
                "expected_model": "model-c",
                "expected_dim": 1536,
            },
            {
                "target": "chunks",
                "recorded_model": "model-b",
                "recorded_dim": 3072,
                "expected_model": "model-c",
                "expected_dim": 1536,
            },
        ],
    )
    life = driver(make_app(error, []))
    await life.startup()
    messages = _lifespan_messages(logs)
    assert "Mismatched targets (2)" in messages
    assert "entities: recorded model 'model-a' dim 768" in messages
    assert "chunks: recorded model 'model-b' dim 3072" in messages


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        pytest.param(
            ConfigurationIdentityError(
                "Refusing to start: the anchor binds this deployment to "
                "PGKVStorage. Fix: set LIGHTRAG_CONFIG_STORAGE=PGKVStorage.",
                cause="backend_mismatch",
            ),
            id="identity",
        ),
        pytest.param(
            ConfigurationAnchorLockError(
                "the configuration migration holds the anchor lock"
            ),
            id="anchor-lock",
        ),
        pytest.param(
            WorkingDirectoryInUseError(
                "working directory '/data' is already in use by another "
                "LightRAG process."
            ),
            id="working-dir",
        ),
    ],
)
async def test_a_configuration_refusal_is_concise_and_keeps_its_own_advice(
    monkeypatch, logs, error
):
    monkeypatch.setattr("sys.stderr.isatty", lambda: False)
    cleaned = []
    life = driver(make_app(error, cleaned))
    await life.startup()
    messages = _lifespan_messages(logs)
    assert life.should_exit and life.startup_failed
    assert cleaned == [True]
    assert f"Startup blocked: {error}" in messages
    assert "space1" in messages
    assert "Traceback" not in messages
    assert "lightrag-rebuild-vdb" not in messages
    assert "FaissVectorDBStorage" not in messages


@pytest.mark.asyncio
async def test_a_configuration_refusal_keeps_the_io_error_it_was_raised_from(
    monkeypatch, logs
):
    monkeypatch.setattr("sys.stderr.isatty", lambda: False)
    try:
        try:
            raise PermissionError(13, "Permission denied", "/data/anchor.json")
        except PermissionError as exc:
            raise ConfigurationIdentityError(
                "Refusing to start: the anchor cannot be read.",
                cause="anchor_unreadable",
            ) from exc
    except ConfigurationIdentityError as chained:
        error = chained
    life = driver(make_app(error, []))
    await life.startup()
    messages = _lifespan_messages(logs)
    assert "Cause: PermissionError:" in messages
    assert "Permission denied" in messages
    assert "Traceback" not in messages
