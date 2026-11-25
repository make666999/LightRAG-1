"""Gunicorn master anchor checks, configuration directory claims, and worker orphan exit."""

from __future__ import annotations

import importlib
import os
import signal
import sys
from types import SimpleNamespace

import pytest

from lightrag import config_anchor as ca
from lightrag.config_shards import json_config_dir
from lightrag.exceptions import ConfigurationIdentityError

if sys.platform != "win32":
    from uvicorn_worker import UvicornWorker

    from lightrag.api.gunicorn_worker import LightRAGUvicornWorker
else:  # pragma: no cover - Windows lacks fcntl needed by uvicorn_worker
    UvicornWorker = None
    LightRAGUvicornWorker = None

pytestmark = pytest.mark.offline


@pytest.fixture
def gunicorn_config(monkeypatch):
    # The module parses argv at import time; pytest's own would fail it.
    monkeypatch.setattr(sys, "argv", ["lightrag-gunicorn"])
    # ``resolved_workspace`` imports ``lightrag.api.config``, whose import
    # runs ``load_dotenv(override=False)``: imported lazily there, it would
    # refill the variables ``_env`` removed from a local ``.env``.
    importlib.import_module("lightrag.api.config")
    module = importlib.import_module("lightrag.api.gunicorn_config")
    monkeypatch.setattr(module, "workers", 4, raising=False)
    monkeypatch.setattr(module, "working_dir", None, raising=False)
    monkeypatch.setattr(module, "workspace", None, raising=False)
    return module


@pytest.fixture(autouse=True)
def _clean_locks():
    from lightrag.kg import anchor_lock as al
    from lightrag.kg import working_dir_lock as wdl

    claims, holds = dict(wdl._claims), dict(al._shared_holds)
    wdl._claims.clear()
    al._shared_holds.clear()
    yield
    for claim in wdl._claims.values():
        try:
            claim.handle.close()
        except OSError:  # pragma: no cover - defensive
            pass
    for hold in al._shared_holds.values():
        if hold.handle is not None:
            hold.handle.close()
    wdl._claims.clear()
    al._shared_holds.clear()
    wdl._claims.update(claims)
    al._shared_holds.update(holds)


def _env(monkeypatch, tmp_path, **overrides):
    monkeypatch.setenv("WORKING_DIR", str(tmp_path))
    for name in ("LIGHTRAG_CONFIG_STORAGE", "LIGHTRAG_CONFIG_DIR", "WORKSPACE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LIGHTRAG_KV_STORAGE", "JsonKVStorage")
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)


def _anchor(tmp_path, backend):
    os.makedirs(ca.anchor_dir(str(tmp_path)), exist_ok=True)
    ca.publish_anchor(
        str(tmp_path),
        ca.StorageAnchor(
            backend=backend,
            storage_uuid=ca.new_storage_uuid(),
            members=() if backend == "JsonKVStorage" else None,
        ),
        replace=False,
    )


# --------------------------------------------------------------------------- #
# Master anchor checks (steps 0a-0c before forking)
# --------------------------------------------------------------------------- #


def test_the_master_refuses_a_type_mismatch_before_forking(
    gunicorn_config, monkeypatch, tmp_path, capsys
):
    """First upgrade bound PG; later only KV was changed to Mongo with CONFIG
    unset -- the candidate moved, and the master refuses."""
    from lightrag.kg.anchor_lock import holds_anchor_lock
    from lightrag.kg.working_dir_lock import holds_working_dir_lock

    _anchor(tmp_path, "JsonKVStorage")
    _env(monkeypatch, tmp_path, LIGHTRAG_KV_STORAGE="MongoKVStorage")
    with pytest.raises(ConfigurationIdentityError) as excinfo:
        gunicorn_config.on_starting(object())
    capsys.readouterr()
    assert excinfo.value.cause == ca.IDENTITY_BACKEND_MISMATCH
    assert "forking workers" not in capsys.readouterr().out
    assert holds_anchor_lock(str(tmp_path)) is False
    assert holds_working_dir_lock(str(tmp_path)) is False


def test_an_explicit_selection_of_the_anchored_backend_passes(
    gunicorn_config, monkeypatch, tmp_path, capsys
):
    """The documented fix: keep KV on Mongo, name the anchored configuration
    backend explicitly. No migration needed."""
    from lightrag.kg.anchor_lock import holds_anchor_lock

    _anchor(tmp_path, "JsonKVStorage")
    _env(
        monkeypatch,
        tmp_path,
        LIGHTRAG_KV_STORAGE="MongoKVStorage",
        LIGHTRAG_CONFIG_STORAGE="JsonKVStorage",
    )
    gunicorn_config.on_starting(object())
    capsys.readouterr()
    assert holds_anchor_lock(str(tmp_path)) is True
    gunicorn_config.on_exit(object())
    capsys.readouterr()
    assert holds_anchor_lock(str(tmp_path)) is False


def test_no_anchor_takes_the_lock_for_the_workers_and_writes_nothing(
    gunicorn_config, monkeypatch, tmp_path, capsys
):
    """The master only checks: binding is each worker's step 1b."""
    from lightrag.kg.anchor_lock import holds_anchor_lock

    _env(monkeypatch, tmp_path)
    gunicorn_config.on_starting(object())
    capsys.readouterr()
    assert holds_anchor_lock(str(tmp_path)) is True
    assert ca.read_anchor(str(tmp_path)) is None
    gunicorn_config.on_exit(object())
    capsys.readouterr()
    assert holds_anchor_lock(str(tmp_path)) is False


def test_an_unreadable_anchor_refuses_the_master(
    gunicorn_config, monkeypatch, tmp_path, capsys
):
    path = ca.anchor_path(str(tmp_path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("{")
    _env(monkeypatch, tmp_path)
    with pytest.raises(ConfigurationIdentityError):
        gunicorn_config.on_starting(object())
    capsys.readouterr()


def test_the_master_follows_the_parsed_working_dir(
    gunicorn_config, monkeypatch, tmp_path, capsys
):
    """``--working-dir`` is never written back to the environment, so the
    master must read the anchor where the workers will."""
    parsed = tmp_path / "parsed"
    _anchor(parsed, "PGKVStorage")
    _env(monkeypatch, tmp_path / "env")
    monkeypatch.setattr(gunicorn_config, "working_dir", str(parsed))
    with pytest.raises(ConfigurationIdentityError):
        gunicorn_config.on_starting(object())
    capsys.readouterr()


# --------------------------------------------------------------------------- #
# Master configuration directory claim (shared across fork)
# --------------------------------------------------------------------------- #


def test_the_master_claims_the_empty_workspace_snapshot_dir(
    gunicorn_config, monkeypatch, tmp_path, capsys
):
    """The empty workspace keeps its snapshot directly under WORKING_DIR."""
    from lightrag.kg.working_dir_lock import holds_working_dir_lock

    _env(monkeypatch, tmp_path)
    gunicorn_config.on_starting(object())
    capsys.readouterr()

    assert json_config_dir(str(tmp_path), "") == str(tmp_path)
    assert holds_working_dir_lock(str(tmp_path)) is True
    assert holds_working_dir_lock(str(tmp_path / "_lightrag_config")) is False


def test_the_master_claims_the_parsed_workspace_snapshot_dir(
    gunicorn_config, monkeypatch, tmp_path, capsys
):
    """``--workspace`` is parsed into ``global_args`` and handed to the
    workers; ``run_with_gunicorn`` copies it to the module, and the master
    claims THAT workspace's directory, not the one the environment names."""
    from lightrag.kg.working_dir_lock import holds_working_dir_lock

    _env(monkeypatch, tmp_path, WORKSPACE="from_env")
    monkeypatch.setattr(gunicorn_config, "workspace", "tenant")

    assert gunicorn_config.resolved_workspace() == "tenant"
    gunicorn_config.on_starting(object())
    capsys.readouterr()

    assert holds_working_dir_lock(json_config_dir(str(tmp_path), "tenant")) is True
    assert holds_working_dir_lock(str(tmp_path / "from_env")) is False
    assert holds_working_dir_lock(str(tmp_path)) is False


def test_the_environment_workspace_is_normalized_like_the_server(
    gunicorn_config, monkeypatch, tmp_path, capsys
):
    """A master started without ``run_with_gunicorn`` falls back to
    ``WORKSPACE``, normalized by the server's own rule -- a raw value would
    name a directory no worker asks for."""
    from lightrag.api.config import normalize_server_workspace
    from lightrag.kg.working_dir_lock import holds_working_dir_lock

    _env(monkeypatch, tmp_path, WORKSPACE="team-a.prod")
    expected = normalize_server_workspace("team-a.prod")
    assert expected == "team_a_prod"
    assert gunicorn_config.resolved_workspace() == expected

    gunicorn_config.on_starting(object())
    capsys.readouterr()

    assert holds_working_dir_lock(json_config_dir(str(tmp_path), expected)) is True
    assert holds_working_dir_lock(str(tmp_path / "team-a.prod")) is False


def test_the_removed_config_dir_setting_does_not_move_the_claim(
    gunicorn_config, monkeypatch, tmp_path, capsys
):
    """``LIGHTRAG_CONFIG_DIR`` is gone: the snapshot path is derived, never
    chosen, so a leftover setting must not redirect the master's claim."""
    from lightrag.kg.working_dir_lock import holds_working_dir_lock

    elsewhere = tmp_path / "conf"
    _env(monkeypatch, tmp_path, LIGHTRAG_CONFIG_DIR=str(elsewhere))
    gunicorn_config.on_starting(object())
    capsys.readouterr()

    assert holds_working_dir_lock(str(tmp_path)) is True
    assert holds_working_dir_lock(str(elsewhere)) is False


def test_a_server_backed_configuration_claims_nothing(
    gunicorn_config, monkeypatch, tmp_path, capsys
):
    """A file-backed BUSINESS store no longer drags the claim in with it: the
    configuration storage is what the claim is for."""
    from lightrag.kg.working_dir_lock import holds_working_dir_lock

    _env(
        monkeypatch,
        tmp_path,
        LIGHTRAG_CONFIG_STORAGE="PGKVStorage",
        WORKSPACE="tenant",
    )
    gunicorn_config.on_starting(object())
    capsys.readouterr()

    assert holds_working_dir_lock(str(tmp_path / "tenant")) is False
    assert holds_working_dir_lock(str(tmp_path)) is False


def test_a_reserved_workspace_name_is_refused_in_the_master(
    gunicorn_config, monkeypatch, tmp_path, capsys
):
    """A workspace named after a deployment-wide root file would collide with
    it; the master refuses before claiming anything."""
    from lightrag.config_shards import RESERVED_WORKSPACE_NAMES
    from lightrag.kg.working_dir_lock import holds_working_dir_lock

    reserved = sorted(RESERVED_WORKSPACE_NAMES)[0]
    _env(monkeypatch, tmp_path)
    monkeypatch.setattr(gunicorn_config, "workspace", reserved)

    with pytest.raises(ValueError, match="reserved|deployment-wide"):
        gunicorn_config.on_starting(object())
    capsys.readouterr()
    assert holds_working_dir_lock(str(tmp_path)) is False
    assert holds_working_dir_lock(str(tmp_path / reserved)) is False


def test_the_master_and_a_worker_agree_on_the_directory(
    gunicorn_config, monkeypatch, tmp_path, capsys
):
    """The property the whole thing rests on, asserted directly rather than
    inferred: what the master claims is what a ``LightRAG`` instance in a
    worker would ask for -- for the workspace the worker serves."""
    import numpy as np

    from lightrag import LightRAG
    from lightrag.config_store import configuration_selection_from_env
    from lightrag.kg.shared_storage import (
        finalize_share_data,
        initialize_share_data,
    )
    from lightrag.utils import EmbeddingFunc, Tokenizer, TokenizerInterface

    class _StubTokenizer(TokenizerInterface):
        def encode(self, content: str) -> list[int]:
            return [ord(c) for c in content]

        def decode(self, tokens: list[int]) -> str:
            return "".join(chr(t) for t in tokens)

    async def _llm(prompt, **kwargs):  # pragma: no cover - never called
        return ""

    async def _embed(texts, **kwargs):  # pragma: no cover - never called
        return np.zeros((len(texts), 8), dtype=np.float32)

    _env(monkeypatch, tmp_path, WORKSPACE="tenant")

    master_storage, master_dir = configuration_selection_from_env(
        kv_storage="JsonKVStorage",
        working_dir=gunicorn_config.resolved_working_dir(),
        workspace=gunicorn_config.resolved_workspace(),
    )
    assert master_dir == json_config_dir(str(tmp_path), "tenant")

    initialize_share_data(workers=1)
    try:
        rag = LightRAG(
            working_dir=str(tmp_path),
            workspace="tenant",
            llm_model_func=_llm,
            embedding_func=EmbeddingFunc(
                embedding_dim=8, max_token_size=1024, func=_embed, model_name="m"
            ),
            tokenizer=Tokenizer("stub", _StubTokenizer()),
        )
        assert rag.config_dir == master_dir
        assert rag.config_storage == master_storage
    finally:
        finalize_share_data()
    capsys.readouterr()


def test_an_unusable_selection_is_refused_in_the_master(
    gunicorn_config, monkeypatch, tmp_path, capsys
):
    """Redis is not in the category. Refusing in the master fails the server
    once, clearly, instead of once per worker."""
    _env(monkeypatch, tmp_path, LIGHTRAG_CONFIG_STORAGE="RedisKVStorage")

    with pytest.raises(ValueError, match="RedisKVStorage"):
        gunicorn_config.on_starting(object())
    capsys.readouterr()


def test_on_exit_releases_what_on_starting_claimed(
    gunicorn_config, monkeypatch, tmp_path, capsys
):
    from lightrag.kg.working_dir_lock import holds_working_dir_lock

    claimed = json_config_dir(str(tmp_path), "tenant")
    _env(monkeypatch, tmp_path)
    monkeypatch.setattr(gunicorn_config, "workspace", "tenant")

    gunicorn_config.on_starting(object())
    assert holds_working_dir_lock(claimed) is True
    gunicorn_config.on_exit(object())
    capsys.readouterr()

    assert holds_working_dir_lock(claimed) is False


def test_the_master_follows_the_parsed_working_dir_over_the_environment(
    gunicorn_config, monkeypatch, tmp_path, capsys
):
    """``--working-dir`` is parsed into ``global_args`` and handed to every
    worker's ``LightRAG``; nothing writes it back to ``WORKING_DIR``."""
    from lightrag.kg.working_dir_lock import holds_working_dir_lock

    parsed = tmp_path / "from-cli"
    parsed.mkdir()
    environment = tmp_path / "from-env"
    environment.mkdir()

    _env(monkeypatch, environment)
    monkeypatch.setattr(gunicorn_config, "working_dir", str(parsed), raising=False)
    monkeypatch.setattr(gunicorn_config, "workspace", "tenant")

    assert gunicorn_config.resolved_working_dir() == str(parsed)

    gunicorn_config.on_starting(object())
    capsys.readouterr()

    assert holds_working_dir_lock(json_config_dir(str(parsed), "tenant")) is True
    assert holds_working_dir_lock(json_config_dir(str(environment), "tenant")) is False


def test_the_environment_is_the_fallback_when_nothing_parsed_a_directory(
    gunicorn_config, monkeypatch, tmp_path
):
    """A master started some other way than ``lightrag-gunicorn`` sets no
    module attribute, and the environment is all there is."""
    _env(monkeypatch, tmp_path)
    monkeypatch.setattr(gunicorn_config, "working_dir", None, raising=False)

    assert gunicorn_config.resolved_working_dir() == str(tmp_path)


@pytest.mark.parametrize("workspace", ["", "tenant"])
def test_redis_default_claims_json_configuration_directory(
    gunicorn_config, monkeypatch, tmp_path, capsys, workspace
):
    """Redis business KV defaults the configuration storage to JSON, so the
    master claims that workspace's snapshot directory."""
    from lightrag.kg.working_dir_lock import holds_working_dir_lock

    _env(monkeypatch, tmp_path, LIGHTRAG_KV_STORAGE="RedisKVStorage")
    monkeypatch.setattr(gunicorn_config, "workspace", workspace)
    gunicorn_config.on_starting(object())
    capsys.readouterr()
    assert holds_working_dir_lock(json_config_dir(str(tmp_path), workspace)) is True


# --------------------------------------------------------------------------- #
# Worker orphan exit (POSIX-only)
# --------------------------------------------------------------------------- #


class _Log:
    def __init__(self):
        self.messages: list[str] = []

    def info(self, message, *args):
        self.messages.append(message % args if args else message)


def _worker(ppid: int):
    """A worker with only the attributes the check touches."""
    worker = LightRAGUvicornWorker.__new__(LightRAGUvicornWorker)
    worker.ppid = ppid
    worker.log = _Log()
    return worker


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="uvicorn_worker imports fcntl; the orphan-exit regression is POSIX-only",
)
class TestGunicornWorkerOrphanExit:
    """A worker whose gunicorn master died must stop serving."""

    def test_gunicorn_graceful_timeout_is_forwarded_to_uvicorn(self, monkeypatch):
        """Without this mapping uvicorn waits forever for active ASGI requests after
        the orphan-triggered SIGTERM because its shutdown timeout defaults to None."""

        def fake_init(worker, *args, **kwargs):
            worker.cfg = SimpleNamespace(graceful_timeout=47)
            worker.config = SimpleNamespace(timeout_graceful_shutdown=None)

        monkeypatch.setattr(UvicornWorker, "__init__", fake_init)

        worker = LightRAGUvicornWorker()

        assert worker.config.timeout_graceful_shutdown == 47

    def test_an_orphaned_worker_requests_a_graceful_shutdown(self, monkeypatch):
        """Fix-proof: nothing used to notice, and the worker kept serving."""
        signals: list[tuple[int, int]] = []
        monkeypatch.setattr(os, "kill", lambda pid, sig: signals.append((pid, sig)))

        worker = _worker(ppid=os.getppid() + 1000)
        assert worker.exit_if_orphaned() is True

        assert signals == [(os.getpid(), signal.SIGTERM)]
        assert any("Parent changed" in message for message in worker.log.messages)

    def test_a_live_master_is_left_alone(self, monkeypatch):
        """The check must be silent on every ordinary tick."""
        signals: list[tuple[int, int]] = []
        monkeypatch.setattr(os, "kill", lambda pid, sig: signals.append((pid, sig)))

        worker = _worker(ppid=os.getppid())
        assert worker.exit_if_orphaned() is False
        assert signals == []
        assert worker.log.messages == []

    @pytest.mark.asyncio
    async def test_the_check_rides_the_notify_tick_and_still_notifies(
        self, monkeypatch
    ):
        """It must hang off uvicorn's existing notify callback AND keep notifying."""
        notified: list[bool] = []
        monkeypatch.setattr(
            LightRAGUvicornWorker, "notify", lambda self: notified.append(True)
        )
        signals: list[tuple[int, int]] = []
        monkeypatch.setattr(os, "kill", lambda pid, sig: signals.append((pid, sig)))

        worker = _worker(ppid=os.getppid() + 1000)
        await worker.callback_notify()

        assert notified == [True]
        assert signals == [(os.getpid(), signal.SIGTERM)]

    def test_the_configured_worker_class_resolves(self, monkeypatch):
        """The class is wired in by dotted path, so a rename surfaces here."""
        from gunicorn.util import load_class

        monkeypatch.setattr("sys.argv", ["pytest"])
        from lightrag.api import gunicorn_config

        assert load_class(gunicorn_config.worker_class) is LightRAGUvicornWorker, (
            gunicorn_config.worker_class
        )
