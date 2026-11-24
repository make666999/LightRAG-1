"""Offline tests for provider binding options, install guards, runtime target validation,
shared aiohttp retry classification, and bounded CPU thread-pool submission.
"""

from __future__ import annotations

import ast
import asyncio
import gc
import importlib
import pathlib
import subprocess
import sys
import threading
import weakref
from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import pipmaster
import pytest

from lightrag import utils as lr_utils
from lightrag.api.runtime_validation import (
    RuntimeEnvironment,
    validate_runtime_target,
    validate_runtime_target_from_env_file,
)
from lightrag.llm._aiohttp_retry import is_transient_aiohttp_error
from lightrag.llm.binding_options import OllamaLLMOptions, OpenAILLMOptions

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# 1. Binding option value parsing (env, role overrides, CLI)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("raw", "expected"), [(" ON ", True), ("  0\n", False)])
def test_bool_option_ignores_surrounding_whitespace(monkeypatch, raw, expected):
    monkeypatch.setenv("EXTRACT_OLLAMA_LLM_LOW_VRAM", raw)
    options = OllamaLLMOptions.options_dict_for_role(Namespace(), "extract")
    assert options["low_vram"] is expected

    monkeypatch.delenv("OLLAMA_LLM_LOW_VRAM", raising=False)
    parser = ArgumentParser()
    OllamaLLMOptions.add_args(parser)
    args = parser.parse_args(["--ollama-llm-low_vram", raw])
    assert args.ollama_llm_low_vram is expected


@pytest.mark.parametrize(
    ("options_cls", "env_name", "raw", "field_name", "base_value"),
    [
        (OllamaLLMOptions, "EXTRACT_OLLAMA_LLM_LOW_VRAM", "ture", "low_vram", True),
        (
            OpenAILLMOptions,
            "EXTRACT_OPENAI_LLM_MAX_COMPLETION_TOKENS",
            "500.0",
            "max_completion_tokens",
            321,
        ),
        (
            OpenAILLMOptions,
            "EXTRACT_OPENAI_LLM_TEMPERATURE",
            "0.7x",
            "temperature",
            0.3,
        ),
        (
            OpenAILLMOptions,
            "EXTRACT_OPENAI_LLM_EXTRA_BODY",
            "not json",
            "extra_body",
            {"base": True},
        ),
        (OpenAILLMOptions, "EXTRACT_OPENAI_LLM_STOP", '{"a": 1}', "stop", ["END"]),
    ],
    ids=["bool", "int", "float", "json-invalid", "json-wrong-shape"],
)
def test_role_override_that_does_not_parse_keeps_base_with_warning(
    monkeypatch,
    lightrag_log_records,
    options_cls,
    env_name,
    raw,
    field_name,
    base_value,
):
    monkeypatch.setenv(env_name, raw)
    args = Namespace(**{f"{options_cls._binding_name}_{field_name}": base_value})

    options = options_cls.options_dict_for_role(args, "extract")

    assert options[field_name] == base_value
    warnings = [
        r.getMessage() for r in lightrag_log_records if env_name in r.getMessage()
    ]
    assert len(warnings) == 1
    assert "base option or provider default" in warnings[0]


def test_bool_option_typo_handling(monkeypatch, capsys, lightrag_log_records):
    """CLI rejects typos with SystemExit; env typos log a warning and keep the default."""
    monkeypatch.delenv("OLLAMA_LLM_LOW_VRAM", raising=False)
    parser = ArgumentParser()
    OllamaLLMOptions.add_args(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--ollama-llm-low_vram", "ture"])
    assert "OLLAMA_LLM_LOW_VRAM" in capsys.readouterr().err

    monkeypatch.setenv("OLLAMA_LLM_LOW_VRAM", "ture")
    env_parser = ArgumentParser()
    OllamaLLMOptions.add_args(env_parser)
    args = env_parser.parse_args([])
    assert not hasattr(args, "ollama_llm_low_vram")
    assert any("OLLAMA_LLM_LOW_VRAM" in r.getMessage() for r in lightrag_log_records)


# ---------------------------------------------------------------------------
# 2. Provider pipmaster install guards
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_LLM_PACKAGE = _REPO_ROOT / "lightrag" / "llm"
_DISTRIBUTION_IMPORT_NAMES = {
    "google-api-core": "google.api_core",
    "google-genai": "google.genai",
}


def _distribution_import_name(package_spec: str) -> str:
    distribution = package_spec.partition("[")[0]
    return _DISTRIBUTION_IMPORT_NAMES.get(distribution, distribution.replace("-", "_"))


def _installed_packages(tree: ast.AST) -> list[tuple[str, str, int]]:
    installs: list[tuple[str, str, int]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "install"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            package_spec = node.args[0].value
            installs.append(
                (package_spec, _distribution_import_name(package_spec), node.lineno)
            )
    return installs


def _first_import_line(tree: ast.AST, package: str) -> int | None:
    first: int | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.module and not node.level:
                names = [node.module]
                names.extend(
                    f"{node.module}.{alias.name}"
                    for alias in node.names
                    if alias.name != "*"
                )
            else:
                names = []
        else:
            continue
        if any(
            name == package or name.startswith(f"{package}.") for name in names
        ) and (first is None or node.lineno < first):
            first = node.lineno
    return first


def _provider_modules() -> list[pathlib.Path]:
    return sorted(p for p in _LLM_PACKAGE.glob("*.py") if p.name != "__init__.py")


def test_install_guard_scanner_helpers():
    """Verify package discovery, distribution-to-import resolution, and namespace AST scans."""
    modules = _provider_modules()
    assert len(modules) >= 5, f"only found {len(modules)} modules in {_LLM_PACKAGE}"

    for package_spec, import_name in [
        ("llama-index", "llama_index"),
        ("lmdeploy[all]", "lmdeploy"),
        ("google-genai", "google.genai"),
        ("google-api-core", "google.api_core"),
    ]:
        assert _distribution_import_name(package_spec) == import_name

    assert _first_import_line(ast.parse("from google import genai\n"), "google.genai") == 1
    assert _installed_packages(
        ast.parse('pm.install("google-genai")\npm.install("google-api-core")\n')
    ) == [
        ("google-genai", "google.genai", 1),
        ("google-api-core", "google.api_core", 2),
    ]


def test_every_install_guard_precedes_the_import_it_protects():
    offenders: list[str] = []
    guarded: list[str] = []
    for path in _provider_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for package_spec, package, install_line in _installed_packages(tree):
            import_line = _first_import_line(tree, package)
            if import_line is None:
                continue
            rel = path.relative_to(_REPO_ROOT)
            guarded.append(f"{rel}:{package}")
            if import_line < install_line:
                offenders.append(
                    f"{rel} imports {package} at line {import_line}, but "
                    f"pm.install({package_spec!r}) is at line {install_line} — the "
                    f"guard never runs"
                )

    assert guarded, "no pipmaster install guards found in lightrag/llm/"
    assert not offenders, "install guard placed after its import:\n" + "\n".join(
        offenders
    )


def test_an_install_guard_run_under_pytest_never_reaches_pip(tmp_path, monkeypatch):
    """Session-level ``tests/conftest.py`` refuses ``pipmaster.install`` under pytest."""
    probe = "_lightrag_install_guard_probe"
    absent = "lightrag-absent-sdk-probe"
    (tmp_path / f"{probe}.py").write_text(
        "import pipmaster as pm\n"
        "\n"
        f"if not pm.is_installed({absent!r}):\n"
        f"    pm.install({absent!r})\n",
        encoding="utf-8",
    )

    attempted: list[str] = []
    session_install = pipmaster.install

    def _record_install(package, *args, **kwargs):
        attempted.append(package)
        return session_install(package, *args, **kwargs)

    ran: list[object] = []
    real_run = subprocess.run

    def _record_run(command, *args, **kwargs):
        parts = command if isinstance(command, (list, tuple)) else str(command).split()
        if "install" not in map(str, parts):
            return real_run(command, *args, **kwargs)
        ran.append(command)
        return subprocess.CompletedProcess(command, returncode=1)

    monkeypatch.setattr(pipmaster, "install", _record_install)
    monkeypatch.setattr(subprocess, "run", _record_run)
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        importlib.import_module(probe)
    finally:
        sys.modules.pop(probe, None)

    assert attempted == [absent]
    assert ran == [], f"an install guard ran pip install under pytest: {ran}"


# ---------------------------------------------------------------------------
# 3. Runtime target validation
# ---------------------------------------------------------------------------

_HOST_ENV = RuntimeEnvironment(in_container=False, in_docker=False, in_kubernetes=False)
_DOCKER_ENV = RuntimeEnvironment(in_container=True, in_docker=True, in_kubernetes=False)
_K8S_ENV = RuntimeEnvironment(in_container=True, in_docker=False, in_kubernetes=True)


@pytest.mark.parametrize(
    ("target", "env"),
    [
        pytest.param(None, None, id="undeclared"),
        pytest.param("host", _HOST_ENV, id="host-on-host"),
        pytest.param("compose", _K8S_ENV, id="compose-in-k8s"),
        pytest.param("docker", _K8S_ENV, id="docker-in-k8s"),
    ],
)
def test_validate_runtime_target_accepts_valid_combinations(target, env) -> None:
    is_valid, error_message = (
        validate_runtime_target(target)
        if env is None
        else validate_runtime_target(target, env)
    )
    assert is_valid is True
    assert error_message is None


@pytest.mark.parametrize(
    ("target", "env", "expected_substrings"),
    [
        pytest.param(
            "host",
            _DOCKER_ENV,
            ["LIGHTRAG_RUNTIME_TARGET=host", "This value from .env", "Docker"],
            id="host-in-docker",
        ),
        pytest.param(
            "docker",
            _HOST_ENV,
            [
                "LIGHTRAG_RUNTIME_TARGET=docker",
                "This value from .env",
                "Docker or Kubernetes",
            ],
            id="docker-on-host",
        ),
        pytest.param(
            "invalid",
            _HOST_ENV,
            ["must be 'host' or 'compose'"],
            id="invalid-value",
        ),
    ],
)
def test_validate_runtime_target_rejects_mismatches(
    target, env, expected_substrings
) -> None:
    is_valid, error_message = validate_runtime_target(target, env)
    assert is_valid is False
    assert "\n" in error_message
    assert "Configuration error in .env" in error_message
    for substring in expected_substrings:
        assert substring in error_message


def test_validate_runtime_target_from_env_file_uses_raw_env_value(
    tmp_path: pathlib.Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("LIGHTRAG_RUNTIME_TARGET=compose\n", encoding="utf-8")

    is_valid, error_message = validate_runtime_target_from_env_file(
        env_file, _DOCKER_ENV
    )
    assert is_valid is True
    assert error_message is None


# ---------------------------------------------------------------------------
# 4. Shared aiohttp retry classification
# ---------------------------------------------------------------------------

_URL = "http://provider.test/v1"


def _status_error(status: int) -> aiohttp.ClientResponseError:
    request_info = aiohttp.RequestInfo(
        url=_URL, method="POST", headers={}, real_url=_URL
    )
    return aiohttp.ClientResponseError(request_info, (), status=status)


@pytest.mark.parametrize(
    ("status", "transient"),
    [
        (408, True),
        (409, True),
        (429, True),
        (500, True),
        (401, False),
        (499, False),
    ],
)
def test_aiohttp_status_classification(status, transient):
    assert is_transient_aiohttp_error(_status_error(status)) is transient


@pytest.mark.parametrize(
    ("exc", "transient"),
    [
        pytest.param(aiohttp.ClientConnectionError("reset"), True, id="connection"),
        pytest.param(aiohttp.ClientPayloadError("truncated"), True, id="payload"),
        pytest.param(asyncio.TimeoutError(), True, id="timeout"),
        pytest.param(aiohttp.InvalidURL("not-a-url"), False, id="invalid-url"),
        pytest.param(asyncio.CancelledError(), False, id="cancelled"),
        pytest.param(ValueError("bug"), False, id="non-aiohttp"),
    ],
)
def test_aiohttp_exception_classification(exc, transient):
    assert is_transient_aiohttp_error(exc) is transient


# ---------------------------------------------------------------------------
# 5. Bounded CPU thread-pool submission & per-loop semaphores
# ---------------------------------------------------------------------------


class _CountingExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor that records how many submissions it accepted."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submitted = 0

    def submit(self, fn, /, *args, **kwargs):
        self.submitted += 1
        return super().submit(fn, *args, **kwargs)


def _blocking(release: threading.Event, marker: str) -> str:
    release.wait(5.0)
    return marker


def test_bounded_submit_basics():
    """Returns results without blocking the loop, forwards kwargs, and propagates errors."""

    async def _main():
        executor = _CountingExecutor(max_workers=1)
        try:
            beats = 0

            async def _heartbeat():
                nonlocal beats
                while True:
                    beats += 1
                    await asyncio.sleep(0.005)

            pulse = asyncio.create_task(_heartbeat())
            semaphore = lr_utils.get_loop_semaphore("test-basic", 4)
            result = await lr_utils.bounded_submit(
                executor, semaphore, lambda: (threading.Event().wait(0.1), "done")[1]
            )
            pulse.cancel()
            assert result == "done"
            assert beats > 1

            summed = await lr_utils.bounded_submit(
                executor, semaphore, lambda a, b=0: a + b, 1, b=41
            )
            assert summed == 42

            def _boom():
                raise ValueError("from the thread")

            with pytest.raises(ValueError, match="from the thread"):
                await lr_utils.bounded_submit(executor, semaphore, _boom)
        finally:
            executor.shutdown(wait=True)

    asyncio.run(_main())


def test_cancelling_a_submission_does_not_return_the_permit_early():
    """Permit ownership stays on the executor future even if the awaiting task cancels."""

    async def _main():
        executor = _CountingExecutor(max_workers=1)
        release = threading.Event()
        try:
            semaphore = lr_utils.get_loop_semaphore("test-cancel", 1)

            first = asyncio.create_task(
                lr_utils.bounded_submit(executor, semaphore, _blocking, release, "one")
            )
            while executor.submitted < 1:
                await asyncio.sleep(0.005)

            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first

            follow_ups = [
                asyncio.create_task(
                    lr_utils.bounded_submit(executor, semaphore, lambda: "later")
                )
                for _ in range(3)
            ]
            await asyncio.sleep(0.1)

            assert executor.submitted == 1
            assert not any(task.done() for task in follow_ups)

            release.set()
            assert (
                await asyncio.wait_for(asyncio.gather(*follow_ups), timeout=5.0)
                == ["later"] * 3
            )
            assert executor.submitted == 4
        finally:
            release.set()
            executor.shutdown(wait=True)

    asyncio.run(_main())


def test_saturation_is_backpressure_not_refusal():
    async def _main():
        executor = _CountingExecutor(max_workers=1)
        release = threading.Event()
        try:
            semaphore = lr_utils.get_loop_semaphore("test-backpressure", 2)
            tasks = [
                asyncio.create_task(
                    lr_utils.bounded_submit(
                        executor, semaphore, _blocking, release, f"m{i}"
                    )
                )
                for i in range(5)
            ]
            await asyncio.sleep(0.05)
            assert executor.submitted == 2

            release.set()
            results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)
            assert results == [f"m{i}" for i in range(5)]
        finally:
            release.set()
            executor.shutdown(wait=True)

    asyncio.run(_main())


def test_per_loop_semaphore_identity_and_cross_loop_isolation():
    """Repeated lookups in one loop return the same semaphore; successive loops stay isolated."""

    async def _main():
        first = lr_utils.get_loop_semaphore("test-identity", 3)
        second = lr_utils.get_loop_semaphore("test-identity", 99)
        assert first is second
        assert second._value == 3

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            semaphore = lr_utils.get_loop_semaphore("test-crossloop", 2)
            return await lr_utils.bounded_submit(executor, semaphore, lambda: "ok")
        finally:
            executor.shutdown(wait=True)

    assert asyncio.run(_main()) == "ok"
    assert asyncio.run(_main()) == "ok"


def test_a_contended_semaphore_does_not_retain_its_closed_loop():
    """Even after contention populates ``Semaphore._loop``, the closed loop can be GC'd."""
    captured: dict[str, object] = {}

    async def _main():
        loop = asyncio.get_running_loop()
        captured["ref"] = weakref.ref(loop)
        executor = ThreadPoolExecutor(max_workers=1)
        release = threading.Event()
        try:
            semaphore = lr_utils.get_loop_semaphore("test-gc", 1)
            tasks = [
                asyncio.create_task(
                    lr_utils.bounded_submit(
                        executor, semaphore, _blocking, release, str(i)
                    )
                )
                for i in range(2)
            ]
            await asyncio.sleep(0.05)
            release.set()
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)
            assert semaphore._loop is loop
        finally:
            release.set()
            executor.shutdown(wait=True)

    asyncio.run(_main())
    gc.collect()
    gc.collect()

    assert captured["ref"]() is None


class _FakeLoop:
    """Minimal stand-in exposing only what the fallback semaphore table needs."""

    def __init__(self):
        self._closed = False

    def is_closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        self._closed = True


class TestBoundedSubmitFallbackTable:
    """Fallback table tests for event loops that reject attribute assignment."""

    @pytest.fixture(autouse=True)
    def _clear_fallback_table(self):
        lr_utils._LOOP_SEMAPHORE_FALLBACK.clear()
        yield
        lr_utils._LOOP_SEMAPHORE_FALLBACK.clear()

    def test_fallback_table_reuse_sweep_and_id_guard(self):
        loop = _FakeLoop()
        first = lr_utils._fallback_semaphore(loop, "a", 2)
        assert lr_utils._fallback_semaphore(loop, "a", 2) is first
        assert lr_utils._fallback_semaphore(loop, "b", 2) is not first
        assert len(lr_utils._LOOP_SEMAPHORE_FALLBACK) == 1

        # Sweeps closed loops without needing contention.
        loop.close()
        survivor = _FakeLoop()
        lr_utils._fallback_semaphore(survivor, "a", 2)
        assert list(lr_utils._LOOP_SEMAPHORE_FALLBACK) == [id(survivor)]

        # Rejects stale entry when an id() is reused by a different loop object.
        stale = lr_utils._fallback_semaphore(survivor, "a", 2)
        impostor = _FakeLoop()
        lr_utils._LOOP_SEMAPHORE_FALLBACK[id(impostor)] = (survivor, {"a": stale})
        assert lr_utils._fallback_semaphore(impostor, "a", 2) is not stale
