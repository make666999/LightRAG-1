"""Shared helpers for interactive setup tests."""

from __future__ import annotations

from collections.abc import Sequence
import functools
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]
PRESERVED_HEADER = (
    "### ----- Preserved custom environment variables from previous .env  -----"
)
PRESERVED_NOTICE = (
    "### ----- Comments in this session will persist across regenerations -----"
)

FILE_STORAGE_ENV: tuple[str, ...] = (
    "LIGHTRAG_KV_STORAGE=JsonKVStorage",
    "LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage",
    "LIGHTRAG_GRAPH_STORAGE=NetworkXStorage",
    "LIGHTRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage",
)
OPENSEARCH_STORAGE_ENV: tuple[str, ...] = (
    "LIGHTRAG_KV_STORAGE=OpenSearchKVStorage",
    "LIGHTRAG_VECTOR_STORAGE=OpenSearchVectorDBStorage",
    "LIGHTRAG_GRAPH_STORAGE=OpenSearchGraphStorage",
    "LIGHTRAG_DOC_STATUS_STORAGE=OpenSearchDocStatusStorage",
)
MONGO_STORAGE_ENV: tuple[str, ...] = (
    "LIGHTRAG_KV_STORAGE=MongoKVStorage",
    "LIGHTRAG_VECTOR_STORAGE=MongoVectorDBStorage",
    "LIGHTRAG_GRAPH_STORAGE=MongoGraphStorage",
    "LIGHTRAG_DOC_STATUS_STORAGE=MongoDocStatusStorage",
)
PG_STORAGE_ENV: tuple[str, ...] = (
    "LIGHTRAG_KV_STORAGE=PGKVStorage",
    "LIGHTRAG_VECTOR_STORAGE=PGVectorStorage",
    "LIGHTRAG_GRAPH_STORAGE=PGGraphStorage",
    "LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage",
)

DEFAULT_PROMPT_STUBS = """
prompt_choice() { printf '%s' "$2"; }
prompt_with_default() { printf '%s' "$2"; }
prompt_until_valid() { printf '%s' "$2"; }
prompt_secret_with_default() { printf '%s' "$2"; }
prompt_secret_until_valid_with_default() { printf '%s' "$2"; }
""".strip()


class ValidateEnvResult(NamedTuple):
    """Outcome of running ``validate_env_file`` in an isolated bash subprocess."""

    valid: bool
    stdout: str
    stderr: str
    returncode: int


def _bash_version_major(candidate: str) -> int | None:
    result = subprocess.run(
        [candidate, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    first_line = result.stdout.splitlines()[0] if result.stdout else ""
    match = re.search(r"version ([0-9]+)", first_line)
    return int(match.group(1)) if match else None


@functools.cache
def bash_bin() -> str:
    """Path to a bash >= 4 interpreter, resolved once per process.

    ``scripts/setup/setup.sh`` uses associative arrays (``declare -A``), which
    need bash 4+. macOS still ships bash 3.2 as ``/bin/bash``, so a plain
    ``"bash"`` picks an interpreter that cannot source the wizard at all.

    Deliberately lazy (and cached): resolution shells out to ``bash --version``
    once per candidate, and a module-level constant made every test session that
    merely imports this helper pay those forks — including sessions that run no
    setup test. Override with ``LIGHTRAG_TEST_BASH``.
    """
    candidates = [
        os.environ.get("LIGHTRAG_TEST_BASH"),
        "/opt/homebrew/bin/bash",
        "/usr/local/bin/bash",
        shutil.which("bash"),
        "/bin/bash",
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        candidate_path = Path(candidate)
        if candidate_path.is_absolute() and not candidate_path.exists():
            continue
        major = _bash_version_major(candidate)
        if major is not None and major >= 4:
            return candidate
    # No bash 4+ found. Return the best available so the caller fails with the
    # script's own error rather than a confusing FileNotFoundError.
    return shutil.which("bash") or "bash"


def run_bash_process(
    script: str, cwd: Path | None = None, stdin: str | None = ""
) -> subprocess.CompletedProcess[str]:
    """Run a bash snippet and return the completed process."""
    return subprocess.run(
        [bash_bin(), "--norc", "--noprofile", "-c", script],
        cwd=cwd or REPO_ROOT,
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )


def run_bash(script: str, cwd: Path | None = None) -> str:
    """Run a bash snippet and return stdout."""
    result = run_bash_process(script, cwd=cwd)
    if result.returncode != 0:
        raise AssertionError(f"""bash script failed with code {result.returncode}
stdout:
{result.stdout}
stderr:
{result.stderr}""")
    return result.stdout


def parse_lines(output: str) -> dict[str, str]:
    """Parse KEY=value lines into a dictionary."""
    values: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def run_bash_lines(script: str, cwd: Path | None = None) -> dict[str, str]:
    """Run a bash snippet and parse KEY=value lines from stdout."""
    return parse_lines(run_bash(script, cwd=cwd))


def write_text_lines(path: Path, lines: Sequence[str]) -> Path:
    """Write lines to a fixture file with a trailing newline."""
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def copy_repo_template(tmp_path: Path, filename: str) -> Path:
    """Copy a repository template file (e.g. ``env.example``) into ``tmp_path``."""
    target = tmp_path / filename
    target.write_text(
        (REPO_ROOT / filename).read_text(encoding="utf-8"), encoding="utf-8"
    )
    return target


def write_setup_workspace(
    tmp_path: Path,
    *,
    env_lines: Sequence[str] | None = None,
    compose_lines: Sequence[str] | None = None,
    compose_name: str = "docker-compose.final.yml",
    copy_env_example: bool = False,
    copy_compose_template: bool = False,
) -> None:
    """Write common setup fixture files (``.env``, templates, compose) into ``tmp_path``."""
    if env_lines is not None:
        write_text_lines(tmp_path / ".env", env_lines)
    if copy_env_example:
        copy_repo_template(tmp_path, "env.example")
    if copy_compose_template:
        copy_repo_template(tmp_path, "docker-compose.yml")
    if compose_lines is not None:
        write_text_lines(tmp_path / compose_name, compose_lines)


def assert_single_compose_backup(tmp_path: Path, expected_content: str) -> Path:
    """Assert that a single compose backup exists with the expected content."""
    backups = sorted(tmp_path.glob("docker-compose.backup*.yml"))
    assert len(backups) == 1
    assert re.fullmatch("docker-compose\\.backup\\d{8}_\\d{6}\\.yml", backups[0].name)
    assert backups[0].read_text(encoding="utf-8") == expected_content
    return backups[0]


def write_storage_setup_files(
    tmp_path: Path, env_lines: Sequence[str], compose_lines: Sequence[str]
) -> None:
    """Write the minimal env/example/compose fixtures used by storage-flow tests."""
    write_setup_workspace(
        tmp_path,
        env_lines=env_lines,
        compose_lines=compose_lines,
        copy_env_example=True,
    )


def _setup_preamble(
    *,
    tmp_path: Path | None = None,
    reset: bool = True,
    load_env: bool = False,
) -> str:
    lines = [
        "set -euo pipefail",
        f'source "{REPO_ROOT}/scripts/setup/setup.sh"',
    ]
    if tmp_path is not None:
        lines.append(f'REPO_ROOT="{tmp_path}"')
    if reset:
        lines.append("reset_state")
    if load_env:
        lines.append("load_existing_env_if_present")
    return "\n".join(lines)


def run_setup_bash(
    body: str,
    *,
    tmp_path: Path | None = None,
    reset: bool = True,
    load_env: bool = False,
    cwd: Path | None = None,
) -> str:
    """Source ``setup.sh`` with standard boilerplate, run ``body``, and return stdout."""
    preamble = _setup_preamble(tmp_path=tmp_path, reset=reset, load_env=load_env)
    return run_bash(f"{preamble}\n{body}\n", cwd=cwd)


def run_setup_bash_process(
    body: str,
    *,
    tmp_path: Path | None = None,
    reset: bool = True,
    load_env: bool = False,
    cwd: Path | None = None,
    stdin: str | None = "",
) -> subprocess.CompletedProcess[str]:
    """Source ``setup.sh`` with standard boilerplate and return the completed process."""
    preamble = _setup_preamble(tmp_path=tmp_path, reset=reset, load_env=load_env)
    return run_bash_process(f"{preamble}\n{body}\n", cwd=cwd, stdin=stdin)


def run_validate_env(
    tmp_path: Path,
    env_content: str | Sequence[str],
    *,
    env_example_lines: Sequence[str] | None = ("LLM_BINDING=openai",),
) -> ValidateEnvResult:
    """Write ``.env`` in ``tmp_path``, invoke ``validate_env_file``, and return results."""
    if isinstance(env_content, str):
        (tmp_path / ".env").write_text(env_content, encoding="utf-8")
    else:
        write_text_lines(tmp_path / ".env", env_content)
    if env_example_lines is not None:
        write_text_lines(tmp_path / "env.example", env_example_lines)
    result = run_bash_process(
        f"""
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
reset_state
validate_env_file
"""
    )
    return ValidateEnvResult(
        valid=result.returncode == 0,
        stdout=result.stdout,
        stderr=result.stderr,
        returncode=result.returncode,
    )


def run_security_check(
    tmp_path: Path, env_lines: Sequence[str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run ``security_check_env_file`` against ``tmp_path/.env``."""
    if env_lines is not None:
        write_text_lines(tmp_path / ".env", env_lines)
    return run_bash_process(
        f"""
source "{REPO_ROOT}/scripts/setup/setup.sh"
REPO_ROOT="{tmp_path}"
security_check_env_file
"""
    )


def _bash_case_body(answers: dict[str, str]) -> str:
    if not answers:
        return "  printf '%s' \"$2\""
    branches = "\n".join(
        f"    {shlex.quote(prompt)}) printf '%s' {shlex.quote(answer)} ;;"
        for prompt, answer in answers.items()
    )
    return f'  case "$1" in\n{branches}\n    *) printf \'%s\' "$2" ;;\n  esac'


def run_collect_bash(
    body: str,
    *,
    tmp_path: Path | None = None,
    env_lines: Sequence[str] | None = None,
    env_values: dict[str, str] | None = None,
    original_env_values: dict[str, str] | None = None,
    stubs: str = "",
    prompt_answers: dict[str, str] | None = None,
    until_valid_answers: dict[str, str] | None = None,
    secret_answer: str | None = None,
    log_prompts: bool = False,
    log_defaults: bool = False,
    log_secret_prefix: bool = False,
    print_env: Sequence[str] = (),
    print_set: Sequence[str] = (),
    cwd: Path | None = None,
) -> dict[str, str]:
    """Run a ``collect_*`` snippet with optional prompt logging, seeding, and KEY=value output."""
    if env_lines is not None:
        if tmp_path is None:
            raise ValueError("tmp_path is required when env_lines is provided")
        write_text_lines(tmp_path / ".env", env_lines)

    parts = [
        _setup_preamble(
            tmp_path=tmp_path,
            reset=True,
            load_env=env_lines is not None,
        )
    ]

    if log_prompts:
        parts.append(
            'PROMPT_LOG_FILE="$(mktemp)"\n'
            "trap 'rm -f \"$PROMPT_LOG_FILE\"' EXIT\n"
            ': > "$PROMPT_LOG_FILE"'
        )
        log_fmt = "%s[%s]\\n" if log_defaults else "%s\\n"
        log_args = '"$1" "$2"' if log_defaults else '"$1"'
        secret_fmt = f"secret:{log_fmt}" if log_secret_prefix else log_fmt
        secret_reply = (
            f"  printf '%s' {shlex.quote(secret_answer)}"
            if secret_answer is not None
            else "  printf '%s' \"$2\""
        )
        parts.append(
            f"""
prompt_choice() {{
  printf '{log_fmt}' {log_args} >> "$PROMPT_LOG_FILE"
  printf '%s' "$2"
}}
prompt_with_default() {{
  printf '{log_fmt}' {log_args} >> "$PROMPT_LOG_FILE"
{_bash_case_body(prompt_answers or {})}
}}
prompt_until_valid() {{
  printf '{log_fmt}' {log_args} >> "$PROMPT_LOG_FILE"
{_bash_case_body(until_valid_answers or {})}
}}
prompt_secret_with_default() {{
  printf '{secret_fmt}' {log_args} >> "$PROMPT_LOG_FILE"
{secret_reply}
}}
""".strip()
        )
    else:
        if prompt_answers is not None:
            parts.append(
                f"prompt_with_default() {{\n{_bash_case_body(prompt_answers)}\n}}"
            )
        if until_valid_answers is not None:
            parts.append(
                f"prompt_until_valid() {{\n{_bash_case_body(until_valid_answers)}\n}}"
            )
        if secret_answer is not None:
            parts.append(
                f"prompt_secret_with_default() {{ printf '%s' {shlex.quote(secret_answer)}; }}"
            )

    if stubs:
        parts.append(stubs.strip())

    if original_env_values:
        parts.extend(
            f"ORIGINAL_ENV_VALUES[{key}]={shlex.quote(value)}"
            for key, value in original_env_values.items()
        )
    if env_values:
        parts.extend(
            f"ENV_VALUES[{key}]={shlex.quote(value)}"
            for key, value in env_values.items()
        )

    parts.append(body.strip())

    for key in print_env:
        parts.append(f"printf '{key}=%s\\n' \"${{ENV_VALUES[{key}]:-}}\"")
    for key in print_set:
        parts.append(f"printf '{key}_SET=%s\\n' \"${{ENV_VALUES[{key}]+set}}\"")
    if log_prompts:
        parts.append(
            "printf 'PROMPT_LOG=%s\\n' \"$(paste -sd '|' \"$PROMPT_LOG_FILE\")\""
        )

    return run_bash_lines("\n\n".join(parts) + "\n", cwd=cwd)
