# Regression tests for interactive setup wizard.
# Classification: keep tests here when they target collect_* prompt/normalization logic for one config area before env/compose files are finalized.

from __future__ import annotations

import shlex
from pathlib import Path

import pytest

from tests.setup._helpers import (
    REPO_ROOT,
    copy_repo_template,
    parse_lines,
    run_bash_process,
    run_collect_bash,
    run_setup_bash,
    write_text_lines,
)

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    ("invocation", "stdin", "expected"),
    [
        ('prompt_until_valid "Port" "9621" validate_port', "invalid\n9622\n", "9622"),
        (
            'prompt_secret_until_valid "Key" validate_non_empty',
            "\nvalid-key\n",
            "valid-key",
        ),
        (
            'prompt_secret_until_valid_with_default "Key" "" validate_non_empty',
            "\nvalid-key\n",
            "valid-key",
        ),
        ('prompt_required_secret "Key"', "\nvalid-key\n", "valid-key"),
    ],
)
def test_retry_diagnostics_stay_out_of_captured_value(
    invocation: str, stdin: str, expected: str
) -> None:
    """Validation messages must not become the wizard's captured configuration."""
    prompts = (REPO_ROOT / "scripts/setup/lib/prompts.sh").as_posix()
    validation = (REPO_ROOT / "scripts/setup/lib/validation.sh").as_posix()
    result = run_bash_process(
        f"""source "{prompts}"
source "{validation}"
value=$(printf '%s' {shlex.quote(stdin)} | {invocation})
printf '%s' "$value"
""",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == expected
    assert "Please try again." in result.stderr


def test_collect_postgres_config_uses_fixed_bundled_port_and_compose_overrides() -> (
    None
):
    """Bundled PostgreSQL should use the fixed service port and compose overrides."""
    values = run_collect_bash(
        """
collect_postgres_config yes
printf 'COMPOSE_POSTGRES_HOST=%s\\n' "${COMPOSE_ENV_OVERRIDES[POSTGRES_HOST]}"
printf 'COMPOSE_POSTGRES_PORT=%s\\n' "${COMPOSE_ENV_OVERRIDES[POSTGRES_PORT]}"
printf 'DOCKER_SERVICE=%s\\n' "${DOCKER_SERVICES[0]}"
""",
        prompt_answers={
            "PostgreSQL host": "localhost",
            "PostgreSQL user": "lightrag",
            "PostgreSQL database": "lightrag",
        },
        stubs="""
confirm_default_yes() { return 0; }
prompt_secret_with_default() { printf '%s' "$2"; }
mask_sensitive_input() { printf 'supersecret'; }
""",
        print_env=("POSTGRES_HOST", "POSTGRES_PORT"),
    )
    assert values["POSTGRES_HOST"] == "localhost"
    assert values["POSTGRES_PORT"] == "5432"
    assert values["COMPOSE_POSTGRES_HOST"] == "postgres"
    assert values["COMPOSE_POSTGRES_PORT"] == "5432"
    assert values["DOCKER_SERVICE"] == "postgres"


@pytest.mark.parametrize(
    (
        "collector_call",
        "original_env",
        "confirm_stub",
        "prompt_answers",
        "until_valid_answers",
        "secret_answer",
        "log_secret_prefix",
        "expected_values",
        "expected_prompt_log",
    ),
    [
        (
            "collect_postgres_config yes",
            {"POSTGRES_USER": "", "POSTGRES_PASSWORD": "", "POSTGRES_DATABASE": ""},
            "confirm_default_yes() { return 0; }",
            {"PostgreSQL host": "localhost"},
            {},
            None,
            True,
            {
                "POSTGRES_USER": "rag",
                "POSTGRES_PASSWORD": "rag",
                "POSTGRES_DATABASE": "lightrag",
            },
            "PostgreSQL host[localhost]|PostgreSQL user[rag]|secret:PostgreSQL password: [rag]|PostgreSQL database[lightrag]",
        ),
        (
            "collect_postgres_config yes",
            {
                "POSTGRES_USER": "existing-user",
                "POSTGRES_PASSWORD": "existing-password",
                "POSTGRES_DATABASE": "existing-db",
            },
            "confirm_default_yes() { return 0; }",
            {
                "PostgreSQL host": "localhost",
                "PostgreSQL user": "updated-user",
                "PostgreSQL database": "updated-db",
            },
            {},
            "updated-password",
            False,
            {
                "POSTGRES_USER": "updated-user",
                "POSTGRES_PASSWORD": "updated-password",
                "POSTGRES_DATABASE": "updated-db",
            },
            "PostgreSQL host[localhost]|PostgreSQL user[existing-user]|PostgreSQL password: [existing-password]|PostgreSQL database[existing-db]",
        ),
        (
            "collect_postgres_config no",
            {"POSTGRES_USER": "", "POSTGRES_PASSWORD": ""},
            "confirm_default_no() { return 1; }",
            {
                "PostgreSQL host": "db.internal",
                "PostgreSQL user": "host-user",
                "PostgreSQL database": "host-db",
            },
            {"PostgreSQL port": "6543"},
            "host-password",
            False,
            {
                "POSTGRES_HOST": "db.internal",
                "POSTGRES_PORT": "6543",
                "POSTGRES_USER": "host-user",
                "POSTGRES_PASSWORD": "host-password",
            },
            "PostgreSQL host[localhost]|PostgreSQL port[5432]|PostgreSQL user[rag]|PostgreSQL password: [rag]|PostgreSQL database[lightrag]",
        ),
    ],
    ids=["docker-empty-creds", "docker-existing-creds", "host-empty-creds"],
)
def test_collect_postgres_config_prompts_for_credentials(
    collector_call: str,
    original_env: dict[str, str],
    confirm_stub: str,
    prompt_answers: dict[str, str],
    until_valid_answers: dict[str, str],
    secret_answer: str | None,
    log_secret_prefix: bool,
    expected_values: dict[str, str],
    expected_prompt_log: str,
) -> None:
    """PostgreSQL credential prompts should apply host/saved defaults across docker and host modes."""
    values = run_collect_bash(
        collector_call,
        original_env_values=original_env,
        stubs=confirm_stub,
        prompt_answers=prompt_answers,
        until_valid_answers=until_valid_answers,
        secret_answer=secret_answer,
        log_prompts=True,
        log_defaults=True,
        log_secret_prefix=log_secret_prefix,
        print_env=tuple(expected_values),
    )
    for key, expected in expected_values.items():
        assert values[key] == expected
    assert values["PROMPT_LOG"] == expected_prompt_log


def test_collect_server_config_includes_summary_language_last() -> None:
    """Server config should prompt for summary language after the WebUI fields."""
    values = run_collect_bash(
        "collect_server_config",
        prompt_answers={
            "Server host": "127.0.0.1",
            "WebUI title": "Custom KB",
            "WebUI description": "Custom description",
            "Summary language": "Chinese",
        },
        until_valid_answers={"Server port": "9630"},
        log_prompts=True,
        print_env=(
            "HOST",
            "PORT",
            "WEBUI_TITLE",
            "WEBUI_DESCRIPTION",
            "SUMMARY_LANGUAGE",
        ),
    )
    assert values["HOST"] == "127.0.0.1"
    assert values["PORT"] == "9630"
    assert values["WEBUI_TITLE"] == "Custom KB"
    assert values["WEBUI_DESCRIPTION"] == "Custom description"
    assert values["SUMMARY_LANGUAGE"] == "Chinese"
    assert (
        values["PROMPT_LOG"]
        == "Server host|Server port|WebUI title|WebUI description|Summary language"
    )


@pytest.mark.parametrize(
    ("env_values", "collector_call", "env_key", "expected_value"),
    [
        (
            {"POSTGRES_HOST": "db.example.com", "POSTGRES_PORT": "6543"},
            "collect_postgres_config yes",
            "POSTGRES_HOST",
            "localhost",
        ),
        (
            {"POSTGRES_HOST": "db.example.com", "POSTGRES_PORT": "6543"},
            "collect_postgres_config yes",
            "POSTGRES_PORT",
            "5432",
        ),
        (
            {"NEO4J_URI": "neo4j+s://graph.example.com"},
            "collect_neo4j_config yes",
            "NEO4J_URI",
            "neo4j://localhost:7687",
        ),
        (
            {"MONGO_URI": "mongodb://mongo.example.com:27018/"},
            "collect_mongodb_config yes",
            "MONGO_URI",
            "mongodb://localhost:27017/?directConnection=true",
        ),
        (
            {"REDIS_URI": "redis://cache.example.com:6380/1"},
            "collect_redis_config yes",
            "REDIS_URI",
            "redis://localhost:6379",
        ),
        (
            {"MILVUS_URI": "http://milvus.example.com:19530"},
            "collect_milvus_config yes",
            "MILVUS_URI",
            "http://localhost:19530",
        ),
        (
            {"QDRANT_URL": "http://qdrant.example.com:6333"},
            "collect_qdrant_config yes",
            "QDRANT_URL",
            "http://localhost:6333",
        ),
        (
            {"MEMGRAPH_URI": "bolt://memgraph.example.com:7687"},
            "collect_memgraph_config yes",
            "MEMGRAPH_URI",
            "bolt://localhost:7687",
        ),
        (
            {"NEO4J_URI": "neo4j://localhost:7777"},
            "collect_neo4j_config yes",
            "NEO4J_URI",
            "neo4j://localhost:7687",
        ),
        (
            {"MILVUS_URI": "http://localhost:29530"},
            "collect_milvus_config yes",
            "MILVUS_URI",
            "http://localhost:19530",
        ),
        (
            {"QDRANT_URL": "http://localhost:16333"},
            "collect_qdrant_config yes",
            "QDRANT_URL",
            "http://localhost:6333",
        ),
        (
            {"MEMGRAPH_URI": "bolt://localhost:17687"},
            "collect_memgraph_config yes",
            "MEMGRAPH_URI",
            "bolt://localhost:7687",
        ),
    ],
    ids=[
        "postgres-remote-host",
        "postgres-port-reset-to-bundled-default",
        "neo4j-remote-uri",
        "mongodb-remote-uri",
        "redis-remote-uri",
        "milvus-remote-uri",
        "qdrant-remote-uri",
        "memgraph-remote-uri",
        "neo4j-local-port",
        "milvus-local-port",
        "qdrant-local-port",
        "memgraph-local-port",
    ],
)
def test_collect_local_service_configs_normalize_stale_values(
    env_values: dict[str, str], collector_call: str, env_key: str, expected_value: str
) -> None:
    """Bundled services should normalize stale remote or localhost endpoints on rerun."""
    values = run_collect_bash(
        collector_call,
        env_values=env_values,
        prompt_answers={
            "PostgreSQL user": "lightrag",
            "PostgreSQL database": "lightrag",
            "Neo4j database": "neo4j",
            "MongoDB database": "LightRAG",
            "Milvus database name": "lightrag",
        },
        until_valid_answers={},
        stubs="""
confirm_default_yes() { return 0; }
prompt_choice() { printf '%s' "$2"; }
prompt_secret_with_default() { printf '%s' "$2"; }
""",
        print_env=(env_key,),
    )
    assert values[env_key] == expected_value


def test_collect_ssl_config_can_disable_loaded_ssl_values(tmp_path: Path) -> None:
    """Declining SSL should clear previously loaded cert paths and staged sources."""
    cert_path = tmp_path / "cert.pem"
    cert_path.write_text("cert", encoding="utf-8")
    key_path = tmp_path / "key.pem"
    key_path.write_text("key", encoding="utf-8")
    values = run_collect_bash(
        """
collect_ssl_config
printf 'SSL_CERT_SOURCE_PATH=%s\\n' "$SSL_CERT_SOURCE_PATH"
printf 'SSL_KEY_SOURCE_PATH=%s\\n' "$SSL_KEY_SOURCE_PATH"
""",
        tmp_path=tmp_path,
        env_lines=["SSL=true", f"SSL_CERTFILE={cert_path}", f"SSL_KEYFILE={key_path}"],
        stubs="confirm_default_yes() { return 1; }",
        print_set=("SSL", "SSL_CERTFILE", "SSL_KEYFILE"),
    )
    assert values["SSL_SET"] == ""
    assert values["SSL_CERTFILE_SET"] == ""
    assert values["SSL_KEYFILE_SET"] == ""
    assert values["SSL_CERT_SOURCE_PATH"] == ""
    assert values["SSL_KEY_SOURCE_PATH"] == ""


@pytest.mark.parametrize(
    ("collector_name", "binding_prefix", "env_lines"),
    [
        (
            "collect_llm_config",
            "LLM",
            [
                "LLM_BINDING=openai",
                "LLM_MODEL=gpt-4o",
                "LLM_BINDING_HOST=https://api.openai.com/v1",
                "LLM_BINDING_API_KEY=${OPENAI_API_KEY}",
            ],
        ),
        (
            "collect_embedding_config",
            "EMBEDDING",
            [
                "EMBEDDING_BINDING=openai",
                "EMBEDDING_MODEL=text-embedding-3-large",
                "EMBEDDING_DIM=3072",
                "EMBEDDING_BINDING_HOST=https://api.openai.com/v1",
                "EMBEDDING_BINDING_API_KEY=${OPENAI_API_KEY}",
            ],
        ),
    ],
    ids=["llm-bedrock-clears-api-key", "embedding-bedrock-clears-api-key"],
)
def test_collect_provider_config_clears_stale_api_key_for_bedrock(
    tmp_path: Path, collector_name: str, binding_prefix: str, env_lines: list[str]
) -> None:
    """Switching a provider to Bedrock should remove stale API-key settings."""
    values = run_collect_bash(
        f"""
{collector_name}
if validate_sensitive_env_literals; then
  printf 'VALID=yes\\n'
else
  printf 'VALID=no\\n'
fi
""",
        tmp_path=tmp_path,
        env_lines=env_lines,
        prompt_answers={},
        stubs="""
prompt_choice() { printf 'bedrock'; }
prompt_required_secret() { printf 'dummy-secret'; }
mask_sensitive_input() { printf ''; }
confirm_default_yes() { return 0; }
""",
        print_env=(f"{binding_prefix}_BINDING",),
        print_set=(f"{binding_prefix}_BINDING_API_KEY",),
    )
    assert values[f"{binding_prefix}_BINDING"] == "bedrock"
    assert values[f"{binding_prefix}_BINDING_API_KEY_SET"] == ""
    assert values["VALID"] == "yes"


@pytest.mark.parametrize(
    (
        "collector_name",
        "binding_prefix",
        "provider_choice",
        "secret_stub",
        "expected_binding",
        "expected_model",
        "expected_host",
        "expected_dim",
        "expected_api_key_set",
    ),
    [
        (
            "collect_llm_config",
            "LLM",
            "ollama",
            "",
            "ollama",
            "mistral-nemo:latest",
            "http://localhost:11434",
            "",
            "",
        ),
        (
            "collect_embedding_config",
            "EMBEDDING",
            "jina",
            "prompt_secret_until_valid_with_default() { printf 'jina-secret-key'; }",
            "jina",
            "jina-embeddings-v4",
            "https://api.jina.ai/v1/embeddings",
            "2048",
            "set",
        ),
        (
            "collect_llm_config",
            "LLM",
            "gemini",
            "prompt_secret_until_valid_with_default() { printf 'gemini-secret-key'; }",
            "gemini",
            "gemini-flash-latest",
            "DEFAULT_GEMINI_ENDPOINT",
            "",
            "set",
        ),
        (
            "collect_llm_config",
            "LLM",
            "bedrock",
            """
prompt_clearable_with_default() { printf ''; }
prompt_required_secret() { return 1; }
confirm_default_yes() { return 1; }
""",
            "bedrock",
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "DEFAULT_BEDROCK_ENDPOINT",
            "",
            "",
        ),
        (
            "collect_embedding_config",
            "EMBEDDING",
            "gemini",
            "prompt_secret_until_valid_with_default() { printf 'gemini-secret-key'; }",
            "gemini",
            "gemini-embedding-001",
            "DEFAULT_GEMINI_ENDPOINT",
            "1536",
            "set",
        ),
        (
            "collect_embedding_config",
            "EMBEDDING",
            "bedrock",
            """
prompt_clearable_with_default() { printf ''; }
prompt_required_secret() { return 1; }
confirm_default_yes() { return 1; }
""",
            "bedrock",
            "amazon.titan-embed-text-v2:0",
            "DEFAULT_BEDROCK_ENDPOINT",
            "1024",
            "",
        ),
    ],
    ids=[
        "llm-provider-defaults",
        "embedding-provider-defaults",
        "llm-gemini-sentinel-default",
        "llm-bedrock-sentinel-default",
        "embedding-gemini-sentinel-default",
        "embedding-bedrock-sentinel-default",
    ],
)
def test_collect_provider_config_uses_provider_specific_defaults(
    collector_name: str,
    binding_prefix: str,
    provider_choice: str,
    secret_stub: str,
    expected_binding: str,
    expected_model: str,
    expected_host: str,
    expected_dim: str,
    expected_api_key_set: str,
) -> None:
    """Fresh provider selection should pick provider-specific defaults."""
    values = run_collect_bash(
        collector_name,
        prompt_answers={},
        stubs=f"prompt_choice() {{ printf '{provider_choice}'; }}\n{secret_stub}",
        print_env=(
            f"{binding_prefix}_BINDING",
            f"{binding_prefix}_MODEL",
            f"{binding_prefix}_BINDING_HOST",
            f"{binding_prefix}_DIM",
        ),
        print_set=(f"{binding_prefix}_BINDING_API_KEY",),
    )
    assert values[f"{binding_prefix}_BINDING"] == expected_binding
    assert values[f"{binding_prefix}_MODEL"] == expected_model
    assert values[f"{binding_prefix}_BINDING_HOST"] == expected_host
    assert values[f"{binding_prefix}_DIM"] == expected_dim
    assert values[f"{binding_prefix}_BINDING_API_KEY_SET"] == expected_api_key_set


@pytest.mark.parametrize(
    (
        "collector_name",
        "binding_prefix",
        "env_lines",
        "expected_host",
        "expected_api_key",
    ),
    [
        (
            "collect_llm_config",
            "LLM",
            [
                "LLM_BINDING=gemini",
                "LLM_MODEL=gemini-flash-latest",
                "LLM_BINDING_HOST=https://generativelanguage.googleapis.com",
                "LLM_BINDING_API_KEY=gemini-existing-key",
            ],
            "https://generativelanguage.googleapis.com",
            "gemini-existing-key",
        ),
        (
            "collect_embedding_config",
            "EMBEDDING",
            [
                "EMBEDDING_BINDING=bedrock",
                "EMBEDDING_MODEL=amazon.titan-embed-text-v2:0",
                "EMBEDDING_DIM=1024",
                "EMBEDDING_BINDING_HOST=https://bedrock.amazonaws.com",
            ],
            "https://bedrock.amazonaws.com",
            "",
        ),
    ],
    ids=[
        "llm-rerun-preserves-explicit-gemini-host",
        "embedding-rerun-preserves-explicit-bedrock-host",
    ],
)
def test_collect_provider_config_preserves_explicit_host_on_rerun(
    tmp_path: Path,
    collector_name: str,
    binding_prefix: str,
    env_lines: list[str],
    expected_host: str,
    expected_api_key: str,
) -> None:
    """Reruns should keep saved explicit provider hosts instead of swapping to sentinels."""
    values = run_collect_bash(
        collector_name,
        tmp_path=tmp_path,
        env_lines=env_lines,
        prompt_answers={},
        stubs="""
prompt_choice() { printf '%s' "$2"; }
prompt_secret_until_valid_with_default() { printf '%s' "$2"; }
prompt_clearable_with_default() { printf ''; }
prompt_required_secret() { return 1; }
confirm_default_yes() { return 1; }
""",
        print_env=(
            f"{binding_prefix}_BINDING_HOST",
            f"{binding_prefix}_BINDING_API_KEY",
        ),
    )
    assert values[f"{binding_prefix}_BINDING_HOST"] == expected_host
    assert values[f"{binding_prefix}_BINDING_API_KEY"] == expected_api_key


@pytest.mark.parametrize(
    (
        "collector_name",
        "binding_prefix",
        "env_lines",
        "prompt_stubs",
        "expected_binding",
        "expected_model",
        "expected_host",
        "expected_dim",
        "expected_api_key",
    ),
    [
        (
            "collect_llm_config",
            "LLM",
            [
                "LLM_BINDING=openai-ollama",
                "LLM_MODEL=llama3.1:8b",
                "LLM_BINDING_HOST=http://localhost:11434/v1",
                "LLM_BINDING_API_KEY=sk-local-test-key",
            ],
            "prompt_secret_until_valid_with_default() { printf '%s' \"$2\"; }",
            "openai-ollama",
            "llama3.1:8b",
            "http://localhost:11434/v1",
            "",
            "sk-local-test-key",
        ),
        (
            "collect_embedding_config",
            "EMBEDDING",
            [
                "EMBEDDING_BINDING=lollms",
                "EMBEDDING_MODEL=lollms_embedding_model",
                "EMBEDDING_DIM=1024",
                "EMBEDDING_BINDING_HOST=http://localhost:9600",
            ],
            "",
            "lollms",
            "lollms_embedding_model",
            "http://localhost:9600",
            "1024",
            "",
        ),
    ],
    ids=["llm-rerun-preserves-openai-ollama", "embedding-rerun-preserves-lollms"],
)
def test_collect_provider_config_preserves_supported_binding_on_rerun(
    tmp_path: Path,
    collector_name: str,
    binding_prefix: str,
    env_lines: list[str],
    prompt_stubs: str,
    expected_binding: str,
    expected_model: str,
    expected_host: str,
    expected_dim: str,
    expected_api_key: str,
) -> None:
    """Reruns should preserve supported provider bindings and their saved settings."""
    values = run_collect_bash(
        collector_name,
        tmp_path=tmp_path,
        env_lines=env_lines,
        prompt_answers={},
        stubs=prompt_stubs,
        print_env=(
            f"{binding_prefix}_BINDING",
            f"{binding_prefix}_MODEL",
            f"{binding_prefix}_BINDING_HOST",
            f"{binding_prefix}_DIM",
            f"{binding_prefix}_BINDING_API_KEY",
        ),
    )
    assert values[f"{binding_prefix}_BINDING"] == expected_binding
    assert values[f"{binding_prefix}_MODEL"] == expected_model
    assert values[f"{binding_prefix}_BINDING_HOST"] == expected_host
    assert values[f"{binding_prefix}_DIM"] == expected_dim
    assert values[f"{binding_prefix}_BINDING_API_KEY"] == expected_api_key


def test_collect_embedding_config_forces_ollama_for_openai_ollama_llm(
    tmp_path: Path,
) -> None:
    """`openai-ollama` should not preserve a conflicting embedding provider."""
    values = run_collect_bash(
        "collect_embedding_config",
        tmp_path=tmp_path,
        env_lines=[
            "LLM_BINDING=openai-ollama",
            "EMBEDDING_BINDING=openai",
            "EMBEDDING_MODEL=text-embedding-3-large",
            "EMBEDDING_DIM=3072",
            "EMBEDDING_BINDING_HOST=https://api.openai.com/v1",
            "EMBEDDING_BINDING_API_KEY=local-key",
        ],
        prompt_answers={},
        stubs="prompt_secret_until_valid_with_default() { printf '%s' \"$2\"; }",
        print_env=(
            "EMBEDDING_BINDING",
            "EMBEDDING_MODEL",
            "EMBEDDING_DIM",
            "EMBEDDING_BINDING_HOST",
        ),
        print_set=("EMBEDDING_BINDING_API_KEY",),
    )
    assert values["EMBEDDING_BINDING"] == "ollama"
    assert values["EMBEDDING_MODEL"] == "bge-m3:latest"
    assert values["EMBEDDING_DIM"] == "1024"
    assert values["EMBEDDING_BINDING_HOST"] == "http://localhost:11434"
    assert values["EMBEDDING_BINDING_API_KEY_SET"] == ""


def test_collect_llm_config_allows_bedrock_ambient_credential_chain() -> None:
    """Bedrock setup should allow IAM roles, AWS profiles, or SSO without saved keys."""
    values = run_collect_bash(
        "collect_llm_config",
        prompt_answers={},
        stubs="""
prompt_choice() { printf 'bedrock'; }
prompt_clearable_with_default() { printf ''; }
prompt_required_secret() { return 1; }
confirm_default_yes() { return 1; }
""",
        print_env=("LLM_BINDING",),
        print_set=(
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_REGION",
        ),
    )
    assert values["LLM_BINDING"] == "bedrock"
    assert values["AWS_ACCESS_KEY_ID_SET"] == ""
    assert values["AWS_SECRET_ACCESS_KEY_SET"] == ""
    assert values["AWS_SESSION_TOKEN_SET"] == ""
    assert values["AWS_REGION_SET"] == ""


@pytest.mark.parametrize(
    (
        "env_lines",
        "prompt_answers",
        "secret_stub",
        "expected_keyword",
        "expected_query",
    ),
    [
        (
            None,
            {"LLM model": "gpt-4o"},
            "prompt_secret_until_valid_with_default() { printf 'fake-key'; }",
            "gpt-4o",
            "gpt-4o",
        ),
        (
            [
                "LLM_BINDING=openai",
                "LLM_MODEL=gpt-4o",
                "LLM_BINDING_HOST=https://api.openai.com/v1",
                "LLM_BINDING_API_KEY=existing-key",
                "KEYWORD_LLM_MODEL=custom-keyword-model",
                "QUERY_LLM_MODEL=custom-query-model",
            ],
            {},
            "prompt_secret_until_valid_with_default() { printf '%s' \"$2\"; }",
            "custom-keyword-model",
            "custom-query-model",
        ),
    ],
    ids=["default-to-base-model-when-unset", "reuse-existing-values"],
)
def test_collect_llm_config_role_models(
    tmp_path: Path,
    env_lines: list[str] | None,
    prompt_answers: dict[str, str],
    secret_stub: str,
    expected_keyword: str,
    expected_query: str,
) -> None:
    """KEYWORD/QUERY_LLM_MODEL default to LLM_MODEL when unset and survive reruns when set."""
    values = run_collect_bash(
        "collect_llm_config",
        tmp_path=tmp_path if env_lines is not None else None,
        env_lines=env_lines,
        prompt_answers=prompt_answers,
        stubs=f"prompt_choice() {{ printf 'openai'; }}\n{secret_stub}",
        print_env=("LLM_MODEL", "KEYWORD_LLM_MODEL", "QUERY_LLM_MODEL"),
    )
    assert values["LLM_MODEL"] == "gpt-4o"
    assert values["KEYWORD_LLM_MODEL"] == expected_keyword
    assert values["QUERY_LLM_MODEL"] == expected_query


def test_collect_rerank_config_preserves_api_key_when_disabled(tmp_path: Path) -> None:
    """Disabling reranking should preserve credentials so they survive re-enable."""
    values = run_collect_bash(
        """
collect_rerank_config
if validate_sensitive_env_literals; then
  printf 'VALID=yes\\n'
else
  printf 'VALID=no\\n'
fi
""",
        tmp_path=tmp_path,
        env_lines=[
            "RERANK_BINDING=cohere",
            "RERANK_MODEL=rerank-v3.5",
            "RERANK_BINDING_HOST=https://api.cohere.com/v1/rerank",
            "RERANK_BINDING_API_KEY=test-api-key-literal",
        ],
        stubs="confirm_default_yes() { return 1; }",
        print_env=("RERANK_BINDING",),
        print_set=("RERANK_BINDING_API_KEY",),
    )
    assert values["RERANK_BINDING"] == "null"
    assert values["RERANK_BINDING_API_KEY_SET"] == "set"
    assert values["VALID"] == "yes"


def test_collect_rerank_config_does_not_offer_vllm_provider_option() -> None:
    """The generic rerank provider prompt should only expose valid RERANK_BINDING values."""
    values = run_collect_bash(
        "collect_rerank_config",
        env_values={"RERANK_BINDING": "cohere"},
        prompt_answers={},
        stubs="""
confirm_default_no() { return 0; }
prompt_choice() {
  case "$1" in
    "Rerank provider")
      shift 2
      for option in "$@"; do
        if [[ "$option" == "vllm" ]]; then
          echo "unexpected vllm option" >&2
          return 91
        fi
      done
      printf 'cohere'
      ;;
    *) printf '%s' "$2" ;;
  esac
}
prompt_secret_until_valid_with_default() { printf 'cohere-secret-123'; }
""",
        print_env=("RERANK_BINDING",),
    )
    assert values["RERANK_BINDING"] == "cohere"


def test_collect_rerank_config_switching_from_vllm_clears_local_defaults() -> None:
    """Switching from local vLLM to hosted rerank should replace stale vLLM values with provider defaults."""
    values = run_collect_bash(
        "collect_rerank_config",
        env_values={
            "LIGHTRAG_SETUP_RERANK_PROVIDER": "vllm",
            "RERANK_BINDING": "cohere",
            "RERANK_MODEL": "BAAI/bge-reranker-v2-m3",
            "RERANK_BINDING_HOST": "http://localhost:8000/rerank",
        },
        prompt_answers={},
        stubs="""
confirm_default_no() { return 0; }
prompt_choice() {
  case "$1" in
    "Rerank provider") printf 'cohere' ;;
    *) printf '%s' "$2" ;;
  esac
}
prompt_secret_until_valid_with_default() { printf 'cohere-secret-123'; }
""",
        print_env=(
            "RERANK_BINDING",
            "LIGHTRAG_SETUP_RERANK_PROVIDER",
            "RERANK_MODEL",
            "RERANK_BINDING_HOST",
        ),
    )
    assert values["RERANK_BINDING"] == "cohere"
    assert values["LIGHTRAG_SETUP_RERANK_PROVIDER"] == ""
    assert values["RERANK_MODEL"] != "BAAI/bge-reranker-v2-m3"
    assert values["RERANK_MODEL"] == "rerank-v3.5"
    assert "localhost:8000" not in values["RERANK_BINDING_HOST"]
    assert "cohere" in values["RERANK_BINDING_HOST"]


def test_collect_rerank_config_ignores_vllm_marker_when_docker_is_predeclined() -> None:
    """A predeclined Docker path should default the provider prompt to the binding, not the setup marker."""
    values = run_collect_bash(
        'collect_rerank_config "yes" "no"',
        env_values={
            "LIGHTRAG_SETUP_RERANK_PROVIDER": "vllm",
            "RERANK_BINDING": "cohere",
        },
        prompt_answers={},
        stubs="""
prompt_choice() {
  case "$1" in
    "Rerank provider")
      if [[ "$2" != "cohere" ]]; then
        echo "unexpected rerank provider default: $2" >&2
        return 91
      fi
      printf 'cohere'
      ;;
    *) printf '%s' "$2" ;;
  esac
}
prompt_secret_until_valid_with_default() { printf 'cohere-secret-123'; }
""",
        print_env=("RERANK_BINDING",),
    )
    assert values["RERANK_BINDING"] == "cohere"


def test_collect_milvus_config_defaults_to_existing_database_name() -> None:
    """Milvus database prompt should preserve the documented default database."""
    values = run_collect_bash(
        "collect_milvus_config no",
        prompt_answers={},
        until_valid_answers={},
        stubs="confirm_default_yes() { return 1; }",
        print_env=("MILVUS_DB_NAME",),
    )
    assert values["MILVUS_DB_NAME"] == "lightrag"


def test_collect_milvus_config_initializes_minio_credentials_for_local_docker(
    tmp_path: Path,
) -> None:
    """Local Docker Milvus should write default MinIO credentials when none exist yet."""
    copy_repo_template(tmp_path, "env.example")
    run_collect_bash(
        """
collect_milvus_config yes
generate_env_file "$REPO_ROOT/env.example" "$REPO_ROOT/.env"
""",
        tmp_path=tmp_path,
        prompt_answers={},
        until_valid_answers={},
        stubs="""
confirm_default_yes() { return 0; }
prompt_choice() { printf '%s' "$2"; }
""",
        cwd=tmp_path,
    )
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "MINIO_ACCESS_KEY_ID=minioadmin" in env_text
    assert "MINIO_SECRET_ACCESS_KEY=minioadmin" in env_text


@pytest.mark.parametrize(
    ("env_values", "nvidia_impl", "expected_device"),
    [
        ({"MILVUS_DEVICE": "cpu"}, "nvidia-smi() { return 0; }", "cpu"),
        ({"MILVUS_DEVICE": "cuda"}, "nvidia-smi() { return 1; }", "cuda"),
        ({}, "nvidia-smi() { return 0; }", "cuda"),
    ],
    ids=["saved-cpu-wins", "saved-cuda-wins", "gpu-host-defaults-to-cuda"],
)
def test_collect_milvus_config_resolves_device_default_for_local_docker(
    env_values: dict[str, str], nvidia_impl: str, expected_device: str
) -> None:
    """Milvus device defaults should prefer saved state and otherwise use host CUDA detection."""
    values = run_collect_bash(
        "collect_milvus_config yes",
        env_values=env_values,
        prompt_answers={},
        until_valid_answers={},
        stubs=f"""
{nvidia_impl}
confirm_default_yes() {{ return 0; }}
prompt_choice() {{ printf '%s' "$2"; }}
""",
        print_env=("MILVUS_DEVICE",),
    )
    assert values["MILVUS_DEVICE"] == expected_device


@pytest.mark.parametrize(
    ("collector_call", "device_prompt", "endpoint_prompt"),
    [
        ("collect_milvus_config yes", "Milvus device", "Milvus URI"),
        ("collect_qdrant_config yes", "Qdrant device", "Qdrant URL"),
    ],
)
def test_local_vector_db_device_prompt_is_first_follow_up_after_docker_choice(
    collector_call: str, device_prompt: str, endpoint_prompt: str
) -> None:
    """GPU/CPU selection should be the first prompt after choosing local Docker deployment."""
    values = run_collect_bash(
        collector_call,
        stubs="confirm_default_yes() { return 0; }",
        log_prompts=True,
    )
    assert values["PROMPT_LOG"].startswith(f"{device_prompt}|{endpoint_prompt}")


def test_collect_mongodb_config_local_service_strips_stale_credentials_on_rerun() -> (
    None
):
    """Bundled MongoDB should keep host `.env` aligned with the unauthenticated template."""
    values = run_collect_bash(
        """
collect_mongodb_config yes
printf 'COMPOSE_MONGO_URI=%s\\n' "${COMPOSE_ENV_OVERRIDES[MONGO_URI]}"
printf 'DOCKER_SERVICE=%s\\n' "${DOCKER_SERVICES[0]}"
""",
        env_values={"MONGO_URI": "mongodb://root:secret@localhost:27018/"},
        prompt_answers={"MongoDB database": "LightRAG"},
        until_valid_answers={},
        stubs="confirm_default_yes() { return 0; }",
        print_env=("MONGO_URI",),
    )
    assert values["MONGO_URI"] == "mongodb://localhost:27017/?directConnection=true"
    assert (
        values["COMPOSE_MONGO_URI"] == "mongodb://mongodb:27017/?directConnection=true"
    )
    assert values["DOCKER_SERVICE"] == "mongodb"


@pytest.mark.parametrize(
    ("initial_uri", "expected_uri"),
    [
        (
            "mongodb://localhost:27017/?directConnection=true",
            "mongodb+srv://cluster.example.mongodb.net/",
        ),
        (
            "mongodb://atlas-local.example.com:27017/LightRAG?replicaSet=rs0&directConnection=true",
            "mongodb://atlas-local.example.com:27017/LightRAG?replicaSet=rs0&directConnection=true",
        ),
    ],
    ids=["resets-wizard-managed-local-uri", "preserves-external-atlas-local-uri"],
)
def test_collect_mongodb_config_switching_off_docker_handles_mongo_uri(
    initial_uri: str, expected_uri: str
) -> None:
    """Switching off bundled MongoDB resets the managed local URI while keeping external Atlas Local URIs."""
    values = run_collect_bash(
        """
collect_mongodb_config yes
printf 'COMPOSE_MONGO_URI=%s\\n' "${COMPOSE_ENV_OVERRIDES[MONGO_URI]-}"
""",
        env_values={
            "LIGHTRAG_VECTOR_STORAGE": "MongoVectorDBStorage",
            "MONGO_URI": initial_uri,
        },
        prompt_answers={},
        until_valid_answers={},
        stubs="confirm_default_yes() { return 1; }",
        print_env=("MONGO_URI",),
    )
    assert values["MONGO_URI"] == expected_uri
    assert values["COMPOSE_MONGO_URI"] == ""


def test_collect_redis_config_local_service_normalizes_custom_host_port() -> None:
    """Bundled Redis should keep host `.env` aligned with the published local port."""
    values = run_collect_bash(
        """
collect_redis_config yes
printf 'COMPOSE_REDIS_URI=%s\\n' "${COMPOSE_ENV_OVERRIDES[REDIS_URI]}"
printf 'DOCKER_SERVICE=%s\\n' "${DOCKER_SERVICES[0]}"
""",
        env_values={"REDIS_URI": "redis://localhost:6380/1"},
        until_valid_answers={},
        stubs="confirm_default_yes() { return 0; }",
        print_env=("REDIS_URI",),
    )
    assert values["REDIS_URI"] == "redis://localhost:6379/1"
    assert values["COMPOSE_REDIS_URI"] == "redis://redis:6379"
    assert values["DOCKER_SERVICE"] == "redis"


def test_collect_security_config_can_clear_existing_values_on_rerun(
    tmp_path: Path,
) -> None:
    """Rerunning security setup should be able to remove previously saved values."""
    values = run_collect_bash(
        f"""
collect_security_config yes no
generate_env_file "{REPO_ROOT}/env.example" "$REPO_ROOT/.env.generated"
""",
        tmp_path=tmp_path,
        env_lines=[
            "AUTH_ACCOUNTS=admin:secret",
            "TOKEN_SECRET=jwt-secret",
            "TOKEN_EXPIRE_HOURS=72",
            "LIGHTRAG_API_KEY=api-key",
            "WHITELIST_PATHS=/health,/api/*,/docs",
        ],
        stubs="""
confirm_default_no() { return 0; }
prompt_clearable_with_default() { printf '%s' "$CLEAR_INPUT_SENTINEL"; }
prompt_clearable_secret_with_default() { printf '%s' "$CLEAR_INPUT_SENTINEL"; }
""",
        print_set=(
            "AUTH_ACCOUNTS",
            "TOKEN_SECRET",
            "TOKEN_EXPIRE_HOURS",
            "LIGHTRAG_API_KEY",
            "WHITELIST_PATHS",
        ),
    )
    generated_lines = (
        (tmp_path / ".env.generated").read_text(encoding="utf-8").splitlines()
    )
    assert values["AUTH_ACCOUNTS_SET"] == ""
    assert values["TOKEN_SECRET_SET"] == ""
    assert values["TOKEN_EXPIRE_HOURS_SET"] == ""
    assert values["LIGHTRAG_API_KEY_SET"] == ""
    assert values["WHITELIST_PATHS_SET"] == "set"
    assert not any(line.startswith("AUTH_ACCOUNTS=") for line in generated_lines)
    assert not any(line.startswith("TOKEN_SECRET=") for line in generated_lines)
    assert not any(line.startswith("TOKEN_EXPIRE_HOURS=") for line in generated_lines)
    assert not any(line.startswith("LIGHTRAG_API_KEY=") for line in generated_lines)
    assert "WHITELIST_PATHS=" in generated_lines


def test_collect_security_config_preserves_explicit_empty_whitelist_on_rerun(
    tmp_path: Path,
) -> None:
    """Rerunning security setup should keep an explicitly empty whitelist unchanged."""
    values = run_collect_bash(
        "collect_security_config no no",
        tmp_path=tmp_path,
        env_lines=["WHITELIST_PATHS="],
        stubs="""
prompt_clearable_with_default() { printf '%s' "$2"; }
prompt_clearable_secret_with_default() { printf '%s' "$2"; }
""",
        print_env=("WHITELIST_PATHS",),
        print_set=("WHITELIST_PATHS",),
    )
    assert values["WHITELIST_PATHS_SET"] == "set"
    assert values["WHITELIST_PATHS"] == ""


def test_collect_observability_config_clears_existing_values_on_rerun(
    tmp_path: Path,
) -> None:
    """Rerunning setup should remove saved Langfuse settings when observability is declined."""
    values = run_collect_bash(
        f"""
collect_observability_config
generate_env_file "{REPO_ROOT}/env.example" "$REPO_ROOT/.env.generated"
""",
        tmp_path=tmp_path,
        env_lines=[
            "LANGFUSE_ENABLE_TRACE=true",
            "LANGFUSE_SECRET_KEY=old-secret",
            "LANGFUSE_PUBLIC_KEY=old-public",
            "LANGFUSE_HOST=https://langfuse.example",
        ],
        print_set=(
            "LANGFUSE_ENABLE_TRACE",
            "LANGFUSE_SECRET_KEY",
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_HOST",
        ),
    )
    generated_lines = (
        (tmp_path / ".env.generated").read_text(encoding="utf-8").splitlines()
    )
    assert values["LANGFUSE_ENABLE_TRACE_SET"] == ""
    assert values["LANGFUSE_SECRET_KEY_SET"] == ""
    assert values["LANGFUSE_PUBLIC_KEY_SET"] == ""
    assert values["LANGFUSE_HOST_SET"] == ""
    assert not any(
        line.startswith("LANGFUSE_ENABLE_TRACE=") for line in generated_lines
    )
    assert not any(line.startswith("LANGFUSE_SECRET_KEY=") for line in generated_lines)
    assert not any(line.startswith("LANGFUSE_PUBLIC_KEY=") for line in generated_lines)
    assert not any(line.startswith("LANGFUSE_HOST=") for line in generated_lines)


def test_collect_neo4j_config_bundled_service_keeps_username_editable(
    tmp_path: Path,
) -> None:
    """Bundled Neo4j should preserve editable credentials and existing database overrides."""
    write_text_lines(
        tmp_path / "docker-compose.yml",
        [
            "services:",
            "  lightrag:",
            "    image: example/lightrag:test",
            "    env_file:",
            "      - .env",
        ],
    )
    values = run_collect_bash(
        """
collect_neo4j_config yes
generate_docker_compose "$REPO_ROOT/docker-compose.generated.yml"
printf 'DOCKER_SERVICE=%s\\n' "${DOCKER_SERVICES[0]}"
""",
        tmp_path=tmp_path,
        env_values={
            "NEO4J_USERNAME": "custom-user",
            "NEO4J_PASSWORD": "existing-password",
            "NEO4J_DATABASE": "custom-db",
        },
        prompt_answers={"Neo4j database": "custom-db-2"},
        log_prompts=True,
        stubs="""
confirm_default_yes() { return 0; }
prompt_secret_until_valid_with_default() { printf '%s' "$2"; }
""",
        print_env=("NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE"),
    )
    generated_compose = (tmp_path / "docker-compose.generated.yml").read_text(
        encoding="utf-8"
    )
    assert values["NEO4J_USERNAME"] == "custom-user"
    assert values["NEO4J_PASSWORD"] == "existing-password"
    assert values["NEO4J_DATABASE"] == "custom-db-2"
    assert values["DOCKER_SERVICE"] == "neo4j"
    assert values["PROMPT_LOG"].split("|").count("Neo4j database") == 1
    assert (
        "NEO4J_AUTH: ${NEO4J_USERNAME:?missing}/${NEO4J_PASSWORD:?missing}"
        in generated_compose
    )
    assert 'NEO4J_dbms_default__database: "custom-db-2"' in generated_compose


def test_collect_neo4j_config_bundled_service_defaults_database_when_unset() -> None:
    """Bundled Neo4j should pin the community default database when no prior value exists."""
    values = run_collect_bash(
        "collect_neo4j_config yes",
        log_prompts=True,
        stubs="""
confirm_default_yes() { return 0; }
prompt_secret_until_valid_with_default() { printf 'secure-password'; }
""",
        print_env=("NEO4J_DATABASE",),
    )
    assert values["NEO4J_DATABASE"] == "neo4j"
    assert "Neo4j database" not in values["PROMPT_LOG"].split("|")


def test_collect_neo4j_config_uses_existing_password_as_default_in_external_mode() -> (
    None
):
    """External Neo4j should preserve the existing password when the default is accepted."""
    values = run_collect_bash(
        "collect_neo4j_config no",
        env_values={"NEO4J_PASSWORD": "from-env-password"},
        prompt_answers={},
        until_valid_answers={},
        stubs="""
confirm_default_no() { return 1; }
prompt_secret_with_default() { printf '%s' "$2"; }
""",
        print_env=("NEO4J_PASSWORD",),
    )
    assert values["NEO4J_PASSWORD"] == "from-env-password"


def test_collect_neo4j_config_bundled_service_reprompts_for_empty_credentials() -> None:
    """Bundled Neo4j should reject empty username and password values."""
    values = run_collect_bash(
        """
collect_neo4j_config yes
printf 'USERNAME_CALLS=%s\\n' "$(grep -c '^username$' "$PROMPT_LOG_FILE")"
printf 'PASSWORD_CALLS=%s\\n' "$(grep -c '^password$' "$PROMPT_LOG_FILE")"
""",
        log_prompts=True,
        stubs="""
confirm_default_yes() { return 0; }
prompt_until_valid() {
  local prompt="$1"
  local default="$2"
  local validator="$3"
  shift 3
  local value=""

  while true; do
    if [[ "$prompt" == "Neo4j URI" ]]; then
      value="$default"
    else
      printf 'username\\n' >> "$PROMPT_LOG_FILE"
      if [[ "$(grep -c '^username$' "$PROMPT_LOG_FILE")" -eq 1 ]]; then
        value=""
      else
        value="neo4j-user"
      fi
    fi

    if "$validator" "$value" "$@"; then
      printf '%s' "$value"
      return 0
    fi
  done
}
prompt_secret_until_valid_with_default() {
  local prompt="$1"
  local default="$2"
  local validator="$3"
  shift 3
  local value=""

  while true; do
    printf 'password\\n' >> "$PROMPT_LOG_FILE"
    if [[ "$(grep -c '^password$' "$PROMPT_LOG_FILE")" -eq 1 ]]; then
      value=""
    else
      value="secure-password"
    fi

    if "$validator" "$value" "$@"; then
      printf '%s' "$value"
      return 0
    fi
  done
}
""",
        print_env=("NEO4J_USERNAME", "NEO4J_PASSWORD"),
    )
    assert values["NEO4J_USERNAME"] == "neo4j-user"
    assert values["NEO4J_PASSWORD"] == "secure-password"
    assert values["USERNAME_CALLS"] == "2"
    assert values["PASSWORD_CALLS"] == "2"


def test_collect_neo4j_config_external_service_still_uses_standard_prompts() -> None:
    """External Neo4j setup should keep the non-Docker prompt behavior unchanged."""
    values = run_collect_bash(
        """
collect_neo4j_config no
printf 'USERNAME_PROMPTS=%s\\n' "$(grep -c '^with_default$' "$PROMPT_LOG_FILE")"
printf 'PASSWORD_PROMPTS=%s\\n' "$(grep -c '^secret_with_default$' "$PROMPT_LOG_FILE")"
""",
        log_prompts=True,
        stubs="""
confirm_default_no() { return 1; }
prompt_until_valid() { printf '%s' "$2"; }
prompt_with_default() {
  printf 'with_default\\n' >> "$PROMPT_LOG_FILE"
  if [[ "$1" == "Neo4j username" ]]; then
    printf 'external-user'
  elif [[ "$1" == "Neo4j database" ]]; then
    printf 'external-db'
  else
    printf '%s' "$2"
  fi
}
prompt_secret_with_default() {
  printf 'secret_with_default\\n' >> "$PROMPT_LOG_FILE"
  printf 'external-password'
}
""",
        print_env=("NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE"),
    )
    assert values["NEO4J_USERNAME"] == "external-user"
    assert values["NEO4J_PASSWORD"] == "external-password"
    assert values["NEO4J_DATABASE"] == "external-db"
    assert values["USERNAME_PROMPTS"] == "2"
    assert values["PASSWORD_PROMPTS"] == "1"


@pytest.mark.parametrize(
    ("env_values", "expected_env", "expected_graphlookup_set"),
    [
        (
            {"OPENSEARCH_USE_PPL_GRAPHLOOKUP": "true"},
            {"OPENSEARCH_USE_PPL_GRAPHLOOKUP": "true"},
            "set",
        ),
        (
            {"OPENSEARCH_USE_SSL": "false", "OPENSEARCH_VERIFY_CERTS": "true"},
            {"OPENSEARCH_USE_SSL": "false", "OPENSEARCH_VERIFY_CERTS": "false"},
            "",
        ),
        (
            {},
            {"OPENSEARCH_USE_SSL": "true", "OPENSEARCH_VERIFY_CERTS": "false"},
            "",
        ),
    ],
    ids=[
        "preserves-explicit-graphlookup-override",
        "forces-docker-verify-certs-false",
        "defaults-docker-tls-flags-when-unset",
    ],
)
def test_collect_opensearch_config_docker_flags(
    env_values: dict[str, str],
    expected_env: dict[str, str],
    expected_graphlookup_set: str,
) -> None:
    """collect_opensearch_config should default/force Docker TLS flags and preserve PPL graphlookup overrides."""
    values = run_collect_bash(
        'collect_opensearch_config "yes"',
        env_values=env_values,
        prompt_answers={},
        until_valid_answers={},
        stubs="""
confirm_default_yes() { return 0; }
confirm_default_no() { return 1; }
prompt_secret_until_valid_with_default() { printf '%s' "$2"; }
""",
        print_env=tuple(expected_env),
        print_set=("OPENSEARCH_USE_PPL_GRAPHLOOKUP",),
    )
    for key, expected in expected_env.items():
        assert values[key] == expected
    # PPL graphlookup stays unset (auto-detected) unless explicitly configured.
    assert values["OPENSEARCH_USE_PPL_GRAPHLOOKUP_SET"] == expected_graphlookup_set


def test_collect_opensearch_config_uses_original_index_settings_as_defaults() -> None:
    """collect_opensearch_config should prefer ORIGINAL_ENV_VALUES for shard/replica defaults."""
    values = run_collect_bash(
        'collect_opensearch_config "no"',
        original_env_values={
            "OPENSEARCH_NUMBER_OF_SHARDS": "3",
            "OPENSEARCH_NUMBER_OF_REPLICAS": "2",
        },
        env_values={
            "OPENSEARCH_NUMBER_OF_SHARDS": "9",
            "OPENSEARCH_NUMBER_OF_REPLICAS": "8",
        },
        log_prompts=True,
        log_defaults=True,
        stubs="""
confirm_default_yes() { return 1; }
confirm_default_no() { return 1; }
prompt_secret_until_valid_with_default() { printf '%s' "$2"; }
""",
        print_env=("OPENSEARCH_NUMBER_OF_SHARDS", "OPENSEARCH_NUMBER_OF_REPLICAS"),
    )
    logged = values["PROMPT_LOG"].split("|")
    assert values["OPENSEARCH_NUMBER_OF_SHARDS"] == "3"
    assert values["OPENSEARCH_NUMBER_OF_REPLICAS"] == "2"
    assert "Number of index shards[3]" in logged
    assert "Number of index replicas (use 2 for 3-AZ clusters)[2]" in logged


@pytest.mark.parametrize(
    ("default_docker", "confirm_yes_rc", "expected_validators"),
    [
        (
            "no",
            1,
            (
                "Number of index shards=validate_positive_integer",
                "Number of index replicas (use 2 for 3-AZ clusters)=validate_non_negative_integer",
            ),
        ),
        (
            "yes",
            0,
            ("validate_opensearch_hosts_format",),
        ),
        (
            "yes",
            0,
            ("secret:validate_opensearch_password_strength",),
        ),
    ],
    ids=["index-settings", "hosts", "password"],
)
def test_collect_opensearch_config_validates_inputs_during_prompt(
    default_docker: str,
    confirm_yes_rc: int,
    expected_validators: tuple[str, ...],
) -> None:
    """collect_opensearch_config should attach prompt-time validators for index settings, hosts, and password."""
    values = run_collect_bash(
        f"""
collect_opensearch_config "{default_docker}"
printf 'VALIDATORS=%s\\n' "$(paste -sd '|' "$PROMPT_LOG_FILE")"
""",
        log_prompts=True,
        stubs=f"""
confirm_default_yes() {{ return {confirm_yes_rc}; }}
confirm_default_no() {{ return 1; }}
prompt_until_valid() {{
  printf '%s=%s\\n' "$1" "$3" >> "$PROMPT_LOG_FILE"
  printf '%s' "$2"
}}
prompt_secret_until_valid_with_default() {{
  printf 'secret:%s\\n' "$3" >> "$PROMPT_LOG_FILE"
  printf '%s' "$2"
}}
""",
    )
    for expected in expected_validators:
        assert expected in values["VALIDATORS"]


def test_collect_llm_config_offers_openai_responses() -> None:
    """The Responses binding is a selectable LLM provider with OpenAI defaults."""
    output = run_setup_bash(
        """
OPTIONS_FILE="$(mktemp)"
trap 'rm -f "$OPTIONS_FILE"' EXIT
prompt_choice() {
  shift 2
  printf '%s\\n' "$@" > "$OPTIONS_FILE"
  printf 'openai_responses'
}
prompt_with_default() { printf '%s' "$2"; }
prompt_secret_until_valid_with_default() { printf 'sk-test-key'; }

collect_llm_config

printf 'OFFERED=%s\\n' "$(grep -cx openai_responses "$OPTIONS_FILE")"
printf 'LLM_BINDING=%s\\n' "${ENV_VALUES[LLM_BINDING]}"
printf 'LLM_MODEL=%s\\n' "${ENV_VALUES[LLM_MODEL]}"
printf 'LLM_BINDING_HOST=%s\\n' "${ENV_VALUES[LLM_BINDING_HOST]}"
printf 'LLM_BINDING_API_KEY=%s\\n' "${ENV_VALUES[LLM_BINDING_API_KEY]}"
"""
    )
    values = parse_lines(output)
    assert values["OFFERED"] == "1"
    assert values["LLM_BINDING"] == "openai_responses"
    assert values["LLM_MODEL"] == "gpt-6-luna"
    assert values["LLM_BINDING_HOST"] == "https://api.openai.com/v1"
    assert values["LLM_BINDING_API_KEY"] == "sk-test-key"
    assert "/v1/responses" in output


def test_collect_llm_config_keeps_existing_openai_responses_settings(
    tmp_path: Path,
) -> None:
    """A rerun keeps the binding as the default choice and its model/host/key.

    The real prompt_choice only pre-selects a default that is one of the
    offered options; otherwise Enter is rejected and the user is forced to
    pick another binding. The stub models exactly that: accept the default
    only when it is offered, fail otherwise.
    """
    values = run_collect_bash(
        "collect_llm_config",
        tmp_path=tmp_path,
        env_lines=[
            "LLM_BINDING=openai_responses",
            "LLM_MODEL=gpt-5.6",
            "LLM_BINDING_HOST=https://gateway.example/v1",
            "LLM_BINDING_API_KEY=existing-key",
        ],
        prompt_answers={},
        stubs="""
prompt_choice() {
  local default="$2"
  shift 2
  local option
  for option in "$@"; do
    if [[ "$option" == "$default" ]]; then
      printf '%s' "$default"
      return 0
    fi
  done
  echo "DEFAULT_NOT_OFFERED: $default" >&2
  return 1
}
prompt_secret_until_valid_with_default() { printf '%s' "$2"; }
""",
        print_env=(
            "LLM_BINDING",
            "LLM_MODEL",
            "LLM_BINDING_HOST",
            "LLM_BINDING_API_KEY",
        ),
    )
    assert values["LLM_BINDING"] == "openai_responses"
    assert values["LLM_MODEL"] == "gpt-5.6"
    assert values["LLM_BINDING_HOST"] == "https://gateway.example/v1"
    assert values["LLM_BINDING_API_KEY"] == "existing-key"


def test_collect_embedding_config_reuses_openai_responses_llm_credentials(
    tmp_path: Path,
) -> None:
    """openai_responses has no embedding side; an openai embedding next to it
    is the same provider and reuses the LLM endpoint and key."""
    values = run_collect_bash(
        "collect_embedding_config",
        tmp_path=tmp_path,
        env_lines=[
            "LLM_BINDING=openai_responses",
            "LLM_BINDING_HOST=https://gateway.example/v1",
            "LLM_BINDING_API_KEY=shared-key",
        ],
        prompt_answers={},
        stubs="""
prompt_choice() { printf 'openai'; }
prompt_secret_until_valid_with_default() { printf '%s' "$2"; }
""",
        print_env=(
            "EMBEDDING_BINDING",
            "EMBEDDING_BINDING_HOST",
            "EMBEDDING_BINDING_API_KEY",
        ),
    )
    assert values["EMBEDDING_BINDING"] == "openai"
    assert values["EMBEDDING_BINDING_HOST"] == "https://gateway.example/v1"
    assert values["EMBEDDING_BINDING_API_KEY"] == "shared-key"
