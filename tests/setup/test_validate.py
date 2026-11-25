# Regression tests for interactive setup wizard.
# Classification: keep tests here when they verify validate_* and related checks that accept or reject final env/security/runtime configuration values.

from __future__ import annotations

from pathlib import Path

import pytest

from tests.setup._helpers import (
    FILE_STORAGE_ENV,
    MONGO_STORAGE_ENV,
    OPENSEARCH_STORAGE_ENV,
    parse_lines,
    run_security_check,
    run_setup_bash,
    run_setup_bash_process,
    run_validate_env,
)

pytestmark = pytest.mark.offline


def test_validate_env_file_rejects_missing_ssl_files(tmp_path: Path) -> None:
    """Validation should fail when SSL is enabled with missing cert/key paths."""
    res = run_validate_env(
        tmp_path,
        [
            "SSL=true",
            "SSL_CERTFILE=/missing/cert.pem",
            "SSL_KEYFILE=/missing/key.pem",
            *FILE_STORAGE_ENV,
        ],
        env_example_lines=None,
    )
    assert res.returncode == 1
    assert "Invalid SSL_CERTFILE" in res.stderr
    assert "Invalid SSL_KEYFILE" in res.stderr


@pytest.mark.parametrize(
    ("storage_name", "offered"),
    [
        ("PGTableGraphStorage", True),
        ("PGGraphStorage", False),
    ],
    ids=["pgtable-offered", "age-not-offered-to-new-deployments"],
)
def test_postgres_graph_storage_registration(storage_name: str, offered: bool) -> None:
    """PGTableGraphStorage is offered; deprecated PGGraphStorage stays registered for existing .env files."""
    values = parse_lines(
        run_setup_bash(
            f"""
printf 'OPTIONS=%s\\n' "${{GRAPH_STORAGE_OPTIONS[*]}}"
printf 'REQ=%s\\n' "${{STORAGE_ENV_REQUIREMENTS[{storage_name}]:-}}"
printf 'DB=%s\\n' "${{STORAGE_DB_TYPES[{storage_name}]:-}}"
""",
            reset=False,
        )
    )
    assert (storage_name in values["OPTIONS"].split()) is offered
    assert values["REQ"] == "POSTGRES_USER POSTGRES_PASSWORD POSTGRES_DATABASE"
    assert values["DB"] == "postgresql"


@pytest.mark.parametrize(
    ("existing_graph", "offered"),
    [
        ("PGGraphStorage", True),
        ("PGTableGraphStorage", False),
        ("NetworkXStorage", False),
        ("", False),
    ],
)
def test_age_graph_storage_is_offered_only_to_a_deployment_that_uses_it(
    existing_graph: str, offered: bool
) -> None:
    """An existing PGGraphStorage deployment can keep it: it is offered and it
    is the default, so Enter keeps it. Nobody else is offered it."""
    result = run_setup_bash_process(
        f"""
ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]="{existing_graph}"
prompt_choice() {{
  local prompt="$1" default="$2"
  shift 2
  if [[ "$prompt" == "Graph storage" ]]; then
    printf 'GRAPH_OPTIONS=%s\\n' "$*" >&2
    printf 'GRAPH_DEFAULT=%s\\n' "$default" >&2
  fi
  printf '%s' "$default"
}}
check_storage_compatibility() {{ return 0; }}
select_config_storage() {{ SELECTED_CONFIG_STORAGE=JsonKVStorage; }}
select_storage_backends development
printf 'GRAPH=%s\\n' "${{ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]}}"
"""
    )
    assert result.returncode == 0, result.stderr
    seen = parse_lines(result.stderr)
    assert ("PGGraphStorage" in seen["GRAPH_OPTIONS"].split()) is offered
    expected = existing_graph or "NetworkXStorage"
    assert seen["GRAPH_DEFAULT"] == expected
    assert parse_lines(result.stdout)["GRAPH"] == expected
    output = result.stdout + result.stderr
    assert ("PGGraphStorage (Apache AGE), which is deprecated" in output) is offered
    assert ("lightrag.tools.migrate_graph_storage --apply" in output) is offered


@pytest.mark.parametrize(
    ("runtime_target", "expected_rc", "expected_stderr"),
    [
        (
            "host",
            1,
            ("Invalid SSL_CERTFILE", "Invalid SSL_KEYFILE"),
        ),
        (
            None,
            1,
            ("Invalid SSL_CERTFILE", "Invalid SSL_KEYFILE"),
        ),
        (
            "compose",
            0,
            (),
        ),
    ],
    ids=["host-target", "default-host-target", "compose-target"],
)
def test_validate_env_file_container_ssl_paths_depend_on_runtime_target(
    tmp_path: Path,
    runtime_target: str | None,
    expected_rc: int,
    expected_stderr: tuple[str, ...],
) -> None:
    """Only compose targets may use /app/data/certs/* paths even when staged files exist."""
    (tmp_path / "data" / "certs").mkdir(parents=True)
    (tmp_path / "data" / "certs" / "cert.pem").write_text("cert", encoding="utf-8")
    (tmp_path / "data" / "certs" / "key.pem").write_text("key", encoding="utf-8")
    env_lines = [
        "SSL=true",
        "SSL_CERTFILE=/app/data/certs/cert.pem",
        "SSL_KEYFILE=/app/data/certs/key.pem",
        *([f"LIGHTRAG_RUNTIME_TARGET={runtime_target}"] if runtime_target else []),
        *FILE_STORAGE_ENV,
    ]
    res = run_validate_env(tmp_path, env_lines, env_example_lines=None)
    assert res.returncode == expected_rc
    for fragment in expected_stderr:
        assert fragment in res.stderr


def test_validate_sensitive_env_literals_rejects_interpolation_syntax() -> None:
    """Sensitive values should reject `${...}` so default dotenv interpolation stays safe."""
    output = run_setup_bash(
        """
ENV_VALUES[TOKEN_SECRET]='${JWT_SECRET}'
ENV_VALUES[LIGHTRAG_API_KEY]='plain$token'
ENV_VALUES[WEBUI_DESCRIPTION]='${ALLOWED_MACRO}'

if validate_sensitive_env_literals; then
  printf 'VALID=yes\\n'
else
  printf 'VALID=no\\n'
fi
"""
    )
    assert parse_lines(output)["VALID"] == "no"


def test_validate_env_file_allows_predictable_auth_passwords_and_leaves_them_to_audit(
    tmp_path: Path,
) -> None:
    """validate_env_file should allow risky-but-runnable auth settings."""
    res = run_validate_env(
        tmp_path,
        [
            "AUTH_ACCOUNTS=admin:Passw0rd!",
            "TOKEN_SECRET=jwt-secret",
            *FILE_STORAGE_ENV,
        ],
    )
    assert res.valid
    audit_result = run_security_check(tmp_path)
    assert audit_result.returncode == 1
    assert "AUTH_ACCOUNTS uses a predictable password prefix." in audit_result.stdout


def test_validate_uri_accepts_neo4j_self_signed_tls_scheme() -> None:
    """Neo4j self-signed TLS URIs should pass validation."""
    output = run_setup_bash(
        """
if validate_uri "neo4j+ssc://db.example.com:7687" neo4j; then
  printf 'VALID=yes\\n'
else
  printf 'VALID=no\\n'
fi
""",
        reset=False,
    )
    assert parse_lines(output)["VALID"] == "yes"


def test_validate_security_config_rejects_malformed_auth_accounts() -> None:
    """Security validation should reject auth entries the API cannot parse."""
    output = run_setup_bash(
        """
if validate_security_config "admin" "token-secret" "" no "/health"; then
  printf 'MISSING_COLON=yes\\n'
else
  printf 'MISSING_COLON=no\\n'
fi

if validate_security_config "admin:secret," "token-secret" "" no "/health"; then
  printf 'TRAILING_COMMA=yes\\n'
else
  printf 'TRAILING_COMMA=no\\n'
fi

if validate_security_config "admin:secret,reader:hunter2" "token-secret" "" no "/health"; then
  printf 'VALID_FORMAT=yes\\n'
else
  printf 'VALID_FORMAT=no\\n'
fi

if validate_security_config 'admin:{bcrypt}$2b$12$abcdefghijklmnopqrstuuuuuuuuuuuuuuuuuuuuuuuuuuuu' "token-secret" "" no "/health"; then
  printf 'BCRYPT_FORMAT=yes\\n'
else
  printf 'BCRYPT_FORMAT=no\\n'
fi

if validate_security_config "admin:admin123!" "token-secret" "" no "/health"; then
  printf 'ADMIN_PREFIX=yes\\n'
else
  printf 'ADMIN_PREFIX=no\\n'
fi

if validate_security_config "admin:Passw0rd!" "token-secret" "" no "/health"; then
  printf 'PASS_PREFIX=yes\\n'
else
  printf 'PASS_PREFIX=no\\n'
fi
"""
    )
    values = parse_lines(output)
    assert values["MISSING_COLON"] == "no"
    assert values["TRAILING_COMMA"] == "no"
    assert values["VALID_FORMAT"] == "yes"
    assert values["BCRYPT_FORMAT"] == "yes"
    assert values["ADMIN_PREFIX"] == "no"
    assert values["PASS_PREFIX"] == "no"


@pytest.mark.parametrize(
    ("extra_lines", "expected_valid", "expected_stderr"),
    [
        (
            [
                "LIGHTRAG_GRAPH_STORAGE=Neo4JStorage",
                "NEO4J_URI=http://localhost:7687",
                "NEO4J_USERNAME=neo4j",
                "NEO4J_PASSWORD=secret",
            ],
            False,
            "Invalid NEO4J_URI",
        ),
        # Both Redis cases explicitly pin JSON configuration while testing
        # the business Redis URI scheme.
        (
            [
                "LIGHTRAG_KV_STORAGE=RedisKVStorage",
                "LIGHTRAG_CONFIG_STORAGE=JsonKVStorage",
                "REDIS_URI=tcp://localhost:6379",
            ],
            False,
            "Invalid REDIS_URI",
        ),
        (
            [
                "LIGHTRAG_KV_STORAGE=RedisKVStorage",
                "LIGHTRAG_CONFIG_STORAGE=JsonKVStorage",
                "REDIS_URI=rediss://localhost:6380",
            ],
            True,
            "",
        ),
    ],
    ids=["invalid-neo4j-scheme", "invalid-redis-scheme", "valid-rediss-scheme"],
)
def test_validate_env_file_handles_supported_and_unsupported_uri_schemes(
    tmp_path: Path,
    extra_lines: list[str],
    expected_valid: bool,
    expected_stderr: str,
) -> None:
    """validate_env_file should reject malformed schemes and allow supported TLS variants."""
    res = run_validate_env(tmp_path, [*FILE_STORAGE_ENV, *extra_lines])
    assert res.valid is expected_valid
    if expected_stderr:
        assert expected_stderr in res.stderr


def test_validate_env_file_rejects_invalid_runtime_target(tmp_path: Path) -> None:
    """validate_env_file should reject unsupported LIGHTRAG_RUNTIME_TARGET values."""
    res = run_validate_env(
        tmp_path, ["LIGHTRAG_RUNTIME_TARGET=laptop", *FILE_STORAGE_ENV]
    )
    assert not res.valid
    assert "Invalid LIGHTRAG_RUNTIME_TARGET" in res.stderr


def test_validate_required_variables_requires_opensearch_basic_auth() -> None:
    """OpenSearch storages should require both OPENSEARCH_USER and OPENSEARCH_PASSWORD."""
    values = parse_lines(
        run_setup_bash(
            """
ENV_VALUES[LIGHTRAG_KV_STORAGE]="OpenSearchKVStorage"
ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]="OpenSearchVectorDBStorage"
ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]="OpenSearchGraphStorage"
ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]="OpenSearchDocStatusStorage"
ENV_VALUES[OPENSEARCH_HOSTS]="localhost:9200"

if validate_required_variables \
  "${ENV_VALUES[LIGHTRAG_KV_STORAGE]}" \
  "${ENV_VALUES[LIGHTRAG_VECTOR_STORAGE]}" \
  "${ENV_VALUES[LIGHTRAG_GRAPH_STORAGE]}" \
  "${ENV_VALUES[LIGHTRAG_DOC_STATUS_STORAGE]}"; then
  printf 'VALID=yes\\n'
else
  printf 'VALID=no\\n'
fi
"""
        )
    )
    assert values["VALID"] == "no"


@pytest.mark.parametrize(
    ("extra_lines", "expected_stderr"),
    [
        (
            [
                "OPENSEARCH_HOSTS=localhost:9200",
                "OPENSEARCH_USER=admin",
                "OPENSEARCH_PASSWORD=StrongPass1!",
                "OPENSEARCH_NUMBER_OF_SHARDS=abc",
                "OPENSEARCH_NUMBER_OF_REPLICAS=-1",
            ],
            ("OPENSEARCH_NUMBER_OF_SHARDS must be a positive integer.",),
        ),
        (
            [
                "OPENSEARCH_HOSTS=localhost:9200",
                "OPENSEARCH_USER=admin",
                "OPENSEARCH_PASSWORD=StrongPass1!",
                "OPENSEARCH_NUMBER_OF_SHARDS=",
                "OPENSEARCH_NUMBER_OF_REPLICAS=",
            ],
            ("OPENSEARCH_NUMBER_OF_SHARDS must be a positive integer.",),
        ),
        (
            ["OPENSEARCH_HOSTS="],
            ("Empty OPENSEARCH_HOSTS",),
        ),
        (
            [
                "OPENSEARCH_HOSTS=   ,   ",
                "OPENSEARCH_USER=admin",
                "OPENSEARCH_PASSWORD=StrongPass1!",
            ],
            ("OPENSEARCH_HOSTS must not contain empty host entries.",),
        ),
        (
            [
                "LIGHTRAG_SETUP_OPENSEARCH_DEPLOYMENT=docker",
                "OPENSEARCH_HOSTS=localhost:9200",
                "OPENSEARCH_USER=admin",
            ],
            ("Bundled OpenSearch requires OPENSEARCH_USER and OPENSEARCH_PASSWORD",),
        ),
        (
            [
                "OPENSEARCH_HOSTS=localhost:9200",
                "OPENSEARCH_USER=admin",
                "OPENSEARCH_PASSWORD=weakpass",
            ],
            ("OpenSearch requires a strong OPENSEARCH_PASSWORD",),
        ),
        (
            ["OPENSEARCH_HOSTS=localhost:9200"],
            ("OPENSEARCH_USER", "OPENSEARCH_PASSWORD"),
        ),
        (
            [
                "OPENSEARCH_HOSTS=https://localhost:9200",
                "OPENSEARCH_USER=admin",
                "OPENSEARCH_PASSWORD=StrongPass1!",
            ],
            ("OPENSEARCH_HOSTS must use bare host:port entries, not URLs.",),
        ),
    ],
    ids=[
        "invalid-index-settings",
        "blank-index-settings",
        "empty-hosts",
        "whitespace-only-hosts",
        "docker-without-password",
        "weak-host-password",
        "unauthenticated-host",
        "hosts-with-uri-scheme",
    ],
)
def test_validate_env_file_rejects_invalid_opensearch_settings(
    tmp_path: Path,
    extra_lines: list[str],
    expected_stderr: tuple[str, ...],
) -> None:
    """validate_env_file should reject invalid OpenSearch index, host, and auth settings."""
    res = run_validate_env(tmp_path, [*OPENSEARCH_STORAGE_ENV, *extra_lines])
    assert not res.valid
    for fragment in expected_stderr:
        assert fragment in res.stderr


@pytest.mark.parametrize(
    ("env_lines", "expected_valid", "expected_stderr"),
    [
        (
            [
                "LIGHTRAG_KV_STORAGE=JsonKVStorage",
                "LIGHTRAG_VECTOR_STORAGE=MongoVectorDBStorage",
                "LIGHTRAG_GRAPH_STORAGE=NetworkXStorage",
                "LIGHTRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage",
                "MONGO_URI=mongodb://localhost:27017",
            ],
            False,
            "MongoVectorDBStorage requires an Atlas-capable MongoDB URI",
        ),
        (
            [
                "LIGHTRAG_SETUP_MONGODB_DEPLOYMENT=docker",
                *MONGO_STORAGE_ENV,
                "MONGO_URI=mongodb://localhost:27017/?directConnection=true",
                "MONGO_DATABASE=LightRAG",
            ],
            True,
            "",
        ),
        (
            [
                *MONGO_STORAGE_ENV,
                "MONGO_URI=mongodb://atlas-local.example.com:27017/LightRAG?replicaSet=rs0&directConnection=true",
                "MONGO_DATABASE=LightRAG",
            ],
            True,
            "",
        ),
        (
            [
                "LIGHTRAG_SETUP_MONGODB_DEPLOYMENT=docker",
                *MONGO_STORAGE_ENV,
                "MONGO_URI=mongodb://mongo.example.com:27017/?directConnection=true",
                "MONGO_DATABASE=LightRAG",
            ],
            False,
            "MongoVectorDBStorage requires the bundled Atlas Local endpoint",
        ),
        (
            [
                "LIGHTRAG_SETUP_MONGODB_DEPLOYMENT=docker",
                *MONGO_STORAGE_ENV,
                "MONGO_URI=mongodb://localhost:27017/",
                "MONGO_DATABASE=LightRAG",
            ],
            False,
            "MongoVectorDBStorage requires the bundled Atlas Local endpoint",
        ),
        (
            [
                "LIGHTRAG_SETUP_MONGODB_DEPLOYMENT=docker",
                *MONGO_STORAGE_ENV,
                "MONGO_URI=mongodb://localhost:9999/?directConnection=true",
                "MONGO_DATABASE=LightRAG",
            ],
            False,
            "MongoVectorDBStorage requires the bundled Atlas Local endpoint",
        ),
    ],
    ids=[
        "without-atlas-capable-uri",
        "wizard-managed-atlas-local",
        "external-atlas-local",
        "remote-uri-with-docker-marker",
        "stale-local-uri-without-direct-connection",
        "wrong-local-port-with-docker-marker",
    ],
)
def test_validate_env_file_mongo_vector_storage_uri_rules(
    tmp_path: Path,
    env_lines: list[str],
    expected_valid: bool,
    expected_stderr: str,
) -> None:
    """validate_env_file should enforce Atlas/Atlas Local URI rules for MongoVectorDBStorage."""
    res = run_validate_env(tmp_path, env_lines)
    assert res.valid is expected_valid
    if expected_stderr:
        assert expected_stderr in res.stderr


@pytest.mark.parametrize(
    ("unused_lines", "forbidden_stderr"),
    [
        (
            [
                "NEO4J_URI=http://localhost:7687",
                "MONGO_URI=not-a-mongo-uri",
                "REDIS_URI=tcp://localhost:6379",
                "MILVUS_URI=tcp://localhost:19530",
                "QDRANT_URL=tcp://localhost:6333",
                "MEMGRAPH_URI=http://localhost:7687",
                "POSTGRES_PORT=99999",
            ],
            (
                "Invalid NEO4J_URI",
                "Invalid MONGO_URI",
                "Invalid REDIS_URI",
                "Invalid MILVUS_URI",
                "Invalid QDRANT_URL",
                "Invalid MEMGRAPH_URI",
                "Invalid POSTGRES_PORT",
            ),
        ),
        (
            ["OPENSEARCH_HOSTS="],
            ("Empty OPENSEARCH_HOSTS",),
        ),
    ],
    ids=["other-backends", "empty-opensearch-hosts"],
)
def test_validate_env_file_ignores_invalid_unused_storage_settings(
    tmp_path: Path,
    unused_lines: list[str],
    forbidden_stderr: tuple[str, ...],
) -> None:
    """validate_env_file should ignore malformed settings for backends not selected by storage."""
    res = run_validate_env(tmp_path, [*FILE_STORAGE_ENV, *unused_lines])
    assert res.valid
    for fragment in forbidden_stderr:
        assert fragment not in res.stderr
