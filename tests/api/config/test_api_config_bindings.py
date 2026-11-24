"""Binding configuration: Bedrock auth/aliases, CLI binding selection, default hosts, LoLLMs host kwargs, OpenAI Responses wiring, and embedding dimension guards."""

from __future__ import annotations

import asyncio
import re
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from lightrag.api.config import get_default_host, parse_args
from lightrag.llm.binding_options import OpenAIResponsesLLMOptions
from lightrag.llm_roles import ROLES

pytestmark = pytest.mark.offline


# --------------------------------------------------------------------------- #
# Bedrock binding aliases and credential validation
# --------------------------------------------------------------------------- #


class TestBedrockConfig:
    @pytest.fixture(autouse=True)
    def _clear_bedrock_auth_env(self, monkeypatch):
        for key in (
            "LLM_BINDING",
            "EMBEDDING_BINDING",
            "QUERY_LLM_BINDING",
            "QUERY_LLM_BINDING_API_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_BEARER_TOKEN_BEDROCK",
            "AWS_ROLE_ARN",
            "AWS_WEB_IDENTITY_TOKEN_FILE",
            "QUERY_AWS_ACCESS_KEY_ID",
            "QUERY_AWS_SECRET_ACCESS_KEY",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(sys, "argv", ["lightrag-server"])

    def test_bedrock_env_binding_alias_is_normalized(self, monkeypatch):
        monkeypatch.setenv("LLM_BINDING", "aws_bedrock")
        monkeypatch.setenv("EMBEDDING_BINDING", "aws_bedrock")
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "absk-test")

        args = parse_args()

        assert args.llm_binding == "bedrock"
        assert args.embedding_binding == "bedrock"

    def test_bedrock_cli_binding_alias_is_not_supported(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["lightrag-server", "--llm-binding", "aws_bedrock"]
        )

        with pytest.raises(SystemExit):
            parse_args()

    def test_bedrock_role_env_binding_alias_is_normalized(self, monkeypatch):
        monkeypatch.setenv("QUERY_LLM_BINDING", "aws_bedrock")
        monkeypatch.setenv("QUERY_LLM_MODEL", "us.amazon.nova-lite-v1:0")
        monkeypatch.setenv("QUERY_AWS_REGION", "us-west-2")
        monkeypatch.setenv("QUERY_AWS_ACCESS_KEY_ID", "akid")
        monkeypatch.setenv("QUERY_AWS_SECRET_ACCESS_KEY", "secret")

        args = parse_args()

        assert args.query_llm_binding == "bedrock"
        assert args.query_aws_region == "us-west-2"

    def test_bedrock_role_api_key_is_rejected(self, monkeypatch):
        monkeypatch.setenv("QUERY_LLM_BINDING", "bedrock")
        monkeypatch.setenv("QUERY_LLM_MODEL", "us.amazon.nova-lite-v1:0")
        monkeypatch.setenv("QUERY_LLM_BINDING_API_KEY", "absk-role")

        with pytest.raises(
            SystemExit, match="does not support QUERY_LLM_BINDING_API_KEY"
        ):
            parse_args()

    def test_bedrock_binding_accepts_default_aws_credential_provider_chain(
        self, monkeypatch
    ):
        monkeypatch.setenv("LLM_BINDING", "bedrock")
        monkeypatch.setenv("EMBEDDING_BINDING", "ollama")
        monkeypatch.setenv(
            "AWS_ROLE_ARN", "arn:aws:iam::123456789012:role/lightrag-bedrock"
        )
        monkeypatch.setenv(
            "AWS_WEB_IDENTITY_TOKEN_FILE",
            "/var/run/secrets/eks.amazonaws.com/serviceaccount/token",
        )

        assert parse_args().llm_binding == "bedrock"

    def test_bedrock_role_accepts_default_aws_credential_provider_chain(
        self, monkeypatch
    ):
        monkeypatch.setenv("QUERY_LLM_BINDING", "bedrock")
        monkeypatch.setenv("QUERY_LLM_MODEL", "us.amazon.nova-lite-v1:0")
        monkeypatch.setenv(
            "AWS_ROLE_ARN", "arn:aws:iam::123456789012:role/lightrag-bedrock"
        )
        monkeypatch.setenv(
            "AWS_WEB_IDENTITY_TOKEN_FILE",
            "/var/run/secrets/eks.amazonaws.com/serviceaccount/token",
        )

        assert parse_args().query_llm_binding == "bedrock"

    @pytest.mark.parametrize("bearer_token", [None, "absk-test"])
    def test_bedrock_binding_rejects_partial_sigv4_pair(
        self, monkeypatch, bearer_token
    ):
        monkeypatch.setenv("LLM_BINDING", "bedrock")
        monkeypatch.setenv("EMBEDDING_BINDING", "ollama")
        if bearer_token:
            monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", bearer_token)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "akid")

        with pytest.raises(ValueError, match="Bedrock LLM binding requires"):
            parse_args()

    def test_bedrock_binding_accepts_sigv4_pair(self, monkeypatch):
        monkeypatch.setenv("LLM_BINDING", "bedrock")
        monkeypatch.setenv("EMBEDDING_BINDING", "ollama")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "akid")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")

        assert parse_args().llm_binding == "bedrock"

    @pytest.mark.parametrize("bearer_token", [None, "absk-test"])
    def test_bedrock_role_requires_complete_role_sigv4_pair(
        self, monkeypatch, bearer_token
    ):
        monkeypatch.setenv("QUERY_LLM_BINDING", "bedrock")
        monkeypatch.setenv("QUERY_LLM_MODEL", "us.amazon.nova-lite-v1:0")
        if bearer_token:
            monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", bearer_token)
        monkeypatch.setenv("QUERY_AWS_ACCESS_KEY_ID", "akid")

        with pytest.raises(ValueError, match="Bedrock role 'query' requires"):
            parse_args()


# --------------------------------------------------------------------------- #
# CLI binding selector pre-parse
# --------------------------------------------------------------------------- #


class TestBindingSelection:
    @pytest.mark.parametrize(
        "selector,options,expected_binding,expected_option",
        [
            (
                ["--llm-binding=openai"],
                ["--openai-llm-temperature", "0.25"],
                "openai",
                "openai_llm_temperature",
            ),
            (
                ["--llm-binding", "openai"],
                ["--openai-llm-temperature", "0.25"],
                "openai",
                "openai_llm_temperature",
            ),
            (
                ["--llm-binding", "ollama", "--llm-binding=openai"],
                ["--openai-llm-temperature", "0.25"],
                "openai",
                "openai_llm_temperature",
            ),
            (
                ["--llm-binding", "ollama", "--llm-binding", "openai"],
                ["--openai-llm-temperature", "0.25"],
                "openai",
                "openai_llm_temperature",
            ),
            (
                ["--llm-bind", "openai"],
                ["--openai-llm-temperature", "0.25"],
                "openai",
                "openai_llm_temperature",
            ),
            (
                ["--llm-b=openai"],
                ["--openai-llm-temperature", "0.25"],
                "openai",
                "openai_llm_temperature",
            ),
        ],
    )
    def test_llm_options_follow_effective_cli_binding(
        self, monkeypatch, selector, options, expected_binding, expected_option
    ):
        monkeypatch.setenv("LLM_BINDING", "ollama")
        monkeypatch.setenv("EMBEDDING_BINDING", "ollama")
        monkeypatch.setattr(sys, "argv", ["lightrag-server", *selector, *options])

        args = parse_args()

        assert args.llm_binding == expected_binding
        assert getattr(args, expected_option) == 0.25

    @pytest.mark.parametrize(
        "selector",
        [
            ["--embedding-binding=ollama"],
            ["--embedding-binding", "ollama"],
            ["--embedding-binding", "openai", "--embedding-binding=ollama"],
            ["--embedding-binding", "openai", "--embedding-binding", "ollama"],
            ["--embedding-bind", "ollama"],
        ],
    )
    def test_embedding_options_follow_effective_cli_binding(
        self, monkeypatch, selector
    ):
        monkeypatch.setenv("LLM_BINDING", "openai")
        monkeypatch.setenv("EMBEDDING_BINDING", "openai")
        monkeypatch.setattr(
            sys,
            "argv",
            ["lightrag-server", *selector, "--ollama-embedding-num_ctx", "4096"],
        )

        args = parse_args()

        assert args.embedding_binding == "ollama"
        assert args.ollama_embedding_num_ctx == 4096

    @pytest.mark.parametrize(
        "selector",
        [
            ["--llm-binding=openai"],
            ["--llm-binding", "ollama", "--llm-binding", "openai"],
            ["--llm-bind", "openai"],
        ],
    )
    def test_env_only_llm_options_follow_effective_cli_binding(
        self, monkeypatch, selector
    ):
        monkeypatch.setenv("LLM_BINDING", "ollama")
        monkeypatch.setenv("EMBEDDING_BINDING", "ollama")
        monkeypatch.setenv("OPENAI_LLM_TEMPERATURE", "0.3")
        monkeypatch.setattr(sys, "argv", ["lightrag-server", *selector])

        args = parse_args()

        assert args.llm_binding == "openai"
        assert args.openai_llm_temperature == 0.3

    def test_env_only_embedding_options_follow_effective_cli_binding(self, monkeypatch):
        monkeypatch.setenv("LLM_BINDING", "openai")
        monkeypatch.setenv("EMBEDDING_BINDING", "openai")
        monkeypatch.setenv("OLLAMA_EMBEDDING_NUM_CTX", "4096")
        monkeypatch.setattr(
            sys, "argv", ["lightrag-server", "--embedding-binding=ollama"]
        )

        args = parse_args()

        assert args.embedding_binding == "ollama"
        assert args.ollama_embedding_num_ctx == 4096

    @pytest.mark.parametrize(
        "argv",
        [
            ["--llm-binding"],
            ["--llm-binding", "--host", "0.0.0.0"],
            ["--embedding-binding"],
        ],
    )
    def test_malformed_selector_is_reported_by_the_full_parser(
        self, monkeypatch, capsys, argv
    ):
        monkeypatch.setenv("LLM_BINDING", "ollama")
        monkeypatch.setenv("EMBEDDING_BINDING", "ollama")
        monkeypatch.setattr(sys, "argv", ["lightrag-server", *argv])

        with pytest.raises(SystemExit) as exc_info:
            parse_args()

        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "expected one argument" in err
        assert "--working-dir" in err


# --------------------------------------------------------------------------- #
# Default hosts & role host resolution
# --------------------------------------------------------------------------- #


class TestDefaultHost:
    @pytest.mark.parametrize(
        ("binding", "default"),
        [
            ("gemini", "DEFAULT_GEMINI_ENDPOINT"),
            ("bedrock", "DEFAULT_BEDROCK_ENDPOINT"),
            ("openai_responses", "https://api.openai.com/v1"),
        ],
    )
    def test_default_host_and_its_override(self, monkeypatch, binding, default):
        monkeypatch.delenv("LLM_BINDING_HOST", raising=False)
        assert get_default_host(binding) == default

        monkeypatch.setenv("LLM_BINDING_HOST", "https://gateway.example/v1")
        assert get_default_host(binding) == "https://gateway.example/v1"

    def test_azure_default_ignores_llm_host(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://azure.example")
        monkeypatch.setenv("LLM_BINDING_HOST", "https://gateway.example/v1")

        assert get_default_host("azure_openai") == "https://azure.example"

    @pytest.mark.parametrize(
        ("binding", "default", "api_key", "role_host"),
        [
            ("gemini", "DEFAULT_GEMINI_ENDPOINT", "gemini-test-key", None),
            ("bedrock", "DEFAULT_BEDROCK_ENDPOINT", None, None),
            (
                "gemini",
                "https://role-gateway.example",
                "gemini-test-key",
                "https://role-gateway.example",
            ),
            ("openai", None, None, None),
        ],
    )
    def test_role_host_respects_provider_and_explicit_override(
        self, monkeypatch, lightrag_log_records, binding, default, api_key, role_host
    ):
        monkeypatch.setattr(sys, "argv", ["lightrag-server"])
        for spec in ROLES:
            for suffix in (
                "LLM_BINDING",
                "LLM_MODEL",
                "LLM_BINDING_HOST",
                "LLM_BINDING_API_KEY",
            ):
                monkeypatch.delenv(f"{spec.env_prefix}_{suffix}", raising=False)
        monkeypatch.setenv("VLM_PROCESS_ENABLE", "false")
        monkeypatch.setenv("LLM_BINDING", "openai")
        monkeypatch.setenv("LLM_BINDING_HOST", "https://base-gateway.example/v1")
        monkeypatch.setenv("QUERY_LLM_BINDING", binding)
        monkeypatch.setenv("QUERY_LLM_MODEL", "role-specific-model")
        if api_key is not None:
            monkeypatch.setenv("QUERY_LLM_BINDING_API_KEY", api_key)
        if role_host is not None:
            monkeypatch.setenv("QUERY_LLM_BINDING_HOST", role_host)

        args = parse_args()

        assert args.llm_binding_host == "https://base-gateway.example/v1"
        assert args.query_llm_binding_host == default
        warned = [
            r
            for r in lightrag_log_records
            if r.levelname == "WARNING" and "QUERY_LLM_BINDING_HOST" in r.getMessage()
        ]
        assert len(warned) == (1 if role_host is None and binding != "openai" else 0)


# --------------------------------------------------------------------------- #
# _create_llm_model_kwargs(): LoLLMs base_url vs Ollama host
# --------------------------------------------------------------------------- #

_LOLLMS_ENV_VARS = (
    "LLM_BINDING",
    "LLM_BINDING_HOST",
    "LLM_BINDING_API_KEY",
    "LLM_MODEL",
    "VLM_PROCESS_ENABLE",
    *(
        f"{spec.env_prefix}_{suffix}"
        for spec in ROLES
        for suffix in (
            "LLM_BINDING",
            "LLM_BINDING_HOST",
            "LLM_BINDING_API_KEY",
            "LLM_MODEL",
        )
    ),
)


class TestLollmsHostKwargs:
    @pytest.fixture(autouse=True)
    def _isolated_config_env(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["lightrag-server"])
        import lightrag.api.lightrag_server  # noqa: F401

        for var in _LOLLMS_ENV_VARS:
            monkeypatch.delenv(var, raising=False)

    @pytest.fixture
    def create_llm_model_kwargs(self):
        from lightrag.api.lightrag_server import _create_llm_model_kwargs

        return _create_llm_model_kwargs

    @pytest.mark.parametrize(
        "binding, expected_key, other_key, host",
        [
            ("lollms", "base_url", "host", "http://myserver:9999"),
            ("ollama", "host", "base_url", "http://myserver:11434"),
        ],
    )
    def test_custom_host_uses_the_binding_specific_key(
        self, monkeypatch, create_llm_model_kwargs, binding, expected_key, other_key, host
    ):
        monkeypatch.setenv("LLM_BINDING", binding)
        monkeypatch.setenv("LLM_BINDING_HOST", host)

        kwargs = create_llm_model_kwargs(binding, parse_args(), llm_timeout=30)

        assert kwargs[expected_key] == host
        assert other_key not in kwargs

    def test_default_lollms_host_and_metadata_kwargs(
        self, monkeypatch, create_llm_model_kwargs
    ):
        monkeypatch.setenv("LLM_BINDING", "lollms")
        monkeypatch.delenv("LLM_BINDING_HOST", raising=False)
        monkeypatch.setenv("LLM_BINDING_API_KEY", "secret")

        kwargs = create_llm_model_kwargs("lollms", parse_args(), llm_timeout=42)

        assert kwargs["base_url"] == "http://localhost:9600"
        assert "host" not in kwargs
        assert kwargs["timeout"] == 42
        assert kwargs["api_key"] == "secret"
        assert kwargs["options"] == {}

    def test_unsupported_binding_returns_empty_dict(
        self, monkeypatch, create_llm_model_kwargs
    ):
        monkeypatch.setenv("LLM_BINDING", "openai")
        assert create_llm_model_kwargs("openai", parse_args(), llm_timeout=30) == {}

    def test_configured_lollms_host_reaches_the_actual_request(
        self, monkeypatch, create_llm_model_kwargs
    ):
        monkeypatch.setenv("LLM_BINDING", "lollms")
        monkeypatch.setenv("LLM_BINDING_HOST", "http://myserver:9999")

        model_kwargs = create_llm_model_kwargs("lollms", parse_args(), llm_timeout=30)

        from lightrag.llm.lollms import lollms_model_complete

        captured = {}

        class FakeResponse:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def text(self):
                return "ok"

        class FakeSession:
            def __init__(self, timeout=None, headers=None, *, raise_for_status=False):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            def post(self, url, json=None):
                captured["url"] = url
                return FakeResponse()

        class FakeHashingKV:
            global_config = {"llm_model_name": "fake-model"}

        with patch("lightrag.llm.lollms.aiohttp.ClientSession", FakeSession):
            asyncio.run(
                lollms_model_complete(
                    "hello", hashing_kv=FakeHashingKV(), **model_kwargs
                )
            )

        assert captured["url"] == "http://myserver:9999/lollms_generate"


# --------------------------------------------------------------------------- #
# openai_responses binding config wiring
# --------------------------------------------------------------------------- #

_OPENAI_RESPONSES_AMBIENT_ENV = (
    "LLM_BINDING",
    "VLM_PROCESS_ENABLE",
    "OPENAI_LLM_MAX_COMPLETION_TOKENS",
    "OPENAI_LLM_REASONING_EFFORT",
    "OPENAI_RESPONSES_LLM_MAX_OUTPUT_TOKENS",
    "OPENAI_RESPONSES_LLM_REASONING_EFFORT",
)


class TestOpenAIResponsesConfig:
    @pytest.fixture
    def clean_env(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["lightrag-server"])
        for name in _OPENAI_RESPONSES_AMBIENT_ENV:
            monkeypatch.delenv(name, raising=False)
        return monkeypatch

    def test_binding_is_accepted_and_options_isolated_from_chat_completions(
        self, clean_env
    ):
        clean_env.setenv("LLM_BINDING", "openai_responses")
        clean_env.setenv("OPENAI_RESPONSES_LLM_MAX_OUTPUT_TOKENS", "4096")
        clean_env.setenv("OPENAI_LLM_MAX_COMPLETION_TOKENS", "999")

        args = parse_args()
        options = OpenAIResponsesLLMOptions.options_dict(args)

        assert args.llm_binding == "openai_responses"
        assert options["max_output_tokens"] == 4096
        assert "max_completion_tokens" not in options

    def test_chat_completions_max_tokens_does_not_leak_when_responses_unset(
        self, clean_env
    ):
        clean_env.setenv("LLM_BINDING", "openai_responses")
        clean_env.setenv("OPENAI_LLM_MAX_COMPLETION_TOKENS", "999")

        options = OpenAIResponsesLLMOptions.options_dict(parse_args())

        assert "max_completion_tokens" not in options
        assert options.get("max_output_tokens") != 999

    def test_startup_routes_the_binding_to_the_responses_driver(
        self, clean_env, prepare_startup
    ):
        import lightrag.llm.openai as chat_completions
        import lightrag.llm.openai_responses as responses

        calls: list[str] = []

        async def fake_responses(model, prompt, **kwargs):
            calls.append("responses")
            return "ok"

        async def fake_chat_completions(model, prompt, **kwargs):
            calls.append("chat_completions")
            return "ok"

        clean_env.setattr(
            responses, "openai_responses_complete_if_cache", fake_responses
        )
        clean_env.setattr(
            chat_completions, "openai_complete_if_cache", fake_chat_completions
        )
        startup = prepare_startup(
            {"LLM_BINDING": "openai_responses", "LLM_BINDING_API_KEY": "a-key"}
        )

        llm_model_func = startup.constructor_kwargs()["llm_model_func"]
        asyncio.run(llm_model_func("hello"))

        assert calls == ["responses"]


# --------------------------------------------------------------------------- #
# Startup validation for custom embedding dimensions across all bindings
# --------------------------------------------------------------------------- #


def _create_embedding_function(
    *, binding: str, model: str | None, embedding_dim: int | None
):
    with patch.object(sys, "argv", ["lightrag-server"]):
        from lightrag.api.lightrag_server import create_optimized_embedding_function

    args = SimpleNamespace(
        embedding_dim=embedding_dim,
        embedding_token_limit=None,
        embedding_asymmetric=False,
        embedding_asymmetric_configured=False,
        embedding_query_prefix_configured=False,
        embedding_document_prefix_configured=False,
    )
    return create_optimized_embedding_function(
        config_cache=MagicMock(ollama_embedding_options=None),
        binding=binding,
        model=model,
        host="http://localhost:11434",
        api_key="test-key",
        args=args,
    )


class TestEmbeddingDimensionGuard:
    @pytest.mark.parametrize(
        ("binding", "model"),
        [
            ("ollama", "nomic-embed-text"),
            ("openai", "text-embedding-3-large"),
            ("jina", "jina-embeddings-v3"),
            ("gemini", "text-embedding-004"),
            ("bedrock", "cohere.embed-english-v3"),
            ("voyageai", "voyage-3-lite"),
        ],
    )
    def test_custom_model_requires_explicit_dim(self, binding, model):
        with pytest.raises(ValueError, match=rf"EMBEDDING_DIM.*{re.escape(model)}"):
            _create_embedding_function(
                binding=binding, model=model, embedding_dim=None
            )

    @pytest.mark.parametrize(
        ("binding", "model", "provider_dim"),
        [
            ("ollama", None, 1024),
            ("ollama", "bge-m3", 1024),
            ("ollama", "bge-m3:latest", 1024),
            ("openai", None, 1536),
            ("jina", None, 2048),
            ("gemini", None, 1536),
            ("bedrock", None, 1024),
            ("voyageai", None, 1024),
        ],
    )
    def test_default_model_keeps_provider_dimension(
        self, binding, model, provider_dim
    ):
        func = _create_embedding_function(
            binding=binding, model=model, embedding_dim=None
        )
        assert func.embedding_dim == provider_dim

    def test_custom_model_accepts_explicit_dim(self):
        func = _create_embedding_function(
            binding="openai",
            model="text-embedding-3-large",
            embedding_dim=3072,
        )
        assert func.embedding_dim == 3072

    def test_azure_any_model_requires_explicit_dim(self):
        with pytest.raises(ValueError, match=r"EMBEDDING_DIM.*Azure OpenAI"):
            _create_embedding_function(
                binding="azure_openai",
                model="my-custom-embedding-deployment",
                embedding_dim=None,
            )

    def test_azure_configured_model_accepts_explicit_dim(self):
        func = _create_embedding_function(
            binding="azure_openai",
            model="my-custom-embedding-deployment",
            embedding_dim=1536,
        )
        assert func.embedding_dim == 1536

    def test_azure_env_deployment_without_model_requires_explicit_dim(self):
        with patch.dict(
            "os.environ", {"AZURE_EMBEDDING_DEPLOYMENT": "text-embedding-3-large"}
        ):
            with pytest.raises(ValueError, match=r"EMBEDDING_DIM.*Azure OpenAI"):
                _create_embedding_function(
                    binding="azure_openai",
                    model=None,
                    embedding_dim=None,
                )

    def test_azure_message_names_the_deployment_that_wins_at_runtime(self):
        with patch.dict("os.environ", {"AZURE_EMBEDDING_DEPLOYMENT": "env-deployment"}):
            with pytest.raises(ValueError) as excinfo:
                _create_embedding_function(
                    binding="azure_openai",
                    model="model-deployment",
                    embedding_dim=None,
                )
        assert "env-deployment" in str(excinfo.value)
        assert "model-deployment" not in str(excinfo.value)

    @pytest.mark.parametrize(
        ("binding", "model", "match"),
        [
            ("openai", "text-embedding-3-large", r"EMBEDDING_DIM.*text-embedding-3-large"),
            (
                "azure_openai",
                "my-custom-embedding-deployment",
                r"EMBEDDING_DIM.*Azure OpenAI",
            ),
        ],
    )
    def test_zero_dim_does_not_satisfy_guard(self, binding, model, match):
        with pytest.raises(ValueError, match=match):
            _create_embedding_function(binding=binding, model=model, embedding_dim=0)

    @pytest.mark.parametrize("value", ["0", "-1"])
    def test_config_layer_rejects_non_positive_dim(self, monkeypatch, value):
        monkeypatch.setattr(sys, "argv", ["lightrag-server"])
        monkeypatch.setenv("EMBEDDING_DIM", value)

        with pytest.raises(SystemExit, match=r"EMBEDDING_DIM must be a positive"):
            parse_args()

    def test_config_layer_accepts_positive_dim(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["lightrag-server"])
        monkeypatch.setenv("EMBEDDING_DIM", "1024")

        assert parse_args().embedding_dim == 1024

    def test_lollms_has_no_dim_guard(self):
        func = _create_embedding_function(
            binding="lollms",
            model="some-model",
            embedding_dim=None,
        )
        assert func is not None
