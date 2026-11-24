"""API server CLI/env parsing and validation for admission, scan batch, chunking, workspace, and timeout."""

from __future__ import annotations

import importlib
import sys
from argparse import Namespace
from types import SimpleNamespace

import pytest

from lightrag.api.config import (
    initialize_config,
    parse_args,
    validate_admission_configuration,
    validate_scan_batch_configuration,
)
from lightrag.constants import (
    DEFAULT_EMBEDDING_CHUNK_OVERLAP_TOKEN_SIZE,
    DEFAULT_MAX_PENDING_DOCUMENTS,
    DEFAULT_MAX_REQUEST_BODY_BYTES,
    DEFAULT_MAX_TEXTS_PER_REQUEST,
    DEFAULT_SCAN_ENQUEUE_BATCH_SIZE,
)

pytestmark = pytest.mark.offline


def _parse_env(monkeypatch, var: str, value: str | None):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    if value is None:
        monkeypatch.delenv(var, raising=False)
    else:
        monkeypatch.setenv(var, value)
    return parse_args()


# --------------------------------------------------------------------------- #
# Admission controls: MAX_PENDING_DOCUMENTS, MAX_REQUEST_BODY_BYTES,
# MAX_TEXTS_PER_REQUEST, PIPELINE_REQUIRE_STRICT_STORAGE_READS
# --------------------------------------------------------------------------- #


class TestAdmissionConfig:
    def test_defaults_when_unset(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["lightrag-server"])
        for var in (
            "MAX_PENDING_DOCUMENTS",
            "MAX_REQUEST_BODY_BYTES",
            "MAX_TEXTS_PER_REQUEST",
        ):
            monkeypatch.delenv(var, raising=False)

        args = parse_args()

        assert args.max_pending_documents == DEFAULT_MAX_PENDING_DOCUMENTS == 0
        assert args.max_request_body_bytes == DEFAULT_MAX_REQUEST_BODY_BYTES
        assert args.max_request_body_bytes > 0
        assert args.max_request_body_bytes_explicit is False
        assert args.max_texts_per_request == DEFAULT_MAX_TEXTS_PER_REQUEST == 0
        validate_admission_configuration(args)

    @pytest.mark.parametrize(
        ("var", "raw", "attr", "expected"),
        [
            ("MAX_PENDING_DOCUMENTS", "250", "max_pending_documents", 250),
            ("MAX_REQUEST_BODY_BYTES", "0", "max_request_body_bytes", 0),
            ("MAX_REQUEST_BODY_BYTES", "1048576", "max_request_body_bytes", 1048576),
            ("MAX_TEXTS_PER_REQUEST", "50", "max_texts_per_request", 50),
        ],
        ids=[
            "pending-docs-250",
            "body-limit-disabled-0",
            "body-limit-1mib",
            "texts-limit-50",
        ],
    )
    def test_env_value_is_honoured(self, monkeypatch, var, raw, attr, expected):
        args = _parse_env(monkeypatch, var, raw)
        assert getattr(args, attr) == expected
        validate_admission_configuration(args)

    @pytest.mark.parametrize(
        ("var", "raw"),
        [
            ("MAX_PENDING_DOCUMENTS", "-5"),
            ("MAX_REQUEST_BODY_BYTES", "-1"),
            ("MAX_TEXTS_PER_REQUEST", "-1"),
        ],
    )
    def test_negative_limit_fails_fast(self, monkeypatch, var, raw):
        args = _parse_env(monkeypatch, var, raw)
        with pytest.raises(ValueError, match=var):
            validate_admission_configuration(args)

    @pytest.mark.parametrize("bad", [True, 1.5, "many", None])
    def test_non_integer_capacity_fails_fast(self, bad):
        with pytest.raises(ValueError, match="MAX_PENDING_DOCUMENTS"):
            validate_admission_configuration(Namespace(max_pending_documents=bad))

    @pytest.mark.parametrize(
        "ns",
        [
            pytest.param(Namespace(), id="empty-namespace"),
            pytest.param(
                Namespace(max_pending_documents=5), id="only-capacity-present"
            ),
        ],
    )
    def test_partial_namespace_is_not_an_operator_error(self, ns):
        """Programmatic callers may pass a namespace missing admission knobs."""
        validate_admission_configuration(ns)

    def test_initialize_config_rejects_a_negative_capacity(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["lightrag-server"])
        monkeypatch.setenv("MAX_PENDING_DOCUMENTS", "-1")
        with pytest.raises(ValueError, match="MAX_PENDING_DOCUMENTS"):
            initialize_config(force=True)

    def test_body_limit_provenance_survives_value_equal_to_default(self, monkeypatch):
        """Setting MAX_REQUEST_BODY_BYTES to the default still marks it explicit
        so /documents/text(s) use the configured limit rather than the 50 MiB tier."""
        args = _parse_env(
            monkeypatch, "MAX_REQUEST_BODY_BYTES", str(DEFAULT_MAX_REQUEST_BODY_BYTES)
        )
        assert args.max_request_body_bytes == DEFAULT_MAX_REQUEST_BODY_BYTES
        assert args.max_request_body_bytes_explicit is True

    def test_unparseable_body_limit_falls_back_to_default_tiering(self, monkeypatch):
        args = _parse_env(monkeypatch, "MAX_REQUEST_BODY_BYTES", "not-a-number")
        assert args.max_request_body_bytes == DEFAULT_MAX_REQUEST_BODY_BYTES
        assert args.max_request_body_bytes_explicit is False

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(None, False), ("true", True), ("false", False)],
    )
    def test_strict_reads_gate_parses_as_a_bool(self, monkeypatch, value, expected):
        args = _parse_env(monkeypatch, "PIPELINE_REQUIRE_STRICT_STORAGE_READS", value)
        assert args.pipeline_require_strict_storage_reads is expected


# --------------------------------------------------------------------------- #
# SCAN_ENQUEUE_BATCH_SIZE
# --------------------------------------------------------------------------- #


class TestScanBatchConfig:
    def test_defaults_when_unset(self, monkeypatch):
        args = _parse_env(monkeypatch, "SCAN_ENQUEUE_BATCH_SIZE", None)
        assert args.scan_enqueue_batch_size == DEFAULT_SCAN_ENQUEUE_BATCH_SIZE
        validate_scan_batch_configuration(args)

    def test_env_value_is_honoured(self, monkeypatch):
        args = _parse_env(monkeypatch, "SCAN_ENQUEUE_BATCH_SIZE", "25")
        assert args.scan_enqueue_batch_size == 25
        validate_scan_batch_configuration(args)

    @pytest.mark.parametrize("value", ["0", "-1"])
    def test_non_positive_batch_size_fails_fast(self, monkeypatch, value):
        args = _parse_env(monkeypatch, "SCAN_ENQUEUE_BATCH_SIZE", value)
        with pytest.raises(ValueError, match="SCAN_ENQUEUE_BATCH_SIZE"):
            validate_scan_batch_configuration(args)

    def test_initialize_config_enforces_the_batch_size(self, monkeypatch):
        args = _parse_env(monkeypatch, "SCAN_ENQUEUE_BATCH_SIZE", "0")
        with pytest.raises(ValueError, match="SCAN_ENQUEUE_BATCH_SIZE"):
            initialize_config(args, force=True)

    def test_partial_namespace_is_accepted_but_bool_rejected(self):
        validate_scan_batch_configuration(SimpleNamespace())
        with pytest.raises(ValueError, match="SCAN_ENQUEUE_BATCH_SIZE"):
            validate_scan_batch_configuration(
                SimpleNamespace(scan_enqueue_batch_size=True)
            )


# --------------------------------------------------------------------------- #
# CUSTOM_CHUNKER
# --------------------------------------------------------------------------- #


class TestCustomChunkerConfig:
    @pytest.mark.parametrize("value", ["", "company-chunker"])
    def test_custom_chunker_env(self, monkeypatch, value):
        assert _parse_env(monkeypatch, "CUSTOM_CHUNKER", value).custom_chunker == value

    def test_custom_chunker_cli_overrides_env(self, monkeypatch):
        monkeypatch.setenv("CUSTOM_CHUNKER", "env-chunker")
        monkeypatch.setattr(
            sys, "argv", ["lightrag-server", "--custom-chunker", "cli-chunker"]
        )
        assert parse_args().custom_chunker == "cli-chunker"


# --------------------------------------------------------------------------- #
# EMBEDDING_CHUNK_OVERLAP_TOKEN_SIZE
# --------------------------------------------------------------------------- #


class TestEmbeddingChunkOverlapConfig:
    def test_defaults_when_unset(self, monkeypatch):
        args = _parse_env(monkeypatch, "EMBEDDING_CHUNK_OVERLAP_TOKEN_SIZE", None)
        assert (
            args.embedding_chunk_overlap_token_size
            == DEFAULT_EMBEDDING_CHUNK_OVERLAP_TOKEN_SIZE
            == 100
        )

    @pytest.mark.parametrize(("value", "expected"), [("0", 0), ("42", 42)])
    def test_env_value_is_honoured(self, monkeypatch, value, expected):
        args = _parse_env(monkeypatch, "EMBEDDING_CHUNK_OVERLAP_TOKEN_SIZE", value)
        assert args.embedding_chunk_overlap_token_size == expected

    def test_does_not_share_a_value_with_chunk_overlap_size(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["lightrag-server"])
        monkeypatch.setenv("CHUNK_OVERLAP_SIZE", "7")
        monkeypatch.delenv("EMBEDDING_CHUNK_OVERLAP_TOKEN_SIZE", raising=False)
        args = parse_args()
        assert args.chunk_overlap_size == 7
        assert (
            args.embedding_chunk_overlap_token_size
            == DEFAULT_EMBEDDING_CHUNK_OVERLAP_TOKEN_SIZE
        )


# --------------------------------------------------------------------------- #
# WORKSPACE
# --------------------------------------------------------------------------- #


class TestWorkspaceConfig:
    def _parse_workspace(self, monkeypatch, workspace: str):
        monkeypatch.setenv("WORKSPACE", workspace)
        monkeypatch.setattr(sys, "argv", ["lightrag-server"])
        config = importlib.import_module("lightrag.api.config")
        return config.parse_args()

    @pytest.mark.parametrize("workspace", ["$prod", "$default"])
    def test_a_reserved_workspace_exits_with_one_line(self, monkeypatch, workspace):
        with pytest.raises(SystemExit) as error:
            self._parse_workspace(monkeypatch, workspace)
        message = str(error.value.code)
        assert message.startswith("WORKSPACE: ")
        assert "reserved" in message

    def test_other_characters_are_still_rewritten(self, monkeypatch):
        assert self._parse_workspace(monkeypatch, "team-a.prod").workspace == (
            "team_a_prod"
        )


# --------------------------------------------------------------------------- #
# --timeout help text
# --------------------------------------------------------------------------- #


def test_timeout_help_does_not_advertise_none_or_infinite(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["lightrag-server", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    timeout_help = help_text.partition("\n  --timeout TIMEOUT")[2].split("\n  --", 1)[0]
    assert timeout_help, "the --timeout entry was not found in the help output"
    assert "None" not in timeout_help
    assert "infinite" not in timeout_help
