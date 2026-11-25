"""G0 wiring tests for the native ``smart_heading`` engine parameter.

Covers the parameter plumbing added ahead of the smart-heading algorithm
itself: ``parse_engine`` decode at parse start, the ``NativeExtractRuntime``
handed to ``extract``, persist-time re-encode, loud failure on malformed
directives, and the markdown warn-and-ignore path.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from lightrag.constants import FULL_DOCS_FORMAT_PENDING_PARSE
from lightrag.parser.base import ParseContext, ParseResult
from lightrag.parser.debug import build_debug_rag
from lightrag.parser.registry import get_parser

from tests.parser.docx._smart_heading_helpers import log_messages as _log_messages

pytestmark = pytest.mark.offline


@pytest.fixture
def clean_smart_heading_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DOCX_SMART_HEADING", raising=False)
    monkeypatch.delenv("DOCX_SMART_HEADING_MAX_CHARS", raising=False)


_STUB_BLOCKS = [
    {
        "uuid": "p1",
        "heading": "Chapter One",
        "content": "# Chapter One\nBody text.",
        "type": "text",
        "parent_headings": [],
        "level": 1,
    }
]


def _stub_extract(
    file_path,
    *,
    drawing_context=None,
    parse_warnings=None,
    parse_metadata=None,
    **_kwargs,
):
    return [dict(b) for b in _STUB_BLOCKS]


def _parse_docx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    parse_engine: Any = "__absent__",
) -> tuple[Any, ParseResult, list[dict[str, Any]]]:
    """Drive ``get_parser("native").parse`` on a stub docx.

    Returns ``(rag, result, extract_calls)`` where ``extract_calls`` records
    the keyword arguments (notably ``runtime``) each ``extract`` call saw.
    """
    from lightrag.parser.docx.parser import NativeDocxParser

    input_dir = tmp_path / "inputs"
    input_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("INPUT_DIR", str(input_dir))

    source_path = input_dir / "doc.docx"
    source_path.write_bytes(b"fake-docx")

    content_data: dict[str, Any] = {
        "parse_format": FULL_DOCS_FORMAT_PENDING_PARSE,
        "content": "",
    }
    if parse_engine != "__absent__":
        content_data["parse_engine"] = parse_engine

    extract_calls: list[dict[str, Any]] = []
    orig_extract = NativeDocxParser.extract

    def _spy_extract(self, source, **kwargs):
        extract_calls.append(dict(kwargs))
        return orig_extract(self, source, **kwargs)

    rag = build_debug_rag()

    with (
        mock.patch.object(NativeDocxParser, "extract", _spy_extract),
        mock.patch(
            "lightrag.parser.docx.parse_document.extract_docx_blocks",
            _stub_extract,
        ),
    ):
        result = asyncio.run(
            get_parser("native").parse(
                ParseContext(rag, "doc-1", str(source_path), content_data)
            )
        )
    return rag, result, extract_calls


def test_no_params_persists_bare_engine(tmp_path, monkeypatch) -> None:
    rag, result, calls = _parse_docx(tmp_path, monkeypatch)
    assert rag.full_docs.data["doc-1"]["parse_engine"] == "native"
    assert result.parse_engine == "native"
    assert len(calls) == 1
    runtime = calls[0]["runtime"]
    assert dict(runtime.engine_params) == {}
    assert runtime.llm_invoke is None


def test_bare_native_persists_bare_engine(tmp_path, monkeypatch) -> None:
    rag, _result, calls = _parse_docx(tmp_path, monkeypatch, parse_engine="native")
    assert rag.full_docs.data["doc-1"]["parse_engine"] == "native"
    assert dict(calls[0]["runtime"].engine_params) == {}


def test_smart_heading_param_reaches_extract_and_persists(
    tmp_path, monkeypatch
) -> None:
    rag, _result, calls = _parse_docx(
        tmp_path, monkeypatch, parse_engine="native(smart_heading=true)"
    )
    # G0-4: persist re-encodes the directive so the stored value keeps params.
    assert rag.full_docs.data["doc-1"]["parse_engine"] == "native(smart_heading=true)"
    # G0-2: the decoded params reach extract via the runtime.
    assert dict(calls[0]["runtime"].engine_params) == {"smart_heading": True}


def test_malformed_parse_engine_fails_loudly(tmp_path, monkeypatch) -> None:
    with pytest.raises(ValueError, match="invalid parse_engine"):
        _parse_docx(tmp_path, monkeypatch, parse_engine="native(smart_heading=true")


def test_wants_llm_bridge_gates_on_smart_heading() -> None:
    from lightrag.parser.docx.parser import NativeDocxParser
    from lightrag.parser.markdown.parser import NativeMarkdownParser

    docx = NativeDocxParser()
    assert docx.wants_llm_bridge({"smart_heading": True}) is True
    assert docx.wants_llm_bridge({"smart_heading": False}) is False
    assert docx.wants_llm_bridge({}) is False
    assert NativeMarkdownParser().wants_llm_bridge({"smart_heading": True}) is False


def test_markdown_warns_and_ignores_params(
    tmp_path, monkeypatch, lightrag_caplog
) -> None:
    """G0-3: a .md routed with smart_heading parses normally, warns once."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    monkeypatch.setenv("INPUT_DIR", str(input_dir))

    source_path = input_dir / "notes.md"
    source_path.write_text("# Title\n\nSome body text.\n", encoding="utf-8")

    rag = build_debug_rag()
    with lightrag_caplog.at_level(logging.WARNING, logger="lightrag"):
        result = asyncio.run(
            get_parser("native").parse(
                ParseContext(
                    rag,
                    "doc-md",
                    str(source_path),
                    {
                        "parse_format": FULL_DOCS_FORMAT_PENDING_PARSE,
                        "content": "",
                        "parse_engine": "native(smart_heading=true)",
                    },
                )
            )
        )

    assert (
        result.parse_warnings
        and result.parse_warnings.get("engine_params_ignored") == 1
    )
    assert any(
        "only apply to .docx" in msg for msg in _log_messages(lightrag_caplog)
    ), "expected a warn-and-ignore log line"
    # Parsing succeeded and the directive is still persisted verbatim.
    assert rag.full_docs.data["doc-md"]["parse_engine"] == "native(smart_heading=true)"
    assert "Title" in rag.full_docs.data["doc-md"]["content"]


def test_markdown_does_not_warn_on_falsy_param(
    tmp_path, monkeypatch, lightrag_caplog
) -> None:
    """Review: a blanket opt-out (smart_heading=false) turned nothing on, so a
    .md routed through it must NOT emit the warn-and-ignore noise."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    monkeypatch.setenv("INPUT_DIR", str(input_dir))

    source_path = input_dir / "notes.md"
    source_path.write_text("# Title\n\nSome body text.\n", encoding="utf-8")

    rag = build_debug_rag()
    with lightrag_caplog.at_level(logging.WARNING, logger="lightrag"):
        result = asyncio.run(
            get_parser("native").parse(
                ParseContext(
                    rag,
                    "doc-md-off",
                    str(source_path),
                    {
                        "parse_format": FULL_DOCS_FORMAT_PENDING_PARSE,
                        "content": "",
                        "parse_engine": "native(smart_heading=false)",
                    },
                )
            )
        )

    assert not (result.parse_warnings or {}).get("engine_params_ignored")
    assert not any(
        "only apply to .docx" in msg for msg in _log_messages(lightrag_caplog)
    )
    assert "Title" in rag.full_docs.data["doc-md-off"]["content"]


def test_i4_cache_disabled_surfaces_parse_warning(tmp_path, monkeypatch) -> None:
    """The I4 determinism waiver reaches the sidecar smart_audit.json.

    It is a smart-heading diagnostic, so it is diverted to the audit file
    (under a ``parse_warnings`` key) rather than doc_status — and it must not
    be limited to the process log.
    """

    async def _fake_llm(prompt: str, **_kw) -> str:
        return "{}"

    input_dir = tmp_path / "inputs"
    input_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    source_path = input_dir / "doc.docx"
    source_path.write_bytes(b"fake-docx")

    content_data: dict[str, Any] = {
        "parse_format": FULL_DOCS_FORMAT_PENDING_PARSE,
        "content": "",
        "parse_engine": "native(smart_heading=true)",
    }

    rag = build_debug_rag(extract_llm_func=_fake_llm)
    orig_config = rag._build_global_config
    rag._build_global_config = lambda: {
        **orig_config(),
        "enable_llm_cache_for_entity_extract": False,
    }

    with mock.patch(
        "lightrag.parser.docx.parse_document.extract_docx_blocks",
        _stub_extract,
    ):
        result = asyncio.run(
            get_parser("native").parse(
                ParseContext(rag, "doc-1", str(source_path), content_data)
            )
        )
    # The waiver no longer rides doc_status parse_warnings ...
    assert "smart_i4_cache_disabled" not in (result.parse_warnings or {})
    # ... it lands in the sidecar audit file instead.
    audits = list(input_dir.glob("**/*.smart_audit.json"))
    assert len(audits) == 1, f"expected one smart_audit.json, got {audits}"
    audit = json.loads(audits[0].read_text(encoding="utf-8"))
    assert audit["parse_warnings"]["smart_i4_cache_disabled"] == 1


# --- parse_warnings split: smart-heading → smart_audit.json, rest → doc_status -


def _make_stub(*, warnings=None, ledger=None):
    """``extract_docx_blocks`` stand-in that seeds parse_warnings / the audit
    ledger, so the merge path (ledger + warnings) is exercised end to end."""

    def _stub(
        file_path,
        *,
        drawing_context=None,
        parse_warnings=None,
        parse_metadata=None,
        **_kwargs,
    ):
        if warnings and parse_warnings is not None:
            parse_warnings.update(warnings)
        if ledger is not None and parse_metadata is not None:
            parse_metadata["smart_audit"] = ledger
        return [dict(b) for b in _STUB_BLOCKS]

    return _stub


def _parse_with_stub(tmp_path, monkeypatch, stub, *, parse_engine="native"):
    """Drive ``parse()`` on a stub docx; return ``(result, audit_or_None)``
    where ``audit`` is the parsed ``<base>.smart_audit.json`` (or None)."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    source_path = input_dir / "doc.docx"
    source_path.write_bytes(b"fake-docx")
    content_data: dict[str, Any] = {
        "parse_format": FULL_DOCS_FORMAT_PENDING_PARSE,
        "content": "",
        "parse_engine": parse_engine,
    }
    rag = build_debug_rag()
    with mock.patch("lightrag.parser.docx.parse_document.extract_docx_blocks", stub):
        result = asyncio.run(
            get_parser("native").parse(
                ParseContext(rag, "doc-1", str(source_path), content_data)
            )
        )
    audits = list(input_dir.glob("**/*.smart_audit.json"))
    audit = json.loads(audits[0].read_text(encoding="utf-8")) if audits else None
    return result, audit


def test_split_ledger_plus_smart_and_nonsmart(tmp_path, monkeypatch) -> None:
    """ledger + smart + non-smart: ledger preserved verbatim, smart warnings
    merged under a ``parse_warnings`` key, non-smart stays on doc_status."""
    result, audit = _parse_with_stub(
        tmp_path,
        monkeypatch,
        _make_stub(
            warnings={
                "smart_cb1_tripped": 3,
                "heading_softbreak_split_count": 1,
                "missing_paraid_count": 2,
            },
            ledger={"shadow_diff": {"x": 1}, "fallback_sub_documents": []},
        ),
    )
    assert result.parse_warnings == {
        "heading_softbreak_split_count": 1,
        "missing_paraid_count": 2,
    }
    assert audit == {
        "shadow_diff": {"x": 1},
        "fallback_sub_documents": [],
        "parse_warnings": {"smart_cb1_tripped": 3},
    }


def test_split_lone_nonsmart_warning_not_dropped(tmp_path, monkeypatch) -> None:
    """Regression: a lone non-smart warning (no smart_ key, no missing_paraId)
    must still reach doc_status. The pre-refactor surface path returned it as
    ``None`` and silently dropped it."""
    result, audit = _parse_with_stub(
        tmp_path,
        monkeypatch,
        _make_stub(
            warnings={"heading_softbreak_split_count": 1},
            ledger={"shadow_diff": {}},
        ),
    )
    assert result.parse_warnings == {"heading_softbreak_split_count": 1}
    # ledger still written, no parse_warnings key (no smart warnings this run).
    assert audit == {"shadow_diff": {}}


def test_split_smart_warning_without_ledger_still_writes_audit(
    tmp_path, monkeypatch
) -> None:
    result, audit = _parse_with_stub(
        tmp_path,
        monkeypatch,
        _make_stub(warnings={"smart_toc_removed_lines": 5}),
    )
    assert result.parse_warnings is None
    assert audit == {"parse_warnings": {"smart_toc_removed_lines": 5}}


def test_split_no_warnings_no_ledger_writes_no_audit(tmp_path, monkeypatch) -> None:
    result, audit = _parse_with_stub(tmp_path, monkeypatch, _make_stub())
    assert result.parse_warnings is None
    assert audit is None


def test_markdown_warning_stays_on_doc_status_no_audit_file(
    tmp_path, monkeypatch
) -> None:
    """markdown emits only non-smart warnings (engine_params_ignored): they
    ride doc_status parse_warnings and no smart_audit.json is written."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    source_path = input_dir / "notes.md"
    source_path.write_text("# Title\n\nSome body text.\n", encoding="utf-8")

    rag = build_debug_rag()
    result = asyncio.run(
        get_parser("native").parse(
            ParseContext(
                rag,
                "doc-md",
                str(source_path),
                {
                    "parse_format": FULL_DOCS_FORMAT_PENDING_PARSE,
                    "content": "",
                    "parse_engine": "native(smart_heading=true)",
                },
            )
        )
    )
    assert (result.parse_warnings or {}).get("engine_params_ignored") == 1
    assert list(input_dir.glob("**/*.smart_audit.json")) == []


def test_smart_audit_json_is_byte_stable_across_reparse(tmp_path, monkeypatch) -> None:
    """I4: re-parsing the same document yields a byte-identical smart_audit.json.

    Compares raw bytes (not parsed objects) so a regression in ``sort_keys`` /
    indent / trailing newline is caught. The ledger keys are deliberately given
    out of sorted order to prove ``sort_keys`` normalizes them.
    """
    stub = _make_stub(
        warnings={"smart_cb1_tripped": 3, "smart_toc_removed_lines": 5},
        ledger={"shadow_diff": {"b": 2, "a": 1}, "decisions": [{"z": 1}]},
    )

    def _audit_bytes(sub: str) -> bytes:
        input_dir = tmp_path / sub
        input_dir.mkdir()
        monkeypatch.setenv("INPUT_DIR", str(input_dir))
        source_path = input_dir / "doc.docx"
        source_path.write_bytes(b"fake-docx")
        rag = build_debug_rag()
        with mock.patch(
            "lightrag.parser.docx.parse_document.extract_docx_blocks", stub
        ):
            asyncio.run(
                get_parser("native").parse(
                    ParseContext(
                        rag,
                        "doc-1",
                        str(source_path),
                        {
                            "parse_format": FULL_DOCS_FORMAT_PENDING_PARSE,
                            "content": "",
                            "parse_engine": "native",
                        },
                    )
                )
            )
        (audit,) = input_dir.glob("**/*.smart_audit.json")
        return audit.read_bytes()

    assert _audit_bytes("a") == _audit_bytes("b")


def test_smart_audit_json_uses_lf_line_endings(
    tmp_path, monkeypatch, emulate_windows_newlines
) -> None:
    """The audit artifact sits in the ``*.parsed/`` directory next to the
    sidecars, so it carries the same floor: LF on every platform.

    ``Path.write_text`` translates ``\\n`` to ``os.linesep`` unless
    ``newline`` is passed, and both the indented payload and the trailing
    newline are newlines — on Windows the file would otherwise be CRLF while
    the sidecars it is filed beside stay LF.
    ``emulate_windows_newlines`` applies that translation on Linux as well,
    so this test goes red in CI if the pin is dropped.
    """
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    source_path = input_dir / "doc.docx"
    source_path.write_bytes(b"fake-docx")
    rag = build_debug_rag()
    with mock.patch(
        "lightrag.parser.docx.parse_document.extract_docx_blocks",
        _make_stub(warnings={"smart_cb1_tripped": 1}),
    ):
        asyncio.run(
            get_parser("native").parse(
                ParseContext(
                    rag,
                    "doc-lf",
                    str(source_path),
                    {
                        "parse_format": FULL_DOCS_FORMAT_PENDING_PARSE,
                        "content": "",
                        "parse_engine": "native",
                    },
                )
            )
        )

    (audit,) = input_dir.glob("**/*.smart_audit.json")
    raw = audit.read_bytes()
    assert b"parse_warnings" in raw  # the audit really was written
    assert b"\n" in raw
    assert b"\r\n" not in raw, "smart_audit.json was written with CRLF"


# ---------------------------------------------------------------------------
# DOCX_SMART_HEADING global default, directive seeding & startup validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(None, False, id="unset-default-off"),
        pytest.param("1", True, id="truthy-1"),
        pytest.param("on", True, id="truthy-on"),
        pytest.param("Yes", True, id="truthy-Yes"),
        pytest.param("false", False, id="falsy-false"),
    ],
)
def test_smart_heading_switch_parsing(
    monkeypatch, clean_smart_heading_env, raw: str | None, expected: bool
) -> None:
    from lightrag.parser.routing import smart_heading_default_enabled

    if raw is not None:
        monkeypatch.setenv("DOCX_SMART_HEADING", raw)
    assert smart_heading_default_enabled() is expected


def test_smart_heading_switch_invalid_raises(
    monkeypatch, clean_smart_heading_env
) -> None:
    from lightrag.parser.routing import (
        ParserRoutingConfigError,
        smart_heading_default_enabled,
    )

    monkeypatch.setenv("DOCX_SMART_HEADING", "maybe")
    with pytest.raises(ParserRoutingConfigError, match="DOCX_SMART_HEADING"):
        smart_heading_default_enabled()


@pytest.mark.parametrize(
    ("switch", "filename", "rules", "exp_engine", "exp_params", "exp_encoded"),
    [
        pytest.param(
            "true",
            "doc.[native].docx",
            "",
            "native",
            {"smart_heading": True},
            "native(smart_heading=true)",
            id="seed-injected-native-docx",
        ),
        pytest.param(
            None,
            "doc.[native].docx",
            "",
            "native",
            {},
            "native",
            id="switch-off-bare-encoding",
        ),
        pytest.param(
            "true",
            "notes.[native].md",
            "",
            "native",
            {},
            None,
            id="not-injected-for-md",
        ),
        pytest.param(
            "true",
            "bundle.textpack",
            "",
            "native",
            {},
            None,
            id="not-injected-for-textpack",
        ),
        pytest.param(
            "true",
            "doc.docx",
            "",
            "legacy",
            {},
            None,
            id="not-injected-for-legacy-docx",
        ),
        pytest.param(
            "true",
            "doc.[native(smart_heading=false)].docx",
            "",
            "native",
            {"smart_heading": False},
            "native(smart_heading=false)",
            id="hint-false-overrides-seed",
        ),
        pytest.param(
            "true",
            "doc.docx",
            "docx:native(smart_heading=false)",
            "native",
            {"smart_heading": False},
            None,
            id="rule-false-overrides-seed",
        ),
    ],
)
def test_smart_heading_directive_seed_and_overrides(
    monkeypatch,
    clean_smart_heading_env,
    switch: str | None,
    filename: str,
    rules: str,
    exp_engine: str,
    exp_params: dict[str, Any],
    exp_encoded: str | None,
) -> None:
    from lightrag.parser.routing import encode_parse_engine, resolve_parser_directives

    if switch is not None:
        monkeypatch.setenv("DOCX_SMART_HEADING", switch)
    d = resolve_parser_directives(filename, parser_rules=rules)
    assert d.engine == exp_engine
    assert d.engine_params == exp_params
    if exp_encoded is not None:
        assert encode_parse_engine(d.engine, d.engine_params) == exp_encoded


@pytest.mark.parametrize(
    ("switch", "engine", "initial", "filename", "expected"),
    [
        pytest.param(
            "true",
            "native",
            {},
            "a.docx",
            {"smart_heading": True},
            id="fills-native-docx",
        ),
        pytest.param(
            "true",
            "native",
            {"smart_heading": False},
            "a.docx",
            {"smart_heading": False},
            id="respects-explicit-false",
        ),
        pytest.param("true", "native", {}, "notes.md", {}, id="skips-non-docx"),
        pytest.param("true", "mineru", {}, "a.docx", {}, id="skips-non-native"),
        pytest.param(None, "native", {}, "a.docx", {}, id="noop-when-switch-off"),
    ],
)
def test_seed_smart_heading_param_chokepoint(
    monkeypatch,
    clean_smart_heading_env,
    switch: str | None,
    engine: str,
    initial: dict[str, Any],
    filename: str,
    expected: dict[str, Any],
) -> None:
    from lightrag.parser.routing import seed_smart_heading_param

    if switch is not None:
        monkeypatch.setenv("DOCX_SMART_HEADING", switch)
    params = dict(initial)
    seed_smart_heading_param(engine, params, filename)
    assert params == expected


@pytest.mark.parametrize(
    ("switch", "max_chars", "rules", "missing_models"),
    [
        pytest.param(None, None, "", ["zh_core_web_sm"], id="disabled-silent"),
        pytest.param(
            None,
            None,
            "docx:native(smart_heading=false)",
            ["zh_core_web_sm"],
            id="rule-false-silent",
        ),
        pytest.param(
            None,
            None,
            "md:native(smart_heading=true)",
            ["zh_core_web_sm"],
            id="non-docx-rule-silent",
        ),
        pytest.param("true", None, "", [], id="enabled-with-models-ok"),
        pytest.param(None, "2", "", [], id="disabled-ignores-bad-max-chars"),
    ],
)
def test_startup_dependency_check_passing_cases(
    monkeypatch,
    clean_smart_heading_env,
    switch: str | None,
    max_chars: str | None,
    rules: str,
    missing_models: list[str],
) -> None:
    from lightrag.parser.docx.smart_heading import nlp
    from lightrag.parser.routing import validate_smart_heading_dependencies

    if switch is not None:
        monkeypatch.setenv("DOCX_SMART_HEADING", switch)
    if max_chars is not None:
        monkeypatch.setenv("DOCX_SMART_HEADING_MAX_CHARS", max_chars)
    monkeypatch.setattr(nlp, "missing_spacy_models", lambda: list(missing_models))
    validate_smart_heading_dependencies(parser_rules=rules)


@pytest.mark.parametrize(
    ("switch", "max_chars", "rules", "missing_models", "exc_kind", "match"),
    [
        pytest.param(
            "true",
            None,
            "",
            ["zh_core_web_sm"],
            "nlp",
            "zh_core_web_sm",
            id="env-on-missing-models",
        ),
        pytest.param(
            None,
            None,
            "docx:native(smart_heading=true)",
            ["en_core_web_sm", "zh_core_web_sm"],
            "nlp",
            "lightrag-download-cache",
            id="explicit-rule-missing-models",
        ),
        pytest.param(
            None,
            None,
            "docx:native(smart_heading)",
            ["zh_core_web_sm"],
            "nlp",
            "zh_core_web_sm",
            id="bare-flag-rule-triggers",
        ),
        pytest.param(
            None,
            None,
            "do*:native(smart_heading=true)",
            ["zh_core_web_sm"],
            "nlp",
            "zh_core_web_sm",
            id="wildcard-rule-triggers",
        ),
        pytest.param(
            "treu",
            None,
            "",
            [],
            "config",
            "DOCX_SMART_HEADING",
            id="invalid-switch-surfaces",
        ),
        pytest.param(
            "true",
            "2",
            "",
            [],
            "config",
            "too small",
            id="enabled-validates-max-chars",
        ),
    ],
)
def test_startup_dependency_check_error_cases(
    monkeypatch,
    clean_smart_heading_env,
    switch: str | None,
    max_chars: str | None,
    rules: str,
    missing_models: list[str],
    exc_kind: str,
    match: str,
) -> None:
    from lightrag.parser.docx.smart_heading import nlp
    from lightrag.parser.routing import (
        ParserRoutingConfigError,
        validate_smart_heading_dependencies,
    )

    if switch is not None:
        monkeypatch.setenv("DOCX_SMART_HEADING", switch)
    if max_chars is not None:
        monkeypatch.setenv("DOCX_SMART_HEADING_MAX_CHARS", max_chars)
    monkeypatch.setattr(nlp, "missing_spacy_models", lambda: list(missing_models))

    exc_type = (
        nlp.SmartHeadingNLPError if exc_kind == "nlp" else ParserRoutingConfigError
    )
    with pytest.raises(exc_type, match=match):
        validate_smart_heading_dependencies(parser_rules=rules)


def test_missing_spacy_models_probes(monkeypatch) -> None:
    from lightrag.parser.docx.smart_heading import nlp

    real_find_spec = nlp.importlib.util.find_spec
    monkeypatch.setattr(
        nlp.importlib.util,
        "find_spec",
        lambda name, *a, **k: (
            None if name == "spacy" else real_find_spec(name, *a, **k)
        ),
    )
    assert nlp.missing_spacy_models() == ["en_core_web_sm", "zh_core_web_sm"]


def test_missing_spacy_models_reports_uninstalled(monkeypatch) -> None:
    from lightrag.parser.docx.smart_heading import nlp

    spacy_util = pytest.importorskip("spacy.util")
    monkeypatch.setattr(spacy_util, "is_package", lambda name: name == "en_core_web_sm")
    assert nlp.missing_spacy_models() == ["zh_core_web_sm"]


def test_max_chars_env_validation(monkeypatch, clean_smart_heading_env) -> None:
    from lightrag.parser import routing
    from lightrag.parser.routing import (
        ParserRoutingConfigError,
        _validate_smart_heading_max_chars,
    )

    # Unset and sane values pass silently.
    _validate_smart_heading_max_chars()
    for good in ("180", "90", "300"):
        monkeypatch.setenv("DOCX_SMART_HEADING_MAX_CHARS", good)
        _validate_smart_heading_max_chars()

    # Usable-but-tiny cap warns without raising.
    calls: list = []
    monkeypatch.setattr(routing.logger, "warning", lambda *a, **k: calls.append(a))
    monkeypatch.setenv("DOCX_SMART_HEADING_MAX_CHARS", "50")
    _validate_smart_heading_max_chars()
    assert calls, "expected a warning for a below-title-width cap"

    # Non-integer value raises ParserRoutingConfigError.
    monkeypatch.setenv("DOCX_SMART_HEADING_MAX_CHARS", "1.5")
    with pytest.raises(ParserRoutingConfigError, match="DOCX_SMART_HEADING_MAX_CHARS"):
        _validate_smart_heading_max_chars()
