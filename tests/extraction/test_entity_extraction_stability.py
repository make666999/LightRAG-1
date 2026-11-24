"""Tests for entity extraction prompt stability, prompt profiles, and section context.

Covers:
- Default/custom ``entity_types_guidance`` and prompt-file injection (text & JSON modes)
- Format-only built-in templates and quantity-limit prompt rules
- ``ENTITY_TYPES`` env var guard and prompt-file validation / runtime refresh
- Section Context (heading breadcrumb) injection, cleaning, per-level capping,
  and token-budget truncation across extraction and query stages
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from lightrag.utils import EmbeddingFunc, Tokenizer, TokenizerInterface

from ._helpers import make_chunks, make_dummy_tokenizer, make_extract_global_config

pytestmark = pytest.mark.offline


def _make_global_config(
    addon_params: dict | None = None,
    use_json: bool = False,
    max_gleaning: int = 0,
    prompt_profile: dict | None = None,
) -> dict:
    return make_extract_global_config(
        addon_params=addon_params,
        use_json=use_json,
        entity_extract_max_gleaning=max_gleaning,
        prompt_profile=prompt_profile,
        max_extract_input_tokens=20480,
    )


def _make_chunks(content: str = "Alice founded Acme Corp in 1990.") -> dict[str, dict]:
    return make_chunks(1, content=content)


def _require_yaml() -> None:
    pytest.importorskip("yaml")


def _write_prompt_profile(
    path: Path,
    *,
    guidance: str | None = None,
    text_examples: list[str] | None = None,
    json_examples: list[str] | None = None,
) -> None:
    lines: list[str] = []

    def _append_block(key: str, value: str) -> None:
        lines.append(f"{key}: |")
        for line in value.strip("\n").splitlines():
            lines.append(f"  {line}")

    def _append_examples(key: str, values: list[str]) -> None:
        lines.append(f"{key}:")
        for value in values:
            lines.append("  - |")
            for line in value.strip("\n").splitlines():
                lines.append(f"    {line}")

    if guidance is not None:
        _append_block("entity_types_guidance", guidance)
    if text_examples is not None:
        _append_examples("entity_extraction_examples", text_examples)
    if json_examples is not None:
        _append_examples("entity_extraction_json_examples", json_examples)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dummy_embedding_func() -> EmbeddingFunc:
    async def _embed(texts):
        return [[0.0, 0.0, 0.0] for _ in texts]

    return EmbeddingFunc(embedding_dim=3, func=_embed)


def _patch_prompt_dir(path: Path):
    return patch("lightrag.prompt.get_entity_type_prompt_dir", return_value=path)


def _text_profile_example(label: str) -> str:
    return f"""---Entity Types---
- ExampleType: Test type

---Input Text---
```
{label}
```

---Output---
entity{{tuple_delimiter}}{label}{{tuple_delimiter}}ExampleType{{tuple_delimiter}}{label} description.
{{completion_delimiter}}"""


def _json_profile_example(label: str) -> str:
    return f"""---Entity Types---
- ExampleType: Test type

---Input Text---
```
{label}
```

---Output---
{{
  "entities": [
    {{"name": "{label}", "type": "ExampleType", "description": "{label} description."}}
  ],
  "relationships": []
}}"""


_TEXT_MODE_RESPONSE = (
    "entity<|#|>Alice<|#|>Person<|#|>Alice is the founder of Acme Corp."
    "\nentity<|#|>Acme Corp<|#|>Organization<|#|>Acme Corp is a company founded by Alice."
    "\nrelation<|#|>Alice<|#|>Acme Corp<|#|>founded<|#|>Alice founded Acme Corp."
    "\n<|COMPLETE|>"
)

_TEXT_MODE_MISPREFIXED_RELATION_RESPONSE = (
    "entity<|#|>Alice<|#|>Person<|#|>Alice is the founder of Acme Corp."
    "\nentity<|#|>Acme Corp<|#|>Organization<|#|>Acme Corp is a company founded by Alice."
    "\nentity<|#|>Alice<|#|>Acme Corp<|#|>founded<|#|>Alice founded Acme Corp."
    "\n<|COMPLETE|>"
)

_TEXT_MODE_GLEANED_RELATION_RESPONSES = [
    _TEXT_MODE_MISPREFIXED_RELATION_RESPONSE,
    "\nrelation<|#|>Alice<|#|>Acme Corp<|#|>founded<|#|>Alice founded Acme Corp.\n<|COMPLETE|>",
]

_TEXT_MODE_CROSS_PASS_RELATION_RESPONSES = [
    "entity<|#|>Alice<|#|>Person<|#|>Alice founded a company.\n<|COMPLETE|>",
    "entity<|#|>Acme Corp<|#|>Organization<|#|>Acme Corp was founded by Alice."
    "\nrelation<|#|>Alice<|#|>Acme Corp<|#|>founded<|#|>Alice founded Acme Corp.\n<|COMPLETE|>",
]

_JSON_MODE_RESPONSE = json.dumps(
    {
        "entities": [
            {
                "name": "Alice",
                "type": "Person",
                "description": "Alice is the founder of Acme Corp.",
            },
            {
                "name": "Acme Corp",
                "type": "Organization",
                "description": "Acme Corp is a company founded by Alice.",
            },
        ],
        "relationships": [
            {
                "source": "Alice",
                "target": "Acme Corp",
                "keywords": "founded",
                "description": "Alice founded Acme Corp.",
            },
        ],
    }
)


class _DummyTextChunksStorage:
    async def get_by_id(self, chunk_id: str):
        return {"file_path": "test.md"}


# --------------------------------------------------------------------------- #
# 1. Built-in prompt templates & guidance constants
# --------------------------------------------------------------------------- #


def test_default_entity_types_guidance_covers_all_types():
    """Default guidance must mention all 11 canonical entity types."""
    from lightrag.prompt import PROMPTS

    guidance = PROMPTS["default_entity_types_guidance"]
    expected_types = [
        "Person",
        "Creature",
        "Organization",
        "Location",
        "Event",
        "Concept",
        "Method",
        "Content",
        "Data",
        "Artifact",
        "NaturalObject",
    ]
    for entity_type in expected_types:
        assert entity_type in guidance, (
            f"Expected entity type '{entity_type}' missing from default_entity_types_guidance"
        )


def test_builtin_entity_extraction_examples_are_format_only():
    """Built-in text and JSON examples must be placeholder-only templates."""
    from lightrag.prompt import PROMPTS

    section_markers = ("---Input Text---", "---Output---", "---Entity Types---")
    placeholder = re.compile(r"<[^<>]+>")
    tuple_delimiter = PROMPTS["DEFAULT_TUPLE_DELIMITER"]
    completion_delimiter = PROMPTS["DEFAULT_COMPLETION_DELIMITER"]

    for example in PROMPTS["entity_extraction_examples"]:
        for marker in section_markers:
            assert marker not in example
        rendered = example.format(
            tuple_delimiter=tuple_delimiter,
            completion_delimiter=completion_delimiter,
        )
        for line in rendered.splitlines():
            line = line.strip()
            if not line or line == completion_delimiter:
                continue
            tag, *fields = line.split(tuple_delimiter)
            assert tag in {"entity", "relation"}
            assert fields
            for field in fields:
                assert placeholder.fullmatch(field), field

    for example in PROMPTS["entity_extraction_json_examples"]:
        for marker in section_markers:
            assert marker not in example
        parsed = json.loads(example)
        records = parsed["entities"] + parsed["relationships"]
        assert records
        for record in records:
            for value in record.values():
                assert placeholder.fullmatch(value), value


def test_entity_extraction_system_prompts_label_examples_as_format_templates():
    from lightrag.prompt import PROMPTS

    for prompt_key in (
        "entity_extraction_system_prompt",
        "entity_extraction_json_system_prompt",
    ):
        prompt = PROMPTS[prompt_key]
        assert "---Output Format Template---" in prompt
        assert "---Examples---" not in prompt
        assert "output format template only" in prompt
        assert "not source text" in prompt
        assert "must never be used as extraction content" in prompt


def test_text_examples_render_tuple_and_completion_delimiters():
    from lightrag.prompt import PROMPTS

    rendered = "\n".join(PROMPTS["entity_extraction_examples"]).format(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
    )

    assert (
        "entity<|#|><entity_name><|#|><entity_type><|#|><entity_description>"
        in rendered
    )
    assert (
        "relation<|#|><source_entity><|#|><target_entity><|#|>"
        "<relationship_keywords><|#|><relationship_description>" in rendered
    )
    assert "<|COMPLETE|>" in rendered
    assert "{tuple_delimiter}" not in rendered
    assert "{completion_delimiter}" not in rendered


def test_json_examples_are_parseable_format_templates():
    """JSON examples must be raw JSON templates with valid endpoint references."""
    from lightrag.prompt import PROMPTS

    for example in PROMPTS["entity_extraction_json_examples"]:
        parsed = json.loads(example)
        assert set(parsed) == {"entities", "relationships"}
        assert isinstance(parsed["entities"], list)
        assert isinstance(parsed["relationships"], list)
        assert parsed["entities"]
        assert parsed["relationships"]

        entity_names = {
            entity["name"] for entity in parsed.get("entities", []) if entity
        }
        for relationship in parsed.get("relationships", []):
            assert relationship["source"] in entity_names
            assert relationship["target"] in entity_names
        assert "<entity_name>" in entity_names


@pytest.mark.parametrize(
    ("prompt_key", "required_substrings", "forbidden_substrings"),
    [
        pytest.param(
            "entity_extraction_user_prompt",
            (
                "output at most {max_total_records} total rows and at most {max_entity_records} entity rows",
                "If the row limit is reached, output `{completion_delimiter}` immediately",
            ),
            (),
            id="text_user",
        ),
        pytest.param(
            "entity_continue_extraction_user_prompt",
            (
                "Any corrected relationship row must be emitted with the literal `relation` prefix",
                "output at most {max_total_records} total rows and at most {max_entity_records} entity rows",
                "may reference entities that were already extracted correctly in the previous response",
            ),
            ("whose source and target entities are both included in this response",),
            id="text_continue",
        ),
        pytest.param(
            "entity_extraction_json_user_prompt",
            (
                "output at most {max_total_records} total records and at most {max_entity_records} entity objects",
                "Only output relationship objects whose `source` and `target` are both included",
            ),
            (),
            id="json_user",
        ),
        pytest.param(
            "entity_continue_extraction_json_user_prompt",
            (
                "output at most {max_total_records} total records and at most {max_entity_records} entity objects",
                "may reference entities already extracted correctly in the previous response",
            ),
            (),
            id="json_continue",
        ),
    ],
)
def test_extraction_user_and_continue_prompts_enforce_limits_and_rules(
    prompt_key: str,
    required_substrings: tuple[str, ...],
    forbidden_substrings: tuple[str, ...],
):
    from lightrag.prompt import PROMPTS

    prompt = PROMPTS[prompt_key]
    for expected in required_substrings:
        assert expected in prompt
    for forbidden in forbidden_substrings:
        assert forbidden not in prompt


# --------------------------------------------------------------------------- #
# 2. ENTITY_TYPES env var guard & text/JSON guidance injection
# --------------------------------------------------------------------------- #


def test_entity_types_env_var_raises_system_exit(tmp_path):
    """LightRAG.__post_init__ raises SystemExit when ENTITY_TYPES env var is set."""
    from lightrag import LightRAG

    with patch.dict(os.environ, {"ENTITY_TYPES": '["Person"]'}):
        with pytest.raises(SystemExit) as exc_info:
            LightRAG(
                working_dir=str(tmp_path),
                llm_model_func=AsyncMock(),
                embedding_func=None,
            )
    assert "ENTITY_TYPES" in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("use_json", "response", "mode_substrings"),
    [
        pytest.param(
            False,
            _TEXT_MODE_RESPONSE,
            (
                "must start with `relation`, never `entity`",
                "After the last entity row, switch prefixes to `relation`",
                "Output at most 100 total rows",
                "Output at most 40 entity rows",
            ),
            id="text_mode",
        ),
        pytest.param(
            True,
            _JSON_MODE_RESPONSE,
            (
                "Output at most 100 total records",
                "Output at most 40 entity objects",
            ),
            id="json_mode",
        ),
    ],
)
async def test_default_guidance_injected_into_prompt(
    use_json: bool, response: str, mode_substrings: tuple[str, ...]
):
    """Default entity_types_guidance and mode-specific kwargs/limits reach the LLM."""
    from lightrag.operate import extract_entities
    from lightrag.prompt import PROMPTS

    global_config = _make_global_config(use_json=use_json)
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = response

    with patch("lightrag.operate.logger"):
        await extract_entities(chunks=_make_chunks(), global_config=global_config)

    assert llm_func.await_count >= 1
    call_kwargs = llm_func.call_args_list[0][1]
    system_prompt = call_kwargs.get("system_prompt", "")
    assert PROMPTS["default_entity_types_guidance"] in system_prompt
    for substring in mode_substrings:
        assert substring in system_prompt
    if use_json:
        assert call_kwargs.get("response_format") == {"type": "json_object"}
        assert call_kwargs.get("entity_extraction") is not True
    else:
        assert not call_kwargs.get("entity_extraction", False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("use_json", "response", "custom_guidance"),
    [
        pytest.param(
            False,
            _TEXT_MODE_RESPONSE,
            "- Widget: A test widget type",
            id="text_mode",
        ),
        pytest.param(
            True,
            _JSON_MODE_RESPONSE,
            "- Gadget: A test gadget type",
            id="json_mode",
        ),
    ],
)
async def test_custom_guidance_overrides_default(
    use_json: bool, response: str, custom_guidance: str
):
    """Custom entity_types_guidance in addon_params overrides default in both modes."""
    from lightrag.operate import extract_entities

    global_config = _make_global_config(
        addon_params={"entity_types_guidance": custom_guidance},
        use_json=use_json,
    )
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = response

    with patch("lightrag.operate.logger"):
        await extract_entities(chunks=_make_chunks(), global_config=global_config)

    system_prompt = llm_func.call_args_list[0][1].get("system_prompt", "")
    assert custom_guidance in system_prompt


@pytest.mark.asyncio
async def test_rebuild_from_cached_fenced_json_uses_json_parser():
    """Cached JSON wrapped in markdown fences must not fall back to text parsing."""
    from lightrag import operate

    fenced_json = f"```json\n{_JSON_MODE_RESPONSE}\n```"

    with patch(
        "lightrag.operate._process_extraction_result",
        side_effect=AssertionError("text parser should not be used"),
    ):
        nodes, edges = await operate._rebuild_from_extraction_result(
            text_chunks_storage=_DummyTextChunksStorage(),
            extraction_result=fenced_json,
            chunk_id="chunk-001",
            timestamp=123,
        )

    assert set(nodes) == {"Alice", "Acme Corp"}
    assert ("Alice", "Acme Corp") in edges
    assert nodes["Alice"][0]["file_path"] == "test.md"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("max_gleaning", "responses"),
    [
        pytest.param(
            0,
            [_TEXT_MODE_MISPREFIXED_RELATION_RESPONSE],
            id="misprefixed_relation_recovered",
        ),
        pytest.param(
            1,
            _TEXT_MODE_GLEANED_RELATION_RESPONSES,
            id="gleaned_relation_merges_after_recovery",
        ),
        pytest.param(
            1,
            _TEXT_MODE_CROSS_PASS_RELATION_RESPONSES,
            id="gleaned_relation_references_prior_entity",
        ),
    ],
)
async def test_text_mode_relation_recovery_and_gleaning(
    max_gleaning: int, responses: list[str]
):
    from lightrag.operate import extract_entities

    global_config = _make_global_config(use_json=False, max_gleaning=max_gleaning)
    llm_func = global_config["llm_model_func"]
    llm_func.side_effect = responses

    with patch("lightrag.operate.logger"):
        chunk_results = await extract_entities(
            chunks=_make_chunks(),
            global_config=global_config,
        )

    entities, relationships = chunk_results[0]
    assert set(entities.keys()) == {"Alice", "Acme Corp"}
    assert len(relationships) == 1
    assert next(iter(relationships.keys())) == ("Alice", "Acme Corp")
    relation_data = next(iter(relationships.values()))[0]
    assert relation_data["src_id"] == "Alice"
    assert relation_data["tgt_id"] == "Acme Corp"


# --------------------------------------------------------------------------- #
# 3. Prompt profile files, validation, and runtime cache refresh
# --------------------------------------------------------------------------- #


def test_addon_params_default_includes_entity_type_prompt_file_env(tmp_path):
    _require_yaml()

    from lightrag import LightRAG

    prompt_dir = tmp_path / "entity_type"
    prompt_dir.mkdir()
    _write_prompt_profile(
        prompt_dir / "entity_type_prompt.sample.yml",
        text_examples=[_text_profile_example("Env Default Example")],
    )

    with patch.dict(
        os.environ,
        {
            "SUMMARY_LANGUAGE": "English",
            "ENTITY_TYPE_PROMPT_FILE": "entity_type_prompt.sample.yml",
        },
    ):
        with _patch_prompt_dir(prompt_dir):
            rag = LightRAG(
                working_dir=str(tmp_path / "rag-default-env"),
                llm_model_func=AsyncMock(),
                embedding_func=_dummy_embedding_func(),
                entity_extraction_use_json=False,
            )

    assert (
        rag.addon_params["entity_type_prompt_file"] == "entity_type_prompt.sample.yml"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("use_json", "guidance", "example_label", "response"),
    [
        pytest.param(
            False,
            "- ExampleType: Injected guidance",
            "Custom Text Example",
            _TEXT_MODE_RESPONSE,
            id="text_mode",
        ),
        pytest.param(
            True,
            "- ExampleType: Injected JSON guidance",
            "Custom Json Example",
            _JSON_MODE_RESPONSE,
            id="json_mode",
        ),
    ],
)
async def test_prompt_file_injects_examples_and_guidance(
    use_json: bool, guidance: str, example_label: str, response: str
):
    _require_yaml()

    from lightrag.operate import extract_entities

    prompt_profile = {
        "entity_types_guidance": guidance,
        "entity_extraction_examples": (
            [] if use_json else [_text_profile_example(example_label)]
        ),
        "entity_extraction_json_examples": (
            [_json_profile_example(example_label)] if use_json else []
        ),
    }

    global_config = _make_global_config(
        prompt_profile=prompt_profile,
        use_json=use_json,
    )
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = response

    with patch("lightrag.operate.logger"):
        await extract_entities(chunks=_make_chunks(), global_config=global_config)

    system_prompt = llm_func.call_args_list[0][1].get("system_prompt", "")
    assert guidance in system_prompt
    assert example_label in system_prompt


@pytest.mark.asyncio
async def test_prompt_file_guidance_falls_back_to_default_when_missing():
    _require_yaml()

    from lightrag.operate import extract_entities
    from lightrag.prompt import PROMPTS

    global_config = _make_global_config(
        prompt_profile={
            "entity_types_guidance": PROMPTS["default_entity_types_guidance"].rstrip(),
            "entity_extraction_examples": [
                _text_profile_example("Fallback Guidance Example")
            ],
            "entity_extraction_json_examples": [],
        },
        use_json=False,
    )
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _TEXT_MODE_RESPONSE

    with patch("lightrag.operate.logger"):
        await extract_entities(chunks=_make_chunks(), global_config=global_config)

    system_prompt = llm_func.call_args_list[0][1].get("system_prompt", "")
    assert PROMPTS["default_entity_types_guidance"] in system_prompt


@pytest.mark.asyncio
async def test_cached_prompt_profile_supplies_merged_guidance():
    from lightrag.operate import extract_entities

    merged_guidance = "- ExampleType: Addon override"
    global_config = _make_global_config(
        prompt_profile={
            "entity_types_guidance": merged_guidance,
            "entity_extraction_examples": [_text_profile_example("Override Example")],
            "entity_extraction_json_examples": [],
        },
        use_json=False,
    )
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _TEXT_MODE_RESPONSE

    with patch("lightrag.operate.logger"):
        await extract_entities(chunks=_make_chunks(), global_config=global_config)

    system_prompt = llm_func.call_args_list[0][1].get("system_prompt", "")
    assert merged_guidance in system_prompt


@pytest.mark.parametrize(
    ("use_json", "filename", "writer_kwargs"),
    [
        pytest.param(
            False,
            "text_only.yml",
            {"text_examples": [_text_profile_example("Text Only Example")]},
            id="text_mode_omits_json_examples",
        ),
        pytest.param(
            True,
            "json_only.yml",
            {"json_examples": [_json_profile_example("Json Only Example")]},
            id="json_mode_omits_text_examples",
        ),
    ],
)
def test_prompt_file_can_omit_other_mode_examples(
    tmp_path, use_json: bool, filename: str, writer_kwargs: dict
):
    _require_yaml()

    from lightrag import LightRAG

    prompt_dir = tmp_path / "entity_type"
    prompt_dir.mkdir()
    _write_prompt_profile(prompt_dir / filename, **writer_kwargs)

    with _patch_prompt_dir(prompt_dir):
        rag = LightRAG(
            working_dir=str(tmp_path / f"rag-{use_json}"),
            llm_model_func=AsyncMock(),
            embedding_func=_dummy_embedding_func(),
            entity_extraction_use_json=use_json,
            addon_params={"entity_type_prompt_file": filename},
        )

    assert rag.addon_params["entity_type_prompt_file"] == filename


@pytest.mark.parametrize(
    ("use_json", "filename", "writer_kwargs", "expected_missing_key"),
    [
        pytest.param(
            False,
            "missing_text_examples.yml",
            {"json_examples": [_json_profile_example("Wrong Mode Only")]},
            "entity_extraction_examples",
            id="text_mode_requires_text_examples",
        ),
        pytest.param(
            True,
            "missing_json_examples.yml",
            {"text_examples": [_text_profile_example("Wrong Mode Only")]},
            "entity_extraction_json_examples",
            id="json_mode_requires_json_examples",
        ),
    ],
)
def test_prompt_file_requires_active_mode_examples(
    tmp_path,
    use_json: bool,
    filename: str,
    writer_kwargs: dict,
    expected_missing_key: str,
):
    _require_yaml()

    from lightrag import LightRAG

    prompt_dir = tmp_path / "entity_type"
    prompt_dir.mkdir()
    _write_prompt_profile(prompt_dir / filename, **writer_kwargs)

    with _patch_prompt_dir(prompt_dir):
        with pytest.raises(ValueError) as exc_info:
            LightRAG(
                working_dir=str(tmp_path / f"rag-missing-{use_json}"),
                llm_model_func=AsyncMock(),
                embedding_func=None,
                entity_extraction_use_json=use_json,
                addon_params={"entity_type_prompt_file": filename},
            )

    assert expected_missing_key in str(exc_info.value)


@pytest.mark.parametrize(
    ("kind", "expected_match"),
    [
        pytest.param("directory_segment", "file name only", id="directory_segments"),
        pytest.param("absolute_path", "file name only", id="absolute_paths"),
        pytest.param(
            "unsupported_extension", "'.yml' or '.yaml'", id="unsupported_extension"
        ),
        pytest.param("malformed_yaml", "invalid YAML", id="malformed_yaml"),
    ],
)
def test_prompt_file_rejects_invalid_inputs(tmp_path, kind: str, expected_match: str):
    _require_yaml()

    from lightrag import LightRAG
    from lightrag.prompt import load_entity_extraction_prompt_profile

    if kind == "malformed_yaml":
        bad_file = tmp_path / "broken.yml"
        bad_file.write_text("entity_types_guidance: [unclosed", encoding="utf-8")
        with pytest.raises(ValueError, match=expected_match):
            load_entity_extraction_prompt_profile(bad_file)
        return

    bad_name = {
        "directory_segment": "../outside.yml",
        "absolute_path": str(tmp_path / "abs.yml"),
        "unsupported_extension": "profile.txt",
    }[kind]
    with pytest.raises(ValueError, match=expected_match):
        LightRAG(
            working_dir=str(tmp_path / f"rag-{kind}"),
            llm_model_func=AsyncMock(),
            embedding_func=None,
            addon_params={"entity_type_prompt_file": bad_name},
        )


@pytest.mark.asyncio
async def test_extract_entities_uses_cached_prompt_profile_without_reloading():
    from lightrag.operate import extract_entities

    cached_profile = {
        "entity_types_guidance": "- ExampleType: Cached guidance",
        "entity_extraction_examples": [_text_profile_example("Cached Text Example")],
        "entity_extraction_json_examples": [],
    }
    global_config = _make_global_config(use_json=False, prompt_profile=cached_profile)
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _TEXT_MODE_RESPONSE

    with patch(
        "lightrag.operate.resolve_entity_extraction_prompt_profile",
        side_effect=AssertionError("should not resolve profile when cache exists"),
    ):
        with patch("lightrag.operate.logger"):
            await extract_entities(chunks=_make_chunks(), global_config=global_config)
            await extract_entities(chunks=_make_chunks(), global_config=global_config)

    system_prompt = llm_func.call_args_list[0][1].get("system_prompt", "")
    assert "Cached Text Example" in system_prompt
    assert "Cached guidance" in system_prompt


def test_sample_prompt_file_matches_builtin_prompt_data():
    _require_yaml()

    from lightrag.prompt import (
        get_default_entity_extraction_prompt_profile,
        load_entity_extraction_prompt_profile,
    )

    sample_file = (
        Path(__file__).resolve().parents[2]
        / "prompts"
        / "samples"
        / "entity_type_prompt.sample.yml"
    )

    loaded_profile = load_entity_extraction_prompt_profile(sample_file)
    assert loaded_profile == get_default_entity_extraction_prompt_profile()


def test_prompt_dir_env_var_overrides_default(tmp_path, monkeypatch):
    _require_yaml()

    from lightrag.prompt import (
        get_entity_type_prompt_dir,
        resolve_entity_type_prompt_path,
    )

    monkeypatch.setenv("PROMPT_DIR", str(tmp_path))
    expected_dir = (tmp_path / "entity_type").resolve()
    assert get_entity_type_prompt_dir() == expected_dir
    resolved = resolve_entity_type_prompt_path("custom.yml")
    assert resolved == expected_dir / "custom.yml"


def test_prompt_dir_defaults_to_cwd_relative(tmp_path, monkeypatch):
    _require_yaml()

    from lightrag.prompt import get_entity_type_prompt_dir

    monkeypatch.delenv("PROMPT_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    assert (
        get_entity_type_prompt_dir() == (tmp_path / "prompts" / "entity_type").resolve()
    )


def test_addon_guidance_overrides_file_profile(tmp_path):
    _require_yaml()

    from lightrag.prompt import resolve_entity_extraction_prompt_profile

    prompt_dir = tmp_path / "entity_type"
    prompt_dir.mkdir()
    _write_prompt_profile(
        prompt_dir / "profile.yml",
        guidance="- FileType: from file",
        text_examples=[_text_profile_example("Merged Example")],
    )

    with _patch_prompt_dir(prompt_dir):
        profile = resolve_entity_extraction_prompt_profile(
            addon_params={
                "entity_type_prompt_file": "profile.yml",
                "entity_types_guidance": "- AddonType: from addon_params",
            },
            use_json=False,
        )

    assert profile["entity_types_guidance"] == "- AddonType: from addon_params"
    assert any(
        "Merged Example" in example for example in profile["entity_extraction_examples"]
    )


def test_explicit_addon_params_still_picks_up_env_defaults(tmp_path, monkeypatch):
    """Passing addon_params explicitly must not drop env-based defaults."""
    _require_yaml()

    from lightrag import LightRAG

    prompt_dir = tmp_path / "entity_type"
    prompt_dir.mkdir()
    _write_prompt_profile(
        prompt_dir / "from_env.yml",
        text_examples=[_text_profile_example("Env Example")],
    )

    monkeypatch.setenv("ENTITY_TYPE_PROMPT_FILE", "from_env.yml")

    with _patch_prompt_dir(prompt_dir):
        rag = LightRAG(
            working_dir=str(tmp_path / "rag-env-default"),
            llm_model_func=AsyncMock(),
            embedding_func=_dummy_embedding_func(),
            entity_extraction_use_json=False,
            addon_params={"language": "English"},
        )

    assert rag.addon_params["entity_type_prompt_file"] == "from_env.yml"


def test_runtime_addon_params_item_update_refreshes_cached_values(tmp_path):
    _require_yaml()

    from lightrag import LightRAG

    prompt_dir = tmp_path / "entity_type"
    prompt_dir.mkdir()
    _write_prompt_profile(
        prompt_dir / "initial.yml",
        text_examples=[_text_profile_example("Initial Example")],
    )
    _write_prompt_profile(
        prompt_dir / "updated.yml",
        guidance="- UpdatedType: runtime update",
        text_examples=[_text_profile_example("Updated Example")],
    )

    with _patch_prompt_dir(prompt_dir):
        rag = LightRAG(
            working_dir=str(tmp_path / "rag-runtime-update"),
            llm_model_func=AsyncMock(),
            embedding_func=_dummy_embedding_func(),
            entity_extraction_use_json=False,
            addon_params={
                "entity_type_prompt_file": "initial.yml",
                "language": "English",
            },
        )
        rag.addon_params["entity_type_prompt_file"] = "updated.yml"
        rag.addon_params["language"] = "French"
        global_config = rag._build_global_config()

    assert global_config["addon_params"]["language"] == "French"
    assert global_config["_resolved_summary_language"] == "French"
    assert (
        global_config["_entity_extraction_prompt_profile"]["entity_types_guidance"]
        == "- UpdatedType: runtime update"
    )
    assert any(
        "Updated Example" in example
        for example in global_config["_entity_extraction_prompt_profile"][
            "entity_extraction_examples"
        ]
    )


def test_runtime_addon_params_replacement_refreshes_cached_values(tmp_path):
    _require_yaml()

    from lightrag import LightRAG

    rag = LightRAG(
        working_dir=str(tmp_path / "rag-runtime-replace"),
        llm_model_func=AsyncMock(),
        embedding_func=_dummy_embedding_func(),
        entity_extraction_use_json=False,
        addon_params={"language": "English"},
    )

    rag.addon_params = {
        "language": "German",
        "entity_types_guidance": "- ReplacementType: runtime replace",
    }
    global_config = rag._build_global_config()

    assert global_config["addon_params"]["language"] == "German"
    assert global_config["_resolved_summary_language"] == "German"
    assert (
        global_config["_entity_extraction_prompt_profile"]["entity_types_guidance"]
        == "- ReplacementType: runtime replace"
    )


def test_runtime_mode_flip_invalidates_cached_prompt_profile(tmp_path):
    _require_yaml()

    from lightrag import LightRAG

    prompt_dir = tmp_path / "entity_type"
    prompt_dir.mkdir()
    _write_prompt_profile(
        prompt_dir / "text_only.yml",
        text_examples=[_text_profile_example("Text Only Example")],
    )

    with _patch_prompt_dir(prompt_dir):
        rag = LightRAG(
            working_dir=str(tmp_path / "rag-mode-flip"),
            llm_model_func=AsyncMock(),
            embedding_func=_dummy_embedding_func(),
            entity_extraction_use_json=False,
            addon_params={"entity_type_prompt_file": "text_only.yml"},
        )

        rag._build_global_config()
        rag.entity_extraction_use_json = True

        with pytest.raises(ValueError) as exc_info:
            rag._build_global_config()

    assert "entity_extraction_json_examples" in str(exc_info.value)


# --------------------------------------------------------------------------- #
# 4. Section Context (heading breadcrumb) injection, cleaning, and truncation
# --------------------------------------------------------------------------- #

_SECTION_MARKER = "---Section Context---"


def _render_user_prompt(use_json: bool, heading_context_block: str) -> str:
    from lightrag.prompt import PROMPTS

    if use_json:
        return PROMPTS["entity_extraction_json_user_prompt"].format(
            max_total_records=100,
            max_entity_records=40,
            language="English",
            entity_types_guidance="- Person: humans",
            input_text="Alice founded Acme Corp.",
            heading_context_block=heading_context_block,
        )
    return PROMPTS["entity_extraction_user_prompt"].format(
        max_total_records=100,
        max_entity_records=40,
        completion_delimiter="<|COMPLETE|>",
        language="English",
        input_text="Alice founded Acme Corp.",
        heading_context_block=heading_context_block,
    )


def _section_block(heading_path: str) -> str:
    from lightrag.prompt import PROMPTS

    return PROMPTS["entity_extraction_section_context"].format(
        heading_path=heading_path
    )


def test_user_prompts_keep_single_real_input_text_section():
    """Only the rendered task prompt should carry the real input section marker."""
    for use_json in (False, True):
        markers = [
            line
            for line in _render_user_prompt(use_json, "").splitlines()
            if line == "---Input Text---"
        ]
        assert len(markers) == 1


@pytest.mark.parametrize("use_json", [False, True], ids=["text_mode", "json_mode"])
def test_user_prompt_section_context_hidden_and_byte_identical_when_no_heading(
    use_json: bool,
):
    """Empty heading_context_block disappears and matches the placeholder-free template."""
    from lightrag.prompt import PROMPTS

    rendered = _render_user_prompt(use_json, "")
    assert _SECTION_MARKER not in rendered

    key = (
        "entity_extraction_json_user_prompt"
        if use_json
        else "entity_extraction_user_prompt"
    )
    baseline_template = PROMPTS[key].replace("{heading_context_block}", "")
    format_kwargs = {
        "max_total_records": 100,
        "max_entity_records": 40,
        "language": "English",
        "input_text": "Alice founded Acme Corp.",
    }
    if use_json:
        format_kwargs["entity_types_guidance"] = "- Person: humans"
    else:
        format_kwargs["completion_delimiter"] = "<|COMPLETE|>"
    assert rendered == baseline_template.format(**format_kwargs)


def test_json_user_prompt_includes_section_context_when_heading_present():
    rendered = _render_user_prompt(True, _section_block("Methods → Data Collection"))
    assert _SECTION_MARKER in rendered
    assert "Methods → Data Collection" in rendered
    assert "Methods → Data Collection\n\n---Input Text---" in rendered


def test_section_context_breadcrumb_is_not_at_line_start():
    """A heading resembling a prompt marker stays inline and cannot forge a section."""
    block = _section_block("---Output---")
    assert "\n---Output---" not in block
    assert "---Output---" in block


def test_extraction_system_prompts_reference_section_context():
    """Both system prompts carry the static conditional instruction."""
    from lightrag.prompt import PROMPTS

    for key in (
        "entity_extraction_system_prompt",
        "entity_extraction_json_system_prompt",
    ):
        assert _SECTION_MARKER in PROMPTS[key]
        assert "only as background" in PROMPTS[key]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("heading", "expect_present"),
    [
        pytest.param(
            {"level": 2, "heading": "Data Collection", "parent_headings": ["Methods"]},
            True,
            id="with_heading",
        ),
        pytest.param(None, False, id="without_heading"),
    ],
)
async def test_extract_entities_section_context_by_heading_presence(
    heading: dict | None, expect_present: bool
):
    from lightrag.operate import extract_entities

    global_config = _make_global_config(use_json=False)
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _TEXT_MODE_RESPONSE

    chunks = _make_chunks("Alice founded Acme Corp.")
    if heading is not None:
        chunks["chunk-001"]["heading"] = heading

    with patch("lightrag.operate.logger"):
        await extract_entities(chunks=chunks, global_config=global_config)

    assert llm_func.await_count >= 1
    user_prompt = llm_func.call_args_list[0][0][0]
    if expect_present:
        assert _SECTION_MARKER in user_prompt
        assert "Methods → Data Collection\n\n---Input Text---" in user_prompt
    else:
        assert _SECTION_MARKER not in user_prompt


@pytest.mark.parametrize(
    ("formatter_name", "disable_cap"),
    [
        pytest.param("format_heading_context", False, id="heading_context_capped"),
        pytest.param("format_heading_context", True, id="heading_context_uncapped"),
        pytest.param("format_parent_headings", False, id="parent_headings_capped"),
        pytest.param("format_parent_headings", True, id="parent_headings_uncapped"),
    ],
)
def test_heading_formatter_per_level_char_cap(formatter_name: str, disable_cap: bool):
    from lightrag import chunk_schema

    formatter = getattr(chunk_schema, formatter_name)
    max_chars = chunk_schema.DEFAULT_HEADING_LEVEL_MAX_CHARS
    long_title = "B" * 300 if disable_cap else "A" * (max_chars + 50)
    if formatter_name == "format_heading_context":
        chunk = {"heading": {"level": 1, "heading": long_title, "parent_headings": []}}
    else:
        chunk = {
            "heading": {"level": 2, "heading": "Leaf", "parent_headings": [long_title]}
        }

    if disable_cap:
        assert formatter(chunk, max_heading_len=0) == long_title
    else:
        out = formatter(chunk)
        assert out.endswith("…")
        assert len(out) == max_chars


@pytest.mark.parametrize(
    ("formatter_name", "chunk", "expected", "expected_levels"),
    [
        pytest.param(
            "format_parent_headings",
            {"content": "...", "chunk_order_index": 0},
            "",
            None,
            id="parent_headings_empty",
        ),
        pytest.param(
            "format_parent_headings",
            {"heading": {"level": 2, "heading": "Leaf", "parent_headings": ["h1", "h2"]}},
            "h1 → h2",
            None,
            id="parent_headings_excludes_leaf",
        ),
        pytest.param(
            "format_parent_headings",
            {
                "heading": {
                    "level": 2,
                    "heading": "Leaf",
                    "parent_headings": ["A→B", "x" + chr(0) + "y" + chr(0x200B) + "z"],
                }
            },
            "A B → xyz",
            None,
            id="parent_headings_cleaning_matches_extraction",
        ),
        pytest.param(
            "format_heading_context",
            {"heading": {"level": 2, "heading": "C", "parent_headings": ["A→B"]}},
            "A B → C",
            ["A B", "C"],
            id="heading_context_arrow_does_not_forge_level",
        ),
    ],
)
def test_heading_formatting_preserves_structure_and_cleans_symbols(
    formatter_name: str,
    chunk: dict,
    expected: str,
    expected_levels: list[str] | None,
):
    from lightrag import chunk_schema

    out = getattr(chunk_schema, formatter_name)(chunk)
    assert out == expected
    if expected_levels is not None:
        assert out.split(chunk_schema.HEADING_BREADCRUMB_SEP) == expected_levels


class _FakeChunksDB:
    """Minimal text_chunks_db for _attach_content_headings: get_by_ids + config."""

    def __init__(self, data_by_id: dict, tokenizer):
        self._data = data_by_id
        self.global_config = {"tokenizer": tokenizer}

    async def get_by_ids(self, ids):
        return [self._data.get(i) for i in ids]


@pytest.mark.asyncio
@pytest.mark.parametrize("deep_path", [True, False], ids=["deep_collapsed", "short_intact"])
async def test_attach_content_headings_budgeting(deep_path: bool):
    from lightrag.chunk_schema import HEADING_BREADCRUMB_SEP
    from lightrag.constants import DEFAULT_MAX_SECTION_CONTEXT_TOKENS
    from lightrag.operate import _attach_content_headings

    tok = make_dummy_tokenizer()
    parents = [f"Level{i:02d}" for i in range(100)] if deep_path else ["h1", "h2"]
    level = 99 if deep_path else 2
    db = _FakeChunksDB(
        {"c1": {"heading": {"level": level, "heading": "Leaf", "parent_headings": parents}}},
        tok,
    )
    chunks = [{"chunk_id": "c1"}]

    await _attach_content_headings(chunks, db)

    out = chunks[0]["content_headings"]
    if deep_path:
        assert len(tok.encode(out)) <= DEFAULT_MAX_SECTION_CONTEXT_TOKENS
        assert f"{HEADING_BREADCRUMB_SEP}…{HEADING_BREADCRUMB_SEP}" in out
        assert "Level50" not in out
    else:
        assert out == "h1 → h2"


@pytest.mark.parametrize(
    "case",
    [
        "collapse_middle",
        "hard_cap_dense_short",
        "hard_cap_collapsed",
        "budget_zero_disabled",
        "no_tokenizer_noop",
    ],
)
def test_truncate_section_context_budget_modes(case: str):
    from lightrag.chunk_schema import HEADING_BREADCRUMB_SEP
    from lightrag.operate import _truncate_section_context

    tok = make_dummy_tokenizer()
    if case == "collapse_middle":
        levels = [f"Level{i:02d}" for i in range(100)]
        path = HEADING_BREADCRUMB_SEP.join(levels)
        out = _truncate_section_context(path, tok, 40)
        assert out == (
            f"{levels[0]}{HEADING_BREADCRUMB_SEP}…{HEADING_BREADCRUMB_SEP}{levels[-1]}"
        )
        assert "Level50" not in out
        assert len(tok.encode(out)) <= 40
    elif case == "hard_cap_dense_short":
        path = HEADING_BREADCRUMB_SEP.join(["A" * 50, "B" * 50])
        out = _truncate_section_context(path, tok, 10)
        assert out != path
        assert out.endswith("…")
        assert len(tok.encode(out)) <= 10
    elif case == "hard_cap_collapsed":
        levels = [f"Level{i:02d}" for i in range(100)]
        path = HEADING_BREADCRUMB_SEP.join(levels)
        out = _truncate_section_context(path, tok, 8)
        assert out.endswith("…")
        assert len(tok.encode(out)) <= 8
    elif case == "budget_zero_disabled":
        path = "X" * 1000
        assert _truncate_section_context(path, tok, 0) == path
    else:
        path = "X" * 1000
        assert _truncate_section_context(path, None, 256) == path


def test_truncate_section_context_accounts_for_multitoken_ellipsis():
    """The hard cap must reserve the tokenizer's actual ellipsis cost."""
    from lightrag.operate import _truncate_section_context

    class TwoTokenEllipsisTokenizer(TokenizerInterface):
        def encode(self, content: str):
            tokens = []
            for ch in content:
                if ch == "…":
                    tokens.extend([0x110000, 0x110001])
                else:
                    tokens.append(ord(ch))
            return tokens

        def decode(self, tokens):
            return "".join(chr(token) for token in tokens if token <= 0x10FFFF)

    tok = Tokenizer("two-token-ellipsis", TwoTokenEllipsisTokenizer())
    out = _truncate_section_context("A" * 20, tok, 10)
    assert out == "A" * 8 + "…"
    assert len(tok.encode(out)) <= 10


def test_heading_level_cap_below_one_third_of_token_budget():
    """Invariant guard: collapsed first+leaf must fit the token budget."""
    from lightrag.constants import (
        DEFAULT_HEADING_LEVEL_MAX_CHARS,
        DEFAULT_MAX_SECTION_CONTEXT_TOKENS,
    )

    assert DEFAULT_HEADING_LEVEL_MAX_CHARS * 3 < DEFAULT_MAX_SECTION_CONTEXT_TOKENS


@pytest.mark.asyncio
async def test_extract_entities_bounds_pathological_heading_in_prompt():
    """A chunk with an absurdly long heading must not inject it verbatim."""
    from lightrag.chunk_schema import DEFAULT_HEADING_LEVEL_MAX_CHARS
    from lightrag.operate import extract_entities

    global_config = _make_global_config(use_json=False)
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _TEXT_MODE_RESPONSE

    long_title = "Z" * 500
    chunks = _make_chunks("Alice founded Acme Corp.")
    chunks["chunk-001"]["heading"] = {
        "level": 1,
        "heading": long_title,
        "parent_headings": [],
    }

    with patch("lightrag.operate.logger"):
        await extract_entities(chunks=chunks, global_config=global_config)

    user_prompt = llm_func.call_args_list[0][0][0]
    assert _SECTION_MARKER in user_prompt
    assert long_title not in user_prompt
    assert "Z" * DEFAULT_HEADING_LEVEL_MAX_CHARS not in user_prompt


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("a\x00b\u200bc\ufeff", "abc", id="nul_zwsp_bom"),
        pytest.param("x\x07y", "xy", id="bel"),
        pytest.param("p\x1c\x1fq", "pq", id="file_unit_separators"),
        pytest.param("方法 → 数据采集 (2024)!", "方法 数据采集 (2024)!", id="cjk_and_arrow"),
        pytest.param("数据采集", "数据采集", id="adjacent_cjk"),
        pytest.param("a\nb\tc", "a b c", id="newline_tab_collapse"),
        pytest.param("a \x00 b", "a b", id="control_between_spaces"),
    ],
)
def test_clean_heading_text(raw: str, expected: str):
    from lightrag.chunk_schema import _clean_heading_text

    assert _clean_heading_text(raw) == expected
