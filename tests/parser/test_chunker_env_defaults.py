"""Empty or malformed CHUNK_* environment variables must not crash defaults.

Covers:
- ``CHUNK_R_SEPARATORS`` (fallback, honoring valid JSON, warning cache + test isolation),
- ``CHUNK_*_SIZE`` / ``CHUNK_*_OVERLAP_SIZE`` (fallback, valid value, malformed ValueError),
- ``CHUNK_V_BUFFER_SIZE`` (fallback and valid value),
- ``LightRAG`` init handling of empty vs. malformed ``CHUNK_*`` variables.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lightrag.constants import DEFAULT_CHUNK_P_SIZE, DEFAULT_R_SEPARATORS
from lightrag.parser.routing import default_chunker_config

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# CHUNK_R_SEPARATORS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_value", ["", "  ", "not-json", "[1,2]"])
def test_empty_or_invalid_chunk_r_separators_falls_back(
    env_value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHUNK_R_SEPARATORS", env_value)
    separators = default_chunker_config()["recursive_character"]["separators"]
    assert separators == list(DEFAULT_R_SEPARATORS)


def test_valid_chunk_r_separators_are_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_R_SEPARATORS", '["\\n\\n", "\\n", " "]')
    separators = default_chunker_config()["recursive_character"]["separators"]
    assert separators == ["\n\n", "\n", " "]


def test_out_of_bounds_env_cascade_is_cached_and_warned_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A static bad environment value must not warn once per document/config read."""
    assert _warnings_for_one_env_value(monkeypatch) == 1


def test_the_env_cache_does_not_leak_between_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deliberately reuses the value of the test above, byte for byte.

    The cache is process-wide and keyed on the raw environment string, so
    without the ``_reset_r_separator_caches`` autouse fixture this test sees
    zero warnings. Tests must not have to invent globally-unique separator
    strings to stay independent of one another.
    """
    assert _warnings_for_one_env_value(monkeypatch) == 1


def _warnings_for_one_env_value(monkeypatch: pytest.MonkeyPatch) -> int:
    """Load one out-of-bounds cascade twice; return the correction-warning count."""
    import lightrag.chunker.recursive_character as recursive_character

    separators = [f"env-cache-{index}" for index in range(70)]
    monkeypatch.setenv("CHUNK_R_SEPARATORS", json.dumps(separators))
    warnings: list[str] = []
    monkeypatch.setattr(recursive_character.logger, "warning", warnings.append)

    first = default_chunker_config()["recursive_character"]["separators"]
    second = default_chunker_config()["recursive_character"]["separators"]

    assert first == separators[:64]
    assert second == separators[:64]
    return sum(
        "[CHUNK_R_SEPARATORS] separator cascade" in message for message in warnings
    )


# ---------------------------------------------------------------------------
# CHUNK_*_SIZE / CHUNK_*_OVERLAP_SIZE
# ---------------------------------------------------------------------------

# Every variable is read by the shared ``_chunk_env_int``, so each variable
# is checked once (its own default and slot) and the blank and malformed
# spellings are swept on one of them: the spellings only differ inside it.
_CHUNK_ENVS = [
    ("CHUNK_P_SIZE", "paragraph_semantic", "chunk_token_size", DEFAULT_CHUNK_P_SIZE),
    ("CHUNK_F_SIZE", "fixed_token", "chunk_token_size", None),
    ("CHUNK_R_SIZE", "recursive_character", "chunk_token_size", None),
    ("CHUNK_V_SIZE", "semantic_vector", "chunk_token_size", None),
    ("CHUNK_F_OVERLAP_SIZE", "fixed_token", "chunk_overlap_token_size", None),
    ("CHUNK_R_OVERLAP_SIZE", "recursive_character", "chunk_overlap_token_size", None),
    ("CHUNK_P_OVERLAP_SIZE", "paragraph_semantic", "chunk_overlap_token_size", None),
]
_BLANKS = ["", "  ", "\t"]
_MALFORMED = ["20OO", "1.5", "abc"]


def _one_spelling_per_env(spellings):
    """Every env with the first spelling, plus the other spellings on the first env."""
    cases = [
        pytest.param(*env, spellings[0], id=f"{env[0]}-{spellings[0]!r}")
        for env in _CHUNK_ENVS
    ]
    cases += [
        pytest.param(*_CHUNK_ENVS[0], value, id=f"{_CHUNK_ENVS[0][0]}-{value!r}")
        for value in spellings[1:]
    ]
    return cases


@pytest.mark.parametrize(
    ("env_key", "strategy", "field", "expected", "env_value"),
    _one_spelling_per_env(_BLANKS),
)
def test_empty_strategy_chunk_envs_fall_back(
    env_key: str,
    strategy: str,
    field: str,
    expected: int | None,
    env_value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(env_key, env_value)
    cfg = default_chunker_config()[strategy]
    if expected is None:
        assert field not in cfg
    else:
        assert cfg[field] == expected


def test_valid_chunk_f_size_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_F_SIZE", "900")
    assert default_chunker_config()["fixed_token"]["chunk_token_size"] == 900


@pytest.mark.parametrize(
    ("env_key", "strategy", "field", "expected", "env_value"),
    _one_spelling_per_env(_MALFORMED),
)
def test_malformed_chunk_size_envs_raise(
    env_key: str,
    strategy: str,
    field: str,
    expected: int | None,
    env_value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del strategy, field, expected  # shared case table; only the key matters here
    monkeypatch.setenv(env_key, env_value)
    with pytest.raises(ValueError, match=env_key):
        default_chunker_config()


def _patch_lightrag_storages(monkeypatch: pytest.MonkeyPatch):
    """Stub storage verification/instantiation and return a minimal EmbeddingFunc."""
    from lightrag.utils import EmbeddingFunc

    monkeypatch.setattr(
        "lightrag.lightrag.verify_storage_implementation", lambda *a: None
    )
    monkeypatch.setattr("lightrag.lightrag.check_storage_env_vars", lambda *a: None)

    class _DummyStorage:
        """No-op storage double for LightRAG.__init__ env-parsing tests."""

        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(
        "lightrag.lightrag.get_storage_class", lambda *a, **k: _DummyStorage
    )

    async def _emb(texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), 8))

    return EmbeddingFunc(embedding_dim=8, max_token_size=100, func=_emb)


@pytest.mark.parametrize("env_value", [""])
def test_empty_chunk_size_env_does_not_crash_lightrag_init(
    env_value: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lightrag.lightrag import LightRAG

    monkeypatch.setenv("CHUNK_SIZE", env_value)
    monkeypatch.setenv("CHUNK_OVERLAP_SIZE", env_value)
    emb = _patch_lightrag_storages(monkeypatch)

    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        embedding_func=emb,
        llm_model_func=lambda *a, **k: "ok",
    )
    assert rag.chunk_token_size == 1200
    assert rag.chunk_overlap_token_size == 100


@pytest.mark.parametrize(
    "env_key",
    [
        "CHUNK_SIZE",
        "CHUNK_OVERLAP_SIZE",
        "CHUNK_P_SIZE",
        "CHUNK_F_SIZE",
        "CHUNK_R_SIZE",
        "CHUNK_V_SIZE",
    ],
)
def test_malformed_chunk_size_envs_raise_in_lightrag_init(
    env_key: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed non-empty env values must raise ValueError on LightRAG init."""
    from lightrag.lightrag import LightRAG

    monkeypatch.setenv(env_key, "20OO")
    emb = _patch_lightrag_storages(monkeypatch)

    with pytest.raises(ValueError, match=env_key):
        LightRAG(
            working_dir=str(tmp_path / "wd"),
            addon_params={"chunker": {"paragraph_semantic": {}}},
            embedding_func=emb,
            llm_model_func=lambda *a, **k: "ok",
        )


# ---------------------------------------------------------------------------
# CHUNK_V_BUFFER_SIZE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_value", [""])
def test_empty_chunk_v_buffer_size_falls_back(
    env_value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHUNK_V_BUFFER_SIZE", env_value)
    assert default_chunker_config()["semantic_vector"]["buffer_size"] == 1


def test_valid_chunk_v_buffer_size_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_V_BUFFER_SIZE", "3")
    assert default_chunker_config()["semantic_vector"]["buffer_size"] == 3
