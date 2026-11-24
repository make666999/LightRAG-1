"""LMDeploy tests: do_sample/max_tokens forwarding, version guard, truncation, and retry policy."""

import asyncio
import sys
from types import SimpleNamespace

import pytest

from lightrag.llm.lmdeploy import lmdeploy_model_if_cache
from lightrag.utils import TruncatedResponse

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


class FakeRequestError(RuntimeError):
    """Stand-in for lmdeploy.serve.core.exceptions.RequestError."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"fake lmdeploy request error ({status_code})")


async def _instant_sleep(_seconds: float) -> None:
    return None


def _install_fake_lmdeploy(
    monkeypatch, *, less_than_0_6_0: bool = False, generate=None
) -> dict:
    captured_gen_config_kwargs: dict = {}

    class FakeGenerationConfig:
        def __init__(self, **kwargs):
            captured_gen_config_kwargs.update(kwargs)

    if generate is None:

        async def generate(*_args, **_kwargs):
            yield SimpleNamespace(response="{}", finish_reason="stop")

    monkeypatch.setattr(
        "lightrag.llm.lmdeploy.initialize_lmdeploy_pipeline",
        lambda **_kwargs: SimpleNamespace(generate=generate),
    )
    version_info = (0, 5, 3) if less_than_0_6_0 else (0, 6, 1)
    monkeypatch.setitem(
        sys.modules,
        "lmdeploy",
        SimpleNamespace(
            __version__=".".join(map(str, version_info)),
            version_info=version_info,
            GenerationConfig=FakeGenerationConfig,
        ),
    )
    monkeypatch.setitem(sys.modules, "lmdeploy.serve", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "lmdeploy.serve.core", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "lmdeploy.serve.core.exceptions",
        SimpleNamespace(RequestError=FakeRequestError),
    )
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    return captured_gen_config_kwargs


@pytest.mark.parametrize(
    "call_kwargs,expected_do_sample",
    [
        pytest.param({}, False, id="omitted-defaults-to-false"),
        pytest.param({"do_sample": False}, False, id="explicit-false-preserved"),
        pytest.param({"do_sample": True}, True, id="explicit-true-preserved"),
    ],
)
async def test_do_sample_forwarded_unchanged_on_modern_lmdeploy(
    monkeypatch, call_kwargs, expected_do_sample
):
    captured = _install_fake_lmdeploy(monkeypatch, less_than_0_6_0=False)

    result = await lmdeploy_model_if_cache(
        model="lmdeploy-model", prompt="hello", **call_kwargs
    )

    assert result == "{}"
    assert captured["do_sample"] is expected_do_sample
    assert captured["max_new_tokens"] == 512
    assert "max_tokens" not in captured


@pytest.mark.parametrize(
    ("generation_kwargs", "expected_max_new_tokens"),
    [
        pytest.param({"max_tokens": 37}, 37, id="generic-max-tokens"),
        pytest.param(
            {"max_tokens": 37, "max_new_tokens": 41},
            41,
            id="native-value-takes-precedence",
        ),
    ],
)
async def test_generation_token_limit_precedence(
    monkeypatch, generation_kwargs, expected_max_new_tokens
):
    captured = _install_fake_lmdeploy(monkeypatch, less_than_0_6_0=False)

    result = await lmdeploy_model_if_cache(
        model="lmdeploy-model", prompt="hello", **generation_kwargs
    )

    assert result == "{}"
    assert captured["max_new_tokens"] == expected_max_new_tokens
    assert "max_tokens" not in captured


@pytest.mark.parametrize(
    "do_sample_kwarg",
    [
        pytest.param({}, id="omitted"),
        pytest.param({"do_sample": False}, id="explicit-false"),
        pytest.param({"do_sample": True}, id="explicit-true"),
    ],
)
async def test_pre_0_6_0_version_guard_still_raises(monkeypatch, do_sample_kwarg):
    _install_fake_lmdeploy(monkeypatch, less_than_0_6_0=True)

    with pytest.raises(RuntimeError, match="do_sample.*not supported"):
        await lmdeploy_model_if_cache(
            model="lmdeploy-model", prompt="hello", **do_sample_kwarg
        )


@pytest.mark.parametrize(
    ("finish_reason", "expected_type"),
    [
        pytest.param("stop", str, id="natural-stop"),
        pytest.param("length", TruncatedResponse, id="token-limit"),
    ],
)
async def test_generation_finish_reason_controls_truncation_marker(
    monkeypatch, finish_reason, expected_type
):
    async def fake_generate(*_args, **_kwargs):
        yield SimpleNamespace(text="generated text", finish_reason=finish_reason)

    _install_fake_lmdeploy(monkeypatch, generate=fake_generate)

    result = await lmdeploy_model_if_cache(model="lmdeploy-model", prompt="hello")

    assert type(result) is expected_type
    assert result == "generated text"


async def test_legacy_response_attribute_remains_supported(monkeypatch):
    async def fake_generate(*_args, **_kwargs):
        yield SimpleNamespace(response="legacy text", finish_reason="stop")

    _install_fake_lmdeploy(monkeypatch, generate=fake_generate)

    result = await lmdeploy_model_if_cache(model="lmdeploy-model", prompt="hello")

    assert type(result) is str
    assert result == "legacy text"


async def test_retries_on_transient_engine_unavailable_error(monkeypatch):
    attempts = {"count": 0}

    async def fake_generate(*_args, **_kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise FakeRequestError(503)
        yield SimpleNamespace(text="recovered", finish_reason="stop")

    _install_fake_lmdeploy(monkeypatch, generate=fake_generate)

    result = await lmdeploy_model_if_cache(model="lmdeploy-model", prompt="hello")

    assert result == "recovered"
    assert attempts["count"] == 3


@pytest.mark.parametrize(
    ("status_code", "expected_attempts"),
    [
        pytest.param(400, 1, id="permanent-400-fails-fast"),
        pytest.param(503, 3, id="transient-503-exhausts-and-reraises"),
    ],
)
async def test_retry_error_classification_and_reraise(
    monkeypatch, status_code, expected_attempts
):
    attempts = {"count": 0}

    async def fake_generate(*_args, **_kwargs):
        attempts["count"] += 1
        raise FakeRequestError(status_code)
        yield  # pragma: no cover

    _install_fake_lmdeploy(monkeypatch, generate=fake_generate)

    with pytest.raises(FakeRequestError, match=rf"\({status_code}\)"):
        await lmdeploy_model_if_cache(model="lmdeploy-model", prompt="hello")

    assert attempts["count"] == expected_attempts
