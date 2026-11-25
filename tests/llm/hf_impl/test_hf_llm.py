"""HuggingFace generation tests: device placement, token-limit precedence, truncation, and chat-template fallback."""

from __future__ import annotations

import importlib
import sys
import types
from copy import deepcopy

import pytest

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


class FakeDevice:
    def __init__(self, name):
        self.type = name.type if isinstance(name, FakeDevice) else name.split(":")[0]

    def __eq__(self, other):
        return isinstance(other, FakeDevice) and other.type == self.type

    def __hash__(self):
        return hash(self.type)

    def __repr__(self):
        return f"FakeDevice({self.type!r})"


class FakeTensor:
    """Minimal stand-in for torch.Tensor device/indexing semantics in hf_model_if_cache."""

    def __init__(self, data, device="cpu", host_has_cuda=True):
        self.data = data
        self.device = FakeDevice(device)
        self._host_has_cuda = host_has_cuda

    def to(self, device):
        device = FakeDevice(device)
        if device.type == "cuda" and not self._host_has_cuda:
            raise RuntimeError("Torch not compiled with CUDA enabled")
        return FakeTensor(self.data, device, self._host_has_cuda)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return FakeTensor(self.data[idx], self.device, self._host_has_cuda)

    def item(self):
        return self.data

    def __repr__(self):
        return f"FakeTensor(data={self.data!r}, device={self.device.type!r})"


class FakeModel:
    def __init__(self, device):
        self.device = FakeDevice(device)
        self.received_kwargs = None
        self.generated_ids = [901, 902]
        self.generation_config = types.SimpleNamespace(eos_token_id=0)

    def generate(self, **kwargs):
        self.received_kwargs = kwargs
        input_ids = kwargs["input_ids"]
        if input_ids.device.type != self.device.type:
            raise RuntimeError(
                "Expected all tensors to be on the same device, but found "
                f"at least two devices, {self.device.type} and "
                f"{input_ids.device.type}:0!"
            )
        prompt_ids = input_ids.data[0]
        generated_ids = prompt_ids + self.generated_ids
        return FakeTensor([generated_ids], self.device, input_ids._host_has_cuda)


class FakeTokenizer:
    def __init__(self, prompt_ids, host_has_cuda=True):
        self._prompt_ids = list(prompt_ids)
        self._host_has_cuda = host_has_cuda
        self.eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "<rendered-prompt>"

    def __call__(self, text, return_tensors="pt", padding=True, truncation=True):
        return {
            "input_ids": FakeTensor(
                [list(self._prompt_ids)], "cpu", self._host_has_cuda
            ),
            "attention_mask": FakeTensor(
                [[1] * len(self._prompt_ids)], "cpu", self._host_has_cuda
            ),
        }

    def decode(self, tensor, skip_special_tokens=True):
        return f"decoded:{tensor.data}"


class PromptTokenizer(FakeTokenizer):
    """Tokenizer double exercising apply_chat_template fallback branches."""

    def __init__(self, mode: str):
        super().__init__([11, 12, 13], host_has_cuda=False)
        self.mode = mode
        self.rendered_messages = []
        self.input_prompt = None

    def apply_chat_template(self, messages, **kwargs):
        self.rendered_messages.append(deepcopy(messages))
        if self.mode == "missing":
            raise ValueError("tokenizer.chat_template is not set")
        if self.mode == "no_system" and messages[0]["role"] == "system":
            raise ValueError("System role not supported")
        return " | ".join(message["content"] for message in messages)

    def __call__(self, text, **kwargs):
        self.input_prompt = text
        return super().__call__(text, **kwargs)


def _make_hf_module(
    monkeypatch, *, model_device, host_has_cuda, prompt_ids=(11, 12, 13)
):
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = object
    fake_transformers.AutoModelForCausalLM = object
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: host_has_cuda)
    fake_torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    import pipmaster as pm

    monkeypatch.setattr(pm, "is_installed", lambda name: True)
    sys.modules.pop("lightrag.llm.hf", None)
    hf_module = importlib.import_module("lightrag.llm.hf")

    fake_model = FakeModel(model_device)
    fake_tokenizer = FakeTokenizer(prompt_ids, host_has_cuda)
    monkeypatch.setattr(
        hf_module,
        "initialize_hf_model",
        lambda model_name: (fake_model, fake_tokenizer),
    )
    return hf_module, fake_model, fake_tokenizer


@pytest.mark.parametrize(
    ("model_device", "host_has_cuda", "prompt_ids"),
    [
        pytest.param("cpu", False, (11, 12, 13), id="cpu-only-host"),
        pytest.param("cuda", True, (11, 12, 13), id="cuda-model"),
        pytest.param("cpu", True, (11, 12, 13), id="cpu-offload-on-cuda-host"),
        pytest.param("cpu", False, (1, 2, 3, 4, 5), id="prompt-length-slicing"),
    ],
)
async def test_inputs_follow_model_device_and_slice_prompt_tokens(
    monkeypatch, model_device, host_has_cuda, prompt_ids
):
    hf_module, fake_model, _ = _make_hf_module(
        monkeypatch,
        model_device=model_device,
        host_has_cuda=host_has_cuda,
        prompt_ids=prompt_ids,
    )

    result = await hf_module.hf_model_if_cache("fake-model", "hello world")

    assert fake_model.received_kwargs["input_ids"].device.type == model_device
    assert fake_model.received_kwargs["attention_mask"].device.type == model_device
    assert result == "decoded:[901, 902]"
    assert fake_model.received_kwargs["max_new_tokens"] == 512
    assert "max_tokens" not in fake_model.received_kwargs


@pytest.mark.parametrize(
    ("generation_kwargs", "expected_max_new_tokens"),
    [
        ({"max_tokens": 37}, 37),
        ({"max_tokens": 37, "max_new_tokens": 41}, 41),
    ],
)
async def test_generation_token_limit_precedence(
    monkeypatch, generation_kwargs, expected_max_new_tokens
):
    hf_module, fake_model, _ = _make_hf_module(
        monkeypatch, model_device="cpu", host_has_cuda=False
    )

    await hf_module.hf_model_if_cache("fake-model", "hello world", **generation_kwargs)

    assert fake_model.received_kwargs["max_new_tokens"] == expected_max_new_tokens
    assert "max_tokens" not in fake_model.received_kwargs


@pytest.mark.parametrize(
    ("generated_ids", "expect_truncated"),
    [
        pytest.param([901, 902], True, id="hit-limit-without-eos"),
        pytest.param([901, 0], False, id="eos-at-limit"),
    ],
)
async def test_truncation_marker_follows_eos_at_limit(
    monkeypatch, generated_ids, expect_truncated
):
    hf_module, fake_model, _ = _make_hf_module(
        monkeypatch, model_device="cpu", host_has_cuda=False
    )
    fake_model.generated_ids = generated_ids

    result = await hf_module.hf_model_if_cache(
        "fake-model", "hello world", max_new_tokens=2
    )

    assert isinstance(result, hf_module.TruncatedResponse) is expect_truncated
    if not expect_truncated:
        assert type(result) is str


@pytest.mark.parametrize("mode", ["native", "no_system", "missing"])
@pytest.mark.parametrize("system_prompt", [None, "Be concise"])
@pytest.mark.parametrize("with_history", [False, True])
async def test_chat_template_fallback_preserves_prompt_and_history(
    monkeypatch, mode, system_prompt, with_history
):
    hf, model, _ = _make_hf_module(monkeypatch, model_device="cpu", host_has_cuda=False)
    tokenizer = PromptTokenizer(mode)
    monkeypatch.setattr(hf, "initialize_hf_model", lambda _: (model, tokenizer))
    history = (
        [
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
        ]
        if with_history
        else []
    )
    original_history = deepcopy(history)

    result = await hf.hf_model_if_cache(
        "fake-model", "Current question", system_prompt, history
    )

    assert result == "decoded:[901, 902]"
    assert history == original_history
    if mode == "missing":
        expected = "<system>Be concise</system>\n" if system_prompt else ""
        if with_history:
            expected += (
                "<user>Earlier question</user>\n<assistant>Earlier answer</assistant>\n"
            )
        expected += "<user>Current question</user>\n"
    else:
        contents = [message["content"] for message in original_history]
        contents.append("Current question")
        if system_prompt:
            if mode == "native":
                contents.insert(0, "Be concise")
            else:
                contents[0] = "<system>Be concise</system>\n" + contents[0]
        expected = " | ".join(contents)
    assert tokenizer.input_prompt == expected
    assert len(tokenizer.rendered_messages) == (
        2 if system_prompt and mode != "native" else 1
    )
