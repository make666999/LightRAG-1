"""HuggingFace embedding tests: attention-masked float32 pooling, dtype cast-back, and max_length truncation."""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]

UNSET_SENTINEL = int(1e30)


class _FakeDType:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, _FakeDType) and other.name == self.name

    def __hash__(self):
        return hash(self.name)


FLOAT32 = _FakeDType("float32")
FLOAT16 = _FakeDType("float16")
BFLOAT16 = _FakeDType("bfloat16")
_GENUINE_NUMPY_DTYPE = {"float16": np.float16}


class FakeTensor:
    """Numpy-backed stand-in for torch.Tensor operations in hf_embed()."""

    def __init__(self, array, dtype=FLOAT32):
        backing = _GENUINE_NUMPY_DTYPE.get(dtype.name, np.float64)
        self.array = np.asarray(array, dtype=backing)
        self.dtype = dtype

    @property
    def shape(self):
        return self.array.shape

    def unsqueeze(self, dim):
        return FakeTensor(np.expand_dims(self.array, dim), self.dtype)

    def to(self, target):
        if isinstance(target, _FakeDType):
            backing = _GENUINE_NUMPY_DTYPE.get(target.name, np.float64)
            return FakeTensor(self.array.astype(backing), target)
        return self

    def sum(self, dim):
        return FakeTensor(self.array.sum(axis=dim), self.dtype)

    def clamp_min(self, value):
        return FakeTensor(np.clip(self.array, a_min=value, a_max=None), self.dtype)

    def mean(self, dim):
        return FakeTensor(self.array.mean(axis=dim), self.dtype)

    def __mul__(self, other):
        return FakeTensor(self.array * other.array, self.dtype)

    def __truediv__(self, other):
        return FakeTensor(self.array / other.array, self.dtype)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        backing = _GENUINE_NUMPY_DTYPE.get(self.dtype.name, np.float32)
        return self.array.astype(backing)


def zeros(*shape):
    return FakeTensor(np.zeros(shape))


def ones(*shape):
    return FakeTensor(np.ones(shape))


def randn(*shape, rng):
    return FakeTensor(rng.standard_normal(shape))


def full(shape, value):
    return FakeTensor(np.full(shape, value))


def tensor(data):
    return FakeTensor(np.array(data))


def cat(tensors, dim):
    return FakeTensor(np.concatenate([t.array for t in tensors], axis=dim))


@pytest.fixture
def hf_module(monkeypatch):
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = object
    fake_transformers.AutoModelForCausalLM = object
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    class _NullContext:
        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = FLOAT32
    fake_torch.float16 = FLOAT16
    fake_torch.bfloat16 = BFLOAT16
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False)
    )
    fake_torch.device = lambda name: name
    fake_torch.no_grad = _NullContext
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    import pipmaster as pm

    monkeypatch.setattr(pm, "is_installed", lambda name: True)
    sys.modules.pop("lightrag.llm.hf", None)
    return importlib.import_module("lightrag.llm.hf")


class _FakeTokenizerOutput(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    def __init__(self, encoded):
        self._encoded = encoded

    def __call__(
        self, texts, return_tensors="pt", padding=True, truncation=True, max_length=None
    ):
        return _FakeTokenizerOutput(self._encoded)


class _FakeModelOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _FakeEmbedModel:
    def __init__(self, hidden_states):
        self._hidden_states = hidden_states

    def to(self, device):
        return self

    def __call__(self, input_ids, attention_mask):
        return _FakeModelOutput(self._hidden_states)

    def parameters(self):
        yield zeros(1)


class _OversizedTokenizer:
    """Tokenizer double recording truncation/max_length kwargs."""

    def __init__(self, raw_token_count=1277, model_max_length=UNSET_SENTINEL):
        self.model_max_length = model_max_length
        self.raw_token_count = raw_token_count
        self.last_kwargs = None

    def __call__(
        self, texts, return_tensors="pt", padding=True, truncation=True, max_length=None
    ):
        self.last_kwargs = {"truncation": truncation, "max_length": max_length}
        seq_len = self.raw_token_count
        if truncation and max_length is not None:
            seq_len = min(seq_len, max_length)
        batch = len(texts)
        return _FakeTokenizerOutput(
            {"input_ids": zeros(batch, seq_len), "attention_mask": ones(batch, seq_len)}
        )


class _PositionLimitedEmbedModel:
    """BERT-like model double enforcing config.max_position_embeddings."""

    def __init__(self, max_position_embeddings, dim=1024):
        self.config = types.SimpleNamespace(
            max_position_embeddings=max_position_embeddings
        )
        self._dim = dim

    def to(self, device):
        return self

    def parameters(self):
        yield zeros(1)

    def __call__(self, input_ids, attention_mask):
        seq_len = input_ids.shape[1]
        if seq_len > self.config.max_position_embeddings:
            raise RuntimeError(
                f"The expanded size of the tensor ({seq_len}) must match the "
                f"existing size ({self.config.max_position_embeddings}) at "
                "non-singleton dimension 1."
            )
        batch = input_ids.shape[0]
        hidden = np.ones((batch, seq_len, self._dim), dtype=np.float64)
        return _FakeModelOutput(FakeTensor(hidden))


# --------------------------------------------------------------------------- #
# Masked pooling & float32 reduction
# --------------------------------------------------------------------------- #


async def test_same_text_gets_the_same_embedding_regardless_of_batch_padding(
    hf_module,
):
    dim = 1024
    rng = np.random.default_rng(0)
    real_tokens = randn(1, 3, dim, rng=rng)
    pad_tokens = full((1, 3, dim), 5.0)

    emb_alone = await hf_module.hf_embed(
        ["hello"],
        _FakeTokenizer({"input_ids": zeros(1, 3), "attention_mask": ones(1, 3)}),
        _FakeEmbedModel(real_tokens),
    )
    emb_batched = await hf_module.hf_embed(
        ["hello"],
        _FakeTokenizer(
            {
                "input_ids": zeros(1, 6),
                "attention_mask": tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]),
            }
        ),
        _FakeEmbedModel(cat([real_tokens, pad_tokens], dim=1)),
    )

    assert emb_alone.shape == emb_batched.shape
    assert np.allclose(emb_alone, emb_batched, atol=1e-8)


async def test_no_padding_matches_plain_mean_and_preserves_shape_and_dtype(hf_module):
    dim = 1024
    rng = np.random.default_rng(1)
    hidden = randn(3, 5, dim, rng=rng)
    plain_mean = hidden.mean(dim=1).numpy()

    result = await hf_module.hf_embed(
        ["a", "b", "c"],
        _FakeTokenizer({"input_ids": zeros(3, 5), "attention_mask": ones(3, 5)}),
        _FakeEmbedModel(hidden),
    )

    assert result.shape == (3, dim)
    assert result.dtype == np.float32
    assert np.allclose(result, plain_mean, atol=1e-8)


async def test_fully_masked_row_does_not_produce_nan_or_inf(hf_module):
    dim = 1024
    rng = np.random.default_rng(2)
    hidden = randn(1, 3, dim, rng=rng)

    result = await hf_module.hf_embed(
        [""],
        _FakeTokenizer({"input_ids": zeros(1, 3), "attention_mask": zeros(1, 3)}),
        _FakeEmbedModel(hidden),
    )

    assert bool(np.isfinite(result).all())


@pytest.mark.parametrize(
    ("input_dtype", "expected_numpy_dtype"),
    [
        pytest.param(BFLOAT16, np.float32, id="bfloat16-converts-to-float32"),
        pytest.param(FLOAT16, np.float16, id="float16-casts-back-to-float16"),
    ],
)
async def test_pooled_embedding_output_dtype(
    hf_module, input_dtype, expected_numpy_dtype
):
    dim = 1024
    hidden = FakeTensor(np.full((1, 3, dim), 2.0), dtype=input_dtype)

    result = await hf_module.hf_embed(
        ["x"],
        _FakeTokenizer({"input_ids": zeros(1, 3), "attention_mask": ones(1, 3)}),
        _FakeEmbedModel(hidden),
    )

    assert result.dtype == expected_numpy_dtype


async def test_fp16_hidden_states_upcast_to_float32_avoid_overflow(hf_module):
    seq, dim = 8192, 1024
    with np.errstate(over="ignore"):
        naive_fp16_sum = np.full(seq, 10.0, dtype=np.float16).sum()
    assert not np.isfinite(naive_fp16_sum)

    hidden = FakeTensor(np.full((1, seq, dim), 10.0, dtype=np.float16), dtype=FLOAT16)
    result = await hf_module.hf_embed(
        ["x"],
        _FakeTokenizer({"input_ids": zeros(1, seq), "attention_mask": ones(1, seq)}),
        _FakeEmbedModel(hidden),
    )

    assert np.isfinite(result).all()
    assert np.allclose(result, 10.0, atol=1e-2)


async def test_token_count_and_hidden_state_reduction_occur_in_float32(hf_module):
    seq, dim = 3001, 1024
    rng = np.random.default_rng(7)
    hidden_np = rng.uniform(0.5, 2.0, size=(1, seq, dim)).astype(np.float16)

    true_mean = hidden_np.astype(np.float64).mean(axis=1)
    naive_fp16_mean = hidden_np.mean(axis=1, dtype=np.float16)
    assert not np.allclose(naive_fp16_mean, true_mean, atol=2e-3)

    hidden = FakeTensor(hidden_np, dtype=FLOAT16)
    result = await hf_module.hf_embed(
        ["x"],
        _FakeTokenizer({"input_ids": zeros(1, seq), "attention_mask": ones(1, seq)}),
        _FakeEmbedModel(hidden),
    )

    assert np.allclose(result, true_mean, atol=2e-3)


# --------------------------------------------------------------------------- #
# Tokenizer max_length resolution & truncation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw_token_count", "model_max_length", "expected_max_length"),
    [
        pytest.param(1277, UNSET_SENTINEL, 512, id="oversized-unset-sentinel-falls-back"),
        pytest.param(900, 384, 384, id="bounded-tokenizer-limit-used-directly"),
        pytest.param(100, UNSET_SENTINEL, 512, id="short-input-keeps-model-cap"),
    ],
)
async def test_hf_embed_resolves_truncation_max_length(
    hf_module, raw_token_count, model_max_length, expected_max_length
):
    tokenizer = _OversizedTokenizer(
        raw_token_count=raw_token_count, model_max_length=model_max_length
    )
    embed_model = _PositionLimitedEmbedModel(max_position_embeddings=512)

    result = await hf_module.hf_embed(["x" * 5000], tokenizer, embed_model)

    assert result.shape == (1, 1024)
    assert tokenizer.last_kwargs == {
        "truncation": True,
        "max_length": expected_max_length,
    }
