"""Offline tests for the Tokenizer thread-safety, deepcopy, and split/truncate contracts.

Covers two interrelated contracts on ``Tokenizer`` / ``TiktokenTokenizer``:
1. Thread-safety and deepcopy (GHSA-r8jh / GHSA-26pm): built-in ``TiktokenTokenizer``
   tolerates concurrent calls across threads (all instances share one encoding /
   core BPE), LightRAG never serializes callers with an internal lock, and
   ``_build_global_config`` preserves the injected tokenizer identity while
   requiring lock-holding custom implementations to declare ``__deepcopy__``.
2. Safe split/truncate contract (replaces #3559): ``truncate_by_token_limit``,
   ``split_by_token_limit``, and ``truncate_list_by_token_size`` re-encode every
   candidate span, maintain coverage/no-gaps/real-progress invariants across
   Unicode/special tokens/non-monotonic BPE/heterogeneous token density, clamp
   huge overlaps up front, and keep probe counts O(log length).
"""

from __future__ import annotations

import copy
import dataclasses
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from lightrag import LightRAG

# Accessed via the module (not ``from``-imports) on purpose: another test in the
# suite reloads lightrag.utils in place, which would leave from-imported class
# references pointing at the pre-reload class.
from lightrag import utils as lr_utils
from lightrag.utils import (
    EmbeddingFunc,
    TokenBudgetError,
    Tokenizer,
    TokenizerInterface,
    TokenSpan,
    truncate_list_by_token_size,
)

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Test tokenizer doubles
# ---------------------------------------------------------------------------


class _PlainTokenizer:
    """Minimal conforming implementation: immutable, therefore thread-safe."""

    def encode(self, content: str) -> list[int]:
        return [len(content)]

    def decode(self, tokens: list[int]) -> str:
        return "x" * sum(tokens)


class _OverlapDetectingTokenizer:
    """Records whether ``encode`` is ever entered by two threads at once."""

    def __init__(self, hold: float = 0.0):
        self._hold = hold
        self._inside = 0
        self._guard = threading.Lock()
        self.max_concurrency = 0

    def encode(self, content: str) -> list[int]:
        with self._guard:
            self._inside += 1
            self.max_concurrency = max(self.max_concurrency, self._inside)
        try:
            if self._hold:
                threading.Event().wait(self._hold)
            return [len(content)]
        finally:
            with self._guard:
                self._inside -= 1

    def decode(self, tokens: list[int]) -> str:
        return "x" * sum(tokens)


class _LockHoldingTokenizer:
    """Achieves thread safety with an internal lock, optionally defining ``__deepcopy__``."""

    def __init__(self, deepcopyable: bool = True):
        self._lock = threading.Lock()
        if deepcopyable:
            self.__deepcopy__ = lambda _memo: self

    def encode(self, content: str) -> list[int]:
        with self._lock:
            return [len(content)]

    def decode(self, tokens: list[int]) -> str:
        with self._lock:
            return "x" * sum(tokens)


class _CharTokenizer(TokenizerInterface):
    """One token per character; deterministic and cheap for invariant checks."""

    def encode(self, content: str) -> list[int]:
        return [ord(ch) % 1000 for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


class _HeavyTokenizer(TokenizerInterface):
    """Five tokens per character, used to trigger TokenBudgetError."""

    def encode(self, content: str) -> list[int]:
        return [0] * (5 * len(content))

    def decode(self, tokens: list[int]) -> str:
        return ""


class _WordCountingTokenizer(TokenizerInterface):
    """BPE-like tokenizer that recognizes each ``ab`` pair as one token."""

    def encode(self, content: str) -> list[int]:
        tokens: list[int] = []
        i = 0
        while i < len(content):
            if content[i : i + 2] == "ab":
                tokens.append(-1)
                i += 2
            else:
                tokens.append(ord(content[i]))
                i += 1
        return tokens

    def decode(self, tokens: list[int]) -> str:
        return "".join("ab" if t == -1 else chr(t) for t in tokens)


class _NonMonotonicPrefixTokenizer(TokenizerInterface):
    """Tokenizer where ``abcd`` prefix counts are 1, 2, 1, 4 (``abc`` == 1 token)."""

    def encode(self, content: str) -> list[int]:
        if content == "abc":
            return [-1]
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join("abc" if token == -1 else chr(token) for token in tokens)


class _DensityVaryingTokenizer(TokenizerInterface):
    """Token density differs 16:1 between ``D`` (4 tok/char) and other chars (1 tok/4 chars)."""

    def encode(self, content: str) -> list[int]:
        tokens: list[int] = []
        index = 0
        while index < len(content):
            if content[index] == "D":
                tokens.extend([1, 1, 1, 1])
                index += 1
            else:
                run = 0
                while index < len(content) and content[index] != "D":
                    run += 1
                    index += 1
                tokens.extend([2] * ((run + 3) // 4))
        return tokens

    def decode(self, tokens: list[int]) -> str:  # pragma: no cover
        raise AssertionError("the generic contract must never call decode()")


class _DensityCliffTokenizer(TokenizerInterface):
    """Steep 10,000:1 density cliff (``D`` = 10 tok/char, other = 1 tok/1000 chars)."""

    def encode(self, content: str) -> list[int]:
        tokens: list[int] = []
        index = 0
        while index < len(content):
            if content[index] == "D":
                tokens.extend([1] * 10)
                index += 1
            else:
                run = 0
                while index < len(content) and content[index] != "D":
                    run += 1
                    index += 1
                tokens.extend([2] * ((run + 999) // 1000))
        return tokens

    def decode(self, tokens: list[int]) -> str:  # pragma: no cover
        raise AssertionError("the generic contract must never call decode()")


class _CountingTokenizer(TokenizerInterface):
    """Wraps another tokenizer (or defaults to 1 token/char) and counts ``encode`` calls."""

    def __init__(self, inner: TokenizerInterface | None = None):
        self.inner = inner or _CharTokenizer()
        self.encode_calls = 0

    def encode(self, content: str) -> list[int]:
        self.encode_calls += 1
        return self.inner.encode(content)

    def decode(self, tokens: list[int]) -> str:
        return self.inner.decode(tokens)


def _char_tok() -> Tokenizer:
    return Tokenizer("char", _CharTokenizer())


def _tiktoken_tok():
    return lr_utils.TiktokenTokenizer("gpt-4o-mini")


def _density_tokenizer() -> Tokenizer:
    return Tokenizer(model_name="density", tokenizer=_DensityVaryingTokenizer())


async def _mock_embedding(texts: list[str]) -> np.ndarray:
    return np.zeros((len(texts), 16))


async def _mock_llm(*_args, **_kwargs) -> str:
    return "mock"


def _make_rag(tmp_path, tokenizer):
    return LightRAG(
        working_dir=str(tmp_path / "tokenizer-contract"),
        workspace="tokenizer-contract",
        llm_model_func=_mock_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=16, max_token_size=4096, func=_mock_embedding
        ),
        tokenizer=tokenizer,
    )


def _assert_split_invariants(tok, text: str, spans: list[TokenSpan], max_tokens: int):
    assert spans, "split of non-empty content must not be empty"
    assert spans[0].start == 0
    covered_end = 0
    for i, span in enumerate(spans):
        assert span.start < span.end, "every span must be non-empty"
        assert span.start <= covered_end, "no coverage gap"
        piece = text[span.start : span.end]
        assert len(tok.encode(piece)) == span.token_count
        assert span.token_count <= max_tokens
        if i < len(spans) - 1:
            assert span.end > covered_end, "non-final span must make real progress"
        covered_end = span.end
    assert covered_end == len(text), "split must fully cover the content"


# ---------------------------------------------------------------------------
# 1. Thread-safety, shared encoding, and LightRAG deepcopy contract
# ---------------------------------------------------------------------------


def test_builtin_tokenizer_survives_concurrent_encode_and_decode():
    """Guards the assumption that tiktoken tolerates concurrent calls across threads."""
    tokenizer = lr_utils.TiktokenTokenizer()
    texts = [f"hello {i} 中文测试 " * 200 for i in range(8)]
    expected = [tokenizer.encode(text) for text in texts]

    def round_trip(text: str) -> tuple[list[int], str]:
        tokens = tokenizer.encode(text)
        return tokens, tokenizer.decode(tokens)

    def work(index: int) -> list[tuple[list[int], str]]:
        return [round_trip(texts[index]) for _ in range(10)]

    with ThreadPoolExecutor(max_workers=len(texts)) as pool:
        results = list(pool.map(work, range(len(texts))))

    mismatches = [
        (index, kind)
        for index, rounds in enumerate(results)
        for tokens, decoded in rounds
        for kind, ok in (
            ("encode", tokens == expected[index]),
            ("decode", decoded == texts[index]),
        )
        if not ok
    ]
    assert mismatches == []


def test_builtin_tokenizers_share_encoding_and_core_bpe_across_deepcopy():
    """``tiktoken`` caches encodings process-wide, and deepcopy still shares ``_core_bpe``."""
    first = lr_utils.TiktokenTokenizer()
    second = lr_utils.TiktokenTokenizer()
    assert first.tokenizer is second.tokenizer

    clone = copy.deepcopy(first)
    assert clone.tokenizer is not first.tokenizer
    assert vars(clone.tokenizer)["_core_bpe"] is vars(first.tokenizer)["_core_bpe"]
    assert clone.encode("hello world") == first.encode("hello world")


def test_concurrent_callers_are_not_serialized_by_the_wrapper():
    """Two threads must genuinely overlap inside one Tokenizer wrapper (no wrapper lock)."""
    underlying = _OverlapDetectingTokenizer(hold=0.15)
    tokenizer = Tokenizer("test-model", underlying)
    barrier = threading.Barrier(2)

    def work() -> None:
        barrier.wait(timeout=5)
        tokenizer.encode("abcdef")

    threads = [threading.Thread(target=work) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert underlying.max_concurrency == 2


def test_lightrag_tokenizer_identity_and_wrapper_deepcopy(tmp_path):
    """``_build_global_config`` hands out the original tokenizer, and the wrapper is deepcopyable."""

    @dataclasses.dataclass
    class Holder:
        tokenizer: object

    tokenizer = Tokenizer("test-model", _PlainTokenizer())
    rag = _make_rag(tmp_path, tokenizer)

    assert rag._build_global_config()["tokenizer"] is tokenizer
    assert copy.deepcopy(tokenizer).encode("abc") == [3]
    assert dataclasses.asdict(Holder(tokenizer=tokenizer))["tokenizer"].encode(
        "abcd"
    ) == [4]


def test_lock_based_tokenizer_deepcopy_boundary(tmp_path):
    """A lock-holding tokenizer requires ``__deepcopy__`` to pass ``LightRAG`` construction."""
    injectable = Tokenizer("test-model", _LockHoldingTokenizer(deepcopyable=True))
    rag = _make_rag(tmp_path, injectable)
    global_config = rag._build_global_config()
    assert global_config["tokenizer"] is injectable
    assert global_config["tokenizer"].encode("abcd") == [4]

    non_deepcopyable = Tokenizer(
        "test-model", _LockHoldingTokenizer(deepcopyable=False)
    )
    with pytest.raises(TypeError, match="cannot pickle"):
        _make_rag(tmp_path, non_deepcopyable)


# ---------------------------------------------------------------------------
# 2. truncate_by_token_limit and argument validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tok_factory", [_char_tok, _tiktoken_tok], ids=["char", "tiktoken"]
)
def test_truncate_empty_whole_and_safe_prefix(tok_factory):
    tok = tok_factory()
    assert tok.truncate_by_token_limit("", 10) == TokenSpan(0, 0, 0)

    short = "hello world"
    whole = tok.truncate_by_token_limit(short, 10_000)
    assert whole == TokenSpan(0, len(short), len(tok.encode(short)))

    long_text = "abcdefghijklmnopqrstuvwxyz" * 5
    span = tok.truncate_by_token_limit(long_text, 7)
    assert span.start == 0
    assert len(tok.encode(long_text[span.start : span.end])) == span.token_count
    assert span.token_count <= 7


@pytest.mark.parametrize(
    "tok_factory", [_char_tok, _tiktoken_tok], ids=["char", "tiktoken"]
)
def test_truncate_and_split_reject_invalid_arguments(tok_factory):
    tok = tok_factory()
    for bad_budget in (0, -1):
        with pytest.raises(ValueError):
            tok.truncate_by_token_limit("hello", bad_budget)
    with pytest.raises(ValueError):
        tok.split_by_token_limit("hello", 0)
    with pytest.raises(ValueError):
        tok.split_by_token_limit("hello", 5, overlap_tokens=-1)


@pytest.mark.parametrize(
    ("tok_factory", "text", "budget", "expected_cp_tokens"),
    [
        pytest.param(
            lambda: Tokenizer("heavy", _HeavyTokenizer()),
            "hello",
            1,
            5,
            id="heavy-tokenizer",
        ),
        pytest.param(
            _density_tokenizer,
            "D" + "s" * 100,
            3,
            4,
            id="density-dense-head",
        ),
    ],
)
def test_truncate_raises_token_budget_error_below_one_code_point(
    tok_factory, text, budget, expected_cp_tokens
):
    tok = tok_factory()
    with pytest.raises(TokenBudgetError) as exc_info:
        tok.truncate_by_token_limit(text, budget)
    assert exc_info.value.max_tokens == budget
    assert exc_info.value.code_point_token_count == expected_cp_tokens


def test_truncate_and_split_survive_non_monotonic_bpe_tokenizer():
    """Bracketing guarantees every returned span fits even when longer prefixes are cheaper."""
    tok = Tokenizer("non-monotonic", _NonMonotonicPrefixTokenizer())

    # Prefix token counts for "abcd" are [1, 2, 1, 4]: budget 1 safely returns "a"
    # even though "abc" also fits.
    content = "abcd"
    assert [len(tok.encode(content[:end])) for end in range(1, 5)] == [1, 2, 1, 4]
    span = tok.truncate_by_token_limit(content, 1)
    assert span == TokenSpan(0, 1, 1)
    assert len(tok.encode(content[: span.end])) == span.token_count <= 1
    assert len(tok.encode(content[:3])) <= 1

    text = "abc" + "x" * 20
    for budget in (1, 2, 3, 5, 10):
        s = tok.truncate_by_token_limit(text, budget)
        assert len(tok.encode(text[s.start : s.end])) == s.token_count <= budget

    repeated = ("abc" + "x" * 8) * 6
    spans = tok.split_by_token_limit(repeated, 4)
    _assert_split_invariants(tok, repeated, spans, 4)


# ---------------------------------------------------------------------------
# 3. split_by_token_limit: coverage, overlap, clamping, bounded search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tok_factory", "text", "max_tokens"),
    [
        pytest.param(
            _char_tok,
            "The quick brown fox jumps over the lazy dog. " * 30,
            12,
            id="ascii-char",
        ),
        pytest.param(
            _tiktoken_tok,
            "The quick brown fox jumps over the lazy dog. " * 30,
            12,
            id="ascii-tiktoken",
        ),
        pytest.param(
            _char_tok,
            "这是一个测试句子,用来验证安全切分的不变式。" * 20,
            9,
            id="cjk-char",
        ),
        pytest.param(
            _tiktoken_tok,
            "这是一个测试句子,用来验证安全切分的不变式。" * 20,
            9,
            id="cjk-tiktoken",
        ),
        pytest.param(
            _tiktoken_tok,
            ("e\u0301 combining " + "👨‍👩‍👧‍👦 zwj family ") * 10,
            6,
            id="combining-and-zwj-emoji",
        ),
        pytest.param(
            _tiktoken_tok,
            "🇯🇵🇰🇷🇨🇳🇺🇸🇬🇧 flags " * 15,
            5,
            id="regional-indicators",
        ),
        pytest.param(
            _tiktoken_tok,
            ("plain text <|endoftext|> more plain text " * 10) + "<|endoftext|>",
            8,
            id="special-token-literal",
        ),
    ],
)
def test_split_covers_corpus_with_no_gaps_and_real_progress(
    tok_factory, text, max_tokens
):
    tok = tok_factory()
    spans = tok.split_by_token_limit(text, max_tokens)
    _assert_split_invariants(tok, text, spans, max_tokens)


def test_split_empty_and_zero_overlap_fixed_windows():
    tok = _char_tok()
    assert tok.split_by_token_limit("", 5) == []

    text = "abcdefghij"
    spans = tok.split_by_token_limit(text, 4, overlap_tokens=0)
    assert [text[s.start : s.end] for s in spans] == ["abcd", "efgh", "ij"]


@pytest.mark.parametrize(
    "tok_factory", [_char_tok, _tiktoken_tok], ids=["char", "tiktoken"]
)
def test_split_produces_real_overlap_and_strict_forward_progress(tok_factory):
    tok = tok_factory()
    text = "The quick brown fox jumps over the lazy dog. " * 6
    spans = tok.split_by_token_limit(text, 6, overlap_tokens=2)
    _assert_split_invariants(tok, text, spans, 6)
    assert len(spans) > 1
    assert any(spans[i].start < spans[i - 1].end for i in range(1, len(spans)))

    # Counter-example guard: high overlap must never yield a non-extending window.
    rep = "x" * 500
    high_overlap_spans = tok.split_by_token_limit(rep, 20, overlap_tokens=15)
    for i in range(1, len(high_overlap_spans) - 1):
        assert high_overlap_spans[i].end > high_overlap_spans[i - 1].end


def test_split_clamps_huge_overlap_target_before_retreating():
    """Requested overlap >> max_tokens is clamped to previous window size - 1 up front."""
    underlying = _CountingTokenizer()
    tok = Tokenizer("counting", underlying)
    text = "y" * 2000

    spans = tok.split_by_token_limit(text, max_tokens=6, overlap_tokens=10**9)
    assert len(spans) > 1
    _assert_split_invariants(tok, text, spans, 6)
    assert underlying.encode_calls < len(spans) * 15

    tt = _tiktoken_tok()
    prose = "The quick brown fox jumps over the lazy dog. " * 20
    tt_spans = tt.split_by_token_limit(prose, max_tokens=10, overlap_tokens=5000)
    _assert_split_invariants(tt, prose, tt_spans, 10)


def test_generic_and_tiktoken_bounded_truncation():
    """Generic bracket search is O(log length), and tiktoken exponential retreat handles large budgets."""
    underlying = _CountingTokenizer()
    tok = Tokenizer("counting", underlying)
    span = tok.truncate_by_token_limit("y" * 5000, 37)
    assert span.token_count <= 37
    assert underlying.encode_calls < 40

    tt = _tiktoken_tok()
    prose = "The quick brown fox jumps over the lazy dog. " * 200
    assert len(tt.encode(prose)) > 2000
    tt_span = tt.truncate_by_token_limit(prose, 500)
    assert tt_span.token_count <= 500
    assert len(tt.encode(prose[tt_span.start : tt_span.end])) == tt_span.token_count


def test_third_party_encode_decode_only_tokenizer_needs_no_new_methods():
    """Encode/decode-only TokenizerInterface implementations inherit safe split/truncate."""
    tok = Tokenizer("legacy-custom", _CharTokenizer())
    assert not hasattr(_CharTokenizer, "split_by_token_limit")
    assert not hasattr(_CharTokenizer, "truncate_by_token_limit")

    assert tok.truncate_by_token_limit("hello world", 3).token_count <= 3
    text = "hello world " * 5
    _assert_split_invariants(tok, text, tok.split_by_token_limit(text, 4), 4)


def test_tokenizer_split_state_does_not_leak_across_calls_or_deepcopy():
    """Calls have no hidden cross-call mutable state and deepcopy produces identical spans."""
    tok = _tiktoken_tok()
    text_a = "a" * 3000
    text_b = "b" * 3000

    spans_a1 = tok.split_by_token_limit(text_a, 10)
    tok.split_by_token_limit(text_b, 10)
    assert tok.split_by_token_limit(text_a, 10) == spans_a1
    assert copy.deepcopy(tok).split_by_token_limit(text_a, 10) == spans_a1


# ---------------------------------------------------------------------------
# 4. truncate_list_by_token_size (#3559)
# ---------------------------------------------------------------------------


def test_truncate_list_by_token_size_contract():
    """List truncation counts separators, keeps only whole items, re-verifies
    after item-boundary mapping, and returns [] on empty input or <=0 budget.
    """
    tok = _char_tok()
    # Separator tokens count: "aaa" (3) + "||" (2) + "bbb" (3) = 8 > 7 -> only ["aaa"].
    assert truncate_list_by_token_size(
        ["aaa", "bbb", "ccc"],
        key=lambda x: x,
        separator="||",
        max_token_size=7,
        tokenizer=tok,
    ) == ["aaa"]
    # Never splits a partial item.
    items = ["aaaa", "bbbb", "cccc", "dddd"]
    whole_only = truncate_list_by_token_size(
        items, key=lambda x: x, separator="\n", max_token_size=6, tokenizer=tok
    )
    assert whole_only == ["aaaa"]
    assert all(item in items for item in whole_only)

    # Empty list and non-positive budget return [].
    assert (
        truncate_list_by_token_size(
            [], key=lambda x: x, separator="\n", max_token_size=10, tokenizer=tok
        )
        == []
    )
    assert (
        truncate_list_by_token_size(
            ["a"], key=lambda x: x, separator="\n", max_token_size=0, tokenizer=tok
        )
        == []
    )

    # Re-verification after mapping to item boundaries on BPE-like tokenizer.
    word_tok = Tokenizer("word", _WordCountingTokenizer())
    reverified = truncate_list_by_token_size(
        ["abxxx", "abxxx", "abxxx"],
        key=lambda x: x,
        separator="",
        max_token_size=4,
        tokenizer=word_tok,
    )
    assert reverified == ["abxxx"]
    assert len(word_tok.encode("".join(reverified))) <= 4


# ---------------------------------------------------------------------------
# 5. Heterogeneous token density & logarithmic probe bound
# ---------------------------------------------------------------------------


def test_density_heterogeneous_truncation_and_split():
    """Dense/sparse heads neither collapse to 1 char nor drop budget, and
    the generic path works end-to-end on real tiktoken encodings.
    """
    tokenizer = _density_tokenizer()

    # Dense head, sparse tail: 500 tokens buys 125 dense chars, never 1 char.
    dense_head = "D" * 400 + "s" * 40000
    span = tokenizer.truncate_by_token_limit(dense_head, 500)
    assert span.start == 0
    assert span.end >= 100
    assert span.token_count <= 500
    assert len(tokenizer.encode(dense_head[: span.end])) == span.token_count

    # Sparse head, dense tail: spends full 500-token budget.
    sparse_head = "s" * 40000 + "D" * 400
    sparse_span = tokenizer.truncate_by_token_limit(sparse_head, 500)
    assert sparse_span.token_count == 500
    assert len(tokenizer.encode(sparse_head[: sparse_span.end])) == 500

    # Split over alternating dense/sparse blocks emits no 1-char spans and stays compact.
    alternating = ("D" * 200 + "s" * 8000) * 3
    spans = tokenizer.split_by_token_limit(alternating, 256)
    _assert_split_invariants(tokenizer, alternating, spans, 256)
    assert min(s.end - s.start for s in spans) > 1
    assert len(spans) < 100

    # End-to-end on real tiktoken through the generic base-class path.
    tiktoken = pytest.importorskip("tiktoken")
    generic_tt = Tokenizer(
        model_name="gpt-4o-mini",
        tokenizer=tiktoken.encoding_for_model("gpt-4o-mini"),
    )
    cjk_dense = (
        "\u8bed\u8a00\u6a21\u578b\u7684\u77e5\u8bc6\u56fe\u8c31" * 200
        + "the quick brown fox jumps over the lazy dog " * 400
    )
    tt_span = generic_tt.truncate_by_token_limit(cjk_dense, 512)
    assert tt_span.start == 0
    assert tt_span.end > 1
    assert 500 <= tt_span.token_count <= 512
    assert len(generic_tt.encode(cjk_dense[: tt_span.end])) == tt_span.token_count


@pytest.mark.parametrize(
    "tokenizer_factory",
    [_DensityVaryingTokenizer, _DensityCliffTokenizer],
    ids=["ratio_16", "ratio_10000"],
)
@pytest.mark.parametrize(
    "content,budget",
    [
        ("s" * 60000 + "D" * 4000, 4096),
        ("D" * 4000 + "s" * 60000, 4096),
        (("D" * 200 + "s" * 8000) * 8, 2048),
        ("s" * 100000 + "D" * 5000, 8192),
    ],
    # Explicit ids avoid embedding 105k chars into PYTEST_CURRENT_TEST (Windows 32k cap).
    ids=["sparse_head", "dense_head", "alternating_blocks", "long_sparse"],
)
def test_probe_count_stays_logarithmic_on_heterogeneous_input(
    content, budget, tokenizer_factory
):
    """Bisection closes the bracket in O(log length) probes even across a 10,000:1 cliff."""
    inner = _CountingTokenizer(tokenizer_factory())
    tokenizer = Tokenizer(model_name="density", tokenizer=inner)

    inner.encode_calls = 0
    span = tokenizer.truncate_by_token_limit(content, budget)

    assert span.token_count <= budget
    assert len(tokenizer.encode(content[: span.end])) == span.token_count
    assert inner.encode_calls <= 40
