import asyncio
from pathlib import Path
import random
import re

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.chunker import chunking_by_fixed_token, chunking_by_token_size
from lightrag.constants import MIN_CHUNK_TOKEN_SIZE
from lightrag.exceptions import ChunkTokenLimitExceededError
from lightrag.parser.routing import validate_chunk_token_sizes
from lightrag.rerank import chunk_documents_for_rerank
from lightrag.utils import (
    EmbeddingFunc,
    TiktokenTokenizer,
    Tokenizer,
    TokenizerInterface,
)

pytestmark = pytest.mark.offline


class DummyTokenizer(TokenizerInterface):
    """Simple 1:1 character-to-token mapping."""

    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


class MultiTokenCharacterTokenizer(TokenizerInterface):
    """
    Tokenizer where character-to-token ratio is non-uniform.
    This helps catch bugs where code incorrectly counts characters instead of tokens.

    Mapping:
    - Uppercase letters: 2 tokens each
    - Punctuation (!, ?, .): 3 tokens each
    - Other characters: 1 token each
    """

    def encode(self, content: str):
        tokens = []
        for ch in content:
            if ch.isupper():  # Uppercase = 2 tokens
                tokens.extend([ord(ch), ord(ch) + 1000])
            elif ch in ["!", "?", "."]:  # Punctuation = 3 tokens
                tokens.extend([ord(ch), ord(ch) + 2000, ord(ch) + 3000])
            else:  # Regular chars = 1 token
                tokens.append(ord(ch))
        return tokens

    def decode(self, tokens):
        # Simplified decode for testing
        result = []
        i = 0
        while i < len(tokens):
            base_token = tokens[i]
            # Check if this is part of a multi-token sequence
            if (
                i + 2 < len(tokens)
                and tokens[i + 1] == base_token + 2000
                and tokens[i + 2] == base_token + 3000
            ):
                # 3-token punctuation
                result.append(chr(base_token))
                i += 3
            elif i + 1 < len(tokens) and tokens[i + 1] == base_token + 1000:
                # 2-token uppercase
                result.append(chr(base_token))
                i += 2
            else:
                # Single token
                result.append(chr(base_token))
                i += 1
        return "".join(result)


def make_tokenizer() -> Tokenizer:
    return Tokenizer(model_name="dummy", tokenizer=DummyTokenizer())


def make_multi_token_tokenizer() -> Tokenizer:
    return Tokenizer(model_name="multi", tokenizer=MultiTokenCharacterTokenizer())


# ============================================================================
# Tests for split_by_character_only=True (raises error on oversized chunks)
# ============================================================================


@pytest.mark.offline
def test_chunk_error_includes_preview():
    """Test that error message includes chunk preview."""
    tokenizer = make_tokenizer()
    oversized = "x" * 100

    with pytest.raises(ChunkTokenLimitExceededError) as excinfo:
        chunking_by_token_size(
            tokenizer,
            oversized,
            split_by_character="\n\n",
            split_by_character_only=True,
            chunk_token_size=10,
        )

    err = excinfo.value
    # Preview should be first 80 chars of a 100-char string
    assert err.chunk_preview == "x" * 80
    assert "Preview:" in str(err)


@pytest.mark.offline
def test_split_by_character_only_at_exact_limit():
    """Test chunking when chunk is exactly at token limit."""
    tokenizer = make_tokenizer()
    exact_size = "a" * 10

    chunks = chunking_by_token_size(
        tokenizer,
        exact_size,
        split_by_character="\n\n",
        split_by_character_only=True,
        chunk_token_size=10,
    )

    assert len(chunks) == 1
    assert chunks[0]["content"] == exact_size
    assert chunks[0]["tokens"] == 10


@pytest.mark.offline
def test_split_by_character_only_one_over_limit():
    """Test that chunk with one token over limit raises error."""
    tokenizer = make_tokenizer()
    one_over = "a" * 11

    with pytest.raises(ChunkTokenLimitExceededError) as excinfo:
        chunking_by_token_size(
            tokenizer,
            one_over,
            split_by_character="\n\n",
            split_by_character_only=True,
            chunk_token_size=10,
        )

    err = excinfo.value
    assert err.chunk_tokens == 11
    assert err.chunk_token_limit == 10


# ============================================================================
# Tests for split_by_character_only=False (recursive splitting)
# ============================================================================


@pytest.mark.offline
def test_split_with_chunk_overlap():
    """
    Test chunk splitting with overlap using distinctive content.

    With distinctive characters, we can verify overlap positions are exact.
    Misaligned overlap would produce wrong content and fail the test.
    """
    tokenizer = make_tokenizer()
    # Each character is unique - enables exact position verification
    content = "0123456789abcdefghijklmno"  # 25 chars

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="\n\n",
        split_by_character_only=False,
        chunk_token_size=10,
        chunk_overlap_token_size=3,
    )

    # With overlap=3, step size = chunk_size - overlap = 10 - 3 = 7
    # Chunks start at positions: 0, 7, 14, 21
    assert len(chunks) == 4

    # Verify exact content and token counts
    assert chunks[0]["tokens"] == 10
    assert chunks[0]["content"] == "0123456789"  # [0:10]

    assert chunks[1]["tokens"] == 10
    assert chunks[1]["content"] == "789abcdefg"  # [7:17] - overlaps with "789"

    assert chunks[2]["tokens"] == 10
    assert chunks[2]["content"] == "efghijklmn"  # [14:24] - overlaps with "efg"

    assert chunks[3]["tokens"] == 4
    assert chunks[3]["content"] == "lmno"  # [21:25] - overlaps with "lmn"


@pytest.mark.offline
def test_split_multiple_chunks_with_mixed_sizes():
    """Test splitting text with multiple chunks of different sizes."""
    tokenizer = make_tokenizer()
    # "small\n\nlarge_chunk_here\n\nmedium"
    # small: 5 tokens, large_chunk_here: 16 tokens, medium: 6 tokens
    content = "small\n\n" + "a" * 16 + "\n\nmedium"

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="\n\n",
        split_by_character_only=False,
        chunk_token_size=10,
        chunk_overlap_token_size=2,
    )

    # First chunk "small" should be kept as is (5 tokens)
    # Second chunk (16 tokens) should be split into 2 chunks
    # Third chunk "medium" should be kept as is (6 tokens)
    assert len(chunks) == 4
    assert chunks[0]["content"] == "small"
    assert chunks[0]["tokens"] == 5


# ============================================================================
# Edge Cases
# ============================================================================


@pytest.mark.offline
def test_overlap_greater_than_chunk_size_raises():
    """Overlap larger than chunk size makes the stride negative, which yields an
    empty range and silently drops the entire document. Fail closed instead."""
    tokenizer = make_tokenizer()
    content = "abcdefghijklmnopqrstuvwxyz0123456789"

    with pytest.raises(ValueError) as excinfo:
        chunking_by_token_size(
            tokenizer,
            content,
            chunk_token_size=50,
            chunk_overlap_token_size=100,
        )

    message = str(excinfo.value)
    assert "chunk_overlap_token_size (100)" in message
    assert "chunk_token_size (50)" in message


@pytest.mark.offline
def test_overlap_equal_to_chunk_size_raises():
    """Overlap equal to chunk size makes the stride zero, which otherwise raises
    an opaque ``range() arg 3 must not be zero``. Raise a clear message instead."""
    tokenizer = make_tokenizer()
    content = "abcdefghijklmnopqrstuvwxyz0123456789"

    with pytest.raises(ValueError) as excinfo:
        chunking_by_token_size(
            tokenizer,
            content,
            chunk_token_size=10,
            chunk_overlap_token_size=10,
        )

    message = str(excinfo.value)
    assert "chunk_overlap_token_size (10)" in message
    assert "chunk_token_size (10)" in message


@pytest.mark.offline
def test_negative_overlap_raises():
    """Negative overlap makes the stride larger than chunk_token_size (e.g.
    step = 10 - (-1) = 11 for a 10-token chunk), silently skipping a token
    between every window instead of the intended overlap. Fail closed,
    matching the >= chunk_size case above."""
    tokenizer = make_tokenizer()
    content = "abcdefghijklmnopqrstuvwxyz0123456789"

    with pytest.raises(ValueError) as excinfo:
        chunking_by_token_size(
            tokenizer,
            content,
            chunk_token_size=10,
            chunk_overlap_token_size=-1,
        )

    message = str(excinfo.value)
    assert "chunk_overlap_token_size must be non-negative" in message
    assert "-1" in message


@pytest.mark.offline
@pytest.mark.parametrize("chunk_overlap_token_size", [0, 5])
def test_valid_overlap_still_covers_every_token(chunk_overlap_token_size):
    """Control: zero and a valid positive overlap must still produce
    complete, gap-free coverage of the source content -- guards the
    negative-overlap rejection above against accidentally tightening or
    loosening a legitimate case.

    DummyTokenizer maps each character to its own token (ord(ch)), so a
    string of distinct printable characters lets coverage be checked by
    set membership without collisions.
    """
    tokenizer = make_tokenizer()
    content = "".join(chr(33 + i) for i in range(90))

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        chunk_token_size=10,
        chunk_overlap_token_size=chunk_overlap_token_size,
    )

    covered = set()
    for chunk in chunks:
        covered.update(chunk["content"])

    assert covered == set(content)


@pytest.mark.offline
@pytest.mark.parametrize("split_by_character", [None, "|"])
def test_no_trailing_window_inside_previous_window(split_by_character):
    """A window that reaches the end of the text is the last one.

    17 tokens, size 10, overlap 3 (step 7): the windows [0:10] and [7:17]
    already cover everything. The next start (14) used to emit [14:17],
    which is the tail of [7:17] and adds no text. The "|" case sends the
    same text through the split_by_character path as one oversize segment.
    """
    tokenizer = make_tokenizer()
    content = "0123456789abcdefg"  # 17 chars

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character=split_by_character,
        chunk_token_size=10,
        chunk_overlap_token_size=3,
    )

    assert [chunk["content"] for chunk in chunks] == ["0123456789", "789abcdefg"]


@pytest.mark.offline
@pytest.mark.parametrize("split_by_character", [None, "|"])
def test_every_window_adds_new_text(split_by_character):
    """Each window must end past the previous one, the last must reach the end
    of the text, and the count must be the minimum that covers it.

    A value sweep (see *Failures name the case* in
    docs/design/BackendTestingGuide.md): ``split_by_character`` selects the
    code path and stays a node; length and overlap only change the window
    arithmetic, so every combination is checked against the formula below
    and every mismatch is reported at once.
    """
    tokenizer = make_tokenizer()
    chunk_token_size = 10
    mismatches = []
    for chunk_overlap_token_size in (0, 3, 6, 9):
        step = chunk_token_size - chunk_overlap_token_size
        for length in range(1, 40):
            content = "".join(chr(33 + i) for i in range(length))
            chunks = chunking_by_token_size(
                tokenizer,
                content,
                split_by_character=split_by_character,
                chunk_token_size=chunk_token_size,
                chunk_overlap_token_size=chunk_overlap_token_size,
                _emit_source_span=True,
            )
            ends = [chunk["_source_span"]["end"] for chunk in chunks]
            expected_count = 1 + max(0, -(-(length - chunk_token_size) // step))
            if (
                not ends
                or ends != sorted(set(ends))
                or ends[-1] != length
                or len(chunks) != expected_count
            ):
                mismatches.append(
                    (length, chunk_overlap_token_size, ends, expected_count)
                )

    assert mismatches == []


@pytest.mark.offline
@pytest.mark.parametrize("split_by_character_only", [False, True])
def test_negative_overlap_raises_with_split_by_character(split_by_character_only):
    """Negative overlap must be rejected even when every split_by_character
    segment stays under chunk_token_size -- that path never reaches
    _window_step()'s inline calls, so the check has to run up front."""
    tokenizer = make_tokenizer()
    content = "abc|def|ghi"

    with pytest.raises(ValueError) as excinfo:
        chunking_by_token_size(
            tokenizer,
            content,
            split_by_character="|",
            split_by_character_only=split_by_character_only,
            chunk_token_size=10,
            chunk_overlap_token_size=-1,
        )

    assert "chunk_overlap_token_size must be non-negative" in str(excinfo.value)


@pytest.mark.offline
def test_empty_content():
    """Test chunking with empty content."""
    tokenizer = make_tokenizer()

    chunks = chunking_by_token_size(
        tokenizer,
        "",
        split_by_character="\n\n",
        split_by_character_only=True,
        chunk_token_size=10,
    )

    assert len(chunks) == 1
    assert chunks[0]["content"] == ""
    assert chunks[0]["tokens"] == 0


# ============================================================================
# Parameter Combinations
# ============================================================================


@pytest.mark.offline
def test_multi_character_delimiter_splitting():
    """
    Verify that multi-character delimiters are correctly recognized and not partially matched.

    Tests various multi-character delimiter scenarios to ensure the entire delimiter
    sequence is used for splitting, not individual characters.
    """
    tokenizer = make_tokenizer()

    # Test 1: Multi-character delimiter that contains single chars also present elsewhere
    content = "data<SEP>more<SEP>final"
    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="<SEP>",
        split_by_character_only=True,
        chunk_token_size=50,
    )

    assert len(chunks) == 3
    assert chunks[0]["content"] == "data"
    assert chunks[1]["content"] == "more"
    assert chunks[2]["content"] == "final"
    # Verify full delimiter is not in chunks, not just parts
    for chunk in chunks:
        assert "<SEP>" not in chunk["content"]

    # Test 2: Delimiter appears in middle of content
    content = "first><second><third"
    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="><",  # Multi-char delimiter
        split_by_character_only=True,
        chunk_token_size=50,
    )

    # Should split at "><" delimiter
    assert len(chunks) == 3
    assert chunks[0]["content"] == "first"
    assert chunks[1]["content"] == "second"
    assert chunks[2]["content"] == "third"

    # Test 3: Three-character delimiter
    content = "section1[***]section2[***]section3"
    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="[***]",
        split_by_character_only=True,
        chunk_token_size=50,
    )

    assert len(chunks) == 3
    assert chunks[0]["content"] == "section1"
    assert chunks[1]["content"] == "section2"
    assert chunks[2]["content"] == "section3"

    # Test 4: Delimiter with special regex characters (should be treated literally)
    content = "partA...partB...partC"
    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="...",
        split_by_character_only=True,
        chunk_token_size=50,
    )

    assert len(chunks) == 3
    assert chunks[0]["content"] == "partA"
    assert chunks[1]["content"] == "partB"
    assert chunks[2]["content"] == "partC"


@pytest.mark.offline
def test_delimiter_partial_match_not_split():
    """
    Verify that partial matches of multi-character delimiters don't cause splits.

    Only the complete delimiter sequence should trigger a split.
    """
    tokenizer = make_tokenizer()

    # Content contains "||" delimiter but also contains single "|"
    content = "data|single||data|with|pipes||final"

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="||",  # Only split on double pipe
        split_by_character_only=True,
        chunk_token_size=50,
    )

    # Should split only at "||", not at single "|"
    assert len(chunks) == 3
    assert chunks[0]["content"] == "data|single"
    assert chunks[1]["content"] == "data|with|pipes"
    assert chunks[2]["content"] == "final"

    # Single "|" should remain in content, but not double "||"
    assert "|" in chunks[0]["content"]
    assert "|" in chunks[1]["content"]
    assert "||" not in chunks[0]["content"]
    assert "||" not in chunks[1]["content"]


@pytest.mark.offline
def test_no_delimiter_forces_token_based_split():
    """
    Verify that when split_by_character doesn't appear in content,
    chunking falls back to token-based splitting.
    """
    tokenizer = make_tokenizer()

    # Content without the specified delimiter
    content = "0123456789abcdefghijklmnop"  # 26 chars, no "\n\n"

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="\n\n",  # Delimiter not in content
        split_by_character_only=False,
        chunk_token_size=10,
        chunk_overlap_token_size=0,
    )

    # Should fall back to token-based splitting
    assert len(chunks) == 3
    assert chunks[0]["content"] == "0123456789"  # [0:10]
    assert chunks[1]["content"] == "abcdefghij"  # [10:20]
    assert chunks[2]["content"] == "klmnop"  # [20:26]

    # Verify it didn't somehow split at the delimiter that doesn't exist
    for chunk in chunks:
        assert "\n\n" not in chunk["content"]


# ============================================================================
# Chunk Order Index Tests
# ============================================================================


@pytest.mark.offline
def test_chunk_order_index_simple():
    """Test that chunk_order_index is correctly assigned."""
    tokenizer = make_tokenizer()
    content = "a\n\nb\n\nc"

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="\n\n",
        split_by_character_only=True,
        chunk_token_size=10,
    )

    assert len(chunks) == 3
    assert chunks[0]["chunk_order_index"] == 0
    assert chunks[1]["chunk_order_index"] == 1
    assert chunks[2]["chunk_order_index"] == 2


@pytest.mark.offline
def test_chunk_order_index_with_splitting():
    """Test chunk_order_index with recursive splitting."""
    tokenizer = make_tokenizer()
    content = "a" * 30

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character=None,
        split_by_character_only=False,
        chunk_token_size=10,
        chunk_overlap_token_size=0,
    )

    assert len(chunks) == 3
    assert chunks[0]["chunk_order_index"] == 0
    assert chunks[1]["chunk_order_index"] == 1
    assert chunks[2]["chunk_order_index"] == 2


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.offline
def test_whitespace_handling():
    """Test that whitespace is properly handled in chunk content."""
    tokenizer = make_tokenizer()
    content = "  alpha  \n\n  beta  "

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="\n\n",
        split_by_character_only=True,
        chunk_token_size=20,
    )

    # Content should be stripped
    assert chunks[0]["content"] == "alpha"
    assert chunks[1]["content"] == "beta"


@pytest.mark.offline
def test_consecutive_delimiters():
    """Test handling of consecutive delimiters."""
    tokenizer = make_tokenizer()
    content = "alpha\n\n\n\nbeta"

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="\n\n",
        split_by_character_only=True,
        chunk_token_size=20,
    )

    # Should split on delimiter and include empty chunks
    assert len(chunks) >= 2
    assert "alpha" in [c["content"] for c in chunks]
    assert "beta" in [c["content"] for c in chunks]


# ============================================================================
# Token vs Character Counting Tests (Multi-Token Characters)
# ============================================================================


@pytest.mark.offline
def test_token_counting_not_character_counting():
    """
    Verify chunking uses token count, not character count.

    With MultiTokenCharacterTokenizer:
    - "aXa" = 3 chars but 4 tokens (a=1, X=2, a=1)

    This test would PASS if code incorrectly used character count (3 <= 3)
    but correctly FAILS because token count (4 > 3).
    """
    tokenizer = make_multi_token_tokenizer()

    # "aXa" = 3 characters, 4 tokens
    content = "aXa"

    with pytest.raises(ChunkTokenLimitExceededError) as excinfo:
        chunking_by_token_size(
            tokenizer,
            content,
            split_by_character="\n\n",
            split_by_character_only=True,
            chunk_token_size=3,  # 3 token limit
        )

    err = excinfo.value
    assert err.chunk_tokens == 4  # Should be 4 tokens, not 3 characters
    assert err.chunk_token_limit == 3


@pytest.mark.offline
def test_recursive_split_with_multi_token_chars():
    """
    Test recursive splitting respects token boundaries, not character boundaries.

    "AAAAA" = 5 chars but 10 tokens (each A = 2 tokens)
    With chunk_size=6, should split at token positions, not character positions.
    """
    tokenizer = make_multi_token_tokenizer()

    # "AAAAA" = 5 characters, 10 tokens
    content = "AAAAA"

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="\n\n",
        split_by_character_only=False,
        chunk_token_size=6,
        chunk_overlap_token_size=0,
    )

    # Should split into: [0:6]=3 chars, [6:10]=2 chars
    # Not [0:3]=6 tokens, [3:5]=4 tokens (character-based would be wrong)
    assert len(chunks) == 2
    assert chunks[0]["tokens"] == 6
    assert chunks[1]["tokens"] == 4


@pytest.mark.offline
def test_mixed_multi_token_content():
    """Test chunking with mixed single and multi-token characters."""
    tokenizer = make_multi_token_tokenizer()

    # "hello\n\nWORLD!" = 12 chars
    # hello = 5 tokens, WORLD = 10 tokens (5 chars × 2), ! = 3 tokens
    # Total = 18 tokens
    content = "hello\n\nWORLD!"

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="\n\n",
        split_by_character_only=True,
        chunk_token_size=20,
    )

    assert len(chunks) == 2
    assert chunks[0]["content"] == "hello"
    assert chunks[0]["tokens"] == 5
    assert chunks[1]["content"] == "WORLD!"
    assert chunks[1]["tokens"] == 13  # 10 + 3


# ============================================================================
# Tests: the split_by_character path must not encode the whole document
# (regression guard for the deferred whole-document encode)
# ============================================================================


class CountingTokenizer(DummyTokenizer):
    """DummyTokenizer that records every string passed to encode()."""

    def __init__(self):
        self.encoded: list[str] = []

    def encode(self, content: str):
        self.encoded.append(content)
        return super().encode(content)


def _make_counting_tokenizer() -> tuple[CountingTokenizer, Tokenizer]:
    inner = CountingTokenizer()
    return inner, Tokenizer(model_name="counting", tokenizer=inner)


@pytest.mark.offline
def test_split_by_character_does_not_encode_whole_document():
    """The split_by_character path only encodes each segment, never the
    whole document (the fixed-window ``else`` branch is the only consumer
    of the whole-document token list)."""
    inner, tokenizer = _make_counting_tokenizer()
    content = "alpha\n\nbeta\n\ngamma"

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="\n\n",
        chunk_token_size=10,
    )

    assert [chunk["content"] for chunk in chunks] == ["alpha", "beta", "gamma"]
    # Exactly one encode per split segment; in particular no call with the
    # full document, whose token list the split path never reads.
    assert inner.encoded == ["alpha", "beta", "gamma"]


@pytest.mark.offline
def test_split_by_character_only_does_not_encode_whole_document():
    """Same guarantee for split_by_character_only=True."""
    inner, tokenizer = _make_counting_tokenizer()
    content = "one\ntwo\nthree"

    chunks = chunking_by_token_size(
        tokenizer,
        content,
        split_by_character="\n",
        split_by_character_only=True,
        chunk_token_size=10,
    )

    assert [chunk["content"] for chunk in chunks] == ["one", "two", "three"]
    assert inner.encoded == ["one", "two", "three"]


# --------------------------------------------------------------------------- #
# Fixed-token character windows (no split UTF-8 bytes or lossy decode rewrites)
# --------------------------------------------------------------------------- #


class _WindowMultiTokenTokenizer(TokenizerInterface):
    """Uppercase letters are 2 tokens, ``.?!`` 3, everything else 1."""

    def encode(self, content: str) -> list[int]:
        tokens: list[int] = []
        for ch in content:
            width = 2 if ch.isupper() else 3 if ch in ".?!" else 1
            tokens.extend([ord(ch)] * width)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        raise AssertionError("F must not decode token windows")


class _ByteTokenizer(TokenizerInterface):
    """One token per UTF-8 byte; a split character decodes to U+FFFD."""

    errors = "replace"

    def encode(self, content: str) -> list[int]:
        return list(content.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8", errors=self.errors)


class _DropInvalidTokenizer(_ByteTokenizer):
    """Byte tokens; the decoder silently drops incomplete characters."""

    errors = "ignore"


class _LossyMetaspaceTokenizer(TokenizerInterface):
    """Word pieces whose decoder drops a leading space and rewrites ``□``."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.pieces: list[str] = []

    def encode(self, content: str) -> list[int]:
        tokens = []
        for piece in re.findall(r" ?\S+|\s", content):
            if piece not in self.vocab:
                self.vocab[piece] = len(self.pieces)
                self.pieces.append(piece)
            tokens.append(self.vocab[piece])
        return tokens

    def decode(self, tokens: list[int]) -> str:
        decoded = "".join(self.pieces[token] for token in tokens)
        return decoded.removeprefix(" ").replace("□", "[UNK]")


def _window_tokenizer(kind: str) -> Tokenizer:
    if kind == "tiktoken":
        return TiktokenTokenizer()
    impl = {
        "char": DummyTokenizer,
        "multi-token": _WindowMultiTokenTokenizer,
        "bytes-replace": _ByteTokenizer,
        "bytes-drop": _DropInvalidTokenizer,
        "lossy-metaspace": _LossyMetaspaceTokenizer,
    }[kind]()
    return Tokenizer(kind, impl)


_WINDOW_ALPHABET = [
    "a", "a", "b", "A", ".", " ", " ", "\n", "x", "😀", "中", "é", "\ufffd",
    "the ", "👨‍👩", "中文", "🇨🇳", "→", "□",
]  # fmt: skip


def _random_window_text(rng: random.Random) -> str:
    parts = []
    for _ in range(rng.randint(1, 6)):
        if rng.random() < 0.4:
            parts.append(rng.choice(_WINDOW_ALPHABET) * rng.randint(1, 30))
        else:
            parts.append("".join(rng.choice(_WINDOW_ALPHABET) for _ in range(20)))
    return "".join(parts)


@pytest.mark.parametrize(
    "kind",
    [
        "char",
        "multi-token",
        "bytes-replace",
        "bytes-drop",
        "lossy-metaspace",
        "tiktoken",
    ],
)
@pytest.mark.parametrize("split_by_character", [None, "|"])
def test_every_window_is_a_character_slice_of_the_source(
    kind: str, split_by_character: str | None
) -> None:
    """Seeded audit against the source itself: every chunk is the exact text
    at its span, counts its own tokens, and together the chunks lose no
    non-whitespace character."""
    tok = _window_tokenizer(kind)
    rng = random.Random(4623)
    mismatches = []
    for case in range(60):
        segments = [
            _random_window_text(rng) for _ in range(2 if split_by_character else 1)
        ]
        text = "|".join(segments)
        size = rng.randint(4, 24)
        overlap = rng.randint(0, size - 1)
        chunks = chunking_by_fixed_token(
            tok,
            text,
            chunk_token_size=size,
            chunk_overlap_token_size=overlap,
            split_by_character=split_by_character,
            _emit_source_span=True,
        )
        covered = set()
        for k, chunk in enumerate(chunks):
            body, span = chunk["content"], chunk.get("_source_span")
            where = (case, k, text, size, overlap)
            if not body:
                if span is not None or chunk["tokens"] != 0:
                    mismatches.append(("empty", where, chunk))
                continue
            if span is None or text[span["start"] : span["end"]] != body:
                mismatches.append(("slice", where, chunk))
                continue
            if chunk["tokens"] != len(tok.encode(body)):
                mismatches.append(("tokens", where, chunk))
            if kind != "tiktoken" and chunk["tokens"] > size:
                mismatches.append(("size", where, chunk))
            covered.update(range(span["start"], span["end"]))
        lost = [
            i
            for i, ch in enumerate(text)
            if not ch.isspace() and ch != "|" and i not in covered
        ]
        if lost:
            mismatches.append(("lost", (case, text, size, overlap), lost[:5]))
    assert mismatches == []


# --------------------------------------------------------------------------- #
# Rerank overlap_tokens validation (infinite loop prevention)
# --------------------------------------------------------------------------- #


class TestOverlapValidation:
    """Test suite for overlap_tokens validation in chunk_documents_for_rerank."""

    @pytest.mark.parametrize(
        "overlap_tokens",
        [
            pytest.param(32, id="greater_than_max"),
            pytest.param(30, id="equal_to_max"),
            pytest.param(29, id="one_below_max"),
        ],
    )
    def test_overlap_boundary_does_not_hang(self, overlap_tokens):
        """overlap_tokens >= max_tokens is clamped to max_tokens - 1; < works normally."""
        documents = [" ".join([f"word{i}" for i in range(100)])]
        chunked_docs, doc_indices = chunk_documents_for_rerank(
            documents, max_tokens=30, overlap_tokens=overlap_tokens
        )
        assert len(chunked_docs) > 0
        assert all(idx == 0 for idx in doc_indices)

    def test_normal_operation_unaffected(self):
        """Test that normal cases continue to work correctly."""
        documents = [
            " ".join([f"word{i}" for i in range(100)]),
            "short doc",
        ]
        chunked_docs, doc_indices = chunk_documents_for_rerank(
            documents, max_tokens=50, overlap_tokens=10
        )
        assert len(chunked_docs) > 2
        assert "short doc" in chunked_docs
        assert doc_indices[-1] == 1


# --------------------------------------------------------------------------- #
# MIN_CHUNK_TOKEN_SIZE at core configuration ingress
# --------------------------------------------------------------------------- #

_SIZE_ENVS = (
    "CHUNK_SIZE",
    "CHUNK_F_SIZE",
    "CHUNK_R_SIZE",
    "CHUNK_V_SIZE",
    "CHUNK_P_SIZE",
)
_TOO_SMALL = MIN_CHUNK_TOKEN_SIZE - 1


async def _minsize_mock_llm(*args, **kwargs) -> str:
    return ""


async def _minsize_mock_embedding(texts: list[str]) -> np.ndarray:
    return np.zeros((len(texts), 8))


def _minsize_new_rag(tmp_path: Path, **kwargs) -> LightRAG:
    return LightRAG(
        working_dir=str(tmp_path),
        workspace=f"minsize-{tmp_path.name}",
        llm_model_func=_minsize_mock_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=4096, func=_minsize_mock_embedding
        ),
        tokenizer=Tokenizer("char-tokenizer", DummyTokenizer()),
        **kwargs,
    )


def _minsize_enqueue_and_read(tmp_path, mutate, **enqueue_kwargs):
    async def _run():
        rag = _minsize_new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            mutate(rag)
            with pytest.raises(ValueError, match="must be >= ") as exc:
                await rag.apipeline_enqueue_documents(
                    "Body text for the minimum chunk size test.",
                    ids=["doc-min"],
                    file_paths="min.txt",
                    **enqueue_kwargs,
                )
            return exc.value, await rag.full_docs.get_by_id("doc-min")
        finally:
            await rag.finalize_storages()

    return asyncio.run(_run())


class TestChunkSizeMinimum:
    """``MIN_CHUNK_TOKEN_SIZE`` enforced at constructor, setter, and enqueue."""

    @pytest.fixture(autouse=True)
    def _clean_size_env(self, monkeypatch):
        for name in _SIZE_ENVS:
            monkeypatch.delenv(name, raising=False)

    @pytest.mark.parametrize(
        "config,slot",
        [
            ({"chunk_token_size": _TOO_SMALL}, "chunker.chunk_token_size"),
            (
                {"fixed_token": {"chunk_token_size": _TOO_SMALL}},
                "chunker.fixed_token.chunk_token_size",
            ),
            # bool is an int subclass: True is 1, below the minimum.
            ({"chunk_token_size": True}, "chunker.chunk_token_size"),
        ],
    )
    def test_validator_rejects_a_size_below_the_minimum(self, config, slot):
        with pytest.raises(
            ValueError, match=rf"{slot} must be >= {MIN_CHUNK_TOKEN_SIZE}"
        ):
            validate_chunk_token_sizes(config)

    @pytest.mark.parametrize(
        "config",
        [
            None,
            {},
            {"chunk_token_size": None, "recursive_character": {}},
            {
                "chunk_token_size": MIN_CHUNK_TOKEN_SIZE,
                "fixed_token": {"chunk_token_size": MIN_CHUNK_TOKEN_SIZE},
                "recursive_character": {"chunk_token_size": MIN_CHUNK_TOKEN_SIZE},
                "semantic_vector": {"chunk_token_size": MIN_CHUNK_TOKEN_SIZE},
                "paragraph_semantic": {"chunk_token_size": MIN_CHUNK_TOKEN_SIZE},
            },
        ],
    )
    def test_validator_accepts_the_minimum_and_absent_slots(self, config):
        assert validate_chunk_token_sizes(config) is None

    @pytest.mark.parametrize(
        "env_name,slot",
        [
            ("CHUNK_SIZE", "chunker.chunk_token_size"),
            ("CHUNK_F_SIZE", "chunker.fixed_token.chunk_token_size"),
            ("CHUNK_R_SIZE", "chunker.recursive_character.chunk_token_size"),
            ("CHUNK_V_SIZE", "chunker.semantic_vector.chunk_token_size"),
            ("CHUNK_P_SIZE", "chunker.paragraph_semantic.chunk_token_size"),
        ],
    )
    def test_constructor_rejects_a_small_size_env(
        self, tmp_path, monkeypatch, env_name, slot
    ):
        monkeypatch.setenv(env_name, str(_TOO_SMALL))

        with pytest.raises(
            ValueError, match=rf"{slot} must be >= {MIN_CHUNK_TOKEN_SIZE}"
        ):
            _minsize_new_rag(tmp_path)

    def test_constructor_rejects_a_small_legacy_chunk_token_size(self, tmp_path):
        with pytest.raises(ValueError, match=r"chunker\.chunk_token_size must be >= "):
            _minsize_new_rag(
                tmp_path, chunk_token_size=_TOO_SMALL, chunk_overlap_token_size=0
            )

    def test_constructor_rejects_a_small_addon_params_size(self, tmp_path):
        with pytest.raises(
            ValueError,
            match=r"chunker\.recursive_character\.chunk_token_size must be",
        ):
            _minsize_new_rag(
                tmp_path,
                addon_params={
                    "chunker": {"recursive_character": {"chunk_token_size": 8}}
                },
            )

    def test_constructor_accepts_the_minimum(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHUNK_R_SIZE", str(MIN_CHUNK_TOKEN_SIZE))

        rag = _minsize_new_rag(
            tmp_path,
            chunk_token_size=MIN_CHUNK_TOKEN_SIZE,
            chunk_overlap_token_size=0,
        )

        chunker = rag.addon_params["chunker"]
        assert chunker["chunk_token_size"] == MIN_CHUNK_TOKEN_SIZE
        assert (
            chunker["recursive_character"]["chunk_token_size"] == MIN_CHUNK_TOKEN_SIZE
        )

    def test_setter_rejection_keeps_the_previous_configuration(self, tmp_path):
        rag = _minsize_new_rag(
            tmp_path, chunk_token_size=500, chunk_overlap_token_size=50
        )
        before_params = rag.addon_params
        before_chunker = dict(before_params["chunker"])

        with pytest.raises(ValueError, match=r"chunker\.chunk_token_size must be >= "):
            rag.addon_params = {"chunker": {"chunk_token_size": _TOO_SMALL}}

        assert rag.addon_params is before_params
        assert dict(rag.addon_params["chunker"]) == before_chunker
        assert (rag.chunk_token_size, rag.chunk_overlap_token_size) == (500, 50)

    def test_enqueue_rejects_small_caller_chunk_options_before_writing(self, tmp_path):
        error, row = _minsize_enqueue_and_read(
            tmp_path,
            lambda rag: None,
            chunk_options=[{"fixed_token": {"chunk_token_size": _TOO_SMALL}}],
        )

        assert "chunker.fixed_token.chunk_token_size" in str(error)
        assert row is None

    def test_enqueue_rejects_a_nested_addon_params_mutation(self, tmp_path):
        def mutate(rag):
            rag.addon_params["chunker"]["recursive_character"]["chunk_token_size"] = 8

        error, row = _minsize_enqueue_and_read(tmp_path, mutate, process_options="R")

        assert "chunker.recursive_character.chunk_token_size" in str(error)
        assert row is None
