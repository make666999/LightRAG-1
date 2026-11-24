"""Unit tests for ``chunking_by_recursive_character`` (process_options=R)."""

import json

import pytest

pytest.importorskip("langchain_text_splitters")
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

from lightrag.chunker import chunking_by_recursive_character  # noqa: E402
from lightrag.chunker.recursive_character import (  # noqa: E402
    _split_text_with_spans,
    normalize_r_separators,
)
from lightrag.constants import (  # noqa: E402
    DEFAULT_R_SEPARATORS,
    MAX_R_SEPARATOR_CHARS,
    MAX_R_SEPARATORS,
)
from lightrag.utils import Tokenizer, TokenizerInterface  # noqa: E402

pytestmark = pytest.mark.offline


class _CharTokenizer(TokenizerInterface):
    """1 char ≈ 1 token; lets assertions reason in terms of input length."""

    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


def _tok() -> Tokenizer:
    return Tokenizer("char-tokenizer", _CharTokenizer())


@pytest.mark.offline
def test_empty_input_returns_empty_list():
    chunks = chunking_by_recursive_character(_tok(), "")
    assert chunks == []


@pytest.mark.offline
def test_short_input_single_chunk():
    body = "Para A.\n\nPara B."
    chunks = chunking_by_recursive_character(_tok(), body, chunk_token_size=1000)

    assert len(chunks) == 1
    assert chunks[0]["content"] == body
    assert chunks[0]["_source_span"] == {"start": 0, "end": len(body)}
    assert chunks[0]["chunk_order_index"] == 0
    assert chunks[0]["tokens"] == len(body)


@pytest.mark.offline
def test_paragraph_separator_used_first():
    """``\\n\\n`` is the first separator in the default cascade — three
    paragraphs that each fit under the cap should split exactly there."""
    body = "Alpha section.\n\nBeta section.\n\nGamma section."
    chunks = chunking_by_recursive_character(
        _tok(),
        body,
        chunk_token_size=20,
        chunk_overlap_token_size=0,
    )

    assert [c["chunk_order_index"] for c in chunks] == list(range(len(chunks)))
    assert all(c["content"].strip() for c in chunks)
    # Reconstructed (joined with the splitter's separator semantics) must
    # at least contain each original paragraph as a substring.
    joined = "\n\n".join(c["content"] for c in chunks)
    for para in ("Alpha section.", "Beta section.", "Gamma section."):
        assert para in joined


@pytest.mark.offline
def test_token_field_matches_tokenizer_encode_length():
    chunks = chunking_by_recursive_character(
        _tok(),
        "X" * 50 + "\n\n" + "Y" * 50,
        chunk_token_size=40,
        chunk_overlap_token_size=5,
    )
    tok = _tok()
    for c in chunks:
        assert c["tokens"] == len(tok.encode(c["content"]))


@pytest.mark.offline
def test_custom_separators_are_honored():
    body = "alpha|beta|gamma|delta"
    chunks = chunking_by_recursive_character(
        _tok(),
        body,
        chunk_token_size=10,
        chunk_overlap_token_size=0,
        separators=["|", ""],
    )
    contents = [c["content"] for c in chunks]
    # With "|" as the primary separator and a 10-token cap, each 5-char
    # token name must land in its own chunk.
    assert any("alpha" in c for c in contents)
    assert any("delta" in c for c in contents)
    # Every chunk fits the cap.
    for c in chunks:
        assert c["tokens"] <= 10


@pytest.mark.offline
def test_recursive_chunks_carry_exact_source_spans_with_overlap():
    body = "Alpha section.\n\nBeta section.\n\nGamma section."
    chunks = chunking_by_recursive_character(
        _tok(),
        body,
        chunk_token_size=22,
        chunk_overlap_token_size=6,
    )

    assert len(chunks) >= 2
    for chunk in chunks:
        span = chunk["_source_span"]
        assert body[span["start"] : span["end"]] == chunk["content"]


@pytest.mark.offline
def test_long_unique_text_keeps_every_source_span():
    """Every chunk of a long, non-repeating document must carry a span.

    Regression for the LangChain ``add_start_index`` unit mismatch: with a
    token-based length function its search cursor overshot each chunk's true
    start, dropping ``_source_span`` (and failing span-first backfill) on
    roughly half of a unique document's chunks.
    """
    body = " ".join(f"word{i}" for i in range(800))
    chunks = chunking_by_recursive_character(
        _tok(),
        body,
        chunk_token_size=50,
        chunk_overlap_token_size=10,
    )

    assert len(chunks) > 5
    for chunk in chunks:
        span = chunk["_source_span"]
        assert body[span["start"] : span["end"]] == chunk["content"]


@pytest.mark.offline
def test_repeated_text_spans_tile_whole_document():
    """Spans over heavily repeated text must tile the document, not cluster.

    Regression for the repeated-content ambiguity: a naive forward ``find``
    matches every identical overlapping chunk to the nearest repetition, so all
    spans collapse into the document head and the tail gets no provenance. The
    offset-aware splitter mirror must instead advance spans across the whole text.
    """
    unit = "ABCDE fghij. "
    body = unit * 60
    chunks = chunking_by_recursive_character(
        _tok(),
        body,
        chunk_token_size=40,
        chunk_overlap_token_size=10,
    )

    spans = [c["_source_span"] for c in chunks]
    assert len(spans) == len(chunks)  # no chunk lost its span

    # Each span is exact and starts are monotonically non-decreasing.
    for chunk, span in zip(chunks, spans):
        assert body[span["start"] : span["end"]] == chunk["content"]
    starts = [s["start"] for s in spans]
    assert starts == sorted(starts)

    # Spans reach the document tail rather than clustering at the head. Under
    # the old nearest-repetition behaviour the furthest end stalled near the
    # first few hundred chars; tiling reaches within one ``unit`` of the end.
    assert max(s["end"] for s in spans) >= len(body) - len(unit)


@pytest.mark.offline
def test_span_mirror_matches_langchain_split_text():
    """Drift guard for the offset-aware mirror of LangChain's splitter."""
    body = (
        "Alpha repeats.\n\n"
        "ff hh aa hh ee\n"
        "ff hh aa hh ee\n\n"
        "Tail repeats. Tail repeats. Tail repeats."
    )
    tok = _tok()

    def length_function(text: str) -> int:
        return len(tok.encode(text))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=36,
        chunk_overlap=14,
        length_function=length_function,
        strip_whitespace=True,
    )

    mirrored = _split_text_with_spans(
        body,
        base_offset=0,
        separators=splitter._separators,
        chunk_size=splitter._chunk_size,
        chunk_overlap=splitter._chunk_overlap,
        length_function=length_function,
        keep_separator=splitter._keep_separator,
        is_separator_regex=splitter._is_separator_regex,
        strip_whitespace=splitter._strip_whitespace,
    )

    assert [piece for piece, _, _ in mirrored] == splitter.split_text(body)
    for piece, start, end in mirrored:
        assert body[start:end] == piece


@pytest.mark.offline
def test_chunk_repeating_an_earlier_block_maps_to_its_own_block():
    """A chunk whose text also appears in an earlier, different block must map
    to *its own* occurrence, not the earlier copy.

    Regression for cross-block duplicate provenance: merged text ``"ff aa\\n\\naa"``
    splits into ``["ff aa", "aa"]``. The second chunk ``"aa"`` is block 2, so its
    span must point at the *second* ``"aa"`` (offset 7), not the ``"aa"`` inside
    ``"ff aa"`` (offset 3). The earlier locator anchored on a predicted offset
    and snapped to the nearest copy, silently emitting the wrong — but
    legal-looking — span that strict backfill would trust.
    """
    merged = "ff aa\n\naa"
    chunks = chunking_by_recursive_character(
        _tok(),
        merged,
        chunk_token_size=5,
        chunk_overlap_token_size=1,
    )

    assert [c["content"] for c in chunks] == ["ff aa", "aa"]
    assert chunks[0]["_source_span"] == {"start": 0, "end": 5}
    # The second "aa" lives in block 2 at offset 7, after the "\n\n" separator —
    # not the "aa" embedded in "ff aa" at offset 3.
    assert chunks[1]["_source_span"] == {"start": 7, "end": 9}
    assert merged[7:9] == "aa"


@pytest.mark.offline
def test_overlap_at_least_chunk_size_is_rejected_cleanly():
    """``chunk_overlap_token_size >= chunk_token_size`` must fail closed.

    Regression for a crash reachable from addon_params, direct SDK calls, and
    persisted per-doc snapshots — none of which pass through the REST layer's
    ``_OverlapChunkParams`` validator. Without a backstop here, the same
    invalid combination that validator rejects instead reaches LangChain's
    ``RecursiveCharacterTextSplitter`` constructor and raises its own
    uncontrolled ``ValueError``, whose message is an implementation detail of
    a third-party library rather than a message this codebase controls. The
    sibling fixed-token chunker (``lightrag/chunker/token_size.py``) already
    guards this exact case for the same reason.
    """
    body = "Alpha section.\n\nBeta section.\n\nGamma section."

    with pytest.raises(
        ValueError,
        match=r"chunk_overlap_token_size \(20\) must be < chunk_token_size \(20\)",
    ):
        chunking_by_recursive_character(
            _tok(), body, chunk_token_size=20, chunk_overlap_token_size=20
        )

    with pytest.raises(
        ValueError,
        match=r"chunk_overlap_token_size \(25\) must be < chunk_token_size \(20\)",
    ):
        chunking_by_recursive_character(
            _tok(), body, chunk_token_size=20, chunk_overlap_token_size=25
        )


@pytest.mark.offline
def test_repeated_block_window_does_not_shift_to_previous_duplicate():
    """A repeated multi-block chunk must not slide back into the prior duplicate."""
    block = "ff hh aa hh ee"
    merged = "\n\n".join([block] * 4)
    chunks = chunking_by_recursive_character(
        _tok(),
        merged,
        chunk_token_size=36,
        chunk_overlap_token_size=14,
    )

    assert [c["content"] for c in chunks] == [
        f"{block}\n\n{block}",
        f"{block}\n\n{block}",
    ]
    assert chunks[0]["_source_span"] == {"start": 0, "end": 30}
    assert chunks[1]["_source_span"] == {"start": 32, "end": 62}


# --------------------------------------------------------------------------- #
# Argument floors (non-positive chunk_token_size, negative overlap clamp)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("chunk_token_size", [0, -1])
def test_non_positive_chunk_token_size_raises(chunk_token_size):
    """A non-positive size is rejected, never floored to one-token chunks."""
    with pytest.raises(
        ValueError, match=rf"chunk_token_size must be >= 1, got {chunk_token_size}"
    ):
        chunking_by_recursive_character(
            _tok(), "abc", chunk_token_size=chunk_token_size, chunk_overlap_token_size=0
        )


def test_small_positive_chunk_token_size_is_allowed_on_direct_call():
    """Size 1 is the direct-call floor; ``MIN_CHUNK_TOKEN_SIZE`` is not applied."""
    chunks = chunking_by_recursive_character(
        _tok(), "abc", chunk_token_size=1, chunk_overlap_token_size=0
    )

    assert [c["content"] for c in chunks] == ["a", "b", "c"]
    assert [c["chunk_order_index"] for c in chunks] == [0, 1, 2]


@pytest.mark.parametrize("chunk_overlap_token_size", [-1, -(10**6)])
def test_negative_chunk_overlap_token_size_floors_to_zero(chunk_overlap_token_size):
    """Any negative overlap gives exactly the output of an explicit 0."""
    body = "abcdefghij"

    floored = chunking_by_recursive_character(
        _tok(),
        body,
        chunk_token_size=5,
        chunk_overlap_token_size=chunk_overlap_token_size,
    )
    explicit_zero = chunking_by_recursive_character(
        _tok(), body, chunk_token_size=5, chunk_overlap_token_size=0
    )

    assert [c["content"] for c in floored] == ["abcde", "fghij"]
    assert floored == explicit_zero


def test_overlap_guard_compares_the_floored_overlap():
    """A negative overlap passes the guard even against size 1 (0 < 1)."""
    chunks = chunking_by_recursive_character(
        _tok(), "ab", chunk_token_size=1, chunk_overlap_token_size=-7
    )

    assert [c["content"] for c in chunks] == ["a", "b"]


def test_overlap_guard_still_fires_on_small_size():
    """The default overlap (100) against a small size raises from the guard."""
    with pytest.raises(
        ValueError,
        match=r"chunk_overlap_token_size \(100\) must be < chunk_token_size \(5\)",
    ):
        chunking_by_recursive_character(_tok(), "abc", chunk_token_size=5)


# --------------------------------------------------------------------------- #
# Separator-cascade bounds and no-op amplification guard (GHSA-26pm-px5v-8c4w)
# --------------------------------------------------------------------------- #


class _CountingTokenizer:
    """Counts encodes so cost can be asserted without timing anything."""

    def __init__(self):
        self.encodes = 0

    def encode(self, content: str):
        self.encodes += 1
        # One token per 4 characters, deterministic and cheap.
        return list(range(max(len(content) // 4, 1)))

    def decode(self, tokens):
        return "x" * (len(tokens) * 4)


def _counting_tok() -> tuple[Tokenizer, _CountingTokenizer]:
    underlying = _CountingTokenizer()
    return Tokenizer("test-model", underlying), underlying


@pytest.mark.parametrize(
    "separators, expected",
    [
        pytest.param(None, None, id="none_passthrough"),
        pytest.param(
            list(DEFAULT_R_SEPARATORS),
            list(DEFAULT_R_SEPARATORS),
            id="within_limits_verbatim",
        ),
        pytest.param(
            ["\n\n", "x" * (MAX_R_SEPARATOR_CHARS + 1), "\n"],
            ["\n\n", "\n"],
            id="over_long_dropped",
        ),
        pytest.param(
            [f"s{i}" for i in range(900)],
            [f"s{i}" for i in range(MAX_R_SEPARATORS)],
            id="long_cascade_truncated_without_adding_sentinel",
        ),
        pytest.param(
            [f"s{i}" for i in range(MAX_R_SEPARATORS)] + [""],
            [f"s{i}" for i in range(MAX_R_SEPARATORS - 1)] + [""],
            id="truncation_preserves_char_sentinel",
        ),
        pytest.param(
            ["x" * (MAX_R_SEPARATOR_CHARS + 1)] * 3,
            [],
            id="all_over_long_normalizes_to_empty",
        ),
    ],
)
def test_normalize_r_separators_bounds(separators, expected):
    """``normalize_r_separators`` bounds count and entry length while keeping
    ``None`` untouched and preserving a trailing char-level ``""`` sentinel."""
    bounded = normalize_r_separators(separators)
    assert bounded == expected
    if bounded is not None and separators is not None and "" not in separators:
        assert "" not in bounded


def test_runtime_normalizer_is_silent_for_an_invalid_snapshot(monkeypatch):
    """The hot-path backstop must not recreate a warning per document."""
    import lightrag.chunker.recursive_character as recursive_character

    warnings: list[str] = []
    monkeypatch.setattr(recursive_character.logger, "warning", warnings.append)

    over_long = ["x" * (MAX_R_SEPARATOR_CHARS + 1)] * 3
    too_many = [f"s{index}" for index in range(MAX_R_SEPARATORS + 6)]

    for _ in range(2):
        assert normalize_r_separators(over_long) == []
        assert len(normalize_r_separators(too_many)) == MAX_R_SEPARATORS

    assert warnings == []


def test_the_sentinel_still_splits_an_otherwise_unsplittable_text():
    """A long run with no ordinary separator can only be broken at the char level."""
    tokenizer, _ = _counting_tok()
    text = "A" * 4000
    separators = [f"s{i}" for i in range(MAX_R_SEPARATORS)] + [""]

    chunks = chunking_by_recursive_character(
        tokenizer, text, 100, chunk_overlap_token_size=0, separators=separators
    )

    assert len(chunks) > 1


def test_an_all_over_long_cascade_falls_back_instead_of_crashing():
    """``_split_text_with_spans`` reads ``separators[-1]`` on its first line, so
    an empty cascade must fall back rather than raising IndexError."""
    tokenizer, _ = _counting_tok()
    separators = ["x" * (MAX_R_SEPARATOR_CHARS + 1)] * 3

    chunks = chunking_by_recursive_character(
        tokenizer,
        "hello world " * 200,
        50,
        chunk_overlap_token_size=0,
        separators=separators,
    )

    assert chunks


def test_the_chunker_entry_point_is_silent_across_repeated_calls(monkeypatch):
    """One bad SDK argument must not produce one WARNING per document."""
    import lightrag.chunker.recursive_character as recursive_character

    warnings: list[object] = []
    monkeypatch.setattr(
        recursive_character.logger, "warning", lambda *a, **k: warnings.append(a)
    )

    tokenizer, _ = _counting_tok()
    over_long = ["x" * (MAX_R_SEPARATOR_CHARS + 1)] * 3

    for _ in range(3):
        assert chunking_by_recursive_character(
            tokenizer,
            "hello world\n\nsecond block",
            1200,
            chunk_overlap_token_size=100,
            separators=over_long,
        )

    assert warnings == []


def test_load_chunk_separators_falls_back_to_the_repo_cascade(monkeypatch):
    """``load_chunk_separators`` falls back to ``DEFAULT_R_SEPARATORS`` minus
    the char sentinel and warns once per distinct invalid env value."""
    import lightrag.chunker.recursive_character as recursive_character
    import lightrag.multimodal_context as multimodal_context

    warnings: list[str] = []
    monkeypatch.setattr(recursive_character.logger, "warning", warnings.append)
    monkeypatch.setattr(multimodal_context.logger, "warning", warnings.append)

    monkeypatch.setenv(
        "CHUNK_R_SEPARATORS",
        json.dumps(["multimodal-warning-cache" * (MAX_R_SEPARATOR_CHARS + 1)] * 3),
    )

    assert multimodal_context.load_chunk_separators() == [
        s for s in DEFAULT_R_SEPARATORS if s
    ]
    assert multimodal_context.load_chunk_separators() == [
        s for s in DEFAULT_R_SEPARATORS if s
    ]
    assert sum("[CHUNK_R_SEPARATORS] dropped" in message for message in warnings) == 1
    assert (
        sum(
            "[load_chunk_separators] no usable separators" in message
            for message in warnings
        )
        == 1
    )


def test_the_all_over_long_fallback_matches_separators_none():
    """All-overlong cascade falls back to the splitter's own default, not
    DEFAULT_R_SEPARATORS."""
    text = "第一句。第二句！第三句？" * 40 + "\n\nEnglish paragraph here. " * 20
    tokenizer_a, _ = _counting_tok()
    tokenizer_b, _ = _counting_tok()

    fallback = chunking_by_recursive_character(
        tokenizer_a,
        text,
        60,
        chunk_overlap_token_size=0,
        separators=["x" * (MAX_R_SEPARATOR_CHARS + 1)],
    )
    default = chunking_by_recursive_character(
        tokenizer_b, text, 60, chunk_overlap_token_size=0, separators=None
    )

    assert [c["content"] for c in fallback] == [c["content"] for c in default]


def test_separators_none_is_not_rewritten_to_the_repo_cascade():
    """Direct SDK ``separators=None`` keeps LangChain's default (no CJK)."""
    text = "第一句。第二句！第三句？" * 40
    tokenizer_a, _ = _counting_tok()
    tokenizer_b, _ = _counting_tok()

    as_none = chunking_by_recursive_character(
        tokenizer_a, text, 60, chunk_overlap_token_size=0, separators=None
    )
    as_repo_cascade = chunking_by_recursive_character(
        tokenizer_b,
        text,
        60,
        chunk_overlap_token_size=0,
        separators=list(DEFAULT_R_SEPARATORS),
    )

    assert [c["content"] for c in as_none] != [c["content"] for c in as_repo_cascade]


def _sep_bomb_text() -> str:
    """The advisory's payload shape: the separator occurs once, at offset 0."""
    return "Q" + ("X " * 4000)


@pytest.mark.parametrize("count", [1, 8, MAX_R_SEPARATORS])
def test_a_no_op_cascade_costs_the_same_regardless_of_its_length(count):
    """A separator matching only at offset 0 must not re-encode per level."""
    tokenizer, underlying = _counting_tok()

    chunking_by_recursive_character(
        tokenizer, _sep_bomb_text(), 1200, separators=["Q"] * count
    )

    assert underlying.encodes <= 4


def test_the_no_op_skip_does_not_change_the_output():
    """Skipping a level in place must reach the same state the recursion did."""
    tokenizer, _ = _counting_tok()
    text = _sep_bomb_text()

    with_bomb = chunking_by_recursive_character(
        tokenizer, text, 1200, separators=["Q"] * MAX_R_SEPARATORS
    )

    assert len(with_bomb) == 1
    assert with_bomb[0]["content"].startswith("QX")


@pytest.mark.parametrize("chunk_size", [16, 17])
def test_no_op_skip_sends_a_piece_below_chunk_size_to_the_merge_step(chunk_size):
    """The skip must not push a short piece down to weaker separators."""
    text = "Q" + "a" * 64
    separators = ["Q"] * (MAX_R_SEPARATORS - 1) + [""]
    tokenizer, underlying = _counting_tok()
    expected = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        separators=separators,
        length_function=lambda value: len(tokenizer.encode(value)),
    ).split_text(text)
    underlying.encodes = 0

    chunks = chunking_by_recursive_character(
        tokenizer,
        text,
        chunk_size,
        chunk_overlap_token_size=0,
        separators=separators,
    )

    assert [chunk["content"] for chunk in chunks] == expected
    for chunk in chunks:
        span = chunk["_source_span"]
        assert text[span["start"] : span["end"]] == chunk["content"]
    if chunk_size == 17:
        assert expected == [text]
        assert underlying.encodes <= 5
    else:
        assert len(expected) > 1


def test_a_separator_that_really_splits_is_unaffected():
    """The skip must only trigger on a genuine no-op."""
    tokenizer, _ = _counting_tok()
    text = "alpha|beta|gamma|delta " * 50

    chunks = chunking_by_recursive_character(
        tokenizer, text, 20, chunk_overlap_token_size=0, separators=["|", ""]
    )

    assert len(chunks) > 1
    assert "".join(c["content"] for c in chunks).replace(" ", "").count("alpha") == 50


def test_default_cascade_output_is_unchanged_by_the_skip():
    """Sanity net over ordinary text: the optimisation is cost-only."""
    tokenizer, _ = _counting_tok()
    text = "Para one.\n\nPara two is longer and wordier.\n\nPara three.\n" * 20

    chunks = chunking_by_recursive_character(
        tokenizer,
        text,
        40,
        chunk_overlap_token_size=0,
        separators=list(DEFAULT_R_SEPARATORS),
    )

    rejoined = "".join(c["content"] for c in chunks)
    assert "Para one." in rejoined and "Para three." in rejoined
    assert len(chunks) > 1

