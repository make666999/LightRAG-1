"""Tests for token tracking, token splitting, truncation tallying, and prompt resolution.

Covers:
- ``TokenTracker.add_usage`` nullable/optional counts and cached prompt tokens.
- ``split_text_by_token_limit`` non-positive limits, empty text, and splitting.
- ``TokenLimitTruncationTally`` and ``merge_truncation_metadata`` aggregation
  and sample capping.
- ``resolve_user_prompt`` prefix composition, whitespace preservation, cache
  key backward compatibility, and ``disable_prefix``.
"""

from __future__ import annotations

import pytest

from lightrag.utils import (
    LLM_TRUNCATION_METADATA_KEY,
    TRUNCATION_METADATA_SAMPLE_LIMIT,
    TokenLimitTruncationTally,
    TokenTracker,
    Tokenizer,
    TokenizerInterface,
    merge_truncation_metadata,
    resolve_user_prompt,
    split_text_by_token_limit,
)

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# TokenTracker
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reported_total", "expected_total"),
    [
        pytest.param(None, 2, id="none-total-is-summed"),
        pytest.param(0, 0, id="reported-zero-is-kept"),
    ],
)
def test_add_usage_treats_none_counts_as_missing(reported_total, expected_total):
    tracker = TokenTracker()

    tracker.add_usage(
        {"prompt_tokens": 2, "completion_tokens": None, "total_tokens": reported_total}
    )

    assert tracker.get_usage() == {
        "prompt_tokens": 2,
        "completion_tokens": 0,
        "total_tokens": expected_total,
        "call_count": 1,
    }


def test_cached_prompt_tokens_sum_only_reported_calls():
    tracker = TokenTracker()
    usage = {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}

    tracker.add_usage({**usage, "cached_prompt_tokens": 2})
    tracker.add_usage(usage)
    tracker.add_usage({**usage, "cached_prompt_tokens": 3})

    assert tracker.get_usage()["cached_prompt_tokens"] == 5
    assert str(tracker).endswith(", Cached prompt tokens: 5")
    tracker.reset()
    assert "cached_prompt_tokens" not in tracker.get_usage()


# ---------------------------------------------------------------------------
# split_text_by_token_limit
# ---------------------------------------------------------------------------


class _DummyTokenizer(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) % 1000 for ch in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


def _tok() -> Tokenizer:
    return Tokenizer(model_name="dummy", tokenizer=_DummyTokenizer())


@pytest.mark.parametrize(
    ("text", "limit", "expected"),
    [
        pytest.param(
            "Hello world. This is a second sentence that needs splitting!",
            0,
            [],
            id="zero-limit",
        ),
        pytest.param("Hello world. Another sentence.", -5, [], id="negative-limit"),
        pytest.param("", 10, [], id="empty-text"),
        pytest.param("abcdefghij", 4, ["abcd", "efgh", "ij"], id="positive-limit"),
    ],
)
def test_split_text_by_token_limit(text, limit, expected):
    pieces = split_text_by_token_limit(text, _tok(), limit)
    assert pieces == expected
    if limit > 0:
        assert all(len(_tok().encode(p)) <= limit for p in pieces)


# ---------------------------------------------------------------------------
# TokenLimitTruncationTally & merge_truncation_metadata
# ---------------------------------------------------------------------------


def test_a_clean_run_records_nothing():
    tally = TokenLimitTruncationTally()

    assert not tally
    assert tally.events == 0
    assert tally.affected == 0
    assert tally.as_metadata() is None
    # An empty fragment clears a previous attempt's summary in doc_status.
    assert tally.as_metadata_extra() == {}


def test_only_the_first_event_is_announced():
    tally = TokenLimitTruncationTally()

    assert tally.record("initial", "chunk-001") is True
    assert tally.record("initial", "chunk-002") is False
    assert tally.record("gleaning", "chunk-001") is False


def test_events_count_responses_while_subjects_deduplicate():
    """A chunk truncated at both stages is 2 events but 1 affected chunk."""
    tally = TokenLimitTruncationTally()
    tally.record("initial", "chunk-001")
    tally.record("gleaning", "chunk-001")
    tally.record("initial", "chunk-002")

    assert tally.events == 3
    assert tally.affected == 2
    assert tally.as_metadata()["stages"] == {"initial": 2, "gleaning": 1}
    assert tally.stage_breakdown() == "initial: 2, gleaning: 1"


def test_absorb_folds_a_stage_tally_into_the_document_tally():
    extraction = TokenLimitTruncationTally()
    extraction.record("initial", "chunk-001")
    extraction.record("gleaning", "chunk-001")
    merge = TokenLimitTruncationTally()
    merge.record("summary", "Entity:ALICE")

    document = TokenLimitTruncationTally()
    document.absorb(extraction)
    document.absorb(merge)

    assert document.events == 3
    assert document.affected == 2
    assert document.as_metadata()["stages"] == {
        "initial": 1,
        "gleaning": 1,
        "summary": 1,
    }
    assert document.as_metadata()["samples"] == ["chunk-001", "Entity:ALICE"]


def test_absorb_deduplicates_subjects_across_stages():
    document = TokenLimitTruncationTally()
    document.record("initial", "chunk-001")

    other = TokenLimitTruncationTally()
    other.record("summary", "chunk-001")
    document.absorb(other)

    assert document.events == 2
    assert document.affected == 1


def test_absorb_tolerates_an_absent_tally():
    document = TokenLimitTruncationTally()
    document.absorb(None)
    document.absorb(TokenLimitTruncationTally())

    assert not document


def test_the_metadata_sample_is_bounded_by_document_size():
    tally = TokenLimitTruncationTally()
    total = TRUNCATION_METADATA_SAMPLE_LIMIT + 7
    for index in range(total):
        tally.record("initial", f"chunk-{index:03d}")

    payload = tally.as_metadata()

    assert payload["events"] == total
    assert payload["affected"] == total
    assert len(payload["samples"]) == TRUNCATION_METADATA_SAMPLE_LIMIT
    assert payload["samples"][0] == "chunk-000"
    assert payload["samples_omitted"] == 7


def test_no_omitted_counter_when_every_subject_fits():
    tally = TokenLimitTruncationTally()
    tally.record("initial", "chunk-001")

    assert "samples_omitted" not in tally.as_metadata()


def test_metadata_extra_is_keyed_for_the_doc_status_transition():
    tally = TokenLimitTruncationTally()
    tally.record("initial", "chunk-001")

    extra = tally.as_metadata_extra()

    assert set(extra) == {LLM_TRUNCATION_METADATA_KEY}
    assert extra[LLM_TRUNCATION_METADATA_KEY]["events"] == 1


class TestMergeTruncationMetadata:
    """Combining persisted payloads (custom-chunk patch semantics)."""

    @staticmethod
    def _payload(**overrides):
        payload = {
            "events": 1,
            "affected": 1,
            "stages": {"initial": 1},
            "samples": ["chunk-001"],
        }
        payload.update(overrides)
        return payload

    def test_either_side_absent_returns_a_copy_of_the_other(self):
        base = self._payload()
        assert merge_truncation_metadata(None, None) is None
        assert merge_truncation_metadata(base, None) == base
        assert merge_truncation_metadata(base, None) is not base
        assert merge_truncation_metadata(None, base) == base

    def test_complete_samples_merge_exactly(self):
        base = self._payload(
            events=3,
            affected=2,
            stages={"initial": 2, "summary": 1},
            samples=["chunk-001", "Entity:ALICE"],
        )
        extra = self._payload(
            events=2,
            affected=2,
            stages={"initial": 1, "summary": 1},
            samples=["chunk-101", "Entity:ALICE"],
        )

        merged = merge_truncation_metadata(base, extra)

        assert merged["events"] == 5
        assert merged["stages"] == {"initial": 3, "summary": 2}
        assert merged["affected"] == 3
        assert merged["samples"] == ["chunk-001", "Entity:ALICE", "chunk-101"]
        assert "samples_omitted" not in merged

    def test_incomplete_samples_fall_back_to_summed_counts(self):
        limit = TRUNCATION_METADATA_SAMPLE_LIMIT
        base = self._payload(
            events=limit + 5,
            affected=limit + 5,
            samples=[f"chunk-{i:03d}" for i in range(limit)],
            samples_omitted=5,
            stages={"initial": limit + 5},
        )
        extra = self._payload(samples=["chunk-000"])

        merged = merge_truncation_metadata(base, extra)

        assert merged["events"] == limit + 6
        assert merged["affected"] == limit + 5
        assert len(merged["samples"]) == limit
        assert merged["samples_omitted"] == 5

    def test_merged_samples_stay_capped(self):
        limit = TRUNCATION_METADATA_SAMPLE_LIMIT
        base = self._payload(
            events=limit,
            affected=limit,
            samples=[f"base-{i}" for i in range(limit)],
            stages={"initial": limit},
        )
        extra = self._payload(samples=["patch-0"])

        merged = merge_truncation_metadata(base, extra)

        assert len(merged["samples"]) == limit
        assert merged["samples"] == base["samples"]
        assert merged["affected"] == limit + 1
        assert merged["samples_omitted"] == 1


# ---------------------------------------------------------------------------
# resolve_user_prompt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("user_prompt", "prefix", "expected"),
    [
        pytest.param(
            "Be concise.",
            "Answer in Chinese.\n\n",
            "Answer in Chinese.\n\nBe concise.",
            id="prefix-precedes-user-prompt",
        ),
        pytest.param(
            "Be concise.",
            "Answer in Chinese.",
            "Answer in Chinese.Be concise.",
            id="no-separator-injected",
        ),
        pytest.param(
            "Be concise.",
            "  Answer in Chinese.  \n\n",
            "  Answer in Chinese.  \n\nBe concise.",
            id="prefix-trailing-whitespace-preserved",
        ),
        pytest.param(
            "  Be concise.  ",
            "",
            "  Be concise.  ",
            id="user-prompt-not-stripped",
        ),
    ],
)
def test_resolve_user_prompt_composition_and_whitespace(user_prompt, prefix, expected):
    assert resolve_user_prompt(user_prompt, prefix).text == expected


@pytest.mark.parametrize("prefix", [None, "", "   "])
@pytest.mark.parametrize("user_prompt", [None, "", "   "])
def test_empty_and_whitespace_combinations(prefix, user_prompt):
    result = resolve_user_prompt(user_prompt, prefix)
    assert result.text == (prefix or "") + (user_prompt or "")
    if result.text:
        assert result.slot == f"\n\n{result.text}"
    else:
        assert result.slot == "n/a"


@pytest.mark.parametrize("empty", [None, ""])
def test_empty_user_prompt_makes_the_prefix_the_whole_instruction(empty):
    prefix = "Answer in the language of the question.\n\n"
    result = resolve_user_prompt(empty, prefix)
    assert result.text == prefix
    assert result.slot == f"\n\n{prefix}"


def test_prefix_only_is_indistinguishable_from_a_caller_sending_it():
    prefix_only = resolve_user_prompt(None, "House style.")
    caller_sent = resolve_user_prompt("House style.", None)
    assert prefix_only == caller_sent
    assert prefix_only.slot == "\n\nHouse style."


@pytest.mark.parametrize(
    ("user_prompt", "prefix"),
    [(user_prompt, None) for user_prompt in ("Be concise.", "多行\n文本")]
    + [("Be concise.", "")],
)
def test_unconfigured_prefix_preserves_legacy_cache_key_component(user_prompt, prefix):
    legacy = user_prompt or ""
    assert resolve_user_prompt(user_prompt, prefix).text == legacy


def test_disable_prefix_behavior():
    with_user = resolve_user_prompt(
        "Be concise.", "Answer in Chinese.\n\n", disable_prefix=True
    )
    assert with_user.text == "Be concise."
    assert with_user.slot == "\n\nBe concise."

    without_user = resolve_user_prompt(
        None, "Answer in Chinese.\n\n", disable_prefix=True
    )
    assert without_user.text == ""
    assert without_user.slot == "n/a"

    disabled = resolve_user_prompt("Be concise.", "House style.", disable_prefix=True)
    unconfigured = resolve_user_prompt("Be concise.", None)
    assert disabled == unconfigured
