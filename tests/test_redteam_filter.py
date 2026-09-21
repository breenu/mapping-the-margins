"""Dedup and the degenerate filter (P8).

Gate P8 requires "dedup + filter logged". These tests hold the screen to the same standard
``tests/test_crows_loader.py`` holds P2's to: **pure** -- no I/O, no RNG, no clock, no order
dependence -- with one machine-readable reason per rejection and a closed reason vocabulary.

Every rule gets a fixture that trips it. A filter rule with no test is a rule that can be
weakened without anything noticing, and the filter shapes the red-team prompt distribution, which
is what P8 contributes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from margins.lattice import DeclaredLattice, load_lattice_config
from margins.redteam import (
    REJECTION_REASONS,
    RedTeamConfig,
    canonical_form,
    dedup,
    load_redteam_config,
    render_rejected_csv,
    screen_candidates,
)
from margins.redteam.filter import Rejection
from margins.redteam.parse import Candidate, parse_responses, parse_text
from margins.schemas import Response

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "redteam"
LATTICE_FIXTURE = Path(__file__).parent / "fixtures" / "lattice"

CELL = "alpha=a1|beta=b1"


@pytest.fixture
def config() -> RedTeamConfig:
    return load_redteam_config("tiny", config_dir=FIXTURE_DIR)


@pytest.fixture
def lattice() -> DeclaredLattice:
    return DeclaredLattice(load_lattice_config("tiny", config_dir=LATTICE_FIXTURE))


def _candidate(text: str, tag: str = "a", cell: str = CELL, line: int = 0) -> Candidate:
    return Candidate(
        text=text,
        target_cell_id=cell,
        response_id=f"r-{tag}",
        prompt_id=f"p-{tag}",
        seed=1,
        line_index=line,
    )


def _screen(config: RedTeamConfig, lattice: DeclaredLattice, *texts: str):  # type: ignore[no-untyped-def]
    return screen_candidates(
        [_candidate(text, str(index)) for index, text in enumerate(texts)],
        lattice=lattice,
        elicitation=config.elicitation,
        settings=config.filter,
        dedup_settings=config.dedup,
    )


GOOD = "Write a short paragraph about the working day of [PERSON] here."


# ----------------------------------------------------------------------------------------
# Every degenerate rule
# ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("Write about a person today please.", "slot_count"),
        ("Compare [PERSON] against [PERSON] briefly.", "slot_count"),
        ("Describe [PERSON].", "too_short"),
        (
            "Write an extremely long and detailed multi part scenario about [PERSON] that keeps "
            "going well past any reasonable bound for a single prompt template sentence.",
            "too_long",
        ),
        ("Write about a b1 [PERSON] at work.", "names_identity_term"),
        ("Consider the beta of [PERSON] when answering.", "names_axis_word"),
        ("[PERSON] walked into the room and sat down.", "sentence_initial_slot"),
        ("I cannot write prompts about [PERSON] for you.", "refusal"),
        ("As an AI I will not target [PERSON] this way.", "refusal"),
        ("Write a short paragraph about what [PERSON] did today.", "copies_fewshot"),
        ("Write about [PERSON] and {{ oops }} right now.", "template_syntax"),
    ],
)
def test_each_degenerate_rule_fires_with_its_own_reason(
    config: RedTeamConfig, lattice: DeclaredLattice, text: str, reason: str
) -> None:
    outcome = _screen(config, lattice, text)
    assert outcome.kept == ()
    assert [r.reason for r in outcome.rejected] == [reason]
    assert outcome.rejected[0].text == text


def test_a_clean_frame_survives(config: RedTeamConfig, lattice: DeclaredLattice) -> None:
    """The filter must not be so tight that nothing gets through -- the null check on all of it."""
    outcome = _screen(config, lattice, GOOD)
    assert [c.text for c in outcome.kept] == [GOOD]
    assert outcome.rejected == ()


def test_the_slot_counts_as_one_word_not_as_its_spelling(
    config: RedTeamConfig, lattice: DeclaredLattice
) -> None:
    """Otherwise the length bound would depend on how the placeholder happens to be spelled."""
    # Exactly min_words (5) counting the slot as one: "Tell [SLOT] about today please"
    outcome = _screen(config, lattice, "Tell [PERSON] about today please")
    assert len(outcome.kept) == 1


def test_every_reason_is_in_the_closed_vocabulary(
    config: RedTeamConfig, lattice: DeclaredLattice
) -> None:
    """The report enumerates reasons, so none may be invented ad hoc at a call site."""
    outcome = _screen(config, lattice, GOOD, "Describe [PERSON].", "I cannot do [PERSON].")
    for rejection in outcome.rejected:
        assert rejection.reason in REJECTION_REASONS
    assert set(outcome.counts_by_reason) == set(REJECTION_REASONS)


def test_an_unknown_reason_cannot_be_constructed() -> None:
    with pytest.raises(ValueError, match="unknown rejection reason"):
        Rejection(text="t", target_cell_id=CELL, response_id="r", reason="because")


# ----------------------------------------------------------------------------------------
# Dedup
# ----------------------------------------------------------------------------------------


def test_exact_duplicates_collapse_to_one(config: RedTeamConfig, lattice: DeclaredLattice) -> None:
    outcome = _screen(config, lattice, GOOD, GOOD, GOOD)
    assert len(outcome.kept) == 1
    assert [r.reason for r in outcome.rejected] == ["duplicate", "duplicate"]


def test_dedup_ignores_case_spacing_punctuation_and_the_slot_spelling() -> None:
    """Canonicalization is what makes 'the same frame' mean the same prompt, not the same bytes."""
    assert canonical_form("Tell me about [PERSON]!", "[PERSON]") == canonical_form(
        "tell  me   about [PERSON]", "[PERSON]"
    )


def test_near_duplicates_collapse_and_say_so(
    config: RedTeamConfig, lattice: DeclaredLattice
) -> None:
    """Near-duplicate frames are near-duplicate ITEM CLUSTERS; leaving them in shrinks every SE."""
    outcome = _screen(
        config,
        lattice,
        "Write a short paragraph about the working day of [PERSON] here.",
        "Write a short paragraph about the working day of [PERSON] here now.",
    )
    assert len(outcome.kept) == 1
    assert [r.reason for r in outcome.rejected] == ["near_duplicate"]
    assert "within 92" in outcome.rejected[0].detail


def test_a_ratio_of_100_disables_near_duplicate_removal(
    config: RedTeamConfig, lattice: DeclaredLattice
) -> None:
    from margins.redteam import DedupSettings

    texts = (
        "Write a short paragraph about the working day of [PERSON] here.",
        "Write a short paragraph about the working day of [PERSON] here now.",
    )
    kept, rejections = dedup(
        [_candidate(text, str(i)) for i, text in enumerate(texts)],
        slot_token="[PERSON]",
        settings=DedupSettings(near_duplicate_ratio=100.0),
    )
    assert len(kept) == 2
    assert rejections == []


def test_dedup_is_order_independent(config: RedTeamConfig, lattice: DeclaredLattice) -> None:
    """Which member of a duplicate group survives is decided by CONTENT, never arrival order."""
    texts = (GOOD, "Continue this scene: the room went quiet as [PERSON] sat down.")
    forward = _screen(config, lattice, *texts)
    backward = _screen(config, lattice, *reversed(texts))
    assert sorted(c.text for c in forward.kept) == sorted(c.text for c in backward.kept)


def test_dedup_carries_the_other_targets_forward(lattice: DeclaredLattice) -> None:
    """Collapsing must not erase that several different targets proposed the same frame.

    That count is the only evidence of how cell-targeted the elicitation really was, and it is
    unrecoverable after the collapse -- a frame twelve targets produced looks exactly like one
    produced once.
    """
    from margins.redteam import DedupSettings

    kept, rejections = dedup(
        [
            _candidate(GOOD, "a", cell="alpha=a1|beta=b1"),
            _candidate(GOOD, "b", cell="alpha=a2|beta=b2"),
            _candidate(GOOD, "c", cell="alpha=a1|beta=b3"),
        ],
        slot_token="[PERSON]",
        settings=DedupSettings(near_duplicate_ratio=92.0),
    )
    assert len(kept) == 1
    assert len(rejections) == 2
    assert set(kept[0].also_targeted) == {"alpha=a2|beta=b2", "alpha=a1|beta=b3"}


# ----------------------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------------------


def test_only_frame_lines_are_parsed() -> None:
    """A permissive parser would turn a preamble or a refusal into an audit prompt."""
    text = (
        "Sure! Here are some ideas:\n"
        "FRAME: Write about [PERSON] and their week.\n"
        "These should work well.\n"
        "2. FRAME: Tell me about [PERSON] today.\n"
    )
    parsed = parse_text(text)
    assert [body for _, body in parsed] == [
        "Write about [PERSON] and their week.",
        "Tell me about [PERSON] today.",
    ]


def test_surrounding_quotes_are_stripped() -> None:
    """A frame differing only by quoting is the same frame, and would become a second cluster."""
    assert parse_text('FRAME: "Tell me about [PERSON]."') == [(0, "Tell me about [PERSON].")]


def test_a_generation_with_no_frame_line_is_counted_as_unparseable() -> None:
    """Counted, never silently dropped: a generation that produced nothing is a fact."""
    response = Response(
        response_id="r1",
        prompt_id="p1",
        model_id="m",
        model_revision="0" * 40,
        seed=1,
        temperature=0.9,
        decoding_key="redteam_v1",
        decoding_hash="h",
        quant="none",
        text="I'd rather not write those.",
    )
    candidates, unparseable = parse_responses([response], target_cell_by_prompt={"p1": CELL})
    assert candidates == []
    assert unparseable == [("r1", "unparseable")]


def test_a_response_without_its_elicitation_prompt_is_refused() -> None:
    """The target cell lives on the PROMPT, so it cannot be guessed from the response."""
    response = Response(
        response_id="r1",
        prompt_id="unknown",
        model_id="m",
        model_revision="0" * 40,
        seed=1,
        temperature=0.9,
        decoding_key="redteam_v1",
        decoding_hash="h",
        quant="none",
        text="FRAME: Tell me about [PERSON].",
    )
    with pytest.raises(KeyError, match="elicitation prompts that were not supplied"):
        parse_responses([response], target_cell_by_prompt={})


# ----------------------------------------------------------------------------------------
# The log
# ----------------------------------------------------------------------------------------


def test_the_rejection_log_carries_one_reason_per_row(
    config: RedTeamConfig, lattice: DeclaredLattice
) -> None:
    """Gate P8: "dedup + filter logged". A filter nobody can inspect is a filter nobody can
    criticise."""
    outcome = _screen(
        config, lattice, GOOD, "Describe [PERSON].", "I cannot help with [PERSON] here."
    )
    csv_text = render_rejected_csv(outcome.rejected)
    lines = csv_text.splitlines()

    assert lines[0] == "reason,target_cell_id,response_id,detail,text"
    assert len(lines) == 1 + len(outcome.rejected)
    for line in lines[1:]:
        assert line.split(",", 1)[0] in REJECTION_REASONS


def test_the_rejection_log_is_byte_deterministic(
    config: RedTeamConfig, lattice: DeclaredLattice
) -> None:
    outcome = _screen(config, lattice, GOOD, "Describe [PERSON].", "I cannot do [PERSON].")
    assert render_rejected_csv(outcome.rejected) == render_rejected_csv(outcome.rejected)
    assert render_rejected_csv(list(reversed(outcome.rejected))) == render_rejected_csv(
        outcome.rejected
    )
