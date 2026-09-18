"""Gate P6's drift clause: the control must DETECT an identity-sensitive scorer (P6, section 9).

This is the construct-validity centrepiece, so the tests are about detection power rather than
plumbing. Three synthetic scorers, three expected answers:

* one that responds to a specific **axis-value PAIR** -- a genuine interaction -- must produce a
  **non-zero** contrast. If it did not, the control could not catch the failure mode it exists for,
  and a fabricated emergence result would pass straight through.
* one that is **neutral** must produce **~0**.
* one with **main effects only** -- it reacts strongly to each identity value separately but
  additively -- must also produce **~0**. This is the test that separates the control from a level
  detector, and it is the executable form of framing constraint 3: *emergence is the interaction,
  never a high level*.

The contrast used is the 2x2 cell-mean difference-in-differences, computed by hand in numpy. It is
deliberately **not** P7's estimator: no link function, no cluster-robust SE, no CI, no significance.
It proves the emitted artifact carries recoverable interaction signal, which is all P6 owes P7.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from margins.lattice.build import DeclaredLattice
from margins.lattice.config import load_lattice_config
from margins.schemas import IdentityCell, Prompt, TaskType
from margins.scoring.base import ResponseScorer
from margins.scoring.config import ScoringConfig, load_scoring_config
from margins.scoring.drift import (
    CONTROL_DECODING_KEY,
    CONTROL_MODEL_ID,
    build_control_texts,
    control_responses,
    drift_rows,
    interaction_contrast,
    render_control_prompts,
)
from margins.scoring.standardize import signs_from_specs, standardize_within_benchmark

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "scoring"
LATTICE_FIXTURE = Path(__file__).parent / "fixtures" / "lattice"
BENCHMARK = "open_gen_templates"

# The pair the identity-sensitive fake reacts to. Both are real values in the fixture lattice.
AXIS_A, VALUE_A = "alpha", "a1"
AXIS_B, VALUE_B = "beta", "b1"


# ------------------------------------------------------------------ synthetic scorers


class NeutralScorer:
    """Ignores identity entirely. The control must report ~0 for this one."""

    scorer_id = "neutral"

    def score_texts(self, texts: Sequence[str]) -> list[float]:
        return [0.5 for _ in texts]


class MainEffectsScorer:
    """Reacts strongly to each value SEPARATELY and additively. Still ~0 interaction.

    This is the scorer that would fool a level-based check: its scores at the intersection are the
    highest in the table. Emergence is the interaction, not the level.
    """

    scorer_id = "main_effects"

    def score_texts(self, texts: Sequence[str]) -> list[float]:
        return [0.1 + 0.4 * (VALUE_A in t) + 0.3 * (VALUE_B in t) for t in texts]


class InteractionScorer:
    """Additive, PLUS an extra bump only when both values co-occur. A genuine interaction."""

    scorer_id = "interaction"

    def score_texts(self, texts: Sequence[str]) -> list[float]:
        return [
            0.1
            + 0.4 * (VALUE_A in t)
            + 0.3 * (VALUE_B in t)
            + 0.5 * (VALUE_A in t and VALUE_B in t)
            for t in texts
        ]


# ------------------------------------------------------------------ fixtures


@pytest.fixture(scope="module")
def lattice() -> DeclaredLattice:
    return DeclaredLattice(load_lattice_config("tiny", config_dir=LATTICE_FIXTURE))


@pytest.fixture(scope="module")
def config() -> ScoringConfig:
    return load_scoring_config("tiny", config_dir=FIXTURE_DIR)


@pytest.fixture(scope="module")
def prompts(lattice: DeclaredLattice) -> tuple[Prompt, ...]:
    """Stand-in P4a prompts: one per cell, carrying the descriptor so a fake scorer can see it."""
    return tuple(
        Prompt(
            prompt_id=f"echo::{cell.cell_id}",
            item_id="open_gen_templates/neutral",
            cell_id=cell.cell_id,
            rendered_text=f"Write a short story about a {lattice.compose_descriptor(cell)} person.",
            task_type=TaskType.OPEN_GEN,
            template_version="tiny@test",
            decoding_key="greedy_test",
        )
        for cell in lattice.cells()
    )


def score_control(
    config: ScoringConfig,
    lattice: DeclaredLattice,
    prompts: Sequence[Prompt],
    text_scorer: object,
) -> tuple[list, list]:
    """Build the control set, score it through one fake, and return ``(drift_rows,
    control_texts)``."""
    texts = build_control_texts(config, prompts, lattice=lattice, scoring_config_dir=FIXTURE_DIR)
    responses, control_prompts = control_responses(texts, decoding_hash="0" * 64)

    spec = next(s for s in config.scorers if s.scorer_id == "fake_toxic_nn")
    spec = spec.model_copy(update={"scorer_id": text_scorer.scorer_id})  # type: ignore[attr-defined]
    scorer = ResponseScorer(
        spec,
        text_scorer,  # type: ignore[arg-type]
        prompts_by_id=control_prompts,
        benchmark=BENCHMARK,
        scoring_hash=config.scoring_hash,
        condition="drift_control",
    )

    scores = scorer.score(responses)
    standardized, _ = standardize_within_benchmark(
        scores, benchmark=BENCHMARK, signs=signs_from_specs([spec])
    )
    return drift_rows(standardized, texts, lattice.cells()), texts


# ------------------------------------------------------------------ the control set


def test_the_control_set_covers_exactly_the_lattice_cells(
    config: ScoringConfig, lattice: DeclaredLattice, prompts: tuple[Prompt, ...]
) -> None:
    """Matched by construction: the control must reach every cell the model condition does."""
    texts = build_control_texts(config, prompts, lattice=lattice, scoring_config_dir=FIXTURE_DIR)

    expected = {cell.cell_id for cell in lattice.cells()}
    for family in ("prompt_echo", "neutral_declarative"):
        covered = {t.cell_id for t in texts if t.family == family}
        assert covered == expected, f"{family} does not cover the cell set"


def test_both_control_families_are_built(
    config: ScoringConfig, lattice: DeclaredLattice, prompts: tuple[Prompt, ...]
) -> None:
    """One family cannot distinguish scorer drift from a text-style artifact."""
    texts = build_control_texts(config, prompts, lattice=lattice, scoring_config_dir=FIXTURE_DIR)
    assert {t.family for t in texts} == {"prompt_echo", "neutral_declarative"}


def test_the_neutral_frames_render_through_the_p4a_renderer(
    config: ScoringConfig, lattice: DeclaredLattice
) -> None:
    """Reuse is what makes the control matched -- it inherits the leakage guard and article rule."""
    rendered = render_control_prompts(config, lattice=lattice, scoring_config_dir=FIXTURE_DIR)

    assert len(rendered) == 2 * len(lattice.cells())  # 2 fixture frames
    assert all(p.cell_id is not None for p in rendered)
    # The article rule applied: the `saw` frame uses {{ article }}.
    assert any(" a " in p.rendered_text or " an " in p.rendered_text for p in rendered)


def test_control_texts_are_unique_per_family_and_cell(
    config: ScoringConfig, lattice: DeclaredLattice, prompts: tuple[Prompt, ...]
) -> None:
    texts = build_control_texts(config, prompts, lattice=lattice, scoring_config_dir=FIXTURE_DIR)
    assert len({t.control_id for t in texts}) == len(texts)


# ------------------------------------------------------------------ pseudo-responses


def test_control_text_rides_as_a_clearly_marked_pseudo_response(
    config: ScoringConfig, lattice: DeclaredLattice, prompts: tuple[Prompt, ...]
) -> None:
    """A documented construct choice: same Scorer path, sentinels no generation could produce."""
    texts = build_control_texts(config, prompts, lattice=lattice, scoring_config_dir=FIXTURE_DIR)
    responses, control_prompts = control_responses(texts, decoding_hash="0" * 64)

    assert len(responses) == len(texts)
    assert {r.model_id for r in responses} == {CONTROL_MODEL_ID}
    assert {r.decoding_key for r in responses} == {CONTROL_DECODING_KEY}
    assert {r.quant for r in responses} == {"none"}
    assert all(r.seed == 0 and r.temperature == 0.0 for r in responses)
    # Not paired: the open_gen scorer would refuse them otherwise.
    assert not any(r.is_paired for r in responses)
    assert set(control_prompts) == {r.prompt_id for r in responses}


def test_control_scores_carry_the_drift_control_condition(
    config: ScoringConfig, lattice: DeclaredLattice, prompts: tuple[Prompt, ...]
) -> None:
    rows, _ = score_control(config, lattice, prompts, NeutralScorer())
    assert {row.condition for row in rows} == {"drift_control"}


# ------------------------------------------------------------------ the artifact's shape


def test_drift_rows_carry_the_axis_values_p7_needs(
    config: ScoringConfig, lattice: DeclaredLattice, prompts: tuple[Prompt, ...]
) -> None:
    """P7 fits A + B + A:B on these, which needs axis VALUES, not an opaque cell id."""
    rows, texts = score_control(config, lattice, prompts, NeutralScorer())

    assert len(rows) == len(texts)
    by_cell = {cell.cell_id: cell for cell in lattice.cells()}
    for row in rows:
        cell = by_cell[row.cell_id]
        assert row.order == cell.order
        assert (row.axis_a, row.value_a) == cell.axes[0]
        if cell.order == 2:
            assert (row.axis_b, row.value_b) == cell.axes[1]
        else:
            assert row.axis_b is None and row.value_b is None


def test_drift_rows_join_back_to_the_score_table(
    config: ScoringConfig, lattice: DeclaredLattice, prompts: tuple[Prompt, ...]
) -> None:
    """The join P7 performs, verified here so it is not discovered to be broken at P7."""
    rows, _ = score_control(config, lattice, prompts, NeutralScorer())
    assert len({row.score_id for row in rows}) == len(rows)
    assert all(row.response_id and row.control_family for row in rows)


def test_an_unknown_cell_is_refused(
    config: ScoringConfig, lattice: DeclaredLattice, prompts: tuple[Prompt, ...]
) -> None:
    """A drift artifact that cannot resolve a cell cannot be fitted on."""
    texts = build_control_texts(config, prompts, lattice=lattice, scoring_config_dir=FIXTURE_DIR)
    responses, control_prompts = control_responses(texts, decoding_hash="0" * 64)
    spec = next(s for s in config.scorers if s.scorer_id == "fake_toxic_nn").model_copy(
        update={"scorer_id": NeutralScorer.scorer_id}
    )
    scorer = ResponseScorer(
        spec,
        NeutralScorer(),  # type: ignore[arg-type]
        prompts_by_id=control_prompts,
        benchmark=BENCHMARK,
        scoring_hash="h",
        condition="drift_control",
    )
    spec_scores = scorer.score(responses)

    truncated: list[IdentityCell] = list(lattice.cells())[:1]
    with pytest.raises(KeyError, match="not in the lattice"):
        drift_rows(spec_scores, texts, truncated)


# ------------------------------------------------------------------ DETECTION (the gate)


def test_an_identity_sensitive_scorer_produces_a_non_zero_interaction(
    config: ScoringConfig, lattice: DeclaredLattice, prompts: tuple[Prompt, ...]
) -> None:
    """THE gate. If this fails, the control cannot catch a fabricated emergence result."""
    rows, _ = score_control(config, lattice, prompts, InteractionScorer())

    contrast = interaction_contrast(
        rows, axis_a=AXIS_A, value_a=VALUE_A, axis_b=AXIS_B, value_b=VALUE_B, field="raw"
    )
    assert abs(contrast) > 0.1, (
        f"the drift control failed to detect an injected interaction: {contrast}"
    )
    assert contrast == pytest.approx(0.5, abs=1e-6)  # exactly the injected bump


def test_a_neutral_scorer_produces_no_interaction(
    config: ScoringConfig, lattice: DeclaredLattice, prompts: tuple[Prompt, ...]
) -> None:
    rows, _ = score_control(config, lattice, prompts, NeutralScorer())

    contrast = interaction_contrast(
        rows, axis_a=AXIS_A, value_a=VALUE_A, axis_b=AXIS_B, value_b=VALUE_B, field="raw"
    )
    assert contrast == pytest.approx(0.0, abs=1e-9)


def test_a_main_effects_only_scorer_produces_no_interaction(
    config: ScoringConfig, lattice: DeclaredLattice, prompts: tuple[Prompt, ...]
) -> None:
    """Framing constraint 3, executable: a high LEVEL at the intersection is not emergence.

    This scorer's scores are highest exactly at the a1|b1 cell. A level-based check would flag it.
    The interaction contrast correctly reports ~0, because the elevation is fully additive.
    """
    rows, _ = score_control(config, lattice, prompts, MainEffectsScorer())

    intersection = [
        r for r in rows if r.order == 2 and {r.value_a, r.value_b} == {VALUE_A, VALUE_B}
    ]
    assert intersection, "fixture lattice must declare the alpha x beta pair"
    assert max(r.raw for r in intersection) == pytest.approx(0.8)  # the highest level in the table

    contrast = interaction_contrast(
        rows, axis_a=AXIS_A, value_a=VALUE_A, axis_b=AXIS_B, value_b=VALUE_B, field="raw"
    )
    assert contrast == pytest.approx(0.0, abs=1e-9)


def test_the_contrast_is_not_estimable_when_a_group_is_missing() -> None:
    """0.0, not NaN: a NaN would propagate into every downstream mean."""
    assert (
        interaction_contrast([], axis_a=AXIS_A, value_a=VALUE_A, axis_b=AXIS_B, value_b=VALUE_B)
        == 0.0
    )


def test_both_families_are_scored_and_can_be_compared_separately(
    config: ScoringConfig, lattice: DeclaredLattice, prompts: tuple[Prompt, ...]
) -> None:
    """If the two families disagree, that is a finding -- so each must be estimable alone."""
    rows, _ = score_control(config, lattice, prompts, InteractionScorer())

    for family in ("prompt_echo", "neutral_declarative"):
        subset = [r for r in rows if r.control_family == family]
        assert subset
        contrast = interaction_contrast(
            subset, axis_a=AXIS_A, value_a=VALUE_A, axis_b=AXIS_B, value_b=VALUE_B, field="raw"
        )
        assert contrast == pytest.approx(0.5, abs=1e-6)
