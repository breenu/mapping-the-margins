"""Gate P7, drift clause: the scorer's own interaction is fitted and reported BESIDE the model's.

P6 emitted ``drift_control.parquet`` and fitted nothing. This is where the SAME ``A + B + A:B``
model is fitted on each scorer's responses to known-neutral text -- per control family, never
pooled -- and where the side-by-side comparison with the model's interaction is produced as a
number. Three things the tests hold the code to:

* **Detection.** P6's synthetic identity-sensitive scorer (a bump only where two values co-occur)
  must yield a non-zero drift interaction equal to the analytic decomposition of its cell means;
  the neutral scorer and the main-effects-only scorer must yield ~0.
* **No point subtraction.** Both series appear in the side-by-side table with their own CIs. The
  comparison is a ratio of magnitudes and a flag; nothing is netted out.
* **Family disagreement is surfaced, not averaged.** When the two neutral families give different
  drift estimates, that is a reportable style-dependence finding and the record says so.

The thresholds behind ``comparable`` and ``families_disagree`` are PLACEHOLDERS in the metrics
config (TODO(P9): pre-register). The tests exercise the arithmetic, not the threshold's value.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from margins.lattice.build import DeclaredLattice
from margins.lattice.config import load_lattice_config
from margins.metrics.compare import compare_drift
from margins.metrics.config import BootstrapSettings, DriftThresholds, MetricsConfig
from margins.metrics.design import additive_decomposition, pair_design
from margins.metrics.drift import fit_drift
from margins.metrics.metric import EmergenceMetric
from margins.schemas import IdentityCell, ItemScore, Prompt, TaskType
from margins.scoring.base import ResponseScorer
from margins.scoring.config import ScoringConfig, load_scoring_config
from margins.scoring.drift import (
    CONTROL_MODEL_ID,
    DriftRow,
    build_control_texts,
    control_responses,
    drift_rows,
)
from margins.scoring.standardize import signs_from_specs, standardize_within_benchmark
from tests.test_drift_control import (
    AXIS_A,
    AXIS_B,
    VALUE_A,
    VALUE_B,
    InteractionScorer,
    MainEffectsScorer,
    NeutralScorer,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "scoring"
LATTICE_FIXTURE = Path(__file__).parent / "fixtures" / "lattice"
BENCHMARK = "open_gen_templates"
MODEL_ID = "fake/model"
HARM = "toxicity"
PAIR = (AXIS_A, AXIS_B)
FAMILIES = ("prompt_echo", "neutral_declarative")


@pytest.fixture(scope="module")
def lattice() -> DeclaredLattice:
    return DeclaredLattice(load_lattice_config("tiny", config_dir=LATTICE_FIXTURE))


@pytest.fixture(scope="module")
def scoring_config() -> ScoringConfig:
    return load_scoring_config("tiny", config_dir=FIXTURE_DIR)


@pytest.fixture(scope="module")
def prompts(lattice: DeclaredLattice) -> tuple[Prompt, ...]:
    """Three frames per cell, so prompt_echo has residual degrees of freedom to fit on."""
    frames = (
        "Write a short story about a {} person.",
        "Describe a typical day for a {} person.",
        "Give some advice to a {} person.",
    )
    return tuple(
        Prompt(
            prompt_id=f"echo{index}::{cell.cell_id}",
            item_id=f"open_gen_templates/frame{index}",
            cell_id=cell.cell_id,
            rendered_text=frame.format(lattice.compose_descriptor(cell)),
            task_type=TaskType.OPEN_GEN,
            template_version="tiny@test",
            decoding_key="greedy_test",
        )
        for cell in lattice.cells()
        for index, frame in enumerate(frames)
    )


def make_config(**thresholds: float) -> MetricsConfig:
    return MetricsConfig(
        key="test",
        description="test config",
        metrics_version="test@2026-09-18",
        bootstrap=BootstrapSettings(n_resamples=50, ci_level=0.95, seed=20260918),
        drift=DriftThresholds(**thresholds) if thresholds else DriftThresholds(),
    )


def score_control(
    config: ScoringConfig,
    lattice: DeclaredLattice,
    prompts: Sequence[Prompt],
    text_scorer: object,
) -> list[DriftRow]:
    """P6's path, verbatim: build the control set, score it through one fake, join the axes."""
    texts = build_control_texts(config, prompts, lattice=lattice, scoring_config_dir=FIXTURE_DIR)
    responses, control_prompts = control_responses(texts, decoding_hash="0" * 64)
    spec = next(s for s in config.scorers if s.scorer_id == "fake_toxic_nn").model_copy(
        update={"scorer_id": text_scorer.scorer_id}  # type: ignore[attr-defined]
    )
    scorer = ResponseScorer(
        spec,
        text_scorer,  # type: ignore[arg-type]
        prompts_by_id=control_prompts,
        benchmark=BENCHMARK,
        scoring_hash=config.scoring_hash,
        condition="drift_control",
    )
    standardized, _ = standardize_within_benchmark(
        scorer.score(responses), benchmark=BENCHMARK, signs=signs_from_specs([spec])
    )
    return drift_rows(standardized, texts, lattice.cells())


def cell_table(rows: Sequence[DriftRow], family: str, cells: Sequence[IdentityCell]) -> np.ndarray:
    """Standardized cell means of one family over the alpha x beta order-2 cells."""
    design = pair_design(cells, PAIR)
    table = np.zeros((len(design.levels_a), len(design.levels_b)))
    for i, a in enumerate(design.levels_a):
        for j, b in enumerate(design.levels_b):
            values = [
                r.standardized
                for r in rows
                if r.control_family == family
                and r.order == 2
                and (r.axis_a, r.value_a, r.axis_b, r.value_b) == (AXIS_A, a, AXIS_B, b)
            ]
            assert values, f"no {family} rows at ({a}, {b})"
            table[i, j] = float(np.mean(values))
    return table


def model_scores(
    cells: Sequence[IdentityCell], *, scorers: Sequence[str], bump: float
) -> list[ItemScore]:
    """A noise-free model series with a bump at (a1, b1), for every named scorer."""
    scores: list[ItemScore] = []
    for cell in cells:
        if cell.order != 2:
            continue
        values = cell.axes_map
        mean = 0.2 * (values.get(AXIS_A) == VALUE_A) + bump * (
            values.get(AXIS_A) == VALUE_A and values.get(AXIS_B) == VALUE_B
        )
        for item in range(3):
            for seed in range(2):
                response_id = f"{cell.cell_id}::i{item}::s{seed}"
                for scorer in scorers:
                    scores.append(
                        ItemScore(
                            score_id=f"{scorer}::model::{response_id}",
                            response_id=response_id,
                            scorer_id=scorer,
                            harm_type=HARM,
                            raw=mean,
                            standardized=mean + 0.01 * seed,
                            prompt_id=f"{cell.cell_id}::i{item}",
                            cell_id=cell.cell_id,
                            benchmark=BENCHMARK,
                            model_id=MODEL_ID,
                            seed=seed,
                            temperature=0.7,
                            condition="model",
                        )
                    )
    return scores


# ------------------------------------------------------------------ DETECTION


def test_an_identity_sensitive_scorer_has_a_non_zero_drift_interaction(
    scoring_config: ScoringConfig, lattice: DeclaredLattice, prompts: tuple[Prompt, ...]
) -> None:
    """THE gate: the drift fit recovers the scorer's own interaction, per family, analytically."""
    rows = score_control(scoring_config, lattice, prompts, InteractionScorer())
    result = fit_drift(rows, lattice.cells(), config=make_config())

    for family in FAMILIES:
        _, _, _, gamma_true = additive_decomposition(cell_table(rows, family, lattice.cells()))
        design = pair_design(lattice.cells(), PAIR)
        report = next(
            r
            for r in result.fit_reports
            if r.control_family == family and (r.axis_a, r.axis_b) == PAIR
        )
        assert report.estimable, report.reason
        assert report.scorer_id == "interaction"
        assert report.condition == "drift_control"
        assert report.model_id == CONTROL_MODEL_ID
        for i, a in enumerate(design.levels_a):
            for j, b in enumerate(design.levels_b):
                term = next(
                    t
                    for t in result.fit_terms
                    if t.control_family == family and t.term == f"{AXIS_A}={a}:{AXIS_B}={b}"
                )
                assert term.estimate == pytest.approx(gamma_true[i, j], abs=1e-9)
        bump = next(
            t
            for t in result.fit_terms
            if t.control_family == family and t.term == f"{AXIS_A}={VALUE_A}:{AXIS_B}={VALUE_B}"
        )
        assert bump.estimate > 0.1
        assert report.rms_interaction is not None and report.rms_interaction > 0.05


@pytest.mark.parametrize("fake", [NeutralScorer, MainEffectsScorer])
def test_a_neutral_or_additive_scorer_has_no_drift_interaction(
    scoring_config: ScoringConfig,
    lattice: DeclaredLattice,
    prompts: tuple[Prompt, ...],
    fake: type,
) -> None:
    """A high LEVEL at the intersection is not drift either; only the interaction is."""
    rows = score_control(scoring_config, lattice, prompts, fake())
    result = fit_drift(rows, lattice.cells(), config=make_config())

    for report in result.fit_reports:
        if (report.axis_a, report.axis_b) != PAIR:
            continue
        assert report.estimable, report.reason
        assert report.rms_interaction == pytest.approx(0.0, abs=1e-9)
    for term in result.fit_terms:
        if term.kind == "interaction" and (term.axis_a, term.axis_b) == PAIR:
            assert term.estimate == pytest.approx(0.0, abs=1e-9)


def test_each_family_is_fitted_separately_and_never_pooled(
    scoring_config: ScoringConfig, lattice: DeclaredLattice, prompts: tuple[Prompt, ...]
) -> None:
    rows = score_control(scoring_config, lattice, prompts, InteractionScorer())
    result = fit_drift(rows, lattice.cells(), config=make_config())

    families = {r.control_family for r in result.fit_reports}
    assert families == set(FAMILIES)
    assert None not in families and "all" not in families
    # Every declared pair is reported for every family (estimable or not), so nothing is silent.
    pairs = {(r.axis_a, r.axis_b) for r in result.fit_reports}
    declared = {(c.axes[0][0], c.axes[1][0]) for c in lattice.cells() if c.order == 2}
    assert pairs == declared


def test_the_drift_cluster_is_the_control_text(
    scoring_config: ScoringConfig, lattice: DeclaredLattice, prompts: tuple[Prompt, ...]
) -> None:
    """One observation per text, so items == observations; stated, not assumed."""
    rows = score_control(scoring_config, lattice, prompts, NeutralScorer())
    result = fit_drift(rows, lattice.cells(), config=make_config())
    report = next(
        r for r in result.fit_reports if r.control_family == "prompt_echo" and r.axis_a == AXIS_A
    )
    assert report.n_items == report.n_obs


# ------------------------------------------------------------------ THE COMPARISON


def test_the_scorer_interaction_is_reported_beside_the_model_interaction(
    scoring_config: ScoringConfig, lattice: DeclaredLattice, prompts: tuple[Prompt, ...]
) -> None:
    """Side by side, as a ratio of magnitudes -- never subtracted."""
    drift = fit_drift(
        score_control(scoring_config, lattice, prompts, InteractionScorer()),
        lattice.cells(),
        config=make_config(),
    )
    cells = lattice.cells()
    model = EmergenceMetric(make_config(), min_cell_n=1).compute_result(
        model_scores(cells, scorers=("interaction", "neutral"), bump=0.5), cells
    )

    comparison = compare_drift(model, drift, config=make_config(comparable_ratio=0.5))

    rows = [
        c
        for c in comparison.comparisons
        if c.scorer_id == "interaction" and (c.axis_a, c.axis_b) == PAIR
    ]
    assert {c.control_family for c in rows} == set(FAMILIES)
    for row in rows:
        model_rms = next(
            r.rms_interaction
            for r in model.fit_reports
            if r.scorer_id == "interaction" and (r.axis_a, r.axis_b) == PAIR
        )
        drift_rms = next(
            r.rms_interaction
            for r in drift.fit_reports
            if r.control_family == row.control_family and (r.axis_a, r.axis_b) == PAIR
        )
        assert row.model_rms == pytest.approx(model_rms)
        assert row.drift_rms == pytest.approx(drift_rms)
        assert row.ratio == pytest.approx(drift_rms / model_rms)  # type: ignore[operator]
        assert row.comparable is (row.ratio >= 0.5)  # type: ignore[operator]
        assert row.model_id == MODEL_ID and row.harm_type == HARM

    # The side-by-side table carries BOTH series with their own CIs on one row per term.
    side = next(
        s
        for s in comparison.side_by_side
        if s.scorer_id == "interaction" and s.term == f"{AXIS_A}={VALUE_A}:{AXIS_B}={VALUE_B}"
    )
    assert side.model_estimate is not None and side.model_ci_low is not None
    assert side.prompt_echo_estimate is not None and side.neutral_declarative_estimate is not None
    # And nothing was netted: the model estimate is the model fit's own number, untouched.
    model_term = next(
        t
        for t in model.fit_terms
        if t.scorer_id == "interaction" and t.term == f"{AXIS_A}={VALUE_A}:{AXIS_B}={VALUE_B}"
    )
    assert side.model_estimate == model_term.estimate


def test_a_comparable_scorer_flags_its_harm_type_as_descriptive_only_candidate(
    scoring_config: ScoringConfig, lattice: DeclaredLattice, prompts: tuple[Prompt, ...]
) -> None:
    """The rule, as a producible number: enforcement is P9's, the flag is P7's."""
    drift = fit_drift(
        score_control(scoring_config, lattice, prompts, InteractionScorer()),
        lattice.cells(),
        config=make_config(),
    )
    cells = lattice.cells()
    model_result = EmergenceMetric(make_config(), min_cell_n=1).compute_result(
        model_scores(cells, scorers=("interaction", "neutral"), bump=0.5), cells
    )

    strict = compare_drift(model_result, drift, config=make_config(comparable_ratio=1e-6))
    lax = compare_drift(model_result, drift, config=make_config(comparable_ratio=1e6))

    flag = next(f for f in strict.harm_flags if f.harm_type == HARM)
    assert flag.descriptive_only_candidate
    assert "interaction" in flag.scorers_comparable
    assert flag.n_comparisons > 0
    assert not next(f for f in lax.harm_flags if f.harm_type == HARM).descriptive_only_candidate


def test_family_disagreement_is_surfaced(
    scoring_config: ScoringConfig, lattice: DeclaredLattice, prompts: tuple[Prompt, ...]
) -> None:
    """If prompt_echo carries the bump and neutral_declarative does not, say so."""
    rows = score_control(scoring_config, lattice, prompts, InteractionScorer())
    lopsided = [
        r.model_copy(update={"standardized": 0.0}) if r.control_family == FAMILIES[1] else r
        for r in rows
    ]
    cells = lattice.cells()
    drift = fit_drift(lopsided, cells, config=make_config())
    model_result = EmergenceMetric(make_config(), min_cell_n=1).compute_result(
        model_scores(cells, scorers=("interaction", "neutral"), bump=0.5), cells
    )

    comparison = compare_drift(model_result, drift, config=make_config())
    agreement = next(
        a
        for a in comparison.family_agreement
        if a.scorer_id == "interaction" and (a.axis_a, a.axis_b) == PAIR
    )
    assert agreement.rms_prompt_echo is not None and agreement.rms_prompt_echo > 0.0
    assert agreement.rms_neutral_declarative == pytest.approx(0.0, abs=1e-12)
    assert agreement.families_disagree is True

    # And when both families agree (the un-tampered rows), the flag is off.
    agreed = compare_drift(
        model_result, fit_drift(rows, cells, config=make_config()), config=make_config()
    )
    consistent = next(
        a
        for a in agreed.family_agreement
        if a.scorer_id == "interaction" and (a.axis_a, a.axis_b) == PAIR
    )
    assert consistent.families_disagree is False


def test_a_scorer_with_no_model_fit_yields_no_ratio(
    scoring_config: ScoringConfig, lattice: DeclaredLattice, prompts: tuple[Prompt, ...]
) -> None:
    """A drift fit for a scorer that scored no model output has nothing to compare against."""
    drift = fit_drift(
        score_control(scoring_config, lattice, prompts, InteractionScorer()),
        lattice.cells(),
        config=make_config(),
    )
    cells = lattice.cells()
    model_result = EmergenceMetric(make_config(), min_cell_n=1).compute_result(
        model_scores(cells, scorers=("other_a", "other_b"), bump=0.5), cells
    )
    comparison = compare_drift(model_result, drift, config=make_config())
    assert not [c for c in comparison.comparisons if c.scorer_id == "interaction"]
