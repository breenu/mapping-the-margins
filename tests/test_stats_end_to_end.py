"""Gate P9: the whole stage runs, on records shaped exactly like P7's, and round-trips.

The real dev-scale P7 artifact contains no estimable pair, so it cannot exercise the correction
path at all. This builds a synthetic ``StatsInputs`` with the same record types P7 writes -- real
``EffectEstimate``, ``FitTerm``, ``FitReport`` and ``DriftComparison`` objects, not mocks -- and
runs the stage end to end, so the join, the p-values, the correction, the enforcement, the gate
and the export are all covered by one test that would fail on any of them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from margins.lattice.build import DeclaredLattice
from margins.lattice.config import load_lattice_config
from margins.metrics.design import pair_designs
from margins.metrics.records import DriftComparison, FitReport, FitTerm
from margins.schemas import EffectEstimate
from margins.stats.analysis import run_stats_analysis
from margins.stats.config import (
    DriftEnforcement,
    FdrSettings,
    PermutationSettings,
    PowerSettings,
    StatsConfig,
)
from margins.stats.export import EFFECTS_CORRECTED_FILENAME, export_stats, read_corrected_effects
from margins.stats.family import free_interaction_terms
from margins.stats.inputs import StatsInputs, check_effects_match_fits
from margins.stats.manifest import stats_record

LATTICE_FIXTURE = Path(__file__).parent / "fixtures" / "lattice"
BENCHMARK = "open_gen_templates"
MODEL = "fake/model"
HARMS = ("toxicity", "regard")
N_ITEMS = 60


@pytest.fixture(scope="module")
def lattice() -> DeclaredLattice:
    return DeclaredLattice(load_lattice_config("tiny", config_dir=LATTICE_FIXTURE))


def make_config(**overrides: object) -> StatsConfig:
    data: dict[str, object] = {
        "key": "test",
        "description": "test",
        "stats_version": "test@2026-09-19",
        "fdr": FdrSettings(alpha=0.05, prds_justification="stated for the test"),
        "permutation": PermutationSettings(n_permutations=199, seed=1),
        "power": PowerSettings(
            frames_per_cell=27,
            frames_grid=(27,),
            seeds_grid=(1, 4),
            icc_rho=(0.18, 0.35),
            effect_size=0.5,
            patterns=("concentrated",),
            target_power=0.8,
            n_simulations=100,
            pair=("alpha", "beta"),
            seed=1,
        ),
        "drift": DriftEnforcement(descriptive_only_ratio=0.5),
    }
    data.update(overrides)
    return StatsConfig.model_validate(data)


def build_inputs(lattice: DeclaredLattice, *, drift_rms: dict[str, float]) -> StatsInputs:
    """Effects, their matching fit terms and reports, and a drift comparison per harm type."""
    designs = pair_designs(lattice.cells())
    confirmatory = set(lattice.config.confirmatory_pairs)
    effects: list[EffectEstimate] = []
    terms: list[FitTerm] = []
    reports: list[FitReport] = []
    comparisons: list[DriftComparison] = []

    for harm_index, harm in enumerate(HARMS):
        for design in designs:
            if design.axis_pair not in confirmatory:
                continue
            reports.append(
                FitReport(
                    benchmark=BENCHMARK,
                    model_id=MODEL,
                    harm_type=harm,
                    condition="model",
                    scorer_id=None,
                    axis_a=design.axis_a,
                    axis_b=design.axis_b,
                    estimable=True,
                    n_obs=N_ITEMS * 4,
                    n_items=N_ITEMS,
                    n_cells=design.n_cells,
                    n_params=design.n_params,
                    cov_type="cluster",
                    n_resamples=200,
                    rms_interaction=0.4,
                    powered=True,
                )
            )
            for index, term in enumerate(free_interaction_terms(design)):
                # A couple of genuinely large effects per harm type, the rest near zero.
                estimate = 0.9 if index == 0 else 0.02 * (index + harm_index)
                se = 0.12
                effects.append(
                    EffectEstimate(
                        benchmark=BENCHMARK,
                        model=MODEL,
                        axis_pair=design.axis_pair,
                        term=term,
                        harm_type=harm,
                        estimate=estimate,
                        se=se,
                        ci_low=estimate - 2 * se,
                        ci_high=estimate + 2 * se,
                        powered=True,
                    )
                )
                terms.append(
                    FitTerm(
                        benchmark=BENCHMARK,
                        model_id=MODEL,
                        harm_type=harm,
                        condition="model",
                        scorer_id=None,
                        axis_a=design.axis_a,
                        axis_b=design.axis_b,
                        term=term,
                        kind="interaction",
                        estimate=estimate,
                        se=se,
                        ci_low=estimate - 2 * se,
                        ci_high=estimate + 2 * se,
                        n_obs=N_ITEMS * 4,
                        n_items=N_ITEMS,
                    )
                )
        comparisons.append(
            DriftComparison(
                benchmark=BENCHMARK,
                model_id=MODEL,
                harm_type=harm,
                scorer_id=f"{harm}_neural",
                axis_a="alpha",
                axis_b="beta",
                control_family="prompt_echo",
                model_rms=1.0,
                drift_rms=drift_rms[harm],
                ratio=drift_rms[harm],
                comparable=False,
                n_model_obs=240,
                n_drift_obs=60,
            )
        )
    return StatsInputs(
        effects=effects,
        fit_terms=terms,
        fit_reports=reports,
        cell_stats=[],
        comparisons=comparisons,
        harm_flags=[],
        paths={"effects": Path("synthetic")},
    )


def run(lattice: DeclaredLattice, *, drift_rms: dict[str, float]):  # type: ignore[no-untyped-def]
    config = make_config()
    inputs = build_inputs(lattice, drift_rms=drift_rms)
    result = run_stats_analysis(
        inputs, pair_designs(lattice.cells()), lattice.config, config=config, min_cell_n=108
    )
    return config, inputs, result


# ------------------------------------------------------------------ the whole stage


def test_the_stage_runs_and_corrects(lattice: DeclaredLattice) -> None:
    _, inputs, result = run(lattice, drift_rms={"toxicity": 0.1, "regard": 0.1})
    assert result.family_size == len(inputs.effects)
    assert result.n_significant_pre >= result.n_significant_post > 0
    assert all(e.q is not None for e in result.corrected)
    assert {e.family_size for e in result.corrected} == {result.family_size}


def test_the_join_is_checked(lattice: DeclaredLattice) -> None:
    """A fit term that disagrees with its effect must stop the run, not feed a wrong df."""
    inputs = build_inputs(lattice, drift_rms={"toxicity": 0.1, "regard": 0.1})
    check_effects_match_fits(inputs)
    inputs.fit_terms[0] = inputs.fit_terms[0].model_copy(update={"se": 0.99})
    with pytest.raises(ValueError, match="do not match their fit terms"):
        check_effects_match_fits(inputs)


def test_a_high_drift_harm_type_loses_every_claim(lattice: DeclaredLattice) -> None:
    """The enforced consequence, end to end: the q survives, the claim does not."""
    _, _, result = run(lattice, drift_rms={"toxicity": 0.9, "regard": 0.1})
    assert result.descriptive_only_harm_types == ("toxicity",)
    toxic = [c for c in result.claims if c.harm_type == "toxicity"]
    regard = [c for c in result.claims if c.harm_type == "regard"]
    assert toxic and not any(c.claimable for c in toxic)
    assert all(c.status == "descriptive_only" for c in toxic)
    assert all(c.q is not None for c in toxic), "the number survives; only the claim is refused"
    assert any(c.claimable for c in regard)


def test_exploratory_pairs_never_reach_the_family(lattice: DeclaredLattice) -> None:
    _, _inputs, result = run(lattice, drift_rms={"toxicity": 0.1, "regard": 0.1})
    exploratory = set(lattice.config.exploratory_pairs)
    assert exploratory
    assert not any((m.axis_a, m.axis_b) in exploratory for m in result.family), (
        "an exploratory pair entered the FDR family"
    )


def test_a_significance_claim_requires_power(lattice: DeclaredLattice) -> None:
    config = make_config()
    inputs = build_inputs(lattice, drift_rms={"toxicity": 0.1, "regard": 0.1})
    inputs.effects[:] = [e.model_copy(update={"powered": False}) for e in inputs.effects]
    result = run_stats_analysis(
        inputs, pair_designs(lattice.cells()), lattice.config, config=config, min_cell_n=108
    )
    assert result.n_claimable == 0
    assert all(c.status == "underpowered" for c in result.claims)


# ------------------------------------------------------------------ export and manifest


def test_corrected_effects_round_trip_with_their_stamp(
    lattice: DeclaredLattice, tmp_path: Path
) -> None:
    """The mirror of P7's null test: q, family_size and correction must come back PRESENT."""
    config, _, result = run(lattice, drift_rms={"toxicity": 0.1, "regard": 0.1})
    paths = export_stats(result, config=config, out_dir=tmp_path)
    reloaded = read_corrected_effects(tmp_path / EFFECTS_CORRECTED_FILENAME)
    assert len(reloaded) == len(result.corrected)
    assert all(e.q is not None for e in reloaded)
    assert all(e.family_size == result.family_size for e in reloaded)
    assert all(e.correction == "fdr_bh" for e in reloaded)
    assert set(paths) == {
        "effects_corrected",
        "family",
        "descriptive_only",
        "claims",
        "fdr_report_md",
    }


def test_the_markdown_report_is_byte_deterministic(
    lattice: DeclaredLattice, tmp_path: Path
) -> None:
    config, _, result = run(lattice, drift_rms={"toxicity": 0.9, "regard": 0.1})
    first = export_stats(result, config=config, out_dir=tmp_path / "a")["fdr_report_md"]
    second = export_stats(result, config=config, out_dir=tmp_path / "b")["fdr_report_md"]
    assert first.read_bytes() == second.read_bytes()
    text = first.read_text(encoding="utf-8")
    # The downgrade must be legible in the artifact, both as a verdict and per claim.
    assert "descriptive-only" in text.lower()
    assert "**YES**" in text, "the downgraded harm type is not marked in the decision table"
    assert "descriptive_only" in text, "refused claims must carry their status in the table"
    assert "no significance claim" in text


def test_the_manifest_block_states_the_estimand(lattice: DeclaredLattice) -> None:
    config, _, result = run(lattice, drift_rms={"toxicity": 0.9, "regard": 0.1})
    record = stats_record(config, result, stats_name="test", benchmark=BENCHMARK, min_cell_n=108)
    assert "fdr_bh" in record.correction_method
    assert "CONFIRMATORY" in record.family_definition
    assert "cluster-robust Wald" in record.p_source
    assert "item clusters - 1" in record.p_source
    assert record.family_size == result.family_size
    assert record.descriptive_only_harm_types == ("toxicity",)
    assert record.min_cell_n == 108
    assert record.effect_size == 0.5
