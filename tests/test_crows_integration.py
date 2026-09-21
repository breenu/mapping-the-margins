"""CrowS-Pairs records flow through P7 and P9 exactly as open_gen ones do (P2, scope item 5).

**This is the whole point of the shared record schema.** PROJECT_CONTEXT.md section 2 argues that
the "unified intersectional metric layer" is not an algorithm bolted on at the end -- it is the
record contract plus per-source standardization. If that is true, adding a second, maximally
different paradigm should require *no* change to the metric or stats layers, and a CrowS
``EffectEstimate`` should be corrected by exactly the same code path as a templated one. This file
is the test of that claim.

The records here are **synthetic**, deliberately. Measured on the pinned file, the real extractive
artifact reports every confirmatory pair NOT ESTIMABLE -- each CrowS item is annotated with one
attribute, so the order-2 grid is never populated -- which means it cannot exercise the correction
path at all. A gate that only passes when the data happens to cooperate is not a gate. The *real*
artifact's behaviour is asserted separately, as the excused path below.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from margins.lattice.build import DeclaredLattice
from margins.lattice.config import load_lattice_config
from margins.metrics.design import pair_designs
from margins.metrics.records import FitReport, FitTerm
from margins.schemas import EffectEstimate
from margins.stats.analysis import run_stats_analysis
from margins.stats.config import (
    DriftEnforcement,
    FdrSettings,
    PermutationSettings,
    PowerSettings,
    StatsConfig,
)
from margins.stats.family import build_family_plan, free_interaction_terms
from margins.stats.inputs import StatsInputs, check_effects_match_fits

LATTICE_FIXTURE = Path(__file__).parent / "fixtures" / "lattice"

OPEN_GEN = "open_gen_templates"
CROWS = "crows_pairs"
MODEL = "fake/model"
N_ITEMS = 60


@pytest.fixture(scope="module")
def lattice() -> DeclaredLattice:
    return DeclaredLattice(load_lattice_config("tiny", config_dir=LATTICE_FIXTURE))


def make_config(family_benchmarks: tuple[str, ...]) -> StatsConfig:
    return StatsConfig.model_validate(
        {
            "key": "test",
            "description": "test",
            "stats_version": "test@2026-09-20",
            "fdr": FdrSettings(
                alpha=0.05,
                prds_justification="stated for the test",
                family_benchmarks=family_benchmarks,
            ),
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
    )


def series_for(
    lattice: DeclaredLattice, benchmark: str, harm: str, *, estimable: bool = True
) -> tuple[list[EffectEstimate], list[FitTerm], list[FitReport]]:
    """One source's worth of P7 output, in the real record types.

    ``estimable=False`` produces what the real CrowS artifact produces: a not-estimable report and
    NO effects for that pair, which is P9's excuse path.
    """
    designs = pair_designs(lattice.cells())
    confirmatory = set(lattice.config.confirmatory_pairs)
    effects: list[EffectEstimate] = []
    terms: list[FitTerm] = []
    reports: list[FitReport] = []

    for design in designs:
        if design.axis_pair not in confirmatory:
            continue
        reports.append(
            FitReport(
                benchmark=benchmark,
                model_id=MODEL,
                harm_type=harm,
                condition="model",
                scorer_id=None,
                axis_a=design.axis_a,
                axis_b=design.axis_b,
                estimable=estimable,
                reason=None
                if estimable
                else f"axis pair {design.axis_pair} has empty cell(s): the grid is not identified",
                n_obs=N_ITEMS if estimable else 0,
                n_items=N_ITEMS if estimable else 0,
                n_cells=design.n_cells,
                n_params=design.n_params,
                cov_type="cluster",
                n_resamples=200,
                rms_interaction=0.4 if estimable else None,
                powered=estimable,
            )
        )
        if not estimable:
            continue
        for index, term in enumerate(free_interaction_terms(design)):
            estimate = 0.9 if index == 0 else 0.02 * index
            se = 0.12
            effects.append(
                EffectEstimate(
                    benchmark=benchmark,
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
                    benchmark=benchmark,
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
                    n_obs=N_ITEMS,
                    n_items=N_ITEMS,
                )
            )
    return effects, terms, reports


def inputs_for(*series: tuple[list, list, list]) -> StatsInputs:  # type: ignore[type-arg]
    effects: list[EffectEstimate] = []
    terms: list[FitTerm] = []
    reports: list[FitReport] = []
    for e, t, r in series:
        effects.extend(e)
        terms.extend(t)
        reports.extend(r)
    return StatsInputs(
        effects=effects,
        fit_terms=terms,
        fit_reports=reports,
        cell_stats=[],
        comparisons=[],
        harm_flags=[],
        paths={"effects": Path("synthetic")},
    )


def run(lattice: DeclaredLattice, inputs: StatsInputs, config: StatsConfig):  # type: ignore[no-untyped-def]
    return run_stats_analysis(
        inputs, pair_designs(lattice.cells()), lattice.config, config=config, min_cell_n=108
    )


# --------------------------------------------------------------------------------------
# A CrowS effect is corrected exactly like an open_gen one
# --------------------------------------------------------------------------------------


def test_a_crows_effect_joins_its_fit_term_like_any_other(lattice: DeclaredLattice) -> None:
    """`check_effects_match_fits` joins on (benchmark, model, harm, axis_a, axis_b, term).

    A mis-join would silently supply a wrong df and therefore a wrong q, which is why P9 asserts
    the join to 1e-12 rather than trusting it.
    """
    inputs = inputs_for(series_for(lattice, CROWS, "stereotype_preference"))
    check_effects_match_fits(inputs)


def test_crows_effects_are_corrected_by_the_same_code_path(lattice: DeclaredLattice) -> None:
    config = make_config((CROWS,))
    inputs = inputs_for(series_for(lattice, CROWS, "stereotype_preference"))
    result = run(lattice, inputs, config)

    corrected = [e for e in result.corrected if e.benchmark == CROWS]
    assert corrected, "CrowS effects must reach the correction stage"
    for effect in corrected:
        assert effect.corrected, "q, family_size and correction are stamped by StatsEngine only"
        assert effect.p is not None
        assert effect.family_size == result.family_size
        assert effect.correction == config.fdr.method


def test_the_two_paradigms_are_corrected_in_one_family(lattice: DeclaredLattice) -> None:
    """Section 8: the family is `benchmark x model x axis_pair x term x harm_type`, **never
    per-source**. Correcting each paradigm in its own family would be correcting a subset twice.
    """
    config = make_config((OPEN_GEN, CROWS))
    inputs = inputs_for(
        series_for(lattice, OPEN_GEN, "toxicity"),
        series_for(lattice, CROWS, "stereotype_preference"),
    )
    result = run(lattice, inputs, config)

    benchmarks = {e.benchmark for e in result.corrected}
    assert benchmarks == {OPEN_GEN, CROWS}
    sizes = {e.family_size for e in result.corrected}
    assert len(sizes) == 1, "one family, one size, stamped on every member of both paradigms"

    alone = run(
        lattice, inputs_for(series_for(lattice, OPEN_GEN, "toxicity")), make_config((OPEN_GEN,))
    )
    assert result.family_size > alone.family_size, (
        "adding a second source must GROW the family; if it did not, the BH threshold would be "
        "computed over fewer tests than were actually run"
    )


def test_adding_crows_never_loosens_the_threshold(lattice: DeclaredLattice) -> None:
    """The invariant that makes a growing family safe: more tests cannot mean more discoveries."""
    open_only = run(
        lattice, inputs_for(series_for(lattice, OPEN_GEN, "toxicity")), make_config((OPEN_GEN,))
    )
    both = run(
        lattice,
        inputs_for(
            series_for(lattice, OPEN_GEN, "toxicity"),
            series_for(lattice, CROWS, "stereotype_preference"),
        ),
        make_config((OPEN_GEN, CROWS)),
    )

    def significant(result: object, benchmark: str) -> int:  # type: ignore[no-untyped-def]
        return sum(
            1
            for e in result.corrected  # type: ignore[attr-defined]
            if e.benchmark == benchmark and e.q is not None and e.q <= 0.05
        )

    assert significant(both, OPEN_GEN) <= significant(open_only, OPEN_GEN)


# --------------------------------------------------------------------------------------
# The REAL CrowS behaviour: not estimable, and excused
# --------------------------------------------------------------------------------------


def test_a_not_estimable_crows_pair_is_excused_and_shrinks_the_family(
    lattice: DeclaredLattice,
) -> None:
    """What the real artifact does. Measured: every confirmatory pair has empty cells.

    A pair P7 reported not estimable is EXCUSED, carries P7's own reason, and shrinks the family.
    It must not raise -- that is reserved for a member that merely went missing.
    """
    config = make_config((OPEN_GEN, CROWS))
    inputs = inputs_for(
        series_for(lattice, OPEN_GEN, "toxicity"),
        series_for(lattice, CROWS, "stereotype_preference", estimable=False),
    )
    result = run(lattice, inputs, config)

    excused = [m for m in result.family if m.status == "excused_not_estimable"]
    assert excused, "the not-estimable CrowS pairs must be excused, not missing"
    assert all(m.benchmark == CROWS for m in excused)
    assert all(m.reason for m in excused), "the excuse carries P7's own reason"
    assert all(e.benchmark == OPEN_GEN for e in result.corrected)


# --------------------------------------------------------------------------------------
# The hole P2 closed: the family's benchmark dimension is DECLARED
# --------------------------------------------------------------------------------------


def test_a_declared_benchmark_that_never_arrives_raises(lattice: DeclaredLattice) -> None:
    """The hole this closes: `groups` used to be derived from the effects that arrived, so the
    expected set was built FROM the offered effects and could not notice a missing source.

    Running P9 on an open_gen-only directory would then quietly produce a smaller family and MORE
    discoveries -- a pre-registered `family_size` that is a function of which files are on disk.
    """
    config = make_config((OPEN_GEN, CROWS))
    inputs = inputs_for(series_for(lattice, OPEN_GEN, "toxicity"))
    with pytest.raises(ValueError, match="contains no fit report"):
        run(lattice, inputs, config)


def test_an_undeclared_benchmark_that_arrives_raises(lattice: DeclaredLattice) -> None:
    """The same inflation in the other direction: a source corrected inside a family that never
    announced it."""
    config = make_config((OPEN_GEN,))
    inputs = inputs_for(
        series_for(lattice, OPEN_GEN, "toxicity"),
        series_for(lattice, CROWS, "stereotype_preference"),
    )
    with pytest.raises(ValueError, match="does not declare"):
        run(lattice, inputs, config)


def test_an_empty_declaration_keeps_the_pre_p2_behaviour(lattice: DeclaredLattice) -> None:
    """Empty means 'infer', so a fixture config need not name sources it does not have."""
    config = make_config(())
    inputs = inputs_for(series_for(lattice, CROWS, "stereotype_preference"))
    result = run(lattice, inputs, config)
    assert {e.benchmark for e in result.corrected} == {CROWS}


def test_each_source_brings_its_own_harm_types(lattice: DeclaredLattice) -> None:
    """The family must not be a cartesian product of benchmarks and harm types.

    The two paradigms measure different things -- `toxicity`/`regard` against
    `stereotype_preference` -- so crossing every declared source with every observed harm type
    would invent an (open_gen_templates, stereotype_preference) test that no stage ever ran, and
    then raise because it never arrived.
    """
    plan = build_family_plan(
        pair_designs(lattice.cells()),
        lattice_config=lattice.config,
        groups=[(OPEN_GEN, MODEL, "toxicity"), (CROWS, MODEL, "stereotype_preference")],
        settings=make_config((OPEN_GEN, CROWS)).fdr,
    )
    pairs = {(m.benchmark, m.harm_type) for m in plan.members}
    assert pairs == {(OPEN_GEN, "toxicity"), (CROWS, "stereotype_preference")}


def test_the_shipped_config_declares_both_paradigms() -> None:
    """The pre-registration amendment, checked against the live config rather than the document."""
    from margins.stats.config import DEFAULT_STATS, load_stats_config

    config = load_stats_config(DEFAULT_STATS)
    assert set(config.fdr.family_benchmarks) == {OPEN_GEN, CROWS}


def _write_metrics_dir(
    out_dir: Path,
    effects: list[EffectEstimate],
    terms: list[FitTerm],
    reports: list[FitReport],
) -> Path:
    """The three P7 tables P9 reads, written the way `run_metrics.py` writes them."""
    from margins.metrics.export import (
        FIT_REPORTS_FILENAME,
        FIT_TERMS_FILENAME,
        write_effects,
        write_records,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_effects(effects, out_dir)
    write_records(terms, out_dir / FIT_TERMS_FILENAME, ["benchmark", "harm_type", "term"])
    write_records(reports, out_dir / FIT_REPORTS_FILENAME, ["benchmark", "harm_type", "axis_a"])
    return out_dir


# --------------------------------------------------------------------------------------
# Reading both paradigms into ONE family
# --------------------------------------------------------------------------------------


def test_both_paradigms_can_be_loaded_into_one_correction(
    lattice: DeclaredLattice, tmp_path: Path
) -> None:
    """Section 8: the family is corrected across the whole set, **never per-source**.

    `run_metrics.py` writes one directory per benchmark, so P9 has to read more than one -- or the
    two paradigms would be corrected in two families, which is correcting a subset twice.
    """
    from margins.stats.inputs import load_stats_inputs_many

    dirs = [
        _write_metrics_dir(tmp_path / benchmark, *series_for(lattice, benchmark, harm))
        for benchmark, harm in ((OPEN_GEN, "toxicity"), (CROWS, "stereotype_preference"))
    ]

    merged = load_stats_inputs_many(dirs)
    assert {e.benchmark for e in merged.effects} == {OPEN_GEN, CROWS}
    assert {b for b, _, _ in merged.attempted_groups} == {OPEN_GEN, CROWS}


def test_the_same_effect_in_two_directories_raises(
    lattice: DeclaredLattice, tmp_path: Path
) -> None:
    """Double-counting a test inflates the BH denominator's weighting as surely as a subset
    deflates it."""
    from margins.stats.inputs import load_stats_inputs_many

    dirs = [
        _write_metrics_dir(tmp_path / name, *series_for(lattice, CROWS, "stereotype_preference"))
        for name in ("a", "b")
    ]

    with pytest.raises(ValueError, match="appears in both"):
        load_stats_inputs_many(dirs)
