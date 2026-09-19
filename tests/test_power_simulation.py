"""Gate P9: the power simulation measures the real pipeline and invents nothing.

These tests were written before the simulation. The failure mode they exist to prevent is not a
crash -- it is a **plausible power number produced by a simulation that quietly measured a
different design than the one that will run**. Three ways that could happen, each with a test:

1. the simulation fits iid instead of clustered, and reports a design effect of 1 (spy test);
2. the injected effect is scaled before the sum-to-zero projection, so the reported minimum
   detectable effect is about a quantity nobody tests (projection tests);
3. the simulation treats seeds as if they added independent information, hiding the fact that a
   cell has a FIXED number of item clusters (the ceiling test).

The ceiling test is the important one and it is executable: at a fixed template-frame count, power
converges as seeds grow, so ``power(k=20)`` and ``power(k=100)`` must be close. If they are not,
either the clustering is being ignored or the variance model is wrong -- and either way the
rendering budget the analysis recommends would be fiction.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from margins.lattice.build import DeclaredLattice
from margins.lattice.config import load_lattice_config
from margins.metrics import estimator as estimator_module
from margins.metrics.design import PairDesign, additive_decomposition, pair_design
from margins.metrics.estimator import PairFit
from margins.stats.family import family_size, free_interaction_terms
from margins.stats.power import (
    PowerDesign,
    cell_mean_table,
    interaction_table,
    power_curve,
    simulate_frame,
    simulate_power_point,
    wilson_interval,
)

LATTICE_FIXTURE = Path(__file__).parent / "fixtures" / "lattice"
ALPHA = 0.05
PAIR = ("alpha", "beta")


@pytest.fixture(scope="module")
def design() -> PairDesign:
    """The 2 x 3 fixture pair. Small enough to simulate repeatedly in a test suite."""
    lattice = DeclaredLattice(load_lattice_config("tiny", config_dir=LATTICE_FIXTURE))
    return pair_design(lattice.cells(), PAIR)


@pytest.fixture(scope="module")
def real_design() -> PairDesign:
    """The shipped confirmatory 4 x 4 pair, used where the real arithmetic is the point."""
    lattice = DeclaredLattice(load_lattice_config("confirmatory_v1"))
    return pair_design(lattice.cells(), ("socioeconomic_class", "race_ethnicity"))


def rng(seed: int = 20260919) -> np.random.Generator:
    return np.random.default_rng(seed)


# ------------------------------------------------------------------ the injected truth


@pytest.mark.parametrize("pattern", ["concentrated", "diffuse"])
def test_injected_table_is_sum_to_zero(design: PairDesign, pattern: str) -> None:
    """A gamma table that does not project to zero margins is not an interaction."""
    gamma = interaction_table(design, effect_size=0.3, pattern=pattern)  # type: ignore[arg-type]
    assert gamma.sum(axis=0) == pytest.approx(np.zeros(gamma.shape[1]), abs=1e-12)
    assert gamma.sum(axis=1) == pytest.approx(np.zeros(gamma.shape[0]), abs=1e-12)


def test_concentrated_scales_the_largest_cell(design: PairDesign) -> None:
    gamma = interaction_table(design, effect_size=0.3, pattern="concentrated")
    assert float(np.abs(gamma).max()) == pytest.approx(0.3, abs=1e-12)
    # The maximum lands on the first cell, which is always inside the free contrast set.
    assert np.unravel_index(int(np.abs(gamma).argmax()), gamma.shape) == (0, 0)


def test_diffuse_scales_the_rms(design: PairDesign) -> None:
    gamma = interaction_table(design, effect_size=0.3, pattern="diffuse")
    assert float(np.sqrt(np.mean(gamma**2))) == pytest.approx(0.3, abs=1e-12)


def test_scaling_happens_after_projection(design: PairDesign) -> None:
    """Scaling a raw table first would leave max|gamma| BELOW the target after projection."""
    raw = np.zeros((len(design.levels_a), len(design.levels_b)))
    raw[0, 0] = 0.3
    _, _, _, naive = additive_decomposition(raw)
    assert float(np.abs(naive).max()) < 0.3  # the projection shrinks it
    correct = interaction_table(design, effect_size=0.3, pattern="concentrated")
    assert float(np.abs(correct).max()) == pytest.approx(0.3, abs=1e-12)


def test_cell_means_recover_the_injected_gamma(design: PairDesign) -> None:
    """The truth is an answer key: decomposing the mean table returns what was injected."""
    gamma = interaction_table(design, effect_size=0.3, pattern="concentrated")
    mu, _, _, recovered = additive_decomposition(cell_mean_table(design, gamma))
    assert mu == pytest.approx(0.0, abs=1e-12)
    assert recovered == pytest.approx(gamma, abs=1e-12)


# ------------------------------------------------------------------ the simulated data


def test_total_variance_is_one_on_the_standardized_scale(design: PairDesign) -> None:
    """Scores are z-scored, so effect_size can only be read as SD units if Var(y) == 1."""
    power_design = PowerDesign(design=design, frames_per_cell=40, seeds_per_item=40)
    zero = np.zeros((len(design.levels_a), len(design.levels_b)))
    frame = simulate_frame(power_design, zero, icc_rho=0.3, rng=rng())
    assert float(frame.y.var()) == pytest.approx(1.0, abs=0.05)


def test_empirical_icc_matches_the_assumption(design: PairDesign) -> None:
    """The between-item share of variance must be the rho that was asked for."""
    power_design = PowerDesign(design=design, frames_per_cell=60, seeds_per_item=30)
    zero = np.zeros((len(design.levels_a), len(design.levels_b)))
    frame = simulate_frame(power_design, zero, icc_rho=0.3, rng=rng())
    by_item: dict[str, list[float]] = {}
    for value, item in zip(frame.y, frame.item, strict=True):
        by_item.setdefault(item, []).append(float(value))
    item_means = np.array([np.mean(v) for v in by_item.values()])
    within = np.mean([np.var(v) for v in by_item.values()])
    assert float(item_means.var()) == pytest.approx(0.3, abs=0.05)
    assert float(within) == pytest.approx(0.7, abs=0.05)


def test_seeds_of_one_item_share_a_cluster(design: PairDesign) -> None:
    """Whole items must move together; an item that spans cells could not be clustered."""
    power_design = PowerDesign(design=design, frames_per_cell=4, seeds_per_item=5)
    gamma = interaction_table(design, effect_size=0.3, pattern="concentrated")
    frame = simulate_frame(power_design, cell_mean_table(design, gamma), icc_rho=0.3, rng=rng())
    cells_of_item: dict[str, set[tuple[str, str]]] = {}
    for item, a, b in zip(frame.item, frame.value_a, frame.value_b, strict=True):
        cells_of_item.setdefault(item, set()).add((a, b))
    assert all(len(cells) == 1 for cells in cells_of_item.values())
    assert len(cells_of_item) == 4 * design.n_cells
    assert frame.n_obs == 4 * 5 * design.n_cells


def test_items_per_cell_does_not_move_with_seeds(design: PairDesign) -> None:
    """The fact the whole analysis turns on, asserted rather than commented."""
    few = PowerDesign(design=design, frames_per_cell=9, seeds_per_item=2)
    many = PowerDesign(design=design, frames_per_cell=9, seeds_per_item=40)
    assert few.items_per_cell == many.items_per_cell == 9
    assert few.n_item_clusters == many.n_item_clusters
    assert few.renderings_per_cell == 18
    assert many.renderings_per_cell == 360


# ------------------------------------------------------------------ the estimator it calls


def test_simulation_uses_the_real_clustered_estimator(
    design: PairDesign, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A simulation that quietly fitted iid would report fantasy power. Spy on the call."""
    calls: list[dict[str, object]] = []
    real = estimator_module.estimate_pair

    def spy(frame: object, pair: object, **kwargs: object) -> PairFit:
        calls.append(kwargs)
        return real(frame, pair, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("margins.stats.power.estimate_pair", spy)
    simulate_power_point(
        PowerDesign(design=design, frames_per_cell=4, seeds_per_item=2),
        pattern="concentrated",
        effect_size=0.3,
        icc_rho=0.3,
        alpha=ALPHA,
        method="fdr_bh",
        family_size=12,
        n_simulations=3,
        rng=rng(),
    )
    assert len(calls) == 3
    assert all(call["cov_type"] == "cluster" for call in calls)
    assert all(call["n_resamples"] == 0 for call in calls)
    assert all(call["rng"] is None for call in calls)


def test_degrees_of_freedom_come_from_clusters_not_observations(design: PairDesign) -> None:
    """Mean SE must barely move between 6 and 60 seeds; with iid df it would fall by sqrt(10)."""
    common = {
        "pattern": "concentrated",
        "effect_size": 0.3,
        "icc_rho": 0.5,
        "alpha": ALPHA,
        "method": "fdr_bh",
        "family_size": 12,
        "n_simulations": 40,
    }
    few = simulate_power_point(
        PowerDesign(design=design, frames_per_cell=12, seeds_per_item=6),
        rng=rng(),
        **common,  # type: ignore[arg-type]
    )
    many = simulate_power_point(
        PowerDesign(design=design, frames_per_cell=12, seeds_per_item=60),
        rng=rng(),
        **common,  # type: ignore[arg-type]
    )
    assert many.mean_se / few.mean_se > 0.8
    assert few.n_item_clusters == many.n_item_clusters


# ------------------------------------------------------------------ the curve


def test_power_rises_with_seeds_and_with_effect_size(design: PairDesign) -> None:
    common = {
        "pattern": "concentrated",
        "icc_rho": 0.2,
        "alpha": ALPHA,
        "method": "fdr_bh",
        "family_size": 12,
        "n_simulations": 200,
    }
    small = simulate_power_point(
        PowerDesign(design=design, frames_per_cell=12, seeds_per_item=1),
        effect_size=0.3,
        rng=rng(),
        **common,  # type: ignore[arg-type]
    )
    more_seeds = simulate_power_point(
        PowerDesign(design=design, frames_per_cell=12, seeds_per_item=8),
        effect_size=0.3,
        rng=rng(),
        **common,  # type: ignore[arg-type]
    )
    bigger = simulate_power_point(
        PowerDesign(design=design, frames_per_cell=12, seeds_per_item=1),
        effect_size=0.9,
        rng=rng(),
        **common,  # type: ignore[arg-type]
    )
    assert more_seeds.power >= small.power
    assert bigger.power > small.power


def test_seeds_hit_a_ceiling_the_cluster_count_sets(design: PairDesign) -> None:
    """THE structural finding, made executable: past some k, more generation buys nothing."""
    common = {
        "pattern": "concentrated",
        "effect_size": 0.3,
        "icc_rho": 0.4,
        "alpha": ALPHA,
        "method": "fdr_bh",
        "family_size": 12,
        "n_simulations": 200,
    }
    at_20 = simulate_power_point(
        PowerDesign(design=design, frames_per_cell=9, seeds_per_item=20),
        rng=rng(),
        **common,  # type: ignore[arg-type]
    )
    at_100 = simulate_power_point(
        PowerDesign(design=design, frames_per_cell=9, seeds_per_item=100),
        rng=rng(),
        **common,  # type: ignore[arg-type]
    )
    assert abs(at_100.power - at_20.power) < 0.05


def test_limiting_point_is_the_ceiling(design: PairDesign) -> None:
    """The seeds -> infinity design is simulated directly, so the ceiling is measured."""
    common = {
        "pattern": "concentrated",
        "effect_size": 0.3,
        "icc_rho": 0.4,
        "alpha": ALPHA,
        "method": "fdr_bh",
        "family_size": 12,
        "n_simulations": 200,
    }
    at_40 = simulate_power_point(
        PowerDesign(design=design, frames_per_cell=9, seeds_per_item=40),
        rng=rng(),
        **common,  # type: ignore[arg-type]
    )
    limit = simulate_power_point(
        PowerDesign(design=design, frames_per_cell=9, seeds_per_item=1),
        rng=rng(),
        limiting=True,
        **common,  # type: ignore[arg-type]
    )
    assert limit.limiting is True
    assert limit.power >= at_40.power - 0.06


def test_more_frames_raise_the_ceiling(design: PairDesign) -> None:
    """The only lever that beats the ceiling is clusters, i.e. template frames."""
    common = {
        "pattern": "concentrated",
        "effect_size": 0.3,
        "icc_rho": 0.4,
        "alpha": ALPHA,
        "method": "fdr_bh",
        "family_size": 12,
        "n_simulations": 200,
        "limiting": True,
    }
    nine = simulate_power_point(
        PowerDesign(design=design, frames_per_cell=9, seeds_per_item=1),
        rng=rng(),
        **common,  # type: ignore[arg-type]
    )
    many = simulate_power_point(
        PowerDesign(design=design, frames_per_cell=90, seeds_per_item=1),
        rng=rng(),
        **common,  # type: ignore[arg-type]
    )
    assert many.power > nine.power


def test_correction_costs_power(design: PairDesign) -> None:
    """A family of one is the uncorrected test; the real family must be no easier."""
    common = {
        "pattern": "concentrated",
        "effect_size": 0.3,
        "icc_rho": 0.2,
        "alpha": ALPHA,
        "method": "fdr_bh",
        "n_simulations": 200,
    }
    pair_terms = len(free_interaction_terms(design))
    alone = simulate_power_point(
        PowerDesign(design=design, frames_per_cell=12, seeds_per_item=4),
        family_size=pair_terms,
        rng=rng(),
        **common,  # type: ignore[arg-type]
    )
    corrected = simulate_power_point(
        PowerDesign(design=design, frames_per_cell=12, seeds_per_item=4),
        family_size=168,
        rng=rng(),
        **common,  # type: ignore[arg-type]
    )
    assert alone.power >= corrected.power


def test_family_smaller_than_the_pair_is_refused(design: PairDesign) -> None:
    with pytest.raises(ValueError, match="smaller than the simulated pair"):
        simulate_power_point(
            PowerDesign(design=design, frames_per_cell=4, seeds_per_item=2),
            pattern="concentrated",
            effect_size=0.3,
            icc_rho=0.2,
            alpha=ALPHA,
            method="fdr_bh",
            family_size=1,
            n_simulations=2,
            rng=rng(),
        )


# ------------------------------------------------------------------ the reported minimum


def _curve(design: PairDesign, *, seeds: Sequence[int], effect: float, rho: float, sims: int):  # type: ignore[no-untyped-def]
    return power_curve(
        design,
        frames_per_cell=12,
        seeds_grid=seeds,
        pattern="concentrated",
        effect_size=effect,
        icc_rho=rho,
        alpha=ALPHA,
        method="fdr_bh",
        family_size=12,
        target_power=0.8,
        n_simulations=sims,
        rng=rng(),
    )


def test_reported_minimum_is_self_consistent(design: PairDesign) -> None:
    """Re-simulating at the reported minimum, with a fresh seed, must reach the target."""
    result = _curve(design, seeds=(1, 2, 4, 8, 16), effect=1.2, rho=0.2, sims=300)
    assert result.min_seeds_per_item is not None
    replay = simulate_power_point(
        PowerDesign(design=design, frames_per_cell=12, seeds_per_item=result.min_seeds_per_item),
        pattern="concentrated",
        effect_size=1.2,
        icc_rho=0.2,
        alpha=ALPHA,
        method="fdr_bh",
        family_size=12,
        n_simulations=300,
        rng=rng(4242),
    )
    assert replay.power >= 0.8 - 0.05


def test_minimum_is_one_decision_with_min_cell_n(design: PairDesign) -> None:
    """A rendering budget and the threshold it must clear are ONE number (section 8)."""
    result = _curve(design, seeds=(1, 2, 4, 8, 16), effect=1.2, rho=0.2, sims=200)
    assert result.min_renderings_per_cell == 12 * (result.min_seeds_per_item or 0)
    assert result.min_cell_n == result.min_renderings_per_cell


def test_an_unreachable_target_invents_no_number(design: PairDesign) -> None:
    """The ceiling below target must produce None plus a stated reason, never an extrapolation."""
    result = _curve(design, seeds=(1, 2, 4), effect=0.02, rho=0.5, sims=200)
    assert result.min_seeds_per_item is None
    assert result.min_renderings_per_cell is None
    assert result.min_cell_n is None
    assert result.reason is not None
    assert "template frames" in result.reason
    assert result.power_ceiling < 0.8


def test_bisection_finds_the_same_minimum_as_the_linear_walk(design: PairDesign) -> None:
    """The speedup must be a cost decision, not an approximation.

    Bisection is only valid because power is monotone in seeds. If the two searches ever disagree,
    either the monotonicity assumption or the search is wrong, and the reported budget is wrong
    with it.
    """
    grid = (1, 2, 3, 4, 6, 8, 12, 18, 24)
    common = {
        "frames_per_cell": 6,
        "seeds_grid": grid,
        "pattern": "concentrated",
        "effect_size": 0.8,
        "icc_rho": 0.1,
        "alpha": ALPHA,
        "method": "fdr_bh",
        "family_size": 12,
        "target_power": 0.8,
        "n_simulations": 400,
    }
    linear = power_curve(design, search="linear", rng=rng(11), **common)  # type: ignore[arg-type]
    bisect = power_curve(design, search="bisect", rng=rng(11), **common)  # type: ignore[arg-type]
    assert linear.min_seeds_per_item is not None
    assert bisect.min_seeds_per_item == linear.min_seeds_per_item

    # And the cost is BOUNDED, which is the reason it exists. A linear walk can touch every grid
    # point, and cost grows linearly in seeds, so the tail of the grid dominates a whole run.
    probes = [p for p in bisect.curve if not p.limiting]
    assert len(probes) <= math.ceil(math.log2(len(grid))) + 2
    assert max(p.seeds_per_item for p in probes) <= max(grid)


def test_bisection_confirms_at_full_precision(design: PairDesign) -> None:
    """A coarse probe may search; only a full-precision point may be reported."""
    result = power_curve(
        design,
        frames_per_cell=12,
        seeds_grid=(1, 2, 4, 8, 16),
        pattern="concentrated",
        effect_size=1.2,
        icc_rho=0.2,
        alpha=ALPHA,
        method="fdr_bh",
        family_size=12,
        target_power=0.8,
        n_simulations=400,
        search_simulations=80,
        rng=rng(),
    )
    assert result.min_seeds_per_item is not None
    reported = [
        p for p in result.curve if p.seeds_per_item == result.min_seeds_per_item and not p.limiting
    ]
    assert any(p.n_simulations == 400 and p.power >= 0.8 for p in reported)


def test_curve_is_deterministic_at_a_fixed_seed(design: PairDesign) -> None:
    first = _curve(design, seeds=(1, 2, 4), effect=0.6, rho=0.2, sims=100)
    second = _curve(design, seeds=(1, 2, 4), effect=0.6, rho=0.2, sims=100)
    assert [p.power for p in first.curve] == [p.power for p in second.curve]
    assert first.min_seeds_per_item == second.min_seeds_per_item


# ------------------------------------------------------------------ arithmetic and plumbing


def test_the_shipped_family_size_is_168(real_design: PairDesign) -> None:
    """The decided family, computed from the real lattice rather than asserted as a constant."""
    from margins.metrics.design import pair_designs

    lattice = DeclaredLattice(load_lattice_config("confirmatory_v1"))
    config = lattice.config
    confirmatory = set(config.confirmatory_pairs)
    designs = [d for d in pair_designs(lattice.cells()) if d.axis_pair in confirmatory]
    assert len(designs) == 6
    # 2 models x 2 harm types.
    assert family_size(designs, family_terms="free_interaction", n_groups=4) == 168
    assert family_size(designs, family_terms="all_interaction", n_groups=4) == 320
    assert family_size(designs, family_terms="all_terms", n_groups=4) == 496


def test_free_contrasts_exclude_the_implied_level(real_design: PairDesign) -> None:
    terms = free_interaction_terms(real_design)
    assert len(terms) == (len(real_design.levels_a) - 1) * (len(real_design.levels_b) - 1) == 9
    assert real_design.levels_a[-1] not in " ".join(terms).replace("=", " ").split()


@pytest.mark.parametrize(("successes", "trials"), [(0, 50), (50, 50), (25, 50)])
def test_wilson_interval_stays_inside_the_unit_interval(successes: int, trials: int) -> None:
    low, high = wilson_interval(successes, trials)
    assert 0.0 <= low <= high <= 1.0


def test_power_design_refuses_a_single_cluster(design: PairDesign) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        PowerDesign(design=design, frames_per_cell=1, seeds_per_item=10)
