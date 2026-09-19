"""Gate: the rho estimator recovers a known variance split and refuses a design that cannot.

Everything downstream of rho is a design decision -- how many template frames the sampled E1 arm
needs, whether 0.3 SD is reachable at all, what ``min_cell_n`` becomes. A rho biased by 0.1 would
move the recommended frame count by a factor of two, so these tests check against injected truth
rather than against a number the estimator once produced.

The two mistakes that would matter most, each with a test:

1. **Pooling between-cell variance into rho.** Prompts in different cells differ because the
   cells differ, and that is signal the model fits, not clustering noise. An estimator that
   ignored the cell factor would report a much larger rho on data with real cell effects, and
   would condemn a design that is actually fine.
2. **A normal-approximation interval.** rho is bounded and skewed near 0, which is exactly where
   the decision boundary sits, so coverage is asserted over repeated datasets rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from margins.stats.icc import IccObservation, estimate_icc, icc_interval

CELLS = 16
FRAMES = 9
SEEDS = 12


def synthetic(
    *,
    rho: float,
    n_cells: int = CELLS,
    n_frames: int = FRAMES,
    n_seeds: int = SEEDS,
    cell_spread: float = 0.0,
    rng: np.random.Generator,
) -> list[IccObservation]:
    """``y = mu_cell + u_item + e`` with total within-cell variance 1 and a known rho.

    ``cell_spread`` adds a fixed between-CELL effect. It must not change rho: that variation is
    the model's signal, and the estimator treats cells as a fixed factor for exactly that reason.
    """
    sd_item = float(np.sqrt(rho))
    sd_noise = float(np.sqrt(1.0 - rho))
    cell_means = np.linspace(-cell_spread, cell_spread, n_cells)
    out: list[IccObservation] = []
    for cell_index in range(n_cells):
        for frame in range(n_frames):
            u = rng.normal(0.0, sd_item) if sd_item > 0 else 0.0
            for _ in range(n_seeds):
                out.append(
                    IccObservation(
                        cell_id=f"cell{cell_index}",
                        item_id=f"cell{cell_index}::frame{frame}",
                        value=float(cell_means[cell_index] + u + rng.normal(0.0, sd_noise)),
                    )
                )
    return out


def measure(rho: float, *, seed: int = 0, **kwargs: object) -> float:
    rng = np.random.default_rng(seed)
    estimate = estimate_icc(
        synthetic(rho=rho, rng=rng, **kwargs),  # type: ignore[arg-type]
        harm_type="toxicity",
    )
    assert estimate.estimable and estimate.rho is not None
    return estimate.rho


# ------------------------------------------------------------------ recovery


@pytest.mark.parametrize("true_rho", [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8])
def test_recovers_the_injected_rho_on_average(true_rho: float) -> None:
    """Asserted over repeated datasets, never one seed: a single draw of 144 items is noisy."""
    measured = [measure(true_rho, seed=seed) for seed in range(20)]
    assert float(np.mean(measured)) == pytest.approx(true_rho, abs=0.03)


def test_interval_covers_the_truth_at_about_the_nominal_rate() -> None:
    covered = 0
    trials = 60
    for seed in range(trials):
        rng = np.random.default_rng(1000 + seed)
        estimate = estimate_icc(synthetic(rho=0.2, rng=rng), harm_type="toxicity")
        assert estimate.ci_low is not None and estimate.ci_high is not None
        covered += int(estimate.ci_low <= 0.2 <= estimate.ci_high)
    assert covered >= int(0.85 * trials)


def test_between_cell_variation_does_not_enter_rho() -> None:
    """THE failure that would condemn a workable design. Cells are a fixed factor."""
    flat = [measure(0.1, seed=seed, cell_spread=0.0) for seed in range(12)]
    spread = [measure(0.1, seed=seed, cell_spread=2.0) for seed in range(12)]
    assert float(np.mean(spread)) == pytest.approx(float(np.mean(flat)), abs=0.02)
    assert float(np.mean(spread)) == pytest.approx(0.1, abs=0.03)


def test_variance_components_are_on_the_right_scale() -> None:
    rng = np.random.default_rng(7)
    estimate = estimate_icc(synthetic(rho=0.3, rng=rng), harm_type="toxicity")
    assert estimate.var_between is not None and estimate.var_within is not None
    assert estimate.var_between == pytest.approx(0.3, abs=0.12)
    assert estimate.var_within == pytest.approx(0.7, abs=0.06)


def test_n0_is_the_seed_count_in_a_balanced_design() -> None:
    rng = np.random.default_rng(3)
    estimate = estimate_icc(synthetic(rho=0.2, rng=rng), harm_type="toxicity")
    assert estimate.n0 == pytest.approx(float(SEEDS), abs=1e-9)
    assert estimate.df_between == CELLS * (FRAMES - 1)
    assert estimate.df_within == CELLS * FRAMES * (SEEDS - 1)
    assert estimate.n_items == CELLS * FRAMES
    assert estimate.n_obs == CELLS * FRAMES * SEEDS


def test_rho_is_invariant_to_affine_rescaling() -> None:
    """P6 z-scores within benchmark; a variance RATIO cannot move under that."""
    rng = np.random.default_rng(11)
    raw = synthetic(rho=0.25, rng=rng)
    scaled = [
        IccObservation(cell_id=o.cell_id, item_id=o.item_id, value=3.7 * o.value - 2.1) for o in raw
    ]
    first = estimate_icc(raw, harm_type="toxicity")
    second = estimate_icc(scaled, harm_type="toxicity")
    assert first.rho is not None and second.rho is not None
    assert second.rho == pytest.approx(first.rho, abs=1e-10)


def test_unbalanced_design_is_handled() -> None:
    """A dropped generation must not silently bias n0, and therefore rho."""
    rng = np.random.default_rng(5)
    observations = synthetic(rho=0.2, rng=rng)
    thinned = [o for index, o in enumerate(observations) if index % 17 != 0]
    estimate = estimate_icc(thinned, harm_type="toxicity")
    assert estimate.estimable and estimate.rho is not None
    assert estimate.rho == pytest.approx(0.2, abs=0.08)
    assert estimate.n0 is not None and SEEDS - 1.5 < estimate.n0 < SEEDS


# ------------------------------------------------------------------ refusals


def test_one_response_per_prompt_is_refused_with_the_reason_named() -> None:
    """Exactly the dev run's shape. It must say why, not return zero."""
    rng = np.random.default_rng(2)
    estimate = estimate_icc(synthetic(rho=0.2, n_seeds=1, rng=rng), harm_type="toxicity")
    assert not estimate.estimable
    assert estimate.rho is None
    assert estimate.reason is not None
    assert "within-prompt variance" in estimate.reason


def test_one_item_per_cell_is_refused() -> None:
    rng = np.random.default_rng(2)
    estimate = estimate_icc(synthetic(rho=0.2, n_frames=1, rng=rng), harm_type="toxicity")
    assert not estimate.estimable
    assert estimate.reason is not None
    assert "fewer than two items" in estimate.reason


def test_identical_generations_are_refused() -> None:
    """Sampled decoding that produced identical text measured greedy, not rho."""
    observations = [
        IccObservation(cell_id=f"c{c}", item_id=f"c{c}::f{f}", value=float(f))
        for c in range(4)
        for f in range(3)
        for _ in range(5)
    ]
    estimate = estimate_icc(observations, harm_type="toxicity")
    assert not estimate.estimable
    assert estimate.reason is not None
    assert "did not actually vary" in estimate.reason


def test_no_observations_is_refused() -> None:
    estimate = estimate_icc([], harm_type="toxicity")
    assert not estimate.estimable
    assert estimate.n_obs == 0


def test_interval_refuses_degenerate_degrees_of_freedom() -> None:
    with pytest.raises(ValueError, match="degrees of freedom"):
        icc_interval(f_statistic=2.0, df_between=0, df_within=10, n0=5.0)
    with pytest.raises(ValueError, match="n0 must exceed 1"):
        icc_interval(f_statistic=2.0, df_between=5, df_within=10, n0=1.0)


def test_interval_matches_the_stage_a_planning_arithmetic() -> None:
    """The CI width that justified 12 seeds over 24 must be reproducible from this function."""
    n0 = 12.0
    f_at_rho_02 = (1 + (n0 - 1) * 0.2) / (1 - 0.2)
    low, high = icc_interval(
        f_statistic=f_at_rho_02,
        df_between=CELLS * (FRAMES - 1),
        df_within=CELLS * FRAMES * (SEEDS - 1),
        n0=n0,
    )
    assert low == pytest.approx(0.151, abs=0.005)
    assert high == pytest.approx(0.261, abs=0.005)
