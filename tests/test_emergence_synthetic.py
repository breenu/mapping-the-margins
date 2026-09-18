"""Gate P7: the emergence estimator recovers what was injected and invents nothing (P7, section 10).

These tests were written BEFORE the estimator, against synthetic data whose answer is known in
closed form. A silent numerical bug here is not a failing test -- it is a false paper claim -- so
every assertion is against an analytic truth rather than against a number the estimator once
produced.

What the synthetic data has, and why:

* **A known cell-mean table** for one axis pair: ``mu + alpha_a + beta_b + gamma_ab``. The truth
  the estimator must recover is the sum-to-zero decomposition of that table, computed by
  :func:`additive_decomposition` -- the same arithmetic P9's additive null (B2) presents.
* **Item clustering**: each prompt carries a random intercept shared by every one of its sampled
  generations, which is exactly why the ~50 seeds per prompt are not independent observations.
  The cluster-robust standard error must be WIDER than the naive iid one, or the clustering is
  being ignored.
* **Two scorers per harm type**, so the triangulated (mean-of-standardized) series is exercised.

The four questions the gate asks:

1. an injected interaction is recovered within its cluster-bootstrap CI;
2. a main-effects-only table -- with the intersection cell the HIGHEST level in the table --
   yields an interaction of ~0 (framing constraint 3, executable: a level is not emergence);
3. item-clustered SEs are demonstrably wider than iid SEs, and the gap grows with seeds per item;
4. the bootstrap is deterministic at a fixed seed, an order-1 cell yields no EffectEstimate, an
   empty cell is reported rather than crashed on, underpowered cells are ``powered=False``, and
   nothing here sets ``p``/``q``/``family_size``/``correction`` (invariant 3).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest

from margins.interfaces import Metric
from margins.lattice.build import DeclaredLattice
from margins.lattice.config import load_lattice_config
from margins.metrics.config import BootstrapSettings, MetricsConfig
from margins.metrics.design import PairDesign, additive_decomposition, pair_design
from margins.metrics.estimator import (
    NotEstimableError,
    PairFrame,
    estimate_pair,
    lstsq_params,
)
from margins.metrics.metric import EmergenceMetric, EmergenceResult
from margins.schemas import CellStat, EffectEstimate, IdentityCell, ItemScore

LATTICE_FIXTURE = Path(__file__).parent / "fixtures" / "lattice"
BENCHMARK = "open_gen_templates"
MODEL_ID = "fake/model"
HARM = "toxicity"
PAIR = ("alpha", "beta")
SCORERS = ("s_neural", "s_lexicon")

# Main effects chosen so the (a1, b1) intersection is the HIGHEST cell in every table below.
MAIN_A = {"a1": 0.8, "a2": -0.8}
MAIN_B = {"b1": 0.6, "b2": 0.0, "b3": -0.6}


# ------------------------------------------------------------------ synthetic data


@pytest.fixture(scope="module")
def lattice() -> DeclaredLattice:
    return DeclaredLattice(load_lattice_config("tiny", config_dir=LATTICE_FIXTURE))


@pytest.fixture(scope="module")
def pair_cells(lattice: DeclaredLattice) -> list[IdentityCell]:
    """The alpha x beta order-2 cells plus the alpha and beta order-1 cells: one pair's world."""
    return [c for c in lattice.cells() if {axis for axis, _ in c.axes} <= set(PAIR)]


def make_config(n_resamples: int = 200, seed: int = 20260918) -> MetricsConfig:
    return MetricsConfig(
        key="test",
        description="test config",
        metrics_version="test@2026-09-18",
        bootstrap=BootstrapSettings(n_resamples=n_resamples, ci_level=0.95, seed=seed),
    )


def cell_mean(cell: IdentityCell, gamma: Mapping[tuple[str, str], float], mu: float) -> float:
    values = cell.axes_map
    mean = mu
    if "alpha" in values:
        mean += MAIN_A[values["alpha"]]
    if "beta" in values:
        mean += MAIN_B[values["beta"]]
    if cell.order == 2:
        mean += gamma.get((values["alpha"], values["beta"]), 0.0)
    return mean


def synthetic_scores(
    cells: Sequence[IdentityCell],
    *,
    n_items: int,
    n_seeds: int,
    gamma: Mapping[tuple[str, str], float],
    item_sd: float,
    noise_sd: float,
    rng: np.random.Generator,
    mu: float = 0.0,
    scorer_noise: float = 0.0,
    scorers: Sequence[str] = SCORERS,
) -> list[ItemScore]:
    """Scores with a known cell-mean table and a random intercept per ITEM (prompt).

    ``y = mean(cell) + u_item + e``; ``u_item ~ N(0, item_sd^2)`` is shared by every seed of the
    same prompt, which is the clustering the estimator must respect.
    """
    scores: list[ItemScore] = []
    for cell in cells:
        mean = cell_mean(cell, gamma, mu)
        for item_index in range(n_items):
            item = f"{cell.cell_id}::item{item_index}"
            u_item = rng.normal(0.0, item_sd) if item_sd > 0 else 0.0
            for seed in range(n_seeds):
                y = mean + u_item + (rng.normal(0.0, noise_sd) if noise_sd > 0 else 0.0)
                response_id = f"{item}::seed{seed}"
                for scorer in scorers:
                    wobble = rng.normal(0.0, scorer_noise) if scorer_noise > 0 else 0.0
                    scores.append(
                        ItemScore(
                            score_id=f"{scorer}::model::{response_id}",
                            response_id=response_id,
                            scorer_id=scorer,
                            harm_type=HARM,
                            raw=y,
                            standardized=y + wobble,
                            prompt_id=item,
                            cell_id=cell.cell_id,
                            benchmark=BENCHMARK,
                            model_id=MODEL_ID,
                            seed=seed,
                            temperature=0.7,
                            condition="model",
                        )
                    )
    return scores


def truth(
    design: PairDesign, gamma: Mapping[tuple[str, str], float], mu: float = 0.0
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """The sum-to-zero decomposition of the injected cell-mean table: the answer key."""
    table = np.array(
        [
            [mu + MAIN_A[a] + MAIN_B[b] + gamma.get((a, b), 0.0) for b in design.levels_b]
            for a in design.levels_a
        ]
    )
    return additive_decomposition(table)


def frame_from_scores(scores: Sequence[ItemScore], scorer: str) -> PairFrame:
    rows = [s for s in scores if s.scorer_id == scorer and s.cell_id and "|" in s.cell_id]
    cells = [dict(part.split("=") for part in str(s.cell_id).split("|")) for s in rows]
    return PairFrame(
        y=np.array([s.standardized for s in rows]),
        value_a=tuple(c["alpha"] for c in cells),
        value_b=tuple(c["beta"] for c in cells),
        item=tuple(s.prompt_id for s in rows),
    )


def interactions(result: EmergenceResult) -> dict[tuple[str, str], EffectEstimate]:
    """The interaction EffectEstimates keyed by ``(value_a, value_b)``."""
    out: dict[tuple[str, str], EffectEstimate] = {}
    for effect in result.effects:
        if ":" not in effect.term:
            continue
        left, right = effect.term.split(":")
        out[(left.split("=")[1], right.split("=")[1])] = effect
    return out


# ------------------------------------------------------------------ the design itself


def test_sum_coding_reproduces_the_balanced_difference_in_differences(
    pair_cells: list[IdentityCell],
) -> None:
    """With no noise the fitted gammas ARE the cell-mean decomposition, to floating-point.

    gamma_ij = mu_ij - mu_i. - mu_.j + mu.. is the textbook two-way interaction; sum-to-zero
    coding must reproduce it exactly, or the parameterization is not the one pre-registered.
    """
    design = pair_design(pair_cells, PAIR)
    gamma = {("a1", "b1"): 0.5, ("a2", "b3"): -0.2}
    scores = synthetic_scores(
        pair_cells,
        n_items=2,
        n_seeds=2,
        gamma=gamma,
        item_sd=0.0,
        noise_sd=0.0,
        rng=np.random.default_rng(0),
    )
    frame = frame_from_scores(scores, SCORERS[0])
    fit = estimate_pair(frame, design, n_resamples=0, ci_level=0.95, rng=None)

    mu, alpha, beta, gamma_true = truth(design, gamma)
    for i, a in enumerate(design.levels_a):
        for j, b in enumerate(design.levels_b):
            assert fit.interaction(a, b).estimate == pytest.approx(gamma_true[i, j], abs=1e-10)
        assert fit.main("alpha", a).estimate == pytest.approx(alpha[i], abs=1e-10)
    for j, b in enumerate(design.levels_b):
        assert fit.main("beta", b).estimate == pytest.approx(beta[j], abs=1e-10)
    assert fit.params[0] == pytest.approx(mu, abs=1e-10)
    # Every implied constraint holds: the gammas sum to zero along both margins.
    table = np.array(
        [[fit.interaction(a, b).estimate for b in design.levels_b] for a in design.levels_a]
    )
    assert np.allclose(table.sum(axis=0), 0.0, atol=1e-10)
    assert np.allclose(table.sum(axis=1), 0.0, atol=1e-10)


def test_the_bootstrap_refit_path_matches_statsmodels_exactly(
    pair_cells: list[IdentityCell],
) -> None:
    """The bootstrap refits with numpy lstsq for speed; it must agree with the point fit."""
    design = pair_design(pair_cells, PAIR)
    scores = synthetic_scores(
        pair_cells,
        n_items=4,
        n_seeds=3,
        gamma={("a1", "b1"): 0.3},
        item_sd=0.4,
        noise_sd=0.5,
        rng=np.random.default_rng(1),
    )
    frame = frame_from_scores(scores, SCORERS[0])
    fit = estimate_pair(frame, design, n_resamples=0, ci_level=0.95, rng=None)

    design_matrix = design.matrix(frame.value_a, frame.value_b)
    assert np.allclose(lstsq_params(design_matrix, frame.y), fit.params, atol=1e-10)


def test_the_cluster_robust_covariance_matches_a_hand_computed_sandwich(
    pair_cells: list[IdentityCell],
) -> None:
    """Guards the cov_kwds wiring: a mis-passed ``groups`` would still return a covariance."""
    design = pair_design(pair_cells, PAIR)
    scores = synthetic_scores(
        pair_cells,
        n_items=5,
        n_seeds=4,
        gamma={("a1", "b1"): 0.3},
        item_sd=0.6,
        noise_sd=0.4,
        rng=np.random.default_rng(2),
    )
    frame = frame_from_scores(scores, SCORERS[0])
    fit = estimate_pair(frame, design, n_resamples=0, ci_level=0.95, rng=None)

    design_matrix = design.matrix(frame.value_a, frame.value_b)
    y = frame.y
    bread = np.linalg.inv(design_matrix.T @ design_matrix)
    residuals = y - design_matrix @ fit.params
    meat = np.zeros_like(bread)
    items = sorted(set(frame.item))
    for item in items:
        mask = np.array([i == item for i in frame.item])
        score = design_matrix[mask].T @ residuals[mask]
        meat += np.outer(score, score)
    n, k = design_matrix.shape
    g = len(items)
    # statsmodels' default small-sample correction for cov_type="cluster".
    correction = (g / (g - 1)) * ((n - 1) / (n - k))
    by_hand = correction * bread @ meat @ bread

    assert fit.cov_type == "cluster"
    assert fit.n_items == g
    assert np.allclose(fit.cov, by_hand, atol=1e-10)


# ------------------------------------------------------------------ RECOVERY (the gate)


def test_an_injected_interaction_is_recovered_within_its_bootstrap_ci(
    pair_cells: list[IdentityCell],
) -> None:
    """THE gate, half one: what was put in comes out, and the CI knows it."""
    gamma = {("a1", "b1"): 1.0}
    scores = synthetic_scores(
        pair_cells,
        n_items=12,
        n_seeds=10,
        gamma=gamma,
        item_sd=0.3,
        noise_sd=0.5,
        rng=np.random.default_rng(3),
        scorer_noise=0.05,
    )
    metric = EmergenceMetric(make_config(n_resamples=300), min_cell_n=10)
    result = metric.compute_result(scores, pair_cells)

    design = pair_design(pair_cells, PAIR)
    _, _, _, gamma_true = truth(design, gamma)
    found = interactions(result)
    assert set(found) == {(a, b) for a in design.levels_a for b in design.levels_b}
    for i, a in enumerate(design.levels_a):
        for j, b in enumerate(design.levels_b):
            effect = found[(a, b)]
            assert abs(effect.estimate - gamma_true[i, j]) <= 3.0 * effect.se, (
                f"gamma[{a},{b}] est {effect.estimate:.3f} vs truth {gamma_true[i, j]:.3f} "
                f"(se {effect.se:.3f})"
            )
    # The injected bump is the largest interaction in the table, and its CI excludes zero.
    bump = found[("a1", "b1")]
    assert bump.estimate == pytest.approx(gamma_true[0, 0], abs=0.2)
    assert bump.ci_low > 0.0
    assert max(found.values(), key=lambda e: abs(e.estimate)) is bump

    report = next(r for r in result.fit_reports if r.scorer_id is None)
    assert report.estimable
    assert report.rms_interaction is not None and report.rms_ci_low is not None


def test_the_bootstrap_ci_is_calibrated_over_repeated_datasets(
    pair_cells: list[IdentityCell],
) -> None:
    """Coverage is a property of the procedure, not of one dataset: check it over many.

    A single fixed-seed dataset either covers or it does not, and asserting joint coverage of
    seven terms at one seed fails ~25% of the time for a CORRECT estimator. So the check is the
    honest one: over 24 independent datasets the 95% CI for the injected bump covers the truth at
    least 19 times (a binomial(24, 0.95) miss of five or more has probability under 1%), and no
    estimate is ever further than four SEs from the truth.
    """
    gamma = {("a1", "b1"): 1.0}
    design = pair_design(pair_cells, PAIR)
    _, _, _, gamma_true = truth(design, gamma)
    truth_rms = float(np.sqrt(np.mean(gamma_true**2)))

    covered_bump = covered_rms = 0
    n_datasets = 24
    for dataset in range(n_datasets):
        scores = synthetic_scores(
            pair_cells,
            n_items=10,
            n_seeds=8,
            gamma=gamma,
            item_sd=0.3,
            noise_sd=0.5,
            rng=np.random.default_rng(100 + dataset),
        )
        result = EmergenceMetric(make_config(n_resamples=200, seed=dataset), min_cell_n=10)
        out = result.compute_result(scores, pair_cells)
        bump = interactions(out)[("a1", "b1")]
        assert abs(bump.estimate - gamma_true[0, 0]) <= 4.0 * bump.se
        covered_bump += bump.ci_low <= gamma_true[0, 0] <= bump.ci_high
        report = next(r for r in out.fit_reports if r.scorer_id is None)
        assert report.rms_ci_low is not None and report.rms_ci_high is not None
        covered_rms += report.rms_ci_low <= truth_rms <= report.rms_ci_high

    assert covered_bump >= 19, f"bump CI covered the truth in only {covered_bump}/{n_datasets}"
    assert covered_rms >= 17, f"rms CI covered the truth in only {covered_rms}/{n_datasets}"


def test_a_main_effects_only_table_yields_no_interaction(
    pair_cells: list[IdentityCell],
) -> None:
    """THE gate, half two: the estimator must not fabricate emergence.

    The (a1, b1) cell is the HIGHEST level in the table -- a level-based check would flag it as
    the emergent cell. The interaction is ~0 because the elevation is fully additive. This is
    framing constraint 3 in executable form, and it mirrors the fabricated-emergence failure mode
    the drift control exists for: the estimator itself must be innocent of it.
    """
    scores = synthetic_scores(
        pair_cells,
        n_items=20,
        n_seeds=10,
        gamma={},
        item_sd=0.2,
        noise_sd=0.3,
        rng=np.random.default_rng(4),
        scorer_noise=0.05,
    )
    metric = EmergenceMetric(make_config(n_resamples=300), min_cell_n=10)
    result = metric.compute_result(scores, pair_cells)

    # The level trap: the intersection cell really is the highest mean in the table.
    stats = {s.cell_id: s for s in result.cell_stats}
    top = max(stats.values(), key=lambda s: s.mean_standardized)
    assert top.cell_id == "alpha=a1|beta=b1"
    assert top.mean_standardized > 1.0

    found = interactions(result)
    for (a, b), effect in found.items():
        assert abs(effect.estimate) < 0.12, f"fabricated interaction at ({a},{b}): {effect}"
        assert abs(effect.estimate) <= 4.0 * effect.se
    report = next(r for r in result.fit_reports if r.scorer_id is None)
    assert report.rms_interaction is not None and report.rms_interaction < 0.1

    # And over repeated datasets the estimator is unbiased at zero: the mean interaction across
    # 24 independent draws is ~0, and the 95% CI covers zero at least 19 times (see the
    # calibration test above for the binomial arithmetic).
    n_datasets = 24
    estimates: list[float] = []
    covered = 0
    for dataset in range(n_datasets):
        scores = synthetic_scores(
            pair_cells,
            n_items=10,
            n_seeds=8,
            gamma={},
            item_sd=0.2,
            noise_sd=0.3,
            rng=np.random.default_rng(200 + dataset),
        )
        out = EmergenceMetric(make_config(n_resamples=200, seed=dataset), min_cell_n=10)
        bump = interactions(out.compute_result(scores, pair_cells))[("a1", "b1")]
        estimates.append(bump.estimate)
        covered += bump.ci_low <= 0.0 <= bump.ci_high
    assert abs(float(np.mean(estimates))) < 0.03, f"biased at zero: {np.mean(estimates):.4f}"
    assert covered >= 19, f"zero covered in only {covered}/{n_datasets} datasets"


def test_item_clustered_standard_errors_are_wider_than_iid(
    pair_cells: list[IdentityCell],
) -> None:
    """Seed-samples within a prompt are NOT independent, and the SE must say so.

    A strong item intercept with many seeds per item makes the design effect large. The iid SE
    treats every seed as new information and is far too small; the cluster-robust SE is not.
    The gap must also GROW with seeds per item, which is what distinguishes clustering from a
    constant inflation.
    """
    design = pair_design(pair_cells, PAIR)

    def se_ratio(n_seeds: int) -> tuple[float, float, float]:
        scores = synthetic_scores(
            pair_cells,
            n_items=6,
            n_seeds=n_seeds,
            gamma={("a1", "b1"): 0.5},
            item_sd=1.0,
            noise_sd=0.3,
            rng=np.random.default_rng(5),
        )
        frame = frame_from_scores(scores, SCORERS[0])
        clustered = estimate_pair(
            frame, design, cov_type="cluster", n_resamples=0, ci_level=0.95, rng=None
        )
        iid = estimate_pair(
            frame, design, cov_type="nonrobust", n_resamples=0, ci_level=0.95, rng=None
        )
        se_c = np.array([t.se for t in clustered.terms if t.kind == "interaction"])
        se_i = np.array([t.se for t in iid.terms if t.kind == "interaction"])
        return float(np.min(se_c / se_i)), float(np.mean(se_c)), float(np.mean(se_i))

    ratio_few, _, _ = se_ratio(5)
    ratio_many, se_clustered, se_iid = se_ratio(20)

    assert ratio_many > 1.5, f"clustered/iid SE ratio only {ratio_many:.2f}; clustering ignored?"
    assert ratio_many > ratio_few, "the design effect must grow with seeds per item"

    # And the clustered SE is the RIGHT size, not merely larger: compare with the analytic
    # sampling SD of a cell-mean contrast under the generating model.
    n_items, n_seeds = 6, 20
    var_cell_mean = 1.0**2 / n_items + 0.3**2 / (n_items * n_seeds)
    # gamma_11 in a 2x3 table is a contrast with weights (1-1/2)(1-1/3) on the cell itself and
    # correspondingly smaller ones elsewhere; its variance is sum(w^2) * var_cell_mean.
    weights = np.outer([1 - 1 / 2, -1 / 2], [1 - 1 / 3, -1 / 3, -1 / 3])
    analytic_se = float(np.sqrt(np.sum(weights**2) * var_cell_mean))
    assert 0.5 * analytic_se < se_clustered < 1.6 * analytic_se
    assert se_iid < 0.5 * analytic_se


# ------------------------------------------------------------------ sampling structure


def test_seeds_are_not_pre_averaged(pair_cells: list[IdentityCell]) -> None:
    """The fit sees every sampled observation; n counts renderings, not prompts."""
    n_items, n_seeds = 4, 6
    scores = synthetic_scores(
        pair_cells,
        n_items=n_items,
        n_seeds=n_seeds,
        gamma={},
        item_sd=0.2,
        noise_sd=0.2,
        rng=np.random.default_rng(6),
    )
    result = EmergenceMetric(make_config(n_resamples=50), min_cell_n=10).compute_result(
        scores, pair_cells
    )
    for stat in result.cell_stats:
        assert stat.n == n_items * n_seeds
    report = next(r for r in result.fit_reports if r.scorer_id is None)
    order_2 = sum(1 for c in pair_cells if c.order == 2)
    assert report.n_obs == order_2 * n_items * n_seeds
    assert report.n_items == order_2 * n_items


def test_the_triangulated_series_is_the_mean_of_the_scorers(
    pair_cells: list[IdentityCell],
) -> None:
    """harm_std for the harm-type fit is the equal-weight mean of the scorers' standardized
    scores; the per-scorer fits are emitted beside it."""
    scores = synthetic_scores(
        pair_cells,
        n_items=3,
        n_seeds=2,
        gamma={("a1", "b1"): 0.4},
        item_sd=0.0,
        noise_sd=0.0,
        rng=np.random.default_rng(7),
        scorer_noise=0.3,
    )
    result = EmergenceMetric(make_config(n_resamples=20), min_cell_n=1).compute_result(
        scores, pair_cells
    )
    series = {r.scorer_id for r in result.fit_reports}
    assert series == {None, *SCORERS}

    def estimate(scorer_id: str | None) -> float:
        return next(
            t.estimate
            for t in result.fit_terms
            if t.scorer_id == scorer_id and t.term == "alpha=a1:beta=b1"
        )

    combined = estimate(None)
    per_scorer = [estimate(s) for s in SCORERS]
    # Balanced, noise-free design: the combined gamma is exactly the mean of the two scorers'.
    assert combined == pytest.approx(float(np.mean(per_scorer)), abs=1e-9)
    assert per_scorer[0] != pytest.approx(per_scorer[1])  # the scorers genuinely differ


def test_a_response_missing_a_scorer_is_refused(pair_cells: list[IdentityCell]) -> None:
    """A row scored by one scorer of two would be single-classifier-dependent (constraint 6)."""
    scores = synthetic_scores(
        pair_cells,
        n_items=2,
        n_seeds=2,
        gamma={},
        item_sd=0.0,
        noise_sd=0.1,
        rng=np.random.default_rng(8),
    )
    dropped = [s for s in scores if not (s.scorer_id == SCORERS[1] and s.seed == 0)]
    with pytest.raises(ValueError, match="scored by"):
        EmergenceMetric(make_config(n_resamples=10), min_cell_n=1).compute(dropped, pair_cells)


# ------------------------------------------------------------------ determinism


def test_the_bootstrap_is_deterministic_at_a_fixed_seed(pair_cells: list[IdentityCell]) -> None:
    scores = synthetic_scores(
        pair_cells,
        n_items=5,
        n_seeds=4,
        gamma={("a1", "b1"): 0.5},
        item_sd=0.3,
        noise_sd=0.4,
        rng=np.random.default_rng(9),
    )
    first = EmergenceMetric(make_config(n_resamples=100, seed=11), min_cell_n=1).compute(
        scores, pair_cells
    )
    second = EmergenceMetric(make_config(n_resamples=100, seed=11), min_cell_n=1).compute(
        scores, pair_cells
    )
    other = EmergenceMetric(make_config(n_resamples=100, seed=12), min_cell_n=1).compute(
        scores, pair_cells
    )

    assert first == second
    # A different seed moves the CI endpoints (the bootstrap is real) but not the estimate.
    for a, b in zip(first[1], other[1], strict=True):
        assert a.estimate == b.estimate and a.se == b.se
    assert any(
        a.ci_low != b.ci_low or a.ci_high != b.ci_high
        for a, b in zip(first[1], other[1], strict=True)
    )
    assert first[0] == second[0]


# ------------------------------------------------------------------ edge cases


def test_order_one_cells_get_a_cell_stat_but_no_effect_estimate(
    pair_cells: list[IdentityCell],
) -> None:
    """An interaction belongs to an axis PAIR; an order-1 cell has none."""
    scores = synthetic_scores(
        pair_cells,
        n_items=3,
        n_seeds=2,
        gamma={},
        item_sd=0.1,
        noise_sd=0.1,
        rng=np.random.default_rng(10),
    )
    cell_stats, effects = EmergenceMetric(make_config(n_resamples=20), min_cell_n=1).compute(
        scores, pair_cells
    )
    described = {s.cell_id for s in cell_stats}
    assert described == {c.cell_id for c in pair_cells}  # order-1 AND order-2 cells described
    assert effects
    for effect in effects:
        assert effect.axis_pair == PAIR


def test_an_empty_cell_makes_the_pair_not_estimable_without_crashing(
    pair_cells: list[IdentityCell],
) -> None:
    """No NaN ever reaches an EffectEstimate; the pair is reported with the cell named."""
    scores = synthetic_scores(
        pair_cells,
        n_items=3,
        n_seeds=2,
        gamma={},
        item_sd=0.1,
        noise_sd=0.1,
        rng=np.random.default_rng(11),
    )
    scores = [s for s in scores if s.cell_id != "alpha=a2|beta=b3"]

    result = EmergenceMetric(make_config(n_resamples=20), min_cell_n=1).compute_result(
        scores, pair_cells
    )
    assert result.effects == []
    report = next(r for r in result.fit_reports if r.scorer_id is None)
    assert not report.estimable
    assert report.reason is not None and "alpha=a2|beta=b3" in report.reason
    assert report.rms_interaction is None
    assert "alpha=a2|beta=b3" not in {s.cell_id for s in result.cell_stats}


def test_a_saturated_design_is_not_estimable(pair_cells: list[IdentityCell]) -> None:
    """One observation per cell leaves no residual degrees of freedom: no SE is definable."""
    design = pair_design(pair_cells, PAIR)
    scores = synthetic_scores(
        pair_cells,
        n_items=1,
        n_seeds=1,
        gamma={},
        item_sd=0.0,
        noise_sd=0.1,
        rng=np.random.default_rng(12),
    )
    frame = frame_from_scores(scores, SCORERS[0])
    with pytest.raises(NotEstimableError, match="degrees of freedom"):
        estimate_pair(frame, design, n_resamples=0, ci_level=0.95, rng=None)


def test_underpowered_cells_and_pairs_are_marked_powered_false(
    pair_cells: list[IdentityCell],
) -> None:
    """Invariant 4: the flag comes from the ACTUAL n against min_cell_n, never from a projection."""
    scores = synthetic_scores(
        pair_cells,
        n_items=3,
        n_seeds=2,
        gamma={},
        item_sd=0.1,
        noise_sd=0.1,
        rng=np.random.default_rng(13),
    )
    powered = EmergenceMetric(make_config(n_resamples=20), min_cell_n=6).compute(scores, pair_cells)
    under = EmergenceMetric(make_config(n_resamples=20), min_cell_n=7).compute(scores, pair_cells)
    assert all(s.powered for s in powered[0]) and all(e.powered for e in powered[1])
    assert not any(s.powered for s in under[0]) and not any(e.powered for e in under[1])
    assert under[1], "underpowered pairs are still ESTIMATED -- just descriptive-only"


def test_drift_condition_rows_are_not_fitted_by_compute(pair_cells: list[IdentityCell]) -> None:
    """The drift series has its own fit; compute() reports what it skipped rather than mixing."""
    scores = synthetic_scores(
        pair_cells,
        n_items=2,
        n_seeds=2,
        gamma={},
        item_sd=0.0,
        noise_sd=0.1,
        rng=np.random.default_rng(14),
    )
    drift = [
        s.model_copy(update={"condition": "drift_control", "score_id": f"d::{s.score_id}"})
        for s in scores[:8]
    ]
    result = EmergenceMetric(make_config(n_resamples=10), min_cell_n=1).compute_result(
        [*scores, *drift], pair_cells
    )
    assert result.n_skipped_rows == 8
    report = next(r for r in result.fit_reports if r.scorer_id is None)
    order_2 = sum(1 for c in pair_cells if c.order == 2)
    assert report.n_obs == order_2 * 2 * 2  # model-condition responses only


# ------------------------------------------------------------------ the contract


def test_the_metric_satisfies_the_protocol_and_sets_no_significance(
    pair_cells: list[IdentityCell],
) -> None:
    """Invariant 3: p, q, family_size and correction are StatsEngine's, and only StatsEngine's."""
    metric = EmergenceMetric(make_config(n_resamples=20), min_cell_n=1)
    assert isinstance(metric, Metric)

    scores = synthetic_scores(
        pair_cells,
        n_items=3,
        n_seeds=2,
        gamma={("a1", "b1"): 0.3},
        item_sd=0.1,
        noise_sd=0.1,
        rng=np.random.default_rng(15),
    )
    cell_stats, effects = metric.compute(scores, pair_cells)

    assert isinstance(cell_stats, list) and all(isinstance(s, CellStat) for s in cell_stats)
    assert isinstance(effects, list) and all(isinstance(e, EffectEstimate) for e in effects)
    assert effects
    for effect in effects:
        assert effect.p is None
        assert effect.q is None
        assert effect.family_size is None
        assert effect.correction is None
        assert not effect.corrected
        assert effect.ci_low <= effect.ci_high
        assert effect.benchmark == BENCHMARK and effect.model == MODEL_ID
        assert effect.harm_type == HARM
    assert not {"p", "q", "family_size", "correction"} & set(CellStat.model_fields)

    # Main effects and interactions are both emitted, keyed by term, and the interaction count
    # is the full |A| x |B| table (the dropped level is implied, not omitted).
    terms = {e.term for e in effects}
    assert {"alpha=a1", "alpha=a2", "beta=b1", "beta=b2", "beta=b3"} <= terms
    assert sum(1 for t in terms if ":" in t) == 2 * 3


def test_effect_estimates_carry_the_bootstrap_ci_not_a_normal_approximation(
    pair_cells: list[IdentityCell],
) -> None:
    """The CI is the percentile cluster bootstrap; it need not be symmetric about the estimate."""
    scores = synthetic_scores(
        pair_cells,
        n_items=5,
        n_seeds=3,
        gamma={("a1", "b1"): 0.5},
        item_sd=0.5,
        noise_sd=0.3,
        rng=np.random.default_rng(16),
    )
    _, effects = EmergenceMetric(make_config(n_resamples=200), min_cell_n=1).compute(
        scores, pair_cells
    )
    asymmetric = [
        e for e in effects if abs((e.ci_high - e.estimate) - (e.estimate - e.ci_low)) > 1e-6
    ]
    assert asymmetric, "every CI is exactly symmetric, which a percentile bootstrap never is"
