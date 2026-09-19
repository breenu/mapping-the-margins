"""The power analysis: a SIMULATION on the actual pipeline, not a formula (P9, section 8).

Locked decision, 19 Sep 2026: power is measured by injecting a known interaction into synthetic
data with the real clustered structure, pushing it through the **real P7 estimator** and the
**real P9 FDR**, and counting how often the planted term survives correction. A closed-form power
formula would have to assume away the two things that actually bind here -- item clustering and
multiplicity -- and the whole point of the exercise is that neither is negligible.

**The structural fact this module exists to measure.** A cell is rendered by ``frames_per_cell``
template frames. Each ``(frame, cell)`` is one prompt, i.e. one **item cluster**, and the sampled
seeds are repeated observations *inside* that cluster. So a cell has ``frames_per_cell`` clusters
however many seeds are drawn. With total standardized variance 1 split by an intraclass
correlation ``rho``, the variance of a cell mean is::

    rho / frames  +  (1 - rho) / (frames * seeds)

which **floors at ``rho / frames``**. Past some seed count, more generation buys nothing. That
ceiling is simulated explicitly (:func:`ceiling_point`) rather than argued, and when it sits below
the target power the result is ``min_seeds_per_item=None`` with the reason named -- never an
extrapolated number, because an extrapolated number here is a fabricated compute budget.

**``rho`` is an assumption, reported as a strip.** It cannot be estimated from anything that
exists: the dev run has one response per prompt, so there is no within-item variance to observe.
Sweeping it and printing the whole strip is the honest treatment; collapsing it to one hidden
constant would present an assumption as a measurement.

**Two injected patterns, because "an interaction of 0.3 SD" is ambiguous on a gamma table.**
``concentrated`` puts the magnitude in a single intersection (``max|gamma| = effect_size``) and is
the headline, matching a per-term FDR family; ``diffuse`` spreads it (``RMS(gamma) = effect_size``).
Both are built as a raw table, projected through :func:`~margins.metrics.design.
additive_decomposition`, and **rescaled after projection** -- putting the number in a corner and
calling it the effect would be wrong, because the sum-to-zero projection shrinks it and the
reported minimum detectable effect would then be about a different quantity than the one tested.

**No bootstrap.** :func:`~margins.metrics.estimator.estimate_pair` is called with
``n_resamples=0``: power depends on the cluster-robust Wald standard error, not on the percentile
CI, and skipping the bootstrap is what makes a grid of this size affordable. It is also why the
test suite spies on the call -- a simulation that quietly fitted iid would report fantasy power.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

from margins.metrics.design import PairDesign, additive_decomposition, interaction_term
from margins.metrics.estimator import PairFrame, estimate_pair
from margins.stats.config import CorrectionMethod, EffectPattern, PowerSettings
from margins.stats.family import free_interaction_terms
from margins.stats.records import PowerPoint, PowerResult
from margins.stats.wald import wald_p

#: How the minimum is located on the seeds grid. ``bisect`` exploits monotonicity in seeds and is
#: the default because a linear walk spends its budget on the expensive large points it passes
#: through; ``linear`` returns the whole curve and is what a report or a plot wants.
SearchStrategy = Literal["bisect", "linear"]

#: Called with a one-line status before each simulated point, so a long grid is observable.
ProgressHook = Callable[[str], None]

__all__ = [
    "PowerDesign",
    "ProgressHook",
    "SearchStrategy",
    "ceiling_point",
    "cell_mean_table",
    "interaction_table",
    "minimum_renderings",
    "power_curve",
    "power_settings_grid",
    "simulate_frame",
    "simulate_power_point",
    "wilson_interval",
]

#: Which norm of the projected gamma table each pattern scales to ``effect_size``.
_SCALE_OF_PATTERN: Final[dict[EffectPattern, str]] = {
    "concentrated": "max_cell",
    "diffuse": "rms",
}

#: Main-effect spread injected under every pattern, in SD units. Present so the simulated table has
#: real *levels* to be distinguished from its *interaction* (framing constraint 3), and so a cell
#: mean is not a bare gamma. Balanced designs make the interaction's variance independent of these,
#: which the tests check.
_MAIN_A_SPREAD: Final[float] = 0.8
_MAIN_B_SPREAD: Final[float] = 0.6

#: The unused percentile level handed to ``estimate_pair`` when the bootstrap is switched off.
_UNUSED_CI_LEVEL: Final[float] = 0.95


@dataclass(frozen=True, eq=False)
class PowerDesign:
    """What one simulated design point holds fixed.

    ``frames_per_cell`` is the ITEM CLUSTER count per cell and ``seeds_per_item`` is the number of
    observations inside each cluster. They are separate fields, and ``renderings_per_cell`` is
    their product, so the distinction the whole analysis turns on is visible in the type rather
    than buried in a comment.
    """

    design: PairDesign
    frames_per_cell: int
    seeds_per_item: int

    def __post_init__(self) -> None:
        if self.frames_per_cell < 2:
            raise ValueError(
                f"frames_per_cell must be at least 2, got {self.frames_per_cell}. A cell with one "
                "item cluster cannot support a cluster-robust standard error."
            )
        if self.seeds_per_item < 1:
            raise ValueError(f"seeds_per_item must be at least 1, got {self.seeds_per_item}")

    @property
    def items_per_cell(self) -> int:
        """Item clusters per cell. Equal to the template-frame count; seeds do not add any."""
        return self.frames_per_cell

    @property
    def n_item_clusters(self) -> int:
        return self.frames_per_cell * self.design.n_cells

    @property
    def renderings_per_cell(self) -> int:
        return self.frames_per_cell * self.seeds_per_item

    @property
    def n_obs(self) -> int:
        return self.n_item_clusters * self.seeds_per_item


# --------------------------------------------------------------------------------------
# the injected truth
# --------------------------------------------------------------------------------------


def interaction_table(
    design: PairDesign, *, effect_size: float, pattern: EffectPattern
) -> np.ndarray:
    """The injected gamma table: already sum-to-zero, scaled so the pattern's norm is the target.

    ``concentrated`` is a unit bump at the first cell, projected; its largest gamma lands on that
    cell, which is always inside the free contrast set. ``diffuse`` is the outer product of two
    centred alternating vectors, so every cell carries the same magnitude.

    **The rescale happens after the projection**, so ``max|gamma|`` (concentrated) or
    ``RMS(gamma)`` (diffuse) is exactly ``effect_size``. Scaling a raw table before projecting
    would report a minimum detectable effect for a quantity nobody tests.
    """
    if effect_size <= 0.0:
        raise ValueError(f"effect_size must be positive, got {effect_size}")
    n_a, n_b = len(design.levels_a), len(design.levels_b)

    if pattern == "concentrated":
        raw = np.zeros((n_a, n_b))
        raw[0, 0] = 1.0
    else:
        vector_a = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n_a)])
        vector_b = np.array([1.0 if j % 2 == 0 else -1.0 for j in range(n_b)])
        raw = np.outer(vector_a - vector_a.mean(), vector_b - vector_b.mean())

    _, _, _, gamma = additive_decomposition(raw)
    scale = _SCALE_OF_PATTERN[pattern]
    norm = float(np.abs(gamma).max()) if scale == "max_cell" else float(np.sqrt(np.mean(gamma**2)))
    if norm <= 0.0:  # pragma: no cover - both patterns are non-degenerate by construction
        raise ValueError(f"pattern {pattern!r} produced a zero interaction table")
    return gamma * (effect_size / norm)


def cell_mean_table(design: PairDesign, gamma: np.ndarray) -> np.ndarray:
    """``mu_ij = alpha_i + beta_j + gamma_ij`` with fixed, centred main effects.

    The main effects exist so the simulated table has *levels*. They are centred, so the grand mean
    is zero and the injected gammas are recovered unchanged -- which is what makes the truth an
    answer key rather than an approximation.
    """
    n_a, n_b = len(design.levels_a), len(design.levels_b)
    alpha = np.linspace(_MAIN_A_SPREAD, -_MAIN_A_SPREAD, n_a)
    beta = np.linspace(_MAIN_B_SPREAD, -_MAIN_B_SPREAD, n_b)
    alpha = alpha - alpha.mean()
    beta = beta - beta.mean()
    return np.asarray(alpha[:, None] + beta[None, :] + gamma, dtype=float)


# --------------------------------------------------------------------------------------
# the simulated data
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class _Layout:
    """Everything about a design point that does not change between replicates.

    Built once and reused, because the labels and the cell index of each row are pure functions of
    the design; regenerating them per replicate would dominate the runtime of the whole grid.
    """

    value_a: tuple[str, ...]
    value_b: tuple[str, ...]
    item: tuple[str, ...]
    cell_of_item: np.ndarray
    n_items: int
    seeds: int


def _layout(power_design: PowerDesign, *, limiting: bool) -> _Layout:
    design = power_design.design
    seeds = 1 if limiting else power_design.seeds_per_item
    value_a: list[str] = []
    value_b: list[str] = []
    item: list[str] = []
    cell_of_item: list[int] = []
    for index, (a, b, cell_id) in enumerate(design.cells):
        for frame in range(power_design.frames_per_cell):
            cell_of_item.append(index)
            item_id = f"{cell_id}::frame{frame}"
            for _ in range(seeds):
                value_a.append(a)
                value_b.append(b)
                item.append(item_id)
    return _Layout(
        value_a=tuple(value_a),
        value_b=tuple(value_b),
        item=tuple(item),
        cell_of_item=np.asarray(cell_of_item, dtype=int),
        n_items=len(cell_of_item),
        seeds=seeds,
    )


def simulate_frame(
    power_design: PowerDesign,
    means: np.ndarray,
    *,
    icc_rho: float,
    rng: np.random.Generator,
    limiting: bool = False,
    layout: _Layout | None = None,
) -> PairFrame:
    """One synthetic dataset: ``y = mu_cell + u_item + e``, total variance exactly 1.

    ``u_item ~ N(0, rho)`` is shared by every seed of the same prompt and ``e ~ N(0, 1 - rho)`` is
    drawn per rendering, so ``Var(y) = 1`` -- the z-scored scale the estimator consumes, which is
    what lets ``effect_size`` be read as SD units without a further conversion.

    ``limiting=True`` is the seeds -> infinity design: one observation per item cluster and no
    within-item noise, the item carrying variance ``rho``. It is the exact limit of the clustered
    design as seeds grow, and simulating it directly is how the ceiling is measured rather than
    extrapolated.
    """
    if not 0.0 <= icc_rho < 1.0:
        raise ValueError(f"icc_rho must satisfy 0 <= rho < 1, got {icc_rho}")
    resolved = layout if layout is not None else _layout(power_design, limiting=limiting)

    sd_item = float(np.sqrt(icc_rho))
    sd_noise = 0.0 if limiting else float(np.sqrt(1.0 - icc_rho))

    flat_means = np.asarray(means, dtype=float).reshape(-1)
    per_item = flat_means[resolved.cell_of_item]
    u = rng.normal(0.0, sd_item, size=resolved.n_items) if sd_item > 0.0 else 0.0
    base = (per_item + u)[:, None]
    if sd_noise > 0.0:
        y = base + rng.normal(0.0, sd_noise, size=(resolved.n_items, resolved.seeds))
    else:
        y = np.repeat(base, resolved.seeds, axis=1)

    return PairFrame(
        y=y.reshape(-1),
        value_a=resolved.value_a,
        value_b=resolved.value_b,
        item=resolved.item,
    )


# --------------------------------------------------------------------------------------
# one simulated design point
# --------------------------------------------------------------------------------------


def wilson_interval(successes: int, trials: int, *, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion. Never returns a bound outside ``[0, 1]``.

    A normal-approximation interval would put the bound outside the unit interval at exactly the
    place that matters here -- a power near 0 or near 1 -- and a power of "1.04" in an artifact is
    worse than no interval.
    """
    from scipy import stats

    if trials < 1:
        raise ValueError("a proportion needs at least one trial")
    z = float(stats.norm.isf((1.0 - level) / 2.0))
    phat = successes / trials
    denominator = 1.0 + z**2 / trials
    centre = (phat + z**2 / (2 * trials)) / denominator
    half = (z / denominator) * np.sqrt(phat * (1 - phat) / trials + z**2 / (4 * trials**2))
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def simulate_power_point(
    power_design: PowerDesign,
    *,
    pattern: EffectPattern,
    effect_size: float,
    icc_rho: float,
    alpha: float,
    method: CorrectionMethod,
    family_size: int,
    n_simulations: int,
    rng: np.random.Generator,
    limiting: bool = False,
) -> PowerPoint:
    """Simulate one design point and return the power the whole pipeline actually achieved.

    Each replicate: generate a dataset, fit it with the REAL
    :func:`~margins.metrics.estimator.estimate_pair` under ``cov_type="cluster"`` and
    ``n_resamples=0``, compute a cluster-robust Wald p for every free interaction contrast of the
    pair, and correct over the full family.

    **The rest of the family is drawn as independent uniforms.** Only the simulated pair's terms
    are fitted; the other confirmatory pairs' members are true nulls under the planted design, and
    a true null p-value is uniform. This is a stated approximation: the real family's nulls are
    correlated through the shared prompt set, which moves the *variance* of the BH threshold rather
    than its mean, in no systematic direction. Simulating all six pairs per replicate would
    multiply the grid's cost by six to refine a second-order term.

    Detection is ``q <= alpha`` on the planted target term, which is the first cell of the design
    and is always inside the free contrast set.
    """
    from statsmodels.stats.multitest import multipletests

    design = power_design.design
    gamma = interaction_table(design, effect_size=effect_size, pattern=pattern)
    means = cell_mean_table(design, gamma)
    layout = _layout(power_design, limiting=limiting)

    free_terms = free_interaction_terms(design)
    if family_size < len(free_terms):
        raise ValueError(
            f"family_size={family_size} is smaller than the simulated pair's own "
            f"{len(free_terms)} free interaction contrasts; the family must contain the pair."
        )
    target_term = interaction_term(
        design.axis_a, design.levels_a[0], design.axis_b, design.levels_b[0]
    )
    target_index = free_terms.index(target_term)
    target_gamma = float(gamma[0, 0])
    n_companions = family_size - len(free_terms)
    df = layout.n_items - 1

    detected = 0
    ses: list[float] = []
    estimates: list[float] = []
    for _ in range(n_simulations):
        frame = simulate_frame(
            power_design, means, icc_rho=icc_rho, rng=rng, limiting=limiting, layout=layout
        )
        fit = estimate_pair(
            frame,
            design,
            cov_type="cluster",
            n_resamples=0,
            ci_level=_UNUSED_CI_LEVEL,
            rng=None,
        )
        by_term = {term.term: term for term in fit.terms}
        pair_p = np.array([wald_p(by_term[t].estimate, by_term[t].se, df=df) for t in free_terms])
        companions = rng.uniform(0.0, 1.0, size=n_companions)
        _, q_values, _, _ = multipletests(
            np.concatenate([pair_p, companions]), alpha=alpha, method=method
        )
        if float(q_values[target_index]) <= alpha:
            detected += 1
        ses.append(by_term[target_term].se)
        estimates.append(by_term[target_term].estimate)

    low, high = wilson_interval(detected, n_simulations)
    return PowerPoint(
        axis_a=design.axis_a,
        axis_b=design.axis_b,
        pattern=pattern,
        effect_size=effect_size,
        icc_rho=icc_rho,
        frames_per_cell=power_design.frames_per_cell,
        seeds_per_item=power_design.seeds_per_item,
        renderings_per_cell=power_design.renderings_per_cell,
        items_per_cell=power_design.items_per_cell,
        n_item_clusters=layout.n_items,
        n_obs=len(layout.item),
        alpha=alpha,
        method=method,
        family_size=family_size,
        n_simulations=n_simulations,
        n_detected=detected,
        power=detected / n_simulations,
        power_ci_low=low,
        power_ci_high=high,
        target_term=target_term,
        target_gamma=target_gamma,
        mean_se=float(np.mean(ses)),
        mean_estimate=float(np.mean(estimates)),
        limiting=limiting,
    )


def ceiling_point(
    design: PairDesign,
    *,
    frames_per_cell: int,
    pattern: EffectPattern,
    effect_size: float,
    icc_rho: float,
    alpha: float,
    method: CorrectionMethod,
    family_size: int,
    n_simulations: int,
    rng: np.random.Generator,
) -> PowerPoint:
    """Power as seeds -> infinity at this frame count. **No seed budget can exceed it.**

    Simulated, not extrapolated: the limit of ``y = mu + u_item + e`` averaged over infinitely many
    seeds is one observation per cluster at ``y = mu + u_item``, which is exactly what
    :func:`simulate_frame` produces under ``limiting=True``.
    """
    return simulate_power_point(
        PowerDesign(design=design, frames_per_cell=frames_per_cell, seeds_per_item=1),
        pattern=pattern,
        effect_size=effect_size,
        icc_rho=icc_rho,
        alpha=alpha,
        method=method,
        family_size=family_size,
        n_simulations=n_simulations,
        rng=rng,
        limiting=True,
    )


# --------------------------------------------------------------------------------------
# the curve and its minimum
# --------------------------------------------------------------------------------------


def power_curve(
    design: PairDesign,
    *,
    frames_per_cell: int,
    seeds_grid: Sequence[int],
    pattern: EffectPattern,
    effect_size: float,
    icc_rho: float,
    alpha: float,
    method: CorrectionMethod,
    family_size: int,
    target_power: float,
    n_simulations: int,
    rng: np.random.Generator,
    early_exit: bool = True,
    search: SearchStrategy = "bisect",
    search_simulations: int | None = None,
    progress: ProgressHook | None = None,
) -> PowerResult:
    """Sweep ``seeds_grid`` at one frame count and read off the minimum that reaches the target.

    The ceiling is simulated **first**. If it sits below ``target_power`` the seeds sweep is
    skipped entirely and the result says so: no number of seeds can reach a target the cluster
    count already rules out, and spending the grid's budget to demonstrate that one point at a time
    would be waste.

    **``search="bisect"`` is the default and it is a cost decision, not an approximation.** The
    cost of one design point grows linearly in ``seeds``, so a linear walk spends most of its
    budget on the large points it passes through on the way up -- and the whole grid can run to
    hours. Power is monotone in seeds (asserted by the test suite), so a bisection over the grid
    finds the same minimum in ``log2(|grid|)`` points instead of up to ``|grid|``. The chosen
    point is then **re-simulated at full ``n_simulations``**, and the grid point below it is
    checked to be under target, so a noisy coarse probe cannot promote a design that does not
    actually clear the bar.

    ``search="linear"`` keeps the old ascending walk, which returns the whole curve rather than a
    bracket; ``early_exit=False`` with it produces every point, for a report or a plot.

    Args:
        search_simulations: replicates for the coarse search probes. Defaults to ``n_simulations``;
            a smaller value is the other half of the speedup, and does not affect the reported
            power, which always comes from a full-precision re-simulation.
        progress: called with a one-line status before each simulated point, so a long grid is
            observable rather than silent.
    """
    coarse = search_simulations or n_simulations
    ceiling = ceiling_point(
        design,
        frames_per_cell=frames_per_cell,
        pattern=pattern,
        effect_size=effect_size,
        icc_rho=icc_rho,
        alpha=alpha,
        method=method,
        family_size=family_size,
        n_simulations=n_simulations,
        rng=rng,
    )

    def result(points: Sequence[PowerPoint], seeds: int | None, reason: str | None) -> PowerResult:
        renderings = None if seeds is None else frames_per_cell * seeds
        return PowerResult(
            axis_a=design.axis_a,
            axis_b=design.axis_b,
            pattern=pattern,
            effect_size=effect_size,
            icc_rho=icc_rho,
            frames_per_cell=frames_per_cell,
            target_power=target_power,
            min_seeds_per_item=seeds,
            min_renderings_per_cell=renderings,
            min_cell_n=renderings,
            power_ceiling=ceiling.power,
            reason=reason,
            curve=tuple(points),
        )

    if ceiling.power < target_power:
        return result(
            [ceiling],
            None,
            (
                f"unreachable at {frames_per_cell} template frames per cell: power converges to "
                f"{ceiling.power:.3f} as seeds grow (95% CI [{ceiling.power_ci_low:.3f}, "
                f"{ceiling.power_ci_high:.3f}]), below the target of {target_power:.2f}. Item "
                f"clusters, not renderings, are the binding constraint -- a cell has "
                f"{frames_per_cell} clusters however many seeds are drawn, so the lever is more "
                "template frames, a larger target effect, or a narrower family."
            ),
        )

    grid = sorted(set(seeds_grid))
    points: list[PowerPoint] = []

    def probe(seeds: int, replicates: int) -> PowerPoint:
        if progress is not None:
            progress(
                f"    frames={frames_per_cell:<3} seeds={seeds:<4} "
                f"rows={frames_per_cell * design.n_cells * seeds:<7} x{replicates}"
            )
        point = simulate_power_point(
            PowerDesign(design=design, frames_per_cell=frames_per_cell, seeds_per_item=seeds),
            pattern=pattern,
            effect_size=effect_size,
            icc_rho=icc_rho,
            alpha=alpha,
            method=method,
            family_size=family_size,
            n_simulations=replicates,
            rng=rng,
        )
        points.append(point)
        return point

    found: int | None = None
    if search == "linear":
        for seeds in grid:
            if probe(seeds, n_simulations).power >= target_power:
                found = seeds
                if early_exit:
                    break
    else:
        # Bisect on the monotone power curve: `low` is the largest grid point known to be under
        # target, `high` the smallest known to reach it.
        low, high = 0, len(grid)
        while low < high:
            middle = (low + high) // 2
            if probe(grid[middle], coarse).power >= target_power:
                high = middle
            else:
                low = middle + 1
        if low < len(grid):
            candidate = grid[low]
            # Re-simulate at full precision, and confirm the point below really is under target.
            # A coarse probe that got lucky cannot promote a design that does not clear the bar.
            if probe(candidate, n_simulations).power >= target_power:
                found = candidate
            else:
                for seeds in grid[low + 1 :]:
                    if probe(seeds, n_simulations).power >= target_power:
                        found = seeds
                        break
    points.append(ceiling)

    if found is None:
        return result(
            points,
            None,
            (
                f"no seed count in {grid} reached power {target_power:.2f} at "
                f"{frames_per_cell} frames per cell, although the ceiling ({ceiling.power:.3f}) "
                "is above it. Extend seeds_grid."
            ),
        )
    return result(points, found, None)


def minimum_renderings(result: PowerResult) -> int | None:
    """The minimum renderings per cell a curve reached, or ``None`` when it reached none."""
    return result.min_renderings_per_cell


def power_settings_grid(
    settings: PowerSettings,
) -> tuple[tuple[EffectPattern, float, int], ...]:
    """Every ``(pattern, rho, frames)`` the shipped settings ask for, in report order."""
    return tuple(
        (pattern, rho, frames)
        for pattern in settings.patterns
        for rho in settings.icc_rho
        for frames in settings.frames_grid
    )
