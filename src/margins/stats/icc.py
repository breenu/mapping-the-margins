"""The intraclass correlation: what share of harm-score variance sits BETWEEN prompts (P9).

The Stage-A power analysis showed the whole feasibility question turns on one number that no
existing artifact contains. The estimator clusters by item (the prompt) and the sampled seeds are
repeated observations inside that cluster, so the variance of a cell mean is::

    rho / frames  +  (1 - rho) / (frames * seeds)

``rho`` is the between-item share. At ``rho = 0.05`` the shipped nine template frames reach 80%
power on a 0.3 SD interaction; at ``rho = 0.2`` they cannot reach it at any seed budget. **It was
assumed in Stage A because nothing could measure it** -- the dev run has one response per prompt,
so it contains no within-prompt variance at all. This module is what turns it into a measurement.

**The estimand, precisely: the between-item variance share WITHIN a cell.** Not the raw between-
prompt share over the whole corpus. Prompts in different cells differ because the cells differ,
and that between-cell variation is *signal* the model fits, not the noise the cluster-robust
standard error is defending against. Pooling it in would inflate ``rho`` and make the design look
worse than it is. So cells enter as a fixed factor and the random item effect is nested inside
them:

    y_cij = mu_c + u_ci + e_cij,   u ~ N(0, var_between),  e ~ N(0, var_within)

**Scale invariance, stated because it licenses a shortcut.** ``rho`` is a ratio of variances, so
it is unchanged by any affine transform of the score. Computing it on P6's ``standardized`` column
and computing it on ``raw`` give the same answer, and the within-benchmark z-scoring P6 applies
cannot move it. Standardized is used because that is the scale the estimator and the power
simulation both work on.

**A degenerate design is reported, never returned as a NaN** -- the same rule P7's estimator
follows. A cell with one item, or a series with no within-item variation, yields
``IccEstimate(estimable=False, reason=...)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["IccEstimate", "IccObservation", "estimate_icc", "icc_interval"]


@dataclass(frozen=True)
class IccObservation:
    """One scored generation: which cell it belongs to, which prompt produced it, and its score."""

    cell_id: str
    item_id: str
    value: float


class IccEstimate(BaseModel):
    """The measured between-item variance share for one series, with an exact interval.

    ``rho`` is what the power simulation's ``icc_rho`` was assuming. ``n0`` is the effective
    observations per item; in a balanced design it is exactly the seeds per prompt, and it is
    reported because the interval's width depends on it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    harm_type: str = Field(min_length=1)
    scorer_id: str | None = Field(
        default=None, description="None for the triangulated (mean-of-scorers) series."
    )
    condition: str = "model"

    estimable: bool
    reason: str | None = Field(default=None, description="Why not, when estimable is False.")

    n_obs: int = Field(ge=0)
    n_items: int = Field(ge=0)
    n_cells: int = Field(ge=0)

    rho: float | None = Field(default=None, ge=0.0, le=1.0)
    ci_low: float | None = Field(default=None, ge=0.0, le=1.0)
    ci_high: float | None = Field(default=None, ge=0.0, le=1.0)
    ci_level: float = Field(default=0.95, gt=0.0, lt=1.0)

    var_between: float | None = Field(default=None, ge=0.0)
    var_within: float | None = Field(default=None, ge=0.0)
    ms_between: float | None = Field(default=None, ge=0.0)
    ms_within: float | None = Field(default=None, ge=0.0)
    df_between: int = Field(default=0, ge=0)
    df_within: int = Field(default=0, ge=0)
    n0: float | None = Field(
        default=None, gt=0.0, description="Effective observations per item; = seeds if balanced."
    )

    @model_validator(mode="after")
    def _consistent(self) -> IccEstimate:
        if not self.estimable:
            if self.reason is None:
                raise ValueError("an estimate that is not estimable must say why")
            if self.rho is not None:
                raise ValueError("an estimate that is not estimable cannot carry a rho")
            return self
        if self.rho is None:
            raise ValueError("an estimable result must carry a rho")
        if self.ci_low is None or self.ci_high is None:
            raise ValueError("an estimable result must carry an interval")
        if self.ci_low > self.ci_high:
            raise ValueError("ci_low must be <= ci_high")
        return self


def icc_interval(
    *, f_statistic: float, df_between: int, df_within: int, n0: float, level: float = 0.95
) -> tuple[float, float]:
    """Exact interval for the intraclass correlation (Searle), from the F ratio.

    Built from the F distribution rather than from a normal approximation on ``rho``. ``rho`` is
    bounded in ``[0, 1]`` and its sampling distribution is badly skewed near the boundaries --
    which is exactly where the design decision sits, since 0.05 and 0.2 are the values that
    separate "feasible at nine frames" from "not feasible at any seed budget".
    """
    from scipy import stats

    if df_between < 1 or df_within < 1:
        raise ValueError(
            f"an interval needs positive degrees of freedom, got between={df_between}, "
            f"within={df_within}"
        )
    if n0 <= 1.0:
        raise ValueError(f"n0 must exceed 1 for an interval to be defined, got {n0}")
    tail = (1.0 - level) / 2.0
    upper_quantile = float(stats.f.ppf(1.0 - tail, df_between, df_within))
    lower_quantile = float(stats.f.ppf(tail, df_between, df_within))
    bounded_low = f_statistic / upper_quantile
    bounded_high = f_statistic / lower_quantile
    low = (bounded_low - 1.0) / (bounded_low + n0 - 1.0)
    high = (bounded_high - 1.0) / (bounded_high + n0 - 1.0)
    return float(np.clip(low, 0.0, 1.0)), float(np.clip(high, 0.0, 1.0))


def estimate_icc(
    observations: Sequence[IccObservation],
    *,
    harm_type: str,
    scorer_id: str | None = None,
    condition: str = "model",
    level: float = 0.95,
) -> IccEstimate:
    """Between-item-within-cell variance share, by nested one-way ANOVA.

    Cells are a fixed factor and items are random within them, so between-cell differences -- the
    signal the model fits -- never enter ``rho``. Handles an unbalanced design through the standard
    ``n0`` coefficient, which reduces to the seeds per prompt when every item has the same count.

    Returns:
        An :class:`IccEstimate`. A design that cannot support the estimate comes back with
        ``estimable=False`` and a named reason, never a NaN.
    """

    def refused(reason: str, **counts: int) -> IccEstimate:
        return IccEstimate(
            harm_type=harm_type,
            scorer_id=scorer_id,
            condition=condition,
            estimable=False,
            reason=reason,
            ci_level=level,
            **counts,
        )

    if not observations:
        return refused("no observations", n_obs=0, n_items=0, n_cells=0)

    by_cell: dict[str, dict[str, list[float]]] = {}
    for observation in observations:
        by_cell.setdefault(observation.cell_id, {}).setdefault(observation.item_id, []).append(
            observation.value
        )

    n_obs = len(observations)
    n_cells = len(by_cell)
    n_items = sum(len(items) for items in by_cell.values())
    counts = {"n_obs": n_obs, "n_items": n_items, "n_cells": n_cells}

    thin = sorted(cell for cell, items in by_cell.items() if len(items) < 2)
    if thin:
        return refused(
            f"{len(thin)} cell(s) carry fewer than two items, e.g. {thin[:3]}. A between-item "
            "variance needs at least two items inside a cell; with one, item and cell are the "
            "same thing and the two variance components are not separable.",
            **counts,
        )

    df_between = n_items - n_cells
    df_within = n_obs - n_items
    if df_within < 1:
        return refused(
            f"no within-item degrees of freedom ({n_obs} observations over {n_items} items). "
            "Every prompt was generated once, so there is no within-prompt variance to measure -- "
            "this is precisely the gap the sampled rho run exists to fill.",
            **counts,
        )

    ss_between = 0.0
    ss_within = 0.0
    sum_of_squared_sizes = 0.0
    for items in by_cell.values():
        sizes = np.array([len(values) for values in items.values()], dtype=float)
        item_means = np.array([float(np.mean(values)) for values in items.values()])
        cell_total = float(sum(sum(values) for values in items.values()))
        cell_n = float(sizes.sum())
        cell_mean = cell_total / cell_n
        ss_between += float(np.sum(sizes * (item_means - cell_mean) ** 2))
        for values in items.values():
            array = np.asarray(values, dtype=float)
            ss_within += float(np.sum((array - array.mean()) ** 2))
        sum_of_squared_sizes += float(np.sum(sizes**2) / cell_n)

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    # The standard nested coefficient. Balanced (every item with S observations) gives n0 == S.
    n0 = (n_obs - sum_of_squared_sizes) / df_between

    if ms_within <= 0.0:
        return refused(
            "within-item variance is exactly zero: every seed of a prompt produced the same "
            "score. Under sampled decoding that means the generations did not actually vary, so "
            "the run measured greedy behaviour and rho is not defined from it.",
            **counts,
        )
    if n0 <= 1.0:
        return refused(
            f"effective observations per item is {n0:.3f}, which cannot support an interval. "
            "Each prompt needs more than one generation.",
            **counts,
        )

    var_between = max(0.0, (ms_between - ms_within) / n0)
    var_within = ms_within
    rho = var_between / (var_between + var_within)
    low, high = icc_interval(
        f_statistic=ms_between / ms_within,
        df_between=df_between,
        df_within=df_within,
        n0=n0,
        level=level,
    )
    return IccEstimate(
        harm_type=harm_type,
        scorer_id=scorer_id,
        condition=condition,
        estimable=True,
        reason=None,
        n_obs=n_obs,
        n_items=n_items,
        n_cells=n_cells,
        rho=float(rho),
        ci_low=low,
        ci_high=high,
        ci_level=level,
        var_between=float(var_between),
        var_within=float(var_within),
        ms_between=float(ms_between),
        ms_within=float(ms_within),
        df_between=df_between,
        df_within=df_within,
        n0=float(n0),
    )
