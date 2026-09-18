"""The interaction estimator: OLS with item-clustered SEs and a cluster bootstrap over items (P7).

Three pre-registration facts are implemented here and nowhere else:

* **Identity link, ordinary least squares.** The response is the standardized harm score
  (higher = more harm, SD units), so no transform is applied.
* **Standard errors are cluster-robust, clustered by ITEM (the prompt).** The ~50 sampled
  generations per prompt are repeated observations within one item; a naive iid standard error
  would count them as new information and be anti-conservative by roughly the design effect
  ``1 + (m - 1) * rho``. Template-level clustering is deferred to P9's robustness checks and is
  deliberately NOT in the primary model. ``statsmodels`` computes the sandwich
  (``cov_type="cluster"``); the test suite recomputes it by hand for a small case so a mis-passed
  ``groups`` argument cannot silently return an iid covariance under the cluster label.
* **Confidence intervals are a percentile cluster bootstrap whose resampling unit is the item,
  resampled with replacement WITHIN each cell.** Whole items move together, so the within-item
  dependence is preserved in every replicate; stratifying by cell keeps every cell populated in
  every replicate, so the design stays identified (an unstratified resample of ~990 prompts drops
  an entire cell with probability about ``e^-9`` per cell per replicate -- rare, but a replicate
  with an unidentified interaction is not a replicate). Replicates are refitted with numpy least
  squares on the same design matrix for speed; the tests hold that path to the statsmodels point
  fit.

**Seeds are never pre-averaged.** The fit sees every sampled observation; aggregation across seeds
happens through the model and the cluster structure, not by collapsing rows first -- collapsing
would discard the within-item variance the clustering exists to account for.

**Not estimable is a result, not an exception to hide.** A pair whose design has an empty cell, a
single cluster, or no residual degrees of freedom raises :class:`NotEstimableError` with the reason
named; the metric layer records it. Nothing here ever emits a NaN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

from margins.metrics.design import PairDesign
from margins.metrics.records import TermKind

__all__ = [
    "COV_TYPES",
    "CovType",
    "NotEstimableError",
    "PairFit",
    "PairFrame",
    "TermEstimate",
    "estimate_pair",
    "lstsq_params",
    "percentile_interval",
]

#: ``cluster`` is the primary model. ``nonrobust`` exists so the test suite can DEMONSTRATE that
#: the clustered SE is wider than the iid one -- it must never be used for a reported number.
CovType = Literal["cluster", "nonrobust"]
COV_TYPES: Final[tuple[CovType, ...]] = ("cluster", "nonrobust")


class NotEstimableError(ValueError):
    """The pair cannot be fitted on this data; the message names why."""


@dataclass(frozen=True, eq=False)
class PairFrame:
    """The observations for one pair fit: response, the two axis values, and the ITEM of each.

    ``item`` is the cluster key. For model output it is the prompt id; for the drift control it is
    the control text id (one observation per text).
    """

    y: np.ndarray
    value_a: tuple[str, ...]
    value_b: tuple[str, ...]
    item: tuple[str, ...]

    def __post_init__(self) -> None:
        n = len(self.y)
        if not (len(self.value_a) == len(self.value_b) == len(self.item) == n):
            raise ValueError("y, value_a, value_b and item must have the same length")
        if self.y.ndim != 1:
            raise ValueError("y must be one-dimensional")

    @property
    def n_obs(self) -> int:
        return len(self.y)


@dataclass(frozen=True)
class TermEstimate:
    """One reported term: estimate, cluster-robust SE, and (if bootstrapped) a percentile CI."""

    term: str
    kind: TermKind
    value_a: str | None
    value_b: str | None
    estimate: float
    se: float
    ci_low: float | None
    ci_high: float | None


@dataclass(frozen=True, eq=False)
class PairFit:
    """A fitted pair: parameters, covariance, every reported term, and the interaction magnitude."""

    design: PairDesign
    cov_type: CovType
    n_obs: int
    n_items: int
    df_resid: int
    params: np.ndarray
    cov: np.ndarray
    terms: tuple[TermEstimate, ...]
    rms_interaction: float
    rms_ci: tuple[float, float] | None
    n_resamples: int
    cell_counts: dict[str, int]

    @property
    def interactions(self) -> tuple[TermEstimate, ...]:
        return tuple(t for t in self.terms if t.kind == "interaction")

    def interaction(self, value_a: str, value_b: str) -> TermEstimate:
        for term in self.terms:
            if term.kind == "interaction" and (term.value_a, term.value_b) == (value_a, value_b):
                return term
        raise KeyError(f"no interaction term for ({value_a!r}, {value_b!r})")

    def main(self, axis: str, value: str) -> TermEstimate:
        if axis == self.design.axis_a:
            kind, key = "main_a", value
        elif axis == self.design.axis_b:
            kind, key = "main_b", value
        else:
            raise KeyError(f"axis {axis!r} is not in {self.design.axis_pair}")
        for term in self.terms:
            if term.kind == kind and (term.value_a if kind == "main_a" else term.value_b) == key:
                return term
        raise KeyError(f"no main-effect term for {axis}={value}")


def lstsq_params(design_matrix: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS parameters by least squares. The bootstrap refit path; held to statsmodels by test."""
    params, _, _, _ = np.linalg.lstsq(design_matrix, y, rcond=None)
    return np.asarray(params, dtype=float)


def percentile_interval(samples: np.ndarray, ci_level: float) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided percentile bounds along axis 0."""
    tail = 100.0 * (1.0 - ci_level) / 2.0
    low, high = np.percentile(samples, [tail, 100.0 - tail], axis=0)
    return np.asarray(low, dtype=float), np.asarray(high, dtype=float)


def _check_estimable(frame: PairFrame, design: PairDesign) -> dict[str, int]:
    """Raise :class:`NotEstimableError` unless the pair can be fitted; return per-cell counts."""
    counts: dict[str, int] = {cell_id: 0 for _, _, cell_id in design.cells}
    for a, b in zip(frame.value_a, frame.value_b, strict=True):
        counts[design.cell_id(a, b)] += 1

    empty = [cell_id for cell_id, n in counts.items() if n == 0]
    if empty:
        raise NotEstimableError(
            f"axis pair {design.axis_pair} has {len(empty)} empty cell(s): {empty}. An interaction "
            "over a table with a missing cell is not identified."
        )
    n_items = len(set(frame.item))
    if n_items < 2:
        raise NotEstimableError(
            f"axis pair {design.axis_pair} has {n_items} item cluster(s); a cluster-robust "
            "standard error needs at least two."
        )
    df_resid = frame.n_obs - design.n_params
    if df_resid < 1:
        raise NotEstimableError(
            f"axis pair {design.axis_pair} has no residual degrees of freedom "
            f"({frame.n_obs} observations for {design.n_params} parameters); no standard error "
            "is definable."
        )
    return counts


def estimate_pair(
    frame: PairFrame,
    design: PairDesign,
    *,
    cov_type: CovType = "cluster",
    n_resamples: int,
    ci_level: float,
    rng: np.random.Generator | None,
) -> PairFit:
    """Fit ``harm ~ A + B + A:B`` on ``frame`` and bootstrap its CIs.

    Args:
        frame: the observations. Every ``(value_a, value_b)`` must be a cell of ``design``.
        design: the pair's sum-to-zero design.
        cov_type: ``"cluster"`` (the primary model, clustered by ``frame.item``) or
            ``"nonrobust"`` (iid, for the demonstration in the tests only).
        n_resamples: bootstrap replicates; ``0`` skips the bootstrap and leaves the CIs ``None``.
        ci_level: two-sided percentile CI level.
        rng: the generator the bootstrap draws from. Required when ``n_resamples > 0``; it must
            come from :func:`margins.repro.set_global_seed` so the path has no hidden global RNG.

    Raises:
        NotEstimableError: if a cell is empty, there are fewer than two items, or the residual
            degrees of freedom are zero.
    """
    import statsmodels.api as sm

    if cov_type not in COV_TYPES:
        raise ValueError(f"cov_type must be one of {COV_TYPES}, got {cov_type!r}")
    if n_resamples > 0 and rng is None:
        raise ValueError("a bootstrap needs a seeded generator; pass rng from set_global_seed")

    counts = _check_estimable(frame, design)
    design_matrix = design.matrix(frame.value_a, frame.value_b)
    y = np.asarray(frame.y, dtype=float)
    _, item_codes = np.unique(np.asarray(frame.item), return_inverse=True)

    model = sm.OLS(y, design_matrix)
    if cov_type == "cluster":
        result = model.fit(cov_type="cluster", cov_kwds={"groups": item_codes})
    else:
        result = model.fit()
    params = np.asarray(result.params, dtype=float)
    cov = np.asarray(result.cov_params(), dtype=float)

    contrasts = design.contrasts
    contrast_matrix = design.contrast_matrix
    estimates = contrast_matrix @ params
    ses = np.sqrt(
        np.clip(np.einsum("ij,jk,ik->i", contrast_matrix, cov, contrast_matrix), 0.0, None)
    )
    is_interaction = np.array([c.kind == "interaction" for c in contrasts])
    rms = float(np.sqrt(np.mean(estimates[is_interaction] ** 2)))

    ci_low: np.ndarray | None = None
    ci_high: np.ndarray | None = None
    rms_ci: tuple[float, float] | None = None
    if n_resamples > 0:
        assert rng is not None
        replicates = _bootstrap(
            design_matrix, y, frame, design, contrast_matrix, n_resamples=n_resamples, rng=rng
        )
        ci_low, ci_high = percentile_interval(replicates, ci_level)
        rms_samples = np.sqrt(np.mean(replicates[:, is_interaction] ** 2, axis=1))
        low, high = percentile_interval(rms_samples, ci_level)
        rms_ci = (float(low), float(high))

    terms = tuple(
        TermEstimate(
            term=c.term,
            kind=c.kind,
            value_a=c.value_a,
            value_b=c.value_b,
            estimate=float(estimates[k]),
            se=float(ses[k]),
            ci_low=None if ci_low is None else float(ci_low[k]),
            ci_high=None if ci_high is None else float(ci_high[k]),
        )
        for k, c in enumerate(contrasts)
    )
    return PairFit(
        design=design,
        cov_type=cov_type,
        n_obs=frame.n_obs,
        n_items=int(item_codes.max()) + 1,
        df_resid=frame.n_obs - design.n_params,
        params=params,
        cov=cov,
        terms=terms,
        rms_interaction=rms,
        rms_ci=rms_ci,
        n_resamples=n_resamples,
        cell_counts=counts,
    )


def _bootstrap(
    design_matrix: np.ndarray,
    y: np.ndarray,
    frame: PairFrame,
    design: PairDesign,
    contrast_matrix: np.ndarray,
    *,
    n_resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Cluster bootstrap over ITEMS, stratified by CELL. Returns ``(n_resamples, n_terms)``.

    Each replicate draws, within every cell, as many items as the cell has, with replacement, and
    takes every row of each drawn item. Whole items move together (the within-item dependence is
    preserved); every cell keeps its item count (the design stays identified).
    """
    cell_of_row = [design.cell_id(a, b) for a, b in zip(frame.value_a, frame.value_b, strict=True)]
    # cell_id -> list of row-index arrays, one per item in that cell.
    rows_by_cell_item: dict[str, dict[str, list[int]]] = {}
    for index, (cell_id, item) in enumerate(zip(cell_of_row, frame.item, strict=True)):
        rows_by_cell_item.setdefault(cell_id, {}).setdefault(item, []).append(index)
    strata = [
        [np.asarray(rows, dtype=int) for _, rows in sorted(items.items())]
        for _, items in sorted(rows_by_cell_item.items())
    ]

    out = np.empty((n_resamples, contrast_matrix.shape[0]))
    for r in range(n_resamples):
        pieces: list[np.ndarray] = []
        for items in strata:
            draws = rng.integers(0, len(items), size=len(items))
            pieces.extend(items[d] for d in draws)
        chosen = np.concatenate(pieces)
        out[r] = contrast_matrix @ lstsq_params(design_matrix[chosen], y[chosen])
    return out
