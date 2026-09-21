"""The emergence metric: ``Metric.compute`` on the model condition (P7, section 10).

This is the stage that turns :class:`~margins.schemas.ItemScore` rows into the two records the
paper is written against, and it keeps them apart on purpose:

* :class:`~margins.schemas.CellStat` -- **descriptive**. Per cell: n renderings, the mean
  standardized harm, a cluster-bootstrap CI over items, and ``powered``. A high mean here is a
  high LEVEL. It is not emergence and carries no ``p``.
* :class:`~margins.schemas.EffectEstimate` -- **inferential**. Per axis pair and term: the
  coefficients of ``harm_std ~ A + B + A:B`` fitted over the pair's order-2 cells. The
  ``interaction`` terms ARE the emergent intersectional bias. ``p``, ``q``, ``family_size`` and
  ``correction`` are left unset; only ``StatsEngine`` (P9) may fill them (invariant 3).

**What ``harm_std`` is.** Every harm type has at least two scorers, each producing its own
``standardized`` score per response. The harm-type-level series fitted into ``EffectEstimate`` is
the **equal-weight mean of the scorers' standardized scores per response** (decided 18 Sep 2026):
each leg is already z-scored within benchmark x scorer x condition, so the mean lives on one scale.
A response missing any of its harm type's scorers is **refused**, not imputed -- a row scored by one
classifier of two would be single-classifier-dependent (framing constraint 6). The per-scorer fits
are ALSO produced, as :class:`~margins.metrics.records.FitTerm` / :class:`FitReport` rows, because
the drift comparison is scorer-specific: a scorer's own drift interaction is compared with *that
scorer's* view of the model, not with the triangulated one.

**Aggregation across seeds happens here, through the model.** Each response is one row; each
prompt is one cluster. Nothing is averaged before fitting.

**``compute`` fits the model condition only.** Rows with ``condition="drift_control"`` are counted
and reported as skipped; the drift series has its own fit in :mod:`margins.metrics.drift`, on the
P6 artifact that carries the axis values. Mixing the two conditions into one fit would be the
point-subtraction section 9 forbids, arrived at by accident.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from margins.metrics.config import MetricsConfig
from margins.metrics.design import PairDesign, pair_designs
from margins.metrics.estimator import (
    NotEstimableError,
    PairFit,
    PairFrame,
    estimate_pair,
    percentile_interval,
)
from margins.metrics.records import FitReport, FitTerm
from margins.repro import set_global_seed
from margins.schemas import CellStat, Condition, EffectEstimate, IdentityCell, ItemScore

__all__ = [
    "EmergenceMetric",
    "EmergenceResult",
    "Observation",
    "SeriesKey",
    "fit_report",
    "fit_terms",
    "not_estimable_report",
    "triangulate",
]


@dataclass(frozen=True)
class Observation:
    """One response after triangulation: the per-scorer scores and their mean."""

    response_id: str
    prompt_id: str
    cell_id: str
    by_scorer: dict[str, float]
    combined: float


@dataclass
class EmergenceResult:
    """Everything ``compute_result`` produces. ``compute`` returns the first two."""

    cell_stats: list[CellStat] = field(default_factory=list)
    effects: list[EffectEstimate] = field(default_factory=list)
    fit_terms: list[FitTerm] = field(default_factory=list)
    fit_reports: list[FitReport] = field(default_factory=list)
    n_skipped_rows: int = 0

    @property
    def pair(self) -> tuple[list[CellStat], list[EffectEstimate]]:
        return self.cell_stats, self.effects


class EmergenceMetric:
    """Satisfies :class:`~margins.interfaces.Metric`.

    Args:
        config: bootstrap settings and drift thresholds.
        min_cell_n: the pre-registered minimum renderings per cell. ``powered`` is computed from
            the ACTUAL n of each cell against it -- never from the lattice's projection. Passed in
            rather than read from a lattice config so the metric has one source for it and a test
            can vary it.
    """

    def __init__(self, config: MetricsConfig, *, min_cell_n: int) -> None:
        if min_cell_n < 1:
            raise ValueError("min_cell_n must be at least 1")
        self._config = config
        self._min_cell_n = min_cell_n

    @property
    def config(self) -> MetricsConfig:
        return self._config

    @property
    def min_cell_n(self) -> int:
        return self._min_cell_n

    # ---------------------------------------------------------------- the Metric Protocol

    def compute(
        self, scores: Sequence[ItemScore], cells: Sequence[IdentityCell]
    ) -> tuple[list[CellStat], list[EffectEstimate]]:
        """Return ``(cell_stats, effect_estimates)``, uncorrected."""
        return self.compute_result(scores, cells).pair

    # ---------------------------------------------------------------- the full result

    def compute_result(
        self, scores: Sequence[ItemScore], cells: Sequence[IdentityCell]
    ) -> EmergenceResult:
        """Fit everything and return the full result, including per-scorer fits and reports."""
        _, rng = set_global_seed(self._config.bootstrap.seed)
        result = EmergenceResult()

        model_rows = [s for s in scores if s.condition == "model"]

        # A NULL cell_id is a legal, documented state on the shared record -- "None where a source
        # attests no identity cell" -- and it is skipped, not refused.
        #
        # It first arose at P2. The generation vertical injects a cell into every prompt, so every
        # row it produces has one; the extractive vertical DERIVES the cell from its source item,
        # and most CrowS-Pairs items attest no lattice value at all (their descriptors are not
        # lattice values, or their category has no lattice axis). Those rows are real measurements
        # of real sentence pairs that the lattice simply cannot place.
        #
        # Counting them beside the drift-control rows is the honest handling: they are reported in
        # `n_skipped_rows`, and the coverage report says why. Raising instead would have made an
        # expected property of the second paradigm look like a data error, and would have taken
        # the whole run down -- including the open_gen rows in the same file.
        placeable = [s for s in model_rows if s.cell_id is not None]
        result.n_skipped_rows = len(scores) - len(placeable)
        if not placeable:
            raise ValueError(
                "no model-condition rows with a cell to fit; the drift series is fitted separately,"
                " and rows whose source attests no identity cell cannot enter a design matrix"
            )

        by_cell = {cell.cell_id: cell for cell in cells}
        unknown = sorted({str(s.cell_id) for s in placeable if s.cell_id not in by_cell})
        if unknown:
            raise KeyError(
                f"{len(unknown)} score row(s) reference cells not in the lattice passed in, e.g. "
                f"{unknown[:3]}. The metric resolves cell_id -> axes through `cells`; a row it "
                "cannot place cannot be fitted. (A NULL cell_id is different and is skipped: it "
                "means the source attests no cell, not that it names an unknown one.)"
            )
        model_rows = placeable

        designs = pair_designs(cells)
        groups = sorted({(s.benchmark, s.model_id, s.harm_type) for s in model_rows})
        for benchmark, model_id, harm_type in groups:
            rows = [
                s
                for s in model_rows
                if (s.benchmark, s.model_id, s.harm_type) == (benchmark, model_id, harm_type)
            ]
            scorer_ids, observations = triangulate(rows)
            self._cell_stats(result, observations, cells, benchmark, model_id, harm_type, rng)
            for design in designs:
                self._fit_series(
                    result,
                    observations,
                    design,
                    by_cell,
                    scorer_ids,
                    benchmark=benchmark,
                    model_id=model_id,
                    harm_type=harm_type,
                    rng=rng,
                )
        return result

    # ---------------------------------------------------------------- pieces

    def _cell_stats(
        self,
        result: EmergenceResult,
        observations: Sequence[Observation],
        cells: Sequence[IdentityCell],
        benchmark: str,
        model_id: str,
        harm_type: str,
        rng: np.random.Generator,
    ) -> None:
        """Descriptive per-cell summaries, CI by cluster bootstrap over the cell's items."""
        settings = self._config.bootstrap
        by_cell: dict[str, list[Observation]] = {}
        for obs in observations:
            by_cell.setdefault(obs.cell_id, []).append(obs)

        for cell in cells:
            group = by_cell.get(cell.cell_id)
            if not group:
                continue  # an empty cell has nothing to describe; the pair report names it
            values = np.array([o.combined for o in group])
            mean = float(values.mean())

            items: dict[str, list[float]] = {}
            for obs in group:
                items.setdefault(obs.prompt_id, []).append(obs.combined)
            item_values = [np.asarray(v) for _, v in sorted(items.items())]
            replicates = np.empty(settings.n_resamples)
            for r in range(settings.n_resamples):
                draws = rng.integers(0, len(item_values), size=len(item_values))
                replicates[r] = float(np.concatenate([item_values[d] for d in draws]).mean())
            low, high = percentile_interval(replicates, settings.ci_level)

            result.cell_stats.append(
                CellStat(
                    cell_id=cell.cell_id,
                    benchmark=benchmark,
                    model=model_id,
                    harm_type=harm_type,
                    n=len(group),
                    mean_standardized=mean,
                    ci_low=float(low),
                    ci_high=float(high),
                    powered=len(group) >= self._min_cell_n,
                )
            )

    def _fit_series(
        self,
        result: EmergenceResult,
        observations: Sequence[Observation],
        design: PairDesign,
        by_cell: dict[str, IdentityCell],
        scorer_ids: Sequence[str],
        *,
        benchmark: str,
        model_id: str,
        harm_type: str,
        rng: np.random.Generator,
    ) -> None:
        """Fit the triangulated series (-> EffectEstimate) and each scorer's series (-> FitTerm)."""
        settings = self._config.bootstrap
        cell_ids = {cell_id for _, _, cell_id in design.cells}
        in_pair = [o for o in observations if o.cell_id in cell_ids]
        value_a = tuple(by_cell[o.cell_id].axes_map[design.axis_a] for o in in_pair)
        value_b = tuple(by_cell[o.cell_id].axes_map[design.axis_b] for o in in_pair)
        items = tuple(o.prompt_id for o in in_pair)

        counts: dict[str, int] = dict.fromkeys(cell_ids, 0)
        for obs in in_pair:
            counts[obs.cell_id] += 1
        powered = bool(in_pair) and all(n >= self._min_cell_n for n in counts.values())

        series: list[str | None] = [None, *scorer_ids]
        for scorer_id in series:
            y = np.array(
                [o.combined if scorer_id is None else o.by_scorer[scorer_id] for o in in_pair]
            )
            frame = PairFrame(y=y, value_a=value_a, value_b=value_b, item=items)
            key = SeriesKey(
                benchmark=benchmark,
                model_id=model_id,
                harm_type=harm_type,
                condition="model",
                scorer_id=scorer_id,
                control_family=None,
                axis_a=design.axis_a,
                axis_b=design.axis_b,
            )
            try:
                fit = estimate_pair(
                    frame,
                    design,
                    n_resamples=settings.n_resamples,
                    ci_level=settings.ci_level,
                    rng=rng,
                )
            except NotEstimableError as exc:
                result.fit_reports.append(
                    not_estimable_report(key, design, frame, reason=str(exc), powered=powered)
                )
                continue

            result.fit_reports.append(fit_report(fit, key, powered=powered))
            result.fit_terms.extend(fit_terms(fit, key))
            if scorer_id is None:
                result.effects.extend(_effects(fit, benchmark, model_id, harm_type, powered))


# --------------------------------------------------------------------------------------
# helpers shared with the drift fit
# --------------------------------------------------------------------------------------


def triangulate(rows: Sequence[ItemScore]) -> tuple[tuple[str, ...], list[Observation]]:
    """Mean of the scorers' standardized scores per response; refuse an incomplete response."""
    scorer_ids = tuple(sorted({s.scorer_id for s in rows}))
    by_response: dict[str, dict[str, ItemScore]] = {}
    for score in rows:
        slot = by_response.setdefault(score.response_id, {})
        if score.scorer_id in slot:
            raise ValueError(
                f"response {score.response_id!r} has two rows from scorer {score.scorer_id!r}; "
                "one score per (response, scorer) is the P6 contract."
            )
        slot[score.scorer_id] = score

    observations: list[Observation] = []
    for response_id, slot in by_response.items():
        missing = [sid for sid in scorer_ids if sid not in slot]
        if missing:
            raise ValueError(
                f"response {response_id!r} was scored by {len(slot)} of {len(scorer_ids)} "
                f"scorers (missing {missing}). The triangulated score is the mean over EVERY "
                "scorer of the harm type; a row missing one would be single-classifier-dependent "
                "(CLAUDE.md framing constraint 6). Refusing rather than imputing."
            )
        first = slot[scorer_ids[0]]
        by_scorer = {sid: slot[sid].standardized for sid in scorer_ids}
        observations.append(
            Observation(
                response_id=response_id,
                prompt_id=first.prompt_id,
                cell_id=str(first.cell_id),
                by_scorer=by_scorer,
                combined=float(np.mean(list(by_scorer.values()))),
            )
        )
    observations.sort(key=lambda o: (o.cell_id, o.prompt_id, o.response_id))
    return scorer_ids, observations


@dataclass(frozen=True)
class SeriesKey:
    """What identifies one fitted series: the grouping keys every FitTerm/FitReport carries."""

    benchmark: str
    model_id: str
    harm_type: str
    condition: Condition
    scorer_id: str | None
    control_family: str | None
    axis_a: str
    axis_b: str


def not_estimable_report(
    key: SeriesKey, design: PairDesign, frame: PairFrame, *, reason: str, powered: bool | None
) -> FitReport:
    """The report for a pair that could not be fitted: the reason is named, nothing is NaN."""
    return FitReport(
        benchmark=key.benchmark,
        model_id=key.model_id,
        harm_type=key.harm_type,
        condition=key.condition,
        scorer_id=key.scorer_id,
        control_family=key.control_family,
        axis_a=key.axis_a,
        axis_b=key.axis_b,
        estimable=False,
        reason=reason,
        n_obs=frame.n_obs,
        n_items=len(set(frame.item)),
        n_cells=design.n_cells,
        n_params=design.n_params,
        cov_type="cluster",
        n_resamples=0,
        powered=powered,
    )


def fit_report(fit: PairFit, key: SeriesKey, *, powered: bool | None) -> FitReport:
    return FitReport(
        benchmark=key.benchmark,
        model_id=key.model_id,
        harm_type=key.harm_type,
        condition=key.condition,
        scorer_id=key.scorer_id,
        control_family=key.control_family,
        axis_a=key.axis_a,
        axis_b=key.axis_b,
        estimable=True,
        reason=None,
        n_obs=fit.n_obs,
        n_items=fit.n_items,
        n_cells=fit.design.n_cells,
        n_params=fit.design.n_params,
        cov_type=fit.cov_type,
        n_resamples=fit.n_resamples,
        rms_interaction=fit.rms_interaction,
        rms_ci_low=None if fit.rms_ci is None else fit.rms_ci[0],
        rms_ci_high=None if fit.rms_ci is None else fit.rms_ci[1],
        powered=powered,
    )


def fit_terms(fit: PairFit, key: SeriesKey) -> list[FitTerm]:
    out: list[FitTerm] = []
    for term in fit.terms:
        if term.ci_low is None or term.ci_high is None:
            raise ValueError("a reported term needs a bootstrap CI; n_resamples must be >= 1")
        out.append(
            FitTerm(
                benchmark=key.benchmark,
                model_id=key.model_id,
                harm_type=key.harm_type,
                condition=key.condition,
                scorer_id=key.scorer_id,
                control_family=key.control_family,
                axis_a=key.axis_a,
                axis_b=key.axis_b,
                term=term.term,
                kind=term.kind,
                value_a=term.value_a,
                value_b=term.value_b,
                estimate=term.estimate,
                se=term.se,
                ci_low=term.ci_low,
                ci_high=term.ci_high,
                n_obs=fit.n_obs,
                n_items=fit.n_items,
            )
        )
    return out


def _effects(
    fit: PairFit, benchmark: str, model_id: str, harm_type: str, powered: bool
) -> list[EffectEstimate]:
    """EffectEstimates from the triangulated fit. p/q/family_size/correction deliberately unset."""
    out: list[EffectEstimate] = []
    for term in fit.terms:
        if term.ci_low is None or term.ci_high is None:
            raise ValueError("an EffectEstimate needs a bootstrap CI; n_resamples must be >= 1")
        out.append(
            EffectEstimate(
                benchmark=benchmark,
                model=model_id,
                axis_pair=fit.design.axis_pair,
                term=term.term,
                harm_type=harm_type,
                estimate=term.estimate,
                se=term.se,
                ci_low=term.ci_low,
                ci_high=term.ci_high,
                powered=powered,
                # p, q, family_size, correction: NOT set here. StatsEngine only (invariant 3).
            )
        )
    return out
