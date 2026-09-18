"""Fitting the scorer identity-drift control: the SAME model on the scorer's own series (P7).

P6 scored known-neutral text through every scorer and emitted ``drift_control.parquet`` with the
axis values resolved from the lattice. This module fits ``standardized ~ A + B + A:B`` on those
rows, per scorer, per harm type, per **control family** -- through the very same
:func:`~margins.metrics.estimator.estimate_pair` that fits the model condition, so the two series
cannot have been produced by different estimators.

**Each family is fitted separately and never pooled.** ``prompt_echo`` is the imperative
instruction register; ``neutral_declarative`` is flat prose. If a scorer's drift interaction differs
between them, that is a style-dependence finding to report (:mod:`margins.metrics.compare`
surfaces it), not noise to average out.

**The cluster is the control text.** Each text is scored once per scorer, so items and
observations coincide and the cluster-robust covariance reduces to a heteroskedasticity-robust one;
the bootstrap resamples texts within cell. This is stated rather than assumed because the model
condition's clustering (many seeds per prompt) does not carry over.

The result rows carry ``condition="drift_control"``, ``model_id`` set to P6's
:data:`~margins.scoring.drift.CONTROL_MODEL_ID` sentinel (no model produced this text), and
``powered=None`` -- power is a question about the model series.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from margins.metrics.config import MetricsConfig
from margins.metrics.design import pair_designs
from margins.metrics.estimator import NotEstimableError, PairFrame, estimate_pair
from margins.metrics.metric import SeriesKey, fit_report, fit_terms, not_estimable_report
from margins.metrics.records import FitReport, FitTerm
from margins.repro import set_global_seed
from margins.schemas import IdentityCell
from margins.scoring.drift import CONTROL_MODEL_ID, DriftRow

__all__ = ["DriftFitResult", "fit_drift"]


@dataclass
class DriftFitResult:
    """Every drift fit: terms with CIs, and a report per (scorer, harm type, family, pair)."""

    fit_terms: list[FitTerm] = field(default_factory=list)
    fit_reports: list[FitReport] = field(default_factory=list)


def fit_drift(
    rows: Sequence[DriftRow], cells: Sequence[IdentityCell], *, config: MetricsConfig
) -> DriftFitResult:
    """Fit the scorer's own interaction on the P6 drift artifact, per family.

    Every declared pair is reported for every (scorer, harm type, family) present -- estimable or
    not -- so a missing drift number is never silent.
    """
    _, rng = set_global_seed(config.bootstrap.seed)
    settings = config.bootstrap
    result = DriftFitResult()
    designs = pair_designs(cells)

    groups = sorted({(r.benchmark, r.scorer_id, r.harm_type, r.control_family) for r in rows})
    for benchmark, scorer_id, harm_type, family in groups:
        group = [
            r
            for r in rows
            if (r.benchmark, r.scorer_id, r.harm_type, r.control_family)
            == (benchmark, scorer_id, harm_type, family)
            and r.order == 2
        ]
        group.sort(key=lambda r: (r.cell_id, r.response_id, r.score_id))
        for design in designs:
            in_pair = [r for r in group if (r.axis_a, r.axis_b) == design.axis_pair and r.value_b]
            frame = PairFrame(
                y=np.array([r.standardized for r in in_pair]),
                value_a=tuple(r.value_a for r in in_pair),
                value_b=tuple(str(r.value_b) for r in in_pair),
                item=tuple(r.response_id for r in in_pair),
            )
            key = SeriesKey(
                benchmark=benchmark,
                model_id=CONTROL_MODEL_ID,
                harm_type=harm_type,
                condition="drift_control",
                scorer_id=scorer_id,
                control_family=family,
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
                    not_estimable_report(key, design, frame, reason=str(exc), powered=None)
                )
                continue
            result.fit_reports.append(fit_report(fit, key, powered=None))
            result.fit_terms.extend(fit_terms(fit, key))
    return result
