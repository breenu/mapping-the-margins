"""The scorer's own interaction BESIDE the model's, as producible numbers (P7, section 9).

Section 9's rule: *if a scorer's interaction is comparable in magnitude to the model's, that harm
type is descriptive-only.* This module makes that rule computable and stops there.

* :class:`~margins.metrics.records.DriftComparison` -- per scorer, pair and control family: the
  model-condition interaction magnitude **as seen by that scorer** (its own per-scorer fit, so the
  comparison is apples to apples), the drift magnitude, their ratio, and a ``comparable`` flag at
  the config's PLACEHOLDER threshold.
* :class:`~margins.metrics.records.FamilyAgreement` -- per scorer and pair: do the two neutral
  families agree? Disagreement is a reportable style-dependence finding, surfaced, never averaged.
* :class:`~margins.metrics.records.HarmTypeFlag` -- per harm type: ``descriptive_only_candidate``
  if ANY scorer on ANY pair in ANY family is comparable.
* :class:`~margins.metrics.records.SideBySide` -- per term: the model coefficient and each
  family's drift coefficient, with their own CIs, on one row.

**Nothing is point-subtracted.** No "net" interaction is computed anywhere in this package. The
magnitudes are the root-mean-square of a pair's interaction coefficients, so the comparison is
between whole tables rather than between whichever single coefficient happened to be largest.
Enforcement -- the downgrade -- is P9's, against pre-registered thresholds.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from margins.metrics.config import MetricsConfig
from margins.metrics.drift import DriftFitResult
from margins.metrics.metric import EmergenceResult
from margins.metrics.records import (
    DriftComparison,
    FamilyAgreement,
    FitReport,
    FitTerm,
    HarmTypeFlag,
    SideBySide,
)
from margins.scoring.config import CONTROL_FAMILIES

__all__ = ["DriftComparisonResult", "compare_drift"]

_ModelKey = tuple[str, str, str, str, str, str]  # benchmark, model_id, harm, scorer, axis_a, axis_b
_DriftKey = tuple[str, str, str, str, str, str]  # benchmark, scorer, harm, family, axis_a, axis_b


@dataclass
class DriftComparisonResult:
    comparisons: list[DriftComparison] = field(default_factory=list)
    family_agreement: list[FamilyAgreement] = field(default_factory=list)
    harm_flags: list[HarmTypeFlag] = field(default_factory=list)
    side_by_side: list[SideBySide] = field(default_factory=list)


def _model_reports(model: EmergenceResult) -> dict[_ModelKey, FitReport]:
    """The per-SCORER model-condition fits (the triangulated one has scorer_id None)."""
    return {
        (r.benchmark, r.model_id, r.harm_type, r.scorer_id, r.axis_a, r.axis_b): r
        for r in model.fit_reports
        if r.condition == "model" and r.scorer_id is not None and r.estimable
    }


def _drift_reports(drift: DriftFitResult) -> dict[_DriftKey, FitReport]:
    return {
        (r.benchmark, str(r.scorer_id), r.harm_type, str(r.control_family), r.axis_a, r.axis_b): r
        for r in drift.fit_reports
        if r.condition == "drift_control" and r.estimable
    }


def _families_disagree(
    echo: float | None, declarative: float | None, factor: float
) -> tuple[float | None, bool | None]:
    """``(ratio, disagree)``. One family zero and the other not is a disagreement by definition."""
    if echo is None or declarative is None:
        return None, None
    if echo == 0.0 and declarative == 0.0:
        return None, False
    if echo == 0.0 or declarative == 0.0:
        return None, True
    ratio = echo / declarative
    return ratio, bool(ratio > factor or ratio < 1.0 / factor)


def compare_drift(
    model: EmergenceResult, drift: DriftFitResult, *, config: MetricsConfig
) -> DriftComparisonResult:
    """Join the per-scorer model fits to the drift fits and produce the section 9 numbers."""
    thresholds = config.drift
    result = DriftComparisonResult()
    model_reports = _model_reports(model)
    drift_reports = _drift_reports(drift)

    for key in sorted(model_reports):
        benchmark, model_id, harm_type, scorer_id, axis_a, axis_b = key
        report = model_reports[key]
        model_rms = float(report.rms_interaction or 0.0)
        per_family: dict[str, float | None] = {}
        for family in CONTROL_FAMILIES:
            drift_report = drift_reports.get(
                (benchmark, scorer_id, harm_type, family, axis_a, axis_b)
            )
            if drift_report is None:
                per_family[family] = None
                continue
            drift_rms = float(drift_report.rms_interaction or 0.0)
            per_family[family] = drift_rms
            ratio = drift_rms / model_rms if model_rms > 0.0 else None
            comparable = (
                ratio >= thresholds.comparable_ratio if ratio is not None else drift_rms > 0.0
            )
            result.comparisons.append(
                DriftComparison(
                    benchmark=benchmark,
                    model_id=model_id,
                    harm_type=harm_type,
                    scorer_id=scorer_id,
                    axis_a=axis_a,
                    axis_b=axis_b,
                    control_family=family,
                    model_rms=model_rms,
                    drift_rms=drift_rms,
                    ratio=ratio,
                    comparable=comparable,
                    n_model_obs=report.n_obs,
                    n_drift_obs=drift_report.n_obs,
                )
            )
        if any(value is not None for value in per_family.values()):
            echo = per_family.get("prompt_echo")
            declarative = per_family.get("neutral_declarative")
            ratio, disagree = _families_disagree(
                echo, declarative, thresholds.family_disagreement_ratio
            )
            result.family_agreement.append(
                FamilyAgreement(
                    benchmark=benchmark,
                    model_id=model_id,
                    harm_type=harm_type,
                    scorer_id=scorer_id,
                    axis_a=axis_a,
                    axis_b=axis_b,
                    rms_prompt_echo=echo,
                    rms_neutral_declarative=declarative,
                    ratio=ratio,
                    families_disagree=disagree,
                )
            )

    # Harm-type flags: any comparable comparison anywhere under the harm type.
    for benchmark, model_id, harm_type in sorted({k[:3] for k in model_reports}):
        rows = [
            c
            for c in result.comparisons
            if (c.benchmark, c.model_id, c.harm_type) == (benchmark, model_id, harm_type)
        ]
        comparable_rows = [c for c in rows if c.comparable]
        result.harm_flags.append(
            HarmTypeFlag(
                benchmark=benchmark,
                model_id=model_id,
                harm_type=harm_type,
                descriptive_only_candidate=bool(comparable_rows),
                n_comparisons=len(rows),
                n_comparable=len(comparable_rows),
                scorers_comparable=tuple(sorted({c.scorer_id for c in comparable_rows})),
            )
        )

    result.side_by_side = _side_by_side(model.fit_terms, drift.fit_terms, model_reports)
    return result


def _side_by_side(
    model_terms: Sequence[FitTerm],
    drift_terms: Sequence[FitTerm],
    model_reports: dict[_ModelKey, FitReport],
) -> list[SideBySide]:
    drift_by_key: dict[tuple[str, str, str, str, str, str, str], FitTerm] = {
        (
            t.benchmark,
            str(t.scorer_id),
            t.harm_type,
            str(t.control_family),
            t.axis_a,
            t.axis_b,
            t.term,
        ): t
        for t in drift_terms
    }
    rows: list[SideBySide] = []
    for term in model_terms:
        if term.condition != "model" or term.scorer_id is None:
            continue
        key: _ModelKey = (
            term.benchmark,
            term.model_id,
            term.harm_type,
            term.scorer_id,
            term.axis_a,
            term.axis_b,
        )
        if key not in model_reports:
            continue
        matches = {
            family: drift_by_key.get(
                (
                    term.benchmark,
                    term.scorer_id,
                    term.harm_type,
                    family,
                    term.axis_a,
                    term.axis_b,
                    term.term,
                )
            )
            for family in CONTROL_FAMILIES
        }
        echo = matches["prompt_echo"]
        declarative = matches["neutral_declarative"]
        rows.append(
            SideBySide(
                benchmark=term.benchmark,
                model_id=term.model_id,
                harm_type=term.harm_type,
                scorer_id=term.scorer_id,
                axis_a=term.axis_a,
                axis_b=term.axis_b,
                term=term.term,
                kind=term.kind,
                model_estimate=term.estimate,
                model_ci_low=term.ci_low,
                model_ci_high=term.ci_high,
                prompt_echo_estimate=None if echo is None else echo.estimate,
                prompt_echo_ci_low=None if echo is None else echo.ci_low,
                prompt_echo_ci_high=None if echo is None else echo.ci_high,
                neutral_declarative_estimate=None if declarative is None else declarative.estimate,
                neutral_declarative_ci_low=None if declarative is None else declarative.ci_low,
                neutral_declarative_ci_high=None if declarative is None else declarative.ci_high,
            )
        )
    rows.sort(
        key=lambda r: (
            r.benchmark,
            r.model_id,
            r.harm_type,
            r.scorer_id,
            r.axis_a,
            r.axis_b,
            r.term,
        )
    )
    return rows
