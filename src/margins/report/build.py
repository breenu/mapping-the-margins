"""Turning P7/P9 records into the committed result tables (P10).

Pure functions from records to rows. The I/O -- reading Parquet, writing CSV and LaTeX -- lives in
``scripts/build_result_tables.py``, for the same reason the metric and stats layers keep their
orchestration out of their scripts: the transformation is what deserves a test, and a test should
not need a Parquet file on disk to run one.

Two joins here are load-bearing and are checked rather than assumed:

* **``EffectEstimate`` carries a term string but not the term's shape.** ``kind``, ``value_a`` and
  ``value_b`` live on the triangulated :class:`~margins.metrics.records.FitTerm`, and the figures
  need them to lay a gamma table out as a grid. A term that finds no matching fit term raises --
  it would otherwise silently become an interaction with no cell coordinates and vanish from the
  heatmap.
* **Claim status comes from P9's own ``claims.parquet``**, never recomputed here. The figure layer
  re-runs the gate and cross-checks it (:mod:`margins.viz.claims`); if this module recomputed the
  status too, the cross-check would compare one derivation against another rather than against
  the stats run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from margins.metrics.records import DriftComparison, FitReport, FitTerm
from margins.schemas import CellStat, EffectEstimate
from margins.stats.records import DescriptiveOnlyDecision, FamilyMember, SignificanceClaim

__all__ = [
    "MissingFitTermError",
    "descriptive_only_rows",
    "drift_rows",
    "emergence_rows",
    "fit_report_rows",
    "single_axis_rows",
]

_REAL: Mapping[str, str] = {"data_status": "real"}


class MissingFitTermError(KeyError):
    """An estimate names a term that the triangulated fit never emitted."""


def _term_key(
    benchmark: str, model_id: str, harm_type: str, axis_a: str, axis_b: str, term: str
) -> tuple[str, ...]:
    return (benchmark, model_id, harm_type, axis_a, axis_b, term)


def emergence_rows(
    effects: Sequence[EffectEstimate],
    fit_terms: Sequence[FitTerm],
    claims: Sequence[SignificanceClaim],
    family: Sequence[FamilyMember],
) -> list[dict[str, Any]]:
    """One row per emitted term, with its claim status and whether it is in the FDR family."""
    shapes = {
        _term_key(
            term.benchmark, term.model_id, term.harm_type, term.axis_a, term.axis_b, term.term
        ): term
        for term in fit_terms
        # The triangulated series is the one EffectEstimates are made from; a per-scorer fit of
        # the same term would supply the same shape and a different estimate.
        if term.condition == "model" and term.scorer_id is None
    }
    statuses = {
        _term_key(
            claim.benchmark, claim.model_id, claim.harm_type, claim.axis_a, claim.axis_b, claim.term
        ): claim
        for claim in claims
    }
    members = {
        _term_key(
            member.benchmark,
            member.model_id,
            member.harm_type,
            member.axis_a,
            member.axis_b,
            member.term,
        )
        for member in family
    }

    rows: list[dict[str, Any]] = []
    for effect in effects:
        axis_a, axis_b = effect.axis_pair
        key = _term_key(
            effect.benchmark, effect.model, effect.harm_type, axis_a, axis_b, effect.term
        )
        shape = shapes.get(key)
        if shape is None:
            raise MissingFitTermError(
                f"no triangulated fit term for {key}. The estimate exists and its shape does "
                "not, so a figure could not place it in a grid. Rebuild data/metrics/ and "
                "data/stats/ from the same run."
            )
        claim = statuses.get(key)
        rows.append(
            {
                "benchmark": effect.benchmark,
                "model_id": effect.model,
                "harm_type": effect.harm_type,
                "axis_a": axis_a,
                "axis_b": axis_b,
                "term": effect.term,
                "kind": shape.kind,
                "value_a": shape.value_a,
                "value_b": shape.value_b,
                "estimate": effect.estimate,
                "se": effect.se,
                "ci_low": effect.ci_low,
                "ci_high": effect.ci_high,
                "p": effect.p,
                "q": effect.q,
                # B3, beside the GLM p and never corrected. Carried from the claim row P9 wrote;
                # empty for an effect the gate never saw (a descriptive arm) or a non-member term.
                "p_permutation": claim.p_permutation if claim is not None else None,
                "family_size": effect.family_size,
                "correction": effect.correction,
                "powered": effect.powered,
                # An effect with no claim row was never gated; leaving the column empty says so
                # rather than inventing a status the stats run did not record.
                "status": claim.status if claim is not None else "",
                "claimable": bool(claim.claimable) if claim is not None else False,
                "family_member": key in members,
                **_REAL,
            }
        )
    return rows


def descriptive_only_rows(decisions: Sequence[DescriptiveOnlyDecision]) -> list[dict[str, Any]]:
    return [
        {
            "benchmark": decision.benchmark,
            "model_id": decision.model_id,
            "harm_type": decision.harm_type,
            "descriptive_only": decision.descriptive_only,
            "threshold": decision.threshold,
            "max_ratio": decision.max_ratio,
            "worst_scorer": decision.worst_scorer,
            "n_comparisons": decision.n_comparisons,
            "n_over_threshold": decision.n_over_threshold,
            **_REAL,
        }
        for decision in decisions
    ]


def drift_rows(comparisons: Sequence[DriftComparison]) -> list[dict[str, Any]]:
    return [
        {
            "benchmark": comparison.benchmark,
            "model_id": comparison.model_id,
            "harm_type": comparison.harm_type,
            "scorer_id": comparison.scorer_id,
            "axis_a": comparison.axis_a,
            "axis_b": comparison.axis_b,
            "control_family": comparison.control_family,
            "model_rms": comparison.model_rms,
            "drift_rms": comparison.drift_rms,
            "ratio": comparison.ratio,
            "n_model_obs": comparison.n_model_obs,
            "n_drift_obs": comparison.n_drift_obs,
            **_REAL,
        }
        for comparison in comparisons
    ]


def fit_report_rows(reports: Sequence[FitReport]) -> list[dict[str, Any]]:
    """The triangulated model-condition reports only -- including the not-estimable ones.

    The not-estimable rows are the point: they are how the extractive paradigm's empty
    interaction slot reaches a figure as a reason rather than as a missing row.
    """
    return [
        {
            "benchmark": report.benchmark,
            "model_id": report.model_id,
            "harm_type": report.harm_type,
            "axis_a": report.axis_a,
            "axis_b": report.axis_b,
            "estimable": report.estimable,
            "reason": report.reason,
            "n_obs": report.n_obs,
            "n_items": report.n_items,
            "n_cells": report.n_cells,
            "rms_interaction": report.rms_interaction,
            "powered": bool(report.powered),
            **_REAL,
        }
        for report in reports
        if report.condition == "model" and report.scorer_id is None
    ]


def single_axis_rows(stats: Sequence[CellStat]) -> list[dict[str, Any]]:
    """Order-1 cells only, with the axis and value parsed back out of the cell id.

    Order-2 cells are deliberately excluded: this table feeds the cross-paradigm figure, whose
    whole point is the level at which the two paradigms ARE comparable. An order-2 cell mean on
    that figure would be a harm level standing in for an interaction, which is the conflation the
    project exists to avoid.
    """
    rows: list[dict[str, Any]] = []
    for stat in stats:
        if "|" in stat.cell_id or "=" not in stat.cell_id:
            continue
        axis, value = stat.cell_id.split("=", 1)
        rows.append(
            {
                "benchmark": stat.benchmark,
                "model_id": stat.model,
                "harm_type": stat.harm_type,
                "cell_id": stat.cell_id,
                "axis": axis,
                "value": value,
                "n": stat.n,
                "mean_standardized": stat.mean_standardized,
                "ci_low": stat.ci_low,
                "ci_high": stat.ci_high,
                "powered": stat.powered,
                **_REAL,
            }
        )
    return rows
