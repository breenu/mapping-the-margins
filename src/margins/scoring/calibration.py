"""Human-calibration scaffolding (P6): sampling, the label schema, and the metrics.

Gate P6 requires "scorer precision/recall on 150-300 labeled items". **The labeling is the human's
work**; this module builds the sample, defines what a label is, and computes the metrics once labels
come back. It deliberately does not attempt to generate labels -- an LLM judge treated as ground
truth is exactly what CLAUDE.md's guardrails forbid, and a scorer calibrated against another scorer
is not calibrated at all.

**Why the sample is stratified rather than uniform.** A uniform sample of a corpus where most
generations are benign spends nearly all of a human's limited attention confirming true negatives.
The strata used are:

* **cell order** (1 vs 2), so the intersectional cells -- the ones carrying the estimand -- are not
  swamped by the single-axis ones;
* **scorer disagreement**, bucketed, so the sample is informative about precisely the cases
  triangulation splits on. Where the scorers agree, the label mostly confirms what is already known;
  where they disagree, the label is what decides which of them to believe.

Both strata are recorded on each sampled row, so the report can say what the sample was and a
reviewer can judge whether precision estimated on it transfers.

**Determinism.** Sampling routes through :func:`margins.repro.set_global_seed` and threads the
returned generator, per CLAUDE.md's rule that no stochastic path may reach for a hidden global RNG.
The same responses, config and seed give the same sample on any machine.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from margins.repro import set_global_seed
from margins.schemas import ItemScore
from margins.scoring.config import ScoringConfig

__all__ = [
    "CALIBRATION_REPORT_FILENAME",
    "CALIBRATION_SAMPLE_FILENAME",
    "DISAGREEMENT_BUCKETS",
    "CalibrationMetric",
    "CalibrationUnit",
    "HumanLabel",
    "calibration_metrics",
    "read_labels",
    "render_calibration_report",
    "sample_for_calibration",
    "write_calibration_sample",
]

CALIBRATION_SAMPLE_FILENAME: Final[str] = "calibration_sample.jsonl"
CALIBRATION_REPORT_FILENAME: Final[str] = "calibration_report.md"

#: Disagreement strata, as quantile cut points on the per-response spread of standardized scores.
#: Three buckets rather than deciles: with n<=300 split across cell order too, deciles would leave
#: single-digit cells and the stratification would be noise.
DISAGREEMENT_BUCKETS: Final[tuple[str, ...]] = ("low", "mid", "high")


class CalibrationUnit(BaseModel):
    """One response selected for hand labeling, with the strata it was drawn from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    response_id: str = Field(min_length=1)
    harm_type: str = Field(min_length=1)
    text: str
    cell_id: str | None = None
    cell_order: int = Field(ge=0, le=2, description="0 when the response carries no cell.")
    disagreement_bucket: str
    scorer_raw: dict[str, float] = Field(
        description="Each scorer's raw score, so a labeler can be shown what the machines said."
    )


class HumanLabel(BaseModel):
    """One human judgement. ``label`` is 1 for harmful, 0 for not.

    Binary on purpose. Precision and recall need a binary ground truth, and asking a labeler for a
    graded score would invite disagreement about the scale rather than about the text -- the
    graded question is what the continuous scorers already answer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    response_id: str = Field(min_length=1)
    harm_type: str = Field(min_length=1)
    label: int = Field(ge=0, le=1)
    labeler: str = Field(min_length=1)
    note: str | None = None

    @field_validator("label")
    @classmethod
    def _label_is_binary(cls, value: int) -> int:
        if value not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {value}")
        return value


class CalibrationMetric(BaseModel):
    """One scorer's agreement with the human labels for one harm type."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scorer_id: str = Field(min_length=1)
    harm_type: str = Field(min_length=1)
    threshold: float
    n: int = Field(ge=0)
    n_positive: int = Field(ge=0, description="Human-labeled positives in the evaluated set.")
    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None


def _disagreement_bucket(spread: float, cuts: tuple[float, float]) -> str:
    if spread <= cuts[0]:
        return "low"
    if spread <= cuts[1]:
        return "mid"
    return "high"


def sample_for_calibration(
    scores: Sequence[ItemScore],
    texts: Mapping[str, str],
    config: ScoringConfig,
    *,
    harm_type: str,
    cell_order: Mapping[str, int] | None = None,
) -> list[CalibrationUnit]:
    """Draw a deterministic, stratified calibration sample for one harm type.

    Args:
        scores: model-condition scores. Drift-control rows are excluded: calibration asks whether a
            scorer identifies harm in *model output*, and control text has no harm to identify.
        texts: ``response_id -> text``, for the labeler to read.
        config: supplies ``sample_size`` and ``seed``.
        harm_type: the harm being calibrated. One call per harm type; precision is not
            comparable across harms and a pooled sample would hide that.
        cell_order: ``cell_id -> order``, so order-2 cells can be stratified for. Missing ids are
            treated as order 0 rather than guessed at.

    Returns:
        Up to ``config.calibration.sample_size`` units, sorted by ``response_id`` so the output file
        is stable regardless of which stratum contributed each row.
    """
    model_scores = [s for s in scores if s.condition == "model" and s.harm_type == harm_type]
    if not model_scores:
        raise ValueError(
            f"no model-condition scores for harm_type {harm_type!r}; nothing to calibrate against"
        )

    by_response: dict[str, list[ItemScore]] = {}
    for score in model_scores:
        by_response.setdefault(score.response_id, []).append(score)

    # Spread across scorers is the disagreement signal. One scorer means no disagreement, which is
    # a roster the config already refuses -- but a SUBSET run could produce it, so it is handled.
    spreads = {
        response_id: (float(np.ptp([s.standardized for s in rows])) if len(rows) > 1 else 0.0)
        for response_id, rows in by_response.items()
    }
    values = np.asarray(sorted(spreads.values()), dtype=float)
    cuts = (float(np.quantile(values, 1 / 3)), float(np.quantile(values, 2 / 3)))

    orders = cell_order or {}
    units: dict[str, CalibrationUnit] = {}
    strata: dict[tuple[int, str], list[str]] = {}
    for response_id, rows in by_response.items():
        first = rows[0]
        order = orders.get(first.cell_id or "", 0)
        bucket = _disagreement_bucket(spreads[response_id], cuts)
        strata.setdefault((order, bucket), []).append(response_id)
        units[response_id] = CalibrationUnit(
            response_id=response_id,
            harm_type=harm_type,
            text=texts.get(response_id, ""),
            cell_id=first.cell_id,
            cell_order=order,
            disagreement_bucket=bucket,
            scorer_raw={s.scorer_id: s.raw for s in sorted(rows, key=lambda r: r.scorer_id)},
        )

    _, generator = set_global_seed(config.calibration.seed)
    target = min(config.calibration.sample_size, len(units))

    # Proportional allocation with a floor of one per non-empty stratum, so a small stratum (the
    # order-2 high-disagreement cases, typically) is never rounded out of existence.
    keys = sorted(strata)
    allocation = {key: max(1, round(target * len(strata[key]) / len(units))) for key in keys}
    chosen: list[str] = []
    for key in keys:
        pool = sorted(strata[key])
        take = min(allocation[key], len(pool))
        picked = generator.choice(np.asarray(pool, dtype=object), size=take, replace=False)
        chosen.extend(str(value) for value in picked)

    # Proportional rounding overshoots or undershoots; trim or top up deterministically.
    chosen = sorted(set(chosen))
    if len(chosen) > target:
        picked = generator.choice(np.asarray(chosen, dtype=object), size=target, replace=False)
        chosen = sorted(str(value) for value in picked)
    elif len(chosen) < target:
        remaining = sorted(set(units) - set(chosen))
        shortfall = min(target - len(chosen), len(remaining))
        if shortfall:
            picked = generator.choice(
                np.asarray(remaining, dtype=object), size=shortfall, replace=False
            )
            chosen = sorted(chosen + [str(value) for value in picked])

    return [units[response_id] for response_id in sorted(chosen)]


def write_calibration_sample(units: Sequence[CalibrationUnit], path: Path) -> Path:
    """Write the sample as JSONL, ready to hand-fill. Canonical dump, LF newlines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            unit.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        for unit in units
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(lines) + ("\n" if lines else ""))
    return path


def read_labels(path: Path, *, units: Sequence[CalibrationUnit] | None = None) -> list[HumanLabel]:
    """Read hand-filled labels, optionally checking them against the sample they belong to.

    Raises:
        ValueError: on a label for a response that is not in the sample, or a duplicate label for
            one ``(response_id, harm_type)``. Both would silently distort precision -- the first by
            evaluating on something that was never sampled, the second by double-counting.
    """
    if not path.is_file():
        raise FileNotFoundError(f"label file not found: {path}")

    labels = [
        HumanLabel.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    seen: set[tuple[str, str]] = set()
    for label in labels:
        key = (label.response_id, label.harm_type)
        if key in seen:
            raise ValueError(f"duplicate label for {key}; each response is labeled once per harm")
        seen.add(key)

    if units is not None:
        known = {(u.response_id, u.harm_type) for u in units}
        unknown = sorted(str(key) for key in seen - known)
        if unknown:
            raise ValueError(
                f"{len(unknown)} label(s) reference responses not in the sample, e.g. "
                f"{unknown[:3]}. Evaluating on unsampled responses breaks the stratification the "
                "precision estimate is conditioned on."
            )

    return labels


def calibration_metrics(
    scores: Sequence[ItemScore],
    labels: Sequence[HumanLabel],
    config: ScoringConfig,
) -> list[CalibrationMetric]:
    """Precision / recall / F1 for every scorer against the human labels.

    A scorer fires when ``raw >= threshold`` (the roster's declared cut). Precision and recall are
    ``None`` rather than 0.0 when their denominator is empty -- "the scorer never fired" and "the
    scorer fired and was always wrong" are different facts, and reporting both as 0.0 would conflate
    a sample with no positives with a scorer that has none.
    """
    truth = {(label.response_id, label.harm_type): label.label for label in labels}
    thresholds = {spec.scorer_id: spec.threshold for spec in config.scorers}

    metrics: list[CalibrationMetric] = []
    for spec in sorted(config.scorers, key=lambda s: (s.harm_type, s.scorer_id)):
        rows = [
            s
            for s in scores
            if s.scorer_id == spec.scorer_id
            and s.condition == "model"
            and (s.response_id, s.harm_type) in truth
        ]
        tp = fp = fn = positives = 0
        for row in rows:
            actual = truth[(row.response_id, row.harm_type)]
            predicted = int(row.raw >= thresholds[spec.scorer_id])
            positives += actual
            if predicted and actual:
                tp += 1
            elif predicted and not actual:
                fp += 1
            elif not predicted and actual:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and (precision + recall) > 0
            else None
        )

        metrics.append(
            CalibrationMetric(
                scorer_id=spec.scorer_id,
                harm_type=spec.harm_type,
                threshold=spec.threshold,
                n=len(rows),
                n_positive=positives,
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )
    return metrics


def _cell(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def render_calibration_report(
    metrics: Sequence[CalibrationMetric],
    *,
    scoring_version: str,
    n_labels: int,
) -> str:
    """Render the calibration report. Generated file; never hand-edited."""
    lines = [
        "# Scorer calibration against human labels",
        "",
        f"Scoring roster: `{scoring_version}`. Human labels: **{n_labels}**.",
        "",
        "Generated by `scripts/calibration_report.py`. **Do not edit by hand.**",
        "",
        "A scorer fires when `raw >= threshold`. `—` means the denominator was empty: no",
        "positives in the sample (recall), or the scorer never fired (precision). That is not the",
        "same as a score of zero, and is deliberately not reported as one.",
        "",
        "| harm type | scorer | threshold | n | labeled + | TP | FP | FN |"
        " precision | recall | F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in metrics:
        lines.append(
            f"| {metric.harm_type} | `{metric.scorer_id}` | {metric.threshold:g} | {metric.n} | "
            f"{metric.n_positive} | {metric.true_positives} | {metric.false_positives} | "
            f"{metric.false_negatives} | {_cell(metric.precision)} | {_cell(metric.recall)} | "
            f"{_cell(metric.f1)} |"
        )
    lines += [
        "",
        "## How to read this",
        "",
        "These numbers say how well each scorer reproduces a human judgement on a stratified",
        "sample that deliberately over-weights scorer disagreement and order-2 cells. Precision on",
        "such a sample is **not** an estimate of precision on the corpus, and must not be reported",
        "as one — the sample was chosen to be hard.",
        "",
        "Calibration is one of three construct-validity checks and answers only one question:",
        "does the scorer find what a human calls harm? It says nothing about whether the scorer",
        "responds to identity terms independently of harm — that is the identity-drift control",
        "(`drift_control.parquet`) — nor about whether two scorers behave alike, which is the",
        "agreement table. A harm type needs all three before it is load-bearing.",
        "",
    ]
    return "\n".join(lines) + "\n"
