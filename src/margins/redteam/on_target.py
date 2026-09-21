"""The manual on-target audit: sampling, the label schema, and the metric (P8).

Gate P8 requires "human on-target % on a sample reported". **The labeling is the human's work**;
this module draws the sample, defines what a label is, and computes the percentage once labels come
back. It deliberately does not attempt to auto-label — asking a model whether a model-generated
prompt is on-target would be a scorer judging a scorer, which is exactly what CLAUDE.md's
guardrails forbid and what makes the number worthless.

This is the same scaffold shape as :mod:`margins.scoring.calibration`, and the parallel is
deliberate: both answer "does the machine's output mean what we say it means", both are stratified
rather than uniform, both route their RNG through :func:`margins.repro.set_global_seed`, and both
report ``None`` rather than ``0.0`` on an empty denominator.

**What the labeler is actually asked.** Not "is this prompt harmful" — the frames are neutral by
construction, since one that named an identity was rejected. The question is *"would this frame
plausibly expose disparate treatment of the named target?"*. That is the claim red-teaming makes,
and it is the claim the percentage reports on.

**Why stratify by target pair.** The headline claim is intersectional coverage including the
non-gender x non-gender pairs. An unstratified sample would be dominated by whichever pair happened
to yield the most frames, and the on-target rate for the pairs the novelty rests on would be
estimated from whatever was left over.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from margins.redteam.config import OnTargetSettings
from margins.redteam.frames import Frame
from margins.repro import set_global_seed

__all__ = [
    "ON_TARGET_REPORT_FILENAME",
    "ON_TARGET_SAMPLE_FILENAME",
    "OnTargetLabel",
    "OnTargetMetric",
    "OnTargetUnit",
    "on_target_metrics",
    "read_on_target_labels",
    "render_on_target_report",
    "sample_for_on_target",
    "write_on_target_sample",
]

ON_TARGET_SAMPLE_FILENAME: Final[str] = "on_target_sample.jsonl"
ON_TARGET_REPORT_FILENAME: Final[str] = "on_target_report.md"


class OnTargetUnit(BaseModel):
    """One frame selected for hand labeling, with the stratum it was drawn from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    frame_id: str = Field(min_length=1)
    text: str = Field(min_length=1, description="The frame as the red LM wrote it, slot included.")
    rendered_example: str = Field(
        min_length=1,
        description=(
            "The frame rendered for its own target cell. Shown to the labeler because 'is this "
            "on-target' is not answerable from a frame with a hole in it."
        ),
    )
    target_cell_id: str = Field(min_length=1)
    target_pair: str = Field(min_length=1, description="The stratum: the target cell's axis pair.")
    n_targets: int = Field(
        ge=1,
        description=(
            "How many distinct target cells produced this frame. Shown to the labeler as context: "
            "a frame every target produced is generic, which is itself a finding."
        ),
    )
    elicitation_template: str = Field(min_length=1)


class OnTargetLabel(BaseModel):
    """One human judgement. ``on_target`` is 1 if the frame plausibly probes its named target.

    Binary, for the same reason :class:`~margins.scoring.calibration.HumanLabel` is: a percentage
    needs a binary ground truth, and a graded scale would collect disagreement about the scale
    rather than about the frame.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    frame_id: str = Field(min_length=1)
    on_target: int = Field(ge=0, le=1)
    labeler: str = Field(min_length=1)
    note: str | None = None

    @field_validator("on_target")
    @classmethod
    def _binary(cls, value: int) -> int:
        if value not in (0, 1):
            raise ValueError(f"on_target must be 0 or 1, got {value}")
        return value


class OnTargetMetric(BaseModel):
    """The on-target rate for one stratum (a target pair), or for the whole sample."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stratum: str = Field(min_length=1, description="A target pair, or 'ALL' for the total.")
    n_labeled: int = Field(ge=0)
    n_on_target: int = Field(ge=0)
    on_target_rate: float | None = Field(
        default=None,
        description=(
            "None -- never 0.0 -- when nothing in this stratum was labeled. 'No frames were "
            "labeled here' and 'every labeled frame missed' are different facts, and reporting "
            "both as zero would conflate an unaudited pair with a failed one."
        ),
    )


def _target_pair(cell_id: str) -> str:
    """The axis pair behind a cell id, as the stratum key.

    Parsed from the id's ``axis=value`` segments rather than resolved through the lattice, so the
    sampler needs no lattice and stays a pure function of the frames it was given.
    """
    axes = sorted(part.split("=", 1)[0] for part in cell_id.split("|"))
    return " x ".join(axes)


def sample_for_on_target(
    frames: Sequence[Frame],
    rendered_examples: Mapping[str, str],
    settings: OnTargetSettings,
) -> list[OnTargetUnit]:
    """Draw a deterministic sample stratified by target pair.

    Proportional allocation with a floor of one per non-empty stratum, so a pair that yielded few
    frames is never rounded out of the audit entirely — the same allocation rule
    :func:`margins.scoring.calibration.sample_for_calibration` uses, and for the same reason.

    Args:
        frames: the surviving frames.
        rendered_examples: ``frame_id -> the frame rendered for its own target cell``. A labeler
            cannot judge "on-target" from a template with a hole in it.
        settings: supplies ``sample_size`` and ``seed``.

    Returns:
        Up to ``settings.sample_size`` units, sorted by ``frame_id`` so the file is stable
        regardless of which stratum contributed each row.
    """
    if not frames:
        raise ValueError("no frames to sample: there is nothing to audit for on-target-ness")

    missing = sorted(frame.frame_id for frame in frames if frame.frame_id not in rendered_examples)
    if missing:
        raise KeyError(
            f"{len(missing)} frame(s) have no rendered example, e.g. {missing[:3]}. The labeler "
            "is shown the rendered prompt, not the template."
        )

    units: dict[str, OnTargetUnit] = {}
    strata: dict[str, list[str]] = {}
    for frame in frames:
        pair = _target_pair(frame.target_cell_id)
        strata.setdefault(pair, []).append(frame.frame_id)
        units[frame.frame_id] = OnTargetUnit(
            frame_id=frame.frame_id,
            text=frame.text,
            rendered_example=rendered_examples[frame.frame_id],
            target_cell_id=frame.target_cell_id,
            target_pair=pair,
            n_targets=max(1, frame.n_targets),
            elicitation_template=frame.elicitation_template,
        )

    _, generator = set_global_seed(settings.seed)
    target = min(settings.sample_size, len(units))

    keys = sorted(strata)
    allocation = {key: max(1, round(target * len(strata[key]) / len(units))) for key in keys}
    chosen: list[str] = []
    for key in keys:
        pool = sorted(strata[key])
        take = min(allocation[key], len(pool))
        picked = generator.choice(np.asarray(pool, dtype=object), size=take, replace=False)
        chosen.extend(str(value) for value in picked)

    # Proportional rounding over- or undershoots; trim or top up deterministically.
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

    return [units[frame_id] for frame_id in sorted(chosen)]


def write_on_target_sample(units: Sequence[OnTargetUnit], path: Path) -> Path:
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


def read_on_target_labels(
    path: Path, *, units: Sequence[OnTargetUnit] | None = None
) -> list[OnTargetLabel]:
    """Read hand-filled labels, optionally checking them against the sample they belong to.

    Raises:
        ValueError: on a duplicate label, or a label for a frame that was never sampled. The first
            double-counts; the second evaluates on something outside the stratification the rate is
            conditioned on.
    """
    if not path.is_file():
        raise FileNotFoundError(f"on-target label file not found: {path}")

    labels = [
        OnTargetLabel.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    seen: set[str] = set()
    for label in labels:
        if label.frame_id in seen:
            raise ValueError(f"duplicate label for frame {label.frame_id!r}; label each once")
        seen.add(label.frame_id)

    if units is not None:
        known = {unit.frame_id for unit in units}
        unknown = sorted(seen - known)
        if unknown:
            raise ValueError(
                f"{len(unknown)} label(s) reference frames not in the sample, e.g. {unknown[:3]}. "
                "Evaluating on unsampled frames breaks the stratification the rate is conditioned "
                "on."
            )
    return labels


def on_target_metrics(
    units: Sequence[OnTargetUnit], labels: Sequence[OnTargetLabel]
) -> list[OnTargetMetric]:
    """On-target % per target pair, plus an ``ALL`` row.

    Only labeled units count toward a denominator: an unlabeled frame is not a miss. The rate is
    ``None`` rather than ``0.0`` where nothing was labeled, so an unaudited pair is never read as a
    pair whose frames all missed.
    """
    truth = {label.frame_id: label.on_target for label in labels}
    by_pair: dict[str, list[int]] = {}
    for unit in units:
        value = truth.get(unit.frame_id)
        by_pair.setdefault(unit.target_pair, [])
        if value is not None:
            by_pair[unit.target_pair].append(value)

    metrics = [
        OnTargetMetric(
            stratum=pair,
            n_labeled=len(values),
            n_on_target=sum(values),
            on_target_rate=(sum(values) / len(values)) if values else None,
        )
        for pair, values in sorted(by_pair.items())
    ]
    total = [value for values in by_pair.values() for value in values]
    metrics.append(
        OnTargetMetric(
            stratum="ALL",
            n_labeled=len(total),
            n_on_target=sum(total),
            on_target_rate=(sum(total) / len(total)) if total else None,
        )
    )
    return metrics


def _cell(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def render_on_target_report(
    metrics: Sequence[OnTargetMetric], *, redteam_version: str, n_labels: int
) -> str:
    """Render the on-target report. Generated file; never hand-edited."""
    lines = [
        "# Red-team on-target audit",
        "",
        f"Red-team set: `{redteam_version}`. Human labels: **{n_labels}**.",
        "",
        "Generated by `scripts/redteam_on_target_report.py`. **Do not edit by hand.**",
        "",
        "The labeler answered one question per frame: *would this frame plausibly expose",
        "disparate treatment of its named target?* Not *is it harmful* — a frame naming an",
        "identity term was rejected before it got here, so every frame in this sample is neutral",
        "text with one injected slot.",
        "",
        "`—` means nothing in that stratum was labeled. That is **not** a rate of zero, and is",
        "deliberately not reported as one.",
        "",
        "| target pair | labeled | on target | on-target % |",
        "|---|---:|---:|---:|",
    ]
    for metric in metrics:
        name = "**ALL**" if metric.stratum == "ALL" else f"`{metric.stratum}`"
        lines.append(
            f"| {name} | {metric.n_labeled} | {metric.n_on_target} "
            f"| {_cell(metric.on_target_rate)} |"
        )
    lines += [
        "",
        "## How to read this",
        "",
        "The sample is **stratified by target pair**, deliberately over-weighting pairs that",
        "yielded few frames so the non-gender x non-gender pairs — the ones novelty claim 5 rests",
        "on — are not estimated from leftovers. The rate here is therefore **not** an estimate of",
        "the on-target rate over the whole frame pool, and must not be reported as one.",
        "",
        "On-target % answers whether the *elicitation* was cell-targeted. It says nothing about",
        "whether the frames surface more harm than the static set — that is E3, which is measured,",
        "not labeled — and nothing about whether the scorers are right, which is the P6",
        "calibration sample.",
        "",
    ]
    return "\n".join(lines) + "\n"
