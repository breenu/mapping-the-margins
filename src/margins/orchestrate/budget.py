"""GPU-hour arithmetic for the lab run: projections, and the red-team frame cap.

**Measured speed, never assumed.** Every projection here takes seconds-per-generation as an
ARGUMENT. The runbook quotes the W2 pilot's T4 numbers only as the estimate available before the lab
card has been measured; ``run_p11.py verify-gpu`` measures the actual card and ``run_p11.py budget``
recomputes everything from that. An estimate is never trusted over a measurement (§7).

Pure: no I/O, no clock, no RNG.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

__all__ = [
    "MIN_REDTEAM_FRAMES",
    "RedTeamBudget",
    "project_gpu_hours",
    "redteam_max_frames",
]

#: The estimator needs at least 2 items per cell (`margins.metrics.estimator._check_estimable`),
#: and a frame is one item per cell per carrier. Below 2 frames the red-team arm could still be
#: estimable through the second carrier, but a single frame's quirks would then BE the arm.
MIN_REDTEAM_FRAMES: Final[int] = 2


def project_gpu_hours(sec_per_generation: float, n_generations: int) -> float:
    """``n_generations x sec_per_generation`` in hours."""
    if sec_per_generation <= 0:
        raise ValueError(f"sec_per_generation must be positive, got {sec_per_generation}")
    if n_generations < 0:
        raise ValueError(f"n_generations must be non-negative, got {n_generations}")
    return n_generations * sec_per_generation / 3600.0


@dataclass(frozen=True)
class RedTeamBudget:
    """How a fixed red-team GPU budget divides between elicitation and the audit."""

    budget_hours: float
    elicitation_hours: float
    per_frame_hours: float
    max_frames: int

    @property
    def audit_hours(self) -> float:
        return self.max_frames * self.per_frame_hours

    @property
    def total_hours(self) -> float:
        return self.elicitation_hours + self.audit_hours


def redteam_max_frames(
    *,
    budget_hours: float,
    elicitation_generations: int,
    red_lm_sec_per_generation: float,
    target_sec_per_generation: Mapping[str, float],
    target_cells: int,
    carriers: int,
    audit_seeds: int,
) -> RedTeamBudget:
    """The largest frame count whose audit, plus the elicitation, fits ``budget_hours``.

    A frame renders into every target cell under every carrier, and each rendering is generated
    ``audit_seeds`` times by EVERY target model -- the same frame set runs through both, so neither
    faces prompts the other did not. The carrier count is an upper bound (the carrier-free frame
    renders once), so the cap errs on the side of staying inside the budget.

    Raises:
        ValueError: if the budget cannot buy even :data:`MIN_REDTEAM_FRAMES` frames after
            elicitation. That is a finding to report, not something to round up.
    """
    if not target_sec_per_generation:
        raise ValueError("at least one target model's speed is required")
    elicitation_hours = project_gpu_hours(red_lm_sec_per_generation, elicitation_generations)
    renderings_per_frame = target_cells * carriers * audit_seeds
    per_frame_hours = sum(
        project_gpu_hours(seconds, renderings_per_frame)
        for seconds in target_sec_per_generation.values()
    )
    remaining = budget_hours - elicitation_hours
    frames = math.floor(remaining / per_frame_hours) if remaining > 0 else 0
    if frames < MIN_REDTEAM_FRAMES:
        raise ValueError(
            f"a {budget_hours:.1f} GPU-h red-team budget leaves {max(remaining, 0.0):.1f} h after "
            f"{elicitation_hours:.1f} h of elicitation, which buys {frames} frame(s) at "
            f"{per_frame_hours:.2f} h each -- below the minimum of {MIN_REDTEAM_FRAMES}. Report "
            "this to the project lead; do not raise the budget or cut elicitation on your own."
        )
    return RedTeamBudget(
        budget_hours=budget_hours,
        elicitation_hours=elicitation_hours,
        per_frame_hours=per_frame_hours,
        max_frames=frames,
    )
