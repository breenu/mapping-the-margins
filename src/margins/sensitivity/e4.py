"""E4, the sensitivity subset: the class x race interaction across temperature and seed (RQ4).

The estimands, decided before any E4 analysis ran (pre-registration §10, 16 Oct 2026):

1. **Every arm is standardized WITHIN itself**, per (model, scorer, harm type) -- the T=0.7 arm
   (E1's class x race responses, reused) exactly like the T=1.0 and greedy arms. A temperature
   difference therefore cannot be a standardization difference. (23 Sep 2026: never pooled
   across arms, because pooling would normalize away the temperature effect E4 measures.)
2. **Each arm is fitted per seed AND pooled.** ``seed_group`` 1..n are the arm's seeds in
   ascending order, each on its own (27 items per cell in every group); ``seed_group`` 0 is all
   of the arm's seeds together. The per-seed groups are RQ4's seed variance; the pooled fit is the
   arm's best estimate. A greedy arm has one seed, so it has only the pooled group.
3. **Every free interaction term is reported**, so nothing is selected after seeing the data.

E4 is descriptive: an estimate and its cluster-bootstrap CI, no p and no q.

Standardization happens ONCE per arm, before the seed split: a seed group is a subset of an arm,
and re-standardizing it would put each group on its own scale.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from margins.schemas import IdentityCell, ItemScore
from margins.scoring.standardize import standardize_within_benchmark
from margins.sensitivity.fit import fit_pair, free_terms, restrict_to_cells

__all__ = ["POOLED", "ArmScores", "seed_groups", "sensitivity_rows"]

#: ``seed_group`` of the fit that pools all of an arm's seeds.
POOLED = 0

#: (scores, cells) -> fitted EmergenceResult; injectable so tests can run without a bootstrap.
Fitter = Callable[[Sequence[ItemScore], Sequence[IdentityCell]], Any]


@dataclass(frozen=True)
class ArmScores:
    """One model's raw item scores for one temperature arm, on any cells (restricted here)."""

    model_id: str
    arm: str
    scores: Sequence[ItemScore]


def _temperature(arm: ArmScores) -> float:
    # What the backend RECORDED on every response, never the profile's name: the smoke run's
    # eighth clause exists because a profile and what actually ran can disagree.
    temperatures = {s.temperature for s in arm.scores}
    if len(temperatures) != 1:
        raise ValueError(
            f"{arm.model_id}/{arm.arm} mixes recorded temperatures {sorted(temperatures)}; an arm "
            "is one temperature"
        )
    return temperatures.pop()


def seed_groups(scores: Sequence[ItemScore]) -> list[tuple[int, list[ItemScore]]]:
    """``[(0, all), (1, first seed), ...]``; just ``[(0, all)]`` when there is one seed."""
    seeds = sorted({s.seed for s in scores})
    groups = [(POOLED, list(scores))]
    if len(seeds) > 1:
        groups += [(i, [s for s in scores if s.seed == seed]) for i, seed in enumerate(seeds, 1)]
    return groups


def sensitivity_rows(
    arms: Sequence[ArmScores],
    cells: Sequence[IdentityCell],
    *,
    benchmark: str,
    signs: dict[str, int],
    fitter: Fitter | None = None,
    config: Any = None,
    min_cell_n: int | None = None,
) -> list[dict[str, Any]]:
    """The ``sensitivity`` table's rows, one per (model, harm, term, temperature, seed group).

    Pass either ``fitter`` or (``config``, ``min_cell_n``) for :func:`fit_pair`.
    """
    if fitter is None:
        if config is None or min_cell_n is None:
            raise ValueError("pass a fitter, or the metrics config and min_cell_n for fit_pair")

        def fitter(scores: Sequence[ItemScore], grid: Sequence[IdentityCell]) -> Any:
            return fit_pair(scores, grid, config=config, min_cell_n=min_cell_n)

    axis_a, axis_b = cells[0].axes[0][0], cells[0].axes[1][0]
    terms = free_terms(cells)
    seen: set[tuple[str, float]] = set()
    rows: list[dict[str, Any]] = []
    for arm in arms:
        in_grid = restrict_to_cells(arm.scores, cells)
        if not in_grid:
            raise ValueError(f"{arm.model_id}/{arm.arm} has no scores in the pair's grid")
        temperature = _temperature(ArmScores(arm.model_id, arm.arm, in_grid))
        if (arm.model_id, temperature) in seen:
            raise ValueError(f"two arms for {arm.model_id} at temperature {temperature}")
        seen.add((arm.model_id, temperature))
        standardized, _ = standardize_within_benchmark(in_grid, benchmark=benchmark, signs=signs)
        for group, subset in seed_groups(standardized):
            result = fitter(subset, cells)
            fitted = {
                (e.harm_type, e.term): e
                for e in result.effects
                if e.axis_pair == (axis_a, axis_b) and e.model == arm.model_id
            }
            harms = sorted({s.harm_type for s in subset})
            for harm in harms:
                for term in terms:
                    effect = fitted.get((harm, term))
                    if effect is None:
                        raise ValueError(
                            f"{arm.model_id}/{arm.arm}/seed group {group}: {harm} {term} was not "
                            "estimated; a sensitivity row cannot be silently missing"
                        )
                    rows.append(
                        {
                            "benchmark": benchmark,
                            "model_id": arm.model_id,
                            "harm_type": harm,
                            "axis_a": axis_a,
                            "axis_b": axis_b,
                            "term": term,
                            "temperature": temperature,
                            "seed_group": group,
                            "estimate": effect.estimate,
                            "ci_low": effect.ci_low,
                            "ci_high": effect.ci_high,
                            "data_status": "real",
                        }
                    )
    return rows
