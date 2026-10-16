"""R2, the reproducibility rerun: does an independent regeneration give back the reported numbers?

Pre-registered 16 Oct 2026 (§10), before any rerun data existed:

* the rerun's raw scores are standardized with the ORIGINAL run's constants
  (:func:`~margins.scoring.standardize.standardize_with_reference`), so its estimates sit on the
  scale of the reported ones;
* the verdict is recomputed by re-running P9 with the class x race series REPLACED by the rerun's
  and every other member unchanged -- family size, drift downgrades and all
  (:func:`substitute_series`);
* **PASS iff every class x race family term's rerun estimate lies inside the original run's
  reported 95% CI AND every ClaimGate verdict is identical.** Max and RMS |difference| in units
  of the original SE, and the greedy exact-token-match rate, are reported pass or fail.

What R2 can and cannot show is stated in the paper, not here, but one consequence belongs beside
the code: when every verdict is ``descriptive_only`` -- a property of the SCORERS, which R2 does
not regenerate -- the verdict half of the rule is close to automatic and the CI half carries it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from margins.metrics.metric import EmergenceResult
from margins.metrics.records import FitReport, FitTerm
from margins.schemas import EffectEstimate
from margins.stats.inputs import StatsInputs
from margins.stats.records import SignificanceClaim

__all__ = [
    "R2Comparison",
    "R2Verdict",
    "compare_to_original",
    "exact_match_rate",
    "substitute_series",
    "summarize",
    "verdict_changes",
]

GroupKey = tuple[str, str, str]  # (benchmark, model, harm_type)


@dataclass(frozen=True)
class R2Comparison:
    """One family term: the original reported estimate against the rerun's."""

    model_id: str
    harm_type: str
    term: str
    original_estimate: float
    original_se: float
    original_ci_low: float
    original_ci_high: float
    rerun_estimate: float
    rerun_ci_low: float
    rerun_ci_high: float

    @property
    def difference(self) -> float:
        return self.rerun_estimate - self.original_estimate

    @property
    def difference_in_se(self) -> float:
        """|rerun - original| in units of the ORIGINAL SE -- the reported scale of the claim."""
        return abs(self.difference) / self.original_se if self.original_se > 0 else float("inf")

    @property
    def inside_original_ci(self) -> bool:
        return self.original_ci_low <= self.rerun_estimate <= self.original_ci_high


@dataclass(frozen=True)
class R2Verdict:
    """One family term's ClaimGate status, originally and with the rerun substituted."""

    model_id: str
    harm_type: str
    term: str
    original_status: str
    rerun_status: str

    @property
    def same(self) -> bool:
        return self.original_status == self.rerun_status


def _family(
    effects: Sequence[EffectEstimate], axis_pair: tuple[str, str], terms: Sequence[str]
) -> dict[tuple[str, str, str], EffectEstimate]:
    wanted = set(terms)
    return {
        (e.model, e.harm_type, e.term): e
        for e in effects
        if e.axis_pair == axis_pair and e.term in wanted
    }


def compare_to_original(
    original: Sequence[EffectEstimate],
    rerun: Sequence[EffectEstimate],
    *,
    axis_pair: tuple[str, str],
    terms: Sequence[str],
) -> list[R2Comparison]:
    """Pair every original family term of ``axis_pair`` with the rerun's. Both sides must agree.

    Raises:
        ValueError: if a term is on one side only. A rerun that silently lost a term would pass
            on the terms it kept.
    """
    before = _family(original, axis_pair, terms)
    after = _family(rerun, axis_pair, terms)
    if set(before) != set(after):
        only = sorted(set(before) ^ set(after))
        raise ValueError(f"original and rerun disagree on which terms exist: {only[:4]}")
    if not before:
        raise ValueError(f"no family terms of {axis_pair} in the original estimates")
    return [
        R2Comparison(
            model_id=key[0],
            harm_type=key[1],
            term=key[2],
            original_estimate=before[key].estimate,
            original_se=before[key].se,
            original_ci_low=before[key].ci_low,
            original_ci_high=before[key].ci_high,
            rerun_estimate=after[key].estimate,
            rerun_ci_low=after[key].ci_low,
            rerun_ci_high=after[key].ci_high,
        )
        for key in sorted(before)
    ]


def substitute_series(
    inputs: StatsInputs, rerun: EmergenceResult, *, axis_pair: tuple[str, str]
) -> StatsInputs:
    """``inputs`` with the ``axis_pair`` model series replaced by the rerun's, all else unchanged.

    Replaced, per (benchmark, model, harm type) the rerun fitted: that pair's effects, and its
    TRIANGULATED fit terms and fit report (``scorer_id is None``, ``condition == "model"``), which
    are what P9's join check and its degrees of freedom come from. Kept: every other pair, every
    per-scorer series, and the drift comparisons -- so the drift downgrade is the original one, as
    pre-registered ("same drift downgrades").

    Raises:
        ValueError: if the rerun fitted a group the original run never had.
    """
    rerun_groups: set[GroupKey] = {
        (r.benchmark, r.model_id, r.harm_type)
        for r in rerun.fit_reports
        if r.scorer_id is None and r.condition == "model" and (r.axis_a, r.axis_b) == axis_pair
    }
    original_groups: set[GroupKey] = {
        (r.benchmark, r.model_id, r.harm_type)
        for r in inputs.fit_reports
        if r.scorer_id is None and r.condition == "model" and (r.axis_a, r.axis_b) == axis_pair
    }
    if not rerun_groups or not rerun_groups <= original_groups:
        raise ValueError(
            f"rerun groups {sorted(rerun_groups)} are not a non-empty subset of the original "
            f"run's {sorted(original_groups)}"
        )

    def replaced_series(record: FitTerm | FitReport) -> bool:
        return (
            (record.benchmark, record.model_id, record.harm_type) in rerun_groups
            and record.scorer_id is None
            and record.condition == "model"
            and (record.axis_a, record.axis_b) == axis_pair
        )

    def replaced_effect(effect: EffectEstimate) -> bool:
        return (
            effect.benchmark,
            effect.model,
            effect.harm_type,
        ) in rerun_groups and effect.axis_pair == axis_pair

    new_terms = [t for t in rerun.fit_terms if replaced_series(t)]
    new_reports = [r for r in rerun.fit_reports if replaced_series(r)]
    new_effects = [e for e in rerun.effects if replaced_effect(e)]
    return replace(
        inputs,
        effects=[e for e in inputs.effects if not replaced_effect(e)] + new_effects,
        fit_terms=[t for t in inputs.fit_terms if not replaced_series(t)] + new_terms,
        fit_reports=[r for r in inputs.fit_reports if not replaced_series(r)] + new_reports,
    )


def verdict_changes(
    original: Sequence[SignificanceClaim],
    rerun: Sequence[SignificanceClaim],
    *,
    axis_pair: tuple[str, str],
    terms: Sequence[str],
) -> list[R2Verdict]:
    """Every family term of ``axis_pair``: its status before and after substitution."""

    def index(claims: Sequence[SignificanceClaim]) -> dict[tuple[str, str, str], str]:
        wanted = set(terms)
        return {
            (c.model_id, c.harm_type, c.term): c.status
            for c in claims
            if (c.axis_a, c.axis_b) == axis_pair and c.term in wanted
        }

    before, after = index(original), index(rerun)
    if set(before) != set(after) or not before:
        raise ValueError("original and rerun claims disagree on which family terms exist")
    return [R2Verdict(key[0], key[1], key[2], before[key], after[key]) for key in sorted(before)]


def exact_match_rate(
    original: Mapping[str, str], rerun: Mapping[str, str]
) -> tuple[int, int, float]:
    """``(n compared, n byte-identical, rate)`` over the rerun's response ids.

    Raises:
        ValueError: if a rerun id has no original -- the rerun must reproduce existing responses.
    """
    missing = sorted(set(rerun) - set(original))
    if missing:
        raise ValueError(f"{len(missing)} rerun response(s) have no original, e.g. {missing[:2]}")
    if not rerun:
        raise ValueError("no rerun responses to compare")
    identical = sum(1 for rid, text in rerun.items() if original[rid] == text)
    return len(rerun), identical, identical / len(rerun)


def summarize(
    comparisons: Sequence[R2Comparison], verdicts: Sequence[R2Verdict]
) -> dict[str, float | int | bool]:
    """The pre-registered verdict and the always-reported magnitudes."""
    in_se = np.array([c.difference_in_se for c in comparisons])
    n_inside = sum(1 for c in comparisons if c.inside_original_ci)
    n_same = sum(1 for v in verdicts if v.same)
    return {
        "n_terms": len(comparisons),
        "n_inside_ci": n_inside,
        "n_verdicts_same": n_same,
        "max_abs_diff_se": float(in_se.max()),
        "rms_diff_se": float(np.sqrt(np.mean(in_se**2))),
        "passed": n_inside == len(comparisons) and n_same == len(verdicts),
    }
