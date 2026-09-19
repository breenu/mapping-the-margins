"""Section 9's rule, enforced: the scorer identity-drift downgrade (P9).

P7 measured each scorer's own interaction on known-neutral text carrying the same descriptors,
beside that scorer's view of the model, and emitted a ``comparable`` flag under a **placeholder**
threshold. This module turns that into a decision against the **pre-registered** threshold.

The rule, in the form section 9 states it: *if a scorer's interaction is comparable in magnitude
to the model's, that harm type is descriptive-only.* The reasoning is that such a scorer cannot
distinguish emergence in the model from emergence in the instrument, so no inferential claim
about the model survives.

**P7's ``comparable`` column is deliberately ignored.** It was computed under the old number, and
inheriting it would mean the enforced decision silently depended on a threshold nobody
pre-registered. The ratio is recomputed from the raw ``drift_rms`` and ``model_rms`` columns
every time. Where P7's flag and P9's decision disagree, the disagreement is **reported**, not
quietly resolved: two phases differing about a scorer's validity is exactly the kind of thing
that should reach a human.

One edge case worth naming: P7 records ``ratio=None`` when ``model_rms`` is exactly zero. That is
not "no information" -- a scorer with zero interaction on the model and a non-zero one on neutral
text is the worst case, entirely instrument, and it is treated as descriptive-only rather than
skipped.
"""

from __future__ import annotations

from collections.abc import Sequence

from margins.metrics.records import DriftComparison, HarmTypeFlag
from margins.stats.records import DescriptiveOnlyDecision

__all__ = [
    "compare_with_metrics_flags",
    "descriptive_only_decisions",
    "descriptive_only_keys",
]

#: What identifies a harm type for the downgrade: ``(benchmark, model, harm_type)``.
HarmKey = tuple[str, str, str]


def descriptive_only_decisions(
    comparisons: Sequence[DriftComparison],
    flags: Sequence[HarmTypeFlag] = (),
    *,
    threshold: float,
) -> list[DescriptiveOnlyDecision]:
    """One decision per ``(benchmark, model, harm_type)``, against the pre-registered ratio."""
    metrics_flag = {(f.benchmark, f.model_id, f.harm_type): f for f in flags}
    grouped: dict[HarmKey, list[DriftComparison]] = {}
    for comparison in comparisons:
        grouped.setdefault(
            (comparison.benchmark, comparison.model_id, comparison.harm_type), []
        ).append(comparison)

    decisions: list[DescriptiveOnlyDecision] = []
    for key in sorted(grouped):
        rows = grouped[key]
        over: list[DriftComparison] = []
        best: tuple[float, DriftComparison] | None = None
        for row in rows:
            if row.model_rms > 0.0:
                ratio = row.drift_rms / row.model_rms
            elif row.drift_rms > 0.0:
                # The instrument moves where the model does not. The worst case, not a gap.
                ratio = float("inf")
            else:
                continue
            if best is None or ratio > best[0]:
                best = (ratio, row)
            if ratio >= threshold:
                over.append(row)

        benchmark, model_id, harm_type = key
        worst = best[1] if best else None
        max_ratio = best[0] if best else None
        decision = DescriptiveOnlyDecision(
            benchmark=benchmark,
            model_id=model_id,
            harm_type=harm_type,
            descriptive_only=bool(over),
            threshold=threshold,
            max_ratio=None if max_ratio is None or max_ratio == float("inf") else max_ratio,
            worst_scorer=worst.scorer_id if worst else None,
            worst_axis_a=worst.axis_a if worst else None,
            worst_axis_b=worst.axis_b if worst else None,
            worst_control_family=worst.control_family if worst else None,
            n_comparisons=len(rows),
            n_over_threshold=len(over),
            scorers_over_threshold=tuple(sorted({row.scorer_id for row in over})),
            disagrees_with_metrics_flag=(
                key in metrics_flag and metrics_flag[key].descriptive_only_candidate != bool(over)
            ),
        )
        decisions.append(decision)
    return decisions


def descriptive_only_keys(decisions: Sequence[DescriptiveOnlyDecision]) -> frozenset[HarmKey]:
    """The harm types the claim gate must refuse."""
    return frozenset(
        (d.benchmark, d.model_id, d.harm_type) for d in decisions if d.descriptive_only
    )


def compare_with_metrics_flags(decisions: Sequence[DescriptiveOnlyDecision]) -> list[str]:
    """Human-readable lines for every disagreement with P7's candidate flag.

    Returned rather than logged so the run script can print them and the report can carry them.
    A silent disagreement between the phase that measures drift and the phase that enforces it
    would be the worst available outcome.
    """
    lines: list[str] = []
    for decision in decisions:
        if not decision.disagrees_with_metrics_flag:
            continue
        verdict = "DESCRIPTIVE-ONLY" if decision.descriptive_only else "clear"
        lines.append(
            f"{decision.harm_type} ({decision.model_id}): P9 says {verdict} at threshold "
            f"{decision.threshold:g}, P7's candidate flag said the opposite. Max ratio "
            f"{decision.max_ratio if decision.max_ratio is not None else 'infinite'}."
        )
    return lines
