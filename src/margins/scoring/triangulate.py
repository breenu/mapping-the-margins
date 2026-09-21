"""Running the roster, and reporting inter-scorer agreement (P6).

Agreement is a **first-class reported artifact**, not an internal sanity check. Gate P6 requires
"inter-scorer alpha/kappa reported", and section 9 requires poor-agreement harm types to be
downgraded to descriptive -- neither is possible if the number only ever exists inside an assertion.
So :meth:`Triangulator.agreement` returns records, :mod:`margins.scoring.export` writes them, and
the manifest carries the summary.

**What agreement does and does not license.** Two scorers agreeing means they behave alike. It does
NOT mean they are measuring the harm correctly: section 9 is explicit that two identity-sensitive
classifiers agree with each other and are wrong in the same direction. High alpha is therefore a
*precondition* for treating a harm type as load-bearing, never evidence that it is valid -- that
question belongs to the drift control (:mod:`margins.scoring.drift`) and to human calibration
(:mod:`margins.scoring.calibration`). The roster's mechanism-diversity rule exists so that agreement
here is at least agreement between two different ways of being wrong.

**Two statistics, because they answer different questions.**

* **Krippendorff's alpha** (interval) over ``standardized``, units = responses, raters = scorers.
  It is the headline: it handles the continuous signal the metric layer actually consumes, and it is
  defined for any number of raters.
* **Cohen's kappa** (nominal) per scorer *pair*, after binarizing ``raw`` at each scorer's declared
  threshold. Pairwise and categorical, so it says which specific pair disagrees, and it speaks the
  language of the calibration report.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from margins.schemas import Condition, ItemScore, Prompt, Response
from margins.scoring.base import RosterScorer
from margins.scoring.config import ScoringConfig
from margins.scoring.standardize import (
    StandardizationReport,
    signs_from_specs,
    standardize_within_benchmark,
)

__all__ = [
    "MIN_UNITS_FOR_AGREEMENT",
    "AgreementRecord",
    "Triangulator",
    "build_triangulator",
    "cohen_kappa",
    "krippendorff_alpha",
]

#: Below this, an agreement coefficient is noise. Reported as ``None`` with ``n_units`` recorded
#: rather than as a number, because a kappa computed on four responses invites exactly the
#: over-reading that the powered/descriptive split exists to prevent.
MIN_UNITS_FOR_AGREEMENT: Final[int] = 10


class AgreementRecord(BaseModel):
    """One agreement coefficient.

    A stage-local record, deliberately not in :mod:`margins.schemas`: invariant 2 keeps the shared
    contract layer free of per-stage shapes, and nothing downstream of P6 consumes this as a record
    -- P7 reads the emitted table.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    benchmark: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    harm_type: str = Field(min_length=1)
    condition: Condition
    metric: str = Field(min_length=1, description="'krippendorff_alpha' or 'cohen_kappa'.")
    scorer_a: str = Field(
        min_length=1, description="Single scorer pair member, or 'ALL' for alpha."
    )
    scorer_b: str | None = Field(default=None, description="None for the all-rater alpha.")
    value: float | None = Field(
        default=None, description="None when n_units is below MIN_UNITS_FOR_AGREEMENT."
    )
    n_units: int = Field(ge=0, description="Responses with a score from every scorer compared.")
    n_raters: int = Field(ge=0)


def krippendorff_alpha(matrix: np.ndarray) -> float:
    """Krippendorff's alpha for interval data. ``matrix`` is ``(raters, units)``, NaN for missing.

    **Interval, not nominal**: the standardized score is a continuous signed index, and treating it
    as categories would discard the magnitude of a disagreement -- which is exactly what matters
    when deciding whether a harm type is load-bearing.

    **Computed in closed form rather than through the `krippendorff` package, for a reason that only
    shows up at scale.** That package builds a coincidence matrix over *distinct observed values*.
    With continuous scores every value is distinct, so V grows with the number of units and the
    matrix is O(n * V**2): 1,430 units already asks for 22 GiB, and the confirmatory design is
    ~49,500 renderings. Subsampling to fit would make the reported alpha a property of the
    subsample, so the arithmetic is done directly instead.

    For the interval metric d(c,k) = (c - k)**2 the coincidence sums collapse to sums of squares:

        sum_{i != j} (v_i - v_j)**2  ==  2 * (m * sum(v**2) - sum(v)**2)

    which makes both disagreement terms O(n). ``tests/test_triangulation.py`` checks this against
    the `krippendorff` package on inputs small enough for it, so the dependency is still what
    validates the implementation -- it is an oracle here rather than the production path.

    Returns NaN when expected disagreement is zero (no variance anywhere); callers report that as
    "not estimable" rather than as a number.
    """
    observed = 0.0
    n = 0
    pooled_sum = 0.0
    pooled_square_sum = 0.0

    for unit in range(matrix.shape[1]):
        values = matrix[:, unit]
        values = values[~np.isnan(values)]
        count = values.size
        if count < 2:
            # A unit rated once carries no information about disagreement and is excluded, which
            # is what the coincidence-matrix construction does with it too.
            continue
        total = float(values.sum())
        square_total = float((values**2).sum())
        observed += 2.0 * (count * square_total - total**2) / (count - 1)
        pooled_sum += total
        pooled_square_sum += square_total
        n += count

    if n < 2:
        return float("nan")

    expected = 2.0 * (n * pooled_square_sum - pooled_sum**2) / (n - 1)
    if expected == 0.0:
        return float("nan")
    return 1.0 - (observed / expected)


def cohen_kappa(labels_a: Sequence[int], labels_b: Sequence[int]) -> float:
    """Cohen's kappa between two binary raters.

    Returns 0.0 when both raters give a single constant label across every unit. scikit-learn
    returns NaN there (the expected-agreement denominator vanishes); 0.0 is the honest reading --
    two raters who never vary agree no better than chance would predict, and NaN would propagate
    into the report as a blank where a reader expects a number.
    """
    from sklearn.metrics import cohen_kappa_score

    if len(set(labels_a)) == 1 and len(set(labels_b)) == 1:
        return 0.0
    value = float(cohen_kappa_score(list(labels_a), list(labels_b)))
    return 0.0 if np.isnan(value) else value


class Triangulator:
    """Runs every scorer over every response, then reports how much they agreed.

    Scorers are constructed once and reused across conditions via
    :meth:`~margins.scoring.base.ResponseScorer.for_condition`, so the drift control measures the
    same instrument that produced the model scores rather than a second instance of it.
    """

    def __init__(
        self,
        config: ScoringConfig,
        scorers: Sequence[RosterScorer],
        *,
        benchmark: str,
    ) -> None:
        if not scorers:
            raise ValueError("Triangulator needs at least one scorer")

        roster_ids = {spec.scorer_id for spec in config.scorers}
        built_ids = {scorer.scorer_id for scorer in scorers}
        if not built_ids <= roster_ids:
            raise ValueError(
                f"scorers {sorted(built_ids - roster_ids)} are not in roster {config.key!r}. "
                "Every score row's scorer_id must resolve to a pinned roster entry, or the "
                "manifest cannot say what produced it."
            )

        # The >=2-per-harm rule is enforced on the CONFIG at load, but a caller can build a subset
        # of it. Re-check on what was actually built: a run that scored one harm type with one
        # scorer would satisfy the config and still violate framing constraint 6.
        for harm_type in sorted({scorer.harm_type for scorer in scorers}):
            group = [s for s in scorers if s.harm_type == harm_type]
            if len(group) < 2:
                raise ValueError(
                    f"harm_type {harm_type!r} was built with {len(group)} scorer(s). No harm type "
                    "may depend on a single classifier (CLAUDE.md framing constraint 6), and a "
                    "subset of a valid roster can still violate it."
                )

        self._config = config
        self._scorers = tuple(scorers)
        self._benchmark = benchmark

    @property
    def benchmark(self) -> str:
        return self._benchmark

    @property
    def scorers(self) -> tuple[RosterScorer, ...]:
        return self._scorers

    def score_all(
        self, responses: Sequence[Response], *, condition: Condition = "model"
    ) -> tuple[list[ItemScore], list[StandardizationReport]]:
        """Score every response with every scorer, then standardize within the benchmark.

        Returns ``(scores, standardization_reports)``. One row per (response, scorer) -- **never an
        average over responses sharing a prompt**, because distinct seeds are distinct observations
        and collapsing them is the metric layer's decision to make, not this one's.
        """
        raw_scores: list[ItemScore] = []
        for scorer in self._scorers:
            active = scorer if scorer.condition == condition else scorer.for_condition(condition)
            raw_scores.extend(active.score(responses))

        return standardize_within_benchmark(
            raw_scores,
            benchmark=self._benchmark,
            signs=signs_from_specs([s.spec for s in self._scorers]),
        )

    def agreement(self, scores: Sequence[ItemScore]) -> list[AgreementRecord]:
        """Inter-scorer agreement per (model, harm type, condition). Alpha over all, kappa
        pairwise."""
        records: list[AgreementRecord] = []

        keys = sorted({(s.model_id, s.harm_type, s.condition) for s in scores})
        for model_id, harm_type, condition in keys:
            group = [
                s
                for s in scores
                if s.model_id == model_id and s.harm_type == harm_type and s.condition == condition
            ]
            scorer_ids = sorted({s.scorer_id for s in group})
            if len(scorer_ids) < 2:
                continue

            # Units are responses scored by EVERY scorer being compared. A response missing one
            # scorer's row is dropped rather than imputed: alpha tolerates missing data, but a
            # kappa computed on differently-sized vectors is not a kappa.
            by_scorer: dict[str, dict[str, ItemScore]] = {
                scorer_id: {s.response_id: s for s in group if s.scorer_id == scorer_id}
                for scorer_id in scorer_ids
            }
            common = sorted(set.intersection(*(set(rows) for rows in by_scorer.values())))
            n_units = len(common)
            enough = n_units >= MIN_UNITS_FOR_AGREEMENT

            matrix = np.asarray(
                [[by_scorer[sid][rid].standardized for rid in common] for sid in scorer_ids],
                dtype=float,
            )
            alpha: float | None = None
            if enough:
                # Alpha is undefined when expected disagreement is zero -- every rating identical,
                # which happens with a degenerate standardization group. The closed form returns
                # NaN there; report None, because a blank reads as "not estimable" where a number
                # would read as a finding.
                value = krippendorff_alpha(matrix)
                alpha = None if np.isnan(value) else value

            records.append(
                AgreementRecord(
                    benchmark=self._benchmark,
                    model_id=model_id,
                    harm_type=harm_type,
                    condition=condition,
                    metric="krippendorff_alpha",
                    scorer_a="ALL",
                    scorer_b=None,
                    value=alpha,
                    n_units=n_units,
                    n_raters=len(scorer_ids),
                )
            )

            thresholds = {
                spec.scorer_id: spec.threshold
                for spec in self._config.scorers
                if spec.scorer_id in by_scorer
            }
            for index, first in enumerate(scorer_ids):
                for second in scorer_ids[index + 1 :]:
                    labels_a = [
                        int(by_scorer[first][rid].raw >= thresholds[first]) for rid in common
                    ]
                    labels_b = [
                        int(by_scorer[second][rid].raw >= thresholds[second]) for rid in common
                    ]
                    records.append(
                        AgreementRecord(
                            benchmark=self._benchmark,
                            model_id=model_id,
                            harm_type=harm_type,
                            condition=condition,
                            metric="cohen_kappa",
                            scorer_a=first,
                            scorer_b=second,
                            value=cohen_kappa(labels_a, labels_b) if enough else None,
                            n_units=n_units,
                            n_raters=2,
                        )
                    )

        return records


def build_triangulator(
    config: ScoringConfig,
    *,
    prompts_by_id: dict[str, Prompt],
    benchmark: str,
    data_dir: object = None,
) -> Triangulator:
    """Construct every scorer in a roster and wrap them in a :class:`Triangulator`.

    Dispatches on the roster entry's mechanism, so a mixed roster would build a mixed triangulator.
    In practice the two paradigms ship separate rosters -- ``scoring_hash`` rides in every
    ``score_id``, so folding the paired scorers into ``default_v1`` would change the id of every
    score the generation vertical already produced.
    """
    from margins.scoring.base import scorer_from_spec
    from margins.scoring.paired import paired_scorer_from_spec

    scorers: list[RosterScorer] = []
    for spec in config.scorers:
        if spec.reads_logprobs:
            scorers.append(
                paired_scorer_from_spec(
                    spec,
                    prompts_by_id=prompts_by_id,
                    benchmark=benchmark,
                    scoring_hash=config.scoring_hash,
                )
            )
        else:
            scorers.append(
                scorer_from_spec(
                    spec,
                    prompts_by_id=prompts_by_id,
                    benchmark=benchmark,
                    scoring_hash=config.scoring_hash,
                    data_dir=data_dir,
                )
            )
    return Triangulator(config, scorers, benchmark=benchmark)
