"""Within-benchmark standardization, and the sign contract (P6).

Two rules from CLAUDE.md meet here, and both are enforced rather than documented:

1. **"Never standardize harm scores across sources -- standardize WITHIN a source (the function
   requires a source key)."** :func:`standardize_within_benchmark` takes ``benchmark`` as a required
   keyword and raises if any row disagrees with it. Pooling two sources into one z-score would make
   the cross-paradigm comparison a comparison of the pooling, not of the paradigms -- and it is the
   kind of mistake that produces entirely reasonable-looking output.

2. **"higher = more harm, guaranteed by test."** Each scorer declares a ``sign``; the standardized
   value is ``sign * z``. A ``sign`` alone is an assertion, so every roster entry also carries a
   :class:`~margins.scoring.config.SignProbe` and :func:`verify_sign` turns the assertion into a
   check that can fail. A mis-signed source silently inverts every effect downstream while the
   pipeline stays green, which is why this is a gate and not a comment.

**Standardization is per (benchmark, scorer, harm_type, condition) group.** Per scorer because raw
scales are incomparable -- a toxicity probability lives in [0, 1] and a valence score in [-1, 1], so
one pooled z-score would be dominated by whichever has the wider spread. Per condition because the
drift control's neutral texts and the model's generations are different populations; z-scoring them
together would subtract a mean that is partly the control's, which would smuggle in exactly the
point-subtraction section 9 says not to do.

**Degenerate groups.** A group whose raw scores are all identical has zero variance and no defined
z-score. Rather than emit NaN (which propagates silently into every mean downstream) or divide by a
fudge factor (which invents spread that is not there), such a group standardizes to all-zeros and is
reported in the returned :class:`StandardizationReport`. Zero is the honest answer: every row is
exactly at its group mean.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from margins.schemas import ItemScore
from margins.scoring.base import TextScorer
from margins.scoring.config import ScorerSpec

__all__ = [
    "MIN_GROUP_FOR_VARIANCE",
    "SignContractError",
    "StandardizationReport",
    "signs_from_specs",
    "standardize_within_benchmark",
    "verify_sign",
]


class SignContractError(ValueError):
    """Raised when a scorer's declared sign contradicts its own probe.

    Its own class because this is not an input-validation problem a caller can fix by passing
    different data -- it means the roster describes the instrument incorrectly, and every number the
    instrument has produced has the wrong sign.
    """


#: Below this a standard deviation is not meaningfully estimable. Two points always have a defined
#: sd, but z-scoring them just relabels them +/-1 regardless of how far apart they are.
MIN_GROUP_FOR_VARIANCE: Final[int] = 3


class StandardizationReport(BaseModel):
    """What standardization did, per group. Emitted so degenerate groups are visible, not silent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    benchmark: str
    scorer_id: str
    harm_type: str
    condition: str
    n: int = Field(ge=0)
    raw_mean: float
    raw_sd: float
    sign: int
    degenerate: bool = Field(
        description="True when sd was 0 or n was too small; every row standardized to 0.0."
    )


def _group_key(score: ItemScore) -> tuple[str, str, str, str]:
    return (score.benchmark, score.scorer_id, score.harm_type, score.condition)


def standardize_within_benchmark(
    scores: Sequence[ItemScore],
    *,
    benchmark: str,
    signs: dict[str, int],
) -> tuple[list[ItemScore], list[StandardizationReport]]:
    """Standardize ``raw`` into ``standardized`` within one benchmark, sign-harmonized.

    Args:
        scores: rows to standardize. Every row must carry ``benchmark == benchmark``.
        benchmark: the source key. **Required** -- see rule 1 in the module docstring. There is
            deliberately no default and no inference of it from the rows: a function that can guess
            its own grouping key is a function that can be called wrong without anyone noticing.
        signs: ``scorer_id -> +1 | -1``, from the roster. Every scorer present must appear.

    Returns:
        ``(standardized_scores, reports)``. New frozen records via ``model_copy`` (invariant 1);
        the inputs are not mutated. Output order matches input order.

    Raises:
        ValueError: if any row belongs to another benchmark, or a scorer has no declared sign.
    """
    if not benchmark:
        raise ValueError("benchmark must be a non-empty source key")

    foreign = sorted({s.benchmark for s in scores if s.benchmark != benchmark})
    if foreign:
        raise ValueError(
            f"standardize_within_benchmark({benchmark!r}) was given rows from {foreign}. "
            "Standardization is WITHIN a source, never across: pooling sources into one z-score "
            "makes the cross-paradigm comparison a property of the pooling (CLAUDE.md guardrails)."
        )

    unsigned = sorted({s.scorer_id for s in scores if s.scorer_id not in signs})
    if unsigned:
        raise ValueError(
            f"no declared sign for scorer(s) {unsigned}. Every scorer must declare whether its raw "
            "output rises or falls with harm, or 'higher = more harm' cannot be guaranteed."
        )

    groups: dict[tuple[str, str, str, str], list[int]] = {}
    for index, score in enumerate(scores):
        groups.setdefault(_group_key(score), []).append(index)

    standardized: list[float] = [0.0] * len(scores)
    reports: list[StandardizationReport] = []

    for key in sorted(groups):
        indices = groups[key]
        raw = np.asarray([scores[i].raw for i in indices], dtype=float)
        sign = signs[key[1]]
        mean = float(raw.mean())
        # Population sd (ddof=0): these are all the rows in the group, not a sample from it.
        sd = float(raw.std(ddof=0))
        degenerate = sd == 0.0 or len(indices) < MIN_GROUP_FOR_VARIANCE

        values = np.zeros_like(raw) if degenerate else sign * (raw - mean) / sd
        for position, index in enumerate(indices):
            standardized[index] = float(values[position])

        reports.append(
            StandardizationReport(
                benchmark=key[0],
                scorer_id=key[1],
                harm_type=key[2],
                condition=key[3],
                n=len(indices),
                raw_mean=mean,
                raw_sd=sd,
                sign=sign,
                degenerate=degenerate,
            )
        )

    out = [
        score.model_copy(update={"standardized": value})
        for score, value in zip(scores, standardized, strict=True)
    ]
    return out, reports


def verify_sign(spec: ScorerSpec, text_scorer: TextScorer) -> None:
    """Check a scorer's declared sign against its own probe. Raises on a mis-signed scorer.

    This is the executable form of "higher = more harm, guaranteed by test". It runs the probe's
    two texts through the real instrument and asserts that, **after applying the declared sign**,
    the harmful one scores above the benign one.

    Note it checks the sign-adjusted *raw* scores rather than standardized ones. Two points cannot
    be standardized meaningfully (:data:`MIN_GROUP_FOR_VARIANCE`), and the polarity question is
    about the scorer's own direction, which z-scoring preserves exactly.

    Raises:
        SignContractError: if the probe's harmful text does not score above its benign text.
    """
    probe = spec.sign_probe
    harmful_raw, benign_raw = text_scorer.score_texts([probe.harmful, probe.benign])
    harmful = spec.sign * harmful_raw
    benign = spec.sign * benign_raw

    if harmful <= benign:
        raise SignContractError(
            f"scorer {spec.scorer_id!r} declares sign={spec.sign:+d}, but its own probe comes out "
            f"the wrong way round: harmful text scores {harmful:.6g} and benign text {benign:.6g} "
            f"after applying the sign (raw {harmful_raw:.6g} vs {benign_raw:.6g}). "
            "'higher = more harm' is violated, so every effect computed from this scorer "
            "would have the wrong sign -- and would look entirely plausible. "
            "Fix `sign` in the roster, or the "
            "probe if the probe is what is wrong."
        )


def signs_from_specs(specs: Sequence[ScorerSpec]) -> dict[str, int]:
    """``scorer_id -> sign`` for a roster, in the shape :func:`standardize_within_benchmark`
    wants."""
    return {spec.scorer_id: spec.sign for spec in specs}
