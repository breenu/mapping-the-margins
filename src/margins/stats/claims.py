"""The claim gate: which effects may be cited, and the mechanical refusal of the rest (P9).

Four separate rules say an estimate must not be reported as significant, and the project had them
written down in four places: exploratory pairs carry no significance claim (section 8),
underpowered cells are descriptive-only (invariant 4), descriptive-only harm types cannot
distinguish model from instrument (section 9), and an uncorrected estimate has no ``q`` at all
(invariant 3). This module is the single place they are applied.

**The refusal is mechanical, and that is the whole design.** :meth:`ClaimGate.is_significant`
**raises** on a refused effect rather than returning ``False``. A table builder that forgets to
consult the gate therefore crashes instead of quietly printing a star; a builder that returns
``False`` would let the same bug ship a table that looks complete and is wrong. This is the same
pattern as ``powered_only=True`` on the report API, extended to the other three rules.

**A refusal is a row, never an absence.** :meth:`ClaimGate.claims` emits a
:class:`~margins.stats.records.SignificanceClaim` for *every* effect, refused ones included, with
the reason named. Filtering them out instead would make "we refused to claim this" look identical
to "we never looked", which is precisely the ambiguity a descriptive-only downgrade exists to
avoid.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from margins.schemas import EffectEstimate
from margins.stats.records import ClaimStatus, SignificanceClaim

__all__ = ["ClaimGate", "ClaimRefusedError"]


class ClaimRefusedError(ValueError):
    """Significance was asked of an effect that may not carry a significance claim."""


_REASONS: dict[ClaimStatus, str] = {
    "exploratory": (
        "the axis pair is exploratory. The pre-registered confirmatory set IS the FDR family; "
        "everything outside it is reported descriptively and carries no significance claim."
    ),
    "descriptive_only": (
        "the harm type is DESCRIPTIVE-ONLY: a scorer's own interaction on known-neutral text is "
        "comparable to the model's, so emergence in the model cannot be distinguished from "
        "emergence in the instrument (PROJECT_CONTEXT.md section 9)."
    ),
    "underpowered": (
        "the cell is underpowered against the pre-registered minimum N, and underpowered cells "
        "are descriptive-only and never load-bearing (invariant 4)."
    ),
    "uncorrected": (
        "the estimate carries no q. Significance is set only by StatsEngine (invariant 3), and "
        "an uncorrected estimate has not been through the family."
    ),
}


@dataclass(frozen=True)
class ClaimGate:
    """Decides, for one effect, whether a significance claim is permitted.

    Args:
        exploratory_pairs: axis pairs outside the pre-registered confirmatory set.
        descriptive_only: ``(benchmark, model, harm_type)`` downgraded by the drift rule.
        alpha: the FDR level a claim is judged at.
        powered_only: whether ``powered=False`` refuses a claim. Default True, and opting out
            has to be explicit.
    """

    exploratory_pairs: frozenset[tuple[str, str]]
    descriptive_only: frozenset[tuple[str, str, str]]
    alpha: float
    powered_only: bool = True

    def status(self, effect: EffectEstimate) -> ClaimStatus:
        """The first applicable refusal, in precedence order, or ``claimable``."""
        if effect.axis_pair in self.exploratory_pairs:
            return "exploratory"
        if (effect.benchmark, effect.model, effect.harm_type) in self.descriptive_only:
            return "descriptive_only"
        if self.powered_only and not effect.powered:
            return "underpowered"
        if effect.q is None:
            return "uncorrected"
        return "claimable"

    def refusal_reason(self, effect: EffectEstimate) -> str | None:
        status = self.status(effect)
        return None if status == "claimable" else _REASONS[status]

    def require_claimable(self, effect: EffectEstimate) -> EffectEstimate:
        """Return the effect, or raise naming why it may not be cited."""
        reason = self.refusal_reason(effect)
        if reason is not None:
            raise ClaimRefusedError(
                f"{effect.harm_type} {effect.axis_pair} {effect.term!r}: {reason}"
            )
        return effect

    def is_significant(self, effect: EffectEstimate) -> bool:
        """``q <= alpha``, but **raises** for an effect that may not carry a claim.

        Returning ``False`` would be the friendlier API and the wrong one: a caller that forgot
        the gate would silently render every refused effect as "not significant", which is a
        claim about the model rather than an abstention.
        """
        self.require_claimable(effect)
        return effect.q is not None and effect.q <= self.alpha

    def claims(
        self,
        effects: Sequence[EffectEstimate],
        *,
        permutation_p: Mapping[tuple[str, ...], float] | None = None,
        fdr_by: Mapping[tuple[str, ...], float] | None = None,
    ) -> list[SignificanceClaim]:
        """A row per effect, refused ones included, with the reason on the row."""
        permutation = permutation_p or {}
        by_values = fdr_by or {}
        rows: list[SignificanceClaim] = []
        for effect in effects:
            status = self.status(effect)
            key = tuple(
                str(part)
                for part in (
                    effect.benchmark,
                    effect.model,
                    effect.axis_pair,
                    effect.term,
                    effect.harm_type,
                )
            )
            rows.append(
                SignificanceClaim(
                    benchmark=effect.benchmark,
                    model_id=effect.model,
                    harm_type=effect.harm_type,
                    axis_a=effect.axis_pair[0],
                    axis_b=effect.axis_pair[1],
                    term=effect.term,
                    estimate=effect.estimate,
                    se=effect.se,
                    ci_low=effect.ci_low,
                    ci_high=effect.ci_high,
                    p=effect.p,
                    q=effect.q,
                    q_fdr_by=by_values.get(key),
                    p_permutation=permutation.get(key),
                    powered=effect.powered,
                    status=status,
                    claimable=status == "claimable",
                    refusal_reason=None if status == "claimable" else _REASONS[status],
                )
            )
        return rows
