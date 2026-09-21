"""Benjamini-Hochberg over the pre-registered family. **The only place significance is set** (P9).

Invariant 3, in one sentence: ``q``, ``family_size`` and ``correction`` are written here and
nowhere else in the codebase. ``tests/test_fdr_correction.py`` enforces that by scanning the
source tree rather than by trusting the convention.

Three properties this module is built around.

**It is pure.** No file access, no RNG, no clock. Everything it needs beyond its argument is
constructor state: a :class:`~margins.stats.config.StatsConfig` and a prebuilt
:class:`~margins.stats.family.FamilyPlan`. That is what makes "correcting a subset raises"
checkable at all -- the plan knows the whole family, so a missing member is detectable without
going back to disk to look for it.

**Exploratory pairs pass through untouched, as the same objects.** Not copied with nulls: the
identical records. An exploratory effect that came out of ``correct()`` carrying a ``q`` would be
a significance claim the pre-registration forbids, and returning the input object makes that
impossible rather than merely unlikely.

**Benjamini-Yekutieli is computed alongside, never instead.** Section 8 requires either an
explicit PRDS statement or the unconditional guarantee; the config forces the statement and this
reports BY as a sensitivity column, so a claim that survives only under the weaker assumption is
visible as such.
"""

from __future__ import annotations

from collections.abc import Sequence

from margins.schemas import EffectEstimate
from margins.stats.config import StatsConfig
from margins.stats.family import FamilyPlan
from margins.stats.inputs import effect_key

__all__ = [
    "AlreadyCorrectedError",
    "BenjaminiHochbergEngine",
    "MissingPValueError",
    "fdr_by_values",
]


class AlreadyCorrectedError(ValueError):
    """An estimate already carries a ``q``; correcting twice would restate the family."""


class MissingPValueError(ValueError):
    """A confirmatory estimate reached the engine with no ``p`` to correct."""


class BenjaminiHochbergEngine:
    """Satisfies :class:`~margins.interfaces.StatsEngine`.

    Args:
        config: supplies ``alpha`` and the method. Frozen for the run.
        plan: the declared family. Built from the lattice before any estimate is read.
    """

    def __init__(self, config: StatsConfig, *, plan: FamilyPlan) -> None:
        self._config = config
        self._plan = plan

    @property
    def config(self) -> StatsConfig:
        return self._config

    @property
    def plan(self) -> FamilyPlan:
        return self._plan

    def correct(self, effects: Sequence[EffectEstimate]) -> list[EffectEstimate]:
        """Stamp ``q``, ``family_size`` and ``correction`` on the confirmatory family.

        Exploratory estimates are returned unchanged and in place. Input order is preserved, so a
        caller can zip the result against what it passed in.

        **Membership, not the axis pair, selects what gets corrected** (fixed 21 Sep 2026). A
        confirmatory pair's fit also emits main effects and every dependent interaction cell, and
        those are NOT family members: they are returned unchanged, with their CIs, carrying no p
        and no q -- exactly as an exploratory pair's estimates are. Selecting by pair offered them
        for correction and made :meth:`FamilyPlan.validate` raise on any real P7 artifact.

        Raises:
            MissingPValueError: a family member has no ``p``.
            AlreadyCorrectedError: a family member already has a ``q``.
            IncompleteFamilyError / UnknownFamilyMemberError: the offered set is not the family.
        """
        from statsmodels.stats.multitest import multipletests

        confirmatory_index = [
            index
            for index, effect in enumerate(effects)
            if self._plan.is_family_member(effect_key(effect))
        ]
        confirmatory = [effects[index] for index in confirmatory_index]

        uncorrected = [e for e in confirmatory if e.p is None]
        if uncorrected:
            raise MissingPValueError(
                f"{len(uncorrected)} confirmatory estimate(s) have no p-value, e.g. "
                f"{effect_key(uncorrected[0])}. A p must be attached (margins.stats.wald) before "
                "the family can be corrected; there is nothing to correct otherwise."
            )
        already = [e for e in confirmatory if e.q is not None]
        if already:
            raise AlreadyCorrectedError(
                f"{len(already)} estimate(s) already carry a q, e.g. {effect_key(already[0])}. "
                "Correcting twice would compute a threshold over an already-thresholded family."
            )

        self._plan.validate([effect_key(effect) for effect in confirmatory])

        p_values = [float(effect.p) for effect in confirmatory if effect.p is not None]
        method = self._config.fdr.method
        alpha = self._config.fdr.alpha
        _, q_values, _, _ = multipletests(p_values, alpha=alpha, method=method)
        size = self._plan.size

        corrected = list(effects)
        for position, index in enumerate(confirmatory_index):
            corrected[index] = effects[index].model_copy(
                update={
                    "q": float(q_values[position]),
                    "family_size": size,
                    "correction": method,
                }
            )
        return corrected


def fdr_by_values(effects: Sequence[EffectEstimate]) -> dict[tuple[str, ...], float]:
    """Benjamini-Yekutieli q-values for the same set, as a **sensitivity column**.

    Deliberately NOT returned as ``EffectEstimate.q``: that field records the pre-registered
    correction, and a second set of q-values living in the same field would make "which
    correction produced this number" unanswerable from the record. Keyed by the effect key so
    the report layer can put the two side by side.
    """
    from statsmodels.stats.multitest import multipletests

    usable = [effect for effect in effects if effect.p is not None]
    if not usable:
        return {}
    _, q_values, _, _ = multipletests(
        [float(effect.p) for effect in usable if effect.p is not None], method="fdr_by"
    )
    return {
        tuple(str(part) for part in effect_key(effect)): float(q)
        for effect, q in zip(usable, q_values, strict=True)
    }
