"""Which terms constitute the FDR family, and how big it is (P9, section 8).

The confirmatory axis pairs **are** the family. That is the single highest-leverage decision in the
project: an unbounded family over the whole lattice drives the Benjamini-Hochberg threshold so low
that realized power collapses regardless of per-cell N. This module is where the set is enumerated,
from the lattice's own ``confirmatory`` declaration, before any estimate is looked at.

**Which terms of a pair — decided 19 Sep 2026 with the human, before any P9 code.** The default is
``free_interaction``: the ``(|A|-1)(|B|-1)`` identified interaction contrasts, which are exactly
the design's own free interaction columns. The choice is canonical rather than arbitrary, and the
two wider options were rejected on the record:

* ``all_interaction`` (every ``|A| x |B|`` cell) tests a **linearly dependent** set -- P7 emits
  every level's coefficient and the gamma table's rows and columns sum to zero -- so the extra
  members carry no extra information while costing power on the primary claim;
* ``all_terms`` additionally folds in the main effects, which are B1's single-axis baseline and not
  the emergence estimand (section 10).

Both remain selectable so the choice is a config line a reviewer can see, not a hard-coded
assumption, and so a sensitivity analysis can be run under either. The dependent gammas that fall
outside the family are still **reported**, with their cluster-bootstrap CIs, descriptively.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from margins.lattice.config import LatticeConfig
from margins.metrics.design import PairDesign, interaction_term, main_term
from margins.stats.config import FamilyTerms, FdrSettings
from margins.stats.inputs import EffectKey
from margins.stats.records import FamilyMember

__all__ = [
    "FamilyPlan",
    "IncompleteFamilyError",
    "UnknownFamilyMemberError",
    "build_family_plan",
    "family_size",
    "family_terms_of",
    "free_interaction_terms",
]


def free_interaction_terms(design: PairDesign) -> tuple[str, ...]:
    """The ``(|A|-1)(|B|-1)`` identified interaction contrasts of a pair, in design order.

    These are the design's free interaction parameters: levels ``0 .. |A|-2`` of A crossed with
    levels ``0 .. |B|-2`` of B. The omitted last level of each axis is implied by the sum-to-zero
    constraint, so nothing is lost -- its coefficient is still emitted and reported, it simply does
    not appear twice in a family that controls a false-discovery rate.
    """
    return tuple(
        interaction_term(design.axis_a, a, design.axis_b, b)
        for a in design.levels_a[:-1]
        for b in design.levels_b[:-1]
    )


def family_terms_of(design: PairDesign, *, family_terms: FamilyTerms) -> tuple[str, ...]:
    """The family's term strings for one pair, under the configured selection rule."""
    if family_terms == "free_interaction":
        return free_interaction_terms(design)

    interactions = tuple(
        interaction_term(design.axis_a, a, design.axis_b, b)
        for a in design.levels_a
        for b in design.levels_b
    )
    if family_terms == "all_interaction":
        return interactions

    mains = tuple(main_term(design.axis_a, a) for a in design.levels_a) + tuple(
        main_term(design.axis_b, b) for b in design.levels_b
    )
    return mains + interactions


def family_size(designs: Sequence[PairDesign], *, family_terms: FamilyTerms, n_groups: int) -> int:
    """``n_groups`` x the total term count over ``designs``.

    ``n_groups`` is the number of ``(benchmark, model, harm_type)`` combinations the family spans;
    section 8 defines the family as ``benchmark x model x axis_pair x term x harm_type`` across the
    entire pre-registered set, never per-source.
    """
    if n_groups < 1:
        raise ValueError("a family spans at least one (benchmark, model, harm_type) group")
    per_pair = sum(len(family_terms_of(d, family_terms=family_terms)) for d in designs)
    return n_groups * per_pair


# --------------------------------------------------------------------------------------
# The declared family, and the subset guard
# --------------------------------------------------------------------------------------


class IncompleteFamilyError(ValueError):
    """A declared family member did not arrive, and was not excused by P7."""


class UnknownFamilyMemberError(ValueError):
    """An estimate was offered for correction that the declared family does not contain."""


@dataclass(frozen=True)
class FamilyPlan:
    """The FDR family, enumerated from the LATTICE before any estimate is looked at.

    This is what makes "correcting a subset must raise" checkable without ``correct()`` touching
    a file. The plan knows every slot the family should have; an estimate set that is missing one
    is a smaller family than was pre-registered, and a smaller family means a more lenient BH
    threshold and more discoveries. That is the single easiest way to accidentally inflate a false
    discovery rate, so it is an error rather than a warning.

    A pair P7 reported as **not estimable** is a different case and is recorded as
    ``excused_not_estimable``, carrying P7's own reason. Those slots leave the family, and
    ``size`` shrinks accordingly -- which is correct: a test that could not be run is not a test
    that was run and ignored.
    """

    members: tuple[FamilyMember, ...]
    confirmatory_pairs: frozenset[tuple[str, str]]
    exploratory_pairs: frozenset[tuple[str, str]]

    @property
    def size(self) -> int:
        """Members actually present. This is what is stamped as ``family_size``."""
        return sum(1 for member in self.members if member.status == "present")

    @property
    def excused(self) -> tuple[FamilyMember, ...]:
        return tuple(m for m in self.members if m.status == "excused_not_estimable")

    def expected_keys(self) -> frozenset[EffectKey]:
        return frozenset(
            (m.benchmark, m.model_id, (m.axis_a, m.axis_b), m.term, m.harm_type)
            for m in self.members
            if m.status == "present"
        )

    def is_confirmatory(self, key: EffectKey) -> bool:
        return key[2] in self.confirmatory_pairs

    def validate(self, keys: Collection[EffectKey]) -> None:
        """Refuse a family that is not the declared one, in either direction."""
        offered = frozenset(keys)
        expected = self.expected_keys()
        missing = sorted(expected - offered)
        unknown = sorted(offered - expected)
        if unknown:
            raise UnknownFamilyMemberError(
                f"{len(unknown)} estimate(s) are not in the declared family, e.g. {unknown[0]}. "
                "The family is the lattice's confirmatory pairs and their free interaction "
                "contrasts; anything else is exploratory and carries no significance claim."
            )
        if missing:
            raise IncompleteFamilyError(
                f"{len(missing)} declared family member(s) did not arrive, e.g. {missing[0]}. "
                "Correcting a subset would apply a Benjamini-Hochberg threshold computed over a "
                "SMALLER family than was pre-registered, which inflates the realized false "
                "discovery rate. If a pair is genuinely not estimable, P7 must say so in "
                "fit_reports.parquet and it will be excused."
            )


def build_family_plan(
    designs: Sequence[PairDesign],
    *,
    lattice_config: LatticeConfig,
    groups: Sequence[tuple[str, str, str]],
    settings: FdrSettings,
    not_estimable: Mapping[tuple[str, str, str, str, str], str] | None = None,
) -> FamilyPlan:
    """Enumerate the family from the lattice's declared pairs.

    Args:
        designs: every pair design, confirmatory and exploratory alike.
        lattice_config: supplies which pairs are confirmatory. **The single source.**
        groups: the ``(benchmark, model, harm_type)`` combinations the family spans.
        settings: which terms of a pair enter.
        not_estimable: ``series -> reason`` from P7, for pairs that could not be fitted.
    """
    confirmatory = frozenset(lattice_config.confirmatory_pairs)
    exploratory = frozenset(lattice_config.exploratory_pairs)
    excuses = dict(not_estimable or {})

    members: list[FamilyMember] = []
    for benchmark, model_id, harm_type in groups:
        for design in designs:
            if design.axis_pair not in confirmatory:
                continue
            reason = excuses.get((benchmark, model_id, harm_type, design.axis_a, design.axis_b))
            for term in family_terms_of(design, family_terms=settings.family_terms):
                members.append(
                    FamilyMember(
                        benchmark=benchmark,
                        model_id=model_id,
                        harm_type=harm_type,
                        axis_a=design.axis_a,
                        axis_b=design.axis_b,
                        term=term,
                        status="present" if reason is None else "excused_not_estimable",
                        reason=reason,
                    )
                )
    return FamilyPlan(
        members=tuple(members),
        confirmatory_pairs=confirmatory,
        exploratory_pairs=exploratory,
    )
