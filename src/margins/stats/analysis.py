"""Orchestration: read P7, attach p, correct, permute, enforce, gate (P9).

Kept out of ``scripts/run_stats.py`` for the reason the metric layer keeps ``compute_result``
out of its script: the whole pipeline should be importable and testable without argparse, and a
run script should be printing and manifest-writing only.

The order is not arbitrary and each step depends on the one before:

1. **join and check** -- every effect must match its triangulated fit term to 1e-12, or the
   degrees of freedom about to be attached belong to a different series;
2. **declare the family from the lattice**, before looking at what arrived, so a missing member
   is detectable;
3. **attach p** to confirmatory effects only, because an exploratory p with no q beside it
   invites the reading the split exists to prevent;
4. **correct**, which is the only place q, family_size and correction are set;
5. **permute (B3)** every family term from the item-level scores, when they are given -- the
   assumption-light second opinion that is reported beside the GLM p and never corrected;
6. **enforce the drift rule** against the pre-registered threshold, recomputed from raw columns;
7. **gate**, emitting a row for every effect including the refused ones.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from margins.lattice.config import LatticeConfig
from margins.metrics.design import PairDesign
from margins.schemas import EffectEstimate, IdentityCell, ItemScore
from margins.stats.b3 import B3Result, run_b3
from margins.stats.claims import ClaimGate
from margins.stats.config import StatsConfig
from margins.stats.correct import BenjaminiHochbergEngine, fdr_by_values
from margins.stats.descriptive import (
    compare_with_metrics_flags,
    descriptive_only_decisions,
    descriptive_only_keys,
)
from margins.stats.family import FamilyPlan, build_family_plan
from margins.stats.inputs import (
    StatsInputs,
    check_effects_match_fits,
    cluster_df_by_series,
    effect_key,
    not_estimable_reasons,
)
from margins.stats.records import DescriptiveOnlyDecision, FamilyMember, SignificanceClaim
from margins.stats.wald import attach_wald_p

__all__ = ["StatsResult", "run_stats_analysis"]


@dataclass
class StatsResult:
    """Everything the stats stage produces."""

    corrected: list[EffectEstimate] = field(default_factory=list)
    family: list[FamilyMember] = field(default_factory=list)
    descriptive_only: list[DescriptiveOnlyDecision] = field(default_factory=list)
    claims: list[SignificanceClaim] = field(default_factory=list)
    drift_disagreements: list[str] = field(default_factory=list)
    family_size: int = 0
    n_significant_pre: int = 0
    n_significant_post: int = 0
    n_exploratory: int = 0
    #: The B3 permutation result. None only when the caller declined B3 explicitly.
    b3: B3Result | None = None

    @property
    def n_claimable(self) -> int:
        return sum(1 for claim in self.claims if claim.claimable)

    @property
    def n_refused(self) -> int:
        return sum(1 for claim in self.claims if not claim.claimable)

    @property
    def descriptive_only_harm_types(self) -> tuple[str, ...]:
        return tuple(sorted({d.harm_type for d in self.descriptive_only if d.descriptive_only}))


def run_stats_analysis(
    inputs: StatsInputs,
    designs: Sequence[PairDesign],
    lattice_config: LatticeConfig,
    *,
    config: StatsConfig,
    min_cell_n: int,
    scores: Sequence[ItemScore] | None = None,
    cells: Sequence[IdentityCell] | None = None,
) -> StatsResult:
    """Run the whole stage and return its result. Pure of I/O beyond what ``inputs`` holds.

    ``scores`` (with ``cells``) are the item-level rows P7 fitted; given them, B3 runs on every
    family term and its p lands on the claim rows. Omitting them skips B3 -- the script refuses to
    do that unless told to explicitly.
    """
    if scores is not None and cells is None:
        raise ValueError("B3 needs the lattice cells to place the item-level scores")
    check_effects_match_fits(inputs)

    # `attempted_groups`, not `groups`: the family spans what P7 TRIED, so a source whose every
    # pair came back not-estimable still takes its place in the family and is excused rather than
    # vanishing from it and shrinking the Benjamini-Hochberg denominator (P2).
    plan: FamilyPlan = build_family_plan(
        designs,
        lattice_config=lattice_config,
        groups=inputs.attempted_groups or inputs.groups,
        settings=config.fdr,
        not_estimable=not_estimable_reasons(inputs),
    )
    engine = BenjaminiHochbergEngine(config, plan=plan)

    with_p = attach_wald_p(
        inputs.effects,
        df_by_series=cluster_df_by_series(inputs),
        # Membership, not the pair: a confirmatory pair's main effects and dependent
        # interaction cells are OUTSIDE the family and must carry no p (see
        # FamilyPlan.is_family_member, fixed 21 Sep 2026).
        only=lambda effect: plan.is_family_member(effect_key(effect)),
    )
    # Pre-correction count, on the SAME alpha, so the "FDR reduces the significant count" gate
    # compares like with like.
    n_pre = sum(
        1
        for effect in with_p
        if plan.is_family_member(effect_key(effect))
        and effect.p is not None
        and effect.p <= config.fdr.alpha
    )
    corrected = engine.correct(with_p)
    b3 = (
        run_b3(
            scores,
            cells or (),
            designs,
            corrected,
            is_family_member=lambda key: plan.is_family_member(key),
            settings=config.permutation,
        )
        if scores is not None
        else None
    )
    n_post = sum(1 for effect in corrected if effect.q is not None and effect.q <= config.fdr.alpha)

    decisions = descriptive_only_decisions(
        inputs.comparisons, inputs.harm_flags, threshold=config.drift.descriptive_only_ratio
    )
    gate = ClaimGate(
        exploratory_pairs=plan.exploratory_pairs,
        descriptive_only=descriptive_only_keys(decisions),
        alpha=config.fdr.alpha,
        powered_only=True,
    )
    claims = gate.claims(
        corrected,
        fdr_by=fdr_by_values(corrected),
        permutation_p=None if b3 is None else b3.p_values,
    )

    return StatsResult(
        corrected=corrected,
        family=list(plan.members),
        descriptive_only=decisions,
        claims=claims,
        drift_disagreements=compare_with_metrics_flags(decisions),
        family_size=plan.size,
        n_significant_pre=n_pre,
        n_significant_post=n_post,
        b3=b3,
        n_exploratory=sum(
            # Everything outside the family: exploratory pairs AND the confirmatory pairs'
            # non-member terms. Both are reported descriptively and carry no significance.
            1
            for effect in corrected
            if not plan.is_family_member(effect_key(effect))
        ),
    )
