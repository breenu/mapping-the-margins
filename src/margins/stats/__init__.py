"""The statistics layer (P9): FDR, power, permutation, and the enforced drift downgrade.

**This is the only stage permitted to declare significance** (invariant 3). ``q``,
``family_size`` and ``correction`` are set in :mod:`margins.stats.correct` and nowhere else in the
codebase, and a test asserts that by scanning the source rather than by convention.

What lives here, and the one-line reason each is here rather than upstream:

* :mod:`margins.stats.family` -- the FDR family is the lattice's **confirmatory** pairs and
  nothing else. An unbounded family over the whole lattice would push the Benjamini-Hochberg
  threshold low enough that nothing survives, regardless of per-cell N (section 8).
* :mod:`margins.stats.wald` -- the p-value. P7 left it unset because a p-value on a clustered fit
  is a claim about degrees of freedom, and ``df`` is the ITEM cluster count minus one, not
  ``n_obs``.
* :mod:`margins.stats.power` -- a SIMULATION on the real estimator and the real FDR, not a
  formula. A formula would have to assume away item clustering and multiplicity, which are the two
  things that actually bind.
* :mod:`margins.stats.correct` -- Benjamini-Hochberg, with Benjamini-Yekutieli reported alongside.
* :mod:`margins.stats.permute` -- B3, the genuinely independent null. The additive null (B2) is a
  *presentation* of the interaction, not corroboration of it (section 10).
* :mod:`margins.stats.descriptive` / :mod:`margins.stats.claims` -- section 9's rule, enforced:
  a harm type whose scorer's own interaction is comparable to the model's cannot distinguish
  emergence in the model from emergence in the instrument, so it becomes descriptive-only and the
  claim layer **refuses** to render it as significant.

No torch, and no new dependency: ``statsmodels``, ``scipy`` and ``numpy`` were already core.
"""

from __future__ import annotations

from margins.stats.analysis import StatsResult, run_stats_analysis
from margins.stats.claims import ClaimGate, ClaimRefusedError
from margins.stats.config import (
    DEFAULT_STATS,
    STATS_INPUT_KEY,
    CorrectionMethod,
    DriftEnforcement,
    EffectPattern,
    FamilyTerms,
    FdrSettings,
    PermutationScheme,
    PermutationSettings,
    PowerSettings,
    StatsConfig,
    default_stats_config_dir,
    load_stats_config,
    stats_config_from_mapping,
)
from margins.stats.correct import (
    AlreadyCorrectedError,
    BenjaminiHochbergEngine,
    MissingPValueError,
    fdr_by_values,
)
from margins.stats.descriptive import descriptive_only_decisions, descriptive_only_keys
from margins.stats.export import DEFAULT_STATS_DIR, export_stats, read_corrected_effects
from margins.stats.family import (
    FamilyPlan,
    IncompleteFamilyError,
    UnknownFamilyMemberError,
    build_family_plan,
    family_size,
    family_terms_of,
    free_interaction_terms,
)
from margins.stats.icc import IccEstimate, IccObservation, estimate_icc, icc_interval
from margins.stats.inputs import EffectKey, StatsInputs, load_stats_inputs
from margins.stats.manifest import stats_inputs, stats_record
from margins.stats.numbers import (
    PAPER_NUMBERS_PATH,
    PaperNumber,
    PaperNumbers,
    read_paper_numbers,
    write_paper_numbers,
)
from margins.stats.permute import (
    PermutationPlan,
    permutation_p,
    permutation_plan,
    permute_interaction,
)
from margins.stats.power import (
    PowerDesign,
    ceiling_point,
    cell_mean_table,
    interaction_table,
    minimum_renderings,
    power_curve,
    power_settings_grid,
    simulate_frame,
    simulate_power_point,
    wilson_interval,
)
from margins.stats.records import (
    ClaimStatus,
    DescriptiveOnlyDecision,
    FamilyMember,
    FamilyMemberStatus,
    PowerPoint,
    PowerResult,
    SignificanceClaim,
)
from margins.stats.wald import attach_wald_p, wald_p, wald_t

__all__ = [
    "DEFAULT_STATS",
    "DEFAULT_STATS_DIR",
    "PAPER_NUMBERS_PATH",
    "STATS_INPUT_KEY",
    "AlreadyCorrectedError",
    "BenjaminiHochbergEngine",
    "ClaimGate",
    "ClaimRefusedError",
    "ClaimStatus",
    "CorrectionMethod",
    "DescriptiveOnlyDecision",
    "DriftEnforcement",
    "EffectKey",
    "EffectPattern",
    "FamilyMember",
    "FamilyMemberStatus",
    "FamilyPlan",
    "FamilyTerms",
    "FdrSettings",
    "IccEstimate",
    "IccObservation",
    "IncompleteFamilyError",
    "MissingPValueError",
    "PaperNumber",
    "PaperNumbers",
    "PermutationPlan",
    "PermutationScheme",
    "PermutationSettings",
    "PowerDesign",
    "PowerPoint",
    "PowerResult",
    "PowerSettings",
    "SignificanceClaim",
    "StatsConfig",
    "StatsInputs",
    "StatsResult",
    "UnknownFamilyMemberError",
    "attach_wald_p",
    "build_family_plan",
    "ceiling_point",
    "cell_mean_table",
    "default_stats_config_dir",
    "descriptive_only_decisions",
    "descriptive_only_keys",
    "estimate_icc",
    "export_stats",
    "family_size",
    "family_terms_of",
    "fdr_by_values",
    "free_interaction_terms",
    "icc_interval",
    "interaction_table",
    "load_stats_config",
    "load_stats_inputs",
    "minimum_renderings",
    "permutation_p",
    "permutation_plan",
    "permute_interaction",
    "power_curve",
    "power_settings_grid",
    "read_corrected_effects",
    "read_paper_numbers",
    "run_stats_analysis",
    "simulate_frame",
    "simulate_power_point",
    "stats_config_from_mapping",
    "stats_inputs",
    "stats_record",
    "wald_p",
    "wald_t",
    "wilson_interval",
    "write_paper_numbers",
]
