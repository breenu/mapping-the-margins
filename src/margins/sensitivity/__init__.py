"""E4 and R2: re-estimating one pre-registered pair under a stated change (P11 analysis).

Both analyses ask the same question of the class x race grid -- "does the interaction move?" --
under a different change: E4 changes the decoding temperature and the seed (RQ4); R2 regenerates
the same prompts on other hardware and asks whether the reported numbers come back (§10, 16 Oct
2026). So both reduce to one operation, kept in :mod:`margins.sensitivity.fit`: take item scores,
standardize them under a declared rule, and fit them with P7's OWN
:class:`~margins.metrics.metric.EmergenceMetric`. There is deliberately no second estimator here,
which is what lets an E4 or R2 number be compared with a headline number at all.

Pure: no I/O, no clock. The scripts (``run_e4_analysis.py``, ``run_r2_analysis.py``) read and
write; these modules compute.
"""

from margins.sensitivity.e4 import ArmScores, seed_groups, sensitivity_rows
from margins.sensitivity.fit import fit_pair, free_terms, restrict_to_cells
from margins.sensitivity.r2 import (
    R2Comparison,
    R2Verdict,
    compare_to_original,
    exact_match_rate,
    substitute_series,
    summarize,
    verdict_changes,
)

__all__ = [
    "ArmScores",
    "R2Comparison",
    "R2Verdict",
    "compare_to_original",
    "exact_match_rate",
    "fit_pair",
    "free_terms",
    "restrict_to_cells",
    "seed_groups",
    "sensitivity_rows",
    "substitute_series",
    "summarize",
    "verdict_changes",
]
