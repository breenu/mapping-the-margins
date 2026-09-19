"""Manifest wiring for the statistics stage (P9).

The P5/P6/P7 pattern once more, and the argument is the same one phase later. An
:class:`~margins.schemas.EffectEstimate` carries a ``q``, but not the family it was corrected
over, not the alpha, not the degrees of freedom behind its p-value, and not the threshold
``powered`` was judged against. A reviewer's first question about an FDR claim is "how many tests
was this controlled over", and the honest place to answer it is the committed manifest rather
than a gitignored Parquet.

The measured intraclass correlation is recorded here too. It is the input the whole design was
derived from, it was measured rather than assumed, and a run that did not record it would leave
the minimum detectable effect looking like a free choice.
"""

from __future__ import annotations

from pathlib import Path

from margins.config import StatsRecord
from margins.stats.analysis import StatsResult
from margins.stats.config import (
    CELL_STATS_INPUT_KEY,
    DRIFT_COMPARISON_INPUT_KEY,
    EFFECTS_INPUT_KEY,
    FIT_REPORTS_INPUT_KEY,
    FIT_TERMS_INPUT_KEY,
    HARM_FLAGS_INPUT_KEY,
    STATS_INPUT_KEY,
    StatsConfig,
)

__all__ = ["stats_inputs", "stats_record"]


def stats_record(
    config: StatsConfig,
    result: StatsResult,
    *,
    stats_name: str,
    benchmark: str,
    min_cell_n: int,
) -> StatsRecord:
    """Assemble the manifest's stats block."""
    return StatsRecord(
        stats_name=stats_name,
        stats_version=config.stats_version,
        stats_hash=config.stats_hash,
        benchmark=benchmark,
        alpha=config.fdr.alpha,
        method=config.fdr.method,
        prds_justification=config.fdr.prds_justification,
        family_terms=config.fdr.family_terms,
        family_size=result.family_size,
        n_present=sum(1 for member in result.family if member.status == "present"),
        n_excused_not_estimable=sum(
            1 for member in result.family if member.status == "excused_not_estimable"
        ),
        n_significant_pre=result.n_significant_pre,
        n_significant_post=result.n_significant_post,
        n_exploratory_passthrough=result.n_exploratory,
        permutation_scheme=config.permutation.scheme,
        n_permutations=config.permutation.n_permutations,
        permutation_seed=config.permutation.seed,
        descriptive_only_ratio=config.drift.descriptive_only_ratio,
        descriptive_only_harm_types=result.descriptive_only_harm_types,
        n_drift_disagreements=len(result.drift_disagreements),
        icc_rho=config.power.icc_rho,
        effect_size=config.power.effect_size,
        target_power=config.power.target_power,
        min_cell_n=min_cell_n,
        n_claimable=result.n_claimable,
        n_refused=result.n_refused,
    )


def stats_inputs(*, stats_config_path: Path, metrics_paths: dict[str, Path]) -> dict[str, str]:
    """``{source_name: sha256}`` for the config and every P7 table actually consumed.

    Parquet is not byte-deterministic, so these digests identify the files THIS run read rather
    than any file a regeneration would produce, which is the honest thing to record.
    """
    from margins.prompts.config import file_sha256

    keys = {
        "effects": EFFECTS_INPUT_KEY,
        "fit_terms": FIT_TERMS_INPUT_KEY,
        "fit_reports": FIT_REPORTS_INPUT_KEY,
        "cell_stats": CELL_STATS_INPUT_KEY,
        "drift_comparison": DRIFT_COMPARISON_INPUT_KEY,
        "harm_flags": HARM_FLAGS_INPUT_KEY,
    }
    out = {STATS_INPUT_KEY: file_sha256(stats_config_path)}
    for name, key in keys.items():
        path = metrics_paths.get(name)
        if path is not None and path.is_file():
            out[key] = file_sha256(path)
    return out
