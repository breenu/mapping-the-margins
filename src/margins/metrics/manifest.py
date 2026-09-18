"""Manifest wiring for the metric stage (P7).

The P5/P6 pattern one phase later. An :class:`~margins.schemas.EffectEstimate` records an
estimate, an SE and a CI; it does not record HOW they were produced. The estimand is
pre-registration material -- identity link, sum-to-zero coding over order-2 cells, cluster-robust
SEs by item, a percentile cluster bootstrap over items stratified by cell, the triangulated
harm score -- and a reproducer reading ``effects.parquet`` cannot tell a cluster-bootstrap CI from
an iid one. So the manifest says, in words, which estimator ran, and it records the two placeholder
drift thresholds the flags were computed under, because the flags mean nothing without them.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from margins.config import MetricsRecord
from margins.metrics.config import (
    DRIFT_INPUT_KEY,
    METRICS_INPUT_KEY,
    SCORES_INPUT_KEY,
    MetricsConfig,
)
from margins.metrics.records import HarmTypeFlag

__all__ = ["metrics_inputs", "metrics_record"]


def metrics_record(
    config: MetricsConfig,
    *,
    metrics_name: str,
    benchmark: str,
    min_cell_n: int,
    n_pairs: int,
    n_effects: int,
    n_not_estimable: int,
    n_drift_fits: int,
    harm_flags: Sequence[HarmTypeFlag],
    n_family_disagreements: int,
) -> MetricsRecord:
    """Assemble the manifest's metrics block."""
    return MetricsRecord(
        metrics_name=metrics_name,
        metrics_version=config.metrics_version,
        metrics_hash=config.metrics_hash,
        benchmark=benchmark,
        n_resamples=config.bootstrap.n_resamples,
        ci_level=config.bootstrap.ci_level,
        bootstrap_seed=config.bootstrap.seed,
        min_cell_n=min_cell_n,
        comparable_ratio=config.drift.comparable_ratio,
        family_disagreement_ratio=config.drift.family_disagreement_ratio,
        n_pairs=n_pairs,
        n_effects=n_effects,
        n_not_estimable=n_not_estimable,
        n_drift_fits=n_drift_fits,
        descriptive_only_candidates=tuple(
            sorted({f.harm_type for f in harm_flags if f.descriptive_only_candidate})
        ),
        n_family_disagreements=n_family_disagreements,
    )


def metrics_inputs(
    *, metrics_config_path: Path, scores_path: Path, drift_path: Path
) -> dict[str, str]:
    """``{source_name: sha256}`` for the config and the two P6 tables actually consumed.

    Parquet is not byte-deterministic, so these digests identify the files THIS run read rather
    than any file a regeneration would produce -- which is the honest thing to record.
    """
    from margins.prompts.config import file_sha256

    return {
        METRICS_INPUT_KEY: file_sha256(metrics_config_path),
        SCORES_INPUT_KEY: file_sha256(scores_path),
        DRIFT_INPUT_KEY: file_sha256(drift_path),
    }
