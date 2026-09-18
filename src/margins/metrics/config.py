"""The metrics config: bootstrap settings and the drift-comparison thresholds (P7).

Why a config and not two constants: the numbers here are **pre-registration material**. The
bootstrap's replicate count and seed decide what every headline CI is, and the two drift thresholds
decide which harm types get flagged descriptive-only. Both belong in a versioned YAML whose sha256
the run manifest records, so a reproducer can see exactly which thresholds produced a flag -- a
Python literal would be invisible to the manifest.

**The drift thresholds are PLACEHOLDERS.** ``comparable_ratio`` (a scorer's own interaction
magnitude divided by the model's, above which the scorer is "comparable") and
``family_disagreement_ratio`` (the factor by which the two neutral families may differ before the
drift estimate is reported as style-dependent) are declared conventions until P9 pre-registers them.
P7 produces the numbers and the flags; nothing here enforces a downgrade, and nothing should until
the thresholds are tagged.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, cast

from pydantic import BaseModel, ConfigDict, Field

from margins.repro import MAX_SEED

__all__ = [
    "DEFAULT_METRICS",
    "DRIFT_INPUT_KEY",
    "METRICS_INPUT_KEY",
    "SCORES_INPUT_KEY",
    "BootstrapSettings",
    "DriftThresholds",
    "MetricsConfig",
    "default_metrics_config_dir",
    "load_metrics_config",
    "metrics_config_from_mapping",
]

#: The config this project's confirmatory runs use.
DEFAULT_METRICS: Final[str] = "emergence_v1"

#: Manifest ``inputs`` keys. The two Parquet tables are what the metric layer CONSUMED, so their
#: digests are recorded even though Parquet is not byte-deterministic: a digest of the file that was
#: actually read is a true statement about this run, whatever a regenerated file would hash to.
METRICS_INPUT_KEY: Final[str] = "metrics_config"
SCORES_INPUT_KEY: Final[str] = "scores_parquet"
DRIFT_INPUT_KEY: Final[str] = "drift_parquet"


class BootstrapSettings(BaseModel):
    """The cluster bootstrap over items (PROJECT_CONTEXT.md section 8).

    The resampling unit is not configurable, deliberately: it is the **item (prompt)**, resampled
    with replacement within each cell. That is the pre-registered unit, and a config that could
    switch it to rows would be a config that could quietly understate every variance.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_resamples: int = Field(
        ge=1,
        description=(
            "Bootstrap replicates. At least one, because every EffectEstimate must carry a CI; "
            "a headline number wants thousands."
        ),
    )
    ci_level: float = Field(gt=0.0, lt=1.0, description="Two-sided percentile CI level.")
    seed: int = Field(
        ge=0, le=MAX_SEED, description="Routed through set_global_seed; no hidden global RNG."
    )


class DriftThresholds(BaseModel):
    """PLACEHOLDER thresholds for the drift comparison. TODO(P9): pre-register both."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    comparable_ratio: float = Field(
        default=0.5,
        gt=0.0,
        description=(
            "drift_rms / model_rms at or above which a scorer's own interaction counts as "
            "'comparable in magnitude' to the model's, flagging the harm type as a "
            "descriptive-only candidate (section 9)."
        ),
    )
    family_disagreement_ratio: float = Field(
        default=2.0,
        ge=1.0,
        description=(
            "The two neutral families' interaction magnitudes may differ by at most this factor "
            "before the drift estimate is reported as style-dependent."
        ),
    )


class MetricsConfig(BaseModel):
    """The validated metrics config. Frozen: a run's estimator settings cannot change mid-run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(min_length=1, description="Config name; must match the filename.")
    description: str = Field(min_length=1)
    metrics_version: str = Field(
        min_length=1, description="Bumped whenever the estimator or its thresholds change."
    )
    bootstrap: BootstrapSettings
    drift: DriftThresholds = DriftThresholds()

    def canonical_json(self) -> str:
        """The exact preimage of :attr:`metrics_hash`, exposed so a reproducer can recompute it."""
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    @property
    def metrics_hash(self) -> str:
        """sha256 of the resolved config."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------
# The Hydra seam
# --------------------------------------------------------------------------------------


def metrics_config_from_mapping(data: Mapping[str, Any]) -> MetricsConfig:
    """Validate a plain mapping into a :class:`MetricsConfig`."""
    return MetricsConfig.model_validate(dict(data))


def default_metrics_config_dir() -> Path:
    """``<repo>/configs/metrics``. Override it for tests or an installed package."""
    return Path(__file__).resolve().parents[3] / "configs" / "metrics"


def load_metrics_config(
    name: str = DEFAULT_METRICS, *, config_dir: Path | None = None
) -> MetricsConfig:
    """Compose ``<config_dir>/<name>.yaml`` with Hydra and validate it.

    The same seam as :func:`margins.scoring.config.load_scoring_config`: Hydra composes YAML into
    a plain container and Pydantic validates it. No Hydra structured configs anywhere.
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    directory = (config_dir or default_metrics_config_dir()).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"metrics config dir does not exist: {directory}")

    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        composed = compose(config_name=name)
    data = OmegaConf.to_container(composed, resolve=True)
    if not isinstance(data, dict):
        raise TypeError(f"composed metrics config {name!r} is not a mapping, got {type(data)!r}")
    payload = cast(dict[str, Any], data)
    payload.setdefault("key", name)
    config = metrics_config_from_mapping(payload)
    if config.key != name:
        raise ValueError(
            f"metrics config {name!r} declares key={config.key!r}. The filename and the key must "
            "agree, or the manifest records a config name that resolves to a different file."
        )
    return config
