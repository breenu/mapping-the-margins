"""The stats config: the FDR family, the permutation scheme, power, and the drift rule (P9).

Why a config and not constants: every number here is **pre-registration material**. ``alpha`` and
the family definition decide what may be called significant; ``descriptive_only_ratio`` decides
which harm types may be called anything at all; the power settings decide ``min_cell_n`` and
therefore the rendering budget. All of it belongs in a versioned YAML whose sha256 the run manifest
records, so a reproducer can see exactly which thresholds produced a claim. A Python literal would
be invisible to the manifest.

**What is deliberately NOT configurable here**, because it is the estimand rather than a setting:

* the FDR family is the lattice's ``confirmatory`` pairs and nothing else -- a config able to widen
  it to exploratory pairs would be a config able to manufacture the significance claim the
  pre-registration forbids;
* the p-value is a two-sided cluster-robust Wald t on ``estimate/se`` with ``df`` = item clusters
  minus one;
* the permutation exchange unit is the ITEM, with all of its sampled seeds moving together.

**``prds_justification`` is required under ``fdr_bh``.** Section 8 says BH assumes independence or
PRDS and that the project must "either state the PRDS assumption explicitly or use ``fdr_by``". A
config choosing BH without writing the sentence therefore **fails to load**, rather than leaving an
unstated assumption to be discovered by a reviewer.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from margins.repro import MAX_SEED

__all__ = [
    "CELL_STATS_INPUT_KEY",
    "CORRECTION_METHODS",
    "DEFAULT_STATS",
    "DRIFT_COMPARISON_INPUT_KEY",
    "EFFECTS_INPUT_KEY",
    "FIT_REPORTS_INPUT_KEY",
    "FIT_TERMS_INPUT_KEY",
    "HARM_FLAGS_INPUT_KEY",
    "SCORES_INPUT_KEY",
    "STATS_INPUT_KEY",
    "CorrectionMethod",
    "DriftEnforcement",
    "EffectPattern",
    "FamilyTerms",
    "FdrSettings",
    "PermutationScheme",
    "PermutationSettings",
    "PowerSettings",
    "StatsConfig",
    "default_stats_config_dir",
    "load_stats_config",
    "stats_config_from_mapping",
]

#: The config this project's confirmatory runs use.
DEFAULT_STATS: Final[str] = "confirmatory_v1"

#: Manifest ``inputs`` keys. Every P7/P6 table P9 actually consumed is digested, for the same
#: reason P7 digested its two: Parquet is not byte-deterministic, so the digest identifies the file
#: THIS run read rather than any file a regeneration would produce.
STATS_INPUT_KEY: Final[str] = "stats_config"
EFFECTS_INPUT_KEY: Final[str] = "effects_parquet"
FIT_TERMS_INPUT_KEY: Final[str] = "fit_terms_parquet"
FIT_REPORTS_INPUT_KEY: Final[str] = "fit_reports_parquet"
CELL_STATS_INPUT_KEY: Final[str] = "cell_stats_parquet"
DRIFT_COMPARISON_INPUT_KEY: Final[str] = "drift_comparison_parquet"
HARM_FLAGS_INPUT_KEY: Final[str] = "harm_flags_parquet"
SCORES_INPUT_KEY: Final[str] = "scores_parquet"

#: ``fdr_bh`` is primary; ``fdr_by`` is the unconditional guarantee, reported alongside.
CorrectionMethod = Literal["fdr_bh", "fdr_by"]
CORRECTION_METHODS: Final[tuple[CorrectionMethod, ...]] = ("fdr_bh", "fdr_by")

#: Which terms of a confirmatory pair enter the family. See the shipped config for the decision
#: and for why the two wider options were rejected.
FamilyTerms = Literal["free_interaction", "all_interaction", "all_terms"]

#: ``within_stratum`` is section 10's wording made structural. ``freedman_lane`` preserves both
#: main effects exactly and serves as the calibration oracle.
PermutationScheme = Literal["within_stratum", "freedman_lane"]

#: How the injected interaction is shaped before it is scaled to the target magnitude.
EffectPattern = Literal["concentrated", "diffuse"]


class FdrSettings(BaseModel):
    """Benjamini-Hochberg over the confirmatory family (PROJECT_CONTEXT.md section 8)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    alpha: float = Field(gt=0.0, lt=1.0, description="The FDR level the family is controlled at.")
    method: CorrectionMethod = "fdr_bh"
    report_fdr_by: bool = Field(
        default=True,
        description="Also compute fdr_by as a sensitivity column. Never replaces `method`.",
    )
    family_terms: FamilyTerms = "free_interaction"
    family_benchmarks: tuple[str, ...] = Field(
        default=(),
        description=(
            "The sources the family spans, PRE-REGISTERED. Added at P2, and it closes a real hole: "
            "the benchmark dimension used to be inferred from whichever effects happened to land "
            "in `effects.parquet`, so `family_size` was a function of which files were on disk. "
            "The 'correcting a subset RAISES' guard could not catch a wholly absent benchmark, "
            "because the expected set was built FROM the offered effects -- running P9 on an "
            "open_gen-only directory would quietly produce a smaller family and more discoveries. "
            "Declared here, a benchmark that arrives with neither effects nor a P7 not-estimable "
            "excuse raises instead. Empty means 'infer', which is the pre-P2 behaviour and is kept "
            "only so a fixture config need not name sources it does not have."
        ),
    )
    prds_justification: str = Field(
        default="",
        description=(
            "Why BH's positive-dependency assumption is defensible for this family. REQUIRED "
            "(non-empty) under fdr_bh -- section 8 demands the assumption be stated or fdr_by used."
        ),
    )

    @model_validator(mode="after")
    def _prds_stated(self) -> FdrSettings:
        if self.method == "fdr_bh" and not self.prds_justification.strip():
            raise ValueError(
                "method='fdr_bh' requires a non-empty prds_justification. BH assumes independence "
                "or PRDS, and the tests here share items, templates and a model "
                "(PROJECT_CONTEXT.md section 8): state the assumption or set method='fdr_by'."
            )
        return self


class PermutationSettings(BaseModel):
    """The B3 permutation baseline: a structure-respecting shuffle, never a global one."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scheme: PermutationScheme = "within_stratum"
    stratum_axis: Literal["a", "b"] = Field(
        default="a",
        description=(
            "Which axis of the pair supplies the strata. NOT symmetric: the other axis's labels "
            "are what gets permuted, and its main effect is what the shuffle destroys."
        ),
    )
    n_permutations: int = Field(
        ge=199,
        description=(
            "Replicates. At least 199, because the smallest attainable p is 1/(B+1) and a "
            "coarser null cannot resolve an alpha of 0.05 after correction."
        ),
    )
    seed: int = Field(ge=0, le=MAX_SEED, description="Routed through set_global_seed.")


class PowerSettings(BaseModel):
    """The simulation, not a formula (locked decision, 19 Sep 2026).

    ``frames_per_cell`` is the ITEM CLUSTER count per cell and is fixed by the P4a template set.
    Seeds are repeated observations inside a cluster, so they cannot buy clusters -- which is why
    ``frames_grid`` is swept beside ``seeds_grid`` rather than held constant.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    frames_per_cell: int = Field(
        ge=2, description="Template frames per cell in the shipped prompt set = item clusters."
    )
    frames_grid: tuple[int, ...] = Field(min_length=1)
    seeds_grid: tuple[int, ...] = Field(min_length=1)
    icc_rho: tuple[float, ...] = Field(
        min_length=1,
        description=(
            "The between-item share of standardized score variance. An ASSUMPTION: the dev run "
            "has one response per prompt, so no within-item variance exists to estimate it from. "
            "Swept as a strip and never collapsed to one hidden constant."
        ),
    )
    effect_size: float = Field(gt=0.0, description="Minimum detectable interaction, in SD units.")
    patterns: tuple[EffectPattern, ...] = Field(min_length=1)
    target_power: float = Field(gt=0.0, lt=1.0)
    n_simulations: int = Field(ge=100)
    pair: tuple[str, str] = Field(description="Which confirmatory pair's real design to simulate.")
    seed: int = Field(ge=0, le=MAX_SEED)

    @model_validator(mode="after")
    def _check_grids(self) -> PowerSettings:
        if any(k < 1 for k in self.seeds_grid):
            raise ValueError("every seeds_grid entry must be at least 1")
        if any(f < 2 for f in self.frames_grid):
            raise ValueError("every frames_grid entry must be at least 2")
        if self.frames_per_cell not in self.frames_grid:
            raise ValueError(
                f"frames_per_cell={self.frames_per_cell} must appear in frames_grid "
                f"{self.frames_grid}; the shipped design has to be one of the priced points."
            )
        if any(not 0.0 <= rho < 1.0 for rho in self.icc_rho):
            raise ValueError("every icc_rho must satisfy 0 <= rho < 1")
        if self.pair[0] == self.pair[1]:
            raise ValueError(f"pair must name two distinct axes, got {self.pair}")
        return self


class DriftEnforcement(BaseModel):
    """Section 9's rule, promoted from P7's placeholder to a pre-registered consequence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    descriptive_only_ratio: float = Field(
        gt=0.0,
        description=(
            "drift_rms / model_rms at or above which a harm type becomes DESCRIPTIVE-ONLY and the "
            "claim layer refuses to emit an FDR-significance claim for it. Enforced, not flagged."
        ),
    )


class StatsConfig(BaseModel):
    """The validated stats config. Frozen: a run's thresholds cannot change mid-run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(min_length=1, description="Config name; must match the filename.")
    description: str = Field(min_length=1)
    stats_version: str = Field(
        min_length=1, description="Bumped whenever a threshold or the family definition changes."
    )
    fdr: FdrSettings
    permutation: PermutationSettings
    power: PowerSettings
    drift: DriftEnforcement

    def canonical_json(self) -> str:
        """The exact preimage of :attr:`stats_hash`, exposed so a reproducer can recompute it."""
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    @property
    def stats_hash(self) -> str:
        """sha256 of the resolved config."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------
# The Hydra seam
# --------------------------------------------------------------------------------------


def stats_config_from_mapping(data: Mapping[str, Any]) -> StatsConfig:
    """Validate a plain mapping into a :class:`StatsConfig`."""
    return StatsConfig.model_validate(dict(data))


def default_stats_config_dir() -> Path:
    """``<repo>/configs/stats``. Override it for tests or an installed package."""
    return Path(__file__).resolve().parents[3] / "configs" / "stats"


def load_stats_config(name: str = DEFAULT_STATS, *, config_dir: Path | None = None) -> StatsConfig:
    """Compose ``<config_dir>/<name>.yaml`` with Hydra and validate it.

    The same seam as :func:`margins.metrics.config.load_metrics_config`: Hydra composes YAML into
    a plain container and Pydantic validates it. No Hydra structured configs anywhere.
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    directory = (config_dir or default_stats_config_dir()).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"stats config dir does not exist: {directory}")

    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        composed = compose(config_name=name)
    data = OmegaConf.to_container(composed, resolve=True)
    if not isinstance(data, dict):
        raise TypeError(f"composed stats config {name!r} is not a mapping, got {type(data)!r}")
    payload = cast(dict[str, Any], data)
    payload.setdefault("key", name)
    config = stats_config_from_mapping(payload)
    if config.key != name:
        raise ValueError(
            f"stats config {name!r} declares key={config.key!r}. The filename and the key must "
            "agree, or the manifest records a config name that resolves to a different file."
        )
    return config
