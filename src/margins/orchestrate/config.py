"""The P11 run declaration, validated (``configs/experiment/p11_v1.yaml``).

**Why a declaration at all.** The pre-registration fixes the design -- 27 prompts per cell x 4
sampled generations, ``sampled_v1``, both confirmatory models -- but not the operational details
that decide whether the run that happens IS that design: the four seeds, the batch size, which
directory each arm writes to. Chosen at the lab, those would be unrecorded decisions made under
time pressure. Declared here and committed before the run, they are checkable: the orchestrator
reads them, the runbook is written against them, and a test fails if the runbook drifts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from margins.repro import MAX_SEED

__all__ = [
    "DEFAULT_P11",
    "CrowsArm",
    "E1Arm",
    "P11Config",
    "RedTeamArm",
    "SmokeSettings",
    "default_p11_config_dir",
    "load_p11_config",
]

DEFAULT_P11: Final[str] = "p11_v1"


def _distinct_seeds(seeds: tuple[int, ...]) -> tuple[int, ...]:
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError(
            f"seeds must be DISTINCT, got {seeds}: distinct samples of one prompt come from "
            "distinct seeds, and the harness rejects a repeated one"
        )
    for seed in seeds:
        if not 0 <= seed <= MAX_SEED:
            raise ValueError(f"seed {seed} is outside [0, {MAX_SEED}]")
    return seeds


class E1Arm(BaseModel):
    """The static open_gen arm."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_set: str = Field(min_length=1)
    decoding: str = Field(min_length=1)
    prompts_dir: Path
    seeds: tuple[int, ...]
    batch_size: int = Field(ge=1, le=64)
    models: tuple[str, ...] = Field(min_length=1)

    @field_validator("seeds")
    @classmethod
    def _check_seeds(cls, seeds: tuple[int, ...]) -> tuple[int, ...]:
        return _distinct_seeds(seeds)


class CrowsArm(BaseModel):
    """The extractive arm. Declared-family source, expected to be excused as not estimable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    extractive_set: str = Field(min_length=1)
    prompts: Path
    decoding: str = Field(min_length=1)
    scoring: str = Field(min_length=1)
    benchmark: str = Field(min_length=1)


class RedTeamArm(BaseModel):
    """The DESCRIPTIVE red-team arm: powered=False, never in the FDR family."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    set: str = Field(min_length=1)
    audit_seeds: tuple[int, ...]
    budget_gpu_hours: float = Field(gt=0.0)
    target_cells: int = Field(ge=1)

    @field_validator("audit_seeds")
    @classmethod
    def _check_seeds(cls, seeds: tuple[int, ...]) -> tuple[int, ...]:
        return _distinct_seeds(seeds)


class SmokeSettings(BaseModel):
    """The multi-rendering smoke run. Must be ESTIMABLE, or it tests nothing it exists to test."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pair: tuple[str, str]
    frames: tuple[str, ...] = Field(min_length=2)
    carrier: str = Field(min_length=1)
    seeds: tuple[int, ...] = Field(min_length=2)
    crows_limit: int = Field(ge=1)
    n_resamples: int = Field(ge=50)

    @field_validator("seeds")
    @classmethod
    def _check_seeds(cls, seeds: tuple[int, ...]) -> tuple[int, ...]:
        return _distinct_seeds(seeds)

    @field_validator("frames")
    @classmethod
    def _distinct_frames(cls, frames: tuple[str, ...]) -> tuple[str, ...]:
        # Two items per cell is the estimator's floor (`_check_estimable`), and each frame is one
        # item per cell under a single carrier. A repeated frame would be one item counted twice.
        if len(set(frames)) != len(frames):
            raise ValueError(f"smoke frames must be distinct, got {frames}")
        return frames


class P11Config(BaseModel):
    """The whole P11 declaration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(min_length=1)
    p11_version: str = Field(min_length=1)
    lattice: str = Field(min_length=1)
    e1: E1Arm
    crows: CrowsArm
    redteam: RedTeamArm
    smoke: SmokeSettings
    pilot_sec_per_generation: dict[str, float]

    @model_validator(mode="after")
    def _pilot_covers_the_models(self) -> P11Config:
        missing = [m for m in self.e1.models if m not in self.pilot_sec_per_generation]
        if missing:
            raise ValueError(
                f"pilot_sec_per_generation has no entry for {missing}; the runbook quotes it as "
                "the pre-measurement estimate for every confirmatory model"
            )
        return self


def default_p11_config_dir() -> Path:
    """``<repo>/configs/experiment``. Override it for tests or an installed package."""
    return Path(__file__).resolve().parents[3] / "configs" / "experiment"


def load_p11_config(name: str = DEFAULT_P11, *, config_dir: Path | None = None) -> P11Config:
    """Compose ``<config_dir>/<name>.yaml`` with Hydra and validate it.

    The same seam as every other loader here: Hydra composes YAML into a plain container and
    Pydantic validates it. No Hydra structured configs anywhere.
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    directory = (config_dir or default_p11_config_dir()).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"P11 config dir does not exist: {directory}")
    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        composed = compose(config_name=name)
    data = OmegaConf.to_container(composed, resolve=True)
    if not isinstance(data, dict):
        raise TypeError(f"composed P11 config {name!r} is not a mapping, got {type(data)!r}")
    config = P11Config.model_validate(cast(dict[str, Any], data))
    if config.key != name:
        raise ValueError(
            f"P11 config {name!r} declares key={config.key!r}. The filename and the key must agree."
        )
    return config
