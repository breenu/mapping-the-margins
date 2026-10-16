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
from typing import Any, Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from margins.repro import MAX_SEED

__all__ = [
    "DEFAULT_P11",
    "CrowsArm",
    "E1Arm",
    "E4Arm",
    "E4DecodingArm",
    "P11Config",
    "R2Arm",
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


class E4DecodingArm(BaseModel):
    """One generated E4 temperature arm."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    kind: Literal["sampled", "greedy"]
    decoding: str = Field(min_length=1)
    seeds: tuple[int, ...]
    prompts_dir: Path

    @field_validator("seeds")
    @classmethod
    def _check_seeds(cls, seeds: tuple[int, ...]) -> tuple[int, ...]:
        return _distinct_seeds(seeds)

    @model_validator(mode="after")
    def _seed_rule(self) -> E4DecodingArm:
        # Pre-reg §10 clarification (23 Sep 2026): the >= 3-seed rule is for SAMPLED arms. A
        # greedy arm has exactly one seed, because more would be identical outputs presented as
        # independent samples.
        if self.kind == "greedy" and len(self.seeds) != 1:
            raise ValueError(
                f"E4 arm {self.name!r} is greedy and must have exactly ONE seed, got {self.seeds}: "
                "greedy decoding is deterministic, so extra seeds are identical outputs"
            )
        if self.kind == "sampled" and len(self.seeds) < 3:
            raise ValueError(f"E4 sampled arm {self.name!r} needs >= 3 distinct seeds")
        return self


class E4Arm(BaseModel):
    """E4, the sensitivity subset. The 0.7 arm is E1's, reused; only these arms generate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pair: tuple[str, str]
    reuses_e1_decoding: str = Field(min_length=1)
    arms: tuple[E4DecodingArm, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _distinct_arms(self) -> E4Arm:
        names = [arm.name for arm in self.arms]
        decodings = [arm.decoding for arm in self.arms]
        if len(set(names)) != len(names) or len(set(decodings)) != len(decodings):
            raise ValueError("E4 arms must have distinct names and distinct decoding profiles")
        if self.reuses_e1_decoding in decodings:
            raise ValueError(
                f"{self.reuses_e1_decoding!r} is E1's profile and is REUSED, never regenerated"
            )
        return self


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
    e4_limit: int = Field(ge=1)

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


class R2Arm(BaseModel):
    """R2, the reproducibility rerun (pre-registration §10, 16 Oct 2026).

    Regenerates the E4 subset under E1's own sampled profile and seeds -- compared against E1's
    responses -- and under the E4 greedy arm's profile and seed -- compared against E4's. Nothing
    here is chosen freely: the validator on :class:`P11Config` pins every value to the arm it is
    compared against, because a rerun under different settings would not be a rerun.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pair: tuple[str, str]
    machines: tuple[str, ...] = Field(min_length=1)
    sampled_decoding: str = Field(min_length=1)
    sampled_seeds: tuple[int, ...]
    sampled_prompts_dir: Path
    greedy_arm: str = Field(min_length=1)

    @field_validator("sampled_seeds")
    @classmethod
    def _check_seeds(cls, seeds: tuple[int, ...]) -> tuple[int, ...]:
        return _distinct_seeds(seeds)

    @field_validator("machines")
    @classmethod
    def _machine_names(cls, machines: tuple[str, ...]) -> tuple[str, ...]:
        # A machine name becomes a directory name (data/responses/r2_<machine>_sampled).
        for machine in machines:
            if not machine.isidentifier() or machine != machine.lower():
                raise ValueError(f"R2 machine names must be lower-case identifiers: {machine!r}")
        if len(set(machines)) != len(machines):
            raise ValueError(f"R2 machine names must be distinct, got {machines}")
        return machines


class P11Config(BaseModel):
    """The whole P11 declaration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(min_length=1)
    p11_version: str = Field(min_length=1)
    lattice: str = Field(min_length=1)
    e1: E1Arm
    e4: E4Arm
    crows: CrowsArm
    redteam: RedTeamArm
    smoke: SmokeSettings
    r2: R2Arm
    pilot_sec_per_generation: dict[str, float]

    @model_validator(mode="after")
    def _r2_reruns_what_it_compares_against(self) -> P11Config:
        r2 = self.r2
        if r2.pair != self.e4.pair:
            raise ValueError(f"R2 reruns the E4 subset {self.e4.pair}, not {r2.pair}")
        if r2.sampled_decoding != self.e1.decoding or r2.sampled_seeds != self.e1.seeds:
            raise ValueError(
                "R2's sampled arm is compared against E1's own responses, so it must use E1's "
                f"decoding and seeds ({self.e1.decoding}, {self.e1.seeds}); got "
                f"({r2.sampled_decoding}, {r2.sampled_seeds})"
            )
        greedy = [arm for arm in self.e4.arms if arm.name == r2.greedy_arm]
        if not greedy or greedy[0].kind != "greedy":
            raise ValueError(f"R2's greedy_arm {r2.greedy_arm!r} must name a greedy E4 arm")
        return self

    @property
    def r2_greedy(self) -> E4DecodingArm:
        """The E4 greedy arm R2 regenerates (its prompts, profile and single seed)."""
        return next(arm for arm in self.e4.arms if arm.name == self.r2.greedy_arm)

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
