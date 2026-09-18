"""Stage-local records for the metric layer (P7).

None of these is a shared schema, deliberately (invariant 2): the contract layer already has the
two records the paper is written against -- :class:`~margins.schemas.CellStat` (descriptive) and
:class:`~margins.schemas.EffectEstimate` (inferential). What P7 additionally has to emit is
*bookkeeping about the fits*: which series was fitted (the triangulated harm score, one scorer's
own view of the model, or one scorer's response to neutral text), whether the fit was estimable,
and how the scorer's own interaction compares with the model's. Those are per-stage shapes, and
P9 reads them from the emitted tables rather than through the contract.

**Why one ``FitTerm`` shape for three kinds of series.** The model-condition triangulated fit, the
model-condition per-scorer fits, and the drift-condition per-scorer-per-family fits are the SAME
model on different rows. Giving them one record keyed by ``(condition, scorer_id, control_family)``
is what makes the side-by-side comparison a join rather than a reconciliation, and it is what
guarantees the drift series was produced by the same estimator as the model's -- there is no second
code path for it to have diverged in.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from margins.schemas import Condition

__all__ = [
    "DriftComparison",
    "FamilyAgreement",
    "FitReport",
    "FitTerm",
    "HarmTypeFlag",
    "SideBySide",
    "TermKind",
]

#: What a fitted term is. ``interaction`` is the estimand; the two main effects are B1's single-axis
#: baseline and are emitted so a reader can see the level a cell sits at is NOT its interaction.
TermKind = Literal["main_a", "main_b", "interaction"]


class FitTerm(BaseModel):
    """One estimated term of one ``A + B + A:B`` fit, with its cluster-robust SE and bootstrap CI.

    ``scorer_id`` is ``None`` for the triangulated (mean-of-scorers) series -- the series the
    :class:`~margins.schemas.EffectEstimate` records are made from -- and a scorer id otherwise.
    ``control_family`` is ``None`` on the model condition and names the neutral family on the
    drift condition.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    benchmark: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    harm_type: str = Field(min_length=1)
    condition: Condition
    scorer_id: str | None = None
    control_family: str | None = None

    axis_a: str = Field(min_length=1)
    axis_b: str = Field(min_length=1)
    term: str = Field(min_length=1)
    kind: TermKind
    value_a: str | None = None
    value_b: str | None = None

    estimate: float
    se: float = Field(ge=0.0)
    ci_low: float
    ci_high: float
    n_obs: int = Field(ge=1)
    n_items: int = Field(ge=1)

    @model_validator(mode="after")
    def _check_interval(self) -> FitTerm:
        if self.ci_low > self.ci_high:
            raise ValueError(f"ci_low ({self.ci_low}) must be <= ci_high ({self.ci_high})")
        return self


class FitReport(BaseModel):
    """What happened when one series was fitted for one axis pair -- including when it could not be.

    A pair with an empty cell, or with no residual degrees of freedom, is **reported** as not
    estimable with the reason named, never silently dropped and never given a NaN. A reader of
    ``effects.parquet`` who finds a pair missing looks here and finds out why.

    ``rms_interaction`` is the root-mean-square of the pair's interaction coefficients: one scalar
    magnitude per pair, in SD units, which is what the drift comparison (section 9: "comparable in
    magnitude") is computed on. ``powered`` is ``None`` on the drift condition, where power is not
    the question being asked.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    benchmark: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    harm_type: str = Field(min_length=1)
    condition: Condition
    scorer_id: str | None = None
    control_family: str | None = None

    axis_a: str = Field(min_length=1)
    axis_b: str = Field(min_length=1)
    estimable: bool
    reason: str | None = Field(default=None, description="Why not, when estimable is False.")

    n_obs: int = Field(ge=0)
    n_items: int = Field(ge=0)
    n_cells: int = Field(ge=0)
    n_params: int = Field(ge=0)
    cov_type: str = Field(min_length=1)
    n_resamples: int = Field(ge=0)

    rms_interaction: float | None = Field(default=None, ge=0.0)
    rms_ci_low: float | None = None
    rms_ci_high: float | None = None
    powered: bool | None = None

    @model_validator(mode="after")
    def _consistent(self) -> FitReport:
        if not self.estimable and self.reason is None:
            raise ValueError("a report that is not estimable must say why")
        if not self.estimable and self.rms_interaction is not None:
            raise ValueError("a report that is not estimable cannot carry an interaction magnitude")
        if (self.rms_ci_low is None) != (self.rms_ci_high is None):
            raise ValueError("rms_ci_low and rms_ci_high are set together or not at all")
        if self.rms_ci_low is not None and self.rms_ci_high is not None:
            if self.rms_ci_low > self.rms_ci_high:
                raise ValueError("rms_ci_low must be <= rms_ci_high")
        return self


class DriftComparison(BaseModel):
    """The scorer's own interaction magnitude BESIDE the model's, for one scorer, pair and family.

    ``model_rms`` is that scorer's *own* view of the model (the per-scorer model-condition fit), not
    the triangulated one: the comparison is apples to apples. ``ratio`` is ``drift_rms /
    model_rms`` and is ``None`` when the model magnitude is exactly zero. **Nothing is
    subtracted.**
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    benchmark: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    harm_type: str = Field(min_length=1)
    scorer_id: str = Field(min_length=1)
    axis_a: str = Field(min_length=1)
    axis_b: str = Field(min_length=1)
    control_family: str = Field(min_length=1)

    model_rms: float = Field(ge=0.0)
    drift_rms: float = Field(ge=0.0)
    ratio: float | None = Field(default=None, ge=0.0)
    comparable: bool = Field(
        description=(
            "True when the scorer's own interaction is comparable in magnitude to the model's "
            "under the config's PLACEHOLDER threshold. A flag, not an enforcement."
        )
    )
    n_model_obs: int = Field(ge=1)
    n_drift_obs: int = Field(ge=1)


class FamilyAgreement(BaseModel):
    """Do the two neutral families agree on a scorer's drift interaction for one pair?

    If they do not, that is a reportable style-dependence finding: the scorer's response to
    identity terms depends on the register of the text around them, and a drift estimate measured
    on one register may not transfer to the other. Surfaced, never averaged away.

    ``families_disagree`` is ``None`` when either family's fit was not estimable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    benchmark: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    harm_type: str = Field(min_length=1)
    scorer_id: str = Field(min_length=1)
    axis_a: str = Field(min_length=1)
    axis_b: str = Field(min_length=1)

    rms_prompt_echo: float | None = Field(default=None, ge=0.0)
    rms_neutral_declarative: float | None = Field(default=None, ge=0.0)
    ratio: float | None = Field(
        default=None, ge=0.0, description="prompt_echo / neutral_declarative, when both are > 0."
    )
    families_disagree: bool | None = None


class HarmTypeFlag(BaseModel):
    """The producible form of section 9's rule, per (benchmark, model, harm type).

    ``descriptive_only_candidate`` is True when ANY scorer of the harm type, on ANY pair and ANY
    family, has a drift interaction comparable to the model's. It is a *candidate* flag: P9 decides
    the downgrade against the pre-registered threshold. P7 only makes the number exist.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    benchmark: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    harm_type: str = Field(min_length=1)
    descriptive_only_candidate: bool
    n_comparisons: int = Field(ge=0)
    n_comparable: int = Field(ge=0)
    scorers_comparable: tuple[str, ...] = ()


class SideBySide(BaseModel):
    """One row per term: the model's coefficient and each family's drift coefficient, side by side.

    The two family columns are named after the two declared control families
    (:data:`margins.scoring.config.CONTROL_FAMILIES`) rather than generically, so the table reads
    without a legend. A ``None`` means that series was not estimable for this pair.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    benchmark: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    harm_type: str = Field(min_length=1)
    scorer_id: str = Field(min_length=1)
    axis_a: str = Field(min_length=1)
    axis_b: str = Field(min_length=1)
    term: str = Field(min_length=1)
    kind: TermKind

    model_estimate: float | None = None
    model_ci_low: float | None = None
    model_ci_high: float | None = None
    prompt_echo_estimate: float | None = None
    prompt_echo_ci_low: float | None = None
    prompt_echo_ci_high: float | None = None
    neutral_declarative_estimate: float | None = None
    neutral_declarative_ci_low: float | None = None
    neutral_declarative_ci_high: float | None = None
