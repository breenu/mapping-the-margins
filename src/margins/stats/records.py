"""Stage-local records for the statistics layer (P9).

None of these is a shared schema, deliberately (invariant 2). The contract layer already has the
record the paper's inferential claims are written against --
:class:`~margins.schemas.EffectEstimate`, which P9 stamps with ``q``, ``family_size`` and
``correction`` and nothing else. What P9 additionally has to emit is **bookkeeping about the
decisions**: which family was corrected over, what the permutation null said, what the power
simulation found, which harm types the drift rule downgraded, and which effects may therefore be
cited at all. Those are per-stage shapes, and P10 reads them from the emitted tables rather than
through the contract.

**Why a refusal is a record and not an absence.** A descriptive-only harm type, an exploratory
pair, and an underpowered cell all produce estimates that exist and must not be cited. If P9
expressed that by *filtering them out*, the artifact would be indistinguishable from one where the
fit never happened, and a reader could not tell "we refused to claim this" from "we did not look".
:class:`SignificanceClaim` therefore carries a row for every effect, with ``claimable`` and a named
``refusal_reason``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from margins.stats.config import CorrectionMethod, EffectPattern

__all__ = [
    "ClaimStatus",
    "DescriptiveOnlyDecision",
    "FamilyMember",
    "FamilyMemberStatus",
    "PowerPoint",
    "PowerResult",
    "SignificanceClaim",
]

#: Why an effect may or may not be cited as a significance claim. Ordered by precedence: the first
#: applicable reason is the one reported.
ClaimStatus = Literal[
    "claimable",
    "exploratory",
    "descriptive_only",
    "underpowered",
    "uncorrected",
    "not_in_family",
]

#: Whether a declared family slot was filled, legitimately excused, or genuinely missing.
FamilyMemberStatus = Literal["present", "excused_not_estimable", "missing"]


class FamilyMember(BaseModel):
    """One declared slot of the FDR family, and whether an estimate arrived to fill it.

    The family is declared from the lattice BEFORE the estimates are looked at, so a member that
    never arrives is a detectable fact rather than a silently smaller family. ``excused_not
    _estimable`` carries P7's own reason string: a pair with an empty cell produced no estimate for
    a stated reason, which is different from one that went missing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    benchmark: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    harm_type: str = Field(min_length=1)
    axis_a: str = Field(min_length=1)
    axis_b: str = Field(min_length=1)
    term: str = Field(min_length=1)
    status: FamilyMemberStatus
    reason: str | None = Field(default=None, description="Why not, when status is not 'present'.")

    @model_validator(mode="after")
    def _consistent(self) -> FamilyMember:
        if self.status != "present" and self.reason is None:
            raise ValueError("a family member that is not present must say why")
        return self


class PowerPoint(BaseModel):
    """One simulated design point: what power the REAL estimator plus the REAL FDR achieved.

    ``items_per_cell`` is the number of ITEM CLUSTERS, which is the template-frame count and does
    not move with ``seeds_per_item``. Both are carried so a reader cannot mistake observations for
    clusters -- the whole point of the simulation is that they are not the same thing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    axis_a: str = Field(min_length=1)
    axis_b: str = Field(min_length=1)
    pattern: EffectPattern
    effect_size: float = Field(gt=0.0, description="The injected magnitude, in SD units.")
    icc_rho: float = Field(ge=0.0, lt=1.0)

    frames_per_cell: int = Field(ge=2, description="Item clusters per cell. Seeds do not add any.")
    seeds_per_item: int = Field(ge=1)
    renderings_per_cell: int = Field(ge=1, description="frames_per_cell * seeds_per_item.")
    items_per_cell: int = Field(ge=2)
    n_item_clusters: int = Field(ge=2, description="Across the whole pair.")
    n_obs: int = Field(ge=1)

    alpha: float = Field(gt=0.0, lt=1.0)
    method: CorrectionMethod
    family_size: int = Field(ge=1)
    n_simulations: int = Field(ge=1)
    n_detected: int = Field(ge=0)
    power: float = Field(ge=0.0, le=1.0)
    power_ci_low: float = Field(ge=0.0, le=1.0)
    power_ci_high: float = Field(ge=0.0, le=1.0)

    target_term: str = Field(min_length=1, description="The planted term detection is measured on.")
    target_gamma: float = Field(description="Its true value in the injected table.")
    mean_se: float = Field(ge=0.0, description="Mean cluster-robust SE of the target term.")
    mean_estimate: float
    limiting: bool = Field(
        default=False,
        description=(
            "True for the seeds -> infinity point: one observation per item cluster, all "
            "within-item noise removed. This is the CEILING the design cannot beat."
        ),
    )

    @model_validator(mode="after")
    def _consistent(self) -> PowerPoint:
        if self.power_ci_low > self.power_ci_high:
            raise ValueError("power_ci_low must be <= power_ci_high")
        if self.n_detected > self.n_simulations:
            raise ValueError("n_detected cannot exceed n_simulations")
        if not self.limiting and self.renderings_per_cell != (
            self.frames_per_cell * self.seeds_per_item
        ):
            raise ValueError("renderings_per_cell must equal frames_per_cell * seeds_per_item")
        return self


class PowerResult(BaseModel):
    """The minimum renderings per cell for one (pattern, rho, frames) curve, or that there is none.

    **A curve that never reaches the target returns ``None``, never a number.** The honest report
    for a design whose cluster count caps its power is "not reachable at this frame count", with
    the ceiling named; inventing an extrapolated minimum would be a fabricated compute budget.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    axis_a: str = Field(min_length=1)
    axis_b: str = Field(min_length=1)
    pattern: EffectPattern
    effect_size: float = Field(gt=0.0)
    icc_rho: float = Field(ge=0.0, lt=1.0)
    frames_per_cell: int = Field(ge=2)
    target_power: float = Field(gt=0.0, lt=1.0)

    min_seeds_per_item: int | None = Field(default=None, ge=1)
    min_renderings_per_cell: int | None = Field(default=None, ge=1)
    #: The pre-registered threshold this curve implies. Identical to
    #: ``min_renderings_per_cell`` by construction -- named separately because that identity IS
    #: the "a budget and the minimum N it must clear are one decision" rule (section 8).
    min_cell_n: int | None = Field(default=None, ge=1)

    power_ceiling: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Power as seeds -> infinity at this frame count: one observation per cluster, no "
            "within-item noise. No seed budget can exceed it."
        ),
    )
    reason: str | None = Field(
        default=None, description="Why no minimum was reached, when there is none."
    )
    curve: tuple[PowerPoint, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> PowerResult:
        fields = (self.min_seeds_per_item, self.min_renderings_per_cell, self.min_cell_n)
        if any(f is None for f in fields) and any(f is not None for f in fields):
            raise ValueError("the three minimum fields are set together or not at all")
        if self.min_seeds_per_item is None and self.reason is None:
            raise ValueError("a curve that reached no minimum must say why")
        if self.min_seeds_per_item is not None:
            if self.min_renderings_per_cell != self.frames_per_cell * self.min_seeds_per_item:
                raise ValueError(
                    "min_renderings_per_cell must equal frames_per_cell * min_seeds_per_item"
                )
            if self.min_cell_n != self.min_renderings_per_cell:
                raise ValueError(
                    "min_cell_n and min_renderings_per_cell are ONE decision and must agree "
                    "(PROJECT_CONTEXT.md section 8)"
                )
        return self


class DescriptiveOnlyDecision(BaseModel):
    """The enforced form of section 9's rule, per (benchmark, model, harm type).

    P7 emitted a *candidate* flag under a placeholder. This is the decision, taken against the
    pre-registered ``descriptive_only_ratio`` and recomputed from the raw ``drift_rms`` /
    ``model_rms`` columns rather than from P7's ``comparable`` boolean -- which was computed under
    the old number and must not be inherited.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    benchmark: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    harm_type: str = Field(min_length=1)
    descriptive_only: bool
    threshold: float = Field(gt=0.0, description="The pre-registered ratio this was decided at.")
    max_ratio: float | None = Field(
        default=None,
        ge=0.0,
        description="Largest drift_rms / model_rms seen. None when no ratio was computable.",
    )
    worst_scorer: str | None = None
    worst_axis_a: str | None = None
    worst_axis_b: str | None = None
    worst_control_family: str | None = None
    n_comparisons: int = Field(ge=0)
    n_over_threshold: int = Field(ge=0)
    scorers_over_threshold: tuple[str, ...] = ()
    #: True when P7's own candidate flag disagreed with this decision. Surfaced, never swallowed.
    disagrees_with_metrics_flag: bool = False


class SignificanceClaim(BaseModel):
    """One effect and whether it may be cited -- including when it may not, and why.

    ``claimable`` is the gate the report layer must pass through. A refused effect keeps its ``q``
    (it was corrected; the number exists) but cannot be rendered as significant, and the reason is
    on the row so a reader of the artifact sees the refusal rather than an absence.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    benchmark: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    harm_type: str = Field(min_length=1)
    axis_a: str = Field(min_length=1)
    axis_b: str = Field(min_length=1)
    term: str = Field(min_length=1)

    estimate: float
    se: float = Field(ge=0.0)
    ci_low: float
    ci_high: float
    p: float | None = Field(default=None, ge=0.0, le=1.0)
    q: float | None = Field(default=None, ge=0.0, le=1.0)
    q_fdr_by: float | None = Field(
        default=None, ge=0.0, le=1.0, description="The BY sensitivity column, never the primary."
    )
    p_permutation: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="B3's assumption-light second opinion. Never enters the FDR family.",
    )

    powered: bool
    status: ClaimStatus
    claimable: bool
    refusal_reason: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> SignificanceClaim:
        if self.ci_low > self.ci_high:
            raise ValueError("ci_low must be <= ci_high")
        if self.claimable != (self.status == "claimable"):
            raise ValueError("claimable must agree with status")
        if not self.claimable and self.refusal_reason is None:
            raise ValueError("a refused claim must name its reason")
        if self.claimable and self.refusal_reason is not None:
            raise ValueError("a claimable effect cannot carry a refusal reason")
        return self
