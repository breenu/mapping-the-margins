"""The contract layer: frozen, source-agnostic records shared by every pipeline stage.

Why this module carries so much weight: the "unified intersectional metric layer" this project
promises is **not** an algorithm bolted on at the end. It is these records plus per-source
standardization (P6/P7). Every module depends only on these types, never on another module's
internals — that is what makes sources, scorers, and backends pluggable. Get the schema right
and cross-paradigm comparability is nearly free; get it wrong and no downstream cleverness
recovers it.

Three design rules run through the whole file:

1. **Frozen, and genuinely so.** Pydantic's ``frozen=True`` blocks attribute reassignment but
   does *not* deep-freeze containers: a plain ``dict`` field stays mutable through its
   reference and makes the record unhashable. So ``axes`` and ``token_logprobs`` are tuples
   and ``payload`` is a :class:`FrozenDict` that freezes all the way down.
2. **No source-specific field on a shared record.** Source quirks live only in
   ``Item.payload``, typed at the loader boundary by :data:`PAYLOAD_TYPES`. Every record uses
   ``extra="forbid"``.
3. **Descriptive and inferential records are different types.** :class:`CellStat` describes a
   cell and carries no ``p``/``q`` at all; :class:`EffectEstimate` tests a hypothesis. An
   interaction belongs to an axis *pair* and a *term*, not to a single cell — order-1 cells
   have no interaction — so collapsing the two would leave a field undefined for half of its
   legal instances and make the FDR family undefinable.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Final, Literal, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    field_validator,
    model_validator,
)
from pydantic_core import core_schema

from margins.config import QuantSetting
from margins.repro import MAX_SEED

__all__ = [
    "PAYLOAD_TYPES",
    "AxisValue",
    "AxisValueRegistry",
    "CellStat",
    "Condition",
    "EffectEstimate",
    "FrozenDict",
    "IdentityCell",
    "Item",
    "ItemScore",
    "Prompt",
    "Provenance",
    "Response",
    "TaskType",
    "TemplatePayload",
]

_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")

#: Distinguishes scores of model output from scores of the matched neutral control texts used
#: by the scorer identity-drift control (PROJECT_CONTEXT.md section 9). Without this split the
#: drift control cannot be estimated, and RQ2's estimand is not credible.
Condition = Literal["model", "drift_control"]


# --------------------------------------------------------------------------------------
# Deep immutability
# --------------------------------------------------------------------------------------


def _deep_freeze(value: Any) -> Any:
    """Recursively convert mutable containers into hashable immutable ones."""
    if isinstance(value, FrozenDict):
        return value
    if isinstance(value, Mapping):
        return FrozenDict(value)
    if isinstance(value, list | tuple):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    """Inverse of :func:`_deep_freeze`, for JSON serialization."""
    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_deep_thaw(item) for item in value]
    if isinstance(value, frozenset | set):
        return sorted(_deep_thaw(item) for item in value)
    return value


class FrozenDict(Mapping[str, Any]):
    """A hashable, *deeply* immutable mapping.

    Exists because ``Item.payload`` is the one field carrying arbitrary source-specific data,
    and invariant 1 requires records to stay immutable and hashable. A plain ``dict`` field on
    a frozen model satisfies neither: it can still be mutated through its reference, and it
    makes the whole record unhashable.

    Nested containers are frozen on construction, so a payload cannot smuggle in a mutable
    list two levels down.
    """

    __slots__ = ("_data", "_hash")

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        source: Mapping[str, Any] = data if data is not None else {}
        self._data: dict[str, Any] = {
            str(key): _deep_freeze(value) for key, value in source.items()
        }
        self._hash: int | None = None

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __hash__(self) -> int:
        cached = self._hash
        if cached is None:
            cached = hash(frozenset(self._data.items()))
            self._hash = cached
        return cached

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FrozenDict):
            return self._data == other._data
        if isinstance(other, Mapping):
            return self._data == dict(other)
        return NotImplemented

    def __repr__(self) -> str:
        return f"FrozenDict({self._data!r})"

    @classmethod
    def _validate(cls, value: Mapping[str, Any]) -> FrozenDict:
        return value if isinstance(value, cls) else cls(value)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        mapping_schema = core_schema.dict_schema(core_schema.str_schema(), core_schema.any_schema())
        return core_schema.no_info_after_validator_function(
            cls._validate,
            core_schema.union_schema([core_schema.is_instance_schema(cls), mapping_schema]),
            serialization=core_schema.plain_serializer_function_ser_schema(
                _deep_thaw, return_schema=mapping_schema, when_used="always"
            ),
        )


# --------------------------------------------------------------------------------------
# Enums and payload typing
# --------------------------------------------------------------------------------------


class TaskType(str, Enum):
    """What a record asks the model to do.

    ``open_gen`` is the **primary** task type: it carries the main audit and red-teaming, it is
    where RQ2 lives, and it is the only path that can reach arbitrary lattice cells.
    """

    #: Sentence-pair likelihood. For decoder-only models this is autoregressive sentence
    #: log-likelihood rather than true PLL — a documented construct choice, not an oversight.
    PAIR_PLL = "pair_pll"
    #: WinoBias, reformulated as a prompted task. Out of confirmatory scope unless ahead.
    COREF = "coref"
    #: BBQ. Calibration-only, never load-bearing for an intersectional claim.
    QA = "qa"
    #: Templated open generation and red-teaming. The primary task type.
    OPEN_GEN = "open_gen"


#: Task types that have a single categorical answer, and therefore may carry ``Item.gold``.
#: ``pair_pll`` deliberately does not: both sentences live in the payload and there is no
#: single gold string.
GOLD_BEARING_TASKS: Final[frozenset[TaskType]] = frozenset({TaskType.QA, TaskType.COREF})


class TemplatePayload(TypedDict):
    """Payload of a template :class:`Item` (``benchmark="open_gen_templates"``)."""

    template_text: str
    slots: tuple[str, ...]


#: Registry typing ``Item.payload`` at the **loader boundary**. The shared record stays
#: source-agnostic, but mypy is not blind on the one field carrying the real data. P2 (extractive
#: loaders) and P8 (red-team) register their own payload types here as they are built.
PAYLOAD_TYPES: Final[dict[str, type]] = {
    "open_gen_templates": TemplatePayload,
}


# --------------------------------------------------------------------------------------
# Provenance and the lattice value registry
# --------------------------------------------------------------------------------------


class Provenance(BaseModel):
    """Where a record's content came from, and at which version.

    Required on every :class:`Item` **and** every :class:`AxisValue`: a descriptor source needs
    provenance exactly as much as an ingested dataset does, and gate P3 ("every lattice value
    traces to a provenance-carrying source") is unmeetable without it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = Field(min_length=1, description="Human-readable source identifier.")
    revision: str | None = Field(
        default=None, description="Upstream VCS or dataset revision (a commit or dataset SHA)."
    )
    version: str | None = Field(
        default=None, description="Upstream release label (e.g. a tag or version string)."
    )
    license: str = Field(
        min_length=1,
        description=(
            "Required, and deliberately not defaulted: an unrecorded license blocks the Zenodo "
            "release. Record 'unknown' explicitly rather than leaving it implicit."
        ),
    )
    sha256: str = Field(description="Content checksum of the retrieved source, lowercase hex.")
    retrieved_at: datetime

    @field_validator("sha256")
    @classmethod
    def _check_sha256(cls, value: str) -> str:
        if not _SHA256_RE.match(value):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        return value

    @model_validator(mode="after")
    def _require_a_version_anchor(self) -> Provenance:
        """At least one of revision/version must be present.

        A source pinned by neither cannot be re-retrieved by a reproducer, which makes the
        checksum a record of something unrecoverable. Sources carrying only one of the two
        record that one and leave the other None.
        """
        if self.revision is None and self.version is None:
            raise ValueError("provenance needs at least one of revision or version")
        return self


class AxisValue(BaseModel):
    """One value on one identity axis, carrying its own provenance.

    Provenance attaches to the **value**, never to a combination of values — see
    :class:`IdentityCell`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    axis: str = Field(min_length=1)
    value: str = Field(min_length=1)
    provenance: Provenance


class IdentityCell(BaseModel):
    """A point in the identity lattice: one axis value, or an intersection of two.

    Carries **no** provenance field. An order-2 cell combines two values that may come from
    different sources, so a single provenance on the cell would be ambiguous; provenance lives
    on the constituent :class:`AxisValue`\\ s instead.

    Hashable by contract — it is used as a grouping key throughout the metric layer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    cell_id: str = Field(min_length=1)
    axes: tuple[tuple[str, str], ...] = Field(
        description="Ordered ``(axis, value)`` pairs. A tuple so the record stays hashable."
    )
    order: int = Field(ge=1, le=2, description="Intersection depth. Fixed at 2 by invariant 6.")

    @model_validator(mode="after")
    def _check_axes(self) -> IdentityCell:
        if not self.axes:
            raise ValueError("axes must be non-empty")
        if self.order != len(self.axes):
            raise ValueError(f"order ({self.order}) must equal len(axes) ({len(self.axes)})")
        names = [axis for axis, _ in self.axes]
        if len(set(names)) != len(names):
            raise ValueError(f"axes must be key-unique, got duplicate axis in {names}")
        for axis, value in self.axes:
            if not axis or not value:
                raise ValueError("axis and value must both be non-empty")
        return self

    @property
    def axes_map(self) -> dict[str, str]:
        """The axes as a plain dict, for ergonomics. A copy: mutating it changes nothing."""
        return dict(self.axes)


class AxisValueRegistry:
    """The set of attested ``(axis, value)`` pairs a lattice may legally build cells from.

    Gate P3 requires that every lattice value trace to a provenance-carrying source and that
    every cell's axis pair resolve here. The registry is defined in P1 because it is part of
    the contract; P3 is what populates it from the descriptor source.
    """

    def __init__(self, values: tuple[AxisValue, ...] = ()) -> None:
        self._values: dict[tuple[str, str], AxisValue] = {(v.axis, v.value): v for v in values}

    def __len__(self) -> int:
        return len(self._values)

    def __contains__(self, key: tuple[str, str]) -> bool:
        return key in self._values

    @property
    def values(self) -> tuple[AxisValue, ...]:
        return tuple(self._values.values())

    def get(self, axis: str, value: str) -> AxisValue | None:
        return self._values.get((axis, value))

    def validate_cell(self, cell: IdentityCell) -> IdentityCell:
        """Raise unless every ``(axis, value)`` in ``cell`` is attested here."""
        missing = [pair for pair in cell.axes if pair not in self._values]
        if missing:
            raise ValueError(
                f"cell {cell.cell_id!r} uses values absent from the AxisValue registry: {missing}"
            )
        return cell

    def validate_cells(self, cells: tuple[IdentityCell, ...]) -> tuple[IdentityCell, ...]:
        for cell in cells:
            self.validate_cell(cell)
        return cells


# --------------------------------------------------------------------------------------
# Pipeline records
# --------------------------------------------------------------------------------------


class Item(BaseModel):
    """A unit of source material, for **every** path — not only ingested datasets.

    A template is an ``Item`` (``benchmark="open_gen_templates"``), and so is a red-team seed
    (``benchmark="redteam"``, provenance naming the generator model, revision, and seed). That
    is what keeps :attr:`Prompt.item_id` required rather than optional, makes
    ``TemplateRenderer.render(items, cells)`` a genuine cross-product of templates x cells, and
    preserves the property that every prompt traces back to something carrying provenance.
    Without it, templated prompts would have no audit trail and no version anchor.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    benchmark: str = Field(
        min_length=1,
        description=(
            "The *source* label: an ingested dataset, the templated open_gen set, or the "
            "red-team set. Ranges over sources, not only over ingested datasets — this is the "
            "key for the FDR family and for 'standardize within, never across'."
        ),
    )
    task_type: TaskType
    payload: FrozenDict = Field(
        default_factory=FrozenDict,
        description="The ONLY place source specifics live. Typed by PAYLOAD_TYPES at the loader.",
    )
    target_axes: tuple[str, ...] = ()
    gold: str | None = Field(
        default=None,
        description="Categorical answer. Legal only for qa and coref — see the validator.",
    )
    provenance: Provenance

    @model_validator(mode="after")
    def _gold_only_for_categorical_tasks(self) -> Item:
        if self.gold is not None and self.task_type not in GOLD_BEARING_TASKS:
            raise ValueError(
                f"gold is only meaningful for {sorted(t.value for t in GOLD_BEARING_TASKS)}; "
                f"task_type={self.task_type.value} must carry its content in payload instead "
                "(for pair_pll, both sentences live there and there is no single gold string)"
            )
        return self


class Prompt(BaseModel):
    """A rendered model input. Deterministic given its item, cell, and template version."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1, description="Required: every prompt traces to an Item.")
    cell_id: str | None = Field(
        default=None,
        description="None where a source attests no identity cell (e.g. an unlabelled item).",
    )
    rendered_text: str
    task_type: TaskType
    template_version: str = Field(min_length=1)
    decoding_key: str = Field(min_length=1)


class Response(BaseModel):
    """Model output, self-describing by design.

    ``decoding_hash`` exists because ``top_p``, ``top_k``, ``max_new_tokens``, and
    ``repetition_penalty`` are otherwise recorded nowhere, which would leave a response
    uninterpretable without the config that produced it. Determinism is a contract
    (invariant 5), so a response must carry enough to check it.

    **Paired-task shape (open decision 6, resolved 8 Sep 2026 as option (a)).** A ``pair_pll``
    measurement compares *two* sequences, which does not fit one score per response. Rather
    than split :class:`ItemScore` into two shapes — forcing every downstream consumer to branch
    on task type — the pair lives on this single record as
    :attr:`logprob_stereo` / :attr:`logprob_antistereo`, and the scorer reads the preference off
    it. ``open_gen`` responses leave both null.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    response_id: str = Field(min_length=1)
    prompt_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(
        min_length=1, description="Pinned upstream revision. An unpinned repo can move."
    )
    seed: int = Field(ge=0, le=MAX_SEED)
    temperature: float = Field(ge=0.0)
    decoding_key: str = Field(min_length=1)
    decoding_hash: str = Field(
        min_length=1, description="Hash of the fully resolved decoding config."
    )
    quant: QuantSetting = Field(description="Recorded constant, never an experimental axis.")
    text: str
    token_logprobs: tuple[float, ...] | None = Field(
        default=None, description="A tuple, not a list, so the record stays hashable."
    )

    # --- paired-task log-probabilities (pair_pll only; null on open_gen) ---
    # Named for what they MEASURE, not for any one dataset's columns: CrowS-Pairs'
    # sent_more/sent_less vocabulary is source-specific and would violate invariant 2 on a
    # shared record. These are whole-sequence log-probabilities, distinct from the per-token
    # `token_logprobs` above.
    logprob_stereo: float | None = Field(
        default=None,
        description=(
            "Sequence log-probability of the MORE-stereotyping member of the pair. Null for "
            "non-paired tasks."
        ),
    )
    logprob_antistereo: float | None = Field(
        default=None,
        description=(
            "Sequence log-probability of the LESS-stereotyping (anti-stereotypical) member. "
            "Null for non-paired tasks. SIGN CONVENTION: the bias signal is "
            "`logprob_stereo - logprob_antistereo`, positive when the model prefers the "
            "stereotyping sequence -- i.e. higher = more harm, matching the standardization "
            "rule. Reversing this pair silently flips the sign of every pair_pll effect."
        ),
    )

    @model_validator(mode="after")
    def _pair_logprobs_move_together(self) -> Response:
        """Half a pair is not a measurement.

        Note what this does *not* check: that the fields are null exactly when the task is
        ``open_gen``. ``Response`` carries no ``task_type`` (it lives on :class:`Prompt`), so
        that constraint spans two records and is enforced at the P5/P6 boundary, not here.
        """
        present = (self.logprob_stereo is not None, self.logprob_antistereo is not None)
        if any(present) and not all(present):
            raise ValueError(
                "logprob_stereo and logprob_antistereo are set together or not at all; got "
                f"logprob_stereo={self.logprob_stereo}, "
                f"logprob_antistereo={self.logprob_antistereo}"
            )
        return self

    @property
    def is_paired(self) -> bool:
        """True when this response carries a ``pair_pll`` measurement."""
        return self.logprob_stereo is not None and self.logprob_antistereo is not None

    @property
    def stereo_preference(self) -> float | None:
        """``logprob_stereo - logprob_antistereo``; higher = more harm. None if not paired."""
        if self.logprob_stereo is None or self.logprob_antistereo is None:
            return None
        return self.logprob_stereo - self.logprob_antistereo


class ItemScore(BaseModel):
    """One scorer's judgement of one response, standardized **within** its source.

    The trailing keys are deliberately **denormalized**. Without them ``Metric.compute`` would
    need a four-hop join (ItemScore -> Response -> Prompt -> Item) and E4's sensitivity analysis
    would have no access to seed/temperature at all. The redundancy buys flat Parquet, a
    joinless metric layer, and self-describing rows.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    score_id: str = Field(min_length=1)
    response_id: str = Field(min_length=1)
    scorer_id: str = Field(min_length=1)
    harm_type: str = Field(min_length=1)
    raw: float
    standardized: float = Field(
        description=(
            "Signed bias index standardized WITHIN source, higher = more harm. Never "
            "standardized across sources."
        )
    )

    # Denormalized grouping keys.
    #
    # CONTRACT CHANGE 18 Sep 2026 (P7): `prompt_id` added. The emergence estimator's standard
    # errors are cluster-robust by ITEM (the prompt), and its confidence intervals come from a
    # cluster bootstrap that resamples whole items -- the ~50 sampled generations per prompt are
    # repeated observations within one item and must never be counted as independent. Without
    # the key on the row, the metric layer needed the response file to cluster, and a score table
    # on its own could not be fitted. Required, with no default: a row without an item is
    # unclusterable, and "unknown" is not a cluster.
    prompt_id: str = Field(min_length=1, description="The item (prompt) this score belongs to.")
    cell_id: str | None = None
    benchmark: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    seed: int = Field(ge=0, le=MAX_SEED)
    temperature: float = Field(ge=0.0)
    condition: Condition = "model"


class CellStat(BaseModel):
    """Descriptive summary of one cell. **Tests no hypothesis.**

    Carries no ``p`` and no ``q``, by design and not by omission: significance is set only by
    ``StatsEngine``, and a cell-level description is not the unit an interaction is estimated
    on. Underpowered cells (``powered=False``) are descriptive-only and never load-bearing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    cell_id: str = Field(min_length=1)
    benchmark: str = Field(min_length=1)
    model: str = Field(min_length=1)
    harm_type: str = Field(min_length=1)
    n: int = Field(ge=0)
    mean_standardized: float
    ci_low: float
    ci_high: float
    powered: bool

    @model_validator(mode="after")
    def _check_interval(self) -> CellStat:
        if self.ci_low > self.ci_high:
            raise ValueError(f"ci_low ({self.ci_low}) must be <= ci_high ({self.ci_high})")
        return self


class EffectEstimate(BaseModel):
    """An inferential estimate for one ``axis_pair x term``.

    The unit is an axis *pair* and a *term*, not a cell: an interaction is undefined for a
    single order-1 cell. This is what makes the FDR family (benchmark x model x axis_pair x
    term x harm_type) definable and the pre-registration writable.

    ``q``, ``family_size`` and ``correction`` are set **only** by ``StatsEngine`` (invariant 3).
    They are validated to move together, because a partially corrected estimate is not
    interpretable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    benchmark: str = Field(min_length=1)
    model: str = Field(min_length=1)
    axis_pair: tuple[str, str]
    term: str = Field(min_length=1, description="e.g. an axis name for a main effect, or 'A:B'.")
    harm_type: str = Field(min_length=1)
    estimate: float
    se: float = Field(ge=0.0)
    ci_low: float
    ci_high: float
    p: float | None = Field(default=None, ge=0.0, le=1.0)
    q: float | None = Field(default=None, ge=0.0, le=1.0)
    powered: bool
    family_size: int | None = Field(default=None, ge=1)
    correction: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _check_estimate(self) -> EffectEstimate:
        if self.ci_low > self.ci_high:
            raise ValueError(f"ci_low ({self.ci_low}) must be <= ci_high ({self.ci_high})")

        axis_a, axis_b = self.axis_pair
        if not axis_a or not axis_b:
            raise ValueError("axis_pair entries must be non-empty")
        if axis_a == axis_b:
            raise ValueError(f"axis_pair must name two distinct axes, got {self.axis_pair}")

        correction_fields = (self.q, self.family_size, self.correction)
        if any(f is not None for f in correction_fields) and not all(
            f is not None for f in correction_fields
        ):
            raise ValueError(
                "q, family_size and correction are set together by StatsEngine or not at all; "
                f"got q={self.q}, family_size={self.family_size}, correction={self.correction}"
            )
        if self.q is not None and self.p is None:
            raise ValueError("q cannot be set without p: there is nothing to correct")
        return self

    @property
    def corrected(self) -> bool:
        """True once ``StatsEngine`` has stamped this estimate. The report layer requires it."""
        return self.q is not None and self.family_size is not None and self.correction is not None


#: Convenience alias used by the interfaces module and tests.
AnyRecord = Annotated[
    Item | Prompt | Response | ItemScore | CellStat | EffectEstimate | AxisValue | IdentityCell,
    Field(description="Any shared record in the contract layer."),
]
