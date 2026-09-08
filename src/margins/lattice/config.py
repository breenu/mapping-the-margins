"""Declarative configuration for the identity lattice (P3).

Why a declarative config rather than code: gate P3 requires that **every lattice value trace
to a documented source** and that the confirmatory axis pairs be **named in advance**. Both are
pre-registration facts, not implementation details — they have to live in a file a human writes,
reviews, and git-tags, not in a Python literal buried in a builder.

Three things this module refuses to allow, structurally:

1. **A value with no provenance.** Every value names an upstream source key, an upstream axis, a
   bucket, and the exact term. ``verify_lattice_values`` (in
   :mod:`margins.lattice.descriptors.holistic_bias`) checks those coordinates against the pinned
   upstream file, so a transcription error is a test failure rather than a fabricated descriptor
   in a paper.
2. **A lattice that cannot meet gate P3.** ``>= 2`` confirmatory pairs must be
   non-gender x non-gender (novelty claim 5). This is a validator, not a convention: a config that
   would fail the gate does not load.
3. **An unbounded confirmatory family.** The confirmatory pairs *are* the FDR family
   (PROJECT_CONTEXT.md section 8), so their number is capped. An unbounded family drives the BH
   threshold low enough that realized power collapses regardless of per-cell N.

**Construct caveat.** Choosing which descriptors represent an axis is a construct choice, not a
neutral sampling operation: HolisticBias offers dozens of terms per axis and this config keeps
three to five. The selection rule is recorded in the config's own ``description`` and reproduced
in ``value_provenance.md``; the ``preference`` annotation is carried verbatim from upstream so a
reader can see which terms upstream marked dispreferred and check that none were used.

Hydra composes the plain YAML, the result is converted to a plain dict, and validation happens
here in Pydantic — no Hydra structured configs, no hydra-zen (PROJECT_CONTEXT.md section 6).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from margins.schemas import Provenance

__all__ = [
    "DEFAULT_LATTICE",
    "MAX_CONFIRMATORY_PAIRS",
    "MAX_VALUES_PER_AXIS",
    "MIN_NON_GENDER_CONFIRMATORY_PAIRS",
    "MIN_VALUES_PER_AXIS",
    "AxisPairSpec",
    "AxisSpec",
    "AxisValueSpec",
    "DescriptorSource",
    "LatticeConfig",
    "LatticeValue",
    "PairRenderingOverride",
    "PowerSpec",
    "RenderingBudget",
    "lattice_config_from_mapping",
    "load_lattice_config",
]

#: How a value is realized in surface text. ``descriptor`` injects an explicit identity phrase
#: ("working-class"); ``name`` proxies an identity through a given name. Name-proxying is
#: **discouraged** — it confounds the identity signal with name frequency, orthography, and
#: nationality — so a name-realized value must carry an explicit note recording that confound.
Realization = Literal["descriptor", "name"]

#: ``descriptor_taxonomy`` is an external, pinned descriptor source (HolisticBias).
#: ``custom`` is a value this project introduced, which must carry a written justification.
SourceKind = Literal["descriptor_taxonomy", "custom"]

#: ``confirmatory`` pairs are pre-registered and constitute the entire FDR family.
#: ``exploratory`` pairs are reported descriptively and carry no significance claim.
PairStatus = Literal["confirmatory", "exploratory"]

#: The confirmatory pairs are the FDR family, so the family size is bounded here rather than
#: discovered at P9 (PROJECT_CONTEXT.md section 8 pre-registers 6-8 pairs).
MAX_CONFIRMATORY_PAIRS: Final[int] = 8

#: Novelty claim 5 and gate P3: the primary intersections must reach beyond gender-anchored
#: cells. Enforced as a validator so no lattice config can silently fail the gate.
MIN_NON_GENDER_CONFIRMATORY_PAIRS: Final[int] = 2

#: Values per axis are deliberately few. Cell count grows multiplicatively and projected N per
#: cell falls with it, so breadth here is power spent (PROJECT_CONTEXT.md section 14). The
#: confirmatory config aims for 3-5; the hard bounds leave room for small test fixtures.
MIN_VALUES_PER_AXIS: Final[int] = 2
MAX_VALUES_PER_AXIS: Final[int] = 6

#: The confirmatory lattice shipped with the repo.
DEFAULT_LATTICE: Final[str] = "confirmatory_v1"


class DescriptorSource(BaseModel):
    """A pinned source that lattice values may be drawn from.

    Mirrors :class:`margins.schemas.Provenance` and converts to one via :meth:`to_provenance`.
    It exists separately because the config also records *where* to fetch the source (``url``,
    ``path``), which a record's provenance does not carry.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: SourceKind
    source: str = Field(min_length=1)
    url: str | None = Field(default=None, description="Where a reproducer retrieves the source.")
    path: str | None = Field(default=None, description="Path within the upstream repository.")
    revision: str | None = None
    version: str | None = None
    license: str = Field(min_length=1)
    sha256: str = Field(min_length=64, max_length=64)
    retrieved_at: datetime

    def to_provenance(self) -> Provenance:
        """The shared-record provenance every value drawn from this source carries.

        ``url`` and ``path`` are dropped deliberately: they are retrieval instructions, and
        adding them to a shared record would violate invariant 2.
        """
        return Provenance(
            source=self.source,
            revision=self.revision,
            version=self.version,
            license=self.license,
            sha256=self.sha256,
            retrieved_at=self.retrieved_at,
        )


class AxisValueSpec(BaseModel):
    """One declared value, as written in YAML.

    ``value`` and ``surface`` both default to ``source_term``, so the common case — the registry
    value, the injected phrase, and the upstream term are the same string — is a single line of
    YAML and cannot drift between the three.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_term: str = Field(
        min_length=1, description="The descriptor term EXACTLY as it appears upstream."
    )
    source_bucket: str = Field(
        min_length=1, description="Upstream bucket the term sits in (e.g. 'working_class')."
    )
    source_axis: str | None = Field(
        default=None, description="Upstream axis key; defaults to the axis-level declaration."
    )
    source: str | None = Field(
        default=None, description="Source key; defaults to the axis-level declaration."
    )
    value: str | None = Field(
        default=None, description="Registry value and cell-id token; defaults to source_term."
    )
    surface: str | None = Field(
        default=None, description="Phrase injected downstream; defaults to source_term."
    )
    realization: Realization = "descriptor"
    preference: str = Field(
        default="none",
        min_length=1,
        description=(
            "The upstream preference annotation, recorded VERBATIM ('reviewed', 'dispreferred', "
            "'polarizing', or 'none'). Verified against the pinned file, so it cannot drift."
        ),
    )
    note: str | None = Field(
        default=None,
        description="Construct note. Required for name-realized values and for custom sources.",
    )

    @model_validator(mode="after")
    def _name_realization_needs_a_confound_note(self) -> AxisValueSpec:
        """Name-proxied identity is a documented construct choice or it is not allowed.

        A given name carries nationality, cohort, class and orthography signal alongside the
        identity it proxies, so an effect estimated on names is not cleanly attributable to the
        axis. Requiring the note forces that confound into the provenance table rather than into
        a footnote nobody writes.
        """
        if self.realization == "name" and not (self.note or "").strip():
            raise ValueError(
                f"value {self.source_term!r} is name-realized and must carry a `note` recording "
                "the construct choice and the confound (name frequency, orthography, "
                "nationality) that name-proxying introduces"
            )
        return self


class LatticeValue(BaseModel):
    """A fully resolved lattice value: what enters the registry and the provenance table.

    Separate from :class:`AxisValueSpec` because the spec's axis-level defaults are optional and
    this is not: every field here is required, so nothing downstream has to re-derive a default
    or handle a ``None``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    axis: str = Field(min_length=1)
    value: str = Field(min_length=1)
    surface: str = Field(min_length=1)
    realization: Realization
    source_key: str = Field(min_length=1)
    source_axis: str = Field(min_length=1)
    source_bucket: str = Field(min_length=1)
    source_term: str = Field(min_length=1)
    preference: str = Field(min_length=1)
    note: str | None = None


class AxisSpec(BaseModel):
    """One identity axis and the small set of values representing it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str = Field(min_length=1, description="Human-readable axis name for tables and figures.")
    gender_anchored: bool = Field(
        description=(
            "True for axes that are about gender or sex. Gate P3 counts confirmatory pairs whose "
            "BOTH axes are non-gender-anchored, so this flag is load-bearing for novelty claim 5."
        )
    )
    compose_rank: int = Field(
        ge=0,
        description=(
            "Canonical position of this axis, used for cell-id ordering, axis-pair ordering, and "
            "descriptor composition. Unique across axes. A CONSTRUCT CHOICE for composition: it "
            "fixes English adjective order, which is not a neutral operation."
        ),
    )
    source: str = Field(min_length=1, description="Default source key for this axis's values.")
    source_axis: str = Field(min_length=1, description="Default upstream axis key.")
    values: tuple[AxisValueSpec, ...]
    note: str | None = None

    @model_validator(mode="after")
    def _check_values(self) -> AxisSpec:
        count = len(self.values)
        if not MIN_VALUES_PER_AXIS <= count <= MAX_VALUES_PER_AXIS:
            raise ValueError(
                f"axis has {count} values; keep it between {MIN_VALUES_PER_AXIS} and "
                f"{MAX_VALUES_PER_AXIS}. Cell count grows multiplicatively, so extra values are "
                "power spent, and exhaustive coverage is exploratory-only."
            )
        resolved = [spec.value or spec.source_term for spec in self.values]
        if len(set(resolved)) != count:
            raise ValueError(f"axis values must be unique, got {resolved}")
        return self

    def lattice_values(self, axis: str) -> tuple[LatticeValue, ...]:
        """Resolve this axis's declared values against its axis-level defaults."""
        return tuple(
            LatticeValue(
                axis=axis,
                value=spec.value or spec.source_term,
                surface=spec.surface or spec.source_term,
                realization=spec.realization,
                source_key=spec.source or self.source,
                source_axis=spec.source_axis or self.source_axis,
                source_bucket=spec.source_bucket,
                source_term=spec.source_term,
                preference=spec.preference,
                note=spec.note,
            )
            for spec in self.values
        )


class AxisPairSpec(BaseModel):
    """A DECLARED axis pair. Order-2 cells exist only for these.

    This is the anti-cartesian guarantee: the lattice never crosses every axis with every other
    axis. The full product would multiply the cell count, collapse projected N per cell, and
    (for confirmatory pairs) inflate the FDR family until nothing survives correction.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    axes: tuple[str, str]
    status: PairStatus
    rationale: str = Field(
        min_length=1, description="Why this pair. Pre-registration text, not a comment."
    )

    @model_validator(mode="after")
    def _check_axes(self) -> AxisPairSpec:
        first, second = self.axes
        if first == second:
            raise ValueError(f"an axis pair must name two distinct axes, got {self.axes}")
        return self


class RenderingBudget(BaseModel):
    """Projected renderings per cell, by intersection order.

    An *estimate* fed to power bookkeeping, not a measurement: the real per-cell N is known only
    after P4a fixes the template set and P5 runs. Its job here is to mark cells powered vs
    descriptive **before** any run, which gate P3 requires.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    order_1: int = Field(gt=0)
    order_2: int = Field(gt=0)


class PairRenderingOverride(BaseModel):
    """A per-pair rendering budget, for pairs that cannot afford the default."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    axes: tuple[str, str]
    renderings_per_cell: int = Field(gt=0)
    reason: str = Field(min_length=1)


class PowerSpec(BaseModel):
    """Inputs to power bookkeeping. **No power analysis happens in P3.**"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_cell_n: int = Field(
        gt=0,
        description=(
            "PRE-REGISTERED PLACEHOLDER. TODO(P9): replace with the minimum cell N derived from "
            "the real power analysis (alpha=0.05, power=0.8, minimum detectable effect fixed "
            "there). Until then this threshold is a declared convention, and any cell it marks "
            "powered is provisional."
        ),
    )
    renderings_per_cell: RenderingBudget
    pair_overrides: tuple[PairRenderingOverride, ...] = ()


class LatticeConfig(BaseModel):
    """The validated lattice declaration.

    Frozen, like every config in this project: a lattice cannot change mid-run, because the cell
    inventory it produced is what the pre-registration and the FDR family were written against.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    lattice_version: str = Field(
        min_length=1, description="Bump on any change to axes, values, or pairs."
    )
    description: str = Field(
        min_length=1,
        description="The selection rule. Reproduced verbatim in value_provenance.md.",
    )
    sources: dict[str, DescriptorSource]
    axes: dict[str, AxisSpec]
    pairs: tuple[AxisPairSpec, ...]
    power: PowerSpec

    # ---------------------------------------------------------------- validation

    @model_validator(mode="after")
    def _check_sources_and_axes(self) -> LatticeConfig:
        if not self.sources:
            raise ValueError("a lattice config must declare at least one source")
        if not self.axes:
            raise ValueError("a lattice config must declare at least one axis")

        ranks = [axis.compose_rank for axis in self.axes.values()]
        if len(set(ranks)) != len(ranks):
            raise ValueError(f"compose_rank must be unique across axes, got {ranks}")

        for axis_name, axis in self.axes.items():
            for value in axis.lattice_values(axis_name):
                source = self.sources.get(value.source_key)
                if source is None:
                    raise ValueError(
                        f"{axis_name}={value.value!r} names source {value.source_key!r}, which is "
                        f"not declared; declared sources are {sorted(self.sources)}"
                    )
                if source.kind == "custom" and not (value.note or "").strip():
                    raise ValueError(
                        f"{axis_name}={value.value!r} comes from custom source "
                        f"{value.source_key!r} and must carry a `note` justifying it; a custom "
                        "value with no justification cannot satisfy gate P3"
                    )
        return self

    @model_validator(mode="after")
    def _check_pairs(self) -> LatticeConfig:
        if not self.pairs:
            raise ValueError("a lattice config must declare at least one axis pair")

        seen: set[tuple[str, str]] = set()
        for pair in self.pairs:
            for axis_name in pair.axes:
                if axis_name not in self.axes:
                    raise ValueError(
                        f"pair {pair.axes} names undeclared axis {axis_name!r}; declared axes are "
                        f"{sorted(self.axes)}"
                    )
            canonical = self.canonical_pair(pair.axes)
            if canonical in seen:
                raise ValueError(f"axis pair {canonical} is declared more than once")
            seen.add(canonical)

        for override in self.power.pair_overrides:
            canonical = self.canonical_pair(override.axes)
            if canonical not in seen:
                raise ValueError(
                    f"pair_overrides names {canonical}, which is not a declared axis pair"
                )

        confirmatory = self.confirmatory_pairs
        if not confirmatory:
            raise ValueError("a lattice config must declare at least one confirmatory axis pair")
        if len(confirmatory) > MAX_CONFIRMATORY_PAIRS:
            raise ValueError(
                f"{len(confirmatory)} confirmatory pairs exceeds the cap of "
                f"{MAX_CONFIRMATORY_PAIRS}. The confirmatory set IS the FDR family; an unbounded "
                "family drives the BH threshold low enough that realized power collapses."
            )

        non_gender = self.non_gender_confirmatory_pairs
        if len(non_gender) < MIN_NON_GENDER_CONFIRMATORY_PAIRS:
            raise ValueError(
                f"gate P3 requires at least {MIN_NON_GENDER_CONFIRMATORY_PAIRS} confirmatory "
                f"pairs with both axes non-gender-anchored; this config has {len(non_gender)} "
                f"({sorted(non_gender)}). Non-gender x non-gender coverage is novelty claim 5 "
                "and is reachable only through the open_gen and red-team paths."
            )
        return self

    # ---------------------------------------------------------------- accessors

    def axis_rank(self, axis: str) -> int:
        """Canonical rank of ``axis``. Raises :class:`KeyError` for an undeclared axis."""
        return self.axes[axis].compose_rank

    def canonical_pair(self, axes: tuple[str, str]) -> tuple[str, str]:
        """Order an axis pair by ``compose_rank``.

        Every artifact — cell ids, ``axis_pair`` columns, the inventory — uses this order, so the
        order a human happened to type in YAML never reaches an output.
        """
        first, second = sorted(axes, key=self.axis_rank)
        return (first, second)

    def is_non_gender_pair(self, axes: tuple[str, str]) -> bool:
        """True when neither axis is gender-anchored."""
        return not any(self.axes[axis].gender_anchored for axis in axes)

    @property
    def ordered_axes(self) -> tuple[str, ...]:
        """Axis names in canonical (``compose_rank``) order."""
        return tuple(sorted(self.axes, key=self.axis_rank))

    @property
    def ordered_pairs(self) -> tuple[AxisPairSpec, ...]:
        """Declared pairs in canonical order, so output never depends on YAML ordering."""

        def rank(pair: AxisPairSpec) -> tuple[int, int]:
            first, second = self.canonical_pair(pair.axes)
            return (self.axis_rank(first), self.axis_rank(second))

        return tuple(sorted(self.pairs, key=rank))

    @property
    def confirmatory_pairs(self) -> tuple[tuple[str, str], ...]:
        """The pre-registered confirmatory pairs. **This set is the FDR family.**"""
        return tuple(
            self.canonical_pair(pair.axes)
            for pair in self.ordered_pairs
            if pair.status == "confirmatory"
        )

    @property
    def exploratory_pairs(self) -> tuple[tuple[str, str], ...]:
        """Declared pairs carrying no significance claim."""
        return tuple(
            self.canonical_pair(pair.axes)
            for pair in self.ordered_pairs
            if pair.status == "exploratory"
        )

    @property
    def non_gender_confirmatory_pairs(self) -> tuple[tuple[str, str], ...]:
        """Confirmatory pairs with both axes non-gender-anchored (gate P3, novelty claim 5)."""
        return tuple(pair for pair in self.confirmatory_pairs if self.is_non_gender_pair(pair))

    @property
    def confirmatory_axes(self) -> frozenset[str]:
        """Axes appearing in at least one confirmatory pair."""
        return frozenset(axis for pair in self.confirmatory_pairs for axis in pair)

    def lattice_values(self) -> tuple[LatticeValue, ...]:
        """Every resolved value, in canonical axis order then declaration order."""
        return tuple(
            value
            for axis_name in self.ordered_axes
            for value in self.axes[axis_name].lattice_values(axis_name)
        )

    def provenance_for(self, value: LatticeValue) -> Provenance:
        """The shared-record provenance for ``value``. Validated to exist at load time."""
        return self.sources[value.source_key].to_provenance()

    def renderings_for_pair(self, axes: tuple[str, str]) -> int:
        """Projected renderings per order-2 cell of ``axes``, honouring any override."""
        canonical = self.canonical_pair(axes)
        for override in self.power.pair_overrides:
            if self.canonical_pair(override.axes) == canonical:
                return override.renderings_per_cell
        return self.power.renderings_per_cell.order_2


# --------------------------------------------------------------------------------------
# The Hydra seam
# --------------------------------------------------------------------------------------


def lattice_config_from_mapping(data: Mapping[str, Any]) -> LatticeConfig:
    """Validate a plain mapping into a :class:`LatticeConfig`.

    This is where Hydra stops and Pydantic starts: Hydra/OmegaConf composes YAML, the caller
    converts to a plain container, and every rule above is enforced here. Keeping Hydra out of
    the type system is the cheap side of the Hydra/Pydantic friction.
    """
    return LatticeConfig.model_validate(dict(data))


def default_lattice_config_dir() -> Path:
    """``<repo>/configs/lattice``. Override it for tests or an installed package."""
    return Path(__file__).resolve().parents[3] / "configs" / "lattice"


def load_lattice_config(
    name: str = DEFAULT_LATTICE, *, config_dir: Path | None = None
) -> LatticeConfig:
    """Compose ``<config_dir>/<name>.yaml`` with Hydra and validate it.

    Args:
        name: Config name without the ``.yaml`` suffix.
        config_dir: Directory holding the lattice config group. Defaults to the repository's
            ``configs/lattice``.

    Returns:
        The validated config.
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    directory = (config_dir or default_lattice_config_dir()).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"lattice config dir does not exist: {directory}")

    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        composed = compose(config_name=name)
    data = OmegaConf.to_container(composed, resolve=True)
    if not isinstance(data, dict):
        raise TypeError(f"composed lattice config {name!r} is not a mapping, got {type(data)!r}")
    return lattice_config_from_mapping(cast(Mapping[str, Any], data))
