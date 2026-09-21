"""Construct-validity screening for the extractive source (P2).

PROJECT_CONTEXT.md section 9 asks for exactly this: *"For CrowS-Pairs run minimal-pair checks
(edit distance, single-attribute swap); flag non-minimal pairs; ... log every exclusion with a
machine-readable reason."* Gate P2 adds the property that makes such a log trustworthy:
**screening is deterministic.**

Deterministic here means something stronger than "no random seed". :func:`screen` is a pure
function of ``(rows, config, lattice)``: no I/O, no RNG, no clock, no dependence on input order,
and every decision is reached by the same ordered sequence of checks for every row. That is why
the exclusion reasons and the order they are applied in are **not** configurable -- a config that
could reorder them could change which reason a given item is excluded under, and an exclusion log
whose reasons depend on settings is not a record of anything.

The screen answers two separate questions, and keeping them separate is what makes the report
readable:

* **Is this a usable minimal pair at all?** Identical sentences, duplicated rows, more than one
  differing region, an over-wide swap, or a contrast that is not about the mapped axis -- these are
  EXCLUSIONS. The item does not become a record.
* **What does the pair attest?** An out-of-lattice category, or a category whose descriptor is not
  a lattice value, is NOT an exclusion. The item is kept, marked, and reported. It simply carries
  no cell, and a record with no cell is an honest record of a sentence pair we cannot place in the
  lattice -- which is a finding about the benchmark, not a defect in the row.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import NamedTuple

from margins.benchmarks.attest import (
    AttestedValue,
    DiffSummary,
    attested_values,
    build_cell_index,
    build_surface_index,
    declared_cell,
    diff_summary,
    values_in_tokens,
)
from margins.benchmarks.config import BenchmarkConfig
from margins.benchmarks.crows_pairs import CrowsPairsPayload, CrowsRow
from margins.lattice import DeclaredLattice
from margins.schemas import Item, TaskType

__all__ = [
    "CoverageRow",
    "Exclusion",
    "ExclusionReason",
    "KeptWithoutCell",
    "ScreeningResult",
    "screen",
]


class ExclusionReason(str, Enum):
    """Machine-readable exclusion reasons. Every excluded row carries exactly one.

    A string enum so the value lands in CSV and Parquet as itself, and so a downstream filter is
    written against a name rather than a spelling.
    """

    #: The two sentences are token-identical. There is no contrast to measure.
    IDENTICAL_SENTENCES = "identical_sentences"
    #: An earlier row already carried this exact (sent_more, sent_less) pair. Kept once.
    DUPLICATE_PAIR = "duplicate_pair"
    #: More than one contiguous differing region: the pair differs in more than one thing, so the
    #: likelihood difference is not attributable to the intended attribute.
    NOT_MINIMAL_PAIR = "not_minimal_pair"
    #: One region, but too wide. A long phrase swapped for a short one changes sequence length and
    #: token frequency, which a likelihood difference cannot distinguish from identity preference.
    SWAP_TOO_WIDE = "swap_too_wide"
    #: The differing region contains no lattice value on the category's mapped axis, so the
    #: contrast is not about the cell the item would be assigned to.
    CONTRAST_NOT_ON_MAPPED_AXIS = "contrast_not_on_mapped_axis"


class KeptWithoutCell(str, Enum):
    """Why a kept item carries no cell. **These are not exclusions.**

    The distinction matters for the report and for the claim: an excluded row is one we judged
    unusable, a kept-without-cell row is one that is perfectly usable and simply is not about
    anything the lattice declares. Conflating them would let a coverage gap read as a quality
    problem.
    """

    #: The upstream category has no lattice axis (religion, nationality, ...).
    OUT_OF_LATTICE_CATEGORY = "out_of_lattice_category"
    #: The category maps to an axis, but the descriptor contrasted is not one of that axis's
    #: lattice values -- CrowS' "The poor"/"The rich" against `working-class`/`wealthy`.
    NO_ATTESTED_LATTICE_VALUE = "no_attested_lattice_value"


class Exclusion(NamedTuple):
    """One excluded row, with everything needed to audit the decision without the source file."""

    row_index: int
    bias_type: str
    reason: ExclusionReason
    detail: str
    sent_stereo: str
    sent_antistereo: str


class CoverageRow(NamedTuple):
    """One kept item's placement, for the coverage report."""

    row_index: int
    bias_type: str
    mapped_axis: str | None
    cell_id: str | None
    order: int
    out_of_lattice: bool
    no_cell_reason: str | None
    second_axis_declined: str | None


class ScreeningResult(NamedTuple):
    """Everything :func:`screen` decided, ready to export."""

    items: tuple[Item, ...]
    exclusions: tuple[Exclusion, ...]
    coverage: tuple[CoverageRow, ...]
    #: Row indices admitted only because a differing region was pure grammatical concord. Reported
    #: so the exclusion count stays comparable to a stricter reading of the same data.
    admitted_by_agreement: tuple[int, ...] = ()

    @property
    def n_kept(self) -> int:
        return len(self.items)

    @property
    def n_excluded(self) -> int:
        return len(self.exclusions)

    def counts_by_reason(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for exclusion in self.exclusions:
            counts[exclusion.reason.value] = counts.get(exclusion.reason.value, 0) + 1
        return dict(sorted(counts.items()))

    def counts_by_order(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.coverage:
            key = f"order_{row.order}" if row.order else "no_cell"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def order_2_cells(self) -> dict[str, int]:
        """``cell_id -> item count`` for the order-2 cells, the quantity that decides whether the
        extractive paradigm can support an interaction estimate at all."""
        counts: dict[str, int] = {}
        for row in self.coverage:
            if row.order == 2 and row.cell_id is not None:
                counts[row.cell_id] = counts.get(row.cell_id, 0) + 1
        return dict(sorted(counts.items()))


def _item_id(config: BenchmarkConfig, row: CrowsRow) -> str:
    """A readable natural key. The row index is the upstream row, which the pin makes stable.

    Separated by ``/``, matching P4a's ``open_gen_templates/describe``, and NOT by ``::``: the
    prompt id joins its four fields with ``::``, so an item id containing that separator would make
    the id unsplittable -- readable to a human and ambiguous to everything else.
    """
    return f"{config.benchmark}/{row.row_index:04d}"


def screen(
    rows: Sequence[CrowsRow], *, config: BenchmarkConfig, lattice: DeclaredLattice
) -> ScreeningResult:
    """Screen raw rows into ``Item`` records, an exclusion log, and a coverage table.

    Pure: no I/O, no RNG, no clock. Rows are processed in the order given and every decision
    depends only on the row and the two configs, so the same inputs produce byte-identical outputs
    and a shuffled input produces the same per-row decisions.
    """
    config.validate_against_lattice(lattice.config.ordered_axes)

    surfaces = build_surface_index(lattice)
    cells = build_cell_index(lattice)
    axis_for = config.axis_for
    provenance = config.provenance()
    settings = config.screening

    agreement = config.screening.agreement

    items: list[Item] = []
    exclusions: list[Exclusion] = []
    coverage: list[CoverageRow] = []
    admitted_by_agreement: list[int] = []
    seen: set[tuple[str, str]] = set()

    def exclude(row: CrowsRow, reason: ExclusionReason, detail: str) -> None:
        exclusions.append(
            Exclusion(
                row_index=row.row_index,
                bias_type=row.bias_type,
                reason=reason,
                detail=detail,
                sent_stereo=row.sent_stereo,
                sent_antistereo=row.sent_antistereo,
            )
        )

    for row in rows:
        if row.bias_type not in axis_for:
            raise ValueError(
                f"row {row.row_index} carries bias_type {row.bias_type!r}, which "
                f"{config.key!r} does not declare. Known categories are "
                f"{sorted(axis_for)}. Upstream gaining a category is a human decision: add it to "
                "the config's `attributes`, mapped to an axis or explicitly out-of-lattice. "
                "Silently skipping it would drop items from a published dataset with no record."
            )

        # --- 1. Is this a usable minimal pair? Order is fixed and not configurable. ---
        summary = diff_summary(row.sent_stereo, row.sent_antistereo)
        if summary.identical:
            exclude(row, ExclusionReason.IDENTICAL_SENTENCES, "tokenized sentences are equal")
            continue

        key = (row.sent_more, row.sent_less)
        if key in seen:
            exclude(row, ExclusionReason.DUPLICATE_PAIR, "an earlier row carries this exact pair")
            continue
        seen.add(key)

        changes = summary.content_changes(agreement)
        n_spans = DiffSummary.spans(changes)
        if n_spans > settings.max_differing_spans:
            exclude(
                row,
                ExclusionReason.NOT_MINIMAL_PAIR,
                f"{n_spans} differing spans, limit {settings.max_differing_spans}",
            )
            continue

        # Recorded, not merely allowed: a screen that quietly admits rows a stricter reading would
        # have excluded must say how many, or the exclusion count is not comparable to anyone
        # else's.
        if agreement and DiffSummary.spans(summary.changes) > n_spans:
            admitted_by_agreement.append(row.row_index)

        width = DiffSummary.width(changes)
        if width > settings.max_differing_tokens:
            exclude(
                row,
                ExclusionReason.SWAP_TOO_WIDE,
                f"widest differing span is {width} tokens, limit {settings.max_differing_tokens}",
            )
            continue

        # --- 2. What does the pair attest? Not attesting is KEPT AND MARKED, never excluded. ---
        mapped_axis = axis_for[row.bias_type]
        out_of_lattice = mapped_axis is None

        primary: AttestedValue | None = None
        no_cell_reason: str | None = None

        if out_of_lattice:
            no_cell_reason = KeptWithoutCell.OUT_OF_LATTICE_CATEGORY.value
        else:
            assert mapped_axis is not None  # narrowed by `out_of_lattice`
            contrasted = values_in_tokens(
                DiffSummary.stereo_tokens(changes), surfaces, axis=mapped_axis
            )
            if contrasted:
                primary = contrasted[0]
            elif settings.require_contrast_on_mapped_axis:
                no_cell_reason = KeptWithoutCell.NO_ATTESTED_LATTICE_VALUE.value

        # --- 3. A second axis, only if it is genuinely SHARED by both sentences. ---
        cell_id: str | None = None
        order = 0
        second_declined: str | None = None
        attested: list[tuple[str, str]] = []

        if primary is not None:
            attested.append((primary.axis, primary.value))
            in_stereo = {(v.axis, v.value) for v in attested_values(row.sent_stereo, surfaces)}
            in_anti = {(v.axis, v.value) for v in attested_values(row.sent_antistereo, surfaces)}
            shared = sorted(pair for pair in (in_stereo & in_anti) if pair[0] != primary.axis)
            shared_axes = sorted({axis for axis, _ in shared})

            order_2 = None
            if len(shared_axes) == 1 and len(shared) == 1:
                order_2 = declared_cell(cells, (*attested, shared[0]))
                if order_2 is None:
                    second_declined = (
                        f"{primary.axis} x {shared[0][0]} is not a declared lattice pair"
                    )
            elif len(shared_axes) > 1:
                # Declining is not the same as excluding: the order-1 reading is still valid, and
                # CHOOSING one of several attested second axes would be inventing the cell this
                # module exists to prevent.
                second_declined = f"{len(shared_axes)} candidate second axes: {shared_axes}"
            elif len(shared) > 1:
                second_declined = f"two values on one second axis: {shared}"

            if order_2 is not None:
                attested.append(shared[0])
                cell_id = order_2.cell_id
                order = 2
            else:
                order_1 = declared_cell(cells, tuple(attested))
                if order_1 is None:  # pragma: no cover - a lattice value always has an order-1 cell
                    raise ValueError(
                        f"row {row.row_index} attests {attested}, which resolves to no declared "
                        "order-1 cell. The lattice and the surface index disagree, which should "
                        "be impossible: both are built from `lattice.values()`."
                    )
                cell_id = order_1.cell_id
                order = 1

        payload: CrowsPairsPayload = {
            "row_index": row.row_index,
            "sent_more": row.sent_more,
            "sent_less": row.sent_less,
            "stereo_antistereo": row.stereo_antistereo,
            "bias_type": row.bias_type,
            "sent_stereo": row.sent_stereo,
            "sent_antistereo": row.sent_antistereo,
            "mapped_axis": mapped_axis,
            "out_of_lattice": out_of_lattice,
            "attested": tuple(attested),
        }

        items.append(
            Item(
                id=_item_id(config, row),
                benchmark=config.benchmark,
                task_type=TaskType.PAIR_PLL,
                payload=dict(payload),
                target_axes=tuple(sorted({axis for axis, _ in attested})),
                gold=None,
                provenance=provenance,
            )
        )
        coverage.append(
            CoverageRow(
                row_index=row.row_index,
                bias_type=row.bias_type,
                mapped_axis=mapped_axis,
                cell_id=cell_id,
                order=order,
                out_of_lattice=out_of_lattice,
                no_cell_reason=no_cell_reason,
                second_axis_declined=second_declined,
            )
        )

    return ScreeningResult(
        items=tuple(items),
        exclusions=tuple(exclusions),
        coverage=tuple(coverage),
        admitted_by_agreement=tuple(admitted_by_agreement),
    )
