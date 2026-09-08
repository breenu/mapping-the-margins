"""The identity lattice: cells, descriptor composition, and power bookkeeping (P3).

This module turns a validated :class:`~margins.lattice.config.LatticeConfig` into the
:class:`~margins.schemas.IdentityCell` set the rest of the pipeline audits. It renders nothing,
runs nothing, and scores nothing — P4a injects these cells into templates, and it is the first
stage allowed to produce a :class:`~margins.schemas.Prompt`.

**The anti-cartesian rule.** Order-2 cells are generated for **declared axis pairs only**, never
for the full product of every axis against every other. That is a power decision, not a
convenience: crossing five axes exhaustively would multiply the cell count while dividing the
rendering budget per cell, and — for confirmatory pairs, which *are* the FDR family
(PROJECT_CONTEXT.md section 8) — would push the BH threshold low enough that nothing survives
correction. :func:`DeclaredLattice.cells` therefore emits exactly
``sum over declared pairs of |A| * |B|`` order-2 cells and no others.

**Descriptor composition is a CONSTRUCT CHOICE, not a neutral operation.** ``compose_descriptor``
concatenates the constituent surface phrases in ``compose_rank`` order, so
``{age: older, ability: blind}`` becomes ``"older blind"`` and never ``"blind older"``. Three
things follow, and all three are caveats a reader is entitled to:

* English adjective order is not free. A fixed rank produces fluent phrases for some pairs and
  stilted ones for others, and fluency is itself a property a language model reacts to. The rank
  is constant across cells, so it cannot manufacture a *main* effect, but an ordering that is
  awkward for one combination and natural for another can contribute to an *interaction* — which
  is precisely the estimand. The rank chosen in the shipped config is the one that reads
  naturally across every declared pair; the residual risk is real and is stated in the paper.
* A conjunction ("older and blind") or an appositive ("older, blind") would encode a different
  relation between the two identities than juxtaposition does. Juxtaposition was chosen because
  it matches HolisticBias's own ``<descriptor> <noun>`` construction; it is a choice, not the
  only option.
* No carrier noun is attached here. ``"older blind"`` is a modifier phrase; whether it becomes
  "an older blind person", "an older blind woman", or something else is P4a's decision, made in
  the template, where the audit trail for it belongs.

**Power bookkeeping marks cells, it does not analyse power.** ``projected_n`` comes from the
config's rendering estimate and ``powered`` is ``projected_n >= min_cell_n`` against a
*pre-registered placeholder* threshold. TODO(P9): the real minimum N comes from the power
analysis (alpha=0.05, power=0.8). Until then every ``powered=True`` flag is provisional — but the
flags exist *before* any run, which is what gate P3 asks for, and underpowered cells are
descriptive-only by invariant 4 regardless.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from margins.lattice.config import LatticeConfig, LatticeValue
from margins.schemas import AxisValue, AxisValueRegistry, IdentityCell

__all__ = [
    "AXIS_SEPARATOR",
    "DESCRIPTOR_JOINER",
    "VALUE_SEPARATOR",
    "CellRecord",
    "DeclaredLattice",
    "make_cell_id",
]

#: ``cell_id`` grammar: ``axis=value`` joined by ``|`` in canonical ``compose_rank`` order. The
#: id is a *label*, never something to parse structure back out of — ``Metric.compute`` takes the
#: cells themselves for exactly that reason (PROJECT_CONTEXT.md section 3).
VALUE_SEPARATOR = "="
AXIS_SEPARATOR = "|"

#: Surface phrases are joined by a single space. See the module docstring: this is a construct
#: choice, not the only defensible one.
DESCRIPTOR_JOINER = " "


def make_cell_id(axes: Sequence[tuple[str, str]]) -> str:
    """Build a cell id from already-canonically-ordered ``(axis, value)`` pairs."""
    return AXIS_SEPARATOR.join(f"{axis}{VALUE_SEPARATOR}{value}" for axis, value in axes)


@dataclass(frozen=True, slots=True)
class CellRecord:
    """A cell plus everything the inventory reports about it before any run.

    Kept separate from :class:`~margins.schemas.IdentityCell` on purpose: ``projected_n``,
    ``powered`` and ``confirmatory`` are properties of *this lattice configuration*, not of the
    cell itself, and a shared record must not grow a field that only one phase understands
    (invariant 2).
    """

    cell: IdentityCell
    descriptor: str
    axis_pair: tuple[str, str] | None
    projected_n: int
    powered: bool
    confirmatory: bool

    @property
    def cell_id(self) -> str:
        return self.cell.cell_id

    @property
    def order(self) -> int:
        return self.cell.order


class DeclaredLattice:
    """The identity lattice declared by a :class:`LatticeConfig`.

    Satisfies the :class:`~margins.interfaces.Lattice` Protocol. Deterministic by construction:
    the same config always yields the same cells, in the same order, with the same ids — nothing
    here consults a clock, an RNG, or the filesystem.

    Ordering is canonical rather than incidental. Order-1 cells come first, sorted by axis
    ``compose_rank`` then by the order values were declared in (so a human can put the reference
    category first and have it stay first). Order-2 cells follow, sorted by the canonical rank of
    their axis pair, then by the two constituent value positions. Reordering the ``pairs`` list
    in YAML therefore changes nothing about the output.
    """

    def __init__(self, config: LatticeConfig) -> None:
        self._config = config
        self._values: dict[str, tuple[LatticeValue, ...]] = {
            axis: config.axes[axis].lattice_values(axis) for axis in config.ordered_axes
        }
        self._surfaces: dict[tuple[str, str], str] = {
            (value.axis, value.value): value.surface
            for values in self._values.values()
            for value in values
        }
        self._registry = AxisValueRegistry(
            tuple(
                AxisValue(
                    axis=value.axis, value=value.value, provenance=config.provenance_for(value)
                )
                for value in config.lattice_values()
            )
        )
        self._records: tuple[CellRecord, ...] = tuple(self._build_records())
        # Every cell must resolve in the registry -- gate P3's second clause, asserted at
        # construction so an unattested value can never leave this module.
        self._registry.validate_cells(tuple(record.cell for record in self._records))

    # ---------------------------------------------------------------- construction

    def _build_records(self) -> Iterator[CellRecord]:
        config = self._config
        confirmatory_axes = config.confirmatory_axes
        order_1_n = config.power.renderings_per_cell.order_1
        min_n = config.power.min_cell_n

        for axis in config.ordered_axes:
            for value in self._values[axis]:
                axes = ((axis, value.value),)
                cell = IdentityCell(cell_id=make_cell_id(axes), axes=axes, order=1)
                yield CellRecord(
                    cell=cell,
                    descriptor=value.surface,
                    axis_pair=None,
                    projected_n=order_1_n,
                    powered=order_1_n >= min_n,
                    # An order-1 cell is load-bearing when its axis carries the main effect of a
                    # confirmatory interaction. It is not itself a pre-registered hypothesis.
                    confirmatory=axis in confirmatory_axes,
                )

        for pair in config.ordered_pairs:
            axis_a, axis_b = config.canonical_pair(pair.axes)
            projected_n = config.renderings_for_pair(pair.axes)
            confirmatory = pair.status == "confirmatory"
            for value_a in self._values[axis_a]:
                for value_b in self._values[axis_b]:
                    paired = ((axis_a, value_a.value), (axis_b, value_b.value))
                    cell = IdentityCell(cell_id=make_cell_id(paired), axes=paired, order=2)
                    yield CellRecord(
                        cell=cell,
                        descriptor=DESCRIPTOR_JOINER.join((value_a.surface, value_b.surface)),
                        axis_pair=(axis_a, axis_b),
                        projected_n=projected_n,
                        powered=projected_n >= min_n,
                        confirmatory=confirmatory,
                    )

    # ---------------------------------------------------------------- the Lattice Protocol

    def cells(self) -> list[IdentityCell]:
        """Every cell in the lattice, in canonical order (the ``Lattice`` Protocol)."""
        return [record.cell for record in self._records]

    # ---------------------------------------------------------------- accessors

    @property
    def config(self) -> LatticeConfig:
        return self._config

    @property
    def registry(self) -> AxisValueRegistry:
        """The attested ``(axis, value)`` registry every cell was validated against."""
        return self._registry

    def records(self) -> list[CellRecord]:
        """Every cell with its inventory bookkeeping, in the same order as :meth:`cells`."""
        return list(self._records)

    def axis_values(self, axis: str) -> tuple[LatticeValue, ...]:
        """Resolved values for one axis, in declaration order."""
        return self._values[axis]

    def values(self) -> tuple[LatticeValue, ...]:
        """Every resolved value, in canonical axis order then declaration order."""
        return self._config.lattice_values()

    def powered_cells(self) -> list[IdentityCell]:
        """Cells meeting the pre-registered minimum N. Only these are load-bearing."""
        return [record.cell for record in self._records if record.powered]

    def confirmatory_cells(self) -> list[IdentityCell]:
        """Cells belonging to the pre-registered confirmatory set."""
        return [record.cell for record in self._records if record.confirmatory]

    # ---------------------------------------------------------------- composition

    def compose_descriptor(self, cell: IdentityCell) -> str:
        """Compose ``cell`` into its surface descriptor phrase.

        **A construct choice, not a neutral operation** — see the module docstring for the three
        caveats (adjective ordering, juxtaposition vs conjunction, no carrier noun). The rule
        itself is one line: take each constituent's surface phrase, order by the axis's
        ``compose_rank``, join with a single space.

        Raises:
            KeyError: If the cell uses an ``(axis, value)`` this lattice does not declare.
        """
        ordered = sorted(cell.axes, key=lambda pair: self._config.axis_rank(pair[0]))
        return DESCRIPTOR_JOINER.join(self._surfaces[pair] for pair in ordered)

    # ---------------------------------------------------------------- summary

    def summary(self) -> dict[str, int]:
        """Counts a human checks before committing a rendering budget.

        ``projected_renderings`` is the number that decides whether E1 fits the free-tier quota,
        which is why it is surfaced here rather than discovered during the W2 timing pilot.
        """
        records = self._records
        return {
            "axes": len(self._config.axes),
            "values": len(self._registry),
            "cells": len(records),
            "cells_order_1": sum(1 for record in records if record.order == 1),
            "cells_order_2": sum(1 for record in records if record.order == 2),
            "declared_pairs": len(self._config.pairs),
            "confirmatory_pairs": len(self._config.confirmatory_pairs),
            "non_gender_confirmatory_pairs": len(self._config.non_gender_confirmatory_pairs),
            "exploratory_pairs": len(self._config.exploratory_pairs),
            "powered_cells": sum(1 for record in records if record.powered),
            "descriptive_cells": sum(1 for record in records if not record.powered),
            "confirmatory_cells": sum(1 for record in records if record.confirmatory),
            "projected_renderings": sum(record.projected_n for record in records),
        }
