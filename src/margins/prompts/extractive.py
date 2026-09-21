"""The ``ExtractiveRenderer``: a cell DERIVED from its item, never injected (P4b).

The counterpart to P4a's ``TemplateRenderer``, and its exact inverse. ``interfaces.PromptGenerator``
spells out why one signature has two incompatible semantics:

    ``render(items, cells)`` reads as a cross-product, which is simply wrong for extractive
    sources: you cannot assign "Black woman" to a CrowS-Pairs item about religion.

So this renderer never consults the cell list for *material*. A CrowS-Pairs sentence pair is about
whatever it is about; the only honest cell is the one its own text attests, which
:mod:`margins.benchmarks.screening` derived and recorded in the item's payload. What the cell list
is used for here is the opposite of injection -- **validation**: gate P4b is
*"emits no cell unattested in its source item"*, and :meth:`ExtractiveRenderer._check_attestation`
is that sentence made executable. A payload that claims a cell its own sentences do not attest
raises rather than rendering.

**Open decision 7 is closed here (20 Sep 2026).** A paired prompt carries both sequences on one
record -- ``rendered_text_stereo`` and ``rendered_text_antistereo`` -- mirroring decision 6's one
``Response`` carrying both log-probabilities. The pair travels as a single unit: one ``Prompt``,
one ``Response``, one score.

**Nothing is rendered, in the P4a sense.** There is no frame, no carrier, no article rule and no
leakage guard, because no text is composed: the two sequences ARE the upstream sentences, copied
verbatim. That is deliberate. Rewriting a benchmark's sentences -- even to normalise punctuation --
would mean scoring a text the published dataset does not contain, and the comparison to prior work
would be to a number nobody else can reproduce. The only transformation applied anywhere in this
vertical is the ``stereo_antistereo`` resolution, which happens once at the loader boundary.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Final

from margins.benchmarks.attest import build_surface_index
from margins.lattice import DeclaredLattice
from margins.prompts.config import ExtractivePromptSetConfig
from margins.schemas import IdentityCell, Item, Prompt, TaskType

__all__ = [
    "PROMPT_ID_DIGEST_LENGTH",
    "ExtractiveRenderer",
    "make_extractive_prompt_id",
]

#: Same length as P4a's, so the two prompt families read alike in a table.
PROMPT_ID_DIGEST_LENGTH: Final[int] = 12

_PROMPT_ID_SEPARATOR: Final[str] = "::"
#: A unit separator, so no field's content can forge a boundary in the digest preimage.
_DIGEST_SEPARATOR: Final[str] = "\x1f"

#: Stands in for the cell slot of the readable id when an item attests none. A literal rather than
#: an empty string, so the id keeps four fields and stays greppable.
NO_CELL: Final[str] = "nocell"


def make_extractive_prompt_id(
    *,
    set_key: str,
    item_id: str,
    cell_id: str | None,
    template_version: str,
    decoding_key: str,
    stereo: str,
    antistereo: str,
) -> str:
    """``<set_key>::<cell_id|nocell>::<item_id>::<digest12>``.

    A readable natural key plus a digest over the full rendering key -- the same construction P4a
    uses, and for the same reason: a config edit that forgets to bump ``template_version`` shows up
    as changed ids in a diff rather than as silent drift.

    The four fields are joined with ``::``, so no field may contain it. Item ids use ``/`` for
    exactly that reason (``crows_pairs/0000``), matching P4a's ``open_gen_templates/describe``.

    **Both sequences enter the preimage, and that is load-bearing.** ``make_response_id``'s own
    preimage does not cover the scored texts, so ``prompt_id`` is the only place a pair's identity
    can live; two different pairs sharing an item would otherwise collide on one response id.
    """
    preimage = _DIGEST_SEPARATOR.join(
        (
            template_version,
            set_key,
            item_id,
            cell_id or NO_CELL,
            decoding_key,
            stereo,
            antistereo,
        )
    )
    fields = (set_key, cell_id or NO_CELL, item_id)
    offenders = [f for f in fields if _PROMPT_ID_SEPARATOR in f]
    if offenders:
        raise ValueError(
            f"prompt id fields may not contain {_PROMPT_ID_SEPARATOR!r}, which joins them: "
            f"{offenders}. An id that cannot be split is readable to a human and ambiguous to "
            "everything else."
        )
    digest = hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:PROMPT_ID_DIGEST_LENGTH]
    return _PROMPT_ID_SEPARATOR.join((*fields, digest))


class ExtractiveRenderer:
    """Renders ``pair_pll`` items into paired prompts, deriving each cell from its item.

    Satisfies :class:`margins.interfaces.PromptGenerator`. Deterministic: the same items, the same
    lattice and the same config reproduce byte-identical prompts.
    """

    def __init__(self, config: ExtractivePromptSetConfig, lattice: DeclaredLattice) -> None:
        self._config = config
        self._lattice = lattice
        self._surfaces = build_surface_index(lattice)
        cells = lattice.cells()
        self._cells: dict[str, IdentityCell] = {c.cell_id: c for c in cells}
        self._by_axes: dict[frozenset[tuple[str, str]], IdentityCell] = {
            frozenset(c.axes): c for c in cells
        }

    @property
    def config(self) -> ExtractivePromptSetConfig:
        return self._config

    # ----------------------------------------------------------------------------------
    # Validation -- gate P4b lives here
    # ----------------------------------------------------------------------------------

    def _check_item(self, item: Item) -> None:
        """Refuse anything this renderer must not derive a cell from."""
        if item.benchmark != self._config.benchmark:
            raise ValueError(
                f"item {item.id!r} has benchmark {item.benchmark!r}; this ExtractiveRenderer is "
                f"configured for {self._config.benchmark!r}. A cell DERIVED from one source's "
                "items must not be rendered under another source's prompt set: the two would "
                "share a template_version and become indistinguishable downstream."
            )
        if item.task_type is not TaskType.PAIR_PLL:
            raise ValueError(
                f"item {item.id!r} has task_type {item.task_type.value!r}; ExtractiveRenderer "
                f"renders {TaskType.PAIR_PLL.value!r} only. An item whose cell must be INJECTED "
                "belongs to TemplateRenderer (P4a)."
            )

    def _check_attestation(self, item: Item, cell_id: str | None, stereo: str, anti: str) -> None:
        """**Gate P4b: emit no cell unattested in the source item.**

        Three checks, and each has caught a different way of being wrong:

        1. The cell must be one the lattice declares. A cell minted anywhere else cannot be
           resolved by the metric layer, which raises a hard ``KeyError`` for an unknown
           ``cell_id`` and would take the whole run down with it.
        2. Every ``(axis, value)`` of the cell must actually occur in the item's sentences. This is
           the sentence "never emit a prompt whose cell is not attested in the source item",
           checked against the TEXT rather than against the payload that claims it -- a payload is
           an assertion, and an assertion is what this guard exists to verify.
        3. For an order-2 cell, the second value must occur in **both** sentences. A value present
           in only one of them is part of the contrast, not shared context, so treating it as a
           second axis would attribute the likelihood difference to a cell that only half the pair
           is in.
        """
        if cell_id is None:
            return

        cell = self._cells.get(cell_id)
        if cell is None:
            raise ValueError(
                f"item {item.id!r} claims cell {cell_id!r}, which the lattice "
                f"({self._lattice.config.lattice_version}) does not declare. A renderer may not "
                "mint a cell: the metric layer resolves cell_id -> axes through the lattice and "
                "raises for one it cannot place."
            )

        matched = {
            (entry.axis, entry.value): (
                bool(pattern.search(stereo)),
                bool(pattern.search(anti)),
            )
            for entry, pattern in self._surfaces.entries
        }
        for axis, value in cell.axes:
            in_stereo, in_anti = matched.get((axis, value), (False, False))
            if not (in_stereo or in_anti):
                raise ValueError(
                    f"item {item.id!r} claims cell {cell_id!r}, but neither of its sentences "
                    f"attests ({axis}={value}). A cell is DERIVED from its item, never assigned "
                    "to it -- this is gate P4b."
                )

        if cell.order == 2:
            contrasted = {(a, v) for a, v in cell.axes if matched[(a, v)] != (True, True)}
            shared = {(a, v) for a, v in cell.axes} - contrasted
            if not shared:
                raise ValueError(
                    f"item {item.id!r} claims order-2 cell {cell_id!r}, but neither value appears "
                    "in BOTH sentences. An order-2 cell needs one contrasted axis and one SHARED "
                    "axis; two contrasted axes mean the pair is not minimal and the likelihood "
                    "difference is not attributable to either."
                )

    # ----------------------------------------------------------------------------------
    # Rendering
    # ----------------------------------------------------------------------------------

    def render(self, items: Sequence[Item], cells: Sequence[IdentityCell]) -> list[Prompt]:
        """One paired prompt per item, each carrying both sequences.

        ``cells`` is accepted because the Protocol declares it and is used **only** to validate:
        this renderer derives, and a cell list it consulted for material would be a cell list it
        could inject from. Passing a narrower list than the lattice declares narrows what may be
        emitted; passing a wider one cannot widen it, because the lattice is the authority.
        """
        allowed = {cell.cell_id for cell in cells}
        prompts: list[Prompt] = []

        for item in items:
            self._check_item(item)

            stereo = str(item.payload["sent_stereo"])
            antistereo = str(item.payload["sent_antistereo"])
            attested = tuple(tuple(pair) for pair in item.payload.get("attested", ()))
            cell_id = self._cell_id_for(item, attested)

            if cell_id is not None and cell_id not in allowed:
                raise ValueError(
                    f"item {item.id!r} attests cell {cell_id!r}, which is not in the cell list "
                    "passed to render(). The renderer does not silently drop it: a cell the "
                    "caller excluded but the item attests is a disagreement a human should see."
                )
            self._check_attestation(item, cell_id, stereo, antistereo)

            prompts.append(
                Prompt(
                    prompt_id=make_extractive_prompt_id(
                        set_key=self._config.key,
                        item_id=item.id,
                        cell_id=cell_id,
                        template_version=self._config.template_version,
                        decoding_key=self._config.decoding_key,
                        stereo=stereo,
                        antistereo=antistereo,
                    ),
                    item_id=item.id,
                    cell_id=cell_id,
                    rendered_text="",
                    task_type=TaskType.PAIR_PLL,
                    template_version=self._config.template_version,
                    decoding_key=self._config.decoding_key,
                    rendered_text_stereo=stereo,
                    rendered_text_antistereo=antistereo,
                )
            )
        return prompts

    def _cell_id_for(self, item: Item, attested: Sequence[tuple[str, ...]]) -> str | None:
        """The declared cell for an item's attested values, or None when it attests nothing."""
        if not attested:
            return None
        pairs = frozenset((str(a), str(v)) for a, v in attested)
        cell = self._by_axes.get(pairs)
        if cell is None:
            raise ValueError(
                f"item {item.id!r} attests {sorted(pairs)}, which resolves to no declared cell. "
                "The screen and the lattice disagree, which should be impossible: both are built "
                "from `lattice.values()`."
            )
        return cell.cell_id
