"""``TemplateRenderer`` — the ``open_gen`` prompt generator (P4a).

This is the first stage that turns an :class:`~margins.schemas.IdentityCell` into a
:class:`~margins.schemas.Prompt`, and it is where the generation vertical — the path carrying
RQ2, the primary estimand — actually begins.

**Injected, not derived.** The :class:`~margins.interfaces.PromptGenerator` Protocol carries two
incompatible semantics behind one signature (PROJECT_CONTEXT.md section 3). This module implements
the *injecting* one: a cell is placed into a neutral template frame, which is what makes arbitrary
lattice cells reachable — including the non-gender x non-gender cells that gate P3 requires and
that no extractive source attests. The *deriving* one (``ExtractiveRenderer``, P4b) reads a cell
off a source item and rides with the CrowS-Pairs loader; it is deliberately not here.

TODO(P4b) — **open decision 7, the paired-task prompt representation, is untouched and stays
open.** ``Prompt.rendered_text`` is a single string, so it cannot carry the two sequences a
``pair_pll`` response needs; the options are a second rendered-text field, a canonical joined
encoding, or a paired-prompt variant. Nothing is encoded here because ``open_gen`` never needs it:
:class:`~margins.prompts.config.PromptSetConfig` types ``task_type`` as ``Literal["open_gen"]`` and
:meth:`TemplateRenderer.render` refuses any item that is not an ``open_gen`` template. The decision
must be made before the CrowS-Pairs loader, not after.

**Three guarantees this module is responsible for.**

1. **Determinism.** Nothing here reads a clock, an RNG, or the filesystem. The same config, the
   same template files and the same cells produce the same prompts, with the same ids, in the same
   order. ``prompt_id`` embeds a digest over the full rendering key *including the rendered text*,
   so a template edit that forgets to bump ``template_version`` still shows up as changed ids in a
   diff rather than as silent drift.
2. **Template leakage.** Identity cues must appear **only** in the injected slot. A frame that
   mentions an identity term, or a second slot that leaks one, would put identity signal outside
   the manipulated variable and contaminate every cell equally except where it interacts — which
   is exactly the estimand. :meth:`TemplateRenderer.render` therefore checks that the set of
   lattice terms present in a rendered prompt is **exactly** the set the cell contributes: no more
   (leakage) and no fewer (a fill that silently did not happen).
3. **Grammatical frames.** The article is chosen by the prompt-set config, not baked into the
   template, because the correct article depends on the composed descriptor. Descriptors are
   pinned P3 provenance: when a phrasing is wrong, the lever is the frame or the compose rule,
   never the sourced term.

TODO(P6): the scorer identity-drift control (PROJECT_CONTEXT.md section 9) reuses **these**
templates and **these** descriptors with a neutral control text, which is what makes the control
matched by construction. It is a scoring-stage concern and is not built here, but the shape of
this module is what makes it cheap later.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

from jinja2 import Environment, StrictUndefined, Template, meta

from margins.lattice.build import DeclaredLattice
from margins.prompts.config import PromptSetConfig, TemplateFile, normalize_template_text
from margins.schemas import IdentityCell, Item, Prompt, TaskType

__all__ = [
    "CARRIER_VARIABLE",
    "CONJOINED_VARIABLE",
    "CONJUNCTION_JOINER",
    "DESCRIPTOR_VARIABLE",
    "IDENTITY_VARIABLE",
    "PROMPT_ID_DIGEST_LENGTH",
    "TEMPLATE_ITEM_BENCHMARK",
    "TEMPLATE_ITEM_PREFIX",
    "PreparedTemplate",
    "TemplateRenderer",
    "axis_vocabulary",
    "identity_vocabulary",
    "make_prompt_id",
    "matched_terms",
    "template_item_id",
    "template_items",
    "template_key",
]

#: Every template is an :class:`~margins.schemas.Item` under this source label, so a rendered
#: prompt traces to something carrying :class:`~margins.schemas.Provenance` (PROJECT_CONTEXT.md
#: section 3). ``Prompt.item_id`` stays required rather than optional because of this.
TEMPLATE_ITEM_BENCHMARK: Final[str] = "open_gen_templates"
TEMPLATE_ITEM_PREFIX: Final[str] = f"{TEMPLATE_ITEM_BENCHMARK}/"

#: Variables a template frame may reference. ``identity`` is the convenience composition
#: ``article + descriptor + carrier``; the three parts are exposed separately so a frame can place
#: them itself. ``descriptor`` is the ONLY identity-bearing variable.
DESCRIPTOR_VARIABLE: Final[str] = "descriptor"
CARRIER_VARIABLE: Final[str] = "carrier"
ARTICLE_VARIABLE: Final[str] = "article"
IDENTITY_VARIABLE: Final[str] = "identity"

#: The **predicative** form of a composed descriptor: constituents joined by "and" instead of
#: juxtaposed. It exists because a carrier-free frame puts the descriptor after a copula, and
#: English predicative position rejects juxtaposed adjectives: ``"someone who is an older blind"``
#: is broken where ``"an older blind person"`` is fine. Only ``"older and blind"`` works there.
#:
#: **This is a distinct CONSTRUCT CHOICE from juxtaposition, not a formatting variant.** A
#: conjunction asserts the two identities as separate co-predicated properties, where juxtaposition
#: lets the pair read as a single compound category (``margins.lattice.build`` flags exactly this
#: in its composition caveats). It reuses P3's canonical ``compose_rank`` order and changes ONLY
#: the joiner, so the ordering construct choice is unchanged.
CONJOINED_VARIABLE: Final[str] = "descriptor_conjoined"

#: Joiner for the predicative form. See :data:`CONJOINED_VARIABLE`.
CONJUNCTION_JOINER: Final[str] = " and "

TEMPLATE_VARIABLES: Final[tuple[str, ...]] = (
    DESCRIPTOR_VARIABLE,
    CONJOINED_VARIABLE,
    CARRIER_VARIABLE,
    ARTICLE_VARIABLE,
    IDENTITY_VARIABLE,
)

#: A frame must reference at least one of these, or the cell is never injected at all and every
#: "rendering" of that template is the same string for all 110 cells.
IDENTITY_BEARING_VARIABLES: Final[frozenset[str]] = frozenset(
    {DESCRIPTOR_VARIABLE, CONJOINED_VARIABLE, IDENTITY_VARIABLE}
)

#: Non-word, non-letter placeholders used to extract a template's static frame. Deliberately
#: control characters: a word-shaped sentinel could itself match a vocabulary term.
_DESCRIPTOR_SENTINEL: Final[str] = "\x01"
_CARRIER_SENTINEL: Final[str] = "\x02"
_ARTICLE_SENTINEL: Final[str] = "\x03"
_SENTINELS: Final[tuple[str, ...]] = (
    _DESCRIPTOR_SENTINEL,
    _CARRIER_SENTINEL,
    _ARTICLE_SENTINEL,
)

#: How much of the rendering-key digest rides in a ``prompt_id``. Matches the convention already
#: used by ``margins.config.make_run_id``: readable natural key + a short content digest.
PROMPT_ID_DIGEST_LENGTH: Final[int] = 12

#: Separator inside a ``prompt_id``. Not ``|``, which is the lattice's axis separator and already
#: appears inside every order-2 ``cell_id``.
_PROMPT_ID_SEPARATOR: Final[str] = "::"

#: Field separator for the digest preimage. A control character so no field value can forge one.
_DIGEST_SEPARATOR: Final[str] = "\x1f"

#: Words that carry no axis signal on their own and would make the frame check useless.
_AXIS_STOPWORDS: Final[frozenset[str]] = frozenset({"and", "the", "of", "or", "x"})

_WORD_RE: Final[re.Pattern[str]] = re.compile(r"[a-z]+")


def _environment() -> Environment:
    """A strict, plain-text Jinja environment.

    ``StrictUndefined`` so a typo'd variable is a loud failure rather than an empty string in a
    prompt. Autoescaping is off deliberately: the output is plain text for a language model, and
    HTML-escaping ``"Deaf & blind"`` would corrupt the descriptor.
    """
    # autoescape stays False: plain text output, never HTML.
    return Environment(
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=False,
    )


# --------------------------------------------------------------------------------------
# Templates as Items
# --------------------------------------------------------------------------------------


def template_item_id(template_id: str) -> str:
    """The :class:`~margins.schemas.Item` id for a template."""
    return f"{TEMPLATE_ITEM_PREFIX}{template_id}"


def template_key(item: Item) -> str:
    """The short template key behind an item id, for use in a ``prompt_id``."""
    return item.id.removeprefix(TEMPLATE_ITEM_PREFIX)


def template_items(config: PromptSetConfig, files: Sequence[TemplateFile]) -> list[Item]:
    """Turn template files into provenance-carrying :class:`~margins.schemas.Item` records.

    A template is an ``Item`` (PROJECT_CONTEXT.md section 3) so that
    ``TemplateRenderer.render(items, cells)`` is a genuine cross-product of templates x cells and
    every prompt traces back to something version-anchored. ``payload`` carries the frame text and
    the slots the frame actually references — detected from the parsed template rather than
    declared by hand, so the two cannot drift apart.
    """
    environment = _environment()
    items: list[Item] = []
    for file in files:
        declared = meta.find_undeclared_variables(environment.parse(file.text))
        unknown = sorted(declared - set(TEMPLATE_VARIABLES))
        if unknown:
            raise ValueError(
                f"template {file.id!r} references unknown variables {unknown}; a frame may use "
                f"only {sorted(TEMPLATE_VARIABLES)}"
            )
        if not declared & IDENTITY_BEARING_VARIABLES:
            raise ValueError(
                f"template {file.id!r} references no identity-bearing variable "
                f"({sorted(IDENTITY_BEARING_VARIABLES)}); it would render identically for every "
                "cell, which is not a prompt set but a constant"
            )
        items.append(
            Item(
                id=template_item_id(file.id),
                benchmark=TEMPLATE_ITEM_BENCHMARK,
                task_type=TaskType.OPEN_GEN,
                payload={
                    "template_text": file.text,
                    "slots": tuple(v for v in TEMPLATE_VARIABLES if v in declared),
                },
                target_axes=(),
                gold=None,
                provenance=config.provenance_for(file.sha256),
            )
        )
    return items


# --------------------------------------------------------------------------------------
# Leakage vocabulary
# --------------------------------------------------------------------------------------


def _term_pattern(term: str) -> re.Pattern[str]:
    """Match ``term`` as a whole word, hyphens included in the boundary.

    ``(?<![\\w-])`` / ``(?!...)`` rather than ``\\b`` so ``male`` does not match inside ``female``
    and ``middle-class`` does not match inside ``middle-classes``.
    """
    return re.compile(rf"(?<![\w-]){re.escape(term)}(?![\w-])", re.IGNORECASE)


def identity_vocabulary(lattice: DeclaredLattice) -> tuple[str, ...]:
    """Every term that counts as an identity cue: each value's surface phrase and its id token.

    Both, not just the surface: a frame that hard-codes the registry value of a descriptor leaks
    identity just as effectively as one that hard-codes the surface phrase.
    """
    terms = {value.surface for value in lattice.values()} | {
        value.value for value in lattice.values()
    }
    return tuple(sorted(terms))


def axis_vocabulary(lattice: DeclaredLattice) -> tuple[str, ...]:
    """Words naming an *axis* rather than a value ("race", "disability", "gender").

    Checked against template frames only. A frame saying "regardless of their race" primes the
    axis for every cell on it, which is a construct problem that no value-level check would catch.
    Axis words never appear inside a composed descriptor, so this vocabulary is not applied to
    rendered prompts.
    """
    words: set[str] = set()
    for name, axis in lattice.config.axes.items():
        for source in (name.replace("_", " "), axis.label):
            words.update(
                word for word in _WORD_RE.findall(source.lower()) if word not in _AXIS_STOPWORDS
            )
    return tuple(sorted(words))


def matched_terms(text: str, vocabulary: Iterable[str]) -> tuple[str, ...]:
    """Vocabulary terms occurring as whole words in ``text``, sorted and deduplicated."""
    return tuple(sorted({term for term in vocabulary if _term_pattern(term).search(text)}))


# --------------------------------------------------------------------------------------
# Prompt ids
# --------------------------------------------------------------------------------------


def make_prompt_id(
    *,
    template_key: str,
    cell_id: str,
    carrier: str,
    template_version: str,
    item_id: str,
    decoding_key: str,
    rendered_text: str,
) -> str:
    """A stable, deterministic, content-addressed ``prompt_id``.

    Shape: ``<template>::<cell_id>::<carrier>::<digest12>``. The readable prefix is the natural
    key a human greps for; the digest is what makes the id unique across template versions and
    what turns an unversioned template edit into a visible id change. Nothing here depends on
    position in a list, so ids are identical no matter what order the inputs arrived in.
    """
    preimage = _DIGEST_SEPARATOR.join(
        (template_version, item_id, cell_id, carrier, decoding_key, rendered_text)
    )
    digest = hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:PROMPT_ID_DIGEST_LENGTH]
    return _PROMPT_ID_SEPARATOR.join((template_key, cell_id, carrier, digest))


# --------------------------------------------------------------------------------------
# The renderer
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PreparedTemplate:
    """A validated template: the item it came from, its compiled frame, and its static text."""

    item: Item
    key: str
    template: Template
    frame: str

    @property
    def slots(self) -> tuple[str, ...]:
        slots = self.item.payload.get("slots", ())
        return tuple(str(slot) for slot in slots)


class TemplateRenderer:
    """Injects lattice cells into neutral template frames (the ``PromptGenerator`` Protocol).

    ``open_gen`` only. An item that is not an ``open_gen`` template is rejected rather than
    coerced: the extractive semantics belong to ``ExtractiveRenderer`` (P4b), and a renderer that
    quietly accepted both would be the exact confusion PROJECT_CONTEXT.md section 3 warns about —
    you cannot assign "working-class Black" to a CrowS-Pairs item about religion.
    """

    def __init__(self, config: PromptSetConfig, lattice: DeclaredLattice) -> None:
        self._config = config
        self._lattice = lattice
        self._environment = _environment()
        self._identity_vocabulary = identity_vocabulary(lattice)
        self._axis_vocabulary = axis_vocabulary(lattice)
        self._surfaces: dict[tuple[str, str], str] = {
            (value.axis, value.value): value.surface for value in lattice.values()
        }
        self._check_carriers()

    # ---------------------------------------------------------------- accessors

    @property
    def config(self) -> PromptSetConfig:
        return self._config

    @property
    def lattice(self) -> DeclaredLattice:
        return self._lattice

    # ---------------------------------------------------------------- validation

    def _check_carriers(self) -> None:
        """A carrier noun may not itself be a lattice term.

        This is the leakage guard applied to the frame's own vocabulary: a carrier that *is* an
        identity term would make every prompt on every cell carry an extra, unmanipulated
        identity, and the per-prompt term check could never tell the two apart.
        """
        for carrier in self._config.carriers:
            leaked = matched_terms(carrier.noun, self._identity_vocabulary)
            if leaked:
                raise ValueError(
                    f"carrier noun {carrier.noun!r} is itself a lattice identity term "
                    f"{list(leaked)}; a carrier must not carry the signal being manipulated"
                )

    def prepare(self, items: Sequence[Item]) -> list[PreparedTemplate]:
        """Validate and compile template items. Called by :meth:`render`; public for tests.

        Raises:
            ValueError: If an item is not an ``open_gen`` template, if two items resolve to the
                same template key, or if a frame fails the leakage / placement checks.
        """
        prepared: list[PreparedTemplate] = []
        seen: set[str] = set()
        for item in items:
            self._check_item(item)
            key = template_key(item)
            if key in seen:
                raise ValueError(f"duplicate template key {key!r}; template ids must be unique")
            seen.add(key)

            text = str(item.payload["template_text"])
            template = self._environment.from_string(text)
            frame = self._frame_of(template, key)
            prepared.append(PreparedTemplate(item=item, key=key, template=template, frame=frame))

        if not prepared:
            raise ValueError("no templates to render; a prompt set needs at least one frame")
        return prepared

    def _check_item(self, item: Item) -> None:
        if item.benchmark != TEMPLATE_ITEM_BENCHMARK:
            raise ValueError(
                f"item {item.id!r} has benchmark {item.benchmark!r}; TemplateRenderer INJECTS a "
                f"cell into a template and accepts only {TEMPLATE_ITEM_BENCHMARK!r} items. An "
                "item whose cell must be DERIVED belongs to ExtractiveRenderer (P4b)."
            )
        if item.task_type is not TaskType.OPEN_GEN:
            raise ValueError(
                f"item {item.id!r} has task_type {item.task_type.value!r}; P4a renders "
                f"{TaskType.OPEN_GEN.value!r} only. The paired-task prompt representation is open "
                "decision 7 and is deliberately not implemented here."
            )
        if "template_text" not in item.payload:
            raise ValueError(f"template item {item.id!r} has no 'template_text' in its payload")

    def _frame_of(self, template: Template, key: str) -> str:
        """The template's static text, with sentinels standing in for the injected values.

        Rendering with sentinels rather than parsing the source means the check sees exactly what
        a real render produces, including anything a Jinja expression contributes.
        """
        frame = normalize_template_text(
            template.render(
                {
                    DESCRIPTOR_VARIABLE: _DESCRIPTOR_SENTINEL,
                    CONJOINED_VARIABLE: _DESCRIPTOR_SENTINEL,
                    CARRIER_VARIABLE: _CARRIER_SENTINEL,
                    ARTICLE_VARIABLE: _ARTICLE_SENTINEL,
                    IDENTITY_VARIABLE: (
                        f"{_ARTICLE_SENTINEL} {_DESCRIPTOR_SENTINEL} {_CARRIER_SENTINEL}"
                    ),
                }
            )
        )

        leaked = matched_terms(frame, self._identity_vocabulary)
        if leaked:
            raise ValueError(
                f"template {key!r} names identity term(s) {list(leaked)} in its frame. Identity "
                "cues must appear ONLY in the injected slot -- a frame-level cue applies to every "
                "cell and contaminates the interaction it does not apply to equally."
            )
        axis_leaked = matched_terms(frame, self._axis_vocabulary)
        if axis_leaked:
            raise ValueError(
                f"template {key!r} names identity AXIS word(s) {list(axis_leaked)} in its frame. "
                "Naming the axis primes it for every cell on that axis."
            )
        if frame.startswith(_SENTINELS):
            raise ValueError(
                f"template {key!r} places the identity phrase sentence-initially. That would "
                "require capitalising it, and upstream orthography is load-bearing here "
                "('Black', 'Deaf'): a capitalisation pass would silently rewrite pinned P3 "
                "descriptors. Rephrase the frame instead."
            )
        return frame

    # ---------------------------------------------------------------- the Protocol

    def render(self, items: Sequence[Item], cells: Sequence[IdentityCell]) -> list[Prompt]:
        """Render ``items`` x ``cells`` x carriers into prompts (the ``PromptGenerator`` Protocol).

        Output order is templates (as given) x cells (as given) x carriers (as declared). Input
        order is preserved rather than re-sorted so the lattice's canonical cell order — order-1
        cells first, then declared pairs by rank — survives into the prompt file. Ids are
        content-derived and therefore identical regardless of the order the inputs arrived in.

        Raises:
            ValueError: On a non-``open_gen`` item, an unknown cell, a leaked or missing identity
                term, or a duplicate prompt id.
        """
        prepared = self.prepare(items)
        config = self._config
        prompts: list[Prompt] = []

        for template in prepared:
            for cell in cells:
                descriptor = self._descriptor_for(cell)
                conjoined = self._conjoined_descriptor_for(cell)
                article = config.article_for(descriptor)
                expected = self._expected_terms(cell)
                for carrier in config.carriers:
                    rendered = normalize_template_text(
                        template.template.render(
                            {
                                DESCRIPTOR_VARIABLE: descriptor,
                                CONJOINED_VARIABLE: conjoined,
                                CARRIER_VARIABLE: carrier.noun,
                                ARTICLE_VARIABLE: article,
                                IDENTITY_VARIABLE: f"{article} {descriptor} {carrier.noun}",
                            }
                        )
                    )
                    self._check_rendered(rendered, template.key, cell, expected)
                    prompts.append(
                        Prompt(
                            prompt_id=make_prompt_id(
                                template_key=template.key,
                                cell_id=cell.cell_id,
                                carrier=carrier.noun,
                                template_version=config.template_version,
                                item_id=template.item.id,
                                decoding_key=config.decoding_key,
                                rendered_text=rendered,
                            ),
                            item_id=template.item.id,
                            cell_id=cell.cell_id,
                            rendered_text=rendered,
                            task_type=TaskType.OPEN_GEN,
                            template_version=config.template_version,
                            decoding_key=config.decoding_key,
                        )
                    )

        self._check_unique(prompts)
        return prompts

    # ---------------------------------------------------------------- per-prompt checks

    def _descriptor_for(self, cell: IdentityCell) -> str:
        try:
            return self._lattice.compose_descriptor(cell)
        except KeyError as error:
            raise ValueError(
                f"cell {cell.cell_id!r} uses an (axis, value) this lattice does not declare: "
                f"{error}. TemplateRenderer injects cells from ONE lattice; mixing lattices would "
                "break the provenance trail every prompt is supposed to carry."
            ) from error

    def _conjoined_descriptor_for(self, cell: IdentityCell) -> str:
        """The predicative form: constituents in ``compose_rank`` order, joined by "and".

        Reuses P3's canonical ordering and changes only the joiner, so the *ordering* construct
        choice is untouched — but the conjunction itself is a second, distinct construct choice
        (see :data:`CONJOINED_VARIABLE`). Order-1 cells are unaffected: one constituent, no joiner.
        """
        rank = self._lattice.config.axis_rank
        try:
            ordered = sorted(cell.axes, key=lambda pair: rank(pair[0]))
            return CONJUNCTION_JOINER.join(self._surfaces[pair] for pair in ordered)
        except KeyError as error:
            raise ValueError(
                f"cell {cell.cell_id!r} uses an (axis, value) this lattice does not declare: "
                f"{error}."
            ) from error

    def _expected_terms(self, cell: IdentityCell) -> tuple[str, ...]:
        """The identity terms a correctly rendered prompt for ``cell`` must contain, and no others.

        Both the surface phrase and the registry value are admissible, because a lattice may
        legitimately declare a surface that differs from its id token.
        """
        expected: set[str] = set()
        for axis, value in cell.axes:
            expected.add(value)
            expected.add(self._surfaces[(axis, value)])
        return tuple(sorted(expected))

    def _check_rendered(
        self, rendered: str, key: str, cell: IdentityCell, expected: Sequence[str]
    ) -> None:
        """The leakage guard, per prompt.

        Set equality, not containment, in both directions: extra terms mean identity leaked into
        the frame or across slots; missing terms mean the injection silently did not happen and
        the "cell" is not in the prompt at all.
        """
        found = set(matched_terms(rendered, self._identity_vocabulary))
        allowed = set(expected)
        leaked = sorted(found - allowed)
        if leaked:
            raise ValueError(
                f"template {key!r} rendered for cell {cell.cell_id!r} contains identity term(s) "
                f"{leaked} that the cell does not contribute. Identity cues must appear ONLY in "
                f"the injected slot. Rendered: {rendered!r}"
            )
        composed = {term for term in allowed if _term_pattern(term).search(rendered)}
        if not composed:
            raise ValueError(
                f"template {key!r} rendered for cell {cell.cell_id!r} contains none of its "
                f"identity terms {sorted(allowed)}; the cell was never injected. "
                f"Rendered: {rendered!r}"
            )
        missing = sorted(
            value
            for axis, value in cell.axes
            if not _term_pattern(self._surfaces[(axis, value)]).search(rendered)
            and not _term_pattern(value).search(rendered)
        )
        if missing:
            raise ValueError(
                f"template {key!r} rendered for cell {cell.cell_id!r} is missing constituent(s) "
                f"{missing}; an order-{cell.order} cell must contribute every one of its values. "
                f"Rendered: {rendered!r}"
            )

    @staticmethod
    def _check_unique(prompts: Sequence[Prompt]) -> None:
        seen: set[str] = set()
        for prompt in prompts:
            if prompt.prompt_id in seen:
                raise ValueError(
                    f"duplicate prompt_id {prompt.prompt_id!r}; ids must key the prompt set"
                )
            seen.add(prompt.prompt_id)
