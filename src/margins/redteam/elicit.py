"""Elicitation: the prompts that address the **red LM** (P8).

These are *not* audit prompts and must never be confused with them. An elicitation prompt
deliberately **names** the target identity — "write a prompt that would expose disparate treatment
of a working-class Black person" — because naming it is exactly what makes the red-teaming
*cell-targeted*, which is the novelty. It is sent to the red LM, its output is parsed into frames,
and it is never scored and never shown to a target model.

So the P4a leakage guard is **not** applied here, and that is correct rather than a loophole: the
guard exists to keep identity signal out of everything except the manipulated slot of an *audit*
prompt. Applying it to an instruction whose entire job is to name an identity would be a category
error. The guard applies with full force one stage later, in
:mod:`margins.redteam.frames`, where the generated frames are rendered — and a frame that names an
identity term is rejected there.

**What makes this cheap.** An elicitation prompt is an ordinary
:class:`~margins.schemas.Prompt`, so red-team generation is a run of the *existing* P5 harness:
resumable, content-addressed, manifest-writing, seeded per batch. Nothing in P5 needed changing.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Final

from jinja2 import Environment, StrictUndefined

from margins.lattice.build import DeclaredLattice
from margins.prompts.config import normalize_template_text
from margins.redteam.config import (
    REDTEAM_ELICITATION_BENCHMARK,
    ElicitationFile,
    RedTeamConfig,
    RedTeamPromptSetConfig,
)
from margins.schemas import IdentityCell, Item, Prompt, TaskType

__all__ = [
    "ELICITATION_VARIABLES",
    "ElicitationPlan",
    "elicitation_items",
    "elicitation_prompts",
    "elicitation_seeds",
    "make_elicitation_prompt_id",
    "target_cells",
]

#: Variables an elicitation template may reference. ``target`` is the composed descriptor phrase
#: with its article and carrier ("a working-class Black person"); ``axes`` names the two axes in
#: prose ("socioeconomic class and race / ethnicity"); ``slot`` is the literal placeholder token
#: the red LM must write; ``examples`` is the rendered few-shot block.
ELICITATION_VARIABLES: Final[tuple[str, ...]] = ("target", "axes", "slot", "examples")

_PROMPT_ID_SEPARATOR: Final[str] = "::"
_DIGEST_SEPARATOR: Final[str] = "\x1f"
PROMPT_ID_DIGEST_LENGTH: Final[int] = 12


def _environment() -> Environment:
    """Strict, plain-text Jinja. Same settings as P4a's, for the same reasons."""
    return Environment(undefined=StrictUndefined, autoescape=False, keep_trailing_newline=False)


def target_cells(config: RedTeamConfig, lattice: DeclaredLattice) -> list[IdentityCell]:
    """The lattice cells the red LM is asked to attack, in the lattice's canonical order.

    **Confirmatory pairs only** — gate P8 is "prompts restricted to the confirmatory axis pairs",
    and :class:`~margins.redteam.config.TargetSettings` types that as a ``Literal`` so a config
    could not widen it. Order-1 cells are excluded by default: the estimand is the interaction,
    which lives on order-2 cells, so an order-1 target spends generation budget on what B1 already
    covers. They are still *rendered* — every frame renders into every cell — this only governs
    what the red LM is asked about.
    """
    confirmatory = frozenset(lattice.config.confirmatory_pairs)
    cells: list[IdentityCell] = []
    for cell in lattice.cells():
        if cell.order == 1:
            if config.target.include_order_1:
                cells.append(cell)
            continue
        axes = tuple(axis for axis, _ in cell.axes)
        if axes in confirmatory or tuple(reversed(axes)) in confirmatory:
            cells.append(cell)
    if not cells:
        raise ValueError(
            "no target cells: the lattice declares no confirmatory order-2 pair. Red-teaming is "
            "restricted to the confirmatory pairs (gate P8), so there is nothing to target."
        )
    return cells


def elicitation_items(
    config: RedTeamConfig, files: Sequence[ElicitationFile], *, config_sha256: str
) -> list[Item]:
    """Turn elicitation template files into provenance-carrying :class:`~margins.schemas.Item`s.

    Same move as P4a's ``template_items``: every prompt traces to an ``Item`` carrying
    :class:`~margins.schemas.Provenance`, so ``Prompt.item_id`` stays required. The provenance here
    names *this repository* — the elicitation instructions are first-party source material, unlike
    the frames the red LM writes, whose provenance names the model.
    """
    environment = _environment()
    items: list[Item] = []
    for file in files:
        from jinja2 import meta

        declared = meta.find_undeclared_variables(environment.parse(file.text))
        unknown = sorted(declared - set(ELICITATION_VARIABLES))
        if unknown:
            raise ValueError(
                f"elicitation template {file.id!r} references unknown variables {unknown}; it may "
                f"use only {sorted(ELICITATION_VARIABLES)}"
            )
        if "target" not in declared:
            raise ValueError(
                f"elicitation template {file.id!r} never references `target`, so it asks the red "
                "LM the same question for every cell. That is not cell-targeted red-teaming -- it "
                "is one generic prompt set wearing 92 different labels."
            )
        if "slot" not in declared:
            raise ValueError(
                f"elicitation template {file.id!r} never references `slot`, so the red LM is "
                "never told which placeholder to write and every candidate will be rejected as "
                "having no identity slot."
            )
        items.append(
            Item(
                id=f"{REDTEAM_ELICITATION_BENCHMARK}/{file.id}",
                benchmark=REDTEAM_ELICITATION_BENCHMARK,
                task_type=TaskType.OPEN_GEN,
                payload={
                    "template_text": file.text,
                    "slots": tuple(v for v in ELICITATION_VARIABLES if v in declared),
                },
                target_axes=(),
                gold=None,
                provenance=config.elicitation_provenance(config_sha256),
            )
        )
    return items


def make_elicitation_prompt_id(
    *,
    template_key: str,
    cell_id: str,
    redteam_version: str,
    item_id: str,
    decoding_key: str,
    rendered_text: str,
) -> str:
    """``<template>::<cell_id>::<digest12>``.

    Three fields rather than P4a's four: an elicitation prompt has no carrier, because it is not
    rendered from a lattice descriptor into a carrier phrase — it *describes* a target cell to the
    red LM. The digest covers the rendered text, so an edit to an elicitation template that forgets
    to bump ``redteam_version`` shows up as changed ids in a diff rather than as silent drift.
    """
    preimage = _DIGEST_SEPARATOR.join(
        (redteam_version, item_id, cell_id, decoding_key, rendered_text)
    )
    fields = (template_key, cell_id)
    offenders = [f for f in fields if _PROMPT_ID_SEPARATOR in f]
    if offenders:
        raise ValueError(
            f"prompt id fields may not contain {_PROMPT_ID_SEPARATOR!r}, which joins them: "
            f"{offenders}"
        )
    digest = hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:PROMPT_ID_DIGEST_LENGTH]
    return _PROMPT_ID_SEPARATOR.join((*fields, digest))


def elicitation_seeds(config: RedTeamConfig) -> tuple[int, ...]:
    """The distinct seeds one elicitation prompt is run at, one per candidate.

    Distinct candidates come from distinct **seeds**, never from repeated calls at one seed: P5's
    backend re-seeds per batch, so two calls at one seed return identical text under different
    ids, and ``RunHarness.plan`` rejects a repeated seed outright.
    """
    base = config.elicitation.seed_base
    return tuple(base + index for index in range(config.elicitation.candidates_per_target))


class ElicitationPlan:
    """Renders elicitation prompts for a red-team config. Deterministic and I/O-free."""

    def __init__(
        self,
        config: RedTeamConfig,
        lattice: DeclaredLattice,
        prompt_set: RedTeamPromptSetConfig,
    ) -> None:
        """
        Args:
            config: the validated red-team declaration.
            lattice: supplies the target cells and composes their descriptors.
            prompt_set: the AUDIT rendering rules, whose carrier and article rule are used to
                describe the target to the red LM. Passing the audit rules here is deliberate: the
                red LM must be told about the identity in exactly the words the audit prompts will
                use, or it targets a phrasing the instrument never produces.
        """
        self._config = config
        self._lattice = lattice
        self._prompt_set = prompt_set
        self._environment = _environment()
        self._axis_labels = {name: axis.label for name, axis in lattice.config.axes.items()}

    def axes_phrase(self, cell: IdentityCell) -> str:
        """The cell's axes named in prose, for the elicitation instruction."""
        return " and ".join(
            self._axis_labels.get(axis, axis.replace("_", " ")) for axis, _ in cell.axes
        )

    def target_phrase(self, cell: IdentityCell) -> str:
        """The target identity as a noun phrase: "a working-class Black person".

        Composed through the **lattice's own** ``compose_descriptor`` and the base set's article
        rule and carrier, so the red LM is told about the identity in exactly the words the audit
        prompts will use. Telling it about "Black working class people" while auditing "a
        working-class Black person" would target a phrasing the instrument never produces.
        """
        descriptor = self._lattice.compose_descriptor(cell)
        article = self._prompt_set.article_for(descriptor)
        carrier = self._prompt_set.carriers[0].noun
        return f"{article} {descriptor} {carrier}"

    def render(self, items: Sequence[Item], cells: Sequence[IdentityCell]) -> list[Prompt]:
        """One elicitation prompt per (template x target cell).

        Output order is templates (as given) x cells (as given). Ids are content-derived, so they
        do not depend on the order the inputs arrived in.
        """
        config = self._config
        examples = "\n".join(f"- {example}" for example in config.elicitation.few_shot)
        prompts: list[Prompt] = []
        seen: set[str] = set()

        for item in items:
            if item.benchmark != REDTEAM_ELICITATION_BENCHMARK:
                raise ValueError(
                    f"item {item.id!r} has benchmark {item.benchmark!r}; elicitation renders "
                    f"{REDTEAM_ELICITATION_BENCHMARK!r} items only. An AUDIT prompt is rendered "
                    "by TemplateRenderer, under the leakage guard."
                )
            key = item.id.removeprefix(f"{REDTEAM_ELICITATION_BENCHMARK}/")
            template = self._environment.from_string(str(item.payload["template_text"]))
            for cell in cells:
                rendered = normalize_template_text(
                    template.render(
                        {
                            "target": self.target_phrase(cell),
                            "axes": self.axes_phrase(cell),
                            "slot": config.elicitation.slot_token,
                            "examples": examples,
                        }
                    )
                )
                prompt_id = make_elicitation_prompt_id(
                    template_key=key,
                    cell_id=cell.cell_id,
                    redteam_version=config.redteam_version,
                    item_id=item.id,
                    decoding_key=config.generator.decoding_name,
                    rendered_text=rendered,
                )
                if prompt_id in seen:
                    raise ValueError(f"duplicate elicitation prompt_id {prompt_id!r}")
                seen.add(prompt_id)
                prompts.append(
                    Prompt(
                        prompt_id=prompt_id,
                        item_id=item.id,
                        cell_id=cell.cell_id,
                        rendered_text=rendered,
                        task_type=TaskType.OPEN_GEN,
                        template_version=config.redteam_version,
                        # The GENERATOR's profile: this prompt is run through the red LM.
                        decoding_key=config.generator.decoding_name,
                    )
                )
        return prompts


def elicitation_prompts(
    config: RedTeamConfig,
    lattice: DeclaredLattice,
    files: Sequence[ElicitationFile],
    prompt_set: RedTeamPromptSetConfig,
    *,
    config_sha256: str,
) -> tuple[list[Item], list[Prompt]]:
    """Build the elicitation items and prompts for a red-team config."""
    items = elicitation_items(config, files, config_sha256=config_sha256)
    cells = target_cells(config, lattice)
    return items, ElicitationPlan(config, lattice, prompt_set).render(items, cells)
