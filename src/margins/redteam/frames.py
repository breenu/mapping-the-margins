"""Generated frames as :class:`~margins.schemas.Item` records, and the leakage-guard trial (P8).

This is where a candidate stops being text the red LM wrote and becomes a **frame**: a
provenance-carrying ``Item`` whose ``Provenance`` names the pinned generator, its resolved
revision, and the red-team version — exactly what PROJECT_CONTEXT.md section 3 anticipated when it
said "a red-team seed is an ``Item`` (``benchmark="redteam"``, provenance -> the generator model,
revision, and seed)".

---

**THE RIGOR POINT, and it decides the whole design.**

A frame elicited for one target cell is rendered into **every** target cell, not only its own. The
target survives as provenance and as the unit of the on-target audit, and nothing else.

If frames were rendered only into the cell they were elicited for, then "frame targeted at cell X"
would be perfectly confounded with "cell X": every cell would be measured through its own private
set of frames, and *any* difference between frames — length, register, topic, how aggressive the
red LM happened to be that seed — would land on that cell's mean and be read as an identity
effect. The estimator would report an interaction that the frame distribution, not the model,
produced. Rendering the shared pool into every cell makes frames an item factor crossed with the
cell factor, which is precisely what the cluster-robust design assumes.

It is also what makes red-team effects *estimable at all*. A frame present in every cell of a
pair's grid is one item cluster spanning that grid; a per-cell prompt is a singleton. The latter is
the structure that made CrowS-Pairs not estimable (``docs/methods.md`` section 2.5), and it would
have made red-teaming not estimable for the same reason.

---

**The leakage guard is authoritative here.** :func:`trial_render` puts every surviving frame
through the unchanged :class:`~margins.prompts.render.TemplateRenderer` across every target cell.
A frame that raises is dropped with reason ``leakage_guard``; a frame that reaches the artifact has
already rendered cleanly into every cell it will ever be used in. There is no path by which a
red-team prompt reaches a target model without having passed the same check a static prompt passes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, TypedDict

from jinja2 import TemplateError

from margins.inference.models import ModelSpec
from margins.lattice.build import DeclaredLattice
from margins.prompts.render import IDENTITY_VARIABLE, TemplateRenderer, template_item_id
from margins.redteam.config import REDTEAM_BENCHMARK, RedTeamConfig, RedTeamPromptSetConfig
from margins.redteam.filter import Rejection
from margins.redteam.parse import Candidate
from margins.schemas import IdentityCell, Item, Prompt, TaskType, register_payload_type

__all__ = [
    "FRAME_ID_DIGEST_LENGTH",
    "Frame",
    "RedTeamFramePayload",
    "frame_id",
    "frame_items",
    "render_audit_prompts",
    "to_template_text",
    "trial_render",
]

#: Frames are content-addressed, like every other id in this pipeline.
FRAME_ID_DIGEST_LENGTH: Final[int] = 12


class RedTeamFramePayload(TypedDict):
    """Payload of a generated-frame :class:`~margins.schemas.Item` (``benchmark="redteam"``).

    Mirrors ``TemplatePayload`` and adds the generation provenance a static frame does not have.
    Registered in ``PAYLOAD_TYPES`` at import, so mypy is not blind on the one field carrying the
    real data — the same discipline the CrowS-Pairs loader follows.
    """

    template_text: str
    slots: tuple[str, ...]
    target_cell_id: str
    n_targets: int
    elicitation_template: str
    generator_model_id: str
    generator_revision: str
    seed: int


register_payload_type(REDTEAM_BENCHMARK, RedTeamFramePayload)


@dataclass(frozen=True, slots=True)
class Frame:
    """One surviving generated frame, ready to become an ``Item``."""

    frame_id: str
    text: str
    template_text: str
    target_cell_id: str
    target_cell_ids: tuple[str, ...]
    elicitation_template: str
    response_id: str
    seed: int

    @property
    def n_targets(self) -> int:
        """How many distinct target cells produced this frame.

        An honest audit signal rather than bookkeeping: a frame that many different targets
        produced is a *generic* frame, which says something about how cell-targeted the elicitation
        actually was. It is reported in the coverage table.
        """
        return len(self.target_cell_ids)


def frame_id(template_text: str, *, redteam_version: str) -> str:
    """``f<digest12>`` over the frame's rendered template text and the red-team version.

    Content-addressed, so the same frame text always gets the same id no matter which target cell
    or which seed produced it — which is what lets a frame proposed for several cells collapse to
    one item rather than several identical ones. The version is in the preimage so a red-team set
    bump produces different ids rather than silently reusing the old ones.
    """
    preimage = f"{redteam_version}\x1f{template_text}"
    digest = hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:FRAME_ID_DIGEST_LENGTH]
    return f"f{digest}"


def to_template_text(text: str, *, slot_token: str) -> str:
    """Replace the red LM's literal placeholder with the Jinja identity variable.

    The red LM writes ``[PERSON]`` because a mid-size instruct model emits a bracketed placeholder
    far more reliably than it emits Jinja syntax. The translation happens once, here, after the
    filter has already confirmed the slot occurs exactly once — so this cannot produce a frame with
    two slots or none.

    ``identity`` is the variable used rather than ``descriptor``, because it composes
    ``article + descriptor + carrier`` and is what makes the rendered prompt grammatical for every
    cell without the frame having to supply an article it cannot know.
    """
    return text.replace(slot_token, f"{{{{ {IDENTITY_VARIABLE} }}}}")


def trial_render(
    candidates: Sequence[Candidate],
    *,
    config: RedTeamConfig,
    prompt_set: RedTeamPromptSetConfig,
    lattice: DeclaredLattice,
    cells: Sequence[IdentityCell],
    generator: ModelSpec,
) -> tuple[list[Frame], list[Rejection]]:
    """Turn surviving candidates into frames, collapsing by content and enforcing the guard.

    Every frame is rendered into **every** cell in ``cells`` through the unchanged
    ``TemplateRenderer``. A frame that raises anywhere is rejected with reason ``leakage_guard``
    and never reaches the artifact — so the guarantee is not "we filtered for leakage" but "every
    emitted frame has already rendered cleanly into every cell it will be used in".

    Returns:
        ``(frames, rejections)``, frames sorted by ``frame_id`` so the artifact is byte-stable.
    """
    slot = config.elicitation.slot_token
    renderer = TemplateRenderer(prompt_set, lattice)

    # Collapse by content first: the same frame proposed for several targets is ONE frame.
    by_id: dict[str, Frame] = {}
    targets: dict[str, list[str]] = {}
    for candidate in sorted(candidates, key=lambda c: c.sort_key):
        template_text = to_template_text(candidate.text, slot_token=slot)
        identifier = frame_id(template_text, redteam_version=config.redteam_version)
        # `also_targeted` is the dedup group's other targets, carried forward by the filter
        # precisely because collapsing duplicates would otherwise erase it.
        targets.setdefault(identifier, []).append(candidate.target_cell_id)
        targets[identifier].extend(candidate.also_targeted)
        if identifier in by_id:
            continue
        by_id[identifier] = Frame(
            frame_id=identifier,
            text=candidate.text,
            template_text=template_text,
            target_cell_id=candidate.target_cell_id,
            target_cell_ids=(),
            elicitation_template=_elicitation_key(candidate.prompt_id),
            response_id=candidate.response_id,
            seed=candidate.seed,
        )

    frames: list[Frame] = []
    rejections: list[Rejection] = []
    for identifier in sorted(by_id):
        frame = by_id[identifier]
        resolved = Frame(
            frame_id=frame.frame_id,
            text=frame.text,
            template_text=frame.template_text,
            target_cell_id=frame.target_cell_id,
            target_cell_ids=tuple(sorted(set(targets[identifier]))),
            elicitation_template=frame.elicitation_template,
            response_id=frame.response_id,
            seed=frame.seed,
        )
        item = _frame_item(resolved, config=config, generator=generator)
        try:
            renderer.render([item], cells)
        except (ValueError, TemplateError) as error:
            # TemplateError as well as ValueError, because a frame IS compiled as a Jinja
            # template: an unbalanced delimiter raises TemplateSyntaxError and an unknown
            # variable raises UndefinedError, neither of which is a ValueError. The filter's
            # `template_syntax` rule catches these earlier, but this is the backstop -- one
            # malformed candidate must cost its own frame, never the whole run.
            rejections.append(
                Rejection(
                    text=resolved.text,
                    target_cell_id=resolved.target_cell_id,
                    response_id=resolved.response_id,
                    reason="leakage_guard",
                    detail=str(error).replace("\n", " ")[:300],
                )
            )
            continue
        frames.append(resolved)

    rejections.sort(key=lambda r: (r.reason, r.target_cell_id, r.text))
    return frames, rejections


def _elicitation_key(prompt_id: str) -> str:
    """The elicitation template key from an elicitation prompt id (``<template>::<cell>::<d>``)."""
    return prompt_id.split("::", 1)[0]


def _frame_item(frame: Frame, *, config: RedTeamConfig, generator: ModelSpec) -> Item:
    return Item(
        id=template_item_id(frame.frame_id, benchmark=REDTEAM_BENCHMARK),
        benchmark=REDTEAM_BENCHMARK,
        task_type=TaskType.OPEN_GEN,
        payload={
            "template_text": frame.template_text,
            "slots": (IDENTITY_VARIABLE,),
            "target_cell_id": frame.target_cell_id,
            "n_targets": frame.n_targets,
            "elicitation_template": frame.elicitation_template,
            "generator_model_id": generator.model_id,
            "generator_revision": generator.revision,
            "seed": frame.seed,
        },
        target_axes=(),
        gold=None,
        provenance=config.provenance_for(
            hashlib.sha256(frame.template_text.encode("utf-8")).hexdigest(),
            model_id=generator.model_id,
            revision=generator.revision,
        ),
    )


def frame_items(
    frames: Sequence[Frame], *, config: RedTeamConfig, generator: ModelSpec
) -> list[Item]:
    """The provenance-carrying ``Item`` for each frame, in frame-id order."""
    return [_frame_item(frame, config=config, generator=generator) for frame in frames]


def render_audit_prompts(
    frames: Sequence[Frame],
    *,
    config: RedTeamConfig,
    prompt_set: RedTeamPromptSetConfig,
    lattice: DeclaredLattice,
    cells: Sequence[IdentityCell],
    generator: ModelSpec,
) -> list[Prompt]:
    """Render every frame into every cell — the audit prompt set the target model sees.

    Straight through the unchanged ``TemplateRenderer``: same leakage guard, same article rule,
    same ``compose_rank`` order, same ``prompt_id`` construction. The **only** thing that
    distinguishes these from static prompts downstream is the ``redteam`` benchmark label they
    carry through ``Item.benchmark``, which is exactly the tag the E3 comparison reads.
    """
    items = frame_items(frames, config=config, generator=generator)
    return TemplateRenderer(prompt_set, lattice).render(items, cells)
