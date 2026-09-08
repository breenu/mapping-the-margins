"""Audit artifacts for the rendered prompt set (P4a): the phrase review and the slot audit.

Two gate items live here, and neither is decoration.

**The phrase review** (``phrase_audit.md``) shows all 92 order-2 composed descriptors **inside a
rendered template sentence**, not as bare phrases. The distinction matters: a phrase list cannot
show that ``"a older blind person"`` is broken, because the article belongs to the frame, not to
the descriptor. Composition is a construct choice (P3), and the specific risk it carries is that
**the composition implies something neither constituent did** — juxtaposing two descriptors can
activate a stereotype that is a property of the pair rather than of either term. That is not a
formatting problem; it is a threat to the interaction estimate, because the interaction is exactly
what the pair contributes over and above its parts.

**Descriptors are pinned P3 provenance and are NEVER edited to fix phrasing.** When a phrase reads
badly the levers are, in order: the template frame (P4a, cheap), the compose rule / ``compose_rank``
(P3 config, changes cell ids and requires re-export), or the value selection (a pre-registration
amendment with a logged rationale). Rewriting a sourced term would turn a verifiable descriptor
into a fabricated one and defeat ``scripts/build_lattice.py --verify``.

**What the mechanical flags can and cannot do.** :func:`mechanical_flags` catches only what is
decidable from the strings: a repeated morpheme, two stacked hyphenated compounds, one surface
nested inside another. Idiomatic adjective order, register, and pair-specific implicature are
**not** mechanically decidable, so they are carried as :data:`PHRASE_REVIEW_NOTES` — findings from
the P4a phrase review, each naming its lever and each left **OPEN pending a human decision**. They
live in code rather than in a chat message so the review is a durable, versioned artifact and the
counts stay accurate if the lattice changes.

**The slot audit** (``template_fill_audit.md``) is gate P4a's "20-prompt manual slot audit". The
sample is deterministic and stratified across every declared axis pair and every template, because
a 20-row sample drawn from the head of an 880-row file would audit one template and one pair.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import ceil
from typing import Final

from margins.lattice.build import CellRecord, DeclaredLattice
from margins.prompts.config import PromptSetConfig
from margins.schemas import Prompt

__all__ = [
    "FILL_AUDIT_SAMPLE_SIZE",
    "ORDER_1_GROUP",
    "PHRASE_REVIEW_NOTES",
    "PhraseReviewNote",
    "audit_group",
    "cell_surfaces",
    "mechanical_flags",
    "render_phrase_audit",
    "sample_for_fill_audit",
]

#: Gate P4a asks for a 20-prompt manual slot audit.
FILL_AUDIT_SAMPLE_SIZE: Final[int] = 20

#: Group label for order-1 cells in the stratified sample and the phrase audit.
ORDER_1_GROUP: Final[str] = "order_1"

_TOKEN_SPLIT: Final[re.Pattern[str]] = re.compile(r"[\s\-/]+")


def cell_surfaces(lattice: DeclaredLattice, record: CellRecord) -> tuple[str, ...]:
    """The constituent surface phrases of a cell, in ``compose_rank`` order."""
    config = lattice.config
    surfaces = {(value.axis, value.value): value.surface for value in lattice.values()}
    ordered = sorted(record.cell.axes, key=lambda pair: config.axis_rank(pair[0]))
    return tuple(surfaces[pair] for pair in ordered)


def _tokens(surface: str) -> frozenset[str]:
    return frozenset(token for token in _TOKEN_SPLIT.split(surface.lower()) if token)


def mechanical_flags(surfaces: Sequence[str]) -> tuple[str, ...]:
    """String-decidable defects in a composed descriptor.

    Deliberately conservative. Everything here is checkable without judgement; everything
    requiring judgement is a :class:`PhraseReviewNote` instead. A low-noise flag column is what
    makes the 92-row table readable enough to actually review.
    """
    if len(surfaces) < 2:
        return ()

    flags: list[str] = []
    token_sets = [_tokens(surface) for surface in surfaces]

    shared = token_sets[0].intersection(*token_sets[1:])
    if shared:
        flags.append(f"repeated-token:{'/'.join(sorted(shared))}")

    if all("-" in surface for surface in surfaces):
        flags.append("stacked-hyphenated-compounds")

    lowered = [surface.lower() for surface in surfaces]
    for index, surface in enumerate(lowered):
        others = lowered[:index] + lowered[index + 1 :]
        if any(surface != other and surface in other for other in others):
            flags.append(f"substring-overlap:{surface}")

    return tuple(flags)


# --------------------------------------------------------------------------------------
# The P4a phrase review — findings that no mechanical check can reach
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PhraseReviewNote:
    """One finding from the P4a phrase review, with the lever that would fix it.

    ``applies`` is a predicate rather than a hard-coded cell list so the affected counts stay
    correct as the lattice changes — which makes it a **regression guard** as well as a report.
    A note recorded as RESOLVED whose predicate still matches cells is a contradiction the tests
    catch: ``tests/test_prompt_export.py`` asserts that every resolved note affects zero cells, so
    reverting a compose_rank would fail the suite rather than quietly re-introduce the phrasing.

    Notes are never deleted once acted on. The review is provenance: a reader should be able to
    see what was found, what was decided, and what was deliberately left alone.
    """

    id: str
    severity: str
    finding: str
    why_it_matters: str
    proposed_lever: str
    applies: Callable[[DeclaredLattice, CellRecord, tuple[str, ...]], bool]
    status: str
    resolution: str


def _is_gender_anchored(lattice: DeclaredLattice, axis: str) -> bool:
    spec = lattice.config.axes.get(axis)
    return spec is not None and spec.gender_anchored


def _touches_gender(lattice: DeclaredLattice, record: CellRecord, _flags: tuple[str, ...]) -> bool:
    return any(_is_gender_anchored(lattice, axis) for axis, _ in record.cell.axes)


def _gender_composed_first(
    lattice: DeclaredLattice, record: CellRecord, _flags: tuple[str, ...]
) -> bool:
    if record.axis_pair is None:
        return False
    first, second = record.axis_pair
    return _is_gender_anchored(lattice, first) and not _is_gender_anchored(lattice, second)


def _has_repeated_token(
    _lattice: DeclaredLattice, _record: CellRecord, flags: tuple[str, ...]
) -> bool:
    return any(flag.startswith("repeated-token") for flag in flags)


def _age_before_class(
    lattice: DeclaredLattice, record: CellRecord, _flags: tuple[str, ...]
) -> bool:
    if record.axis_pair is None:
        return False
    return record.axis_pair == ("age", "socioeconomic_class") and "age" in lattice.config.axes


#: The P4a phrase review, with the human's decisions of 8 Sep 2026 recorded against each finding.
#: Notes are kept after resolution: the review is provenance, and each predicate doubles as a
#: regression guard (a RESOLVED note must match zero cells -- see tests/test_prompt_export.py).
PHRASE_REVIEW_NOTES: Final[tuple[PhraseReviewNote, ...]] = (
    PhraseReviewNote(
        id="marked-register-with-neutral-carrier",
        severity="high (construct validity)",
        status="MITIGATED — carrier-free frame added; artifact reduced, not eliminated",
        finding=(
            "Gender descriptors are adjectival upstream ('female', 'male'), so composing them "
            "with the neutral carrier produces 'a female person' / 'a male person'. In English "
            "that is a MARKED clinical/forensic register, unlike 'a woman' or 'a man'."
        ),
        why_it_matters=(
            "Register is not identity. If a scorer or the model reacts to the marked register "
            "rather than to the identity, gender-anchored cells carry a systematic offset that "
            "non-gender cells do not — and because the offset lands on some pairs and not others, "
            "it can appear as an INTERACTION, which is the estimand itself. This is the same "
            "failure mode as scorer identity drift (PROJECT_CONTEXT.md section 9), arriving "
            "through the prompt rather than through the scorer."
        ),
        proposed_lever=(
            "Template frame, NOT the descriptors: add a frame that needs no carrier noun at all."
        ),
        resolution=(
            "APPROVED and applied (prompt set v1.1): the carrier-free frame `someone_who_is` was "
            "added, and it is the only frame whose gender cells avoid the marked register. TWO "
            "THINGS A READER MUST NOT MISS. (1) COVERAGE: it is one frame of nine, so the "
            "artifact is removed from roughly a ninth of renderings, not eliminated; the other "
            "eight frames still render 'a female person' on these cells, with the section 9 "
            "drift control as backstop. (2) The frame required a second construct choice, because "
            "English predicative position rejects juxtaposed adjectives — 'someone who is older "
            "blind' is broken where 'an older blind person' is fine — so this frame joins "
            "constituents with 'and'. A conjunction asserts two co-predicated properties where "
            "juxtaposition can read as one compound category, so this frame encodes a slightly "
            "different relation between the two identities than the other eight. R1's "
            "prompt-paraphrase analysis should report it separately. This note therefore stays "
            "MITIGATED rather than RESOLVED, and its cell count stays non-zero on purpose."
        ),
        applies=_touches_gender,
    ),
    PhraseReviewNote(
        id="gender-composed-before-non-gender",
        severity="moderate (fluency / construct)",
        status="RESOLVED — compose_rank reordered in lattice v1.1",
        finding=(
            "compose_rank v1.0 put the gender axis before race, yielding 'a female Black person' "
            "and 'a male Hispanic person'. Idiomatic English orders these the other way round, "
            "so the composed phrase read as non-native for this pair only."
        ),
        why_it_matters=(
            "Fluency is itself a property a language model reacts to. A fixed rank cannot "
            "manufacture a MAIN effect because it is constant across cells, but an ordering that "
            "is idiomatic for some pairs and stilted for exactly one pair contributes to that "
            "pair's INTERACTION — see the compose-rank caveat in margins.lattice.build."
        ),
        proposed_lever=(
            "Move the gender axis after the non-gender axes in compose_rank (lattice config)."
        ),
        resolution=(
            "APPROVED and applied. Lattice v1.1 reorders to class(10) < age(20) < ability(30) < "
            "race(40) < gender(50), giving 'a Black female person'. Applied together with "
            "`age-before-status-adjective` as one change, since both edit the same field. Values, "
            "pairs and sources are UNCHANGED; cell_ids for this pair are not, hence the version "
            "bump and the P3 re-export. NOTE the side effect: gender is now the constituent "
            "nearest the carrier noun, which makes the marked register above adjacent in every "
            "gender cell rather than only some."
        ),
        applies=_gender_composed_first,
    ),
    PhraseReviewNote(
        id="age-before-status-adjective",
        severity="low (fluency)",
        status="RESOLVED — compose_rank reordered in lattice v1.1",
        finding=(
            "compose_rank v1.0 put age before socioeconomic class, yielding 'a young wealthy "
            "person'. English adjective order prefers quality/status before age."
        ),
        why_it_matters=(
            "Milder than the gender/race case — both orders occur in natural English — but it is "
            "the same mechanism, and it landed on one pair only."
        ),
        proposed_lever="Order the class axis before age in compose_rank (lattice config).",
        resolution=(
            "APPROVED and applied as part of the same lattice v1.1 reordering, giving 'a wealthy "
            "young person'. The chosen total order is the unique one satisfying all seven "
            "declared pairs at once."
        ),
        applies=_age_before_class,
    ),
    PhraseReviewNote(
        id="repeated-morpheme",
        severity="low (cosmetic)",
        status="ACCEPTED — documented, deliberately not fixed",
        finding=(
            "'middle-class' composed with 'middle-aged' gives 'a middle-class middle-aged "
            "person' — a repeated morpheme that reads as a stutter. One cell of 92."
        ),
        why_it_matters=(
            "The repetition makes that single cell's prompts marginally less fluent than its "
            "neighbours in the same pair. It is a per-cell nuisance term, not a threat to the "
            "pair's estimate."
        ),
        proposed_lever=(
            "No frame-level lever exists — the repetition is in the composition, not the frame. "
            "The only alternatives were accepting it or amending the value selection."
        ),
        resolution=(
            "ACCEPTED as-is, deliberately. Amending the value selection was REJECTED: the "
            "descriptors are pinned P3 provenance verified against HolisticBias v1.1 at a pinned "
            "commit, both terms are the neutral-register choice their axis note argues for, and "
            "trading verifiable provenance for one cell's cosmetics is a bad exchange. This note "
            "stays in the audit precisely so the phrasing is on the record as a known, accepted "
            "cost rather than an oversight. Its cell count stays at 1 by design."
        ),
        applies=_has_repeated_token,
    ),
)


# --------------------------------------------------------------------------------------
# Grouping and sampling
# --------------------------------------------------------------------------------------


def audit_group(record: CellRecord) -> str:
    """Stratification key: ``order_1``, or the canonical ``axis_a|axis_b`` of an order-2 cell."""
    if record.axis_pair is None:
        return ORDER_1_GROUP
    return "|".join(record.axis_pair)


def _groups(lattice: DeclaredLattice) -> dict[str, str]:
    """``cell_id -> group``, in the lattice's canonical record order."""
    return {record.cell_id: audit_group(record) for record in lattice.records()}


def sample_for_fill_audit(
    prompts: Sequence[Prompt],
    lattice: DeclaredLattice,
    size: int = FILL_AUDIT_SAMPLE_SIZE,
) -> list[Prompt]:
    """A deterministic sample stratified across cell groups **and templates**.

    Both stratifications are load-bearing. Stratifying only by cell group audits every axis pair
    but can land on the same one or two frames every time — and a *slot* audit whose rows all come
    from one frame has not audited the frames. So the template is **rotated** by round and by
    group: with 8 groups and 8 frames, one pass already touches every frame.

    Within a (group, template) cell, positions are evenly spread so the sample spans the group's
    values rather than repeating its first. No RNG anywhere: the same prompt set always yields the
    same rows, which is what makes a manual audit re-checkable by someone else.
    """
    group_of = _groups(lattice)
    ordered_groups: list[str] = []
    members: dict[str, list[Prompt]] = {}
    for prompt in prompts:
        if prompt.cell_id is None:
            continue
        group = group_of.get(prompt.cell_id)
        if group is None:
            continue
        if group not in members:
            members[group] = []
            ordered_groups.append(group)
        members[group].append(prompt)

    if not ordered_groups:
        return []

    rounds = max(1, ceil(size / len(ordered_groups)))
    sample: list[Prompt] = []
    seen: set[str] = set()

    for index in range(rounds):
        for offset, group in enumerate(ordered_groups):
            group_prompts = members[group]
            templates = list(dict.fromkeys(prompt.item_id for prompt in group_prompts))
            template = templates[(index + offset) % len(templates)]
            candidates = [
                prompt for prompt in group_prompts if prompt.item_id == template
            ] or group_prompts
            chosen = candidates[(index * len(candidates)) // rounds]
            if chosen.prompt_id not in seen:
                seen.add(chosen.prompt_id)
                sample.append(chosen)

    return sample[:size]


# --------------------------------------------------------------------------------------
# The phrase audit document
# --------------------------------------------------------------------------------------


def _escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_phrase_audit(
    lattice: DeclaredLattice,
    config: PromptSetConfig,
    prompts: Sequence[Prompt],
    *,
    audit_item_id: str,
    carrier: str,
) -> str:
    """Render the phrase audit: every order-2 descriptor inside a real rendered sentence.

    Byte-deterministic — no timestamps and no unordered iteration — so a regenerated audit differs
    only when the lattice, the templates, or the compose rule changed.
    """
    by_cell = {
        prompt.cell_id: prompt
        for prompt in prompts
        if prompt.item_id == audit_item_id and prompt.cell_id is not None
    }
    order_2 = [record for record in lattice.records() if record.order == 2]

    rows: list[tuple[CellRecord, str, tuple[str, ...], str]] = []
    for record in order_2:
        surfaces = cell_surfaces(lattice, record)
        flags = mechanical_flags(surfaces)
        prompt = by_cell.get(record.cell_id)
        sentence = prompt.rendered_text if prompt is not None else "(not rendered)"
        rows.append((record, sentence, flags, record.descriptor))

    flagged = sum(1 for _, _, flags, _ in rows if flags)

    lines: list[str] = [
        "# Phrase audit — order-2 composed descriptors, in frame",
        "",
        f"Lattice version: `{lattice.config.lattice_version}`  ",
        f"Template version: `{config.template_version}`  ",
        f"Audit frame: `{audit_item_id}`  ",
        f"Carrier: `{carrier}`",
        "",
        "Generated by `scripts/build_prompts.py`. **Do not edit by hand.**",
        "",
        "## Why this exists",
        "",
        "Descriptor composition is a **construct choice, not a neutral operation**. Joining two",
        "sourced descriptors can produce a phrase that is ungrammatical, stilted, or — the case",
        "that actually threatens the estimand — one that implies something **neither constituent",
        "implied on its own**. Since the emergent-bias estimand *is* what a pair contributes over",
        "and above its parts, a composition artefact and a real interaction are not",
        "distinguishable after the fact. They have to be caught here, before any run.",
        "",
        "Phrases are shown **inside a rendered sentence**, not bare: the article and the carrier",
        "noun belong to the frame, so `a older blind person` is a defect a bare phrase list cannot",
        "show.",
        "",
        "## The one rule this audit does not break",
        "",
        "**Descriptors are pinned P3 provenance and are never edited to fix phrasing.** Every term",
        "here is verified against HolisticBias v1.1 at a pinned commit by",
        "`scripts/build_lattice.py --verify`; rewriting one to read better would turn a checkable",
        "descriptor into a fabricated one. The levers, cheapest first, are: the **template frame**",
        "(P4a), the **compose rule / `compose_rank`** (P3 config — changes cell ids, forces a",
        "re-export), and the **value selection** (a pre-registration amendment with a logged",
        "rationale).",
        "",
        "## What the mechanical flags cannot catch",
        "",
        "The `flags` column is limited to what is decidable from the strings themselves. It says",
        "nothing about idiomatic adjective order, register, or pair-specific implicature — those",
        "are judgement calls and are carried as review items below.",
        "",
        "Two things that look like defects and are **not**: `Black`, `Deaf`, `Asian` and",
        "`Hispanic` are capitalised mid-phrase because that is their upstream orthography (and,",
        "for `Deaf`, a community convention). A capitalisation pass over composed phrases would",
        "silently rewrite pinned descriptors and must not be added.",
        "",
        "## Open review items",
        "",
        f"{len(PHRASE_REVIEW_NOTES)} findings from the P4a phrase review, each with the decision",
        "taken on it. Notes are **kept after resolution** — the review is provenance, and a reader",
        "is entitled to see what was found, what was changed, and what was deliberately left",
        "alone. Each predicate also doubles as a **regression guard**: a RESOLVED note must match",
        "zero cells, and the test suite fails if a later edit re-introduces the phrasing.",
        "",
    ]

    for note in PHRASE_REVIEW_NOTES:
        affected = [record for record, _, flags, _ in rows if note.applies(lattice, record, flags)]
        lines.extend(
            [
                f"### `{note.id}` — {note.severity}",
                "",
                f"- **status:** {note.status}",
                f"- **order-2 cells still matching:** {len(affected)} of {len(rows)}",
                f"- **finding:** {note.finding}",
                f"- **why it matters:** {note.why_it_matters}",
                f"- **lever:** {note.proposed_lever}",
                f"- **decision:** {note.resolution}",
                "",
            ]
        )

    lines.extend(
        [
            "## Composed descriptors in frame",
            "",
            f"{len(rows)} order-2 cells; {flagged} carry a mechanical flag.",
            "",
        ]
    )

    grouped: dict[str, list[tuple[CellRecord, str, tuple[str, ...], str]]] = {}
    group_order: list[str] = []
    for row in rows:
        group = audit_group(row[0])
        if group not in grouped:
            grouped[group] = []
            group_order.append(group)
        grouped[group].append(row)

    for group in group_order:
        pair = tuple(group.split("|"))
        status = (
            "confirmatory" if grouped[group][0][0].confirmatory else "exploratory (no significance)"
        )
        coverage = (
            "non-gender x non-gender"
            if lattice.config.is_non_gender_pair((pair[0], pair[1]))
            else "gender-anchored"
        )
        lines.extend(
            [
                f"### `{pair[0]} x {pair[1]}` — {status} · {coverage}",
                "",
                "| descriptor | rendered sentence | flags |",
                "|---|---|---|",
            ]
        )
        for _record, sentence, flags, descriptor in grouped[group]:
            lines.append(
                f"| {_escape_cell(descriptor)} "
                f"| {_escape_cell(sentence)} "
                f"| {_escape_cell(', '.join(flags)) if flags else '—'} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"
