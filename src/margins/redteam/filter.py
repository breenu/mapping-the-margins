"""Quality control on generated frames: the degenerate filter and dedup (P8).

Gate P8 requires "dedup + filter logged". This module is that, and it is **pure** — no I/O, no
RNG, no clock, no order dependence — exactly as :mod:`margins.benchmarks.screening` is. Every
rejection carries one machine-readable reason, and every reason lands in
``rejected_frames.csv`` beside the candidate that earned it.

**Why each rule exists.** A filter whose rules are arbitrary is a filter that shapes the red-team
distribution in undocumented ways, and the distribution *is* the contribution:

* ``unparseable`` — the generation contained no frame line at all. Counted, not ignored.
* ``slot_count`` — a frame must contain the slot **exactly once**. Zero means the cell is never
  injected and the "prompt set" is one constant string. Two or more means the identity is stated
  twice, which doubles the identity signal relative to every static frame and makes the red-team
  arm non-comparable in E3.
* ``too_short`` / ``too_long`` — below the floor a frame is a stub with no task; above the ceiling
  it is a scenario, and the generated text becomes a response to the scenario rather than to the
  identity.
* ``names_identity_term`` / ``names_axis_word`` — the frame itself contains a lattice value or an
  axis word. These are **exactly** the P4a leakage-guard conditions, checked here so a red-team run
  reports "47 frames leaked identity" instead of dying on the first one.
* ``sentence_initial_slot`` — P4a refuses a sentence-initial identity phrase, because capitalising
  it would silently rewrite pinned orthography (``Black``, ``Deaf``).
* ``refusal`` — the red LM declined or editorialised. A refusal parsed as a frame becomes an audit
  prompt asking the target model about the red LM's safety policy.
* ``copies_fewshot`` — the candidate is one of the examples it was shown. A model echoing its
  prompt has generated nothing.
* ``duplicate`` / ``near_duplicate`` — see :func:`dedup`. This one is not hygiene: duplicate frames
  are duplicate **item clusters**, and a cluster-robust standard error computed over them is too
  small in the anti-conservative direction.

The **leakage guard remains authoritative**; this is a pre-screen. A frame surviving here is still
trial-rendered through the unchanged :class:`~margins.prompts.render.TemplateRenderer` in
:mod:`margins.redteam.frames`, and one that raises there is rejected with reason
``leakage_guard``. Nothing reaches the frame artifact that has not already rendered cleanly into
every target cell.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Final

from margins.lattice.build import DeclaredLattice
from margins.prompts.render import axis_vocabulary, identity_vocabulary, matched_terms
from margins.redteam.config import DedupSettings, ElicitationSettings, FilterSettings
from margins.redteam.parse import Candidate

__all__ = [
    "REJECTION_REASONS",
    "FilterOutcome",
    "Rejection",
    "canonical_form",
    "dedup",
    "screen_candidates",
]

#: Every reason this module can emit. A closed vocabulary so the report can enumerate them and a
#: test can assert none is invented ad hoc at a call site.
REJECTION_REASONS: Final[tuple[str, ...]] = (
    "unparseable",
    "slot_count",
    "too_short",
    "too_long",
    "names_identity_term",
    "names_axis_word",
    "sentence_initial_slot",
    "template_syntax",
    "refusal",
    "copies_fewshot",
    "duplicate",
    "near_duplicate",
    "leakage_guard",
)

_WORD_RE: Final[re.Pattern[str]] = re.compile(r"\S+")

#: Canonicalization sentinel. A control character, so no frame's own text can forge one.
_SLOT_SENTINEL: Final[str] = "\x01"

_PUNCTUATION_RE: Final[re.Pattern[str]] = re.compile(r"[^\w\s\x01]+")

#: Jinja delimiters. A frame IS a template, so a candidate containing one of these would be
#: compiled rather than read as text -- see the ``template_syntax`` rule.
_TEMPLATE_DELIMITERS: Final[tuple[str, ...]] = ("{{", "}}", "{%", "%}", "{#", "#}")


@dataclass(frozen=True, slots=True)
class Rejection:
    """One rejected candidate and why."""

    text: str
    target_cell_id: str
    response_id: str
    reason: str
    detail: str = ""

    def __post_init__(self) -> None:
        if self.reason not in REJECTION_REASONS:
            raise ValueError(
                f"unknown rejection reason {self.reason!r}; add it to REJECTION_REASONS so the "
                f"report can enumerate it. Known: {list(REJECTION_REASONS)}"
            )


@dataclass(frozen=True, slots=True)
class FilterOutcome:
    """What the screen kept and what it threw away."""

    kept: tuple[Candidate, ...]
    rejected: tuple[Rejection, ...]

    @property
    def counts_by_reason(self) -> dict[str, int]:
        counts: dict[str, int] = dict.fromkeys(REJECTION_REASONS, 0)
        for rejection in self.rejected:
            counts[rejection.reason] += 1
        return counts


def canonical_form(text: str, slot_token: str) -> str:
    """The comparison key for dedup: case-, spacing-, punctuation- and slot-insensitive.

    The slot collapses to a sentinel so two frames differing only in which placeholder spelling
    the model used are one frame. Punctuation is dropped for the same reason a trailing period is
    not a different prompt.
    """
    lowered = text.replace(slot_token, _SLOT_SENTINEL).lower()
    stripped = _PUNCTUATION_RE.sub(" ", lowered)
    return " ".join(stripped.split())


def _word_count(text: str, slot_token: str) -> int:
    """Words in a frame, counting the slot as one word.

    The slot stands for a descriptor phrase of one to three words depending on the cell, so
    counting its literal characters would make the length bound depend on the placeholder spelling
    rather than on the frame.
    """
    return len(_WORD_RE.findall(text.replace(slot_token, _SLOT_SENTINEL)))


def screen_candidates(
    candidates: Sequence[Candidate],
    *,
    lattice: DeclaredLattice,
    elicitation: ElicitationSettings,
    settings: FilterSettings,
    dedup_settings: DedupSettings,
) -> FilterOutcome:
    """Apply every degenerate rule, then dedup. Deterministic and order-independent.

    Candidates are sorted canonically first, so which member of a duplicate group survives is a
    property of the *content*, never of the order responses were read in.
    """
    slot = elicitation.slot_token
    identity_terms = identity_vocabulary(lattice)
    axis_words = axis_vocabulary(lattice)
    few_shot = {canonical_form(example, slot) for example in elicitation.few_shot}
    refusals = tuple(pattern.lower() for pattern in settings.refusal_patterns)

    ordered = sorted(candidates, key=lambda candidate: candidate.sort_key)
    survivors: list[Candidate] = []
    rejected: list[Rejection] = []

    def reject(candidate: Candidate, reason: str, detail: str = "") -> None:
        rejected.append(
            Rejection(
                text=candidate.text,
                target_cell_id=candidate.target_cell_id,
                response_id=candidate.response_id,
                reason=reason,
                detail=detail,
            )
        )

    for candidate in ordered:
        text = candidate.text

        occurrences = text.count(slot)
        if occurrences != 1:
            reject(candidate, "slot_count", f"{occurrences} occurrences of {slot!r}")
            continue

        words = _word_count(text, slot)
        if words < settings.min_words:
            reject(candidate, "too_short", f"{words} words < {settings.min_words}")
            continue
        if words > settings.max_words:
            reject(candidate, "too_long", f"{words} words > {settings.max_words}")
            continue

        lowered = text.lower()
        hit = next((pattern for pattern in refusals if pattern in lowered), None)
        if hit is not None:
            reject(candidate, "refusal", f"matched {hit!r}")
            continue

        # The frame with its slot removed is what P4a's guard sees as the static frame.
        frame_only = text.replace(slot, " ")
        leaked = matched_terms(frame_only, identity_terms)
        if leaked:
            reject(candidate, "names_identity_term", ", ".join(leaked))
            continue
        axis_leaked = matched_terms(frame_only, axis_words)
        if axis_leaked:
            reject(candidate, "names_axis_word", ", ".join(axis_leaked))
            continue

        if text.startswith(slot):
            reject(candidate, "sentence_initial_slot", "slot at position 0")
            continue

        delimiter = next((d for d in _TEMPLATE_DELIMITERS if d in text), None)
        if delimiter is not None:
            # A frame becomes a Jinja template, so a candidate containing Jinja delimiters would
            # be COMPILED rather than treated as text: `{{ descriptor }}` would inject a second
            # identity, and `{{ oops }}` would raise an undefined-variable error out of the
            # renderer. The red LM is shown `[PERSON]` and never asked for Jinja, so anything
            # with a delimiter in it is malformed output rather than a frame.
            reject(candidate, "template_syntax", f"contains {delimiter!r}")
            continue

        canonical = canonical_form(text, slot)
        if canonical in few_shot:
            reject(candidate, "copies_fewshot", "identical to a few-shot example")
            continue

        survivors.append(candidate)

    kept, duplicate_rejections = dedup(survivors, slot_token=slot, settings=dedup_settings)
    rejected.extend(duplicate_rejections)
    rejected.sort(key=lambda r: (r.reason, r.target_cell_id, r.text))
    return FilterOutcome(kept=tuple(kept), rejected=tuple(rejected))


def dedup(
    candidates: Sequence[Candidate], *, slot_token: str, settings: DedupSettings
) -> tuple[list[Candidate], list[Rejection]]:
    """Collapse exact and near-duplicate frames, keeping the canonically first of each group.

    Two stages. **Exact** dedup on :func:`canonical_form` is cheap and catches the common case: the
    same frame proposed for several target cells, which is expected and is itself a useful signal
    (a frame every cell produces is a generic frame). **Near-duplicate** removal then runs
    ``rapidfuzz.fuzz.token_sort_ratio`` against the survivors so far, greedily, over the canonical
    ordering.

    Greedy-over-a-canonical-order is what makes this deterministic: the survivor of a near-duplicate
    group is decided by content ordering, never by iteration order. A ratio of 100 disables the
    second stage and leaves exact dedup only.

    **Why this is not hygiene.** Near-duplicate frames become near-duplicate item clusters in every
    cell, and the estimator's cluster-robust standard error treats them as independent items. Two
    copies of one frame do not carry two frames' worth of information, so leaving them in shrinks
    every standard error in the anti-conservative direction.

    **What dedup must not throw away.** Collapsing a group discards the losing candidates, and with
    them the fact that *several different targets* proposed the same frame. That is a real signal
    about how cell-targeted the elicitation was, so the group's other target cells are carried onto
    the survivor as ``Candidate.also_targeted`` instead of being lost. Counting it afterwards is
    impossible: after the collapse, a frame twelve targets produced looks exactly like one produced
    once.
    """
    from rapidfuzz import fuzz

    ordered = sorted(candidates, key=lambda candidate: candidate.sort_key)
    kept: list[Candidate] = []
    kept_canonical: list[str] = []
    seen: dict[str, int] = {}
    group_targets: list[set[str]] = []
    rejections: list[Rejection] = []

    for candidate in ordered:
        canonical = canonical_form(candidate.text, slot_token)

        index = seen.get(canonical)
        if index is not None:
            group_targets[index].add(candidate.target_cell_id)
            rejections.append(
                Rejection(
                    text=candidate.text,
                    target_cell_id=candidate.target_cell_id,
                    response_id=candidate.response_id,
                    reason="duplicate",
                    detail=f"same frame as {kept[index].response_id}",
                )
            )
            continue

        if settings.near_duplicate_ratio < 100.0:
            match = next(
                (
                    position
                    for position, other_canonical in enumerate(kept_canonical)
                    if fuzz.token_sort_ratio(canonical, other_canonical)
                    >= settings.near_duplicate_ratio
                ),
                None,
            )
            if match is not None:
                group_targets[match].add(candidate.target_cell_id)
                rejections.append(
                    Rejection(
                        text=candidate.text,
                        target_cell_id=candidate.target_cell_id,
                        response_id=candidate.response_id,
                        reason="near_duplicate",
                        detail=(
                            f"within {settings.near_duplicate_ratio:g} of {kept[match].response_id}"
                        ),
                    )
                )
                continue

        seen[canonical] = len(kept)
        kept.append(candidate)
        kept_canonical.append(canonical)
        group_targets.append({candidate.target_cell_id})

    kept = [
        replace(candidate, also_targeted=tuple(sorted(targets - {candidate.target_cell_id})))
        for candidate, targets in zip(kept, group_targets, strict=True)
    ]

    return kept, rejections
