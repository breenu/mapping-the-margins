"""Parsing the red LM's output into candidate frames (P8).

The red LM is asked to emit one frame per line in a fixed shape (``FRAME: ...``), and this module
is the strict reader of that shape. Strictness is the point: a permissive parser that "recovers"
a frame from a paragraph of commentary would silently turn the model's refusal, its preamble, or
its numbered list header into an audit prompt. Anything that does not match is **rejected with a
reason**, never repaired.

**Pure.** No I/O, no RNG, no clock, no order dependence — the same discipline
:mod:`margins.benchmarks.screening` is held to, and for the same reason: a screen that is not a
function of its input cannot be re-checked by a reviewer.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from margins.schemas import Response

__all__ = [
    "FRAME_PREFIX",
    "Candidate",
    "parse_responses",
    "parse_text",
]

#: The line prefix the red LM is instructed to use. Uppercase and colon-terminated so it cannot be
#: confused with prose, and matched case-insensitively so a model that lowercases it still parses.
FRAME_PREFIX: Final[str] = "FRAME:"

_FRAME_RE: Final[re.Pattern[str]] = re.compile(
    rf"^\s*(?:[-*\d.)\s]*)?{re.escape(FRAME_PREFIX)}\s*(?P<body>.+?)\s*$",
    re.IGNORECASE,
)

#: Surrounding quotes a model habitually adds around a requested string. Stripped symmetrically:
#: a frame differing from another only by quoting is the same frame, and leaving both in would put
#: two near-identical item clusters in every cell.
_QUOTE_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ('"', '"'),
    ("'", "'"),
    ("“", "”"),
    ("‘", "’"),  # noqa: RUF001 - likewise; escapes kept so the pair stays legible
)


@dataclass(frozen=True, slots=True)
class Candidate:
    """One parsed frame candidate, with the provenance of the generation that produced it.

    ``target_cell_id`` is the cell the red LM was *asked* about. It is retained for the on-target
    audit and as provenance — **not** as a restriction on where the frame is later rendered. See
    :mod:`margins.redteam.frames` for why rendering a frame only into its own target cell would
    confound the frame with the cell and fabricate an interaction.
    """

    text: str
    target_cell_id: str
    response_id: str
    prompt_id: str
    seed: int
    line_index: int
    also_targeted: tuple[str, ...] = ()
    """Other target cells whose elicitation produced this same frame.

    Empty until :func:`margins.redteam.filter.dedup` fills it, and filled there rather than
    counted later **because dedup is the only stage that can see it**: collapsing duplicates
    discards the losing candidates, so a frame proposed for twelve different targets and a frame
    proposed once are indistinguishable afterwards. That difference is a real signal — a frame
    many targets produce is a *generic* frame, which says how cell-targeted the elicitation
    actually was — so it is carried forward rather than thrown away.
    """

    @property
    def sort_key(self) -> tuple[str, str, int, int]:
        """Canonical ordering key: target cell, then prompt, then seed, then line.

        Every downstream stage sorts by this before doing anything order-sensitive (dedup keeps
        the first survivor, the audit sample walks strata in order), so the output never depends
        on the order responses happened to be read in.
        """
        return (self.target_cell_id, self.prompt_id, self.seed, self.line_index)


def _strip_quotes(text: str) -> str:
    for opening, closing in _QUOTE_PAIRS:
        if len(text) >= 2 and text.startswith(opening) and text.endswith(closing):
            return text[1:-1].strip()
    return text


def parse_text(text: str) -> list[tuple[int, str]]:
    """Extract ``(line_index, frame_text)`` pairs from one generation.

    Returns every line matching the frame shape, in file order. A generation containing no such
    line yields an empty list, and the caller records that as an ``unparseable`` rejection — a
    generation that produced nothing is a fact worth counting, not a silent zero.
    """
    found: list[tuple[int, str]] = []
    for index, line in enumerate(text.splitlines()):
        match = _FRAME_RE.match(line)
        if match is None:
            continue
        body = _strip_quotes(" ".join(match.group("body").split()))
        if body:
            found.append((index, body))
    return found


def parse_responses(
    responses: Sequence[Response], *, target_cell_by_prompt: dict[str, str]
) -> tuple[list[Candidate], list[tuple[str, str]]]:
    """Parse every generation into candidates.

    Args:
        responses: the red LM's generations, as written by the P5 harness.
        target_cell_by_prompt: ``prompt_id -> cell_id``, from the elicitation prompt file. The
            target cell lives on the *prompt*, not on the response, so a response whose prompt is
            unknown cannot be attributed to a target and is refused rather than guessed at.

    Returns:
        ``(candidates, unparseable)`` where ``unparseable`` is ``(response_id, reason)`` for every
        generation that yielded no frame at all. Sorted canonically, so the result is a function of
        the input set and not of its order.
    """
    missing = sorted({r.prompt_id for r in responses if r.prompt_id not in target_cell_by_prompt})
    if missing:
        raise KeyError(
            f"{len(missing)} response(s) reference elicitation prompts that were not supplied, "
            f"e.g. {missing[:3]}. The target cell lives on the prompt, so a candidate cannot be "
            "attributed to a target without it."
        )

    candidates: list[Candidate] = []
    unparseable: list[tuple[str, str]] = []
    for response in responses:
        parsed = parse_text(response.text)
        if not parsed:
            unparseable.append((response.response_id, "unparseable"))
            continue
        for line_index, body in parsed:
            candidates.append(
                Candidate(
                    text=body,
                    target_cell_id=target_cell_by_prompt[response.prompt_id],
                    response_id=response.response_id,
                    prompt_id=response.prompt_id,
                    seed=response.seed,
                    line_index=line_index,
                )
            )

    candidates.sort(key=lambda candidate: candidate.sort_key)
    unparseable.sort()
    return candidates, unparseable
