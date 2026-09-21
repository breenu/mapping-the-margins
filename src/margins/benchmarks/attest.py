"""Deriving an identity cell FROM a source item, never injecting one into it (P2).

This is the counterpart to P4a's leakage guard, and it runs in the opposite direction. The
templated path *injects* a cell into a frame and then checks that exactly the injected terms came
back out. The extractive path has no such freedom: a CrowS-Pairs sentence is about whatever it is
about, and the only honest cell is the one its own text attests. ``interfaces.PromptGenerator``
states the rule plainly -- "you cannot assign 'Black woman' to a CrowS-Pairs item about religion" --
and gate P4b is exactly that rule made executable: *the renderer emits no cell unattested in its
source item*.

**Attestation is literal.** A value is attested when its lattice surface form occurs in the
sentence, matched case-insensitively on word boundaries. There is no alias table and no inference:
CrowS row 1 contrasts "The poor" with "The rich", and since neither is a lattice value that row
attests nothing on the class axis and is logged as such. Mapping it to ``working-class`` would be a
construct claim *we* invented, and the entire point of resolving the lattice's values from
HolisticBias at a pinned commit is that we do not invent them.

**Order-2 needs a second axis that is genuinely shared.** The contrasted attribute supplies one
axis. A second can only come from an identity term that appears **unchanged in both sentences** --
if it appeared in only one, it would be a second contrast and the pair would not be minimal, so the
likelihood difference could not be attributed to the intended attribute. This is why the extractive
paradigm's order-2 yield is low, and why that yield is MEASURED here rather than assumed.

Case-insensitive matching deserves a note, because the lattice pins orthography: its values are
``Black`` and ``Deaf``, and CrowS writes "black kids" in lower case. Matching case-sensitively would
score the dataset's typography rather than its content. The lattice's own capitalisation is what is
recorded on the emitted cell; the sentence's is not rewritten.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Sequence
from typing import Final, NamedTuple

from margins.lattice import DeclaredLattice
from margins.schemas import IdentityCell

__all__ = [
    "AttestedValue",
    "CellIndex",
    "DiffSummary",
    "SpanChange",
    "SurfaceIndex",
    "attested_values",
    "build_cell_index",
    "build_surface_index",
    "declared_cell",
    "diff_summary",
    "tokenize",
    "values_in_tokens",
]

#: Word-boundary matching that treats a hyphen as part of the word, so ``middle-class`` does not
#: match inside ``middle-classes`` and ``male`` does not match inside ``female``. Same pattern the
#: P4a leakage guard uses, for the same reason.
_BOUNDARY: Final[str] = r"(?<![\w-]){term}(?![\w-])"

#: Words and standalone punctuation. Punctuation is kept as its own token so that a pair differing
#: only in a comma is visible as a differing span rather than silently folded into a word.
_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[\w'-]+|[^\w\s]")


class AttestedValue(NamedTuple):
    """One ``(axis, value)`` a sentence attests, with the surface that attested it."""

    axis: str
    value: str
    surface: str


class SurfaceIndex(NamedTuple):
    """Compiled surface patterns for every lattice value, built once per run."""

    entries: tuple[tuple[AttestedValue, re.Pattern[str]], ...]

    @property
    def axes(self) -> tuple[str, ...]:
        return tuple(sorted({entry.axis for entry, _ in self.entries}))


def build_surface_index(lattice: DeclaredLattice) -> SurfaceIndex:
    """Compile a matcher for every declared lattice value.

    Built from ``lattice.values()`` rather than from any list of our own, so a value that is not in
    the lattice cannot be attested and therefore cannot reach a cell. Longer surfaces are compiled
    first so that, where two values would both match, the more specific one is reported -- the
    ordering is deterministic (length, then the value itself) rather than dictionary order.
    """
    values = sorted(lattice.values(), key=lambda v: (-len(v.surface), v.axis, v.value))
    entries = tuple(
        (
            AttestedValue(axis=value.axis, value=value.value, surface=value.surface),
            re.compile(_BOUNDARY.format(term=re.escape(value.surface)), re.IGNORECASE),
        )
        for value in values
    )
    return SurfaceIndex(entries=entries)


def attested_values(text: str, index: SurfaceIndex) -> tuple[AttestedValue, ...]:
    """Every lattice value whose surface occurs in ``text``, in deterministic order."""
    return tuple(entry for entry, pattern in index.entries if pattern.search(text))


def tokenize(text: str) -> tuple[str, ...]:
    """Words and punctuation, lower-cased, with sentence-final punctuation dropped.

    Three normalizations, each of which removes a *typographic* difference that would otherwise be
    counted as a difference in content:

    * **Lower-casing** stops a sentence-initial capital from registering as a differing token when
      the pair is otherwise identical.
    * **Dropping trailing punctuation** matters more than it sounds. CrowS-Pairs is
      crowd-authored, and many pairs differ in whether the writer typed a final full stop --
      row 22's "My mom spent all day cooking for Thanksgiving" against "My dad ... Thanksgiving."
      is one contrast plus one keystroke. Counting the keystroke would exclude the row as
      non-minimal for a reason that has nothing to do with what it measures.
    * Both affect only the *diff*. Nothing recorded on an ``Item`` is rewritten: the payload keeps
      the upstream sentences verbatim, and the model is scored on them verbatim.
    """
    tokens = [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]
    while tokens and not any(char.isalnum() for char in tokens[-1]):
        tokens.pop()
    return tuple(tokens)


class SpanChange(NamedTuple):
    """One contiguous region where the two sentences differ."""

    stereo: tuple[str, ...]
    antistereo: tuple[str, ...]

    @property
    def width(self) -> int:
        """Tokens replaced, on whichever side replaced more."""
        return max(len(self.stereo), len(self.antistereo))

    def is_agreement_only(self, agreement: frozenset[str]) -> bool:
        """True when every token on both sides is a closed-class agreement token.

        Such a span is **obligatory grammatical concord**, not an independent second contrast:
        swapping ``woman`` for ``man`` forces ``she`` to become ``he``, and English gives the
        writer no choice about it. Counting it as a second differing region would label grammar as
        a construct flaw. A span mixing agreement with content (``her mother`` / ``his father``) is
        not agreement-only and still counts.
        """
        tokens = set(self.stereo) | set(self.antistereo)
        return bool(tokens) and tokens <= agreement


class DiffSummary(NamedTuple):
    """How two sentences differ, in token terms.

    Carries every differing region rather than a count, because the minimality test needs to ask
    *which* regions are substantive -- see :meth:`SpanChange.is_agreement_only`.
    """

    changes: tuple[SpanChange, ...]
    identical: bool

    def content_changes(self, agreement: frozenset[str]) -> tuple[SpanChange, ...]:
        """The differing regions that are not pure grammatical concord.

        Falls back to **all** regions when every region is agreement-only: a pair whose entire
        difference is function words still differs, and reporting zero differing regions for it
        would be false. In practice such a pair attests no lattice value and is kept without a
        cell.
        """
        content = tuple(c for c in self.changes if not c.is_agreement_only(agreement))
        return content or self.changes

    @staticmethod
    def spans(changes: tuple[SpanChange, ...]) -> int:
        return len(changes)

    @staticmethod
    def width(changes: tuple[SpanChange, ...]) -> int:
        return max((c.width for c in changes), default=0)

    @staticmethod
    def stereo_tokens(changes: tuple[SpanChange, ...]) -> tuple[str, ...]:
        return tuple(token for change in changes for token in change.stereo)


def diff_summary(stereo: str, antistereo: str) -> DiffSummary:
    """Summarise the difference between the two members of a pair.

    Uses :class:`difflib.SequenceMatcher` over token sequences rather than a character edit
    distance: the screen's question is "how many contiguous regions differ, and how wide are they",
    which is about alignment, and an edit distance answers a different question.
    """
    left = tokenize(stereo)
    right = tokenize(antistereo)
    if left == right:
        return DiffSummary(changes=(), identical=True)

    matcher = difflib.SequenceMatcher(a=left, b=right, autojunk=False)
    changes = tuple(
        SpanChange(stereo=left[i1:i2], antistereo=right[j1:j2])
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    )
    return DiffSummary(changes=changes, identical=False)


def values_in_tokens(
    tokens: Sequence[str], index: SurfaceIndex, *, axis: str | None = None
) -> tuple[AttestedValue, ...]:
    """Lattice values attested by a bag of tokens, optionally restricted to one axis.

    Used on the *differing* tokens, to answer "is the contrast itself about the mapped axis?". The
    tokens are re-joined with spaces before matching so a multi-word surface can still be found.
    """
    haystack = " ".join(tokens)
    found = attested_values(haystack, index)
    if axis is None:
        return found
    return tuple(entry for entry in found if entry.axis == axis)


#: ``frozenset of (axis, value) -> the declared cell``. Built once; see :func:`build_cell_index`.
CellIndex = dict[frozenset[tuple[str, str]], IdentityCell]


def build_cell_index(lattice: DeclaredLattice) -> CellIndex:
    """Index the lattice's own cells by their ``(axis, value)`` content."""
    return {frozenset(cell.axes): cell for cell in lattice.cells()}


def declared_cell(index: CellIndex, pairs: Sequence[tuple[str, str]]) -> IdentityCell | None:
    """The declared :class:`IdentityCell` for these ``(axis, value)`` pairs, or None.

    **Looked up, never constructed.** Every cell this can return came out of ``lattice.cells()``,
    so an emitted cell is by definition one the lattice declares -- including the anti-cartesian
    guarantee that only seven of the ten possible axis pairs have order-2 cells at all. Two
    attested axes whose pair is not declared therefore yield None here, and the caller falls back
    to the order-1 cell rather than minting a cell the lattice does not contain.
    """
    return index.get(frozenset(pairs))
