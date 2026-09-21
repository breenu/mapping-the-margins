"""Shared drawing helpers: labels, grouped positions, encodings, notes (P10).

Nothing here decides anything. It exists so that six figure modules format an axis pair the same
way, place a grouped forest plot the same way, and mark furniture as furniture -- because a figure
set that is inconsistent in those things reads as six figures rather than as one instrument.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

from margins.report.source import Traced
from margins.viz.palette import CATEGORICAL, MUTED, RULE
from margins.viz.trace import Encoding, decorate

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes

__all__ = [
    "DESCRIPTIVE_SUFFIX",
    "add_note",
    "axis_label",
    "encoding_for_harm",
    "grouped_positions",
    "pair_label",
    "term_label",
    "zero_rule",
]

#: Appended to any series the drift rule downgraded, in the legend itself. A reader should not
#: need the caption to learn that a series carries no claim.
DESCRIPTIVE_SUFFIX: Final[str] = " — DESCRIPTIVE-ONLY"

_AXIS_LABELS: Final[Mapping[str, str]] = {
    "socioeconomic_class": "class",
    "race_ethnicity": "race",
    "gender_and_sex": "gender",
    "ability": "ability",
    "age": "age",
}


def axis_label(axis: str) -> str:
    """The short axis name used on figures. The full name stays in the tables."""
    return _AXIS_LABELS.get(axis, axis.replace("_", " "))


def pair_label(axis_a: str, axis_b: str) -> str:
    return f"{axis_label(axis_a)} × {axis_label(axis_b)}"


def term_label(value_a: str, value_b: str) -> str:
    """One interaction cell, as a reader reads it: ``wealthy × white``."""
    return f"{value_a} × {value_b}"


def grouped_positions(
    groups: Sequence[str], *, gap: float = 1.0
) -> tuple[dict[int, float], dict[str, tuple[float, float]]]:
    """Positions for items laid out in labelled groups, top to bottom.

    Returns ``(position_by_index, span_by_group)``, where a span is the first and last position
    of a group -- which is what a group label and a separating rule need.
    """
    positions: dict[int, float] = {}
    spans: dict[str, tuple[float, float]] = {}
    cursor = 0.0
    previous: str | None = None
    first_in_group = 0.0
    for index, group in enumerate(groups):
        if previous is not None and group != previous:
            spans[previous] = (first_in_group, cursor - 1.0)
            cursor += gap
            first_in_group = cursor
        elif previous is None:
            first_in_group = cursor
        positions[index] = cursor
        previous = group
        cursor += 1.0
    if previous is not None:
        spans[previous] = (first_in_group, cursor - 1.0)
    return positions, spans


#: Harm types get the two leading palette entries, which are also the two that separate best in
#: greyscale. The marker is the channel that survives a monochrome print.
_HARM_ENCODING: Final[Mapping[str, tuple[str, str]]] = {
    "regard": (CATEGORICAL[0], "o"),
    "toxicity": (CATEGORICAL[1], "s"),
    "stereotype_preference": (CATEGORICAL[2], "D"),
}


def encoding_for_harm(harm_type: str, *, descriptive_only: bool) -> Encoding:
    """The encoding for one harm type, muted and relabelled if the drift rule downgraded it."""
    colour, marker = _HARM_ENCODING.get(harm_type, (CATEGORICAL[3], "^"))
    label = harm_type.replace("_", " ")
    if descriptive_only:
        return Encoding(
            label=label + DESCRIPTIVE_SUFFIX, colour=MUTED, marker=marker, linestyle="--"
        )
    return Encoding(label=label, colour=colour, marker=marker, linestyle="-")


def zero_rule(ax: Axes, *, vertical: bool = True) -> None:
    """The additive null. Drawn, because the estimand is a departure from it."""
    line = (
        ax.axvline(0.0, color=RULE, linewidth=0.7, zorder=0)
        if vertical
        else ax.axhline(0.0, color=RULE, linewidth=0.7, zorder=0)
    )
    decorate(line)


def add_note(ax: Axes, text: str, *, y: float = -0.14) -> None:
    """A caption-in-the-figure.

    Deliberately NOT marked as decoration: a note is where a hand-typed number would most
    plausibly appear, so it stays inside the traceability audit. Any figure that wants to write a
    number into a note has to obtain it from :class:`~margins.report.source.NumberSource` first.
    """
    note = ax.text(
        0.0,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.0,
        color=MUTED,
        linespacing=1.5,
    )
    note.set_gid(None)


def traced_pairs(values: Sequence[Traced]) -> list[float]:
    """Unwrap traced values for matplotlib."""
    return [item.value for item in values]
