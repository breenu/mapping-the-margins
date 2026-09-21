"""Figure provenance, and the audit that checks it against the canvas (P10).

:class:`FigureSpec` is what a figure module returns: the rendered figure, the claim it carries,
the panels whose data may be audited, the ledger of traced values that went into it, and the
encodings it used. :func:`audit_figure` then reads the numbers back off the *rendered artists* and
reports any that the ledger does not account for.

**Why audit the artists rather than trust the builder.** A ledger on its own proves the figure
*asked* for traced values; it does not prove those are the values it drew. Walking the canvas
closes that gap, and it is the only form of the check that catches the realistic failure -- a
rounded number typed into an annotation, or a threshold written as a literal into a reference
line. The audit is deliberately hard to satisfy by accident: ``tests/test_viz_traceability.py``
draws an unbacked value and asserts the audit reports it.

**Structural values.** A few numbers on a canvas are not measurements: the origin a bar starts
from, the unity line of a ratio plot. Those are declared per figure as
``structural={value: "why this needs no backing"}`` -- a mapping rather than a set, because an
exemption that does not say why is how an audit becomes a formality. :data:`COMMON_STRUCTURAL` is
the short list every figure inherits.

**Redundant encoding.** :class:`Encoding` pairs a colour with a marker, hatch or linestyle, and
``tests/test_viz_palette.py`` asserts the non-colour channel alone distinguishes every series in
every figure. That is what makes the figures greyscale-legible; the palette alone cannot, and
:mod:`margins.viz.palette` says so with the luminance numbers.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal

import numpy as np

from margins.report.source import Traced
from margins.report.tables import DataStatus
from margins.viz.palette import MUTED

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "COMMON_STRUCTURAL",
    "DECORATION",
    "AuditedPanel",
    "Encoding",
    "FigureSpec",
    "Unbacked",
    "audit_figure",
    "decorate",
    "stamp_data_status",
]

#: ``gid`` marking an artist as furniture rather than data: a shaded band, a bracket, a leader
#: line. Furniture carries no number and is skipped by the audit.
DECORATION: Final[str] = "margins-decoration"

#: Numbers every figure may draw without a ledger entry, each with the reason it is not a
#: measurement. Figures extend this; they do not replace it.
COMMON_STRUCTURAL: Final[Mapping[float, str]] = {
    0.0: "the origin, and the additive null the interaction is measured against",
    1.0: "unity: a ratio of one, or the top of a probability axis",
    2.0: "the lattice's fixed intersection depth (invariant 6), as in 'order-2'",
    95.0: "the nominal confidence level, written as a percentage in prose",
}

ValueAxis = Literal["x", "y", "both", "array"]

#: A sign only counts as a sign when nothing word-like precedes it, so ``order-2`` reads as the
#: number 2 rather than as -2. Without the lookbehind every hyphenated word in a figure would
#: enter the audit as a negative quantity nobody drew.
_NUMBER_RE: Final[re.Pattern[str]] = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?%?")


@dataclass(frozen=True)
class Encoding:
    """One series' visual encoding. Colour is never the only channel.

    Exactly one of ``marker``, ``hatch`` or ``linestyle`` must differ between any two series of a
    figure -- that is what the greyscale test checks, and it is why this is a record rather than
    a colour string.
    """

    label: str
    colour: str
    marker: str | None = None
    hatch: str | None = None
    linestyle: str | None = None
    #: Filled or open. A secondary channel: it doubles a marker set but cannot carry meaning on
    #: its own, so it does not satisfy the "not colour alone" requirement by itself.
    filled: bool | None = None

    @property
    def non_colour(self) -> tuple[str | None, str | None, str | None, bool | None]:
        """The channels that survive a greyscale print."""
        return (self.marker, self.hatch, self.linestyle, self.filled)

    def __post_init__(self) -> None:
        if all(channel is None for channel in (self.marker, self.hatch, self.linestyle)):
            raise ValueError(
                f"encoding {self.label!r} carries meaning by colour alone. Give it a marker, a "
                "hatch or a linestyle: the palette's closest two colours are 0.011 apart in "
                "greyscale luminance and are indistinguishable in print "
                "(margins.viz.palette)."
            )


@dataclass(frozen=True)
class AuditedPanel:
    """One axes whose drawn numbers must appear in the ledger.

    ``value_axis`` says which coordinate carries measurements: ``"x"`` for a horizontal forest
    plot, ``"y"`` for vertical bars, ``"array"`` for a heatmap, whose cell values are the data
    and whose coordinates are category indices.
    """

    ax: Axes
    value_axis: ValueAxis = "y"
    #: Skip the axis titles and labels. For a panel whose title is pure prose this changes
    #: nothing; it exists so a panel can opt out when a label legitimately carries an id.
    audit_text: bool = True


@dataclass(frozen=True)
class Unbacked:
    """A number found on the canvas that no ledger entry or structural exemption accounts for."""

    value: float
    where: str

    def __str__(self) -> str:
        return f"{self.value!r} drawn at {self.where}"


@dataclass
class FigureSpec:
    """A finished figure and everything needed to check it."""

    name: str
    title: str
    #: The one claim this figure carries. A figure whose claim cannot be written in a sentence
    #: is a generic plot and does not belong in the paper.
    claim: str
    caption: str
    figure: Figure
    panels: tuple[AuditedPanel, ...]
    ledger: tuple[Traced, ...]
    encodings: tuple[Encoding, ...] = ()
    data_status: DataStatus = "real"
    structural: Mapping[float, str] = field(default_factory=lambda: dict(COMMON_STRUCTURAL))

    @property
    def backed_values(self) -> tuple[float, ...]:
        return tuple(traced.value for traced in self.ledger)


def decorate(artist: Artist | Sequence[Artist]) -> None:
    """Mark furniture so the audit skips it. Furniture may carry no number."""
    for item in artist if isinstance(artist, list | tuple) else [artist]:
        item.set_gid(DECORATION)  # type: ignore[union-attr]


def stamp_data_status(figure: Figure, status: DataStatus) -> None:
    """Put a visible banner on a figure drawn from synthetic rows.

    The DATA NOTE for P10 is that the confirmatory run has not happened, so several figures are
    built against fixture rows to prove they render and to freeze their code. A reader must not
    have to know that from a commit message.
    """
    if status == "real":
        return
    text = figure.text(
        0.5,
        0.005,
        "FIXTURE DATA — synthetic rows so the figure renders and is tested; "
        "the confirmatory run (P11) drops real numbers through this same code.",
        ha="center",
        va="bottom",
        fontsize=6.0,
        color=MUTED,
    )
    decorate(text)


# --------------------------------------------------------------------------------------
# the audit
# --------------------------------------------------------------------------------------


def _finite(values: object) -> list[float]:
    array = np.asarray(values, dtype=float).ravel()
    return [float(value) for value in array if np.isfinite(value)]


def _collection_values(collection: Artist, axis: Literal["x", "y"]) -> list[float]:
    index = 0 if axis == "x" else 1
    out: list[float] = []
    offsets = getattr(collection, "get_offsets", None)
    if callable(offsets):
        data = np.asarray(offsets())
        if data.ndim == 2 and data.shape[1] == 2 and data.size:
            out += _finite(data[:, index])
    segments = getattr(collection, "get_segments", None)
    if callable(segments):
        for segment in segments():
            data = np.asarray(segment)
            if data.ndim == 2 and data.shape[1] == 2 and data.size:
                out += _finite(data[:, index])
    if not out:
        paths = getattr(collection, "get_paths", None)
        if callable(paths):
            for path in paths():
                vertices = np.asarray(path.vertices)
                if vertices.ndim == 2 and vertices.shape[1] == 2 and vertices.size:
                    out += _finite(vertices[:, index])
    return out


def _panel_values(panel: AuditedPanel) -> list[tuple[tuple[float, ...], str]]:
    """Every number the panel draws, as *alternative readings* of one drawn thing.

    A plotted coordinate has exactly one reading. A printed ``"34%"`` has two -- ``34`` and
    ``0.34`` -- and the audit passes if either is backed, because both are honest renderings of
    the same measurement and the figures use both.
    """
    ax = panel.ax
    label = ax.get_label() or ax.get_title() or "panel"
    found: list[tuple[tuple[float, ...], str]] = []

    def add(values: Sequence[float], where: str) -> None:
        found.extend(((value,), f"{label}:{where}") for value in values)

    if panel.value_axis == "array":
        for image in list(ax.images) + list(ax.collections):
            if image.get_gid() == DECORATION:
                continue
            array = getattr(image, "get_array", None)
            if callable(array):
                data = array()
                if data is not None:
                    add(_finite(np.ma.filled(np.ma.asarray(data), np.nan)), "array")
    else:
        axes: tuple[Literal["x", "y"], ...] = (
            ("x", "y") if panel.value_axis == "both" else (panel.value_axis,)
        )
        for which in axes:
            for line in ax.lines:
                if line.get_gid() == DECORATION:
                    continue
                data = line.get_xdata() if which == "x" else line.get_ydata()
                add(_finite(data), f"line-{which}")
            for patch in ax.patches:
                if patch.get_gid() == DECORATION:
                    continue
                extent = getattr(patch, "get_width" if which == "x" else "get_height", None)
                if callable(extent):
                    add(_finite([extent()]), f"bar-{which}")
            for collection in ax.collections:
                if collection.get_gid() == DECORATION:
                    continue
                add(_collection_values(collection, which), f"collection-{which}")

    if panel.audit_text:
        prose = [ax.get_title(), ax.get_xlabel(), ax.get_ylabel()]
        for text in ax.texts:
            if text.get_gid() != DECORATION:
                prose.append(text.get_text())
        for item in prose:
            for readings, where in _numbers_in(item):
                found.append((readings, f"{label}:text:{where}"))
    return found


def _numbers_in(text: str) -> list[tuple[tuple[float, ...], str]]:
    """Every number a string asserts, each with its alternative readings."""
    out: list[tuple[tuple[float, ...], str]] = []
    for match in _NUMBER_RE.finditer(text):
        literal = match.group(0)
        percent = literal.endswith("%")
        try:
            value = float(literal.rstrip("%"))
        except ValueError:  # pragma: no cover - the regex cannot produce this
            continue
        readings = (value, value / 100.0) if percent else (value,)
        out.append((readings, literal))
    return out


def _tolerance(value: float) -> float:
    """How close a drawn number must be to a ledger value to count as the same number.

    Generous for an annotation (which is rounded for print) and tight for a plotted coordinate
    (which is the raw float). One rule covers both: half of the last printed digit, floored at a
    value that only forgives binary rounding.
    """
    magnitude = abs(value)
    return max(1e-9, 5e-4 * max(magnitude, 1.0))


def audit_figure(spec: FigureSpec) -> tuple[Unbacked, ...]:
    """Every number on the canvas that the ledger does not account for.

    An empty result is the passing state. The figure tests assert it for every built figure.
    """
    backed = sorted(spec.backed_values)
    exempt = sorted(spec.structural)

    def accounted(value: float) -> bool:
        tolerance = _tolerance(value)
        for candidates in (backed, exempt):
            if any(abs(value - candidate) <= tolerance for candidate in candidates):
                return True
        # A printed annotation is rounded, so also accept a ledger value that rounds to it.
        return any(
            abs(value - round(candidate, decimals)) <= 1e-9
            for candidate in backed
            for decimals in range(0, 5)
        )

    return tuple(
        Unbacked(readings[0], where)
        for panel in spec.panels
        for readings, where in _panel_values(panel)
        if not any(accounted(reading) for reading in readings)
    )
