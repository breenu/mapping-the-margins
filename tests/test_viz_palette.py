"""Gate P10: the palette is colourblind-safe and no figure carries meaning by colour alone.

Two separate properties, and the second is the one that actually delivers greyscale legibility.

The palette check is arithmetic on the colours themselves: simulate protanopia, deuteranopia and
tritanopia and require every pair of categorical colours to stay perceptually apart. The
greyscale check is structural, because it has to be -- six hues chosen for hue separation are
*not* separated in luminance, and this file proves that with a number rather than asserting a
palette can do something it cannot. What makes a figure readable in monochrome is that its
non-colour channel (marker, hatch, linestyle) identifies every series on its own.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from margins.viz.palette import (
    CATEGORICAL,
    CVD_KINDS,
    greyscale,
    min_pairwise_delta_e,
    relative_luminance,
)
from margins.viz.trace import Encoding

#: A just-noticeable difference in CIE76 is about 2.3. The shipped palette's worst pair under any
#: simulated deficiency measured 16.2 on 21 Sep 2026; the gate sits below that with room for a
#: future palette edit, and far above "technically different".
MIN_DELTA_E = 15.0


def test_the_palette_survives_simulated_colour_vision_deficiency() -> None:
    for kind in (None, *CVD_KINDS):
        closest = min_pairwise_delta_e(CATEGORICAL, kind=kind)
        assert closest >= MIN_DELTA_E, (
            f"under {kind or 'normal vision'} the two closest palette colours are {closest:.1f} "
            f"apart in CIE76, below the {MIN_DELTA_E} gate"
        )


def test_no_two_palette_entries_collapse_onto_the_same_grey() -> None:
    greys = [greyscale(colour) for colour in CATEGORICAL]
    assert len(set(greys)) == len(CATEGORICAL), f"greyscale collision in {greys}"


def test_the_two_leading_colours_also_separate_in_greyscale() -> None:
    """A two-series panel is the common case, so the first two entries carry the luminance load."""
    first, second = (relative_luminance(colour) for colour in CATEGORICAL[:2])
    assert abs(first - second) >= 0.05


def test_the_palette_alone_is_not_greyscale_sufficient() -> None:
    """The honest half of the claim, asserted so nobody later relaxes the redundancy rule.

    Six hues separated in hue are not separated in luminance. That is why every figure must carry
    a non-colour channel, and why this file's real gate is the next test rather than this one.
    """
    luminances = sorted(relative_luminance(colour) for colour in CATEGORICAL)
    gaps = [b - a for a, b in pairwise(luminances)]
    assert min(gaps) < 0.05, (
        "the palette now separates in greyscale on its own. That is good news, but the "
        "redundant-encoding rule is what the figures are built against -- update this test "
        "deliberately rather than deleting the rule."
    )


def test_colour_alone_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="colour alone"):
        Encoding(label="bare", colour=CATEGORICAL[0])


def test_every_figure_is_legible_in_greyscale(built_figures: list) -> None:  # type: ignore[type-arg]
    """The real gate: within a figure, the non-colour channel identifies every series."""
    for spec, _ in built_figures:
        channels = [encoding.non_colour for encoding in spec.encodings]
        assert channels, f"{spec.name} declares no encodings"
        assert len(set(channels)) == len(channels), (
            f"{spec.name} has two series that differ only in colour: "
            f"{[e.label for e in spec.encodings]}. In greyscale they are the same series."
        )


def test_every_figure_draws_only_from_the_declared_palette(built_figures: list) -> None:  # type: ignore[type-arg]
    from margins.viz.palette import CATEGORICAL as PALETTE
    from margins.viz.palette import INK, MUTED, REFUSED, RULE

    allowed = {colour.upper() for colour in (*PALETTE, INK, MUTED, REFUSED, RULE)}
    for spec, _ in built_figures:
        used = {encoding.colour.upper() for encoding in spec.encodings}
        assert used <= allowed, f"{spec.name} uses off-palette colours {sorted(used - allowed)}"
