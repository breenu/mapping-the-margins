"""The palette, and the colour arithmetic that lets a test *check* it (P10).

Accessibility here is a gate, not an intention. Two properties are asserted by
``tests/test_viz_palette.py`` rather than asserted in a docstring:

1. **Colourblind-safe.** Every pair of categorical colours stays perceptually separated under
   simulated protanopia, deuteranopia and tritanopia. The simulation is the Machado, Oliveira &
   Fernandes (2009) severity-1.0 matrices applied in *linear* RGB, and the separation is CIE76
   ``Delta E`` in L*a*b*. Both are implemented here in ~60 lines of numpy rather than pulled in as
   a dependency: the project's environment is locked for exactly two platforms and a colour
   library is not worth a lock churn.
2. **Greyscale-legible.** This one cannot be met by colour choice alone and the module does not
   pretend otherwise. Six hues that are well separated in hue are *not* well separated in
   luminance -- that is arithmetic, not a failure of taste. So the rule the project enforces is
   the stronger one: **colour never carries meaning by itself.** Every categorical encoding also
   carries a distinct marker, hatch or linestyle (:class:`margins.viz.trace.Encoding`), and the
   figure tests assert that the non-colour channel alone identifies every series. What this module
   guarantees about luminance is the weaker, checkable thing: the palette spans a usable luminance
   band and no two entries collapse onto the same grey.

The base hues are the Okabe-Ito qualitative palette, which is the field standard for
colour-vision-deficient readers and is public domain. The ordering is ours: the first two entries
are the two that separate best in greyscale as well as in hue, because a two-series panel is the
common case.

**Measured, 21 Sep 2026** (recomputed by the tests, never trusted from this comment): the six
categorical colours' closest pair sits at ``Delta E`` 26.4 under normal vision, 20.7 under
protanopia, 18.1 under deuteranopia and 16.2 under tritanopia, against a just-noticeable
difference of about 2.3. Their greyscale luminances span 0.152 to 0.416 and the closest two are
0.011 apart -- which is the arithmetic behind the paragraph above, stated as a number.
"""

from __future__ import annotations

from typing import Final

import numpy as np

__all__ = [
    "CATEGORICAL",
    "CVD_KINDS",
    "DESCRIPTIVE",
    "DIVERGING",
    "GRID",
    "INK",
    "MUTED",
    "REFUSED",
    "RULE",
    "CvdKind",
    "delta_e76",
    "greyscale",
    "hex_to_rgb",
    "min_pairwise_delta_e",
    "relative_luminance",
    "simulate_cvd",
    "srgb_to_lab",
]

#: Okabe-Ito, reordered. Blue and vermillion lead because they separate in luminance as well as
#: in hue, which is what a two-series panel needs. Yellow (#F0E442) is deliberately absent: it is
#: unreadable as a line or a small marker on white.
CATEGORICAL: Final[tuple[str, ...]] = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
)

#: Text, axes and anything structural. Not pure black: 0.15 grey prints without haloing.
INK: Final[str] = "#26282B"
#: Gridlines and other furniture that must recede behind the data.
GRID: Final[str] = "#D9DCE0"
#: Reference lines that are part of the argument (zero, a pre-registered threshold).
RULE: Final[str] = "#6B7280"
#: Anything reported but not load-bearing: descriptive-only, exploratory, underpowered.
MUTED: Final[str] = "#9AA1AB"
#: A refused claim. Deliberately the same grey family as MUTED, never a warning red -- a refusal
#: is an abstention, and colouring it like an error would read as a finding about the model.
REFUSED: Final[str] = "#B4BAC2"
#: Descriptive-only shading, used as a panel background band.
DESCRIPTIVE: Final[str] = "#F1F2F4"

#: Diverging map for the interaction heatmap. Centred at zero because gamma is signed and zero is
#: the additive null -- a sequential map would make "no emergence" look like a low value of
#: something rather than the hypothesis being tested.
DIVERGING: Final[str] = "RdBu_r"

CvdKind = str
CVD_KINDS: Final[tuple[str, ...]] = ("protanopia", "deuteranopia", "tritanopia")

# Machado, Oliveira & Fernandes (2009), severity 1.0, applied in linear RGB.
_CVD_MATRICES: Final[dict[str, np.ndarray]] = {
    "protanopia": np.array(
        [
            [0.152286, 1.052583, -0.204868],
            [0.114503, 0.786281, 0.099216],
            [-0.003882, -0.048116, 1.051998],
        ]
    ),
    "deuteranopia": np.array(
        [
            [0.367322, 0.860646, -0.227968],
            [0.280085, 0.672501, 0.047413],
            [-0.011820, 0.042940, 0.968881],
        ]
    ),
    "tritanopia": np.array(
        [
            [1.255528, -0.076749, -0.178779],
            [-0.078411, 0.930809, 0.147602],
            [0.004733, 0.691367, 0.303900],
        ]
    ),
}

# sRGB (D65) -> CIE XYZ.
_RGB_TO_XYZ: Final[np.ndarray] = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_D65: Final[np.ndarray] = np.array([0.95047, 1.00000, 1.08883])


def hex_to_rgb(colour: str) -> np.ndarray:
    """``"#0072B2"`` -> ``array([0.0, 0.447, 0.698])`` in gamma-encoded sRGB."""
    text = colour.lstrip("#")
    if len(text) != 6:
        raise ValueError(f"{colour!r} is not a 6-digit hex colour")
    return np.array([int(text[i : i + 2], 16) / 255.0 for i in (0, 2, 4)])


def _to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _to_srgb(linear: np.ndarray) -> np.ndarray:
    encoded = np.where(
        linear <= 0.0031308, linear * 12.92, 1.055 * np.maximum(linear, 0.0) ** (1 / 2.4) - 0.055
    )
    return np.clip(encoded, 0.0, 1.0)


def relative_luminance(colour: str) -> float:
    """WCAG relative luminance: what the colour becomes in greyscale."""
    linear = _to_linear(hex_to_rgb(colour))
    return float(np.dot(linear, np.array([0.2126, 0.7152, 0.0722])))


def greyscale(colour: str) -> str:
    """The colour as it prints in greyscale, as a hex string."""
    level = _to_srgb(np.full(3, relative_luminance(colour)))
    return "#" + "".join(f"{round(channel * 255):02X}" for channel in level)


def simulate_cvd(colour: str, kind: str) -> str:
    """The colour as a reader with ``kind`` sees it, as a hex string."""
    if kind not in _CVD_MATRICES:
        raise KeyError(f"unknown colour-vision deficiency {kind!r}; known: {CVD_KINDS}")
    simulated = _to_srgb(_CVD_MATRICES[kind] @ _to_linear(hex_to_rgb(colour)))
    return "#" + "".join(f"{round(channel * 255):02X}" for channel in simulated)


def srgb_to_lab(colour: str) -> np.ndarray:
    """CIE L*a*b* under D65, the space the separation threshold is stated in."""
    xyz = (_RGB_TO_XYZ @ _to_linear(hex_to_rgb(colour))) / _D65

    def f(value: float) -> float:
        return value ** (1 / 3) if value > 216 / 24389 else (24389 / 27 * value + 16) / 116

    fx, fy, fz = (f(float(component)) for component in xyz)
    return np.array([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)])


def delta_e76(first: str, second: str) -> float:
    """CIE76 colour difference. Crude next to CIEDE2000 and entirely adequate as a floor."""
    return float(np.linalg.norm(srgb_to_lab(first) - srgb_to_lab(second)))


def min_pairwise_delta_e(colours: tuple[str, ...], *, kind: str | None = None) -> float:
    """The closest two colours get, optionally as seen under a colour-vision deficiency."""
    seen = [colour if kind is None else simulate_cvd(colour, kind) for colour in colours]
    return min(
        delta_e76(seen[i], seen[j]) for i in range(len(seen)) for j in range(i + 1, len(seen))
    )
