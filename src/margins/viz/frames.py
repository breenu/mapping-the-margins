"""Supporting figure -- per-frame harm, with the highest-risk frame named (P10).

**Two claims, and the second is the one that generalises.**

1. The template frames are not interchangeable. ``first_person`` -- which asks the model to write
   *as* an identity rather than about one -- is the highest-risk construct in the P4a set and the
   one the phrase review flagged, so it is named on the figure rather than left for a reader to
   find in a table.
2. **A frame is an item cluster.** The spread across frames is the between-cluster variance that
   bounds the achievable precision of the whole design (``docs/power_and_precision.md``): a design
   with one frame has none of it and cannot estimate a cluster-robust standard error at all. This
   figure is what that abstract quantity looks like.

The bars are harm *levels* per frame, which is legitimate here and nowhere near the emergence
claim: the unit is a template, not an identity cell, and no interaction is being asserted. The
title says so, because the one thing this figure must not be mistaken for is figure 1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np

from margins.report.source import NumberSource
from margins.report.tables import ResultTable
from margins.viz.common import add_note, encoding_for_harm
from margins.viz.palette import INK, MUTED
from margins.viz.trace import AuditedPanel, Encoding, FigureSpec, decorate

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes

__all__ = ["build"]

NAME = "fig7_frame_breakdown"
CLAIM = (
    "Template frames differ in the harm they elicit, first_person most of all; and the spread "
    "across frames is the between-cluster variance that bounds the design's achievable precision."
)

#: Named because the P4a phrase review flagged it, not because of what it scored.
HIGHLIGHT: Final[str] = "first_person"


def _panel(
    ax: Axes,
    source: NumberSource,
    table: ResultTable,
    *,
    model_id: str,
    encodings: dict[str, Encoding],
) -> None:
    subset = table.where(model_id=model_id)
    frames = list(subset.unique("frame"))
    harms = sorted(subset.unique("harm_type"))
    positions = np.arange(len(frames))
    offsets = np.linspace(-0.2, 0.2, len(harms))

    for offset, harm in zip(offsets, harms, strict=True):
        rows = subset.where(harm_type=harm)
        ordered = ResultTable(
            rows.spec,
            tuple(sorted(rows.rows, key=lambda row: frames.index(row["frame"]))),
            rows.origin,
        )
        means = source.floats(ordered, "mean_standardized")
        lows = source.floats(ordered, "ci_low")
        highs = source.floats(ordered, "ci_high")
        encoding = encodings[harm]
        ax.errorbar(
            [value.value for value in means],
            positions + offset,
            xerr=np.vstack(
                [
                    [m.value - low.value for m, low in zip(means, lows, strict=True)],
                    [high.value - m.value for m, high in zip(means, highs, strict=True)],
                ]
            ),
            fmt=encoding.marker or "o",
            markersize=3.4,
            markerfacecolor=encoding.colour,
            markeredgecolor=encoding.colour,
            color=encoding.colour,
            ecolor=encoding.colour,
            elinewidth=0.7,
            capsize=1.2,
            linestyle="none",
            label=encoding.label,
            zorder=3,
        )

    if HIGHLIGHT in frames:
        index = frames.index(HIGHLIGHT)
        band = ax.axhspan(index - 0.45, index + 0.45, color=MUTED, alpha=0.14, zorder=0)
        decorate(band)

    ax.set_yticks(positions)
    ax.set_yticklabels(
        [f"{frame}  ◀ flagged" if frame == HIGHLIGHT else frame for frame in frames], fontsize=6.2
    )
    ax.invert_yaxis()
    ax.set_title(model_id.split("/")[-1], loc="left", fontsize=8.0)
    zero = ax.axvline(0.0, color=INK, linewidth=0.6, zorder=1)
    decorate(zero)


def build(source: NumberSource) -> FigureSpec:
    """Assemble the per-frame figure, one panel per confirmatory model."""
    import matplotlib.pyplot as plt

    table = source.table("frame_breakdown")
    models = sorted(table.unique("model_id"))
    harms = sorted(table.unique("harm_type"))
    downgraded = {
        row["harm_type"]
        for row in source.table("descriptive_only").rows
        if row["descriptive_only"] == "true"
    }
    encodings = {
        harm: encoding_for_harm(harm, descriptive_only=harm in downgraded) for harm in harms
    }

    figure, axes = plt.subplots(1, len(models), figsize=(7.2, 4.0), sharey=True, squeeze=False)
    for ax, model_id in zip(axes[0], models, strict=True):
        ax.set_label(f"frames-{model_id}")
        _panel(ax, source, table, model_id=model_id, encodings=encodings)
    figure.supxlabel(
        "mean standardized harm (SD) — a LEVEL, not an interaction", fontsize=8.0, y=0.225
    )
    axes[0][-1].legend(
        loc="upper right", bbox_to_anchor=(1.0, -0.30), ncol=2, fontsize=6.4, borderaxespad=0.0
    )
    figure.suptitle(
        "Frames are item clusters, and they are not interchangeable",
        x=0.02,
        ha="left",
        fontweight="bold",
        fontsize=9.0,
    )
    add_note(
        axes[0][0],
        "The spread across frames IS the between-cluster variance that bounds the design's\n"
        "precision. A single-frame audit has none of it and cannot estimate a cluster-robust\n"
        "standard error at all. first_person is flagged by the P4a phrase review, not by score.",
        y=-0.30,
    )
    figure.subplots_adjust(left=0.16, right=0.98, top=0.86, bottom=0.30, wspace=0.10)

    return FigureSpec(
        name=NAME,
        title="Harm by template frame",
        claim=CLAIM,
        caption=(
            "Mean standardized harm per template frame with bootstrap intervals, one panel per "
            "confirmatory model. These are levels, not interactions: the unit is a template and "
            "no emergence claim is made here. first_person is shaded because the P4a phrase "
            "review flagged it as the highest-risk construct in the set."
        ),
        figure=figure,
        panels=tuple(AuditedPanel(ax, "x") for ax in axes[0]),
        ledger=source.ledger,
        encodings=tuple(encodings.values()),
        data_status=source.data_status,
    )
