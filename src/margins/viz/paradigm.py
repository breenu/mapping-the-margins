"""Figure 4 -- RQ1, scoped: what each paradigm can and cannot express (P10).

**The claim, in the form the 21 Sep 2026 sign-off fixed it.** The cross-paradigm comparison is
generation vs. extraction *at the single-axis, descriptive level*, **plus** a structural finding:
a standard sentence-pair benchmark cannot populate an intersectional cell at all. Of 1,352 kept
CrowS-Pairs items, three attest an order-2 cell, because each item is annotated with exactly one
attribute and an order-2 cell requires a second identity term present unchanged in *both*
sentences. Every confirmatory pair therefore comes back not estimable, with an explicit count of
empty cells.

**What this figure must never do, and structurally cannot.** It must never draw an extractive
interaction estimate -- not as a point, and above all not as a zero with an interval around it.
"CrowS-Pairs shows no intersectional effect" would be false; CrowS-Pairs cannot express one. The
interaction panel therefore has a slot for the extractive paradigm that carries the machine-
readable reason from ``fit_reports`` in place of a marker, and :func:`build` **raises** if an
extractive row ever arrives claiming to be estimable, rather than silently plotting it.

The left panel is the comparison that *is* available: order-1 cell means with intervals, per
paradigm, on the cells both attest. The right panel is the structural result, on a log axis
because the counts span three orders of magnitude and a linear axis would render the order-2 bar
invisible -- which would be the wrong kind of honest.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np

from margins.report.source import NumberSource
from margins.report.tables import ResultTable
from margins.viz.common import add_note, axis_label
from margins.viz.palette import CATEGORICAL, INK, MUTED, REFUSED
from margins.viz.trace import AuditedPanel, Encoding, FigureSpec, decorate

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes

__all__ = ["ExtractiveEstimableError", "build"]

NAME = "fig4_cross_paradigm"
CLAIM = (
    "Generation and extraction are comparable at the single-axis level. They are not comparable "
    "at the intersectional level, because the extractive paradigm structurally cannot populate an "
    "order-2 cell -- which is a result about sentence-pair benchmarks, not a null result about "
    "the model."
)

_GENERATION: Final[str] = "open_gen_templates"
_EXTRACTION: Final[str] = "crows_pairs"


class ExtractiveEstimableError(AssertionError):
    """An extractive row claimed an estimable interaction. The figure refuses to draw it."""


def _single_axis(
    ax: Axes,
    source: NumberSource,
    cells: ResultTable,
    encodings: dict[tuple[str, str], Encoding],
) -> None:
    """Order-1 cell means per paradigm, on the cells both paradigms attest."""
    # One series per (paradigm, harm type). The two paradigms measure DIFFERENT harm constructs
    # -- open generation is scored for regard and toxicity, an extractive pair for stereotype
    # preference -- and each is standardized within its own source. Collapsing them into one
    # series per paradigm would hide exactly that.
    series = sorted({(row["benchmark"], row["harm_type"]) for row in cells.rows})
    by_series = {key: cells.where(benchmark=key[0], harm_type=key[1]) for key in series}
    shared = set.intersection(*(set(table.strings("cell_id")) for table in by_series.values()))
    generation = next(key for key in series if key[0] == _GENERATION)
    ordered = [cell for cell in by_series[generation].unique("cell_id") if cell in shared]
    positions = {cell: index for index, cell in enumerate(ordered)}
    offsets = np.linspace(-0.26, 0.26, len(by_series))

    for offset, (key, table) in zip(offsets, by_series.items(), strict=True):
        rows = [row for row in table.rows if row["cell_id"] in positions]
        subset = ResultTable(table.spec, tuple(rows), table.origin)
        means = source.floats(subset, "mean_standardized")
        lows = source.floats(subset, "ci_low")
        highs = source.floats(subset, "ci_high")
        y = [positions[row["cell_id"]] + offset for row in rows]
        encoding = encodings[key]
        ax.errorbar(
            [value.value for value in means],
            y,
            xerr=np.vstack(
                [
                    [m.value - low.value for m, low in zip(means, lows, strict=True)],
                    [high.value - m.value for m, high in zip(means, highs, strict=True)],
                ]
            ),
            fmt=encoding.marker or "o",
            markersize=3.4,
            markerfacecolor=encoding.colour if encoding.filled else "white",
            markeredgecolor=encoding.colour,
            color=encoding.colour,
            ecolor=encoding.colour,
            elinewidth=0.7,
            capsize=1.2,
            linestyle="none",
            label=encoding.label,
            zorder=3,
        )

    labels = []
    for cell in ordered:
        axis, value = cell.split("=", 1)
        labels.append(f"{value}  ({axis_label(axis)})")
    ax.set_yticks(list(positions.values()))
    ax.set_yticklabels(labels, fontsize=6.0)
    ax.invert_yaxis()
    ax.set_xlabel("mean standardized harm (SD, within source)")
    ax.set_title("Comparable: single-axis cells", loc="left")
    zero = ax.axvline(0.0, color=MUTED, linewidth=0.7, zorder=0)
    decorate(zero)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, -0.10),
        ncol=3,
        fontsize=5.8,
        columnspacing=1.0,
        handletextpad=0.4,
        borderaxespad=0.0,
    )


def _structure(ax: Axes, source: NumberSource, encoding: Encoding) -> None:
    """The measured order-2 yield of the extractive source, against what it would need."""
    bars = (
        ("kept items", "crows.kept"),
        ("attest an order-1 cell", "crows.order_1_items"),
        ("attest an order-2 cell", "crows.order_2_items"),
    )
    values = [source.number(key) for _, key in bars]
    positions = np.arange(len(bars))
    ax.barh(
        positions,
        [value.value for value in values],
        height=0.55,
        color="white",
        edgecolor=encoding.colour,
        hatch=encoding.hatch,
        linewidth=0.9,
        zorder=2,
    )
    for position, value in zip(positions, values, strict=True):
        label = ax.text(
            value.value * 1.25,
            float(position),
            f"{value.value:.0f}",
            va="center",
            fontsize=6.6,
            color=INK,
        )
        label.set_gid(None)

    ax.set_xscale("log")
    ax.set_yticks(positions)
    ax.set_yticklabels([label for label, _ in bars], fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlabel("items (log scale)")
    ax.set_title("Not comparable: the order-2 yield", loc="left")


def _not_estimable(ax: Axes, source: NumberSource, reports: ResultTable) -> None:
    """The extractive interaction slot: a reason, never a point and never a zero."""
    extractive = reports.where(benchmark=_EXTRACTION).sorted_by("axis_a", "axis_b")
    estimable = [row for row in extractive.rows if row["estimable"] == "true"]
    if estimable:
        raise ExtractiveEstimableError(
            f"{len(estimable)} extractive rows claim an estimable interaction. P2 measured that "
            "the paradigm attests three order-2 items in total; a figure that drew this would "
            "report a null result about the model where the truth is that the instrument cannot "
            "express the effect."
        )
    pair_set = sorted({(row["axis_a"], row["axis_b"]) for row in extractive.rows})
    pairs = source.derived(
        len(pair_set),
        key="extractive_pairs_attempted",
        origin=extractive.origin,
        how="distinct confirmatory axis pairs the extractive source was fitted for",
    )
    cells = source.floats(extractive.where(harm_type=extractive.strings("harm_type")[0]), "n_cells")
    total = source.derived(
        sum(value.value for value in cells),
        key="extractive_grid_cells",
        origin=extractive.origin,
        how="order-2 cells across the confirmatory pairs the extractive source was asked to fill",
    )
    attested = source.number("crows.order_2_items")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    box = ax.text(
        0.0,
        1.0,
        "Extractive interaction estimate\n\nNOT ESTIMABLE",
        ha="left",
        va="top",
        fontsize=8.0,
        color=INK,
        fontweight="bold",
    )
    box.set_gid(None)
    detail = ax.text(
        0.0,
        0.52,
        f"All {pairs.value:.0f} confirmatory pairs, both reductions.\n"
        f"{total.value:.0f} order-2 cells to fill; {attested.value:.0f} items in the whole "
        "source attest one.\n"
        "The estimator needs every cell of a pair's grid non-empty,\n"
        "so it reports a reason rather than a number.\n\n"
        "This slot stays empty on purpose. Drawing a zero here would\n"
        "say the model has no intersectional effect. What it says is\n"
        "that this instrument cannot express one.",
        ha="left",
        va="top",
        fontsize=6.4,
        color=INK,
    )
    detail.set_gid(None)
    from matplotlib.patches import Rectangle

    frame = ax.add_patch(
        Rectangle(
            (-0.05, -0.85),
            1.10,
            1.95,
            transform=ax.transAxes,
            fill=True,
            facecolor=REFUSED,
            alpha=0.18,
            edgecolor=MUTED,
            linewidth=0.8,
            linestyle="--",
            zorder=0,
            # The panel is an annotation slot rather than a plot; its text deliberately runs
            # past the axes box, and the frame has to follow it.
            clip_on=False,
        )
    )
    decorate(frame)


def build(source: NumberSource, *, model_id: str) -> FigureSpec:
    """Assemble figure 4 from the committed tables plus the measured CrowS coverage numbers."""
    import matplotlib.pyplot as plt

    cells = source.table("single_axis_cells").where(model_id=model_id)
    reports = source.table("fit_reports")

    #: Colour names the paradigm, marker names the harm construct. Colour is the redundant
    #: channel: the marker alone distinguishes every series in greyscale.
    paradigm_colour = {_GENERATION: CATEGORICAL[0], _EXTRACTION: CATEGORICAL[2]}
    markers = ("o", "s", "D", "^", "v", "P")
    series = sorted({(row["benchmark"], row["harm_type"]) for row in cells.rows})
    encodings = {
        key: Encoding(
            label=f"{'generation' if key[0] == _GENERATION else 'extraction'} · {key[1]}",
            colour=paradigm_colour.get(key[0], CATEGORICAL[3]),
            marker=markers[index % len(markers)],
            filled=key[0] == _GENERATION,
        )
        for index, key in enumerate(series)
    }
    structure_encoding = Encoding(
        "extraction (CrowS-Pairs)", CATEGORICAL[2], marker="D", hatch="///", filled=False
    )

    figure = plt.figure(figsize=(7.2, 4.9))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.15, 1.0),
        height_ratios=(1.0, 1.0),
        wspace=0.34,
        hspace=0.95,
        left=0.16,
        right=0.98,
        top=0.88,
        bottom=0.23,
    )
    cells_ax = figure.add_subplot(grid[:, 0], label="single-axis")
    structure_ax = figure.add_subplot(grid[0, 1], label="structure")
    slot_ax = figure.add_subplot(grid[1, 1], label="not-estimable")

    _single_axis(cells_ax, source, cells, encodings)
    _structure(structure_ax, source, structure_encoding)
    _not_estimable(slot_ax, source, reports)

    add_note(
        cells_ax,
        "Standardized WITHIN source, never across: the two paradigms measure on different\n"
        "instruments and a shared z-score would assert a comparability that was not tested.",
        y=-0.205,
    )
    figure.suptitle(
        "Sentence-pair benchmarks cannot express an intersectional cell",
        x=0.02,
        ha="left",
        fontweight="bold",
    )

    return FigureSpec(
        name=NAME,
        title="Cross-paradigm comparison, scoped by what each paradigm can express",
        claim=CLAIM,
        caption=(
            "Left: order-1 cell means with intervals for both paradigms, on the cells both "
            "attest, each standardized within its own source. Top right: the measured order-2 "
            "yield of the pinned CrowS-Pairs revision. Bottom right: the extractive "
            "interaction slot, which carries the estimator's machine-readable reason instead of "
            "an estimate. The absence is the finding and it is about the benchmark, not the "
            "model."
        ),
        figure=figure,
        panels=(
            AuditedPanel(cells_ax, "x"),
            AuditedPanel(structure_ax, "x"),
            AuditedPanel(slot_ax, "x"),
        ),
        ledger=source.ledger,
        encodings=(*encodings.values(), structure_encoding),
        data_status=source.data_status,
    )
