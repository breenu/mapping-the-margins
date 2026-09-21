"""Figure 2 -- construct validity: the scorer's own interaction, beside the model's (P10).

**The claim.** A classifier that scores text higher merely for naming an identity has an
interaction of its own, and if that interaction is comparable in magnitude to the model's then the
measured "emergence" cannot be attributed to the model at all. This figure is the evidence for
every descriptive-only downgrade the paper makes, and the reason those downgrades are a design
feature rather than a caveat.

**Nothing is point-subtracted, and the figure is drawn so that it cannot be.** Section 9 permits
either netting the drift out or reporting the two side by side; the project reports side by side,
because a difference of two noisy magnitudes has a variance nobody computed and a sign nobody
checked. The left panel therefore plots the pair ``(model RMS, drift RMS)`` as a point --
a reader can see both coordinates -- rather than plotting a difference, which would show neither.

**Reading it.** A point on the diagonal is a scorer whose response to identity terms on
known-neutral text is as structured as its response to the model's output. The shallower ray is
the pre-registered ``descriptive_only_ratio``; anything above it costs its harm type every
significance claim. The two marker fills are the two neutral control families -- imperative
prompt echo and flat declarative prose -- kept apart rather than averaged, because a scorer whose
drift depends on register is a finding in its own right.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from margins.report.source import NumberSource
from margins.report.tables import ResultTable
from margins.viz.common import add_note
from margins.viz.palette import CATEGORICAL, INK, MUTED, REFUSED, RULE
from margins.viz.trace import AuditedPanel, Encoding, FigureSpec, decorate

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes

__all__ = ["build"]

NAME = "fig2_scorer_drift"
CLAIM = (
    "Each scorer has its own interaction over identity descriptors on known-neutral text. Where "
    "that interaction is a pre-registered fraction of the model's, emergence in the model cannot "
    "be distinguished from emergence in the instrument, and the harm type is descriptive-only."
)

#: Marker per scorer, fill per control family. Colour is the redundant channel here, not the
#: load-bearing one.
_MARKERS = ("o", "s", "D", "^", "v", "P")
_FAMILY_FILL: Mapping[str, bool] = {"prompt_echo": True, "neutral_declarative": False}


def _scatter(
    ax: Axes,
    source: NumberSource,
    table: ResultTable,
    *,
    threshold: float,
    encodings: Mapping[tuple[str, str], Encoding],
) -> None:
    model_all = source.floats(table, "model_rms")
    drift_all = source.floats(table, "drift_rms")
    limit = source.derived(
        max(value.value for value in model_all + drift_all) * 1.08,
        key="drift_axis_limit",
        origin=table.origin,
        how="1.08 x the largest RMS plotted, so the diagonal spans the data",
    )

    for (scorer, family), encoding in encodings.items():
        subset = table.where(scorer_id=scorer, control_family=family)
        if not len(subset):
            continue
        x = [value.value for value in source.floats(subset, "model_rms")]
        y = [value.value for value in source.floats(subset, "drift_rms")]
        ax.plot(
            x,
            y,
            linestyle="none",
            marker=encoding.marker,
            markersize=3.6,
            markerfacecolor=encoding.colour if encoding.filled else "white",
            markeredgecolor=encoding.colour,
            markeredgewidth=0.8,
            label=encoding.label,
            zorder=3,
        )

    diagonal = ax.plot(
        [0.0, limit.value],
        [0.0, limit.value],
        color=RULE,
        linewidth=0.8,
        linestyle=":",
        zorder=1,
    )
    ray = ax.plot(
        [0.0, limit.value],
        [0.0, threshold * limit.value],
        color=INK,
        linewidth=0.8,
        linestyle="--",
        zorder=1,
    )
    equal = ax.text(
        limit.value * 0.62,
        limit.value * 0.68,
        "scorer = model",
        fontsize=6.0,
        color=RULE,
        rotation=38,
    )
    downgrade = ax.text(
        limit.value * 0.62,
        limit.value * threshold * 0.70,
        "DESCRIPTIVE-ONLY above this ray",
        fontsize=6.0,
        color=INK,
        rotation=22,
    )
    decorate([*diagonal, *ray])
    equal.set_gid(None)
    downgrade.set_gid(None)

    ax.set_xlim(0.0, limit.value)
    ax.set_ylim(0.0, limit.value)
    ax.set_xlabel("model's interaction, RMS γ (SD)")
    ax.set_ylabel("that scorer's own interaction\non neutral text, RMS γ (SD)")
    ax.set_title("Instrument beside model — nothing subtracted", loc="left")
    ax.legend(
        loc="upper left", bbox_to_anchor=(0.0, -0.20), ncol=2, fontsize=6.0, borderaxespad=0.0
    )


def _ratio_bars(
    ax: Axes,
    source: NumberSource,
    table: ResultTable,
    decisions: ResultTable,
    *,
    threshold: float,
    colours: Mapping[str, str],
) -> None:
    """Largest drift ratio per scorer, with every individual ratio behind it."""
    keys = sorted({(row["harm_type"], row["scorer_id"]) for row in table.rows})
    downgraded = {row["harm_type"] for row in decisions.rows if row["descriptive_only"] == "true"}
    positions = np.arange(len(keys))
    for position, (harm, scorer) in zip(positions, keys, strict=True):
        subset = table.where(harm_type=harm, scorer_id=scorer)
        ratios = source.floats(subset, "ratio")
        worst = source.derived(
            max(value.value for value in ratios),
            key=f"max_drift_ratio[{harm}|{scorer}]",
            origin=table.origin,
            how="largest drift_rms / model_rms over the pair x control-family grid",
        )
        over = worst.value >= threshold
        ax.barh(
            position,
            worst.value,
            height=0.6,
            color=REFUSED if over else "white",
            edgecolor=colours[scorer],
            linewidth=0.8,
            hatch="xxx" if over else None,
            zorder=2,
        )
        ax.plot(
            [value.value for value in ratios],
            np.full(len(ratios), position),
            linestyle="none",
            marker="|",
            markersize=4.0,
            color=colours[scorer],
            zorder=3,
        )
        if over:
            flag = ax.text(
                worst.value - 0.02,
                position - 0.36,
                "no significance claim",
                ha="right",
                va="bottom",
                fontsize=5.8,
                color=INK,
            )
            flag.set_gid(None)

    rule = ax.axvline(threshold, color=INK, linewidth=0.9, linestyle="--", zorder=1)
    decorate(rule)
    ax.set_yticks(positions)
    ax.set_yticklabels([f"{harm} · {scorer}" for harm, scorer in keys], fontsize=6.2)
    ax.invert_yaxis()
    ax.set_xlabel("drift RMS ÷ model RMS")
    ax.set_title("What the pre-registered threshold decides", loc="left")
    ax.text(
        threshold,
        len(keys) - 0.42,
        f"pre-registered threshold {threshold:g}",
        ha="center",
        va="top",
        fontsize=6.0,
        color=INK,
    ).set_gid(None)
    ax.set_xlim(0.0, 1.0)
    for harm in sorted(downgraded):
        rows = [index for index, key in enumerate(keys) if key[0] == harm]
        band = ax.axhspan(min(rows) - 0.5, max(rows) + 0.5, color=MUTED, alpha=0.10, zorder=0)
        decorate(band)


def build(source: NumberSource, *, benchmark: str, model_id: str) -> FigureSpec:
    """Assemble figure 2 from the committed drift tables."""
    import matplotlib.pyplot as plt

    drift = source.table("drift_side_by_side").where(benchmark=benchmark, model_id=model_id)
    decisions = source.table("descriptive_only").where(benchmark=benchmark, model_id=model_id)
    thresholds = source.floats(decisions, "threshold")
    unique = {value.value for value in thresholds}
    if len(unique) != 1:
        raise ValueError(
            f"the drift rule was applied at {sorted(unique)} in one run. The threshold is "
            "pre-registered and is one number; a figure cannot draw two."
        )
    threshold = next(iter(unique))

    scorers = sorted({row["scorer_id"] for row in drift.rows})
    colours = {
        scorer: CATEGORICAL[index % len(CATEGORICAL)] for index, scorer in enumerate(scorers)
    }
    families = sorted({row["control_family"] for row in drift.rows})
    encodings = {
        (scorer, family): Encoding(
            label=f"{scorer} · {family}",
            colour=colours[scorer],
            marker=_MARKERS[index % len(_MARKERS)],
            filled=_FAMILY_FILL.get(family, True),
        )
        for index, scorer in enumerate(scorers)
        for family in families
    }

    figure = plt.figure(figsize=(7.2, 4.8))
    grid = figure.add_gridspec(
        1, 2, width_ratios=(1.0, 1.05), wspace=0.36, left=0.10, right=0.98, top=0.88, bottom=0.36
    )
    scatter_ax = figure.add_subplot(grid[0, 0], label="drift-scatter")
    bar_ax = figure.add_subplot(grid[0, 1], label="drift-ratio")

    _scatter(scatter_ax, source, drift, threshold=threshold, encodings=encodings)
    _ratio_bars(bar_ax, source, drift, decisions, threshold=threshold, colours=colours)

    add_note(
        bar_ax,
        "A tick is one axis pair in one neutral control family; the bar is the largest of them.\n"
        "The two families are kept apart, never averaged: a scorer whose drift depends on the\n"
        "register of the text around an identity term is a reportable finding of its own.",
        y=-0.36,
    )
    figure.suptitle(
        "A scorer's own interaction is not the model's", x=0.02, ha="left", fontweight="bold"
    )

    return FigureSpec(
        name=NAME,
        title="Scorer identity drift beside the model's interaction",
        claim=CLAIM,
        caption=(
            "Left: for every scorer, axis pair and neutral control family, the root-mean-square "
            "interaction of the model's output as that scorer sees it against the same scorer's "
            "interaction over identical descriptors in known-neutral text. Points are plotted, "
            "not differenced. Right: the largest ratio per scorer against the pre-registered "
            "threshold that makes a harm type descriptive-only. A shaded row band marks a harm "
            "type that lost its claims; it keeps every estimate it had."
        ),
        figure=figure,
        panels=(AuditedPanel(scatter_ax, "both"), AuditedPanel(bar_ax, "x")),
        ledger=source.ledger,
        encodings=tuple(encodings.values()),
        data_status=source.data_status,
    )
