"""Supporting figure -- RQ4: how much the headline interaction moves with decoding (P10).

**The claim.** The reported interaction is a property of the model and the lattice, not of the
temperature and the seeds that happened to be drawn. RQ4 is answered by the *spread*, so the
figure plots the spread: one band per model and harm type across the pre-registered temperature
grid, with the individual seed groups drawn inside it rather than averaged away.

The band is min-to-max of the seed groups' cluster-bootstrap intervals, not a standard error of
the mean over three groups -- three is too few to estimate a variance from, and a narrow interval
computed that way would overstate stability in exactly the direction that flatters the result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from margins.report.source import NumberSource
from margins.report.tables import ResultTable
from margins.viz.common import add_note, encoding_for_harm, pair_label
from margins.viz.palette import MUTED
from margins.viz.trace import AuditedPanel, Encoding, FigureSpec, decorate

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes

__all__ = ["build"]

NAME = "fig6_sensitivity"
CLAIM = (
    "The headline interaction is stable across the pre-registered temperature grid and across "
    "seed groups; the reported spread is the min-to-max of the groups' intervals, not a standard "
    "error over three of them."
)


def _panel(
    ax: Axes,
    source: NumberSource,
    table: ResultTable,
    *,
    model_id: str,
    encodings: dict[str, Encoding],
) -> None:
    subset = table.where(model_id=model_id)
    for harm in sorted(subset.unique("harm_type")):
        rows = subset.where(harm_type=harm).sorted_by("temperature", "seed_group")
        temperatures = sorted({float(row["temperature"]) for row in rows.rows})
        encoding = encodings[harm]
        centres: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for temperature in temperatures:
            at = rows.where(temperature=f"{temperature:.10g}")
            estimates = source.floats(at, "estimate")
            low = source.floats(at, "ci_low")
            high = source.floats(at, "ci_high")
            centres.append(
                source.derived(
                    sum(value.value for value in estimates) / len(estimates),
                    key=f"sensitivity_mean[{model_id}|{harm}|{temperature:g}]",
                    origin=rows.origin,
                    how="mean estimate over the seed groups at this temperature",
                ).value
            )
            lows.append(min(value.value for value in low))
            highs.append(max(value.value for value in high))
            ax.plot(
                np.full(len(estimates), temperature),
                [value.value for value in estimates],
                linestyle="none",
                marker=encoding.marker,
                markersize=3.0,
                markerfacecolor="white",
                markeredgecolor=encoding.colour,
                markeredgewidth=0.7,
                zorder=4,
            )
        band = ax.fill_between(
            temperatures, lows, highs, color=encoding.colour, alpha=0.16, linewidth=0.0, zorder=1
        )
        band.set_gid(None)
        ax.plot(
            temperatures,
            centres,
            color=encoding.colour,
            linestyle=encoding.linestyle or "-",
            marker=encoding.marker,
            markersize=3.6,
            label=encoding.label,
            zorder=3,
        )

    ax.set_xticks(sorted({float(row["temperature"]) for row in subset.rows}))
    ax.set_xlabel("decoding temperature")
    ax.set_title(model_id.split("/")[-1], loc="left", fontsize=8.0)
    zero = ax.axhline(0.0, color=MUTED, linewidth=0.6, zorder=0)
    decorate(zero)


def build(source: NumberSource) -> FigureSpec:
    """Assemble the sensitivity figure, one panel per confirmatory model."""
    import matplotlib.pyplot as plt

    table = source.table("sensitivity")
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

    figure, axes = plt.subplots(1, len(models), figsize=(7.2, 3.3), sharey=True, squeeze=False)
    for ax, model_id in zip(axes[0], models, strict=True):
        ax.set_label(f"sensitivity-{model_id}")
        _panel(ax, source, table, model_id=model_id, encodings=encodings)
    axes[0][0].set_ylabel("interaction coefficient γ (SD)")
    axes[0][-1].legend(
        loc="upper right", bbox_to_anchor=(1.0, -0.20), ncol=2, fontsize=6.4, borderaxespad=0.0
    )

    first = table.rows[0]
    pair = pair_label(first["axis_a"], first["axis_b"])
    values = [part.split("=", 1)[1] for part in first["term"].split(":")]
    figure.suptitle(
        f"The headline interaction across decoding: {pair} · {' × '.join(values)}",
        x=0.02,
        ha="left",
        fontweight="bold",
        fontsize=9.0,
    )
    add_note(
        axes[0][0],
        "Open markers are individual seed groups; the band is min-to-max of their\n"
        "cluster-bootstrap intervals. Three groups is too few to estimate a variance\n"
        "from, so no standard error over them is drawn.",
        y=-0.20,
    )
    figure.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.32, wspace=0.12)

    return FigureSpec(
        name=NAME,
        title="Seed and temperature sensitivity of the headline interaction",
        claim=CLAIM,
        caption=(
            "The pre-registered confirmatory interaction term re-estimated across the sensitivity "
            "subset's temperature grid, one panel per confirmatory model. Bands are the "
            "min-to-max of the seed groups' bootstrap intervals; markers are the groups "
            "themselves. A descriptive-only harm type is muted and dashed here as everywhere."
        ),
        figure=figure,
        panels=tuple(AuditedPanel(ax, "y") for ax in axes[0]),
        ledger=source.ledger,
        encodings=tuple(encodings.values()),
        data_status=source.data_status,
    )
