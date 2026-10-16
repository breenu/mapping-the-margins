"""Supporting figure -- RQ4: how much the interaction moves with decoding and seed (P10, E4).

**The claim.** The class x race interaction terms are a property of the model and the lattice,
not of the temperature and seeds that happened to be drawn. RQ4 is answered by the *spread*, so
the figure plots it: one panel per model x harm type, every free interaction term along the
x-axis (pre-registration §10, 16 Oct 2026: all nine, so no term is selected after seeing the
data), and at each term the three temperature arms side by side -- the pooled estimate with its
cluster-bootstrap CI, and each seed group as a small open point beside it.

**Why no band and no SE over seeds.** Three or four seed groups are too few to estimate a variance
from, and a narrow interval computed that way would overstate stability in exactly the direction
that flatters the result. The seed groups are drawn, not summarized.

**The headline reference.** Each arm is standardized within itself (§10), so the T=0.7 arm is not
numerically the headline E1 estimate (standardized over all 110 cells). The headline estimate is
drawn beside it, as a reference only, so a reader can see how much the standardization subset
alone moves the number.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from margins.report.source import NumberSource
from margins.report.tables import ResultTable
from margins.viz.common import DESCRIPTIVE_SUFFIX, add_note, pair_label
from margins.viz.palette import CATEGORICAL, MUTED
from margins.viz.trace import AuditedPanel, Encoding, FigureSpec, decorate

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes

__all__ = ["build"]

NAME = "fig6_sensitivity"
CLAIM = (
    "Every free class x race interaction term, re-estimated across the pre-registered temperature "
    "grid and seed groups; the spread is drawn point by point, never summarized as a standard "
    "error over three or four groups."
)

#: The pooled-seeds fit of an arm (``margins.sensitivity.e4.POOLED``).
POOLED = "0"
#: Horizontal offset of each arm within a term's slot, in term-index units.
_DODGE = 0.24
_ARMS: tuple[tuple[float, str, str], ...] = (
    (0.0, "o", "greedy (T = 0)"),
    (0.7, "s", "T = 0.7 (E1's own responses)"),
    (1.0, "^", "T = 1.0"),
)


def _encodings() -> dict[float, Encoding]:
    return {
        temperature: Encoding(label=label, colour=CATEGORICAL[index], marker=marker)
        for index, (temperature, marker, label) in enumerate(_ARMS)
    }


HEADLINE = Encoding(label="headline E1 estimate (reference)", colour=MUTED, marker="x")


def _short(term: str) -> str:
    return " × ".join(part.split("=", 1)[1] for part in term.split(":"))


def _panel(
    ax: Axes,
    source: NumberSource,
    table: ResultTable,
    headline: ResultTable | None,
    *,
    model_id: str,
    harm: str,
    terms: tuple[str, ...],
    encodings: dict[float, Encoding],
) -> None:
    subset = table.where(model_id=model_id, harm_type=harm)
    for index, term in enumerate(terms):
        at_term = subset.where(term=term)
        for offset, (temperature, _, _) in zip((-_DODGE, 0.0, _DODGE), _ARMS, strict=True):
            arm = at_term.where(temperature=f"{temperature:.10g}")
            if not arm.rows:
                continue
            encoding = encodings[temperature]
            x = index + offset
            pooled = arm.where(seed_group=POOLED)
            if pooled.rows:
                estimate = source.floats(pooled, "estimate")[0]
                low = source.floats(pooled, "ci_low")[0]
                high = source.floats(pooled, "ci_high")[0]
                ax.vlines(x, low.value, high.value, color=encoding.colour, linewidth=0.9, zorder=2)
                ax.plot(
                    [x],
                    [estimate.value],
                    linestyle="none",
                    marker=encoding.marker,
                    markersize=3.6,
                    color=encoding.colour,
                    zorder=3,
                )
            seeds = arm.where(seed_group=[g for g in arm.unique("seed_group") if g != POOLED])
            if seeds.rows:
                values = source.floats(seeds, "estimate")
                ax.plot(
                    [x + 0.07] * len(values),
                    [value.value for value in values],
                    linestyle="none",
                    marker=encoding.marker,
                    markersize=2.0,
                    markerfacecolor="white",
                    markeredgecolor=encoding.colour,
                    markeredgewidth=0.5,
                    zorder=4,
                )
        if headline is not None:
            reference = headline.where(model_id=model_id, harm_type=harm, term=term)
            if reference.rows:
                value = source.floats(reference, "estimate")[0]
                ax.plot(
                    [index - _DODGE - 0.12],
                    [value.value],
                    linestyle="none",
                    marker=HEADLINE.marker,
                    markersize=3.0,
                    color=HEADLINE.colour,
                    zorder=3,
                )

    ax.set_xticks(range(len(terms)))
    ax.set_xticklabels([_short(term) for term in terms], rotation=60, ha="right", fontsize=5.6)
    ax.set_xlim(-0.6, len(terms) - 0.4)
    zero = ax.axhline(0.0, color=MUTED, linewidth=0.6, zorder=0)
    decorate(zero)


def build(source: NumberSource) -> FigureSpec:
    """Assemble the sensitivity figure: rows = harm types, columns = confirmatory models."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    table = source.table("sensitivity")
    models = sorted(table.unique("model_id"))
    harms = sorted(table.unique("harm_type"))
    first = table.rows[0]
    terms = tuple(
        sorted(
            table.unique("term"),
            key=lambda term: [part.split("=", 1)[1] for part in term.split(":")],
        )
    )
    downgraded = {
        row["harm_type"]
        for row in source.table("descriptive_only").rows
        if row["descriptive_only"] == "true"
    }
    effects = source.table("emergence_effects").where(
        axis_a=first["axis_a"], axis_b=first["axis_b"]
    )
    headline = effects if effects.rows else None
    encodings = _encodings()

    figure, axes = plt.subplots(
        len(harms), len(models), figsize=(7.2, 2.5 * len(harms) + 0.9), sharey="row", squeeze=False
    )
    for row, harm in enumerate(harms):
        for column, model_id in enumerate(models):
            ax = axes[row][column]
            ax.set_label(f"sensitivity-{model_id}-{harm}")
            _panel(
                ax,
                source,
                table,
                headline,
                model_id=model_id,
                harm=harm,
                terms=terms,
                encodings=encodings,
            )
            suffix = DESCRIPTIVE_SUFFIX if harm in downgraded else ""
            ax.set_title(
                f"{model_id.split('/')[-1]} · {harm.replace('_', ' ')}{suffix}",
                loc="left",
                fontsize=7.4,
            )
        axes[row][0].set_ylabel("interaction γ (SD)")

    used = [*encodings.values(), *([HEADLINE] if headline is not None else [])]
    handles = [
        Line2D(
            [], [], linestyle="none", marker=e.marker, color=e.colour, markersize=4, label=e.label
        )
        for e in used
    ]
    figure.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.945),
        ncol=2,
        fontsize=6.2,
        frameon=False,
    )
    figure.suptitle(
        f"How the {pair_label(first['axis_a'], first['axis_b'])} interaction moves with decoding",
        x=0.02,
        ha="left",
        fontweight="bold",
        fontsize=9.0,
    )
    add_note(
        axes[0][0],
        "Filled markers with bars: each arm's pooled estimate and cluster-bootstrap CI.\n"
        "Small open markers: the arm's individual seed groups (too few for a variance).",
        y=1.40,
    )
    figure.subplots_adjust(left=0.08, right=0.99, top=0.80, bottom=0.17, hspace=0.95, wspace=0.08)

    return FigureSpec(
        name=NAME,
        title="Temperature and seed sensitivity of the class x race interaction (E4)",
        claim=CLAIM,
        caption=(
            "Every free interaction term of the E4 pair, re-estimated per temperature arm (each "
            "standardized within itself) and per seed group, one panel per model x harm type. "
            "Bars are the pooled arm's 95% cluster-bootstrap CI; open points are seed groups; the "
            "grey cross is the headline E1 estimate, for reference. Descriptive: no p, no q."
        ),
        figure=figure,
        panels=tuple(AuditedPanel(ax, "y") for row in axes for ax in row),
        ledger=source.ledger,
        encodings=tuple(used),
        data_status=source.data_status,
    )
