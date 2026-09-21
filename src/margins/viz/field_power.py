"""Figure 3 -- the field result: conventional designs cannot see a moderate interaction (P10).

**The claim, and it is a contribution rather than a methods footnote.** At the intraclass
correlation this project *measured*, a conventional generative audit -- nine prompts per cell,
a handful of samples each -- reaches between 18% and 50% power against a **moderate** 0.5 SD
interaction, through the real estimator and the real Benjamini-Hochberg correction. Below that
power, "no significant interaction" is the expected outcome whether or not an interaction exists,
and a reported significant one is at elevated risk of being noise.

**Every number here is real and measured**, unlike several of the other P10 figures: rho comes
from 1,728 sampled generations of the class x race pair, and the power figures come from the
simulation in ``scripts/run_power_analysis.py``. Both are registered in
``results/paper_numbers.json`` under ``rho.*``, ``field.power_at_9_prompts.*`` and
``power.ceiling_at_27_prompts.*``, and this module reads them by key.

**Why the ceiling is drawn beside the conventional bars.** The finding is not "spend more
compute". Precision floors at ``rho / m`` for ``m`` prompts per cell, so a nine-prompt design is
bounded no matter how many generations it draws; the adopted 27-prompt design clears the target at
every measured rho including the upper bounds. Drawing only the conventional bars would invite the
reading that the field needs a bigger budget. It needs a different allocation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np

from margins.report.source import NumberSource, Traced
from margins.viz.common import add_note
from margins.viz.palette import CATEGORICAL, INK, MUTED, RULE
from margins.viz.trace import AuditedPanel, Encoding, FigureSpec, decorate

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes

__all__ = ["build"]

NAME = "fig3_field_underpowered"
CLAIM = (
    "At the measured intraclass correlation, a conventional nine-prompt-per-cell generative audit "
    "reaches 18-50% power on a moderate 0.5 SD interaction. Null results from such designs are "
    "close to uninformative, and the binding resource is prompt count, not compute."
)

#: ``(rho, paper-number suffix, harm label)``. The two measured values and their upper confidence
#: bounds, which is what the design was chosen against.
_RHO_POINTS: Final[tuple[tuple[float, str, str], ...]] = (
    (0.20, "rho20", "toxicity\nρ = 0.20"),
    (0.23, "rho23", "toxicity, upper\nρ = 0.23"),
    (0.35, "rho35", "regard\nρ = 0.35"),
    (0.43, "rho43", "regard, upper\nρ = 0.43"),
)

_CONVENTIONAL: Final[tuple[tuple[int, str], ...]] = (
    (27, "9 prompts × 3 samples  (27 renderings/cell)"),
    (45, "9 prompts × 5 samples  (45 renderings/cell)"),
)


def _power_bars(ax: Axes, source: NumberSource, encodings: dict[str, Encoding]) -> None:
    positions = np.arange(len(_RHO_POINTS))
    width = 0.26

    series: list[tuple[str, list[Traced]]] = []
    for renderings, label in _CONVENTIONAL:
        series.append(
            (
                label,
                [
                    source.number(f"field.power_at_9_prompts.r{renderings}_{suffix}")
                    for _, suffix, _ in _RHO_POINTS
                ],
            )
        )
    ceiling = [
        source.number(f"power.ceiling_at_27_prompts.{suffix}") for _, suffix, _ in _RHO_POINTS
    ]
    series.append(("27 prompts/cell — the adopted design", ceiling))

    for index, (label, values) in enumerate(series):
        encoding = encodings[label]
        ax.bar(
            positions + (index - 1) * width,
            [value.value for value in values],
            width=width * 0.92,
            color=encoding.colour if index == len(series) - 1 else "white",
            edgecolor=encoding.colour,
            linewidth=0.9,
            hatch=encoding.hatch,
            label=label,
            zorder=2,
        )

    target = source.number("power.target_power")
    rule = ax.axhline(target.value, color=INK, linewidth=0.9, linestyle="--", zorder=3)
    decorate(rule)
    ax.text(
        len(_RHO_POINTS) - 0.45,
        target.value + 0.015,
        f"pre-registered target power {target.value:g}",
        ha="right",
        va="bottom",
        fontsize=6.2,
        color=INK,
    ).set_gid(None)

    effect = source.number("power.effect_size")
    ax.set_xticks(positions)
    ax.set_xticklabels([label for _, _, label in _RHO_POINTS], fontsize=6.4)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(f"simulated power on a {effect.value:g} SD interaction")
    ax.set_title("What each design can detect", loc="left")
    ax.grid(axis="y", zorder=0)
    ax.legend(
        loc="upper left", bbox_to_anchor=(0.0, -0.16), ncol=1, fontsize=6.4, borderaxespad=0.0
    )


def _rho_panel(ax: Axes, source: NumberSource, encodings: dict[str, Encoding]) -> None:
    """The measured intraclass correlation with its exact F interval. One pair, stated as one."""
    harms = ("regard", "toxicity")
    positions = np.arange(len(harms))
    for index, harm in enumerate(harms):
        point = source.number(f"rho.{harm}")
        low, high = source.interval(f"rho.{harm}")
        encoding = encodings[harm]
        ax.errorbar(
            [point.value],
            [positions[index]],
            xerr=[[point.value - low.value], [high.value - point.value]],
            fmt=encoding.marker or "o",
            markersize=4.5,
            color=encoding.colour,
            markerfacecolor=encoding.colour,
            ecolor=encoding.colour,
            elinewidth=1.0,
            capsize=2.0,
            linestyle="none",
            label=encoding.label,
        )
        annotation = ax.text(
            high.value + 0.012,
            positions[index],
            f"{point.value:.2f}  [{low.value:.2f}, {high.value:.2f}]",
            va="center",
            fontsize=6.2,
            color=INK,
        )
        annotation.set_gid(None)

    generations = source.number("rho.n_generations")
    pairs = source.number("rho.measured_on_pairs")
    ax.set_yticks(positions)
    ax.set_yticklabels(harms, fontsize=7.0)
    ax.invert_yaxis()
    ax.set_ylim(len(harms) - 0.4, -0.6)
    ax.set_xlim(0.0, 0.62)
    ax.set_xlabel("intraclass correlation ρ\nvariance between prompts within a cell")
    ax.set_title("ρ was measured, not assumed", loc="left")
    add_note(
        ax,
        f"{generations.value:.0f} sampled generations of {pairs.value:.0f} axis pair — the\n"
        "lattice's largest grid — through the pinned confirmatory\n"
        "checkpoint, quantized. NOT assumed uniform across pairs: the\n"
        "design was chosen against the upper bound, the row that binds.",
        y=-0.30,
    )


def build(source: NumberSource) -> FigureSpec:
    """Assemble figure 3 entirely from ``results/paper_numbers.json``."""
    import matplotlib.pyplot as plt

    encodings = {
        _CONVENTIONAL[0][1]: Encoding(_CONVENTIONAL[0][1], MUTED, hatch="///"),
        _CONVENTIONAL[1][1]: Encoding(_CONVENTIONAL[1][1], RULE, hatch="..."),
        "27 prompts/cell — the adopted design": Encoding(
            "27 prompts/cell — the adopted design", CATEGORICAL[0], linestyle="-"
        ),
        "regard": Encoding("regard", CATEGORICAL[0], marker="o"),
        "toxicity": Encoding("toxicity", CATEGORICAL[1], marker="s"),
    }

    figure = plt.figure(figsize=(7.2, 4.6))
    grid = figure.add_gridspec(
        1, 2, width_ratios=(1.55, 1.0), wspace=0.30, left=0.09, right=0.98, top=0.86, bottom=0.30
    )
    bars_ax = figure.add_subplot(grid[0, 0], label="power")
    rho_ax = figure.add_subplot(grid[0, 1], label="rho")

    _power_bars(bars_ax, source, encodings)
    _rho_panel(rho_ax, source, encodings)

    low = source.number("field.power_at_9_prompts.min")
    high = source.number("field.power_at_9_prompts.max")
    figure.suptitle(
        f"A conventional intersectional audit reaches {low.value:.0%}–{high.value:.0%} power "
        "on a moderate effect",
        x=0.02,
        ha="left",
        fontweight="bold",
    )
    add_note(
        bars_ax,
        "Precision floors at ρ/m for m prompts per cell, so the hollow bars are\n"
        "ceilings no sampling budget beats. The filled bars are the adopted design\n"
        "at the same alpha, estimator and correction: what made the target\n"
        "reachable was allocation, not compute.",
        y=-0.39,
    )
    return FigureSpec(
        name=NAME,
        title="Power of conventional generative audits at the measured rho",
        claim=CLAIM,
        caption=(
            "Simulated power against a 0.5 SD single-cell interaction, run through the project's "
            "own estimator and its Benjamini-Hochberg correction over the pre-registered family. "
            "Hollow bars are conventional nine-prompt designs at two rendering budgets; filled "
            "bars are the ceiling of the adopted 27-prompt design. Right: the intraclass "
            "correlation measured on one axis pair, with the exact F interval the design was "
            "chosen against. Every value is a key in results/paper_numbers.json."
        ),
        figure=figure,
        panels=(AuditedPanel(bars_ax, "y"), AuditedPanel(rho_ax, "x")),
        ledger=source.ledger,
        encodings=tuple(encodings.values()),
        data_status="real",
    )
