"""Figure 5 -- RQ3: what cell-targeted red-teaming surfaces that static prompts do not (P10).

**The claim, and the label it must carry.** Red-team frames reach the same lattice cells as the
static set and produce an interaction vector over the same terms, so the two arms can be compared
directly. The comparison is **exploratory**: the red-team arm runs as a descriptive subset at
``powered=False``, stays out of ``family_benchmarks``, and carries no p and no q. That is a
budget-driven scope choice -- a powered red-team arm costs the same ~95 GPU-hours as the whole
main sweep -- and it is **never** evidence that red-team effects are small. The figure says so in
its own banner, not only in its caption.

**Why a scatter of the two vectors rather than a difference.** The same argument as the drift
figure: a difference of two interaction coefficients hides both of them, and the question here is
which cells move, not by how much on average. A point far above the diagonal is a cell the static
set scores as ordinary and the red-team set does not -- which is the entire content of RQ3.

**Why every frame is rendered into every cell, and why that matters here.** A red-team frame
elicited for one target cell is rendered across the whole grid (``margins.redteam.frames``). If
each cell were measured through its own private frames, any difference between frames -- length,
register, topic -- would land on that cell's mean and read as an identity effect, and the
estimator would report an interaction the frame distribution produced. The comparison this figure
draws is only meaningful because that confound was designed out upstream.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

import numpy as np

from margins.report.source import NumberSource
from margins.report.tables import ResultTable
from margins.viz.common import add_note, pair_label, term_label
from margins.viz.palette import CATEGORICAL, DESCRIPTIVE, INK, MUTED, RULE
from margins.viz.trace import AuditedPanel, Encoding, FigureSpec, decorate

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes

__all__ = ["RedTeamPoweredError", "build"]

NAME = "fig5_redteam_vs_static"
CLAIM = (
    "Cell-targeted red-teaming produces an interaction vector over the same terms as the static "
    "set, and surfaces cells the static set does not. The arm is a descriptive subset by "
    "pre-registered budget decision: it carries confidence intervals and no significance claim."
)

_STATIC: Final[str] = "open_gen_templates"
_REDTEAM: Final[str] = "redteam"
_N_EXAMPLES: Final[int] = 3


class RedTeamPoweredError(AssertionError):
    """A red-team row arrived claiming a significance claim the pre-registration withholds."""


def _check_descriptive(table: ResultTable) -> None:
    offenders = [row for row in table.rows if row["claimable"] == "true" or row["q"] != ""]
    if offenders:
        raise RedTeamPoweredError(
            f"{len(offenders)} red-team rows carry a q or a claimable flag. The red-team arm is "
            "a descriptive subset at powered=False and is deliberately absent from "
            "family_benchmarks (CLAUDE.md, 21 Sep 2026). A figure must not promote it."
        )


def _paired(
    static: ResultTable, redteam: ResultTable
) -> tuple[tuple[Mapping[str, str], ...], tuple[Mapping[str, str], ...]]:
    """Rows of the two arms matched on ``(harm, pair, term)``, in a stable order."""
    key = ("harm_type", "axis_a", "axis_b", "term")
    by_key = {tuple(row[column] for column in key): row for row in redteam.rows}
    matched = [
        (row, by_key[tuple(row[column] for column in key)])
        for row in static.rows
        if tuple(row[column] for column in key) in by_key
    ]
    matched.sort(key=lambda pair: tuple(pair[0][column] for column in key))
    return tuple(row for row, _ in matched), tuple(row for _, row in matched)


def _vector_scatter(
    ax: Axes,
    source: NumberSource,
    static: ResultTable,
    redteam: ResultTable,
    encodings: dict[str, Encoding],
) -> list[tuple[str, float, float]]:
    """The two arms' interaction vectors against each other. Returns the surfaced examples."""
    static_rows, redteam_rows = _paired(static, redteam)
    examples: list[tuple[str, float, float]] = []

    for harm in sorted({row["harm_type"] for row in static_rows}):
        indices = [i for i, row in enumerate(static_rows) if row["harm_type"] == harm]
        static_subset = ResultTable(
            static.spec, tuple(static_rows[i] for i in indices), static.origin
        )
        redteam_subset = ResultTable(
            redteam.spec, tuple(redteam_rows[i] for i in indices), redteam.origin
        )
        x = source.floats(static_subset, "estimate")
        y = source.floats(redteam_subset, "estimate")
        encoding = encodings[harm]
        ax.plot(
            [value.value for value in x],
            [value.value for value in y],
            linestyle="none",
            marker=encoding.marker,
            markersize=3.4,
            markerfacecolor="white",
            markeredgecolor=encoding.colour,
            markeredgewidth=0.8,
            label=encoding.label,
            zorder=3,
        )
        for index, (left, right) in enumerate(zip(x, y, strict=True)):
            row = redteam_subset.rows[index]
            examples.append(
                (
                    f"{pair_label(row['axis_a'], row['axis_b'])} · "
                    f"{term_label(row['value_a'], row['value_b'])} · {harm}",
                    left.value,
                    right.value,
                )
            )

    limit = source.derived(
        max(abs(value) for _, left, right in examples for value in (left, right)) * 1.12,
        key="redteam_axis_limit",
        origin=redteam.origin,
        how="1.12 x the largest |gamma| in either arm, so the diagonal spans the data",
    )
    diagonal = ax.plot(
        [-limit.value, limit.value],
        [-limit.value, limit.value],
        color=RULE,
        linestyle=":",
        linewidth=0.8,
        zorder=1,
    )
    decorate(diagonal)
    for rule in (
        ax.axhline(0.0, color=MUTED, linewidth=0.6, zorder=0),
        ax.axvline(0.0, color=MUTED, linewidth=0.6, zorder=0),
    ):
        decorate(rule)
    ax.set_xlim(-limit.value, limit.value)
    ax.set_ylim(-limit.value, limit.value)
    ax.set_aspect("equal")
    ax.set_xlabel("static templated set — γ (SD)")
    ax.set_ylabel("cell-targeted red-team set — γ (SD)")
    ax.set_title("The two arms' interaction vectors", loc="left")
    label = ax.text(
        -limit.value * 0.95,
        limit.value * 0.86,
        "above the diagonal:\nred-teaming surfaces more",
        fontsize=6.0,
        color=INK,
        va="top",
    )
    label.set_gid(None)
    ax.legend(loc="lower right", fontsize=6.2, borderaxespad=0.3)

    examples.sort(key=lambda item: abs(item[2]) - abs(item[1]), reverse=True)
    return examples[:_N_EXAMPLES]


def _rms_bars(
    ax: Axes, source: NumberSource, reports: ResultTable, encodings: dict[str, Encoding]
) -> None:
    """Per-pair interaction magnitude, static beside red-team. Descriptive, no correction."""
    arms = (_STATIC, _REDTEAM)
    # Matched pairs only. E3 is a comparison on shared cells, and the red-team arm is restricted
    # to the confirmatory pairs by construction, so an unmatched pair would put a bar beside
    # nothing and invite the reading that the other arm measured zero there.
    pairs = sorted(
        set.intersection(
            *(
                {(row["axis_a"], row["axis_b"]) for row in reports.where(benchmark=arm).rows}
                for arm in arms
            )
        )
    )
    positions = np.arange(len(pairs))
    height = 0.34
    for index, arm in enumerate(arms):
        values: list[float] = []
        for axis_a, axis_b in pairs:
            subset = reports.where(benchmark=arm, axis_a=axis_a, axis_b=axis_b)
            rms = source.floats(subset, "rms_interaction")
            values.append(
                source.derived(
                    sum(value.value for value in rms) / len(rms),
                    key=f"rms_mean[{arm}|{axis_a}x{axis_b}]",
                    origin=reports.origin,
                    how="mean RMS interaction over the arm's harm types for this pair",
                ).value
            )
        encoding = encodings[arm]
        ax.barh(
            positions + (index - 0.5) * height,
            values,
            height=height * 0.9,
            color=encoding.colour if arm == _STATIC else "white",
            edgecolor=encoding.colour,
            linewidth=0.8,
            hatch=encoding.hatch,
            label=encoding.label,
            zorder=2,
        )
    ax.set_yticks(positions)
    ax.set_yticklabels([pair_label(a, b) for a, b in pairs], fontsize=6.2)
    ax.invert_yaxis()
    ax.set_xlabel("RMS interaction magnitude (SD)")
    ax.set_title("Per-pair magnitude, both arms", loc="left")
    ax.legend(
        loc="upper left", bbox_to_anchor=(0.0, -0.30), ncol=2, fontsize=6.2, borderaxespad=0.0
    )


def _examples(ax: Axes, examples: list[tuple[str, float, float]]) -> None:
    """The cells red-teaming surfaced, named. An effect vector needs worked examples.

    The values are the ones already handed out by the ledger for the scatter, so naming them here
    adds no untraced number -- the panel re-prints points the figure has already drawn.
    """
    from matplotlib.patches import Rectangle

    ax.axis("off")
    band = ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            1.0,
            1.0,
            transform=ax.transAxes,
            facecolor=DESCRIPTIVE,
            edgecolor=MUTED,
            linewidth=0.7,
            linestyle="--",
            zorder=0,
            clip_on=False,
        )
    )
    decorate(band)
    header = ax.text(
        0.03,
        0.92,
        "EXPLORATORY — cells the red-team arm surfaces most",
        fontsize=7.0,
        fontweight="bold",
        color=INK,
        va="top",
    )
    header.set_gid(None)
    for index, (label, static_value, redteam_value) in enumerate(examples):
        line = ax.text(
            0.03,
            0.70 - index * 0.24,
            f"{label}\n      static γ {static_value:+.2f}      red-team γ {redteam_value:+.2f}",
            fontsize=6.2,
            color=INK,
            va="top",
            ha="left",
            multialignment="left",
        )
        line.set_gid(None)


def build(source: NumberSource, *, model_id: str) -> FigureSpec:
    """Assemble figure 5 from the committed tables. Descriptive by construction."""
    import matplotlib.pyplot as plt

    effects = source.table("emergence_effects").where(model_id=model_id, kind="interaction")
    static = effects.where(benchmark=_STATIC)
    redteam = effects.where(benchmark=_REDTEAM)
    _check_descriptive(redteam)

    reports = source.table("fit_reports").where(
        model_id=model_id, benchmark=(_STATIC, _REDTEAM), estimable="true"
    )

    encodings = {
        "regard": Encoding("regard", CATEGORICAL[0], marker="o"),
        "toxicity": Encoding("toxicity", CATEGORICAL[1], marker="s"),
        _STATIC: Encoding("static templated set", CATEGORICAL[0], linestyle="-"),
        _REDTEAM: Encoding("cell-targeted red-team set", CATEGORICAL[4], hatch="\\\\\\"),
    }

    figure = plt.figure(figsize=(7.2, 4.6))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.0, 1.1),
        height_ratios=(1.0, 0.85),
        wspace=0.34,
        hspace=0.62,
        left=0.09,
        right=0.98,
        top=0.86,
        bottom=0.22,
    )
    scatter_ax = figure.add_subplot(grid[:, 0], label="vectors")
    bars_ax = figure.add_subplot(grid[0, 1], label="rms")
    examples_ax = figure.add_subplot(grid[1, 1], label="examples")

    surfaced = _vector_scatter(scatter_ax, source, static, redteam, encodings)
    _rms_bars(bars_ax, source, reports, encodings)
    _examples(examples_ax, surfaced)

    add_note(
        scatter_ax,
        "Every frame is rendered into EVERY target cell, not only the cell it was elicited for.\n"
        "Confining a frame to its target would confound 'frame aimed at X' with 'cell X', and the\n"
        "estimator would report an interaction the frame distribution produced.",
        y=-0.16,
    )
    figure.suptitle(
        "Red-teaming reaches cells the static set does not — reported descriptively",
        x=0.02,
        ha="left",
        fontweight="bold",
    )

    return FigureSpec(
        name=NAME,
        title="Cell-targeted red-teaming against the static templated set",
        claim=CLAIM,
        caption=(
            "Left: the two arms' interaction coefficients over matched terms; a point above the "
            "diagonal is a cell the red-team set scores higher than the static set. Top right: "
            "per-pair RMS interaction magnitude for both arms. Bottom right: the cells red-teaming "
            "surfaces most, named. The red-team arm runs as a descriptive subset at powered=False "
            "and is absent from the FDR family by a budget decision, so nothing here carries a p "
            "or a q. That is a scope choice and not evidence that red-team effects are small."
        ),
        figure=figure,
        panels=(
            AuditedPanel(scatter_ax, "both"),
            AuditedPanel(bars_ax, "x"),
            AuditedPanel(examples_ax, "x"),
        ),
        ledger=source.ledger,
        encodings=tuple(encodings.values()),
        data_status=source.data_status,
    )
