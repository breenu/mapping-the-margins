"""Figure 1 -- RQ2: emergent intersectional bias is the interaction (P10).

**The claim.** For the pre-registered confirmatory axis pairs, the departure from additivity --
the interaction coefficient gamma -- is estimated with a cluster-bootstrap interval, corrected
over the whole family, and is non-zero at some cells and not others. That is emergence. A high
harm *level* at an intersection is not.

**Why there is no level anywhere on this figure.** Section 10 names conflating a level with an
interaction as the single most common way intersectional audits overclaim, and the project split
:class:`~margins.schemas.CellStat` from :class:`~margins.schemas.EffectEstimate` so the two could
not be confused in the data. Drawing a cell mean here would undo that split in the one artifact a
reader actually looks at. The heatmap panel is a gamma table -- it sums to zero along every row
and column by construction, which a table of levels never does.

**Three things the figure must show and does.**

* *Which effects may be claimed.* Significance markers come from :mod:`margins.viz.claims`, which
  runs the real :class:`~margins.stats.claims.ClaimGate`. A descriptive-only harm type keeps its
  q and gets no star -- the figure draws the refusal rather than hiding the row.
* *Which are load-bearing.* Powered, claimable series are drawn in the categorical palette with
  filled markers; refused series are muted, open, and labelled ``DESCRIPTIVE-ONLY`` in the legend
  itself, so a reader who never reaches the caption still cannot mistake them for evidence.
* *That the family is the family.* Only the ``(|A|-1)(|B|-1)`` free contrasts of the confirmatory
  pairs are eligible for a star, because the emitted gamma set is linearly dependent and the
  pre-registration corrects over the free ones.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from margins.report.source import NumberSource, Traced
from margins.report.tables import ResultTable
from margins.viz.claims import RowClaim, exploratory_pairs_from_lattice, significance
from margins.viz.common import (
    add_note,
    encoding_for_harm,
    grouped_positions,
    pair_label,
    term_label,
    zero_rule,
)
from margins.viz.palette import DIVERGING, INK, MUTED, RULE
from margins.viz.trace import AuditedPanel, Encoding, FigureSpec, decorate

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["build"]

NAME = "fig1_emergence"
CLAIM = (
    "Emergent intersectional bias is the interaction term of harm ~ A + B + A:B, and for the "
    "pre-registered confirmatory pairs some of those terms are non-zero after FDR correction "
    "while the harm levels they sit at are not the evidence."
)


def _descriptive_only(source: NumberSource, benchmark: str, model_id: str) -> frozenset[str]:
    table = source.table("descriptive_only").where(benchmark=benchmark, model_id=model_id)
    return frozenset(row["harm_type"] for row in table.rows if row["descriptive_only"] == "true")


def _forest(
    ax: Axes,
    source: NumberSource,
    rows_by_harm: Mapping[str, ResultTable],
    claims_by_harm: Mapping[str, tuple[RowClaim, ...]],
    encodings: Mapping[str, Encoding],
) -> None:
    """All free interaction contrasts, grouped by axis pair, one series per harm type."""
    reference = next(iter(rows_by_harm.values()))
    groups = [pair_label(row["axis_a"], row["axis_b"]) for row in reference.rows]
    positions, spans = grouped_positions(groups)
    labels = [term_label(row["value_a"], row["value_b"]) for row in reference.rows]

    offsets = np.linspace(-0.22, 0.22, len(rows_by_harm)) if len(rows_by_harm) > 1 else [0.0]
    for offset, (harm, table) in zip(offsets, rows_by_harm.items(), strict=True):
        encoding = encodings[harm]
        estimates = source.floats(table, "estimate")
        lows = source.floats(table, "ci_low")
        highs = source.floats(table, "ci_high")
        claims = claims_by_harm[harm]
        # Filled markers are load-bearing evidence; open markers are description. The status is
        # a property of the harm type, so it is uniform across the series by construction.
        load_bearing = all(claim.load_bearing for claim in claims)
        y = [positions[index] + offset for index in range(len(table.rows))]
        x = [value.value for value in estimates]
        ax.errorbar(
            x,
            y,
            xerr=np.vstack(
                [
                    [e.value - lo.value for e, lo in zip(estimates, lows, strict=True)],
                    [hi.value - e.value for e, hi in zip(estimates, highs, strict=True)],
                ]
            ),
            fmt=encoding.marker or "o",
            markersize=3.0,
            markerfacecolor=encoding.colour if load_bearing else "white",
            markeredgecolor=encoding.colour,
            markeredgewidth=0.8,
            color=encoding.colour,
            ecolor=encoding.colour,
            elinewidth=0.7,
            capsize=1.2,
            linestyle="none",
            label=encoding.label,
            zorder=3,
        )
        for index, claim in enumerate(claims):
            if not claim.significant:
                continue
            star = ax.text(
                highs[index].value + 0.03,
                positions[index] + offset,
                "*",
                ha="left",
                va="center",
                fontsize=9.0,
                color=encoding.colour,
            )
            star.set_gid(None)

    ax.set_yticks([positions[index] for index in range(len(labels))])
    ax.set_yticklabels(labels, fontsize=5.6)
    ax.invert_yaxis()
    ax.set_xlabel("interaction coefficient γ  (SD of the standardized harm score)")
    ax.set_title("Departure from additivity, per confirmatory cell", loc="left")
    zero_rule(ax)

    for group, (first, last) in spans.items():
        mid = (first + last) / 2.0
        text = ax.text(
            -0.66,
            mid,
            group,
            transform=ax.get_yaxis_transform(),
            ha="center",
            va="center",
            fontsize=7.0,
            color=INK,
            rotation=90,
        )
        decorate(text)
        rule = ax.axhline(last + 0.5, color=RULE, linewidth=0.4, alpha=0.6, zorder=0)
        decorate(rule)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, -0.055),
        ncol=2,
        handletextpad=0.4,
        columnspacing=1.4,
        borderaxespad=0.0,
    )


def _heatmap(
    figure: Figure,
    ax: Axes,
    source: NumberSource,
    table: ResultTable,
    *,
    harm_type: str,
) -> None:
    """The gamma table of one pair. Rows and columns sum to zero; a level table never would."""
    levels_a = table.unique("value_a")
    levels_b = table.unique("value_b")
    grid = np.full((len(levels_a), len(levels_b)), np.nan)
    traced: list[Traced] = []
    for row in table.rows:
        value = source.cell(table, row, "estimate")
        traced.append(value)
        grid[levels_a.index(row["value_a"]), levels_b.index(row["value_b"])] = value.value

    limit = float(np.nanmax(np.abs(grid)))
    image = ax.imshow(grid, cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(levels_b)))
    ax.set_xticklabels(levels_b, rotation=35, ha="right", fontsize=6.0)
    ax.set_yticks(range(len(levels_a)))
    ax.set_yticklabels(levels_a, fontsize=6.0)
    for i in range(len(levels_a)):
        for j in range(len(levels_b)):
            cell = ax.text(
                j,
                i,
                f"{grid[i, j]:+.2f}",
                ha="center",
                va="center",
                fontsize=5.6,
                color="white" if abs(grid[i, j]) > 0.62 * limit else INK,
            )
            cell.set_gid(None)
    axis_a = table.strings("axis_a")[0]
    axis_b = table.strings("axis_b")[0]
    ax.set_title(f"γ table — {pair_label(axis_a, axis_b)} ({harm_type})", loc="left")
    bar = figure.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    bar.ax.tick_params(labelsize=6.0)
    bar.set_label("γ (SD)", fontsize=6.5)


def _survival(
    ax: Axes,
    source: NumberSource,
    per_pair: Mapping[str, tuple[int, int]],
    origin: str,
    encoding: Encoding,
) -> None:
    """How many of each pair's family terms survived the correction. A count, derived and traced."""
    pairs = list(per_pair)
    eligible = [
        source.derived(
            per_pair[pair][1],
            key=f"family_terms[{pair}]",
            origin=origin,
            how="rows with family_member=true",
        ).value
        for pair in pairs
    ]
    survived = [
        source.derived(
            per_pair[pair][0],
            key=f"significant_terms[{pair}]",
            origin=origin,
            how="rows the ClaimGate permits with q <= alpha",
        ).value
        for pair in pairs
    ]
    y = np.arange(len(pairs))
    ax.barh(
        y,
        eligible,
        color="white",
        edgecolor=MUTED,
        linewidth=0.7,
        hatch="///",
        height=0.62,
        label="free contrasts in the FDR family",
    )
    ax.barh(
        y,
        survived,
        color=encoding.colour,
        edgecolor=encoding.colour,
        height=0.62,
        label="survive BH correction",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(pairs, fontsize=6.0)
    ax.invert_yaxis()
    ax.set_xlabel("free interaction contrasts")
    ax.set_title("Family membership and survival", loc="left")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, -0.22),
        fontsize=6.0,
        handlelength=1.6,
        borderaxespad=0.0,
    )


def build(
    source: NumberSource, *, benchmark: str, model_id: str, alpha_key: str = "fdr.alpha"
) -> FigureSpec:
    """Assemble figure 1 from the committed tables."""
    import matplotlib.pyplot as plt

    alpha = source.number(alpha_key)
    downgraded = _descriptive_only(source, benchmark, model_id)
    exploratory = exploratory_pairs_from_lattice()
    descriptive_keys = frozenset((benchmark, model_id, harm) for harm in downgraded)

    effects = source.table("emergence_effects").where(
        benchmark=benchmark, model_id=model_id, kind="interaction", family_member="true"
    )
    harms = effects.unique("harm_type")
    rows_by_harm: dict[str, ResultTable] = {}
    claims_by_harm: dict[str, tuple[RowClaim, ...]] = {}
    for harm in harms:
        table = effects.where(harm_type=harm).sorted_by("axis_a", "axis_b", "term")
        rows_by_harm[harm] = table
        claims_by_harm[harm] = significance(
            table.rows,
            alpha=alpha.value,
            exploratory_pairs=exploratory,
            descriptive_only=descriptive_keys,
        )

    encodings = {
        harm: encoding_for_harm(harm, descriptive_only=harm in downgraded) for harm in harms
    }

    figure = plt.figure(figsize=(7.2, 8.0))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.45, 1.0),
        height_ratios=(1.3, 1.0),
        wspace=0.46,
        hspace=0.52,
        left=0.17,
        right=0.97,
        top=0.93,
        bottom=0.145,
    )
    forest_ax = figure.add_subplot(grid[:, 0], label="forest")
    heat_ax = figure.add_subplot(grid[0, 1], label="gamma")
    survival_ax = figure.add_subplot(grid[1, 1], label="survival")

    _forest(forest_ax, source, rows_by_harm, claims_by_harm, encodings)

    load_bearing = next((harm for harm in harms if harm not in downgraded), harms[0])
    flagship = _flagship_pair(rows_by_harm[load_bearing])
    heat_rows = (
        source.table("emergence_effects")
        .where(
            benchmark=benchmark,
            model_id=model_id,
            kind="interaction",
            harm_type=load_bearing,
            axis_a=flagship[0],
            axis_b=flagship[1],
        )
        .sorted_by("value_a", "value_b")
    )
    _heatmap(figure, heat_ax, source, heat_rows, harm_type=load_bearing)

    per_pair = _survival_counts(rows_by_harm[load_bearing], claims_by_harm[load_bearing])
    _survival(
        survival_ax, source, per_pair, rows_by_harm[load_bearing].origin, encodings[load_bearing]
    )

    add_note(
        forest_ax,
        "A star marks a term the claim gate permits AND that survives Benjamini-Hochberg\n"
        "over the whole pre-registered family. A refused series keeps every estimate and\n"
        "every interval and can never be starred: the gate raises rather than returning false.",
        y=-0.105,
    )
    figure.suptitle(
        "Emergence is the interaction, not the level", x=0.02, ha="left", fontweight="bold"
    )

    return FigureSpec(
        name=NAME,
        title="Emergent intersectional bias per confirmatory axis pair",
        claim=CLAIM,
        caption=(
            "Interaction coefficients of harm ~ A + B + A:B over each confirmatory pair's order-2 "
            "cells, with cluster-bootstrap intervals. Only the free contrasts enter the FDR "
            "family. Harm types the scorer identity-drift rule downgraded are drawn muted and "
            "carry no significance marker regardless of their q. The right panel is a gamma "
            "table, not a table of harm levels: its rows and columns sum to zero by construction."
        ),
        figure=figure,
        panels=(
            AuditedPanel(forest_ax, "x"),
            AuditedPanel(heat_ax, "array"),
            AuditedPanel(survival_ax, "x"),
        ),
        ledger=source.ledger,
        encodings=(
            *encodings.values(),
            Encoding("free contrasts in the FDR family", MUTED, hatch="///"),
            Encoding("survive BH correction", encodings[load_bearing].colour, linestyle="-"),
        ),
        data_status=source.data_status,
    )


def _flagship_pair(table: ResultTable) -> tuple[str, str]:
    """The pair whose grid the heatmap shows.

    Chosen **pre-registered**, not by the results: it is the pair the power analysis was run on
    (``power.pair`` in ``configs/stats/``), which is the lattice's primary non-gender x
    non-gender pair, its largest grid, and the pair the intraclass correlation was measured on.
    Picking the pair with the largest interaction instead would be choosing a figure after seeing
    the data. If that pair is absent from this arm the rule falls back to the largest grid, ties
    broken alphabetically.
    """
    from margins.stats.config import DEFAULT_STATS, load_stats_config

    present = {(row["axis_a"], row["axis_b"]) for row in table.rows}
    declared = load_stats_config(DEFAULT_STATS).power.pair
    if declared in present:
        return declared
    sizes: dict[tuple[str, str], int] = {}
    for row in table.rows:
        sizes[(row["axis_a"], row["axis_b"])] = sizes.get((row["axis_a"], row["axis_b"]), 0) + 1
    return max(sorted(sizes), key=lambda pair: sizes[pair])


def _survival_counts(table: ResultTable, claims: Sequence[RowClaim]) -> dict[str, tuple[int, int]]:
    """``{pair label: (significant, eligible)}`` in table order."""
    out: dict[str, tuple[int, int]] = {}
    for row, claim in zip(table.rows, claims, strict=True):
        label = pair_label(row["axis_a"], row["axis_b"])
        significant, eligible = out.get(label, (0, 0))
        out[label] = (significant + int(claim.significant), eligible + 1)
    return out
