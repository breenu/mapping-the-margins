"""B3: the structure-respecting permutation null, the one genuinely independent check (P9).

Section 10 is blunt about why this exists. The additive null (B2) is *the same quantity* as the
interaction coefficient, so it is a presentation of E1 rather than corroboration of it. The
permutation test is assumption-light, commits to no link function, and is cheap -- and it is the
only baseline that can disagree with the GLM.

**The shuffle must respect the structure or the null is a lie.** A naive global shuffle of axis
labels would break two things at once: it would split an item's sampled generations across
different labels, destroying the clustering the standard errors depend on, and it would let cells
empty out so that some replicates are not even identified. Both make the null distribution too
narrow, and a too-narrow null manufactures significance.

So the exchange unit is a **whole item**, with all of its generations moving together, and items
are permuted only **within a block**. A block is one ``(template frame, stratum-axis value)``
pair: inside it, the items differ only in the axis being permuted, so exchanging them is exactly
the null hypothesis of no interaction at that stratum. Items never cross a block, so every
replicate keeps every cell's item count and every replicate stays estimable.

Two schemes ship. ``within_stratum`` is section 10's wording made structural; it destroys the
permuted axis's main effect, which makes it **conservative** for the interaction rather than
anti-conservative. ``freedman_lane`` permutes item-level residuals of the additive ``A + B`` fit
and refits the full model, preserving both main effects exactly; it is the calibration oracle the
tests hold the default against.

The reported p is ``(1 + count) / (B + 1)``, which can never be zero -- a zero would assert
infinite evidence from a finite number of replicates. **It is reported beside the GLM p and never
enters the FDR family**: mixing an assumption-light second opinion into the corrected family
would double-count the same hypothesis.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from margins.metrics.design import PairDesign
from margins.metrics.estimator import PairFrame, lstsq_params
from margins.stats.config import PermutationScheme

__all__ = [
    "PermutationPlan",
    "permutation_p",
    "permutation_plan",
    "permute_interaction",
]


@dataclass(frozen=True, eq=False)
class PermutationPlan:
    """What may be exchanged with what, and which rows move together.

    ``blocks`` holds, per block, the indices into ``items`` that may be permuted among
    themselves. ``item_rows`` gives each item's row indices, so a permutation moves whole items.
    """

    blocks: tuple[tuple[int, ...], ...]
    item_rows: tuple[np.ndarray, ...]
    item_labels: tuple[str, ...]

    @property
    def n_items(self) -> int:
        return len(self.item_rows)

    @property
    def exchangeable_items(self) -> int:
        """Items sitting in a block with at least one other item. The rest never move."""
        return sum(len(block) for block in self.blocks if len(block) > 1)


def permutation_plan(
    frame: PairFrame,
    design: PairDesign,
    *,
    frame_of_item: Mapping[str, str],
    stratum_axis: str = "a",
) -> PermutationPlan:
    """Group items into exchangeable blocks.

    Args:
        frame: the observations, whose ``item`` column is the cluster key.
        design: the pair's design, used only to validate the stratum choice.
        frame_of_item: ``item -> template frame``. Two items of the same frame and the same
            stratum value differ only in the permuted axis, which is what makes them
            exchangeable under the null.
        stratum_axis: ``"a"`` or ``"b"``. The OTHER axis is the one whose labels move. Not
            symmetric, so it is pre-registered rather than chosen per run.
    """
    if stratum_axis not in ("a", "b"):
        raise ValueError(f"stratum_axis must be 'a' or 'b', got {stratum_axis!r}")
    if design.n_cells < 2:  # pragma: no cover - a design this small cannot be fitted anyway
        raise ValueError("a permutation needs a design with at least two cells")

    rows_by_item: dict[str, list[int]] = {}
    stratum_by_item: dict[str, str] = {}
    for index, item in enumerate(frame.item):
        rows_by_item.setdefault(item, []).append(index)
        stratum_by_item[item] = (
            frame.value_a[index] if stratum_axis == "a" else frame.value_b[index]
        )

    items = sorted(rows_by_item)
    missing = [item for item in items if item not in frame_of_item]
    if missing:
        raise KeyError(
            f"{len(missing)} item(s) have no template frame, e.g. {missing[:2]}. Blocks are "
            "(frame, stratum value) pairs; without the frame an item cannot be placed and "
            "permuting it across frames would exchange things that are not exchangeable."
        )

    grouped: dict[tuple[str, str], list[int]] = {}
    for position, item in enumerate(items):
        grouped.setdefault((frame_of_item[item], stratum_by_item[item]), []).append(position)

    return PermutationPlan(
        blocks=tuple(tuple(block) for _, block in sorted(grouped.items())),
        item_rows=tuple(np.asarray(rows_by_item[item], dtype=int) for item in items),
        item_labels=tuple(items),
    )


def permute_interaction(
    frame: PairFrame,
    design: PairDesign,
    plan: PermutationPlan,
    *,
    scheme: PermutationScheme,
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """The null distribution of the interaction terms. ``(n_permutations, n_interaction_terms)``.

    Under ``within_stratum`` the permutation relabels whole items inside their block, which is
    equivalent to permuting the non-stratum axis's labels within strata. Under ``freedman_lane``
    the additive ``A + B`` fit is removed first and its item-level residuals are permuted, so both
    main effects are preserved exactly in every replicate.
    """
    y = np.asarray(frame.y, dtype=float)
    matrix = design.matrix(frame.value_a, frame.value_b)
    contrasts = design.contrasts
    contrast_matrix = design.contrast_matrix
    is_interaction = np.array([c.kind == "interaction" for c in contrasts])
    interaction_rows = contrast_matrix[is_interaction]

    if scheme == "freedman_lane":
        # Additive model: drop the interaction columns.
        n_a, n_b = len(design.levels_a), len(design.levels_b)
        additive = matrix[:, : 1 + (n_a - 1) + (n_b - 1)]
        fitted = additive @ lstsq_params(additive, y)
        residual = y - fitted
        base = fitted
    else:
        residual = y
        base = np.zeros_like(y)

    out = np.empty((n_permutations, int(is_interaction.sum())))
    for replicate in range(n_permutations):
        permuted = np.arange(len(y))
        for block in plan.blocks:
            if len(block) < 2:
                continue
            order = rng.permutation(len(block))
            for source, target in zip(block, [block[i] for i in order], strict=True):
                # Move the SOURCE item's rows onto the TARGET item's row slots. Equal-length by
                # construction only when the design is balanced, so guard rather than assume.
                source_rows = plan.item_rows[source]
                target_rows = plan.item_rows[target]
                width = min(len(source_rows), len(target_rows))
                permuted[target_rows[:width]] = source_rows[:width]
        shuffled = base + residual[permuted]
        out[replicate] = interaction_rows @ lstsq_params(matrix, shuffled)
    return out


def permutation_p(observed: float, null: np.ndarray) -> float:
    """``(1 + #{|null| >= |observed|}) / (B + 1)``. Never zero.

    The ``+1`` is not a fudge: with ``B`` replicates the smallest supportable p is ``1/(B+1)``,
    and reporting 0 would claim more evidence than a finite resampling can provide.
    """
    array = np.abs(np.asarray(null, dtype=float))
    count = int(np.sum(array >= abs(observed)))
    return float((1 + count) / (len(array) + 1))
