"""The one operation E4 and R2 share: fit a pair's grid with P7's own metric.

**Why P7's metric and not a call to the estimator.** :class:`EmergenceMetric` is what produced
every headline number: it triangulates the harm type's scorers into ``harm_std``, clusters by
prompt, bootstraps within cell, and decides ``powered`` from the actual n. Re-implementing any of
that here would make an E4 or R2 estimate differ from the headline one for a reason that is not
the change being studied.

**Why only the pair's cells are passed.** The metric fits every axis pair present among the cells
it is given. Giving it the pair's grid alone means it fits that pair and nothing else, rather than
reporting six other pairs as "not estimable" on data that was never meant to cover them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from margins.metrics.config import MetricsConfig
from margins.metrics.design import PairDesign, pair_design
from margins.metrics.metric import EmergenceMetric, EmergenceResult
from margins.schemas import IdentityCell, ItemScore
from margins.stats.family import free_interaction_terms

__all__ = ["fit_pair", "free_terms", "restrict_to_cells"]


def restrict_to_cells(
    scores: Iterable[ItemScore], cells: Sequence[IdentityCell]
) -> list[ItemScore]:
    """The MODEL-condition rows whose cell is one of ``cells``. Order is preserved."""
    wanted = {cell.cell_id for cell in cells}
    return [s for s in scores if s.condition == "model" and s.cell_id in wanted]


def _design(cells: Sequence[IdentityCell]) -> PairDesign:
    pairs = {(cell.axes[0][0], cell.axes[1][0]) for cell in cells if cell.order == 2}
    if len(pairs) != 1 or any(cell.order != 2 for cell in cells):
        raise ValueError(f"expected the order-2 grid of exactly one pair, got pairs {pairs}")
    return pair_design(cells, pairs.pop())


def free_terms(cells: Sequence[IdentityCell]) -> tuple[str, ...]:
    """The pair's ``(|A|-1)(|B|-1)`` free interaction terms -- its FDR-family members."""
    return free_interaction_terms(_design(cells))


def fit_pair(
    scores: Sequence[ItemScore],
    cells: Sequence[IdentityCell],
    *,
    config: MetricsConfig,
    min_cell_n: int,
) -> EmergenceResult:
    """Fit one pair's grid exactly as P7 fitted it.

    Args:
        scores: standardized item scores, already restricted to the pair's cells and to one
            standardization rule (the caller's job, because the rule is what E4 and R2 differ in).
        cells: the pair's order-2 grid.
        config: the metrics config the headline run used (its bootstrap settings and seed).
        min_cell_n: the pre-registered minimum renderings per cell.
    """
    _design(cells)  # refuse anything but one pair's grid before fitting
    outside = sorted(
        {str(s.cell_id) for s in scores if s.cell_id not in {c.cell_id for c in cells}}
    )
    if outside:
        raise ValueError(f"{len(outside)} score cell(s) are outside the pair's grid: {outside[:3]}")
    return EmergenceMetric(config, min_cell_n=min_cell_n).compute_result(scores, cells)
