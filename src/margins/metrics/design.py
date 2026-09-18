"""The design matrix for ``harm ~ A + B + A:B`` over one axis pair (P7, section 10).

**What the estimand is, in one sentence.** For an axis pair, fit a two-way linear model over the
pair's order-2 cells; the coefficients on ``A:B`` are the emergent intersectional bias. **Never
the level.** A cell whose mean harm is the highest in the table has a high *level*; it is emergent
only if that level departs from what the two main effects predict additively. This module is where
that distinction is made concrete, because the interaction *is* a set of contrasts on the design
matrix and nothing else.

**Sum-to-zero (deviation) coding, over the order-2 cells only** -- decided 18 Sep 2026, before the
pre-registration tag:

* Each axis with ``L`` levels contributes ``L - 1`` columns; level ``k < L - 1`` is coded as the
  unit vector ``e_k`` and the last level as ``-1`` everywhere. Under this coding the intercept is
  the grand mean of cell means, each main effect is a level's deviation from it, and each
  interaction coefficient ``gamma_ij`` is the departure of cell ``(i, j)`` from the additive fit:
  in a balanced design, exactly ``mu_ij - mu_i. - mu_.j + mu..``, the textbook two-way interaction
  (:func:`additive_decomposition` computes that closed form and the tests hold the fit to it).
* No reference level is needed. Treatment coding would express every interaction as a
  difference-in-differences against one reference cell, and two of the five axes (age, gender)
  have no defensible reference.
* Order-1 cells do not enter. The lattice has no "no identity" cell, so an order-1 cell can only be
  coded as an *absent* level of the other axis -- and the (absent, absent) corner does not exist,
  which leaves the model rank-deficient by one and confounds "absent" with "one fewer descriptor
  in the text". Order-1 cells are described by :class:`~margins.schemas.CellStat` instead.

**Every level's coefficient is emitted, not only the free ones.** With ``L - 1`` free parameters
per factor, the last level's effect is implied by the sum constraint. Emitting all ``|A|`` main
effects and all ``|A| x |B|`` interaction cells means no cell is arbitrarily the one without a
number, at the cost that the emitted set is linearly dependent (rows and columns of the gamma table
sum to zero). That dependency is stated here so P9 can account for it when it defines the FDR
family; it is not hidden by dropping a level.

The matrix is built explicitly rather than through a formula string: axis values such as
``working-class`` and ``Deaf`` would need escaping in patsy, and the coding is the estimand -- it
should be readable in the code that defines it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from margins.metrics.records import TermKind
from margins.schemas import IdentityCell

__all__ = [
    "PairDesign",
    "TermContrast",
    "additive_decomposition",
    "interaction_term",
    "main_term",
    "pair_design",
    "pair_designs",
]


def main_term(axis: str, value: str) -> str:
    """The ``term`` string for a main effect: ``<axis>=<value>``."""
    return f"{axis}={value}"


def interaction_term(axis_a: str, value_a: str, axis_b: str, value_b: str) -> str:
    """The ``term`` string for one interaction cell: ``<axis_a>=<a>:<axis_b>=<b>``."""
    return f"{main_term(axis_a, value_a)}:{main_term(axis_b, value_b)}"


def _coding(n_levels: int, index: int) -> np.ndarray:
    """Sum-to-zero code for level ``index`` of a factor with ``n_levels`` levels."""
    if n_levels < 2:
        raise ValueError("a factor needs at least two levels")
    code = np.zeros(n_levels - 1)
    if index == n_levels - 1:
        code[:] = -1.0
    else:
        code[index] = 1.0
    return code


@dataclass(frozen=True, eq=False)
class TermContrast:
    """A linear combination of the parameters that yields one reported term."""

    term: str
    kind: TermKind
    value_a: str | None
    value_b: str | None
    vector: np.ndarray


@dataclass(frozen=True, eq=False)
class PairDesign:
    """The two-way design for one axis pair, over its order-2 cells.

    Level order follows the order the cells were given in, which for a
    :class:`~margins.lattice.build.DeclaredLattice` is the canonical declaration order. The order
    changes which level is the implied one internally but changes NO emitted number, because every
    level's coefficient is reported.
    """

    axis_a: str
    axis_b: str
    levels_a: tuple[str, ...]
    levels_b: tuple[str, ...]
    #: ``(value_a, value_b, cell_id)`` for every cell, in ``levels_a x levels_b`` order.
    cells: tuple[tuple[str, str, str], ...]

    @property
    def axis_pair(self) -> tuple[str, str]:
        return (self.axis_a, self.axis_b)

    @property
    def n_params(self) -> int:
        a, b = len(self.levels_a), len(self.levels_b)
        return 1 + (a - 1) + (b - 1) + (a - 1) * (b - 1)

    @property
    def n_cells(self) -> int:
        return len(self.levels_a) * len(self.levels_b)

    def cell_id(self, value_a: str, value_b: str) -> str:
        for a, b, cell_id in self.cells:
            if a == value_a and b == value_b:
                return cell_id
        raise KeyError(f"({value_a!r}, {value_b!r}) is not a cell of this design")

    def _codes(self, value_a: str, value_b: str) -> tuple[np.ndarray, np.ndarray]:
        try:
            index_a = self.levels_a.index(value_a)
            index_b = self.levels_b.index(value_b)
        except ValueError as exc:
            raise KeyError(
                f"({value_a!r}, {value_b!r}) is not a level pair of {self.axis_pair}: levels are "
                f"{self.levels_a} x {self.levels_b}"
            ) from exc
        return _coding(len(self.levels_a), index_a), _coding(len(self.levels_b), index_b)

    def row(self, value_a: str, value_b: str) -> np.ndarray:
        """The design row for one observation at ``(value_a, value_b)``."""
        code_a, code_b = self._codes(value_a, value_b)
        return np.concatenate(([1.0], code_a, code_b, np.kron(code_a, code_b)))

    def matrix(self, values_a: Sequence[str], values_b: Sequence[str]) -> np.ndarray:
        """The full design matrix, one row per observation."""
        if len(values_a) != len(values_b):
            raise ValueError("values_a and values_b must have the same length")
        if not values_a:
            return np.zeros((0, self.n_params))
        return np.vstack([self.row(a, b) for a, b in zip(values_a, values_b, strict=True)])

    @property
    def contrasts(self) -> tuple[TermContrast, ...]:
        """Every reported term as a contrast on the parameter vector.

        Main effects first (all levels of A, then all of B), then every interaction cell in
        ``levels_a x levels_b`` order. Zero rows of the parameter vector are the intercept and the
        other factor's columns.
        """
        n_a, n_b = len(self.levels_a), len(self.levels_b)
        offset_a = 1
        offset_b = offset_a + (n_a - 1)
        offset_ab = offset_b + (n_b - 1)
        p = self.n_params

        out: list[TermContrast] = []
        for i, a in enumerate(self.levels_a):
            vector = np.zeros(p)
            vector[offset_a:offset_b] = _coding(n_a, i)
            out.append(TermContrast(main_term(self.axis_a, a), "main_a", a, None, vector))
        for j, b in enumerate(self.levels_b):
            vector = np.zeros(p)
            vector[offset_b:offset_ab] = _coding(n_b, j)
            out.append(TermContrast(main_term(self.axis_b, b), "main_b", None, b, vector))
        for i, a in enumerate(self.levels_a):
            for j, b in enumerate(self.levels_b):
                vector = np.zeros(p)
                vector[offset_ab:] = np.kron(_coding(n_a, i), _coding(n_b, j))
                out.append(
                    TermContrast(
                        interaction_term(self.axis_a, a, self.axis_b, b),
                        "interaction",
                        a,
                        b,
                        vector,
                    )
                )
        return tuple(out)

    @property
    def contrast_matrix(self) -> np.ndarray:
        """The contrasts stacked: ``(n_terms, n_params)``."""
        return np.vstack([c.vector for c in self.contrasts])


def pair_design(cells: Sequence[IdentityCell], axis_pair: tuple[str, str]) -> PairDesign:
    """Build the design for ``axis_pair`` from the order-2 cells that carry it.

    The design must be a complete product of the levels observed on each axis. Cells are the
    lattice's declaration (P3 emits ``|A| x |B|`` cells per declared pair), so an incomplete
    product means the caller passed a subset -- a design error, distinct from an *empty* cell in
    the data, which the estimator reports.

    Raises:
        ValueError: if no order-2 cell carries the pair, or the cells do not form a full product.
    """
    wanted = frozenset(axis_pair)
    matching = [c for c in cells if c.order == 2 and frozenset(a for a, _ in c.axes) == wanted]
    if not matching:
        raise ValueError(f"no order-2 cell carries axis pair {axis_pair}")

    # Canonical order is the order the cell's axes are declared in; every cell of a pair agrees.
    axis_a, axis_b = (axis for axis, _ in matching[0].axes)
    levels_a: list[str] = []
    levels_b: list[str] = []
    by_values: dict[tuple[str, str], str] = {}
    for cell in matching:
        values = cell.axes_map
        a, b = values[axis_a], values[axis_b]
        if a not in levels_a:
            levels_a.append(a)
        if b not in levels_b:
            levels_b.append(b)
        by_values[(a, b)] = cell.cell_id

    missing = [(a, b) for a in levels_a for b in levels_b if (a, b) not in by_values]
    if missing:
        raise ValueError(
            f"axis pair {(axis_a, axis_b)} is not a full product of its levels: missing cells "
            f"{missing}. The lattice declares every combination; pass all of its cells."
        )
    if len(levels_a) < 2 or len(levels_b) < 2:
        raise ValueError(
            f"axis pair {(axis_a, axis_b)} needs at least two levels per axis to carry an "
            f"interaction, got {levels_a} x {levels_b}"
        )
    return PairDesign(
        axis_a=axis_a,
        axis_b=axis_b,
        levels_a=tuple(levels_a),
        levels_b=tuple(levels_b),
        cells=tuple((a, b, by_values[(a, b)]) for a in levels_a for b in levels_b),
    )


def pair_designs(cells: Sequence[IdentityCell]) -> list[PairDesign]:
    """One design per axis pair present among the order-2 cells, in first-appearance order."""
    seen: list[tuple[str, str]] = []
    for cell in cells:
        if cell.order != 2:
            continue
        pair = (cell.axes[0][0], cell.axes[1][0])
        if pair not in seen:
            seen.append(pair)
    return [pair_design(cells, pair) for pair in seen]


def additive_decomposition(
    table: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """The sum-to-zero decomposition of a cell-mean table: ``(mu, alpha, beta, gamma)``.

    ``mu`` is the grand mean, ``alpha_i = mu_i. - mu``, ``beta_j = mu_.j - mu`` and
    ``gamma_ij = mu_ij - mu_i. - mu_.j + mu``. In a balanced design these are exactly the sum-coded
    OLS coefficients; the tests use this as the answer key. ``gamma`` is also what the additive
    null (B2) leaves as residual -- which is why B2 is a presentation of the interaction and not an
    independent baseline (section 10).
    """
    means = np.asarray(table, dtype=float)
    if means.ndim != 2:
        raise ValueError("table must be two-dimensional (levels_a x levels_b)")
    mu = float(means.mean())
    alpha = means.mean(axis=1) - mu
    beta = means.mean(axis=0) - mu
    gamma = means - alpha[:, None] - beta[None, :] - mu
    return mu, alpha, beta, gamma
