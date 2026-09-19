"""The p-value, and where it comes from (P9).

P7 deliberately left :attr:`~margins.schemas.EffectEstimate.p` unset: nothing in
:mod:`margins.metrics` computes a p-value, because a p-value on a clustered fit is a claim about
degrees of freedom, and the degrees of freedom are a property of the *design* rather than of the
estimate. This module makes that claim explicit and testable.

**The test is a two-sided Wald t on the cluster-robust standard error, with ``df`` equal to the
number of ITEM clusters minus one.** That is what ``statsmodels`` itself uses under
``cov_type="cluster"``, and the test suite holds :func:`wald_p` to ``OLS(...).fit(cov_type=
"cluster").pvalues`` on a real fit rather than to a reimplementation of the same formula.

Two consequences worth stating, because both are counter-intuitive and both are the point:

* **``df`` does not grow with seeds.** Drawing more samples per prompt adds observations inside a
  cluster, not clusters, so a design with 9 template frames per cell has the same ``df`` at 3
  seeds and at 30. A p-value computed against ``n_obs`` would be anti-conservative by the design
  effect, which is exactly the failure cluster-robust inference exists to prevent.
* **A missing cluster count is refused, never defaulted.** Substituting ``n_obs`` or a large
  constant when the item count cannot be resolved would silently return a smaller p. The lookup
  raises instead.

Setting ``p`` is not restricted to ``StatsEngine`` -- invariant 3 names ``q``, ``family_size`` and
``correction``, and only those are set in :mod:`margins.stats.correct`. A ``p`` is a test
statistic; a ``q`` is a significance claim.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from margins.schemas import EffectEstimate
from margins.stats.inputs import SeriesKey

__all__ = ["attach_wald_p", "wald_p", "wald_t"]


def wald_t(estimate: float, se: float) -> float:
    """The Wald statistic ``estimate / se``.

    Raises:
        ValueError: if ``se`` is zero or negative. A zero standard error is a degenerate fit, not
            an infinitely significant one, and returning ``inf`` would let it become a claim.
    """
    if se <= 0.0:
        raise ValueError(
            f"a Wald statistic needs a positive standard error, got se={se!r}. A zero SE is a "
            "degenerate fit (no residual variation), not infinite evidence."
        )
    return estimate / se


def wald_p(estimate: float, se: float, *, df: int) -> float:
    """Two-sided Student-t p-value for one cluster-robust coefficient.

    Args:
        estimate: the coefficient.
        se: its cluster-robust standard error.
        df: residual degrees of freedom. For a cluster-robust fit this is the number of item
            clusters minus one -- **not** ``n_obs - n_params``.

    Raises:
        ValueError: if ``se <= 0`` or ``df < 1``.
    """
    from scipy import stats

    if df < 1:
        raise ValueError(
            f"a t test needs at least one degree of freedom, got df={df!r}. With cluster-robust "
            "standard errors df is the ITEM cluster count minus one, which a design with fewer "
            "than two clusters cannot supply."
        )
    statistic = wald_t(estimate, se)
    return float(2.0 * stats.t.sf(abs(statistic), df))


def attach_wald_p(
    effects: Sequence[EffectEstimate],
    *,
    df_by_series: Mapping[SeriesKey, int],
    only: Callable[[EffectEstimate], bool] | None = None,
) -> list[EffectEstimate]:
    """Return new estimates carrying ``p``. Sets ``p`` and nothing else.

    ``only`` restricts which estimates get a p-value. The stats layer passes the confirmatory
    predicate, because an exploratory pair carries **no p and no q**: a bare p with no q beside
    it invites exactly the "well, it was significant" reading that the confirmatory/exploratory
    split exists to prevent.

    Raises:
        KeyError: if an estimate's series has no recorded item count. Defaulting a degrees of
            freedom would silently produce a smaller p, so it is refused instead.
    """
    out: list[EffectEstimate] = []
    for effect in effects:
        if only is not None and not only(effect):
            out.append(effect)
            continue
        key: SeriesKey = (
            effect.benchmark,
            effect.model,
            effect.harm_type,
            effect.axis_pair[0],
            effect.axis_pair[1],
        )
        if key not in df_by_series:
            raise KeyError(
                f"no item-cluster count for series {key}. The degrees of freedom of a "
                "cluster-robust Wald test come from the triangulated fit report; without it "
                "there is no defensible df, and defaulting one would understate every p-value."
            )
        p_value = wald_p(effect.estimate, effect.se, df=df_by_series[key])
        out.append(effect.model_copy(update={"p": p_value}))
    return out
