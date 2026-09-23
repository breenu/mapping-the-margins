"""B3 on real data: the permutation null for every FDR-family term, from the item-level scores.

:mod:`margins.stats.permute` has shipped the permutation machinery since P9, but nothing ran it on
a real run: ``run_stats.py`` reads only P7's fitted outputs, and a permutation null needs the
OBSERVATIONS the fit was made from. So after the lab run every ``SignificanceClaim.p_permutation``
would have been empty -- and B3 is in the minimum claim set (E1 + B1 + B3 + E4), because it is the
only baseline that can DISAGREE with the GLM (section 10). This module closes that (P11).

**Same rows as the estimator, by construction.** Observations are triangulated by P7's own
:func:`~margins.metrics.metric.triangulate` and cut to a pair by P7's own
:func:`~margins.metrics.metric.pair_observations`, so the permuted frame is the fitted frame. That
is then CHECKED, not assumed: the observed interaction contrasts are recomputed from the frame and
must equal P7's ``EffectEstimate.estimate`` to 1e-8, or this raises -- a p-value attached to a
different series than the estimate beside it would be a silent mis-join.

**Exchangeability blocks = (template, carrier, stratum value)** (decided 23 Sep 2026, pre-reg §10).
Since ``open_gen_v1.2`` each frame renders under two carriers; blocking on the template alone would
exchange a "... person" item with a "... individual" item, i.e. across a second source of variation
that is not the permuted axis. With the carrier in the block, items in a block differ ONLY in the
permuted axis, which is exactly the null of no interaction. The carrier-free frame forms its own
blocks.

**Scope.** Family members only -- the confirmatory pairs' free interaction contrasts. The p is
reported BESIDE the GLM p and never enters the FDR family (§5: mixing an assumption-light second
opinion into the corrected family would double-count the hypothesis). Exploratory pairs get none,
for the reason they get no GLM p.

**Deterministic and order-independent.** Each series draws from its own generator, seeded by the
pre-registered permutation seed plus a digest of the series key, so adding a model or a pair never
changes another series' null.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from margins.metrics.design import PairDesign
from margins.metrics.estimator import PairFrame, lstsq_params
from margins.metrics.metric import pair_observations, triangulate
from margins.repro import set_global_seed
from margins.schemas import EffectEstimate, IdentityCell, ItemScore
from margins.stats.config import PermutationSettings
from margins.stats.inputs import EffectKey, effect_key
from margins.stats.permute import permutation_p, permutation_plan, permute_interaction

__all__ = [
    "JOIN_TOLERANCE",
    "B3Result",
    "B3Series",
    "block_frame_of_item",
    "claim_key",
    "run_b3",
]

#: How closely the recomputed observed contrast must match P7's estimate. Both are the same OLS on
#: the same rows; anything larger than float noise means the rows are not the same rows.
JOIN_TOLERANCE = 1e-8

_ID_SEPARATOR = "::"


def block_frame_of_item(prompt_id: str) -> str:
    """``<template>::<carrier>`` from a ``<template>::<cell>::<carrier>::<digest>`` prompt id.

    The template AND the carrier: two items sharing both differ only in the identity cell.

    Raises:
        ValueError: on an id of another shape. Placing an item in the wrong block would exchange
            things that are not exchangeable, so an unparseable id stops the run.
    """
    parts = prompt_id.split(_ID_SEPARATOR)
    if len(parts) != 4 or not parts[0] or not parts[2]:
        raise ValueError(
            f"cannot place item {prompt_id!r} in a permutation block: expected "
            "'<template>::<cell>::<carrier>::<digest>'"
        )
    return f"{parts[0]}{_ID_SEPARATOR}{parts[2]}"


def claim_key(key: EffectKey) -> tuple[str, ...]:
    """The key :meth:`~margins.stats.claims.ClaimGate.claims` looks ``permutation_p`` up by."""
    return tuple(str(part) for part in key)


@dataclass(frozen=True)
class B3Series:
    """One permuted series: which fit, how many items could move, how many terms got a p."""

    benchmark: str
    model_id: str
    harm_type: str
    axis_a: str
    axis_b: str
    n_items: int
    exchangeable_items: int
    n_terms: int


@dataclass(frozen=True)
class B3Result:
    """The permutation p per family term, keyed as the claim gate reads it."""

    p_values: Mapping[tuple[str, ...], float]
    series: tuple[B3Series, ...]
    n_permutations: int
    scheme: str

    @property
    def n_terms(self) -> int:
        return len(self.p_values)


def _series_rng(seed: int, key: Sequence[str]) -> np.random.Generator:
    digest = hashlib.blake2b("\x1f".join(key).encode("utf-8"), digest_size=8).digest()
    return np.random.default_rng(np.random.SeedSequence([seed, int.from_bytes(digest, "big")]))


def run_b3(
    scores: Sequence[ItemScore],
    cells: Sequence[IdentityCell],
    designs: Sequence[PairDesign],
    effects: Sequence[EffectEstimate],
    *,
    is_family_member: Callable[[EffectKey], bool],
    settings: PermutationSettings,
) -> B3Result:
    """Permutation p-values for every family member present in ``effects``.

    Args:
        scores: the item-level P6 score rows the P7 fits were made from.
        cells: the lattice, to resolve ``cell_id`` -> axis values.
        designs: the pair designs P7 fitted.
        effects: P7's estimates. Only family members get a p; each is also the join check.
        is_family_member: the FDR family's membership test.
        settings: the pre-registered scheme, stratum axis, replicate count and seed.

    Raises:
        ValueError: if a family member's series cannot be rebuilt from ``scores``, or its observed
            contrast does not match P7's estimate.
    """
    set_global_seed(settings.seed)
    members = {effect_key(e): e for e in effects if is_family_member(effect_key(e))}
    if not members:
        return B3Result({}, (), settings.n_permutations, settings.scheme)

    wanted_series = {(k[0], k[1], k[4], k[2]) for k in members}  # benchmark, model, harm, pair
    by_cell = {cell.cell_id: cell for cell in cells}
    design_by_pair = {design.axis_pair: design for design in designs}
    model_rows = [s for s in scores if s.condition == "model" and s.cell_id is not None]

    p_values: dict[tuple[str, ...], float] = {}
    series_out: list[B3Series] = []
    groups = sorted({(benchmark, model, harm) for benchmark, model, harm, _ in wanted_series})
    for benchmark, model, harm in groups:
        rows = [
            s
            for s in model_rows
            if (s.benchmark, s.model_id, s.harm_type) == (benchmark, model, harm)
        ]
        if not rows:
            raise ValueError(
                f"no item-level scores for {benchmark}/{model}/{harm}, but P7 estimated family "
                "terms for it. B3 needs the scores the fit was made from: pass that run's "
                "scores.parquet."
            )
        _, observations = triangulate(rows)
        pairs = sorted(
            pair for b, m, h, pair in wanted_series if (b, m, h) == (benchmark, model, harm)
        )
        for pair in pairs:
            design = design_by_pair[pair]
            in_pair, value_a, value_b, items = pair_observations(observations, design, by_cell)
            frame = PairFrame(
                y=np.array([o.combined for o in in_pair]),
                value_a=value_a,
                value_b=value_b,
                item=items,
            )
            matrix = design.matrix(frame.value_a, frame.value_b)
            interaction = [c for c in design.contrasts if c.kind == "interaction"]
            observed = np.vstack([c.vector for c in interaction]) @ lstsq_params(
                matrix, np.asarray(frame.y, dtype=float)
            )

            plan = permutation_plan(
                frame,
                design,
                frame_of_item={item: block_frame_of_item(item) for item in set(items)},
                stratum_axis=settings.stratum_axis,
            )
            series_key = (benchmark, model, harm, pair[0], pair[1])
            null = permute_interaction(
                frame,
                design,
                plan,
                scheme=settings.scheme,
                n_permutations=settings.n_permutations,
                rng=_series_rng(settings.seed, series_key),
            )

            n_terms = 0
            for index, contrast in enumerate(interaction):
                key: EffectKey = (benchmark, model, pair, contrast.term, harm)
                effect = members.get(key)
                if effect is None:
                    continue  # a dependent gamma: reported descriptively, never tested
                if abs(float(observed[index]) - effect.estimate) > JOIN_TOLERANCE:
                    raise ValueError(
                        f"B3 join check failed for {key}: the permuted frame's observed contrast "
                        f"is {float(observed[index])!r} but P7 estimated {effect.estimate!r}. "
                        "The scores passed are not the ones this metrics directory was fitted "
                        "from; a permutation p on a different series would be a silent mis-join."
                    )
                p_values[claim_key(key)] = permutation_p(float(observed[index]), null[:, index])
                n_terms += 1
            series_out.append(
                B3Series(
                    benchmark=benchmark,
                    model_id=model,
                    harm_type=harm,
                    axis_a=pair[0],
                    axis_b=pair[1],
                    n_items=plan.n_items,
                    exchangeable_items=plan.exchangeable_items,
                    n_terms=n_terms,
                )
            )

    missing = [key for key in members if claim_key(key) not in p_values]
    if missing:
        raise ValueError(
            f"{len(missing)} family term(s) got no permutation p, e.g. {missing[:2]}. Every family "
            "member with a GLM p must have a B3 p beside it."
        )
    return B3Result(p_values, tuple(series_out), settings.n_permutations, settings.scheme)
