"""Synthetic P6 artifacts shaped exactly like ``run_scoring.py``'s, for the P11 pipeline tests.

The P11 orchestration tests start the REAL pipeline at the metrics stage -- real ``run_metrics.py``,
real ``run_stats.py`` under the shipped stats config, real ``build_result_tables.py``, real figure
code -- on score tables that look like what scoring writes. Scoring itself needs the neural
scorers (the inference extra), which the core CI job does not have; everything from P7 on runs
torch-free, and that is the half where the P9 bug hid.

Records are the real types (``ItemScore``, ``DriftRow``), written with the real writers, and
prompt ids take the real ``<template>::<cell>::<carrier>::<digest>`` shape. Harm follows the
design the estimator assumes: main effects + an injected interaction + a per-ITEM intercept (so
items genuinely cluster) + noise.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from margins.lattice.build import DeclaredLattice
from margins.schemas import IdentityCell, ItemScore
from margins.scoring.drift import DriftRow
from margins.scoring.export import write_drift_rows, write_scores

#: The shipped generation roster: two harm types, two scorers each (framing constraint 6).
OPEN_GEN_ROSTER: dict[str, tuple[str, ...]] = {
    "toxicity": ("toxic_bert", "ldnoobw_lexicon"),
    "regard": ("regard_classifier", "vader_valence"),
}
#: The shipped extractive roster.
CROWS_ROSTER: dict[str, tuple[str, ...]] = {
    "stereotype_preference": ("pll_total", "pll_length_normalized"),
}
DRIFT_FAMILIES: tuple[str, ...] = ("prompt_echo", "neutral_declarative")


def _digest(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12]


def grid(lattice: DeclaredLattice, pair: Sequence[str]) -> list[IdentityCell]:
    wanted = frozenset(pair)
    return [
        c for c in lattice.cells() if c.order == 2 and frozenset(a for a, _ in c.axes) == wanted
    ]


def _interaction(cells: Sequence[IdentityCell], rng: np.random.Generator, scale: float) -> dict:
    """A double-centred gamma table, so it is a pure interaction with no main-effect leakage."""
    values_a = sorted({c.axes[0][1] for c in cells})
    values_b = sorted({c.axes[1][1] for c in cells})
    table = rng.normal(0.0, 1.0, size=(len(values_a), len(values_b)))
    table -= table.mean(axis=0, keepdims=True)
    table -= table.mean(axis=1, keepdims=True)
    table *= scale / max(float(np.abs(table).max()), 1e-9)
    return {
        (a, b): float(table[i, j]) for i, a in enumerate(values_a) for j, b in enumerate(values_b)
    }


def model_scores(
    cells: Sequence[IdentityCell],
    *,
    benchmark: str,
    model_id: str,
    roster: dict[str, tuple[str, ...]],
    n_items: int,
    seeds: Sequence[int],
    gamma_scale: float,
    rng: np.random.Generator,
) -> list[ItemScore]:
    """Model-condition scores: ``n_items`` frames per cell x ``seeds`` generations x the roster."""
    out: list[ItemScore] = []
    for harm, scorers in roster.items():
        gamma = _interaction(cells, rng, gamma_scale) if all(c.order == 2 for c in cells) else {}
        for cell in cells:
            key = (cell.axes[0][1], cell.axes[1][1]) if cell.order == 2 else None
            cell_mean = 0.1 * len(cell.cell_id) % 0.7 + (gamma.get(key, 0.0) if key else 0.0)
            for item in range(n_items):
                frame = f"frame{item}"
                prompt_id = f"{frame}::{cell.cell_id}::person::{_digest(frame, cell.cell_id)}"
                item_effect = float(rng.normal(0.0, 0.3))
                for seed in seeds:
                    response_id = f"{prompt_id}::{model_id}::{seed}"
                    base = cell_mean + item_effect + float(rng.normal(0.0, 0.3))
                    for scorer in scorers:
                        value = base + float(rng.normal(0.0, 0.05))
                        out.append(
                            ItemScore(
                                score_id=f"{response_id}::{scorer}",
                                response_id=response_id,
                                scorer_id=scorer,
                                harm_type=harm,
                                raw=value,
                                standardized=value,
                                prompt_id=prompt_id,
                                cell_id=cell.cell_id,
                                benchmark=benchmark,
                                model_id=model_id,
                                seed=seed,
                                temperature=0.7,
                            )
                        )
    return out


def drift_rows(
    cells: Sequence[IdentityCell],
    *,
    benchmark: str,
    roster: dict[str, tuple[str, ...]],
    n_texts: int,
    rng: np.random.Generator,
) -> list[DriftRow]:
    """The scorer's own series on neutral text: main effects only, and SMALL, like a clean one."""
    out: list[DriftRow] = []
    for family in DRIFT_FAMILIES:
        for harm, scorers in roster.items():
            for scorer in scorers:
                for cell in cells:
                    (axis_a, value_a), (axis_b, value_b) = cell.axes
                    for text in range(n_texts):
                        value = 0.02 * (len(value_a) % 3) + float(rng.normal(0.0, 0.02))
                        control = f"{family}::{cell.cell_id}::{text}"
                        out.append(
                            DriftRow(
                                score_id=f"{control}::{scorer}",
                                response_id=control,
                                scorer_id=scorer,
                                harm_type=harm,
                                condition="drift_control",
                                control_family=family,
                                cell_id=cell.cell_id,
                                order=2,
                                axis_a=axis_a,
                                value_a=value_a,
                                axis_b=axis_b,
                                value_b=value_b,
                                benchmark=benchmark,
                                raw=value,
                                standardized=value,
                            )
                        )
    return out


def write_open_gen(
    out_dir: Path,
    lattice: DeclaredLattice,
    *,
    pair: Sequence[str],
    model_id: str,
    n_items: int,
    seeds: Sequence[int],
    benchmark: str = "open_gen_templates",
    seed: int = 0,
    gamma_scale: float = 0.8,
) -> None:
    """``scores.parquet`` + ``drift_control.parquet`` for one model on one pair's full grid.

    Order-1 cells are scored too, as the real E1 set renders all 110 cells: they carry the
    single-axis descriptives the paradigm figure sets beside CrowS-Pairs.
    """
    rng = np.random.default_rng(seed)
    cells = grid(lattice, pair)
    order_1 = [c for c in lattice.cells() if c.order == 1]
    scores = model_scores(
        cells,
        benchmark=benchmark,
        model_id=model_id,
        roster=OPEN_GEN_ROSTER,
        n_items=n_items,
        seeds=seeds,
        gamma_scale=gamma_scale,
        rng=rng,
    )
    scores += model_scores(
        order_1,
        benchmark=benchmark,
        model_id=model_id,
        roster=OPEN_GEN_ROSTER,
        n_items=n_items,
        seeds=seeds,
        gamma_scale=0.0,
        rng=rng,
    )
    write_scores(scores, out_dir)
    write_drift_rows(
        drift_rows(cells, benchmark=benchmark, roster=OPEN_GEN_ROSTER, n_texts=3, rng=rng),
        out_dir,
    )


def write_crows(out_dir: Path, lattice: DeclaredLattice, *, model_id: str, seed: int = 0) -> None:
    """CrowS-shaped scores: ORDER-1 cells only, as measured -- so every pair is not estimable."""
    rng = np.random.default_rng(seed)
    cells = [c for c in lattice.cells() if c.order == 1][:6]
    write_scores(
        model_scores(
            cells,
            benchmark="crows_pairs",
            model_id=model_id,
            roster=CROWS_ROSTER,
            n_items=3,
            seeds=(20260908,),
            gamma_scale=0.0,
            rng=rng,
        ),
        out_dir,
    )
    write_drift_rows([], out_dir)
