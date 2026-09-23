"""The multi-rendering smoke run: which prompts it uses, and what counts as passing (P11).

**Why this exists.** Decided 21 Sep 2026, before any full run: the smoke run must reach
``attach_wald_p -> correct -> ClaimGate -> a figure`` on REAL P7 output. The dev artifact never did
-- one rendering per cell makes every pair not estimable, P9 prints "nothing to correct", and the
correction path is never executed. That is exactly where the P9 family-membership bug hid through
four green phases. A smoke run that cannot reach the correction path does not test it.

**So the pass criterion is about REACHING things, not about results.** A smoke run that prints
"NOTHING TO CORRECT" is a FAIL here, however cleanly it exited. And a smoke run whose every claim
is refused as ``underpowered`` PASSES: at ~9 renderings per cell against ``min_cell_n = 108`` that
is the correct verdict, and what matters is that the gate RAN on corrected estimates and said so.

**The smoke prompts are a SUBSET of the real E1 prompt set**, selected here rather than re-rendered,
so their ids are E1's ids and the smoke run exercises the exact records the sweep will run.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from margins.lattice.build import DeclaredLattice
from margins.schemas import IdentityCell, Prompt

__all__ = [
    "SMOKE_FIGURE_BANNER",
    "SmokeCheck",
    "SmokeVerdict",
    "check_smoke_pass",
    "pair_cells",
    "prompt_carrier",
    "prompt_template",
    "select_smoke_prompts",
]

#: Stamped on the smoke figure. No digits: the traceability audit reads every number on a canvas,
#: and a banner must not be one.
SMOKE_FIGURE_BANNER: Final[str] = (
    "SMOKE RUN - pipeline check on a tiny subset, NOT a result. Do not cite or reuse."
)

_ID_SEPARATOR: Final[str] = "::"


def prompt_template(prompt: Prompt) -> str:
    """The template frame of an open_gen prompt, from its ``item_id`` (``<source>/<frame>``)."""
    return prompt.item_id.rsplit("/", 1)[-1]


def prompt_carrier(prompt: Prompt) -> str:
    """The carrier of an open_gen prompt: field 3 of ``<template>::<cell>::<carrier>::<digest>``."""
    parts = prompt.prompt_id.split(_ID_SEPARATOR)
    if len(parts) != 4:
        raise ValueError(f"not an open_gen prompt id: {prompt.prompt_id!r}")
    return parts[2]


def pair_cells(lattice: DeclaredLattice, pair: Sequence[str]) -> list[IdentityCell]:
    """The order-2 cells of ``pair`` -- the full grid, in the lattice's canonical order.

    Raises:
        ValueError: if the pair is not a CONFIRMATORY pair. A smoke run on the exploratory pair
            would pass through P9 with no p and no q, i.e. never reach the correction it exists
            to exercise.
    """
    wanted = frozenset(pair)
    confirmatory = {frozenset(p) for p in lattice.config.confirmatory_pairs}
    if wanted not in confirmatory:
        raise ValueError(
            f"smoke pair {tuple(pair)} is not a confirmatory pair of this lattice; only a "
            "confirmatory pair's terms enter the FDR family, so only one exercises the correction."
        )
    return [
        cell
        for cell in lattice.cells()
        if cell.order == 2 and frozenset(axis for axis, _ in cell.axes) == wanted
    ]


def select_smoke_prompts(
    prompts: Iterable[Prompt],
    *,
    cells: Sequence[IdentityCell],
    frames: Sequence[str],
    carrier: str,
    decoding_key: str,
) -> list[Prompt]:
    """The smoke subset: every cell of the grid x every listed frame, under one carrier.

    Returned in input order. Raises unless the result is a FULL grid with exactly ``len(frames)``
    items per cell, all stamped ``decoding_key`` -- because an incomplete grid is not estimable,
    and a smoke run that is not estimable tests nothing it exists to test.
    """
    cell_ids = {cell.cell_id for cell in cells}
    frame_set = set(frames)
    chosen = [
        prompt
        for prompt in prompts
        if prompt.cell_id in cell_ids
        and prompt_template(prompt) in frame_set
        and prompt_carrier(prompt) == carrier
    ]
    per_cell = Counter(prompt.cell_id for prompt in chosen)
    short = sorted(cell for cell in cell_ids if per_cell.get(cell, 0) != len(frames))
    if short:
        raise ValueError(
            f"the smoke subset is not a full grid: {len(short)} cell(s) do not have exactly "
            f"{len(frames)} prompts ({short[:3]}...). Check the frames {sorted(frames)} and the "
            f"carrier {carrier!r} exist in the prompt set."
        )
    wrong = sorted({p.decoding_key for p in chosen if p.decoding_key != decoding_key})
    if wrong:
        raise ValueError(
            f"smoke prompts are stamped {wrong}, not {decoding_key!r}. Build the set with "
            f"`build_prompts.py --decoding {decoding_key}` first."
        )
    return chosen


@dataclass(frozen=True)
class SmokeCheck:
    """One clause of the pass criterion."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class SmokeVerdict:
    """Every clause, and the conjunction."""

    checks: tuple[SmokeCheck, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    def render(self) -> str:
        lines = [f"SMOKE {'PASS' if self.passed else 'FAIL'}"]
        for check in self.checks:
            lines.append(f"  [{'ok' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
        return "\n".join(lines)


def check_smoke_pass(
    *,
    open_gen_metrics_dir: Path,
    stats_dir: Path,
    figure_paths: Sequence[Path],
    model_id: str,
    lattice: DeclaredLattice,
    open_gen_benchmark: str,
    crows_benchmark: str,
    correction: str,
) -> SmokeVerdict:
    """Read the smoke run's artifacts back and evaluate the pass criterion clause by clause.

    Every clause reads a file the pipeline wrote, never an in-memory value, so a stage that
    "succeeded" without writing what it should have cannot pass.
    """
    from margins.metrics.export import FIT_REPORTS_FILENAME, read_records
    from margins.metrics.records import FitReport
    from margins.schemas import EffectEstimate
    from margins.stats.export import (
        CLAIMS_FILENAME,
        EFFECTS_CORRECTED_FILENAME,
        FAMILY_FILENAME,
    )
    from margins.stats.records import FamilyMember, SignificanceClaim

    confirmatory = {frozenset(pair) for pair in lattice.config.confirmatory_pairs}
    checks: list[SmokeCheck] = []

    # 1. An estimable confirmatory pair. Without one there is nothing for P9 to correct.
    reports_path = open_gen_metrics_dir / FIT_REPORTS_FILENAME
    reports = read_records(reports_path, FitReport) if reports_path.is_file() else []
    estimable = sorted(
        {
            f"{r.axis_a} x {r.axis_b}"
            for r in reports
            if r.estimable
            and r.condition == "model"
            and r.scorer_id is None
            and r.model_id == model_id
            and frozenset((r.axis_a, r.axis_b)) in confirmatory
        }
    )
    checks.append(
        SmokeCheck(
            "estimable confirmatory pair",
            bool(estimable),
            ", ".join(estimable) if estimable else f"none in {reports_path}",
        )
    )

    # 2. The correction ran on real P7 output: a family member carrying q, family_size, correction.
    effects_path = stats_dir / EFFECTS_CORRECTED_FILENAME
    effects = read_records(effects_path, EffectEstimate) if effects_path.is_file() else []
    corrected = [
        e
        for e in effects
        if e.benchmark == open_gen_benchmark
        and e.model == model_id
        and e.q is not None
        and (e.family_size or 0) >= 1
        and e.correction == correction
    ]
    if not effects_path.is_file():
        detail = (
            f"{effects_path} was never written -- P9 found NOTHING TO CORRECT, which is the "
            "one-rendering failure this run exists to rule out"
        )
    elif not corrected:
        detail = f"no {open_gen_benchmark} effect in {effects_path} carries a q"
    else:
        sizes = sorted({e.family_size for e in corrected if e.family_size is not None})
        detail = f"{len(corrected)} family terms corrected ({correction}), family_size {sizes}"
    checks.append(SmokeCheck("FDR correction reached", bool(corrected), detail))

    # 3. The claim gate ran on every corrected member, and its verdicts are recorded.
    claims_path = stats_dir / CLAIMS_FILENAME
    claims = read_records(claims_path, SignificanceClaim) if claims_path.is_file() else []
    by_key = {(c.benchmark, c.model_id, c.harm_type, c.axis_a, c.axis_b, c.term): c for c in claims}
    missing = [
        e
        for e in corrected
        if (e.benchmark, e.model, e.harm_type, *e.axis_pair, e.term) not in by_key
    ]
    statuses = Counter(
        by_key[(e.benchmark, e.model, e.harm_type, *e.axis_pair, e.term)].status
        for e in corrected
        if (e.benchmark, e.model, e.harm_type, *e.axis_pair, e.term) in by_key
    )
    gate_ok = bool(corrected) and not missing
    gate_detail = (
        f"verdicts {dict(sorted(statuses.items()))} (underpowered is EXPECTED at smoke n)"
        if gate_ok
        else f"{len(missing)} corrected member(s) have no claim row in {claims_path}"
    )
    checks.append(SmokeCheck("ClaimGate ran on every corrected term", gate_ok, gate_detail))

    # 4. The extractive source is present in the family and EXCUSED, not raised and not missing.
    family_path = stats_dir / FAMILY_FILENAME
    family = read_records(family_path, FamilyMember) if family_path.is_file() else []
    crows = [m for m in family if m.benchmark == crows_benchmark]
    crows_ok = bool(crows) and all(m.status == "excused_not_estimable" for m in crows)
    checks.append(
        SmokeCheck(
            f"{crows_benchmark} members excused as not estimable",
            crows_ok,
            f"{len(crows)} member(s), statuses {sorted({m.status for m in crows})}",
        )
    )

    # 5. The figure exists in every vector format.
    absent = [path for path in figure_paths if not path.is_file()]
    checks.append(
        SmokeCheck(
            "emergence figure written (PDF + SVG, traceability audit clean)",
            bool(figure_paths) and not absent,
            ", ".join(str(p) for p in figure_paths) if not absent else f"missing {absent}",
        )
    )
    return SmokeVerdict(tuple(checks))
