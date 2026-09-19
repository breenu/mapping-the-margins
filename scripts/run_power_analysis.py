"""Run the P9 power simulation at the real design parameters and print the decision tables.

    uv run python scripts/run_power_analysis.py

This is **Stage A** of P9. It answers one question -- how many renderings per cell are needed to
detect the pre-registered minimum interaction at 80% power, on the CLUSTERED design -- and it stops
there. It deliberately does **not** write ``min_cell_n`` into ``configs/lattice/confirmatory_v1
.yaml``: if the answer does not fit the compute budget, the response (a larger target effect, a
narrower family, more template frames, or more descriptive-only cells) is a human decision.

No run manifest is written here. The manifest's ``stats`` block and the ``stats_name`` config field
arrive with the rest of the layer in Stage B; this script emits its full result as JSON so the
numbers behind the printed tables are inspectable, and Stage B re-runs the analysis under the
manifest.

What is printed, in order:

1. the family arithmetic, derived from the lattice rather than asserted;
2. the CEILING table -- power as seeds go to infinity, per template-frame count and per assumed
   intraclass correlation. **A ceiling below the target cannot be beaten by any seed budget**;
3. the minimum renderings per cell wherever the ceiling allows one;
4. power at the budgets currently on the table (the ~50/cell target and the 60 placeholder);
5. the implied total generation count and GPU-hours from the measured pilot rates.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from margins.lattice.build import DeclaredLattice
from margins.lattice.config import DEFAULT_LATTICE, load_lattice_config
from margins.metrics.design import PairDesign, pair_design, pair_designs
from margins.repro import set_global_seed
from margins.stats.config import DEFAULT_STATS, StatsConfig, load_stats_config
from margins.stats.family import family_size
from margins.stats.power import PowerDesign, ceiling_point, power_curve, simulate_power_point

#: Measured on Kaggle, Tesla T4, NF4, 18 Sep 2026 (CLAUDE.md decision #8). Seconds per prompt.
PILOT_RATES: dict[str, float] = {
    "meta-llama/Llama-3.1-8B-Instruct": 14.0,
    "mistralai/Mistral-7B-Instruct-v0.3": 14.8,
}

#: (benchmark, model, harm_type) combinations the family spans: 1 benchmark x 2 models x 2 harms.
N_FAMILY_GROUPS = 4

#: Rendering budgets already named in the project, priced for comparison.
NAMED_BUDGETS: tuple[tuple[int, str], ...] = (
    (27, "9 frames x 3 seeds"),
    (45, "9 frames x 5 seeds -- the low end of the ~50 target"),
    (54, "9 frames x 6 seeds -- the high end of the ~50 target"),
    (63, "9 frames x 7 seeds -- first budget clearing the 60 placeholder"),
    (108, "9 frames x 12 seeds"),
    (216, "9 frames x 24 seeds"),
)


@dataclass(frozen=True)
class Budget:
    """What one renderings-per-cell figure costs across both confirmatory models."""

    renderings_per_cell: int
    n_cells: int
    generations_per_model: int
    total_generations: int
    gpu_hours: float

    @property
    def kaggle_weeks(self) -> float:
        """Free-tier Kaggle is ~30 GPU-h per week (PROJECT_CONTEXT.md section 7)."""
        return self.gpu_hours / 30.0


def price(renderings_per_cell: int, n_cells: int) -> Budget:
    """Total generations and GPU-hours for one renderings-per-cell figure.

    Total generations is ``n_cells x renderings_per_cell`` per model, independent of how the
    renderings split between template frames and sampled seeds -- more frames means more prompts
    with fewer samples each, for the same total. The split changes the STATISTICS, not the bill.
    """
    per_model = n_cells * renderings_per_cell
    seconds = sum(per_model * rate for rate in PILOT_RATES.values())
    return Budget(
        renderings_per_cell=renderings_per_cell,
        n_cells=n_cells,
        generations_per_model=per_model,
        total_generations=per_model * len(PILOT_RATES),
        gpu_hours=seconds / 3600.0,
    )


def confirmatory_designs(lattice: DeclaredLattice) -> list[PairDesign]:
    confirmatory = set(lattice.config.confirmatory_pairs)
    return [d for d in pair_designs(lattice.cells()) if d.axis_pair in confirmatory]


def report_family(lattice: DeclaredLattice, config: StatsConfig) -> int:
    designs = confirmatory_designs(lattice)
    print("FDR FAMILY (the confirmatory pairs ARE the family; section 8)\n")
    print(f"  {'axis pair':<42} {'levels':>9} {'free':>6} {'all int':>8} {'all':>5}")
    for design in designs:
        n_a, n_b = len(design.levels_a), len(design.levels_b)
        pair = f"{design.axis_a} x {design.axis_b}"
        print(
            f"  {pair:<42} {f'{n_a}x{n_b}':>9} {(n_a - 1) * (n_b - 1):>6} "
            f"{n_a * n_b:>8} {n_a + n_b + n_a * n_b:>5}"
        )
    sizes = {
        terms: family_size(designs, family_terms=terms, n_groups=N_FAMILY_GROUPS)
        for terms in ("free_interaction", "all_interaction", "all_terms")
    }
    print(
        f"\n  x {N_FAMILY_GROUPS} groups (1 benchmark x 2 models x 2 harm types):  "
        + "   ".join(f"{name} = {size}" for name, size in sizes.items())
    )
    chosen = sizes[config.fdr.family_terms]
    print(f"  CONFIGURED: {config.fdr.family_terms} -> family_size = {chosen}\n")
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", default=DEFAULT_STATS, help="Stats config name.")
    parser.add_argument("--lattice", default=DEFAULT_LATTICE)
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=None,
        help="Override the config's replicate count (smoke runs only; recorded as used).",
    )
    parser.add_argument(
        "--search-simulations",
        type=int,
        default=None,
        help=(
            "Replicates for the coarse bisection probes (default: a quarter of --n-simulations, "
            "floored at 100). The reported power always comes from a full-precision re-simulation."
        ),
    )
    # --- overrides, for exploring a re-target before the config is edited --------------
    #
    # The shipped stats config is PRE-REGISTRATION MATERIAL, and editing it to try a different
    # target effect or a measured rho would churn its version history with candidates that were
    # never adopted. These flags let the analysis be re-run at other settings; whatever is chosen
    # is then written into the config once, as a decision.
    parser.add_argument(
        "--effect-size",
        type=float,
        default=None,
        help="Override the target effect, in SD units (the config ships 0.3).",
    )
    parser.add_argument(
        "--icc-rho",
        type=float,
        nargs="+",
        default=None,
        help="Override the rho strip. Pass the MEASURED values once rho is known.",
    )
    parser.add_argument(
        "--frames", type=int, nargs="+", default=None, help="Override the frames-per-cell grid."
    )
    parser.add_argument(
        "--seeds-grid", type=int, nargs="+", default=None, help="Override the seeds grid."
    )
    parser.add_argument("--patterns", nargs="+", default=None, choices=("concentrated", "diffuse"))
    parser.add_argument("--out", type=Path, default=Path("data") / "stats" / "power_analysis.json")
    args = parser.parse_args(argv)

    config = load_stats_config(args.stats)
    settings = config.power
    overrides: dict[str, Any] = {}
    if args.effect_size is not None:
        overrides["effect_size"] = args.effect_size
    if args.icc_rho is not None:
        overrides["icc_rho"] = tuple(args.icc_rho)
    if args.frames is not None:
        overrides["frames_grid"] = tuple(args.frames)
        # frames_per_cell must stay inside the grid; the validator enforces that the shipped
        # design is one of the priced points.
        if settings.frames_per_cell not in args.frames:
            overrides["frames_per_cell"] = min(args.frames)
    if args.seeds_grid is not None:
        overrides["seeds_grid"] = tuple(args.seeds_grid)
    if args.patterns is not None:
        overrides["patterns"] = tuple(args.patterns)
    if overrides:
        settings = settings.model_copy(update=overrides)
        # Re-validate: model_copy bypasses validators, and a grid that violates them would only
        # surface as a confusing failure deep in the sweep.
        settings = type(settings).model_validate(settings.model_dump())
        print(f"OVERRIDES   {', '.join(f'{k}={v}' for k, v in sorted(overrides.items()))}\n")

    n_simulations = args.n_simulations or settings.n_simulations
    search_simulations = args.search_simulations or max(100, n_simulations // 4)

    lattice = DeclaredLattice(load_lattice_config(args.lattice))
    design = pair_design(lattice.cells(), settings.pair)
    n_cells_total = len(lattice.cells())

    print(f"stats      {config.key}  {config.stats_version}  hash={config.stats_hash[:16]}...")
    print(f"lattice    {lattice.config.lattice_version}  {n_cells_total} cells")
    print(
        f"simulated  {design.axis_a} x {design.axis_b}  "
        f"{len(design.levels_a)}x{len(design.levels_b)} = {design.n_cells} order-2 cells, "
        f"{design.n_params} parameters"
    )
    print(
        f"target     detect {settings.effect_size} SD at power {settings.target_power:.2f}, "
        f"alpha {config.fdr.alpha}, {config.fdr.method}"
    )
    print(f"replicates {n_simulations} per reported point, {search_simulations} per search probe\n")

    size = report_family(lattice, config)

    def log(line: str) -> None:
        """Progress, flushed, so a grid that takes half an hour is observable while it runs."""
        print(line, flush=True)

    _, rng = set_global_seed(settings.seed)
    common = {
        "effect_size": settings.effect_size,
        "alpha": config.fdr.alpha,
        "method": config.fdr.method,
        "family_size": size,
        "n_simulations": n_simulations,
    }

    # --- 1. the ceiling ---------------------------------------------------------------
    print(
        "CEILING -- power as seeds -> infinity. A cell has `frames` ITEM CLUSTERS however many\n"
        "seeds are drawn, so cell-mean variance floors at rho/frames. No seed budget beats this.\n"
    )
    ceilings: dict[tuple[str, float, int], float] = {}
    for pattern in settings.patterns:
        print(f"  pattern = {pattern}")
        header = "    rho  " + "".join(f"{f'{f} frames':>12}" for f in settings.frames_grid)
        print(header)
        for rho in settings.icc_rho:
            row = f"   {rho:<5.2f}"
            for frames in settings.frames_grid:
                point = ceiling_point(
                    design,
                    frames_per_cell=frames,
                    pattern=pattern,
                    icc_rho=rho,
                    rng=rng,
                    **common,  # type: ignore[arg-type]
                )
                ceilings[(pattern, rho, frames)] = point.power
                mark = "*" if point.power >= settings.target_power else " "
                row += f"{point.power:>11.3f}{mark}"
            print(row, flush=True)
        print()
    print(f"  * = ceiling reaches the {settings.target_power:.0%} target at that frame count.\n")

    # --- 2. the minimum, where a ceiling allows one -----------------------------------
    print("MINIMUM RENDERINGS PER CELL (only where the ceiling allows one)\n")
    results = []
    for pattern in settings.patterns:
        for rho in settings.icc_rho:
            for frames in settings.frames_grid:
                if ceilings[(pattern, rho, frames)] < settings.target_power:
                    continue
                log(f"  searching {pattern} rho={rho:g} frames={frames}")
                result = power_curve(
                    design,
                    frames_per_cell=frames,
                    seeds_grid=settings.seeds_grid,
                    pattern=pattern,
                    icc_rho=rho,
                    target_power=settings.target_power,
                    rng=rng,
                    search_simulations=search_simulations,
                    progress=log,
                    **common,  # type: ignore[arg-type]
                )
                results.append(result)
                if result.min_renderings_per_cell is None:
                    log(
                        f"  {pattern:<13} rho={rho:<5.2f} frames={frames:<3} "
                        f"-> none in grid (ceiling {result.power_ceiling:.3f})"
                    )
                    continue
                budget = price(result.min_renderings_per_cell, n_cells_total)
                log(
                    f"  {pattern:<13} rho={rho:<5.2f} frames={frames:<3} "
                    f"-> {result.min_seeds_per_item:>3} seeds = "
                    f"{result.min_renderings_per_cell:>4} renderings/cell   "
                    f"{budget.total_generations:>8,} generations   "
                    f"{budget.gpu_hours:>7.1f} GPU-h"
                )
    if not results:
        print("  none: every ceiling in the grid sits below the target.\n")
    else:
        print()

    # --- 3. what the budgets currently on the table actually buy -----------------------
    frames = settings.frames_per_cell
    print(
        f"POWER AT THE BUDGETS ALREADY ON THE TABLE ({frames} template frames, pattern = "
        f"{settings.patterns[0]})\n"
    )
    header = "  renderings/cell  " + "".join(f"{f'rho={r:g}':>10}" for r in settings.icc_rho)
    print(header + "     GPU-h   note")
    named_powers: dict[int, dict[float, float]] = {}
    for renderings, label in NAMED_BUDGETS:
        seeds, remainder = divmod(renderings, frames)
        if remainder or seeds < 1:
            continue
        row = f"  {renderings:>15}  "
        named_powers[renderings] = {}
        for rho in settings.icc_rho:
            point = simulate_power_point(
                PowerDesign(design=design, frames_per_cell=frames, seeds_per_item=seeds),
                pattern=settings.patterns[0],
                icc_rho=rho,
                rng=rng,
                **common,  # type: ignore[arg-type]
            )
            named_powers[renderings][rho] = point.power
            row += f"{point.power:>10.3f}"
        budget = price(renderings, n_cells_total)
        log(f"{row}  {budget.gpu_hours:>8.1f}   {label}")
    print()

    # --- 4. the bill ------------------------------------------------------------------
    print("COMPUTE, from the measured pilot rates (T4, NF4; 14.0 s and 14.8 s per prompt)\n")
    print(f"  {'renderings/cell':>16} {'generations':>14} {'GPU-h':>9} {'Kaggle weeks':>14}")
    for renderings, _ in NAMED_BUDGETS:
        budget = price(renderings, n_cells_total)
        print(
            f"  {renderings:>16} {budget.total_generations:>14,} "
            f"{budget.gpu_hours:>9.1f} {budget.kaggle_weeks:>14.1f}"
        )
    print(
        f"\n  Both models, all {n_cells_total} cells. Total generations is independent of how the\n"
        "  renderings split between frames and seeds; the split changes the statistics, not the\n"
        "  bill. Kaggle free tier is ~30 GPU-h/week and is the reproducibility path only.\n"
    )

    # --- 5. the record ----------------------------------------------------------------
    payload = {
        "stats_config": config.key,
        "stats_version": config.stats_version,
        "stats_hash": config.stats_hash,
        "lattice_version": lattice.config.lattice_version,
        "axis_pair": list(design.axis_pair),
        "family_terms": config.fdr.family_terms,
        "family_size": size,
        "alpha": config.fdr.alpha,
        "method": config.fdr.method,
        "effect_size": settings.effect_size,
        "target_power": settings.target_power,
        "n_simulations": n_simulations,
        "ceilings": [
            {"pattern": p, "icc_rho": r, "frames_per_cell": f, "power": v}
            for (p, r, f), v in sorted(ceilings.items())
        ],
        "named_budget_power": [
            {"renderings_per_cell": n, "icc_rho": r, "power": v}
            for n, by_rho in sorted(named_powers.items())
            for r, v in sorted(by_rho.items())
        ],
        "minimums": [result.model_dump(mode="json") for result in results],
        "budgets": [
            {
                "renderings_per_cell": n,
                "total_generations": price(n, n_cells_total).total_generations,
                "gpu_hours": round(price(n, n_cells_total).gpu_hours, 2),
            }
            for n, _ in NAMED_BUDGETS
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out}")
    print(
        "\nSTAGE A STOPS HERE. min_cell_n is NOT written into the lattice config; the 96/60\n"
        "stale markers stand until the human decides the response."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
