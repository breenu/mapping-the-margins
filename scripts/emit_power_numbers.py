"""Register the measured rho and the power-ceiling result in ``results/paper_numbers.json``.

    uv run python scripts/emit_power_numbers.py

These are the numbers behind ``docs/power_and_precision.md``, and the project's guardrail is that
the manuscript cites numbers **by key** rather than retyping digits. The field-underpowered table
in particular is a headline result rather than a methods footnote, so it is registered key by key
with its own prefix: ``field.power_at_9_prompts.*``.

Every entry names the run manifest it came from. The power figures come from
``scripts/run_power_analysis.py``; the rho figures come from ``scripts/measure_rho.py`` on the
Kaggle generation run. Where a number is an arithmetic identity of the design rather than a run's
output -- prompts per cell, say -- it is recorded against the no-run sentinel instead of
borrowing a run id it did not come from.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from margins.lattice.build import DeclaredLattice
from margins.lattice.config import DEFAULT_LATTICE, load_lattice_config
from margins.stats.config import DEFAULT_STATS, load_stats_config
from margins.stats.numbers import (
    NO_RUN,
    PAPER_NUMBERS_PATH,
    PaperNumber,
    read_paper_numbers,
    write_paper_numbers,
)

#: MEASURED, 19 Sep 2026, on the class x race pair through Llama-3.1-8B-Instruct at NF4.
#: ``scripts/measure_rho.py`` produced these from the Kaggle generation run; they are transcribed
#: here rather than recomputed because the run is frozen and its responses are not in the repo.
MEASURED_RHO: dict[str, tuple[float, float, float]] = {
    "regard": (0.35, 0.29, 0.43),
    "toxicity": (0.18, 0.13, 0.23),
}

#: THE FIELD RESULT. Simulated power at the 9 prompts per cell a conventional design uses, for a
#: MODERATE 0.5 SD interaction, at the measured rho. From scripts/run_power_analysis.py.
FIELD_POWER: tuple[tuple[int, float, float], ...] = (
    # (renderings per cell, rho, power)
    (27, 0.20, 0.340),
    (27, 0.23, 0.273),
    (27, 0.35, 0.217),
    (27, 0.43, 0.183),
    (45, 0.20, 0.493),
    (45, 0.23, 0.503),
    (45, 0.35, 0.300),
    (45, 0.43, 0.270),
)

#: The power ceiling at the SHIPPED 27 prompts per cell, 0.5 SD, by rho.
CEILING_AT_27: tuple[tuple[float, float], ...] = (
    (0.20, 1.000),
    (0.23, 1.000),
    (0.35, 0.983),
    (0.43, 0.957),
)

#: The run that produced the power figures. A power analysis writes no manifest of its own at
#: Stage A, so these carry the sentinel and name the script and its committed JSON output.
POWER_SOURCE = "scripts/run_power_analysis.py -> data/stats/power_analysis_0p5sd.json"
RHO_SOURCE = "scripts/measure_rho.py -> data/stats/rho_measurement.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PAPER_NUMBERS_PATH)
    parser.add_argument("--stats", default=DEFAULT_STATS)
    parser.add_argument("--lattice", default=DEFAULT_LATTICE)
    args = parser.parse_args(argv)

    config = load_stats_config(args.stats)
    lattice = DeclaredLattice(load_lattice_config(args.lattice))
    power = config.power

    def number(key: str, value: object, unit: str, source: str, **extra: object) -> PaperNumber:
        return PaperNumber(
            key=key,
            value=value,  # type: ignore[arg-type]
            unit=unit,  # type: ignore[arg-type]
            run_id=NO_RUN,
            manifest_path=None,
            source=source,
            **extra,  # type: ignore[arg-type]
        )

    entries: list[PaperNumber] = []

    # --- the measured rho -------------------------------------------------------------
    for harm, (point, low, high) in MEASURED_RHO.items():
        entries.append(
            number(
                f"rho.{harm}",
                point,
                "ratio",
                RHO_SOURCE,
                ci_low=low,
                ci_high=high,
                note=(
                    "Measured intraclass correlation: the share of standardized harm-score "
                    "variance between prompts rather than between samples of one prompt. "
                    "Measured on ONE axis pair (class x race, the 4x4) and not assumed uniform."
                ),
            )
        )
    entries.append(
        number(
            "rho.n_generations",
            1728,
            "count",
            RHO_SOURCE,
            note="16 cells x 9 frames x 12 sampled seeds, Llama-3.1-8B-Instruct at NF4.",
        )
    )
    entries.append(number("rho.measured_on_pairs", 1, "count", RHO_SOURCE))

    # --- THE FIELD RESULT -------------------------------------------------------------
    for renderings, rho, value in FIELD_POWER:
        entries.append(
            number(
                f"field.power_at_9_prompts.r{renderings}_rho{int(rho * 100):02d}",
                value,
                "probability",
                POWER_SOURCE,
                note=(
                    "Simulated power of a CONVENTIONAL design -- 9 prompts per cell -- on a "
                    "MODERATE 0.5 SD interaction at the measured rho. This is the quantified "
                    "form of the claim that the field is underpowered."
                ),
            )
        )
    field_values = [value for _, _, value in FIELD_POWER]
    entries.append(
        number(
            "field.power_at_9_prompts.min",
            min(field_values),
            "probability",
            POWER_SOURCE,
            note="Worst case across the measured rho range at conventional rendering budgets.",
        )
    )
    entries.append(
        number(
            "field.power_at_9_prompts.max",
            max(field_values),
            "probability",
            POWER_SOURCE,
            note="Best case across the measured rho range at conventional rendering budgets.",
        )
    )

    # --- the ceiling, and the design it forced ----------------------------------------
    for rho, value in CEILING_AT_27:
        entries.append(
            number(
                f"power.ceiling_at_27_prompts.rho{int(rho * 100):02d}",
                value,
                "probability",
                POWER_SOURCE,
            )
        )
    entries += [
        number(
            "power.prompts_per_cell",
            power.frames_per_cell,
            "count",
            "configs/prompts/open_gen_v1.yaml",
            note="13 carrier-using frames x 2 carriers + 1 carrier-free frame emitted once.",
        ),
        number("power.effect_size", power.effect_size, "sd", POWER_SOURCE),
        number("power.target_power", power.target_power, "probability", POWER_SOURCE),
        number(
            "power.min_cell_n",
            lattice.config.power.min_cell_n,
            "renderings",
            POWER_SOURCE,
            note="27 prompts per cell x 4 sampled seeds. Derived, not a placeholder.",
        ),
        number("power.alpha", config.fdr.alpha, "probability", POWER_SOURCE),
        # The two pre-registered thresholds the FIGURES draw as reference lines. They are read
        # off the stats config, not authored here, and they are registered so that a figure can
        # cite them by key like every other number rather than hard-coding a literal into a rule.
        number(
            "fdr.alpha",
            config.fdr.alpha,
            "probability",
            "configs/stats/",
            note="The FDR level the pre-registered family is controlled at.",
        ),
        number(
            "drift.descriptive_only_ratio",
            config.drift.descriptive_only_ratio,
            "ratio",
            "configs/stats/",
            note=(
                "drift_rms / model_rms at or above which a harm type becomes DESCRIPTIVE-ONLY "
                "and the claim layer refuses to emit a significance claim for it."
            ),
        ),
        number(
            "power.gpu_hours",
            95.0,
            "gpu_hours",
            POWER_SOURCE,
            note="Both confirmatory models, 110 cells, at the pilot-measured 14.0/14.8 s each.",
        ),
        number(
            "power.total_generations",
            110 * lattice.config.power.min_cell_n * 2,
            "count",
            POWER_SOURCE,
        ),
        number(
            "design.prompt_set_size",
            27 * len(lattice.cells()),
            "count",
            "configs/prompts/open_gen_v1.yaml",
        ),
        number("design.n_cells", len(lattice.cells()), "count", "configs/lattice/"),
    ]

    existing = read_paper_numbers(args.out)
    merged = existing.merged(entries)
    write_paper_numbers(merged, args.out)
    print(f"registered {len(entries)} numbers into {args.out} ({len(merged.numbers)} total)")
    print("\nTHE FIELD RESULT, as registered:")
    for renderings, rho, value in FIELD_POWER:
        print(f"  {renderings:>3} renderings/cell at rho {rho:<5} -> power {value:.3f}")
    print(
        f"\n  range: {min(field_values):.0%} to {max(field_values):.0%} power on a moderate "
        "0.5 SD interaction, at 9 prompts per cell."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
