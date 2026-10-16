"""E4: the class x race interaction across temperature and seed -> the `sensitivity` table (RQ4).

    uv run python scripts/run_e4_analysis.py

Reads each confirmatory model's item scores for the three E4 arms --

* T=0.7: E1's own scores (``data/scoring/e1/<model>/``), restricted to the class x race grid;
* T=1.0 and greedy: the E4 arms' scores (``data/scoring/e4_<arm>/<model>/``), written by
  ``run_scoring.py`` on the E4 response files (``run_p11.py e4-process`` runs it) --

standardizes each arm WITHIN itself, fits it per seed and pooled with P7's own metric, and writes
``results/tables/sensitivity.{csv,tex}`` plus descriptive ``e4.*`` paper numbers. The estimands
were fixed before this ran (pre-registration §10, 16 Oct 2026); see :mod:`margins.sensitivity.e4`.

**Descriptive.** No p, no q: RQ4 is answered by the spread.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from margins.config import (
    default_config_dir,
    load_experiment_config,
    manifest_path,
    write_run_manifest,
)
from margins.lattice.build import DeclaredLattice
from margins.lattice.config import default_lattice_config_dir, load_lattice_config
from margins.metrics.config import DEFAULT_METRICS, default_metrics_config_dir, load_metrics_config
from margins.orchestrate.config import load_p11_config
from margins.orchestrate.plan import OPEN_GEN_BENCHMARK, Layout
from margins.orchestrate.smoke import pair_cells
from margins.prompts.config import file_sha256
from margins.report.tables import (
    DEFAULT_TABLE_DIR,
    read_table,
    spec_for,
    write_table_csv,
    write_table_latex,
)
from margins.scoring.config import DEFAULT_SCORING, load_scoring_config
from margins.scoring.export import SCORES_FILENAME, read_scores
from margins.scoring.standardize import signs_from_specs
from margins.sensitivity.e4 import POOLED, ArmScores, sensitivity_rows
from margins.stats.numbers import (
    PAPER_NUMBERS_PATH,
    PaperNumber,
    read_paper_numbers,
    write_paper_numbers,
)

T07 = "t07"


def arm_score_paths(layout: Layout, model: str, arm_names: list[str]) -> dict[str, Path]:
    """``{arm: scores.parquet}`` -- the T=0.7 arm is E1's, the others are E4's own."""
    paths = {T07: layout.scoring("e1", model) / SCORES_FILENAME}
    for name in arm_names:
        paths[name] = layout.scoring(f"e4_{name}", model) / SCORES_FILENAME
    return paths


def e4_numbers(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Descriptive summaries in SD units: how far each arm's pooled estimate sits from T=0.7's,
    and how far apart one arm's seed groups land. Maxima over models, harm types and terms."""
    pooled: dict[tuple[str, str, str], dict[float, float]] = defaultdict(dict)
    by_seed: dict[tuple[str, str, str, float], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["model_id"], row["harm_type"], row["term"])
        if row["seed_group"] == POOLED:
            pooled[key][row["temperature"]] = row["estimate"]
        else:
            by_seed[(*key, row["temperature"])].append(row["estimate"])
    out: dict[str, float] = {}
    for temperature, label in ((1.0, "t10"), (0.0, "t00")):
        shifts = [abs(t[temperature] - t[0.7]) for t in pooled.values() if temperature in t]
        if shifts:
            out[f"e4.max_abs_shift_{label}_vs_t07"] = max(shifts)
    for temperature, label in ((0.7, "t07"), (1.0, "t10")):
        ranges = [max(v) - min(v) for k, v in by_seed.items() if k[3] == temperature]
        if ranges:
            out[f"e4.max_seed_range_{label}"] = max(ranges)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p11", default="p11_v1", help="P11 declaration (configs/experiment).")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--paper-numbers", type=Path, default=PAPER_NUMBERS_PATH)
    parser.add_argument("--metrics", default=DEFAULT_METRICS)
    parser.add_argument("--scoring", default=DEFAULT_SCORING)
    parser.add_argument(
        "--n-resamples", type=int, default=None, help="Override the bootstrap size (smoke only)."
    )
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--run-name", default="e4-analysis")
    parser.add_argument("--no-manifest", action="store_true")
    parser.add_argument("--no-paper-numbers", action="store_true")
    parser.add_argument("--no-latex", action="store_true")
    args = parser.parse_args(argv)

    p11 = load_p11_config(args.p11)
    lattice = DeclaredLattice(load_lattice_config(p11.lattice))
    cells = pair_cells(lattice, p11.e4.pair)
    metrics = load_metrics_config(args.metrics)
    if args.n_resamples is not None:
        bootstrap = metrics.bootstrap.model_copy(update={"n_resamples": args.n_resamples})
        metrics = metrics.model_copy(update={"bootstrap": bootstrap})
    signs = signs_from_specs(load_scoring_config(args.scoring).scorers)
    min_cell_n = lattice.config.power.min_cell_n
    layout = Layout(data=args.data_root)

    arm_names = [arm.name for arm in p11.e4.arms]
    inputs: dict[str, Path] = {}
    arms: list[ArmScores] = []
    for model in p11.e1.models:
        for arm, path in arm_score_paths(layout, model, arm_names).items():
            if not path.is_file():
                parser.error(
                    f"{path} does not exist. Score the E4 arms first: run_p11.py e4-process"
                )
            scores = read_scores(path)
            model_ids = {s.model_id for s in scores}
            if len(model_ids) != 1:
                parser.error(f"{path} holds {sorted(model_ids)}; one model per score file")
            arms.append(ArmScores(model_ids.pop(), arm, scores))
            inputs[f"scores:{model}:{arm}"] = file_sha256(path)
            print(f"  {model:<24} {arm:<7} {len(scores):>7} rows  {path}")

    print(
        f"\nE4 on {' x '.join(p11.e4.pair)}: {len(cells)} cells, {metrics.bootstrap.n_resamples} "
        f"bootstrap replicates, min_cell_n {min_cell_n}"
    )
    rows = sensitivity_rows(
        arms,
        cells,
        benchmark=OPEN_GEN_BENCHMARK,
        signs=signs,
        config=metrics,
        min_cell_n=min_cell_n,
    )
    spec = spec_for("sensitivity")
    path = write_table_csv(spec, rows, args.out_dir / "sensitivity.csv")
    print(f"  sensitivity {len(rows):>5} rows  -> {path}")
    if not args.no_latex:
        write_table_latex(
            read_table("sensitivity", directory=args.out_dir), args.out_dir / "sensitivity.tex"
        )

    summary = e4_numbers(rows)
    for key, value in summary.items():
        print(f"  {key:<34} {value:.3f} SD")

    if args.no_manifest:
        return 0
    overrides = [f"run_name={args.run_name}", f"metrics_name={args.metrics}"]
    if args.results_dir is not None:
        overrides.append(f"results_dir={args.results_dir.as_posix()}")
    cfg = load_experiment_config(
        config_dir=args.config_dir or default_config_dir(), overrides=overrides
    )
    manifest = write_run_manifest(
        cfg,
        inputs={
            "lattice_config": file_sha256(default_lattice_config_dir() / f"{p11.lattice}.yaml"),
            "metrics_config": file_sha256(default_metrics_config_dir() / f"{args.metrics}.yaml"),
            **inputs,
        },
    )
    where = manifest_path(cfg, manifest.run_id)
    print(f"\nmanifest   {where}")
    if not args.no_paper_numbers:
        common = {
            "run_id": manifest.run_id,
            "manifest_path": where.as_posix(),
            "source": "scripts/run_e4_analysis.py",
            "unit": "sd",
            "claimable": False,
            "note": "E4 is descriptive (RQ4): a spread, not a test.",
        }
        registered = [PaperNumber(key=k, value=v, **common) for k, v in summary.items()]
        write_paper_numbers(
            read_paper_numbers(args.paper_numbers).merged(registered), args.paper_numbers
        )
        print(f"paper      {len(registered)} numbers -> {args.paper_numbers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
