"""Fit the emergence estimator on a scored run and emit every P7 artifact.

    uv run python scripts/run_metrics.py

Reads ``data/scoring/scores.parquet`` and ``data/scoring/drift_control.parquet`` (P6) plus the
lattice, and writes under ``--out-dir`` (default ``data/metrics``):

* ``cell_stats.parquet``        -- descriptive per-cell means, cluster-bootstrap CI, powered
* ``effects.parquet``           -- UNCORRECTED EffectEstimates (p/q/family_size/correction unset)
* ``fit_terms.parquet``         -- every fitted term of every series (triangulated, scorer, drift)
* ``fit_reports.parquet``       -- one row per fit attempted, estimable or not, with the reason
* ``drift_comparison.parquet``  -- the scorer's own interaction beside the model's, as a ratio
* ``family_agreement.parquet``, ``harm_flags.parquet``, ``drift_side_by_side.parquet``
* ``drift_comparison.md``, ``fit_report.md``

and a run manifest naming the estimator, the resampling unit, and the PLACEHOLDER thresholds.

**Nothing here declares significance.** ``effects.parquet`` is what P9's StatsEngine reads; the
FDR family, ``q``, ``family_size`` and ``correction`` are set there and only there.

The manifest is written BEFORE the fits, as ``run_scoring.py`` and ``run_inference.py`` do: an
interrupted run still leaves a complete record of what it was.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from margins.config import (
    default_config_dir,
    load_experiment_config,
    manifest_path,
    write_run_manifest,
)
from margins.lattice.build import DeclaredLattice
from margins.lattice.config import DEFAULT_LATTICE, default_lattice_config_dir, load_lattice_config
from margins.metrics.compare import compare_drift
from margins.metrics.config import DEFAULT_METRICS, default_metrics_config_dir, load_metrics_config
from margins.metrics.drift import fit_drift
from margins.metrics.export import DEFAULT_METRICS_DIR, export_metrics
from margins.metrics.manifest import metrics_inputs, metrics_record
from margins.metrics.metric import EmergenceMetric
from margins.prompts.config import file_sha256
from margins.scoring.export import (
    DEFAULT_SCORING_DIR,
    DRIFT_FILENAME,
    SCORES_FILENAME,
    read_drift_rows,
    read_scores,
)

BENCHMARK = "open_gen_templates"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORING_DIR / SCORES_FILENAME)
    parser.add_argument("--drift", type=Path, default=DEFAULT_SCORING_DIR / DRIFT_FILENAME)
    parser.add_argument("--metrics", default=DEFAULT_METRICS, help="Metrics config name.")
    parser.add_argument("--lattice", default=DEFAULT_LATTICE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--run-name", default="metrics")
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=None,
        help="Override the config's bootstrap replicates (smoke runs only; recorded as used).",
    )
    parser.add_argument(
        "--no-manifest", action="store_true", help="Skip the run manifest (debugging only)."
    )
    args = parser.parse_args(argv)

    config = load_metrics_config(args.metrics, config_dir=default_metrics_config_dir())
    if args.n_resamples is not None:
        bootstrap = config.bootstrap.model_copy(update={"n_resamples": args.n_resamples})
        config = config.model_copy(update={"bootstrap": bootstrap})
    lattice_config = load_lattice_config(args.lattice)
    lattice = DeclaredLattice(lattice_config)
    min_cell_n = lattice_config.power.min_cell_n

    scores = read_scores(args.scores)
    drift_rows = read_drift_rows(args.drift)
    if not scores:
        parser.error(f"no scores in {args.scores}")

    print(f"metrics    {config.key}  {config.metrics_version}  hash={config.metrics_hash[:16]}...")
    print(f"bootstrap  {config.bootstrap.n_resamples} replicates, seed {config.bootstrap.seed}")
    print(f"scores     {len(scores)} rows from {args.scores}")
    print(f"drift      {len(drift_rows)} rows from {args.drift}")
    print(f"min_cell_n {min_cell_n} (from the lattice config; a PLACEHOLDER until P9)")

    # --- manifest first --------------------------------------------------------------
    cfg = None
    if not args.no_manifest:
        overrides = [f"run_name={args.run_name}", f"metrics_name={args.metrics}"]
        if args.results_dir is not None:
            overrides.append(f"results_dir={args.results_dir.as_posix()}")
        cfg = load_experiment_config(
            config_dir=args.config_dir or default_config_dir(), overrides=overrides
        )

    # --- the fits --------------------------------------------------------------------
    metric = EmergenceMetric(config, min_cell_n=min_cell_n)
    cells = lattice.cells()
    result = metric.compute_result(scores, cells)
    print(
        f"\nmodel      {len(result.cell_stats)} cell stats, {len(result.effects)} effect "
        f"estimates, {len(result.fit_reports)} fits attempted"
    )
    drift = fit_drift(drift_rows, cells, config=config)
    print(f"drift      {len(drift.fit_reports)} fits attempted")
    comparison = compare_drift(result, drift, config=config)

    not_estimable = [r for r in [*result.fit_reports, *drift.fit_reports] if not r.estimable]
    if not_estimable:
        print(f"\nnot estimable ({len(not_estimable)}):")
        for report in not_estimable[:12]:
            series = (
                f"drift:{report.scorer_id}:{report.control_family}"
                if report.condition == "drift_control"
                else (report.scorer_id or "triangulated")
            )
            print(f"  {report.harm_type:<10} {series:<40} {report.axis_a} x {report.axis_b}")
            print(f"    {report.reason}")
        if len(not_estimable) > 12:
            print(f"  ... and {len(not_estimable) - 12} more (see fit_report.md)")

    print("\ndrift comparison (scorer's own interaction beside the model's; nothing subtracted)")
    for flag in comparison.harm_flags:
        verdict = "DESCRIPTIVE-ONLY candidate" if flag.descriptive_only_candidate else "clear"
        print(
            f"  {flag.harm_type:<10} {verdict:<28} comparable {flag.n_comparable}/"
            f"{flag.n_comparisons}  scorers {list(flag.scorers_comparable)}"
        )
    disagreements = [a for a in comparison.family_agreement if a.families_disagree]
    print(f"  family disagreements: {len(disagreements)}")

    paths = export_metrics(result, drift, comparison, config=config, out_dir=args.out_dir)
    print("\nwrote")
    for path in paths.values():
        print(f"  {path}")

    if cfg is None:
        return 0

    inputs = {
        "lattice_config": file_sha256(default_lattice_config_dir() / f"{args.lattice}.yaml"),
        **metrics_inputs(
            metrics_config_path=default_metrics_config_dir() / f"{args.metrics}.yaml",
            scores_path=args.scores,
            drift_path=args.drift,
        ),
    }
    manifest = write_run_manifest(
        cfg,
        inputs=inputs,
        metrics=metrics_record(
            config,
            metrics_name=args.metrics,
            benchmark=BENCHMARK,
            min_cell_n=min_cell_n,
            n_pairs=len({(r.axis_a, r.axis_b) for r in result.fit_reports}),
            n_effects=len(result.effects),
            n_not_estimable=len(not_estimable),
            n_drift_fits=sum(1 for r in drift.fit_reports if r.estimable),
            harm_flags=comparison.harm_flags,
            n_family_disagreements=len(disagreements),
        ),
    )
    print(f"\nmanifest   {manifest_path(cfg, manifest.run_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
