"""R2: does an independent rerun give back the reported class x race numbers? (pre-reg §10)

    uv run python scripts/run_r2_analysis.py --machine kaggle

Reads, for one machine:

* the rerun's scores (``data/scoring/r2_<machine>_sampled/<model>/``, written by
  ``run_scoring.py`` -- ``run_p11.py r2-process`` runs it) and the ORIGINAL E1 scores, whose
  standardization constants the rerun is mapped through;
* the original P7 directories and P9 artifacts (``data/metrics/{e1,crows}/<model>/``,
  ``data/stats/``), into which the rerun's class x race series is substituted;
* the original and rerun response files, for the exact-match rates.

and applies the rule pre-registered on 16 Oct 2026, before any rerun data existed: **PASS iff
every class x race family term's rerun estimate lies inside the original run's reported 95% CI
AND every ClaimGate verdict is identical.** It writes the machine's rows of
``results/tables/reproducibility.{csv,tex}`` and ``r2.<machine>.*`` paper numbers -- the
magnitudes and the exact-match rates are reported whatever the verdict.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from margins.config import (
    default_config_dir,
    load_experiment_config,
    manifest_path,
    write_run_manifest,
)
from margins.inference.harness import read_responses
from margins.lattice.build import DeclaredLattice
from margins.lattice.config import default_lattice_config_dir, load_lattice_config
from margins.metrics.config import DEFAULT_METRICS, default_metrics_config_dir, load_metrics_config
from margins.metrics.design import pair_designs
from margins.metrics.export import read_records
from margins.orchestrate.config import load_p11_config
from margins.orchestrate.plan import OPEN_GEN_BENCHMARK, Layout, r2_arm_dir
from margins.orchestrate.smoke import pair_cells
from margins.prompts.config import file_sha256
from margins.report.tables import (
    DEFAULT_TABLE_DIR,
    read_table,
    spec_for,
    write_table_csv,
    write_table_latex,
)
from margins.schemas import EffectEstimate, ItemScore
from margins.scoring.config import DEFAULT_SCORING, load_scoring_config
from margins.scoring.export import SCORES_FILENAME, read_scores
from margins.scoring.standardize import (
    signs_from_specs,
    standardize_with_reference,
    standardize_within_benchmark,
)
from margins.sensitivity import (
    compare_to_original,
    exact_match_rate,
    fit_pair,
    free_terms,
    restrict_to_cells,
    substitute_series,
    summarize,
    verdict_changes,
)
from margins.stats.analysis import run_stats_analysis
from margins.stats.config import DEFAULT_STATS, default_stats_config_dir, load_stats_config
from margins.stats.export import CLAIMS_FILENAME, EFFECTS_CORRECTED_FILENAME
from margins.stats.inputs import load_stats_inputs_many
from margins.stats.numbers import (
    PAPER_NUMBERS_PATH,
    PaperNumber,
    read_paper_numbers,
    write_paper_numbers,
)
from margins.stats.records import SignificanceClaim


def _one_file(directory: Path) -> Path:
    files = sorted(directory.glob("*.jsonl")) if directory.is_dir() else []
    if len(files) != 1:
        raise SystemExit(f"STOP: expected exactly one run file in {directory}, found {len(files)}")
    return files[0]


def _texts(path: Path) -> dict[str, str]:
    return {response.response_id: response.text for response in read_responses(path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", required=True, help="An R2 machine declared in the config.")
    parser.add_argument("--p11", default="p11_v1")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--paper-numbers", type=Path, default=PAPER_NUMBERS_PATH)
    parser.add_argument("--metrics", default=DEFAULT_METRICS)
    parser.add_argument("--stats", default=DEFAULT_STATS)
    parser.add_argument("--scoring", default=DEFAULT_SCORING)
    parser.add_argument("--n-resamples", type=int, default=None, help="Smoke runs only.")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-manifest", action="store_true")
    parser.add_argument("--no-paper-numbers", action="store_true")
    parser.add_argument("--no-latex", action="store_true")
    args = parser.parse_args(argv)

    p11 = load_p11_config(args.p11)
    if args.machine not in p11.r2.machines:
        parser.error(f"machine {args.machine!r} is not declared; one of {p11.r2.machines}")
    layout = Layout(data=args.data_root)
    lattice = DeclaredLattice(load_lattice_config(p11.lattice))
    cells = pair_cells(lattice, p11.r2.pair)
    axis_pair = (cells[0].axes[0][0], cells[0].axes[1][0])
    terms = free_terms(cells)
    metrics = load_metrics_config(args.metrics)
    if args.n_resamples is not None:
        bootstrap = metrics.bootstrap.model_copy(update={"n_resamples": args.n_resamples})
        metrics = metrics.model_copy(update={"bootstrap": bootstrap})
    stats_config = load_stats_config(args.stats)
    signs = signs_from_specs(load_scoring_config(args.scoring).scorers)
    min_cell_n = lattice.config.power.min_cell_n
    sampled, greedy = r2_arm_dir(args.machine, "sampled"), r2_arm_dir(args.machine, "greedy")

    inputs: dict[str, str] = {}
    rerun: list[ItemScore] = []
    exact: dict[str, tuple[int, int, float]] = {}
    for model in p11.e1.models:
        original_path = layout.scoring("e1", model) / SCORES_FILENAME
        rerun_path = layout.scoring(sampled, model) / SCORES_FILENAME
        for path in (original_path, rerun_path):
            if not path.is_file():
                parser.error(f"{path} does not exist; run `run_p11.py r2-process` first")
        _, reference = standardize_within_benchmark(
            read_scores(original_path), benchmark=OPEN_GEN_BENCHMARK, signs=signs
        )
        rerun += standardize_with_reference(
            restrict_to_cells(read_scores(rerun_path), cells),
            benchmark=OPEN_GEN_BENCHMARK,
            reference=reference,
        )
        inputs[f"scores:{model}:original"] = file_sha256(original_path)
        inputs[f"scores:{model}:rerun"] = file_sha256(rerun_path)

        e1_file = _one_file(layout.responses("e1") / model)
        e4_greedy = _one_file(layout.responses(f"e4_{p11.r2.greedy_arm}") / model)
        rerun_sampled = _one_file(layout.responses(sampled) / model)
        rerun_greedy = _one_file(layout.responses(greedy) / model)
        exact[f"{model}:sampled"] = exact_match_rate(_texts(e1_file), _texts(rerun_sampled))
        exact[f"{model}:greedy"] = exact_match_rate(_texts(e4_greedy), _texts(rerun_greedy))

    print(f"R2 {args.machine}: {len(rerun)} rerun score rows on {len(cells)} cells, ")
    print(f"   {metrics.bootstrap.n_resamples} bootstrap replicates, family terms {len(terms)}\n")

    fitted = fit_pair(rerun, cells, config=metrics, min_cell_n=min_cell_n)
    stats_dir = args.data_root / "stats"
    original_effects = read_records(stats_dir / EFFECTS_CORRECTED_FILENAME, EffectEstimate)
    original_claims = read_records(stats_dir / CLAIMS_FILENAME, SignificanceClaim)
    comparisons = compare_to_original(
        original_effects, fitted.effects, axis_pair=axis_pair, terms=terms
    )

    metric_dirs = [layout.metrics(arm, m) for arm in ("e1", "crows") for m in p11.e1.models]
    substituted = substitute_series(
        load_stats_inputs_many(metric_dirs), fitted, axis_pair=axis_pair
    )
    result = run_stats_analysis(
        substituted,
        pair_designs(lattice.cells()),
        lattice.config,
        config=stats_config,
        min_cell_n=min_cell_n,
    )
    reported_size = read_paper_numbers(args.paper_numbers).lookup("fdr.family_size").value
    if result.family_size != reported_size:
        raise SystemExit(
            f"STOP: the substituted family has {result.family_size} members, the reported one "
            f"{reported_size}. R2 substitutes estimates; it must not change the family."
        )
    verdicts = verdict_changes(original_claims, result.claims, axis_pair=axis_pair, terms=terms)
    summary = summarize(comparisons, verdicts)

    status = {(v.model_id, v.harm_type, v.term): v for v in verdicts}
    rows: list[dict[str, Any]] = []
    for c in comparisons:
        verdict = status[(c.model_id, c.harm_type, c.term)]
        rows.append(
            {
                "machine": args.machine,
                "benchmark": OPEN_GEN_BENCHMARK,
                "model_id": c.model_id,
                "harm_type": c.harm_type,
                "axis_a": axis_pair[0],
                "axis_b": axis_pair[1],
                "term": c.term,
                "original_estimate": c.original_estimate,
                "original_se": c.original_se,
                "original_ci_low": c.original_ci_low,
                "original_ci_high": c.original_ci_high,
                "rerun_estimate": c.rerun_estimate,
                "difference_in_se": c.difference_in_se,
                "inside_original_ci": c.inside_original_ci,
                "original_status": verdict.original_status,
                "rerun_status": verdict.rerun_status,
                "data_status": "real",
            }
        )
    table_path = args.out_dir / "reproducibility.csv"
    if table_path.is_file():  # keep the OTHER machines' rows; this run owns only its own
        rows = [
            dict(r)
            for r in read_table("reproducibility", directory=args.out_dir).rows
            if r["machine"] != args.machine
        ] + rows
    write_table_csv(spec_for("reproducibility"), rows, table_path)
    if not args.no_latex:
        write_table_latex(
            read_table("reproducibility", directory=args.out_dir),
            args.out_dir / "reproducibility.tex",
        )

    verdict_word = "PASS" if summary["passed"] else "FAIL"
    print(f"R2 {args.machine}: {verdict_word}")
    print(f"  inside original 95% CI   {summary['n_inside_ci']}/{summary['n_terms']}")
    print(f"  identical verdicts       {summary['n_verdicts_same']}/{len(verdicts)}")
    print(f"  max |diff| (original SE) {summary['max_abs_diff_se']:.3f}")
    print(f"  RMS |diff| (original SE) {summary['rms_diff_se']:.3f}")
    for key, (n, same, rate) in sorted(exact.items()):
        print(f"  exact text match {key:<40} {same}/{n} = {rate:.3f}")
    statuses = sorted({v.original_status for v in verdicts})
    if statuses == ["descriptive_only"]:
        print(
            "  NOTE: every original verdict is descriptive_only (scorer-driven, which R2 does\n"
            "  not regenerate), so the verdict half of the rule is near-automatic; the CI half\n"
            "  carries it."
        )

    if args.no_manifest:
        return 0
    run_name = args.run_name or f"r2-analysis-{args.machine}"
    overrides = [f"run_name={run_name}", f"metrics_name={args.metrics}", f"stats_name={args.stats}"]
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
            "stats_config": file_sha256(default_stats_config_dir() / f"{args.stats}.yaml"),
            **inputs,
        },
    )
    where = manifest_path(cfg, manifest.run_id)
    print(f"\nmanifest   {where}")
    if args.no_paper_numbers:
        return 0
    common = {
        "run_id": manifest.run_id,
        "manifest_path": where.as_posix(),
        "source": "scripts/run_r2_analysis.py",
        "claimable": False,
    }
    prefix = f"r2.{args.machine}"
    registered = [
        PaperNumber(key=f"{prefix}.passed", value=bool(summary["passed"]), **common),
        PaperNumber(key=f"{prefix}.n_terms", value=summary["n_terms"], unit="count", **common),
        PaperNumber(
            key=f"{prefix}.n_inside_ci", value=summary["n_inside_ci"], unit="count", **common
        ),
        PaperNumber(
            key=f"{prefix}.n_verdicts_same",
            value=summary["n_verdicts_same"],
            unit="count",
            **common,
        ),
        PaperNumber(
            key=f"{prefix}.max_abs_diff_se",
            value=summary["max_abs_diff_se"],
            unit="ratio",
            note="in units of the original SE",
            **common,
        ),
        PaperNumber(
            key=f"{prefix}.rms_diff_se",
            value=summary["rms_diff_se"],
            unit="ratio",
            note="in units of the original SE",
            **common,
        ),
    ]
    for arm in ("sampled", "greedy"):
        n = sum(exact[f"{m}:{arm}"][0] for m in p11.e1.models)
        same = sum(exact[f"{m}:{arm}"][1] for m in p11.e1.models)
        registered.append(
            PaperNumber(
                key=f"{prefix}.exact_match_{arm}",
                value=same / n,
                unit="probability",
                note=f"{same}/{n} rerun responses byte-identical to the original",
                **common,
            )
        )
    write_paper_numbers(
        read_paper_numbers(args.paper_numbers).merged(registered), args.paper_numbers
    )
    print(f"paper      {len(registered)} numbers -> {args.paper_numbers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
