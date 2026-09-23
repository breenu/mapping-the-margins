"""Correct the emergence estimates over the pre-registered family and emit every P9 artifact.

    uv run python scripts/run_stats.py

Reads ``data/metrics/`` (P7) plus the lattice, and writes under ``--out-dir``
(default ``data/stats``):

* ``effects_corrected.parquet`` -- EffectEstimates with q, family_size and correction stamped
* ``family.parquet``           -- every declared family slot, present or excused, with reasons
* ``descriptive_only.parquet`` -- the enforced section 9 downgrade, per harm type
* ``claims.parquet``           -- one row per effect INCLUDING refused ones, with the reason and
                                  B3's permutation p beside the GLM p
* ``fdr_report.md``            -- the correction in words

plus ``results/paper_numbers.json`` and a run manifest.

**B3 runs here, on the item-level scores (P11).** The permutation null needs the observations the
fits were made from, which P7's outputs do not carry, so pass every run's ``scores.parquet`` with
``--scores``. Leaving B3 out takes an explicit ``--no-permutation``, never a silent default: B3 is
in the minimum claim set, and an empty permutation column must be a choice someone made.

**This is the only stage that declares significance.** The family is the lattice's confirmatory
pairs and nothing else; exploratory pairs pass through with no p and no q. A harm type the drift
rule downgrades keeps its estimates and loses its claims.

The manifest is written AFTER the analysis here, unlike the earlier stages, because the block it
carries is largely a summary of what the correction found and an empty one would say nothing.
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
from margins.metrics.design import pair_designs
from margins.metrics.export import DEFAULT_METRICS_DIR
from margins.prompts.config import file_sha256
from margins.scoring.export import read_scores
from margins.stats.analysis import run_stats_analysis
from margins.stats.config import DEFAULT_STATS, default_stats_config_dir, load_stats_config
from margins.stats.export import DEFAULT_STATS_DIR, export_stats
from margins.stats.inputs import load_stats_inputs_many
from margins.stats.manifest import stats_inputs, stats_record
from margins.stats.numbers import (
    PAPER_NUMBERS_PATH,
    PaperNumber,
    read_paper_numbers,
    write_paper_numbers,
)

BENCHMARK = "open_gen_templates"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        nargs="+",
        default=[DEFAULT_METRICS_DIR],
        help="One or more P7 output directories. Pass BOTH paradigms' directories: the FDR "
        "family spans benchmark x model x axis_pair x term x harm_type across the whole "
        "pre-registered set and is never corrected per-source (section 8).",
    )
    parser.add_argument("--stats", default=DEFAULT_STATS, help="Stats config name.")
    parser.add_argument("--lattice", default=DEFAULT_LATTICE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_STATS_DIR)
    parser.add_argument("--paper-numbers", type=Path, default=PAPER_NUMBERS_PATH)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--run-name", default="stats")
    parser.add_argument(
        "--scores",
        type=Path,
        nargs="+",
        default=[],
        help="The P6 scores.parquet of every run in --metrics-dir. B3 (the permutation null) is "
        "computed from these item-level rows, which must match the P7 fits exactly.",
    )
    parser.add_argument(
        "--no-permutation",
        action="store_true",
        help="Skip B3. Debugging only: B3 is in the minimum claim set.",
    )
    parser.add_argument("--no-manifest", action="store_true")
    parser.add_argument(
        "--no-paper-numbers", action="store_true", help="Skip results/paper_numbers.json."
    )
    args = parser.parse_args(argv)

    config = load_stats_config(args.stats)
    lattice = DeclaredLattice(load_lattice_config(args.lattice))
    lattice_config = lattice.config
    min_cell_n = lattice_config.power.min_cell_n

    inputs = load_stats_inputs_many(args.metrics_dir)
    designs = pair_designs(lattice.cells())

    print(f"stats      {config.key}  {config.stats_version}  hash={config.stats_hash[:16]}...")
    print(f"lattice    {lattice_config.lattice_version}  min_cell_n={min_cell_n}")
    print(f"effects    {len(inputs.effects)} uncorrected, from {inputs.paths['effects']}")
    print(f"groups     {len(inputs.groups)} (benchmark x model x harm type)")
    print(
        f"family     {config.fdr.family_terms} over "
        f"{len(lattice_config.confirmatory_pairs)} confirmatory pairs, "
        f"{config.fdr.method} at alpha {config.fdr.alpha}\n"
    )

    if not inputs.effects:
        # The dev-scale P7 artifact is exactly this: one rendering per cell, so every pair is
        # reported not estimable and no EffectEstimate exists. Correcting nothing is not a
        # degenerate success -- it is a run that had no hypotheses to test, and saying so is
        # more useful than emitting an empty family of size zero.
        print(
            "NOTHING TO CORRECT.\n"
            f"  {inputs.paths['effects']} holds no EffectEstimate rows, which means P7 found "
            "every axis pair not estimable.\n"
            "  That is what a dev-scale run looks like: with one rendering per cell there are "
            "too few items per\n"
            "  cell to fit an interaction. See fit_reports.parquet for the per-pair reason.\n"
            "  Re-run scripts/run_metrics.py against a full sampled run and this stage will "
            "have a family to correct."
        )
        return 0

    if not args.scores and not args.no_permutation:
        parser.error(
            "B3 needs the item-level scores: pass every run's scores.parquet with --scores (or "
            "--no-permutation to skip B3 deliberately, which leaves p_permutation empty)."
        )
    scores = None
    if args.scores:
        scores = [row for path in args.scores for row in read_scores(path)]
        print(f"scores     {len(scores)} item-level rows from {len(args.scores)} file(s), for B3")

    result = run_stats_analysis(
        inputs,
        designs,
        lattice_config,
        config=config,
        min_cell_n=min_cell_n,
        scores=scores,
        cells=lattice.cells() if scores is not None else None,
    )

    print(f"FAMILY SIZE            {result.family_size}")
    print(f"  declared slots       {len(result.family)}")
    print(
        f"  excused not estimable "
        f"{sum(1 for m in result.family if m.status == 'excused_not_estimable')}"
    )
    print(f"significant pre-FDR    {result.n_significant_pre}")
    print(f"significant post-FDR   {result.n_significant_post}")
    assert result.n_significant_post <= result.n_significant_pre, (
        "post-FDR significance exceeded pre-FDR, which is arithmetically impossible for BH; "
        "something is wrong with the correction."
    )
    print(f"exploratory (no q)     {result.n_exploratory}")
    print(f"claimable / refused    {result.n_claimable} / {result.n_refused}")
    if result.b3 is None:
        print("B3 PERMUTATION         NOT RUN (--no-permutation): p_permutation is empty\n")
    else:
        below = sum(1 for p in result.b3.p_values.values() if p <= config.fdr.alpha)
        print(
            f"B3 PERMUTATION         {result.b3.n_terms} family terms, {result.b3.scheme}, "
            f"{result.b3.n_permutations} replicates; p_perm <= {config.fdr.alpha}: {below}"
        )
        for series in result.b3.series:
            print(
                f"  {series.model_id} {series.harm_type:<10} {series.axis_a} x {series.axis_b}: "
                f"{series.exchangeable_items}/{series.n_items} items exchangeable"
            )
        print()

    print("DESCRIPTIVE-ONLY (section 9, enforced at the pre-registered threshold)")
    for decision in result.descriptive_only:
        verdict = "DESCRIPTIVE-ONLY" if decision.descriptive_only else "clear"
        ratio = "—" if decision.max_ratio is None else f"{decision.max_ratio:.2f}"
        print(
            f"  {decision.harm_type:<12} {verdict:<18} max ratio {ratio:>6}  "
            f"over {decision.n_over_threshold}/{decision.n_comparisons}"
        )
    for line in result.drift_disagreements:
        print(f"  ! DISAGREES WITH P7: {line}")
    print()

    paths = export_stats(result, config=config, out_dir=args.out_dir)
    print("wrote")
    for path in paths.values():
        print(f"  {path}")

    cfg = None
    if not args.no_manifest:
        overrides = [f"run_name={args.run_name}", f"stats_name={args.stats}"]
        if args.results_dir is not None:
            overrides.append(f"results_dir={args.results_dir.as_posix()}")
        cfg = load_experiment_config(
            config_dir=args.config_dir or default_config_dir(), overrides=overrides
        )
        manifest = write_run_manifest(
            cfg,
            inputs={
                "lattice_config": file_sha256(
                    default_lattice_config_dir() / f"{args.lattice}.yaml"
                ),
                **stats_inputs(
                    stats_config_path=default_stats_config_dir() / f"{args.stats}.yaml",
                    metrics_paths=inputs.paths,
                ),
            },
            stats=stats_record(
                config,
                result,
                stats_name=args.stats,
                benchmark=BENCHMARK,
                min_cell_n=min_cell_n,
            ),
        )
        print(f"\nmanifest   {manifest_path(cfg, manifest.run_id)}")

    if not args.no_paper_numbers and cfg is not None:
        registered = _paper_numbers(
            result,
            config=config,
            min_cell_n=min_cell_n,
            run_id=manifest.run_id,
            manifest_path=manifest_path(cfg, manifest.run_id),
        )
        existing = read_paper_numbers(args.paper_numbers)
        write_paper_numbers(existing.merged(registered), args.paper_numbers)
        print(f"paper      {len(registered)} numbers -> {args.paper_numbers}")
    return 0


def _paper_numbers(
    result: object,
    *,
    config: object,
    min_cell_n: int,
    run_id: str,
    manifest_path: Path,
) -> list[PaperNumber]:
    """The manuscript-facing numbers this stage owns."""
    from margins.stats.analysis import StatsResult
    from margins.stats.config import StatsConfig

    assert isinstance(result, StatsResult) and isinstance(config, StatsConfig)
    common = {
        "run_id": run_id,
        "manifest_path": manifest_path.as_posix(),
        "source": "scripts/run_stats.py",
    }
    return [
        PaperNumber(key="fdr.family_size", value=result.family_size, unit="count", **common),
        PaperNumber(key="fdr.alpha", value=config.fdr.alpha, unit="probability", **common),
        PaperNumber(key="fdr.method", value=config.fdr.method, unit="none", **common),
        PaperNumber(
            key="fdr.n_significant_pre", value=result.n_significant_pre, unit="count", **common
        ),
        PaperNumber(
            key="fdr.n_significant_post", value=result.n_significant_post, unit="count", **common
        ),
        PaperNumber(key="fdr.n_claimable", value=result.n_claimable, unit="count", **common),
        PaperNumber(key="fdr.n_refused", value=result.n_refused, unit="count", **common),
        PaperNumber(
            key="drift.descriptive_only_harm_types",
            value=", ".join(result.descriptive_only_harm_types) or "none",
            unit="none",
            **common,
        ),
        PaperNumber(key="power.min_cell_n", value=min_cell_n, unit="renderings", **common),
        PaperNumber(key="power.effect_size", value=config.power.effect_size, unit="sd", **common),
        *(
            []
            if result.b3 is None
            else [
                PaperNumber(
                    key="b3.n_terms_tested", value=result.b3.n_terms, unit="count", **common
                ),
                PaperNumber(
                    key="b3.n_permutations",
                    value=result.b3.n_permutations,
                    unit="count",
                    **common,
                ),
                PaperNumber(
                    key="b3.n_p_le_alpha",
                    value=sum(1 for p in result.b3.p_values.values() if p <= config.fdr.alpha),
                    unit="count",
                    **common,
                ),
            ]
        ),
    ]


if __name__ == "__main__":
    raise SystemExit(main())
