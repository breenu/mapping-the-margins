"""Assemble the committed result tables from a real P7 + P9 run.

    uv run python scripts/build_result_tables.py

Reads ``data/stats/`` (P9) and one or more ``data/metrics/`` directories (P7) and writes
``results/tables/*.csv`` plus their LaTeX forms. Those tables are what the manuscript prints and
what ``scripts/make_figures.py`` draws, so the paper's numbers, its tables and its figures all
come out of one place and cannot disagree.

``data/`` is gitignored and regenerable; ``results/`` is committed. These tables are the seam
between the two: small, final, and worth keeping in git beside ``paper_numbers.json`` and the run
manifests.

**Before the P11 confirmatory run there is nothing here to read**, and the script says so and
exits non-zero rather than writing empty tables. Until then the figures draw from
``results/fixtures/`` (see ``scripts/build_fixture_results.py``), every such figure carries a
banner, and no figure code differs between the two.

The ``sensitivity`` and ``frame_breakdown`` tables are E4 and per-frame analyses that P7 and P9 do
not emit; they arrive with the P11 experiment runs. The script reports them as pending rather than
pretending they are empty.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from margins.metrics.export import (
    CELL_STATS_FILENAME,
    DEFAULT_METRICS_DIR,
    DRIFT_COMPARISON_FILENAME,
    EFFECTS_FILENAME,
    FIT_REPORTS_FILENAME,
    FIT_TERMS_FILENAME,
    read_records,
)
from margins.metrics.records import DriftComparison, FitReport, FitTerm
from margins.report.build import (
    descriptive_only_rows,
    drift_rows,
    emergence_rows,
    fit_report_rows,
    single_axis_rows,
)
from margins.report.tables import (
    DEFAULT_TABLE_DIR,
    spec_for,
    write_table_csv,
    write_table_latex,
)
from margins.report.tables import read_table as read_result_table
from margins.schemas import CellStat, EffectEstimate
from margins.stats.config import DEFAULT_STATS, StatsConfig, load_stats_config
from margins.stats.export import (
    CLAIMS_FILENAME,
    DEFAULT_STATS_DIR,
    DESCRIPTIVE_ONLY_FILENAME,
    EFFECTS_CORRECTED_FILENAME,
    FAMILY_FILENAME,
)
from margins.stats.records import DescriptiveOnlyDecision, FamilyMember, SignificanceClaim

#: Tables that only exist once the P11 experiment runs have happened.
PENDING = ("sensitivity", "frame_breakdown")


def _read_many(paths: list[Path], record: type):  # type: ignore[no-untyped-def]
    out = []
    for path in paths:
        if path.is_file():
            out.extend(read_records(path, record))
    return out


def _refuse_significance_on_descriptive(
    effects: list[EffectEstimate], stats_config: StatsConfig
) -> None:
    """A descriptive arm may not carry a p or a q, nor masquerade as a family source.

    The red-team arm is descriptive by decision (21 Sep 2026): its estimates go to the tables with
    CIs and nothing else. An effect arriving here with significance metadata was set somewhere
    other than P9 -- invariant 3 -- and a family-source effect arriving here would bypass the
    correction it belongs in.
    """
    family_sources = set(stats_config.fdr.family_benchmarks)
    for effect in effects:
        if effect.benchmark in family_sources:
            raise ValueError(
                f"a {effect.benchmark!r} effect was passed as DESCRIPTIVE, but that benchmark is "
                "an FDR-family source. Pass its directory with --metrics-dir, through P9."
            )
        carried = {
            name: value
            for name, value in (
                ("p", effect.p),
                ("q", effect.q),
                ("family_size", effect.family_size),
                ("correction", effect.correction),
            )
            if value is not None
        }
        if carried:
            raise ValueError(
                f"descriptive effect {effect.benchmark}/{effect.model}/{effect.term} carries "
                f"{sorted(carried)}. Only StatsEngine sets significance (invariant 3), and a "
                "descriptive arm never goes through it."
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS_DIR)
    parser.add_argument("--metrics-dir", type=Path, nargs="+", default=[DEFAULT_METRICS_DIR])
    parser.add_argument(
        "--descriptive-metrics-dir",
        type=Path,
        nargs="+",
        default=[],
        help="P7 directories of a DESCRIPTIVE arm (the red-team subset) that P9 never corrects. "
        "Their uncorrected effects join the tables with no p and no q; any that carries either, or "
        "whose benchmark is an FDR-family source, is refused.",
    )
    parser.add_argument("--stats", default=DEFAULT_STATS, help="Stats config naming the family.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--no-latex", action="store_true")
    args = parser.parse_args(argv)

    effects_path = args.stats_dir / EFFECTS_CORRECTED_FILENAME
    if not effects_path.is_file():
        print(
            f"NOTHING TO ASSEMBLE.\n"
            f"  {effects_path} does not exist, so no corrected estimate has been produced yet.\n"
            "  Run scripts/run_metrics.py and scripts/run_stats.py against a full sampled run "
            "first.\n"
            "  Until then the figures draw from results/fixtures/ and say so on their face:\n"
            "    uv run python scripts/build_fixture_results.py\n"
            "    uv run python scripts/make_figures.py"
        )
        return 1

    effects = read_records(effects_path, EffectEstimate)
    descriptive = _read_many(
        [d / EFFECTS_FILENAME for d in args.descriptive_metrics_dir], EffectEstimate
    )
    _refuse_significance_on_descriptive(descriptive, load_stats_config(args.stats))
    effects = [*effects, *descriptive]

    claims = read_records(args.stats_dir / CLAIMS_FILENAME, SignificanceClaim)
    family = read_records(args.stats_dir / FAMILY_FILENAME, FamilyMember)
    decisions = read_records(args.stats_dir / DESCRIPTIVE_ONLY_FILENAME, DescriptiveOnlyDecision)
    metric_dirs = [*args.metrics_dir, *args.descriptive_metrics_dir]
    fit_terms = _read_many([d / FIT_TERMS_FILENAME for d in metric_dirs], FitTerm)
    reports = _read_many([d / FIT_REPORTS_FILENAME for d in metric_dirs], FitReport)
    comparisons = _read_many([d / DRIFT_COMPARISON_FILENAME for d in metric_dirs], DriftComparison)
    cells = _read_many([d / CELL_STATS_FILENAME for d in metric_dirs], CellStat)

    tables = {
        "emergence_effects": emergence_rows(effects, fit_terms, claims, family),
        "descriptive_only": descriptive_only_rows(decisions),
        "drift_side_by_side": drift_rows(comparisons),
        "fit_reports": fit_report_rows(reports),
        "single_axis_cells": single_axis_rows(cells),
    }

    print(f"stats      {args.stats_dir}")
    print(f"metrics    {', '.join(str(d) for d in args.metrics_dir)}")
    print(f"out        {args.out_dir}\n")
    for name in sorted(tables):
        spec = spec_for(name)
        path = write_table_csv(spec, tables[name], args.out_dir / f"{name}.csv")
        print(f"  {name:<22} {len(tables[name]):>5} rows  -> {path}")
        if not args.no_latex:
            write_table_latex(
                read_result_table(name, directory=args.out_dir), args.out_dir / f"{name}.tex"
            )

    for name in PENDING:
        print(f"  {name:<22} {'—':>5}       pending: emitted by the P11 experiment runs")

    print(
        "\nscripts/make_figures.py now prefers results/tables/ over results/fixtures/ "
        "automatically."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
