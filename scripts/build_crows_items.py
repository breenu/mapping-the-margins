"""Ingest and screen CrowS-Pairs, then write the P2 artifacts + a run manifest (P2).

Regenerates ``data/benchmarks/crows_pairs/items_v1.jsonl``, ``excluded_items.csv``,
``screening_report.md`` and ``coverage_report.md`` from the pinned source file plus a benchmark
config and the lattice. Every output is byte-deterministic, so re-running on unchanged inputs
produces unchanged files -- which is how gate P2's "screening is deterministic" is checked.

The manifest records the sha256 of **both** the benchmark config and the pinned CSV. The config
digest alone would not notice a swapped data file, and the data digest alone would not notice a
changed screening threshold; the two together are what make the item set reconstructible.

Nothing here loads a model. P2 is ingestion only -- the prompts are P4b
(``scripts/build_crows_prompts.py``) and inference is P5.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from margins.benchmarks import (
    DEFAULT_BENCHMARK,
    CrowsPairsLoader,
    default_benchmark_config_dir,
    load_benchmark_config,
)
from margins.benchmarks.crows_pairs import crows_pairs_path
from margins.benchmarks.export import benchmark_inputs, export_benchmark
from margins.config import (
    default_config_dir,
    load_experiment_config,
    manifest_path,
    write_run_manifest,
)
from margins.lattice import DEFAULT_LATTICE, DeclaredLattice, load_lattice_config
from margins.lattice.config import default_lattice_config_dir
from margins.stats.numbers import (
    PAPER_NUMBERS_PATH,
    PaperNumber,
    read_paper_numbers,
    write_paper_numbers,
)

#: The measured structural facts about the extractive paradigm, registered by key so the
#: manuscript and the cross-paradigm figure quote them rather than retyping them. They are the
#: evidence for a claim about **sentence-pair benchmarks**, never a null result about the model:
#: CrowS-Pairs cannot express an intersectional effect, which is a different statement from not
#: finding one.
_COVERAGE_NOTE = (
    "Measured by the P2 screen against the pinned CrowS-Pairs revision. Each item is annotated "
    "with exactly one attribute, so an order-2 cell needs a second identity term present "
    "unchanged in BOTH sentences. A fact about what sentence-pair benchmarks can support, never "
    "a null result about the model."
)


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK, help="Benchmark config name.")
    parser.add_argument("--lattice", default=DEFAULT_LATTICE, help="Lattice config name.")
    parser.add_argument("--benchmark-config-dir", type=Path, default=None)
    parser.add_argument("--lattice-config-dir", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=None, help="Root Hydra config dir.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Where the pinned CSV lives.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/benchmarks/crows_pairs"),
        help="Where to write the exports.",
    )
    parser.add_argument("--results-dir", type=Path, default=None, help="Override results_dir.")
    parser.add_argument("--run-name", default="build-crows-items")
    parser.add_argument(
        "--timestamp",
        type=_parse_timestamp,
        default=None,
        help="ISO timestamp recorded in the manifest. Pass it to make run_id reproducible.",
    )
    parser.add_argument(
        "--no-manifest", action="store_true", help="Skip the run manifest (exports only)."
    )
    parser.add_argument("--paper-numbers", type=Path, default=PAPER_NUMBERS_PATH)
    parser.add_argument(
        "--no-paper-numbers", action="store_true", help="Skip results/paper_numbers.json."
    )
    args = parser.parse_args(argv)

    benchmark_dir = args.benchmark_config_dir or default_benchmark_config_dir()
    lattice_dir = args.lattice_config_dir or default_lattice_config_dir()
    benchmark_file = benchmark_dir / f"{args.benchmark}.yaml"

    config = load_benchmark_config(args.benchmark, config_dir=benchmark_dir)
    lattice = DeclaredLattice(load_lattice_config(args.lattice, config_dir=lattice_dir))

    loader = CrowsPairsLoader(config, lattice, data_dir=args.data_dir)
    result = loader.screen()
    n_rows = result.n_kept + result.n_excluded

    paths = export_benchmark(result, config, args.out_dir, n_rows=n_rows)

    print(f"benchmark_version: {config.benchmark_version}")
    print(f"source revision:   {config.source.revision}")
    print(f"source sha256:     {config.source.sha256}")
    print(f"lattice_version:   {lattice.config.lattice_version}")

    print("\nscreening")
    print(f"  {'rows read':<34} {n_rows}")
    print(f"  {'kept':<34} {result.n_kept}")
    print(f"  {'excluded':<34} {result.n_excluded}")
    for reason, count in result.counts_by_reason().items():
        print(f"    {reason:<32} {count}")
    print(f"  {'admitted by concord allowance':<34} {len(result.admitted_by_agreement)}")

    print("\nlattice coverage")
    for order, count in result.counts_by_order().items():
        print(f"  {order:<34} {count}")
    order_2 = result.order_2_cells()
    print(f"  {'order-2 cells populated':<34} {len(order_2)}")
    for cell_id, count in order_2.items():
        print(f"    {cell_id:<32} {count}")

    print("\nwrote")
    for path in paths.values():
        print(f"  {path}")

    if args.no_manifest:
        print("\nno manifest written (--no-manifest)")
        return 0

    overrides = [
        f"run_name={args.run_name}",
        f"lattice_name={args.lattice}",
        f"benchmark_name={args.benchmark}",
    ]
    if args.results_dir is not None:
        overrides.append(f"results_dir={args.results_dir.as_posix()}")
    cfg = load_experiment_config(
        config_dir=args.config_dir or default_config_dir(), overrides=overrides
    )

    inputs = benchmark_inputs(config, config_path=benchmark_file, data_dir=args.data_dir)
    manifest = write_run_manifest(cfg, inputs=inputs, timestamp=args.timestamp)

    print("\nrun manifest")
    print(f"  {'run_id':<34} {manifest.run_id}")
    print(f"  {'config.benchmark_name':<34} {manifest.config.benchmark_name}")
    for key in sorted(manifest.input_checksums):
        print(f"  {key:<34} {manifest.input_checksums[key]}")
    print(f"\nwrote {manifest_path(cfg, manifest.run_id)}")
    print(f"\nsource file: {crows_pairs_path(args.data_dir)}")

    if not args.no_paper_numbers:
        by_order = result.counts_by_order()
        common = {
            "run_id": manifest.run_id,
            "manifest_path": manifest_path(cfg, manifest.run_id).as_posix(),
            "source": "scripts/build_crows_items.py",
            "note": _COVERAGE_NOTE,
        }
        entries = [
            PaperNumber(key="crows.rows_read", value=n_rows, unit="count", **common),
            PaperNumber(key="crows.kept", value=result.n_kept, unit="count", **common),
            PaperNumber(key="crows.excluded", value=result.n_excluded, unit="count", **common),
            PaperNumber(
                key="crows.order_1_items", value=by_order.get("order_1", 0), unit="count", **common
            ),
            PaperNumber(
                key="crows.order_2_items", value=by_order.get("order_2", 0), unit="count", **common
            ),
            PaperNumber(
                key="crows.no_cell_items", value=by_order.get("no_cell", 0), unit="count", **common
            ),
            PaperNumber(
                key="crows.order_2_cells_populated", value=len(order_2), unit="count", **common
            ),
        ]
        existing = read_paper_numbers(args.paper_numbers)
        write_paper_numbers(existing.merged(entries), args.paper_numbers)
        print(f"\npaper numbers: {len(entries)} -> {args.paper_numbers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
