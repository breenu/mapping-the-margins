"""Render the screened CrowS-Pairs items into paired prompts + a run manifest (P4b).

Regenerates ``data/prompts/crows_pll_v1.jsonl`` from the item file
``scripts/build_crows_items.py`` wrote, plus the extractive prompt-set config and the lattice. The
output is byte-deterministic, so re-running on unchanged inputs produces unchanged bytes -- that is
gate P4b's "byte-identical regeneration from the same config".

The other half of gate P4b -- *emits no cell unattested in its source item* -- is enforced inside
the renderer on every prompt, not checked here. A guarantee that only holds when a script
remembers to call a checker is not a guarantee.

Nothing here loads a model. Inference is P5 (``scripts/run_inference.py``).
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from margins.benchmarks import DEFAULT_BENCHMARK, default_benchmark_config_dir
from margins.benchmarks.config import BENCHMARK_INPUT_KEY, file_sha256
from margins.benchmarks.export import ITEMS_FILENAME, read_items
from margins.config import (
    LATTICE_INPUT_KEY,
    default_config_dir,
    load_experiment_config,
    manifest_path,
    write_run_manifest,
)
from margins.lattice import DEFAULT_LATTICE, DeclaredLattice, load_lattice_config
from margins.lattice.config import default_lattice_config_dir
from margins.prompts import (
    DEFAULT_EXTRACTIVE_SET,
    EXTRACTIVE_SET_INPUT_KEY,
    ExtractiveRenderer,
    default_prompt_config_dir,
    load_extractive_config,
    render_prompts_jsonl,
)


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-set", default=DEFAULT_EXTRACTIVE_SET, help="Prompt set name.")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK, help="Benchmark config name.")
    parser.add_argument("--lattice", default=DEFAULT_LATTICE, help="Lattice config name.")
    parser.add_argument("--prompt-config-dir", type=Path, default=None)
    parser.add_argument("--benchmark-config-dir", type=Path, default=None)
    parser.add_argument("--lattice-config-dir", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=None, help="Root Hydra config dir.")
    parser.add_argument(
        "--items",
        type=Path,
        default=Path("data/benchmarks/crows_pairs") / ITEMS_FILENAME,
        help="Item JSONL written by scripts/build_crows_items.py.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/prompts"), help="Where to write the prompts."
    )
    parser.add_argument(
        "--celled-only",
        action="store_true",
        help="Render only items that attest a lattice cell. Off by default: an item carrying no "
        "cell is still a scoreable pair and is reported, it simply cannot be fitted.",
    )
    parser.add_argument("--results-dir", type=Path, default=None, help="Override results_dir.")
    parser.add_argument("--run-name", default="build-crows-prompts")
    parser.add_argument("--timestamp", type=_parse_timestamp, default=None)
    parser.add_argument("--no-manifest", action="store_true")
    args = parser.parse_args(argv)

    prompt_dir = args.prompt_config_dir or default_prompt_config_dir()
    lattice_dir = args.lattice_config_dir or default_lattice_config_dir()
    benchmark_dir = args.benchmark_config_dir or default_benchmark_config_dir()

    config = load_extractive_config(args.prompt_set, config_dir=prompt_dir)
    lattice = DeclaredLattice(load_lattice_config(args.lattice, config_dir=lattice_dir))

    items = read_items(args.items)
    if args.celled_only:
        items = [item for item in items if item.payload.get("attested")]

    renderer = ExtractiveRenderer(config, lattice)
    prompts = renderer.render(items, lattice.cells())

    out_path = args.out_dir / f"{args.prompt_set}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(render_prompts_jsonl(prompts))

    celled = sum(1 for p in prompts if p.cell_id is not None)
    print(f"template_version: {config.template_version}")
    print(f"decoding_key:     {config.decoding_key}")
    print(f"lattice_version:  {lattice.config.lattice_version}")
    print("\nsummary")
    print(f"  {'items read':<28} {len(items)}")
    print(f"  {'prompts':<28} {len(prompts)}")
    print(f"  {'unique prompt ids':<28} {len({p.prompt_id for p in prompts})}")
    print(f"  {'carrying a cell':<28} {celled}")
    print(f"  {'carrying no cell':<28} {len(prompts) - celled}")
    print(f"\nwrote {out_path}")

    if args.no_manifest:
        print("\nno manifest written (--no-manifest)")
        return 0

    overrides = [
        f"run_name={args.run_name}",
        f"lattice_name={args.lattice}",
        f"benchmark_name={args.benchmark}",
        f"decoding_name={config.decoding_key}",
    ]
    if args.results_dir is not None:
        overrides.append(f"results_dir={args.results_dir.as_posix()}")
    cfg = load_experiment_config(
        config_dir=args.config_dir or default_config_dir(), overrides=overrides
    )

    inputs = {
        LATTICE_INPUT_KEY: file_sha256(lattice_dir / f"{args.lattice}.yaml"),
        BENCHMARK_INPUT_KEY: file_sha256(benchmark_dir / f"{args.benchmark}.yaml"),
        EXTRACTIVE_SET_INPUT_KEY: file_sha256(prompt_dir / f"{args.prompt_set}.yaml"),
    }
    manifest = write_run_manifest(cfg, inputs=inputs, timestamp=args.timestamp)

    print("\nrun manifest")
    print(f"  {'run_id':<28} {manifest.run_id}")
    for key in sorted(manifest.input_checksums):
        print(f"  {key:<28} {manifest.input_checksums[key]}")
    print(f"\nwrote {manifest_path(cfg, manifest.run_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
