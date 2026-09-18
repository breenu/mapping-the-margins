"""Render the templated ``open_gen`` prompt set and write its audits + run manifest (P4a).

Regenerates ``data/prompts/prompts_v1.jsonl``, ``data/prompts/template_fill_audit.md`` and
``data/prompts/phrase_audit.md`` from a lattice config plus a prompt-set config. Every output is
byte-deterministic, so re-running on unchanged inputs produces unchanged files -- that is gate
P4a's "byte-identical prompt regeneration".

It also writes a **run manifest** recording the sha256 of the lattice config, the prompt-set
config, and every template file. Before P4a a manifest recorded the experiment config but not
*which identity lattice* produced the prompts; this closes that hole. The lattice digest is the
same one ``scripts/build_lattice.py`` prints, so the two are checkably the same file.

Nothing here loads a model. P4a is prompt generation only -- inference is P5.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

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
    DEFAULT_PROMPT_SET,
    TemplateRenderer,
    default_prompt_config_dir,
    export_prompt_set,
    load_prompt_config,
    load_template_files,
    prompt_set_inputs,
    template_items,
)
from margins.prompts.config import TEMPLATE_DIRNAME


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lattice", default=DEFAULT_LATTICE, help="Lattice config name.")
    parser.add_argument("--prompt-set", default=DEFAULT_PROMPT_SET, help="Prompt config name.")
    parser.add_argument("--lattice-config-dir", type=Path, default=None)
    parser.add_argument("--prompt-config-dir", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=None, help="Root Hydra config dir.")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/prompts"), help="Where to write the exports."
    )
    parser.add_argument("--results-dir", type=Path, default=None, help="Override results_dir.")
    parser.add_argument("--run-name", default="build-prompts")
    parser.add_argument(
        "--timestamp",
        type=_parse_timestamp,
        default=None,
        help="ISO timestamp recorded in the manifest. Pass it to make run_id reproducible.",
    )
    parser.add_argument(
        "--no-manifest", action="store_true", help="Skip the run manifest (exports only)."
    )
    args = parser.parse_args(argv)

    lattice_dir = args.lattice_config_dir or default_lattice_config_dir()
    prompt_dir = args.prompt_config_dir or default_prompt_config_dir()
    lattice_file = lattice_dir / f"{args.lattice}.yaml"
    prompt_file = prompt_dir / f"{args.prompt_set}.yaml"

    lattice = DeclaredLattice(load_lattice_config(args.lattice, config_dir=lattice_dir))
    prompt_config = load_prompt_config(args.prompt_set, config_dir=prompt_dir)
    files = load_template_files(prompt_config, prompt_dir / TEMPLATE_DIRNAME)

    items = template_items(prompt_config, files)
    renderer = TemplateRenderer(prompt_config, lattice)
    prompts = renderer.render(items, lattice.cells())

    prompts_path, fill_audit, phrase_audit = export_prompt_set(
        lattice, prompt_config, prompts, files, args.out_dir
    )

    print(f"lattice_version:  {lattice.config.lattice_version}")
    print(f"template_version: {prompt_config.template_version}")
    print(f"decoding_key:     {prompt_config.decoding_key}  (resolved at P5: configs/decoding/)")
    print("\nsummary")
    print(f"  {'templates':<32} {len(files)}")
    print(f"  {'cells':<32} {len(lattice.cells())}")
    print(f"  {'carriers':<32} {len(prompt_config.carriers)}")
    print(f"  {'prompts':<32} {len(prompts)}")
    print(f"  {'unique prompt ids':<32} {len({p.prompt_id for p in prompts})}")

    print("\nwrote")
    for path in (prompts_path, fill_audit, phrase_audit):
        print(f"  {path}")

    if args.no_manifest:
        print("\nno manifest written (--no-manifest)")
        return 0

    overrides = [f"run_name={args.run_name}", f"lattice_name={args.lattice}"]
    if args.results_dir is not None:
        overrides.append(f"results_dir={args.results_dir.as_posix()}")
    cfg = load_experiment_config(
        config_dir=args.config_dir or default_config_dir(), overrides=overrides
    )

    inputs = prompt_set_inputs(
        lattice_config_path=lattice_file, prompt_config_path=prompt_file, files=files
    )
    manifest = write_run_manifest(cfg, inputs=inputs, timestamp=args.timestamp)

    print("\nrun manifest")
    print(f"  {'run_id':<32} {manifest.run_id}")
    print(f"  {'config.lattice_name':<32} {manifest.config.lattice_name}")
    lattice_key = LATTICE_INPUT_KEY
    print(f"  {lattice_key:<32} {manifest.input_checksums[lattice_key]}")
    for key in sorted(manifest.input_checksums):
        if key != LATTICE_INPUT_KEY:
            print(f"  {key:<32} {manifest.input_checksums[key]}")
    print(f"\nwrote {manifest_path(cfg, manifest.run_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
