"""Build the elicitation prompts that ask the red LM for frames (P8, stage 1 of 3).

Writes ``data/redteam/elicitation_prompts.jsonl`` plus a run manifest. Byte-deterministic: the
same config and lattice produce the same prompts with the same ids.

**These prompts are not audit prompts.** They address the *red LM*, they deliberately NAME the
target identity -- that is what makes the red-teaming cell-targeted -- and they are never scored.
The P4a leakage guard is deliberately not applied to them; it applies with full force one stage
later, to the frames the red LM writes back (``scripts/build_redteam_frames.py``).

Nothing here loads a model. To actually generate, run the EXISTING P5 harness against this file::

    uv run python scripts/run_inference.py --model llama31_8b_instruct \\
        --prompts data/redteam/elicitation_prompts.jsonl --decoding redteam_v1 \\
        --seeds <the candidates_per_target seeds this script prints>

That is the whole point of elicitation prompts being ordinary ``Prompt`` records: red-team
generation is resumable, content-addressed and manifest-writing because P5 already is.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from margins.config import (
    default_config_dir,
    load_experiment_config,
    manifest_path,
    write_run_manifest,
)
from margins.lattice import DEFAULT_LATTICE, DeclaredLattice, load_lattice_config
from margins.lattice.config import default_lattice_config_dir
from margins.prompts import default_prompt_config_dir, render_prompts_jsonl
from margins.prompts.config import file_sha256
from margins.redteam import (
    DEFAULT_REDTEAM_SET,
    default_elicitation_dir,
    default_redteam_config_dir,
    elicitation_prompts,
    elicitation_seeds,
    load_base_prompt_set,
    load_redteam_config,
    redteam_inputs,
    target_cells,
)
from margins.redteam.config import load_elicitation_files

ELICITATION_FILENAME = "elicitation_prompts.jsonl"


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redteam", default=DEFAULT_REDTEAM_SET, help="Red-team set config name.")
    parser.add_argument("--lattice", default=DEFAULT_LATTICE, help="Lattice config name.")
    parser.add_argument("--redteam-config-dir", type=Path, default=None)
    parser.add_argument("--elicitation-dir", type=Path, default=None)
    parser.add_argument("--lattice-config-dir", type=Path, default=None)
    parser.add_argument("--prompt-config-dir", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=None, help="Root Hydra config dir.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/redteam"))
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--run-name", default="build-redteam-elicitation")
    parser.add_argument("--timestamp", type=_parse_timestamp, default=None)
    parser.add_argument("--no-manifest", action="store_true")
    args = parser.parse_args(argv)

    redteam_dir = args.redteam_config_dir or default_redteam_config_dir()
    elicitation_dir = args.elicitation_dir or default_elicitation_dir()
    lattice_dir = args.lattice_config_dir or default_lattice_config_dir()
    prompt_dir = args.prompt_config_dir or default_prompt_config_dir()

    config = load_redteam_config(args.redteam, config_dir=redteam_dir)
    lattice = DeclaredLattice(load_lattice_config(args.lattice, config_dir=lattice_dir))
    base = load_base_prompt_set(config, config_dir=prompt_dir)
    prompt_set = config.prompt_set(base)

    files = load_elicitation_files(config.elicitation.templates, elicitation_dir)
    config_path = redteam_dir / f"{args.redteam}.yaml"
    _, prompts = elicitation_prompts(
        config, lattice, files, prompt_set, config_sha256=file_sha256(config_path)
    )

    out_path = args.out_dir / ELICITATION_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(render_prompts_jsonl(prompts))

    cells = target_cells(config, lattice)
    seeds = elicitation_seeds(config)
    print(f"redteam_version:  {config.redteam_version}")
    print(f"generator:        {config.generator.model_name} @ {config.generator.decoding_name}")
    print(f"lattice_version:  {lattice.config.lattice_version}")
    print(f"base prompt set:  {config.base_prompt_set} ({base.template_version})")
    print("\nsummary")
    print(f"  {'elicitation templates':<28} {len(files)}")
    print(f"  {'target cells':<28} {len(cells)}")
    print(f"  {'elicitation prompts':<28} {len(prompts)}")
    print(f"  {'seeds per prompt':<28} {len(seeds)}  {list(seeds)}")
    print(f"  {'planned generations':<28} {len(prompts) * len(seeds)}")
    print(f"\nwrote {out_path}")
    # `--seeds` is NOT optional here. run_inference.py defaults to the config's single seed, which
    # would draw ONE candidate per (template x target) instead of the configured
    # `candidates_per_target` -- a sixth of the frame pool, silently. The hint omitted it until P11.
    print(
        "\nnext: uv run python scripts/run_inference.py "
        f"--model {config.generator.model_name} --prompts {out_path.as_posix()} "
        f"--decoding {config.generator.decoding_name} "
        f"--seeds {' '.join(str(seed) for seed in seeds)}"
    )

    if args.no_manifest:
        print("\nno manifest written (--no-manifest)")
        return 0

    overrides = [
        f"run_name={args.run_name}",
        f"lattice_name={args.lattice}",
        f"redteam_name={args.redteam}",
        f"decoding_name={config.generator.decoding_name}",
    ]
    if args.results_dir is not None:
        overrides.append(f"results_dir={args.results_dir.as_posix()}")
    cfg = load_experiment_config(
        config_dir=args.config_dir or default_config_dir(), overrides=overrides
    )

    inputs = redteam_inputs(
        lattice_config_path=lattice_dir / f"{args.lattice}.yaml",
        redteam_config_path=config_path,
        elicitation_paths={file.id: file.path for file in files},
    )
    manifest = write_run_manifest(cfg, inputs=inputs, timestamp=args.timestamp)

    print("\nrun manifest")
    print(f"  {'run_id':<28} {manifest.run_id}")
    for key in sorted(manifest.input_checksums):
        print(f"  {key:<28} {manifest.input_checksums[key]}")
    print(f"\nwrote {manifest_path(cfg, manifest.run_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
