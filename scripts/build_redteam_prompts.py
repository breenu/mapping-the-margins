"""Render the screened frames into the red-team audit prompt set (P8, stage 3 of 3).

Writes ``data/prompts/<set>.jsonl`` — the prompts the **target** model sees. Byte-deterministic:
re-running on unchanged frames produces unchanged bytes.

**Every frame renders into every target cell.** Not only the cell it was elicited for. That is the
design decision the whole phase rests on: a frame confined to its own target would make "frame
targeted at cell X" perfectly confounded with "cell X", so any difference between frames would land
on that cell's mean and read as an identity effect. It is also what makes red-team effects
estimable — a frame present in every cell of a pair's grid is one item cluster spanning it, where a
per-cell prompt is a singleton. See :mod:`margins.redteam.frames`.

The leakage guard runs on every prompt emitted here, inside the unchanged
:class:`~margins.prompts.render.TemplateRenderer`. It is not checked by this script, and that is
deliberate: a guarantee that holds only when a script remembers to call a checker is not a
guarantee.

From here the red-team set flows down the **existing** path, with no new scoring or metric code::

    uv run python scripts/run_inference.py --model llama31_8b_instruct \\
        --prompts data/prompts/redteam_v1.jsonl --decoding sampled_v1
    uv run python scripts/run_scoring.py --responses <file> \\
        --prompts data/prompts/redteam_v1.jsonl --benchmark redteam \\
        --out-dir data/scoring/redteam
    uv run python scripts/run_metrics.py --scores data/scoring/redteam/scores.parquet \\
        --benchmark redteam --out-dir data/metrics/redteam
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
from margins.inference.models import load_model_config
from margins.lattice import DEFAULT_LATTICE, DeclaredLattice, load_lattice_config
from margins.lattice.config import default_lattice_config_dir
from margins.prompts import default_prompt_config_dir, render_prompts_jsonl
from margins.prompts.config import file_sha256
from margins.redteam import (
    DEFAULT_REDTEAM_SET,
    FRAMES_FILENAME,
    default_elicitation_dir,
    default_redteam_config_dir,
    frame_stratum,
    load_base_prompt_set,
    load_redteam_config,
    read_frames,
    redteam_inputs,
    render_audit_prompts,
    select_frames,
    target_cells,
)
from margins.redteam.config import load_elicitation_files


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redteam", default=DEFAULT_REDTEAM_SET, help="Red-team set config name.")
    parser.add_argument("--lattice", default=DEFAULT_LATTICE, help="Lattice config name.")
    parser.add_argument(
        "--frames",
        type=Path,
        default=Path("data/redteam") / FRAMES_FILENAME,
        help="Frame JSONL written by scripts/build_redteam_frames.py.",
    )
    parser.add_argument("--redteam-config-dir", type=Path, default=None)
    parser.add_argument("--elicitation-dir", type=Path, default=None)
    parser.add_argument("--lattice-config-dir", type=Path, default=None)
    parser.add_argument("--prompt-config-dir", type=Path, default=None)
    parser.add_argument("--model-config-dir", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=None, help="Root Hydra config dir.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/prompts"))
    parser.add_argument(
        "--all-cells",
        action="store_true",
        help="Render into every lattice cell, not only the confirmatory targets. Off by default: "
        "gate P8 restricts red-teaming to the confirmatory axis pairs.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Render only this many frames, chosen round-robin over (elicitation angle x target "
        "pair) strata in content order (P11's DESCRIPTIVE subset). Size it with "
        "`scripts/run_p11.py budget`. Default: every screened frame.",
    )
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--run-name", default="build-redteam-prompts")
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
    generator = load_model_config(config.generator.model_name, config_dir=args.model_config_dir)

    frames = read_frames(args.frames)
    if not frames:
        raise SystemExit(f"no frames in {args.frames}; run scripts/build_redteam_frames.py first")
    n_screened = len(frames)
    if args.max_frames is not None:
        pair_of = {
            cell.cell_id: (cell.axes[0][0], cell.axes[1][0])
            for cell in lattice.cells()
            if cell.order == 2
        }
        frames = select_frames(frames, args.max_frames, pair_of=pair_of)
        selection_path = args.out_dir / f"{args.redteam}_frame_selection.csv"
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        with selection_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("frame_id,elicitation_template,axis_a,axis_b,target_cell_id\n")
            for frame in frames:
                angle, axis_a, axis_b = frame_stratum(frame, pair_of)
                handle.write(f"{frame.frame_id},{angle},{axis_a},{axis_b},{frame.target_cell_id}\n")

    cells = lattice.cells() if args.all_cells else target_cells(config, lattice)
    prompts = render_audit_prompts(
        frames,
        config=config,
        prompt_set=prompt_set,
        lattice=lattice,
        cells=cells,
        generator=generator,
    )

    out_path = args.out_dir / f"{args.redteam}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(render_prompts_jsonl(prompts))

    per_cell = len(prompts) / len(cells) if cells else 0.0
    print(f"redteam_version:  {config.redteam_version}")
    print(f"decoding_key:     {config.decoding_key}  (the TARGET model's profile)")
    print(f"lattice_version:  {lattice.config.lattice_version}")
    print(f"base prompt set:  {config.base_prompt_set} ({base.template_version})")
    print("\nsummary")
    print(f"  {'frames screened':<28} {n_screened}")
    print(f"  {'frames rendered':<28} {len(frames)}")
    if args.max_frames is not None:
        print(
            f"  (capped at --max-frames {args.max_frames}: the DESCRIPTIVE red-team subset, "
            "powered=False by design)"
        )
    print(f"  {'cells rendered into':<28} {len(cells)}")
    print(f"  {'carriers':<28} {len(prompt_set.carriers)}")
    print(f"  {'prompts':<28} {len(prompts)}")
    print(f"  {'unique prompt ids':<28} {len({p.prompt_id for p in prompts})}")
    print(f"  {'prompts per cell':<28} {per_cell:.1f}")
    print(
        f"\n  Every frame is rendered into every one of the {len(cells)} cells, so each frame is "
        "one\n  ITEM CLUSTER spanning the grid. Renderings per cell at run time is this times the\n"
        "  number of sampled seeds; the `powered` threshold it must clear is the lattice's\n"
        f"  min_cell_n = {lattice.config.power.min_cell_n}."
    )
    print(f"\nwrote {out_path}")

    if args.no_manifest:
        print("\nno manifest written (--no-manifest)")
        return 0

    overrides = [
        f"run_name={args.run_name}",
        f"lattice_name={args.lattice}",
        f"redteam_name={args.redteam}",
        f"decoding_name={config.decoding_key}",
    ]
    if args.results_dir is not None:
        overrides.append(f"results_dir={args.results_dir.as_posix()}")
    cfg = load_experiment_config(
        config_dir=args.config_dir or default_config_dir(), overrides=overrides
    )

    files = load_elicitation_files(config.elicitation.templates, elicitation_dir)
    inputs = redteam_inputs(
        lattice_config_path=lattice_dir / f"{args.lattice}.yaml",
        redteam_config_path=redteam_dir / f"{args.redteam}.yaml",
        elicitation_paths={file.id: file.path for file in files},
    )
    inputs["base_prompt_set_config"] = file_sha256(prompt_dir / f"{config.base_prompt_set}.yaml")
    manifest = write_run_manifest(cfg, inputs=inputs, timestamp=args.timestamp)

    print("\nrun manifest")
    print(f"  {'run_id':<28} {manifest.run_id}")
    for key in sorted(manifest.input_checksums):
        print(f"  {key:<28} {manifest.input_checksums[key]}")
    print(f"\nwrote {manifest_path(cfg, manifest.run_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
