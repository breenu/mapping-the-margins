"""Turn the red LM's generations into screened frames (P8, stage 2 of 3).

Reads the red LM's response file, parses ``FRAME:`` lines, applies the documented degenerate
filter, dedups, and **trial-renders every survivor through the unchanged P4a TemplateRenderer
across every target cell**. Writes:

* ``data/redteam/frames_v1.jsonl`` — the frames that rendered cleanly into every target cell
* ``data/redteam/rejected_frames.csv`` — every discarded candidate, one machine-readable reason each
* ``data/redteam/redteam_report.md`` — counts, rejections by reason, and measured per-pair coverage
* ``data/redteam/on_target_sample.jsonl`` — the stratified sample for the human on-target audit

All byte-deterministic. The filter is pure: no I/O, no RNG, no clock, no order dependence. Only the
on-target sample draws, and it routes through ``set_global_seed``.

**The leakage guard is authoritative here, not advisory.** A frame that raises when rendered is
rejected with reason ``leakage_guard`` rather than taking the run down — so a red-team run reports
"47 frames leaked identity" instead of dying on the first one, and nothing reaches the frame file
that has not already rendered cleanly into every cell it will be used in.

Nothing here loads a model: the generations were produced by ``scripts/run_inference.py``.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from margins.config import (
    RedTeamRecord,
    default_config_dir,
    load_experiment_config,
    manifest_path,
    write_run_manifest,
)
from margins.inference.decoding import DecodingConfig, load_decoding_config
from margins.inference.harness import read_responses
from margins.inference.models import load_model_config
from margins.lattice import DEFAULT_LATTICE, DeclaredLattice, load_lattice_config
from margins.lattice.config import default_lattice_config_dir
from margins.prompts import default_prompt_config_dir
from margins.prompts.config import file_sha256
from margins.prompts.export import read_prompts
from margins.redteam import (
    DEFAULT_REDTEAM_SET,
    ON_TARGET_SAMPLE_FILENAME,
    REDTEAM_BENCHMARK,
    default_elicitation_dir,
    default_redteam_config_dir,
    load_base_prompt_set,
    load_redteam_config,
    redteam_inputs,
    render_audit_prompts,
    render_redteam_report,
    sample_for_on_target,
    screen_candidates,
    target_cells,
    trial_render,
    write_frames,
    write_on_target_sample,
    write_redteam_report,
    write_rejected,
)
from margins.redteam.config import load_elicitation_files
from margins.redteam.filter import Rejection
from margins.redteam.parse import parse_responses

ELICITATION_FILENAME = "elicitation_prompts.jsonl"


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--responses",
        type=Path,
        required=True,
        help="Red LM response JSONL written by scripts/run_inference.py.",
    )
    parser.add_argument(
        "--elicitation",
        type=Path,
        default=Path("data/redteam") / ELICITATION_FILENAME,
        help="Elicitation prompt file; supplies each response's target cell.",
    )
    parser.add_argument("--redteam", default=DEFAULT_REDTEAM_SET, help="Red-team set config name.")
    parser.add_argument("--lattice", default=DEFAULT_LATTICE, help="Lattice config name.")
    parser.add_argument("--redteam-config-dir", type=Path, default=None)
    parser.add_argument("--elicitation-dir", type=Path, default=None)
    parser.add_argument("--lattice-config-dir", type=Path, default=None)
    parser.add_argument("--prompt-config-dir", type=Path, default=None)
    parser.add_argument("--model-config-dir", type=Path, default=None)
    parser.add_argument("--decoding-config-dir", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=None, help="Root Hydra config dir.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/redteam"))
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--run-name", default="build-redteam-frames")
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
    decoding: DecodingConfig = load_decoding_config(
        config.generator.decoding_name, config_dir=args.decoding_config_dir
    )

    cells = target_cells(config, lattice)
    elicitation = read_prompts(args.elicitation)
    target_by_prompt = {
        prompt.prompt_id: prompt.cell_id for prompt in elicitation if prompt.cell_id is not None
    }

    responses = read_responses(args.responses)
    if not responses:
        raise SystemExit(f"no responses in {args.responses}; nothing to build frames from")

    candidates, unparseable = parse_responses(responses, target_cell_by_prompt=target_by_prompt)
    outcome = screen_candidates(
        candidates,
        lattice=lattice,
        elicitation=config.elicitation,
        settings=config.filter,
        dedup_settings=config.dedup,
    )
    frames, guard_rejections = trial_render(
        outcome.kept,
        config=config,
        prompt_set=prompt_set,
        lattice=lattice,
        cells=cells,
        generator=generator,
    )

    rejections = [
        *outcome.rejected,
        *guard_rejections,
        *(
            Rejection(text="", target_cell_id="", response_id=response_id, reason=reason)
            for response_id, reason in unparseable
        ),
    ]

    # Every frame is rendered into EVERY target cell. See margins.redteam.frames: rendering a
    # frame only into the cell it was elicited for would confound the frame with the cell.
    audit_prompts = render_audit_prompts(
        frames,
        config=config,
        prompt_set=prompt_set,
        lattice=lattice,
        cells=cells,
        generator=generator,
    )

    frames_path = write_frames(frames, args.out_dir)
    rejected_path = write_rejected(rejections, args.out_dir)
    report_path = write_redteam_report(
        render_redteam_report(
            config,
            frames=frames,
            rejections=rejections,
            target_cells=[cell.cell_id for cell in cells],
            n_candidates=len(candidates),
            n_responses=len(responses),
            generator_model_id=generator.model_id,
            generator_revision=generator.revision,
            n_audit_prompts=len(audit_prompts),
        ),
        args.out_dir,
    )

    sample_path: Path | None = None
    units: list = []
    if frames:
        # The labeler judges the frame rendered for its OWN target cell: "is this on-target" is
        # not answerable from a template with a hole in it.
        example_by_frame: dict[str, str] = {}
        for frame in frames:
            match = next(
                (
                    prompt
                    for prompt in audit_prompts
                    if prompt.item_id.endswith(f"/{frame.frame_id}")
                    and prompt.cell_id == frame.target_cell_id
                ),
                None,
            )
            if match is not None:
                example_by_frame[frame.frame_id] = match.rendered_text
        sampled = [frame for frame in frames if frame.frame_id in example_by_frame]
        if sampled:
            units = sample_for_on_target(sampled, example_by_frame, config.on_target)
            sample_path = write_on_target_sample(units, args.out_dir / ON_TARGET_SAMPLE_FILENAME)

    counts = dict.fromkeys(sorted({r.reason for r in rejections}), 0)
    for rejection in rejections:
        counts[rejection.reason] += 1

    print(f"redteam_version:  {config.redteam_version}")
    print(f"generator:        {generator.model_id} @ {generator.revision}")
    print("\nsummary")
    print(f"  {'generations read':<28} {len(responses)}")
    print(f"  {'candidates parsed':<28} {len(candidates)}")
    print(f"  {'rejected':<28} {len(rejections)}")
    for reason, count in sorted(counts.items()):
        print(f"    {reason:<26} {count}")
    print(f"  {'frames kept':<28} {len(frames)}")
    print(f"  {'target cells':<28} {len(cells)}")
    print(f"  {'audit prompts':<28} {len(audit_prompts)}")
    print(f"  {'on-target sample':<28} {len(units)}")
    print(f"\nwrote {frames_path}")
    print(f"wrote {rejected_path}")
    print(f"wrote {report_path}")
    if sample_path is not None:
        print(f"wrote {sample_path}")
        print(
            "\nthe on-target sample is UNLABELED. Fill in `on_target` (0/1) per frame, then run "
            "scripts/redteam_on_target_report.py"
        )

    if args.no_manifest:
        print("\nno manifest written (--no-manifest)")
        return 0

    overrides = [
        f"run_name={args.run_name}",
        f"lattice_name={args.lattice}",
        f"redteam_name={args.redteam}",
        f"model_name={config.generator.model_name}",
        f"decoding_name={config.generator.decoding_name}",
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

    record = RedTeamRecord(
        redteam_name=args.redteam,
        redteam_version=config.redteam_version,
        benchmark=REDTEAM_BENCHMARK,
        generator_model_name=config.generator.model_name,
        generator_model_id=generator.model_id,
        generator_revision=generator.revision,
        generator_decoding_key=decoding.key,
        generator_decoding_hash=decoding.decoding_hash,
        audit_decoding_key=config.decoding_key,
        base_prompt_set=config.base_prompt_set,
        base_template_version=base.template_version,
        n_target_cells=len(cells),
        n_elicitation_prompts=len(elicitation),
        n_responses=len(responses),
        n_candidates=len(candidates),
        n_rejected=len(rejections),
        rejections_by_reason=counts,
        n_frames=len(frames),
        n_audit_prompts=len(audit_prompts),
        n_on_target_sample=len(units),
    )
    manifest = write_run_manifest(cfg, inputs=inputs, timestamp=args.timestamp, redteam=record)

    print("\nrun manifest")
    print(f"  {'run_id':<28} {manifest.run_id}")
    for key in sorted(manifest.input_checksums):
        print(f"  {key:<28} {manifest.input_checksums[key]}")
    print(f"\nwrote {manifest_path(cfg, manifest.run_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
