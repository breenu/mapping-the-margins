"""Run the P4a prompt set through a pinned checkpoint, resumably (P5).

The inference entrypoint. It consumes ``data/prompts/prompts_v1.jsonl`` -- the artifact P4a's
byte-identity gate checks -- rather than re-rendering prompts, so what a run audited is literally
the prompt set that was reviewed.

**Resumption is the point.** The confirmatory sweep runs in 12-hour free-tier Kaggle sessions
against a ~30 GPU-h weekly quota. Re-running the same command with ``--resume`` picks up the newest
run file for that model and continues it: finished responses are skipped, a truncated final record
is repaired, and nothing is duplicated or dropped. The manifest is written **before** generation
starts, using the same ``run_id`` as the response file, so an interrupted run still leaves a
complete record of what it was doing.

    # local smoke test against the dev model (unquantized on Windows -- recorded as such)
    uv run python scripts/run_inference.py --model llama32_1b_instruct --limit 20

    # resume a killed sweep
    uv run python scripts/run_inference.py --model llama31_8b_instruct --resume

Requires the ``inference`` extra. The 7-8B checkpoints never run on the Windows dev box
(PROJECT_CONTEXT.md section 7); use Kaggle.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from margins.config import (
    LATTICE_INPUT_KEY,
    config_hash,
    default_config_dir,
    load_experiment_config,
    make_run_id,
    manifest_path,
    write_run_manifest,
)
from margins.inference import (
    DEFAULT_RESPONSES_DIR,
    DEV_MODEL,
    HFBackend,
    RunHarness,
    default_decoding_config_dir,
    default_model_config_dir,
    inference_inputs,
    inference_record,
    load_decoding_config,
    load_model_config,
    response_path,
)
from margins.lattice import DEFAULT_LATTICE
from margins.lattice.config import default_lattice_config_dir
from margins.prompts import (
    DEFAULT_EXTRACTIVE_SET,
    DEFAULT_PROMPT_SET,
    EXTRACTIVE_SET_INPUT_KEY,
    PROMPTS_FILENAME,
    default_prompt_config_dir,
    load_extractive_config,
    load_prompt_config,
    load_template_files,
    prompt_set_inputs,
    read_prompts,
)
from margins.prompts.config import TEMPLATE_DIRNAME, file_sha256
from margins.schemas import Prompt, TaskType


def _latest_run_id(out_dir: Path, model_name: str) -> str | None:
    """The newest existing run file for this model, or None."""
    directory = out_dir / model_name
    if not directory.is_dir():
        return None
    files = sorted(directory.glob("*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)
    return files[0].stem if files else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEV_MODEL, help="Model config name.")
    parser.add_argument("--decoding", default=None, help="Decoding profile (default: from config).")
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("data/prompts") / PROMPTS_FILENAME,
        help="Prompt JSONL from scripts/build_prompts.py.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "One generation per (prompt, seed). Seeds must be distinct: under a sampled profile "
            "they are what produce distinct samples, and under greedy they produce identical text. "
            "Defaults to the config's single seed."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="First N prompts (smoke tests).")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--token-logprobs",
        action="store_true",
        help="Record per-token log-probabilities. Off by default: one float per generated token.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESPONSES_DIR)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--run-name", default="inference")
    parser.add_argument(
        "--resume", action="store_true", help="Continue the newest run file for this model."
    )
    parser.add_argument("--run-id", default=None, help="Continue this exact run id.")
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--lattice", default=DEFAULT_LATTICE)
    parser.add_argument("--prompt-set", default=DEFAULT_PROMPT_SET)
    parser.add_argument(
        "--extractive-set",
        default=DEFAULT_EXTRACTIVE_SET,
        help="Extractive prompt-set config name, used when the prompt file is pair_pll.",
    )
    args = parser.parse_args(argv)

    spec = load_model_config(args.model, config_dir=default_model_config_dir())

    overrides = [f"run_name={args.run_name}", f"model_name={args.model}"]
    if args.decoding is not None:
        overrides.append(f"decoding_name={args.decoding}")
    if args.results_dir is not None:
        overrides.append(f"results_dir={args.results_dir.as_posix()}")
    cfg = load_experiment_config(
        config_dir=args.config_dir or default_config_dir(), overrides=overrides
    )
    # The manifest's serialised config must state the exact checkpoint the run loaded, not a name
    # that resolves to whatever the config file says today.
    cfg = cfg.model_copy(update={"model_id": spec.model_id, "model_revision": spec.revision})

    decoding = load_decoding_config(cfg.decoding_name, config_dir=default_decoding_config_dir())
    prompts: list[Prompt] = read_prompts(args.prompts)
    if args.limit is not None:
        prompts = prompts[: args.limit]
    if not prompts:
        parser.error(f"no prompts to run from {args.prompts}")

    seeds = args.seeds if args.seeds else [cfg.seed]

    run_id = args.run_id
    if run_id is None and args.resume:
        run_id = _latest_run_id(args.out_dir, spec.name)
        if run_id is None:
            print(f"--resume: no existing run for {spec.name}; starting a new one")
    if run_id is None:
        run_id = make_run_id(cfg, config_hash(cfg), datetime.now(UTC))

    print(f"model      {spec.model_id}@{spec.revision}  ({spec.role})")
    print(f"decoding   {decoding.key}  hash={decoding.decoding_hash[:16]}...")
    print(f"prompts    {len(prompts)} from {args.prompts}")
    print(f"seeds      {seeds}  batch_size={args.batch_size}")
    print(f"run_id     {run_id}")

    backend = HFBackend(spec, decoding, seed=cfg.seed, quant=cfg.quant)
    report = backend.load_report  # loads the model; everything below reports what it actually did
    print(
        f"loaded     device={report.device} dtype={report.dtype} "
        f"quant={report.quant_actual} (requested {report.quant_requested})"
    )
    if report.quant_fallback_reason:
        print(f"  NOTE: {report.quant_fallback_reason}")

    # Manifest inputs: P4a's lattice + prompt set + frames, plus P5's checkpoint + decoding profile.
    #
    # Branches on the PARADIGM since P2. An extractive prompt set has no template files to digest,
    # so the open_gen `inputs` builder does not apply to it; recording the lattice and the
    # extractive set's own config is the equivalent record. Detected from the prompts themselves
    # rather than from a flag -- a run whose manifest described the wrong prompt set would be
    # wrong in exactly the way the manifest exists to prevent.
    prompt_dir = default_prompt_config_dir()
    lattice_path = default_lattice_config_dir() / f"{args.lattice}.yaml"
    if all(prompt.task_type is TaskType.PAIR_PLL for prompt in prompts):
        extractive = load_extractive_config(args.extractive_set, config_dir=prompt_dir)
        inputs = {
            LATTICE_INPUT_KEY: file_sha256(lattice_path),
            EXTRACTIVE_SET_INPUT_KEY: file_sha256(prompt_dir / f"{args.extractive_set}.yaml"),
        }
        if extractive.decoding_key != cfg.decoding_name:
            parser.error(
                f"prompt set {extractive.key!r} is stamped decoding_key "
                f"{extractive.decoding_key!r} but this run holds {cfg.decoding_name!r}. Pass "
                f"--decoding {extractive.decoding_key}."
            )
    else:
        prompt_config = load_prompt_config(args.prompt_set, config_dir=prompt_dir)
        files = load_template_files(prompt_config, prompt_dir / TEMPLATE_DIRNAME)
        inputs = prompt_set_inputs(
            lattice_config_path=lattice_path,
            prompt_config_path=prompt_dir / f"{args.prompt_set}.yaml",
            files=files,
        )
    inputs.update(
        inference_inputs(
            model_config_path=default_model_config_dir() / f"{args.model}.yaml",
            decoding_config_path=default_decoding_config_dir() / f"{cfg.decoding_name}.yaml",
            model_name=spec.name,
            chat_template_sha256=report.chat_template_sha256,
        )
    )

    # Written BEFORE generation: an interrupted run still leaves a complete record of what it was.
    manifest = write_run_manifest(
        cfg,
        inputs=inputs,
        run_id=run_id,
        inference=inference_record(spec, decoding, report, batch_size=args.batch_size, seeds=seeds),
    )
    print(f"manifest   {manifest_path(cfg, manifest.run_id)}")

    harness = RunHarness(
        backend,
        run_id=run_id,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        token_logprobs=args.token_logprobs,
    )
    print(f"responses  {response_path(args.out_dir, spec.name, run_id)}\n")

    def progress(done: int, total: int) -> None:
        print(f"  {done}/{total}", end="\r", flush=True)

    result = harness.run(prompts, seeds=seeds, progress=progress)

    print("\n\nrun complete")
    print(f"  {'planned':<16} {result.planned}")
    print(f"  {'resumed':<16} {result.resumed}")
    print(f"  {'generated':<16} {result.generated}")
    print(f"  {'file':<16} {result.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
