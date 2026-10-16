"""Run the P11 experiment on a lab GPU, one declared step at a time (the runbook's entry point).

    uv run python scripts/run_p11.py prepare            # fetch pinned inputs, build prompts
    uv run python scripts/run_p11.py verify-gpu --model llama31_8b_instruct   # FAIL FAST
    uv run python scripts/run_p11.py budget             # GPU-hours from the MEASURED speed
    uv run python scripts/run_p11.py smoke --model llama31_8b_instruct   # MUST PASS before E1
    uv run python scripts/run_p11.py e1 --model llama31_8b_instruct      # resumable
    uv run python scripts/run_p11.py crows --model llama31_8b_instruct   # resumable
    uv run python scripts/run_p11.py e4 --model llama31_8b_instruct      # resumable
    uv run python scripts/run_p11.py redteam-elicit     # red LM -> frames
    uv run python scripts/run_p11.py redteam-prompts --max-frames N      # descriptive subset
    uv run python scripts/run_p11.py redteam-audit --model llama31_8b_instruct
    uv run python scripts/run_p11.py process            # scoring -> ... -> figures (CPU ok)

Every step accepts ``--dry-run``, which prints the exact ``uv run python scripts/...`` lines it
would execute and runs nothing. Those lines are built in :mod:`margins.orchestrate.plan`, the one
place a P11 argv is written, and ``tests/test_p11_orchestration.py`` parses every one of them
against the real script's parser.

**This script measures nothing itself.** It calls the existing P2-P10 scripts' ``main(argv)`` in
order, echoing each call so the terminal log is a complete, replayable record. The scripts that
load a model onto the GPU run as a child process instead (``SUBPROCESS_SCRIPTS``), so each one's
GPU memory is released when it exits. What it adds is
refusal: it will not start a confirmatory generation step from a modified worktree, from a commit
older than ``prereg-v1``, or on a model that did not load at NF4 -- the three ways a lab run could
spend days producing something that is not the pre-registered measurement.

**Resuming.** Every generation step passes ``--resume`` to ``run_inference.py``, and every arm
writes to its own directory, so re-running the identical command after an interruption continues
exactly where it stopped. Nothing is regenerated and nothing is duplicated.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from margins.orchestrate import plan
from margins.orchestrate.budget import project_gpu_hours, redteam_max_frames
from margins.orchestrate.config import DEFAULT_P11, P11Config, load_p11_config
from margins.orchestrate.preflight import (
    PREREG_TAG,
    dirty_tree_reason,
    nf4_unavailable_reason,
    prereg_reason,
    quant_reason,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
GPU_CHECK_DIR = Path("data") / "p11"
SMOKE_STAGES = (
    "prompts",
    "inference",
    "e4",
    "scoring",
    "metrics",
    "stats",
    "tables",
    "figure",
    "verdict",
)
PROCESS_STAGES = ("scoring", "metrics", "stats", "tables", "figures")


class StepFailedError(RuntimeError):
    """A called script exited non-zero. The message names the command."""


# ---------------------------------------------------------------------------------------------
# running the existing scripts
# ---------------------------------------------------------------------------------------------


def _load_script(name: str) -> ModuleType:
    """Load ``scripts/<name>.py`` by path. ``scripts/`` is not a package."""
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_p11_{name}", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: Scripts that load a model onto the GPU. Each runs in its OWN process, so the OS reclaims every
#: byte of GPU memory when it exits. Found by the first lab smoke run (12 Oct 2026, RTX A4000):
#: in-process, each NF4 model load left its ~4.8 GB allocated after `main()` returned, and the
#: fourth load in one smoke run (open_gen, crows, E4 t1.0, E4 greedy) OOMed at 14.4 GB = 3 x 4.8.
#: Not reproducible on CPU, where the model is freed on return, so the release is enforced by
#: process boundary rather than by trusting gc/empty_cache to find whatever holds it on CUDA.
SUBPROCESS_SCRIPTS = frozenset({"run_inference", "run_scoring"})


def execute(command: plan.Command, *, dry_run: bool = False) -> None:
    """Echo, then run, one command. Raises :class:`StepFailedError` on a non-zero exit."""
    print(f"\n$ {command.render()}", flush=True)
    if dry_run:
        return
    if command.script in SUBPROCESS_SCRIPTS:
        # The same interpreter (the uv venv) and the same environment, so HF_TOKEN/HF_HOME carry
        # over; the working directory is inherited, so the relative paths in argv resolve alike.
        # Unbuffered, because a child writing into a pipe (`| tee run.log`, as the lab ran it)
        # otherwise holds its progress back in a block buffer and the terminal looks frozen.
        script = SCRIPTS_DIR / f"{command.script}.py"
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        argv = [sys.executable, str(script), *command.argv]
        code = subprocess.run(argv, check=False, env=env).returncode
    else:
        try:
            code = _load_script(command.script).main(list(command.argv))
        except SystemExit as exit_:  # argparse errors and explicit SystemExit(...) calls
            code = exit_.code if isinstance(exit_.code, int) else 1
    if code:
        raise StepFailedError(f"exited {code}: {command.render()}")


def newest_run_file(out_dir: Path, model: str, *, dry_run: bool = False) -> Path | str:
    """The one response file of an arm, for scoring. Refuses zero or several."""
    directory = out_dir / model
    if dry_run:
        return f"<the run file in {directory.as_posix()}/>"
    files = sorted(directory.glob("*.jsonl")) if directory.is_dir() else []
    if not files:
        raise StepFailedError(f"no response file in {directory}: has that generation step run?")
    if len(files) > 1:
        names = ", ".join(path.name for path in files)
        raise StepFailedError(
            f"{directory} holds {len(files)} run files ({names}). Scoring takes exactly one. "
            "Keep the complete run and move the others OUT of data/ (do not delete them) -- see "
            "docs/runbook_p11.md section F."
        )
    return files[0]


def _count_lines(path: Path | str) -> int:
    if not isinstance(path, Path) or not path.is_file():
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


# ---------------------------------------------------------------------------------------------
# preflight for confirmatory generation
# ---------------------------------------------------------------------------------------------


def _git(*args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(  # fixed argv, no shell
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return None


def confirmatory_preflight(*, allow_dirty: bool) -> None:
    """Refuse a confirmatory generation step from a modified tree or a pre-registration commit."""
    from margins.config import git_state

    _, dirty = git_state(REPO_ROOT, ignore_untracked_under="results/run_manifests")
    reason = dirty_tree_reason(dirty)
    if reason is not None and not allow_dirty:
        raise StepFailedError(f"STOP: {reason}")
    if reason is not None:
        print(f"WARNING (--allow-dirty): {reason}")

    verify = _git("rev-parse", "--verify", "--quiet", f"refs/tags/{PREREG_TAG}")
    if verify is None or verify.returncode != 0:
        is_descendant: bool | None = None
    else:
        ancestor = _git("merge-base", "--is-ancestor", PREREG_TAG, "HEAD")
        is_descendant = None if ancestor is None else ancestor.returncode == 0
    reason = prereg_reason(is_descendant)
    if reason is not None:
        raise StepFailedError(f"STOP: {reason}")


# ---------------------------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------------------------


def cmd_prepare(cfg: P11Config, args: argparse.Namespace) -> int:
    for command in plan.prepare_commands(cfg):
        execute(command, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"\nprepared. E1 prompts: {plan.Layout().e1_prompts(cfg)}")
    return 0


def cmd_verify_gpu(cfg: P11Config, args: argparse.Namespace) -> int:
    """Fail fast: can THIS card load THIS model at NF4 and generate? Measure its speed."""
    from margins.inference.models import load_model_config

    spec = load_model_config(args.model)
    print(f"model      {spec.model_id}@{spec.revision}  ({spec.role})")
    if args.dry_run:
        print("(dry run: would check NF4, the HF token and licence, disk, load, and time a batch)")
        return 0

    reason = nf4_unavailable_reason()
    if reason is not None:
        print(f"\nSTOP: {reason}", file=sys.stderr)
        return 2

    from huggingface_hub import get_token

    if get_token() is None:
        print(
            "\nSTOP: no Hugging Face token. `export HF_TOKEN=<your token>` in this shell (the "
            "account must have accepted the model's licence on its Hugging Face page).",
            file=sys.stderr,
        )
        return 2

    pin = _load_script("pin_models").main(["--models", args.model])
    if pin == 2:
        print("\nSTOP: the pinned revision is unreachable (see above). Do not substitute.")
        return 2

    cache = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    cache.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(cache).free / 1e9
    print(f"disk       {free_gb:.0f} GB free at {cache}")
    if free_gb < 40:
        print("  WARNING: under 40 GB free. Each 7-8B checkpoint downloads ~15-16 GB. Set HF_HOME.")

    import torch

    from margins.inference import HFBackend, load_decoding_config
    from margins.prompts import read_prompts
    from margins.repro import environment_fingerprint

    prompts_path = plan.Layout().e1_prompts(cfg)
    if not prompts_path.is_file():
        print(f"\nSTOP: {prompts_path} is missing. Run `run_p11.py prepare` first.")
        return 2
    batch = read_prompts(prompts_path)[: cfg.e1.batch_size]
    decoding = load_decoding_config(cfg.e1.decoding)
    backend = HFBackend(spec, decoding, seed=cfg.e1.seeds[0], quant="nf4")
    try:
        report = backend.load_report
    except torch.cuda.OutOfMemoryError:
        print("\nSTOP: out of GPU memory while LOADING the model. See runbook section F.")
        return 2
    print(f"loaded     device={report.device} dtype={report.dtype} quant={report.quant_actual}")
    refusal = quant_reason(
        requested="nf4",
        actual=report.quant_actual,
        fallback_reason=report.quant_fallback_reason,
        role=spec.role,
        allow_fallback=False,
    )
    if refusal is not None:
        print(f"\nSTOP: {refusal}", file=sys.stderr)
        return 2

    torch.cuda.reset_peak_memory_stats()
    try:
        backend.generate(batch, seed=cfg.e1.seeds[0], batch_size=len(batch))  # warm-up
        start = time.perf_counter()
        backend.generate(batch, seed=cfg.e1.seeds[1], batch_size=len(batch))
        elapsed = time.perf_counter() - start
    except torch.cuda.OutOfMemoryError:
        print(
            f"\nSTOP: out of GPU memory generating a batch of {len(batch)}. Lower `batch_size` in "
            "configs/experiment/p11_v1.yaml is a CONFIG CHANGE -- report it; see section F."
        )
        return 2
    sec_per_gen = elapsed / len(batch)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    fingerprint = environment_fingerprint()
    n_e1 = len(read_prompts(prompts_path)) * len(cfg.e1.seeds)
    projected = project_gpu_hours(sec_per_gen, n_e1)

    print(
        f"gpu        {fingerprint.cuda_device_in_use}  driver {fingerprint.nvidia_driver_version}"
    )
    print(f"cuda       runtime {fingerprint.torch_cuda_version}  cudnn {fingerprint.cudnn_version}")
    print(f"torch      {fingerprint.torch_version}  blas {fingerprint.torch_blas_info}")
    print(f"speed      {sec_per_gen:.2f} s/generation at batch {len(batch)} (steady state)")
    print(f"vram       peak {peak_gb:.1f} GB allocated")
    print(f"E1 budget  {n_e1} generations -> ~{projected:.1f} GPU-h for this model on this card")

    record = {
        "model": args.model,
        "model_id": spec.model_id,
        "revision": spec.revision,
        "batch_size": len(batch),
        "sec_per_generation": sec_per_gen,
        "peak_vram_gb": peak_gb,
        "e1_generations": n_e1,
        "e1_projected_gpu_hours": projected,
        "environment": fingerprint.model_dump(mode="json"),
    }
    GPU_CHECK_DIR.mkdir(parents=True, exist_ok=True)
    out = GPU_CHECK_DIR / f"gpu_check_{args.model}.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nVERIFY OK -- wrote {out}")
    return 0


def _speeds(cfg: P11Config) -> dict[str, tuple[float, str]]:
    """Per-model sec/generation: measured by verify-gpu where available, else the T4 pilot."""
    speeds: dict[str, tuple[float, str]] = {}
    for model in cfg.e1.models:
        path = GPU_CHECK_DIR / f"gpu_check_{model}.json"
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            speeds[model] = (float(data["sec_per_generation"]), "MEASURED on this card")
        else:
            speeds[model] = (
                cfg.pilot_sec_per_generation[model],
                "ESTIMATE (T4 pilot, batch 1) -- run verify-gpu to measure",
            )
    return speeds


def cmd_budget(cfg: P11Config, args: argparse.Namespace) -> int:
    from margins.lattice import DeclaredLattice, load_lattice_config
    from margins.prompts import read_prompts
    from margins.redteam import (
        elicitation_seeds,
        load_base_prompt_set,
        load_redteam_config,
        target_cells,
    )

    speeds = _speeds(cfg)
    prompts_path = plan.Layout().e1_prompts(cfg)
    if not prompts_path.is_file():
        print(f"STOP: {prompts_path} is missing. Run `run_p11.py prepare` first.")
        return 2
    e1_prompts = read_prompts(prompts_path)
    n_e1 = len(e1_prompts) * len(cfg.e1.seeds)
    print(f"E1: {n_e1} generations per model ({len(cfg.e1.seeds)} seeds)\n")
    for model, (seconds, source) in speeds.items():
        hours = project_gpu_hours(seconds, n_e1)
        print(f"  {model:<26} {seconds:5.2f} s/gen  -> {hours:6.1f} GPU-h   [{source}]")

    lattice = DeclaredLattice(load_lattice_config(cfg.lattice))
    from margins.orchestrate.smoke import pair_cells

    e4_cells = {cell.cell_id for cell in pair_cells(lattice, cfg.e4.pair)}
    n_e4_prompts = sum(1 for prompt in e1_prompts if prompt.cell_id in e4_cells)
    n_e4 = n_e4_prompts * sum(len(arm.seeds) for arm in cfg.e4.arms)
    arms = " + ".join(f"{arm.decoding} x {len(arm.seeds)}" for arm in cfg.e4.arms)
    print(
        f"\nE4: {n_e4_prompts} prompts ({' x '.join(cfg.e4.pair)}) x ({arms}) = {n_e4} "
        f"generations per model;\n    the temperature-0.7 arm is E1's "
        f"{cfg.e4.reuses_e1_decoding}, reused (0 extra)"
    )
    for model, (seconds, source) in speeds.items():
        hours = project_gpu_hours(seconds, n_e4)
        print(f"  {model:<26} {seconds:5.2f} s/gen  -> {hours:6.1f} GPU-h   [{source}]")

    redteam = load_redteam_config(cfg.redteam.set)
    cells = target_cells(redteam, lattice)
    n_elicit = len(redteam.elicitation.templates) * len(cells) * len(elicitation_seeds(redteam))
    carriers = len(load_base_prompt_set(redteam).carriers)
    generator = redteam.generator.model_name
    red_speed = speeds.get(generator, (cfg.pilot_sec_per_generation.get(generator, 0.0), ""))[0]
    print(f"\nred-team (DESCRIPTIVE, budget {cfg.redteam.budget_gpu_hours:.0f} GPU-h)")
    try:
        budget = redteam_max_frames(
            budget_hours=cfg.redteam.budget_gpu_hours,
            elicitation_generations=n_elicit,
            red_lm_sec_per_generation=red_speed,
            target_sec_per_generation={m: s for m, (s, _) in speeds.items()},
            target_cells=len(cells),
            carriers=carriers,
            audit_seeds=len(cfg.redteam.audit_seeds),
        )
    except ValueError as error:
        print(f"  STOP: {error}")
        return 2
    print(f"  elicitation   {n_elicit} red-LM generations  -> {budget.elicitation_hours:.1f} h")
    print(
        f"  audit         {len(cells)} cells x {carriers} carriers x "
        f"{len(cfg.redteam.audit_seeds)} seed x {len(speeds)} models per frame "
        f"-> {budget.per_frame_hours:.2f} h/frame"
    )
    print(f"  --max-frames  {budget.max_frames}   (total {budget.total_hours:.1f} h)")
    print(
        f"\nnext: uv run python scripts/run_p11.py redteam-prompts --max-frames {budget.max_frames}"
    )
    return 0


def cmd_smoke(cfg: P11Config, args: argparse.Namespace) -> int:
    from margins.inference.models import load_model_config
    from margins.lattice import DeclaredLattice, load_lattice_config
    from margins.orchestrate.figure import build_smoke_figure
    from margins.orchestrate.smoke import check_smoke_pass, pair_cells, select_smoke_prompts
    from margins.prompts import read_prompts, render_prompts_jsonl
    from margins.stats.config import load_stats_config
    from margins.viz import emergence
    from margins.viz.style import VECTOR_FORMATS

    spec = load_model_config(args.model)
    layout = plan.SmokeLayout(args.work_dir)
    lattice = DeclaredLattice(load_lattice_config(cfg.lattice))
    first = SMOKE_STAGES.index(args.from_stage)
    stages = SMOKE_STAGES[first:]
    record: dict[str, Any] = {"model": args.model, "model_id": spec.model_id, "stages": {}}
    figure_paths = [
        layout.figures / f"smoke_{emergence.NAME}.{fmt}" for fmt in sorted(VECTOR_FORMATS)
    ]
    failed: str | None = None

    print(f"SMOKE RUN  {spec.model_id} ({spec.role})  work dir {layout.work}")
    print(f"  pair {cfg.smoke.pair}  frames {list(cfg.smoke.frames)}  carrier {cfg.smoke.carrier}")
    print(f"  seeds {list(cfg.smoke.seeds)}  crows items {cfg.smoke.crows_limit}")

    def stage(name: str) -> bool:
        return name in stages and failed is None

    def timed(name: str, body: Any) -> None:
        nonlocal failed
        start = time.perf_counter()
        try:
            extra = body() or {}
        except Exception as error:  # every failure becomes a recorded FAIL
            failed = name
            record["stages"][name] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            print(f"\nSTAGE {name} FAILED: {type(error).__name__}: {error}", file=sys.stderr)
            return
        record["stages"][name] = {"ok": True, "seconds": round(time.perf_counter() - start, 1)}
        record["stages"][name].update(extra)

    if stage("prompts"):

        def prompts_stage() -> dict[str, Any]:
            execute(plan.smoke_prompt_build(cfg, layout), dry_run=args.dry_run)
            if args.dry_run:
                return {}
            shipped = plan.Layout().e1_prompts(cfg)
            if shipped.is_file() and shipped.read_bytes() != layout.sampled_prompts.read_bytes():
                raise StepFailedError(
                    f"{shipped} differs from a fresh render of the same config -- the E1 prompt "
                    "file on disk is stale or edited. Re-run `run_p11.py prepare`."
                )
            chosen = select_smoke_prompts(
                read_prompts(layout.sampled_prompts),
                cells=pair_cells(lattice, cfg.smoke.pair),
                frames=cfg.smoke.frames,
                carrier=cfg.smoke.carrier,
                decoding_key=cfg.e1.decoding,
            )
            layout.open_gen_prompts.write_text(render_prompts_jsonl(chosen), encoding="utf-8")
            if not cfg.crows.prompts.is_file():
                raise StepFailedError(
                    f"{cfg.crows.prompts} is missing. Run `run_p11.py prepare` first."
                )
            crows = read_prompts(cfg.crows.prompts)[: cfg.smoke.crows_limit]
            layout.crows_prompts.write_text(render_prompts_jsonl(crows), encoding="utf-8")
            print(f"\nsmoke prompts: {len(chosen)} open_gen, {len(crows)} CrowS")
            return {"open_gen_prompts": len(chosen), "crows_prompts": len(crows)}

        timed("prompts", prompts_stage)

    if stage("inference"):

        def inference_stage() -> dict[str, Any]:
            timings: dict[str, Any] = {}
            for arm, command in zip(
                ("open_gen", "crows"), plan.smoke_inference(cfg, args.model, layout), strict=True
            ):
                directory = layout.responses(arm) / args.model
                before = sum(_count_lines(p) for p in directory.glob("*.jsonl"))
                start = time.perf_counter()
                execute(command, dry_run=args.dry_run)
                elapsed = time.perf_counter() - start
                after = sum(_count_lines(p) for p in directory.glob("*.jsonl"))
                timings[arm] = {"generated": after - before, "seconds": round(elapsed, 1)}
            generated = timings["open_gen"]["generated"]
            if generated:
                # An UPPER bound: it includes loading the model. verify-gpu measures steady state.
                timings["open_gen"]["sec_per_generation_upper_bound"] = round(
                    timings["open_gen"]["seconds"] / generated, 2
                )
            return timings

        timed("inference", inference_stage)

    if stage("e4"):

        def e4_stage() -> dict[str, Any]:
            # The E4 arms through the lab's own prompt builder and decoding profiles, capped at
            # `e4_limit` prompts. They are not scored: E4's analysis is not the smoke run's job,
            # proving that each E4 profile loads and generates under its own settings is.
            for command in plan.smoke_e4_commands(cfg, args.model, layout):
                execute(command, dry_run=args.dry_run)
            return {"limit": cfg.smoke.e4_limit, "arms": [arm.name for arm in cfg.e4.arms]}

        timed("e4", e4_stage)

    if stage("scoring"):

        def scoring_stage() -> None:
            responses: dict[str, Path | str] = {
                arm: newest_run_file(layout.responses(arm), args.model, dry_run=args.dry_run)
                for arm in ("open_gen", "crows")
            }
            for command in plan.smoke_stage_commands(cfg, "scoring", layout, responses=responses):
                execute(command, dry_run=args.dry_run)

        timed("scoring", scoring_stage)

    for name in ("metrics", "stats", "tables"):
        if stage(name):

            def script_stage(name: str = name) -> None:
                for command in plan.smoke_stage_commands(cfg, name, layout):
                    execute(command, dry_run=args.dry_run)

            timed(name, script_stage)

    if stage("figure"):

        def figure_stage() -> dict[str, Any]:
            if args.dry_run:
                print("\n(figure: build the emergence figure from the smoke tables, bannered)")
                return {}
            written = build_smoke_figure(
                table_dir=layout.tables,
                paper_numbers_path=layout.paper_numbers,
                model_id=spec.model_id,
                benchmark=plan.OPEN_GEN_BENCHMARK,
                out_dir=layout.figures,
            )
            return {"figures": {fmt: path.as_posix() for fmt, path in sorted(written.items())}}

        timed("figure", figure_stage)

    if args.dry_run:
        return 0

    stats_config = load_stats_config()
    from margins.inference import load_decoding_config
    from margins.orchestrate.smoke import E4Expectation

    e4_ran = "e4" in stages or any(
        (layout.responses(f"e4_{arm.name}") / args.model).is_dir() for arm in cfg.e4.arms
    )
    e4_expected = (
        [
            E4Expectation(
                name=arm.name,
                directory=layout.responses(f"e4_{arm.name}") / args.model,
                n_responses=cfg.smoke.e4_limit * len(arm.seeds),
                decoding_key=arm.decoding,
                temperature=load_decoding_config(arm.decoding).temperature,
                seeds=arm.seeds,
            )
            for arm in cfg.e4.arms
        ]
        if e4_ran
        else []
    )
    verdict = check_smoke_pass(
        open_gen_metrics_dir=layout.metrics("open_gen"),
        stats_dir=layout.stats,
        figure_paths=figure_paths,
        model_id=spec.model_id,
        lattice=lattice,
        open_gen_benchmark=plan.OPEN_GEN_BENCHMARK,
        crows_benchmark=cfg.crows.benchmark,
        correction=stats_config.fdr.method,
        e4=e4_expected,
    )
    record["failed_stage"] = failed
    record["passed"] = verdict.passed and failed is None
    record["checks"] = [
        {"name": check.name, "passed": check.passed, "detail": check.detail}
        for check in verdict.checks
    ]
    layout.work.mkdir(parents=True, exist_ok=True)
    layout.result.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n" + "=" * 96)
    if failed is not None:
        print(f"STAGE FAILED: {failed} -- {record['stages'][failed]['error']}")
    print(verdict.render())
    print(f"record: {layout.result}")
    if record["passed"]:
        print("\nThe correction path, the claim gate and the figure all ran on real P7 output.")
        print("You may proceed to the full run (runbook section C).")
        return 0
    print("\nDO NOT PROCEED to the full run. See docs/runbook_p11.md section F.")
    return 1


def cmd_e1(cfg: P11Config, args: argparse.Namespace) -> int:
    if not args.dry_run:
        confirmatory_preflight(allow_dirty=args.allow_dirty)
    execute(plan.e1_inference(cfg, args.model), dry_run=args.dry_run)
    return 0


def cmd_e4(cfg: P11Config, args: argparse.Namespace) -> int:
    """E4's generated arms for one model. The temperature-0.7 arm is E1's, reused."""
    if not args.dry_run:
        confirmatory_preflight(allow_dirty=args.allow_dirty)
        missing = [
            plan.Layout.e4_prompts(arm)
            for arm in cfg.e4.arms
            if not plan.Layout.e4_prompts(arm).is_file()
        ]
        if missing:
            raise StepFailedError(
                f"STOP: {', '.join(str(m) for m in missing)} missing. Run `run_p11.py prepare`."
            )
    print(
        f"E4 on {' x '.join(cfg.e4.pair)}: generating "
        + ", ".join(f"{arm.decoding} x {len(arm.seeds)} seed(s)" for arm in cfg.e4.arms)
        + f"; temperature 0.7 is E1's {cfg.e4.reuses_e1_decoding}, reused, not regenerated."
    )
    for command in plan.e4_inference(cfg, args.model):
        execute(command, dry_run=args.dry_run)
    return 0


def cmd_crows(cfg: P11Config, args: argparse.Namespace) -> int:
    if not args.dry_run:
        confirmatory_preflight(allow_dirty=args.allow_dirty)
    execute(plan.crows_inference(cfg, args.model), dry_run=args.dry_run)
    return 0


def cmd_redteam_elicit(cfg: P11Config, args: argparse.Namespace) -> int:
    from margins.redteam import elicitation_seeds, load_redteam_config

    if not args.dry_run:
        confirmatory_preflight(allow_dirty=args.allow_dirty)
    redteam = load_redteam_config(cfg.redteam.set)
    layout = plan.Layout()
    commands = plan.redteam_elicitation_commands(
        cfg,
        generator=redteam.generator.model_name,
        generator_decoding=redteam.generator.decoding_name,
        elicitation_seeds=elicitation_seeds(redteam),
    )
    for command in commands:
        execute(command, dry_run=args.dry_run)
    responses = newest_run_file(
        layout.responses("redteam_elicitation"), redteam.generator.model_name, dry_run=args.dry_run
    )
    execute(plan.redteam_frames(cfg, responses), dry_run=args.dry_run)
    return 0


def cmd_redteam_prompts(cfg: P11Config, args: argparse.Namespace) -> int:
    execute(plan.redteam_audit_prompts(cfg, args.max_frames), dry_run=args.dry_run)
    return 0


def cmd_redteam_audit(cfg: P11Config, args: argparse.Namespace) -> int:
    if not args.dry_run:
        confirmatory_preflight(allow_dirty=args.allow_dirty)
    execute(plan.redteam_audit_inference(cfg, args.model), dry_run=args.dry_run)
    return 0


def cmd_process(cfg: P11Config, args: argparse.Namespace) -> int:
    layout = plan.Layout(data=args.data_root, results=args.results_root, figures=args.figures_dir)
    stages = PROCESS_STAGES[PROCESS_STAGES.index(args.from_stage) :]
    arms = ["e1", "crows", *([] if args.no_redteam else ["redteam"])]
    responses: dict[tuple[str, str], Path | str] = {}
    if "scoring" in stages:
        for arm in arms:
            for model in cfg.e1.models:
                responses[(arm, model)] = newest_run_file(
                    layout.responses(arm), model, dry_run=args.dry_run
                )
    for command in plan.process_commands(
        cfg,
        responses=responses,
        layout=layout,
        include_redteam=not args.no_redteam,
        stages=stages,
    ):
        execute(command, dry_run=args.dry_run)
    return 0


# ---------------------------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", default=DEFAULT_P11, help="P11 declaration in configs/experiment."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_text: str) -> argparse.ArgumentParser:
        child = sub.add_parser(name, help=help_text)
        child.add_argument(
            "--dry-run", action="store_true", help="Print the exact commands; run nothing."
        )
        return child

    add("prepare", "Fetch pinned inputs and render every prompt set (no GPU).")

    verify = add("verify-gpu", "Fail fast: load the model at NF4 on this card and time a batch.")
    verify.add_argument("--model", required=True)

    add("budget", "GPU-hours per arm from the measured speed; sizes the red-team --max-frames.")

    smoke = add("smoke", "The multi-rendering smoke run. MUST PASS before any full run.")
    smoke.add_argument("--model", required=True)
    smoke.add_argument("--work-dir", type=Path, default=Path("data/smoke"))
    smoke.add_argument("--from-stage", choices=SMOKE_STAGES, default=SMOKE_STAGES[0])

    for name, text in (
        ("e1", "E1: the static open_gen arm for one model (resumable)."),
        ("crows", "E1: the CrowS-Pairs log-probability arm for one model (resumable)."),
        ("e4", "E4: the temperature-1.0 and greedy arms on class x race (resumable)."),
        ("redteam-audit", "Red-team audit prompts through one target model (resumable)."),
    ):
        child = add(name, text)
        child.add_argument("--model", required=True)
        child.add_argument(
            "--allow-dirty", action="store_true", help="Debugging only. Never for a real run."
        )

    elicit = add("redteam-elicit", "Red LM elicitation, then screen its frames (resumable).")
    elicit.add_argument("--allow-dirty", action="store_true", help="Debugging only.")

    prompts = add("redteam-prompts", "Render the DESCRIPTIVE red-team subset of frames.")
    prompts.add_argument("--max-frames", type=int, required=True)

    process = add("process", "Scoring -> metrics -> stats -> tables -> figures. CPU is fine.")
    process.add_argument("--no-redteam", action="store_true", help="Process E1 + CrowS only.")
    process.add_argument("--from-stage", choices=PROCESS_STAGES, default=PROCESS_STAGES[0])
    process.add_argument("--data-root", type=Path, default=Path("data"))
    process.add_argument("--results-root", type=Path, default=Path("results"))
    process.add_argument("--figures-dir", type=Path, default=Path("figures"))
    return parser


HANDLERS = {
    "prepare": cmd_prepare,
    "verify-gpu": cmd_verify_gpu,
    "budget": cmd_budget,
    "smoke": cmd_smoke,
    "e1": cmd_e1,
    "crows": cmd_crows,
    "e4": cmd_e4,
    "redteam-elicit": cmd_redteam_elicit,
    "redteam-prompts": cmd_redteam_prompts,
    "redteam-audit": cmd_redteam_audit,
    "process": cmd_process,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_p11_config(args.config)
    try:
        return HANDLERS[args.command](cfg, args)
    except StepFailedError as error:
        print(f"\n{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
