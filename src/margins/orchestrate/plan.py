"""Every command the P11 lab run executes, built in one place.

**This module is the only place a P11 argv is written.** ``scripts/run_p11.py`` executes these
commands; ``docs/runbook_p11.md`` prints them; ``tests/test_p11_orchestration.py`` parses every
one of them against the REAL script's argparse parser, so a flag that does not exist fails the
suite rather than failing at the lab on a booked GPU. The runbook's command list is checked the
same way.

**Layout: one directory per (arm, model).** ``run_inference.py --resume`` continues the NEWEST
response file under ``<out-dir>/<model>/``. Giving each arm its own out-dir is what makes "newest"
unambiguous: the smoke run, the E1 arm, the CrowS arm and the red-team arms can never resume each
other's files. Scoring and metrics are per model as well, because standardization is PER MODEL
(decided 23 Sep 2026): each model's harm scores are z-scored on that model's own SD, the unit the
power analysis was run in, and P9 then corrects every model's terms together as one family.

Pure: builders take every path and value as an argument and return data. Nothing here runs,
reads a clock, or touches the filesystem.
"""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from margins.orchestrate.config import E4DecodingArm, P11Config

__all__ = [
    "DEFAULT_LAYOUT",
    "OPEN_GEN_BENCHMARK",
    "REDTEAM_BENCHMARK",
    "Command",
    "Layout",
    "SmokeLayout",
    "crows_inference",
    "e1_inference",
    "e4_inference",
    "e4_process_commands",
    "e4_prompt_build",
    "prepare_commands",
    "process_commands",
    "r2_arm_dir",
    "r2_inference",
    "r2_process_commands",
    "r2_prompt_builds",
    "redteam_audit_inference",
    "redteam_audit_prompts",
    "redteam_elicitation_commands",
    "redteam_frames",
    "smoke_e4_commands",
    "smoke_inference",
    "smoke_prompt_build",
    "smoke_stage_commands",
]

OPEN_GEN_BENCHMARK: Final[str] = "open_gen_templates"
REDTEAM_BENCHMARK: Final[str] = "redteam"

ARMS: Final[tuple[str, ...]] = ("e1", "crows", "redteam")


@dataclass(frozen=True)
class Command:
    """One invocation of one script's ``main(argv)``."""

    script: str
    argv: tuple[str, ...]

    def render(self) -> str:
        """The exact shell line, copy-pastable into a terminal."""
        parts = ["uv", "run", "python", f"scripts/{self.script}.py", *self.argv]
        return " ".join(shlex.quote(part) for part in parts)


def _p(path: Path | str) -> str:
    return path.as_posix() if isinstance(path, Path) else path


@dataclass(frozen=True)
class Layout:
    """Where the confirmatory run reads and writes. Defaults are the repository's own."""

    data: Path = Path("data")
    results: Path = Path("results")
    figures: Path = Path("figures")

    def e1_prompts(self, cfg: P11Config) -> Path:
        return cfg.e1.prompts_dir / "prompts_v1.jsonl"

    @staticmethod
    def e4_prompts(arm: E4DecodingArm) -> Path:
        return arm.prompts_dir / "prompts_v1.jsonl"

    @property
    def redteam_dir(self) -> Path:
        return self.data / "redteam"

    @property
    def elicitation_prompts(self) -> Path:
        return self.redteam_dir / "elicitation_prompts.jsonl"

    @property
    def frames(self) -> Path:
        return self.redteam_dir / "frames_v1.jsonl"

    def audit_prompts(self, cfg: P11Config) -> Path:
        return self.data / "prompts" / f"{cfg.redteam.set}.jsonl"

    def responses(self, arm: str) -> Path:
        """``data/responses/<arm>``; ``run_inference.py`` adds ``/<model>/<run_id>.jsonl``."""
        return self.data / "responses" / arm

    def scoring(self, arm: str, model: str) -> Path:
        return self.data / "scoring" / arm / model

    def metrics(self, arm: str, model: str) -> Path:
        return self.data / "metrics" / arm / model

    @property
    def stats(self) -> Path:
        return self.data / "stats"

    @property
    def tables(self) -> Path:
        return self.results / "tables"

    @property
    def paper_numbers(self) -> Path:
        return self.results / "paper_numbers.json"


#: The repository's own layout, as the module-level default every builder falls back to.
DEFAULT_LAYOUT: Final[Layout] = Layout()


# ---------------------------------------------------------------------------------------------
# Section A: preparing a fresh lab checkout
# ---------------------------------------------------------------------------------------------


def prepare_commands(cfg: P11Config, layout: Layout = DEFAULT_LAYOUT) -> list[Command]:
    """Fetch the pinned inputs and render every prompt set the run needs. No GPU, no model.

    The CrowS ingestion runs with ``--no-manifest --no-paper-numbers``: its record already exists
    in git (``results/run_manifests/build-crows-items-…``) and re-registering it would rewrite a
    committed file, dirtying the tree that every confirmatory manifest then records. The item file
    it regenerates is byte-deterministic from a pinned, digest-checked CSV.
    """
    return [
        Command("fetch_lexicons", ("--fetch-pinned",)),
        Command("fetch_crows_pairs", ("--fetch-pinned",)),
        Command("build_crows_items", ("--no-manifest", "--no-paper-numbers")),
        Command("build_crows_prompts", ("--no-manifest",)),
        Command(
            "build_prompts",
            (
                "--prompt-set",
                cfg.e1.prompt_set,
                "--decoding",
                cfg.e1.decoding,
                "--out-dir",
                _p(cfg.e1.prompts_dir),
                "--run-name",
                "build-prompts-e1-sampled",
            ),
        ),
        *(e4_prompt_build(cfg, arm) for arm in cfg.e4.arms),
    ]


def e4_prompt_build(
    cfg: P11Config,
    arm: E4DecodingArm,
    *,
    out_dir: Path | None = None,
    results_dir: Path | None = None,
) -> Command:
    """One E4 arm's prompt file: the S1 pair's grid, stamped with that arm's decoding profile.

    Rendered by the same renderer as E1 and restricted with ``--pair``, so every E4 prompt's TEXT is
    its E1 counterpart's, byte for byte; only the decoding key (and so the id) differs.
    """
    argv = [
        "--prompt-set",
        cfg.e1.prompt_set,
        "--decoding",
        arm.decoding,
        "--pair",
        *cfg.e4.pair,
        "--out-dir",
        _p(out_dir if out_dir is not None else arm.prompts_dir),
        "--run-name",
        f"build-prompts-e4-{arm.name}",
    ]
    if results_dir is not None:
        argv += ["--results-dir", _p(results_dir)]
    return Command("build_prompts", tuple(argv))


# ---------------------------------------------------------------------------------------------
# Sections C and D: generation (the only GPU work)
# ---------------------------------------------------------------------------------------------


def _inference(
    *,
    model: str,
    prompts: Path | str,
    decoding: str,
    seeds: Sequence[int] | None,
    batch_size: int,
    out_dir: Path,
    run_name: str,
    extra: Sequence[str] = (),
    results_dir: Path | None = None,
) -> Command:
    argv = ["--model", model, "--prompts", _p(prompts), "--decoding", decoding]
    if seeds:
        argv += ["--seeds", *(str(seed) for seed in seeds)]
    argv += ["--batch-size", str(batch_size), "--out-dir", _p(out_dir), "--run-name", run_name]
    argv += list(extra)
    if results_dir is not None:
        argv += ["--results-dir", _p(results_dir)]
    # ALWAYS --resume: re-running the identical command after an interruption continues the run
    # rather than starting a second one. On a first invocation it prints that there is nothing to
    # resume and starts fresh.
    argv.append("--resume")
    return Command("run_inference", tuple(argv))


def e1_inference(cfg: P11Config, model: str, layout: Layout = DEFAULT_LAYOUT) -> Command:
    """The static E1 arm for one model: 2,970 prompts x the 4 declared seeds."""
    return _inference(
        model=model,
        prompts=layout.e1_prompts(cfg),
        decoding=cfg.e1.decoding,
        seeds=cfg.e1.seeds,
        batch_size=cfg.e1.batch_size,
        out_dir=layout.responses("e1"),
        run_name=f"e1-{model}",
    )


def e4_inference(cfg: P11Config, model: str, layout: Layout = DEFAULT_LAYOUT) -> list[Command]:
    """E4's GENERATED arms for one model: temperature 1.0 x 3 seeds, then greedy x 1 seed.

    The temperature-0.7 arm is not here on purpose: it is E1's own ``sampled_v1`` x 4 seeds on the
    same prompts, already generated, and regenerating it would spend GPU time to reproduce data.
    """
    return [
        _inference(
            model=model,
            prompts=layout.e4_prompts(arm),
            decoding=arm.decoding,
            seeds=arm.seeds,
            batch_size=cfg.e1.batch_size,
            out_dir=layout.responses(f"e4_{arm.name}"),
            run_name=f"e4-{arm.name}-{model}",
        )
        for arm in cfg.e4.arms
    ]


def r2_arm_dir(machine: str, arm: str) -> str:
    """``r2_<machine>_<arm>``: one response/scoring directory per (machine, arm), never shared.

    Kept apart from every original arm so a rerun can never resume -- or be mistaken for -- the
    run it is meant to reproduce.
    """
    return f"r2_{machine}_{arm}"


def r2_prompt_builds(cfg: P11Config) -> list[Command]:
    """R2's two prompt files. Deterministic and cheap, so the ``r2`` step builds them itself.

    The sampled file is the E4 grid stamped with E1's profile: the same text AND the same stamp as
    E1's rows, so the same ``prompt_id``s. The greedy file is the E4 greedy arm's own.
    """
    sampled = Command(
        "build_prompts",
        (
            "--prompt-set",
            cfg.e1.prompt_set,
            "--decoding",
            cfg.r2.sampled_decoding,
            "--pair",
            *cfg.r2.pair,
            "--out-dir",
            _p(cfg.r2.sampled_prompts_dir),
            "--run-name",
            "build-prompts-r2-sampled",
        ),
    )
    return [sampled, e4_prompt_build(cfg, cfg.r2_greedy)]


def r2_inference(
    cfg: P11Config, model: str, machine: str, layout: Layout = DEFAULT_LAYOUT
) -> list[Command]:
    """R2 for one model on one machine: the sampled rerun, then the greedy rerun. Resumable."""
    if machine not in cfg.r2.machines:
        raise ValueError(f"R2 machine {machine!r} is not declared; one of {cfg.r2.machines}")
    greedy = cfg.r2_greedy
    return [
        _inference(
            model=model,
            prompts=cfg.r2.sampled_prompts_dir / "prompts_v1.jsonl",
            decoding=cfg.r2.sampled_decoding,
            seeds=cfg.r2.sampled_seeds,
            batch_size=cfg.e1.batch_size,
            out_dir=layout.responses(r2_arm_dir(machine, "sampled")),
            run_name=f"r2-{machine}-sampled-{model}",
        ),
        _inference(
            model=model,
            prompts=layout.e4_prompts(greedy),
            decoding=greedy.decoding,
            seeds=greedy.seeds,
            batch_size=cfg.e1.batch_size,
            out_dir=layout.responses(r2_arm_dir(machine, "greedy")),
            run_name=f"r2-{machine}-greedy-{model}",
        ),
    ]


def r2_process_commands(
    cfg: P11Config,
    machine: str,
    *,
    responses: dict[str, Path | str],
    layout: Layout = DEFAULT_LAYOUT,
    n_resamples: int | None = None,
) -> list[Command]:
    """R2's CPU half for one machine: score each model's sampled rerun, then the R2 analysis.

    Only the SAMPLED rerun is scored -- the pass rule is about its estimates; the greedy rerun is
    compared as text (exact-match rate) and needs no scorer.
    """
    sampled = r2_arm_dir(machine, "sampled")
    commands = [
        _scoring(
            responses=responses[model],
            prompts=cfg.r2.sampled_prompts_dir / "prompts_v1.jsonl",
            benchmark=OPEN_GEN_BENCHMARK,
            out_dir=layout.scoring(sampled, model),
            run_name=f"scoring-{sampled}-{model}",
        )
        for model in cfg.e1.models
    ]
    analysis = ["--machine", machine, "--data-root", _p(layout.data)]
    analysis += ["--out-dir", _p(layout.tables), "--paper-numbers", _p(layout.paper_numbers)]
    if n_resamples is not None:
        analysis += ["--n-resamples", str(n_resamples)]
    commands.append(Command("run_r2_analysis", tuple(analysis)))
    return commands


def crows_inference(cfg: P11Config, model: str, layout: Layout = DEFAULT_LAYOUT) -> Command:
    """The CrowS-Pairs log-probability pass for one model. Deterministic: one seed suffices."""
    return _inference(
        model=model,
        prompts=cfg.crows.prompts,
        decoding=cfg.crows.decoding,
        seeds=None,
        batch_size=cfg.e1.batch_size,
        out_dir=layout.responses("crows"),
        run_name=f"crows-{model}",
        extra=("--extractive-set", cfg.crows.extractive_set),
    )


def redteam_elicitation_commands(
    cfg: P11Config,
    *,
    generator: str,
    generator_decoding: str,
    elicitation_seeds: Sequence[int],
    layout: Layout = DEFAULT_LAYOUT,
) -> list[Command]:
    """Stage 1 + the red LM's generation. Passes EVERY elicitation seed -- see the P11 fix."""
    return [
        Command("build_redteam_elicitation", ("--redteam", cfg.redteam.set)),
        _inference(
            model=generator,
            prompts=layout.elicitation_prompts,
            decoding=generator_decoding,
            seeds=elicitation_seeds,
            batch_size=cfg.e1.batch_size,
            out_dir=layout.responses("redteam_elicitation"),
            run_name="redteam-elicit",
        ),
    ]


def redteam_frames(cfg: P11Config, responses: Path | str) -> Command:
    """Stage 2: parse, filter, dedup and guard the red LM's frames."""
    return Command(
        "build_redteam_frames", ("--responses", _p(responses), "--redteam", cfg.redteam.set)
    )


def redteam_audit_prompts(cfg: P11Config, max_frames: int) -> Command:
    """Stage 3, capped: the DESCRIPTIVE subset of frames x every target cell."""
    return Command(
        "build_redteam_prompts", ("--redteam", cfg.redteam.set, "--max-frames", str(max_frames))
    )


def redteam_audit_inference(cfg: P11Config, model: str, layout: Layout = DEFAULT_LAYOUT) -> Command:
    """The target model on the red-team audit prompts, ONE seed (the P9 precision argument)."""
    return _inference(
        model=model,
        prompts=layout.audit_prompts(cfg),
        decoding=cfg.e1.decoding,
        seeds=cfg.redteam.audit_seeds,
        batch_size=cfg.e1.batch_size,
        out_dir=layout.responses("redteam"),
        run_name=f"redteam-{model}",
    )


# ---------------------------------------------------------------------------------------------
# Section E: processing (CPU-capable)
# ---------------------------------------------------------------------------------------------


def _scoring(
    *,
    responses: Path | str,
    prompts: Path | str,
    benchmark: str,
    out_dir: Path,
    run_name: str,
    scoring: str | None = None,
    results_dir: Path | None = None,
) -> Command:
    argv = ["--responses", _p(responses), "--prompts", _p(prompts)]
    if scoring is not None:
        argv += ["--scoring", scoring]
    argv += ["--benchmark", benchmark, "--out-dir", _p(out_dir), "--run-name", run_name]
    if results_dir is not None:
        argv += ["--results-dir", _p(results_dir)]
    return Command("run_scoring", tuple(argv))


def _metrics(
    *,
    scoring_dir: Path,
    benchmark: str,
    out_dir: Path,
    run_name: str,
    n_resamples: int | None = None,
    results_dir: Path | None = None,
) -> Command:
    argv = [
        "--scores",
        _p(scoring_dir / "scores.parquet"),
        "--drift",
        _p(scoring_dir / "drift_control.parquet"),
        "--benchmark",
        benchmark,
        "--out-dir",
        _p(out_dir),
        "--run-name",
        run_name,
    ]
    if n_resamples is not None:
        argv += ["--n-resamples", str(n_resamples)]
    if results_dir is not None:
        argv += ["--results-dir", _p(results_dir)]
    return Command("run_metrics", tuple(argv))


def e4_process_commands(
    cfg: P11Config,
    *,
    responses: dict[tuple[str, str], Path | str],
    layout: Layout = DEFAULT_LAYOUT,
    n_resamples: int | None = None,
) -> list[Command]:
    """E4's CPU half: score each generated arm per model, then the E4 analysis, then figures.

    ``process`` never touches the E4 folders (a test pins that); this is their own path. Each arm
    is scored in its OWN call, so ``run_scoring.py`` standardizes it within itself -- the 23 Sep
    2026 rule. The T=0.7 arm is E1's scores, already written by ``process``.
    """
    commands: list[Command] = []
    for arm in cfg.e4.arms:
        for model in cfg.e1.models:
            name = f"e4_{arm.name}"
            commands.append(
                _scoring(
                    responses=responses[(name, model)],
                    prompts=layout.e4_prompts(arm),
                    benchmark=OPEN_GEN_BENCHMARK,
                    out_dir=layout.scoring(name, model),
                    run_name=f"scoring-{name}-{model}",
                )
            )
    analysis = ["--data-root", _p(layout.data), "--out-dir", _p(layout.tables)]
    analysis += ["--paper-numbers", _p(layout.paper_numbers)]
    if n_resamples is not None:
        analysis += ["--n-resamples", str(n_resamples)]
    commands.append(Command("run_e4_analysis", tuple(analysis)))
    # On the repository's own layout, let make_figures RESOLVE the tables: an explicit --table-dir
    # is strict, so a table still pending after E4 (frame_breakdown) would stop the whole build,
    # where the default resolution falls back for that one figure and stamps it FIXTURE.
    figures: list[str] = []
    if layout.tables != DEFAULT_LAYOUT.tables:
        figures += ["--table-dir", _p(layout.tables)]
    figures += ["--paper-numbers", _p(layout.paper_numbers), "--out-dir", _p(layout.figures)]
    commands.append(Command("make_figures", tuple(figures)))
    return commands


def process_commands(
    cfg: P11Config,
    *,
    responses: dict[tuple[str, str], Path | str],
    layout: Layout = DEFAULT_LAYOUT,
    include_redteam: bool = True,
    stages: Sequence[str] = ("scoring", "metrics", "stats", "tables", "figures"),
) -> list[Command]:
    """Scoring -> metrics per (arm, model), then ONE stats run over every family source.

    ``responses`` maps ``(arm, model)`` to that arm's response file. The red-team directories go
    to the tables as DESCRIPTIVE and never to ``run_stats.py``: ``redteam`` is not in
    ``family_benchmarks``, and P9 would refuse it as an undeclared source.
    """
    arms = ["e1", "crows", *(["redteam"] if include_redteam else [])]
    prompts = {
        "e1": layout.e1_prompts(cfg),
        "crows": cfg.crows.prompts,
        "redteam": layout.audit_prompts(cfg),
    }
    benchmark = {
        "e1": OPEN_GEN_BENCHMARK,
        "crows": cfg.crows.benchmark,
        "redteam": REDTEAM_BENCHMARK,
    }
    # Manifests follow the layout. The default layout's results dir IS the scripts' default, so it
    # is left implicit there and the lab's command lines stay short; any other layout (the tests)
    # passes it, so no manifest can land in the committed results/ by accident.
    manifests = None if layout.results == DEFAULT_LAYOUT.results else layout.results
    commands: list[Command] = []
    if "scoring" in stages:
        for arm in arms:
            for model in cfg.e1.models:
                commands.append(
                    _scoring(
                        responses=responses[(arm, model)],
                        prompts=prompts[arm],
                        scoring=cfg.crows.scoring if arm == "crows" else None,
                        benchmark=benchmark[arm],
                        out_dir=layout.scoring(arm, model),
                        run_name=f"scoring-{arm}-{model}",
                        results_dir=manifests,
                    )
                )
    if "metrics" in stages:
        for arm in arms:
            for model in cfg.e1.models:
                commands.append(
                    _metrics(
                        scoring_dir=layout.scoring(arm, model),
                        benchmark=benchmark[arm],
                        out_dir=layout.metrics(arm, model),
                        run_name=f"metrics-{arm}-{model}",
                        results_dir=manifests,
                    )
                )
    family_dirs = [
        _p(layout.metrics(arm, model)) for arm in ("e1", "crows") for model in cfg.e1.models
    ]
    # B3 permutes the item-level scores the family's fits were made from, in the same order.
    family_scores = [
        _p(layout.scoring(arm, model) / "scores.parquet")
        for arm in ("e1", "crows")
        for model in cfg.e1.models
    ]
    descriptive_dirs = [_p(layout.metrics("redteam", model)) for model in cfg.e1.models]
    if "stats" in stages:
        commands.append(
            Command(
                "run_stats",
                (
                    "--metrics-dir",
                    *family_dirs,
                    "--scores",
                    *family_scores,
                    "--out-dir",
                    _p(layout.stats),
                    "--paper-numbers",
                    _p(layout.paper_numbers),
                    "--run-name",
                    "stats-e1",
                    *(("--results-dir", _p(manifests)) if manifests is not None else ()),
                ),
            )
        )
    if "tables" in stages:
        argv = ["--stats-dir", _p(layout.stats), "--metrics-dir", *family_dirs]
        if include_redteam:
            argv += ["--descriptive-metrics-dir", *descriptive_dirs]
        argv += ["--out-dir", _p(layout.tables)]
        commands.append(Command("build_result_tables", tuple(argv)))
    if "figures" in stages:
        # `essential` only: the two supporting figures read E4 sensitivity and per-frame tables
        # that P7/P9 do not emit; they arrive with the E4 runs, not with this processing pass.
        commands.append(
            Command(
                "make_figures",
                (
                    "--only",
                    "essential",
                    "--table-dir",
                    _p(layout.tables),
                    "--paper-numbers",
                    _p(layout.paper_numbers),
                    "--out-dir",
                    _p(layout.figures),
                ),
            )
        )
    return commands


# ---------------------------------------------------------------------------------------------
# Section B: the smoke run, fully isolated under one work directory
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SmokeLayout:
    """Everything the smoke run writes lives under ``work``. Nothing reaches ``results/``."""

    work: Path = Path("data/smoke")

    @property
    def results(self) -> Path:
        return self.work / "results"

    @property
    def sampled_prompts_dir(self) -> Path:
        return self.work / "prompts" / "e1_sampled"

    @property
    def sampled_prompts(self) -> Path:
        return self.sampled_prompts_dir / "prompts_v1.jsonl"

    @property
    def open_gen_prompts(self) -> Path:
        return self.work / "prompts" / "smoke_open_gen.jsonl"

    @property
    def crows_prompts(self) -> Path:
        return self.work / "prompts" / "smoke_crows.jsonl"

    def e4_prompts_dir(self, arm: str) -> Path:
        return self.work / "prompts" / f"e4_{arm}"

    def responses(self, arm: str) -> Path:
        return self.work / "responses" / arm

    def scoring(self, arm: str) -> Path:
        return self.work / "scoring" / arm

    def metrics(self, arm: str) -> Path:
        return self.work / "metrics" / arm

    @property
    def stats(self) -> Path:
        return self.work / "stats"

    @property
    def tables(self) -> Path:
        return self.work / "tables"

    @property
    def figures(self) -> Path:
        return self.work / "figures"

    @property
    def paper_numbers(self) -> Path:
        return self.work / "paper_numbers.json"

    @property
    def result(self) -> Path:
        return self.work / "smoke_result.json"


def smoke_prompt_build(cfg: P11Config, layout: SmokeLayout) -> Command:
    """Render the real E1 sampled set into the work dir; the smoke subset is selected from it."""
    return Command(
        "build_prompts",
        (
            "--prompt-set",
            cfg.e1.prompt_set,
            "--decoding",
            cfg.e1.decoding,
            "--out-dir",
            _p(layout.sampled_prompts_dir),
            "--results-dir",
            _p(layout.results),
            "--run-name",
            "smoke-build-prompts",
        ),
    )


def smoke_inference(cfg: P11Config, model: str, layout: SmokeLayout) -> list[Command]:
    """Open generation on the smoke grid, then a small CrowS pass (a declared family source)."""
    return [
        _inference(
            model=model,
            prompts=layout.open_gen_prompts,
            decoding=cfg.e1.decoding,
            seeds=cfg.smoke.seeds,
            batch_size=cfg.e1.batch_size,
            out_dir=layout.responses("open_gen"),
            run_name="smoke-open-gen",
            results_dir=layout.results,
        ),
        _inference(
            model=model,
            prompts=layout.crows_prompts,
            decoding=cfg.crows.decoding,
            seeds=None,
            batch_size=cfg.e1.batch_size,
            out_dir=layout.responses("crows"),
            run_name="smoke-crows",
            extra=("--extractive-set", cfg.crows.extractive_set),
            results_dir=layout.results,
        ),
    ]


def smoke_e4_commands(cfg: P11Config, model: str, layout: SmokeLayout) -> list[Command]:
    """The E4 arms through the lab's own builder and profiles, on their first few prompts.

    Proves each E4 decoding profile loads and generates through the real harness -- the
    temperature-1.0 profile is new, and greedy has not run through the P11 path at all -- without
    measuring anything. ``--limit`` is the ONLY difference from the lab command.
    """
    commands: list[Command] = []
    for arm in cfg.e4.arms:
        commands.append(
            e4_prompt_build(
                cfg,
                arm,
                out_dir=layout.e4_prompts_dir(arm.name),
                results_dir=layout.results,
            )
        )
        commands.append(
            _inference(
                model=model,
                prompts=layout.e4_prompts_dir(arm.name) / "prompts_v1.jsonl",
                decoding=arm.decoding,
                seeds=arm.seeds,
                batch_size=cfg.e1.batch_size,
                out_dir=layout.responses(f"e4_{arm.name}"),
                run_name=f"smoke-e4-{arm.name}",
                extra=("--limit", str(cfg.smoke.e4_limit)),
                results_dir=layout.results,
            )
        )
    return commands


def smoke_stage_commands(
    cfg: P11Config,
    stage: str,
    layout: SmokeLayout,
    *,
    responses: dict[str, Path | str] | None = None,
) -> list[Command]:
    """The smoke run's scoring / metrics / stats / tables commands, all inside ``layout.work``."""
    if stage == "scoring":
        if responses is None:
            raise ValueError("the scoring stage needs the response files")
        return [
            _scoring(
                responses=responses["open_gen"],
                prompts=layout.open_gen_prompts,
                benchmark=OPEN_GEN_BENCHMARK,
                out_dir=layout.scoring("open_gen"),
                run_name="smoke-scoring-open-gen",
                results_dir=layout.results,
            ),
            _scoring(
                responses=responses["crows"],
                prompts=layout.crows_prompts,
                scoring=cfg.crows.scoring,
                benchmark=cfg.crows.benchmark,
                out_dir=layout.scoring("crows"),
                run_name="smoke-scoring-crows",
                results_dir=layout.results,
            ),
        ]
    if stage == "metrics":
        return [
            _metrics(
                scoring_dir=layout.scoring(arm),
                benchmark=bench,
                out_dir=layout.metrics(arm),
                run_name=f"smoke-metrics-{arm}",
                n_resamples=cfg.smoke.n_resamples,
                results_dir=layout.results,
            )
            for arm, bench in (("open_gen", OPEN_GEN_BENCHMARK), ("crows", cfg.crows.benchmark))
        ]
    metric_dirs = (_p(layout.metrics("open_gen")), _p(layout.metrics("crows")))
    if stage == "stats":
        # The REAL stats config, unchanged: the smoke run must exercise the family the sweep will
        # be corrected over, not a smoke-shaped substitute.
        return [
            Command(
                "run_stats",
                (
                    "--metrics-dir",
                    *metric_dirs,
                    "--scores",
                    _p(layout.scoring("open_gen") / "scores.parquet"),
                    _p(layout.scoring("crows") / "scores.parquet"),
                    "--out-dir",
                    _p(layout.stats),
                    "--paper-numbers",
                    _p(layout.paper_numbers),
                    "--results-dir",
                    _p(layout.results),
                    "--run-name",
                    "smoke-stats",
                ),
            )
        ]
    if stage == "tables":
        return [
            Command(
                "build_result_tables",
                (
                    "--stats-dir",
                    _p(layout.stats),
                    "--metrics-dir",
                    *metric_dirs,
                    "--out-dir",
                    _p(layout.tables),
                    "--no-latex",
                ),
            )
        ]
    raise ValueError(f"no script commands for smoke stage {stage!r}")
