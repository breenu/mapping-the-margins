"""P11 gate: the lab-run orchestration, the smoke run's pass criterion, and the runbook's commands.

Three things are asserted here that no earlier phase could assert, because each only exists once
the scripts are chained for a real run:

1. **The smoke run reaches the correction path, or FAILS.** Real ``run_metrics`` -> ``run_stats``
   (under the SHIPPED stats config) -> ``build_result_tables`` -> the emergence figure, on
   multi-rendering score tables shaped exactly like scoring's output. And the negative: one
   rendering per cell -- the dev artifact's shape, where the P9 bug hid -- must be a FAIL, never a
   vacuous pass.
2. **Every command the orchestrator or the runbook will run parses against the REAL script.** A
   flag that does not exist fails here, not on a booked GPU.
3. **Every refusal fires**: dirty tree, pre-registration ancestry, a confirmatory model that fell
   back from NF4, a resume at a different batch size, a descriptive arm carrying significance.

Core environment only: no torch, no model, no network.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shlex
import shutil
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from margins.lattice import DeclaredLattice, load_lattice_config
from margins.orchestrate import plan
from margins.orchestrate.budget import (
    MIN_REDTEAM_FRAMES,
    project_gpu_hours,
    redteam_max_frames,
)
from margins.orchestrate.config import load_p11_config
from margins.orchestrate.preflight import (
    batch_size_conflict,
    dirty_tree_reason,
    prereg_reason,
    quant_reason,
)
from margins.orchestrate.smoke import pair_cells, select_smoke_prompts
from margins.prompts import read_prompts
from margins.redteam import Frame, load_redteam_config, select_frames, target_cells
from margins.schemas import EffectEstimate
from tests.p11_synthetic import write_crows, write_open_gen

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
RUNBOOK = REPO / "docs" / "runbook_p11.md"
DEV_MODEL = "llama32_1b_instruct"
DEV_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
LLAMA_8B = "meta-llama/Llama-3.1-8B-Instruct"
MISTRAL = "mistralai/Mistral-7B-Instruct-v0.3"


def _script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"_test_{name}", SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cfg():  # type: ignore[no-untyped-def]
    return load_p11_config()


@pytest.fixture(scope="module")
def lattice() -> DeclaredLattice:
    return DeclaredLattice(load_lattice_config("confirmatory_v1"))


# ---------------------------------------------------------------------------------------------
# the declaration
# ---------------------------------------------------------------------------------------------


def test_e1_seeds_are_the_config_seed_plus_zero_to_three(cfg) -> None:  # type: ignore[no-untyped-def]
    """DECIDED 23 Sep 2026: four DISTINCT seeds, derived from the root config's seed."""
    from margins.config import load_experiment_config

    base = load_experiment_config().seed
    assert cfg.e1.seeds == tuple(base + i for i in range(4))
    assert cfg.e1.decoding == "sampled_v1"


def test_the_declared_design_matches_the_preregistered_one(cfg, lattice) -> None:  # type: ignore[no-untyped-def]
    """27 prompts per cell x 4 seeds = min_cell_n 108, and the red-team cell count is real."""
    assert len(cfg.e1.seeds) * 27 == lattice.config.power.min_cell_n == 108
    assert set(cfg.e1.models) == {"llama31_8b_instruct", "mistral7b_instruct_v03"}
    redteam = load_redteam_config(cfg.redteam.set)
    assert cfg.redteam.target_cells == len(target_cells(redteam, lattice))


def test_the_redteam_arm_is_not_in_the_fdr_family(cfg) -> None:  # type: ignore[no-untyped-def]
    """The 21 Sep 2026 decision: descriptive, powered=False, outside family_benchmarks."""
    from margins.stats.config import load_stats_config

    assert plan.REDTEAM_BENCHMARK not in load_stats_config().fdr.family_benchmarks
    assert len(cfg.redteam.audit_seeds) == 1


def test_the_smoke_design_is_estimable(cfg, lattice) -> None:  # type: ignore[no-untyped-def]
    """A full confirmatory grid, >= 2 items per cell, n_obs > n_params -- or it tests nothing."""
    cells = pair_cells(lattice, cfg.smoke.pair)
    assert cells and len(cfg.smoke.frames) >= 2 and len(cfg.smoke.seeds) >= 2
    assert len(cells) * len(cfg.smoke.frames) * len(cfg.smoke.seeds) > len(cells)


def test_an_exploratory_smoke_pair_is_refused(lattice) -> None:
    with pytest.raises(ValueError, match="not a confirmatory pair"):
        pair_cells(lattice, ("age", "race_ethnicity"))


# ---------------------------------------------------------------------------------------------
# the E1 prompt re-stamp and the smoke subset
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sampled_set(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("sampled")
    code = _script("build_prompts").main(
        ["--decoding", "sampled_v1", "--out-dir", str(out), "--no-manifest"]
    )
    assert code == 0
    return out / "prompts_v1.jsonl"


def test_decoding_restamp_changes_keys_and_ids_but_never_text(
    sampled_set: Path, tmp_path: Path
) -> None:
    assert _script("build_prompts").main(["--out-dir", str(tmp_path), "--no-manifest"]) == 0
    greedy = read_prompts(tmp_path / "prompts_v1.jsonl")
    sampled = read_prompts(sampled_set)
    assert len(greedy) == len(sampled) == 2970
    assert {p.decoding_key for p in greedy} == {"greedy_v1"}
    assert {p.decoding_key for p in sampled} == {"sampled_v1"}
    assert [p.rendered_text for p in greedy] == [p.rendered_text for p in sampled]
    assert not {p.prompt_id for p in greedy} & {p.prompt_id for p in sampled}


def test_decoding_restamp_is_byte_deterministic(sampled_set: Path, tmp_path: Path) -> None:
    argv = ["--decoding", "sampled_v1", "--out-dir", str(tmp_path), "--no-manifest"]
    assert _script("build_prompts").main(argv) == 0
    assert (tmp_path / "prompts_v1.jsonl").read_bytes() == sampled_set.read_bytes()


def test_an_unknown_decoding_profile_is_refused_before_anything_is_written(
    tmp_path: Path,
) -> None:
    with pytest.raises(Exception, match="no_such_profile"):
        _script("build_prompts").main(
            ["--decoding", "no_such_profile", "--out-dir", str(tmp_path), "--no-manifest"]
        )
    assert not (tmp_path / "prompts_v1.jsonl").exists()


def test_smoke_subset_is_a_full_grid_of_real_e1_prompts(cfg, lattice, sampled_set: Path) -> None:  # type: ignore[no-untyped-def]
    full = read_prompts(sampled_set)
    cells = pair_cells(lattice, cfg.smoke.pair)
    chosen = select_smoke_prompts(
        full,
        cells=cells,
        frames=cfg.smoke.frames,
        carrier=cfg.smoke.carrier,
        decoding_key=cfg.e1.decoding,
    )
    assert len(chosen) == len(cells) * len(cfg.smoke.frames) == 36
    assert {p.prompt_id for p in chosen} <= {p.prompt_id for p in full}
    assert {p.cell_id for p in chosen} == {c.cell_id for c in cells}


def test_smoke_subset_refuses_greedy_prompts_and_missing_frames(
    cfg, lattice, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    assert _script("build_prompts").main(["--out-dir", str(tmp_path), "--no-manifest"]) == 0
    greedy = read_prompts(tmp_path / "prompts_v1.jsonl")
    cells = pair_cells(lattice, cfg.smoke.pair)
    with pytest.raises(ValueError, match="stamped"):
        select_smoke_prompts(
            greedy,
            cells=cells,
            frames=cfg.smoke.frames,
            carrier="person",
            decoding_key="sampled_v1",
        )
    with pytest.raises(ValueError, match="not a full grid"):
        select_smoke_prompts(
            greedy,
            cells=cells,
            frames=("no_such_frame",),
            carrier="person",
            decoding_key="greedy_v1",
        )


# ---------------------------------------------------------------------------------------------
# THE GATE: the smoke run reaches correction -> ClaimGate -> figure, and FAILS when it cannot
# ---------------------------------------------------------------------------------------------


def _smoke_from_metrics(work: Path, *, n_items: int, seeds: tuple[int, ...]) -> tuple[int, dict]:  # type: ignore[type-arg]
    lattice = DeclaredLattice(load_lattice_config("confirmatory_v1"))
    cfg = load_p11_config()
    write_open_gen(
        work / "scoring" / "open_gen",
        lattice,
        pair=cfg.smoke.pair,
        model_id=DEV_MODEL_ID,
        n_items=n_items,
        seeds=seeds,
    )
    write_crows(work / "scoring" / "crows", lattice, model_id=DEV_MODEL_ID)
    code = _script("run_p11").main(
        ["smoke", "--model", DEV_MODEL, "--work-dir", str(work), "--from-stage", "metrics"]
    )
    return code, json.loads((work / "smoke_result.json").read_text(encoding="utf-8"))


def test_the_smoke_run_reaches_correction_claim_gate_and_figure(tmp_path: Path) -> None:
    code, record = _smoke_from_metrics(tmp_path, n_items=3, seeds=(1, 2, 3))
    checks = {check["name"]: check for check in record["checks"]}
    assert code == 0 and record["passed"], json.dumps(record["checks"], indent=1)
    assert "family_size" in checks["FDR correction reached"]["detail"]
    # The expected verdict at smoke n, and the proof the gate RAN rather than being skipped.
    assert "underpowered" in checks["ClaimGate ran on every corrected term"]["detail"]
    assert (tmp_path / "figures" / "smoke_fig1_emergence.pdf").is_file()
    assert (tmp_path / "figures" / "smoke_fig1_emergence.svg").is_file()
    # Isolation: every manifest went under the work dir, none into the committed results/.
    assert list((tmp_path / "results" / "run_manifests").glob("smoke-stats-*.json"))


def test_one_rendering_per_cell_is_a_fail_not_a_vacuous_pass(tmp_path: Path) -> None:
    """The dev artifact's shape. P9 prints NOTHING TO CORRECT; the smoke run must say FAIL."""
    code, record = _smoke_from_metrics(tmp_path, n_items=1, seeds=(1,))
    checks = {check["name"]: check for check in record["checks"]}
    assert code == 1 and not record["passed"]
    assert not checks["estimable confirmatory pair"]["passed"]
    assert not checks["FDR correction reached"]["passed"]
    assert "NOTHING TO CORRECT" in checks["FDR correction reached"]["detail"]


def test_the_smoke_run_leaves_the_committed_results_untouched(tmp_path: Path) -> None:
    watched = [REPO / "results" / "paper_numbers.json", *(REPO / "figures").glob("*.svg")]
    before = {path: path.read_bytes() for path in watched}
    manifests = set((REPO / "results" / "run_manifests").glob("*.json"))
    _smoke_from_metrics(tmp_path, n_items=3, seeds=(1, 2))
    assert {path: path.read_bytes() for path in watched} == before
    assert set((REPO / "results" / "run_manifests").glob("*.json")) == manifests
    assert not (REPO / "results" / "tables").exists() or not any(
        (REPO / "results" / "tables").iterdir()
    )


# ---------------------------------------------------------------------------------------------
# E: processing, two models, per-model standardization, the red-team arm DESCRIPTIVE
# ---------------------------------------------------------------------------------------------


def test_processing_two_models_corrects_one_family_and_keeps_redteam_descriptive(
    tmp_path: Path,
) -> None:
    """Four P7 dirs into ONE P9 family; red-team into the tables with no p and no q; figures."""
    lattice = DeclaredLattice(load_lattice_config("confirmatory_v1"))
    layout = plan.Layout(data=tmp_path / "data", results=tmp_path / "results")
    pair = ("socioeconomic_class", "race_ethnicity")
    for index, (name, model_id) in enumerate(
        (("llama31_8b_instruct", LLAMA_8B), ("mistral7b_instruct_v03", MISTRAL))
    ):
        for arm, benchmark in (("e1", "open_gen_templates"), ("redteam", "redteam")):
            write_open_gen(
                layout.scoring(arm, name),
                lattice,
                pair=pair,
                model_id=model_id,
                n_items=4,
                seeds=(1, 2),
                benchmark=benchmark,
                seed=10 * index + len(arm),
            )
        write_crows(layout.scoring("crows", name), lattice, model_id=model_id, seed=index)
    layout.results.mkdir(parents=True)
    # fig3 draws the MEASURED field result from the committed paper numbers.
    shutil.copy(REPO / "results" / "paper_numbers.json", layout.paper_numbers)

    code = _script("run_p11").main(
        [
            "process",
            "--from-stage",
            "metrics",
            "--data-root",
            str(layout.data),
            "--results-root",
            str(layout.results),
            "--figures-dir",
            str(tmp_path / "figures"),
        ]
    )
    assert code == 0

    from margins.metrics.export import read_records
    from margins.report.tables import read_table
    from margins.stats.export import EFFECTS_CORRECTED_FILENAME

    corrected = read_records(layout.stats / EFFECTS_CORRECTED_FILENAME, EffectEstimate)
    members = [e for e in corrected if e.q is not None]
    assert {e.model for e in members} == {LLAMA_8B, MISTRAL}
    assert {e.benchmark for e in corrected} == {"open_gen_templates"}
    # One family across both models (9 free terms x 2 harm types x 2 models), never per model.
    assert {e.family_size for e in members} == {36}

    effects = read_table("emergence_effects", directory=layout.tables)
    redteam = effects.where(benchmark="redteam")
    assert redteam.rows, "the red-team arm never reached the tables"
    assert all(row["q"] == "" and row["p"] == "" for row in redteam.rows)
    assert len(list((tmp_path / "figures").glob("*.pdf"))) == 5
    # Manifests went where the layout said, not into the committed results/.
    assert list((layout.results / "run_manifests").glob("stats-e1-*.json"))


def test_a_descriptive_arm_carrying_significance_is_refused(cfg) -> None:  # type: ignore[no-untyped-def]
    from margins.stats.config import load_stats_config

    refuse = _script("build_result_tables")._refuse_significance_on_descriptive
    stats = load_stats_config()
    base: dict[str, Any] = {
        "benchmark": "redteam",
        "model": LLAMA_8B,
        "axis_pair": ("age", "ability"),
        "term": "age=young:ability=blind",
        "harm_type": "regard",
        "estimate": 0.1,
        "se": 0.1,
        "ci_low": -0.1,
        "ci_high": 0.3,
        "powered": False,
    }
    refuse([EffectEstimate(**base)], stats)  # clean: passes
    with pytest.raises(ValueError, match="invariant 3"):
        refuse([EffectEstimate(**{**base, "p": 0.01})], stats)
    with pytest.raises(ValueError, match="FDR-family source"):
        refuse([EffectEstimate(**{**base, "benchmark": "open_gen_templates"})], stats)


# ---------------------------------------------------------------------------------------------
# every command is a real interface
# ---------------------------------------------------------------------------------------------


class _Parsed(Exception):  # noqa: N818 - a control-flow signal, not an error
    def __init__(self, namespace: argparse.Namespace) -> None:
        self.namespace = namespace


@pytest.fixture
def capture_parse(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Make every ArgumentParser stop right after parsing, and reject unknown arguments."""

    def parse_args(self: argparse.ArgumentParser, args: Any = None, namespace: Any = None) -> Any:
        parsed, extras = self.parse_known_args(args, namespace)
        if extras:
            raise AssertionError(f"{self.prog}: unrecognized arguments {extras}")
        raise _Parsed(parsed)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_args)
    yield


def _parses(command: plan.Command) -> argparse.Namespace:
    with pytest.raises(_Parsed) as caught:
        _script(command.script).main(list(command.argv))
    return caught.value.namespace


def _every_planned_command(cfg) -> list[plan.Command]:  # type: ignore[no-untyped-def]
    redteam = load_redteam_config(cfg.redteam.set)
    from margins.redteam import elicitation_seeds

    responses = {
        (arm, model): f"data/responses/{arm}/{model}/run.jsonl"
        for arm in plan.ARMS
        for model in cfg.e1.models
    }
    smoke = plan.SmokeLayout()
    commands = [
        *plan.prepare_commands(cfg),
        *(plan.e1_inference(cfg, m) for m in cfg.e1.models),
        *(plan.crows_inference(cfg, m) for m in cfg.e1.models),
        *plan.redteam_elicitation_commands(
            cfg,
            generator=redteam.generator.model_name,
            generator_decoding=redteam.generator.decoding_name,
            elicitation_seeds=elicitation_seeds(redteam),
        ),
        plan.redteam_frames(cfg, "data/responses/redteam_elicitation/x/run.jsonl"),
        plan.redteam_audit_prompts(cfg, 9),
        *(plan.redteam_audit_inference(cfg, m) for m in cfg.e1.models),
        *plan.process_commands(cfg, responses=responses),
        plan.smoke_prompt_build(cfg, smoke),
        *plan.smoke_inference(cfg, DEV_MODEL, smoke),
        *plan.smoke_stage_commands(
            cfg, "scoring", smoke, responses={"open_gen": "a.jsonl", "crows": "b.jsonl"}
        ),
        *plan.smoke_stage_commands(cfg, "metrics", smoke),
        *plan.smoke_stage_commands(cfg, "stats", smoke),
        *plan.smoke_stage_commands(cfg, "tables", smoke),
    ]
    return commands


def test_every_planned_command_parses_against_the_real_script(cfg, capture_parse) -> None:  # type: ignore[no-untyped-def]
    commands = _every_planned_command(cfg)
    assert len(commands) > 30
    for command in commands:
        _parses(command)


def test_the_e1_command_carries_the_declared_design(cfg, capture_parse) -> None:  # type: ignore[no-untyped-def]
    namespace = _parses(plan.e1_inference(cfg, "llama31_8b_instruct"))
    assert namespace.seeds == list(cfg.e1.seeds)
    assert namespace.decoding == "sampled_v1"
    assert namespace.resume is True
    assert namespace.batch_size == cfg.e1.batch_size
    assert namespace.prompts == Path("data/prompts/e1_sampled/prompts_v1.jsonl")


def test_the_elicitation_run_carries_every_candidate_seed(cfg, capture_parse) -> None:  # type: ignore[no-untyped-def]
    """The P11 fix: without --seeds the red LM drew ONE candidate per target instead of six."""
    from margins.redteam import elicitation_seeds

    redteam = load_redteam_config(cfg.redteam.set)
    _, inference = plan.redteam_elicitation_commands(
        cfg,
        generator=redteam.generator.model_name,
        generator_decoding=redteam.generator.decoding_name,
        elicitation_seeds=elicitation_seeds(redteam),
    )
    assert _parses(inference).seeds == list(elicitation_seeds(redteam))
    assert len(elicitation_seeds(redteam)) == redteam.elicitation.candidates_per_target


def test_stats_never_receives_the_redteam_arm(cfg) -> None:  # type: ignore[no-untyped-def]
    commands = plan.process_commands(cfg, responses={}, stages=("stats", "tables"))
    stats = next(c for c in commands if c.script == "run_stats")
    assert not any("redteam" in part for part in stats.argv)
    tables = next(c for c in commands if c.script == "build_result_tables")
    descriptive = tables.argv[tables.argv.index("--descriptive-metrics-dir") + 1 :]
    assert all("redteam" in part for part in descriptive[: len(cfg.e1.models)])


def _runbook_commands() -> list[str]:
    """Every ``uv run python scripts/...`` line in the runbook's code blocks."""
    text = RUNBOOK.read_text(encoding="utf-8")
    blocks = re.findall(r"```(?:bash|sh)?\n(.*?)```", text, flags=re.DOTALL)
    commands: list[str] = []
    for block in blocks:
        joined = re.sub(r"\\\n\s*", " ", block)
        for line in joined.splitlines():
            line = line.split(" #", 1)[0].strip()
            if line.startswith("uv run python scripts/"):
                commands.append(line)
    return commands


def test_every_runbook_command_is_a_real_interface(cfg, capture_parse) -> None:  # type: ignore[no-untyped-def]
    """The runbook is followed literally by someone at a lab machine. No invented flag survives."""
    commands = _runbook_commands()
    assert len(commands) >= 20, "the runbook lost its command list"
    for line in commands:
        # A `<placeholder>` becomes "1": parsable both as a path and as an int like --max-frames.
        tokens = shlex.split(re.sub(r"<[^>]+>", "1", line))[3:]
        script = Path(tokens[0]).stem
        argv = tokens[1:]
        with pytest.raises(_Parsed):
            if script == "run_p11":
                _script("run_p11").build_parser().parse_args(argv)
            else:
                _script(script).main(argv)


def test_the_runbook_quotes_the_declared_seeds_and_nothing_else(cfg) -> None:  # type: ignore[no-untyped-def]
    for line in _runbook_commands():
        if "--seeds" in line and "sampled_v1" in line and "smoke" not in line:
            seeds = [int(s) for s in re.findall(r"--seeds ((?:\d+ ?)+)", line)[0].split()]
            assert tuple(seeds) in {cfg.e1.seeds, cfg.redteam.audit_seeds}, line


def test_the_runbook_states_the_pass_criterion_and_the_stop_rule() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "Do not proceed to the full run until this passes" in text
    assert "NOTHING TO CORRECT" in text
    assert "never unpin" in text.lower()


# ---------------------------------------------------------------------------------------------
# refusals and arithmetic
# ---------------------------------------------------------------------------------------------


def test_preflight_refusals() -> None:
    assert dirty_tree_reason(False) is None
    assert "uncommitted" in (dirty_tree_reason(True) or "")
    assert "not a git checkout" in (dirty_tree_reason(None) or "")
    assert prereg_reason(True) is None
    assert "earlier commit" in (prereg_reason(False) or "")
    assert "git fetch --tags" in (prereg_reason(None) or "")


@pytest.mark.parametrize(
    ("role", "actual", "allow", "refused"),
    [
        ("confirmatory", "none", False, True),  # the silent fp16 fallback, caught
        ("confirmatory", "nf4", False, False),
        ("confirmatory", "none", True, False),  # explicit, recorded override
        ("dev", "none", False, False),  # the dev box's normal state
        ("ci", "none", False, False),
    ],
)
def test_quant_guard(role: str, actual: str, allow: bool, refused: bool) -> None:
    reason = quant_reason(
        requested="nf4",
        actual=actual,
        fallback_reason="bitsandbytes is not installed",
        role=role,
        allow_fallback=allow,
    )
    assert (reason is not None) is refused


def test_a_resume_at_a_different_batch_size_is_refused(tmp_path: Path) -> None:
    manifest = tmp_path / "run.json"
    assert batch_size_conflict(manifest, 8) is None  # a fresh run
    manifest.write_text(json.dumps({"inference": {"batch_size": 8}}), encoding="utf-8")
    assert batch_size_conflict(manifest, 8) is None
    assert "batch size 8, not 4" in (batch_size_conflict(manifest, 4) or "")


def test_untracked_manifests_do_not_make_the_tree_dirty() -> None:
    """Every stage writes a manifest; counting them would stamp git_dirty on all but the first."""
    from margins.config import dirty_status_lines

    status = "\n".join(
        [
            "?? results/run_manifests/e1-llama-20260930T000000Z-abc.json",
            " M results/run_manifests/committed.json",
            "?? scripts/stray.py",
            " M src/margins/config.py",
        ]
    )
    kept = dirty_status_lines(status, ignore_untracked_under="results/run_manifests")
    assert kept == status.splitlines()[1:]
    assert (
        dirty_status_lines(status.splitlines()[0], ignore_untracked_under="results/run_manifests")
        == []
    )
    assert len(dirty_status_lines(status)) == 4


def test_gpu_hour_projection_matches_the_pilot_arithmetic(cfg) -> None:  # type: ignore[no-untyped-def]
    """The runbook's per-model estimate: 2,970 prompts x 4 seeds at the T4 pilot speed."""
    n = 2970 * len(cfg.e1.seeds)
    assert project_gpu_hours(
        cfg.pilot_sec_per_generation["llama31_8b_instruct"], n
    ) == pytest.approx(46.2)
    assert project_gpu_hours(
        cfg.pilot_sec_per_generation["mistral7b_instruct_v03"], n
    ) == pytest.approx(48.84)
    with pytest.raises(ValueError):
        project_gpu_hours(0.0, n)


def test_the_redteam_frame_cap_fits_the_budget_and_refuses_below_the_minimum(cfg) -> None:  # type: ignore[no-untyped-def]
    speeds = dict(cfg.pilot_sec_per_generation)
    budget = redteam_max_frames(
        budget_hours=20.0,
        elicitation_generations=1440,
        red_lm_sec_per_generation=speeds["llama31_8b_instruct"],
        target_sec_per_generation=speeds,
        target_cells=80,
        carriers=2,
        audit_seeds=1,
    )
    assert budget.max_frames >= MIN_REDTEAM_FRAMES
    assert budget.total_hours <= 20.0
    assert budget.total_hours + budget.per_frame_hours > 20.0  # the cap is the largest that fits
    with pytest.raises(ValueError, match="below the minimum"):
        redteam_max_frames(
            budget_hours=6.0,
            elicitation_generations=1440,
            red_lm_sec_per_generation=14.0,
            target_sec_per_generation=speeds,
            target_cells=80,
            carriers=2,
            audit_seeds=1,
        )


def _frame(fid: str, angle: str, cell: str) -> Frame:
    return Frame(
        frame_id=fid,
        text=f"text {fid} [PERSON]",
        template_text=f"text {fid} {{{{ identity }}}}",
        target_cell_id=cell,
        target_cell_ids=(cell,),
        elicitation_template=angle,
        response_id=f"r{fid}",
        seed=1,
    )


def test_frame_selection_is_stratified_content_ordered_and_capped() -> None:
    pair_of = {"c1": ("a", "b"), "c2": ("c", "d")}
    frames = [
        _frame("f9", "narrate", "c1"),
        _frame("f1", "narrate", "c1"),
        _frame("f5", "narrate", "c1"),
        _frame("f2", "decide", "c2"),
        _frame("f3", "decide", "c2"),
    ]
    chosen = select_frames(frames, 2, pair_of=pair_of)
    # One per stratum before any second, lowest frame_id first, returned in input order.
    assert [f.frame_id for f in chosen] == ["f1", "f2"]
    assert select_frames(list(reversed(frames)), 2, pair_of=pair_of) == [
        f for f in reversed(frames) if f.frame_id in {"f1", "f2"}
    ]
    assert select_frames(frames, 99, pair_of=pair_of) == frames
    with pytest.raises(KeyError, match="not an order-2"):
        select_frames([*frames, _frame("f0", "x", "nope")], 2, pair_of=pair_of)
    with pytest.raises(ValueError):
        select_frames(frames, 0, pair_of=pair_of)
