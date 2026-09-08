"""The Kaggle timing pilot: drift guards and extrapolation arithmetic (P5 / W2).

The pilot runs somewhere this suite cannot reach, so what *is* testable here matters more than
usual:

* **It cannot drift.** The prompt sample, the model pins and the decoding profile are generated into
  the script from the same configs the pipeline uses. Regenerating them here and comparing is what
  stops the pilot from timing a stale prompt set or an unpinned checkpoint -- which would produce a
  budget for an experiment nobody is going to run.
* **It stays self-contained.** Kaggle has no checkout of this repo, so an ``import margins`` would
  turn a one-click notebook into a debugging session in a place Claude Code cannot reach.
* **Its arithmetic is right.** The measurement happens on Kaggle, but the extrapolation from
  seconds-per-prompt to GPU-hours is ordinary arithmetic and is checked here against synthetic
  timings with a known answer -- the same test-first discipline the metric layer gets.

What is deliberately NOT tested: any throughput figure. There is none in the file, and there must
not be: the whole point of the pilot is that the number is measured rather than assumed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_SCRIPT = REPO_ROOT / "scripts" / "kaggle_timing_pilot.py"
PILOT_NOTEBOOK = REPO_ROOT / "notebooks" / "kaggle_timing_pilot.ipynb"


@pytest.fixture(scope="module")
def pilot() -> ModuleType:
    """Import the pilot as a module. It must load with no `margins` and no torch present."""
    spec = importlib.util.spec_from_file_location("kaggle_timing_pilot", PILOT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------------------
# Self-containment
# --------------------------------------------------------------------------------------


def test_the_pilot_imports_nothing_from_margins() -> None:
    """Kaggle has no checkout of this repo; an import would strand the notebook."""
    source = PILOT_SCRIPT.read_text(encoding="utf-8")
    for line in source.splitlines():
        stripped = line.strip()
        assert not stripped.startswith(("import margins", "from margins")), (
            f"the pilot must be self-contained, found: {stripped!r}"
        )


def test_torch_is_imported_lazily(pilot: ModuleType) -> None:
    """Importing the module must not require the inference stack -- this test just did it."""
    assert pilot.PROMPT_SAMPLE, "the generated block is empty"


# --------------------------------------------------------------------------------------
# Drift guards
# --------------------------------------------------------------------------------------


def test_embedded_block_and_notebook_match_the_current_configs() -> None:
    """Regenerate and compare. A stale pilot budgets an experiment nobody will run."""
    from scripts.build_pilot_sample import main as build

    assert build(["--check"]) == 0, (
        "the pilot's embedded data is stale. Run `uv run python scripts/build_pilot_sample.py`."
    )


def test_embedded_prompts_are_the_real_rendered_prompts(pilot: ModuleType) -> None:
    """Not paraphrases: every embedded prompt must appear verbatim in the live prompt set."""
    from margins.lattice import DEFAULT_LATTICE, DeclaredLattice, load_lattice_config
    from margins.lattice.config import default_lattice_config_dir
    from margins.prompts import (
        DEFAULT_PROMPT_SET,
        TemplateRenderer,
        default_prompt_config_dir,
        load_prompt_config,
        load_template_files,
        template_items,
    )
    from margins.prompts.config import TEMPLATE_DIRNAME

    lattice = DeclaredLattice(
        load_lattice_config(DEFAULT_LATTICE, config_dir=default_lattice_config_dir())
    )
    prompt_dir = default_prompt_config_dir()
    config = load_prompt_config(DEFAULT_PROMPT_SET, config_dir=prompt_dir)
    files = load_template_files(config, prompt_dir / TEMPLATE_DIRNAME)
    renderer = TemplateRenderer(config, lattice)
    rendered = renderer.render(template_items(config, files), lattice.cells())
    live = {prompt.prompt_id: prompt.rendered_text for prompt in rendered}

    for prompt_id, text in pilot.PROMPT_SAMPLE:
        assert prompt_id in live, f"{prompt_id!r} is not in the current prompt set"
        assert live[prompt_id] == text, f"{prompt_id!r} renders differently now"


def test_embedded_pins_match_the_model_configs(pilot: ModuleType) -> None:
    """A pilot timing an unpinned or differently-pinned checkpoint measures the wrong model."""
    from margins.inference.models import CONFIRMATORY_MODELS, load_model_config

    embedded = {entry["name"]: entry for entry in pilot.MODELS}
    assert set(embedded) == set(CONFIRMATORY_MODELS)
    for name in CONFIRMATORY_MODELS:
        spec = load_model_config(name)
        assert embedded[name]["model_id"] == spec.model_id
        assert embedded[name]["revision"] == spec.revision
        assert len(embedded[name]["revision"]) == 40


def test_embedded_decoding_matches_the_shipped_profile(pilot: ModuleType) -> None:
    from margins.inference.decoding import DEFAULT_DECODING, load_decoding_config

    config = load_decoding_config(DEFAULT_DECODING)
    assert pilot.DECODING == config.model_dump(mode="json")
    assert pilot.PILOT_META["decoding_hash"] == config.decoding_hash


def test_prompt_set_size_constant_matches_the_real_set(pilot: ModuleType) -> None:
    """The extrapolation is per-prompt x this number; a stale 880 would understate the budget."""
    assert pilot.PROMPT_SET_SIZE == pilot.PILOT_META["prompt_set_size"] == 990


def test_the_sample_spans_many_cell_groups_and_frames(pilot: ModuleType) -> None:
    """Generation cost varies with prompt length, so a one-frame sample would mis-time the set."""
    frames = {prompt_id.split("::")[0] for prompt_id, _ in pilot.PROMPT_SAMPLE}
    cells = {prompt_id.split("::")[1] for prompt_id, _ in pilot.PROMPT_SAMPLE}
    assert len(pilot.PROMPT_SAMPLE) == 50
    assert len(frames) >= 5, f"sample covers only {frames}"
    assert len(cells) >= 25, f"sample covers only {len(cells)} cells"


def test_notebook_is_two_cells_wrapping_the_script(pilot: ModuleType) -> None:
    notebook = json.loads(PILOT_NOTEBOOK.read_text(encoding="utf-8"))
    assert [cell["cell_type"] for cell in notebook["cells"]] == ["markdown", "code"]

    readme = "".join(notebook["cells"][0]["source"])
    for required in ("GPU T4", "Internet", "HF_TOKEN", "Run All", "Colab"):
        assert required in readme, f"the setup instructions lost {required!r}"

    code = "".join(notebook["cells"][1]["source"])
    assert "raise SystemExit(main())" not in code, "SystemExit surfaces as a traceback in a kernel"
    assert code.rstrip().endswith("main([])"), (
        "the notebook's last statement must be `main([])`. Anything that reaches argparse with the "
        "kernel's argv dies on `-f /root/.../kernel-<uuid>.json` -- the failure this pilot hit."
    )
    assert code.count("PROMPT_SAMPLE: tuple") == 1


# --------------------------------------------------------------------------------------
# Notebook hosts: the failure this pilot actually hit in the field
# --------------------------------------------------------------------------------------


def test_pasting_the_script_into_a_notebook_does_not_parse_the_kernels_argv(
    pilot: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pasted cell has ``__name__ == "__main__"`` and the KERNEL's argv.

    Reported from a real session::

        colab_kernel_launcher.py: error: unrecognized arguments:
            -f /root/.local/share/jupyter/runtime/kernel-4fbcf93b-....json

    Pasting the file into one cell is a documented way to run the pilot, so ``main([])`` -- not
    ``main()`` -- must be what a kernel reaches. Checked on the source text because the guard runs
    under ``__main__``, which importing the module deliberately does not trigger.
    """
    source = PILOT_SCRIPT.read_text(encoding="utf-8")
    tail = source[source.index('if __name__ == "__main__":') :]
    assert "in_notebook()" in tail, "the entrypoint does not check for a notebook kernel"
    assert "main([])" in tail, "a notebook must be handed an explicit empty argv"

    # And argparse really does reject a kernel's argv, which is why the guard has to exist.
    monkeypatch.setattr(
        sys, "argv", ["colab_kernel_launcher.py", "-f", "/root/.../kernel-abc.json"]
    )
    with pytest.raises(SystemExit):
        pilot.main()  # argv=None -> reads sys.argv -> argparse rejects `-f kernel.json`


def test_token_lookup_covers_both_notebook_hosts() -> None:
    """Kaggle and Colab expose secrets through completely different APIs.

    Looking in only one place means the gated Meta checkpoints 401 several minutes into a session,
    after the weight download has already started -- on the host PROJECT_CONTEXT.md section 7 names
    as the debugging backup.
    """
    source = PILOT_SCRIPT.read_text(encoding="utf-8")
    assert "kaggle_secrets" in source
    assert "google.colab" in source
    assert 'os.environ.get("HF_TOKEN")' in source


def test_pinned_packages_come_from_the_lockfile(pilot: ModuleType) -> None:
    """CLAUDE.md forbids installing unpinned "to make it work".

    Kaggle's image has no bitsandbytes, so the pilot has to install one -- and the only version it
    may install is the one the locked linux_x86_64 environment resolves to. Read from ``uv.lock``,
    never typed, so the two cannot drift.
    """
    from scripts.build_pilot_sample import LOCK_PINNED_PACKAGES, locked_versions

    assert pilot.PINNED_PACKAGES == locked_versions(LOCK_PINNED_PACKAGES)
    assert "bitsandbytes" in pilot.PINNED_PACKAGES, (
        "bitsandbytes must be pinned: its absence on Kaggle is what broke the first pilot run"
    )
    for version in pilot.PINNED_PACKAGES.values():
        assert version[0].isdigit(), f"{version!r} is not an exact version"


def test_dependencies_are_checked_before_any_weight_download(pilot: ModuleType) -> None:
    """Order matters: discovering a missing package from inside from_pretrained costs minutes."""
    source = PILOT_SCRIPT.read_text(encoding="utf-8")
    body = source[source.index("def main(") : source.index("def in_notebook(")]
    assert body.index("ensure_dependencies(") < body.index("time_model("), (
        "the dependency preflight must run before the first model load, or a missing package is "
        "discovered only after the tokenizer download has started"
    )


def test_a_missing_pinned_package_can_be_reported_instead_of_installed(
    pilot: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--no-install: name the exact pip command rather than changing someone's environment."""
    monkeypatch.setattr(pilot, "package_version", lambda name: None)
    with pytest.raises(SystemExit, match="bitsandbytes=="):
        pilot.ensure_dependencies(install=False)


def test_an_already_satisfied_pin_installs_nothing(
    pilot: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pilot, "package_version", lambda name: pilot.PINNED_PACKAGES.get(name))
    monkeypatch.setattr(pilot, "_verify_bitsandbytes_visible_to_transformers", lambda: None)
    versions = pilot.ensure_dependencies(install=True)
    assert versions["bitsandbytes"] == pilot.PINNED_PACKAGES["bitsandbytes"]


def _fake_transformers(monkeypatch: pytest.MonkeyPatch, *, available: bool) -> None:
    utils = ModuleType("transformers.utils")
    utils.is_bitsandbytes_available = lambda: available  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", ModuleType("transformers"))
    monkeypatch.setitem(sys.modules, "transformers.utils", utils)


def test_transformers_is_asked_whether_the_4bit_path_works(
    pilot: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _fake_transformers(monkeypatch, available=True)
    pilot._verify_bitsandbytes_visible_to_transformers()
    assert "can see bitsandbytes" in capsys.readouterr().out


def test_bitsandbytes_invisible_to_transformers_fails_before_downloading(
    pilot: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Covers BOTH ways this ends in the same ImportError 16 GB later.

    Either bitsandbytes was never installed (Kaggle's image ships none), or it was installed just
    now but this cell already ran once, so transformers cached the old answer -- which looks like
    the install failed. It did not; the kernel needs restarting, and the script must say so.
    """
    _fake_transformers(monkeypatch, available=False)
    with pytest.raises(SystemExit, match=r"(?i)restart the session"):
        pilot._verify_bitsandbytes_visible_to_transformers()


def test_the_install_does_not_let_pip_touch_torch(pilot: ModuleType) -> None:
    """Resolving bitsandbytes' torch dependency could swap the host's CUDA build for a generic one.

    That is a multi-GB download which can leave the GPU unusable -- on a host whose whole purpose
    here is measuring GPU throughput.
    """
    source = PILOT_SCRIPT.read_text(encoding="utf-8")
    install = source[source.index("subprocess.check_call(") :][:600]
    assert '"--no-deps"' in install, "pip must not be allowed to resolve torch on a notebook host"


def test_the_measurement_environment_is_recorded(pilot: ModuleType) -> None:
    """A throughput figure is uninterpretable without the stack that produced it."""
    source = PILOT_SCRIPT.read_text(encoding="utf-8")
    assert '"package_versions": package_versions' in source
    assert set(pilot.RECORDED_PACKAGES) >= {"torch", "transformers"}


def test_platform_is_detected_and_recorded(pilot: ModuleType) -> None:
    """A Colab run's GPU-hours are real; its percentage column is still a KAGGLE allowance."""
    assert pilot.detect_platform() in {"kaggle", "colab", "other"}
    source = PILOT_SCRIPT.read_text(encoding="utf-8")
    assert '"platform": platform' in source, "pilot_results.json must record where it ran"


def test_a_non_kaggle_run_says_the_quota_column_is_kaggles(
    pilot: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    timing = pilot.ModelTiming(
        name="solo",
        model_id="owner/solo",
        revision="0" * 40,
        load_seconds=1.0,
        quant="nf4",
        prompts_timed=50,
        total_generate_seconds=50.0,
        seconds_per_prompt=1.0,
        generated_tokens_total=100,
        generated_tokens_mean=2.0,
        generated_tokens_median=2.0,
        tokens_per_second=2.0,
        peak_vram_gb=1.0,
    )
    pilot.print_extrapolation([timing], "colab")
    printed = capsys.readouterr().out
    assert "RAN ON COLAB, NOT KAGGLE" in printed
    assert "KAGGLE allowance" in printed

    pilot.print_extrapolation([timing], "kaggle")
    assert "NOT KAGGLE" not in capsys.readouterr().out


# --------------------------------------------------------------------------------------
# The extrapolation arithmetic
# --------------------------------------------------------------------------------------


def test_scenarios_cover_the_grid_the_budget_decision_needs(pilot: ModuleType) -> None:
    keys = [scenario.key for scenario in pilot.scenarios()]
    assert keys == ["E1_greedy", "E1_sampled", "E4_sensitivity"]
    assert all(scenario.assumption for scenario in pilot.scenarios()), (
        "every scenario must state the assumption that makes it that size"
    )


def test_extrapolation_arithmetic_against_a_known_answer(
    pilot: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    """Synthetic timings with a hand-computable answer, the way the metric layer is tested.

    Two models at exactly 1 s/prompt: greedy E1 is 990 x 1 x 2 = 1,980 generation-seconds = 0.55
    GPU-h, which is 1.833% of a 30 GPU-h week.
    """
    timings = [
        pilot.ModelTiming(
            name=f"model_{index}",
            model_id=f"owner/model-{index}",
            revision="0" * 40,
            load_seconds=10.0,
            quant="nf4",
            prompts_timed=50,
            total_generate_seconds=50.0,
            seconds_per_prompt=1.0,
            generated_tokens_total=5000,
            generated_tokens_mean=100.0,
            generated_tokens_median=100.0,
            tokens_per_second=100.0,
            peak_vram_gb=5.5,
        )
        for index in range(2)
    ]
    rows = {row["key"]: row for row in pilot.print_extrapolation(timings)}

    greedy = rows["E1_greedy"]
    assert greedy["generations"] == 990 * 1 * 2
    assert greedy["gpu_hours"] == pytest.approx(1980 / 3600)
    assert greedy["percent_of_weekly_quota"] == pytest.approx(100 * (1980 / 3600) / 30)

    sampled = rows["E1_sampled"]
    assert sampled["generations"] == 990 * pilot.SAMPLED_E1_SEEDS * 2
    assert sampled["gpu_hours"] == pytest.approx(greedy["gpu_hours"] * pilot.SAMPLED_E1_SEEDS)

    subset = round(990 * pilot.E4_SUBSET_FRACTION)
    e4 = rows["E4_sensitivity"]
    assert e4["generations"] == subset * pilot.E4_TEMPERATURES * pilot.E4_SEEDS * 2

    printed = capsys.readouterr().out
    assert "GPU-h" in printed and "% of 30 GPU-h/wk" in printed
    assert "Shared assumptions" in printed, "the assumptions must be printed with the numbers"
    assert "Red-teaming (P8) is NOT included" in printed


def test_single_model_extrapolation_says_it_covers_one_model(
    pilot: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    """--models restricts the run; the report must not silently imply it covered the pair."""
    timing = pilot.ModelTiming(
        name="solo",
        model_id="owner/solo",
        revision="0" * 40,
        load_seconds=1.0,
        quant="nf4",
        prompts_timed=50,
        total_generate_seconds=50.0,
        seconds_per_prompt=1.0,
        generated_tokens_total=100,
        generated_tokens_mean=2.0,
        generated_tokens_median=2.0,
        tokens_per_second=2.0,
        peak_vram_gb=1.0,
    )
    pilot.print_extrapolation([timing])
    assert "ONLY ONE MODEL WAS TIMED" in capsys.readouterr().out
