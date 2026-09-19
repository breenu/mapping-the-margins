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


def test_the_pilot_is_frozen() -> None:
    """Gate: the W2 timing pilot is a FROZEN RECORD and must not be regenerated.

    **This pilot has been RUN.** Its numbers are the record of that run, taken against prompt set
    ``open_gen_v1.1``. On 19 Sep 2026 the prompt set moved to ``open_gen_v1.2`` -- five new frames
    and a second carrier, forced by the P9 power analysis -- which changes every ``prompt_id`` and
    the set size. Regenerating this block would therefore make the file describe a corpus the run
    never saw, and would quietly invalidate the measurement it exists to record.

    So the drift guard INVERTS at this point in the project's life. Before the run it asserted
    "matches the live config"; after it, it asserts "has not changed at all". Re-running this
    pilot means checking out the revision it was generated at, not regenerating it here.
    """
    import hashlib

    digest = hashlib.sha256(generated_block(PILOT_SCRIPT).encode("utf-8")).hexdigest()
    assert digest == FROZEN_BLOCK_SHA256, (
        "the pilot's embedded data changed. This pilot is a FROZEN RECORD of a completed run "
        "against open_gen_v1.1; it must not be regenerated against the current prompt set. If "
        "the change was deliberate, the measurement it records is no longer the measurement the "
        "file describes."
    )


def test_embedded_prompts_are_shaped_like_real_rendered_prompts(pilot: ModuleType) -> None:
    """The prompts are real P4a renderings, checked structurally now that the set has moved on.

    Before ``open_gen_v1.2`` this compared every embedded prompt against the LIVE prompt set. It
    cannot any more: the set was regenerated with five new frames and a second carrier, so every
    id changed. What survives, and is still worth asserting, is that these are renderings of the
    real frames against real lattice cells rather than hand-typed paraphrases -- the id grammar
    carries the template key, the cell id and the carrier, and all three must resolve.
    """
    from margins.lattice import DEFAULT_LATTICE, DeclaredLattice, load_lattice_config

    lattice = DeclaredLattice(load_lattice_config(DEFAULT_LATTICE))
    cell_ids = {cell.cell_id for cell in lattice.cells()}

    assert pilot.PROMPT_SAMPLE, "the generated block is empty"
    for prompt_id, text in pilot.PROMPT_SAMPLE:
        template_key, cell_id, carrier, digest = prompt_id.split("::")
        assert template_key, prompt_id
        assert cell_id in cell_ids, f"{cell_id!r} is not a cell of the lattice"
        assert carrier == "person", "the frozen run used the single shipped carrier"
        assert len(digest) == 12, prompt_id
        assert text.strip() == text and text, prompt_id


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
    monkeypatch.setattr(pilot, "_verify_pinned_stack_is_usable", lambda: None)
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
    pilot._verify_pinned_stack_is_usable()
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
        pilot._verify_pinned_stack_is_usable()


def test_the_whole_inference_stack_is_pinned_not_just_bitsandbytes(pilot: ModuleType) -> None:
    """Kaggle ships transformers 5.x; this project locks 4.57.6.

    Measuring throughput on a major version we do not ship answers the wrong question -- and in
    practice its rewritten quantized-loading path OOMed an 8B NF4 load on a T4, where NF4 needs
    ~5.6 GB of 15. tokenizers/huggingface-hub/safetensors ride along because transformers will not
    import against the versions Kaggle pairs with 5.x.
    """
    for package in ("bitsandbytes", "transformers", "tokenizers", "huggingface-hub", "safetensors"):
        assert package in pilot.PINNED_PACKAGES, f"{package} must be pinned for the pilot"
    assert pilot.PINNED_PACKAGES["transformers"].startswith("4."), (
        "the pilot must run the transformers major version this project locks"
    )
    # torch is deliberately NOT pinned: replacing a host's CUDA-matched build is the bigger risk.
    assert "torch" not in pilot.PINNED_PACKAGES
    assert "torch" in pilot.RECORDED_PACKAGES


def test_a_stale_import_of_a_pinned_package_is_reloaded_in_place(
    pilot: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """pip replacing a module on disk does not change what Python has already imported.

    Two hosts hit this. A kernel that ran this cell once already holds Kaggle's transformers 5.x;
    and Colab imports ``huggingface_hub`` at kernel STARTUP, so the pin is stale before the cell
    runs at all. The second one is why a restart is not the first answer -- the host preimports it
    again on the next session, which reported as "I restarted many times and it still fails".

    Reloading happens before any weight download, so getting it wrong costs seconds, and the version
    is re-checked afterwards rather than assumed.
    """
    stale = ModuleType("transformers")
    stale.__version__ = "5.0.0"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", stale)
    pinned = pilot.PINNED_PACKAGES["transformers"]

    def reloaded(package: str) -> str:
        assert package == "transformers", f"reloaded the wrong package: {package}"
        assert "transformers" not in sys.modules, "the stale module must be dropped first"
        _fake_transformers(monkeypatch, available=True)
        sys.modules["transformers"].__version__ = pinned  # type: ignore[attr-defined]
        return str(pinned)

    monkeypatch.setattr(pilot, "_reimport", reloaded)
    pilot._verify_pinned_stack_is_usable()

    printed = capsys.readouterr().out
    assert "stale in this kernel: transformers 5.0.0" in printed
    assert "reloaded" in printed
    assert "can see bitsandbytes" in printed


def test_a_stale_import_that_will_not_reload_demands_a_restart_and_says_where_it_came_from(
    pilot: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reload that does not take means pip's install is being shadowed on sys.path.

    "Restart the session" alone is the wrong advice there -- it has already been tried. The message
    has to name the paths, because the paths are the only thing that distinguishes a stale import
    (restarting fixes it) from a second copy earlier on sys.path (restarting never will).
    """
    stale = ModuleType("bitsandbytes")
    stale.__version__ = "0.99.0"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bitsandbytes", stale)
    monkeypatch.setattr(pilot, "_reimport", lambda package: "0.99.0")

    with pytest.raises(SystemExit) as raised:
        pilot._verify_pinned_stack_is_usable()

    message = str(raised.value)
    assert "bitsandbytes: 0.99.0" in message
    assert pilot.PINNED_PACKAGES["bitsandbytes"] in message
    assert "copy(ies) visible" in message, "the failure must name where the package comes from"
    assert "sys.path" in message
    assert "Restart the session" in message


def test_the_pilot_picks_its_compute_dtype_from_the_device(pilot: ModuleType) -> None:
    """Mirrors margins.inference.backend: a T4 is Turing (7.5) and has no bfloat16."""
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    for capability, expected in (((7, 5), "float16"), ((8, 0), "bfloat16")):
        fake = SimpleNamespace(
            cuda=SimpleNamespace(get_device_capability=lambda c=capability: c),
            bfloat16=torch.bfloat16,
            float16=torch.float16,
        )
        chosen = pilot.cuda_compute_dtype(fake)
        assert str(chosen).removeprefix("torch.") == expected

    source = PILOT_SCRIPT.read_text(encoding="utf-8")
    assert "bnb_4bit_compute_dtype=dtype" in source
    assert "bnb_4bit_compute_dtype=torch.bfloat16" not in source


def test_the_install_does_not_let_pip_touch_torch(pilot: ModuleType) -> None:
    """Resolving bitsandbytes' torch dependency could swap the host's CUDA build for a generic one.

    That is a multi-GB download which can leave the GPU unusable -- on a host whose whole purpose
    here is measuring GPU throughput.
    """
    source = PILOT_SCRIPT.read_text(encoding="utf-8")
    install = source[source.index("subprocess.check_call(") :][:1600]
    assert '"--no-deps"' in install, "pip must not be allowed to resolve torch on a notebook host"


def test_the_measurement_environment_is_recorded(pilot: ModuleType) -> None:
    """A throughput figure is uninterpretable without the stack that produced it."""
    source = PILOT_SCRIPT.read_text(encoding="utf-8")
    assert '"package_versions": package_versions' in source
    # transformers is PINNED rather than merely recorded; both sets land in the JSON.
    reported = set(pilot.RECORDED_PACKAGES) | set(pilot.PINNED_PACKAGES)
    assert reported >= {"torch", "transformers", "bitsandbytes"}


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


#: The generated block, frozen at the state the pilot was RUN against. See
#: ``test_the_pilot_is_frozen``.
FROZEN_BLOCK_SHA256 = "e90bf57230718964ac0eab333cfac9faf56a70dabc985515a5bb33b0a681892f"


def generated_block(path: Path) -> str:
    """The ``# --- BEGIN/END GENERATED ---`` region of a pilot script."""
    import re

    text = path.read_text(encoding="utf-8")
    pattern = r"# --- BEGIN GENERATED ---\n.*?\n# --- END GENERATED ---"
    match = re.search(pattern, text, re.DOTALL)
    assert match is not None, f"{path} has no generated region"
    return match.group(0)
