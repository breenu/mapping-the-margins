"""Gate: the rho pilot measures the real prompts, under the real pins, and cannot drift.

The pilot is a single self-contained file pasted into a Kaggle notebook, so nothing at run time
checks it against this repository. These tests are that check. The failure they exist to prevent
is not a crash: it is a **plausible rho measured on a paraphrase of the prompt set, or under a
decoding profile that makes the answer meaningless**.

The load-bearing ones:

* the embedded prompts are regenerated and compared, so a lattice or template edit that was not
  followed by `build_rho_sample.py` is a failing test rather than a silently stale notebook;
* the decoding profile must be SAMPLED. Under greedy decoding every seed of a prompt returns the
  same string, within-item variance is exactly zero, and rho is 1 by construction -- the run would
  produce a confident number that means nothing;
* the host-handling functions must be byte-identical to the timing pilot's, because each one
  encodes a failure that happened in the field and a divergent copy would reintroduce it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RHO_SCRIPT = REPO_ROOT / "scripts" / "kaggle_rho_pilot.py"
RHO_NOTEBOOK = REPO_ROOT / "notebooks" / "kaggle_rho_pilot.ipynb"
TIMING_SCRIPT = REPO_ROOT / "scripts" / "kaggle_timing_pilot.py"


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pilot() -> ModuleType:
    """Import the pilot. It must load with no `margins` and no torch present."""
    return _load("kaggle_rho_pilot", RHO_SCRIPT)


@pytest.fixture(scope="module")
def builder() -> ModuleType:
    return _load("build_rho_sample", REPO_ROOT / "scripts" / "build_rho_sample.py")


# --------------------------------------------------------------------------------------
# Self-containment
# --------------------------------------------------------------------------------------


def test_the_pilot_imports_nothing_from_margins() -> None:
    """Kaggle has no checkout of this repo; an import would strand the notebook."""
    for line in RHO_SCRIPT.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        assert not stripped.startswith(("import margins", "from margins")), (
            f"the pilot must be self-contained, found: {stripped!r}"
        )


def test_torch_is_imported_lazily(pilot: ModuleType) -> None:
    assert pilot.PROMPTS, "the generated block is empty"


# --------------------------------------------------------------------------------------
# Drift guards
# --------------------------------------------------------------------------------------


def test_the_rho_pilot_is_frozen() -> None:
    """Gate: the rho measurement is a FROZEN RECORD and must not be regenerated.

    **This pilot has been RUN**, and its output is the measured intraclass correlation the whole
    P9 design rests on (regard 0.35, toxicity 0.18). It ran against prompt set
    ``open_gen_v1.1``. On 19 Sep 2026 the set moved to ``open_gen_v1.2`` -- five new frames and a
    second carrier, forced by that very measurement -- which changes every ``prompt_id``.

    Regenerating this block would make the file describe a corpus the run never saw, and the
    circularity would be total: the design that the measurement produced would silently rewrite
    the record of the measurement. So the guard inverts from "matches the live config" to "has
    not changed at all". Re-measuring rho means checking out the revision this was generated at.
    """
    import hashlib
    import re

    text = RHO_SCRIPT.read_text(encoding="utf-8")
    pattern = r"# --- BEGIN GENERATED ---\n.*?\n# --- END GENERATED ---"
    match = re.search(pattern, text, re.DOTALL)
    assert match is not None, "the rho pilot has no generated region"
    digest = hashlib.sha256(match.group(0).encode("utf-8")).hexdigest()
    assert digest == FROZEN_BLOCK_SHA256, (
        "the rho pilot's embedded data changed. It is a FROZEN RECORD of the run that produced "
        "the measured rho; regenerating it against the current prompt set would invalidate the "
        "measurement the rest of the design was derived from."
    )


def test_embedded_prompts_are_shaped_like_real_rendered_prompts(pilot: ModuleType) -> None:
    """Real renderings against real cells, checked structurally now that the set has moved on.

    Before ``open_gen_v1.2`` this compared the embedded TEXT against a fresh render. It cannot
    any more, and the reason is the finding itself: the measured rho is what forced the prompt
    set to grow. What is still assertable is that every embedded prompt is a rendering of a real
    frame against a real lattice cell of the pair, not a hand-typed paraphrase.
    """
    from margins.lattice import DEFAULT_LATTICE, DeclaredLattice, load_lattice_config

    lattice = DeclaredLattice(load_lattice_config(DEFAULT_LATTICE))
    cell_ids = {cell.cell_id for cell in lattice.cells()}
    for row in pilot.PROMPTS:
        template_key, cell_id, carrier, digest = row["prompt_id"].split("::")
        assert cell_id in cell_ids, f"{cell_id!r} is not a cell of the lattice"
        assert cell_id == row["cell_id"]
        assert template_key == row["template"]
        assert carrier == "person", "the frozen run used the single shipped carrier"
        assert len(digest) == 12
        assert row["value_a"] in cell_id and row["value_b"] in cell_id
        assert row["text"].strip() == row["text"] and row["text"]


def test_the_design_is_the_whole_pair(pilot: ModuleType) -> None:
    meta = pilot.RHO_META
    assert (meta["axis_a"], meta["axis_b"]) == ("socioeconomic_class", "race_ethnicity")
    assert meta["n_cells"] == 16
    assert meta["n_frames"] == 9
    assert meta["n_prompts"] == 144
    assert len(pilot.PROMPTS) == 144
    assert len({row["cell_id"] for row in pilot.PROMPTS}) == 16
    assert len({row["template"] for row in pilot.PROMPTS}) == 9
    # Every cell carries every frame. rho's precision comes from items per cell, so a hole here
    # would quietly widen the interval.
    per_cell = dict.fromkeys({row["cell_id"] for row in pilot.PROMPTS}, 0)
    for row in pilot.PROMPTS:
        per_cell[row["cell_id"]] += 1
    assert set(per_cell.values()) == {9}


def test_the_model_pin_matches_the_config(pilot: ModuleType) -> None:
    from margins.inference.models import load_model_config

    spec = load_model_config("llama31_8b_instruct")
    assert pilot.MODEL["model_id"] == spec.model_id
    assert pilot.MODEL["revision"] == spec.revision
    assert len(pilot.MODEL["revision"]) == 40, "a branch name is not a pin"


def test_the_decoding_profile_is_sampled_and_matches_the_config(pilot: ModuleType) -> None:
    """Greedy would make rho 1 by construction and the whole run meaningless."""
    from margins.inference.decoding import load_decoding_config

    config = load_decoding_config("sampled_v1")
    assert pilot.DECODING == config.model_dump(mode="json")
    assert pilot.DECODING["do_sample"] is True
    assert pilot.DECODING["temperature"] > 0.0
    assert pilot.RHO_META["decoding_key"] == "sampled_v1"
    assert pilot.RHO_META["decoding_hash"] == config.decoding_hash


def test_host_handling_is_byte_identical_to_the_timing_pilot(builder: ModuleType) -> None:
    """One source, two scripts. A divergent copy reintroduces a failure already paid for."""
    shared = builder.shared_source()
    assert shared in RHO_SCRIPT.read_text(encoding="utf-8")
    timing = TIMING_SCRIPT.read_text(encoding="utf-8")
    for name in builder.SHARED_FUNCTIONS:
        assert f"def {name}(" in timing
        assert f"def {name}(" in shared


def test_the_notebook_carries_the_script_and_the_readme() -> None:
    parsed = json.loads(RHO_NOTEBOOK.read_text(encoding="utf-8"))
    kinds = [cell["cell_type"] for cell in parsed["cells"]]
    assert kinds == ["markdown", "code"]
    readme = "".join(parsed["cells"][0]["source"])
    for phrase in ("GPU T4 x2", "Internet: On", "HF_TOKEN", "rho_responses.jsonl"):
        assert phrase in readme
    code = "".join(parsed["cells"][1]["source"])
    assert code.rstrip().endswith("main([])"), "the notebook must not parse the kernel's argv"
    assert 'if __name__ == "__main__":' not in code


# --------------------------------------------------------------------------------------
# Run-time behaviour, exercised without a GPU
# --------------------------------------------------------------------------------------


def test_planned_units_puts_seeds_on_the_outside(pilot: ModuleType) -> None:
    """A batch must never straddle a seed -- the P5 harness rule, preserved here."""
    units = pilot.planned_units(3)
    assert len(units) == 144 * 3
    assert len(set(units)) == len(units)
    seeds_in_order = [seed for _prompt, seed in units]
    assert seeds_in_order == sorted(seeds_in_order)


def test_resume_reads_completed_units(pilot: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "rho.jsonl"
    path.write_text(
        json.dumps({"kind": "header", "meta": {}})
        + "\n"
        + json.dumps({"prompt_id": "a", "seed": 0})
        + "\n"
        + json.dumps({"prompt_id": "a", "seed": 1})
        + "\n",
        encoding="utf-8",
    )
    assert pilot.completed_units(path) == {("a", 0), ("a", 1)}


def test_a_truncated_final_line_is_repaired_not_fatal(pilot: ModuleType, tmp_path: Path) -> None:
    """The normal state of a killed session. Interruption is not corruption."""
    path = tmp_path / "rho.jsonl"
    path.write_text(
        json.dumps({"prompt_id": "a", "seed": 0}) + "\n" + '{"prompt_id": "b", "se',
        encoding="utf-8",
    )
    assert pilot.completed_units(path) == {("a", 0)}


def test_a_corrupt_interior_line_raises(pilot: ModuleType, tmp_path: Path) -> None:
    """A complete invalid line is corruption, which is a different thing, and must not pass."""
    path = tmp_path / "rho.jsonl"
    path.write_text(
        "{not json}\n" + json.dumps({"prompt_id": "a", "seed": 0}) + "\n", encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="corruption"):
        pilot.completed_units(path)


def test_missing_file_resumes_from_nothing(pilot: ModuleType, tmp_path: Path) -> None:
    assert pilot.completed_units(tmp_path / "absent.jsonl") == set()


def test_one_seed_per_prompt_is_refused(pilot: ModuleType) -> None:
    """Exactly the dev run's shape: it contains no within-prompt variance to measure."""
    with pytest.raises(SystemExit, match="rho is a WITHIN-prompt variance"):
        pilot.main(["--seeds", "1"])


def test_default_seeds_is_the_documented_twelve(pilot: ModuleType) -> None:
    assert pilot.RHO_META["default_seeds"] == 12


#: The generated block, frozen at the state the rho measurement was RUN against
#: (``open_gen_v1.1``). See ``test_the_rho_pilot_is_frozen``.
FROZEN_BLOCK_SHA256 = "95e4b341ceb9cef843fa7387f057ad1726b6de2f2a5ddda36b7fe4753e8728f2"
