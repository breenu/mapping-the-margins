"""Generate the pinned data block inside ``scripts/kaggle_rho_pilot.py`` (P9 / rho measurement).

**Why the rho pilot embeds its data instead of reading it.** The same reason the timing pilot does:
Kaggle has no checkout of this repository and ``data/`` is gitignored, so
``data/prompts/prompts_v1.jsonl`` does not exist there. A hand-typed prompt list would measure a
*paraphrase* of the audited prompt set. This script renders the real P4a prompts for one
confirmatory pair and writes them into the pilot verbatim, with the model pin and the decoding
profile. ``tests/test_kaggle_rho_pilot.py`` regenerates the block and fails on any difference.

**The host plumbing is COPIED FROM THE TIMING PILOT, not rewritten.** Secret lookup on two hosts,
the locked-version preflight, the stale-import reload, the Turing-safe compute dtype and the NF4
load are all hard-won: each one encodes a failure that happened in the field. Retyping them would
lose that. They are extracted by name from ``scripts/kaggle_timing_pilot.py`` at generation time,
so there is exactly one source for them and the drift check covers both files.

**Which prompts.** The ``socioeconomic_class x race_ethnicity`` pair: 4 x 4 = 16 order-2 cells x 9
template frames = 144 prompts. Every frame of every cell, not a sample -- rho is a *within-cell,
between-item* variance share, so its precision comes from the item count per cell, and dropping
frames would be the one cut that directly widens the interval.

**Stamped ``sampled_v1``, deliberately.** ``decoding_key`` is part of the ``prompt_id`` preimage,
so the ids here are the ones the SAMPLED arm will carry, not the greedy set's. The rendered TEXT is
identical either way -- the key enters the digest, never the template -- so the measurement is of
the real prompts regardless. **``configs/prompts/open_gen_v1.yaml`` still declares
``decoding_key: greedy_v1`` and will need a versioned re-stamp before E1 runs sampled.** That is a
P4a follow-up and is deliberately NOT done here.

    uv run python scripts/build_rho_sample.py
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from margins.inference.decoding import load_decoding_config
from margins.inference.models import load_model_config
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


def _pilot_generator() -> Any:
    """Load ``scripts/build_pilot_sample.py`` by path.

    ``scripts/`` is not a package -- the pilot's own tests load it by path too -- so a plain
    ``from scripts.build_pilot_sample import ...`` would work under pytest (which puts the repo
    root on the path) and fail when this file is run directly. Loading by path works from any
    working directory and keeps ONE copy of the literal emitters, the lockfile reader and the
    notebook-cell builder rather than a second, drifting set.
    """
    import importlib.util

    path = Path(__file__).resolve().parent / "build_pilot_sample.py"
    spec = importlib.util.spec_from_file_location("build_pilot_sample", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_pilot = _pilot_generator()
LOCK_PINNED_PACKAGES: tuple[str, ...] = _pilot.LOCK_PINNED_PACKAGES
RECORDED_PACKAGES: tuple[str, ...] = _pilot.RECORDED_PACKAGES
locked_versions = _pilot.locked_versions
notebook_sources = _pilot.notebook_sources
_scalar = _pilot._scalar
_dict = _pilot._dict
_cell = _pilot._cell

#: The pair whose cells are measured. The first non-gender x non-gender confirmatory pair and the
#: largest table (4x4), which is also the pair the Stage-A power simulation used -- so the measured
#: rho and the simulated power curve describe the same design.
RHO_PAIR: tuple[str, str] = ("socioeconomic_class", "race_ethnicity")

#: Sampled generation. rho is a property of SAMPLED variation: under greedy decoding every seed of
#: a prompt returns the same string, within-item variance is exactly zero, and rho is 1 by
#: construction. Measuring it under anything but the profile E1 will use answers a different
#: question.
RHO_DECODING = "sampled_v1"

#: Default generations per prompt. 12, not the 20-30 first proposed, because the interval's width
#: is driven by the BETWEEN-item degrees of freedom -- 16 cells x 8 = 128 -- and not by seeds.
#: At a true rho of 0.2 the expected 95% interval is [0.151, 0.261] at 12 seeds and [0.159, 0.253]
#: at 24: the extra 1728 generations buy about one percentage point of half-width, and cost about
#: 6.7 GPU-hours. Overridable at run time with `--seeds`.
RHO_SEEDS = 12

#: Module-level constants the shared functions close over. Extracted too, or the copy would be a
#: ``NameError`` the moment a notebook hit the restart path -- which is the path that only fires
#: on a host that is already misbehaving, i.e. the worst place to discover a missing name.
SHARED_CONSTANTS: tuple[str, ...] = ("_RESTART_HINT",)

#: Host-handling functions lifted verbatim from the timing pilot. Order matters: this is the order
#: they are written into the generated region.
SHARED_FUNCTIONS: tuple[str, ...] = (
    "detect_platform",
    "hf_token",
    "package_version",
    "ensure_dependencies",
    "_module_name",
    "_live_version",
    "_stale_imports",
    "_purge_modules",
    "_reimport",
    "_import_diagnostics",
    "_verify_pinned_stack_is_usable",
    "require_cuda",
    "cuda_compute_dtype",
    "load_model",
    "format_prompt",
    "in_notebook",
)

REPO_ROOT = Path(__file__).resolve().parents[1]

TIMING_PILOT = Path("scripts/kaggle_timing_pilot.py")
RHO_SCRIPT = Path("scripts/kaggle_rho_pilot.py")
RHO_NOTEBOOK = Path("notebooks/kaggle_rho_pilot.ipynb")
BEGIN = "# --- BEGIN GENERATED ---"
END = "# --- END GENERATED ---"

NOTEBOOK_README = """# Kaggle rho measurement (P9)

Measures the **intraclass correlation** of harm scores within a prompt: what share of the variance
sits BETWEEN prompts rather than between repeated samples of the same prompt. The Stage-A power
analysis showed the entire feasibility of the sampled E1 arm turns on this one number, and nothing
in the project measures it yet -- the dev run has one response per prompt, so it contains no
within-prompt variance at all.

At rho = 0.05 the shipped nine template frames reach 80% power on a 0.3 SD interaction. At
rho = 0.2 they cannot reach it at **any** seed budget. So this is measured, not assumed.

**What it does:** loads `Llama-3.1-8B-Instruct` at 4-bit NF4 at its pinned revision, and generates
16 cells x 9 template frames x 12 seeds = **1,728 sampled generations** under the real
`sampled_v1` profile (temperature 0.7, top_p 0.9, 256 new tokens). It writes
`rho_responses.jsonl` and **does no scoring** -- scoring runs locally on CPU through the real P6
scorers, so the roster, the pins and the sign contract are not duplicated here.

**Before you Run All:**

1. **Settings -> Accelerator: GPU T4 x2**, and **Internet: On**.
2. **Add-ons -> Secrets:** add `HF_TOKEN` (a Hugging Face token whose account has **accepted the
   Meta Llama licence** -- `Llama-3.1-8B-Instruct` is gated) and attach it to this notebook.
3. **Run All, in a fresh session.** Expect roughly **3-6 hours**, most of it generation. The first
   cell aligns the inference stack with `uv.lock` before any weight download, for the same reasons
   the timing pilot does: Kaggle ships no `bitsandbytes`, and ships a `transformers` major version
   whose quantized-loading path OOMed an 8B NF4 load on a T4. **A fresh session matters** --
   Python keeps whatever it already imported.
4. **Download `rho_responses.jsonl` from the output panel when it finishes.**

**It is resumable.** Every batch is flushed to `rho_responses.jsonl` as it completes. If the
session dies, re-run the notebook in a new session with the file restored as an input, or simply
re-run and pass `--resume`: completed (prompt, seed) pairs are skipped. The progress line prints a
running ETA after the first batch, so you can tell within a couple of minutes whether it will fit.

**Tight on session time?** `main(["--seeds", "8"])` cuts it to 1,152 generations. That widens the
expected interval at rho = 0.2 from about [0.151, 0.261] to [0.144, 0.269] -- a real but small
loss. `main(["--seeds", "24"])` doubles the cost for roughly one percentage point of half-width,
which is why 12 is the default.

**On Colab instead?** Works the same way; the script reads Kaggle and Colab secrets both.

*Generated from `scripts/kaggle_rho_pilot.py` by `scripts/build_rho_sample.py`. Do not edit here --
edit the script and regenerate, or the two will disagree about what was measured.*
"""

_BLOCK_RE = re.compile(rf"{re.escape(BEGIN)}\n.*?\n{re.escape(END)}", re.DOTALL)


def shared_source(
    names: tuple[str, ...] = SHARED_FUNCTIONS,
    constants: tuple[str, ...] = SHARED_CONSTANTS,
    *,
    pilot: Path | None = None,
) -> str:
    """Extract named top-level constants and functions from the timing pilot, verbatim.

    Parsed with ``ast`` rather than matched with a regex so a docstring containing ``def`` cannot
    truncate a function, and so a renamed function is a loud ``KeyError`` instead of a silently
    missing one.
    """
    path = pilot or (REPO_ROOT / TIMING_PILOT)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    tree = ast.parse(text)

    spans: dict[str, tuple[int, int]] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            # `decorator_list` is empty throughout the pilot; end_lineno covers the whole body.
            spans[node.name] = (node.lineno - 1, node.end_lineno or node.lineno)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    spans[target.id] = (node.lineno - 1, node.end_lineno or node.lineno)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            spans[node.target.id] = (node.lineno - 1, node.end_lineno or node.lineno)

    wanted = (*constants, *names)
    missing = [name for name in wanted if name not in spans]
    if missing:
        raise KeyError(
            f"{path} has no top-level {missing}. The rho pilot copies its host handling from the "
            "timing pilot; a rename there must be reflected in SHARED_FUNCTIONS / "
            "SHARED_CONSTANTS here."
        )
    chunks = []
    for name in wanted:
        start, end = spans[name]
        chunks.append("\n".join(lines[start:end]).rstrip())
    return "\n\n\n".join(chunks)


def rho_prompt_records(
    *,
    lattice_name: str = DEFAULT_LATTICE,
    prompt_set: str = DEFAULT_PROMPT_SET,
    pair: tuple[str, str] = RHO_PAIR,
    decoding_name: str = RHO_DECODING,
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    """``(prompts, cells, meta)`` as real records.

    Exposed separately from :func:`rho_prompts` because ``scripts/measure_rho.py`` needs genuine
    :class:`~margins.schemas.Prompt` objects to hand the P6 scorers -- the same records the
    pipeline would pass -- while the generated block needs plain dicts it can write into a file
    that imports nothing.
    """
    lattice = DeclaredLattice(
        load_lattice_config(lattice_name, config_dir=default_lattice_config_dir())
    )
    prompt_dir = default_prompt_config_dir()
    base = load_prompt_config(prompt_set, config_dir=prompt_dir)
    # decoding_key is part of the prompt_id preimage, so stamping it here is what makes these the
    # ids the sampled arm will carry. The rendered TEXT is identical either way.
    config = base.model_copy(update={"decoding_key": decoding_name})
    files = load_template_files(config, prompt_dir / TEMPLATE_DIRNAME)
    renderer = TemplateRenderer(config, lattice)

    wanted = frozenset(pair)
    cells = [
        cell
        for cell in lattice.cells()
        if cell.order == 2 and frozenset(axis for axis, _ in cell.axes) == wanted
    ]
    if not cells:
        raise ValueError(f"the lattice declares no order-2 cells for {pair}")
    prompts = renderer.render(template_items(config, files), cells)

    axis_a, axis_b = (axis for axis, _ in cells[0].axes)
    ordered = sorted(prompts, key=lambda prompt: prompt.prompt_id)
    meta = {
        "lattice_version": lattice.config.lattice_version,
        "template_version": config.template_version,
        "axis_a": axis_a,
        "axis_b": axis_b,
        "n_cells": len(cells),
        "n_frames": len({p.prompt_id.split("::", 1)[0] for p in ordered}),
        "n_prompts": len(ordered),
        "default_seeds": RHO_SEEDS,
        "generated_by": "scripts/build_rho_sample.py",
    }
    return ordered, cells, meta


def rho_prompts(**kwargs: Any) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """The same prompts as plain dicts, which is what the generated block embeds."""
    prompts, cells, meta = rho_prompt_records(**kwargs)
    by_cell = {cell.cell_id: cell for cell in cells}
    axis_a, axis_b = meta["axis_a"], meta["axis_b"]
    rows = [
        {
            "prompt_id": prompt.prompt_id,
            "cell_id": str(prompt.cell_id),
            "value_a": by_cell[str(prompt.cell_id)].axes_map[axis_a],
            "value_b": by_cell[str(prompt.cell_id)].axes_map[axis_b],
            "template": prompt.prompt_id.split("::", 1)[0],
            "text": prompt.rendered_text,
        }
        for prompt in prompts
    ]
    return rows, meta


def render_block(**kwargs: Any) -> str:
    """Render the generated region: metadata, pins, decoding profile, prompts, host plumbing."""
    rows, meta = rho_prompts(**kwargs)
    decoding = load_decoding_config(RHO_DECODING)
    spec = load_model_config("llama31_8b_instruct")
    meta = {**meta, "decoding_key": decoding.key, "decoding_hash": decoding.decoding_hash}

    model = {
        "name": spec.name,
        "model_id": spec.model_id,
        "revision": spec.revision,
        "license": spec.license,
    }

    lines = [
        BEGIN,
        f"RHO_META: dict[str, Any] = {_dict(meta, '')}",
        f"PINNED_PACKAGES: dict[str, str] = {_dict(locked_versions(LOCK_PINNED_PACKAGES), '')}",
        "RECORDED_PACKAGES: tuple[str, ...] = ("
        + ", ".join(_scalar(name) for name in RECORDED_PACKAGES)
        + ")",
        f"MODEL: dict[str, str] = {_dict(model, '')}",
        f"DECODING: dict[str, Any] = {_dict(decoding.model_dump(mode='json'), '')}",
        "PROMPTS: tuple[dict[str, str], ...] = (",
    ]
    lines.extend(f"    {_dict(row, '    ')}," for row in rows)
    lines.extend(
        [
            ")",
            "",
            "",
            "# ---------------------------------------------------------------------------------",
            "# Host handling, copied VERBATIM from scripts/kaggle_timing_pilot.py by",
            "# scripts/build_rho_sample.py. Each of these encodes a failure that happened in the",
            "# field -- the missing bitsandbytes, the transformers major version that OOMed an NF4",
            "# load, the preimported module pip cannot replace, bfloat16 on a Turing T4. One",
            "# source, two scripts; tests/test_kaggle_rho_pilot.py fails if they diverge.",
            "# ---------------------------------------------------------------------------------",
            "",
            shared_source(),
            # Two blank lines before the marker: the last shared chunk is a function definition,
            # and without them `ruff format` would want to insert them on every run -- which would
            # make the formatter and the generator disagree forever.
            "",
            "",
            END,
        ]
    )
    return "\n".join(lines)


def splice(script_text: str, block: str) -> str:
    if not _BLOCK_RE.search(script_text):
        raise ValueError(f"{RHO_SCRIPT} has no generated region delimited by {BEGIN!r} / {END!r}")
    return _BLOCK_RE.sub(lambda _match: block, script_text, count=1)


def render_notebook(script_text: str) -> str:
    """The Kaggle notebook: a README cell plus the rho script verbatim."""
    marker = 'if __name__ == "__main__":'
    if marker not in script_text:
        raise ValueError(f"{RHO_SCRIPT} has no {marker!r} entrypoint to replace")
    body = script_text[: script_text.index(marker)] + (
        "# Notebook entrypoint. The script detects a kernel on its own; the notebook states it\n"
        "# outright: no argv parsing, and no SystemExit (which surfaces as a traceback in a\n"
        '# kernel). Fewer generations: main(["--seeds", "8"]). Resume a killed run:\n'
        '# main(["--resume"]).\n'
        "main([])\n"
    )
    notebook = {
        "cells": [_cell("markdown", NOTEBOOK_README), _cell("code", body)],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=1, ensure_ascii=False, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, default=RHO_SCRIPT)
    parser.add_argument("--notebook", type=Path, default=RHO_NOTEBOOK)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the embedded block is stale, without writing.",
    )
    args = parser.parse_args(argv)

    block = render_block()
    current = args.script.read_text(encoding="utf-8")
    updated = splice(current, block)
    notebook = render_notebook(updated)

    if args.check:
        stale: list[str] = []
        if updated != current:
            stale.append("script")
        on_disk = args.notebook.read_text(encoding="utf-8") if args.notebook.is_file() else ""
        if not on_disk or notebook_sources(on_disk) != notebook_sources(notebook):
            stale.append("notebook")
        if stale:
            print(f"STALE: {stale} do not match the current configs.")
            return 1
        print("the embedded block and the notebook are current")
        return 0

    with args.script.open("w", encoding="utf-8", newline="") as handle:
        handle.write(updated)
    args.notebook.parent.mkdir(parents=True, exist_ok=True)
    with args.notebook.open("w", encoding="utf-8", newline="") as handle:
        handle.write(notebook)

    rows, meta = rho_prompts()
    digest = hashlib.sha256(block.encode("utf-8")).hexdigest()
    print(f"wrote generated block into {args.script}")
    print(f"wrote notebook          {args.notebook}")
    print(f"  {'pair':<24} {meta['axis_a']} x {meta['axis_b']}")
    print(f"  {'cells x frames':<24} {meta['n_cells']} x {meta['n_frames']} = {len(rows)} prompts")
    print(f"  {'decoding':<24} {RHO_DECODING}")
    print(f"  {'generations at 12 seeds':<24} {len(rows) * RHO_SEEDS}")
    print(f"  {'block sha256':<24} {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
