"""Generate the pinned data block inside ``scripts/kaggle_timing_pilot.py`` (P5 / W2).

**Why the pilot embeds its data instead of reading it.** Kaggle has no checkout of this repository,
and ``data/`` is gitignored, so ``data/prompts/prompts_v1.jsonl`` does not exist there. The pilot
must be a single file that runs on its own. But a hand-typed prompt list would be a *paraphrase* of
the audited prompt set, timed instead of the real thing -- so this script renders the real P4a
prompts, samples them the way the P4a slot audit does, and writes them into the pilot verbatim,
along with the model pins and the decoding profile.

``tests/test_kaggle_pilot.py`` regenerates the block and fails on any difference, so the embedded
copy cannot drift from ``configs/model/``, ``configs/decoding/`` or the rendered prompt set.

The sample reuses :func:`margins.prompts.audit.sample_for_fill_audit`: already deterministic,
already stratified across every declared axis pair *and* every template, and RNG-free. A
head-of-file sample would time one frame and one axis pair, and generation cost varies with prompt
length -- so stratification is a measurement concern here, not only an audit one.

    uv run python scripts/build_pilot_sample.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from margins.inference.decoding import DEFAULT_DECODING, load_decoding_config
from margins.inference.models import CONFIRMATORY_MODELS, load_model_config
from margins.lattice import DEFAULT_LATTICE, DeclaredLattice, load_lattice_config
from margins.lattice.config import default_lattice_config_dir
from margins.prompts import (
    DEFAULT_PROMPT_SET,
    TemplateRenderer,
    default_prompt_config_dir,
    load_prompt_config,
    load_template_files,
    sample_for_fill_audit,
    template_items,
)
from margins.prompts.config import TEMPLATE_DIRNAME

#: Gate-relevant size: "~50 prompts spanning cell groups".
PILOT_SAMPLE_SIZE = 50

PILOT_SCRIPT = Path("scripts/kaggle_timing_pilot.py")
PILOT_NOTEBOOK = Path("notebooks/kaggle_timing_pilot.ipynb")
BEGIN = "# --- BEGIN GENERATED ---"
END = "# --- END GENERATED ---"

#: The markdown cell Kaggle shows first. Setup steps only -- the rationale lives in the script.
NOTEBOOK_README = """# Kaggle timing pilot (P5 / W2)

Measures real generation throughput for the two confirmatory checkpoints at 4-bit NF4, then
extrapolates the E1 budget in GPU-hours and as a share of Kaggle's ~30 GPU-h/week.

**Before you Run All:**

1. **Settings -> Accelerator: GPU T4 x2**, and **Internet: On**.
2. **Add-ons -> Secrets:** add `HF_TOKEN` (a Hugging Face token whose account has **accepted the
   Meta Llama licence** -- `Llama-3.1-8B-Instruct` is gated) and attach it to this notebook.
3. **Run All.** Roughly 20-40 minutes, most of it downloading ~30 GB of weights.
4. Bring back the printed table and the `pilot_results.json` it writes.

Tight on session time? Change the last line to
`main(["--models", "llama31_8b_instruct"])` to time one model instead of two.

*Generated from `scripts/kaggle_timing_pilot.py` by `scripts/build_pilot_sample.py`. Do not edit
here -- edit the script and regenerate, or the two will disagree about what was measured.*
"""

_BLOCK_RE = re.compile(
    rf"{re.escape(BEGIN)}\n.*?\n{re.escape(END)}",
    re.DOTALL,
)


def _scalar(value: Any) -> str:
    """A **Python** literal for one scalar, in the style ``ruff format`` would leave alone.

    Neither ``json.dumps`` nor ``repr`` alone will do. JSON writes ``false``/``null``/``true``,
    which are not Python and would make the generated block a syntax error the moment a decoding
    profile contained a boolean. ``repr`` writes single-quoted strings, which the formatter would
    immediately rewrite to double quotes -- and then ``ruff format --check`` would fail on a file
    that had just been generated.
    """
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int | float):
        return repr(value)
    raise TypeError(f"cannot emit {type(value).__name__} into the generated block: {value!r}")


def _dict(value: dict[str, Any], indent: str) -> str:
    """An exploded dict literal with a magic trailing comma.

    Exploded on purpose: the trailing comma pins the multi-line form, so ``ruff format`` leaves the
    generated block exactly as written. A single-line dict would be reformatted on the next format
    run, and the drift check would then report a file it had just generated as stale.
    """
    inner = indent + "    "
    lines = [f"{inner}{_scalar(key)}: {_scalar(value[key])}," for key in sorted(value)]
    return "{\n" + "\n".join(lines) + f"\n{indent}}}"


def render_block(
    *,
    lattice_name: str = DEFAULT_LATTICE,
    prompt_set: str = DEFAULT_PROMPT_SET,
    decoding_name: str = DEFAULT_DECODING,
    size: int = PILOT_SAMPLE_SIZE,
) -> str:
    """Render the generated region: metadata, model pins, decoding profile, prompt sample."""
    lattice = DeclaredLattice(
        load_lattice_config(lattice_name, config_dir=default_lattice_config_dir())
    )
    prompt_dir = default_prompt_config_dir()
    prompt_config = load_prompt_config(prompt_set, config_dir=prompt_dir)
    files = load_template_files(prompt_config, prompt_dir / TEMPLATE_DIRNAME)
    renderer = TemplateRenderer(prompt_config, lattice)
    prompts = renderer.render(template_items(prompt_config, files), lattice.cells())

    sample = sample_for_fill_audit(prompts, lattice, size)
    decoding = load_decoding_config(decoding_name)

    models = [
        {
            "name": spec.name,
            "model_id": spec.model_id,
            "revision": spec.revision,
            "license": spec.license,
        }
        for spec in (load_model_config(name) for name in CONFIRMATORY_MODELS)
    ]

    meta = {
        "lattice_version": lattice.config.lattice_version,
        "template_version": prompt_config.template_version,
        "decoding_key": decoding.key,
        "decoding_hash": decoding.decoding_hash,
        "prompt_set_size": len(prompts),
        "sample_size": len(sample),
        "generated_by": "scripts/build_pilot_sample.py",
    }

    lines = [
        BEGIN,
        f"PILOT_META: dict[str, Any] = {_dict(meta, '')}",
        "MODELS: tuple[dict[str, str], ...] = (",
    ]
    lines.extend(f"    {_dict(entry, '    ')}," for entry in models)
    lines.extend(
        [
            ")",
            f"DECODING: dict[str, Any] = {_dict(decoding.model_dump(mode='json'), '')}",
            "PROMPT_SAMPLE: tuple[tuple[str, str], ...] = (",
        ]
    )
    for prompt in sample:
        lines.extend(
            [
                "    (",
                f"        {_scalar(prompt.prompt_id)},",
                f"        {_scalar(prompt.rendered_text)},",
                "    ),",
            ]
        )
    lines.extend([")", END])
    return "\n".join(lines)


def splice(script_text: str, block: str) -> str:
    """Replace the generated region of the pilot script."""
    if not _BLOCK_RE.search(script_text):
        raise ValueError(f"{PILOT_SCRIPT} has no generated region delimited by {BEGIN!r} / {END!r}")
    return _BLOCK_RE.sub(lambda _match: block, script_text, count=1)


def _cell(cell_type: str, source: str) -> dict[str, Any]:
    """One notebook cell. ``source`` is a list of lines *with* their newlines, per nbformat."""
    lines = source.splitlines(keepends=True)
    cell: dict[str, Any] = {"cell_type": cell_type, "metadata": {}, "source": lines}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def render_notebook(script_text: str) -> str:
    """The Kaggle notebook: a README cell plus the pilot script verbatim.

    The script is the single source of truth and the notebook is a wrapper around it, so there is
    exactly one copy of the measurement logic. Nothing imports from ``notebooks/`` and nothing here
    writes to ``results/``, so the no-load-bearing-notebooks guardrail holds.
    """
    # `raise SystemExit(main())` would kill the kernel on Kaggle; in a notebook we just call it.
    body = script_text.replace(
        'if __name__ == "__main__":\n    raise SystemExit(main())\n',
        "# Notebook entrypoint. `raise SystemExit` would kill the Kaggle kernel, so call main()\n"
        '# directly. Pass e.g. ["--models", "llama31_8b_instruct"] to time a single model.\n'
        "main([])\n",
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
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, default=PILOT_SCRIPT)
    parser.add_argument("--notebook", type=Path, default=PILOT_NOTEBOOK)
    parser.add_argument("--size", type=int, default=PILOT_SAMPLE_SIZE)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the embedded block is stale, without writing.",
    )
    args = parser.parse_args(argv)

    block = render_block(size=args.size)
    current = args.script.read_text(encoding="utf-8")
    updated = splice(current, block)
    notebook = render_notebook(updated)

    if args.check:
        stale = [
            name
            for name, expected, actual in (
                ("script", updated, current),
                (
                    "notebook",
                    notebook,
                    args.notebook.read_text(encoding="utf-8") if args.notebook.is_file() else "",
                ),
            )
            if expected != actual
        ]
        if stale:
            print(f"STALE: {stale} do not match the current configs.")
            return 1
        print("the embedded block and the notebook are current")
        return 0

    # newline="" keeps LF endings, matching .gitattributes, so the file's digest does not depend on
    # which of the two locked environments wrote it.
    with args.script.open("w", encoding="utf-8", newline="") as handle:
        handle.write(updated)
    args.notebook.parent.mkdir(parents=True, exist_ok=True)
    with args.notebook.open("w", encoding="utf-8", newline="") as handle:
        handle.write(notebook)

    digest = hashlib.sha256(block.encode("utf-8")).hexdigest()
    sample_lines = block.count("\n    (")
    print(f"wrote generated block into {args.script}")
    print(f"wrote notebook          {args.notebook}")
    print(f"  {'prompts sampled':<24} {sample_lines}")
    print(f"  {'block sha256':<24} {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
