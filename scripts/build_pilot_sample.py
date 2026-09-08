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
import tomllib
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

#: Packages the pilot must install on a notebook host, pinned to whatever ``uv.lock`` resolves for
#: ``linux_x86_64``. Kaggle's image ships no ``bitsandbytes`` at all, so the NF4 load raises
#: ``ImportError`` several minutes in, *after* the tokenizer download has started.
#:
#: The version is READ FROM THE LOCKFILE, never typed here. CLAUDE.md forbids installing unpinned
#: "to make it work"; installing the exact version the locked Linux environment resolves to is the
#: opposite of unpinning, and it is the only way to make the notebook host resemble that
#: environment. A hand-typed version here would drift from the lock silently.
#: `transformers` and its own pinned chain are here for a second reason. Kaggle's image now ships
#: **transformers 5.0.0**, a major version this project has never run: its rewritten loading path
#: reported CONVERSION errors and OOMed an 8B NF4 load on a 14.7 GB T4, where NF4 needs ~5.6 GB --
#: i.e. it was materialising weights largely unquantized. Measuring throughput on a stack we do not
#: ship would also answer the wrong question. `tokenizers`, `huggingface-hub` and `safetensors` come
#: along because transformers 4.57.6 will not import against the 1.x/0.23 versions Kaggle installs.
LOCK_PINNED_PACKAGES: tuple[str, ...] = (
    "bitsandbytes",
    "transformers",
    "tokenizers",
    "huggingface-hub",
    "safetensors",
)

#: Packages whose installed version the pilot RECORDS but does not touch. Reinstalling **torch** on
#: a notebook host is slow and can break its CUDA build, and throughput is not sensitive enough to
#: the exact torch build to justify that risk -- but the measurement is only interpretable if the
#: versions are known, so they are reported and written into pilot_results.json.
RECORDED_PACKAGES: tuple[str, ...] = ("torch", "accelerate")

REPO_ROOT = Path(__file__).resolve().parents[1]

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
3. **Run All, in a fresh session.** Roughly 20-40 minutes, most of it downloading ~30 GB.
   The cell first aligns the inference stack with `uv.lock`: Kaggle ships no `bitsandbytes` at all,
   and ships a `transformers` **major version this project does not use**, whose quantized-loading
   path OOMed an 8B NF4 load on a T4. Both are installed at their locked versions before any weight
   download. **A fresh session matters** -- Python keeps whatever it already imported, so a kernel
   that has run this cell before will keep the old `transformers` no matter what pip does. The
   script detects that and tells you to restart, instead of failing ten minutes later.
4. Bring back the printed table and the `pilot_results.json` it writes.

Out of VRAM on the 8B? An 8B at NF4 needs ~5.6 GB and a T4 has ~15, so an OOM means the weights were
not being quantized -- check the `environment:` line shows the pinned transformers. As a last
resort, `main(["--device-map", "auto"])` spreads the model across both T4s.

Tight on session time? Change the last line to
`main(["--models", "llama31_8b_instruct"])` to time one model instead of two.

**On Colab instead?** It works -- Colab is the documented debugging backup. Add `HF_TOKEN` in the
secrets panel (the key icon in the left sidebar) and enable notebook access; the script reads Kaggle
and Colab secrets both. One caveat it prints for you: the `% of 30 GPU-h/wk` column is a **Kaggle**
allowance, so on Colab read it as *"what this would cost on Kaggle"* -- which is the question the
budget decision needs anyway. The measured seconds/prompt are equally valid on the same T4, and the
GPU actually used is recorded in `pilot_results.json`.

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


def locked_versions(names: tuple[str, ...], lock_path: Path | None = None) -> dict[str, str]:
    """``{package: version}`` as resolved in ``uv.lock``.

    The pilot's pins come from here rather than from a literal, so "install bitsandbytes" on a
    notebook host means "install exactly what the locked Linux environment has" and cannot drift
    from it unnoticed.
    """
    path = lock_path or (REPO_ROOT / "uv.lock")
    lock = tomllib.loads(path.read_text(encoding="utf-8"))
    versions = {package["name"]: package["version"] for package in lock["package"]}
    missing = [name for name in names if name not in versions]
    if missing:
        raise ValueError(f"{path} has no entry for {missing}; is the inference extra still locked?")
    return {name: versions[name] for name in names}


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
        f"PINNED_PACKAGES: dict[str, str] = {_dict(locked_versions(LOCK_PINNED_PACKAGES), '')}",
        # _scalar, not repr(): repr writes single-quoted strings that `ruff format` immediately
        # rewrites, which puts the generator and the formatter in a loop.
        "RECORDED_PACKAGES: tuple[str, ...] = ("
        + ", ".join(_scalar(name) for name in RECORDED_PACKAGES)
        + ")",
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
    """One notebook cell. ``source`` is a list of lines *with* their newlines, per nbformat.

    The ``id`` is REQUIRED by nbformat 4.5+ and is derived from the cell's content rather than
    generated randomly: a random id would make every regeneration differ, and an absent one makes
    the file invalid, so any tool that opens the notebook (VS Code, Jupyter, nbformat) rewrites it
    -- which is exactly how this generator's output first drifted from what was on disk.
    """
    lines = source.splitlines(keepends=True)
    cell: dict[str, Any] = {
        "cell_type": cell_type,
        "id": hashlib.sha256(source.encode("utf-8")).hexdigest()[:8],
        "metadata": {},
        "source": lines,
    }
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
    # Replace the script's dual entrypoint with the unconditional notebook one. The script already
    # detects a kernel and does the right thing when pasted, so this is belt-and-braces -- but it
    # also puts `main([])` on the last line, where the README tells the reader to edit it.
    #
    # Sliced rather than string-replaced: a `.replace()` that stops matching after an edit to the
    # script's tail fails SILENTLY, shipping a notebook whose last cell parses the KERNEL's argv and
    # dies on `-f kernel.json`. That is exactly the bug this pilot hit in the field.
    marker = 'if __name__ == "__main__":'
    if marker not in script_text:
        raise ValueError(f"{PILOT_SCRIPT} has no {marker!r} entrypoint to replace")
    body = script_text[: script_text.index(marker)] + (
        "# Notebook entrypoint. The script detects a kernel on its own; the notebook states it\n"
        "# outright: no argv parsing, and no SystemExit (which surfaces as a traceback in a\n"
        '# kernel). Time a single model with: main(["--models", "llama31_8b_instruct"])\n'
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


def notebook_sources(notebook_json: str) -> list[tuple[str, str]]:
    """``[(cell_type, source)]`` for a notebook, ignoring ids, metadata and key order.

    The drift check compares notebooks THIS way rather than byte for byte. What has to be true is
    that the notebook carries the same instructions and the same code as the script; editors and
    nbformat legitimately rewrite ids, key order and execution counts, and a byte comparison would
    report that harmless churn as staleness while adding nothing.
    """
    parsed = json.loads(notebook_json)
    return [(cell["cell_type"], "".join(cell["source"])) for cell in parsed["cells"]]


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
        stale: list[str] = []
        if updated != current:
            stale.append("script")
        # Compared by cell content, not bytes: see notebook_sources().
        on_disk = args.notebook.read_text(encoding="utf-8") if args.notebook.is_file() else ""
        if not on_disk or notebook_sources(on_disk) != notebook_sources(notebook):
            stale.append("notebook")
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
