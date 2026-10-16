"""Generate the Kaggle R2 notebook from the P11 plan, so it cannot drift from what the lab runs.

    uv run python scripts/build_r2_notebook.py            # write notebooks/r2_kaggle.ipynb
    uv run python scripts/build_r2_notebook.py --check    # fail if the committed one is stale

**What the notebook is.** R2's primary evidence (pre-registration §10, 16 Oct 2026): the E4 subset
regenerated on Kaggle's free-tier T4s, so the paper can say "reproducible on free-tier commodity
hardware" as a demonstrated fact rather than an argument from the budget. It is a THIN driver: it
clones the repository at a pinned tag, installs the LOCKED environment, and calls the same
``run_p11.py`` steps the lab runs. It measures nothing itself, and nothing imports from it.

**Why generated.** The commands are rendered from :mod:`margins.orchestrate.plan`, the one place a
P11 argv is written, and ``tests/test_r2_analysis.py`` parses every one against the real parsers,
exactly as it does for the runbook. A hand-edited notebook would be the one copy nobody checks.

**The two T4s.** Kaggle gives two GPUs; the notebook runs one model per GPU, in parallel, which
halves the wall time. That is why the prompt files are built ONCE before the two runs start and
each run passes ``--no-prompt-build``: two processes rewriting one file is a race.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from margins.orchestrate.config import load_p11_config
from margins.orchestrate.plan import r2_arm_dir, r2_prompt_builds

NOTEBOOK = Path("notebooks/r2_kaggle.ipynb")
#: The ref the notebook checks out. A TAG, created on the commit that carries this notebook (a
#: commit cannot name its own hash). Every R2 manifest records the resolved commit regardless.
R2_REF = "r2-v1"
REPO = "github.com/breenu/mapping-the-margins.git"
MACHINE = "kaggle"

README = """\
# Mapping the Margins: R2 reproducibility rerun (Kaggle, free tier)

Regenerates the **E4 subset** (class x race, 432 prompts) through **both** confirmatory models on
Kaggle's free-tier T4s, as pre-registered on 16 Oct 2026 (`docs/preregistration.md` §10). It runs
the repository's own pipeline at the pinned tag `{ref}`; nothing is measured here.

**Before running**
1. *Settings -> Accelerator -> GPU T4 x2*, and *Internet -> On* (needs a phone-verified account).
2. *Add-ons -> Secrets*: attach **`HF_TOKEN`** (read; the account must have accepted the Llama 3.1
   licence) and **`GITHUB_TOKEN`** (a fine-grained, **read-only** token for
   `breenu/mapping-the-margins`). The token is never printed and is removed from the clone's remote.
3. Run with **Save Version -> Save & Run All (Commit)**, so it keeps running with the tab closed.
   Expect ~1 h of downloads and setup, then roughly 3-5 h of generation (both models at once).

**Output**: `r2_kaggle_outputs.tar.gz` in the notebook's *Output* tab, plus one log per model.
Download it and hand it back; `run_p11.py r2-process --machine kaggle` analyses it on any CPU.

If a step fails, the notebook stops with the reason. Do not edit the cells to work around it:
never unpin, never `pip install`, never swap a model. Report the last ~50 lines of the cell.
"""

SETUP = """\
import os, subprocess, time
from pathlib import Path

REPO = {repo!r}
REF = {ref!r}
MODELS = {models!r}
WORK = Path("/tmp/mapping-the-margins")
OUT = Path("/kaggle/working")


def sh(command, cwd=None):
    print(f"\\n$ {{command}}", flush=True)
    subprocess.run(command, shell=True, check=True, cwd=cwd)


try:
    from kaggle_secrets import UserSecretsClient

    _secrets = UserSecretsClient()
    os.environ["HF_TOKEN"] = _secrets.get_secret("HF_TOKEN")
    _github_token = _secrets.get_secret("GITHUB_TOKEN")
except Exception as error:
    message = "Attach the HF_TOKEN and GITHUB_TOKEN secrets (Add-ons -> Secrets)."
    raise SystemExit(message) from error

if not WORK.exists():
    # The token goes in the clone URL only, is never echoed, and is stripped from the remote.
    subprocess.run(
        [
            "git", "clone", "--quiet",
            f"https://x-access-token:{{_github_token}}@{{REPO}}", str(WORK),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(WORK), "remote", "set-url", "origin", f"https://{{REPO}}"], check=True
    )
sh(f"git checkout --quiet {{REF}}", cwd=WORK)
sh("git log -1 --format='%H %s'", cwd=WORK)
"""

ENVIRONMENT = """\
# The LOCKED environment, exactly as the lab installed it. `--locked` fails rather than resolve.
sh("curl -LsSf https://astral.sh/uv/install.sh | sh")
os.environ["PATH"] = f"{Path.home()}/.local/bin:" + os.environ["PATH"]
sh("uv sync --locked --extra inference", cwd=WORK)
sh("nvidia-smi")
"""

VERIFY = """\
# Both models must load at NF4 on a T4 and generate, or nothing later is the measurement.
# (First run downloads ~30 GB of weights.)
for model in MODELS:
    sh(
        f"CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_p11.py verify-gpu --model {model}",
        cwd=WORK,
    )
"""

PROMPTS = """\
# R2's prompt files, built ONCE here: the two parallel runs below must not both rewrite them.
for command in {commands!r}:
    sh(command, cwd=WORK)
"""

GENERATE = """\
# One model per T4, in parallel. Each run is the lab's own `r2` step (resumable, preflighted).
running = {{}}
for gpu, model in enumerate(MODELS):
    log = open(OUT / f"r2_{{model}}.log", "w")
    command = (
        f"CUDA_VISIBLE_DEVICES={{gpu}} uv run python scripts/run_p11.py r2 --model {{model}} "
        f"--machine {machine} --no-prompt-build"
    )
    print(f"$ {{command}}  > {{log.name}}", flush=True)
    process = subprocess.Popen(
        command, shell=True, cwd=WORK, stdout=log, stderr=subprocess.STDOUT
    )
    running[model] = (process, log)

responses = WORK / "data" / "responses"
while any(process.poll() is None for process, _ in running.values()):
    time.sleep(300)
    counts = {{
        f"{{path.parent.parent.name}}/{{path.parent.name}}": len(
            path.read_text(encoding="utf-8").splitlines()
        )
        for path in sorted(responses.glob("r2_{machine}_*/*/*.jsonl"))
    }}
    print(time.strftime("%H:%M"), counts, flush=True)

for model, (process, log) in running.items():
    log.close()
    tail = Path(log.name).read_text(encoding="utf-8").splitlines()[-25:]
    print(f"\\n==== {{model}}: exit {{process.returncode}} ====", *tail, sep="\\n")
failed = [model for model, (process, _) in running.items() if process.returncode]
if failed:
    raise SystemExit(f"R2 generation FAILED for {{failed}}; see the logs above. Report them.")
"""

PACK = """\
sh(
    "tar czf /kaggle/working/r2_kaggle_outputs.tar.gz "
    "{sampled} {greedy} data/p11 results/run_manifests",
    cwd=WORK,
)
sh("ls -lh /kaggle/working")
"""


def _cell(cell_type: str, source: str) -> dict[str, Any]:
    """One nbformat-4.5 cell with a content-derived id (a random one would churn every build)."""
    cell: dict[str, Any] = {
        "cell_type": cell_type,
        "id": hashlib.sha256(source.encode("utf-8")).hexdigest()[:8],
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def notebook_commands() -> list[str]:
    """Every shell command the notebook runs through the repository's scripts, as rendered."""
    cfg = load_p11_config()
    commands = [f"uv run python scripts/run_p11.py verify-gpu --model {m}" for m in cfg.e1.models]
    commands += [command.render() for command in r2_prompt_builds(cfg)]
    commands += [
        f"uv run python scripts/run_p11.py r2 --model {m} --machine {MACHINE} --no-prompt-build"
        for m in cfg.e1.models
    ]
    return commands


def render_notebook() -> str:
    cfg = load_p11_config()
    sampled = f"data/responses/{r2_arm_dir(MACHINE, 'sampled')}"
    greedy = f"data/responses/{r2_arm_dir(MACHINE, 'greedy')}"
    cells = [
        _cell("markdown", README.format(ref=R2_REF)),
        _cell("code", SETUP.format(repo=REPO, ref=R2_REF, models=list(cfg.e1.models))),
        _cell("code", ENVIRONMENT),
        _cell("code", VERIFY),
        _cell("code", PROMPTS.format(commands=[c.render() for c in r2_prompt_builds(cfg)])),
        _cell("code", GENERATE.format(machine=MACHINE)),
        _cell("code", PACK.format(sampled=sampled, greedy=greedy)),
    ]
    notebook = {
        "cells": cells,
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
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK)
    parser.add_argument("--check", action="store_true", help="Fail if the notebook is stale.")
    args = parser.parse_args(argv)
    rendered = render_notebook()
    if args.check:
        current = args.notebook.read_text(encoding="utf-8") if args.notebook.is_file() else ""
        if current != rendered:
            print(f"{args.notebook} is stale; run scripts/build_r2_notebook.py", file=sys.stderr)
            return 1
        print(f"{args.notebook} is up to date")
        return 0
    args.notebook.parent.mkdir(parents=True, exist_ok=True)
    with args.notebook.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(rendered)
    print(f"wrote {args.notebook}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
