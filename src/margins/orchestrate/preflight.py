"""Checks that must pass BEFORE a confirmatory generation run spends GPU time.

Each check is a pure function of facts the caller gathers, so every refusal is unit-testable on
the core environment with no GPU and no git. Each returns ``None`` when the check passes and a
plain-language reason when it does not: the reason is printed to whoever is at the lab machine,
and it says what to do, because the person reading it may be running this for the first time.

Why each exists:

* **Clean tree / pre-registration ancestry.** ``CLAUDE.md``: "NO FULL EXPERIMENTAL RUN MAY BE
  EXECUTED AGAINST AN EARLIER COMMIT" than ``prereg-v1``, and a manifest from a dirty tree that
  claims a clean SHA is worse than no manifest.
* **Quantization.** A CUDA host without a working bitsandbytes silently loads fp16 (P5's fallback
  records it but does not stop). For the dev model that is the right behaviour; for a confirmatory
  model it is ~46 GPU-hours at a quantization that is not the pre-registered constant (framing
  constraint 4).
* **Batch size on resume.** Batch size is part of the determinism scope, and a run resumed at a
  different one would hold responses produced under two batchings while its manifest -- which is
  rewritten on resume -- names only the second.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Final

__all__ = [
    "PREREG_TAG",
    "batch_size_conflict",
    "dirty_tree_reason",
    "nf4_unavailable_reason",
    "prereg_reason",
    "quant_reason",
]

#: The pre-registration tag every full run must descend from.
PREREG_TAG: Final[str] = "prereg-v1"


def dirty_tree_reason(dirty: bool | None) -> str | None:
    """Refuse a confirmatory run from a modified worktree."""
    if dirty is None:
        return (
            "this is not a git checkout (or git is missing), so the run manifest cannot record "
            "which code produced the results. Clone the repository with git; do not copy files."
        )
    if dirty:
        return (
            "the git worktree has uncommitted changes, so the manifest would record a SHA that is "
            "not the code that ran. Run `git status`, and do not edit tracked files at the lab: "
            "report the change to the project lead instead."
        )
    return None


def prereg_reason(is_descendant: bool | None, *, tag: str = PREREG_TAG) -> str | None:
    """Refuse a HEAD that does not descend from the pre-registration tag."""
    if is_descendant is None:
        return (
            f"could not find the tag {tag!r}. Run `git fetch --tags` and try again. A full run may "
            "not execute against a commit older than the pre-registration."
        )
    if not is_descendant:
        return (
            f"HEAD does not descend from {tag!r}. NO full experimental run may execute against an "
            "earlier commit. Check out `main` (`git checkout main && git pull`)."
        )
    return None


def nf4_unavailable_reason() -> str | None:
    """Whether NF4 can load HERE, checked before a ~16 GB download rather than after it."""
    if importlib.util.find_spec("bitsandbytes") is None:
        return (
            "bitsandbytes is not installed, so NF4 cannot load. It is Linux-only in this project's "
            "lock: on Linux run `uv sync --locked --extra dev --extra inference`; on Windows, NF4 "
            "is not available at all (use WSL2 or the Kaggle path)."
        )
    if importlib.util.find_spec("torch") is None:
        return "torch is not installed: run `uv sync --locked --extra dev --extra inference`."
    import torch

    if not torch.cuda.is_available():
        return (
            "torch sees no CUDA device. Check `nvidia-smi` works; if it does not, the NVIDIA "
            "driver is missing or broken, which needs whoever administers the machine."
        )
    return None


def quant_reason(
    *,
    requested: str,
    actual: str,
    fallback_reason: str | None,
    role: str,
    allow_fallback: bool,
) -> str | None:
    """Refuse a confirmatory model that did not load at the requested quantization."""
    if actual == requested or role != "confirmatory" or allow_fallback:
        return None
    detail = f" ({fallback_reason})" if fallback_reason else ""
    return (
        f"a CONFIRMATORY model loaded at quant={actual!r} but {requested!r} was requested"
        f"{detail}. Quantization is a fixed pre-registered constant, so this run would not be the "
        "pre-registered measurement. Stopping before any generation. Fix the environment (see the "
        "runbook, section F); never pass --allow-quant-fallback for a real run."
    )


def batch_size_conflict(manifest_path: Path, requested: int) -> str | None:
    """Refuse to resume a run whose manifest records a different batch size."""
    if not manifest_path.is_file():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    inference = data.get("inference") if isinstance(data, dict) else None
    recorded = inference.get("batch_size") if isinstance(inference, dict) else None
    if recorded is None or recorded == requested:
        return None
    return (
        f"{manifest_path.name} was started at batch size {recorded}, not {requested}. Batch size "
        "is part of the determinism scope, so one run may not mix two. Resume at "
        f"--batch-size {recorded}, or start a NEW run (different --run-name) at {requested}."
    )
