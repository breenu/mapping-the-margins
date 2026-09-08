"""P0 exit-gate demonstration: seeding determinism + an example run manifest.

Run with ``uv run python scripts/p0_demo.py``. Writes its manifest to a temporary directory
so the demo never pollutes committed ``results/``.

This demo deliberately uses **no torch**. The P0 determinism gate is the seeding utility; the
toy-model check belongs to P5 (PROJECT_CONTEXT.md section 11).
"""

from __future__ import annotations

import json
import random
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from margins.config import config_from_mapping, manifest_path, write_run_manifest
from margins.repro import environment_fingerprint, residual_nondeterminism_note, set_global_seed

SEED = 20260908


def stochastic_function(rng: np.random.Generator) -> np.ndarray:
    """The same pure-numpy function the P0 gate test reproduces bit-identically."""
    draws = rng.normal(size=(64, 8))
    weights = rng.dirichlet(np.ones(8))
    return np.asarray(draws @ weights + rng.exponential(scale=2.0, size=64))


def demo_determinism() -> None:
    print("=" * 78)
    print("1. Seeding determinism (no torch involved)")
    print("=" * 78)

    report_a, rng_a = set_global_seed(SEED)
    stdlib_a = [round(random.random(), 12) for _ in range(3)]
    array_a = stochastic_function(rng_a)

    report_b, rng_b = set_global_seed(SEED)
    stdlib_b = [round(random.random(), 12) for _ in range(3)]
    array_b = stochastic_function(rng_b)

    _, rng_c = set_global_seed(SEED + 1)
    array_c = stochastic_function(rng_c)

    print(f"  seed                       : {SEED}")
    print(f"  stdlib random, run A       : {stdlib_a}")
    print(f"  stdlib random, run B       : {stdlib_b}")
    print(f"  stdlib identical           : {stdlib_a == stdlib_b}")
    print(f"  numpy fn first 3, run A    : {np.round(array_a[:3], 12).tolist()}")
    print(f"  numpy fn first 3, run B    : {np.round(array_b[:3], 12).tolist()}")
    print(f"  BIT-identical (tobytes)    : {array_a.tobytes() == array_b.tobytes()}")
    print(f"  differs at seed {SEED + 1}   : {array_a.tobytes() != array_c.tobytes()}")
    print(f"  PYTHONHASHSEED set         : {report_a.python_hash_seed}")
    print(f"  ...effective this process  : {report_a.python_hash_seed_effective_this_process}")
    print("    (False is correct: set at runtime, so it governs CHILD processes only)")
    print(f"  torch seeded               : {report_b.torch_seeded} (core env has no torch)")
    print(f"  cuBLAS workspace config    : {report_b.cublas_workspace_config}")

    fingerprint = environment_fingerprint()
    print(f"  BLAS backend               : {fingerprint.blas_backend}")
    print(f"  torch version              : {fingerprint.torch_version}")

    print()
    print("  Determinism claim actually being made:")
    for line in _wrap(residual_nondeterminism_note(), 72):
        print(f"    {line}")
    print()


def demo_manifest() -> None:
    print("=" * 78)
    print("2. Example run manifest")
    print("=" * 78)

    with tempfile.TemporaryDirectory() as tmp:
        cfg = config_from_mapping(
            {
                "seed": SEED,
                "run_name": "p0-demo",
                "results_dir": str(Path(tmp) / "results"),
                "data_dir": str(Path(tmp) / "data"),
                "output_dir": str(Path(tmp) / "outputs"),
                "quant": "nf4",
            }
        )
        manifest = write_run_manifest(
            cfg,
            inputs={},
            timestamp=datetime(2026, 9, 8, 12, 0, 0, tzinfo=UTC),
        )
        path = manifest_path(cfg, manifest.run_id)

        print(f"  written to : <tmpdir>/{path.relative_to(tmp).as_posix()}")
        print(f"  run_id     : {manifest.run_id}  (derived from config hash + timestamp)")
        print(f"  git_sha    : {manifest.git_sha}")
        print(f"  git_dirty  : {manifest.git_dirty}")
        print()
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        print("  round-trip: returned object == file on disk :", end=" ")
        print(manifest.model_dump(mode="json") == on_disk)
        print()
        print(json.dumps(on_disk, indent=2, sort_keys=True))


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        if sum(len(w) + 1 for w in current) + len(word) > width:
            lines.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def main() -> None:
    demo_determinism()
    demo_manifest()


if __name__ == "__main__":
    main()
