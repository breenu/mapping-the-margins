"""Seeding and determinism reporting.

Why this module exists: determinism is a *contract* in this project (CLAUDE.md invariant 5),
and a determinism claim that a reproducer cannot check is worse than no claim at all. So this
module does two things that are easy to conflate and must not be:

1. It *sets* every seed and determinism switch we can reach, and
2. it *records what was actually set* -- including that ``torch`` was absent -- so the run
   manifest states truth rather than assumption.

Construct caveat, stated plainly because the paper depends on it: seeding does **not** buy
bit-identical output everywhere. What it buys is spelled out by
:func:`residual_nondeterminism_note`, and nothing in this codebase may claim more.

``torch`` is imported lazily and is genuinely optional. The Windows dev environment installs
core + dev only (PROJECT_CONTEXT.md section 7) and this module is fully functional there.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import random
from typing import Any, Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

#: numpy's legacy global seeder accepts ``[0, 2**32)``; we take that as the project-wide bound
#: so that a seed valid for one RNG is valid for all of them.
MAX_SEED: Final[int] = 2**32 - 1

#: Required by cuBLAS for deterministic GEMM on CUDA >= 10.2. Must be set *before* the CUDA
#: context is created, which is why it is set here rather than at the call site.
CUBLAS_WORKSPACE_CONFIG: Final[str] = ":4096:8"

#: Thread-count environment variables that materially change floating-point reduction order.
#: Recorded, never set -- silently pinning a user's thread count would hide the real config.
_THREAD_ENV_VARS: Final[tuple[str, ...]] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


class SeedReport(BaseModel):
    """What :func:`set_global_seed` actually did, as opposed to what it tried to do.

    Recorded into the run manifest. ``torch_seeded=False`` with ``torch_version=None`` is the
    normal, expected state on the dev box; it is information, not a failure.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = Field(ge=0, le=MAX_SEED)
    python_hash_seed: str
    python_hash_seed_effective_this_process: bool = Field(
        description=(
            "False whenever PYTHONHASHSEED was set at runtime rather than before interpreter "
            "start: this process's str hash randomisation is already fixed and only CHILD "
            "processes inherit the value."
        ),
    )
    torch_seeded: bool
    torch_version: str | None
    torch_deterministic_algorithms: bool
    torch_cudnn_deterministic: bool
    cublas_workspace_config: str


class EnvironmentFingerprint(BaseModel):
    """The environment facts that decide whether a determinism claim is checkable.

    Every field degrades to ``None`` rather than raising: this is called while writing a
    manifest, and a manifest that fails to write because ``numpy`` moved a private attribute
    is a worse outcome than a manifest with a null in it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    torch_version: str | None = None
    torch_num_threads: int | None = None
    torch_cuda_available: bool | None = None
    blas_backend: str | None = None
    thread_env: dict[str, str | None] = Field(default_factory=dict)


def _torch_module() -> Any | None:
    """Return the imported ``torch`` module, or ``None`` when the inference extra is absent."""
    if importlib.util.find_spec("torch") is None:
        return None
    try:
        return importlib.import_module("torch")
    except ImportError:  # pragma: no cover - a broken install, not a missing one
        return None


def set_global_seed(seed: int) -> tuple[SeedReport, np.random.Generator]:
    """Seed every global RNG we can reach and hand back an *explicit* numpy generator.

    The returned :class:`numpy.random.Generator` is the point of the tuple: CLAUDE.md forbids
    hidden global RNG on stochastic paths, so callers are expected to thread this object
    through rather than reach for ``np.random.*``. The legacy global is seeded anyway, because
    third-party code (statsmodels, seaborn, scipy) still reads it.

    Args:
        seed: Integer in ``[0, MAX_SEED]``.

    Returns:
        The :class:`SeedReport` to record in the manifest, and a fresh seeded generator.

    Raises:
        TypeError: If ``seed`` is not an ``int`` (``bool`` is rejected explicitly).
        ValueError: If ``seed`` is outside ``[0, MAX_SEED]``.

    Caveat -- ``PYTHONHASHSEED``:
        Setting it here affects **child processes only**. This interpreter's string-hash
        randomisation was fixed at startup and cannot be changed retroactively. The report
        records which case applies rather than pretending the difference away.

    Caveat -- ``CUBLAS_WORKSPACE_CONFIG``:
        Must precede CUDA context creation to have any effect. Call this before touching a
        model.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be in [0, {MAX_SEED}], got {seed}")

    hash_seed_was_preset = os.environ.get("PYTHONHASHSEED") == str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)

    random.seed(seed)
    np.random.seed(seed)
    generator = np.random.default_rng(seed)

    torch = _torch_module()
    torch_deterministic = False
    cudnn_deterministic = False
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)
            torch_deterministic = True
        except (RuntimeError, AttributeError):  # pragma: no cover - needs a torch build
            torch_deterministic = False
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            cudnn_deterministic = True
        except AttributeError:  # pragma: no cover - needs a torch build
            cudnn_deterministic = False

    report = SeedReport(
        seed=seed,
        python_hash_seed=str(seed),
        python_hash_seed_effective_this_process=hash_seed_was_preset,
        torch_seeded=torch is not None,
        torch_version=None if torch is None else str(torch.__version__),
        torch_deterministic_algorithms=torch_deterministic,
        torch_cudnn_deterministic=cudnn_deterministic,
        cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
    )
    return report, generator


def residual_nondeterminism_note() -> str:
    """The exact determinism claim this project is entitled to make.

    Copied into every run manifest so the scope of the claim travels with the results. It is
    deliberately hedged in three places; weakening the hedges would overstate invariant 5.
    """
    return (
        "Determinism scope. Same-seed greedy decoding produces identical token sequences "
        "on CPU at a fixed thread count AND a fixed batch size, with the torch version and "
        "BLAS backend recorded in this manifest. Batch size is load-bearing here: padding a "
        "batch changes floating-point reduction order, so the same prompt run alone and run "
        "inside a batch may diverge -- the batch size used is recorded in this manifest's "
        "inference block. Log-probabilities match to a documented tolerance rather than "
        "bit-exactly. On GPU the guarantee is metric-level only, within a reported "
        "tolerance: kernel selection, atomic reduction order, and cuBLAS workspace reuse are "
        "not fully controllable. Changing thread count, batch size, BLAS backend, torch "
        "version, or hardware invalidates bit-level comparison. No unconditional bit-identity "
        "is claimed."
    )


def _blas_backend() -> str | None:
    """Best-effort BLAS identification from ``numpy``'s build config."""
    config_module: Any = getattr(np, "__config__", None)
    if config_module is None:  # pragma: no cover - numpy always ships this today
        return None
    show = getattr(config_module, "show", None)
    if show is not None:
        config: Any = None
        try:
            config = show(mode="dicts")
        except (TypeError, ValueError):  # numpy < 2 has no dict mode
            config = None
        if isinstance(config, dict):
            build_deps = config.get("Build Dependencies", {})
            blas = build_deps.get("blas", {}) if isinstance(build_deps, dict) else {}
            name = blas.get("name") if isinstance(blas, dict) else None
            version = blas.get("version") if isinstance(blas, dict) else None
            if name:
                return f"{name} {version}" if version else str(name)
    for attr in ("openblas64__info", "openblas_info", "blas_opt_info", "blas_mkl_info"):
        info: Any = getattr(config_module, attr, None)
        if isinstance(info, dict) and info:
            return attr.removesuffix("_info")
    return None


def environment_fingerprint() -> EnvironmentFingerprint:
    """Collect the environment facts a reproducer needs to check the determinism claim.

    Never raises. Absent ``torch`` yields ``None`` fields, which is the normal dev-box state.
    """
    torch = _torch_module()
    torch_version: str | None = None
    torch_threads: int | None = None
    cuda_available: bool | None = None
    if torch is not None:
        torch_version = str(torch.__version__)
        try:
            torch_threads = int(torch.get_num_threads())
        except (RuntimeError, AttributeError):  # pragma: no cover - needs a torch build
            torch_threads = None
        try:
            cuda_available = bool(torch.cuda.is_available())
        except (RuntimeError, AttributeError):  # pragma: no cover - needs a torch build
            cuda_available = None

    return EnvironmentFingerprint(
        torch_version=torch_version,
        torch_num_threads=torch_threads,
        torch_cuda_available=cuda_available,
        blas_backend=_blas_backend(),
        thread_env={var: os.environ.get(var) for var in _THREAD_ENV_VARS},
    )
