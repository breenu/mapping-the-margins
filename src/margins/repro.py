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
import re
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

    # --- the accelerator (added at P11) -------------------------------------------------------
    #
    # The confirmatory sweep moved to a university GPU (open decision #8), which makes
    # reproduction METRIC-level within a reported tolerance rather than bit-level. That scoping is
    # only checkable if the manifest says what hardware produced the numbers, so these are recorded
    # on every run. All default to None: the dev box and the core CI job have no GPU at all.
    torch_blas_info: str | None = Field(
        default=None,
        description=(
            "torch's OWN BLAS, parsed from torch.__config__.show(). `blas_backend` is numpy's, "
            "which is not the library a model's matmuls run through."
        ),
    )
    torch_cuda_version: str | None = Field(
        default=None, description="The CUDA runtime torch was built against (torch.version.cuda)."
    )
    cudnn_version: int | None = None
    nvidia_driver_version: str | None = Field(
        default=None,
        description="From `nvidia-smi`, because torch exposes no driver API. None if absent.",
    )
    cuda_devices: tuple[str, ...] = Field(
        default=(),
        description="Every visible GPU as '<name>, <total memory>', from `nvidia-smi`.",
    )
    cuda_device_in_use: str | None = Field(
        default=None,
        description=(
            "The device torch actually initialised, with its compute capability. Read only when a "
            "CUDA context already exists: creating one here would precede set_global_seed's "
            "CUBLAS_WORKSPACE_CONFIG and silently weaken the determinism it sets up."
        ),
    )


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


#: ``nvidia-smi`` query for the facts torch cannot give: the driver, and every visible card.
NVIDIA_SMI_QUERY: Final[tuple[str, ...]] = (
    "nvidia-smi",
    "--query-gpu=name,memory.total,driver_version",
    "--format=csv,noheader",
)


def _nvidia_smi() -> tuple[tuple[str, ...], str | None]:
    """``(devices, driver_version)`` from ``nvidia-smi``, or ``((), None)``. Never raises.

    A subprocess rather than ``pynvml`` because pynvml is not a dependency, and a missing binary
    (every machine without an NVIDIA driver, including CI) is the normal case, not an error.
    """
    import subprocess

    try:
        completed = subprocess.run(  # fixed argv, no shell
            list(NVIDIA_SMI_QUERY), capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return (), None
    if completed.returncode != 0:
        return (), None
    devices: list[str] = []
    driver: str | None = None
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3 or not parts[0]:
            continue
        devices.append(f"{parts[0]}, {parts[1]}")
        driver = driver or parts[2]
    return tuple(devices), driver


def _torch_blas_info(torch: Any) -> str | None:
    """The ``BLAS_INFO=`` entry of ``torch.__config__.show()``, e.g. ``mkl``."""
    try:
        shown = str(torch.__config__.show())
    except (AttributeError, RuntimeError):  # pragma: no cover - needs a torch build
        return None
    # Real output buries it mid-line: "  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, ...".
    match = re.search(r"\bBLAS_INFO=([^,\s]+)", shown)
    return match.group(1) if match else None


def _cuda_device_in_use(torch: Any) -> str | None:
    """The initialised device's name and capability -- only if a CUDA context already exists."""
    try:
        if not torch.cuda.is_initialized():
            return None
        index = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        return f"{torch.cuda.get_device_name(index)} (cuda:{index}, sm_{major}{minor})"
    except (AttributeError, RuntimeError, AssertionError):
        return None


def environment_fingerprint() -> EnvironmentFingerprint:
    """Collect the environment facts a reproducer needs to check the determinism claim.

    Never raises. Absent ``torch`` yields ``None`` fields, which is the normal dev-box state, and
    an absent ``nvidia-smi`` yields no devices and no driver, which is every machine without a GPU.
    """
    torch = _torch_module()
    torch_version: str | None = None
    torch_threads: int | None = None
    cuda_available: bool | None = None
    torch_blas: str | None = None
    torch_cuda: str | None = None
    cudnn: int | None = None
    in_use: str | None = None
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
        torch_blas = _torch_blas_info(torch)
        version = getattr(getattr(torch, "version", None), "cuda", None)
        torch_cuda = None if version is None else str(version)
        try:
            raw = torch.backends.cudnn.version() if cuda_available else None
            cudnn = None if raw is None else int(raw)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            cudnn = None
        in_use = _cuda_device_in_use(torch)

    devices, driver = _nvidia_smi()
    return EnvironmentFingerprint(
        torch_version=torch_version,
        torch_num_threads=torch_threads,
        torch_cuda_available=cuda_available,
        blas_backend=_blas_backend(),
        thread_env={var: os.environ.get(var) for var in _THREAD_ENV_VARS},
        torch_blas_info=torch_blas,
        torch_cuda_version=torch_cuda,
        cudnn_version=cudnn,
        nvidia_driver_version=driver,
        cuda_devices=devices,
        cuda_device_in_use=in_use,
    )
