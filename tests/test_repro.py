"""P0 gate: the seeding utility is the determinism check for this phase.

Deliberately torch-free. The toy-model determinism check belongs to P5
(PROJECT_CONTEXT.md section 11); asserting it here would claim a guarantee P0 has not earned.
"""

from __future__ import annotations

import importlib.util
import os
import random

import numpy as np
import pytest

from margins.repro import (
    MAX_SEED,
    SeedReport,
    environment_fingerprint,
    residual_nondeterminism_note,
    set_global_seed,
)


def _stochastic_function(rng: np.random.Generator) -> np.ndarray:
    """A pure-numpy stochastic function with enough arithmetic to expose ordering drift."""
    draws = rng.normal(loc=0.0, scale=1.0, size=(64, 8))
    weights = rng.dirichlet(np.ones(8))
    return np.asarray(draws @ weights + rng.exponential(scale=2.0, size=64))


def test_same_seed_reproduces_stdlib_and_legacy_numpy() -> None:
    set_global_seed(1234)
    first = ([random.random() for _ in range(20)], np.random.random(20).tolist())
    set_global_seed(1234)
    second = ([random.random() for _ in range(20)], np.random.random(20).tolist())
    assert first == second


def test_different_seed_gives_different_draws() -> None:
    set_global_seed(1234)
    first = [random.random() for _ in range(20)]
    set_global_seed(4321)
    second = [random.random() for _ in range(20)]
    assert first != second


def test_stochastic_function_reproduces_bit_identically() -> None:
    """Bit equality, not ``allclose`` -- the invariant-5 claim is about bits, not tolerance."""
    _, rng_a = set_global_seed(20260908)
    a = _stochastic_function(rng_a)
    _, rng_b = set_global_seed(20260908)
    b = _stochastic_function(rng_b)

    assert a.dtype == b.dtype
    assert a.shape == b.shape
    assert a.tobytes() == b.tobytes()


def test_stochastic_function_differs_across_seeds() -> None:
    _, rng_a = set_global_seed(1)
    _, rng_b = set_global_seed(2)
    assert _stochastic_function(rng_a).tobytes() != _stochastic_function(rng_b).tobytes()


def test_returned_generator_is_explicit_and_independent_of_global_state() -> None:
    """The returned generator must not be affected by later global-RNG use."""
    _, rng_a = set_global_seed(7)
    np.random.random(100)
    random.random()
    first = rng_a.random(10).tobytes()

    _, rng_b = set_global_seed(7)
    first_again = rng_b.random(10).tobytes()
    assert first == first_again


def test_python_hash_seed_is_set_and_caveat_recorded() -> None:
    report, _ = set_global_seed(99)
    assert os.environ["PYTHONHASHSEED"] == "99"
    assert report.python_hash_seed == "99"
    # Set at runtime, so it governs child processes only, never this interpreter.
    assert report.python_hash_seed_effective_this_process is False


def test_seed_report_is_frozen_and_records_torch_truthfully() -> None:
    report, _ = set_global_seed(5)
    assert isinstance(report, SeedReport)
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError on frozen assignment
        report.seed = 6  # type: ignore[misc]

    torch_present = importlib.util.find_spec("torch") is not None
    assert report.torch_seeded is torch_present
    if not torch_present:
        assert report.torch_version is None
        assert report.torch_deterministic_algorithms is False


def test_module_functions_without_torch() -> None:
    """Core env has no torch; nothing here may require it."""
    fingerprint = environment_fingerprint()
    if importlib.util.find_spec("torch") is None:
        assert fingerprint.torch_version is None
        assert fingerprint.torch_num_threads is None
        assert fingerprint.torch_cuda_available is None
    assert set(fingerprint.thread_env) >= {"OMP_NUM_THREADS", "MKL_NUM_THREADS"}


def test_gpu_facts_degrade_to_empty_without_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    """No driver binary is normal on the dev box and in CI: record nothing, and never raise."""
    import subprocess

    def missing(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", missing)
    fingerprint = environment_fingerprint()
    assert fingerprint.cuda_devices == ()
    assert fingerprint.nvidia_driver_version is None


def test_gpu_facts_are_read_from_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    """The manifest must NAME the hardware: cross-host reproduction is metric-level only."""
    import subprocess

    from margins import repro

    def fake(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert tuple(argv) == repro.NVIDIA_SMI_QUERY
        stdout = "NVIDIA A100-SXM4-40GB, 40960 MiB, 550.54.15\nTesla T4, 15360 MiB, 550.54.15\n"
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake)
    fingerprint = environment_fingerprint()
    assert fingerprint.nvidia_driver_version == "550.54.15"
    assert fingerprint.cuda_devices == ("NVIDIA A100-SXM4-40GB, 40960 MiB", "Tesla T4, 15360 MiB")


def test_torch_cuda_facts_are_recorded_without_creating_a_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """torch's CUDA/cuDNN/BLAS are recorded; the device name only if a context ALREADY exists."""
    from types import SimpleNamespace

    from margins import repro

    calls: list[str] = []

    def device_name(index: int) -> str:
        calls.append("name")
        return "Fake GPU"

    cuda = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: False,
        current_device=lambda: 0,
        get_device_capability=lambda index: (8, 6),
        get_device_name=device_name,
    )
    fake_torch = SimpleNamespace(
        __version__="9.9.9",
        get_num_threads=lambda: 4,
        cuda=cuda,
        version=SimpleNamespace(cuda="12.4"),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 90100)),
        __config__=SimpleNamespace(show=lambda: "PyTorch built with:\n  - BLAS_INFO=mkl, X=1\n"),
    )
    monkeypatch.setattr(repro, "_torch_module", lambda: fake_torch)
    fingerprint = environment_fingerprint()
    assert fingerprint.torch_cuda_version == "12.4"
    assert fingerprint.cudnn_version == 90100
    assert fingerprint.torch_blas_info == "mkl"
    assert fingerprint.cuda_device_in_use is None
    assert calls == [], "reading the device name would have created a CUDA context"

    monkeypatch.setattr(cuda, "is_initialized", lambda: True)
    assert environment_fingerprint().cuda_device_in_use == "Fake GPU (cuda:0, sm_86)"


def test_blas_backend_is_actually_identified() -> None:
    """The determinism note promises the BLAS backend is recorded, so a silent None is a bug.

    Guards the numpy-config probe specifically: it previously swallowed a TypeError from an
    unsupported kwarg and reported None while claiming the backend was recorded.
    """
    config = getattr(np, "__config__", None)
    show = getattr(config, "show", None)
    if show is None:  # pragma: no cover - numpy always ships this today
        pytest.skip("numpy exposes no build config")
    try:
        dict_mode_supported = isinstance(show(mode="dicts"), dict)
    except (TypeError, ValueError):
        dict_mode_supported = False
    if not dict_mode_supported:  # pragma: no cover - numpy < 2
        pytest.skip("numpy build config has no dict mode")

    backend = environment_fingerprint().blas_backend
    assert isinstance(backend, str) and backend.strip()


def test_cublas_workspace_config_set() -> None:
    report, _ = set_global_seed(11)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == report.cublas_workspace_config
    assert report.cublas_workspace_config


@pytest.mark.parametrize("bad", [-1, MAX_SEED + 1])
def test_out_of_range_seed_rejected(bad: int) -> None:
    with pytest.raises(ValueError):
        set_global_seed(bad)


@pytest.mark.parametrize("bad", [1.5, "7", None, True])
def test_non_int_seed_rejected(bad: object) -> None:
    with pytest.raises(TypeError):
        set_global_seed(bad)  # type: ignore[arg-type]


def test_residual_nondeterminism_note_cannot_be_silently_weakened() -> None:
    """Guards the honesty of the claim, not its prose.

    If someone rewrites this into an unconditional bit-identity promise, this test fails.
    """
    note = residual_nondeterminism_note().lower()
    for required in (
        "identical token sequences",
        "cpu",
        "thread count",
        # Added at P5: batching pads, padding changes reduction order, so the claim is only true
        # at a fixed batch size. Dropping this qualifier would overstate what the harness delivers.
        "batch size",
        "tolerance",
        "metric-level",
        "gpu",
    ):
        assert required in note, f"determinism note lost its {required!r} qualifier"
    assert "no unconditional bit-identity is claimed" in note
