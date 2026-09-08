"""Validated experiment configuration and the run manifest.

Why this module exists: the run manifest is the *only* experiment-tracking mechanism in this
project (PROJECT_CONTEXT.md section 6 -- no MLflow, no W&B, no DVC). It therefore has to carry
everything needed to reconstruct a run, and it has to be honest about what it could not
determine. Two design points follow from that:

* ``run_id`` is **derived**, not random. It is a function of the config hash and an explicitly
  passed timestamp, so the same config is traceable across runs and a test can pin it.
* :func:`write_run_manifest` returns the manifest **re-read from disk**, not the in-memory
  object it serialised. Anything JSON cannot round-trip (tuple to list, ``Path`` to ``str``)
  is then impossible to hide: what the caller gets back is literally what a reproducer will
  read.

Scope caveat: :class:`ExperimentConfig` is deliberately small. Axes, cells, benchmarks, and
scorers do not exist until P3-P9 and must not be anticipated here -- a config field invented
before its phase is a field whose validation is untested.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from margins.repro import (
    MAX_SEED,
    EnvironmentFingerprint,
    environment_fingerprint,
    residual_nondeterminism_note,
)

#: Recorded in every manifest so a reproducer knows the exact dependency surface. Packages
#: absent from the environment are recorded as ``NOT_INSTALLED`` rather than omitted --
#: "torch was not here" is a fact worth keeping, especially for the determinism claim.
TRACKED_PACKAGES: Final[tuple[str, ...]] = (
    "pydantic",
    "numpy",
    "pandas",
    "pyarrow",
    "scipy",
    "statsmodels",
    "pingouin",
    "hydra-core",
    "omegaconf",
    "jinja2",
    "matplotlib",
    "seaborn",
    "rapidfuzz",
    "torch",
    "transformers",
    "bitsandbytes",
)

#: Sentinel for a tracked package that is not installed in this environment.
NOT_INSTALLED: Final[str] = "not-installed"

#: Quantization settings. This is a **scalar constant recorded per run**, never an
#: experimental axis (CLAUDE.md framing constraint 4).
QuantSetting = Literal["nf4", "gguf_q4_k_m", "none"]

#: Manifest ``inputs`` key for the lattice config's sha256. A constant rather than a literal so
#: the key is greppable and a test asserts the key, not a spelling of it. Added at P4a: before it,
#: a manifest recorded the experiment config but not WHICH identity lattice produced the prompt
#: set, so a run could not be traced back to the cell inventory it audited.
LATTICE_INPUT_KEY: Final[str] = "lattice_config"

#: Default lattice name. Kept as a literal rather than imported from ``margins.lattice.config``:
#: that module imports ``margins.schemas``, which imports this one, so the import would cycle.
#: ``tests/test_lattice_in_manifest.py`` pins the two together.
DEFAULT_LATTICE_NAME: Final[str] = "confirmatory_v1"

_RUN_NAME_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class ManifestSettings(BaseModel):
    """Where and whether the run manifest is written."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    write_manifest: bool = True
    manifest_subdir: str = Field(default="run_manifests", min_length=1)
    record_environment_fingerprint: bool = True


class ExperimentConfig(BaseModel):
    """The validated root config. Frozen: a run's config cannot change mid-run.

    Hydra composes plain YAML into a plain dict, and :func:`config_from_mapping` validates it
    into this. There are no Hydra structured configs and no hydra-zen anywhere in the project
    (PROJECT_CONTEXT.md section 6).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = Field(ge=0, le=MAX_SEED)
    run_name: str = Field(min_length=1, max_length=128)

    output_dir: Path = Path("outputs")
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")

    quant: QuantSetting = Field(
        default="nf4",
        description=(
            "Fixed, recorded CONSTANT -- deliberately a scalar and never a list. Quantization "
            "is not an experimental axis and not a finding (CLAUDE.md framing constraint 4)."
        ),
    )

    # Added at P4a, and deliberately the ONLY field added there: a run's manifest must record
    # which identity lattice produced its prompt set. The name resolves to
    # `configs/lattice/<name>.yaml`, whose sha256 goes into `write_run_manifest(inputs=...)`
    # under LATTICE_INPUT_KEY -- the name alone would not survive an edit to the file.
    #
    # Named `lattice_name`, not `lattice`, ON PURPOSE: `configs/lattice/` is a directory under the
    # Hydra config root and is therefore a config GROUP. A key colliding with a group name cannot
    # be overridden at all -- Hydra rejects `lattice=x` ("no match in the defaults list") and
    # `++lattice=x` ("force-add of config groups is not supported"), so the field would be frozen
    # at whatever config.yaml said. The `_name` suffix keeps the two namespaces apart.
    lattice_name: str = Field(
        default=DEFAULT_LATTICE_NAME,
        min_length=1,
        description="Lattice config name; its sha256 is recorded in the manifest inputs.",
    )

    # Open decision #2 (model checkpoint + pinned revision) is not settled until P5, so these
    # stay None. A revision is required alongside an id precisely because an unpinned HF repo
    # can change under a reproducer.
    model_id: str | None = None
    model_revision: str | None = None

    manifest: ManifestSettings = ManifestSettings()


class RunManifest(BaseModel):
    """Everything needed to reconstruct a run, including what could not be determined.

    ``git_sha`` and ``git_dirty`` are both ``None`` outside a repository. A manifest from a
    dirty tree that claims a clean SHA is worse than no manifest (PROJECT_CONTEXT.md section
    4), which is why the dirty flag is a first-class field rather than an afterthought.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    created_at: str
    config_hash: str
    config: ExperimentConfig
    git_sha: str | None
    git_dirty: bool | None
    package_versions: dict[str, str]
    python_version: str
    platform: str
    seed: int
    environment: EnvironmentFingerprint
    determinism_note: str
    input_checksums: dict[str, str]
    quant: QuantSetting


def config_hash(cfg: ExperimentConfig) -> str:
    """Content-addressed hash of a config.

    Hashes the JSON-mode dump with sorted keys, so the value depends on the config's *content*
    and not on the key order of the YAML it was composed from.
    """
    payload = json.dumps(
        cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def config_from_mapping(data: Mapping[str, Any]) -> ExperimentConfig:
    """Validate a plain mapping into an :class:`ExperimentConfig`.

    This is the Hydra seam: Hydra/OmegaConf composes YAML, the caller converts to a plain
    container, and validation happens here in Pydantic. Keeping Hydra out of the type system
    is the cheap side of the Hydra/Pydantic friction.
    """
    return ExperimentConfig.model_validate(dict(data))


def default_config_dir() -> Path:
    """``<repo>/configs``. Override it for tests or an installed package."""
    return Path(__file__).resolve().parents[2] / "configs"


def load_experiment_config(
    name: str = "config",
    *,
    config_dir: Path | None = None,
    overrides: Sequence[str] = (),
) -> ExperimentConfig:
    """Compose ``<config_dir>/<name>.yaml`` with Hydra and validate it.

    The same seam as :func:`margins.lattice.config.load_lattice_config`, hoisted here so a script
    that needs to write a manifest does not have to re-implement the Hydra composition (and get
    the override syntax subtly different).
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    directory = (config_dir or default_config_dir()).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"config dir does not exist: {directory}")

    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        composed = compose(config_name=name, overrides=list(overrides))
    data = OmegaConf.to_container(composed, resolve=True)
    if not isinstance(data, dict):
        raise TypeError(f"composed config {name!r} is not a mapping, got {type(data)!r}")
    return config_from_mapping({str(key): value for key, value in data.items()})


def _package_versions() -> dict[str, str]:
    """Installed version of each tracked package, or :data:`NOT_INSTALLED`."""
    versions: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = NOT_INSTALLED
    return versions


def _git(args: list[str], repo_root: Path) -> str | None:
    """Run a git command, returning stripped stdout or ``None`` on any failure."""
    try:
        completed = subprocess.run(  # fixed argv, no shell
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _git_state(repo_root: Path) -> tuple[str | None, bool | None]:
    """Return ``(sha, dirty)`` for the worktree at ``repo_root``.

    Both are ``None`` outside a git worktree. Inside one they are reported independently,
    because they can legitimately disagree: a freshly initialised repo with an unborn HEAD has
    no SHA (``None``) while still being dirty (``True``). Coupling them would force one of the
    two to be a lie.
    """
    if _git(["rev-parse", "--is-inside-work-tree"], repo_root) != "true":
        return None, None
    sha = _git(["rev-parse", "HEAD"], repo_root)
    status = _git(["status", "--porcelain"], repo_root)
    dirty = None if status is None else bool(status)
    return sha, dirty


def make_run_id(cfg: ExperimentConfig, config_digest: str, timestamp: datetime) -> str:
    """Derive a run id from the config and a recorded timestamp.

    Deterministic by construction: the same config and timestamp always yield the same id, and
    a changed config always yields a different one. Nothing here consults a clock or an RNG.
    """
    stamp = timestamp.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_name = _RUN_NAME_SAFE.sub("-", cfg.run_name).strip("-") or "run"
    return f"{safe_name}-{stamp}-{config_digest[:12]}"


def write_run_manifest(
    cfg: ExperimentConfig,
    *,
    inputs: Mapping[str, str] | None = None,
    timestamp: datetime | None = None,
    repo_root: Path | None = None,
) -> RunManifest:
    """Write ``<results_dir>/<manifest_subdir>/<run_id>.json`` and return what was written.

    Args:
        cfg: The validated config for this run.
        inputs: ``{source_name: sha256}`` for every input consumed. Empty until P2 has
            sources to checksum.
        timestamp: Recorded creation time; defaults to now (UTC). Pass it explicitly to make
            ``run_id`` reproducible.
        repo_root: Where to ask git about the worktree. Defaults to the package's repository.

    Returns:
        The manifest **re-read and re-validated from the file on disk**, so the returned object
        cannot drift from what a reproducer will actually read.

    Raises:
        ValueError: If ``cfg.manifest.write_manifest`` is False -- writing is the point of the
            call, so a silent no-op returning a manifest that is not on disk would be a lie.
    """
    if not cfg.manifest.write_manifest:
        raise ValueError(
            "write_run_manifest called with manifest.write_manifest=False; refusing to return "
            "a manifest that was never written."
        )

    created = (timestamp or datetime.now(UTC)).astimezone(UTC)
    digest = config_hash(cfg)
    run_id = make_run_id(cfg, digest, created)
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[2]
    git_sha, git_dirty = _git_state(root)

    manifest = RunManifest(
        run_id=run_id,
        created_at=created.isoformat().replace("+00:00", "Z"),
        config_hash=digest,
        config=cfg,
        git_sha=git_sha,
        git_dirty=git_dirty,
        package_versions=_package_versions(),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        seed=cfg.seed,
        environment=(
            environment_fingerprint()
            if cfg.manifest.record_environment_fingerprint
            else EnvironmentFingerprint()
        ),
        determinism_note=residual_nondeterminism_note(),
        input_checksums=dict(inputs or {}),
        quant=cfg.quant,
    )

    destination = cfg.results_dir / cfg.manifest.manifest_subdir / f"{run_id}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    # Read back rather than trusting the in-memory object: this is what makes JSON round-trip
    # drift impossible to ship unnoticed.
    return RunManifest.model_validate_json(destination.read_text(encoding="utf-8"))


def manifest_path(cfg: ExperimentConfig, run_id: str) -> Path:
    """Where :func:`write_run_manifest` puts the manifest for ``run_id``."""
    return cfg.results_dir / cfg.manifest.manifest_subdir / f"{run_id}.json"
