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
    # P6 scoring. Core deps, so these are always present -- but recorded anyway, because an
    # agreement coefficient depends on the implementation that computed it.
    "krippendorff",
    "scikit-learn",
    "vaderSentiment",
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

#: Default decoding profile name, and the one P4a already stamps on every prompt. A literal for the
#: same reason as :data:`DEFAULT_LATTICE_NAME`; ``tests/test_inference_manifest.py`` pins it against
#: ``margins.inference.decoding.DEFAULT_DECODING``.
DEFAULT_DECODING_NAME: Final[str] = "greedy_v1"

#: Default scoring roster name. A literal for the same import-cycle reason as the two above;
#: ``tests/test_scoring_manifest.py`` pins it against ``margins.scoring.config.DEFAULT_SCORING``.
DEFAULT_SCORING_NAME: Final[str] = "default_v1"

#: Default metrics config name. A literal for the same import-cycle reason as the three above;
#: ``tests/test_metrics_manifest.py`` pins it against ``margins.metrics.config.DEFAULT_METRICS``.
DEFAULT_METRICS_NAME: Final[str] = "emergence_v1"
#: Pinned to margins.stats.config.DEFAULT_STATS by test_stats_manifest.py. A literal here for
#: the same import-cycle reason as the four above.
DEFAULT_STATS_NAME: Final[str] = "confirmatory_v1"

#: Default extractive-benchmark config name. A literal for the same import-cycle reason as the
#: five above; ``tests/test_crows_loader.py`` pins it against
#: ``margins.benchmarks.config.DEFAULT_BENCHMARK``.
DEFAULT_BENCHMARK_NAME: Final[str] = "crows_pairs_v1"

#: Default red-team set name (P8). A literal for the same import-cycle reason as the six above;
#: ``tests/test_redteam_config.py`` pins it against ``margins.redteam.config.DEFAULT_REDTEAM_SET``.
DEFAULT_REDTEAM_NAME: Final[str] = "cell_targeted_v1"

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

    # Added at P5, and named with the `_name` suffix for exactly the reason `lattice_name` is:
    # `configs/model/` and `configs/decoding/` are Hydra config GROUPS, and a key colliding with a
    # group name cannot be overridden from the command line at all. Each name resolves to
    # `configs/<group>/<name>.yaml`, whose sha256 goes into the manifest inputs -- a name alone
    # would not survive an edit to the file.
    model_name: str | None = Field(
        default=None,
        description="Model config name. None for runs that load no model (e.g. build_prompts).",
    )
    decoding_name: str = Field(
        default=DEFAULT_DECODING_NAME,
        min_length=1,
        description="Decoding profile name; must match the decoding_key stamped on the prompts.",
    )

    # Added at P6. `_name` suffix for the same reason as the three above: `configs/scoring/` is a
    # Hydra config GROUP, and a key colliding with a group name cannot be overridden at all.
    scoring_name: str = Field(
        default=DEFAULT_SCORING_NAME,
        min_length=1,
        description="Scoring roster name; its sha256 is recorded in the manifest inputs.",
    )

    # Added at P7. `_name` suffix once more: `configs/metrics/` is a Hydra config GROUP.
    metrics_name: str = Field(
        default=DEFAULT_METRICS_NAME,
        min_length=1,
        description="Metrics config name; its sha256 is recorded in the manifest inputs.",
    )

    # Added at P9. `_name` suffix once more, and for the same reason every other one has it:
    # `configs/stats/` is a Hydra config GROUP, and a key colliding with a group name cannot be
    # overridden from the command line at all.
    stats_name: str = Field(
        default=DEFAULT_STATS_NAME,
        min_length=1,
        description="Stats config name; its sha256 is recorded in the manifest inputs.",
    )

    # Added at P2 (the EXTRACTIVE vertical). `_name` suffix once more: `configs/benchmark/` is a
    # Hydra config GROUP. Unlike the five above it is OPTIONAL and defaults to None, because most
    # runs ingest no benchmark at all -- the generation vertical never touches one, and a run that
    # named a source it did not read would put a checksum in the manifest for a file it never
    # opened. `scripts/build_crows_items.py` sets it.
    benchmark_name: str | None = Field(
        default=None,
        description=(
            "Extractive benchmark config name, or None for runs that ingest no benchmark. Its "
            "sha256, and the pinned source file's, are recorded in the manifest inputs."
        ),
    )

    # Added at P8. `_name` suffix once more: `configs/redteam/` is a Hydra config GROUP. Optional
    # and None by default, like `benchmark_name` and for the same reason -- most runs generate no
    # red-team frames, and a run naming a set it never read would put a checksum in its manifest
    # for a file it never opened. The red-team build scripts set it.
    redteam_name: str | None = Field(
        default=None,
        description=(
            "Red-team set config name, or None for runs that generate no red-team frames. Its "
            "sha256, and each elicitation template's, are recorded in the manifest inputs."
        ),
    )

    # Open decision #2 was settled at P5 (9 Sep 2026), but these stay optional and are populated
    # PER RUN from the resolved ModelSpec rather than being authored here -- the pinned sha lives in
    # `configs/model/<name>.yaml`, resolved by `scripts/pin_models.py`, and is copied in so the
    # manifest's serialised config states the exact checkpoint the run used.
    model_id: str | None = None
    model_revision: str | None = None

    manifest: ManifestSettings = ManifestSettings()


class InferenceRecord(BaseModel):
    """What the inference stage actually did, for runs that loaded a model (P5).

    Exists because :attr:`ExperimentConfig.quant` records the *intended* constant and
    ``environment`` records the *machine*, but neither says what the load resolved to. NF4 is a
    bitsandbytes/CUDA path, so a Windows dev run falls back to unquantized -- and a manifest in
    which that is indistinguishable from a real NF4 run would make the recorded constant worthless
    (CLAUDE.md framing constraint 4).

    ``batch_size`` is here for the determinism claim, not for performance bookkeeping: padding
    changes floating-point reduction order, so identical token sequences are guaranteed at a *fixed*
    batch size and thread count, and a reproducer cannot check that without knowing the batch size.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    model_name: str
    model_id: str
    model_revision: str = Field(description="The RESOLVED commit sha the run actually loaded.")
    quant_requested: QuantSetting
    quant_actual: QuantSetting
    quant_fallback_reason: str | None = None
    device: str
    dtype: str
    prompt_format: str
    chat_template_sha256: str | None = None
    decoding_key: str
    decoding_hash: str
    batch_size: int = Field(ge=1)
    seeds: tuple[int, ...]


class ScorerRecord(BaseModel):
    """One scorer as it was actually used, for runs that scored anything (P6).

    Carries the pin rather than only the id, because "which scorer" is not answerable from a
    ``scorer_id`` alone: the same id under a different classifier revision, a different lexicon
    digest, a flipped sign or a moved threshold is a different instrument producing different
    numbers. A manifest that recorded only the id could not tell two such runs apart.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    scorer_id: str
    harm_type: str
    mechanism: str
    sign: int
    threshold: float
    pin_digest: str
    model_repo: str | None = None
    model_revision: str | None = None
    model_label: str | None = None
    lexicon_name: str | None = None
    lexicon_revision: str | None = None
    lexicon_version: str | None = None
    lexicon_sha256: str | None = None
    lexicon_size: int | None = Field(
        default=None, description="Terms actually loaded. A silently empty wordlist scores 0.0."
    )
    license: str | None = None


class ScoringRecord(BaseModel):
    """What the scoring stage actually did (P6).

    ``agreement_summary`` is in the manifest, not only in the emitted table, because framing
    constraint 6 makes agreement a precondition for a harm type being load-bearing -- and a run
    whose agreement is only in a gitignored artifact cannot be audited from its manifest alone.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    scoring_name: str
    scoring_version: str
    scoring_hash: str
    scorers: tuple[ScorerRecord, ...]
    harm_types: tuple[str, ...]
    benchmark: str
    n_responses: int = Field(ge=0)
    n_scores: int = Field(ge=0)
    control_families: tuple[str, ...] = ()
    n_control_texts: int = Field(default=0, ge=0)
    degenerate_groups: int = Field(
        default=0,
        ge=0,
        description="Standardization groups with no variance; every row became 0.0.",
    )
    agreement_summary: dict[str, float] = Field(
        default_factory=dict,
        description="harm_type -> Krippendorff alpha on the model condition, where estimable.",
    )


class MetricsRecord(BaseModel):
    """What the metric stage actually did (P7): which estimator, under which thresholds.

    The estimator's identity is spelled out in fixed fields rather than left implicit, because an
    ``EffectEstimate`` carries an estimate, an SE and a CI but not how they were produced -- and a
    cluster-bootstrap CI over items is a different claim from an iid one. The two drift thresholds
    are recorded because they are PLACEHOLDERS (TODO(P9): pre-register) and the descriptive-only
    candidate flags mean nothing without the numbers they were computed against.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    metrics_name: str
    metrics_version: str
    metrics_hash: str
    benchmark: str

    # The estimand, as pre-registered. Constants, not settings: nothing in the config can change
    # them, and the manifest states them so a reproducer does not have to read the code to know.
    link: str = "identity"
    model_formula: str = "harm_std ~ A + B + A:B, over an axis pair's order-2 cells"
    coding: str = "sum_to_zero (deviation) contrasts; every level's coefficient emitted"
    harm_series: str = "equal-weight mean of the harm type's scorers' standardized scores"
    cov_type: str = "cluster-robust, clustered by item (prompt); statsmodels cov_type='cluster'"
    resampling_unit: str = (
        "item (prompt), resampled with replacement within each cell; percentile interval"
    )

    n_resamples: int = Field(ge=1)
    ci_level: float = Field(gt=0.0, lt=1.0)
    bootstrap_seed: int = Field(ge=0, le=MAX_SEED)
    min_cell_n: int = Field(ge=1, description="The powered/descriptive threshold used.")
    comparable_ratio: float = Field(gt=0.0, description="PLACEHOLDER drift threshold used.")
    family_disagreement_ratio: float = Field(ge=1.0, description="PLACEHOLDER threshold used.")

    n_pairs: int = Field(ge=0)
    n_effects: int = Field(ge=0)
    n_not_estimable: int = Field(ge=0, description="Fits reported as not estimable (any series).")
    n_drift_fits: int = Field(ge=0)
    descriptive_only_candidates: tuple[str, ...] = Field(
        default=(),
        description=(
            "Harm types with a scorer whose drift interaction is comparable to the model's."
        ),
    )
    n_family_disagreements: int = Field(ge=0)


class StatsRecord(BaseModel):
    """What the statistics stage actually did (P9): which family, which correction, what power.

    The estimand-level facts are FIXED TEXT rather than settings, as :class:`MetricsRecord` does
    it, because none of them is configurable and a reproducer should not have to read the code to
    learn that the p-value is a cluster-robust Wald t on ``item clusters - 1`` degrees of freedom.

    ``family_size`` is here as well as on every corrected estimate because a manifest should be
    answerable on its own: "how many tests was this controlled over" is the first question a
    reviewer asks about an FDR claim, and a gitignored Parquet is not an answer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    stats_name: str
    stats_version: str
    stats_hash: str
    benchmark: str

    # Constants, not settings. Nothing in the config can change them.
    correction_method: str = "benjamini_hochberg; statsmodels multipletests(method='fdr_bh')"
    family_definition: str = (
        "benchmark x model x axis_pair x term x harm_type, over the PRE-REGISTERED CONFIRMATORY "
        "pairs only; exploratory pairs carry no p and no q"
    )
    p_source: str = (
        "two-sided cluster-robust Wald t on estimate/se; df = item clusters - 1, from the "
        "triangulated model-condition fit"
    )
    permutation_definition: str = (
        "B3: whole items exchanged within (template frame, stratum-axis value) blocks; "
        "p = (1 + count) / (B + 1); reported beside the GLM p and never in the FDR family"
    )

    alpha: float = Field(gt=0.0, lt=1.0)
    method: str
    prds_justification: str
    family_terms: str
    family_size: int = Field(ge=0)
    n_present: int = Field(ge=0)
    n_excused_not_estimable: int = Field(ge=0)

    n_significant_pre: int = Field(ge=0)
    n_significant_post: int = Field(ge=0)
    n_exploratory_passthrough: int = Field(ge=0)

    permutation_scheme: str
    n_permutations: int = Field(ge=0)
    permutation_seed: int = Field(ge=0, le=MAX_SEED)

    descriptive_only_ratio: float = Field(gt=0.0)
    descriptive_only_harm_types: tuple[str, ...] = ()
    n_drift_disagreements: int = Field(
        default=0, description="Where P9's enforced decision differs from P7's candidate flag."
    )

    icc_rho: tuple[float, ...] = Field(
        default=(), description="The rho strip the power analysis ran at. MEASURED, not assumed."
    )
    effect_size: float = Field(gt=0.0, description="The pre-registered minimum detectable effect.")
    target_power: float = Field(gt=0.0, lt=1.0)
    min_cell_n: int = Field(ge=1, description="The threshold this run judged `powered` against.")
    n_claimable: int = Field(ge=0)
    n_refused: int = Field(ge=0)


class RedTeamRecord(BaseModel):
    """What the red-team generation stage actually did (P8).

    Carries the **generator pin**, not only its name, for the same reason :class:`ScorerRecord`
    carries a scorer's: "which red LM" is not answerable from a config name alone, and a re-pinned
    generator is a different instrument producing a different prompt *distribution* — which is the
    contribution of P8, so it is the thing a manifest most needs to state.

    The rejection counts are here rather than only in the gitignored report because gate P8 is
    "dedup + filter logged", and a run whose filter yield is knowable only from a regenerable
    artifact cannot be audited from its manifest alone. The same argument put
    ``agreement_summary`` on :class:`ScoringRecord`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    redteam_name: str
    redteam_version: str
    benchmark: str = Field(description="The AUDIT source label, i.e. the tag E3 reads.")

    generator_model_name: str
    generator_model_id: str
    generator_revision: str = Field(description="The RESOLVED commit sha, never a branch.")
    generator_decoding_key: str
    generator_decoding_hash: str
    audit_decoding_key: str = Field(
        description="What the TARGET model runs the audit prompts under. Matched to the static arm."
    )

    base_prompt_set: str = Field(description="Where the carriers and article rule were inherited.")
    base_template_version: str

    n_target_cells: int = Field(ge=0)
    n_elicitation_prompts: int = Field(ge=0)
    n_responses: int = Field(ge=0, description="Red LM generations read.")
    n_candidates: int = Field(ge=0, description="Frame lines parsed out of those generations.")
    n_rejected: int = Field(ge=0)
    rejections_by_reason: dict[str, int] = Field(
        default_factory=dict,
        description="Closed vocabulary; see margins.redteam.filter.REJECTION_REASONS.",
    )
    n_frames: int = Field(ge=0, description="Frames that rendered cleanly into EVERY target cell.")
    n_audit_prompts: int = Field(ge=0)
    n_on_target_sample: int = Field(
        default=0, ge=0, description="Frames drawn for the human on-target audit."
    )


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
    quant: QuantSetting = Field(description="The INTENDED constant, from the config.")
    inference: InferenceRecord | None = Field(
        default=None,
        description=(
            "What the inference stage actually did, including the quantization it resolved to. "
            "None for runs that load no model, such as prompt generation."
        ),
    )
    scoring: ScoringRecord | None = Field(
        default=None,
        description=(
            "What the scoring stage actually did, including every scorer's pin and the agreement "
            "it achieved. None for runs that score nothing, such as inference or prompt generation."
        ),
    )
    metrics: MetricsRecord | None = Field(
        default=None,
        description=(
            "What the metric stage actually did: which estimator, under which thresholds. None "
            "for runs that fit nothing."
        ),
    )
    stats: StatsRecord | None = Field(
        default=None,
        description=(
            "What the statistics stage actually did: which family, which correction, what power. "
            "None for runs that declare no significance, which is most of them."
        ),
    )
    redteam: RedTeamRecord | None = Field(
        default=None,
        description=(
            "What the red-team generation stage actually did: which red LM, asked what, and what "
            "survived the filter. None for runs that generate no frames, which is most of them."
        ),
    )


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
    run_id: str | None = None,
    inference: InferenceRecord | None = None,
    scoring: ScoringRecord | None = None,
    metrics: MetricsRecord | None = None,
    stats: StatsRecord | None = None,
    redteam: RedTeamRecord | None = None,
) -> RunManifest:
    """Write ``<results_dir>/<manifest_subdir>/<run_id>.json`` and return what was written.

    Args:
        cfg: The validated config for this run.
        inputs: ``{source_name: sha256}`` for every input consumed. Empty until P2 has
            sources to checksum.
        timestamp: Recorded creation time; defaults to now (UTC). Pass it explicitly to make
            ``run_id`` reproducible.
        repo_root: Where to ask git about the worktree. Defaults to the package's repository.
        run_id: Reuse an existing run id instead of deriving a fresh one. This is what makes an
            inference run **resumable**: a killed sweep restarts against the same response file and
            the same manifest, rather than orphaning both. Still recorded, never random.
        inference: What the inference stage actually did. None for runs that load no model.
        scoring: What the scoring stage actually did. None for runs that score nothing.
        metrics: What the metric stage actually did. None for runs that fit nothing.
        stats: What the statistics stage actually did. None for runs that declare nothing.
        redteam: What the red-team generation stage actually did. None for runs that generate
            no frames.

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
    resolved_run_id = run_id if run_id is not None else make_run_id(cfg, digest, created)
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[2]
    git_sha, git_dirty = _git_state(root)

    manifest = RunManifest(
        run_id=resolved_run_id,
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
        inference=inference,
        scoring=scoring,
        metrics=metrics,
        stats=stats,
        redteam=redteam,
    )

    destination = cfg.results_dir / cfg.manifest.manifest_subdir / f"{resolved_run_id}.json"
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
