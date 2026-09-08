"""``margins`` — a reproducible instrument for intersectional bias auditing of open-weight LLMs.

The deliverable of this project is the instrument and its rigor, not the finding that
intersectional bias exists (that is settled prior art). Four modules carry the load that
everything downstream leans on:

* :mod:`margins.repro` (P0) — seeding and the *honest* statement of what determinism this
  pipeline does and does not buy.
* :mod:`margins.config` (P0) — the validated :class:`~margins.config.ExperimentConfig` and the
  run manifest, which is the sole experiment-tracking mechanism in this project.
* :mod:`margins.schemas` (P1) — the frozen shared records. These *are* the unified metric
  layer: every stage depends on them and on nothing else of any other stage.
* :mod:`margins.interfaces` (P1) — the Protocol per pipeline stage, so sources, scorers, and
  backends stay swappable and fakeable.
"""

from margins.config import (
    ExperimentConfig,
    ManifestSettings,
    RunManifest,
    config_from_mapping,
    config_hash,
    write_run_manifest,
)
from margins.interfaces import (
    BenchmarkLoader,
    InferenceBackend,
    Lattice,
    Metric,
    PromptGenerator,
    Scorer,
    StatsEngine,
)
from margins.repro import (
    EnvironmentFingerprint,
    SeedReport,
    environment_fingerprint,
    residual_nondeterminism_note,
    set_global_seed,
)
from margins.schemas import (
    AxisValue,
    AxisValueRegistry,
    CellStat,
    EffectEstimate,
    FrozenDict,
    IdentityCell,
    Item,
    ItemScore,
    Prompt,
    Provenance,
    Response,
    TaskType,
)

__all__ = [
    "AxisValue",
    "AxisValueRegistry",
    "BenchmarkLoader",
    "CellStat",
    "EffectEstimate",
    "EnvironmentFingerprint",
    "ExperimentConfig",
    "FrozenDict",
    "IdentityCell",
    "InferenceBackend",
    "Item",
    "ItemScore",
    "Lattice",
    "ManifestSettings",
    "Metric",
    "Prompt",
    "PromptGenerator",
    "Provenance",
    "Response",
    "RunManifest",
    "Scorer",
    "SeedReport",
    "StatsEngine",
    "TaskType",
    "config_from_mapping",
    "config_hash",
    "environment_fingerprint",
    "residual_nondeterminism_note",
    "set_global_seed",
    "write_run_manifest",
]

__version__ = "0.1.0"
