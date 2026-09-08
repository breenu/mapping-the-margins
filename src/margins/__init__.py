"""``margins`` — a reproducible instrument for intersectional bias auditing of open-weight LLMs.

The deliverable of this project is the instrument and its rigor, not the finding that
intersectional bias exists (that is settled prior art). Two P0 modules carry the load that
everything downstream leans on:

* :mod:`margins.repro` — seeding and the *honest* statement of what determinism this
  pipeline does and does not buy.
* :mod:`margins.config` — the validated :class:`~margins.config.ExperimentConfig` and the
  run manifest, which is the sole experiment-tracking mechanism in this project.
"""

from margins.config import (
    ExperimentConfig,
    ManifestSettings,
    RunManifest,
    config_from_mapping,
    config_hash,
    write_run_manifest,
)
from margins.repro import (
    EnvironmentFingerprint,
    SeedReport,
    environment_fingerprint,
    residual_nondeterminism_note,
    set_global_seed,
)

__all__ = [
    "EnvironmentFingerprint",
    "ExperimentConfig",
    "ManifestSettings",
    "RunManifest",
    "SeedReport",
    "config_from_mapping",
    "config_hash",
    "environment_fingerprint",
    "residual_nondeterminism_note",
    "set_global_seed",
    "write_run_manifest",
]

__version__ = "0.1.0"
