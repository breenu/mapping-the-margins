"""Extractive benchmark ingestion and construct-validity screening (P2).

**Extractive sources only.** HolisticBias is a descriptor/value taxonomy, not a benchmark, and
must never appear here: ``open_gen`` is a *generation* path, not an ingestion one. It supplies
lattice axis values (P3) and template slot fillers (P4a) instead.

The confirmatory scope contains exactly one extractive source, CrowS-Pairs. WinoBias, StereoSet and
BBQ-calibration are out of scope unless ahead of schedule, and BBQ's intersectional subsets remain
calibration-only in any case (framing constraint 5).
"""

from __future__ import annotations

from margins.benchmarks.config import (
    BENCHMARK_INPUT_KEY,
    BENCHMARK_SOURCE_PREFIX,
    DEFAULT_BENCHMARK,
    AttributeMap,
    BenchmarkConfig,
    BenchmarkSource,
    ScreeningSettings,
    benchmark_config_from_mapping,
    default_benchmark_config_dir,
    file_sha256,
    load_benchmark_config,
)
from margins.benchmarks.crows_pairs import (
    CROWS_PAIRS_RELEASE,
    CrowsPairsPayload,
    CrowsPairsRelease,
    CrowsRow,
    crows_pairs_path,
    read_rows,
    source_rows,
)
from margins.benchmarks.loader import CrowsPairsLoader
from margins.benchmarks.screening import (
    Exclusion,
    ExclusionReason,
    KeptWithoutCell,
    ScreeningResult,
    screen,
)

__all__ = [
    "BENCHMARK_INPUT_KEY",
    "BENCHMARK_SOURCE_PREFIX",
    "CROWS_PAIRS_RELEASE",
    "DEFAULT_BENCHMARK",
    "AttributeMap",
    "BenchmarkConfig",
    "BenchmarkSource",
    "CrowsPairsLoader",
    "CrowsPairsPayload",
    "CrowsPairsRelease",
    "CrowsRow",
    "Exclusion",
    "ExclusionReason",
    "KeptWithoutCell",
    "ScreeningResult",
    "ScreeningSettings",
    "benchmark_config_from_mapping",
    "crows_pairs_path",
    "default_benchmark_config_dir",
    "file_sha256",
    "load_benchmark_config",
    "read_rows",
    "screen",
    "source_rows",
]
