"""Descriptor/value sources feeding the lattice.

A descriptor source supplies **identity terms with provenance**, never items. Nothing here is a
``BenchmarkLoader``: ``open_gen`` is a generation path, not an ingestion one
(PROJECT_CONTEXT.md section 5).
"""

from __future__ import annotations

from margins.lattice.descriptors.holistic_bias import (
    DEFAULT_DESCRIPTOR_PATH,
    HOLISTIC_BIAS_V1_1,
    DescriptorIndex,
    DescriptorTerm,
    HolisticBiasRelease,
    descriptor_path,
    load_descriptor_index,
    verify_lattice_values,
)

__all__ = [
    "DEFAULT_DESCRIPTOR_PATH",
    "HOLISTIC_BIAS_V1_1",
    "DescriptorIndex",
    "DescriptorTerm",
    "HolisticBiasRelease",
    "descriptor_path",
    "load_descriptor_index",
    "verify_lattice_values",
]
