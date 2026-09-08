"""Model inference (P5): the pinned backend and the resumable run harness.

This is the only GPU-touching, non-portable stage, and the one that turns P4a's bare
``decoding_key = "greedy_v1"`` into a resolved, hashed, recorded decoding profile.

Import layering matters here. ``decoding`` and ``models`` are pure configuration and import no
model stack at all, so a manifest, a test, or the Windows core environment can read them with no
``torch`` installed. ``backend`` imports ``torch``/``transformers`` **lazily**, inside the functions
that need them, for the same reason :mod:`margins.repro` does: ``mypy src``, ``ruff`` and the
non-inference suite all have to pass on a machine with no inference extra.
"""

from __future__ import annotations

from margins.inference.backend import (
    HFBackend,
    LoadReport,
    SequenceLogprob,
    make_response_id,
)
from margins.inference.decoding import (
    DECODING_INPUT_KEY,
    DEFAULT_DECODING,
    DecodingConfig,
    decoding_config_from_mapping,
    default_decoding_config_dir,
    load_decoding_config,
)
from margins.inference.harness import (
    DEFAULT_RESPONSES_DIR,
    GenerativeBackend,
    RunHarness,
    RunResult,
    RunTask,
    read_responses,
    response_path,
)
from margins.inference.manifest import inference_inputs, inference_record
from margins.inference.models import (
    CHAT_TEMPLATE_INPUT_PREFIX,
    CI_MODEL,
    CONFIRMATORY_MODELS,
    DEV_MODEL,
    MODEL_INPUT_KEY,
    ModelSpec,
    default_model_config_dir,
    load_model_config,
    model_config_from_mapping,
)

__all__ = [
    "CHAT_TEMPLATE_INPUT_PREFIX",
    "CI_MODEL",
    "CONFIRMATORY_MODELS",
    "DECODING_INPUT_KEY",
    "DEFAULT_DECODING",
    "DEFAULT_RESPONSES_DIR",
    "DEV_MODEL",
    "MODEL_INPUT_KEY",
    "DecodingConfig",
    "GenerativeBackend",
    "HFBackend",
    "LoadReport",
    "ModelSpec",
    "RunHarness",
    "RunResult",
    "RunTask",
    "SequenceLogprob",
    "decoding_config_from_mapping",
    "default_decoding_config_dir",
    "default_model_config_dir",
    "inference_inputs",
    "inference_record",
    "load_decoding_config",
    "load_model_config",
    "make_response_id",
    "model_config_from_mapping",
    "read_responses",
    "response_path",
]
