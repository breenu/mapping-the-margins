"""Manifest wiring for the inference stage (P5).

The mirror image of ``margins.prompts.export.prompt_set_inputs``: P4a closed the "which lattice
produced these prompts" hole, and this closes the "which checkpoint, at which revision, under which
decoding profile, and did the quantization it claims actually happen" hole.

Two halves, matching P4a's shape:

* :func:`inference_record` -- what the run *did*, including ``quant_actual``. The intended constant
  lives on ``ExperimentConfig.quant``; this is what the load resolved to, with a reason when the two
  differ.
* :func:`inference_inputs` -- ``{name: sha256}`` for every file that decided what the model was
  asked and how. The chat template is checksummed **per model**, under
  ``chat_template:<model_name>``, because the two pinned families wrap the same prompt differently
  and that wrapper is part of the input.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from margins.config import InferenceRecord
from margins.inference.backend import LoadReport
from margins.inference.decoding import DECODING_INPUT_KEY, DecodingConfig
from margins.inference.models import CHAT_TEMPLATE_INPUT_PREFIX, MODEL_INPUT_KEY, ModelSpec
from margins.prompts.config import file_sha256

__all__ = ["inference_inputs", "inference_record"]


def inference_record(
    spec: ModelSpec,
    decoding: DecodingConfig,
    report: LoadReport,
    *,
    batch_size: int,
    seeds: Sequence[int],
) -> InferenceRecord:
    """Assemble the manifest's inference block from what the backend reported."""
    return InferenceRecord(
        model_name=spec.name,
        model_id=spec.model_id,
        model_revision=spec.revision,
        quant_requested=report.quant_requested,
        quant_actual=report.quant_actual,
        quant_fallback_reason=report.quant_fallback_reason,
        device=report.device,
        dtype=report.dtype,
        prompt_format=report.prompt_format,
        chat_template_sha256=report.chat_template_sha256,
        decoding_key=decoding.key,
        decoding_hash=decoding.decoding_hash,
        batch_size=batch_size,
        seeds=tuple(seeds),
    )


def inference_inputs(
    *,
    model_config_path: Path,
    decoding_config_path: Path,
    model_name: str,
    chat_template_sha256: str | None,
) -> dict[str, str]:
    """``{source_name: sha256}`` for every input that decided how the model was addressed.

    Merged into the mapping P4a already builds (lattice, prompt set, one entry per template) and
    handed to ``write_run_manifest(inputs=...)``. Together they mean a response file can be traced
    back to the cell inventory, the frames, the checkpoint revision and the decoding profile that
    produced it -- which is the whole of what "reconstruct the run" requires.
    """
    inputs = {
        MODEL_INPUT_KEY: file_sha256(model_config_path),
        DECODING_INPUT_KEY: file_sha256(decoding_config_path),
    }
    if chat_template_sha256 is not None:
        inputs[f"{CHAT_TEMPLATE_INPUT_PREFIX}{model_name}"] = chat_template_sha256
    return inputs
