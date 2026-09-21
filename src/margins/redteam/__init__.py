"""Automated intersectional red-teaming (P8): cell-targeted prompt generation.

Perez et al. (2022) automated red-teaming by having one LM write test cases for another. That
mechanism is prior art; **what is novel here is that the generation is demographically and
intersectionally targeted** — the red LM is asked for prompts that would expose disparate treatment
of a *named lattice cell*, including the non-gender x non-gender pairs no static benchmark reaches.

**Red-team is a prompt SOURCE, not a new measurement path.** Its output enters the existing P4a
renderer, is scored by the existing P6 roster, is fitted by the existing P7 estimator, and is
corrected by the existing P9 family. Nothing here computes a score, a standard error or a q. The
only thing that distinguishes red-team output downstream is the ``redteam`` benchmark label, which
is exactly the tag the E3 static-vs-red-team comparison reads.

**The red LM writes FRAMES, not finished prompts** — a slot-bearing template that is then rendered
across the whole grid. Two reasons, and the second is decisive:

1. A frame is checked by the *unchanged* P4a leakage guard, so a red-team prompt cannot reach a
   target model with identity signal outside the injected slot.
2. A frame is **one item cluster present in every cell of a pair's grid**. A finished per-cell
   prompt would be a singleton, which is exactly the structure that made CrowS-Pairs not estimable
   (``docs/methods.md`` section 2.5). Frames are what make red-team interactions estimable at all.

See :mod:`margins.redteam.frames` for why a frame is rendered into *every* target cell rather than
only its own — the confound that decision avoids is the one that would fabricate an interaction.
"""

from __future__ import annotations

from margins.redteam.config import (
    DEFAULT_REDTEAM_SET,
    ELICITATION_INPUT_PREFIX,
    REDTEAM_BENCHMARK,
    REDTEAM_ELICITATION_BENCHMARK,
    REDTEAM_SET_INPUT_KEY,
    DedupSettings,
    ElicitationSettings,
    ElicitationTemplateSpec,
    FilterSettings,
    GeneratorSpec,
    OnTargetSettings,
    RedTeamConfig,
    RedTeamPromptSetConfig,
    TargetSettings,
    default_elicitation_dir,
    default_redteam_config_dir,
    load_base_prompt_set,
    load_redteam_config,
    redteam_config_from_mapping,
)
from margins.redteam.elicit import (
    ELICITATION_VARIABLES,
    ElicitationPlan,
    elicitation_items,
    elicitation_prompts,
    elicitation_seeds,
    make_elicitation_prompt_id,
    target_cells,
)
from margins.redteam.export import (
    FRAMES_FILENAME,
    REDTEAM_REPORT_FILENAME,
    REJECTED_FILENAME,
    coverage_rows,
    read_frames,
    redteam_inputs,
    render_frames_jsonl,
    render_redteam_report,
    render_rejected_csv,
    write_frames,
    write_redteam_report,
    write_rejected,
)
from margins.redteam.filter import (
    REJECTION_REASONS,
    FilterOutcome,
    Rejection,
    canonical_form,
    dedup,
    screen_candidates,
)
from margins.redteam.frames import (
    Frame,
    RedTeamFramePayload,
    frame_id,
    frame_items,
    render_audit_prompts,
    to_template_text,
    trial_render,
)
from margins.redteam.on_target import (
    ON_TARGET_REPORT_FILENAME,
    ON_TARGET_SAMPLE_FILENAME,
    OnTargetLabel,
    OnTargetMetric,
    OnTargetUnit,
    on_target_metrics,
    read_on_target_labels,
    render_on_target_report,
    sample_for_on_target,
    write_on_target_sample,
)

__all__ = [
    "DEFAULT_REDTEAM_SET",
    "ELICITATION_INPUT_PREFIX",
    "ELICITATION_VARIABLES",
    "FRAMES_FILENAME",
    "ON_TARGET_REPORT_FILENAME",
    "ON_TARGET_SAMPLE_FILENAME",
    "REDTEAM_BENCHMARK",
    "REDTEAM_ELICITATION_BENCHMARK",
    "REDTEAM_REPORT_FILENAME",
    "REDTEAM_SET_INPUT_KEY",
    "REJECTED_FILENAME",
    "REJECTION_REASONS",
    "DedupSettings",
    "ElicitationPlan",
    "ElicitationSettings",
    "ElicitationTemplateSpec",
    "FilterOutcome",
    "FilterSettings",
    "Frame",
    "GeneratorSpec",
    "OnTargetLabel",
    "OnTargetMetric",
    "OnTargetSettings",
    "OnTargetUnit",
    "RedTeamConfig",
    "RedTeamFramePayload",
    "RedTeamPromptSetConfig",
    "Rejection",
    "TargetSettings",
    "canonical_form",
    "coverage_rows",
    "dedup",
    "default_elicitation_dir",
    "default_redteam_config_dir",
    "elicitation_items",
    "elicitation_prompts",
    "elicitation_seeds",
    "frame_id",
    "frame_items",
    "load_base_prompt_set",
    "load_redteam_config",
    "make_elicitation_prompt_id",
    "on_target_metrics",
    "read_frames",
    "read_on_target_labels",
    "redteam_config_from_mapping",
    "redteam_inputs",
    "render_audit_prompts",
    "render_frames_jsonl",
    "render_on_target_report",
    "render_redteam_report",
    "render_rejected_csv",
    "sample_for_on_target",
    "screen_candidates",
    "target_cells",
    "to_template_text",
    "trial_render",
    "write_frames",
    "write_on_target_sample",
    "write_redteam_report",
    "write_rejected",
]
