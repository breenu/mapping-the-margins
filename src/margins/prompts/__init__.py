"""Prompt generation (P4a): the templated ``open_gen`` path.

``TemplateRenderer`` **injects** a lattice cell into a neutral template frame — the only renderer
that can reach arbitrary lattice cells, and therefore the only path to the non-gender x non-gender
coverage that is novelty claim 5. It is the first stage that produces a
:class:`~margins.schemas.Prompt`, and it starts the generation vertical that carries RQ2.

``ExtractiveRenderer`` (P4b) **derives** a cell from a source item and rides with the CrowS-Pairs
vertical. Added 20 Sep 2026 together with the resolution of open decision 7: a paired prompt
carries BOTH sequences on one record, mirroring decision 6's one ``Response`` carrying both
log-probabilities.
"""

from __future__ import annotations

from margins.prompts.audit import (
    FILL_AUDIT_SAMPLE_SIZE,
    PHRASE_REVIEW_NOTES,
    PhraseReviewNote,
    cell_surfaces,
    mechanical_flags,
    render_phrase_audit,
    sample_for_fill_audit,
)
from margins.prompts.config import (
    DEFAULT_EXTRACTIVE_SET,
    DEFAULT_PROMPT_SET,
    EXTRACTIVE_SET_INPUT_KEY,
    MAX_CARRIERS,
    PROMPT_SET_INPUT_KEY,
    TEMPLATE_INPUT_PREFIX,
    CarrierSpec,
    ExtractivePromptSetConfig,
    PromptSetConfig,
    TemplateFile,
    TemplateSpec,
    default_prompt_config_dir,
    extractive_config_from_mapping,
    file_sha256,
    load_extractive_config,
    load_prompt_config,
    load_template_files,
    prompt_config_from_mapping,
)
from margins.prompts.export import (
    FILL_AUDIT_FILENAME,
    PHRASE_AUDIT_FILENAME,
    PROMPTS_FILENAME,
    export_prompt_set,
    prompt_set_inputs,
    read_prompts,
    render_fill_audit,
    render_prompts_jsonl,
    write_fill_audit,
    write_phrase_audit,
    write_prompts,
)
from margins.prompts.extractive import ExtractiveRenderer, make_extractive_prompt_id
from margins.prompts.render import (
    TEMPLATE_ITEM_BENCHMARK,
    TemplateRenderer,
    axis_vocabulary,
    identity_vocabulary,
    make_prompt_id,
    matched_terms,
    template_item_id,
    template_items,
    template_key,
)

__all__ = [
    "DEFAULT_EXTRACTIVE_SET",
    "DEFAULT_PROMPT_SET",
    "EXTRACTIVE_SET_INPUT_KEY",
    "FILL_AUDIT_FILENAME",
    "FILL_AUDIT_SAMPLE_SIZE",
    "MAX_CARRIERS",
    "PHRASE_AUDIT_FILENAME",
    "PHRASE_REVIEW_NOTES",
    "PROMPTS_FILENAME",
    "PROMPT_SET_INPUT_KEY",
    "TEMPLATE_INPUT_PREFIX",
    "TEMPLATE_ITEM_BENCHMARK",
    "CarrierSpec",
    "ExtractivePromptSetConfig",
    "ExtractiveRenderer",
    "PhraseReviewNote",
    "PromptSetConfig",
    "TemplateFile",
    "TemplateRenderer",
    "TemplateSpec",
    "axis_vocabulary",
    "cell_surfaces",
    "default_prompt_config_dir",
    "export_prompt_set",
    "extractive_config_from_mapping",
    "file_sha256",
    "identity_vocabulary",
    "load_extractive_config",
    "load_prompt_config",
    "load_template_files",
    "make_extractive_prompt_id",
    "make_prompt_id",
    "matched_terms",
    "mechanical_flags",
    "prompt_config_from_mapping",
    "prompt_set_inputs",
    "read_prompts",
    "render_fill_audit",
    "render_phrase_audit",
    "render_prompts_jsonl",
    "sample_for_fill_audit",
    "template_item_id",
    "template_items",
    "template_key",
    "write_fill_audit",
    "write_phrase_audit",
    "write_prompts",
]
