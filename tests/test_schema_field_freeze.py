"""Pins the exact field set of every shared record (gate P1).

This is deliberately a brittle test. Adding, renaming, or removing a field on a shared record
is a change to the contract every other phase is written against, and it must be a conscious
edit here rather than a side effect noticed three phases later.

Two guardrails are enforced structurally rather than by review:

* ``CellStat`` may never gain ``p``/``q``/``family_size``/``correction`` (invariant 3).
* No shared record may gain a source-specific field; source quirks live only in
  ``Item.payload`` (invariant 2).
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from margins.schemas import (
    AxisValue,
    CellStat,
    EffectEstimate,
    IdentityCell,
    Item,
    ItemScore,
    Prompt,
    Provenance,
    Response,
)

EXPECTED_FIELDS: dict[type[BaseModel], set[str]] = {
    Provenance: {"source", "revision", "version", "license", "sha256", "retrieved_at"},
    AxisValue: {"axis", "value", "provenance"},
    IdentityCell: {"cell_id", "axes", "order"},
    Item: {"id", "benchmark", "task_type", "payload", "target_axes", "gold", "provenance"},
    Prompt: {
        "prompt_id",
        "item_id",
        "cell_id",
        "rendered_text",
        "task_type",
        "template_version",
        "decoding_key",
    },
    # CONTRACT CHANGE 8 Sep 2026 (open decision 6, option (a)): logprob_stereo and
    # logprob_antistereo added so a pair_pll pair is one Response rather than a second
    # ItemScore shape. Named for what they measure, never for a dataset's columns.
    Response: {
        "response_id",
        "prompt_id",
        "model_id",
        "model_revision",
        "seed",
        "temperature",
        "decoding_key",
        "decoding_hash",
        "quant",
        "text",
        "token_logprobs",
        "logprob_stereo",
        "logprob_antistereo",
    },
    ItemScore: {
        "score_id",
        "response_id",
        "scorer_id",
        "harm_type",
        "raw",
        "standardized",
        "cell_id",
        "benchmark",
        "model_id",
        "seed",
        "temperature",
        "condition",
    },
    CellStat: {
        "cell_id",
        "benchmark",
        "model",
        "harm_type",
        "n",
        "mean_standardized",
        "ci_low",
        "ci_high",
        "powered",
    },
    EffectEstimate: {
        "benchmark",
        "model",
        "axis_pair",
        "term",
        "harm_type",
        "estimate",
        "se",
        "ci_low",
        "ci_high",
        "p",
        "q",
        "powered",
        "family_size",
        "correction",
    },
}

#: Fields naming a specific upstream source. None may appear on a shared record: they belong in
#: ``Item.payload``, typed at the loader boundary.
SOURCE_SPECIFIC_FIELDS = {
    "sent_more",
    "sent_less",
    "bias_type",
    "stereo_antistereo",
    "context",
    "sentence",
    "descriptor",
    "bucket",
    "answer_info",
    "question_polarity",
    "context_condition",
}


@pytest.mark.parametrize(
    ("record", "expected"),
    list(EXPECTED_FIELDS.items()),
    ids=[record.__name__ for record in EXPECTED_FIELDS],
)
def test_field_set_is_pinned(record: type[BaseModel], expected: set[str]) -> None:
    assert set(record.model_fields) == expected, (
        f"{record.__name__}'s field set changed. This is a contract change: update "
        "EXPECTED_FIELDS here and check every phase that reads the record."
    )


@pytest.mark.parametrize("record", list(EXPECTED_FIELDS), ids=[r.__name__ for r in EXPECTED_FIELDS])
def test_no_shared_record_carries_a_source_specific_field(record: type[BaseModel]) -> None:
    """Invariant 2: source quirks live ONLY in Item.payload."""
    offenders = set(record.model_fields) & SOURCE_SPECIFIC_FIELDS
    assert not offenders, f"{record.__name__} leaked source-specific fields: {sorted(offenders)}"


@pytest.mark.parametrize("record", list(EXPECTED_FIELDS), ids=[r.__name__ for r in EXPECTED_FIELDS])
def test_every_shared_record_forbids_extras_and_is_frozen(record: type[BaseModel]) -> None:
    assert record.model_config.get("extra") == "forbid", f"{record.__name__} allows extra fields"
    assert record.model_config.get("frozen") is True, f"{record.__name__} is not frozen"


def test_only_effect_estimate_may_carry_significance() -> None:
    """Invariant 3, structurally: significance lives on exactly one record."""
    significance = {"p", "q", "family_size", "correction"}
    for record, fields in EXPECTED_FIELDS.items():
        overlap = fields & significance
        if record is EffectEstimate:
            assert overlap == significance
        else:
            assert not overlap, f"{record.__name__} must not carry significance fields: {overlap}"
