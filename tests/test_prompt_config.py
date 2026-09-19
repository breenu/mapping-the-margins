"""The prompt-set config: carrier construct rules, the article rule, and template loading (P4a).

The carrier validators are the interesting ones. A carrier noun that carries identity signal is a
second axis of variation in every cell, and the whole point of typing it is that such a carrier can
only enter the design *declared*, never by accident.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from margins.prompts.config import (
    ARTICLE_RISKY_PREFIXES,
    DEFAULT_PROMPT_SET,
    MAX_CARRIERS,
    TEMPLATE_DIRNAME,
    PromptSetConfig,
    load_prompt_config,
    load_template_files,
    normalize_template_text,
    prompt_config_from_mapping,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "prompts"


def _base(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "template_version": "unit@test",
        "description": "unit test prompt set",
        "license": "MIT",
        "authored_at": "2026-09-08T00:00:00Z",
        "decoding_key": "greedy_test",
        "templates": [{"id": "t1", "file": "t1.jinja", "note": "a frame"}],
        "carriers": [{"noun": "person", "axis_neutral": True, "note": "neutral"}],
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------- carriers


def test_non_neutral_carrier_must_declare_its_axis() -> None:
    """An identity-bearing carrier is a second axis; it enters declared or not at all."""
    with pytest.raises(ValidationError, match="declared_axis"):
        prompt_config_from_mapping(
            _base(carriers=[{"noun": "nurse", "axis_neutral": False, "note": "role carrier"}])
        )


def test_non_neutral_carrier_is_allowed_once_its_axis_is_named() -> None:
    cfg = prompt_config_from_mapping(
        _base(
            carriers=[
                {"noun": "person", "axis_neutral": True, "note": "neutral"},
                {
                    "noun": "nurse",
                    "axis_neutral": False,
                    "declared_axis": "occupation",
                    "note": "declared occupational axis of variation",
                },
            ]
        )
    )
    assert cfg.declared_carrier_axes == ("occupation",)


def test_carrier_cannot_be_both_neutral_and_declare_an_axis() -> None:
    with pytest.raises(ValidationError, match="cannot be both"):
        prompt_config_from_mapping(
            _base(
                carriers=[
                    {
                        "noun": "person",
                        "axis_neutral": True,
                        "declared_axis": "occupation",
                        "note": "contradictory",
                    }
                ]
            )
        )


def test_a_prompt_set_needs_at_least_one_axis_neutral_carrier() -> None:
    with pytest.raises(ValidationError, match="axis-neutral"):
        prompt_config_from_mapping(
            _base(
                carriers=[
                    {
                        "noun": "nurse",
                        "axis_neutral": False,
                        "declared_axis": "occupation",
                        "note": "role",
                    }
                ]
            )
        )


def test_carrier_count_is_capped() -> None:
    """Carriers multiply the prompt set and spend the per-cell rendering budget."""
    carriers = [
        {"noun": f"carrier{index}", "axis_neutral": True, "note": "n"}
        for index in range(MAX_CARRIERS + 1)
    ]
    with pytest.raises(ValidationError, match="exceeds the cap"):
        prompt_config_from_mapping(_base(carriers=carriers))


def test_carrier_note_is_required() -> None:
    with pytest.raises(ValidationError):
        prompt_config_from_mapping(_base(carriers=[{"noun": "person", "axis_neutral": True}]))


# ---------------------------------------------------------------------------- articles


@pytest.mark.parametrize(
    ("descriptor", "expected"),
    [
        ("young", "a"),
        ("middle-aged", "a"),
        ("older", "an"),
        ("older blind", "an"),
        ("autistic Asian", "an"),
        ("working-class Black", "a"),
        ("nondisabled white", "a"),
        ("Asian", "an"),
        ("blind female", "a"),
    ],
)
def test_article_rule_matches_english_for_every_shipped_descriptor_shape(
    descriptor: str, expected: str
) -> None:
    assert prompt_config_from_mapping(_base()).article_for(descriptor) == expected


@pytest.mark.parametrize("prefix", ARTICLE_RISKY_PREFIXES)
def test_article_rule_refuses_to_guess_where_the_letter_rule_is_wrong(prefix: str) -> None:
    """'a European', 'a university', 'an hour': the letter rule is wrong, so it must not guess."""
    with pytest.raises(ValueError, match="article_overrides"):
        prompt_config_from_mapping(_base()).article_for(f"{prefix}xample thing")


def test_article_override_resolves_a_risky_word() -> None:
    cfg = prompt_config_from_mapping(_base(article_overrides={"unemployed": "an"}))
    assert cfg.article_for("unemployed older") == "an"


def test_article_override_values_are_validated() -> None:
    with pytest.raises(ValidationError, match="must be one of"):
        prompt_config_from_mapping(_base(article_overrides={"older": "the"}))


def test_article_override_keys_must_be_lowercase() -> None:
    with pytest.raises(ValidationError, match="lowercased"):
        prompt_config_from_mapping(_base(article_overrides={"Older": "an"}))


def test_empty_descriptor_has_no_article() -> None:
    with pytest.raises(ValueError, match="empty descriptor"):
        prompt_config_from_mapping(_base()).article_for("   ")


# ---------------------------------------------------------------------------- templates


def test_template_ids_and_files_must_be_unique() -> None:
    with pytest.raises(ValidationError, match="unique"):
        prompt_config_from_mapping(
            _base(
                templates=[
                    {"id": "t1", "file": "a.jinja", "note": "n"},
                    {"id": "t1", "file": "b.jinja", "note": "n"},
                ]
            )
        )


def test_task_type_is_open_gen_only() -> None:
    """P4a is the generation path. A paired-task prompt set fails to LOAD, not just review."""
    with pytest.raises(ValidationError):
        prompt_config_from_mapping(_base(task_type="pair_pll"))


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError):
        prompt_config_from_mapping(_base(carrier_noun="person"))


def test_config_is_frozen() -> None:
    cfg = prompt_config_from_mapping(_base())
    with pytest.raises(ValidationError):
        cfg.template_version = "other"  # type: ignore[misc]


def test_missing_template_file_is_an_error_not_a_silent_skip() -> None:
    """Skipping a frame would change the per-cell rendering count without saying so."""
    cfg = prompt_config_from_mapping(_base())
    with pytest.raises(FileNotFoundError, match="missing file"):
        load_template_files(cfg, FIXTURE_DIR / TEMPLATE_DIRNAME)


def test_template_text_is_whitespace_normalized() -> None:
    """So rendered bytes cannot depend on how a file happens to be wrapped or checked out."""
    assert normalize_template_text("  Write   a\n story about\r\n{{ identity }}.\n") == (
        "Write a story about {{ identity }}."
    )


# ---------------------------------------------------------------------------- shipped config


@pytest.fixture(scope="module")
def shipped() -> PromptSetConfig:
    return load_prompt_config(DEFAULT_PROMPT_SET)


def test_shipped_config_declares_only_axis_neutral_carriers(shipped: PromptSetConfig) -> None:
    """Two carriers at v1.2, both axis-neutral, and no undeclared axis among them.

    The count grew from one to two on 19 Sep 2026 because a carrier multiplies PROMPTS PER CELL
    exactly as a frame does, and prompts per cell is what bounds precision at the measured
    intraclass correlation. What must not grow is the number of axes smuggled in: every carrier
    here is `axis_neutral`, and the schema refuses a non-neutral one that does not name the axis
    it introduces.
    """
    assert [carrier.noun for carrier in shipped.carriers] == ["person", "individual"]
    assert all(carrier.axis_neutral for carrier in shipped.carriers)
    assert all(carrier.note.strip() for carrier in shipped.carriers)
    assert shipped.declared_carrier_axes == ()
    assert len(shipped.carriers) <= MAX_CARRIERS


def test_shipped_templates_all_load_and_carry_a_note(shipped: PromptSetConfig) -> None:
    """14 frames at v1.2, up from 9. See the config's version note for why."""
    files = load_template_files(shipped)
    assert len(files) == len(shipped.templates) == 14
    for file in files:
        assert file.text and "\n" not in file.text
        assert file.spec.note.strip()
        assert len(file.sha256) == 64


def test_shipped_config_needs_no_article_overrides(shipped: PromptSetConfig) -> None:
    """The vowel-letter rule is correct for all 18 shipped descriptors; nothing is patched."""
    assert shipped.article_overrides == {}
