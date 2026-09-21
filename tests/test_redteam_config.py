"""The red-team declaration validates what it claims to (P8).

The red-team prompt *distribution* is what P8 contributes, and `docs/preregistration.md` section 11
deliberately does not pre-register it -- so the config file plus the run manifest are the only
records of it. A config that accepted a distribution nobody declared would leave the contribution
undocumented, which is why these are load failures rather than review items.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from margins.config import DEFAULT_REDTEAM_NAME, ExperimentConfig
from margins.inference.models import load_model_config
from margins.prompts.config import load_prompt_config
from margins.redteam import (
    DEFAULT_REDTEAM_SET,
    REDTEAM_BENCHMARK,
    RedTeamConfig,
    default_elicitation_dir,
    default_redteam_config_dir,
    load_base_prompt_set,
    load_redteam_config,
)
from margins.redteam.config import load_elicitation_files, redteam_config_from_mapping

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "redteam"


def _fixture() -> RedTeamConfig:
    return load_redteam_config("tiny", config_dir=FIXTURE_DIR)


def _payload() -> dict[str, object]:
    return dict(_fixture().model_dump(mode="json"))


# ----------------------------------------------------------------------------------------
# The shipped set
# ----------------------------------------------------------------------------------------


def test_shipped_set_loads_and_pins_its_generator() -> None:
    """The shipped red-team set resolves, and its generator is a real pinned checkpoint."""
    config = load_redteam_config()
    assert config.key == DEFAULT_REDTEAM_SET
    assert config.redteam_version.startswith("cell_targeted_v1@")

    spec = load_model_config(config.generator.model_name)
    # ModelSpec already refuses a branch name; this asserts the red-team set names a real pin
    # rather than a repo id, because a generator that can move is not a recorded constant.
    assert len(spec.revision) == 40
    assert spec.model_id


def test_default_name_matches_the_config_module() -> None:
    """``margins.config`` keeps a literal copy to avoid an import cycle; they must agree."""
    assert DEFAULT_REDTEAM_NAME == DEFAULT_REDTEAM_SET


def test_experiment_config_carries_an_optional_redteam_name() -> None:
    """Optional and None by default: most runs generate no frames (the `benchmark_name` shape)."""
    assert "redteam_name" in ExperimentConfig.model_fields
    field = ExperimentConfig.model_fields["redteam_name"]
    assert field.default is None
    # `redteam`, the manifest BLOCK, must never become a config field: it records what a run did.
    assert "redteam" not in ExperimentConfig.model_fields


def test_shipped_elicitation_templates_exist_and_are_digestible() -> None:
    config = load_redteam_config()
    files = load_elicitation_files(config.elicitation.templates, default_elicitation_dir())
    assert len(files) == len(config.elicitation.templates)
    for file in files:
        assert len(file.sha256) == 64
        assert config.elicitation.slot_token not in file.text, (
            "an elicitation template must reference `slot`, not hard-code the token: the config "
            "is the single authority on the placeholder"
        )


def test_filename_and_key_must_agree() -> None:
    """A mismatch would put a set name in the manifest that resolves to a different file."""
    with pytest.raises(ValueError, match="filename and the key must agree"):
        load_redteam_config("tiny_wrong_key", config_dir=FIXTURE_DIR)


# ----------------------------------------------------------------------------------------
# Inheritance from the static prompt set
# ----------------------------------------------------------------------------------------


def test_carriers_and_articles_are_inherited_not_redeclared() -> None:
    """E3 compares matched cells, so a red-team carrier of its own would confound the comparison."""
    config = load_redteam_config()
    base = load_base_prompt_set(config)
    prompt_set = config.prompt_set(base)

    assert prompt_set.carriers == base.carriers
    assert prompt_set.article_overrides == base.article_overrides
    assert prompt_set.base_template_version == base.template_version
    # The article RULE is shared, not copied: both route through `indefinite_article`.
    for phrase in ("older blind", "autistic Asian", "wealthy young"):
        assert prompt_set.article_for(phrase) == base.article_for(phrase)


def test_the_red_team_set_stamps_its_own_version_not_the_base_set_s() -> None:
    """Red-team prompts must be distinguishable from static ones in an artifact."""
    config = load_redteam_config()
    base = load_base_prompt_set(config)
    prompt_set = config.prompt_set(base)

    assert prompt_set.template_version == config.redteam_version
    assert prompt_set.template_version != base.template_version


def test_audit_prompts_carry_the_redteam_benchmark_label() -> None:
    """The E3 tag. `item_benchmark` is a property, not a config field, so YAML cannot relabel it."""
    config = load_redteam_config()
    prompt_set = config.prompt_set(load_base_prompt_set(config))
    assert prompt_set.item_benchmark == REDTEAM_BENCHMARK
    assert "item_benchmark" not in type(prompt_set).model_fields

    static = load_prompt_config()
    assert static.item_benchmark == "open_gen_templates"


# ----------------------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------------------


def test_few_shot_examples_must_contain_the_slot_token() -> None:
    """Showing the model an example without the slot teaches it to omit the slot."""
    payload = _payload()
    elicitation = dict(payload["elicitation"])  # type: ignore[arg-type]
    elicitation["few_shot"] = ["Write a paragraph about someone."]
    payload["elicitation"] = elicitation
    with pytest.raises(ValueError, match="does not contain the slot token"):
        redteam_config_from_mapping(payload)


def test_few_shot_may_not_be_empty() -> None:
    payload = _payload()
    elicitation = dict(payload["elicitation"])  # type: ignore[arg-type]
    elicitation["few_shot"] = []
    payload["elicitation"] = elicitation
    with pytest.raises(ValueError, match="few_shot must be non-empty"):
        redteam_config_from_mapping(payload)


def test_word_bounds_must_be_ordered() -> None:
    payload = _payload()
    payload["filter"] = {"min_words": 30, "max_words": 10, "refusal_patterns": []}
    with pytest.raises(ValueError, match=r"min_words .* must be < max_words"):
        redteam_config_from_mapping(payload)


def test_target_pairs_cannot_be_widened_past_confirmatory() -> None:
    """Gate P8 restricts red-teaming to the confirmatory axis pairs, as a LOAD failure."""
    payload = _payload()
    payload["target"] = {"pairs": "all", "include_order_1": False}
    with pytest.raises(ValueError):
        redteam_config_from_mapping(payload)


def test_duplicate_elicitation_ids_are_refused() -> None:
    payload = _payload()
    elicitation = dict(payload["elicitation"])  # type: ignore[arg-type]
    spec = dict(elicitation["templates"][0])  # type: ignore[index]
    elicitation["templates"] = [spec, dict(spec)]
    payload["elicitation"] = elicitation
    with pytest.raises(ValueError, match="ids must be unique"):
        redteam_config_from_mapping(payload)


def test_unknown_keys_are_forbidden() -> None:
    payload = _payload()
    payload["temperature"] = 0.9
    with pytest.raises(ValueError):
        redteam_config_from_mapping(payload)


def test_config_is_frozen() -> None:
    config = _fixture()
    with pytest.raises(ValueError):
        config.redteam_version = "other"  # type: ignore[misc]


def test_authored_at_is_recorded_not_read_from_a_clock() -> None:
    """A clock here would make frames_v1.jsonl non-deterministic through frame provenance."""
    config = load_redteam_config()
    assert config.authored_at < datetime.now(UTC)
    provenance = config.provenance_for("a" * 64, model_id="m", revision="r")
    assert provenance.retrieved_at == config.authored_at


def test_missing_elicitation_file_raises_rather_than_being_skipped() -> None:
    """A silently skipped angle changes the distribution without saying so."""
    payload = _payload()
    elicitation = dict(payload["elicitation"])  # type: ignore[arg-type]
    elicitation["templates"] = [
        {"id": "ghost", "file": "ghost.jinja", "note": "not on disk"},
    ]
    payload["elicitation"] = elicitation
    config = redteam_config_from_mapping(payload)
    with pytest.raises(FileNotFoundError, match="declares missing file"):
        load_elicitation_files(config.elicitation.templates, FIXTURE_DIR / "templates")


def test_default_config_dir_points_at_the_repo_configs() -> None:
    assert default_redteam_config_dir().name == "redteam"
    assert (default_redteam_config_dir() / f"{DEFAULT_REDTEAM_SET}.yaml").is_file()
    assert default_elicitation_dir().is_dir()
