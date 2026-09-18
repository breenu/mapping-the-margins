"""The roster's rules are enforced at load, not in prose (P6).

CLAUDE.md framing constraint 6 says no harm type may depend on a single classifier, and
PROJECT_CONTEXT.md section 9 says two scorers sharing a mechanism are not independent. Both are the
kind of rule that a refactor can quietly drop when it lives only in a comment, so both are
validators -- and these tests are what stop someone from "fixing" a failing config by relaxing them.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from margins.scoring.config import (
    DEFAULT_SCORING,
    LexiconPin,
    ModelPin,
    ScoringConfig,
    default_scoring_config_dir,
    load_scoring_config,
    scoring_config_from_mapping,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "scoring"
REAL_SHA = "4d6c22e74ba2fdd26bc4f7238f50766b045a0d94"


@pytest.fixture(scope="module")
def tiny() -> ScoringConfig:
    return load_scoring_config("tiny", config_dir=FIXTURE_DIR)


def _payload(config: ScoringConfig) -> dict[str, Any]:
    """A mutable dict of the fixture, for building deliberately-broken variants."""
    return copy.deepcopy(config.model_dump(mode="json"))


# ------------------------------------------------------------------ the shipped roster


def test_the_shipped_roster_loads() -> None:
    """The real config must be loadable, not just the fixture -- pins included."""
    config = load_scoring_config(DEFAULT_SCORING)
    assert config.key == DEFAULT_SCORING
    assert config.harm_types == ("regard", "toxicity")


def test_every_shipped_pin_is_resolved() -> None:
    """No placeholder revision may ship. A zero sha is not a pin."""
    config = load_scoring_config(DEFAULT_SCORING)
    for spec in config.scorers:
        if spec.model_pin is not None:
            assert set(spec.model_pin.revision) != {"0"}, (
                f"{spec.scorer_id} still carries the unresolved placeholder revision; "
                "run scripts/pin_scorers.py --write"
            )
        if spec.lexicon_pin is not None and spec.lexicon_pin.path is not None:
            assert spec.lexicon_pin.sha256 is not None


def test_config_dir_points_at_the_repo_configs() -> None:
    assert (default_scoring_config_dir() / f"{DEFAULT_SCORING}.yaml").is_file()


# ------------------------------------------------------------------ the >=2 rules


def test_a_harm_type_with_one_scorer_fails_to_load(tiny: ScoringConfig) -> None:
    """Framing constraint 6, as a load failure rather than a convention."""
    payload = _payload(tiny)
    payload["scorers"] = [s for s in payload["scorers"] if s["harm_type"] != "regard"] + [
        next(s for s in payload["scorers"] if s["scorer_id"] == "fake_regard_nn")
    ]

    with pytest.raises(ValidationError, match="at least 2 are required"):
        scoring_config_from_mapping(payload)


def test_two_scorers_sharing_a_mechanism_fail_to_load(tiny: ScoringConfig) -> None:
    """The subtler half of the rule: same mechanism means they fail together, and agree while doing
    it."""
    payload = _payload(tiny)
    for scorer in payload["scorers"]:
        if scorer["scorer_id"] == "fake_toxic_lex":
            scorer["mechanism"] = "neural_classifier"
            scorer["lexicon_pin"] = None
            scorer["model_pin"] = {
                "repo": "unitary/toxic-bert",
                "revision": REAL_SHA,
                "license": "apache-2.0",
                "label": "toxic",
                "max_input_tokens": 512,
            }

    with pytest.raises(ValidationError, match="not independent"):
        scoring_config_from_mapping(payload)


def test_duplicate_scorer_ids_fail_to_load(tiny: ScoringConfig) -> None:
    payload = _payload(tiny)
    payload["scorers"][1]["scorer_id"] = payload["scorers"][0]["scorer_id"]

    with pytest.raises(ValidationError, match="unique"):
        scoring_config_from_mapping(payload)


# ------------------------------------------------------------------ pins


def test_a_branch_name_is_not_a_pin() -> None:
    with pytest.raises(ValidationError, match="40-character commit sha"):
        ModelPin(repo="unitary/toxic-bert", revision="main", license="apache-2.0", label="toxic")


def test_a_fetched_lexicon_without_a_digest_is_refused() -> None:
    """A file we retrieve and do not digest can change under us with nothing to say so."""
    with pytest.raises(ValidationError, match="not pinned"):
        LexiconPin(
            name="x", source="https://example.invalid", license="MIT", path=Path("lexicons/x.txt")
        )


def test_a_lexicon_with_neither_a_file_nor_a_version_is_refused() -> None:
    with pytest.raises(ValidationError, match="neither a path"):
        LexiconPin(name="x", source="https://example.invalid", license="MIT")


def test_a_packaged_lexicon_needs_only_a_version() -> None:
    pin = LexiconPin(
        name="vader-lexicon", source="https://example.invalid", license="MIT", version="3.3.2"
    )
    assert pin.sha256 is None and pin.path is None


def test_mechanism_must_match_its_pin(tiny: ScoringConfig) -> None:
    payload = _payload(tiny)
    for scorer in payload["scorers"]:
        if scorer["scorer_id"] == "fake_toxic_nn":
            scorer["model_pin"] = None

    with pytest.raises(ValidationError, match="no model_pin"):
        scoring_config_from_mapping(payload)


# ------------------------------------------------------------------ hashing


def test_scoring_hash_is_stable_and_content_addressed(tiny: ScoringConfig) -> None:
    """Same content, same hash; changed content, changed hash."""
    again = load_scoring_config("tiny", config_dir=FIXTURE_DIR)
    assert tiny.scoring_hash == again.scoring_hash

    payload = _payload(tiny)
    payload["scorers"][0]["threshold"] = 0.75
    assert scoring_config_from_mapping(payload).scoring_hash != tiny.scoring_hash


def test_pin_digest_moves_when_the_instrument_moves(tiny: ScoringConfig) -> None:
    """A changed revision, sign or threshold is a changed instrument and must change the digest."""
    spec = tiny.scorers[0]
    assert spec.model_pin is not None

    moved = spec.model_copy(
        update={"model_pin": spec.model_pin.model_copy(update={"revision": "b" * 40})}
    )
    assert moved.pin_digest != spec.pin_digest

    assert spec.model_copy(update={"sign": -1}).pin_digest != spec.pin_digest
    assert spec.model_copy(update={"threshold": 0.9}).pin_digest != spec.pin_digest


def test_records_are_frozen_and_forbid_extras(tiny: ScoringConfig) -> None:
    with pytest.raises(ValidationError):
        ScoringConfig.model_validate({**_payload(tiny), "unexpected": 1})

    with pytest.raises(ValidationError):
        tiny.scorers[0].__setattr__("sign", -1)


def test_calibration_sample_size_is_bounded_to_the_preregistered_band(tiny: ScoringConfig) -> None:
    """150-300 (section 9). A sample quietly shrunk to 20 would still compute a precision."""
    payload = _payload(tiny)
    payload["calibration"]["sample_size"] = 20
    with pytest.raises(ValidationError):
        scoring_config_from_mapping(payload)

    payload["calibration"]["sample_size"] = 5000
    with pytest.raises(ValidationError):
        scoring_config_from_mapping(payload)


def test_filename_and_key_must_agree() -> None:
    """Otherwise the manifest records a roster name that resolves to a different file.

    ``wrong_key.yaml`` is a fully valid roster whose ``key`` says something else -- so this test
    isolates the filename check rather than tripping over a schema error on the way to it.
    """
    with pytest.raises(ValueError, match="filename and the key must agree"):
        load_scoring_config("wrong_key", config_dir=FIXTURE_DIR)
