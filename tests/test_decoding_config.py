"""The resolved decoding profile and its hash (P5).

P4a stamped ``decoding_key = "greedy_v1"`` on 990 prompts with nothing behind the name. These tests
pin what is behind it now, and pin the two properties that make a ``Response`` interpretable
without its config: the hash covers everything that decides what the model was asked, and a profile
cannot be half-greedy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from margins.inference.decoding import (
    DEFAULT_DECODING,
    DecodingConfig,
    decoding_config_from_mapping,
    default_decoding_config_dir,
    load_decoding_config,
)


def _greedy(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "key": "greedy_test",
        "description": "test profile",
        "prompt_format": "chat",
        "pinned_chat_date": "08 Sep 2026",
        "max_input_tokens": 512,
        "do_sample": False,
        "temperature": 0.0,
        "max_new_tokens": 16,
    }
    data.update(overrides)
    return data


# --------------------------------------------------------------------------------------
# The shipped configs
# --------------------------------------------------------------------------------------


def test_shipped_greedy_profile_matches_the_key_p4a_stamps() -> None:
    """The prompt set is already stamped `greedy_v1`; a mismatch here would strand 990 prompts."""
    config = load_decoding_config(DEFAULT_DECODING)
    assert config.key == DEFAULT_DECODING
    assert config.do_sample is False
    assert config.temperature == 0.0


def test_shipped_sampled_profile_is_genuinely_sampled() -> None:
    config = load_decoding_config("sampled_v1")
    assert config.do_sample is True
    assert config.temperature > 0.0


def test_the_e4_temperature_profile_differs_from_e1s_only_in_temperature() -> None:
    """E4 isolates temperature: sampled_t10_v1 must equal sampled_v1 in every other setting."""
    e1 = load_decoding_config("sampled_v1").model_dump()
    e4 = load_decoding_config("sampled_t10_v1").model_dump()
    differing = {name for name in e1 if e1[name] != e4[name]}
    assert differing <= {"key", "description", "temperature"}
    assert (e1["temperature"], e4["temperature"]) == (0.7, 1.0)
    assert e4["top_p"] == e1["top_p"] == 0.9


def test_every_shipped_profile_loads() -> None:
    for path in sorted(default_decoding_config_dir().glob("*.yaml")):
        config = load_decoding_config(path.stem)
        assert config.key == path.stem, "a profile's key must match its filename"


# --------------------------------------------------------------------------------------
# Greedy and sampling cannot be mixed
# --------------------------------------------------------------------------------------


def test_temperature_has_no_default() -> None:
    """The user-facing rule: never let temperature default silently.

    Enforced as a required field rather than as a review item, because a defaulted temperature is
    the easiest way to publish a sampled result while believing it was greedy.
    """
    data = _greedy()
    del data["temperature"]
    with pytest.raises(ValueError, match="temperature"):
        decoding_config_from_mapping(data)


def test_greedy_with_a_nonzero_temperature_is_rejected() -> None:
    """transformers would ignore it, and the Response would record a temperature nothing used."""
    with pytest.raises(ValueError, match="Greedy profiles must set temperature"):
        decoding_config_from_mapping(_greedy(temperature=0.7))


@pytest.mark.parametrize(("field", "value"), [("top_p", 0.9), ("top_k", 50)])
def test_greedy_with_truncation_warpers_is_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError, match="greedy but declares"):
        decoding_config_from_mapping(_greedy(**{field: value}))


def test_sampling_at_temperature_zero_is_rejected() -> None:
    with pytest.raises(ValueError, match="greedy decoding wearing a sampling label"):
        decoding_config_from_mapping(_greedy(do_sample=True, temperature=0.0))


# --------------------------------------------------------------------------------------
# The hash
# --------------------------------------------------------------------------------------


def test_hash_is_stable_across_key_order() -> None:
    """Content-addressed, not order-addressed: the YAML's key order must not change the hash."""
    forward = decoding_config_from_mapping(_greedy())
    reordered = decoding_config_from_mapping(dict(reversed(list(_greedy().items()))))
    assert forward.decoding_hash == reordered.decoding_hash


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_new_tokens", 32),
        ("repetition_penalty", 1.1),
        ("max_input_tokens", 256),
        ("prompt_format", "raw"),
        ("pinned_chat_date", "09 Sep 2026"),
        ("key", "other"),
    ],
)
def test_every_field_that_changes_the_ask_changes_the_hash(field: str, value: object) -> None:
    """The point of the hash: nothing that decides what the model was asked may be invisible."""
    base = decoding_config_from_mapping(_greedy())
    changed = decoding_config_from_mapping(_greedy(**{field: value}))
    assert base.decoding_hash != changed.decoding_hash, f"{field} does not reach the hash"


def test_hash_is_recomputable_from_the_published_preimage() -> None:
    """A reproducer must be able to check the hash, not just read it."""
    import hashlib

    config = decoding_config_from_mapping(_greedy())
    recomputed = hashlib.sha256(config.canonical_json().encode("utf-8")).hexdigest()
    assert recomputed == config.decoding_hash


# --------------------------------------------------------------------------------------
# Generation kwargs
# --------------------------------------------------------------------------------------


def test_greedy_kwargs_do_not_smuggle_a_temperature() -> None:
    """A temperature passed under do_sample=False draws a warning and means nothing."""
    kwargs = decoding_config_from_mapping(_greedy()).generation_kwargs()
    assert kwargs["do_sample"] is False
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs


def test_sampled_kwargs_state_everything_explicitly() -> None:
    """Nothing is left to the checkpoint's bundled generation_config."""
    config = decoding_config_from_mapping(
        _greedy(key="s", do_sample=True, temperature=0.7, top_p=0.9, top_k=50)
    )
    kwargs = config.generation_kwargs()
    assert kwargs == {
        "do_sample": True,
        "max_new_tokens": 16,
        "repetition_penalty": 1.0,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
    }


def test_config_is_frozen_and_forbids_extras() -> None:
    assert DecodingConfig.model_config.get("frozen") is True
    assert DecodingConfig.model_config.get("extra") == "forbid"


def test_missing_config_dir_is_an_explicit_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_decoding_config("greedy_v1", config_dir=tmp_path / "nope")
