"""The metrics config: bootstrap settings and the drift thresholds, validated at load (P7).

Two of the numbers in this config are PLACEHOLDERS -- ``comparable_ratio`` and
``family_disagreement_ratio`` -- and they decide which harm types are flagged descriptive-only.
They are pre-registration material (TODO(P9)), so they live in a versioned YAML whose sha256 the
manifest records, not in a Python literal. These tests pin the shape and the bounds, not the values.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from margins.metrics.config import (
    DEFAULT_METRICS,
    BootstrapSettings,
    DriftThresholds,
    MetricsConfig,
    load_metrics_config,
    metrics_config_from_mapping,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "metrics"


def _payload(config: MetricsConfig) -> dict[str, Any]:
    return copy.deepcopy(config.model_dump(mode="json"))


def test_the_shipped_config_loads() -> None:
    config = load_metrics_config(DEFAULT_METRICS)
    assert config.key == DEFAULT_METRICS
    assert config.bootstrap.n_resamples >= 1000  # a headline CI is not built on 50 replicates
    assert 0.0 < config.bootstrap.ci_level < 1.0
    assert config.drift.comparable_ratio > 0.0
    assert config.drift.family_disagreement_ratio >= 1.0


def test_the_fixture_config_loads() -> None:
    config = load_metrics_config("tiny", config_dir=FIXTURE_DIR)
    assert config.key == "tiny"
    assert config.bootstrap.n_resamples == 50


def test_the_key_must_match_the_filename() -> None:
    """A manifest names the config by key; a key pointing at another file would mislead it."""
    with pytest.raises(ValueError, match="declares key"):
        load_metrics_config("wrong_key", config_dir=FIXTURE_DIR)


@pytest.mark.parametrize("ci_level", [0.0, 1.0, 1.5, -0.1])
def test_ci_level_is_strictly_inside_the_unit_interval(ci_level: float) -> None:
    with pytest.raises(ValidationError):
        BootstrapSettings(n_resamples=10, ci_level=ci_level, seed=1)


def test_at_least_one_resample_is_required() -> None:
    """An EffectEstimate must carry a CI; zero replicates cannot produce one."""
    with pytest.raises(ValidationError):
        BootstrapSettings(n_resamples=0, ci_level=0.95, seed=1)


def test_thresholds_are_bounded() -> None:
    with pytest.raises(ValidationError):
        DriftThresholds(comparable_ratio=0.0)
    with pytest.raises(ValidationError):
        DriftThresholds(family_disagreement_ratio=0.5)


def test_the_hash_tracks_content() -> None:
    config = load_metrics_config("tiny", config_dir=FIXTURE_DIR)
    same = metrics_config_from_mapping(_payload(config))
    assert same.metrics_hash == config.metrics_hash

    changed = _payload(config)
    changed["bootstrap"]["n_resamples"] = 51
    assert metrics_config_from_mapping(changed).metrics_hash != config.metrics_hash


def test_the_config_is_frozen_and_forbids_extras() -> None:
    config = load_metrics_config("tiny", config_dir=FIXTURE_DIR)
    with pytest.raises(ValidationError):
        config.__setattr__("key", "other")
    with pytest.raises(ValidationError):
        metrics_config_from_mapping({**_payload(config), "unexpected": 1})
