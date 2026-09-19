"""Gate P9: the stats config is pre-registration material and refuses to be vague.

The shipped config decides what may be called significant (``alpha``, the family definition) and
what may be called anything at all (``descriptive_only_ratio``). Two properties matter more than
the values themselves:

1. the file's key matches its filename, so a manifest recording a config NAME resolves to the file
   that actually ran;
2. **choosing ``fdr_bh`` without writing the PRDS justification is a LOAD FAILURE.** Section 8
   says BH assumes independence or PRDS and that the project must state the assumption or use
   ``fdr_by``. An unstated assumption discovered by a reviewer is a desk reject; an unstated
   assumption caught by the loader is a config edit.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from margins.stats.config import (
    DEFAULT_STATS,
    DriftEnforcement,
    FdrSettings,
    PermutationSettings,
    PowerSettings,
    StatsConfig,
    default_stats_config_dir,
    load_stats_config,
    stats_config_from_mapping,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "stats"


def make_power(**overrides: object) -> PowerSettings:
    data: dict[str, object] = {
        "frames_per_cell": 9,
        "frames_grid": (9, 12),
        "seeds_grid": (1, 2, 4),
        "icc_rho": (0.1, 0.3),
        "effect_size": 0.3,
        "patterns": ("concentrated",),
        "target_power": 0.8,
        "n_simulations": 100,
        "pair": ("alpha", "beta"),
        "seed": 7,
    }
    data.update(overrides)
    return PowerSettings.model_validate(data)


def make_config(**overrides: object) -> StatsConfig:
    data: dict[str, object] = {
        "key": "test",
        "description": "test config",
        "stats_version": "test@2026-09-19",
        "fdr": FdrSettings(alpha=0.05, prds_justification="stated for the test"),
        "permutation": PermutationSettings(n_permutations=199, seed=1),
        "power": make_power(),
        "drift": DriftEnforcement(descriptive_only_ratio=0.5),
    }
    data.update(overrides)
    return StatsConfig.model_validate(data)


# ------------------------------------------------------------------ the shipped config


def test_shipped_config_loads() -> None:
    config = load_stats_config(DEFAULT_STATS)
    assert config.key == DEFAULT_STATS
    assert config.fdr.alpha == 0.05
    assert config.fdr.method == "fdr_bh"
    assert config.fdr.family_terms == "free_interaction"
    assert config.drift.descriptive_only_ratio == 0.5
    # 27 prompts per cell and a 0.5 SD target: both DERIVED on 19 Sep 2026 from the measured
    # intraclass correlation, not chosen. See docs/power_and_precision.md.
    assert config.power.frames_per_cell == 27
    assert config.power.effect_size == 0.5
    assert config.power.target_power == 0.8


def test_shipped_config_states_prds() -> None:
    """Section 8's requirement, met in prose rather than assumed."""
    config = load_stats_config(DEFAULT_STATS)
    assert len(config.fdr.prds_justification.strip()) > 200
    assert "PRDS" in config.fdr.prds_justification
    # BY must be reported alongside, which is the other half of section 8's either/or.
    assert config.fdr.report_fdr_by is True


def test_shipped_config_lives_where_the_loader_looks() -> None:
    assert (default_stats_config_dir() / f"{DEFAULT_STATS}.yaml").is_file()


def test_key_must_match_filename() -> None:
    with pytest.raises(ValueError, match="declares key"):
        load_stats_config("mismatched_key", config_dir=FIXTURE_DIR)


def test_hash_tracks_content() -> None:
    base = make_config()
    assert base.stats_hash == make_config().stats_hash
    moved = make_config(drift=DriftEnforcement(descriptive_only_ratio=0.4))
    assert moved.stats_hash != base.stats_hash


# ------------------------------------------------------------------ the PRDS guard


def test_fdr_bh_without_a_prds_justification_fails_to_load() -> None:
    with pytest.raises(ValidationError, match="prds_justification"):
        FdrSettings(alpha=0.05, method="fdr_bh", prds_justification="   ")


def test_fdr_by_needs_no_justification() -> None:
    """BY carries the unconditional guarantee, so there is no assumption left to state."""
    settings = FdrSettings(alpha=0.05, method="fdr_by")
    assert settings.prds_justification == ""


# ------------------------------------------------------------------ validation


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_alpha_must_be_a_probability(alpha: float) -> None:
    with pytest.raises(ValidationError):
        FdrSettings(alpha=alpha, prds_justification="stated")


def test_permutation_needs_enough_replicates() -> None:
    """The smallest attainable p is 1/(B+1); a coarser null cannot resolve alpha=0.05."""
    with pytest.raises(ValidationError):
        PermutationSettings(n_permutations=100, seed=1)


def test_frames_per_cell_must_be_priced() -> None:
    with pytest.raises(ValidationError, match="must appear in frames_grid"):
        make_power(frames_per_cell=9, frames_grid=(12, 18))


def test_icc_rho_must_be_a_variance_share() -> None:
    with pytest.raises(ValidationError, match="icc_rho"):
        make_power(icc_rho=(0.3, 1.0))


def test_pair_must_name_two_axes() -> None:
    with pytest.raises(ValidationError, match="distinct axes"):
        make_power(pair=("alpha", "alpha"))


def test_config_is_frozen_and_forbids_extras() -> None:
    config = make_config()
    with pytest.raises(ValidationError):
        config.fdr.alpha = 0.1  # type: ignore[misc]
    with pytest.raises(ValidationError):
        stats_config_from_mapping({**config.model_dump(), "surprise": 1})
