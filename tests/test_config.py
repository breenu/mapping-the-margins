"""ExperimentConfig validation, the config hash, and the Hydra -> dict -> Pydantic seam."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from margins.config import ExperimentConfig, config_from_mapping, config_hash

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def _base(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {"seed": 42, "run_name": "unit-test"}
    data.update(overrides)
    return data


def test_minimal_config_validates_with_defaults() -> None:
    cfg = config_from_mapping(_base())
    assert cfg.seed == 42
    assert cfg.quant == "nf4"
    assert cfg.model_id is None
    assert cfg.manifest.write_manifest is True


@pytest.mark.parametrize("bad_seed", [-1, 2**32])
def test_seed_bounds_rejected(bad_seed: int) -> None:
    with pytest.raises(ValidationError):
        config_from_mapping(_base(seed=bad_seed))


def test_unknown_field_rejected() -> None:
    """extra='forbid': a typo in YAML must fail loudly, not be silently ignored."""
    with pytest.raises(ValidationError):
        config_from_mapping(_base(temperture=0.7))


def test_unknown_nested_field_rejected() -> None:
    with pytest.raises(ValidationError):
        config_from_mapping(_base(manifest={"write_manifest": True, "subdir": "x"}))


def test_bad_quant_rejected() -> None:
    with pytest.raises(ValidationError):
        config_from_mapping(_base(quant="int8"))


def test_quant_is_scalar_not_a_list() -> None:
    """Framing constraint 4: quantization is a recorded constant, never an axis."""
    with pytest.raises(ValidationError):
        config_from_mapping(_base(quant=["nf4", "none"]))


@pytest.mark.parametrize("bad_name", ["", "x" * 129])
def test_run_name_bounds_rejected(bad_name: str) -> None:
    with pytest.raises(ValidationError):
        config_from_mapping(_base(run_name=bad_name))


def test_config_is_frozen() -> None:
    cfg = config_from_mapping(_base())
    with pytest.raises(ValidationError):
        cfg.seed = 43  # type: ignore[misc]


def test_hash_is_stable_across_identical_configs() -> None:
    assert config_hash(config_from_mapping(_base())) == config_hash(config_from_mapping(_base()))


def test_hash_is_content_sensitive() -> None:
    assert config_hash(config_from_mapping(_base())) != config_hash(
        config_from_mapping(_base(seed=43))
    )
    assert config_hash(config_from_mapping(_base())) != config_hash(
        config_from_mapping(_base(quant="none"))
    )


def test_hash_is_insensitive_to_key_order() -> None:
    ordered = config_from_mapping({"seed": 42, "run_name": "unit-test", "quant": "none"})
    shuffled = config_from_mapping({"quant": "none", "run_name": "unit-test", "seed": 42})
    assert config_hash(ordered) == config_hash(shuffled)


def test_paths_are_parsed_as_paths() -> None:
    cfg = config_from_mapping(_base(results_dir="some/results"))
    assert isinstance(cfg.results_dir, Path)
    assert cfg.results_dir == Path("some/results")


def test_hydra_composes_the_root_config_and_it_validates() -> None:
    """The whole point of the seam: plain YAML -> Hydra -> dict -> Pydantic."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        composed = compose(config_name="config")

    raw = OmegaConf.to_container(composed, resolve=True)
    assert isinstance(raw, dict)
    cfg = config_from_mapping({str(k): v for k, v in raw.items()})

    assert isinstance(cfg, ExperimentConfig)
    assert cfg.quant == "nf4"
    assert cfg.model_id is None, "model checkpoint is open decision #2, unset until P5"


def test_hydra_override_flows_through_validation() -> None:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        composed = compose(config_name="config", overrides=["seed=7", "run_name=overridden"])

    raw = OmegaConf.to_container(composed, resolve=True)
    assert isinstance(raw, dict)
    cfg = config_from_mapping({str(k): v for k, v in raw.items()})
    assert cfg.seed == 7
    assert cfg.run_name == "overridden"
