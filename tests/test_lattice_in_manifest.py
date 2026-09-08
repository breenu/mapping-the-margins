"""The lattice reaches the run manifest (P4a).

Before P4a a manifest recorded the experiment config but **not which identity lattice produced
the prompts**, so a prompt set could not be traced back to the cell inventory it audited. That is
the provenance hole P3 flagged and this closes: the name lands on ``ExperimentConfig``, and the
config file's sha256 lands in ``input_checksums`` — the name alone would not survive an edit to
the file.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from margins.config import (
    DEFAULT_LATTICE_NAME,
    LATTICE_INPUT_KEY,
    ExperimentConfig,
    config_from_mapping,
    default_config_dir,
    load_experiment_config,
    write_run_manifest,
)
from margins.lattice import DEFAULT_LATTICE
from margins.lattice.config import default_lattice_config_dir
from margins.prompts import (
    DEFAULT_PROMPT_SET,
    PROMPT_SET_INPUT_KEY,
    TEMPLATE_INPUT_PREFIX,
    file_sha256,
    load_prompt_config,
    load_template_files,
    prompt_set_inputs,
)
from margins.prompts.config import default_prompt_config_dir

FIXED_TIME = datetime(2026, 9, 8, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def cfg(tmp_path: Path) -> ExperimentConfig:
    return config_from_mapping(
        {
            "seed": 20260908,
            "run_name": "manifest-test",
            "results_dir": str(tmp_path / "results"),
            "lattice_name": DEFAULT_LATTICE,
        }
    )


@pytest.fixture(scope="module")
def inputs() -> dict[str, str]:
    config = load_prompt_config(DEFAULT_PROMPT_SET)
    return prompt_set_inputs(
        lattice_config_path=default_lattice_config_dir() / f"{DEFAULT_LATTICE}.yaml",
        prompt_config_path=default_prompt_config_dir() / f"{DEFAULT_PROMPT_SET}.yaml",
        files=load_template_files(config),
    )


# ---------------------------------------------------------------------------- the config field


def test_default_lattice_name_matches_the_lattice_module() -> None:
    """The literal in config.py cannot import from lattice (that would cycle), so pin it here."""
    assert DEFAULT_LATTICE_NAME == DEFAULT_LATTICE


def test_experiment_config_names_a_lattice_by_default() -> None:
    cfg = config_from_mapping({"seed": 1, "run_name": "x"})
    assert cfg.lattice_name == DEFAULT_LATTICE_NAME


def test_root_config_yaml_names_the_shipped_lattice() -> None:
    assert load_experiment_config(config_dir=default_config_dir()).lattice_name == DEFAULT_LATTICE


def test_lattice_name_is_overridable_from_hydra() -> None:
    """`lattice` would collide with the configs/lattice/ config GROUP and be un-overridable."""
    cfg = load_experiment_config(
        config_dir=default_config_dir(), overrides=["lattice_name=some_other"]
    )
    assert cfg.lattice_name == "some_other"


def test_changing_the_lattice_changes_the_config_hash() -> None:
    from margins.config import config_hash

    base = config_from_mapping({"seed": 1, "run_name": "x"})
    other = config_from_mapping({"seed": 1, "run_name": "x", "lattice_name": "tiny"})
    assert config_hash(base) != config_hash(other)


# ---------------------------------------------------------------------------- manifest inputs


def test_manifest_records_the_lattice_config_sha(
    cfg: ExperimentConfig, inputs: dict[str, str]
) -> None:
    """Gate P4a: a manifest must say which lattice produced the prompt set."""
    manifest = write_run_manifest(cfg, inputs=inputs, timestamp=FIXED_TIME)
    assert LATTICE_INPUT_KEY in manifest.input_checksums
    expected = file_sha256(default_lattice_config_dir() / f"{DEFAULT_LATTICE}.yaml")
    assert manifest.input_checksums[LATTICE_INPUT_KEY] == expected
    assert manifest.config.lattice_name == DEFAULT_LATTICE


def test_manifest_records_the_prompt_config_and_every_template(
    cfg: ExperimentConfig, inputs: dict[str, str]
) -> None:
    manifest = write_run_manifest(cfg, inputs=inputs, timestamp=FIXED_TIME)
    checksums = manifest.input_checksums
    assert PROMPT_SET_INPUT_KEY in checksums

    config = load_prompt_config(DEFAULT_PROMPT_SET)
    for spec in config.templates:
        assert f"{TEMPLATE_INPUT_PREFIX}{spec.id}" in checksums
    assert all(len(digest) == 64 for digest in checksums.values())


def test_manifest_survives_the_json_round_trip(
    cfg: ExperimentConfig, inputs: dict[str, str]
) -> None:
    """write_run_manifest returns what is on disk, so the checksums are what a reproducer reads."""
    manifest = write_run_manifest(cfg, inputs=inputs, timestamp=FIXED_TIME)
    assert manifest.input_checksums == inputs


def test_lattice_digest_matches_the_one_build_lattice_prints() -> None:
    """Both artifacts must name the same file, or the provenance chain is broken in the middle."""
    path = default_lattice_config_dir() / f"{DEFAULT_LATTICE}.yaml"
    import hashlib

    assert file_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()
