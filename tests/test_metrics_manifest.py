"""The manifest's metrics block: which estimator, which resampling unit, which thresholds (P7).

The P5/P6 pattern one phase later. The estimand is pre-registration material -- identity link,
sum-to-zero coding, cluster-robust SEs by item, a cluster bootstrap over items stratified by
cell -- and an EffectEstimate does not say which of those produced it. The manifest must, or a
reproducer reading ``effects.parquet`` cannot tell a cluster-bootstrap CI from an iid one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from margins.config import (
    DEFAULT_METRICS_NAME,
    ExperimentConfig,
    MetricsRecord,
    RunManifest,
    manifest_path,
    write_run_manifest,
)
from margins.metrics.config import (
    DEFAULT_METRICS,
    DRIFT_INPUT_KEY,
    METRICS_INPUT_KEY,
    SCORES_INPUT_KEY,
    MetricsConfig,
    load_metrics_config,
)
from margins.metrics.manifest import metrics_inputs, metrics_record
from margins.metrics.records import HarmTypeFlag
from tests.conftest import FIXED_TIME

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "metrics"
BENCHMARK = "open_gen_templates"


@pytest.fixture(scope="module")
def config() -> MetricsConfig:
    return load_metrics_config("tiny", config_dir=FIXTURE_DIR)


def test_the_default_metrics_name_matches_the_metrics_package() -> None:
    """Held as a literal in config.py to avoid an import cycle; pinned together here."""
    assert DEFAULT_METRICS_NAME == DEFAULT_METRICS


def test_the_experiment_config_field_is_name_suffixed() -> None:
    """`configs/metrics/` is a Hydra config GROUP; a colliding key cannot be overridden at all."""
    assert "metrics_name" in ExperimentConfig.model_fields
    assert "metrics" not in ExperimentConfig.model_fields


def test_the_record_names_the_estimand(config: MetricsConfig) -> None:
    record = metrics_record(
        config,
        metrics_name="tiny",
        benchmark=BENCHMARK,
        min_cell_n=60,
        n_pairs=7,
        n_effects=140,
        n_not_estimable=1,
        n_drift_fits=56,
        harm_flags=[
            HarmTypeFlag(
                benchmark=BENCHMARK,
                model_id="m",
                harm_type="toxicity",
                descriptive_only_candidate=True,
                n_comparisons=14,
                n_comparable=3,
                scorers_comparable=("toxic_bert",),
            )
        ],
        n_family_disagreements=2,
    )
    assert record.metrics_hash == config.metrics_hash
    assert record.link == "identity"
    assert "sum" in record.coding
    assert "cluster" in record.cov_type and "item" in record.cov_type
    assert "item" in record.resampling_unit and "cell" in record.resampling_unit
    assert record.n_resamples == config.bootstrap.n_resamples
    assert record.bootstrap_seed == config.bootstrap.seed
    assert record.comparable_ratio == config.drift.comparable_ratio
    assert record.descriptive_only_candidates == ("toxicity",)
    assert record.n_family_disagreements == 2
    assert record.min_cell_n == 60


def test_the_inputs_digest_the_config_and_both_consumed_tables(
    config: MetricsConfig, tmp_path: Path
) -> None:
    scores = tmp_path / "scores.parquet"
    drift = tmp_path / "drift_control.parquet"
    scores.write_bytes(b"scores")
    drift.write_bytes(b"drift")

    inputs = metrics_inputs(
        metrics_config_path=FIXTURE_DIR / "tiny.yaml", scores_path=scores, drift_path=drift
    )
    assert set(inputs) == {METRICS_INPUT_KEY, SCORES_INPUT_KEY, DRIFT_INPUT_KEY}
    assert all(len(value) == 64 for value in inputs.values())
    assert inputs[SCORES_INPUT_KEY] != inputs[DRIFT_INPUT_KEY]


def test_the_metrics_block_reaches_the_manifest_on_disk(
    config: MetricsConfig, tmp_path: Path
) -> None:
    cfg = ExperimentConfig(
        seed=20260918, run_name="metrics-test", results_dir=tmp_path, metrics_name="tiny"
    )
    record = metrics_record(
        config,
        metrics_name="tiny",
        benchmark=BENCHMARK,
        min_cell_n=60,
        n_pairs=1,
        n_effects=11,
        n_not_estimable=0,
        n_drift_fits=2,
        harm_flags=[],
        n_family_disagreements=0,
    )
    manifest = write_run_manifest(cfg, timestamp=FIXED_TIME, metrics=record)

    assert manifest.metrics is not None
    assert manifest.metrics.metrics_name == "tiny"
    on_disk = RunManifest.model_validate_json(
        manifest_path(cfg, manifest.run_id).read_text(encoding="utf-8")
    )
    assert on_disk.metrics == manifest.metrics
    assert on_disk.config.metrics_name == "tiny"


def test_a_run_that_fits_nothing_records_no_metrics_block(tmp_path: Path) -> None:
    cfg = ExperimentConfig(seed=1, run_name="prompts", results_dir=tmp_path)
    manifest = write_run_manifest(cfg, timestamp=FIXED_TIME)
    assert manifest.metrics is None


def test_the_stats_packages_are_tracked(tmp_path: Path) -> None:
    """A cluster-robust covariance depends on the implementation that computed it."""
    cfg = ExperimentConfig(seed=1, run_name="x", results_dir=tmp_path)
    manifest = write_run_manifest(cfg, timestamp=FIXED_TIME)
    for package in ("statsmodels", "scipy", "numpy"):
        assert package in manifest.package_versions


def test_the_record_is_frozen_and_forbids_extras(config: MetricsConfig) -> None:
    record = metrics_record(
        config,
        metrics_name="tiny",
        benchmark=BENCHMARK,
        min_cell_n=1,
        n_pairs=0,
        n_effects=0,
        n_not_estimable=0,
        n_drift_fits=0,
        harm_flags=[],
        n_family_disagreements=0,
    )
    with pytest.raises(Exception):  # noqa: B017 - pydantic raises ValidationError on a frozen set
        record.__setattr__("n_effects", 99)
    with pytest.raises(Exception):  # noqa: B017
        MetricsRecord.model_validate({**record.model_dump(mode="json"), "unexpected": 1})
