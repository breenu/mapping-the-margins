"""The manifest's scoring block: which instruments, at which pins, achieving what agreement (P6).

The P5 pattern, one phase later. A response file is traceable to its checkpoint because the manifest
records the revision; a score file must be traceable to its *scorers* the same way. A ``scorer_id``
alone does not do it: the same id under a different classifier revision, a different lexicon digest,
a flipped sign or a moved threshold is a different instrument producing different numbers.

Agreement rides here too, not only in the emitted table, because ``results/run_manifests/`` is the
committed record and ``data/scoring/`` is gitignored -- a run whose agreement lives only in a
gitignored artifact cannot be audited from what is in git.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from margins.config import (
    DEFAULT_SCORING_NAME,
    ExperimentConfig,
    RunManifest,
    ScoringRecord,
    manifest_path,
    write_run_manifest,
)
from margins.scoring.config import (
    DEFAULT_SCORING,
    SCORER_INPUT_PREFIX,
    SCORING_INPUT_KEY,
    ScoringConfig,
    default_scoring_config_dir,
    load_scoring_config,
)
from margins.scoring.manifest import scorer_record, scoring_inputs, scoring_record
from margins.scoring.triangulate import AgreementRecord
from tests.conftest import FIXED_TIME

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "scoring"
BENCHMARK = "open_gen_templates"


@pytest.fixture(scope="module")
def config() -> ScoringConfig:
    return load_scoring_config("tiny", config_dir=FIXTURE_DIR)


def make_agreement(
    harm_type: str, value: float | None, condition: str = "model"
) -> AgreementRecord:
    return AgreementRecord(
        benchmark=BENCHMARK,
        model_id="m",
        harm_type=harm_type,
        condition=condition,  # type: ignore[arg-type]
        metric="krippendorff_alpha",
        scorer_a="ALL",
        value=value,
        n_units=30,
        n_raters=2,
    )


# ------------------------------------------------------------------ the default name


def test_the_default_scoring_name_matches_the_scoring_package() -> None:
    """Held as a literal in config.py to avoid an import cycle; pinned together here."""
    assert DEFAULT_SCORING_NAME == DEFAULT_SCORING


def test_the_experiment_config_field_is_name_suffixed() -> None:
    """`configs/scoring/` is a Hydra config GROUP; a colliding key cannot be overridden at all."""
    assert "scoring_name" in ExperimentConfig.model_fields
    assert "scoring" not in ExperimentConfig.model_fields


# ------------------------------------------------------------------ the record


def test_a_scorer_record_carries_its_pin(config: ScoringConfig) -> None:
    """The id alone cannot answer 'which instrument'."""
    spec = next(s for s in config.scorers if s.mechanism == "neural_classifier")
    record = scorer_record(spec)

    assert record.scorer_id == spec.scorer_id
    assert record.model_repo == "unitary/toxic-bert"
    assert record.model_revision == spec.model_pin.revision  # type: ignore[union-attr]
    assert record.model_label == "toxic"
    assert record.sign == spec.sign
    assert record.threshold == spec.threshold
    assert record.pin_digest == spec.pin_digest
    assert record.license == "apache-2.0"


def test_a_lexicon_scorer_records_its_source_and_size(config: ScoringConfig) -> None:
    spec = next(s for s in config.scorers if s.mechanism == "lexicon")
    record = scorer_record(spec, lexicon_size=403)

    assert record.lexicon_name == spec.lexicon_pin.name  # type: ignore[union-attr]
    assert record.lexicon_version == "1.0"
    assert record.lexicon_size == 403  # a silently empty wordlist scores 0.0 for everything
    assert record.model_repo is None


def test_the_scoring_record_summarizes_the_run(config: ScoringConfig) -> None:
    record = scoring_record(
        config,
        scoring_name="tiny",
        benchmark=BENCHMARK,
        n_responses=100,
        n_scores=400,
        agreement=[make_agreement("toxicity", 0.87), make_agreement("regard", 0.42)],
        n_control_texts=58,
    )

    assert record.scoring_hash == config.scoring_hash
    assert record.harm_types == ("regard", "toxicity")
    assert len(record.scorers) == len(config.scorers)
    assert (record.n_responses, record.n_scores) == (100, 400)
    assert record.control_families == ("prompt_echo", "neutral_declarative")
    assert record.n_control_texts == 58
    assert record.agreement_summary == {"toxicity": 0.87, "regard": 0.42}


def test_the_summary_reports_the_model_condition_only(config: ScoringConfig) -> None:
    """Agreement about neutral control text is a different question and would mislead under this
    key."""
    record = scoring_record(
        config,
        scoring_name="tiny",
        benchmark=BENCHMARK,
        n_responses=10,
        n_scores=40,
        agreement=[
            make_agreement("toxicity", 0.87),
            make_agreement("toxicity", 0.05, condition="drift_control"),
        ],
    )
    assert record.agreement_summary == {"toxicity": 0.87}


def test_an_unestimable_alpha_is_omitted_not_zeroed(config: ScoringConfig) -> None:
    record = scoring_record(
        config,
        scoring_name="tiny",
        benchmark=BENCHMARK,
        n_responses=3,
        n_scores=12,
        agreement=[make_agreement("toxicity", None)],
    )
    assert record.agreement_summary == {}


# ------------------------------------------------------------------ inputs


def test_the_roster_is_checksummed_into_the_manifest_inputs(config: ScoringConfig) -> None:
    inputs = scoring_inputs(
        config,
        scoring_config_path=FIXTURE_DIR / "tiny.yaml",
        data_dir=Path("data"),
    )

    assert SCORING_INPUT_KEY in inputs
    assert len(inputs[SCORING_INPUT_KEY]) == 64


def test_a_packaged_lexicon_contributes_no_fabricated_digest(config: ScoringConfig) -> None:
    """VADER has no file of ours to digest; it is pinned by the package version instead."""
    inputs = scoring_inputs(
        config, scoring_config_path=FIXTURE_DIR / "tiny.yaml", data_dir=Path("data")
    )
    assert not any(key.startswith(SCORER_INPUT_PREFIX) for key in inputs)


def test_the_shipped_roster_digests_its_fetched_lexicon() -> None:
    """The real roster has a fetched wordlist, so it must contribute a scorer: entry."""
    shipped = load_scoring_config(DEFAULT_SCORING)
    inputs = scoring_inputs(
        shipped,
        scoring_config_path=default_scoring_config_dir() / f"{DEFAULT_SCORING}.yaml",
        data_dir=Path("data"),
    )

    key = f"{SCORER_INPUT_PREFIX}ldnoobw_lexicon"
    assert key in inputs
    assert len(inputs[key]) == 64


def test_a_lexicon_that_does_not_match_its_pin_is_refused(
    config: ScoringConfig, tmp_path: Path
) -> None:
    """Digest the file ON DISK and check it, rather than copying the declared value into the
    record."""
    spec = next(s for s in config.scorers if s.scorer_id == "fake_toxic_lex")
    pin = spec.lexicon_pin.model_copy(  # type: ignore[union-attr]
        update={"path": Path("lexicons/x.txt"), "sha256": "b" * 64}
    )
    broken = config.model_copy(
        update={
            "scorers": tuple(
                s.model_copy(update={"lexicon_pin": pin}) if s.scorer_id == spec.scorer_id else s
                for s in config.scorers
            )
        }
    )
    (tmp_path / "lexicons").mkdir()
    (tmp_path / "lexicons" / "x.txt").write_text("different content\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match its pin"):
        scoring_inputs(broken, scoring_config_path=FIXTURE_DIR / "tiny.yaml", data_dir=tmp_path)


# ------------------------------------------------------------------ the manifest


def test_the_scoring_block_reaches_the_manifest_on_disk(
    config: ScoringConfig, tmp_path: Path
) -> None:
    """Round-tripped through the file, because write_run_manifest re-reads what it wrote."""
    cfg = ExperimentConfig(
        seed=20260918, run_name="scoring-test", results_dir=tmp_path, scoring_name="tiny"
    )
    record = scoring_record(
        config,
        scoring_name="tiny",
        benchmark=BENCHMARK,
        n_responses=10,
        n_scores=40,
        agreement=[make_agreement("toxicity", 0.9)],
    )

    manifest = write_run_manifest(cfg, timestamp=FIXED_TIME, scoring=record)

    assert manifest.scoring is not None
    assert manifest.scoring.scoring_name == "tiny"
    assert manifest.scoring.agreement_summary == {"toxicity": 0.9}
    assert {s.scorer_id for s in manifest.scoring.scorers} == {s.scorer_id for s in config.scorers}

    on_disk = RunManifest.model_validate_json(
        manifest_path(cfg, manifest.run_id).read_text(encoding="utf-8")
    )
    assert on_disk.scoring == manifest.scoring


def test_a_run_that_scores_nothing_records_no_scoring_block(tmp_path: Path) -> None:
    """None, not an empty record: 'this run did not score' is a fact worth keeping."""
    cfg = ExperimentConfig(seed=1, run_name="prompts", results_dir=tmp_path)
    manifest = write_run_manifest(cfg, timestamp=FIXED_TIME)
    assert manifest.scoring is None


def test_the_scoring_packages_are_tracked(tmp_path: Path) -> None:
    """An agreement coefficient depends on the implementation that computed it."""
    cfg = ExperimentConfig(seed=1, run_name="x", results_dir=tmp_path)
    manifest = write_run_manifest(cfg, timestamp=FIXED_TIME)

    for package in ("krippendorff", "scikit-learn", "vaderSentiment"):
        assert package in manifest.package_versions


def test_the_record_is_frozen_and_forbids_extras(config: ScoringConfig) -> None:
    record = scoring_record(
        config, scoring_name="tiny", benchmark=BENCHMARK, n_responses=1, n_scores=4
    )
    with pytest.raises(Exception):  # noqa: B017 - pydantic raises ValidationError on a frozen set
        record.__setattr__("n_scores", 99)

    with pytest.raises(Exception):  # noqa: B017
        ScoringRecord.model_validate({**record.model_dump(mode="json"), "unexpected": 1})
