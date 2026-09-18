"""The calibration scaffold: deterministic sampling, and correct precision/recall (P6).

The labeling itself is the human's work. What is tested here is everything around it -- that the
sample is reproducible, that it over-weights the cases triangulation splits on, that a label file
cannot silently evaluate on responses that were never sampled, and that the arithmetic is right
against a fixture whose answer is computed by hand in the test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from margins.schemas import ItemScore
from margins.scoring.calibration import (
    CalibrationUnit,
    HumanLabel,
    calibration_metrics,
    read_labels,
    render_calibration_report,
    sample_for_calibration,
    write_calibration_sample,
)
from margins.scoring.config import ScoringConfig, load_scoring_config

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "scoring"
BENCHMARK = "open_gen_templates"
N = 400


@pytest.fixture(scope="module")
def config() -> ScoringConfig:
    return load_scoring_config("tiny", config_dir=FIXTURE_DIR)


def make_score(
    index: int,
    *,
    scorer_id: str,
    raw: float,
    harm_type: str = "toxicity",
    standardized: float | None = None,
    condition: str = "model",
) -> ItemScore:
    return ItemScore(
        score_id=f"{scorer_id}::{condition}::r{index}",
        response_id=f"r{index}",
        scorer_id=scorer_id,
        harm_type=harm_type,
        raw=raw,
        standardized=raw if standardized is None else standardized,
        prompt_id=f"p{index}",
        cell_id="alpha=a1|beta=b1" if index % 3 == 0 else "alpha=a1",
        benchmark=BENCHMARK,
        model_id="m",
        seed=20260918 + index,
        temperature=0.7,
        condition=condition,  # type: ignore[arg-type]
    )


@pytest.fixture(scope="module")
def scores() -> list[ItemScore]:
    """Two scorers over N responses, disagreeing more as the index rises."""
    rows: list[ItemScore] = []
    for index in range(N):
        drift = (index % 10) / 20.0
        rows.append(make_score(index, scorer_id="fake_toxic_nn", raw=0.4, standardized=0.0))
        rows.append(make_score(index, scorer_id="fake_toxic_lex", raw=0.6, standardized=drift))
    return rows


@pytest.fixture(scope="module")
def texts() -> dict[str, str]:
    return {f"r{index}": f"generation number {index}" for index in range(N)}


@pytest.fixture(scope="module")
def cell_order() -> dict[str, int]:
    return {"alpha=a1": 1, "alpha=a1|beta=b1": 2}


# ------------------------------------------------------------------ sampling


def test_the_sample_is_the_configured_size(
    config: ScoringConfig,
    scores: list[ItemScore],
    texts: dict[str, str],
    cell_order: dict[str, int],
) -> None:
    units = sample_for_calibration(
        scores, texts, config, harm_type="toxicity", cell_order=cell_order
    )
    assert len(units) == config.calibration.sample_size == 150


def test_the_sample_is_deterministic_at_a_fixed_seed(
    config: ScoringConfig,
    scores: list[ItemScore],
    texts: dict[str, str],
    cell_order: dict[str, int],
) -> None:
    """Routed through set_global_seed; no hidden global RNG anywhere on the path."""
    first = sample_for_calibration(
        scores, texts, config, harm_type="toxicity", cell_order=cell_order
    )
    second = sample_for_calibration(
        scores, texts, config, harm_type="toxicity", cell_order=cell_order
    )
    assert [u.response_id for u in first] == [u.response_id for u in second]


def test_a_different_seed_draws_a_different_sample(
    config: ScoringConfig,
    scores: list[ItemScore],
    texts: dict[str, str],
    cell_order: dict[str, int],
) -> None:
    """The assertion is that the seed changes something, not everything."""
    other = config.model_copy(
        update={"calibration": config.calibration.model_copy(update={"seed": 999})}
    )
    first = sample_for_calibration(
        scores, texts, config, harm_type="toxicity", cell_order=cell_order
    )
    second = sample_for_calibration(
        scores, texts, other, harm_type="toxicity", cell_order=cell_order
    )
    assert [u.response_id for u in first] != [u.response_id for u in second]


def test_the_sample_is_stratified_across_disagreement_and_cell_order(
    config: ScoringConfig,
    scores: list[ItemScore],
    texts: dict[str, str],
    cell_order: dict[str, int],
) -> None:
    """A uniform sample would spend a human's attention confirming true negatives."""
    units = sample_for_calibration(
        scores, texts, config, harm_type="toxicity", cell_order=cell_order
    )

    assert {u.disagreement_bucket for u in units} == {"low", "mid", "high"}
    assert {u.cell_order for u in units} == {1, 2}


def test_each_unit_carries_what_the_machines_said(
    config: ScoringConfig,
    scores: list[ItemScore],
    texts: dict[str, str],
    cell_order: dict[str, int],
) -> None:
    units = sample_for_calibration(
        scores, texts, config, harm_type="toxicity", cell_order=cell_order
    )
    unit = units[0]
    assert set(unit.scorer_raw) == {"fake_toxic_nn", "fake_toxic_lex"}
    assert unit.text.startswith("generation number")


def test_drift_control_rows_are_never_sampled(config: ScoringConfig, texts: dict[str, str]) -> None:
    """Calibration asks whether a scorer finds harm in MODEL OUTPUT. Control text has none to
    find."""
    control_only = [
        make_score(i, scorer_id="fake_toxic_nn", raw=0.5, condition="drift_control")
        for i in range(20)
    ]

    with pytest.raises(ValueError, match="nothing to calibrate"):
        sample_for_calibration(control_only, texts, config, harm_type="toxicity")


# ------------------------------------------------------------------ the label file


def test_the_sample_writes_as_jsonl_and_labels_read_back(
    config: ScoringConfig,
    scores: list[ItemScore],
    texts: dict[str, str],
    cell_order: dict[str, int],
    tmp_path: Path,
) -> None:
    units = sample_for_calibration(
        scores, texts, config, harm_type="toxicity", cell_order=cell_order
    )
    path = write_calibration_sample(units, tmp_path / "sample.jsonl")

    assert path.is_file()
    assert len(path.read_text(encoding="utf-8").splitlines()) == len(units)
    # Bytes, not text: read_text would translate the newlines and hide a CRLF regression.
    assert b"\r" not in path.read_bytes()

    label_path = tmp_path / "labels.jsonl"
    label_path.write_text(
        "\n".join(
            HumanLabel(
                response_id=unit.response_id, harm_type="toxicity", label=index % 2, labeler="rb"
            ).model_dump_json()
            for index, unit in enumerate(units[:5])
        ),
        encoding="utf-8",
    )

    labels = read_labels(label_path, units=units)
    assert len(labels) == 5


def test_a_label_for_an_unsampled_response_is_refused(tmp_path: Path) -> None:
    """Evaluating off-sample breaks the stratification the precision estimate is conditioned on."""
    units = [
        CalibrationUnit(
            response_id="r1",
            harm_type="toxicity",
            text="t",
            cell_order=1,
            disagreement_bucket="low",
            scorer_raw={},
        )
    ]
    path = tmp_path / "labels.jsonl"
    path.write_text(
        HumanLabel(
            response_id="r999", harm_type="toxicity", label=1, labeler="rb"
        ).model_dump_json(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not in the sample"):
        read_labels(path, units=units)


def test_a_duplicate_label_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    line = HumanLabel(
        response_id="r1", harm_type="toxicity", label=1, labeler="rb"
    ).model_dump_json()
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate label"):
        read_labels(path)


# ------------------------------------------------------------------ the metrics


def test_precision_and_recall_are_computed_correctly(config: ScoringConfig) -> None:
    """Hand-computed against a tiny fixture, so the test checks arithmetic and not the code itself.

    Threshold for fake_toxic_nn is 0.5. Five responses:

        r0  raw 0.9 -> fires,  human 1  -> TP
        r1  raw 0.8 -> fires,  human 1  -> TP
        r2  raw 0.7 -> fires,  human 0  -> FP
        r3  raw 0.2 -> silent, human 1  -> FN
        r4  raw 0.1 -> silent, human 0  -> TN

    precision = 2/3, recall = 2/3, F1 = 2/3.
    """
    raws = [0.9, 0.8, 0.7, 0.2, 0.1]
    truth = [1, 1, 0, 1, 0]
    scores = [make_score(i, scorer_id="fake_toxic_nn", raw=raw) for i, raw in enumerate(raws)]
    labels = [
        HumanLabel(response_id=f"r{i}", harm_type="toxicity", label=value, labeler="rb")
        for i, value in enumerate(truth)
    ]

    metrics = {m.scorer_id: m for m in calibration_metrics(scores, labels, config)}
    metric = metrics["fake_toxic_nn"]

    assert (metric.true_positives, metric.false_positives, metric.false_negatives) == (2, 1, 1)
    assert metric.precision == pytest.approx(2 / 3)
    assert metric.recall == pytest.approx(2 / 3)
    assert metric.f1 == pytest.approx(2 / 3)
    assert metric.n == 5
    assert metric.n_positive == 3


def test_an_empty_denominator_is_none_not_zero(config: ScoringConfig) -> None:
    """'The scorer never fired' and 'it fired and was always wrong' are different facts."""
    scores = [make_score(i, scorer_id="fake_toxic_nn", raw=0.1) for i in range(3)]
    labels = [
        HumanLabel(response_id=f"r{i}", harm_type="toxicity", label=0, labeler="rb")
        for i in range(3)
    ]

    metric = next(
        m for m in calibration_metrics(scores, labels, config) if m.scorer_id == "fake_toxic_nn"
    )
    assert metric.precision is None  # never fired
    assert metric.recall is None  # no positives to find


def test_every_roster_scorer_appears_in_the_report(config: ScoringConfig) -> None:
    metrics = calibration_metrics([], [], config)
    assert {m.scorer_id for m in metrics} == {s.scorer_id for s in config.scorers}


def test_the_report_states_what_the_sample_is_and_is_not(config: ScoringConfig) -> None:
    """The sample is deliberately hard; reporting its precision as corpus precision would
    mislead."""
    metrics = calibration_metrics([], [], config)
    text = render_calibration_report(metrics, scoring_version="v1", n_labels=0)

    assert "Do not edit by hand" in text
    assert "not** an estimate of precision on the corpus" in text
    assert "identity-drift control" in text
    assert "—" in text  # empty denominators render as a dash, not 0.000
