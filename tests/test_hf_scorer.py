"""The real pinned classifiers, on CPU (P6).

**Marked ``inference`` and skipped on the core environment**, like
``tests/test_inference_backend.py``
-- it downloads and runs real checkpoints, so it belongs to the nightly job and to a dev box with
the
extra installed, never to the core CI job. Everything that can be checked without a model already
is,
in the other P6 test files; what is left here is the part that genuinely needs one:

* the configured label resolves **by name** at the pinned revision, and a wrong label raises rather
  than silently reading a different column;
* the sign contract holds against the **real** instrument, not a fixture. A sign that is right for a
  stand-in and wrong for the checkpoint would invert every effect while every core test stayed
  green.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from margins.scoring.config import DEFAULT_SCORING, ScorerSpec, ScoringConfig, load_scoring_config
from margins.scoring.hf_classifier import HFClassifierScorer
from margins.scoring.standardize import SignContractError, verify_sign

pytestmark = [
    pytest.mark.inference,
    pytest.mark.skipif(
        importlib.util.find_spec("transformers") is None,
        reason="needs the inference extra: uv sync --extra dev --extra inference",
    ),
]


@pytest.fixture(scope="module")
def config() -> ScoringConfig:
    return load_scoring_config(DEFAULT_SCORING)


def neural_specs(config: ScoringConfig) -> list[ScorerSpec]:
    return [spec for spec in config.scorers if spec.mechanism == "neural_classifier"]


def test_every_neural_scorer_loads_at_its_pinned_revision(config: ScoringConfig) -> None:
    for spec in neural_specs(config):
        scores = HFClassifierScorer(spec).score_texts(["A perfectly ordinary sentence."])
        assert len(scores) == 1
        assert 0.0 <= scores[0] <= 1.0


def test_every_neural_scorer_passes_its_sign_probe(config: ScoringConfig) -> None:
    """The sign contract against the REAL instrument. This is the one a fixture cannot check."""
    for spec in neural_specs(config):
        verify_sign(spec, HFClassifierScorer(spec))


def test_a_flipped_sign_is_caught_against_the_real_classifier(config: ScoringConfig) -> None:
    spec = neural_specs(config)[0]
    flipped = spec.model_copy(update={"sign": -1})

    with pytest.raises(SignContractError):
        verify_sign(flipped, HFClassifierScorer(spec))


def test_an_unknown_label_raises_rather_than_reading_a_column(config: ScoringConfig) -> None:
    """Label order is a checkpoint property; reading column 0 and naming it is how a re-pin
    drifts."""
    spec = neural_specs(config)[0]
    assert spec.model_pin is not None
    wrong = spec.model_copy(
        update={"model_pin": spec.model_pin.model_copy(update={"label": "not_a_real_label"})}
    )

    with pytest.raises(ValueError, match="does not expose"):
        HFClassifierScorer(wrong).score_texts(["text"])


def test_scoring_is_deterministic_and_batch_order_is_preserved(config: ScoringConfig) -> None:
    """Classification is a forward pass with no sampling; the caller zips on returned order."""
    spec = neural_specs(config)[0]
    texts = [f"Sentence number {index} about an ordinary afternoon." for index in range(5)]

    small = HFClassifierScorer(spec, batch_size=2).score_texts(texts)
    large = HFClassifierScorer(spec, batch_size=16).score_texts(texts)

    assert small == pytest.approx(large, abs=1e-5)
    # Re-running gives the same answer.
    assert small == pytest.approx(
        HFClassifierScorer(spec, batch_size=2).score_texts(texts), abs=1e-6
    )


def test_empty_input_scores_nothing_without_loading(config: ScoringConfig) -> None:
    """The model is loaded on first use, so an empty call must not download anything."""
    assert HFClassifierScorer(neural_specs(config)[0]).score_texts([]) == []


def test_the_lexicon_scorers_agree_with_their_probes_too(config: ScoringConfig) -> None:
    """Runs here as well as on core, so the shipped roster is verified end to end in one place."""
    from margins.scoring.lexicon import build_lexicon_scorer

    for spec in config.scorers:
        if spec.mechanism != "lexicon":
            continue
        verify_sign(spec, build_lexicon_scorer(spec, data_dir=Path("data")))
