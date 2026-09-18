"""The scorer contract: one row per response per scorer, and never an average (P6).

**Why this runs without a model.** Every test here drives a fake :class:`TextScorer` -- three lines,
no imports. Scoring arithmetic is a property of the pipeline, not of any classifier, so the whole
file runs on the core environment with no ``torch`` and no network, exactly as
``tests/test_harness_resume.py`` does for the run harness.

The load-bearing case is :func:`test_responses_sharing_a_prompt_yield_one_score_each`. Open decision
#8 makes E1 sampled: each prompt is generated many times at distinct seeds. A scorer that collapsed
those into one row would destroy the within-cell variance E4's sensitivity analysis measures and the
cluster bootstrap resamples -- and it would do so invisibly, because the output would still be a
perfectly well-formed score table.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from margins.interfaces import Scorer
from margins.schemas import ItemScore, Prompt, Response, TaskType
from margins.scoring.base import ResponseScorer
from margins.scoring.config import ScoringConfig, load_scoring_config
from margins.scoring.ids import SCORE_ID_DIGEST_LENGTH, make_score_id

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "scoring"
BENCHMARK = "open_gen_templates"
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
SEEDS = (20260918, 20260919, 20260920)


# ------------------------------------------------------------------ fakes


class FakeTextScorer:
    """Scores by text length, deterministically. Satisfies the narrow ``TextScorer`` Protocol."""

    def __init__(self, scorer_id: str, *, scale: float = 0.01) -> None:
        self._scorer_id = scorer_id
        self._scale = scale
        self.calls: list[int] = []

    @property
    def scorer_id(self) -> str:
        return self._scorer_id

    def score_texts(self, texts: Sequence[str]) -> list[float]:
        self.calls.append(len(texts))
        return [len(text) * self._scale for text in texts]


class WrongLengthScorer(FakeTextScorer):
    """Returns the wrong number of scores. The caller zips, so this must be caught, not zipped."""

    def score_texts(self, texts: Sequence[str]) -> list[float]:
        return [0.0] * (len(texts) + 1)


# ------------------------------------------------------------------ fixtures


@pytest.fixture(scope="module")
def config() -> ScoringConfig:
    return load_scoring_config("tiny", config_dir=FIXTURE_DIR)


def make_prompt(prompt_id: str, cell_id: str | None = "alpha=a1") -> Prompt:
    return Prompt(
        prompt_id=prompt_id,
        item_id="open_gen_templates/neutral",
        cell_id=cell_id,
        rendered_text=f"Write about {cell_id}.",
        task_type=TaskType.OPEN_GEN,
        template_version="tiny@test",
        decoding_key="greedy_test",
    )


def make_response(
    response_id: str, prompt_id: str, *, seed: int, text: str = "some output"
) -> Response:
    return Response(
        response_id=response_id,
        prompt_id=prompt_id,
        model_id=MODEL_ID,
        model_revision="9213176726f574b556790deb65791e0c5aa438b6",
        seed=seed,
        temperature=0.7,
        decoding_key="sampled_v1",
        decoding_hash="0" * 64,
        quant="nf4",
        text=text,
    )


def build(config: ScoringConfig, scorer_id: str, prompts: dict[str, Prompt]) -> ResponseScorer:
    spec = next(s for s in config.scorers if s.scorer_id == scorer_id)
    return ResponseScorer(
        spec,
        FakeTextScorer(scorer_id),
        prompts_by_id=prompts,
        benchmark=BENCHMARK,
        scoring_hash=config.scoring_hash,
    )


# ------------------------------------------------------------------ the Protocol


def test_a_response_scorer_satisfies_the_scorer_protocol(config: ScoringConfig) -> None:
    scorer = build(config, "fake_toxic_nn", {})
    assert isinstance(scorer, Scorer)
    assert scorer.scorer_id == "fake_toxic_nn"
    assert scorer.harm_type == "toxicity"


def test_one_score_per_response_with_denormalized_keys(config: ScoringConfig) -> None:
    prompts = {"p1": make_prompt("p1", "alpha=a1|beta=b1")}
    responses = [make_response("r1", "p1", seed=SEEDS[0], text="abcde")]

    scores = build(config, "fake_toxic_nn", prompts).score(responses)

    assert len(scores) == 1
    score = scores[0]
    assert score.response_id == "r1"
    assert score.scorer_id == "fake_toxic_nn"
    assert score.harm_type == "toxicity"
    assert score.raw == pytest.approx(0.05, abs=1e-9)
    # The denormalized keys are what keep the metric layer joinless. prompt_id is the item the
    # P7 estimator clusters by; a row that lost it could not be fitted.
    assert score.prompt_id == "p1"
    assert score.cell_id == "alpha=a1|beta=b1"
    assert score.benchmark == BENCHMARK
    assert score.model_id == MODEL_ID
    assert score.seed == SEEDS[0]
    assert score.temperature == pytest.approx(0.7)
    assert score.condition == "model"


def test_responses_sharing_a_prompt_yield_one_score_each(config: ScoringConfig) -> None:
    """Sampled generation: N seeds on one prompt must give N rows, NOT one averaged row.

    Aggregation across seeds belongs to the metric layer (P7). Doing it here would destroy the
    within-cell variance E4 measures, and would look like a perfectly good score table.
    """
    prompts = {"p1": make_prompt("p1")}
    responses = [
        make_response(f"r{index}", "p1", seed=seed, text="x" * (index + 1))
        for index, seed in enumerate(SEEDS)
    ]

    scores = build(config, "fake_toxic_nn", prompts).score(responses)

    assert len(scores) == len(SEEDS)
    assert {s.seed for s in scores} == set(SEEDS)
    assert len({s.response_id for s in scores}) == len(SEEDS)
    # Distinct raw values survive; nothing was collapsed.
    assert sorted(s.raw for s in scores) == pytest.approx([0.01, 0.02, 0.03], abs=1e-9)


def test_the_text_scorer_is_called_once_for_the_whole_batch(config: ScoringConfig) -> None:
    """Batching is the scorer's business, but the row count is not affected by it."""
    prompts = {"p1": make_prompt("p1")}
    responses = [make_response(f"r{i}", "p1", seed=SEEDS[0] + i) for i in range(3)]

    spec = next(s for s in config.scorers if s.scorer_id == "fake_toxic_nn")
    text_scorer = FakeTextScorer("fake_toxic_nn")
    scorer = ResponseScorer(
        spec, text_scorer, prompts_by_id=prompts, benchmark=BENCHMARK, scoring_hash="h"
    )

    assert len(scorer.score(responses)) == 3
    assert text_scorer.calls == [3]


def test_empty_input_scores_nothing(config: ScoringConfig) -> None:
    assert build(config, "fake_toxic_nn", {}).score([]) == []


# ------------------------------------------------------------------ refusals


def test_a_response_without_its_prompt_is_refused(config: ScoringConfig) -> None:
    """cell_id and benchmark live on the Prompt; scoring without it emits ungroupable rows."""
    responses = [make_response("r1", "missing", seed=SEEDS[0])]

    with pytest.raises(KeyError, match="not given"):
        build(config, "fake_toxic_nn", {}).score(responses)


def test_a_paired_response_is_refused_not_text_scored(config: ScoringConfig) -> None:
    """TODO(P2): pair_pll reads stereo_preference off the record. Refusing beats a meaningless
    number."""
    prompts = {"p1": make_prompt("p1")}
    paired = make_response("r1", "p1", seed=SEEDS[0]).model_copy(
        update={"logprob_stereo": -3.0, "logprob_antistereo": -5.0}
    )

    with pytest.raises(ValueError, match="pair_pll"):
        build(config, "fake_toxic_nn", prompts).score([paired])


def test_a_text_scorer_returning_the_wrong_count_is_refused(config: ScoringConfig) -> None:
    prompts = {"p1": make_prompt("p1")}
    spec = next(s for s in config.scorers if s.scorer_id == "fake_toxic_nn")
    scorer = ResponseScorer(
        spec,
        WrongLengthScorer("fake_toxic_nn"),
        prompts_by_id=prompts,
        benchmark=BENCHMARK,
        scoring_hash="h",
    )

    with pytest.raises(ValueError, match="one score per text"):
        scorer.score([make_response("r1", "p1", seed=SEEDS[0])])


def test_a_mismatched_scorer_id_is_refused(config: ScoringConfig) -> None:
    """The id is recorded on every row; a mismatch attributes scores to the wrong instrument."""
    spec = next(s for s in config.scorers if s.scorer_id == "fake_toxic_nn")

    with pytest.raises(ValueError, match="does not match roster entry"):
        ResponseScorer(
            spec,
            FakeTextScorer("something_else"),
            prompts_by_id={},
            benchmark=BENCHMARK,
            scoring_hash="h",
        )


# ------------------------------------------------------------------ ids


def test_score_id_is_readable_and_content_addressed() -> None:
    score_id = make_score_id(
        response_id="r1",
        scorer_id="toxic_bert",
        harm_type="toxicity",
        condition="model",
        pin_digest="abc",
        scoring_hash="h",
    )
    prefix, condition, response_id, digest = score_id.split("::")
    assert (prefix, condition, response_id) == ("toxic_bert", "model", "r1")
    assert len(digest) == SCORE_ID_DIGEST_LENGTH


@pytest.mark.parametrize("field", ["pin_digest", "scoring_hash", "condition", "harm_type"])
def test_score_id_moves_when_the_instrument_moves(field: str) -> None:
    """Re-pin, re-threshold, or re-condition and the id must change, or rows collide."""
    base = {
        "response_id": "r1",
        "scorer_id": "toxic_bert",
        "harm_type": "toxicity",
        "condition": "model",
        "pin_digest": "abc",
        "scoring_hash": "h",
    }
    changed = {**base, field: "drift_control" if field == "condition" else "CHANGED"}
    assert make_score_id(**base) != make_score_id(**changed)  # type: ignore[arg-type]


def test_score_ids_are_unique_across_a_run(config: ScoringConfig) -> None:
    prompts = {f"p{i}": make_prompt(f"p{i}") for i in range(3)}
    responses = [
        make_response(f"r{i}-{seed}", f"p{i}", seed=seed) for i in range(3) for seed in SEEDS
    ]

    scores: list[ItemScore] = []
    for scorer_id in ("fake_toxic_nn", "fake_toxic_lex", "fake_regard_nn", "fake_regard_lex"):
        scores.extend(build(config, scorer_id, prompts).score(responses))

    assert len({s.score_id for s in scores}) == len(scores) == 9 * 4


def test_the_two_conditions_do_not_collide(config: ScoringConfig) -> None:
    """The drift control scores different text through the same scorer; ids must separate them."""
    prompts = {"p1": make_prompt("p1")}
    responses = [make_response("r1", "p1", seed=SEEDS[0])]

    scorer = build(config, "fake_toxic_nn", prompts)
    model = scorer.score(responses)[0]
    control = scorer.for_condition("drift_control").score(responses)[0]

    assert model.score_id != control.score_id
    assert (model.condition, control.condition) == ("model", "drift_control")


def test_for_condition_shares_the_underlying_instrument(config: ScoringConfig) -> None:
    """A second instance could load a different revision; the control must measure the same one."""
    spec = next(s for s in config.scorers if s.scorer_id == "fake_toxic_nn")
    text_scorer = FakeTextScorer("fake_toxic_nn")
    scorer = ResponseScorer(
        spec, text_scorer, prompts_by_id={}, benchmark=BENCHMARK, scoring_hash="h"
    )

    assert scorer.for_condition("drift_control")._text_scorer is text_scorer
