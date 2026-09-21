"""The paired-likelihood scoring path (P2).

Four things are checked here, in rising order of how badly they would fail silently:

1. **The reductions compute what they say.** Hand-computed values, no model.
2. **The sign contract holds** -- and, more importantly, *fails* when it is violated. A reversed
   subtraction inverts every CrowS effect while producing entirely plausible output, which is the
   single most dangerous bug this vertical can have.
3. **The two shapes refuse each other.** A text scorer refuses a paired record and a paired scorer
   refuses a generated one; either way round, scoring the wrong shape yields a number that means
   nothing.
4. **Framing constraint 6 is still enforced** on the new mechanisms, including the roster rule that
   a harm type may not be measured by two scorers sharing a mechanism.

Nothing here loads a model, by design. A paired scorer has no instrument -- the log-probabilities
are the model's -- so what can be wrong is our arithmetic, and that is checkable on the torch-free
core environment, which is where the sign contract most needs enforcing.
"""

from __future__ import annotations

import math

import pytest

from margins.schemas import ItemScore, Prompt, Response, TaskType
from margins.scoring.base import RosterScorer, scorer_from_spec
from margins.scoring.config import ScorerSpec, load_scoring_config
from margins.scoring.paired import (
    LengthNormalizedScorer,
    PairedScorer,
    TotalLogprobScorer,
    paired_reduction_for,
    paired_scorer_from_spec,
)
from margins.scoring.standardize import (
    SignContractError,
    signs_from_specs,
    standardize_within_benchmark,
    verify_paired_sign,
    verify_sign,
)

BENCHMARK = "crows_pairs"
ROSTER = "crows_pll_v1"


@pytest.fixture(scope="module")
def roster():  # type: ignore[no-untyped-def]
    return load_scoring_config(ROSTER)


def _prompt(index: int = 0, *, stereo: str = "a b c", antistereo: str = "d e") -> Prompt:
    return Prompt(
        prompt_id=f"crows_pll_v1::race_ethnicity=Black::crows_pairs/{index:04d}::{index:012d}",
        item_id=f"crows_pairs/{index:04d}",
        cell_id="race_ethnicity=Black",
        rendered_text="",
        task_type=TaskType.PAIR_PLL,
        template_version="crows_pll_v1@2026-09-20",
        decoding_key="pll_v1",
        rendered_text_stereo=stereo,
        rendered_text_antistereo=antistereo,
    )


def _response(prompt: Prompt, stereo: float, antistereo: float) -> Response:
    return Response(
        response_id=f"{prompt.prompt_id}::m::0::0::deadbeef1234",
        prompt_id=prompt.prompt_id,
        model_id="test/model",
        model_revision="0" * 40,
        seed=0,
        temperature=0.0,
        decoding_key="pll_v1",
        decoding_hash="hash",
        quant="none",
        text="",
        logprob_stereo=stereo,
        logprob_antistereo=antistereo,
    )


def _generated(prompt_id: str) -> Response:
    return Response(
        response_id="gen-1",
        prompt_id=prompt_id,
        model_id="test/model",
        model_revision="0" * 40,
        seed=0,
        temperature=0.7,
        decoding_key="sampled_v1",
        decoding_hash="hash",
        quant="none",
        text="a generated sentence",
    )


# --------------------------------------------------------------------------------------
# 1. The reductions, hand-computed
# --------------------------------------------------------------------------------------


def test_total_logprob_is_the_difference() -> None:
    prompt = _prompt()
    response = _response(prompt, stereo=-4.0, antistereo=-10.0)
    assert TotalLogprobScorer("pll_total").reduce(response, prompt) == pytest.approx(6.0)
    assert response.stereo_preference == pytest.approx(6.0)


def test_length_normalization_divides_each_side_by_its_own_length() -> None:
    """Not ``difference / length``: the two members can differ in length, which is the point.

    Dividing the difference by one length would leave the confound in.
    """
    prompt = _prompt(stereo="a b c", antistereo="d e")  # 3 words vs 2
    response = _response(prompt, stereo=-6.0, antistereo=-8.0)

    expected = (-6.0 / 3) - (-8.0 / 2)  # -2.0 + 4.0 = 2.0
    assert LengthNormalizedScorer("pll_length_normalized").reduce(
        response, prompt
    ) == pytest.approx(expected)
    assert expected == pytest.approx(2.0)


def test_the_two_reductions_disagree_exactly_where_length_differs() -> None:
    """Their disagreement IS the length confound, which is why triangulating them is informative.

    The unnormalised reduction prefers the stereo sequence; the normalised one does not, because
    the stereo sequence is simply longer.
    """
    prompt = _prompt(stereo="one two three four five six", antistereo="one two")
    response = _response(prompt, stereo=-6.0, antistereo=-4.0)

    total = TotalLogprobScorer("pll_total").reduce(response, prompt)
    normalized = LengthNormalizedScorer("pll_length_normalized").reduce(response, prompt)

    assert total < 0, "unnormalised: the longer stereo sequence looks less preferred"
    assert normalized > 0, "per word, the stereo sequence is MORE preferred"


def test_a_reduction_refuses_an_unpaired_response() -> None:
    prompt = _prompt()
    with pytest.raises(ValueError, match="carries no paired log-probabilities"):
        TotalLogprobScorer("pll_total").reduce(_generated(prompt.prompt_id), prompt)


def test_an_unknown_mechanism_is_not_silently_built(roster) -> None:  # type: ignore[no-untyped-def]
    spec = roster.scorers[0].model_copy(update={"mechanism": "lexicon"})
    with pytest.raises(ValueError, match="not a paired reduction"):
        paired_reduction_for(spec)


# --------------------------------------------------------------------------------------
# 2. THE SIGN CONTRACT
# --------------------------------------------------------------------------------------


def test_the_shipped_paired_probes_pass_without_a_model(roster) -> None:  # type: ignore[no-untyped-def]
    for spec in roster.scorers:
        verify_paired_sign(spec, paired_reduction_for(spec))


def test_a_reversed_sign_is_caught_by_its_own_probe(roster) -> None:  # type: ignore[no-untyped-def]
    """The check that matters. A mis-signed scorer inverts every effect and looks fine doing it."""
    spec = roster.scorers[0].model_copy(update={"sign": -1})
    with pytest.raises(SignContractError, match="wrong way round"):
        verify_paired_sign(spec, paired_reduction_for(spec))


def test_a_reversed_subtraction_is_caught_by_its_own_probe(roster) -> None:  # type: ignore[no-untyped-def]
    """The other way the sign can break: correct `sign`, reversed arithmetic."""

    class Reversed(TotalLogprobScorer):
        def reduce(self, response: Response, prompt: Prompt) -> float:
            return -super().reduce(response, prompt)

    with pytest.raises(SignContractError, match="reversed subtraction"):
        verify_paired_sign(roster.scorers[0], Reversed("pll_total"))


def test_a_text_probe_cannot_be_run_against_a_likelihood_scorer(roster) -> None:  # type: ignore[no-untyped-def]
    class Fake:
        scorer_id = "pll_total"

        def score_texts(self, texts):  # type: ignore[no-untyped-def]
            return [0.0 for _ in texts]

    with pytest.raises(SignContractError, match="no text sign_probe"):
        verify_sign(roster.scorers[0], Fake())


# --------------------------------------------------------------------------------------
# 3. The two shapes refuse each other
# --------------------------------------------------------------------------------------


def test_a_paired_scorer_refuses_a_generated_response(roster) -> None:  # type: ignore[no-untyped-def]
    prompt = _prompt()
    scorer = paired_scorer_from_spec(
        roster.scorers[0],
        prompts_by_id={prompt.prompt_id: prompt},
        benchmark=BENCHMARK,
        scoring_hash=roster.scoring_hash,
    )
    with pytest.raises(ValueError, match="carry no pair_pll log-probabilities"):
        scorer.score([_generated(prompt.prompt_id)])


def test_a_text_scorer_still_refuses_a_paired_response() -> None:
    """P6's refusal survives P2 -- it is now one half of a symmetric pair, not a placeholder."""
    text_roster = load_scoring_config("default_v1")
    spec = next(s for s in text_roster.scorers if s.mechanism == "lexicon")
    prompt = _prompt()

    class Fake:
        scorer_id = spec.scorer_id

        def score_texts(self, texts):  # type: ignore[no-untyped-def]
            return [0.0 for _ in texts]

    from margins.scoring.base import ResponseScorer

    scorer = ResponseScorer(
        spec,
        Fake(),
        prompts_by_id={prompt.prompt_id: prompt},
        benchmark=BENCHMARK,
        scoring_hash=text_roster.scoring_hash,
    )
    with pytest.raises(ValueError, match="pair_pll"):
        scorer.score([_response(prompt, -1.0, -2.0)])


def test_a_likelihood_spec_cannot_be_built_as_a_text_scorer(roster) -> None:  # type: ignore[no-untyped-def]
    """The exhaustive-dispatch fix: this used to fall through to HFClassifierScorer silently."""
    with pytest.raises(ValueError, match="paired_scorer_from_spec"):
        scorer_from_spec(roster.scorers[0], prompts_by_id={}, benchmark=BENCHMARK, scoring_hash="h")


def test_a_text_spec_cannot_be_built_as_a_paired_scorer() -> None:
    text_roster = load_scoring_config("default_v1")
    spec = next(s for s in text_roster.scorers if s.mechanism == "lexicon")
    with pytest.raises(ValueError, match="scores TEXT"):
        PairedScorer(
            spec,
            TotalLogprobScorer(spec.scorer_id),
            prompts_by_id={},
            benchmark=BENCHMARK,
            scoring_hash="h",
        )


# --------------------------------------------------------------------------------------
# 4. One item -> one Response -> one score, and the roster rules
# --------------------------------------------------------------------------------------


def test_one_item_one_response_one_score_per_scorer(roster) -> None:  # type: ignore[no-untyped-def]
    """The pair travels as a single unit end to end (open decisions 6 and 7)."""
    prompt = _prompt()
    response = _response(prompt, stereo=-4.0, antistereo=-10.0)

    for spec in roster.scorers:
        scorer = paired_scorer_from_spec(
            spec,
            prompts_by_id={prompt.prompt_id: prompt},
            benchmark=BENCHMARK,
            scoring_hash=roster.scoring_hash,
        )
        scores = scorer.score([response])
        assert len(scores) == 1
        row = scores[0]
        assert isinstance(row, ItemScore)
        assert row.response_id == response.response_id
        assert row.prompt_id == prompt.prompt_id, "the cluster key must reach the score row"
        assert row.cell_id == prompt.cell_id
        assert row.benchmark == BENCHMARK
        assert row.condition == "model"


def test_paired_scorers_satisfy_the_roster_protocol(roster) -> None:  # type: ignore[no-untyped-def]
    prompt = _prompt()
    scorer = paired_scorer_from_spec(
        roster.scorers[0],
        prompts_by_id={prompt.prompt_id: prompt},
        benchmark=BENCHMARK,
        scoring_hash=roster.scoring_hash,
    )
    assert isinstance(scorer, RosterScorer)
    assert scorer.for_condition("drift_control").condition == "drift_control"


def test_standardization_is_unchanged_by_the_new_paradigm(roster) -> None:  # type: ignore[no-untyped-def]
    """`standardize_within_benchmark` is paradigm-agnostic; CrowS rows flow through untouched."""
    prompts = {}
    responses = []
    for index in range(4):
        prompt = _prompt(index)
        prompts[prompt.prompt_id] = prompt
        responses.append(_response(prompt, stereo=-float(index), antistereo=-5.0))

    spec = roster.scorers[0]
    scorer = paired_scorer_from_spec(
        spec, prompts_by_id=prompts, benchmark=BENCHMARK, scoring_hash=roster.scoring_hash
    )
    raw = scorer.score(responses)
    standardized, reports = standardize_within_benchmark(
        raw, benchmark=BENCHMARK, signs=signs_from_specs([spec])
    )

    assert len(standardized) == 4
    assert not reports[0].degenerate
    assert math.isclose(sum(s.standardized for s in standardized), 0.0, abs_tol=1e-9)


def test_standardization_still_refuses_to_pool_the_two_paradigms(roster) -> None:  # type: ignore[no-untyped-def]
    """The cross-paradigm guardrail, exercised now that a second paradigm actually exists."""
    prompt = _prompt()
    scorer = paired_scorer_from_spec(
        roster.scorers[0],
        prompts_by_id={prompt.prompt_id: prompt},
        benchmark=BENCHMARK,
        scoring_hash=roster.scoring_hash,
    )
    rows = scorer.score([_response(prompt, -1.0, -2.0)])
    with pytest.raises(ValueError, match="WITHIN a source, never across"):
        standardize_within_benchmark(
            rows, benchmark="open_gen_templates", signs={roster.scorers[0].scorer_id: 1}
        )


def test_the_roster_still_refuses_a_single_scorer_harm_type(roster) -> None:  # type: ignore[no-untyped-def]
    """Framing constraint 6, on the new mechanisms."""
    payload = roster.model_dump(mode="json")
    payload["scorers"] = [
        payload["scorers"][0],
        {**payload["scorers"][0], "scorer_id": "other", "harm_type": "something_else"},
    ]
    from margins.scoring.config import scoring_config_from_mapping

    with pytest.raises(ValueError, match="at least"):
        scoring_config_from_mapping(payload)


def test_the_roster_still_refuses_two_scorers_sharing_a_mechanism(roster) -> None:  # type: ignore[no-untyped-def]
    """Two reductions that reduce the same way are one reduction twice."""
    payload = roster.model_dump(mode="json")
    payload["scorers"][1]["mechanism"] = payload["scorers"][0]["mechanism"]
    from margins.scoring.config import scoring_config_from_mapping

    with pytest.raises(ValueError, match="mechanism"):
        scoring_config_from_mapping(payload)


def test_a_likelihood_scorer_may_not_declare_a_pin(roster) -> None:  # type: ignore[no-untyped-def]
    """Its instrument is the model under test, pinned by ModelSpec and on every Response."""
    payload = roster.model_dump(mode="json")
    payload["scorers"][0]["lexicon_pin"] = {
        "name": "x",
        "source": "https://example.invalid",
        "revision": "0" * 40,
        "license": "MIT",
        "sha256": "a" * 64,
        "path": "lexicons/x.txt",
    }
    with pytest.raises(ValueError, match="carries NO pin"):
        ScorerSpec.model_validate(payload["scorers"][0])


def test_a_likelihood_scorer_must_declare_a_paired_probe(roster) -> None:  # type: ignore[no-untyped-def]
    payload = roster.model_dump(mode="json")
    payload["scorers"][0]["paired_sign_probe"] = None
    with pytest.raises(ValueError, match="declares no paired_sign_probe"):
        ScorerSpec.model_validate(payload["scorers"][0])


def test_the_two_reductions_have_distinct_pin_digests(roster) -> None:  # type: ignore[no-untyped-def]
    """`pin_digest` rides in every score_id; two reductions must not collide."""
    digests = {spec.pin_digest for spec in roster.scorers}
    assert len(digests) == len(roster.scorers)
