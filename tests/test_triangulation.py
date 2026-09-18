"""Gate P6's agreement clause: alpha and kappa are computed and REPORTED (P6).

Agreement is a first-class artifact, not an internal assertion -- section 9 makes it a precondition
for a harm type being load-bearing, and a number that only exists inside an `assert` cannot be
reported, downgraded against, or audited from a manifest.

These tests also pin what agreement does NOT license. `test_two_identical_scorers_agree_perfectly`
exists to make the limitation concrete: two scorers that are the same function score alpha = 1.0,
which is exactly the case section 9 warns about. High agreement means shared behaviour; validity is
the drift control's question and the calibration's, not this one's.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from margins.schemas import Prompt, Response, TaskType
from margins.scoring.base import ResponseScorer
from margins.scoring.config import ScoringConfig, load_scoring_config
from margins.scoring.triangulate import (
    MIN_UNITS_FOR_AGREEMENT,
    Triangulator,
    cohen_kappa,
    krippendorff_alpha,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "scoring"
BENCHMARK = "open_gen_templates"
N_RESPONSES = 30


class LinearScorer:
    """Raw score rises with text length. Deterministic, no model."""

    def __init__(self, scorer_id: str, *, slope: float = 0.02, offset: float = 0.0) -> None:
        self._scorer_id = scorer_id
        self._slope = slope
        self._offset = offset

    @property
    def scorer_id(self) -> str:
        return self._scorer_id

    def score_texts(self, texts: Sequence[str]) -> list[float]:
        return [min(1.0, self._offset + self._slope * len(t)) for t in texts]


class InvertedScorer(LinearScorer):
    """Raw score FALLS with text length -- maximal disagreement with a LinearScorer."""

    def score_texts(self, texts: Sequence[str]) -> list[float]:
        return [max(0.0, 1.0 - 0.02 * len(t)) for t in texts]


class ConstantScorer(LinearScorer):
    def score_texts(self, texts: Sequence[str]) -> list[float]:
        return [0.5 for _ in texts]


@pytest.fixture(scope="module")
def config() -> ScoringConfig:
    return load_scoring_config("tiny", config_dir=FIXTURE_DIR)


@pytest.fixture(scope="module")
def prompts() -> dict[str, Prompt]:
    return {
        f"p{i}": Prompt(
            prompt_id=f"p{i}",
            item_id="open_gen_templates/neutral",
            cell_id="alpha=a1" if i % 2 else "alpha=a1|beta=b1",
            rendered_text="Write about someone.",
            task_type=TaskType.OPEN_GEN,
            template_version="tiny@test",
            decoding_key="greedy_test",
        )
        for i in range(N_RESPONSES)
    }


@pytest.fixture(scope="module")
def responses() -> list[Response]:
    return [
        Response(
            response_id=f"r{i}",
            prompt_id=f"p{i}",
            model_id="m",
            model_revision="0" * 40,
            seed=20260918 + i,
            temperature=0.7,
            decoding_key="sampled_v1",
            decoding_hash="0" * 64,
            quant="nf4",
            text="x" * (i + 1),
        )
        for i in range(N_RESPONSES)
    ]


def make_triangulator(
    config: ScoringConfig, prompts: dict[str, Prompt], text_scorers: dict[str, object]
) -> Triangulator:
    scorers = []
    for spec in config.scorers:
        scorers.append(
            ResponseScorer(
                spec,
                text_scorers[spec.scorer_id],  # type: ignore[arg-type]
                prompts_by_id=prompts,
                benchmark=BENCHMARK,
                scoring_hash=config.scoring_hash,
            )
        )
    return Triangulator(config, scorers, benchmark=BENCHMARK)


def default_scorers() -> dict[str, object]:
    return {
        "fake_toxic_nn": LinearScorer("fake_toxic_nn"),
        "fake_toxic_lex": LinearScorer("fake_toxic_lex", slope=0.019, offset=0.01),
        "fake_regard_nn": LinearScorer("fake_regard_nn"),
        "fake_regard_lex": InvertedScorer("fake_regard_lex"),
    }


# ------------------------------------------------------------------ scoring


def test_every_scorer_scores_every_response(
    config: ScoringConfig, prompts: dict[str, Prompt], responses: list[Response]
) -> None:
    scores, _ = make_triangulator(config, prompts, default_scorers()).score_all(responses)

    assert len(scores) == N_RESPONSES * len(config.scorers)
    assert {s.scorer_id for s in scores} == {s.scorer_id for s in config.scorers}


def test_scores_are_standardized_within_the_benchmark(
    config: ScoringConfig, prompts: dict[str, Prompt], responses: list[Response]
) -> None:
    scores, reports = make_triangulator(config, prompts, default_scorers()).score_all(responses)

    assert {s.benchmark for s in scores} == {BENCHMARK}
    # One report per (benchmark, scorer, harm, condition) group.
    assert len(reports) == len(config.scorers)
    for spec in config.scorers:
        group = [s.standardized for s in scores if s.scorer_id == spec.scorer_id]
        assert sum(group) == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------------ agreement


def test_agreement_is_computed_and_appears_in_the_output(
    config: ScoringConfig, prompts: dict[str, Prompt], responses: list[Response]
) -> None:
    """Gate P6: inter-scorer alpha/kappa REPORTED, per harm type."""
    triangulator = make_triangulator(config, prompts, default_scorers())
    scores, _ = triangulator.score_all(responses)

    records = triangulator.agreement(scores)

    metrics = {(r.harm_type, r.metric) for r in records}
    assert metrics == {
        ("toxicity", "krippendorff_alpha"),
        ("toxicity", "cohen_kappa"),
        ("regard", "krippendorff_alpha"),
        ("regard", "cohen_kappa"),
    }
    assert all(r.n_units == N_RESPONSES for r in records)
    assert all(r.value is not None for r in records)


def test_near_identical_scorers_agree_and_opposed_scorers_do_not(
    config: ScoringConfig, prompts: dict[str, Prompt], responses: list[Response]
) -> None:
    """The alpha must actually track agreement, or reporting it is theatre."""
    triangulator = make_triangulator(config, prompts, default_scorers())
    scores, _ = triangulator.score_all(responses)
    records = triangulator.agreement(scores)

    alpha = {r.harm_type: r.value for r in records if r.metric == "krippendorff_alpha" and r.value}
    # toxicity: two near-identical linear scorers. regard: one linear, one inverted.
    assert alpha["toxicity"] > 0.9
    assert alpha["regard"] < 0.0


def test_two_identical_scorers_agree_perfectly(
    config: ScoringConfig, prompts: dict[str, Prompt], responses: list[Response]
) -> None:
    """Documents the LIMIT of agreement: the same function twice scores alpha = 1.0.

    This is section 9's warning made concrete. Two identity-sensitive classifiers would look exactly
    like this, and be wrong together. It is why the roster forbids two scorers sharing a mechanism,
    and why the drift control exists beside this number rather than downstream of it.
    """
    scorers = default_scorers()
    scorers["fake_toxic_lex"] = LinearScorer("fake_toxic_lex")  # identical to fake_toxic_nn
    triangulator = make_triangulator(config, prompts, scorers)
    scores, _ = triangulator.score_all(responses)

    records = triangulator.agreement(scores)
    alpha = next(
        r.value for r in records if r.harm_type == "toxicity" and r.metric == "krippendorff_alpha"
    )
    assert alpha == pytest.approx(1.0, abs=1e-6)


def test_agreement_is_withheld_below_the_minimum_unit_count(
    config: ScoringConfig, prompts: dict[str, Prompt], responses: list[Response]
) -> None:
    """A kappa on four responses invites exactly the over-reading the powered split prevents."""
    few = responses[: MIN_UNITS_FOR_AGREEMENT - 1]
    triangulator = make_triangulator(config, prompts, default_scorers())
    scores, _ = triangulator.score_all(few)

    records = triangulator.agreement(scores)
    assert records
    assert all(r.value is None for r in records)
    assert all(r.n_units == len(few) for r in records)


def test_a_zero_variance_group_reports_no_alpha(
    config: ScoringConfig, prompts: dict[str, Prompt], responses: list[Response]
) -> None:
    """Undefined, not 1.0: no observed disagreement AND no expected disagreement."""
    scorers = default_scorers()
    scorers["fake_toxic_nn"] = ConstantScorer("fake_toxic_nn")
    scorers["fake_toxic_lex"] = ConstantScorer("fake_toxic_lex")
    triangulator = make_triangulator(config, prompts, scorers)
    scores, _ = triangulator.score_all(responses)

    records = triangulator.agreement(scores)
    alpha = next(
        r for r in records if r.harm_type == "toxicity" and r.metric == "krippendorff_alpha"
    )
    assert alpha.value is None


def test_agreement_separates_the_two_conditions(
    config: ScoringConfig, prompts: dict[str, Prompt], responses: list[Response]
) -> None:
    """Agreement about neutral control text is a different question from agreement about output."""
    triangulator = make_triangulator(config, prompts, default_scorers())
    model_scores, _ = triangulator.score_all(responses)
    control_scores, _ = triangulator.score_all(responses, condition="drift_control")

    records = triangulator.agreement([*model_scores, *control_scores])
    assert {r.condition for r in records} == {"model", "drift_control"}


def test_cohen_kappa_handles_two_constant_raters() -> None:
    """scikit-learn returns NaN; 0.0 is the honest reading and keeps the report readable."""
    assert cohen_kappa([1, 1, 1], [1, 1, 1]) == 0.0


# ------------------------------------------------------------------ alpha, against an oracle


def test_the_closed_form_alpha_matches_the_krippendorff_package() -> None:
    """The dependency is the ORACLE; the closed form is the production path.

    `krippendorff.alpha` at interval level builds a coincidence matrix over DISTINCT OBSERVED
    VALUES. With continuous scores every value is distinct, so it is O(n * V**2): at 1,430 units it
    asks for 22 GiB, and the confirmatory design is ~49,500 renderings. Subsampling to fit would
    make the reported alpha a property of the subsample, so the arithmetic is done directly.

    This test is what keeps "done directly" honest -- it checks the two agree to floating-point
    precision, on inputs small enough for the package, with and without missing ratings.
    """
    krippendorff = pytest.importorskip("krippendorff")
    rng = np.random.default_rng(20260918)

    for trial in range(8):
        raters = int(rng.integers(2, 5))
        units = int(rng.integers(5, 40))
        matrix = rng.normal(size=(raters, units))
        if trial % 2:
            matrix[rng.random(matrix.shape) < 0.2] = np.nan

        mine = krippendorff_alpha(matrix)
        oracle = float(krippendorff.alpha(reliability_data=matrix, level_of_measurement="interval"))
        assert mine == pytest.approx(oracle, abs=1e-12)


def test_alpha_is_one_for_identical_raters_and_not_estimable_for_constant_ones() -> None:
    identical = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    assert krippendorff_alpha(identical) == pytest.approx(1.0, abs=1e-12)

    # No variance anywhere: expected disagreement is 0 and alpha is undefined, not 1.0.
    assert np.isnan(krippendorff_alpha(np.full((2, 5), 0.5)))


def test_alpha_scales_past_the_packages_memory_limit() -> None:
    """The case that motivated the closed form: the package cannot even run this one."""
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(4, 20_000))

    value = krippendorff_alpha(matrix)
    assert -1.0 <= value <= 1.0


def test_units_rated_only_once_are_excluded() -> None:
    """A single rating carries no information about disagreement, as in the coincidence matrix."""
    with_singleton = np.array([[1.0, 2.0, 3.0, 9.0], [1.5, 2.5, 2.0, np.nan]])
    without = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 2.0]])

    assert krippendorff_alpha(with_singleton) == pytest.approx(krippendorff_alpha(without))


# ------------------------------------------------------------------ the >=2 rule, again


def test_a_single_scorer_subset_is_refused(
    config: ScoringConfig, prompts: dict[str, Prompt]
) -> None:
    """A valid roster can still be SUBSET into a violation, so the check runs on what was built."""
    spec = config.scorers[0]
    lone = ResponseScorer(
        spec,
        LinearScorer(spec.scorer_id),  # type: ignore[arg-type]
        prompts_by_id=prompts,
        benchmark=BENCHMARK,
        scoring_hash="h",
    )

    with pytest.raises(ValueError, match="single classifier"):
        Triangulator(config, [lone], benchmark=BENCHMARK)


def test_a_scorer_outside_the_roster_is_refused(
    config: ScoringConfig, prompts: dict[str, Prompt]
) -> None:
    """Every row's scorer_id must resolve to a pinned roster entry, or the manifest cannot say what
    produced it."""
    stranger = config.scorers[0].model_copy(update={"scorer_id": "not_in_roster"})
    scorers = [
        ResponseScorer(
            stranger,
            LinearScorer("not_in_roster"),  # type: ignore[arg-type]
            prompts_by_id=prompts,
            benchmark=BENCHMARK,
            scoring_hash="h",
        ),
        ResponseScorer(
            config.scorers[1],
            LinearScorer(config.scorers[1].scorer_id),  # type: ignore[arg-type]
            prompts_by_id=prompts,
            benchmark=BENCHMARK,
            scoring_hash="h",
        ),
    ]

    with pytest.raises(ValueError, match="not in roster"):
        Triangulator(config, scorers, benchmark=BENCHMARK)
