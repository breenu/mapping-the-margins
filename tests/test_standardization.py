"""The two standardization guardrails, as tests (P6).

CLAUDE.md states both as rules; this file is what makes them fail loudly instead of silently:

* **"standardize WITHIN a source -- the function requires a source key."** Pooling two sources into
  one z-score turns the cross-paradigm comparison into a property of the pooling, and produces
  output that looks entirely reasonable.
* **"higher = more harm, guaranteed by test."** A mis-signed scorer inverts every effect downstream
  while the pipeline stays green. ``verify_sign`` is the guarantee; these tests are its teeth.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from margins.schemas import ItemScore
from margins.scoring.config import ScorerSpec, SignProbe
from margins.scoring.standardize import (
    SignContractError,
    signs_from_specs,
    standardize_within_benchmark,
    verify_sign,
)

BENCHMARK = "open_gen_templates"
OTHER_BENCHMARK = "crows_pairs"


class FixedScorer:
    """Returns pre-set values in order. Enough to drive the sign contract with no model."""

    def __init__(self, scorer_id: str, values: Sequence[float]) -> None:
        self._scorer_id = scorer_id
        self._values = list(values)

    @property
    def scorer_id(self) -> str:
        return self._scorer_id

    def score_texts(self, texts: Sequence[str]) -> list[float]:
        return self._values[: len(texts)]


def make_score(
    index: int,
    raw: float,
    *,
    scorer_id: str = "s1",
    benchmark: str = BENCHMARK,
    harm_type: str = "toxicity",
    condition: str = "model",
) -> ItemScore:
    return ItemScore(
        score_id=f"{scorer_id}::{condition}::r{index}",
        response_id=f"r{index}",
        scorer_id=scorer_id,
        harm_type=harm_type,
        raw=raw,
        standardized=raw,
        prompt_id=f"p{index}",
        cell_id="alpha=a1",
        benchmark=benchmark,
        model_id="m",
        seed=1,
        temperature=0.7,
        condition=condition,  # type: ignore[arg-type]
    )


def make_spec(scorer_id: str, *, sign: int, harmful: str = "H", benign: str = "B") -> ScorerSpec:
    return ScorerSpec(
        scorer_id=scorer_id,
        harm_type="toxicity",
        mechanism="lexicon",
        description="fixture",
        sign=sign,  # type: ignore[arg-type]
        threshold=0.5,
        sign_probe=SignProbe(harmful=harmful, benign=benign),
        lexicon_pin={  # type: ignore[arg-type]
            "name": "fixture",
            "source": "https://example.invalid",
            "license": "MIT",
            "version": "1.0",
        },
    )


# ------------------------------------------------------------------ the source key


def test_the_benchmark_key_is_required() -> None:
    """Keyword-only and no default: a function that can guess its grouping key can be called
    wrong."""
    with pytest.raises(TypeError):
        standardize_within_benchmark([], signs={})  # type: ignore[call-arg]


def test_rows_from_another_source_are_refused() -> None:
    """The whole guardrail: never standardize ACROSS sources."""
    scores = [make_score(0, 1.0), make_score(1, 2.0, benchmark=OTHER_BENCHMARK)]

    with pytest.raises(ValueError, match="WITHIN a source"):
        standardize_within_benchmark(scores, benchmark=BENCHMARK, signs={"s1": 1})


def test_a_scorer_with_no_declared_sign_is_refused() -> None:
    with pytest.raises(ValueError, match="no declared sign"):
        standardize_within_benchmark([make_score(0, 1.0)], benchmark=BENCHMARK, signs={})


def test_an_empty_benchmark_key_is_refused() -> None:
    with pytest.raises(ValueError, match="non-empty source key"):
        standardize_within_benchmark([], benchmark="", signs={})


# ------------------------------------------------------------------ the arithmetic


def test_standardization_is_a_z_score_within_group() -> None:
    scores = [make_score(index, raw) for index, raw in enumerate([1.0, 2.0, 3.0, 4.0, 5.0])]

    out, reports = standardize_within_benchmark(scores, benchmark=BENCHMARK, signs={"s1": 1})

    values = [s.standardized for s in out]
    assert sum(values) == pytest.approx(0.0, abs=1e-9)
    assert values[0] < values[2] < values[-1]
    assert reports[0].raw_mean == pytest.approx(3.0)
    assert reports[0].degenerate is False


def test_a_negative_sign_flips_the_order() -> None:
    """Sign harmonization is what makes 'higher = more harm' hold for a valence-style scorer."""
    scores = [make_score(index, raw) for index, raw in enumerate([1.0, 2.0, 3.0, 4.0, 5.0])]

    positive, _ = standardize_within_benchmark(scores, benchmark=BENCHMARK, signs={"s1": 1})
    negative, _ = standardize_within_benchmark(scores, benchmark=BENCHMARK, signs={"s1": -1})

    assert [s.standardized for s in negative] == pytest.approx(
        [-s.standardized for s in positive], abs=1e-9
    )


def test_groups_are_separate_per_scorer_harm_and_condition() -> None:
    """Raw scales are incomparable across scorers, and the two conditions are different
    populations."""
    scores = [make_score(i, float(i)) for i in range(5)]
    scores += [make_score(i, float(i) * 100, scorer_id="s2") for i in range(5)]
    scores += [make_score(i, float(i), condition="drift_control") for i in range(5)]

    out, reports = standardize_within_benchmark(
        scores, benchmark=BENCHMARK, signs={"s1": 1, "s2": 1}
    )

    assert len(reports) == 3
    s1_model = [s.standardized for s in out if s.scorer_id == "s1" and s.condition == "model"]
    s2_model = [s.standardized for s in out if s.scorer_id == "s2"]
    # Wildly different raw scales, identical standardized values: each was z-scored in its own
    # group.
    assert s1_model == pytest.approx(s2_model, abs=1e-9)


def test_a_zero_variance_group_standardizes_to_zero_and_is_reported() -> None:
    """Not NaN, which would poison every downstream mean, and not a fudged denominator."""
    scores = [make_score(index, 0.5) for index in range(5)]

    out, reports = standardize_within_benchmark(scores, benchmark=BENCHMARK, signs={"s1": 1})

    assert all(s.standardized == 0.0 for s in out)
    assert reports[0].degenerate is True
    assert reports[0].raw_sd == 0.0


def test_a_tiny_group_is_degenerate_rather_than_relabelled() -> None:
    """Two points always have a defined sd, but z-scoring them just relabels them +/-1."""
    out, reports = standardize_within_benchmark(
        [make_score(0, 1.0), make_score(1, 9.0)], benchmark=BENCHMARK, signs={"s1": 1}
    )

    assert reports[0].degenerate is True
    assert all(s.standardized == 0.0 for s in out)


def test_inputs_are_not_mutated_and_order_is_preserved() -> None:
    """Invariant 1: transformations return NEW frozen records."""
    scores = [make_score(index, float(index)) for index in range(5)]
    originals = [s.raw for s in scores]

    out, _ = standardize_within_benchmark(scores, benchmark=BENCHMARK, signs={"s1": 1})

    assert [s.raw for s in scores] == originals
    assert all(s.standardized == s.raw for s in scores)
    assert [s.response_id for s in out] == [s.response_id for s in scores]
    assert all(new is not old for new, old in zip(out, scores, strict=True))


def test_raw_survives_standardization() -> None:
    """Cohen's kappa and the calibration report binarize `raw`, so it must not be overwritten."""
    scores = [make_score(index, float(index)) for index in range(5)]
    out, _ = standardize_within_benchmark(scores, benchmark=BENCHMARK, signs={"s1": 1})
    assert [s.raw for s in out] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_signs_from_specs_builds_the_mapping() -> None:
    specs = [make_spec("a", sign=1), make_spec("b", sign=-1)]
    assert signs_from_specs(specs) == {"a": 1, "b": -1}


# ------------------------------------------------------------------ the sign contract


def test_a_correctly_signed_scorer_passes_its_probe() -> None:
    spec = make_spec("s1", sign=1)
    verify_sign(spec, FixedScorer("s1", [0.9, 0.1]))  # harmful high, benign low


def test_a_mis_signed_scorer_is_caught() -> None:
    """THE gate. The scorer is fine; the roster describes it backwards, and every effect inverts."""
    spec = make_spec("s1", sign=-1)

    with pytest.raises(SignContractError, match="wrong way round"):
        verify_sign(spec, FixedScorer("s1", [0.9, 0.1]))


def test_a_negative_polarity_scorer_needs_the_negative_sign() -> None:
    """A valence-style scorer whose raw output FALLS with harm passes only with sign=-1."""
    falling = FixedScorer("s1", [-0.8, 0.6])  # harmful text scores LOW

    verify_sign(make_spec("s1", sign=-1), falling)

    with pytest.raises(SignContractError):
        verify_sign(make_spec("s1", sign=1), falling)


def test_a_scorer_that_cannot_tell_the_probe_apart_is_caught() -> None:
    """Equal scores are a failure, not a pass: 'higher' must be strict."""
    with pytest.raises(SignContractError):
        verify_sign(make_spec("s1", sign=1), FixedScorer("s1", [0.5, 0.5]))
