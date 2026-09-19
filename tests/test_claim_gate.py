"""Gate P9: the drift downgrade is enforced, and a refused claim crashes rather than passing.

Two gates in one file because they are one mechanism: :mod:`margins.stats.descriptive` decides
which harm types lose their claims, and :mod:`margins.stats.claims` is what makes that decision
impossible to bypass.

The design point the tests exist to protect is that ``is_significant`` **raises** on a refused
effect. Returning ``False`` would be the friendlier API and the wrong one: a report builder that
forgot the gate would then render every descriptive-only effect as "not significant", which is a
claim about the model where the honest output is an abstention.
"""

from __future__ import annotations

import pytest

from margins.metrics.records import DriftComparison, HarmTypeFlag
from margins.schemas import EffectEstimate
from margins.stats.claims import ClaimGate, ClaimRefusedError
from margins.stats.descriptive import (
    compare_with_metrics_flags,
    descriptive_only_decisions,
    descriptive_only_keys,
)

BENCHMARK = "open_gen_templates"
MODEL = "fake/model"


def comparison(
    *, harm_type: str = "toxicity", model_rms: float = 1.0, drift_rms: float = 0.1, **overrides
) -> DriftComparison:
    data = {
        "benchmark": BENCHMARK,
        "model_id": MODEL,
        "harm_type": harm_type,
        "scorer_id": "toxic_bert",
        "axis_a": "alpha",
        "axis_b": "beta",
        "control_family": "prompt_echo",
        "model_rms": model_rms,
        "drift_rms": drift_rms,
        # DELIBERATELY WRONG on some fixtures: P9 must not inherit it.
        "ratio": None if model_rms == 0 else drift_rms / model_rms,
        "comparable": False,
        "n_model_obs": 100,
        "n_drift_obs": 50,
    }
    data.update(overrides)
    return DriftComparison.model_validate(data)


def effect(**overrides) -> EffectEstimate:
    data = {
        "benchmark": BENCHMARK,
        "model": MODEL,
        "axis_pair": ("alpha", "beta"),
        "term": "alpha=a1:beta=b1",
        "harm_type": "toxicity",
        "estimate": 0.6,
        "se": 0.1,
        "ci_low": 0.4,
        "ci_high": 0.8,
        "p": 0.001,
        "q": 0.01,
        "powered": True,
        "family_size": 168,
        "correction": "fdr_bh",
    }
    data.update(overrides)
    return EffectEstimate.model_validate(data)


def gate(**overrides) -> ClaimGate:
    data = {
        "exploratory_pairs": frozenset(),
        "descriptive_only": frozenset(),
        "alpha": 0.05,
    }
    data.update(overrides)
    return ClaimGate(**data)  # type: ignore[arg-type]


# ------------------------------------------------------------------ the drift decision


def test_at_or_above_the_threshold_is_descriptive_only() -> None:
    decisions = descriptive_only_decisions([comparison(drift_rms=0.5)], threshold=0.5)
    assert decisions[0].descriptive_only is True
    assert decisions[0].max_ratio == pytest.approx(0.5)


def test_strictly_below_the_threshold_is_clear() -> None:
    decisions = descriptive_only_decisions([comparison(drift_rms=0.49)], threshold=0.5)
    assert decisions[0].descriptive_only is False


def test_any_scorer_pair_or_family_over_threshold_downgrades_the_harm_type() -> None:
    rows = [
        comparison(drift_rms=0.1),
        comparison(
            drift_rms=0.9, scorer_id="ldnoobw_lexicon", control_family="neutral_declarative"
        ),
    ]
    decision = descriptive_only_decisions(rows, threshold=0.5)[0]
    assert decision.descriptive_only is True
    assert decision.scorers_over_threshold == ("ldnoobw_lexicon",)
    assert decision.worst_scorer == "ldnoobw_lexicon"
    assert decision.n_over_threshold == 1 and decision.n_comparisons == 2


def test_p7s_comparable_column_is_ignored() -> None:
    """The fixture's flag says the opposite of the truth. P9 must recompute from raw columns."""
    lying = comparison(drift_rms=0.9, comparable=False)
    assert descriptive_only_decisions([lying], threshold=0.5)[0].descriptive_only is True
    other = comparison(drift_rms=0.01, comparable=True)
    assert descriptive_only_decisions([other], threshold=0.5)[0].descriptive_only is False


def test_zero_model_interaction_with_nonzero_drift_is_descriptive_only() -> None:
    """P7 records ratio=None here. It is the WORST case, not a missing one."""
    row = comparison(model_rms=0.0, drift_rms=0.3)
    assert row.ratio is None
    decision = descriptive_only_decisions([row], threshold=0.5)[0]
    assert decision.descriptive_only is True
    assert decision.max_ratio is None  # infinite is reported as absent, never as a number


def test_a_disagreement_with_p7_is_reported_not_swallowed() -> None:
    flag = HarmTypeFlag(
        benchmark=BENCHMARK,
        model_id=MODEL,
        harm_type="toxicity",
        descriptive_only_candidate=False,
        n_comparisons=1,
        n_comparable=0,
    )
    decisions = descriptive_only_decisions([comparison(drift_rms=0.9)], [flag], threshold=0.5)
    assert decisions[0].disagrees_with_metrics_flag is True
    lines = compare_with_metrics_flags(decisions)
    assert len(lines) == 1 and "DESCRIPTIVE-ONLY" in lines[0]


def test_threshold_is_recorded_on_the_decision() -> None:
    assert descriptive_only_decisions([comparison()], threshold=0.42)[0].threshold == 0.42


# ------------------------------------------------------------------ the gate


def test_a_clean_effect_is_claimable() -> None:
    assert gate().is_significant(effect()) is True
    assert gate().status(effect()) == "claimable"
    assert gate().refusal_reason(effect()) is None


def test_is_significant_raises_rather_than_returning_false() -> None:
    """THE design point. A builder that forgets the gate must crash, not print a star."""
    downgraded = gate(descriptive_only=frozenset({(BENCHMARK, MODEL, "toxicity")}))
    with pytest.raises(ClaimRefusedError, match="DESCRIPTIVE-ONLY"):
        downgraded.is_significant(effect())


@pytest.mark.parametrize(
    ("kwargs", "overrides", "status", "fragment"),
    [
        ({"exploratory_pairs": frozenset({("alpha", "beta")})}, {}, "exploratory", "exploratory"),
        (
            {"descriptive_only": frozenset({(BENCHMARK, MODEL, "toxicity")})},
            {},
            "descriptive_only",
            "DESCRIPTIVE-ONLY",
        ),
        ({}, {"powered": False}, "underpowered", "underpowered"),
        ({}, {"q": None, "family_size": None, "correction": None}, "uncorrected", "no q"),
    ],
)
def test_each_refusal_names_its_own_reason(kwargs, overrides, status, fragment) -> None:
    subject = effect(**overrides)
    engine = gate(**kwargs)
    assert engine.status(subject) == status
    with pytest.raises(ClaimRefusedError, match=fragment):
        engine.require_claimable(subject)


def test_powered_only_is_the_default_and_opting_out_is_explicit() -> None:
    unpowered = effect(powered=False)
    assert gate().status(unpowered) == "underpowered"
    assert gate(powered_only=False).status(unpowered) == "claimable"


def test_a_claim_row_exists_for_every_effect_including_refused_ones() -> None:
    """A refusal must be visible in the artifact, not an absence indistinguishable from silence."""
    engine = gate(descriptive_only=frozenset({(BENCHMARK, MODEL, "toxicity")}))
    effects = [effect(), effect(term="alpha=a1:beta=b2")]
    claims = engine.claims(effects)
    assert len(claims) == len(effects)
    assert all(not claim.claimable for claim in claims)
    assert all(claim.refusal_reason for claim in claims)
    # And the number itself survives: a refusal is not a deletion.
    assert all(claim.q == 0.01 and claim.estimate == 0.6 for claim in claims)


def test_descriptive_only_keys_feeds_the_gate() -> None:
    decisions = descriptive_only_decisions([comparison(drift_rms=0.9)], threshold=0.5)
    keys = descriptive_only_keys(decisions)
    assert keys == frozenset({(BENCHMARK, MODEL, "toxicity")})
    with pytest.raises(ClaimRefusedError):
        gate(descriptive_only=keys).is_significant(effect())
