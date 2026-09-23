"""B3 on real-shaped data: the permutation null from item-level scores, joined to P7's estimates.

The permutation MACHINERY (``margins.stats.permute``) is tested in ``test_permutation.py``. What is
tested here is the P11 wiring that makes it run on an actual run: scores -> P7's own triangulation
-> the same pair frame the estimator fitted -> a p per FDR-family term, on the claim rows.
"""

from __future__ import annotations

import numpy as np
import pytest

from margins.lattice import DeclaredLattice, load_lattice_config
from margins.metrics.config import load_metrics_config
from margins.metrics.design import pair_designs
from margins.metrics.metric import EmergenceMetric
from margins.schemas import EffectEstimate, ItemScore
from margins.stats.b3 import block_frame_of_item, run_b3
from margins.stats.config import PermutationSettings
from margins.stats.family import free_interaction_terms
from margins.stats.inputs import EffectKey
from tests.p11_synthetic import OPEN_GEN_ROSTER, grid, model_scores

PAIR = ("age", "socioeconomic_class")
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SETTINGS = PermutationSettings(n_permutations=199, seed=20260919)


@pytest.fixture(scope="module")
def lattice() -> DeclaredLattice:
    return DeclaredLattice(load_lattice_config("confirmatory_v1"))


def _scores(lattice: DeclaredLattice, *, gamma: float, seed: int) -> list[ItemScore]:
    return model_scores(
        grid(lattice, PAIR),
        benchmark="open_gen_templates",
        model_id=MODEL,
        roster=OPEN_GEN_ROSTER,
        n_items=4,
        seeds=(1, 2, 3),
        gamma_scale=gamma,
        rng=np.random.default_rng(seed),
    )


def _effects(lattice: DeclaredLattice, scores: list[ItemScore]) -> list[EffectEstimate]:
    config = load_metrics_config()
    config = config.model_copy(
        update={"bootstrap": config.bootstrap.model_copy(update={"n_resamples": 50})}
    )
    return EmergenceMetric(config, min_cell_n=108).compute_result(scores, lattice.cells()).effects


def _member(lattice: DeclaredLattice):  # type: ignore[no-untyped-def]
    free = {
        (design.axis_pair, term)
        for design in pair_designs(lattice.cells())
        if frozenset(design.axis_pair) == frozenset(PAIR)
        for term in free_interaction_terms(design)
    }
    return lambda key: (key[2], key[3]) in free


def _run(lattice: DeclaredLattice, scores, effects, settings=SETTINGS):  # type: ignore[no-untyped-def]
    return run_b3(
        scores,
        lattice.cells(),
        pair_designs(lattice.cells()),
        effects,
        is_family_member=_member(lattice),
        settings=settings,
    )


def test_block_is_template_and_carrier() -> None:
    """Decided 23 Sep 2026: items in a block differ ONLY in the permuted axis."""
    assert block_frame_of_item("describe::age=young|x=y::person::abc123") == "describe::person"
    assert block_frame_of_item("describe::age=young::individual::abc") == "describe::individual"
    with pytest.raises(ValueError, match="permutation block"):
        block_frame_of_item("crows_pairs/0001")


def test_b3_attaches_a_p_to_every_family_term_and_only_those(lattice: DeclaredLattice) -> None:
    scores = _scores(lattice, gamma=0.8, seed=1)
    effects = _effects(lattice, scores)
    result = _run(lattice, scores, effects)
    member = _member(lattice)
    members = [
        e for e in effects if member((e.benchmark, e.model, e.axis_pair, e.term, e.harm_type))
    ]
    assert result.n_terms == len(members) == 6 * 2  # (3-1)(4-1) free terms x 2 harm types
    assert all(1 / 200 <= p <= 1.0 for p in result.p_values.values())  # never zero
    assert {s.harm_type for s in result.series} == {"toxicity", "regard"}
    assert all(s.exchangeable_items == s.n_items for s in result.series)


def test_b3_detects_an_injected_interaction_and_is_quiet_without_one(
    lattice: DeclaredLattice,
) -> None:
    strong = _scores(lattice, gamma=1.5, seed=2)
    p_strong = _run(lattice, strong, _effects(lattice, strong)).p_values.values()
    null = _scores(lattice, gamma=0.0, seed=3)
    p_null = list(_run(lattice, null, _effects(lattice, null)).p_values.values())
    assert min(p_strong) <= 0.01
    # A null that fabricated significance would pile up small p-values.
    assert float(np.median(p_null)) > 0.2


def test_b3_is_deterministic_and_order_independent(lattice: DeclaredLattice) -> None:
    scores = _scores(lattice, gamma=0.8, seed=4)
    effects = _effects(lattice, scores)
    first = dict(_run(lattice, scores, effects).p_values)
    assert dict(_run(lattice, list(reversed(scores)), list(reversed(effects))).p_values) == first


def test_b3_refuses_scores_that_are_not_the_fitted_ones(lattice: DeclaredLattice) -> None:
    """The join check: a p beside an estimate from a DIFFERENT series is a silent mis-join."""
    scores = _scores(lattice, gamma=0.8, seed=5)
    effects = _effects(lattice, scores)
    other = _scores(lattice, gamma=0.8, seed=6)
    with pytest.raises(ValueError, match="join check failed"):
        _run(lattice, other, effects)
    with pytest.raises(ValueError, match="no item-level scores"):
        _run(lattice, [s for s in scores if s.harm_type != "regard"], effects)


def test_b3_skips_cleanly_when_the_family_is_empty(lattice: DeclaredLattice) -> None:
    scores = _scores(lattice, gamma=0.8, seed=7)
    effects = _effects(lattice, scores)

    def nobody(key: EffectKey) -> bool:
        return False

    result = run_b3(
        scores,
        lattice.cells(),
        pair_designs(lattice.cells()),
        effects,
        is_family_member=nobody,
        settings=SETTINGS,
    )
    assert result.n_terms == 0
