"""Gate P9: the correction is Benjamini-Hochberg, over the whole family, and nowhere else.

A silent bug here is a false paper claim, so nothing is asserted against a number the engine once
produced. ``q`` is checked against an independent ``multipletests`` call, the monotonicity gate is
arithmetic, and the "only StatsEngine sets significance" invariant is enforced by **scanning the
source tree** rather than by trusting anyone to remember it.

The subset guard is the load-bearing one. Correcting 100 tests when 168 were pre-registered
applies a more lenient threshold than the pre-registration promised, so more things come out
significant. It is the easiest way to inflate a false-discovery rate by accident, and it looks
like nothing at all in the output.
"""

from __future__ import annotations

import ast
import itertools
from pathlib import Path

import numpy as np
import pytest

from margins.lattice.build import DeclaredLattice
from margins.lattice.config import load_lattice_config
from margins.metrics.design import pair_designs
from margins.schemas import EffectEstimate
from margins.stats.config import (
    DriftEnforcement,
    FdrSettings,
    PermutationSettings,
    PowerSettings,
    StatsConfig,
)
from margins.stats.correct import (
    AlreadyCorrectedError,
    BenjaminiHochbergEngine,
    MissingPValueError,
)
from margins.stats.family import (
    IncompleteFamilyError,
    UnknownFamilyMemberError,
    build_family_plan,
    free_interaction_terms,
)
from margins.stats.inputs import effect_key

REPO_ROOT = Path(__file__).resolve().parents[1]
LATTICE_FIXTURE = Path(__file__).parent / "fixtures" / "lattice"
GROUPS = [("open_gen_templates", "fake/model", "toxicity")]


def make_config(**overrides: object) -> StatsConfig:
    data: dict[str, object] = {
        "key": "test",
        "description": "test",
        "stats_version": "test@2026-09-19",
        "fdr": FdrSettings(alpha=0.05, prds_justification="stated for the test"),
        "permutation": PermutationSettings(n_permutations=199, seed=1),
        "power": PowerSettings(
            frames_per_cell=9,
            frames_grid=(9,),
            seeds_grid=(1, 2),
            icc_rho=(0.2,),
            effect_size=0.5,
            patterns=("concentrated",),
            target_power=0.8,
            n_simulations=100,
            pair=("alpha", "beta"),
            seed=1,
        ),
        "drift": DriftEnforcement(descriptive_only_ratio=0.5),
    }
    data.update(overrides)
    return StatsConfig.model_validate(data)


@pytest.fixture(scope="module")
def lattice() -> DeclaredLattice:
    return DeclaredLattice(load_lattice_config("tiny", config_dir=LATTICE_FIXTURE))


def build(lattice: DeclaredLattice, config: StatsConfig):  # type: ignore[no-untyped-def]
    designs = pair_designs(lattice.cells())
    plan = build_family_plan(
        designs, lattice_config=lattice.config, groups=GROUPS, settings=config.fdr
    )
    return designs, plan


def effects_for(lattice: DeclaredLattice, plan, p_values=None):  # type: ignore[no-untyped-def]
    """One EffectEstimate per declared family slot, with supplied or uniform p-values."""
    members = [m for m in plan.members if m.status == "present"]
    values = p_values if p_values is not None else np.linspace(0.001, 0.9, len(members))
    return [
        EffectEstimate(
            benchmark=member.benchmark,
            model=member.model_id,
            axis_pair=(member.axis_a, member.axis_b),
            term=member.term,
            harm_type=member.harm_type,
            estimate=0.3,
            se=0.1,
            ci_low=0.1,
            ci_high=0.5,
            p=float(value),
            powered=True,
        )
        for member, value in zip(members, values, strict=True)
    ]


# ------------------------------------------------------------------ the family


def test_the_family_is_the_confirmatory_pairs_only(lattice: DeclaredLattice) -> None:
    config = make_config()
    _designs, plan = build(lattice, config)
    pairs = {(m.axis_a, m.axis_b) for m in plan.members}
    assert pairs == set(lattice.config.confirmatory_pairs)
    assert not pairs & set(lattice.config.exploratory_pairs)


def test_family_size_is_the_derived_arithmetic(lattice: DeclaredLattice) -> None:
    """Computed from the designs, never asserted as a magic constant."""
    config = make_config()
    designs, plan = build(lattice, config)
    expected = len(GROUPS) * sum(
        len(free_interaction_terms(d))
        for d in designs
        if d.axis_pair in set(lattice.config.confirmatory_pairs)
    )
    assert plan.size == expected


def test_a_pair_p7_could_not_fit_is_excused_and_shrinks_the_family(
    lattice: DeclaredLattice,
) -> None:
    config = make_config()
    designs = pair_designs(lattice.cells())
    pair = lattice.config.confirmatory_pairs[0]
    reason = "axis pair has 1 empty cell(s)"
    plan = build_family_plan(
        designs,
        lattice_config=lattice.config,
        groups=GROUPS,
        settings=config.fdr,
        not_estimable={(*GROUPS[0], pair[0], pair[1]): reason},
    )
    full = build(lattice, config)[1]
    assert plan.size < full.size
    assert plan.excused
    assert all(member.reason == reason for member in plan.excused)


# ------------------------------------------------------------------ the correction


def test_q_matches_an_independent_multipletests_call(lattice: DeclaredLattice) -> None:
    from statsmodels.stats.multitest import multipletests

    config = make_config()
    _, plan = build(lattice, config)
    effects = effects_for(lattice, plan)
    corrected = BenjaminiHochbergEngine(config, plan=plan).correct(effects)

    _, expected, _, _ = multipletests(
        [float(e.p) for e in effects if e.p is not None], alpha=0.05, method="fdr_bh"
    )
    assert [e.q for e in corrected] == pytest.approx(list(expected))


def test_post_fdr_significance_never_exceeds_pre_fdr(lattice: DeclaredLattice) -> None:
    """Gate P9's headline arithmetic, over many random families rather than one."""
    config = make_config()
    _, plan = build(lattice, config)
    rng = np.random.default_rng(0)
    for _ in range(30):
        values = rng.uniform(0, 1, size=plan.size)
        effects = effects_for(lattice, plan, values)
        corrected = BenjaminiHochbergEngine(config, plan=plan).correct(effects)
        pre = sum(1 for e in effects if e.p is not None and e.p <= 0.05)
        post = sum(1 for e in corrected if e.q is not None and e.q <= 0.05)
        assert post <= pre


def test_every_corrected_estimate_is_fully_stamped(lattice: DeclaredLattice) -> None:
    config = make_config()
    _, plan = build(lattice, config)
    corrected = BenjaminiHochbergEngine(config, plan=plan).correct(effects_for(lattice, plan))
    assert all(e.q is not None for e in corrected)
    assert {e.family_size for e in corrected} == {plan.size}
    assert {e.correction for e in corrected} == {"fdr_bh"}


def test_input_order_is_preserved_and_inputs_are_not_mutated(
    lattice: DeclaredLattice,
) -> None:
    config = make_config()
    _, plan = build(lattice, config)
    effects = effects_for(lattice, plan)
    before = [e.model_dump() for e in effects]
    corrected = BenjaminiHochbergEngine(config, plan=plan).correct(effects)
    assert [e.term for e in corrected] == [e.term for e in effects]
    assert [e.model_dump() for e in effects] == before


def test_q_is_monotone_in_p(lattice: DeclaredLattice) -> None:
    config = make_config()
    _, plan = build(lattice, config)
    effects = effects_for(lattice, plan)
    corrected = BenjaminiHochbergEngine(config, plan=plan).correct(effects)
    pairs = sorted((e.p, e.q) for e in corrected if e.p is not None and e.q is not None)
    assert all(a[1] <= b[1] + 1e-12 for a, b in itertools.pairwise(pairs))


def test_realized_false_discovery_rate_sits_at_or_below_alpha(
    lattice: DeclaredLattice,
) -> None:
    """Under a GLOBAL null every discovery is false, so the rate is checkable directly."""
    config = make_config()
    _, plan = build(lattice, config)
    rng = np.random.default_rng(7)
    families = 200
    any_discovery = 0
    for _ in range(families):
        effects = effects_for(lattice, plan, rng.uniform(0, 1, size=plan.size))
        corrected = BenjaminiHochbergEngine(config, plan=plan).correct(effects)
        if any(e.q is not None and e.q <= 0.05 for e in corrected):
            any_discovery += 1
    # Under the global null the FDR equals the family-wise error rate, bounded by alpha.
    assert any_discovery / families <= 0.10


# ------------------------------------------------------------------ refusals


def test_correcting_a_subset_raises(lattice: DeclaredLattice) -> None:
    config = make_config()
    _, plan = build(lattice, config)
    effects = effects_for(lattice, plan)
    with pytest.raises(IncompleteFamilyError, match="did not arrive"):
        BenjaminiHochbergEngine(config, plan=plan).correct(effects[:-1])


def test_an_unknown_member_raises(lattice: DeclaredLattice) -> None:
    config = make_config()
    _, plan = build(lattice, config)
    effects = effects_for(lattice, plan)
    stranger = effects[0].model_copy(update={"term": "not=a:real=term"})
    with pytest.raises(UnknownFamilyMemberError, match="not in the declared family"):
        BenjaminiHochbergEngine(config, plan=plan).correct([*effects, stranger])


def test_a_confirmatory_estimate_without_a_p_raises(lattice: DeclaredLattice) -> None:
    config = make_config()
    _, plan = build(lattice, config)
    effects = effects_for(lattice, plan)
    effects[0] = effects[0].model_copy(update={"p": None})
    with pytest.raises(MissingPValueError, match="no p-value"):
        BenjaminiHochbergEngine(config, plan=plan).correct(effects)


def test_correcting_twice_raises(lattice: DeclaredLattice) -> None:
    config = make_config()
    _, plan = build(lattice, config)
    engine = BenjaminiHochbergEngine(config, plan=plan)
    once = engine.correct(effects_for(lattice, plan))
    with pytest.raises(AlreadyCorrectedError, match="already carry a q"):
        engine.correct(once)


def test_exploratory_estimates_pass_through_as_the_same_objects(
    lattice: DeclaredLattice,
) -> None:
    """Not copied with nulls: identical objects, so a q cannot appear on one by accident."""
    config = make_config()
    _, plan = build(lattice, config)
    exploratory_pair = lattice.config.exploratory_pairs[0]
    stray = EffectEstimate(
        benchmark=GROUPS[0][0],
        model=GROUPS[0][1],
        axis_pair=exploratory_pair,
        term="alpha=a1:gamma=g1",
        harm_type=GROUPS[0][2],
        estimate=0.9,
        se=0.1,
        ci_low=0.7,
        ci_high=1.1,
        powered=True,
    )
    effects = [*effects_for(lattice, plan), stray]
    corrected = BenjaminiHochbergEngine(config, plan=plan).correct(effects)
    assert corrected[-1] is stray
    assert stray.q is None and stray.family_size is None and stray.correction is None
    assert not plan.is_confirmatory(effect_key(stray))


# ------------------------------------------------------------------ invariant 3, executable


def test_only_the_correct_module_sets_q_family_size_or_correction() -> None:
    """Invariant 3, enforced by scanning the source rather than by convention.

    The invariant is about :class:`~margins.schemas.EffectEstimate` specifically, so the scan is
    too: it flags an ``EffectEstimate(...)`` construction carrying one of the three fields, and
    any ``model_copy(update={...})`` that writes one. Other records legitimately have a
    ``family_size`` of their own -- the manifest block records it, a simulated power point
    carries the family it was corrected against -- and flagging those would be noise that trains
    people to ignore the check.
    """
    forbidden = {"q", "family_size", "correction"}
    allowed = {Path("src/margins/stats/correct.py")}
    offenders: list[str] = []

    def keys_of(node: ast.Dict) -> set[str]:
        return {
            k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)
        }

    for path in (REPO_ROOT / "src").rglob("*.py"):
        relative = path.relative_to(REPO_ROOT)
        if relative in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = node.func
            constructs_effect = isinstance(callee, ast.Name) and callee.id == "EffectEstimate"
            is_model_copy = isinstance(callee, ast.Attribute) and callee.attr == "model_copy"

            if constructs_effect:
                for keyword in node.keywords:
                    if keyword.arg in forbidden:
                        offenders.append(
                            f"{relative}:{node.lineno} EffectEstimate({keyword.arg}=...)"
                        )
            if is_model_copy:
                for keyword in node.keywords:
                    if keyword.arg == "update" and isinstance(keyword.value, ast.Dict):
                        for name in sorted(keys_of(keyword.value) & forbidden):
                            offenders.append(
                                f"{relative}:{node.lineno} model_copy(update={{{name!r}: ...}})"
                            )
    assert not offenders, (
        "significance is set outside margins.stats.correct (invariant 3): " + "; ".join(offenders)
    )
