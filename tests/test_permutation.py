"""Gate P9 (B3, deferred from P7): the permutation null respects structure and invents nothing.

Section 10 defers this here and gate P7 leaves the clause open, so this is where it closes. The
failure mode is specific and quiet: a shuffle that breaks the clustering produces a null
distribution that is too NARROW, and a too-narrow null manufactures significance out of noise.
Nothing about the output looks wrong when that happens.

So the structure is asserted on the permuted index arrays themselves, not inferred from the
resulting p-values: whole items move, cell item counts are preserved, and the stratum margin is
untouched. Then calibration is checked over repeated datasets rather than one seed, because a
correct null still produces an extreme p about 5% of the time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from margins.lattice.build import DeclaredLattice
from margins.lattice.config import load_lattice_config
from margins.metrics.design import PairDesign, pair_design
from margins.metrics.estimator import PairFrame
from margins.stats.permute import permutation_p, permutation_plan, permute_interaction

LATTICE_FIXTURE = Path(__file__).parent / "fixtures" / "lattice"
PAIR = ("alpha", "beta")
FRAMES = 6
SEEDS = 4


@pytest.fixture(scope="module")
def design() -> PairDesign:
    lattice = DeclaredLattice(load_lattice_config("tiny", config_dir=LATTICE_FIXTURE))
    return pair_design(lattice.cells(), PAIR)


def synthetic(
    design: PairDesign,
    *,
    interaction: float,
    rng: np.random.Generator,
    main_a: float = 0.8,
    main_b: float = 0.6,
    item_sd: float = 0.5,
    noise_sd: float = 0.5,
) -> tuple[PairFrame, dict[str, str]]:
    """Clustered data with real MAIN effects and a controllable interaction."""
    levels_a, levels_b = design.levels_a, design.levels_b
    y: list[float] = []
    value_a: list[str] = []
    value_b: list[str] = []
    item: list[str] = []
    frame_of_item: dict[str, str] = {}
    for i, a in enumerate(levels_a):
        for j, b in enumerate(levels_b):
            mean = main_a * (i - (len(levels_a) - 1) / 2) + main_b * (j - (len(levels_b) - 1) / 2)
            if i == 0 and j == 0:
                mean += interaction
            for frame in range(FRAMES):
                item_id = f"{a}|{b}::frame{frame}"
                frame_of_item[item_id] = f"frame{frame}"
                u = rng.normal(0.0, item_sd)
                for _ in range(SEEDS):
                    y.append(mean + u + rng.normal(0.0, noise_sd))
                    value_a.append(a)
                    value_b.append(b)
                    item.append(item_id)
    return (
        PairFrame(
            y=np.asarray(y), value_a=tuple(value_a), value_b=tuple(value_b), item=tuple(item)
        ),
        frame_of_item,
    )


def observed_interaction(frame: PairFrame, design: PairDesign) -> np.ndarray:
    from margins.metrics.estimator import lstsq_params

    matrix = design.matrix(frame.value_a, frame.value_b)
    contrasts = design.contrasts
    rows = design.contrast_matrix[[c.kind == "interaction" for c in contrasts]]
    return rows @ lstsq_params(matrix, frame.y)


# ------------------------------------------------------------------ structure


def test_whole_items_move_together_and_cells_keep_their_counts(design: PairDesign) -> None:
    """Asserted on the plan itself. A split item would destroy the clustering silently."""
    rng = np.random.default_rng(0)
    frame, frame_of_item = synthetic(design, interaction=0.0, rng=rng)
    plan = permutation_plan(frame, design, frame_of_item=frame_of_item, stratum_axis="a")

    # Every item's rows are contiguous in the plan and belong to exactly one cell.
    for rows in plan.item_rows:
        cells = {(frame.value_a[r], frame.value_b[r]) for r in rows}
        assert len(cells) == 1
        assert len(rows) == SEEDS

    # A block is (frame, stratum value): its items share the A value and differ in B.
    for block in plan.blocks:
        a_values = {frame.value_a[plan.item_rows[i][0]] for i in block}
        b_values = {frame.value_b[plan.item_rows[i][0]] for i in block}
        frames = {frame_of_item[plan.item_labels[i]] for i in block}
        assert len(a_values) == 1, "a block must not cross the stratum axis"
        assert len(frames) == 1, "a block must not cross template frames"
        assert len(b_values) == len(block), "the permuted axis varies within a block"


def test_every_replicate_keeps_the_design_identified(design: PairDesign) -> None:
    """A replicate that empties a cell is not a replicate; the block structure prevents it."""
    rng = np.random.default_rng(1)
    frame, frame_of_item = synthetic(design, interaction=0.0, rng=rng)
    plan = permutation_plan(frame, design, frame_of_item=frame_of_item, stratum_axis="a")
    null = permute_interaction(
        frame, design, plan, scheme="within_stratum", n_permutations=50, rng=rng
    )
    assert np.isfinite(null).all()
    assert null.shape == (50, design.n_cells)


def test_an_item_with_no_frame_is_refused(design: PairDesign) -> None:
    rng = np.random.default_rng(2)
    frame, frame_of_item = synthetic(design, interaction=0.0, rng=rng)
    frame_of_item.pop(next(iter(frame_of_item)))
    with pytest.raises(KeyError, match="no template frame"):
        permutation_plan(frame, design, frame_of_item=frame_of_item, stratum_axis="a")


def test_the_stratum_axis_must_be_named(design: PairDesign) -> None:
    rng = np.random.default_rng(3)
    frame, frame_of_item = synthetic(design, interaction=0.0, rng=rng)
    with pytest.raises(ValueError, match="stratum_axis"):
        permutation_plan(frame, design, frame_of_item=frame_of_item, stratum_axis="c")


# ------------------------------------------------------------------ calibration


def test_the_null_is_centred_on_zero_with_real_main_effects(design: PairDesign) -> None:
    """Main effects present and large. A null that shifted with them would be wrong."""
    rng = np.random.default_rng(4)
    frame, frame_of_item = synthetic(design, interaction=0.0, rng=rng)
    plan = permutation_plan(frame, design, frame_of_item=frame_of_item, stratum_axis="a")
    null = permute_interaction(
        frame, design, plan, scheme="within_stratum", n_permutations=400, rng=rng
    )
    assert abs(float(null.mean())) < 0.05


def test_pure_noise_does_not_fabricate_significance(design: PairDesign) -> None:
    """Over repeated DATASETS, not one seed: a correct null is extreme ~5% of the time."""
    small = 0
    trials = 24
    for seed in range(trials):
        rng = np.random.default_rng(500 + seed)
        frame, frame_of_item = synthetic(design, interaction=0.0, rng=rng)
        plan = permutation_plan(frame, design, frame_of_item=frame_of_item, stratum_axis="a")
        null = permute_interaction(
            frame, design, plan, scheme="within_stratum", n_permutations=199, rng=rng
        )
        observed = observed_interaction(frame, design)[0]
        if permutation_p(float(observed), null[:, 0]) <= 0.05:
            small += 1
    assert small <= 4, f"{small}/{trials} false positives at alpha 0.05 is too many"


def test_a_planted_interaction_beats_the_null(design: PairDesign) -> None:
    rng = np.random.default_rng(6)
    frame, frame_of_item = synthetic(design, interaction=4.0, rng=rng)
    plan = permutation_plan(frame, design, frame_of_item=frame_of_item, stratum_axis="a")
    null = permute_interaction(
        frame, design, plan, scheme="within_stratum", n_permutations=199, rng=rng
    )
    observed = observed_interaction(frame, design)[0]
    assert permutation_p(float(observed), null[:, 0]) == pytest.approx(1 / 200)


def test_freedman_lane_preserves_both_main_effects(design: PairDesign) -> None:
    """The calibration oracle: the default scheme's null must be no NARROWER than this one."""
    rng = np.random.default_rng(8)
    frame, frame_of_item = synthetic(design, interaction=0.0, rng=rng)
    plan = permutation_plan(frame, design, frame_of_item=frame_of_item, stratum_axis="a")
    default = permute_interaction(
        frame,
        design,
        plan,
        scheme="within_stratum",
        n_permutations=300,
        rng=np.random.default_rng(9),
    )
    oracle = permute_interaction(
        frame,
        design,
        plan,
        scheme="freedman_lane",
        n_permutations=300,
        rng=np.random.default_rng(9),
    )
    assert float(default[:, 0].std()) >= float(oracle[:, 0].std()) * 0.9


# ------------------------------------------------------------------ the p-value itself


def test_p_is_never_zero() -> None:
    """(1 + count)/(B + 1). A zero would claim more evidence than B replicates can supply."""
    assert permutation_p(1e9, np.zeros(199)) == pytest.approx(1 / 200)
    assert permutation_p(0.0, np.zeros(199)) == pytest.approx(1.0)


def test_p_is_deterministic_at_a_fixed_seed(design: PairDesign) -> None:
    rng = np.random.default_rng(10)
    frame, frame_of_item = synthetic(design, interaction=0.5, rng=rng)
    plan = permutation_plan(frame, design, frame_of_item=frame_of_item, stratum_axis="a")
    first = permute_interaction(
        frame,
        design,
        plan,
        scheme="within_stratum",
        n_permutations=100,
        rng=np.random.default_rng(3),
    )
    second = permute_interaction(
        frame,
        design,
        plan,
        scheme="within_stratum",
        n_permutations=100,
        rng=np.random.default_rng(3),
    )
    assert np.allclose(first, second)
    third = permute_interaction(
        frame,
        design,
        plan,
        scheme="within_stratum",
        n_permutations=100,
        rng=np.random.default_rng(4),
    )
    assert not np.allclose(first, third)
