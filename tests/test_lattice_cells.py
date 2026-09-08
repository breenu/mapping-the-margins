"""Cell generation, the anti-cartesian guarantee, determinism, and power flags (P3).

The load-bearing test in this file is
:func:`test_no_order_2_cell_exists_for_an_undeclared_axis_pair`. Everything else checks that the
lattice does what it says; that one checks it does *not* do the thing that would quietly destroy
the project's power budget and inflate the FDR family.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pytest

from margins.interfaces import Lattice
from margins.lattice import DEFAULT_LATTICE, DeclaredLattice, LatticeConfig, load_lattice_config
from margins.schemas import IdentityCell

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "lattice"

# Hand-computed from tests/fixtures/lattice/tiny.yaml.
#   order-1: alpha 2 + beta 3 + gamma 2 + gender_x 2                     =  9
#   order-2: alpha x beta 2*3=6, alpha x gamma 2*2=4,
#            alpha x gender_x 2*2=4, beta x gamma 3*2=6                  = 20
EXPECTED_ORDER_1 = 9
EXPECTED_ORDER_2 = 20
EXPECTED_TOTAL = 29

# Hand-computed from configs/lattice/confirmatory_v1.yaml.
#   order-1: age 3 + class 4 + ability 4 + gender 3 + race 4             =  18
#   order-2: class x race 16, age x ability 12, ability x race 16,
#            age x class 12, gender x race 12, ability x gender 12,
#            age x race (exploratory) 12                                 =  92
SHIPPED_ORDER_1 = 18
SHIPPED_ORDER_2 = 92


@pytest.fixture(scope="module")
def tiny_config() -> LatticeConfig:
    return load_lattice_config("tiny", config_dir=FIXTURE_DIR)


@pytest.fixture(scope="module")
def tiny(tiny_config: LatticeConfig) -> DeclaredLattice:
    return DeclaredLattice(tiny_config)


@pytest.fixture(scope="module")
def shipped() -> DeclaredLattice:
    return DeclaredLattice(load_lattice_config(DEFAULT_LATTICE))


# ---------------------------------------------------------------------------- cardinality


def test_cell_count_matches_the_hand_computed_fixture(tiny: DeclaredLattice) -> None:
    cells = tiny.cells()
    assert len(cells) == EXPECTED_TOTAL
    assert sum(1 for cell in cells if cell.order == 1) == EXPECTED_ORDER_1
    assert sum(1 for cell in cells if cell.order == 2) == EXPECTED_ORDER_2


def test_shipped_cell_count_matches_the_hand_computed_config(shipped: DeclaredLattice) -> None:
    summary = shipped.summary()
    assert summary["cells_order_1"] == SHIPPED_ORDER_1
    assert summary["cells_order_2"] == SHIPPED_ORDER_2
    assert summary["cells"] == SHIPPED_ORDER_1 + SHIPPED_ORDER_2


def test_order_1_cells_are_exactly_one_per_declared_value(tiny: DeclaredLattice) -> None:
    order_1 = [cell for cell in tiny.cells() if cell.order == 1]
    assert len(order_1) == len(tiny.values())
    assert {cell.axes[0] for cell in order_1} == {
        (value.axis, value.value) for value in tiny.values()
    }
    for cell in order_1:
        assert len(cell.axes) == 1


def test_order_2_cells_are_the_full_product_within_each_declared_pair(
    tiny: DeclaredLattice, tiny_config: LatticeConfig
) -> None:
    for pair in tiny_config.pairs:
        axis_a, axis_b = tiny_config.canonical_pair(pair.axes)
        expected = {
            ((axis_a, a.value), (axis_b, b.value))
            for a in tiny.axis_values(axis_a)
            for b in tiny.axis_values(axis_b)
        }
        actual = {
            cell.axes
            for cell in tiny.cells()
            if cell.order == 2 and (cell.axes[0][0], cell.axes[1][0]) == (axis_a, axis_b)
        }
        assert actual == expected, f"{axis_a} x {axis_b}"


# ---------------------------------------------------------------------------- anti-cartesian


def test_no_order_2_cell_exists_for_an_undeclared_axis_pair(
    tiny: DeclaredLattice, tiny_config: LatticeConfig
) -> None:
    """THE anti-cartesian guarantee: the lattice is not the product of every axis with every other.

    The fixture leaves ``beta x gender_x`` and ``gamma x gender_x`` undeclared precisely so this
    test has something to fail on.
    """
    declared = set(tiny_config.confirmatory_pairs) | set(tiny_config.exploratory_pairs)
    all_pairs = {
        tiny_config.canonical_pair(pair) for pair in combinations(tiny_config.ordered_axes, 2)
    }
    undeclared = all_pairs - declared
    assert undeclared, "the fixture must leave some axis pair undeclared for this test to bite"

    emitted = {(cell.axes[0][0], cell.axes[1][0]) for cell in tiny.cells() if cell.order == 2}
    assert emitted == declared
    assert not emitted & undeclared


def test_shipped_lattice_is_not_the_full_cartesian_product(shipped: DeclaredLattice) -> None:
    config = shipped.config
    all_pairs = {config.canonical_pair(pair) for pair in combinations(config.ordered_axes, 2)}
    declared = set(config.confirmatory_pairs) | set(config.exploratory_pairs)
    assert declared < all_pairs, "the lattice must not cross every axis with every other axis"


def test_intersection_depth_never_exceeds_two(shipped: DeclaredLattice) -> None:
    """Invariant 6. Enforced by IdentityCell too; asserted here on real output."""
    assert {cell.order for cell in shipped.cells()} == {1, 2}
    assert all(len(cell.axes) == cell.order for cell in shipped.cells())


# ---------------------------------------------------------------------------- validity


def test_every_cell_resolves_in_the_axis_value_registry(shipped: DeclaredLattice) -> None:
    """Gate P3's second clause."""
    registry = shipped.registry
    assert len(registry) == len(shipped.values())
    for cell in shipped.cells():
        registry.validate_cell(cell)
        for pair in cell.axes:
            assert pair in registry


def test_cells_are_hashable_and_usable_as_keys(shipped: DeclaredLattice) -> None:
    """IdentityCell is a grouping key throughout the metric layer (invariant 1)."""
    cells = shipped.cells()
    assert len(set(cells)) == len(cells)
    lookup = {cell: cell.cell_id for cell in cells}
    assert lookup[cells[0]] == cells[0].cell_id


def test_cell_ids_are_unique_and_canonically_ordered(shipped: DeclaredLattice) -> None:
    config = shipped.config
    ids = [cell.cell_id for cell in shipped.cells()]
    assert len(set(ids)) == len(ids)
    for cell in shipped.cells():
        ranks = [config.axis_rank(axis) for axis, _ in cell.axes]
        assert ranks == sorted(ranks)
        assert cell.cell_id == "|".join(f"{axis}={value}" for axis, value in cell.axes)


def test_the_lattice_satisfies_the_lattice_protocol(shipped: DeclaredLattice) -> None:
    assert isinstance(shipped, Lattice)
    assert all(isinstance(cell, IdentityCell) for cell in shipped.cells())


# ---------------------------------------------------------------------------- determinism


def test_same_config_yields_identical_cells(tiny_config: LatticeConfig) -> None:
    first = DeclaredLattice(tiny_config).cells()
    second = DeclaredLattice(tiny_config).cells()
    assert first == second
    assert [cell.cell_id for cell in first] == [cell.cell_id for cell in second]


def test_reloading_the_config_from_disk_yields_identical_cells() -> None:
    first = DeclaredLattice(load_lattice_config("tiny", config_dir=FIXTURE_DIR))
    second = DeclaredLattice(load_lattice_config("tiny", config_dir=FIXTURE_DIR))
    assert first.cells() == second.cells()
    assert first.records() == second.records()


# ---------------------------------------------------------------------------- power flags


def test_powered_flag_is_projected_n_against_min_cell_n(tiny: DeclaredLattice) -> None:
    min_n = tiny.config.power.min_cell_n
    for record in tiny.records():
        assert record.powered == (record.projected_n >= min_n)


def test_the_overridden_pair_is_underpowered_and_the_rest_are_not(tiny: DeclaredLattice) -> None:
    """beta x gamma carries a below-threshold override, so it is descriptive-only (invariant 4)."""
    underpowered = {record.cell_id for record in tiny.records() if not record.powered}
    expected = {
        record.cell_id for record in tiny.records() if record.axis_pair == ("beta", "gamma")
    }
    assert underpowered == expected
    assert len(underpowered) == 6
    assert all(record.projected_n == 4 for record in tiny.records() if not record.powered)


def test_powered_cells_exclude_the_descriptive_ones(tiny: DeclaredLattice) -> None:
    powered = tiny.powered_cells()
    assert len(powered) == EXPECTED_TOTAL - 6
    assert all(
        cell.order == 1 or cell.axes[0][0] != "beta" or cell.axes[1][0] != "gamma"
        for cell in powered
    )


def test_shipped_non_gender_confirmatory_pairs_project_as_powered(shipped: DeclaredLattice) -> None:
    """Go/no-go criterion 5 (PROJECT_CONTEXT.md section 13), checked at the config stage."""
    config = shipped.config
    non_gender = config.non_gender_confirmatory_pairs
    assert len(non_gender) >= 2
    for pair in non_gender:
        records = [record for record in shipped.records() if record.axis_pair == pair]
        assert records
        assert all(record.powered for record in records), pair


def test_confirmatory_flags_track_pair_status(tiny: DeclaredLattice) -> None:
    config = tiny.config
    for record in tiny.records():
        if record.order == 2:
            assert record.axis_pair is not None
            assert record.confirmatory == (record.axis_pair in config.confirmatory_pairs)
        else:
            axis = record.cell.axes[0][0]
            assert record.confirmatory == (axis in config.confirmatory_axes)


def test_the_exploratory_pair_is_not_marked_confirmatory(tiny: DeclaredLattice) -> None:
    exploratory = [record for record in tiny.records() if record.axis_pair == ("alpha", "gamma")]
    assert len(exploratory) == 4
    assert not any(record.confirmatory for record in exploratory)


# ---------------------------------------------------------------------------- composition


def test_descriptor_composition_orders_by_compose_rank(tiny: DeclaredLattice) -> None:
    by_id = {record.cell_id: record for record in tiny.records()}
    assert by_id["alpha=a1"].descriptor == "a1"
    assert by_id["alpha=a1|beta=b1"].descriptor == "a1 b1"
    # gamma's g1 declares a surface different from its value, so composition must use the surface.
    assert by_id["beta=b1|gamma=g1"].descriptor == "b1 gee-one"
    assert by_id["alpha=a2|gender_x=x2"].descriptor == "a2 x2"


def test_compose_descriptor_matches_the_recorded_descriptor(shipped: DeclaredLattice) -> None:
    for record in shipped.records():
        assert shipped.compose_descriptor(record.cell) == record.descriptor


def test_shipped_composition_reads_as_expected(shipped: DeclaredLattice) -> None:
    """The rank was chosen so every declared pair composes into a fluent phrase."""
    by_id = {record.cell_id: record for record in shipped.records()}
    assert by_id["socioeconomic_class=working-class|race_ethnicity=Black"].descriptor == (
        "working-class Black"
    )
    assert by_id["age=older|ability=blind"].descriptor == "older blind"
    assert by_id["ability=blind|gender_and_sex=female"].descriptor == "blind female"
    assert by_id["age=older|socioeconomic_class=working-class"].descriptor == (
        "older working-class"
    )


def test_compose_descriptor_rejects_a_cell_this_lattice_does_not_declare(
    shipped: DeclaredLattice,
) -> None:
    alien = IdentityCell(cell_id="age=ancient", axes=(("age", "ancient"),), order=1)
    with pytest.raises(KeyError):
        shipped.compose_descriptor(alien)


# ---------------------------------------------------------------------------- summary


def test_summary_is_internally_consistent(shipped: DeclaredLattice) -> None:
    summary = shipped.summary()
    records = shipped.records()
    assert summary["cells"] == len(records)
    assert summary["powered_cells"] + summary["descriptive_cells"] == summary["cells"]
    assert summary["confirmatory_pairs"] + summary["exploratory_pairs"] == summary["declared_pairs"]
    assert summary["non_gender_confirmatory_pairs"] >= 2
    assert summary["projected_renderings"] == sum(record.projected_n for record in records)
