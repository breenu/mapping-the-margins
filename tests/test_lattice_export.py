"""The cell inventory and value-provenance exports (P3).

These two files are gate P3's evidence, so the tests check both *what* they say (powered vs
descriptive marked before any run; every value's source present) and *that they are stable* —
a regenerated inventory must be byte-identical, or a diff cannot tell a real lattice change from
a line-ending difference between the two locked environments.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from margins.lattice import (
    CELL_INVENTORY_COLUMNS,
    CELL_INVENTORY_FILENAME,
    DEFAULT_LATTICE,
    VALUE_PROVENANCE_FILENAME,
    DeclaredLattice,
    export_lattice,
    load_lattice_config,
    render_cell_inventory,
    render_value_provenance,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "lattice"


@pytest.fixture(scope="module")
def tiny() -> DeclaredLattice:
    return DeclaredLattice(load_lattice_config("tiny", config_dir=FIXTURE_DIR))


@pytest.fixture(scope="module")
def shipped() -> DeclaredLattice:
    return DeclaredLattice(load_lattice_config(DEFAULT_LATTICE))


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text, newline="")))


# ---------------------------------------------------------------------------- cell inventory


def test_inventory_has_the_declared_columns_and_one_row_per_cell(tiny: DeclaredLattice) -> None:
    text = render_cell_inventory(tiny)
    header = text.splitlines()[0]
    assert header == ",".join(CELL_INVENTORY_COLUMNS)
    rows = _rows(text)
    assert len(rows) == len(tiny.records())


def test_inventory_marks_powered_and_confirmatory_before_any_run(tiny: DeclaredLattice) -> None:
    rows = {row["cell_id"]: row for row in _rows(render_cell_inventory(tiny))}
    for record in tiny.records():
        row = rows[record.cell_id]
        assert row["order"] == str(record.order)
        assert row["projected_n"] == str(record.projected_n)
        assert row["powered"] == ("true" if record.powered else "false")
        assert row["confirmatory"] == ("true" if record.confirmatory else "false")
        assert row["descriptor"] == record.descriptor


def test_inventory_records_the_axis_pair_only_for_order_2_cells(tiny: DeclaredLattice) -> None:
    for row in _rows(render_cell_inventory(tiny)):
        if row["order"] == "1":
            assert row["axis_pair"] == ""
        else:
            assert len(row["axis_pair"].split("|")) == 2


def test_inventory_contains_descriptive_only_cells(tiny: DeclaredLattice) -> None:
    """Underpowered cells appear, flagged -- they are not silently dropped (invariant 4)."""
    rows = _rows(render_cell_inventory(tiny))
    assert any(row["powered"] == "false" for row in rows)
    assert any(row["powered"] == "true" for row in rows)


def test_shipped_inventory_marks_every_confirmatory_pair(shipped: DeclaredLattice) -> None:
    rows = _rows(render_cell_inventory(shipped))
    pairs = {
        row["axis_pair"] for row in rows if row["order"] == "2" and row["confirmatory"] == "true"
    }
    assert pairs == {"|".join(pair) for pair in shipped.config.confirmatory_pairs}


# ---------------------------------------------------------------------------- determinism


def test_inventory_regenerates_byte_identically(tiny: DeclaredLattice, tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_inventory, first_provenance = export_lattice(tiny, first_dir)
    second_inventory, second_provenance = export_lattice(
        DeclaredLattice(load_lattice_config("tiny", config_dir=FIXTURE_DIR)), second_dir
    )
    assert first_inventory.read_bytes() == second_inventory.read_bytes()
    assert first_provenance.read_bytes() == second_provenance.read_bytes()


def test_export_writes_both_files_with_unix_line_endings(
    tiny: DeclaredLattice, tmp_path: Path
) -> None:
    inventory, provenance = export_lattice(tiny, tmp_path / "out")
    assert inventory.name == CELL_INVENTORY_FILENAME
    assert provenance.name == VALUE_PROVENANCE_FILENAME
    assert b"\r\n" not in inventory.read_bytes()
    assert b"\r\n" not in provenance.read_bytes()


def test_overwriting_an_existing_export_is_stable(tiny: DeclaredLattice, tmp_path: Path) -> None:
    inventory, _ = export_lattice(tiny, tmp_path)
    before = inventory.read_bytes()
    export_lattice(tiny, tmp_path)
    assert inventory.read_bytes() == before


# ---------------------------------------------------------------------------- value provenance


def test_provenance_table_lists_every_value_with_its_source(shipped: DeclaredLattice) -> None:
    text = render_value_provenance(shipped)
    for value in shipped.values():
        assert f"`{value.source_term}`" in text
        assert f"`{value.source_bucket}`" in text
        assert f"`{value.source_key}`" in text


def test_provenance_table_pins_the_source(shipped: DeclaredLattice) -> None:
    text = render_value_provenance(shipped)
    source = shipped.config.sources["holistic_bias_v1_1"]
    assert source.sha256 in text
    assert source.revision is not None
    assert source.revision in text
    assert source.license in text


def test_provenance_table_records_the_selection_rule_and_pairs(shipped: DeclaredLattice) -> None:
    text = render_value_provenance(shipped)
    assert "## Selection rule" in text
    assert shipped.config.description.strip().splitlines()[0] in text
    for pair in shipped.config.confirmatory_pairs:
        assert f"`{pair[0]} x {pair[1]}`" in text
    assert "confirmatory" in text
    assert "exploratory" in text


def test_provenance_table_carries_no_timestamp_of_its_own(shipped: DeclaredLattice) -> None:
    """Determinism: only the source's recorded retrieval time may appear, never 'now'."""
    first = render_value_provenance(shipped)
    second = render_value_provenance(shipped)
    assert first == second
