"""The identity lattice (P3): axes, values with provenance, cells, and power bookkeeping.

Produces :class:`~margins.schemas.IdentityCell` records plus a cell inventory and a value
provenance table. It renders no prompts, runs no models, and scores nothing — P4a's
``TemplateRenderer`` is the first stage that turns a cell into a
:class:`~margins.schemas.Prompt`.
"""

from __future__ import annotations

from margins.lattice.build import CellRecord, DeclaredLattice, make_cell_id
from margins.lattice.config import (
    DEFAULT_LATTICE,
    AxisPairSpec,
    AxisSpec,
    AxisValueSpec,
    DescriptorSource,
    LatticeConfig,
    LatticeValue,
    PowerSpec,
    lattice_config_from_mapping,
    load_lattice_config,
)
from margins.lattice.export import (
    CELL_INVENTORY_COLUMNS,
    CELL_INVENTORY_FILENAME,
    VALUE_PROVENANCE_FILENAME,
    export_lattice,
    render_cell_inventory,
    render_value_provenance,
    write_cell_inventory,
    write_value_provenance,
)

__all__ = [
    "CELL_INVENTORY_COLUMNS",
    "CELL_INVENTORY_FILENAME",
    "DEFAULT_LATTICE",
    "VALUE_PROVENANCE_FILENAME",
    "AxisPairSpec",
    "AxisSpec",
    "AxisValueSpec",
    "CellRecord",
    "DeclaredLattice",
    "DescriptorSource",
    "LatticeConfig",
    "LatticeValue",
    "PowerSpec",
    "export_lattice",
    "lattice_config_from_mapping",
    "load_lattice_config",
    "make_cell_id",
    "render_cell_inventory",
    "render_value_provenance",
    "write_cell_inventory",
    "write_value_provenance",
]
