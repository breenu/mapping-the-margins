"""Build the identity lattice and write its two exports (P3).

Regenerates ``data/lattice/cell_inventory.csv`` and ``data/lattice/value_provenance.md`` from a
lattice config. Both outputs are byte-deterministic, so re-running this on an unchanged config
produces an unchanged file.

``--verify`` additionally checks every declared value against the pinned HolisticBias file
(fetch it first with ``scripts/fetch_holistic_bias.py``). Run it before tagging the
pre-registration: it is the mechanism behind "never invent a descriptor".

The printed ``config_sha256`` is what a later phase passes to
``margins.config.write_run_manifest(inputs=...)`` so a run's manifest records which lattice it
audited.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from margins.lattice import DEFAULT_LATTICE, DeclaredLattice, export_lattice, load_lattice_config
from margins.lattice.config import default_lattice_config_dir
from margins.lattice.descriptors import (
    HOLISTIC_BIAS_V1_1,
    load_descriptor_index,
    verify_lattice_values,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lattice", default=DEFAULT_LATTICE, help="Lattice config name.")
    parser.add_argument("--config-dir", type=Path, default=None, help="Lattice config directory.")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/lattice"), help="Where to write the exports."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check every value against the pinned HolisticBias descriptors file.",
    )
    args = parser.parse_args(argv)

    config_dir = args.config_dir or default_lattice_config_dir()
    config_file = config_dir / f"{args.lattice}.yaml"
    config = load_lattice_config(args.lattice, config_dir=config_dir)
    lattice = DeclaredLattice(config)

    if args.verify:
        index = load_descriptor_index()
        for source_key, source in config.sources.items():
            if source.kind != "descriptor_taxonomy":
                continue
            verify_lattice_values(lattice.values(), index, source_key=source_key)
        print(
            f"verified every value against {HOLISTIC_BIAS_V1_1.repo_path} "
            f"@ {HOLISTIC_BIAS_V1_1.revision}"
        )

    inventory, provenance = export_lattice(lattice, args.out_dir)

    print(f"lattice_version: {config.lattice_version}")
    if config_file.is_file():
        digest = hashlib.sha256(config_file.read_bytes()).hexdigest()
        print(f"config_sha256:   {digest}  ({config_file})")

    print("\nsummary")
    for key, value in lattice.summary().items():
        print(f"  {key:<32} {value}")

    print("\nconfirmatory axis pairs (THE FDR FAMILY)")
    for pair in config.confirmatory_pairs:
        marker = "non-gender x non-gender" if config.is_non_gender_pair(pair) else "gender-anchored"
        print(f"  {pair[0]} x {pair[1]:<24} {marker}")
    if config.exploratory_pairs:
        print("\nexploratory pairs (descriptive only, NO significance claim)")
        for pair in config.exploratory_pairs:
            print(f"  {pair[0]} x {pair[1]}")

    print(
        f"\nmin_cell_n: {config.power.min_cell_n}  "
        "(PRE-REGISTERED PLACEHOLDER -- TODO(P9): derive from the real power analysis)"
    )
    print(f"\nwrote {inventory}")
    print(f"wrote {provenance}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
