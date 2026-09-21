"""Regenerate every figure from the committed result tables. One command (P10).

    uv run python scripts/make_figures.py

Writes vector PDF **and** SVG for each figure into ``figures/``. There is no raster path:
``.gitignore`` drops ``figures/*.png`` and :func:`margins.viz.style.save_figure` refuses a raster
format outright, so a rasterised headline figure cannot enter the repo by accident.

**Every number on every canvas is audited as it is built.** :func:`margins.viz.make.build_figure`
renders the figure, then walks its artists and fails if anything drawn traces to neither
``results/paper_numbers.json`` nor a committed result table nor a declared structural constant. A
figure that would have printed a hand-typed number does not get written -- the command stops.

**Which tables it draws from.** ``results/tables/`` when a real P7+P9 run has produced them,
otherwise ``results/fixtures/``. The fallback is stated on the console and stamped on every
affected figure, because before P11 several of these figures necessarily show synthetic rows and
a reader must not have to know that from a commit message.

Runs on the core environment: matplotlib and numpy only, no torch, no GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from margins.report.tables import DEFAULT_FIXTURE_DIR, DEFAULT_TABLE_DIR
from margins.stats.numbers import PAPER_NUMBERS_PATH
from margins.viz.make import ESSENTIAL, FIGURES, SUPPORTING, build_all
from margins.viz.style import DEFAULT_FIGURE_DIR, VECTOR_FORMATS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=None,
        help="Override the result tables. Default: results/tables if built, else results/fixtures.",
    )
    parser.add_argument("--paper-numbers", type=Path, default=PAPER_NUMBERS_PATH)
    parser.add_argument(
        "--only",
        choices=("all", "essential", "supporting"),
        default="all",
        help="Which figures to build. 'essential' is the five that carry a headline claim.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=list(VECTOR_FORMATS),
        help="Vector formats only; a raster format is refused.",
    )
    args = parser.parse_args(argv)

    recipes = {"all": FIGURES, "essential": ESSENTIAL, "supporting": SUPPORTING}[args.only]
    built = build_all(
        out_dir=args.out_dir,
        table_dir=args.table_dir,
        paper_numbers_path=args.paper_numbers,
        recipes=recipes,
        formats=tuple(args.formats),
    )

    source = args.table_dir
    if source is None:
        source = DEFAULT_TABLE_DIR if any(DEFAULT_TABLE_DIR.glob("*.csv")) else DEFAULT_FIXTURE_DIR
    print(f"tables     {source}")
    print(f"numbers    {args.paper_numbers}")
    print(f"out        {args.out_dir}\n")

    for spec, paths in built:
        marker = "REAL   " if spec.data_status == "real" else "FIXTURE"
        print(f"  [{marker}] {spec.name:<26} {len(spec.ledger):>4} traced values")
        print(f"            claim: {spec.claim.splitlines()[0][:96]}")
        for fmt in sorted(paths):
            print(f"            {paths[fmt]}")

    fixture = [spec for spec, _ in built if spec.data_status == "fixture"]
    print(f"\n{len(built)} figures, every drawn number traced to a committed artifact.")
    if fixture:
        print(
            f"{len(fixture)} of them show FIXTURE rows and are banner-stamped: "
            f"{', '.join(spec.name for spec in fixture)}.\n"
            "The P11 confirmatory run replaces the tables under them; no figure code changes."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
