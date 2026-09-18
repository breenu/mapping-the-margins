"""Draw the human-calibration sample for hand labeling (P6).

    uv run python scripts/build_calibration_sample.py --harm-type toxicity

Writes ``data/scoring/calibration_sample_<harm_type>.jsonl``: one line per response to label, with
the text, the cell, and what each scorer said about it. **Labeling is the human's work** -- this
script only decides WHICH responses are worth a person's attention, and
``scripts/calibration_report.py`` computes precision/recall once the labels come back.

The sample is deterministic at the roster's configured seed and **stratified** by cell order and by
scorer disagreement: a uniform draw from a mostly-benign corpus would spend nearly all of a limited
labeling budget confirming true negatives, and would be least informative exactly where
triangulation splits.

To label: add one JSON object per line to a separate file, e.g.

    {"response_id": "...", "harm_type": "toxicity", "label": 1, "labeler": "rb", "note": "slur"}
"""

from __future__ import annotations

import argparse
from pathlib import Path

from margins.inference.harness import read_responses
from margins.lattice.build import DeclaredLattice
from margins.lattice.config import DEFAULT_LATTICE, load_lattice_config
from margins.prompts.export import PROMPTS_FILENAME
from margins.scoring.calibration import sample_for_calibration, write_calibration_sample
from margins.scoring.config import DEFAULT_SCORING, default_scoring_config_dir, load_scoring_config
from margins.scoring.export import DEFAULT_SCORING_DIR, SCORES_FILENAME, read_scores

DEFAULT_PROMPTS = Path("data") / "prompts" / PROMPTS_FILENAME


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harm-type", required=True, help="One harm type per sample.")
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORING_DIR / SCORES_FILENAME)
    parser.add_argument(
        "--responses", type=Path, required=True, help="Response JSONL for the text."
    )
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--scoring", default=DEFAULT_SCORING)
    parser.add_argument("--lattice", default=DEFAULT_LATTICE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_SCORING_DIR)
    args = parser.parse_args(argv)

    config = load_scoring_config(args.scoring, config_dir=default_scoring_config_dir())
    if args.harm_type not in config.harm_types:
        parser.error(f"{args.harm_type!r} is not in roster {config.key!r}: {config.harm_types}")

    scores = read_scores(args.scores)
    responses = read_responses(args.responses)
    texts = {r.response_id: r.text for r in responses}

    # cell_id -> order, resolved from the lattice rather than parsed out of the id string.
    lattice = DeclaredLattice(load_lattice_config(args.lattice))
    cell_order = {cell.cell_id: cell.order for cell in lattice.cells()}

    units = sample_for_calibration(
        scores, texts, config, harm_type=args.harm_type, cell_order=cell_order
    )
    path = write_calibration_sample(
        units, args.out_dir / f"calibration_sample_{args.harm_type}.jsonl"
    )

    print(f"harm type  {args.harm_type}")
    print(f"sampled    {len(units)} of {len({s.response_id for s in scores})} responses")
    print(f"seed       {config.calibration.seed}")
    print("\nstrata")
    for order in sorted({u.cell_order for u in units}):
        for bucket in ("low", "mid", "high"):
            count = sum(
                1 for u in units if u.cell_order == order and u.disagreement_bucket == bucket
            )
            if count:
                print(f"  order-{order} disagreement={bucket:<5} {count}")
    print(f"\nwrote {path}")
    print("Label into a separate JSONL, then run scripts/calibration_report.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
