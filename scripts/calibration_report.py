"""Score the scorers against the human labels (P6).

    uv run python scripts/calibration_report.py --labels data/scoring/labels_toxicity.jsonl

Reads hand-filled labels, computes each scorer's precision / recall / F1 at its declared threshold,
and writes ``data/scoring/calibration_report.md``.

**What this answers, and what it does not.** Calibration asks whether a scorer finds what a person
calls harm. It says nothing about whether the scorer responds to identity terms independently of
harm -- that is the drift control -- nor about whether two scorers behave alike, which is the
agreement table. A harm type needs all three before it is load-bearing, and the generated report
says so on its face rather than leaving a reader to remember it.

The sample was deliberately drawn to over-weight scorer disagreement and order-2 cells, so precision
measured on it is **not** an estimate of precision on the corpus. The report states that too.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from margins.scoring.calibration import (
    CALIBRATION_REPORT_FILENAME,
    calibration_metrics,
    read_labels,
    render_calibration_report,
)
from margins.scoring.config import DEFAULT_SCORING, default_scoring_config_dir, load_scoring_config
from margins.scoring.export import DEFAULT_SCORING_DIR, SCORES_FILENAME, read_scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True, help="Hand-filled label JSONL.")
    parser.add_argument(
        "--scores", type=Path, nargs="+", default=[DEFAULT_SCORING_DIR / SCORES_FILENAME]
    )
    parser.add_argument("--scoring", default=DEFAULT_SCORING)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_SCORING_DIR)
    args = parser.parse_args(argv)

    config = load_scoring_config(args.scoring, config_dir=default_scoring_config_dir())
    labels = read_labels(args.labels)
    scores = [row for path in args.scores for row in read_scores(path)]

    metrics = calibration_metrics(scores, labels, config)

    print(f"labels     {len(labels)}")
    print(f"scores     {len(scores)}\n")
    for metric in metrics:
        precision = "—" if metric.precision is None else f"{metric.precision:.3f}"
        recall = "—" if metric.recall is None else f"{metric.recall:.3f}"
        print(
            f"  {metric.harm_type:<10} {metric.scorer_id:<20} "
            f"n={metric.n:<5} precision={precision:<7} recall={recall}"
        )

    text = render_calibration_report(
        metrics, scoring_version=config.scoring_version, n_labels=len(labels)
    )
    path = args.out_dir / CALIBRATION_REPORT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)

    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
