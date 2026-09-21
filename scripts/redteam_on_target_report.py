"""Compute the on-target % from the human labels and write the report (P8).

Gate P8 requires "human on-target % on a sample reported". ``scripts/build_redteam_frames.py``
writes the unlabeled sample; **the labeling is the human's work**; this turns the returned labels
into the reported number.

The labeler answers one question per frame: *would this frame plausibly expose disparate treatment
of its named target?* Not *is it harmful* — a frame naming an identity term was rejected before it
reached the sample, so every frame here is neutral text with one injected slot.

No model is loaded and nothing is auto-labeled. Asking a model whether a model-generated prompt is
on-target would be a scorer judging a scorer, which is what CLAUDE.md's guardrails forbid and what
would make the number worthless.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from margins.redteam import (
    DEFAULT_REDTEAM_SET,
    ON_TARGET_REPORT_FILENAME,
    ON_TARGET_SAMPLE_FILENAME,
    default_redteam_config_dir,
    load_redteam_config,
    on_target_metrics,
    read_on_target_labels,
    render_on_target_report,
)
from margins.redteam.on_target import OnTargetUnit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Hand-filled JSONL: one {frame_id, on_target, labeler} record per line.",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=Path("data/redteam") / ON_TARGET_SAMPLE_FILENAME,
        help="The sample the labels belong to. Labels outside it are refused.",
    )
    parser.add_argument("--redteam", default=DEFAULT_REDTEAM_SET)
    parser.add_argument("--redteam-config-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("data/redteam"))
    args = parser.parse_args(argv)

    config = load_redteam_config(
        args.redteam, config_dir=args.redteam_config_dir or default_redteam_config_dir()
    )

    if not args.sample.is_file():
        raise SystemExit(
            f"sample not found: {args.sample}. Run scripts/build_redteam_frames.py first."
        )
    units = [
        OnTargetUnit.model_validate_json(line)
        for line in args.sample.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels = read_on_target_labels(args.labels, units=units)
    metrics = on_target_metrics(units, labels)

    report = render_on_target_report(
        metrics, redteam_version=config.redteam_version, n_labels=len(labels)
    )
    out_path = args.out_dir / ON_TARGET_REPORT_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(report)

    print(f"redteam_version:  {config.redteam_version}")
    print(f"sampled:          {len(units)}")
    print(f"labeled:          {len(labels)}")
    print("\non-target rate")
    for metric in metrics:
        rate = "—" if metric.on_target_rate is None else f"{metric.on_target_rate:.1%}"
        print(f"  {metric.stratum:<36} {metric.n_on_target}/{metric.n_labeled}  {rate}")
    print(f"\nwrote {out_path}")

    unlabeled = len(units) - len(labels)
    if unlabeled:
        print(
            f"\nNOTE: {unlabeled} sampled frame(s) carry no label. They count toward no "
            "denominator -- an unlabeled frame is not a miss."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
