"""Turn a labelling sample into a spreadsheet, and the filled spreadsheet back into label JSONL.

    uv run python scripts/label_sheets.py export-calibration --harm-type toxicity
    uv run python scripts/label_sheets.py import-calibration --harm-type toxicity --labeler rb
    uv run python scripts/label_sheets.py export-on-target
    uv run python scripts/label_sheets.py import-on-target --labeler rb

**Why a spreadsheet.** The labels are a person's work, done in Excel or Google Sheets, not in a
JSONL editor. ``export-*`` writes a CSV with one row per item and an empty label column; the human
fills it in; ``import-*`` validates every row against the real label record
(:class:`~margins.scoring.calibration.HumanLabel` /
:class:`~margins.redteam.on_target.OnTargetLabel`) and writes the JSONL the report scripts read.
What to mark is defined in ``docs/labelling_guide.md``.

**Blind on purpose.** The calibration sheet shows the TEXT and nothing else: not the scorers'
outputs, not the cell. A labeller who can see that a classifier said 0.9 is no longer an
independent check on that classifier.

**Strict on import.** A blank or non-binary label is refused rather than skipped: a calibration
estimated on the rows somebody happened to finish is conditioned on something nobody recorded.
The CSV is written with a UTF-8 byte-order mark so Excel opens it as UTF-8 (identity terms and
curly quotes survive), and is read back the same way.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

from margins.redteam.on_target import OnTargetLabel
from margins.scoring.calibration import HumanLabel

SHEETS_DIR = Path("data") / "labels"
CALIBRATION_SAMPLE_DIR = Path("data") / "scoring" / "e1"
ON_TARGET_SAMPLE = Path("data") / "redteam" / "on_target_sample.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise SystemExit(f"STOP: {path} does not exist; build the sample first")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"STOP: {path} does not exist")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _binary(value: str, *, where: str, column: str) -> int:
    cleaned = value.strip()
    if cleaned not in ("0", "1"):
        raise SystemExit(
            f"STOP: {where} has {column}={value!r}. Every row needs 0 or 1; nothing is imported "
            "until the sheet is complete (a partial sample is a different sample)."
        )
    return int(cleaned)


def _write_jsonl(path: Path, records: Sequence[HumanLabel | OnTargetLabel]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(record.model_dump_json(exclude_none=True) + "\n")
    return path


def export_calibration(harm_type: str, sample_dir: Path, out_dir: Path) -> Path:
    units = _read_jsonl(sample_dir / f"calibration_sample_{harm_type}.jsonl")
    rows = [[u["response_id"], harm_type, u["text"], "", ""] for u in units]
    header = ["response_id", "harm_type", "text", "label", "note"]
    return _write_csv(out_dir / f"calibration_{harm_type}.csv", header, rows)


def import_calibration(harm_type: str, labeler: str, sheets: Path, out_dir: Path) -> Path:
    labels: list[HumanLabel] = []
    for number, row in enumerate(_read_csv(sheets / f"calibration_{harm_type}.csv"), start=2):
        if row["harm_type"] != harm_type:
            raise SystemExit(f"STOP: row {number} is {row['harm_type']!r}, not {harm_type!r}")
        labels.append(
            HumanLabel(
                response_id=row["response_id"],
                harm_type=harm_type,
                label=_binary(row["label"], where=f"row {number}", column="label"),
                labeler=labeler,
                note=row["note"].strip() or None,
            )
        )
    return _write_jsonl(out_dir / f"labels_{harm_type}.jsonl", labels)


def export_on_target(sample: Path, out_dir: Path) -> Path:
    units = _read_jsonl(sample)
    rows = [
        [
            u["frame_id"],
            u["target_pair"],
            u["target_cell_id"],
            u["text"],
            u["rendered_example"],
            "",
            "",
        ]
        for u in units
    ]
    header = [
        "frame_id",
        "target_pair",
        "target_cell",
        "frame",
        "rendered_for_target",
        "on_target",
        "note",
    ]
    return _write_csv(out_dir / "on_target.csv", header, rows)


def import_on_target(labeler: str, sheets: Path, out_dir: Path) -> Path:
    labels = [
        OnTargetLabel(
            frame_id=row["frame_id"],
            on_target=_binary(row["on_target"], where=f"row {number}", column="on_target"),
            labeler=labeler,
            note=row["note"].strip() or None,
        )
        for number, row in enumerate(_read_csv(sheets / "on_target.csv"), start=2)
    ]
    return _write_jsonl(out_dir / "on_target_labels.jsonl", labels)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("export-calibration", "import-calibration"):
        child = sub.add_parser(name)
        child.add_argument("--harm-type", required=True, choices=("toxicity", "regard"))
        child.add_argument("--sample-dir", type=Path, default=CALIBRATION_SAMPLE_DIR)
        child.add_argument("--sheets", type=Path, default=SHEETS_DIR)
        if name.startswith("import"):
            child.add_argument(
                "--labeler", required=True, help="Your initials, recorded per label."
            )
    for name in ("export-on-target", "import-on-target"):
        child = sub.add_parser(name)
        child.add_argument("--sample", type=Path, default=ON_TARGET_SAMPLE)
        child.add_argument("--sheets", type=Path, default=SHEETS_DIR)
        if name.startswith("import"):
            child.add_argument(
                "--labeler", required=True, help="Your initials, recorded per label."
            )
    args = parser.parse_args(argv)

    if args.command == "export-calibration":
        path = export_calibration(args.harm_type, args.sample_dir, args.sheets)
    elif args.command == "import-calibration":
        path = import_calibration(args.harm_type, args.labeler, args.sheets, args.sample_dir)
    elif args.command == "export-on-target":
        path = export_on_target(args.sample, args.sheets)
    else:
        path = import_on_target(args.labeler, args.sheets, args.sample.parent)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
