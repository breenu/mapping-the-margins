"""The labelling round trip: sample -> spreadsheet -> human -> label JSONL the reports accept.

The spreadsheet is blind (text only) and the import is strict (every row 0/1, or nothing).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import scripts.label_sheets as sheets
from margins.redteam.on_target import OnTargetLabel
from margins.scoring.calibration import HumanLabel


def _sample(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _fill(path: Path, column: str, values: list[str]) -> None:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row, value in zip(rows, values, strict=True):
        row[column] = value
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_the_calibration_sheet_is_blind_and_round_trips(tmp_path: Path) -> None:
    units = [
        {
            "response_id": f"r{i}",
            "harm_type": "regard",
            "text": f"Text {i} — with “curly” quotes, a Deaf person",
            "cell_id": "c",
            "cell_order": 2,
            "disagreement_bucket": "high",
            "scorer_raw": {"regard_classifier": 0.9},
        }
        for i in range(3)
    ]
    _sample(tmp_path / "calibration_sample_regard.jsonl", units)
    sheet = sheets.export_calibration("regard", tmp_path, tmp_path / "labels")
    header = sheet.read_text(encoding="utf-8-sig").splitlines()[0]
    assert header == "response_id,harm_type,text,label,note"  # no scorer output, no cell
    assert "0.9" not in sheet.read_text(encoding="utf-8-sig")

    _fill(sheet, "label", ["1", "0", " 1 "])
    out = sheets.import_calibration("regard", "rb", tmp_path / "labels", tmp_path)
    labels = [HumanLabel.model_validate_json(line) for line in out.read_text().splitlines()]
    assert [(lab.response_id, lab.label, lab.labeler) for lab in labels] == [
        ("r0", 1, "rb"),
        ("r1", 0, "rb"),
        ("r2", 1, "rb"),
    ]


@pytest.mark.parametrize("bad", ["", "2", "yes"])
def test_an_incomplete_or_non_binary_sheet_imports_nothing(tmp_path: Path, bad: str) -> None:
    _sample(
        tmp_path / "calibration_sample_toxicity.jsonl",
        [{"response_id": f"r{i}", "text": "t"} for i in range(2)],
    )
    sheet = sheets.export_calibration("toxicity", tmp_path, tmp_path / "labels")
    _fill(sheet, "label", ["1", bad])
    with pytest.raises(SystemExit, match="0 or 1"):
        sheets.import_calibration("toxicity", "rb", tmp_path / "labels", tmp_path)
    assert not (tmp_path / "labels_toxicity.jsonl").exists()


def test_the_on_target_sheet_round_trips(tmp_path: Path) -> None:
    sample = tmp_path / "redteam" / "on_target_sample.jsonl"
    _sample(
        sample,
        [
            {
                "frame_id": "f1",
                "target_pair": "ability x race_ethnicity",
                "target_cell_id": "ability=blind|race_ethnicity=Asian",
                "text": "Consider [PERSON] applying for a loan.",
                "rendered_example": "Consider a blind Asian person applying for a loan.",
            }
        ],
    )
    sheet = sheets.export_on_target(sample, tmp_path / "labels")
    _fill(sheet, "on_target", ["1"])
    out = sheets.import_on_target("rb", tmp_path / "labels", sample.parent)
    label = OnTargetLabel.model_validate_json(out.read_text().splitlines()[0])
    assert (label.frame_id, label.on_target, label.labeler) == ("f1", 1, "rb")
