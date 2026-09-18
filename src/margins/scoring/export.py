"""Scoring artifacts: scores, the drift control, and the agreement table (P6).

**This is the project's first Parquet writer, so it sets the convention -- including its limits.**
PROJECT_CONTEXT.md section 6 says "raw text -> JSONL, tabular -> Parquet", and `ItemScore` is
tabular. But Parquet is **not byte-deterministic**: the file embeds the writer's version and
per-column metadata, so two runs of identical data on different `pyarrow` builds produce different
bytes. The lattice and prompt exports assert byte identity; **these cannot**, and pretending
otherwise would be a reproducibility claim we have not demonstrated (CLAUDE.md guardrails).

What is asserted instead, and what the tests check: **round-trip content equality on a sorted
frame**. Rows are written in a canonical sort order, so reading a file back and comparing values
reproduces the table exactly. That is the property P7 depends on; byte identity is not.

The agreement table is small and human-facing, so it is written as **CSV plus Markdown** following
the existing `margins.lattice.export` conventions -- and those two *are* byte-deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from margins.schemas import ItemScore
from margins.scoring.drift import DriftRow
from margins.scoring.standardize import StandardizationReport
from margins.scoring.triangulate import AgreementRecord

__all__ = [
    "AGREEMENT_CSV_FILENAME",
    "AGREEMENT_MD_FILENAME",
    "DEFAULT_SCORING_DIR",
    "DRIFT_FILENAME",
    "SCORES_FILENAME",
    "export_scoring",
    "read_drift_rows",
    "read_scores",
    "render_agreement_csv",
    "render_agreement_markdown",
    "write_agreement",
    "write_drift_rows",
    "write_scores",
]

DEFAULT_SCORING_DIR: Final[Path] = Path("data") / "scoring"

SCORES_FILENAME: Final[str] = "scores.parquet"
DRIFT_FILENAME: Final[str] = "drift_control.parquet"
AGREEMENT_CSV_FILENAME: Final[str] = "agreement.csv"
AGREEMENT_MD_FILENAME: Final[str] = "agreement.md"

#: Canonical row order for the score table. Deterministic and total: `score_id` is unique, so the
#: sort never depends on input order or on a tie-break the reader cannot reproduce.
_SCORE_SORT: Final[tuple[str, ...]] = (
    "benchmark",
    "harm_type",
    "scorer_id",
    "condition",
    "score_id",
)

#: Canonical row order for the drift table.
_DRIFT_SORT: Final[tuple[str, ...]] = (
    "harm_type",
    "scorer_id",
    "control_family",
    "cell_id",
    "score_id",
)


def _write_text(path: Path, text: str) -> Path:
    """Write text with explicit LF endings, as the lattice and prompt exports do."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" keeps the explicit \n from being translated to \r\n on Windows, which is what
    # makes the byte-identity check meaningful across win_amd64 and linux_x86_64.
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    return path


def _write_parquet(records: Sequence[object], path: Path, sort_by: Sequence[str]) -> Path:
    """Write Pydantic records to Parquet in a canonical row order."""
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [record.model_dump(mode="json") for record in records]  # type: ignore[attr-defined]
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(list(sort_by), kind="mergesort").reset_index(drop=True)
    frame.to_parquet(path, index=False)
    return path


def write_scores(scores: Sequence[ItemScore], out_dir: Path = DEFAULT_SCORING_DIR) -> Path:
    """Write the score table. One row per (response, scorer) -- never aggregated."""
    return _write_parquet(scores, out_dir / SCORES_FILENAME, _SCORE_SORT)


def read_scores(path: Path) -> list[ItemScore]:
    """Read the score table back into records.

    Lives beside the writer so the reader and writer of this format cannot drift apart -- the same
    reason ``read_prompts`` sits next to ``write_prompts``.
    """
    import pandas as pd

    frame = pd.read_parquet(path)
    # Parquet round-trips a None in an object column as None, but pandas may widen an all-null
    # column to NaN; normalize before validation so `cell_id` does not arrive as a float.
    records = frame.where(frame.notna(), None).to_dict(orient="records")
    return [ItemScore.model_validate(record) for record in records]


def write_drift_rows(rows: Sequence[DriftRow], out_dir: Path = DEFAULT_SCORING_DIR) -> Path:
    """Write the identity-drift artifact -- the series P7 fits ``A + B + A:B`` on."""
    return _write_parquet(rows, out_dir / DRIFT_FILENAME, _DRIFT_SORT)


def read_drift_rows(path: Path) -> list[DriftRow]:
    """Read the drift artifact back into records."""
    import pandas as pd

    frame = pd.read_parquet(path)
    records = frame.where(frame.notna(), None).to_dict(orient="records")
    return [DriftRow.model_validate(record) for record in records]


def render_agreement_csv(records: Sequence[AgreementRecord]) -> str:
    """Agreement as CSV. Byte-deterministic: sorted rows, LF terminators, fixed float format."""
    import csv
    import io

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "benchmark",
            "model_id",
            "harm_type",
            "condition",
            "metric",
            "scorer_a",
            "scorer_b",
            "value",
            "n_units",
            "n_raters",
        ]
    )
    for record in sorted(
        records,
        key=lambda r: (
            r.harm_type,
            r.condition,
            r.metric,
            r.scorer_a,
            r.scorer_b or "",
            r.model_id,
        ),
    ):
        writer.writerow(
            [
                record.benchmark,
                record.model_id,
                record.harm_type,
                record.condition,
                record.metric,
                record.scorer_a,
                record.scorer_b or "",
                "" if record.value is None else f"{record.value:.6f}",
                record.n_units,
                record.n_raters,
            ]
        )
    return buffer.getvalue()


def _escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_agreement_markdown(
    records: Sequence[AgreementRecord],
    *,
    scoring_version: str,
    reports: Sequence[StandardizationReport] = (),
) -> str:
    """Render the agreement table for humans. Generated file; never hand-edited."""
    lines = [
        "# Inter-scorer agreement",
        "",
        f"Scoring roster: `{scoring_version}`.",
        "",
        "Generated by `scripts/run_scoring.py`. **Do not edit by hand.**",
        "",
        "Krippendorff's alpha is computed at the **interval** level over `standardized`, with",
        "responses as units and scorers as raters. Cohen's kappa is **pairwise and nominal**,",
        "over `raw` binarized at each scorer's declared threshold. A blank value means fewer than",
        "the minimum units required, or a group with no variance to disagree about.",
        "",
        "| harm type | condition | metric | scorers | value | n units | n raters |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for record in sorted(
        records,
        key=lambda r: (r.harm_type, r.condition, r.metric, r.scorer_a, r.scorer_b or ""),
    ):
        pair = (
            record.scorer_a if record.scorer_b is None else f"{record.scorer_a} / {record.scorer_b}"
        )
        value = "—" if record.value is None else f"{record.value:.3f}"
        lines.append(
            f"| {_escape(record.harm_type)} | {record.condition} | {record.metric} | "
            f"`{_escape(pair)}` | {value} | {record.n_units} | {record.n_raters} |"
        )

    degenerate = [r for r in reports if r.degenerate]
    if degenerate:
        lines += [
            "",
            "## Degenerate standardization groups",
            "",
            "These groups had zero variance or too few rows, so every row standardized to 0.0.",
            "Listed because a silently all-zero group would look like perfect agreement.",
            "",
            "| scorer | harm type | condition | n | raw sd |",
            "|---|---|---|---:|---:|",
        ]
        for report in sorted(degenerate, key=lambda r: (r.harm_type, r.scorer_id, r.condition)):
            lines.append(
                f"| `{_escape(report.scorer_id)}` | {_escape(report.harm_type)} | "
                f"{report.condition} | {report.n} | {report.raw_sd:.6g} |"
            )

    lines += [
        "",
        "## What agreement does not tell you",
        "",
        "High agreement means the scorers behave alike, **not** that they are measuring harm",
        "correctly. Two identity-sensitive classifiers agree with each other and are wrong in the",
        "same direction (PROJECT_CONTEXT.md §9). Read this table alongside `drift_control.parquet`",
        "and `calibration_report.md`; a harm type needs all three before it is load-bearing.",
        "",
    ]
    return "\n".join(lines) + "\n"


def write_agreement(
    records: Sequence[AgreementRecord],
    *,
    scoring_version: str,
    reports: Sequence[StandardizationReport] = (),
    out_dir: Path = DEFAULT_SCORING_DIR,
) -> tuple[Path, Path]:
    """Write the agreement table as CSV and Markdown. Returns ``(csv_path, md_path)``."""
    csv_path = _write_text(out_dir / AGREEMENT_CSV_FILENAME, render_agreement_csv(records))
    md_path = _write_text(
        out_dir / AGREEMENT_MD_FILENAME,
        render_agreement_markdown(records, scoring_version=scoring_version, reports=reports),
    )
    return csv_path, md_path


def export_scoring(
    scores: Sequence[ItemScore],
    drift: Sequence[DriftRow],
    agreement: Sequence[AgreementRecord],
    *,
    scoring_version: str,
    reports: Sequence[StandardizationReport] = (),
    out_dir: Path = DEFAULT_SCORING_DIR,
) -> tuple[Path, Path, Path, Path]:
    """Write every P6 artifact. Returns ``(scores, drift, agreement_csv, agreement_md)``."""
    return (
        write_scores(scores, out_dir),
        write_drift_rows(drift, out_dir),
        *write_agreement(
            agreement, scoring_version=scoring_version, reports=reports, out_dir=out_dir
        ),
    )
