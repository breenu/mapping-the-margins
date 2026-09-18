"""The scoring artifacts, and what determinism means for each (P6).

**This is the project's first Parquet, so the file pins the weaker guarantee explicitly.** The
lattice and prompt exports assert BYTE identity. Parquet cannot: it embeds writer version and
per-column metadata, so the same table written by two `pyarrow` builds differs in bytes. What is
asserted instead -- and what P7 actually depends on -- is **round-trip content equality on a
canonically sorted frame**. Claiming byte identity here would be claiming a reproducibility property
we have not demonstrated.

The agreement CSV and Markdown *are* byte-deterministic, and are tested as such.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from margins.schemas import ItemScore
from margins.scoring.drift import DriftRow
from margins.scoring.export import (
    AGREEMENT_CSV_FILENAME,
    AGREEMENT_MD_FILENAME,
    DRIFT_FILENAME,
    SCORES_FILENAME,
    export_scoring,
    read_drift_rows,
    read_scores,
    render_agreement_csv,
    render_agreement_markdown,
    write_drift_rows,
    write_scores,
)
from margins.scoring.standardize import StandardizationReport
from margins.scoring.triangulate import AgreementRecord

BENCHMARK = "open_gen_templates"


def make_score(index: int, *, cell_id: str | None = "alpha=a1", scorer_id: str = "s1") -> ItemScore:
    return ItemScore(
        score_id=f"{scorer_id}::model::r{index}::abc123def456",
        response_id=f"r{index}",
        scorer_id=scorer_id,
        harm_type="toxicity",
        raw=index * 0.1,
        standardized=index * 0.1 - 0.2,
        cell_id=cell_id,
        benchmark=BENCHMARK,
        model_id="m",
        seed=20260918 + index,
        temperature=0.7,
        condition="model",
    )


def make_drift_row(index: int, *, order: int = 2) -> DriftRow:
    return DriftRow(
        score_id=f"s1::drift_control::c{index}::abc123def456",
        response_id=f"c{index}",
        scorer_id="s1",
        harm_type="toxicity",
        condition="drift_control",
        control_family="prompt_echo" if index % 2 else "neutral_declarative",
        cell_id="alpha=a1|beta=b1" if order == 2 else "alpha=a1",
        order=order,
        axis_a="alpha",
        value_a="a1",
        axis_b="beta" if order == 2 else None,
        value_b="b1" if order == 2 else None,
        benchmark=BENCHMARK,
        raw=index * 0.05,
        standardized=index * 0.05 - 0.1,
    )


def make_agreement(
    metric: str = "krippendorff_alpha", value: float | None = 0.87
) -> AgreementRecord:
    return AgreementRecord(
        benchmark=BENCHMARK,
        model_id="m",
        harm_type="toxicity",
        condition="model",
        metric=metric,
        scorer_a="ALL" if metric == "krippendorff_alpha" else "s1",
        scorer_b=None if metric == "krippendorff_alpha" else "s2",
        value=value,
        n_units=30,
        n_raters=2,
    )


# ------------------------------------------------------------------ parquet round-trips


def test_scores_round_trip_through_parquet(tmp_path: Path) -> None:
    """The property P7 depends on: values survive, exactly."""
    scores = [make_score(index) for index in range(5)]

    path = write_scores(scores, tmp_path)
    back = read_scores(path)

    assert {s.score_id for s in back} == {s.score_id for s in scores}
    by_id = {s.score_id: s for s in back}
    for original in scores:
        recovered = by_id[original.score_id]
        assert recovered.raw == pytest.approx(original.raw)
        assert recovered.standardized == pytest.approx(original.standardized)
        assert recovered.seed == original.seed
        assert recovered.condition == original.condition


def test_a_null_cell_id_survives_the_round_trip(tmp_path: Path) -> None:
    """pandas widens an all-null object column; cell_id must not come back as a float NaN."""
    path = write_scores([make_score(0, cell_id=None)], tmp_path)
    assert read_scores(path)[0].cell_id is None


def test_drift_rows_round_trip_with_their_axis_columns(tmp_path: Path) -> None:
    rows = [make_drift_row(index, order=1 if index % 2 else 2) for index in range(6)]

    back = read_drift_rows(write_drift_rows(rows, tmp_path))

    assert len(back) == len(rows)
    order_1 = [r for r in back if r.order == 1]
    assert order_1 and all(r.axis_b is None and r.value_b is None for r in order_1)
    order_2 = [r for r in back if r.order == 2]
    assert order_2 and all(r.axis_b == "beta" and r.value_b == "b1" for r in order_2)


def test_the_row_order_is_canonical_and_not_input_order(tmp_path: Path) -> None:
    """Sorted on write, so two runs that built the table in different orders agree."""
    scores = [make_score(index) for index in range(5)]

    forward = read_scores(write_scores(scores, tmp_path / "a"))
    reverse = read_scores(write_scores(list(reversed(scores)), tmp_path / "b"))

    assert [s.score_id for s in forward] == [s.score_id for s in reverse]


def test_an_empty_table_writes_and_reads(tmp_path: Path) -> None:
    assert read_scores(write_scores([], tmp_path)) == []


# ------------------------------------------------------------------ agreement (byte-deterministic)


def test_agreement_csv_is_byte_deterministic() -> None:
    records = [make_agreement(), make_agreement("cohen_kappa", 0.42)]

    first = render_agreement_csv(records)
    second = render_agreement_csv(list(reversed(records)))

    assert first == second
    assert "\r" not in first
    assert first.splitlines()[0].startswith("benchmark,model_id,harm_type")


def test_a_withheld_agreement_value_is_blank_not_zero() -> None:
    """None means 'not estimable'. Writing 0.0 would read as 'no agreement', which is different."""
    csv = render_agreement_csv([make_agreement(value=None)])
    assert ",,30,2" in csv


def test_agreement_markdown_names_what_agreement_does_not_tell_you() -> None:
    """The limitation is part of the artifact, not a thing a reader is expected to remember."""
    text = render_agreement_markdown([make_agreement()], scoring_version="v1")

    assert "Do not edit by hand" in text
    assert "0.870" in text
    assert "not** that they are measuring harm" in text
    assert "drift_control.parquet" in text


def test_degenerate_groups_are_listed_in_the_report() -> None:
    """An all-zero group looks like perfect agreement unless it is named."""
    report = StandardizationReport(
        benchmark=BENCHMARK,
        scorer_id="s1",
        harm_type="toxicity",
        condition="model",
        n=4,
        raw_mean=0.5,
        raw_sd=0.0,
        sign=1,
        degenerate=True,
    )

    text = render_agreement_markdown([make_agreement()], scoring_version="v1", reports=[report])
    assert "Degenerate standardization groups" in text
    assert "`s1`" in text


# ------------------------------------------------------------------ the full export


def test_export_writes_every_artifact(tmp_path: Path) -> None:
    scores = [make_score(index) for index in range(3)]
    drift = [make_drift_row(index) for index in range(3)]
    agreement = [make_agreement()]

    paths = export_scoring(scores, drift, agreement, scoring_version="v1", out_dir=tmp_path)

    assert [p.name for p in paths] == [
        SCORES_FILENAME,
        DRIFT_FILENAME,
        AGREEMENT_CSV_FILENAME,
        AGREEMENT_MD_FILENAME,
    ]
    assert all(p.is_file() for p in paths)
