"""Gate P10: the committed result tables are declared, deterministic, and match their specs.

These tables are the seam between gitignored ``data/`` and the committed ``results/``: they are
what the manuscript prints, what the figures draw, and therefore the one place a figure and a
table can be made to disagree. The tests here are about that seam rather than about any number.

``test_the_row_builders_round_trip_real_records`` is the one that will matter at P11: it runs
``margins.report.build`` over genuine ``EffectEstimate`` / ``FitTerm`` / ``SignificanceClaim``
records -- the same types P7 and P9 write -- so the path from a real run to a committed table is
exercised now, months before a real run exists.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from margins.metrics.records import DriftComparison, FitReport, FitTerm
from margins.report.build import (
    MissingFitTermError,
    descriptive_only_rows,
    drift_rows,
    emergence_rows,
    fit_report_rows,
    single_axis_rows,
)
from margins.report.tables import (
    DATA_STATUS_COLUMN,
    TABLE_SPECS,
    Column,
    ResultTable,
    TableSpec,
    read_table,
    spec_for,
    write_table_csv,
    write_table_latex,
)
from margins.schemas import CellStat, EffectEstimate
from margins.stats.records import DescriptiveOnlyDecision, FamilyMember, SignificanceClaim
from tests.conftest import FIXTURE_TABLES

BENCHMARK = "open_gen_templates"
MODEL = "fake/model"


# ------------------------------------------------------------------ the specs


def test_every_shipped_table_matches_its_declared_spec() -> None:
    for spec in TABLE_SPECS:
        table = read_table(spec.name, directory=FIXTURE_TABLES)
        assert table.spec is spec
        assert len(table) > 0, f"{spec.name} is empty"
        for row in table.rows:
            assert set(row) == set(spec.column_names)


def test_every_table_declares_a_data_status_column() -> None:
    with pytest.raises(ValueError, match="data_status"):
        TableSpec(
            name="bad",
            title="t",
            caption="c",
            columns=(Column("x"),),
            key_columns=("x",),
        )


def test_a_table_that_mixes_real_and_fixture_rows_is_refused() -> None:
    spec = spec_for("descriptive_only")
    rows = (
        dict.fromkeys(spec.column_names, "") | {DATA_STATUS_COLUMN: "real"},
        dict.fromkeys(spec.column_names, "") | {DATA_STATUS_COLUMN: "fixture"},
    )
    table = ResultTable(spec, rows, origin="test")
    with pytest.raises(ValueError, match="mixes data_status"):
        _ = table.data_status


def test_an_undeclared_column_is_refused(tmp_path: Path) -> None:
    spec = spec_for("descriptive_only")
    with pytest.raises(ValueError, match="undeclared columns"):
        write_table_csv(spec, [{"nonsense": 1}], tmp_path / "x.csv")


def test_a_header_that_disagrees_with_the_spec_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "descriptive_only.csv"
    path.write_text("benchmark,model_id\nx,y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="declares"):
        read_table("descriptive_only", directory=tmp_path)


# ------------------------------------------------------------------ determinism and LaTeX


def test_the_csv_and_latex_forms_are_byte_deterministic(tmp_path: Path) -> None:
    for name in ("descriptive_only", "emergence_effects"):
        table = read_table(name, directory=FIXTURE_TABLES)
        digests = []
        for run in ("a", "b"):
            out = tmp_path / run
            out.mkdir(exist_ok=True)
            rows = [dict(row) for row in table.rows]
            csv = write_table_csv(table.spec, rows, out / f"{name}.csv")
            tex = write_table_latex(table, out / f"{name}.tex")
            digests.append(tuple(hashlib.sha256(p.read_bytes()).hexdigest() for p in (csv, tex)))
        assert digests[0] == digests[1]


def test_reading_a_written_table_back_gives_the_same_floats(tmp_path: Path) -> None:
    """The table is the authority for what a figure plots, so the round trip must be exact."""
    table = read_table("emergence_effects", directory=FIXTURE_TABLES)
    rows = [dict(row) for row in table.rows]
    write_table_csv(table.spec, rows, tmp_path / "emergence_effects.csv")
    reloaded = read_table("emergence_effects", directory=tmp_path)
    for before, after in zip(table.rows, reloaded.rows, strict=True):
        assert before == after


def test_the_latex_caption_admits_when_the_rows_are_fixtures(tmp_path: Path) -> None:
    table = read_table("descriptive_only", directory=FIXTURE_TABLES)
    text = write_table_latex(table, tmp_path / "t.tex").read_text(encoding="utf-8")
    assert "synthetic fixtures" in text
    assert "\\toprule" in text and "\\bottomrule" in text


def test_latex_escapes_the_underscores_in_a_model_id(tmp_path: Path) -> None:
    table = read_table("drift_side_by_side", directory=FIXTURE_TABLES)
    text = write_table_latex(table, tmp_path / "t.tex").read_text(encoding="utf-8")
    assert "toxic\\_bert" in text


# ------------------------------------------------------------------ the row builders


def _fit_term(term: str, kind: str, **overrides: object) -> FitTerm:
    data: dict[str, object] = {
        "benchmark": BENCHMARK,
        "model_id": MODEL,
        "harm_type": "regard",
        "condition": "model",
        "scorer_id": None,
        "axis_a": "age",
        "axis_b": "ability",
        "term": term,
        "kind": kind,
        "value_a": "young",
        "value_b": "blind",
        "estimate": 0.4,
        "se": 0.1,
        "ci_low": 0.2,
        "ci_high": 0.6,
        "n_obs": 400,
        "n_items": 100,
    }
    data.update(overrides)
    return FitTerm.model_validate(data)


def _effect(term: str, **overrides: object) -> EffectEstimate:
    data: dict[str, object] = {
        "benchmark": BENCHMARK,
        "model": MODEL,
        "axis_pair": ("age", "ability"),
        "term": term,
        "harm_type": "regard",
        "estimate": 0.4,
        "se": 0.1,
        "ci_low": 0.2,
        "ci_high": 0.6,
        "p": 0.001,
        "q": 0.01,
        "powered": True,
        "family_size": 168,
        "correction": "fdr_bh",
    }
    data.update(overrides)
    return EffectEstimate.model_validate(data)


def _claim(term: str) -> SignificanceClaim:
    return SignificanceClaim(
        benchmark=BENCHMARK,
        model_id=MODEL,
        harm_type="regard",
        axis_a="age",
        axis_b="ability",
        term=term,
        estimate=0.4,
        se=0.1,
        ci_low=0.2,
        ci_high=0.6,
        p=0.001,
        q=0.01,
        powered=True,
        status="claimable",
        claimable=True,
    )


TERM = "age=young:ability=blind"


def test_the_row_builders_round_trip_real_records(tmp_path: Path) -> None:
    rows = emergence_rows(
        [_effect(TERM)],
        [_fit_term(TERM, "interaction")],
        [_claim(TERM)],
        [
            FamilyMember(
                benchmark=BENCHMARK,
                model_id=MODEL,
                harm_type="regard",
                axis_a="age",
                axis_b="ability",
                term=TERM,
                status="present",
            )
        ],
    )
    path = write_table_csv(spec_for("emergence_effects"), rows, tmp_path / "emergence_effects.csv")
    table = read_table("emergence_effects", directory=path.parent)
    assert table.data_status == "real"
    (row,) = table.rows
    assert row["kind"] == "interaction"
    assert row["value_a"] == "young" and row["value_b"] == "blind"
    assert row["status"] == "claimable" and row["family_member"] == "true"


def test_an_estimate_with_no_matching_fit_term_raises() -> None:
    """Its shape would be missing, and a figure would silently drop it out of the grid."""
    with pytest.raises(MissingFitTermError):
        emergence_rows([_effect(TERM)], [], [_claim(TERM)], [])


def test_a_term_the_stats_run_never_gated_gets_an_empty_status() -> None:
    """Empty says 'never gated'. Inventing a status would be the figure outvoting P9."""
    (row,) = emergence_rows([_effect(TERM)], [_fit_term(TERM, "interaction")], [], [])
    assert row["status"] == ""
    assert row["claimable"] is False


def test_only_the_triangulated_model_series_reaches_the_tables() -> None:
    """A per-scorer fit of the same term carries a different estimate and must not be mixed in."""
    per_scorer = _fit_term(TERM, "interaction", scorer_id="regard_classifier", estimate=0.9)
    with pytest.raises(MissingFitTermError):
        emergence_rows([_effect(TERM)], [per_scorer], [_claim(TERM)], [])


def test_fit_report_rows_keep_the_not_estimable_ones() -> None:
    """The extractive paradigm's empty slot reaches the figure as a reason, not as a gap."""
    reports = [
        FitReport(
            benchmark="crows_pairs",
            model_id=MODEL,
            harm_type="stereotype_preference",
            condition="model",
            scorer_id=None,
            axis_a="age",
            axis_b="ability",
            estimable=False,
            reason="12 empty cells: no item attests them",
            n_obs=3,
            n_items=3,
            n_cells=12,
            n_params=6,
            cov_type="cluster",
            n_resamples=0,
        )
    ]
    (row,) = fit_report_rows(reports)
    assert row["estimable"] is False
    assert "empty cells" in row["reason"]
    assert row["rms_interaction"] is None


def test_single_axis_rows_exclude_order_two_cells() -> None:
    """An order-2 cell MEAN on the cross-paradigm figure would be a level standing in for γ."""
    stats = [
        CellStat(
            cell_id="age=young",
            benchmark=BENCHMARK,
            model=MODEL,
            harm_type="regard",
            n=108,
            mean_standardized=0.1,
            ci_low=0.0,
            ci_high=0.2,
            powered=True,
        ),
        CellStat(
            cell_id="age=young|ability=blind",
            benchmark=BENCHMARK,
            model=MODEL,
            harm_type="regard",
            n=108,
            mean_standardized=0.3,
            ci_low=0.2,
            ci_high=0.4,
            powered=True,
        ),
    ]
    rows = single_axis_rows(stats)
    assert [row["cell_id"] for row in rows] == ["age=young"]
    assert rows[0]["axis"] == "age" and rows[0]["value"] == "young"


def test_the_drift_and_decision_builders_carry_their_thresholds() -> None:
    comparison = DriftComparison(
        benchmark=BENCHMARK,
        model_id=MODEL,
        harm_type="toxicity",
        scorer_id="toxic_bert",
        axis_a="age",
        axis_b="ability",
        control_family="prompt_echo",
        model_rms=0.4,
        drift_rms=0.3,
        ratio=0.75,
        comparable=True,
        n_model_obs=100,
        n_drift_obs=40,
    )
    (row,) = drift_rows([comparison])
    assert row["ratio"] == pytest.approx(0.75)

    decision = DescriptiveOnlyDecision(
        benchmark=BENCHMARK,
        model_id=MODEL,
        harm_type="toxicity",
        descriptive_only=True,
        threshold=0.5,
        max_ratio=0.75,
        worst_scorer="toxic_bert",
        n_comparisons=14,
        n_over_threshold=7,
    )
    (decision_row,) = descriptive_only_rows([decision])
    assert decision_row["threshold"] == pytest.approx(0.5)
    assert decision_row["descriptive_only"] is True
