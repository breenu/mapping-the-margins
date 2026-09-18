"""Metric artifacts: Parquet round-trips and byte-deterministic Markdown (P7).

The P6 Parquet convention applies unchanged: **round-trip content equality on a canonically sorted
frame**, not byte identity. What is new here is that ``effects.parquet`` carries the unset
significance fields -- ``p``, ``q``, ``family_size``, ``correction`` -- as nulls, and they must
come back as ``None`` rather than as NaN or 0.0: an EffectEstimate that lost its "uncorrected"
status on the way through a file would let the report layer treat it as stamped.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from margins.metrics.export import (
    CELL_STATS_FILENAME,
    DRIFT_COMPARISON_MD_FILENAME,
    EFFECTS_FILENAME,
    read_cell_stats,
    read_effects,
    render_drift_comparison_markdown,
    write_cell_stats,
    write_effects,
)
from margins.metrics.records import DriftComparison, FamilyAgreement, HarmTypeFlag
from margins.schemas import CellStat, EffectEstimate

BENCHMARK = "open_gen_templates"


def make_effect(index: int, *, term: str = "alpha=a1:beta=b1") -> EffectEstimate:
    return EffectEstimate(
        benchmark=BENCHMARK,
        model="m",
        axis_pair=("alpha", "beta"),
        term=term,
        harm_type="toxicity",
        estimate=0.1 * index,
        se=0.05,
        ci_low=0.1 * index - 0.1,
        ci_high=0.1 * index + 0.12,
        powered=index % 2 == 0,
    )


def make_cell_stat(index: int) -> CellStat:
    return CellStat(
        cell_id=f"alpha=a{index}",
        benchmark=BENCHMARK,
        model="m",
        harm_type="toxicity",
        n=10 + index,
        mean_standardized=0.2 * index,
        ci_low=0.2 * index - 0.1,
        ci_high=0.2 * index + 0.1,
        powered=True,
    )


def test_effects_round_trip_with_significance_fields_still_unset(tmp_path: Path) -> None:
    effects = [make_effect(i, term=f"alpha=a{i}:beta=b1") for i in range(4)]
    path = write_effects(effects, tmp_path)
    assert path.name == EFFECTS_FILENAME

    back = read_effects(path)
    assert {e.term for e in back} == {e.term for e in effects}
    for recovered in back:
        original = next(e for e in effects if e.term == recovered.term)
        assert recovered.estimate == pytest.approx(original.estimate)
        assert recovered.ci_low == pytest.approx(original.ci_low)
        assert recovered.axis_pair == ("alpha", "beta")
        assert recovered.powered == original.powered
        assert recovered.p is None and recovered.q is None
        assert recovered.family_size is None and recovered.correction is None
        assert not recovered.corrected


def test_cell_stats_round_trip(tmp_path: Path) -> None:
    stats = [make_cell_stat(i) for i in range(3)]
    path = write_cell_stats(stats, tmp_path)
    assert path.name == CELL_STATS_FILENAME
    back = read_cell_stats(path)
    assert sorted(s.cell_id for s in back) == sorted(s.cell_id for s in stats)
    assert all(isinstance(s.n, int) for s in back)


def test_the_comparison_markdown_is_byte_deterministic_and_carries_both_series() -> None:
    comparisons = [
        DriftComparison(
            benchmark=BENCHMARK,
            model_id="m",
            harm_type="toxicity",
            scorer_id="toxic_bert",
            axis_a="alpha",
            axis_b="beta",
            control_family=family,
            model_rms=0.40,
            drift_rms=drift,
            ratio=drift / 0.40,
            comparable=drift / 0.40 >= 0.5,
            n_model_obs=100,
            n_drift_obs=30,
        )
        for family, drift in (("prompt_echo", 0.30), ("neutral_declarative", 0.05))
    ]
    agreement = [
        FamilyAgreement(
            benchmark=BENCHMARK,
            model_id="m",
            harm_type="toxicity",
            scorer_id="toxic_bert",
            axis_a="alpha",
            axis_b="beta",
            rms_prompt_echo=0.30,
            rms_neutral_declarative=0.05,
            ratio=6.0,
            families_disagree=True,
        )
    ]
    flags = [
        HarmTypeFlag(
            benchmark=BENCHMARK,
            model_id="m",
            harm_type="toxicity",
            descriptive_only_candidate=True,
            n_comparisons=2,
            n_comparable=1,
            scorers_comparable=("toxic_bert",),
        )
    ]

    first = render_drift_comparison_markdown(
        comparisons, agreement, flags, metrics_version="v", comparable_ratio=0.5
    )
    second = render_drift_comparison_markdown(
        list(reversed(comparisons)), agreement, flags, metrics_version="v", comparable_ratio=0.5
    )
    assert first == second  # input order does not reach the output
    assert "\r" not in first
    assert "prompt_echo" in first and "neutral_declarative" in first
    assert "DESCRIPTIVE-ONLY" in first.upper()
    assert "disagree" in first.lower()
    assert "not" in first.lower() and "subtract" in first.lower()
    assert DRIFT_COMPARISON_MD_FILENAME.endswith(".md")
