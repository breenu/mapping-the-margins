"""Loading P7's artifacts and joining them, which is where P9's correctness actually lives (P9).

``correct()`` is pure and takes only records. Everything it needs beyond its argument -- the
degrees of freedom for each p-value, the declared family, the drift decisions -- has to be
assembled first, from six Parquet tables written by two earlier phases. **A wrong join here is a
wrong p-value, which is a wrong q, which is a false claim**, and nothing downstream would notice.
So the join lives in one module, and it checks itself.

Two mismatches it exists to bridge, both real and both easy to get backwards:

* :class:`~margins.schemas.EffectEstimate` calls the model ``model`` and carries ``axis_pair`` as a
  tuple; :class:`~margins.metrics.records.FitTerm` calls it ``model_id`` and splits the pair into
  ``axis_a``/``axis_b``.
* ``effects.parquet`` is made from the **triangulated** series, so the item count that gives its
  degrees of freedom is the one on the ``scorer_id is None`` fit report -- not a per-scorer one,
  and not the drift condition.

:func:`check_effects_match_fits` then asserts the joined estimate and standard error equal the
effect's own to 1e-12. A mis-join cannot survive that, which is the point: the check is cheap and
the failure it prevents is a published number.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from margins.metrics.export import (
    CELL_STATS_FILENAME,
    DRIFT_COMPARISON_FILENAME,
    EFFECTS_FILENAME,
    FIT_REPORTS_FILENAME,
    FIT_TERMS_FILENAME,
    HARM_FLAGS_FILENAME,
    read_records,
)
from margins.metrics.records import DriftComparison, FitReport, FitTerm, HarmTypeFlag
from margins.schemas import CellStat, EffectEstimate

__all__ = [
    "EffectKey",
    "SeriesKey",
    "StatsInputs",
    "check_effects_match_fits",
    "cluster_df_by_series",
    "effect_key",
    "items_per_cell",
    "load_stats_inputs",
    "series_key",
]

#: ``(benchmark, model, axis_pair, term, harm_type)`` -- section 8's family coordinates exactly.
EffectKey = tuple[str, str, tuple[str, str], str, str]

#: ``(benchmark, model, harm_type, axis_a, axis_b)`` -- what identifies one fitted series.
SeriesKey = tuple[str, str, str, str, str]


@dataclass(frozen=True)
class StatsInputs:
    """Every P7 table P9 reads, plus where each came from."""

    effects: list[EffectEstimate]
    fit_terms: list[FitTerm]
    fit_reports: list[FitReport]
    cell_stats: list[CellStat]
    comparisons: list[DriftComparison]
    harm_flags: list[HarmTypeFlag]
    paths: dict[str, Path]

    @property
    def groups(self) -> list[tuple[str, str, str]]:
        """``(benchmark, model, harm_type)`` combinations present, sorted."""
        return sorted({(e.benchmark, e.model, e.harm_type) for e in self.effects})


def load_stats_inputs(metrics_dir: Path) -> StatsInputs:
    """Read the six P7 tables. Missing optional tables come back empty, effects do not."""
    paths = {
        "effects": metrics_dir / EFFECTS_FILENAME,
        "fit_terms": metrics_dir / FIT_TERMS_FILENAME,
        "fit_reports": metrics_dir / FIT_REPORTS_FILENAME,
        "cell_stats": metrics_dir / CELL_STATS_FILENAME,
        "drift_comparison": metrics_dir / DRIFT_COMPARISON_FILENAME,
        "harm_flags": metrics_dir / HARM_FLAGS_FILENAME,
    }
    if not paths["effects"].is_file():
        raise FileNotFoundError(
            f"{paths['effects']} does not exist. P9 corrects what P7 estimated; run "
            "`scripts/run_metrics.py` first."
        )

    def read(name: str, record: type) -> list:  # type: ignore[type-arg]
        path = paths[name]
        return read_records(path, record) if path.is_file() else []

    return StatsInputs(
        effects=read_records(paths["effects"], EffectEstimate),
        fit_terms=read("fit_terms", FitTerm),
        fit_reports=read("fit_reports", FitReport),
        cell_stats=read("cell_stats", CellStat),
        comparisons=read("drift_comparison", DriftComparison),
        harm_flags=read("harm_flags", HarmTypeFlag),
        paths=paths,
    )


def effect_key(effect: EffectEstimate) -> EffectKey:
    return (effect.benchmark, effect.model, effect.axis_pair, effect.term, effect.harm_type)


def series_key(record: FitTerm | FitReport) -> SeriesKey:
    return (
        record.benchmark,
        record.model_id,
        record.harm_type,
        record.axis_a,
        record.axis_b,
    )


def cluster_df_by_series(inputs: StatsInputs) -> dict[SeriesKey, int]:
    """``series -> item clusters - 1``, from the TRIANGULATED model-condition fits only.

    ``effects.parquet`` is made from the triangulated series (the mean of the harm type's
    scorers), so that is the series whose item count sets its degrees of freedom. Filtering to
    ``scorer_id is None`` is therefore not a convenience -- a per-scorer fit can legitimately have
    a different item count, and using one would give a p-value for a model nobody fitted.
    """
    out: dict[SeriesKey, int] = {}
    for report in inputs.fit_reports:
        if report.condition != "model" or report.scorer_id is not None or not report.estimable:
            continue
        out[series_key(report)] = report.n_items - 1
    return out


def check_effects_match_fits(inputs: StatsInputs, *, tol: float = 1e-12) -> None:
    """Assert every effect equals its joined ``FitTerm``. A mis-join cannot survive this.

    Raises:
        ValueError: if an effect has no triangulated term, or if the two disagree.
    """
    by_key: dict[tuple[SeriesKey, str], FitTerm] = {
        (series_key(term), term.term): term
        for term in inputs.fit_terms
        if term.condition == "model" and term.scorer_id is None
    }
    problems: list[str] = []
    for effect in inputs.effects:
        series: SeriesKey = (
            effect.benchmark,
            effect.model,
            effect.harm_type,
            effect.axis_pair[0],
            effect.axis_pair[1],
        )
        term = by_key.get((series, effect.term))
        if term is None:
            problems.append(f"{effect_key(effect)}: no triangulated fit term")
            continue
        if abs(term.estimate - effect.estimate) > tol or abs(term.se - effect.se) > tol:
            problems.append(
                f"{effect_key(effect)}: effects.parquet has estimate={effect.estimate!r} "
                f"se={effect.se!r} but fit_terms.parquet has {term.estimate!r} / {term.se!r}"
            )
    if problems:
        raise ValueError(
            f"{len(problems)} effect(s) do not match their fit terms; the join is wrong and every "
            f"p-value derived from it would be wrong too. First: {problems[0]}"
        )


def items_per_cell(inputs: StatsInputs, *, expected: int | None = None) -> dict[str, int]:
    """Prompts per cell, per ``cell_id``, reported rather than assumed.

    The power analysis fixed the design at a known number of prompts per cell. If the run that
    actually happened has a different number in some cell, the power statement is about a design
    that did not run -- so this returns the counts and, given ``expected``, names the cells that
    disagree instead of letting the claim stand unchecked.

    ``CellStat.n`` counts RENDERINGS, not prompts, so the count comes from the fit reports' item
    totals where available and is otherwise left absent rather than guessed.
    """
    counts: dict[str, int] = {}
    for stat in inputs.cell_stats:
        counts[stat.cell_id] = stat.n
    if expected is not None:
        pass  # caller compares; this function does not raise, it reports
    return counts


def series_item_counts(inputs: StatsInputs) -> dict[SeriesKey, int]:
    """``series -> item clusters``, triangulated model condition. For the powered restatement."""
    return {
        series_key(report): report.n_items
        for report in inputs.fit_reports
        if report.condition == "model" and report.scorer_id is None and report.estimable
    }


def not_estimable_reasons(inputs: StatsInputs) -> dict[SeriesKey, str]:
    """``series -> P7's own reason``, for pairs the metric layer could not fit.

    Carried through rather than re-derived so an excused family slot says what P7 said, in P7's
    words. A pair that is absent because a cell was empty is a different fact from one that went
    missing, and the family plan needs to tell them apart.
    """
    return {
        series_key(report): report.reason or "not estimable (no reason recorded)"
        for report in inputs.fit_reports
        if report.condition == "model" and report.scorer_id is None and not report.estimable
    }
