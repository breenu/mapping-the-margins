"""Metric artifacts under ``data/metrics/`` (P7).

The P6 Parquet convention applies unchanged: rows are written in a canonical sort order and what is
asserted is **round-trip content equality on a sorted frame**, not byte identity (Parquet embeds
writer metadata). The two human-facing Markdown files ARE byte-deterministic, following
:mod:`margins.scoring.export`.

Two things specific to this stage:

* ``effects.parquet`` carries ``p``, ``q``, ``family_size`` and ``correction`` as nulls, and the
  reader must hand them back as ``None`` -- an :class:`~margins.schemas.EffectEstimate` that came
  back "corrected" from a file would let the report layer treat an uncorrected estimate as stamped.
  P9 reads this file, corrects, and writes its own.
* ``drift_comparison.md`` is the reviewer-facing form of section 9's rule. It says, in words, that
  nothing was subtracted and that the thresholds are placeholders, because a table of ratios and
  flags on its own invites exactly the over-reading the descriptive-only rule exists to prevent.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final, TypeVar

from pydantic import BaseModel

from margins.metrics.compare import DriftComparisonResult
from margins.metrics.config import MetricsConfig
from margins.metrics.drift import DriftFitResult
from margins.metrics.metric import EmergenceResult
from margins.metrics.records import (
    DriftComparison,
    FamilyAgreement,
    FitReport,
    FitTerm,
    HarmTypeFlag,
    SideBySide,
)
from margins.schemas import CellStat, EffectEstimate

__all__ = [
    "CELL_STATS_FILENAME",
    "DEFAULT_METRICS_DIR",
    "DRIFT_COMPARISON_FILENAME",
    "DRIFT_COMPARISON_MD_FILENAME",
    "EFFECTS_FILENAME",
    "FAMILY_AGREEMENT_FILENAME",
    "FIT_REPORTS_FILENAME",
    "FIT_REPORT_MD_FILENAME",
    "FIT_TERMS_FILENAME",
    "HARM_FLAGS_FILENAME",
    "SIDE_BY_SIDE_FILENAME",
    "export_metrics",
    "read_cell_stats",
    "read_effects",
    "read_records",
    "render_drift_comparison_markdown",
    "render_fit_report_markdown",
    "write_cell_stats",
    "write_effects",
    "write_records",
]

DEFAULT_METRICS_DIR: Final[Path] = Path("data") / "metrics"

CELL_STATS_FILENAME: Final[str] = "cell_stats.parquet"
EFFECTS_FILENAME: Final[str] = "effects.parquet"
FIT_TERMS_FILENAME: Final[str] = "fit_terms.parquet"
FIT_REPORTS_FILENAME: Final[str] = "fit_reports.parquet"
DRIFT_COMPARISON_FILENAME: Final[str] = "drift_comparison.parquet"
FAMILY_AGREEMENT_FILENAME: Final[str] = "family_agreement.parquet"
HARM_FLAGS_FILENAME: Final[str] = "harm_flags.parquet"
SIDE_BY_SIDE_FILENAME: Final[str] = "drift_side_by_side.parquet"
DRIFT_COMPARISON_MD_FILENAME: Final[str] = "drift_comparison.md"
FIT_REPORT_MD_FILENAME: Final[str] = "fit_report.md"

#: Canonical sort keys. ``axis_pair`` on EffectEstimate is a list column, so the effects sort goes
#: through a derived scalar key (see ``_write_parquet``).
_EFFECTS_SORT: Final[tuple[str, ...]] = ("benchmark", "model", "harm_type", "_axis_pair", "term")
_CELL_STATS_SORT: Final[tuple[str, ...]] = ("benchmark", "model", "harm_type", "cell_id")
_FIT_SORT: Final[tuple[str, ...]] = (
    "benchmark",
    "model_id",
    "harm_type",
    "condition",
    "_scorer_id",
    "_control_family",
    "axis_a",
    "axis_b",
)
_TERM_SORT: Final[tuple[str, ...]] = (*_FIT_SORT, "term")
_COMPARISON_SORT: Final[tuple[str, ...]] = (
    "benchmark",
    "model_id",
    "harm_type",
    "scorer_id",
    "axis_a",
    "axis_b",
    "control_family",
)
_AGREEMENT_SORT: Final[tuple[str, ...]] = _COMPARISON_SORT[:-1]
_FLAGS_SORT: Final[tuple[str, ...]] = ("benchmark", "model_id", "harm_type")
_SIDE_SORT: Final[tuple[str, ...]] = (*_AGREEMENT_SORT, "term")

R = TypeVar("R", bound=BaseModel)


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    return path


def _write_parquet(records: Sequence[BaseModel], path: Path, sort_by: Sequence[str]) -> Path:
    """Write records in a canonical order. Derived ``_``-prefixed sort keys are dropped after."""
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [record.model_dump(mode="json") for record in records]
    frame = pd.DataFrame(rows)
    if not frame.empty:
        if "_axis_pair" in sort_by:
            frame["_axis_pair"] = frame["axis_pair"].map(lambda pair: "|".join(pair))
        if "_scorer_id" in sort_by:
            frame["_scorer_id"] = frame["scorer_id"].fillna("")
        if "_control_family" in sort_by:
            frame["_control_family"] = frame["control_family"].fillna("")
        frame = frame.sort_values(list(sort_by), kind="mergesort").reset_index(drop=True)
        frame = frame.drop(columns=[c for c in frame.columns if c.startswith("_")])
    frame.to_parquet(path, index=False)
    return path


def read_records(path: Path, record: type[R]) -> list[R]:
    """Read a Parquet table written by this module back into records.

    Nulls come back as ``None`` (not NaN), which is what keeps an uncorrected EffectEstimate
    uncorrected across the file boundary.
    """
    import pandas as pd

    frame = pd.read_parquet(path)
    out: list[R] = []
    for row in frame.to_dict(orient="records"):
        cleaned = {
            key: (None if _is_missing(value) else _to_python(value)) for key, value in row.items()
        }
        out.append(record.model_validate(cleaned))
    return out


def _is_missing(value: object) -> bool:
    import pandas as pd

    if isinstance(value, list | tuple):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # pragma: no cover - arrays and other exotica
        return False


def _to_python(value: object) -> object:
    import numpy as np

    if isinstance(value, np.ndarray):
        return [_to_python(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_records(records: Sequence[BaseModel], path: Path, sort_by: Sequence[str]) -> Path:
    return _write_parquet(records, path, sort_by)


def write_effects(effects: Sequence[EffectEstimate], out_dir: Path = DEFAULT_METRICS_DIR) -> Path:
    """Write the uncorrected estimates. P9 reads this and writes its own corrected table."""
    return _write_parquet(effects, out_dir / EFFECTS_FILENAME, _EFFECTS_SORT)


def read_effects(path: Path) -> list[EffectEstimate]:
    return read_records(path, EffectEstimate)


def write_cell_stats(stats: Sequence[CellStat], out_dir: Path = DEFAULT_METRICS_DIR) -> Path:
    return _write_parquet(stats, out_dir / CELL_STATS_FILENAME, _CELL_STATS_SORT)


def read_cell_stats(path: Path) -> list[CellStat]:
    return read_records(path, CellStat)


# --------------------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------------------


def _escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_drift_comparison_markdown(
    comparisons: Sequence[DriftComparison],
    agreement: Sequence[FamilyAgreement],
    flags: Sequence[HarmTypeFlag],
    *,
    metrics_version: str,
    comparable_ratio: float,
) -> str:
    """The section 9 comparison for humans. Generated; never hand-edited; byte-deterministic."""
    lines = [
        "# Scorer identity drift beside the model's interaction",
        "",
        f"Metrics config: `{metrics_version}`.",
        "",
        "Generated by `scripts/run_metrics.py`. **Do not edit by hand.**",
        "",
        "For each scorer and axis pair, `model_rms` is the root-mean-square of the interaction",
        "coefficients of `harm ~ A + B + A:B` fitted on the MODEL's output **as scored by that",
        "scorer**, and `drift_rms` is the same quantity fitted on the scorer's own scores of",
        "known-neutral text carrying the same descriptors (PROJECT_CONTEXT.md §9). Both are in SD",
        "units of the scorer's standardized scale. **Nothing is subtracted**: the two series are",
        "reported side by side, and `ratio = drift_rms / model_rms`.",
        "",
        f"`comparable` is `ratio >= {comparable_ratio:g}`. That threshold is a **PLACEHOLDER**",
        "(TODO(P9): pre-register). A harm type with any comparable row is a candidate for",
        "**DESCRIPTIVE-ONLY** status; the downgrade decision is P9's, not this table's.",
        "",
        "## Harm-type flags",
        "",
        "| benchmark | model | harm type | descriptive-only candidate | comparable / comparisons "
        "| scorers |",
        "|---|---|---|---|---:|---|",
    ]
    for flag in sorted(flags, key=lambda f: (f.benchmark, f.model_id, f.harm_type)):
        scorers = ", ".join(f"`{_escape(s)}`" for s in flag.scorers_comparable) or "—"
        lines.append(
            f"| {_escape(flag.benchmark)} | `{_escape(flag.model_id)}` | {_escape(flag.harm_type)} "
            f"| {'**YES**' if flag.descriptive_only_candidate else 'no'} "
            f"| {flag.n_comparable} / {flag.n_comparisons} | {scorers} |"
        )

    lines += [
        "",
        "## Per scorer, pair and control family",
        "",
        "| harm type | scorer | axis pair | family | model rms | drift rms | ratio | comparable "
        "| n model | n drift |",
        "|---|---|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for comparison in sorted(
        comparisons,
        key=lambda c: (
            c.benchmark,
            c.model_id,
            c.harm_type,
            c.scorer_id,
            c.axis_a,
            c.axis_b,
            c.control_family,
        ),
    ):
        lines.append(
            f"| {_escape(comparison.harm_type)} | `{_escape(comparison.scorer_id)}` "
            f"| {_escape(comparison.axis_a)} x {_escape(comparison.axis_b)} "
            f"| {comparison.control_family} "
            f"| {_fmt(comparison.model_rms)} | {_fmt(comparison.drift_rms)} "
            f"| {_fmt(comparison.ratio, 2)} | {'yes' if comparison.comparable else 'no'} "
            f"| {comparison.n_model_obs} | {comparison.n_drift_obs} |"
        )

    lines += [
        "",
        "## Do the two neutral families agree?",
        "",
        "`prompt_echo` is the imperative instruction register; `neutral_declarative` is flat",
        "prose. A scorer whose drift differs between them responds to identity terms differently",
        "depending on the text around them -- a style-dependence finding, reported here rather",
        "than averaged away. `ratio` is prompt_echo / neutral_declarative.",
        "",
        "| harm type | scorer | axis pair | prompt_echo rms | neutral_declarative rms | ratio "
        "| disagree |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for pair_agreement in sorted(
        agreement,
        key=lambda a: (a.benchmark, a.model_id, a.harm_type, a.scorer_id, a.axis_a, a.axis_b),
    ):
        if pair_agreement.families_disagree is None:
            disagree = "—"
        else:
            disagree = "**yes**" if pair_agreement.families_disagree else "no"
        lines.append(
            f"| {_escape(pair_agreement.harm_type)} | `{_escape(pair_agreement.scorer_id)}` "
            f"| {_escape(pair_agreement.axis_a)} x {_escape(pair_agreement.axis_b)} "
            f"| {_fmt(pair_agreement.rms_prompt_echo)} "
            f"| {_fmt(pair_agreement.rms_neutral_declarative)} "
            f"| {_fmt(pair_agreement.ratio, 2)} | {disagree} |"
        )

    lines += [
        "",
        "## What this table does not do",
        "",
        "It does not net the drift out of the model's interaction, and it does not decide.",
        "A scorer whose own interaction is comparable to the model's cannot distinguish emergence",
        "in the model from emergence in the instrument; the honest consequence is that the harm",
        "type becomes descriptive-only, and that consequence is applied at P9 against the",
        "pre-registered threshold, not here.",
        "",
    ]
    return "\n".join(lines) + "\n"


def render_fit_report_markdown(reports: Sequence[FitReport], *, metrics_version: str) -> str:
    """Every fit attempted, estimable or not, with the reason when not. Byte-deterministic."""
    lines = [
        "# Fit report",
        "",
        f"Metrics config: `{metrics_version}`.",
        "",
        "Generated by `scripts/run_metrics.py`. **Do not edit by hand.**",
        "",
        "One row per series and axis pair. `series` is `triangulated` (the mean of the harm",
        "type's scorers, the series `effects.parquet` is made from), a scorer id (that scorer's",
        "own view of the model), or `drift:<scorer>:<family>`. A pair that is not estimable is",
        "listed with the reason; it produced no EffectEstimate and no NaN.",
        "",
        "| harm type | series | axis pair | estimable | n obs | n items | rms interaction | CI "
        "| powered | reason |",
        "|---|---|---|---|---:|---:|---:|---|---|---|",
    ]

    def series(report: FitReport) -> str:
        if report.condition == "drift_control":
            return f"drift:{report.scorer_id}:{report.control_family}"
        return "triangulated" if report.scorer_id is None else str(report.scorer_id)

    for report in sorted(
        reports,
        key=lambda r: (
            r.benchmark,
            r.model_id,
            r.harm_type,
            r.condition,
            r.scorer_id or "",
            r.control_family or "",
            r.axis_a,
            r.axis_b,
        ),
    ):
        ci = (
            "—"
            if report.rms_ci_low is None or report.rms_ci_high is None
            else f"[{report.rms_ci_low:.3f}, {report.rms_ci_high:.3f}]"
        )
        powered = "—" if report.powered is None else ("yes" if report.powered else "no")
        lines.append(
            f"| {_escape(report.harm_type)} | `{_escape(series(report))}` "
            f"| {_escape(report.axis_a)} x {_escape(report.axis_b)} "
            f"| {'yes' if report.estimable else '**no**'} | {report.n_obs} | {report.n_items} "
            f"| {_fmt(report.rms_interaction)} | {ci} | {powered} "
            f"| {_escape(report.reason or '')} |"
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
# everything
# --------------------------------------------------------------------------------------


def export_metrics(
    result: EmergenceResult,
    drift: DriftFitResult,
    comparison: DriftComparisonResult,
    *,
    config: MetricsConfig,
    out_dir: Path = DEFAULT_METRICS_DIR,
) -> dict[str, Path]:
    """Write every P7 artifact. Returns ``{name: path}``."""
    all_terms: list[FitTerm] = [*result.fit_terms, *drift.fit_terms]
    all_reports: list[FitReport] = [*result.fit_reports, *drift.fit_reports]
    side: list[SideBySide] = comparison.side_by_side
    return {
        "cell_stats": write_cell_stats(result.cell_stats, out_dir),
        "effects": write_effects(result.effects, out_dir),
        "fit_terms": _write_parquet(all_terms, out_dir / FIT_TERMS_FILENAME, _TERM_SORT),
        "fit_reports": _write_parquet(all_reports, out_dir / FIT_REPORTS_FILENAME, _FIT_SORT),
        "drift_comparison": _write_parquet(
            comparison.comparisons, out_dir / DRIFT_COMPARISON_FILENAME, _COMPARISON_SORT
        ),
        "family_agreement": _write_parquet(
            comparison.family_agreement, out_dir / FAMILY_AGREEMENT_FILENAME, _AGREEMENT_SORT
        ),
        "harm_flags": _write_parquet(
            comparison.harm_flags, out_dir / HARM_FLAGS_FILENAME, _FLAGS_SORT
        ),
        "side_by_side": _write_parquet(side, out_dir / SIDE_BY_SIDE_FILENAME, _SIDE_SORT),
        "drift_comparison_md": _write_text(
            out_dir / DRIFT_COMPARISON_MD_FILENAME,
            render_drift_comparison_markdown(
                comparison.comparisons,
                comparison.family_agreement,
                comparison.harm_flags,
                metrics_version=config.metrics_version,
                comparable_ratio=config.drift.comparable_ratio,
            ),
        ),
        "fit_report_md": _write_text(
            out_dir / FIT_REPORT_MD_FILENAME,
            render_fit_report_markdown(all_reports, metrics_version=config.metrics_version),
        ),
    }
