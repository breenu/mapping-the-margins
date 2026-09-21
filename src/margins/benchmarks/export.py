"""The P2 artifacts: items, the exclusion log, and the two reports.

Gate P2 asks for "checksums recorded; 5-item eyeball; every exclusion logged with a reason;
screening is deterministic". Three of those four are properties of what this module writes, so
every artifact here is **byte-deterministic**: canonical JSON with sorted keys, ``newline=""`` so
an explicit ``\\n`` is never translated to ``\\r\\n`` on Windows, and no clock or path baked into
the contents. Regenerating on either locked environment produces identical bytes, which is what
makes "screening is deterministic" checkable rather than asserted.

The **coverage report** is the artifact that earns its place beyond the gate. The extractive
paradigm's order-2 yield decides whether CrowS-Pairs can support an interaction estimate at all,
and that yield is a fact about the dataset that had to be MEASURED. Writing it down as a first-
class artifact, rather than discovering it in a notebook, is the difference between a reported
result and an unexplained absence.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Sequence
from pathlib import Path

from margins.benchmarks.config import (
    BENCHMARK_INPUT_KEY,
    BENCHMARK_SOURCE_PREFIX,
    BenchmarkConfig,
    file_sha256,
)
from margins.benchmarks.crows_pairs import crows_pairs_path
from margins.benchmarks.screening import ScreeningResult
from margins.schemas import Item

__all__ = [
    "COVERAGE_FILENAME",
    "EXCLUSIONS_FILENAME",
    "ITEMS_FILENAME",
    "SCREENING_FILENAME",
    "benchmark_inputs",
    "export_benchmark",
    "read_items",
    "render_coverage_report",
    "render_exclusions_csv",
    "render_items_jsonl",
    "render_screening_report",
]

#: Raw text goes to JSONL, tabular to CSV (PROJECT_CONTEXT.md section 6). The exclusion log is
#: `excluded_items.csv` because section 9 names that file by name.
ITEMS_FILENAME = "items_v1.jsonl"
EXCLUSIONS_FILENAME = "excluded_items.csv"
SCREENING_FILENAME = "screening_report.md"
COVERAGE_FILENAME = "coverage_report.md"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    return path


def render_items_jsonl(items: Sequence[Item]) -> str:
    """Render items as JSONL, one record per line, in the order given."""
    return "".join(
        json.dumps(
            item.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        + "\n"
        for item in items
    )


def read_items(path: Path) -> list[Item]:
    """Read an item JSONL file back into records, in file order.

    The inverse of :func:`render_items_jsonl`, kept beside it so the reader and the writer of this
    format cannot drift apart -- the same argument ``prompts.export.read_prompts`` makes.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"item file not found: {path}. It is gitignored and regenerable -- run "
            "`uv run python scripts/build_crows_items.py`."
        )
    return [
        Item.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def render_exclusions_csv(result: ScreeningResult) -> str:
    """The machine-readable exclusion log (PROJECT_CONTEXT.md section 9).

    One row per excluded item, carrying the reason as a stable enum **value** plus enough of the
    source to audit the decision without re-reading the pinned file. ``lineterminator="\\n"`` and
    the ``newline=""`` in :func:`_write` together keep the bytes identical on both environments.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        ["row_index", "bias_type", "reason", "detail", "sent_stereo", "sent_antistereo"]
    )
    for exclusion in result.exclusions:
        writer.writerow(
            [
                exclusion.row_index,
                exclusion.bias_type,
                exclusion.reason.value,
                exclusion.detail,
                exclusion.sent_stereo,
                exclusion.sent_antistereo,
            ]
        )
    return buffer.getvalue()


def _cell(value: object) -> str:
    """One table cell, with pipes escaped.

    Cell ids join their axes with ``|`` (``race_ethnicity=Black|gender_and_sex=female``), which is
    also the markdown column separator -- unescaped, every order-2 row silently renders as two
    columns and the table stops meaning anything.
    """
    return str(value).replace("|", "\\|")


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(_cell(cell) for cell in row) + " |" for row in rows)
    return lines


def render_screening_report(
    result: ScreeningResult, config: BenchmarkConfig, *, n_rows: int
) -> str:
    """The human-facing screening summary. Byte-deterministic."""
    lines: list[str] = [
        f"# Screening report - {config.benchmark}",
        "",
        f"Source `{config.source.source}` at revision `{config.source.revision}`, "
        f"sha256 `{config.source.sha256[:12]}...`, {n_rows} rows.",
        "",
        "Screening is a deterministic function of the pinned rows and this config: no I/O, no RNG,",
        "no clock, and no dependence on row order. Every excluded row carries exactly one",
        f"machine-readable reason, logged in `{EXCLUSIONS_FILENAME}`.",
        "",
        "## Outcome",
        "",
    ]
    lines.extend(
        _table(
            ["outcome", "items"],
            [
                ["read from source", n_rows],
                ["kept", result.n_kept],
                ["excluded", result.n_excluded],
            ],
        )
    )
    lines.extend(["", "## Exclusions by reason", ""])
    counts = result.counts_by_reason()
    if counts:
        lines.extend(_table(["reason", "items"], sorted(counts.items())))
    else:
        lines.append("_No exclusions._")

    lines.extend(
        [
            "",
            "## Grammatical-concord allowance",
            "",
            "A differing region made only of closed-class agreement tokens is obligatory concord,",
            "not an independent second contrast: swapping `woman` for `man` forces `she` to become",
            "`he`. Such a region does not count toward `max_differing_spans`. The allowance is",
            "declared in the config so it is visible, and its effect is stated here so the",
            "exclusion count stays comparable to a stricter reading of the same data.",
            "",
        ]
    )
    lines.extend(
        _table(
            ["quantity", "value"],
            [
                ["agreement tokens declared", len(config.screening.agreement_tokens)],
                ["rows admitted by the allowance", len(result.admitted_by_agreement)],
                [
                    "`not_minimal_pair` without it",
                    counts.get("not_minimal_pair", 0) + len(result.admitted_by_agreement),
                ],
            ],
        )
    )
    return "\n".join(lines) + "\n"


def render_coverage_report(result: ScreeningResult, config: BenchmarkConfig) -> str:
    """What the kept items attest, and -- the load-bearing number -- the order-2 yield."""
    order_counts = result.counts_by_order()
    order_2 = result.order_2_cells()
    no_cell: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for row in result.coverage:
        if row.order == 0 and row.no_cell_reason is not None:
            no_cell[row.no_cell_reason] = no_cell.get(row.no_cell_reason, 0) + 1
        if row.order:
            by_category[row.bias_type] = by_category.get(row.bias_type, 0) + 1

    lines: list[str] = [
        f"# Lattice coverage - {config.benchmark}",
        "",
        "How many kept items the lattice can actually place, and at what order. **The order-2",
        "count is the number that decides whether this paradigm can support an interaction",
        "estimate at all**, because the emergence estimator fits order-2 cells only and requires",
        "every cell of a pair's grid to be non-empty.",
        "",
        "Attestation is literal surface matching against the lattice's own values. There is no",
        "alias table: `poor` is not read as `working-class`, `Mexican` is not read as `Hispanic`.",
        "An item whose descriptor is not a lattice value attests nothing and is recorded as such.",
        "",
        "## Kept items by cell order",
        "",
    ]
    lines.extend(_table(["order", "items"], sorted(order_counts.items())))

    lines.extend(["", "## Kept items carrying no cell", "", "**These are not exclusions.**", ""])
    if no_cell:
        lines.extend(_table(["reason", "items"], sorted(no_cell.items())))
    else:
        lines.append("_Every kept item carries a cell._")

    lines.extend(["", "## Celled items by upstream category", ""])
    if by_category:
        lines.extend(_table(["bias_type", "items"], sorted(by_category.items())))
    else:
        lines.append("_No item attested a lattice value._")

    lines.extend(["", "## Order-2 cells populated", ""])
    if order_2:
        lines.extend(_table(["cell_id", "items"], sorted(order_2.items())))
    else:
        lines.append("_None._")

    lines.extend(
        [
            "",
            "## Out-of-lattice categories",
            "",
            "Kept and marked, never dropped and never forced into a cell. The lattice declares no",
            "axis for them, and adding one would mean authoring a descriptor taxonomy rather than",
            "resolving it from a pinned source.",
            "",
        ]
    )
    lines.extend(
        _table(
            ["category"],
            [[category] for category in config.out_of_lattice_categories],
        )
    )
    return "\n".join(lines) + "\n"


def benchmark_inputs(
    config: BenchmarkConfig, *, config_path: Path, data_dir: Path | None
) -> dict[str, str]:
    """``input_checksums`` for the run manifest: the config file and the pinned source file."""
    return {
        BENCHMARK_INPUT_KEY: file_sha256(config_path),
        f"{BENCHMARK_SOURCE_PREFIX}{config.benchmark}": file_sha256(crows_pairs_path(data_dir)),
    }


def export_benchmark(
    result: ScreeningResult, config: BenchmarkConfig, out_dir: Path, *, n_rows: int
) -> dict[str, Path]:
    """Write every P2 artifact and return ``{name: path}``."""
    return {
        "items": _write(out_dir / ITEMS_FILENAME, render_items_jsonl(result.items)),
        "exclusions": _write(out_dir / EXCLUSIONS_FILENAME, render_exclusions_csv(result)),
        "screening": _write(
            out_dir / SCREENING_FILENAME, render_screening_report(result, config, n_rows=n_rows)
        ),
        "coverage": _write(out_dir / COVERAGE_FILENAME, render_coverage_report(result, config)),
    }
