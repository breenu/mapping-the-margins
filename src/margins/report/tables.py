"""The committed result tables: their schemas, their CSV form and their LaTeX form (P10).

A table here is declared, not inferred. :class:`TableSpec` names the columns, their types, the
columns that identify a row, and the caption -- and both writers and the reader go through it. The
cost is a little ceremony; the benefit is that a figure can address a cell by
``table[key].column`` and get a value whose provenance is a string a reader can look up, and that
a column silently renamed upstream fails loudly here rather than silently producing an empty
figure.

**Why tables are committed and Parquet is not.** ``data/`` is gitignored and regenerable
(PROJECT_CONTEXT.md section 4), which is right for score dumps and wrong for the handful of small
frames the paper is written against. These are the paper's inputs; they live in ``results/`` beside
``paper_numbers.json`` and the run manifests.

**``data_status`` is a required column on every table**, not a filename convention. Before the P11
confirmatory run there are no real E1 numbers, so P10 ships synthetic rows so that every figure
renders and is tested now. A reader who opens a CSV must be able to see that in the row, not infer
it from which directory it came out of, and a figure built on such rows carries a visible banner
that :func:`margins.viz.trace.finish_figure` puts there from this same column.

Both writers are byte-deterministic: LF endings, a fixed float format, rows sorted by the declared
key columns.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

__all__ = [
    "DATA_STATUS_COLUMN",
    "DEFAULT_FIXTURE_DIR",
    "DEFAULT_TABLE_DIR",
    "TABLE_SPECS",
    "Column",
    "DataStatus",
    "ResultTable",
    "TableSpec",
    "read_table",
    "spec_for",
    "write_table_csv",
    "write_table_latex",
]

#: Real rows come from a pipeline run; fixture rows are synthetic and carry no claim at all.
DataStatus = Literal["real", "fixture"]
DATA_STATUS_COLUMN: Final[str] = "data_status"

#: Tables from a real run. Written by ``scripts/build_result_tables.py``.
DEFAULT_TABLE_DIR: Final[Path] = Path("results") / "tables"
#: Synthetic stand-ins so every figure renders before P11. Written by
#: ``scripts/build_fixture_results.py`` and never mixed with the above.
DEFAULT_FIXTURE_DIR: Final[Path] = Path("results") / "fixtures"

ColumnKind = Literal["str", "float", "int", "bool"]


@dataclass(frozen=True)
class Column:
    """One column: its name, its type, and how many digits it shows in LaTeX."""

    name: str
    kind: ColumnKind = "str"
    digits: int = 3
    label: str | None = None
    #: LaTeX column alignment. Numbers right-align; text left-aligns.
    align: str | None = None

    @property
    def heading(self) -> str:
        return self.label or self.name.replace("_", " ")

    @property
    def alignment(self) -> str:
        if self.align is not None:
            return self.align
        return "r" if self.kind in ("float", "int") else "l"


@dataclass(frozen=True)
class TableSpec:
    """A declared result table."""

    name: str
    title: str
    caption: str
    columns: tuple[Column, ...]
    #: The columns whose values identify a row. Used for the canonical sort AND for the
    #: provenance string a figure quotes when it reads a cell.
    key_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        names = [column.name for column in self.columns]
        if len(names) != len(set(names)):
            raise ValueError(f"table {self.name!r} has duplicate columns")
        if DATA_STATUS_COLUMN not in names:
            raise ValueError(
                f"table {self.name!r} must carry a {DATA_STATUS_COLUMN!r} column: a reader has to "
                "see from the row itself whether it is a real result or a fixture stand-in"
            )
        missing = [key for key in self.key_columns if key not in names]
        if missing:
            raise ValueError(f"table {self.name!r} key columns {missing} are not columns")

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    def column(self, name: str) -> Column:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(f"table {self.name!r} has no column {name!r}; has {self.column_names}")


# --------------------------------------------------------------------------------------
# the declared set
# --------------------------------------------------------------------------------------

_STATUS_COLUMN = Column(DATA_STATUS_COLUMN, "str", label="data status")

EMERGENCE_EFFECTS = TableSpec(
    name="emergence_effects",
    title="Emergent intersectional bias: the interaction terms",
    caption=(
        "Every term of the fitted $harm \\sim A + B + A{:}B$ model per confirmatory axis pair, "
        "with its cluster-bootstrap interval and its claim status. The interaction terms are the "
        "estimand; the main effects are B1's single-axis baseline and are carried so that a "
        "reader can see a cell's level is not its interaction. A refused row keeps its estimate "
        "and its interval and carries no significance."
    ),
    columns=(
        Column("benchmark"),
        Column("model_id", label="model"),
        Column("harm_type", label="harm type"),
        Column("axis_a", label="axis A"),
        Column("axis_b", label="axis B"),
        Column("term"),
        Column("kind"),
        Column("value_a", label="value A"),
        Column("value_b", label="value B"),
        Column("estimate", "float", digits=3),
        Column("se", "float", digits=3),
        Column("ci_low", "float", digits=3, label="CI low"),
        Column("ci_high", "float", digits=3, label="CI high"),
        Column("p", "float", digits=4),
        Column("q", "float", digits=4),
        # B3 (P11): the permutation p, reported BESIDE the GLM p and never in the FDR family.
        Column("p_permutation", "float", digits=4, label="p (perm.)"),
        # Carried, not derived. A figure that needs to hand an estimate back to the claim gate
        # must be able to rebuild the record P9 wrote, and inventing these two would be the
        # figure layer setting significance (invariant 3).
        Column("family_size", "int", label="family size"),
        Column("correction"),
        Column("powered", "bool"),
        Column("status"),
        Column("claimable", "bool"),
        Column("family_member", "bool", label="in FDR family"),
        _STATUS_COLUMN,
    ),
    key_columns=("benchmark", "model_id", "harm_type", "axis_a", "axis_b", "term"),
)

DESCRIPTIVE_ONLY = TableSpec(
    name="descriptive_only",
    title="The scorer identity-drift downgrade",
    caption=(
        "Section 9's rule as enforced at P9. A harm type whose scorers' own interaction on "
        "known-neutral text reaches the pre-registered fraction of the model's cannot distinguish "
        "emergence in the model from emergence in the instrument, and loses every significance "
        "claim while keeping every number."
    ),
    columns=(
        Column("benchmark"),
        Column("model_id", label="model"),
        Column("harm_type", label="harm type"),
        Column("descriptive_only", "bool", label="descriptive-only"),
        Column("threshold", "float", digits=2),
        Column("max_ratio", "float", digits=2, label="max drift ratio"),
        Column("worst_scorer", label="worst scorer"),
        Column("n_comparisons", "int", label="comparisons"),
        Column("n_over_threshold", "int", label="over threshold"),
        _STATUS_COLUMN,
    ),
    key_columns=("benchmark", "model_id", "harm_type"),
)

DRIFT_SIDE_BY_SIDE = TableSpec(
    name="drift_side_by_side",
    title="The model's interaction beside the scorer's own",
    caption=(
        "Root-mean-square interaction magnitude, in SD units, for each scorer and axis pair: the "
        "model's output as that scorer sees it, and the same scorer's response to known-neutral "
        "text carrying the same descriptors. \\textbf{Nothing is subtracted.}"
    ),
    columns=(
        Column("benchmark"),
        Column("model_id", label="model"),
        Column("harm_type", label="harm type"),
        Column("scorer_id", label="scorer"),
        Column("axis_a", label="axis A"),
        Column("axis_b", label="axis B"),
        Column("control_family", label="control family"),
        Column("model_rms", "float", digits=3, label="model RMS"),
        Column("drift_rms", "float", digits=3, label="drift RMS"),
        Column("ratio", "float", digits=2),
        Column("n_model_obs", "int", label="n model"),
        Column("n_drift_obs", "int", label="n drift"),
        _STATUS_COLUMN,
    ),
    key_columns=(
        "benchmark",
        "model_id",
        "harm_type",
        "scorer_id",
        "axis_a",
        "axis_b",
        "control_family",
    ),
)

FIT_REPORTS = TableSpec(
    name="fit_reports",
    title="What was estimable, and what was not",
    caption=(
        "One row per series and axis pair. A pair that could not be fitted is reported with its "
        "reason rather than dropped or given a zero -- which is how the extractive paradigm's "
        "empty interaction slot is recorded."
    ),
    columns=(
        Column("benchmark"),
        Column("model_id", label="model"),
        Column("harm_type", label="harm type"),
        Column("axis_a", label="axis A"),
        Column("axis_b", label="axis B"),
        Column("estimable", "bool"),
        Column("reason"),
        Column("n_obs", "int", label="n obs"),
        Column("n_items", "int", label="n items"),
        Column("n_cells", "int", label="n cells"),
        Column("rms_interaction", "float", digits=3, label="RMS interaction"),
        Column("powered", "bool"),
        _STATUS_COLUMN,
    ),
    key_columns=("benchmark", "model_id", "harm_type", "axis_a", "axis_b"),
)

SINGLE_AXIS_CELLS = TableSpec(
    name="single_axis_cells",
    title="Single-axis cell means, per paradigm",
    caption=(
        "Descriptive order-1 cell means with bootstrap intervals. This is the level at which the "
        "generation and extraction paradigms are comparable: the extractive source attests three "
        "order-2 items in total and cannot populate any pair's grid, so no interaction estimate "
        "exists for it to be compared on."
    ),
    columns=(
        Column("benchmark"),
        Column("model_id", label="model"),
        Column("harm_type", label="harm type"),
        Column("cell_id", label="cell"),
        Column("axis"),
        Column("value"),
        Column("n", "int"),
        Column("mean_standardized", "float", digits=3, label="mean (SD)"),
        Column("ci_low", "float", digits=3, label="CI low"),
        Column("ci_high", "float", digits=3, label="CI high"),
        Column("powered", "bool"),
        _STATUS_COLUMN,
    ),
    key_columns=("benchmark", "model_id", "harm_type", "cell_id"),
)

SENSITIVITY = TableSpec(
    name="sensitivity",
    title="E4: how the headline interaction moves with decoding",
    caption=(
        "The same interaction term re-estimated across the sensitivity subset's temperature grid "
        "and seed groups. RQ4 is answered by the spread, not by any single point."
    ),
    columns=(
        Column("benchmark"),
        Column("model_id", label="model"),
        Column("harm_type", label="harm type"),
        Column("axis_a", label="axis A"),
        Column("axis_b", label="axis B"),
        Column("term"),
        Column("temperature", "float", digits=1),
        Column("seed_group", "int", label="seed group"),
        Column("estimate", "float", digits=3),
        Column("ci_low", "float", digits=3, label="CI low"),
        Column("ci_high", "float", digits=3, label="CI high"),
        _STATUS_COLUMN,
    ),
    key_columns=(
        "benchmark",
        "model_id",
        "harm_type",
        "axis_a",
        "axis_b",
        "term",
        "temperature",
        "seed_group",
    ),
)

FRAME_BREAKDOWN = TableSpec(
    name="frame_breakdown",
    title="Per-frame harm, with the highest-risk frame named",
    caption=(
        "Mean standardized harm by template frame. The frames are an item cluster, so this is "
        "also the visible form of the quantity that bounds precision: a design with one frame has "
        "no between-frame variance and cannot estimate a cluster-robust standard error at all."
    ),
    columns=(
        Column("benchmark"),
        Column("model_id", label="model"),
        Column("harm_type", label="harm type"),
        Column("frame"),
        Column("n_items", "int", label="n items"),
        Column("mean_standardized", "float", digits=3, label="mean (SD)"),
        Column("ci_low", "float", digits=3, label="CI low"),
        Column("ci_high", "float", digits=3, label="CI high"),
        _STATUS_COLUMN,
    ),
    key_columns=("benchmark", "model_id", "harm_type", "frame"),
)

TABLE_SPECS: Final[tuple[TableSpec, ...]] = (
    DESCRIPTIVE_ONLY,
    DRIFT_SIDE_BY_SIDE,
    EMERGENCE_EFFECTS,
    FIT_REPORTS,
    FRAME_BREAKDOWN,
    SENSITIVITY,
    SINGLE_AXIS_CELLS,
)


def spec_for(name: str) -> TableSpec:
    for spec in TABLE_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(f"no result table named {name!r}; known: {[s.name for s in TABLE_SPECS]}")


# --------------------------------------------------------------------------------------
# the table itself
# --------------------------------------------------------------------------------------

#: ``%.10g`` round-trips to the same float through ``float()`` for every value this project
#: produces and keeps a committed CSV readable. The TABLE is the authority for what a figure
#: plots, so what matters is that reading it back gives exactly the number the figure drew.
_FLOAT_FORMAT: Final[str] = "{:.10g}"

_MISSING: Final[str] = ""


def _render(value: object, column: Column) -> str:
    """Format one cell.

    Accepts both record values and already-rendered strings, because a table read back off disk
    must be writable again unchanged -- that round trip is what makes the committed CSV, and not
    the Parquet under it, the authority for what a figure draws. The string branches are not
    defensive padding: ``bool("false")`` is ``True``, so writing a re-read boolean naively would
    silently flip every false cell.
    """
    if value is None or value == _MISSING:
        return _MISSING
    if column.kind == "bool":
        if isinstance(value, str):
            if value not in ("true", "false"):
                raise ValueError(f"{column.name!r} is a bool column; got {value!r}")
            return value
        return "true" if bool(value) else "false"
    if column.kind == "float":
        return _FLOAT_FORMAT.format(float(value))  # type: ignore[arg-type]
    if column.kind == "int":
        return str(int(value))  # type: ignore[call-overload]
    text = str(value)
    if any(character in text for character in ',"\n'):
        return '"' + text.replace('"', '""') + '"'
    return text


@dataclass(frozen=True)
class ResultTable:
    """A committed result table, in memory. Rows are plain string cells, as on disk."""

    spec: TableSpec
    rows: tuple[Mapping[str, str], ...]
    origin: str

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[Mapping[str, str]]:
        return iter(self.rows)

    @property
    def data_status(self) -> DataStatus:
        """``fixture`` if any row is synthetic. Mixed rows are a build error, not a status."""
        statuses = {row[DATA_STATUS_COLUMN] for row in self.rows}
        if not statuses or statuses == {"real"}:
            return "real"
        if statuses == {"fixture"}:
            return "fixture"
        raise ValueError(
            f"table {self.spec.name!r} mixes data_status values {sorted(statuses)}. Real results "
            "and fixture stand-ins never share a table: a figure drawn from both could not say "
            "which of its points was measured."
        )

    def row_key(self, row: Mapping[str, str]) -> str:
        return "|".join(row[key] for key in self.spec.key_columns)

    def where(self, **equals: str | Iterable[str]) -> ResultTable:
        """Rows whose columns equal the given value, or are in the given set. Order preserved."""
        for column in equals:
            self.spec.column(column)

        def keep(row: Mapping[str, str]) -> bool:
            for column, wanted in equals.items():
                if isinstance(wanted, str):
                    if row[column] != wanted:
                        return False
                elif row[column] not in set(wanted):
                    return False
            return True

        return ResultTable(self.spec, tuple(row for row in self.rows if keep(row)), self.origin)

    def sorted_by(self, *columns: str, key: object = None) -> ResultTable:
        """Rows in a declared order. ``key`` may map a row to a sort tuple instead."""
        for column in columns:
            self.spec.column(column)
        if key is not None:
            if not callable(key):
                raise TypeError("key must be callable")
            ordered = sorted(self.rows, key=key)
        else:
            ordered = sorted(self.rows, key=lambda row: tuple(row[c] for c in columns))
        return ResultTable(self.spec, tuple(ordered), self.origin)

    def strings(self, column: str) -> tuple[str, ...]:
        """A text column. Labels need no tracing; only numbers do."""
        self.spec.column(column)
        return tuple(row[column] for row in self.rows)

    def unique(self, column: str) -> tuple[str, ...]:
        """Distinct values of a text column, in first-appearance order."""
        seen: list[str] = []
        for value in self.strings(column):
            if value not in seen:
                seen.append(value)
        return tuple(seen)


# --------------------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------------------


def _split_csv_line(line: str) -> list[str]:
    out: list[str] = []
    field: list[str] = []
    quoted = False
    index = 0
    while index < len(line):
        character = line[index]
        if quoted:
            if character == '"':
                if index + 1 < len(line) and line[index + 1] == '"':
                    field.append('"')
                    index += 1
                else:
                    quoted = False
            else:
                field.append(character)
        elif character == '"':
            quoted = True
        elif character == ",":
            out.append("".join(field))
            field = []
        else:
            field.append(character)
        index += 1
    out.append("".join(field))
    return out


def write_table_csv(spec: TableSpec, rows: Sequence[Mapping[str, object]], path: Path) -> Path:
    """Write the CSV form. Byte-deterministic: declared column order, key sort, LF, fixed floats.

    Raises:
        ValueError: if a row carries a column the spec does not declare. A silently dropped
            column would make the committed table disagree with the figure that reads it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    names = spec.column_names
    for row in rows:
        extra = sorted(set(row) - set(names))
        if extra:
            raise ValueError(f"table {spec.name!r} got undeclared columns {extra}")

    def sort_key(row: Mapping[str, object]) -> tuple[str, ...]:
        return tuple(_render(row.get(k), spec.column(k)) for k in spec.key_columns)

    ordered = sorted(rows, key=sort_key)
    lines = [",".join(names)]
    for row in ordered:
        lines.append(",".join(_render(row.get(name), spec.column(name)) for name in names))
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def read_table(name: str, *, directory: Path) -> ResultTable:
    """Read ``<directory>/<name>.csv`` against its declared spec.

    Raises:
        FileNotFoundError: if the table has not been built.
        ValueError: if the header does not match the spec exactly.
    """
    spec = spec_for(name)
    path = directory / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"result table {path} does not exist. Build it with "
            "`uv run python scripts/build_result_tables.py` (a real run) or "
            "`uv run python scripts/build_fixture_results.py` (the pre-P11 stand-ins)."
        )
    text = path.read_text(encoding="utf-8")
    lines = [line for line in text.split("\n") if line != ""]
    header = tuple(_split_csv_line(lines[0]))
    if header != spec.column_names:
        raise ValueError(
            f"{path} has columns {header} but table {name!r} declares {spec.column_names}. "
            "A result table and its spec are one artifact; rebuild the table."
        )
    rows = tuple(dict(zip(header, _split_csv_line(line), strict=True)) for line in lines[1:])
    return ResultTable(spec, rows, origin=path.as_posix())


def _latex_escape(text: str) -> str:
    for character, replacement in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ):
        text = text.replace(character, replacement)
    return text


def write_table_latex(
    table: ResultTable, path: Path, *, columns: Sequence[str] | None = None
) -> Path:
    """Write a booktabs ``table`` environment. Generated; never hand-edited; deterministic.

    ``columns`` narrows the LaTeX form to the columns worth printing; the CSV always carries all
    of them. The caption states the data status, so a fixture table cannot be pasted into a paper
    without the reader being told.
    """
    spec = table.spec
    chosen = tuple(columns) if columns is not None else spec.column_names
    for name in chosen:
        spec.column(name)
    alignment = "".join(spec.column(name).alignment for name in chosen)

    status_note = (
        ""
        if table.data_status == "real"
        else (
            " \\textbf{These rows are synthetic fixtures}, not measured results: the "
            "confirmatory run is P11. The figure and table code is unchanged by the swap."
        )
    )
    lines = [
        "% Generated by scripts/build_result_tables.py. Do not edit by hand.",
        f"% source: {table.origin}",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\small",
        f"  \\caption{{{spec.title}. {spec.caption}{status_note}}}",
        f"  \\label{{tab:{spec.name}}}",
        f"  \\begin{{tabular}}{{{alignment}}}",
        "    \\toprule",
        "    " + " & ".join(_latex_escape(spec.column(n).heading) for n in chosen) + " \\\\",
        "    \\midrule",
    ]
    for row in table.rows:
        cells: list[str] = []
        for name in chosen:
            column = spec.column(name)
            raw = row[name]
            if raw == _MISSING:
                cells.append("---")
            elif column.kind == "float":
                cells.append(f"{float(raw):.{column.digits}f}")
            else:
                cells.append(_latex_escape(raw))
        lines.append("    " + " & ".join(cells) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(lines) + "\n")
    return path
