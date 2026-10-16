"""``NumberSource``: the only way a figure is allowed to obtain a number (P10).

The guardrail is "never invent a statistic", and P9 already made the manuscript cite numbers by
key rather than by typing digits. A figure is the other half of that: a plotted point is a claim
in exactly the way a sentence is, and a hand-typed axis annotation is as much a fabrication as a
hand-typed p-value.

So figures here never see a float literal. They ask a :class:`NumberSource` for a paper number by
key or for a table column, and get :class:`Traced` values back -- each carrying the value, the key
it came from and the artifact it came out of. The source keeps a **ledger** of everything it hands
out, and :func:`margins.viz.trace.audit_figure` walks the rendered canvas and fails if any number
drawn on it is absent from that ledger.

That is a real check and not a ceremonial one: it catches a rounded constant typed into a label, a
threshold hard-coded into a reference line, and an axis annotation that drifted when the table
under it changed. ``tests/test_viz_traceability.py`` additionally proves the audit *can* fail, by
drawing an unbacked value and asserting it is reported.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from margins.report.tables import (
    DEFAULT_FIXTURE_DIR,
    DEFAULT_TABLE_DIR,
    DataStatus,
    ResultTable,
    read_table,
)
from margins.stats.numbers import PAPER_NUMBERS_PATH, PaperNumbers, read_paper_numbers

__all__ = ["MissingNumberError", "NumberSource", "Traced", "resolve_table_dir"]

#: The origin string a paper number carries in a ledger entry.
PAPER_ORIGIN: Final[str] = "results/paper_numbers.json"


class MissingNumberError(KeyError):
    """A figure asked for a number that no committed artifact supplies."""


@dataclass(frozen=True)
class Traced:
    """One number, and where it came from. The unit of the figure ledger."""

    value: float
    key: str
    origin: str

    def __float__(self) -> float:
        return self.value


def values(traced: Sequence[Traced | None]) -> list[float]:
    """Unwrap for plotting. ``None`` becomes NaN, which matplotlib skips and the audit ignores."""
    return [float("nan") if item is None else item.value for item in traced]


def resolve_table_dir(
    explicit: Path | None = None,
    *,
    real: Path = DEFAULT_TABLE_DIR,
    fixture: Path = DEFAULT_FIXTURE_DIR,
) -> Path:
    """Prefer real result tables; fall back to the fixture stand-ins while P11 has not run.

    The fallback is deliberate and loud rather than silent: the fixture rows carry
    ``data_status="fixture"``, every figure drawn from them is banner-stamped, and
    ``scripts/make_figures.py`` prints which directory it used.
    """
    if explicit is not None:
        return explicit
    if real.is_dir() and any(real.glob("*.csv")):
        return real
    return fixture


class NumberSource:
    """Hands out traced values from ``paper_numbers.json`` and the committed result tables."""

    def __init__(
        self,
        *,
        numbers: PaperNumbers | None = None,
        table_dir: Path | None = None,
        paper_numbers_path: Path = PAPER_NUMBERS_PATH,
    ) -> None:
        self._numbers = numbers if numbers is not None else read_paper_numbers(paper_numbers_path)
        self._paper_origin = paper_numbers_path.as_posix()
        self._table_dir = resolve_table_dir(table_dir)
        # Per-TABLE fallback, only when the directory was resolved by default. A real run can
        # produce most tables before all of them (E4's `sensitivity` is built after the lab
        # run), and one pending table must not stop the figures whose tables are real. The
        # fallback rows carry data_status="fixture", so `data_status` below turns that figure,
        # and only that figure, into a banner-stamped FIXTURE. An explicit --table-dir is strict.
        self._fallback_dir = (
            DEFAULT_FIXTURE_DIR
            if table_dir is None and self._table_dir != DEFAULT_FIXTURE_DIR
            else None
        )
        self._tables: dict[str, ResultTable] = {}
        self._ledger: list[Traced] = []

    # -- provenance ------------------------------------------------------------------

    @property
    def table_dir(self) -> Path:
        return self._table_dir

    @property
    def ledger(self) -> tuple[Traced, ...]:
        """Everything handed out so far, in order."""
        return tuple(self._ledger)

    @property
    def data_status(self) -> DataStatus:
        """``fixture`` if any table read so far is synthetic; ``real`` otherwise."""
        return (
            "fixture"
            if any(table.data_status == "fixture" for table in self._tables.values())
            else "real"
        )

    def _record(self, traced: Traced) -> Traced:
        self._ledger.append(traced)
        return traced

    # -- paper numbers ---------------------------------------------------------------

    def number(self, key: str) -> Traced:
        """One entry of ``paper_numbers.json``, by key.

        Raises:
            MissingNumberError: if the key is absent, or present but not numeric. A figure that
                needs a number the pipeline never emitted must fail, not fall back.
        """
        try:
            entry = self._numbers.lookup(key)
        except KeyError as exc:
            raise MissingNumberError(str(exc)) from exc
        if not isinstance(entry.value, int | float) or isinstance(entry.value, bool):
            raise MissingNumberError(
                f"paper number {key!r} is {entry.value!r}, which is not plottable"
            )
        return self._record(Traced(float(entry.value), key, self._paper_origin))

    def interval(self, key: str) -> tuple[Traced, Traced]:
        """The ``ci_low``/``ci_high`` of a paper number, traced separately."""
        entry = self._numbers.lookup(key)
        if entry.ci_low is None or entry.ci_high is None:
            raise MissingNumberError(f"paper number {key!r} carries no interval")
        return (
            self._record(Traced(entry.ci_low, f"{key}.ci_low", self._paper_origin)),
            self._record(Traced(entry.ci_high, f"{key}.ci_high", self._paper_origin)),
        )

    def has(self, key: str) -> bool:
        try:
            self._numbers.lookup(key)
        except KeyError:
            return False
        return True

    def text(self, key: str) -> str:
        """A non-numeric paper number, for a label. Not traced: a label is not a measurement."""
        return str(self._numbers.lookup(key).value)

    # -- result tables ---------------------------------------------------------------

    def table(self, name: str) -> ResultTable:
        """Read a committed result table once and remember it."""
        if name not in self._tables:
            try:
                self._tables[name] = read_table(name, directory=self._table_dir)
            except FileNotFoundError:
                if self._fallback_dir is None:
                    raise
                self._tables[name] = read_table(name, directory=self._fallback_dir)
        return self._tables[name]

    def floats(self, table: ResultTable, column: str) -> tuple[Traced, ...]:
        """A numeric column of a (possibly filtered) table, one traced value per row.

        Raises:
            MissingNumberError: on an empty cell. Use :meth:`optional_floats` where a null is a
                legal state -- an exploratory term's ``q``, say.
        """
        out: list[Traced] = []
        for traced in self.optional_floats(table, column):
            if traced is None:
                raise MissingNumberError(
                    f"{table.spec.name}.{column} has an empty cell; if that is legal here use "
                    "optional_floats and decide explicitly what the figure does with it"
                )
            out.append(traced)
        return tuple(out)

    def optional_floats(self, table: ResultTable, column: str) -> tuple[Traced | None, ...]:
        """As :meth:`floats`, but an empty cell yields ``None`` rather than raising."""
        spec = table.spec.column(column)
        if spec.kind not in ("float", "int"):
            raise TypeError(
                f"{table.spec.name}.{column} is a {spec.kind} column; only numbers are traced"
            )
        out: list[Traced | None] = []
        for row in table.rows:
            raw = row[column]
            if raw == "":
                out.append(None)
                continue
            key = f"{table.spec.name}[{table.row_key(row)}].{column}"
            out.append(self._record(Traced(float(raw), key, table.origin)))
        return tuple(out)

    def derived(self, value: float, *, key: str, origin: str, how: str) -> Traced:
        """Register a number computed from traced inputs -- a count of rows, a column mean.

        A derivation is not a fabrication, but it is also not self-evident, so ``how`` is
        required and lands in the ledger key. This is the only way a number that appears in no
        artifact may be drawn, and it exists so that "16 of 42 family terms survived correction"
        can be plotted without either inventing it or adding a table for every arithmetic step.
        """
        return self._record(Traced(float(value), f"{key} ({how})", origin))

    def cell(self, table: ResultTable, row: Mapping[str, str], column: str) -> Traced:
        """One numeric cell of one row."""
        table.spec.column(column)
        raw = row[column]
        if raw == "":
            raise MissingNumberError(f"{table.spec.name}.{column} is empty for this row")
        key = f"{table.spec.name}[{table.row_key(row)}].{column}"
        return self._record(Traced(float(raw), key, table.origin))
