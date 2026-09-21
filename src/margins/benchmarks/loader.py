"""The ``BenchmarkLoader`` for CrowS-Pairs: read the pinned file, screen it, return Items (P2).

Thin on purpose. The three things that could go wrong live elsewhere and are tested there: the pin
(:mod:`margins.benchmarks.crows_pairs`), the attestation rule
(:mod:`margins.benchmarks.attest`), and the screen (:mod:`margins.benchmarks.screening`). What this
module adds is the ``BenchmarkLoader`` contract -- ``load()`` returns the **screened** items, and
is deterministic: same source revision, same output.
"""

from __future__ import annotations

from pathlib import Path

from margins.benchmarks.config import BenchmarkConfig
from margins.benchmarks.crows_pairs import source_rows
from margins.benchmarks.screening import ScreeningResult, screen
from margins.lattice import DeclaredLattice
from margins.schemas import Item

__all__ = ["CrowsPairsLoader"]


class CrowsPairsLoader:
    """Ingests CrowS-Pairs into screened :class:`~margins.schemas.Item` records.

    Satisfies :class:`margins.interfaces.BenchmarkLoader`. The lattice is a constructor argument
    rather than something the loader reaches for, because screening *derives* a cell from each item
    and must do so against the same declared lattice the rest of the run uses -- a loader that
    built its own would be able to attest a cell no other stage recognises.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        lattice: DeclaredLattice,
        *,
        data_dir: Path | None = None,
    ) -> None:
        self._config = config
        self._lattice = lattice
        self._data_dir = data_dir

    @property
    def name(self) -> str:
        """Source label, matching ``Item.benchmark``."""
        return self._config.benchmark

    @property
    def config(self) -> BenchmarkConfig:
        return self._config

    def screen(self) -> ScreeningResult:
        """The full screening result: items, the exclusion log, and the coverage table.

        ``load()`` returns only the items, because that is the Protocol's shape; every caller that
        needs to *report* what screening did -- which is gate P2's actual requirement -- calls this
        instead.
        """
        rows = source_rows(self._config, data_dir=self._data_dir)
        return screen(rows, config=self._config, lattice=self._lattice)

    def load(self) -> list[Item]:
        """Return the screened items. Deterministic: same source revision, same output."""
        return list(self.screen().items)
