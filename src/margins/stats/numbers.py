"""``results/paper_numbers.json``: every number destined for the manuscript, keyed (P9).

The guardrail this implements is blunt: *never invent a statistic*, enforced by making the
manuscript reference numbers **by key** rather than by typing them. A figure that appears in the
paper and not in this file is a figure nobody can trace; a figure that appears in both and
disagrees is a diff.

Three properties the format buys, each a deliberate cost:

* **Every entry names its run.** ``run_id`` and ``manifest_path`` are required, so "where did this
  number come from" is a field rather than an archaeology exercise. A number produced outside any
  run -- an analytic identity, say -- says so in ``source`` and carries the sentinel run id.
* **Duplicate keys with different values RAISE.** Last-write-wins would let a re-run silently
  replace a number the manuscript had already been written around. Re-registering the same key
  with the same value is fine and idempotent.
* **Byte-deterministic.** Unlike the Parquet artifacts, this file is committed under
  ``results/``, so it is written with sorted keys and a fixed indent and two writes of the same
  content produce identical bytes.

``claimable`` exists because P9 refuses significance claims for exploratory pairs, underpowered
cells and descriptive-only harm types. A number may be perfectly well estimated and still not be
citable as evidence, and the file records which -- rather than omitting the refused ones, which
would make a refusal indistinguishable from never having looked.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "NO_RUN",
    "PAPER_NUMBERS_PATH",
    "PAPER_NUMBERS_SCHEMA_VERSION",
    "PaperNumber",
    "PaperNumbers",
    "read_paper_numbers",
    "write_paper_numbers",
]

PAPER_NUMBERS_PATH: Final[Path] = Path("results") / "paper_numbers.json"
PAPER_NUMBERS_SCHEMA_VERSION: Final[str] = "1"

#: For a number that is an analytic identity or a design constant rather than a run's output.
#: Spelled out rather than left blank, so "no run" is a statement and not a missing field.
NO_RUN: Final[str] = "not-from-a-run"

#: Dotted, lowercase, no spaces. A key is quoted verbatim in the manuscript source, so a loose
#: key is a typo waiting to silently resolve to nothing.
_SEGMENT = r"[a-z0-9]+(?:_[a-z0-9]+)*"
_KEY_RE: Final[re.Pattern[str]] = re.compile(rf"^{_SEGMENT}(?:\.{_SEGMENT})+$")

Unit = Literal["sd", "count", "probability", "ratio", "renderings", "gpu_hours", "none"]


class PaperNumber(BaseModel):
    """One number the manuscript may cite, with where it came from and whether it may be cited."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(min_length=3)
    value: float | int | str | bool | None
    unit: Unit = "none"
    run_id: str = Field(min_length=1)
    manifest_path: str | None = Field(
        default=None, description="None only when run_id is the no-run sentinel."
    )
    source: str = Field(min_length=1, description="Which stage or script produced it.")
    ci_low: float | None = None
    ci_high: float | None = None
    claimable: bool = Field(
        default=False,
        description=(
            "Whether this number may be cited as an inferential claim. Descriptive numbers, "
            "exploratory pairs and refused harm types are False and are still recorded."
        ),
    )
    note: str | None = None

    @field_validator("key")
    @classmethod
    def _well_formed_key(cls, value: str) -> str:
        if not _KEY_RE.match(value):
            raise ValueError(
                f"{value!r} is not a well-formed key. Use dotted lowercase segments, e.g. "
                "'power.min_cell_n' -- the manuscript quotes these verbatim."
            )
        return value

    @model_validator(mode="after")
    def _consistent(self) -> PaperNumber:
        if (self.ci_low is None) != (self.ci_high is None):
            raise ValueError("ci_low and ci_high are set together or not at all")
        if self.ci_low is not None and self.ci_high is not None and self.ci_low > self.ci_high:
            raise ValueError("ci_low must be <= ci_high")
        if self.run_id != NO_RUN and self.manifest_path is None:
            raise ValueError(
                f"{self.key!r} names run {self.run_id!r} but no manifest path. A number that "
                "claims a run must say where that run's manifest is."
            )
        return self


class PaperNumbers(BaseModel):
    """The whole file. Keys are unique and the entries are stored sorted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = PAPER_NUMBERS_SCHEMA_VERSION
    numbers: tuple[PaperNumber, ...] = ()

    @model_validator(mode="after")
    def _unique_and_sorted(self) -> PaperNumbers:
        keys = [number.key for number in self.numbers]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError(f"duplicate keys: {duplicates}")
        if keys != sorted(keys):
            raise ValueError("numbers must be stored in key order; use PaperNumbers.of()")
        return self

    @classmethod
    def of(
        cls, numbers: Sequence[PaperNumber], *, schema_version: str | None = None
    ) -> PaperNumbers:
        """Build a sorted, validated collection from any order of entries."""
        return cls(
            schema_version=schema_version or PAPER_NUMBERS_SCHEMA_VERSION,
            numbers=tuple(sorted(numbers, key=lambda number: number.key)),
        )

    def lookup(self, key: str) -> PaperNumber:
        for number in self.numbers:
            if number.key == key:
                return number
        near = [n.key for n in self.numbers if key.split(".")[0] == n.key.split(".")[0]]
        hint = f"; did you mean one of {near[:5]}?" if near else ""
        raise KeyError(f"no paper number {key!r}{hint}")

    def merged(self, others: Sequence[PaperNumber]) -> PaperNumbers:
        """Add entries, refusing a duplicate key whose value disagrees.

        Re-registering the same key with the same value is idempotent, which is what makes a
        re-run of an unchanged stage safe. A disagreement is a real conflict and raises, because
        last-write-wins would quietly replace a number the manuscript was written around.
        """
        merged = {number.key: number for number in self.numbers}
        for number in others:
            existing = merged.get(number.key)
            if existing is not None and existing.value != number.value:
                raise ValueError(
                    f"paper number {number.key!r} already recorded as {existing.value!r} "
                    f"(run {existing.run_id}) and would be replaced with {number.value!r} "
                    f"(run {number.run_id}). Refusing: the manuscript may already cite the "
                    "first. Bump the key or remove the stale file deliberately."
                )
            merged[number.key] = number
        return PaperNumbers.of(list(merged.values()), schema_version=self.schema_version)


def write_paper_numbers(numbers: PaperNumbers, path: Path = PAPER_NUMBERS_PATH) -> Path:
    """Write the file. Byte-deterministic: sorted keys, fixed indent, LF, trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        numbers.model_dump(mode="json"), indent=2, sort_keys=True, ensure_ascii=False
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(payload + "\n")
    return path


def read_paper_numbers(path: Path = PAPER_NUMBERS_PATH) -> PaperNumbers:
    """Read the file back, or an empty collection if it does not exist yet."""
    if not path.is_file():
        return PaperNumbers()
    return PaperNumbers.model_validate_json(path.read_text(encoding="utf-8"))
