"""Gate: every number the paper cites traces to a run, and cannot be silently replaced.

The guardrail is "never invent a statistic", and the mechanism is that the manuscript quotes keys
rather than digits. Two failures would defeat that quietly:

1. a key recorded twice with different values, where a re-run replaces a number the manuscript was
   already written around -- so a disagreeing duplicate RAISES rather than winning;
2. a number that claims a run whose manifest does not exist, which looks traceable and is not.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from margins.stats.numbers import (
    NO_RUN,
    PaperNumber,
    PaperNumbers,
    read_paper_numbers,
    write_paper_numbers,
)


def number(key: str = "power.min_cell_n", value: object = 135, **overrides: object) -> PaperNumber:
    data: dict[str, object] = {
        "key": key,
        "value": value,
        "unit": "renderings",
        "run_id": "stats-20260919T000000Z-abcdef123456",
        "manifest_path": "results/run_manifests/stats-20260919T000000Z-abcdef123456.json",
        "source": "scripts/run_power_analysis.py",
    }
    data.update(overrides)
    return PaperNumber.model_validate(data)


# ------------------------------------------------------------------ round trip


def test_round_trips_exactly(tmp_path: Path) -> None:
    original = PaperNumbers.of(
        [
            number(),
            number("power.icc_rho_regard", 0.35, unit="ratio", ci_low=0.29, ci_high=0.43),
            number("power.unreachable", None, unit="none"),
            number("power.note", "measured on one pair", unit="none"),
            number("power.ceiling_clears", True, unit="none"),
        ]
    )
    path = write_paper_numbers(original, tmp_path / "paper_numbers.json")
    assert read_paper_numbers(path) == original


def test_is_byte_deterministic(tmp_path: Path) -> None:
    """Committed under results/, so unlike Parquet this file IS byte-comparable."""
    numbers = PaperNumbers.of([number(), number("power.frames_per_cell", 27, unit="count")])
    first = write_paper_numbers(numbers, tmp_path / "a.json").read_bytes()
    second = write_paper_numbers(numbers, tmp_path / "b.json").read_bytes()
    assert first == second
    assert first.endswith(b"\n")


def test_missing_file_reads_as_empty(tmp_path: Path) -> None:
    assert read_paper_numbers(tmp_path / "absent.json").numbers == ()


def test_entries_are_stored_sorted(tmp_path: Path) -> None:
    numbers = PaperNumbers.of([number("z.last", 1), number("a.first", 2)])
    assert [n.key for n in numbers.numbers] == ["a.first", "z.last"]
    written = json.loads(write_paper_numbers(numbers, tmp_path / "n.json").read_text("utf-8"))
    assert [n["key"] for n in written["numbers"]] == ["a.first", "z.last"]


# ------------------------------------------------------------------ traceability


def test_every_number_names_a_run_and_its_manifest() -> None:
    with pytest.raises(ValidationError, match="no manifest path"):
        number(manifest_path=None)


def test_a_number_that_came_from_no_run_says_so() -> None:
    """An analytic identity is legitimate, but it must not pretend to a run."""
    analytic = number(run_id=NO_RUN, manifest_path=None, source="analytic identity")
    assert analytic.run_id == NO_RUN


def test_manifest_paths_resolve_to_real_manifests() -> None:
    """The committed file's entries must point at manifests that exist in this checkout."""
    repo_root = Path(__file__).resolve().parents[1]
    committed = read_paper_numbers(repo_root / "results" / "paper_numbers.json")
    for entry in committed.numbers:
        if entry.run_id == NO_RUN:
            continue
        assert entry.manifest_path is not None
        assert (repo_root / entry.manifest_path).is_file(), (
            f"{entry.key} cites {entry.manifest_path}, which is not in this checkout"
        )


# ------------------------------------------------------------------ conflicts


def test_a_duplicate_key_with_a_different_value_raises() -> None:
    base = PaperNumbers.of([number("power.min_cell_n", 135)])
    with pytest.raises(ValueError, match="already recorded"):
        base.merged([number("power.min_cell_n", 216)])


def test_re_registering_the_same_value_is_idempotent() -> None:
    base = PaperNumbers.of([number("power.min_cell_n", 135)])
    merged = base.merged([number("power.min_cell_n", 135)])
    assert len(merged.numbers) == 1


def test_duplicate_keys_in_one_collection_are_refused() -> None:
    with pytest.raises(ValidationError, match="duplicate keys"):
        PaperNumbers(numbers=(number("a.b", 1), number("a.b", 2)))


def test_unsorted_construction_is_refused() -> None:
    with pytest.raises(ValidationError, match="key order"):
        PaperNumbers(numbers=(number("z.z", 1), number("a.a", 2)))


# ------------------------------------------------------------------ lookup and keys


def test_lookup_names_near_misses() -> None:
    numbers = PaperNumbers.of([number("power.min_cell_n", 135), number("power.frames", 27)])
    assert numbers.lookup("power.min_cell_n").value == 135
    with pytest.raises(KeyError, match="did you mean"):
        numbers.lookup("power.min_cell")


@pytest.mark.parametrize("bad", ["Power.X", "power min_cell_n", "power", "", "power..x", "power.X"])
def test_malformed_keys_are_refused(bad: str) -> None:
    with pytest.raises(ValidationError):
        number(bad, 1)


def test_intervals_move_together() -> None:
    with pytest.raises(ValidationError, match="together"):
        number(ci_low=0.1)
    with pytest.raises(ValidationError, match="ci_low must be"):
        number(ci_low=0.9, ci_high=0.1)


def test_claimable_defaults_to_false() -> None:
    """A recorded number is not automatically a citable claim."""
    assert number().claimable is False
