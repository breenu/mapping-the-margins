"""Binding figures to the claim gate: no star a refusal would have withheld (P10).

Invariant 3 says only ``StatsEngine`` sets significance, and :mod:`margins.stats.claims` makes
the refusal mechanical -- ``ClaimGate.is_significant`` *raises* for an effect that may not carry a
claim, so a table builder that forgets to ask crashes rather than quietly printing a star. A
figure is a table builder. This module is how a figure asks.

**It runs the real gate, and then checks the gate against P9's own record.** Reading the ``status``
column out of the committed table and trusting it would be the easy version and a circular one: if
the figure believed the artifact, a figure and a stats run could drift apart with nothing
detecting it. Instead :func:`significance` rebuilds an :class:`~margins.schemas.EffectEstimate`
from each row, constructs the gate from the *pre-registered* inputs (the lattice's exploratory
pairs and the committed descriptive-only decisions), runs it, and **raises** if its verdict
disagrees with the one P9 wrote down.

What comes back is a marker per row. Only ``claimable`` rows with ``q <= alpha`` get one. Every
other row is drawn, with its estimate and its interval, and with its refusal reason available for
the caption -- because a refusal is a row and never an absence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from margins.report.tables import ResultTable
from margins.schemas import EffectEstimate
from margins.stats.claims import ClaimGate
from margins.stats.records import ClaimStatus

__all__ = [
    "SIGNIFICANT_MARKER",
    "RowClaim",
    "descriptive_only_keys_from_table",
    "effect_from_row",
    "exploratory_pairs_from_lattice",
    "significance",
]

#: The only mark a figure may use for "survived the correction". One glyph, one meaning.
SIGNIFICANT_MARKER: Final[str] = "*"


class ClaimMismatchError(ValueError):
    """The gate's verdict at figure time disagrees with the one the stats run recorded."""


@dataclass(frozen=True)
class RowClaim:
    """What a figure may say about one row."""

    status: ClaimStatus
    claimable: bool
    significant: bool
    refusal_reason: str | None

    @property
    def marker(self) -> str:
        """``"*"`` only for a claim the gate permits and the correction survived."""
        return SIGNIFICANT_MARKER if self.significant else ""

    @property
    def load_bearing(self) -> bool:
        """Whether this row may be drawn as evidence rather than as description."""
        return self.claimable


def _optional_float(raw: str) -> float | None:
    return None if raw == "" else float(raw)


def effect_from_row(row: Mapping[str, str]) -> EffectEstimate:
    """**Deserialize** P9's estimate out of a committed ``emergence_effects`` row.

    Nothing here *sets* significance. ``q``, ``family_size`` and ``correction`` are read off the
    row exactly as the stats run wrote them -- the same read-back that
    :func:`margins.stats.export.read_corrected_effects` performs across the Parquet boundary,
    and the reason those three are columns of the table rather than something a figure could
    reconstruct. An earlier version of this function passed ``family_size=1`` so the record would
    validate, which is precisely the invariant-3 violation the AST scan in
    ``tests/test_fdr_correction.py`` exists to catch, and it caught it.
    """
    return EffectEstimate.model_validate(
        {
            "benchmark": row["benchmark"],
            "model": row["model_id"],
            "axis_pair": (row["axis_a"], row["axis_b"]),
            "term": row["term"],
            "harm_type": row["harm_type"],
            "estimate": float(row["estimate"]),
            "se": float(row["se"]),
            "ci_low": float(row["ci_low"]),
            "ci_high": float(row["ci_high"]),
            "p": _optional_float(row["p"]),
            "q": _optional_float(row["q"]),
            "family_size": None if row["family_size"] == "" else int(row["family_size"]),
            "correction": row["correction"] or None,
            "powered": row["powered"] == "true",
        }
    )


def descriptive_only_keys_from_table(table: ResultTable) -> frozenset[tuple[str, str, str]]:
    """``(benchmark, model, harm_type)`` triples the drift rule downgraded, from the table."""
    return frozenset(
        (row["benchmark"], row["model_id"], row["harm_type"])
        for row in table.rows
        if row["descriptive_only"] == "true"
    )


def exploratory_pairs_from_lattice(name: str | None = None) -> frozenset[tuple[str, str]]:
    """The pairs outside the pre-registered confirmatory set, read from the lattice config.

    The lattice is where the confirmatory set is *declared*, and the declaration is what the gate
    is entitled to consult. Deriving it from the artifact instead would let a figure inherit a
    mistake the artifact made.
    """
    from margins.lattice.config import DEFAULT_LATTICE, load_lattice_config

    config = load_lattice_config(name or DEFAULT_LATTICE)
    return frozenset(config.exploratory_pairs)


def significance(
    rows: Sequence[Mapping[str, str]],
    *,
    alpha: float,
    exploratory_pairs: frozenset[tuple[str, str]],
    descriptive_only: frozenset[tuple[str, str, str]],
) -> tuple[RowClaim, ...]:
    """Run the real gate over the rows and cross-check it against their recorded status.

    Raises:
        ClaimMismatchError: if the gate's verdict differs from the one in the table. That is
            either a stale table or a figure built against a different pre-registration, and both
            are reasons to stop rather than to draw.
    """
    gate = ClaimGate(
        exploratory_pairs=exploratory_pairs,
        descriptive_only=descriptive_only,
        alpha=alpha,
        powered_only=True,
    )
    effects = [effect_from_row(row) for row in rows]
    claims = gate.claims(effects)

    out: list[RowClaim] = []
    for row, effect, claim in zip(rows, effects, claims, strict=True):
        recorded = row["status"]
        if recorded and recorded != claim.status:
            raise ClaimMismatchError(
                f"{row['harm_type']} {row['axis_a']}x{row['axis_b']} {row['term']!r}: the stats "
                f"run recorded status {recorded!r} and the gate says {claim.status!r} at figure "
                "time. Rebuild the result tables; a figure must not outvote P9."
            )
        # Short-circuit deliberately: `is_significant` RAISES for a refused effect, and that
        # refusal is the point. The figure asks only once the gate has already permitted a claim.
        significant = claim.claimable and gate.is_significant(effect)
        out.append(
            RowClaim(
                status=claim.status,
                claimable=claim.claimable,
                significant=significant,
                refusal_reason=claim.refusal_reason,
            )
        )
    return tuple(out)
