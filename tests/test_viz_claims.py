"""Gate P10: a figure never displays a claim the gate would refuse.

Invariant 3 says only ``StatsEngine`` sets significance, and ``ClaimGate.is_significant`` RAISES
for a refused effect rather than returning False so that a table builder which forgets to ask
crashes instead of quietly printing a star. A figure is a table builder, and this file holds it to
the same rule.

The case that matters is the one that looks fine: a descriptive-only harm type keeps its ``q``,
because it WAS corrected -- only the claim is refused. A figure that filtered on ``q <= alpha``
would star it. Nothing here is hypothetical: the shipped fixture has exactly that shape, with
toxicity downgraded by the drift rule and several of its q values below alpha.
"""

from __future__ import annotations

import pytest

from margins.report.source import NumberSource
from margins.report.tables import read_table
from margins.stats.claims import ClaimGate, ClaimRefusedError
from margins.viz.claims import (
    ClaimMismatchError,
    descriptive_only_keys_from_table,
    effect_from_row,
    exploratory_pairs_from_lattice,
    significance,
)
from margins.viz.emergence import NAME as EMERGENCE
from tests.conftest import FIXTURE_TABLES, PAPER_NUMBERS

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BENCHMARK = "open_gen_templates"


@pytest.fixture(scope="module")
def effects():  # type: ignore[no-untyped-def]
    return read_table("emergence_effects", directory=FIXTURE_TABLES)


@pytest.fixture(scope="module")
def decisions():  # type: ignore[no-untyped-def]
    return read_table("descriptive_only", directory=FIXTURE_TABLES)


@pytest.fixture(scope="module")
def alpha() -> float:
    return (
        NumberSource(table_dir=FIXTURE_TABLES, paper_numbers_path=PAPER_NUMBERS)
        .number("fdr.alpha")
        .value
    )


def _claims(effects, decisions, alpha):  # type: ignore[no-untyped-def]
    return significance(
        effects.rows,
        alpha=alpha,
        exploratory_pairs=exploratory_pairs_from_lattice(),
        descriptive_only=descriptive_only_keys_from_table(decisions),
    )


# ------------------------------------------------------------------ the fixture has teeth


def test_the_fixture_actually_contains_a_refused_effect_with_a_small_q(effects, alpha) -> None:  # type: ignore[no-untyped-def]
    """Without this, every other test in the file passes vacuously."""
    trap = [
        row
        for row in effects.rows
        if row["claimable"] == "false" and row["q"] != "" and float(row["q"]) <= alpha
    ]
    assert trap, (
        "no refused effect carries q <= alpha, so 'the figure must not star it' is untested. "
        "Rebuild results/fixtures with a downgraded harm type that still reaches significance."
    )


# ------------------------------------------------------------------ the binding


def test_a_refused_effect_is_never_marked_significant(effects, decisions, alpha) -> None:  # type: ignore[no-untyped-def]
    for row, claim in zip(effects.rows, _claims(effects, decisions, alpha), strict=True):
        if claim.claimable:
            continue
        assert not claim.significant, f"{row['term']} was starred despite status {claim.status!r}"
        assert claim.marker == ""
        assert claim.refusal_reason


def test_the_gate_raises_rather_than_returning_false(effects, decisions, alpha) -> None:  # type: ignore[no-untyped-def]
    """The mechanism itself: asking a refused effect for significance is an error, not a no."""
    gate = ClaimGate(
        exploratory_pairs=exploratory_pairs_from_lattice(),
        descriptive_only=descriptive_only_keys_from_table(decisions),
        alpha=alpha,
    )
    refused = next(effect_from_row(row) for row in effects.rows if row["claimable"] == "false")
    with pytest.raises(ClaimRefusedError):
        gate.is_significant(refused)


def test_a_claimable_effect_is_starred_exactly_when_it_survives_correction(  # type: ignore[no-untyped-def]
    effects, decisions, alpha
) -> None:
    starred = 0
    for row, claim in zip(effects.rows, _claims(effects, decisions, alpha), strict=True):
        if not claim.claimable:
            continue
        expected = row["q"] != "" and float(row["q"]) <= alpha
        assert claim.significant is expected
        starred += int(claim.significant)
    assert starred > 0, "no claimable effect survives correction; the marker path is untested"


def test_the_figure_gate_must_agree_with_the_stats_run(effects, decisions, alpha) -> None:  # type: ignore[no-untyped-def]
    """A stale table is a reason to stop, not to draw. Circular trust is the failure mode."""
    row = dict(effects.rows[0])
    row["status"] = "claimable" if row["status"] != "claimable" else "exploratory"
    with pytest.raises(ClaimMismatchError, match="must not outvote P9"):
        significance(
            [row],
            alpha=alpha,
            exploratory_pairs=exploratory_pairs_from_lattice(),
            descriptive_only=descriptive_only_keys_from_table(decisions),
        )


def test_an_exploratory_pair_carries_no_significance(effects, decisions, alpha) -> None:  # type: ignore[no-untyped-def]
    exploratory = exploratory_pairs_from_lattice()
    assert exploratory, "the lattice declares no exploratory pair; the split is untested"
    rows = [row for row in effects.rows if (row["axis_a"], row["axis_b"]) in exploratory]
    assert rows
    for row in rows:
        assert row["q"] == "", "an exploratory estimate was given a q"
        assert row["family_member"] == "false"


def test_the_red_team_arm_carries_no_q_at_all(effects) -> None:  # type: ignore[no-untyped-def]
    """Settled 21 Sep 2026: the red-team arm is a descriptive subset outside the FDR family."""
    rows = [row for row in effects.rows if row["benchmark"] == "redteam"]
    assert rows
    assert all(row["q"] == "" and row["p"] == "" for row in rows)
    assert all(row["claimable"] == "false" for row in rows)


# ------------------------------------------------------------------ on the canvas


def test_the_emergence_figure_stars_only_permitted_effects(built_figures) -> None:  # type: ignore[no-untyped-def]
    """Count the stars actually drawn and hold them to what the gate permits."""
    spec = next(spec for spec, _ in built_figures if spec.name == EMERGENCE)
    stars = sum(text.get_text().count("*") for panel in spec.panels for text in panel.ax.texts)
    effects_table = read_table("emergence_effects", directory=FIXTURE_TABLES)
    decisions_table = read_table("descriptive_only", directory=FIXTURE_TABLES)
    alpha_value = (
        NumberSource(table_dir=FIXTURE_TABLES, paper_numbers_path=PAPER_NUMBERS)
        .number("fdr.alpha")
        .value
    )
    eligible = effects_table.where(
        benchmark=BENCHMARK, model_id=MODEL, kind="interaction", family_member="true"
    )
    permitted = sum(
        1 for claim in _claims(eligible, decisions_table, alpha_value) if claim.significant
    )
    assert stars == permitted, (
        f"the figure drew {stars} significance markers and the gate permits {permitted}"
    )
    assert permitted > 0


def test_the_emergence_figure_plots_interactions_and_not_levels(built_figures) -> None:  # type: ignore[no-untyped-def]
    """Emergence-is-not-level, checked on the artifact rather than trusted from the caption."""
    spec = next(spec for spec, _ in built_figures if spec.name == EMERGENCE)
    effects_table = read_table("emergence_effects", directory=FIXTURE_TABLES)
    levels = {
        float(row["estimate"]) for row in effects_table.rows if row["kind"] in ("main_a", "main_b")
    }
    interactions = {
        float(row["estimate"]) for row in effects_table.rows if row["kind"] == "interaction"
    }
    drawn = {traced.value for traced in spec.ledger}
    assert drawn & interactions, "the emergence figure drew no interaction coefficient"
    stray = (drawn & levels) - interactions
    assert not stray, (
        f"the emergence figure drew main-effect values {sorted(stray)[:5]}. The claim is about "
        "the interaction; a level on this figure is the conflation section 10 names as the most "
        "common way intersectional audits overclaim."
    )


def test_the_gamma_grid_the_heatmap_draws_sums_to_zero() -> None:
    """A table of interactions sums to zero along every row and column.

    A table of LEVELS does not, which is the whole distinction.
    """
    import numpy as np

    table = read_table("emergence_effects", directory=FIXTURE_TABLES).where(
        benchmark=BENCHMARK,
        model_id=MODEL,
        kind="interaction",
        harm_type="regard",
        axis_a="socioeconomic_class",
        axis_b="race_ethnicity",
    )
    levels_a = table.unique("value_a")
    levels_b = table.unique("value_b")
    grid = np.zeros((len(levels_a), len(levels_b)))
    for row in table.rows:
        grid[levels_a.index(row["value_a"]), levels_b.index(row["value_b"])] = float(
            row["estimate"]
        )
    assert np.allclose(grid.sum(axis=0), 0.0, atol=1e-5)
    assert np.allclose(grid.sum(axis=1), 0.0, atol=1e-5)
