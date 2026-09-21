"""Gate P10: no figure draws a number that no committed artifact accounts for.

The guardrail is "never invent a statistic". P9 enforced it for the manuscript by making it cite
numbers by key; this enforces it for the figures, which assert just as much as a sentence does.

The check is not a promise that the builder *asked* for traced values -- that would be satisfied
by asking and then drawing something else. It reads the numbers back off the rendered artists and
matches them against the ledger of what :class:`~margins.report.source.NumberSource` handed out.
``test_the_audit_can_fail`` is the load-bearing test here: an audit that cannot fail is
decoration, so this file proves it catches a hand-typed value, a hard-coded reference line and a
rounded annotation that drifted from its table.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from margins.report.source import MissingNumberError, NumberSource
from margins.viz.make import FIGURES, UntracedValueError, build_figure
from margins.viz.style import figure_style
from margins.viz.trace import AuditedPanel, Encoding, FigureSpec, audit_figure, decorate
from tests.conftest import FIXTURE_TABLES, PAPER_NUMBERS


def _source() -> NumberSource:
    return NumberSource(table_dir=FIXTURE_TABLES, paper_numbers_path=PAPER_NUMBERS)


# ------------------------------------------------------------------ the figures


def test_every_figure_traces_every_number_it_draws(built_figures: list) -> None:  # type: ignore[type-arg]
    for spec, _ in built_figures:
        unbacked = audit_figure(spec)
        assert not unbacked, f"{spec.name} drew untraced values: {[str(u) for u in unbacked]}"


def test_every_figure_carries_a_named_claim(built_figures: list) -> None:  # type: ignore[type-arg]
    """A figure whose claim cannot be written in a sentence is a generic plot."""
    for spec, _ in built_figures:
        assert len(spec.claim.split()) >= 12, f"{spec.name}'s claim is not a sentence"
        assert spec.caption.strip()
        assert spec.ledger, f"{spec.name} obtained no traced values at all"


def test_every_ledger_entry_names_its_origin(built_figures: list) -> None:  # type: ignore[type-arg]
    for spec, _ in built_figures:
        for traced in spec.ledger:
            assert traced.key, f"{spec.name} recorded a value with no key"
            assert traced.origin.endswith((".json", ".csv")), (
                f"{spec.name} traced {traced.key!r} to {traced.origin!r}, which is neither "
                "paper_numbers.json nor a committed result table"
            )


def test_every_structural_exemption_states_a_reason(built_figures: list) -> None:  # type: ignore[type-arg]
    for spec, _ in built_figures:
        for value, reason in spec.structural.items():
            assert reason.strip(), f"{spec.name} exempts {value} with no reason"


# ------------------------------------------------------------------ the audit itself


def _tiny_spec(draw) -> FigureSpec:  # type: ignore[no-untyped-def]
    import matplotlib.pyplot as plt

    source = _source()
    figure, ax = plt.subplots()
    ax.set_label("tiny")
    draw(ax, source)
    return FigureSpec(
        name="tiny",
        title="tiny",
        claim="A test figure that exists only so the traceability audit has something to read.",
        caption="test",
        figure=figure,
        panels=(AuditedPanel(ax, "y"),),
        ledger=source.ledger,
        encodings=(Encoding("series", "#0072B2", marker="o"),),
    )


def test_the_audit_can_fail_on_a_hand_typed_coordinate() -> None:
    with figure_style():
        spec = _tiny_spec(lambda ax, source: ax.plot([0, 1], [0.0, 0.4242]))
    unbacked = audit_figure(spec)
    assert any(abs(item.value - 0.4242) < 1e-9 for item in unbacked), (
        "the audit accepted a value that came from nowhere"
    )


def test_the_audit_can_fail_on_a_hard_coded_reference_line() -> None:
    with figure_style():
        spec = _tiny_spec(lambda ax, source: ax.axhline(0.73))
    assert any(abs(item.value - 0.73) < 1e-9 for item in audit_figure(spec))


def test_the_audit_can_fail_on_an_annotation_that_drifted_from_its_table() -> None:
    """The realistic failure: a number was right once, the table moved, the text did not."""

    def draw(ax, source):  # type: ignore[no-untyped-def]
        traced = source.number("rho.regard")
        ax.plot([0, 1], [traced.value, traced.value])
        ax.text(0.5, traced.value, "rho = 0.41")  # stale: the measured value is 0.35

    with figure_style():
        spec = _tiny_spec(draw)
    assert any(abs(item.value - 0.41) < 1e-9 for item in audit_figure(spec))


def test_a_traced_value_passes_and_a_rounded_print_of_it_passes() -> None:
    def draw(ax, source):  # type: ignore[no-untyped-def]
        traced = source.number("rho.regard")
        ax.plot([0, 1], [traced.value, traced.value])
        ax.text(0.5, traced.value, f"rho = {traced.value:.2f}")

    with figure_style():
        spec = _tiny_spec(draw)
    assert audit_figure(spec) == ()


def test_decoration_is_skipped_but_its_text_is_not_exempt() -> None:
    """Furniture may be skipped as geometry; it may not smuggle a number in as a label."""

    def draw(ax, source):  # type: ignore[no-untyped-def]
        del source
        decorate(ax.axhspan(0.11, 0.97, color="#D9DCE0"))
        ax.text(0.5, 0.5, "measured at 0.6789")

    with figure_style():
        spec = _tiny_spec(draw)
    values = {round(item.value, 4) for item in audit_figure(spec)}
    assert 0.6789 in values
    assert 0.11 not in values and 0.97 not in values


def test_a_hyphenated_word_is_not_read_as_a_negative_number() -> None:
    with figure_style():
        spec = _tiny_spec(lambda ax, source: ax.set_title("order-2 cells, FDR-corrected"))
    assert audit_figure(spec) == ()


# ------------------------------------------------------------------ NumberSource


def test_a_missing_paper_number_raises_rather_than_defaulting() -> None:
    with pytest.raises(MissingNumberError):
        _source().number("power.no_such_key")


def test_a_non_numeric_paper_number_is_not_plottable() -> None:
    """``fdr.method`` is a real entry and a string. A figure asking for it must stop."""
    from margins.stats.numbers import NO_RUN, PaperNumber, PaperNumbers

    numbers = PaperNumbers.of(
        [PaperNumber(key="fdr.method", value="fdr_bh", run_id=NO_RUN, source="configs/stats/")]
    )
    source = NumberSource(numbers=numbers, table_dir=FIXTURE_TABLES)
    with pytest.raises(MissingNumberError, match="not plottable"):
        source.number("fdr.method")


def test_a_derivation_records_how_it_was_derived() -> None:
    source = _source()
    traced = source.derived(42.0, key="count", origin="test.csv", how="rows that matched")
    assert "rows that matched" in traced.key
    assert source.ledger[-1] is traced


def test_a_missing_table_names_the_script_that_builds_it(tmp_path: Path) -> None:
    source = NumberSource(table_dir=tmp_path, paper_numbers_path=PAPER_NUMBERS)
    with pytest.raises(FileNotFoundError, match=r"build_fixture_results\.py"):
        source.table("emergence_effects")


def test_the_figure_builder_refuses_to_return_an_untraced_figure() -> None:
    """`build_figure` audits inside the build, so an advisory check cannot be skipped."""
    import margins.viz.emergence as emergence_module
    from margins.viz.make import FigureRecipe

    original = next(r for r in FIGURES if r.name == emergence_module.NAME)

    def sabotage(source: NumberSource) -> FigureSpec:
        spec = original.build(source)
        spec.panels[0].ax.axvline(0.61803)
        return spec

    recipe = FigureRecipe(name="sabotaged", claim_id="test", build=sabotage)
    with figure_style(), pytest.raises(UntracedValueError, match="trace to no paper number"):
        build_figure(recipe, table_dir=FIXTURE_TABLES, paper_numbers_path=PAPER_NUMBERS)
