"""Gate P10: one command regenerates every figure, as deterministic vector artifacts.

``scripts/make_figures.py`` is the command. The properties asserted here are the ones the exit
gate names, plus the two that make a figure an *artifact* rather than a picture somebody made
once: it is byte-identical on a second run, and it cannot be written as a raster.

Determinism is not decoration. Matplotlib stamps a creation date into PDF and SVG and derives SVG
element ids from a per-process salt, so the naive version of this pipeline produces a different
file every run and no reviewer can tell a re-render from an edit.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from margins.report.tables import DEFAULT_FIXTURE_DIR, DEFAULT_TABLE_DIR
from margins.viz.make import ESSENTIAL, FIGURES, SUPPORTING, build_all
from margins.viz.style import VECTOR_FORMATS, figure_style, save_figure
from tests.conftest import FIXTURE_TABLES, PAPER_NUMBERS, REPO_ROOT


def _digests(directory: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


# ------------------------------------------------------------------ the registry


def test_the_five_essential_figures_are_the_ones_the_brief_names() -> None:
    assert [recipe.claim_id for recipe in ESSENTIAL] == [
        "RQ2",
        "construct validity",
        "field result",
        "RQ1",
        "RQ3",
    ]
    assert all(recipe.essential for recipe in ESSENTIAL)
    assert not any(recipe.essential for recipe in SUPPORTING)
    assert len({recipe.name for recipe in FIGURES}) == len(FIGURES)


def test_every_recipe_actually_builds(built_figures) -> None:  # type: ignore[no-untyped-def]
    assert [spec.name for spec, _ in built_figures] == [recipe.name for recipe in FIGURES]


# ------------------------------------------------------------------ the command


def test_one_command_writes_every_figure_as_vector_pdf_and_svg(tmp_path: Path) -> None:
    import scripts.make_figures as make_figures

    code = make_figures.main(
        [
            "--out-dir",
            str(tmp_path),
            "--table-dir",
            str(FIXTURE_TABLES),
            "--paper-numbers",
            str(PAPER_NUMBERS),
        ]
    )
    assert code == 0
    for recipe in FIGURES:
        for fmt in VECTOR_FORMATS:
            path = tmp_path / f"{recipe.name}.{fmt}"
            assert path.is_file(), f"{path} was not written"
            assert path.stat().st_size > 0
    assert not list(tmp_path.glob("*.png")), "a raster figure reached the output directory"


def test_the_figures_are_byte_deterministic(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    for out in (first, second):
        build_all(out_dir=out, table_dir=FIXTURE_TABLES, paper_numbers_path=PAPER_NUMBERS)
    assert _digests(first) == _digests(second)


def test_a_raster_format_is_refused(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    with figure_style():
        figure = plt.figure()
        with pytest.raises(ValueError, match="not a vector format"):
            save_figure(figure, "raster", out_dir=tmp_path, formats=("png",))
        plt.close(figure)


def test_the_svg_carries_no_timestamp_and_no_random_ids(tmp_path: Path) -> None:
    """The two sources of spurious diffs, checked directly rather than inferred."""
    built = build_all(
        out_dir=tmp_path,
        table_dir=FIXTURE_TABLES,
        paper_numbers_path=PAPER_NUMBERS,
        recipes=ESSENTIAL[:1],
    )
    path = built[0][1]["svg"]
    assert "<dc:date>" not in path.read_text(encoding="utf-8"), "the SVG embeds a creation date"
    assert b"\r\n" not in path.read_bytes(), (
        "the SVG was written with CRLF. The repo normalises text to LF, so a Windows "
        "regeneration would dirty the worktree on every run and determinism would hold only "
        "within one operating system."
    )


# ------------------------------------------------------------------ fixture vs real


def test_every_fixture_backed_figure_says_so_on_its_face(built_figures) -> None:  # type: ignore[no-untyped-def]
    for spec, _ in built_figures:
        banner = [
            text.get_text() for text in spec.figure.texts if "FIXTURE DATA" in text.get_text()
        ]
        if spec.data_status == "fixture":
            assert banner, (
                f"{spec.name} is drawn from synthetic rows and carries no banner. A reader must "
                "not have to learn that from a commit message."
            )
        else:
            assert not banner, f"{spec.name} is real and claims to be a fixture"


def test_the_field_result_figure_is_built_from_real_measurements(built_figures) -> None:  # type: ignore[no-untyped-def]
    """Figure 3 is the one that must not be fixture-backed: rho and the power sim are measured."""
    spec = next(spec for spec, _ in built_figures if spec.name == "fig3_field_underpowered")
    assert spec.data_status == "real"
    assert all(traced.origin.endswith("paper_numbers.json") for traced in spec.ledger)


def test_no_fixture_value_ever_reaches_paper_numbers() -> None:
    """The guardrail against invented statistics, at its one enforcement point."""
    from margins.stats.numbers import read_paper_numbers

    numbers = read_paper_numbers(PAPER_NUMBERS)
    sources = {entry.source for entry in numbers.numbers}
    assert not any("fixture" in source for source in sources), (
        f"a fixture-derived entry reached paper_numbers.json: {sorted(sources)}"
    )


def test_the_table_directory_falls_back_to_fixtures_until_a_real_run_exists() -> None:
    from margins.report.source import resolve_table_dir

    assert resolve_table_dir(Path("explicit")) == Path("explicit")
    real = REPO_ROOT / "results" / "no-such-tables"
    assert resolve_table_dir(None, real=real, fixture=DEFAULT_FIXTURE_DIR) == DEFAULT_FIXTURE_DIR
    assert resolve_table_dir(None, real=FIXTURE_TABLES, fixture=DEFAULT_TABLE_DIR) == FIXTURE_TABLES


def test_a_pending_real_table_falls_back_alone_and_stamps_only_its_figure() -> None:
    """After P11, `sensitivity` (E4) is built later than the rest. One pending table must fall
    back to its fixture -- marking that figure FIXTURE -- without demoting the real ones; and an
    explicit --table-dir stays strict."""
    from margins.report.source import NumberSource

    if not (DEFAULT_TABLE_DIR / "emergence_effects.csv").is_file():
        pytest.skip("no real result tables committed yet")
    if (DEFAULT_TABLE_DIR / "sensitivity.csv").is_file():
        pytest.skip("the sensitivity table is real now; nothing is pending")

    real_only = NumberSource(paper_numbers_path=PAPER_NUMBERS)
    real_only.table("emergence_effects")
    assert real_only.data_status == "real"

    mixed = NumberSource(paper_numbers_path=PAPER_NUMBERS)
    mixed.table("emergence_effects")
    assert mixed.table("sensitivity").data_status == "fixture"
    assert mixed.data_status == "fixture"

    strict = NumberSource(table_dir=DEFAULT_TABLE_DIR, paper_numbers_path=PAPER_NUMBERS)
    with pytest.raises(FileNotFoundError):
        strict.table("sensitivity")


# ------------------------------------------------------------------ environment


def test_the_figure_layer_never_reaches_for_the_gpu_stack() -> None:
    """P10 runs on the core CI job, which has no torch at all.

    Scanned statically rather than by checking ``sys.modules``: the dev box has the inference
    extra installed and some other test in the session will have imported torch, so a runtime
    check here would pass on CI for the wrong reason and be meaningless locally.
    """
    import ast

    gpu = {"torch", "transformers", "bitsandbytes", "accelerate", "detoxify", "huggingface_hub"}
    offenders: list[str] = []
    for package in ("viz", "report"):
        for path in (REPO_ROOT / "src" / "margins" / package).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    if name.split(".")[0] in gpu:
                        offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} {name}")
    assert not offenders, (
        "the figure and report layers import the GPU stack: "
        + "; ".join(offenders)
        + ". The core environment has no torch and the core CI job runs on both OSes without it."
    )
