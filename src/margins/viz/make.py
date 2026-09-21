"""The figure registry: one command builds every figure, and the audit runs on each (P10).

Gate P10 asks for one-command regeneration from committed tables. This module is the registry
that makes it one command, and it is deliberately the only place the figure set is enumerated --
``scripts/make_figures.py`` is argparse and printing, and the tests build from here, so a figure
that renders in CI but is missing from the script cannot exist.

**The audit runs inside the build, not beside it.** :func:`build_figure` renders, audits, and
raises if any number on the canvas is unaccounted for. Making the check optional would make it
advisory, and an advisory traceability rule is one somebody skips at deadline.

**Each figure gets its own ``NumberSource``.** Sharing one would pool the ledgers, and a value
figure 1 legitimately drew would then excuse an identical stray number in figure 4. Separate
ledgers keep each figure answerable for its own canvas.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from margins.report.source import NumberSource
from margins.stats.numbers import PAPER_NUMBERS_PATH
from margins.viz import drift, emergence, field_power, frames, paradigm, redteam, sensitivity
from margins.viz.style import DEFAULT_FIGURE_DIR, VECTOR_FORMATS, figure_style, save_figure
from margins.viz.trace import FigureSpec, audit_figure, stamp_data_status

if TYPE_CHECKING:  # pragma: no cover - typing only
    from margins.viz.trace import Unbacked

__all__ = [
    "ESSENTIAL",
    "FIGURES",
    "SUPPORTING",
    "FigureRecipe",
    "UntracedValueError",
    "build_all",
    "build_figure",
]

#: The confirmatory model the single-model panels show. The pinned 8B Llama is the project's
#: primary confirmatory checkpoint; Mistral appears wherever a figure shows both.
PRIMARY_MODEL: Final[str] = "meta-llama/Llama-3.1-8B-Instruct"
GENERATION_BENCHMARK: Final[str] = "open_gen_templates"


class UntracedValueError(AssertionError):
    """A figure drew a number that no committed artifact accounts for."""


@dataclass(frozen=True)
class FigureRecipe:
    """One figure: its name, the claim it carries, and how to build it."""

    name: str
    claim_id: str
    build: Callable[[NumberSource], FigureSpec]
    essential: bool = True


ESSENTIAL: Final[tuple[FigureRecipe, ...]] = (
    FigureRecipe(
        name=emergence.NAME,
        claim_id="RQ2",
        build=lambda source: emergence.build(
            source, benchmark=GENERATION_BENCHMARK, model_id=PRIMARY_MODEL
        ),
    ),
    FigureRecipe(
        name=drift.NAME,
        claim_id="construct validity",
        build=lambda source: drift.build(
            source, benchmark=GENERATION_BENCHMARK, model_id=PRIMARY_MODEL
        ),
    ),
    FigureRecipe(name=field_power.NAME, claim_id="field result", build=field_power.build),
    FigureRecipe(
        name=paradigm.NAME,
        claim_id="RQ1",
        build=lambda source: paradigm.build(source, model_id=PRIMARY_MODEL),
    ),
    FigureRecipe(
        name=redteam.NAME,
        claim_id="RQ3",
        build=lambda source: redteam.build(source, model_id=PRIMARY_MODEL),
    ),
)

SUPPORTING: Final[tuple[FigureRecipe, ...]] = (
    FigureRecipe(name=sensitivity.NAME, claim_id="RQ4", build=sensitivity.build, essential=False),
    FigureRecipe(name=frames.NAME, claim_id="precision", build=frames.build, essential=False),
)

FIGURES: Final[tuple[FigureRecipe, ...]] = (*ESSENTIAL, *SUPPORTING)


def build_figure(
    recipe: FigureRecipe,
    *,
    table_dir: Path | None = None,
    paper_numbers_path: Path = PAPER_NUMBERS_PATH,
) -> tuple[FigureSpec, tuple[Unbacked, ...]]:
    """Render one figure, stamp its data status, and audit its canvas.

    Raises:
        UntracedValueError: if the canvas carries a number the ledger does not account for.
    """
    source = NumberSource(table_dir=table_dir, paper_numbers_path=paper_numbers_path)
    spec = recipe.build(source)
    stamp_data_status(spec.figure, spec.data_status)
    unbacked = audit_figure(spec)
    if unbacked:
        raise UntracedValueError(
            f"{spec.name} drew {len(unbacked)} number(s) that trace to no paper number and no "
            f"committed table: {[str(item) for item in unbacked[:6]]}. Obtain them through "
            "NumberSource, or declare them structural with a reason."
        )
    return spec, unbacked


def build_all(
    *,
    out_dir: Path = DEFAULT_FIGURE_DIR,
    table_dir: Path | None = None,
    paper_numbers_path: Path = PAPER_NUMBERS_PATH,
    recipes: Sequence[FigureRecipe] = FIGURES,
    formats: tuple[str, ...] = VECTOR_FORMATS,
    save: bool = True,
) -> list[tuple[FigureSpec, dict[str, Path]]]:
    """Build every figure under the project theme. Returns ``[(spec, {format: path})]``."""
    import matplotlib.pyplot as plt

    out: list[tuple[FigureSpec, dict[str, Path]]] = []
    with figure_style():
        for recipe in recipes:
            spec, _ = build_figure(
                recipe, table_dir=table_dir, paper_numbers_path=paper_numbers_path
            )
            written = (
                save_figure(spec.figure, spec.name, out_dir=out_dir, formats=formats)
                if save
                else {}
            )
            out.append((spec, written))
            plt.close(spec.figure)
    return out
