"""The smoke run's one figure: the REAL emergence figure, on the smoke tables, bannered.

It is built through :func:`margins.viz.make.build_figure`, so the traceability audit and the claim
binding both run exactly as they will on the confirmatory tables -- in particular
:mod:`margins.viz.claims` runs the real ``ClaimGate`` at figure time and raises if its verdict
disagrees with the one P9 recorded. That is the last link of the chain the smoke run exists to
exercise: ``attach_wald_p -> correct -> ClaimGate -> figure``.

Two differences from ``scripts/make_figures.py``, both deliberate:

* **The model is the smoke model.** The committed recipe draws the primary confirmatory checkpoint;
  a smoke run on the ~1B dev model would find no rows for it.
* **A banner says it is a smoke run.** The smoke tables carry ``data_status = real`` -- they ARE
  real pipeline output -- so the fixture banner never fires. A figure of 108 generations on one
  pair must not be mistakable for a result, so it is stamped, with no digits (the audit reads every
  number on a canvas), and then audited AGAIN, so the stamp cannot smuggle one in.
"""

from __future__ import annotations

from pathlib import Path

from margins.orchestrate.smoke import SMOKE_FIGURE_BANNER

__all__ = ["build_smoke_figure"]


def build_smoke_figure(
    *,
    table_dir: Path,
    paper_numbers_path: Path,
    model_id: str,
    benchmark: str,
    out_dir: Path,
) -> dict[str, Path]:
    """Build, banner, re-audit and save the smoke emergence figure. Returns ``{format: path}``.

    Raises:
        margins.viz.make.UntracedValueError: if the canvas carries an untraced number, before or
            after the banner.
    """
    import matplotlib.pyplot as plt

    from margins.viz import emergence
    from margins.viz.make import FigureRecipe, UntracedValueError, build_figure
    from margins.viz.style import figure_style, save_figure
    from margins.viz.trace import audit_figure

    recipe = FigureRecipe(
        name=emergence.NAME,
        claim_id="RQ2 (smoke)",
        build=lambda source: emergence.build(source, benchmark=benchmark, model_id=model_id),
    )
    with figure_style():
        spec, _ = build_figure(recipe, table_dir=table_dir, paper_numbers_path=paper_numbers_path)
        spec.figure.text(
            0.5,
            0.995,
            SMOKE_FIGURE_BANNER,
            ha="center",
            va="top",
            fontsize=7.0,
            color="#9b1c1c",
            fontweight="bold",
        )
        unbacked = audit_figure(spec)
        if unbacked:
            raise UntracedValueError(
                f"the smoke banner put untraced numbers on the canvas: {unbacked[:3]}"
            )
        written = save_figure(spec.figure, f"smoke_{spec.name}", out_dir=out_dir)
        plt.close(spec.figure)
    return written
