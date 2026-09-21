"""Figure style and deterministic vector output (P10).

Two jobs, both of which exist because of a project guardrail rather than a taste preference.

**Determinism.** Invariant 5 says determinism is a contract, and a figure is an artifact like any
other: ``scripts/make_figures.py`` must produce the same bytes from the same tables. Matplotlib
does not do that by default -- it stamps a creation date into PDF and SVG metadata and derives SVG
element ids from a per-process hash salt. :func:`save_figure` pins the salt, drops the dates, and
writes text as text rather than as outlined glyph paths, so two runs differ in no byte.
``tests/test_make_figures.py`` asserts it.

**Print legibility.** Nothing here is a raster. Headline figures go out as PDF *and* SVG at a
column width that matches a two-column venue, with font sizes chosen so the smallest label is
still 7pt on the printed page. Seaborn is a declared dependency of the project and is deliberately
*not* used for drawing: it ships no type information (``mypy --strict`` runs over ``src``) and its
themes change defaults between minor versions, which is exactly the kind of invisible drift the
determinism test exists to catch. The rcParams below are the theme, written out.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

from margins.viz.palette import GRID, INK

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.figure import Figure

__all__ = [
    "COLUMN_WIDTH",
    "DEFAULT_FIGURE_DIR",
    "FULL_WIDTH",
    "HASH_SALT",
    "VECTOR_FORMATS",
    "figure_style",
    "save_figure",
]

#: Where ``scripts/make_figures.py`` writes. Committed: vector figures are small, and a figure
#: nobody can open from the repo is a figure nobody can check against the tables.
DEFAULT_FIGURE_DIR: Final[Path] = Path("figures")

#: Vector only. ``.gitignore`` drops ``figures/*.png``, so a rasterised headline figure cannot be
#: committed even by accident.
VECTOR_FORMATS: Final[tuple[str, ...]] = ("pdf", "svg")

#: Fixed so SVG element ids are a function of the content and not of the process. Any constant
#: works; changing it changes every SVG's bytes, so it is pinned here and never derived.
HASH_SALT: Final[str] = "mapping-the-margins"

#: Inches. A single column of a two-column venue, and the full text width.
COLUMN_WIDTH: Final[float] = 3.4
FULL_WIDTH: Final[float] = 7.0

_RC: Final[dict[str, Any]] = {
    # DejaVu Sans ships with matplotlib, so the same glyphs are available on both locked
    # platforms. A system font would make the output host-dependent.
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
    "figure.titlesize": 10.0,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "lines.linewidth": 1.2,
    "lines.markersize": 4.0,
    "figure.dpi": 150.0,
    "savefig.dpi": 300.0,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "savefig.transparent": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    # Determinism and vector fidelity.
    "svg.hashsalt": HASH_SALT,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "pdf.compression": 6,
    # A headline figure is never rasterised, whatever a call site asks for.
    "image.composite_image": False,
}

#: Passed to ``savefig`` so no timestamp reaches the file. Matplotlib treats ``None`` as
#: "omit this key" for both backends.
_METADATA: Final[dict[str, dict[str, None]]] = {
    "pdf": {"CreationDate": None},
    "svg": {"Date": None},
}


@contextmanager
def figure_style(overrides: Mapping[str, Any] | None = None) -> Iterator[None]:
    """Apply the project theme for the duration of the block.

    A context manager rather than a module-level ``rcParams.update`` so importing
    :mod:`margins.viz` never mutates a caller's matplotlib state -- a notebook that imports a
    figure module should not silently acquire our fonts.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    settings = dict(_RC)
    settings.update(overrides or {})
    # The stubs type rc_context's mapping as a Literal-keyed TypedDict; our theme is a plain
    # dict assembled from two sources, and every key in it is checked by matplotlib at runtime.
    with matplotlib.rc_context(cast("Any", settings)):
        yield


def save_figure(
    figure: Figure,
    name: str,
    *,
    out_dir: Path = DEFAULT_FIGURE_DIR,
    formats: tuple[str, ...] = VECTOR_FORMATS,
) -> dict[str, Path]:
    """Write ``name`` in every vector format. Byte-deterministic; returns ``{format: path}``.

    Raises:
        ValueError: on a raster format. The exit gate says vector, and a helpful fallback here
            would be the one place a rasterised headline figure could enter the repo.
    """
    raster = [fmt for fmt in formats if fmt not in VECTOR_FORMATS]
    if raster:
        raise ValueError(
            f"{raster} is not a vector format. Headline figures ship as {VECTOR_FORMATS}; "
            "figures/*.png is gitignored precisely so this cannot be worked around."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        figure.savefig(path, format=fmt, metadata=_METADATA.get(fmt))
        _normalise_newlines(path)
        written[fmt] = path
    return written


def _normalise_newlines(path: Path) -> None:
    """Rewrite a text-format figure with LF endings, whatever platform wrote it.

    SVG is text and matplotlib opens it in text mode, so a figure written on Windows arrives with
    CRLF and the same figure written on Linux with LF. The repo normalises text to LF
    (``.gitattributes``), so without this a Windows regeneration would dirty the worktree on every
    run and the determinism claim would hold only within one operating system. PDF is binary and
    is left alone.
    """
    if path.suffix.lower() != ".svg":
        return
    raw = path.read_bytes()
    if b"\r\n" in raw:
        path.write_bytes(raw.replace(b"\r\n", b"\n"))
