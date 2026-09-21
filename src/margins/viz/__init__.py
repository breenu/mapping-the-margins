"""Publication figures (P10).

Every figure in this package carries one of the paper's claims, and the module that draws it is
named after the claim rather than after the chart type. That is the whole organising principle:
a figure here is an argument, and a figure nobody can name the claim for does not belong.

Three rules are enforced mechanically rather than by review, because all three have been violated
by published intersectional audits:

* **No number is typed into a figure.** Every plotted value comes through
  :class:`~margins.report.source.NumberSource`, which hands out :class:`~margins.report.source.
  Traced` values and keeps a ledger of what it handed out.
  :func:`margins.viz.trace.audit_figure` then walks the rendered artists and fails if anything on
  the canvas is not in that ledger.
* **Emergence is the interaction, never the level.** The emergence figure plots gamma and says so
  in its own subtitle; a bare cell mean is drawn nowhere in this package.
* **A figure may not display a claim the gate would refuse.** Significance markers are produced by
  :mod:`margins.viz.claims`, which runs the real :class:`~margins.stats.claims.ClaimGate` at figure
  time and cross-checks its verdict against the one P9 recorded.
"""

from __future__ import annotations

__all__ = ["__doc__"]
