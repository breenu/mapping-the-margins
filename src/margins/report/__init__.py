"""Result tables and the traceability seam between the pipeline and the paper (P10).

The package has one idea in it. **A figure and a table are the same artifact seen twice**, so they
must come from the same place: a small set of *committed* result tables under ``results/``, plus
``results/paper_numbers.json``. Nothing in :mod:`margins.viz` reads a Parquet file, a config, or a
constant in a Python module; it reads a table or a paper number through
:class:`~margins.report.source.NumberSource`, which keeps a ledger of every value it hands out so
the figure tests can check the canvas against it.

That indirection buys three things:

* the manuscript, the tables and the figures cannot disagree, because there is one source;
* ``data/`` stays gitignored and regenerable while the paper's inputs stay in git;
* a figure built today against fixture rows becomes a figure built against the real E1 run at P11
  with **no change to the figure code** -- only the table under it changes.
"""

from __future__ import annotations

__all__ = ["__doc__"]
