"""P11 orchestration: the lab run, as declared commands over the existing scripts.

Nothing here measures anything. Every number the experiment produces still comes from the P5-P10
scripts, unchanged; this package decides *which* invocations run, in what order, with what
arguments, and refuses to start one that would not be the pre-registered measurement:

* :mod:`~margins.orchestrate.config` -- the committed run declaration (seeds, batch size, arms).
* :mod:`~margins.orchestrate.plan` -- the ONE place every argv is written; the runbook prints
  these and a test parses each against the real script.
* :mod:`~margins.orchestrate.preflight` -- refusals: dirty tree, pre-registration ancestry, a
  confirmatory model that fell back from NF4, a resume at a different batch size.
* :mod:`~margins.orchestrate.smoke` -- the multi-rendering smoke subset and its pass criterion.
* :mod:`~margins.orchestrate.budget` -- GPU-hour projections from MEASURED speed.
* :mod:`~margins.orchestrate.figure` -- the smoke run's bannered emergence figure.

The entry point is ``scripts/run_p11.py``; the human procedure is ``docs/runbook_p11.md``.
"""

from __future__ import annotations

from margins.orchestrate.budget import (
    MIN_REDTEAM_FRAMES,
    RedTeamBudget,
    project_gpu_hours,
    redteam_max_frames,
)
from margins.orchestrate.config import (
    DEFAULT_P11,
    P11Config,
    default_p11_config_dir,
    load_p11_config,
)
from margins.orchestrate.plan import Command, Layout, SmokeLayout
from margins.orchestrate.smoke import SmokeCheck, SmokeVerdict, check_smoke_pass

__all__ = [
    "DEFAULT_P11",
    "MIN_REDTEAM_FRAMES",
    "Command",
    "Layout",
    "P11Config",
    "RedTeamBudget",
    "SmokeCheck",
    "SmokeLayout",
    "SmokeVerdict",
    "check_smoke_pass",
    "default_p11_config_dir",
    "load_p11_config",
    "project_gpu_hours",
    "redteam_max_frames",
]
