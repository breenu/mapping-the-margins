"""Guardrail: production logic never depends on notebooks.

CLAUDE.md forbids load-bearing notebooks -- ``src/`` and ``scripts/`` must not import from
``notebooks/``. Cheap to enforce here; expensive to discover in review.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_IMPORT = re.compile(r"^\s*(?:from|import)\s+notebooks\b", re.MULTILINE)


def test_src_and_scripts_do_not_import_from_notebooks() -> None:
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for directory in ("src", "scripts")
        for path in (REPO_ROOT / directory).rglob("*.py")
        if NOTEBOOK_IMPORT.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"production code imports from notebooks/: {offenders}"
