"""Gate: the pinned lexicon fetch materialises the PIN, and refuses anything else.

``scripts/fetch_lexicons.py`` is deliberately kept out of the network-touching test suite, so
what is tested here is the decision logic with the download stubbed out. That is where the risk
lives: the CI job that calls this runs unattended, and a mode that quietly wrote the wrong bytes
would put an unpinned wordlist behind every toxicity score in the nightly run.

The distinction the tests defend is between the two modes. ``--write`` resolves upstream HEAD and
is how a human re-pins; ``--fetch-pinned`` downloads the revision the roster already records. Using
the first where the second belongs is the exact shape of "unpinning to make it work".
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "fetch_lexicons.py"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fetch_lexicons", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def roster():  # type: ignore[no-untyped-def]
    from margins.scoring.config import (
        DEFAULT_SCORING,
        default_scoring_config_dir,
        load_scoring_config,
    )

    return load_scoring_config(DEFAULT_SCORING, config_dir=default_scoring_config_dir())


def pinned_spec(roster):  # type: ignore[no-untyped-def]
    for spec in roster.scorers:
        if spec.lexicon_pin is not None and spec.lexicon_pin.path is not None:
            return spec
    raise AssertionError("the shipped roster has no fetched lexicon to exercise")


def test_it_downloads_at_the_pinned_revision_and_never_resolves_head(
    script: ModuleType, roster, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE behaviour. Fetching HEAD would score against a wordlist nobody recorded.

    Asserted on the revision actually requested and on ``resolve_head`` never being reached,
    rather than on the write succeeding: synthesising bytes with a chosen sha256 is not possible,
    and the write path is covered by the mismatch and already-present tests either side of this.
    """
    spec = pinned_spec(roster)
    pin = spec.lexicon_pin
    requested: list[str] = []

    def fake_fetch(source, revision, *, timeout=60):  # type: ignore[no-untyped-def]
        requested.append(revision)
        return b"arbitrary bytes\n"

    def explode(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("resolve_head must NOT be called: it resolves upstream HEAD")

    monkeypatch.setattr(script, "fetch", fake_fetch)
    monkeypatch.setattr(script, "resolve_head", explode)
    monkeypatch.setattr(script, "_normalize", lambda payload: payload)

    script.fetch_pinned(roster, data_dir=tmp_path)
    assert requested == [pin.revision], "the roster's recorded revision, not HEAD"
    assert len(pin.revision) == 40, "a branch name is not a pin"


def test_the_real_fetch_round_trips_against_the_committed_pin(
    script: ModuleType, roster, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bytes on disk in this checkout must satisfy the pin the roster records.

    No network: it replays the local copy through the same digest check the fetch performs, so a
    roster whose sha256 no longer matches the wordlist it names fails here rather than in the
    nightly job.
    """
    spec = pinned_spec(roster)
    pin = spec.lexicon_pin
    real = REPO_ROOT / "data" / pin.path
    if not real.is_file():
        pytest.skip("the real lexicon is not fetched in this checkout")
    monkeypatch.setattr(script, "fetch", lambda *a, **k: real.read_bytes())
    assert script.fetch_pinned(roster, data_dir=tmp_path) == 0
    written = tmp_path / pin.path
    assert hashlib.sha256(written.read_bytes()).hexdigest() == pin.sha256


def test_a_digest_mismatch_refuses_to_write(
    script: ModuleType, roster, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upstream rewriting history at the pinned revision must STOP the run, not overwrite."""
    spec = pinned_spec(roster)
    pin = spec.lexicon_pin
    monkeypatch.setattr(script, "fetch", lambda *a, **k: b"not the pinned wordlist\n")
    monkeypatch.setattr(script, "_normalize", lambda payload: payload)

    assert script.fetch_pinned(roster, data_dir=tmp_path) == 1
    assert not (tmp_path / pin.path).exists(), "a mismatched lexicon must never be written"


def test_an_existing_correct_file_is_left_alone(
    script: ModuleType, roster, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Idempotent, and it must not hit the network when the pin is already satisfied."""
    spec = pinned_spec(roster)
    pin = spec.lexicon_pin
    real = REPO_ROOT / "data" / pin.path
    if not real.is_file():
        pytest.skip("the real lexicon is not fetched in this checkout")
    destination = tmp_path / pin.path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(real.read_bytes())

    monkeypatch.setattr(
        script, "fetch", lambda *a, **k: pytest.fail("must not download an already-pinned file")
    )
    assert script.fetch_pinned(roster, data_dir=tmp_path) == 0


def test_write_and_fetch_pinned_are_mutually_exclusive(script: ModuleType) -> None:
    """They express opposite intentions; silently preferring one would be a trap."""
    with pytest.raises(SystemExit):
        script.main(["--write", "--fetch-pinned"])


def test_the_nightly_ci_job_fetches_the_pinned_lexicons() -> None:
    """The regression this fixes: the job had no fetch step, so the probe test could not run."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "scripts/fetch_lexicons.py --fetch-pinned" in text
    # Only executed lines matter: the comment above the step legitimately explains why `--write`
    # is the wrong mode here, and a naive substring check would flag its own rationale.
    commands = [
        line.split("run:", 1)[1]
        for line in text.splitlines()
        if line.strip().startswith("run:") or " run: " in line
    ]
    assert not any("--write" in command for command in commands), "CI must never re-pin"
    # It has to come before the tests that need the file.
    assert text.index("fetch_lexicons.py --fetch-pinned") < text.index(
        "pytest (includes the inference-marked P5 gate tests)"
    )
