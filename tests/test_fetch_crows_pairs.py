"""The CrowS-Pairs fetch script's three modes, without touching the network (P2).

The same discipline ``tests/test_fetch_lexicons.py`` applies to the scoring wordlists, for the same
reason: *the download is pinned to a commit, never a branch, and the bytes are verified before
anything is written*. A benchmark that changed under us would change every extractive effect
computed from it, silently, while the pipeline stayed green.

The script is loaded **by path** -- ``scripts/`` is not an importable package -- and every network
call is monkeypatched. The sharpest test makes ``resolve_head`` explode: ``--fetch-pinned`` must
never consult upstream HEAD, because doing so would fetch whatever the dataset looks like today
and score against a revision nobody recorded.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from margins.benchmarks import crows_pairs_path, load_benchmark_config

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_crows_pairs.py"
FIXTURE_CSV = Path(__file__).parent / "fixtures" / "benchmarks" / "crows_eyeball.csv"
FIXTURE_CONFIG_DIR = Path(__file__).parent / "fixtures" / "benchmark"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fetch_crows_pairs", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def config():  # type: ignore[no-untyped-def]
    return load_benchmark_config("tiny", config_dir=FIXTURE_CONFIG_DIR)


def _place(data_dir: Path) -> Path:
    destination = crows_pairs_path(data_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(FIXTURE_CSV.read_bytes())
    return destination


# --------------------------------------------------------------------------------------
# --verify
# --------------------------------------------------------------------------------------


def test_verify_passes_on_the_pinned_bytes(script: ModuleType, config: Any, tmp_path: Path) -> None:
    _place(tmp_path)
    assert script.verify(config, data_dir=tmp_path) == 0


def test_verify_fails_on_a_digest_mismatch(script: ModuleType, config: Any, tmp_path: Path) -> None:
    path = _place(tmp_path)
    path.write_bytes(path.read_bytes() + b"7,a,b,stereo,age,x,a0,x\n")
    assert script.verify(config, data_dir=tmp_path) == 1


def test_verify_fails_when_the_file_is_missing(
    script: ModuleType, config: Any, tmp_path: Path
) -> None:
    assert script.verify(config, data_dir=tmp_path) == 1


# --------------------------------------------------------------------------------------
# --fetch-pinned
# --------------------------------------------------------------------------------------


def test_fetch_pinned_downloads_at_the_pin_and_never_resolves_head(
    script: ModuleType, config: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The distinction that matters: `--write` resolves HEAD, `--fetch-pinned` must not.

    Using HEAD to provision a machine would silently fetch whatever upstream looks like today and
    score against a revision nobody pinned.
    """
    seen: dict[str, str] = {}

    def fake_fetch(source: Any, revision: str, **_: Any) -> bytes:
        seen["revision"] = revision
        return FIXTURE_CSV.read_bytes()

    def explode(*_: Any, **__: Any) -> str:
        raise AssertionError("--fetch-pinned must never resolve upstream HEAD")

    monkeypatch.setattr(script, "fetch", fake_fetch)
    monkeypatch.setattr(script, "resolve_head", explode)

    assert script.fetch_pinned(config, data_dir=tmp_path) == 0
    assert seen["revision"] == config.source.revision
    assert crows_pairs_path(tmp_path).is_file()


def test_fetch_pinned_refuses_to_write_on_a_digest_mismatch(
    script: ModuleType, config: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing is written unless the bytes match. A mismatch is a human decision, not a retry."""
    monkeypatch.setattr(script, "fetch", lambda *a, **k: b"not the pinned dataset\n")

    assert script.fetch_pinned(config, data_dir=tmp_path) == 1
    assert not crows_pairs_path(tmp_path).exists(), "a rejected download must leave no file"


def test_fetch_pinned_stops_rather_than_substituting_on_a_404(
    script: ModuleType, config: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLAUDE.md: never work around a blocked domain; surface it."""
    import urllib.error

    def not_found(*_: Any, **__: Any) -> bytes:
        raise urllib.error.HTTPError(
            url="https://example.invalid", code=404, msg="Not Found", hdrs=None, fp=None
        )

    monkeypatch.setattr(script, "fetch", not_found)
    assert script.fetch_pinned(config, data_dir=tmp_path) == 1
    assert not crows_pairs_path(tmp_path).exists()


def test_an_already_pinned_file_is_not_re_downloaded(
    script: ModuleType, config: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _place(tmp_path)
    monkeypatch.setattr(
        script, "fetch", lambda *a, **k: pytest.fail("must not re-download a pinned file")
    )
    assert script.fetch_pinned(config, data_dir=tmp_path) == 0


def test_the_real_pin_round_trips_against_the_committed_config(
    script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shipped pin, replayed from the local checkout instead of from the network."""
    real = crows_pairs_path()
    if not real.is_file():
        pytest.skip("pinned crows_pairs_anonymized.csv absent; run scripts/fetch_crows_pairs.py")

    monkeypatch.setattr(script, "fetch", lambda *a, **k: real.read_bytes())
    assert script.fetch_pinned(load_benchmark_config(), data_dir=tmp_path) == 0
    assert (
        hashlib.sha256(crows_pairs_path(tmp_path).read_bytes()).hexdigest()
        == load_benchmark_config().source.sha256
    )


# --------------------------------------------------------------------------------------
# Mode exclusivity, and byte normalization
# --------------------------------------------------------------------------------------


def test_write_and_fetch_pinned_are_mutually_exclusive(script: ModuleType) -> None:
    with pytest.raises(SystemExit):
        script.main(["--write", "--fetch-pinned"])


def test_verify_cannot_be_combined_with_another_mode(script: ModuleType) -> None:
    with pytest.raises(SystemExit):
        script.main(["--verify", "--write"])


def test_normalization_makes_the_digest_platform_independent(script: ModuleType) -> None:
    """Without this the same CSV digests differently on the two locked environments, and the pin
    would fail verification on one of them."""
    crlf = b"a,b\r\n1,2\r\n\r\n"
    lf = b"a,b\n1,2\n"
    assert script._normalize(crlf) == lf
    assert script._normalize(lf) == lf


def test_row_counting_excludes_the_header(script: ModuleType) -> None:
    assert script._count_rows(FIXTURE_CSV.read_bytes()) == 7


def test_the_source_table_matches_the_release_constant(script: ModuleType) -> None:
    from margins.benchmarks import CROWS_PAIRS_RELEASE

    assert script.SOURCE.repo == CROWS_PAIRS_RELEASE.repo
    assert script.SOURCE.path_in_repo == CROWS_PAIRS_RELEASE.repo_path
