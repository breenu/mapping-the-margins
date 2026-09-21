"""Fetch and pin the CrowS-Pairs dataset into ``data/`` (P2).

The mirror of ``scripts/fetch_lexicons.py`` and ``scripts/fetch_holistic_bias.py``, following the
same discipline for the same reason: **the download is pinned to a commit, never a branch, and the
bytes are verified before anything is written.** A benchmark that changed under us would change
every extractive effect computed from it, silently, while the pipeline stayed green.

Three modes:

* ``--verify`` (default) -- check the file on disk against the sha256 and row count the config
  records. This is what a reproducer runs.
* ``--fetch-pinned`` -- download at the revision the config ALREADY records, verify the bytes, and
  only then write. This is what CI and a fresh machine run. A 404 or a digest mismatch **stops**
  rather than substituting another source or revision.
* ``--write`` -- resolve the upstream commit at HEAD, download at that commit, and print the YAML
  block and the Python constant to paste. **It prints rather than edits**: a pin is a logged
  decision, and a script that silently rewrote the config would make re-pinning invisible in a
  diff. This mode RE-PINS, so it is a human action.

Kept out of the test suite's network path on purpose -- tests monkeypatch :func:`fetch` and the
core CI job has no network at all.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

from margins.benchmarks.config import DEFAULT_BENCHMARK, BenchmarkConfig, load_benchmark_config
from margins.benchmarks.crows_pairs import CROWS_PAIRS_RELEASE, crows_pairs_path


class BenchmarkFile(NamedTuple):
    """Where an extractive source's bytes come from, and where they land."""

    name: str
    repo: str
    path_in_repo: str
    license: str


#: The one extractive source in the confirmatory scope. WinoBias, StereoSet and BBQ are out of
#: scope unless ahead of schedule (PROJECT_CONTEXT.md section 1); each would add an entry here.
SOURCE: BenchmarkFile = BenchmarkFile(
    name="crows-pairs",
    repo=CROWS_PAIRS_RELEASE.repo,
    path_in_repo=CROWS_PAIRS_RELEASE.repo_path,
    license=CROWS_PAIRS_RELEASE.license,
)

_API = "https://api.github.com/repos/{repo}/commits?per_page=1"
_RAW = "https://raw.githubusercontent.com/{repo}/{revision}/{path}"


def resolve_head(repo: str, *, timeout: int = 60) -> str:
    """The current HEAD commit sha of a GitHub repo.

    Resolved, never authored -- the same rule ``ModelSpec.revision`` enforces for checkpoints. The
    sha this returns is what gets pinned; ``master`` itself is never recorded, because it moves.
    """
    request = urllib.request.Request(
        _API.format(repo=repo), headers={"Accept": "application/vnd.github+json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    sha = str(payload[0]["sha"])
    if len(sha) != 40:  # pragma: no cover - defensive
        raise ValueError(f"resolved revision for {repo} is not a 40-char sha: {sha!r}")
    return sha


def fetch(source: BenchmarkFile, revision: str, *, timeout: int = 60) -> bytes:
    """Download the source file at a pinned revision."""
    url = _RAW.format(repo=source.repo, revision=revision, path=source.path_in_repo)
    print(f"fetching {url}")
    with urllib.request.urlopen(url, timeout=timeout) as response:
        data: bytes = response.read()
    return data


def _normalize(payload: bytes) -> bytes:
    """LF endings and a single trailing newline, so the digest is platform-independent.

    Without this the sha256 of the same CSV differs between a Windows and a Linux checkout, and
    the pin would fail verification on one of the two locked environments.
    """
    text = payload.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return (text.rstrip("\n") + "\n").encode("utf-8")


def _count_rows(payload: bytes) -> int:
    """Data rows in the CSV, excluding the header. Counted, never assumed."""
    return sum(1 for _ in csv.DictReader(io.StringIO(payload.decode("utf-8"))))


def verify(config: BenchmarkConfig, *, data_dir: Path) -> int:
    """Check the file on disk against the recorded pin. Returns a process exit code."""
    path = crows_pairs_path(data_dir)
    if not path.is_file():
        print(
            f"FAIL {config.benchmark}: {path} is missing. Run with --fetch-pinned to retrieve it "
            "at the recorded revision.",
            file=sys.stderr,
        )
        return 1

    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != config.source.sha256:
        print(
            f"FAIL {config.benchmark}: {path} has sha256 {digest}, the config pins "
            f"{config.source.sha256}. The file on disk is NOT the one these effects were defined "
            "against.",
            file=sys.stderr,
        )
        return 1

    rows = _count_rows(payload)
    if rows != config.expected_rows:
        print(
            f"FAIL {config.benchmark}: {path} holds {rows} rows, the config records "
            f"{config.expected_rows}.",
            file=sys.stderr,
        )
        return 1

    print(f"ok   {config.benchmark:<16} {rows} rows  {digest[:12]}  {path}")
    return 0


def fetch_pinned(config: BenchmarkConfig, *, data_dir: Path, timeout: int = 60) -> int:
    """Download the source AT ITS RECORDED REVISION and verify before writing.

    The distinction from :func:`write` is the whole point. ``write`` resolves upstream HEAD, which
    is correct when a human is deliberately re-pinning and wrong everywhere else; this fetches the
    revision the config already names, so a machine provisioned by it holds exactly the bytes the
    recorded effects were defined against.
    """
    revision = config.source.revision
    if revision is None:
        print(
            f"FAIL {config.benchmark}: the config records no revision, so there is no pinned "
            "version to fetch. Re-pin with --write first.",
            file=sys.stderr,
        )
        return 1

    destination = crows_pairs_path(data_dir)
    if destination.is_file():
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest == config.source.sha256:
            print(f"ok   {config.benchmark:<16} already present and pinned  {destination}")
            return verify(config, data_dir=data_dir)

    try:
        payload = _normalize(fetch(SOURCE, revision, timeout=timeout))
    except (urllib.error.HTTPError, urllib.error.URLError) as error:
        print(
            f"STOPPING: could not fetch {SOURCE.name} at the pinned revision {revision}: {error}. "
            "Refusing to substitute another source or revision.",
            file=sys.stderr,
        )
        return 1

    digest = hashlib.sha256(payload).hexdigest()
    if digest != config.source.sha256:
        print(
            f"STOPPING: {SOURCE.name} at pinned revision {revision} has sha256 {digest}, but the "
            f"config pins {config.source.sha256}. NOT writing it. Upstream rewrote history at that "
            "revision, or the config is wrong; either way this is a human decision.",
            file=sys.stderr,
        )
        return 1

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    print(f"wrote {destination}  (sha256 {digest[:12]}, pinned revision {revision[:12]})")
    return verify(config, data_dir=data_dir)


def write(data_dir: Path, *, timeout: int = 60) -> int:
    """Resolve, download and digest the source, then print the pin blocks to paste."""
    try:
        revision = resolve_head(SOURCE.repo, timeout=timeout)
    except (urllib.error.HTTPError, urllib.error.URLError) as error:
        print(
            f"STOPPING: could not resolve a revision for {SOURCE.repo}: {error}. Refusing to "
            "substitute another source or to leave the pin unresolved -- surface this to a human "
            "(CLAUDE.md: never work around a blocked domain).",
            file=sys.stderr,
        )
        return 1

    try:
        payload = _normalize(fetch(SOURCE, revision, timeout=timeout))
    except (urllib.error.HTTPError, urllib.error.URLError) as error:
        print(f"STOPPING: could not fetch {SOURCE.name} at {revision}: {error}", file=sys.stderr)
        return 1

    destination = crows_pairs_path(data_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    rows = _count_rows(payload)

    print(f"wrote {destination}  ({rows} rows, sha256 {digest[:12]})")
    print(
        "\nPaste into configs/benchmark/crows_pairs_v1.yaml (a pin is a LOGGED decision, so this "
        "script\nprints rather than edits -- a silent rewrite would make a re-pin invisible in a "
        "diff):\n"
    )
    print(
        "\n".join(
            [
                "source:",
                f"  source: {CROWS_PAIRS_RELEASE.source}",
                f"  url: https://github.com/{SOURCE.repo}/blob/{revision}/{SOURCE.path_in_repo}",
                f"  path: {SOURCE.path_in_repo}",
                f'  revision: "{revision}"',
                f'  version: "{CROWS_PAIRS_RELEASE.version}"',
                f"  license: {SOURCE.license}",
                f'  sha256: "{digest}"',
                "  retrieved_at: <ISO-8601 UTC>",
                "",
                f"expected_rows: {rows}",
            ]
        )
    )
    print("\nAnd into margins/benchmarks/crows_pairs.py's CROWS_PAIRS_RELEASE:\n")
    print(f'    revision="{revision}",')
    print(f'    sha256="{digest}",')
    print(f"    rows={rows},")
    print(
        "\nNOTE: the LICENCE is not resolved by this script. Read it off the upstream repository "
        "and\nrecord what it actually says -- a licence nobody checked is a licence that blocks "
        "the release."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK, help="Benchmark config name.")
    parser.add_argument("--config-dir", type=Path, default=None, help="Benchmark config directory.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Where the CSV lives.")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check the file on disk against the recorded pin. The default; accepted explicitly "
        "so the documented invocation and the implemented one are the same string.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Resolve upstream HEAD, download, and print the pin blocks. RE-PINS: a human action. "
        "Default is --verify.",
    )
    parser.add_argument(
        "--fetch-pinned",
        action="store_true",
        help="Download AT THE REVISION THE CONFIG RECORDS and verify before writing. What CI "
        "runs; never re-pins.",
    )
    args = parser.parse_args(argv)

    if args.write and args.fetch_pinned:
        parser.error(
            "--write and --fetch-pinned are different intentions: --write RE-PINS to upstream "
            "HEAD, --fetch-pinned materialises the pin already recorded. Choose one."
        )
    if args.verify and (args.write or args.fetch_pinned):
        parser.error("--verify is the default and cannot be combined with another mode.")
    if args.write:
        return write(args.data_dir)

    config = load_benchmark_config(args.benchmark, config_dir=args.config_dir)
    if args.fetch_pinned:
        return fetch_pinned(config, data_dir=args.data_dir)
    return verify(config, data_dir=args.data_dir)


if __name__ == "__main__":
    raise SystemExit(main())
