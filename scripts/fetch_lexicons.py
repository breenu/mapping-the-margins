"""Fetch and pin the scoring lexicons into ``data/`` (P6).

The mirror of ``scripts/fetch_holistic_bias.py``, and it follows the same discipline for the same
reason: **the download is pinned to a commit, never a branch, and the bytes are verified before
anything is written.** A wordlist that changed under us would change every toxicity score computed
from it, silently, while the pipeline stayed green.

Two modes, matching ``scripts/pin_models.py``:

* ``--verify`` (default) -- check that every lexicon declared in the roster is present on disk and
  matches its recorded sha256. This is what a reproducer runs.
* ``--write`` -- resolve the upstream commit at HEAD, download at that commit, and print the YAML
  block to paste into ``configs/scoring/<roster>.yaml``. **It prints rather than edits**: a pin is
  a logged decision, and a script that silently rewrote the roster would make re-pinning invisible
  in a diff. This mode RE-PINS, so it is a human action.
* ``--fetch-pinned`` -- download each lexicon at the revision the roster ALREADY records, verify
  the bytes against the recorded sha256, and only then write. This is what CI runs. It is
  deliberately a separate mode from ``--write``: ``--write`` resolves HEAD, so using it to
  populate a machine would silently fetch whatever upstream looks like today and score against a
  wordlist nobody pinned. A digest mismatch here STOPS rather than writing (CLAUDE.md: never
  unpin to make it work).

Kept out of the test suite on purpose -- tests must not depend on the network, and the core CI job
has none.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

from margins.scoring.config import DEFAULT_SCORING, ScoringConfig, load_scoring_config


class LexiconSource(NamedTuple):
    """Where a lexicon's bytes come from, and where they land."""

    scorer_id: str
    name: str
    repo: str
    path_in_repo: str
    destination: Path
    license: str


#: Every fetchable lexicon the project uses. Keyed by scorer_id so the roster and this table cannot
#: drift apart silently -- ``--verify`` cross-checks them and fails if the roster names a fetched
#: lexicon that is not here.
SOURCES: tuple[LexiconSource, ...] = (
    LexiconSource(
        scorer_id="ldnoobw_lexicon",
        name="ldnoobw-en",
        repo="LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words",
        path_in_repo="en",
        destination=Path("data") / "lexicons" / "ldnoobw_en.txt",
        license="CC-BY-4.0",
    ),
)

_API = "https://api.github.com/repos/{repo}/commits?per_page=1"
_RAW = "https://raw.githubusercontent.com/{repo}/{revision}/{path}"


def resolve_head(repo: str, *, timeout: int = 60) -> str:
    """The current HEAD commit sha of a GitHub repo.

    Resolved, never authored -- the same rule ``ModelSpec.revision`` enforces for checkpoints. The
    sha this returns is what gets pinned; ``main`` itself is never recorded, because it moves.
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


def fetch(source: LexiconSource, revision: str, *, timeout: int = 60) -> bytes:
    """Download one lexicon at a pinned revision."""
    url = _RAW.format(repo=source.repo, revision=revision, path=source.path_in_repo)
    print(f"fetching {url}")
    with urllib.request.urlopen(url, timeout=timeout) as response:
        data: bytes = response.read()
    return data


def _normalize(payload: bytes) -> bytes:
    """LF endings and a single trailing newline, so the digest is platform-independent.

    Without this the sha256 of the same wordlist differs between a Windows and a Linux checkout,
    and the pin would fail verification on one of the two locked environments.
    """
    text = payload.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return (text.rstrip("\n") + "\n").encode("utf-8")


def _sources_by_scorer() -> dict[str, LexiconSource]:
    return {source.scorer_id: source for source in SOURCES}


def verify(config: ScoringConfig, *, data_dir: Path) -> int:
    """Check every fetched lexicon in the roster against its pin. Returns a process exit code."""
    known = _sources_by_scorer()
    failures = 0
    checked = 0

    for spec in config.scorers:
        pin = spec.lexicon_pin
        if pin is None or pin.path is None:
            continue  # packaged lexicon (VADER): pinned by package version, nothing to fetch.
        checked += 1

        if spec.scorer_id not in known:
            print(
                f"FAIL {spec.scorer_id}: roster declares a fetched lexicon this script does not "
                "know how to retrieve. Add it to SOURCES rather than fetching it by hand.",
                file=sys.stderr,
            )
            failures += 1
            continue

        path = pin.path if pin.path.is_absolute() else data_dir / pin.path
        if not path.is_file():
            print(
                f"FAIL {spec.scorer_id}: {path} is missing. Run with --write to fetch it.",
                file=sys.stderr,
            )
            failures += 1
            continue

        if pin.sha256 is None:
            print(
                f"FAIL {spec.scorer_id}: roster records no sha256 for {pin.name}", file=sys.stderr
            )
            failures += 1
            continue

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != pin.sha256:
            print(
                f"FAIL {spec.scorer_id}: {path} has sha256 {digest}, roster pins {pin.sha256}. "
                "The file on disk is NOT the one these scores were defined against.",
                file=sys.stderr,
            )
            failures += 1
            continue

        print(f"ok   {spec.scorer_id:<20} {pin.name:<16} {digest[:12]}  {path}")

    if checked == 0:
        print("no fetched lexicons in this roster (all packaged); nothing to verify")
    return 1 if failures else 0


def fetch_pinned(config: ScoringConfig, *, data_dir: Path, timeout: int = 60) -> int:
    """Download every fetched lexicon AT ITS RECORDED REVISION and verify before writing.

    The distinction from :func:`write` is the whole point. ``write`` resolves upstream HEAD, which
    is correct when a human is deliberately re-pinning and wrong everywhere else; this fetches the
    revision the roster already names, so a machine provisioned by it holds exactly the bytes the
    recorded scores were defined against.

    Nothing is written unless its sha256 matches the pin. A mismatch means upstream rewrote
    history at that revision, or the roster is wrong, and either way the honest response is to
    stop and surface it rather than to score against a wordlist that is not the pinned one.
    """
    known = _sources_by_scorer()
    failures = 0
    fetched = 0

    for spec in config.scorers:
        pin = spec.lexicon_pin
        if pin is None or pin.path is None:
            continue  # packaged lexicon (VADER): pinned by package version, nothing to fetch.
        source = known.get(spec.scorer_id)
        if source is None:
            print(
                f"FAIL {spec.scorer_id}: roster declares a fetched lexicon this script does not "
                "know how to retrieve. Add it to SOURCES rather than fetching it by hand.",
                file=sys.stderr,
            )
            failures += 1
            continue
        if pin.revision is None or pin.sha256 is None:
            print(
                f"FAIL {spec.scorer_id}: the roster records no revision/sha256 for {pin.name}, so "
                "there is no pinned version to fetch. Re-pin with --write first.",
                file=sys.stderr,
            )
            failures += 1
            continue

        destination = pin.path if pin.path.is_absolute() else data_dir / pin.path
        if destination.is_file():
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            if digest == pin.sha256:
                print(f"ok   {spec.scorer_id:<20} already present and pinned  {destination}")
                continue

        try:
            payload = _normalize(fetch(source, pin.revision, timeout=timeout))
        except (urllib.error.HTTPError, urllib.error.URLError) as error:
            print(
                f"STOPPING: could not fetch {source.name} at the pinned revision "
                f"{pin.revision}: {error}. Refusing to substitute another source or revision.",
                file=sys.stderr,
            )
            return 1

        digest = hashlib.sha256(payload).hexdigest()
        if digest != pin.sha256:
            print(
                f"STOPPING: {source.name} at pinned revision {pin.revision} has sha256 {digest}, "
                f"but the roster pins {pin.sha256}. NOT writing it. Upstream rewrote history at "
                "that revision, or the roster is wrong; either way this is a human decision.",
                file=sys.stderr,
            )
            return 1

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        fetched += 1
        print(f"wrote {destination}  (sha256 {digest[:12]}, pinned revision {pin.revision[:12]})")

    if failures:
        return 1
    print(f"{fetched} lexicon(s) fetched at their pinned revisions; all digests match")
    return 0


def write(data_dir: Path, *, timeout: int = 60) -> int:
    """Resolve, download and digest every lexicon, then print the YAML to paste into the roster."""
    blocks: list[str] = []

    for source in SOURCES:
        try:
            revision = resolve_head(source.repo, timeout=timeout)
        except (urllib.error.HTTPError, urllib.error.URLError) as error:
            print(
                f"STOPPING: could not resolve a revision for {source.repo}: {error}. "
                "Refusing to substitute another source or to leave the pin unresolved -- surface "
                "this to a human (CLAUDE.md: never work around a blocked domain).",
                file=sys.stderr,
            )
            return 1

        try:
            payload = _normalize(fetch(source, revision, timeout=timeout))
        except (urllib.error.HTTPError, urllib.error.URLError) as error:
            print(
                f"STOPPING: could not fetch {source.name} at {revision}: {error}", file=sys.stderr
            )
            return 1

        destination = (
            source.destination
            if source.destination.is_absolute()
            else data_dir / source.destination.relative_to("data")
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        entries = len([line for line in payload.decode("utf-8").splitlines() if line.strip()])

        print(f"wrote {destination}  ({entries} entries, sha256 {digest[:12]})")
        blocks.append(
            "\n".join(
                [
                    f"    # {source.scorer_id}",
                    "    lexicon_pin:",
                    f"      name: {source.name}",
                    f"      source: https://github.com/{source.repo}",
                    f'      revision: "{revision}"',
                    f"      license: {source.license}",
                    f'      sha256: "{digest}"',
                    f"      path: {destination.relative_to(data_dir).as_posix()}",
                ]
            )
        )

    print("\nPaste into configs/scoring/<roster>.yaml (a pin is a LOGGED decision, so this script")
    print(
        "prints rather than edits -- a silent rewrite would make a re-pin invisible in a diff):\n"
    )
    print("\n\n".join(blocks))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scoring", default=DEFAULT_SCORING, help="Scoring roster config name.")
    parser.add_argument("--config-dir", type=Path, default=None, help="Scoring config directory.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Where lexicons live.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Resolve upstream HEAD, download, and print the YAML pin block. RE-PINS: a human "
        "action. Default is --verify.",
    )
    parser.add_argument(
        "--fetch-pinned",
        action="store_true",
        help="Download each lexicon AT THE REVISION THE ROSTER RECORDS and verify before "
        "writing. What CI runs; never re-pins.",
    )
    args = parser.parse_args(argv)

    if args.write and args.fetch_pinned:
        parser.error(
            "--write and --fetch-pinned are different intentions: --write RE-PINS to upstream "
            "HEAD, --fetch-pinned materialises the pin already recorded. Choose one."
        )
    if args.write:
        return write(args.data_dir)

    config = load_scoring_config(args.scoring, config_dir=args.config_dir)
    if args.fetch_pinned:
        code = fetch_pinned(config, data_dir=args.data_dir)
        if code != 0:
            return code
    return verify(config, data_dir=args.data_dir)


if __name__ == "__main__":
    raise SystemExit(main())
