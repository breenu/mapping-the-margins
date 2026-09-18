"""Resolve and verify the pinned scorer-classifier revisions (P6).

The direct analogue of ``scripts/pin_models.py``, and it enforces the same one rule: **a commit sha
is RESOLVED, never authored.** Every revision in ``configs/scoring/*.yaml`` is what
``huggingface_hub`` reported for that repo at pin time, and this script is both how it got there
(``--write``) and how a reproducer checks it is still what we claim (default ``--verify``).

It also checks something ``pin_models.py`` does not need to: **that the configured label exists on
the checkpoint**. A scorer reads one named output, and label order is a property of the checkpoint
that can differ between revisions. A re-pin that quietly changed which column ``negative`` refers to
would keep producing numbers under the same ``scorer_id`` while measuring a different construct.

**It stops rather than substituting.** A 404, a gated repo without access, or a missing label exits
non-zero. There is no fallback classifier: the roster is a construct choice, and swapping one leg of
a triangulated pair for whatever is reachable would silently break the independence the pair exists
to provide.

Requires the ``inference`` extra for ``huggingface_hub``. Deliberately NOT part of the test suite --
tests must not depend on the network, and the core CI job has none.

    uv run python scripts/pin_scorers.py            # verify every pin against upstream
    uv run python scripts/pin_scorers.py --write    # re-pin to current heads (a DELIBERATE act)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from margins.scoring.config import DEFAULT_SCORING, ScorerSpec, ScoringConfig, load_scoring_config

#: An unresolved placeholder. All-zero on purpose: obviously not a real sha, where a plausible
#: hand-typed one would be indistinguishable from a resolved pin.
PLACEHOLDER_REVISION: str = "0" * 40


class AccessError(RuntimeError):
    """The repo could not be reached, or is gated and this account has no access."""


def resolve_head(repo: str) -> str:
    """The repo's current head sha, via ``huggingface_hub``."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    try:
        info = HfApi().model_info(repo)
    except GatedRepoError as error:
        raise AccessError(
            f"{repo} is GATED and this account has no accepted licence. STOP: accept the licence "
            "with the account behind your HF token, then re-run. Do NOT substitute another "
            "classifier -- the neural/rule-based pairing is a construct choice, not a default."
        ) from error
    except RepositoryNotFoundError as error:
        raise AccessError(
            f"{repo} does not exist (404). STOP and tell a human. Do NOT substitute another "
            "classifier: a silently swapped scorer invalidates every score built on it while "
            "leaving the manifest looking correct."
        ) from error
    except HfHubHTTPError as error:
        raise AccessError(f"could not reach {repo}: {error}") from error

    sha = info.sha
    if not sha:  # pragma: no cover - defensive
        raise AccessError(f"{repo} returned no revision sha")
    return str(sha)


def check_label(repo: str, revision: str, label: str) -> list[str]:
    """Return the checkpoint's labels, raising if the configured one is not among them."""
    import json

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, filename="config.json", revision=revision)
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    id2label = config.get("id2label") or {}
    labels = [str(value) for value in id2label.values()]

    if not any(name.lower() == label.lower() for name in labels):
        raise AccessError(
            f"{repo}@{revision[:12]} does not expose label {label!r}. Available: {sorted(labels)}. "
            "STOP: a substituted label would measure a different construct under the same "
            "scorer_id."
        )
    return labels


def _neural_scorers(config: ScoringConfig) -> list[ScorerSpec]:
    return [spec for spec in config.scorers if spec.model_pin is not None]


def verify(config: ScoringConfig, *, check_labels: bool = True) -> int:
    """Check every declared revision against upstream. Returns a process exit code."""
    failures = 0

    for spec in _neural_scorers(config):
        pin = spec.model_pin
        assert pin is not None  # narrowed by _neural_scorers

        if pin.revision == PLACEHOLDER_REVISION:
            print(
                f"FAIL {spec.scorer_id}: revision is the unresolved placeholder. Run --write. "
                "Nothing may score against a placeholder pin.",
                file=sys.stderr,
            )
            failures += 1
            continue

        try:
            head = resolve_head(pin.repo)
        except AccessError as error:
            print(f"FAIL {spec.scorer_id}: {error}", file=sys.stderr)
            failures += 1
            continue

        moved = "" if head == pin.revision else f"  (upstream head is now {head[:12]})"
        print(f"ok   {spec.scorer_id:<20} {pin.repo:<28} {pin.revision[:12]}{moved}")

        if check_labels:
            try:
                check_label(pin.repo, pin.revision, pin.label)
            except AccessError as error:
                print(f"FAIL {spec.scorer_id}: {error}", file=sys.stderr)
                failures += 1
            except Exception as error:
                print(
                    f"FAIL {spec.scorer_id}: could not read config.json: {error}", file=sys.stderr
                )
                failures += 1

    if not _neural_scorers(config):
        print("no neural scorers in this roster; nothing to pin")
    return 1 if failures else 0


def write(config: ScoringConfig) -> int:
    """Resolve every head and print the YAML to paste into the roster.

    Prints rather than edits, exactly as ``scripts/fetch_lexicons.py`` does: a re-pin is a logged
    decision, and any score produced under the old revision was produced by a different instrument.
    """
    blocks: list[str] = []

    for spec in _neural_scorers(config):
        pin = spec.model_pin
        assert pin is not None

        try:
            head = resolve_head(pin.repo)
            labels = check_label(pin.repo, head, pin.label)
        except AccessError as error:
            print(f"STOPPING: {error}", file=sys.stderr)
            return 1

        print(f"{spec.scorer_id:<20} {pin.repo:<28} -> {head}")
        print(f"{'':<20} labels: {sorted(labels)}")
        blocks.append(
            "\n".join(
                [
                    f"    # {spec.scorer_id}",
                    "    model_pin:",
                    f"      repo: {pin.repo}",
                    f'      revision: "{head}"',
                    f"      license: {pin.license}",
                    f"      label: {pin.label}",
                    f"      max_input_tokens: {pin.max_input_tokens}",
                ]
            )
        )

    print("\nPaste into configs/scoring/<roster>.yaml (a re-pin is a LOGGED decision -- any score")
    print("produced under the old revision was produced by a different instrument):\n")
    print("\n\n".join(blocks))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scoring", default=DEFAULT_SCORING, help="Scoring roster config name.")
    parser.add_argument("--config-dir", type=Path, default=None, help="Scoring config directory.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Resolve current heads and print the YAML pin block. Default is --verify.",
    )
    parser.add_argument(
        "--no-label-check",
        action="store_true",
        help="Skip downloading config.json to check the configured label exists.",
    )
    args = parser.parse_args(argv)

    config = load_scoring_config(args.scoring, config_dir=args.config_dir)
    if args.write:
        return write(config)
    return verify(config, check_labels=not args.no_label_check)


if __name__ == "__main__":
    raise SystemExit(main())
