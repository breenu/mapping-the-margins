"""Resolve and verify the pinned model revisions (P5 -- open decision #2).

**The one rule this script exists to enforce: a commit sha is RESOLVED, never authored.** Nothing in
this project may invent a revision. Every sha in ``configs/model/*.yaml`` is what
``huggingface_hub`` reported for that repo's head at pin time, and this script is both how it got
there (``--write``) and how a reproducer checks it is still what we claim (default ``--verify``).

**It stops rather than substituting.** If a repo 404s, or is gated and the caller has no accepted
licence, the script exits non-zero and says so. It never falls back to a different checkpoint, a
different revision, or an unpinned branch -- a silently substituted model would invalidate every
result built on it while leaving the manifest looking correct.

Requires the ``inference`` extra (``uv sync --extra dev --extra inference``) for
``huggingface_hub``. Deliberately NOT part of the test suite: tests must not depend on the network,
and the core CI job has none.

    uv run python scripts/pin_models.py             # verify every pin against upstream
    uv run python scripts/pin_models.py --write     # re-pin to current heads (a DELIBERATE act)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from margins.inference.models import (
    CI_MODEL,
    CONFIRMATORY_MODELS,
    DEV_MODEL,
    ModelSpec,
    default_model_config_dir,
    load_model_config,
)

#: Every model the project declares, in reporting order.
DECLARED_MODELS: tuple[str, ...] = (*CONFIRMATORY_MODELS, DEV_MODEL, CI_MODEL)

_REVISION_LINE = re.compile(r"^revision:\s*\S+\s*$", re.MULTILINE)


class AccessError(RuntimeError):
    """The repo could not be reached, or is gated and this account has no access."""


def resolve_head(spec: ModelSpec) -> str:
    """The repo's current head sha, via ``huggingface_hub``.

    Raises:
        AccessError: On any failure to reach the repo. The message names the STOP condition
            explicitly, because the correct response is to tell a human -- never to substitute.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.utils import (
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
    )

    try:
        info = HfApi().model_info(spec.model_id)
    except GatedRepoError as error:
        raise AccessError(
            f"{spec.model_id} is GATED and this account has no accepted licence. STOP: accept the "
            "licence on the model page with the account behind your HF token, then re-run. Do NOT "
            "substitute another model -- the pinned pair is a pre-registered design choice."
        ) from error
    except RepositoryNotFoundError as error:
        raise AccessError(
            f"{spec.model_id} does not exist (404). STOP and tell a human. Do NOT substitute "
            "another model: a silently swapped checkpoint invalidates every result built on it "
            "while leaving the manifest looking correct."
        ) from error
    except HfHubHTTPError as error:
        raise AccessError(f"{spec.model_id}: Hugging Face returned an error: {error}") from error

    sha = info.sha
    if not sha:
        raise AccessError(f"{spec.model_id}: the API returned no commit sha to pin to")
    return str(sha)


def rewrite_revision(path: Path, sha: str) -> None:
    """Replace the ``revision:`` line in a model config, leaving every comment intact."""
    text = path.read_text(encoding="utf-8")
    if not _REVISION_LINE.search(text):
        raise ValueError(f"{path} has no `revision:` line to rewrite")
    updated = _REVISION_LINE.sub(f"revision: {sha}", text, count=1)
    # newline="" keeps the LF endings .gitattributes pins, so the config's sha256 -- which the run
    # manifest records -- does not depend on which of the two locked environments wrote the file.
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(updated)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Re-pin each config to the repo's current head. A deliberate act, not a routine one.",
    )
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument(
        "--models", nargs="*", default=None, help=f"Subset of {list(DECLARED_MODELS)}."
    )
    args = parser.parse_args(argv)

    directory = args.config_dir or default_model_config_dir()
    names = tuple(args.models) if args.models else DECLARED_MODELS

    print(f"{'model':<24} {'role':<13} {'gated':<6} status")
    print("-" * 96)

    drifted: list[str] = []
    failed: list[str] = []
    for name in names:
        spec = load_model_config(name, config_dir=directory)
        try:
            head = resolve_head(spec)
        except AccessError as error:
            failed.append(name)
            print(f"{name:<24} {spec.role:<13} {spec.gated!s:<6} UNREACHABLE")
            print(f"  {error}", file=sys.stderr)
            continue

        if head == spec.revision:
            print(f"{name:<24} {spec.role:<13} {spec.gated!s:<6} pinned  {spec.revision}")
            continue

        drifted.append(name)
        path = directory / f"{name}.yaml"
        if args.write:
            rewrite_revision(path, head)
            print(f"{name:<24} {spec.role:<13} {spec.gated!s:<6} RE-PINNED")
            print(f"  {spec.revision} -> {head}")
        else:
            print(f"{name:<24} {spec.role:<13} {spec.gated!s:<6} DRIFTED")
            print(f"  pinned {spec.revision}")
            print(f"  head   {head}")

    if failed:
        print(
            f"\nSTOP: {len(failed)} model(s) could not be resolved: {failed}. "
            "Tell a human. Do not substitute a model and do not unpin.",
            file=sys.stderr,
        )
        return 2

    if drifted and not args.write:
        print(
            f"\n{len(drifted)} pin(s) differ from upstream head: {drifted}.\n"
            "This is INFORMATION, not necessarily a problem -- a pin is supposed to hold still "
            "while upstream moves. Re-pin only as a deliberate, logged decision (`--write`), and "
            "note that any result produced under the old revision was produced under the OLD "
            "model.",
        )
        return 1

    if not drifted:
        print("\nevery pin matches upstream head")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
