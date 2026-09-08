"""Fetch the pinned HolisticBias descriptors file into ``data/``.

Kept out of the test suite on purpose: tests must not depend on the network, and the core CI job
has none. This script is what a reproducer runs once, after which
``scripts/build_lattice.py --verify`` can check every lattice value against the real file.

The download is pinned to a **commit**, never a branch, and the checksum is verified before the
file is written. A file that does not match the pin is not saved: a descriptors file that is not
the pinned one makes every provenance claim built on it false.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

from margins.lattice.descriptors import HOLISTIC_BIAS_V1_1, descriptor_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Destination file. Defaults to data/descriptors/holistic_bias/v1.1/descriptors.json",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if the file already verifies."
    )
    args = parser.parse_args(argv)

    release = HOLISTIC_BIAS_V1_1
    destination = args.out if args.out is not None else descriptor_path()

    if destination.is_file() and not args.force:
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest == release.sha256:
            print(f"already present and verified: {destination}")
            return 0
        print(f"present but sha256 {digest} != pin {release.sha256}; re-downloading")

    print(f"fetching {release.raw_url}")
    with urllib.request.urlopen(release.raw_url, timeout=60) as response:
        payload: bytes = response.read()

    digest = hashlib.sha256(payload).hexdigest()
    if digest != release.sha256:
        print(
            f"REFUSING TO WRITE: downloaded sha256 {digest} != pinned {release.sha256}. "
            "Upstream moved or the download was corrupted; do not proceed until this is "
            "resolved by a human.",
            file=sys.stderr,
        )
        return 1

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    print(f"wrote {destination} ({len(payload)} bytes, sha256 {digest})")
    print(f"pinned revision {release.revision} ({release.version}, {release.license})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
