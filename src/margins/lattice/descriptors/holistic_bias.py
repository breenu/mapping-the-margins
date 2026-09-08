"""HolisticBias as a pinned descriptor/value source for the lattice (P3).

**HolisticBias is not a benchmark and must never be implemented as one.** It supplies identity
*descriptor terms* with provenance; it does not supply items to ingest. ``open_gen`` is a
generation path, and modelling this source as a ``BenchmarkLoader`` returning ``list[Item]``
would misfile the whole generation vertical (PROJECT_CONTEXT.md section 5). Nothing in this
module produces an :class:`~margins.schemas.Item`.

What it does provide:

* :data:`HOLISTIC_BIAS_V1_1` — the pinned release: repository path, commit revision, release
  label, license, and the SHA-256 of the exact file the lattice config's terms were taken from.
* :func:`load_descriptor_index` — parse a local copy of ``descriptors.json``, verifying that
  checksum first.
* :func:`verify_lattice_values` — check that every value declared in the lattice config really
  exists at its declared ``(source_axis, source_bucket, source_term)`` coordinates upstream, with
  the ``preference`` annotation the config claims.

The last one is the point. "Never invent a citation, statistic, or model result" needs a
mechanism for descriptors just as ``paper_numbers.json`` provides one for statistics: without it,
a plausible-looking term that upstream does not contain would travel unchallenged into the
prompt set and then into the paper. This turns that into a failing check.

The descriptors file itself lives under ``data/`` (gitignored and regenerable) and is fetched by
``scripts/fetch_holistic_bias.py``. Verification therefore runs wherever that file is present —
locally and in any reproduction — and is skipped in the hermetic CI job, which has no network.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from margins.lattice.config import LatticeValue

__all__ = [
    "DEFAULT_DESCRIPTOR_PATH",
    "HOLISTIC_BIAS_V1_1",
    "DescriptorIndex",
    "DescriptorTerm",
    "HolisticBiasRelease",
    "descriptor_path",
    "load_descriptor_index",
    "verify_lattice_values",
]


@dataclass(frozen=True, slots=True)
class HolisticBiasRelease:
    """A pinned HolisticBias descriptor release.

    Mirrors what a lattice config's ``sources`` entry must declare, so a test can assert the two
    agree and a drifting config cannot quietly claim a revision it was not built from.
    """

    source: str
    repo: str
    repo_path: str
    revision: str
    version: str
    license: str
    sha256: str

    @property
    def raw_url(self) -> str:
        """Revision-pinned raw URL. Pinned to the commit, never to a branch."""
        return f"https://raw.githubusercontent.com/{self.repo}/{self.revision}/{self.repo_path}"

    @property
    def blob_url(self) -> str:
        """Human-browsable, revision-pinned URL for the provenance table."""
        return f"https://github.com/{self.repo}/blob/{self.revision}/{self.repo_path}"


#: The release the shipped lattice config draws every value from. ``revision`` is the commit that
#: last touched this file; ``sha256`` is the checksum of its bytes.
HOLISTIC_BIAS_V1_1: Final[HolisticBiasRelease] = HolisticBiasRelease(
    source="HolisticBias descriptors (facebookresearch/ResponsibleNLP)",
    repo="facebookresearch/ResponsibleNLP",
    repo_path="holistic_bias/dataset/v1.1/descriptors.json",
    revision="6d76f31a6631aa882fd00fadb6dd8600577533d1",
    version="v1.1",
    license="MIT",
    sha256="c1328f0edfdeefb76302d360a01878df3db55f18026f4ea1956b4a0f52daf943",
)

#: Where ``scripts/fetch_holistic_bias.py`` puts the file, relative to the repository root.
DEFAULT_DESCRIPTOR_PATH: Final[Path] = Path("data/descriptors/holistic_bias/v1.1/descriptors.json")


def descriptor_path(repo_root: Path | None = None) -> Path:
    """Absolute path to the local descriptors file."""
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[4]
    return root / DEFAULT_DESCRIPTOR_PATH


@dataclass(frozen=True, slots=True)
class DescriptorTerm:
    """One upstream descriptor term with its annotations.

    ``preference`` is HolisticBias's own judgement of a term (``reviewed``, ``dispreferred``,
    ``polarizing``); terms carrying no annotation are recorded as ``"none"`` rather than dropped,
    so "unannotated" and "annotated as fine" stay distinguishable. ``gender`` marks terms that are
    gendered upstream (``Latina``, ``FTM``) — a non-gender axis must not use one, or the gender
    signal leaks into the axis and the non-gender x non-gender claim is no longer clean.
    """

    term: str
    preference: str
    gender: str | None


#: ``{(upstream_axis, bucket): terms}``.
DescriptorIndex = dict[tuple[str, str], tuple[DescriptorTerm, ...]]


def _parse_term(entry: Any) -> DescriptorTerm:
    """Upstream stores a term as either a bare string or an annotated object."""
    if isinstance(entry, str):
        return DescriptorTerm(term=entry, preference="none", gender=None)
    if isinstance(entry, dict):
        term = entry.get("descriptor")
        if not isinstance(term, str):
            raise ValueError(f"descriptor entry has no string 'descriptor' field: {entry!r}")
        preference = entry.get("preference", "none")
        gender = entry.get("gender")
        return DescriptorTerm(
            term=term,
            preference=preference if isinstance(preference, str) else "none",
            gender=gender if isinstance(gender, str) else None,
        )
    raise ValueError(f"unrecognised descriptor entry: {entry!r}")


def load_descriptor_index(
    path: Path | None = None,
    *,
    release: HolisticBiasRelease = HOLISTIC_BIAS_V1_1,
    verify_checksum: bool = True,
) -> DescriptorIndex:
    """Load and index the pinned descriptors file.

    Args:
        path: The local file. Defaults to :func:`descriptor_path`.
        release: The pin to check against.
        verify_checksum: Whether to require the file's SHA-256 to match the pin. Leave it on:
            a descriptors file that is *not* the pinned one makes every downstream provenance
            claim false, and that is worth an exception rather than a warning.

    Returns:
        ``{(upstream_axis, bucket): terms}``.

    Raises:
        FileNotFoundError: If the file is absent. Fetch it with
            ``scripts/fetch_holistic_bias.py``.
        ValueError: If the checksum does not match the pin, or the file is not shaped as
            ``{axis: {bucket: [term, ...]}}``.
    """
    target = path if path is not None else descriptor_path()
    if not target.is_file():
        raise FileNotFoundError(
            f"HolisticBias descriptors not found at {target}. Run "
            "`uv run python scripts/fetch_holistic_bias.py` to retrieve the pinned file."
        )

    raw = target.read_bytes()
    if verify_checksum:
        digest = hashlib.sha256(raw).hexdigest()
        if digest != release.sha256:
            raise ValueError(
                f"{target} has sha256 {digest}, expected {release.sha256} for "
                f"{release.version} at {release.revision}. This is not the pinned file; every "
                "provenance claim built on it would be wrong."
            )

    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{target} is not a JSON object")

    index: DescriptorIndex = {}
    for axis, buckets in data.items():
        if not isinstance(buckets, dict):
            raise ValueError(f"axis {axis!r} does not map buckets to term lists")
        for bucket, entries in buckets.items():
            if not isinstance(entries, list):
                raise ValueError(f"bucket {axis!r}/{bucket!r} is not a list of terms")
            index[(axis, bucket)] = tuple(_parse_term(entry) for entry in entries)
    return index


def verify_lattice_values(
    values: Sequence[LatticeValue],
    index: DescriptorIndex,
    *,
    source_key: str,
) -> None:
    """Check every value drawn from ``source_key`` against the upstream index.

    Verifies three things per value: the ``(source_axis, source_bucket)`` coordinates exist
    upstream, the exact ``source_term`` is present there, and the recorded ``preference`` matches
    upstream's annotation. Values from other sources are skipped — a custom value has a written
    justification instead, and checking it against HolisticBias would be meaningless.

    Raises:
        ValueError: Listing every mismatch found, rather than the first, so one run tells the
            human everything that needs fixing.
    """
    problems: list[str] = []
    for value in values:
        if value.source_key != source_key:
            continue
        coordinates = (value.source_axis, value.source_bucket)
        terms = index.get(coordinates)
        if terms is None:
            problems.append(
                f"{value.axis}={value.value!r}: upstream has no bucket "
                f"{value.source_axis}/{value.source_bucket}"
            )
            continue
        match = next((term for term in terms if term.term == value.source_term), None)
        if match is None:
            problems.append(
                f"{value.axis}={value.value!r}: term {value.source_term!r} is not in upstream "
                f"{value.source_axis}/{value.source_bucket} "
                f"(has {sorted(term.term for term in terms)})"
            )
            continue
        if match.preference != value.preference:
            problems.append(
                f"{value.axis}={value.value!r}: recorded preference {value.preference!r} but "
                f"upstream says {match.preference!r}"
            )
    if problems:
        raise ValueError(
            "lattice values do not match the pinned descriptor source:\n  " + "\n  ".join(problems)
        )
