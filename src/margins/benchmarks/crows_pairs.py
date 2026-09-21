"""The CrowS-Pairs source: its pin, its reader, and its loader (P2).

CrowS-Pairs (Nangia et al., EMNLP 2020) is the project's **extractive** paradigm -- the second of
the two sources RQ1 compares. Each row is a minimal sentence pair differing in one identity
attribute, and the measurement is which member the model finds more likely.

**Two construct decisions are made here and must travel with every number built on them.**

1. **This is autoregressive sentence log-likelihood, not masked pseudo-log-likelihood.** The
   original metric conditions on the unmodified tokens and masks the modified ones, which a
   decoder-only model cannot do. Llama and Mistral are decoder-only, so a whole-sequence
   autoregressive log-likelihood is used instead. PROJECT_CONTEXT.md section 9 names this as a
   caveat to state, not a detail to absorb; ``docs/methods.md`` states it.
2. **The stereo/anti-stereo direction is resolved HERE, once.** The upstream ``sent_more`` column
   is *not* always the stereotyping sentence: on rows where ``stereo_antistereo == "antistereo"``
   it is the anti-stereotypical one. :class:`CrowsRow` therefore exposes derived
   :attr:`~CrowsRow.sent_stereo` / :attr:`~CrowsRow.sent_antistereo`, and nothing downstream is
   allowed to read the raw columns for direction. Getting this backwards would flip the sign of
   every CrowS effect while producing entirely plausible output -- precisely the failure
   ``Response.logprob_antistereo``'s field description warns about.

**The pin is resolved, never authored.** ``scripts/fetch_crows_pairs.py --write`` obtained the
revision and the sha256 below from upstream; ``--verify`` re-checks them, and ``--fetch-pinned``
materialises them on a fresh machine. A 404 or a digest mismatch stops rather than substituting.
"""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, NamedTuple, TypedDict

from margins.benchmarks.config import BenchmarkConfig
from margins.schemas import register_payload_type

__all__ = [
    "CROWS_PAIRS_RELEASE",
    "DEFAULT_CROWS_PATH",
    "STEREO_DIRECTIONS",
    "CrowsPairsPayload",
    "CrowsPairsRelease",
    "CrowsRow",
    "crows_pairs_path",
    "read_rows",
]

#: The two legal values of the upstream ``stereo_antistereo`` column. An unrecognised value raises:
#: it decides which sentence is the stereotyping one, so guessing would flip a sign silently.
StereoDirection = Literal["stereo", "antistereo"]
STEREO_DIRECTIONS: Final[tuple[str, ...]] = ("stereo", "antistereo")

#: The upstream columns this loader reads. Checked on load, so an upstream schema change is a
#: failure rather than a `KeyError` three functions later.
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "sent_more",
    "sent_less",
    "stereo_antistereo",
    "bias_type",
)


@dataclass(frozen=True, slots=True)
class CrowsPairsRelease:
    """A pinned CrowS-Pairs release. Mirrors ``HolisticBiasRelease``.

    Duplicated in ``configs/benchmark/crows_pairs_v1.yaml`` on purpose: the YAML is what gets
    checksummed into the run manifest, this constant is what the fetch script uses, and
    ``tests/test_crows_loader.py`` asserts the two agree so neither can drift alone.
    """

    source: str
    repo: str
    repo_path: str
    revision: str
    version: str
    license: str
    sha256: str
    rows: int

    @property
    def raw_url(self) -> str:
        """Revision-pinned raw URL. Pinned to the commit, never to a branch."""
        return f"https://raw.githubusercontent.com/{self.repo}/{self.revision}/{self.repo_path}"

    @property
    def blob_url(self) -> str:
        """Human-browsable URL, for the provenance table."""
        return f"https://github.com/{self.repo}/blob/{self.revision}/{self.repo_path}"


#: RESOLVED 20 Sep 2026 by `python scripts/fetch_crows_pairs.py --write`, never hand-authored.
#:
#: The LICENCE was READ off upstream rather than resolved by the script, because the repository
#: ships no LICENSE file and the GitHub API therefore reports none. Its README at this very commit
#: states: "CrowS-Pairs is licensed under a Creative Commons Attribution-ShareAlike 4.0
#: International License." That sentence is the authority, and it is recorded here so a reproducer
#: can check the same file at the same revision rather than take our word for it.
CROWS_PAIRS_RELEASE: Final[CrowsPairsRelease] = CrowsPairsRelease(
    source="CrowS-Pairs (nyu-mll/crows-pairs)",
    repo="nyu-mll/crows-pairs",
    repo_path="data/crows_pairs_anonymized.csv",
    revision="8aaac11c485473159ec9328a65253a5be9a479dc",
    version="v1.0",
    license="CC-BY-SA-4.0",
    sha256="dfb36986ce0502abbaf7055b9176da3d08d48e07df1251991b5dfbcbceab9d0c",
    rows=1508,
)

DEFAULT_CROWS_PATH: Final[Path] = (
    Path("data") / "benchmarks" / "crows_pairs" / "crows_pairs_anonymized.csv"
)


def crows_pairs_path(data_dir: Path | None = None) -> Path:
    """Where the pinned CSV lives. ``data/`` is gitignored and regenerable."""
    if data_dir is None:
        return Path(__file__).resolve().parents[3] / DEFAULT_CROWS_PATH
    return data_dir / DEFAULT_CROWS_PATH.relative_to("data")


class CrowsPairsPayload(TypedDict):
    """Payload of a CrowS-Pairs :class:`~margins.schemas.Item` (``benchmark="crows_pairs"``).

    Registered in ``schemas.PAYLOAD_TYPES``: the shared record stays source-agnostic, but mypy is
    not blind on the one field carrying the real data (invariant 2).

    Both the raw upstream columns and the **derived** direction are carried. The raw ones preserve
    the audit trail back to a row of the pinned file; the derived ones are what every downstream
    stage reads, so the ``stereo_antistereo`` resolution happens exactly once, here.
    """

    row_index: int
    sent_more: str
    sent_less: str
    stereo_antistereo: str
    bias_type: str
    #: Derived. The MORE-stereotyping member, whichever upstream column it came from.
    sent_stereo: str
    #: Derived. The LESS-stereotyping member.
    sent_antistereo: str
    #: The lattice axis this row's category maps to, or None when out-of-lattice.
    mapped_axis: str | None
    #: True when the category has no lattice axis. Such items are kept, marked, and never celled.
    out_of_lattice: bool
    #: Every ``(axis, value)`` the pair attests, as ordered pairs. The cell is derived from THIS.
    attested: tuple[tuple[str, str], ...]


#: The loader boundary where ``Item.payload`` stops being ``Any`` (invariant 2).
register_payload_type("crows_pairs", CrowsPairsPayload)


class CrowsRow(NamedTuple):
    """One raw row of the pinned CSV, with the stereo direction resolved."""

    row_index: int
    sent_more: str
    sent_less: str
    stereo_antistereo: str
    bias_type: str

    @property
    def sent_stereo(self) -> str:
        """The MORE-stereotyping sentence.

        On an ``antistereo`` row upstream's ``sent_more`` is the *anti*-stereotypical member, so
        the columns swap. This property is the only place that swap is expressed.
        """
        return self.sent_less if self.stereo_antistereo == "antistereo" else self.sent_more

    @property
    def sent_antistereo(self) -> str:
        """The LESS-stereotyping sentence. The complement of :attr:`sent_stereo`."""
        return self.sent_more if self.stereo_antistereo == "antistereo" else self.sent_less


def read_rows(
    path: Path | None = None,
    *,
    expected_sha256: str | None = None,
    expected_rows: int | None = None,
    verify_checksum: bool = True,
) -> tuple[CrowsRow, ...]:
    """Read the pinned CSV, verifying the bytes before parsing them.

    Verification is on by default and refusing is the point: a file that is not the pinned one
    makes every provenance claim built on it wrong, and the failure would otherwise surface as a
    slightly different row count nobody looks at.
    """
    target = crows_pairs_path() if path is None else path
    if not target.is_file():
        raise FileNotFoundError(
            f"CrowS-Pairs source file not found at {target}. `data/` is gitignored and "
            "regenerable: run `python scripts/fetch_crows_pairs.py --fetch-pinned`."
        )

    payload = target.read_bytes()
    if verify_checksum:
        pinned = expected_sha256 or CROWS_PAIRS_RELEASE.sha256
        digest = hashlib.sha256(payload).hexdigest()
        if digest != pinned:
            raise ValueError(
                f"{target} has sha256 {digest}, but the pin records {pinned}. This is not the "
                "pinned file; every provenance claim built on it would be wrong. Re-fetch with "
                "`python scripts/fetch_crows_pairs.py --fetch-pinned`, and do not proceed until a "
                "human has resolved the difference."
            )

    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or ())]
    if missing:
        raise ValueError(
            f"{target} is missing required column(s) {missing}; found "
            f"{list(reader.fieldnames or ())}. Upstream changed its schema, which is a human "
            "decision, not something this loader may adapt to."
        )

    rows: list[CrowsRow] = []
    for index, record in enumerate(reader):
        direction = (record["stereo_antistereo"] or "").strip()
        if direction not in STEREO_DIRECTIONS:
            raise ValueError(
                f"row {index} of {target} has stereo_antistereo={direction!r}, which is not one of "
                f"{list(STEREO_DIRECTIONS)}. That column decides WHICH sentence is the "
                "stereotyping one, so an unrecognised value would flip a sign silently."
            )
        rows.append(
            CrowsRow(
                row_index=index,
                sent_more=(record["sent_more"] or "").strip(),
                sent_less=(record["sent_less"] or "").strip(),
                stereo_antistereo=direction,
                bias_type=(record["bias_type"] or "").strip(),
            )
        )

    expected = expected_rows if expected_rows is not None else CROWS_PAIRS_RELEASE.rows
    if len(rows) != expected:
        raise ValueError(
            f"{target} holds {len(rows)} rows, but the pin records {expected}. Gate P2 requires "
            "the row count to match the published dataset; a mismatch means the file is not the "
            "one that was pinned, whatever its name says."
        )
    return tuple(rows)


def source_rows(config: BenchmarkConfig, *, data_dir: Path | None = None) -> tuple[CrowsRow, ...]:
    """Read the rows this config pins, checking its digest and its row count."""
    return read_rows(
        crows_pairs_path(data_dir),
        expected_sha256=config.source.sha256,
        expected_rows=config.expected_rows,
    )
