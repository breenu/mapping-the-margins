"""The extractive benchmark's ingestion rules, as validated configuration (P2).

The sibling of :mod:`margins.lattice.config` and :mod:`margins.scoring.config`, and it exists for
the same reason they do: the rules that decide **which source items become auditable records** are
exactly the rules that are easiest to violate in prose and impossible to notice downstream. A
loader that quietly dropped half its rows, or quietly guessed which identity a sentence was about,
would produce a perfectly clean pipeline and a fabricated result.

Three rules are therefore enforced here rather than documented:

* **Every upstream attribute category is declared.** A ``bias_type`` absent from :attr:`attributes`
  makes the run **fail**, rather than being silently skipped. Upstream gaining a tenth category is
  a thing a human must look at, not a thing a filter should swallow.
* **A category maps to a lattice axis or to nothing, and "nothing" is a recorded outcome.** Items
  on an unmapped category (religion, nationality, sexual orientation, physical appearance) are
  **kept and marked out-of-lattice**, never forced into a cell and never dropped. The lattice has
  no axis for them; inventing one would be authoring a descriptor taxonomy, which
  ``configs/lattice/`` spends a pinned file and a ``--verify`` mode avoiding.
* **Nothing here maps a surface term to a lattice value.** There is no alias table, on purpose:
  ``poor -> working-class`` and ``Mexican -> Hispanic`` are construct claims *we* would be making,
  and the whole point of resolving values from HolisticBias at a pinned commit is that we do not
  make them. Attestation is literal surface matching (:mod:`margins.benchmarks.attest`), and an
  item whose descriptor is not a lattice value simply attests no cell on that axis.

**The pin is resolved, never authored.** :attr:`BenchmarkSource` records the upstream commit and
the sha256 of the bytes that were actually fetched; ``scripts/fetch_crows_pairs.py`` is what put
them there and what checks them. Same discipline as the HolisticBias descriptor file and the
scoring lexicons.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from margins.schemas import Provenance

__all__ = [
    "BENCHMARK_INPUT_KEY",
    "BENCHMARK_SOURCE_PREFIX",
    "DEFAULT_BENCHMARK",
    "AttributeMap",
    "BenchmarkConfig",
    "BenchmarkSource",
    "ScreeningSettings",
    "benchmark_config_from_mapping",
    "default_benchmark_config_dir",
    "load_benchmark_config",
]

#: The extractive source this project's confirmatory runs ingest.
DEFAULT_BENCHMARK: Final[str] = "crows_pairs_v1"

#: Manifest ``inputs`` keys, alongside :data:`~margins.config.LATTICE_INPUT_KEY`. Constants so the
#: manifest vocabulary is greppable and a test can assert the key rather than a spelling of it.
BENCHMARK_INPUT_KEY: Final[str] = "benchmark_config"
BENCHMARK_SOURCE_PREFIX: Final[str] = "benchmark_source:"

_SHA_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")


def file_sha256(path: Path) -> str:
    """sha256 of a file's raw bytes, for the run manifest's ``input_checksums``.

    Raw bytes, not normalized text: ``.gitattributes`` pins ``eol=lf`` for files in the repo, and
    the fetched source file is written already normalized by ``scripts/fetch_crows_pairs.py``. Both
    ends of the digest therefore agree across the two locked environments.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BenchmarkSource(BaseModel):
    """Where the ingested dataset came from, at which revision, under which licence.

    Mirrors :class:`margins.lattice.config.DescriptorSource`. The duplication between this block
    and the release constant in :mod:`margins.benchmarks.crows_pairs` is deliberate: the YAML is
    what gets checksummed into the run manifest, the constant is what the fetch script uses, and a
    test asserts the two agree so neither can drift alone.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = Field(min_length=1, description="Human-readable source identifier.")
    url: str | None = Field(default=None, description="Revision-pinned, human-browsable URL.")
    path: str | None = Field(default=None, description="Path within the upstream repository.")
    revision: str | None = Field(default=None, description="Upstream commit sha, never a branch.")
    version: str | None = Field(default=None, description="Upstream release label, if any.")
    license: str = Field(
        min_length=1,
        description="Required and never defaulted: an unrecorded licence blocks the release.",
    )
    sha256: str = Field(min_length=64, max_length=64, description="Digest of the fetched bytes.")
    retrieved_at: datetime

    @field_validator("revision")
    @classmethod
    def _a_pin_is_a_commit(cls, value: str | None) -> str | None:
        """Reject a branch name. ``master`` moves, and a source that moved is a different source."""
        if value is not None and not _SHA_RE.match(value):
            raise ValueError(
                f"revision {value!r} is not a 40-character commit sha. A branch name is not a pin: "
                "re-resolve it with `python scripts/fetch_crows_pairs.py --write`."
            )
        return value

    @field_validator("sha256")
    @classmethod
    def _sha256_is_a_digest(cls, value: str) -> str:
        if not _SHA256_RE.match(value):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        return value

    def to_provenance(self) -> Provenance:
        """The :class:`~margins.schemas.Provenance` every ingested ``Item`` carries.

        ``url`` and ``path`` are dropped deliberately: they are retrieval instructions, and putting
        them on a shared record would violate invariant 2.
        """
        return Provenance(
            source=self.source,
            revision=self.revision,
            version=self.version,
            license=self.license,
            sha256=self.sha256,
            retrieved_at=self.retrieved_at,
        )


class AttributeMap(BaseModel):
    """One upstream attribute category, and the lattice axis it corresponds to -- or none.

    ``axis: null`` is a **declared outcome**, not a gap. It says: this category is real, we ingest
    it, and the lattice has no axis for it, so its items are marked out-of-lattice and reported
    rather than celled. An undeclared category, by contrast, makes the load fail.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    category: str = Field(min_length=1, description="The upstream category label, verbatim.")
    axis: str | None = Field(
        default=None,
        description="The lattice axis this category corresponds to, or null for out-of-lattice.",
    )
    note: str | None = Field(default=None, description="Why, where the mapping deserves a reason.")

    @model_validator(mode="after")
    def _an_unmapped_category_says_why(self) -> AttributeMap:
        """An out-of-lattice category must state its reason.

        Otherwise ``axis: null`` is indistinguishable from someone forgetting to fill it in, which
        is exactly the failure this record exists to make impossible.
        """
        if self.axis is None and not (self.note and self.note.strip()):
            raise ValueError(
                f"category {self.category!r} maps to no lattice axis but records no note. An "
                "out-of-lattice category is a DECLARED outcome and must say why it is one; "
                "without a note it is indistinguishable from an unfinished mapping."
            )
        return self


class ScreeningSettings(BaseModel):
    """The construct-validity screen's thresholds (PROJECT_CONTEXT.md section 9).

    Every threshold is here rather than in code so the screen is inspectable and its effect on the
    exclusion counts is a config diff. What is NOT configurable: the exclusion *reasons* and the
    order they are applied in. Both are part of what "screening is deterministic" means, and a
    config that could reorder them could change which reason a given item is excluded under.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_differing_spans: int = Field(
        default=1,
        ge=1,
        description=(
            "A minimal pair differs in ONE contiguous region. More than one differing span means "
            "the two sentences differ in more than the intended attribute, so the log-probability "
            "difference is not attributable to it."
        ),
    )
    max_differing_tokens: int = Field(
        default=4,
        ge=1,
        description=(
            "Upper bound on the size of that one span, on either side. A swap of a long phrase for "
            "a short one changes sequence length and token frequency, which the likelihood "
            "difference cannot distinguish from a preference about identity."
        ),
    )
    require_contrast_on_mapped_axis: bool = Field(
        default=True,
        description=(
            "The differing span must itself contain a lattice value on the category's mapped axis. "
            "Without this a pair could be screened in on an attestation found elsewhere in the "
            "sentence, and the contrast would not be about the cell the item is assigned to."
        ),
    )
    agreement_tokens: tuple[str, ...] = Field(
        default=(),
        description=(
            "Closed-class tokens whose form is FORCED by the attribute being swapped. A differing "
            "region made only of these is obligatory grammatical concord -- swapping `woman` for "
            "`man` forces `she` to become `he` -- not an independent second contrast, so it does "
            "not count toward `max_differing_spans`. Listed in the config rather than in code "
            "because it is a short list WE author (a fact about English, not a claim about "
            "identity) and it must therefore be visible and criticisable. Empty by default: the "
            "allowance is opt-in, and the screening report states how many rows it admitted."
        ),
    )

    @field_validator("agreement_tokens")
    @classmethod
    def _agreement_tokens_are_lowercase(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Matching happens on lower-cased tokens, so an upper-case entry would never fire."""
        offenders = sorted({t for t in value if t != t.lower() or not t.strip()})
        if offenders:
            raise ValueError(
                f"agreement_tokens must be non-empty and lower-case; got {offenders}. Tokens are "
                "compared against lower-cased text, so an upper-case entry would silently never "
                "match."
            )
        return value

    @property
    def agreement(self) -> frozenset[str]:
        return frozenset(self.agreement_tokens)


class BenchmarkConfig(BaseModel):
    """One extractive source's ingestion contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(min_length=1, description="Must equal the config filename.")
    benchmark_version: str = Field(min_length=1)
    description: str = Field(default="")

    benchmark: str = Field(
        min_length=1, description="The source label stamped on every Item.benchmark."
    )
    task_type: Literal["pair_pll"] = Field(
        default="pair_pll",
        description=(
            "Typed this narrowly on purpose, exactly as PromptSetConfig types its own. This config "
            "governs a sentence-pair likelihood source; a coref or qa source has a different "
            "screen and a different payload, and should fail to LOAD here rather than fail review."
        ),
    )

    source: BenchmarkSource
    expected_rows: int = Field(
        ge=1,
        description=(
            "Row count of the pinned file, RESOLVED by the fetch script at pin time. Redundant "
            "with the sha256 and deliberately so: a digest failure says 'these are not the bytes', "
            "a row-count failure says 'and here is how they differ', which is what gate P2 asks "
            "to be reported."
        ),
    )

    attributes: tuple[AttributeMap, ...] = Field(
        min_length=1, description="Every upstream category, mapped to an axis or declared unmapped."
    )
    screening: ScreeningSettings = Field(default_factory=ScreeningSettings)

    @model_validator(mode="after")
    def _categories_are_unique(self) -> BenchmarkConfig:
        seen = [entry.category for entry in self.attributes]
        duplicates = sorted({name for name in seen if seen.count(name) > 1})
        if duplicates:
            raise ValueError(f"attribute categories must be unique, got duplicates: {duplicates}")
        return self

    @property
    def axis_for(self) -> dict[str, str | None]:
        """``category -> axis or None``. A category absent from this map is an error, not a skip."""
        return {entry.category: entry.axis for entry in self.attributes}

    @property
    def mapped_axes(self) -> tuple[str, ...]:
        return tuple(sorted({e.axis for e in self.attributes if e.axis is not None}))

    @property
    def out_of_lattice_categories(self) -> tuple[str, ...]:
        return tuple(sorted(e.category for e in self.attributes if e.axis is None))

    def provenance(self) -> Provenance:
        return self.source.to_provenance()

    def validate_against_lattice(self, axes: Sequence[str]) -> None:
        """Raise unless every mapped axis is a declared lattice axis.

        Kept out of the model validator because it needs the lattice, which this module
        deliberately does not import: the config must be loadable and inspectable on its own.
        """
        declared = set(axes)
        unknown = sorted({a for a in self.mapped_axes if a not in declared})
        if unknown:
            raise ValueError(
                f"benchmark config {self.key!r} maps categories onto axes the lattice does not "
                f"declare: {unknown}. Declared axes are {sorted(declared)}. An axis that is not in "
                "the lattice cannot carry a cell, so this mapping could never attest anything."
            )


def benchmark_config_from_mapping(data: Mapping[str, Any]) -> BenchmarkConfig:
    """Where Hydra stops and Pydantic starts. No Hydra structured configs, no hydra-zen."""
    return BenchmarkConfig.model_validate(dict(data))


def default_benchmark_config_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "benchmark"


def load_benchmark_config(
    name: str = DEFAULT_BENCHMARK, *, config_dir: Path | None = None
) -> BenchmarkConfig:
    """Compose a benchmark config group entry and validate it."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    directory = (config_dir or default_benchmark_config_dir()).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"benchmark config dir does not exist: {directory}")

    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        composed = compose(config_name=name)
    data = OmegaConf.to_container(composed, resolve=True)
    if not isinstance(data, dict):
        raise TypeError(f"benchmark config {name!r} did not compose to a mapping")
    payload = cast(dict[str, Any], data)
    payload.setdefault("key", name)

    config = benchmark_config_from_mapping(payload)
    if config.key != name:
        raise ValueError(
            f"benchmark config {name!r} declares key={config.key!r}. The filename and the key must "
            "agree, or the manifest records a source name that resolves to a different file."
        )
    return config
