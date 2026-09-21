"""Declarative configuration for the templated ``open_gen`` prompt set (P4a).

Why a declarative config rather than literals in code: gate P4a requires **byte-identical prompt
regeneration**, and the prompt set is the input to the primary estimand. Everything that decides
what a prompt says — the template frames, the carrier noun, the article rule, the version stamped
on every :class:`~margins.schemas.Prompt` — therefore has to live in a file a human writes,
reviews, and git-tags, exactly as the lattice does.

**The carrier noun is a CONSTRUCT CHOICE, and it is the choice this module exists to make
explicit.** P3's ``compose_descriptor`` deliberately stops at a modifier phrase — ``"older
blind"``, ``"working-class Black"`` — with no carrier noun, leaving the noun to P4a where the
audit trail for it belongs. The shipped confirmatory config attaches exactly one carrier,
``person``, because:

* Every alternative in the obvious set **is itself an identity axis**. ``man``/``woman`` carry
  gender, which would confound the ``gender_and_sex`` axis and silently gender every *non*-gender
  cell; ``kid``/``teen``/``elder`` carry age; ``patient``/``client`` carry a role relation.
* An **occupational** carrier (``nurse``, ``engineer``, ``cleaner``) carries dense class, gender
  and race stereotype content of its own. Using one would add an **undeclared second axis of
  variation** to every cell and make the interaction estimate uninterpretable — it would no longer
  be "harm varies non-additively across identity cells" but "harm varies across identity cells
  crossed with an occupation we never pre-registered". :class:`CarrierSpec` therefore refuses a
  non-neutral carrier that does not name the axis of variation it introduces, and
  :data:`MAX_CARRIERS` keeps the confirmatory set small: carriers multiply the prompt set and every
  extra one spends power on breadth.

``person`` is not neutral in an absolute sense — nothing is — and the register it creates with
some descriptors is a live review item (see ``data/prompts/phrase_audit.md``). It is the least
axis-loaded option available, which is a different and weaker claim, and that is the claim made.

**The article rule is the second construct choice, and it belongs here, not in the lattice.**
``"a older blind person"`` is broken English and a language model reacts to broken English. The
fix is a template-frame concern; the descriptors are pinned P3 provenance and are never edited to
repair phrasing. :meth:`PromptSetConfig.article_for` implements the plain vowel-letter rule and
**refuses** to guess on the words where that rule is known to fail (``eu-``, ``u-``, ``one-``,
``hon-``), forcing an explicit override instead of a silent error.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from margins.schemas import Provenance

__all__ = [
    "ARTICLE_RISKY_PREFIXES",
    "DEFAULT_EXTRACTIVE_SET",
    "DEFAULT_PROMPT_SET",
    "EXTRACTIVE_SET_INPUT_KEY",
    "LEGAL_ARTICLES",
    "MAX_CARRIERS",
    "PROMPT_SET_INPUT_KEY",
    "TEMPLATE_DIRNAME",
    "TEMPLATE_INPUT_PREFIX",
    "VOWEL_LETTERS",
    "CarrierSpec",
    "ExtractivePromptSetConfig",
    "PromptSetConfig",
    "TemplateFile",
    "TemplateSpec",
    "default_prompt_config_dir",
    "extractive_config_from_mapping",
    "file_sha256",
    "indefinite_article",
    "load_extractive_config",
    "load_prompt_config",
    "load_template_files",
    "normalize_template_text",
    "prompt_config_from_mapping",
]

#: The prompt set shipped with the repo.
DEFAULT_PROMPT_SET: Final[str] = "open_gen_v1"

#: Template files live in a subdirectory of the prompt config dir (PROJECT_CONTEXT.md section 4).
TEMPLATE_DIRNAME: Final[str] = "templates"

#: Carriers multiply the prompt set: |templates| x |cells| x |carriers|. The confirmatory set is
#: deliberately tiny -- the shipped config declares one. The cap is not a style preference; extra
#: carriers spend rendering budget that the power bookkeeping already allocated per cell.
MAX_CARRIERS: Final[int] = 3

LEGAL_ARTICLES: Final[frozenset[str]] = frozenset({"a", "an"})

VOWEL_LETTERS: Final[frozenset[str]] = frozenset("aeiou")

#: Word beginnings where the vowel-*letter* rule disagrees with the vowel-*sound* rule that
#: English articles actually follow ("a European", "a university", "a one-time thing", "an hour").
#: A descriptor starting with one of these must carry an explicit override: guessing here produces
#: broken English in a prompt, and a model's reaction to broken English is not the estimand.
ARTICLE_RISKY_PREFIXES: Final[tuple[str, ...]] = ("eu", "u", "one", "hon", "hour")

#: Manifest ``inputs`` keys. Constants rather than string literals so the manifest vocabulary is
#: greppable and a test can assert the key rather than a spelling of it.
PROMPT_SET_INPUT_KEY: Final[str] = "prompt_set_config"
TEMPLATE_INPUT_PREFIX: Final[str] = "template:"


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's raw bytes.

    Raw bytes, not normalized text, so the digest matches what ``scripts/build_lattice.py``
    already prints for the lattice config. This is only safe because ``.gitattributes`` pins
    ``eol=lf`` for every text file in the repository -- without that, a checksum would depend on
    which of the two locked environments checked the file out.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def indefinite_article(phrase: str, overrides: Mapping[str, str]) -> str:
    """The indefinite article for ``phrase``: the vowel-*letter* rule plus an override table.

    Factored out of :meth:`PromptSetConfig.article_for` at P8 so the red-team prompt set applies
    the **same** rule rather than a second copy of it. Two copies of an article rule is two places
    for "a older blind person" to appear, and a model's reaction to broken English is not the
    estimand.

    English actually follows the vowel-*sound*, so the letter rule is wrong for a small known set
    of word beginnings (:data:`ARTICLE_RISKY_PREFIXES`). Those **raise** rather than guess.

    Raises:
        ValueError: If ``phrase`` is empty, or begins with a word where the letter rule is known to
            fail and no override is declared.
    """
    words = phrase.split()
    if not words:
        raise ValueError("cannot choose an article for an empty descriptor phrase")
    first = words[0].lower()

    override = overrides.get(first)
    if override is not None:
        return override

    if first.startswith(ARTICLE_RISKY_PREFIXES):
        raise ValueError(
            f"descriptor phrase {phrase!r} starts with {first!r}, where the vowel-letter rule "
            "disagrees with English pronunciation ('a European', 'a university', 'an hour'). "
            "Declare an explicit entry in `article_overrides` -- do not edit the descriptor, "
            "which is pinned P3 provenance."
        )
    return "an" if first[0] in VOWEL_LETTERS else "a"


def normalize_template_text(text: str) -> str:
    """Collapse a template file into a single whitespace-normalized line.

    A template file ends with a newline and may be wrapped for readability; neither should reach
    the model, and neither should be able to change the rendered bytes. Collapsing here makes the
    byte-identity guarantee independent of how the file happens to be formatted or checked out.
    """
    return " ".join(text.split())


class TemplateSpec(BaseModel):
    """One declared template frame."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, description="Stable key; appears in item ids and prompt ids.")
    file: str = Field(min_length=1, description="Filename inside the template directory.")
    note: str = Field(
        min_length=1,
        description=(
            "What this frame elicits and why it is neutral. Required: an unexplained frame is an "
            "unexamined construct choice, and the frame is half of what a prompt measures."
        ),
    )


class CarrierSpec(BaseModel):
    """The noun a composed descriptor attaches to. **A construct choice, never a default.**

    See the module docstring. ``axis_neutral=False`` is permitted but must name the axis of
    variation it introduces, so an occupational or role carrier can only enter the design as a
    *declared* axis and never by accident.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    noun: str = Field(min_length=1)
    axis_neutral: bool = Field(
        description=(
            "True only for a carrier that carries no identity signal of its own. "
            "'man'/'woman' (gender), 'kid'/'elder' (age) and every occupational noun are NOT "
            "axis-neutral."
        )
    )
    declared_axis: str | None = Field(
        default=None,
        description="Required for a non-neutral carrier: the axis of variation it introduces.",
    )
    note: str = Field(min_length=1, description="The construct rationale. Always required.")

    @model_validator(mode="after")
    def _non_neutral_carrier_declares_its_axis(self) -> CarrierSpec:
        if not self.axis_neutral and not (self.declared_axis or "").strip():
            raise ValueError(
                f"carrier {self.noun!r} is not axis-neutral and must set `declared_axis`. A "
                "carrier that carries identity signal is a second axis of variation; it may enter "
                "the design declared and documented, never silently."
            )
        if self.axis_neutral and self.declared_axis:
            raise ValueError(
                f"carrier {self.noun!r} is marked axis_neutral but declares axis "
                f"{self.declared_axis!r}; it cannot be both."
            )
        return self


class PromptSetConfig(BaseModel):
    """The validated ``open_gen`` prompt-set declaration.

    Frozen, like every config here: the prompt set a run audited is what its manifest and its
    ``template_version`` claim it was.

    ``task_type`` is a ``Literal["open_gen"]`` rather than the full :class:`~margins.schemas.
    TaskType` enum **on purpose**. P4a builds the generation path only; the extractive
    ``ExtractiveRenderer`` (P4b) derives a cell from a source item and cannot share this config.
    Typing the field this narrowly means a paired-task prompt set does not merely fail a review,
    it fails to load.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    template_version: str = Field(
        min_length=1,
        description=(
            "Stamped on every Prompt. Bump on ANY change to a template file, the carrier set, or "
            "the article rule -- the prompt id embeds a digest of the rendered text, so an "
            "unbumped change shows up as changed ids in a diff rather than as silent drift."
        ),
    )
    description: str = Field(min_length=1, description="Frame-selection rule, in prose.")
    task_type: Literal["open_gen"] = "open_gen"
    license: str = Field(min_length=1, description="License of the template files themselves.")
    authored_at: datetime = Field(
        description="Recorded, never clock-read: template provenance must be deterministic."
    )
    decoding_key: str = Field(
        min_length=1,
        description=(
            "Named decoding profile stamped on every Prompt. P4a only NAMES it; the resolved "
            "config and its `decoding_hash` landed at P5 in `configs/decoding/<key>.yaml` and "
            "`margins.inference.decoding`. The backend refuses a prompt whose key does not match "
            "the profile it holds, so the two cannot drift apart."
        ),
    )
    templates: tuple[TemplateSpec, ...]
    carriers: tuple[CarrierSpec, ...]
    article_overrides: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Lowercased FIRST WORD of a composed descriptor -> 'a' or 'an'. The lever for the "
            "words where the vowel-letter rule is wrong. Descriptors are pinned P3 provenance and "
            "are never edited to fix an article."
        ),
    )

    # ---------------------------------------------------------------- validation

    @model_validator(mode="after")
    def _check_templates(self) -> PromptSetConfig:
        if not self.templates:
            raise ValueError("a prompt set must declare at least one template")
        ids = [spec.id for spec in self.templates]
        if len(set(ids)) != len(ids):
            raise ValueError(f"template ids must be unique, got {ids}")
        files = [spec.file for spec in self.templates]
        if len(set(files)) != len(files):
            raise ValueError(f"template files must be unique, got {files}")
        return self

    @model_validator(mode="after")
    def _check_carriers(self) -> PromptSetConfig:
        if not self.carriers:
            raise ValueError("a prompt set must declare at least one carrier noun")
        nouns = [carrier.noun for carrier in self.carriers]
        if len(set(nouns)) != len(nouns):
            raise ValueError(f"carrier nouns must be unique, got {nouns}")
        if len(self.carriers) > MAX_CARRIERS:
            raise ValueError(
                f"{len(self.carriers)} carriers exceeds the cap of {MAX_CARRIERS}. Carriers "
                "multiply the prompt set (|templates| x |cells| x |carriers|) and spend the "
                "per-cell rendering budget the lattice's power bookkeeping already allocated."
            )
        if not any(carrier.axis_neutral for carrier in self.carriers):
            raise ValueError(
                "at least one carrier must be axis-neutral; a prompt set built entirely from "
                "identity-bearing carriers has an undeclared second axis in every cell"
            )
        return self

    @model_validator(mode="after")
    def _check_article_overrides(self) -> PromptSetConfig:
        for word, article in self.article_overrides.items():
            if not word or word != word.lower():
                raise ValueError(f"article override key {word!r} must be a lowercased word")
            if article not in LEGAL_ARTICLES:
                raise ValueError(
                    f"article override for {word!r} is {article!r}; must be one of "
                    f"{sorted(LEGAL_ARTICLES)}"
                )
        return self

    # ---------------------------------------------------------------- accessors

    @property
    def item_benchmark(self) -> str:
        """The source label of the items this prompt set renders (the ``RenderRules`` Protocol).

        A constant for the static set, and deliberately **not** a config field: a prompt set whose
        source label could be edited in YAML could relabel the static paradigm as the red-team one,
        which is precisely the distinction the E3 comparison rests on. P8's red-team prompt set is a
        separate class returning ``"redteam"``.
        """
        from margins.prompts.render import TEMPLATE_ITEM_BENCHMARK

        return TEMPLATE_ITEM_BENCHMARK

    @property
    def axis_neutral_carriers(self) -> tuple[CarrierSpec, ...]:
        return tuple(carrier for carrier in self.carriers if carrier.axis_neutral)

    @property
    def declared_carrier_axes(self) -> tuple[str, ...]:
        """Axes of variation introduced by non-neutral carriers. Empty in the confirmatory set."""
        return tuple(
            carrier.declared_axis for carrier in self.carriers if carrier.declared_axis is not None
        )

    def article_for(self, phrase: str) -> str:
        """The indefinite article for ``phrase``.

        Plain vowel-*letter* rule plus an explicit override table. English actually follows the
        vowel-*sound*, so this rule is wrong for a small, known set of word beginnings
        (:data:`ARTICLE_RISKY_PREFIXES`); those raise rather than guess. A broken article is
        broken English inside an audit prompt, and a model's reaction to broken English is not
        what this instrument is measuring.

        Raises:
            ValueError: If ``phrase`` is empty, or begins with a word where the letter rule is
                known to fail and no override is declared.
        """
        return indefinite_article(phrase, self.article_overrides)

    def provenance_for(self, sha256: str) -> Provenance:
        """Provenance for a template file with content digest ``sha256``.

        Templates are first-party source material, so their provenance names this repository and
        the prompt-set version. It is a real :class:`~margins.schemas.Provenance` and not a
        formality: it is what keeps ``Prompt.item_id`` pointing at something version-anchored,
        which is the whole reason a template is an :class:`~margins.schemas.Item`.
        """
        return Provenance(
            source=f"mapping-the-margins open_gen templates ({self.template_version})",
            revision=None,
            version=self.template_version,
            license=self.license,
            sha256=sha256,
            retrieved_at=self.authored_at,
        )


@dataclass(frozen=True, slots=True)
class TemplateFile:
    """A declared template plus the file content and digest behind it."""

    spec: TemplateSpec
    text: str
    sha256: str
    path: Path

    @property
    def id(self) -> str:
        return self.spec.id


def load_template_files(
    config: PromptSetConfig, template_dir: Path | None = None
) -> tuple[TemplateFile, ...]:
    """Read every declared template file, in declaration order.

    Args:
        config: The validated prompt-set config.
        template_dir: Directory holding the ``.jinja`` files. Defaults to
            ``<prompt config dir>/templates``.

    Raises:
        FileNotFoundError: If a declared template file is missing. A prompt set that silently
            skips a missing frame would change the per-cell rendering count without saying so.
    """
    directory = template_dir if template_dir is not None else default_template_dir()
    files: list[TemplateFile] = []
    for spec in config.templates:
        path = directory / spec.file
        if not path.is_file():
            raise FileNotFoundError(f"template {spec.id!r} declares missing file: {path}")
        text = normalize_template_text(path.read_text(encoding="utf-8"))
        if not text:
            raise ValueError(f"template {spec.id!r} is empty: {path}")
        files.append(TemplateFile(spec=spec, text=text, sha256=file_sha256(path), path=path))
    return tuple(files)


# --------------------------------------------------------------------------------------
# The Hydra seam
# --------------------------------------------------------------------------------------


def prompt_config_from_mapping(data: Mapping[str, Any]) -> PromptSetConfig:
    """Validate a plain mapping into a :class:`PromptSetConfig`."""
    return PromptSetConfig.model_validate(dict(data))


def default_prompt_config_dir() -> Path:
    """``<repo>/configs/prompts``. Override it for tests or an installed package."""
    return Path(__file__).resolve().parents[3] / "configs" / "prompts"


def default_template_dir() -> Path:
    """``<repo>/configs/prompts/templates``."""
    return default_prompt_config_dir() / TEMPLATE_DIRNAME


def load_prompt_config(
    name: str = DEFAULT_PROMPT_SET, *, config_dir: Path | None = None
) -> PromptSetConfig:
    """Compose ``<config_dir>/<name>.yaml`` with Hydra and validate it.

    Mirrors :func:`margins.lattice.config.load_lattice_config`: Hydra composes plain YAML, the
    result becomes a plain dict, and Pydantic validates. No Hydra structured configs.
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    directory = (config_dir or default_prompt_config_dir()).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"prompt config dir does not exist: {directory}")

    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        composed = compose(config_name=name)
    data = OmegaConf.to_container(composed, resolve=True)
    if not isinstance(data, dict):
        raise TypeError(f"composed prompt config {name!r} is not a mapping, got {type(data)!r}")
    return prompt_config_from_mapping(cast(Mapping[str, Any], data))


# --------------------------------------------------------------------------------------
# The extractive prompt set (P4b)
# --------------------------------------------------------------------------------------

#: The extractive prompt set this project's confirmatory runs render.
DEFAULT_EXTRACTIVE_SET: Final[str] = "crows_pll_v1"

#: Manifest ``inputs`` key for the extractive prompt-set config's sha256.
EXTRACTIVE_SET_INPUT_KEY: Final[str] = "extractive_set_config"


class ExtractivePromptSetConfig(BaseModel):
    """The validated ``pair_pll`` prompt-set declaration (P4b).

    A **separate class** from :class:`PromptSetConfig`, not a widening of it. That config types
    ``task_type`` as ``Literal["open_gen"]`` precisely so a paired prompt set fails to *load*
    rather than failing review, and the two have almost nothing in common: this one has no
    templates, no carriers, no article rule and no leakage guard, because an extractive renderer
    composes no text at all -- it copies the source sentences verbatim. Sharing a config class
    would mean every field on it was optional for one of the two renderers, which is how a config
    stops validating anything.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(min_length=1, description="Must equal the config filename.")
    template_version: str = Field(
        min_length=1,
        description=(
            "Stamped on every Prompt, and part of the prompt id's digest preimage. 'Template' is a "
            "slight misnomer here -- there is no template -- but the field is the shared record's, "
            "and it means the same thing: the version of the rendering rules that produced this "
            "prompt. Bump it on any change that alters what gets scored."
        ),
    )
    description: str = Field(min_length=1, description="What this prompt set is, in prose.")
    task_type: Literal["pair_pll"] = "pair_pll"
    benchmark: str = Field(
        min_length=1,
        description=(
            "The source whose items this set renders. The renderer refuses an item from any other "
            "benchmark: two sources sharing a template_version would be indistinguishable "
            "downstream."
        ),
    )
    license: str = Field(
        min_length=1,
        description=(
            "Licence of the RENDERED prompts. For an extractive set the sequences are the upstream "
            "dataset's own sentences, so this is the dataset's licence, not ours."
        ),
    )
    authored_at: datetime = Field(
        description="Recorded, never clock-read: prompt provenance must be deterministic."
    )
    decoding_key: str = Field(
        min_length=1,
        description=(
            "Named decoding profile stamped on every Prompt. For a paired set this is a "
            "log-probability SCORING profile, not a generation one: it records whether the chat "
            "template is applied and what the input bound is, both of which change the measured "
            "sequence log-likelihood. The backend refuses a prompt whose key does not match the "
            "profile it holds."
        ),
    )


def extractive_config_from_mapping(data: Mapping[str, Any]) -> ExtractivePromptSetConfig:
    """Validate a plain mapping into an :class:`ExtractivePromptSetConfig`."""
    return ExtractivePromptSetConfig.model_validate(dict(data))


def load_extractive_config(
    name: str = DEFAULT_EXTRACTIVE_SET, *, config_dir: Path | None = None
) -> ExtractivePromptSetConfig:
    """Compose ``<config_dir>/<name>.yaml`` with Hydra and validate it."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    directory = (config_dir or default_prompt_config_dir()).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"prompt config dir does not exist: {directory}")

    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        composed = compose(config_name=name)
    data = OmegaConf.to_container(composed, resolve=True)
    if not isinstance(data, dict):
        raise TypeError(f"composed prompt config {name!r} is not a mapping, got {type(data)!r}")
    payload = cast(dict[str, Any], data)
    payload.setdefault("key", name)

    config = extractive_config_from_mapping(payload)
    if config.key != name:
        raise ValueError(
            f"extractive prompt set {name!r} declares key={config.key!r}. The filename and the key "
            "must agree, or the manifest records a set name that resolves to a different file."
        )
    return config
