"""Declarative configuration for cell-targeted red-teaming (P8).

Everything that decides *what the red LM is asked*, *what survives the filter* and *how the
survivors are rendered* lives in a file a human writes, reviews and git-tags — the same standard
``margins.prompts.config`` holds the static prompt set to, and for the same reason: the red-team
set is an input to an estimand.

**Two prompt populations, and this module declares both.** They must never be confused:

* **Elicitation** prompts address the *red LM*. They deliberately **name** the target identity —
  that is the whole point of cell-targeting — and they are never scored. Their source label is
  ``redteam_elicitation``.
* **Audit** prompts are what the *target model* sees. They are rendered from generated frames
  through the unchanged :class:`~margins.prompts.render.TemplateRenderer`, so they pass the P4a
  leakage guard, and they are scored. Their source label is ``redteam``.

**The carriers and the article rule are INHERITED from the static prompt set, not re-declared.**
E3 compares red-team against static on matched cells, so a red-team set with its own carrier noun
would make the comparison confound "red-team frame" with "different carrier". The base set's name
is declared here and its sha256 is recorded in the manifest, so the inheritance is checkable rather
than assumed.

**`template_version` is the red-team version, not the base set's.** Red-team prompts must be
distinguishable from static ones in an artifact, and the version is half of what a ``prompt_id``
digests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from margins.prompts.config import (
    CarrierSpec,
    PromptSetConfig,
    file_sha256,
    indefinite_article,
    load_prompt_config,
    normalize_template_text,
)
from margins.schemas import Provenance

__all__ = [
    "DEFAULT_REDTEAM_SET",
    "ELICITATION_INPUT_PREFIX",
    "REDTEAM_BENCHMARK",
    "REDTEAM_ELICITATION_BENCHMARK",
    "REDTEAM_SET_INPUT_KEY",
    "DedupSettings",
    "ElicitationFile",
    "ElicitationSettings",
    "ElicitationTemplateSpec",
    "FilterSettings",
    "GeneratorSpec",
    "OnTargetSettings",
    "RedTeamConfig",
    "RedTeamPromptSetConfig",
    "TargetSettings",
    "default_elicitation_dir",
    "default_redteam_config_dir",
    "load_base_prompt_set",
    "load_elicitation_files",
    "load_redteam_config",
    "redteam_config_from_mapping",
]

#: The red-team set shipped with the repo.
DEFAULT_REDTEAM_SET: Final[str] = "cell_targeted_v1"

#: Source label of the AUDIT prompts — what the target model sees, what gets scored, and the tag
#: the E3 static-vs-red-team comparison reads off every ``ItemScore``.
REDTEAM_BENCHMARK: Final[str] = "redteam"

#: Source label of the ELICITATION prompts — what the red LM is asked. Never scored. A separate
#: label because these deliberately name identity and would fail every guard the audit prompts pass.
REDTEAM_ELICITATION_BENCHMARK: Final[str] = "redteam_elicitation"

#: Manifest ``inputs`` keys.
REDTEAM_SET_INPUT_KEY: Final[str] = "redteam_config"
ELICITATION_INPUT_PREFIX: Final[str] = "elicitation:"

#: Elicitation templates live in a subdirectory of the red-team config dir.
ELICITATION_DIRNAME: Final[str] = "templates"


class ElicitationTemplateSpec(BaseModel):
    """One way of asking the red LM for frames."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, description="Stable key; appears in elicitation prompt ids.")
    file: str = Field(min_length=1, description="Filename inside the elicitation template dir.")
    note: str = Field(
        min_length=1,
        description=(
            "What this elicitation angle asks for and why. Required: an unexplained elicitation "
            "strategy is an unexamined construct choice, and it is half of what the red-team "
            "distribution IS."
        ),
    )


@dataclass(frozen=True, slots=True)
class ElicitationFile:
    """A declared elicitation template plus the file content and digest behind it.

    A sibling of :class:`~margins.prompts.config.TemplateFile` rather than a reuse of it: that
    record's ``spec`` is a static :class:`~margins.prompts.config.TemplateSpec`, and an elicitation
    template is a different thing with a different variable vocabulary. Borrowing the type would
    have been a lie the type checker could not catch.
    """

    spec: ElicitationTemplateSpec
    text: str
    sha256: str
    path: Path

    @property
    def id(self) -> str:
        return self.spec.id


def load_elicitation_files(
    specs: Sequence[ElicitationTemplateSpec], directory: Path
) -> tuple[ElicitationFile, ...]:
    """Read every declared elicitation template, in declaration order.

    Raises:
        FileNotFoundError: If a declared file is missing. A silently skipped elicitation angle
            changes the red-team distribution without saying so, and the distribution is what P8
            contributes.
    """
    files: list[ElicitationFile] = []
    for spec in specs:
        path = directory / spec.file
        if not path.is_file():
            raise FileNotFoundError(
                f"elicitation template {spec.id!r} declares missing file: {path}"
            )
        text = normalize_template_text(path.read_text(encoding="utf-8"))
        if not text:
            raise ValueError(f"elicitation template {spec.id!r} is empty: {path}")
        files.append(ElicitationFile(spec=spec, text=text, sha256=file_sha256(path), path=path))
    return tuple(files)


class GeneratorSpec(BaseModel):
    """The red LM: which pinned checkpoint proposes frames, under which decoding profile.

    **The generator is a recorded constant, never a default.** A re-pin is a different instrument
    producing a different prompt distribution, exactly as a re-pinned scorer is a different
    instrument producing different scores.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    model_name: str = Field(
        min_length=1, description="Name under configs/model/; its pin is resolved from that file."
    )
    decoding_name: str = Field(
        min_length=1,
        description=(
            "Name under configs/decoding/. Must be a SAMPLED profile: distinct candidates come "
            "from distinct seeds, and a greedy profile returns the same frame every time."
        ),
    )
    note: str = Field(
        min_length=1,
        description=(
            "The construct rationale for this choice of red LM, INCLUDING its relationship to the "
            "target models. Required, because a generator sharing a family with a target is a "
            "limitation that must be stated rather than discovered."
        ),
    )


class ElicitationSettings(BaseModel):
    """How the red LM is asked, and how many candidates come back."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    templates: tuple[ElicitationTemplateSpec, ...]
    candidates_per_target: int = Field(
        ge=1,
        le=64,
        description=(
            "Distinct seeds per (elicitation template x target cell). P5's harness rejects a "
            "repeated seed, so the candidate count IS the seed count."
        ),
    )
    seed_base: int = Field(
        ge=0, description="Seeds are seed_base + index. Recorded, so the run is reproducible."
    )
    slot_token: str = Field(
        min_length=1,
        description=(
            "The literal placeholder the red LM is told to write, e.g. '[PERSON]'. Translated to "
            "the `identity` template variable after parsing. A plain token rather than Jinja "
            "syntax because a mid-size instruct model emits a bracketed placeholder far more "
            "reliably than it emits '{{ identity }}', and a malformed slot is an unparseable "
            "candidate rather than a broken prompt."
        ),
    )
    few_shot: tuple[str, ...] = Field(
        description=(
            "Example frames shown to the red LM, each containing the slot token. They are ALSO "
            "the copy-detector's reference set: a candidate identical to an example is rejected, "
            "because a model echoing the prompt has generated nothing."
        )
    )

    @model_validator(mode="after")
    def _check(self) -> ElicitationSettings:
        if not self.templates:
            raise ValueError("a red-team set must declare at least one elicitation template")
        ids = [spec.id for spec in self.templates]
        if len(set(ids)) != len(ids):
            raise ValueError(f"elicitation template ids must be unique, got {ids}")
        files = [spec.file for spec in self.templates]
        if len(set(files)) != len(files):
            raise ValueError(f"elicitation template files must be unique, got {files}")
        if not self.few_shot:
            raise ValueError(
                "few_shot must be non-empty: an unconstrained 'write a prompt' instruction "
                "produces unparseable output, and the examples are what make the slot format "
                "reliable enough to filter on"
            )
        for example in self.few_shot:
            if self.slot_token not in example:
                raise ValueError(
                    f"few-shot example {example!r} does not contain the slot token "
                    f"{self.slot_token!r}. Showing the model an example without the slot teaches "
                    "it to omit the slot."
                )
        return self


class TargetSettings(BaseModel):
    """Which lattice cells the elicitation targets."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pairs: Literal["confirmatory"] = Field(
        default="confirmatory",
        description=(
            "Gate P8 is 'prompts restricted to the confirmatory axis pairs'. A Literal rather "
            "than a free string so the restriction is enforced by the config failing to load, not "
            "by a reviewer noticing."
        ),
    )
    include_order_1: bool = Field(
        default=False,
        description=(
            "Whether order-1 cells are targeted too. False by default: the estimand is the "
            "interaction, which lives on order-2 cells, and order-1 targets spend generation "
            "budget on what B1 already covers. Order-1 cells are still RENDERED (they carry the "
            "descriptive CellStat rows); this only controls what the red LM is asked to attack."
        ),
    )


class FilterSettings(BaseModel):
    """The degenerate-candidate bounds. Every one of these produces a logged rejection reason."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_words: int = Field(
        ge=1,
        description="Frames shorter than this are stubs ('Describe [PERSON].' with no task).",
    )
    max_words: int = Field(
        ge=2,
        description=(
            "Frames longer than this stop being a prompt and start being a scenario, which makes "
            "the generated text a response to the scenario rather than to the identity."
        ),
    )
    refusal_patterns: tuple[str, ...] = Field(
        description=(
            "Case-insensitive substrings marking a red LM refusal or meta-commentary ('I cannot', "
            "'As an AI'). A refusal parsed as a frame would become an audit prompt asking the "
            "target model about the red LM's safety policy."
        )
    )

    @model_validator(mode="after")
    def _check(self) -> FilterSettings:
        if self.min_words >= self.max_words:
            raise ValueError(f"min_words ({self.min_words}) must be < max_words ({self.max_words})")
        return self


class DedupSettings(BaseModel):
    """How near-duplicate frames are collapsed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    near_duplicate_ratio: float = Field(
        ge=0.0,
        le=100.0,
        description=(
            "rapidfuzz token_sort_ratio at or above which two canonicalized frames are one frame. "
            "100 disables near-duplicate removal and keeps exact dedup only. This matters more "
            "than it looks: near-duplicate frames are near-duplicate ITEM CLUSTERS, and a "
            "cluster-robust standard error computed over them is too small."
        ),
    )


class OnTargetSettings(BaseModel):
    """The manual on-target audit (gate P8: 'human on-target % on a sample reported')."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_size: int = Field(
        ge=1,
        le=1000,
        description="Frames drawn for hand labeling. The binding constraint is human time.",
    )
    seed: int = Field(ge=0, description="Recorded: the sample must be re-drawable by a reviewer.")


class RedTeamConfig(BaseModel):
    """The validated red-team declaration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(min_length=1, description="Must equal the config filename.")
    redteam_version: str = Field(
        min_length=1,
        description=(
            "Stamped on every red-team Prompt as `template_version` and folded into every prompt "
            "id's digest. Bump on ANY change to an elicitation template, the filter bounds, the "
            "dedup threshold or the generator pin."
        ),
    )
    description: str = Field(min_length=1, description="What this red-team set is, in prose.")
    license: str = Field(
        min_length=1,
        description=(
            "Licence of the GENERATED frames. They are model output, so this follows the "
            "generator's licence, not this repository's."
        ),
    )
    authored_at: datetime = Field(
        description=(
            "Recorded, never clock-read. It is the `retrieved_at` of every generated frame's "
            "provenance, so a clock here would make the frame artifact non-deterministic."
        )
    )
    base_prompt_set: str = Field(
        min_length=1,
        description=(
            "The static prompt set whose carriers and article rule are INHERITED. E3 compares "
            "red-team against static on matched cells, so a different carrier here would confound "
            "'red-team frame' with 'different carrier noun'."
        ),
    )
    decoding_key: str = Field(
        min_length=1,
        description=(
            "The decoding profile the AUDIT prompts are stamped with -- the profile the TARGET "
            "model runs under, not the one the red LM generated under. Matching the static set's "
            "profile is what keeps E3 a comparison of prompts rather than of decoding."
        ),
    )

    generator: GeneratorSpec
    elicitation: ElicitationSettings
    target: TargetSettings = TargetSettings()
    filter: FilterSettings
    dedup: DedupSettings
    on_target: OnTargetSettings

    def prompt_set(self, base: PromptSetConfig) -> RedTeamPromptSetConfig:
        """The rendering rules for the audit prompts, inheriting ``base``'s carriers and articles.

        ``base`` is loaded **by the name declared here** (:func:`load_base_prompt_set`), so the
        inheritance recorded in the manifest is the inheritance that happened. The base set's own
        ``template_version`` is copied onto the result, which is what makes "matched carriers" a
        checkable claim rather than a convention.
        """
        return RedTeamPromptSetConfig(
            key=self.key,
            template_version=self.redteam_version,
            decoding_key=self.decoding_key,
            base_prompt_set=self.base_prompt_set,
            base_template_version=base.template_version,
            carriers=base.carriers,
            article_overrides=dict(base.article_overrides),
            license=self.license,
            authored_at=self.authored_at,
        )

    def elicitation_provenance(self, sha256: str) -> Provenance:
        """Provenance for an elicitation template.

        First-party source material — *we* wrote the instruction that asks the red LM for frames —
        so this names the repository, unlike :meth:`provenance_for`, which names the model that
        wrote the frame.
        """
        return Provenance(
            source=f"mapping-the-margins red-team elicitation ({self.key})",
            revision=None,
            version=self.redteam_version,
            license=self.license,
            sha256=sha256,
            retrieved_at=self.authored_at,
        )

    def provenance_for(self, sha256: str, *, model_id: str, revision: str) -> Provenance:
        """Provenance for one generated frame.

        The source is the **red LM**, because that is what produced the text. ``sha256`` digests
        the frame itself, so a frame traces to the exact checkpoint, revision and red-team version
        that emitted it — which is what PROJECT_CONTEXT.md section 3 means by "a red-team seed is
        an Item whose provenance names the generator model, revision, and seed".
        """
        return Provenance(
            source=f"{model_id} (red-team generator, {self.redteam_version})",
            revision=revision,
            version=self.redteam_version,
            license=self.license,
            sha256=sha256,
            retrieved_at=self.authored_at,
        )


class RedTeamPromptSetConfig(BaseModel):
    """Rendering rules for red-team AUDIT prompts (satisfies ``RenderRules``).

    A **separate class** from :class:`~margins.prompts.config.PromptSetConfig`, for the same reason
    P4b's :class:`~margins.prompts.config.ExtractivePromptSetConfig` is one: this set declares no
    template files, because its frames are generated rather than read from disk. Widening the
    static config to cover both would have made ``templates`` optional, which is how a config stops
    validating anything.

    What it does **not** change is the renderer. The leakage guard, the article rule and the
    ``compose_rank`` order all run exactly as they do for the static set — a red-team prompt is
    checked by the same code, against the same lattice vocabulary, for the same set equality.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(min_length=1)
    template_version: str = Field(min_length=1)
    decoding_key: str = Field(min_length=1)
    base_prompt_set: str = Field(min_length=1, description="Where the carriers came from.")
    base_template_version: str = Field(
        min_length=1,
        description=(
            "The base set's version at inheritance time. Recorded so 'matched carriers' is a "
            "checkable claim: if the static set is bumped and the red-team set is not, this says "
            "which static set the red-team prompts were actually matched to."
        ),
    )
    carriers: tuple[CarrierSpec, ...]
    article_overrides: dict[str, str] = Field(default_factory=dict)
    license: str = Field(min_length=1)
    authored_at: datetime

    @property
    def item_benchmark(self) -> str:
        """``redteam`` — the audit-prompt source label (the ``RenderRules`` Protocol)."""
        return REDTEAM_BENCHMARK

    def article_for(self, phrase: str) -> str:
        """The indefinite article, by the **same** rule the static set uses."""
        return indefinite_article(phrase, self.article_overrides)


# --------------------------------------------------------------------------------------
# The Hydra seam
# --------------------------------------------------------------------------------------


def redteam_config_from_mapping(data: Mapping[str, Any]) -> RedTeamConfig:
    """Validate a plain mapping into a :class:`RedTeamConfig`."""
    return RedTeamConfig.model_validate(dict(data))


def default_redteam_config_dir() -> Path:
    """``<repo>/configs/redteam``."""
    return Path(__file__).resolve().parents[3] / "configs" / "redteam"


def default_elicitation_dir() -> Path:
    """``<repo>/configs/redteam/templates``."""
    return default_redteam_config_dir() / ELICITATION_DIRNAME


def load_redteam_config(
    name: str = DEFAULT_REDTEAM_SET, *, config_dir: Path | None = None
) -> RedTeamConfig:
    """Compose ``<config_dir>/<name>.yaml`` with Hydra and validate it.

    Mirrors :func:`margins.prompts.config.load_prompt_config`: Hydra composes plain YAML, the
    result becomes a plain dict, and Pydantic validates. No Hydra structured configs.
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    directory = (config_dir or default_redteam_config_dir()).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"red-team config dir does not exist: {directory}")

    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        composed = compose(config_name=name)
    data = OmegaConf.to_container(composed, resolve=True)
    if not isinstance(data, dict):
        raise TypeError(f"composed red-team config {name!r} is not a mapping, got {type(data)!r}")
    payload = cast(dict[str, Any], data)
    payload.setdefault("key", name)

    config = redteam_config_from_mapping(payload)
    if config.key != name:
        raise ValueError(
            f"red-team set {name!r} declares key={config.key!r}. The filename and the key must "
            "agree, or the manifest records a set name that resolves to a different file."
        )
    return config


def load_base_prompt_set(
    config: RedTeamConfig, *, config_dir: Path | None = None
) -> PromptSetConfig:
    """Load the static prompt set whose carriers and article rule this red-team set inherits."""
    return load_prompt_config(config.base_prompt_set, config_dir=config_dir)
