"""The scoring roster and its rules, as validated configuration (P6).

**The rules this file enforces are the ones that are easiest to violate in prose.** CLAUDE.md's
framing constraint 6 and PROJECT_CONTEXT.md section 9 require at least two *independent* scorers per
harm type, with reported agreement and a drift control. A roster that quietly ships one scorer for a
harm type, or two scorers that are the same classifier twice, satisfies neither -- and no downstream
test would notice, because the pipeline would run perfectly and produce a number.

So the roster is validated, not documented:

* a harm type with fewer than two scorers **fails to load**;
* two scorers for one harm type that declare the same ``mechanism`` **fail to load**, because
  "two independent scorers" means two ways of being wrong, not two instances of one way. Section 9
  is explicit that agreement between two identity-sensitive classifiers measures shared behaviour,
  not shared validity;
* every scorer declares a ``sign``, and a ``SignProbe`` that makes the sign checkable rather than
  asserted (see :mod:`margins.scoring.standardize`).

**Pins are resolved, never authored.** A neural scorer names a repo plus a ``revision`` that
``scripts/pin_scorers.py`` obtained from ``huggingface_hub``; a lexicon names an upstream commit and
the sha256 of the file that was actually fetched. The revision validator rejects a branch name for
the same reason :class:`~margins.inference.models.ModelSpec` does: ``main`` moves, and a scorer that
moved is a different instrument measuring a different thing.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "CONTROL_FAMILIES",
    "DEFAULT_SCORING",
    "MIN_SCORERS_PER_HARM",
    "SCORER_INPUT_PREFIX",
    "SCORING_INPUT_KEY",
    "CalibrationSettings",
    "ControlFamily",
    "DriftSettings",
    "LexiconPin",
    "Mechanism",
    "ModelPin",
    "ScorerSpec",
    "ScoringConfig",
    "SignProbe",
    "default_scoring_config_dir",
    "load_scoring_config",
    "resolve_control_template_dir",
    "scorer_specs_by_id",
    "scoring_config_from_mapping",
    "sorted_scorer_ids",
]

#: The roster this project's confirmatory runs use.
DEFAULT_SCORING: Final[str] = "default_v1"

#: Manifest ``inputs`` keys, alongside :data:`~margins.config.LATTICE_INPUT_KEY`. Constants so the
#: manifest vocabulary is greppable and a test asserts the key rather than a spelling of it.
SCORING_INPUT_KEY: Final[str] = "scoring_config"
SCORER_INPUT_PREFIX: Final[str] = "scorer:"

#: How a scorer decides. The distinction is load-bearing, not descriptive: two scorers for one harm
#: type must not share a mechanism, because that is what "independent" means here. A transformer and
#: a wordlist fail differently; two transformers fail together and agree while doing it.
Mechanism = Literal["neural_classifier", "lexicon"]

#: The two families of known-neutral text the drift control scores (section 9).
#:
#: ``prompt_echo`` -- the P4a rendered prompts themselves. Exactly matched to the model condition by
#: construction, and introduces no new construct choice at all, because these strings already exist
#: and were already audited.
#:
#: ``neutral_declarative`` -- flat declarative sentences built from the SAME descriptors through the
# : SAME renderer. Present because a prompt is an imperative instruction and model output is not, so
# a
#: drift estimate measured only on ``prompt_echo`` might not transfer to the register it is meant to
#: correct. Two families make a style artifact visible; one family makes it invisible.
ControlFamily = Literal["prompt_echo", "neutral_declarative"]

CONTROL_FAMILIES: Final[tuple[ControlFamily, ...]] = ("prompt_echo", "neutral_declarative")

#: A 40-character hex sha. Same rule as `ModelSpec.revision`: a branch name is not a pin.
_SHA_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")

#: Minimum scorers per harm type (framing constraint 6). Not a tunable -- a config that lowers it
#: would be a config that disables the rule, so it is not a config field.
MIN_SCORERS_PER_HARM: Final[int] = 2


class SignProbe(BaseModel):
    """Two texts whose relative harm is not in dispute, used to check a scorer's sign.

    The standardization rule (CLAUDE.md, PROJECT_CONTEXT.md section 8) is "higher = more harm,
    guaranteed by test". A `sign` field alone is an assertion; a probe makes it falsifiable, because
    a flipped sign turns ``harmful > benign`` into ``benign > harmful`` and
    :func:`margins.scoring.standardize.verify_sign` raises.

    Kept deliberately blunt and short. The probe is a wiring check on polarity, not a calibration
    set -- calibration against human labels is :mod:`margins.scoring.calibration`, on 150-300 real
    responses.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    harmful: str = Field(min_length=1, description="Text the scorer must rate as MORE harmful.")
    benign: str = Field(min_length=1, description="Text the scorer must rate as LESS harmful.")


class ModelPin(BaseModel):
    """A pinned Hugging Face classifier checkpoint."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    repo: str = Field(min_length=1, description="Hub repo id, e.g. 'unitary/toxic-bert'.")
    revision: str = Field(
        min_length=1,
        description="RESOLVED commit sha. Written by scripts/pin_scorers.py, never by hand.",
    )
    license: str = Field(min_length=1)
    label: str = Field(
        min_length=1,
        description=(
            "Which output label's score is read. A multi-label classifier exposes several; naming "
            "the one used is the difference between a reproducible number and a plausible one."
        ),
    )
    max_input_tokens: int = Field(default=512, ge=1)

    @field_validator("revision")
    @classmethod
    def _revision_is_a_pin(cls, value: str) -> str:
        if not _SHA_RE.match(value):
            raise ValueError(
                f"revision must be a 40-character commit sha, got {value!r}. A branch name is not "
                "a pin: 'main' moves, and a scorer that moved is a different instrument. Resolve "
                "it with `uv run python scripts/pin_scorers.py --write`."
            )
        return value


class LexiconPin(BaseModel):
    """A pinned wordlist or lexicon file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    source: str = Field(min_length=1, description="Upstream URL or repo the file came from.")
    revision: str | None = Field(
        default=None, description="Upstream commit sha where one exists. None for a packaged asset."
    )
    version: str | None = Field(
        default=None,
        description="Upstream release label where one exists (e.g. a package version).",
    )
    license: str = Field(min_length=1)
    sha256: str | None = Field(
        default=None,
        description=(
            "sha256 of the fetched file. None only for a lexicon that ships inside a pinned "
            "package rather than as a file we retrieve (the package version is the pin there)."
        ),
    )
    path: Path | None = Field(
        default=None, description="Location under data/. None for a packaged lexicon."
    )

    @field_validator("sha256")
    @classmethod
    def _sha256_is_a_digest(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256_RE.match(value):
            raise ValueError(f"sha256 must be 64 hex characters, got {value!r}")
        return value

    @model_validator(mode="after")
    def _a_fetched_file_is_pinned_by_its_bytes(self) -> LexiconPin:
        """A file we retrieve must carry a digest; a packaged asset is pinned by its version."""
        if self.path is not None and self.sha256 is None:
            raise ValueError(
                f"lexicon {self.name!r} names a path but no sha256. A fetched file that is not "
                "digested is not pinned -- upstream can change it under us and nothing "
                "would say so."
            )
        if self.path is None and self.version is None:
            raise ValueError(
                f"lexicon {self.name!r} has neither a path+sha256 nor a version. One of the two is "
                "required, or there is no record of WHICH lexicon produced a score."
            )
        return self


class ScorerSpec(BaseModel):
    """One scorer: what it measures, how it decides, what it is pinned to, and its polarity."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    scorer_id: str = Field(
        min_length=1, description="Stable id, recorded on every ItemScore it produces."
    )
    harm_type: str = Field(min_length=1, description="The harm this scorer measures.")
    mechanism: Mechanism
    description: str = Field(min_length=1, description="What it measures and what it does not.")

    sign: Literal[1, -1] = Field(
        description=(
            "Polarity harmonization. +1 when the scorer's raw output already rises with harm (a "
            "toxicity probability); -1 when it falls with harm (a valence score, negative for "
            "hostile text). Multiplied into the standardized value so that `higher = more harm` "
            "holds for EVERY scorer -- a mis-signed source silently inverts every cross-paradigm "
            "comparison while producing entirely plausible output."
        )
    )
    threshold: float = Field(
        description=(
            "Raw-score cut used ONLY to binarize for Cohen's kappa and for the calibration "
            "precision/recall. Never used to produce `standardized`, which stays continuous."
        )
    )
    sign_probe: SignProbe

    model_pin: ModelPin | None = None
    lexicon_pin: LexiconPin | None = None

    @model_validator(mode="after")
    def _mechanism_matches_its_pin(self) -> ScorerSpec:
        if self.mechanism == "neural_classifier" and self.model_pin is None:
            raise ValueError(
                f"scorer {self.scorer_id!r} is a neural_classifier but has no model_pin"
            )
        if self.mechanism == "lexicon" and self.lexicon_pin is None:
            raise ValueError(
                f"scorer {self.scorer_id!r} is a lexicon scorer but has no lexicon_pin"
            )
        if self.model_pin is not None and self.lexicon_pin is not None:
            raise ValueError(
                f"scorer {self.scorer_id!r} declares both a model_pin and a lexicon_pin. A scorer "
                "has one mechanism, and the manifest records one pin per scorer_id."
            )
        return self

    @property
    def pin_digest(self) -> str:
        """A short, stable digest of what this scorer IS, for the ``score_id`` preimage.

        Covers the pin, so re-scoring after a re-pin produces different score ids rather than rows
        that collide with the old ones. Covers the sign and threshold too: both change the number a
        row reports, and a row whose value changed but whose id did not is the exact failure the
        content-addressed ids elsewhere in this project exist to prevent.
        """
        pin: object
        if self.model_pin is not None:
            pin = (self.model_pin.repo, self.model_pin.revision, self.model_pin.label)
        elif self.lexicon_pin is not None:
            pin = (
                self.lexicon_pin.name,
                self.lexicon_pin.revision,
                self.lexicon_pin.version,
                self.lexicon_pin.sha256,
            )
        else:  # pragma: no cover - unreachable: _mechanism_matches_its_pin requires one.
            pin = ()
        payload = json.dumps(
            [self.scorer_id, self.harm_type, self.mechanism, self.sign, self.threshold, pin],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class DriftSettings(BaseModel):
    """How the identity-drift control's neutral text set is built (section 9)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    families: tuple[ControlFamily, ...] = Field(
        default=CONTROL_FAMILIES,
        min_length=1,
        description="Which control families to build. Both, unless a run is deliberately narrowed.",
    )
    control_prompt_set: str = Field(
        default="drift_control_v1",
        min_length=1,
        description=(
            "Name of the prompt-set config describing the neutral declarative frames, resolved "
            "under the SCORING config dir. It is a full PromptSetConfig rendered by the ordinary "
            "TemplateRenderer, so the control inherits P4a's leakage guard, article rule and "
            "compose order rather than reimplementing them -- which is what makes it matched."
        ),
    )
    control_template_dir: Path = Field(
        default=Path("control_templates"),
        description="Neutral declarative frames, resolved relative to the scoring config dir.",
    )

    @field_validator("families")
    @classmethod
    def _families_are_unique(cls, value: tuple[ControlFamily, ...]) -> tuple[ControlFamily, ...]:
        if len(set(value)) != len(value):
            raise ValueError(f"control families must be unique, got {value!r}")
        return value


class CalibrationSettings(BaseModel):
    """The human-calibration sample (section 9: 150-300 hand-labeled responses)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_size: int = Field(
        default=200,
        ge=150,
        le=300,
        description=(
            "Bounded by the range PROJECT_CONTEXT.md section 9 pre-registers. Bounds rather than a "
            "free integer: a sample quietly shrunk to 20 would still compute a precision."
        ),
    )
    seed: int = Field(
        default=20260918, ge=0, description="Sampling seed. Routed via set_global_seed."
    )


class ScoringConfig(BaseModel):
    """A fully resolved scoring roster. Frozen: a run's scorers cannot change mid-run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(min_length=1, description="Roster name; must match the config filename.")
    description: str = Field(min_length=1)
    scoring_version: str = Field(
        min_length=1, description="Bumped whenever the roster changes what a score MEANS."
    )

    scorers: tuple[ScorerSpec, ...] = Field(min_length=2)
    drift: DriftSettings = DriftSettings()
    calibration: CalibrationSettings = CalibrationSettings()

    @model_validator(mode="after")
    def _scorer_ids_are_unique(self) -> ScoringConfig:
        ids = [spec.scorer_id for spec in self.scorers]
        duplicates = sorted({name for name in ids if ids.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"scorer_id must be unique across the roster, got duplicates {duplicates}. "
                "score_id is built from it, so a duplicate would collide two scorers' rows."
            )
        return self

    @model_validator(mode="after")
    def _every_harm_type_is_triangulated(self) -> ScoringConfig:
        """Framing constraint 6, enforced at load: no single classifier is ever load-bearing.

        Two separate rules, because they fail differently. **Count** catches a harm type that is
        measured once. **Mechanism diversity** catches the subtler one: two scorers that are the
        same kind of thing agree with each other and are wrong together, which is precisely what
        section 9 says agreement cannot detect. A roster that violates either is unloadable, so the
        rule cannot be lost in a refactor the way a comment can.
        """
        for harm_type in sorted({spec.harm_type for spec in self.scorers}):
            group = [spec for spec in self.scorers if spec.harm_type == harm_type]
            if len(group) < MIN_SCORERS_PER_HARM:
                raise ValueError(
                    f"harm_type {harm_type!r} has {len(group)} scorer(s); at least "
                    f"{MIN_SCORERS_PER_HARM} are required. No harm type may depend on a single "
                    "classifier (CLAUDE.md framing constraint 6)."
                )
            mechanisms = {spec.mechanism for spec in group}
            if len(mechanisms) < 2:
                only = next(iter(mechanisms))
                raise ValueError(
                    f"harm_type {harm_type!r} is measured by {len(group)} scorers that all use "
                    f"mechanism {only!r}. Two scorers sharing a mechanism are not independent: "
                    "they fail the same way and agree while doing it, and agreement measures "
                    "shared behaviour, not shared validity (PROJECT_CONTEXT.md section 9)."
                )
        return self

    @property
    def harm_types(self) -> tuple[str, ...]:
        """Every harm type in the roster, sorted. Deterministic: it keys grouped output."""
        return tuple(sorted({spec.harm_type for spec in self.scorers}))

    def scorers_for(self, harm_type: str) -> tuple[ScorerSpec, ...]:
        """The scorers measuring one harm type, in roster order."""
        return tuple(spec for spec in self.scorers if spec.harm_type == harm_type)

    def canonical_json(self) -> str:
        """The exact preimage of :attr:`scoring_hash`, exposed so a reproducer can recompute it."""
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    @property
    def scoring_hash(self) -> str:
        """sha256 of the resolved roster.

        Rides in every ``score_id`` preimage, so editing a threshold, a sign, or a pin produces new
        score ids instead of silently changing what an existing id means.
        """
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------
# The Hydra seam
# --------------------------------------------------------------------------------------


def scoring_config_from_mapping(data: Mapping[str, Any]) -> ScoringConfig:
    """Validate a plain mapping into a :class:`ScoringConfig`."""
    return ScoringConfig.model_validate(dict(data))


def default_scoring_config_dir() -> Path:
    """``<repo>/configs/scoring``. Override it for tests or an installed package."""
    return Path(__file__).resolve().parents[3] / "configs" / "scoring"


def load_scoring_config(
    name: str = DEFAULT_SCORING, *, config_dir: Path | None = None
) -> ScoringConfig:
    """Compose ``<config_dir>/<name>.yaml`` with Hydra and validate it.

    The same seam as :func:`margins.inference.decoding.load_decoding_config`: Hydra composes YAML
    into a plain container and Pydantic validates it. No Hydra structured configs anywhere.
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    directory = (config_dir or default_scoring_config_dir()).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"scoring config dir does not exist: {directory}")

    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        composed = compose(config_name=name)
    data = OmegaConf.to_container(composed, resolve=True)
    if not isinstance(data, dict):
        raise TypeError(f"composed scoring config {name!r} is not a mapping, got {type(data)!r}")
    payload = cast(dict[str, Any], data)
    payload.setdefault("key", name)
    config = scoring_config_from_mapping(payload)
    if config.key != name:
        raise ValueError(
            f"scoring config {name!r} declares key={config.key!r}. The filename and the key must "
            "agree, or the manifest records a roster name that resolves to a different file."
        )
    return config


def resolve_control_template_dir(config: ScoringConfig, *, config_dir: Path | None = None) -> Path:
    """Where the neutral declarative frames live, relative to the scoring config directory."""
    base = (config_dir or default_scoring_config_dir()).resolve()
    return (base / config.drift.control_template_dir).resolve()


def scorer_specs_by_id(config: ScoringConfig) -> dict[str, ScorerSpec]:
    """The roster keyed by ``scorer_id``. Uniqueness is guaranteed by the validator above."""
    return {spec.scorer_id: spec for spec in config.scorers}


def sorted_scorer_ids(specs: Sequence[ScorerSpec]) -> tuple[str, ...]:
    """Scorer ids, sorted. Used wherever output order must not depend on roster order."""
    return tuple(sorted(spec.scorer_id for spec in specs))
