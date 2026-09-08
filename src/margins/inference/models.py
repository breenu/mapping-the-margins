"""Pinned model checkpoints (P5 -- closes open decision #2).

**Why a revision is required, not optional.** A Hugging Face repo id names a moving branch. A repo
can be re-uploaded, re-quantized, or have its chat template edited under a reproducer's feet, and
every one of those changes what the model said without changing anything we recorded. So
:attr:`ModelSpec.revision` is a required 40-hex commit sha, a branch name is rejected by a
validator, and the sha is stamped onto every :class:`~margins.schemas.Response` via
``model_revision`` and into every run manifest. A run is tied to the exact checkpoint it used or it
is not a reproducible run.

**Revisions are RESOLVED, never authored.** Nothing in this project may invent a commit sha.
``scripts/pin_models.py`` resolves each repo's head through ``huggingface_hub`` and writes it into
the YAML; if a repo is missing or gated without access it stops and says so rather than substituting
another model. The configs under ``configs/model/`` are the output of that script, and
``tests/test_model_config.py`` pins what they currently say.

**Instruct checkpoints are a CONSTRUCT CHOICE, not a default.** The P4a prompts are instructions
("Write a short story about ...", "What advice would you give ..."). A base model handed an
instruction tends to continue it as text rather than perform it, which would make the harm signal a
measurement of off-task drift. The confirmatory pair is therefore instruct-tuned, and the cost is
recorded: the two families wrap a prompt in different chat templates, so the wrapper is part of what
each model was asked (see :data:`~margins.inference.decoding.PromptFormat`).

**Quantization is a fixed recorded constant, never an axis** (CLAUDE.md framing constraint 4). It is
not a field here: it lives on ``ExperimentConfig.quant`` as a scalar, and what the backend *actually
did* is recorded separately, because a Linux NF4 intent and a Windows unquantized fallback must not
be indistinguishable in the record.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "CHAT_TEMPLATE_INPUT_PREFIX",
    "CI_MODEL",
    "CONFIRMATORY_MODELS",
    "DEV_MODEL",
    "MODEL_INPUT_KEY",
    "ModelRole",
    "ModelSpec",
    "default_model_config_dir",
    "load_model_config",
    "model_config_from_mapping",
]

#: Manifest ``inputs`` key for the model config's sha256, alongside
#: :data:`~margins.config.LATTICE_INPUT_KEY` and
#: :data:`~margins.inference.decoding.DECODING_INPUT_KEY`.
MODEL_INPUT_KEY: Final[str] = "model_config"

#: Prefix for the per-model chat-template digest in the manifest inputs. Per MODEL, because the
#: template is the checkpoint's, not the decoding profile's -- mirrors P4a's ``template:<id>``.
CHAT_TEMPLATE_INPUT_PREFIX: Final[str] = "chat_template:"

#: What a checkpoint is for. ``confirmatory`` models carry the published result; ``dev`` is the
#: laptop-runnable model every local smoke test uses; ``ci`` is the tiny random checkpoint the
#: nightly job loads (no weights worth downloading, no gated repo, no HF token).
ModelRole = Literal["confirmatory", "dev", "ci"]

#: The confirmatory 7-8B pair (open decision #2, resolved 9 Sep 2026). One family was the locked
#: scope; two are declared because the budget arithmetic and the pilot both range over the pair.
CONFIRMATORY_MODELS: Final[tuple[str, ...]] = (
    "llama31_8b_instruct",
    "mistral7b_instruct_v03",
)
#: Laptop-runnable development model. Every local determinism / resumability test targets this.
DEV_MODEL: Final[str] = "llama32_1b_instruct"
#: Tiny random checkpoint for CI. PROJECT_CONTEXT.md section 7 asked for its availability to be
#: verified at P5; it was (9 Sep 2026).
CI_MODEL: Final[str] = "tiny_random_llama"

_REVISION_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")
_REPO_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


class ModelSpec(BaseModel):
    """One pinned checkpoint. Frozen: a run's model cannot change mid-run."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    name: str = Field(
        min_length=1,
        description="Config name (`configs/model/<name>.yaml`). Appears in response ids and paths.",
    )
    model_id: str = Field(min_length=1, description="Hugging Face repo id, `owner/repo`.")
    revision: str = Field(
        description=(
            "RESOLVED 40-hex commit sha. Never a branch, never hand-authored -- "
            "`scripts/pin_models.py` resolves it through huggingface_hub and writes it here."
        )
    )
    license: str = Field(
        min_length=1,
        description=(
            "Required and deliberately not defaulted: an unrecorded license blocks the Zenodo "
            "release, exactly as for Provenance."
        ),
    )
    role: ModelRole
    family: str = Field(min_length=1, description="Architecture family, e.g. 'llama' or 'mistral'.")
    gated: bool = Field(
        description=(
            "Whether the upstream repo requires an accepted licence before download. Recorded so a "
            "reproducer is told up front that they need their own access, rather than hitting a "
            "401 mid-run."
        )
    )
    note: str = Field(
        min_length=1,
        description="Why this checkpoint is in the design. An undocumented model is an unexamined "
        "construct choice.",
    )

    @field_validator("revision")
    @classmethod
    def _revision_is_a_resolved_sha(cls, value: str) -> str:
        if not _REVISION_RE.match(value):
            raise ValueError(
                f"revision {value!r} is not a 40-character lowercase hex commit sha. A branch name "
                "('main') is a MOVING target: the weights, the tokenizer or the chat template can "
                "change under a reproducer without anything we recorded changing. Run "
                "`scripts/pin_models.py --write` to resolve it."
            )
        return value

    @field_validator("model_id")
    @classmethod
    def _repo_id_is_well_formed(cls, value: str) -> str:
        if not _REPO_ID_RE.match(value):
            raise ValueError(f"model_id {value!r} is not a Hugging Face 'owner/repo' id")
        return value

    @property
    def short_name(self) -> str:
        """Filesystem- and id-safe label used in response ids and response file paths."""
        return self.name


# --------------------------------------------------------------------------------------
# The Hydra seam
# --------------------------------------------------------------------------------------


def model_config_from_mapping(data: Mapping[str, Any]) -> ModelSpec:
    """Validate a plain mapping into a :class:`ModelSpec`."""
    return ModelSpec.model_validate(dict(data))


def default_model_config_dir() -> Path:
    """``<repo>/configs/model``. Override it for tests or an installed package."""
    return Path(__file__).resolve().parents[3] / "configs" / "model"


def load_model_config(name: str, *, config_dir: Path | None = None) -> ModelSpec:
    """Compose ``<config_dir>/<name>.yaml`` with Hydra and validate it.

    The composed mapping does not carry its own name, so ``name`` is injected -- that keeps the YAML
    from repeating its filename and from being able to disagree with it.
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    directory = (config_dir or default_model_config_dir()).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"model config dir does not exist: {directory}")

    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        composed = compose(config_name=name)
    data = OmegaConf.to_container(composed, resolve=True)
    if not isinstance(data, dict):
        raise TypeError(f"composed model config {name!r} is not a mapping, got {type(data)!r}")
    payload = dict(cast(Mapping[str, Any], data))
    payload.setdefault("name", name)
    if payload["name"] != name:
        raise ValueError(
            f"model config {name!r} declares name {payload['name']!r}; the file name is the "
            "authority and the two must not disagree"
        )
    return model_config_from_mapping(payload)
