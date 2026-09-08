"""The resolved decoding configuration and its hash (P5).

P4a stamps ``decoding_key = "greedy_v1"`` on every :class:`~margins.schemas.Prompt` and left a
``TODO(P5)`` saying the resolved config lands with the backend. This module is that landing.

**Why a hash at all.** ``Response.decoding_key`` names a profile; it does not say what the profile
*was*. ``top_p``, ``top_k``, ``max_new_tokens`` and ``repetition_penalty`` are otherwise recorded
nowhere, so a response would be uninterpretable without the config that produced it. Determinism is
a contract (CLAUDE.md invariant 5), which means a response has to be self-describing:
:attr:`DecodingConfig.decoding_hash` is what makes ``greedy_v1`` checkable rather than merely
labelled.

**Temperature is required and never defaulted.** A silently defaulted temperature is the single
easiest way to publish a sampled result while believing it was greedy. The field has no default, and
the validators refuse anything that is neither cleanly greedy (``do_sample=False``, ``temperature``
exactly 0, no ``top_p``/``top_k``) nor cleanly sampled (``do_sample=True``, ``temperature > 0``).

**What is deliberately NOT here.**

* ``batch_size`` and the device belong to the *run*, not to the decoding profile: they change
  floating-point reduction order without changing decoding semantics. They are recorded in the run
  manifest, and the determinism claim is scoped to a fixed batch size accordingly
  (:func:`margins.repro.residual_nondeterminism_note`).
* The chat template's own sha256 is per-*model*, so folding it in here would make two models sharing
  ``greedy_v1`` disagree on a hash that is supposed to name a decoding profile. It rides in the run
  manifest under ``chat_template:<model>`` instead, mirroring P4a's ``template:<id>`` convention.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "DECODING_INPUT_KEY",
    "DEFAULT_DECODING",
    "DecodingConfig",
    "PromptFormat",
    "decoding_config_from_mapping",
    "default_decoding_config_dir",
    "load_decoding_config",
]

#: The profile P4a already stamps on every prompt. Named here so the two cannot drift apart in
#: spelling: ``TemplateRenderer`` writes the string, ``HFBackend`` refuses a prompt whose
#: ``decoding_key`` does not match the profile it was handed.
DEFAULT_DECODING: Final[str] = "greedy_v1"

#: Manifest ``inputs`` key for the decoding config's sha256, alongside
#: :data:`~margins.config.LATTICE_INPUT_KEY`. A constant so the manifest vocabulary is greppable.
DECODING_INPUT_KEY: Final[str] = "decoding_config"

#: How a rendered prompt reaches the model.
#:
#: ``chat`` -- apply the checkpoint's own chat template as a single user turn. This is a CONSTRUCT
#: CHOICE, and the one the confirmatory set makes: the P4a prompts are instructions ("Write a short
#: story about..."), the pinned checkpoints are instruct-tuned, and feeding an instruct model raw
#: completion text puts it off-distribution. The cost is that the two model families wrap the same
#: prompt differently, so the wrapper is part of what each model was asked -- which is why the
#: applied template's sha256 is recorded per model in the run manifest rather than assumed equal.
#:
#: ``raw`` -- feed ``Prompt.rendered_text`` verbatim. Kept for base models and for the tiny CI
#: checkpoint, which has no chat template at all.
PromptFormat = Literal["chat", "raw"]


class DecodingConfig(BaseModel):
    """A fully resolved decoding profile. Frozen: a run's decoding cannot change mid-run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(
        min_length=1,
        description=(
            "Profile name. Must equal the `decoding_key` stamped on the prompts this profile is "
            "used for -- the backend checks, rather than trusting that they match."
        ),
    )
    description: str = Field(min_length=1, description="What this profile is for, in prose.")

    prompt_format: PromptFormat = Field(
        description="How a rendered prompt reaches the model. A construct choice; see PromptFormat."
    )
    pinned_chat_date: str = Field(
        min_length=1,
        description=(
            "Date handed to the chat template as `date_string`. Llama's stock template injects a "
            "LIVE 'Today Date' header, which would make the same prompt render differently every "
            "day and quietly break reproducibility. Pinned here and verified at render time."
        ),
    )
    max_input_tokens: int = Field(
        ge=1, description="Truncation bound on the tokenized prompt. Recorded: it can change input."
    )

    do_sample: bool = Field(description="Greedy when False. No default: this is half the profile.")
    temperature: float = Field(
        ge=0.0,
        description=(
            "REQUIRED and never defaulted. A silently defaulted temperature is the easiest way to "
            "publish a sampled result while believing it was greedy."
        ),
    )
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    max_new_tokens: int = Field(ge=1)
    repetition_penalty: float = Field(
        default=1.0, gt=0.0, description="1.0 means no penalty. Recorded either way."
    )

    @model_validator(mode="after")
    def _greedy_and_sampling_are_not_mixed(self) -> DecodingConfig:
        """Reject a profile that is neither cleanly greedy nor cleanly sampled.

        Transformers will happily accept ``do_sample=False`` alongside ``temperature=0.7`` and
        silently ignore the temperature. The resulting ``Response`` would then record a temperature
        the run never used, which is worse than a crash: it is a plausible-looking lie in the
        provenance trail.
        """
        if self.do_sample:
            if self.temperature <= 0.0:
                raise ValueError(
                    f"decoding {self.key!r} sets do_sample=True with temperature="
                    f"{self.temperature}; sampling at temperature 0 is greedy decoding wearing a "
                    "sampling label. Set do_sample=False instead."
                )
            return self

        if self.temperature != 0.0:
            raise ValueError(
                f"decoding {self.key!r} is greedy (do_sample=False) but sets temperature="
                f"{self.temperature}. Generation would IGNORE it and the Response would record a "
                "temperature the run never used. Greedy profiles must set temperature: 0.0."
            )
        offenders = {
            name: value
            for name, value in (("top_p", self.top_p), ("top_k", self.top_k))
            if value is not None
        }
        if offenders:
            raise ValueError(
                f"decoding {self.key!r} is greedy but declares {offenders}; truncation warpers do "
                "nothing under greedy decoding and recording them implies a sampling run."
            )
        return self

    # ---------------------------------------------------------------- the hash

    def canonical_json(self) -> str:
        """The exact preimage of :attr:`decoding_hash`, exposed so a reproducer can recompute it."""
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @property
    def decoding_hash(self) -> str:
        """Content-addressed hash of the fully resolved profile.

        Same construction as :func:`margins.config.config_hash`: sorted-key JSON of the JSON-mode
        dump, so the value depends on the profile's content and not on the key order of the YAML it
        was composed from. Stamped on every :class:`~margins.schemas.Response`.
        """
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    # ---------------------------------------------------------------- generation kwargs

    def generation_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``model.generate``, stated explicitly rather than defaulted.

        Only the parameters this profile actually declares are passed, and nothing is left to the
        checkpoint's bundled ``generation_config`` -- several instruct checkpoints ship a non-zero
        default temperature and a ``top_p``, which would otherwise apply silently.
        """
        kwargs: dict[str, Any] = {
            "do_sample": self.do_sample,
            "max_new_tokens": self.max_new_tokens,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.do_sample:
            kwargs["temperature"] = self.temperature
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
            if self.top_k is not None:
                kwargs["top_k"] = self.top_k
        return kwargs


# --------------------------------------------------------------------------------------
# The Hydra seam
# --------------------------------------------------------------------------------------


def decoding_config_from_mapping(data: Mapping[str, Any]) -> DecodingConfig:
    """Validate a plain mapping into a :class:`DecodingConfig`."""
    return DecodingConfig.model_validate(dict(data))


def default_decoding_config_dir() -> Path:
    """``<repo>/configs/decoding``. Override it for tests or an installed package."""
    return Path(__file__).resolve().parents[3] / "configs" / "decoding"


def load_decoding_config(
    name: str = DEFAULT_DECODING, *, config_dir: Path | None = None
) -> DecodingConfig:
    """Compose ``<config_dir>/<name>.yaml`` with Hydra and validate it.

    Mirrors :func:`margins.prompts.config.load_prompt_config`: Hydra composes plain YAML, the result
    becomes a plain dict, and Pydantic validates. No Hydra structured configs.
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    directory = (config_dir or default_decoding_config_dir()).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"decoding config dir does not exist: {directory}")

    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        composed = compose(config_name=name)
    data = OmegaConf.to_container(composed, resolve=True)
    if not isinstance(data, dict):
        raise TypeError(f"composed decoding config {name!r} is not a mapping, got {type(data)!r}")
    return decoding_config_from_mapping(cast(Mapping[str, Any], data))
