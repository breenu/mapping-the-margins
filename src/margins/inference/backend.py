"""``HFBackend`` -- the Hugging Face inference backend (P5).

The only GPU-touching, non-portable stage in the pipeline, and the first one that produces a
:class:`~margins.schemas.Response`. Three things it is responsible for, none of which can be
retrofitted later:

1. **A self-describing response.** ``model_revision``, ``seed``, ``temperature``, ``decoding_key``,
   ``decoding_hash`` and ``quant`` exist on the record so a response is interpretable without the
   config that produced it. This module is what fills them with truth rather than intent -- in
   particular ``quant`` records what the load ACTUALLY did, so a Windows unquantized fallback is
   never mistaken for the Linux NF4 constant.
2. **Both measurement paths.** ``generate`` for ``open_gen`` (the MVP path, where RQ2 lives) and
   ``sequence_logprob`` for sequence log-likelihood, which P2 exercises for real with CrowS-Pairs.
   The log-prob path scores the **raw** sequence: no chat template is applied, which is how
   CrowS-Pairs log-likelihood is computed in the literature and what keeps the figure comparable
   to published work. That is a construct choice, recorded in ``configs/decoding/pll_v1.yaml`` and
   folded into ``decoding_hash`` rather than left as an accident of which code path ran.
3. **A determinism claim that is scoped honestly.** Same seed and greedy decoding give identical
   token sequences *at a fixed thread count and batch size*, on the recorded torch version and BLAS
   backend. Nothing here claims unconditional bit-identity, and
   :func:`margins.repro.residual_nondeterminism_note` is the exact wording that travels with every
   manifest.

**Lazy imports are deliberate.** ``torch`` and ``transformers`` are an optional extra. This module
must import cleanly on the core environment so that ``mypy src``, ``ruff`` and the non-inference
test suite all run on a machine with no model stack, exactly as :mod:`margins.repro` already does.

**Open decision #7 is CLOSED (20 Sep 2026, with P2/P4b).** A ``pair_pll`` prompt carries both
sequences on one record -- ``rendered_text_stereo`` / ``rendered_text_antistereo`` -- mirroring
decision #6's one ``Response`` carrying both sequence log-probs. :meth:`HFBackend.run` no longer
refuses a paired prompt; it routes it to :meth:`HFBackend.score_pairs`. What it *does* refuse is a
**mixed** call, because the two paradigms run under different decoding profiles and this backend
holds one.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from margins.config import QuantSetting
from margins.inference.decoding import DecodingConfig
from margins.inference.models import ModelSpec
from margins.repro import MAX_SEED, environment_fingerprint, set_global_seed
from margins.schemas import Prompt, Response, TaskType

__all__ = [
    "RESPONSE_ID_DIGEST_LENGTH",
    "HFBackend",
    "LoadReport",
    "SequenceLogprob",
    "cuda_compute_dtype",
    "make_response_id",
]

#: Matches ``make_prompt_id``'s convention: readable natural key + a short content digest.
RESPONSE_ID_DIGEST_LENGTH: Final[int] = 12

_RESPONSE_ID_SEPARATOR: Final[str] = "::"
_DIGEST_SEPARATOR: Final[str] = "\x1f"

#: Date shapes a chat template might inject. ``%d %b %Y`` is Llama's, ISO is the other common one.
#: "December 2023" (Llama's static cutting-knowledge line) deliberately matches neither -- it is a
#: fixed string in the template and carries no clock.
_DATE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\b\d{1,2} [A-Z][a-z]{2} \d{4}\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
)


def cuda_compute_dtype(torch: Any) -> Any:
    """The float type to compute in on THIS GPU: bfloat16 on Ampere+, float16 before it.

    **bfloat16 requires compute capability 8.0.** The reference hardware for this project is a
    free-tier Kaggle **T4**, which is Turing (7.5) and has no bf16 support at all. Handing
    ``bnb_4bit_compute_dtype=torch.bfloat16`` to bitsandbytes there is not a performance choice, it
    is a wrong one -- and it is silent, because nothing in the API rejects it.

    Selected from the device rather than hardcoded, so the same code is correct on a T4 and on the
    A100-class card a reproducer might use. The dtype actually chosen is recorded in
    :class:`LoadReport` and in the run manifest, because it is part of what produced the numbers.
    """
    major, _minor = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def make_response_id(
    *,
    prompt_id: str,
    model_name: str,
    model_id: str,
    revision: str,
    seed: int,
    sample_index: int,
    decoding_hash: str,
) -> str:
    """A deterministic, content-addressed ``response_id``.

    Shape: ``<prompt_id>::<model_name>::<seed>::<sample_index>::<digest12>``, mirroring
    :func:`margins.prompts.render.make_prompt_id`.

    Determinism is what makes the run harness resumable: the id of a unit of work is computable
    **before** the work is done, so a restart can tell what is already on disk from what still has
    to run without re-generating anything to find out. The digest covers the pinned revision and the
    decoding hash, so re-pinning a model or editing a decoding profile produces different ids rather
    than silently appending incompatible rows to an existing run file.
    """
    preimage = _DIGEST_SEPARATOR.join(
        (
            prompt_id,
            model_name,
            model_id,
            revision,
            str(seed),
            str(sample_index),
            decoding_hash,
        )
    )
    digest = hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:RESPONSE_ID_DIGEST_LENGTH]
    return _RESPONSE_ID_SEPARATOR.join(
        (prompt_id, model_name, str(seed), str(sample_index), digest)
    )


@dataclass(frozen=True, slots=True)
class SequenceLogprob:
    """The result of scoring a sequence: per-token log-probabilities and their sum."""

    token_logprobs: tuple[float, ...]
    total: float
    n_tokens: int


class LoadReport(BaseModel):
    """What loading the model actually did, as opposed to what was requested.

    Recorded into the run manifest. The ``quant_requested``/``quant_actual`` split is the point:
    CLAUDE.md framing constraint 4 makes quantization a *fixed recorded constant*, and a constant
    that silently differs from what ran is worse than no constant. A Windows dev load falls back to
    unquantized, and this says so, with the reason.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    model_id: str
    revision: str
    quant_requested: QuantSetting
    quant_actual: QuantSetting
    quant_fallback_reason: str | None = Field(
        default=None, description="Why the actual quantization differs from the requested one."
    )
    device: str
    dtype: str
    torch_version: str
    torch_num_threads: int | None
    blas_backend: str | None
    prompt_format: str
    chat_template_sha256: str | None = Field(
        default=None,
        description=(
            "sha256 of the checkpoint's chat template source. Per MODEL, not per decoding profile: "
            "two families wrap the same prompt differently, and the wrapper is part of what each "
            "model was asked."
        ),
    )
    pad_token_source: str = Field(
        description="Where the pad token came from. 'eos' means the tokenizer declared none."
    )


class HFBackend:
    """Runs prompts through a pinned Hugging Face checkpoint (the ``InferenceBackend`` Protocol).

    Args:
        spec: The pinned checkpoint. Its ``revision`` is a resolved commit sha, never a branch.
        decoding: The fully resolved decoding profile whose hash is stamped on every response.
        seed: Seed for the run. Routed through :func:`margins.repro.set_global_seed` before every
            batch, so no stochastic path reaches a hidden global RNG.
        quant: The *requested* constant. What actually happened is recorded on
            :attr:`load_report` and on every ``Response``.
        device: Force a device. Defaults to CUDA when available, else CPU.
        model, tokenizer: Pre-built objects, for tests. When given, nothing is downloaded.
    """

    def __init__(
        self,
        spec: ModelSpec,
        decoding: DecodingConfig,
        *,
        seed: int,
        quant: QuantSetting = "nf4",
        device: str | None = None,
        model: Any = None,
        tokenizer: Any = None,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"seed must be an int, got {type(seed).__name__}")
        if not 0 <= seed <= MAX_SEED:
            raise ValueError(f"seed must be in [0, {MAX_SEED}], got {seed}")

        self._spec = spec
        self._decoding = decoding
        self._seed = seed
        self._quant_requested: QuantSetting = quant
        self._device_override = device
        self._model = model
        self._tokenizer = tokenizer
        self._load_report: LoadReport | None = None

    # ---------------------------------------------------------------- accessors

    @property
    def model_id(self) -> str:
        """The ``InferenceBackend`` Protocol's required attribute."""
        return self._spec.model_id

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    @property
    def decoding(self) -> DecodingConfig:
        return self._decoding

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def load_report(self) -> LoadReport:
        """The load report, loading the model first if it has not been loaded yet."""
        self._ensure_loaded()
        assert self._load_report is not None  # set by _ensure_loaded
        return self._load_report

    # ---------------------------------------------------------------- loading

    @staticmethod
    def _torch() -> Any:
        import torch

        return torch

    def _resolve_quant(self) -> tuple[QuantSetting, str | None]:
        """Decide what quantization can ACTUALLY be used here, and say why if it differs.

        NF4 is a bitsandbytes/CUDA path. On the Windows dev box neither is present, so the dev model
        loads unquantized -- which is fine for what the dev box is for, and is recorded rather than
        papered over.
        """
        if self._quant_requested != "nf4":
            return self._quant_requested, None
        if importlib.util.find_spec("bitsandbytes") is None:
            return "none", (
                "bitsandbytes is not installed (it is Linux-gated in this project's lock, "
                "PROJECT_CONTEXT.md section 7); loaded unquantized instead of NF4"
            )
        if not self._torch().cuda.is_available():
            return "none", (
                "no CUDA device available and bitsandbytes NF4 requires one; "
                "loaded unquantized instead of NF4"
            )
        return "nf4", None

    def _ensure_loaded(self) -> None:
        if self._load_report is not None:
            return

        torch = self._torch()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        quant_actual, fallback_reason = self._resolve_quant()
        device = self._device_override or ("cuda" if torch.cuda.is_available() else "cpu")
        # float32 on CPU: bf16 CPU kernels are slower AND vary more across builds, and the CPU path
        # is exactly where the bit-level determinism claim lives. On CUDA the type is chosen from
        # the device -- see cuda_compute_dtype; a T4 cannot do bf16 at all.
        dtype = cuda_compute_dtype(torch) if device == "cuda" else torch.float32

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._spec.model_id, revision=self._spec.revision
            )
        tokenizer = self._tokenizer

        pad_token_source = "tokenizer"
        if tokenizer.pad_token_id is None:
            # Decoder-only checkpoints usually declare no pad token. Batched generation needs one;
            # reusing eos is the standard choice and is recorded so it is not an invisible default.
            tokenizer.pad_token = tokenizer.eos_token
            pad_token_source = "eos"

        if self._model is None:
            kwargs: dict[str, Any] = {"revision": self._spec.revision, "dtype": dtype}
            if quant_actual == "nf4":
                from transformers import BitsAndBytesConfig

                # The recorded constant is plain NF4. Double quantization is deliberately OFF: it
                # would make "4-bit NF4" ambiguous in the manifest, and the ~0.4 GB it saves is not
                # needed for an 8B model on a 16 GB T4.
                #
                # The compute dtype MUST match `dtype` and must come from the device: a T4 has no
                # bf16, and passing it there is silently wrong rather than an error.
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=False,
                )
                kwargs["device_map"] = {"": 0}
            self._model = AutoModelForCausalLM.from_pretrained(self._spec.model_id, **kwargs)
            if quant_actual != "nf4":
                self._model.to(device)
        self._model.eval()
        self._neutralize_bundled_generation_config()

        template = getattr(tokenizer, "chat_template", None)
        fingerprint = environment_fingerprint()
        self._load_report = LoadReport(
            model_id=self._spec.model_id,
            revision=self._spec.revision,
            quant_requested=self._quant_requested,
            quant_actual=quant_actual,
            quant_fallback_reason=fallback_reason,
            device=device,
            dtype=str(dtype).removeprefix("torch."),
            torch_version=str(torch.__version__),
            torch_num_threads=fingerprint.torch_num_threads,
            blas_backend=fingerprint.blas_backend,
            prompt_format=self._decoding.prompt_format,
            chat_template_sha256=(
                None
                if template is None
                else hashlib.sha256(str(template).encode("utf-8")).hexdigest()
            ),
            pad_token_source=pad_token_source,
        )

    def _neutralize_bundled_generation_config(self) -> None:
        """Clear the sampling parameters the checkpoint ships with.

        Instruct checkpoints bundle a ``generation_config.json`` -- Llama-3.2-1B-Instruct ships
        ``temperature=0.6, top_p=0.9`` -- which transformers applies to any parameter the caller
        does not pass. Under greedy decoding those are ignored (transformers warns about it), but
        that is luck rather than design: the moment a profile sets ``do_sample=True`` without naming
        a ``top_p``, the checkpoint's 0.9 would silently apply and would appear in **no**
        ``decoding_hash``, because it never came from our config.

        Clearing them makes :class:`~margins.inference.decoding.DecodingConfig` the sole
        authority on how tokens are chosen. A parameter this project did not declare is not merely
        unrecorded -- it is not used.
        """
        config = getattr(self._model, "generation_config", None)
        if config is None:  # pragma: no cover - a stub model in tests
            return
        for parameter in ("temperature", "top_p", "top_k", "typical_p", "repetition_penalty"):
            if hasattr(config, parameter):
                setattr(config, parameter, None)
        if hasattr(config, "do_sample"):
            config.do_sample = False

    # ---------------------------------------------------------------- prompt formatting

    def format_prompt(self, text: str) -> str:
        """Wrap a rendered prompt the way this checkpoint expects to be addressed.

        ``chat`` applies the checkpoint's own template as a single user turn with no system message
        -- a construct choice, documented at :data:`~margins.inference.decoding.PromptFormat`. The
        date handed to the template is PINNED and then verified, because a template that injects a
        live "Today Date" would render the same prompt differently every day and quietly destroy
        reproducibility.
        """
        if self._decoding.prompt_format == "raw":
            return text

        self._ensure_loaded()
        tokenizer = self._tokenizer
        if getattr(tokenizer, "chat_template", None) is None:
            raise ValueError(
                f"{self._spec.model_id!r} declares no chat template, but the decoding profile "
                f"{self._decoding.key!r} sets prompt_format='chat'. Use prompt_format='raw' for a "
                "base or randomly-initialised checkpoint."
            )
        wrapped = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            date_string=self._decoding.pinned_chat_date,
        )
        self._check_no_clock_drift(str(wrapped))
        return str(wrapped)

    def _check_no_clock_drift(self, wrapped: str) -> None:
        """Refuse a chat wrapper carrying any date other than the pinned one.

        Llama's stock template injects ``Today Date: <today>``. If a template version ignores the
        ``date_string`` variable and reads the clock instead, the prompt text -- and therefore every
        response and every prompt-level digest -- changes from one day to the next. That is exactly
        the kind of drift the determinism contract exists to catch, so it fails loudly here rather
        than showing up as unexplained variance weeks later.
        """
        pinned = self._decoding.pinned_chat_date
        for pattern in _DATE_PATTERNS:
            for found in pattern.findall(wrapped):
                if found != pinned:
                    raise ValueError(
                        f"chat template for {self._spec.model_id!r} rendered date {found!r}, which "
                        f"is not the pinned {pinned!r}. The template is reading a clock, so the "
                        "same prompt would render differently tomorrow and the run would not be "
                        "reproducible. Pin the template version or set prompt_format='raw'."
                    )

    # ---------------------------------------------------------------- generation

    def _check_prompt(self, prompt: Prompt, *, expect: TaskType) -> None:
        if prompt.task_type is not expect:
            raise ValueError(
                f"prompt {prompt.prompt_id!r} has task_type {prompt.task_type.value!r}, expected "
                f"{expect.value!r}"
            )
        if prompt.decoding_key != self._decoding.key:
            raise ValueError(
                f"prompt {prompt.prompt_id!r} is stamped decoding_key "
                f"{prompt.decoding_key!r} but this backend holds profile "
                f"{self._decoding.key!r}. Running a prompt set under a profile it was not stamped "
                "for would put a decoding_hash on the Response that does not describe the prompt "
                "that produced it."
            )

    def generate(
        self,
        prompts: Sequence[Prompt],
        *,
        seed: int | None = None,
        sample_index: int = 0,
        batch_size: int = 1,
        token_logprobs: bool = False,
    ) -> list[Response]:
        """Generate one response per prompt.

        Args:
            prompts: ``open_gen`` prompts stamped with this backend's decoding key.
            seed: Overrides the backend's run seed. Under a sampled profile, distinct seeds are
                what produce distinct samples of the same prompt.
            sample_index: Which sample of each prompt this is. Part of the response id, so repeated
                sampling of one prompt yields distinct, predictable ids.
            batch_size: Prompts per forward pass. **The determinism claim is scoped to a fixed batch
                size**: padding changes reduction order, so two batchings of the same prompt can
                differ in the last bits. Recorded in the run manifest.
            token_logprobs: Record per-token log-probabilities of the generated text. Off by
                default -- it adds one float per generated token to every JSONL row and nothing in
                P6 reads it.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        for prompt in prompts:
            self._check_prompt(prompt, expect=TaskType.OPEN_GEN)

        self._ensure_loaded()
        torch = self._torch()
        effective_seed = self._seed if seed is None else seed
        report = self.load_report

        responses: list[Response] = []
        for start in range(0, len(prompts), batch_size):
            batch = list(prompts[start : start + batch_size])
            # Re-seeded per batch, from the seed alone, so a resumed run reaches every batch with
            # exactly the RNG state an uninterrupted run would have had. Without this, resumption
            # would silently change sampled output -- and the point of the harness is that a
            # restart is invisible in the results.
            #
            # NOTE this is why distinct samples of one prompt come from distinct SEEDS, never from
            # a repeated call at the same seed: two such calls are re-seeded identically and would
            # return the same text while carrying different response ids.
            set_global_seed(effective_seed)
            texts, logprob_rows = self._generate_batch(batch, torch, token_logprobs)
            for prompt, text, row in zip(batch, texts, logprob_rows, strict=True):
                responses.append(
                    Response(
                        response_id=make_response_id(
                            prompt_id=prompt.prompt_id,
                            model_name=self._spec.name,
                            model_id=self._spec.model_id,
                            revision=self._spec.revision,
                            seed=effective_seed,
                            sample_index=sample_index,
                            decoding_hash=self._decoding.decoding_hash,
                        ),
                        prompt_id=prompt.prompt_id,
                        model_id=self._spec.model_id,
                        model_revision=self._spec.revision,
                        seed=effective_seed,
                        temperature=self._decoding.temperature,
                        decoding_key=self._decoding.key,
                        decoding_hash=self._decoding.decoding_hash,
                        # What ACTUALLY happened, never the intended constant.
                        quant=report.quant_actual,
                        text=text,
                        token_logprobs=row,
                    )
                )
        return responses

    def _generate_batch(
        self, batch: Sequence[Prompt], torch: Any, want_logprobs: bool
    ) -> tuple[list[str], list[tuple[float, ...] | None]]:
        tokenizer = self._tokenizer
        model = self._model
        chat = self._decoding.prompt_format == "chat"
        texts = [self.format_prompt(prompt.rendered_text) for prompt in batch]

        # Left padding: with a decoder-only model, right padding would put pad tokens between the
        # prompt and the first generated token.
        tokenizer.padding_side = "left"
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._decoding.max_input_tokens,
            # A chat template already emits BOS; re-adding it would double the special tokens.
            add_special_tokens=not chat,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                **self._decoding.generation_kwargs(),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=want_logprobs,
            )

        prompt_length = int(encoded["input_ids"].shape[1])
        generated = outputs.sequences[:, prompt_length:]
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

        rows: list[tuple[float, ...] | None] = [None] * len(batch)
        if want_logprobs:
            transitions = model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            for index in range(len(batch)):
                finite = [
                    float(value)
                    for value in transitions[index].tolist()
                    if value == value and value != float("-inf")  # drop NaN/-inf padding
                ]
                rows[index] = tuple(finite)
        return [text.strip() for text in decoded], rows

    # ---------------------------------------------------------------- log-probabilities

    def sequence_logprob(self, text: str, *, prefix: str | None = None) -> SequenceLogprob:
        """Autoregressive log-probability of ``text``, optionally conditioned on ``prefix``.

        Construct caveat, stated because the paper depends on it: CrowS-Pairs' PLL was designed for
        *masked* LMs. The pinned checkpoints are decoder-only, so this is autoregressive sentence
        log-likelihood, a documented substitute rather than the same quantity
        (PROJECT_CONTEXT.md section 9).

        The first token is excluded from the sum: nothing predicts it, so it has no conditional
        log-probability. When ``prefix`` is given, only the tokens of ``text`` are scored.
        """
        if not text:
            raise ValueError("cannot score an empty sequence")
        self._ensure_loaded()
        torch = self._torch()
        tokenizer = self._tokenizer
        model = self._model

        full = prefix + text if prefix else text
        encoded = tokenizer(full, return_tensors="pt", add_special_tokens=True).to(model.device)
        input_ids = encoded["input_ids"]
        if int(input_ids.shape[1]) < 2:
            raise ValueError(
                f"sequence {full!r} tokenizes to fewer than 2 tokens; there is no conditional "
                "log-probability to compute"
            )

        start = 1
        if prefix:
            prefix_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=True)[
                "input_ids"
            ]
            # Token boundaries can shift where a prefix and its continuation merge into one token;
            # max() keeps the slice legal, and the caller is scoring a documented approximation.
            start = max(1, int(prefix_ids.shape[1]))

        with torch.no_grad():
            logits = model(**encoded).logits
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        targets = input_ids[0, start:]
        predictors = logprobs[0, start - 1 : -1, :]
        per_token = predictors.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        values = tuple(float(value) for value in per_token.tolist())
        return SequenceLogprob(
            token_logprobs=values, total=float(sum(values)), n_tokens=len(values)
        )

    @staticmethod
    def _pair_sequences(
        prompt: Prompt, stereo_text: str | None, antistereo_text: str | None
    ) -> tuple[str, str]:
        """The two sequences to score: the caller's if given, otherwise the prompt's own."""
        given = (stereo_text is not None, antistereo_text is not None)
        if any(given) and not all(given):
            raise ValueError(
                f"score_pair({prompt.prompt_id!r}) was given one sequence but not the other. A "
                "pair whose halves come from different places is not a pair: pass both, or pass "
                "neither and let them be read off the prompt."
            )
        if stereo_text is not None and antistereo_text is not None:
            return stereo_text, antistereo_text
        if prompt.rendered_text_stereo is None or prompt.rendered_text_antistereo is None:
            raise ValueError(
                f"prompt {prompt.prompt_id!r} is pair_pll but carries no sequences. Since open "
                "decision #7 was resolved (20 Sep 2026) a paired prompt carries both on the "
                "record; this one does not, so it was built outside ExtractiveRenderer."
            )
        return prompt.rendered_text_stereo, prompt.rendered_text_antistereo

    def score_pair(
        self,
        prompt: Prompt,
        *,
        stereo_text: str | None = None,
        antistereo_text: str | None = None,
        seed: int | None = None,
        sample_index: int = 0,
        prefix: str | None = None,
    ) -> Response:
        """Score a sentence pair into **one** :class:`~margins.schemas.Response`.

        This is open decision #6, option (a), as built: a pair is one measurement, so it belongs on
        one record. ``logprob_stereo - logprob_antistereo`` is positive when the model prefers the
        stereotyping sequence -- higher = more harm, matching the standardization rule. Reversing
        the two arguments silently flips the sign of every ``pair_pll`` effect.

        **Open decision #7 was resolved on 20 Sep 2026**, so the two sequences now ride on the
        prompt itself as ``rendered_text_stereo`` / ``rendered_text_antistereo`` and are read from
        there by default. The explicit arguments survive for tests and for a caller holding
        sequences from somewhere else; passing one without the other raises, because a pair whose
        halves come from different places is not a pair.

        ``sample_index`` is a parameter rather than a hard-coded 0 because the resumable harness
        computes every ``response_id`` up front from ``(prompt, seed, sample_index)`` and then
        checks what the backend produced against that plan. A backend that always stamped 0 would
        fail that check for every seed after the first -- silently, as an id mismatch rather than
        as a wrong number.
        """
        self._check_prompt(prompt, expect=TaskType.PAIR_PLL)
        stereo_text, antistereo_text = self._pair_sequences(prompt, stereo_text, antistereo_text)
        effective_seed = self._seed if seed is None else seed
        report = self.load_report

        stereo = self.sequence_logprob(stereo_text, prefix=prefix)
        antistereo = self.sequence_logprob(antistereo_text, prefix=prefix)

        return Response(
            response_id=make_response_id(
                prompt_id=prompt.prompt_id,
                model_name=self._spec.name,
                model_id=self._spec.model_id,
                revision=self._spec.revision,
                seed=effective_seed,
                sample_index=sample_index,
                decoding_hash=self._decoding.decoding_hash,
            ),
            prompt_id=prompt.prompt_id,
            model_id=self._spec.model_id,
            model_revision=self._spec.revision,
            seed=effective_seed,
            temperature=self._decoding.temperature,
            decoding_key=self._decoding.key,
            decoding_hash=self._decoding.decoding_hash,
            quant=report.quant_actual,
            # A pair_pll measurement is a likelihood comparison, not a generation: there is no
            # generated text, and inventing one would misrepresent what was measured.
            text="",
            token_logprobs=None,
            logprob_stereo=stereo.total,
            logprob_antistereo=antistereo.total,
        )

    # ---------------------------------------------------------------- the Protocol

    def score_pairs(
        self, prompts: Sequence[Prompt], *, seed: int | None = None, sample_index: int = 0
    ) -> list[Response]:
        """Score paired prompts, one :class:`Response` per prompt.

        Deliberately not batched. ``sequence_logprob`` is one forward pass per sequence, and
        batching would require padding -- which changes reduction order and therefore the
        log-probabilities, for no gain on inputs this short. The determinism note already scopes
        bit-identity to a fixed batch size; here there is nothing to fix.
        """
        return [self.score_pair(prompt, seed=seed, sample_index=sample_index) for prompt in prompts]

    def run(self, prompts: Sequence[Prompt]) -> list[Response]:
        """Run prompts of either paradigm (the ``InferenceBackend`` Protocol).

        **Open decision #7 was resolved on 20 Sep 2026**, and this method is where that shows.
        ``pair_pll`` used to be refused here, because ``Prompt.rendered_text`` is one string and
        cannot carry two sequences; a paired prompt now carries both on the record as
        ``rendered_text_stereo`` / ``rendered_text_antistereo``, mirroring decision #6's one
        ``Response`` carrying both log-probabilities. So paired prompts route to
        :meth:`score_pairs` and generative ones to :meth:`generate`.

        A **mixed** sequence is refused. The two paradigms need different decoding profiles -- the
        extractive one scores raw sequences under ``pll_v1``, the generative one wraps prompts in a
        chat template under ``greedy_v1``/``sampled_v1`` -- and this backend holds exactly one
        profile, so a mixed batch could only be run by ignoring that mismatch for half of it.
        """
        paired = [p for p in prompts if p.task_type is TaskType.PAIR_PLL]
        if paired and len(paired) != len(prompts):
            raise ValueError(
                f"run() was given {len(paired)} pair_pll prompt(s) and "
                f"{len(prompts) - len(paired)} of other task types. The two paradigms are scored "
                "under different decoding profiles and this backend holds one, so a mixed call "
                "could only proceed by ignoring the mismatch for half of them. Split the call."
            )
        if paired:
            return self.score_pairs(prompts)
        return self.generate(prompts)
