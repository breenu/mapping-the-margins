"""Gate P5's determinism and log-prob clauses, against a real checkpoint.

Skipped wholesale when the ``inference`` extra is absent, which is the normal state of the core
environment and of the core CI job. They run locally against the **dev model** (the ~1B pin) and in
the nightly Linux job against the tiny random checkpoint, selected by ``MARGINS_TEST_MODEL``:

    uv run pytest -m inference                       # the dev model (one 2.5 GB download)
    MARGINS_TEST_MODEL=tiny_random_llama uv run pytest -m inference

**What is and is not claimed.** Same seed, greedy, fixed thread count, fixed batch size gives
identical token sequences. That is the whole claim -- see
:func:`margins.repro.residual_nondeterminism_note`, which these tests exist to keep honest.

The hand-computed log-prob check deliberately uses a **stub with fixed logits** rather than a real
checkpoint: gate P5 asks for a value matched against a hand computation, and no hand computation is
possible against real weights. Uniform-ish logits over a two-token vocabulary give an exact expected
answer, so the test checks the arithmetic (log-softmax, the off-by-one between predictor and target,
the gather) rather than merely checking the code against itself.
"""

from __future__ import annotations

import importlib.util
import math
import os
from types import SimpleNamespace
from typing import Any

import pytest

from margins.inference.decoding import DecodingConfig, decoding_config_from_mapping
from margins.inference.models import DEV_MODEL, ModelSpec, load_model_config
from margins.schemas import Prompt, TaskType

pytestmark = [
    pytest.mark.inference,
    pytest.mark.skipif(
        importlib.util.find_spec("torch") is None,
        reason="needs the inference extra: uv sync --extra dev --extra inference",
    ),
]

#: Which pinned config the torch tests target. Defaults to the dev model, per the P5 plan; CI sets
#: it to the tiny random checkpoint, which is ungated and needs no HF token.
TEST_MODEL = os.environ.get("MARGINS_TEST_MODEL", DEV_MODEL)

SEED = 20260908
OTHER_SEED = 20260909


@pytest.fixture(scope="module")
def spec() -> ModelSpec:
    return load_model_config(TEST_MODEL)


@pytest.fixture(scope="module")
def has_chat_template(spec: ModelSpec) -> bool:
    """The tiny CI checkpoint has no chat template; the pinned instruct models do."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision)
    return getattr(tokenizer, "chat_template", None) is not None


def _decoding(has_chat: bool, **overrides: Any) -> DecodingConfig:
    data: dict[str, Any] = {
        "key": "greedy_v1",
        "description": "test profile",
        "prompt_format": "chat" if has_chat else "raw",
        "pinned_chat_date": "08 Sep 2026",
        "max_input_tokens": 256,
        "do_sample": False,
        "temperature": 0.0,
        # Small on purpose: these tests run on CPU and prove determinism, not fluency.
        "max_new_tokens": 8,
    }
    data.update(overrides)
    return decoding_config_from_mapping(data)


def _prompt(index: int = 0, task_type: TaskType = TaskType.OPEN_GEN) -> Prompt:
    """One prompt of either paradigm.

    Since open decision #7 was resolved (20 Sep 2026) a ``pair_pll`` prompt carries BOTH sequences
    and an empty ``rendered_text``; the schema refuses any other shape, which is why this helper
    branches rather than passing ``task_type`` straight through.
    """
    if task_type is TaskType.PAIR_PLL:
        return Prompt(
            prompt_id=f"crows_pll::axis=value-{index}::crows_pairs-{index:04d}::{index:012d}",
            item_id=f"crows_pairs::{index:04d}",
            cell_id=f"axis=value-{index}",
            rendered_text="",
            task_type=task_type,
            template_version="crows_pll_v1@2026-09-20",
            decoding_key="greedy_v1",
            rendered_text_stereo="abab",
            rendered_text_antistereo="aba",
        )
    return Prompt(
        prompt_id=f"describe::axis=value-{index}::person::{index:012d}",
        item_id="open_gen_templates/describe",
        cell_id=f"axis=value-{index}",
        rendered_text=f"Describe a person who is number {index}.",
        task_type=task_type,
        template_version="open_gen_v1.1@2026-09-08",
        decoding_key="greedy_v1",
    )


@pytest.fixture(scope="module")
def backend(spec: ModelSpec, has_chat_template: bool) -> Any:
    from margins.inference.backend import HFBackend

    instance = HFBackend(spec, _decoding(has_chat_template), seed=SEED, quant="nf4")
    # Force the load once for the whole module rather than per test.
    assert instance.load_report is not None
    return instance


# --------------------------------------------------------------------------------------
# Determinism (gate P5, clause 1)
# --------------------------------------------------------------------------------------


def test_same_seed_greedy_produces_identical_output(backend: Any) -> None:
    prompts = [_prompt(index) for index in range(3)]
    first = backend.generate(prompts, seed=SEED, batch_size=1)
    second = backend.generate(prompts, seed=SEED, batch_size=1)

    assert [r.text for r in first] == [r.text for r in second]
    assert [r.response_id for r in first] == [r.response_id for r in second]


def test_different_seeds_under_sampling_diverge(spec: ModelSpec, has_chat_template: bool) -> None:
    """The other half of the claim: seeds must actually do something once sampling is on.

    Compared across several prompts rather than one, because any single short generation can
    coincide by chance -- the assertion is that the seed changes *something*, not everything.
    """
    from margins.inference.backend import HFBackend

    sampled = _decoding(
        has_chat_template, key="sampled_v1", do_sample=True, temperature=1.0, top_p=0.95
    )
    backend = HFBackend(spec, sampled, seed=SEED, quant="nf4")
    prompts = [
        _prompt(index).model_copy(update={"decoding_key": "sampled_v1"}) for index in range(4)
    ]

    first = [r.text for r in backend.generate(prompts, seed=SEED)]
    second = [r.text for r in backend.generate(prompts, seed=OTHER_SEED)]
    assert first != second, "changing the seed changed nothing; sampling is not actually on"


def test_a_prompt_stamped_for_another_profile_is_refused(backend: Any) -> None:
    """Otherwise the Response's decoding_hash would not describe the prompt that produced it."""
    stray = _prompt().model_copy(update={"decoding_key": "sampled_v1"})
    with pytest.raises(ValueError, match="is stamped decoding_key"):
        backend.generate([stray])


# --------------------------------------------------------------------------------------
# Compute dtype must come from the device, not from a constant
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("capability", "expected", "gpu"),
    [
        ((7, 5), "float16", "Tesla T4 (Turing) -- the project's reference hardware, NO bf16"),
        ((8, 0), "bfloat16", "A100 (Ampere)"),
        ((8, 6), "bfloat16", "RTX 3090 (Ampere)"),
        ((9, 0), "bfloat16", "H100 (Hopper)"),
        ((6, 1), "float16", "GTX 1080 (Pascal)"),
    ],
)
def test_compute_dtype_is_chosen_from_the_device(
    capability: tuple[int, int], expected: str, gpu: str
) -> None:
    """bfloat16 requires compute capability 8.0, and a free-tier Kaggle T4 is 7.5.

    Hardcoding bf16 is silently wrong there -- nothing in the bitsandbytes or transformers API
    rejects it -- and the T4 is exactly where the confirmatory sweep runs. Parametrised over real
    cards because the CI box and the dev box have no GPU at all, so this is the only way the rule
    gets checked before it reaches Kaggle.
    """
    import torch

    from margins.inference.backend import cuda_compute_dtype

    fake = SimpleNamespace(
        cuda=SimpleNamespace(get_device_capability=lambda: capability),
        bfloat16=torch.bfloat16,
        float16=torch.float16,
    )
    chosen = cuda_compute_dtype(fake)
    assert str(chosen).removeprefix("torch.") == expected, f"wrong dtype for {gpu}"


def test_the_quantization_compute_dtype_matches_the_model_dtype(spec: ModelSpec) -> None:
    """A compute dtype that disagrees with the weights' dtype is a silent correctness bug."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "src/margins/inference/backend.py").read_text(
        encoding="utf-8"
    )
    # Code lines only: the docstring quotes the wrong form (inside ``...``) to explain the bug.
    assignments = [
        line.strip()
        for line in source.splitlines()
        if "bnb_4bit_compute_dtype=" in line and "``" not in line
    ]
    assert assignments == ["bnb_4bit_compute_dtype=dtype,"], (
        "the bitsandbytes compute dtype must be the same `dtype` chosen for the device, never a "
        f"hardcoded bfloat16. Found: {assignments}"
    )


# --------------------------------------------------------------------------------------
# The Response is self-describing
# --------------------------------------------------------------------------------------


def test_response_carries_the_resolved_revision_and_the_actual_quantization(
    backend: Any, spec: ModelSpec
) -> None:
    """`quant` must be what happened, never the intended constant (framing constraint 4)."""
    response = backend.generate([_prompt()], seed=SEED)[0]
    report = backend.load_report

    assert response.model_revision == spec.revision
    assert len(response.model_revision) == 40
    assert response.decoding_hash == backend.decoding.decoding_hash
    assert response.decoding_key == backend.decoding.key
    assert response.seed == SEED
    assert response.temperature == 0.0
    assert response.quant == report.quant_actual
    if report.quant_actual != report.quant_requested:
        assert report.quant_fallback_reason, "a fallback must say why"


def test_the_checkpoints_bundled_sampling_parameters_are_cleared(backend: Any) -> None:
    """The decoding config must be the SOLE authority on how tokens are chosen.

    Llama-3.2-1B-Instruct ships ``temperature=0.6, top_p=0.9`` in its own generation_config.
    Greedy decoding ignores them, but a future sampled profile that did not name a ``top_p`` would
    inherit 0.9 silently -- and it would appear in no ``decoding_hash``, because it never came from
    our config. An undeclared parameter must not merely go unrecorded; it must not be used.
    """
    config = backend._model.generation_config
    for parameter in ("temperature", "top_p", "top_k"):
        assert getattr(config, parameter, None) is None, (
            f"the checkpoint's bundled {parameter} survived the load and could apply unrecorded"
        )


def test_load_report_records_the_environment_the_claim_depends_on(backend: Any) -> None:
    report = backend.load_report
    assert report.torch_version
    assert report.device in {"cpu", "cuda"}
    assert report.dtype in {"float32", "bfloat16"}
    assert report.pad_token_source in {"tokenizer", "eos"}


def test_generated_responses_leave_the_paired_logprobs_null(backend: Any) -> None:
    """open_gen is not a paired measurement; half a pair would be a lie about what was measured."""
    response = backend.generate([_prompt()], seed=SEED)[0]
    assert response.logprob_stereo is None
    assert response.logprob_antistereo is None
    assert response.is_paired is False


def test_token_logprobs_are_off_by_default_and_available_on_request(backend: Any) -> None:
    assert backend.generate([_prompt()], seed=SEED)[0].token_logprobs is None
    with_logprobs = backend.generate([_prompt()], seed=SEED, token_logprobs=True)[0]
    assert with_logprobs.token_logprobs is not None
    assert all(value <= 0.0 for value in with_logprobs.token_logprobs)


# --------------------------------------------------------------------------------------
# sequence_logprob (gate P5, clause 2)
# --------------------------------------------------------------------------------------


class _StubEncoding(dict):  # type: ignore[type-arg]
    def to(self, device: object) -> _StubEncoding:
        return self


class _StubTokenizer:
    """Maps single characters to token ids. Two-token vocabulary: 'a' -> 0, 'b' -> 1."""

    pad_token_id = 0
    eos_token_id = 1
    chat_template = None

    def __call__(self, text: str, **kwargs: Any) -> _StubEncoding:
        import torch

        ids = torch.tensor([[0 if char == "a" else 1 for char in text]])
        return _StubEncoding(input_ids=ids)


class _StubModel:
    """Emits the same fixed logits at every position, so the answer is computable by hand.

    Logits ``[0, ln 3]`` over a two-token vocabulary give softmax ``[1/4, 3/4]``, i.e.
    ``log p(0) = ln 0.25`` and ``log p(1) = ln 0.75`` at every position.
    """

    device = "cpu"

    def eval(self) -> _StubModel:
        return self

    def __call__(self, input_ids: Any = None, **kwargs: Any) -> Any:
        import torch

        length = int(input_ids.shape[1])
        row = torch.tensor([0.0, math.log(3.0)])
        return SimpleNamespace(logits=row.repeat(1, length, 1))


def test_sequence_logprob_matches_a_hand_computed_value(spec: ModelSpec) -> None:
    """Gate P5: 'logprob path matches a hand-computed toy value within the documented tolerance'.

    Sequence ``a b a`` is token ids ``[0, 1, 0]``. The first token has no predictor and is excluded,
    so the scored tokens are ``1`` then ``0``:

        ln 0.75 + ln 0.25 = -0.287682 + -1.386294 = -1.673976
    """
    from margins.inference.backend import HFBackend

    backend = HFBackend(
        spec,
        _decoding(has_chat=False),
        seed=SEED,
        quant="none",
        model=_StubModel(),
        tokenizer=_StubTokenizer(),
    )
    result = backend.sequence_logprob("aba")

    expected_tokens = (math.log(0.75), math.log(0.25))
    assert result.n_tokens == 2, "the first token has no predictor and must not be scored"
    assert result.token_logprobs == pytest.approx(expected_tokens, abs=1e-6)
    assert result.total == pytest.approx(math.log(0.75) + math.log(0.25), abs=1e-6)


def test_sequence_logprob_is_internally_consistent_on_the_real_checkpoint(backend: Any) -> None:
    """Against real weights the only checkable facts are structural -- so check exactly those."""
    result = backend.sequence_logprob("The nurse examined the patient carefully.")

    assert result.n_tokens == len(result.token_logprobs) > 0
    assert all(value <= 0.0 for value in result.token_logprobs)
    assert result.total == pytest.approx(sum(result.token_logprobs), abs=1e-6)
    repeated = backend.sequence_logprob("The nurse examined the patient carefully.")
    assert repeated.total == pytest.approx(result.total, abs=1e-6)


def test_a_more_likely_sentence_scores_higher(backend: Any, spec: ModelSpec) -> None:
    """Sanity on the sign convention, skipped for the random CI checkpoint which has no ordering."""
    if spec.role == "ci":
        pytest.skip("the tiny random checkpoint has no meaningful likelihood ordering")
    fluent = backend.sequence_logprob("The cat sat on the mat.")
    scrambled = backend.sequence_logprob("Mat the on sat cat the.")
    assert fluent.total > scrambled.total


def test_an_empty_or_single_token_sequence_is_refused(spec: ModelSpec) -> None:
    from margins.inference.backend import HFBackend

    backend = HFBackend(
        spec,
        _decoding(has_chat=False),
        seed=SEED,
        quant="none",
        model=_StubModel(),
        tokenizer=_StubTokenizer(),
    )
    with pytest.raises(ValueError, match="empty sequence"):
        backend.sequence_logprob("")
    with pytest.raises(ValueError, match="fewer than 2 tokens"):
        backend.sequence_logprob("a")


# --------------------------------------------------------------------------------------
# The pair_pll shape (decision #6) and the representation that closed with it (#7)
# --------------------------------------------------------------------------------------


def test_a_pair_yields_one_response_carrying_both_logprobs(spec: ModelSpec) -> None:
    """Decision #6 option (a): a pair is ONE measurement, so it belongs on ONE record."""
    from margins.inference.backend import HFBackend

    backend = HFBackend(
        spec,
        _decoding(has_chat=False),
        seed=SEED,
        quant="none",
        model=_StubModel(),
        tokenizer=_StubTokenizer(),
    )
    prompt = _prompt(task_type=TaskType.PAIR_PLL)
    response = backend.score_pair(prompt, stereo_text="abab", antistereo_text="aba")

    assert response.is_paired
    assert response.logprob_stereo is not None
    assert response.logprob_antistereo is not None
    # Sign convention: positive means the model prefers the stereotyping sequence (= more harm).
    assert response.stereo_preference == pytest.approx(
        response.logprob_stereo - response.logprob_antistereo
    )
    assert response.text == "", "a likelihood comparison generates no text"


def test_score_pair_reads_the_two_sequences_off_the_prompt(spec: ModelSpec) -> None:
    """Decision #7 as built: the prompt carries the pair, so the caller need not.

    The explicit-argument form is the same measurement, which is what makes the record's
    ``rendered_text_stereo``/``antistereo`` the single source of truth rather than a second one.
    """
    from margins.inference.backend import HFBackend

    backend = HFBackend(
        spec,
        _decoding(has_chat=False),
        seed=SEED,
        quant="none",
        model=_StubModel(),
        tokenizer=_StubTokenizer(),
    )
    prompt = _prompt(task_type=TaskType.PAIR_PLL)

    from_prompt = backend.score_pair(prompt)
    explicit = backend.score_pair(prompt, stereo_text="abab", antistereo_text="aba")

    assert from_prompt.logprob_stereo == explicit.logprob_stereo
    assert from_prompt.logprob_antistereo == explicit.logprob_antistereo


def test_score_pair_refuses_half_a_pair(spec: ModelSpec) -> None:
    """A pair whose halves come from different places is not a pair."""
    from margins.inference.backend import HFBackend

    backend = HFBackend(
        spec,
        _decoding(has_chat=False),
        seed=SEED,
        quant="none",
        model=_StubModel(),
        tokenizer=_StubTokenizer(),
    )
    with pytest.raises(ValueError, match="one sequence but not the other"):
        backend.score_pair(_prompt(task_type=TaskType.PAIR_PLL), stereo_text="abab")


def test_run_routes_paired_prompts_to_the_logprob_path(backend: Any) -> None:
    """Open decision #7 is CLOSED: run() no longer refuses a paired prompt, it scores it."""
    responses = backend.run([_prompt(task_type=TaskType.PAIR_PLL)])

    assert len(responses) == 1
    assert responses[0].is_paired
    assert responses[0].text == "", "a likelihood comparison generates no text"


def test_run_refuses_a_mixed_batch(backend: Any) -> None:
    """The two paradigms run under different decoding profiles; this backend holds one."""
    with pytest.raises(ValueError, match="Split the call"):
        backend.run([_prompt(), _prompt(task_type=TaskType.PAIR_PLL)])


def test_run_generates_for_open_gen(backend: Any) -> None:
    responses = backend.run([_prompt(), _prompt(1)])
    assert len(responses) == 2
    assert all(response.text is not None for response in responses)


# --------------------------------------------------------------------------------------
# The chat wrapper
# --------------------------------------------------------------------------------------


def test_chat_wrapper_pins_the_date(backend: Any, has_chat_template: bool) -> None:
    """A template that reads the clock would re-render every prompt differently tomorrow."""
    if not has_chat_template:
        pytest.skip("checkpoint declares no chat template")
    wrapped = backend.format_prompt("Describe a person.")

    assert "Describe a person." in wrapped
    assert wrapped != "Describe a person.", "the chat wrapper did nothing"
    assert backend.format_prompt("Describe a person.") == wrapped
    assert backend.load_report.chat_template_sha256 is not None


def test_raw_format_passes_the_prompt_through_untouched(spec: ModelSpec) -> None:
    from margins.inference.backend import HFBackend

    backend = HFBackend(
        spec,
        _decoding(has_chat=False),
        seed=SEED,
        quant="none",
        model=_StubModel(),
        tokenizer=_StubTokenizer(),
    )
    assert backend.format_prompt("Describe a person.") == "Describe a person."


def test_chat_format_against_a_template_less_checkpoint_is_refused(spec: ModelSpec) -> None:
    from margins.inference.backend import HFBackend

    backend = HFBackend(
        spec,
        _decoding(has_chat=True),
        seed=SEED,
        quant="none",
        model=_StubModel(),
        tokenizer=_StubTokenizer(),
    )
    with pytest.raises(ValueError, match="declares no chat template"):
        backend.format_prompt("Describe a person.")
