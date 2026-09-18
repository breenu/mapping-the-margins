"""The scorer seam: a narrow text Protocol, and the adapter that satisfies ``Scorer`` (P6).

Two layers, on purpose.

**:class:`TextScorer` is the narrow one** -- ``score_texts(texts) -> list[float]`` and nothing else.
Every mechanism (a pinned transformer, a wordlist, a valence lexicon) fits behind it, which means
the whole of P6's judgement-carrying logic -- standardization, triangulation, the drift control,
calibration -- is testable with a fake that is three lines long and imports nothing. This is the
same move ``tests/test_harness_resume.py`` makes with ``GenerativeBackend``: resumability is a
property of the harness, not of any checkpoint, and here *scoring arithmetic is a property of the
pipeline, not of any classifier*. The core test suite runs on the torch-free environment because of
this split, and so does the core CI job on Windows.

**:class:`ResponseScorer` is the adapter** that satisfies the locked
:class:`~margins.interfaces.Scorer` Protocol. It exists because ``score(responses)`` cannot populate
an ``ItemScore`` on its own: ``cell_id`` and ``benchmark`` live on :class:`~margins.schemas.Prompt`,
not on :class:`~margins.schemas.Response`. The context is injected at construction, exactly as the
P1 walking skeleton's ``FakeScorer`` does.

**One ItemScore per Response per scorer -- never an average.** The confirmatory design generates
each prompt many times at distinct seeds (open decision #8). A scorer that averaged across those
would destroy the within-cell variance E4's sensitivity analysis measures and the cluster bootstrap
resamples. Aggregation across seeds is the METRIC layer's job (P7). This module's contract is
one row in, one row out, per scorer, and a test pins it.

TODO(P2): this is where the paired path will diverge. A ``pair_pll`` response carries no text worth
scoring -- the measurement is already on the record as
:attr:`~margins.schemas.Response.stereo_preference` (``logprob_stereo - logprob_antistereo``,
positive when the model prefers the stereotyping sequence, per open decision 6). A ``PairedScorer``
will read that property instead of calling :meth:`TextScorer.score_texts`, and will standardize it
through the same function with ``sign=+1``. **Nothing is implemented here**: open decision 7 --
how a paired prompt is *represented* -- is still open, and the extractive vertical rides with
P2/P4b.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from margins.schemas import Condition, ItemScore, Prompt, Response
from margins.scoring.config import ScorerSpec
from margins.scoring.ids import make_score_id

__all__ = ["ResponseScorer", "TextScorer", "scorer_from_spec"]


@runtime_checkable
class TextScorer(Protocol):
    """Turns text into a raw harm number. The whole of a mechanism's surface.

    Deliberately says nothing about batching, devices, or models: an implementation may be a
    transformer on a GPU or a set-membership count, and every caller in this package treats them
    identically.
    """

    @property
    def scorer_id(self) -> str:
        """Stable id, matching the roster entry this scorer was built from."""

    def score_texts(self, texts: Sequence[str]) -> list[float]:
        """One raw score per text, in input order.

        **Order is part of the contract**, because the caller zips the result back against the
        responses that produced it. An implementation that reorders for batching must restore the
        input order before returning.
        """


class ResponseScorer:
    """Adapts a :class:`TextScorer` to the locked :class:`~margins.interfaces.Scorer` Protocol.

    ``standardized`` is left equal to ``raw`` here and is filled in by
    :func:`margins.scoring.standardize.standardize_within_benchmark` afterwards. That ordering is
    forced rather than stylistic: standardization is *within benchmark across the whole set*, so it
    cannot be computed one response at a time. Leaving the field to be overwritten in a second pass
    keeps the sign contract in exactly one place instead of in every scorer.
    """

    def __init__(
        self,
        spec: ScorerSpec,
        text_scorer: TextScorer,
        *,
        prompts_by_id: dict[str, Prompt],
        benchmark: str,
        scoring_hash: str,
        condition: Condition = "model",
    ) -> None:
        if text_scorer.scorer_id != spec.scorer_id:
            raise ValueError(
                f"text scorer id {text_scorer.scorer_id!r} does not match roster entry "
                f"{spec.scorer_id!r}. The id is recorded on every row and is what the agreement "
                "analysis groups by; a mismatch would attribute scores to the wrong instrument."
            )
        self._spec = spec
        self._text_scorer = text_scorer
        self._prompts = prompts_by_id
        self._benchmark = benchmark
        self._scoring_hash = scoring_hash
        self._condition: Condition = condition

    @property
    def scorer_id(self) -> str:
        return self._spec.scorer_id

    @property
    def harm_type(self) -> str:
        return self._spec.harm_type

    @property
    def spec(self) -> ScorerSpec:
        return self._spec

    @property
    def condition(self) -> Condition:
        return self._condition

    def for_condition(self, condition: Condition) -> ResponseScorer:
        """The same scorer, stamping a different ``condition``.

        The drift control scores neutral text through the *same instrument* as model output. Sharing
        the underlying :class:`TextScorer` instance rather than rebuilding one is what guarantees
        that -- a second instance could load a different revision or a differently-seeded model and
        the control would no longer be measuring the scorer that produced the model scores.
        """
        return ResponseScorer(
            self._spec,
            self._text_scorer,
            prompts_by_id=self._prompts,
            benchmark=self._benchmark,
            scoring_hash=self._scoring_hash,
            condition=condition,
        )

    def score(self, responses: Sequence[Response]) -> list[ItemScore]:
        """One :class:`~margins.schemas.ItemScore` per response. No averaging, ever."""
        if not responses:
            return []

        missing = sorted({r.prompt_id for r in responses if r.prompt_id not in self._prompts})
        if missing:
            raise KeyError(
                f"{len(missing)} response(s) reference prompts this scorer was not given, e.g. "
                f"{missing[:3]}. ItemScore.cell_id and .benchmark come from the Prompt, so scoring "
                "without it would emit rows the metric layer cannot group."
            )

        paired = [r for r in responses if r.is_paired]
        if paired:
            raise ValueError(
                f"{len(paired)} response(s) carry pair_pll log-probabilities, which this scorer "
                "does not measure. The paired path rides with P2/P4b -- see the TODO(P2) in this "
                "module. Refusing rather than scoring the text of a paired record, which would "
                "silently produce a number that means nothing."
            )

        raw_scores = self._text_scorer.score_texts([r.text for r in responses])
        if len(raw_scores) != len(responses):
            raise ValueError(
                f"scorer {self.scorer_id!r} returned {len(raw_scores)} scores for "
                f"{len(responses)} responses. score_texts must return one score per text, in order."
            )

        scores: list[ItemScore] = []
        for response, raw in zip(responses, raw_scores, strict=True):
            prompt = self._prompts[response.prompt_id]
            scores.append(
                ItemScore(
                    score_id=make_score_id(
                        response_id=response.response_id,
                        scorer_id=self._spec.scorer_id,
                        harm_type=self._spec.harm_type,
                        condition=self._condition,
                        pin_digest=self._spec.pin_digest,
                        scoring_hash=self._scoring_hash,
                    ),
                    response_id=response.response_id,
                    scorer_id=self._spec.scorer_id,
                    harm_type=self._spec.harm_type,
                    raw=float(raw),
                    # Overwritten by standardize_within_benchmark, which needs the whole set.
                    standardized=float(raw),
                    prompt_id=prompt.prompt_id,
                    cell_id=prompt.cell_id,
                    benchmark=self._benchmark,
                    model_id=response.model_id,
                    seed=response.seed,
                    temperature=response.temperature,
                    condition=self._condition,
                )
            )
        return scores


def scorer_from_spec(
    spec: ScorerSpec,
    *,
    prompts_by_id: dict[str, Prompt],
    benchmark: str,
    scoring_hash: str,
    data_dir: object = None,
) -> ResponseScorer:
    """Build the :class:`ResponseScorer` a roster entry describes.

    The mechanism-specific imports happen here and nowhere else, so a caller that only needs lexicon
    scorers never touches the transformers import path.
    """
    from pathlib import Path

    if spec.mechanism == "lexicon":
        from margins.scoring.lexicon import build_lexicon_scorer

        root = Path(str(data_dir)) if data_dir is not None else Path("data")
        text_scorer: TextScorer = build_lexicon_scorer(spec, data_dir=root)
    else:
        from margins.scoring.hf_classifier import HFClassifierScorer

        text_scorer = HFClassifierScorer(spec)

    return ResponseScorer(
        spec,
        text_scorer,
        prompts_by_id=prompts_by_id,
        benchmark=benchmark,
        scoring_hash=scoring_hash,
    )
