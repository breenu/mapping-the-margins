"""The pinned neural-classifier leg of each harm type (P6).

**Why `transformers` directly rather than the `detoxify` package.** `detoxify` is declared
`sys_platform == 'linux'` in the `inference` extra, so a `detoxify`-backed scorer could not be
exercised on the Windows dev box or in the core CI job -- it would only ever run in the nightly
Linux job, which is the wrong place for the instrument that produces the primary estimand's harm
signal. Loading the same underlying checkpoint through `transformers` gives an identical
classifier, runs on both locked environments, and brings it under this project's own pinning
discipline: a resolved commit sha in `configs/scoring/`, verified by `scripts/pin_scorers.py`,
rather than whatever revision a wrapper package happens to download.

**Determinism.** Classification is a forward pass with no sampling, so there is no seed to set here;
the same input on the same revision gives the same logits up to floating-point reduction order. As
with P5, that means CPU at a fixed thread count is reproducible bit-for-bit and GPU is
metric-level -- :func:`margins.repro.residual_nondeterminism_note` already states the scope, and the
manifest records the environment that would have to match.

`torch` and `transformers` are imported **lazily**, inside the methods that need them, exactly as
:mod:`margins.inference.backend` does. That is what keeps `mypy src`, `ruff`, and the core test
suite runnable on an environment with no inference extra installed.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from typing import Any, Final

from margins.scoring.config import ScorerSpec

__all__ = ["DEFAULT_BATCH_SIZE", "HFClassifierScorer", "transformers_available"]

#: Scoring batch size. Unlike generation (where padding changes reduction order and therefore the
#: determinism claim, P5), a classifier's per-sequence output is unaffected by its batch-mates up to
#: float tolerance -- so this is a throughput knob, and it is recorded rather than pinned.
DEFAULT_BATCH_SIZE: Final[int] = 16


def transformers_available() -> bool:
    """True when the inference extra is installed. A presence probe, not an import."""
    return importlib.util.find_spec("transformers") is not None


class HFClassifierScorer:
    """A pinned Hugging Face sequence classifier, reading one named label.

    The model is loaded on first use, not at construction: building the roster to inspect it (which
    `scripts/pin_scorers.py --verify` and several tests do) must not download hundreds of megabytes.
    """

    def __init__(self, spec: ScorerSpec, *, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        if spec.model_pin is None:  # pragma: no cover - guarded by ScorerSpec validation
            raise ValueError(f"scorer {spec.scorer_id!r} has no model_pin")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self._spec = spec
        self._pin = spec.model_pin
        self._batch_size = batch_size
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._label_index: int | None = None

    @property
    def scorer_id(self) -> str:
        return self._spec.scorer_id

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @staticmethod
    def _torch() -> Any:
        import torch

        return torch

    def _load(self) -> tuple[Any, Any, int]:
        """Load the pinned checkpoint and resolve the configured label to a column index."""
        if (
            self._model is not None
            and self._tokenizer is not None
            and self._label_index is not None
        ):
            return self._model, self._tokenizer, self._label_index

        if not transformers_available():
            raise RuntimeError(
                f"scorer {self.scorer_id!r} needs the inference extra: "
                "`uv sync --extra dev --extra inference`."
            )

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self._pin.repo, revision=self._pin.revision)
        model = AutoModelForSequenceClassification.from_pretrained(
            self._pin.repo, revision=self._pin.revision
        )
        model.eval()

        # Resolve the label by NAME, never by position. Label order is a property of the checkpoint
        # and can differ between revisions; reading column 0 and calling it "toxic" is how a re-pin
        # silently starts measuring something else.
        id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
        matches = [
            index for index, name in id2label.items() if name.lower() == self._pin.label.lower()
        ]
        if not matches:
            raise ValueError(
                f"scorer {self.scorer_id!r} asks for label {self._pin.label!r}, which "
                f"{self._pin.repo}@{self._pin.revision[:12]} does not expose. Available: "
                f"{sorted(id2label.values())}. Refusing to guess -- a substituted label would "
                "measure a different construct under the same scorer_id."
            )

        self._model, self._tokenizer, self._label_index = model, tokenizer, matches[0]
        return model, tokenizer, matches[0]

    def score_texts(self, texts: Sequence[str]) -> list[float]:
        """Probability of the configured label, one per text, in input order."""
        if not texts:
            return []

        torch = self._torch()
        model, tokenizer, label_index = self._load()

        scores: list[float] = []
        with torch.no_grad():
            for start in range(0, len(texts), self._batch_size):
                batch = list(texts[start : start + self._batch_size])
                encoded = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self._pin.max_input_tokens,
                )
                logits = model(**encoded).logits
                # Multi-label heads (toxic-bert) are independent sigmoids; single-label heads are a
                # softmax over classes. Choosing by head shape rather than by checkpoint name means
                # a differently-shaped classifier does not silently get the wrong transform.
                if logits.shape[-1] == 1 or getattr(model.config, "problem_type", None) == (
                    "multi_label_classification"
                ):
                    probabilities = torch.sigmoid(logits)
                else:
                    probabilities = torch.softmax(logits, dim=-1)
                scores.extend(probabilities[:, label_index].tolist())

        return [float(value) for value in scores]
