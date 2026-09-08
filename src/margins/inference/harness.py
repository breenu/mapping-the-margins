"""The resumable run harness (P5.T2).

**Why this is a gate item and not a convenience.** The confirmatory sweep runs on free-tier Kaggle:
12-hour sessions, ~30 GPU-hours a week (PROJECT_CONTEXT.md section 7). A session that dies at hour
eleven must not cost eleven hours of GPU time, and the top-listed risk in section 14 is exactly
"compute over free quota". A sweep that cannot resume is a sweep that cannot finish.

**The contract, stated as an assertion rather than a comment.**

* *No drop.* Every planned unit of work is either already on disk or is run. The harness verifies
  the planned id set is a subset of the file's ids before it returns.
* *No duplicate.* Ids are content-addressed and computed **before** the work is done
  (:func:`~margins.inference.backend.make_response_id`), so a restart can tell finished work from
  outstanding work without re-generating anything to find out.
* *No silent widening.* A run file containing responses outside the current plan is an error, not a
  merge. Two different plans writing into one file would produce a response set that no manifest
  describes.

**The failure mode this is actually built for is a kill mid-write.** A truncated final line is the
normal state of an interrupted JSONL file, so the reader repairs it: the file is truncated back to
its last complete record, and that record's work is re-run. A *complete* line that fails to
validate is different -- that is corruption, and it raises rather than being discarded, because
quietly dropping a malformed record would lose data while reporting success.

Output lands under ``data/responses/<model>/<run_id>.jsonl`` -- gitignored and regenerable, and
never committed (raw response dumps ship as a Zenodo artifact at release).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

from margins.inference.backend import make_response_id
from margins.inference.decoding import DecodingConfig
from margins.inference.models import ModelSpec
from margins.schemas import Prompt, Response

__all__ = [
    "DEFAULT_RESPONSES_DIR",
    "GenerativeBackend",
    "RunHarness",
    "RunResult",
    "RunTask",
    "read_responses",
    "response_path",
]


@runtime_checkable
class GenerativeBackend(Protocol):
    """What the harness needs from a backend -- deliberately narrower than ``HFBackend``.

    Typing against this rather than the concrete class is what lets the resumability gate be tested
    without a model: the kill/restart behaviour is a property of the harness, not of any checkpoint,
    and a test that had to load weights to exercise it would be testing the wrong thing. It is also
    the seam the documented llama.cpp/GGUF fallback backend (PROJECT_CONTEXT.md section 6) would
    slot into.
    """

    @property
    def spec(self) -> ModelSpec:
        """The pinned checkpoint. Names the output subdirectory and part of every response id."""

    @property
    def decoding(self) -> DecodingConfig:
        """The resolved decoding profile whose hash is stamped on every response."""

    def generate(
        self,
        prompts: Sequence[Prompt],
        *,
        seed: int | None = ...,
        sample_index: int = ...,
        batch_size: int = ...,
        token_logprobs: bool = ...,
    ) -> list[Response]:
        """Generate one response per prompt."""


#: Raw text goes to JSONL (PROJECT_CONTEXT.md section 6); ``data/`` is gitignored and regenerable.
DEFAULT_RESPONSES_DIR: Final[Path] = Path("data") / "responses"


@dataclass(frozen=True, slots=True)
class RunTask:
    """One unit of work: this prompt, at this seed, as this sample.

    ``response_id`` is computed up front, which is the whole basis of resumability.
    """

    prompt: Prompt
    seed: int
    sample_index: int
    response_id: str


@dataclass(frozen=True, slots=True)
class RunResult:
    """What a run did, including how much of it was already on disk."""

    path: Path
    run_id: str
    planned: int
    resumed: int
    generated: int
    batch_size: int
    seeds: tuple[int, ...]

    @property
    def complete(self) -> bool:
        return self.resumed + self.generated == self.planned


def response_path(out_dir: Path, model_name: str, run_id: str) -> Path:
    """``<out_dir>/<model_name>/<run_id>.jsonl``.

    Partitioned by model because the sweep runs the same prompt set through several checkpoints,
    and a single interleaved file would make "resume this model" a filter rather than a file.
    """
    return out_dir / model_name / f"{run_id}.jsonl"


def _render_line(response: Response) -> str:
    """One JSONL record, in the same canonical form P4a uses for prompts."""
    return (
        json.dumps(
            response.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    )


def read_responses(path: Path, *, repair: bool = False) -> list[Response]:
    """Read a response file, optionally repairing a truncated final record.

    Args:
        path: The JSONL file. A missing file reads as empty.
        repair: When True, a trailing partial line (the normal result of a kill mid-write) is
            removed from the file on disk and the record it belonged to is treated as never having
            been written. When False the partial line is ignored but the file is left alone.

    Raises:
        ValueError: If a *complete* line fails to parse or validate. That is corruption rather than
            interruption, and discarding it would lose data while reporting success.
    """
    if not path.is_file():
        return []

    raw = path.read_bytes()
    if not raw:
        return []

    segments = raw.split(b"\n")
    trailing = segments.pop()  # empty when the file ends with a newline
    responses: list[Response] = []
    for number, segment in enumerate(segments, start=1):
        if not segment.strip():
            continue
        try:
            responses.append(Response.model_validate_json(segment.decode("utf-8")))
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError(
                f"{path}: line {number} is complete but does not validate as a Response. This is "
                "corruption, not an interrupted write; refusing to discard it silently."
            ) from error

    if trailing.strip() and repair:
        # Truncate back to the end of the last complete record. Its work is simply re-run.
        keep = len(raw) - len(trailing)
        with path.open("r+b") as handle:
            handle.truncate(keep)
    return responses


class RunHarness:
    """Batches prompts through a backend and checkpoints responses as they complete."""

    def __init__(
        self,
        backend: GenerativeBackend,
        *,
        run_id: str,
        out_dir: Path = DEFAULT_RESPONSES_DIR,
        batch_size: int = 1,
        token_logprobs: bool = False,
        strict: bool = True,
    ) -> None:
        """
        Args:
            backend: The loaded backend. Its ``ModelSpec`` decides the output subdirectory.
            run_id: **Recorded, not generated per call.** It comes from the run manifest, so the
                responses on disk and the manifest describing them share one identifier. Resuming
                means passing the same run_id.
            out_dir: Root for response files.
            batch_size: Prompts per forward pass, and also the checkpoint granularity -- a kill
                loses at most one batch. The determinism claim is scoped to a fixed batch size.
            token_logprobs: Record per-token log-probabilities of generated text.
            strict: Refuse to write into a file that already holds responses outside this plan.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if not run_id:
            raise ValueError("run_id must be non-empty; it ties responses to their manifest")
        self._backend = backend
        self._run_id = run_id
        self._out_dir = out_dir
        self._batch_size = batch_size
        self._token_logprobs = token_logprobs
        self._strict = strict

    @property
    def path(self) -> Path:
        return response_path(self._out_dir, self._backend.spec.name, self._run_id)

    def plan(self, prompts: Sequence[Prompt], *, seeds: Sequence[int]) -> list[RunTask]:
        """Enumerate every unit of work, with its id, before any of it runs.

        One generation per ``(prompt, seed)``. Distinct samples come from distinct **seeds** and
        never from repeated calls at one seed: the backend re-seeds per batch, so a second call at
        the same seed returns the same text under a different id -- a duplicate wearing a fresh
        label. Requiring distinct seeds makes that unrepresentable.
        """
        if not seeds:
            raise ValueError("a run needs at least one seed")
        if len(set(seeds)) != len(seeds):
            raise ValueError(
                f"seeds must be distinct, got {list(seeds)}. Two generations at one seed are "
                "identical by construction (the backend re-seeds per batch), so a repeated seed "
                "would write the same text twice under two ids."
            )
        spec = self._backend.spec
        decoding_hash = self._backend.decoding.decoding_hash
        return [
            RunTask(
                prompt=prompt,
                seed=seed,
                sample_index=index,
                response_id=make_response_id(
                    prompt_id=prompt.prompt_id,
                    model_name=spec.name,
                    model_id=spec.model_id,
                    revision=spec.revision,
                    seed=seed,
                    sample_index=index,
                    decoding_hash=decoding_hash,
                ),
            )
            for index, seed in enumerate(seeds)
            for prompt in prompts
        ]

    def run(
        self,
        prompts: Sequence[Prompt],
        *,
        seeds: Sequence[int],
        progress: Callable[[int, int], None] | None = None,
    ) -> RunResult:
        """Run every outstanding task, appending to the run file as batches complete.

        Raises:
            ValueError: If the existing file holds duplicate ids, or (under ``strict``) responses
                outside this plan, or if the no-drop check fails at the end.
        """
        tasks = self.plan(prompts, seeds=seeds)
        planned_ids = {task.response_id for task in tasks}
        if len(planned_ids) != len(tasks):
            raise ValueError(
                "planned tasks contain duplicate response ids; the prompt set has repeated "
                "prompt_ids, which P4a's uniqueness check should have prevented"
            )

        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = read_responses(path, repair=True)
        done_ids = {response.response_id for response in existing}
        if len(done_ids) != len(existing):
            raise ValueError(
                f"{path} contains duplicate response ids; it was written by more than one run. "
                "Refusing to append -- start a new run_id."
            )
        extra = done_ids - planned_ids
        if extra and self._strict:
            raise ValueError(
                f"{path} holds {len(extra)} response(s) outside this plan, e.g. "
                f"{sorted(extra)[0]!r}. Two plans writing into one file produce a response set no "
                "manifest describes. Use a fresh run_id, or pass strict=False if the wider set is "
                "deliberate."
            )

        outstanding = [task for task in tasks if task.response_id not in done_ids]
        generated = 0
        with path.open("a", encoding="utf-8", newline="") as handle:
            for batch in self._batches(outstanding):
                responses = self._run_batch(batch)
                for response in responses:
                    handle.write(_render_line(response))
                # flush + fsync per batch: a kill loses at most this batch, never a written record.
                handle.flush()
                os.fsync(handle.fileno())
                generated += len(responses)
                if progress is not None:
                    progress(len(done_ids) + generated, len(tasks))

        self._verify_no_drop(path, planned_ids)
        return RunResult(
            path=path,
            run_id=self._run_id,
            planned=len(tasks),
            resumed=len(done_ids & planned_ids),
            generated=generated,
            batch_size=self._batch_size,
            seeds=tuple(seeds),
        )

    def _batches(self, tasks: Sequence[RunTask]) -> list[list[RunTask]]:
        """Chunk outstanding work into batches that never straddle a seed.

        The seed drives the RNG for a whole batch, so mixing two seeds in one forward pass would
        generate both under whichever seed happened to be first -- silently, and only when the
        outstanding count is not a multiple of the batch size, which is exactly the state a
        *resumed* run is in. Grouping by seed first makes that unrepresentable.
        """
        batches: list[list[RunTask]] = []
        group: list[RunTask] = []
        for task in tasks:
            if group and (group[0].seed != task.seed or len(group) == self._batch_size):
                batches.append(group)
                group = []
            group.append(task)
        if group:
            batches.append(group)
        return batches

    def _run_batch(self, batch: Sequence[RunTask]) -> list[Response]:
        """Generate one batch. Every task in it shares a seed, since the seed drives the RNG."""
        seeds = {task.seed for task in batch}
        if len(seeds) != 1:  # pragma: no cover - _batches() guarantees this
            raise ValueError(f"a batch must share one seed, got {sorted(seeds)}")
        task = batch[0]
        responses = self._backend.generate(
            [item.prompt for item in batch],
            seed=task.seed,
            sample_index=task.sample_index,
            batch_size=len(batch),
            token_logprobs=self._token_logprobs,
        )
        produced = {response.response_id for response in responses}
        expected = {item.response_id for item in batch}
        if produced != expected:
            raise ValueError(
                "backend returned response ids that do not match the planned ones; the id "
                f"derivation has drifted. missing={sorted(expected - produced)} "
                f"unexpected={sorted(produced - expected)}"
            )
        return responses

    @staticmethod
    def _verify_no_drop(path: Path, planned_ids: set[str]) -> None:
        """The no-drop half of the contract, checked against the file rather than against memory."""
        on_disk = {response.response_id for response in read_responses(path)}
        missing = planned_ids - on_disk
        if missing:
            raise ValueError(
                f"{path} is missing {len(missing)} planned response(s) after the run, e.g. "
                f"{sorted(missing)[0]!r}. The run did not complete and the file must not be "
                "treated as a finished response set."
            )
