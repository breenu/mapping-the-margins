"""Gate P5's resumability clause: kill mid-run, restart, no duplicate and no drop.

**Why this runs without a model.** Resumability is a property of the harness, not of any
checkpoint. A test that had to load weights to exercise a kill/restart would be slow, would need
network and an accepted licence, and would be testing the wrong thing. The harness is therefore
typed against :class:`~margins.inference.harness.GenerativeBackend`, a Protocol narrow enough for a
fake to satisfy honestly -- so this whole file runs on the core environment with no ``torch``.

The failure this is built for is a **kill mid-write**, which is what actually happens when a
free-tier Kaggle session dies: the file's last line is half a JSON record. That case is repaired.
A *complete* line that does not validate is different, and raises -- discarding it would lose data
while reporting success.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from margins.inference.decoding import DecodingConfig, decoding_config_from_mapping
from margins.inference.harness import (
    GenerativeBackend,
    RunHarness,
    read_responses,
    response_path,
)
from margins.inference.manifest import inference_record
from margins.inference.models import ModelSpec, model_config_from_mapping
from margins.schemas import Prompt, Response, TaskType

RUN_ID = "resume-test-20260909T120000Z-abcdef123456"
SEEDS = (20260908, 20260909)


@pytest.fixture
def spec() -> ModelSpec:
    return model_config_from_mapping(
        {
            "name": "fake_model",
            "model_id": "fake/model",
            "revision": "a" * 40,
            "license": "MIT",
            "role": "dev",
            "family": "llama",
            "gated": False,
            "note": "fake",
        }
    )


@pytest.fixture
def decoding() -> DecodingConfig:
    return decoding_config_from_mapping(
        {
            "key": "greedy_v1",
            "description": "test",
            "prompt_format": "raw",
            "pinned_chat_date": "08 Sep 2026",
            "max_input_tokens": 128,
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": 8,
        }
    )


@pytest.fixture
def prompts() -> list[Prompt]:
    return [
        Prompt(
            prompt_id=f"frame::cell-{index}::person::{index:012d}",
            item_id="open_gen_templates/frame",
            cell_id=f"axis=value-{index}",
            rendered_text=f"Write about person {index}.",
            task_type=TaskType.OPEN_GEN,
            template_version="open_gen_v1.1@2026-09-08",
            decoding_key="greedy_v1",
        )
        for index in range(7)
    ]


class FakeBackend:
    """A backend that can be told to die partway through, like a killed session.

    Satisfies :class:`GenerativeBackend` structurally, inheriting from nothing -- the same shape the
    P1 walking skeleton uses for every other Protocol.
    """

    def __init__(
        self,
        spec: ModelSpec,
        decoding: DecodingConfig,
        *,
        die_after: int | None = None,
        text: str = "generated",
    ) -> None:
        self._spec = spec
        self._decoding = decoding
        self._die_after = die_after
        self._text = text
        self.produced = 0

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    @property
    def decoding(self) -> DecodingConfig:
        return self._decoding

    def generate(
        self,
        prompts: Sequence[Prompt],
        *,
        seed: int | None = None,
        sample_index: int = 0,
        batch_size: int = 1,
        token_logprobs: bool = False,
    ) -> list[Response]:
        from margins.inference.backend import make_response_id

        if self._die_after is not None and self.produced >= self._die_after:
            raise RuntimeError("session died (simulated)")

        responses = [
            Response(
                response_id=make_response_id(
                    prompt_id=prompt.prompt_id,
                    model_name=self._spec.name,
                    model_id=self._spec.model_id,
                    revision=self._spec.revision,
                    seed=seed if seed is not None else 0,
                    sample_index=sample_index,
                    decoding_hash=self._decoding.decoding_hash,
                ),
                prompt_id=prompt.prompt_id,
                model_id=self._spec.model_id,
                model_revision=self._spec.revision,
                seed=seed if seed is not None else 0,
                temperature=self._decoding.temperature,
                decoding_key=self._decoding.key,
                decoding_hash=self._decoding.decoding_hash,
                quant="none",
                text=f"{self._text} for {prompt.prompt_id} @ {seed}",
            )
            for prompt in prompts
        ]
        self.produced += len(responses)
        return responses


def _harness(backend: Any, tmp_path: Path, **kwargs: Any) -> RunHarness:
    return RunHarness(backend, run_id=RUN_ID, out_dir=tmp_path / "responses", **kwargs)


def test_fake_backend_satisfies_the_protocol(spec: ModelSpec, decoding: DecodingConfig) -> None:
    assert isinstance(FakeBackend(spec, decoding), GenerativeBackend)


# --------------------------------------------------------------------------------------
# The gate: kill -> restart -> no dup, no drop
# --------------------------------------------------------------------------------------


def test_kill_midrun_then_restart_loses_and_duplicates_nothing(
    spec: ModelSpec, decoding: DecodingConfig, prompts: list[Prompt], tmp_path: Path
) -> None:
    dying = FakeBackend(spec, decoding, die_after=5)
    harness = _harness(dying, tmp_path, batch_size=2)
    with pytest.raises(RuntimeError, match="session died"):
        harness.run(prompts, seeds=SEEDS)

    partial = read_responses(harness.path)
    assert 0 < len(partial) < len(prompts) * len(SEEDS), "the kill must land mid-run to be a test"
    before = harness.path.read_bytes()

    result = _harness(FakeBackend(spec, decoding), tmp_path, batch_size=2).run(prompts, seeds=SEEDS)

    assert result.planned == len(prompts) * len(SEEDS)
    assert result.resumed == len(partial), "already-finished work must not be re-run"
    assert result.generated == result.planned - result.resumed
    assert result.complete

    final = read_responses(result.path)
    ids = [response.response_id for response in final]
    assert len(ids) == len(set(ids)), "no duplicates"
    assert set(ids) == {task.response_id for task in harness.plan(prompts, seeds=SEEDS)}, "no drop"
    # The resumed prefix is untouched, byte for byte: a restart must be invisible in the output.
    assert result.path.read_bytes().startswith(before)


def test_a_batch_never_straddles_two_seeds(
    spec: ModelSpec, decoding: DecodingConfig, prompts: list[Prompt], tmp_path: Path
) -> None:
    """The seed drives the RNG for a whole batch, so a mixed batch would run under one of them.

    This is specifically a RESUMED-run hazard: it only appears when the outstanding count is not a
    multiple of the batch size, which is precisely the state a restart leaves behind. 7 prompts at
    batch_size 2 is that case.
    """
    harness = _harness(FakeBackend(spec, decoding), tmp_path, batch_size=2)
    tasks = harness.plan(prompts, seeds=SEEDS)
    assert len(tasks) % 2 == 0 and len(prompts) % 2 == 1, "the fixture must exercise the odd case"
    for batch in harness._batches(tasks):
        assert len({task.seed for task in batch}) == 1
        assert len(batch) <= 2


def test_a_second_run_over_finished_work_generates_nothing(
    spec: ModelSpec, decoding: DecodingConfig, prompts: list[Prompt], tmp_path: Path
) -> None:
    first = _harness(FakeBackend(spec, decoding), tmp_path).run(prompts, seeds=SEEDS)
    backend = FakeBackend(spec, decoding)
    second = _harness(backend, tmp_path).run(prompts, seeds=SEEDS)

    assert second.generated == 0
    assert backend.produced == 0, "a completed run must not touch the model at all"
    assert second.resumed == first.planned


# --------------------------------------------------------------------------------------
# Interrupted writes vs. corruption
# --------------------------------------------------------------------------------------


def test_a_truncated_final_record_is_repaired_not_duplicated(
    spec: ModelSpec, decoding: DecodingConfig, prompts: list[Prompt], tmp_path: Path
) -> None:
    """The normal state of a JSONL file whose writer was killed mid-line."""
    harness = _harness(FakeBackend(spec, decoding), tmp_path)
    harness.run(prompts[:3], seeds=(SEEDS[0],))

    complete = harness.path.read_bytes()
    with harness.path.open("ab") as handle:
        handle.write(b'{"response_id":"half-writ')

    result = _harness(FakeBackend(spec, decoding), tmp_path).run(prompts[:3], seeds=(SEEDS[0],))
    assert result.path.read_bytes() == complete, "the partial line is removed, nothing re-appended"
    assert result.generated == 0


def test_a_complete_but_invalid_line_raises_rather_than_being_dropped(tmp_path: Path) -> None:
    """Corruption is not interruption. Silently discarding it would lose data and report success."""
    path = tmp_path / "run.jsonl"
    path.write_text('{"response_id": "x", "not": "a response"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="complete but does not validate"):
        read_responses(path)


def test_missing_file_reads_as_empty(tmp_path: Path) -> None:
    assert read_responses(tmp_path / "absent.jsonl") == []


# --------------------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------------------


def test_strict_refuses_a_file_holding_work_outside_this_plan(
    spec: ModelSpec, decoding: DecodingConfig, prompts: list[Prompt], tmp_path: Path
) -> None:
    """Two plans writing into one file produce a response set no manifest describes."""
    _harness(FakeBackend(spec, decoding), tmp_path).run(prompts, seeds=SEEDS)
    with pytest.raises(ValueError, match="outside this plan"):
        _harness(FakeBackend(spec, decoding), tmp_path).run(prompts[:2], seeds=SEEDS)


def test_repeated_seeds_are_refused(
    spec: ModelSpec, decoding: DecodingConfig, prompts: list[Prompt], tmp_path: Path
) -> None:
    """Two generations at one seed are identical by construction: a duplicate wearing a new id."""
    with pytest.raises(ValueError, match="seeds must be distinct"):
        _harness(FakeBackend(spec, decoding), tmp_path).plan(prompts, seeds=(7, 7))


def test_a_backend_returning_unexpected_ids_is_caught(
    spec: ModelSpec, decoding: DecodingConfig, prompts: list[Prompt], tmp_path: Path
) -> None:
    """If id derivation ever drifts between plan and backend, resumption silently breaks."""

    class DriftingBackend(FakeBackend):
        def generate(self, prompts: Sequence[Prompt], **kwargs: Any) -> list[Response]:
            responses = super().generate(prompts, **kwargs)
            return [
                r.model_copy(update={"response_id": r.response_id + "-drift"}) for r in responses
            ]

    with pytest.raises(ValueError, match="do not match the planned ones"):
        _harness(DriftingBackend(spec, decoding), tmp_path).run(prompts, seeds=(SEEDS[0],))


def test_empty_seeds_and_bad_batch_size_are_refused(
    spec: ModelSpec, decoding: DecodingConfig, prompts: list[Prompt], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="at least one seed"):
        _harness(FakeBackend(spec, decoding), tmp_path).plan(prompts, seeds=())
    with pytest.raises(ValueError, match="batch_size"):
        _harness(FakeBackend(spec, decoding), tmp_path, batch_size=0)


# --------------------------------------------------------------------------------------
# Ids and paths
# --------------------------------------------------------------------------------------


def test_response_ids_are_computable_before_the_work_is_done(
    spec: ModelSpec, decoding: DecodingConfig, prompts: list[Prompt], tmp_path: Path
) -> None:
    """The whole basis of resumability: knowing what is outstanding without generating it."""
    harness = _harness(FakeBackend(spec, decoding), tmp_path)
    planned = harness.plan(prompts, seeds=SEEDS)
    assert len({task.response_id for task in planned}) == len(prompts) * len(SEEDS)

    result = harness.run(prompts, seeds=SEEDS)
    produced = {response.response_id for response in read_responses(result.path)}
    assert produced == {task.response_id for task in planned}


def test_repinning_the_model_changes_every_response_id(
    spec: ModelSpec, decoding: DecodingConfig, prompts: list[Prompt], tmp_path: Path
) -> None:
    """A re-pin must not silently append rows from a different checkpoint to an existing run."""
    harness = _harness(FakeBackend(spec, decoding), tmp_path)
    original = {task.response_id for task in harness.plan(prompts, seeds=SEEDS)}

    repinned = spec.model_copy(update={"revision": "b" * 40})
    other = _harness(FakeBackend(repinned, decoding), tmp_path)
    assert original.isdisjoint({task.response_id for task in other.plan(prompts, seeds=SEEDS)})


def test_editing_the_decoding_profile_changes_every_response_id(
    spec: ModelSpec, decoding: DecodingConfig, prompts: list[Prompt], tmp_path: Path
) -> None:
    harness = _harness(FakeBackend(spec, decoding), tmp_path)
    original = {task.response_id for task in harness.plan(prompts, seeds=SEEDS)}

    edited = decoding.model_copy(update={"max_new_tokens": 9})
    other = _harness(FakeBackend(spec, edited), tmp_path)
    assert original.isdisjoint({task.response_id for task in other.plan(prompts, seeds=SEEDS)})


def test_responses_are_partitioned_by_model(
    spec: ModelSpec, decoding: DecodingConfig, tmp_path: Path
) -> None:
    path = response_path(tmp_path, spec.name, RUN_ID)
    assert path.parent.name == spec.name
    assert path.name == f"{RUN_ID}.jsonl"


def test_run_id_is_required(spec: ModelSpec, decoding: DecodingConfig, tmp_path: Path) -> None:
    """run_id is recorded, never random-per-call: it ties responses to their manifest."""
    with pytest.raises(ValueError, match="run_id must be non-empty"):
        RunHarness(FakeBackend(spec, decoding), run_id="", out_dir=tmp_path)


# --------------------------------------------------------------------------------------
# What the manifest records about the run
# --------------------------------------------------------------------------------------


def test_inference_record_reports_the_actual_quantization(
    spec: ModelSpec, decoding: DecodingConfig
) -> None:
    """Framing constraint 4: the recorded constant must be what happened, not what was intended."""
    from margins.inference.backend import LoadReport

    report = LoadReport(
        model_id=spec.model_id,
        revision=spec.revision,
        quant_requested="nf4",
        quant_actual="none",
        quant_fallback_reason="no CUDA device available",
        device="cpu",
        dtype="float32",
        torch_version="2.14.0",
        torch_num_threads=8,
        blas_backend="openblas",
        prompt_format="raw",
        chat_template_sha256=None,
        pad_token_source="eos",
    )
    record = inference_record(spec, decoding, report, batch_size=4, seeds=SEEDS)

    assert record.quant_requested == "nf4"
    assert record.quant_actual == "none"
    assert record.quant_fallback_reason
    assert record.model_revision == spec.revision
    assert record.decoding_hash == decoding.decoding_hash
    assert record.batch_size == 4
    assert record.seeds == SEEDS


def test_manifest_carries_the_inference_block(tmp_path: Path, spec: ModelSpec) -> None:
    from margins.config import config_from_mapping, write_run_manifest
    from margins.inference.backend import LoadReport
    from margins.inference.decoding import DEFAULT_DECODING, load_decoding_config

    cfg = config_from_mapping(
        {
            "seed": 20260908,
            "run_name": "inference-manifest-test",
            "results_dir": str(tmp_path / "results"),
            "model_name": spec.name,
            "model_id": spec.model_id,
            "model_revision": spec.revision,
        }
    )
    report = LoadReport(
        model_id=spec.model_id,
        revision=spec.revision,
        quant_requested="nf4",
        quant_actual="nf4",
        device="cuda",
        dtype="bfloat16",
        torch_version="2.14.0",
        torch_num_threads=8,
        blas_backend="openblas",
        prompt_format="chat",
        chat_template_sha256="c" * 64,
        pad_token_source="tokenizer",
    )
    manifest = write_run_manifest(
        cfg,
        run_id=RUN_ID,
        timestamp=datetime(2026, 9, 9, 12, 0, tzinfo=UTC),
        inference=inference_record(
            spec, load_decoding_config(DEFAULT_DECODING), report, batch_size=8, seeds=SEEDS
        ),
    )

    assert manifest.run_id == RUN_ID, "an explicit run_id is what makes a resumed run traceable"
    assert manifest.inference is not None
    assert manifest.inference.model_revision == spec.revision
    assert manifest.inference.quant_actual == "nf4"
    assert manifest.inference.batch_size == 8
    # torch / thread count / BLAS already ride in `environment`; no second copy is kept.
    assert manifest.environment.model_fields_set or manifest.environment is not None
    assert "batch size" in manifest.determinism_note.lower()


def test_manifest_inference_block_is_none_for_runs_that_load_no_model(tmp_path: Path) -> None:
    from margins.config import config_from_mapping, write_run_manifest

    cfg = config_from_mapping(
        {"seed": 1, "run_name": "no-model", "results_dir": str(tmp_path / "results")}
    )
    manifest = write_run_manifest(cfg, timestamp=datetime(2026, 9, 9, tzinfo=UTC))
    assert manifest.inference is None
    assert manifest.config.model_name is None
