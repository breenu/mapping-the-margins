"""The P1 gate: no-op fakes for every Protocol, composed end to end on fake data.

The point is to prove the **contracts compose** before any real logic exists — that the output
of each stage is a legal input to the next, that no stage needs a field the previous one cannot
supply, and that the awkward parts of the design (Metric needing `cells`, StatsEngine being the
only stage that may stamp significance) actually work when wired together.

Nothing here computes anything meaningful. Every number is arbitrary. A green run means the
plumbing fits, not that the pipeline is correct.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from margins.interfaces import (
    BenchmarkLoader,
    InferenceBackend,
    Lattice,
    Metric,
    PromptGenerator,
    Scorer,
    StatsEngine,
)
from margins.schemas import (
    AxisValueRegistry,
    CellStat,
    EffectEstimate,
    IdentityCell,
    Item,
    ItemScore,
    Prompt,
    Provenance,
    Response,
)
from tests.conftest import make_provenance

TEMPLATE_VERSION = "skeleton-v1"
DECODING_KEY = "greedy"
DECODING_HASH = "0" * 16
MODEL_ID = "fake/tiny-model"
MODEL_REVISION = "revision-abc123"
SEED = 20260908
HARM_TYPES = ("toxicity", "stereotype")


# --------------------------------------------------------------------------------------
# Fakes — each satisfies one Protocol structurally, inheriting from nothing
# --------------------------------------------------------------------------------------


class FakeTemplateLoader:
    """Stands in for a source of template Items (P4's input)."""

    def __init__(self, items: Sequence[Item]) -> None:
        self._items = tuple(items)

    @property
    def name(self) -> str:
        return "open_gen_templates"

    def load(self) -> list[Item]:
        return list(self._items)


class FakeLattice:
    def __init__(self, cells: Sequence[IdentityCell], registry: AxisValueRegistry) -> None:
        self._cells = registry.validate_cells(tuple(cells))

    def cells(self) -> list[IdentityCell]:
        return list(self._cells)


class FakeTemplateRenderer:
    """The *injecting* semantics: a cell is placed into a template.

    This is the renderer that can reach arbitrary lattice cells, including the
    non-gender x non-gender ones gate P3 requires.
    """

    def render(self, items: Sequence[Item], cells: Sequence[IdentityCell]) -> list[Prompt]:
        prompts: list[Prompt] = []
        for item in items:
            template = str(item.payload["template_text"])
            for cell in cells:
                identity = " ".join(value for _, value in cell.axes)
                prompts.append(
                    Prompt(
                        prompt_id=f"{item.id}::{cell.cell_id}",
                        item_id=item.id,
                        cell_id=cell.cell_id,
                        rendered_text=template.replace("{identity}", identity),
                        task_type=item.task_type,
                        template_version=TEMPLATE_VERSION,
                        decoding_key=DECODING_KEY,
                    )
                )
        return prompts


class FakeBackend:
    @property
    def model_id(self) -> str:
        return MODEL_ID

    def run(self, prompts: Sequence[Prompt]) -> list[Response]:
        return [
            Response(
                response_id=f"resp::{prompt.prompt_id}",
                prompt_id=prompt.prompt_id,
                model_id=MODEL_ID,
                model_revision=MODEL_REVISION,
                seed=SEED,
                temperature=0.0,
                decoding_key=prompt.decoding_key,
                decoding_hash=DECODING_HASH,
                quant="nf4",
                text=f"<generated for {prompt.prompt_id}>",
                token_logprobs=(-0.1, -0.2, -0.3),
            )
            for prompt in prompts
        ]


class FakeScorer:
    """One scorer for one harm type. Two are instantiated, per the >=2-scorers rule."""

    def __init__(self, scorer_id: str, harm_type: str, prompts_by_id: dict[str, Prompt]) -> None:
        self._scorer_id = scorer_id
        self._harm_type = harm_type
        self._prompts = prompts_by_id

    @property
    def scorer_id(self) -> str:
        return self._scorer_id

    @property
    def harm_type(self) -> str:
        return self._harm_type

    def score(self, responses: Sequence[Response]) -> list[ItemScore]:
        scores: list[ItemScore] = []
        for index, response in enumerate(responses):
            prompt = self._prompts[response.prompt_id]
            # Both conditions, because the drift control needs matched neutral controls.
            for condition in ("model", "drift_control"):
                scores.append(
                    ItemScore(
                        score_id=f"{self._scorer_id}::{condition}::{response.response_id}",
                        response_id=response.response_id,
                        scorer_id=self._scorer_id,
                        harm_type=self._harm_type,
                        raw=float(index % 5) / 10.0,
                        standardized=float(index % 5) / 10.0 - 0.2,
                        cell_id=prompt.cell_id,
                        benchmark="open_gen_templates",
                        model_id=response.model_id,
                        seed=response.seed,
                        temperature=response.temperature,
                        condition=condition,  # type: ignore[arg-type]
                    )
                )
        return scores


class FakeMetric:
    """Uses `cells` to resolve cell_id -> axes, which is the whole reason it takes them."""

    def compute(
        self, scores: Sequence[ItemScore], cells: Sequence[IdentityCell]
    ) -> tuple[list[CellStat], list[EffectEstimate]]:
        by_cell = {cell.cell_id: cell for cell in cells}
        model_scores = [s for s in scores if s.condition == "model"]

        cell_stats = [
            CellStat(
                cell_id=cell_id,
                benchmark="open_gen_templates",
                model=MODEL_ID,
                harm_type=harm_type,
                n=len(group),
                mean_standardized=sum(s.standardized for s in group) / len(group),
                ci_low=-1.0,
                ci_high=1.0,
                powered=len(group) >= 4,
            )
            for harm_type in HARM_TYPES
            for cell_id, group in (
                (
                    cid,
                    [s for s in model_scores if s.cell_id == cid and s.harm_type == harm_type],
                )
                for cid in by_cell
            )
            if group
        ]

        # An interaction belongs to an axis PAIR, so only order-2 cells contribute one.
        effects: list[EffectEstimate] = []
        for harm_type in HARM_TYPES:
            for cell in cells:
                if cell.order != 2:
                    continue
                axis_a, axis_b = (axis for axis, _ in cell.axes)
                effects.append(
                    EffectEstimate(
                        benchmark="open_gen_templates",
                        model=MODEL_ID,
                        axis_pair=(axis_a, axis_b),
                        term=f"{axis_a}:{axis_b}",
                        harm_type=harm_type,
                        estimate=0.15,
                        se=0.05,
                        ci_low=0.05,
                        ci_high=0.25,
                        p=0.03,
                        powered=True,
                        # q / family_size / correction deliberately unset: only StatsEngine
                        # may stamp those.
                    )
                )
        return cell_stats, effects


class FakeStatsEngine:
    """The only stage that stamps q, family_size and correction."""

    def correct(self, effects: Sequence[EffectEstimate]) -> list[EffectEstimate]:
        family_size = len(effects)
        return [
            effect.model_copy(
                update={
                    "q": min(1.0, (effect.p or 1.0) * family_size),
                    "family_size": family_size,
                    "correction": "BH",
                }
            )
            for effect in effects
        ]


# --------------------------------------------------------------------------------------
# The skeleton
# --------------------------------------------------------------------------------------


@pytest.fixture
def pipeline(
    template_items: tuple[Item, ...],
    cells: tuple[IdentityCell, ...],
    registry: AxisValueRegistry,
) -> dict[str, object]:
    loader = FakeTemplateLoader(template_items)
    lattice = FakeLattice(cells, registry)
    renderer = FakeTemplateRenderer()

    items = loader.load()
    lattice_cells = lattice.cells()
    prompts = renderer.render(items, lattice_cells)
    responses = FakeBackend().run(prompts)

    prompts_by_id = {p.prompt_id: p for p in prompts}
    scorers = [
        FakeScorer("fake-detoxify", "toxicity", prompts_by_id),
        FakeScorer("fake-perspective", "toxicity", prompts_by_id),
        FakeScorer("fake-stereo-a", "stereotype", prompts_by_id),
        FakeScorer("fake-stereo-b", "stereotype", prompts_by_id),
    ]
    scores = [score for scorer in scorers for score in scorer.score(responses)]

    cell_stats, effects = FakeMetric().compute(scores, lattice_cells)
    corrected = FakeStatsEngine().correct(effects)

    return {
        "items": items,
        "cells": lattice_cells,
        "prompts": prompts,
        "responses": responses,
        "scores": scores,
        "cell_stats": cell_stats,
        "effects": effects,
        "corrected": corrected,
        "scorers": scorers,
    }


def test_fakes_satisfy_their_protocols(
    template_items: tuple[Item, ...],
    cells: tuple[IdentityCell, ...],
    registry: AxisValueRegistry,
) -> None:
    """isinstance against a runtime_checkable Protocol checks attribute presence only."""
    assert isinstance(FakeTemplateLoader(template_items), BenchmarkLoader)
    assert isinstance(FakeLattice(cells, registry), Lattice)
    assert isinstance(FakeTemplateRenderer(), PromptGenerator)
    assert isinstance(FakeBackend(), InferenceBackend)
    assert isinstance(FakeScorer("s", "toxicity", {}), Scorer)
    assert isinstance(FakeMetric(), Metric)
    assert isinstance(FakeStatsEngine(), StatsEngine)


def test_skeleton_runs_end_to_end(pipeline: dict[str, object]) -> None:
    for stage in ("items", "cells", "prompts", "responses", "scores", "cell_stats", "corrected"):
        assert pipeline[stage], f"stage {stage!r} produced nothing"


def test_every_stage_links_to_the_previous_one(pipeline: dict[str, object]) -> None:
    """No orphans: each record's foreign key resolves in the stage before it."""
    items: list[Item] = pipeline["items"]  # type: ignore[assignment]
    prompts: list[Prompt] = pipeline["prompts"]  # type: ignore[assignment]
    responses: list[Response] = pipeline["responses"]  # type: ignore[assignment]
    scores: list[ItemScore] = pipeline["scores"]  # type: ignore[assignment]
    lattice_cells: list[IdentityCell] = pipeline["cells"]  # type: ignore[assignment]

    item_ids = {item.id for item in items}
    cell_ids = {cell.cell_id for cell in lattice_cells}
    prompt_ids = {p.prompt_id for p in prompts}
    response_ids = {r.response_id for r in responses}

    assert all(p.item_id in item_ids for p in prompts)
    assert all(p.cell_id in cell_ids for p in prompts)
    assert all(r.prompt_id in prompt_ids for r in responses)
    assert all(s.response_id in response_ids for s in scores)


def test_prompt_count_is_the_template_by_cell_cross_product(pipeline: dict[str, object]) -> None:
    items: list[Item] = pipeline["items"]  # type: ignore[assignment]
    lattice_cells: list[IdentityCell] = pipeline["cells"]  # type: ignore[assignment]
    prompts: list[Prompt] = pipeline["prompts"]  # type: ignore[assignment]
    assert len(prompts) == len(items) * len(lattice_cells)


def test_every_prompt_traces_to_a_provenance_carrying_item(pipeline: dict[str, object]) -> None:
    """The reason templates are Items: without it, templated prompts have no audit trail."""
    items: list[Item] = pipeline["items"]  # type: ignore[assignment]
    prompts: list[Prompt] = pipeline["prompts"]  # type: ignore[assignment]
    by_id = {item.id: item for item in items}
    for prompt in prompts:
        assert isinstance(by_id[prompt.item_id].provenance, Provenance)


def test_skeleton_reaches_a_non_gender_by_non_gender_cell(pipeline: dict[str, object]) -> None:
    """Gate P3's shape: this coverage is reachable only via the open_gen / red-team path."""
    prompts: list[Prompt] = pipeline["prompts"]  # type: ignore[assignment]
    lattice_cells: list[IdentityCell] = pipeline["cells"]  # type: ignore[assignment]
    by_id = {c.cell_id: c for c in lattice_cells}

    reached = {
        tuple(sorted(axis for axis, _ in by_id[p.cell_id].axes))
        for p in prompts
        if p.cell_id is not None and by_id[p.cell_id].order == 2
    }
    assert any("gender" not in pair for pair in reached), f"only gender-anchored pairs: {reached}"


def test_each_harm_type_has_at_least_two_scorers(pipeline: dict[str, object]) -> None:
    scores: list[ItemScore] = pipeline["scores"]  # type: ignore[assignment]
    by_harm: dict[str, set[str]] = {}
    for score in scores:
        by_harm.setdefault(score.harm_type, set()).add(score.scorer_id)
    assert by_harm, "no scores produced"
    for harm_type, scorer_ids in by_harm.items():
        assert len(scorer_ids) >= 2, f"{harm_type} has only {scorer_ids}"


def test_drift_control_condition_is_present(pipeline: dict[str, object]) -> None:
    """Without matched control scores the drift control cannot be estimated at all."""
    scores: list[ItemScore] = pipeline["scores"]  # type: ignore[assignment]
    assert {s.condition for s in scores} == {"model", "drift_control"}


def test_metric_emits_uncorrected_estimates(pipeline: dict[str, object]) -> None:
    """Invariant 3: nothing but StatsEngine may set q/family_size/correction."""
    effects: list[EffectEstimate] = pipeline["effects"]  # type: ignore[assignment]
    assert effects
    for effect in effects:
        assert not effect.corrected
        assert effect.q is None
        assert effect.family_size is None
        assert effect.correction is None


def test_stats_engine_stamps_every_estimate(pipeline: dict[str, object]) -> None:
    corrected: list[EffectEstimate] = pipeline["corrected"]  # type: ignore[assignment]
    assert corrected
    for effect in corrected:
        assert effect.corrected
        assert effect.family_size == len(corrected)
        assert effect.correction == "BH"


def test_stats_engine_does_not_mutate_its_input(pipeline: dict[str, object]) -> None:
    """Invariant 1: transformations return NEW records; inputs stay untouched."""
    effects: list[EffectEstimate] = pipeline["effects"]  # type: ignore[assignment]
    corrected: list[EffectEstimate] = pipeline["corrected"]  # type: ignore[assignment]
    assert all(not e.corrected for e in effects)
    assert all(c.corrected for c in corrected)
    assert effects[0] is not corrected[0]


def test_cell_stats_mark_underpowered_cells(pipeline: dict[str, object]) -> None:
    """Invariant 4: powered is populated so the report layer can filter on it."""
    cell_stats: list[CellStat] = pipeline["cell_stats"]  # type: ignore[assignment]
    assert cell_stats
    assert all(isinstance(stat.powered, bool) for stat in cell_stats)


def test_interaction_estimates_exist_only_for_order_two_cells(
    pipeline: dict[str, object],
) -> None:
    """An order-1 cell has no interaction; that is why EffectEstimate is keyed by axis pair."""
    effects: list[EffectEstimate] = pipeline["effects"]  # type: ignore[assignment]
    lattice_cells: list[IdentityCell] = pipeline["cells"]  # type: ignore[assignment]
    order_two = [c for c in lattice_cells if c.order == 2]
    assert len(effects) == len(order_two) * len(HARM_TYPES)
    for effect in effects:
        assert effect.axis_pair[0] != effect.axis_pair[1]


def test_records_survive_serialization_at_every_stage(pipeline: dict[str, object]) -> None:
    """Every stage's output must reach Parquet/JSONL and come back intact."""
    for stage in ("items", "prompts", "responses", "scores", "cell_stats", "corrected"):
        records = pipeline[stage]
        assert isinstance(records, list)
        for record in records:
            rebuilt = type(record).model_validate_json(record.model_dump_json())
            assert rebuilt == record


def test_skeleton_uses_at_least_three_fake_items(pipeline: dict[str, object]) -> None:
    items: list[Item] = pipeline["items"]  # type: ignore[assignment]
    assert len(items) >= 3
    assert all(item.provenance == make_provenance() for item in items)
