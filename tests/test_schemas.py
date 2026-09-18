"""Contract-layer tests: round-trip, immutability, hashability, and the validators.

The P1 gate names these four explicitly. Each guards a property some later phase silently
relies on, so a failure here is a failure of something downstream, not a style complaint.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from margins.schemas import (
    AxisValue,
    AxisValueRegistry,
    CellStat,
    EffectEstimate,
    FrozenDict,
    IdentityCell,
    Item,
    ItemScore,
    Prompt,
    Provenance,
    Response,
    TaskType,
)
from tests.conftest import make_provenance

# --------------------------------------------------------------------------------------
# FrozenDict: the mechanism behind "frozen means frozen"
# --------------------------------------------------------------------------------------


def test_frozen_dict_is_hashable_and_deeply_frozen() -> None:
    raw: dict[str, object] = {"a": 1, "nested": {"b": [1, 2, 3]}, "tags": ["x", "y"]}
    payload = FrozenDict(raw)
    assert hash(payload) == hash(FrozenDict(raw))
    assert isinstance(payload["nested"], FrozenDict)
    assert payload["nested"]["b"] == (1, 2, 3)
    assert payload["tags"] == ("x", "y")


def test_frozen_dict_rejects_mutation() -> None:
    payload = FrozenDict({"a": 1})
    with pytest.raises(TypeError):
        payload["a"] = 2  # type: ignore[index]


def test_frozen_dict_does_not_alias_its_input() -> None:
    """Mutating the source dict after construction must not change the record."""
    source: dict[str, object] = {"a": 1}
    payload = FrozenDict(source)
    source["a"] = 999
    source["b"] = "new"
    assert payload["a"] == 1
    assert "b" not in payload


def test_frozen_dict_equals_plain_mapping() -> None:
    assert FrozenDict({"a": 1}) == {"a": 1}
    assert FrozenDict({"a": 1}) != {"a": 2}


# --------------------------------------------------------------------------------------
# Round-trip, immutability, hashability
# --------------------------------------------------------------------------------------


def _sample_records(provenance: Provenance) -> tuple[object, ...]:
    return (
        provenance,
        AxisValue(axis="gender", value="woman", provenance=provenance),
        IdentityCell(cell_id="c1", axes=(("gender", "woman"),), order=1),
        Item(
            id="i1",
            benchmark="open_gen_templates",
            task_type=TaskType.OPEN_GEN,
            payload={"template_text": "Hello {identity}.", "slots": ["identity"]},
            target_axes=("gender",),
            provenance=provenance,
        ),
        Prompt(
            prompt_id="p1",
            item_id="i1",
            cell_id="c1",
            rendered_text="Hello woman.",
            task_type=TaskType.OPEN_GEN,
            template_version="v1",
            decoding_key="greedy",
        ),
        Response(
            response_id="r1",
            prompt_id="p1",
            model_id="fake/model",
            model_revision="abc123",
            seed=7,
            temperature=0.0,
            decoding_key="greedy",
            decoding_hash="d" * 16,
            quant="nf4",
            text="some text",
            token_logprobs=(-0.1, -0.2),
        ),
        ItemScore(
            score_id="s1",
            response_id="r1",
            scorer_id="fake-scorer",
            harm_type="toxicity",
            raw=0.5,
            standardized=0.1,
            prompt_id="p1",
            cell_id="c1",
            benchmark="open_gen_templates",
            model_id="fake/model",
            seed=7,
            temperature=0.0,
            condition="model",
        ),
        CellStat(
            cell_id="c1",
            benchmark="open_gen_templates",
            model="fake/model",
            harm_type="toxicity",
            n=10,
            mean_standardized=0.1,
            ci_low=-0.2,
            ci_high=0.4,
            powered=True,
        ),
        EffectEstimate(
            benchmark="open_gen_templates",
            model="fake/model",
            axis_pair=("gender", "race"),
            term="gender:race",
            harm_type="toxicity",
            estimate=0.3,
            se=0.1,
            ci_low=0.1,
            ci_high=0.5,
            p=0.02,
            powered=True,
        ),
    )


def test_every_record_round_trips_through_json(provenance: Provenance) -> None:
    for record in _sample_records(provenance):
        assert hasattr(record, "model_validate_json")
        rebuilt = type(record).model_validate_json(record.model_dump_json())  # type: ignore[attr-defined]
        assert rebuilt == record, f"{type(record).__name__} did not survive a JSON round-trip"


def test_every_record_is_immutable(provenance: Provenance) -> None:
    for record in _sample_records(provenance):
        field = next(iter(type(record).model_fields))  # type: ignore[attr-defined]
        with pytest.raises(ValidationError):
            setattr(record, field, None)


def test_every_record_is_hashable(provenance: Provenance) -> None:
    """Invariant 1: container fields are tuples / immutable mappings so records stay hashable."""
    for record in _sample_records(provenance):
        assert isinstance(hash(record), int), f"{type(record).__name__} is not hashable"


def test_identity_cell_is_usable_as_a_grouping_key() -> None:
    """IdentityCell is a dict key throughout the metric layer, so this is load-bearing."""
    a = IdentityCell(cell_id="c1", axes=(("gender", "woman"),), order=1)
    b = IdentityCell(cell_id="c1", axes=(("gender", "woman"),), order=1)
    counts = {a: 1}
    counts[b] = counts[b] + 1
    assert counts == {a: 2}


def test_extra_fields_are_forbidden_on_every_record(provenance: Provenance) -> None:
    with pytest.raises(ValidationError):
        Item(
            id="i1",
            benchmark="b",
            task_type=TaskType.OPEN_GEN,
            provenance=provenance,
            sentiment=0.5,  # type: ignore[call-arg]
        )


# --------------------------------------------------------------------------------------
# Validators
# --------------------------------------------------------------------------------------


def test_provenance_requires_a_version_anchor() -> None:
    with pytest.raises(ValidationError, match="revision or version"):
        make_provenance(revision=None, version=None)
    assert make_provenance(revision=None).version == "v1.0"
    assert make_provenance(version=None).revision == "0" * 40


def test_provenance_rejects_a_malformed_checksum() -> None:
    for bad in ("", "xyz", "A" * 64, "a" * 63):
        with pytest.raises(ValidationError, match="sha256"):
            make_provenance(sha256=bad)


def test_provenance_requires_an_explicit_license() -> None:
    with pytest.raises(ValidationError):
        Provenance(
            source="s",
            version="v1",
            license="",
            sha256="a" * 64,
            retrieved_at=datetime(2026, 9, 8, tzinfo=UTC),
        )


@pytest.mark.parametrize(
    ("axes", "order"),
    [
        ((("gender", "woman"),), 2),
        ((("gender", "woman"), ("race", "Black")), 1),
        ((), 1),
    ],
)
def test_identity_cell_order_must_equal_axis_count(
    axes: tuple[tuple[str, str], ...], order: int
) -> None:
    with pytest.raises(ValidationError):
        IdentityCell(cell_id="c", axes=axes, order=order)


def test_identity_cell_depth_is_capped_at_two() -> None:
    """Invariant 6: no depth-3 cells."""
    with pytest.raises(ValidationError):
        IdentityCell(
            cell_id="c",
            axes=(("gender", "woman"), ("race", "Black"), ("religion", "Muslim")),
            order=3,
        )


def test_identity_cell_axes_must_be_key_unique() -> None:
    with pytest.raises(ValidationError, match="key-unique"):
        IdentityCell(cell_id="c", axes=(("gender", "woman"), ("gender", "man")), order=2)


def test_identity_cell_axes_map_is_a_detached_copy() -> None:
    cell = IdentityCell(cell_id="c", axes=(("gender", "woman"), ("race", "Black")), order=2)
    assert cell.axes_map == {"gender": "woman", "race": "Black"}
    cell.axes_map["gender"] = "man"
    assert cell.axes_map["gender"] == "woman"


def test_identity_cell_carries_no_provenance_field() -> None:
    """An order-2 cell spans two possibly-different sources, so cell-level provenance is
    ambiguous. Provenance attaches to the AxisValue instead."""
    assert "provenance" not in IdentityCell.model_fields
    assert "provenance" in AxisValue.model_fields


def test_registry_rejects_cells_with_unattested_values(
    registry: AxisValueRegistry, cells: tuple[IdentityCell, ...]
) -> None:
    assert registry.validate_cells(cells) == cells

    unattested = IdentityCell(cell_id="c", axes=(("gender", "not-a-descriptor"),), order=1)
    with pytest.raises(ValueError, match="absent from the AxisValue registry"):
        registry.validate_cell(unattested)


def test_gold_is_rejected_for_non_categorical_tasks(provenance: Provenance) -> None:
    """pair_pll carries both sentences in payload; there is no single gold string."""
    for task in (TaskType.PAIR_PLL, TaskType.OPEN_GEN):
        with pytest.raises(ValidationError, match="gold is only meaningful"):
            Item(
                id="i",
                benchmark="b",
                task_type=task,
                gold="answer",
                provenance=provenance,
            )
    for task in (TaskType.QA, TaskType.COREF):
        assert Item(id="i", benchmark="b", task_type=task, gold="a", provenance=provenance).gold


def test_cell_stat_carries_no_significance_fields() -> None:
    """Invariant 3: CellStat is descriptive and carries no p/q at all."""
    assert "p" not in CellStat.model_fields
    assert "q" not in CellStat.model_fields
    assert "family_size" not in CellStat.model_fields
    assert "correction" not in CellStat.model_fields


@pytest.mark.parametrize("record", [CellStat, EffectEstimate])
def test_confidence_intervals_must_be_ordered(record: type) -> None:
    common = {"benchmark": "b", "harm_type": "toxicity", "ci_low": 1.0, "ci_high": 0.0}
    kwargs = (
        {**common, "cell_id": "c", "model": "m", "n": 5, "mean_standardized": 0.0, "powered": True}
        if record is CellStat
        else {
            **common,
            "model": "m",
            "axis_pair": ("gender", "race"),
            "term": "gender:race",
            "estimate": 0.0,
            "se": 0.1,
            "powered": True,
        }
    )
    with pytest.raises(ValidationError, match="ci_low"):
        record(**kwargs)


def _effect(**overrides: object) -> EffectEstimate:
    data: dict[str, object] = {
        "benchmark": "b",
        "model": "m",
        "axis_pair": ("gender", "race"),
        "term": "gender:race",
        "harm_type": "toxicity",
        "estimate": 0.3,
        "se": 0.1,
        "ci_low": 0.1,
        "ci_high": 0.5,
        "powered": True,
    }
    data.update(overrides)
    return EffectEstimate.model_validate(data)


def test_correction_fields_move_together() -> None:
    """A partially corrected estimate is not interpretable, so it is not representable."""
    with pytest.raises(ValidationError, match="set together"):
        _effect(p=0.01, q=0.04)
    with pytest.raises(ValidationError, match="set together"):
        _effect(p=0.01, family_size=10, correction="BH")

    assert _effect(p=0.01, q=0.04, family_size=10, correction="BH").corrected


def test_q_requires_p() -> None:
    with pytest.raises(ValidationError, match="nothing to correct"):
        _effect(q=0.04, family_size=10, correction="BH")


def test_uncorrected_estimate_is_not_marked_corrected() -> None:
    assert not _effect(p=0.02).corrected
    assert not _effect().corrected


def test_axis_pair_must_name_two_distinct_axes() -> None:
    with pytest.raises(ValidationError, match="distinct axes"):
        _effect(axis_pair=("gender", "gender"))
    with pytest.raises(ValidationError):
        _effect(axis_pair=("gender", ""))


def test_response_token_logprobs_is_a_tuple() -> None:
    response = Response(
        response_id="r",
        prompt_id="p",
        model_id="m",
        model_revision="rev",
        seed=1,
        temperature=0.0,
        decoding_key="greedy",
        decoding_hash="h",
        quant="nf4",
        text="t",
        token_logprobs=[-0.1, -0.2],  # type: ignore[arg-type]
    )
    assert isinstance(response.token_logprobs, tuple)
    assert isinstance(hash(response), int)


def _response(**overrides: object) -> Response:
    data: dict[str, object] = {
        "response_id": "r",
        "prompt_id": "p",
        "model_id": "m",
        "model_revision": "rev",
        "seed": 1,
        "temperature": 0.0,
        "decoding_key": "greedy",
        "decoding_hash": "h",
        "quant": "nf4",
        "text": "t",
    }
    data.update(overrides)
    return Response.model_validate(data)


def test_pair_logprobs_default_to_null_for_open_gen() -> None:
    """open_gen responses carry no pair, which is the default rather than a special case."""
    response = _response()
    assert response.logprob_stereo is None
    assert response.logprob_antistereo is None
    assert not response.is_paired
    assert response.stereo_preference is None


def test_pair_logprobs_must_move_together() -> None:
    """Half a pair is not a measurement."""
    with pytest.raises(ValidationError, match="set together or not at all"):
        _response(logprob_stereo=-3.0)
    with pytest.raises(ValidationError, match="set together or not at all"):
        _response(logprob_antistereo=-4.0)


def test_stereo_preference_sign_convention_is_higher_equals_more_harm() -> None:
    """Guards the direction. Swapping the two fields silently flips every pair_pll effect."""
    prefers_stereotype = _response(logprob_stereo=-2.0, logprob_antistereo=-5.0)
    assert prefers_stereotype.is_paired
    assert prefers_stereotype.stereo_preference == pytest.approx(3.0)
    assert prefers_stereotype.stereo_preference is not None
    assert prefers_stereotype.stereo_preference > 0, "preferring the stereotype must score higher"

    prefers_antistereotype = _response(logprob_stereo=-5.0, logprob_antistereo=-2.0)
    assert prefers_antistereotype.stereo_preference == pytest.approx(-3.0)

    indifferent = _response(logprob_stereo=-3.0, logprob_antistereo=-3.0)
    assert indifferent.stereo_preference == pytest.approx(0.0)


def test_paired_response_round_trips_and_stays_hashable() -> None:
    response = _response(logprob_stereo=-2.0, logprob_antistereo=-5.0)
    assert Response.model_validate_json(response.model_dump_json()) == response
    assert isinstance(hash(response), int)


def test_item_score_condition_is_constrained() -> None:
    """The drift control depends on this split existing and being spelled exactly."""
    base: dict[str, object] = {
        "score_id": "s",
        "response_id": "r",
        "scorer_id": "sc",
        "harm_type": "toxicity",
        "raw": 0.1,
        "standardized": 0.0,
        "prompt_id": "p",
        "benchmark": "b",
        "model_id": "m",
        "seed": 1,
        "temperature": 0.0,
    }
    assert ItemScore.model_validate({**base, "condition": "drift_control"}).condition
    with pytest.raises(ValidationError):
        ItemScore.model_validate({**base, "condition": "control"})
