"""The on-target audit scaffold and the on-target-% metric (P8).

Gate P8 requires "human on-target % on a sample reported". The labeling is the human's; what is
tested here is the instrument that draws the sample and computes the number -- the same split, and
the same tests, as ``tests/test_calibration.py`` applies to the P6 calibration scaffold.

The one behaviour worth naming: **an empty denominator reports ``None``, never ``0.0``.** "No
frames were labeled for this pair" and "every labeled frame missed" are different facts, and
reporting both as zero would let an unaudited pair read as a failed one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from margins.redteam import (
    OnTargetLabel,
    OnTargetUnit,
    on_target_metrics,
    read_on_target_labels,
    render_on_target_report,
    sample_for_on_target,
    write_on_target_sample,
)
from margins.redteam.config import OnTargetSettings
from margins.redteam.frames import Frame

PAIR_A = "alpha x beta"
PAIR_B = "alpha x gamma"


def _frame(index: int, cell_id: str, n_targets: int = 1) -> Frame:
    return Frame(
        frame_id=f"f{index:04d}",
        text=f"Write about [PERSON] in situation {index}.",
        template_text=f"Write about {{{{ identity }}}} in situation {index}.",
        target_cell_id=cell_id,
        target_cell_ids=tuple(f"{cell_id}-{n}" for n in range(n_targets)) or (cell_id,),
        elicitation_template="direct",
        response_id=f"r{index}",
        seed=1,
    )


def _frames() -> list[Frame]:
    """Deliberately lopsided: eight frames on one pair, two on the other.

    Stratification is what stops the small pair being estimated from leftovers, so the fixture
    has to be imbalanced for the test to mean anything.
    """
    return [_frame(i, "alpha=a1|beta=b1") for i in range(8)] + [
        _frame(100 + i, "alpha=a1|gamma=g1") for i in range(2)
    ]


def _examples(frames: list[Frame]) -> dict[str, str]:
    return {frame.frame_id: f"rendered: {frame.text}" for frame in frames}


def _settings(size: int = 4, seed: int = 99) -> OnTargetSettings:
    return OnTargetSettings(sample_size=size, seed=seed)


# ----------------------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------------------


def test_the_sample_is_deterministic() -> None:
    """A reviewer must be able to redraw the exact sample a reported number came from."""
    frames = _frames()
    first = sample_for_on_target(frames, _examples(frames), _settings())
    second = sample_for_on_target(frames, _examples(frames), _settings())
    assert [u.frame_id for u in first] == [u.frame_id for u in second]


def test_a_different_seed_draws_a_different_sample() -> None:
    frames = _frames()
    a = sample_for_on_target(frames, _examples(frames), _settings(seed=1))
    b = sample_for_on_target(frames, _examples(frames), _settings(seed=2))
    assert [u.frame_id for u in a] != [u.frame_id for u in b] or len(frames) <= 4


def test_every_target_pair_is_represented() -> None:
    """Proportional allocation with a floor of one: the small pair is never rounded away.

    The non-gender x non-gender pairs are what novelty claim 5 rests on, and they are exactly the
    pairs a proportional-only allocation would drop when their yield is low.
    """
    frames = _frames()
    units = sample_for_on_target(frames, _examples(frames), _settings(size=4))
    assert {u.target_pair for u in units} == {PAIR_A, PAIR_B}


def test_the_sample_never_exceeds_the_budget() -> None:
    """The binding constraint is human time, so the size is a cap, not a target to overshoot."""
    frames = _frames()
    for size in (1, 3, 4, 7, 10, 50):
        units = sample_for_on_target(frames, _examples(frames), _settings(size=size))
        assert len(units) == min(size, len(frames))


def test_the_labeler_is_shown_a_rendered_prompt_not_a_template() -> None:
    """ "Is this on-target" is not answerable from a frame with a hole in it."""
    frames = _frames()
    units = sample_for_on_target(frames, _examples(frames), _settings())
    for unit in units:
        assert unit.rendered_example.startswith("rendered: ")
        assert "[PERSON]" in unit.text


def test_a_frame_without_a_rendered_example_is_refused() -> None:
    frames = _frames()
    examples = _examples(frames)
    examples.pop(frames[0].frame_id)
    with pytest.raises(KeyError, match="no rendered example"):
        sample_for_on_target(frames, examples, _settings())


def test_sampling_nothing_raises_rather_than_returning_an_empty_audit() -> None:
    with pytest.raises(ValueError, match="nothing to audit"):
        sample_for_on_target([], {}, _settings())


def test_the_sample_file_is_byte_deterministic(tmp_path: Path) -> None:
    frames = _frames()
    units = sample_for_on_target(frames, _examples(frames), _settings())
    first = write_on_target_sample(units, tmp_path / "a.jsonl").read_bytes()
    second = write_on_target_sample(units, tmp_path / "b.jsonl").read_bytes()
    assert first == second
    assert first.endswith(b"\n")
    assert b"\r\n" not in first


# ----------------------------------------------------------------------------------------
# The metric
# ----------------------------------------------------------------------------------------


def test_on_target_rate_against_a_tiny_labeled_fixture() -> None:
    units = [
        OnTargetUnit(
            frame_id=f"f{i}",
            text="Write about [PERSON].",
            rendered_example="Write about a person.",
            target_cell_id="alpha=a1|beta=b1",
            target_pair=PAIR_A,
            n_targets=1,
            elicitation_template="direct",
        )
        for i in range(4)
    ]
    labels = [
        OnTargetLabel(frame_id="f0", on_target=1, labeler="human"),
        OnTargetLabel(frame_id="f1", on_target=1, labeler="human"),
        OnTargetLabel(frame_id="f2", on_target=1, labeler="human"),
        OnTargetLabel(frame_id="f3", on_target=0, labeler="human"),
    ]
    metrics = on_target_metrics(units, labels)
    total = next(m for m in metrics if m.stratum == "ALL")
    assert total.n_labeled == 4
    assert total.n_on_target == 3
    assert total.on_target_rate == pytest.approx(0.75)


def test_an_unlabeled_stratum_reports_none_not_zero() -> None:
    """The distinction the whole scaffold turns on: unaudited is not failed."""
    units = [
        OnTargetUnit(
            frame_id="f0",
            text="[PERSON]",
            rendered_example="x",
            target_cell_id="alpha=a1|beta=b1",
            target_pair=PAIR_A,
            n_targets=1,
            elicitation_template="direct",
        ),
        OnTargetUnit(
            frame_id="f1",
            text="[PERSON]",
            rendered_example="x",
            target_cell_id="alpha=a1|gamma=g1",
            target_pair=PAIR_B,
            n_targets=1,
            elicitation_template="direct",
        ),
    ]
    metrics = on_target_metrics(units, [OnTargetLabel(frame_id="f0", on_target=0, labeler="h")])

    unlabeled = next(m for m in metrics if m.stratum == PAIR_B)
    assert unlabeled.n_labeled == 0
    assert unlabeled.on_target_rate is None

    labeled_and_missed = next(m for m in metrics if m.stratum == PAIR_A)
    assert labeled_and_missed.n_labeled == 1
    assert labeled_and_missed.on_target_rate == 0.0, "a real zero must still read as zero"


def test_an_unlabeled_frame_is_not_a_miss() -> None:
    """It counts toward no denominator at all, rather than being scored against the labeler."""
    units = [
        OnTargetUnit(
            frame_id=f"f{i}",
            text="[PERSON]",
            rendered_example="x",
            target_cell_id="alpha=a1|beta=b1",
            target_pair=PAIR_A,
            n_targets=1,
            elicitation_template="direct",
        )
        for i in range(3)
    ]
    metrics = on_target_metrics(units, [OnTargetLabel(frame_id="f0", on_target=1, labeler="h")])
    total = next(m for m in metrics if m.stratum == "ALL")
    assert total.n_labeled == 1
    assert total.on_target_rate == 1.0


def test_a_non_binary_label_is_refused() -> None:
    with pytest.raises(ValueError):
        OnTargetLabel(frame_id="f0", on_target=2, labeler="h")


def test_duplicate_labels_are_refused(tmp_path: Path) -> None:
    """A double-counted frame silently distorts the rate."""
    path = tmp_path / "labels.jsonl"
    rows = [
        {"frame_id": "f0", "on_target": 1, "labeler": "h"},
        {"frame_id": "f0", "on_target": 0, "labeler": "h"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate label"):
        read_on_target_labels(path)


def test_labels_outside_the_sample_are_refused(tmp_path: Path) -> None:
    """Evaluating on unsampled frames breaks the stratification the rate is conditioned on."""
    frames = _frames()
    units = sample_for_on_target(frames, _examples(frames), _settings())
    path = tmp_path / "labels.jsonl"
    path.write_text(
        json.dumps({"frame_id": "not-sampled", "on_target": 1, "labeler": "h"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not in the sample"):
        read_on_target_labels(path, units=units)


# ----------------------------------------------------------------------------------------
# The report
# ----------------------------------------------------------------------------------------


def test_the_report_shows_an_em_dash_for_an_empty_denominator() -> None:
    units = [
        OnTargetUnit(
            frame_id="f0",
            text="[PERSON]",
            rendered_example="x",
            target_cell_id="alpha=a1|beta=b1",
            target_pair=PAIR_A,
            n_targets=1,
            elicitation_template="direct",
        )
    ]
    report = render_on_target_report(
        on_target_metrics(units, []), redteam_version="tiny-rt@test", n_labels=0
    )
    assert "—" in report
    assert "0.0%" not in report
    assert "not** an estimate of" in report, "the sample is chosen to be hard; say so"


def test_the_report_is_byte_deterministic() -> None:
    units = [
        OnTargetUnit(
            frame_id="f0",
            text="[PERSON]",
            rendered_example="x",
            target_cell_id="alpha=a1|beta=b1",
            target_pair=PAIR_A,
            n_targets=1,
            elicitation_template="direct",
        )
    ]
    metrics = on_target_metrics(units, [OnTargetLabel(frame_id="f0", on_target=1, labeler="h")])
    first = render_on_target_report(metrics, redteam_version="v", n_labels=1)
    second = render_on_target_report(metrics, redteam_version="v", n_labels=1)
    assert first == second
    assert first.endswith("\n")
