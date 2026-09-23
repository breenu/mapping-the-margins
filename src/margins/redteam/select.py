"""Choose the frames the DESCRIPTIVE red-team audit renders, under a GPU budget (P11).

The red-team arm is descriptive by decision (21 Sep 2026): a powered arm would cost as much as the
main sweep. The budget therefore buys a fixed number of frames, and the question this module
answers is *which* ones -- a choice that must be neither the analyst's nor an accident of file
order.

**Stratified round-robin over (elicitation angle x target pair).** The screened frame file is
ordered by how the red LM happened to answer, so its first N frames could all come from one angle
or all target one pair. Taking frames in turn from every (angle, pair) stratum spreads the budget
across the three elicitation angles and the six confirmatory pairs -- including the four
non-gender x non-gender pairs novelty claim 5 rests on -- before any stratum gets a second frame.

**Pure and content-ordered.** Within a stratum, frames are taken in ``frame_id`` order. A frame id
is a digest of the frame's text, so the order is fixed by content: no RNG, no clock, no dependence
on file order, and the same frame file always yields the same selection.

Every selected frame is still rendered into EVERY target cell (``margins.redteam.frames``); a cap
changes how many item clusters span the grid, never which cells a frame reaches.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from margins.redteam.frames import Frame

__all__ = ["frame_stratum", "select_frames"]


def frame_stratum(frame: Frame, pair_of: Mapping[str, tuple[str, str]]) -> tuple[str, str, str]:
    """``(elicitation angle, axis_a, axis_b)`` for a frame's elicitation target."""
    try:
        axis_a, axis_b = pair_of[frame.target_cell_id]
    except KeyError as error:
        raise KeyError(
            f"frame {frame.frame_id} targets {frame.target_cell_id!r}, which is not an order-2 "
            "cell of this lattice. The frame file and the lattice disagree; rebuild the frames."
        ) from error
    return (frame.elicitation_template, axis_a, axis_b)


def select_frames(
    frames: Sequence[Frame], max_frames: int, *, pair_of: Mapping[str, tuple[str, str]]
) -> list[Frame]:
    """Up to ``max_frames`` frames, round-robin over strata, returned in their input order.

    Returns every frame unchanged when ``max_frames`` is at least the number available.

    Raises:
        ValueError: if ``max_frames`` is below 1.
    """
    if max_frames < 1:
        raise ValueError(f"max_frames must be at least 1, got {max_frames}")
    if max_frames >= len(frames):
        return list(frames)

    strata: dict[tuple[str, str, str], list[Frame]] = defaultdict(list)
    for frame in frames:
        strata[frame_stratum(frame, pair_of)].append(frame)
    queues = [sorted(strata[key], key=lambda f: f.frame_id) for key in sorted(strata)]

    chosen: set[str] = set()
    depth = 0
    while len(chosen) < max_frames:
        for queue in queues:
            if depth < len(queue) and len(chosen) < max_frames:
                chosen.add(queue[depth].frame_id)
        depth += 1
    return [frame for frame in frames if frame.frame_id in chosen]
