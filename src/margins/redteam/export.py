"""Red-team artifacts: the frame file, the rejection log, and the audit report (P8).

Held to the same standard as the P3/P4a exports — no timestamps, no iteration over unordered
containers, an explicit ``\\n`` terminator, UTF-8 without a BOM — so a regenerated frame file that
differs byte-for-byte means the frames changed, rather than that the file was written on the other
operating system.

Three artifacts beyond the frame file, and each exists because a specific question has to be
answerable without re-running a GPU:

* ``rejected_frames.csv`` — every discarded candidate with one machine-readable reason. Gate P8 is
  "dedup + filter logged", and a filter whose rejections are not inspectable is a filter nobody can
  criticise.
* ``redteam_report.md`` — the counts: candidates in, rejections by reason, frames out, and the
  **per-pair coverage table**. Coverage had to be *measured* rather than assumed, exactly as the
  CrowS order-2 yield did: an elicitation strategy that quietly produces nothing for one pair would
  otherwise show up only as a thin cell weeks later.
* ``on_target_sample.jsonl`` — written by :mod:`margins.redteam.on_target`.

Outputs land under ``data/redteam/`` — gitignored and regenerable.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from margins.config import LATTICE_INPUT_KEY
from margins.prompts.config import file_sha256
from margins.redteam.config import (
    ELICITATION_INPUT_PREFIX,
    REDTEAM_SET_INPUT_KEY,
    RedTeamConfig,
)
from margins.redteam.filter import REJECTION_REASONS, Rejection
from margins.redteam.frames import Frame

__all__ = [
    "FRAMES_FILENAME",
    "REDTEAM_REPORT_FILENAME",
    "REJECTED_FILENAME",
    "coverage_rows",
    "read_frames",
    "redteam_inputs",
    "render_frames_jsonl",
    "render_redteam_report",
    "render_rejected_csv",
    "write_frames",
    "write_redteam_report",
    "write_rejected",
]

FRAMES_FILENAME: Final[str] = "frames_v1.jsonl"
REJECTED_FILENAME: Final[str] = "rejected_frames.csv"
REDTEAM_REPORT_FILENAME: Final[str] = "redteam_report.md"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" keeps the explicit \n from becoming \r\n on Windows, which is what makes the
    # byte-identity check meaningful across win_amd64 and linux_x86_64.
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    return path


# --------------------------------------------------------------------------------------
# The frame file
# --------------------------------------------------------------------------------------


def _frame_record(frame: Frame) -> dict[str, object]:
    return {
        "frame_id": frame.frame_id,
        "text": frame.text,
        "template_text": frame.template_text,
        "target_cell_id": frame.target_cell_id,
        "target_cell_ids": list(frame.target_cell_ids),
        "elicitation_template": frame.elicitation_template,
        "response_id": frame.response_id,
        "seed": frame.seed,
    }


def render_frames_jsonl(frames: Sequence[Frame]) -> str:
    """Render frames as JSONL, one record per line, in the order given (frame-id order)."""
    return "".join(
        json.dumps(_frame_record(frame), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
        for frame in frames
    )


def write_frames(frames: Sequence[Frame], out_dir: Path) -> Path:
    """Write ``<out_dir>/frames_v1.jsonl`` and return its path."""
    return _write(out_dir / FRAMES_FILENAME, render_frames_jsonl(frames))


def read_frames(path: Path) -> list[Frame]:
    """Read the frame file back, in file order. The inverse of :func:`render_frames_jsonl`.

    Lives beside the writer so the two cannot drift apart — the same reason
    ``margins.prompts.export.read_prompts`` does.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"frame file not found: {path}. It is gitignored and regenerable -- run "
            "`uv run python scripts/build_redteam_frames.py`."
        )
    frames: list[Frame] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        frames.append(
            Frame(
                frame_id=str(record["frame_id"]),
                text=str(record["text"]),
                template_text=str(record["template_text"]),
                target_cell_id=str(record["target_cell_id"]),
                target_cell_ids=tuple(str(value) for value in record["target_cell_ids"]),
                elicitation_template=str(record["elicitation_template"]),
                response_id=str(record["response_id"]),
                seed=int(record["seed"]),
            )
        )
    return frames


# --------------------------------------------------------------------------------------
# The rejection log
# --------------------------------------------------------------------------------------


def render_rejected_csv(rejections: Sequence[Rejection]) -> str:
    """Render ``rejected_frames.csv``: one row per discarded candidate, with its reason."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["reason", "target_cell_id", "response_id", "detail", "text"])
    for rejection in sorted(
        rejections, key=lambda r: (r.reason, r.target_cell_id, r.text, r.response_id)
    ):
        writer.writerow(
            [
                rejection.reason,
                rejection.target_cell_id,
                rejection.response_id,
                rejection.detail,
                rejection.text,
            ]
        )
    return buffer.getvalue()


def write_rejected(rejections: Sequence[Rejection], out_dir: Path) -> Path:
    """Write ``<out_dir>/rejected_frames.csv`` and return its path."""
    return _write(out_dir / REJECTED_FILENAME, render_rejected_csv(rejections))


# --------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------


def coverage_rows(
    frames: Sequence[Frame], target_cells: Sequence[str]
) -> list[tuple[str, int, int]]:
    """``(target pair, cells targeted, frames elicited)`` — the coverage that had to be measured.

    Counted over ``target_cell_ids``, i.e. every cell that elicited a frame, not only the canonical
    first. A pair with cells but no frames appears with a zero rather than being absent, because
    "this pair produced nothing" is the finding the table exists to surface.
    """

    def pair_of(cell_id: str) -> str:
        return " x ".join(sorted(part.split("=", 1)[0] for part in cell_id.split("|")))

    cells_by_pair: dict[str, set[str]] = {}
    for cell_id in target_cells:
        cells_by_pair.setdefault(pair_of(cell_id), set()).add(cell_id)

    frames_by_pair: dict[str, int] = dict.fromkeys(cells_by_pair, 0)
    for frame in frames:
        for cell_id in frame.target_cell_ids or (frame.target_cell_id,):
            frames_by_pair[pair_of(cell_id)] = frames_by_pair.get(pair_of(cell_id), 0) + 1

    return [
        (pair, len(cells_by_pair.get(pair, set())), frames_by_pair.get(pair, 0))
        for pair in sorted(set(cells_by_pair) | set(frames_by_pair))
    ]


def render_redteam_report(
    config: RedTeamConfig,
    *,
    frames: Sequence[Frame],
    rejections: Sequence[Rejection],
    target_cells: Sequence[str],
    n_candidates: int,
    n_responses: int,
    generator_model_id: str,
    generator_revision: str,
    n_audit_prompts: int,
) -> str:
    """Render the red-team generation report. Generated file; never hand-edited."""
    counts: dict[str, int] = dict.fromkeys(REJECTION_REASONS, 0)
    for rejection in rejections:
        counts[rejection.reason] += 1

    generic = [frame for frame in frames if frame.n_targets > 1]
    lines = [
        "# Red-team generation report",
        "",
        f"Red-team set: `{config.redteam_version}`  ",
        f"Generator: `{generator_model_id}` @ `{generator_revision}`  ",
        f"Elicitation decoding profile: `{config.generator.decoding_name}`  ",
        f"Audit decoding profile: `{config.decoding_key}`",
        "",
        "Generated by `scripts/build_redteam_frames.py`. **Do not edit by hand.**",
        "",
        "## What this set is",
        "",
        "Frames are generated by a red LM **targeting a named lattice cell**, then rendered into",
        "**every** target cell through the unchanged P4a `TemplateRenderer`. The target survives",
        "as provenance and as the unit of the on-target audit, and nothing else: rendering a frame",
        "only into its own target would confound 'frame targeted at cell X' with 'cell X', and any",
        "difference between frames would land on that cell's mean and read as an identity effect.",
        "",
        "## Counts",
        "",
        f"- red LM generations read: **{n_responses}**",
        f"- candidate frames parsed: **{n_candidates}**",
        f"- rejected: **{len(rejections)}**",
        f"- frames kept: **{len(frames)}**",
        f"- target cells: **{len(set(target_cells))}**",
        f"- audit prompts rendered: **{n_audit_prompts}** "
        "= frames x cells x carriers (carrier-free frames emitted once)",
        f"- frames elicited by more than one target: **{len(generic)}** "
        f"of {len(frames)} — a frame many targets produce is a *generic* frame, which is a fact "
        "about how cell-targeted the elicitation really was",
        "",
        "## Rejections by reason",
        "",
        "| reason | n |",
        "|---|---:|",
    ]
    for reason in REJECTION_REASONS:
        lines.append(f"| `{reason}` | {counts[reason]} |")

    lines += [
        "",
        "`leakage_guard` is the authoritative one. Every other reason is a **pre-screen**: the",
        "guard itself runs on every surviving frame, rendered into every target cell, and a frame",
        "that raises there is dropped here rather than taking the run down. Nothing in",
        "`frames_v1.jsonl` has failed to render cleanly into every cell it will be used in.",
        "",
        "Full detail, one row per discarded candidate, is in `rejected_frames.csv`.",
        "",
        "## Coverage by target pair",
        "",
        "Measured, not assumed — an elicitation strategy that quietly produces nothing for one",
        "pair would otherwise surface only as a thin cell weeks later.",
        "",
        "A zero in the last column means no surviving frame was ever elicited for that pair, which",
        "is a coverage hole, not a rounding artifact. The count is frame x target-cell pairs, so a",
        "frame elicited for six cells of one axis pair contributes six.",
        "",
        "| target pair | cells | frame elicitations |",
        "|---|---:|---:|",
    ]
    for pair, n_cells, n_frames in coverage_rows(frames, target_cells):
        lines.append(f"| `{pair}` | {n_cells} | {n_frames} |")

    lines += [
        "",
        "## What is NOT in this file",
        "",
        "The **on-target %** — whether a frame plausibly probes the target it names — is a human",
        "judgement and lives in `on_target_report.md`, which is not written until real labels",
        "exist. `on_target_sample.jsonl` is the sheet to fill in.",
        "",
        "Whether red-team prompts surface more harm than static ones is **E3**, and it is measured",
        "through the unified P6 -> P7 -> P9 path, not asserted here.",
        "",
    ]
    return "\n".join(lines) + "\n"


def write_redteam_report(text: str, out_dir: Path) -> Path:
    """Write ``<out_dir>/redteam_report.md`` and return its path."""
    return _write(out_dir / REDTEAM_REPORT_FILENAME, text)


# --------------------------------------------------------------------------------------
# Manifest inputs
# --------------------------------------------------------------------------------------


def redteam_inputs(
    *,
    lattice_config_path: Path,
    redteam_config_path: Path,
    elicitation_paths: Mapping[str, Path],
) -> dict[str, str]:
    """``{source_name: sha256}`` for every input that decided what the red LM was asked.

    Passed to ``write_run_manifest(inputs=...)``. The elicitation templates are in here
    individually, not folded into the config digest, because *what the red LM was asked* is the
    red-team distribution — the single thing about P8 a reproducer most needs to be able to check.
    """
    inputs = {
        LATTICE_INPUT_KEY: file_sha256(lattice_config_path),
        REDTEAM_SET_INPUT_KEY: file_sha256(redteam_config_path),
    }
    for template_id, path in sorted(elicitation_paths.items()):
        inputs[f"{ELICITATION_INPUT_PREFIX}{template_id}"] = file_sha256(path)
    return inputs
