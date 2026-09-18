"""The scorer identity-drift control (P6) -- a designed experiment, not a caveat.

**The problem it exists for.** Detoxify and classifiers like it are documented to score text higher
purely for *mentioning* identity terms, independent of any harm. Our estimand is "harm varies
non-additively across identity cells", estimated as the interaction in ``harm ~ A + B + A:B``. A
scorer whose own response to identity mentions is non-additive across term pairs therefore produces
a **non-zero interaction from text that is identical across cells** -- a fully fabricated emergence
result, arrived at through a pipeline that was working correctly the whole time.

**Why "two scorers with reported agreement" does not address it.** Two identity-sensitive
classifiers agree with each other and are both wrong in the same direction. Agreement measures
shared behaviour, not shared validity (PROJECT_CONTEXT.md section 9). The roster's
mechanism-diversity rule helps, but only the control actually measures the thing.

**The control.** Score a matched set of known-neutral texts -- same descriptors, same templates,
no model output involved -- through **every** scorer, and emit the result keyed so the *same*
``A + B + A:B`` model can be fitted on the scorer's own responses to identity terms. The scorer's
interaction is then reported **beside** the model's.

**This module does NOT subtract them.** Both series are produced and both are emitted; the fitting,
the CIs and the enforcement belong to P7/P9. The rule those phases enforce, recorded here so it
cannot be lost: *if a scorer's own interaction is comparable in magnitude to the model's, that harm
type is DESCRIPTIVE-ONLY.* P6's job is to produce numbers that make that check possible.

**Two control families**, because one is not enough to tell drift from a style artifact:

* ``prompt_echo`` -- the P4a rendered prompts themselves. Exactly matched by construction, adds no
  new construct choice, and every string in it has already been through the P4a phrase audit.
* ``neutral_declarative`` -- flat declarative sentences over the same cells, rendered through the
  **ordinary** :class:`~margins.prompts.render.TemplateRenderer` from frames in
  ``configs/scoring/control_templates/``. Present because a prompt is an imperative instruction and
  model output is not; a drift estimate measured only on instructions may not transfer to the
  register it is meant to correct.

If the two families disagree, that is itself a finding and it gets reported, not averaged away.

**Control texts ride as pseudo-``Response`` records -- a documented CONSTRUCT CHOICE.** The
:class:`~margins.interfaces.Scorer` Protocol is ``score(responses)``, and routing control text
through that exact path is the point: the scorer cannot treat control text differently from model
text, because it is the same code and the same loaded instrument. The cost is that a
:class:`~margins.schemas.Response` field set describes text no model produced, so those fields are
filled with explicit sentinels (:data:`CONTROL_MODEL_ID`, :data:`CONTROL_DECODING_KEY`, ``seed=0``,
``temperature=0.0``, ``quant="none"``) that could never be mistaken for a real generation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from margins.config import QuantSetting
from margins.lattice.build import DeclaredLattice
from margins.schemas import IdentityCell, ItemScore, Prompt, Response
from margins.scoring.config import ControlFamily, ScoringConfig

__all__ = [
    "CONTROL_BENCHMARK",
    "CONTROL_DECODING_KEY",
    "CONTROL_MODEL_ID",
    "ControlText",
    "DriftRow",
    "build_control_texts",
    "control_responses",
    "drift_rows",
    "interaction_contrast",
    "render_control_prompts",
]

#: Sentinel ``model_id`` for control text. Deliberately not a real model name and deliberately not
#: empty: it must be impossible to mistake a control row for a generation, and impossible for a
#: groupby on ``model_id`` to silently merge the two.
CONTROL_MODEL_ID: Final[str] = "drift_control::template"

#: Sentinel ``decoding_key``. No decoding happened; the text was rendered, not generated.
CONTROL_DECODING_KEY: Final[str] = "control_v1"

#: Benchmark label for control rows. The control is matched to the ``open_gen`` set and is
#: standardized within it, so it shares the source key -- ``condition`` is what separates them.
CONTROL_BENCHMARK: Final[str] = "open_gen_templates"

#: Quantization on a control row. Nothing was quantized because nothing was loaded.
_CONTROL_QUANT: QuantSetting = "none"


class ControlText(BaseModel):
    """One known-neutral text, with the cell it carries and the family it came from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    control_id: str = Field(min_length=1)
    family: ControlFamily
    cell_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_prompt_id: str = Field(
        min_length=1, description="The Prompt this text came from, for traceability."
    )


class DriftRow(BaseModel):
    """One scored control text, keyed so P7 can fit ``A + B + A:B`` on the scorer's own series.

    The axis columns live **here and not on** :class:`~margins.schemas.ItemScore`. Invariant 2 keeps
    source-specific fields off shared records, and ``tests/test_schema_field_freeze.py`` explicitly
    blocklists ``descriptor``/``bucket``. They are resolved from the lattice rather than parsed out
    of ``cell_id``, which ``margins.lattice.build`` documents as a label and not a structure.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    score_id: str = Field(min_length=1)
    response_id: str = Field(min_length=1)
    scorer_id: str = Field(min_length=1)
    harm_type: str = Field(min_length=1)
    condition: str = Field(min_length=1)
    control_family: str = Field(min_length=1)

    cell_id: str = Field(min_length=1)
    order: int = Field(ge=1, le=2)
    axis_a: str = Field(min_length=1)
    value_a: str = Field(min_length=1)
    axis_b: str | None = None
    value_b: str | None = None

    benchmark: str = Field(min_length=1)
    raw: float
    standardized: float


def _control_id(family: str, prompt_id: str) -> str:
    """A deterministic id for a control text. Digest so distinct prompts cannot collide."""
    digest = hashlib.sha256(f"{family}\x1f{prompt_id}".encode()).hexdigest()[:12]
    return f"{family}::{prompt_id}::{digest}"


def build_control_texts(
    config: ScoringConfig,
    prompts: Sequence[Prompt],
    *,
    lattice: DeclaredLattice,
    scoring_config_dir: Path | None = None,
) -> list[ControlText]:
    """Build the known-neutral text set, one entry per (family, cell, frame).

    ``prompt_echo`` reuses ``prompts`` verbatim. ``neutral_declarative`` renders the control frames
    over the same cells through the ordinary :class:`~margins.prompts.render.TemplateRenderer`, so
    the leakage guard, the article rule and P3's ``compose_rank`` order all apply unchanged.

    Raises:
        ValueError: if a requested family produces nothing, which would leave the drift estimate
            silently resting on one family.
    """
    texts: list[ControlText] = []

    if "prompt_echo" in config.drift.families:
        for prompt in prompts:
            if prompt.cell_id is None:
                continue
            texts.append(
                ControlText(
                    control_id=_control_id("prompt_echo", prompt.prompt_id),
                    family="prompt_echo",
                    cell_id=prompt.cell_id,
                    text=prompt.rendered_text,
                    source_prompt_id=prompt.prompt_id,
                )
            )
        if not texts:
            raise ValueError(
                "the prompt_echo control family produced no texts: every prompt given had "
                "cell_id=None. A drift control with no identity variation measures nothing."
            )

    if "neutral_declarative" in config.drift.families:
        control_prompts = render_control_prompts(
            config, lattice=lattice, scoring_config_dir=scoring_config_dir
        )
        before = len(texts)
        for prompt in control_prompts:
            if prompt.cell_id is None:  # pragma: no cover - the renderer always sets it
                continue
            texts.append(
                ControlText(
                    control_id=_control_id("neutral_declarative", prompt.prompt_id),
                    family="neutral_declarative",
                    cell_id=prompt.cell_id,
                    text=prompt.rendered_text,
                    source_prompt_id=prompt.prompt_id,
                )
            )
        if len(texts) == before:
            raise ValueError(
                "the neutral_declarative control family produced no texts. Check "
                f"configs/scoring/{config.drift.control_prompt_set}.yaml and its frames."
            )

    return texts


def render_control_prompts(
    config: ScoringConfig,
    *,
    lattice: DeclaredLattice,
    scoring_config_dir: Path | None = None,
) -> list[Prompt]:
    """Render the neutral declarative frames over every lattice cell.

    Imports the P4a machinery rather than reimplementing it. That reuse is what
    ``margins/prompts/render.py``'s ``TODO(P6)`` anticipated, and it is the entire reason the
    control is "matched by construction" rather than matched by careful copying.
    """
    from margins.prompts.config import load_prompt_config, load_template_files
    from margins.prompts.render import TemplateRenderer, template_items
    from margins.scoring.config import default_scoring_config_dir

    base = (scoring_config_dir or default_scoring_config_dir()).resolve()
    prompt_config = load_prompt_config(config.drift.control_prompt_set, config_dir=base)
    files = load_template_files(prompt_config, base / config.drift.control_template_dir)
    items = template_items(prompt_config, files)
    renderer = TemplateRenderer(prompt_config, lattice)
    return renderer.render(items, lattice.cells())


def control_responses(
    texts: Sequence[ControlText], *, decoding_hash: str
) -> tuple[list[Response], dict[str, Prompt]]:
    """Wrap control texts as pseudo-:class:`~margins.schemas.Response` records and their prompts.

    See the module docstring for why this shape rather than a separate scoring path: it routes the
    control through the *same* ``Scorer.score(responses)`` code as model output, which is the only
    way to guarantee the two conditions are measured by the same instrument.

    Returns ``(responses, prompts_by_id)`` -- the prompts are what a
    :class:`~margins.scoring.base.ResponseScorer` reads ``cell_id`` and ``benchmark`` from.
    """
    responses: list[Response] = []
    prompts: dict[str, Prompt] = {}

    for control in texts:
        prompt = Prompt(
            prompt_id=control.control_id,
            item_id=f"drift_control/{control.family}",
            cell_id=control.cell_id,
            rendered_text=control.text,
            task_type="open_gen",
            template_version=control.family,
            decoding_key=CONTROL_DECODING_KEY,
        )
        prompts[prompt.prompt_id] = prompt
        responses.append(
            Response(
                response_id=control.control_id,
                prompt_id=control.control_id,
                model_id=CONTROL_MODEL_ID,
                model_revision=CONTROL_DECODING_KEY,
                seed=0,
                temperature=0.0,
                decoding_key=CONTROL_DECODING_KEY,
                decoding_hash=decoding_hash,
                quant=_CONTROL_QUANT,
                text=control.text,
                token_logprobs=None,
            )
        )

    return responses, prompts


def drift_rows(
    scores: Sequence[ItemScore],
    texts: Sequence[ControlText],
    cells: Sequence[IdentityCell],
) -> list[DriftRow]:
    """Join scored control text to its cell's axis values -- the artifact P7 fits on.

    ``cells`` is taken rather than parsed out of ``cell_id`` for the same reason
    :meth:`margins.interfaces.Metric.compute` takes it: a cell id is a label, and
    ``EffectEstimate.axis_pair`` cannot be reconstructed from one.
    """
    by_cell = {cell.cell_id: cell for cell in cells}
    family_by_id = {text.control_id: text.family for text in texts}

    rows: list[DriftRow] = []
    for score in scores:
        family = family_by_id.get(score.response_id)
        if family is None:
            continue
        if score.cell_id is None:
            continue
        cell = by_cell.get(score.cell_id)
        if cell is None:
            raise KeyError(
                f"control score references cell {score.cell_id!r}, which is not in the lattice "
                "passed in. The drift artifact must resolve every cell to its axis values, or P7 "
                "cannot build a design matrix from it."
            )
        axes = cell.axes
        rows.append(
            DriftRow(
                score_id=score.score_id,
                response_id=score.response_id,
                scorer_id=score.scorer_id,
                harm_type=score.harm_type,
                condition=score.condition,
                control_family=family,
                cell_id=cell.cell_id,
                order=cell.order,
                axis_a=axes[0][0],
                value_a=axes[0][1],
                axis_b=axes[1][0] if len(axes) > 1 else None,
                value_b=axes[1][1] if len(axes) > 1 else None,
                benchmark=score.benchmark,
                raw=score.raw,
                standardized=score.standardized,
            )
        )
    return rows


def interaction_contrast(
    rows: Sequence[DriftRow],
    *,
    axis_a: str,
    value_a: str,
    axis_b: str,
    value_b: str,
    field: str = "standardized",
) -> float:
    """The 2x2 cell-mean interaction contrast for one axis-value pair.

    ``mean(11) - mean(10) - mean(01) + mean(00)``, where ``11`` is the order-2 cell carrying both
    values, ``10``/``01`` are the two order-1 cells, and ``00`` is the grand mean of the remaining
    order-1 cells on those axes (the "neither value" baseline).

    **This is deliberately NOT P7's estimator.** No link function, no cluster-robust standard error,
    no confidence interval, no significance -- those belong to the metric and stats layers, and
    building them here would be building P7. It exists so P6 can *demonstrate* that the artifact it
    emits carries recoverable interaction signal: the drift tests use it to show that a synthetic
    identity-sensitive scorer produces a non-zero contrast while a neutral one produces ~0.

    Returns 0.0 when any of the four groups is empty, which is the honest answer for a contrast that
    is not estimable rather than a NaN that would propagate.
    """
    import numpy as np

    def mean_of(selector: object) -> float | None:
        values = [
            float(getattr(row, field))
            for row in rows
            if selector(row)  # type: ignore[operator]
        ]
        return float(np.mean(values)) if values else None

    both = mean_of(
        lambda r: r.order == 2
        and {(r.axis_a, r.value_a), (r.axis_b, r.value_b)} == {(axis_a, value_a), (axis_b, value_b)}
    )
    only_a = mean_of(lambda r: r.order == 1 and r.axis_a == axis_a and r.value_a == value_a)
    only_b = mean_of(lambda r: r.order == 1 and r.axis_a == axis_b and r.value_a == value_b)
    neither = mean_of(
        lambda r: r.order == 1
        and (
            (r.axis_a == axis_a and r.value_a != value_a)
            or (r.axis_a == axis_b and r.value_a != value_b)
        )
    )

    if both is None or only_a is None or only_b is None or neither is None:
        return 0.0
    return both - only_a - only_b + neither
