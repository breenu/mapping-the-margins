"""Structural Protocols for every pipeline stage.

Why Protocols rather than base classes: modules must be swappable and fakeable without
inheriting from anything, so a test double is a plain object that happens to have the right
shape. The P1 walking-skeleton test is exactly that — no-op fakes satisfying every Protocol,
composed end to end, proving the contracts fit together before any real logic exists.

Every stage depends only on the records in :mod:`margins.schemas`, never on another stage's
internals.

Note on ``runtime_checkable``: ``isinstance`` against these Protocols checks only that the
required attributes *exist*, never their signatures or types. It is a smoke test, not a proof
of conformance. ``issubclass`` raises ``TypeError`` for the Protocols carrying non-method
members (``BenchmarkLoader``, ``InferenceBackend``, ``Scorer``) — that is a Python restriction,
so use ``isinstance``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from margins.schemas import (
    CellStat,
    EffectEstimate,
    IdentityCell,
    Item,
    ItemScore,
    Prompt,
    Response,
)

__all__ = [
    "BenchmarkLoader",
    "InferenceBackend",
    "Lattice",
    "Metric",
    "PromptGenerator",
    "Scorer",
    "StatsEngine",
]


@runtime_checkable
class BenchmarkLoader(Protocol):
    """Ingests an **extractive** source into :class:`~margins.schemas.Item` records.

    Extractive sources only. HolisticBias is a descriptor/value taxonomy, not a benchmark, and
    must never appear as a loader: ``open_gen`` is a *generation* path, not an ingestion one.
    HolisticBias supplies lattice axis values (P3) and template slot fillers (P4) instead.
    """

    @property
    def name(self) -> str:
        """Source label, matching ``Item.benchmark``."""

    def load(self) -> list[Item]:
        """Return the screened items. Deterministic: same source revision, same output."""


@runtime_checkable
class Lattice(Protocol):
    """Produces the identity cells to audit, from a provenance-carrying descriptor source."""

    def cells(self) -> list[IdentityCell]:
        """Return the cell inventory. Every ``(axis, value)`` must resolve in the registry."""


@runtime_checkable
class PromptGenerator(Protocol):
    """Renders items x cells into prompts. The only place identities meet templates.

    **Two incompatible semantics live behind this one signature — both are implemented in P4
    and both must be documented wherever they are used.** ``render(items, cells)`` reads as a
    cross-product, which is simply wrong for extractive sources: you cannot assign "Black
    woman" to a CrowS-Pairs item about religion.

    * ``ExtractiveRenderer`` — the cell is **derived from** the item (``pair_pll``, ``coref``).
      It must never emit a prompt whose cell is unattested in the source item.
    * ``TemplateRenderer`` — the cell is **injected into** a template (``open_gen``, red-team).
      This is the renderer the MVP exercises first, and the only one that can reach arbitrary
      lattice cells — including the non-gender x non-gender cells gate P3 requires.

    Rendering is deterministic: the same items, cells, and template version must reproduce
    byte-identical prompts.
    """

    def render(self, items: Sequence[Item], cells: Sequence[IdentityCell]) -> list[Prompt]:
        """Return the rendered prompt set."""


@runtime_checkable
class InferenceBackend(Protocol):
    """Runs prompts through a model. The only GPU-touching, non-portable stage.

    Implementations must be resumable: a killed run restarts without duplicating or dropping
    responses (gate P5).
    """

    @property
    def model_id(self) -> str:
        """The model identifier recorded on every :class:`~margins.schemas.Response`."""

    def run(self, prompts: Sequence[Prompt]) -> list[Response]:
        """Generate responses, each self-describing via ``decoding_hash`` and ``quant``."""


@runtime_checkable
class Scorer(Protocol):
    """Assigns a harm score to responses.

    **At least two independent scorers per harm type**, with reported agreement and a scorer
    identity-drift control. No harm type may depend on a single classifier; poor-agreement or
    high-drift harm types are downgraded to descriptive. An LLM judge, if used at all, is one
    scorer of two or more with a fixed documented prompt and seed — never ground truth.
    """

    @property
    def scorer_id(self) -> str:
        """Stable identifier, recorded on every score for the agreement analysis."""

    @property
    def harm_type(self) -> str:
        """The harm this scorer measures."""

    def score(self, responses: Sequence[Response]) -> list[ItemScore]:
        """Score responses, standardizing WITHIN source so higher always means more harm."""


@runtime_checkable
class Metric(Protocol):
    """Computes descriptive cell statistics and inferential effect estimates.

    Takes ``cells`` as well as ``scores`` **deliberately**: ``ItemScore.cell_id`` is an opaque
    string, and fitting ``harm ~ A + B + A:B`` needs the axis values behind it — as does
    populating ``EffectEstimate.axis_pair``, which cannot be derived from a cell id. Passing the
    lattice (small, already in memory) resolves ``cell_id -> axes`` without parsing structure
    out of an id string or denormalizing axis values onto every score row. The denormalized keys
    on ``ItemScore`` cover the *grouping* keys; ``cells`` covers the *design matrix*.

    Emitted estimates are **uncorrected**: ``q``, ``family_size`` and ``correction`` stay None
    until :class:`StatsEngine` sets them.
    """

    def compute(
        self, scores: Sequence[ItemScore], cells: Sequence[IdentityCell]
    ) -> tuple[list[CellStat], list[EffectEstimate]]:
        """Return ``(cell_stats, effect_estimates)``."""


@runtime_checkable
class StatsEngine(Protocol):
    """The **only** stage permitted to declare significance (invariant 3).

    Sets ``q``, ``family_size`` and ``correction``; they are set nowhere else, and the report
    layer raises if any is missing. The FDR family is
    ``benchmark x model x axis_pair x term x harm_type`` over the pre-registered confirmatory
    axis pairs — never within a single source. Anything outside that set is exploratory and
    carries no significance claim.
    """

    def correct(self, effects: Sequence[EffectEstimate]) -> list[EffectEstimate]:
        """Return new, corrected estimates. Inputs are frozen and must not be mutated."""
