"""Synthetic stand-in rows, so every P10 figure renders and is tested before P11 (P10).

**These are not results and no number here may be cited.** The confirmatory run is P11; P10's job
is to build the figures and freeze their code so that the real run drops into them unchanged. A
figure that cannot be rendered cannot be reviewed, and a figure first written in the week the
results land is a figure written under deadline pressure against numbers its author has already
seen -- which is how an honest analysis becomes a persuasive one.

Three rules keep the stand-ins from ever being mistaken for measurements:

* every row carries ``data_status="fixture"``, machine-readably, in the table itself;
* every figure drawn from such a table is banner-stamped by
  :func:`margins.viz.trace.stamp_data_status`;
* **no fixture value is registered in ``results/paper_numbers.json``.** That file holds numbers
  the manuscript may cite, and a fabricated entry in it would defeat the one mechanism the
  project has against invented statistics.

Everything *structural* is real: the axis pairs, their levels, the confirmatory/exploratory split,
the harm types, the scorer ids, the model pins, the frame names, and the extractive paradigm's
measured inability to populate an order-2 grid. Only the estimates are drawn from a seeded RNG.
The shapes are chosen to exercise the paths a figure must get right -- one harm type downgraded by
the drift rule, one exploratory pair carrying no q, and a red-team arm that is descriptive by
pre-registered design rather than by accident.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Final

import numpy as np

from margins.lattice.build import DeclaredLattice
from margins.metrics.design import PairDesign, interaction_term, main_term, pair_designs
from margins.stats.family import free_interaction_terms

__all__ = [
    "CONFIRMATORY_MODELS",
    "DRIFT_THRESHOLD",
    "FIXTURE_ALPHA",
    "FIXTURE_SEED",
    "GENERATION_BENCHMARK",
    "HARM_TYPES",
    "REDTEAM_BENCHMARK",
    "SCORERS",
    "build_fixture_tables",
]

#: Real names throughout: the pinned confirmatory models, the P6 roster, the P2 source.
CONFIRMATORY_MODELS: Final[tuple[str, ...]] = (
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
)
GENERATION_BENCHMARK: Final[str] = "open_gen_templates"
REDTEAM_BENCHMARK: Final[str] = "redteam"
EXTRACTIVE_BENCHMARK: Final[str] = "crows_pairs"

HARM_TYPES: Final[tuple[str, ...]] = ("regard", "toxicity")
#: harm type -> (neural scorer, lexicon scorer). The P6 roster, unchanged.
SCORERS: Final[Mapping[str, tuple[str, str]]] = {
    "regard": ("regard_classifier", "vader_valence"),
    "toxicity": ("toxic_bert", "ldnoobw_lexicon"),
}
CONTROL_FAMILIES: Final[tuple[str, ...]] = ("prompt_echo", "neutral_declarative")

#: The pre-registered drift threshold from ``configs/stats/confirmatory_v1.yaml``. Real; it is a
#: threshold, not an estimate.
DRIFT_THRESHOLD: Final[float] = 0.5
FIXTURE_ALPHA: Final[float] = 0.05
FIXTURE_SEED: Final[int] = 20260921

#: The harm type the fixture downgrades. Chosen to match P6's real early signal: ``toxic_bert``
#: showed a large own-interaction on text containing no harm at all.
DOWNGRADED_HARM: Final[str] = "toxicity"

#: The pair the fixture plants a clearly emergent cell in, and the one the single-pair panels
#: show. It is the lattice's primary non-gender x non-gender pair and its largest grid (4x4) --
#: the pair novelty claim 5 rests on, and the one rho was measured on.
FLAGSHIP_PAIR: Final[tuple[str, str]] = ("socioeconomic_class", "race_ethnicity")

#: The P4a frames, with the one the phrase review flagged as highest-risk first.
FRAMES: Final[tuple[str, ...]] = (
    "first_person",
    "advice",
    "biography",
    "conversation",
    "daily_life",
    "describe",
    "first_meeting",
    "questions",
    "scene",
    "sentence_completion",
    "short_story",
    "someone_who_is",
    "three_words",
)

_FIXTURE: Final[dict[str, str]] = {"data_status": "fixture"}


def _rng(*parts: object) -> np.random.Generator:
    """A generator keyed by the row group, so a row's value never depends on iteration order.

    Keyed through blake2b rather than ``hash()``: string hashing is salted per process, so the
    obvious version would have produced a different fixture table on every run and quietly broken
    the byte-determinism the figures are tested for.
    """
    key = "\x1f".join(("margins-fixture", str(FIXTURE_SEED), *(str(part) for part in parts)))
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=4).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


def _interval(estimate: float, se: float) -> tuple[float, float]:
    return (round(estimate - 1.96 * se, 6), round(estimate + 1.96 * se, 6))


def _bh(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg, so the fixture's q column is at least internally consistent."""
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    q = [0.0] * n
    running = 1.0
    for rank, index in reversed(list(enumerate(order, start=1))):
        running = min(running, p_values[index] * n / rank)
        q[index] = round(running, 6)
    return q


def _wald_p(estimate: float, se: float) -> float:
    from scipy import stats

    # df = item clusters - 1, at the shipped 27 frames x 4 cells' worth of clusters.
    return float(2.0 * stats.t.sf(abs(estimate) / se, df=107))


# --------------------------------------------------------------------------------------
# emergence_effects
# --------------------------------------------------------------------------------------


def _arms() -> tuple[tuple[str, str, bool], ...]:
    """``(benchmark, model, powered)`` for each arm the figures draw.

    The red-team arm is ``powered=False`` by the 21 Sep 2026 decision: a powered red-team arm
    costs the same ~95 GPU-h as the main sweep, so it runs as a descriptive subset, stays out of
    ``family_benchmarks``, and carries no p and no q. That is a budget-driven scope choice and
    never evidence that red-team effects are small.
    """
    return (
        (GENERATION_BENCHMARK, CONFIRMATORY_MODELS[0], True),
        (GENERATION_BENCHMARK, CONFIRMATORY_MODELS[1], True),
        (REDTEAM_BENCHMARK, CONFIRMATORY_MODELS[0], False),
    )


def _double_centre(table: np.ndarray) -> np.ndarray:
    """Project onto the interaction subspace: rows and columns sum to zero.

    The same operation :func:`margins.metrics.design.additive_decomposition` performs, which is
    why a fixture built this way is a legal gamma table rather than an arbitrary matrix.
    """
    centred = (
        table - table.mean(axis=1, keepdims=True) - table.mean(axis=0, keepdims=True) + table.mean()
    )
    return np.asarray(centred, dtype=float)


def _gamma_table(design: PairDesign, rng: np.random.Generator, scale: float) -> np.ndarray:
    """A sum-to-zero interaction table: rows and columns sum to zero, as the estimand requires."""
    shape = (len(design.levels_a), len(design.levels_b))
    return _double_centre(rng.normal(0.0, scale, size=shape))


def _planted(shape: tuple[int, int], magnitude: float) -> np.ndarray:
    """A sum-to-zero table whose ``(0, 0)`` cell is exactly ``magnitude``.

    A single cell raised in isolation is not an interaction table -- its rows and columns would
    not sum to zero and no estimator could return it. Projecting the indicator onto the
    interaction subspace and rescaling gives the real shape of "one cell departs from additivity".
    """
    indicator = np.zeros(shape)
    indicator[0, 0] = 1.0
    projected = _double_centre(indicator)
    return np.asarray(projected * (magnitude / projected[0, 0]), dtype=float)


def _emergence_rows(
    designs: Sequence[PairDesign],
    confirmatory: frozenset[tuple[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for benchmark, model, powered in _arms():
        for harm in HARM_TYPES:
            descriptive_only = harm == DOWNGRADED_HARM
            family: list[dict[str, Any]] = []
            for design in designs:
                is_confirmatory = design.axis_pair in confirmatory
                # P8 restricts red-team targeting to the confirmatory pairs as a Literal, so a
                # red-team row for an exploratory pair could not exist.
                if benchmark == REDTEAM_BENCHMARK and not is_confirmatory:
                    continue
                rng = _rng(benchmark, model, harm, design.axis_pair)
                # A planted, clearly emergent cell in one confirmatory pair per harm type, so
                # the figure has something to mark and the marking can be checked.
                scale = 0.22 if is_confirmatory else 0.12
                gamma = _gamma_table(design, rng, scale)
                if is_confirmatory and design.axis_pair == FLAGSHIP_PAIR:
                    # Comfortably above the pre-registered 0.5 SD minimum detectable effect, so
                    # the figure has an unambiguous positive to mark and the marking is testable.
                    gamma = gamma + _planted(gamma.shape, 0.72 if harm == "regard" else 0.58)
                free = set(free_interaction_terms(design))
                se_scale = 0.09 if benchmark == GENERATION_BENCHMARK else 0.16

                for i, value_a in enumerate(design.levels_a):
                    for j, value_b in enumerate(design.levels_b):
                        estimate = round(float(gamma[i, j]), 6)
                        se = round(float(rng.uniform(0.7, 1.3) * se_scale), 6)
                        term = interaction_term(design.axis_a, value_a, design.axis_b, value_b)
                        low, high = _interval(estimate, se)
                        member = is_confirmatory and term in free and powered
                        row = {
                            "benchmark": benchmark,
                            "model_id": model,
                            "harm_type": harm,
                            "axis_a": design.axis_a,
                            "axis_b": design.axis_b,
                            "term": term,
                            "kind": "interaction",
                            "value_a": value_a,
                            "value_b": value_b,
                            "estimate": estimate,
                            "se": se,
                            "ci_low": low,
                            "ci_high": high,
                            "p": _wald_p(estimate, se) if member else None,
                            "q": None,
                            "family_size": None,
                            "correction": None,
                            "powered": powered,
                            "status": "",
                            "claimable": False,
                            "family_member": member,
                            **_FIXTURE,
                        }
                        rows.append(row)
                        if member:
                            family.append(row)

                # Main effects: B1's baseline, emitted so the figure can show that a high level
                # is not an interaction. Never in the family.
                for axis, levels, kind in (
                    (design.axis_a, design.levels_a, "main_a"),
                    (design.axis_b, design.levels_b, "main_b"),
                ):
                    raw = rng.normal(0.0, 0.3, size=len(levels))
                    for value, estimate in zip(levels, raw - raw.mean(), strict=True):
                        se = round(float(rng.uniform(0.7, 1.3) * se_scale), 6)
                        low, high = _interval(float(estimate), se)
                        rows.append(
                            {
                                "benchmark": benchmark,
                                "model_id": model,
                                "harm_type": harm,
                                "axis_a": design.axis_a,
                                "axis_b": design.axis_b,
                                "term": main_term(axis, value),
                                "kind": kind,
                                "value_a": value if kind == "main_a" else None,
                                "value_b": value if kind == "main_b" else None,
                                "estimate": round(float(estimate), 6),
                                "se": se,
                                "ci_low": low,
                                "ci_high": high,
                                "p": None,
                                "q": None,
                                "family_size": None,
                                "correction": None,
                                "powered": powered,
                                "status": "",
                                "claimable": False,
                                "family_member": False,
                                **_FIXTURE,
                            }
                        )

            # The family is corrected together, across pairs, exactly as section 8 requires.
            if family:
                for row, q in zip(family, _bh([float(r["p"]) for r in family]), strict=True):
                    row["q"] = q
                    # The three move together on the record, so they move together here.
                    row["family_size"] = len(family)
                    row["correction"] = "fdr_bh"

            _stamp_status(rows, benchmark, model, harm, descriptive_only, confirmatory)
    return rows


def _stamp_status(
    rows: list[dict[str, Any]],
    benchmark: str,
    model: str,
    harm: str,
    descriptive_only: bool,
    confirmatory: frozenset[tuple[str, str]],
) -> None:
    """Apply the gate's precedence, so the table's status column matches what P9 would record."""
    for row in rows:
        if (row["benchmark"], row["model_id"], row["harm_type"]) != (benchmark, model, harm):
            continue
        if (row["axis_a"], row["axis_b"]) not in confirmatory:
            status = "exploratory"
        elif descriptive_only:
            status = "descriptive_only"
        elif not row["powered"]:
            status = "underpowered"
        elif row["q"] is None:
            status = "uncorrected"
        else:
            status = "claimable"
        row["status"] = status
        row["claimable"] = status == "claimable"


# --------------------------------------------------------------------------------------
# the drift tables
# --------------------------------------------------------------------------------------


def _drift_rows(designs: Sequence[PairDesign]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for benchmark, model, _ in _arms():
        for harm in HARM_TYPES:
            neural, lexicon = SCORERS[harm]
            for scorer in (neural, lexicon):
                for design in designs:
                    for family in CONTROL_FAMILIES:
                        rng = _rng(
                            "drift", benchmark, model, harm, scorer, design.axis_pair, family
                        )
                        model_rms = round(float(rng.uniform(0.25, 0.55)), 6)
                        # A lexicon scorer has no identity sensitivity to speak of; the neural
                        # toxicity scorer does, which is what downgrades that harm type.
                        if scorer == lexicon:
                            drift = float(rng.uniform(0.0, 0.02))
                        elif harm == DOWNGRADED_HARM:
                            drift = float(rng.uniform(0.55, 0.95)) * model_rms
                        else:
                            drift = float(rng.uniform(0.05, 0.30)) * model_rms
                        drift_rms = round(drift, 6)
                        rows.append(
                            {
                                "benchmark": benchmark,
                                "model_id": model,
                                "harm_type": harm,
                                "scorer_id": scorer,
                                "axis_a": design.axis_a,
                                "axis_b": design.axis_b,
                                "control_family": family,
                                "model_rms": model_rms,
                                "drift_rms": drift_rms,
                                "ratio": round(drift_rms / model_rms, 6),
                                "n_model_obs": 1728,
                                "n_drift_obs": 432,
                                **_FIXTURE,
                            }
                        )
    return rows


def _descriptive_only_rows(drift: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = sorted({(r["benchmark"], r["model_id"], r["harm_type"]) for r in drift})
    for benchmark, model, harm in keys:
        group = [
            r
            for r in drift
            if (r["benchmark"], r["model_id"], r["harm_type"]) == (benchmark, model, harm)
        ]
        over = [r for r in group if r["ratio"] >= DRIFT_THRESHOLD]
        worst = max(group, key=lambda r: float(r["ratio"]))
        rows.append(
            {
                "benchmark": benchmark,
                "model_id": model,
                "harm_type": harm,
                "descriptive_only": bool(over),
                "threshold": DRIFT_THRESHOLD,
                "max_ratio": round(float(worst["ratio"]), 6),
                "worst_scorer": worst["scorer_id"],
                "n_comparisons": len(group),
                "n_over_threshold": len(over),
                **_FIXTURE,
            }
        )
    return rows


# --------------------------------------------------------------------------------------
# estimability, single-axis cells, sensitivity, frames
# --------------------------------------------------------------------------------------

#: The REAL reason the extractive paradigm reports every confirmatory pair not estimable. P2
#: measured it: 3 order-2 items across 1,352 kept rows cannot fill a 3x4 or 4x4 grid.
_CROWS_REASON: Final[str] = "12 empty cells: no item attests them"


def _fit_report_rows(
    designs: Sequence[PairDesign], confirmatory: frozenset[tuple[str, str]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for benchmark, model, powered in _arms():
        for harm in HARM_TYPES:
            for design in designs:
                if benchmark == REDTEAM_BENCHMARK and design.axis_pair not in confirmatory:
                    continue
                rng = _rng("fit", benchmark, model, harm, design.axis_pair)
                rows.append(
                    {
                        "benchmark": benchmark,
                        "model_id": model,
                        "harm_type": harm,
                        "axis_a": design.axis_a,
                        "axis_b": design.axis_b,
                        "estimable": True,
                        "reason": None,
                        "n_obs": design.n_cells * 108,
                        "n_items": design.n_cells * 27,
                        "n_cells": design.n_cells,
                        "rms_interaction": round(float(rng.uniform(0.15, 0.45)), 6),
                        "powered": powered,
                        **_FIXTURE,
                    }
                )
    # The extractive paradigm, and its measured emptiness. Not a zero: a reason.
    for harm in HARM_TYPES:
        for design in designs:
            if design.axis_pair not in confirmatory:
                continue
            empty = design.n_cells - (1 if design.axis_pair[0] == "age" else 0)
            rows.append(
                {
                    "benchmark": EXTRACTIVE_BENCHMARK,
                    "model_id": CONFIRMATORY_MODELS[0],
                    "harm_type": "stereotype_preference" if harm == "regard" else harm,
                    "axis_a": design.axis_a,
                    "axis_b": design.axis_b,
                    "estimable": False,
                    "reason": _CROWS_REASON.replace("12", str(empty)),
                    "n_obs": 3,
                    "n_items": 3,
                    "n_cells": design.n_cells,
                    "rms_interaction": None,
                    "powered": False,
                    **_FIXTURE,
                }
            )
    return rows


def _single_axis_rows(lattice: DeclaredLattice) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    order_one = [cell for cell in lattice.cells() if cell.order == 1]
    for benchmark, model, n, harm_types in (
        (GENERATION_BENCHMARK, CONFIRMATORY_MODELS[0], 108, HARM_TYPES),
        (EXTRACTIVE_BENCHMARK, CONFIRMATORY_MODELS[0], 11, ("stereotype_preference",)),
    ):
        for harm in harm_types:
            for cell in order_one:
                axis, value = cell.axes[0]
                rng = _rng("cell", benchmark, model, harm, cell.cell_id)
                mean = round(float(rng.normal(0.0, 0.35)), 6)
                half = round(float(rng.uniform(0.12, 0.30)) * (1.0 if n > 50 else 2.2), 6)
                rows.append(
                    {
                        "benchmark": benchmark,
                        "model_id": model,
                        "harm_type": harm,
                        "cell_id": cell.cell_id,
                        "axis": axis,
                        "value": value,
                        "n": n,
                        "mean_standardized": mean,
                        "ci_low": round(mean - half, 6),
                        "ci_high": round(mean + half, 6),
                        "powered": n >= 108,
                        **_FIXTURE,
                    }
                )
    return rows


def _sensitivity_rows(designs: Sequence[PairDesign]) -> list[dict[str, Any]]:
    """E4: the headline term re-estimated across the temperature grid and seed groups."""
    design = next(d for d in designs if d.axis_pair == FLAGSHIP_PAIR)
    term = free_interaction_terms(design)[0]
    rows: list[dict[str, Any]] = []
    for model in CONFIRMATORY_MODELS:
        for harm in HARM_TYPES:
            for temperature in (0.0, 0.7, 1.0):
                for seed_group in (1, 2, 3):
                    rng = _rng("sens", model, harm, temperature, seed_group)
                    centre = 0.62 if harm == "regard" else 0.44
                    spread = 0.04 + 0.14 * temperature
                    estimate = round(float(centre + rng.normal(0.0, spread)), 6)
                    half = round(float(0.10 + 0.09 * temperature), 6)
                    rows.append(
                        {
                            "benchmark": GENERATION_BENCHMARK,
                            "model_id": model,
                            "harm_type": harm,
                            "axis_a": design.axis_a,
                            "axis_b": design.axis_b,
                            "term": term,
                            "temperature": temperature,
                            "seed_group": seed_group,
                            "estimate": estimate,
                            "ci_low": round(estimate - half, 6),
                            "ci_high": round(estimate + half, 6),
                            **_FIXTURE,
                        }
                    )
    return rows


def _frame_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in CONFIRMATORY_MODELS:
        for harm in HARM_TYPES:
            for frame in FRAMES:
                rng = _rng("frame", model, harm, frame)
                # `first_person` is the frame the P4a review flagged: asking a model to speak as
                # an identity is the highest-risk construct in the set, and the figure exists to
                # show whether it behaves differently from the rest.
                centre = 0.45 if frame == "first_person" else float(rng.normal(0.0, 0.18))
                half = round(float(rng.uniform(0.08, 0.16)), 6)
                mean = round(float(centre + rng.normal(0.0, 0.05)), 6)
                rows.append(
                    {
                        "benchmark": GENERATION_BENCHMARK,
                        "model_id": model,
                        "harm_type": harm,
                        "frame": frame,
                        "n_items": 110 * 2,
                        "mean_standardized": mean,
                        "ci_low": round(mean - half, 6),
                        "ci_high": round(mean + half, 6),
                        **_FIXTURE,
                    }
                )
    return rows


# --------------------------------------------------------------------------------------


def build_fixture_tables(lattice: DeclaredLattice) -> dict[str, list[dict[str, Any]]]:
    """Every fixture table, keyed by table name. Deterministic for a fixed lattice."""
    designs = pair_designs(lattice.cells())
    confirmatory = frozenset(lattice.config.confirmatory_pairs)
    drift = _drift_rows(designs)
    return {
        "emergence_effects": _emergence_rows(designs, confirmatory),
        "drift_side_by_side": drift,
        "descriptive_only": _descriptive_only_rows(drift),
        "fit_reports": _fit_report_rows(designs, confirmatory),
        "single_axis_cells": _single_axis_rows(lattice),
        "sensitivity": _sensitivity_rows(designs),
        "frame_breakdown": _frame_rows(),
    }


def iter_fixture_tables(
    lattice: DeclaredLattice,
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    yield from sorted(build_fixture_tables(lattice).items())
