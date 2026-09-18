"""Scoring, triangulation, the identity-drift control, and human calibration (P6).

This is the stage that turns :class:`~margins.schemas.Response` records into
:class:`~margins.schemas.ItemScore` rows -- and the stage that decides whether the primary estimand
is measurable at all. Three checks live here, and they answer three different questions:

* **Triangulation** (:mod:`~margins.scoring.triangulate`) -- do two independent scorers behave
  alike? Reported as Krippendorff's alpha and Cohen's kappa.
* **The identity-drift control** (:mod:`~margins.scoring.drift`) -- does a scorer respond to
  identity terms *independently of harm*, non-additively, and therefore manufacture an interaction
  from text that is constant across cells? This is the construct-validity centrepiece, and the one
  that agreement cannot substitute for.
* **Human calibration** (:mod:`~margins.scoring.calibration`) -- does the scorer find what a person
  calls harm? Scaffolding and metrics here; the labeling is the human's.

A harm type is load-bearing only if all three hold. Any one of them failing downgrades it to
descriptive, and P7/P9 enforce that consequence -- P6's job is to emit the numbers that make it
checkable.

**Import layering**, the same discipline :mod:`margins.inference` uses: ``config``, ``ids``,
``base``, ``standardize``, ``drift``, ``calibration``, ``export`` and ``manifest`` import no model
stack at all, so the whole numeric core runs on the torch-free environment and in the core CI job on
both Windows and Linux. Only :mod:`~margins.scoring.hf_classifier` touches ``transformers``, and it
imports lazily inside the methods that need it. :mod:`~margins.scoring.lexicon` imports
``vaderSentiment`` lazily for the same reason -- constructing the analyzer reads a bundled lexicon
off disk, which is wasted work for a caller that only wants to inspect the roster.

**Two rules are enforced in code rather than in prose**, because both are easy to violate silently:
a harm type measured by fewer than two scorers, or by two scorers sharing a mechanism, makes
:class:`~margins.scoring.config.ScoringConfig` **fail to load**; and
:func:`~margins.scoring.standardize.standardize_within_benchmark` requires a benchmark key and
refuses rows from any other source.
"""

from __future__ import annotations

from margins.scoring.base import ResponseScorer, TextScorer, scorer_from_spec
from margins.scoring.calibration import (
    CalibrationMetric,
    CalibrationUnit,
    HumanLabel,
    calibration_metrics,
    read_labels,
    render_calibration_report,
    sample_for_calibration,
    write_calibration_sample,
)
from margins.scoring.config import (
    DEFAULT_SCORING,
    MIN_SCORERS_PER_HARM,
    SCORER_INPUT_PREFIX,
    SCORING_INPUT_KEY,
    ControlFamily,
    LexiconPin,
    Mechanism,
    ModelPin,
    ScorerSpec,
    ScoringConfig,
    SignProbe,
    load_scoring_config,
)
from margins.scoring.drift import (
    CONTROL_BENCHMARK,
    CONTROL_DECODING_KEY,
    CONTROL_MODEL_ID,
    ControlText,
    DriftRow,
    build_control_texts,
    control_responses,
    drift_rows,
    interaction_contrast,
    render_control_prompts,
)
from margins.scoring.export import (
    DEFAULT_SCORING_DIR,
    export_scoring,
    read_drift_rows,
    read_scores,
    write_agreement,
    write_drift_rows,
    write_scores,
)
from margins.scoring.ids import SCORE_ID_DIGEST_LENGTH, make_score_id
from margins.scoring.lexicon import ValenceScorer, WordlistScorer, build_lexicon_scorer
from margins.scoring.manifest import scoring_inputs, scoring_record
from margins.scoring.standardize import (
    SignContractError,
    StandardizationReport,
    signs_from_specs,
    standardize_within_benchmark,
    verify_sign,
)
from margins.scoring.triangulate import (
    AgreementRecord,
    Triangulator,
    build_triangulator,
)

__all__ = [
    "CONTROL_BENCHMARK",
    "CONTROL_DECODING_KEY",
    "CONTROL_MODEL_ID",
    "DEFAULT_SCORING",
    "DEFAULT_SCORING_DIR",
    "MIN_SCORERS_PER_HARM",
    "SCORER_INPUT_PREFIX",
    "SCORE_ID_DIGEST_LENGTH",
    "SCORING_INPUT_KEY",
    "AgreementRecord",
    "CalibrationMetric",
    "CalibrationUnit",
    "ControlFamily",
    "ControlText",
    "DriftRow",
    "HumanLabel",
    "LexiconPin",
    "Mechanism",
    "ModelPin",
    "ResponseScorer",
    "ScorerSpec",
    "ScoringConfig",
    "SignContractError",
    "SignProbe",
    "StandardizationReport",
    "TextScorer",
    "Triangulator",
    "ValenceScorer",
    "WordlistScorer",
    "build_control_texts",
    "build_lexicon_scorer",
    "build_triangulator",
    "calibration_metrics",
    "control_responses",
    "drift_rows",
    "export_scoring",
    "interaction_contrast",
    "load_scoring_config",
    "make_score_id",
    "read_drift_rows",
    "read_labels",
    "read_scores",
    "render_calibration_report",
    "render_control_prompts",
    "sample_for_calibration",
    "scorer_from_spec",
    "scoring_inputs",
    "scoring_record",
    "signs_from_specs",
    "standardize_within_benchmark",
    "verify_sign",
    "write_agreement",
    "write_calibration_sample",
    "write_drift_rows",
    "write_scores",
]
