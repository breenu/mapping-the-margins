"""The emergence estimator and the scorer-drift comparison (P7) -- the intellectual core.

This is the stage that turns :class:`~margins.schemas.ItemScore` rows into the primary estimand.
For each (benchmark, model, harm type, axis pair) it fits

    harm_std ~ A + B + A:B

by ordinary least squares on the standardized harm score (identity link), over the pair's order-2
cells with sum-to-zero coding, with standard errors **cluster-robust by item (prompt)** and
confidence intervals from a **cluster bootstrap over items, stratified by cell**. The coefficients
on ``A:B`` -- and only those -- are the emergent intersectional bias. A high harm *level* at a cell
is described by :class:`~margins.schemas.CellStat` and is not emergence (framing constraint 3).

The same model is fitted on each scorer's own responses to known-neutral text (P6's drift
artifact), per control family, and reported **beside** the model's interaction with a ratio and a
flag. Nothing is point-subtracted; nothing here sets ``p``/``q``/``family_size``/``correction``
(invariant 3). Enforcement of the descriptive-only rule is P9's.

**Test-first.** ``tests/test_emergence_synthetic.py`` was written before this package against
synthetic data with a closed-form answer: an injected interaction is recovered within its CI, a
main-effects-only table yields ~0, and the item-clustered SE is demonstrably wider than the iid one.
A silent numerical bug here is a false paper claim, which is why those tests are the gate.

Import layering: nothing here touches a model stack. ``statsmodels``/``scipy``/``numpy``/``pandas``
are core dependencies, so the whole package runs on the torch-free environment and in the core CI
job on both Windows and Linux.
"""

from __future__ import annotations

from margins.metrics.compare import DriftComparisonResult, compare_drift
from margins.metrics.config import (
    DEFAULT_METRICS,
    DRIFT_INPUT_KEY,
    METRICS_INPUT_KEY,
    SCORES_INPUT_KEY,
    BootstrapSettings,
    DriftThresholds,
    MetricsConfig,
    load_metrics_config,
)
from margins.metrics.design import (
    PairDesign,
    additive_decomposition,
    interaction_term,
    main_term,
    pair_design,
    pair_designs,
)
from margins.metrics.drift import DriftFitResult, fit_drift
from margins.metrics.estimator import (
    NotEstimableError,
    PairFit,
    PairFrame,
    TermEstimate,
    estimate_pair,
)
from margins.metrics.export import DEFAULT_METRICS_DIR, export_metrics, read_effects
from margins.metrics.manifest import metrics_inputs, metrics_record
from margins.metrics.metric import EmergenceMetric, EmergenceResult, triangulate
from margins.metrics.records import (
    DriftComparison,
    FamilyAgreement,
    FitReport,
    FitTerm,
    HarmTypeFlag,
    SideBySide,
)

__all__ = [
    "DEFAULT_METRICS",
    "DEFAULT_METRICS_DIR",
    "DRIFT_INPUT_KEY",
    "METRICS_INPUT_KEY",
    "SCORES_INPUT_KEY",
    "BootstrapSettings",
    "DriftComparison",
    "DriftComparisonResult",
    "DriftFitResult",
    "DriftThresholds",
    "EmergenceMetric",
    "EmergenceResult",
    "FamilyAgreement",
    "FitReport",
    "FitTerm",
    "HarmTypeFlag",
    "MetricsConfig",
    "NotEstimableError",
    "PairDesign",
    "PairFit",
    "PairFrame",
    "SideBySide",
    "TermEstimate",
    "additive_decomposition",
    "compare_drift",
    "estimate_pair",
    "export_metrics",
    "fit_drift",
    "interaction_term",
    "load_metrics_config",
    "main_term",
    "metrics_inputs",
    "metrics_record",
    "pair_design",
    "pair_designs",
    "read_effects",
    "triangulate",
]
