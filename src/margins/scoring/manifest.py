"""Manifest wiring for the scoring stage (P6).

The same two halves as :mod:`margins.inference.manifest`:

* :func:`scoring_record` -- what the run *did*: every scorer's pin, sign and threshold, how many
  rows it produced, how many standardization groups were degenerate, and the agreement it achieved.
* :func:`scoring_inputs` -- ``{name: sha256}`` for the roster file and for every fetched lexicon,
  under ``scorer:<scorer_id>``.

**Why agreement rides in the manifest and not only in the emitted table.** Framing constraint 6
makes agreement a precondition for a harm type being load-bearing. A run whose agreement exists only
in a gitignored Parquet artifact cannot be audited from its manifest, and ``results/run_manifests/``
is the committed record. The full table is still exported; this is the summary that survives.

**Why a classifier's revision is recorded per scorer rather than once.** Two scorers can name the
same repo at different revisions, and a re-pin of one is not a re-pin of the other. The same logic
that gives ``chat_template:<model>`` a per-model key applies here.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from margins.config import ScorerRecord, ScoringRecord
from margins.scoring.config import (
    SCORER_INPUT_PREFIX,
    SCORING_INPUT_KEY,
    ScorerSpec,
    ScoringConfig,
)
from margins.scoring.standardize import StandardizationReport
from margins.scoring.triangulate import AgreementRecord

__all__ = ["scorer_record", "scoring_inputs", "scoring_record"]


def scorer_record(spec: ScorerSpec, *, lexicon_size: int | None = None) -> ScorerRecord:
    """Flatten one roster entry into its manifest shape."""
    model_pin = spec.model_pin
    lexicon_pin = spec.lexicon_pin
    return ScorerRecord(
        scorer_id=spec.scorer_id,
        harm_type=spec.harm_type,
        mechanism=spec.mechanism,
        sign=spec.sign,
        threshold=spec.threshold,
        pin_digest=spec.pin_digest,
        model_repo=model_pin.repo if model_pin else None,
        model_revision=model_pin.revision if model_pin else None,
        model_label=model_pin.label if model_pin else None,
        lexicon_name=lexicon_pin.name if lexicon_pin else None,
        lexicon_revision=lexicon_pin.revision if lexicon_pin else None,
        lexicon_version=lexicon_pin.version if lexicon_pin else None,
        lexicon_sha256=lexicon_pin.sha256 if lexicon_pin else None,
        lexicon_size=lexicon_size,
        license=model_pin.license if model_pin else (lexicon_pin.license if lexicon_pin else None),
    )


def scoring_record(
    config: ScoringConfig,
    *,
    scoring_name: str,
    benchmark: str,
    n_responses: int,
    n_scores: int,
    agreement: Sequence[AgreementRecord] = (),
    reports: Sequence[StandardizationReport] = (),
    n_control_texts: int = 0,
    lexicon_sizes: dict[str, int] | None = None,
) -> ScoringRecord:
    """Assemble the manifest's scoring block.

    ``agreement_summary`` carries the **model-condition** Krippendorff alpha per harm type. Model
    condition specifically: the drift control's alpha describes agreement about neutral text, which
    is a different question and would be misleading under the same key.
    """
    sizes = lexicon_sizes or {}
    summary = {
        record.harm_type: record.value
        for record in agreement
        if record.metric == "krippendorff_alpha"
        and record.condition == "model"
        and record.value is not None
    }

    return ScoringRecord(
        scoring_name=scoring_name,
        scoring_version=config.scoring_version,
        scoring_hash=config.scoring_hash,
        scorers=tuple(
            scorer_record(spec, lexicon_size=sizes.get(spec.scorer_id)) for spec in config.scorers
        ),
        harm_types=config.harm_types,
        benchmark=benchmark,
        n_responses=n_responses,
        n_scores=n_scores,
        control_families=tuple(config.drift.families),
        n_control_texts=n_control_texts,
        degenerate_groups=sum(1 for report in reports if report.degenerate),
        agreement_summary=summary,
    )


def scoring_inputs(
    config: ScoringConfig,
    *,
    scoring_config_path: Path,
    data_dir: Path,
) -> dict[str, str]:
    """``{source_name: sha256}`` for the roster and every fetched lexicon behind it.

    A lexicon that ships inside a pinned package (VADER) has no file to digest; it is pinned by the
    package version, which the manifest already records in ``package_versions``. Such a scorer
    contributes no entry here rather than a fabricated one.
    """
    from margins.prompts.config import file_sha256

    inputs = {SCORING_INPUT_KEY: file_sha256(scoring_config_path)}

    for spec in config.scorers:
        pin = spec.lexicon_pin
        if pin is None or pin.sha256 is None:
            continue
        key = f"{SCORER_INPUT_PREFIX}{spec.scorer_id}"
        if pin.path is not None:
            path = pin.path if pin.path.is_absolute() else data_dir / pin.path
            if path.is_file():
                # Digest the file that is actually on disk, then check it against the pin. A
                # manifest that copied the declared sha256 without looking would record a digest
                # for a file that may have been replaced.
                actual = file_sha256(path)
                if actual != pin.sha256:
                    raise ValueError(
                        f"lexicon {pin.name!r} for scorer {spec.scorer_id!r} does not match its "
                        f"pin: file is {actual}, config declares {pin.sha256}. Re-fetch with "
                        "`uv run python scripts/fetch_lexicons.py`, or the roster is describing a "
                        "different wordlist from the one that produced these scores."
                    )
                inputs[key] = actual
                continue
        inputs[key] = pin.sha256

    return inputs
