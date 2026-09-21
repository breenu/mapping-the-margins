"""Score a response file through the whole roster, both conditions, and emit every P6 artifact.

    uv run python scripts/run_scoring.py --responses data/responses/<model>/<run_id>.jsonl

Produces, under ``--out-dir`` (default ``data/scoring``):

* ``scores.parquet``          -- one row per (response, scorer), model condition
* ``drift_control.parquet``   -- the scorer's own series over known-neutral text, keyed for P7
* ``agreement.csv`` / ``.md`` -- inter-scorer alpha and kappa, a first-class reported artifact

and a run manifest carrying the roster, every scorer's pin, and the agreement summary.

**The drift control runs on every invocation, not on a flag.** It is not an optional diagnostic:
without it the interaction P7 estimates cannot be distinguished from the scorer's own response to
identity terms, and RQ2's estimand is not credible (PROJECT_CONTEXT.md section 9). Making it
skippable would make the credible path the one you have to remember to ask for.

**The EXTRACTIVE paradigm is the one exception, and it is a structural one (P2).** A roster of
paired log-probability scorers has no drift control because it has neither text nor instrument:
there is no neutral analogue of "how much more likely is this sentence than its minimally
different counterpart", and no classifier reading identity terms for the control to catch. The
run reports the absence rather than emitting an empty control series. See
``configs/scoring/crows_pll_v1.yaml`` and ``docs/methods.md``.

The manifest is written BEFORE scoring, as ``run_inference.py`` does: an interrupted run still
leaves a complete record of what it was.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from margins.config import (
    LATTICE_INPUT_KEY,
    default_config_dir,
    load_experiment_config,
    manifest_path,
    write_run_manifest,
)
from margins.inference.harness import read_responses
from margins.lattice.build import DeclaredLattice
from margins.lattice.config import DEFAULT_LATTICE, default_lattice_config_dir, load_lattice_config
from margins.prompts.config import (
    DEFAULT_PROMPT_SET,
    TEMPLATE_DIRNAME,
    default_prompt_config_dir,
    file_sha256,
    load_prompt_config,
    load_template_files,
)
from margins.prompts.export import PROMPTS_FILENAME, prompt_set_inputs, read_prompts
from margins.scoring.config import DEFAULT_SCORING, default_scoring_config_dir, load_scoring_config
from margins.scoring.drift import build_control_texts, control_responses, drift_rows
from margins.scoring.export import DEFAULT_SCORING_DIR, export_scoring
from margins.scoring.lexicon import WordlistScorer
from margins.scoring.manifest import scoring_inputs, scoring_record
from margins.scoring.standardize import verify_paired_sign, verify_sign
from margins.scoring.triangulate import build_triangulator

DEFAULT_PROMPTS = Path("data") / "prompts" / PROMPTS_FILENAME

#: The generation vertical's source label. A DEFAULT rather than a constant since P2: the
#: extractive vertical scores under `crows_pairs`, and standardization is within a source, so a
#: hard-coded label would have pooled the two paradigms into one z-score -- the exact thing
#: `standardize_within_benchmark` takes a required keyword to prevent.
DEFAULT_BENCHMARK_LABEL = "open_gen_templates"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True, help="Response JSONL to score.")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--scoring", default=DEFAULT_SCORING, help="Scoring roster config name.")
    parser.add_argument("--lattice", default=DEFAULT_LATTICE)
    parser.add_argument("--prompt-set", default=DEFAULT_PROMPT_SET)
    parser.add_argument(
        "--benchmark",
        default=DEFAULT_BENCHMARK_LABEL,
        help="Source label stamped on every score. Standardization is WITHIN this source.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_SCORING_DIR)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--run-name", default="scoring")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N responses.")
    parser.add_argument(
        "--no-manifest", action="store_true", help="Skip the run manifest (debugging only)."
    )
    args = parser.parse_args(argv)

    config = load_scoring_config(args.scoring, config_dir=default_scoring_config_dir())
    lattice = DeclaredLattice(load_lattice_config(args.lattice))

    prompts = read_prompts(args.prompts)
    prompts_by_id = {p.prompt_id: p for p in prompts}
    responses = read_responses(args.responses)
    if args.limit is not None:
        responses = responses[: args.limit]
    if not responses:
        parser.error(f"no responses to score in {args.responses}")

    print(f"roster     {config.key}  {config.scoring_version}  hash={config.scoring_hash[:16]}...")
    print(f"scorers    {len(config.scorers)} over harm types {list(config.harm_types)}")
    print(f"responses  {len(responses)} from {args.responses}")

    triangulator = build_triangulator(
        config, prompts_by_id=prompts_by_id, benchmark=args.benchmark, data_dir=args.data_dir
    )

    # The sign contract, checked before anything is scored. A mis-signed scorer inverts every
    # effect downstream while producing entirely plausible output, so this runs first and raises
    # rather than warning.
    #
    # Two probe kinds, because there are two kinds of scorer. A text scorer's polarity is a
    # property of an INSTRUMENT and is checked by running the real one. A paired scorer has no
    # instrument -- the log-probabilities are the model's -- so what is checked is OUR arithmetic,
    # against supplied log-probabilities and without loading anything.
    for scorer in triangulator.scorers:
        if scorer.spec.reads_logprobs:
            verify_paired_sign(scorer.spec, scorer._reduction)
        else:
            verify_sign(scorer.spec, scorer._text_scorer)
    print(f"sign       verified for all {len(triangulator.scorers)} scorers")

    # --- model condition -------------------------------------------------------------
    scores, reports = triangulator.score_all(responses)
    print(f"scored     {len(scores)} rows (model)")

    # --- drift control ---------------------------------------------------------------
    #
    # Skipped for a paired roster, and skipped STRUCTURALLY rather than by flag: the control scores
    # known-neutral TEXT through the same instrument, and a log-probability difference has neither.
    # Printing the reason is the point -- an absent control that says nothing reads like a control
    # that returned nothing.
    paired_roster = all(spec.reads_logprobs for spec in config.scorers)
    control_texts: list[object] = []
    control_scores: list[object] = []
    control_reports: list[object] = []
    drift: list[object] = []

    if paired_roster:
        print(
            "drift      NOT APPLICABLE for a paired roster: the identity-drift control scores\n"
            "           known-neutral TEXT through the same INSTRUMENT, and a sequence\n"
            "           log-probability difference has neither. Recorded in docs/methods.md,\n"
            "           not silently omitted."
        )
    else:
        control_texts = list(build_control_texts(config, prompts, lattice=lattice))
        control_records, control_prompts = control_responses(
            control_texts, decoding_hash=config.scoring_hash
        )
        control_triangulator = build_triangulator(
            config, prompts_by_id=control_prompts, benchmark=args.benchmark, data_dir=args.data_dir
        )
        control_scores, control_reports = control_triangulator.score_all(
            control_records, condition="drift_control"
        )
        drift = list(drift_rows(control_scores, control_texts, lattice.cells()))
        print(f"drift      {len(control_texts)} control texts -> {len(drift)} rows")
        for family in config.drift.families:
            print(f"  {family:<22} {sum(1 for t in control_texts if t.family == family)}")

    # --- agreement -------------------------------------------------------------------
    agreement = triangulator.agreement([*scores, *control_scores])
    print("\nagreement (model condition)")
    for record in agreement:
        if record.metric != "krippendorff_alpha" or record.condition != "model":
            continue
        value = "not estimable" if record.value is None else f"{record.value:.3f}"
        print(f"  {record.harm_type:<12} alpha {value:>14}  (n={record.n_units})")

    all_reports = [*reports, *control_reports]
    paths = export_scoring(
        scores,
        drift,
        agreement,
        scoring_version=config.scoring_version,
        reports=all_reports,
        out_dir=args.out_dir,
    )
    print("\nwrote")
    for path in paths:
        print(f"  {path}")

    if args.no_manifest:
        return 0

    overrides = [f"run_name={args.run_name}", f"scoring_name={args.scoring}"]
    if args.results_dir is not None:
        overrides.append(f"results_dir={args.results_dir.as_posix()}")
    cfg = load_experiment_config(
        config_dir=args.config_dir or default_config_dir(), overrides=overrides
    )

    prompt_dir = default_prompt_config_dir()
    if paired_roster:
        # The extractive prompt set has no templates to digest, so the open_gen `inputs` builder
        # does not apply. Recording the lattice alone is honest here: `build_crows_prompts.py`
        # already recorded the prompt-set and benchmark digests when it rendered the prompts.
        inputs = {
            LATTICE_INPUT_KEY: file_sha256(default_lattice_config_dir() / f"{args.lattice}.yaml")
        }
    else:
        prompt_config = load_prompt_config(args.prompt_set, config_dir=prompt_dir)
        files = load_template_files(prompt_config, prompt_dir / TEMPLATE_DIRNAME)
        inputs = prompt_set_inputs(
            lattice_config_path=default_lattice_config_dir() / f"{args.lattice}.yaml",
            prompt_config_path=prompt_dir / f"{args.prompt_set}.yaml",
            files=files,
        )
    inputs.update(
        scoring_inputs(
            config,
            scoring_config_path=default_scoring_config_dir() / f"{args.scoring}.yaml",
            data_dir=args.data_dir,
        )
    )

    lexicon_sizes = {
        scorer.scorer_id: text_scorer.size
        for scorer in triangulator.scorers
        # getattr, not attribute access: a PairedScorer has no `_text_scorer` at all, and since P2
        # a roster can hold either kind.
        if isinstance((text_scorer := getattr(scorer, "_text_scorer", None)), WordlistScorer)
    }

    manifest = write_run_manifest(
        cfg,
        inputs=inputs,
        scoring=scoring_record(
            config,
            scoring_name=args.scoring,
            benchmark=args.benchmark,
            n_responses=len(responses),
            n_scores=len(scores),
            agreement=agreement,
            reports=all_reports,
            n_control_texts=len(control_texts),
            lexicon_sizes=lexicon_sizes,
        ),
    )
    print(f"\nmanifest   {manifest_path(cfg, manifest.run_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
