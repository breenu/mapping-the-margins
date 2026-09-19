"""Score the Kaggle rho run locally and measure the intraclass correlation (P9).

    uv run --extra inference python scripts/measure_rho.py --responses rho_responses.jsonl

**CPU only, and no model.** ``scripts/kaggle_rho_pilot.py`` does the GPU half on Kaggle and brings
back raw text; this does the rest on a laptop. The two neural scorers need ``--extra inference``
for ``transformers``, but they run on CPU and there is no 8B anywhere in this path.

**Why the scoring is here and not in the notebook.** Scoring in the notebook would mean copying
the P6 roster, its pinned classifier revisions, its lexicon digest and its sign contract into a
file that none of the P6 tests cover. A re-pin would then change rho silently. Here, the real
:mod:`margins.scoring` roster runs, ``verify_sign`` runs first on the real instruments, and
standardization is the same function the pipeline uses.

**The prompts are re-rendered locally and CHECKED against the run.** The notebook embedded them at
generation time; if the lattice, the templates or the decoding profile have moved since, the ids
will not match and this refuses rather than scoring a corpus it cannot describe.

What it emits, under ``--out-dir`` (default ``data/stats``):

* ``rho_measurement.json``  -- every estimate with its interval and variance components
* ``rho_measurement.md``    -- the same, for a human, byte-deterministic

and a run manifest. **It does not write ``min_cell_n``** and it does not edit the lattice config.
It prints what the Stage-A power tables imply at the measured rho, and stops there.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from margins.config import (
    default_config_dir,
    load_experiment_config,
    manifest_path,
    write_run_manifest,
)
from margins.inference.backend import make_response_id
from margins.lattice.build import DeclaredLattice
from margins.lattice.config import DEFAULT_LATTICE, default_lattice_config_dir, load_lattice_config
from margins.metrics.metric import triangulate
from margins.schemas import Response
from margins.scoring.config import DEFAULT_SCORING, default_scoring_config_dir, load_scoring_config
from margins.scoring.standardize import verify_sign
from margins.scoring.triangulate import build_triangulator
from margins.stats.icc import IccEstimate, IccObservation, estimate_icc

BENCHMARK = "open_gen_templates"
DEFAULT_OUT_DIR = Path("data") / "stats"
MEASUREMENT_JSON = "rho_measurement.json"
MEASUREMENT_MD = "rho_measurement.md"

#: The Stage-A ceiling table (simulated, 500 replicates, 0.3 SD concentrated, family 168).
#: Reproduced here ONLY to print what the measured rho implies. It is a lookup for a printout,
#: never an input to a decision: `scripts/run_power_analysis.py` is the authority.
STAGE_A_CEILING: dict[float, dict[int, float]] = {
    0.05: {9: 0.938, 12: 0.988, 18: 1.000, 27: 1.000},
    0.10: {9: 0.572, 12: 0.742, 18: 0.936, 27: 1.000},
    0.20: {9: 0.210, 12: 0.316, 18: 0.530, 27: 0.832},
    0.30: {9: 0.116, 12: 0.156, 18: 0.328, 27: 0.578},
    0.50: {9: 0.036, 12: 0.044, 18: 0.128, 27: 0.276},
}


def read_run(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Split the JSONL into its header and its generation rows."""
    header: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise SystemExit(f"{path} line {index + 1} is not valid JSON: {error}") from error
        if record.get("kind") == "header":
            header = record
            continue
        rows.append(record)
    if header is None:
        raise SystemExit(
            f"{path} carries no header record. It was not written by kaggle_rho_pilot.py, or "
            "the first line was lost."
        )
    if not rows:
        raise SystemExit(f"{path} carries a header but no generations.")
    return header, rows


def load_builder() -> Any:
    """Load ``scripts/build_rho_sample.py`` by path. ``scripts/`` is not a package."""
    import importlib.util

    path = Path(__file__).resolve().parent / "build_rho_sample.py"
    spec = importlib.util.spec_from_file_location("build_rho_sample", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_prompts_match(
    rows: list[dict[str, Any]], header: dict[str, Any], prompts: list[Any], meta: dict[str, Any]
) -> dict[str, str]:
    """Refuse if the run's prompt ids are not the ones this checkout renders.

    The alternative is scoring a corpus whose provenance cannot be stated. An id mismatch means
    the lattice, the templates or the decoding profile moved after the notebook was generated, and
    the honest response is to regenerate the notebook and re-run, not to proceed.
    """
    local = {prompt.prompt_id: prompt.rendered_text for prompt in prompts}
    seen = {row["prompt_id"] for row in rows}
    unknown = sorted(seen - set(local))
    if unknown:
        raise SystemExit(
            f"{len(unknown)} prompt id(s) in the run are not in the locally re-rendered set, e.g. "
            f"{unknown[:2]}.\nThe lattice ({meta['lattice_version']}), templates "
            f"({meta['template_version']}) or decoding profile have moved since the notebook was "
            "generated. Regenerate with `uv run python scripts/build_rho_sample.py` and re-run "
            "the Kaggle job; do not score a corpus whose provenance cannot be stated."
        )
    header_meta = header.get("meta", {})
    for key in ("lattice_version", "template_version"):
        if header_meta.get(key) and header_meta[key] != meta[key]:
            raise SystemExit(
                f"the run was generated under {key}={header_meta[key]!r} but this checkout has "
                f"{meta[key]!r}. Check out the matching revision or regenerate and re-run."
            )
    return local


def build_responses(rows: list[dict[str, Any]], header: dict[str, Any]) -> list[Response]:
    """Reconstruct :class:`~margins.schemas.Response` records from the raw generations.

    ``quant`` is taken from the run header rather than assumed: the notebook records what the load
    actually resolved to, exactly as the P5 harness does, and a Windows-style unquantized fallback
    must not be recorded here as NF4.
    """
    model = header["model"]
    decoding = header["decoding"]
    # `decoding_hash` is a PROPERTY of DecodingConfig, not a field, so it is absent from the
    # profile dump and lives in the run metadata instead. Read from there rather than recomputed,
    # so the response records carry the hash the run was actually generated under.
    decoding_hash = header["meta"]["decoding_hash"]
    quant = header.get("quant", "none")
    out: list[Response] = []
    for row in rows:
        response_id = make_response_id(
            prompt_id=row["prompt_id"],
            model_name=model["name"],
            model_id=model["model_id"],
            revision=model["revision"],
            seed=int(row["seed"]),
            sample_index=0,
            decoding_hash=decoding_hash,
        )
        out.append(
            Response(
                response_id=response_id,
                prompt_id=row["prompt_id"],
                model_id=model["model_id"],
                model_revision=model["revision"],
                seed=int(row["seed"]),
                temperature=float(decoding["temperature"]),
                decoding_key=decoding["key"],
                decoding_hash=decoding_hash,
                quant=quant,
                text=row["text"],
            )
        )
    return out


def measure(scores: list[Any], *, level: float) -> list[IccEstimate]:
    """One estimate per scorer, plus one for each harm type's triangulated series."""
    estimates: list[IccEstimate] = []
    by_harm: dict[str, list[Any]] = {}
    for score in scores:
        by_harm.setdefault(score.harm_type, []).append(score)

    for harm_type in sorted(by_harm):
        rows = by_harm[harm_type]
        for scorer_id in sorted({row.scorer_id for row in rows}):
            estimates.append(
                estimate_icc(
                    [
                        IccObservation(
                            cell_id=str(row.cell_id),
                            item_id=row.prompt_id,
                            value=row.standardized,
                        )
                        for row in rows
                        if row.scorer_id == scorer_id
                    ],
                    harm_type=harm_type,
                    scorer_id=scorer_id,
                    level=level,
                )
            )
        # The triangulated series -- the one the emergence estimator actually fits, and therefore
        # the rho the power simulation was assuming. Same rule as the metric layer, same function.
        _, observations = triangulate(rows)
        estimates.append(
            estimate_icc(
                [
                    IccObservation(
                        cell_id=observation.cell_id,
                        item_id=observation.prompt_id,
                        value=observation.combined,
                    )
                    for observation in observations
                ],
                harm_type=harm_type,
                scorer_id=None,
                level=level,
            )
        )
    return estimates


def implied_design(rho: float) -> str:
    """What the Stage-A tables say at this rho. A printout, never a decision."""
    nearest = min(STAGE_A_CEILING, key=lambda candidate: abs(candidate - rho))
    ceilings = STAGE_A_CEILING[nearest]
    workable = [frames for frames, power in sorted(ceilings.items()) if power >= 0.8]
    if not workable:
        return (
            f"at rho ~ {nearest:g}, NO frame count in the Stage-A grid (9/12/18/27) reaches 80% "
            "power on a 0.3 SD interaction at any seed budget"
        )
    return (
        f"at rho ~ {nearest:g}, the smallest Stage-A frame count whose ceiling clears 80% is "
        f"{workable[0]} frames per cell (ceiling {ceilings[workable[0]]:.3f})"
    )


def render_markdown(
    estimates: list[IccEstimate], header: dict[str, Any], *, n_responses: int
) -> str:
    meta = header.get("meta", {})
    lines = [
        "# Measured intraclass correlation (rho)",
        "",
        "Generated by `scripts/measure_rho.py`. **Do not edit by hand.**",
        "",
        f"- pair: `{meta.get('axis_a')} x {meta.get('axis_b')}`",
        f"- design: {meta.get('n_cells')} cells x {meta.get('n_frames')} frames x "
        f"{header.get('seeds')} seeds = {n_responses} generations",
        f"- model: `{header['model']['model_id']}` @ `{header['model']['revision'][:12]}` "
        f"(quant `{header.get('quant')}`)",
        f"- decoding: `{header['decoding']['key']}` temperature "
        f"{header['decoding']['temperature']}, top_p {header['decoding']['top_p']}",
        f"- lattice: `{meta.get('lattice_version')}`  templates: `{meta.get('template_version')}`",
        f"- gpu: `{header.get('gpu')}`  platform: `{header.get('platform')}`",
        "",
        "`rho` is the share of standardized harm-score variance sitting BETWEEN prompts rather",
        "than between repeated samples of one prompt, measured **within** a cell so that",
        "between-cell differences (the signal the model fits) never enter it. The interval is the",
        "exact F-based one, because rho is bounded and skewed near zero, which is where the",
        "design boundary sits.",
        "",
        "| harm type | series | rho | 95% CI | var between | var within | items | obs |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for estimate in sorted(
        estimates, key=lambda e: (e.harm_type, e.scorer_id is not None, e.scorer_id or "")
    ):
        series = "**triangulated**" if estimate.scorer_id is None else f"`{estimate.scorer_id}`"
        if not estimate.estimable:
            lines.append(
                f"| {estimate.harm_type} | {series} | not estimable | — | — | — "
                f"| {estimate.n_items} | {estimate.n_obs} |"
            )
            continue
        assert estimate.rho is not None
        lines.append(
            f"| {estimate.harm_type} | {series} | {estimate.rho:.4f} "
            f"| [{estimate.ci_low:.4f}, {estimate.ci_high:.4f}] "
            f"| {estimate.var_between:.4f} | {estimate.var_within:.4f} "
            f"| {estimate.n_items} | {estimate.n_obs} |"
        )

    lines += ["", "## What this implies, and what it does not", ""]
    for estimate in estimates:
        if estimate.scorer_id is None and estimate.estimable and estimate.rho is not None:
            lines.append(f"- {estimate.harm_type}: {implied_design(estimate.rho)}.")
    lines += [
        "",
        "Those lines are a LOOKUP into the Stage-A ceiling table, not a decision. `min_cell_n`",
        "and the rendering budget are one decision, taken together against the pre-committed",
        "frames-and-renderings rule, and neither is written by this script.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True, help="rho_responses.jsonl")
    parser.add_argument("--scoring", default=DEFAULT_SCORING)
    parser.add_argument("--lattice", default=DEFAULT_LATTICE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--run-name", default="rho")
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--limit", type=int, default=None, help="Smoke runs only.")
    parser.add_argument("--no-manifest", action="store_true")
    args = parser.parse_args(argv)

    header, rows = read_run(args.responses)
    if args.limit is not None:
        rows = rows[: args.limit]
    builder = load_builder()
    prompt_records, _cells, local_meta = builder.rho_prompt_records()
    local_texts = check_prompts_match(rows, header, prompt_records, local_meta)

    config = load_scoring_config(args.scoring, config_dir=default_scoring_config_dir())
    lattice = DeclaredLattice(load_lattice_config(args.lattice))

    meta = header.get("meta", {})
    print(f"run        {args.responses}")
    print(f"pair       {meta.get('axis_a')} x {meta.get('axis_b')}")
    print(
        f"design     {meta.get('n_cells')} cells x {meta.get('n_frames')} frames x "
        f"{header.get('seeds')} seeds -> {len(rows)} generations"
    )
    print(f"model      {header['model']['model_id']} @ {header['model']['revision'][:12]}")
    print(f"decoding   {header['decoding']['key']} temp {header['decoding']['temperature']}")
    print(f"roster     {config.key} {config.scoring_version}")

    # The locally re-rendered Prompt records: the scorers see exactly what P6 would hand them.
    prompts_by_id = {prompt.prompt_id: prompt for prompt in prompt_records}

    responses = build_responses(rows, header)
    triangulator = build_triangulator(
        config, prompts_by_id=prompts_by_id, benchmark=BENCHMARK, data_dir=args.data_dir
    )
    for scorer in triangulator.scorers:
        verify_sign(scorer.spec, scorer._text_scorer)
    print(f"sign       verified for all {len(triangulator.scorers)} scorers")

    scores, _reports = triangulator.score_all(responses)
    print(f"scored     {len(scores)} rows over {len(responses)} responses\n")

    estimates = measure(list(scores), level=args.ci_level)

    print("MEASURED rho (between-prompt share of standardized harm-score variance, within cell)\n")
    print(f"  {'harm type':<12} {'series':<22} {'rho':>8} {'95% CI':>20} {'items':>7} {'obs':>7}")
    for estimate in estimates:
        series = "triangulated" if estimate.scorer_id is None else estimate.scorer_id
        if not estimate.estimable:
            print(
                f"  {estimate.harm_type:<12} {series:<22} {'n/a':>8} {'not estimable':>20} "
                f"{estimate.n_items:>7} {estimate.n_obs:>7}"
            )
            print(f"      {estimate.reason}")
            continue
        assert estimate.rho is not None
        interval = f"[{estimate.ci_low:.3f}, {estimate.ci_high:.3f}]"
        print(
            f"  {estimate.harm_type:<12} {series:<22} {estimate.rho:>8.4f} {interval:>20} "
            f"{estimate.n_items:>7} {estimate.n_obs:>7}"
        )

    print("\nWHAT THE STAGE-A TABLES SAY AT THIS rho (a lookup, not a decision)\n")
    for estimate in estimates:
        if estimate.scorer_id is None and estimate.estimable and estimate.rho is not None:
            print(f"  {estimate.harm_type:<12} {implied_design(estimate.rho)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "responses_file": str(args.responses),
        "header": header,
        "n_generations": len(rows),
        "n_prompts": len(local_texts),
        "scoring_version": config.scoring_version,
        "scoring_hash": config.scoring_hash,
        "lattice_version": lattice.config.lattice_version,
        "ci_level": args.ci_level,
        "estimates": [estimate.model_dump(mode="json") for estimate in estimates],
    }
    json_path = args.out_dir / MEASUREMENT_JSON
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md_path = args.out_dir / MEASUREMENT_MD
    with md_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(render_markdown(estimates, header, n_responses=len(rows)))
    print(f"\nwrote {json_path}\nwrote {md_path}")

    if args.no_manifest:
        return 0

    overrides = [f"run_name={args.run_name}", f"scoring_name={args.scoring}"]
    if args.results_dir is not None:
        overrides.append(f"results_dir={args.results_dir.as_posix()}")
    cfg = load_experiment_config(
        config_dir=args.config_dir or default_config_dir(), overrides=overrides
    )
    digest = hashlib.sha256(args.responses.read_bytes()).hexdigest()
    manifest = write_run_manifest(
        cfg,
        inputs={
            "rho_responses_jsonl": digest,
            "lattice_config": _sha256(default_lattice_config_dir() / f"{args.lattice}.yaml"),
            "scoring_config": _sha256(default_scoring_config_dir() / f"{args.scoring}.yaml"),
        },
    )
    print(f"manifest   {manifest_path(cfg, manifest.run_id)}")
    print(
        "\nmin_cell_n is NOT written. The frames-and-renderings rule is applied to this rho "
        "separately, and the lattice config's 96/60 markers stand until it is."
    )
    return 0


def _sha256(path: Path) -> str:
    from margins.prompts.config import file_sha256

    return file_sha256(path)


if __name__ == "__main__":
    raise SystemExit(main())
