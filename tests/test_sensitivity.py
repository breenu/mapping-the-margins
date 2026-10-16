"""E4 and R2: the analyses fixed in the pre-registration's §10 on 16 Oct 2026, held to them.

Each test pins one of the pre-registered rules on synthetic scores shaped like ``run_scoring.py``'s
(``tests.p11_synthetic``), so the rule is checked on the CI job without the real run's data:

* E4 standardizes each arm WITHIN itself, fits per seed and pooled, and reports every free term;
* R2 maps the rerun through the ORIGINAL constants, substitutes it into the family without
  changing the family, and PASSES only if every estimate is inside the original CI and every
  verdict is unchanged -- and an identical rerun passes with a difference of exactly zero, while
  a different one fails.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from margins.lattice import DeclaredLattice, load_lattice_config
from margins.metrics.config import load_metrics_config
from margins.orchestrate.config import load_p11_config
from margins.orchestrate.smoke import pair_cells
from margins.schemas import EffectEstimate, ItemScore
from margins.scoring.config import load_scoring_config
from margins.scoring.standardize import (
    signs_from_specs,
    standardize_with_reference,
    standardize_within_benchmark,
)
from margins.sensitivity import (
    ArmScores,
    compare_to_original,
    exact_match_rate,
    fit_pair,
    free_terms,
    seed_groups,
    sensitivity_rows,
    summarize,
    verdict_changes,
)
from margins.sensitivity.r2 import R2Verdict
from tests.p11_synthetic import OPEN_GEN_ROSTER, model_scores

BENCHMARK = "open_gen_templates"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PAIR = ("socioeconomic_class", "race_ethnicity")


@pytest.fixture(scope="module")
def cells():  # type: ignore[no-untyped-def]
    return pair_cells(DeclaredLattice(load_lattice_config("confirmatory_v1")), PAIR)


@pytest.fixture(scope="module")
def signs() -> dict[str, int]:
    return signs_from_specs(load_scoring_config("default_v1").scorers)


@pytest.fixture(scope="module")
def metrics():  # type: ignore[no-untyped-def]
    config = load_metrics_config()
    return config.model_copy(
        update={"bootstrap": config.bootstrap.model_copy(update={"n_resamples": 60})}
    )


def _scores(cells, *, seeds, temperature=0.7, seed=0, gamma_scale=0.8) -> list[ItemScore]:  # type: ignore[no-untyped-def]
    rows = model_scores(
        cells,
        benchmark=BENCHMARK,
        model_id=MODEL,
        roster=OPEN_GEN_ROSTER,
        n_items=3,
        seeds=seeds,
        gamma_scale=gamma_scale,
        rng=np.random.default_rng(seed),
    )
    return [r.model_copy(update={"temperature": temperature}) for r in rows]


# ------------------------------------------------------------------ standardization (R2's rule)


def test_reference_standardization_reproduces_the_original_exactly(cells, signs) -> None:  # type: ignore[no-untyped-def]
    rows = _scores(cells, seeds=(1, 2))
    original, reference = standardize_within_benchmark(rows, benchmark=BENCHMARK, signs=signs)
    mapped = standardize_with_reference(rows, benchmark=BENCHMARK, reference=reference)
    assert [r.standardized for r in mapped] == pytest.approx([r.standardized for r in original])


def test_reference_standardization_refuses_another_source_and_an_uncovered_group(
    cells, signs
) -> None:  # type: ignore[no-untyped-def]
    rows = _scores(cells, seeds=(1, 2))
    _, reference = standardize_within_benchmark(rows, benchmark=BENCHMARK, signs=signs)
    foreign = [rows[0].model_copy(update={"benchmark": "redteam"})]
    with pytest.raises(ValueError, match="never across"):
        standardize_with_reference(foreign, benchmark=BENCHMARK, reference=reference)
    unknown = [rows[0].model_copy(update={"scorer_id": "nobody"})]
    with pytest.raises(ValueError, match="no reference constants"):
        standardize_with_reference(unknown, benchmark=BENCHMARK, reference=reference)


def test_a_degenerate_reference_group_maps_to_zero(cells, signs) -> None:  # type: ignore[no-untyped-def]
    rows = [r.model_copy(update={"raw": 0.0}) for r in _scores(cells, seeds=(1, 2))]
    _, reference = standardize_within_benchmark(rows, benchmark=BENCHMARK, signs=signs)
    shifted = [r.model_copy(update={"raw": 5.0}) for r in rows]
    mapped = standardize_with_reference(shifted, benchmark=BENCHMARK, reference=reference)
    assert {r.standardized for r in mapped} == {0.0}


# ------------------------------------------------------------------ E4


def test_seed_groups_are_pooled_then_each_seed_and_greedy_is_pooled_only(cells) -> None:  # type: ignore[no-untyped-def]
    sampled = seed_groups(_scores(cells, seeds=(9, 3, 5)))
    assert [group for group, _ in sampled] == [0, 1, 2, 3]
    assert {s.seed for s in sampled[1][1]} == {3}  # groups follow ascending seed order
    assert len(sampled[0][1]) == sum(len(rows) for _, rows in sampled[1:])
    assert [group for group, _ in seed_groups(_scores(cells, seeds=(7,)))] == [0]


def test_e4_reports_every_free_term_per_arm_per_seed_group(cells, signs, metrics) -> None:  # type: ignore[no-untyped-def]
    arms = [
        ArmScores(MODEL, "t07", _scores(cells, seeds=(1, 2), temperature=0.7, seed=1)),
        ArmScores(MODEL, "greedy", _scores(cells, seeds=(1,), temperature=0.0, seed=2)),
    ]
    rows = sensitivity_rows(
        arms, cells, benchmark=BENCHMARK, signs=signs, config=metrics, min_cell_n=108
    )
    terms = free_terms(cells)
    assert len(terms) == 9
    # t07: pooled + 2 seeds; greedy: pooled only. Two harm types, nine terms each.
    assert len(rows) == (3 + 1) * 2 * 9
    assert {r["term"] for r in rows} == set(terms)
    assert {(r["temperature"], r["seed_group"]) for r in rows} == {
        (0.7, 0),
        (0.7, 1),
        (0.7, 2),
        (0.0, 0),
    }
    assert all(r["ci_low"] <= r["estimate"] <= r["ci_high"] for r in rows)
    assert {r["data_status"] for r in rows} == {"real"}


def test_e4_standardizes_each_arm_within_itself(cells, signs, metrics) -> None:  # type: ignore[no-untyped-def]
    """Shifting one arm's raw scale must not move its estimates: it is z-scored on its own."""
    base = _scores(cells, seeds=(1, 2), seed=4)
    shifted = [r.model_copy(update={"raw": 3.0 * r.raw + 10.0}) for r in base]

    def pooled(rows):  # type: ignore[no-untyped-def]
        out = sensitivity_rows(
            [ArmScores(MODEL, "t07", rows)],
            cells,
            benchmark=BENCHMARK,
            signs=signs,
            config=metrics,
            min_cell_n=108,
        )
        return [r["estimate"] for r in out if r["seed_group"] == 0]

    assert pooled(shifted) == pytest.approx(pooled(base))


def test_e4_refuses_an_arm_with_two_recorded_temperatures(cells, signs, metrics) -> None:  # type: ignore[no-untyped-def]
    rows = _scores(cells, seeds=(1, 2))
    mixed = [*rows[:10], *(r.model_copy(update={"temperature": 1.0}) for r in rows[10:])]
    with pytest.raises(ValueError, match="mixes recorded temperatures"):
        sensitivity_rows(
            [ArmScores(MODEL, "t07", mixed)],
            cells,
            benchmark=BENCHMARK,
            signs=signs,
            config=metrics,
            min_cell_n=108,
        )


# ------------------------------------------------------------------ R2


def _fit(rows, cells, metrics):  # type: ignore[no-untyped-def]
    return fit_pair(rows, cells, config=metrics, min_cell_n=108)


def test_an_identical_rerun_passes_with_a_difference_of_exactly_zero(cells, signs, metrics) -> None:  # type: ignore[no-untyped-def]
    raw = _scores(cells, seeds=(1, 2), seed=7)
    original, reference = standardize_within_benchmark(raw, benchmark=BENCHMARK, signs=signs)
    rerun = standardize_with_reference(raw, benchmark=BENCHMARK, reference=reference)
    terms = free_terms(cells)
    comparisons = compare_to_original(
        _fit(original, cells, metrics).effects,
        _fit(rerun, cells, metrics).effects,
        axis_pair=PAIR,
        terms=terms,
    )
    verdicts = [
        R2Verdict(c.model_id, c.harm_type, c.term, "descriptive_only", "descriptive_only")
        for c in comparisons
    ]
    summary = summarize(comparisons, verdicts)
    assert summary["passed"] is True
    assert summary["max_abs_diff_se"] == pytest.approx(0.0, abs=1e-9)
    assert summary["n_terms"] == 2 * len(terms)


def test_a_different_rerun_fails_the_ci_half(cells, signs, metrics) -> None:  # type: ignore[no-untyped-def]
    raw = _scores(cells, seeds=(1, 2), seed=7)
    original, reference = standardize_within_benchmark(raw, benchmark=BENCHMARK, signs=signs)
    other = _scores(cells, seeds=(1, 2), seed=99, gamma_scale=2.5)  # a different interaction
    rerun = standardize_with_reference(other, benchmark=BENCHMARK, reference=reference)
    comparisons = compare_to_original(
        _fit(original, cells, metrics).effects,
        _fit(rerun, cells, metrics).effects,
        axis_pair=PAIR,
        terms=free_terms(cells),
    )
    verdicts = [R2Verdict(c.model_id, c.harm_type, c.term, "x", "x") for c in comparisons]
    summary = summarize(comparisons, verdicts)
    assert summary["n_inside_ci"] < summary["n_terms"]
    assert summary["passed"] is False


def test_a_changed_verdict_fails_even_inside_every_ci() -> None:
    verdicts = [R2Verdict("m", "regard", "t", "descriptive_only", "claimable")]
    assert not verdicts[0].same
    from margins.sensitivity.r2 import R2Comparison

    inside = R2Comparison("m", "regard", "t", 0.1, 0.05, 0.0, 0.2, 0.11, 0.0, 0.2)
    assert summarize([inside], verdicts)["passed"] is False


def test_a_term_missing_from_one_side_is_refused(cells, signs, metrics) -> None:  # type: ignore[no-untyped-def]
    raw = _scores(cells, seeds=(1, 2), seed=7)
    original, _ = standardize_within_benchmark(raw, benchmark=BENCHMARK, signs=signs)
    effects = _fit(original, cells, metrics).effects
    terms = free_terms(cells)
    lost = next(e for e in effects if e.term == terms[0])
    with pytest.raises(ValueError, match="disagree on which terms"):
        compare_to_original(
            effects, [e for e in effects if e is not lost], axis_pair=PAIR, terms=terms
        )


def test_verdict_changes_requires_the_same_family_terms() -> None:
    from margins.stats.records import SignificanceClaim

    def claim(term: str, status: str) -> SignificanceClaim:
        return SignificanceClaim(
            benchmark=BENCHMARK,
            model_id="m",
            harm_type="regard",
            axis_a=PAIR[0],
            axis_b=PAIR[1],
            term=term,
            estimate=0.0,
            se=0.1,
            ci_low=-0.2,
            ci_high=0.2,
            powered=True,
            status=status,
            claimable=False,
            refusal_reason="r",
        )

    before = [claim("a", "descriptive_only"), claim("b", "descriptive_only")]
    after = [claim("a", "descriptive_only"), claim("b", "underpowered")]
    changes = verdict_changes(before, after, axis_pair=PAIR, terms=("a", "b"))
    assert [v.same for v in changes] == [True, False]
    with pytest.raises(ValueError):
        verdict_changes(before, after[:1], axis_pair=PAIR, terms=("a", "b"))


def test_exact_match_counts_byte_identical_texts_and_refuses_strangers() -> None:
    assert exact_match_rate({"a": "x", "b": "y"}, {"a": "x", "b": "z"}) == (2, 1, 0.5)
    with pytest.raises(ValueError, match="no original"):
        exact_match_rate({"a": "x"}, {"q": "x"})


# ------------------------------------------------------------------ config + notebook


def test_r2_config_is_pinned_to_the_arms_it_is_compared_against() -> None:
    cfg = load_p11_config()
    assert cfg.r2.sampled_seeds == cfg.e1.seeds
    assert cfg.r2.sampled_decoding == cfg.e1.decoding
    assert cfg.r2.pair == cfg.e4.pair
    assert cfg.r2_greedy.kind == "greedy"
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="E1's decoding and seeds"):
        cfg.model_validate(
            cfg.model_dump() | {"r2": cfg.r2.model_dump() | {"sampled_seeds": (1, 2, 3, 4)}}
        )


def test_the_kaggle_notebook_is_generated_and_up_to_date() -> None:
    import scripts.build_r2_notebook as builder

    assert builder.main(["--check"]) == 0


def test_the_kaggle_notebook_never_puts_the_github_token_in_a_command() -> None:
    """Found on the first real Kaggle run: the token rode in the clone URL, the clone failed, and
    CalledProcessError printed the full argv -- token included. It must never be in an argv."""
    import scripts.build_r2_notebook as builder

    cells = json.loads(builder.render_notebook())["cells"]
    code = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    assert "x-access-token:{_github_token}@" not in code  # not in a URL
    assert "@{REPO}" not in code
    assert "extraheader" in code and "GIT_CONFIG_VALUE_0" in code  # sent as a header via env
    assert 'replace(_github_token, "***")' in code  # scrubbed from any error text
    assert ".strip()" in code  # the leading-space secret that broke the first clone
    assert "check=True" not in code.split("_clone = subprocess.run(")[1].split(")")[0]
    assert "--locked" in code and "pip install" not in code


def test_every_notebook_command_parses_against_the_real_script(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import argparse
    import importlib.util
    import shlex

    import scripts.build_r2_notebook as builder

    class Parsed(Exception):  # noqa: N818
        pass

    def parse_args(self, args=None, namespace=None):  # type: ignore[no-untyped-def]
        _, extras = self.parse_known_args(args, namespace)
        assert not extras, extras
        raise Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_args)
    for line in builder.notebook_commands():
        tokens = shlex.split(line)[3:]
        path = Path("scripts") / Path(tokens[0]).name
        spec = importlib.util.spec_from_file_location(f"_nb_{path.stem}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with pytest.raises(Parsed):
            if path.stem == "run_p11":
                module.build_parser().parse_args(tokens[1:])
            else:
                module.main(tokens[1:])


def test_effect_estimates_carry_no_significance_from_e4_or_r2(cells, signs, metrics) -> None:  # type: ignore[no-untyped-def]
    """Invariant 3: only StatsEngine sets q/family_size/correction. A fit here sets none."""
    raw = _scores(cells, seeds=(1, 2), seed=3)
    rows, _ = standardize_within_benchmark(raw, benchmark=BENCHMARK, signs=signs)
    effects: list[EffectEstimate] = _fit(rows, cells, metrics).effects
    assert effects and all(
        e.q is None and e.family_size is None and e.correction is None for e in effects
    )


def test_e4_process_lets_make_figures_fall_back_on_the_default_layout() -> None:
    """Found running it for real: an explicit --table-dir is strict, so the still-pending
    frame_breakdown table stopped the figure build after E4's table had been written."""
    from margins.orchestrate import plan

    cfg = load_p11_config()
    responses = {(f"e4_{a.name}", m): "x.jsonl" for a in cfg.e4.arms for m in cfg.e1.models}
    default = plan.e4_process_commands(cfg, responses=responses)[-1]
    assert default.script == "make_figures" and "--table-dir" not in default.argv
    elsewhere = plan.Layout(results=Path("elsewhere"))
    moved = plan.e4_process_commands(cfg, responses=responses, layout=elsewhere)[-1]
    assert "--table-dir" in moved.argv


def test_the_notebook_builds_the_e1_prompts_before_the_gpu_check() -> None:
    """Found on the first real Kaggle run: verify-gpu times a batch of E1 prompts, which a fresh
    clone does not have, so it stopped before loading a model."""
    import scripts.build_r2_notebook as builder

    commands = builder.notebook_commands()
    build = next(i for i, c in enumerate(commands) if "build-prompts-e1-sampled" in c)
    verify = next(i for i, c in enumerate(commands) if "verify-gpu" in c)
    assert build < verify
    cells = ["".join(c["source"]) for c in json.loads(builder.render_notebook())["cells"]]
    prompts = next(i for i, c in enumerate(cells) if "build-prompts-e1-sampled" in c)
    gpu = next(i for i, c in enumerate(cells) if "run_p11.py verify-gpu" in c)
    assert prompts < gpu
