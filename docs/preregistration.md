# Pre-registration — Mapping the Margins

**Status: LOCKED. Tagged `prereg-v1` at the commit that introduced this line, 19 Sep 2026.**

The tag predates every full experimental run in this repository's history, which is the only
property that makes a pre-registration one. **No full run may be executed against an earlier
commit than the tag.** Versioned amendments with a logged rationale are permitted and expected
(§10); a pre-registration written blind is not honest, and an amendment log is.

Every locked number below was verified against the live configuration before tagging, by
recomputing it rather than by reading this document: the family size from the lattice, the
thresholds from `configs/stats/confirmatory_v1.yaml`, `min_cell_n` from
`configs/lattice/confirmatory_v1.yaml`. A pre-registration that disagrees with the code it
governs is worse than none.

Every number below is emitted by the pipeline and registered in `results/paper_numbers.json`. The
manuscript cites them by key.

---

## 1. Hypotheses

- **RQ2 (primary).** Open-weight LLMs exhibit **emergent** intersectional bias in open-ended
  generation: the interaction term of `harm ~ A + B + A:B` over an axis pair's order-2 cells is
  non-zero, over and above both constituent main effects.
- **RQ1.** A unified intersectional bias score can be computed comparably across two maximally
  different paradigms (templated `open_gen`, extractive `pair_pll`). At k = 2 this is answered by
  the effect vector plus an agreement statistic, with **no pooling, no tau-squared, no I-squared**.
- **RQ3.** Intersectionally targeted red-teaming surfaces biases static benchmarks miss.
- **RQ4.** Measurements are sensitive to seeds and decoding, quantified as variance and CI on
  every headline metric.

**Emergence is the interaction, never the level.** A cell with the highest mean harm in a table is
not evidence of emergence unless its departure from the additive fit is.

## 2. The estimand

Decided 18 Sep 2026, implemented in `margins.metrics`, unchanged since:

- **Identity link, OLS on the standardized harm score.** Scores are z-scored within
  benchmark x scorer x condition, so they are already on an SD scale.
- **`harm_std ~ A + B + A:B` over an axis pair's order-2 cells**, with **sum-to-zero (deviation)
  coding**. No reference level: age and gender have none.
- **Order-1 cells do not enter.** The lattice has no "no identity" cell, so an absent level would
  leave the (absent, absent) corner unidentified and confound absence with one fewer descriptor.
- **`harm_std` is the equal-weight mean of the harm type's scorers' standardized scores.** A
  response missing a scorer is **refused, not imputed**.
- **Standard errors are cluster-robust by ITEM (the prompt).** The sampled generations of one
  prompt are repeated observations, not independent draws.
- **Confidence intervals are a percentile cluster bootstrap over items, stratified by cell.**
- **Every level's coefficient is emitted**, so no cell is arbitrarily the one without a number.
  The emitted set is linearly dependent; §4 says how the FDR family accounts for that.

**The additive null (B2) is a presentation of the interaction, not an independent baseline.** The
genuinely independent check is the permutation test (§5).

## 3. Minimum detectable effect, power, and the design

**This section was rewritten on 19 Sep 2026 after a measurement contradicted its first draft.**
The full derivation, with the tables, is `docs/power_and_precision.md`.

### 3.1 The measured intraclass correlation

| harm type | rho | 95% CI |
|---|---:|---|
| regard | 0.35 | 0.29 to 0.43 |
| toxicity | 0.18 | 0.13 to 0.23 |

Measured, not assumed, on 1,728 sampled generations of the `socioeconomic_class x race_ethnicity`
pair through `meta-llama/Llama-3.1-8B-Instruct` at NF4, scored through the full P6 roster.
**Measured on ONE axis pair** and explicitly not assumed uniform across pairs; the design is
chosen against the **upper** confidence bound.

### 3.2 The consequence

Cell-mean variance is `rho/m + (1-rho)/(m*k)` for `m` prompts per cell and `k` samples per prompt,
which **floors at `rho/m`**. Prompts per cell bound precision; samples cannot. At the measured rho,
the original 9 prompts per cell could not reach 80% power on a 0.3 SD interaction at **any**
sampling budget.

### 3.3 The pre-registered minimum detectable effect

**0.5 SD**, re-targeted from 0.3 SD on 19 Sep 2026 because 0.3 SD is unreachable at accessible
compute at the measured rho. Alpha 0.05, power 0.80.

**Effects smaller than 0.5 SD are reported DESCRIPTIVELY**, with cluster-bootstrap confidence
intervals, clearly labelled, and carry **no significance claim**. They are not discarded.

### 3.4 The design this fixes

| quantity | value |
|---|---|
| prompts per cell | 27 (13 carrier-using frames x 2 carriers + 1 carrier-free frame) |
| sampled generations per prompt | 4, at distinct seeds |
| **`min_cell_n`** | **108 renderings per cell** |
| `renderings_per_cell` | 108 |
| prompt set | 2,970 prompts (`open_gen_v1.2`) |
| decoding | `sampled_v1`, temperature 0.7, top_p 0.9 |
| compute | ~95 GPU-hours across both confirmatory models |

`min_cell_n` and `renderings_per_cell` are the same number by construction: a rendering budget and
the minimum N it must clear are one decision.

## 4. The FDR family

**The pre-registered confirmatory axis pairs ARE the family.** Anything outside is exploratory,
reported descriptively, and carries **no p and no q**.

- **Members:** the `(|A|-1)(|B|-1)` **free interaction contrasts** of each confirmatory pair,
  crossed with benchmark x model x harm type. **Family size 168** = 42 terms x 2 models x 2 harm
  types, for the generative paradigm.
- **The benchmarks the family spans are DECLARED, not inferred** (amended 20 Sep 2026, §10).
  `configs/stats/confirmatory_v1.yaml` names them: `open_gen_templates` and `crows_pairs`. Until
  P2 the benchmark component was taken from whichever effects happened to reach
  `effects.parquet`, which made `family_size` a function of which files were on disk and left the
  subset guard unable to notice that an entire source never arrived. Now a declared source that
  P7 never attempted **raises**, and an attempted source that is not declared **raises** too.
  Each source brings its own harm types, so no member is invented for a test that was never run.
- **`crows_pairs` members are expected to be EXCUSED, and the corrected family size stays 168.**
  Measured on the pinned file: each CrowS item is annotated with one attribute, so the order-2
  grid is never populated and every confirmatory pair comes back not estimable. Its members
  therefore take their declared place in the family and are excused with P7's own reason. This is
  reported as a result about the paradigm, never as a null result about the model
  (`docs/methods.md` §2.5).
- **Why the free contrasts:** P7 emits every level's coefficient and that set is linearly
  dependent (gamma rows and columns sum to zero), so correcting over all of them would add tests
  carrying no information while costing power. The free contrasts are the design's own interaction
  parameters, so the subset is canonical rather than an arbitrary choice of which cells get a q.
  Main effects are B1's single-axis baseline, not the emergence estimand, and are excluded.
- **Correction:** Benjamini-Hochberg, `statsmodels.stats.multitest.multipletests(method='fdr_bh')`,
  alpha 0.05. **Benjamini-Yekutieli is reported alongside** as a sensitivity column; a claim that
  survives only under BH is reported as such. The PRDS justification is written into
  `configs/stats/confirmatory_v1.yaml` and the config **fails to load** without it.
- **Correcting a subset is an error, not a warning.** The family is declared from the lattice
  before any estimate is read; a member that does not arrive and is not excused by a P7
  not-estimable report raises.
- `family_size` and `correction` are stamped on every corrected estimate and recorded in the run
  manifest.

**Significance is set only by `StatsEngine`.** A test scans the source tree to enforce it.

## 5. The permutation baseline (B3)

Whole **items** are exchanged within `(template frame, stratum-axis value)` blocks. Items never
cross a block, so every replicate preserves each cell's item count and stays identified, and an
item's sampled generations always move together. The reported p is `(1 + count) / (B + 1)`, which
can never be zero. B = 2000, seed recorded.

**It is reported beside the GLM p and never enters the FDR family.** Freedman-Lane is implemented
as a calibration oracle; the pre-registered scheme is `within_stratum`, which destroys the
permuted axis's main effect and is therefore conservative for the interaction.

## 6. Scorer identity drift, and the downgrade

**Threshold 0.5, pre-registered** (promoted from P7's placeholder on 19 Sep 2026).

A harm type is **DESCRIPTIVE-ONLY** if any of its scorers, on any axis pair and any control
family, has an interaction magnitude on known-neutral text reaching half that scorer's own
interaction on the model's output. Such a scorer cannot distinguish emergence in the model from
emergence in the instrument.

The consequence is **enforced, not advisory**: the claim layer refuses to emit a significance
claim for a descriptive-only harm type, and `is_significant` raises rather than returning false,
so a report that forgets the gate crashes instead of printing a star. The ratio is recomputed from
the raw magnitudes and never inherited from P7's flag, which was computed under the placeholder.

## 7. Bootstrap resampling unit

The **item (prompt)**, resampled with replacement **within each cell**. Not configurable: a config
that could switch it to rows could quietly understate every variance.

## 8. Seeds and the decoding grid

- Confirmatory: `sampled_v1`, temperature 0.7, top_p 0.9, 256 new tokens, 4 distinct seeds per
  prompt. Distinct samples come from distinct seeds; the harness rejects a repeated seed.
- E4 sensitivity subset: temperatures {0.0, 0.7, 1.0} x at least 3 seeds, on a subset, never the
  whole corpus.

## 9. What would falsify, and what we will report either way

- A confirmatory pair whose interaction does not survive FDR is reported as not surviving. The
  post-correction significant count is reported beside the pre-correction count.
- A harm type downgraded to descriptive-only is reported as downgraded, with the drift ratio that
  caused it.
- If **no** harm type survives the drift rule, the primary estimand is not measurable with the
  available scorers and we say so. Go/no-go criterion 3 needs one harm type to survive.
- Underpowered cells are reported as underpowered and are never load-bearing.

## 10. Amendment log

| date | change | rationale |
|---|---|---|
| 18 Sep 2026 | Estimand fixed (§2) | P7 implementation; decided before any full run |
| 18 Sep 2026 | E1 grid: sampled, not greedy | Measured Kaggle timing pilot |
| 19 Sep 2026 | FDR family = free interaction contrasts, size 168 | The emitted gamma set is linearly dependent; correcting over it costs power for no information |
| 19 Sep 2026 | Drift threshold 0.5 promoted from placeholder to pre-registered and **enforced** | P9 owns the downgrade decision; P7 only produced the number |
| 19 Sep 2026 | **MDE re-targeted 0.3 SD -> 0.5 SD** | The measured rho makes 0.3 SD unreachable at accessible compute. Recorded rather than quietly absorbed |
| 19 Sep 2026 | Prompts per cell 9 -> 27; `min_cell_n` 60 -> 108 | Derived from the measured rho; the 96/60 placeholders are resolved |
| 19 Sep 2026 | Sub-0.5 SD effects are exploratory/descriptive | Consequence of the re-target, stated so it is not read as selective reporting |
| 20 Sep 2026 | **Open decision 7 resolved**: a `pair_pll` Prompt carries BOTH sequences on one record (`rendered_text_stereo` / `rendered_text_antistereo`) | Mirrors decision 6's one Response carrying both log-probabilities. Splitting a pair into two linked Prompts would have made `prompt_id` ambiguous as the metric layer's cluster key, since half a pair is not an item |
| 20 Sep 2026 | FDR family: the **benchmarks it spans are declared** in `configs/stats/`, not inferred from the artifacts present | The inferred form made `family_size` depend on which parquet files were on disk, and the "correcting a subset raises" guard could not detect a wholly absent source. A pre-registered number must not be a function of the file system |
| 20 Sep 2026 | New harm type `stereotype_preference` for the extractive paradigm, with two reductions (`pll_total`, `pll_length_normalized`) under a **separate roster** | A paired measurement has no external instrument, so framing constraint 6 is satisfied by two reductions that fail differently (length/frequency confound) rather than by two classifiers. Separate roster because `scoring_hash` rides in every `score_id` |
| 20 Sep 2026 | Extractive sequences are scored **raw**, with no chat template, under `pll_v1` | Matches how CrowS-Pairs log-likelihood is computed in the literature; wrapping would fold the per-model template into the measurement. The asymmetry with the generative path is recorded in `docs/methods.md`, not hidden |
| 20 Sep 2026 | **No identity-drift control for the extractive paradigm** | Structural, not an omission: the control scores known-neutral text through the same instrument, and a log-probability difference has neither text nor instrument |
| 21 Sep 2026 | **CrowS-Pairs' role is SCOPED: single-axis calibration + evidence that sentence-pair benchmarks structurally cannot support intersectional auditing. RQ1's cross-paradigm claim is scoped accordingly.** ✅ **Signed off by the supervisor, 21 Sep 2026.** | Measured on the pinned file, not discovered late: 1,352 kept items yield 198 order-1 cells and **3 order-2**, because each item is annotated with ONE attribute and an order-2 cell needs a second identity term present unchanged in BOTH sentences. Every confirmatory pair is therefore `estimable=False` and excused (§4), and the corrected family size is unchanged. RQ1's *interaction* comparison consequently runs **generation vs. red-team**; generation vs. extraction is compared at the single-axis / descriptive level and reported beside the structural finding. A result about the PARADIGM, never a null result about the model (`docs/methods.md` §2.5, `docs/PROJECT_CONTEXT.md` §1) |

| 21 Sep 2026 | **The red-team arm (P8) is DESCRIPTIVE: a subset at `powered=False`, carrying no p and no q. `redteam` stays OUT of `family_benchmarks`; the family stays 168 and `min_cell_n = 108` is unchanged.** | A powered red-team arm costs ~95 GPU-h — the same as the main sweep, against the ~15–25 GPU-h budgeted for it. Rather than spend the sweep's budget twice, red-team effects are reported with cluster-bootstrap CIs and no significance claim, and E3 is answered by the effect vector plus agreement against the static arm (§10's k = 2 machinery). **A budget-driven scope choice, stated as one** — never evidence that red-team effects are small. No re-run power analysis is needed, because the family size the design was derived against does not move |
| 23 Sep 2026 | **Clarification, not a design change: the four E1 seeds are `20260908, 20260909, 20260910, 20260911`** (the root config seed + 0..3), declared in `configs/experiment/p11_v1.yaml` before any run | §8 fixed "4 distinct seeds" without naming them. Naming them before the run, in a committed file the orchestrator reads, means no seed is chosen at the lab |
| 23 Sep 2026 | **Clarification: the E1 prompt set is `open_gen_v1.2` re-stamped `decoding_key: sampled_v1`** at build time (`build_prompts.py --decoding sampled_v1`), with byte-identical rendered text | The prompt-set config is stamped `greedy_v1`, and `decoding_key` is in the `prompt_id` preimage, so the sampled arm needs its own stamp. Overriding at build time (recorded in the manifest) rather than editing the config keeps the greedy set for E4's temperature-0 arm and avoids a second config file that could drift. The ids equal those the rho pilot measured |
| 23 Sep 2026 | **Clarification: harm scores are standardized PER MODEL.** Each confirmatory model's responses are scored in their own run, so standardization is within benchmark x scorer x harm type x condition *for that model*, and every gamma is in that model's own SD units. P9 then corrects both models' terms together as ONE family (size 168, unchanged) | §8 said "within source"; with two models it had to be decided whether models share a scale. Per model is the unit the power analysis was run in (rho and the SD were measured on one model's scores). Pooling would let the model with more harm variance set the unit for both, which was never simulated |
| 23 Sep 2026 | **Clarification: the descriptive red-team subset is sized by a FRAME CAP**, computed by `scripts/run_p11.py budget` from the lab card's MEASURED speed against the ~20 GPU-h red-team budget. Frames are chosen by a pure, content-ordered round-robin over (elicitation angle x target pair); each is rendered into all 80 target cells with ONE sampled generation per prompt. The arm stays `powered=False`, outside `family_benchmarks`, with no p and no q | Operationalizes the 21 Sep 2026 decision. One sample per prompt because prompts per cell bound precision and samples cannot (§3); the budget is therefore spent on frames. A cap chosen by content order and stratum, never by the analyst or the file order |
| 23 Sep 2026 | **Clarification: B3's exchangeability block is (template, CARRIER, stratum value).** §5's "template frame" is read as the rendering template, carrier included. B3 runs on the real run's item-level scores inside the stats stage and is required there (`run_stats.py` refuses to skip it without an explicit `--no-permutation`); its p is joined to P7's estimate to 1e-8 and reported beside the GLM p, never in the family | Since `open_gen_v1.2` each frame renders under two carriers. Blocking on the template alone would exchange a "... person" item with a "... individual" item, i.e. across a second source of variation that is not the permuted axis. With the carrier in the block, items in a block differ only in the permuted axis, which is exactly the null of no interaction. Until P11 no script ran B3 on real data, so every `p_permutation` would have been empty |
| 23 Sep 2026 | **Clarification: E4's sensitivity subset and arms.** Subset = the `socioeconomic_class x race_ethnicity` grid (16 cells x 27 prompts = 432 prompts), through BOTH confirmatory models. Temperatures {0.0, 0.7, 1.0}: **0.7 is E1's own `sampled_v1` x 4 seeds on these prompts, reused, not regenerated**; **1.0 is the new `sampled_t10_v1`, identical to `sampled_v1` except the temperature (top_p 0.9 held fixed)**, x 3 seeds (`20260908-10`); **0.0 is `greedy_v1` x ONE seed**. §8's "at least 3 seeds" applies to the SAMPLED arms; a greedy arm takes exactly one. ~13.8 GPU-h at T4 speed, approved into the P11 lab run | §8 left the subset unnamed. The pair rho was measured on is also the largest grid (4 x 4). Both models, because a sensitivity result on one invites the question of whether the other behaves the same. Only temperature varies between the sampled arms, so any difference is attributable to it. Greedy decoding is deterministic: three greedy seeds are three identical outputs, and presenting them as independent samples would understate the variance E4 exists to measure. The ">= 3 seeds" rule is enforced per arm kind in `configs/experiment/p11_v1.yaml`'s validator |
| 23 Sep 2026 | **Decision, before any E4 data exists: E4 scores are standardized SEPARATELY per (model x temperature arm)**, never pooled across arms within a model. The E4 analysis is built after the lab run | Temperature changes output variance, so pooling arms onto one scale would normalize away the very temperature effect E4 / RQ4 measures. Standardizing each arm on its own keeps each measurement independent, so a difference between temperatures is a property of the model rather than of the standardization. It extends the per-model rule (clarified above) one level down |
| 12 Oct 2026 | **Clarification: `bitsandbytes` 0.47.0 -> 0.50.2 in `uv.lock`, before any confirmatory generation.** The quantization scheme is unchanged (4-bit NF4, the recorded constant); only the library build that executes it moved | Found by the first `verify-gpu` on the lab card: the locked torch 2.14 links CUDA 13.0, and 0.47.0 ships no CUDA 13.0 binary, so NF4 could not load at all. 0.50.2 is the release contemporary with torch 2.14 that ships one; no other package moved. The installed version is recorded in every run manifest. The frozen Kaggle pilots (incl. the measured rho) ran under 0.47.0 and stay records of that |
| 12 Oct 2026 | **Clarification: `protobuf` added (and `sentencepiece` declared) in the Linux inference environment, before any confirmatory generation.** No package already in the lock moved | The pinned Mistral revision's fast tokenizer can only be built from its sentencepiece model, which needs both packages; without `protobuf` the second `verify-gpu` could not load the tokenizer at all. There is no alternative tokenizer path that could tokenize differently: without `sentencepiece` the load fails outright. Kaggle's image ships both, so this is the path the frozen timing pilot used |
| 16 Oct 2026 | **Decision, before any R2 data exists: R2 design and pass rule.** *Rerun:* the `socioeconomic_class x race_ethnicity` grid (432 prompts), both confirmatory models: (a) `sampled_v1` x E1's four seeds, compared against E1's own responses; (b) `greedy_v1` x one seed, compared against E4's greedy arm. Run on the Kaggle free tier (T4) as the primary evidence and on the lab A4000 as a same-hardware reference; each machine's comparison is reported separately. *Scale:* the rerun's raw scores are standardized with the ORIGINAL run's constants per (model, scorer, harm type, condition), so its estimates sit on the scale of the reported E1 estimates. *Verdicts:* P9 is re-run with the class x race family terms replaced by the rerun's estimates and every other member unchanged (family size 168, same drift downgrades). **PASS iff every class x race family term's rerun interaction estimate lies inside the original run's reported 95% CI (P7's cluster-bootstrap percentile interval) AND every ClaimGate verdict is identical.** Always reported, pass or fail: max and RMS |difference| in units of the original SE, and the greedy exact-token-match rate. If the locked environment cannot run on Kaggle, no dependency is changed: R2 runs on the lab GPU only and the reproducibility claim is scoped to a same-hardware rerun | The pre-registration named R2 but fixed neither its subset nor its criterion; both are fixed here so the criterion cannot be chosen after seeing the result. Original-run constants and family substitution make R2 test the numbers and verdicts the paper actually reports, not a refit of a subset. The same-hardware rerun supplies the run-to-run baseline without which a cross-hardware difference is uninterpretable. Decided with the human 16 Oct 2026 |

## 11. What is NOT pre-registered here

- The exploratory pair (`age x race_ethnicity`) and anything else outside the confirmatory set.
- The red-team prompt distribution (P8), which is scoped to the confirmatory pairs but whose
  generation is not fixed here. It is recorded instead in `configs/redteam/cell_targeted_v1.yaml`
  and in every red-team run manifest.
- **Any red-team significance claim.** `redteam` is absent from `family_benchmarks` in
  `configs/stats/confirmatory_v1.yaml` **by decision** (§10, 21 Sep 2026), so no red-team estimate
  carries a p or a q: the arm is descriptive, and E3 is answered by the effect vector plus
  agreement. The FDR machinery is built and tested against a fixture config, so admitting red-team
  later is possible — but it takes the family 168 -> 336 and moves the Benjamini-Hochberg threshold
  `min_cell_n = 108` was derived against (§3), making it a logged amendment that must ride with a
  re-run power analysis.
- Any claim about a second model family, additional benchmarks, or ablations A1-A3, all of which
  are out of scope unless ahead of schedule.
