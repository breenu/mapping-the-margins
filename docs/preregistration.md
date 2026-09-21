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
| 21 Sep 2026 | **CrowS-Pairs' role is SCOPED: single-axis calibration + evidence that sentence-pair benchmarks structurally cannot support intersectional auditing. RQ1's cross-paradigm claim is scoped accordingly.** ⚠ **Awaiting human / supervisor sign-off.** | Measured on the pinned file, not discovered late: 1,352 kept items yield 198 order-1 cells and **3 order-2**, because each item is annotated with ONE attribute and an order-2 cell needs a second identity term present unchanged in BOTH sentences. Every confirmatory pair is therefore `estimable=False` and excused (§4), and the corrected family size is unchanged. RQ1's *interaction* comparison consequently runs **generation vs. red-team**; generation vs. extraction is compared at the single-axis / descriptive level and reported beside the structural finding. A result about the PARADIGM, never a null result about the model (`docs/methods.md` §2.5, `docs/PROJECT_CONTEXT.md` §1) |

## 11. What is NOT pre-registered here

- The exploratory pair (`age x race_ethnicity`) and anything else outside the confirmatory set.
- The red-team prompt distribution (P8), which is scoped to the confirmatory pairs but whose
  generation is not fixed here. **Nor is red-team's FDR-family membership**: `redteam` is
  deliberately absent from `family_benchmarks` in `configs/stats/confirmatory_v1.yaml`, so no
  red-team estimate can currently carry a q. Admitting it is a logged amendment that must ride with
  a **re-run power analysis**, because 168 -> 336 changes the Benjamini-Hochberg threshold that
  `min_cell_n = 108` was derived against (§3). P8 builds and tests the machinery; the declaration
  is a P11 decision taken against real red-team fits.
- Any claim about a second model family, additional benchmarks, or ablations A1-A3, all of which
  are out of scope unless ahead of schedule.
