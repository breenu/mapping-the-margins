# CLAUDE.md — Mapping the Margins

Lean, always-on project memory for Claude Code. Deep reference (architecture, schemas, phase
plan, experiments, validation gates) lives in **`docs/PROJECT_CONTEXT.md`** — read it before
implementing a new phase.

---

## Status & next action

**P0 ✅ · P1 ✅ · P2 ✅ · P3 ✅ · P4a ✅ · P4b ✅ · P5 ✅ · P6 ✅ · P7 ✅ · P8 ✅ · P9 ✅ · P10 ✅ complete.** P0: environment,
`margins.repro`, `margins.config` + run manifest, CI. P1: `margins.schemas` (frozen records) +
`margins.interfaces` (Protocols), walking skeleton, schema field-freeze test. P3:
`margins.lattice`. P4a: `margins.prompts` — `TemplateRenderer`, the leakage guard, the phrase
review. P5: `margins.inference`. P6: `margins.scoring` — the 4-scorer roster, standardization with
an executable sign contract, triangulation + agreement, the identity-drift control. P7:
`margins.metrics` — the emergence estimator, item-clustered SEs, the cluster bootstrap, the drift
comparison. P2/P4b: `margins.benchmarks` (the CrowS-Pairs loader + screening),
`margins.prompts.extractive`, `margins.scoring.paired` — the second paradigm, end to end. P8:
`margins.redteam` — cell-targeted frame generation, the QC screen, the on-target scaffold.

**P9 ✅ `margins.stats` — DONE 19 Sep 2026, and it changed the design.** All four gates green on
`win_amd64` and in CI on both OSes (`core (ubuntu-latest)` and `core (windows-latest)`).

**P2 + P4b ✅ the EXTRACTIVE vertical — DONE 20 Sep 2026, and it produced a measured result.**
CrowS-Pairs pinned, screened, rendered, scored and joined to the existing metric + stats layers.
**Open decision #7 is CLOSED.** The headline finding: **CrowS attests 3 order-2 cells out of 1,352
kept items**, so every confirmatory pair is not estimable and is excused by P9. That is a fact
about what sentence-pair benchmarks can support, not a null result about the model -- see
"P2 + P4b as built" and `docs/methods.md` §2.5.

**✅ PRE-REGISTRATION TAGGED: `prereg-v1` at commit `067b50e`, 19 Sep 2026.** Every locked value
was verified against the live configs by RECOMPUTATION before tagging, not by reading the
document: the 168-term family from the lattice, the thresholds from `configs/stats/`, `min_cell_n`
from `configs/lattice/`. **NO FULL EXPERIMENTAL RUN MAY BE EXECUTED AGAINST AN EARLIER COMMIT.**
Amendments are permitted, versioned and logged in §10 of `docs/preregistration.md`.

**P8 ✅ automated intersectional RED-TEAMING — DONE 21 Sep 2026, and it found a P9 bug.**
`margins.redteam` generates cell-targeted FRAMES through a pinned red LM and renders them across
the confirmatory cells via the **unchanged** `TemplateRenderer`. No schema change, no new scoring
or metric path. See "P8 as built". **It also exposed a real P9 defect that would have crashed the
first full run** — see the ⚠ box below.

**P10 ✅ visualization + reporting — DONE 21 Sep 2026.** `margins.viz` + `margins.report`, seven
figures, one command, every drawn number audited against a committed artifact. See "P10 as built".
All four gates green on `win_amd64` and **in CI on both OSes 22 Sep 2026** (`core (ubuntu-latest)`
and `core (windows-latest)`, run `35694991682` at commit `0f887b6`) — including the new
`make_figures.py` step, so the one-command regeneration gate is confirmed on both platforms and
not only on the dev box. **965 tests.**

**NEXT TASK: P11 — experiments.** Then P12 (paper), P13 (release). **Before any full run, honour
the two things decided 21 Sep 2026 below** — the multi-rendering smoke run that exercises the P9
correction path, and the red-team arm staying `powered=False`.

### 📌 DECIDED 21 Sep 2026 — two things that bind BEFORE the P11 full run

1. **A deliberate end-to-end smoke run on MULTI-RENDERING data, exercising the P9 correction
   path.** Not the existing dev artifact: it has **one rendering per cell**, so every pair comes
   back not estimable and `run_stats.py` correctly reports "nothing to correct" — meaning the
   correction path is never executed. **That is exactly where the family-membership bug hid**
   (see the ⚠ box below), and it survived four green phases because of it. The smoke run must
   produce at least one *estimable* pair so `attach_wald_p` → `correct` → `ClaimGate` actually
   run on real P7 output. §11's golden rule, made specific: the small-scale smoke test catches
   ~90% of pipeline bugs at ~1% of the compute, but only for the paths it reaches.
2. **The red-team P11 arm stays DESCRIPTIVE / subset — `powered=False`, not fully powered.** A
   powered red-team arm costs ~95 GPU-h, the same as the main sweep and far above the ~15–25
   GPU-h budgeted for it. So red-team cells are reported descriptively with CIs and carry no
   significance claim, and `redteam` therefore stays **out** of `family_benchmarks` — which means
   the 168-term family and the derived `min_cell_n = 108` are unchanged, and no re-run power
   analysis is needed. E3 is answered by the effect vector plus agreement against the static arm.

### ✅ SIGNED OFF — the CrowS / RQ1 reframing (registered + approved 21 Sep 2026)

**Approved by the human 21 Sep 2026. This is the scoping the paper is written against; do not
re-open it and do not re-litigate the wording.**

The P2 measurement (3 order-2 cells out of 1,352 kept items) is a **scoping decision**, not a
caveat: CrowS-Pairs' role is **(a) single-axis calibration** and **(b) positive evidence that
standard sentence-pair benchmarks structurally cannot support intersectional auditing**, and
**RQ1's cross-paradigm claim is scoped accordingly** — the *interaction* comparison is generation
vs. red-team, while generation vs. extraction is compared at the single-axis / descriptive level
beside the structural finding.

Written into `docs/PROJECT_CONTEXT.md` §1 ("CrowS-Pairs' measured role"), §11's P2 gate row, §12's
E2 bullet, and `docs/preregistration.md` §10 + §11 — all as approved, unchanged. **It must never be
reported as a null result about the model:** CrowS-Pairs cannot express an intersectional effect,
which is a different statement from not finding one.

### ⚠ THE P9 FINDING — read this before touching the design

The power analysis was meant to fix `min_cell_n`. It instead falsified the design twice, and both
corrections are load-bearing. **Do not re-litigate any of this; it is measured.**

1. **Precision is bounded by PROMPTS PER CELL, not renderings per cell.** Cell-mean variance is
   `rho/m + (1-rho)/(m*k)` for `m` prompts and `k` samples, and it **floors at `rho/m`**. Sampling
   more generations from the same prompts hits a ceiling no budget beats. Executable: at 9 prompts
   and rho 0.4, simulated power at 20 and at 100 samples differs by less than 0.05.
2. **rho was MEASURED (do not re-run, do not re-assume):** regard **0.35** (CI 0.29–0.43),
   toxicity **0.18** (CI 0.13–0.23), from 1,728 sampled generations of the class × race pair
   through Llama-3.1-8B-Instruct at NF4. It is **one pair**, the demanding 4×4, and is NOT assumed
   uniform across pairs. The dev run could never have supplied it: one response per prompt
   contains no within-prompt variance at all.
3. **The MDE was RE-TARGETED from 0.3 SD to 0.5 SD**, because 0.3 SD is unreachable at accessible
   compute at that rho. **Sub-0.5 SD effects are EXPLORATORY/DESCRIPTIVE** — reported with CIs,
   clearly labelled, no significance claim. They are not discarded.
4. **The design: 27 prompts/cell × 4 samples = `min_cell_n` 108 renderings/cell, ~95 GPU-h.** The
   96/60 stale markers in `configs/lattice/confirmatory_v1.yaml` are RESOLVED.
5. **THE FIELD RESULT, a first-class paper claim:** a conventional 9-prompt design reaches only
   **18–50% power on a MODERATE 0.5 SD interaction** at the measured rho. Null results from such
   designs are close to uninformative; a single-template design cannot estimate a cluster-robust
   SE at all. Written up in **`docs/power_and_precision.md`**, keyed in
   `results/paper_numbers.json` under `field.power_at_9_prompts.*`.

### ⚠ THE P9 BUG P8 FOUND — fixed, and worth understanding

**`correct()` selected its inputs by axis PAIR, not by family MEMBERSHIP.** A confirmatory pair's
fit emits its main effects and *every* interaction cell as well as its `(|A|-1)(|B|-1)` free
contrasts — 19 terms per pair on the shipped lattice, of which **6** are family members. All 19
were offered for correction, and `FamilyPlan.validate` then raised `UnknownFamilyMemberError`.

**It would have fired on the first full E1 run, with ~95 GPU-h already spent.** Nothing caught it
earlier because the dev artifact has one rendering per cell, so P7 found every pair *not estimable*
and `run_stats.py` correctly reported "nothing to correct" — the correction path had never once run
on real P7 output. P8's integration test is the first thing that fed it any.

**Fix:** `FamilyPlan.is_family_member(key)` (pair **and** term) now selects who gets a p and a q;
`is_confirmatory` survives as the coarser pair-level question. Non-member terms pass through
**descriptively, with their CIs, carrying no p and no q** — which is what `docs/preregistration.md`
§4 and the "dependent gammas are still reported" line in "P9 as built" always said should happen.
Pinned by `tests/test_fdr_correction.py::test_a_non_member_term_of_a_confirmatory_pair_passes_
through_descriptively`. **The lesson is the §11 golden rule restated: run the real artifact through
the real stage.** Both P6's Krippendorff blow-up and this were found that way, not by a test.

### P10 as built (the FIGURES + TABLES — for P11/P12 to consume)

- **NO schema change, no contract change, no new dependency.** `matplotlib`/`seaborn` were already
  core. P10 reads committed artifacts and writes figures and tables; it computes no estimate and
  sets nothing.
- **One command: `uv run python scripts/make_figures.py`** → seven figures as vector **PDF + SVG**
  in `figures/` (committed; `figures/*.png` is gitignored and `save_figure` REFUSES a raster
  format, so a rasterised headline figure cannot enter the repo). CI runs the command too.
- **⚠ THE TRACEABILITY MECHANISM, and it is not a convention.** A figure obtains numbers only
  through `margins.report.source.NumberSource`, which hands out `Traced` values and keeps a
  **ledger**. `margins.viz.trace.audit_figure` then walks the RENDERED ARTISTS — line data, bar
  extents, collection segments, heatmap arrays, titles, axis labels and annotations — and reports
  every number the ledger does not account for. `build_figure` runs the audit *inside* the build
  and raises, so an untraceable figure is never written. **The audit is proven able to fail**
  (`tests/test_viz_traceability.py`): a hand-typed coordinate, a hard-coded reference line and an
  annotation that drifted from its table are each caught. Exemptions are declared per figure as
  `structural={value: "why this is not a measurement"}` — a mapping, because an exemption that
  does not say why is how an audit becomes a formality.
  - A number computed from traced inputs (a row count, a column max) goes through
    `NumberSource.derived(..., how=...)`, which lands the derivation in the ledger key.
  - `NumberSource.number()` RAISES on a missing or non-numeric key; there is no fallback.
- **⚠ THE CLAIM BINDING: a figure cannot print a star the gate would refuse.** `margins.viz.claims`
  runs the REAL `ClaimGate` at figure time and **raises if its verdict disagrees with the one P9
  recorded** — believing the artifact would be circular. The trap is live in the shipped fixture:
  the downgraded harm type keeps q values below alpha (it WAS corrected; only the claim is
  refused) and gets no star. `test_the_emergence_figure_stars_only_permitted_effects` counts the
  markers actually drawn and holds them to what the gate permits.
- **⚠ THE AST SCAN CAUGHT A REAL P10 BUG.** `effect_from_row` originally passed
  `family_size=1, correction="fdr_bh"` so the record would validate — the figure layer inventing
  significance metadata, exactly the invariant-3 violation
  `test_only_the_correct_module_sets_q_family_size_or_correction` exists to catch. **Fix:**
  `family_size` and `correction` are now COLUMNS of `emergence_effects`, and the figure layer
  *deserializes* what P9 wrote rather than authoring it. Third time a P9-era guard has paid off.
- **Emergence-is-not-level is enforced, not captioned.** No `CellStat` mean reaches figure 1; the
  heatmap is a γ table and `test_the_gamma_grid_the_heatmap_draws_sums_to_zero` checks it sums to
  zero along every row and column, which a table of levels never does. The single-axis table
  deliberately excludes order-2 cells for the same reason.
- **Greyscale legibility is structural, and the palette docstring says so with numbers.** Okabe-Ito
  reordered; worst pair ΔE 26.4 normal / 20.7 protan / 18.1 deutan / **16.2 tritan** (JND ≈ 2.3),
  recomputed by the tests via Machado 2009 matrices + CIE76, no new dependency. But six hues
  separated in HUE are not separated in LUMINANCE — the closest two are **0.011** apart — so the
  real rule is `Encoding`: every series carries a marker, hatch or linestyle, colour-only
  construction RAISES, and a test asserts the non-colour channel alone identifies every series in
  every figure. `test_the_palette_alone_is_not_greyscale_sufficient` asserts the honest half.
- **Byte-deterministic within a platform**, tested: `svg.hashsalt` pinned, `svg.fonttype="none"`,
  savefig metadata drops `CreationDate`/`Date`, and **SVG is rewritten with LF** because
  matplotlib opens it in text mode and a Windows run would otherwise dirty the worktree every
  time. **Cross-OS byte identity is NOT claimed** (PDF font subsets), so CI regenerates but does
  not diff — the same scoping invariant 5 already uses for bit-identity.
- **Seaborn is deliberately UNUSED for drawing**, though it stays a declared dep: it ships no type
  information (`mypy --strict` covers `src`) and its themes shift between minor versions, which is
  the invisible drift the determinism test exists to catch. The rcParams are the theme, written out.
- **`results/` gained two directories.** `results/tables/` holds the real committed result tables
  (CSV **+** LaTeX, booktabs, byte-deterministic, declared column-by-column by a `TableSpec`);
  `results/fixtures/` holds the synthetic stand-ins. **Every table carries a `data_status` column**
  — machine-readable, in the row, not a filename convention — and every figure drawn from fixture
  rows is banner-stamped on its face. `scripts/make_figures.py` prefers `results/tables/` when it
  exists and says which it used.
- **FIXTURE DATA, stated plainly: six of seven figures currently show synthetic rows.** Only
  `fig3_field_underpowered` is real (measured rho + the power simulation, read by key). Everything
  *structural* in the fixtures is real — the axis pairs and their levels, the
  confirmatory/exploratory split, the scorer roster, the model pins, the frame names, the
  pre-registered thresholds, and CrowS's measured order-2 emptiness — only the estimates are from
  a seeded RNG (blake2b-keyed, NOT `hash()`, which is salted per process). **No fixture value is
  registered in `results/paper_numbers.json`,** asserted by a test.
- **`results/paper_numbers.json` gained 9 keys, all real:** `crows.{rows_read,kept,excluded,
  order_1_items,order_2_items,no_cell_items,order_2_cells_populated}` from a re-run of
  `scripts/build_crows_items.py` (which now registers them), plus `fdr.alpha` and
  `drift.descriptive_only_ratio` read off the stats config by `scripts/emit_power_numbers.py`.
- **Figure 4 REFUSES to draw an extractive interaction.** If a `crows_pairs` row ever arrives
  claiming `estimable=true`, `build` raises rather than plotting it: the slot carries the
  estimator's machine-readable reason, because a zero with an interval would report a null result
  about the model where the truth is that the instrument cannot express the effect.
- **Figure 5 REFUSES to promote the red-team arm.** A red-team row carrying a q or a claimable
  flag raises. The arm is descriptive by the 21 Sep 2026 budget decision, and the figure says that
  is a scope choice and not evidence the effects are small.
- **`scripts/build_result_tables.py` exists and is TESTED against real record types** even though
  no real run exists: `margins.report.build` is pure, and the tests feed it genuine
  `EffectEstimate`/`FitTerm`/`SignificanceClaim`/`CellStat` objects. An estimate with no matching
  triangulated fit term RAISES — its shape would be missing and it would vanish from the heatmap.
- **Artifacts:** `figures/*.{pdf,svg}` + `figures/README.md`, `results/fixtures/*.{csv,tex}` +
  a README, and (after P11) `results/tables/*.{csv,tex}`.
- **One stale run manifest was REMOVED, deliberately.** Re-running `scripts/build_crows_items.py`
  to register the `crows.*` numbers produced a second manifest for the same ingestion, because
  `ExperimentConfig` gained `redteam_name` at P8 and the config hash moved. The two recorded
  **identical `input_checksums` and identical config values** — the same run under two schema
  versions — so `build-crows-items-…-3344af7f11ed.json` was deleted (it survives in git at
  `ba38e28`) and `…-d50690c0148c.json` is the one `results/paper_numbers.json` names. **The guard
  that makes this safe already existed:** `test_manifest_paths_resolve_to_real_manifests` fails if
  any paper number cites a manifest absent from the checkout, so deleting the wrong one is a red
  suite, not a silent orphan.

### P8 as built (the RED-TEAM vertical — for P10/P11 to consume)

- **NO schema change.** `Prompt`, `Response`, `ItemScore` are untouched: the E3 tag is the existing
  `benchmark` field, carried from `Item.benchmark = "redteam"` through the `--benchmark` flag that
  `run_scoring.py`/`run_metrics.py` already had. The only contract change is a `redteam` block on
  `RunManifest`, logged in `tests/test_manifest.py` exactly as the P6/P7/P9 blocks were.
- **DECIDED with the human, 21 Sep 2026: the red LM emits slot-bearing FRAMES, not finished
  per-cell prompts.** Decisive reason: **a frame is one item cluster present in every cell of a
  pair's grid**, where a per-cell prompt is a singleton — the exact structure that made CrowS not
  estimable. Frames are what make a red-team interaction estimable at all. Second reason: a frame
  goes through the **unchanged** P4a guard.
- **⚠ THE RIGOR POINT: every frame is rendered into EVERY target cell, not only its own.** The
  elicitation target survives as provenance and as the on-target audit unit, nothing more.
  Confining a frame to its target would confound "frame targeted at X" with "cell X" — every cell
  measured through its own private frames, so any difference between frames (length, register,
  topic) lands on that cell's mean and reads as an identity effect. **The estimator would report an
  interaction the frame distribution produced.** Written into `frames.py`, `methods.md` §2A.2, and
  the generated report.
- **TWO PROMPT POPULATIONS, never to be confused.** *Elicitation* prompts address the red LM,
  deliberately NAME the target identity (that is the cell-targeting), are never scored, and carry
  `benchmark="redteam_elicitation"`. The leakage guard is deliberately NOT applied to them — it
  exists to keep identity out of everything but the slot of an *audit* prompt, and applying it to
  an instruction whose job is to name an identity is a category error. *Audit* prompts are what the
  target model sees, are guarded, and carry `benchmark="redteam"`.
- **The guard is AUTHORITATIVE; the filter is a pre-screen.** Every surviving frame is
  trial-rendered into every target cell through `TemplateRenderer`; one that raises is dropped with
  reason `leakage_guard` rather than taking the run down. `tests/test_redteam_generator.py` proves
  the guard is load-bearing **on its own** by calling `trial_render` directly on a leaking
  candidate, bypassing the pre-screen entirely.
- **The one P4a seam change is small and the guard is untouched.** A `RenderRules` Protocol names
  the five things `TemplateRenderer` actually reads off a config, plus a new `item_benchmark`
  property; `_check_item`/`template_key` read that instead of the module constant.
  `item_benchmark` is a **property, not a config field**, so no YAML can relabel the red-team arm
  as the static one. P4a behaviour is unchanged in every respect.
- **Carriers and the article rule are INHERITED from `open_gen_v1`, never re-declared**, and the
  base set's own `template_version` is recorded. E3 compares matched cells, so a red-team carrier
  of its own would confound "red-team frame" with "different carrier noun".
- **Eleven machine-reasoned filter rules + two-stage dedup, all PURE** (no I/O, RNG, clock or order
  dependence, like `benchmarks.screening`): `unparseable`, `slot_count`, `too_short`, `too_long`,
  `names_identity_term`, `names_axis_word`, `sentence_initial_slot`, `template_syntax`, `refusal`,
  `copies_fewshot`, `duplicate`/`near_duplicate`, plus `leakage_guard`. **Dedup is not hygiene:**
  duplicate frames are duplicate ITEM CLUSTERS, and a cluster-robust SE over them is too small in
  the anti-conservative direction. `rapidfuzz` was already a core dep.
  - **`template_syntax` and a `TemplateError` backstop exist because a frame IS compiled as Jinja.**
    `{{ oops }}` raises `UndefinedError` and a stray `{{` raises `TemplateSyntaxError`, neither of
    which is a `ValueError` — one malformed candidate would otherwise have killed a run whose GPU
    time was already spent.
  - **Dedup carries the group's other targets forward** (`Candidate.also_targeted`). Found by
    reading the generated report: collapsing duplicates erased how many distinct targets proposed
    a frame, so the "generic frame" count was structurally always zero. It is unrecoverable after
    the collapse, so it is preserved during it.
- **On-target scaffold, the P6 calibration pattern exactly.** Deterministic sample **stratified by
  target pair** (so the non-gender × non-gender pairs are not estimated from leftovers), seeded
  through `set_global_seed`, `on_target_metrics` reporting **`None` not `0.0`** on an empty
  denominator. The labeler is shown the frame **rendered for its own target**, because "is this
  on-target" is unanswerable from a template with a hole in it. Nothing is auto-labeled.
  **`on_target_report.md` is unwritten until real labels exist** — the reported-% gate clause is
  the human's, like P6's precision/recall.
- **Red-team is NOT in the shipped FDR family — SETTLED 21 Sep 2026, not deferred.** A powered
  red-team arm (27 frames/cell × 4 samples × 92 order-2 cells × 2 models) costs **~95 GPU-h, the
  same as the main sweep**, against the ~15–25 GPU-h budgeted. So the P11 arm runs as a **subset at
  `powered=False`**: red-team effects are descriptive, with CIs and **no p and no q**, `redteam`
  stays out of `family_benchmarks`, the family stays **168**, `min_cell_n = 108` is unchanged and
  **no re-run power analysis is needed**. E3 is answered by the effect vector + agreement against
  the static arm — the same k = 2 machinery §10 already specifies. The integration test proves the
  FDR machinery against a FIXTURE config (so it is ready if the budget ever changes), and
  `test_the_shipped_family_does_not_yet_declare_red_team` fails with the full explanation if anyone
  adds it. **State it as a budget-driven scope choice, never as evidence red-team effects are
  small.**
- **Generator pinned and recorded:** `meta-llama/Llama-3.1-8B-Instruct` @ `0e9e39f2…`, under the new
  `configs/decoding/redteam_v1.yaml` (sampled, temp 0.9 — diversity is the product, and duplicates
  are discarded anyway). **Stated limitation:** it shares a family with one confirmatory target; the
  same frame set runs through both, so neither faces prompts the other did not, and swapping in
  Mistral is one config line. In `methods.md` §2A.3.
- **Three elicitation angles**, not one: `disparate_treatment`, `consequential_decision`,
  `narrative_completion`. One instruction is one prompt distribution wearing a label.
- **Artifacts** under `data/redteam/` (`frames_v1.jsonl`, `rejected_frames.csv`,
  `redteam_report.md` with a measured **per-pair coverage table**, `on_target_sample.jsonl`) plus
  `data/prompts/<set>.jsonl`. All byte-deterministic, verified by regeneration. No new dependencies.

### P2 + P4b as built (the EXTRACTIVE vertical — for P8/P10 to consume)

- **CONTRACT CHANGE: `Prompt` gained `rendered_text_stereo` + `rendered_text_antistereo`**
  (`str | None`, null on `open_gen`). **This closes open decision #7** — see the decision table
  below. `tests/test_schema_field_freeze.py` updated with a dated comment, as `prompt_id` was.
  Unlike `Response`, `Prompt` carries `task_type`, so "both sequences exactly when paired" is a
  **single-record** validator: a paired prompt must carry both, both non-empty and distinct, and
  `rendered_text == ""` (the two sequences ARE the input, one per forward pass — mirroring
  `Response.text=""`). `ExperimentConfig` gained `benchmark_name` (optional, `None` by default:
  most runs ingest no benchmark).
- **The pin was RESOLVED, never authored.** `nyu-mll/crows-pairs` @
  `8aaac11c485473159ec9328a65253a5be9a479dc`, sha256 `dfb36986ce05…`, **1,508 rows**, CC-BY-SA-4.0.
  `scripts/fetch_crows_pairs.py` has the same three modes as `fetch_lexicons.py` and STOPS on a
  404 or a digest mismatch. **The licence was READ off the upstream README at that commit** — the
  repo ships no LICENSE file, so the GitHub API reports none, and the README sentence is the
  authority. Recorded in `crows_pairs.py` so a reproducer can check the same file.
- **⚠ THE MEASURED FINDING — CrowS cannot estimate an interaction, and that is a RESULT.**
  1,508 rows → **1,352 kept, 156 excluded**; of the kept, **198 order-1 cells and 3 order-2**.
  Each CrowS item is annotated with ONE attribute, so an order-2 cell needs a second identity term
  present *unchanged in both* sentences. The estimator fits order-2 only and needs every grid cell
  non-empty, so **every confirmatory pair comes back `estimable=False`** with an explicit
  "12 empty cells" reason, and P9 excuses them. **Report it as a fact about the paradigm — never
  as a null result about the model.** It is the extractive analogue of the P9 field result and is
  written up in `docs/methods.md` §2.5.
- **Attestation is LITERAL, and there is no alias table.** `poor` is not read as `working-class`,
  `Mexican` is not read as `Hispanic`. Surface matching is case-insensitive on word boundaries
  (the lattice pins `Black`/`Deaf`; CrowS writes "black kids"). The cost is a low yield; the
  benefit is that **no cell assignment in this pipeline was authored**. 761 kept items attest no
  lattice value and 390 are out-of-lattice — both **kept and marked**, never dropped, never celled.
- **Attribute → axis, all nine declared; an undeclared `bias_type` RAISES.** `race-color`→
  `race_ethnicity`, `gender`→`gender_and_sex`, `age`→`age`, `disability`→`ability`,
  `socioeconomic`→`socioeconomic_class`; `religion` / `nationality` / `sexual-orientation` /
  `physical-appearance` → **out-of-lattice with a required note** (`axis: null` without a reason
  fails to load, because it is otherwise indistinguishable from an unfinished mapping).
- **THE SIGN HAZARD IS REAL AND RESOLVED ONCE.** **218 of 1,508 rows are `antistereo`**, meaning
  upstream's `sent_more` is the *anti*-stereotypical member. `CrowsRow.sent_stereo` is the only
  place that swap is expressed; nothing downstream reads the raw columns for direction.
- **Screening is deterministic and pure** — no I/O, no RNG, no clock, no order dependence — and
  every exclusion carries one machine-readable reason. Two artifacts beyond the gate:
  `excluded_items.csv` and a **coverage report**, because the order-2 yield had to be *measured*.
  A **grammatical-concord allowance** is declared in the config, counted, and reported: obligatory
  `she`→`he` concord is not a second contrast. It admitted **68 rows**, so a stricter reading would
  record 221 `not_minimal_pair` rather than 153. Honorifics are deliberately excluded from it.
- **Gate P4b is enforced on every emission, not by a checker a script must remember to call.**
  `ExtractiveRenderer._check_attestation` verifies the claimed cell against the **TEXT**, not
  against the payload that claims it (a payload is an assertion), and additionally requires an
  order-2 cell's second value to appear in **both** sentences. Cells are **looked up** in
  `lattice.cells()`, never constructed. Byte-identical regeneration verified.
- **Item ids use `/`, not `::`** (`crows_pairs/0000`), matching P4a's `open_gen_templates/describe`:
  `prompt_id` joins four fields with `::`, and `make_extractive_prompt_id` now raises if a field
  contains it. **Both sequences enter the digest preimage** — `make_response_id` does not cover the
  scored texts, so `prompt_id` is the only place a pair's identity can live.
- **`pair_pll` scoring: a NEW harm type, a SEPARATE roster, and no drift control.**
  `stereotype_preference` with two reductions — `pll_total` and `pll_length_normalized` —
  under `configs/scoring/crows_pll_v1.yaml`. **Separate roster because `scoring_hash` rides in
  every `score_id`**: folding these into `default_v1` would change the id of every existing
  `open_gen` score for rows whose value did not change. A paired measurement has **no external
  instrument**, so framing constraint 6 is met by two reductions that *fail differently* (the
  length/frequency confound) rather than two classifiers; their α was **0.694** on the dev smoke,
  i.e. genuinely informative. **Length is in whitespace WORDS, not tokens** — per-sequence token
  counts are not on the `Response` — and that limitation is stated in `docs/methods.md`.
  **No drift control exists for this paradigm**, structurally: the control scores known-neutral
  TEXT through the same INSTRUMENT and a log-prob difference has neither. The run prints that.
- **`verify_paired_sign` needs NO model.** A text probe checks an instrument; a paired scorer's
  polarity is a property of *our arithmetic*, so the probe supplies log-probabilities and runs on
  the torch-free core env. Tests cover both a reversed `sign` and a reversed subtraction.
- **`scorer_from_spec` is now EXHAUSTIVE** — it was an `if/else` that silently routed any new
  mechanism to `HFClassifierScorer`. A likelihood spec carries **no pin** (its instrument is the
  model under test, pinned by `ModelSpec`), enforced rather than tolerated.
- **Three minimal fixes at the join, each noted:**
  1. **`metric.compute_result` now SKIPS a null `cell_id`** instead of raising. A null cell is a
     legal documented state ("the source attests no cell"); the generation path never produced one,
     the extractive path produces 1,151. Raising made an expected property of paradigm two look
     like a data error *and* took down the open_gen rows in the same file. They count into
     `n_skipped_rows` beside the drift rows.
  2. **The FDR family's benchmark dimension is now DECLARED** (`family_benchmarks` in
     `configs/stats/`), and comes from `fit_reports` (what P7 *attempted*) rather than `effects`
     (what it *managed*). This closes a real hole: the expected set used to be built FROM the
     offered effects, so a wholly absent source was undetectable and P9 would quietly correct over
     a *smaller* family — i.e. make more discoveries — on a `family_size` that was really a
     function of which files were on disk. Declared source missing → raises; undeclared source
     present → raises. **Each source brings its own harm types**, so no member is invented.
  3. `run_scoring.py` / `run_metrics.py` took `--benchmark`; the hard-coded
     `BENCHMARK = "open_gen_templates"` would have pooled the two paradigms into one z-score.
- **`Triangulator` is typed against a new `RosterScorer` Protocol**, satisfied by both
  `ResponseScorer` and `PairedScorer`, so the second paradigm joins on the shared contract rather
  than on a second code path. `standardize_within_benchmark`, `estimator`, `design` and `correct`
  needed **no changes at all** — which is the unified-metric-layer claim doing its job.
- **Artifacts** under `data/benchmarks/crows_pairs/` (`items_v1.jsonl`, `excluded_items.csv`,
  `screening_report.md`, `coverage_report.md`) and `data/prompts/crows_pll_v1.jsonl`. All
  byte-deterministic, all verified so. `docs/methods.md` is new and carries the causal-model
  caveat. No new dependencies.

### P9 as built (for P2/P10 to consume)

- **CONTRACT CHANGE: `ExperimentConfig` gained `stats_name`, `RunManifest` gained a `stats`
  block.** `tests/test_schema_field_freeze.py` and `tests/test_manifest.py` updated deliberately.
  `margins.metrics.metric._triangulate` was renamed to the public `triangulate` (pure rename) so
  P9 reuses the triangulation rule instead of duplicating it.
- **FDR family = the FREE interaction contrasts of the confirmatory pairs. Size 168** (42 terms ×
  2 models × 2 harm types). The emitted gamma set is linearly dependent, so correcting all 320
  interaction cells would add tests carrying no information; main effects are B1's baseline and
  are excluded. Configurable via `family_terms` so the choice is visible, defaulted to
  `free_interaction`.
- **`correct()` is the ONLY place `q`/`family_size`/`correction` are set**, it is pure (no I/O, no
  RNG, no clock), and `tests/test_fdr_correction.py` enforces that with an **AST scan of `src/`**.
- **Correcting a subset RAISES.** The family is declared from the lattice before any estimate is
  read. A pair P7 reported not estimable is *excused*, carries P7's own reason, and shrinks the
  family; a member that merely went missing is an error.
- **p-values:** two-sided cluster-robust Wald t, **df = item clusters − 1**, taken from the
  *triangulated* fit report. Not `n_obs`: df does not grow with samples. **Exploratory pairs get
  no p and no q** — a bare p invites exactly the reading the split exists to prevent.
- **`check_effects_match_fits` asserts every effect equals its joined FitTerm to 1e-12**, so a
  mis-join cannot silently supply a wrong df and therefore a wrong q.
- **The permutation test (B3) is built** — P7's deferred gate clause is closed. Whole items
  exchange within `(frame, stratum value)` blocks; `p = (1+c)/(B+1)`, never zero; reported beside
  the GLM p and **never in the FDR family**. Freedman-Lane ships as the calibration oracle.
- **The drift downgrade is ENFORCED at the pre-registered 0.5**, recomputed from raw magnitudes
  and never inheriting P7's placeholder flag; disagreements with P7 are reported, not swallowed.
  **`ClaimGate.is_significant` RAISES for a refused effect** rather than returning False, so a
  table that forgets the gate crashes instead of printing a star. A refusal is a ROW, never an
  absence.
- **`results/paper_numbers.json` is emitted and committed**, byte-deterministic, every entry
  naming its run manifest or the explicit no-run sentinel.
- **`margins.stats.icc`** measures the intraclass correlation: nested ANOVA with cells a FIXED
  factor, so between-cell signal never enters rho; exact F-based interval.
- **Artifacts** under `data/stats/`: `effects_corrected`, `family`, `descriptive_only`, `claims`
  (Parquet) plus `fdr_report.md` (byte-deterministic) and `power_analysis*.json`. No new
  dependencies.
- **Both Kaggle pilots are FROZEN.** `open_gen_v1.2` changed every `prompt_id`, so regenerating
  either would make it describe a corpus its run never saw. Their drift guards now assert a
  **pinned block sha256** instead of "matches the live config". Re-running one means checking out
  the revision it was generated at.
- **`scripts/run_stats.py` on the dev artifact reports NOTHING TO CORRECT** — correct, because the
  dev run has one rendering per cell and P7 found every pair not estimable.

### P7 as built (for P9 to consume)

- **CONTRACT CHANGE: `ItemScore` gained `prompt_id`** (required, no default). The estimator
  clusters by ITEM (the prompt) and its bootstrap resamples whole items; a score row that does not
  name its item cannot be clustered, and the score table alone could not be fitted without it.
  `tests/test_schema_field_freeze.py` updated deliberately; `ResponseScorer` populates it from the
  prompt it already holds; `DriftRow` unchanged (each control text is its own item).
  `RunManifest` gained a `metrics` block; `ExperimentConfig` gained `metrics_name`.
- **Four estimand decisions, confirmed with the human 18 Sep 2026, before the pre-registration tag:**
  1. **Order-2 cells only, sum-to-zero (deviation) coding.** Per pair, a two-way layout over its
     |A|×|B| order-2 cells; `γ_ij` = the cell mean's departure from the additive fit (in a balanced
     design exactly `μ_ij − μ_i. − μ_.j + μ..`, checked to 1e-10). No reference level — age and
     gender have none. Order-1 cells are described by `CellStat` only: the lattice has no
     "no identity" cell, so an "absent" level leaves the (absent, absent) corner unidentified and
     confounds absence with "one fewer descriptor".
  2. **`harm_std` = the equal-weight mean of the harm type's scorers' `standardized` per response**
     (each leg already z-scored within benchmark × scorer × condition). A response missing a scorer
     is **refused**, not imputed (constraint 6). Per-scorer fits are ALSO emitted (`fit_terms` /
     `fit_reports`, `scorer_id` set) because the drift comparison is scorer-specific.
  3. **Bootstrap: items resampled with replacement WITHIN each cell** (whole items move together; every
     replicate keeps every cell identified — an unstratified resample drops a cell with p ≈ e⁻⁹ per
     cell per replicate). Percentile CI. Refits by numpy `lstsq`, held to the statsmodels point fit.
  4. **Every level's coefficient is emitted**, not only the (|A|−1)(|B|−1) free ones: `term =
     "A=a:B=b"` for all |A|·|B| interaction cells plus `"A=a"`/`"B=b"` main effects. The set is
     linearly dependent (γ rows/columns sum to zero) — **P9 must account for that when it defines
     the FDR family**; it was not hidden by dropping a level.
- **The clustering demonstrably works, and is the right size.** Synthetic item intercept sd 1.0,
  noise 0.3, 6 items/cell: clustered/iid SE ratio **2.14 at 5 seeds/item, 4.34 at 20** (the design
  effect grows with seeds, as it must), and the clustered SE (0.226) sits on the analytic sampling
  SD (0.236) while the iid SE (0.052) is a quarter of it. The cluster-robust covariance is recomputed
  by hand in the tests (sandwich + statsmodels' G/(G−1)·(N−1)/(N−k) correction) to 1e-10.
- **Zero-interaction → ~0 with the intersection the HIGHEST level in the table** (mean +1.34 SD):
  all six |γ| ≤ 0.06, RMS 0.047; over 24 independent datasets the mean γ is ~0 and the 95% CI
  covers zero ≥ 19/24 times. Calibration is asserted over repeated datasets, not one seed — joint
  coverage of seven terms at one seed fails ~25% of the time for a *correct* estimator.
- **Not estimable is a reported result, never a NaN.** Empty cell, < 2 items, or no residual df →
  `FitReport(estimable=False, reason=…)`, no `EffectEstimate` for that pair. The dev run (24 cells ×
  1 rendering) reports **every** pair not estimable — correct, and the smoke shows it.
- **`powered` comes from the ACTUAL n against `min_cell_n`** (the lattice placeholder, passed in),
  never from the projection. `EffectEstimate.powered` = every cell of the pair powered.
- **Drift: both families fitted separately, never pooled; nothing subtracted.** `compare_drift`
  emits `drift_rms / model_rms` per (scorer, pair, family) — `model_rms` is *that scorer's own*
  model fit, apples to apples — a `comparable` flag at **`comparable_ratio = 0.5`**, a
  `families_disagree` flag at **`family_disagreement_ratio = 2.0`**, and a per-harm-type
  `descriptive_only_candidate`. **Both thresholds are PLACEHOLDERS in
  `configs/metrics/emergence_v1.yaml` — TODO(P9): pre-register.** The downgrade is P9's decision.
- **⚠ KNOWN CAVEAT, found by running the real drift artifact: the percentile-bootstrap CI of
  `rms_interaction` is biased upward near zero.** RMS is a norm, so every replicate's γ's carry
  sampling noise that inflates their RMS, and a scorer whose true interaction is ~0 gets a point
  estimate *below* its own CI's lower bound (e.g. `regard_classifier` on `age × race`, RMS 0.001,
  CI [0.005, 0.017]). Per-coefficient γ CIs are unaffected (γ is linear), and the drift ratio and
  flags use point estimates, so nothing downstream is wrong — but **P9 must not read `rms_ci` as a
  calibrated interval for small magnitudes**; a bias-corrected bootstrap or a coefficient-scale
  comparison is the fix, and it is P9's call.
- **Early signal on the real P6 artifact, still NOT a finding:** `toxic_bert`'s own interaction on
  neutral text is 0.10–0.28 SD RMS across pairs (both families), `regard_classifier` 0.06–0.42
  (largest on `ability × gender`), while both lexicon scorers sit at 0.000. No model comparison
  exists yet (the dev run has one rendering per cell), so no ratio, no flag, no claim.
- **The permutation test (B3) is NOT built.** §11's P7 row names it, but the brief locked P7 to the
  estimator + drift; it belongs with P9's resampling machinery. `EffectEstimate.p` stays `None`
  here too — nothing in `margins.metrics` computes a p-value.
- **Artifacts** under `data/metrics/`: `effects.parquet` (uncorrected — P9 reads it), `cell_stats`,
  `fit_terms`, `fit_reports`, `drift_comparison`, `family_agreement`, `harm_flags`,
  `drift_side_by_side` (Parquet, round-trip-tested), `drift_comparison.md` + `fit_report.md`
  (byte-deterministic). No new dependencies: `statsmodels`/`scipy` were already core.

### P6 as built (for P7 to consume)

- **`schemas.py` was NOT touched.** `ItemScore` already had every field P6 needed, `condition ∈
  {model, drift_control}` included. `tests/test_schema_field_freeze.py` is unedited. The only shared
  record that changed is `RunManifest`, which gained a `scoring` block exactly as P5 gave it
  `inference` — a deliberate contract change, logged in the field-freeze test.
- **Roster = 2 harm types × 2 scorers, and the ≥2 rule is a LOAD FAILURE, not a convention.**
  `ScoringConfig` refuses a harm type with fewer than two scorers **and** one whose scorers share a
  `mechanism` — because two classifiers agree with each other and are wrong together, which is
  precisely what §9 says agreement cannot detect. So each harm type pairs a transformer with a
  rule-based method:

  | harm_type | scorer | mechanism | pin |
  |---|---|---|---|
  | `toxicity` | `toxic_bert` | neural | `unitary/toxic-bert` @ `4d6c22e7…` (apache-2.0), label `toxic` |
  | `toxicity` | `ldnoobw_lexicon` | lexicon | LDNOOBW @ `5faf2ba4…`, 403 terms, CC-BY-4.0 |
  | `regard` | `regard_classifier` | neural | `sasha/regardv3` @ `9232a1d0…` (**cc-by-4.0**), label `negative` |
  | `regard` | `vader_valence` | lexicon | `vaderSentiment` 3.3.2 (MIT) |

  **Revisions and licences were RESOLVED, never authored** — `scripts/pin_scorers.py --write` for the
  checkpoints (it also verifies the configured **label exists at that revision**, so a re-pin cannot
  silently start reading a different column), `scripts/fetch_lexicons.py --write` for the wordlist.
  Both stop rather than substituting. The regard licence was read off the Hub and is CC-BY-4.0; the
  placeholder had said MIT and that was wrong.
- **`detoxify` is deliberately UNUSED.** It is `sys_platform == 'linux'`, so a detoxify-backed scorer
  could never run on the dev box or the core CI job — the wrong place for the instrument producing
  the primary estimand's harm signal. The same checkpoint loads through `transformers`, which is
  ungated in the extra and comes under our own pinning rules.
- **The sign contract is executable.** `standardize_within_benchmark` takes `benchmark` as a
  **required keyword** and refuses rows from any other source. Every scorer declares a `sign` *and* a
  `SignProbe` (a known-harmful / known-benign pair), so `verify_sign` **fails** instead of the sign
  merely being asserted; `scripts/run_scoring.py` runs it on the real instruments before scoring
  anything. Standardization groups by `(benchmark, scorer, harm_type, condition)` — per condition
  because z-scoring model output together with control text would subtract a partly-control mean,
  i.e. smuggle in the point-subtraction §9 forbids. A zero-variance group standardizes to 0.0 and is
  **reported** as degenerate, never NaN.
- **Krippendorff's α is computed in CLOSED FORM, and the package is the test oracle.** `krippendorff`
  at interval level builds a coincidence matrix over *distinct observed values*; with continuous
  scores that is O(n·V²) — **1,430 units already asks for 22 GiB**, and the confirmatory design is
  ~49,500 renderings. Subsampling to fit would make the reported α a property of the subsample. The
  closed form is O(n) and exact; `tests/test_triangulation.py` checks it against the package to
  1e-12 (with and without missing ratings) and separately runs it at 20,000 units. **This was found
  by running the real pipeline, not by a test** — it is the reason to keep doing that.
- **Drift control: two families, both series emitted, NOTHING point-subtracted.**
  `prompt_echo` (the 990 P4a prompts) + `neutral_declarative` (440 = 4 new frames × 110 cells,
  rendered through the **ordinary `TemplateRenderer`**, so the leakage guard, article rule and
  `compose_rank` order all apply unchanged — this is what `render.py`'s `TODO(P6)` anticipated).
  Control texts ride as **pseudo-`Response` records** (a documented CONSTRUCT CHOICE: sentinel
  `model_id="drift_control::template"`, `seed=0`, `quant="none"`) so they flow through the *unchanged*
  `Scorer.score(responses)` path — the same code and the same loaded instrument that scored the model
  output. `data/scoring/drift_control.parquet` carries `axis_a/value_a/axis_b/value_b` **resolved
  from the lattice**, so P7 can fit `A + B + A:B` on the scorer's own series without parsing a
  `cell_id`. P6 fits nothing: no link function, no SE, no CI.
- **⚠ EARLY SIGNAL, NOT A FINDING — `toxic_bert` shows a large own-interaction.** On a real run,
  `blindness × race` came out at **+1.71 SD** from text containing no harm whatsoever, and
  `regard_classifier` at +0.67–0.80 on two pairs. The lexicon scorers sit at ~0.00. This is exactly
  the fabricated-emergence mode §9 predicts, and it is **preliminary**: tiny n, no CI, no
  significance, contrast computed by hand. **Go/no-go criterion 3 needs only ONE harm type to
  survive**, and having two harm types plus two lexicon legs is why that is still reachable. P7/P9
  enforce the consequence: *a scorer whose own interaction is comparable to the model's makes that
  harm type DESCRIPTIVE-ONLY.*
- **Calibration is a SCAFFOLD; the labeling is the human's.** Deterministic, stratified by cell order
  × scorer-disagreement tercile (a uniform draw would spend the budget confirming true negatives),
  seeded through `set_global_seed`, bounded to §9's 150–300 by the schema. Precision/recall are
  `None` rather than 0.0 on an empty denominator — "never fired" and "always wrong" are different
  facts. **No real labels exist yet**; `data/scoring/calibration_report.md` is unwritten until they do.
- **First Parquet in the repo, and its determinism is the WEAKER claim, stated.** Parquet embeds
  writer metadata, so unlike the lattice/prompt exports it is **not byte-deterministic**. What is
  asserted and tested is **round-trip content equality on a canonically sorted frame** — which is what
  P7 depends on. `agreement.csv`/`.md` *are* byte-deterministic.
- ~~**Open decision #7 is still open, by construction.**~~ **CLOSED at P2 (20 Sep 2026).**
  `ResponseScorer.score()` still **refuses** a paired response rather than text-scoring it -- that
  refusal survived and is now one half of a symmetric pair, since `PairedScorer` refuses a
  generated one. The `TODO(P2)` became `margins.scoring.paired`.
- **`configs/config.yaml` gained `scoring_name`** (the `_name` suffix rule again — `configs/scoring/`
  is a Hydra config GROUP). New **core** deps, not the inference extra, because agreement,
  standardization, drift arithmetic and calibration must all run on the torch-free CI job:
  `krippendorff`, `scikit-learn` (previously an undeclared `pingouin` transitive), `vaderSentiment`.

### P5 as built (for P6 to consume)

- **Pinned checkpoints** in `configs/model/*.yaml`, revisions **resolved via `huggingface_hub`, never
  authored**. `scripts/pin_models.py` is both how they got there (`--write`) and how a reproducer
  checks them (default `--verify`); on a 404 or gated-without-access it **stops** and refuses to
  substitute. `ModelSpec.revision` rejects a branch name — `main` moves and nothing we record would
  change. Ids/shas/licences are in decision #2 below and pinned by `tests/test_model_config.py`.
- **`decoding_key` finally has something behind it.** `configs/decoding/greedy_v1.yaml` is the
  profile P4a stamps; `decoding_hash` = sha256 of the canonical JSON dump, on every `Response`.
  **`temperature` is a required field with no default**, and a profile that is neither cleanly greedy
  (`do_sample=False`, temp exactly 0, no `top_p`/`top_k`) nor cleanly sampled fails to load —
  transformers would silently ignore a temperature under greedy and the `Response` would record one
  nothing used. The backend refuses a prompt whose `decoding_key` ≠ the profile it holds.
- **`prompt_format = "chat"` is a CONSTRUCT CHOICE** (native template, one user turn, no system
  message): the prompts are instructions and the pins are instruct-tuned. The cost is that the two
  families wrap a prompt differently, so the **chat-template sha256 is recorded per model** in the
  manifest (`chat_template:<model>`), not assumed equal. `pinned_chat_date` + a render-time guard
  exist because Llama's template injects a live "Today Date" — unpinned, the same prompt would
  render differently tomorrow.
- **`quant` on a `Response` is what ACTUALLY happened, never the intended constant.** NF4 needs
  bitsandbytes + CUDA; a Windows dev load falls back to unquantized and records `quant_actual="none"`
  plus the reason. The manifest's new `inference` block carries requested-vs-actual side by side.
- **Resumable harness contract, asserted not commented:** work unit = `(prompt, seed)`, with a
  content-addressed `response_id` computed **before** the work runs. Output
  `data/responses/<model>/<run_id>.jsonl`, `run_id` from the manifest (never random-per-call);
  flush+fsync per batch; a **truncated final line is repaired** (the normal state of a killed
  session) while a *complete* invalid line **raises** (corruption ≠ interruption); the planned id set
  must be a subset of the file's at the end, and a file holding work outside the plan is refused.
- **Distinct samples come from distinct SEEDS.** The backend re-seeds per batch so a resumed run
  reaches every batch with the RNG state an uninterrupted one would have — which means two calls at
  one seed return identical text, so `plan()` rejects repeated seeds. Batches never straddle a seed.
- **Determinism is now scoped to batch size too.** Padding changes reduction order, so
  `residual_nondeterminism_note()` gained a batch-size clause and the manifest records `batch_size`.
- ~~**Open decision #7 is still open, by construction.**~~ **CLOSED at P2/P4b (20 Sep 2026).**
  `run()` no longer refuses `pair_pll`; it routes to `score_pairs` and refuses a **mixed** call
  instead. `score_pair()` still accepts explicit sequences, but now defaults to reading them off
  the prompt, and `sample_index` became a parameter so the harness's planned ids line up.
- **`schemas.py` was NOT touched** — `Response` already had every field P5 needed.
- **`margins.inference` imports with no torch installed** (lazy imports, as `repro` does), so
  `mypy src` and the core suite still run on the core environment.
- **The checkpoint's bundled `generation_config` is neutralized at load.** Llama-3.2-1B-Instruct
  ships `temperature=0.6, top_p=0.9`; greedy ignores them, but a future sampled profile that did not
  name a `top_p` would inherit 0.9 **silently and unrecorded** — it never came from our config, so it
  would appear in no `decoding_hash`. `DecodingConfig` is now the sole authority on token choice.
- **Environment split, amended (`docs/PROJECT_CONTEXT.md` §7).** `torch`/`transformers`/`accelerate`/
  `huggingface-hub` are no longer Linux-gated, so `uv sync --extra inference` works on `win_amd64`
  and the gates really do run locally against the 1B. `bitsandbytes` (+ `datasets`/`evaluate`/
  `detoxify`) stay Linux-only; **the 7–8B sweep still never runs locally**. Default `uv sync` is
  unchanged. `huggingface-hub` moved from core deps into the `inference` extra; `uv.lock` regenerated
  and still resolves for exactly the two environments.
- **`mypy src` no longer depends on which extras are installed.** With `ignore_missing_imports`
  alone, the check passed on the core env (packages absent ⇒ `Any`) and failed on the inference env
  (partial stubs ⇒ `no-untyped-call`). `follow_imports = "skip"` on the inference stack makes both
  cases identical. A gate whose result depends on the environment is not a gate.

### The Kaggle timing pilot — RUN 18 Sep 2026 (measured; do not re-run, do not re-recommend)

`scripts/kaggle_timing_pilot.py` + the generated `notebooks/kaggle_timing_pilot.ipynb`.

**MEASURED on Kaggle, Tesla T4, both models at NF4:**

| model | sec/prompt | peak VRAM |
|---|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | ~14.0 | ~6.6 GB |
| `mistralai/Mistral-7B-Instruct-v0.3` | ~14.8 | ~8.0 GB |

**Verdict (this is decision #8, recorded in full below):** greedy E1 gives ~9 renderings/cell and is
**statistically underpowered — ruled out**. Sampled ×11 (~99/cell) costs **~87 GPU-h ≈ ~290% of
Kaggle's ~30 GPU-h weekly quota** — unaffordable on free tier. The design target is therefore
**sampled generation at ~50 renderings/cell**, run on a **university GPU**, with Kaggle retained as
the documented reproducibility fallback.

`pilot_results.json` is **not committed** (`data/` is gitignored). Its home is
`data/pilot/pilot_results.json`; the numbers above are the record of it.

- **Self-contained.** Imports nothing from `margins` (Kaggle has no checkout, and `data/` is
  gitignored so the prompt file is absent there). Its 50 prompts, the two model pins and the decoding
  profile are **generated into the file** by `scripts/build_pilot_sample.py` from the same configs
  the pipeline uses; `tests/test_kaggle_pilot.py` fails if any of it drifts, and separately asserts
  each embedded prompt still renders identically. They are the real P4a prompts, sampled by the same
  deterministic, group-stratified sampler the P4a slot audit uses.
- **To run:** GPU T4 ×2, Internet On, `HF_TOKEN` secret attached (the account must have accepted the
  Meta licence), Run All. ~20–40 min, mostly weight downloads. `--models llama31_8b_instruct` halves
  it. Bring back `pilot_results.json`.
- **It measures** sec/prompt, tokens/sec, mean/median generated tokens, peak VRAM and load time per
  model, then prints a **scenario table** — greedy E1 (990×2×1), sampled E1 (990×2×11 @ 0.7, the
  99-renderings/cell arithmetic), E4 subset (3 temps × 3 seeds × 10%) — in GPU-h and % of ~30
  GPU-h/week, with every assumption printed above the numbers. **It refuses to extrapolate without
  CUDA**: a CPU timing presented as a GPU budget would be a fabricated number.
- **What it fed:** hard go/no-go criterion 4 (Sep 28–29) and the **E1 grid decision**, both now
  settled — see decision #8. Red-teaming (P8) is **not** in the pilot's table, so its cost still has
  to be added to the university-GPU budget separately.

### P4a as built (for P5 to consume)

- **990 prompts** = 9 templates × 110 cells × 1 carrier, at `data/prompts/prompts_v1.jsonl`
  (gitignored, regenerable via `scripts/build_prompts.py`). Byte-identical regeneration verified.
- **`prompt_id` = `<template>::<cell_id>::<carrier>::<digest12>`** — readable natural key plus a
  digest over the full rendering key *including the rendered text*, so a template edit that forgets
  to bump `template_version` shows up as changed ids rather than silent drift.
- **Carrier noun = `person`, a documented CONSTRUCT CHOICE** (`configs/prompts/open_gen_v1.yaml`).
  Every obvious alternative is itself an identity axis; an occupational carrier would add an
  undeclared second axis of variation to every cell. `CarrierSpec` refuses a non-neutral carrier
  that does not name the axis it introduces; `MAX_CARRIERS = 3`.
- **Leakage guard is enforced at render time, not by review:** the set of lattice identity terms in
  a rendered prompt must equal *exactly* the set its cell contributes. Frames are additionally
  checked against axis words, rejected if they place identity sentence-initially (capitalising it
  would rewrite pinned `Black`/`Deaf` orthography), and a carrier may not be a lattice term.
- **Article rule lives in the prompt config**, not the lattice: vowel-letter rule plus an override
  table, and it **refuses to guess** on `eu-`/`u-`/`one-`/`hon-`. No overrides are needed for the
  18 shipped descriptors.
- **Budget arithmetic:** renderings/cell = |templates| × |samples|. ~~11 samples ⇒ 99/cell
  (**10,890 renderings**), 12 ⇒ 108; both clear `min_cell_n = 60`.~~ **Superseded by decision #8
  (18 Sep 2026):** the pilot priced the ×11 grid at ~290% of Kaggle's weekly quota, so the target is
  **~50 renderings/cell** — between greedy's underpowered 9 and the unaffordable 99. With 9 frames
  that is 5 or 6 samples (45 or 54/cell). **The exact number is NOT settled here**: it is locked at
  P9 *together with* `min_cell_n`, because a rendering budget and the minimum N it must clear are one
  decision, not two. Note ~50 sits **below** the `min_cell_n = 60` placeholder — that is expected and
  is precisely why both are replaced at the same time.
- ~~**`decoding_key = "greedy_v1"` is a NAME only.**~~ **Resolved at P5:**
  `configs/decoding/greedy_v1.yaml` + `DecodingConfig.decoding_hash`. Editing that file changes the
  prompt-set config's sha256 but **not** `prompt_id` (which digests the rendered text, not the
  profile) — it changes every `response_id` instead, which is the right blast radius.
- **Manifest now records the lattice** (the provenance hole P3 flagged, closed).
  `ExperimentConfig.lattice_name` is the only field added; `input_checksums` carries
  `lattice_config`, `prompt_set_config` and one `template:<id>` per frame. The lattice digest
  matches what `scripts/build_lattice.py` prints. *Named `lattice_name`, not `lattice`, because
  `configs/lattice/` is a Hydra config GROUP and a colliding key cannot be overridden at all.*
- ~~**Open decision #7 is untouched.**~~ **CLOSED at P4b (20 Sep 2026).** `PromptSetConfig` still
  types `task_type` as `Literal["open_gen"]`, and that narrowness is why P4b added a **separate**
  `ExtractivePromptSetConfig` rather than widening it: the two renderers share almost no fields, and
  one config class would have made every field optional for one of them.

### P4a phrase review — all 4 findings resolved (decided 8 Sep 2026)

Full record in `data/prompts/phrase_audit.md` (all 92 order-2 descriptors shown *inside* a rendered
sentence). **No descriptor was edited** — the levers used were the template frame and
`compose_rank`. Notes are kept after resolution and their predicates act as **regression guards**:
a RESOLVED note that matches any cell fails the suite.

| Finding | Decision |
|---|---|
| `gender-composed-before-non-gender` (was 12) | **RESOLVED** — lattice **v1.1** |
| `age-before-status-adjective` (was 12) | **RESOLVED** — same reordering |
| `marked-register-with-neutral-carrier` (24) | **MITIGATED** — carrier-free frame added |
| `repeated-morpheme` (1) | **ACCEPTED** — documented, not fixed |

- **Lattice `confirmatory_v1.1@2026-09-08`** reorders `compose_rank` to **class(10) < age(20) <
  ability(30) < race(40) < gender(50)** — the unique order satisfying all seven declared pairs
  ("a wealthy young person", "a Black female person"). Values, pairs and sources are **unchanged**;
  `--verify` re-run green against the pinned HolisticBias file. **Cell ids changed for those two
  pairs** (`socioeconomic_class=…|age=…`, `race_ethnicity=…|gender_and_sex=…`) and the P3 exports
  were regenerated. Pre-dates the pre-registration tag, so this is an edit, not an amendment.
- **Prompt set `open_gen_v1.1@2026-09-08`** adds the carrier-free frame `someone_who_is`.
  **Two caveats that must not be lost:** (1) it is **1 frame of 9**, so the "a female person"
  register is reduced, not eliminated — the §9 drift control remains the backstop for the other 8;
  (2) it required a **second construct choice** — English predicative position rejects juxtaposed
  adjectives ("someone who is older blind" is broken), so this frame joins constituents with
  **"and"**, which asserts two co-predicated properties where juxtaposition can read as one
  compound category. **R1 should report this frame separately.**
- Side effect worth knowing: the reordering puts gender **last**, i.e. adjacent to the carrier, so
  the marked register is now adjacent in every gender cell rather than only some.

### P3 as built (for P4a to consume)

- **5 axes / 18 values**, every value traced to HolisticBias **v1.1**, pinned to commit
  `6d76f31a6631aa882fd00fadb6dd8600577533d1` (file sha256 `c1328f0e…52daf943`, MIT).
  `scripts/fetch_holistic_bias.py` retrieves it; `scripts/build_lattice.py --verify` checks every
  `(axis, bucket, term, preference)` tuple against that file, so a fabricated descriptor is a
  failing check rather than a paper claim.
- **110 cells** — 18 order-1, 92 order-2 — from **7 declared pairs only** (the anti-cartesian
  guarantee; the full product would be 10 pairs). **6 confirmatory pairs = the FDR family, 4 of
  them non-gender × non-gender** (`class×race`, `age×ability`, `ability×race`, `age×class`)
  against the gate's minimum of 2. 1 exploratory pair (`age×race`), descriptive only.
- **Exports** (regenerable, gitignored): `data/lattice/cell_inventory.csv` and
  `data/lattice/value_provenance.md`. Byte-deterministic.
- **`min_cell_n = 60` is a PRE-REGISTERED PLACEHOLDER.** TODO(P9): derive it from the real power
  analysis (α=0.05, power=0.8). Projected 96 renderings/cell → **10,560 renderings total**; the
  W2 Kaggle timing pilot is what turns that into a budget.
- **Descriptor composition is a construct choice** (`DeclaredLattice.compose_descriptor`):
  surfaces joined in `compose_rank` order, e.g. `working-class Black`, `older blind`. It emits a
  modifier phrase and **no carrier noun** — attaching one is P4a's decision, in the template.
- **Open at P4a:** `ExperimentConfig` does not yet name a lattice. `scripts/build_lattice.py`
  prints the config sha256 for `write_run_manifest(inputs=...)`; wire it when P4a needs a run.

### Build order (DECIDED 8 Sep 2026 — generation vertical first)

Phase IDs are **stable labels, not a sequence**; they deliberately run out of numeric order. The
authority is the `docs/PROJECT_CONTEXT.md` §5 table, which is sorted by build order.

> P0 ✅ → P1 ✅ → P3 ✅ → P4a ✅ → P5 ✅ → P6 ✅ → P7 ✅ → P9 ✅ → P2 ✅ (CrowS-Pairs) →
> P4b ✅ (ExtractiveRenderer) → P8 ✅ → P10 ✅ → **P11** → P12 → P13

**Why:** against a 7-week clock, the phase that must not arrive late is the one carrying the primary
estimand. Generation-first means a slip costs **breadth** (RQ1's second paradigm — already cuttable
under the locked scope); extraction-first would mean a slip costs a **claim**. The cost is explicit:
RQ1/E2 cannot land until P2 + P4b are done, which matches §12's cut order (E2 already ranks below
E1/B1/B3/E4).

**P4 is split along the renderer seam:** `TemplateRenderer` (**P4a**) injects a cell into a template
and needs only P3; `ExtractiveRenderer` (**P4b**) derives a cell from a source item and cannot exist
before a loader, so it rides with P2. Do not skip ahead.

**Hard deadline: FAccT abstract 27 Oct 2026 (mandatory, non-placeholder), paper 3 Nov 2026 — 8 weeks
from ~8 Sep.** *(Verify the deadline year against the official CFP; these dates may belong to FAccT
2027. See the note atop `PROJECT_CONTEXT.md`.)* **Hard go/no-go on Sep 28–29** — five criteria in §13;
if any fails the fallback is ACL Rolling Review or a non-archival workshop, decided at that gate.

**Environment preconditions all met.** Repo at `C:\dev\mapping-the-margins` (outside OneDrive),
git initialised on `main`, remote is **private** `github.com/breenu/mapping-the-margins`, CI runs on
push. Flip the repo to public at P13 release, not before.

**What this is (one line):** a reproducible, statistically rigorous **framework** (Python package
`margins`) for **intersectional bias auditing of open-weight LLMs**. The deliverable is *the
instrument + its rigor*, not the finding that intersectional bias exists.

---

## Confirmatory scope (LOCKED — narrow from day one, breadth-additive)

8 weeks against a fixed deadline. **Breadth gets cut; a claim never does.** Full rationale + the
claim-survival table in §1.

- **IN:** two paradigms (templated `open_gen` from HolisticBias descriptors **+** CrowS-Pairs);
  **one** 7–8B model family + a ~1B dev model; **red-teaming** (claimed novelty); the **scorer
  identity-drift control**.
- **OUT unless ahead:** WinoBias, StereoSet, BBQ-calib, a second model family, ablations A1–A3.
- **k = 2 by design → no pooling.** RQ1 is answered by the cross-paradigm **effect vector +
  agreement**. No τ², no I². Pooling returns only at k ≥ 3.
- **The MVP is the GENERATION path**, not the extractive one — that is where RQ2 and the risk live.

---

## Commands

```bash
# uv is the dependency manager; Python 3.11. (P0 creates pyproject.toml + uv.lock.)
uv sync                      # core deps (no GPU stack) — this is the Windows dev environment
uv sync --extra inference    # torch, transformers, accelerate, huggingface-hub (BOTH environments)
                             #   + bitsandbytes, datasets, evaluate, detoxify (Linux/Kaggle only)
uv sync --extra llamacpp     # pure-CPU / Apple fallback backend
uv sync --extra dev          # pytest, ruff, mypy, pre-commit

# P5 (needs --extra inference). NF4 is Linux/CUDA; a Windows load falls back and RECORDS that.
uv run python scripts/pin_models.py                       # verify every pinned revision upstream
uv run python scripts/run_inference.py --model llama32_1b_instruct --limit 20   # local smoke
uv run python scripts/run_inference.py --model llama32_1b_instruct --resume     # continue a run
uv run python scripts/build_pilot_sample.py               # regenerate the Kaggle pilot's data
# The Kaggle timing pilot was RUN 18 Sep 2026 — decision #8. Do not re-run or re-recommend it.

# P6 scoring. Runs on the CORE env: only the two neural scorers need --extra inference, and their
# tests are `inference`-marked. Both pin verifiers STOP rather than substituting.
uv run python scripts/fetch_lexicons.py                   # --verify (default): check disk vs the pin
uv run python scripts/fetch_lexicons.py --fetch-pinned    # download AT the recorded revision. What
                                                          #   CI runs; refuses on a digest mismatch.
uv run python scripts/fetch_lexicons.py --write           # RE-PINS to upstream HEAD. A human action.
uv run python scripts/pin_scorers.py                      # --verify revisions AND label presence
uv run python scripts/run_scoring.py --responses data/responses/<model>/<run_id>.jsonl
uv run python scripts/build_calibration_sample.py --harm-type toxicity --responses <file>
uv run python scripts/calibration_report.py --labels data/scoring/labels_toxicity.jsonl

# P2 + P4b, the EXTRACTIVE vertical (CrowS-Pairs). Core env for ingestion; the paired scoring path
# needs no torch either -- only the inference run does. The fetch script STOPS rather than
# substituting, exactly like the lexicon one.
uv run python scripts/fetch_crows_pairs.py                # --verify (default): disk vs the pin
uv run python scripts/fetch_crows_pairs.py --fetch-pinned # download AT the recorded revision (CI)
uv run python scripts/fetch_crows_pairs.py --write        # RE-PINS to upstream HEAD. A human action.
uv run python scripts/build_crows_items.py                # screen -> items + excluded_items.csv
uv run python scripts/build_crows_prompts.py              # P4b: one paired Prompt per item
uv run python scripts/run_inference.py --model llama32_1b_instruct \
    --prompts data/prompts/crows_pll_v1.jsonl --decoding pll_v1 --limit 40
uv run python scripts/run_scoring.py --responses <file> --prompts data/prompts/crows_pll_v1.jsonl \
    --scoring crows_pll_v1 --benchmark crows_pairs --out-dir data/scoring/crows_pairs
uv run python scripts/run_metrics.py --scores data/scoring/crows_pairs/scores.parquet \
    --drift data/scoring/crows_pairs/drift_control.parquet --benchmark crows_pairs \
    --out-dir data/metrics/crows_pairs

# P8 RED-TEAM, three stages. Stages 1 and 3 are core-env; only the middle step needs a GPU, and
# it is the EXISTING run_inference.py -- red-team generation is a run of the P5 harness, so it is
# resumable and manifest-writing for free. Heavy generation is P11's budget, NOT CI.
uv run python scripts/build_redteam_elicitation.py        # -> data/redteam/elicitation_prompts.jsonl
uv run python scripts/run_inference.py --model llama31_8b_instruct \
    --prompts data/redteam/elicitation_prompts.jsonl --decoding redteam_v1   # the red LM (GPU)
uv run python scripts/build_redteam_frames.py --responses data/responses/<red-lm>/<run_id>.jsonl
                                                          # parse -> filter -> dedup -> GUARD
uv run python scripts/build_redteam_prompts.py            # frames x EVERY target cell -> prompts
# Then the UNCHANGED path, tagged `redteam` so E3 can tell the two arms apart:
uv run python scripts/run_inference.py --model llama31_8b_instruct \
    --prompts data/prompts/cell_targeted_v1.jsonl --decoding sampled_v1
uv run python scripts/run_scoring.py --responses <file> \
    --prompts data/prompts/cell_targeted_v1.jsonl --benchmark redteam --out-dir data/scoring/redteam
uv run python scripts/run_metrics.py --scores data/scoring/redteam/scores.parquet \
    --benchmark redteam --out-dir data/metrics/redteam
# The on-target % is the HUMAN's: fill in on_target (0/1) in data/redteam/on_target_sample.jsonl.
uv run python scripts/redteam_on_target_report.py --labels data/redteam/on_target_labels.jsonl

# P7 metrics. Core env, no torch. Reads the two P6 parquets; writes data/metrics/ + a manifest.
# --n-resamples overrides the config's 2000 for smoke runs only (recorded as used).
uv run python scripts/run_metrics.py
uv run python scripts/run_metrics.py --n-resamples 200      # smoke

# P9 stats. Core env, no torch. The ONLY stage that declares significance.
# --metrics-dir takes BOTH paradigms' directories: the FDR family spans the whole
# pre-registered set and is NEVER corrected per-source (§8).
# NOTE: data/metrics/redteam is deliberately NOT passed yet -- `redteam` is absent from
# `family_benchmarks`, so P9 would refuse it as an undeclared source. Admitting it is a logged
# amendment that rides with a re-run power analysis at P11 (168 -> 336 moves the BH threshold).
uv run python scripts/run_stats.py --metrics-dir data/metrics data/metrics/crows_pairs
uv run python scripts/run_power_analysis.py     # the power simulation (slow, ~30 min)
uv run python scripts/emit_power_numbers.py     # -> results/paper_numbers.json
# rho was MEASURED 19 Sep 2026 (regard 0.35, toxicity 0.18). Do NOT re-run these two: both
# pilots are frozen records and their drift guards are now pinned-digest tests.
#   scripts/kaggle_rho_pilot.py (Kaggle, GPU)   scripts/measure_rho.py (local, CPU)

# P10 figures + tables. Core env, no torch. Every drawn number is audited against a committed
# artifact as the figure is built, so an untraceable figure is never written.
uv run python scripts/make_figures.py            # -> figures/*.pdf + *.svg, ONE command
uv run python scripts/make_figures.py --only essential   # the five that carry a headline claim
uv run python scripts/build_fixture_results.py   # -> results/fixtures/  (synthetic, pre-P11)
uv run python scripts/build_result_tables.py     # -> results/tables/    (needs a real P7+P9 run)
# make_figures.py prefers results/tables/ when it exists and falls back to results/fixtures/,
# printing which it used. Figures drawn from fixtures are banner-stamped on their face.

# Quality gates — all must pass before crossing a phase boundary.
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy src

# Regenerate the gitignored artifacts (both byte-deterministic).
uv run python scripts/build_lattice.py --verify   # data/lattice/  (P3; --verify needs the pinned file)
uv run python scripts/build_prompts.py            # data/prompts/ + a run manifest (P4a)
```

If a dependency won't resolve or a domain is blocked, surface it to the human — never work around
it by unpinning. `uv.lock` resolves for exactly two environments (`win_amd64`, `linux_x86_64`) via
`tool.uv.environments`; that is what keeps `bitsandbytes` from failing resolution on Windows.

---

## Honest framing constraints (NON-NEGOTIABLE — govern claims, code comments, paper text)

The novelty is narrow, real, and perishable. Violating these turns a publishable artifact into a
desk-reject. Rationale for each is in `docs/PROJECT_CONTEXT.md`.

1. **Instrument, not finding.** Never frame the contribution as "we discovered intersectional bias in
   open-weight LLMs" — that is settled prior art. The contribution is the **instrument + rigor**.
2. **Accessible compute, not "single laptop" literalism.** The claim is *"accessible / commodity
   compute — no industrial GPU; reference runs on free-tier cloud (Kaggle/Colab)."*
3. **Emergence = interaction, not level.** Emergent intersectional bias is the **interaction term** in
   a factorial harm model, never a high bias *level* at a cell.
4. **Quantization is a fixed, recorded CONSTANT** — not an experimental axis and not a finding,
   unless a dedicated quantization-and-bias literature search is done first and justifies it.
5. **BBQ intersectional subsets are calibration-only** — never load-bearing for an intersectional claim.
6. **No single-classifier dependence** — every harm type uses ≥2 independent scorers with reported
   agreement **and** a scorer identity-drift control; poor-agreement or high-drift harm types are
   downgraded to descriptive.

---

## Design invariants (MUST NOT break)

1. **Shared records are frozen (immutable).** Transformations return NEW records via
   `.model_copy(update=...)`. Never mutate in place. Note Pydantic `frozen=True` does **not**
   deep-freeze containers — container fields are tuples / immutable mappings so records stay hashable.
2. **No source-specific field on a shared record.** Source quirks live ONLY in `Item.payload`
   (typed at the loader boundary). Shared records use `extra="forbid"`, and
   `tests/test_schema_field_freeze.py` pins each record's exact field set.
3. **Only `StatsEngine` may set significance.** `EffectEstimate.q`, `.family_size`, `.correction` are
   set nowhere else; the report layer raises if any is missing. **`CellStat` is descriptive and
   carries no `p`/`q` at all.**
4. **Underpowered cells are `powered=False` → descriptive-only**, never load-bearing. The report/table
   API defaults to `powered_only=True`.
5. **Determinism is a contract.** Every run seeds via `margins.repro.set_global_seed` and emits a
   manifest. Same-seed greedy produces **identical token sequences on CPU at a fixed thread count**;
   logprobs match to a documented tolerance; GPU is metric-level within a reported tolerance. The
   manifest records torch version, thread count, and BLAS — otherwise the claim isn't checkable.
6. **Intersection depth is fixed at 2** (`IdentityCell.order ∈ {1, 2}`). No depth-3 cells.

---

## Guardrails (NEVER)

- Never `pip install` unpinned or bypass the lockfile "to make it work."
- Never invent a citation, statistic, or model result. Enforced for statistics: the pipeline emits
  `results/paper_numbers.json` and the manuscript references numbers **by key**. *(Citations remain
  human discipline — there is no mechanism.)*
- Never standardize harm scores *across* sources — standardize **within** a source (the function
  requires a source key), with **higher = more harm** guaranteed by test.
- Never FDR-correct within a single source. The family is **benchmark × model × axis_pair × term ×
  harm_type** over the **6–8 pre-registered confirmatory axis pairs** (≥2 non-gender×non-gender).
  Anything outside that set is exploratory and carries no significance claim.
- Never fit the interaction GLM assuming independent residuals — items and templates are reused across
  cells, so use cluster-robust SEs (or a mixed model) and a **cluster bootstrap over items**.
- Never present the additive null (B2) as an independent baseline — it *is* the interaction estimate.
  The independent check is the **permutation test (B3)**.
- Never declare significance outside `StatsEngine`; never treat an LLM-judge as ground truth (it is one
  scorer of ≥2, with a fixed documented prompt/seed).
- Never commit model weights or raw response dumps to git (`data/` is gitignored + a
  `check-added-large-files` pre-commit hook; dumps ship as a Zenodo artifact at release).
- Never add quantization as an experimental axis (see constraint 4).
- Never claim reproducibility you haven't demonstrated — **R2 reruns the sensitivity subset, not full
  E1**, so the claim must be scoped to exactly that.
- No load-bearing notebooks — production logic lives in `src/margins/`; CI fails if `src/` or
  `scripts/` imports from `notebooks/`, and notebooks may not write to `results/`.

---

## Open decisions — ASK the human, do not silently choose

1. ~~Hardware branch~~ — **DECIDED: Windows laptop for dev (`win_amd64`, core+dev only), Linux GPU
   for inference (`linux_x86_64`, NF4). The 7–8B sweep never runs locally.** §7. **Amended by
   decision #8 (18 Sep 2026):** the inference host for the confirmatory sweep is a **university
   GPU**, not free-tier Kaggle; Kaggle stays as the reproducibility fallback.
2. ~~**Model checkpoints**~~ — **DECIDED 9 Sep 2026.** All four verified reachable at pin time; the
   two Meta repos are `gated: manual` and the project account's licence acceptance is confirmed.
   **Revisions were RESOLVED through `huggingface_hub`, never hand-authored** — re-check with
   `uv run python scripts/pin_models.py`.

   | role | repo | revision | license |
   |---|---|---|---|
   | confirmatory | `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` | llama3.1 (gated) |
   | confirmatory | `mistralai/Mistral-7B-Instruct-v0.3` | `c170c708c41dac9275d15a8fff4eca08d52bab71` | apache-2.0 |
   | dev | `meta-llama/Llama-3.2-1B-Instruct` | `9213176726f574b556790deb65791e0c5aa438b6` | llama3.2 (gated) |
   | ci | `hf-internal-testing/tiny-random-LlamaForCausalLM` | `9fb191250dd56d0ba7ec9785a025ed29c03d5998` | unknown |

   *Rationale:* **instruct** checkpoints are a deliberate construct choice, not a default — the P4a
   prompts are instructions, and a base model continues an instruction as text rather than performing
   it, which would make the harm signal a measurement of off-task drift. The pair spans **two
   families** so the confirmatory result is not one architecture's quirk. A pin that drifts is not a
   pin: `ModelSpec` rejects a branch name, and a re-pin is a logged decision, because any result
   produced under the old revision was produced under a *different* model.
3. ~~**Emergence link/scale + null model**~~ — **DECIDED 18 Sep 2026 (P7), pre-registration
   text.** Identity link, OLS on standardized scores; `harm_std ~ A + B + A:B` over an axis pair's
   order-2 cells with **sum-to-zero coding**; the additive null is the same model without `A:B`
   (B2 = a presentation of the interaction); cluster-robust SEs by item; cluster bootstrap over
   items stratified by cell; `harm_std` = mean of the scorers' standardized scores. See "P7 as
   built". **Still open for P9:** the permutation test's exact scheme (B3) and the two placeholder
   drift thresholds. Tag due end of W4.
4. **Quantization-as-axis:** OFF unless the human runs the dedicated literature search first.
5. **Paper title** is not finalized (distinctiveness risk noted in the deep reference). Working
   repo/package names are `mapping-the-margins` / `margins`. Don't hard-code a paper title anywhere
   load-bearing.
6. ~~**`pair_pll` scoring shape**~~ — **DECIDED 8 Sep 2026: option (a).** One `Response` carries both
   sequence log-probs (`logprob_stereo` / `logprob_antistereo`, `float | None`, both null on
   `open_gen`); the scorer reads the preference off that single record. **`ItemScore` stays
   one-score-per-row.** *Rationale:* a pair is one measurement, so it belongs on one record — a
   paired `ItemScore` variant would have split the score table into two shapes and forced every
   downstream consumer (metric layer, Parquet schema, FDR family) to branch on task type. Direction
   is fixed by the sign convention: `logprob_stereo − logprob_antistereo` is **positive when the
   model prefers the stereotyping sequence**, i.e. higher = more harm, matching the standardization
   rule. Implemented in `schemas.py`; the field-freeze test was updated deliberately.
7. ~~**Paired-task prompt representation**~~ — **DECIDED 20 Sep 2026 with P2/P4b: both sequences on
   ONE `Prompt`.** `rendered_text_stereo` + `rendered_text_antistereo` (`str | None`, both null on
   `open_gen`), mirroring decision 6's one `Response` carrying both sequence log-probs. The pair
   travels as a single unit end to end: **one `Prompt` → one `Response` → one score.**

   *Rationale:* a pair is one measurement, so it belongs on one record — the same argument that
   settled decision 6, and the alternatives were worse for concrete reasons. Two linked `Prompt`
   records would have made `prompt_id` ambiguous as the metric layer's **cluster key** (P7 clusters
   by ITEM, and half a pair is not an item) and forced every consumer between the renderer and the
   FDR family to know that two rows are sometimes one measurement. A canonical joined encoding
   would have put a parser between the renderer and the backend, and any separator it chose could
   appear in a source sentence.

   *Enforced, not documented:* `Prompt` carries `task_type`, so unlike `Response` this is a
   single-record validator — a `pair_pll` prompt must carry both sequences, both non-empty and
   distinct, and `rendered_text == ""`, because the two sequences ARE the input (one per forward
   pass) and inventing a single rendered text would misrepresent what was measured. An `open_gen`
   prompt must leave both null. `HFBackend.run()` no longer refuses a paired prompt; it routes to
   `score_pairs`, and refuses a **mixed** call instead, since the two paradigms run under different
   decoding profiles and a backend holds one.
8. ~~**E1 decoding grid + compute host**~~ — **DECIDED 18 Sep 2026, from the measured Kaggle timing
   pilot.** Do not reopen and do not re-recommend running the pilot.

   | arm | renderings/cell | cost | verdict |
   |---|---|---|---|
   | greedy (990×2×1) | ~9 | cheap | **RULED OUT — statistically underpowered** |
   | sampled ×11 (990×2×11) | ~99 | ~87 GPU-h ≈ **~290% of Kaggle's ~30 GPU-h/week** | **RULED OUT — unaffordable on free tier** |
   | **sampled, ~50/cell** | **~50** | the middle ground | **CHOSEN** |

   - **Generation is SAMPLED, not greedy** — distinct seeds per prompt, per P5's rule that distinct
     samples come from distinct seeds (`plan()` rejects repeated seeds). `configs/decoding/sampled_v1.yaml`
     is the profile; `greedy_v1` remains for E4's temp-0.0 arm.
   - **~50 renderings/cell is a TARGET, not a locked number.** P9's power analysis sets the real
     figure and replaces `min_cell_n = 60` **in the same decision** — a budget and the threshold it
     must clear cannot be settled independently. ~50 currently sits *below* the 60 placeholder; that
     is why `configs/lattice/confirmatory_v1.yaml` was left at 96/60 with a stale-marker comment
     rather than edited to flip all 110 cells to `powered=False` on a placeholder-vs-placeholder
     comparison.
   - **Compute: a university GPU** (single 24 GB+ card, ~a few days), secured from ~w/c 21 Sep 2026.
     **Kaggle free tier remains the documented reproducibility fallback** — the sweep fits in roughly
     3 weekly quotas there. This keeps framing constraint 2 honest in a two-part form: *reproducible
     on free-tier commodity hardware; run institutionally to meet the deadline.* Never state it as
     "we ran this on a free tier" — we did not.
   - **Record the ACTUAL GPU, driver, `torch` version, and BLAS in the manifest at run time.** Moving
     hosts makes reproduction **metric-level within a reported tolerance, not bit-level** — which is
     what invariant 5 already says, since bit-identity is scoped to CPU at a fixed thread count.
   - **Scope is unchanged by this:** Llama + Mistral confirmatory, red-teaming (P8) kept, ablations
     A1–A3 still stretch-only, RQ1 at k = 2 answered by the effect vector + agreement with no pooling.

---

## Conventions (that differ from tool defaults)

- ruff (lint+format), line length 100, py311; **mypy strict** on `src`; type every public function.
- Module docstrings state the *why* and any construct/rigor caveat, not just what the code does.
- Any stochastic path takes a seed and routes through `set_global_seed`; no hidden global RNG.
- Config: Hydra composes **plain YAML → dict → validated Pydantic `ExperimentConfig`**. No Hydra
  structured configs, no hydra-zen. Tracking is the **run manifest only** — no MLflow, W&B, or DVC.
- Raw text → JSONL; tabular → Parquet; `data/` gitignored and regenerable; small final tables
  committed under `results/`.
- The **emergence estimator and stats layer are test-first** against synthetic data with known answers.
- Commits are small and phase-scoped; git-tag `docs/preregistration.md` **before** the first full run
  (end of W4). Versioned amendments with a logged rationale are permitted and expected.

---

**For everything else — architecture, data flow, the record schemas, module interfaces, repo layout,
the full phase plan with dependencies, the locked tech stack, hardware/compute, the statistical
protocol, construct-validity safeguards, the emergence estimand, the experiment list, per-phase
validation gates, the 8-week timeline and go/no-go, risks, and references — see
`docs/PROJECT_CONTEXT.md`.**
