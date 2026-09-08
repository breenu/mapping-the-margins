# PROJECT_CONTEXT.md — Mapping the Margins (deep reference)

Full context and plan for building this project from scratch. The always-on essentials (current
status, commands, non-negotiable framing constraints, design invariants, guardrails, open decisions)
live in the lean root **`CLAUDE.md`**; this file is the detailed reference to read before implementing
a new phase. **Build status is tracked in `CLAUDE.md`, not here** — as of writing, nothing is
implemented; the first task is P0.

> **Deadline premise — verify before relying on it.** This plan is built backwards from a FAccT
> **abstract registration on 27 Oct 2026** (mandatory, non-placeholder) and **paper on 3 Nov 2026**,
> with work starting ~8 Sep 2026 — **8 weeks total, 7 to the abstract**. FAccT 2026 decisions would
> normally predate Sep 2026, so these dates most likely belong to **FAccT 2027**; confirm the year
> against the official CFP and correct §13 if it differs. Everything in §13 keys off these two dates.

---

## 1. What we're building, in one screen

Most LLM bias tooling is **single-axis** (gender *or* race *or* religion). Intersectional bias concerns
**combinations** (Black women, disabled elderly men, working-class queer people) and is **emergent**:
the bias at an intersection is not the sum of its constituents' biases and can differ from them
entirely (Guo & Caliskan 2021, for embeddings). Single-axis auditing structurally cannot see it.

The field has established *that* intersectional bias exists in open-weight LLMs. What it lacks — and
what this project supplies — is a **reproducible instrument** with:
1. native intersectional metrics in a reproducible toolkit (no existing toolkit has this),
2. unification of static benchmarks + automated red-teaming under one comparable intersectional score,
3. a statistical rigor protocol (FDR, power, seed/temperature sensitivity) that all but one prior audit lacks,
4. a generative-level operationalization of emergence,
5. lattice coverage beyond gender-anchored intersections (real non-gender×non-gender cells),
6. all at accessible / free-tier compute.

### Research questions
- **RQ1:** Can a unified intersectional bias score be computed *comparably* across heterogeneous
  **paradigms** — extractive sentence-likelihood benchmarks, templated open generation, and
  red-team-generated prompts?
- **RQ2:** Do open-weight LLMs exhibit **emergent** intersectional bias in open-ended generation
  (bias at intersections exceeding what constituent single-axis biases predict)?
- **RQ3:** Does intersectionally targeted **red-teaming** surface biases static benchmarks miss,
  incl. non-gender×non-gender intersections?
- **RQ4:** How **sensitive/reproducible** are intersectional bias measurements across decoding
  parameters and seeds?

### Objectives
1. Release the framework (native intersectional lattice; generative emergent-bias metric extending
   Guo & Caliskan; unified cross-paradigm score).
2. Embed a mandatory statistical protocol (FDR across the whole pre-registered family; pre-registered
   min cell sizes + power; seed/temperature/decoding sensitivity).
3. Build in construct-validity safeguards (item screening; per-source "what it measures" notes;
   ≥2 independent scorers per harm type; **scorer identity-drift control**).
4. Demonstrate end-to-end on an open-weight 7–8B model at accessible-compute scale with pinned,
   deterministic pipelines.

### Confirmatory scope (LOCKED — narrow from day one, breadth-additive)

The schedule (§13) allows 8 weeks against a hard external deadline. The project therefore **starts
narrow and adds breadth only if ahead**. Breadth is what gets cut; **a claim is never cut**.

**In the confirmatory scope:**
- **Two paradigms, k = 2 sources:** one templated `open_gen` generation path (HolisticBias descriptors
  → templates) **and** CrowS-Pairs (extractive `pair_pll`).
- **One model family** at 7–8B for the confirmatory result, plus a ~1B model for development.
- **Red-teaming (P8) is IN** — it is claimed novelty #2 and #5 and cannot be cut.
- **Scorer identity-drift control is IN** — without it RQ2's estimand is not credible (§9).

**Out of scope unless ahead of schedule** (breadth only, no claim depends on any of these):
WinoBias, StereoSet, BBQ-calibration, a second model family, ablations A1–A3.

Every novelty claim must be demonstrated **inside** the narrow scope:

| Novelty claim (above) | Demonstrated in narrow scope by |
|---|---|
| 1. Native intersectional metrics in a reproducible toolkit | The metric layer + `EffectEstimate`, exercised on both paradigms |
| 2. Benchmarks **+** red-teaming under one comparable score | CrowS-Pairs (extractive `pair_pll`) + templated `open_gen` + red-team, all producing standardized effects on one scale |
| 3. Statistical rigor protocol | Unchanged and full — FDR, power, cluster bootstrap, sensitivity (§8) |
| 4. Generative operationalization of emergence | The `open_gen` path + interaction estimator + permutation baseline (§10) |
| 5. Non-gender × non-gender lattice coverage | Reachable only via `open_gen`/red-team — both in scope; a named gate (§11 P3) |
| 6. Accessible / free-tier compute | Kaggle free tier, demonstrated by the actual runs (§7) |

**RQ1 is a cross-*paradigm* claim, not a cross-*dataset* one.** At k = 2 the two sources differ
maximally *in kind* — extractive sentence log-likelihood vs. templated open generation — rather than
being two similar sentence-pair datasets. Comparability across two maximally different paradigms is a
**stronger** test of the unification claim than agreement among four similar benchmarks. State it that
way; it is a design argument, not an apology for scope.

### Why the framing constraints matter (rationale for the CLAUDE.md non-negotiables)
- The *finding* "intersectional bias exists in open-weight LLMs" is published at top venues (LABE/Wan &
  Chang ACL 2025; IndiBias NAACL 2024; Neutral Is Not Unbiased EMNLP Findings 2025). Claiming it as
  our contribution → correct rejection for non-novelty.
- The novelty is an **engineering + methodology** gap (a general, paradigm-unifying,
  red-teaming-integrated, statistically disciplined, accessible-compute instrument with native
  intersectional metrics). It is real but narrow, verified against primary sources, and perishable —
  protect it with precise wording and a final novelty re-check before submission.

---

## 2. Architecture & data flow

Pipeline **data flow** (the paper's logical order):

```
Raw benchmarks    ─▶ [Loaders + Screening] ─▶ Item[]           (extractive sources only)
Descriptor source ─▶ [Lattice] ─▶ IdentityCell[]               (HolisticBias values, w/ provenance)
                     [PromptGenerator] ─▶ Prompt[] ◀─ [RedTeam] (LM-generated, cell-targeted)
                     [InferenceBackend] ─▶ Response[]   (text + optional token_logprobs)
                     [Scorer ×N + triangulation + drift control] ─▶ ItemScore[]  (standardized WITHIN source)
                     [Metric: emergence + cross-paradigm] ─▶ CellStat[] + EffectEstimate[]
                     [StatsEngine: FDR + power + cluster bootstrap + sensitivity] ─▶ EffectEstimate[] (corrected)
                     [Viz + Report] ─▶ figures/, results/tables/, results/paper_numbers.json
```

**Build order differs from data flow** (§5): inference is built *before* red-teaming because
red-teaming *consumes* inference, and the **generation** path is built before the extractive path
because generation is where RQ2 lives.

**Key structural insight:** the "unified intersectional metric layer" the paper promises is not an
algorithm bolted on at the end — it is the **shared record schema (§3) plus per-source
standardization (P6/P7)**. Every module depends only on the record types, never on another module's
internals. That is what makes sources/scorers/backends pluggable. Get the schema right and
comparability is nearly free; get it wrong and no downstream cleverness recovers it.

**"Source" vs "benchmark".** Throughout, a *source* is anything that produces `Item`s or `Prompt`s
carrying a `benchmark` label: an ingested dataset (CrowS-Pairs), the templated `open_gen` set, or the
red-team set. The `benchmark` field ranges over sources, not only over ingested datasets. This matters
for the FDR family (§8) and for the "standardize within, never across" rule.

---

## 3. The contract layer — record schemas (implement in P1, `src/margins/schemas.py`)

Target design: frozen Pydantic v2 records, `extra="forbid"`, immutable, source-agnostic at the top
level. These records ARE the unified metric layer.

- **`TaskType`** (enum): `pair_pll` (sentence-pair likelihood; for decoder-only models this is
  autoregressive sentence log-likelihood — a documented construct choice), `coref` (WinoBias,
  reformulated as a prompted task; out of scope unless ahead), `qa` (BBQ, calibration-only; out of
  scope unless ahead), `open_gen` (main audit + red-teaming — **the primary task type**).
- **`Provenance`**: `source`, `revision`, `version`, `license`, `sha256`, `retrieved_at`. Required on
  every `Item` and every `AxisValue` — descriptor sources need provenance exactly as ingested datasets
  do (§11 gate P3). Define `revision` as the upstream VCS/dataset revision and `version` as the
  upstream release label; if a source has only one of the two, record it and leave the other `None`
  rather than duplicating.
- **`Item`**: `id`, `benchmark`, `task_type`, `payload: Mapping[str, Any]` (the ONLY place source
  specifics live), `target_axes: tuple[str,...]`, `gold: str|None`, `provenance`.
  - `gold` is for tasks with a categorical answer (`qa`, `coref`) only. For `pair_pll` **both**
    sentences live in `payload`; there is no single gold string.
  - `payload` is typed at the **loader boundary** by a registry of per-source `TypedDict`s. The shared
    record stays source-agnostic, but mypy strict is not blind on the one field carrying the real
    data. `payload` must be a frozen/immutable mapping (see immutability note below).
  - **`Item` is the carrier for every path, not only ingested datasets.** A **template** is an `Item`
    (`benchmark="open_gen_templates"`, `task_type=open_gen`, `payload={template_text, slots}`,
    `provenance` → the template file + version); a **red-team seed** is an `Item`
    (`benchmark="redteam"`, `provenance` → the generator model, revision, and seed). This keeps
    `Prompt.item_id` **required** rather than optional, makes `TemplateRenderer.render(items, cells)`
    a genuine cross-product of templates × cells, and preserves the property that every `Prompt`
    traces to something carrying `Provenance`. Without it, templated prompts would have no audit
    trail and no version anchor.
- **`AxisValue`**: `axis`, `value`, `provenance`. The lattice's value registry — this is where
  descriptor provenance lives and what satisfies gate P3's "every lattice value traces to a source."
  **`IdentityCell` carries no `provenance` field**: an order-2 cell combines two values from
  potentially different sources, so a single provenance field on the cell would be ambiguous.
  Provenance attaches to the value, not to the combination.
- **`IdentityCell`**: `cell_id`, `axes: tuple[tuple[str, str], ...]`, `order: int` — validate
  `order == len(axes)`, `1 ≤ order ≤ 2`, axes non-empty and key-unique, and every `(axis, value)`
  present in the `AxisValue` registry. Expose a `.axes_map` property returning a plain dict for
  ergonomics.
- **`Prompt`**: `prompt_id`, `item_id`, `cell_id: str|None`, `rendered_text`, `task_type`,
  `template_version`, `decoding_key`.
- **`Response`**: `response_id`, `prompt_id`, `model_id`, `model_revision`, `seed`, `temperature`,
  `decoding_key`, `decoding_hash` (hash of the fully resolved decoding config), `quant` (recorded
  constant), `text`, `token_logprobs: tuple[float,...]|None`.
  - `decoding_hash` exists because `top_p`, `top_k`, `max_new_tokens`, and `repetition_penalty` are
    otherwise recorded nowhere, which would make a `Response` uninterpretable without the config that
    produced it. Determinism is a contract (invariant 5); a response must be self-describing.
- **`ItemScore`**: `score_id`, `response_id`, `scorer_id`, `harm_type`, `raw`, `standardized`
  (signed bias index standardized WITHIN source), plus the **denormalized** keys
  `cell_id`, `benchmark`, `model_id`, `seed`, `temperature`, and `condition`.
  - **Why denormalized:** without these, `Metric.compute` would need a four-hop join
    (`ItemScore`→`Response`→`Prompt`→`Item`) and E4's sensitivity analysis would have no access to
    `seed`/`temperature`. The redundancy is deliberate: flat Parquet, joinless metric layer,
    self-describing rows.
  - `condition ∈ {"model", "drift_control"}` distinguishes scores of model output from scores of the
    matched neutral control texts (§9).
- **`CellStat`** — **descriptive only**: `cell_id`, `benchmark`, `model`, `harm_type`, `n`,
  `mean_standardized`, `ci_low`, `ci_high`, `powered`. Validate `ci_low ≤ ci_high`.
  **No `p`, no `q`.** A `CellStat` describes a cell; it does not test a hypothesis.
- **`EffectEstimate`** — **inferential**: `benchmark`, `model`, `axis_pair`, `term`, `harm_type`,
  `estimate`, `se`, `ci_low`, `ci_high`, `p: float|None`, `q: float|None`, `powered`,
  `family_size: int|None`, `correction: str|None`. Validate `ci_low ≤ ci_high`.

**Why `CellStat` was split.** The original single record was keyed by `cell_id` but carried
`effect = "the interaction estimate"`. An interaction belongs to an **axis pair × term**, not to a
single cell, and order-1 cells have no interaction at all — so `effect` was undefined for half the
record's legal instances. The split makes the unit of analysis unambiguous, which in turn makes the
FDR family (§8) definable and the pre-registration writable.

**Immutability note (invariant 1).** Pydantic's `frozen=True` prevents attribute reassignment but does
**not** deep-freeze container fields; a `dict` field stays mutable through its reference and makes the
record unhashable. Hence `axes` and `token_logprobs` are tuples and `payload` is an immutable mapping.
`IdentityCell` must be hashable — it is used as a grouping key throughout the metric layer.

**Module interfaces (P1, `src/margins/interfaces.py`)** — structural runtime-checkable `Protocol`s so
modules are swappable and fakeable in tests:
- `BenchmarkLoader.load() -> list[Item]` (has `name`) — **extractive sources only**
- `Lattice.cells() -> list[IdentityCell]` (consumes a provenance-carrying descriptor source)
- `PromptGenerator.render(items, cells) -> list[Prompt]` (deterministic; only place identities meet templates) — **two semantics, see below**
- `InferenceBackend.run(prompts) -> list[Response]` (has `model_id`; only GPU-touching / non-portable stage)
- `Scorer.score(responses) -> list[ItemScore]` (has `scorer_id`, `harm_type`; ≥2 per harm type)
- `Metric.compute(scores, cells) -> tuple[list[CellStat], list[EffectEstimate]]`
- `StatsEngine.correct(effects) -> list[EffectEstimate]` (the ONLY stage that sets `q`, `family_size`, `correction`)

**`Metric.compute` takes `cells` as well as `scores`, deliberately.** `ItemScore.cell_id` is an
opaque string; fitting `harm ~ A + B + A:B` requires the **axis values** behind it, and
`EffectEstimate.axis_pair` cannot be populated from a cell id alone. Passing the lattice (small, and
already in memory) resolves `cell_id → axes` without either parsing structure out of an id string or
denormalizing the axis values onto every score row. The denormalized keys on `ItemScore` cover the
*grouping* keys; `cells` covers the *design matrix*.

**`PromptGenerator` has two incompatible semantics — implement both, document both.**
`render(items, cells)` reads as a cross-product, which is *wrong* for extractive sources: you cannot
assign "Black woman" to a CrowS-Pairs item about religion.
- **`ExtractiveRenderer`** — the cell is **derived** from the item (`pair_pll`, `coref`). Validator: it
  must never emit a prompt whose cell is not attested in the source item.
- **`TemplateRenderer`** — the cell is **injected** into a template (`open_gen`, red-team). This is the
  renderer the MVP exercises first, and the only one that can reach arbitrary lattice cells.

**Open contract question, must be settled before P6 (W3 — on the critical path):** a PLL score
compares *two sequences*, so one `ItemScore` per `Response` does not fit `pair_pll`. Decide between
(a) representing a pair as one `Response` with two logprob fields, or (b) a paired `ItemScore`
variant. Do not defer this past the CrowS-Pairs loader.

**P1 exit gate = a "walking skeleton":** no-op fakes for every Protocol run end-to-end on ~3 fake items,
proving the contracts compose before any real logic exists.

---

## 4. Repository layout (target — nothing built yet)

```
mapping-the-margins/
├── pyproject.toml            # pinned deps; inference/llamacpp/dev extras; ruff/mypy/pytest config
├── uv.lock                   # exact resolved environment (see §7 for tool.uv.environments)
├── .pre-commit-config.yaml   # ruff, mypy, check-added-large-files
├── CLAUDE.md                 # lean always-on memory (status, commands, invariants, guardrails)
├── README.md                 # human-facing overview + build status
├── configs/                  # Hydra tree: config.yaml + model/ benchmark/ lattice/ prompts/templates/ scoring/ experiment/
├── src/margins/
│   ├── schemas.py            # P1 — frozen records (contract layer)
│   ├── interfaces.py         # P1 — Protocols for every stage
│   ├── repro.py              # P0 — set_global_seed + residual-nondeterminism note
│   ├── config.py             # P0 — ExperimentConfig + run-manifest writer
│   ├── benchmarks/           # P2 — EXTRACTIVE loaders + screening (CrowS-Pairs first)
│   ├── lattice/              # P3 — axes, cells, power bookkeeping
│   │   └── descriptors/      #      HolisticBias descriptor/value source (+ provenance). NOT a loader.
│   ├── prompts/              # P4a Template renderer (with the generation vertical); P4b Extractive (with P2)
│   ├── inference/            # P5 — backend + resumable run harness
│   ├── scoring/              # P6 — scorers + triangulation + calibration + drift control
│   ├── metrics/              # P7 — emergence estimator + cross-paradigm comparison
│   ├── redteam/              # P8 — cell-targeted generation
│   ├── stats/                # P9 — FDR, power, cluster bootstrap, permutation, sensitivity
│   ├── viz/                  # P10 — figures
│   └── report/               # P10 — table/report assembly + paper_numbers.json
├── data/                     # GITIGNORED + regenerable (benchmarks, lattice, prompts, responses, scores, metrics, redteam)
├── results/                  # small committed final tables + run_manifests/ + paper_numbers.json
├── figures/                  # regenerable vector figures
├── scripts/                  # run_experiment.py (hydra entrypoint), make_figures.py, reproduce.sh
├── tests/                    # pytest (schemas, schema field freeze, repro/config, walking_skeleton, + per-module)
├── notebooks/                # exploration only; NEVER load-bearing
└── docs/                     # PROJECT_CONTEXT.md, preregistration.md (tagged before full runs), model_card, data_card, methods, reproduction
```

**`data/` is gitignored and regenerable, not version-controlled.** There is no DVC (§6). The guardrail
"never commit model weights or raw response dumps" is enforced by `.gitignore` plus a
`check-added-large-files` pre-commit hook. Raw response dumps are published as a **Zenodo artifact at
release** (post-submission), which is where a reproducer gets them.

### Run manifest (P0 — written by `margins/config.py`)

Emitted per run to `results/run_manifests/<run_id>.json`. Must contain enough to reconstruct the run:

- `run_id` — **recorded, not random-per-call**; derive it from the config hash plus a recorded
  timestamp so the same config is traceable across runs.
- `config_hash`, plus the fully resolved config
- `git_sha` **and a dirty-worktree flag** (a manifest from a dirty tree that claims a clean SHA is
  worse than no manifest)
- package versions; Python version; platform
- `seed`
- `torch.__version__`, thread count, and BLAS backend when the inference extra is present — without
  these the determinism claim in invariant 5 is not checkable by a reproducer
- input checksums (`sha256`) for every source consumed
- the `quant` constant

---

## 5. Execution plan — phases, build order, dependencies

Phases in **build order** (not data-flow order). Cross a phase only when its validation gate (§11)
passes. Update completion status in `CLAUDE.md`.

**Phase IDs are stable labels, not a sequence.** The table below is sorted by *build order*, so the
IDs deliberately run out of numeric order (P3 before P2). IDs are referenced by §11, §12 and §13 and
are never renumbered; read the leftmost column as a name, not a position.

**The generation vertical is built first, the extractive vertical second** — decided 8 Sep 2026, see
*Spine to build first* below for the rationale. **P4 is split accordingly**: `TemplateRenderer`
depends only on the lattice and rides with the generation vertical, while `ExtractiveRenderer`
cannot be built before a loader exists and rides with CrowS-Pairs.

| Build order | # | Phase | Depends on | Exit criteria |
|---|---|-------|-----------|---------------|
| 1 | P0 | Env + repro + config | — | pinned env (uv), `set_global_seed`, run-manifest writer, CI on clean checkout |
| 2 | P1 | Schemas + interfaces | P0 | frozen records + Protocols + walking-skeleton test + schema field-freeze test |
| 3 | P3 | Identity lattice | P1 | axes/values from the HolisticBias descriptor source w/ provenance; ≥2 non-gender×non-gender cells; cell inventory w/ projected N + power flags |
| 4 | **P4a** | Prompt generation — `TemplateRenderer` (`open_gen`) | P3 | deterministic versioned prompt set from the template renderer; template-fill audit |
| 5 | P5 | Model inference | P0, P4a | quantized backend; `generate()` + `sequence_logprob()`; determinism + resumability tests |
| 6 | P6 | Scoring + triangulation + drift control | P5 | ≥2 scorers/harm; inter-scorer agreement; **scorer identity-drift control quantified**; calibration on hand-labeled sample |
| 7 | P7 | Emergent metric + cross-paradigm comparison | P6 | interaction estimator (synthetic-recovery tested); permutation baseline; cross-paradigm effect vector + agreement |
| 8 | P2 | Extractive benchmark ingestion + validity screening | P1 | **CrowS-Pairs** loader; screening report; excluded-item log. (WinoBias/StereoSet/BBQ-calib: out of scope unless ahead) |
| 9 | **P4b** | Prompt generation — `ExtractiveRenderer` (`pair_pll`) | P2, P4a | `ExtractiveRenderer` emits no cell unattested in its source item; byte-identical regeneration |
| 10 | P8 | Automated red-teaming | P5, P6 | cell-targeted prompts (confirmatory pairs only); manual on-target audit; dedup/filter |
| 11 | P9 | Statistics layer | P6, P7 | FDR + power + cluster bootstrap + sensitivity; pre-registration tagged BEFORE full runs |
| 12 | P10 | Visualization + reporting | P7, P9 | one-command figure regeneration; `paper_numbers.json` emitted |
| 13 | P11 | Experiments | P0–P10 | confirmatory result set (E1, E3, E4, baselines, R1/R2) |
| 14 | P12 | Paper | P11 | manuscript + supplement |
| 15 | P13 | Open-source release | P11 | tagged release + Zenodo DOI + model/data cards + reproduction script (post-submission) |

**Note on E2/RQ1 and the cut order.** RQ1's cross-paradigm claim needs *both* verticals, so it cannot
land until P2 + P4b are done. That is the deliberate cost of front-loading the generation path: if the
schedule slips, RQ1 is what degrades, while RQ2 — the primary estimand — is already secured. This
matches §12's cut order (E1 → B1/B3 → E4 → E2 → E3), which already ranks the cross-paradigm
experiment below the main audit.

### Spine to build first — the MVP is the GENERATION path

The MVP's first vertical is **generation**, because that is where RQ2 lives and where the risk is:

> HolisticBias descriptor values → **lattice (P3)** → **templates (P4a)** → inference (P5, ~1B dev
> model) → scoring + **scorer identity-drift control** (P6) → emergence estimator (P7) → stats (P9)
> → one figure.

**This ordering is DECIDED (8 Sep 2026), not a preference — it is written into the §5 table above.**
Against a 7-week clock to the abstract, the phase that must not be reached late is the one carrying
the primary estimand. Building generation first means a slip costs *breadth* (RQ1's second paradigm),
which the confirmatory scope already treats as cuttable; building extraction first would mean a slip
costs a *claim*.

**CrowS-Pairs (P2, extractive) is the SECOND vertical**, added once the generation path is green. It
is what makes RQ1's cross-paradigm comparison possible, but it is not the risky path and **must not be
built first** — building it first would validate the easy half of the pipeline and leave scoring,
triangulation, and the primary estimand untested.

**P4 is split along the same seam.** `TemplateRenderer` (**P4a**) *injects* a cell into a template and
depends only on P3, so it belongs to the generation vertical and is the next build after the lattice.
`ExtractiveRenderer` (**P4b**) *derives* a cell from a source item and cannot exist before a loader
does, so it depends on P2 and rides with the second vertical. The two renderers were always
documented as having incompatible semantics (§3); the split makes that difference a scheduling fact
rather than a footnote.

**HolisticBias is a descriptor/value taxonomy, NOT a benchmark loader.** `open_gen` is a *generation*
path, not an ingestion path. HolisticBias supplies **lattice axis values with provenance** (P3,
`margins/lattice/descriptors/`) and **template slot fillers** (P4a). It must never appear as a
`BenchmarkLoader`.

**Never build:** red-teaming before inference+scoring; any cross-paradigm comparison before ≥2 sources
produce comparable effects; any significance before StatsEngine; full experiments before the
pre-registration tag.

**Parallelizable once the spine exists:** the extractive loader, extra scorers, red-teaming, viz, docs
(independent behind the interfaces).

**Test-first components:** the emergence estimator (synthetic known-interaction data) and the stats
layer (FDR monotonicity, power sanity). Silent bugs here become false paper claims.

---

## 6. Technical stack (locked — one per slot)

- **Language:** Python 3.11. **Dep manager:** `uv` (pinned + lockfile).
- **Inference backbone:** Hugging Face `transformers` (single backbone — needs both token log-probs
  AND generation). `llama.cpp`/GGUF is the documented CPU/Apple fallback behind the same interface.
- **Quantization:** 4-bit NF4 via `bitsandbytes` on NVIDIA; GGUF Q4_K_M via llama.cpp fallback.
  Fixed constant, recorded in every manifest.
- **Config:** Hydra (composition) + Pydantic (validation). **Hydra composes plain YAML → dict →
  validated into a Pydantic `ExperimentConfig`. No Hydra structured configs, no hydra-zen** — the
  Hydra/Pydantic friction is real and this is the cheap side of it.
- **Experiment tracking:** the **run manifest** (§4) only. No MLflow, no W&B, no DVC — the manifest
  plus Parquet outputs already capture everything, and the UI value is low when runs are few and
  batch. Removing them removes maintenance and a Windows failure surface.
- **Stats:** `statsmodels` (`multipletests` BH-FDR; GLM interaction with cluster-robust SEs; power;
  `stats.meta_analysis.combine_effects` — **only at k ≥ 3**, see §10) + `scipy` (`bootstrap`) +
  `pingouin` (effect sizes) + `krippendorff` (**Krippendorff α is not in `pingouin`**; Cohen κ comes
  from `scikit-learn`).
- **Viz:** `matplotlib` + `seaborn`, vector PDF/SVG.
- **Storage:** JSONL (raw prompts/responses), Parquet (tabular scores/stats), committed small tables
  under `results/`. `data/` is gitignored and regenerable.
- **Layout:** `src/` package, `pip install`-able. **Testing:** `pytest` + synthetic metric tests.

Heavy inference deps (`transformers`, `torch`, `bitsandbytes`, `datasets`, `evaluate`, `detoxify`) are
**optional extras**, so the non-inference stages install/test on any laptop without a GPU stack.

**Scorer cost note:** an LLM-judge scorer carries an unbudgeted compute or API cost. Under the §13
timeline, prefer **two cheap local scorers plus the drift control (§9)** over an LLM judge. If a judge
is used it is one scorer of ≥2, with a fixed documented prompt and seed — never ground truth.

---

## 7. Hardware & compute plan

**Decided (resolves open decision #1): Windows laptop for development, Linux / free-tier Kaggle GPU
for inference.** The 7–8B sweep never runs locally.

- **Windows laptop (`win_amd64`):** everything EXCEPT the 7–8B generative sweep and red-team
  generation — i.e. dev, schemas, lattice, prompts, scoring on cached outputs, metrics, stats, viz,
  paper. The default `uv sync` installs **core + dev extras only**; no `torch`, no `bitsandbytes`.
  - **P5 amendment (9 Sep 2026).** The `inference` extra is now *installable* on `win_amd64`:
    `torch`/`transformers`/`accelerate`/`huggingface-hub` carry no platform marker, while
    `bitsandbytes` (and `datasets`/`evaluate`/`detoxify`) stay Linux-gated. This is what lets the
    P5 determinism, sequence-logprob and resumability gates run locally against the **~1B dev
    model**, and lets `scripts/pin_models.py` resolve a revision from the dev box. **Installable is
    not installed**: the default sync is unchanged, NF4 remains a CUDA/Linux path (a Windows load
    falls back to unquantized and records that it did), and **the 7–8B sweep still never runs
    locally**.
- **Kaggle (`linux_x86_64`), free tier:** the generative sweep (P5) and red-team generation (P8),
  with the `inference` extra and `bitsandbytes` NF4. ~30 GPU-h/week, 12-h sessions, persistent
  datasets, batched and **resumable** harness. **Colab is the debugging backup.**
- **`uv.lock` must resolve for exactly two environments.** Pin `tool.uv.environments` to
  `win_amd64` and `linux_x86_64`. Without this, `uv lock` attempts to resolve `bitsandbytes` for
  Windows, and a resolution failure there creates pressure to unpin — which the guardrails forbid.
  Declaring the environments up front is the supported way to avoid that.
- **University GPU:** convenience only, never a dependency; if used, verify one full config reproduces
  on Kaggle.
- **Paid (only if scaling >8B or needing speed):** spot RTX 3090/4090 on RunPod/Vast (~$0.20–0.45/h);
  a full sweep is single-digit to low-tens of dollars.
- **Scope-to-feasibility rule:** pick model sizes + prompt counts so reference experiments finish within
  free Kaggle quota; keep the full seed×temperature grid to the **sensitivity subset**, not the whole
  corpus. State this as a design choice, not a hidden limitation.
- **Calibrate with a timed 50-prompt pilot on real hardware before committing scale (W2 — §13). Do not
  trust an estimate over a measurement.**

### Environment preconditions (settle these before P0 runs)

- **Move the repository out of OneDrive.** The current location is inside a OneDrive-synced folder.
  OneDrive will attempt to sync `.venv/` and `data/`, hold file locks during sync, and create
  sync-conflict duplicates (`file (2).parquet`) *inside* data directories. That is a reproducibility
  hazard, not a nuisance. If the repo cannot be moved, `.venv/`, `data/`, and `results/` must be
  explicitly excluded from sync.
- **`git init` + a remote is step zero.** The run manifest records a git SHA and the pre-registration
  requires a git tag; neither exists without a repository.
- **`uv` is not installed** on the target machine, and the system Python is 3.14 while the project
  pins 3.11. P0's first action is installing `uv`; `uv` then provisions 3.11.

### CI (part of P0's exit criteria)

Free GitHub Actions has **no GPU**; say so in the docs rather than implying CI covers inference.

- **Core job** — every push: `uv sync` (core + dev), `pytest`, `ruff check`, `ruff format --check`,
  `mypy src`. No `torch`.
- **CPU inference job** — nightly or on-label: `uv sync --extra inference` (CPU torch) and the P5
  determinism, sequence-logprob and resumability tests against a tiny random Llama checkpoint,
  `hf-internal-testing/tiny-random-LlamaForCausalLM` — **availability verified 9 Sep 2026**, pinned
  at `9fb191250dd56d0ba7ec9785a025ed29c03d5998` in `configs/model/tiny_random_llama.yaml`. It is
  ungated, which matters: the job has no HF token and cannot reach the gated Meta pins. `OMP_NUM_
  THREADS` is pinned in the job, because the determinism claim holds at a *fixed* thread count.

---

## 8. Statistical protocol (headline novelty — make these MANDATORY pipeline outputs)

- **Pre-registration first.** Commit + **git-tag** `docs/preregistration.md` before the first full run
  (§13: end of W4). The tag must predate the first full run in git history. **Versioned amendments
  with a logged rationale are permitted and expected** — an amendment log is honest; a pre-registration
  written blind is not.
- **Pre-registration contents:** hypotheses; the emergence estimand + link/scale; the **named
  confirmatory axis pairs**; minimum cell N from power analysis; the FDR family definition; the
  **bootstrap resampling unit**; seeds; the temperature grid.
- **The confirmatory axis-pair set.** Pre-register a small named set of **6–8 axis pairs**, of which
  **≥2 must be non-gender × non-gender** (claim 5). **That set IS the FDR family.** Everything outside
  it is exploratory: reported descriptively, with no significance claims and no FDR membership. This
  is the single highest-leverage decision for the project producing a result at all — an unbounded
  family over the whole lattice drives the BH threshold so low that realized power collapses
  regardless of per-cell N.
- **FDR family:** **benchmark × model × axis_pair × term × harm_type**, across the entire
  pre-registered set — never per-source. Benjamini-Hochberg
  (`statsmodels.stats.multitest.multipletests(method='fdr_bh')`). **BH assumes independence or PRDS**;
  tests here share items, templates, and a model, so either state the PRDS assumption explicitly or
  use `fdr_by` for the unconditional guarantee. Every `EffectEstimate` records `family_size` and
  `correction`.
- **Clustering — do not skip this.** The same item and the same descriptor are rendered across many
  cells, and templates are reused across cells, so GLM residuals are **clustered by item and by
  template**. A plain GLM assuming independence produces anti-conservative standard errors, and FDR
  cannot rescue inputs that are already wrong. Use cluster-robust SEs (`cov_type='cluster'`) or a
  mixed model with a random intercept for item.
- **Power:** `statsmodels.stats.power`; fix minimum detectable effect at α=0.05, power=0.8; derive
  minimum cell N; mark under-N cells `powered=False`.
- **CIs:** bootstrap (`scipy.stats.bootstrap`) on every headline number, as a **cluster bootstrap over
  items/descriptors** — resampling rows rather than clusters understates variance for exactly the
  reason above. The resampling unit is pre-registered; it is the part that matters.
- **Sensitivity (RQ4):** aggregate seed/temperature/decoding variability as variance + CI on every
  headline metric.
- **Sign harmonization:** the within-source standardization function must guarantee **higher = more
  harm**, with a test. A mis-signed source silently inverts the cross-paradigm comparison while
  producing entirely plausible-looking output.
- **Invariant:** post-FDR significant count ≤ pre-FDR; every headline number carries a CI; no
  significance outside StatsEngine.

---

## 9. Construct-validity safeguards (respond to Blodgett et al. 2021 — a design feature)

- **Screen reused items.** For CrowS-Pairs run minimal-pair checks (edit distance, single-attribute
  swap); flag non-minimal pairs; manually review a stratified sample; log every exclusion with a
  machine-readable reason to `excluded_items.csv`.
- **Document per source** a one-paragraph "what this measures / what it does not" note (`docs/methods.md`).
- **Causal-model caveat:** CrowS-Pairs PLL was designed for masked LMs; Llama/Mistral are
  decoder-only → use autoregressive sentence log-likelihood (or a documented pseudo-LL variant) and
  state it.
- **Triangulate:** ≥2 independent scorers per harm type; report Krippendorff α / Cohen κ; calibrate
  against 150–300 hand-labeled responses; downgrade poor-agreement harm types to descriptive.

### Scorer identity-drift control (REQUIRED — not a caveat)

Detoxify and similar classifiers are documented to score text higher **purely for mentioning identity
terms**, independent of harm. Our estimand is "harm varies non-additively across identity cells." A
scorer whose own response to identity mentions is non-additive across term pairs will therefore
produce a **non-zero interaction coefficient from text that is identical across cells** — a fully
fabricated emergence result.

**"≥2 scorers with reported agreement" does not address this.** Two identity-sensitive classifiers
will agree with each other and both be wrong in the same direction. Agreement measures shared
behaviour, not shared validity.

> **The control.** Score a matched set of known-neutral texts containing the same identity descriptors
> through every scorer, fit the same `A + B + A:B` model on *those*, and report each scorer's own
> interaction coefficient. Reported emergence is the model's interaction **net of** the scorer's, or
> the two are presented side by side. If a scorer's interaction is comparable in magnitude to the
> model's, that harm type is **descriptive-only**.

Because the descriptors come from HolisticBias and the prompts from templates (§5), the neutral
control texts **reuse the same descriptor set and the same templates** with a neutral carrier
sentence. The control is therefore exactly matched by construction and costs one extra scoring pass
over cheap text — no additional generation. `ItemScore.condition` distinguishes the two conditions.

---

## 10. The emergence estimand (the intellectual core — spelled out)

Guo & Caliskan (2021) established emergence for embeddings ("intersection ≠ sum of parts"). To make it
testable for **generation**, operationalize it as an **interaction effect**:

> For each source×model, fit `harm_score ~ axisA + axisB + axisA:axisB` (a GLM with a
> **pre-registered link** — identity link for standardized scores, logit for rates — and
> **cluster-robust SEs by item**, §8). The coefficient(s) on `axisA:axisB` **are** the emergent
> intersectional bias for those cells. Report estimate + cluster-bootstrap CI + FDR-corrected p.

**Never conflate a high bias *level* at a cell with emergence.** Emergence is specifically the
interaction — the single most common way intersectional audits overclaim. This is why `CellStat`
(descriptive, no p) and `EffectEstimate` (inferential) are separate records (§3).

**The additive null is not an independent baseline.** The additive-null residual
`observed − (grand mean + main-effect deviations)` **is the same quantity** as the interaction
coefficient. B2 is therefore a *presentation* of E1, not corroboration of it, and must not be
reported as a second line of evidence. The genuinely independent check is a **permutation test**:
shuffle axis-B labels within axis-A strata, refit, and build the null distribution of the
interaction. It is assumption-light, requires no link-function commitment, and is cheap.

### Cross-paradigm comparison (RQ1) — conditional on k

- **At k = 2 (the confirmatory scope): NO POOLING.** RQ1 is answered by the **effect vector across
  paradigms plus their agreement** — the per-source standardized interaction estimates, their CIs,
  and a directional / rank agreement statistic. **No pooled estimate, no τ², no I².** With k = 2 the
  between-source variance is not estimable, and reporting a pooled number or an I² would be false
  precision.
- **At k ≥ 3 only:** random-effects pooling via `statsmodels.stats.meta_analysis.combine_effects`
  with I², reported **alongside** the effect vector and with an explicit small-k caveat (τ² and I²
  are poorly estimated below roughly k = 5–10). This returns only if a third source lands with time
  to spare (§1, out of scope unless ahead).

Comparability is an empirical result (report the vector and the agreement), never an assumption. If
the paradigms disagree, that is the finding and it gets said plainly.

**Open decision before P7/P9:** link/scale (identity vs logit) and the exact null model — chosen and
pre-registered by the human.

---

## 11. Validation gates (gate after every stage — do not proceed until green)

| After | Concrete check |
|-------|----------------|
| P0 | Fresh clone + `uv sync` → identical env. **Seeding determinism:** `random`, `numpy`, `PYTHONHASHSEED` and a pure-numpy stochastic function reproduce bit-identically under `set_global_seed`. Manifest reconstructs the run. **No `torch` required** — the toy-model check belongs to P5. |
| P1 | Walking skeleton runs through all interfaces on fake data; schema round-trip + immutability + hashability + order-validator tests pass; **schema field-freeze test** pins the exact field set of every shared record. |
| P2 | Row counts match published dataset; checksums recorded; 5-item eyeball; every exclusion logged with a reason; screening is deterministic. |
| P3 | Every lattice value traces to a provenance-carrying source via the `AxisValue` registry, and every `IdentityCell` axis pair resolves in it; **≥2 non-gender×non-gender primary cells named**; cell inventory marks powered vs descriptive before any run. Note: with CrowS-Pairs as the only extractive source, these cells are reachable **only** via the templated `open_gen` and red-team paths — this gate cannot be met from extractive items and must not slip. |
| P4a | Byte-identical prompt regeneration from the same config; 20-prompt manual slot audit of template fills. (Template renderer only — the `ExtractiveRenderer` clause moved to P4b with the §5 split.) |
| P4b | `ExtractiveRenderer` emits no cell unattested in its source item; byte-identical regeneration from the same config. |
| P5 | Same-seed greedy produces identical token sequences on CPU at a fixed thread count (tiny model); logprob path matches a hand-computed toy value within the documented tolerance; kill/restart resumes without dup/drop. |
| P6 | Inter-scorer α/κ reported; scorer precision/recall on 150–300 labeled items; **scorer identity-drift interaction quantified per scorer and reported**; poor-agreement and high-drift harms downgraded. |
| P7 | Synthetic recovery test: injected interaction recovered within CI; zero-interaction → ~0. Permutation null centred on ~0 under a no-effect synthetic. |
| P8 | Human on-target % on a sample reported; dedup + filter logged; prompts restricted to the confirmatory axis pairs. |
| P9 | FDR reduces significant count; power table names min detectable effect; **pre-reg tag predates first full run**; `family_size` and `correction` stamped on every corrected estimate. |
| P10 | `make_figures.py` regenerates every figure from committed tables; greyscale/colourblind pass; **`results/paper_numbers.json` emitted and every headline number in the manuscript resolves to a key in it**. |
| P11 | Small-scale end-to-end smoke run (tiny model, ~50 prompts, one source, full pipeline → stats → 1 figure) passes BEFORE any full run; known-direction bias shows expected sign. |

**Golden rule:** the small-scale smoke test catches ~90% of pipeline bugs at ~1% of the compute. Always
run it before a full run.

---

## 12. Experimental plan (each maps to an RQ and a figure/table)

**Confirmatory (in scope):**
- **E1 Main audit** (RQ2): the templated `open_gen` set + CrowS-Pairs, over the pre-registered
  confirmatory axis pairs; one 7–8B model family; greedy + fixed temp; emergence per pair; GLM
  interaction with cluster-robust SEs + cluster-bootstrap CI + BH-FDR across the whole family.
- **E3 Red-team vs static** (RQ3): matched cells incl. non-gender×non-gender; paired comparison + FDR.
- **E4 Sensitivity/reproducibility** (RQ4): sensitivity subset; seeds ∈ {≥3}, temp ∈ {0.0, 0.7, 1.0},
  greedy vs sampling; variance/CV of headline metrics.
- **E2 Cross-paradigm comparability** (RQ1): standardized effects → **effect vector + agreement
  statistic; no pooling at k = 2** (§10). A forest-style plot of the two paradigm estimates with CIs
  is the figure; it carries no pooled diamond. Random-effects pooling and I² only if k reaches 3.
- **Baselines:** B1 single-axis (main effects only); **B2 additive-null — a *presentation* of E1's
  interaction, not an independent baseline (§10)**; **B3 permutation test on the interaction — the
  genuinely independent null.** B1 + B3 are what license the *emergence* claim over "bias exists."
- **Robustness:** R1 prompt-paraphrase; **R2 reproducibility rerun scoped to the sensitivity subset**
  (a full E1 rerun would double E1's compute). The reproducibility claim in the paper must match
  exactly what was rerun — never claim reproducibility you haven't demonstrated.

**Out of scope unless ahead of schedule** (breadth, not claims): additional sources (WinoBias,
StereoSet, BBQ-calib) and the k ≥ 3 meta-analysis; a second model family; ablations A1 (screening
on/off), A2 (single-scorer vs triangulated), A3 (lattice depth 1 vs 2).

**Cut order if compute-limited:** E1 → B1/B3 → E4 → E2 → E3 → R1/R2.
**E1 + B1 + B3 + E4 is the minimum supporting the core claims.**

---

## 13. Timeline — 8 weeks to a hard external deadline

Working back from **abstract registration 27 Oct 2026** (mandatory, non-placeholder) and **paper
3 Nov 2026**, starting ~8 Sep 2026. Weeks run Tue–Mon. *(Verify the deadline year — see the note at
the top of this file.)*

| Week | Dates | Work | Must be true at the end |
|---|---|---|---|
| W1 | Sep 8–14 | P0 + P1 | Env locked, manifest writes, walking skeleton green |
| W2 | Sep 15–21 | P3 lattice (HolisticBias values) + **P4a** templates + P5 spine on ~1B dev model + **Kaggle timing pilot** | Prompts regenerate byte-identically; **measured** GPU-h per 1k prompts |
| W3 | Sep 22–28 | P6 scoring: ≥2 scorers + **drift control**; P2 CrowS-Pairs loader + screening + **P4b** extractive renderer; small-scale smoke run | Smoke run passes end-to-end; drift quantified |
| — | **Sep 28–29** | **HARD GO / NO-GO** (criteria below) | Venue decision made deliberately |
| W4 | Sep 29–Oct 5 | P7 emergence estimator (test-first) + P9 stats; **pre-registration written and git-tagged** | Synthetic recovery passes; tag predates any full run |
| W5 | Oct 6–12 | P8 red-teaming (confirmatory pairs only) + **full E1 run** + E4 sensitivity subset | E1 results in hand |
| W6 | Oct 13–19 | E3 + B1/B3 + R1 + **R2 repro rerun**; **results freeze** | The claim is known and stable |
| W7 | Oct 20–26 | P10 figures + paper draft; abstract text finalized from frozen results | Abstract ready — non-placeholder |
| W8 | Oct 27–Nov 2 | **Abstract registered 27 Oct**; paper completion | **Submitted 3 Nov** |

Post-submission: P13 release + Zenodo DOI + model/data cards. Not on the critical path.

**Critical path:** P0→P1→(lattice+templates)→P5→scorer+drift→P7→P9→E1→viz→paper. Everything else
parallelizes off it or is out of scope.

**Schedule consequences to respect:**
- **The timing pilot is in W2, not later.** It sets E1's prompt budget and therefore everything after
  it. Do not trust an estimate over a measurement.
- **The pre-registration tag is due end of W4**, because the first full run is W5.
- **P8 red-teaming is scoped to the confirmatory axis pairs only**, with a ~100-prompt manual
  on-target audit. The binding constraint there is human time, not GPU time.
- **R2 is scoped to the sensitivity subset**, not a full E1 rerun.

### Hard go/no-go — Sep 28–29

Proceed to the FAccT deadline only if **all five** hold:

1. The small-scale smoke run passes end-to-end (§11 P11): one source, ~50 prompts, full pipeline →
   stats → one figure, and a known-direction bias shows the expected sign.
2. The emergence estimator's synthetic-recovery test passes (injected interaction recovered within
   CI; zero-interaction → ~0).
3. The scorer identity-drift control shows scorer interaction **materially smaller** than model
   interaction for **at least one** harm type. If not, the primary estimand is not measurable with
   the available scorers, and no amount of remaining schedule fixes that.
4. The W2 timing pilot projects E1 + E4 + red-team **within the remaining free Kaggle quota** for
   W5–W6.
5. **≥2 non-gender × non-gender confirmatory pairs project as `powered`** at the pre-registered
   minimum N (claim 5).

If any fails, the fallback is **ACL Rolling Review (no fixed deadline) or a non-archival workshop**,
chosen **deliberately at this gate** — not as a panic cut in W7. Record the decision and the evidence
behind it in `docs/methods.md`.

---

## 14. Risks (top; mitigations baked into the plan)

- **Fixed external deadline — 8 weeks** (High/High): narrow confirmatory scope locked from day one
  (§1); W2 timing pilot; W3/W4 hard go/no-go with a named fallback venue; breadth items explicitly
  out of scope rather than "stretch."
- **Compute over free quota** (High/High): scope to Kaggle ~30 h/wk; sensitivity grid on a subset;
  resumable harness; W2 timing pilot.
- **Power collapse in intersectional cells** (High/High): depth fixed at 2; **6–8 pre-registered
  confirmatory axis pairs as the entire FDR family**; pre-registered min N; under-N cells
  descriptive-only.
- **Scorer identity drift fabricating emergence** (High/High): the §9 drift control; high-drift harm
  types downgraded to descriptive.
- **Inherited construct-validity flaws** (High/High): §9 safeguards.
- **Anti-conservative SEs from clustered residuals** (Med/High): cluster-robust SEs by item; cluster
  bootstrap; PRDS stated or `fdr_by`.
- **Single-scorer/LLM-judge bias** (Med/High): ≥2 scorers + agreement + human calibration.
- **Overclaiming emergence (level vs interaction)** (Med/High): §10; B1/B3 baselines; synthetic
  recovery gate; `CellStat`/`EffectEstimate` split.
- **Nondeterminism / repro failure** (Med/High): pin checkpoints/seeds/deps; record torch/thread/BLAS;
  document GPU tolerance; R2 rerun as evidence.
- **Multiple-comparison false positives** (Med/High): BH-FDR across the whole pre-registered family;
  report post-FDR counts and `family_size`.

---

## 15. Key references (related work + method grounding — verify before citing; do not fabricate)

- **Guo & Caliskan (2021)** — emergent intersectional bias (embeddings). The concept extended to generation.
- **Perez et al. (2022)** — red-teaming LMs with LMs. The automation mechanism (not demographic/intersectional — that's the novelty).
- **Blodgett et al. (2021)** — construct-validity failures in these benchmarks. The critique we *respond* to.
- **LABE / Wan & Chang (ACL 2025)**; **IndiBias (NAACL 2024)**; **Neutral Is Not Unbiased (EMNLP Findings 2025)** — establish the *finding* (so it can't be ours).
- **Compounding Disadvantage / Gupta et al. (FAccT 2026)** — the one prior audit with FDR; proof FAccT accepts the genre; we add generality + reusability + accessible compute.
- **LangFair (JOSS 2025), FairLangProc, HELM (2022)** — toolkit genre; primary-source-verified to lack native intersectional support.
- **HolisticBias** — the descriptor/value taxonomy feeding the lattice and templates (§5). A source of
  identity descriptors, **not** a benchmark we ingest as items.
- Benchmarks: **CrowS-Pairs** (in scope); **WinoBias, StereoSet, BBQ (calibration), HONEST** (out of
  scope unless ahead).
- Held for verification: **HInter** (arXiv, not peer-reviewed), **Invisible Influences**, **Lalor et al. (NAACL 2022)**.

---

## 16. Glossary

- **Source:** anything producing records under a `benchmark` label — an ingested dataset, the templated
  `open_gen` set, or the red-team set.
- **Cell / IdentityCell:** a point in the lattice — order-1 (single axis) or order-2 (pairwise intersection).
- **Emergent intersectional bias:** the interaction term in `harm ~ A + B + A:B`. Not a bias level.
- **`CellStat`:** a *descriptive* summary of one cell (n, mean, CI, powered). Carries no p and no q.
- **`EffectEstimate`:** an *inferential* record for one model term (axis pair × term × harm type).
  The only record that carries p, q, `family_size`, and `correction`.
- **Unified / cross-paradigm score:** per-source interaction effects standardized within source, then
  compared across paradigms. **At k = 2: the effect vector + an agreement statistic, with no pooling,
  no τ², no I².** Random-effects pooling with I² only at k ≥ 3.
- **Confirmatory axis pairs:** the 6–8 pre-registered axis pairs (≥2 non-gender×non-gender) that
  constitute the entire FDR family. Everything outside is exploratory and carries no significance claim.
- **Powered cell:** cell meeting the pre-registered minimum N; only powered cells are load-bearing.
- **Triangulation:** ≥2 independent scorers per harm type with reported agreement.
- **Scorer identity drift:** a scorer's own interaction coefficient over identity descriptors, measured
  on matched neutral texts. Subtracted from, or reported beside, the model's interaction (§9).
- **Manifest:** JSON emitted per run capturing config hash, git SHA + dirty flag, versions, seed,
  platform, torch/BLAS/thread count, checksums — enough to reconstruct the run.
