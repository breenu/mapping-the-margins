# CLAUDE.md — Mapping the Margins

Lean, always-on project memory for Claude Code. Deep reference (architecture, schemas, phase
plan, experiments, validation gates) lives in **`docs/PROJECT_CONTEXT.md`** — read it before
implementing a new phase.

---

## Status & next action

**P0 ✅ · P1 ✅ · P3 ✅ · P4a ✅ · P5 ✅ complete.** P0: environment, `margins.repro`,
`margins.config` + run manifest, CI. P1: `margins.schemas` (frozen records) + `margins.interfaces`
(Protocols), walking skeleton, and the schema field-freeze test. P3: `margins.lattice` —
`LatticeConfig` (`configs/lattice/confirmatory_v1.yaml`), `DeclaredLattice`, HolisticBias v1.1 as a
**pinned descriptor source** (not a loader), and the two exports. P4a: `margins.prompts` —
`TemplateRenderer`, 9 versioned Jinja frames, the leakage guard, and the phrase review (all 4
findings resolved; lattice bumped to **v1.1**). P5: `margins.inference` — `DecodingConfig` +
`decoding_hash`, the pinned `ModelSpec`s (decision #2 closed), `HFBackend`, and the resumable
`RunHarness`. All four gates green on `win_amd64` and in CI on both locked environments.

**NEXT TASK: P6 — scoring, triangulation, and the scorer identity-drift control.** Not started.

**⚠ THE KAGGLE TIMING PILOT IS OUTSTANDING AND IS THE HUMAN'S TO RUN.** `scripts/kaggle_timing_pilot.py`
/ `notebooks/kaggle_timing_pilot.ipynb` are built, self-contained and tested, but **no timing number
exists yet** — Claude Code cannot reach Kaggle and must never fabricate one. Its output is the input
to **hard go/no-go criterion 4** (Sep 28–29) and to the E1 grid decision (greedy vs sampled), both of
which are due before P6's results matter. Run it, bring back `pilot_results.json`, and record the
numbers here.

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
- **Open decision #7 is still open, by construction.** `run()` **refuses** `pair_pll` and names it;
  `HFBackend.score_pair()` builds decision #6's shape (one `Response`, both sequence log-probs) but
  takes both strings from its caller rather than encoding a representation. Settle #7 at P2/P4b.
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

### The Kaggle timing pilot — BUILT, NOT RUN (the human's to execute)

`scripts/kaggle_timing_pilot.py` + the generated `notebooks/kaggle_timing_pilot.ipynb`. **No timing
number exists yet and none may be invented.**

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
- **What it feeds:** hard go/no-go criterion 4 (Sep 28–29) and the **E1 grid decision** — greedy
  gives only 9 renderings/cell against `min_cell_n = 60`, so the sampled arm is what the power
  bookkeeping actually needs, at roughly 11× the cost. Red-teaming (P8) is **not** in the table.

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
- **Budget arithmetic:** renderings/cell = |templates| × |samples|. 9 frames no longer divides
  P3's 96 evenly — 11 samples ⇒ 99/cell (**10,890 renderings**), 12 ⇒ 108. Both clear
  `min_cell_n = 60`, and both that and the 96 are P9 placeholders, so nothing is load-bearing yet.
  Sample count is a P5 decision.
- ~~**`decoding_key = "greedy_v1"` is a NAME only.**~~ **Resolved at P5:**
  `configs/decoding/greedy_v1.yaml` + `DecodingConfig.decoding_hash`. Editing that file changes the
  prompt-set config's sha256 but **not** `prompt_id` (which digests the rendered text, not the
  profile) — it changes every `response_id` instead, which is the right blast radius.
- **Manifest now records the lattice** (the provenance hole P3 flagged, closed).
  `ExperimentConfig.lattice_name` is the only field added; `input_checksums` carries
  `lattice_config`, `prompt_set_config` and one `template:<id>` per frame. The lattice digest
  matches what `scripts/build_lattice.py` prints. *Named `lattice_name`, not `lattice`, because
  `configs/lattice/` is a Hydra config GROUP and a colliding key cannot be overridden at all.*
- **Open decision #7 is untouched.** `open_gen` never needs a second sequence; `PromptSetConfig`
  types `task_type` as `Literal["open_gen"]` and the renderer rejects non-`open_gen` items, so the
  paired path fails to load rather than being half-built. TODO(P4b) markers are in
  `margins/prompts/render.py`.

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

> P0 ✅ → P1 ✅ → P3 ✅ → P4a ✅ → **P5** → P6 → P7 → *then* P2 (CrowS-Pairs) →
> **P4b** (ExtractiveRenderer) → P8 → P9 → P10 → P11 → P12 → P13

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
# The Kaggle timing pilot is run BY THE HUMAN on Kaggle — see notebooks/kaggle_timing_pilot.ipynb.

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

1. ~~Hardware branch~~ — **DECIDED: Windows laptop for dev (`win_amd64`, core+dev only), Linux /
   free-tier Kaggle GPU for inference (`linux_x86_64`, NF4). The 7–8B sweep never runs locally.** §7.
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
3. **Emergence link/scale + null model** (before P7/P9): identity vs logit; the additive null to test
   against. Must be pre-registered (tag due end of W4).
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
7. **Paired-task prompt representation** (before P4b/P6 — *not* hit by the generation MVP): decision
   6 settled the **scoring shape only**. `InferenceBackend.run()` sees only `Prompt`s, and
   `Prompt.rendered_text` is a single string, so it cannot carry the two sequences a `pair_pll`
   response needs. Options: a second rendered-text field on `Prompt`, a canonical joined encoding, or
   a paired-prompt variant. Nothing is encoded yet — do not let this slip past the CrowS-Pairs loader.

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
