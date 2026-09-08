# CLAUDE.md — Mapping the Margins

Lean, always-on project memory for Claude Code. Deep reference (architecture, schemas, phase
plan, experiments, validation gates) lives in **`docs/PROJECT_CONTEXT.md`** — read it before
implementing a new phase.

---

## Status & next action

**P0 ✅ · P1 ✅ · P3 ✅ · P4a ✅ complete.** P0: environment, `margins.repro`, `margins.config` +
run manifest, CI. P1: `margins.schemas` (frozen records) + `margins.interfaces` (Protocols),
walking skeleton, and the schema field-freeze test. P3: `margins.lattice` — `LatticeConfig`
(`configs/lattice/confirmatory_v1.yaml`), `DeclaredLattice`, HolisticBias v1.1 as a **pinned
descriptor source** (not a loader), and the two exports. P4a: `margins.prompts` —
`TemplateRenderer`, 9 versioned Jinja frames, the leakage guard, and the phrase review (all 4
findings resolved; lattice bumped to **v1.1**). 279 tests; all four gates green on `win_amd64` and
in CI on both locked environments.

**NEXT TASK: P5 — model inference.** Not yet started. **Blocked on open decision #2** (exact 7–8B
repo + pinned revision hash + license, plus the ~1B dev model) — settle it before writing the
backend. Also due in W2: the **Kaggle timing pilot**, which converts the 880-prompt set into a
measured GPU-h budget.

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
- **`decoding_key = "greedy_v1"` is a NAME only.** TODO(P5): the resolved decoding config and
  `decoding_hash` land with the backend.
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
uv sync --extra inference    # transformers, torch, bitsandbytes, datasets, evaluate, detoxify (Linux/Kaggle)
uv sync --extra llamacpp     # pure-CPU / Apple fallback backend
uv sync --extra dev          # pytest, ruff, mypy, pre-commit

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
2. **Model checkpoints** (before P5): exact 7–8B repo + **pinned revision hash** + license, plus the
   ~1B dev model; verify current availability at that time.
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
