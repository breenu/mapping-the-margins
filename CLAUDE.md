# CLAUDE.md — Mapping the Margins

Lean, always-on project memory for Claude Code. Deep reference (architecture, schemas, phase
plan, experiments, validation gates) lives in **`docs/PROJECT_CONTEXT.md`** — read it before
implementing a new phase.

---

## Status & next action

**P0 ✅ complete.** Environment, `margins.repro`, `margins.config` + run manifest, tests, CI.
All four quality gates green on `win_amd64` (Python 3.11.16, uv 0.12.10, 117 packages locked).

**NEXT TASK: P1 — schemas + interfaces.** Build strictly in the order in
`docs/PROJECT_CONTEXT.md` §5; do not skip ahead. Update this section as phases complete.

**Hard deadline: FAccT abstract 27 Oct 2026 (mandatory, non-placeholder), paper 3 Nov 2026 — 8 weeks
from ~8 Sep.** *(Verify the deadline year against the official CFP; these dates may belong to FAccT
2027. See the note atop `PROJECT_CONTEXT.md`.)* **Hard go/no-go on Sep 28–29** — five criteria in §13;
if any fails the fallback is ACL Rolling Review or a non-archival workshop, decided at that gate.

**Outstanding env item:** no git remote is configured yet — CI will not actually run until one is
added. Not blocking local work. The repo lives at `C:\dev\mapping-the-margins` (outside OneDrive)
and has **no commits yet**, so manifests currently record `git_sha: null` with `git_dirty: true`.

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
