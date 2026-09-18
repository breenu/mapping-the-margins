# Mapping the Margins

A reproducible, statistically rigorous **framework** (Python package `margins`) for
**intersectional bias auditing of open-weight LLMs**.

The contribution is **the instrument and its rigor**, not the finding that intersectional bias
exists — that is settled prior art. What the field lacks, and what this project supplies, is a
reproducible instrument with native intersectional metrics, cross-paradigm comparability
(static benchmarks *and* automated red-teaming on one scale), a mandatory statistical protocol,
a generative operationalization of emergence, lattice coverage beyond gender-anchored
intersections, and all of it at **accessible compute** — no industrial GPU, and the reference
configuration reproduces on free-tier cloud (see [Compute](#compute)).

## Build status

| Phase | State |
|---|---|
| **P0** — environment, seeding, run manifest, CI | ✅ complete |
| **P1** — schemas + interfaces | ✅ complete |
| **P3** — identity lattice | ✅ complete |
| **P4a** — `TemplateRenderer` + the `open_gen` prompt set | ✅ complete |
| **P5** — inference backend + resumable run harness | ✅ complete |
| P6 — scoring, triangulation, drift control | in progress |
| P7 and later | not started |

Phases are built strictly in the order in [`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md)
§5; each is gated by §11.

## Install

`uv` is the dependency manager; Python 3.11 is pinned and provisioned by `uv`.

```bash
uv sync --extra dev          # Windows dev environment: core + dev, no GPU stack
uv sync --extra inference    # torch, transformers, accelerate (BOTH environments)
                             #   + bitsandbytes, datasets, evaluate, detoxify (Linux only)
uv sync --extra llamacpp     # pure-CPU / Apple fallback backend
```

Within the `inference` extra, `bitsandbytes` (and `datasets`/`evaluate`/`detoxify`) are Linux-gated
by marker while `torch`/`transformers`/`accelerate` are not, so the extra installs on Windows for
dev-scale work. `uv.lock` resolves for exactly two environments (`win_amd64`, `linux_x86_64`), which
is what keeps `bitsandbytes` from failing resolution on Windows — never unpin a dependency to work
around a resolution failure.

## Quality gates

All must pass before crossing a phase boundary:

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

## What P0 gives you

- **`margins.repro.set_global_seed`** — seeds `random`, legacy global `numpy`, `PYTHONHASHSEED`
  and (when present) `torch`, returns a frozen `SeedReport` recording *what was actually set*
  plus an explicit `numpy.random.Generator`. `torch` is optional and lazily imported; the module
  is fully functional without it.
- **`margins.repro.residual_nondeterminism_note`** — the exact determinism claim this project is
  entitled to make, carried into every manifest. A test asserts its qualifiers so it cannot be
  silently strengthened into an unconditional bit-identity promise.
- **`margins.config.ExperimentConfig`** — frozen, `extra="forbid"`, validated from plain YAML
  composed by Hydra (no structured configs, no hydra-zen).
- **`margins.config.write_run_manifest`** — writes `results/run_manifests/<run_id>.json` and
  returns it **re-read from disk**, so the object a caller holds cannot drift from the bytes a
  reproducer will read. `run_id` is derived from the config hash and a recorded timestamp, not
  randomly generated.

Demo both:

```bash
uv run python scripts/p0_demo.py
```

## What P1 gives you

- **`margins.schemas`** — the frozen, source-agnostic records that *are* the unified metric
  layer. Every stage depends on these and on nothing else of any other stage. Records are
  deeply immutable and hashable (`FrozenDict` freezes payloads all the way down, so a nested
  list cannot smuggle mutability into a "frozen" record), and `extra="forbid"` everywhere.
- **`margins.interfaces`** — one `Protocol` per pipeline stage, so sources, scorers, and
  backends are swappable and fakeable without inheriting from anything.
- **`tests/test_walking_skeleton.py`** — no-op fakes for every Protocol, composed end to end
  on fake data. It proves the contracts *compose* before any real logic exists; it computes
  nothing meaningful and asserts nothing about correctness.
- **`tests/test_schema_field_freeze.py`** — pins the exact field set of every shared record.
  Deliberately brittle: a field change is a contract change and must be a conscious edit.

Two invariants are enforced structurally rather than by review: only `EffectEstimate` may carry
significance fields (`CellStat` is descriptive and has no `p`/`q` at all), and no shared record
may carry a source-specific field — source quirks live only in `Item.payload`.

## What P3 gives you

- **`configs/lattice/confirmatory_v1.yaml`** — the lattice declaration, and **pre-registration
  material**: 5 axes, 18 descriptor values, and the named axis pairs. The 6 confirmatory pairs
  *are* the FDR family; 4 of them are non-gender × non-gender (gate P3 / novelty claim 5).
- **`margins.lattice.LatticeConfig`** — Hydra composes the plain YAML, Pydantic validates it, and
  three rules are structural rather than conventional: a value with no source does not load,
  a lattice with fewer than two non-gender × non-gender confirmatory pairs does not load, and the
  confirmatory family is capped so the BH threshold cannot be diluted into uselessness.
- **`margins.lattice.DeclaredLattice`** — 110 `IdentityCell`s (18 order-1, 92 order-2) generated
  for **declared axis pairs only**. That anti-cartesian rule is the point: crossing every axis
  with every other would multiply cells, divide the per-cell budget, and inflate the FDR family.
  `compose_descriptor` is a documented **construct choice**, not a neutral operation.
- **`margins.lattice.descriptors`** — HolisticBias v1.1 as a **pinned descriptor source, never a
  `BenchmarkLoader`** (`open_gen` is a generation path, not an ingestion one). Every value is
  verified against the pinned file at its exact `(axis, bucket, term, preference)` coordinates,
  so an invented descriptor is a failing check rather than a paper claim.
- **Exports** — `data/lattice/cell_inventory.csv` marks every cell powered vs descriptive
  *before* any run; `data/lattice/value_provenance.md` traces every value to its source. Both are
  byte-deterministic across Windows and Linux.

`min_cell_n` is a **pre-registered placeholder**; TODO(P9) derives the real threshold from the
power analysis (α=0.05, power=0.8). No power computation happens in P3.

```bash
uv run python scripts/fetch_holistic_bias.py    # once: retrieve the pinned descriptors
uv run python scripts/build_lattice.py --verify # build, verify every term, write both exports
```

## Compute

Development runs on a Windows laptop (core + dev extras, no `torch`). The 7–8B generative sweep and
red-team generation run on Linux with NF4 quantization, and never locally. Quantization is a
**fixed recorded constant**, not an experimental axis.

**The compute claim is two-part, and both halves matter.** A measured timing pilot (Kaggle, Tesla T4,
NF4) put the confirmatory sweep at ~14 s/prompt per model — roughly 3 weekly free-tier quotas. The
sweep therefore **runs on a university GPU** to meet the deadline, while **free-tier Kaggle remains
the documented reproduction path**: accessible compute is a property of the instrument, not a claim
about where we personally ran it. Because the two hosts differ, cross-hardware reproduction is
**metric-level within a reported tolerance, not bit-level**; the manifest records the actual GPU,
driver, `torch` version, thread count and BLAS so a reproducer can check that.

**CI has no GPU.** GitHub-hosted runners run lint, types, and tests on Linux and Windows; the
gated `inference-cpu` job only verifies that the inference extra installs. Model-level
determinism checks belong to P5.

## Reference

Architecture, record schemas, module interfaces, the full phase plan, the statistical protocol,
the emergence estimand, validation gates, timeline, and risks are in
[`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md). Always-on working memory (status,
commands, framing constraints, design invariants, guardrails, open decisions) is in
[`CLAUDE.md`](CLAUDE.md).

`data/` is gitignored and regenerable. Model weights and raw response dumps are never committed;
raw dumps ship as a Zenodo artifact at release.
