# Mapping the Margins

A reproducible, statistically rigorous **framework** (Python package `margins`) for
**intersectional bias auditing of open-weight LLMs**.

The contribution is **the instrument and its rigor**, not the finding that intersectional bias
exists — that is settled prior art. What the field lacks, and what this project supplies, is a
reproducible instrument with native intersectional metrics, cross-paradigm comparability
(static benchmarks *and* automated red-teaming on one scale), a mandatory statistical protocol,
a generative operationalization of emergence, lattice coverage beyond gender-anchored
intersections, and all of it at **accessible / free-tier compute** — no industrial GPU.

## Build status

| Phase | State |
|---|---|
| **P0** — environment, seeding, run manifest, CI | ✅ complete |
| **P1** — schemas + interfaces | ✅ complete |
| P2–P13 | not started |

Phases are built strictly in the order in [`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md)
§5; each is gated by §11.

## Install

`uv` is the dependency manager; Python 3.11 is pinned and provisioned by `uv`.

```bash
uv sync --extra dev          # Windows dev environment: core + dev, no GPU stack
uv sync --extra inference    # Linux / Kaggle only: transformers, torch, bitsandbytes, ...
uv sync --extra llamacpp     # pure-CPU / Apple fallback backend
```

The `inference` extra is Linux-gated by marker, and `uv.lock` resolves for exactly two
environments (`win_amd64`, `linux_x86_64`). That is what keeps `bitsandbytes` from failing
resolution on Windows — never unpin a dependency to work around a resolution failure.

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

## Compute

Development runs on a Windows laptop (core + dev extras, no `torch`). The 7–8B generative sweep
and red-team generation run on **free-tier Kaggle GPU** (Linux, NF4) and never locally.
Quantization is a **fixed recorded constant**, not an experimental axis.

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
