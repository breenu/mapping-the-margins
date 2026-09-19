# Power and precision in intersectional auditing

*Methods section, paper-ready. Every number here is emitted by
`scripts/run_power_analysis.py` or `scripts/measure_rho.py` and registered in
`results/paper_numbers.json`; the manuscript cites them by key and never retypes a digit.*

---

## 1. Why this section exists

The project's contribution is an instrument and its rigor, not the finding that intersectional
bias exists. This section is the part of that rigor which turned out to have teeth. Setting out to
pre-register a minimum detectable effect, we found that the achievable precision of a templated
generative intersectional audit is **bounded by a quantity almost no published audit reports**: the
number of distinct prompts per identity cell. Sampling more generations from the same prompts does
not relax that bound, and past a modest number of samples it buys essentially nothing.

We state this as a result rather than a limitation. If it holds beyond our design, a substantial
part of the generative bias-audit literature is reporting effects it was not powered to detect.

## 2. The estimand and its variance

Emergent intersectional bias is the interaction term of `harm ~ A + B + A:B` fitted over an axis
pair's order-2 cells, with sum-to-zero coding (`docs/PROJECT_CONTEXT.md` §10). Standard errors are
cluster-robust by **item**, where an item is one rendered prompt, because the sampled generations
of a prompt are repeated observations of that prompt rather than independent draws from the cell.

Write `m` for the prompts per cell and `k` for the sampled generations per prompt. With harm scores
standardized to unit variance and an intraclass correlation `rho` (the share of variance between
prompts within a cell), the variance of a cell mean is

```
Var(cell mean) = rho / m  +  (1 - rho) / (m * k)
```

Two consequences follow, and the second is the finding.

1. **Total renderings per cell is `m * k`.** Compute cost depends only on the product.
2. **Precision does not.** As `k` grows the second term vanishes and the variance **floors at
   `rho / m`**. Prompts per cell, not renderings per cell, set the achievable precision. A design
   can be arbitrarily expensive and still not reach a target effect.

For an interaction coefficient in an `I x J` table the cell-mean variance is scaled by
`(I-1)(J-1) / (I*J)`; the floor is unaffected.

## 3. Measuring rho

`rho` cannot be assumed and, in this literature, is almost never reported. It also cannot be
recovered from a conventional audit run: a design that generates each prompt once contains **no
within-prompt variance at all**, so `rho` is not identified from it. We therefore measured it.

**Design.** The `socioeconomic_class x race_ethnicity` pair (4 x 4 = 16 order-2 cells), all 9
template frames per cell, 12 sampled generations per prompt at distinct seeds: 1,728 generations
from `meta-llama/Llama-3.1-8B-Instruct` at 4-bit NF4, pinned revision, under the sampled decoding
profile the main audit uses (temperature 0.7, top_p 0.9). Scored through the full P6 roster, two
independent scorers per harm type.

**Estimator.** A nested one-way random-effects ANOVA with cells as a **fixed** factor and prompts
random within them, so between-cell variation — the signal the model fits — never enters `rho`.
Exact F-based intervals, because `rho` is bounded and skewed near zero, which is where the design
boundary sits. Implemented in `margins.stats.icc`, tested against injected variance splits.

**Result.**

| harm type | triangulated rho | 95% CI |
|---|---:|---|
| regard | 0.35 | 0.29 to 0.43 |
| toxicity | 0.18 | 0.13 to 0.23 |

**Caveat, stated prominently.** This is measured on **one axis pair**, the largest and most
demanding table in our lattice. It is the best available estimate and we do not assume it is
uniform across pairs. Nothing in the analysis treats it as exact: the design is chosen against the
**upper confidence bound**, and the sensitivity of the recommendation to `rho` is reported below
rather than hidden.

## 4. The ceiling

Because precision floors at `rho / m`, there is a power ceiling per prompt-count that no sampling
budget can exceed. We measured it by simulation rather than deriving it: synthetic data with the
real clustered structure, pushed through the **real** estimator and the **real** Benjamini-Hochberg
correction over the pre-registered 168-term family, with the seeds-to-infinity limit simulated
directly as one observation per prompt.

**Power ceiling for a 0.3 SD single-cell interaction** (alpha 0.05, BH over 168 tests):

| rho | 9 prompts/cell | 12 | 18 | 27 |
|---|---:|---:|---:|---:|
| 0.05 | 0.94 | 0.99 | 1.00 | 1.00 |
| 0.10 | 0.57 | 0.74 | 0.94 | 1.00 |
| 0.20 | 0.21 | 0.32 | 0.53 | 0.83 |
| 0.30 | 0.12 | 0.16 | 0.33 | 0.58 |
| 0.50 | 0.04 | 0.04 | 0.13 | 0.28 |

At the measured `rho` for regard (0.35), a 9-prompt design detecting a 0.3 SD interaction has a
ceiling near 0.12. **No number of generations reaches 80% power.** We verified the convergence
directly: at 9 prompts per cell and rho 0.4, simulated power at 20 and at 100 generations per
prompt differs by less than 0.05.

This is what forced the re-target described in §5.

## 5. The re-targeted design

Given the measured `rho`, 0.3 SD is not reachable at accessible compute. We re-targeted the
confirmatory minimum detectable effect to **0.5 SD**, a moderate interaction, and re-ran the same
simulation.

**Power ceiling for a 0.5 SD single-cell interaction:**

| rho | 9 prompts/cell | 18 | 27 | 36 |
|---|---:|---:|---:|---:|
| 0.20 (toxicity) | 0.72 | 1.00 | 1.00 | 1.00 |
| 0.23 (toxicity, upper) | 0.71 | 0.99 | 1.00 | 1.00 |
| 0.35 (regard) | 0.39 | 0.82 | 0.98 | 1.00 |
| 0.43 (regard, upper) | 0.30 | 0.70 | 0.96 | 0.98 |

The shipped 9-frame design remains unusable for regard even at 0.5 SD. At **27 prompts per cell**
the ceiling clears 0.95 at every measured value including the upper bounds, which is why the
design moves to 27 prompts per cell rather than to more sampling.

**Minimum renderings per cell**, simulated at the measured rho. Blank means the ceiling rules it
out at any sampling budget:

| rho | 9 prompts | 18 prompts | 27 prompts | 36 prompts |
|---|---|---|---|---|
| 0.20 (toxicity) | unreachable | 90/cell, 79 GPU-h | **54/cell, 47 GPU-h** | 72/cell, 63 GPU-h |
| 0.23 (toxicity, upper) | unreachable | 90/cell, 79 GPU-h | **54/cell, 47 GPU-h** | 72/cell, 63 GPU-h |
| 0.35 (regard) | unreachable | 216/cell, 190 GPU-h | **81/cell, 71 GPU-h** | 72/cell, 63 GPU-h |
| 0.43 (regard, upper) | unreachable | unreachable | **108/cell, 95 GPU-h** | 72/cell, 63 GPU-h |

The binding case is regard at its upper bound. **The adopted design is 27 prompts per cell x 4
sampled seeds = 108 renderings per cell, about 95 GPU-hours** across both confirmatory models.
`min_cell_n` is set to the same number, because a rendering budget and the minimum N it must clear
are one decision.

Two things in that table are worth reading carefully. First, operating just above a ceiling is
ruinous: regard at 18 prompts has a ceiling of 0.82, barely clearing the target, and therefore
needs 12 seeds and 190 GPU-hours, where 27 prompts needs 3 seeds and 71. Second, **spending budget
on prompts is cheaper than spending it on samples** at every row of the table. The design change
that made 0.5 SD affordable was not more compute; it was the same compute allocated differently.

Reaching 27 prompts per cell required growing the prompt set from 9 frames to 13 carrier-using
frames plus a second axis-neutral carrier, with the carrier-free frame emitted once
(`open_gen_v1.2`). That is a real cost in human review time, and it is the cost the measurement
converted into a decision.

**Effects below 0.5 SD are reported descriptively**, with cluster-bootstrap confidence intervals
and no significance claim, and are labelled as such in every table. We do not discard small
observed interactions; we decline to make inferential claims about them.

## 6. What this implies for the field

**This is the headline result of the section, not a caveat to it.**

A conventional generative audit uses a small number of templates — often one to five — and either
greedy decoding or a handful of samples per prompt. We simulated exactly that design at the `rho`
we measured, against a **moderate** 0.5 SD interaction, through the real estimator and the real
correction. Power at 9 prompts per cell:

| renderings per cell | rho 0.20 | rho 0.23 | rho 0.35 | rho 0.43 |
|---|---:|---:|---:|---:|
| 27 (9 prompts x 3 samples) | 0.34 | 0.27 | 0.22 | 0.18 |
| 45 (9 prompts x 5 samples) | 0.49 | 0.50 | 0.30 | 0.27 |

**Between 18% and 50% power on a moderate effect.** Not on a subtle one: 0.5 SD is a large
departure from additivity. A design in that range fails to detect a real moderate interaction more
often than it detects one, and — because nothing in the output says so — reports the failure as an
absence of bias.

Three consequences follow, and we state them as claims about the field rather than about our run.

1. **Null results from conventional intersectional audits are close to uninformative.** At this
   power, "no significant interaction" is the expected outcome whether or not an interaction of
   moderate size exists.
2. **Reported significant interactions from such designs are at elevated risk of being false
   positives**, because low power inflates the proportion of discoveries that are noise for any
   given plausible prior on real effects.
3. **The binding resource is not compute.** Our own conventional-shaped design at 45 renderings
   per cell costs 40 GPU-hours and reaches 27% power at the measured rho for regard; reallocating
   a similar budget across 27 prompts reaches the target. Audits are underpowered because of how
   they spend, not only how much.

A single-template design is a special case worth naming: a cell with one prompt has no
between-prompt variance, so it cannot estimate a cluster-robust standard error at all. Such a
design is not merely underpowered, it has no valid uncertainty estimate for the quantity it
reports.

The practical recommendations that follow are cheap to adopt:

1. **Report prompts per cell, separately from total generations.** They are not interchangeable and
   only the first bounds precision.
2. **Report or estimate `rho`.** It costs one small sampled run and it determines whether a design
   can see the effect it is looking for.
3. **Cluster by prompt.** An audit that treats repeated samples of one prompt as independent
   observations will report standard errors that are too small by roughly the design effect.
4. **Spend budget on prompts before samples.** Past a handful of samples per prompt, additional
   generations buy almost nothing.

## 7. Honest scope of these claims

- `rho` was measured on **one axis pair**, one model, one benchmark, at one decoding profile. It is
  an estimate with a stated interval, not a constant of nature.
- The power figures are simulation-based, on synthetic data with the real clustered structure, run
  through the real estimator and the real correction. They inherit the estimator's assumptions:
  identity link, OLS on standardized scores, sum-to-zero coding, clustering by item only.
- The family-companion p-values in the simulation are drawn as independent uniforms. The real
  family's true nulls are correlated through the shared prompt set, which moves the variance of the
  BH threshold rather than its mean.
- The compute figures use per-generation timings measured on one GPU (Tesla T4, NF4) and scale
  linearly. They are budget arithmetic, not a benchmark.
- **We have not yet run the confirmatory audit under this design.** This section reports what the
  design can detect, not what it found.
