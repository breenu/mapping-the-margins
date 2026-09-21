# Methods — what each source measures, and what it does not

PROJECT_CONTEXT.md §9 requires a one-paragraph "what this measures / what it does not" note per
source, as part of the response to Blodgett et al. (2021). This is that file. It is written to be
read by someone deciding whether to believe a number, so every construct choice appears here even
where it also appears in a config comment.

**Numbers are not quoted here.** Every figure the manuscript uses resolves to a key in
`results/paper_numbers.json`; the counts below that come from a run are reproduced from the
committed artifacts and regenerable by the commands in `CLAUDE.md`.

---

## 1. `open_gen_templates` — the templated generation paradigm (P4a)

**What it measures.** The harm a scorer assigns to text a model generates when prompted about a
person described by one or two identity descriptors. 27 prompts per cell (13 carrier-using frames
× 2 carriers + 1 carrier-free frame), 4 sampled generations per prompt at distinct seeds.

**What it does not measure.** Anything about the model's internal representations, and anything
about text the model produces unprompted. The frames are instructions, and the pinned checkpoints
are instruct-tuned, so this measures instructed behaviour.

**Construct choices, all reversible and all recorded.**

- The carrier noun is `person`, because every obvious alternative is itself an identity axis.
- Descriptors compose in `compose_rank` order, which follows English adjective order. This is not
  neutral: a phrasing that is stilted for one pair and fluent for the others contributes to *that
  pair's* interaction, which is the estimand. See `data/prompts/phrase_audit.md`.
- One frame (`someone_who_is`) joins constituents with "and" rather than juxtaposition, because
  English predicative position rejects juxtaposed adjectives. That asserts two co-predicated
  properties where juxtaposition can read as one compound category, so **R1 reports that frame
  separately**.

**The instrument is two classifiers plus two lexicons, and it has its own interaction.** See §3.

---

## 2. `crows_pairs` — the extractive sentence-likelihood paradigm (P2)

CrowS-Pairs (Nangia et al., EMNLP 2020), 1,508 minimal sentence pairs, pinned at commit
`8aaac11c485473159ec9328a65253a5be9a479dc`, CC-BY-SA-4.0.

**What it measures.** Whether the model assigns higher whole-sequence log-likelihood to the more
stereotyping member of a minimal pair than to its minimally different counterpart. Higher =
more harm, by the sign convention `logprob_stereo − logprob_antistereo`.

**What it does not measure.** Harm in generated text. The two paradigms are **not two estimates of
one quantity** — that is precisely why §10 answers RQ1 with an effect vector plus an agreement
statistic and forbids pooling at k = 2. Reporting a pooled number would be false precision about
two things that were never the same thing.

### 2.1 The causal-model caveat — state this wherever a CrowS number appears

The published CrowS-Pairs metric is a **masked pseudo-log-likelihood**: it conditions on the
unmodified tokens and masks the modified ones. Llama and Mistral are **decoder-only**, and cannot
do that. We therefore use **whole-sequence autoregressive log-likelihood** instead.

This is a documented substitute, not an equivalent. It changes what is conditioned on, and it makes
the score sensitive to sequence length and token frequency in a way the masked metric partly
controls for. §3 below is how that sensitivity is measured rather than assumed away.

### 2.2 The sentences are scored RAW — no chat template

`configs/decoding/pll_v1.yaml` sets `prompt_format: raw`, and this is the one field in that file
that changes the number.

CrowS-Pairs log-likelihood is computed on the bare sentence in the literature, so scoring the bare
sentence keeps our figure comparable to published work. Wrapping it in an instruct model's chat
template would put the wrapper's tokens inside every sequence log-probability and make the
per-model template part of the measurement — the same sentence would score differently under Llama
and Mistral for reasons unrelated to bias.

**The cost, stated rather than hidden:** the two paradigms see the model through *different*
interfaces. The generation path wraps (its prompts are instructions); this path does not (its
sequences are declarative sentences, not requests). That asymmetry is a property of the paradigms,
not an inconsistency — but a reader comparing the two effect vectors should know it is there. The
choice is folded into `decoding_hash`, so every paired `Response` is self-describing about it.

### 2.3 The direction is resolved once, at the loader boundary

218 of the 1,508 upstream rows are marked `antistereo`, meaning **`sent_more` is the
*anti*-stereotypical sentence**. Reading that column as "the stereotyping one" would invert the
sign of every CrowS effect while producing entirely plausible output. The swap is therefore
resolved in `CrowsRow.sent_stereo` / `.sent_antistereo`, nothing downstream reads the raw columns
for direction, and a named test asserts it.

### 2.4 Screening: what was excluded, and why

Deterministic, pure, and logged to `data/benchmarks/crows_pairs/excluded_items.csv` with one
machine-readable reason per row. Of 1,508 rows, **1,352 kept and 156 excluded**:

| reason | items |
|---|---|
| `not_minimal_pair` (more than one contiguous differing region) | 153 |
| `swap_too_wide` (one region, over the 4-token bound) | 2 |
| `duplicate_pair` | 1 |

**A grammatical-concord allowance is applied, declared, and counted.** Swapping `woman` for `man`
*forces* `she` → `he`; English gives the writer no choice, so counting the pronoun as an
independent second differing region would label grammar as a construct flaw. A region made only of
closed-class pronouns, possessives and reflexives therefore does not count toward the minimality
bound. The list is in `configs/benchmark/crows_pairs_v1.yaml` where it can be criticised, it
excludes honorifics (`sir`/`ma'am` are a lexical choice, not concord), and its effect is reported:
**68 rows were admitted by it**, so a stricter reading of the same data would record 221
`not_minimal_pair` exclusions rather than 153.

### 2.5 Lattice coverage — and why the extractive paradigm cannot estimate an interaction

**This is a measured finding, not a limitation discovered late.** Of the 1,352 kept items:

| outcome | items |
|---|---|
| order-1 cell | 198 |
| order-2 cell | **3** |
| no cell — descriptor is not a lattice value | 761 |
| no cell — category has no lattice axis | 390 |

Attestation is **literal**: an `(axis, value)` is attested only when a lattice value's surface form
occurs in the sentence. There is no alias table. CrowS row 1 contrasts "The poor" with "The rich",
and since neither is a lattice value (the lattice has `working-class` and `wealthy`, resolved from
HolisticBias at a pinned commit) that row attests nothing on the class axis. Mapping it would be a
construct claim *we* invented, which is exactly what the pinned descriptor file and
`build_lattice.py --verify` exist to prevent. The cost is a low yield; the benefit is that no cell
assignment in this pipeline was authored.

The consequence is structural and worth stating as a result:

> **Each CrowS-Pairs item is annotated with ONE attribute.** An order-2 cell therefore requires a
> *second* identity term present unchanged in *both* sentences — shared context rather than
> contrast — which is rare. The emergence estimator fits order-2 cells only and requires every cell
> of a pair's grid to be non-empty, so with 3 order-2 items across 2 cells, **every confirmatory
> axis pair comes back `estimable=False`** with an explicit "empty cell" reason.

So the extractive paradigm contributes descriptive `CellStat` rows and a not-estimable report, and
P9 excuses its family members rather than correcting over a family that shrank silently. This is
the extractive analogue of the P9 field result: it says something real about what sentence-pair
benchmarks can and cannot support, and it is why intersectional auditing needs a generation path.

**It is reported, not hidden**, and it must not be presented as a null result about the model.

### 2.6 Two reductions, and what triangulating them buys

Framing constraint 6 forbids any harm type depending on a single scorer. A `pair_pll` measurement
has **no external instrument** — the number is the model's own log-probability difference, so there
is no classifier to be wrong. What can still be wrong is the reduction, and the two used here fail
differently:

- **`pll_total`** sums, inheriting raw PLL's documented length and token-frequency confound.
- **`pll_length_normalized`** divides each side by its own length, and over-corrects when the two
  members differ in length for unrelated reasons.

Their **disagreement localises a length confound**, which is the failure mode the CrowS-Pairs
literature warns about — unlike two toxicity classifiers, they cannot be wrong together in the same
direction. On the 40-response dev smoke run their Krippendorff α was 0.694, i.e. they genuinely do
not always agree.

**Length is measured in whitespace words, not tokens, and that is a limitation.** A per-token mean
would be the textbook normalisation, but per-sequence token counts are not on the `Response`:
`logprob_stereo`/`logprob_antistereo` are totals and `token_logprobs` has no recoverable boundary
between the two sequences. Putting them there would mean a second change to a frozen shared record
inside one phase for one scorer's convenience. Word count is a model-independent proxy, it needs no
tokenizer — so the scorer runs on the torch-free core environment — and it targets the same
confound.

### 2.7 There is no identity-drift control for this paradigm

The scorer identity-drift control (§3) scores known-neutral **text** through the same **instrument**
and reports the instrument's own interaction. A paired log-probability difference has neither: there
is no neutral analogue of "how much more likely is this sentence than its minimally different
counterpart", and no classifier reading identity terms for the control to catch.

The run prints this rather than emitting an empty control series, because an absent control that
says nothing reads like a control that returned nothing.

### 2.8 Clustering degenerates here, and the language must not overclaim

A CrowS item is one prompt with one deterministic log-likelihood measurement, so **clusters are
singletons**: `n_items == n_obs`, and the cluster-robust sandwich collapses to a
heteroskedasticity-robust standard error. The cluster bootstrap becomes an ordinary case bootstrap.

That is *correct* — there is nothing to cluster — but it is not what the estimator's
`1 + (m−1)·rho` design-effect rationale describes, and the pre-registration's "standard errors are
cluster-robust by item" must not be read as claiming a cluster correction did work here that it did
not.

---

## 3. The scorer identity-drift control (generation paradigm only)

Detoxify-style classifiers are documented to score text higher purely for mentioning identity
terms. Our estimand is "harm varies non-additively across identity cells", so a scorer whose own
response to identity mentions is non-additive produces a non-zero interaction **from text that is
identical across cells** — a fully fabricated emergence result.

"≥2 scorers with reported agreement" does not address this: two identity-sensitive classifiers
agree with each other and are wrong together. The control scores matched known-neutral texts
through every scorer, fits the same `A + B + A:B` model on those, and reports each scorer's own
interaction **beside** the model's, never subtracted. A harm type whose scorer interaction reaches
half the model's is **descriptive-only**, enforced in the claim layer rather than advised.

---

## 4. Go/no-go and venue decision

Recorded here when the Sep 28–29 gate is taken (§13). The five criteria and the fallback venue are
in `docs/PROJECT_CONTEXT.md`.
