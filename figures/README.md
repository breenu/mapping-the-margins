# `figures/` — generated, never hand-edited

Regenerate everything with one command:

```bash
uv run python scripts/make_figures.py
```

Vector only, PDF **and** SVG. `figures/*.png` is gitignored and
`margins.viz.style.save_figure` refuses a raster format outright, so a rasterised headline figure
cannot enter the repo.

## What each figure argues

| file | claim | source |
|---|---|---|
| `fig1_emergence` | **RQ2.** Emergence is the *interaction* term of `harm ~ A + B + A:B`, not a high harm level at a cell. Free contrasts of the confirmatory pairs with cluster-bootstrap intervals, FDR markers, and the γ table of the pair the design was powered against. | result tables |
| `fig2_scorer_drift` | **Construct validity.** Each scorer's own interaction over identity descriptors on known-neutral text, beside the model's. Nothing subtracted. This is the evidence for every descriptive-only downgrade. | result tables |
| `fig3_field_underpowered` | **The field result.** At the measured ρ, a conventional nine-prompt design reaches 18–50% power on a *moderate* 0.5 SD interaction. | `results/paper_numbers.json` — **real, measured** |
| `fig4_cross_paradigm` | **RQ1, scoped.** Generation and extraction compared at the single-axis level, plus the structural finding that a sentence-pair benchmark cannot populate an order-2 cell. The extractive interaction slot carries a reason, never a zero. | tables + measured CrowS counts |
| `fig5_redteam_vs_static` | **RQ3, exploratory.** The two arms' interaction vectors and the cells red-teaming surfaces. Descriptive by pre-registered budget decision: no p, no q. | result tables |
| `fig6_sensitivity` | **RQ4.** The headline interaction across the temperature grid and seed groups. | result tables |
| `fig7_frame_breakdown` | **Precision.** Frames are item clusters, and the spread across them is what bounds the design's precision. `first_person` is flagged by the P4a phrase review. | result tables |

## Which figures show real data right now

`fig3_field_underpowered` only. Its numbers are the measured intraclass correlation and the power
simulation, read by key from `results/paper_numbers.json`.

**Every other figure currently draws synthetic fixture rows** from `results/fixtures/` and carries
a visible banner saying so. The confirmatory run is P11; `scripts/build_result_tables.py` then
writes the same table names into `results/tables/`, `scripts/make_figures.py` prefers that
directory automatically, and **no figure code changes**.

## The rules these figures are built under

* **No number is typed into a figure.** Every plotted value comes through
  `margins.report.source.NumberSource`, which keeps a ledger; `margins.viz.trace.audit_figure`
  then reads the numbers back off the rendered artists and fails if any is unaccounted for.
  `scripts/make_figures.py` runs that audit as it builds, so an untraceable figure is not written.
* **No star a refusal would have withheld.** Significance markers come from the real
  `ClaimGate`, which *raises* for a refused effect; `margins.viz.claims` additionally fails if its
  verdict disagrees with the one P9 recorded.
* **Never a level where the claim is about interaction.** `fig1` plots γ and its heatmap sums to
  zero along every row and column, which a table of levels never does.
* **Colour never carries meaning alone.** Every series also has a distinct marker, hatch or
  linestyle, because the palette's closest two colours are 0.011 apart in greyscale luminance.
