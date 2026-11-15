# Iteration results: world model + rule vs. city policy

**Candidate:** `models/risk/cands/c_wm_oos_blend.py`, frozen at commit `485616b` (branch `c-world-model`).

**Recipe**
- 0.5 × two-head LightGBM on **out-of-time world-model parcel features**:
  - the SF World Model's predicted next-year sale rate and block-face Ellis/OMI rate;
  - plus units and block-face share, for scale.
- \+ 0.5 × the calibrated rule of thumb. The blend weight is copied, not tuned.

## Test (T = 2018, outcomes 2019–2024; one logged `--final` call)

Households protected, against a budget that is a multiple of the city's actual spend:

| budget | 1× | 2× | 3× | 5× |
|---|---|---|---|---|
| **World model + rule** | **10.5** | **23.3** | **24.8** | **27.0** |
| City policy | 1.9 | 3.1 | 13.7 | 22.9 |
| Random | 1.8 | 3.2 | 5.1 | 7.0 |

- **2.83× city policy** (geometric mean over budgets). It beats the city at every budget.
- **4.7× random.**
- **Compared with run 1** (`reports/backtest/PREREG.md`), the model went from 0.37 households at 1× to 10.5.
- **Bootstrap 95% CIs of (ours − city):**
  - 1×: [0.3, 18.3]
  - 2×: [2.8, 39.0]
  - 3×: [−11.7, 34.8]
  - 5×: [−19.1, 32.8]

  So the gain is statistically clear at 1× and 2×.

The harness wrote a blank `commit` field in `test_log.csv` because the Savio code directory is an rsync copy without `.git`. The frozen commit is `485616b`.

## Dev (T = 2016, outcomes 2017–2019), mean of LightGBM seeds 0–4

| candidate | vs city | beats city at all budgets | seed range |
|---|---|---|---|
| **`c_wm_oos_blend`** (world model + rule) | **2.78×** | 5/5 seeds | 2.71–2.94 |
| `c_wm_oos_only` (world model alone) | 2.40× | 5/5 seeds | 2.33–2.58 |

Households protected for `c_wm_oos_blend` (seed mean):

| budget | 1× | 2× | 3× | 5× |
|---|---|---|---|---|
| World model + rule | 11.5 | 15.0 | 22.0 | 30.2 |
| City policy | 2.2 | 6.6 | 7.1 | 14.2 |
| Random | 2.9 | 6.3 | 8.4 | 14.0 |

Dev and test agree: 2.78× and 2.83× city. Results are stable across seeds.

## The world model

**Architecture**
- A transformer with 5.08M parameters over H3 res-9 cells. There is one sequence per (cell, year) built from:
  - parcels, sales, permits, block-face eviction notices and buyouts.
- **Loss** = (1−λ)·masked-token + λ·next-year. The next-year part has two heads:
  - a cell-level head predicting next year's per-type counts and feature means;
  - a **parcel-level head** predicting each parcel's next-year sale, buyout tenants and block-face Ellis/OMI notices (Poisson).

**Training and λ choice**
- λ was swept over 6 values on cutoff 2016 and picked by parcel-head skill on held-out (2015→2016) pairs. λ = 0.1 was selected.
- That rule uses only data dated ≤ 2016.
- The older selection rule scored year-2016 probes whose outcomes run 2017–2019, so it would have leaked into the dev window. It was not used.

**Probes:** frozen embeddings beat hand-made cell features on year-2016 validation probes at every λ. The mean R²/PR-AUC lift is +0.10 to +0.13, and every CI is above 0.

**Out-of-time features (the key fix)**
- **The problem:** a model with cutoff T is trained on the same t→t+1 outcomes that LightGBM later learns from. Its features are therefore in-sample on training rows, and LightGBM over-trusts them.
- **The fix:** each year t's features come from a separate world model with cutoff t, so no row's feature ever saw its own outcome. That is 14 rolling-origin models: cutoffs 2011–2016 for dev and 2011–2018 for test.
- **Code:** `models/world_model/oos.py`, output file `parcel_oos_embeddings.parquet`.

**Compute:** Modal A10G, about 2 minutes per run. The whole pipeline, including the rolling-origin runs, cost a few dollars.

## Ablation (dev, seed 0 unless noted)

| variant | vs city |
|---|---|
| world-model features, in-sample (PCA + parcel head), alone | 2.56× |
| world-model features, out-of-time, alone | 2.40× (5-seed mean) |
| **out-of-time world model + calibrated rule** | **2.78×** (5-seed mean) |

- Adding the calibrated rule to the world-model model adds about 0.4× city.
- Mixing world-model features into one LightGBM together with ~20 hand-made features hurt in every version tried. The likely cause is too few buyout-era positives for that many correlated features.
- Combining the **scores** of separate models works better than combining their inputs.

## Caveats
- **Dev is small:** 23 city picks, and the bootstrap CIs at the higher budgets include 0.
- **Weak against the rule of thumb at the edges.** The rule-of-thumb heuristic (sold in last 2y × 10 + nearby Ellis/OMI notices) scores 0.93× of ours on the test set by our metric. We are ahead at 2× and 3× and behind at 1× and 5×.
- **Buyout data starts in 2015-03.** The parcel head's buyout output was not used out-of-time, because rolling models with cutoff ≤ 2015 have no buyout targets.
- **Standardization constants.** Rolling-origin models read the cutoff-T token partition filtered to years ≤ t, so their feature standardization uses data up to T. All of it is dated ≤ T, which is legal for the harness, but it is not a fully native cutoff-t build.
- **Other candidates.** Non-world-model candidates from the LightGBM track are reported separately in `reports/risk/b/README.md`.
- **Test calls:** 1 of 3 used, on this candidate only.
