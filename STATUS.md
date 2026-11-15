# Project status — read this first (2026-11-15)

This is the one doc for whoever picks the project up. The pitch and plan are in `PRD.md`, and data schemas are in `docs/CONTRACTS.md`. Older per-stream handoffs (`docs/STREAM_A_STATUS.md`, `docs/agents/*`, `sim/patient/HANDOFF.md`) are history; this file supersedes them.

## Headline numbers

| PRD number | Status | Value | Source |
|---|---|---|---|
| **N× tenants protected vs. city picks** (backtest 2019–24) | **done** (test, 1 logged call) | **2.83× city policy**, beats it at every budget (1×/2×/3×/5×). 4.7× random. | `reports/risk/ITERATE_RESULTS.md` |
| World-model probe lift over hand-made features | **done, locked test** (`eval_final --final`, run once) | mean **+0.10** over 7 cell targets, 95% CI [0.001, 0.19] (validation was +0.12) | `reports/wm/probes_test.csv`, `probes_test_summary.json` |
| UMAP→RGB map of SF | done | `reports/wm/umap_rgb_map.png` / `.html` (2018), `_2016` (dev). Rendered with PCA; umap-learn wasn't in the Modal image. | |
| Risk-model PR-AUC / precision@k | **done, test** (frozen `c_wm_oos_blend`) | buyout at building: PR-AUC **0.094** vs city pattern 0.070, base 0.061. Any displacement: 0.176 vs 0.165, base 0.148; prec@100 0.29 vs 0.23. Top 10% of buildings hold **21%** of displaced households (city pattern 16%) | `web/public/data/headline.json` (Savio `$S/repo-final`) |
| Patient sim: Δ minutes to care, % losing 30-min access | **not done** (r5 run in flight, see below) | — | |

### The candidate to show: `c_wm_oos_blend`
- **What it is:** 0.5 × two-head LightGBM on out-of-time world-model parcel features + 0.5 × calibrated rule of thumb.
- **Frozen at** commit `485616b`.

| households protected | 1× | 2× | 3× | 5× |
|---|---|---|---|---|
| **test** (T=2018, outcomes 2019–24): world model + rule | **10.5** | **23.3** | **24.8** | **27.0** |
| test: city policy | 1.9 | 3.1 | 13.7 | 22.9 |
| test: random | 1.8 | 3.2 | 5.1 | 7.0 |
| dev (T=2016, mean of 5 seeds): world model + rule | 11.5 | 15.0 | 22.0 | 30.2 |
| dev: city policy | 2.2 | 6.6 | 7.1 | 14.2 |

- **Dev and test agree:** 2.78× and 2.83× city.
- **Run 1** (`reports/backtest/PREREG.md`) protected 0.37 households at 1×.
- **Bootstrap CI vs city:** excludes 0 at 1× and 2×, includes 0 at 3× and 5×.

**Presentation rule (from the user):** the headline uses the world-model candidate. Do not put better-scoring non-WM candidates in results. They are documented in `reports/risk/b/README.md`.

## Rules (still in force)
- **No heavy local compute.** The laptop has frozen several times. Locally, only edit, run git and do tiny checks. Anything that trains, routes or loads big parquet files runs on Savio or Modal.
- **Test split:** at most **3** test calls in total. **All 3 are used**: `c_wm_oos_blend` (backtest), the site export (`models/demo/export.py risk`: sweep, PR-AUC/p@k, futures; logged in `reports/risk/test_log.csv`), and `eval_final --final`. No new models were scored; ask the user before any further test read. Ask the user before each one. The world model's own locked probe test (`python -m models.world_model.eval_final --final`) has **not** been run; treat it the same way and ask first.
- **Git:** curt commit messages, **no Claude / Co-Authored-By attribution**. Never push to main; use a branch and a PR. Never commit data.

## Where things live
- **Repo:** branches `b-risk-lgbm` (#3), `b-patient-sim` (#2) and `c-world-model` (#1) are all merged into `main`.
- **Savio** (`ssh savio` via WSL; if it says "Permission denied", ask the user to re-auth, never log in yourself). Root `S=/global/scratch/users/christophervu/dssg`:

  | path | what it is |
  |---|---|
  | `$S/repo` | stream A's checkout. `$S/repo/data/{raw,interim,processed}` is the **shared data**, written only by the data owner. |
  | `$S/repo-c` | world-model code dir. Its `data/processed` is a real directory of symlinks to A's files, plus WM outputs: `parcel_embeddings.parquet`, `parcel_next_embeddings.parquet`, `parcel_oos_embeddings.parquet`, `cell_embeddings.parquet`. **Run risk candidates that use WM features from here.** |
  | `$S/repo-lgbm` | the LightGBM track's code dir. |
  | `$S/repo-b` | the patient sim code dir. |
  | `$S/venv` | harness / LightGBM environment. |
  | `$S/envs/wm` | torch environment, plus `wm.tar.gz` (conda-pack). |
  | `$S/venv-modal` | Modal client. |

  Run a candidate: `REPO=$S/repo-c sbatch $S/repo/slurm/harness.sbatch <cand> dev`. The log goes to `$S/logs/`.
- **Modal** (GPU, faster than the Savio A5000 queue). Profile `christopher-vu` is set up on Savio.
  - **Volume** `wm-data` is laid out like the repo.
  - **App:** `models/world_model/modal_app.py`.
    - `modal run models/world_model/modal_app.py` runs sweep → select λ → full runs → export.
    - `...::oos_main` runs the rolling-origin features.
  - **Budget:** about $3–5 used of $30. The user should rotate the API token, which was pasted in chat.

## Components

**Risk + optimizer** (`models/risk/`)
- `harness.py` is the leak-proof dev/test harness. Do not edit it, or the value, cost, budget or city-policy logic.
- Candidates live in `models/risk/cands/`. The `c_*` files are the world-model ones; `c_lib.py` joins the WM features.

**SF World Model** (`models/world_model/`, details in its `README.md`)
- **Model:** a 5.08M-parameter transformer over H3 res-9 cell-years.
- **Loss:** (1−λ)·masked-token + λ·next-year. The next-year part includes a parcel-level head that predicts next-year sale, buyout and block-face Ellis/OMI notices.
- **λ:** 0.1, selected leak-free by parcel-head skill on the cutoff-2016 sweep.
- **Out-of-time features** (`oos.py`): for each year t, features come from a model with cutoff t, so no feature ever saw its own outcome. Stacking in-sample WM features into one LightGBM with hand-made features hurt (a leak, plus too few positives). Combining at the score level works.
- **Tests:** `tests/world_model` passes 19/19 (Savio fixture job). It includes the leakage locks: no sponsor data and no year > cutoff, and test targets are readable only with `--final`.

**Patient sim** (`sim/patient/`, `etl/sim/`)
- **Pipeline:** PUMS personas → migration model → r5 access to the nearest open clinic, pharmacy and grocery.
- **Migration model:** held-out JSD 0.25, vs 0.42 for the population-proportional baseline.
- **In flight:** Savio job **41328444** (`$S/repo-b`, log `$S/logs/patient_sim_41328444.log`, 6 h limit).
  - Transit is finished.
  - Car routing runs at about 31 min per 400 of 17,128 origins, which is about 22 h, so **it will time out**.
- **Fix before rerunning:**
  - Cap car routing: limit the search radius or max time, or reuse transit origins with a small car POI set, or approximate car times with OSRM.
  - Or submit car as its own array job.

**Data / ETL** (`etl/`)
- Done: spine, sales, labels, tokens for cutoffs 2016, 2018 and 2025 (public only), `probe_targets`, `risk_dataset` and `displaced_households`.
- Not done: permit-text embeddings and sponsor tokens for 2025. Both are optional.

## Final round (branch `a-final`)
- **Site** `web/` (Vite + React + deck.gl, static JSON in `web/public/data/`, gitignored; built by `python -m models.demo.export {risk,futures,wm,sim}` on Savio from `$S/repo-final`). Press **P** for the guided tour used in the video. Hexagons only (picks at H3 res-8, ≥5 eligible buildings).
- **Futures:** hexagon-level gamma-Poisson; level and spread set on dev. Test: real 2019–24 total inside the 80% band in 75% of hexagons. Citywide the model over-forecasts (flat ~440/yr vs actual 348 → ~110/yr through 2020–24); shown on the site as a stated miss.
- **Budget sweep beyond 5×:** the city pattern catches up (ahead at 10× and 15×, behind at 20×). The site shows the pre-registered 0.5–5× range.
- **Not shown (user rule):** LightGBM without the world model scores higher PR-AUC on "any displacement" (0.190); ours is higher on building buyouts and top-10% capture.
- Video/deck plan: `docs/VIDEO_PLAN.md`. Reproducibility: `Makefile`.

## What's left (priority order)
1. **Patient sim numbers.** Fix the car-routing cost and rerun on Savio, then run the hand-check of 20 transit pairs.
2. **Demo / figures:**
   - budget-sweep curve (1×–20× city budget) for `c_wm_oos_blend` vs city and random;
   - the "frozen in 2018" top-risk map;
   - the budget slider app;
   - the λ-sweep plot and UMAP map, which are already in `reports/wm/`.
3. **Fairness table** for `c_wm_oos_blend` picks: supervisor district, size band, ACS tercile. The ACS targets need the Census key.
4. **Optional, ask the user first:** the world model's locked probe test (`eval_final --final`), and risk PR-AUC / precision@k.
5. **Deck + ≤5 min video** following PRD §1. Stretch: Loophole Finder.

## Honest caveats to keep
- **Dev is small:** 23 city picks, and the CIs at high budgets include 0.
- **Rule of thumb:** a one-line rule of thumb (sold in the last 2 years × 10 + nearby Ellis/OMI notices) scores 0.93× of ours on test. We are ahead at 2× and 3× and behind at 1× and 5×.
- **Buyout data starts 2015-03**, so the parcel head's buyout output isn't used out-of-time.
- **Patient sim:** within-SF moves are excluded (66% of movers), and personas skew high-income.
