# Iteration round: beat the city with world model + LightGBM

**Goal:** on the backtest, a candidate built from LightGBM plus world-model embeddings protects more households than **city policy** at every budget, 1×, 2×, 3× and 5×.
- City policy is a model of which buildings the city chose, applied to buildings it did not buy.
- The final story is "WM + LightGBM beats the city, and beats the obvious rule of thumb". Ablations must show what the world model adds.

## Where things stand (run 1, test split, `reports/backtest/PREREG.md`)

At 1× budget (households protected):

| Strategy | Households |
|---|---|
| Ours (LightGBM Tweedie) | 0.37 |
| City policy | 1.9 |
| Random | 1.75 |
| Recently-sold rule of thumb | 12.6 |
| Oracle | 139 |

The rule of thumb is `(sold in the last 2 years)*10 + block-face Ellis/owner-move-in notices in the last 3 years`, divided by cost. It wins because about 2/3 of realized households are **buyouts**, and buyouts follow speculative purchases.

Why run 1 missed it:
1. **Regime shift.** Training rows (t ≤ 2015 with 3-year labels) are from the 2010–15 Ellis/owner-move-in era (the block-face rate was 0.215 in 2014, vs. 0.075 in 2018). The test window is buyout-era.
2. **Buyout data starts 2015-03.** With horizon 3, almost no training rows see buyouts. Shorter horizons (h=1 or 2) on recent snapshots carry the buyout regime.
3. **Objective.** Tweedie on the summed value is dominated by apportioned block-face notices. The buyout part (the part the rule captures) is small in training.
4. **Knapsack on noisy expected values** picks cheap noise. Greedy by expected value per dollar was 1.71 vs. 0.37 for the knapsack.
5. The world model was not used at all in run 1.

## The harness (stream A owns it; do not edit it, the value, cost, budget or city policy)

`models/risk/harness.py`. Read its docstring.

| split | decide | outcomes | city picks (budget + city policy) |
|---|---|---|---|
| `dev` | end of 2016 | 2017–2019 | 2017–2021 (31) |
| `test` | end of 2018 | 2019–2024 | 2019–2023 (prereg) |

- A candidate is `models/risk/cands/<name>.py` with `fit_predict(ctx) -> np.ndarray`. `ctx` only exposes data dated ≤ T: `ctx.rows`, `ctx.score`, `ctx.labels(h)`, `ctx.events`, `ctx.emb(kind)` and `ctx.cost(df)`.
- The baselines `heuristic` and `gbm_tweedie` (run 1) are on the dev leaderboard: `reports/risk/leaderboard_dev.csv`.
- The primary score is `score_vs_city`, the geometric-mean ratio over 1/2/3/5×. Also check `beats_city_all_budgets`, and compare against the heuristic.
- **Iterate on dev freely. Never tune on test.** `--split test --final` gets logged to `reports/risk/test_log.csv`.
  - At most **3 test calls total across both sessions**, each one for a frozen, committed candidate.
  - Tell the user before each test call.

## Savio (you may now submit your own jobs)

- ssh: `wsl -d Ubuntu -u cvudev -e ssh -o BatchMode=yes savio "<cmd>"`. If it fails with "Permission denied", ask the user to re-auth (PIN+OTP). Never try to log in yourself.
- Root: `S=/global/scratch/users/christophervu/dssg`. Shared data lives in `$S/repo/data`, written only by stream A.
- Your own code dir: rsync your worktree there (exclude `.git`, `data`, `.env`), then `ln -s $S/repo/data $YOUR_DIR/data`.
- Run a candidate on CPU: `REPO=$YOUR_DIR sbatch $S/repo/slurm/harness.sbatch <cand> dev`. Logs go to `$S/logs/harness-<id>.out`.
- **No heavy local compute.** The laptop has frozen twice already. Locally you only edit, run git, and do tiny checks.
- Only use `savio3_htc` (CPU) and `savio4_gpu` A5000 (`--qos=a5k_gpu4_ica`). No overnight jobs. Wall time ≤ 2 h.
- Commit rules: frequent commits, curt messages, **no Claude / Co-Authored-By attribution**. Push to your branch and open a PR to main.

---

## Session B: LightGBM track (`models/risk/cands/b_*.py`, code dir `$S/repo-lgbm`, branch `b-risk-lgbm`)

Ideas, in rough priority order:
- **Recent-regime labels:** `ctx.labels(1)` / `labels(2)` on t = T-2..T-1. Weight recent years more.
- **Two heads:**
  - buyout: P(buyout) × E[tenants], trained on 2015+ windows;
  - block-face: P(Ellis/OMI) × bf_share × E[notices].
  - Sum the two heads.
- **Sale-recency features at the building:** years since sale, sale price jump, buyer type if derivable, sale × rent-controlled units. Monotone constraints where they make sense.
- **A ranking objective** (lambdarank within t) or a per-dollar target, value/cost.
- Pick the selector (knapsack vs. greedy) on dev only.
- Keep `gbm_tweedie` and `heuristic` as reference rows. Make one clean `b_best.py` that loads WM embeddings when `use_emb=True`, so C can ablate.

## Session C: world-model track (`models/world_model`, code dir `$S/repo-c`, branch `c-world-model`)

1. **Fix in your branch:** `slurm/common.sh` needs `set +u` around `source activate` and `conda-unpack` (`CONDA_PREFIX` is unbound under `set -u`). Stream A patched the Savio copy so your fixture could run.
2. Run the real pipeline (sweep → pretrain → export) for **cutoff 2016 (the dev model) and cutoff 2018**.
   - Stream A is building tokens `cutoff=2016` in `$S/repo/data/processed/tokens/cutoff=2016` (job 41328606).
   - Export `parcel_embeddings` with years ≤ cutoff, including the cutoff year itself (`ctx.emb` reads `cutoff == T`, `year <= T`).
3. **Shoehorn the signal into the world model.** Add a parcel-level next-year head: from the parcel's own token output, predict its sale, buyout, and Ellis/OMI on the block-face in t+1 (Poisson). That's the rule-of-thumb mechanism, learned rather than hand-written. Keep it in the λ-weighted next-year loss, and keep the leakage locks.
4. Candidate `c_wm_gbm.py`: B's best LightGBM plus a PCA of the parcel embeddings (fit on years ≤ T). Ablations on dev:
   - `c_wm_only` (a linear/GBM model on embeddings alone);
   - `b_best` (no embeddings);
   - `c_wm_gbm`.
   - Report all three.
5. Text embeddings (MiniLM permit descriptions) are optional. If you want them, write the A5000 job that produces `text_emb_{C}.npy` per CONTRACTS.

## Done =
`c_wm_gbm` beats city policy at every budget on dev, beats the heuristic on dev, and is frozen and committed. Then one logged test call and a short write-up in `reports/risk/ITERATE_RESULTS.md`, covering:
- dev and test tables;
- the ablation;
- the honest caveats.

If either track is stuck after real attempts (still under city policy on dev after about 6 distinct ideas), stop and report to the user with the leaderboard. Don't grind.
