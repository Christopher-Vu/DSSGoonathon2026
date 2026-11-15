# Session B (LightGBM track): dev results

These come from Savio `savio3_htc` harness jobs 41330834–37, 41330907–09 and 41330974–76. Dev split: decide at the end of 2016, outcomes 2017–19, 7,773 buildings, 23 city picks, 1,178 households in total. The rule of thumb scores 3.22 vs city.

| cand | vs city | vs rule | 1× | 2× | 3× | 5× | idea |
|---|---|---|---|---|---|---|---|
| b_twohead | 2.81 | 0.87 | 12.3 | 13.5 | 19.6 | 36.5 | buyout h=1 + block-face Poisson; E[tenants] ∝ units |
| b_twohead_nomono | 2.71 | 0.84 | 10.5 | 14.8 | 17.6 | 37.7 | no monotone constraints |
| b_buyout_only | 2.78 | 0.86 | 12.2 | 13.5 | 19.4 | 35.5 | block-face head off, so it adds ~nothing |
| b_twohead_h2 | 1.90 | 0.59 | 8.5 | 9.9 | 12.6 | 22.6 | h=2 buyout windows: worse |
| b_flat | 3.60 | 1.12 | 14.5 | 18.0 | 25.7 | 49.1 | **flat E[tenants \| buyout]**: the key fix |
| b_calheur | 2.96 | 0.92 | 10.7 | 17.0 | 21.5 | 37.9 | rule's 12 cells, rates learned on recent windows |
| **b_best** (= b_blend) | **3.98** | **1.24** | 14.8 | 17.6 | 35.4 | 53.5 | 0.5 × b_flat + 0.5 × b_calheur |
| b_sens_l07 | 3.66 | 1.14 | 14.5 | 18.2 | 26.4 | 50.3 | sensitivity λ=0.7 |
| b_sens_l03 | 2.71 | 0.85 | 4.4 | 15.8 | 23.6 | 59.8 | sensitivity λ=0.3 |
| rule of thumb | 3.22 | 1.00 | 11.2 | 16.6 | 24.8 | 44.6 | |
| city policy | – | – | 2.2 | 6.6 | 7.1 | 14.2 | |

**Caveats**
- The bootstrap CI of the gain over the rule spans 0 at every budget, because dev rests on a handful of buildings.
- At λ ≥ 0.5 the score is stable at ~1.1–1.2. λ=0.5's 3× result (35 vs ~26) is probably partly luck.
- Dev has only 2 buyout-era training snapshots (t=2014, 2015). Test (T=2018) has 4, so the buyout head should be *stronger* on test, while the regime there is more buyout-heavy.
