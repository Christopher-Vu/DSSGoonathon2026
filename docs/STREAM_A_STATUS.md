> Superseded by `STATUS.md` at the repo root (2026-11-15).

# Stream A status (handoff for the next session)

**Rule:** no heavy local compute. Run everything heavier than a quick check on Savio.

## Done (on main)
- `etl/datasf.py`: pulls 7 DataSF tables to `data/raw/datasf/` (domain is now `data.sf.gov`). Local copies exist.
- `etl/addr.py`, `etl/geo.py`: address/block-face normalization and the address→blklot index.
- `etl/spine.py` → `data/interim/spine.parquet` (3.93M parcel-years, 99.4% geocoded) and `sales.parquet` (225k sales).
  - 2018: 32,301 rent-controlled-proxy parcels, 175,714 units (city's commonly cited figure is ~170k).
  - Sales from 2023 on are undercounted, because only the current sale date is recorded.
- `etl/labels.py` → `buyout_events`, `eviction_events`, `blockfaces`, `parcel_blockface`, `ssp_picks` (in data/interim). Results in `reports/join_rates.json`:
  - buyouts→parcel 98.6%;
  - Ellis/owner-move-in notices since 2010 → block-face 99.1%;
  - Small Sites→parcel 96.6%; 52 of 58 have an acquisition year, 20 of them acquired 2019–23.
- `etl/tokens.py`:
  - `tokens/cutoff=2018` holds 2.98M tokens over 1,037 cells, public data only; `cutoff=2025` is public-only so far.
  - Writes `configs/token_features.yaml`. Run with `--no-text`; permit-text embeddings are **not done**, since 809k unique descriptions need a Savio GPU.
- `etl/probes.py` → `probe_targets.parquet` with 5 targets for 2016 and 2018. Splits use the H3 res-7 parent; CONTRACTS updated. The ACS targets need a Census API key.
- `models/risk/dataset.py` → `risk_dataset.parquet` (building × t 2009–2021, labels y_bf, y_bldg, y_val).
- `models/risk/train.py` (rewritten, **NOT yet run**): three heads, rounds chosen out-of-time on 2015, refit on t≤2015, Tweedie value head.
- `models/risk/backtest.py`: hedonic cost model, PuLP knapsack (pin `pulp==2.9.0`), city lookalike, random, heuristic.

## First backtest result (old model): bad
"Ours" captured 0.38 households vs. 12.6 for the recently-sold heuristic and 1.7 for random. The building head had learned "already had a buyout → another one." Fixes are in the train.py rewrite.
- Value is very sparse: 1,138 households over 7,744 buildings, and the budget (~$46M) buys only ~17–23 buildings.
- So report a **budget-sweep curve** (1×–20× city budget) plus bootstrap, not a single point.
- The heuristic's top 500 gets 3× the universe's households per $.

## Next
1. Set up the Savio env: `/global/scratch/users/christophervu/dssg`; rsync repo + `data/interim` + `data/processed` there. Run `models.risk.train` then `models.risk.backtest` as savio3_htc jobs.
2. Add the budget sweep and fairness tables (district, size band, ACS race/income tercile once there's a Census key).
3. Permit-text MiniLM embeddings on an A5000, then re-tokenize with text.
4. Sponsor tokens for cutoff=2025 (Zillow/Places/Reddit/Reviews) on Savio. The sponsor data is at `/global/scratch/users/christophervu/dssg/data/raw/sponsor/` (rclone pull job 41326936; verify it finished). Places is also local in `data/raw/sponsor/google_maps.csv`.
5. Merge the B and C PRs; submit their sbatch jobs; then `displaced_households.parquet` for B.

## Open asks for the user
- Census API key (free, api.census.gov/data/key_signup.html) → `.env` as `CENSUS_API_KEY`.
- The city lookalike design, needed because the city's own picks have censored outcomes.
