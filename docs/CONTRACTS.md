# Interface contracts

These are the files streams exchange. Each file has exactly one producer. Change a schema only by editing this doc in a PR that the main agent (stream A) approves. Every stream builds a **synthetic fixture** with the same schema so it can work before the real file exists.

General rules:
- All files are Parquet with snappy compression and live under `data/processed/`.
- Coordinates are WGS84 lat/lon.
- H3 uses **resolution 9**, `h3-py>=4` (`h3.latlng_to_cell`).
- Years are calendar years, stored as int16.

| File | Producer → consumer | Purpose |
|---|---|---|
| `tokens/cutoff={2018,2025}/*.parquet` + `text_emb_{cutoff}.npy` | A → C | World-model input |
| `configs/token_features.yaml` | A → C | Per-type feature names |
| `probe_targets.parquet` | A → C | Probe eval targets + splits |
| `cell_embeddings.parquet`, `parcel_embeddings.parquet` | C → A | Embeddings for the risk model |
| `displaced_households.parquet` | A (stream D) → B | Who gets displaced, from where |
| `sim_results.parquet`, `sim_summary.json` | B → app | Patient-sim outputs |

---

## 1. `tokens/cutoff=<C>/part-*.parquet` (A → C)

One row per token. The cutoff partition guarantees `year <= C`. **`cutoff=2018` contains no sponsor token types.**

| column | type | notes |
|---|---|---|
| `cell` | string | H3 res-9 index |
| `year` | int16 | year the token belongs to (state as of Dec 31) |
| `token_type` | string | one of `parcel, sale, permit, eviction_bf, buyout` (public); `poi, reddit, review, zillow` (sponsor, **cutoff=2025 only**) |
| `entity_id` | string, nullable | `blklot` for parcel/sale/permit/zillow tokens, else null |
| `f0` … `f15` | float32 | numeric features, standardized per type (fit on years ≤ C), NaN = missing/unused. Meaning per type is in `configs/token_features.yaml` |
| `x_off`, `y_off` | float32 | position within the cell in [-1, 1] (offset from the cell center / cell radius) |
| `text_idx` | int32 | row in `text_emb_<C>.npy` (float16, N×384, all-MiniLM-L6-v2), −1 if none |

`configs/token_features.yaml` maps `token_type → [feature names for f0..fk]`. Consumers must handle any K ≤ 16 and any new type in the yaml.

## 2. `probe_targets.parquet` (A → C)

| column | type | notes |
|---|---|---|
| `cell` | string | |
| `year` | int16 | snapshot year t (features ≤ t) |
| `target` | string | `eviction_rate_3y`, `nonpayment_rate_3y`, `buyout_rate_3y`, `sale_rate_3y`, `any_ellis_omi_3y` (binary); ACS `overcrowding`, `median_income_log` when available. Outcomes in t+1..t+3 |
| `value` | float32 | |
| `task` | string | `regression` or `binary` |
| `split` | string | `train` / `val` / `test`, assigned by **H3 res-7 parent** (spatial block holdout; res-6 has only ~8 parents in SF) |

**Time rule:** rows with `year=2016` (outcomes 2017–2019) are for selection and tuning. Rows with `year=2018` (outcomes 2019–2024) are **test only**, read solely by the final-eval entry point behind a `--final` flag.

## 3. `cell_embeddings.parquet` / `parcel_embeddings.parquet` (C → A)

| column | type | notes |
|---|---|---|
| `cell` / `blklot` | string | |
| `year` | int16 | |
| `cutoff` | int16 | 2018 or 2025 checkpoint |
| `lam` | float32 | λ of the checkpoint |
| `e0` … `e255` | float32 | `[CELL]` output (cell) or contextual token output (parcel) |

## 4. `displaced_households.parquet` (A / stream D → B)

| column | type | notes |
|---|---|---|
| `household_id` | string | unique within scenario |
| `scenario` | string | `realized_2019_2024`, `unprotected__<strategy>` |
| `origin_lat`, `origin_lon` | float64 | block-face or building point |
| `origin_tract` | string | 11-digit GEOID (2020 tracts) |
| `event_year` | int16 | |
| `event_type` | string | `ellis`, `omi`, `buyout`, `other` |
| `weight` | float32 | expected households represented (unit share × occupancy) |

## 5. `sim_results.parquet` (B → app)

One row per simulated household (≥10k per scenario).

| column | type | notes |
|---|---|---|
| `household_id`, `scenario` | string | joins to (4); synthetic households get `syn_<n>` |
| `age_band`, `income_band`, `lang`, `has_car`, `disability` | string/bool | persona from PUMS |
| `dest_puma` | string | 2020 PUMA GEOID |
| `dest_lat`, `dest_lon` | float64 | |
| `mode` | string | `transit_walk` or `car` |
| `depart` | string | e.g. `weekday_1800` |
| `t_{clinic,pharmacy,grocery}_{before,after}` | float32 | minutes to the nearest **open** POI, `inf` if > 120 |
| `acc30_{clinic,pharmacy,grocery}_{before,after}` | bool | reachable within 30 minutes |

`sim_summary.json` holds medians, Δs and %-lost-30-min overall and by `lang`, `has_car`, `age_band`, `income_band`, plus the run config.
