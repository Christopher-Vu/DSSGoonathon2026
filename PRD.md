# PRD — SF World Model: Buy It Before the Flipper Does

**Event:** 8th Annual Datathon for Social Good (Advanced Track) · **Status:** Draft v1, 2026-11-14
**Stretch goal:** Loophole Finder (AI-landlord red team of SF tenant law)

---

## 0. One-line pitch

> We pretrained a foundation model of San Francisco, used it to predict which rent-controlled buildings are about to be emptied, and showed that with the city's own Small Sites budget it would have protected **N× more tenants** from 2019–2024 than the buildings the city actually bought.

Everything in this doc works backwards from that sentence. If a component doesn't make a number in that sentence real, or isn't in the demo, it gets cut.

---

## 1. What gets presented (work backwards from here)

Deliverables: GitHub repo, slide deck, ≤5-minute video. Video beats, in order:

| # | Beat (time) | What's on screen | Machinery it proves |
|---|---|---|---|
| 1 | **Hook** (0:00–0:30) | A real Ellis Act notice on a real Mission block. "This was predictable in 2018." | — |
| 2 | **The world model** (0:30–1:15) | Map of SF colored by the model's learned hexagon embeddings (UMAP→RGB). Neighborhood boundaries show up even though the model never saw any. Table: the same embeddings predict evictions, health violations and overcrowding better than hand-made features. λ-sweep curve: adding next-year prediction to the training objective improves the embeddings. | §4.2 SF World Model |
| 3 | **The risk model** (1:15–2:00) | "Frozen in 2018": the model's top-risk buildings, then a time-lapse of 2019–2024 notices landing on them. PR curve vs. baselines. | §4.3 Displacement risk |
| 4 | **The money shot** (2:00–3:15) | Live app: drag the budget slider, buildings light up, a counter shows "tenants protected" for **our picks vs. the city's actual picks vs. random vs. a heuristic**. A fan of 1,000 sampled futures for one building. | §4.4 Futures + §4.5 Optimizer |
| 5 | **What eviction costs** (3:15–4:00) | 10,000 simulated displaced households: where they move (real Census migration flows) and how their transit time to the nearest clinic or pharmacy changes. "Each eviction costs X more minutes to care, and Y% lose 30-minute access." | §4.6 Patient sim |
| 6 | **Honesty** (4:00–4:40) | Fairness table across neighborhoods, building sizes and tract demographics; 3 failure cases; the ethics choice (we don't publish a building-level "target list"). | §5 Evals |
| 7 | **Stretch** (4:40–5:00) | Loophole Finder: an AI landlord rediscovers a loophole SF actually closed. | §7 |

**Headline numbers we must produce** (the deck is blank without them):
- `N×`: tenants protected, our picks vs. actual Small Sites picks, at the same budget (2019–2024 backtest).
- Risk model PR-AUC and precision@k vs. the best baseline.
- World-model linear-probe lift over hand-made features, averaged over ≥3 downstream targets.
- Patient sim: median Δ minutes to care and % losing 30-minute access for displaced households.

---

## 2. Minimum viable scope (cut line)

**In (MVP, all machinery intact):**
1. Data spine: parcel × year panel 2007–2025 with labels joined (§3).
2. SF World Model: transformer over H3 hexagons trained with masked-token reconstruction + next-year prediction (λ-weighted, with a λ sweep), pretrained on data from 2018 or earlier (§4.2).
3. Displacement risk model: gradient-boosted model on features + world-model embeddings, with baselines (§4.3).
4. Futures: Monte Carlo samples from the calibrated hazard. *Not* a generative diffusion model; sampling is enough for the fan chart and the optimizer (§4.4).
5. Budget optimizer + 2018→2024 counterfactual backtest vs. actual Small Sites picks (§4.5).
6. Patient sim: PUMS-derived households, real migration destinations, Places + transit access (§4.6).
7. Streamlit + pydeck demo app with budget slider, futures fan and patient-sim panel.
8. Advanced Track evals: baseline, held-out test, ≥3 subgroups, failure examples, hand-checked joins, `make all` reproducibility.

**Out (explicitly cut):** a learned diffusion model for futures; a building-level public map; a surrogate model for the patient sim; multi-city coverage; an LLM chatbot.

**Stretch (only after the MVP numbers exist):** Loophole Finder (§7). A surrogate for the patient sim. A 2021→2024 second backtest window.

---

## 3. Data: verified inventory and joins

Verified 2026-11-14 against the live DataSF APIs (`https://data.sfgov.org/resource/<id>.json`).

### 3.1 External (public): verified to exist

| Dataset | DataSF id | Rows / range | Grain & join key | Role |
|---|---|---|---|---|
| Eviction Notices | `5cei-gny5` | 49,195 notices, 1997-01 → 2026-11; 24,622 since 2010 (1,699 Ellis, 3,083 owner move-in, 2,609 non-payment) | **Block-face only** ("1200 Block Of 22nd Avenue") + slightly jittered point; boolean column per eviction reason | Label (block-face) |
| Buyout Agreements | `wmam-7g8d` | 8,761 rows, 2015-03 → 2026-11 (~3.9k have `buyout_agreement_date`; the rest have only a disclosure date; check) | **Full street address** + point; amount, # tenants | Label (building) |
| Assessor Historical Secured Tax Rolls | `wv5m-vpq2` | ~200–213k parcels per year, **every year 2007–2025** | `parcel_number` (block+lot), `property_location` string; units, year built, use, assessed land/improvement value, `current_sales_date`, exemptions | **Spine** + features |
| Parcels – Active and Retired | `acdm-wktn` | all parcels | `blklot` → centroid lat/lon, neighborhood, supervisor district | Geocoding the spine |
| Building Permits | `i98e-djp9` | full history | block/lot; existing vs. proposed units, cost, description | Features (renovations, unit removal) |
| MOHCD Affordable Housing Portfolio | `pyxv-n29e` | **56 Small Sites buildings, 435 units** | address + lat/lon | Benchmark: "city's actual picks" |
| MOHCD Pipeline | `aaxw-2cb8` | 2 more Small Sites (2024, 2025) with `block`/`lot` and `ssp_rad_acquisition_date` | block/lot | Benchmark (recent) |
| Rent Board Housing Inventory | `gdc7-dmcn` | 561,935 unit records, 2022 → 2027 | block-face, rent bucket, occupancy | Forward-looking features **only** (starts 2022, so it would leak into the 2018 backtest) |
| ACS 5-yr (tract) + PUMS microdata | Census API | standard | tract GEOID; PUMS has `MIGPUMA` (where people lived 1 year ago) | Fairness subgroups, sim personas, migration destinations; *not yet pulled* |
| GTFS (Muni, BART, AC Transit) | 511.org / agency feeds | standard | stops/routes | Patient-sim travel times; *not yet pulled* |

**Joins spot-checked:** buyout addresses (`1523 33rd Ave`, `1969 San Jose Ave`) and Small Sites addresses (`1015 Shotwell`, `1201 Powell`) each matched exactly one tax-roll parcel by address-string match. Full join rates get measured in hour 0–2 (§6).

**Free signal:** for both Small Sites buildings checked, the tax roll's `current_sales_date` is 2017, which is the city's actual purchase. The portfolio's `estimated_actual_construction` field is not the acquisition date. **Use the assessor sale date as the acquisition date.** More generally, the year-over-year `current_sales_date` changes in the tax roll give us a **full sale-event history for every parcel**. Under Prop 13 an assessed-value jump at sale ≈ the sale price, so we can estimate prices without Zillow.

**Key gotcha:** evictions are block-face only, not building-level. Handling it:
- Building-level label = buyout at this address (exact).
- Block-face label = Ellis/owner-move-in notice on this block-face, assigned to that block-face's rent-controlled multi-unit buildings in proportion to their units.
- The risk model has two heads (building, block-face); the optimizer credits a purchase with buyouts at that building + its unit-share of block-face notices. Hand-check 50 of each (§5.5).

### 3.2 Sponsor datasets (≥1 required)

| Dataset | Status in shared Drive | Use |
|---|---|---|
| Dataset | Files (verified 2026-11-14) | Use |
|---|---|---|
| **Zillow** | `Zillow_part1-3.csv` (6.9 GB), 139 columns. `parcelId` = **SF assessor block+lot** (exact join, ~85% filled). `priceHistory` back to 1973; `rentZestimate` ~90% filled | Forward model: rent gap (`rentZestimate` vs. Rent Board rent bucket) and flip history. Price-history ablation |
| **Google Maps Places** | `google_maps.csv` (483 MB), ~100k POIs, `open_hours` 53% filled; bounding box includes Santa Cruz | **Patient sim** (clinics, pharmacies, grocery) + forward world-model tokens |
| **Reddit** | `reddit_posts.csv` (4.8 GB), 20 subreddits, **2025+ only** | Forward world-model text tokens (geotagged housing posts) |
| Google Maps Reviews | `google_maps_reviews.csv` (211 MB), mostly recent | Optional forward tokens (clinic quality in the sim) |

**Rule: sponsor data stays out of the 2018 backtest entirely.** Reddit and Reviews are 2025+, and Places and Zillow are current snapshots, so any of them would leak the future. Zillow's pre-2019 `priceHistory` also carries a selection bias: a building appears only if it was listed in 2025–26.

- **Backtest (§4.5):** public data only.
- **Sponsor data powers** the ≤2025 world-model checkpoint, the forward-looking 2025 risk map, and the patient sim. That satisfies the "use ≥1 sponsor dataset" requirement.
- **Optional ablation:** Zillow `priceHistory` features in the backtest, labeled as biased and never in the headline.
- Sponsor samples are not representative (the Zillow sample is CA-wide; the Reddit sample is random subreddits), so check distributions on the full files.

---

## 4. System

```
 raw (DataSF, Census, GTFS, sponsor)
        │  etl/   (parcel×year spine, block-face labels, hex tokens)
        ▼
 ┌────────────────┐   embeddings   ┌─────────────────┐  calibrated hazard  ┌──────────────┐
 │ SF World Model │ ─────────────▶ │ Risk model      │ ──────────────────▶ │ Futures (MC) │
 │ (H3 transformer)│               │ (GBM, 2 heads)  │                     └──────┬───────┘
 └────────────────┘                └─────────────────┘                            │
                                                                                  ▼
                       ┌───────────────────┐   displaced households   ┌─────────────────────┐
                       │ Patient sim       │ ◀─────────────────────── │ Budget optimizer    │
                       │ (PUMS + Places +  │                          │ + 2018→24 backtest  │
                       │  GTFS)            │                          └─────────────────────┘
                       └─────────┬─────────┘                                     │
                                 └──────────────► Streamlit demo ◀───────────────┘
```

### 4.1 Unit of analysis and time discipline
- **Building universe:** parcels likely under rent control: built before 1979-06-13, ≥2 units, residential use. (Rent-control eligibility is approximated; note this in the limitations.)
- **Snapshot year `t`:** features use only data dated ≤ `t`. **Main backtest: `t = 2018`, outcomes 2019-01 → 2024-12.**
- **The world model used in the backtest is pretrained only on ≤2018 data.** A second checkpoint pretrained on ≤2025 data powers the forward-looking demo map.

### 4.2 SF World Model (the catchy part)
- **Cells:** H3 resolution 9 (~0.1 km², ~1.2k cells in SF); resolution 10 as an ablation.
- **Tokens per cell-year:** every entity in the cell becomes a token: `type embedding + numeric features + optional text embedding`. Types: parcel (units, age, use, value, years since sale), permit, sale event, eviction and buyout counts (lagged). **The ≤2025 checkpoint only** adds sponsor tokens: Places POI (category, rating), Reddit post (frozen MiniLM sentence embedding), review, Zillow listing (rent estimate, status). The ≤2018 checkpoint has no sponsor tokens. Add a 2-D sinusoidal position code within the cell and a year embedding.
- **Objective:** `L = (1 − λ) · L_MTM + λ · L_next`
  - `L_MTM`: masked token modeling. Mask 15–30% of tokens; reconstruct the type with cross-entropy and the features with MSE. This term learns what a cell *is*.
  - `L_next` (**the world-model term**): from the cell's year-`t` tokens, predict its year-`t+1` **summary token**. The summary token is an observable vector: counts by token type (sales, permits, POIs, posts), mean and quantiles of the parcel numeric features (assessed value, years since sale), and the counts of new sale events and permits. A small MLP head reads the cell's `[CELL]` token; loss = MSE on standardized features + Poisson NLL on counts. This term learns how a cell *changes*, which is what makes it a world model, not just an encoder.
  - Predict observable features, not the next-year *embedding*. That avoids representation collapse and needs no EMA target encoder.
  - **Leakage rule:** for the backtest checkpoint, every `(t, t+1)` training pair has `t+1 ≤ 2018`.
- **λ ablation sweep (required; a slide in beat 2):**
  - Sweep λ ∈ {0, 0.1, 0.25, 0.5, 0.75, 1.0}. λ=0 is a pure encoder, λ=1 is pure dynamics.
  - Each run uses the same seed and token budget. The 6 runs execute in parallel as a Slurm array on A5000s, so the sweep takes ~15 minutes.
  - **Selection is on the validation window only:** probe PR-AUC on outcomes from `t=2016` (2017–2019) + `L_next` error on 2018. The 2019–2024 test outcomes are never used for selection.
  - The winning λ gets the full-length pretrain. Report a curve of probe lift and next-year error vs. λ; the expected shape is an interior optimum. If λ=0 wins, report that honestly.
- **Bonus once `L_next` exists:** autoregressive rollouts (`t → t+1 → t+2`) give a learned alternative to hazard sampling for the futures fan (§4.4). That's a stretch only; the MVP futures stay Monte Carlo.
- **Size:** **≤10M parameters** (≈6 layers, d=256). About 1.2k cells × 12 years ≈ 15k sequences, a few hundred tokens each. **The full pretrain takes minutes on one A5000** (Savio `savio4_gpu`). There is no overnight training and no Modal.
- **Output:** a 256-d embedding per cell-year; buildings inherit their cell's embedding plus their own parcel token's contextual embedding.
- **Probe eval (beat 2):** frozen embeddings + a ridge/logistic probe on ≥3 targets the model never saw. Compare against hand-made features alone and hand-made + embeddings:
  1. block-face eviction rate (2019–24)
  2. buyout rate
  3. overcrowding share from ACS tract data, aggregated
  4. (optional) DataSF restaurant inspection failure rate, or life expectancy from CDC PLACES

  Split test cells spatially in blocks (H3 res-6 parents) so neighboring cells can't leak into the test set.

### 4.3 Displacement risk model
- **Targets (3-year horizon from `t`):** `y_bldg` = any buyout at the building; `y_bf` = any Ellis or owner-move-in notice on the block-face. Report both; the combined score is `p_bldg + share × p_bf`.
- **Features:** years since last sale, number of sales in 10 years, assessed-value jump at last sale (≈ price premium), units, age, permits (cost, unit removal, capital improvement), prior notices on the block-face, neighborhood, tract ACS, cell embedding, and Zillow gaps if available.
- **Model:** LightGBM, then isotonic calibration on a validation year (`t = 2016` → outcomes 2017–2019) so the futures sampler gets honest probabilities.
- **Baselines (Advanced checklist):**
  - (a) base rate
  - (b) logistic regression on {years since sale, units, age}
  - (c) **"recently sold" rule**: the anti-eviction organizers' rule of thumb (sold in the last 2 years, multi-unit, pre-1979)
  - (d) GBM without embeddings (isolates the world model's contribution)
- **Metrics:** PR-AUC (rare events), precision@k at the budget's k, Brier score + reliability plot, ROC-AUC for reference.

### 4.4 Futures (MVP = sampling, not generation)
- From the calibrated yearly hazard per building, draw 1,000 Monte Carlo 5-year futures (event or no event per year; tenants displaced ~ units × occupancy × ACS household size).
- **Uses:** (1) the fan chart in the demo, (2) the optimizer's expected value and its 10th–90th percentile band, (3) an interval on "tenants protected".
- **The impact link the futures needed:** they feed the acquisition decision directly.

### 4.5 Budget optimizer + counterfactual backtest (the money shot)
- **Eligibility:** Small Sites-like buildings, 5–25 units (verify current program guidelines; also run a 3–25 variant).
- **Cost:** estimated per-unit acquisition cost from the assessor sale price/value, scaled to match the city's real per-unit spending (take total cost / units from public Small Sites reports; if unavailable, use the median estimated value of the 56 portfolio buildings).
- **Objective:** a 0/1 knapsack maximizing Σ E[tenants displaced]_i subject to Σ cost_i ≤ B. Solve exactly with OR-Tools or PuLP (a few thousand items is trivial).
- **Backtest:** at `t = 2018`, set B = the cost of the Small Sites buildings actually bought 2018–2023. Score each strategy by **realized** 2019–2024 displacement in the buildings it bought (buyout tenants + unit-share of block-face Ellis/owner-move-in notices):
  1. **Ours** (GBM + embeddings)
  2. **City's actual picks** (the portfolio buildings purchased in the window, dates from the assessor sale date)
  3. Random eligible (1,000 draws → a distribution)
  4. "Recently sold" heuristic
  5. Ours without embeddings
- **Assumption, stated on the slide:** a purchase prevents all later Ellis, owner-move-in and buyout displacement in that building. Upper bound, same for every strategy.
- **Caveat we'll show, not hide:** Small Sites often buys buildings *already* in trouble (e.g., after an Ellis notice), which is a different goal from prevention. Present it as "a prevention lens finds different buildings," not "the city is dumb."

### 4.6 Patient sim (flashy demo, real data underneath)
- **Who:** the households in the realized 2019–2024 displacement events (beat 5) and in the counterfactual "not protected" set. Personas drawn from SF PUMS records matched on tract and tenure: age, income, car ownership (`VEH`), language (`LANP`), disability.
- **Where they go:** sample destination PUMAs from **real PUMS migration flows** (people who lived in an SF PUMA 1 year earlier, by income band), then a random residential point in that PUMA, weighted by Zillow rentals affordable at their income (or ACS rent if Zillow is missing).
- **Access:** transit and walking time (r5py or OpenTripPlanner on GTFS + OSM) to the nearest **open** clinic, pharmacy and grocery store from Places, at a persona-relevant time (e.g., a weekday evening for workers). Use driving time if the persona has a car.
- **Outputs:** Δ minutes to care, % losing 30-minute access, broken down by language, car ownership and age. Dollar cost is optional, only if there's time.
- **Scale:** 10,000 households. Routing on precomputed origin-destination matrices, so it runs in minutes.

### 4.7 Demo app
Streamlit + pydeck:
- **(a)** world-model embedding map;
- **(b)** budget slider → chosen buildings (shown as hexagons or block-faces, not addresses, in the public build) + "tenants protected" counters for the 4 strategies;
- **(c)** a futures fan for a selected hexagon;
- **(d)** patient-sim before/after flows and Δ-access histogram.
Everything reads from precomputed parquet files, so there's no live training in the demo.

---

## 5. Evals (Advanced Track checklist mapping)

| Checklist item | How we satisfy it |
|---|---|
| **Model** | Pretrained transformer (World Model) + LightGBM risk model; both trained from scratch |
| **Baseline** | Base rate, logistic regression, the "recently sold" rule, GBM without embeddings; for the optimizer, actual city picks + random + heuristic |
| **Held-out test** | Temporal: train/tune on ≤2018 → test on 2019–2024 outcomes, never used in training or tuning. World-model probes use spatial block holdout |
| **Error & fairness (≥3 subgroups)** | Precision/recall and calibration by **(1) supervisor district / neighborhood**, **(2) building size (2–4, 5–9, 10–25, 26+)**, **(3) tract majority race/ethnicity (ACS)**, **(4) tract median income tercile**. Also check whether protected-tenant totals are fairly distributed by group under the optimizer |
| **Failure examples** | 3–5 hand-written cases: top false positives (high risk, nothing happened: why?) and missed events (Ellis on a building the model ranked low) |
| **Data decisions** | Block-face → building apportioning; rent-control approximation; acquisition date from the assessor. **Hand-check 50 eviction→block-face joins, 50 buyout→parcel joins, all 56 Small Sites→parcel joins** and report the error rates |
| **Reproducibility** | `make data && make pretrain && make risk && make backtest && make sim && make app`, raw → reported numbers, fixed seeds, configs in `configs/` |

**Ethics (judges will ask):** a building-level risk list is also a shopping list for speculators. The public repo and app show hexagons and block-faces only; building-level output is framed as a tool for program staff and land trusts. Note the false-positive harm (stigmatizing blocks) and that a model trained on *reported* notices misses informal displacement (unregistered buyouts, harassment).

---

## 6. 48-hour plan (assumes a team of 3–4; compute = Savio A5000s, jobs take minutes)

**Order:** the full MVP (§1 headline numbers + demo) comes first. The Loophole Finder starts only after that.

| Hours | Data/ETL | ML | Sim/App |
|---|---|---|---|
| 0–2 | Pull all DataSF tables; **measure join rates**; confirm the sponsor folders' contents | Set up the repo skeleton, configs, Makefile | Pull GTFS + PUMS; get r5py running |
| 2–8 | Parcel×year spine, sale events, block-face labels, hand-check samples | Hex tokenizer; Reddit geotagging + MiniLM embeddings | Places → POI table; origin-destination matrices |
| 8–14 | Freeze the `t=2018` feature table | **λ sweep (~15 min) → full WM pretrain (≤2018, then ≤2025)** + baseline risk models | Migration sampler + personas |
| 14–22 | Cost model for acquisitions | Probes; GBM ± embeddings; calibration | Sim runs on realized displacement |
| 22–30 | — | **Optimizer + backtest → headline N×** | App v1 (slider + counters) |
| 30–38 | Fairness tables, failure cases | WM (≤2025) checkpoint for the forward map | App v2 (futures fan, sim panel) |
| 38–44 | **Stretch: Loophole Finder** if the MVP numbers are locked | — | Slides |
| 44–48 | Repro run from clean | — | Record video, submit |

**Kill criteria:**
- If the hour-2 join rates for buyouts → parcels are <80%, switch the building label to block-face for everything.
- If the WM probe shows no lift by hour 22, keep it in the deck as "honest negative" and lean on the optimizer story.
- The headline never depends on the stretch goal.

---

## 7. Stretch: Loophole Finder (only after §1 numbers exist)

- **Environment:** a gym-style simulation of a single landlord's portfolio over 10 years. Tenant turnover, buyout acceptance and Ellis/owner-move-in rates come from the fitted risk model; rent dynamics come from Zillow/ACS. Legal actions and their costs are encoded from the **text of the SF Rent Ordinance (Admin. Code ch. 37)**. An LLM converts a snapshot of the ordinance into a rules config (allowed actions, costs, waiting periods, penalties, disclosure requirements).
- **Agent:** PPO landlord rewarded with portfolio net present value.
- **Benchmark:** feed it the ordinance **as it stood before a real amendment** and check whether it finds the exploit the city later closed:
  - unregistered cash-for-keys buyouts before the **2015 buyout disclosure ordinance**;
  - owner-move-in-then-re-rent before the **2017 owner-move-in enforcement changes**.

  Report the timestep at which the strategy appears vs. a fixed-rule landlord baseline, then run the LLM-patched rules and show the exploit's value collapse.
- **Demo line:** "Given 2014 law, our AI landlord invented the buyout loophole SF closed in 2015."

---

## 8. Open questions / risks

1. **Sponsor folders (Zillow, Places, Reviews) look empty to us.** Ask the organizers. Reddit is the guaranteed sponsor dataset.
2. Buyout rows with null `buyout_agreement_date` (~55%): use the disclosure date as a proxy? Check in hour 0–2.
3. Small Sites per-unit cost: find a public figure or estimate from assessor values.
4. 56 Small Sites buildings is a small benchmark. Report the random-pick *distribution* and bootstrap CIs, not just point estimates.
5. Rent-control coverage is approximated by build date and unit count; condos and single-family homes are mostly exempt.
6. Reddit geotagging precision: start with neighborhood-name and street matching, and hand-check 100 posts.

## 9. Repo layout (target)

```
etl/        pull_datasf.py, spine.py, labels.py, tokens.py, reddit_geo.py
models/     world_model/ (tokenizer, model, pretrain.py, probes.py), risk/ (train.py, calibrate.py)
sim/        futures.py, optimizer.py, backtest.py, patient_sim.py
app/        streamlit_app.py
configs/    *.yaml
reports/    figures, fairness tables, hand-check CSVs
Makefile    data → pretrain → risk → backtest → sim → app
```
