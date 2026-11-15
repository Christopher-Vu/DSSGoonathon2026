# Start prompt — Stream B: Patient Sim

You are building the **patient simulation** for the SF World Model datathon project. Read `PRD.md` (especially §1 beat 5 and §4.6) and `docs/CONTRACTS.md` before writing code. Another agent (stream A) owns the data spine and Savio. Stream C owns the world model. You own only what's listed below.

## Goal
For households displaced from SF, simulate **where they move** (real Census migration flows) and **how their access to care changes**: transit or car time to the nearest *open* clinic, pharmacy and grocery store, before (SF origin) vs. after (destination). Output `sim_results.parquet` and `sim_summary.json` per CONTRACTS §5. The demo line this has to support: *"Each eviction costs X more minutes to care, and Y% lose 30-minute access."*

## Rules
- **Git:**
  - work in your own worktree on branch `b-patient-sim`;
  - **commit frequently, with short curt messages** (e.g. `pums loader`, `r5 matrix`), and **no Claude/Co-Authored-By attribution, ever**;
  - push the branch and open a PR to `main` at each milestone. Never push to `main`.
- **Your folders:** `sim/patient/`, `etl/sim/`, `tests/sim/`, `configs/sim*.yaml`, `requirements-sim.txt`. Don't edit other streams' folders. To change `docs/CONTRACTS.md`, propose it in your PR description.
- **Data:**
  - never commit data or `.env`. Raw data goes in `data/raw/sim/`, outputs in `data/processed/`.
  - The 511 API key is `API_511` in `.env`; read it with python-dotenv and never print or log it.
- **Savio:** do **not** SSH to Savio. If something is too heavy for the laptop, write a `sim/patient/slurm/*.sbatch` script for the CPU partition (`--partition=savio3_htc --qos=savio_normal --account=ic_cdss170fall`) and hand it to stream A with the exact command. Assume scratch at `/global/scratch/users/christophervu/dssg`.
- **Environment:** Python 3.11; pin dependencies in `requirements-sim.txt`. r5py needs Java ≥21 (Savio has `module load java/22.0.1`).
- **Work against fixtures until the real inputs exist.** Write `sim/patient/fixtures.py`, which generates a fake `displaced_households.parquet` (e.g. 500 households at random SF residential points across a few tracts and the 4 event types).

## Data sources (all verified reachable on 2026-11-14)
- **PUMS** CA 2023 5-year: `https://www2.census.gov/programs-surveys/acs/data/pums/2023/5-Year/csv_pca.zip` (persons), `csv_hca.zip` (households). Fields: `PUMA`, `MIGPUMA`, `MIGSP`, `HINCP`, `VEH`, `LANP`, `AGEP`, `DIS`, weights `WGTP` and `PWGTP`.
  - **Gotcha:** the 2023 5-year file mixes 2010- and 2020-vintage PUMAs/MIGPUMAs by survey year. Use a crosswalk, or restrict to 2022–2023 survey years, and document which you chose.
- **PUMA boundaries:** Census TIGER/Line 2020 PUMAs for CA.
- **Transit:**
  - 511 regional GTFS: `https://api.511.org/transit/datafeeds?api_key=$API_511&operator_id=RG` (~60 MB zip; covers Muni, AC Transit, Tri Delta, SamTrans, VTA, etc.).
  - BART: `https://www.bart.gov/dev/schedules/google_transit.zip`. Caltrain: `https://data.trilliumtransit.com/gtfs/caltrain-ca-us/caltrain-ca-us.zip`. Skip either if the regional feed already includes it.
  - Stockton/San Joaquin RTD is outside 511. Either find its feed or cap destinations at the 9 Bay Area counties + Sacramento, and document the choice.
- **Roads:** OpenStreetMap `https://download.geofabrik.de/north-america/us/california/norcal-latest.osm.pbf`. Clip to the destination region with osmium to keep r5 memory reasonable.
- **POIs:**
  - Sponsor **Google Maps Places**: `data/raw/sponsor/google_maps.csv` (483 MB; may not be downloaded yet). Columns: `place_id, name, category, all_categories, address, lat, lon, rating, reviews_count, open_hours (~53% filled), permanently_closed, temporarily_closed`. The bounding box includes Santa Cruz, so clip to counties. Category matching is noisy, so map categories to `{clinic, pharmacy, grocery}` with a reviewed list and spot-check 30 per class.
  - **Until Places lands**, use OSM POIs (`amenity=clinic|doctors|hospital`, `amenity=pharmacy`, `shop=supermarket|grocery`) behind the same internal schema, then swap. Places is the sponsor dataset, so the final run must use it.
  - **Open-hours:** use `open_hours` when present. Otherwise use a category default (clinic Mon–Fri 9–17, pharmacy 9–21, grocery 8–22), and report the share of POIs on defaults.

## Method
1. **Personas:** for each displaced household, sample a PUMS household from the origin's SF PUMA, matched on renter tenure. This gives age band, income band, `has_car` (`VEH>0`), language (`LANP` → English/Spanish/Chinese/Vietnamese/Tagalog/Other), and disability.
2. **Destinations:** build an empirical distribution P(dest PUMA | origin SF PUMA, income band) from PUMS movers (`MIGPUMA` = an SF MIGPUMA, current `PUMA` outside SF), weighted with `PWGTP`. Smooth sparse cells toward the income-band marginal. Sample a destination PUMA, then a residential point in it (an OSM residential building centroid; later, Zillow `FOR_RENT` listings affordable at ≤40% of income if stream A provides them).
3. **Access:** use r5py `TravelTimeMatrix`, origins + destinations → POIs of each class.
   - Departure: weekday 18:00 for working-age personas, weekday 10:00 otherwise.
   - Mode: transit+walk, or car if `has_car`.
   - Value: nearest *open-at-departure* POI, capped at 120 minutes.
   - Compute "before" from the SF origin and "after" from the destination.
4. **Scale:** ≥10k households per scenario. Precompute origin–destination matrices on unique H3 res-9 origin cells, not per household.
5. **Summaries:** medians, the Δ distribution, and %-losing-30-min, overall and by `lang`, `has_car`, `age_band`, `income_band`.

## Evals and sanity checks (the deck needs these)
- Unit tests:
  - contract schema check on outputs;
  - a household with a car is never slower by car than by transit;
  - `before` is computed from SF;
  - no NaN where `inf` belongs.
- **Hand-check:** 20 origin–destination–time pairs against a public trip planner. Record them in `reports/sim/handcheck.csv`, along with the error distribution. If you can't query a planner, list the pairs and a planner URL for each so a human can check them.
- **Validation of the migration model:** hold out one PUMS survey year and compare its destination-share distribution to the model's (Jensen-Shannon divergence vs. a uniform-over-Bay-Area baseline).
- Note in the summary any categories where the sponsor Places data and OSM disagree badly.

## Deliverables / done when
- `python -m sim.patient.run --households <parquet> --config configs/sim.yaml` produces both contract outputs. It runs end to end on the fixture in under 10 minutes on the laptop, and has a documented Savio CPU path for the full run.
- Figures in `reports/sim/`: a migration flow map (SF → destinations), a Δ-minutes histogram, and %-lost-30-min by subgroup.
- `etl/sim/README.md` with exact download commands and the PUMA-vintage decision.
- PR description with the headline numbers on the fixture, open questions, and any proposed contract changes.

**Report back** (when done or blocked): branch name, PR link, what runs, numbers, what's stubbed, and what you need from stream A.
