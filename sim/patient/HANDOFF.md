# Stream B handoff (2026-11-14)

## Done (branch `b-patient-sim`, worktree `../dssg-b`, not pushed yet)

**Pipeline:** `python -m sim.patient.run --households <pq> --config configs/sim.yaml [--backend euclid] [--pois-source osm]`
- Personas: PUMS, sampled from the origin's SF PUMA, renters only.
- Destinations: a migration model P(dest PUMA | SF, income band), then a block point sampled ∝ HOUSING20.
- Routing: `access.py` (r5 or euclid) runs on unique H3 cells.
- Car households take min(car, transit).
- Outputs: `summary.py` and `figures.py`.

**ETL** (see `etl/sim/README.md`):
- `pums_extract`, `blocks_extract`, `osm_region`, `gtfs_clean`, `places_pois`.
- Every output is already built under `SIM_DATA_ROOT`: PUMS parquet, blocks, the `region.osm.pbf` road clip, OSM + Places POIs, and cleaned GTFS (`*_clean.zip`).

**Tests:** 19/19 pass (`tests/sim`).

**Euclid fixture run:** works in 6 s. Figures in `reports/sim/fixture/`.

**Migration validation:** held-out 2023 JSD 0.25 (model) vs 0.44 (uniform) vs 0.42 (population-proportional); bootstrap CI of the gap is [0.15, 0.24].

**Places spot check:** clinic 22/30, pharmacy 27/30, grocery 30/30. Places vs OSM is in `reports/sim/places_vs_osm.csv`.

**Handcheck tool:** `sim/patient/handcheck.py` (OSRM for car pairs, Google Maps URLs for transit). It has NOT been run yet because it needs `access_cells.parquet` from an r5 run.

## Not done / next

1. **No r5 run has finished.** Local r5 froze the laptop, so r5 runs **only on Savio**.
   - sbatch is ready: `sim/patient/slurm/sim_full.sbatch {fixture|<hh.parquet>}`.
   - Copy and submit steps are in `sim/patient/slurm/README.md`. Stream A submits.
2. (done) The sbatch script uses `_clean` GTFS, `--pois-source places`, `--max-memory 48G`, and skips ETL steps whose outputs exist.
3. Re-run the handcheck after the Savio run. Hand 20 transit URLs to a human.
4. Push the branch and open a PR. The PR body needs:
   - headline numbers;
   - open questions: within-SF movers excluded (66% of SF movers); personas skew high-income (consider conditioning on tenure length `MV`); "internist" listings counted as clinics;
   - proposed contract additions: extra columns `origin_household_id, origin_lat/lon, origin_puma, event_type, event_year, weight, cell_before/after`.
5. Report back to the user: branch, PR link, numbers, stubs, and asks for stream A (Savio run, real `displaced_households.parquet`).

## Gotchas
- r5py copies inputs into `%LOCALAPPDATA%/r5py` by file name. Rename the file if its contents change.
- Edit Python with the Edit tool. Heredoc string replacement broke `\n` once.
- In git bash, the Windows Python needs `C:/...` paths inside code; `/c/...` only works in env vars.
