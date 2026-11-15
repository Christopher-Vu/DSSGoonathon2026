# etl/sim — patient-sim inputs (stream B)

Everything lands under `$SIM_DATA_ROOT` (default `data/`). From a worktree, point it at the main checkout:
`export SIM_DATA_ROOT=/path/to/main/data`. Nothing here is committed.

## 1. Downloads (all verified reachable 2026-11-14)

```bash
R=${SIM_DATA_ROOT:-data}/raw/sim
mkdir -p $R/pums $R/tiger $R/osm $R/gtfs $R/gtfs_extra
# PUMS CA 2019-2023 5-yr (persons 266 MB, households 98 MB) + data dictionary
curl -fL -o $R/pums/csv_pca.zip https://www2.census.gov/programs-surveys/acs/data/pums/2023/5-Year/csv_pca.zip
curl -fL -o $R/pums/csv_hca.zip https://www2.census.gov/programs-surveys/acs/data/pums/2023/5-Year/csv_hca.zip
curl -fL -o $R/pums/PUMS_Data_Dictionary_2019-2023.txt https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2019-2023.txt
# TIGER 2020: PUMAs (6 MB) and blocks with HOUSING20 (382 MB)
curl -fL -o $R/tiger/tl_2020_06_puma20.zip https://www2.census.gov/geo/tiger/TIGER2020/PUMA20/tl_2020_06_puma20.zip
curl -fL -o $R/tiger/tl_2020_06_tabblock20.zip https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_06_tabblock20.zip
# OSM NorCal (656 MB)
curl -fL -o $R/osm/norcal-latest.osm.pbf https://download.geofabrik.de/north-america/us/california/norcal-latest.osm.pbf
# GTFS: 511 regional (needs API_511 from .env; never echo it), SacRT, San Joaquin RTD
set -a; . ./.env; set +a
curl -fL -o $R/gtfs/511_RG.zip "https://api.511.org/transit/datafeeds?api_key=${API_511}&operator_id=RG"
curl -fL -o $R/gtfs/sacrt.zip  http://iportal.sacrt.com/GTFS/SRTD/google_transit.zip     # https times out; http works
curl -fL -o $R/gtfs/sjrtd.zip  https://sanjoaquinrtd.com/RTD-GTFS/RTD-GTFS.zip
# BART / Caltrain are already inside 511 RG (agencies BA, CT) -> kept only in gtfs_extra, not loaded
curl -fL -o $R/gtfs_extra/bart.zip https://www.bart.gov/dev/schedules/google_transit.zip
curl -fL -o $R/gtfs_extra/caltrain.zip https://data.trilliumtransit.com/gtfs/caltrain-ca-us/caltrain-ca-us.zip
# Sponsor Places (shared Drive): data/raw/sponsor/google_maps.csv
```

## 2. Build steps (≈ 7 min on a laptop, dominated by the OSM clip)

```bash
python -m etl.sim.pums_extract     # 30 s  -> processed/sim/pums_{households,persons}.parquet
python -m etl.sim.blocks_extract   #  3 s  -> processed/sim/blocks.parquet (87k blocks, 3.8M housing units, 80 PUMAs)
python -m etl.sim.osm_region       #  5 min -> raw/sim/osm/region.osm.pbf (roads only, 88 MB) + processed/sim/pois_osm.parquet
python -m etl.sim.gtfs_clean       # 35 s  -> processed/sim/gtfs/*_clean.zip
python -m etl.sim.places_pois      #  5 s  -> processed/sim/pois_places.parquet + reports/sim/places_{spotcheck,vs_osm}.csv
```

r5py needs Java ≥ 21 in `JAVA_HOME` (any JDK 21+; IntelliJ's bundled `jbr` works on Windows; Savio: `module load java/22.0.1`).

## 3. Decisions

**PUMA vintage: no crosswalk needed, all 5 survey years used.** The prompt warned that the 2023 5-yr file
mixes 2010 and 2020 PUMAs. Verified on the actual file: the dictionary defines `PUMA` and `MIGPUMA` on 2020
definitions, and every survey year 2019–2023 uses exactly the 281 CA PUMA codes in TIGER 2020
(`tl_2020_06_puma20`). The release has already been re-coded, so we keep all five years (more movers) and
hold out 2023 for validation.

**SF is one MIGPUMA (`07500`).** "Lived in SF a year ago" can't be split by origin PUMA, so the destination
model is P(dest PUMA | SF, income band). The origin PUMA still drives persona sampling.

**Movers:** persons with `MIG=3`, `MIGSP=006`, `MIGPUMA=07500`, renter household (`TEN=3`), weighted by
`PWGTP`, with income band from household `HINCP×ADJINC`. That gives 3,101 records (73k weighted):
- 66% stay inside SF;
- 31% of those leaving SF go outside the 11-county region;
- 757 records land in region PUMAs outside SF and form the model.

Smoothing: band counts plus α=20 pseudo-records spread by the band-pooled marginal, which itself has a 1% uniform floor.

**Region = 9 Bay Area counties + Sacramento + San Joaquin.** All three GTFS feeds cover Tue 2026-11-17.
Movers to other CA counties are reported as a share but not simulated. Out-of-state movers are invisible in
the CA PUMS file.

**Within-SF moves are excluded** (spec: current PUMA outside SF). That is 2/3 of SF movers, so the sim
describes households that leave SF, not all displaced households. Flip `migration.include_within_sf`
to include them.

**Residential point:** a 2020 Census block internal point, sampled ∝ `HOUSING20` within the destination
PUMA. Blocks have real housing-unit counts; OSM building tags are mostly `building=yes` here. Zillow
affordable-rental weighting is a TODO if stream A provides it.

**POIs.**
- **Places** is the final source:
  - categories are mapped on the primary `category` only; secondary categories pull psychiatrists into "hospital";
  - names are then filtered by class;
  - "doctor" (11.6k individual listings), pediatricians and ER-physician listings are excluded.
- **Coverage:** Places' bbox stops at lon −121.55, which misses east Sacramento and San Joaquin. OSM POIs fill only that uncovered area.
- **Open hours:** taken from `open_hours`, which is JSON `{"Tuesday": "9 AM–7 PM"}` with `\n` between ranges. When missing, the category default applies (clinic 9–17, pharmacy 9–21, grocery 8–22), and hospitals default to 24 h (ER).
- **Spot check** (`reports/sim/places_spotcheck.csv`): four rounds of name filters, then a final judgment of clinic 22/30, pharmacy 27/30, grocery 30/30. The judgment is model-made from name and category only, and needs a human pass. Clinic false positives make care look closer both before and after the move, so they bias losses down.

**GTFS cleaning:** the 511 feed has 232 pathway nodes without `stop_name`, which r5 rejects. `gtfs_clean` fills
them with `stop_id` and drops `pathways.txt` and `levels.txt`. Cleaned files get a `_clean` suffix because r5py
caches inputs by file name and silently reuses stale copies.
