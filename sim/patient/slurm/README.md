# Savio run (stream A submits)

r5 builds a JVM network that takes most of a laptop's RAM, so run it **only on Savio**.

## 1. Copy the code and the prebuilt inputs (~175 MB, from the laptop, via the DTN)

Run this from the main checkout:

```bash
S=christophervu@dtn.brc.berkeley.edu:/global/scratch/users/christophervu/dssg
git -C ../dssg-b archive b-patient-sim | ssh christophervu@dtn.brc.berkeley.edu \
  "mkdir -p /global/scratch/users/christophervu/dssg/repo-b && tar -x -C /global/scratch/users/christophervu/dssg/repo-b"
rsync -avR data/./processed/sim data/./processed/fixtures data/./raw/sim/osm/region.osm.pbf $S/data/
```

You don't need `raw/sim/{pums,tiger,gtfs}`, because the processed parquet files and the `*_clean.zip` files are already built. The sbatch script rebuilds an ETL step only when that step's output is missing.

## 2. Submit

```bash
cd /global/scratch/users/christophervu/dssg/repo-b
sbatch sim/patient/slurm/sim_full.sbatch fixture          # first: fixture, ~2 scenarios x 500 hh
sbatch sim/patient/slurm/sim_full.sbatch $PWD/../data/processed/displaced_households.parquet   # real, once A/D ships it
```

- Log: `/global/scratch/users/christophervu/dssg/logs/patient_sim_<jobid>.log`.
- Resources: 16 CPUs, 64 GB, a 48 GB JVM heap, and a 6 h limit.
- The first job builds the r5 network into `$SCRATCH/cache/r5py`, and later jobs reuse it.

## 3. Bring back

```bash
rsync -av $S/data/processed/sim_fixture/ data/processed/sim_fixture/     # or processed/sim/sim_results.parquet + sim_summary.json
rsync -av $S/repo-b/reports/sim/ ../dssg-b/reports/sim/                   # figures, handcheck.csv
```

`handcheck.csv` has 20 pairs:
- car pairs come with the OSRM time;
- transit pairs have a Google Maps URL, and a human checks those by hand.
