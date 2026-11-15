#!/bin/bash
# Patient-sim r5 run on Savio CPU (stream B; stream A submits). Never run r5 on the laptop.
#   fixture:  sbatch sim/patient/slurm/sim_full.sbatch fixture
#   full:     sbatch sim/patient/slurm/sim_full.sbatch /global/scratch/users/christophervu/dssg/data/processed/displaced_households.parquet
# Inputs expected under $SCRATCH/data (rsync from laptop, see sim/patient/slurm/README.md).
#SBATCH --job-name=patient_sim
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_normal
#SBATCH --account=ic_cdss170fall
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/global/scratch/users/christophervu/dssg/logs/patient_sim_%j.log
set -euo pipefail
ARG=${1:-fixture}
SCRATCH=/global/scratch/users/christophervu/dssg
REPO=${REPO:-$SCRATCH/repo-b}

module load java/22.0.1 || { module avail java; exit 1; }   # r5py needs Java >= 21
module load python/3.11 2>/dev/null || true
cd "$REPO"
if [ ! -x "$SCRATCH/venv-sim/bin/python" ]; then
  python3.11 -m venv "$SCRATCH/venv-sim"
  "$SCRATCH/venv-sim/bin/pip" install -q -r requirements-sim.txt
fi
source "$SCRATCH/venv-sim/bin/activate"
export SIM_DATA_ROOT=$SCRATCH/data
export JAVA_HOME=$(dirname "$(dirname "$(readlink -f "$(which java)")")")
export XDG_CACHE_HOME=$SCRATCH/cache          # r5py copies inputs + caches the network here
mkdir -p "$SCRATCH/logs" "$XDG_CACHE_HOME"
P=$SIM_DATA_ROOT/processed/sim

# ETL only if outputs are missing (normally rsynced prebuilt)
[ -f "$P/pums_households.parquet" ] || python -m etl.sim.pums_extract
[ -f "$P/blocks.parquet" ]          || python -m etl.sim.blocks_extract
ls "$P"/gtfs/*_clean.zip >/dev/null 2>&1 || python -m etl.sim.gtfs_clean
[ -f "$SIM_DATA_ROOT/raw/sim/osm/region.osm.pbf" ] || python -m etl.sim.osm_region
[ -f "$P/pois_places.parquet" ]     || python -m etl.sim.places_pois

if [ "$ARG" = fixture ]; then
  HH=$SIM_DATA_ROOT/processed/fixtures/displaced_households.parquet; CFG=configs/sim_fixture.yaml
else
  HH=$ARG; CFG=configs/sim.yaml
fi
# JVM heap capped under the job allocation; r5py reads --max-memory from sys.argv
python -m sim.patient.run --households "$HH" --config "$CFG" --pois-source places --max-memory 48G
python -m sim.patient.handcheck --config "$CFG" --n 20     # OSRM car checks need outbound https
echo "done: $(date)"
