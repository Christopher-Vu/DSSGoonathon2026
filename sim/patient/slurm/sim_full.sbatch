#!/bin/bash
# Full patient-sim run on Savio CPU (stream B -> handed to stream A to submit).
#   sbatch sim/patient/slurm/sim_full.sbatch data/processed/displaced_households.parquet
#SBATCH --job-name=patient_sim
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_normal
#SBATCH --account=ic_cdss170fall
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/global/scratch/users/christophervu/dssg/logs/patient_sim_%j.log
set -euo pipefail
HH=${1:-data/processed/displaced_households.parquet}
SCRATCH=/global/scratch/users/christophervu/dssg
REPO=${REPO:-$SCRATCH/repo}

module load java/22.0.1
module load python/3.11 2>/dev/null || true
cd "$REPO"
if [ ! -d "$SCRATCH/venv-sim" ]; then
  python3.11 -m venv "$SCRATCH/venv-sim"
  "$SCRATCH/venv-sim/bin/pip" install -q -r requirements-sim.txt
fi
source "$SCRATCH/venv-sim/bin/activate"
export SIM_DATA_ROOT=$SCRATCH/data
export JAVA_HOME=$(dirname "$(dirname "$(readlink -f "$(which java)")")")
export XDG_CACHE_HOME=$SCRATCH/cache          # r5py caches the built network here

# inputs (idempotent; each step skips nothing, but all are minutes): see etl/sim/README.md
python -m etl.sim.pums_extract
python -m etl.sim.blocks_extract
python -m etl.sim.gtfs_clean
[ -f "$SIM_DATA_ROOT/raw/sim/osm/region.osm.pbf" ] || python -m etl.sim.osm_region
[ -f "$SIM_DATA_ROOT/raw/sponsor/google_maps.csv" ] && python -m etl.sim.places_pois

python -m sim.patient.run --households "$HH" --config configs/sim.yaml --max-memory 40G
python -m sim.patient.handcheck --config configs/sim.yaml
