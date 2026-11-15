#!/bin/bash
# Site export on the test split (frozen c_wm_oos_blend) + WM picks -> displaced_households for the sim.
#   COMMIT=<sha> sbatch slurm/final.sbatch      (run from $S/repo-final)
#SBATCH --job-name=final_export
#SBATCH --partition=savio3_htc
#SBATCH --account=ic_cdss170fall
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/global/scratch/users/christophervu/dssg/logs/final-%j.out
set -euo pipefail
S=/global/scratch/users/christophervu/dssg
cd "${REPO:-$S/repo-final}"
module load python/3.11.6-gcc-11.4.0
source $S/venv/bin/activate
python -m models.demo.export risk
python -m etl.displaced --tag wm
cp data/processed/displaced_households.parquet $S/data/processed/displaced_households.parquet
python -m models.demo.export wm
echo "done $(date)"
