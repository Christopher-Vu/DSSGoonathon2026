#!/bin/bash
#SBATCH --job-name=risk
#SBATCH --account=ic_cdss170fall
#SBATCH --partition=savio3_htc
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/global/scratch/users/christophervu/dssg/logs/%x-%j.out
# sbatch slurm/risk.sbatch [extra steps: probes|dataset]
set -e
R=/global/scratch/users/christophervu/dssg
module load anaconda3 2>/dev/null || module load python 2>/dev/null || true
source $R/venv/bin/activate
cd $R/repo
for s in "$@"; do
  case $s in
    probes) python -m etl.probes ;;
    dataset) python -m models.risk.dataset ;;
  esac
done
python -m models.risk.train
python -m models.risk.backtest --tag gbm
