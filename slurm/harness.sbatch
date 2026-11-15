#!/bin/bash
#SBATCH --job-name=harness
#SBATCH --account=ic_cdss170fall
#SBATCH --partition=savio3_htc
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/global/scratch/users/christophervu/dssg/logs/%x-%j.out
# sbatch slurm/harness.sbatch <cand[,cand2]> [dev|test --final]
# REPO env var = your code dir (default: shared repo); data always comes from the shared repo.
set -e
S=/global/scratch/users/christophervu/dssg
REPO=${REPO:-$S/repo}
module load python/3.11.6-gcc-11.4.0
source $S/venv/bin/activate
cd $REPO
[ -e data ] || ln -s $S/repo/data data
python -m models.risk.harness --cand "$1" --split "${2:-dev}" ${3:-}
