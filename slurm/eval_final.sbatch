#!/bin/bash
# Locked world-model probe test (reads year=2018 test rows). Run once.
#SBATCH --job-name=wm_eval_final
#SBATCH --partition=savio3_htc
#SBATCH --account=ic_cdss170fall
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/global/scratch/users/christophervu/dssg/logs/eval_final-%j.out
S=/global/scratch/users/christophervu/dssg
export REPO=${REPO:-$S/repo-final}
source $REPO/models/world_model/slurm/common.sh
cd "$REPO"
python -m models.world_model.eval_final --final
ls -la reports/wm/
