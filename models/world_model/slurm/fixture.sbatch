#!/bin/bash
#SBATCH --job-name=wm_fixture
#SBATCH --account=ic_cdss170fall
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_ica
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
# Synthetic end-to-end check (no real data): fixture -> sweep -> select -> full x2 -> export -> final eval,
# plus the test suite. Outputs land in $REPO/reports/wm/fixture.
source "${SLURM_SUBMIT_DIR}/models/world_model/slurm/common.sh"
python -m pytest tests/world_model -q
bash models/world_model/run_fixture.sh
