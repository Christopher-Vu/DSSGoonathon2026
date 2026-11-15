#!/bin/bash
#SBATCH --job-name=wm_sweep
#SBATCH --account=ic_cdss170fall
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_ica
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x_%A_%a.out
# λ sweep for the cutoff=2018 checkpoint: 6 runs in parallel, one A5000 each, same seed and budget.
LAMS=(0 0.1 0.25 0.5 0.75 1.0)
LAM=${LAMS[$SLURM_ARRAY_TASK_ID]}
source "${SLURM_SUBMIT_DIR}/models/world_model/slurm/common.sh"
stage_inputs 2018
python -m models.world_model.pretrain --config configs/wm.yaml --cutoff 2018 --lam "$LAM" --budget sweep \
  --data-root "$TMP/data" --ckpt-dir "$TMP/wm" --device cuda
mkdir -p "$DATA/wm"
rsync -a "$TMP/wm/" "$DATA/wm/"
