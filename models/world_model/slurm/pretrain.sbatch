#!/bin/bash
#SBATCH --job-name=wm_pretrain
#SBATCH --account=ic_cdss170fall
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_ica
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%A_%a.out
# Full-budget pretrain of the selected λ: task 0 = cutoff 2018 (backtest), task 1 = cutoff 2025 (forward map).
# Reads λ from reports/wm/selected.json (sweep_select), or LAM=<value> to override.
CUTOFFS=(2018 2025)
CUTOFF=${CUTOFFS[$SLURM_ARRAY_TASK_ID]}
source "${SLURM_SUBMIT_DIR}/models/world_model/slurm/common.sh"
if [ -z "${LAM:-}" ]; then
  [ -f reports/wm/selected.json ] || python -m models.world_model.sweep_select --config configs/wm.yaml
  LAM=$(python -c "import json;print(json.load(open('reports/wm/selected.json'))['lam'])")
fi
echo "cutoff=$CUTOFF lam=$LAM"
stage_inputs "$CUTOFF"
python -m models.world_model.pretrain --config configs/wm.yaml --cutoff "$CUTOFF" --lam "$LAM" --budget full \
  --data-root "$TMP/data" --ckpt-dir "$TMP/wm" --device cuda
mkdir -p "$DATA/wm"
rsync -a "$TMP/wm/" "$DATA/wm/"
