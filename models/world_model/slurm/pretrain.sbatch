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
# Full-budget pretrain of the selected λ, one array task per cutoff (CUTOFFS="2016 2018" by default:
# dev model and test/backtest model; add 2025 for the forward map with --array=0-2).
# Reads λ from reports/wm/selected.json (sweep_select), or LAM=<value> to override.
read -r -a CUTOFFS <<< "${CUTOFFS:-2016 2018}"
CUTOFF=${CUTOFFS[$SLURM_ARRAY_TASK_ID]}
source "${SLURM_SUBMIT_DIR}/models/world_model/slurm/common.sh"
if [ -z "${LAM:-}" ]; then
  [ -f reports/wm/selected.json ] || python -m models.world_model.sweep_select --config configs/wm.yaml --cutoff "${SELECT_CUTOFF:-2016}" --rule "${RULE:-parcel}"
  LAM=$(python -c "import json;print(json.load(open('reports/wm/selected.json'))['lam'])")
fi
echo "cutoff=$CUTOFF lam=$LAM"
stage_inputs "$CUTOFF"
python -m models.world_model.pretrain --config configs/wm.yaml --cutoff "$CUTOFF" --lam "$LAM" --budget full \
  --data-root "$TMP/data" --ckpt-dir "$TMP/wm" --interim "$TMP/interim" --device cuda
mkdir -p "$DATA/wm"
rsync -a "$TMP/wm/" "$DATA/wm/"
