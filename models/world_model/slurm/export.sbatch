#!/bin/bash
#SBATCH --job-name=wm_export
#SBATCH --account=ic_cdss170fall
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_ica
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
# Export cell/parcel embeddings (both cutoffs) + UMAP maps. Reads checkpoints from $DATA/wm.
source "${SLURM_SUBMIT_DIR}/models/world_model/slurm/common.sh"
python -m models.world_model.export --config configs/wm.yaml --device cuda
