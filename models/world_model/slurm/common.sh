#!/bin/bash
# Sourced by the sbatch scripts: stage the env + inputs to node-local /tmp (scratch is slow).
set -euo pipefail
SCR=/global/scratch/users/christophervu/dssg
REPO=${REPO:-$SCR/repo}                       # clone of the repo (branch with models/world_model)
DATA=${DATA:-$REPO/data/processed}            # stream A's outputs live here
INTERIM=${INTERIM:-$REPO/data/interim}
ENV_TGZ=${ENV_TGZ:-$SCR/envs/wm.tar.gz}       # built once by setup_env.sh
TMP=/tmp/wm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$TMP/env" "$TMP/data"
trap 'rm -rf "$TMP"' EXIT

tar -xzf "$ENV_TGZ" -C "$TMP/env"
set +u   # activate/conda-unpack read unset vars (CONDA_PREFIX)
source "$TMP/env/bin/activate"
conda-unpack
set -u

stage_inputs() {  # $1 = cutoff
  mkdir -p "$TMP/data/tokens"
  rsync -a "$DATA/tokens/cutoff=$1" "$TMP/data/tokens/"
  [ -f "$DATA/text_emb_$1.npy" ] && rsync -a "$DATA/text_emb_$1.npy" "$TMP/data/"
  [ -f "$DATA/probe_targets.parquet" ] && rsync -a "$DATA/probe_targets.parquet" "$TMP/data/"
  mkdir -p "$TMP/interim"   # event tables for the parcel next-year head
  for f in sales buyout_events eviction_events parcel_blockface; do
    [ -f "$INTERIM/$f.parquet" ] && rsync -a "$INTERIM/$f.parquet" "$TMP/interim/"
  done
  return 0
}

cd "$REPO"
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv || true
