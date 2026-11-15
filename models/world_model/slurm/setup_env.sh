#!/bin/bash
# One-time: build the stream C env and pack it to $SCR/envs/wm.tar.gz (run on a login or interactive node).
set -euo pipefail
SCR=/global/scratch/users/christophervu/dssg
REPO=${REPO:-$SCR/repo}
module load anaconda3    # if the name differs: `module avail anaconda`
mkdir -p "$SCR/envs"
conda create -y -p "$SCR/envs/wm" python=3.11
source activate "$SCR/envs/wm" 2>/dev/null || conda activate "$SCR/envs/wm"
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r "$REPO/requirements-wm.txt" conda-pack
conda-pack -p "$SCR/envs/wm" -o "$SCR/envs/wm.tar.gz" --force
echo "packed $SCR/envs/wm.tar.gz"
