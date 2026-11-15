#!/bin/bash
# one-time: venv on scratch. run on a login node: bash slurm/setup_env.sh
set -e
R=/global/scratch/users/christophervu/dssg
module load python/3.11.6-gcc-11.4.0
python3 -m venv $R/venv
source $R/venv/bin/activate
pip install -q --upgrade pip
pip install -q -r $R/repo/requirements.txt
python -c "import lightgbm, pulp; print('ok', lightgbm.__version__, pulp.__version__)"
