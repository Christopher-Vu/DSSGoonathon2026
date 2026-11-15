#!/usr/bin/env bash
# End-to-end fixture run on CPU (~10 min): fixture -> λ sweep -> select -> full pretrain x2 -> export -> final eval.
set -euo pipefail
CFG=configs/wm_fixture.yaml
python -m models.world_model.fixtures --out data/processed/fixture --k 12
mkdir -p reports/wm/fixture/logs
for l in 0 0.1 0.25 0.5 0.75 1.0; do
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m models.world_model.pretrain --config $CFG --cutoff 2018 --lam $l --budget sweep \
    > reports/wm/fixture/logs/sweep_$l.log 2>&1 &
done
wait
python -m models.world_model.sweep_select --config $CFG
LAM=$(python -c "import json;print(json.load(open('reports/wm/fixture/selected.json'))['lam'])")
for c in 2018 2025; do
  OMP_NUM_THREADS=4 python -m models.world_model.pretrain --config $CFG --cutoff $c --lam $LAM --budget full \
    > reports/wm/fixture/logs/full_$c.log 2>&1 &
done
wait
python -m models.world_model.export --config $CFG
python -m models.world_model.eval_final --config $CFG --final
