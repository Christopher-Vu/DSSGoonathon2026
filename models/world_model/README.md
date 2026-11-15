# SF World Model (stream C)

A transformer with 5.08M parameters over H3 res-9 cells. There is one sequence per (cell, year): a
`[CELL]` token followed by that cell's tokens (parcels, permits, evictions, text, ...). The loss is

    L = (1-λ)·L_MTM + λ·L_next

- **L_MTM** masks 15–30% of tokens and scores cross-entropy on the token type plus MSE on the true
  type's features.
- **L_next** predicts the year t+1 summary of the cell from the `[CELL]` output: Poisson NLL on
  per-type counts and MSE on per-type feature means.

λ is chosen by a 6-run sweep. Embeddings are judged by frozen linear probes against hand-made features.

## Leakage locks (enforced in code, covered by `tests/world_model/test_leakage.py`)
- The cutoff=2018 model never sees year > 2018 tokens or any sponsor-only token type. This is checked
  when tokens are read, when the vocab is built, and when a checkpoint is loaded.
- L_next trains only on pairs with t+1 < cutoff. Its validation pairs are (cutoff-1 → cutoff).
- Probe targets for year 2018 and for `split=test` are filtered out inside the parquet scan. Only
  `eval_final --final` can lift that filter.
- λ selection uses only year-2016 `split=val` rows plus the frozen next-year probe. The rule is
  written out in `sweep_select.py`.

## Entry points
| command | does |
|---|---|
| `python -m models.world_model.pretrain --cutoff C --lam λ --budget {sweep,full}` | trains one run and writes `model.pt`, `metrics.json`, `cell_emb.parquet`, `probes_val.csv` |
| `python -m models.world_model.sweep_select` | writes `reports/wm/lambda_sweep.csv/.png`, `probes.csv`, `selected.json` |
| `python -m models.world_model.export` | writes `data/processed/{cell,parcel}_embeddings.parquet` and `reports/wm/umap_rgb_map{,_2025}.{png,html}` |
| `python -m models.world_model.eval_final --final` | runs the year-2018 test-split probes **once** and writes `reports/wm/probes_test*.{csv,json}` |

## Savio (stream A runs these; nothing here runs on a laptop)
```bash
SCR=/global/scratch/users/christophervu/dssg
cd $SCR/repo && git fetch && git checkout c-world-model   # or main after merge
bash models/world_model/slurm/setup_env.sh                # once: builds + packs $SCR/envs/wm.tar.gz
mkdir -p logs

# optional synthetic check first (~15 min, 1 A5000, no real data needed)
sbatch models/world_model/slurm/fixture.sbatch

# real run (inputs: data/processed/tokens/cutoff={2018,2025}, text_emb_{C}.npy, probe_targets.parquet,
#           configs/token_features.yaml)
J1=$(sbatch --parsable models/world_model/slurm/sweep.sbatch)                          # 6 × A5000, ≤20 min
J2=$(sbatch --parsable --dependency=afterok:$J1 models/world_model/slurm/pretrain.sbatch)  # selects λ, 2 × A5000
sbatch --dependency=afterok:$J2 models/world_model/slurm/export.sbatch
# check reports/wm/lambda_sweep.png + selected.json, then once, manually:
python -m models.world_model.eval_final --config configs/wm.yaml --final    # in the wm env, CPU is fine
```
Set `LAM=0.25 sbatch ...pretrain.sbatch` to override the selected λ. Each job stages the env and its
inputs to `/tmp/wm_<job>_<task>` and syncs the checkpoints back to `data/processed/wm/`.

**Budget:** each sweep run is about 6–8 A5000 minutes, with training capped at 9 min. Each full run is
also capped at 9 min of training. The whole pipeline is about 1 GPU-hour.

## Outputs (contract)
- `cell_embeddings.parquet`: `cell, year:int16, cutoff:int16, lam:float32, e0..e255:float32`
- `parcel_embeddings.parquet`: the same columns, with `blklot` in place of `cell`. Cutoff 2018 covers
  years 2014–2018; cutoff 2025 covers 2025.

## Tests
`tests/world_model` uses a synthetic fixture with planted structure and finishes in about 1 min. It
trains small models, so run it on Savio (`fixture.sbatch`), not on a laptop.
