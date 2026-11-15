# Start prompt — Stream C: SF World Model

You are building the **SF World Model** for the datathon project. Read `PRD.md` (especially §1 beat 2, §4.1–4.2, and §5) and `docs/CONTRACTS.md` before writing code. Stream A owns the data spine, the token files and Savio. Stream B owns the patient sim. You own only what's listed below.

## Goal
A **≤10M-parameter** transformer over H3 res-9 cells that learns what a cell *is* (masked token modeling) and how it *changes* (next-year prediction). It produces per-cell and per-parcel embeddings that the risk model consumes. Deck deliverables:
- a probe table: embeddings beat hand-made features on ≥3 targets the model never saw;
- a **λ-sweep curve**;
- a UMAP→RGB map of SF in which neighborhoods appear without labels.

## Rules
- **No heavy local compute.** The laptop froze once already. Anything beyond tiny fixture tests (seconds, <1 GB RAM) goes to Savio as an sbatch handed to stream A.
- **Git:**
  - work in your own worktree on branch `c-world-model`;
  - **commit frequently, with short curt messages** (e.g. `tokenizer`, `next-year head`), and **no Claude/Co-Authored-By attribution, ever**;
  - push the branch and open a PR to `main` at each milestone. Never push to `main`.
- **Your folders:** `models/world_model/`, `tests/world_model/`, `configs/wm*.yaml`, `requirements-wm.txt`. Don't edit other streams' folders. Propose contract changes in your PR description.
- **Data:** never commit data. Read inputs from `data/processed/`, write outputs to `data/processed/` and `reports/wm/`.
- **Savio:** do **not** SSH to Savio. Write `models/world_model/slurm/*.sbatch` and hand them to stream A with exact commands.
  - GPU target: **A5000**: `--account=ic_cdss170fall --partition=savio4_gpu --qos=a5k_gpu4_ica --gres=gpu:A5000:1 --cpus-per-task=4`.
  - Stage the env and data to node-local `/tmp` (scratch is slow). Scratch root: `/global/scratch/users/christophervu/dssg`.
- **Environment:** Python 3.11, PyTorch ≥2.3; pin dependencies in `requirements-wm.txt`. Code must also run on CPU for tests.
- **Work against fixtures until stream A's tokens exist.** Write `models/world_model/fixtures.py`, which generates synthetic `tokens/cutoff=2018` and `cutoff=2025` data, `text_emb_*.npy`, `configs/token_features.yaml` and `probe_targets.parquet` per CONTRACTS §1–2.
  - Plant structure, e.g. latent "neighborhood" clusters that drive token mixes and a latent trend that drives next-year changes, so a correct model provably beats baselines on the fixture.

## Model (per PRD §4.2)
- **Sequence** = one (cell, year). Prepend a learned `[CELL]` token. Cap at 512 tokens by stratified subsampling (keep every non-parcel token first). Bucket by length; use a padding mask.
- **Token embedding** = `type_emb[token_type]` + `MLP([f0..f15 with NaN→0, isnan mask])` + `Linear(text_emb)` if `text_idx ≥ 0` + `Fourier(x_off, y_off)` + `year_emb`.
  - Unknown types from the yaml must work without code changes.
- **Encoder:** pre-LN transformer, ~6 layers, d=256, 8 heads. Assert ≤10M parameters in a test.
- **Loss** `L = (1−λ)·L_MTM + λ·L_next`:
  - `L_MTM`: mask 15–30% of tokens (replace with `[MASK]` but keep position and year); predict the type (cross-entropy) + features (MSE on non-NaN entries only).
  - `L_next`: from the year-t `[CELL]` output, an MLP head predicts the cell's **year-t+1 summary**, computed by you from the t+1 tokens:
    - per-type token counts (log-rate head, Poisson NLL);
    - per-type mean of each standardized feature (MSE, masked where the type is absent).

    Only pairs where both years are ≤ the cutoff are used.
- **Training:** AdamW, lr 3e-4, cosine schedule with warmup, bf16 on GPU, early stopping on the validation loss. Seed everything. Set the budget so one run takes ≲10 minutes on an A5000.

## Splits, selection, leakage (judges check this)
- **Checkpoints:**
  - `cutoff=2018`: tokens ≤2018 only, **no sponsor types**. Assert this in a test and at load time.
  - `cutoff=2025`: all types, used for the forward-looking map.
- **L_next validation** = the `(2017→2018)` pairs, held out from training for the 2018 checkpoint.
- **Probes:** frozen embeddings + a ridge (regression) or logistic (binary) model. Fit on `split=train`; tune α on `split=val`.
  - Comparisons: (a) hand-made features only (provided as a probe-feature table, or computed by you from parcel tokens: counts, means), (b) embeddings only, (c) both.
  - Report R²/PR-AUC with bootstrap CIs.
- **λ sweep:** λ ∈ {0, 0.1, 0.25, 0.5, 0.75, 1.0} as one Slurm array (6 × A5000 in parallel).
  - Selection uses **only** `year=2016` probe rows + L_next val error. Write the rule into the code.
  - **`year=2018` (test) probe rows are read only by `eval_final.py --final`.** Make reading them without the flag impossible.
- **Ablations:** λ sweep (required); resolution 9 vs. 10 (optional, fixture-ready only).

## Outputs
- `cell_embeddings.parquet` and `parcel_embeddings.parquet` per CONTRACTS §3, for the selected λ at both cutoffs.
- `reports/wm/`:
  - `lambda_sweep.csv` + plot (probe lift and next-year error vs. λ);
  - `probes.csv` (val; test only via `--final`);
  - `umap_rgb_map.png` (cells colored by 3-D UMAP of the embeddings; a folium/pydeck HTML version too).
- Entry points:
  - `python -m models.world_model.pretrain --config configs/wm.yaml --cutoff 2018 --lam 0.25`
  - `python -m models.world_model.sweep_select`
  - `python -m models.world_model.export`
  - `python -m models.world_model.eval_final --final`
- `models/world_model/slurm/sweep.sbatch` (array 0–5) + `pretrain.sbatch`, with a README section giving exact submit commands for stream A.

## Tests (pytest, CPU, <2 minutes total)
- param count ≤10M;
- forward/backward pass on the fixture;
- masking never masks `[CELL]`;
- `L_next` targets are built only from year t+1 tokens;
- the leakage asserts (cutoff year, no sponsor types in 2018, test rows locked);
- export matches the contract schema;
- **on the fixture, embeddings beat hand-made features on the planted targets** (sanity check that learning happens).

## Done when
Everything above runs end to end on the fixture on CPU, the sbatch scripts are ready, and the PR is open.

**Report back:** branch, PR link, fixture results (probe lift, sweep curve), param count, expected A5000 minutes per run, and anything you need from stream A (token feature list, probe targets, hand-made probe features).
