# Raw data -> reported numbers -> site. Every heavy step runs on Savio (sbatch) or Modal, not a laptop.
# Seeds and configs live in configs/; the leak-proof harness is models/risk/harness.py.
#   make all            full pipeline (run on a Savio login node; targets call python directly, wrap in sbatch for heavy ones)
#   make site           export JSON + build web/ (needs the exported numbers)
PY ?= python
FINAL ?=            # set FINAL=--final to run the logged, locked test evaluations

.PHONY: all data pretrain risk backtest sim site test

all: data pretrain risk backtest sim site

data:
	$(PY) -m etl.datasf
	$(PY) -m etl.geo
	$(PY) -m etl.spine
	$(PY) -m etl.labels
	$(PY) -m etl.acs
	$(PY) -m etl.tokens --cutoff 2016
	$(PY) -m etl.tokens --cutoff 2018
	$(PY) -m etl.tokens --cutoff 2025
	$(PY) -m etl.probes
	$(PY) -m models.risk.dataset
	$(PY) -m etl.sim.pums_extract
	$(PY) -m etl.sim.blocks_extract
	$(PY) -m etl.sim.gtfs_clean
	$(PY) -m etl.sim.osm_region
	$(PY) -m etl.sim.places_pois

# λ sweep on cutoff 2016 -> select λ -> full runs (2016, 2018) -> export embeddings + rolling-origin features (GPU)
pretrain:
	modal run models/world_model/modal_app.py
	modal run models/world_model/modal_app.py::oos_main

# dev harness for the headline candidate; FINAL=--final adds the logged test call + locked probe test
risk:
	$(PY) -m models.risk.harness --cand c_wm_oos_blend --split dev
	$(if $(FINAL),$(PY) -m models.risk.harness --cand c_wm_oos_blend --split test --final)
	$(if $(FINAL),$(PY) -m models.world_model.eval_final --final)

backtest:
	$(PY) -m models.demo.export risk
	$(PY) -m etl.displaced --tag wm

sim:
	sbatch sim/patient/slurm/sim_full.sbatch $(CURDIR)/data/processed/displaced_households.parquet

site:
	$(PY) -m models.demo.export wm
	$(PY) -m models.demo.export sim
	cd web && npm ci && npm run build

test:
	$(PY) -m pytest -q tests
