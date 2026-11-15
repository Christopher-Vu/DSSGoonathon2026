"""Run the world-model pipeline on Modal GPUs (alternative to the Savio sbatch chain).

Volume `wm-data` is laid out like the repo root and is the working directory of every function:
  configs/{wm.yaml,token_features.yaml}  data/processed/{tokens/cutoff=C, probe_targets, risk_dataset}
  data/interim/{sales,buyout_events,eviction_events,parcel_blockface}.parquet
Outputs land in the volume at data/processed/wm/, data/processed/*_embeddings.parquet, reports/wm/.

  modal run models/world_model/modal_app.py --cutoffs 2016,2018 --sweep-cutoff 2016
Leakage locks are the same code as on Savio (they live in models/world_model).
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

_here = Path(__file__).resolve()
REPO = _here.parents[2] if len(_here.parents) > 2 else Path("/root/repo")  # container: module sits at /root
app = modal.App("sf-world-model")
vol = modal.Volume.from_name("wm-data", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.3.1", "numpy==1.26.4", "pandas==2.2.2", "pyarrow==16.1.0", "scipy==1.13.1",
                 "scikit-learn==1.5.1", "pyyaml==6.0.1", "h3==4.1.0", "matplotlib==3.9.1", "folium==0.17.0")
    .env({"PYTHONPATH": "/root/repo", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(REPO / "models", "/root/repo/models", ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_file(REPO / "configs" / "wm.yaml", "/root/repo/configs/wm.yaml")
)
V = "/vol"
KW = dict(image=image, volumes={V: vol}, timeout=45 * 60)


def _sh(*args):
    os.chdir(V)
    os.makedirs("configs", exist_ok=True)
    subprocess.run(["cp", "/root/repo/configs/wm.yaml", "configs/wm.yaml"], check=True)
    print("+", " ".join(args), flush=True)
    subprocess.run(["python", "-m", *args], check=True)
    vol.commit()


@app.function(gpu="A10G", **KW)
def pretrain(cutoff: int, lam: float, budget: str):
    _sh("models.world_model.pretrain", "--config", "configs/wm.yaml", "--cutoff", str(cutoff), "--lam", str(lam),
        "--budget", budget, "--device", "cuda")
    return f"{cutoff}/{budget}/lam={lam:g}"


@app.function(cpu=4, memory=16384, **KW)
def select(cutoff: int, rule: str) -> float:
    vol.reload()
    _sh("models.world_model.sweep_select", "--config", "configs/wm.yaml", "--cutoff", str(cutoff), "--rule", rule)
    import json

    return float(json.load(open(f"{V}/reports/wm/selected.json"))["lam"])


@app.function(gpu="A10G", memory=32768, **KW)
def export(cutoffs: list[int], lam: float):
    vol.reload()
    _sh("models.world_model.export", "--config", "configs/wm.yaml", "--device", "cuda", "--lam", str(lam),
        "--cutoffs", *map(str, cutoffs))


@app.local_entrypoint()
def main(cutoffs: str = "2016,2018", sweep_cutoff: int = 2016, rule: str = "parcel", lam: float = -1.0,
         skip_sweep: bool = False):
    cs = [int(c) for c in cutoffs.split(",")]
    if lam < 0:
        lams = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
        if not skip_sweep:
            for r in pretrain.starmap([(sweep_cutoff, l, "sweep") for l in lams], return_exceptions=True):
                print("sweep:", r)
        lam = select.remote(sweep_cutoff, rule)
    print(f"selected lam = {lam:g}")
    for r in pretrain.starmap([(c, lam, "full") for c in cs], return_exceptions=True):
        print("full:", r)
    export.remote(cs, lam)
    print("done: modal volume get wm-data data/processed/parcel_embeddings.parquet ...")
