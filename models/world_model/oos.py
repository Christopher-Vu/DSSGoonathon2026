"""Out-of-time parcel next-year features for the risk harness.

Problem: a world model with cutoff T is trained on (t -> t+1) parcel targets for every t+1 < T, the same
outcomes a downstream GBM learns from. Its features on training rows are in-sample and look far more
predictive than they are at the scoring year T, so the GBM over-trusts them.

Fix (rolling origin): for each year c in [first, T], train a model with cutoff c (tokens from the
cutoff=T partition, years <= c) and keep only its year-c parcel predictions. A year-c row's features then
never saw any outcome after c. All of it is <= T, so it is legal inside the harness context for T.

  python -m models.world_model.oos run --decision 2016 --model-cutoff 2014 --lam 0.1
  python -m models.world_model.oos combine --decision 2016 --decision 2018
Writes data/processed/parcel_oos_embeddings.parquet: blklot, year, cutoff (= decision T), model_cutoff,
lam, pn_sale, pn_buyout, pn_bf_ellis_omi. Read with ctx.emb("parcel_oos").
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, run_dir
from .data import PNEXT, load_store
from .infer import check_store_matches, device_of, load_checkpoint, parcel_embeddings
from .pretrain import run as pretrain_run


def oos_cfg(cfg: dict, decision: int) -> dict:
    cfg = {**cfg, "paths": {**cfg["paths"]}}
    cfg["paths"]["tokens_from"] = decision
    cfg["paths"]["ckpt_dir"] = str(Path(cfg["paths"]["ckpt_dir"]).parent / "wm_oos" / f"T={decision}")
    return cfg


def run_one(cfg: dict, decision: int, c: int, lam: float, budget: str = "sweep", device=None) -> Path:
    assert c <= decision
    cfg = oos_cfg(cfg, decision)
    pretrain_run(cfg, c, lam, budget, device)
    dev = device_of(device)
    model, meta = load_checkpoint(run_dir(cfg, c, lam, budget) / "model.pt", cfg, dev)
    store = load_store(cfg, c)
    check_store_matches(store, meta)
    assert int(store.seq_year.max()) <= c
    blk = None
    if cfg["export"].get("parcel_filter"):
        blk = pd.read_parquet(cfg["export"]["parcel_filter"], columns=["blklot"])["blklot"].astype(str).unique().tolist()
    par = parcel_embeddings(model, store, cfg, dev, years=[c], blklots=blk)
    out = par[["blklot", "year", *[f"pn_{k}" for k in PNEXT]]].assign(
        cutoff=np.int16(decision), model_cutoff=np.int16(c), lam=np.float32(lam))
    p = Path(cfg["paths"]["ckpt_dir"]) / f"pn_{c}.parquet"
    out.to_parquet(p, index=False)
    print(f"T={decision} model_cutoff={c}: {len(out)} parcel rows -> {p}")
    return p


def combine(cfg: dict, decisions: list[int]) -> Path:
    parts = []
    for T in decisions:
        d = Path(oos_cfg(cfg, T)["paths"]["ckpt_dir"])
        for p in sorted(d.glob("pn_*.parquet")):
            f = pd.read_parquet(p)
            assert (f.year == f.model_cutoff).all() and (f.model_cutoff <= T).all()
            parts.append(f)
    df = pd.concat(parts, ignore_index=True)
    df["year"] = df["year"].astype(np.int16)
    out = Path(cfg["paths"]["data_root"]) / "parcel_oos_embeddings.parquet"
    df.to_parquet(out, index=False)
    print(df.groupby(["cutoff", "model_cutoff"]).size().to_string())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "combine"])
    ap.add_argument("--config", default="configs/wm.yaml")
    ap.add_argument("--decision", type=int, action="append", required=True)
    ap.add_argument("--model-cutoff", type=int)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--budget", default="sweep")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.cmd == "run":
        run_one(cfg, a.decision[0], a.model_cutoff, a.lam, a.budget, a.device)
    else:
        combine(cfg, a.decision)


if __name__ == "__main__":
    main()
