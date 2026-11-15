"""Export cell/parcel embeddings for the selected λ at both cutoffs (CONTRACTS §3) + UMAP maps.

  python -m models.world_model.export [--config configs/wm.yaml] [--lam 0.25] [--cutoffs 2018 2025]

Uses the `full` run for the selected λ when it exists, else the `sweep` run (with a warning).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, run_dir
from .data import PNEXT, load_store
from .infer import cell_embeddings, check_store_matches, device_of, load_checkpoint, parcel_embeddings
from .viz import render


def contract_frame(df: pd.DataFrame, key: str, cutoff: int, lam: float, d: int) -> pd.DataFrame:
    out = pd.DataFrame({key: df[key].astype(str).to_numpy(), "year": df["year"].astype(np.int16).to_numpy(),
                        "cutoff": np.int16(cutoff), "lam": np.float32(lam)})
    out["cutoff"] = out["cutoff"].astype(np.int16)
    out["lam"] = out["lam"].astype(np.float32)
    E = df[[f"e{i}" for i in range(d)]].to_numpy(np.float32)
    return pd.concat([out, pd.DataFrame(E, columns=[f"e{i}" for i in range(d)])], axis=1)


def selected_lam(cfg: dict) -> float:
    p = Path(cfg["paths"]["reports"]) / "selected.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run `python -m models.world_model.sweep_select` or pass --lam")
    with open(p) as f:
        return float(json.load(f)["lam"])


def find_run(cfg: dict, cutoff: int, lam: float) -> Path:
    for budget in ("full", "sweep"):
        p = run_dir(cfg, cutoff, lam, budget) / "model.pt"
        if p.exists():
            if budget == "sweep":
                print(f"warning: no full run for cutoff={cutoff} lam={lam:g}; exporting the sweep run")
            return p
    raise FileNotFoundError(f"no checkpoint for cutoff={cutoff} lam={lam:g}")


def export_cutoff(cfg: dict, cutoff: int, lam: float, dev, next_out: list | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    model, meta = load_checkpoint(find_run(cfg, cutoff, lam), cfg, dev)
    store = load_store(cfg, cutoff)
    check_store_matches(store, meta)
    E = cell_embeddings(model, store, cfg, dev)
    cells = pd.DataFrame(E, columns=[f"e{i}" for i in range(E.shape[1])])
    cells.insert(0, "year", store.seq_year)
    cells.insert(0, "cell", store.seq_cell)
    cell_df = contract_frame(cells, "cell", cutoff, lam, model.d)
    blk = None
    filt = cfg["export"].get("parcel_filter")
    if filt:
        blk = pd.read_parquet(filt, columns=["blklot"])["blklot"].astype(str).tolist()
    years = cfg["export"]["parcel_years"].get(cutoff)
    par = parcel_embeddings(model, store, cfg, dev, years=years, blklots=blk)
    par_df = contract_frame(par, "blklot", cutoff, lam, model.d) if len(par) else par
    if len(par) and next_out is not None:  # parcel next-year head outputs (expected t+1 counts)
        next_out.append(par[["blklot", "year", *[f"pn_{k}" for k in PNEXT]]].assign(
            cutoff=np.int16(cutoff), lam=np.float32(lam)))
    rep = Path(cfg["paths"]["reports"])
    uy = cfg["export"]["umap_year"].get(cutoff, int(store.seq_year.max()))
    sel = cell_df[cell_df["year"] == uy]
    suffix = "" if cutoff == int(cfg["backtest_cutoff"]) else f"_{cutoff}"
    how = render(sel["cell"].tolist(), sel[[f"e{i}" for i in range(model.d)]].to_numpy(), rep / f"umap_rgb_map{suffix}.png",
                 rep / f"umap_rgb_map{suffix}.html", title=f"SF World Model, cutoff={cutoff}, year={uy}, λ={lam:g}")
    print(f"cutoff={cutoff}: {len(cell_df)} cell rows, {len(par_df)} parcel rows, map via {how}")
    return cell_df, par_df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wm.yaml")
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--cutoffs", type=int, nargs="+", default=[2018, 2025])
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    lam = a.lam if a.lam is not None else selected_lam(cfg)
    dev = device_of(a.device)
    cells, parcels, nxt = [], [], []
    for c in a.cutoffs:
        cd, pd_ = export_cutoff(cfg, c, lam, dev, nxt)
        cells.append(cd)
        parcels.append(pd_)
    root = Path(cfg["paths"]["data_root"])
    pd.concat(cells, ignore_index=True).to_parquet(root / "cell_embeddings.parquet", compression="snappy", index=False)
    pd.concat(parcels, ignore_index=True).to_parquet(root / "parcel_embeddings.parquet", compression="snappy", index=False)
    if nxt:  # read by candidates as ctx.emb("parcel_next")
        pd.concat(nxt, ignore_index=True).to_parquet(root / "parcel_next_embeddings.parquet", index=False)
    print(f"wrote {root / 'cell_embeddings.parquet'} and {root / 'parcel_embeddings.parquet'}")


if __name__ == "__main__":
    main()
