"""Synthetic stand-ins for stream A's files (CONTRACTS §1-2), with planted structure.

Latent per cell:
  * neighborhood cluster c (spatially contiguous Voronoi regions over H3 cells)
  * trend tau ~ N(0,1)
Planted signal is deliberately hidden from per-type counts/means (the hand-made baseline):
  * cluster sets the sign of corr(f0, f1) among parcel tokens and a direction in permit text
  * tau sets corr(f3, f4) among parcel tokens, a text direction, and drives next-year drift of
    parcel f2 (so L_next must infer tau from year-t tokens)
Probe targets are functions of (c, tau) + noise, so a model that recovers the latents wins.

  python -m models.world_model.fixtures --out data/processed/fixture
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import N_FEAT, TEXT_DIM

PUBLIC = {
    "parcel": ["units", "age", "log_value", "years_since_sale", "lot_area"],
    "sale": ["value_jump", "years_since_prev"],
    "permit": ["log_cost", "unit_change", "is_capital_improvement"],
    "eviction_bf": ["n_ellis_lag", "n_omi_lag"],
    "buyout": ["n_tenants", "log_amount"],
}
SPONSOR = {
    "poi": ["rating", "n_reviews", "is_health"],
    "reddit": [],
    "review": ["rating"],
    "zillow": ["rent_estimate", "is_listed"],
}
TARGETS = [("eviction_rate_3y", "regression"), ("buyout_rate_3y", "regression"),
           ("overcrowding", "regression"), ("any_ellis_3y", "binary")]


def _cells(k: int):
    import h3

    center = h3.latlng_to_cell(37.765, -122.44, 9)
    return sorted(h3.grid_disk(center, k))


def make_world(k: int = 9, n_clusters: int = 4, seed: int = 0):
    import h3

    rng = np.random.default_rng(seed)
    cells = _cells(k)
    seeds = rng.choice(len(cells), n_clusters, replace=False)
    clus = np.array([np.argmin([h3.grid_distance(c, cells[s]) for s in seeds]) for c in cells])
    tau = rng.normal(size=len(cells)).astype(np.float32)
    dirs = rng.normal(size=(n_clusters + 1, TEXT_DIM))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return {"cells": cells, "clus": clus, "tau": tau, "dirs": dirs, "n_clusters": n_clusters, "seed": seed}


def _text(rng, w, ci, t, n):
    v = 0.8 * w["dirs"][ci] + 0.6 * np.tanh(t) * w["dirs"][-1] + 0.6 * rng.normal(size=(n, TEXT_DIM)) / np.sqrt(TEXT_DIM) * 2
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float16)


def make_tokens(w: dict, years: range, include_sponsor: bool, seed: int = 1):
    rng = np.random.default_rng(seed)
    rows, texts = [], []
    density = np.array([22, 30, 26, 34, 28, 24][: w["n_clusters"]])
    rho = np.array([0.85, -0.85, 0.85, -0.85, 0.85, -0.85][: w["n_clusters"]])
    for i, cell in enumerate(w["cells"]):
        c, t = int(w["clus"][i]), float(w["tau"][i])
        n_par = max(4, rng.poisson(density[c]))
        base = rng.normal(size=(n_par, N_FEAT)).astype(np.float32)
        pxy = rng.uniform(-1, 1, size=(n_par, 2)).astype(np.float32)
        for yr in years:
            dt = yr - years[0]

            def add(tt, f, xy, eid=None, tidx=-1):
                rows.append((cell, yr, tt, eid, *f, *xy, tidx))

            # parcels (all planted signals have ~zero cell mean, so counts/means can't see them):
            #   f1 = cluster-specific gradient over in-cell position (cross-token: MTM can learn it)
            #   f4 = tanh(tau) * saddle(x, y)                      (cross-token)
            #   corr(f0, f3) = rho_c                                (within-token)
            #   f2 drifts with tau over years                       (L_next must infer tau)
            z = rng.normal(size=(n_par, 5)).astype(np.float32)
            f = np.full((n_par, N_FEAT), np.nan, dtype=np.float32)
            x, y = pxy[:, 0], pxy[:, 1]
            grad = (x, -x, y, -y)[c % 4]
            f[:, 0] = base[:, 0] * 0.5 + z[:, 0]
            f[:, 1] = 1.5 * grad + 0.5 * z[:, 1]
            f[:, 2] = base[:, 2] * 0.5 + 0.25 * t * dt + 0.3 * z[:, 2]
            f[:, 3] = rho[c] * f[:, 0] + np.sqrt(1 - rho[c] ** 2) * z[:, 3]
            f[:, 4] = 3.0 * np.tanh(t) * x * y + 0.5 * z[:, 4]
            f[rng.random(n_par) < 0.05, 4] = np.nan
            for p in range(n_par):
                add("parcel", f[p], pxy[p], f"{i:04d}{p:03d}")
            n_sale = rng.poisson(1.0 + 0.5 * (c % 2))
            for _ in range(n_sale):
                ff = np.full(N_FEAT, np.nan, np.float32)
                ff[:2] = rng.normal(size=2)
                add("sale", ff, rng.uniform(-1, 1, 2), f"{i:04d}{rng.integers(n_par):03d}")
            n_perm = rng.poisson(2.0)
            if n_perm:
                tv = _text(rng, w, c, t, n_perm)
                for j in range(n_perm):
                    ff = np.full(N_FEAT, np.nan, np.float32)
                    ff[:3] = rng.normal(size=3)
                    texts.append(tv[j])
                    add("permit", ff, rng.uniform(-1, 1, 2), f"{i:04d}{rng.integers(n_par):03d}", len(texts) - 1)
            if rng.random() < 0.3:
                ff = np.full(N_FEAT, np.nan, np.float32)
                ff[:2] = rng.normal(size=2)
                add("eviction_bf", ff, rng.uniform(-1, 1, 2))
            if rng.random() < 0.15:
                ff = np.full(N_FEAT, np.nan, np.float32)
                ff[:2] = rng.normal(size=2)
                add("buyout", ff, rng.uniform(-1, 1, 2))
            if include_sponsor:
                for _ in range(rng.poisson(3 + c)):
                    ff = np.full(N_FEAT, np.nan, np.float32)
                    ff[:3] = rng.normal(size=3)
                    add("poi", ff, rng.uniform(-1, 1, 2))
                if yr >= 2025:  # reddit is 2025+ only, as in the real data
                    n_r = rng.poisson(2)
                    if n_r:
                        tv = _text(rng, w, c, t, n_r)
                        for j in range(n_r):
                            texts.append(tv[j])
                            add("reddit", np.full(N_FEAT, np.nan, np.float32), rng.uniform(-1, 1, 2), None, len(texts) - 1)
                if rng.random() < 0.4:
                    ff = np.full(N_FEAT, np.nan, np.float32)
                    ff[:2] = rng.normal(size=2)
                    add("zillow", ff, rng.uniform(-1, 1, 2), f"{i:04d}{rng.integers(n_par):03d}")
    cols = ["cell", "year", "token_type", "entity_id", *[f"f{j}" for j in range(N_FEAT)], "x_off", "y_off", "text_idx"]
    df = pd.DataFrame(rows, columns=cols)
    df["year"] = df["year"].astype("int16")
    for j in range(N_FEAT):
        df[f"f{j}"] = df[f"f{j}"].astype("float32")
    df["x_off"] = df["x_off"].astype("float32")
    df["y_off"] = df["y_off"].astype("float32")
    df["text_idx"] = df["text_idx"].astype("int32")
    emb = np.stack(texts).astype(np.float16) if texts else np.zeros((0, TEXT_DIM), np.float16)
    return df, emb


def make_probe_targets(w: dict, years=(2014, 2016, 2018), seed: int = 2) -> pd.DataFrame:
    import h3

    rng = np.random.default_rng(seed)
    nc = w["n_clusters"]
    eff = rng.normal(size=(3, nc)) * 1.0
    # spatial block holdout: fixture is small, so split by res-8 parent (real data: res-6)
    parents = sorted({h3.cell_to_parent(c, 8) for c in w["cells"]})
    order = rng.permutation(len(parents))
    split_of = {}
    for r, pi in enumerate(order):
        q = r / len(parents)
        split_of[parents[pi]] = "train" if q < 0.6 else ("val" if q < 0.8 else "test")
    rows = []
    for i, cell in enumerate(w["cells"]):
        c, t = int(w["clus"][i]), float(w["tau"][i])
        sp = split_of[h3.cell_to_parent(cell, 8)]
        for yr in years:
            vals = {
                "eviction_rate_3y": eff[0, c] + 0.7 * t + 0.3 * rng.normal(),
                "buyout_rate_3y": eff[1, c] - 0.5 * t + 0.3 * rng.normal(),
                "overcrowding": eff[2, c] + 0.3 * rng.normal(),
            }
            logit = 1.5 * t + 1.2 * eff[0, c] - 0.5
            vals["any_ellis_3y"] = float(rng.random() < 1 / (1 + np.exp(-logit)))
            for name, task in TARGETS:
                rows.append((cell, yr, name, vals[name], task, sp))
    df = pd.DataFrame(rows, columns=["cell", "year", "target", "value", "task", "split"])
    df["year"] = df["year"].astype("int16")
    df["value"] = df["value"].astype("float32")
    return df


def write_fixture(out: str | Path, k: int = 9, y0: int = 2010, seed: int = 0, cutoffs=(2018, 2025)) -> Path:
    out = Path(out)
    w = make_world(k=k, seed=seed)
    last = max(cutoffs)
    full_pub, emb_pub = make_tokens(w, range(y0, last + 1), include_sponsor=False, seed=seed + 1)
    full_sp, emb_sp = make_tokens(w, range(y0, last + 1), include_sponsor=True, seed=seed + 1)
    for cutoff in cutoffs:
        if cutoff <= 2018:
            df, emb = full_pub[full_pub["year"] <= cutoff], emb_pub
        else:
            df, emb = full_sp[full_sp["year"] <= cutoff], emb_sp
        d = out / "tokens" / f"cutoff={cutoff}"
        d.mkdir(parents=True, exist_ok=True)
        df.to_parquet(d / "part-0.parquet", compression="snappy", index=False)
        np.save(out / f"text_emb_{cutoff}.npy", emb)
    (out / "configs").mkdir(parents=True, exist_ok=True)
    with open(out / "configs" / "token_features.yaml", "w") as f:
        yaml.safe_dump({**PUBLIC, **SPONSOR}, f, sort_keys=False)
    make_probe_targets(w).to_parquet(out / "probe_targets.parquet", compression="snappy", index=False)
    lat = pd.DataFrame({"cell": w["cells"], "cluster": w["clus"], "tau": w["tau"]})
    lat.to_parquet(out / "fixture_latents.parquet", index=False)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/fixture")
    ap.add_argument("--k", type=int, default=12, help="H3 grid_disk radius (k=12 -> 469 cells)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    p = write_fixture(a.out, k=a.k, seed=a.seed)
    print(f"fixture written to {p}")


if __name__ == "__main__":
    main()
