"""Address -> blklot index, parcel centroids, block-face table."""
from pathlib import Path

import h3
import numpy as np
import pandas as pd

from etl.addr import bare, blockface, norm_street, split_addr

RAW = Path("data/raw/datasf")
INT = Path("data/interim")


def parcels() -> pd.DataFrame:
    p = pd.read_parquet(RAW / "parcels.parquet")
    p["lat"] = p.centroid_latitude.astype(float)
    p["lon"] = p.centroid_longitude.astype(float)
    p["block"] = p.blklot.str[:4]
    p["street"] = [norm_street(f"{a} {b}") if isinstance(a, str) else None
                   for a, b in zip(p.street_name, p.street_type.fillna(""))]
    p["lo"] = pd.to_numeric(p.from_address_num, errors="coerce")
    p["hi"] = pd.to_numeric(p.to_address_num, errors="coerce")
    p["cell"] = [h3.latlng_to_cell(a, b, 9) if a == a else None for a, b in zip(p.lat, p.lon)]
    return p


def address_index(p: pd.DataFrame, roll: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rows (street, num, blklot). Ranges expanded on same parity, capped at 60 numbers."""
    src = [p[["blklot", "street", "lo", "hi"]]]
    if roll is not None:
        src.append(roll[["blklot", "street", "lo", "hi"]])
    a = pd.concat(src).dropna().drop_duplicates()
    a = a[(a.hi >= a.lo) & (a.hi - a.lo <= 120) & (a.lo > 0)]
    rows = []
    for b, s, lo, hi in a.itertuples(index=False):
        for n in range(int(lo), int(hi) + 1, 2 if (hi - lo) % 2 == 0 else 1):
            rows.append((s, n, b))
    return pd.DataFrame(rows, columns=["street", "num", "blklot"]).drop_duplicates()


def match_addresses(addrs: pd.Series, idx: pd.DataFrame) -> pd.DataFrame:
    """addrs -> (addr, num, street, blklot list). Multiple blklots = condo/multi-lot building."""
    sp = [split_addr(a) for a in addrs]
    df = pd.DataFrame({"addr": addrs.values, "num": [x[0] for x in sp], "street": [x[1] for x in sp]})
    g = idx.groupby(["street", "num"]).blklot.agg(list).reset_index()
    out = df.merge(g, on=["street", "num"], how="left")
    # fallback: suffix-less street name, only if it maps to one street
    i2 = idx.assign(b=idx.street.map(bare))
    uniq = i2.groupby("b").street.nunique()
    i2 = i2[i2.b.map(uniq) == 1].groupby(["b", "num"]).blklot.agg(list)
    miss = out.blklot.isna() & out.street.notna()
    keys = list(zip(out.loc[miss, "street"].map(bare), out.loc[miss, "num"]))
    out.loc[miss, "blklot"] = pd.Series([i2.get(k) for k in keys], index=out.index[miss], dtype=object)
    out["how"] = np.where(out.blklot.notna(), np.where(miss, "bare", "exact"), None)
    # fallback: nearest same-parity number on the same street + hundred block, within 8
    by_st = {s: g.sort_values("num") for s, g in idx.groupby("street")}
    miss = out.blklot.isna() & out.street.notna()
    for i in out.index[miss]:
        s, n = out.at[i, "street"], out.at[i, "num"]
        g = by_st.get(s)
        if g is None:
            continue
        c = g[(g.num % 2 == n % 2) & (g.num // 100 == n // 100) & ((g.num - n).abs() <= 8)]
        if len(c):
            best = c.loc[(c.num - n).abs() == (c.num - n).abs().min()]
            out.at[i, "blklot"] = sorted(set(best.blklot))
            out.at[i, "how"] = "near"
    return out


def blockfaces(p: pd.DataFrame) -> pd.DataFrame:
    """Block-face (street|hundred) -> centroid, cell, parcel list (from the parcel's low number)."""
    q = p.dropna(subset=["street", "lo", "lat"]).copy()
    q = q[q.lo > 0]
    q["bf"] = [blockface(n, s) for n, s in zip(q.lo, q.street)]
    q["bfb"] = [f"{bare(s)}|{int(n) // 100}" for n, s in zip(q.lo, q.street)]
    g = q.groupby("bf").agg(lat=("lat", "mean"), lon=("lon", "mean"), n_parcels=("blklot", "nunique"))
    g["cell"] = [h3.latlng_to_cell(a, b, 9) for a, b in zip(g.lat, g.lon)]
    return g.reset_index(), q[["blklot", "bf"]].drop_duplicates()
