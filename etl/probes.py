"""Probe targets per (cell, t) for t in {2016 (selection), 2018 (test only)}. CONTRACTS §2.

All targets use outcomes in years t+1..t+3 only; none is a token feature at year <= t.
Split by H3 res-7 parent (res-6 gives only ~8 parents over SF) (spatial block holdout), deterministic hash.
"""
import hashlib
from pathlib import Path

import h3
import numpy as np
import pandas as pd

INT = Path("data/interim")
OUT = Path("data/processed")
YEARS = [2016, 2018]
H = 3
MIN_RC_UNITS = 5


def split_of(cell):
    p = h3.cell_to_parent(cell, 7)
    v = int(hashlib.md5(p.encode()).hexdigest(), 16) % 100
    return "train" if v < 70 else ("val" if v < 85 else "test")


def build():
    sp = pd.read_parquet(INT / "spine.parquet", columns=["blklot", "year", "cell", "units", "rent_ctrl", "use_code"])
    ev = pd.read_parquet(INT / "eviction_events.parquet")
    bf = pd.read_parquet(INT / "blockfaces.parquet")[["bf", "cell"]]
    ev = ev[ev.bf_ok].merge(bf, on="bf")
    bo = pd.read_parquet(INT / "buyout_events.parquet").dropna(subset=["blklot"])
    sales = pd.read_parquet(INT / "sales.parquet")
    geo = sp.dropna(subset=["cell"]).drop_duplicates("blklot", keep="last")[["blklot", "cell"]]
    bo = bo.merge(geo, on="blklot").drop_duplicates(["case_number", "cell"])
    sales = sales.merge(geo, on="blklot")
    resi = sp[sp.use_code.isin(["MRES", "SRES", "MISC"])]

    rows = []
    for t in YEARS:
        s = sp[sp.year == t].dropna(subset=["cell"])
        rc = s[s.rent_ctrl == 1].groupby("cell").units.sum()
        nres = resi[resi.year == t].groupby("cell").blklot.nunique()
        w = lambda d, col="year": d[(d[col] > t) & (d[col] <= t + H)]
        e = w(ev)
        elo = e[e.ellis_omi].groupby("cell").size()
        npb = e[e.non_payment | e.breach | e.nuisance].groupby("cell").size()
        b = w(bo).groupby("cell").size()
        sl = w(sales, "sale_year")
        sl = sl[sl.blklot.isin(set(resi[resi.year == t].blklot))].groupby("cell").blklot.nunique()
        cells = rc[rc >= MIN_RC_UNITS].index
        per100 = 100 / rc.reindex(cells)
        tg = {
            "eviction_rate_3y": (np.log1p(elo.reindex(cells, fill_value=0) * per100), "regression"),
            "nonpayment_rate_3y": (np.log1p(npb.reindex(cells, fill_value=0) * per100), "regression"),
            "buyout_rate_3y": (np.log1p(b.reindex(cells, fill_value=0) * per100), "regression"),
            "any_ellis_omi_3y": ((elo.reindex(cells, fill_value=0) > 0).astype(float), "binary"),
            "sale_rate_3y": ((sl.reindex(cells, fill_value=0) / nres.reindex(cells)).clip(0, 1), "regression"),
        }
        for name, (v, task) in tg.items():
            v = v.dropna()
            rows.append(pd.DataFrame({"cell": v.index, "year": np.int16(t), "target": name,
                                      "value": v.values.astype("float32"), "task": task}))
    df = pd.concat(rows, ignore_index=True)
    acs = INT / "acs_cell.parquet"
    if acs.exists():  # static ACS 2014-18 targets, attached to both years
        a = pd.read_parquet(acs)
        for name in ["overcrowding", "median_income_log"]:
            for t in YEARS:
                df = pd.concat([df, pd.DataFrame({"cell": a.cell, "year": np.int16(t), "target": name,
                                                  "value": a[name].astype("float32"), "task": "regression"})])
    df["split"] = df.cell.map(split_of)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "probe_targets.parquet", index=False)
    return df.groupby(["year", "target"]).value.agg(["size", "mean"]), df.drop_duplicates("cell").split.value_counts()


if __name__ == "__main__":
    for x in build():
        print(x)
