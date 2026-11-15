"""Building x snapshot-year design matrix for the displacement-risk model.

Universe at t: rent-controlled proxy parcels (residential, >=2 units, built <1979), geocoded.
Labels (horizon H=3, years t+1..t+3):
  y_bldg = any buyout disclosure/agreement at the parcel
  y_bf   = any Ellis or owner-move-in notice on the parcel's block-face
Time discipline for the t=2018 decision: train t<=2014, calibrate t=2015 (labels end 2018), test t=2018.
Features use only data dated <= t.
"""
from pathlib import Path

import numpy as np
import pandas as pd

INT = Path("data/interim")
OUT = Path("data/processed")
H = 3
SNAPSHOTS = list(range(2009, 2022))


def _window_count(events, key, year_col, t, lo, hi, name):
    e = events[(events[year_col] > t - lo) & (events[year_col] <= t - hi)] if lo else None
    return e.groupby(key).size().rename(name)


def build(snapshots=SNAPSHOTS):
    sp = pd.read_parquet(INT / "spine.parquet")
    pbf = pd.read_parquet(INT / "parcel_blockface.parquet").drop_duplicates("blklot")
    ev = pd.read_parquet(INT / "eviction_events.parquet")
    ev = ev[ev.bf_ok]
    bo = pd.read_parquet(INT / "buyout_events.parquet").dropna(subset=["blklot", "year"])
    sales = pd.read_parquet(INT / "sales.parquet")
    per = pd.read_parquet(Path("data/raw/datasf/permits.parquet"),
                          columns=["block", "lot", "filed_date", "permit_creation_date", "estimated_cost",
                                   "revised_cost", "existing_units", "proposed_units", "description"])
    per["blklot"] = per.block.astype(str).str.zfill(4) + per.lot.astype(str).str.zfill(3)
    per["year"] = pd.to_datetime(per.filed_date.fillna(per.permit_creation_date), errors="coerce").dt.year
    per["cost"] = pd.to_numeric(per.revised_cost, errors="coerce").fillna(pd.to_numeric(per.estimated_cost, errors="coerce"))
    per["unit_removal"] = pd.to_numeric(per.proposed_units, errors="coerce") < pd.to_numeric(per.existing_units, errors="coerce")
    per["remodel"] = per.description.fillna("").str.lower().str.contains("kitchen|bath|remodel|renovat")
    ev = ev.astype({"year": "float"})
    bo = bo.astype({"year": "float"})

    # rent-controlled units per block-face (for unit-share apportioning)
    rows = []
    for t in snapshots:
        s = sp[(sp.year == t) & (sp.rent_ctrl == 1) & sp.lat.notna()].merge(pbf, on="blklot", how="left")
        s["t"] = t
        bf_units = s.groupby("bf").units.transform("sum")
        s["bf_share"] = s.units / bf_units
        s["bf_rc_units"] = bf_units
        s["sold_2y"] = (s.yrs_since_sale <= 2).astype("int8")
        # past events (<= t)
        def cnt(df, key, lo, mask=None, name=None):
            d = df if mask is None else df[mask]
            d = d[(d.year > t - lo) & (d.year <= t)]
            return d.groupby(key).size().rename(name)
        feats = [
            cnt(ev, "bf", 3, ev.ellis_omi, "bf_ellis_omi_3y"), cnt(ev, "bf", 10, ev.ellis_omi, "bf_ellis_omi_10y"),
            cnt(ev, "bf", 3, ~ev.ellis_omi, "bf_other_evict_3y"),
            cnt(bo, "blklot", 3, None, "bldg_buyout_3y"),
            cnt(per, "blklot", 3, None, "permits_3y"), cnt(per, "blklot", 3, per.remodel, "remodel_3y"),
            cnt(per, "blklot", 10, per.unit_removal, "unit_removal_10y"),
        ]
        for f in feats:
            key = "bf" if f.name.startswith("bf_") else "blklot"
            s = s.merge(f, left_on=key, right_index=True, how="left")
            s[f.name] = s[f.name].fillna(0)
        bf_bo = bo[(bo.year > t - 3) & (bo.year <= t)].merge(pbf, on="blklot").groupby("bf").size().rename("bf_buyout_3y")
        s = s.merge(bf_bo, left_on="bf", right_index=True, how="left").fillna({"bf_buyout_3y": 0})
        pc = per[(per.year > t - 3) & (per.year <= t)].groupby("blklot").cost.sum().rename("permit_cost_3y")
        s = s.merge(pc, left_on="blklot", right_index=True, how="left").fillna({"permit_cost_3y": 0})
        # cell context: sales turnover of resi parcels past 3y
        cs = sales[(sales.sale_year > t - 3) & (sales.sale_year <= t)].merge(sp[sp.year == t][["blklot", "cell"]], on="blklot")
        s = s.merge(cs.groupby("cell").size().rename("cell_sales_3y"), left_on="cell", right_index=True, how="left")
        s["cell_sales_3y"] = s.cell_sales_3y.fillna(0)
        # labels (t+1..t+H); NaN-free but mark observability
        fb = bo[(bo.year > t) & (bo.year <= t + H)]
        fe = ev[ev.ellis_omi & (ev.year > t) & (ev.year <= t + H)]
        s["y_bldg"] = s.blklot.isin(set(fb.blklot)).astype("int8")
        s["y_bf"] = s.bf.isin(set(fe.bf)).astype("int8")
        s["label_end"] = t + H
        rows.append(s)
    d = pd.concat(rows, ignore_index=True)
    d["log_assessed_per_unit"] = np.log1p(d.assessed / d.units.clip(lower=1))
    d["log_permit_cost_3y"] = np.log1p(d.permit_cost_3y.clip(lower=0))
    d["log_last_jump"] = np.log(d.last_jump.clip(0.05, 50))
    d["size_band"] = pd.cut(d.units, [1, 4, 9, 25, 1e9], labels=["2-4", "5-9", "10-25", "26+"]).astype(str)
    keep = ["blklot", "t", "cell", "bf", "lat", "lon", "supervisor_district", "analysis_neighborhood", "size_band",
            "units", "age", "stories" if "stories" in d else "number_of_stories", "owner_occ", "yrs_since_sale",
            "n_sales_10y", "log_last_jump", "sold_2y", "log_assessed_per_unit", "assessed", "bf_share", "bf_rc_units",
            "bf_ellis_omi_3y", "bf_ellis_omi_10y", "bf_other_evict_3y", "bf_buyout_3y", "bldg_buyout_3y",
            "permits_3y", "remodel_3y", "unit_removal_10y", "log_permit_cost_3y", "cell_sales_3y",
            "y_bldg", "y_bf", "label_end"]
    d = d[keep]
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / "risk_dataset.parquet", index=False)
    return d


if __name__ == "__main__":
    d = build()
    print(d.groupby("t")[["y_bldg", "y_bf"]].mean().round(3).assign(n=d.groupby("t").size()))
