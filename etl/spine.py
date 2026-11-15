"""Parcel x year spine from the assessor rolls + sale events.

Time rule: roll year Y (lien date Jan 1 Y) is treated as known at the end of year Y.
Features at snapshot t use only rolls <= t and sale dates <= Dec 31 t.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from etl.addr import norm_street
from etl.geo import parcels

RAW = Path("data/raw/datasf")
INT = Path("data/interim")
_LOC = re.compile(r"^(\d+)\s+(\d+)\s+(.*?)\s*(\d{4})?$")
NUM = ["year_property_built", "number_of_bedrooms", "number_of_rooms", "number_of_stories", "number_of_units",
       "property_area", "lot_area", "percent_of_ownership", "homeowner_exemption_value",
       "assessed_improvement_value", "assessed_land_value"]


def parse_loc(s):
    """'2342 2332 TAYLOR              ST0000' -> (2332, 2342, 'TAYLOR ST')."""
    if not isinstance(s, str):
        return None, None, None
    m = _LOC.match(s.strip())
    if not m:
        return None, None, None
    a, b = int(m.group(1)), int(m.group(2))
    lo, hi = (min(a, b), max(a, b)) if a and b else (a or b, a or b)
    st = norm_street(re.sub(r"\s+", " ", m.group(3)))
    return lo, hi, st


def load_roll() -> pd.DataFrame:
    r = pd.read_parquet(RAW / "taxroll.parquet")
    r["year"] = r.closed_roll_year.astype(int).astype("int16")
    r["blklot"] = r.parcel_number.str.strip()
    for c in NUM:
        r[c] = pd.to_numeric(r[c], errors="coerce")
    r["sale_date"] = pd.to_datetime(r.current_sales_date, errors="coerce")
    r.loc[r.sale_date.dt.year < 1900, "sale_date"] = pd.NaT
    loc = [parse_loc(s) for s in r.property_location]
    r["lo"], r["hi"], r["street"] = zip(*loc)
    r["assessed"] = r.assessed_land_value.fillna(0) + r.assessed_improvement_value.fillna(0)
    return r.drop_duplicates(["blklot", "year"])


def sales(r: pd.DataFrame) -> pd.DataFrame:
    """One row per (blklot, sale_date) seen in any roll. Jump = assessed after / assessed before."""
    r = r.sort_values(["blklot", "year"])
    s = r.dropna(subset=["sale_date"]).groupby(["blklot", "sale_date"]).year.min().rename("first_roll").reset_index()
    ass = r.set_index(["blklot", "year"]).assessed
    sy = s.sale_date.dt.year
    before = ass.reindex(list(zip(s.blklot, sy))).values          # roll Y: lien Jan 1 Y, pre-sale
    after = ass.reindex(list(zip(s.blklot, sy + 1))).values        # roll Y+1 reassessed
    s["assessed_before"], s["assessed_after"] = before, after
    s["jump"] = np.where(before > 0, after / before, np.nan)
    s["sale_year"] = sy.astype("int16")
    return s


def spine(r: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    p = parcels()[["blklot", "lat", "lon", "cell", "supervisor_district", "analysis_neighborhood"]]
    p = p.drop_duplicates("blklot")
    sp = r[["blklot", "year", "use_code", "property_class_code", "lo", "hi", "street", "assessed",
            "assessor_neighborhood" if "assessor_neighborhood" in r else "analysis_neighborhood"] + NUM].copy()
    sp = sp.merge(p, on="blklot", how="left", suffixes=("", "_p"))
    sp["units"] = sp.number_of_units.fillna(0)
    sp["age"] = np.where(sp.year_property_built > 1800, sp.year - sp.year_property_built, np.nan)
    sp["owner_occ"] = (sp.homeowner_exemption_value.fillna(0) > 0).astype("int8")
    resi = sp.use_code.isin(["MRES", "SRES", "MISC"])
    sp["rent_ctrl"] = (resi & (sp.units >= 2) & (sp.year_property_built < 1979)).astype("int8")
    # sale history as of each snapshot year (sale dates <= Dec 31 of year)
    s2 = s[["blklot", "sale_year", "jump"]].sort_values("sale_year")
    sp = sp.sort_values(["blklot", "year"])
    last = pd.merge_asof(sp[["blklot", "year"]].reset_index().sort_values("year"),
                         s2.rename(columns={"sale_year": "year"}).assign(ly=lambda d: d.year),
                         on="year", by="blklot", direction="backward").set_index("index")
    sp["yrs_since_sale"] = sp.year - last.ly.reindex(sp.index)
    sp["last_jump"] = last.jump.reindex(sp.index)
    cnt = s2.groupby(["blklot", "sale_year"]).size().rename("n").reset_index()
    sp["n_sales_10y"] = _rolling_count(sp, cnt, 10)
    return sp


def _rolling_count(sp, cnt, w):
    c = cnt.pivot_table(index="blklot", columns="sale_year", values="n", fill_value=0)
    out = np.zeros(len(sp), dtype="int16")
    yrs = np.array(c.columns)
    cs = c.values.cumsum(1)
    row = pd.Series(np.arange(len(c)), index=c.index)
    ri = row.reindex(sp.blklot).values
    for k, (i, y) in enumerate(zip(ri, sp.year.values)):
        if i != i:
            continue
        hi = np.searchsorted(yrs, y, "right") - 1
        lo = np.searchsorted(yrs, y - w, "right") - 1
        out[k] = (cs[int(i), hi] if hi >= 0 else 0) - (cs[int(i), lo] if lo >= 0 else 0)
    return out


if __name__ == "__main__":
    INT.mkdir(parents=True, exist_ok=True)
    r = load_roll()
    s = sales(r)
    s.to_parquet(INT / "sales.parquet", index=False)
    sp = spine(r, s)
    sp.to_parquet(INT / "spine.parquet", index=False)
    print("roll", len(r), "sales", len(s), "spine", len(sp), "geo", sp.lat.notna().mean().round(3))
    print(sp[sp.year == 2018].rent_ctrl.sum(), "rent-ctrl parcels 2018")
