"""displaced_households.parquet (A/D -> B), contract section 4.

Scenarios:
  realized_2019_2024          every eviction notice (any cause) + buyout, 2019-24, at the event point
  unprotected__<strategy>_<m>x  what remains if <strategy> had bought its backtest picks at m x budget:
                              buyouts at bought lots removed; Ellis/OMI notice weight scaled by
                              1 - (bought unit share of the block-face). Same rule as the backtest value.
Weight: 1 per notice; buyouts use tenants_share (same unit as the backtest's "households").

python -m etl.displaced --fetch-only (once, light; needs internet) ; python -m etl.displaced --tag gbm
"""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import shape

INT, D = Path("data/interim"), Path("data/processed")
LO, HI = 2019, 2024
TIGER20 = ("https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/tigerWMS_Census2020/MapServer/6/query"
           "?where=STATE%3D%2706%27+AND+COUNTY%3D%27075%27&outFields=GEOID&f=geojson&outSR=4326")
STRATS = [("ours", 1), ("city_policy", 1), ("ours", 5), ("city_policy", 5)]


def tracts2020():
    p = INT / "tracts2020.parquet"
    if not p.exists():
        gj = requests.get(TIGER20, timeout=60).json()
        g = gpd.GeoDataFrame([{"tract": f["properties"]["GEOID"], "geometry": shape(f["geometry"])}
                              for f in gj["features"]], crs=4326)
        g.to_parquet(p)
    return gpd.read_parquet(p)


def events():
    ev = pd.read_parquet(INT / "eviction_events.parquet")
    ev = ev[ev.year.between(LO, HI)].copy()
    bf = pd.read_parquet(INT / "blockfaces.parquet")[["bf", "lat", "lon"]].rename(columns={"lat": "blat", "lon": "blon"})
    ev = ev.merge(bf, on="bf", how="left")
    ev["lat"], ev["lon"] = ev.lat.fillna(ev.blat), ev.lon.fillna(ev.blon)
    ev["event_type"] = np.select([ev.ellis_act_withdrawal.fillna(False), ev.owner_move_in.fillna(False)],
                                 ["ellis", "omi"], "other")
    e = pd.DataFrame({"household_id": "ev_" + ev.eviction_id.astype(str), "lat": ev.lat, "lon": ev.lon,
                      "event_year": ev.year, "event_type": ev.event_type, "weight": 1.0,
                      "bf": ev.bf.where(ev.bf_ok & ev.ellis_omi), "blklot": None})
    bo = pd.read_parquet(INT / "buyout_events.parquet").dropna(subset=["blklot"])
    bo = bo[bo.year.between(LO, HI)]
    pc = pd.read_parquet(Path("data/raw/datasf/parcels.parquet"), columns=["blklot", "centroid_latitude",
                                                                         "centroid_longitude"]).drop_duplicates("blklot")
    bo = bo.merge(pc, on="blklot", how="left")
    b = pd.DataFrame({"household_id": "bo_" + bo.case_number.astype(str) + "_" + bo.blklot,
                      "lat": pd.to_numeric(bo.centroid_latitude, errors="coerce"),
                      "lon": pd.to_numeric(bo.centroid_longitude, errors="coerce"),
                      "event_year": bo.year, "event_type": "buyout", "weight": bo.tenants_share.fillna(1.0),
                      "bf": None, "blklot": bo.blklot})
    x = pd.concat([e, b], ignore_index=True).dropna(subset=["lat", "lon", "event_year"])
    return x[x.weight > 0]


def build(tag="gbm"):
    x = events()
    t = tracts2020()
    pts = gpd.GeoDataFrame(x, geometry=gpd.points_from_xy(x.lon, x.lat), crs=4326)
    j = gpd.sjoin_nearest(pts.to_crs(7131), t.to_crs(7131), how="left").drop_duplicates("household_id")
    x["origin_tract"] = j.set_index("household_id").tract.reindex(x.household_id).values
    u = pd.read_parquet(D / f"backtest_universe_{tag}.parquet")
    out = [x.assign(scenario="realized_2019_2024")]
    for s, m in STRATS:
        sel = u[u[f"sel_{s}_{m}x"]]
        bought_share = sel.groupby("bf").bf_share.sum().clip(upper=1)
        y = x[~x.blklot.isin(set(sel.blklot))].copy()
        y["weight"] = y.weight * (1 - y.bf.map(bought_share).fillna(0))
        out.append(y[y.weight > 1e-6].assign(scenario=f"unprotected__{s}_{m}x"))
    d = pd.concat(out, ignore_index=True).rename(columns={"lat": "origin_lat", "lon": "origin_lon"})
    d = d[["household_id", "scenario", "origin_lat", "origin_lon", "origin_tract", "event_year", "event_type",
           "weight"]].astype({"event_year": "int16", "weight": "float32", "origin_lat": "float64",
                              "origin_lon": "float64"})
    d.to_parquet(D / "displaced_households.parquet", index=False)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gbm")
    ap.add_argument("--fetch-only", action="store_true")
    a = ap.parse_args()
    if a.fetch_only:
        print(len(tracts2020()), "tracts")
    else:
        d = build(a.tag)
        print(d.groupby("scenario").weight.agg(["size", "sum"]).round(1))
        print(d[d.scenario == "realized_2019_2024"].event_type.value_counts())
