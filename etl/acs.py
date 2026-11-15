"""ACS 2014-18 5-year tract features -> H3 res-9 cells (centroid in tract). Light: ~200 tracts, ~1k cells.

Writes data/interim/acs_tract.parquet, data/interim/acs_cell.parquet. Needs CENSUS_API_KEY in .env.
python -m etl.acs
"""
import json
import os
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point, shape

INT = Path("data/interim")
TIGER = ("https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/tigerWMS_ACS2018/MapServer/8/query"
         "?where=STATE%3D%2706%27+AND+COUNTY%3D%27075%27&outFields=GEOID&f=geojson&outSR=4326")
VARS = {"B25014_001E": "occ", "B25014_005E": "o1", "B25014_006E": "o2", "B25014_007E": "o3",
        "B25014_011E": "r1", "B25014_012E": "r2", "B25014_013E": "r3", "B19013_001E": "med_income",
        "B03002_001E": "pop", "B03002_003E": "nh_white", "B25003_001E": "tenure", "B25003_003E": "renter"}


def key():
    for line in open(".env"):
        if line.startswith("CENSUS_API_KEY="):
            return line.split("=", 1)[1].strip()
    return os.environ["CENSUS_API_KEY"]


def tracts():
    gj = requests.get(TIGER, timeout=60).json()
    return gpd.GeoDataFrame([{"tract": f["properties"]["GEOID"], "geometry": shape(f["geometry"])} for f in gj["features"]],
                            crs=4326)


def acs():
    r = requests.get("https://api.census.gov/data/2018/acs/acs5",
                     params={"get": ",".join(VARS), "for": "tract:*", "in": "state:06 county:075", "key": key()},
                     timeout=60)
    r.raise_for_status()
    rows = r.json()
    a = pd.DataFrame(rows[1:], columns=rows[0]).rename(columns=VARS)
    a["tract"] = a.state + a.county + a.tract
    for c in VARS.values():
        a[c] = pd.to_numeric(a[c], errors="coerce")
    a.loc[a.med_income < 0, "med_income"] = np.nan  # census sentinel
    a["overcrowding"] = (a[["o1", "o2", "o3", "r1", "r2", "r3"]].sum(axis=1) / a.occ.where(a.occ > 0))
    a["median_income_log"] = np.log(a.med_income)
    a["pct_nonwhite"] = 1 - a.nh_white / a["pop"].where(a["pop"] > 0)
    a["pct_renter"] = a.renter / a.tenure.where(a.tenure > 0)
    return a[["tract", "pop", "occ", "med_income", "overcrowding", "median_income_log", "pct_nonwhite", "pct_renter"]]


def build():
    t, a = tracts(), acs()
    ta = t.merge(a, on="tract")
    cells = set(pd.read_parquet("data/processed/risk_dataset.parquet", columns=["cell"]).cell.dropna())
    cells |= set(pd.read_parquet("data/processed/probe_targets.parquet", columns=["cell"]).cell)
    pts = gpd.GeoDataFrame({"cell": sorted(cells)},
                           geometry=[Point(h3.cell_to_latlng(c)[::-1]) for c in sorted(cells)], crs=4326)
    j = gpd.sjoin(pts, ta, how="left", predicate="within").drop(columns=["geometry", "index_right"])
    # cells whose centroid falls in water/outside: nearest tract
    miss = j.tract.isna()
    if miss.any():
        nn = gpd.sjoin_nearest(pts[miss.values].to_crs(7131), ta.to_crs(7131), how="left").drop(columns="geometry")
        j = pd.concat([j[~miss], nn.drop(columns="index_right")[j.columns]], ignore_index=True)
    w = ta[ta["pop"] > 500]
    edges = lambda c: np.nanpercentile(w[c], [100 / 3, 200 / 3])
    for c, lab in [("med_income", "income_tercile"), ("pct_nonwhite", "nonwhite_tercile")]:
        e = edges(c)
        j[lab] = np.select([j[c] <= e[0], j[c] <= e[1], j[c] > e[1]], ["low", "mid", "high"], "NA")
    INT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ta.drop(columns="geometry")).to_parquet(INT / "acs_tract.parquet", index=False)
    j.drop_duplicates("cell").to_parquet(INT / "acs_cell.parquet", index=False)
    return ta, j


if __name__ == "__main__":
    ta, j = build()
    print(len(ta), "tracts;", len(j), "cells;", j.tract.isna().sum(), "unmatched")
    print(j[["overcrowding", "median_income_log", "pct_nonwhite"]].describe().round(3))
