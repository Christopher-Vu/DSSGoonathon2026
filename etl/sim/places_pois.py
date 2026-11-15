"""Sponsor Google Maps Places -> internal POI schema (the final POI source).

    python -m etl.sim.places_pois [--config configs/sim.yaml]

* maps `category`/`all_categories` to {clinic, pharmacy, grocery} with the reviewed list in sim/patient/pois.py
* clips to the region counties (the raw bbox includes Santa Cruz), drops permanently/temporarily closed
* writes processed/sim/pois_places.parquet
* writes reports/sim/places_spotcheck.csv (30 random per class, `ok` column for a human) and
  reports/sim/places_vs_osm.csv (per-county counts + 150 m same-class match rates in both directions)
* prints the category mix that was mapped / the top unmapped health-ish categories for review
"""
from __future__ import annotations

import argparse

import geopandas as gpd
import numpy as np
import pandas as pd

from etl.sim.osm_region import region_geom
from sim.patient.config import load_config, p
from sim.patient.pois import CLASSES, load_places


def match_rate(a: pd.DataFrame, b: pd.DataFrame, radius_m=150.0) -> pd.Series:
    """Share of `a` POIs with a same-class `b` POI within radius, per class."""
    out = {}
    for c in CLASSES:
        ga = gpd.GeoDataFrame(a[a.cls == c], geometry=gpd.points_from_xy(a[a.cls == c].lon, a[a.cls == c].lat), crs=4326).to_crs(3310)
        gb = gpd.GeoDataFrame(b[b.cls == c], geometry=gpd.points_from_xy(b[b.cls == c].lon, b[b.cls == c].lat), crs=4326).to_crs(3310)
        if ga.empty or gb.empty:
            out[c] = np.nan
            continue
        j = gpd.sjoin_nearest(ga[["geometry"]], gb[["geometry"]], max_distance=radius_m, how="left")
        out[c] = float(j[~j.index.duplicated()]["index_right"].notna().mean())
    return pd.Series(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sim.yaml")
    cfg = load_config(ap.parse_args().config)
    src = p(cfg, "places_csv")
    if not src.exists():
        raise SystemExit(f"Places file not found at {src}; the sim keeps using OSM POIs until it lands.")
    geom = region_geom(cfg)
    df = load_places(src, cfg["pois"]["default_hours"], geom.bounds)
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326)
    df = df[g.within(geom).to_numpy()].reset_index(drop=True)
    # Places' bounding box stops at lon -121.55 / lat 36.90-38.90: east Sacramento + San Joaquin are outside.
    # Fill only that uncovered area with OSM POIs (source='osm') so those destinations aren't spuriously empty.
    raw_bbox = pd.read_csv(src, usecols=["lat", "lon"]).agg(["min", "max"])
    cov = (raw_bbox.lon["min"], raw_bbox.lat["min"], raw_bbox.lon["max"], raw_bbox.lat["max"])
    if p(cfg, "osm_pois").exists():
        osm = pd.read_parquet(p(cfg, "osm_pois"))
        outside = ~(osm.lon.between(cov[0], cov[2]) & osm.lat.between(cov[1], cov[3]))
        df = pd.concat([df, osm[outside]], ignore_index=True)
        print(f"Places coverage bbox {tuple(round(v, 3) for v in cov)}; added {int(outside.sum())} OSM POIs outside it")
    pd.Series({"min_lon": cov[0], "min_lat": cov[1], "max_lon": cov[2], "max_lat": cov[3]}).to_json(
        p(cfg, "interim") / "places_coverage.json")
    dst = p(cfg, "places_pois")
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    print(f"{len(df):,} Places POIs -> {dst}")
    print(df.groupby("cls").agg(n=("poi_id", "size"), default_hours=("hours_default", "mean"),
                                open10=("open_wd1000", "mean"), open18=("open_wd1800", "mean")).round(3))
    print("mapped categories:\n", df[df.source == "places"].groupby(["cls", "raw_category"]).size().sort_values(ascending=False).head(40))

    rep = p(cfg, "reports")
    rep.mkdir(parents=True, exist_ok=True)
    pl = df[df.source == "places"]
    spot = pd.concat([d.sample(min(30, len(d)), random_state=0) for _, d in pl.groupby("cls")])
    spot.assign(ok="", note="")[["cls", "name", "raw_category", "lat", "lon", "hours_default", "ok", "note"]] \
        .to_csv(rep / "places_spotcheck.csv", index=False)

    if p(cfg, "osm_pois").exists():
        osm = pd.read_parquet(p(cfg, "osm_pois"))
        osm_in = osm[osm.lon.between(cov[0], cov[2]) & osm.lat.between(cov[1], cov[3])]
        cmp_ = pd.DataFrame({"n_places": pl.cls.value_counts(), "n_osm": osm_in.cls.value_counts(),
                             "osm_matched_in_places_150m": match_rate(osm_in, pl),
                             "places_matched_in_osm_150m": match_rate(pl, osm_in)})
        cmp_["ratio_places_to_osm"] = cmp_.n_places / cmp_.n_osm
        cmp_.round(3).to_csv(rep / "places_vs_osm.csv")
        print(cmp_.round(3))


if __name__ == "__main__":
    main()
