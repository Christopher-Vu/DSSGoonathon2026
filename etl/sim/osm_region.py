"""Clip NorCal OSM to the sim region (roads only, for r5) and extract stand-in POIs.

    python -m etl.sim.osm_region [--config configs/sim.yaml] [--skip-clip] [--skip-pois]

* region bbox = union of region-county PUMA polygons + 0.05 deg buffer
* region.osm.pbf  : highway=* ways touching the bbox + their nodes (BackReferenceWriter)
* pois_osm.parquet: amenity=clinic|doctors|hospital|pharmacy, healthcare=*, shop=supermarket|grocery|greengrocer
                    (nodes + closed-way centroids) clipped to region counties, with Tuesday open flags.
"""
from __future__ import annotations

import argparse
import time

import geopandas as gpd
import osmium
import pandas as pd

from sim.patient.config import load_config, p
from sim.patient.pois import OSM_TAGS, add_open_flags, parse_osm_hours
from sim.patient.pums import region_pumas


def region_geom(cfg):
    pu = gpd.read_file(f"zip://{p(cfg, 'puma_shapes')}").to_crs("EPSG:4326")
    reg = region_pumas(pu, cfg["migration"]["region_counties"])
    return pu[pu["PUMACE20"].isin(reg)].union_all()


def clip_roads(src, dst, bbox):
    x0, y0, x1, y1 = bbox
    n = 0
    t0 = time.time()
    with osmium.BackReferenceWriter(str(dst), ref_src=str(src), overwrite=True) as w:
        fp = osmium.FileProcessor(str(src), osmium.osm.NODE | osmium.osm.WAY).with_locations().with_filter(
            osmium.filter.EntityFilter(osmium.osm.WAY)).with_filter(osmium.filter.KeyFilter("highway"))
        for way in fp:
            for nd in way.nodes:
                loc = nd.location
                if loc.valid() and x0 <= loc.lon <= x1 and y0 <= loc.lat <= y1:
                    w.add_way(way)
                    n += 1
                    break
    print(f"clipped {n:,} highway ways in {time.time() - t0:.0f}s -> {dst}")


def _cls(tags) -> str | None:
    for (k, v), c in OSM_TAGS.items():
        if tags.get(k) == v:
            return c
    return None


def extract_pois(src, bbox):
    x0, y0, x1, y1 = bbox
    rows = []
    filt = [osmium.filter.KeyFilter("amenity", "shop", "healthcare")]
    fp = osmium.FileProcessor(str(src), osmium.osm.NODE | osmium.osm.WAY).with_locations()
    for f in filt:
        fp = fp.with_filter(f)
    for o in fp:
        tags = dict(o.tags)
        c = _cls(tags)
        if c is None:
            continue
        if o.is_node():
            lat, lon, kind = o.location.lat, o.location.lon, "n"
        else:
            locs = [nd.location for nd in o.nodes if nd.location.valid()]
            if not locs:
                continue
            lat = sum(l.lat for l in locs) / len(locs)
            lon = sum(l.lon for l in locs) / len(locs)
            kind = "w"
        if not (x0 <= lon <= x1 and y0 <= lat <= y1):
            continue
        rows.append({"poi_id": f"osm_{kind}{o.id}", "name": tags.get("name", ""), "cls": c,
                     "raw_category": next(f"{k}={tags[k]}" for k in ("amenity", "healthcare", "shop") if k in tags),
                     "lat": lat, "lon": lon, "opening_hours": tags.get("opening_hours")})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sim.yaml")
    ap.add_argument("--src", default=None, help="default: raw/sim/osm/norcal-latest.osm.pbf")
    ap.add_argument("--skip-clip", action="store_true")
    ap.add_argument("--skip-pois", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    src = a.src or (p(cfg, "osm_pbf").parent / "norcal-latest.osm.pbf")
    geom = region_geom(cfg)
    x0, y0, x1, y1 = geom.bounds
    bbox = (x0 - 0.05, y0 - 0.05, x1 + 0.05, y1 + 0.05)
    print("region bbox", [round(v, 3) for v in bbox])
    if not a.skip_clip:
        clip_roads(src, p(cfg, "osm_pbf"), bbox)
    if not a.skip_pois:
        t0 = time.time()
        df = extract_pois(src, bbox)
        g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
        df = df[g.within(geom.buffer(0.01)).to_numpy()].drop_duplicates("poi_id").reset_index(drop=True)
        ranges = pd.Series([parse_osm_hours(x) for x in df["opening_hours"]], dtype=object)
        out = add_open_flags(df.drop(columns=["opening_hours"]).assign(source="osm"), ranges, cfg["pois"]["default_hours"])
        dst = p(cfg, "osm_pois")
        dst.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(dst, index=False)
        print(f"{len(out):,} OSM POIs in {time.time() - t0:.0f}s -> {dst}")
        print(out.groupby("cls").agg(n=("poi_id", "size"), default_hours=("hours_default", "mean"),
                                     open10=("open_wd1000", "mean"), open18=("open_wd1800", "mean")).round(3))


if __name__ == "__main__":
    main()
