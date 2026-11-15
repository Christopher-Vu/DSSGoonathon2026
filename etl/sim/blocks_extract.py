"""2020 Census block internal points + housing-unit counts, tagged with 2020 PUMA, region counties only.

    python -m etl.sim.blocks_extract [--config configs/sim.yaml]

Used to sample a residential point in a destination PUMA (weighted by HOUSING20). Writes
<data>/processed/sim/blocks.parquet.
"""
from __future__ import annotations

import argparse

import geopandas as gpd
import pyogrio

from sim.patient.config import load_config, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sim.yaml")
    cfg = load_config(ap.parse_args().config)
    counties = cfg["migration"]["region_counties"]
    where = "COUNTYFP20 IN (" + ",".join(f"'{c}'" for c in counties) + ") AND HOUSING20 > 0"
    df = pyogrio.read_dataframe(f"zip://{p(cfg, 'blocks')}", columns=["GEOID20", "COUNTYFP20", "HOUSING20",
                                "POP20", "INTPTLAT20", "INTPTLON20"], read_geometry=False, where=where)
    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["INTPTLON20"].astype(float),
                           df["INTPTLAT20"].astype(float)), crs="EPSG:4326")
    pu = gpd.read_file(f"zip://{p(cfg, 'puma_shapes')}")[["PUMACE20", "geometry"]].to_crs("EPSG:4326")
    j = gpd.sjoin(pts, pu, predicate="within", how="inner")
    out = j.assign(lat=j.geometry.y, lon=j.geometry.x)[["GEOID20", "COUNTYFP20", "PUMACE20", "HOUSING20", "POP20", "lat", "lon"]]
    out = out.rename(columns={"GEOID20": "block", "COUNTYFP20": "county", "PUMACE20": "puma",
                              "HOUSING20": "housing", "POP20": "pop"})
    dst = p(cfg, "interim") / "blocks.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dst, index=False)
    print(f"{len(out)} blocks, {out['housing'].sum():,} housing units, {out['puma'].nunique()} PUMAs -> {dst}")


if __name__ == "__main__":
    main()
