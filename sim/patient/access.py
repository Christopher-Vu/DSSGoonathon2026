"""Travel time from origin H3 cells to the nearest *open* POI of each class.

Routing runs once per unique origin cell, not per household:
  * transit_walk: one r5 run per departure (10:00, 18:00) against POIs open at that time;
  * car: one r5 run (r5 has no traffic, so departure doesn't change times) against all POIs,
    reduced separately per departure's open set.
Result rows: (cell, mode, depart) -> t_<cls> minutes (inf if > max_minutes / unreachable) + nearest poi id.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import time
from pathlib import Path

import h3
import numpy as np
import pandas as pd

from .pois import CLASSES, DEPARTS

log = logging.getLogger("sim.access")


def open_col(depart: str) -> str:
    return "open_" + depart.replace("weekday_", "wd")


def cells_to_points(cells) -> pd.DataFrame:
    ll = np.array([h3.cell_to_latlng(c) for c in cells]) if len(cells) else np.zeros((0, 2))
    return pd.DataFrame({"cell": list(cells), "lat": ll[:, 0], "lon": ll[:, 1]})


def _reduce(long: pd.DataFrame, pois: pd.DataFrame, departs: list[str], max_min: float) -> pd.DataFrame:
    """long: from_id(cell), to_id(poi_id), travel_time -> min per (cell, depart, cls)."""
    long = long.dropna(subset=["travel_time"])
    long = long[long["travel_time"] <= max_min]
    meta = pois.set_index("poi_id")
    long = long.join(meta[["cls"] + [open_col(d) for d in departs]], on="to_id")
    out = []
    for d in departs:
        sub = long[long[open_col(d)]]
        if sub.empty:
            continue
        idx = sub.groupby(["from_id", "cls"])["travel_time"].idxmin()
        best = sub.loc[idx, ["from_id", "cls", "to_id", "travel_time"]].assign(depart=d)
        out.append(best)
    if not out:
        return pd.DataFrame(columns=["from_id", "cls", "to_id", "travel_time", "depart"])
    return pd.concat(out, ignore_index=True)


def _wide(best: pd.DataFrame, cells: list[str], departs: list[str], mode: str) -> pd.DataFrame:
    base = pd.MultiIndex.from_product([cells, departs], names=["cell", "depart"]).to_frame(index=False)
    for c in CLASSES:
        b = best[best["cls"] == c].rename(columns={"from_id": "cell", "travel_time": f"t_{c}", "to_id": f"poi_{c}"})
        base = base.merge(b[["cell", "depart", f"t_{c}", f"poi_{c}"]], on=["cell", "depart"], how="left")
        base[f"t_{c}"] = base[f"t_{c}"].astype("float32").fillna(np.float32(np.inf))
    base["mode"] = mode
    return base


# ---------------------------------------------------------------- euclid backend (tests / no Java)
def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def euclid_long(orig: pd.DataFrame, dest: pd.DataFrame, mode: str, rc: dict) -> pd.DataFrame:
    v = rc["euclid_speeds_kmh"][mode]
    rows = []
    for s in range(0, len(orig), 500):
        o = orig.iloc[s:s + 500]
        km = _haversine_km(o["lat"].to_numpy()[:, None], o["lon"].to_numpy()[:, None],
                           dest["lat"].to_numpy()[None, :], dest["lon"].to_numpy()[None, :]) * rc["detour_factor"]
        t = km / v * 60.0 + (5.0 if mode == "transit_walk" else 2.0)
        if mode == "transit_walk":  # walking beats waiting for short hops
            t = np.minimum(t, km / 4.8 * 60.0)
        ii, jj = np.nonzero(t <= rc["max_minutes"])
        rows.append(pd.DataFrame({"from_id": o["cell"].to_numpy()[ii], "to_id": dest["poi_id"].to_numpy()[jj],
                                  "travel_time": t[ii, jj]}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["from_id", "to_id", "travel_time"])


# ---------------------------------------------------------------- r5 backend
class R5Router:
    def __init__(self, osm_pbf: Path, gtfs: list[Path], rc: dict):
        jh = rc.get("java_home") or os.environ.get("JAVA_HOME")
        if jh:
            os.environ["JAVA_HOME"] = str(jh)
        import r5py  # noqa: import after JAVA_HOME

        self.r5py = r5py
        self.rc = rc
        t0 = time.time()
        log.info("building r5 network: %s + %d GTFS", osm_pbf.name, len(gtfs))
        self.net = r5py.TransportNetwork(str(osm_pbf), [str(g) for g in gtfs])
        log.info("network ready in %.0fs", time.time() - t0)

    def long(self, orig: pd.DataFrame, dest: pd.DataFrame, mode: str, depart: str) -> pd.DataFrame:
        import geopandas as gpd

        r5py, rc = self.r5py, self.rc
        hour = DEPARTS[depart]
        d0 = dt.datetime.fromisoformat(rc["date"]).replace(hour=int(hour), minute=int(round(hour % 1 * 60)))
        modes = [r5py.TransportMode.TRANSIT, r5py.TransportMode.WALK] if mode == "transit_walk" else [r5py.TransportMode.CAR]
        dg = gpd.GeoDataFrame({"id": dest["poi_id"].to_numpy()},
                              geometry=gpd.points_from_xy(dest["lon"], dest["lat"]), crs="EPSG:4326")
        out = []
        step = int(rc.get("origin_chunk", 400))
        for s in range(0, len(orig), step):
            o = orig.iloc[s:s + step]
            og = gpd.GeoDataFrame({"id": o["cell"].to_numpy()},
                                  geometry=gpd.points_from_xy(o["lon"], o["lat"]), crs="EPSG:4326")
            t0 = time.time()
            m = r5py.TravelTimeMatrix(
                self.net, origins=og, destinations=dg, departure=d0,
                departure_time_window=dt.timedelta(minutes=rc["departure_window_min"]),
                transport_modes=modes, max_time=dt.timedelta(minutes=rc["max_minutes"]),
            )
            m = pd.DataFrame(m)[["from_id", "to_id", "travel_time"]].dropna()
            out.append(m)
            log.info("%s %s origins %d-%d: %d pairs in %.0fs", mode, depart, s, s + len(o), len(m), time.time() - t0)
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["from_id", "to_id", "travel_time"])


def access_table(needs: pd.DataFrame, pois: pd.DataFrame, rc: dict, router=None) -> pd.DataFrame:
    """needs: unique (cell, mode, depart) rows. Car rows also need transit rows (car households take
    min(car, transit)), so the caller should include them. Returns wide table keyed by (cell, mode, depart)."""
    max_min = float(rc["max_minutes"])
    backend = rc["backend"]
    pieces = []
    for mode in ("transit_walk", "car"):
        nm = needs[needs["mode"] == mode]
        if nm.empty:
            continue
        if mode == "transit_walk":
            runs = [(d, sorted(nm.loc[nm["depart"] == d, "cell"].unique()), [d]) for d in sorted(nm["depart"].unique())]
        else:  # one run for all departures
            runs = [("weekday_1000", sorted(nm["cell"].unique()), sorted(nm["depart"].unique()))]
        for run_depart, cells, departs in runs:
            orig = cells_to_points(cells)
            if mode == "transit_walk":
                dest = pois[pois[open_col(run_depart)]]
            else:
                dest = pois[pois[[open_col(d) for d in departs]].any(axis=1)]
            t0 = time.time()
            long = (euclid_long(orig, dest, mode, rc) if backend == "euclid"
                    else router.long(orig, dest, mode, run_depart))
            best = _reduce(long, pois, departs, max_min)
            w = _wide(best, cells, departs, mode)
            pieces.append(w)
            log.info("%s/%s: %d cells x %d POIs in %.0fs", mode, run_depart, len(cells), len(dest), time.time() - t0)
    tab = pd.concat(pieces, ignore_index=True)
    return tab.merge(needs[["cell", "mode", "depart"]].drop_duplicates(), on=["cell", "mode", "depart"], how="inner")
