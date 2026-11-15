"""Hand-check 20 origin -> nearest-POI routed times against public planners.

    python -m sim.patient.handcheck --config configs/sim.yaml [--n 20] [--no-osrm]

* car pairs: queried automatically against the public OSRM demo router (free-flow driving, same
  assumption as r5's car mode). Only the two coordinates are sent.
* transit pairs: we can't query Google/511 programmatically, so each row carries a Google Maps
  directions URL; a human fills `planner_min` (set "Depart at" Tue 10:00 / 18:00 to match `depart`).
Writes reports/sim/handcheck.csv and prints the error distribution of the filled rows.
"""
from __future__ import annotations

import argparse
import time

import h3
import numpy as np
import pandas as pd
import requests

from .config import load_config, p
from .pois import CLASSES, load_pois

OSRM = "https://router.project-osrm.org/route/v1/driving/{o_lon},{o_lat};{d_lon},{d_lat}?overview=false"


def gmaps_url(o_lat, o_lon, d_lat, d_lon, mode):
    tm = "driving" if mode == "car" else "transit"
    return (f"https://www.google.com/maps/dir/?api=1&origin={o_lat:.6f},{o_lon:.6f}"
            f"&destination={d_lat:.6f},{d_lon:.6f}&travelmode={tm}")


def osrm_minutes(o_lat, o_lon, d_lat, d_lon) -> float | None:
    try:
        r = requests.get(OSRM.format(o_lat=o_lat, o_lon=o_lon, d_lat=d_lat, d_lon=d_lon), timeout=20,
                         headers={"User-Agent": "patient-sim-handcheck/0.1"})
        j = r.json()
        return round(j["routes"][0]["duration"] / 60.0, 1) if j.get("code") == "Ok" else None
    except Exception:
        return None


def sample_pairs(cfg: dict, n: int, seed: int = 0) -> pd.DataFrame:
    tab = pd.read_parquet(p(cfg, "interim") / "access_cells.parquet")
    pois = load_pois(cfg).set_index("poi_id")
    rows = []
    for c in CLASSES:
        t = tab[np.isfinite(tab[f"t_{c}"])][["cell", "mode", "depart", f"t_{c}", f"poi_{c}"]]
        t = t.rename(columns={f"t_{c}": "sim_min", f"poi_{c}": "poi_id"}).assign(cls=c)
        rows.append(t)
    allp = pd.concat(rows, ignore_index=True)
    rng = np.random.default_rng(seed)
    # half car / half transit, spread over classes; prefer non-trivial trips (> 3 min)
    allp = allp[allp["sim_min"] > 3]
    picks = []
    for mode, k in (("car", n // 2), ("transit_walk", n - n // 2)):
        sub = allp[allp["mode"] == mode]
        picks.append(sub.iloc[rng.choice(len(sub), size=min(k, len(sub)), replace=False)])
    out = pd.concat(picks, ignore_index=True)
    ll = np.array([h3.cell_to_latlng(c) for c in out["cell"]])
    out["o_lat"], out["o_lon"] = ll[:, 0], ll[:, 1]
    out["d_lat"] = out["poi_id"].map(pois["lat"])
    out["d_lon"] = out["poi_id"].map(pois["lon"])
    out["poi_name"] = out["poi_id"].map(pois["name"])
    out["side"] = np.where((out["o_lat"].between(37.70, 37.84)) & (out["o_lon"].between(-122.52, -122.35)), "before(SF)", "after")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sim.yaml")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--no-osrm", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    df = sample_pairs(cfg, a.n)
    df["planner"] = np.where(df["mode"] == "car", "osrm_demo", "google_maps_manual")
    df["planner_min"] = np.nan
    if not a.no_osrm:
        for i in df.index[df["mode"] == "car"]:
            r = df.loc[i]
            df.loc[i, "planner_min"] = osrm_minutes(r.o_lat, r.o_lon, r.d_lat, r.d_lon)
            time.sleep(1.0)  # be polite to the demo server
    df["planner_url"] = [gmaps_url(r.o_lat, r.o_lon, r.d_lat, r.d_lon, r["mode"]) for _, r in df.iterrows()]
    df["err_min"] = df["sim_min"] - df["planner_min"]
    df["err_pct"] = 100 * df["err_min"] / df["planner_min"]
    cols = ["side", "cls", "mode", "depart", "o_lat", "o_lon", "poi_id", "poi_name", "d_lat", "d_lon",
            "sim_min", "planner", "planner_min", "err_min", "err_pct", "planner_url"]
    out = p(cfg, "reports") / "handcheck.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df[cols].round(4).to_csv(out, index=False)
    done = df.dropna(subset=["planner_min"])
    print(f"wrote {len(df)} pairs -> {out}; {len(done)} checked automatically")
    if len(done):
        e = done["err_min"]
        print(f"err_min: median {e.median():+.1f}, MAE {e.abs().mean():.1f}, p90 |err| {e.abs().quantile(.9):.1f}; "
              f"median |err%| {done['err_pct'].abs().median():.0f}%")


if __name__ == "__main__":
    main()
