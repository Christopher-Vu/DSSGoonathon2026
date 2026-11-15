"""Patient sim entry point.

    python -m sim.patient.run --households data/processed/displaced_households.parquet --config configs/sim.yaml

Outputs (CONTRACTS §5): sim_results.parquet, sim_summary.json, figures in reports/sim/.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd

from . import access, pums
from .config import load_config, p
from .pois import CLASSES, load_pois
from .summary import headline, summarize

log = logging.getLogger("sim")
H3_RES = 9


# ---------------------------------------------------------------- inputs
def load_inputs(cfg: dict) -> dict:
    inter = p(cfg, "interim")
    h = pd.read_parquet(inter / "pums_households.parquet")
    pp = pd.read_parquet(inter / "pums_persons.parquet")
    shapes = gpd.read_file(f"zip://{p(cfg, 'puma_shapes')}")[["PUMACE20", "NAMELSAD20", "geometry"]].to_crs("EPSG:4326")
    blocks = pd.read_parquet(inter / "blocks.parquet")
    return {"h": h, "pp": pp, "shapes": shapes, "blocks": blocks, "pois": load_pois(cfg)}


def origin_puma(hh: pd.DataFrame, shapes: gpd.GeoDataFrame) -> np.ndarray:
    sf = shapes[shapes["PUMACE20"].isin(pums.SF_PUMAS)]
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(hh["origin_lon"], hh["origin_lat"]), crs="EPSG:4326")
    j = gpd.sjoin_nearest(pts.to_crs(3310), sf.to_crs(3310)[["PUMACE20", "geometry"]], how="left")
    j = j[~j.index.duplicated()]
    return j["PUMACE20"].to_numpy()


def expand(hh: pd.DataFrame, n_min: int, rng) -> pd.DataFrame:
    """≥ n_min rows per scenario; small scenarios are resampled ∝ weight into syn_<n> households."""
    out = []
    for scen, df in hh.groupby("scenario", sort=True):
        df = df.reset_index(drop=True)
        if len(df) >= n_min:
            out.append(df.assign(origin_household_id=df["household_id"], weight=df["weight"].astype("float32")))
            continue
        w = df["weight"].to_numpy(float)
        idx = rng.choice(len(df), size=n_min, p=w / w.sum())
        s = df.iloc[idx].reset_index(drop=True)
        s["origin_household_id"] = s["household_id"]
        s["household_id"] = [f"syn_{i}" for i in range(n_min)]
        s["weight"] = np.float32(w.sum() / n_min)
        out.append(s)
    return pd.concat(out, ignore_index=True)


def assign_personas(sim: pd.DataFrame, personas: pd.DataFrame, rng) -> pd.DataFrame:
    cols = ["age_band", "income_band", "lang", "has_car", "disability", "depart", "serialno"]
    parts = []
    for pu, idx in sim.groupby("origin_puma").groups.items():
        pool = personas[personas["puma"] == pu]
        if pool.empty:
            pool = personas
        pick = rng.choice(len(pool), size=len(idx), p=(pool["wgtp"] / pool["wgtp"].sum()).to_numpy())
        parts.append(pool.iloc[pick][cols].set_index(pd.Index(idx)))
    return sim.join(pd.concat(parts))


def assign_destinations(sim: pd.DataFrame, model: pums.MigrationModel, blocks: pd.DataFrame, rng) -> pd.DataFrame:
    sim = sim.copy()
    sim["dest_puma"] = model.sample(sim["income_band"].to_numpy(), rng)
    lat = np.empty(len(sim))
    lon = np.empty(len(sim))
    for pu, idx in sim.groupby("dest_puma").groups.items():
        b = blocks[blocks["puma"] == pu]
        pick = rng.choice(len(b), size=len(idx), p=(b["housing"] / b["housing"].sum()).to_numpy())
        pos = sim.index.get_indexer(idx)
        lat[pos], lon[pos] = b["lat"].to_numpy()[pick], b["lon"].to_numpy()[pick]
    sim["dest_lat"], sim["dest_lon"] = lat, lon
    sim["dest_puma"] = "06" + sim["dest_puma"].astype(str)  # 2020 PUMA GEOID (state + PUMACE)
    return sim


def _cells(lat, lon):
    return [h3.latlng_to_cell(a, b, H3_RES) for a, b in zip(lat, lon)]


# ---------------------------------------------------------------- main
def run(households: Path, cfg: dict) -> tuple[pd.DataFrame, dict]:
    t_start = time.time()
    rng = np.random.default_rng(cfg["seed"])
    rc = cfg["routing"]
    inp = load_inputs(cfg)
    pois = inp["pois"]
    hh = pd.read_parquet(households)
    hh["origin_puma"] = origin_puma(hh, inp["shapes"])

    personas = pums.build_personas(inp["h"], inp["pp"], cfg)
    mv = pums.movers(inp["h"], inp["pp"], cfg)
    region = pums.region_pumas(inp["shapes"], cfg["migration"]["region_counties"])
    bands = [b[2] for b in cfg["personas"]["income_bands"]]
    model = pums.fit_migration(mv, region, cfg, bands=bands)
    pop_w = inp["blocks"].groupby("puma")["housing"].sum().to_dict()
    validation = pums.validate_holdout(mv, region, cfg, pop_w)
    log.info("migration: %s | holdout: %s", model.stats, validation)

    sim = expand(hh, int(cfg["min_households_per_scenario"]), rng)
    sim = assign_personas(sim, personas, rng)
    sim = assign_destinations(sim, model, inp["blocks"], rng)
    sim["mode"] = np.where(sim["has_car"], "car", "transit_walk")
    sim["cell_before"] = _cells(sim["origin_lat"], sim["origin_lon"])
    sim["cell_after"] = _cells(sim["dest_lat"], sim["dest_lon"])

    # routing needs: every household needs transit at its depart; car households also need car
    rows = []
    for side in ("cell_before", "cell_after"):
        rows.append(sim[[side, "depart"]].rename(columns={side: "cell"}).assign(mode="transit_walk"))
        rows.append(sim.loc[sim["has_car"], [side, "depart"]].rename(columns={side: "cell"}).assign(mode="car"))
    needs = pd.concat(rows).drop_duplicates()
    log.info("routing needs: %d (cell, mode, depart) rows, %d unique cells", len(needs), needs["cell"].nunique())

    router = None
    if rc["backend"] == "r5":
        gtfs = sorted(p(cfg, "gtfs_dir").glob("*.zip"))
        router = access.R5Router(p(cfg, "osm_pbf"), gtfs, rc)
    tab = access.access_table(needs, pois, rc, router)
    tab.to_parquet(p(cfg, "interim") / "access_cells.parquet", index=False)

    sim = attach_times(sim, tab, float(rc["max_minutes"]))
    res = to_contract(sim)
    out_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    poi_stats = {
        "source": sorted(pois["source"].unique().tolist()),
        "n_by_class": pois["cls"].value_counts().to_dict(),
        "pct_default_hours_by_class": (pois.groupby("cls")["hours_default"].mean() * 100).round(1).to_dict(),
        "pct_open_wd1000": (pois.groupby("cls")["open_wd1000"].mean() * 100).round(1).to_dict(),
        "pct_open_wd1800": (pois.groupby("cls")["open_wd1800"].mean() * 100).round(1).to_dict(),
    }
    summary = {
        "scenarios": summarize(res, float(rc["max_minutes"])),
        "migration": {**model.stats, "n_dest_pumas": len(model.pumas)},
        "migration_validation": validation,
        "pois": poi_stats,
        "notes": [
            "t_* = minutes to nearest POI open at departure; inf if > max_minutes or unreachable.",
            "Deltas cap inf at max_minutes, so they are lower bounds.",
            "Car households use min(car, transit_walk).",
            "Destinations restricted to region PUMAs outside SF; within-SF and out-of-region movers are reported, not simulated.",
            "Out-of-state movers are invisible in the CA PUMS file.",
        ],
        "runtime_s": round(time.time() - t_start, 1),
        "households_input": str(households),
        "config": out_cfg,
    }
    for scen in summary["scenarios"]:
        summary.setdefault("headline", {})[scen] = headline(summary, scen)
    return res, summary


def attach_times(sim: pd.DataFrame, tab: pd.DataFrame, cap: float) -> pd.DataFrame:
    tcols = [f"t_{c}" for c in CLASSES]
    tr = tab[tab["mode"] == "transit_walk"].set_index(["cell", "depart"])[tcols]
    car = tab[tab["mode"] == "car"].set_index(["cell", "depart"])[tcols]
    for side, suffix in (("cell_before", "before"), ("cell_after", "after")):
        key = pd.MultiIndex.from_arrays([sim[side], sim["depart"]])
        t_tr = tr.reindex(key).to_numpy(np.float32)
        t_car = car.reindex(key).to_numpy(np.float32)
        t = np.where(sim["has_car"].to_numpy()[:, None], np.fmin(t_car, t_tr), t_tr)
        t = np.where(np.isnan(t) | (t > cap), np.float32(np.inf), t).astype(np.float32)
        for j, c in enumerate(CLASSES):
            sim[f"t_{c}_{suffix}"] = t[:, j]
            sim[f"acc30_{c}_{suffix}"] = t[:, j] <= 30.0
    return sim


CONTRACT_COLS = (["household_id", "scenario", "age_band", "income_band", "lang", "has_car", "disability",
                  "dest_puma", "dest_lat", "dest_lon", "mode", "depart"]
                 + [f"t_{c}_{s}" for c in CLASSES for s in ("before", "after")]
                 + [f"acc30_{c}_{s}" for c in CLASSES for s in ("before", "after")])
EXTRA_COLS = ["origin_household_id", "origin_lat", "origin_lon", "origin_puma", "event_type", "event_year", "weight",
              "cell_before", "cell_after"]


def to_contract(sim: pd.DataFrame) -> pd.DataFrame:
    res = sim[CONTRACT_COLS + EXTRA_COLS].copy()
    for c in ("household_id", "scenario", "age_band", "income_band", "lang", "dest_puma", "mode", "depart"):
        res[c] = res[c].astype(str)
    res["has_car"] = res["has_car"].astype(bool)
    res["disability"] = res["disability"].astype(bool)
    res["origin_puma"] = "06" + res["origin_puma"].astype(str)
    res["weight"] = res["weight"].astype("float32")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--households", required=True)
    ap.add_argument("--config", default="configs/sim.yaml")
    ap.add_argument("--backend", choices=["r5", "euclid"], help="override routing.backend")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(a.config)
    if a.backend:
        cfg["routing"]["backend"] = a.backend
    res, summary = run(Path(a.households), cfg)
    out_r, out_s = p(cfg, "out_results"), p(cfg, "out_summary")
    out_r.parent.mkdir(parents=True, exist_ok=True)
    out_s.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out_r, compression="snappy", index=False)
    out_s.write_text(json.dumps(summary, indent=1, default=str))
    log.info("wrote %s (%d rows) and %s", out_r, len(res), out_s)
    for h in summary["headline"].values():
        print(h)
    if not a.no_figures:
        from .figures import make_figures
        make_figures(res, summary, cfg)


if __name__ == "__main__":
    main()
