"""Synthetic `displaced_households.parquet` (CONTRACTS §4) so B can run before stream D delivers.

    python -m sim.patient.fixtures --out data/processed/fixtures/displaced_households.parquet --n 500
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# (GEOID-ish 2020 tract, neighborhood, lat, lon). Centers are real SF residential areas; GEOIDs are
# plausible but this is a FIXTURE, not a crosswalk.
TRACTS = [
    ("06075020800", "Mission", 37.7599, -122.4148),
    ("06075012402", "Tenderloin", 37.7843, -122.4140),
    ("06075011300", "Chinatown", 37.7946, -122.4078),
    ("06075045100", "Inner Richmond", 37.7802, -122.4645),
    ("06075032601", "Outer Sunset", 37.7550, -122.4950),
    ("06075023001", "Bayview", 37.7317, -122.3900),
    ("06075026001", "Excelsior", 37.7240, -122.4300),
    ("06075015900", "Western Addition", 37.7810, -122.4330),
]
EVENT_TYPES = ["ellis", "omi", "buyout", "other"]
SCENARIOS = ["realized_2019_2024", "unprotected__greedy_risk"]


def make_fixture(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = rng.integers(0, len(TRACTS), n)
    lat0 = np.array([TRACTS[i][2] for i in t])
    lon0 = np.array([TRACTS[i][3] for i in t])
    r = 0.0025  # ~250 m jitter
    df = pd.DataFrame(
        {
            "household_id": [f"fx_{i:05d}" for i in range(n)],
            "scenario": rng.choice(SCENARIOS, n, p=[0.6, 0.4]),
            "origin_lat": lat0 + rng.uniform(-r, r, n),
            "origin_lon": lon0 + rng.uniform(-r, r, n),
            "origin_tract": [TRACTS[i][0] for i in t],
            "event_year": rng.integers(2019, 2025, n).astype("int16"),
            "event_type": rng.choice(EVENT_TYPES, n, p=[0.25, 0.35, 0.3, 0.1]),
            "weight": rng.uniform(0.5, 2.5, n).astype("float32"),
        }
    )
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/fixtures/displaced_households.parquet")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    make_fixture(a.n, a.seed).to_parquet(out, compression="snappy", index=False)
    print(f"wrote {a.n} fixture households -> {out}")


if __name__ == "__main__":
    main()
