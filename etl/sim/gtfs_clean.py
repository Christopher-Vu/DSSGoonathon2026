"""Make GTFS feeds palatable to r5: fill empty stop_name (511 pathway nodes have none), drop pathways/levels.

    python -m etl.sim.gtfs_clean [--config configs/sim.yaml]

Reads every zip in <data>/raw/sim/gtfs, writes cleaned copies to paths.gtfs_dir (processed/sim/gtfs).
"""
from __future__ import annotations

import argparse
import io
import zipfile

import pandas as pd

from sim.patient.config import data_root, load_config, p

DROP = {"pathways.txt", "levels.txt"}


def clean(src, dst) -> dict:
    fixed = {}
    with zipfile.ZipFile(src) as zi, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zo:
        for name in zi.namelist():
            base = name.rsplit("/", 1)[-1]
            if base in DROP or not base.endswith(".txt"):
                continue
            data = zi.read(name)
            if base == "stops.txt":
                s = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
                empty = s["stop_name"].str.strip() == ""
                s.loc[empty, "stop_name"] = s.loc[empty, "stop_id"]
                if "level_id" in s:
                    s["level_id"] = ""
                fixed["stop_name_filled"] = int(empty.sum())
                data = s.to_csv(index=False).encode()
            zo.writestr(base, data)
    return fixed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sim.yaml")
    cfg = load_config(ap.parse_args().config)
    out = p(cfg, "gtfs_dir")
    out.mkdir(parents=True, exist_ok=True)
    for z in sorted((data_root(cfg) / "raw/sim/gtfs").glob("*.zip")):
        print(z.name, clean(z, out / z.name))


if __name__ == "__main__":
    main()
