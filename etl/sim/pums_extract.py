"""Slim the CA PUMS 2019-2023 5-yr CSVs into two parquet files.

    python -m etl.sim.pums_extract [--config configs/sim.yaml]

Writes <data>/processed/sim/pums_households.parquet and pums_persons.parquet with a `year` column
(survey year = first 4 chars of SERIALNO). No filtering on year here; the sim filters by config.
"""
from __future__ import annotations

import argparse
import zipfile

import pandas as pd

from sim.patient.config import load_config, p

H_COLS = ["SERIALNO", "PUMA", "ADJINC", "WGTP", "NP", "TYPEHUGQ", "TEN", "VEH", "HINCP"]
P_COLS = ["SERIALNO", "SPORDER", "PUMA", "ADJINC", "PWGTP", "AGEP", "MIG", "RELSHIPP", "DIS",
          "LANP", "MIGPUMA", "MIGSP"]
STR_COLS = {"SERIALNO", "PUMA", "MIGPUMA", "MIGSP", "LANP", "TEN", "TYPEHUGQ", "MIG", "RELSHIPP", "DIS"}


def _read(zpath, cols):
    with zipfile.ZipFile(zpath) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        with z.open(name) as f:
            df = pd.read_csv(f, usecols=cols, dtype={c: str for c in cols if c in STR_COLS}, low_memory=False)
    df["year"] = df["SERIALNO"].str[:4].astype("int16")
    for c in ("PUMA", "MIGPUMA"):
        if c in df:
            df[c] = df[c].str.zfill(5)
    if "MIGSP" in df:
        df["MIGSP"] = df["MIGSP"].str.zfill(3)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sim.yaml")
    cfg = load_config(ap.parse_args().config)
    out = p(cfg, "interim")
    out.mkdir(parents=True, exist_ok=True)
    h = _read(p(cfg, "pums_households"), H_COLS)
    h.to_parquet(out / "pums_households.parquet", index=False)
    print("households", h.shape, h["year"].value_counts().sort_index().to_dict())
    pp = _read(p(cfg, "pums_persons"), P_COLS)
    pp.to_parquet(out / "pums_persons.parquet", index=False)
    print("persons", pp.shape)


if __name__ == "__main__":
    main()
