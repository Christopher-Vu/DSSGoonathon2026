"""Pull DataSF tables to data/raw/datasf/<name>.parquet via SODA (selected columns, paged)."""
import argparse
import io
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

BASE = "https://data.sf.gov/resource/{id}.csv"
OUT = Path("data/raw/datasf")
PAGE = 50_000

TABLES = {
    "evictions": ("5cei-gny5", "*", None),
    "buyouts": ("wmam-7g8d", "*", None),
    "parcels": ("acdm-wktn",
                "blklot,block_num,lot_num,from_address_num,to_address_num,street_name,street_type,"
                "active,date_rec_add,date_rec_drop,centroid_latitude,centroid_longitude,"
                "supervisor_district,analysis_neighborhood,zoning_code", None),
    "permits": ("i98e-djp9",
                "permit_number,permit_type,permit_type_definition,permit_creation_date,block,lot,"
                "street_number,street_name,street_suffix,description,status,filed_date,issued_date,"
                "completed_date,estimated_cost,revised_cost,existing_use,existing_units,proposed_use,"
                "proposed_units,adu", None),
    "mohcd_portfolio": ("pyxv-n29e", "*", None),
    "mohcd_pipeline": ("aaxw-2cb8", "*", None),
    "taxroll": ("wv5m-vpq2",
                "closed_roll_year,property_location,parcel_number,block,lot,use_code,use_definition,"
                "property_class_code,property_class_code_definition,year_property_built,number_of_bedrooms,"
                "number_of_rooms,number_of_stories,number_of_units,zoning_code,construction_type,property_area,"
                "lot_area,percent_of_ownership,exemption_code,status_code,homeowner_exemption_value,"
                "current_sales_date,assessed_improvement_value,assessed_land_value,analysis_neighborhood,"
                "supervisor_district", "closed_roll_year"),
}


def fetch(id_, select, where=None):
    frames, off, s = [], 0, requests.Session()
    while True:
        p = {"$select": select, "$limit": PAGE, "$offset": off, "$order": ":id"}
        if where:
            p["$where"] = where
        r = s.get(BASE.format(id=id_), params=p, timeout=300)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), dtype=str)
        frames.append(df)
        if len(df) < PAGE:
            break
        off += PAGE
    return pd.concat(frames, ignore_index=True)


def pull(name):
    id_, select, part = TABLES[name]
    out = OUT / f"{name}.parquet"
    if out.exists():
        print(name, "exists", flush=True)
        return
    if part:  # parallel by year
        years = range(2007, 2026)
        with ThreadPoolExecutor(6) as ex:
            dfs = list(ex.map(lambda y: fetch(id_, select, f"{part}='{y}'"), years))
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = fetch(id_, select)
    df.to_parquet(out, index=False)
    print(name, len(df), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=list(TABLES))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for n in a.names:
        try:
            pull(n)
        except Exception as e:  # keep going
            print(n, "FAILED", e, file=sys.stderr, flush=True)
