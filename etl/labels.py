"""Event tables: buyouts -> blklot, evictions -> block-face, Small Sites picks -> blklot + acquisition year."""
from pathlib import Path

import numpy as np
import pandas as pd

from etl.addr import blockface_from_eviction, split_addr
from etl.geo import RAW, address_index, blockfaces, match_addresses, parcels

INT = Path("data/interim")
EVICT_FLAGS = ["ellis_act_withdrawal", "owner_move_in", "non_payment", "breach", "nuisance", "illegal_use",
               "capital_improvement", "demolition", "substantial_rehab", "condo_conversion", "other_cause"]


def roll_index():
    r = pd.read_parquet(INT / "spine.parquet", columns=["blklot", "street", "lo", "hi"])
    return r.dropna().drop_duplicates()


def build():
    p = parcels()
    idx = address_index(p, roll_index())
    bf, pbf = blockfaces(p)

    # buyouts
    b = pd.read_parquet(RAW / "buyouts.parquet")
    m = match_addresses(b.address, idx)
    b["blklots"] = m.blklot.values
    b["join_how"] = m.how.values
    d = pd.to_datetime(b.buyout_agreement_date, errors="coerce")
    d = d.fillna(pd.to_datetime(b.pre_buyout_disclosure_declaration_date, errors="coerce"))
    b["year"] = d.dt.year.astype("Int16")
    b["tenants"] = pd.to_numeric(b.number_of_tenants, errors="coerce").fillna(1)
    b["amount"] = pd.to_numeric(b.buyout_amount, errors="coerce")
    b["is_agreement"] = b.buyout_agreement_date.notna()
    bx = b.explode("blklots").rename(columns={"blklots": "blklot"})
    nlots = bx.groupby("case_number").blklot.transform("count").clip(lower=1)
    bx["tenants_share"] = bx.tenants / nlots
    bx[["case_number", "address", "blklot", "join_how", "year", "tenants", "tenants_share", "amount",
        "is_agreement"]].to_parquet(INT / "buyout_events.parquet", index=False)

    # evictions -> block-face
    e = pd.read_parquet(RAW / "evictions.parquet")
    e["bf"] = e.address.map(blockface_from_eviction)
    e["year"] = pd.to_datetime(e.file_date, errors="coerce").dt.year.astype("Int16")
    for c in EVICT_FLAGS:
        e[c] = e[c].astype(str).str.lower().eq("true")
    e["ellis_omi"] = e.ellis_act_withdrawal | e.owner_move_in
    e["bf_ok"] = e.bf.isin(set(bf.bf))
    pt = e.shape.str.extract(r"POINT \(([-\d.]+) ([-\d.]+)\)").astype(float)
    e["lon"], e["lat"] = pt[0], pt[1]
    e[["eviction_id", "address", "bf", "bf_ok", "year", "lat", "lon", "ellis_omi"] + EVICT_FLAGS].to_parquet(
        INT / "eviction_events.parquet", index=False)
    bf.to_parquet(INT / "blockfaces.parquet", index=False)
    pbf.to_parquet(INT / "parcel_blockface.parquet", index=False)

    # Small Sites picks
    pf = pd.read_parquet(RAW / "mohcd_portfolio.parquet")
    ss = pf[pf.housing_program_program_name.eq("Small Sites Program")].copy()
    m = match_addresses(ss.marketing_address, idx)
    ss["blklots"], ss["join_how"] = m.blklot.values, m.how.values
    pl = pd.read_parquet(RAW / "mohcd_pipeline.parquet")
    pl = pl[pl.program_area.fillna("").str.contains("Small Sites", case=False)
            | pl.general_housing_program.fillna("").str.contains("Small Sites", case=False)].copy()
    pl["blklots"] = [[f"{int(bk):04d}{str(lt).zfill(3)}"] if str(bk).isdigit() else None
                     for bk, lt in zip(pl.block, pl.lot)]
    sales = pd.read_parquet(INT / "sales.parquet")
    rows = []
    for src, df, addr in [("portfolio", ss, "marketing_address"), ("pipeline", pl, "project_name")]:
        for _, r in df.iterrows():
            lots = r.blklots if isinstance(r.blklots, list) else []
            sl = sales[sales.blklot.isin(lots) & (sales.sale_year >= 2014)]
            acq = sl.sale_year.max() if len(sl) else np.nan
            cost = sl.sort_values("sale_year").assessed_after.iloc[-1] if len(sl) else np.nan
            rows.append(dict(source=src, name=r[addr], blklots=lots, join_how=r.get("join_how", "blocklot"),
                             units=pd.to_numeric(r.get("total_project_units"), errors="coerce"),
                             acq_year=acq, cost_assessed=cost,
                             mohcd_date=r.get("estimated_actual_construction", r.get("ssp_rad_acquisition_date"))))
    pd.DataFrame(rows).to_parquet(INT / "ssp_picks.parquet", index=False)

    rep = {
        "buyout_join_rate": float(m_rate(b.join_how)),
        "buyout_join_how": b.join_how.value_counts(dropna=False).to_dict(),
        "eviction_bf_rate": float(e.bf_ok.mean()),
        "eviction_bf_rate_ellis_omi_2010+": float(e[e.ellis_omi & (e.year >= 2010)].bf_ok.mean()),
        "ssp_join_rate": float(pd.Series([len(x) > 0 for x in pd.DataFrame(rows).blklots]).mean()),
        "ssp_with_acq_year": int(pd.DataFrame(rows).acq_year.notna().sum()),
    }
    return rep


def m_rate(how):
    return how.notna().mean()


if __name__ == "__main__":
    import json
    rep = build()
    Path("reports").mkdir(exist_ok=True)
    json.dump(rep, open("reports/join_rates.json", "w"), indent=1, default=str)
    print(json.dumps(rep, indent=1, default=str))
