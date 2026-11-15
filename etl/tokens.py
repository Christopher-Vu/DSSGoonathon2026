"""Hex tokens for the world model (CONTRACTS §1).

python -m etl.tokens --cutoff 2018   # public only
python -m etl.tokens --cutoff 2025   # public; sponsor types appended by etl.sponsor_tokens
"""
import argparse
import math
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import yaml

INT = Path("data/interim")
OUT = Path("data/processed")
PUBLIC = ["parcel", "sale", "permit", "eviction_bf", "buyout"]
SPONSOR = ["poi", "reddit", "review", "zillow"]
EDGE_M = h3.average_hexagon_edge_length(9, unit="m")
FIRST_YEAR = 2007

FEATURES = {
    "parcel": ["log_units", "age", "log_assessed", "log_assessed_per_unit", "yrs_since_sale", "rent_ctrl",
               "owner_occ", "stories", "log_area", "log_lot_area", "is_mres", "is_sres", "is_comm",
               "n_sales_10y", "log_last_jump", "beds_per_unit"],
    "sale": ["log_jump", "log_assessed_after", "log_units", "yrs_since_prev_sale", "rent_ctrl"],
    "permit": ["log_cost", "unit_delta", "adu", "demolition", "new_construction", "alteration", "remodel_kw",
               "seismic_kw", "completed", "log_existing_units", "unit_removal", "change_of_use"],
    "eviction_bf": ["ellis", "omi", "non_payment", "breach", "nuisance", "illegal_use", "capital_improvement",
                    "demolition", "substantial_rehab", "condo_conversion", "other_cause"],
    "buyout": ["log_amount", "tenants", "is_agreement"],
}


def offsets(cells, lat, lon):
    c = np.array([h3.cell_to_latlng(x) for x in cells])
    dy = (lat - c[:, 0]) * 110_540 / EDGE_M
    dx = (lon - c[:, 1]) * 111_320 * np.cos(np.radians(c[:, 0])) / EDGE_M
    return np.clip(dx, -1, 1).astype("float32"), np.clip(dy, -1, 1).astype("float32")


def frame(tt, cell, year, feats: dict, lat, lon, entity=None, text_idx=None):
    df = pd.DataFrame({"cell": cell, "year": np.asarray(year, "int16"), "token_type": tt})
    df["entity_id"] = entity if entity is not None else None
    for i, name in enumerate(FEATURES[tt]):
        df[f"f{i}"] = np.asarray(feats[name], dtype="float32")
    for i in range(len(FEATURES[tt]), 16):
        df[f"f{i}"] = np.float32("nan")
    df["x_off"], df["y_off"] = offsets(df.cell.values, np.asarray(lat), np.asarray(lon))
    df["text_idx"] = np.int32(-1) if text_idx is None else np.asarray(text_idx, "int32")
    return df


def parcel_tokens(sp):
    sp = sp.dropna(subset=["cell"])
    u = sp.units.clip(lower=0)
    f = dict(
        log_units=np.log1p(u), age=sp.age, log_assessed=np.log1p(sp.assessed),
        log_assessed_per_unit=np.log1p(sp.assessed / u.clip(lower=1)), yrs_since_sale=sp.yrs_since_sale.clip(0, 60),
        rent_ctrl=sp.rent_ctrl, owner_occ=sp.owner_occ, stories=sp.number_of_stories.clip(0, 60),
        log_area=np.log1p(sp.property_area), log_lot_area=np.log1p(sp.lot_area),
        is_mres=sp.use_code.eq("MRES"), is_sres=sp.use_code.eq("SRES"),
        is_comm=sp.use_code.fillna("").str.startswith("COM"), n_sales_10y=sp.n_sales_10y,
        log_last_jump=np.log(sp.last_jump.clip(0.05, 50)), beds_per_unit=(sp.number_of_bedrooms / u.clip(lower=1)).clip(0, 10),
    )
    return frame("parcel", sp.cell.values, sp.year.values, f, sp.lat.values, sp.lon.values, sp.blklot.values)


def sale_tokens(sales, geo):
    s = sales.merge(geo, on="blklot", how="inner").sort_values(["blklot", "sale_date"])
    s["prev"] = s.groupby("blklot").sale_year.shift()
    f = dict(log_jump=np.log(s.jump.clip(0.05, 50)), log_assessed_after=np.log1p(s.assessed_after),
             log_units=np.log1p(s.units), yrs_since_prev_sale=(s.sale_year - s.prev), rent_ctrl=s.rent_ctrl)
    return frame("sale", s.cell.values, s.sale_year.values, f, s.lat.values, s.lon.values, s.blklot.values)


def permit_tokens(per, geo):
    p = per.copy()
    p["blklot"] = p.block.astype(str).str.zfill(4) + p.lot.astype(str).str.zfill(3)
    p = p.merge(geo, on="blklot", how="inner")
    p["year"] = pd.to_datetime(p.filed_date.fillna(p.permit_creation_date), errors="coerce").dt.year
    p = p.dropna(subset=["year"])
    cost = pd.to_numeric(p.revised_cost, errors="coerce").fillna(pd.to_numeric(p.estimated_cost, errors="coerce"))
    ex = pd.to_numeric(p.existing_units, errors="coerce")
    pr = pd.to_numeric(p.proposed_units, errors="coerce")
    d = p.description.fillna("").str.lower()
    t = pd.to_numeric(p.permit_type, errors="coerce")
    f = dict(log_cost=np.log1p(cost.clip(lower=0)), unit_delta=(pr - ex).clip(-50, 500),
             adu=p.adu.astype(str).str.lower().isin(["true", "y", "yes"]) | d.str.contains(r"\badu\b|accessory dwelling"),
             demolition=t.eq(6) | d.str.contains("demoli"), new_construction=t.isin([1, 2]), alteration=t.isin([3, 8]),
             remodel_kw=d.str.contains("kitchen|bath|remodel|renovat"), seismic_kw=d.str.contains("seismic|soft story|soft-story"),
             completed=p.status.fillna("").str.lower().eq("complete"), log_existing_units=np.log1p(ex.clip(lower=0)),
             unit_removal=(pr < ex), change_of_use=(p.existing_use.fillna("") != p.proposed_use.fillna("")) & p.proposed_use.notna())
    out = frame("permit", p.cell.values, p.year.values, f, p.lat.values, p.lon.values, p.blklot.values)
    return out, p.description.fillna("").values


def eviction_tokens(ev, bf):
    e = ev[ev.bf_ok].merge(bf[["bf", "lat", "lon", "cell"]], on="bf", suffixes=("_pt", ""))
    e = e.dropna(subset=["year"])
    f = {k: e[c] for k, c in zip(FEATURES["eviction_bf"],
         ["ellis_act_withdrawal", "owner_move_in", "non_payment", "breach", "nuisance", "illegal_use",
          "capital_improvement", "demolition", "substantial_rehab", "condo_conversion", "other_cause"])}
    return frame("eviction_bf", e.cell.values, e.year.values, f, e.lat.values, e.lon.values)


def buyout_tokens(bo, geo):
    b = bo.dropna(subset=["blklot", "year"]).drop_duplicates("case_number").merge(geo, on="blklot")
    f = dict(log_amount=np.log1p(b.amount), tenants=b.tenants, is_agreement=b.is_agreement)
    return frame("buyout", b.cell.values, b.year.values, f, b.lat.values, b.lon.values, b.blklot.values)


def standardize(df, cutoff):
    for tt, g in df.groupby("token_type"):
        fit = g[g.year <= cutoff]
        for i in range(len(FEATURES.get(tt, []))):
            c = f"f{i}"
            mu, sd = np.nanmean(fit[c]), np.nanstd(fit[c])
            df.loc[g.index, c] = ((g[c] - mu) / (sd if sd > 1e-6 else 1.0)).astype("float32")
    return df


def build(cutoff: int, embed_text: bool):
    sp = pd.read_parquet(INT / "spine.parquet")
    sp = sp[(sp.year >= FIRST_YEAR) & (sp.year <= cutoff)]
    geo = (sp.dropna(subset=["cell"]).sort_values("year").drop_duplicates("blklot", keep="last")
           [["blklot", "cell", "lat", "lon", "units", "rent_ctrl"]])
    sales = pd.read_parquet(INT / "sales.parquet")
    sales = sales[(sales.sale_year >= FIRST_YEAR) & (sales.sale_year <= cutoff)]
    per = pd.read_parquet(Path("data/raw/datasf/permits.parquet"))
    ev = pd.read_parquet(INT / "eviction_events.parquet")
    ev = ev[(ev.year >= FIRST_YEAR) & (ev.year <= cutoff)]
    bf = pd.read_parquet(INT / "blockfaces.parquet")
    bo = pd.read_parquet(INT / "buyout_events.parquet")
    bo = bo[bo.year <= cutoff]

    ptok, desc = permit_tokens(per, geo)
    keep = (ptok.year >= FIRST_YEAR) & (ptok.year <= cutoff)
    ptok, desc = ptok[keep.values].reset_index(drop=True), desc[keep.values]
    if embed_text:
        uniq, inv = np.unique(desc, return_inverse=True)
        emb = embed(list(uniq))
        ptok["text_idx"] = np.where(desc != "", inv, -1).astype("int32")
        OUT.mkdir(parents=True, exist_ok=True)
        np.save(OUT / f"text_emb_{cutoff}.npy", emb.astype("float16"))
    parts = [parcel_tokens(sp), sale_tokens(sales, geo), ptok, eviction_tokens(ev, bf), buyout_tokens(bo, geo)]
    df = pd.concat(parts, ignore_index=True)
    assert df.year.max() <= cutoff and set(df.token_type) <= set(PUBLIC)
    df = standardize(df, cutoff)
    d = OUT / "tokens" / f"cutoff={cutoff}"
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("part-*.parquet"):
        f.unlink()
    for k, (tt, g) in enumerate(df.groupby("token_type")):
        g.to_parquet(d / f"part-{k:03d}-{tt}.parquet", index=False)
    Path("configs").mkdir(exist_ok=True)
    yaml.safe_dump({**FEATURES, **{k: v for k, v in sponsor_features().items()}},
                   open("configs/token_features.yaml", "w"), sort_keys=False)
    return df.groupby("token_type").size().to_dict(), df.cell.nunique()


def sponsor_features():
    p = Path("configs/token_features_sponsor.yaml")
    return yaml.safe_load(open(p)) if p.exists() else {}


def embed(texts, bs=512):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return m.encode(texts, batch_size=bs, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, required=True)
    ap.add_argument("--no-text", action="store_true")
    a = ap.parse_args()
    print(build(a.cutoff, not a.no_text))
