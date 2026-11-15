"""Session B LightGBM track: shared pieces for b_* candidates.

Value = buyout head + block-face head, both as expected households per year:
  buyout:     P(any buyout in t+1..t+h) / h  x  E[tenants | buyout]   (recent windows only; buyout data starts 2015-03)
  block-face: E[Ellis/OMI notices on the block-face per year] x bf_share
Training rows use recent snapshots with short horizons and recency weights (the regime shifted from
Ellis/OMI in 2010-15 to buyouts later).
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

BASE = ["units", "age", "owner_occ", "yrs_since_sale", "n_sales_10y", "log_last_jump", "sold_2y",
        "log_assessed_per_unit", "bf_share", "bf_rc_units", "bf_ellis_omi_3y", "bf_ellis_omi_10y",
        "bf_other_evict_3y", "bf_buyout_3y", "bldg_buyout_3y", "permits_3y", "remodel_3y", "unit_removal_10y",
        "log_permit_cost_3y", "cell_sales_3y"]
SALE = ["sold_1y", "sold_3y", "log_price_per_unit", "price_vs_nbhd", "sold2y_x_units", "nbhd_sales_rate"]


def add_feats(rows, sales):
    """Sale-recency features at the building from ctx.events['sales'] (all dated <= T)."""
    d = rows.copy()
    ys = d.yrs_since_sale
    d["sold_1y"] = (ys <= 1).astype("int8")
    d["sold_3y"] = (ys <= 3).astype("int8")
    d["sold2y_x_units"] = d.sold_2y * d.units
    # last sale price per unit as of snapshot t (assessed_after ~ purchase price under Prop 13)
    s = sales[["blklot", "sale_year", "assessed_after"]].dropna().sort_values("sale_year")
    k = d[["blklot", "t"]].reset_index().sort_values("t")
    m = pd.merge_asof(k, s.rename(columns={"sale_year": "t"}).astype({"t": k.t.dtype}), on="t", by="blklot",
                      direction="backward").set_index("index").sort_index()
    d["log_price_per_unit"] = np.log1p(m.assessed_after / d.units.clip(lower=1))
    nb = d.groupby(["t", "analysis_neighborhood"]).log_price_per_unit.transform("median")
    d["price_vs_nbhd"] = d.log_price_per_unit - nb
    d["nbhd_sales_rate"] = d.groupby(["t", "analysis_neighborhood"]).sold_2y.transform("mean")
    return d


def recency_w(t, t_max, half_life):
    return np.power(0.5, (t_max - t) / half_life) if half_life else np.ones(len(t))


def _train(params, X, y, w, rounds, mono=None):
    p = dict(learning_rate=0.03, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
             bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=0, num_threads=8)
    p.update(params)
    if mono is not None:
        p["monotone_constraints"] = [mono.get(c, 0) for c in X.columns]
    return lgb.train(p, lgb.Dataset(X, y, weight=w), rounds)


MONO = {"sold_2y": 1, "sold_1y": 1, "sold_3y": 1, "yrs_since_sale": -1, "bf_ellis_omi_3y": 1}


def fit_predict_twohead(ctx, h_bo=1, h_bf=2, t_bo_min=2014, t_bf_min=2011, hl_bo=0, hl_bf=2.0,
                        feats=BASE + SALE, mono=True, rounds_bo=300, rounds_bf=400, units=(3, 60), w_bf=1.0,
                        flat_tenants=False, return_parts=False):
    rows = add_feats(ctx.rows, ctx.events["sales"])
    ctx_rows, ctx.rows = ctx.rows, rows
    try:
        sc = ctx.score
        lab_bo, lab_bf = ctx.labels(h_bo), ctx.labels(h_bf)
    finally:
        ctx.rows = ctx_rows
    M = MONO if mono else None
    T = ctx.T

    # buyout head: P(any buyout in window), trained on buyout-era windows only
    b = lab_bo[(lab_bo.t >= t_bo_min) & lab_bo.units.between(*units)]
    mb = _train(dict(objective="binary"), b[feats], b.y_bldg, recency_w(b.t, T - h_bo, hl_bo), rounds_bo, M)
    pos = b[b.y_bldg == 1]
    if flat_tenants:   # buyouts are usually 1-2 households regardless of building size
        e_ten = pos.y_buyout_tenants.mean() if len(pos) else 1.5
    else:
        ten = (pos.y_buyout_tenants / pos.units).median() if len(pos) else 0.2   # tenants per unit given a buyout
        e_ten = np.clip(ten * sc.units.values, 1, None)
    v_bo = mb.predict(sc[feats]) / h_bo * e_ten

    # block-face head: Poisson on notices on the block-face, apportioned by unit share
    f = lab_bf[(lab_bf.t >= t_bf_min) & lab_bf.units.between(*units)]
    mf = _train(dict(objective="poisson"), f[feats], f.y_bf_notices, recency_w(f.t, T - h_bf, hl_bf), rounds_bf, M)
    v_bf = mf.predict(sc[feats]) / h_bf * sc.bf_share.fillna(0).values

    v = v_bo + w_bf * v_bf
    return (v, v_bo, v_bf) if return_parts else v


def heuristic_raw(df):
    return (df.yrs_since_sale <= 2) * 10 + df.bf_ellis_omi_3y.clip(0, 5)


def calibrated_heuristic(ctx, h=1, t_min=2014, prior=20):
    """The rule of thumb's 12 cells (sold_2y x clipped bf notices), each mapped to mean households/yr on recent
    windows, shrunk toward the overall mean. A learned version of the hand-weighted rule."""
    lab = ctx.labels(h)
    lab = lab[lab.t >= t_min]
    key = heuristic_raw(lab)
    g = lab.groupby(key).y_val.agg(["sum", "count"])
    mu = lab.y_val.mean()
    rate = (g["sum"] + prior * mu) / (g["count"] + prior) / h
    return heuristic_raw(ctx.score).map(rate).fillna(mu / h).values


def blend(ctx, lam=0.5, **kw):
    """lam * model + (1 - lam) * calibrated heuristic, both in households/yr."""
    return lam * fit_predict_twohead(ctx, **kw) + (1 - lam) * calibrated_heuristic(ctx)
