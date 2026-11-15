"""Displacement-risk heads + baselines for the end-of-2018 decision.

Time discipline: every label used for fitting ends by 2018 (snapshot t <= 2015, horizon 3).
  rounds chosen by fit on t<=2014 -> validate t=2015; final model refit on all t<=2015 with those rounds.
  binary heads calibrated (isotonic) on out-of-time 2015 predictions of the <=2014 model.
Test: t=2018 (outcomes 2019-2021). Heads:
  y_bf   binary  any Ellis/OMI on block-face        (train t 2009-2015)
  y_bldg binary  any buyout at the parcel            (train t 2013-2015; buyout data starts 2015-03)
  y_val  tweedie realized households (optimizer objective) (train t 2014-2015)

python -m models.risk.train [--emb data/processed/cell_embeddings.parquet]
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D = Path("data/processed")
REP = Path("reports/risk")
FEATS = ["units", "age", "owner_occ", "yrs_since_sale", "n_sales_10y", "log_last_jump", "sold_2y",
         "log_assessed_per_unit", "bf_share", "bf_rc_units", "bf_ellis_omi_3y", "bf_ellis_omi_10y",
         "bf_other_evict_3y", "bf_buyout_3y", "bldg_buyout_3y", "permits_3y", "remodel_3y", "unit_removal_10y",
         "log_permit_cost_3y", "cell_sales_3y", "supervisor_district"]
HEADS = {"y_bf": ("binary", range(2009, 2016)), "y_bldg": ("binary", range(2013, 2016)),
         "y_val": ("tweedie", range(2014, 2016))}
VAL_T, TEST_T = 2015, 2018
BASE = dict(learning_rate=0.03, num_leaves=15, min_data_in_leaf=300, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=0)


def add_emb(d, path, k=32):
    from sklearn.decomposition import PCA
    e = pd.read_parquet(path)
    e = e[e.cutoff == 2018]
    cols = [f"e{i}" for i in range(256) if f"e{i}" in e]
    pca = PCA(k, random_state=0).fit(e[e.year <= 2015][cols].values)
    z = pd.DataFrame(pca.transform(e[cols].values), columns=[f"emb{i}" for i in range(k)])
    z["cell"], z["t"] = e.cell.values, e.year.values
    return d.merge(z, on=["cell", "t"], how="left"), [f"emb{i}" for i in range(k)]


def fit_head(d, feats, y, obj, years):
    p = dict(BASE, objective=obj)
    if obj == "tweedie":
        p["tweedie_variance_power"] = 1.3
    tr, va = d[d.t.isin(years) & (d.t < VAL_T)], d[d.t == VAL_T]
    cat = ["supervisor_district"]
    m0 = lgb.train(p, lgb.Dataset(tr[feats], tr[y], categorical_feature=cat), 3000,
                   valid_sets=[lgb.Dataset(va[feats], va[y], categorical_feature=cat)],
                   callbacks=[lgb.early_stopping(150, verbose=False)])
    rounds = max(50, m0.best_iteration)
    full = d[d.t.isin(years)]
    m = lgb.train(p, lgb.Dataset(full[feats], full[y], categorical_feature=cat), rounds)
    iso = None
    if obj == "binary":
        iso = IsotonicRegression(out_of_bounds="clip").fit(m0.predict(va[feats]), va[y])
    return m, iso, rounds


def rule(d):
    """Organizers' rule of thumb: sold in last 2 years, multi-unit, pre-1979 (universe already pre-1979)."""
    return ((d.yrs_since_sale <= 2).astype(float) + 1e-3 * d.units.clip(0, 100)).values


def bin_metrics(y, p, groups, n_boot=300, seed=0):
    y, p = np.asarray(y), np.asarray(p)
    out = {"pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p), "base_rate": float(y.mean())}
    k = max(1, int(0.05 * len(y)))
    out["lift_at_5pct"] = float(y[np.argsort(-p)[:k]].mean() / y.mean())
    out["pr_auc_ci"] = _boot(lambda ii: average_precision_score(y[ii], p[ii]) if y[ii].sum() else np.nan,
                             groups, n_boot, seed)
    return out


def val_metrics(y, p, groups, n_boot=300, seed=0):
    y, p = np.asarray(y), np.asarray(p)
    def cap(ii, f=0.05):
        k = max(1, int(f * len(ii)))
        top = ii[np.argsort(-p[ii])[:k]]
        return y[top].sum() / max(y[ii].sum(), 1e-9)
    out = {"capture_top5pct": float(cap(np.arange(len(y)))), "spearman": float(spearmanr(y, p).statistic),
           "pred_total": float(p.sum()), "obs_total": float(y.sum())}
    out["capture_ci"] = _boot(cap, groups, n_boot, seed)
    return out


def _boot(fn, groups, n_boot, seed):
    rng = np.random.default_rng(seed)
    g = pd.Series(groups).astype("category").cat.codes.values
    by = pd.Series(np.arange(len(g))).groupby(g).apply(np.array)
    keys = by.index.values
    bs = [fn(np.concatenate(by.loc[rng.choice(keys, len(keys))].values)) for _ in range(n_boot)]
    bs = [b for b in bs if b == b]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def run(emb=None):
    d = pd.read_parquet(D / "risk_dataset.parquet")
    d["supervisor_district"] = pd.to_numeric(d.supervisor_district, errors="coerce").fillna(-1).astype(int)
    feats, tag = list(FEATS), "gbm"
    if emb:
        d, ecols = add_emb(d, emb)
        feats, tag = feats + ecols, "gbm_emb"
    te = d[d.t == TEST_T].copy()
    groups = te.bf.fillna(te.blklot).values
    res, cols = {}, {}
    for y, (obj, years) in HEADS.items():
        m, iso, rounds = fit_head(d, feats, y, obj, years)
        raw = m.predict(te[feats])
        tr = d[d.t.isin(years)]
        if obj == "binary":
            lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1)).fit(tr[FEATS].fillna(-1), tr[y])
            base = lr.predict_proba(te[FEATS].fillna(-1))[:, 1]
            cands = {tag: raw, "logreg": base, "recently_sold": rule(te)}
            res[y] = {k: bin_metrics(te[y], v, groups) for k, v in cands.items()}
            cal = iso.predict(raw)
            res[y]["calibration"] = {"mean_pred": float(cal.mean()), "mean_obs": float(te[y].mean())}
            cols[f"p_{y}_{tag}"] = cal
        else:
            pr = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-2, max_iter=3000)).fit(tr[FEATS].fillna(-1), tr[y])
            cands = {tag: raw, "poisson_glm": pr.predict(te[FEATS].fillna(-1)), "recently_sold": rule(te)}
            res[y] = {k: val_metrics(te[y], v, groups) for k, v in cands.items()}
            cols[f"ev_{tag}"] = raw
        res[y]["rounds"] = rounds
        res[y]["top_features"] = pd.Series(m.feature_importance("gain"), index=feats).nlargest(8).round(0).to_dict()
    te = te.assign(**cols)
    REP.mkdir(parents=True, exist_ok=True)
    te.to_parquet(D / f"risk_scores_2018_{tag}.parquet", index=False)
    json.dump(res, open(REP / f"metrics_{tag}.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb")
    a = ap.parse_args()
    for y, v in run(a.emb).items():
        print(y, "rounds", v["rounds"], "top", list(v["top_features"])[:5])
        for k, m in v.items():
            if isinstance(m, dict) and k not in ("top_features",):
                print(f"  {k:14s} " + " ".join(f"{kk}={vv:.3f}" if isinstance(vv, float) else f"{kk}={vv}"
                                               for kk, vv in m.items()))
