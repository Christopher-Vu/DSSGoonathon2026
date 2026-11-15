"""Two-head displacement-risk model + baselines. Train t<=2014, calibrate t=2015, test t=2018.

python -m models.risk.train [--emb data/processed/cell_embeddings.parquet]
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D = Path("data/processed")
REP = Path("reports/risk")
FEATS = ["units", "age", "owner_occ", "yrs_since_sale", "n_sales_10y", "log_last_jump", "sold_2y",
         "log_assessed_per_unit", "bf_share", "bf_rc_units", "bf_ellis_omi_3y", "bf_ellis_omi_10y",
         "bf_other_evict_3y", "bf_buyout_3y", "bldg_buyout_3y", "permits_3y", "remodel_3y", "unit_removal_10y",
         "log_permit_cost_3y", "cell_sales_3y", "supervisor_district"]
HEADS = {"y_bf": dict(train=range(2009, 2015)), "y_bldg": dict(train=range(2013, 2015))}
CAL_T, TEST_T = 2015, 2018
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0)


def add_emb(d, path, k=32):
    e = pd.read_parquet(path)
    e = e[e.cutoff == 2018]
    cols = [c for c in e.columns if c.startswith("e")][:256]
    # PCA to k dims fit on years <= 2014 to keep the GBM small
    from sklearn.decomposition import PCA
    fit = e[e.year <= 2014][cols].values
    pca = PCA(k, random_state=0).fit(fit)
    z = pd.DataFrame(pca.transform(e[cols].values), columns=[f"emb{i}" for i in range(k)])
    z["cell"], z["t"] = e.cell.values, e.year.values
    return d.merge(z, on=["cell", "t"], how="left"), list(z.columns[:k])


def fit_gbm(tr, cal, feats, y):
    ds = lgb.Dataset(tr[feats], tr[y], categorical_feature=["supervisor_district"])
    dv = lgb.Dataset(cal[feats], cal[y], reference=ds)
    m = lgb.train(PARAMS, ds, 2000, valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False)])
    iso = IsotonicRegression(out_of_bounds="clip").fit(m.predict(cal[feats]), cal[y])
    return m, iso


def recently_sold_rule(d):
    return ((d.yrs_since_sale <= 2) & (d.units >= 2)).astype(float) + 1e-3 * d.units  # tie-break by size


def metrics(y, p, groups, n_boot=300, seed=0):
    out = {"pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p), "base_rate": float(np.mean(y))}
    k = max(1, int(0.05 * len(y)))
    top = np.argsort(-p)[:k]
    out["prec_at_5pct"] = float(np.mean(np.asarray(y)[top]))
    out["lift_at_5pct"] = out["prec_at_5pct"] / out["base_rate"]
    rng = np.random.default_rng(seed)
    g = pd.Series(groups).astype("category").cat.codes.values
    ug = np.unique(g)
    idx_by = pd.Series(np.arange(len(g))).groupby(g).apply(np.array)
    bs = []
    for _ in range(n_boot):  # block-face cluster bootstrap
        ii = np.concatenate(idx_by.loc[rng.choice(ug, len(ug))].values)
        if np.asarray(y)[ii].sum() == 0:
            continue
        bs.append(average_precision_score(np.asarray(y)[ii], np.asarray(p)[ii]))
    out["pr_auc_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return out


def run(emb=None):
    d = pd.read_parquet(D / "risk_dataset.parquet")
    d["supervisor_district"] = pd.to_numeric(d.supervisor_district, errors="coerce").fillna(-1).astype(int)
    feats = list(FEATS)
    tag = "gbm"
    if emb:
        d, ecols = add_emb(d, emb)
        feats += ecols
        tag = "gbm_emb"
    res, preds = {}, {}
    te = d[d.t == TEST_T].copy()
    for y, cfg in HEADS.items():
        tr, cal = d[d.t.isin(cfg["train"])], d[d.t == CAL_T]
        m, iso = fit_gbm(tr, cal, feats, y)
        p = iso.predict(m.predict(te[feats]))
        raw = m.predict(te[feats])
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
        lr.fit(tr[FEATS].fillna(-1), tr[y])
        cands = {tag: raw, "logreg": lr.predict_proba(te[FEATS].fillna(-1))[:, 1],
                 "recently_sold": recently_sold_rule(te).values, "base_rate": np.full(len(te), tr[y].mean())}
        res[y] = {k: metrics(te[y].values, v, te.bf.fillna(te.blklot).values) for k, v in cands.items()
                  if k != "base_rate"}
        res[y]["base_rate"] = {"pr_auc": float(te[y].mean())}
        res[y]["calibration"] = {"mean_pred": float(p.mean()), "mean_obs": float(te[y].mean())}
        imp = pd.Series(m.feature_importance("gain"), index=feats).sort_values(ascending=False)
        res[y]["top_features"] = imp.head(10).round(0).to_dict()
        preds[f"p_{y}_{tag}"] = p
        preds[f"raw_{y}_{tag}"] = raw
    te = te.assign(**preds)
    te["score_" + tag] = te[f"p_y_bldg_{tag}"] + te.bf_share * te[f"p_y_bf_{tag}"]
    REP.mkdir(parents=True, exist_ok=True)
    te.to_parquet(D / f"risk_scores_2018_{tag}.parquet", index=False)
    json.dump(res, open(REP / f"metrics_{tag}.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb")
    a = ap.parse_args()
    r = run(a.emb)
    for y, v in r.items():
        print(y)
        for k, m in v.items():
            if isinstance(m, dict) and "pr_auc" in m:
                print(f"  {k:14s} " + " ".join(f"{kk}={vv:.3f}" if isinstance(vv, float) else f"{kk}={vv}" for kk, vv in m.items()))
        print("  cal", v["calibration"], "\n  top", list(v["top_features"])[:6])
