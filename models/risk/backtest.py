"""Small Sites counterfactual backtest: decide at end of 2018, score on realized 2019-2024 displacement.

Pre-registered (fixed before the first run of this version; see reports/backtest/PREREG.md):
  value_i  = buyout tenants at i (2019-24) + bf_share_i * Ellis/OMI notices on its block-face (2019-24)
  cost_i   = one hedonic price model (2014-18 sales) applied to every building, city picks included
  budget B = predicted cost of the city's 2019-23 Small Sites picks inside the universe
  PRIMARY  = households protected at 1x B: ours vs "city policy" (a model of the city's own selection rule,
             applied to buildings the city did not buy; the city's own picks are censored by their purchase).
             95% CI from a block-face cluster bootstrap of realized outcomes, selections held fixed.
  SECONDARY= the same comparison across budgets 1x..20x B, vs random fills, recently-sold heuristic, and an
             oracle upper bound; plus raw city picks and k-NN lookalikes at 1x.
  Fairness = share of protected households by supervisor district, size band, ACS income / non-white tercile.

python -m models.risk.backtest --tag gbm
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pulp
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

INT, D, REP = Path("data/interim"), Path("data/processed"), Path("reports/backtest")
T, OUT_LO, OUT_HI = 2018, 2019, 2024
UNITS = (5, 25)
SWEEP = (1, 2, 3, 5, 10, 20)
CITY_F = ["units", "log_assessed_per_unit", "age", "yrs_since_sale", "bf_rc_units", "cell_sales_3y", "owner_occ",
          "bf_ellis_omi_3y", "permits_3y"]


def price_model(sp, sales):
    s = sales[(sales.sale_year.between(2014, T)) & (sales.assessed_after > 0) & (sales.jump > 1.2)]
    x = sp[(sp.rent_ctrl == 1)][["blklot", "year", "units", "age", "property_area", "analysis_neighborhood"]]
    s = s.merge(x, left_on=["blklot", "sale_year"], right_on=["blklot", "year"])

    def X(df):
        z = pd.DataFrame({"lu": np.log(df.units.clip(1)), "age": df.age.fillna(100),
                          "la": np.log1p(df.property_area.fillna(0))})
        nb = pd.get_dummies(df.analysis_neighborhood.fillna("NA"), prefix="nb", dtype=float)
        return pd.concat([z.reset_index(drop=True), nb.reset_index(drop=True)], axis=1)
    Xs = X(s)
    m = RidgeCV(alphas=[0.1, 1, 10]).fit(Xs, np.log(s.assessed_after))
    r2 = m.score(Xs, np.log(s.assessed_after))

    def predict(df):
        return np.exp(m.predict(X(df).reindex(columns=Xs.columns, fill_value=0)))
    return predict, {"n_sales": len(s), "r2_in_sample": float(r2)}


def realized_value(u):
    bo = pd.read_parquet(INT / "buyout_events.parquet")
    bo = bo[bo.year.between(OUT_LO, OUT_HI)]
    ev = pd.read_parquet(INT / "eviction_events.parquet")
    ev = ev[ev.bf_ok & ev.ellis_omi & ev.year.between(OUT_LO, OUT_HI)]
    v_b = u.blklot.map(bo.groupby("blklot").tenants_share.sum()).fillna(0).values
    v_e = (u.bf_share * u.bf.map(ev.groupby("bf").size()).fillna(0)).values
    return v_b, v_e


def knapsack(value, cost, budget, time_limit=120):
    prob = pulp.LpProblem("ssp", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(len(value))]
    prob += pulp.lpSum(float(v) * xi for v, xi in zip(value, x))
    prob += pulp.lpSum(float(c) * xi for c, xi in zip(cost, x)) <= budget
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
    return np.array([(xi.value() or 0) > 0.5 for xi in x])


def fill_by_order(order, cost, budget, allowed=None):
    sel, spent = np.zeros(len(cost), bool), 0.0
    for i in order:
        if allowed is not None and not allowed[i]:
            continue
        if spent + cost[i] <= budget:
            sel[i], spent = True, spent + cost[i]
    return sel


def _multipliers():
    """Fallback when no value head: E[buyout tenants | y_bldg] and E[bf notices | y_bf], t=2015 (2016-18)."""
    bo = pd.read_parquet(INT / "buyout_events.parquet")
    mb = bo[bo.year.between(2016, 2018)].groupby("blklot").tenants_share.sum().mean() * 2
    ev = pd.read_parquet(INT / "eviction_events.parquet")
    me = ev[ev.bf_ok & ev.ellis_omi & ev.year.between(2016, 2018)].groupby("bf").size().mean() * 2
    return float(mb), float(me)


def load_universe(tag, units):
    sc = pd.read_parquet(D / f"risk_scores_2018_{tag}.parquet")
    sp = pd.read_parquet(INT / "spine.parquet", columns=["blklot", "year", "units", "age", "property_area",
                                                          "analysis_neighborhood", "rent_ctrl"])
    sales = pd.read_parquet(INT / "sales.parquet")
    predict, pm = price_model(sp, sales)
    picks = pd.read_parquet(INT / "ssp_picks.parquet")
    picks = picks[picks.blklots.map(len) > 0]
    bought_before = {b for ls, y in zip(picks.blklots, picks.acq_year) if y == y and y <= T for b in ls}
    city = picks[picks.acq_year.between(OUT_LO, 2023)]
    city_lots = {b for ls in city.blklots for b in ls}
    u = sc[sc.units.between(*units) & ~sc.blklot.isin(bought_before)].reset_index(drop=True)
    u = u.merge(sp[sp.year == T][["blklot", "property_area"]].drop_duplicates("blklot"), on="blklot", how="left")
    u["bf_share"] = u.bf_share.fillna(0)
    u["cost"] = predict(u)
    v_b, v_e = realized_value(u)
    u["value_buyout"], u["value_bf"] = v_b, v_e
    u["value"] = v_b + v_e
    if f"ev_{tag}" in u:
        u["exp_value"] = u[f"ev_{tag}"].clip(lower=0)
    else:
        mb, me = _multipliers()
        u["exp_value"] = mb * u[f"p_y_bldg_{tag}"] + u.bf_share * me * u[f"p_y_bf_{tag}"]
    acs = INT / "acs_cell.parquet"
    if acs.exists():
        u = u.merge(pd.read_parquet(acs)[["cell", "income_tercile", "nonwhite_tercile"]], on="cell", how="left")
    u["is_city"] = u.blklot.isin(city_lots)
    return u, {"price_model": pm, "city_picks_2019_2023": len(city)}


def city_policy_score(u):
    """P(city picks building | features): the city's revealed selection rule."""
    X = u[CITY_F].fillna(u[CITY_F].median())
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000))
    return m.fit(X, u.is_city).predict_proba(X)[:, 1]


def strategies(u, budget, city_score, n_random, rng):
    c, ev = u.cost.values, u.exp_value.values
    not_city = ~u.is_city.values
    s = {"ours": knapsack(ev, c, budget),
         "ours_greedy": fill_by_order(np.argsort(-ev / c), c, budget),
         "recently_sold_heuristic": fill_by_order(
             np.argsort(-(((u.yrs_since_sale <= 2) * 10 + u.bf_ellis_omi_3y.clip(0, 5)).values / c)), c, budget),
         "city_policy": fill_by_order(np.argsort(-city_score), c, budget, allowed=not_city),
         "oracle": knapsack(u.value.values, c, budget, time_limit=60)}
    rv = np.array([u.value.values[fill_by_order(rng.permutation(len(u)), c, budget)].sum() for _ in range(n_random)])
    return s, rv


def cluster_boot(u, sels, n_boot, seed=0):
    """Resample block-faces; selections fixed. Returns per-strategy household draws."""
    g = u.bf.fillna(u.blklot).astype("category").cat.codes.values
    G = g.max() + 1
    rng = np.random.default_rng(seed)
    per = {k: np.bincount(g, weights=u.value.values * v, minlength=G) for k, v in sels.items()}
    per["_all"] = np.bincount(g, weights=u.value.values, minlength=G)
    per["_cost_all"] = np.bincount(g, weights=u.cost.values, minlength=G)
    draws = {k: [] for k in per}
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, G, G), minlength=G)
        for k, a in per.items():
            draws[k].append(float(w @ a))
    return {k: np.array(v) for k, v in draws.items()}


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def fairness(u, sel, cols=("supervisor_district", "size_band", "income_tercile", "nonwhite_tercile")):
    out = {}
    for c in cols:
        if c not in u:
            continue
        g = u.assign(prot=u.value * sel, sel=sel).groupby(u[c].astype(str))
        t = pd.DataFrame({"share_protected": g.prot.sum() / max((u.value * sel).sum(), 1e-9),
                          "share_displacement": g.value.sum() / u.value.sum(),
                          "share_rc_units": g.units.sum() / u.units.sum(),
                          "n_selected": g.sel.sum()})
        out[c] = t.round(3).to_dict(orient="index")
    return out


def run(tag="gbm", units=UNITS, sweep=SWEEP, n_random=300, n_boot=1000, k_look=5):
    u, meta = load_universe(tag, units)
    B = float(u.cost[u.is_city].sum())
    city_score = city_policy_score(u)
    rng = np.random.default_rng(0)
    res = {"tag": tag, "units_band": units, "universe_n": len(u), "city_n_in_universe": int(u.is_city.sum()),
           "budget_1x_usd": B, "universe_total_households": float(u.value.sum()),
           "universe_total_cost": float(u.cost.sum()), **meta, "sweep": {}}
    rows, sels_1x = [], None
    for m in sweep:
        b = B * m
        sels, rv = strategies(u, b, city_score, n_random, rng)
        bs = cluster_boot(u, sels, n_boot, seed=m)
        r = {"budget_usd": b, "random": {"mean": float(rv.mean()), "p5": float(np.percentile(rv, 5)),
                                         "p95": float(np.percentile(rv, 95))}}
        # random expected value under bootstrap ~ budget share of total households per $ (same draws)
        rand_bs = bs["_all"] * (rv.mean() / max(u.value.sum(), 1e-9))
        for k, s in sels.items():
            r[k] = {"n": int(s.sum()), "cost": float(u.cost[s].sum()), "households": float(u.value[s].sum()),
                    "households_ci": ci(bs[k]), "units": int(u.units[s].sum()),
                    "households_buyout": float(u.value_buyout[s].sum()), "households_bf": float(u.value_bf[s].sum())}
        for k in ["ours", "ours_greedy", "recently_sold_heuristic"]:
            r[f"{k}_vs_city_policy"] = {"diff": r[k]["households"] - r["city_policy"]["households"],
                                        "diff_ci": ci(bs[k] - bs["city_policy"]),
                                        "ratio": r[k]["households"] / max(r["city_policy"]["households"], 1e-9),
                                        "ratio_ci": ci(bs[k] / np.maximum(bs["city_policy"], 1e-9))}
            r[f"{k}_vs_random"] = {"ratio": r[k]["households"] / max(rv.mean(), 1e-9),
                                   "ratio_ci": ci(bs[k] / np.maximum(rand_bs, 1e-9))}
        r["pct_of_oracle_ours"] = r["ours"]["households"] / max(r["oracle"]["households"], 1e-9)
        res["sweep"][f"{m}x"] = r
        for k in sels:
            rows.append({"mult": m, "strategy": k, "households": r[k]["households"], "lo": r[k]["households_ci"][0],
                         "hi": r[k]["households_ci"][1], "n": r[k]["n"]})
        rows.append({"mult": m, "strategy": "random", "households": rv.mean(), "lo": r["random"]["p5"],
                     "hi": r["random"]["p95"], "n": np.nan})
        if m == 1:
            sels_1x = sels
        if m == 5:
            res["fairness_5x"] = {k: fairness(u, sels[k]) for k in ["ours", "city_policy"]}
    # 1x extras: raw city picks (censored) and k-NN lookalikes
    is_city = u.is_city.values
    res["city_raw_1x"] = {"n": int(is_city.sum()), "households": float(u.value[is_city].sum()),
                          "note": "censored: the purchase itself prevented later events"}
    Z = StandardScaler().fit_transform(u[CITY_F].fillna(u[CITY_F].median()))
    cand = np.where(~is_city)[0]
    _, idx = NearestNeighbors(n_neighbors=k_look).fit(Z[cand]).kneighbors(Z[is_city])
    lv, lc = u.value.values[cand][idx].mean(1), u.cost.values[cand][idx].mean(1)
    res["city_lookalike_knn_1x"] = {"households": float(lv.sum()), "households_at_budget": float(lv.sum() / lc.sum() * B)}
    res["fairness_1x"] = {k: fairness(u, sels_1x[k]) for k in ["ours", "city_policy"]}
    res["fairness_1x"]["city_raw"] = fairness(u, is_city)
    REP.mkdir(parents=True, exist_ok=True)
    tagu = f"{tag}_{units[0]}-{units[1]}"
    pd.DataFrame(rows).to_csv(REP / f"sweep_{tagu}.csv", index=False)
    u.assign(city_score=city_score, **{f"sel_{k}_1x": v for k, v in sels_1x.items()}).to_parquet(
        D / f"backtest_universe_{tag}.parquet", index=False)
    json.dump(res, open(REP / f"backtest_{tagu}.json", "w"), indent=1, default=float)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gbm")
    ap.add_argument("--units", default="5-25")
    ap.add_argument("--n_random", type=int, default=300)
    ap.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args()
    lo, hi = map(int, a.units.split("-"))
    r = run(a.tag, (lo, hi), n_random=a.n_random, n_boot=a.n_boot)
    for m, v in r["sweep"].items():
        print(m, {k: round(v[k]["households"], 2) for k in ["ours", "ours_greedy", "recently_sold_heuristic",
                                                             "city_policy", "oracle"]},
              "random", round(v["random"]["mean"], 2), "ours/city", round(v["ours_vs_city_policy"]["ratio"], 2),
              v["ours_vs_city_policy"]["ratio_ci"])
