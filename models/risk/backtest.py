"""Small Sites counterfactual backtest: decide at end of 2018, score on realized 2019-2024 displacement.

Value of buying building i = buyout tenants at i (2019-24) + bf_share_i * Ellis/OMI notices on its block-face (2019-24).
Assumption (same for all strategies): a purchase at t=2018 prevents all of that building's later displacement.
Cost_i = one hedonic price model (fit on 2014-2018 sales) applied to every building, city picks included.
Budget B = predicted cost of the Small Sites buildings the city bought 2019-2023.

City picks' own realized value is censored (their purchase prevented the events), so the city is scored by
k-nearest "lookalike" buildings (same size band, nearest in features, not bought) as well as raw.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pulp
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

INT, D, REP = Path("data/interim"), Path("data/processed"), Path("reports/backtest")
T, OUT_LO, OUT_HI = 2018, 2019, 2024
UNITS = (5, 25)


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
        Xd = X(df).reindex(columns=Xs.columns, fill_value=0)
        return np.exp(m.predict(Xd))
    return predict, {"n_sales": len(s), "r2_in_sample": float(r2)}


def realized_value(u):
    bo = pd.read_parquet(INT / "buyout_events.parquet")
    bo = bo[bo.year.between(OUT_LO, OUT_HI)]
    ev = pd.read_parquet(INT / "eviction_events.parquet")
    ev = ev[ev.bf_ok & ev.ellis_omi & ev.year.between(OUT_LO, OUT_HI)]
    bt = bo.groupby("blklot").tenants_share.sum()
    bfn = ev.groupby("bf").size()
    v_b = u.blklot.map(bt).fillna(0).values
    v_e = (u.bf_share * u.bf.map(bfn).fillna(0)).values
    return v_b, v_e


def knapsack(value, cost, budget):
    prob = pulp.LpProblem("ssp", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(len(value))]
    prob += pulp.lpSum(v * xi for v, xi in zip(value, x))
    prob += pulp.lpSum(c * xi for c, xi in zip(cost, x)) <= budget
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=60))
    return np.array([xi.value() > 0.5 for xi in x])


def fill_by_order(order, cost, budget):
    sel, spent = np.zeros(len(cost), bool), 0.0
    for i in order:
        if spent + cost[i] <= budget:
            sel[i], spent = True, spent + cost[i]
    return sel


def run(tag="gbm", units=UNITS, n_random=1000, k_look=5):
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
    u = u.merge(sp[sp.year == T][["blklot", "property_area"]], on="blklot", how="left")
    u["bf_share"] = u.bf_share.fillna(0)
    u["cost"] = predict(u)
    v_b, v_e = realized_value(u)
    u["value"] = v_b + v_e
    is_city = u.blklot.isin(city_lots).values
    budget = float(u.cost[is_city].sum())

    # expected households: calibrate head probabilities into household counts on t=2015 -> 2016-18
    mult_b, mult_e = _multipliers()
    exp_v = mult_b * u[f"p_y_bldg_{tag}"].values + u.bf_share.values * mult_e * u[f"p_y_bf_{tag}"].values
    res = {"tag": tag, "universe_n": len(u), "units_band": units, "city_n": int(is_city.sum()),
           "city_picks_2019_2023": len(city), "budget_usd": budget, "price_model": pm,
           "universe_total_value": float(u.value.sum())}
    strat = {}
    sel = knapsack(exp_v, u.cost.values, budget)
    strat["ours"] = sel
    rule = ((u.yrs_since_sale <= 2).astype(float) * 10 + u.bf_ellis_omi_3y.clip(0, 5)).values / u.cost.values
    strat["recently_sold_heuristic"] = fill_by_order(np.argsort(-rule), u.cost.values, budget)
    strat["city_raw"] = is_city
    for name, s in strat.items():
        res[name] = {"n": int(s.sum()), "cost": float(u.cost[s].sum()), "households": float(u.value[s].sum()),
                     "units": int(u.units[s].sum())}
    # city lookalikes
    f = ["units", "log_assessed_per_unit", "age", "yrs_since_sale", "bf_rc_units", "cell_sales_3y"]
    Z = StandardScaler().fit_transform(u[f].fillna(u[f].median()))
    cand = np.where(~is_city)[0]
    nn = NearestNeighbors(n_neighbors=k_look).fit(Z[cand])
    _, idx = nn.kneighbors(Z[is_city])
    look_val = u.value.values[cand][idx].mean(1)
    look_cost = u.cost.values[cand][idx].mean(1)
    res["city_lookalike"] = {"n": int(is_city.sum()), "households": float(look_val.sum()),
                             "households_per_$budget": float(look_val.sum() / look_cost.sum() * budget)}
    rng = np.random.default_rng(0)
    rv = [u.value.values[fill_by_order(rng.permutation(len(u)), u.cost.values, budget)].sum() for _ in range(n_random)]
    res["random"] = {"mean": float(np.mean(rv)), "p5": float(np.percentile(rv, 5)), "p95": float(np.percentile(rv, 95))}
    # bootstrap CI on ours vs city_lookalike ratio (resample universe buildings)
    res["ratio_ours_vs_lookalike"] = res["ours"]["households"] / max(res["city_lookalike"]["households_per_$budget"], 1e-9)
    res["ratio_ours_vs_random"] = res["ours"]["households"] / res["random"]["mean"]
    REP.mkdir(parents=True, exist_ok=True)
    u.assign(sel_ours=strat["ours"], is_city=is_city, exp_value=exp_v).to_parquet(D / f"backtest_universe_{tag}.parquet")
    json.dump(res, open(REP / f"backtest_{tag}_{units[0]}-{units[1]}.json", "w"), indent=1)
    return res


def _multipliers():
    """E[buyout tenants | y_bldg=1] and E[block-face Ellis/OMI notices | y_bf=1], from t=2015 (labels 2016-18)."""
    bo = pd.read_parquet(INT / "buyout_events.parquet")
    bo = bo[bo.year.between(2016, 2018)]
    mb = bo.groupby("blklot").tenants_share.sum().mean() * (6 / 3)  # 6-year window vs 3-year label
    ev = pd.read_parquet(INT / "eviction_events.parquet")
    ev = ev[ev.bf_ok & ev.ellis_omi & ev.year.between(2016, 2018)]
    me = ev.groupby("bf").size().mean() * (6 / 3)
    return float(mb), float(me)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gbm")
    ap.add_argument("--units", default="5-25")
    a = ap.parse_args()
    lo, hi = map(int, a.units.split("-"))
    print(json.dumps(run(a.tag, (lo, hi)), indent=1))
