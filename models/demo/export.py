"""Site export: precomputed, hexagon-aggregated JSON for web/ (no building ids, no addresses).

  python -m models.demo.export risk   # test split (T=2018), frozen c_wm_oos_blend: sweep, picks, fairness,
                                      # PR-AUC / precision@k, futures, frozen-2018 risk map, event time-lapse
  python -m models.demo.export wm     # world-model maps (embedding RGB 2016/2018, similarity vectors), probes, λ sweep
  python -m models.demo.export sim    # patient-sim flows, Δ-minutes histograms, breakdowns, persona cards

Run on Savio from a dir whose data/processed has the WM outputs ($S/repo-c layout). Writes web/public/data/*.json.
Privacy: picks are shown only as H3 res-8 counts, and only in hexes with >= MIN_BLDG eligible buildings;
risk / events at res-9 only in hexes with >= MIN_BLDG rent-controlled buildings.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import h3
import numpy as np
import pandas as pd

OUT = Path("web/public/data")
INT, D = Path("data/interim"), Path("data/processed")
CAND = "c_wm_oos_blend"
MIN_BLDG = 5
MULTS = [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
YEARS = list(range(2019, 2025))


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(clean(obj), separators=(",", ":"), allow_nan=False, default=float))
    print(f"wrote {p} ({p.stat().st_size / 1e3:.0f} kB)", flush=True)


def r(x, n=2):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def parent(cells, res):
    return [h3.cell_to_parent(c, res) if isinstance(c, str) else None for c in cells]


# ------------------------------------------------------------------ risk (test split)
def pr_auc(y, s):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s)) if y.sum() else None


def prec_at_k(y, s, k):
    return float(y[np.argsort(-s)[:k]].mean())


def risk():
    from models.risk import backtest as bt
    from models.risk import harness as H
    from models.risk.cands import gbm_tweedie
    from models.risk.cands.c_wm_oos_blend import fit_predict

    ctx = H.make_ctx(2018)
    score = np.asarray(fit_predict(ctx), float)  # households / yr, every rent-controlled building at T
    sc = ctx.score.assign(score=score)
    u, keep = H.universe(ctx, "test")
    u["score"] = np.clip(score[keep], 0, None)
    u["hex8"] = parent(u.cell, 8)
    acs = pd.read_parquet(INT / "acs_cell.parquet")[["cell", "income_tercile", "nonwhite_tercile"]]
    u = u.merge(acs, on="cell", how="left")
    city_score = bt.city_policy_score(u)
    c = u.cost.values
    B1 = float(u.cost[u.is_city].sum())
    rng = np.random.default_rng(0)
    commit = os.environ.get("COMMIT") or subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()

    # --- budget sweep + picks per budget (res-8 counts)
    n8 = u.groupby("hex8").size()
    ok8 = set(n8[n8 >= MIN_BLDG].index)
    sweep, picks = [], {}
    sel_at = {}
    for m in MULTS:
        B = B1 * m
        sels = {"model": bt.knapsack(u.score.values, c, B),
                "city_policy": bt.fill_by_order(np.argsort(-city_score), c, B, allowed=~u.is_city.values),
                "oracle": bt.knapsack(u.value.values, c, B)}
        rv = np.array([u.value.values[bt.fill_by_order(rng.permutation(len(u)), c, B)].sum() for _ in range(300)])
        bs = bt.cluster_boot(u, {k: sels[k] for k in ("model", "city_policy")}, 500, seed=int(m * 10))
        row = {"mult": m, "budget_usd": B, "random": r(rv.mean()), "random_p5": r(np.percentile(rv, 5)),
               "random_p95": r(np.percentile(rv, 95))}
        for k, s in sels.items():
            row[k] = r(u.value[s].sum())
            row[f"{k}_n"] = int(s.sum())
        for k in ("model", "city_policy"):
            row[f"{k}_ci"] = [r(x) for x in bt.ci(bs[k])]
        row["diff_ci"] = [r(x) for x in bt.ci(bs["model"] - bs["city_policy"])]
        sweep.append(row)
        sel_at[m] = sels
        picks[str(m)] = {k: {h: int(v) for h, v in u[sels[k]].groupby("hex8").size().items() if h in ok8}
                         for k in ("model", "city_policy")}
        print(f"{m}x model {row['model']} city {row['city_policy']} random {row['random']}", flush=True)
    uw = u[["blklot", "bf", "bf_share"]].copy()  # picks for etl.displaced -> patient sim (stays on Savio, gitignored)
    for m in (1, 5):
        uw[f"sel_ours_{m}x"] = sel_at[m]["model"]
        uw[f"sel_city_policy_{m}x"] = sel_at[m]["city_policy"]
    uw.to_parquet(D / "backtest_universe_wm.parquet", index=False)
    city_hex = {h: int(v) for h, v in u[u.is_city].groupby("hex8").size().items() if h in ok8}
    hex8 = {h: list(h3.cell_to_latlng(h)) for h in ok8}
    t = pd.DataFrame(sweep)
    main4 = t[t.mult.isin([1, 2, 3, 5])]
    gm = float(np.exp(np.mean(np.log((main4.model + 0.5) / (main4.city_policy + 0.5)))))
    gmr = float(np.exp(np.mean(np.log((main4.model + 0.5) / (main4.random + 0.5)))))
    headline = {"vs_city": r(gm), "vs_random": r(gmr), "beats_city_all_budgets": bool((main4.model > main4.city_policy).all()),
                "budget_1x_usd": B1, "city_picks": int(u.is_city.sum()), "universe_n": len(u),
                "total_households": r(u.value.sum(), 1), "cand": CAND, "commit": commit}

    # --- fairness (model vs city policy at 1x and 5x)
    u["size_band"] = u.size_band.astype(str)
    fair = {f"{k}_{m}x": bt.fairness(u, sel_at[m][k]) for m in (1, 5) for k in ("model", "city_policy")}

    # --- risk metrics on test (frozen candidate; baselines: city-policy score, LightGBM w/o world model, random)
    tw = np.clip(np.asarray(gbm_tweedie.fit_predict(ctx), float)[keep], 0, None)
    ys = {"any_displacement": (u.value > 0).astype(int).values, "building_buyout": (u.value_buyout > 0).astype(int).values}
    scores = {"model": u.score.values, "lightgbm_no_wm": tw, "city_policy": city_score}
    metrics = {}
    for yn, y in ys.items():
        mrow = {"prevalence": r(y.mean(), 4), "n_pos": int(y.sum())}
        for sn, s in scores.items():
            mrow[sn] = {"pr_auc": r(pr_auc(y, s), 4), **{f"p@{k}": r(prec_at_k(y, s, k), 3) for k in (25, 100, 250)}}
        metrics[yn] = mrow
    top = np.argsort(-u.score.values)[: len(u) // 10]
    metrics["capture_top10pct"] = {"model": r(u.value.values[top].sum() / u.value.sum(), 3),
                                   "lightgbm_no_wm": r(u.value.values[np.argsort(-tw)[: len(u) // 10]].sum() / u.value.sum(), 3),
                                   "city_policy": r(u.value.values[np.argsort(-city_score)[: len(u) // 10]].sum() / u.value.sum(), 3)}
    pd.DataFrame([{"cand": CAND, "split": "test", "T": 2018, "what": "demo export: sweep, PR-AUC, p@k, futures",
                   "commit": commit, **{f"pr_auc_{k}": v["model"]["pr_auc"] for k, v in metrics.items() if "model" in v and isinstance(v["model"], dict)}}]) \
        .to_csv(Path("reports/risk/test_log.csv"), mode="a", header=not Path("reports/risk/test_log.csv").exists(), index=False)

    # --- futures: hex-level gamma-Poisson, level + dispersion calibrated on dev (T=2016, outcomes 2017-19)
    cal = calibrate_futures()
    futures = make_futures(ctx, u, ok8, cal)

    # --- frozen-2018 risk map (res-9, all rent-controlled buildings) + 2019-24 event time-lapse
    sc["hex9"] = sc.cell
    g9 = sc.groupby("hex9").agg(n=("blklot", "size"), risk=("score", "sum"), units=("units", "sum"))
    g9 = g9[g9.n >= MIN_BLDG]
    ev = pd.read_parquet(INT / "eviction_events.parquet")
    ev = ev[ev.year.between(2019, 2024)]
    bfp = pd.read_parquet(INT / "blockfaces.parquet")[["bf", "lat", "lon"]].rename(columns={"lat": "blat", "lon": "blon"})
    ev = ev.merge(bfp, on="bf", how="left")
    ev["lat"], ev["lon"] = ev.lat.fillna(ev.blat), ev.lon.fillna(ev.blon)
    ev = ev.dropna(subset=["lat", "lon"])
    ev["hex9"] = [h3.latlng_to_cell(a, b, 9) for a, b in zip(ev.lat, ev.lon)]
    ev["kind"] = np.where(ev.ellis_act_withdrawal.fillna(False), "ellis",
                          np.where(ev.owner_move_in.fillna(False), "omi", "other"))
    bo = pd.read_parquet(INT / "buyout_events.parquet")
    bo = bo[bo.year.between(2019, 2024)]
    cell_of = sc.set_index("blklot").cell
    bo["hex9"] = bo.blklot.map(cell_of)
    bo["kind"] = "buyout"
    e = pd.concat([ev[["hex9", "year", "kind"]], bo.dropna(subset=["hex9"])[["hex9", "year", "kind"]]])
    e = e[e.hex9.isin(g9.index)]
    events = e.groupby(["hex9", "year", "kind"]).size().reset_index(name="n")
    top_hex = set(g9.risk.sort_values(ascending=False).index[: max(1, len(g9) // 10)])
    disp = e[e.kind.isin(["ellis", "omi", "buyout"])]
    risk_map = {"hexes": [{"h": h, "risk": r(v.risk, 3), "n": int(v.n), "units": int(v.units)} for h, v in g9.iterrows()],
                "events": events.astype({"year": int, "n": int}).values.tolist(),
                "top_decile_share": r(disp.hex9.isin(top_hex).mean(), 3), "top_decile_area_share": 0.1}

    dump("headline", {**headline, "metrics": metrics, "coverage_80": futures["coverage_80"]})
    dump("sweep", {"rows": sweep, "hex8": hex8, "picks": picks, "city_actual": city_hex})
    dump("fairness", fair)
    dump("futures", futures)
    dump("risk_map", risk_map)
    print(json.dumps({**headline, "metrics": metrics, "coverage_80": futures["coverage_80"],
                      "top_decile_share": risk_map["top_decile_share"]}, indent=1, default=float))


def event_size(ctx):
    lab = ctx.labels(1)
    lab = lab[lab.t >= 2014]
    return float(lab.y_val[lab.y_val > 0].mean())


def draw(rate, k, years, rng, nd=1000):
    """rate: (n_hex,) events/yr. Each hex gets one persistent Gamma(k, 1/k) multiplier per future (mean 1),
    then independent Poisson counts per year. Returns (nd, n_hex, years) event counts."""
    g = rng.gamma(k, 1 / k, size=(nd, len(rate))) if np.isfinite(k) else np.ones((nd, len(rate)))
    return rng.poisson((g * rate[None, :])[:, :, None], size=(nd, len(rate), years))


def calibrate_futures(grid=(0.25, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0, np.inf)):
    """Leak-free w.r.t. test: run the frozen candidate on dev (decide 2016), compare to 2017-19 outcomes.
    level = realized / predicted households; k = gamma shape whose hex-level 80% band covers ~80% of dev hexes."""
    from models.risk import harness as H
    from models.risk.cands.c_wm_oos_blend import fit_predict
    ctx = H.make_ctx(2016)
    s = np.asarray(fit_predict(ctx), float)
    u, keep = H.universe(ctx, "dev")
    u["score"] = np.clip(s[keep], 0, None)
    u["hex8"] = parent(u.cell, 8)
    size = event_size(ctx)
    ny = 3
    level = float(u.value.sum() / (ny * u.score.sum()))
    n8 = u.groupby("hex8").size()
    hx = n8[n8 >= MIN_BLDG].index
    lam = (u.groupby("hex8").score.sum().reindex(hx) * level / size).to_numpy()
    real = u.groupby("hex8").value.sum().reindex(hx).to_numpy()
    rng = np.random.default_rng(7)
    cov = {}
    for k in grid:
        tot = draw(lam, k, ny, rng).sum(2) * size
        lo, hi = np.percentile(tot, 10, 0), np.percentile(tot, 90, 0)
        cov[k] = float(((real >= lo) & (real <= hi)).mean())
    kbest = min(cov, key=lambda k: abs(cov[k] - 0.8))
    print("futures calibration (dev): level", round(level, 3), "coverage by k", {str(k): round(v, 3) for k, v in cov.items()},
          "-> k", kbest, flush=True)
    return {"level": level, "k": kbest, "dev_coverage": cov[kbest], "dev_size": size}


def make_futures(ctx, u, ok8, cal, nd=1000):
    from models.risk import backtest as bt
    size = event_size(ctx)
    hall = sorted(set(u.hex8.dropna()))
    lam = (u.groupby("hex8").score.sum().reindex(hall) * cal["level"] / size).to_numpy()
    rng = np.random.default_rng(1)
    cum = draw(lam, cal["k"], len(YEARS), rng, nd).astype(np.float32).cumsum(2) * size  # (nd, hex, yr)
    real_y = pd.DataFrame({y: sum(bt.realized_value(u, y, y)) for y in YEARS}).groupby(u.hex8.values).sum()
    real = real_y.reindex(hall).fillna(0).to_numpy().cumsum(1)
    Q = [10, 25, 50, 75, 90]

    def fan(c, rl):
        return {"q": {str(q): [r(x, 2) for x in np.percentile(c, q, axis=0)] for q in Q},
                "paths": [[r(x, 2) for x in c[i]] for i in range(0, nd, nd // 40)], "realized": [r(x, 2) for x in rl]}
    ok = [i for i, h in enumerate(hall) if h in ok8]
    p10, p90 = np.percentile(cum[:, ok, -1], 10, 0), np.percentile(cum[:, ok, -1], 90, 0)
    inside = (real[ok, -1] >= p10) & (real[ok, -1] <= p90)
    city = cum.sum(1)
    city_real = real.sum(0)
    cl, ch = np.percentile(city[:, -1], [10, 90])
    out = {"years": YEARS, "event_size": r(size, 3), "calibration": {k: r(v, 3) for k, v in cal.items()},
           "city": fan(city, city_real), "hex": {hall[i]: fan(cum[:, i, :], real[i]) for i in ok},
           "coverage_80": r(inside.mean(), 3), "n_hex": len(ok), "city_inside_80": bool(cl <= city_real[-1] <= ch)}
    print("futures test: coverage_80", out["coverage_80"], "city", r(np.median(city[:, -1]), 1), [r(cl, 1), r(ch, 1)],
          "real", r(city_real[-1], 1), flush=True)
    return out


def futures_only():
    """Rebuild futures.json without re-running (or re-logging) the sweep / metrics."""
    from models.risk import harness as H
    from models.risk.cands.c_wm_oos_blend import fit_predict
    ctx = H.make_ctx(2018)
    s = np.asarray(fit_predict(ctx), float)
    u, keep = H.universe(ctx, "test")
    u["score"] = np.clip(s[keep], 0, None)
    u["hex8"] = parent(u.cell, 8)
    n8 = u.groupby("hex8").size()
    fut = make_futures(ctx, u, set(n8[n8 >= MIN_BLDG].index), calibrate_futures())
    dump("futures", fut)
    hp = OUT / "headline.json"
    if hp.exists():
        h = json.loads(hp.read_text())
        h["coverage_80"] = fut["coverage_80"]
        dump("headline", h)


# ------------------------------------------------------------------ world model
def wm():
    from sklearn.decomposition import PCA

    e = pd.read_parquet(D / "cell_embeddings.parquet")
    sel = json.loads(Path("reports/wm/selected.json").read_text())
    lam = float(sel.get("lam", sel.get("lambda", 0.1)))
    if "lam" in e:
        e = e[(e.lam - lam).abs() < 1e-6]
    ecols = [c for c in e.columns if c.startswith("e") and c[1:].isdigit()]

    def frame(cut):
        f = e[(e.cutoff == cut) & (e.year == cut)].drop_duplicates("cell").set_index("cell")
        return f[ecols].to_numpy(np.float32), f.index.to_numpy()
    X18, c18 = frame(2018)
    X16, c16 = frame(2016)
    p18 = PCA(16, random_state=0).fit(X18)
    Z18 = p18.transform(X18)
    Z16 = PCA(16, random_state=0).fit_transform(X16)
    common = np.intersect1d(c18, c16)
    i18 = pd.Index(c18).get_indexer(common)
    i16 = pd.Index(c16).get_indexer(common)
    U, _, Vt = np.linalg.svd(Z16[i16].T @ Z18[i18])  # orthogonal Procrustes: 2016 PCA space -> 2018's
    Z16a = Z16 @ (U @ Vt)

    def rgb(Z, ref):
        lo, hi = np.percentile(ref[:, :3], 2, 0), np.percentile(ref[:, :3], 98, 0)
        return (np.clip((Z[:, :3] - lo) / (hi - lo), 0, 1) * 255).round().astype(int)
    rgb18, rgb16 = rgb(Z18, Z18), rgb(Z16a, Z18)
    zn = Z18 / np.linalg.norm(Z18, axis=1, keepdims=True).clip(1e-9)
    q = np.clip(np.round(zn * 127), -127, 127).astype(int)
    m16 = dict(zip(c16, rgb16.tolist()))
    cells = [{"h": c, "rgb": rgb18[i].tolist(), "rgb16": m16.get(c), "v": q[i].tolist()} for i, c in enumerate(c18)]
    probes = pd.read_csv("reports/wm/probes.csv").to_dict(orient="records")
    pt = Path("reports/wm/probes_test.csv")
    final = None
    if pt.exists():  # locked test (models.world_model.eval_final --final)
        t = pd.read_csv(pt)
        t = t[(t.features == "emb") & t.lift.notna()]
        final = {"mean_lift": float(t.lift.mean()), "targets": t[["target", "lift", "lift_lo", "lift_hi"]].to_dict(orient="records")}
        sj = Path("reports/wm/probes_test_summary.json")
        if sj.exists():
            final["summary"] = json.loads(sj.read_text())
    sweep = pd.read_csv("reports/wm/lambda_sweep.csv").to_dict(orient="records")
    dump("wm", {"cells": cells, "lam": lam, "probes": probes, "probes_final": final,
                "lambda_sweep": sweep, "explained_var_3": r(p18.explained_variance_ratio_[:3].sum(), 3)})


# ------------------------------------------------------------------ patient sim
def sim():
    s = pd.read_parquet(D / "sim_results.parquet")
    summ = json.loads((D / "sim_summary.json").read_text())
    cls = ["clinic", "pharmacy", "grocery"]
    out = {"summary": {k: summ[k] for k in ("headline", "migration", "migration_validation") if k in summ}}
    t2 = pd.read_parquet(INT / "tracts2020.parquet") if (INT / "tracts2020.parquet").exists() else None
    # origins: tract-ish (res-7 parent of the origin cell), destinations: PUMA centroid of the sampled destination
    s["o7"] = parent([h3.latlng_to_cell(a, b, 9) for a, b in zip(s.origin_lat, s.origin_lon)], 7)
    s["d7"] = parent([h3.latlng_to_cell(a, b, 9) for a, b in zip(s.dest_lat, s.dest_lon)], 7)
    scen = {}
    for sc_name, g in s.groupby("scenario"):
        cap = 120.0
        dd = {}
        for c in cls:
            b = g[f"t_{c}_before"].clip(upper=cap)
            a = g[f"t_{c}_after"].clip(upper=cap)
            d = (a - b).to_numpy()
            hist, edges = np.histogram(np.clip(d, -30, 90), bins=np.arange(-30, 92, 4))
            lose = (g[f"acc30_{c}_before"] & ~g[f"acc30_{c}_after"])
            dd[c] = {"median_delta": r(np.median(d), 1), "mean_delta": r(d.mean(), 1),
                     "pct_lose_30": r(lose.mean() * 100, 1), "hist": hist.tolist(), "edges": edges.tolist()}
            for grp in ("has_car", "lang", "age_band", "income_band", "disability"):
                dd[c].setdefault("by", {})[grp] = {str(k): {"median_delta": r(np.median((v[f"t_{c}_after"].clip(upper=cap) - v[f"t_{c}_before"].clip(upper=cap))), 1),
                                                         "pct_lose_30": r((v[f"acc30_{c}_before"] & ~v[f"acc30_{c}_after"]).mean() * 100, 1),
                                                         "n": int(len(v))} for k, v in g.groupby(grp)}
        fl = g.groupby(["o7", "d7"]).size().reset_index(name="n")
        fl = fl[fl.n >= 3]
        scen[sc_name] = {"n": int(len(g)), "delta": dd,
                         "flows": [[o, list(h3.cell_to_latlng(o)), list(h3.cell_to_latlng(d_)), int(n)] for o, d_, n in fl.values],
                         "dest_share": g.dest_puma.value_counts(normalize=True).head(15).round(4).to_dict()}
    out["scenarios"] = scen
    # persona cards: synthetic households (PUMS-derived), realized scenario, spread across groups
    g = s[s.scenario == "realized_2019_2024"].copy()
    g["d_clinic"] = g.t_clinic_after.clip(upper=120) - g.t_clinic_before.clip(upper=120)
    rng = np.random.default_rng(3)
    cards = []
    for (car, age), v in g.groupby(["has_car", "age_band"]):
        v = v[np.isfinite(v.d_clinic)]
        if len(v) < 20:
            continue
        x = v.iloc[(v.d_clinic - v.d_clinic.quantile(0.75)).abs().argsort()[:30]].sample(1, random_state=int(rng.integers(1e6)))
        x = x.iloc[0]
        cards.append({"age_band": x.age_band, "income_band": x.income_band, "lang": x.lang, "has_car": bool(x.has_car),
                      "disability": bool(x.disability), "dest_puma": x.dest_puma,
                      **{f"{c}_{k}": r(min(x[f't_{c}_{k}'], 120), 0) for c in cls for k in ("before", "after")}})
    out["personas"] = cards
    dump("sim", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["risk", "wm", "sim", "futures", "all"])
    a = ap.parse_args()
    for w in (["risk", "wm", "sim"] if a.what == "all" else [a.what]):
        (futures_only if w == "futures" else globals()[w])()
