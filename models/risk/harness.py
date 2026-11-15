"""Leak-proof eval harness for Small Sites candidate scorers (LightGBM, world-model embeddings, anything).

Splits (fixed; same value/cost/budget/city-policy rules as reports/backtest/PREREG.md):
  dev   decide end of 2016, outcomes 2017-2019, city picks 2017-2021 (31 picks), embeddings cutoff 2016
  test  decide end of 2018, outcomes 2019-2024, city picks 2019-2023 (prereg), embeddings cutoff 2018
Iterate on dev as much as you like. `--split test` needs `--final`, and every call is logged to
reports/risk/test_log.csv (commit hash + result). Keep test calls rare (~1 per frozen candidate).

Candidate = module models/risk/cands/<name>.py with
    def fit_predict(ctx) -> np.ndarray   # expected households protected per building in ctx.score (any scale)
ctx gives only data dated <= ctx.T:
    ctx.T                decision year
    ctx.rows             risk_dataset features for every snapshot t <= T (no labels)
    ctx.score            ctx.rows[t == T] (all rent-controlled buildings; harness evaluates the 5-25 unit band)
    ctx.labels(h)        labels for rows with t + h <= T (horizon h years), built from events <= T:
                         y_bldg (any buyout), y_bf (any Ellis/OMI on block-face), y_val (households, the objective),
                         y_buyout_tenants, y_bf_notices
    ctx.events           {"buyouts", "evictions", "sales"} dated <= T
    ctx.emb(kind)        "parcel"/"cell" world-model embeddings from the cutoff-T model (raises if absent)
    ctx.cost(df)         the harness's hedonic cost model (fit on sales T-4..T)

python -m models.risk.harness --cand <name> [--split dev|test --final]
python -m models.risk.harness --cand heuristic,city_policy_score  # baselines
"""
import argparse
import importlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from models.risk import backtest as bt

INT, D, REP = Path("data/interim"), Path("data/processed"), Path("reports/risk")
SPLITS = {"dev": dict(T=2016, out=(2017, 2019), city=(2017, 2021)),
          "test": dict(T=2018, out=(2019, 2024), city=(2019, 2023))}
MULTS = (1, 2, 3, 5)
LABELS = ["y_bldg", "y_bf", "y_val", "label_end"]


@dataclass
class Ctx:
    T: int
    rows: pd.DataFrame
    events: dict
    _cost: object

    @property
    def score(self):
        return self.rows[self.rows.t == self.T].reset_index(drop=True)

    def cost(self, df):
        return self._cost(df)

    def labels(self, h=3):
        r = self.rows[self.rows.t + h <= self.T][["blklot", "t", "bf", "bf_share"]].copy()
        bo, ev = self.events["buyouts"], self.events["evictions"]
        ev = ev[ev.bf_ok & ev.ellis_omi]
        out = []
        for t, g in r.groupby("t"):
            fb = bo[(bo.year > t) & (bo.year <= t + h)]
            fe = ev[(ev.year > t) & (ev.year <= t + h)]
            g = g.copy()
            g["y_buyout_tenants"] = g.blklot.map(fb.groupby("blklot").tenants_share.sum()).fillna(0)
            g["y_bf_notices"] = g.bf.map(fe.groupby("bf").size()).fillna(0)
            g["y_bldg"] = (g.y_buyout_tenants > 0).astype("int8")
            g["y_bf"] = (g.y_bf_notices > 0).astype("int8")
            g["y_val"] = (g.y_buyout_tenants + g.bf_share.fillna(0) * g.y_bf_notices).astype("float32")
            out.append(g)
        lab = pd.concat(out)
        return self.rows.merge(lab.drop(columns=["bf", "bf_share"]), on=["blklot", "t"])

    def emb(self, kind="parcel"):
        p = D / f"{kind}_embeddings.parquet"
        e = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["cutoff"])
        e = e[(e.cutoff == self.T) & (e.year <= self.T)]
        if e.empty:
            raise FileNotFoundError(f"no {kind} embeddings with cutoff={self.T} in {p}")
        return e


def make_ctx(T):
    d = pd.read_parquet(D / "risk_dataset.parquet")
    d = d[d.t <= T].drop(columns=LABELS).reset_index(drop=True)
    d["supervisor_district"] = pd.to_numeric(d.supervisor_district, errors="coerce").fillna(-1).astype(int)
    bo = pd.read_parquet(INT / "buyout_events.parquet").dropna(subset=["year"])
    ev = pd.read_parquet(INT / "eviction_events.parquet").dropna(subset=["year"])
    sa = pd.read_parquet(INT / "sales.parquet")
    sp = pd.read_parquet(INT / "spine.parquet", columns=["blklot", "year", "units", "age", "property_area",
                                                          "analysis_neighborhood", "rent_ctrl"])
    sp = sp[sp.year <= T]
    predict, _ = bt.price_model(sp, sa[sa.sale_year <= T], T=T)
    d = d.merge(sp[sp.year == T][["blklot", "property_area"]].drop_duplicates("blklot"), on="blklot", how="left")
    return Ctx(T, d, {"buyouts": bo[bo.year <= T], "evictions": ev[ev.year <= T], "sales": sa[sa.sale_year <= T]},
               predict)


def universe(ctx, split, units=bt.UNITS):
    s = SPLITS[split]
    lo, hi = s["out"]
    picks = pd.read_parquet(INT / "ssp_picks.parquet")
    picks = picks[picks.blklots.map(len) > 0]
    before = {b for ls, y in zip(picks.blklots, picks.acq_year) if y == y and y <= ctx.T for b in ls}
    city = {b for ls, y in zip(picks.blklots, picks.acq_year) if s["city"][0] <= y <= s["city"][1] for b in ls}
    sc = ctx.score
    keep = sc.units.between(*units) & ~sc.blklot.isin(before)
    u = sc[keep].reset_index(drop=True)
    u["bf_share"] = u.bf_share.fillna(0)
    u["cost"] = ctx.cost(u)
    u["value_buyout"], u["value_bf"] = bt.realized_value(u, lo, hi)
    u["value"] = u.value_buyout + u.value_bf
    u["is_city"] = u.blklot.isin(city)
    return u, keep.values


def heuristic(ctx):
    s = ctx.score
    return ((s.yrs_since_sale <= 2) * 10 + s.bf_ellis_omi_3y.clip(0, 5)).values.astype(float)


def evaluate(u, ev_score, n_random=200, n_boot=500, seed=0):
    u = u.assign(exp_value=np.clip(ev_score, 0, None))
    city_score = bt.city_policy_score(u)
    rng = np.random.default_rng(seed)
    out = []
    for m in MULTS:
        B = float(u.cost[u.is_city].sum()) * m
        c = u.cost.values
        sels = {"cand": bt.knapsack(u.exp_value.values, c, B),
                "cand_greedy": bt.fill_by_order(np.argsort(-u.exp_value.values / c), c, B),
                "city_policy": bt.fill_by_order(np.argsort(-city_score), c, B, allowed=~u.is_city.values),
                "heuristic": bt.fill_by_order(np.argsort(-((u.yrs_since_sale <= 2) * 10 +
                                                          u.bf_ellis_omi_3y.clip(0, 5)).values / c), c, B)}
        rv = np.array([u.value.values[bt.fill_by_order(rng.permutation(len(u)), c, B)].sum() for _ in range(n_random)])
        bs = bt.cluster_boot(u, sels, n_boot, seed=m)
        r = {"mult": m, "budget": B, "random": float(rv.mean())}
        for k, s in sels.items():
            r[k] = float(u.value[s].sum())
        r["cand_vs_city_ci"] = bt.ci(bs["cand"] - bs["city_policy"])
        r["cand_vs_heur_ci"] = bt.ci(bs["cand"] - bs["heuristic"])
        out.append(r)
    t = pd.DataFrame(out)
    gm = lambda a, b: float(np.exp(np.mean(np.log((t[a] + 0.5) / (t[b] + 0.5)))))
    summary = {"score_vs_city": gm("cand", "city_policy"), "score_vs_heuristic": gm("cand", "heuristic"),
               "score_vs_random": gm("cand", "random"), "beats_city_all_budgets": bool((t.cand > t.city_policy).all())}
    return t, summary


def run(cand, split="dev", final=False):
    if split == "test" and not final:
        raise SystemExit("test split needs --final (and gets logged)")
    T = SPLITS[split]["T"]
    t0 = time.time()
    ctx = make_ctx(T)
    if cand == "heuristic":
        score = heuristic(ctx)
    else:
        score = np.asarray(importlib.import_module(f"models.risk.cands.{cand}").fit_predict(ctx), float)
    assert len(score) == len(ctx.score), "fit_predict must return one value per ctx.score row"
    u, keep = universe(ctx, split)
    table, summary = evaluate(u, score[keep])
    summary.update(cand=cand, split=split, T=T, minutes=round((time.time() - t0) / 60, 1),
                   universe_n=len(u), city_n=int(u.is_city.sum()), total_households=float(u.value.sum()))
    REP.mkdir(parents=True, exist_ok=True)
    table.assign(cand=cand, split=split).to_csv(REP / f"harness_{split}_{cand}.csv", index=False)
    lb = REP / f"leaderboard_{split}.csv"
    row = pd.DataFrame([{k: v for k, v in summary.items()}])
    if split == "test":
        row["commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        row.to_csv(REP / "test_log.csv", mode="a", header=not (REP / "test_log.csv").exists(), index=False)
    row.to_csv(lb, mode="a", header=not lb.exists(), index=False)
    return table, summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, help="comma-separated module names in models/risk/cands, or heuristic")
    ap.add_argument("--split", default="dev", choices=list(SPLITS))
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    for c in a.cand.split(","):
        table, s = run(c, a.split, a.final)
        print(table.round(2).to_string(index=False))
        print(json.dumps(s, indent=1))
