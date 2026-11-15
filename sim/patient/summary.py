"""sim_summary.json: medians, deltas, %-lost-30-min overall and by subgroup."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .pois import CLASSES

GROUPS = ["lang", "has_car", "age_band", "income_band", "disability", "event_type"]


def wquantile(x: np.ndarray, w: np.ndarray, q: float = 0.5) -> float:
    x, w = np.asarray(x, float), np.asarray(w, float)
    m = ~np.isnan(x)
    x, w = x[m], w[m]
    if not len(x):
        return float("nan")
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w)
    return float(x[min(np.searchsorted(c, q * c[-1]), len(x) - 1)])


def wmedian(x, w) -> float:
    return wquantile(x, w, 0.5)


def _num(v: float):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else ("inf" if np.isinf(v) else round(float(v), 3))


def block(df: pd.DataFrame, cap: float) -> dict:
    w = df["weight"].to_numpy(float)
    W = w.sum()
    out = {"n": int(len(df)), "weight": round(float(W), 2)}
    for c in CLASSES:
        b, a = df[f"t_{c}_before"].to_numpy(float), df[f"t_{c}_after"].to_numpy(float)
        bc, ac = np.minimum(b, cap), np.minimum(a, cap)  # inf -> cap: deltas are lower bounds
        d = ac - bc
        lost = df[f"acc30_{c}_before"].to_numpy() & ~df[f"acc30_{c}_after"].to_numpy()
        gained = ~df[f"acc30_{c}_before"].to_numpy() & df[f"acc30_{c}_after"].to_numpy()
        had = df[f"acc30_{c}_before"].to_numpy()
        out[c] = {
            "median_before": _num(wmedian(b, w)), "median_after": _num(wmedian(a, w)),
            "median_delta_capped": _num(wmedian(d, w)), "mean_delta_capped": _num(float(np.average(d, weights=w))),
            "p90_delta_capped": _num(wquantile(d, w, 0.9)),
            "pct_lost30": _num(100 * w[lost].sum() / W), "pct_gained30": _num(100 * w[gained].sum() / W),
            "pct_lost30_of_had": _num(100 * w[lost].sum() / w[had].sum()) if w[had].sum() else None,
            "pct_acc30_before": _num(100 * w[had].sum() / W),
            "pct_acc30_after": _num(100 * w[df[f"acc30_{c}_after"].to_numpy()].sum() / W),
            "pct_unreachable_after": _num(100 * w[np.isinf(a)].sum() / W),
        }
    lost_any = np.zeros(len(df), bool)
    for c in CLASSES:
        lost_any |= df[f"acc30_{c}_before"].to_numpy() & ~df[f"acc30_{c}_after"].to_numpy()
    out["pct_lost30_any"] = _num(100 * w[lost_any].sum() / W)
    return out


def delta_hist(df: pd.DataFrame, cls: str, cap: float, bins=None) -> dict:
    bins = bins or [-120, -60, -30, -15, -5, 5, 15, 30, 60, 120]
    d = np.minimum(df[f"t_{cls}_after"], cap) - np.minimum(df[f"t_{cls}_before"], cap)
    h, e = np.histogram(d, bins=bins, weights=df["weight"])
    return {"bin_edges": [float(x) for x in e], "weight": [round(float(x), 3) for x in h]}


def summarize(res: pd.DataFrame, cap: float) -> dict:
    out = {}
    for scen, df in res.groupby("scenario"):
        s = {"overall": block(df, cap), "by": {}, "delta_hist": {c: delta_hist(df, c, cap) for c in CLASSES}}
        for g in GROUPS:
            if g in df:
                s["by"][g] = {str(k): block(v, cap) for k, v in df.groupby(g)}
        s["mode_share"] = df.groupby("mode")["weight"].sum().div(df["weight"].sum()).round(4).to_dict()
        out[scen] = s
    return out


def headline(summary: dict, scenario: str) -> str:
    o = summary["scenarios"][scenario]["overall"]
    c, ph = o["clinic"], o["pharmacy"]
    return (f"[{scenario}] Each displacement adds a median {c['median_delta_capped']} min to the nearest open clinic "
            f"({ph['median_delta_capped']} min to a pharmacy); {c['pct_lost30']}% lose 30-minute clinic access, "
            f"{o['pct_lost30_any']}% lose it for at least one of clinic/pharmacy/grocery.")
