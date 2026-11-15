"""Frozen-embedding probes: ridge (regression) / logistic (binary), hand-made vs embeddings vs both,
with paired bootstrap CIs. Also a frozen next-year probe used as the comparable L_next error."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, r2_score
from sklearn.preprocessing import StandardScaler

from .config import N_FEAT

SETS = ("hand", "emb", "both")


# ---------------------------------------------------------------- feature tables
def handmade_frame(store, seq_ids=None) -> pd.DataFrame:
    """Hand-made per-(cell, year) features: log token counts per type and per-type feature means."""
    s = np.arange(store.n_seq) if seq_ids is None else np.asarray(seq_ids)
    cols, names = [np.log1p(store.counts[s])], [f"h_n_{t}" for t in store.types]
    m = store.means[s].reshape(len(s), -1)
    cols += [np.nan_to_num(m), np.isnan(m).astype(np.float32)]
    names += [f"h_mean_{t}_f{j}" for t in store.types for j in range(N_FEAT)]
    names += [f"h_nan_{t}_f{j}" for t in store.types for j in range(N_FEAT)]
    X = np.concatenate(cols, axis=1)
    keep = X.std(0) > 0
    df = pd.DataFrame(X[:, keep], columns=[n for n, k in zip(names, keep) if k])
    df.insert(0, "year", store.seq_year[s].astype(np.int16))
    df.insert(0, "cell", store.seq_cell[s])
    return df


def load_handmade(cfg: dict, store) -> pd.DataFrame:
    p = cfg["paths"].get("probe_features")
    if p and Path(p).exists():
        df = pd.read_parquet(p)
        df["year"] = df["year"].astype(np.int16)
        return df
    return handmade_frame(store)


def emb_frame(cells, years, E: np.ndarray, prefix="e") -> pd.DataFrame:
    df = pd.DataFrame(E.astype(np.float32), columns=[f"{prefix}{i}" for i in range(E.shape[1])])
    df.insert(0, "year", np.asarray(years).astype(np.int16))
    df.insert(0, "cell", np.asarray(cells))
    return df


# ---------------------------------------------------------------- probes
def _metric(task, y, p):
    if task == "binary":
        return average_precision_score(y, p) if 0 < y.sum() < len(y) else np.nan
    return r2_score(y, p)


def _fit(task, alpha, X, y):
    if task == "binary":
        return LogisticRegression(C=1.0 / alpha, max_iter=2000).fit(X, y)
    return Ridge(alpha=alpha).fit(X, y)


def _pred(task, m, X):
    return m.predict_proba(X)[:, 1] if task == "binary" else m.predict(X)


def fit_probe(task, alphas, Xtr, ytr, Xtu, ytu, Xev):
    sc = StandardScaler().fit(Xtr)
    Xtr, Xtu, Xev = sc.transform(Xtr), sc.transform(Xtu), sc.transform(Xev)
    if task == "binary" and len(np.unique(ytr)) < 2:
        return np.full(len(Xev), ytr.mean()), np.nan
    best, best_a = -np.inf, alphas[0]
    for a in alphas:
        sc_a = _metric(task, ytu, _pred(task, _fit(task, a, Xtr, ytr), Xtu))
        if np.isfinite(sc_a) and sc_a > best:
            best, best_a = sc_a, a
    return _pred(task, _fit(task, best_a, Xtr, ytr), Xev), best_a


def run_probes(hand: pd.DataFrame, emb: pd.DataFrame, targets: pd.DataFrame, year: int, alphas,
               n_boot=1000, seed=0, fit_split="train", tune_split="val", eval_split="val"):
    """Returns (per target x feature-set table, summary dict). Rows are paired: only (cell, year)
    present in hand, emb and targets are used, so every comparison is on identical rows."""
    rng = np.random.default_rng(seed)
    tg = targets[targets["year"] == year]
    hcols = [c for c in hand.columns if c not in ("cell", "year")]
    ecols = [c for c in emb.columns if c not in ("cell", "year")]
    feat = hand.merge(emb, on=["cell", "year"], how="inner")
    rows, boot_lifts = [], {"emb": [], "both": []}
    for (name, task), g in tg.groupby(["target", "task"], sort=True):
        d = g.merge(feat, on=["cell", "year"], how="inner")
        parts = {s: d[d["split"] == s] for s in (fit_split, tune_split, eval_split)}
        if any(len(p) < 5 for p in parts.values()):
            continue
        ytr, ytu, yev = (parts[s]["value"].to_numpy(np.float64) for s in (fit_split, tune_split, eval_split))
        preds, alph = {}, {}
        for s, cols in (("hand", hcols), ("emb", ecols), ("both", hcols + ecols)):
            X = {k: parts[k][cols].to_numpy(np.float64) for k in parts}
            preds[s], alph[s] = fit_probe(task, alphas, X[fit_split], ytr, X[tune_split], ytu, X[eval_split])
        point = {s: _metric(task, yev, preds[s]) for s in SETS}
        B = np.full((n_boot, 3), np.nan)
        n = len(yev)
        for b in range(n_boot):
            i = rng.integers(0, n, n)
            B[b] = [_metric(task, yev[i], preds[s][i]) for s in SETS]
        metric = "pr_auc" if task == "binary" else "r2"
        base_rate = float(yev.mean()) if task == "binary" else np.nan
        for k, s in enumerate(SETS):
            lo, hi = np.nanpercentile(B[:, k], [2.5, 97.5])
            row = dict(target=name, task=task, metric=metric, features=s, value=point[s], ci_lo=lo, ci_hi=hi,
                       alpha=alph[s], n_fit=len(ytr), n_eval=n, base_rate=base_rate)
            if s != "hand":
                L = B[:, k] - B[:, 0]
                row.update(lift=point[s] - point["hand"], lift_lo=np.nanpercentile(L, 2.5), lift_hi=np.nanpercentile(L, 97.5))
                boot_lifts[s].append(L)
            rows.append(row)
    table = pd.DataFrame(rows)
    summary = {}
    for s in ("emb", "both"):
        if boot_lifts[s]:
            M = np.nanmean(np.stack(boot_lifts[s]), axis=0)
            pt = table[table["features"] == s]["lift"].mean()
            summary[f"mean_lift_{s}"] = float(pt)
            summary[f"mean_lift_{s}_se"] = float(np.nanstd(M))
            summary[f"mean_lift_{s}_lo"], summary[f"mean_lift_{s}_hi"] = map(float, np.nanpercentile(M, [2.5, 97.5]))
    summary["n_targets"] = int(table["target"].nunique()) if len(table) else 0
    return table, summary


# ---------------------------------------------------------------- frozen next-year probe
def next_targets(store, seq_ids):
    nxt = store.next_seq[seq_ids]
    return np.concatenate([np.log1p(store.counts[nxt]), np.nan_to_num(store.means[nxt].reshape(len(nxt), -1))], axis=1)


def next_year_probe(store, E: np.ndarray, alpha: float = 10.0) -> dict:
    """Ridge from frozen year-t cell embeddings to the year-t+1 summary. Train pairs have t+1 < cutoff;
    validation pairs are (cutoff-1 -> cutoff). Returned as standardized MSE; comparable across λ."""
    ok = store.next_seq >= 0
    tr = np.where(ok & (store.seq_year + 1 < store.cutoff))[0]
    va = np.where(ok & (store.seq_year + 1 == store.cutoff))[0]
    if len(tr) < 5 or len(va) < 5:
        return {"next_probe_mse": np.nan, "next_probe_mse_hand": np.nan, "next_probe_mse_emb_only": np.nan}
    Ytr, Yva = next_targets(store, tr), next_targets(store, va)
    mu, sd = Ytr.mean(0), Ytr.std(0)
    keep = sd > 0
    Ytr, Yva = (Ytr[:, keep] - mu[keep]) / sd[keep], (Yva[:, keep] - mu[keep]) / sd[keep]
    out = {}
    hand = handmade_frame(store).drop(columns=["cell", "year"]).to_numpy(np.float64)
    # Center features within each year: the [CELL] output carries the year embedding, and the val
    # input year (cutoff-1) never appears among the probe's training inputs, so an uncentered linear
    # probe extrapolates along the year direction.
    E = np.asarray(E, dtype=np.float64).copy()
    for y in np.unique(store.seq_year):
        r = store.seq_year == y
        E[r] -= E[r].mean(0)
        hand[r] -= hand[r].mean(0)
    for key, X in (("next_probe_mse", np.concatenate([hand, E], axis=1)), ("next_probe_mse_hand", hand),
                   ("next_probe_mse_emb_only", E)):
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=alpha).fit(sc.transform(X[tr]), Ytr)
        out[key] = float(((m.predict(sc.transform(X[va])) - Yva) ** 2).mean())
    return out
