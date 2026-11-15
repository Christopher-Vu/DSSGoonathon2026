"""Aggregate the λ sweep and pick λ.

  python -m models.world_model.sweep_select [--config configs/wm.yaml]

SELECTION RULE (uses only year=select_year (2016) val-split probe rows + the (C-1 -> C) next-year
validation error; never the 2018 test rows):
  1. score(λ) = mean over probe targets of [metric(embeddings-only) - metric(hand-made)]
     on year=2016, split=val (R² for regression, PR-AUC for binary).
  2. candidates = every λ whose score >= best_score - SE(best), SE from the paired bootstrap.
  3. among candidates pick the lowest frozen next-year probe MSE (hand-made + embedding -> year t+1
     summary, ridge) on the held-out (2017 -> 2018) pairs;
     remaining ties go to the smaller λ.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config

RULE = __doc__.split("SELECTION RULE")[1].strip()
LAMS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def collect(cfg: dict, cutoff: int, budget: str = "sweep") -> pd.DataFrame:
    root = Path(cfg["paths"]["ckpt_dir"]) / f"cutoff={cutoff}" / budget
    rows = []
    for p in sorted(root.glob("lam=*/metrics.json")):
        with open(p) as f:
            m = json.load(f)
        rows.append(m)
    if not rows:
        raise FileNotFoundError(f"no sweep runs under {root}")
    return pd.DataFrame(rows).sort_values("lam").reset_index(drop=True)


def select_lambda(df: pd.DataFrame) -> tuple[float, dict]:
    d = df.dropna(subset=["mean_lift_emb"]).copy()
    if d.empty:
        raise ValueError("no run has probe scores; cannot select lam")
    b = d["mean_lift_emb"].idxmax()
    thresh = d.loc[b, "mean_lift_emb"] - d.loc[b, "mean_lift_emb_se"]
    cand = d[d["mean_lift_emb"] >= thresh].copy()
    cand["_nx"] = cand["next_probe_mse"].fillna(np.inf)
    pick = cand.sort_values(["_nx", "lam"]).iloc[0]
    why = {"best_score_lam": float(d.loc[b, "lam"]), "best_score": float(d.loc[b, "mean_lift_emb"]),
           "threshold": float(thresh), "candidates": [float(x) for x in cand["lam"]],
           "selected_next_probe_mse": float(pick["next_probe_mse"])}
    return float(pick["lam"]), why


def select_lambda_parcel(df: pd.DataFrame) -> tuple[float, dict]:
    cols = [c for c in df.columns if c.startswith("val_parcel_skill_")]
    if not cols:
        raise ValueError("no parcel-head skill in metrics; was the sweep run with the interim event tables?")
    d = df.assign(parcel_skill=df[cols].mean(axis=1)).dropna(subset=["parcel_skill"])
    pick = d.sort_values(["parcel_skill", "lam"], ascending=[False, True]).iloc[0]
    return float(pick["lam"]), {"rule": "parcel", "parcel_skill": float(pick["parcel_skill"]),
                                "skills": {float(l): float(v) for l, v in zip(d["lam"], d["parcel_skill"])}}


def plot(df: pd.DataFrame, path: Path, selected: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4))
    x = df["lam"].to_numpy()
    y, se = df["mean_lift_emb"].to_numpy(), df["mean_lift_emb_se"].to_numpy()
    ax.plot(x, y, "o-", color="#1f6f8b", label="probe lift (emb − hand)")
    ax.fill_between(x, y - se, y + se, color="#1f6f8b", alpha=0.15)
    if "mean_lift_both" in df:
        ax.plot(x, df["mean_lift_both"], "s--", color="#1f6f8b", alpha=0.6, label="probe lift (both − hand)")
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(selected, color="#c0392b", lw=1, ls=":", label=f"selected λ={selected:g}")
    ax.set_xlabel("λ  (0 = masked only, 1 = next-year only)")
    ax.set_ylabel("mean probe lift, year-2016 val")
    ax2 = ax.twinx()
    ax2.plot(x, df["next_probe_mse"], "^-", color="#e67e22", label="next-year error, hand+emb (frozen probe)")
    if "next_probe_mse_hand" in df:
        ax2.axhline(df["next_probe_mse_hand"].iloc[0], color="#e67e22", ls="--", lw=0.8, label="next-year error, hand only")
    ax2.set_ylabel("next-year MSE (2017→2018)", color="#e67e22")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")
    ax.set_title("λ sweep: L = (1−λ)·L_MTM + λ·L_next")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wm.yaml")
    ap.add_argument("--cutoff", type=int, default=2018)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    df = collect(cfg, a.cutoff)
    missing = sorted(set(LAMS) - set(np.round(df["lam"], 6)))
    if missing:
        print(f"warning: missing sweep runs for lam={missing}")
    lam, why = select_lambda_parcel(df) if a.rule == "parcel" else select_lambda(df)
    rep = Path(cfg["paths"]["reports"])
    rep.mkdir(parents=True, exist_ok=True)
    cols = ["lam", "mean_lift_emb", "mean_lift_emb_se", "mean_lift_emb_lo", "mean_lift_emb_hi", "mean_lift_both",
            "mean_lift_both_se", "next_probe_mse", "next_probe_mse_hand", "next_probe_mse_emb_only", "val_next_head", "best_val_loss",
            "train_minutes", "stopped", "n_params"] + sorted(c for c in df.columns if c.startswith("val_parcel_skill_"))
    per_target = [c for c in df.columns if "__" in c]
    df[[c for c in cols if c in df] + per_target].to_csv(rep / "lambda_sweep.csv", index=False)
    plot(df, rep / "lambda_sweep.png", lam)
    if "mean_lift_emb" not in df:
        df["mean_lift_emb"] = np.nan
        df["mean_lift_emb_se"] = np.nan
    if "next_probe_mse" not in df:
        df["next_probe_mse"] = np.nan
    src = Path(cfg["paths"]["ckpt_dir"]) / f"cutoff={a.cutoff}" / "sweep" / f"lam={lam:g}" / "probes_val.csv"
    if src.exists():
        pd.read_csv(src).to_csv(rep / "probes.csv", index=False)
    sel = {"lam": lam, "cutoff": a.cutoff, "rule": RULE if a.rule == "probe" else "parcel (see docstring)", **why}
    with open(rep / "selected.json", "w") as f:
        json.dump(sel, f, indent=2)
    print(df[[c for c in cols[:9] if c in df]].to_string(index=False))
    print(f"selected lam = {lam:g}  ({why})")


if __name__ == "__main__":
    main()
