"""reports/sim figures: migration flow map, delta histogram, %-lost-30 by subgroup."""
from __future__ import annotations

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import p  # noqa: E402
from .pois import CLASSES  # noqa: E402
from .pums import region_pumas  # noqa: E402

INK, MUTED, GRID = "#1f2933", "#7b8794", "#e4e7eb"
ACCENT, WARM = "#2f6fdf", "#d9480f"
CLS_COLOR = {"clinic": "#2f6fdf", "pharmacy": "#7048e8", "grocery": "#2b8a3e"}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def flow_map(res: pd.DataFrame, cfg: dict, scenario: str, out):
    pu = gpd.read_file(f"zip://{p(cfg, 'puma_shapes')}").to_crs("EPSG:4326")
    reg = pu[pu["PUMACE20"].isin(region_pumas(pu, cfg["migration"]["region_counties"]))]
    df = res[res["scenario"] == scenario]
    flows = df.groupby("dest_puma")["weight"].sum()
    flows.index = flows.index.str[2:]
    reg = reg.assign(flow=reg["PUMACE20"].map(flows).fillna(0.0))
    reg = reg.assign(share=reg["flow"] / reg["flow"].sum())
    o_lat, o_lon = np.average(df["origin_lat"], weights=df["weight"]), np.average(df["origin_lon"], weights=df["weight"])
    fig, ax = plt.subplots(figsize=(8, 8))
    reg.plot(column="share", cmap="Blues", ax=ax, edgecolor="white", linewidth=0.4, legend=True,
             legend_kwds={"label": "share of displaced households", "shrink": 0.55})
    sf = reg[reg["PUMACE20"].isin({"07507", "07508", "07509", "07510", "07511", "07512", "07513", "07514"})]
    sf.plot(ax=ax, color="#ffe8cc", edgecolor=WARM, linewidth=0.6)
    top = reg.sort_values("flow", ascending=False).head(12)
    pts = top.geometry.representative_point()
    mx = top["share"].max()
    for (_, r), pt in zip(top.iterrows(), pts):
        ax.annotate("", xy=(pt.x, pt.y), xytext=(o_lon, o_lat),
                    arrowprops=dict(arrowstyle="-|>", color=WARM, lw=0.6 + 4 * r["share"] / mx, alpha=0.75,
                                    connectionstyle="arc3,rad=0.15", shrinkA=0, shrinkB=2))
    x0, y0, x1, y1 = reg.total_bounds
    ax.set_xlim(max(x0, -123.2), x1)  # drop the Farallon Islands (part of an SF PUMA)
    ax.set_ylim(y0, y1)
    ax.set_axis_off()
    ax.set_title(f"Where displaced SF renters go — {scenario}\n(PUMS 2019–23 SF out-movers, top 12 PUMAs arrowed)",
                 color=INK, fontsize=11, loc="left")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def delta_hist(res: pd.DataFrame, cfg: dict, scenario: str, out):
    cap = float(cfg["routing"]["max_minutes"])
    df = res[res["scenario"] == scenario]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    bins = np.arange(-60, 125, 5)
    for ax, c in zip(axes, CLASSES):
        d = np.minimum(df[f"t_{c}_after"], cap) - np.minimum(df[f"t_{c}_before"], cap)
        ax.hist(np.clip(d, -60, 120), bins=bins, weights=df["weight"] / df["weight"].sum() * 100,
                color=CLS_COLOR[c], alpha=0.9)
        med = np.median(d)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.axvline(med, color=INK, lw=1.2, ls="--")
        ax.text(med, ax.get_ylim()[1] * 0.92, f" median {med:+.0f} min", color=INK, fontsize=9)
        ax.set_title(f"nearest open {c}", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("Δ minutes (after − before, capped at 120)", color=INK, fontsize=9)
        _style(ax)
    axes[0].set_ylabel("% of households", color=INK, fontsize=9)
    fig.suptitle(f"Change in travel time to care after displacement — {scenario}", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def lost_by_group(summary: dict, scenario: str, out, cls: str = "clinic"):
    by = summary["scenarios"][scenario]["by"]
    groups = ["lang", "has_car", "age_band", "income_band", "disability"]
    labels, vals, ns = [], [], []
    for g in groups:
        for k, v in by.get(g, {}).items():
            labels.append(f"{g}: {k}")
            vals.append(v[cls]["pct_lost30"] or 0.0)
            ns.append(v["n"])
    overall = summary["scenarios"][scenario]["overall"][cls]["pct_lost30"] or 0.0
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(labels) + 1.2))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=[WARM if v > overall else ACCENT for v in vals], height=0.65)
    ax.axvline(overall, color=INK, lw=1, ls="--")
    ax.text(overall, len(labels) - 0.2, f" overall {overall:.1f}%", color=INK, fontsize=8, va="bottom")
    for yi, v, n in zip(y, vals, ns):
        ax.text(v, yi, f" {v:.1f}% (n={n})", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel(f"% of households losing 30-minute access to an open {cls}", color=INK, fontsize=9)
    _style(ax)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Who loses 30-minute {cls} access — {scenario}", color=INK, fontsize=11, loc="left")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_figures(res: pd.DataFrame, summary: dict, cfg: dict):
    d = p(cfg, "reports")
    d.mkdir(parents=True, exist_ok=True)
    for scen in sorted(res["scenario"].unique()):
        flow_map(res, cfg, scen, d / f"flow_map__{scen}.png")
        delta_hist(res, cfg, scen, d / f"delta_hist__{scen}.png")
        for c in CLASSES:
            lost_by_group(summary, scen, d / f"lost30_{c}_by_group__{scen}.png", c)
    print(f"figures -> {d}")


if __name__ == "__main__":
    import argparse
    import json

    from .config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sim.yaml")
    cfg = load_config(ap.parse_args().config)
    make_figures(pd.read_parquet(p(cfg, "out_results")), json.loads(p(cfg, "out_summary").read_text()), cfg)
