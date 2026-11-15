"""PUMS personas + the SF out-migration model.

Vintage note (verified 2026-11-14 on csv_pca/csv_hca 2019-2023 5-yr): every survey year's `PUMA` is coded
in 2020 PUMA definitions (all 281 CA codes match TIGER 2020 in every year) and `MIGPUMA` is 2020 MIGPUMA,
so no crosswalk is needed. SF is a single MIGPUMA (07500): destinations can be conditioned on income band,
not on origin SF PUMA.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

SF_MIGPUMA = "07500"
SF_PUMAS = {"07507", "07508", "07509", "07510", "07511", "07512", "07513", "07514"}
CA = "006"
LANG = {"1200": "Spanish", "1970": "Chinese", "2000": "Chinese", "2030": "Chinese", "2050": "Chinese",
        "1960": "Vietnamese", "2920": "Tagalog", "2910": "Tagalog"}
COUNTY_NAMES = {"001": "Alameda", "013": "Contra Costa", "041": "Marin", "055": "Napa", "075": "San Francisco",
                "081": "San Mateo", "085": "Santa Clara", "095": "Solano", "097": "Sonoma", "067": "Sacramento",
                "077": "San Joaquin"}


def band(x: float, bands) -> str:
    for lo, hi, name in bands:
        if lo <= x <= hi:
            return name
    return bands[-1][2]


def lang_of(lanp) -> str:
    if lanp is None or (isinstance(lanp, float) and np.isnan(lanp)) or str(lanp).strip() in ("", "nan"):
        return "English"
    return LANG.get(str(lanp).strip().zfill(4), "Other")


def region_pumas(puma_names: pd.DataFrame, counties: list[str]) -> set[str]:
    """PUMA codes whose 2020 name sits in one of the region counties (CA PUMAs nest in counties)."""
    names = [COUNTY_NAMES[c] for c in counties]
    pat = re.compile(r"^(" + "|".join(map(re.escape, names)) + r") County")
    return set(puma_names.loc[puma_names["NAMELSAD20"].str.match(pat), "PUMACE20"])


def _years(df, years):
    return df[df["year"].isin(years)] if years else df


def build_personas(h: pd.DataFrame, pp: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per SF renter household with persona attributes + WGTP."""
    pc = cfg["personas"]
    years = cfg["pums"].get("survey_years")
    hh = _years(h, years)
    hh = hh[hh["PUMA"].isin(SF_PUMAS) & (hh["TYPEHUGQ"] == "1") & (hh["NP"] > 0)]
    if cfg["pums"].get("renters_only", True):
        hh = hh[hh["TEN"] == "3"]
    per = pp[pp["SERIALNO"].isin(hh["SERIALNO"])]
    ref = per[per["RELSHIPP"] == "20"][["SERIALNO", "AGEP", "LANP"]]
    dis = per.groupby("SERIALNO")["DIS"].apply(lambda s: bool((s == "1").any())).rename("disability")
    out = hh.merge(ref, on="SERIALNO", how="inner").merge(dis, on="SERIALNO", how="left")
    inc = out["HINCP"].fillna(0) * out["ADJINC"] / 1e6
    out = pd.DataFrame({
        "serialno": out["SERIALNO"].to_numpy(),
        "puma": out["PUMA"].to_numpy(),
        "wgtp": out["WGTP"].astype(float).to_numpy(),
        "age": out["AGEP"].astype(int).to_numpy(),
        "income": inc.to_numpy(),
        "has_car": (out["VEH"].fillna(0) > 0).to_numpy(),
        "lang": [lang_of(x) for x in out["LANP"]],
        "disability": out["disability"].fillna(False).astype(bool).to_numpy(),
    })
    out["age_band"] = [band(a, pc["age_bands"]) for a in out["age"]]
    out["income_band"] = [band(x, pc["income_bands"]) for x in out["income"]]
    lo, hi = pc["working_age"]
    out["depart"] = np.where((out["age"] >= lo) & (out["age"] <= hi), "weekday_1800", "weekday_1000")
    return out


def movers(h: pd.DataFrame, pp: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Persons who lived in SF 1 year ago and now live elsewhere in CA, with household income band + tenure."""
    m = pp[(pp["MIG"] == "3") & (pp["MIGSP"] == CA) & (pp["MIGPUMA"] == SF_MIGPUMA)]
    m = m.merge(h[["SERIALNO", "TEN", "HINCP", "ADJINC"]].rename(columns={"ADJINC": "HADJ"}), on="SERIALNO")
    if cfg["pums"].get("renters_only", True):
        m = m[m["TEN"] == "3"]
    inc = m["HINCP"].fillna(0) * m["HADJ"] / 1e6
    return pd.DataFrame({
        "year": m["year"].to_numpy(), "puma": m["PUMA"].to_numpy(), "w": m["PWGTP"].astype(float).to_numpy(),
        "income_band": [band(x, cfg["personas"]["income_bands"]) for x in inc],
        "in_sf": m["PUMA"].isin(SF_PUMAS).to_numpy(),
    })


@dataclass
class MigrationModel:
    pumas: list[str]                 # support (region PUMAs, minus SF unless include_within_sf)
    probs: dict[str, np.ndarray]     # income_band -> P(dest puma)
    marginal: np.ndarray
    stats: dict

    def sample(self, income_band: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = np.empty(len(income_band), dtype=object)
        for b in np.unique(income_band):
            idx = np.where(income_band == b)[0]
            pr = self.probs.get(b, self.marginal)
            out[idx] = rng.choice(self.pumas, size=len(idx), p=pr)
        return out


def fit_migration(mv: pd.DataFrame, region: set[str], cfg: dict, years=None,
                  bands: list[str] | None = None) -> MigrationModel:
    mc = cfg["migration"]
    mv = mv if years is None else mv[mv["year"].isin(years)]
    total_w = mv["w"].sum()
    stats = {"movers_n": int(len(mv)), "movers_w": float(total_w),
             "share_within_sf": float(mv.loc[mv["in_sf"], "w"].sum() / total_w) if total_w else None}
    support = sorted(region - (set() if mc.get("include_within_sf") else SF_PUMAS))
    keep = mv[mv["puma"].isin(support)]
    out_mass = mv[~mv["in_sf"]]["w"].sum()
    stats["share_outside_sf_leaving_region"] = float(1 - keep[~keep["in_sf"]]["w"].sum() / out_mass) if out_mass else None
    stats["model_n"] = int(len(keep))
    idx = {p_: i for i, p_ in enumerate(support)}
    k = len(support)
    beta = 1.0  # tiny uniform floor so unseen region PUMAs stay possible
    cnt = np.zeros(k)
    np.add.at(cnt, keep["puma"].map(idx).to_numpy(dtype=np.int64), keep["w"].to_numpy(dtype=float))
    marginal = (cnt + beta * cnt.sum() / max(k, 1) / 100 * np.ones(k))
    marginal = marginal / marginal.sum()
    alpha = float(mc["smoothing_alpha"])
    probs = {}
    bands = bands or sorted(keep["income_band"].unique())
    for b in bands:
        sb = keep[keep["income_band"] == b]
        cb = np.zeros(k)
        np.add.at(cb, sb["puma"].map(idx).to_numpy(dtype=np.int64), sb["w"].to_numpy(dtype=float))
        n_eff = len(sb)  # pseudo-count in records, scaled to weight units
        scale = cb.sum() / n_eff if n_eff else 0.0
        pr = cb + alpha * scale * marginal if n_eff else marginal.copy()
        probs[b] = pr / pr.sum()
        stats[f"n_{b}"] = int(n_eff)
    return MigrationModel(support, probs, marginal, stats)


def jsd(p_: np.ndarray, q: np.ndarray) -> float:
    from scipy.spatial.distance import jensenshannon
    return float(jensenshannon(p_, q, base=2) ** 2)


def validate_holdout(mv: pd.DataFrame, region: set[str], cfg: dict, pop_w: dict[str, float] | None = None) -> dict:
    """Fit on all survey years except holdout; JSD of holdout destination shares vs model / baselines."""
    ho = cfg["pums"]["holdout_year"]
    yrs = sorted(set(mv["year"]) - {ho})
    model = fit_migration(mv, region, cfg, years=yrs)
    test = mv[(mv["year"] == ho) & mv["puma"].isin(model.pumas)]
    idx = {p_: i for i, p_ in enumerate(model.pumas)}

    def shares(df):
        c = np.zeros(len(model.pumas))
        np.add.at(c, df["puma"].map(idx).to_numpy(), df["w"].to_numpy())
        return c / c.sum() if c.sum() else c

    obs = shares(test)
    # model's implied overall share = mix of band distributions by holdout band mix
    mix = test.groupby("income_band")["w"].sum()
    pred = sum(model.probs.get(b, model.marginal) * w for b, w in mix.items()) / mix.sum()
    uni = np.ones(len(model.pumas)) / len(model.pumas)
    res = {"holdout_year": int(ho), "train_years": [int(y) for y in yrs], "holdout_n": int(len(test)),
           "jsd_model": jsd(obs, pred), "jsd_uniform_bay_area": jsd(obs, uni)}
    if pop_w:
        pw = np.array([pop_w.get(p_, 0.0) for p_ in model.pumas])
        res["jsd_population_prop"] = jsd(obs, pw / pw.sum())
    # bootstrap CI on the model / uniform gap (resample holdout records)
    rng = np.random.default_rng(0)
    gaps = []
    for _ in range(200):
        bs = test.sample(len(test), replace=True, random_state=int(rng.integers(1e9)))
        o = shares(bs)
        gaps.append(jsd(o, uni) - jsd(o, pred))
    res["jsd_gap_uniform_minus_model_ci95"] = [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))]
    return res
