"""Session C: world-model features for risk candidates.

attach_wm(ctx) joins, per (blklot, t):
  wm_pc0..k   PCA of the cutoff-T parcel embeddings (PCA fit on embedding rows with year <= T only)
  pn_*        the WM parcel next-year head: expected t+1 sales / buyout tenants / block-face Ellis-OMI
Everything comes from ctx.emb (cutoff == T, year <= T), so nothing after T enters.
"""
import numpy as np
import pandas as pd

PN = ["pn_sale", "pn_buyout", "pn_bf_ellis_omi"]


def wm_frame(ctx, n_pca=16, seed=0):
    from sklearn.decomposition import PCA

    e = ctx.emb("parcel")
    assert (e.cutoff == ctx.T).all() and e.year.max() <= ctx.T
    ecols = [c for c in e.columns if c.startswith("e") and c[1:].isdigit()]
    X = e[ecols].to_numpy(np.float32)
    Z = PCA(n_components=n_pca, random_state=seed).fit_transform(X - X.mean(0))
    out = pd.DataFrame(Z.astype(np.float32), columns=[f"wm_pc{i}" for i in range(n_pca)])
    out.insert(0, "t", e.year.astype(int).to_numpy())
    out.insert(0, "blklot", e.blklot.astype(str).to_numpy())
    try:
        pn = ctx.emb("parcel_next")
        assert (pn.cutoff == ctx.T).all() and pn.year.max() <= ctx.T
        pn = pn.rename(columns={"year": "t"})[["blklot", "t", *PN]].astype({"blklot": str, "t": int})
        for c in PN:
            pn[c] = np.log(pn[c].clip(lower=1e-6))  # log-rates
        out = out.merge(pn, on=["blklot", "t"], how="left")
    except FileNotFoundError:
        pass
    return out.drop_duplicates(["blklot", "t"])


def attach_wm(ctx, n_pca=16, use_pca=True, use_pn=True):
    """Mutates ctx.rows (adds WM columns) and returns the list of added feature names."""
    w = wm_frame(ctx, n_pca)
    cols = ([f"wm_pc{i}" for i in range(n_pca)] if use_pca else []) + ([c for c in PN if c in w] if use_pn else [])
    r = ctx.rows.drop(columns=[c for c in w.columns if c in ctx.rows and c not in ("blklot", "t")])
    r = r.assign(blklot=r.blklot.astype(str)).merge(w[["blklot", "t", *cols]], on=["blklot", "t"], how="left")
    assert len(r) == len(ctx.rows)
    ctx.rows = r
    return cols


OOS = ["oos_sale", "oos_bf_ellis_omi"]


def attach_oos(ctx):
    """Out-of-time WM parcel head features: the year-t row comes from a model with cutoff t (rolling origin),
    so no row's feature was fit on its own future label. pn_buyout is dropped (no buyout targets before 2015,
    so rolling models with cutoff <= 2015 never trained that output)."""
    e = ctx.emb("parcel_oos")
    assert (e.cutoff == ctx.T).all() and (e.year == e.model_cutoff).all() and e.year.max() <= ctx.T
    f = e.rename(columns={"year": "t"}).astype({"blklot": str, "t": int}).drop_duplicates(["blklot", "t"])
    f["oos_sale"] = np.log(f.pn_sale.clip(lower=1e-6))
    f["oos_bf_ellis_omi"] = np.log(f.pn_bf_ellis_omi.clip(lower=1e-6))
    r = ctx.rows.drop(columns=[c for c in OOS if c in ctx.rows])
    r = r.assign(blklot=r.blklot.astype(str)).merge(f[["blklot", "t", *OOS]], on=["blklot", "t"], how="left")
    assert len(r) == len(ctx.rows)
    ctx.rows = r
    return OOS
