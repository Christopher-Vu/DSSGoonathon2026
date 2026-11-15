"""Ablation: two-head LightGBM on world-model features only (PCA + parcel head), no hand-made features, no rule blend."""
from models.risk.cands.b_lib import fit_predict_twohead
from models.risk.cands.c_lib import attach_wm


def fit_predict(ctx):
    wm = attach_wm(ctx)
    return fit_predict_twohead(ctx, feats=wm, mono=False, flat_tenants=True)
