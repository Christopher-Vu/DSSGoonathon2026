"""Ablation: the same two-head LightGBM on world-model features only (no hand-made features)."""
from models.risk.cands.b_lib import fit_predict_twohead
from models.risk.cands.c_lib import attach_wm


def fit_predict(ctx):
    wm = attach_wm(ctx)
    return fit_predict_twohead(ctx, feats=wm, mono=False)
