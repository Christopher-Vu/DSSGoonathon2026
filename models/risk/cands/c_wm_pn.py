"""Ablation: b_best + only the WM parcel next-year head outputs (no embedding PCA)."""
from models.risk.cands.c_wm_gbm import fit_predict as _f


def fit_predict(ctx):
    return _f(ctx, use_pca=False)
