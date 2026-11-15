"""Ablation: b_twohead + only the WM parcel next-year head outputs (no embedding PCA)."""
from models.risk.cands.b_lib import BASE, SALE, fit_predict_twohead
from models.risk.cands.c_lib import attach_wm


def fit_predict(ctx):
    wm = attach_wm(ctx, use_pca=False)
    return fit_predict_twohead(ctx, feats=BASE + SALE + wm)
