"""B's two-head LightGBM (b_twohead) + world-model features: PCA of parcel embeddings + WM parcel next-year head."""
from models.risk.cands.b_lib import BASE, SALE, fit_predict_twohead
from models.risk.cands.c_lib import attach_wm


def fit_predict(ctx, **kw):
    wm = attach_wm(ctx, **kw)
    return fit_predict_twohead(ctx, feats=BASE + SALE + wm)
