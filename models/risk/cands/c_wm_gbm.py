"""b_best (frozen, Session B) + world-model features: 16-dim PCA of cutoff-T parcel embeddings and the
WM parcel next-year head (expected t+1 sale / buyout tenants / block-face Ellis-OMI, as log-rates).
Same blend as b_best: 0.5 x two-head LightGBM + 0.5 x calibrated rule of thumb."""
from models.risk.cands.b_lib import BASE, SALE, blend
from models.risk.cands.c_lib import attach_wm


def fit_predict(ctx, use_pca=True, use_pn=True):
    wm = attach_wm(ctx, use_pca=use_pca, use_pn=use_pn)
    return blend(ctx, lam=0.5, flat_tenants=True, feats=BASE + SALE + wm)
