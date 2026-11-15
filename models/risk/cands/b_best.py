"""Session B best LightGBM candidate (frozen recipe, chosen on dev):
0.5 x two-head LightGBM (buyout head h=1 on t>=2014 with flat E[tenants]; block-face Poisson head h=2,
recency-weighted; sale-recency features; monotone) + 0.5 x calibrated rule of thumb.
use_emb=True adds a 16-dim PCA of the world-model parcel embeddings as features (for C's ablation:
b_best vs c_wm_gbm). Import and call fit_predict(ctx, use_emb=True), or use cand b_best_emb.
"""
from models.risk.cands.b_lib import blend


def fit_predict(ctx, use_emb=False, emb_k=16):
    return blend(ctx, lam=0.5, flat_tenants=True, emb_k=emb_k if use_emb else 0)
