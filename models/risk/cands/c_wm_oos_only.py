"""Ablation: two-head LightGBM on the out-of-time WM parcel head features only (+ units, bf_share for scale)."""
from models.risk.cands.b_lib import fit_predict_twohead
from models.risk.cands.c_lib import attach_oos


def fit_predict(ctx):
    wm = attach_oos(ctx)
    return fit_predict_twohead(ctx, feats=wm + ["units", "bf_share"], mono=False, flat_tenants=True)
