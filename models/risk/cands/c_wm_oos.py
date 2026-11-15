"""b_best + out-of-time world-model parcel head features (rolling-origin models; see models/world_model/oos.py)."""
from models.risk.cands.b_lib import BASE, SALE, blend
from models.risk.cands.c_lib import attach_oos


def fit_predict(ctx):
    wm = attach_oos(ctx)
    return blend(ctx, lam=0.5, flat_tenants=True, feats=BASE + SALE + wm)
