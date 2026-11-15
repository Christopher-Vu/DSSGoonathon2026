"""World model + rule: 0.5 x two-head LightGBM on out-of-time WM parcel features (c_wm_oos_only)
+ 0.5 x calibrated rule of thumb (same blend and weight as b_best; weight not tuned)."""
from models.risk.cands.b_lib import calibrated_heuristic
from models.risk.cands.c_wm_oos_only import fit_predict as wm_only


def fit_predict(ctx, lam=0.5):
    return lam * wm_only(ctx) + (1 - lam) * calibrated_heuristic(ctx)
