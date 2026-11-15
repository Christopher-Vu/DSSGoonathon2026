"""b_twohead with h=2 buyout windows (t>=2013) for more positives."""
from models.risk.cands.b_lib import fit_predict_twohead


def fit_predict(ctx):
    return fit_predict_twohead(ctx, h_bo=2, t_bo_min=2013)
