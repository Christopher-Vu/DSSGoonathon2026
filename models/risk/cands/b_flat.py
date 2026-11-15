"""b_twohead with flat E[tenants | buyout] (not scaled by units)."""
from models.risk.cands.b_lib import fit_predict_twohead


def fit_predict(ctx):
    return fit_predict_twohead(ctx, flat_tenants=True)
