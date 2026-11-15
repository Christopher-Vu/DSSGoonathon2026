"""0.5 x b_flat + 0.5 x calibrated rule of thumb."""
from models.risk.cands.b_lib import blend


def fit_predict(ctx):
    return blend(ctx, flat_tenants=True)
