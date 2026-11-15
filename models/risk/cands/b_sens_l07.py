"""Sensitivity only (not for selection): b_best with lam=0.7."""
from models.risk.cands.b_lib import blend


def fit_predict(ctx):
    return blend(ctx, lam=0.7, flat_tenants=True)
