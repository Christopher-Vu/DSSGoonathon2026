"""Buyout head only (block-face head weight 0): tests whether the buyout mechanism alone carries the score."""
from models.risk.cands.b_lib import fit_predict_twohead


def fit_predict(ctx):
    return fit_predict_twohead(ctx, w_bf=0.0)
