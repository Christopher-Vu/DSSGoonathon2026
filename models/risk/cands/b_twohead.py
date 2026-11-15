"""Two heads (buyout h=1 on 2014+, block-face h=2 recency-weighted), sale features, monotone."""
from models.risk.cands.b_lib import fit_predict_twohead


def fit_predict(ctx):
    return fit_predict_twohead(ctx)
