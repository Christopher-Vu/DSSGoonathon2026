"""b_twohead without monotone constraints."""
from models.risk.cands.b_lib import fit_predict_twohead


def fit_predict(ctx):
    return fit_predict_twohead(ctx, mono=False)
