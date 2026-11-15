"""Calibrated rule of thumb only: the rule's 12 cells mapped to recent households/yr (reference)."""
from models.risk.cands.b_lib import calibrated_heuristic


def fit_predict(ctx):
    return calibrated_heuristic(ctx)
