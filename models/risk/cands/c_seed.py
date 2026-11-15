"""Diagnostic: rerun a candidate with a different LightGBM seed (env C_SEED, C_BASE) to measure dev noise."""
import importlib
import os

from models.risk.cands import b_lib

_orig = b_lib._train


def _train_seeded(params, X, y, w, rounds, mono=None):
    return _orig({**params, "seed": int(os.environ.get("C_SEED", "0"))}, X, y, w, rounds, mono)


def fit_predict(ctx):
    b_lib._train = _train_seeded
    try:
        return importlib.import_module(f"models.risk.cands.{os.environ.get('C_BASE', 'b_best')}").fit_predict(ctx)
    finally:
        b_lib._train = _orig
