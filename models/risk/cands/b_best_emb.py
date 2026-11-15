"""b_best + world-model parcel embeddings (PCA 16)."""
from models.risk.cands.b_best import fit_predict as _fp


def fit_predict(ctx):
    return _fp(ctx, use_emb=True)
