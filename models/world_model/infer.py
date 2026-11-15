"""Checkpoint I/O and frozen-embedding inference."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import N_SPECIAL
from .data import TokenStore, chunk_tokens, collate, make_batches, select_tokens
from .leakage import LeakageError, assert_checkpoint_ok
from .model import WorldModel


def device_of(arg: str | None = None) -> torch.device:
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_dev(batch: dict, dev) -> dict:
    return {k: v.to(dev, non_blocking=True) for k, v in batch.items()}


def text_table(store: TokenStore, dev) -> torch.Tensor | None:
    if store.text_emb is None or len(store.text_emb) == 0:
        return None
    return torch.from_numpy(np.array(store.text_emb, dtype=np.float16)).to(dev)


def save_checkpoint(path: Path, model: WorldModel, cfg: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_cfg": cfg["model"], "meta": meta}, path)


def load_checkpoint(path: Path, cfg: dict, dev=None) -> tuple[WorldModel, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    assert_checkpoint_ok(meta, cfg)  # leakage lock at load time
    model = WorldModel(n_types=len(meta["types"]), **ck["model_cfg"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    if dev is not None:
        model.to(dev)
    return model, meta


def check_store_matches(store: TokenStore, meta: dict) -> None:
    if store.types != list(meta["types"]):
        raise LeakageError(f"store vocabulary {store.types} != checkpoint {meta['types']}")
    if store.cutoff != int(meta["cutoff"]):
        raise LeakageError(f"store cutoff {store.cutoff} != checkpoint cutoff {meta['cutoff']}")


@torch.no_grad()
def cell_embeddings(model: WorldModel, store: TokenStore, cfg: dict, dev, seq_ids=None) -> np.ndarray:
    """[CELL] output for every (cell, year) sequence (deterministic subsample if > max_tokens)."""
    model.eval()
    cap = cfg["model"]["max_tokens"]
    seq_ids = np.arange(store.n_seq) if seq_ids is None else np.asarray(seq_ids)
    lens = np.minimum(store.lengths(), cap - 1) + 1
    tt = text_table(store, dev)
    out = np.zeros((store.n_seq, model.d), dtype=np.float32)
    for b in make_batches(lens, seq_ids, cfg["train"]["batch_tokens"], cfg["train"]["max_batch_seqs"]):
        batch = to_dev(collate(store, b, [select_tokens(store, s, cap, None) for s in b]), dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda" and cfg["train"]["bf16"]):
            h = model(batch, text_table=tt)
        out[b] = h[:, 0].float().cpu().numpy()
    return out[seq_ids]


@torch.no_grad()
def parcel_embeddings(model: WorldModel, store: TokenStore, cfg: dict, dev, years=None, blklots=None) -> pd.DataFrame:
    """Contextual output of every parcel token (chunked so cells with >511 tokens are fully covered)."""
    model.eval()
    cap = cfg["model"]["max_tokens"]
    pv = store.parcel_vid
    if pv < 0:
        return pd.DataFrame(columns=["blklot", "year"])
    sel = np.arange(store.n_seq)
    if years is not None:
        sel = sel[np.isin(store.seq_year, list(years))]
    items = [(s, ch) for s in sel for ch in chunk_tokens(store, s, cap)]
    lens = np.array([len(ch) + 1 for _, ch in items])
    tt = text_table(store, dev)
    keep_set = set(blklots) if blklots is not None else None
    ids, yrs, embs = [], [], []
    for b in make_batches(lens, np.arange(len(items)), cfg["train"]["batch_tokens"], cfg["train"]["max_batch_seqs"]):
        seqs = [items[i][0] for i in b]
        batch = collate(store, seqs, [items[i][1] for i in b])
        tg = batch["tok_global"].numpy()
        ty = batch["type_ids"].numpy()
        batch = to_dev(batch, dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda" and cfg["train"]["bf16"]):
            h = model(batch, text_table=tt).float().cpu().numpy()
        r, c = np.where(ty == pv)
        ent = store.entity[tg[r, c]]
        m = np.array([e is not None and (keep_set is None or e in keep_set) for e in ent], dtype=bool)
        ids.append(ent[m])
        yrs.append(store.seq_year[np.asarray(seqs)[r[m]]])
        embs.append(h[r[m], c[m]])
    if not ids:
        return pd.DataFrame(columns=["blklot", "year"])
    E = np.concatenate(embs)
    df = pd.DataFrame(E, columns=[f"e{i}" for i in range(E.shape[1])])
    df.insert(0, "year", np.concatenate(yrs).astype(np.int16))
    df.insert(0, "blklot", np.concatenate(ids).astype(str))
    # a parcel id can repeat within a cell-year only if A emits duplicates; keep the first
    return df.drop_duplicates(["blklot", "year"]).reset_index(drop=True)


__all__ = ["N_SPECIAL", "cell_embeddings", "parcel_embeddings", "load_checkpoint", "save_checkpoint", "device_of"]
