"""Token store: reads tokens/cutoff=C (CONTRACTS §1), packs one sequence per (cell, year),
builds next-year summary targets, subsampling and batching."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
import yaml

from .config import CELL_ID, N_FEAT, N_SPECIAL, PAD_ID, PARCEL, TEXT_DIM, is_backtest, sponsor_types
from .leakage import assert_tokens_ok, assert_vocab_ok

FEAT_COLS = [f"f{i}" for i in range(N_FEAT)]


def load_feature_schema(path: str | Path) -> dict[str, list[str]]:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    schema = raw.get("types", raw)  # accept {type: [names]} or {types: {type: [names]}}
    out = {}
    for t, names in schema.items():
        names = list(names or [])
        if len(names) > N_FEAT:
            raise ValueError(f"type {t} has {len(names)} features > {N_FEAT}")
        out[str(t)] = names
    return out


def build_vocab(schema: dict, data_types, cfg: dict, cutoff: int) -> list[str]:
    """Real token types in a stable order: yaml order, then any extra types seen in data.
    Backtest cutoffs drop sponsor types entirely."""
    types = list(schema.keys())
    types += sorted(set(map(str, data_types)) - set(types))
    if is_backtest(cfg, cutoff):
        types = [t for t in types if t not in sponsor_types(cfg)]
    assert_vocab_ok(types, cfg, cutoff)
    return types


def tokens_dir(cfg: dict, cutoff: int) -> Path:
    # paths.tokens_from = T reads the cutoff=T partition and keeps years <= cutoff (rolling-origin models
    # for out-of-time features; only standardization constants differ from a native cutoff partition)
    src = int(cfg["paths"].get("tokens_from") or cutoff)
    assert src >= cutoff, f"tokens_from={src} < cutoff={cutoff}"
    return Path(cfg["paths"]["data_root"]) / "tokens" / f"cutoff={src}"


def read_tokens(cfg: dict, cutoff: int) -> pd.DataFrame:
    d = tokens_dir(cfg, cutoff)
    dataset = ds.dataset(str(d), format="parquet")
    have = set(dataset.schema.names)
    cols = [c for c in ["cell", "year", "token_type", "entity_id", *FEAT_COLS, "x_off", "y_off", "text_idx"] if c in have]
    df = dataset.to_table(columns=cols, filter=ds.field("year") <= cutoff).to_pandas()
    for c in FEAT_COLS:
        if c not in df:
            df[c] = np.float32(np.nan)
    for c, v in (("entity_id", None), ("x_off", 0.0), ("y_off", 0.0), ("text_idx", -1)):
        if c not in df:
            df[c] = v
    df["year"] = df["year"].astype("int16")
    df["token_type"] = df["token_type"].astype(str)
    assert_tokens_ok(df, cfg, cutoff)
    return df


@dataclass
class TokenStore:
    cutoff: int
    types: list[str]
    seq_cell: np.ndarray  # [S] str
    seq_year: np.ndarray  # [S] int16
    offsets: np.ndarray  # [S+1] int64
    tok_type: np.ndarray  # [N] int64 vocab id (>= N_SPECIAL)
    feats: np.ndarray  # [N,16] float32, NaN = missing
    text_idx: np.ndarray  # [N] int32
    xy: np.ndarray  # [N,2] float32
    entity: np.ndarray  # [N] object
    counts: np.ndarray  # [S,T] float32, year-t token counts per type (unsubsampled)
    means: np.ndarray  # [S,T,16] float32, per-type feature means, NaN where undefined
    next_seq: np.ndarray  # [S] int64, index of (cell, year+1) or -1
    text_emb: np.ndarray | None = None
    pnext: np.ndarray | None = None  # [N,3] float32 parcel t+1 targets (sale, buyout, bf Ellis/OMI), NaN = n/a
    token_types_seen: list[str] = field(default_factory=list)

    @property
    def n_seq(self) -> int:
        return len(self.seq_year)

    @property
    def n_types(self) -> int:
        return len(self.types)

    @property
    def parcel_vid(self) -> int:
        return N_SPECIAL + self.types.index(PARCEL) if PARCEL in self.types else -1

    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    def seq_index(self) -> dict:
        return {(c, int(y)): i for i, (c, y) in enumerate(zip(self.seq_cell, self.seq_year))}


def summarize(seq_of_tok: np.ndarray, type_idx: np.ndarray, feats: np.ndarray, n_seq: int, n_types: int):
    """Per-(sequence, type) token counts and feature means. Used both for L_next targets (of the
    t+1 sequence) and as hand-made probe features (of the t sequence)."""
    key = seq_of_tok.astype(np.int64) * n_types + type_idx
    m = n_seq * n_types
    counts = np.bincount(key, minlength=m).astype(np.float32).reshape(n_seq, n_types)
    means = np.full((n_seq, n_types, N_FEAT), np.nan, dtype=np.float32)
    for j in range(N_FEAT):
        v = ~np.isnan(feats[:, j])
        if not v.any():
            continue
        s = np.bincount(key[v], weights=feats[v, j], minlength=m)
        n = np.bincount(key[v], minlength=m)
        with np.errstate(invalid="ignore", divide="ignore"):
            means[:, :, j] = np.where(n > 0, s / np.maximum(n, 1), np.nan).reshape(n_seq, n_types)
    return counts, means


def build_store(df: pd.DataFrame, cfg: dict, cutoff: int, schema: dict, text_emb: np.ndarray | None) -> TokenStore:
    assert_tokens_ok(df, cfg, cutoff)
    types = build_vocab(schema, df["token_type"].unique(), cfg, cutoff)
    df = df[df["token_type"].isin(types)]
    df = df.sort_values(["cell", "year"], kind="stable").reset_index(drop=True)
    grp = df.groupby(["cell", "year"], sort=False, observed=True)
    seq_of_tok = grp.ngroup().to_numpy()
    first = df.drop_duplicates(["cell", "year"])
    seq_cell = first["cell"].to_numpy(dtype=object)
    seq_year = first["year"].to_numpy(dtype=np.int16)
    sizes = np.bincount(seq_of_tok, minlength=len(seq_year))
    offsets = np.r_[0, np.cumsum(sizes)].astype(np.int64)

    tmap = {t: i for i, t in enumerate(types)}
    type_idx = df["token_type"].map(tmap).to_numpy(dtype=np.int64)
    feats = df[FEAT_COLS].to_numpy(dtype=np.float32)
    counts, means = summarize(seq_of_tok, type_idx, feats, len(seq_year), len(types))

    idx = {(c, int(y)): i for i, (c, y) in enumerate(zip(seq_cell, seq_year))}
    next_seq = np.array([idx.get((c, int(y) + 1), -1) for c, y in zip(seq_cell, seq_year)], dtype=np.int64)
    # pairs need both years <= cutoff; guaranteed by the token check, asserted again here
    assert all(int(seq_year[n]) <= cutoff for n in next_seq if n >= 0)

    text_idx = df["text_idx"].fillna(-1).to_numpy(dtype=np.int32)
    if text_emb is not None and (text_idx >= 0).any():
        assert text_idx.max() < len(text_emb), "text_idx out of range of text_emb"
        assert text_emb.shape[1] == TEXT_DIM
    elif (text_idx >= 0).any():
        raise FileNotFoundError("tokens reference text_idx but no text_emb file was found")

    return TokenStore(
        cutoff=cutoff,
        types=types,
        seq_cell=seq_cell,
        seq_year=seq_year,
        offsets=offsets,
        tok_type=type_idx + N_SPECIAL,
        feats=feats,
        text_idx=text_idx,
        xy=np.clip(df[["x_off", "y_off"]].fillna(0).to_numpy(dtype=np.float32), -1, 1),
        entity=df["entity_id"].to_numpy(dtype=object),
        counts=counts,
        means=means,
        next_seq=next_seq,
        text_emb=text_emb,
        token_types_seen=sorted(df["token_type"].unique()),
    )


PNEXT = ("sale", "buyout", "bf_ellis_omi")


def read_events(cfg: dict, cutoff: int) -> dict | None:
    """Event tables for the parcel head, cut at year <= cutoff (asserted). None if absent."""
    root = Path(cfg["paths"].get("interim") or "data/interim")
    need = ["sales", "buyout_events", "eviction_events", "parcel_blockface"]
    if not all((root / f"{n}.parquet").exists() for n in need):
        return None
    sa = pd.read_parquet(root / "sales.parquet", columns=["blklot", "sale_year"]).rename(columns={"sale_year": "year"})
    bo = pd.read_parquet(root / "buyout_events.parquet", columns=["blklot", "year", "tenants_share"])
    ev = pd.read_parquet(root / "eviction_events.parquet", columns=["bf", "bf_ok", "year", "ellis_omi"])
    ev = ev[ev.bf_ok.fillna(False).astype(bool) & ev.ellis_omi.fillna(False).astype(bool)][["bf", "year"]]
    out = {}
    for k, d in (("sales", sa), ("buyouts", bo), ("evictions", ev)):
        d = d.dropna(subset=["year"])
        d = d[d.year <= cutoff].copy()
        d["year"] = d.year.astype(int)
        assert len(d) == 0 or d.year.max() <= cutoff
        out[k] = d
    out["pbf"] = pd.read_parquet(root / "parcel_blockface.parquet", columns=["blklot", "bf"]).drop_duplicates("blklot")
    return out


def parcel_next_targets(store: TokenStore, ev: dict, cfg: dict) -> np.ndarray:
    """[N,3] per-token targets for parcel tokens at year t: counts in year t+1 of (sales of the parcel,
    buyout tenants at the parcel, Ellis/OMI notices on its block-face). NaN for non-parcel tokens,
    t+1 > cutoff, buyouts before first_buyout_year, and parcels without a block-face."""
    C, fby = store.cutoff, int(cfg["parcel_next"]["first_buyout_year"])
    N = len(store.tok_type)
    out = np.full((N, 3), np.nan, dtype=np.float32)
    seq_of = np.repeat(np.arange(store.n_seq), np.diff(store.offsets))
    yr = store.seq_year[seq_of].astype(int)
    ip = np.where((store.tok_type == store.parcel_vid) & (yr + 1 <= C))[0]
    if len(ip) == 0:
        return out
    t = pd.DataFrame({"blklot": store.entity[ip].astype(str), "year": yr[ip] + 1})
    t = t.merge(ev["pbf"].astype({"blklot": str}), on="blklot", how="left")
    sale = ev["sales"].astype({"blklot": str}).groupby(["blklot", "year"]).size().rename("v")
    bo = ev["buyouts"].astype({"blklot": str}).groupby(["blklot", "year"]).tenants_share.sum().rename("v")
    bfn = ev["evictions"].groupby(["bf", "year"]).size().rename("v")
    out[ip, 0] = t.join(sale, on=["blklot", "year"]).v.fillna(0).to_numpy(np.float32)
    b = t.join(bo, on=["blklot", "year"]).v.fillna(0).to_numpy(np.float32)
    out[ip, 1] = np.where(t.year.to_numpy() >= fby, b, np.nan)
    f = t.join(bfn, on=["bf", "year"]).v.fillna(0).to_numpy(np.float32)
    out[ip, 2] = np.where(t.bf.notna().to_numpy(), f, np.nan)
    return out


def load_store(cfg: dict, cutoff: int) -> TokenStore:
    schema = load_feature_schema(cfg["paths"]["token_features"])
    df = read_tokens(cfg, cutoff)
    p = Path(cfg["paths"]["data_root"]) / f"text_emb_{cutoff}.npy"
    text_emb = np.load(p, mmap_mode="r") if p.exists() else None
    store = build_store(df, cfg, cutoff, schema, text_emb)
    ev = read_events(cfg, cutoff)
    if ev is not None and store.parcel_vid >= 0:
        store.pnext = parcel_next_targets(store, ev, cfg)
    return store


# ---------------------------------------------------------------- subsampling
def _stratified(idx: np.ndarray, types: np.ndarray, k: int, rng: np.random.Generator | None) -> np.ndarray:
    """Pick k of idx keeping per-type proportions (largest remainder)."""
    if len(idx) <= k:
        return idx
    uniq, inv = np.unique(types, return_inverse=True)
    n = np.bincount(inv)
    quota = n * k / n.sum()
    take = np.floor(quota).astype(int)
    rem = k - take.sum()
    take[np.argsort(-(quota - take), kind="stable")[:rem]] += 1
    out = []
    for g in range(len(uniq)):
        members = idx[inv == g]
        if rng is None:
            sel = members[np.linspace(0, len(members) - 1, take[g]).round().astype(int)] if take[g] else members[:0]
        else:
            sel = rng.choice(members, size=take[g], replace=False)
        out.append(sel)
    return np.sort(np.concatenate(out))


def select_tokens(store: TokenStore, s: int, cap: int, rng: np.random.Generator | None) -> np.ndarray:
    """Global token indices for sequence s, at most cap-1 (room for [CELL]). Every non-parcel
    token is kept first; parcels fill the remaining budget (random, or evenly spaced if rng is None)."""
    lo, hi = store.offsets[s], store.offsets[s + 1]
    idx = np.arange(lo, hi)
    budget = cap - 1
    if len(idx) <= budget:
        return idx
    tt = store.tok_type[idx]
    is_p = tt == store.parcel_vid
    nonp, par = idx[~is_p], idx[is_p]
    if len(nonp) >= budget:
        return _stratified(nonp, tt[~is_p], budget, rng)
    k = budget - len(nonp)
    if rng is None:
        sel = par[np.linspace(0, len(par) - 1, k).round().astype(int)]
    else:
        sel = rng.choice(par, size=k, replace=False)
    return np.sort(np.concatenate([nonp, sel]))


def chunk_tokens(store: TokenStore, s: int, cap: int, min_parcel_slots: int = 128) -> list[np.ndarray]:
    """Deterministic chunks covering every parcel token of sequence s (for parcel export)."""
    lo, hi = store.offsets[s], store.offsets[s + 1]
    idx = np.arange(lo, hi)
    budget = cap - 1
    if len(idx) <= budget:
        return [idx]
    tt = store.tok_type[idx]
    is_p = tt == store.parcel_vid
    nonp, par = idx[~is_p], idx[is_p]
    if len(par) == 0:
        return [_stratified(nonp, tt[~is_p], budget, None)]
    keep_nonp = _stratified(nonp, tt[~is_p], max(budget - min_parcel_slots, 0), None) if len(nonp) > budget - min_parcel_slots else nonp
    room = budget - len(keep_nonp)
    n_chunks = int(np.ceil(len(par) / room))
    parts = np.array_split(par, n_chunks)  # contiguous = parcels in entity order
    return [np.sort(np.concatenate([keep_nonp, p])) for p in parts]


# ---------------------------------------------------------------- batching
def make_batches(lengths: np.ndarray, seq_ids: np.ndarray, batch_tokens: int, max_seqs: int, rng=None) -> list[np.ndarray]:
    seq_ids = np.asarray(seq_ids)
    if len(seq_ids) == 0:
        return []
    L = lengths[seq_ids].astype(float)
    if rng is not None:
        L = L * rng.uniform(0.9, 1.1, size=len(L))  # bucket jitter
    order = seq_ids[np.argsort(L, kind="stable")]
    batches, cur, cur_max = [], [], 0
    for s in order:
        ln = int(lengths[s])
        new_max = max(cur_max, ln)
        if cur and ((len(cur) + 1) * new_max > batch_tokens or len(cur) >= max_seqs):
            batches.append(np.array(cur))
            cur, new_max = [], ln
        cur.append(s)
        cur_max = new_max
    if cur:
        batches.append(np.array(cur))
    if rng is not None:
        rng.shuffle(batches)
    return batches


def collate(store: TokenStore, seq_ids, tok_lists: list[np.ndarray], next_ok: np.ndarray | None = None) -> dict:
    """Pad a batch. Position 0 of every row is [CELL]. Tensors are CPU torch."""
    B = len(seq_ids)
    L = max(len(t) for t in tok_lists) + 1
    T = store.n_types
    type_ids = np.full((B, L), PAD_ID, dtype=np.int64)
    feats = np.full((B, L, N_FEAT), np.nan, dtype=np.float32)
    text_idx = np.full((B, L), -1, dtype=np.int64)
    xy = np.zeros((B, L, 2), dtype=np.float32)
    tok_global = np.full((B, L), -1, dtype=np.int64)
    type_ids[:, 0] = CELL_ID
    for b, t in enumerate(tok_lists):
        n = len(t)
        type_ids[b, 1 : n + 1] = store.tok_type[t]
        feats[b, 1 : n + 1] = store.feats[t]
        text_idx[b, 1 : n + 1] = store.text_idx[t]
        xy[b, 1 : n + 1] = store.xy[t]
        tok_global[b, 1 : n + 1] = t
    seq_ids = np.asarray(seq_ids)
    nxt = store.next_seq[seq_ids]
    has_next = nxt >= 0
    if next_ok is not None:
        has_next &= next_ok[seq_ids]
    safe = np.where(has_next, nxt, 0)
    out = {
        "seq": torch.from_numpy(seq_ids.astype(np.int64)),
        "type_ids": torch.from_numpy(type_ids),
        "feats": torch.from_numpy(feats),
        "text_idx": torch.from_numpy(text_idx),
        "xy": torch.from_numpy(xy),
        "year": torch.from_numpy(store.seq_year[seq_ids].astype(np.int64)),
        "pad": torch.from_numpy(type_ids == PAD_ID),
        "tok_global": torch.from_numpy(tok_global),
        "has_next": torch.from_numpy(has_next),
        # L_next targets: built ONLY from the (cell, t+1) sequence's own tokens
        "next_counts": torch.from_numpy(store.counts[safe].reshape(B, T)),
        "next_means": torch.from_numpy(store.means[safe].reshape(B, T, N_FEAT)),
    }
    if store.pnext is not None:
        # parcel head targets share the (cell, t+1) gate, so they follow the same train/val year split
        pn = np.where((tok_global >= 0)[..., None], store.pnext[np.maximum(tok_global, 0)], np.nan).astype(np.float32)
        pn[~has_next] = np.nan
        out["pnext"] = torch.from_numpy(pn)
    return out
