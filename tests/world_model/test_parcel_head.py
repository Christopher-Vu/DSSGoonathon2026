"""Parcel next-year head: targets come only from events <= cutoff, follow the L_next year split, and train."""
import numpy as np
import pandas as pd
import torch

from models.world_model.data import collate, load_store, read_events, select_tokens
from models.world_model.model import WorldModel
from models.world_model.pretrain import split_store

from wm_testutil import tiny_cfg


def _write_interim(root, store, cutoff_plus=3):
    """Synthetic event tables for the fixture's parcels, including events AFTER the cutoff (must be ignored)."""
    rng = np.random.default_rng(0)
    blk = sorted({e for e, t in zip(store.entity, store.tok_type) if t == store.parcel_vid and e is not None})
    years = np.arange(2010, store.cutoff + cutoff_plus + 1)
    rows = [(b, y) for b in blk for y in years if rng.random() < 0.1]
    sales = pd.DataFrame(rows, columns=["blklot", "sale_year"])
    bo = pd.DataFrame([(b, y, 1.5) for b, y in rows[::3]], columns=["blklot", "year", "tenants_share"])
    pbf = pd.DataFrame({"blklot": blk, "bf": [f"bf{i % 7}" for i in range(len(blk))]})
    ev = pd.DataFrame([(f"bf{i % 7}", True, y, True) for i, y in enumerate(rng.choice(years, 200))],
                      columns=["bf", "bf_ok", "year", "ellis_omi"])
    root.mkdir(parents=True, exist_ok=True)
    sales.to_parquet(root / "sales.parquet")
    bo.to_parquet(root / "buyout_events.parquet")
    ev.to_parquet(root / "eviction_events.parquet")
    pbf.to_parquet(root / "parcel_blockface.parquet")


def test_parcel_targets_locked_and_trainable(fixture_root, tmp_path, store2018):
    cfg = tiny_cfg(fixture_root)
    cfg["paths"]["interim"] = str(tmp_path / "interim")
    _write_interim(tmp_path / "interim", store2018)
    ev = read_events(cfg, 2018)
    assert all(ev[k].year.max() <= 2018 for k in ("sales", "buyouts", "evictions"))
    st = load_store(cfg, 2018)
    assert st.pnext is not None
    seq_of = np.repeat(np.arange(st.n_seq), np.diff(st.offsets))
    yr = st.seq_year[seq_of]
    par = st.tok_type == st.parcel_vid
    assert np.isnan(st.pnext[~par]).all()
    assert np.isnan(st.pnext[par & (yr == 2018)]).all()            # t+1 = 2019 > cutoff: undefined
    assert np.isnan(st.pnext[par & (yr + 1 < 2015), 1]).all()      # no buyout records before 2015
    assert np.nansum(st.pnext[par & (yr < 2018)]) > 0
    sp = split_store(st, 0)
    b = sp["train"][:16]
    batch = collate(st, b, [select_tokens(st, s, 48, None) for s in b], sp["train_next"])
    # training batches never carry (C-1 -> C) parcel targets
    tg = batch["tok_global"].numpy()
    ok = ~np.isnan(batch["pnext"].numpy()).all(-1)
    assert (st.seq_year[seq_of[tg[ok]]] + 1 < 2018).all()
    m = WorldModel(n_types=st.n_types, **cfg["model"])
    h = m(batch)
    loss, parts = m.next_loss(h, batch, 1.0)
    assert torch.isfinite(loss) and "parcel" in parts
    loss.backward()
    assert m.parcel_head[0].weight.grad.abs().sum() > 0
