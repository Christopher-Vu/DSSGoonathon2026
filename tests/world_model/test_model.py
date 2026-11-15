import numpy as np
import pandas as pd
import torch
import yaml

from models.world_model.config import CELL_ID, DEFAULTS, PAD_ID, load_config
from models.world_model.data import build_store, chunk_tokens, collate, load_feature_schema, read_tokens, select_tokens
from models.world_model.fixtures import PUBLIC, SPONSOR
from models.world_model.model import WorldModel, count_params, sample_mask


def test_param_count_under_10m():
    n_types = len(PUBLIC) + len(SPONSOR) + 8  # headroom for new types
    m = WorldModel(n_types=n_types, **DEFAULTS["model"])
    n = count_params(m)
    print("production params:", n)
    assert n <= 10_000_000


def _batch(store, cfg, n=8, rng=None):
    cap = cfg["model"]["max_tokens"]
    ids = np.arange(n)
    return collate(store, ids, [select_tokens(store, s, cap, rng) for s in ids])


def test_forward_backward(cfg, store2018):
    torch.manual_seed(0)
    m = WorldModel(n_types=store2018.n_types, **cfg["model"])
    b = _batch(store2018, cfg)
    mask = sample_mask(b["type_ids"], b["pad"], 0.15, 0.3, torch.Generator().manual_seed(0))
    tt = torch.from_numpy(np.array(store2018.text_emb))
    h = m(b, mask, tt)
    assert h.shape == (8, b["type_ids"].shape[1], cfg["model"]["d_model"])
    l1, _ = m.mtm_loss(h, b, mask)
    l2, _ = m.next_loss(h, b)
    loss = 0.75 * l1 + 0.25 * l2
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads if g is not None)
    assert m.embed.text_proj.weight.grad.abs().sum() > 0
    assert m.next_head[0].weight.grad.abs().sum() > 0


def test_mask_never_masks_cell_or_pad(cfg, store2018):
    b = _batch(store2018, cfg, n=32)
    g = torch.Generator().manual_seed(1)
    for _ in range(50):
        m = sample_mask(b["type_ids"], b["pad"], 0.15, 0.3, g)
        assert not m[:, 0].any()
        assert not (m & (b["type_ids"] == CELL_ID)).any()
        assert not (m & (b["type_ids"] == PAD_ID)).any()
        assert (m.sum(1) >= 1).all()
        real = (~b["pad"]).sum(1) - 1
        rate = m.sum(1).float() / real.float()
        assert rate.mean() < 0.4


def test_cell_token_first(cfg, store2018):
    b = _batch(store2018, cfg)
    assert (b["type_ids"][:, 0] == CELL_ID).all()


def test_subsampling_keeps_nonparcel_and_cap(store2018):
    cap = 16
    pv = store2018.parcel_vid
    for s in range(50):
        sel = select_tokens(store2018, s, cap, np.random.default_rng(s))
        lo, hi = store2018.offsets[s], store2018.offsets[s + 1]
        nonp = np.arange(lo, hi)[store2018.tok_type[lo:hi] != pv]
        assert len(sel) <= cap - 1
        if len(nonp) <= cap - 1:
            assert set(nonp) <= set(sel)
        chunks = chunk_tokens(store2018, s, cap, min_parcel_slots=4)
        par = set(np.arange(lo, hi)[store2018.tok_type[lo:hi] == pv])
        covered = set(np.concatenate(chunks)) & par
        assert covered == par
        assert all(len(c) <= cap - 1 for c in chunks)


def test_next_targets_only_from_t_plus_1(cfg, fixture_root):
    df = read_tokens(cfg, 2018)
    schema = load_feature_schema(cfg["paths"]["token_features"])
    df = df.assign(text_idx=-1)  # text irrelevant here
    base = build_store(df, cfg, 2018, schema, None)
    cell = base.seq_cell[0]
    t = 2014
    s = base.seq_index()[(cell, t)]
    b0 = collate(base, [s], [select_tokens(base, s, 512, None)])
    # ground truth from the raw t+1 rows
    raw = df[(df["cell"] == cell) & (df["year"] == t + 1)]
    want = raw["token_type"].value_counts()
    for i, ty in enumerate(base.types):
        assert b0["next_counts"][0, i].item() == want.get(ty, 0)
    pm = raw[raw["token_type"] == "parcel"]["f2"].mean()
    assert np.isclose(b0["next_means"][0, base.types.index("parcel"), 2].item(), pm, atol=1e-5)
    # perturbing year-t tokens must not change the targets; perturbing t+1 must
    d1 = df.copy()
    sel_t = (d1["cell"] == cell) & (d1["year"] == t)
    d1.loc[sel_t, "f2"] += 100.0
    d1 = pd.concat([d1, d1[sel_t].head(3)])
    s1 = build_store(d1, cfg, 2018, schema, None)
    b1 = collate(s1, [s1.seq_index()[(cell, t)]], [np.arange(1)])
    assert torch.equal(b0["next_counts"], b1["next_counts"])
    assert torch.allclose(b0["next_means"], b1["next_means"], equal_nan=True)
    d2 = df.copy()
    d2.loc[(d2["cell"] == cell) & (d2["year"] == t + 1), "f2"] += 100.0
    s2 = build_store(d2, cfg, 2018, schema, None)
    b2 = collate(s2, [s2.seq_index()[(cell, t)]], [np.arange(1)])
    assert not torch.allclose(b0["next_means"], b2["next_means"], equal_nan=True)


def test_unknown_type_from_yaml_works(cfg, fixture_root, tmp_path):
    df = read_tokens(cfg, 2018).assign(text_idx=-1)
    new = df[df["token_type"] == "permit"].head(200).copy()
    new["token_type"] = "street_tree"
    new["f5"] = 1.0
    df = pd.concat([df, new], ignore_index=True)
    schema = load_feature_schema(cfg["paths"]["token_features"])
    schema["street_tree"] = ["species_code"] * 6
    y = tmp_path / "tf.yaml"
    y.write_text(yaml.safe_dump(schema))
    st = build_store(df, cfg, 2018, load_feature_schema(y), None)
    assert "street_tree" in st.types
    m = WorldModel(n_types=st.n_types, **cfg["model"])
    b = collate(st, np.arange(4), [select_tokens(st, s, 64, None) for s in range(4)])
    h = m(b)
    assert torch.isfinite(h).all()
