import sys

import numpy as np
import pandas as pd
import pytest

from models.world_model import eval_final
from models.world_model.data import build_store, build_vocab, load_feature_schema, read_tokens
from models.world_model.infer import load_checkpoint, save_checkpoint
from models.world_model.leakage import (FinalUnlock, LeakageError, assert_checkpoint_ok, assert_tokens_ok,
                                        issue_final_unlock, load_probe_targets)
from models.world_model.model import WorldModel
from models.world_model.pretrain import split_store


def test_tokens_after_cutoff_rejected(cfg):
    df = read_tokens(cfg, 2018)
    bad = df.head(5).assign(year=np.int16(2019))
    with pytest.raises(LeakageError):
        assert_tokens_ok(pd.concat([df, bad]), cfg, 2018)


def test_sponsor_types_rejected_in_2018(cfg):
    df = read_tokens(cfg, 2018)
    assert not set(df["token_type"]) & {"poi", "reddit", "review", "zillow"}
    for t in ("poi", "reddit", "review", "zillow"):
        bad = df.head(3).assign(token_type=t)
        with pytest.raises(LeakageError):
            assert_tokens_ok(pd.concat([df, bad]), cfg, 2018)
    # the 2025 partition may carry them
    df25 = read_tokens(cfg, 2025)
    assert {"poi", "reddit", "zillow"} <= set(df25["token_type"])


def test_2018_vocab_has_no_sponsor_types_even_if_yaml_does(cfg):
    schema = load_feature_schema(cfg["paths"]["token_features"])
    assert "reddit" in schema
    v18 = build_vocab(schema, [], cfg, 2018)
    v25 = build_vocab(schema, [], cfg, 2025)
    assert not set(v18) & {"poi", "reddit", "review", "zillow"}
    assert {"poi", "reddit"} <= set(v25)


def test_checkpoint_load_asserts(cfg, store2018, tmp_path):
    m = WorldModel(n_types=store2018.n_types, **cfg["model"])
    good = {"cutoff": 2018, "lam": 0.25, "types": store2018.types, "token_types_seen": store2018.types,
            "max_year_seen": 2018, "max_next_target_year": 2017}
    save_checkpoint(tmp_path / "ok.pt", m, cfg, good)
    load_checkpoint(tmp_path / "ok.pt", cfg)
    for bad in ({**good, "types": store2018.types + ["reddit"]}, {**good, "max_year_seen": 2019},
                {**good, "token_types_seen": ["zillow"]}, {**good, "max_next_target_year": 2019}):
        with pytest.raises(LeakageError):
            assert_checkpoint_ok(bad, cfg)


def test_next_pairs_respect_cutoff(store2018):
    sp = split_store(store2018, 0)
    yrs = store2018.seq_year.astype(int)
    assert (yrs[sp["train_next"]] + 1 < 2018).all()  # (2017->2018) never trained on
    assert (yrs[sp["next_val"]] == 2017).all() and len(sp["next_val"]) > 0
    nxt = store2018.next_seq[store2018.next_seq >= 0]
    assert (store2018.seq_year[nxt] <= 2018).all()


def test_probe_targets_locked_without_final(cfg, fixture_root):
    p = fixture_root / "probe_targets.parquet"
    raw = pd.read_parquet(p)
    assert (raw["year"] == 2018).any() and (raw["split"] == "test").any()
    df = load_probe_targets(p, cfg)
    assert not (df["year"] == 2018).any()
    assert not (df["split"] == "test").any()
    assert (df["year"] == 2016).any()


def test_final_unlock_cannot_be_forged(cfg, fixture_root, monkeypatch):
    with pytest.raises(LeakageError):
        FinalUnlock(object())
    with pytest.raises(LeakageError):
        issue_final_unlock(["--final"])  # flag not on the real command line
    monkeypatch.setattr(sys, "argv", ["pytest", "--final"])
    with pytest.raises(LeakageError):
        issue_final_unlock()  # not called from eval_final.main
    with pytest.raises(LeakageError):
        load_probe_targets(fixture_root / "probe_targets.parquet", cfg, unlock=object())


def test_eval_final_requires_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["eval_final"])
    with pytest.raises(SystemExit):
        eval_final.main([])
