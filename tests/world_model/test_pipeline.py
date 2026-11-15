"""End-to-end on the fixture: train, probe (learning sanity check), export schema."""
import json

import numpy as np
import pandas as pd
import pytest

from models.world_model import export, pretrain
from models.world_model.config import run_dir

from wm_testutil import tiny_cfg


@pytest.fixture(scope="module")
def trained(cfg):
    out = pretrain.run(cfg, 2018, 0.25, "sweep", "cpu", log=lambda *_: None)
    with open(out / "metrics.json") as f:
        return out, json.load(f)


def test_embeddings_beat_handmade_on_planted_targets(trained):
    out, m = trained
    probes = pd.read_csv(out / "probes_val.csv")
    piv = probes.pivot(index="target", columns="features", values="value")
    print(piv)
    assert m["n_targets"] >= 3
    assert m["mean_lift_emb"] > 0.05 and m["mean_lift_emb_lo"] > 0
    assert (piv["emb"] > piv["hand"]).sum() >= 3


def test_run_metadata(trained, cfg):
    out, m = trained
    assert m["n_params"] < 10_000_000
    assert m["max_year_seen"] <= 2018 and m["max_next_target_year"] <= 2017
    assert (out / "model.pt").exists()


def test_export_matches_contract(fixture_root, tmp_path):
    # contract needs 256-d embeddings: d_model=256, 1 layer, 1 epoch
    cfg = tiny_cfg(fixture_root, d_model=256, n_layers=1, n_heads=8, d_ff=256)
    cfg["paths"]["ckpt_dir"] = str(tmp_path / "wm")
    cfg["paths"]["reports"] = str(tmp_path / "reports")
    cfg["train"]["epochs"] = {"sweep": 1, "full": 1}
    cfg["export"]["parcel_years"] = {2018: [2018], 2025: [2025]}
    for c in (2018, 2025):
        pretrain.run(cfg, c, 0.5, "full", "cpu", log=lambda *_: None)
    cells, parcels = [], []
    for c in (2018, 2025):
        cd, pdf = export.export_cutoff(cfg, c, 0.5, export.device_of("cpu"))
        cells.append(cd)
        parcels.append(pdf)
    cells, parcels = pd.concat(cells), pd.concat(parcels)
    ecols = [f"e{i}" for i in range(256)]
    for df, key in ((cells, "cell"), (parcels, "blklot")):
        assert list(df.columns) == [key, "year", "cutoff", "lam", *ecols]
        assert df[key].map(type).eq(str).all()
        assert df["year"].dtype == np.int16 and df["cutoff"].dtype == np.int16
        assert df["lam"].dtype == np.float32
        assert all(df[c].dtype == np.float32 for c in ecols)
        assert set(df["cutoff"]) == {2018, 2025}
        assert not df[ecols].isna().any().any()
        assert not df.duplicated([key, "year", "cutoff"]).any()
    assert cells[cells["cutoff"] == 2018]["year"].max() == 2018
    # every 2018 parcel token got an embedding
    from models.world_model.data import read_tokens
    tok = read_tokens(cfg, 2018)
    want = set(tok[(tok["token_type"] == "parcel") & (tok["year"] == 2018)]["entity_id"])
    assert set(parcels[parcels["cutoff"] == 2018]["blklot"]) == want
    assert (tmp_path / "reports" / "umap_rgb_map.png").exists()
    assert run_dir(cfg, 2018, 0.5, "full").exists()
