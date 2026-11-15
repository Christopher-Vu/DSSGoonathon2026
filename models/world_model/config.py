"""Config loading and shared constants for the SF World Model."""
from __future__ import annotations

import copy
import os
import random
from pathlib import Path

import numpy as np
import yaml

N_FEAT = 16
TEXT_DIM = 384
SPECIALS = ["[PAD]", "[CELL]", "[MASK]"]
PAD_ID, CELL_ID, MASK_ID = 0, 1, 2
N_SPECIAL = len(SPECIALS)
PARCEL = "parcel"
DEFAULT_SPONSOR_TYPES = ("poi", "reddit", "review", "zillow")

DEFAULTS: dict = {
    "paths": {
        "data_root": "data/processed",
        "token_features": "configs/token_features.yaml",
        "probe_features": None,  # optional hand-made probe-feature table (cell, year, feature cols)
        "ckpt_dir": "data/processed/wm",
        "reports": "reports/wm",
    },
    "backtest_cutoff": 2018,  # cutoffs <= this may not contain sponsor types
    "sponsor_types": list(DEFAULT_SPONSOR_TYPES),
    "model": {
        "d_model": 256,
        "n_layers": 6,
        "n_heads": 8,
        "d_ff": 1024,
        "dropout": 0.1,
        "n_fourier": 8,
        "year_base": 2000,
        "n_years": 40,
        "max_tokens": 512,  # including [CELL]
    },
    "train": {
        "seed": 0,
        "lr": 3.0e-4,
        "weight_decay": 0.05,
        "warmup_frac": 0.05,
        "epochs": {"sweep": 12, "full": 25},
        "max_minutes": {"sweep": 9.0, "full": 9.0},
        "batch_tokens": 16384,
        "max_batch_seqs": 64,
        "mask_min": 0.15,
        "mask_max": 0.30,
        "patience": 3,
        "grad_clip": 1.0,
        "num_threads": 0,  # 0 = torch default
        "bf16": True,
        "log_every": 50,
    },
    "probe": {
        "select_year": 2016,
        "test_year": 2018,
        "alphas": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        "n_boot": 1000,
        "next_ridge_alpha": 10.0,
    },
    "export": {
        # parcel embeddings are big (200k parcels x 256 floats per year); export only these years
        "parcel_years": {2018: [2014, 2015, 2016, 2017, 2018], 2025: [2025]},
        "parcel_filter": None,  # optional parquet with a `blklot` column
        "umap_year": {2018: 2018, 2025: 2025},
    },
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path) as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = _merge(cfg, overrides)
    # yaml keys come in as ints already; normalise export dict keys to int
    for k in ("parcel_years", "umap_year"):
        cfg["export"][k] = {int(c): v for c, v in cfg["export"][k].items()}
    return cfg


def sponsor_types(cfg: dict) -> set[str]:
    return set(DEFAULT_SPONSOR_TYPES) | set(cfg.get("sponsor_types") or [])


def is_backtest(cfg: dict, cutoff: int) -> bool:
    return int(cutoff) <= int(cfg["backtest_cutoff"])


def run_dir(cfg: dict, cutoff: int, lam: float, budget: str) -> Path:
    return Path(cfg["paths"]["ckpt_dir"]) / f"cutoff={cutoff}" / budget / f"lam={lam:g}"


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
