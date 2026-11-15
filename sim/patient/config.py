"""Config loading for the patient sim."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = (Path.cwd() / path) if (Path.cwd() / path).exists() else REPO / path
    cfg = yaml.safe_load(path.read_text())
    if "inherit" in cfg:
        parent = load_config(path.parent / cfg.pop("inherit"))
        cfg = _merge(parent, cfg)
    cfg["_config_path"] = str(path)
    return cfg


def data_root(cfg: dict) -> Path:
    root = Path(os.environ.get("SIM_DATA_ROOT") or cfg.get("data_root", "data"))
    return root if root.is_absolute() else REPO / root


def p(cfg: dict, key: str) -> Path:
    """Resolve cfg['paths'][key]. `reports` is repo-relative, everything else data-relative."""
    raw = Path(cfg["paths"][key])
    if raw.is_absolute():
        return raw
    return (REPO / raw) if key == "reports" else data_root(cfg) / raw
