import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.world_model.config import load_config  # noqa: E402


def tiny_cfg(root: Path, **model) -> dict:
    return load_config(ROOT / "configs" / "wm_fixture.yaml", {
        "paths": {"data_root": str(root), "token_features": str(root / "configs" / "token_features.yaml"),
                  "ckpt_dir": str(root / "wm"), "reports": str(root / "reports"),
                  "interim": str(root / "interim")},
        "model": model,
        "train": {"epochs": {"sweep": 8, "full": 8}, "num_threads": 4},
        "probe": {"n_boot": 100},
    })
