import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm_testutil import tiny_cfg  # noqa: E402  (also puts the repo root on sys.path)

from models.world_model.fixtures import write_fixture  # noqa: E402


@pytest.fixture(scope="session")
def fixture_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("wm_fixture")
    write_fixture(root, k=10, seed=0)
    return root


@pytest.fixture(scope="session")
def cfg(fixture_root):
    return tiny_cfg(fixture_root)


@pytest.fixture(scope="session")
def store2018(cfg):
    from models.world_model.data import load_store

    return load_store(cfg, 2018)
