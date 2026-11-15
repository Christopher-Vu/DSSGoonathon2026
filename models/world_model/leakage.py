"""Leakage locks. Judges check these; tests in tests/world_model/test_leakage.py pin them.

1. Token loading: the cutoff partition may contain only years <= cutoff, and a backtest
   cutoff (<= 2018) may contain no sponsor token types. Checked at read time and on
   checkpoint load.
2. Probe targets: rows with year == test_year (2018, outcomes 2019-2024) and rows with
   split == "test" are filtered out *inside the parquet reader* unless the caller holds a
   FinalUnlock. A FinalUnlock can only be issued by `eval_final.main` when the process was
   started with `--final` on the command line.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from .config import is_backtest, sponsor_types

_KEY = object()
_FINAL_MODULE = "models.world_model.eval_final"


class LeakageError(AssertionError):
    pass


# ---------------------------------------------------------------- tokens / checkpoints
def assert_tokens_ok(df: pd.DataFrame, cfg: dict, cutoff: int) -> None:
    if len(df) == 0:
        raise LeakageError(f"no tokens for cutoff={cutoff}")
    ymax = int(df["year"].max())
    if ymax > cutoff:
        raise LeakageError(f"cutoff={cutoff} tokens contain year {ymax}")
    if is_backtest(cfg, cutoff):
        bad = set(df["token_type"].unique()) & sponsor_types(cfg)
        if bad:
            raise LeakageError(f"backtest cutoff={cutoff} contains sponsor token types {sorted(bad)}")


def assert_vocab_ok(types: list[str], cfg: dict, cutoff: int) -> None:
    if is_backtest(cfg, cutoff):
        bad = set(types) & sponsor_types(cfg)
        if bad:
            raise LeakageError(f"backtest cutoff={cutoff} model vocabulary has sponsor types {sorted(bad)}")


def assert_checkpoint_ok(meta: dict, cfg: dict) -> None:
    cutoff = int(meta["cutoff"])
    assert_vocab_ok(meta["types"], cfg, cutoff)
    if int(meta["max_year_seen"]) > cutoff:
        raise LeakageError(f"checkpoint cutoff={cutoff} trained on year {meta['max_year_seen']}")
    if is_backtest(cfg, cutoff):
        if set(meta.get("token_types_seen", [])) & sponsor_types(cfg):
            raise LeakageError("backtest checkpoint saw sponsor tokens")
        if int(meta.get("max_next_target_year", cutoff)) > cutoff:
            raise LeakageError("backtest checkpoint has an L_next pair with t+1 > cutoff")


# ---------------------------------------------------------------- probe targets
class FinalUnlock:
    """Capability object. Only `issue_final_unlock` can build one."""

    __slots__ = ("_ok",)

    def __init__(self, key):
        if key is not _KEY:
            raise LeakageError("FinalUnlock can only be issued by eval_final --final")
        self._ok = True


def _called_from_eval_final() -> bool:
    for fr in inspect.stack(0)[2:]:
        if fr.frame.f_globals.get("__name__") in (_FINAL_MODULE, "__main__") and fr.function == "main":
            mod_file = fr.frame.f_globals.get("__file__") or ""
            if Path(mod_file).name == "eval_final.py":
                return True
    return False


def issue_final_unlock(argv: list[str] | None = None) -> FinalUnlock:
    argv = sys.argv if argv is None else argv
    if "--final" not in argv or "--final" not in sys.argv:
        raise LeakageError("test probe rows need `python -m models.world_model.eval_final --final`")
    if not _called_from_eval_final():
        raise LeakageError("FinalUnlock may only be issued from eval_final.main")
    return FinalUnlock(_KEY)


def load_probe_targets(path: str | Path, cfg: dict, unlock: FinalUnlock | None = None) -> pd.DataFrame:
    """Read probe_targets.parquet. Without a valid unlock, test-year and split=test rows are
    never read from disk (filter is pushed into the parquet scan)."""
    test_year = int(cfg["probe"]["test_year"])
    final = isinstance(unlock, FinalUnlock) and getattr(unlock, "_ok", False)
    if unlock is not None and not final:
        raise LeakageError("invalid unlock object")
    dataset = ds.dataset(str(path), format="parquet")
    if final:
        if "--final" not in sys.argv:
            raise LeakageError("unlock used outside a --final process")
        filt = ds.field("year") == test_year
    else:
        filt = (ds.field("year") != test_year) & (ds.field("split") != "test")
    df = dataset.to_table(filter=filt).to_pandas()
    df["year"] = df["year"].astype("int16")
    if not final:
        assert not ((df["year"] == test_year) | (df["split"] == "test")).any()
    return df
