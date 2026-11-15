"""FINAL test-set probe evaluation. The only code path that can read year=2018 (test) probe rows.

  python -m models.world_model.eval_final --final

Protocol: frozen cutoff=2018 embeddings of the selected λ (from cell_embeddings.parquet); year=2018
rows; fit on split=train, tune α on split=val, report on split=test (spatial holdout). λ and every
probe setting were fixed before this runs. Run it once, at the end.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .config import load_config
from .data import load_store
from .export import selected_lam
from .leakage import issue_final_unlock, load_probe_targets
from .probes import load_handmade, run_probes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/wm.yaml")
    ap.add_argument("--final", action="store_true", required=True,
                    help="required: acknowledges that this reads the locked 2018 test rows")
    a = ap.parse_args(argv)
    unlock = issue_final_unlock(sys.argv)
    cfg = load_config(a.config)
    pr = cfg["probe"]
    cutoff = int(cfg["backtest_cutoff"])
    lam = selected_lam(cfg)
    root = Path(cfg["paths"]["data_root"])
    emb = pd.read_parquet(root / "cell_embeddings.parquet")
    emb = emb[(emb["cutoff"] == cutoff) & ((emb["lam"] - lam).abs() < 1e-6)].drop(columns=["cutoff", "lam"])
    if emb.empty:
        raise SystemExit(f"no cutoff={cutoff} lam={lam:g} rows in cell_embeddings.parquet; run export first")
    store = load_store(cfg, cutoff)
    hand = load_handmade(cfg, store)
    targets = load_probe_targets(root / "probe_targets.parquet", cfg, unlock=unlock)
    table, summ = run_probes(hand, emb, targets, pr["test_year"], pr["alphas"], pr["n_boot"], cfg["train"]["seed"],
                             fit_split="train", tune_split="val", eval_split="test")
    table.insert(0, "lam", lam)
    rep = Path(cfg["paths"]["reports"])
    rep.mkdir(parents=True, exist_ok=True)
    table.to_csv(rep / "probes_test.csv", index=False)
    with open(rep / "probes_test_summary.json", "w") as f:
        json.dump({"lam": lam, "cutoff": cutoff, "year": pr["test_year"], **summ}, f, indent=2)
    print(table[["target", "metric", "features", "value", "ci_lo", "ci_hi", "lift"]].to_string(index=False))
    print(summ)


if __name__ == "__main__":
    main()
