"""Pretrain one SF World Model run: L = (1-λ)·L_MTM + λ·L_next.

  python -m models.world_model.pretrain --config configs/wm.yaml --cutoff 2018 --lam 0.25 [--budget sweep|full]

Splits (cutoff C):
  * MTM: train on a fixed 95% of (cell, year) sequences, validate on the other 5%.
  * L_next: train on pairs (t -> t+1) with t+1 < C; validate on (C-1 -> C) pairs (held out).
Writes {ckpt_dir}/cutoff=C/{budget}/lam=λ/{model.pt, metrics.json, cell_emb.parquet, probes_val.csv}.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import is_backtest, load_config, run_dir, seed_everything
from .data import PNEXT, collate, load_store, make_batches, select_tokens
from .infer import cell_embeddings, device_of, save_checkpoint, text_table, to_dev
from .leakage import assert_checkpoint_ok, load_probe_targets
from .model import WorldModel, count_params, sample_mask
from .probes import emb_frame, load_handmade, next_year_probe, run_probes

MTM_VAL_FRAC = 0.05


def split_store(store, seed: int):
    rng = np.random.default_rng(seed + 12345)
    mtm_val = rng.random(store.n_seq) < MTM_VAL_FRAC
    has = store.next_seq >= 0
    nxt_year = store.seq_year.astype(int) + 1
    train_next = has & (nxt_year < store.cutoff) & ~mtm_val
    val_next = has & (nxt_year == store.cutoff)
    return {
        "train": np.where(~mtm_val)[0],
        "train_next": train_next,
        "mtm_val": np.where(mtm_val)[0],
        "next_val": np.where(val_next)[0],
        "val_next_mask": val_next,
    }


def _lr_lambda(warm, total):
    def f(step):
        if step < warm:
            return (step + 1) / warm
        p = min(1.0, (step - warm) / max(1, total - warm))
        return 0.5 * (1 + math.cos(math.pi * p)) * 0.95 + 0.05
    return f


@torch.no_grad()
def evaluate(model, store, cfg, lam, sp, dev, tt) -> dict:
    model.eval()
    cap, tr = cfg["model"]["max_tokens"], cfg["train"]
    lens = np.minimum(store.lengths(), cap - 1) + 1
    gen = torch.Generator().manual_seed(999)  # fixed val masks
    amp = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda" and tr["bf16"])
    mtm, n_m = 0.0, 0
    for b in make_batches(lens, sp["mtm_val"], tr["batch_tokens"], tr["max_batch_seqs"]):
        batch = collate(store, b, [select_tokens(store, s, cap, None) for s in b])
        m = sample_mask(batch["type_ids"], batch["pad"], tr["mask_min"], tr["mask_max"], gen)
        batch = to_dev(batch, dev)
        with torch.autocast(**amp):
            h = model(batch, m.to(dev), tt)
        l, _ = model.mtm_loss(h.float(), batch, m.to(dev))
        mtm += float(l) * len(b)
        n_m += len(b)
    nxt, n_n = 0.0, 0
    pw = float(cfg["parcel_next"]["weight"])
    base = parcel_base_rates(store, sp)
    p_nll, p_base, p_n = np.zeros(3), np.zeros(3), np.zeros(3)
    for b in make_batches(lens, sp["next_val"], tr["batch_tokens"], tr["max_batch_seqs"]):
        batch = to_dev(collate(store, b, [select_tokens(store, s, cap, None) for s in b], sp["val_next_mask"]), dev)
        with torch.autocast(**amp):
            h = model(batch, None, tt)
        h = h.float()
        l, _ = model.next_loss(h, batch, pw)
        nxt += float(l) * len(b)
        n_n += len(b)
        if "pnext" in batch and base is not None:
            v = ~torch.isnan(batch["pnext"])
            rows = v.any(-1)
            if rows.any():
                lr = model.parcel_pred(h[rows]).cpu().numpy()
                y, vv = batch["pnext"][rows].cpu().numpy(), v[rows].cpu().numpy()
                for k in range(3):
                    yk = y[vv[:, k], k]
                    p_nll[k] += float((np.exp(lr[vv[:, k], k]) - yk * lr[vv[:, k], k]).sum())
                    p_base[k] += float((base[k] - yk * np.log(base[k])).sum())
                    p_n[k] += len(yk)
    mtm = mtm / max(n_m, 1) if n_m else float("nan")
    nxt = nxt / max(n_n, 1) if n_n else float("nan")
    tot = (1 - lam) * (0 if lam == 1 else mtm) + lam * (0 if lam == 0 else nxt)
    model.train()
    out = {"val_mtm": mtm, "val_next": nxt, "val_loss": tot}
    for k, name in enumerate(PNEXT):
        if p_n[k]:
            # skill = 1 - NLL(model)/NLL(train-mean rate), on held-out (C-1 -> C) parcel pairs; > 0 beats the base rate
            out[f"val_parcel_skill_{name}"] = 1 - p_nll[k] / p_base[k] if p_base[k] else float("nan")
    return out


def parcel_base_rates(store, sp):
    """Train-split mean t+1 rate per parcel target (floored), for the skill baseline."""
    if store.pnext is None:
        return None
    seq_of = np.repeat(np.arange(store.n_seq), np.diff(store.offsets))
    tr = sp["train_next"][seq_of]
    y = store.pnext[tr]
    return np.maximum(np.nanmean(y, axis=0), 1e-6) if len(y) else None


def train(cfg: dict, store, lam: float, budget: str, dev, log=print):
    tr = cfg["train"]
    seed_everything(tr["seed"])
    if tr.get("num_threads"):
        torch.set_num_threads(int(tr["num_threads"]))
    model = WorldModel(n_types=store.n_types, **cfg["model"]).to(dev)
    n_params = count_params(model)
    assert n_params <= 10_000_000, f"{n_params} params > 10M"
    sp = split_store(store, tr["seed"])
    cap = cfg["model"]["max_tokens"]
    lens = np.minimum(store.lengths(), cap - 1) + 1
    rng = np.random.default_rng(tr["seed"])
    gen = torch.Generator().manual_seed(tr["seed"])
    epochs = int(tr["epochs"][budget])
    steps_per_epoch = len(make_batches(lens, sp["train"], tr["batch_tokens"], tr["max_batch_seqs"]))
    total = max(1, epochs * steps_per_epoch)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"], betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(max(1, int(tr["warmup_frac"] * total)), total))
    tt = text_table(store, dev)
    amp = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda" and tr["bf16"])
    best, best_state, bad, hist, step = float("inf"), None, 0, [], 0
    t0 = time.time()
    deadline = t0 + 60 * float(tr["max_minutes"][budget])
    log(f"params={n_params:,} seqs={store.n_seq} train={len(sp['train'])} mtm_val={len(sp['mtm_val'])} "
        f"next_val={len(sp['next_val'])} steps/epoch={steps_per_epoch} lam={lam}")
    stopped = "epochs"
    for ep in range(epochs):
        model.train()
        run = {"loss": 0.0, "mtm": 0.0, "next": 0.0}
        nb = 0
        for b in make_batches(lens, sp["train"], tr["batch_tokens"], tr["max_batch_seqs"], rng):
            batch = collate(store, b, [select_tokens(store, s, cap, rng) for s in b], sp["train_next"])
            m = sample_mask(batch["type_ids"], batch["pad"], tr["mask_min"], tr["mask_max"], gen)
            batch, m = to_dev(batch, dev), m.to(dev)
            with torch.autocast(**amp):
                h = model(batch, m if lam < 1 else None, tt)
            h = h.float()
            l_mtm, _ = model.mtm_loss(h, batch, m) if lam < 1 else (h.sum() * 0, {})
            # L_next reads the [CELL] output; with λ<1 the input is the masked view (cheap, acts as dropout)
            l_next, _ = model.next_loss(h, batch, float(cfg["parcel_next"]["weight"])) if lam > 0 else (h.sum() * 0, {})
            loss = (1 - lam) * l_mtm + lam * l_next
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
            opt.step()
            sched.step()
            step += 1
            nb += 1
            run["loss"] += loss.item()
            run["mtm"] += l_mtm.item()
            run["next"] += l_next.item()
            if tr["log_every"] and step % tr["log_every"] == 0:
                log(f"  step {step} loss {loss.item():.4f}")
            if time.time() > deadline:
                break
        ev = evaluate(model, store, cfg, lam, sp, dev, tt)
        rec = {"epoch": ep, "step": step, **{f"train_{k}": v / max(nb, 1) for k, v in run.items()}, **ev,
               "minutes": (time.time() - t0) / 60}
        hist.append(rec)
        log(f"ep {ep} train {rec['train_loss']:.4f} val {ev['val_loss']:.4f} (mtm {ev['val_mtm']:.4f} next {ev['val_next']:.4f}) "
            f"{rec['minutes']:.2f}m")
        if ev["val_loss"] < best - 1e-4:
            best, bad = ev["val_loss"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tr["patience"]:
                stopped = "early_stop"
                break
        if time.time() > deadline:
            stopped = "time_budget"
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    info = {"n_params": n_params, "history": hist, "best_val_loss": best, "stopped": stopped,
            "train_minutes": (time.time() - t0) / 60, "steps": step,
            "max_year_seen": int(store.seq_year.max()),
            "max_next_target_year": int(store.seq_year[sp["train"]][sp["train_next"][sp["train"]]].max() + 1)
            if sp["train_next"].any() else int(store.seq_year.min())}
    return model, info


def probe_eval(cfg, store, E, log=print):
    """Selection-time probes: year=select_year rows, fit split=train, tune+report split=val."""
    p = Path(cfg["paths"]["data_root"]) / "probe_targets.parquet"
    if not p.exists():
        log(f"no probe targets at {p}; skipping probes")
        return None, {}
    targets = load_probe_targets(p, cfg)  # locked: no test-year or split=test rows
    hand = load_handmade(cfg, store)
    emb = emb_frame(store.seq_cell, store.seq_year, E)
    pr = cfg["probe"]
    return run_probes(hand, emb, targets, pr["select_year"], pr["alphas"], pr["n_boot"], cfg["train"]["seed"])


def run(cfg: dict, cutoff: int, lam: float, budget: str, device=None, log=print) -> Path:
    dev = device_of(device)
    store = load_store(cfg, cutoff)
    model, info = train(cfg, store, lam, budget, dev, log)
    meta = {"cutoff": cutoff, "lam": lam, "budget": budget, "types": store.types,
            "token_types_seen": store.token_types_seen, **{k: info[k] for k in ("max_year_seen", "max_next_target_year")}}
    assert_checkpoint_ok(meta, cfg)
    out = run_dir(cfg, cutoff, lam, budget)
    out.mkdir(parents=True, exist_ok=True)
    save_checkpoint(out / "model.pt", model, cfg, meta)
    E = cell_embeddings(model, store, cfg, dev)
    emb = emb_frame(store.seq_cell, store.seq_year, E)
    emb.to_parquet(out / "cell_emb.parquet", index=False)
    metrics = {"cutoff": cutoff, "lam": lam, "budget": budget, **{k: v for k, v in info.items() if k != "history"}}
    metrics.update(next_year_probe(store, E, cfg["probe"]["next_ridge_alpha"]))
    metrics["val_next_head"] = min((h["val_next"] for h in info["history"]), default=float("nan"))
    best_ep = min(info["history"], key=lambda h: h["val_loss"]) if info["history"] else {}
    metrics.update({k: v for k, v in best_ep.items() if k.startswith("val_parcel_skill")})
    metrics["parcel_head"] = store.pnext is not None
    if is_backtest(cfg, cutoff):
        table, summ = probe_eval(cfg, store, E, log)
        if table is not None:
            table.insert(0, "lam", lam)
            table.to_csv(out / "probes_val.csv", index=False)
            metrics.update(summ)
            for _, r in table.iterrows():
                metrics[f"{r['target']}__{r['features']}"] = float(r["value"])
    pd.DataFrame(info["history"]).to_csv(out / "history.csv", index=False)
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    log(json.dumps({k: metrics[k] for k in metrics if k.startswith(("mean_lift", "next_probe", "n_params", "train_min"))}, default=float))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wm.yaml")
    ap.add_argument("--cutoff", type=int, required=True)
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--budget", choices=["sweep", "full"], default="full")
    ap.add_argument("--device", default=None)
    ap.add_argument("--data-root", default=None, help="override paths.data_root (e.g. node-local /tmp)")
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--interim", default=None, help="override paths.interim (event tables for the parcel head)")
    a = ap.parse_args(argv)
    over = {"paths": {}}
    if a.data_root:
        over["paths"]["data_root"] = a.data_root
    if a.ckpt_dir:
        over["paths"]["ckpt_dir"] = a.ckpt_dir
    if a.interim:
        over["paths"]["interim"] = a.interim
    cfg = load_config(a.config, over)
    run(cfg, a.cutoff, a.lam, a.budget, a.device)


if __name__ == "__main__":
    main()
