"""Train one Autoformer and score it on the 78 validation origins after every epoch.

    python -m task2.train --mode smoke --seed 0 --exog none
    python -m task2.train --mode full --seed 0 --exog future --epochs 10
    python -m task2.train --mode full --final --seed 0 --epochs 6   # train to 43656, forecast 43657..43824

Writes task2/runs/<name>/: meta.json, log.jsonl (one line per epoch), and either rows.csv +
val_preds.npy (validation runs) or forecast.npy + model.pt (final runs). No early stopping:
every epoch is trained and logged, so the epoch count is read back from log.jsonl.
"""
import task2.guards  # noqa: F401  (runs the data guards before anything loads data)

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from task2.autoformer.model import Autoformer, count_params
from task2.data import VAL_START, load, metrics, prepare, targets, train_starts, val_origins
from task2.guards import FORECAST_IDX, PRED_LEN, TRAIN_END, assert_chronological_split

RUNS = Path(__file__).resolve().parent / "runs"


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["smoke", "full"], required=True)
    p.add_argument("--final", action="store_true", help="train on all 43656 rows and forecast")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exog", choices=["none", "past", "future", "both"], default="none")
    p.add_argument("--log1p", action="store_true", help="model log1p(target) instead of the raw target")
    p.add_argument("--seq-len", type=int, default=336)
    p.add_argument("--label-len", type=int, default=168)
    p.add_argument("--moving-avg", type=int, default=25)
    p.add_argument("--factor", type=float, default=1.0)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--e-layers", type=int, default=2)
    p.add_argument("--d-layers", type=int, default=1)
    p.add_argument("--d-ff", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--name", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def build_model(a):
    n_past = 10 if a.exog in ("past", "both") else 0
    n_future = 10 if a.exog in ("future", "both") else 0
    return Autoformer(n_past, n_future, a.label_len, PRED_LEN, a.d_model, a.n_heads, a.e_layers,
                      a.d_layers, a.d_ff, a.moving_avg, a.factor, a.dropout)


def batch_inputs(a, ys, Xs, starts):
    """Encoder input (target, plus past covariates) and decoder covariates for first-target
    time_idx `starts` [B]. ys, Xs are device tensors indexed by position = time_idx - 1."""
    pos = starts[:, None] - 1
    enc_idx = pos + torch.arange(-a.seq_len, 0, device=ys.device)
    x_enc = ys[enc_idx][..., None]
    if a.exog in ("past", "both"):
        x_enc = torch.cat([x_enc, Xs[enc_idx]], dim=-1)
    dec_exog = None
    if a.exog in ("future", "both"):
        dec_exog = Xs[pos + torch.arange(-a.label_len, PRED_LEN, device=ys.device)]
    return x_enc, dec_exog


def predict(model, a, ys, Xs, starts, scaler):
    model.eval()
    with torch.no_grad():
        out = model(*batch_inputs(a, ys, Xs, torch.as_tensor(starts, device=ys.device)))
    return np.clip(scaler.inverse(out[..., 0].double().cpu().numpy()), 0, None)


def main(argv=None):
    a = parse_args(argv)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    name = a.name or (f"{'final' if a.final else 'val'}_{a.mode}_exog-{a.exog}_L{a.seq_len}"
                      f"_k{a.moving_avg}{'_log1p' if a.log1p else ''}_s{a.seed}")
    out = RUNS / name
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)

    y, X = load()
    train_end = TRAIN_END if a.final else VAL_START - 1
    starts = train_starts(a.seq_len, train_end)
    assert_chronological_split(starts[-1] + PRED_LEN - 1, TRAIN_END + 1 if a.final else VAL_START)
    if a.mode == "smoke":
        starts = rng.choice(starts, 256, replace=False)
        a.epochs = min(a.epochs, 2)
    ys_np, Xs_np, scaler = prepare(y, X, train_end, a.exog, a.log1p)
    ys, Xs = torch.as_tensor(ys_np, device=a.device), torch.as_tensor(Xs_np, device=a.device)

    model = build_model(a).to(a.device)
    P = count_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)   # halve lr each epoch
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (out / "meta.json").write_text(json.dumps({**vars(a), "P": P, "git": git, "n_train_windows": len(starts)}, indent=1))
    print(f"{name}: P={P}, {len(starts)} training windows, device {a.device}")

    origins = val_origins()
    y_val = targets(y, origins)
    rows, preds = [], []
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        perm = torch.as_tensor(rng.permutation(starts), device=a.device)
        for i in range(0, len(perm), a.batch_size):
            b = perm[i:i + a.batch_size]
            pred = model(*batch_inputs(a, ys, Xs, b))[..., 0]
            loss = torch.nn.functional.mse_loss(pred, ys[(b[:, None] - 1) + torch.arange(PRED_LEN, device=a.device)])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        sec = time.time() - t0
        entry = {"epoch": epoch, "epochs_completed": epoch, "train_loss": float(np.mean(losses)), "sec": sec,
                 "steps": len(losses), "sec_per_step": sec / len(losses)}
        if a.device == "cuda":
            entry["max_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
        if not a.final:
            p = predict(model, a, ys, Xs, origins, scaler)
            m = metrics(y_val, p)
            preds.append(p.astype(np.float32))
            rows += [{"epoch": epoch, "seed": a.seed, "origin": int(o), **{k: float(v[j]) for k, v in m.items()}}
                     for j, o in enumerate(origins)]
            entry.update({k: float(v.mean()) for k, v in m.items()})
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(json.dumps(entry))

    if a.final:
        fc = predict(model, a, ys, Xs, np.array([FORECAST_IDX[0]]), scaler)[0]
        np.save(out / "forecast.npy", fc)
        torch.save(model.state_dict(), out / "model.pt")
    else:
        pd.DataFrame(rows).to_csv(out / "rows.csv", index=False)
        np.save(out / "val_preds.npy", np.stack(preds))


if __name__ == "__main__":
    main()
