"""Average the final refit models' forecasts and write the leaderboard submission.

    python -m task2.make_submission   -> results/t2/final/{forecast.csv, submission.txt, manifest.json, forecast.pdf}

P is counted from the rebuilt models and E is read back from each member's log.jsonl; both are
checked against the declared values before anything is written.
"""
import task2.guards  # noqa: F401

import argparse
import hashlib
import json
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from task2.autoformer.model import count_params
from task2.data import load
from task2.guards import (FORECAST_IDX, PRED_LEN, TRAIN_END, assert_epoch_count, assert_param_count,
                          assert_submission_string)
from task2.sweep import SELECTION
from task2.train import RUNS, build_model

OUT = SELECTION.parent / "final"


def main():
    sel = json.loads(SELECTION.read_text())
    names = [f"final_{sel['candidate']}_s{s}" for s in sel["ensemble_seeds"]]
    models, epochs, forecasts = [], [], []
    for n in names:
        meta = json.loads((RUNS / n / "meta.json").read_text())
        m = build_model(argparse.Namespace(**meta))
        m.load_state_dict(torch.load(RUNS / n / "model.pt", map_location="cpu"))
        assert count_params(m) == meta["P"]
        log = [json.loads(line) for line in (RUNS / n / "log.jsonl").read_text().splitlines()]
        assert [e["epoch"] for e in log] == list(range(1, len(log) + 1)) and len(log) == sel["E_star"]
        models.append(m)
        epochs.append(log[-1]["epochs_completed"])
        forecasts.append(np.load(RUNS / n / "forecast.npy"))

    P, E = sum(count_params(m) for m in models), len(names) * sel["E_star"]
    assert_param_count(models, P)
    assert_epoch_count(epochs, E)
    fc = np.mean(forecasts, axis=0)
    assert fc.shape == (PRED_LEN,) and np.isfinite(fc).all() and (fc >= 0).all()

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"time_idx": np.arange(FORECAST_IDX[0], FORECAST_IDX[1] + 1), "value": fc})
    df.to_csv(OUT / "forecast.csv", index=False)
    back = pd.read_csv(OUT / "forecast.csv")
    assert (back.time_idx.values == np.arange(43657, 43825)).all()
    s = ", ".join(f"{v:.6f}" for v in back.value.values)
    assert_submission_string(s, back.value.values)
    (OUT / "submission.txt").write_text(s + "\n")

    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {"git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                "candidate": sel["candidate"], "args": sel["args"], "members": names,
                "P": P, "P_per_member": [count_params(m) for m in models],
                "E": E, "E_per_member": epochs, "E_star": sel["E_star"],
                "forecast_sha256": sha(OUT / "forecast.csv"), "submission_sha256": sha(OUT / "submission.txt"),
                "forecast_mean": float(fc.mean()), "forecast_min": float(fc.min()), "forecast_max": float(fc.max())}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))

    y = load()[0]
    fig, ax = plt.subplots(figsize=(9, 3))
    hist = np.arange(TRAIN_END - 335, TRAIN_END + 1)
    ax.plot(hist, y[hist - 1], lw=0.8, label="observed (last 336 steps)")
    ax.plot(df.time_idx, df.value, lw=0.8, label=f"forecast ({len(names)}-model mean)")
    for f in forecasts:
        ax.plot(df.time_idx, f, lw=0.3, color="grey", alpha=0.5)
    ax.set(xlabel="time_idx", ylabel="value")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "forecast.pdf")


if __name__ == "__main__":
    main()
