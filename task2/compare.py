"""Aggregate per-(seed, origin) rows, apply the tie rule, and choose the final configuration.

    python -m task2.compare --stage screen    -> results/t2/screen_rows.csv, screen_summary.csv, selection.json
    python -m task2.compare --stage confirm   -> results/t2/candidate_rows.csv, candidate_summary.csv,
                                                 ensemble_rows.csv; adds E_star to selection.json

Score of one run at one epoch = mean over the 78 validation origins of the per-origin RMSE
(one origin = one 168-step block, like one leaderboard score). Arms are compared at their own
E* = median over seeds of the best epoch. Two arms are tied when their mean RMSE differs by
less than 2 standard errors of the difference across seeds; ties keep the base setting.
"""
import task2.guards  # noqa: F401

import argparse
import json
import math

import numpy as np
import pandas as pd

from task2.data import load, metrics, targets, val_origins
from task2.sweep import ARMS, FINAL_SEEDS, SCREEN_SEEDS, SELECTION
from task2.train import RUNS

OUT = SELECTION.parent
METRICS = ["rmse", "mae", "smape", "bias"]
BLOCKS = np.arange(0, 78, 7)      # the 12 non-overlapping 168-step blocks among the 78 origins


def read_rows(run_names):
    frames = []
    for name in run_names:
        arm = name.rsplit("_s", 1)[0]
        df = pd.read_csv(RUNS / name / "rows.csv")
        df.insert(0, "arm", arm)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summarise(rows):
    """Per arm: E* and mean/sd over seeds at E*, from the per-(seed, epoch, origin) rows."""
    per = rows.groupby(["arm", "seed", "epoch"])[METRICS].mean().reset_index()
    out = []
    for arm, g in per.groupby("arm", sort=False):
        best = g.loc[g.groupby("seed").rmse.idxmin()]
        e_star = int(math.ceil(best.epoch.median()))
        at = g[g.epoch == e_star]
        last = g[g.epoch == g.epoch.max()]
        blk = rows[(rows.arm == arm) & (rows.epoch == e_star)].groupby("seed").rmse.apply(
            lambda r: r.values[BLOCKS].std(ddof=1))
        P = json.loads((RUNS / f"{arm}_s{int(at.seed.iloc[0])}" / "meta.json").read_text())["P"]
        out.append({"arm": arm, "n_seeds": len(at), "E_star": e_star, "best_epochs": best.epoch.tolist(),
                    **{f"{m}_mean": at[m].mean() for m in METRICS}, **{f"{m}_sd": at[m].std(ddof=1) for m in METRICS},
                    "rmse_last_epoch_mean": last.rmse.mean(), "block_rmse_sd": blk.mean(), "P": P})
    return pd.DataFrame(out)


def verdict(a, b):
    """Tie rule: 'better' / 'worse' when |mean_a - mean_b| > 2 SE of the difference, else 'tie'."""
    diff = a.rmse_mean - b.rmse_mean
    se = math.sqrt(a.rmse_sd ** 2 / a.n_seeds + b.rmse_sd ** 2 / b.n_seeds)
    return diff, se, ("better" if diff < -2 * se else "worse" if diff > 2 * se else "tie")


def write_checked(rows, name):
    """Write rows, read them back, and check the summary recomputes from the file."""
    rows.to_csv(OUT / f"{name}_rows.csv", index=False)
    s = summarise(rows)
    back = summarise(pd.read_csv(OUT / f"{name}_rows.csv"))
    assert np.allclose(s[[c for c in s if c.endswith(("_mean", "_sd"))]].values,
                       back[[c for c in back if c.endswith(("_mean", "_sd"))]].values), "summary does not recompute"
    return s


def screen():
    rows = read_rows([f"{arm}_s{s}" for arm in ARMS for s in SCREEN_SEEDS])
    s = write_checked(rows, "screen").set_index("arm")
    base = s.loc["exog-none"]
    s[["diff_vs_none", "se_diff", "verdict"]] = [verdict(s.loc[a], base) for a in s.index]
    s.loc["exog-none", "verdict"] = "base"
    s.to_csv(OUT / "screen_summary.csv")
    print(s[["n_seeds", "E_star", "rmse_mean", "rmse_sd", "mae_mean", "smape_mean", "bias_mean",
             "block_rmse_sd", "P", "diff_vs_none", "se_diff", "verdict"]].round(3).to_string())

    resolved = (s.verdict.isin(["better", "worse"])).sum()
    print(f"seed-noise gate: {resolved}/{len(s) - 1} arms differ from base by > 2 SE -> "
          f"{'PASS' if resolved else 'BREACH (keep the base config)'}")
    exog_better = [a for a in ("exog-past", "exog-future", "exog-both") if s.loc[a, "verdict"] == "better"]
    exog = min(exog_better, key=lambda a: s.loc[a, "rmse_mean"]) if exog_better else "exog-none"
    print(f"exog gate: chosen {exog} ({'beats none by > 2 SE' if exog_better else 'no arm beats none by > 2 SE'}) -> PASS")
    adopted = [a for a in ("L168", "L720", "log1p", "k49", "lr1e-3") if s.loc[a, "verdict"] == "better"]
    if "L168" in adopted and "L720" in adopted:
        adopted.remove(max(("L168", "L720"), key=lambda a: s.loc[a, "rmse_mean"]))
    changes = ([exog] if exog != "exog-none" else []) + adopted
    name = changes[0] if len(changes) == 1 else ("exog-none" if not changes else "cand-" + "+".join(changes))
    args = [x for c in changes for x in ARMS[c]]
    SELECTION.write_text(json.dumps({"candidate": name, "changes_from_base": changes, "args": args}, indent=1))
    print(f"candidate: {name}  args: {args}")


def confirm():
    sel = json.loads(SELECTION.read_text())
    names = [f"{sel['candidate']}_s{s}" for s in FINAL_SEEDS]
    rows = read_rows(names)
    s = write_checked(rows, "candidate").iloc[0]
    e_star = int(s.E_star)
    # Ensemble of all five seeds: average their validation forecasts at epoch E*.
    y, _ = load()
    preds = np.mean([np.load(RUNS / n / "val_preds.npy")[e_star - 1] for n in names], axis=0)
    m = metrics(targets(y, val_origins()), preds)
    ens = pd.DataFrame({"origin": val_origins(), **{k: v for k, v in m.items()}})
    ens.to_csv(OUT / "ensemble_rows.csv", index=False)
    per_seed = rows[rows.epoch == e_star].groupby("seed").rmse.mean()
    seed0_rank = int((per_seed < per_seed[0]).sum()) + 1
    summary = {**{k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in s.items()},
               "per_seed_rmse": per_seed.round(3).to_dict(), "ensemble_rmse": float(m["rmse"].mean()),
               "ensemble_mae": float(m["mae"].mean()), "ensemble_smape": float(m["smape"].mean()),
               "ensemble_block_rmse_sd": float(m["rmse"][BLOCKS].std(ddof=1)), "seed0_rank_of_5": seed0_rank}
    (OUT / "candidate_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    print(f"ensemble vs single-seed mean RMSE: {m['rmse'].mean():.3f} vs {per_seed.mean():.3f} -> "
          f"{'ensemble' if m['rmse'].mean() < per_seed.mean() else 'single seed'} is better")
    sel["E_star"] = e_star
    sel["ensemble_seeds"] = FINAL_SEEDS if m["rmse"].mean() < per_seed.mean() else [0]
    SELECTION.write_text(json.dumps(sel, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["screen", "confirm"], required=True)
    {"screen": screen, "confirm": confirm}[p.parse_args().stage]()
