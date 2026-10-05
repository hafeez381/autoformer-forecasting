"""Write the Task 2 report tables and figure from results/t2 (and the naive baseline from the data).

Run from the repo root: python report/make_t2_tables.py
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from task2 import data  # noqa: E402
from task2.train import parse_args  # noqa: E402

RES = ROOT / "results" / "t2"
OUT = ROOT / "report" / "tables"
FIG = ROOT / "report" / "figures"


def num(v, d=2):
    return f"${float(v):.{d}f}$"


def count(v):
    return f"${int(v):,}$".replace(",", "{,}")


def write(name, lines):
    (OUT / name).write_text("\n".join(lines) + "\n")
    print("wrote", OUT / name)


# Period recovery on the training split.
eda = json.loads((RES / "eda.json").read_text())
lags = ["1", "12", "24", "48", "168"]
write("t2_period.tex", [
    r"\begin{tabular}{l" + "c" * len(lags) + "}", r"\toprule",
    r"\textbf{ACF at lag} & " + " & ".join(lags) + r" \\", r"\midrule",
    "Raw & " + " & ".join(num(eda["acf_raw"][k], 3) for k in lags) + r" \\",
    "Detrended & " + " & ".join(num(eda["acf_detrended"][k], 3) for k in lags) + r" \\", r"\midrule",
    r"\multicolumn{" + str(len(lags) + 1) + r"}{l}{Largest periodogram peak (detrended) at a period of "
    + f"{eda['periodogram_top_periods'][0]:.1f}" + r" steps} \\",
    r"\bottomrule", r"\end{tabular}"])
shutil.copy(RES / "acf.pdf", FIG / "t2_acf.pdf")

# Naive baseline on the same 78 validation origins (recomputed from the data, as task2/data.py prints it).
y, _ = data.load()
origins = data.val_origins()
naive = data.metrics(data.targets(y, origins),
                     np.stack([np.full(data.PRED_LEN, y[s - 1 - data.PRED_LEN:s - 1].mean()) for s in origins]))
blocks = np.arange(0, len(origins), 7)


def rows(df, names):
    df = df.set_index("arm")
    out = []
    for arm, label in names:
        r = df.loc[arm]
        out.append(f"{label} & {num(r.rmse_mean)} $\\pm$ {num(r.rmse_sd)} & {num(r.mae_mean)} & "
                   f"{num(r.smape_mean)} & {num(r.bias_mean)} & {num(r.block_rmse_sd)} & {count(r.P)} \\\\")
    return out


HEAD = [r"\begin{tabular}{l c c c c c r}", r"\toprule",
        r"\textbf{Model} & RMSE & MAE & sMAPE (\%) & Bias & Block sd & $\mathcal{P}$ \\", r"\midrule"]

exog = pd.read_csv(RES / "exog_summary.csv")
write("t2_exog.tex", HEAD + [
    f"Naive (last-168 mean) & {num(naive['rmse'].mean())} & {num(naive['mae'].mean())} & "
    f"{num(naive['smape'].mean())} & {num(naive['bias'].mean())} & "
    f"{num(naive['rmse'][blocks].std(ddof=1))} & -- \\\\", r"\midrule",
    *rows(exog, [("exog-none", "No covariates"), ("exog-past", "Past covariates"),
                 ("exog-future", "Future covariates"), ("exog-both", "Past and future")]),
    r"\bottomrule", r"\end{tabular}"])

arms = pd.read_csv(RES / "arms_summary.csv")
write("t2_arms.tex", HEAD + rows(arms, [
    ("exog-future", "Base (future covariates)"), ("exog-future+L168", "$L = 168$"),
    ("exog-future+L720", "$L = 720$"), ("exog-future+k49", "Kernel 49"), ("exog-future+d64", "$d = 64$"),
    ("exog-future+lr1e-3", "Learning rate $10^{-3}$"), ("exog-future+log1p", "log1p target")])
    + [r"\bottomrule", r"\end{tabular}"])

# Final configuration: settings, 5-seed check, ensemble and the declared P and E.
sel = json.loads((RES / "selection.json").read_text())
cand = json.loads((RES / "candidate_summary.json").read_text())
man = json.loads((RES / "final" / "manifest.json").read_text())
a = parse_args(["--mode", "full", *sel["args"]])
settings = [("Covariates", a.exog), ("Input length $L$", a.seq_len), ("Label length", a.label_len),
            ("Moving-average kernel", a.moving_avg), ("Factor $c$", f"{a.factor:g}"),
            ("$d_{\\text{model}}$, heads, $d_{\\text{ff}}$", f"{a.d_model}, {a.n_heads}, {a.d_ff}"),
            ("Encoder, decoder layers", f"{a.e_layers}, {a.d_layers}"), ("Dropout", f"{a.dropout:g}"),
            ("Optimiser", f"Adam, lr ${a.lr:g}$ halved each epoch, batch {a.batch_size}"),
            ("Epochs per member $E^{*}$", sel["E_star"]), ("Ensemble seeds", ", ".join(map(str, sel["ensemble_seeds"])))]
write("t2_settings.tex", [r"\begin{tabular}{ll}", r"\toprule", r"\textbf{Setting} & \textbf{Value} \\", r"\midrule",
                          *[f"{k} & {v} \\\\" for k, v in settings], r"\bottomrule", r"\end{tabular}"])

seeds = cand["per_seed_rmse"]
write("t2_final.tex", [
    r"\begin{tabular}{l c c c c r r}", r"\toprule",
    r"\textbf{Model} & RMSE & MAE & sMAPE (\%) & Block sd & $\mathcal{P}$ & $\mathcal{E}$ \\", r"\midrule",
    *[f"Seed {s} & {num(v)} & & & & {count(cand['P'])} & {sel['E_star']} \\\\" for s, v in seeds.items()],
    f"Seeds 0--4, mean $\\pm$ sd & {num(cand['rmse_mean'])} $\\pm$ {num(cand['rmse_sd'])} & {num(cand['mae_mean'])} & "
    f"{num(cand['smape_mean'])} & {num(cand['block_rmse_sd'])} & & \\\\", r"\midrule",
    f"Ensemble of seeds 0--4 & {num(cand['ensemble_rmse'])} & {num(cand['ensemble_mae'])} & "
    f"{num(cand['ensemble_smape'])} & {num(cand['ensemble_block_rmse_sd'])} & {count(man['P'])} & {man['E']} \\\\",
    r"\bottomrule", r"\end{tabular}"])
