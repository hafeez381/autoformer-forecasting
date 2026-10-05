"""Write the Task 1 report tables from the full-preset result files.

Reads task1/results/design/*.csv (written by the executed notebook) and writes
report/tables/t1_*.tex, so every number in Section 2 traces to a persisted result.
Run from anywhere: python report/make_t1_tables.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "task1" / "results" / "design"
OUT = ROOT / "report" / "tables"
RMSE, MSE = "RMSE (m/s²)", "MSE (m/s²)²"


def num(value, digits=3):
    return "--" if value in ("unavailable", None) or pd.isna(value) else f"${float(value):.{digits}f}$"


def count(value):
    return f"${int(value):,}$".replace(",", "{,}")


def write(name, lines):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text("\n".join(lines) + "\n")
    print("wrote", OUT / name)


part1 = pd.read_csv(RESULTS / "1.1.csv")
designs = pd.read_csv(RESULTS / "4.1.csv")

# Outputs 1.1 and 2.3 by population and product: the four Part 1 forecasters, then
# Raw Attention with decomposition. Raw Attention is the shared row of both outputs.
def product_cells(model, overall, est, held):
    return " & ".join([model, num(overall[f"established stations {RMSE}"]),
                       *(num(est[f"Product {p}"]) for p in "ABC"),
                       num(overall[f"held-out station 4 {RMSE}"]),
                       *(num(held[f"Product {p}"]) for p in "ABC")]) + r" \\"


rows = []
for _, r in part1.iterrows():
    est = {f"Product {p}": r[f"established stations, Product {p} RMSE"] for p in "ABC"}
    held = {f"Product {p}": r[f"held-out station 4, Product {p} RMSE"] for p in "ABC"}
    rows.append(product_cells(r["model"], r, est, held))
ablation = pd.read_csv(RESULTS / "2.3.csv")
for model, g in ablation.groupby("model", sort=False):
    est = g[g.population == "established"].set_index("product")[RMSE]
    held = g[g.population == "held-out"].set_index("product")[RMSE]
    row = product_cells(model, designs.set_index("model").loc[model], est, held)
    if model == "Raw Attention":
        assert row in rows, "Raw Attention differs between Outputs 1.1 and 2.3"
    else:
        rows += [r"\midrule", row]
write("t1_products.tex", [
    r"\begin{tabular}{l cccc cccc}", r"\toprule",
    r"& \multicolumn{4}{c}{\textbf{Established (1--3)}} & \multicolumn{4}{c}{\textbf{Station 4}} \\",
    r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
    r"\textbf{Model} & All & A & B & C & All & A & B & C \\", r"\midrule",
    *rows, r"\bottomrule", r"\end{tabular}"])

# Output 2.2, oracle period recovery by representation.
recovery = pd.read_csv(RESULTS / "2.2.csv")
rows = [f"{r['representation'].replace('width', 'width')}{' (selected)' if r['selected'] else ''} & "
        f"{num(r['oracle recovery within one sample'])} & {num(r['median absolute period error (samples)'])} \\\\"
        for _, r in recovery.iterrows()]
write("t1_recovery.tex", [
    r"\begin{tabular}{lcc}", r"\toprule",
    r"\textbf{Representation} & Recovery & Median error \\", r"\midrule",
    *rows, r"\bottomrule", r"\end{tabular}"])

# Cell 32 summary, all seven fitted models on the validation region.
summary = pd.concat([part1[part1.model != "Raw Attention"], designs], ignore_index=True)
order = ["Shared ridge", "Sensor-specific ridge", "Period-routed ridge", "Raw Attention",
         "Attention + decomposition", "Raw delay mixer", "Autoformer-inspired"]
summary = summary.set_index("model").loc[order]
rows = []
for model, r in summary.iterrows():
    rows.append(" & ".join([model, num(r[f"established stations {RMSE}"]),
                            num(r[f"established stations {MSE}"]),
                            num(r[f"held-out station 4 {RMSE}"]),
                            num(r[f"held-out station 4 {MSE}"]),
                            count(r["parameters"]), num(r["fit seconds"], 3 if r["fit seconds"] < 10 else 1)]) + r" \\")
    if model == "Period-routed ridge":
        rows.append(r"\midrule")
write("t1_designs.tex", [
    r"\begin{tabular}{l cc cc rr}", r"\toprule",
    r"& \multicolumn{2}{c}{\textbf{Established (1--3)}} & \multicolumn{2}{c}{\textbf{Station 4}} & & \\",
    r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
    r"\textbf{Model} & RMSE & MSE & RMSE & MSE & Params & Fit (s) \\", r"\midrule",
    *rows, r"\bottomrule", r"\end{tabular}"])

# Output 4.2, RMSE at selected horizon steps.
horizon = pd.read_csv(RESULTS / "4.2.csv")
steps = (1, 12, 24, 48)
rows = []
for model, g in horizon.groupby("model", sort=False):
    cells = [model]
    for population in ("established", "held-out"):
        h = g[g.population == population].set_index("horizon")[RMSE]
        cells += [num(h[s]) for s in steps]
    rows.append(" & ".join(cells) + r" \\")
write("t1_horizon.tex", [
    r"\begin{tabular}{l cccc cccc}", r"\toprule",
    r"& \multicolumn{4}{c}{\textbf{Established (1--3)}} & \multicolumn{4}{c}{\textbf{Station 4}} \\",
    r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
    r"\textbf{Model} & " + " & ".join(f"$h={s}$" for s in steps) * 1 + " & "
    + " & ".join(f"$h={s}$" for s in steps) + r" \\", r"\midrule",
    *rows, r"\bottomrule", r"\end{tabular}"])


# Output 1.3, the notebook's own table, copied unchanged.
write("t1_attention.tex", [(RESULTS / "1.3.tex").read_text().rstrip("\n")])

# Output 3.2, the notebook's own table, copied unchanged.
write("t1_delays.tex", [(RESULTS / "3.2.tex").read_text().rstrip("\n")])

# Output 1.2, the notebook's own table, copied unchanged.
write("t1_windows.tex", [(RESULTS / "1.2.tex").read_text().rstrip("\n")])
