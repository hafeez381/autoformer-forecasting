"""Run training arms as parallel processes on one machine.

    python -m task2.sweep --phase screen  --mode full --jobs 6   # exog ablation + ranking arms, seeds 0-2
    python -m task2.sweep --phase confirm --mode full --jobs 6   # selected candidate, seeds 0-4 (missing ones)
    python -m task2.sweep --phase final   --mode full --jobs 6   # refit seeds 0-4 on all rows for E* epochs

Arms differ from the base config (L2=336, kernel 25, lr 1e-4, raw target, no exog) in exactly one setting.
"""
import task2.guards  # noqa: F401

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from task2.train import RUNS, parse_args

SELECTION = Path(__file__).resolve().parents[1] / "results" / "t2" / "selection.json"
SCREEN_SEEDS, FINAL_SEEDS = [0, 1, 2], [0, 1, 2, 3, 4]
ARMS = {
    # External-data ablation: where the ten covariates enter.
    "exog-none": [], "exog-past": ["--exog", "past"], "exog-future": ["--exog", "future"],
    "exog-both": ["--exog", "both"],
    # Ranking arms, each against exog-none.
    "L168": ["--seq-len", "168"], "L720": ["--seq-len", "720"], "log1p": ["--log1p"],
    "k49": ["--moving-avg", "49"], "lr1e-3": ["--lr", "1e-3"],
}


def check_one_change(arm, args):
    base, arm_cfg = vars(parse_args(["--mode", "full"])), vars(parse_args(["--mode", "full"] + args))
    changed = [k for k in base if base[k] != arm_cfg[k]]
    assert len(changed) <= 1 and (arm == "exog-none") == (not changed), f"{arm} changes {changed}"


def specs(phase):
    if phase == "screen":
        for arm, args in ARMS.items():
            check_one_change(arm, args)
        return [(f"{arm}_s{s}", args + ["--seed", str(s)]) for arm, args in ARMS.items() for s in SCREEN_SEEDS]
    sel = json.loads(SELECTION.read_text())
    if phase == "confirm":
        return [(f"{sel['candidate']}_s{s}", sel["args"] + ["--seed", str(s)]) for s in FINAL_SEEDS
                if not (RUNS / f"{sel['candidate']}_s{s}" / "rows.csv").exists()]
    return [(f"final_{sel['candidate']}_s{s}", sel["args"] + ["--final", "--epochs", str(sel["E_star"]),
                                                              "--seed", str(s)]) for s in sel["ensemble_seeds"]]


def run(spec, mode):
    name, args = spec
    log = RUNS / f"{name}.out"
    RUNS.mkdir(exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "2"}
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run([sys.executable, "-m", "task2.train", "--mode", mode, "--name", name] + args,
                            stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    print(f"{'DONE' if rc == 0 else 'FAILED'} {name} rc={rc} {time.time() - t0:.0f}s", flush=True)
    return rc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["screen", "confirm", "final"], required=True)
    p.add_argument("--mode", choices=["smoke", "full"], required=True)
    p.add_argument("--jobs", type=int, default=4)
    a = p.parse_args()
    todo = specs(a.phase)
    print(f"{a.phase}: {len(todo)} runs, {a.jobs} at a time", flush=True)
    with ThreadPoolExecutor(a.jobs) as ex:
        rcs = list(ex.map(lambda s: run(s, a.mode), todo))
    print(f"ALL FINISHED {a.phase}: {rcs.count(0)}/{len(rcs)} ok", flush=True)
    sys.exit(0 if all(rc == 0 for rc in rcs) else 1)


if __name__ == "__main__":
    main()
