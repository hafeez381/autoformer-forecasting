"""Period recovery from the index alone, on the training split only (time_idx <= 41640).

    python -m task2.eda   -> results/t2/eda.json, results/t2/acf.pdf
"""
import task2.guards  # noqa: F401

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from task2.data import VAL_START, load

OUT = Path(__file__).resolve().parents[1] / "results" / "t2"
MAX_LAG = 800


def acf(x, max_lag):
    x = x - x.mean()
    f = np.fft.rfft(x, n=2 * len(x))
    r = np.fft.irfft(f * np.conj(f))[:max_lag + 1]
    return r / r[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    y = load()[0][:VAL_START - 1].astype(np.float64)
    trend = np.convolve(y, np.ones(169) / 169, mode="same")          # centered 169-step mean
    detr = (y - trend)[84:-84]                                          # drop edge effects
    raw_acf, det_acf = acf(y, MAX_LAG), acf(detr, MAX_LAG)
    # Local maxima of the detrended ACF beyond lag 2, strongest first.
    peaks = [k for k in range(3, MAX_LAG) if det_acf[k] > det_acf[k - 1] and det_acf[k] >= det_acf[k + 1]]
    peaks = sorted(peaks, key=lambda k: -det_acf[k])[:5]
    power = np.abs(np.fft.rfft(detr - detr.mean())) ** 2
    freqs = np.fft.rfftfreq(len(detr))
    top = np.argsort(power[1:])[::-1][:5] + 1
    res = {
        "split": f"time_idx 1..{VAL_START - 1}",
        "acf_raw": {str(k): round(float(raw_acf[k]), 3) for k in (1, 12, 24, 48, 168, 336)},
        "acf_detrended": {str(k): round(float(det_acf[k]), 3) for k in (1, 12, 24, 48, 168, 336)},
        "acf_detrended_top_peaks": {str(k): round(float(det_acf[k]), 3) for k in peaks},
        "periodogram_top_periods": [round(float(1 / freqs[i]), 2) for i in top],
    }
    (OUT / "eda.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].plot(raw_acf, lw=0.8, label="raw")
    ax[0].plot(det_acf, lw=0.8, label="detrended (169-step mean removed)")
    ax[0].set(xlabel="lag (steps)", ylabel="ACF", title="Training-split ACF")
    ax[0].legend(fontsize=7)
    keep = (freqs > 0) & (freqs < 0.1)
    ax[1].semilogy(1 / freqs[keep], power[keep], lw=0.6)
    ax[1].set(xlabel="period (steps)", ylabel="power", title="Periodogram (detrended)", xlim=(2, 400))
    fig.tight_layout()
    fig.savefig(OUT / "acf.pdf")


if __name__ == "__main__":
    main()
