"""Loading, scaling, window construction, validation origins and metrics.

Positions: array position p holds time_idx p + 1. A window "starting at s" means its first
forecast target is time_idx s; its encoder input is the seq_len steps before s.
"""
import numpy as np
import pandas as pd

from task2.guards import (DATA_DIR, EXOG_COLUMNS, PRED_LEN, TRAIN_END, assert_exog_columns,
                          assert_scaler_fit_region)

VAL_START = TRAIN_END - 12 * PRED_LEN + 1      # 41641: last 12 x 168 steps are validation
ORIGIN_STRIDE = 24
CONTINUOUS = EXOG_COLUMNS[:6]                  # A-F are z-scored; G-J are one-hot, left as-is


def load():
    """Returns target y [43656] and exog X [43824, 10], both float32, after checking the
    properties the spec says can be relied on."""
    train = pd.read_csv(DATA_DIR / "student_train.csv")
    test = pd.read_csv(DATA_DIR / "student_test.csv")
    exog = pd.read_csv(DATA_DIR / "optional_external_data.csv")
    assert (train.time_idx.values == np.arange(1, TRAIN_END + 1)).all() and train.value.notna().all()
    assert (train.value >= 0).all()
    assert (test.time_idx.values == np.arange(TRAIN_END + 1, TRAIN_END + PRED_LEN + 1)).all()
    assert test.value.isna().all()
    assert (exog.time_idx.values == np.arange(1, TRAIN_END + PRED_LEN + 1)).all()
    assert exog[EXOG_COLUMNS].notna().all().all()
    assert (exog[EXOG_COLUMNS[6:]].sum(axis=1) == 1).all(), "G-J are not one-hot"
    return (train.value.values.astype(np.float32), exog[EXOG_COLUMNS].values.astype(np.float32))


class Scaler:
    """z-score (optionally after log1p) fitted on time_idx 1..fit_end only."""

    def __init__(self, values, fit_end, log1p=False):
        self.fit_end, self.log1p = fit_end, log1p
        fit = self._fwd(values[:fit_end])
        self.mean, self.std = fit.mean(0), fit.std(0) + 1e-8

    def _fwd(self, v):
        return np.log1p(v) if self.log1p else v

    def transform(self, v):
        return ((self._fwd(v) - self.mean) / self.std).astype(np.float32)

    def inverse(self, z):
        v = z * self.std + self.mean
        return np.expm1(v) if self.log1p else v


def prepare(y, X, train_end, exog, log1p=False):
    """Scale target and covariates on rows up to train_end. Returns the scaled target, the
    scaled covariates (all 43824 rows) and the target scaler."""
    exog_cols = [] if exog == "none" else EXOG_COLUMNS
    assert_exog_columns(exog_cols, exog)
    y_scaler = Scaler(y, train_end, log1p)
    x_scaler = Scaler(X[:, :6], train_end)
    for s in (y_scaler, x_scaler):
        assert_scaler_fit_region(s, train_end)
    Xs = np.concatenate([x_scaler.transform(X[:, :6]), X[:, 6:]], axis=1).astype(np.float32)
    return y_scaler.transform(y), Xs, y_scaler


def val_origins():
    """First-target time_idx of every validation forecast: stride 24, whole 168-block inside
    the validation region -> 78 origins."""
    return np.arange(VAL_START, TRAIN_END - PRED_LEN + 2, ORIGIN_STRIDE)


def train_starts(seq_len, train_end):
    """Every window whose 168 targets end at or before train_end (stride 1)."""
    return np.arange(seq_len + 1, train_end - PRED_LEN + 2)


def metrics(y_true, y_pred):
    """Per-row RMSE, MAE, sMAPE (0/0 terms count as 0) and bias for [N, 168] arrays."""
    err = y_pred - y_true
    denom = np.abs(y_true) + np.abs(y_pred)
    ratio = np.divide(2 * np.abs(err), denom, out=np.zeros_like(denom, dtype=np.float64), where=denom > 0)
    return {"rmse": np.sqrt((err ** 2).mean(1)), "mae": np.abs(err).mean(1),
            "smape": 100 * ratio.mean(1), "bias": err.mean(1)}


def targets(y, starts):
    return np.stack([y[s - 1:s - 1 + PRED_LEN] for s in starts])


if __name__ == "__main__":
    # Naive baseline: the mean of the last 168 observations, repeated over the horizon.
    y, _ = load()
    origins = val_origins()
    assert len(origins) == 78 and origins[0] == 41641 and origins[-1] + PRED_LEN - 1 <= TRAIN_END
    assert y[origins[0] - 1] == pd.read_csv(DATA_DIR / "student_train.csv").set_index("time_idx").value[41641]
    pred = np.stack([np.full(PRED_LEN, y[s - 1 - PRED_LEN:s - 1].mean()) for s in origins])
    m = metrics(targets(y, origins), pred)
    blocks = np.arange(0, 78, 7)              # origins 0, 7, ..., 77: the 12 non-overlapping blocks
    print(f"last-168-mean baseline over {len(origins)} origins: "
          + ", ".join(f"{k} {v.mean():.2f}" for k, v in m.items())
          + f"; block RMSE sd {m['rmse'][blocks].std(ddof=1):.2f} over {len(blocks)} blocks")
