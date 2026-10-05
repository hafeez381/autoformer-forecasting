"""Leakage and reporting guards. Every script that loads data imports this module first.

Each guard raises AssertionError on a breach, so a wrong pipeline stops instead of
producing a number.
"""
import hashlib
import math
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_SHA256 = {
    "student_train.csv": "04658e259dab4f6b2b474cf582c5bfc9ca945674c23350c122abd0f11a63c9f5",
    "student_test.csv": "a1faab0c513a65dc13536452f50d1b483c0089421418154dc8b0381bd9952c63",
    "optional_external_data.csv": "b4af0e3b7e7ca5183c2e0bd1f4d7d5aec16d9a5ef8ae3de67ffdc7e851b61b85",
}
EXOG_COLUMNS = [f"feature_{c}" for c in "ABCDEFGHIJ"]
TRAIN_END = 43656            # last observed time_idx
FORECAST_IDX = (43657, 43824)
PRED_LEN = 168


def assert_no_external_series():
    """Only the three supplied CSVs may sit in the data directory, unchanged."""
    files = sorted(p.name for p in DATA_DIR.iterdir() if not p.name.startswith("."))
    assert files == sorted(DATA_SHA256), f"unexpected files in {DATA_DIR}: {files}"
    for name, sha in DATA_SHA256.items():
        got = hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest()
        assert got == sha, f"{name} changed: sha256 {got}"


def assert_chronological_split(train_target_end, val_start):
    """Every training target lies strictly before the first validation target."""
    assert train_target_end < val_start, (
        f"training target reaches time_idx {train_target_end}, validation starts at {val_start}")


def assert_scaler_fit_region(scaler, last_allowed_idx):
    """Scalers are fit on rows up to the last training time_idx, never later."""
    assert scaler.fit_end <= last_allowed_idx, (
        f"scaler fit up to time_idx {scaler.fit_end}, allowed only up to {last_allowed_idx}")


def assert_decoder_future_target_masked(dec_target):
    """The decoder's last pred_len target slots must be placeholders (zeros), never true values."""
    tail = dec_target[:, -PRED_LEN:]
    assert float(abs(tail).max()) == 0.0, "decoder future target slots carry real values"


def assert_exog_columns(cols, regime):
    """Only feature_A..feature_J may enter as covariates, under a declared regime."""
    assert regime in ("none", "past", "future", "both"), f"unknown exog regime {regime}"
    assert set(cols) <= set(EXOG_COLUMNS), f"non-exog columns in covariates: {set(cols) - set(EXOG_COLUMNS)}"
    assert (regime == "none") == (len(cols) == 0), f"regime {regime} with columns {cols}"


def assert_param_count(models, declared_p):
    got = sum(p.numel() for m in models for p in m.parameters() if p.requires_grad)
    assert got == declared_p, f"counted P={got}, declared P={declared_p}"


def assert_epoch_count(epochs_per_member, declared_e):
    got = sum(epochs_per_member)
    assert got == declared_e, f"logged E={got}, declared E={declared_e}"


def assert_submission_string(s, forecast_values):
    """Exactly 168 finite values, in time_idx order 43657..43824, matching the saved forecast."""
    vals = [float(v) for v in s.split(",")]
    assert len(vals) == PRED_LEN, f"{len(vals)} values, need {PRED_LEN}"
    assert all(math.isfinite(v) for v in vals), "non-finite value in submission"
    assert np.allclose(vals, forecast_values, rtol=1e-5, atol=1e-6), "string does not round-trip to forecast.csv"


assert_no_external_series()
