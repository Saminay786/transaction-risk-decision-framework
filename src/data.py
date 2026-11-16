"""Load the ULB credit-card data and make time-ordered splits."""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT

RAW_PATH = ROOT / "data" / "raw" / "creditcard.csv"
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount", "hour"]
TARGET = "Class"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[TARGET] = df[TARGET].astype(str).str.strip('"').astype(int)
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    # Time is seconds since the first transaction; data spans ~2 days.
    df["hour"] = (df["Time"] // 3600 % 24).astype(int)
    return df


def time_split(df: pd.DataFrame, fracs=(0.6, 0.2, 0.2)) -> dict:
    """Split by row order (already sorted by Time).

    Returns train / val_a / val_b / test. Validation (20%) is cut in half by time:
    val_a -> early stopping + calibration fit; val_b -> calibrator choice + policy tuning.
    """
    assert abs(sum(fracs) - 1) < 1e-9
    n = len(df)
    i_tr = int(n * fracs[0])
    i_va = int(n * (fracs[0] + fracs[1]))
    i_mid = (i_tr + i_va) // 2
    return {
        "train": df.iloc[:i_tr].reset_index(drop=True),
        "val_a": df.iloc[i_tr:i_mid].reset_index(drop=True),
        "val_b": df.iloc[i_mid:i_va].reset_index(drop=True),
        "test": df.iloc[i_va:].reset_index(drop=True),
    }


def split_summary(splits: dict) -> pd.DataFrame:
    rows = []
    for name, d in splits.items():
        rows.append({
            "split": name,
            "rows": len(d),
            "frauds": int(d[TARGET].sum()),
            "fraud_rate_pct": round(100 * d[TARGET].mean(), 4),
            "fraud_amount_eur": round(float(d.loc[d[TARGET] == 1, "Amount"].sum()), 2),
            "time_start_h": round(d["Time"].min() / 3600, 2),
            "time_end_h": round(d["Time"].max() / 3600, 2),
        })
    return pd.DataFrame(rows)


def xy(d: pd.DataFrame):
    return d[FEATURES], d[TARGET].to_numpy(), d["Amount"].to_numpy(dtype=float)
