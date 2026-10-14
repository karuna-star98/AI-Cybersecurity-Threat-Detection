"""Dataset loading and cleaning (Phase 2 + Phase 3)."""
import pandas as pd

from .config import CATEGORICAL, NUMERIC, RAW_DATA, TARGET


def load_dataset(path=RAW_DATA) -> pd.DataFrame:
    """Load a CSV. Your own dataset works too if it has the same column names
    (see docs/ARCHITECTURE.md) plus an `attack_type` label column."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    missing = [c for c in NUMERIC + CATEGORICAL + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def clean_data(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """Remove duplicates, bad labels and impossible values. Missing numbers are
    kept as NaN on purpose - the model pipeline imputes them (avoids data leakage)."""
    before = len(df)
    df = df.drop_duplicates().dropna(subset=[TARGET]).copy()
    df[CATEGORICAL] = df[CATEGORICAL].apply(lambda s: s.astype(str).str.strip().str.lower())
    for col in ["duration", "src_bytes", "dst_bytes", "request_frequency"]:
        df.loc[df[col] < 0, col] = float("nan")
    for col in ["serror_rate", "rerror_rate", "same_srv_rate"]:
        df[col] = df[col].clip(0, 1)
    df = df.reset_index(drop=True)
    if verbose:
        print(f"[clean] rows: {before} -> {len(df)} (removed {before - len(df)} duplicates/bad rows)")
        print(f"[clean] missing values left for the pipeline to impute: {int(df[NUMERIC].isna().sum().sum())}")
    return df
