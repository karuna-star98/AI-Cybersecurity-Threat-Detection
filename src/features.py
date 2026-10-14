"""Feature engineering + the sklearn preprocessing pipeline.

Everything lives INSIDE the saved model pipeline, so the API and the detector
can feed raw records straight in - no separate scaler files to keep in sync.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .config import CATEGORICAL, ENGINEERED, NUMERIC

# Values used when a caller (e.g. the REST API) sends only a few fields
DEFAULTS = {
    "duration": 2.0, "src_bytes": 700.0, "dst_bytes": 1800.0, "packet_count": 20,
    "failed_logins": 0, "logged_in": 1, "request_frequency": 10.0, "count": 8,
    "srv_count": 6, "serror_rate": 0.03, "rerror_rate": 0.03, "same_srv_rate": 0.8,
    "unique_dst_ports": 2, "hour": 12,
    "protocol_type": "tcp", "service": "http", "flag": "SF",
}
# Friendly names -> internal column names (matches the "packet_size" example in the guide)
ALIASES = {"packet_size": "src_bytes", "request_rate": "request_frequency"}


def fill_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Rename aliases and add any missing model columns with safe defaults."""
    df = df.rename(columns=ALIASES).copy()
    for col, default in DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create behaviour-based features that make attacks easier to separate."""
    df = df.copy()
    df["total_bytes"] = df["src_bytes"] + df["dst_bytes"]
    df["bytes_ratio"] = df["src_bytes"] / (df["dst_bytes"] + 1)          # upload-heavy => exfiltration
    df["bytes_per_packet"] = df["total_bytes"] / (df["packet_count"] + 1)
    df["login_fail_ratio"] = df["failed_logins"] / (df["count"] + 1)       # brute force signal
    df["error_rate_sum"] = df["serror_rate"] + df["rerror_rate"]           # scans / floods
    df["is_night"] = (df["hour"] < 6).astype(int)
    df["log_src_bytes"] = np.log1p(df["src_bytes"].clip(lower=0))
    return df


def build_preprocessor() -> Pipeline:
    """engineer features -> impute -> scale numbers / one-hot encode categories."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    columns = ColumnTransformer([
        ("num", numeric, NUMERIC + ENGINEERED),
        ("cat", categorical, CATEGORICAL),
    ])
    return Pipeline([
        ("engineer", FunctionTransformer(add_engineered_features, validate=False)),
        ("columns", columns),
    ])
