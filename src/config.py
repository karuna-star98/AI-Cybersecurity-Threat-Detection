"""Central configuration: paths, column names and class labels."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
IMAGES_DIR = ROOT / "images"

RAW_DATA = DATA_DIR / "network_traffic.csv"      # training data
LIVE_DATA = DATA_DIR / "live_traffic.csv"        # "unseen" traffic for the detection demo
RF_MODEL_PATH = MODELS_DIR / "threat_model.joblib"
ISO_MODEL_PATH = MODELS_DIR / "isolation_forest.joblib"
META_PATH = MODELS_DIR / "meta.json"

RANDOM_STATE = 42
TARGET = "attack_type"
CLASSES = ["Normal", "DoS", "BruteForce", "PortScan", "DataExfiltration"]

# Raw model inputs (what a network sensor / firewall log would give us)
CATEGORICAL = ["protocol_type", "service", "flag"]
NUMERIC = [
    "duration", "src_bytes", "dst_bytes", "packet_count", "failed_logins",
    "logged_in", "request_frequency", "count", "srv_count", "serror_rate",
    "rerror_rate", "same_srv_rate", "unique_dst_ports", "hour",
]
# Created in src/features.py
ENGINEERED = [
    "total_bytes", "bytes_ratio", "bytes_per_packet", "login_fail_ratio",
    "error_rate_sum", "is_night", "log_src_bytes",
]
META_COLS = ["timestamp", "src_ip", "dst_ip"]    # kept for alerts, NOT used by the model
