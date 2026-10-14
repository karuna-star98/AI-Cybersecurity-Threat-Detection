"""Run with:  python -m pytest -q"""
import pandas as pd

from src.config import CLASSES, NUMERIC, CATEGORICAL
from src.data_simulator import generate_traffic
from src.features import add_engineered_features, fill_defaults
from src.preprocessing import clean_data


def test_simulator_has_all_classes_and_is_reproducible():
    a = generate_traffic(1000, seed=1)
    b = generate_traffic(1000, seed=1)
    assert set(a["attack_type"]) == set(CLASSES)
    pd.testing.assert_frame_equal(a, b)


def test_cleaning_removes_duplicates():
    df = generate_traffic(1000, seed=2)
    assert len(clean_data(df, verbose=False)) < len(df)


def test_feature_engineering_adds_columns():
    df = fill_defaults(pd.DataFrame([{"packet_size": 1500, "failed_logins": 5}]))
    out = add_engineered_features(df)
    assert {"total_bytes", "bytes_ratio", "is_night"} <= set(out.columns)


def test_api_detects_attack_and_normal():
    from src.app import app
    client = app.test_client()
    assert client.get("/health").get_json() == {"status": "ok"}

    # the 3-field example from the project guide
    r = client.post("/predict", json={"packet_size": 1500, "failed_logins": 5,
                                      "request_frequency": 250})
    assert r.status_code == 200 and "Threat_Detected" in r.get_json()

    dos = {"duration": 0.05, "src_bytes": 200, "dst_bytes": 0, "packet_count": 320,
           "request_frequency": 420, "count": 160, "srv_count": 150,
           "serror_rate": 0.9, "flag": "S0", "logged_in": 0}
    assert client.post("/predict", json=dos).get_json()["threat_type"] == "DoS"
    assert client.post("/predict", json={}).status_code == 400
