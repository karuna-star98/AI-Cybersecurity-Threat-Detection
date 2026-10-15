"""REST API for real-time prediction (Flask).

Run:  python main.py serve      ->  http://127.0.0.1:5000
"""
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

from .detect import load_models, score_traffic, severity

app = Flask(__name__)

@app.get("/")
def home():
    return {
        "project": "AI Cybersecurity Threat Detection",
        "status": "running",
        "message": "Cybersecurity threat detection API is online.",
        "available_endpoints": [
            "/",
            "/health",
            "/predict"
        ]
    }

_models = {}


def _get_models():
    if not _models:
        _models["rf"], _models["iso"], _models["thr"] = load_models()
    return _models["rf"], _models["iso"], _models["thr"]


def _to_result(row) -> dict:
    return {
        "Threat_Detected": bool(row["threat"]),
        "threat_type": row["threat_type"],
        "confidence": float(row["confidence"]),
        "severity": severity(row["threat_type"], row["confidence"]) if row["threat"] else "None",
        "anomaly_score": float(row["anomaly_score"]),
    }


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    """Body: one JSON object. Any missing field gets a safe default, so the
    3-field example from the guide works: packet_size, failed_logins, request_frequency."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not data:
        return jsonify({"error": "Send a JSON object with traffic features"}), 400
    try:
        scored = score_traffic(pd.DataFrame([data]), *_get_models())
    except Exception as exc:                      # bad types, etc.
        return jsonify({"error": str(exc)}), 400
    return jsonify(_to_result(scored.iloc[0]))


@app.post("/predict_batch")
def predict_batch():
    """Body: JSON list of traffic objects."""
    data = request.get_json(silent=True)
    if not isinstance(data, list) or not data:
        return jsonify({"error": "Send a non-empty JSON list"}), 400
    try:
        scored = score_traffic(pd.DataFrame(data), *_get_models())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify([_to_result(r) for _, r in scored.iterrows()])


if __name__ == "__main__":
    app.run(debug=False)
