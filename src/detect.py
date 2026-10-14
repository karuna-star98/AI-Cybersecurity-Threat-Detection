"""Threat detection logic + alert generation (Phase 7).

Two detectors work together, like a real SOC pipeline:
  * Random Forest  -> names the KNOWN attack type (DoS, BruteForce, ...)
  * Isolation Forest -> catches traffic that is highly abnormal even if the
                        classifier calls it Normal ("zero-day" style anomaly)
"""
import json
import logging
import time

import joblib
import numpy as np
import pandas as pd

from .config import (CATEGORICAL, ISO_MODEL_PATH, META_PATH, NUMERIC, OUTPUTS_DIR,
                     RF_MODEL_PATH)
from .features import fill_defaults

LEVELS = ["Low", "Medium", "High", "Critical"]
BASE_SEVERITY = {"DataExfiltration": "Critical", "DoS": "High", "BruteForce": "High",
                 "PortScan": "Medium", "UnknownAnomaly": "Medium"}


def load_models():
    rf = joblib.load(RF_MODEL_PATH)
    iso = joblib.load(ISO_MODEL_PATH)
    threshold = json.loads(META_PATH.read_text())["iso_alert_threshold"]
    return rf, iso, threshold


def score_traffic(df: pd.DataFrame, rf, iso, iso_threshold) -> pd.DataFrame:
    """Add prediction columns to a traffic DataFrame."""
    X = fill_defaults(df)[NUMERIC + CATEGORICAL]
    proba = rf.predict_proba(X)
    out = df.copy()
    out["pred_type"] = rf.classes_[proba.argmax(axis=1)]
    out["confidence"] = proba.max(axis=1).round(3)
    out["anomaly_score"] = (-iso.score_samples(X)).round(4)
    out["is_anomaly"] = out["anomaly_score"] > iso_threshold
    known_attack = out["pred_type"] != "Normal"
    unknown = (~known_attack) & out["is_anomaly"]
    out["threat_type"] = np.where(known_attack, out["pred_type"],
                                  np.where(unknown, "UnknownAnomaly", "Normal"))
    out["threat"] = out["threat_type"] != "Normal"
    return out


def severity(threat_type: str, confidence: float) -> str:
    """Base severity by attack type; lowered one level when the model is unsure."""
    level = LEVELS.index(BASE_SEVERITY.get(threat_type, "Medium"))
    if confidence < 0.60:
        level = max(0, level - 1)
    return LEVELS[level]


def explain(row) -> str:
    """Human-readable reason, so an analyst can see WHY the alert fired."""
    t = row["threat_type"]
    if t == "DoS":
        return (f"request rate {row['request_frequency']:.0f}/s, "
                f"SYN-error rate {row['serror_rate']:.2f}, {int(row['count'])} connections")
    if t == "BruteForce":
        return f"{int(row['failed_logins'])} failed logins on {row['service']}"
    if t == "PortScan":
        return f"{int(row['unique_dst_ports'])} different ports probed, reject rate {row['rerror_rate']:.2f}"
    if t == "DataExfiltration":
        return f"{row['src_bytes'] / 1e6:.2f} MB sent out over {row['duration']:.0f}s (hour {int(row['hour'])})"
    return f"behaviour far from normal baseline (anomaly score {row['anomaly_score']:.3f})"


def build_alerts(scored: pd.DataFrame) -> pd.DataFrame:
    hits = scored[scored["threat"]].copy()
    if hits.empty:
        return hits
    hits["severity"] = [severity(t, c) for t, c in zip(hits["threat_type"], hits["confidence"])]
    hits["reason"] = hits.apply(explain, axis=1)
    hits = hits.reset_index(drop=True)
    hits.insert(0, "alert_id", [f"ALERT-{i + 1:05d}" for i in range(len(hits))])
    cols = ["alert_id", "timestamp", "severity", "threat_type", "confidence", "anomaly_score",
            "src_ip", "dst_ip", "service", "protocol_type", "reason"]
    if "attack_type" in hits.columns:
        cols.append("attack_type")          # ground truth (only exists in simulation)
    return hits[cols]


def _summary(scored: pd.DataFrame, alerts: pd.DataFrame) -> str:
    lines = ["=" * 62, "  THREAT DETECTION SUMMARY", "=" * 62,
             f"Flows analysed : {len(scored)}", f"Alerts raised  : {len(alerts)}", ""]
    if not alerts.empty:
        lines += ["Alerts by type:", alerts["threat_type"].value_counts().to_string(), "",
                  "Alerts by severity:",
                  alerts["severity"].value_counts().reindex(LEVELS[::-1]).dropna().astype(int).to_string(), "",
                  "Top suspicious source IPs:", alerts["src_ip"].value_counts().head(5).to_string(), ""]
    if "attack_type" in scored.columns:           # simulation only: we know the truth
        real = scored["attack_type"] != "Normal"
        caught = real & scored["threat"]
        false_alarm = (~real) & scored["threat"]
        lines += ["Simulation check (ground truth):",
                  f"  real attacks in traffic : {int(real.sum())}",
                  f"  attacks detected        : {int(caught.sum())} ({caught.sum() / max(real.sum(), 1):.1%} recall)",
                  f"  false alarms            : {int(false_alarm.sum())}"]
    return "\n".join(lines)


def run_detection(df: pd.DataFrame, replay_delay=0.0, show=10):
    """Run the full detection pipeline on a traffic DataFrame and save the alerts."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    rf, iso, threshold = load_models()
    scored = score_traffic(df, rf, iso, threshold)
    alerts = build_alerts(scored)

    logging.basicConfig(filename=OUTPUTS_DIR / "alerts.log", filemode="w",
                        level=logging.INFO, format="%(asctime)s %(message)s")
    for _, a in alerts.iterrows():
        logging.info("[%s] %s %s src=%s dst=%s conf=%.2f | %s", a["severity"].upper(),
                     a["alert_id"], a["threat_type"], a["src_ip"], a["dst_ip"],
                     a["confidence"], a["reason"])

    # Console "SOC feed": replay the first alerts one-by-one (nice for a screen recording)
    for _, a in alerts.head(show).iterrows():
        print(f"[{a['severity'].upper():8}] {a['alert_id']} {a['threat_type']:17} "
              f"{a['src_ip']:>15} -> {a['dst_ip']:<11} conf={a['confidence']:.2f} | {a['reason']}")
        if replay_delay:
            time.sleep(replay_delay)
    if len(alerts) > show:
        print(f"... and {len(alerts) - show} more alerts (see outputs/alerts.csv)")

    alerts.to_csv(OUTPUTS_DIR / "alerts.csv", index=False)
    alerts.assign(timestamp=alerts["timestamp"].astype(str)).to_json(
        OUTPUTS_DIR / "alerts.json", orient="records", indent=2)
    scored.to_csv(OUTPUTS_DIR / "scored_traffic.csv", index=False)
    text = _summary(scored, alerts)
    (OUTPUTS_DIR / "detection_summary.txt").write_text(text)
    print("\n" + text)
    return scored, alerts
