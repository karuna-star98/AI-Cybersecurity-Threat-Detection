"""Visualisation / dashboard (Phase 8). All charts are saved to images/."""
import json

import matplotlib
matplotlib.use("Agg")                      # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import CLASSES, IMAGES_DIR, NUMERIC, OUTPUTS_DIR, RF_MODEL_PATH
from .evaluate import confusion

sns.set_theme(style="whitegrid")
PALETTE = {"Normal": "#4C9F70", "DoS": "#D1495B", "BruteForce": "#EDAE49",
           "PortScan": "#00798C", "DataExfiltration": "#8E4585", "UnknownAnomaly": "#6C757D"}


def _save(fig, name):
    IMAGES_DIR.mkdir(exist_ok=True)
    fig.savefig(IMAGES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---- individual panels (each draws on a given axis so the dashboard can reuse them) ----
def p_distribution(ax, df):
    counts = df["attack_type"].value_counts().reindex(CLASSES)
    ax.bar(counts.index, counts.values, color=[PALETTE[c] for c in counts.index])
    ax.set_title("Traffic class distribution"); ax.set_ylabel("flows")
    ax.tick_params(axis="x", rotation=30)


def p_confusion(ax, y_true, y_pred):
    cm = confusion(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES,
                yticklabels=CLASSES, cbar=False, ax=ax)
    ax.set_title("Confusion matrix - Random Forest"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.tick_params(axis="x", rotation=30)


def p_models(ax, metrics):
    m = pd.DataFrame(metrics["binary"]).T[["precision", "recall", "f1"]]
    m.plot(kind="bar", ax=ax, color=["#00798C", "#D1495B", "#EDAE49"])
    ax.set_ylim(0.8, 1.0); ax.set_title("Model comparison (attack vs normal)")
    ax.tick_params(axis="x", rotation=15); ax.legend(loc="lower right")


def p_importance(ax, top=10):
    import joblib
    pipe = joblib.load(RF_MODEL_PATH)
    names = pipe.named_steps["prep"].named_steps["columns"].get_feature_names_out()
    names = [n.split("__")[-1] for n in names]
    imp = pd.Series(pipe.named_steps["model"].feature_importances_, index=names).nlargest(top)
    imp[::-1].plot(kind="barh", ax=ax, color="#00798C")
    ax.set_title(f"Top {top} features (Random Forest)")


def p_anomaly(ax, preds):
    normal = preds[preds["attack_type"] == "Normal"]["iso_score"]
    attack = preds[preds["attack_type"] != "Normal"]["iso_score"]
    ax.hist(normal, bins=40, alpha=0.7, label="Normal", color=PALETTE["Normal"])
    ax.hist(attack, bins=40, alpha=0.7, label="Attack", color=PALETTE["DoS"])
    ax.set_title("Isolation Forest anomaly scores"); ax.set_xlabel("anomaly score (higher = stranger)")
    ax.legend()


def p_timeline(ax, alerts):
    a = alerts.copy()
    a["timestamp"] = pd.to_datetime(a["timestamp"])
    pivot = (a.set_index("timestamp").groupby([pd.Grouper(freq="6h"), "threat_type"])
             .size().unstack(fill_value=0))
    x = np.arange(len(pivot))
    bottom = np.zeros(len(pivot))
    for col in pivot.columns:                       # stacked bars drawn by hand (plain integer x-axis)
        ax.bar(x, pivot[col].to_numpy(), bottom=bottom, label=col,
               color=PALETTE.get(col, "#999"), width=0.9)
        bottom += pivot[col].to_numpy()
    step = max(1, len(pivot) // 7)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([t.strftime("%d %b %H:%M") for t in pivot.index[::step]],
                       rotation=30, ha="right")
    ax.set_title("Alerts over time (6-hour buckets)")
    ax.set_ylabel("alerts"); ax.legend(fontsize=8)


def p_severity(ax, alerts):
    order = ["Critical", "High", "Medium", "Low"]
    counts = alerts["severity"].value_counts().reindex(order).fillna(0)
    ax.bar(counts.index, counts.values, color=["#7B1E3A", "#D1495B", "#EDAE49", "#8AB17D"])
    ax.set_title("Alerts by severity")


def generate_all():
    """Create every chart. Needs: data CSV, outputs/test_predictions.csv, models, alerts."""
    from .config import RAW_DATA
    df = pd.read_csv(RAW_DATA)
    preds = pd.read_csv(OUTPUTS_DIR / "test_predictions.csv")
    metrics = json.loads((OUTPUTS_DIR / "metrics.json").read_text())
    alerts = pd.read_csv(OUTPUTS_DIR / "alerts.csv")

    # 1 class distribution
    fig, ax = plt.subplots(figsize=(7, 4)); p_distribution(ax, df); _save(fig, "01_class_distribution.png")

    # 2 correlation heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df[NUMERIC].corr(), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature correlation heatmap"); _save(fig, "02_correlation_heatmap.png")

    # 3 feature behaviour by class (log scale shows the attack fingerprints)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.ravel(), ["request_frequency", "src_bytes", "failed_logins", "unique_dst_ports"]):
        sns.boxplot(data=df, x="attack_type", y=col, order=CLASSES, palette=PALETTE, ax=ax, hue="attack_type", legend=False)
        if col in ("request_frequency", "src_bytes", "unique_dst_ports"):
            ax.set_yscale("log")
        ax.set_title(col); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=20)
    fig.suptitle("How each attack looks different from normal traffic"); fig.tight_layout()
    _save(fig, "03_attack_fingerprints.png")

    # 4 confusion matrix
    fig, ax = plt.subplots(figsize=(7, 6)); p_confusion(ax, preds["attack_type"], preds["rf_pred"])
    _save(fig, "04_confusion_matrix.png")

    # 5 model comparison
    fig, ax = plt.subplots(figsize=(7, 4.5)); p_models(ax, metrics); _save(fig, "05_model_comparison.png")

    # 6 feature importance
    fig, ax = plt.subplots(figsize=(7, 5)); p_importance(ax); _save(fig, "06_feature_importance.png")

    # 7 anomaly scores
    fig, ax = plt.subplots(figsize=(7, 4.5)); p_anomaly(ax, preds); _save(fig, "07_anomaly_scores.png")

    # 8 alert timeline + 9 severity
    fig, ax = plt.subplots(figsize=(10, 4.5)); p_timeline(ax, alerts); _save(fig, "08_alert_timeline.png")
    fig, ax = plt.subplots(figsize=(6, 4)); p_severity(ax, alerts); _save(fig, "09_alert_severity.png")

    # 10 SOC dashboard (everything in one picture - use this as the README hero image)
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    p_distribution(axes[0, 0], df); p_confusion(axes[0, 1], preds["attack_type"], preds["rf_pred"])
    p_models(axes[0, 2], metrics); p_importance(axes[1, 0]); p_anomaly(axes[1, 1], preds)
    p_timeline(axes[1, 2], alerts)
    fig.suptitle("AI-Powered Cybersecurity Threat Detection - SOC Dashboard", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96]); _save(fig, "10_soc_dashboard.png")
    print(f"[visualize] 10 charts saved to {IMAGES_DIR}")
