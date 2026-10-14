"""Evaluation metrics (Phase 6)."""
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

from .config import CLASSES


def binary_metrics(y_true_attack, y_pred_attack) -> dict:
    """Attack (1) vs Normal (0). Recall = share of real attacks we caught;
    precision = share of alerts that were real (false-alarm control)."""
    return {
        "accuracy": accuracy_score(y_true_attack, y_pred_attack),
        "precision": precision_score(y_true_attack, y_pred_attack, zero_division=0),
        "recall": recall_score(y_true_attack, y_pred_attack, zero_division=0),
        "f1": f1_score(y_true_attack, y_pred_attack, zero_division=0),
    }


def multiclass_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def report_text(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, labels=CLASSES, zero_division=0, digits=3)


def confusion(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=CLASSES)
