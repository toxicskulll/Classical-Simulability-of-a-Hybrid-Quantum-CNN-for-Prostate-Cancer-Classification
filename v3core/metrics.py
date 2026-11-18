"""Thresholding and per-fold classification metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRICS = ("AUC", "Recall", "Precision", "F1", "Specificity", "BalancedAccuracy")


def f1_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    """F1-optimal cut on probabilities (midpoints between sorted unique scores).
    Always computed on training-fold scores, never on the test fold."""
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=np.float64)
    uniq = np.unique(prob)
    if len(uniq) == 1:
        return float(uniq[0])
    cands = (uniq[:-1] + uniq[1:]) / 2.0
    best_t, best_f1 = float(cands[0]), -1.0
    for t in cands:
        f1 = f1_score(y_true, (prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def pick_threshold(strategy: str, y_true: np.ndarray, prob: np.ndarray) -> float:
    if strategy == "f1":
        return f1_threshold(y_true, prob)
    if strategy == "0.5":
        return 0.5
    raise ValueError(f"unknown threshold strategy: {strategy}")


def fold_metrics(y_true: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    pred = np.asarray(pred, dtype=int)
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    return {
        "AUC": float(roc_auc_score(y_true, prob)),
        "Recall": float(recall_score(y_true, pred, zero_division=0)),
        "Precision": float(precision_score(y_true, pred, zero_division=0)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "Specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "BalancedAccuracy": float(balanced_accuracy_score(y_true, pred)),
    }
