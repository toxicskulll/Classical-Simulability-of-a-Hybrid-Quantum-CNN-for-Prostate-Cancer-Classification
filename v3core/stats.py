"""AUC, paired DeLong test and patient-cluster bootstrap."""

from __future__ import annotations

import numpy as np
from scipy import stats


def auc_mw(y: np.ndarray, s: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U statistic (ties get half credit)."""
    y = np.asarray(y, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _midrank(x: np.ndarray) -> np.ndarray:
    return stats.rankdata(x, method="average")


def delong_paired(y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray) -> dict:
    """Paired DeLong test for AUC(a) - AUC(b) on the same cases (Sun & Xu 2014).
    Assumes cases are independent; lesions clustered within patients violate this
    mildly, so the patient-cluster bootstrap is reported alongside."""
    y = np.asarray(y, dtype=bool)
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    m, n = len(pos), len(neg)
    scores = np.vstack([s_a, s_b]).astype(np.float64)
    aucs, v10, v01 = [], [], []
    for s in scores:
        tx, ty, tz = _midrank(s[pos]), _midrank(s[neg]), _midrank(np.concatenate([s[pos], s[neg]]))
        auc = (tz[:m].sum() - m * (m + 1) / 2.0) / (m * n)
        aucs.append(auc)
        v10.append((tz[:m] - tx) / n)
        v01.append(1.0 - (tz[m:] - ty) / m)
    v10, v01 = np.array(v10), np.array(v01)
    cov = np.cov(v10) / m + np.cov(v01) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    delta = aucs[0] - aucs[1]
    if var <= 0:
        return {"auc_a": aucs[0], "auc_b": aucs[1], "delta": delta, "z": float("nan"), "p": float("nan")}
    z = delta / np.sqrt(var)
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "delta": float(delta),
            "z": float(z), "p": float(2.0 * stats.norm.sf(abs(z)))}


def patient_bootstrap_indices(groups: np.ndarray, n_boot: int, seed: int):
    """Yield lesion index arrays from resampling patients with replacement."""
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([members[g] for g in pick])
