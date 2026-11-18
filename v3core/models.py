"""Model registry: every model is fit on a FoldData and returns train/test probs."""

from __future__ import annotations

import time

import numpy as np

from .hqcnn import HQCNN
from .preprocess import FoldData
from .seeding import component_rng, component_seed


def fit_hqcnn(fd: FoldData, cfg: dict, seed: int, fold: int) -> dict:
    model = HQCNN(cfg["hqcnn"])
    rng = component_rng(seed, fold, "hqcnn")
    t0 = time.perf_counter()
    model.fit(fd.X_fit, fd.y_fit, rng)
    fit_s = time.perf_counter() - t0
    return {
        "prob_train": model.predict_proba(fd.X_train),
        "prob_test": model.predict_proba(fd.X_test),
        "threshold_strategy": cfg["hqcnn"]["threshold"],
        "meta": {
            "fit_seconds": fit_s,
            "n_params": model.n_params,
            "n_quantum_params": model.n_q,
            "readout_qubit": model.readout,
            "loss_history": [float(v) for v in model.loss_history_],
            # trained weights + angle scaler, so a fold can be re-evaluated exactly
            "params": model.params_.tolist(),
            "scaler_lo": model.lo_.tolist(),
            "scaler_hi": model.hi_.tolist(),
        },
    }


def _sklearn_estimator(name: str, bcfg: dict, random_state: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    p = bcfg[name]
    if name == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=p["C"], max_iter=p["max_iter"], class_weight=p.get("class_weight"),
            random_state=random_state))
    if name == "svm_rbf":
        return make_pipeline(StandardScaler(), SVC(
            C=p["C"], gamma=p["gamma"], kernel="rbf", probability=True,
            random_state=random_state))
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            min_samples_leaf=p.get("min_samples_leaf", 1),
            max_features=p.get("max_features", "sqrt"),
            random_state=random_state, n_jobs=1)
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=tuple(p["hidden_layer_sizes"]), alpha=p["alpha"],
            max_iter=p["max_iter"], random_state=random_state))
    raise KeyError(name)


def _n_params(est) -> int | None:
    last = est.steps[-1][1] if hasattr(est, "steps") else est
    if hasattr(last, "coefs_"):
        return int(sum(w.size for w in last.coefs_) + sum(b.size for b in last.intercepts_))
    if hasattr(last, "coef_"):
        return int(last.coef_.size + np.size(last.intercept_))
    return None


def make_sklearn_fitter(name: str):
    """Classical baseline on the SAME resampled PCA inputs the HQCNN trains on."""
    def fit(fd: FoldData, cfg: dict, seed: int, fold: int) -> dict:
        est = _sklearn_estimator(name, cfg["baselines"], component_seed(seed, fold, name))
        t0 = time.perf_counter()
        est.fit(fd.X_fit, fd.y_fit)
        return {
            "prob_train": est.predict_proba(fd.X_train)[:, 1],
            "prob_test": est.predict_proba(fd.X_test)[:, 1],
            "threshold_strategy": cfg["baselines"]["threshold"],
            "meta": {"fit_seconds": time.perf_counter() - t0, "n_params": _n_params(est)},
        }
    return fit


def fit_clinical_lr(fd: FoldData, cfg: dict, seed: int, fold: int) -> dict:
    """Clinical-only LR; fd comes from preprocess.clinical_arm_fold (no resampling)."""
    from sklearn.linear_model import LogisticRegression

    ccfg = cfg["clinical_arms"]
    est = LogisticRegression(C=ccfg["C"], class_weight=ccfg["class_weight"],
                             max_iter=2000, random_state=component_seed(seed, fold, "clin_lr"))
    t0 = time.perf_counter()
    est.fit(fd.X_fit, fd.y_fit)
    return {
        "prob_train": est.predict_proba(fd.X_train)[:, 1],
        "prob_test": est.predict_proba(fd.X_test)[:, 1],
        "threshold_strategy": ccfg["threshold"],
        "meta": {
            "fit_seconds": time.perf_counter() - t0,
            "n_params": _n_params(est),
            "coefficients": dict(zip(fd.meta["fused_features"], est.coef_[0].tolist())),
        },
    }


REGISTRY = {
    "hqcnn": fit_hqcnn,
    "lr": make_sklearn_fitter("lr"),
    "svm_rbf": make_sklearn_fitter("svm_rbf"),
    "rf": make_sklearn_fitter("rf"),
    "mlp": make_sklearn_fitter("mlp"),
    "clin_lr_prebiopsy": fit_clinical_lr,
    "clin_lr_gleason_leakref": fit_clinical_lr,
}


def fit_model(name: str, fd: FoldData, cfg: dict, seed: int, fold: int) -> dict:
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; known: {sorted(REGISTRY)}")
    out = REGISTRY[name](fd, cfg, seed, fold)
    out["prob_train"] = np.asarray(out["prob_train"], dtype=np.float64)
    out["prob_test"] = np.asarray(out["prob_test"], dtype=np.float64)
    return out


# Phase 2C ablation arms (v3core/arms.py) share the same fitter interface.
from .arms import ARM_REGISTRY  # noqa: E402

REGISTRY.update(ARM_REGISTRY)
