"""
Phase 3 nested tuning: search spaces, inner-CV objective, and tuned-config lookup.

Protocol (per model, per seed s in 0-4, per outer fold k of that seed's partition):
  outer training fold -> StratifiedGroupKFold(3) by ProxID (inner) -> Optuna TPE, 25 trials,
  MedianPruner (reports inner-fold AUC after each inner fold), objective = mean inner AUC.
  Every model gets the same budget and the same preprocessing search space
  (SelectKBest k, PCA dim, resampler) plus its own hyperparameters.
Best params per (model, seed, outer fold) are saved to
results/phase3/tuning/<model>/seed<s>_outer<k>.json. Every seed is tuned independently, so
every evaluated fold is truly nested (its config never saw its test lesions).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .config import RESULTS_DIR
from .preprocess import build_fold
from .seeding import component_seed

TUNING_DIR = RESULTS_DIR / "phase3" / "tuning"

# model key -> fitter name in models.REGISTRY
FITTER = {
    "hqcnn": "hqcnn_twin",                  # exact twin == PennyLane HQCNN, faster
    "dephased_control": "dephased_control",
    "classical_tree_net": "classical_tree_net",
    "mlp_matched": "mlp_matched",
    "lr": "lr",
    "svm_rbf": "svm_rbf",
    "rf": "rf",
}
TUNED_MODELS = tuple(FITTER)


def suggest(trial, model: str) -> dict:
    p = {
        "k": trial.suggest_categorical("k", [16, 24, 32]),
        "pca_dim": trial.suggest_categorical("pca_dim", [8, 12, 16]),
        "resampler": trial.suggest_categorical("resampler", ["none", "smote", "borderline"]),
    }
    if model in ("hqcnn", "dephased_control"):
        p.update(
            lr=trial.suggest_float("lr", 5e-3, 8e-2, log=True),
            n_epochs=trial.suggest_categorical("n_epochs", [40, 60, 80, 100]),
            batch_size=trial.suggest_categorical("batch_size", [32, 64]),
            focal_gamma=trial.suggest_float("focal_gamma", 1.0, 3.0),
            conv_layers=trial.suggest_categorical("conv_layers", [1, 2, 3]),
        )
    elif model in ("classical_tree_net", "mlp_matched"):
        # early stopping = the tuned epoch budget (selected on inner validation AUC) + L2
        p.update(
            lr=trial.suggest_float("lr", 1e-3, 1e-1, log=True),
            n_epochs=trial.suggest_categorical("n_epochs", [20, 40, 80, 120]),
            batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
            focal_gamma=trial.suggest_float("focal_gamma", 1.0, 3.0),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
        )
    elif model == "lr":
        p.update(C=trial.suggest_float("C", 1e-3, 1e2, log=True),
                 class_weight=trial.suggest_categorical("class_weight", ["none", "balanced"]))
    elif model == "svm_rbf":
        p.update(C=trial.suggest_float("C", 1e-2, 1e2, log=True),
                 gamma=trial.suggest_float("gamma", 1e-3, 1.0, log=True))
    elif model == "rf":
        p.update(n_estimators=trial.suggest_categorical("n_estimators", [100, 300, 500]),
                 max_depth=trial.suggest_categorical("max_depth", [3, 5, 8, 0]),
                 min_samples_leaf=trial.suggest_categorical("min_samples_leaf", [1, 3, 5, 10]),
                 max_features=trial.suggest_categorical("max_features", ["sqrt", "half"]))
    else:
        raise KeyError(model)
    return p


def apply_params(cfg: dict, model: str, p: dict) -> dict:
    """Config copy with tuned preprocessing + model hyperparameters applied."""
    c = copy.deepcopy(cfg)
    c["preprocessing"].update(k=int(p["k"]), pca_dim=int(p["pca_dim"]), resampler=p["resampler"])
    if model in ("hqcnn", "dephased_control", "classical_tree_net", "mlp_matched"):
        for key in ("lr", "n_epochs", "batch_size", "focal_gamma", "conv_layers", "weight_decay"):
            if key in p:
                c["hqcnn"][key] = p[key]
    if model == "mlp_matched":
        d = int(p["pca_dim"])
        hidden = int(round(140 / (d + 2)))            # keep ~141 parameters for any PCA dim
        c.setdefault("phase2_arms", {}).setdefault("mlp_matched", {})["hidden"] = hidden
    if model == "lr":
        c["baselines"]["lr"].update(C=p["C"], class_weight=None if p["class_weight"] == "none" else "balanced")
    if model == "svm_rbf":
        c["baselines"]["svm_rbf"].update(C=p["C"], gamma=p["gamma"])
    if model == "rf":
        c["baselines"]["rf"].update(
            n_estimators=int(p["n_estimators"]), max_depth=None if p["max_depth"] == 0 else int(p["max_depth"]),
            min_samples_leaf=int(p["min_samples_leaf"]),
            max_features="sqrt" if p["max_features"] == "sqrt" else 0.5)
    return c


def inner_splits(cohort, train_idx: np.ndarray, seed: int, outer_fold: int, n_inner: int = 3):
    cv = StratifiedGroupKFold(n_splits=n_inner, shuffle=True,
                              random_state=component_seed(seed, outer_fold, "inner"))
    y, g = cohort.y[train_idx], cohort.groups[train_idx]
    return [(train_idx[a], train_idx[b]) for a, b in cv.split(np.zeros(len(y)), y, g)]


def make_objective(cohort, cfg: dict, model: str, train_idx: np.ndarray, seed: int, outer_fold: int):
    import optuna

    from .models import REGISTRY
    fitter = REGISTRY[FITTER[model]]
    splits = inner_splits(cohort, train_idx, seed, outer_fold)

    def objective(trial):
        p = suggest(trial, model)
        c = apply_params(cfg, model, p)
        aucs = []
        for j, (itr, iva) in enumerate(splits):
            s = 1000 + 10 * outer_fold + j                       # tuning-only seed stream
            fd = build_fold(cohort, itr, iva, c["preprocessing"], component_seed(seed, s, "resample"))
            res = fitter(fd, c, seed, s)
            aucs.append(float(roc_auc_score(fd.y_test, np.asarray(res["prob_test"]))))
            trial.report(float(np.mean(aucs)), j)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("inner_aucs", aucs)
        return float(np.mean(aucs))
    return objective


def best_path(model: str, seed: int, outer_fold: int) -> Path:
    return TUNING_DIR / model / f"seed{seed}_outer{outer_fold}.json"


def load_best(model: str, seed: int, outer_fold: int) -> dict:
    return json.loads(best_path(model, seed, outer_fold).read_text())


def tuned_cfg(cfg: dict, model: str, seed: int, fold: int) -> dict:
    """Config for evaluating `model` on (seed, fold) with the params tuned by inner CV on
    exactly that outer training fold (fully nested for every seed)."""
    return apply_params(cfg, model, load_best(model, seed, fold)["params"])
