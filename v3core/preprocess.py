"""
Fold-local preprocessing. Everything is fitted on the training fold only:

  radiomics: drop zero-variance -> StandardScaler -> SelectKBest(f_classif, k)
  clinical:  median impute -> StandardScaler
  fusion:    [clinical, selected radiomics] -> PCA(pca_dim)
  resample:  Borderline-SMOTE (or none/smote/adasyn) on the PCA training matrix
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .data import Cohort, assert_no_leakage


@dataclass
class FoldData:
    X_train: np.ndarray           # PCA features, training fold (not resampled)
    y_train: np.ndarray
    X_fit: np.ndarray             # PCA features after resampling (what models train on)
    y_fit: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    meta: dict = field(default_factory=dict)


def _resampler(name: str, k: int, seed: int):
    from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
    if name == "smote":
        return SMOTE(k_neighbors=k, random_state=seed)
    if name == "borderline":
        return BorderlineSMOTE(k_neighbors=k, random_state=seed)
    if name == "adasyn":
        return ADASYN(n_neighbors=k, random_state=seed)
    raise ValueError(f"unknown resampler: {name}")


def resample(X: np.ndarray, y: np.ndarray, name: str, k: int, seed: int):
    if name == "none":
        return X, y
    k_eff = min(k, int((y == 1).sum()) - 1)
    if k_eff < 1:
        raise ValueError("too few positives to resample")
    # No silent fallback: a failed resampler is an error, not "no resampling".
    return _resampler(name, k_eff, seed).fit_resample(X, y)


def array_digest(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def build_fold(
    cohort: Cohort,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    pcfg: dict,
    resample_seed: int,
) -> FoldData:
    assert_no_leakage(cohort.radiomics_cols + cohort.clinical_cols)
    R = cohort.matrix(cohort.radiomics_cols)
    C = cohort.matrix(cohort.clinical_cols)
    y = cohort.y
    R_tr, R_te, C_tr, C_te = R[train_idx], R[test_idx], C[train_idx], C[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    keep = np.ones(R.shape[1], dtype=bool)
    if pcfg.get("drop_zero_variance", True):
        keep = R_tr.var(axis=0) > 0
    R_tr, R_te = R_tr[:, keep], R_te[:, keep]
    radio_names = [n for n, k in zip(cohort.radiomics_cols, keep) if k]

    rs = StandardScaler().fit(R_tr)
    R_tr, R_te = rs.transform(R_tr), rs.transform(R_te)
    if pcfg["selector"] != "selectkbest":
        raise ValueError(f"unsupported selector: {pcfg['selector']}")
    k = min(int(pcfg["k"]), R_tr.shape[1])
    sel = SelectKBest(f_classif, k=k).fit(R_tr, y_tr)
    R_tr, R_te = sel.transform(R_tr), sel.transform(R_te)
    selected = [radio_names[i] for i in np.flatnonzero(sel.get_support())]

    imp = SimpleImputer(strategy="median").fit(C_tr)
    cs = StandardScaler().fit(imp.transform(C_tr))
    C_tr, C_te = cs.transform(imp.transform(C_tr)), cs.transform(imp.transform(C_te))

    if pcfg.get("fusion", "early") != "early":
        raise ValueError("only early fusion is supported")
    M_tr = np.column_stack([C_tr, R_tr])
    M_te = np.column_stack([C_te, R_te])
    fused_names = list(cohort.clinical_cols) + selected
    assert_no_leakage(fused_names)

    n_comp = min(int(pcfg["pca_dim"]), M_tr.shape[0] - 1, M_tr.shape[1])
    pca = PCA(n_components=n_comp, svd_solver="full").fit(M_tr)
    X_tr, X_te = pca.transform(M_tr), pca.transform(M_te)

    X_fit, y_fit = resample(
        X_tr, y_tr, pcfg["resampler"], int(pcfg.get("resampler_k_neighbors", 5)),
        resample_seed,
    )
    meta = {
        "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "n_train_pos": int(y_tr.sum()), "n_test_pos": int(y_te.sum()),
        "n_fit": int(len(y_fit)), "n_fit_pos": int(y_fit.sum()),
        "n_radiomics_kept": int(keep.sum()), "selected_radiomics": selected,
        "fused_features": fused_names, "pca_dim": int(n_comp),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "input_digest": array_digest(X_fit, y_fit, X_te),
    }
    return FoldData(X_tr, y_tr, X_fit, y_fit, X_te, y_te, meta)


def clinical_arm_fold(
    cohort: Cohort,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    columns: list[str],
    allow_leak_reference: bool,
) -> FoldData:
    """Clinical-only arms: median impute + standardise the given columns."""
    if not allow_leak_reference:
        assert_no_leakage(columns)
    C = cohort.matrix(columns)
    y = cohort.y
    imp = SimpleImputer(strategy="median").fit(C[train_idx])
    sc = StandardScaler().fit(imp.transform(C[train_idx]))
    X_tr = sc.transform(imp.transform(C[train_idx]))
    X_te = sc.transform(imp.transform(C[test_idx]))
    y_tr, y_te = y[train_idx], y[test_idx]
    meta = {
        "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "n_train_pos": int(y_tr.sum()), "n_test_pos": int(y_te.sum()),
        "fused_features": list(columns), "leakage_reference": allow_leak_reference,
        "input_digest": array_digest(X_tr, y_tr, X_te),
    }
    return FoldData(X_tr, y_tr, X_tr, y_tr, X_te, y_te, meta)
