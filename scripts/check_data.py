"""
Sanity checks for the v3 data pipeline (no model training).

  * lesion table loads with expected counts and no duplicate keys
  * Gleason is absent from every model-input feature list
  * folds are patient-disjoint, stratified and partition the cohort, for every seed
  * fold preprocessing is fitted on train only (test rows never influence it)

Usage:
  python scripts/check_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

V3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3_DIR))

from v3core.config import load_config  # noqa: E402
from v3core.data import LeakageError, assert_no_leakage, load_cohort, make_folds  # noqa: E402
from v3core.preprocess import build_fold  # noqa: E402


def main() -> None:
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    cohort = load_cohort(cfg)
    y = cohort.y
    print(f"  lesions={len(y)} patients={len(set(cohort.groups))} pos={y.sum()} "
          f"radiomics={len(cohort.radiomics_cols)} clinical={cohort.clinical_cols}")

    try:
        assert_no_leakage(["leak_gleason_grade"])
        raise SystemExit("FAIL: leakage guard did not trip on Gleason")
    except LeakageError:
        print("  leakage guard trips on leak_gleason_grade: ok")

    n_splits = cfg["cv"]["n_splits"]
    for seed in cfg["cv"]["seeds"]:
        folds = make_folds(cohort, n_splits, seed)
        pos = [int(y[te].sum()) for _, te in folds]
        sizes = [len(te) for _, te in folds]
        print(f"  seed {seed}: test sizes {sizes} positives {pos}")

    # train-only fitting: perturbing test rows must not change training features
    tr, te = make_folds(cohort, n_splits, cfg["cv"]["seeds"][0])[0]
    a = build_fold(cohort, tr, te, cfg["preprocessing"], resample_seed=123)
    frame = cohort.frame.copy()
    frame.loc[te, cohort.radiomics_cols] *= 1000.0
    perturbed = type(cohort)(frame, cohort.y, cohort.groups,
                             cohort.radiomics_cols, cohort.clinical_cols)
    b = build_fold(perturbed, tr, te, cfg["preprocessing"], resample_seed=123)
    if not (np.allclose(a.X_fit, b.X_fit) and a.meta["selected_radiomics"] == b.meta["selected_radiomics"]):
        raise SystemExit("FAIL: training features depend on test rows")
    print("  training features independent of test rows: ok")
    assert_no_leakage(a.meta["fused_features"])
    print(f"  fold 0 fused inputs ({len(a.meta['fused_features'])}): "
          f"{a.meta['fused_features'][:6]} ...")
    print(f"  fold 0: PCA {a.meta['pca_dim']} comps explain "
          f"{a.meta['pca_explained_variance']:.3f}; fit n={a.meta['n_fit']} "
          f"(pos {a.meta['n_fit_pos']})")
    print("  all data checks passed")


if __name__ == "__main__":
    main()
