"""
Phase 2A: plug trained Phase 1 HQCNN weights into the classical twin.

Phase 1 did not store weights, but training is deterministic, so for every
(seed, fold) this script:
  1. refits the HQCNN exactly as run_cv.py did (same fold data, same Generator);
  2. checks the refit reproduces the saved Phase 1 OOF probabilities;
  3. saves the weights + angle scaler to results/phase2/twin/trained_params/;
  4. evaluates the classical twin with those weights on the test fold and checks
     the twin's OOF probabilities equal Phase 1's.

Writes results/phase2/twin/trained_check.csv.

Usage:
  python scripts/twin_trained_check.py --n-jobs 5
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

V3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3_DIR))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

OUT = V3_DIR / "results" / "phase2" / "twin"


def check_task(seed: int, fold: int) -> dict:
    import numpy as np
    import pandas as pd
    from pennylane import numpy as pnp

    sys.path.insert(0, str(V3_DIR))
    from v3core.config import load_config
    from v3core.cv import fold_data_for, oof_path
    from v3core.data import load_cohort, make_folds
    from v3core.models import fit_model
    from v3core.seeding import component_seed, seed_everything
    from v3core.twin import twin_expval

    cfg = load_config()
    h = cfg["hqcnn"]
    seed_everything(component_seed(seed, fold, "global:hqcnn"))
    cohort = load_cohort(cfg)
    tr, te = make_folds(cohort, cfg["cv"]["n_splits"], seed)[fold]
    fd = fold_data_for("hqcnn", cohort, tr, te, cfg, seed, fold)
    res = fit_model("hqcnn", fd, cfg, seed, fold)
    meta = res["meta"]

    phase1 = pd.read_csv(oof_path(V3_DIR / "results", "hqcnn", seed, fold))
    assert np.array_equal(phase1["label"].to_numpy(), fd.y_test)
    p1 = phase1["prob"].to_numpy()

    params = np.asarray(meta["params"])
    lo, hi = np.asarray(meta["scaler_lo"]), np.asarray(meta["scaler_hi"])
    a, b = h["angle_range"]
    span = np.where(hi > lo, hi - lo, 1.0)
    Xs = a + np.clip((fd.X_test - lo) / span, 0.0, 1.0) * (b - a)
    n_q = meta["n_quantum_params"]
    z = np.asarray(twin_expval(pnp.array(Xs, requires_grad=False), pnp.array(params[:n_q]),
                               int(h["n_qubits"]), int(h["conv_layers"]), list(h["reupload_gates"]),
                               h["pooling_gate"], h["pooling_survivor"]))
    p_twin = 1.0 / (1.0 + np.exp(-(params[-2] * z + params[-1])))

    OUT.joinpath("trained_params").mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "trained_params" / f"seed{seed}_fold{fold}.npz",
             params=params, scaler_lo=lo, scaler_hi=hi)
    rel = lambda x, y: float(np.max(np.abs(x - y) / np.maximum(np.abs(y), 1e-12)))  # noqa: E731
    return {
        "seed": seed, "fold": fold, "n_test": len(p1),
        "refit_vs_phase1_max_abs": float(np.max(np.abs(res["prob_test"] - p1))),
        "twin_vs_refit_max_abs": float(np.max(np.abs(p_twin - res["prob_test"]))),
        "twin_vs_phase1_max_abs": float(np.max(np.abs(p_twin - p1))),
        "twin_vs_phase1_max_rel": rel(p_twin, p1),
        "fit_seconds": meta["fit_seconds"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=5)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    import pandas as pd

    tasks = [(s, f) for s in args.seeds for f in args.folds]
    rows = []
    with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
        futs = [ex.submit(check_task, s, f) for s, f in tasks]
        for fut in as_completed(futs):
            r = fut.result()
            rows.append(r)
            print(f"    seed={r['seed']} fold={r['fold']}  refit-vs-P1 {r['refit_vs_phase1_max_abs']:.1e}  "
                  f"twin-vs-refit {r['twin_vs_refit_max_abs']:.1e}  "
                  f"twin-vs-P1 {r['twin_vs_phase1_max_abs']:.1e}", flush=True)
    df = pd.DataFrame(rows).sort_values(["seed", "fold"])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "trained_check.csv", index=False)
    print(f"  max |twin - Phase1 OOF| over {len(df)} folds, {df['n_test'].sum()} predictions: "
          f"{df['twin_vs_phase1_max_abs'].max():.2e} (Phase 1 OOF stored with 10 significant digits)")
    print(f"  max |refit - Phase1 OOF|: {df['refit_vs_phase1_max_abs'].max():.2e}")


if __name__ == "__main__":
    main()
