"""
Phase 3 step 1: nested hyperparameter tuning (seeds 0-4, 5 outer folds each, inner 3
grouped folds).

One Optuna study per (model, seed, outer fold), stored in SQLite so an interrupted run resumes.
Studies run in parallel across processes.

Usage:
  python scripts/tune_nested.py --verify            # twin HQCNN == PennyLane, 1 config
  python scripts/tune_nested.py --time-trial hqcnn  # time one trial
  python scripts/tune_nested.py --n-jobs 7          # all models x 5 seeds x 5 outer folds
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

V3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3_DIR))

CONFIG = V3_DIR / "configs" / "phase2.yaml"      # champion + arm definitions


def run_study(model: str, seed: int, outer_fold: int, n_trials: int) -> dict:
    sys.path.insert(0, str(V3_DIR))
    import optuna
    from v3core.config import load_config
    from v3core.data import load_cohort, make_folds
    from v3core.seeding import component_seed
    from v3core.tuning import TUNING_DIR, best_path, make_objective

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cfg = load_config(CONFIG)
    cohort = load_cohort(cfg)
    tr, _ = make_folds(cohort, cfg["cv"]["n_splits"], seed)[outer_fold]
    out = TUNING_DIR / model
    out.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=f"{model}_seed{seed}_outer{outer_fold}", direction="maximize",
        storage=f"sqlite:///{(out / f'seed{seed}_outer{outer_fold}.db').as_posix()}", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=component_seed(seed, outer_fold, f"tpe:{model}")),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    done = [t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)]
    t0 = time.perf_counter()
    if len(done) < n_trials:
        study.optimize(make_objective(cohort, cfg, model, tr, seed, outer_fold), n_trials=n_trials - len(done))
    states = [t.state.name for t in study.trials]
    best = study.best_trial
    payload = {"model": model, "outer_fold": outer_fold, "seed": seed, "params": best.params,
               "inner_mean_auc": best.value, "inner_aucs": best.user_attrs.get("inner_aucs"),
               "n_trials": len(study.trials), "n_complete": states.count("COMPLETE"),
               "n_pruned": states.count("PRUNED"), "seconds_this_run": time.perf_counter() - t0}
    best_path(model, seed, outer_fold).write_text(json.dumps(payload, indent=2))
    study.trials_dataframe().to_csv(out / f"seed{seed}_outer{outer_fold}_trials.csv", index=False)
    return payload


def verify() -> None:
    """HQCNN trained through the twin == PennyLane HQCNN on a non-champion config."""
    import numpy as np
    from v3core.config import load_config
    from v3core.data import load_cohort, make_folds
    from v3core.models import REGISTRY
    from v3core.preprocess import build_fold
    from v3core.seeding import component_seed
    from v3core.tuning import apply_params
    cfg = load_config(CONFIG)
    p = {"k": 16, "pca_dim": 16, "resampler": "smote", "lr": 0.04, "n_epochs": 20,
         "batch_size": 32, "focal_gamma": 1.5, "conv_layers": 1}
    c = apply_params(cfg, "hqcnn", p)
    cohort = load_cohort(c)
    tr, te = make_folds(cohort, 5, 0)[0]
    fd = build_fold(cohort, tr, te, c["preprocessing"], component_seed(0, 0, "resample"))
    t0 = time.perf_counter(); a = REGISTRY["hqcnn"](fd, c, 0, 0); t1 = time.perf_counter()        # noqa: E702
    b = REGISTRY["hqcnn_twin"](fd, c, 0, 0); t2 = time.perf_counter()                               # noqa: E702
    d = float(np.max(np.abs(np.asarray(a["prob_test"]) - np.asarray(b["prob_test"]))))
    print(f"  twin vs PennyLane HQCNN (pca16, 1 layer, 20 epochs): max|dp| = {d:.1e}; "
          f"PennyLane {t1 - t0:.1f}s, twin {t2 - t1:.1f}s")
    out = V3_DIR / "results" / "phase3" / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    (out / "twin_verify.json").write_text(json.dumps({"config": p, "max_abs_dprob": d,
                                                       "seconds_pennylane": t1 - t0, "seconds_twin": t2 - t1}, indent=2))
    assert d < 1e-8


def main() -> None:
    from v3core.tuning import TUNED_MODELS
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(TUNED_MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--outer-folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--n-jobs", type=int, default=5)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--time-trial", default=None, help="time ONE trial of this model (outer fold 0)")
    args = ap.parse_args()
    if args.verify:
        verify()
        return
    if args.time_trial:
        import optuna
        from v3core.config import load_config
        from v3core.data import load_cohort, make_folds
        from v3core.tuning import make_objective
        cfg = load_config(CONFIG)
        cohort = load_cohort(cfg)
        tr, _ = make_folds(cohort, 5, 0)[0]
        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
        t0 = time.perf_counter()
        st.optimize(make_objective(cohort, cfg, args.time_trial, tr, 0, 0), n_trials=1)
        print(f"  one trial of {args.time_trial}: {time.perf_counter() - t0:.1f}s  params={st.trials[0].params} "
              f"inner AUC={st.trials[0].value:.3f}")
        return
    # long studies first
    order = {"hqcnn": 0, "dephased_control": 1}
    tasks = sorted([(m, s, k) for m in args.models for s in args.seeds for k in args.outer_folds],
                   key=lambda t: order.get(t[0], 9))
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
        futs = {ex.submit(run_study, m, s, k, args.trials): (m, s, k) for m, s, k in tasks}
        for f in as_completed(futs):
            r = f.result()
            print(f"    {r['model']:<20} seed{r['seed']} outer{r['outer_fold']}  best inner AUC={r['inner_mean_auc']:.3f} "
                  f"({r['n_complete']} complete, {r['n_pruned']} pruned, {r['seconds_this_run']:.0f}s)  "
                  f"elapsed {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
