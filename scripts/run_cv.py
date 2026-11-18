"""
Run patient-grouped CV for the v3 models.

Each (model, seed, fold) is an independent task; finished tasks are skipped unless
--force, so an interrupted run resumes where it stopped.

Usage:
  python scripts/run_cv.py                        # everything in the config
  python scripts/run_cv.py --models hqcnn --n-jobs 5
  python scripts/run_cv.py --seeds 0 --folds 0 --results-dir results/smoke
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

V3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3_DIR))

# One BLAS/OpenMP thread per worker; parallelism comes from the process pool.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from v3core.config import DEFAULT_CONFIG, RESULTS_DIR, load_config, v3_path  # noqa: E402
from v3core.cv import run_task  # noqa: E402


def _init_worker(v3_dir: str) -> None:
    if v3_dir not in sys.path:
        sys.path.insert(0, v3_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--folds", nargs="+", type=int, default=None)
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    models = args.models or cfg["models"]
    seeds = args.seeds if args.seeds is not None else cfg["cv"]["seeds"]
    folds = args.folds if args.folds is not None else list(range(cfg["cv"]["n_splits"]))
    results_dir = str(v3_path(args.results_dir))

    # HQCNN tasks first: they dominate runtime.
    tasks = [(m, s, f) for m in models for s in seeds for f in folds]
    tasks.sort(key=lambda t: t[0] != "hqcnn")
    print(f"  {len(tasks)} tasks: models={models} seeds={seeds} folds={folds} "
          f"n_jobs={args.n_jobs} -> {results_dir}", flush=True)

    t0 = time.perf_counter()

    def report(r: dict) -> None:
        if r.get("skipped"):
            print(f"    skip {r['model']:<24} seed={r['seed']} fold={r['fold']} (exists)", flush=True)
        else:
            print(f"    done {r['model']:<24} seed={r['seed']} fold={r['fold']} "
                  f"AUC={r['AUC']:.3f} F1={r['F1']:.3f} ({r['wall_seconds']:.1f}s, "
                  f"elapsed {time.perf_counter() - t0:.0f}s)", flush=True)

    if args.n_jobs == 1:
        for m, s, f in tasks:
            report(run_task(args.config, m, s, f, results_dir, args.force))
    else:
        with ProcessPoolExecutor(max_workers=args.n_jobs, initializer=_init_worker,
                                 initargs=(str(V3_DIR),)) as ex:
            futs = [ex.submit(run_task, args.config, m, s, f, results_dir, args.force)
                    for m, s, f in tasks]
            for fut in as_completed(futs):
                report(fut.result())
    print(f"  finished in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
