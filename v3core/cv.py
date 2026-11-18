"""
One CV task = (model, seed, fold). Writes:

  <results>/oof/<model>/seed<S>_fold<F>.csv.gz     per-lesion OOF probabilities
  <results>/oof/<model>/seed<S>_fold<F>.meta.json  fold/preprocessing/model metadata

Tasks are independent and deterministic, so they can run in any order or in parallel.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .data import load_cohort, make_folds
from .metrics import fold_metrics, pick_threshold
from .models import fit_model
from .preprocess import build_fold, clinical_arm_fold
from .seeding import component_seed, seed_everything


def oof_path(results_dir: Path, model: str, seed: int, fold: int) -> Path:
    return Path(results_dir) / "oof" / model / f"seed{seed}_fold{fold}.csv.gz"


def fold_data_for(model: str, cohort, tr, te, cfg: dict, seed: int, fold: int):
    arms = cfg["clinical_arms"]["arms"]
    if model in arms:
        return clinical_arm_fold(
            cohort, tr, te, list(arms[model]),
            allow_leak_reference=model in cfg.get("leakage_reference_models", []),
        )
    # Same resample seed for every model on this (seed, fold): identical inputs.
    return build_fold(
        cohort, tr, te, cfg["preprocessing"],
        resample_seed=component_seed(seed, fold, "resample"),
    )


def run_task(config_path: str, model: str, seed: int, fold: int,
             results_dir: str, force: bool = False) -> dict:
    out = oof_path(Path(results_dir), model, seed, fold)
    meta_path = out.with_name(out.name.replace(".csv.gz", ".meta.json"))
    if out.exists() and meta_path.exists() and not force:
        return {"model": model, "seed": seed, "fold": fold, "skipped": True}

    seed_everything(component_seed(seed, fold, f"global:{model}"))
    cfg = load_config(config_path)
    cohort = load_cohort(cfg)
    folds = make_folds(cohort, cfg["cv"]["n_splits"], seed)
    tr, te = folds[fold]

    fitter_name = model
    tuned = cfg.get("tuned_models", {})
    if model in tuned:                      # Phase 3: per-fold config chosen by nested tuning
        from .tuning import FITTER, tuned_cfg
        cfg = tuned_cfg(cfg, tuned[model], seed, fold)
        fitter_name = FITTER[tuned[model]]

    t0 = time.perf_counter()
    fd = fold_data_for(fitter_name, cohort, tr, te, cfg, seed, fold)
    res = fit_model(fitter_name, fd, cfg, seed, fold)
    thr = pick_threshold(res["threshold_strategy"], fd.y_train, res["prob_train"])
    pred = (res["prob_test"] >= thr).astype(int)
    wall = time.perf_counter() - t0

    lesions = cohort.frame.iloc[te]
    df = pd.DataFrame({
        "ProxID": lesions["ProxID"].to_numpy(),
        "fid": lesions["fid"].to_numpy(),
        "label": fd.y_test,
        "prob": res["prob_test"],
        "pred": pred,
        "threshold": thr,
        "model": model,
        "seed": seed,
        "fold": fold,
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, compression="gzip", float_format="%.10g")

    metrics = fold_metrics(fd.y_test, res["prob_test"], pred)
    meta = {
        "model": model, "seed": seed, "fold": fold, "config": cfg["name"],
        "config_path": str(config_path), "threshold": thr,
        "wall_seconds": wall, "metrics": metrics,
        "preprocessing": fd.meta, "model_meta": res.get("meta", {}),
        "python": platform.python_version(),
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=_json_default))
    return {"model": model, "seed": seed, "fold": fold, "skipped": False,
            "wall_seconds": wall, **metrics}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
