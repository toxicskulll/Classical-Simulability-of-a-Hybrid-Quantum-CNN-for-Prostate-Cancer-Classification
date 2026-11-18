"""
Phase 3 step 3: PI-CAI external evaluation under the v3 protocol.

  * features: the 264 radiomics columns shared by both cohorts (T2W, ADC, DWI; PI-CAI has
    no K^trans) + pre-biopsy clinical (PSA, PSAD, prostate volume, age). No Gleason.
  * train on PROSTATEx only (327 lesions, 204 patients), champion preprocessing and
    hyperparameters, fitted on PROSTATEx.
  * threshold: F1-optimal on PROSTATEx pooled OOF probabilities (patient-grouped 5-fold CV,
    same seed), never on PI-CAI.
  * evaluate ONCE on PI-CAI (1,500 cases, 1,476 patients); 95% CIs from a bootstrap over
    PI-CAI patient_id. No PI-CAI data is used for any fitting or selection.
  * models: HQCNN (exact twin), LR (Phase 1 baseline), classical_tree_net (Phase 2 recipe);
    seeds 0-4 (each seed: its own PROSTATEx folds, resampling and model init).

Known mismatches, reported with the results (not fixable here):
  * ROI: PROSTATEx features describe a LESION (mask or sphere around the finding); the PI-CAI
    cache describes the WHOLE GLAND (one row per case).
  * label: PROSTATEx lesion ClinSig vs PI-CAI case-level csPCa (ISUP >= 2).
  * DWI: PROSTATEx DWI series vs PI-CAI high-b-value (hbv) images.
A domain classifier (PROSTATEx vs PI-CAI, patient-grouped CV) quantifies the shift.

Usage:
  python scripts/picai_transfer.py --n-jobs 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as onp  # noqa: E402
import pandas as pd  # noqa: E402

V3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3_DIR))

REPO = V3_DIR.parents[1]
PICAI_FEATS = V3_DIR / "data" / "picai" / "features_cache.csv"
PICAI_LABELS = V3_DIR / "data" / "picai" / "labels_cache.csv"
OUT = V3_DIR / "results" / "phase3" / "picai"
CONFIG = V3_DIR / "configs" / "phase2.yaml"
MODELS = {"hqcnn": "hqcnn_twin", "lr": "lr", "classical_tree_net": "classical_tree_net"}
SEEDS = [0, 1, 2, 3, 4]
FINAL = 99            # "fold" index used for the final all-PROSTATEx fit
MODS = ("t2w", "adc", "dwi")
CLIN = {"clinical_psa": "clin_psa", "clinical_psad": "clin_psad",
        "clinical_prostate_volume": "clin_prostate_volume", "clinical_patient_age": "clin_patient_age"}


def build_cohorts():
    """(prostatex_cohort, combined_cohort, n_prostatex): shared columns only."""
    from v3core.config import load_config
    from v3core.data import Cohort, load_cohort
    cfg = load_config(CONFIG)
    px = load_cohort(cfg)
    pi = pd.read_csv(PICAI_FEATS).rename(columns=CLIN)
    lab = pd.read_csv(PICAI_LABELS)
    pi = pi.merge(lab, on=["ProxID", "fid"], how="inner", validate="one_to_one")
    shared = [c for c in px.radiomics_cols if c.split("_", 1)[0] in MODS]
    missing = [c for c in shared if c not in pi.columns]
    assert not missing, missing[:5]
    clin = list(px.clinical_cols)
    pi["patient_id"] = pi["ProxID"].astype(str).str.split("_").str[0]
    px_frame = px.frame[["ProxID", "fid", "label"] + clin + shared].copy()
    px_frame["cohort"], px_frame["patient_id"] = "PROSTATEx", px_frame["ProxID"].astype(str)
    pi_frame = pi[["ProxID", "fid", "label"] + clin + shared].copy()
    pi_frame["cohort"], pi_frame["patient_id"] = "PI-CAI", pi["patient_id"].to_numpy()
    both = pd.concat([px_frame, pi_frame], ignore_index=True)
    mk = lambda f: Cohort(frame=f.reset_index(drop=True), y=f["label"].to_numpy(int),  # noqa: E731
                          groups=f["patient_id"].astype(str).to_numpy(), radiomics_cols=shared,
                          clinical_cols=clin)
    return cfg, mk(px_frame), mk(both), len(px_frame)


def task(model: str, seed: int, fold: int) -> dict:
    """fold 0-4: PROSTATEx OOF fold; fold FINAL: train on all PROSTATEx, predict PI-CAI."""
    sys.path.insert(0, str(V3_DIR))
    from v3core.data import make_folds
    from v3core.models import REGISTRY
    from v3core.preprocess import build_fold
    from v3core.seeding import component_seed, seed_everything
    cfg, px, both, n_px = build_cohorts()
    seed_everything(component_seed(seed, fold, f"picai:{model}"))
    if fold == FINAL:
        tr, te, cohort = onp.arange(n_px), onp.arange(n_px, len(both.y)), both
    else:
        tr, te = make_folds(px, cfg["cv"]["n_splits"], seed)[fold]
        cohort = px
    fd = build_fold(cohort, tr, te, cfg["preprocessing"], component_seed(seed, fold, "picai:resample"))
    t0 = time.perf_counter()
    res = REGISTRY[MODELS[model]](fd, cfg, seed, 500 + fold)
    fr = cohort.frame.iloc[te]
    return {"model": model, "seed": seed, "fold": fold, "seconds": time.perf_counter() - t0,
            "pred": pd.DataFrame({"ProxID": fr["ProxID"].to_numpy(), "fid": fr["fid"].to_numpy(),
                                  "patient_id": fr["patient_id"].to_numpy(), "cohort": fr["cohort"].to_numpy(),
                                  "label": fd.y_test, "prob": onp.asarray(res["prob_test"]),
                                  "model": model, "seed": seed, "fold": fold})}


def domain_shift(both) -> dict:
    """How separable are the two cohorts' feature distributions?"""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cols = both.radiomics_cols + both.clinical_cols
    X = both.frame[cols].to_numpy(float)
    d = (both.frame["cohort"] == "PI-CAI").to_numpy(int)
    oof = onp.zeros(len(d))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, d, both.groups):
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(C=0.1, max_iter=5000))
        m.fit(X[tr], d[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    a, b = X[d == 0], X[d == 1]
    smd = onp.abs(onp.nanmean(a, 0) - onp.nanmean(b, 0)) / onp.sqrt((onp.nanvar(a, 0) + onp.nanvar(b, 0)) / 2 + 1e-12)
    return {"domain_classifier_auc": float(roc_auc_score(d, oof)),
            "median_abs_smd": float(onp.nanmedian(smd)), "frac_features_abs_smd_gt_0.5": float(onp.nanmean(smd > 0.5)),
            "n_features": len(cols), "prevalence_prostatex": float(both.y[d == 0].mean()),
            "prevalence_picai": float(both.y[d == 1].mean())}


def summarise(preds: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    from v3core.metrics import f1_threshold, fold_metrics
    from v3core.stats import auc_mw, patient_bootstrap_indices
    rows, thr = [], {}
    for (m, s), g in preds[preds.fold != FINAL].groupby(["model", "seed"]):
        thr[(m, s)] = f1_threshold(g["label"].to_numpy(), g["prob"].to_numpy())
    ext = preds[preds.fold == FINAL].copy()
    ext["thr"] = [thr[(m, s)] for m, s in zip(ext.model, ext.seed)]
    ext["pred"] = (ext["prob"] >= ext["thr"]).astype(int)
    wide = ext.pivot_table(index=["ProxID", "patient_id", "label"], columns=["model", "seed"], values="prob").reset_index()
    y, groups = wide["label"].to_numpy(), wide["patient_id"].to_numpy()
    boots = list(patient_bootstrap_indices(groups, n_boot, 0))
    for m in MODELS:
        px_oof = preds[(preds.model == m) & (preds.fold != FINAL)]
        px_auc = [auc_mw(g["label"].to_numpy(), g["prob"].to_numpy()) for _, g in px_oof.groupby("seed")]
        per_seed = [fold_metrics(g["label"].to_numpy(), g["prob"].to_numpy(), g["pred"].to_numpy())
                    for _, g in ext[ext.model == m].groupby("seed")]
        S = onp.column_stack([wide[(m, s)].to_numpy() for s in SEEDS])
        dist = [onp.mean([auc_mw(y[i], S[i, j]) for j in range(S.shape[1])]) for i in boots if y[i].min() != y[i].max()]
        row = {"model": m, "prostatex_oof_auc_mean": float(onp.mean(px_auc)), "prostatex_oof_auc_std": float(onp.std(px_auc, ddof=1)),
               "threshold_mean": float(onp.mean([thr[(m, s)] for s in SEEDS])),
               "picai_auc_ci95_lo": float(onp.percentile(dist, 2.5)), "picai_auc_ci95_hi": float(onp.percentile(dist, 97.5))}
        for k in ("AUC", "Recall", "Precision", "F1", "Specificity", "BalancedAccuracy"):
            v = [p[k] for p in per_seed]
            row[f"picai_{k}_mean"], row[f"picai_{k}_std"] = float(onp.mean(v)), float(onp.std(v, ddof=1))
        rows.append(row)
    out = pd.DataFrame(rows)
    # paired HQCNN - comparator on PI-CAI (seed-averaged, patient bootstrap)
    H = onp.column_stack([wide[("hqcnn", s)].to_numpy() for s in SEEDS])
    for m in ("lr", "classical_tree_net"):
        C = onp.column_stack([wide[(m, s)].to_numpy() for s in SEEDS])
        d = [onp.mean([auc_mw(y[i], H[i, j]) - auc_mw(y[i], C[i, j]) for j in range(5)])
             for i in boots if y[i].min() != y[i].max()]
        full = onp.mean([auc_mw(y, H[:, j]) - auc_mw(y, C[:, j]) for j in range(5)])
        out.loc[out.model == m, "delta_hqcnn_minus_model"] = full
        out.loc[out.model == m, "delta_ci95_lo"] = float(onp.percentile(d, 2.5))
        out.loc[out.model == m, "delta_ci95_hi"] = float(onp.percentile(d, 97.5))
    ext.to_csv(OUT / "picai_predictions.csv.gz", index=False, compression="gzip", float_format="%.10g")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=5)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg, px, both, n_px = build_cohorts()
    print(f"  PROSTATEx {n_px} lesions / PI-CAI {len(both.y) - n_px} cases; "
          f"{len(px.radiomics_cols)} shared radiomics + {len(px.clinical_cols)} clinical")
    ds = domain_shift(both)
    print(f"  domain shift: classifier AUC {ds['domain_classifier_auc']:.3f}, median |SMD| {ds['median_abs_smd']:.2f}")
    tasks = [(m, s, f) for m in MODELS for s in SEEDS for f in list(range(5)) + [FINAL]]
    tasks.sort(key=lambda t: t[0] != "hqcnn")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
        res = list(ex.map(task, *zip(*tasks)))
    preds = pd.concat([r["pred"] for r in res], ignore_index=True)
    preds.to_csv(OUT / "all_predictions.csv.gz", index=False, compression="gzip", float_format="%.10g")
    summ = summarise(preds)
    summ.to_csv(OUT / "picai_summary.csv", index=False)
    meta = {"domain_shift": ds, "seconds": time.perf_counter() - t0,
            "mismatches": ["ROI: PROSTATEx lesion vs PI-CAI whole gland",
                           "label: lesion ClinSig vs case csPCa (ISUP>=2)",
                           "DWI: PROSTATEx DWI vs PI-CAI high-b (hbv)"],
            "threshold_source": "F1-optimal on PROSTATEx pooled OOF, per seed",
            "picai_used_for_selection": False}
    (OUT / "picai_meta.json").write_text(json.dumps(meta, indent=2))
    from v3core.phase2 import write_tex_table
    names = {"hqcnn": "HQCNN (8q)", "lr": "Logistic regression", "classical_tree_net": "Classical tree net"}
    write_tex_table(
        OUT / "picai.tex",
        "External evaluation on PI-CAI (1,500 cases, 1,476 patients) of models trained only on PROSTATEx "
        "(shared T2W/ADC/DWI radiomics + pre-biopsy clinical). Threshold: F1-optimal on PROSTATEx OOF. "
        "Mean $\\pm$ SD over 5 seeds; AUC 95\\% CI from a patient-level bootstrap. Note: PROSTATEx features are "
        "lesion-level and PI-CAI features whole-gland.",
        "tab:v3_picai",
        ["Model", "PROSTATEx OOF AUC", "PI-CAI AUC", "95\\% CI", "Recall", "Precision", "F1", "Specificity"],
        [[names[r.model], f"${r.prostatex_oof_auc_mean:.3f} \\pm {r.prostatex_oof_auc_std:.3f}$",
          f"${r.picai_AUC_mean:.3f} \\pm {r.picai_AUC_std:.3f}$", f"[{r.picai_auc_ci95_lo:.3f}, {r.picai_auc_ci95_hi:.3f}]",
          f"${r.picai_Recall_mean:.3f}$", f"${r.picai_Precision_mean:.3f}$", f"${r.picai_F1_mean:.3f}$",
          f"${r.picai_Specificity_mean:.3f}$"] for r in summ.itertuples()],
    )
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False))
    print(f"  done in {time.perf_counter() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
