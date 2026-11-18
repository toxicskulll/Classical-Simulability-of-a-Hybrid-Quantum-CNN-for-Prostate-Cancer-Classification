"""
PI-CAI evaluation from the saved probabilities: all cases, and cases that do not come from PROSTATEx.

Single source: results/phase3/picai/all_predictions.csv.gz, the probabilities written by
scripts/picai_transfer.py (10 significant digits). It holds the PROSTATEx out-of-fold predictions (folds
0-4) and the PI-CAI predictions of the model refitted on all of PROSTATEx (fold 99). Every number
in the PI-CAI tables of the paper (main Table VI, supplementary Table S17) comes from this script:

  * decision threshold per (model, seed): F1-optimal on the PROSTATEx out-of-fold probabilities;
  * AUC: mean over seeds; 95% CI from a patient-level bootstrap (2,000 resamples, seed 0) of the
    seed-averaged AUC; the same resamples give the paired dAUC = HQCNN - model;
  * threshold metrics (recall, precision, F1, specificity, balanced accuracy): mean over seeds.

The PI-CAI public training set contains 328 PROSTATEx cases (PI-CAI
additional_resources/ProstateX-mapping.json, ProstateX-XXXX_<date> -> <patient>_<study>); the
subset "not_from_prostatex" excludes them.

Logistic regression saturates on PI-CAI inputs, so the 10-digit rounding of the saved probabilities
adds ties; its all-case AUC therefore differs from the in-memory value in picai_summary.csv in
the third decimal. The saved probabilities are the single source used in the paper.

Writes results/phase3/picai/picai_summary_by_overlap.csv

Usage:
  python scripts/picai_overlap.py [--mapping path/to/ProstateX-mapping.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

V3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3_DIR))

from v3core.metrics import f1_threshold, fold_metrics  # noqa: E402
from v3core.stats import auc_mw, patient_bootstrap_indices  # noqa: E402

OUT = V3_DIR / "results" / "phase3" / "picai"
DEFAULT_MAPPING = V3_DIR / "data" / "ProstateX-mapping.json"
MODELS = ["hqcnn", "lr", "classical_tree_net"]
FINAL = 99
METRICS = ["Recall", "Precision", "F1", "Specificity", "BalancedAccuracy"]


def summarise(ext: pd.DataFrame, px_auc: dict, subset: str, n_boot: int) -> list[dict]:
    wide = ext.pivot_table(index=["ProxID", "patient_id", "label"], columns=["model", "seed"],
                           values="prob").reset_index()
    y, groups = wide["label"].to_numpy(), wide["patient_id"].to_numpy()
    boots = [i for i in patient_bootstrap_indices(groups, n_boot, 0) if y[i].min() != y[i].max()]
    S = {m: np.column_stack([wide[(m, s)].to_numpy() for s in sorted(s for mm, s in wide.columns if mm == m)])
         for m in MODELS}
    rows = []
    for m in MODELS:
        aucs = [auc_mw(y, S[m][:, j]) for j in range(S[m].shape[1])]
        dist = [np.mean([auc_mw(y[i], S[m][i, j]) for j in range(S[m].shape[1])]) for i in boots]
        per_seed = [fold_metrics(g["label"].to_numpy(), g["prob"].to_numpy(), g["pred"].to_numpy())
                    for _, g in ext[ext.model == m].groupby("seed")]
        row = {"subset": subset, "n_cases": len(wide), "n_patients": int(pd.Series(groups).nunique()),
               "prevalence": float(y.mean()), "model": m,
               "prostatex_oof_auc_mean": float(np.mean(px_auc[m])), "prostatex_oof_auc_std": float(np.std(px_auc[m], ddof=1)),
               "AUC_mean": float(np.mean(aucs)), "AUC_std": float(np.std(aucs, ddof=1)),
               "ci95_lo": float(np.percentile(dist, 2.5)), "ci95_hi": float(np.percentile(dist, 97.5))}
        for k in METRICS:
            row[f"{k}_mean"] = float(np.mean([p[k] for p in per_seed]))
        if m != "hqcnn":
            H, C = S["hqcnn"], S[m]
            k = H.shape[1]
            row["delta_hqcnn_minus_model"] = float(np.mean([auc_mw(y, H[:, j]) - auc_mw(y, C[:, j]) for j in range(k)]))
            d = [np.mean([auc_mw(y[i], H[i, j]) - auc_mw(y[i], C[i, j]) for j in range(k)]) for i in boots]
            row["delta_ci95_lo"], row["delta_ci95_hi"] = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", default=str(DEFAULT_MAPPING))
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    from_prostatex = set(json.loads(Path(args.mapping).read_text()).values())
    preds = pd.read_csv(OUT / "all_predictions.csv.gz")
    preds = preds[preds.model.isin(MODELS)]
    oof = preds[preds.fold != FINAL]
    thr = {(m, s): f1_threshold(g["label"].to_numpy(), g["prob"].to_numpy()) for (m, s), g in oof.groupby(["model", "seed"])}
    px_auc = {m: [auc_mw(g["label"].to_numpy(), g["prob"].to_numpy()) for _, g in oof[oof.model == m].groupby("seed")]
              for m in MODELS}
    ext = preds[preds.fold == FINAL].copy()
    ext["pred"] = (ext["prob"] >= [thr[(m, s)] for m, s in zip(ext.model, ext.seed)]).astype(int)
    n_all = ext["ProxID"].nunique()
    rest = ext[~ext["ProxID"].isin(from_prostatex)]
    print(f"  {len(from_prostatex)} PROSTATEx cases in mapping; {n_all - rest['ProxID'].nunique()} of {n_all} "
          "evaluated PI-CAI cases removed")
    rows = summarise(ext, px_auc, "all_cases", args.n_boot) + summarise(rest, px_auc, "not_from_prostatex", args.n_boot)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "picai_summary_by_overlap.csv", index=False)
    print(df.drop(columns=["n_patients"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
