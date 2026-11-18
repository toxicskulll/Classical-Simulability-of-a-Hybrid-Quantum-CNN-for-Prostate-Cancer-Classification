"""
Phase 3 supporting analyses (no training):

  1. noise-aware vs noiseless training at the SAME evaluation noise: paired, seed-averaged
     pooled AUC difference with a patient-level bootstrap (2,000 resamples)
  2. what nested tuning selected, per model: inner-CV objective, preprocessing choices,
     quantum circuit depth, study wall time

Writes results/phase3/noise/noise_aware_vs_noiseless.csv and
results/phase3/tuning/tuning_selection_summary.csv.

Usage:
  python scripts/phase3_extras.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

V3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3_DIR))

from v3core.stats import auc_mw, patient_bootstrap_indices  # noqa: E402

R = V3_DIR / "results" / "phase3"


def noise_aware() -> pd.DataFrame:
    oof = pd.concat([pd.read_csv(p) for p in sorted((R / "noise" / "oof").glob("*.csv.gz"))])
    oof = oof[oof.shots == 0]
    rows = []
    for arch, base in (("8q3L", "phase1_8q3L"), ("6q1L", "6q1L")):
        for p2 in (1e-3, 1e-2):
            a = oof[(oof.source == f"{arch}_trainp2_{p2:g}") & np.isclose(oof.p2, p2)]
            b = oof[(oof.source == base) & np.isclose(oof.p2, p2)]
            wa = a.pivot_table(index=["ProxID", "fid", "label"], columns="seed", values="prob")
            wb = b.pivot_table(index=["ProxID", "fid", "label"], columns="seed", values="prob").loc[wa.index]
            y = wa.index.get_level_values("label").to_numpy()
            g = wa.index.get_level_values("ProxID").to_numpy()
            A, B = wa.to_numpy(), wb.to_numpy()
            d_full = np.mean([auc_mw(y, A[:, j]) - auc_mw(y, B[:, j]) for j in range(A.shape[1])])
            ds = [np.mean([auc_mw(y[i], A[i, j]) - auc_mw(y[i], B[i, j]) for j in range(A.shape[1])])
                  for i in patient_bootstrap_indices(g, 2000, 0) if y[i].min() != y[i].max()]
            rows.append({"arch": arch, "eval_p2": p2,
                         "auc_noise_aware": np.mean([auc_mw(y, A[:, j]) for j in range(A.shape[1])]),
                         "auc_noiseless_trained": np.mean([auc_mw(y, B[:, j]) for j in range(B.shape[1])]),
                         "delta": d_full, "ci95_lo": np.percentile(ds, 2.5), "ci95_hi": np.percentile(ds, 97.5)})
    df = pd.DataFrame(rows)
    df.to_csv(R / "noise" / "noise_aware_vs_noiseless.csv", index=False)
    return df


def tuning_summary() -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(R / "tuning" / "*" / "seed*_outer*.json")):
        j = json.loads(Path(f).read_text())
        rows.append({"model": j["model"], "seed": j["seed"], "outer_fold": j["outer_fold"],
                     "inner_auc": j["inner_mean_auc"], "n_complete": j["n_complete"],
                     "n_pruned": j["n_pruned"], **{f"p_{k}": v for k, v in j["params"].items()}})
    d = pd.DataFrame(rows)
    d.to_csv(R / "tuning" / "tuning_selections.csv", index=False)
    out = []
    for m, g in d.groupby("model"):
        row = {"model": m, "n_studies": len(g), "inner_auc_mean": g.inner_auc.mean(),
               "inner_auc_min": g.inner_auc.min(), "inner_auc_max": g.inner_auc.max(),
               "pruned_frac": g.n_pruned.sum() / (g.n_complete.sum() + g.n_pruned.sum())}
        for p in ("p_pca_dim", "p_k", "p_resampler", "p_conv_layers", "p_n_epochs"):
            if p in g and g[p].notna().any():
                row[p.replace("p_", "sel_")] = "; ".join(f"{k}:{v}" for k, v in g[p].value_counts().sort_index().items())
        out.append(row)
    s = pd.DataFrame(out)
    s.to_csv(R / "tuning" / "tuning_selection_summary.csv", index=False)
    return s


def main() -> None:
    pd.set_option("display.width", 250)
    print(noise_aware().round(4).to_string(index=False))
    print()
    print(tuning_summary().round(3).to_string(index=False))


if __name__ == "__main__":
    main()
