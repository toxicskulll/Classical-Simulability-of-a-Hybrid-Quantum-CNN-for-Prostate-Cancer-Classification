"""
Paired comparison of every tuned model against the pre-biopsy clinical logistic regression.

Same data, seeds and patient-bootstrap resamples as the Phase 3 summary (summarise.py:
2,000 resamples, bootstrap seed 0), with the reference switched to clin_lr_prebiopsy.
dAUC = model - clinical LR (positive: the model is better), seed-averaged pooled AUC.
Per-seed paired DeLong tests are secondary (they treat lesions as independent).

Reads  results/oof/<model>/seed<S>_fold<F>.csv.gz
Writes results/tables_phase3/vs_clinical.csv            one row per tuned model
       results/tables_phase3/vs_clinical_per_seed.csv   per-seed DeLong

Usage:
  python scripts/compare_vs_clinical.py
"""

from __future__ import annotations

import sys
from pathlib import Path

V3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3_DIR))
sys.path.insert(0, str(V3_DIR / "scripts"))

from summarise import delong_tables, load_oof, wide_oof  # noqa: E402
from v3core.config import RESULTS_DIR, load_config, v3_path  # noqa: E402

REF = "clin_lr_prebiopsy"
MODELS = ["hqcnn_tuned", "dephased_tuned", "tree_net_tuned", "mlp_matched_tuned", "lr_tuned", "svm_tuned", "rf_tuned"]


def main() -> None:
    cfg = load_config(str(V3_DIR / "configs" / "phase3.yaml"))
    out = v3_path(cfg["summary"]["tables_dir"])
    n_boot = int(cfg["summary"]["bootstrap_iterations"])
    oof = load_oof(RESULTS_DIR)
    oof = oof[oof["model"].isin([REF] + MODELS) & oof["seed"].isin(cfg["cv"]["seeds"])].copy()
    w, seeds = wide_oof(oof)
    per_seed, dl = delong_tables(w, seeds, REF, [REF] + MODELS, set(), n_boot, 0)
    # delong_tables reports reference - comparator; flip to model - clinical LR
    for c in ("delta_AUC_mean", "delta_ci95_lo", "delta_ci95_hi", "delong_z_mean"):
        dl[c] = -dl[c]
    dl["delta_ci95_lo"], dl["delta_ci95_hi"] = dl["delta_ci95_hi"].copy(), dl["delta_ci95_lo"].copy()
    dl = dl.rename(columns={"reference": "clinical_reference", "comparator": "model",
                            "AUC_ref_mean": "AUC_clinical_mean", "AUC_comp_mean": "AUC_model_mean",
                            "delta_AUC_mean": "delta_model_minus_clinical"})
    dl.to_csv(out / "vs_clinical.csv", index=False)
    per_seed.to_csv(out / "vs_clinical_per_seed.csv", index=False)
    print(dl[["model", "AUC_model_mean", "AUC_clinical_mean", "delta_model_minus_clinical", "delta_ci95_lo",
              "delta_ci95_hi", "delong_p_median", "seeds_p_lt_005"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
