"""
Summarise v3 CV results into paper tables.

Reads  <results>/oof/<model>/seed<S>_fold<F>.csv.gz
Writes <results>/tables/
  fold_metrics.csv                 one row per model x seed x fold (recomputed from OOF)
  cv_metrics.csv / .tex            mean +/- std across seeds x folds
  pooled_auc.csv / .tex            pooled-OOF AUC per seed, mean +/- std, patient-bootstrap 95% CI
  delong_per_seed.csv              paired DeLong, reference model vs each other model, per seed
  delong.csv / .tex                per-comparison summary across seeds + bootstrap CI of dAUC

Checks before summarising (hard errors):
  * every model has every seed x fold, and each seed's OOF covers each lesion exactly once
  * models sharing radiomics inputs saw identical matrices (input_digest in .meta.json)

Usage:
  python scripts/summarise.py
  python scripts/summarise.py --results-dir results/smoke --allow-incomplete
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

V3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3_DIR))

from v3core.config import DEFAULT_CONFIG, RESULTS_DIR, load_config, v3_path  # noqa: E402
from v3core.metrics import METRICS, fold_metrics  # noqa: E402
from v3core.stats import auc_mw, delong_paired, patient_bootstrap_indices  # noqa: E402

LABELS = {
    "hqcnn": "HQCNN (8q, 3 conv layers)",
    "lr": "Logistic regression",
    "svm_rbf": "RBF-SVM",
    "rf": "Random forest",
    "mlp": "MLP (64, 32)",
    "clin_lr_prebiopsy": "Clinical LR (pre-biopsy)",
    "clin_lr_gleason_leakref": "Clinical LR + Gleason",
    "no_entanglement": "HQCNN, no entanglement",
    "dephased_control": "HQCNN, dephased control",
    "frozen_random": "Frozen random circuit + head",
    "frozen_random_multiZ": r"Frozen random circuit, 8$\langle Z\rangle$ + LR",
    "brickwork_8q": "HQCNN brickwork (8q)",
    "classical_tree_net": "Classical tree net (tanh)",
    "relaxed_bilinear_tree": "Relaxed bilinear tree",
    "mlp_matched": "MLP 12-10-1 (HQCNN recipe)",
    "mlp_matched_sklearn": "MLP 12-10-1 (sklearn)",
    "hqcnn_tuned": "HQCNN (tuned)",
    "dephased_tuned": "HQCNN dephased control (tuned)",
    "tree_net_tuned": "Classical tree net (tuned)",
    "mlp_matched_tuned": r"MLP $\sim$141 params (tuned)",
    "lr_tuned": "Logistic regression (tuned)",
    "svm_tuned": "RBF-SVM (tuned)",
    "rf_tuned": "Random forest (tuned)",
}


def param_counts(results_dir: Path) -> dict[str, int]:
    """n_params per model from the first .meta.json that reports it."""
    out: dict[str, int] = {}
    for p in sorted(glob.glob(str(results_dir / "oof" / "*" / "*.meta.json"))):
        m = json.loads(Path(p).read_text())
        n = m.get("model_meta", {}).get("n_params")
        if n is not None and m["model"] not in out:
            out[m["model"]] = int(n)
    return out


# ── loading and integrity ────────────────────────────────────────────────────
def load_oof(results_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(results_dir / "oof" / "*" / "seed*_fold*.csv.gz")))
    if not files:
        raise SystemExit(f"no OOF files under {results_dir / 'oof'}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def check_integrity(oof: pd.DataFrame, cfg: dict, results_dir: Path, allow_incomplete: bool) -> None:
    n_lesions = cfg["data"]["expected_n_lesions"]
    problems = []
    for (model, seed), g in oof.groupby(["model", "seed"]):
        keys = g["ProxID"] + "/" + g["fid"].astype(str)
        if keys.duplicated().any():
            problems.append(f"{model} seed {seed}: lesion predicted twice")
        if len(g) != n_lesions:
            problems.append(f"{model} seed {seed}: {len(g)}/{n_lesions} lesions")
    expected = {(m, s, f) for m in cfg["models"] for s in cfg["cv"]["seeds"]
                for f in range(cfg["cv"]["n_splits"])}
    have = set(map(tuple, oof[["model", "seed", "fold"]].drop_duplicates().to_numpy().tolist()))
    missing = sorted(expected - have)
    if missing:
        problems.append(f"{len(missing)} missing tasks, e.g. {missing[:3]}")
    if problems and not allow_incomplete:
        raise SystemExit("incomplete results:\n  " + "\n  ".join(problems))
    for p in problems:
        print(f"  [warn] {p}")

    clinical = set(cfg["clinical_arms"]["arms"])
    digests: dict[tuple, dict] = {}
    for p in glob.glob(str(results_dir / "oof" / "*" / "*.meta.json")):
        m = json.loads(Path(p).read_text())
        # only the configured models: the OOF directory is shared across phases
        if m["model"] in cfg["models"] and m["model"] not in clinical:
            digests.setdefault((m["seed"], m["fold"]), {})[m["model"]] = m["preprocessing"]["input_digest"]
    bad = {k: v for k, v in digests.items() if len(set(v.values())) > 1}
    if bad and not cfg["summary"].get("require_identical_inputs", True):
        print(f"  note: inputs differ across models on {len(bad)} cells (tuned preprocessing; expected)")
        return
    if bad:
        raise SystemExit(f"models saw different inputs on (seed, fold): {sorted(bad)}")
    print(f"  integrity ok: identical inputs across models on {len(digests)} (seed, fold) cells")


# ── tables ───────────────────────────────────────────────────────────────────
def fold_table(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, seed, fold), g in oof.groupby(["model", "seed", "fold"]):
        rows.append({"model": model, "seed": seed, "fold": fold, "n": len(g),
                     **fold_metrics(g["label"].to_numpy(), g["prob"].to_numpy(), g["pred"].to_numpy())})
    return pd.DataFrame(rows)


def cv_table(folds: pd.DataFrame, order: list[str], leakref: set[str]) -> pd.DataFrame:
    rows = []
    for model in order:
        g = folds[folds["model"] == model]
        if g.empty:
            continue
        row = {"model": model, "role": "leakage reference" if model in leakref else "model",
               "n_seeds": g["seed"].nunique(), "n_folds": len(g)}
        for m in METRICS:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_std"] = g[m].std(ddof=1) if len(g) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def wide_oof(oof: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """One row per lesion; columns (model, seed) -> prob. Lesion order is fixed."""
    w = oof.pivot_table(index=["ProxID", "fid", "label"], columns=["model", "seed"],
                        values="prob", aggfunc="first")
    return w.reset_index(), sorted(oof["seed"].unique())


def pooled_table(w: pd.DataFrame, seeds: list[int], order: list[str], leakref: set[str],
                 n_boot: int, boot_seed: int) -> pd.DataFrame:
    y = w["label"].to_numpy()
    groups = w["ProxID"].to_numpy()
    rows, boots = [], {}
    for model in order:
        cols = [(model, s) for s in seeds if (model, s) in w.columns]
        if not cols:
            continue
        per_seed = [auc_mw(y, w[c].to_numpy()) for c in cols]
        boots[model] = cols
        rows.append({"model": model, "role": "leakage reference" if model in leakref else "model",
                     "n_seeds": len(cols), "pooled_AUC_mean": np.mean(per_seed),
                     "pooled_AUC_std": np.std(per_seed, ddof=1) if len(cols) > 1 else float("nan"),
                     "pooled_AUC_per_seed": ";".join(f"{a:.4f}" for a in per_seed)})
    # patient-cluster bootstrap of the seed-averaged pooled AUC (same resamples for all models)
    dist = {m: [] for m in boots}
    S = {m: np.column_stack([w[c].to_numpy() for c in cols]) for m, cols in boots.items()}
    for idx in patient_bootstrap_indices(groups, n_boot, boot_seed):
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        for m in boots:
            dist[m].append(np.mean([auc_mw(yb, S[m][idx, j]) for j in range(S[m].shape[1])]))
    out = pd.DataFrame(rows)
    out["ci95_lo"] = [np.percentile(dist[m], 2.5) for m in out["model"]]
    out["ci95_hi"] = [np.percentile(dist[m], 97.5) for m in out["model"]]
    out["n_boot"] = [len(dist[m]) for m in out["model"]]
    return out


def delong_tables(w: pd.DataFrame, seeds: list[int], ref: str, order: list[str],
                  leakref: set[str], n_boot: int, boot_seed: int):
    y = w["label"].to_numpy()
    groups = w["ProxID"].to_numpy()
    per_seed, summary = [], []
    others = [m for m in order if m != ref]
    for other in others:
        common = [s for s in seeds if (ref, s) in w.columns and (other, s) in w.columns]
        if not common:
            continue
        res = []
        for s in common:
            r = delong_paired(y, w[(ref, s)].to_numpy(), w[(other, s)].to_numpy())
            res.append(r)
            per_seed.append({"reference": ref, "comparator": other, "seed": s, **r})
        A = np.column_stack([w[(ref, s)].to_numpy() for s in common])
        B = np.column_stack([w[(other, s)].to_numpy() for s in common])
        deltas = []
        for idx in patient_bootstrap_indices(groups, n_boot, boot_seed):
            yb = y[idx]
            if yb.min() == yb.max():
                continue
            deltas.append(np.mean([auc_mw(yb, A[idx, j]) - auc_mw(yb, B[idx, j])
                                   for j in range(len(common))]))
        ps = np.array([r["p"] for r in res])
        summary.append({
            "reference": ref, "comparator": other,
            "role": "leakage reference" if other in leakref else "model",
            "n_seeds": len(common),
            "AUC_ref_mean": np.mean([r["auc_a"] for r in res]),
            "AUC_comp_mean": np.mean([r["auc_b"] for r in res]),
            "delta_AUC_mean": np.mean([r["delta"] for r in res]),
            "delta_ci95_lo": np.percentile(deltas, 2.5),
            "delta_ci95_hi": np.percentile(deltas, 97.5),
            "delong_z_mean": np.mean([r["z"] for r in res]),
            "delong_p_median": float(np.median(ps)),
            "delong_p_min": float(ps.min()), "delong_p_max": float(ps.max()),
            "seeds_p_lt_005": int((ps < 0.05).sum()),
        })
    return pd.DataFrame(per_seed), pd.DataFrame(summary)


# ── LaTeX ────────────────────────────────────────────────────────────────────
def _pm(m: float, s: float) -> str:
    return f"${m:.3f} \\pm {s:.3f}$" if np.isfinite(s) else f"${m:.3f}$"


def _name(model: str, leakref: set[str]) -> str:
    n = LABELS.get(model, model).replace("_", r"\_")
    return n + r"$^{\dagger}$" if model in leakref else n


def _p(p: float) -> str:
    return "$<0.001$" if p < 1e-3 else f"${p:.3f}$"


def write_tex(path: Path, caption: str, label: str, header: list[str], rows: list[list[str]],
              note: str | None) -> None:
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\small", rf"\begin{{tabular}}{{l{'c' * (len(header) - 1)}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines += [rf"\par\smallskip\footnotesize {note}"]
    lines += [r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


LEAK_NOTE = (r"$^{\dagger}$Leakage reference only: Gleason grade is post-biopsy "
             r"information and is not available at screening.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--boot-seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    results_dir = v3_path(args.results_dir)
    out = v3_path(cfg["summary"]["tables_dir"]) if cfg["summary"].get("tables_dir") else results_dir / "tables"
    out.mkdir(parents=True, exist_ok=True)
    order = list(cfg["models"])
    leakref = set(cfg.get("leakage_reference_models", []))
    ref = cfg["summary"]["reference_model"]
    n_boot = args.n_boot or int(cfg["summary"]["bootstrap_iterations"])

    oof = load_oof(results_dir)
    # only the configured models and seeds (the OOF directory is shared across phases)
    oof = oof[oof["model"].isin(order) & oof["seed"].isin(cfg["cv"]["seeds"])].copy()
    check_integrity(oof, cfg, results_dir, args.allow_incomplete)

    folds = fold_table(oof)
    folds.to_csv(out / "fold_metrics.csv", index=False)

    cv = cv_table(folds, order, leakref)
    counts = param_counts(results_dir)
    cv.insert(2, "n_params", [counts.get(m) for m in cv["model"]])
    cv.to_csv(out / "cv_metrics.csv", index=False)
    n_seeds, n_splits = oof["seed"].nunique(), cfg["cv"]["n_splits"]
    write_tex(
        out / "cv_metrics.tex",
        f"Patient-grouped {n_splits}-fold CV on PROSTATEx (327 lesions, 204 patients), "
        f"mean $\\pm$ SD over {n_seeds} seeds $\\times$ {n_splits} folds. "
        "Thresholds are F1-optimal on each training fold.",
        "tab:v3_cv",
        ["Model", "Params", "AUC", "Recall", "Precision", "F1", "Specificity", "Bal. acc."],
        [[_name(r.model, leakref), "--" if pd.isna(r.n_params) else f"{int(r.n_params)}"]
         + [_pm(getattr(r, f"{m}_mean"), getattr(r, f"{m}_std")) for m in METRICS]
         for r in cv.itertuples()],
        LEAK_NOTE if leakref & set(cv["model"]) else None,
    )

    w, seeds = wide_oof(oof)
    pooled = pooled_table(w, seeds, order, leakref, n_boot, args.boot_seed)
    pooled.to_csv(out / "pooled_auc.csv", index=False)
    write_tex(
        out / "pooled_auc.tex",
        "Pooled out-of-fold AUC: mean $\\pm$ SD across seeds, with 95\\% CI from "
        f"a patient-level bootstrap ({n_boot} resamples) of the seed-averaged pooled AUC.",
        "tab:v3_pooled",
        ["Model", "Pooled AUC", "95\\% CI"],
        [[_name(r.model, leakref), _pm(r.pooled_AUC_mean, r.pooled_AUC_std),
          f"[{r.ci95_lo:.3f}, {r.ci95_hi:.3f}]"] for r in pooled.itertuples()],
        LEAK_NOTE if leakref & set(pooled["model"]) else None,
    )

    per_seed, dl = delong_tables(w, seeds, ref, order, leakref, n_boot, args.boot_seed)
    per_seed.to_csv(out / "delong_per_seed.csv", index=False)
    dl.to_csv(out / "delong.csv", index=False)
    write_tex(
        out / "delong.tex",
        f"Paired comparison of {LABELS.get(ref, ref)} against each model on pooled OOF "
        f"predictions. $\\Delta$AUC = reference $-$ comparator, averaged over seeds; the 95\\% CI "
        "is from a patient-level bootstrap. DeLong $p$: median (range) over seeds.",
        "tab:v3_delong",
        ["Comparator", "$\\Delta$AUC", "95\\% CI", "DeLong $p$", "Seeds $p<0.05$"],
        [[_name(r.comparator, leakref), f"${r.delta_AUC_mean:+.3f}$",
          f"[{r.delta_ci95_lo:+.3f}, {r.delta_ci95_hi:+.3f}]",
          f"{_p(r.delong_p_median)} ({_p(r.delong_p_min)}--{_p(r.delong_p_max)})",
          f"{r.seeds_p_lt_005}/{r.n_seeds}"] for r in dl.itertuples()],
        LEAK_NOTE if leakref & set(dl["comparator"]) else None,
    )

    pd.set_option("display.width", 160)
    print("\n  CV metrics (mean over seeds x folds):")
    print(cv[["model", "n_folds", "AUC_mean", "AUC_std", "F1_mean", "F1_std"]].round(3).to_string(index=False))
    print("\n  Pooled OOF AUC:")
    print(pooled[["model", "pooled_AUC_mean", "pooled_AUC_std", "ci95_lo", "ci95_hi"]].round(3).to_string(index=False))
    print(f"\n  {ref} vs others:")
    print(dl[["comparator", "delta_AUC_mean", "delta_ci95_lo", "delta_ci95_hi", "delong_p_median",
              "seeds_p_lt_005"]].round(4).to_string(index=False))
    print(f"\n  wrote tables to {out}")


if __name__ == "__main__":
    main()
