"""
Phase 2D: gradient variance vs qubit count (barren-plateau check). No training.

Runs (each writes results/phase2/gradvar/<tag>/):
  pca_n       inputs = real PCA features with pca_dim = n (one RY per qubit);
              target d<Z>/d(theta); n in {4, 6, ..., 16}              [the original D run]
  champion12  inputs = the champion's 12 PCA features with RY/RZ/... re-upload
              (at n = 16, qubits 12-15 get no feature); target d<Z>/d(theta); n in {8, 12, 16}
  focal8      champion 12-feature inputs, n = 8; target = gradient of the batch focal
              loss (batch 64 from the resampled training set, balanced class weights,
              head at its init a = 1, b = 0), i.e. the actual training signal at step 0

Common: topology in {tree, brickwork}; 500 draws of params ~ U(-pi, pi) (and a random
lesion / mini-batch); real inputs from seed 0 / fold 0, angle-scaled exactly as HQCNN.fit.
Reported parameters: first_conv (index 0: RY on qubit 0, first conv block) and last_pool
(the final pool's CRX); per-parameter variances for all parameters are also saved.

Uncertainty: 2,000-sample bootstrap over draws gives 95% CIs for every variance. For
runs with >= 3 sizes, the exponential rate (ln Var = a - b n) and power-law exponent
(ln Var = c - k ln n) are refitted on every bootstrap replicate (draws resampled
independently at each n) to get 95% CIs.

Outputs per tag: raw/<topo>_<n>.npz (all gradients), gradvar.csv, gradvar_fits.csv,
per_param_var.csv; figure figures/gradient_variance[_<tag>].pdf

Usage:
  python scripts/gradvar.py --run pca_n --n-jobs 4
  python scripts/gradvar.py --run champion12 --n-jobs 4
  python scripts/gradvar.py --run focal8 --n-jobs 2
"""

from __future__ import annotations

import argparse
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

OUT_ROOT = V3_DIR / "results" / "phase2" / "gradvar"
TOPOLOGIES = ["tree", "brickwork"]
RUNS = {
    "pca_n": {"sizes": [4, 6, 8, 10, 12, 14, 16], "encoding": "pca_n", "target": "z"},
    "champion12": {"sizes": [8, 12, 16], "encoding": "champion12", "target": "z"},
    "focal8": {"sizes": [8], "encoding": "champion12", "target": "focal"},
}
N_BOOT = 2000


def task(n: int, topo: str, n_draws: int, encoding: str, target: str, tag: str) -> dict:
    sys.path.insert(0, str(V3_DIR))
    import pennylane as qml
    from pennylane import numpy as np
    from v3core.config import load_config
    from v3core.hqcnn import HQCNN, build_qnode, n_quantum_params, qcnn_plan, weighted_loss
    from v3core.phase2 import hqcnn_variant, real_inputs
    from v3core.seeding import component_rng

    cfg = load_config()
    h = hqcnn_variant(cfg, n_qubits=n, conv_topology=topo)
    data = real_inputs(cfg, pca_dim=n if encoding == "pca_n" else None)
    Xtr = data["train"]
    ops, ro = qcnn_plan(n, int(h["conv_layers"]), h["pooling_survivor"], topo)
    n_q = n_quantum_params(ops, h["pooling_gate"])
    rng = component_rng(n, 0, f"gradvar:{tag}:{topo}")
    grads = onp.empty((n_draws, n_q))
    t0 = time.perf_counter()
    if target == "z":
        circ = build_qnode(n, ops, ro, list(h["reupload_gates"]), h["pooling_gate"],
                           "lightning.qubit", "adjoint")
        for d in range(n_draws):
            x = np.array(Xtr[rng.integers(len(Xtr))], requires_grad=False)
            q = np.array(rng.uniform(-onp.pi, onp.pi, n_q), requires_grad=True)
            grads[d] = onp.asarray(qml.grad(circ, argnum=1)(x, q))
    else:  # focal-loss gradient on a training mini-batch, head at init (a=1, b=0)
        model = HQCNN(h)                       # default.qubit + backprop, batched
        y_fit = data["fold_data"].y_fit.astype(float)
        counts = onp.bincount(y_fit.astype(int), minlength=2)
        w_all = onp.where(y_fit == 1, len(y_fit) / (2 * counts[1]), len(y_fit) / (2 * counts[0]))
        bs, gamma = int(h["batch_size"]), float(h["focal_gamma"])
        for d in range(n_draws):
            bi = rng.choice(len(Xtr), size=bs, replace=False)
            yb = np.array(y_fit[bi], requires_grad=False)
            wb = np.array(w_all[bi], requires_grad=False)
            q = np.array(rng.uniform(-onp.pi, onp.pi, n_q), requires_grad=True)

            def cost(qq, Xb=Xtr[bi], yb=yb, wb=wb):
                return weighted_loss(1.0 * model._z(Xb, qq) + 0.0, yb, wb, h["loss"], gamma)
            grads[d] = onp.asarray(qml.grad(cost)(q))
    secs = time.perf_counter() - t0

    labels, level, prev = [], 0, None
    width_pool = 2 if h["pooling_gate"] in ("crx_crz", "cry") else 1
    for kind, pair in ops:
        if prev == "pool" and kind == "conv":
            level += 1
        width = 6 if kind == "conv" else width_pool
        labels += [(level, kind, pair[0], pair[1], j) for j in range(width)]
        prev = kind
    out = OUT_ROOT / tag / "raw"
    out.mkdir(parents=True, exist_ok=True)
    onp.savez_compressed(out / f"{topo}_{n}.npz", grads=grads)
    var = grads.var(axis=0, ddof=1)
    per_param = pd.DataFrame(labels, columns=["level", "kind", "wire_a", "wire_b", "slot"])
    per_param["var"] = var
    per_param["n_qubits"], per_param["topology"] = n, topo
    return {"n_qubits": n, "topology": topo, "n_params_circuit": n_q, "n_draws": n_draws,
            "seconds": secs, "grads": grads, "per_param": per_param}


def boot_var_ci(g: onp.ndarray, rng: onp.random.Generator) -> tuple[float, float, onp.ndarray]:
    idx = rng.integers(0, len(g), size=(N_BOOT, len(g)))
    bv = g[idx].var(axis=1, ddof=1)
    return float(onp.percentile(bv, 2.5)), float(onp.percentile(bv, 97.5)), bv


def fit_line(x: onp.ndarray, lnv: onp.ndarray):
    A = onp.vstack([onp.ones_like(x), x]).T
    coef, *_ = onp.linalg.lstsq(A, lnv, rcond=None)
    pred = A @ coef
    r2 = 1 - ((lnv - pred) ** 2).sum() / ((lnv - lnv.mean()) ** 2).sum()
    return coef, float(r2)


def summarise(results: list[dict], tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = onp.random.default_rng(12345)
    rows, boots = [], {}
    params = {"first_conv": 0, "last_pool": -1}
    for r in results:
        row = {k: r[k] for k in ("n_qubits", "topology", "n_params_circuit", "n_draws", "seconds")}
        for name, j in params.items():
            g = r["grads"][:, j]
            lo, hi, bv = boot_var_ci(g, rng)
            row[f"var_{name}"] = float(g.var(ddof=1))
            row[f"var_{name}_ci_lo"], row[f"var_{name}_ci_hi"] = lo, hi
            boots[(r["topology"], r["n_qubits"], name)] = bv
        allv = r["grads"].var(axis=0, ddof=1)
        row["var_mean_all_params"] = float(allv.mean())
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["topology", "n_qubits"])

    fits = []
    for topo in sorted(df.topology.unique()):
        d = df[df.topology == topo].sort_values("n_qubits")
        n = d.n_qubits.to_numpy(dtype=float)
        if len(n) < 3:
            continue
        for name in params:
            lnv = onp.log(d[f"var_{name}"].to_numpy())
            (a, b), r2e = fit_line(n, lnv)
            (c, k), r2p = fit_line(onp.log(n), lnv)
            B = onp.vstack([boots[(topo, int(nn), name)] for nn in n])       # (sizes, N_BOOT)
            rates, expos = [], []
            for bcol in range(N_BOOT):
                lb = onp.log(B[:, bcol])
                rates.append(-fit_line(n, lb)[0][1])
                expos.append(-fit_line(onp.log(n), lb)[0][1])
            fits.append({
                "topology": topo, "parameter": name, "n_sizes": len(n),
                "exp_rate_per_qubit": float(-b), "exp_rate_ci_lo": float(onp.percentile(rates, 2.5)),
                "exp_rate_ci_hi": float(onp.percentile(rates, 97.5)), "exp_intercept": float(a), "exp_r2": r2e,
                "power_exponent": float(-k), "power_exponent_ci_lo": float(onp.percentile(expos, 2.5)),
                "power_exponent_ci_hi": float(onp.percentile(expos, 97.5)), "power_intercept": float(c),
                "power_r2": r2p,
            })
    return df, pd.DataFrame(fits)


def plot(df: pd.DataFrame, fits: pd.DataFrame, tag: str, sizes: list[int]) -> None:
    import matplotlib.pyplot as plt
    from v3core.plotting import FIG_DIR, save_pdf, setup_style

    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    params = ("first_conv", "last_pool")   # (a) first-layer conv parameter, (b) last pooling parameter
    for i, (ax, name) in enumerate(zip(axes, params)):
        for topo, c, mk in (("tree", "C0", "o"), ("brickwork", "C3", "s")):
            d = df[df.topology == topo].sort_values("n_qubits")
            v = d[f"var_{name}"]
            yerr = [v - d[f"var_{name}_ci_lo"], d[f"var_{name}_ci_hi"] - v]
            label = topo.capitalize()
            f = fits[(fits.topology == topo) & (fits.parameter == name)] if len(fits) else fits
            if len(f):
                f = f.iloc[0]
                label += (f" (rate {f.exp_rate_per_qubit:.2f} "
                          f"[{f.exp_rate_ci_lo:.2f}, {f.exp_rate_ci_hi:.2f}]/qubit)")
                nn = onp.linspace(d.n_qubits.min(), d.n_qubits.max(), 50)
                ax.plot(nn, onp.exp(f.exp_intercept - f.exp_rate_per_qubit * nn), "-", color=c, lw=1, alpha=0.6)
            ax.errorbar(d.n_qubits, v, yerr=yerr, fmt=mk, color=c, ms=4, capsize=2, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("Number of qubits $n$")
        ax.set_xticks(sizes)
        ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0)
        ax.text(0.98, 0.97, "(" + "ab"[i] + ")", transform=ax.transAxes, ha="right", va="top", weight="bold")
        ax.grid(True, which="major", lw=0.3, alpha=0.5)
    axes[0].set_ylabel(r"Var$[\partial\langle Z\rangle/\partial\theta]$ (95% CI)")
    name = "gradient_variance.pdf" if tag == "pca_n" else f"gradient_variance_{tag}.pdf"
    save_pdf(fig, FIG_DIR / name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=list(RUNS), default="pca_n")
    ap.add_argument("--n-draws", type=int, default=500)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from gradvar*.csv")
    args = ap.parse_args()
    spec = RUNS[args.run]
    out = OUT_ROOT / args.run
    out.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot(pd.read_csv(out / "gradvar.csv"), pd.read_csv(out / "gradvar_fits.csv"), args.run, spec["sizes"])
        return
    tasks = sorted([(n, t) for n in spec["sizes"] for t in TOPOLOGIES], key=lambda t: -t[0])
    k = len(tasks)
    with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
        results = list(ex.map(task, *zip(*tasks), [args.n_draws] * k, [spec["encoding"]] * k,
                              [spec["target"]] * k, [args.run] * k))
    pd.concat([r["per_param"] for r in results]).to_csv(out / "per_param_var.csv", index=False)
    df, fits = summarise(results, args.run)
    df["encoding"], df["target"] = spec["encoding"], spec["target"]
    df.to_csv(out / "gradvar.csv", index=False)
    fits.to_csv(out / "gradvar_fits.csv", index=False)
    if spec["target"] == "z":
        plot(df, fits, args.run, spec["sizes"])
    cols = ["topology", "n_qubits", "var_first_conv", "var_first_conv_ci_lo", "var_first_conv_ci_hi",
            "var_last_pool", "var_last_pool_ci_lo", "var_last_pool_ci_hi", "seconds"]
    print(df[cols].to_string(index=False))
    if len(fits):
        print(fits.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
