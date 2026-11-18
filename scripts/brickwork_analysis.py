"""
Phase 2B: tree vs brickwork HQCNN.

Stages (default: all):
  mismatch   twin algorithm vs statevector for tree and brickwork (random inputs/params);
             tree must match exactly, brickwork must not
  specs      qml.specs for both topologies at 4..12 qubits (params, gates, 2q gates, depth)
  train      train both topologies at 4..12 qubits on seed 0 / fold 0 (for trained weights)
  entropy    von Neumann entropy of the pre-readout state across two bipartitions
             (contiguous half and interleaved even|odd), at random init (real PCA inputs)
             and at trained weights (test-fold inputs); same states for both cuts

Outputs: results/phase2/brickwork/*.csv, *.tex, figures/entanglement_entropy.pdf

Usage:
  python scripts/brickwork_analysis.py --n-jobs 5
  python scripts/brickwork_analysis.py --stages mismatch specs
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

OUT = V3_DIR / "results" / "phase2" / "brickwork"
SIZES = [4, 6, 8, 10, 12]
TOPOLOGIES = ["tree", "brickwork"]


def _setup():
    sys.path.insert(0, str(V3_DIR))
    from v3core.config import load_config
    return load_config()


# ── mismatch ────────────────────────────────────────────────────────────────
def stage_mismatch(n_draws: int) -> pd.DataFrame:
    from pennylane import numpy as np
    from v3core.hqcnn import HQCNN
    from v3core.phase2 import hqcnn_variant
    from v3core.twin import product_state_expval, twin_expval

    cfg = _setup()
    n_feat = int(cfg["preprocessing"]["pca_dim"])
    rng = onp.random.default_rng(1)
    rows = []
    for n in SIZES:
        for topo in TOPOLOGIES:
            h = hqcnn_variant(cfg, n_qubits=n, conv_topology=topo)
            m = HQCNN(h)
            X = rng.uniform(0, onp.pi, size=(n_draws, n_feat))
            err, err_twin, zs, lossy = [], [], [], 0
            for x in X:
                q = np.array(rng.uniform(-onp.pi, onp.pi, m.n_q))
                xs = np.array(x[None, :], requires_grad=False)
                z_sv = float(m._z(xs, q)[0])
                z_ps, lossy = product_state_expval(xs, q, m.ops, m.readout, n, list(h["reupload_gates"]),
                                                   h["pooling_gate"], h["pooling_survivor"])
                err.append(abs(z_sv - float(z_ps[0])))
                zs.append(z_sv)
                if topo == "tree":
                    z_tw = twin_expval(xs, q, n, int(h["conv_layers"]), list(h["reupload_gates"]),
                                       h["pooling_gate"], h["pooling_survivor"])
                    err_twin.append(abs(z_sv - float(z_tw[0])))
            err = onp.array(err)
            rows.append({
                "n_qubits": n, "topology": topo, "n_draws": n_draws,
                "lossy_flushes_per_forward": lossy,
                "max_abs_err": float(err.max()), "mean_abs_err": float(err.mean()),
                "frac_err_gt_1e-6": float((err > 1e-6).mean()),
                "std_z_statevector": float(onp.std(zs)),
                "max_abs_err_exact_twin": float(max(err_twin)) if err_twin else float("nan"),
            })
            r = rows[-1]
            print(f"  mismatch {n:>2}q {topo:<9} lossy={lossy:<3} max|err|={r['max_abs_err']:.2e} "
                  f"mean|err|={r['mean_abs_err']:.2e} (std<Z>={r['std_z_statevector']:.3f})", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "twin_mismatch.csv", index=False)
    return df


# ── specs ───────────────────────────────────────────────────────────────────
def stage_specs() -> pd.DataFrame:
    import pennylane as qml
    from pennylane import numpy as np
    from v3core.hqcnn import HQCNN
    from v3core.phase2 import hqcnn_variant, write_tex_table

    cfg = _setup()
    n_feat = int(cfg["preprocessing"]["pca_dim"])
    rows = []
    for n in SIZES:
        for topo in TOPOLOGIES:
            m = HQCNN(hqcnn_variant(cfg, n_qubits=n, conv_topology=topo))
            x = np.array(onp.full(n_feat, 0.3), requires_grad=False)
            q = np.array(onp.full(m.n_q, 0.1))
            r = qml.specs(m.circuit)(x, q)["resources"]
            two = sum(v for k, v in r.gate_sizes.items() if k >= 2)
            rows.append({"n_qubits": n, "topology": topo, "n_params": m.n_params,
                         "n_params_circuit": m.n_q, "gates": r.num_gates, "gates_2q": two,
                         "depth": r.depth, "n_features": n_feat})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "specs.csv", index=False)
    wide = df.pivot(index="n_qubits", columns="topology")
    write_tex_table(
        OUT / "specs.tex",
        "Circuit resources of the tree and brickwork HQCNN (3 conv layers, 12 angle-encoded "
        "features, from \\texttt{qml.specs}). Parameters include the 2-parameter affine head.",
        "tab:v3_specs",
        ["Qubits", "Params (tree / brick)", "Gates", "2-qubit gates", "Depth"],
        [[str(n)] + [f"{wide.loc[n, (c, 'tree')]} / {wide.loc[n, (c, 'brickwork')]}"
                     for c in ("n_params", "gates", "gates_2q", "depth")] for n in SIZES],
    )
    print(df.to_string(index=False))
    return df


# ── training (one fold per size x topology) ─────────────────────────────────
def train_task(n: int, topo: str) -> dict:
    cfg = _setup()
    from v3core.data import load_cohort, make_folds
    from v3core.metrics import fold_metrics, pick_threshold
    from v3core.models import fit_model
    from v3core.phase2 import hqcnn_variant
    from v3core.preprocess import build_fold
    from v3core.seeding import component_seed, seed_everything

    seed, fold = 0, 0
    seed_everything(component_seed(seed, fold, "global:hqcnn"))
    cfg = dict(cfg)
    cfg["hqcnn"] = hqcnn_variant(cfg, n_qubits=n, conv_topology=topo)
    cohort = load_cohort(cfg)
    tr, te = make_folds(cohort, cfg["cv"]["n_splits"], seed)[fold]
    fd = build_fold(cohort, tr, te, cfg["preprocessing"], component_seed(seed, fold, "resample"))
    res = fit_model("hqcnn", fd, cfg, seed, fold)
    thr = pick_threshold("f1", fd.y_train, res["prob_train"])
    met = fold_metrics(fd.y_test, res["prob_test"], (res["prob_test"] >= thr).astype(int))
    OUT.joinpath("trained").mkdir(parents=True, exist_ok=True)
    meta = res["meta"]
    onp.savez(OUT / "trained" / f"{topo}_{n}q.npz", params=onp.asarray(meta["params"]),
              scaler_lo=onp.asarray(meta["scaler_lo"]), scaler_hi=onp.asarray(meta["scaler_hi"]))
    return {"n_qubits": n, "topology": topo, "seed": seed, "fold": fold,
            "n_params": meta["n_params"], "fit_seconds": meta["fit_seconds"],
            "final_loss": meta["loss_history"][-1], **met}


def stage_train(n_jobs: int) -> pd.DataFrame:
    tasks = [(n, t) for n in SIZES for t in TOPOLOGIES]
    tasks.sort(key=lambda t: -t[0])
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        rows = list(ex.map(train_task, *zip(*tasks)))
    for r in rows:
        print(f"  trained {r['n_qubits']:>2}q {r['topology']:<9} AUC={r['AUC']:.3f} "
              f"loss={r['final_loss']:.4f} ({r['fit_seconds']:.0f}s)", flush=True)
    df = pd.DataFrame(rows).sort_values(["n_qubits", "topology"])
    df.to_csv(OUT / "trained_single_fold.csv", index=False)
    return df


# ── entropy ─────────────────────────────────────────────────────────────────
def stage_entropy(n_random: int) -> pd.DataFrame:
    """Entropy across two bipartitions, computed on the SAME states:
    contiguous (wires 0..n/2-1 | rest) and interleaved (even | odd wires)."""
    from pennylane import numpy as np
    from v3core.hqcnn import build_qnode, n_quantum_params, qcnn_plan
    from v3core.phase2 import CUTS, bipartition_entropy, hqcnn_variant, real_inputs

    cfg = _setup()
    data = real_inputs(cfg)                    # PCA-12, seed 0 / fold 0, HQCNN scaling
    rng = onp.random.default_rng(2)
    rows = []
    for n in SIZES:
        page = n / 2 - 1 / (2 * onp.log(2))    # Page: mean entropy of Haar-random states
        for topo in TOPOLOGIES:
            h = hqcnn_variant(cfg, n_qubits=n, conv_topology=topo)
            ops, ro = qcnn_plan(n, int(h["conv_layers"]), h["pooling_survivor"], topo)
            circ = build_qnode(n, ops, ro, list(h["reupload_gates"]), h["pooling_gate"],
                               "default.qubit", "backprop", measure="state")
            n_q = n_quantum_params(ops, h["pooling_gate"])
            states_rand = []
            for _ in range(n_random):
                x = data["train"][rng.integers(len(data["train"]))]
                q = rng.uniform(-onp.pi, onp.pi, n_q)
                states_rand.append(onp.asarray(circ(np.array(x), np.array(q))))
            f = OUT / "trained" / f"{topo}_{n}q.npz"
            states_tr = []
            if f.exists():
                w = onp.load(f)
                states_tr = [onp.asarray(circ(np.array(x), np.array(w["params"][:n_q]))) for x in data["test"]]
            for cut, wires_fn in CUTS.items():
                wa = wires_fn(n)
                S_r = [bipartition_entropy(s, n, wa) for s in states_rand]
                row = {"n_qubits": n, "topology": topo, "cut": cut, "wires_a": " ".join(map(str, wa)),
                       "max_possible": len(wa), "page_haar": page,
                       "S_random_mean": float(onp.mean(S_r)), "S_random_std": float(onp.std(S_r)),
                       "n_random": n_random}
                if states_tr:
                    S_t = [bipartition_entropy(s, n, wa) for s in states_tr]
                    row.update(S_trained_mean=float(onp.mean(S_t)), S_trained_std=float(onp.std(S_t)),
                               n_trained_inputs=len(S_t))
                rows.append(row)
                print(f"  entropy {n:>2}q {topo:<9} {cut:<11} random {row['S_random_mean']:.3f}"
                      f"±{row['S_random_std']:.3f}  trained {row.get('S_trained_mean', float('nan')):.3f}"
                      f"  (Page {page:.2f}, max {len(wa)})", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "entropy.csv", index=False)
    plot_entropy(df)
    return df


def plot_entropy(df: pd.DataFrame) -> None:
    from v3core.plotting import FIG_DIR, save_pdf, setup_style
    import matplotlib.pyplot as plt

    setup_style()
    cuts = [c for c in ("contiguous", "interleaved") if c in set(df["cut"])]
    fig, axes = plt.subplots(1, len(cuts), figsize=(3.5 * len(cuts), 2.8), sharey=True, squeeze=False)
    # (a) contiguous cut, wires 0..n/2-1 versus the rest; (b) interleaved cut, even versus odd wires
    for i, (ax, cut) in enumerate(zip(axes[0], cuts)):
        d_cut = df[df["cut"] == cut]
        d0 = d_cut[d_cut.topology == "tree"]
        ax.plot(d0.n_qubits, d0.max_possible, ":", color="0.5", lw=1, label="Maximum ($n/2$)")
        ax.plot(d0.n_qubits, d0.page_haar, "--", color="0.5", lw=1, label="Haar (Page)")
        for topo, c in (("tree", "C0"), ("brickwork", "C3")):
            d = d_cut[d_cut.topology == topo]
            ax.errorbar(d.n_qubits, d.S_random_mean, yerr=d.S_random_std, fmt="o-", color=c, ms=3.5,
                        lw=1.2, capsize=2, label=f"{topo.capitalize()}, random init")
            if "S_trained_mean" in d and d.S_trained_mean.notna().any():
                ax.errorbar(d.n_qubits, d.S_trained_mean, yerr=d.S_trained_std, fmt="s--", color=c,
                            ms=3.5, lw=1.0, capsize=2, mfc="white", label=f"{topo.capitalize()}, trained")
        ax.text(0.02, 0.97, "(" + "ab"[i] + ")", transform=ax.transAxes, ha="left", va="top", weight="bold")
        ax.set_xlabel("Number of qubits $n$")
        ax.set_xticks(SIZES)
    axes[0][0].set_ylabel("Entanglement entropy (bits)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    save_pdf(fig, FIG_DIR / "entanglement_entropy.pdf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["mismatch", "specs", "train", "entropy"])
    ap.add_argument("--n-draws", type=int, default=200)
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=5)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if "mismatch" in args.stages:
        stage_mismatch(args.n_draws)
    if "specs" in args.stages:
        stage_specs()
    if "train" in args.stages:
        stage_train(args.n_jobs)
    if "entropy" in args.stages:
        stage_entropy(args.n_random)
    if "plot_entropy" in args.stages:          # redraw from entropy.csv only
        plot_entropy(pd.read_csv(OUT / "entropy.csv"))
        return
    (OUT / "run_info.json").write_text(json.dumps({"stages": args.stages, "seconds": time.perf_counter() - t0}))


if __name__ == "__main__":
    main()
