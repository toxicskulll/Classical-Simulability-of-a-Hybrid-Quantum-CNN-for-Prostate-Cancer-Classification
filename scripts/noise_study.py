"""
Phase 3 step 2: gate noise and shot noise on the tree HQCNN (via the exact twin).

Stages:
  verify     noisy twin == PennyLane default.mixed (DepolarizingChannel, AmplitudeDamping,
             2q depolarising as a 16-Kraus QubitChannel) on random draws
  train6q    noiseless 6-qubit / 1-layer HQCNN on all 25 (seed, fold) cells (weights for mode a)
  eval       mode (a): noiselessly trained weights evaluated under gate noise
             p2 in {0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2}, p1 = p2/10, and under shot noise
             {100, 1000, 10000} at p2 = 0 and 1e-3. Sources: Phase 1 8q/3L weights,
             6q/1L weights, and (if present) the Phase 3 tuned HQCNN weights.
  trainnoisy mode (b): train under noise at p2 in {1e-3, 1e-2} (8q/3L and 6q/1L), same init/
             shuffle stream as the noiseless model; evaluate at the training noise level.
  summary    pooled AUC vs noise (mean +/- SD over seeds, patient-bootstrap 95% CI, paired
             delta vs the same weights noiseless), CSV + LaTeX + figures/noise_auc.pdf

Thresholds: F1-optimal on the training fold's probabilities computed under the SAME noise
(recalibrated), so AUC and F1 both reflect the noisy model.

Usage:
  python scripts/noise_study.py --stages verify train6q eval summary --n-jobs 5
  python scripts/noise_study.py --stages trainnoisy summary --n-jobs 5
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

OUT = V3_DIR / "results" / "phase3" / "noise"
P2_LEVELS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
SHOTS = [100, 1000, 10000]
SHOT_P2 = [0.0, 1e-3]
TRAIN_P2 = [1e-3, 1e-2]
ARCHS = {"8q3L": {"n_qubits": 8, "conv_layers": 3}, "6q1L": {"n_qubits": 6, "conv_layers": 1}}
SEEDS, FOLDS = [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]


def _cfg():
    from v3core.config import load_config
    return load_config()


def _fold(cfg, seed, fold):
    from v3core.data import load_cohort, make_folds
    from v3core.preprocess import build_fold
    from v3core.seeding import component_seed
    cohort = load_cohort(cfg)
    tr, te = make_folds(cohort, cfg["cv"]["n_splits"], seed)[fold]
    fd = build_fold(cohort, tr, te, cfg["preprocessing"], component_seed(seed, fold, "resample"))
    return cohort, te, fd


TUNED_META = V3_DIR / "results" / "oof" / "hqcnn_tuned"


def _weights_path(source: str, seed: int, fold: int) -> Path:
    if source == "tuned_8q":        # weights live in the tuned run's OOF metadata
        return TUNED_META / f"seed{seed}_fold{fold}.meta.json"
    if source == "phase1_8q3L":
        return V3_DIR / "results" / "phase2" / "twin" / "trained_params" / f"seed{seed}_fold{fold}.npz"
    return OUT / "weights" / source / f"seed{seed}_fold{fold}.npz"


# ── verify ───────────────────────────────────────────────────────────────────
def stage_verify() -> dict:
    import itertools

    import pennylane as qml
    from pennylane import numpy as np
    from v3core.hqcnn import encode, n_quantum_params, qcnn_plan
    from v3core.noise import noisy_tree_expval

    cfg = _cfg()
    h = cfg["hqcnn"]
    gates = list(h["reupload_gates"])
    P = {"I": onp.eye(2), "X": onp.array([[0, 1], [1, 0]]), "Y": onp.array([[0, -1j], [1j, 0]]),
         "Z": onp.diag([1, -1])}
    res = {}
    for arch, a in ARCHS.items():
        n, cl = a["n_qubits"], a["conv_layers"]
        ops, ro = qcnn_plan(n, cl, "second", "tree")
        n_q = n_quantum_params(ops, h["pooling_gate"])
        p2, ratio = 3e-2, 1.0
        p1 = p2 / 10
        kraus2 = [onp.sqrt(1 - p2) * onp.eye(4)] + [
            onp.sqrt(p2 / 15) * onp.kron(P[x], P[y]) for x, y in itertools.product("IXYZ", repeat=2) if x + y != "II"]
        dev = qml.device("default.mixed", wires=n)

        def n1(w):
            qml.DepolarizingChannel(p1, wires=w)
            qml.AmplitudeDamping(ratio * p1, wires=w)

        def n2(a_, b_):
            qml.QubitChannel(kraus2, wires=[a_, b_])
            qml.AmplitudeDamping(ratio * p2, wires=a_)
            qml.AmplitudeDamping(ratio * p2, wires=b_)

        @qml.qnode(dev)
        def circ(x, q):
            nf = len(x)
            for r, g in enumerate(gates):
                for w in range(n):
                    k = r * n + w
                    if k < nf:
                        {"RY": qml.RY, "RZ": qml.RZ, "RX": qml.RX}[g](x[k], wires=w)
                        n1(w)
            i = 0
            for kind, (a_, b_) in ops:
                if kind == "conv":
                    t = q[i: i + 6]
                    i += 6
                    qml.RY(t[0], wires=a_); n1(a_); qml.RY(t[1], wires=b_); n1(b_)      # noqa: E702
                    qml.CNOT(wires=[a_, b_]); n2(a_, b_)                                  # noqa: E702
                    qml.RY(t[2], wires=a_); n1(a_); qml.RY(t[3], wires=b_); n1(b_)      # noqa: E702
                    qml.CNOT(wires=[b_, a_]); n2(a_, b_)                                  # noqa: E702
                    qml.RZ(t[4], wires=a_); n1(a_); qml.RZ(t[5], wires=b_); n1(b_)      # noqa: E702
                else:
                    t = q[i: i + 2]
                    i += 2
                    qml.CRZ(t[0], wires=[a_, b_]); n2(a_, b_)                             # noqa: E702
                    qml.CRX(t[1], wires=[a_, b_]); n2(a_, b_)                             # noqa: E702
            return qml.expval(qml.PauliZ(ro))

        rng = onp.random.default_rng(3)
        errs = []
        for _ in range(5):
            x = rng.uniform(0, onp.pi, 12)
            q = rng.uniform(-onp.pi, onp.pi, n_q)
            z_pl = float(circ(np.array(x), np.array(q)))
            z_tw = float(onp.asarray(noisy_tree_expval(x[None, :], q, n, cl, gates, p1, p2, ratio))[0])
            errs.append(abs(z_pl - z_tw))
        res[arch] = float(max(errs))
        print(f"  verify {arch}: noisy twin vs default.mixed (p2=3e-2, p1=3e-3, AD): max|err| = {max(errs):.1e}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verify.json").write_text(json.dumps(res, indent=2))
    assert max(res.values()) < 1e-10
    return res


# ── noiseless 6q/1L training ────────────────────────────────────────────────
def train_task(arch: str, seed: int, fold: int, p2: float) -> dict:
    sys.path.insert(0, str(V3_DIR))
    import copy

    from pennylane import numpy as np
    from v3core.arms import AngleScaler, train_adam
    from v3core.hqcnn import n_quantum_params, qcnn_plan
    from v3core.noise import noisy_tree_expval
    from v3core.seeding import component_rng, seed_everything

    cfg = _cfg()
    h = copy.deepcopy(cfg["hqcnn"])
    h.update(ARCHS[arch])
    n, cl, gates = h["n_qubits"], h["conv_layers"], list(h["reupload_gates"])
    seed_everything(seed * 100 + fold)
    _, _, fd = _fold(cfg, seed, fold)
    ops, _ = qcnn_plan(n, cl, "second", "tree")
    n_q = n_quantum_params(ops, h["pooling_gate"])
    sc = AngleScaler(h["angle_range"]).fit(fd.X_fit)
    rng = component_rng(seed, fold, "hqcnn")                    # same stream as the noiseless model
    p0 = onp.concatenate([rng.uniform(-onp.pi, onp.pi, n_q), [1.0, 0.0]])

    def logit(p, Xb):
        return p[-2] * noisy_tree_expval(np.array(Xb, requires_grad=False), p[:n_q], n, cl, gates,
                                         p2 / 10, p2, 1.0) + p[-1]
    t0 = time.perf_counter()
    params, hist = train_adam(logit, p0, sc.transform(fd.X_fit), fd.y_fit.astype(float), rng, h)
    tag = f"{arch}" if p2 == 0 else f"{arch}_trainp2_{p2:g}"
    path = _weights_path(tag, seed, fold)
    path.parent.mkdir(parents=True, exist_ok=True)
    onp.savez(path, params=params, scaler_lo=sc.lo, scaler_hi=sc.hi)
    return {"arch": arch, "train_p2": p2, "seed": seed, "fold": fold, "seconds": time.perf_counter() - t0,
            "final_loss": hist[-1]}


# ── evaluation under noise ──────────────────────────────────────────────────
def eval_task(source: str, arch: str, seed: int, fold: int, grid: list) -> list[dict]:
    """grid: list of (p2, shots) pairs; shots None = exact expectation."""
    sys.path.insert(0, str(V3_DIR))
    from v3core.metrics import fold_metrics, pick_threshold
    from v3core.noise import noisy_tree_expval, shot_estimate
    from v3core.seeding import component_seed

    cfg = _cfg()
    if source == "tuned_8q":
        # per-fold tuned preprocessing + circuit (conv_layers), weights from the OOF metadata
        from v3core.config import load_config
        from v3core.tuning import tuned_cfg
        cfg = tuned_cfg(load_config(V3_DIR / "configs" / "phase3.yaml"), "hqcnn", seed, fold)
        meta = json.loads(_weights_path(source, seed, fold).read_text())["model_meta"]
        params, lo, hi = (onp.asarray(meta[k]) for k in ("params", "scaler_lo", "scaler_hi"))
        h = dict(cfg["hqcnn"])
    else:
        h = dict(cfg["hqcnn"])
        h.update(ARCHS[arch])
        w = onp.load(_weights_path(source, seed, fold))
        params, lo, hi = w["params"], w["scaler_lo"], w["scaler_hi"]
    n, cl, gates = int(h["n_qubits"]), int(h["conv_layers"]), list(h["reupload_gates"])
    cohort, te, fd = _fold(cfg, seed, fold)
    span = onp.where(hi > lo, hi - lo, 1.0)
    scale = lambda X: onp.clip((X - lo) / span, 0, 1) * onp.pi  # noqa: E731
    n_q = len(params) - 2
    rows, cache = [], {}
    for p2, shots in grid:
        if p2 not in cache:
            cache[p2] = {k: onp.asarray(noisy_tree_expval(scale(X), params[:n_q], n, cl, gates, p2 / 10, p2, 1.0))
                         for k, X in (("train", fd.X_train), ("test", fd.X_test))}
        z = dict(cache[p2])
        if shots:
            rng = onp.random.default_rng(component_seed(seed, fold, f"shots:{source}:{p2}:{shots}"))
            z = {k: shot_estimate(v, shots, rng) for k, v in z.items()}
        prob = {k: 1 / (1 + onp.exp(-(params[-2] * v + params[-1]))) for k, v in z.items()}
        thr = pick_threshold("f1", fd.y_train, prob["train"])
        m = fold_metrics(fd.y_test, prob["test"], (prob["test"] >= thr).astype(int))
        lesions = cohort.frame.iloc[te]
        rows.append({"source": source, "arch": arch, "p2": p2, "p1": p2 / 10, "shots": shots or 0,
                     "seed": seed, "fold": fold, **m,
                     "_oof": pd.DataFrame({"ProxID": lesions["ProxID"].to_numpy(), "fid": lesions["fid"].to_numpy(),
                                           "label": fd.y_test, "prob": prob["test"]})})
    return rows


def run_eval(sources: list[tuple[str, str]], grid_fn, n_jobs: int, tag: str) -> None:
    tasks = [(src, arch, s, f) for src, arch in sources for s in SEEDS for f in FOLDS
             if _weights_path(src, s, f).exists()]
    if not tasks:
        print(f"  [{tag}] no weights found, skipped")
        return
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        results = list(ex.map(eval_task, *zip(*tasks), [grid_fn(t[0]) for t in tasks]))
    rows = [r for rs in results for r in rs]
    oof_dir = OUT / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for r in rows:
        o = r.pop("_oof")
        o = o.assign(source=r["source"], arch=r["arch"], p2=r["p2"], shots=r["shots"], seed=r["seed"], fold=r["fold"])
        frames.append(o)
    pd.concat(frames).to_csv(oof_dir / f"{tag}.csv.gz", index=False, compression="gzip", float_format="%.10g")
    pd.DataFrame(rows).to_csv(OUT / f"{tag}_folds.csv", index=False)
    print(f"  [{tag}] evaluated {len(tasks)} weight sets x {len(grid_fn(tasks[0][0]))} noise settings")


# ── summary ─────────────────────────────────────────────────────────────────
def stage_summary(n_boot: int = 2000) -> None:
    from v3core.stats import auc_mw, patient_bootstrap_indices

    frames = [pd.read_csv(p) for p in sorted((OUT / "oof").glob("*.csv.gz"))]
    if not frames:
        return
    oof = pd.concat(frames)
    oof["shots"] = oof["shots"].astype(int)
    keys = ["source", "arch", "p2", "shots"]
    rows = []
    for key, g in oof.groupby(keys):
        w = g.pivot_table(index=["ProxID", "fid", "label"], columns="seed", values="prob").reset_index()
        seeds = [c for c in w.columns if isinstance(c, (int, onp.integer))]
        y, groups = w["label"].to_numpy(), w["ProxID"].to_numpy()
        per_seed = [auc_mw(y, w[s].to_numpy()) for s in seeds]
        # paired reference: same weights, same arch, noiseless exact
        ref = oof[(oof.source == key[0]) & (oof.arch == key[1]) & (oof.p2 == 0) & (oof.shots == 0)]
        wr = ref.pivot_table(index=["ProxID", "fid", "label"], columns="seed", values="prob").reset_index()
        wr = w[["ProxID", "fid", "label"]].merge(wr, on=["ProxID", "fid", "label"], how="left")
        dist, ddist = [], []
        for idx in patient_bootstrap_indices(groups, n_boot, 0):
            yb = y[idx]
            if yb.min() == yb.max():
                continue
            a = onp.mean([auc_mw(yb, w[s].to_numpy()[idx]) for s in seeds])
            b = onp.mean([auc_mw(yb, wr[s].to_numpy()[idx]) for s in seeds])
            dist.append(a)
            ddist.append(a - b)
        fm = pd.read_csv(OUT / f"{'eval' if 'trainp2' not in key[0] else 'trainnoisy'}_folds.csv")
        fm = fm[(fm.source == key[0]) & (fm.arch == key[1]) & (fm.p2 == key[2]) & (fm.shots == key[3])]
        rows.append({"source": key[0], "arch": key[1], "p2": key[2], "p1": key[2] / 10, "shots": key[3],
                     "n_seeds": len(seeds), "pooled_AUC_mean": float(onp.mean(per_seed)),
                     "pooled_AUC_std": float(onp.std(per_seed, ddof=1)) if len(seeds) > 1 else float("nan"),
                     "ci95_lo": float(onp.percentile(dist, 2.5)), "ci95_hi": float(onp.percentile(dist, 97.5)),
                     "delta_vs_noiseless": float(onp.mean(ddist)),
                     "delta_ci95_lo": float(onp.percentile(ddist, 2.5)),
                     "delta_ci95_hi": float(onp.percentile(ddist, 97.5)),
                     "fold_F1_mean": float(fm["F1"].mean()), "fold_AUC_mean": float(fm["AUC"].mean())})
    df = pd.DataFrame(rows).sort_values(["source", "arch", "shots", "p2"])
    df.to_csv(OUT / "noise_summary.csv", index=False)
    print(df[["source", "arch", "p2", "shots", "pooled_AUC_mean", "pooled_AUC_std", "ci95_lo", "ci95_hi",
              "delta_vs_noiseless", "delta_ci95_lo", "delta_ci95_hi", "fold_F1_mean"]].round(4).to_string(index=False))
    plot(df)


def plot(df: pd.DataFrame) -> None:
    """(a) pooled AUC versus gate noise; (b) pooled AUC versus shots. One shared legend above."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from v3core.plotting import FIG_DIR, save_pdf, setup_style

    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    style = {"phase1_8q3L": ("C0", "o", "8 qubits, 3 layers (untuned)"), "6q1L": ("C2", "s", "6 qubits, 1 layer"),
             "tuned_8q": ("C4", "D", "Tuned HQCNN")}
    trained_noisy = {"8q3L": ("C1", "8 qubits, 3 layers, trained with noise"),
                     "6q1L": ("C3", "6 qubits, 1 layer, trained with noise")}

    def bars(d):
        return [d.pooled_AUC_mean - d.ci95_lo, d.ci95_hi - d.pooled_AUC_mean]

    ax = axes[0]
    for src, (c, mk, _) in style.items():
        d = df[(df.source == src) & (df.shots == 0)].sort_values("p2")
        if not d.empty:
            ax.errorbar(d.p2.replace(0, 3e-5), d.pooled_AUC_mean, yerr=bars(d), fmt=mk + "-", color=c, ms=3.5,
                        lw=1.1, capsize=2)
    for arch, (c, _) in trained_noisy.items():
        d = df[df.source.str.startswith(f"{arch}_trainp2") & (df.shots == 0)]
        d = d[d.apply(lambda r: r.source.endswith(f"{r.p2:g}"), axis=1)].sort_values("p2")
        if not d.empty:
            ax.errorbar(d.p2, d.pooled_AUC_mean, yerr=bars(d), fmt="*", color=c, ms=7, capsize=2)
    ax.set_xscale("log")
    ax.set_xlabel(r"Two-qubit error rate $p_2$ ($p_1 = p_2/10$)")
    ax.set_ylabel("Pooled out-of-fold AUC (95% CI)")
    ax = axes[1]
    for src, (c, mk, _) in style.items():
        for p2, ls in ((0.0, "-"), (1e-3, "--")):
            d = df[(df.source == src) & (df.p2 == p2) & (df.shots > 0)].sort_values("shots")
            if not d.empty:
                ax.errorbar(d.shots, d.pooled_AUC_mean, yerr=bars(d), fmt=mk + ls, color=c, ms=3.5, lw=1.1,
                            capsize=2)
    ax.set_xscale("log")
    ax.set_xlabel(r"Shots per $\langle Z\rangle$ estimate")
    for ax, t in zip(axes, "ab"):
        ax.text(0.0, 1.02, f"({t})", transform=ax.transAxes, ha="left", va="bottom", weight="bold")
    handles = [Line2D([], [], color=c, marker=mk, ms=3.5, lw=1.1, label=lab) for c, mk, lab in style.values()]
    handles += [Line2D([], [], color=c, marker="*", ms=7, lw=0, label=lab) for c, lab in trained_noisy.values()]
    handles += [Line2D([], [], color="0.3", ls="-", lw=1.1, label="(b) no gate noise"),
                Line2D([], [], color="0.3", ls="--", lw=1.1, label=r"(b) $p_2 = 10^{-3}$")]
    fig.legend(handles=handles, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.tight_layout()
    save_pdf(fig, FIG_DIR / "noise_auc.pdf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["verify", "train6q", "eval", "summary"])
    ap.add_argument("--n-jobs", type=int, default=5)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if "verify" in args.stages:
        stage_verify()
    if "train6q" in args.stages:
        tasks = [("6q1L", s, f, 0.0) for s in SEEDS for f in FOLDS if not _weights_path("6q1L", s, f).exists()]
        with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
            rows = list(ex.map(train_task, *zip(*tasks))) if tasks else []
        if rows:
            pd.DataFrame(rows).to_csv(OUT / "train6q.csv", index=False)
            print(f"  trained 6q/1L noiseless on {len(rows)} folds, mean {onp.mean([r['seconds'] for r in rows]):.0f}s")
    if "eval" in args.stages:
        grid = [(p, None) for p in P2_LEVELS] + [(p, s) for p in SHOT_P2 for s in SHOTS]
        srcs = [("phase1_8q3L", "8q3L"), ("6q1L", "6q1L")]
        if TUNED_META.exists():
            srcs.append(("tuned_8q", "8q3L"))
        run_eval(srcs, lambda src: grid, args.n_jobs, "eval")
    if "trainnoisy" in args.stages:
        tasks = [(a, s, f, p) for a in ARCHS for p in TRAIN_P2 for s in SEEDS for f in FOLDS
                 if not _weights_path(f"{a}_trainp2_{p:g}", s, f).exists()]
        tasks.sort(key=lambda t: t[0] != "8q3L")
        with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
            rows = list(ex.map(train_task, *zip(*tasks))) if tasks else []
        if rows:
            pd.DataFrame(rows).to_csv(OUT / "trainnoisy_log.csv", index=False)
        srcs = [(f"{a}_trainp2_{p:g}", a) for a in ARCHS for p in TRAIN_P2]
        run_eval(srcs, lambda src: [(float(src.rsplit("_", 1)[1]), None), (0.0, None)], args.n_jobs, "trainnoisy")
    if "summary" in args.stages:
        stage_summary()
    if "plot" in args.stages:                  # redraw from noise_summary.csv only
        plot(pd.read_csv(OUT / "noise_summary.csv"))
    print(f"  done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
