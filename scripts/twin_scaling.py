"""
Phase 2A: wall time per forward pass vs n_qubits, classical twin vs PennyLane statevector.

Circuit: champion tree HQCNN (3 conv layers, CRZ/CRX pooling), one RY-encoded
feature per qubit, batch of 1 input, random parameters. Timings are the median of
repeated calls after one warm-up call (which includes QNode construction).

PennyLane is run with lightning.qubit (C++ statevector) and default.qubit. A
simulator is stopped for all larger sizes once it fails (e.g. MemoryError) or when
its projected time for the next size exceeds --pl-cap seconds; this is recorded
as status 'failed' / 'skipped_projected'.

Run on an otherwise idle machine. Writes results/phase2/twin/scaling.csv and
figures/twin_scaling.pdf.

Usage:
  python scripts/twin_scaling.py
  python scripts/twin_scaling.py --plot-only
"""

from __future__ import annotations

import argparse
import gc
import os
import platform
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as onp  # noqa: E402
import pandas as pd  # noqa: E402

V3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3_DIR))

import pennylane as qml  # noqa: E402
from pennylane import numpy as np  # noqa: E402

from v3core.config import RESULTS_DIR, load_config  # noqa: E402
from v3core.hqcnn import build_qnode, n_quantum_params, qcnn_plan  # noqa: E402
from v3core.plotting import FIG_DIR, save_pdf, setup_style  # noqa: E402
from v3core.twin import twin_expval  # noqa: E402

OUT = RESULTS_DIR / "phase2" / "twin"
SIZES = [4, 8, 12, 16, 20, 24, 26, 28, 32, 64, 128, 256]


def timeit(fn, min_reps: int = 3, min_seconds: float = 1.0) -> tuple[float, int]:
    fn()  # warm-up
    times, t_start = [], time.perf_counter()
    while len(times) < min_reps or (time.perf_counter() - t_start < min_seconds and len(times) < 200):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(onp.median(times)), len(times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pl-cap", type=float, default=300.0, help="max projected seconds per forward")
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from scaling.csv")
    args = ap.parse_args()
    if args.plot_only:
        plot(pd.read_csv(OUT / "scaling.csv"))
        return
    setup_style()
    h = load_config()["hqcnn"]
    cl, gates = int(h["conv_layers"]), list(h["reupload_gates"])
    rng = onp.random.default_rng(0)

    rows = []
    alive = {"twin": True, "lightning.qubit": True, "default.qubit": True}
    last = {}
    for n in SIZES:
        ops, ro = qcnn_plan(n, cl, h["pooling_survivor"])
        n_q = n_quantum_params(ops, h["pooling_gate"])
        x = np.array(rng.uniform(0, onp.pi, size=(1, n)), requires_grad=False)
        q = np.array(rng.uniform(-onp.pi, onp.pi, n_q), requires_grad=False)
        for sim in ("twin", "lightning.qubit", "default.qubit"):
            row = {"n_qubits": n, "simulator": sim, "n_params_circuit": n_q, "n_ops": len(ops)}
            if not alive[sim]:
                row.update(status="skipped_after_failure", seconds=float("nan"))
                rows.append(row)
                continue
            # project from the last two points (statevector cost ~ 2^n x gates)
            if sim != "twin" and sim in last:
                n_prev, t_prev, ops_prev = last[sim]
                projected = t_prev * 2.0 ** (n - n_prev) * (len(ops) / ops_prev)
                if projected > args.pl_cap:
                    alive[sim] = False
                    row.update(status="skipped_projected", seconds=float("nan"), projected_seconds=projected)
                    rows.append(row)
                    continue
            try:
                if sim == "twin":
                    fn = lambda: twin_expval(x, q, n, cl, gates, h["pooling_gate"], h["pooling_survivor"])  # noqa: E731
                else:
                    circ = build_qnode(n, ops, ro, gates, h["pooling_gate"], sim,
                                       "adjoint" if sim == "lightning.qubit" else "backprop")
                    fn = lambda: circ(x[0], q)  # noqa: E731
                t, reps = timeit(fn, min_reps=3 if sim == "twin" or n < 20 else 1,
                                 min_seconds=1.0 if n < 24 else 0.0)
                row.update(status="ok", seconds=t, reps=reps)
                last[sim] = (n, t, len(ops))
            except (MemoryError, RuntimeError, ValueError) as exc:
                alive[sim] = False
                row.update(status="failed", seconds=float("nan"), error=f"{type(exc).__name__}: {exc}"[:200])
            rows.append(row)
            print(f"  n={n:>3} {sim:<16} {row['status']:<22} "
                  f"{row.get('seconds', float('nan')):.3e}s", flush=True)
            gc.collect()

    df = pd.DataFrame(rows)
    df["machine"] = f"{platform.processor()} | {platform.python_version()} | PL {qml.__version__}"
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "scaling.csv", index=False)
    plot(df)


def plot(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    setup_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    style = {"twin": ("o-", "C0", "Classical twin"),
             "lightning.qubit": ("s-", "C1", "PennyLane lightning.qubit"),
             "default.qubit": ("^-", "C2", "PennyLane default.qubit")}
    for sim, (mk, c, label) in style.items():
        d = df[(df.simulator == sim) & (df.status == "ok")]
        stop = df[(df.simulator == sim) & (df.status.isin(["failed", "skipped_projected"]))]
        if len(stop):
            why = "out of memory" if stop.status.iloc[0] == "failed" else "stopped"
            label += f" ({why} at {int(stop.n_qubits.iloc[0])})"
        ax.plot(d.n_qubits, d.seconds, mk, color=c, ms=3.5, lw=1.2, label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ticks = [4, 8, 16, 32, 64, 128, 256]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Number of qubits $n$")
    ax.set_ylabel("Forward pass time (s)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False)
    ax.grid(True, which="major", lw=0.3, alpha=0.5)
    save_pdf(fig, FIG_DIR / "twin_scaling.pdf")
    print(f"  wrote {OUT / 'scaling.csv'} and {FIG_DIR / 'twin_scaling.pdf'}")


if __name__ == "__main__":
    main()
