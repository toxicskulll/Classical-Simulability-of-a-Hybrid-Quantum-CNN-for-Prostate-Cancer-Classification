"""
Can this machine handle a 20-qubit HQCNN? Measures, on real PCA-20 inputs (seed 0 / fold 0):

  * statevector memory (analytic) for 20 qubits, per state and per batch of 64
  * PennyLane lightning.qubit: forward and adjoint gradient for ONE sample, tree and brickwork
  * exact twin (tree only): forward + backprop for a batch of 64, and one training epoch,
    noiseless and with gate noise (p2 = 1e-3)
  * projected cost of one 80-epoch CV fold and of 25 folds (5 seeds x 5 folds)

Writes results/phase3/qubit20_check.json.

Usage:
  python scripts/qubit20_check.py
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as onp  # noqa: E402

V3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3_DIR))

import pennylane as qml  # noqa: E402
from pennylane import numpy as np  # noqa: E402

from v3core.arms import train_adam  # noqa: E402
from v3core.config import RESULTS_DIR, load_config  # noqa: E402
from v3core.hqcnn import build_qnode, n_quantum_params, qcnn_plan  # noqa: E402
from v3core.noise import noisy_tree_expval  # noqa: E402
from v3core.phase2 import real_inputs  # noqa: E402
from v3core.twin import twin_expval  # noqa: E402

N, CL = 20, 3


def t(fn, reps=1):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def main() -> None:
    cfg = load_config()
    h = cfg["hqcnn"]
    gates = list(h["reupload_gates"])
    data = real_inputs(cfg, pca_dim=N)
    X, y = data["train"], data["fold_data"].y_fit.astype(float)
    n_fit = len(y)
    res = {"n_qubits": N, "conv_layers": CL, "n_features": int(X.shape[1]), "n_fit": n_fit,
           "statevector_MB_per_state": 2 ** N * 16 / 2 ** 20,
           "statevector_GB_batch64": 64 * 2 ** N * 16 / 2 ** 30}
    print(f"  memory: one 20-qubit state = {res['statevector_MB_per_state']:.0f} MB; "
          f"a broadcast batch of 64 = {res['statevector_GB_batch64']:.1f} GB per stored state "
          f"(backprop stores one per gate)")

    for topo in ("tree", "brickwork"):
        ops, ro = qcnn_plan(N, CL, "second", topo)
        n_q = n_quantum_params(ops, h["pooling_gate"])
        circ = build_qnode(N, ops, ro, gates, h["pooling_gate"], "lightning.qubit", "adjoint")
        x = np.array(X[0], requires_grad=False)
        q = np.array(onp.random.default_rng(0).uniform(-onp.pi, onp.pi, n_q), requires_grad=True)
        tf = t(lambda: circ(x, q), 2)
        tg = t(lambda: qml.grad(circ, argnum=1)(x, q), 1)
        per_epoch = tg * n_fit
        res[f"lightning_{topo}"] = {"n_params_circuit": n_q, "n_ops": len(ops), "forward_s": tf,
                                    "grad_s": tg, "epoch_s_est": per_epoch,
                                    "fold_80ep_h_est": per_epoch * 80 / 3600,
                                    "cv25_h_est_7workers": per_epoch * 80 * 25 / 3600 / 7}
        print(f"  lightning {topo:<9}: {n_q} params, forward {tf:.2f}s, adjoint grad {tg:.2f}s per sample "
              f"-> ~{per_epoch / 60:.0f} min/epoch, ~{per_epoch * 80 / 3600:.1f} h per 80-epoch fold")

    ops, ro = qcnn_plan(N, CL, "second", "tree")
    n_q = n_quantum_params(ops, h["pooling_gate"])
    hh = copy.deepcopy(h)
    hh.update(n_qubits=N, n_epochs=1)
    for name, p2 in (("twin_noiseless", 0.0), ("twin_noisy_p2_1e-3", 1e-3)):
        def logit(p, Xb, p2=p2):
            z = (twin_expval(np.array(Xb, requires_grad=False), p[:n_q], N, CL, gates) if p2 == 0 else
                 noisy_tree_expval(np.array(Xb, requires_grad=False), p[:n_q], N, CL, gates, p2 / 10, p2))
            return p[-2] * z + p[-1]
        rng = onp.random.default_rng(0)
        p0 = onp.concatenate([rng.uniform(-onp.pi, onp.pi, n_q), [1.0, 0.0]])
        t0 = time.perf_counter()
        train_adam(logit, p0, X, y, rng, hh)
        ep = time.perf_counter() - t0
        res[name] = {"n_params": n_q + 2, "epoch_s": ep, "fold_80ep_min": ep * 80 / 60,
                     "cv25_h_7workers": ep * 80 * 25 / 3600 / 7}
        print(f"  {name:<20}: {ep:.1f}s per epoch -> {ep * 80 / 60:.1f} min per 80-epoch fold, "
              f"~{ep * 80 * 25 / 3600 / 7:.1f} h for 25 folds on 7 workers")
    out = RESULTS_DIR / "phase3" / "qubit20_check.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
