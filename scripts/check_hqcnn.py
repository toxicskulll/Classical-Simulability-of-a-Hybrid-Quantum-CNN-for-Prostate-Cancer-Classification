"""
Fast HQCNN correctness checks (seconds; no real training).

  1. broadcast default.qubit/backprop == per-sample lightning.qubit/adjoint (forward)
  2. same for the gradient of the training loss
  3. same Generator seed -> bit-identical parameters; different seed -> different
  4. structure: parameter count, read-out qubit, pooling 8 -> 4 -> 2 -> 1

Usage:
  python scripts/check_hqcnn.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

V3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3_DIR))

import pennylane as qml  # noqa: E402
from pennylane import numpy as pnp  # noqa: E402

from v3core.config import load_config  # noqa: E402
from v3core.hqcnn import HQCNN, qcnn_plan, weighted_loss  # noqa: E402


def main() -> None:
    cfg = load_config()
    h = dict(cfg["hqcnn"])
    n_feat = int(cfg["preprocessing"]["pca_dim"])
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, n_feat))
    y = np.array([0, 1] * 5)

    fast = HQCNN(h)
    slow = HQCNN({**h, "simulator": {"device": "lightning.qubit", "diff_method": "adjoint",
                                     "broadcast": False}})
    q = rng.uniform(-np.pi, np.pi, fast.n_q)
    fast._fit_scaler(X)
    slow._fit_scaler(X)
    Xs = fast._scale(X)
    z_fast = np.asarray(fast._z(Xs, pnp.array(q)))
    z_slow = np.asarray(slow._z(Xs, pnp.array(q)))
    print(f"  forward max |diff| = {np.max(np.abs(z_fast - z_slow)):.2e}")
    assert np.allclose(z_fast, z_slow, atol=1e-10)

    def grad(model):
        p = pnp.array(np.concatenate([q, [1.3, -0.2]]), requires_grad=True)
        yb = pnp.array(y.astype(float), requires_grad=False)
        wb = pnp.ones(len(y), requires_grad=False)

        def cost(p):
            raw = p[-2] * model._z(Xs, p[: model.n_q]) + p[-1]
            return weighted_loss(raw, yb, wb, "focal", 2.335)
        return np.asarray(qml.grad(cost)(p))
    g_fast, g_slow = grad(fast), grad(slow)
    print(f"  gradient max |diff| = {np.max(np.abs(g_fast - g_slow)):.2e}")
    assert np.allclose(g_fast, g_slow, atol=1e-8)

    short = {**h, "n_epochs": 2, "batch_size": 4}
    a = HQCNN(short).fit(X, y, np.random.default_rng(7)).params_
    b = HQCNN(short).fit(X, y, np.random.default_rng(7)).params_
    c = HQCNN(short).fit(X, y, np.random.default_rng(8)).params_
    assert np.array_equal(a, b), "same seed gave different parameters"
    assert not np.allclose(a, c), "different seeds gave identical parameters"
    print("  determinism: same seed identical, different seed differs")

    ops, ro = qcnn_plan(h["n_qubits"], h["conv_layers"], h["pooling_survivor"])
    levels, run = [], 0
    for t, _ in ops + [("conv", None)]:
        if t == "pool":
            run += 1
        elif run:
            levels.append(run)
            run = 0
    print(f"  params={fast.n_params} (quantum {fast.n_q} + head 2), readout=q{ro}, "
          f"pool ops per level={levels}")
    assert levels == [4, 2, 1] and ro == h["n_qubits"] - 1
    specs = qml.specs(fast.circuit)(pnp.array(Xs[0]), pnp.array(q))["resources"]
    print(f"  gates={specs.num_gates} depth={specs.depth} types={dict(specs.gate_types)}")
    print("  all HQCNN checks passed")


if __name__ == "__main__":
    main()
