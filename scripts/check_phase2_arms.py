"""
Correctness checks for the Phase 2C arms (fast unless --trainer-check).

  1. tree_channel_expval(dephase_controls=False) == twin_expval         (random draws)
  2. tree_channel_expval(dephase_controls=True) == PennyLane default.mixed with a full
     phase flip (PhaseFlip p=0.5) on the control before every CNOT/CRZ/CRX
  3. dephased control keeps every pair state PPT (negativity 0); full circuit does not
  4. no_entanglement: <Z> on the read-out is unchanged when every feature NOT loaded on
     the read-out qubit is perturbed (information floor)
  5. parameter counts of all arms
  6. (--trainer-check) train_adam + twin reproduces the Phase 1 HQCNN on seed 0 / fold 0

Writes results/phase2/arms_checks.json.

Usage:
  python scripts/check_phase2_arms.py [--trainer-check]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as onp

V3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3_DIR))

import pennylane as qml  # noqa: E402
from pennylane import numpy as np  # noqa: E402

from v3core.arms import pair_negativity, tree_channel_expval  # noqa: E402
from v3core.config import RESULTS_DIR, load_config, v3_path  # noqa: E402
from v3core.hqcnn import HQCNN, encode, n_quantum_params, qcnn_plan  # noqa: E402
from v3core.phase2 import hqcnn_variant  # noqa: E402
from v3core.twin import twin_expval  # noqa: E402

CFG = v3_path("configs/phase2.yaml")


def mixed_dephased_qnode(n, ops, ro, gates):
    dev = qml.device("default.mixed", wires=n)

    @qml.qnode(dev)
    def circ(x, q):
        encode(x, n, gates)
        i = 0
        for kind, (a, b) in ops:
            if kind == "conv":
                p = q[i: i + 6]
                i += 6
                qml.RY(p[0], wires=a); qml.RY(p[1], wires=b)          # noqa: E702
                qml.PhaseFlip(0.5, wires=a); qml.CNOT(wires=[a, b])   # noqa: E702
                qml.RY(p[2], wires=a); qml.RY(p[3], wires=b)          # noqa: E702
                qml.PhaseFlip(0.5, wires=b); qml.CNOT(wires=[b, a])   # noqa: E702
                qml.RZ(p[4], wires=a); qml.RZ(p[5], wires=b)          # noqa: E702
            else:
                p = q[i: i + 2]
                i += 2
                qml.PhaseFlip(0.5, wires=a); qml.CRZ(p[0], wires=[a, b])  # noqa: E702
                qml.PhaseFlip(0.5, wires=a); qml.CRX(p[1], wires=[a, b])  # noqa: E702
        return qml.expval(qml.PauliZ(ro))
    return circ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer-check", action="store_true")
    args = ap.parse_args()
    cfg = load_config(CFG)
    h = cfg["hqcnn"]
    n, cl, gates = int(h["n_qubits"]), int(h["conv_layers"]), list(h["reupload_gates"])
    nf = int(cfg["preprocessing"]["pca_dim"])
    ops, ro = qcnn_plan(n, cl, "second", "tree")
    n_q = n_quantum_params(ops, h["pooling_gate"])
    rng = onp.random.default_rng(7)
    out = {}

    X = rng.uniform(0, onp.pi, (50, nf))
    Q = rng.uniform(-onp.pi, onp.pi, (50, n_q))
    e1 = max(float(onp.max(onp.abs(onp.asarray(tree_channel_expval(X[k:k + 1], Q[k], n, cl, gates))
                                    - onp.asarray(twin_expval(X[k:k + 1], Q[k], n, cl, gates)))))
             for k in range(50))
    out["channel_twin_vs_exact_twin_max_abs"] = e1
    print(f"  1. channel twin (no dephasing) vs twin: max|err| = {e1:.1e}")

    mixed = mixed_dephased_qnode(n, ops, ro, gates)
    e2 = []
    for k in range(10):
        z_mix = float(mixed(np.array(X[k]), np.array(Q[k])))
        z_tw = float(onp.asarray(tree_channel_expval(X[k:k + 1], Q[k], n, cl, gates, dephase_controls=True))[0])
        e2.append(abs(z_mix - z_tw))
    out["dephased_twin_vs_default_mixed_max_abs"] = float(max(e2))
    print(f"  2. dephased twin vs default.mixed (10 draws, 8q): max|err| = {max(e2):.1e}")

    _, neg_d = tree_channel_expval(X, Q[0], n, cl, gates, dephase_controls=True, track_negativity=True)
    _, neg_f = tree_channel_expval(X, Q[0], n, cl, gates, dephase_controls=False, track_negativity=True)
    out["max_pair_negativity_dephased"], out["max_pair_negativity_full"] = neg_d, neg_f
    print(f"  3. max pair negativity: dephased {neg_d:.1e}  vs full {neg_f:.3f}")
    assert neg_d < 1e-12 < neg_f

    m = HQCNN(hqcnn_variant(cfg, entangling="none"))
    on_ro = [i for i in range(nf) if i % n == ro]
    x0 = rng.uniform(0, onp.pi, (1, nf))
    x1 = x0.copy()
    x1[0, [i for i in range(nf) if i not in on_ro]] = rng.uniform(0, onp.pi, nf - len(on_ro))
    q = np.array(rng.uniform(-onp.pi, onp.pi, m.n_q))
    d_other = abs(float(m._z(x0, q)[0]) - float(m._z(x1, q)[0]))
    x2 = x0.copy()
    x2[0, on_ro] = rng.uniform(0, onp.pi, len(on_ro))
    d_own = abs(float(m._z(x0, q)[0]) - float(m._z(x2, q)[0]))
    out["no_entanglement_features_on_readout"] = on_ro
    out["no_entanglement_change_other_features"] = d_other
    out["no_entanglement_change_readout_features"] = d_own
    print(f"  4. no_entanglement: read-out sees features {on_ro}; |dZ| other features {d_other:.1e}, "
          f"own features {d_own:.3f}")
    assert d_other < 1e-12 < d_own

    counts = {"hqcnn": 142, "no_entanglement": HQCNN(hqcnn_variant(cfg, entangling="none")).n_params,
              "dephased_control": n_q + 2, "frozen_random (trained)": 2,
              "frozen_random_multiZ (trained)": n + 1,
              "brickwork_8q": HQCNN(hqcnn_variant(cfg, conv_topology="brickwork")).n_params,
              "classical_tree_net": (n - 1) * 21 + 4,
              "relaxed_bilinear_tree": (n - 1) * 11 * int(cfg["phase2_arms"]["relaxed_bilinear_tree"]["rank"]) + 2,
              "mlp_matched": nf * 10 + 10 + 10 + 1}
    out["param_counts"] = counts
    print(f"  5. params: {counts}")

    if args.trainer_check:
        import pandas as pd
        from v3core.arms import AngleScaler, sigmoid, train_adam
        from v3core.data import load_cohort, make_folds
        from v3core.preprocess import build_fold
        from v3core.seeding import component_rng, component_seed
        cohort = load_cohort(cfg)
        tr, te = make_folds(cohort, cfg["cv"]["n_splits"], 0)[0]
        fd = build_fold(cohort, tr, te, cfg["preprocessing"], component_seed(0, 0, "resample"))
        sc = AngleScaler(h["angle_range"]).fit(fd.X_fit)
        r = component_rng(0, 0, "hqcnn")                 # same stream as the Phase 1 HQCNN
        p0 = onp.concatenate([r.uniform(-onp.pi, onp.pi, n_q), [1.0, 0.0]])

        def logit(p, Xb):
            return p[-2] * tree_channel_expval(np.array(Xb, requires_grad=False), p[:n_q], n, cl, gates) + p[-1]
        params, _ = train_adam(logit, p0, sc.transform(fd.X_fit), fd.y_fit.astype(float), r, h)
        prob = sigmoid(onp.asarray(logit(np.array(params), sc.transform(fd.X_test))))
        p1 = pd.read_csv(RESULTS_DIR / "oof" / "hqcnn" / "seed0_fold0.csv.gz")["prob"].to_numpy()
        d = float(onp.max(onp.abs(prob - p1)))
        out["trainer_twin_vs_phase1_hqcnn_seed0_fold0_max_abs"] = d
        print(f"  6. train_adam + twin vs Phase 1 HQCNN (seed 0 / fold 0): max|dp| = {d:.1e}")

    path = RESULTS_DIR / "phase2" / "arms_checks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(path.read_text()) if path.exists() else {}
    prev.update(out)
    path.write_text(json.dumps(prev, indent=2))
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
