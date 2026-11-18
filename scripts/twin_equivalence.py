"""
Phase 2A: prove the classical twin equals the HQCNN statevector simulation.

For each circuit configuration, draw N random (input, parameter) pairs, with inputs in
[0, pi] and parameters ~ U(-pi, pi). Compare:
  * forward <Z>: twin vs PennyLane default.qubit (v3core.hqcnn)
  * gradient d<Z>/d(theta) for every circuit parameter: twin (autograd) vs
    PennyLane (backprop)

Writes results/phase2/twin/equivalence.csv (per config) and equivalence.json.

Usage:
  python scripts/twin_equivalence.py            # 1000 draws, champion config
  python scripts/twin_equivalence.py --n 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as onp

V3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3_DIR))

import pennylane as qml  # noqa: E402
from pennylane import numpy as np  # noqa: E402

from v3core.config import RESULTS_DIR, load_config  # noqa: E402
from v3core.hqcnn import HQCNN  # noqa: E402
from v3core.twin import twin_expval  # noqa: E402

OUT = RESULTS_DIR / "phase2" / "twin"


def compare(hcfg: dict, n_feat: int, n_draws: int, seed: int) -> dict:
    model = HQCNN(hcfg)
    rng = onp.random.default_rng(seed)
    gates = list(hcfg["reupload_gates"])
    fwd_err, grad_err, zs = [], [], []
    t_pl = t_tw = 0.0
    for _ in range(n_draws):
        x = rng.uniform(0.0, onp.pi, size=(1, n_feat))
        q = np.array(rng.uniform(-onp.pi, onp.pi, model.n_q), requires_grad=True)
        xs = np.array(x, requires_grad=False)

        def f_pl(qq):
            return model._z(xs, qq)[0]

        def f_tw(qq):
            return twin_expval(xs, qq, model.n_qubits, int(hcfg["conv_layers"]), gates,
                               hcfg["pooling_gate"], hcfg["pooling_survivor"])[0]

        t0 = time.perf_counter()
        z_pl, g_pl = float(f_pl(q)), onp.asarray(qml.grad(f_pl)(q))
        t1 = time.perf_counter()
        z_tw, g_tw = float(f_tw(q)), onp.asarray(qml.grad(f_tw)(q))
        t2 = time.perf_counter()
        t_pl += t1 - t0
        t_tw += t2 - t1
        fwd_err.append(abs(z_pl - z_tw))
        grad_err.append(float(onp.max(onp.abs(g_pl - g_tw))))
        zs.append(z_pl)
    return {
        "n_qubits": model.n_qubits, "conv_layers": int(hcfg["conv_layers"]),
        "n_features": n_feat, "n_params_circuit": model.n_q, "n_draws": n_draws,
        "max_abs_err_forward": float(onp.max(fwd_err)),
        "max_abs_err_gradient": float(onp.max(grad_err)),
        "z_range": [float(onp.min(zs)), float(onp.max(zs))],
        "seconds_pennylane": t_pl, "seconds_twin": t_tw,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="draws for the champion config")
    ap.add_argument("--n-extra", type=int, default=100, help="draws for each extra config")
    args = ap.parse_args()

    cfg = load_config()
    base = dict(cfg["hqcnn"])
    n_feat = int(cfg["preprocessing"]["pca_dim"])
    configs = [({**base}, n_feat, args.n, 0)]                             # champion: 8q, 3L, 12 feats
    for nq, cl, nf in [(4, 1, 4), (4, 2, 12), (6, 2, 12), (8, 1, 8), (16, 2, 16), (16, 1, 40)]:
        configs.append(({**base, "n_qubits": nq, "conv_layers": cl}, nf, args.n_extra, nq * 10 + cl))

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for hcfg, nf, n, seed in configs:
        r = compare(hcfg, nf, n, seed)
        rows.append(r)
        print(f"  {r['n_qubits']:>2}q {r['conv_layers']}L {nf:>2} feats x{n:<5} "
              f"fwd max|err|={r['max_abs_err_forward']:.1e}  "
              f"grad max|err|={r['max_abs_err_gradient']:.1e}  "
              f"(PL {r['seconds_pennylane']:.1f}s, twin {r['seconds_twin']:.1f}s)", flush=True)

    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT / "equivalence.csv", index=False)
    summary = {
        "draws_total": int(sum(r["n_draws"] for r in rows)),
        "max_abs_err_forward": max(r["max_abs_err_forward"] for r in rows),
        "max_abs_err_gradient": max(r["max_abs_err_gradient"] for r in rows),
        "configs": rows,
    }
    (OUT / "equivalence.json").write_text(json.dumps(summary, indent=2))
    print(f"  overall max |err|: forward {summary['max_abs_err_forward']:.2e}, "
          f"gradient {summary['max_abs_err_gradient']:.2e} over {summary['draws_total']} draws")
    assert summary["max_abs_err_forward"] < 1e-10 and summary["max_abs_err_gradient"] < 1e-10


if __name__ == "__main__":
    main()
